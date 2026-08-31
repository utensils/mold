//! LTX-2.5 GGUF transformer backend (#1414).
//!
//! The seven `Abiray/LTX-2.5-Distilled-GGUF` tiers store the transformer's
//! block linears as ggml K-quants / Q8_0 with every non-block tensor F32,
//! under bare ComfyUI-native names (no `model.diffusion_model.` prefix —
//! city96's loader has no `ltxv` remap table, so the names in the file are
//! already the names mold's models ask for, modulo
//! [`mold_core::ltx2_weight_index::canonical_tensor_name`]).
//!
//! [`Ltx2GgufBackend`] is the LTX-2 `SimpleBackend` for those files,
//! mirroring the ConvRot and NVFP4 backends: it retains only the parsed
//! header plus a re-seekable reader and materializes ONE tensor per `get` —
//! never `mold_candle::quantized::VarBuilder::from_gguf`, which uploads the
//! whole file to the device eagerly. A float-stored tensor (and, until the
//! quantized arm is wired, a quantized one) is dequantized on the CPU,
//! cast to the requested dtype, and moved to the device — the exact shape of
//! `Ltx2ConvRotBackend::lookup`. For a quantized block linear the backend
//! additionally serves the synthetic side channel `weight.gguf_blocks` (raw
//! ggml payload bytes, U8, CPU) + `weight.gguf_dtype` (the ggml dtype id,
//! U32), which `LtxLinear` rebuilds into a device-resident `QTensor` so the
//! weight stays compact at rest.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use candle_core::quantized::{ggml_file, gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use mold_core::ltx2_weight_index::canonical_tensor_name;

/// `ltx2 linear kind=...` — which arm the GGUF quantized linears execute
/// (the UAT harness greps these exact literals; keep them in sync with
/// `provenance_vocabulary` when that module lands).
pub(crate) const LINEAR_KIND_QMATMUL: &str = "ltx2 linear kind=qmatmul";
pub(crate) const LINEAR_KIND_DEQUANT: &str = "ltx2 linear kind=dequant";

/// Emit the linear-kind provenance line once per literal, mirroring
/// `convrot::log_int8_arm_once`: the env decision is process-frozen and the
/// device class is fixed per engine, so one literal describes a whole render
/// session — but a multi-device process (a Metal engine beside a CPU-placed
/// one) legitimately earns both lines, and swallowing the second would hide
/// which arm the other device took.
pub(crate) fn log_linear_kind_once(kind: &'static str) {
    static LOGGED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|error| error.into_inner());
    if !logged.contains(&kind) {
        logged.push(kind);
        tracing::info!(target: crate::ltx2::provenance::LOG_TARGET, "{kind}");
    }
}

/// Process-frozen `MOLD_LTX2_QMATMUL`, read once through the
/// admission-frozen environment: a truthy value opts CUDA back into candle's
/// quantized fast path. Defaults off per the Qwen-Image / Z-Image precedent —
/// candle's fast MMQ kernels returned non-finite values for both (#1048) —
/// so the dequant arm is what ships until the UAT qualifies the fast path.
pub(crate) fn ltx2_qmatmul_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = crate::quantized_linear::parse_qmatmul_flag(
            crate::runtime_env::value("MOLD_LTX2_QMATMUL").as_deref(),
        );
        if enabled {
            tracing::warn!(
                "ltx2: MOLD_LTX2_QMATMUL=1 — candle's quantized CUDA fast path enabled; \
                 the same kernels return non-finite values for Qwen-Image and Z-Image (#1048)"
            );
        }
        enabled
    })
}

/// Whether `path` is a GGUF container, by magic. Cheap enough for the
/// checkpoint-format dispatch that runs before every transformer load.
pub(crate) fn checkpoint_is_gguf(path: &Path) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).is_ok() && &magic == b"GGUF"
}

/// The quantized storage types an LTX-2.5 GGUF may carry, as
/// `(GgmlDType, ggml type id)`. The id set matches `ltx25_probe`'s accepted
/// dtypes minus the float-stored ones, which take the plain dense arm.
/// Candle's `GgmlDType::{to_u32, from_u32}` are `pub(crate)`, so the side
/// channel carries the id through this table instead.
const ACCEPTED_QUANTIZED: &[(GgmlDType, u32)] = &[
    (GgmlDType::Q8_0, 8),
    (GgmlDType::Q3K, 11),
    (GgmlDType::Q4K, 12),
    (GgmlDType::Q5K, 13),
    (GgmlDType::Q6K, 14),
];

pub(crate) fn gguf_dtype_id(dtype: GgmlDType) -> Option<u32> {
    ACCEPTED_QUANTIZED
        .iter()
        .find(|(candidate, _)| *candidate == dtype)
        .map(|(_, id)| *id)
}

pub(crate) fn gguf_dtype_from_id(id: u32) -> Option<GgmlDType> {
    ACCEPTED_QUANTIZED
        .iter()
        .find(|(_, candidate)| *candidate == id)
        .map(|(dtype, _)| *dtype)
}

/// One component of the synthetic GGUF side channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufComponent {
    /// `weight.gguf_blocks`: the raw ggml payload bytes, U8, on the CPU.
    Blocks,
    /// `weight.gguf_dtype`: the ggml dtype id, U32 scalar-shaped.
    Dtype,
}

/// Split a synthetic side-channel key into the logical weight key it targets
/// and the component it asks for.
fn gguf_component(name: &str) -> Option<(&str, GgufComponent)> {
    if let Some(weight_name) = name.strip_suffix(".gguf_blocks") {
        Some((weight_name, GgufComponent::Blocks))
    } else {
        name.strip_suffix(".gguf_dtype")
            .map(|weight_name| (weight_name, GgufComponent::Dtype))
    }
}

/// Header facts for one tensor: everything a seek-and-read needs.
struct GgufTensorFact {
    dtype: GgmlDType,
    /// PyTorch-order shape (candle's `Content::read` reverses the on-disk
    /// ggml order).
    shape: Vec<usize>,
    /// Offset inside the tensor-data section.
    offset: u64,
    size_in_bytes: usize,
}

pub(super) struct Ltx2GgufBackend {
    file: Mutex<BufReader<File>>,
    tensor_data_offset: u64,
    /// Keyed by canonical tensor name; the value keeps the facts needed to
    /// materialize the tensor without re-parsing the header.
    tensors: HashMap<String, GgufTensorFact>,
    path: PathBuf,
}

impl Ltx2GgufBackend {
    pub(super) fn from_path(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("open LTX-2 GGUF checkpoint at {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)
            .with_context(|| format!("parse LTX-2 GGUF header at {}", path.display()))?;
        let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
        for (name, info) in &content.tensor_infos {
            let elements = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            if !elements.is_multiple_of(block_size) {
                anyhow::bail!(
                    "GGUF tensor {name} in {} has {elements} elements, not a multiple of the {:?} block size {block_size}",
                    path.display(),
                    info.ggml_dtype,
                );
            }
            let size_in_bytes = elements / block_size * info.ggml_dtype.type_size();
            let canonical = canonical_tensor_name(name);
            let previous = tensors.insert(
                canonical.clone(),
                GgufTensorFact {
                    dtype: info.ggml_dtype,
                    shape: info.shape.dims().to_vec(),
                    offset: info.offset,
                    size_in_bytes,
                },
            );
            if previous.is_some() {
                anyhow::bail!(
                    "GGUF checkpoint {} carries two tensors that canonicalize to {canonical}",
                    path.display(),
                );
            }
        }
        Ok(Self {
            file: Mutex::new(reader),
            tensor_data_offset: content.tensor_data_offset,
            tensors,
            path: path.to_path_buf(),
        })
    }

    fn fact(&self, canonical: &str) -> candle_core::Result<&GgufTensorFact> {
        self.tensors.get(canonical).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "LTX-2 GGUF checkpoint {} has no tensor {canonical}",
                self.path.display()
            ))
        })
    }

    /// Seek to and read one tensor's raw ggml payload bytes.
    fn read_raw(&self, fact: &GgufTensorFact) -> candle_core::Result<Vec<u8>> {
        let mut file = self.file.lock().unwrap_or_else(|err| err.into_inner());
        file.seek(SeekFrom::Start(
            self.tensor_data_offset.saturating_add(fact.offset),
        ))?;
        let mut raw = vec![0u8; fact.size_in_bytes];
        file.read_exact(&mut raw)?;
        Ok(raw)
    }

    /// Whether the side channel serves `canonical` — an accepted quantized
    /// storage type. Float-stored tensors take the plain dense arm, exactly
    /// as `QMatMul::from_arc` would eagerly dequantize them anyway.
    fn side_channel_serves(&self, canonical: &str) -> bool {
        self.tensors
            .get(canonical)
            .is_some_and(|fact| gguf_dtype_id(fact.dtype).is_some())
    }

    fn load_side_channel(
        &self,
        canonical: &str,
        component: GgufComponent,
    ) -> candle_core::Result<Tensor> {
        let fact = self.fact(canonical)?;
        let Some(dtype_id) = gguf_dtype_id(fact.dtype) else {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 GGUF side channel is not exposed for {canonical} ({:?})",
                fact.dtype
            )));
        };
        match component {
            GgufComponent::Blocks => {
                let raw = self.read_raw(fact)?;
                let len = raw.len();
                // CPU on purpose: the loader immediately hands the bytes to
                // `qtensor_from_ggml`, which uploads them itself.
                Tensor::from_vec(raw, len, &Device::Cpu)
            }
            GgufComponent::Dtype => Tensor::from_vec(vec![dtype_id], 1, &Device::Cpu),
        }
    }

    fn lookup(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        if let Some((weight_name, component)) = gguf_component(name) {
            return self.load_side_channel(&canonical_tensor_name(weight_name), component);
        }
        let canonical = canonical_tensor_name(name);
        let fact = self.fact(&canonical)?;
        let raw = self.read_raw(fact)?;
        let qtensor =
            ggml_file::qtensor_from_ggml(fact.dtype, &raw, fact.shape.clone(), &Device::Cpu)?;
        // Keep only the returned model-precision tensor: streaming callers
        // drop each block after use, and a backend cache would silently
        // reconstruct the compact checkpoint into a dense-model-sized one.
        qtensor
            .dequantize(&Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(dev)
    }

    /// The logical shape of one tensor, for tests and diagnostics.
    #[cfg(test)]
    fn shape_of(&self, canonical: &str) -> Option<Shape> {
        self.tensors
            .get(canonical)
            .map(|fact| Shape::from(fact.shape.clone()))
    }
}

/// Rebuild a device-resident `QTensor` from the side channel's raw payload.
/// `dims` is the module's own `[out, in]` expectation; `qtensor_from_ggml`
/// re-validates it against the payload length.
pub(crate) fn qtensor_from_side_channel(
    blocks: &Tensor,
    dtype_id: u32,
    dims: Vec<usize>,
    device: &Device,
) -> candle_core::Result<QTensor> {
    let Some(dtype) = gguf_dtype_from_id(dtype_id) else {
        return Err(candle_core::Error::Msg(format!(
            "unknown LTX-2 GGUF side-channel dtype id {dtype_id}"
        )));
    };
    let raw = blocks.flatten_all()?.to_vec1::<u8>()?;
    ggml_file::qtensor_from_ggml(dtype, &raw, dims, device)
}

impl SimpleBackend for Ltx2GgufBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hints: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.lookup(name, dtype, dev)?;
        if tensor.shape() != &shape {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("LTX-2 GGUF shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.lookup(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if let Some((weight_name, _)) = gguf_component(name) {
            return self.side_channel_serves(&canonical_tensor_name(weight_name));
        }
        self.tensors.contains_key(&canonical_tensor_name(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ltx2::model::video_transformer::tests::{
        assert_tensors_close, av_transformer_tensors, tiny_av_config,
    };
    use crate::ltx2::model::video_transformer::Ltx2AvTransformer3DModel;
    use candle_nn::VarBuilder;
    use std::collections::HashMap as StdHashMap;

    fn magic_file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn checkpoint_is_gguf_dispatches_on_the_magic() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = magic_file(&dir, "model.gguf", b"GGUFxxxxxxxx");
        let safetensors = magic_file(&dir, "model.safetensors", &8u64.to_le_bytes());
        let short = magic_file(&dir, "short.bin", b"GG");
        assert!(checkpoint_is_gguf(&gguf));
        assert!(!checkpoint_is_gguf(&safetensors));
        assert!(!checkpoint_is_gguf(&short));
        assert!(!checkpoint_is_gguf(&dir.path().join("missing.gguf")));
    }

    #[test]
    fn gguf_component_splits_the_side_channel_keys() {
        assert_eq!(
            gguf_component("transformer_blocks.0.attn1.to_q.weight.gguf_blocks"),
            Some((
                "transformer_blocks.0.attn1.to_q.weight",
                GgufComponent::Blocks
            ))
        );
        assert_eq!(
            gguf_component("weight.gguf_dtype"),
            Some(("weight", GgufComponent::Dtype))
        );
        assert_eq!(gguf_component("weight"), None);
        assert_eq!(gguf_component("weight.nvfp4_packed"), None);
    }

    #[test]
    fn side_channel_dtype_ids_round_trip_the_accepted_set() {
        for (dtype, id) in ACCEPTED_QUANTIZED {
            assert_eq!(gguf_dtype_id(*dtype), Some(*id));
            assert_eq!(gguf_dtype_from_id(*id), Some(*dtype));
        }
        assert_eq!(gguf_dtype_id(GgmlDType::F32), None);
        assert_eq!(gguf_dtype_id(GgmlDType::F16), None);
        assert_eq!(gguf_dtype_id(GgmlDType::BF16), None);
        assert_eq!(gguf_dtype_from_id(0), None);
        assert_eq!(gguf_dtype_from_id(30), None);
    }

    /// A tensor-map entry is written quantized iff it is a rank-2 block
    /// linear weight whose row width the Q8_0 block size divides — the same
    /// split the real tiers use (block linears quantized, everything else
    /// float-stored).
    fn quantize_for_fixture(name: &str, tensor: &Tensor) -> GgmlDType {
        let quantizable = name.starts_with("transformer_blocks.")
            && name.ends_with(".weight")
            && tensor.rank() == 2
            && tensor
                .dims()
                .last()
                .is_some_and(|cols| cols.is_multiple_of(32));
        if quantizable {
            GgmlDType::Q8_0
        } else {
            GgmlDType::F32
        }
    }

    /// Write `tensors` as a GGUF checkpoint with real metadata keys, and
    /// return the path plus the dequantized twin of every tensor (what the
    /// file actually stores, quantization error included).
    fn write_gguf_fixture(
        dir: &tempfile::TempDir,
        tensors: &StdHashMap<String, Tensor>,
    ) -> (PathBuf, StdHashMap<String, Tensor>) {
        let mut quantized: Vec<(String, QTensor)> = Vec::new();
        let mut dequantized = StdHashMap::new();
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        for name in names {
            let tensor = &tensors[name];
            let dtype = quantize_for_fixture(name, tensor);
            let qtensor = QTensor::quantize(tensor, dtype).unwrap();
            dequantized.insert(name.clone(), qtensor.dequantize(&Device::Cpu).unwrap());
            quantized.push((name.clone(), qtensor));
        }
        let path = dir.path().join("tiny-av.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("ltxv".to_string()),
            ),
            ("general.file_type", gguf_file::Value::U32(7)),
            (
                "model_version",
                gguf_file::Value::String("2.5.0".to_string()),
            ),
        ];
        let metadata_refs: Vec<(&str, &gguf_file::Value)> = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect();
        let tensor_refs: Vec<(&str, &QTensor)> = quantized
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        gguf_file::write(&mut file, &metadata_refs, &tensor_refs).unwrap();
        (path, dequantized)
    }

    /// The shared AV fixture map, normalized to what a GGUF container can
    /// carry: transformer_blocks.1's FP8 flavor (F8E4M3 weights plus SCALAR
    /// `input_scale`/`weight_scale` sidecars) becomes plain F32 — a rank-0
    /// tensor is unrepresentable in GGUF and the real tiers store only
    /// float and ggml-quantized arrays.
    fn gguf_fixture_tensors(
        config: crate::ltx2::model::video_transformer::Ltx2VideoTransformer3DModelConfig,
    ) -> StdHashMap<String, Tensor> {
        av_transformer_tensors(config, false)
            .into_iter()
            .filter(|(name, _)| !name.ends_with(".input_scale") && !name.ends_with(".weight_scale"))
            .map(|(name, tensor)| {
                let tensor = if tensor.dtype() == DType::F8E4M3 {
                    tensor.to_dtype(DType::F32).unwrap()
                } else {
                    tensor
                };
                (name, tensor)
            })
            .collect()
    }

    fn gguf_test_config() -> crate::ltx2::model::video_transformer::Ltx2VideoTransformer3DModelConfig
    {
        let mut config = tiny_av_config();
        // Q8_0 needs row widths on the 32-element grid, so widen the tiny
        // config until the block linears are quantizable, mirroring the real
        // tiers where every quantized tensor has `in % 256 == 0`.
        config.attention_head_dim = 32;
        config.audio_attention_head_dim = 32;
        config.cross_attention_dim = 32;
        config.audio_cross_attention_dim = 32;
        config
    }

    /// The backend must reproduce the safetensors twin's forward exactly:
    /// the reference model is built from the DEQUANTIZED tensors (the values
    /// the file actually stores), so both paths run identical dense F32
    /// arithmetic and quantization error cancels out of the comparison.
    #[test]
    fn synthetic_gguf_forward_matches_the_safetensors_twin() {
        let device = Device::Cpu;
        let config = gguf_test_config();
        let tensors = gguf_fixture_tensors(config.clone());
        let dir = tempfile::tempdir().unwrap();
        let (path, dequantized) = write_gguf_fixture(&dir, &tensors);

        let backend = Ltx2GgufBackend::from_path(&path).unwrap();
        assert_eq!(
            backend.shape_of("transformer_blocks.0.attn1.to_q.weight"),
            Some(Shape::from(vec![32usize, 32])),
            "GGUF dims must land in PyTorch order"
        );
        let gguf_vb = VarBuilder::from_backend(Box::new(backend), DType::F32, device.clone());
        let gguf_model = Ltx2AvTransformer3DModel::new_streaming(&config, gguf_vb, None).unwrap();

        let reference_vb = VarBuilder::from_tensors(dequantized, DType::F32, &device);
        let reference_model =
            Ltx2AvTransformer3DModel::new_streaming(&config, reference_vb, None).unwrap();

        let video_hidden_states = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6],
            (1, 3, config.in_channels),
            &device,
        )
        .unwrap();
        let audio_hidden_states = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.5, -0.4],
            (1, 2, config.audio_in_channels),
            &device,
        )
        .unwrap();
        let video_encoder_hidden_states = Tensor::from_vec(
            (0..16).map(|i| ((i % 19) as f32 - 9.0) / 16.0).collect(),
            (1, 4, config.caption_channels),
            &device,
        )
        .unwrap();
        let audio_encoder_hidden_states = Tensor::from_vec(
            (0..16)
                .map(|i| (((i + 9) % 19) as f32 - 9.0) / 16.0)
                .collect(),
            (1, 4, config.caption_channels),
            &device,
        )
        .unwrap();
        let timestep = Tensor::new(&[0.75f32], &device).unwrap();
        let video_positions = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0,
                2.0, 3.0,
            ],
            (1, 3, 3, 2),
            &device,
        )
        .unwrap();
        let audio_positions =
            Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 2.0], (1, 1, 2, 2), &device).unwrap();

        let forward = |model: &Ltx2AvTransformer3DModel| {
            model
                .forward(
                    &video_hidden_states,
                    Some(&audio_hidden_states),
                    &video_encoder_hidden_states,
                    Some(&audio_encoder_hidden_states),
                    &timestep,
                    &timestep,
                    Some(&timestep),
                    Some(&timestep),
                    None,
                    None,
                    None,
                    None,
                    &video_positions,
                    Some(&audio_positions),
                    None,
                )
                .unwrap()
        };

        let (gguf_video, gguf_audio) = forward(&gguf_model);
        let (reference_video, reference_audio) = forward(&reference_model);
        assert_tensors_close(&gguf_video, &reference_video, 1e-5);
        let (gguf_audio, reference_audio) = (gguf_audio.unwrap(), reference_audio.unwrap());
        assert_tensors_close(&gguf_audio, &reference_audio, 1e-5);
    }

    /// The side channel serves exactly the quantized weights: raw payload
    /// bytes that rebuild into the same values, an id that round-trips, and
    /// refusals (not fabrications) for float-stored tensors.
    #[test]
    fn side_channel_serves_quantized_weights_and_refuses_dense_ones() {
        let config = gguf_test_config();
        let tensors = gguf_fixture_tensors(config.clone());
        let dir = tempfile::tempdir().unwrap();
        let (path, dequantized) = write_gguf_fixture(&dir, &tensors);
        let backend = Ltx2GgufBackend::from_path(&path).unwrap();

        let quantized_key = "transformer_blocks.0.attn1.to_q.weight";
        assert!(backend.contains_tensor(&format!("{quantized_key}.gguf_blocks")));
        assert!(backend.contains_tensor(&format!("{quantized_key}.gguf_dtype")));
        // patchify_proj has in_features 2: float-stored, plain arm only.
        assert!(backend.contains_tensor("patchify_proj.weight"));
        assert!(!backend.contains_tensor("patchify_proj.weight.gguf_blocks"));
        assert!(!backend.contains_tensor("missing.weight.gguf_blocks"));

        let blocks = backend
            .lookup(
                &format!("{quantized_key}.gguf_blocks"),
                DType::U8,
                &Device::Cpu,
            )
            .unwrap();
        let dtype_id = backend
            .lookup(
                &format!("{quantized_key}.gguf_dtype"),
                DType::U32,
                &Device::Cpu,
            )
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0];
        assert_eq!(gguf_dtype_from_id(dtype_id), Some(GgmlDType::Q8_0));

        let rebuilt =
            qtensor_from_side_channel(&blocks, dtype_id, vec![32, 32], &Device::Cpu).unwrap();
        let rebuilt_dense = rebuilt.dequantize(&Device::Cpu).unwrap();
        assert_tensors_close(&rebuilt_dense, &dequantized[quantized_key], 0.0);

        assert!(backend
            .lookup("patchify_proj.weight.gguf_blocks", DType::U8, &Device::Cpu)
            .is_err());
    }
    /// Real-checkpoint smoke (#1414 deliverable): resolve tensors from an
    /// INSTALLED GGUF tier and run one quantized linear forward on the
    /// first CUDA device (CPU when none). Ignored by default — it needs
    /// `MOLD_LTX25_GGUF_SMOKE=<absolute path to the .gguf>` and is run
    /// manually; nothing committed depends on the real file.
    #[test]
    #[ignore = "needs MOLD_LTX25_GGUF_SMOKE=<installed .gguf> (manual smoke)"]
    fn real_gguf_smoke_resolves_and_forwards_one_quantized_linear() {
        use candle_core::Module;
        use std::sync::Arc;

        let path = PathBuf::from(
            std::env::var_os("MOLD_LTX25_GGUF_SMOKE")
                .expect("set MOLD_LTX25_GGUF_SMOKE to an installed LTX-2.5 GGUF"),
        );
        assert!(checkpoint_is_gguf(&path), "{}", path.display());
        let backend = Ltx2GgufBackend::from_path(&path).unwrap();

        // The video AdaLN table is F32 and non-block: the plain dense arm.
        let adaln_shape = backend
            .shape_of("adaln_single.linear.weight")
            .expect("real tier carries the video AdaLN table");
        let adaln = backend
            .lookup("adaln_single.linear.weight", DType::F32, &Device::Cpu)
            .unwrap();
        assert_eq!(adaln.shape(), &adaln_shape);
        assert_eq!(adaln_shape.dims()[0], 36_864, "22B nine-component AdaLN");

        // A block linear serves the packed side channel.
        let key = "transformer_blocks.0.attn1.to_q.weight";
        let weight_shape = backend.shape_of(key).expect("block 0 to_q present");
        let dims = weight_shape.dims().to_vec();
        assert!(
            backend.contains_tensor(&format!("{key}.gguf_blocks")),
            "quantized block linear must expose the side channel"
        );
        let blocks = backend
            .lookup(&format!("{key}.gguf_blocks"), DType::U8, &Device::Cpu)
            .unwrap();
        let dtype_id = backend
            .lookup(&format!("{key}.gguf_dtype"), DType::U32, &Device::Cpu)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0];

        let device = if candle_core::utils::metal_is_available() {
            Device::new_metal(0).unwrap()
        } else if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap()
        } else {
            Device::Cpu
        };
        let weight = qtensor_from_side_channel(&blocks, dtype_id, dims.clone(), &device).unwrap();
        let kernel_dtype = crate::ltx2::backend::compute_dtype(&device);
        let linear = crate::quantized_linear::QuantizedLinear::new(
            Arc::new(weight),
            None,
            &device,
            kernel_dtype,
            ltx2_qmatmul_enabled(),
        )
        .unwrap();

        let (out_dim, in_dim) = (dims[0], dims[1]);
        let xs_values: Vec<f32> = (0..4 * in_dim)
            .map(|index| ((index % 29) as f32 - 14.0) / 32.0)
            .collect();
        let xs = Tensor::from_vec(xs_values, (1, 4, in_dim), &device)
            .unwrap()
            .to_dtype(kernel_dtype)
            .unwrap();
        let out = linear
            .forward(&xs)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        assert_eq!(out.dims(), [1, 4, out_dim]);

        // Reference: the same weight dequantized on the CPU, dense F32.
        let dense = backend.lookup(key, DType::F32, &Device::Cpu).unwrap();
        let reference = candle_nn::Linear::new(dense, None)
            .forward(
                &xs.to_dtype(DType::F32)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap(),
            )
            .unwrap();
        let peak = reference
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let diff = (out.clone() - reference)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(peak.is_finite() && diff.is_finite(), "non-finite output");
        assert!(
            diff <= 0.02 * peak + 1e-3,
            "device forward diverges from the CPU dense reference: {diff} vs peak {peak}"
        );
        println!(
            "real GGUF smoke: {} — {key} {dims:?} dtype id {dtype_id} on {device:?}, \
             max |diff| {diff:.3e} against peak {peak:.3e}",
            path.display()
        );
    }
}
