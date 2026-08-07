//! fp8-scaled safetensors for the Wan DiT.
//!
//! ComfyUI's lineage of fp8 checkpoints stores each quantized projection as
//! `float8_e4m3fn` weights plus a per-tensor F32 `<module>.scale_weight`, and
//! marks the file with a single `scaled_fp8` tensor. That marker is the whole
//! contract, and **neither of its two fields is a boolean**
//! (`comfy/utils.py:1409-1420`):
//!
//! - its **dtype** names the fp8 flavour (an `F32` marker means e4m3 — the
//!   fallback ComfyUI keeps for its oldest files, `utils.py:1414-1415`);
//! - its **element count** selects the matmul path: 2 means "dequantize to the
//!   compute dtype and matmul there", anything else means the weights may feed
//!   a native fp8 matmul.
//!
//! The two artifacts mold has to read disagree on both fields, which is why a
//! single "is this fp8" flag picks the wrong path for one of them. Headers
//! re-parsed here, not inferred (companion recon:
//! `tmp/wan-research/report-layer6-formats.md`):
//!
//! | File | marker dtype | marker shape | scale shape |
//! | --- | --- | --- | --- |
//! | `Wan2_2-I2V-A14B-{HIGH,LOW}_fp8_e4m3fn_scaled_KJ` | `F8_E4M3` | `[2]` | `[1]` |
//! | `umt5_xxl_fp8_e4m3fn_scaled` | `F8_E4M3` | `[0]` | `[]` |
//!
//! Dequantization is a plain multiply — upstream's scaled-fp8 linear is
//! `F.linear(input, weight * scale_weight, bias)` and its inverse `set_weight`
//! divides by the same scalar, which pins the direction
//! (`comfy/ops.py:361-364,368-379` at v0.3.30, before the `QuantizedTensor`
//! rewrite moved the arithmetic into `comfy_kitchen`). Upstream scales the
//! *input* instead when the input is the smaller tensor; that is only valid
//! because the scale is one scalar for the whole weight, which is also why a
//! per-channel scale is refused here rather than broadcast.
//!
//! Kijai publishes an `fp8_e5m2` twin of every Wan DiT in the same directory.
//! candle has no `DType` for e5m2 at all, so the flavour is read and refused by
//! name — otherwise the failure surfaces as an unsupported-dtype error from
//! inside a memory map, naming neither the file nor the fix.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{bail, DType, Error, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::wan::model::transformer::WanLinear;

/// The marker's key, after whatever prefix the checkpoint puts on its weights
/// (`comfy/utils.py:1409` formats it as `"{}scaled_fp8".format(model_prefix)`).
const MARKER_KEY: &str = "scaled_fp8";

/// The prefix the Comfy-Org full-pipeline repacks put on every DiT key.
const DIFFUSION_MODEL_PREFIX: &str = "model.diffusion_model.";

/// Per-module scale suffix.
const SCALE_SUFFIX: &str = "scale_weight";

/// The fp8 flavour a `scaled_fp8` marker names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fp8Flavour {
    /// `float8_e4m3fn` — 4 exponent bits, 3 mantissa bits. The only flavour
    /// candle can represent, and the one every file mold ships uses.
    E4M3,
    /// `float8_e5m2` — wider range, 2 mantissa bits. Recognized so it can be
    /// refused by name.
    E5M2,
}

impl Fp8Flavour {
    fn parse(dtype: &str) -> Result<Self> {
        match dtype {
            // ComfyUI treats an F32 marker as e4m3 (`utils.py:1414-1415`); the
            // marker's payload is never read, only its dtype and length.
            "F8_E4M3" | "F32" => Ok(Self::E4M3),
            "F8_E5M2" => Ok(Self::E5M2),
            other => bail!(
                "Wan fp8: `{MARKER_KEY}` has dtype {other}, which names no fp8 flavour — \
                 expected F8_E4M3, F8_E5M2, or the legacy F32 spelling of e4m3"
            ),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::E4M3 => "e4m3",
            Self::E5M2 => "e5m2",
        }
    }
}

/// Which matmul path the marker selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fp8Matmul {
    /// Marker element count 2. The quantizer decided this checkpoint must not
    /// feed a native fp8 matmul — both Kijai A14B experts are marked this way.
    FullPrecision,
    /// Any other element count (0 in every shipped file). The weights are
    /// calibrated for `torch._scaled_mm`-style execution.
    Native,
}

impl Fp8Matmul {
    fn label(self) -> &'static str {
        match self {
            Self::FullPrecision => "full-precision matmul",
            Self::Native => "native fp8 matmul",
        }
    }
}

/// A parsed `scaled_fp8` marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScaledFp8Marker {
    pub flavour: Fp8Flavour,
    pub matmul: Fp8Matmul,
}

impl ScaledFp8Marker {
    /// `dtype` and `elements` come straight from the safetensors header entry.
    /// A rank-0 shape is one element (torch's `nelement()`), a `[0]` shape is
    /// zero — the shipped files use both spellings.
    fn parse(dtype: &str, elements: usize) -> Result<Self> {
        Ok(Self {
            flavour: Fp8Flavour::parse(dtype)?,
            matmul: if elements == 2 {
                Fp8Matmul::FullPrecision
            } else {
                Fp8Matmul::Native
            },
        })
    }
}

impl std::fmt::Display for ScaledFp8Marker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}, {}", self.flavour.label(), self.matmul.label())
    }
}

/// What a header probe learned about an fp8-scaled checkpoint.
#[derive(Clone, Debug)]
pub(crate) struct ScaledFp8Checkpoint {
    pub marker: ScaledFp8Marker,
    /// `__metadata__.model_type`, when the exporter wrote one. Kijai puts the
    /// expert role there — but the HIGH and LOW files of one pair spell it
    /// `Wan2_2-I2V-A14B-high` and `Wan22-I2V-A14B-low`, so only the trailing
    /// role is dependable and an exact match on the whole string is not.
    pub model_type: Option<String>,
}

impl ScaledFp8Checkpoint {
    /// Refuse an fp8-scaled checkpoint this loader cannot honour.
    pub fn ensure_supported(&self, path: &Path) -> Result<()> {
        if self.marker.flavour == Fp8Flavour::E5M2 {
            bail!(
                "{}{} is fp8-e5m2-scaled and mold reads the e4m3 flavour only — every \
                 repository that publishes an e5m2 Wan DiT publishes the matching \
                 `_fp8_e4m3fn_scaled_` file beside it",
                path.display(),
                self.model_type_suffix()
            );
        }
        Ok(())
    }

    fn model_type_suffix(&self) -> String {
        match &self.model_type {
            Some(model_type) => format!(" (model_type {model_type:?})"),
            None => String::new(),
        }
    }
}

/// Read the `scaled_fp8` marker out of a checkpoint's header, or `None` when
/// the file is not fp8-scaled.
///
/// Header-only: no tensor is materialized, and the raw JSON is parsed rather
/// than going through candle, because candle cannot represent an e5m2 tensor
/// and would fail before the flavour could be reported.
pub(crate) fn probe_scaled_fp8(paths: &[PathBuf]) -> Result<Option<ScaledFp8Checkpoint>> {
    for path in paths {
        let header = crate::weight_loader::read_safetensors_header(path)
            .map_err(|err| Error::Msg(format!("{err:#}")))?;
        // A sharded export puts the marker in whichever shard the quantizer
        // wrote first; look in all of them before concluding it is absent.
        let Some(entry) = header
            .get(MARKER_KEY)
            .or_else(|| header.get(&format!("{DIFFUSION_MODEL_PREFIX}{MARKER_KEY}")))
        else {
            continue;
        };
        let dtype = entry.get("dtype").and_then(serde_json::Value::as_str);
        let shape = entry.get("shape").and_then(serde_json::Value::as_array);
        let (Some(dtype), Some(shape)) = (dtype, shape) else {
            bail!(
                "{}: `{MARKER_KEY}` is not a safetensors tensor entry",
                path.display()
            );
        };
        let elements = shape
            .iter()
            .map(|dim| dim.as_u64().unwrap_or(0) as usize)
            .product::<usize>();
        let model_type = header
            .get("__metadata__")
            .and_then(|meta| meta.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        return Ok(Some(ScaledFp8Checkpoint {
            marker: ScaledFp8Marker::parse(dtype, elements)?,
            model_type,
        }));
    }
    Ok(None)
}

/// A biased linear whose weight is fp8 with one per-tensor scale.
///
/// The weight stays `F8E4M3` resident — 1 byte per parameter, which is what
/// makes a 14B expert 14 GB instead of 28 GB — and is dequantized into the
/// activation dtype on each call, then dropped. That is ComfyUI's "manual
/// cast", and mold's own [`crate::qwen_image`] fp8 precedent.
struct WanFp8Linear {
    /// `[out, in]`, `F8E4M3`.
    weight: Tensor,
    /// One element, already in the compute dtype.
    scale: Tensor,
    /// `[out]`, already in the compute dtype.
    bias: Tensor,
}

impl Module for WanFp8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        // `weight * scale_weight` (`comfy/ops.py:362`), materialized for this
        // call only.
        let weight = self
            .weight
            .to_dtype(dtype)?
            .broadcast_mul(&self.scale.to_dtype(dtype)?)?;

        // Flatten the leading axes rather than broadcasting the weight the way
        // `candle_nn::Linear` does: Wan feeds these `[batch, tokens, dim]`, and
        // a broadcast weight would materialize one copy per batch entry.
        let dims = x.dims();
        let (&in_dim, leading) = dims
            .split_last()
            .ok_or_else(|| Error::Msg("Wan fp8: linear input is a scalar".to_string()))?;
        let rows: usize = leading.iter().product();
        let out_dim = self.bias.dim(0)?;
        let out = x.reshape((rows, in_dim))?.matmul(&weight.t()?)?;
        let mut out_dims = leading.to_vec();
        out_dims.push(out_dim);
        out.reshape(out_dims)?
            .broadcast_add(&self.bias.to_dtype(dtype)?)
    }
}

/// Load one biased projection out of an fp8-scaled checkpoint.
///
/// The file is mixed: in the Kijai A14B experts exactly the 400 block
/// projections are fp8 and the remaining 695 tensors — every norm, modulation,
/// bias, the patch embedding, and the five top-level projections — are F32. So
/// the dispatch is per module, on the weight's own dtype, and the unquantized
/// half is cast down to the compute dtype the way ComfyUI's `cast_bias_weight`
/// does.
pub(crate) fn load_linear(
    vb: &VarBuilder<'_>,
    in_dim: usize,
    out_dim: usize,
    compute: DType,
) -> Result<WanLinear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?.to_dtype(compute)?;
    if weight.dtype() != DType::F8E4M3 {
        return Ok(Arc::new(candle_nn::Linear::new(
            weight.to_dtype(compute)?,
            Some(bias),
        )));
    }

    let prefix = vb.prefix();
    // Deliberately stricter than the qwen path's `.ok()`: an fp8 weight whose
    // scale went missing would silently run at scale 1.0, which is not a
    // subtle error but does not look like one either. Every shipped file pairs
    // each fp8 weight with exactly one scale, with no orphans in either
    // direction.
    let scale = vb.get_unchecked(SCALE_SUFFIX).map_err(|_| {
        Error::Msg(format!(
            "Wan fp8: `{prefix}.weight` is fp8 but `{prefix}.{SCALE_SUFFIX}` is missing — \
             a scaled-fp8 weight carries its own per-tensor scale, and defaulting to 1.0 \
             would rescale the whole projection"
        ))
    })?;
    if scale.elem_count() != 1 {
        bail!(
            "Wan fp8: `{prefix}.{SCALE_SUFFIX}` has {} elements; mold reads the per-tensor \
             scale ComfyUI writes (shape [] or [1]), not a per-channel one",
            scale.elem_count()
        );
    }

    Ok(Arc::new(WanFp8Linear {
        weight,
        // `[]` and `[1]` both occur; normalize so the multiply broadcasts.
        scale: scale.reshape((1,))?.to_dtype(compute)?,
        bias,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;

    /// Write a header-only fixture: the tensor payloads are never read by the
    /// probe, so zero bytes of the right length are enough.
    fn write_header_fixture(
        path: &Path,
        entries: &[(&str, SafeDtype, Vec<usize>, Vec<u8>)],
        metadata: Option<HashMap<String, String>>,
    ) {
        let mut map: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, dtype, shape, data) in entries {
            map.insert(
                (*name).to_string(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            );
        }
        serialize_to_file(&map, &metadata, path).unwrap();
    }

    /// The two fields of the marker are independent, and the two files mold
    /// reads disagree on the element count. Reading it as a boolean picks the
    /// wrong matmul path for one of them.
    #[test]
    fn marker_dtype_names_the_flavour_and_element_count_names_the_matmul() {
        // The Kijai A14B experts.
        assert_eq!(
            ScaledFp8Marker::parse("F8_E4M3", 2).unwrap(),
            ScaledFp8Marker {
                flavour: Fp8Flavour::E4M3,
                matmul: Fp8Matmul::FullPrecision,
            }
        );
        // umt5_xxl_fp8_e4m3fn_scaled: same flavour, shape [0].
        assert_eq!(
            ScaledFp8Marker::parse("F8_E4M3", 0).unwrap(),
            ScaledFp8Marker {
                flavour: Fp8Flavour::E4M3,
                matmul: Fp8Matmul::Native,
            }
        );
        // A rank-0 marker is one element, which is not 2.
        assert_eq!(
            ScaledFp8Marker::parse("F8_E4M3", 1).unwrap().matmul,
            Fp8Matmul::Native
        );
        // ComfyUI's legacy F32 spelling still means e4m3.
        assert_eq!(
            ScaledFp8Marker::parse("F32", 2).unwrap().flavour,
            Fp8Flavour::E4M3
        );
        // The e5m2 twin is recognized rather than mistaken for e4m3.
        assert_eq!(
            ScaledFp8Marker::parse("F8_E5M2", 2).unwrap().flavour,
            Fp8Flavour::E5M2
        );

        let err = ScaledFp8Marker::parse("BF16", 2).unwrap_err().to_string();
        assert!(err.contains("BF16"), "{err}");
        assert!(err.contains("no fp8 flavour"), "{err}");
    }

    #[test]
    fn the_marker_reads_as_its_flavour_and_matmul_path() {
        assert_eq!(
            ScaledFp8Marker::parse("F8_E4M3", 2).unwrap().to_string(),
            "e4m3, full-precision matmul"
        );
        assert_eq!(
            ScaledFp8Marker::parse("F8_E4M3", 0).unwrap().to_string(),
            "e4m3, native fp8 matmul"
        );
    }

    /// A file with no marker is not fp8-scaled, and must not be routed as one.
    #[test]
    fn a_checkpoint_without_a_marker_probes_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plain.safetensors");
        write_header_fixture(
            &path,
            &[(
                "patch_embedding.weight",
                SafeDtype::F32,
                vec![2],
                vec![0; 8],
            )],
            None,
        );
        assert!(probe_scaled_fp8(&[path]).unwrap().is_none());
    }

    /// The marker, the `model.diffusion_model.` prefix, and the metadata
    /// discriminator all come out of one header read.
    #[test]
    fn the_probe_reads_the_marker_the_prefix_and_the_model_type() {
        let dir = tempfile::tempdir().unwrap();

        // Bare keys, as the Kijai A14B files actually ship.
        let bare = dir.path().join("bare.safetensors");
        write_header_fixture(
            &bare,
            &[(MARKER_KEY, SafeDtype::F8_E4M3, vec![2], vec![0; 2])],
            Some(HashMap::from([(
                "model_type".to_string(),
                "Wan2_2-I2V-A14B-high".to_string(),
            )])),
        );
        let probed = probe_scaled_fp8(&[bare]).unwrap().unwrap();
        assert_eq!(probed.marker.matmul, Fp8Matmul::FullPrecision);
        assert_eq!(probed.model_type.as_deref(), Some("Wan2_2-I2V-A14B-high"));

        // Prefixed keys, as a full-pipeline repack would ship.
        let prefixed = dir.path().join("prefixed.safetensors");
        write_header_fixture(
            &prefixed,
            &[(
                &format!("{DIFFUSION_MODEL_PREFIX}{MARKER_KEY}"),
                SafeDtype::F8_E4M3,
                vec![0],
                Vec::new(),
            )],
            None,
        );
        let probed = probe_scaled_fp8(&[prefixed]).unwrap().unwrap();
        assert_eq!(probed.marker.matmul, Fp8Matmul::Native);
        assert!(probed.model_type.is_none());
    }

    /// candle has no e5m2 dtype, so the refusal has to happen at the marker —
    /// and it has to name the file and the alternative.
    #[test]
    fn an_e5m2_checkpoint_is_refused_by_name() {
        let checkpoint = ScaledFp8Checkpoint {
            marker: ScaledFp8Marker::parse("F8_E5M2", 2).unwrap(),
            model_type: Some("Wan2_2-I2V-A14B-high".to_string()),
        };
        let err = checkpoint
            .ensure_supported(Path::new("/models/HIGH_fp8_e5m2_scaled_KJ.safetensors"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("e5m2"), "{err}");
        assert!(err.contains("e4m3"), "{err}");
        assert!(err.contains("HIGH_fp8_e5m2_scaled_KJ"), "{err}");
        assert!(err.contains("Wan2_2-I2V-A14B-high"), "{err}");

        // The e4m3 twin of the same file loads.
        let supported = ScaledFp8Checkpoint {
            marker: ScaledFp8Marker::parse("F8_E4M3", 2).unwrap(),
            model_type: None,
        };
        assert!(supported
            .ensure_supported(Path::new("ok.safetensors"))
            .is_ok());
    }

    /// Build a one-module fp8 fixture and load it through [`load_linear`].
    ///
    /// `scale` is written under `shape`, which is the tolerance this exercises:
    /// the DiT files use `[1]` and the UMT5 file uses `[]`, and both have to
    /// reach the same multiply.
    fn fp8_linear_fixture(
        dir: &Path,
        name: &str,
        scale: f32,
        shape: Vec<usize>,
    ) -> VarBuilder<'static> {
        let device = Device::Cpu;
        let path = dir.join(name);
        let weight =
            Tensor::from_vec(vec![1.0f32, -2.0, 0.5, 4.0, -0.25, 8.0], (2, 3), &device).unwrap();
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "proj.weight".to_string(),
            weight.to_dtype(DType::F8E4M3).unwrap(),
        );
        tensors.insert(
            "proj.bias".to_string(),
            Tensor::from_vec(vec![0.5f32, -0.5], 2, &device).unwrap(),
        );
        tensors.insert(
            "proj.scale_weight".to_string(),
            Tensor::from_vec(vec![scale], (), &device)
                .unwrap()
                .reshape(shape)
                .unwrap(),
        );
        candle_core::safetensors::save(&tensors, &path).unwrap();

        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.unwrap();
        let backend = crate::weight_loader::NativeFp8Backend::from_mmap(mmap);
        VarBuilder::from_backend(Box::new(backend), DType::F32, device)
    }

    /// The dequantized forward must equal `(fp8_weight * scale) @ x + bias`
    /// computed by hand, for both scale spellings.
    #[test]
    fn an_fp8_projection_dequantizes_by_multiplying_its_per_tensor_scale() {
        let dir = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &device).unwrap();

        for (index, shape) in [vec![1usize], Vec::new()].into_iter().enumerate() {
            let vb =
                fp8_linear_fixture(dir.path(), &format!("fp8-{index}.safetensors"), 2.0, shape);
            let linear = load_linear(&vb.pp("proj"), 3, 2, DType::F32).unwrap();
            let got: Vec<f32> = linear
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            // Every fixture value is exactly representable in e4m3, so the
            // expectation is exact rather than approximate.
            let want = vec![
                2.0 * (1.0 * 1.0 + -2.0 * 2.0 + 0.5 * 3.0) + 0.5,
                2.0 * (4.0 * 1.0 + -0.25 * 2.0 + 8.0 * 3.0) - 0.5,
            ];
            assert_eq!(got, want, "scale spelling {index}");
        }
    }

    /// An fp8 weight without its scale must fail loudly. Silently applying 1.0
    /// produces a model that runs and is wrong by the scale factor.
    #[test]
    fn an_fp8_weight_without_a_scale_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orphan.safetensors");
        let device = Device::Cpu;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "proj.weight".to_string(),
            Tensor::zeros((2, 3), DType::F32, &device)
                .unwrap()
                .to_dtype(DType::F8E4M3)
                .unwrap(),
        );
        tensors.insert(
            "proj.bias".to_string(),
            Tensor::zeros(2, DType::F32, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.unwrap();
        let backend = crate::weight_loader::NativeFp8Backend::from_mmap(mmap);
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, device);

        let err = load_linear(&vb.pp("proj"), 3, 2, DType::F32)
            .err()
            .expect("an fp8 weight without a scale must not load")
            .to_string();
        assert!(err.contains("proj.scale_weight"), "{err}");
        assert!(err.contains("missing"), "{err}");
    }

    /// A per-channel scale would broadcast against the wrong axis and quietly
    /// rescale by column. Refuse it instead.
    #[test]
    fn a_per_channel_scale_is_refused_rather_than_broadcast() {
        let dir = tempfile::tempdir().unwrap();
        let vb = fp8_linear_fixture(dir.path(), "per-channel.safetensors", 2.0, vec![1]);
        // Rebuild the fixture with a 2-element scale.
        let path = dir.path().join("per-channel-2.safetensors");
        let device = Device::Cpu;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "proj.weight".to_string(),
            Tensor::zeros((2, 3), DType::F32, &device)
                .unwrap()
                .to_dtype(DType::F8E4M3)
                .unwrap(),
        );
        tensors.insert(
            "proj.bias".to_string(),
            Tensor::zeros(2, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "proj.scale_weight".to_string(),
            Tensor::from_vec(vec![2.0f32, 3.0], 2, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.unwrap();
        let backend = crate::weight_loader::NativeFp8Backend::from_mmap(mmap);
        let per_channel = VarBuilder::from_backend(Box::new(backend), DType::F32, device);

        assert!(load_linear(&vb.pp("proj"), 3, 2, DType::F32).is_ok());
        let err = load_linear(&per_channel.pp("proj"), 3, 2, DType::F32)
            .err()
            .expect("a per-channel scale must not load")
            .to_string();
        assert!(err.contains("2 elements"), "{err}");
        assert!(err.contains("per-channel"), "{err}");
    }

    /// The unquantized half of an fp8 file still has to load. In the shipped
    /// experts that is 695 tensors, all F32 while the compute dtype is bf16, so
    /// the cast is the load-bearing part — the qwen precedent this mirrors gets
    /// away without one only because its files are bf16 throughout.
    ///
    /// The fixture inverts the direction (F16 on disk, F32 compute) because
    /// candle's CPU backend has no bf16 matmul; what it pins is that the
    /// on-disk dtype is never assumed to be the compute dtype.
    #[test]
    fn an_unquantized_weight_in_an_fp8_file_loads_at_the_compute_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.safetensors");
        let device = Device::Cpu;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "proj.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap(),
        );
        tensors.insert(
            "proj.bias".to_string(),
            Tensor::from_vec(vec![1.0f32, 1.0], 2, &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap(),
        );
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.unwrap();
        let backend = crate::weight_loader::NativeFp8Backend::from_mmap(mmap);
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, device.clone());

        let linear = load_linear(&vb.pp("proj"), 3, 2, DType::F32).unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (1, 3), &device).unwrap();
        let out = linear.forward(&x).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, vec![7.0, 16.0]);
    }
}
