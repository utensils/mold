//! One authority for "what does an LTX-2 transformer file weigh".
//!
//! Admission (`mold-server`) and the runtime (`mold-inference`) both need the
//! same answers from a transformer checkpoint before anything is loaded: how
//! many blocks it has, how many bytes each block and the non-block remainder
//! occupy at rest / once materialized at the compute dtype / when the loader
//! keeps the packed form, the AdaLN table width that sets the conditioned
//! activation cost, and which storage format the file is. Two independent
//! readers answered those questions differently (INT8 ConvRot was priced at
//! its raw bytes while the loader widened it to BF16; the AdaLN width was read
//! from whichever `*adaln_single.linear.weight` a `HashMap` yielded last), so
//! every consumer now reads this one index and a parity test pins them.
//!
//! Pure std + `serde_json` (no candle) so it lives in `mold-core`: safetensors
//! comes from the 8-byte length prefix + JSON header, GGUF from
//! [`crate::gguf_probe`] plus the ggml block/type-size table below. Tensor
//! payloads are never read.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use crate::gguf_probe::{read_gguf_header, GgufHeader};

/// Prefix the diffusers/ComfyUI single-file layouts put in front of every
/// transformer tensor. Canonical names never carry it.
pub const DIFFUSION_MODEL_PREFIX: &str = "model.diffusion_model.";

/// Largest safetensors header this reader will buffer. The 2.5 22B header is
/// ~1 MB; anything near this bound is not a checkpoint mold knows.
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 512 * 1024 * 1024;

/// Top-level prefixes inside an LTX-2 single-file checkpoint that are *not*
/// transformer weights. A combined 19B export carries ~2.4 GB of `vae.*`
/// next to the transformer; charging those to the transformer's GPU
/// residency would be as wrong as charging its non-block tensors nothing.
/// Prompt-encoder-side weights (`text_encoder*`, `prompt_encoder`,
/// `tokenizer`) are separate phases with their own placement policy and
/// never co-reside with the denoise peak.
pub const LTX2_NON_TRANSFORMER_PREFIXES: &[&str] = &[
    "vae",
    "audio_vae",
    "vocoder",
    "text_encoder",
    "text_encoders",
    "prompt_encoder",
    "tokenizer",
    "conditioner",
    "first_stage_model",
    "cond_stage_model",
    "latent_upsampler",
    "spatial_upsampler",
    "temporal_upsampler",
];

/// Heads that count toward [`Ltx2TransformerWeightIndex::vae_bytes_at_rest`].
const LTX2_VAE_PREFIXES: &[&str] = &["vae", "first_stage_model"];

/// The transformer-side prompt connectors. They are loaded by the prompt
/// encoder (`ltx2/text/prompt_encoder.rs`), not by the denoiser, and are
/// reported under their own role so a consumer can decide whether they are
/// resident during denoise.
const LTX2_PROMPT_ENCODER_PREFIXES: &[&str] =
    &["video_embeddings_connector", "audio_embeddings_connector"];

/// The video branch's own modulation table. `prompt_adaln_single`,
/// `audio_adaln_single`, `audio_prompt_adaln_single`, and the four `av_ca_*`
/// gates all share the `adaln_single.linear.weight` suffix, so this is an
/// EXACT canonical key, never an `ends_with`.
pub const LTX2_VIDEO_ADALN_KEY: &str = "adaln_single.linear.weight";

/// Storage format of an LTX-2 transformer checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ltx2WeightFormat {
    /// Dense safetensors (BF16/F16/F32 tensors); everything materializes at
    /// the compute dtype.
    Bf16,
    /// Block linears stored as `F8_E4M3` with per-tensor scales; the loader
    /// keeps the float8 tensor resident and widens per call.
    Fp8,
    /// ComfyUI `int8_tensorwise` + ConvRot: `I8 .weight` with an `F32
    /// .weight_scale` and a `U8 .comfy_quant` marker per linear.
    Int8ConvRot,
    /// NVFP4 (packed `U8 .weight`, `.weight_scale` block scales,
    /// `.weight_scale_2` tensor scale).
    Nvfp4,
    /// GGUF (`ggml` K-quants / Q8_0 for the block linears, F32 elsewhere).
    Gguf,
}

/// Where a tensor sits in the transformer's residency model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ltx2TensorRole {
    /// Inside `transformer_blocks.{i}` / `blocks.{i}`.
    Block(usize),
    /// Transformer tensor outside every block — `patchify_proj`,
    /// `adaln_single.linear`, `proj_out`, the modulation tables, and their
    /// audio twins.
    NonBlock,
    /// `video_embeddings_connector.*` / `audio_embeddings_connector.*`.
    PromptEncoder,
    /// VAE, vocoder, text encoder, upsampler tensors bundled in a single-file
    /// export. Never transformer residency.
    NonTransformer,
}

/// How a tensor is priced on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ltx2TensorKind {
    /// A parameter the loader materializes at the compute dtype.
    Dense,
    /// A quantized parameter: packed bytes at rest, `elements × elem_size`
    /// once widened.
    Quantized,
    /// A quantization sidecar (`.weight_scale`, `.weight_scale_2`) the loader
    /// consumes: never resident on the widened path, at rest on the packed
    /// path.
    Scale,
    /// A quantization marker (`.comfy_quant`) that never reaches the device.
    Marker,
}

/// One tensor's header facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ltx2TensorFact {
    /// Canonical (prefix-stripped, ComfyUI-named) tensor name.
    pub name: String,
    /// Shape in PyTorch order.
    pub shape: Vec<u64>,
    /// Logical element count (for a packed NVFP4 weight this is the
    /// unpacked count, twice the stored byte count).
    pub elements: u64,
    /// Storage dtype as spelled by the container (`BF16`, `I8`, `Q4_K`, …).
    pub dtype: String,
    /// Bytes the tensor occupies in the file.
    pub at_rest_bytes: u64,
    pub kind: Ltx2TensorKind,
    pub role: Ltx2TensorRole,
}

impl Ltx2TensorFact {
    /// Bytes once the tensor is materialized at a compute dtype of
    /// `elem_size` bytes per element.
    pub fn widened_bytes(&self, elem_size: u64) -> u64 {
        match self.kind {
            Ltx2TensorKind::Dense | Ltx2TensorKind::Quantized => {
                self.elements.saturating_mul(elem_size)
            }
            Ltx2TensorKind::Scale | Ltx2TensorKind::Marker => 0,
        }
    }

    /// Bytes when a loader keeps the packed form: quantized weights and their
    /// scales at rest, dense tensors at the compute dtype, markers nothing.
    pub fn packed_bytes(&self, elem_size: u64) -> u64 {
        match self.kind {
            Ltx2TensorKind::Dense => self.elements.saturating_mul(elem_size),
            Ltx2TensorKind::Quantized | Ltx2TensorKind::Scale => self.at_rest_bytes,
            Ltx2TensorKind::Marker => 0,
        }
    }
}

/// Header-derived facts about one LTX-2 transformer checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ltx2TransformerWeightIndex {
    format: Ltx2WeightFormat,
    /// Sorted by canonical name, so every derived total is deterministic.
    tensors: Vec<Ltx2TensorFact>,
    num_layers: usize,
}

fn invalid(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// Strip the single-file prefixes and rewrite the diffusers-style segments
/// (`proj_in`, `time_embed`, `norm_q`, `norm_k`) to the ComfyUI names the
/// checkpoints and mold's loaders use. Idempotent on an already canonical
/// name, so it accepts either the model's logical name or a file key.
pub fn canonical_tensor_name(name: &str) -> String {
    let core = name
        .strip_prefix(DIFFUSION_MODEL_PREFIX)
        .or_else(|| name.strip_prefix("diffusion_model."))
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);
    core.split('.')
        .map(|component| match component {
            "proj_in" => "patchify_proj",
            "time_embed" => "adaln_single",
            "norm_q" => "q_norm",
            "norm_k" => "k_norm",
            _ => component,
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Block index of a canonical transformer tensor name, if it lives inside
/// `transformer_blocks.{i}` / `blocks.{i}`. The connectors' own
/// `transformer_1d_blocks.{i}` deliberately do not match.
pub fn ltx2_transformer_block_index(canonical: &str) -> Option<usize> {
    let mut components = canonical.split('.');
    while let Some(component) = components.next() {
        if component == "transformer_blocks" || component == "blocks" {
            return components.next()?.parse().ok();
        }
    }
    None
}

fn head(canonical: &str) -> &str {
    canonical.split('.').next().unwrap_or_default()
}

/// Role of a canonical tensor name.
pub fn ltx2_tensor_role(canonical: &str) -> Ltx2TensorRole {
    let head = head(canonical);
    if LTX2_NON_TRANSFORMER_PREFIXES.contains(&head) {
        return Ltx2TensorRole::NonTransformer;
    }
    if LTX2_PROMPT_ENCODER_PREFIXES.contains(&head) {
        return Ltx2TensorRole::PromptEncoder;
    }
    match ltx2_transformer_block_index(canonical) {
        Some(index) => Ltx2TensorRole::Block(index),
        None => Ltx2TensorRole::NonBlock,
    }
}

fn safetensors_dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    })
}

/// ggml `(block_size, type_size)` for the storage types an LTX-2.5 GGUF may
/// carry — the same eight ids `ltx25_probe` accepts.
fn ggml_type_layout(ggml_type: u32) -> Option<(&'static str, u64, u64)> {
    Some(match ggml_type {
        0 => ("F32", 1, 4),
        1 => ("F16", 1, 2),
        8 => ("Q8_0", 32, 34),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        30 => ("BF16", 1, 2),
        _ => return None,
    })
}

fn ggml_type_is_quantized(ggml_type: u32) -> bool {
    !matches!(ggml_type, 0 | 1 | 30)
}

fn element_count(shape: &[u64]) -> io::Result<u64> {
    shape
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| invalid("tensor shape overflows u64"))
}

#[derive(serde::Deserialize)]
struct SafetensorsHeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

impl Ltx2TransformerWeightIndex {
    /// Read the header of a safetensors or GGUF transformer checkpoint.
    /// Dispatches on the `GGUF` magic; never reads tensor payloads.
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        drop(file);
        if &magic == b"GGUF" {
            let header = read_gguf_header(path)?;
            Self::from_gguf_header(&header)
        } else {
            let mut file = File::open(path)?;
            let mut len_bytes = [0u8; 8];
            file.read_exact(&mut len_bytes)?;
            let header_len = u64::from_le_bytes(len_bytes);
            if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
                return Err(invalid(format!(
                    "implausible safetensors header length {header_len} in {}",
                    path.display()
                )));
            }
            let mut header = vec![0u8; header_len as usize];
            file.read_exact(&mut header)?;
            Self::from_safetensors_header(&header)
        }
    }

    /// Build from the JSON bytes of a safetensors header.
    pub fn from_safetensors_header(header: &[u8]) -> io::Result<Self> {
        let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header)
            .map_err(|error| invalid(format!("parse safetensors header: {error}")))?;
        let mut parsed: BTreeMap<String, SafetensorsHeaderTensor> = BTreeMap::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let tensor: SafetensorsHeaderTensor = serde_json::from_value(value)
                .map_err(|error| invalid(format!("tensor metadata for {name}: {error}")))?;
            parsed.insert(canonical_tensor_name(&name), tensor);
        }
        let keys: BTreeSet<&str> = parsed.keys().map(String::as_str).collect();
        let format = detect_safetensors_format(&parsed, &keys);

        let mut tensors = Vec::with_capacity(parsed.len());
        for (name, tensor) in &parsed {
            let span = tensor
                .data_offsets
                .1
                .checked_sub(tensor.data_offsets.0)
                .ok_or_else(|| invalid(format!("invalid data offsets for tensor {name}")))?;
            let dtype_size = safetensors_dtype_size(&tensor.dtype)
                .ok_or_else(|| invalid(format!("unknown dtype {} for {name}", tensor.dtype)))?;
            let stored = element_count(&tensor.shape)?;
            let expected = stored
                .checked_mul(dtype_size)
                .ok_or_else(|| invalid(format!("tensor {name} byte size overflows")))?;
            if expected != span {
                return Err(invalid(format!(
                    "safetensors tensor {name} reports {span} bytes but shape/dtype imply {expected}"
                )));
            }
            let (kind, elements) = classify_safetensors_tensor(format, name, tensor, stored, &keys);
            tensors.push(Ltx2TensorFact {
                name: name.clone(),
                shape: tensor.shape.clone(),
                elements,
                dtype: tensor.dtype.clone(),
                at_rest_bytes: span,
                kind,
                role: ltx2_tensor_role(name),
            });
        }
        Ok(Self::finish(format, tensors))
    }

    /// Build from a parsed GGUF header.
    pub fn from_gguf_header(header: &GgufHeader) -> io::Result<Self> {
        let mut tensors = Vec::with_capacity(header.tensors.len());
        let mut sorted: Vec<(&String, &crate::gguf_probe::GgufTensorInfo)> =
            header.tensors.iter().collect();
        sorted.sort_by(|left, right| left.0.cmp(right.0));
        for (name, info) in sorted {
            let canonical = canonical_tensor_name(name);
            let (dtype, block_size, type_size) =
                ggml_type_layout(info.ggml_type).ok_or_else(|| {
                    invalid(format!(
                        "unsupported GGML dtype {} for tensor {name}",
                        info.ggml_type
                    ))
                })?;
            let elements = element_count(&info.shape)?;
            if elements % block_size != 0 {
                return Err(invalid(format!(
                    "GGUF tensor {name} has {elements} elements, not a multiple of the {dtype} block size {block_size}"
                )));
            }
            let at_rest_bytes = (elements / block_size)
                .checked_mul(type_size)
                .ok_or_else(|| invalid(format!("GGUF tensor {name} byte size overflows")))?;
            let kind = if ggml_type_is_quantized(info.ggml_type) {
                Ltx2TensorKind::Quantized
            } else {
                Ltx2TensorKind::Dense
            };
            tensors.push(Ltx2TensorFact {
                role: ltx2_tensor_role(&canonical),
                name: canonical,
                shape: info.shape.clone(),
                elements,
                dtype: dtype.to_string(),
                at_rest_bytes,
                kind,
            });
        }
        Ok(Self::finish(Ltx2WeightFormat::Gguf, tensors))
    }

    fn finish(format: Ltx2WeightFormat, mut tensors: Vec<Ltx2TensorFact>) -> Self {
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        let num_layers = tensors
            .iter()
            .filter_map(|tensor| match tensor.role {
                Ltx2TensorRole::Block(index) => Some(index + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        Self {
            format,
            tensors,
            num_layers,
        }
    }

    pub fn format(&self) -> Ltx2WeightFormat {
        self.format
    }

    pub fn is_gguf(&self) -> bool {
        self.format == Ltx2WeightFormat::Gguf
    }

    /// Whether the transformer block linears are stored as float8. Decided
    /// from `transformer_blocks.1.attn1.to_q.weight` / `…ff.net.0.proj.weight`
    /// exactly as the loader's own probe was.
    pub fn is_fp8(&self) -> bool {
        self.format == Ltx2WeightFormat::Fp8
    }

    pub fn is_convrot(&self) -> bool {
        self.format == Ltx2WeightFormat::Int8ConvRot
    }

    pub fn is_nvfp4(&self) -> bool {
        self.format == Ltx2WeightFormat::Nvfp4
    }

    /// Highest block index present plus one (0 for a file with no blocks).
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn tensors(&self) -> &[Ltx2TensorFact] {
        &self.tensors
    }

    /// Look a tensor up by canonical name.
    pub fn tensor(&self, canonical: &str) -> Option<&Ltx2TensorFact> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(canonical))
            .ok()
            .map(|index| &self.tensors[index])
    }

    /// Output width of the video branch's `adaln_single.linear` — the number
    /// of AdaLN components × the inner dim (six for the 19B, 24,576; nine for
    /// the 22B, 36,864). Read from the exact canonical key so
    /// `prompt_adaln_single` (`[8192, 4096]`) can never be mistaken for it.
    pub fn adaln_dim(&self) -> Option<u64> {
        self.tensor(LTX2_VIDEO_ADALN_KEY)
            .and_then(|tensor| tensor.shape.first().copied())
    }

    fn block_totals(&self, price: impl Fn(&Ltx2TensorFact) -> u64) -> Vec<u64> {
        let mut blocks = vec![0u64; self.num_layers];
        for tensor in &self.tensors {
            if let Ltx2TensorRole::Block(index) = tensor.role {
                blocks[index] = blocks[index].saturating_add(price(tensor));
            }
        }
        blocks
    }

    fn total_where(
        &self,
        select: impl Fn(&Ltx2TensorFact) -> bool,
        price: impl Fn(&Ltx2TensorFact) -> u64,
    ) -> u64 {
        self.tensors
            .iter()
            .filter(|tensor| select(tensor))
            .fold(0u64, |acc, tensor| acc.saturating_add(price(tensor)))
    }

    /// Per-block bytes in the file.
    pub fn block_bytes_at_rest(&self) -> Vec<u64> {
        self.block_totals(|tensor| tensor.at_rest_bytes)
    }

    /// Per-block bytes once every parameter is materialized at a compute
    /// dtype of `elem_size` bytes.
    pub fn block_bytes_widened(&self, elem_size: u64) -> Vec<u64> {
        self.block_totals(|tensor| tensor.widened_bytes(elem_size))
    }

    /// Per-block bytes when the loader keeps quantized weights packed.
    pub fn block_bytes_packed(&self, elem_size: u64) -> Vec<u64> {
        self.block_totals(|tensor| tensor.packed_bytes(elem_size))
    }

    fn is_non_block(tensor: &Ltx2TensorFact) -> bool {
        matches!(
            tensor.role,
            Ltx2TensorRole::NonBlock | Ltx2TensorRole::PromptEncoder
        )
    }

    /// Transformer tensors outside every block, connectors included (the
    /// loaders allocate these after every resident block, so they are
    /// reserved rather than discovered). See [`Self::prompt_encoder_bytes_at_rest`]
    /// for the connector share.
    pub fn non_block_bytes_at_rest(&self) -> u64 {
        self.total_where(Self::is_non_block, |tensor| tensor.at_rest_bytes)
    }

    pub fn non_block_bytes_widened(&self, elem_size: u64) -> u64 {
        self.total_where(Self::is_non_block, |tensor| tensor.widened_bytes(elem_size))
    }

    pub fn non_block_bytes_packed(&self, elem_size: u64) -> u64 {
        self.total_where(Self::is_non_block, |tensor| tensor.packed_bytes(elem_size))
    }

    fn is_prompt_encoder(tensor: &Ltx2TensorFact) -> bool {
        tensor.role == Ltx2TensorRole::PromptEncoder
    }

    /// The `*_embeddings_connector.*` share of the non-block total.
    pub fn prompt_encoder_bytes_at_rest(&self) -> u64 {
        self.total_where(Self::is_prompt_encoder, |tensor| tensor.at_rest_bytes)
    }

    pub fn prompt_encoder_bytes_widened(&self, elem_size: u64) -> u64 {
        self.total_where(Self::is_prompt_encoder, |tensor| {
            tensor.widened_bytes(elem_size)
        })
    }

    pub fn prompt_encoder_bytes_packed(&self, elem_size: u64) -> u64 {
        self.total_where(Self::is_prompt_encoder, |tensor| {
            tensor.packed_bytes(elem_size)
        })
    }

    /// Bytes of a bundled video VAE (`vae.*` / `first_stage_model.*`) in a
    /// combined export; zero for the split 2.5 packs.
    pub fn vae_bytes_at_rest(&self) -> u64 {
        self.total_where(
            |tensor| {
                tensor.role == Ltx2TensorRole::NonTransformer
                    && LTX2_VAE_PREFIXES.contains(&head(&tensor.name))
            },
            |tensor| tensor.at_rest_bytes,
        )
    }

    fn largest_quantized_elements(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|tensor| tensor.kind == Ltx2TensorKind::Quantized)
            .map(|tensor| tensor.elements)
            .max()
            .unwrap_or(0)
    }

    /// Per-forward scratch the loader needs beside the resident weights:
    /// one dequantized linear. GGUF dequantizes through an F32 copy and
    /// narrows it (6 B/element); INT8 ConvRot widens one linear to BF16
    /// (2 B/element); dense and float8 checkpoints need none.
    pub fn transient_bytes(&self) -> u64 {
        let per_element = match self.format {
            Ltx2WeightFormat::Gguf => 6,
            Ltx2WeightFormat::Int8ConvRot => 2,
            Ltx2WeightFormat::Bf16 | Ltx2WeightFormat::Fp8 | Ltx2WeightFormat::Nvfp4 => 0,
        };
        self.largest_quantized_elements()
            .saturating_mul(per_element)
    }

    /// Per-block bytes today's loaders actually keep on the device while
    /// blocks are resident: dense and float8 checkpoints at their packed
    /// size (float8 stays float8 and widens per call), GGUF and NVFP4 at
    /// their packed size, INT8 ConvRot fully widened — every ConvRot arm
    /// reconstructs BF16 weights on the device, which is exactly the 2×
    /// the raw-byte pricing missed.
    pub fn resident_block_bytes(&self, elem_size: u64) -> Vec<u64> {
        match self.format {
            Ltx2WeightFormat::Int8ConvRot => self.block_bytes_widened(elem_size),
            Ltx2WeightFormat::Bf16
            | Ltx2WeightFormat::Fp8
            | Ltx2WeightFormat::Nvfp4
            | Ltx2WeightFormat::Gguf => self.block_bytes_packed(elem_size),
        }
    }

    /// Non-block counterpart of [`Self::resident_block_bytes`].
    pub fn resident_non_block_bytes(&self, elem_size: u64) -> u64 {
        match self.format {
            Ltx2WeightFormat::Int8ConvRot => self.non_block_bytes_widened(elem_size),
            Ltx2WeightFormat::Bf16
            | Ltx2WeightFormat::Fp8
            | Ltx2WeightFormat::Nvfp4
            | Ltx2WeightFormat::Gguf => self.non_block_bytes_packed(elem_size),
        }
    }
}

/// The loader's float8 probe: block 1's `attn1.to_q` / `ff.net.0.proj`.
const FP8_PROBE_KEYS: &[&str] = &[
    "transformer_blocks.1.attn1.to_q.weight",
    "transformer_blocks.1.ff.net.0.proj.weight",
];

fn detect_safetensors_format(
    tensors: &BTreeMap<String, SafetensorsHeaderTensor>,
    keys: &BTreeSet<&str>,
) -> Ltx2WeightFormat {
    if keys.iter().any(|key| key.ends_with(".weight_scale_2")) {
        return Ltx2WeightFormat::Nvfp4;
    }
    let convrot = tensors.iter().any(|(name, tensor)| {
        name.ends_with(".comfy_quant")
            || (tensor.dtype == "I8"
                && name
                    .strip_suffix(".weight")
                    .is_some_and(|base| keys.contains(format!("{base}.weight_scale").as_str())))
    });
    if convrot {
        return Ltx2WeightFormat::Int8ConvRot;
    }
    for key in FP8_PROBE_KEYS {
        if let Some(tensor) = tensors.get(*key) {
            return if tensor.dtype == "F8_E4M3" {
                Ltx2WeightFormat::Fp8
            } else {
                Ltx2WeightFormat::Bf16
            };
        }
    }
    Ltx2WeightFormat::Bf16
}

fn classify_safetensors_tensor(
    format: Ltx2WeightFormat,
    name: &str,
    tensor: &SafetensorsHeaderTensor,
    stored_elements: u64,
    keys: &BTreeSet<&str>,
) -> (Ltx2TensorKind, u64) {
    if name.ends_with(".comfy_quant") {
        return (Ltx2TensorKind::Marker, stored_elements);
    }
    if name.ends_with(".weight_scale") || name.ends_with(".weight_scale_2") {
        return (Ltx2TensorKind::Scale, stored_elements);
    }
    let base = name.strip_suffix(".weight");
    match format {
        Ltx2WeightFormat::Int8ConvRot => {
            let quantized = tensor.dtype == "I8"
                && base.is_some_and(|base| {
                    keys.contains(format!("{base}.weight_scale").as_str())
                        || keys.contains(format!("{base}.comfy_quant").as_str())
                });
            if quantized {
                (Ltx2TensorKind::Quantized, stored_elements)
            } else {
                (Ltx2TensorKind::Dense, stored_elements)
            }
        }
        Ltx2WeightFormat::Nvfp4 => {
            let packed = tensor.dtype == "U8"
                && base
                    .is_some_and(|base| keys.contains(format!("{base}.weight_scale_2").as_str()));
            if packed {
                // Two 4-bit codes per stored byte.
                (Ltx2TensorKind::Quantized, stored_elements.saturating_mul(2))
            } else {
                (Ltx2TensorKind::Dense, stored_elements)
            }
        }
        Ltx2WeightFormat::Fp8 | Ltx2WeightFormat::Bf16 => {
            if tensor.dtype == "F8_E4M3" || tensor.dtype == "F8_E5M2" {
                (Ltx2TensorKind::Quantized, stored_elements)
            } else {
                (Ltx2TensorKind::Dense, stored_elements)
            }
        }
        Ltx2WeightFormat::Gguf => unreachable!("GGUF tensors are classified from ggml types"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const BF16: u64 = 2;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/ltx25")
            .join(name)
    }

    fn int8_conv_index() -> Ltx2TransformerWeightIndex {
        Ltx2TransformerWeightIndex::read(&fixture("distilled-int8-convrot.header.safetensors"))
            .unwrap()
    }

    fn q4_k_m_index() -> Ltx2TransformerWeightIndex {
        Ltx2TransformerWeightIndex::read(&fixture("distilled-q4-k-m.header.gguf")).unwrap()
    }

    /// A safetensors header whose tensor map is built in the given order and
    /// serialized in that order, so the AdaLN read is exercised against both
    /// orderings a JSON object can arrive in.
    fn header_with_two_adaln_tables(reverse: bool) -> Vec<u8> {
        let mut entries = vec![
            (
                "model.diffusion_model.adaln_single.linear.weight",
                serde_json::json!({"dtype": "BF16", "shape": [36864, 4096], "data_offsets": [0, 301989888]}),
            ),
            (
                "model.diffusion_model.prompt_adaln_single.linear.weight",
                serde_json::json!({"dtype": "BF16", "shape": [8192, 4096], "data_offsets": [301989888, 369098752]}),
            ),
            (
                "model.diffusion_model.audio_adaln_single.linear.weight",
                serde_json::json!({"dtype": "BF16", "shape": [18432, 2048], "data_offsets": [369098752, 444596224]}),
            ),
        ];
        if reverse {
            entries.reverse();
        }
        let body = entries
            .into_iter()
            .map(|(name, value)| format!("{}:{}", serde_json::to_string(name).unwrap(), value))
            .collect::<Vec<_>>()
            .join(",");
        format!("{{{body}}}").into_bytes()
    }

    #[test]
    fn adaln_width_is_the_exact_video_key_regardless_of_header_order() {
        for reverse in [false, true] {
            let index = Ltx2TransformerWeightIndex::from_safetensors_header(
                &header_with_two_adaln_tables(reverse),
            )
            .unwrap();
            assert_eq!(index.adaln_dim(), Some(36_864), "reverse={reverse}");
        }
        assert_eq!(int8_conv_index().adaln_dim(), Some(36_864));
        assert_eq!(q4_k_m_index().adaln_dim(), Some(36_864));
        // Neither look-alike qualifies on its own.
        let only_prompt = br#"{"model.diffusion_model.prompt_adaln_single.linear.weight":{"dtype":"BF16","shape":[8192,4096],"data_offsets":[0,67108864]}}"#;
        assert_eq!(
            Ltx2TransformerWeightIndex::from_safetensors_header(only_prompt)
                .unwrap()
                .adaln_dim(),
            None
        );
    }

    #[test]
    fn int8_convrot_fixture_is_priced_widened_and_packed_by_one_rule() {
        let index = int8_conv_index();
        assert_eq!(index.format(), Ltx2WeightFormat::Int8ConvRot);
        assert!(index.is_convrot());
        assert!(!index.is_fp8());
        assert!(!index.is_gguf());
        // Blocks 1..46 were cut from the golden header; the index still
        // reports the checkpoint's 48-layer geometry from block 47.
        assert_eq!(index.num_layers(), 48);
        let at_rest = index.block_bytes_at_rest();
        let widened = index.block_bytes_widened(BF16);
        let packed = index.block_bytes_packed(BF16);
        assert_eq!(at_rest[0], 388_065_632);
        assert_eq!(at_rest[47], 388_065_632);
        assert_eq!(at_rest[1], 0);
        // I8 weights ×2, biases/norms at BF16, `.weight_scale` and
        // `.comfy_quant` consumed by the backend — never resident widened.
        assert_eq!(widened[0], 773_349_760);
        assert_eq!(widened[47], widened[0]);
        // Packed keeps the I8 bytes and the F32 scales; markers cost nothing.
        assert_eq!(packed[0], 387_867_008);
        assert_eq!(index.resident_block_bytes(BF16), widened);
        assert_eq!(index.non_block_bytes_at_rest(), 2_875_797_760);
        assert_eq!(index.non_block_bytes_widened(BF16), 4_887_262_720);
        assert_eq!(index.non_block_bytes_packed(BF16), 2_875_766_272);
        assert_eq!(
            index.resident_non_block_bytes(BF16),
            index.non_block_bytes_widened(BF16)
        );
        assert_eq!(index.prompt_encoder_bytes_at_rest(), 2_020_843_264);
        assert_eq!(index.prompt_encoder_bytes_widened(BF16), 4_032_332_800);
        assert_eq!(index.vae_bytes_at_rest(), 0);
        // One widened `ff.net.2` / `ff.net.0.proj` linear: 16384 × 4096 × 2.
        assert_eq!(index.transient_bytes(), 134_217_728);
    }

    #[test]
    fn q4_k_m_fixture_is_priced_from_ggml_block_layouts() {
        let index = q4_k_m_index();
        assert_eq!(index.format(), Ltx2WeightFormat::Gguf);
        assert!(index.is_gguf());
        assert!(!index.is_fp8());
        assert!(!index.is_convrot());
        assert_eq!(index.num_layers(), 48);
        assert_eq!(index.tensors().len(), 4_349);
        let at_rest = index.block_bytes_at_rest();
        assert_eq!(at_rest.len(), 48);
        assert_eq!(at_rest[0], 249_582_336);
        assert_eq!(at_rest.iter().sum::<u64>(), 12_624_023_552);
        assert_eq!(index.block_bytes_widened(BF16)[0], 773_349_760);
        // Packed keeps every K-quant at its ggml size and narrows the block's
        // F32 norms/tables/biases to the compute dtype: 418,176 B per block
        // less than at rest.
        let packed = index.block_bytes_packed(BF16);
        assert_eq!(packed[0], 249_164_160);
        assert_eq!(at_rest[0] - packed[0], 418_176);
        assert_eq!(packed.iter().sum::<u64>(), 12_603_951_104);
        assert_eq!(index.resident_block_bytes(BF16), packed);
        // Every non-block linear is F32 in the Abiray export; packed pricing
        // narrows the dense tensors to BF16.
        assert_eq!(index.non_block_bytes_at_rest(), 3_063_213_056);
        assert_eq!(index.non_block_bytes_widened(BF16), 4_887_262_720);
        // The connectors are quantized in this export while every other
        // non-block tensor is F32, so the packed figure sits between at rest
        // and fully widened.
        assert_eq!(index.non_block_bytes_packed(BF16), 2_207_200_768);
        assert_eq!(index.resident_non_block_bytes(BF16), 2_207_200_768);
        assert_eq!(index.prompt_encoder_bytes_packed(BF16), 1_352_270_848);
        assert_eq!(index.prompt_encoder_bytes_at_rest(), 1_353_353_216);
        // F32 dequantize + BF16 narrow of the largest Q4_K linear (67,108,864
        // elements) is the per-forward scratch a quantized arm needs.
        assert_eq!(index.transient_bytes(), 402_653_184);
        let q = index
            .tensor("transformer_blocks.0.ff.net.0.proj.weight")
            .unwrap();
        assert_eq!(q.dtype, "Q4_K");
        assert_eq!(q.kind, Ltx2TensorKind::Quantized);
        assert_eq!(q.shape, vec![16384, 4096]);
        assert_eq!(q.at_rest_bytes, 16384 * 4096 / 256 * 144);
    }

    #[test]
    fn connectors_are_prompt_encoder_tensors_in_both_containers() {
        for index in [int8_conv_index(), q4_k_m_index()] {
            let video = index
                .tensor("video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight")
                .unwrap();
            assert_eq!(video.role, Ltx2TensorRole::PromptEncoder);
            let audio = index
                .tensor("audio_embeddings_connector.learnable_registers")
                .unwrap();
            assert_eq!(audio.role, Ltx2TensorRole::PromptEncoder);
            assert_eq!(
                index.tensor("patchify_proj.weight").unwrap().role,
                Ltx2TensorRole::NonBlock
            );
            assert_eq!(
                index
                    .tensor("transformer_blocks.47.attn1.to_q.weight")
                    .unwrap()
                    .role,
                Ltx2TensorRole::Block(47)
            );
            assert!(index.prompt_encoder_bytes_at_rest() > 0);
            assert!(index.prompt_encoder_bytes_at_rest() < index.non_block_bytes_at_rest());
        }
    }

    #[test]
    fn canonical_names_absorb_the_single_file_prefixes_and_diffusers_segments() {
        assert_eq!(
            canonical_tensor_name("model.diffusion_model.proj_in.weight"),
            "patchify_proj.weight"
        );
        assert_eq!(
            canonical_tensor_name("blocks.0.norm_q.weight"),
            "blocks.0.q_norm.weight"
        );
        assert_eq!(
            canonical_tensor_name("model.transformer_blocks.0.attn1.to_q.weight"),
            "transformer_blocks.0.attn1.to_q.weight"
        );
        assert_eq!(
            canonical_tensor_name("blocks.0.patchify_proj_in.weight"),
            "blocks.0.patchify_proj_in.weight"
        );
        assert_eq!(
            canonical_tensor_name("time_embed.linear.weight"),
            "adaln_single.linear.weight"
        );
        assert_eq!(
            canonical_tensor_name("patchify_proj.weight"),
            "patchify_proj.weight"
        );
        assert_eq!(
            ltx2_transformer_block_index("transformer_blocks.12.attn1.to_q.weight"),
            Some(12)
        );
        assert_eq!(
            ltx2_transformer_block_index(
                "video_embeddings_connector.transformer_1d_blocks.3.ff.net.2.weight"
            ),
            None
        );
        assert_eq!(
            ltx2_tensor_role("vae.decoder.conv_out.weight"),
            Ltx2TensorRole::NonTransformer
        );
        assert_eq!(
            ltx2_tensor_role("text_encoders.gemma.weight"),
            Ltx2TensorRole::NonTransformer
        );
        assert_eq!(
            ltx2_tensor_role("proj_out.weight"),
            Ltx2TensorRole::NonBlock
        );
    }

    #[test]
    fn dense_and_fp8_headers_take_the_loader_probe_key() {
        let fp8 = br#"{"model.diffusion_model.transformer_blocks.1.attn1.to_q.weight":{"dtype":"F8_E4M3","shape":[4096,4096],"data_offsets":[0,16777216]},"model.diffusion_model.transformer_blocks.1.attn1.k_norm.weight":{"dtype":"BF16","shape":[4096],"data_offsets":[16777216,16785408]},"vae.decoder.conv_out.weight":{"dtype":"BF16","shape":[3,128],"data_offsets":[16785408,16786176]}}"#;
        let index = Ltx2TransformerWeightIndex::from_safetensors_header(fp8).unwrap();
        assert_eq!(index.format(), Ltx2WeightFormat::Fp8);
        assert!(index.is_fp8());
        assert_eq!(index.num_layers(), 2);
        // Float8 stays float8 on the device; the BF16 norm is materialized
        // at the compute dtype. A wider compute dtype only widens the norm.
        assert_eq!(index.block_bytes_at_rest(), vec![0, 16_777_216 + 8_192]);
        assert_eq!(
            index.resident_block_bytes(BF16),
            vec![0, 16_777_216 + 8_192]
        );
        assert_eq!(index.resident_block_bytes(4), vec![0, 16_777_216 + 16_384]);
        assert_eq!(index.block_bytes_widened(BF16), vec![0, 33_554_432 + 8_192]);
        assert_eq!(index.vae_bytes_at_rest(), 768);
        assert_eq!(index.transient_bytes(), 0);

        let dense = br#"{"transformer_blocks.1.attn1.to_q.weight":{"dtype":"BF16","shape":[4096,4096],"data_offsets":[0,33554432]},"transformer_blocks.0.attn1.to_q.weight":{"dtype":"F32","shape":[4096,4096],"data_offsets":[33554432,100663296]}}"#;
        let index = Ltx2TransformerWeightIndex::from_safetensors_header(dense).unwrap();
        assert_eq!(index.format(), Ltx2WeightFormat::Bf16);
        assert!(!index.is_fp8());
        // An F32 tensor in a dense checkpoint narrows to the compute dtype.
        assert_eq!(index.block_bytes_at_rest(), vec![67_108_864, 33_554_432]);
        assert_eq!(
            index.resident_block_bytes(BF16),
            vec![33_554_432, 33_554_432]
        );
        assert_eq!(index.transient_bytes(), 0);
    }

    #[test]
    fn nvfp4_headers_report_unpacked_elements_and_packed_residency() {
        let nvfp4 = br#"{"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight":{"dtype":"U8","shape":[4096,2048],"data_offsets":[0,8388608]},"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale":{"dtype":"F8_E4M3","shape":[4096,256],"data_offsets":[8388608,9437184]},"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[9437184,9437188]}}"#;
        let index = Ltx2TransformerWeightIndex::from_safetensors_header(nvfp4).unwrap();
        assert_eq!(index.format(), Ltx2WeightFormat::Nvfp4);
        assert!(index.is_nvfp4());
        let weight = index
            .tensor("transformer_blocks.0.attn1.to_q.weight")
            .unwrap();
        assert_eq!(weight.kind, Ltx2TensorKind::Quantized);
        assert_eq!(weight.elements, 4096 * 4096);
        assert_eq!(index.block_bytes_at_rest(), vec![9_437_188]);
        assert_eq!(index.block_bytes_packed(BF16), vec![9_437_188]);
        assert_eq!(index.block_bytes_widened(BF16), vec![4096 * 4096 * 2]);
        assert_eq!(index.transient_bytes(), 0);
    }

    #[test]
    fn a_header_that_lies_about_its_byte_span_is_refused() {
        let lying = br#"{"transformer_blocks.0.attn1.to_q.weight":{"dtype":"BF16","shape":[4096,4096],"data_offsets":[0,16]}}"#;
        let error = Ltx2TransformerWeightIndex::from_safetensors_header(lying).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let unknown = br#"{"transformer_blocks.0.attn1.to_q.weight":{"dtype":"Q4_0","shape":[4096,4096],"data_offsets":[0,16]}}"#;
        assert!(Ltx2TransformerWeightIndex::from_safetensors_header(unknown).is_err());
    }

    #[test]
    fn read_dispatches_on_the_gguf_magic() {
        let dir = tempfile::tempdir().unwrap();
        let not_gguf = dir.path().join("dense.safetensors");
        let header = br#"{"transformer_blocks.0.attn1.to_q.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        std::fs::write(&not_gguf, bytes).unwrap();
        let index = Ltx2TransformerWeightIndex::read(&not_gguf).unwrap();
        assert_eq!(index.format(), Ltx2WeightFormat::Bf16);
        assert_eq!(index.block_bytes_at_rest(), vec![8]);

        // The golden GGUF header is a truncated file (no tensor data); the
        // bounded reader must accept exactly that.
        assert!(q4_k_m_index().is_gguf());

        let truncated = dir.path().join("truncated.gguf");
        std::fs::write(&truncated, b"GGUF\x03\0").unwrap();
        assert!(Ltx2TransformerWeightIndex::read(&truncated).is_err());
    }
}
