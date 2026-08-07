//! Header-only contracts for ComfyUI's pruned MiniMax H3 DiT checkpoints.
//!
//! The schema is derived from ComfyUI `a464ac33588ae182f81a090d910cfbf21e255b73`:
//! `comfy/ldm/minimax/model.py`, `comfy/model_detection.py`,
//! `comfy/quant_ops.py`, `comfy/ops.py`, and `comfy/utils.py`. Public artifact
//! sizes and content digests are pinned to the Comfy-Org repository revision
//! below. No model tensor data is needed by this inspector.
//!
//! This module deliberately produces a *candidate* strategy, not a runnable
//! checkpoint. The current legal gate forbids capturing the production headers
//! or verifying the full content digest in this repository's deployment
//! territory. A future compliance-approved change must pin those header
//! identities and qualify the runtime before any candidate can become an
//! execution authority.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use candle::DType;
use serde::de::{DeserializeOwned, Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

use super::dit::{
    expected_h3_weight_specs, H3AdaLnMode, H3PrecisionProfile, H3QkvLayout, H3TransformerConfig,
    H3TransformerTask,
};

pub const H3_COMFYUI_SOURCE_REVISION: &str = "a464ac33588ae182f81a090d910cfbf21e255b73";
pub const H3_COMFY_ORG_SOURCE_REVISION: &str = "eb8a16107c595128b3a578f82d2ce2f75920c355";

const MAX_HEADER_BYTES: u64 = 8 * 1024 * 1024;
const MAX_TENSORS: usize = 4_096;
const MAX_TENSOR_KEY_BYTES: usize = 4_096;
const MAX_TENSOR_RANK: usize = 8;
const CONVROT_GROUP_SIZE: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3ComfyPrunedFormat {
    Bf16,
    Fp8Scaled,
    Int8ConvRot,
}

impl H3ComfyPrunedFormat {
    pub const fn stable_id(self) -> &'static str {
        match self {
            Self::Bf16 => "minimax-h3.dit.comfy-pruned-adaln.bf16.v1",
            Self::Fp8Scaled => "minimax-h3.dit.comfy-pruned-adaln.fp8-scaled.v1",
            Self::Int8ConvRot => "minimax-h3.dit.comfy-pruned-adaln.int8-convrot.v1",
        }
    }
}

/// Exact public artifact authority. Detection never uses a filename: the
/// independently inspected header must agree with this source manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3ComfyPublishedArtifact {
    Fl2VaPrunedBf16,
    Fl2VaPrunedFp8Scaled,
    Fl2VaPrunedInt8ConvRot,
    Ref2VaPrunedBf16,
    Ref2VaPrunedFp8Scaled,
    Ref2VaPrunedInt8ConvRot,
}

impl H3ComfyPublishedArtifact {
    pub const fn task(self) -> H3TransformerTask {
        match self {
            Self::Fl2VaPrunedBf16 | Self::Fl2VaPrunedFp8Scaled | Self::Fl2VaPrunedInt8ConvRot => {
                H3TransformerTask::T2VaFl2Va
            }
            Self::Ref2VaPrunedBf16
            | Self::Ref2VaPrunedFp8Scaled
            | Self::Ref2VaPrunedInt8ConvRot => H3TransformerTask::Ref2Va,
        }
    }

    pub const fn format(self) -> H3ComfyPrunedFormat {
        match self {
            Self::Fl2VaPrunedBf16 | Self::Ref2VaPrunedBf16 => H3ComfyPrunedFormat::Bf16,
            Self::Fl2VaPrunedFp8Scaled | Self::Ref2VaPrunedFp8Scaled => {
                H3ComfyPrunedFormat::Fp8Scaled
            }
            Self::Fl2VaPrunedInt8ConvRot | Self::Ref2VaPrunedInt8ConvRot => {
                H3ComfyPrunedFormat::Int8ConvRot
            }
        }
    }

    pub const fn file_bytes(self) -> u64 {
        match self.format() {
            H3ComfyPrunedFormat::Bf16 => 40_225_724_176,
            H3ComfyPrunedFormat::Fp8Scaled => 20_958_205_608,
            H3ComfyPrunedFormat::Int8ConvRot => 20_970_379_616,
        }
    }

    pub const fn content_sha256(self) -> &'static str {
        match self {
            Self::Fl2VaPrunedBf16 => {
                "a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4"
            }
            Self::Fl2VaPrunedFp8Scaled => {
                "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993"
            }
            Self::Fl2VaPrunedInt8ConvRot => {
                "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
            }
            Self::Ref2VaPrunedBf16 => {
                "37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe"
            }
            Self::Ref2VaPrunedFp8Scaled => {
                "f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9"
            }
            Self::Ref2VaPrunedInt8ConvRot => {
                "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3ComfyRuntimeBackend {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3ComfyAccuracyTier {
    /// ComfyUI describes the precomputed curve as functionally equivalent to
    /// the removed full-width AdaLN branches; Mold has not independently run
    /// the gated model-quality qualification yet.
    SourceClaimedEquivalentPrunedBf16,
    UnqualifiedPrunedFp8,
    UnqualifiedPrunedInt8ConvRot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3ComfyCurveInterpolation {
    ClampUnitThenAdjacentLinear,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3ComfyMemoryAccounting {
    pub artifact_file_bytes: u64,
    pub header_bytes: u64,
    pub encoded_tensor_bytes: u64,
    pub host_mapped_file_bytes: Option<u64>,
    pub host_staging_bytes: Option<u64>,
    pub resident_device_weight_bytes: Option<u64>,
    pub dequantization_workspace_bytes: Option<u64>,
    pub streamed_weight_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3ComfyQuantizationPolicy {
    pub quantized_layers: Vec<String>,
    pub protected_weight_tensors: Vec<String>,
    pub policy_sha256: String,
}

/// Immutable metadata that placement can eventually snapshot. It is not an
/// execution token and cannot be passed to [`super::dit::H3Transformer::load`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3ComfyFrozenStrategyMetadata {
    pub stable_id: String,
    pub task: H3TransformerTask,
    pub format: H3ComfyPrunedFormat,
    pub adaln_mode: H3AdaLnMode,
    pub interpolation: H3ComfyCurveInterpolation,
    pub qkv_layout: H3QkvLayout,
    pub dense_compute_precision: H3PrecisionProfile,
    pub accuracy_tier: H3ComfyAccuracyTier,
    pub quantization_policy: H3ComfyQuantizationPolicy,
    pub memory: H3ComfyMemoryAccounting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3ComfyRuntimeRequirement {
    ComplianceApprovedHeaderIdentity,
    VerifiedFullContentSha256,
    RuntimeFactoryActivation,
    H3ScaledFp8KernelOrQualifiedDequantization,
    H3Int8ConvRotKernelOrQualifiedDequantization,
    QuantizedBackendQualification,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3ComfyRuntimeRejection {
    pub backend: H3ComfyRuntimeBackend,
    pub requirements: Vec<H3ComfyRuntimeRequirement>,
}

impl fmt::Display for H3ComfyRuntimeRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MiniMax H3 Comfy candidate is not executable on {:?}; unmet requirements: {:?}",
            self.backend, self.requirements
        )
    }
}

impl StdError for H3ComfyRuntimeRejection {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3ComfyCheckpointCandidate {
    pub artifact: H3ComfyPublishedArtifact,
    pub source_repository_revision: String,
    pub implementation_revision: String,
    pub expected_content_sha256: String,
    pub header_identity_sha256: String,
    pub tensor_count: usize,
    pub strategy: H3ComfyFrozenStrategyMetadata,
}

impl H3ComfyCheckpointCandidate {
    /// Fail closed until the legal/schema/content/kernel work named in the
    /// returned requirements has landed. Quantized formats additionally name
    /// the exact missing runtime capability instead of falling back to BF16.
    pub fn require_supported_runtime(
        &self,
        backend: H3ComfyRuntimeBackend,
    ) -> Result<(), H3ComfyRuntimeRejection> {
        let mut requirements = vec![
            H3ComfyRuntimeRequirement::ComplianceApprovedHeaderIdentity,
            H3ComfyRuntimeRequirement::VerifiedFullContentSha256,
            H3ComfyRuntimeRequirement::RuntimeFactoryActivation,
        ];
        match self.strategy.format {
            H3ComfyPrunedFormat::Bf16 => {}
            H3ComfyPrunedFormat::Fp8Scaled => {
                requirements
                    .push(H3ComfyRuntimeRequirement::H3ScaledFp8KernelOrQualifiedDequantization);
                if backend != H3ComfyRuntimeBackend::Cuda {
                    requirements.push(H3ComfyRuntimeRequirement::QuantizedBackendQualification);
                }
            }
            H3ComfyPrunedFormat::Int8ConvRot => {
                requirements
                    .push(H3ComfyRuntimeRequirement::H3Int8ConvRotKernelOrQualifiedDequantization);
                if backend != H3ComfyRuntimeBackend::Cuda {
                    requirements.push(H3ComfyRuntimeRequirement::QuantizedBackendQualification);
                }
            }
        }
        Err(H3ComfyRuntimeRejection {
            backend,
            requirements,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3ComfyCheckpointErrorCode {
    Io,
    InvalidHeader,
    InvalidMetadata,
    ConfigMismatch,
    TaskAuthorityMismatch,
    SourceSizeMismatch,
    FormatMismatch,
    TensorSchemaMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3ComfyCheckpointError {
    pub code: H3ComfyCheckpointErrorCode,
    pub message: String,
}

impl fmt::Display for H3ComfyCheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for H3ComfyCheckpointError {}

type InspectionResult<T> = Result<T, H3ComfyCheckpointError>;

fn failure(code: H3ComfyCheckpointErrorCode, message: impl Into<String>) -> H3ComfyCheckpointError {
    H3ComfyCheckpointError {
        code,
        message: message.into(),
    }
}

/// Inspect only a local safetensors header against a source-pinned published
/// artifact. This never reads tensor payload bytes and never returns an
/// executable checkpoint authority.
pub fn inspect_h3_comfy_published_header(
    path: &Path,
    requested_task: H3TransformerTask,
    artifact: H3ComfyPublishedArtifact,
) -> InspectionResult<H3ComfyCheckpointCandidate> {
    if requested_task != artifact.task() {
        return Err(failure(
            H3ComfyCheckpointErrorCode::TaskAuthorityMismatch,
            format!(
                "H3 requested task {requested_task:?} does not match source authority {:?}",
                artifact.task()
            ),
        ));
    }
    let parsed = read_safetensors_header(path)?;
    if parsed.file_len != artifact.file_bytes() {
        return Err(failure(
            H3ComfyCheckpointErrorCode::SourceSizeMismatch,
            format!(
                "H3 Comfy artifact size {} does not match source-pinned {}",
                parsed.file_len,
                artifact.file_bytes()
            ),
        ));
    }
    inspect_parsed_header(
        parsed,
        H3TransformerConfig::default(),
        requested_task,
        artifact.format(),
        Some(artifact),
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug)]
struct ParsedHeader {
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, HeaderTensor>,
    header_len: u64,
    file_len: u64,
    header_identity_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn read_safetensors_header(path: &Path) -> InspectionResult<ParsedHeader> {
    let mut file = File::open(path).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::Io,
            format!("failed to open H3 Comfy checkpoint: {error}"),
        )
    })?;
    let file_len = file
        .metadata()
        .map_err(|error| failure(H3ComfyCheckpointErrorCode::Io, error.to_string()))?
        .len();
    let mut length = [0u8; 8];
    file.read_exact(&mut length).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            format!("failed to read safetensors header length: {error}"),
        )
    })?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            format!("invalid H3 Comfy safetensors header length {header_len}"),
        ));
    }
    let mut bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut bytes).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            format!("failed to read H3 Comfy safetensors header: {error}"),
        )
    })?;
    let root = parse_strict_json(&bytes, "safetensors header")?;
    let object = root.as_object().ok_or_else(|| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            "H3 Comfy safetensors header must be a JSON object",
        )
    })?;
    if object.len() > MAX_TENSORS + 1 {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            "H3 Comfy safetensors tensor count exceeds the header bound",
        ));
    }
    let metadata = match object.get("__metadata__") {
        Some(value) => metadata_strings(value)?,
        None => BTreeMap::new(),
    };
    let data_len = file_len - header_len - 8;
    let mut tensors = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        if name.is_empty() || name.len() > MAX_TENSOR_KEY_BYTES {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidHeader,
                "H3 Comfy safetensors contains an empty or oversized tensor key",
            ));
        }
        let raw: RawTensorHeader = from_value(value.clone(), "tensor header")?;
        if raw.shape.len() > MAX_TENSOR_RANK {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidHeader,
                format!("H3 Comfy tensor {name:?} exceeds the rank bound"),
            ));
        }
        if raw.data_offsets[0] > raw.data_offsets[1] || raw.data_offsets[1] > data_len {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidHeader,
                format!("H3 Comfy tensor {name:?} has invalid data offsets"),
            ));
        }
        let elements = raw.shape.iter().try_fold(1u64, |total, dimension| {
            total.checked_mul(*dimension as u64).ok_or_else(|| {
                failure(
                    H3ComfyCheckpointErrorCode::InvalidHeader,
                    format!("H3 Comfy tensor {name:?} shape overflows"),
                )
            })
        })?;
        let expected_bytes = elements
            .checked_mul(dtype_size(&raw.dtype)?)
            .ok_or_else(|| {
                failure(
                    H3ComfyCheckpointErrorCode::InvalidHeader,
                    format!("H3 Comfy tensor {name:?} byte size overflows"),
                )
            })?;
        if raw.data_offsets[1] - raw.data_offsets[0] != expected_bytes {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidHeader,
                format!("H3 Comfy tensor {name:?} dtype/shape does not match its byte range"),
            ));
        }
        tensors.insert(
            name.clone(),
            HeaderTensor {
                dtype: raw.dtype,
                shape: raw.shape,
                data_offsets: raw.data_offsets,
            },
        );
    }
    if tensors.is_empty() {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            "H3 Comfy safetensors header contains no tensors",
        ));
    }
    let mut cursor = 0u64;
    let mut ranges = tensors
        .iter()
        .map(|(name, tensor)| (tensor.data_offsets, name.as_str()))
        .collect::<Vec<_>>();
    ranges.sort_by_key(|(offsets, _)| offsets[0]);
    for (offsets, name) in ranges {
        if offsets[0] != cursor {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidHeader,
                format!("H3 Comfy tensor data is non-contiguous before {name:?}"),
            ));
        }
        cursor = offsets[1];
    }
    if cursor != data_len {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            "H3 Comfy safetensors has unclaimed trailing tensor data",
        ));
    }
    let mut identity = Sha256::new();
    identity.update(length);
    identity.update(&bytes);
    Ok(ParsedHeader {
        metadata,
        tensors,
        header_len,
        file_len,
        header_identity_sha256: hex_digest(identity.finalize()),
    })
}

fn metadata_strings(value: &Value) -> InspectionResult<BTreeMap<String, String>> {
    let object = value.as_object().ok_or_else(|| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            "H3 Comfy __metadata__ must be a string map",
        )
    })?;
    object
        .iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_owned()))
                .ok_or_else(|| {
                    failure(
                        H3ComfyCheckpointErrorCode::InvalidMetadata,
                        format!("H3 Comfy metadata {key:?} must be a string"),
                    )
                })
        })
        .collect()
}

fn dtype_size(dtype: &str) -> InspectionResult<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Ok(1),
        "I16" | "U16" | "F16" | "BF16" => Ok(2),
        "I32" | "U32" | "F32" => Ok(4),
        "I64" | "U64" | "F64" => Ok(8),
        other => Err(failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            format!("unsupported H3 Comfy safetensors dtype {other:?}"),
        )),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigEnvelope {
    transformer: PrunedTransformerConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrunedTransformerConfig {
    image_model: String,
    hidden_size: usize,
    num_layers: usize,
    token_refiner_num_layers: usize,
    num_attention_heads: usize,
    attention_head_dim: usize,
    ffn_hidden_size: usize,
    latents_dim: usize,
    audio_latents_dim: usize,
    patch_size: [usize; 3],
    text_dim: usize,
    adaln_curve_grid: usize,
    time_embed_dim: usize,
    rope_inv_freq_len: usize,
    norm_eps: f64,
    qk_norm_eps: f64,
    final_norm_eps: f64,
    sigma_shift_video: f64,
    sigma_shift_audio: f64,
}

impl PrunedTransformerConfig {
    fn validate(&self, expected: &H3TransformerConfig, mode: H3AdaLnMode) -> InspectionResult<()> {
        let H3AdaLnMode::Curve { grid, basis_dim } = mode else {
            return Err(failure(
                H3ComfyCheckpointErrorCode::ConfigMismatch,
                "H3 Comfy pruned config requires curve AdaLN mode",
            ));
        };
        let matches = self.image_model == "minimax_h3"
            && self.hidden_size == expected.hidden_size
            && self.num_layers == expected.num_layers
            && self.token_refiner_num_layers == expected.token_refiner_num_layers
            && self.num_attention_heads == expected.num_attention_heads
            && self.attention_head_dim == expected.attention_head_dim
            && self.ffn_hidden_size == expected.ffn_hidden_size
            && self.latents_dim == expected.video_latent_channels
            && self.audio_latents_dim == expected.audio_latent_channels
            && self.patch_size == expected.patch_size
            && self.text_dim == expected.text_dim
            && self.adaln_curve_grid == grid
            && self.time_embed_dim == basis_dim
            && self.rope_inv_freq_len == expected.rope_inv_freq_len
            && self.norm_eps == expected.norm_eps
            && self.qk_norm_eps == expected.qk_norm_eps
            && self.final_norm_eps == expected.final_norm_eps
            && self.sigma_shift_video == 12.0
            && self.sigma_shift_audio == 3.0;
        if !matches {
            return Err(failure(
                H3ComfyCheckpointErrorCode::ConfigMismatch,
                "H3 Comfy transformer metadata does not match the frozen architecture/curve",
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantizationMetadata {
    format_version: String,
    layers: BTreeMap<String, QuantizedLayerMetadata>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct QuantizedLayerMetadata {
    format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    full_precision_matrix_mult: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    convrot: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    convrot_groupsize: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpectedTensor {
    dtype: &'static str,
    shape: Vec<usize>,
}

fn inspect_parsed_header(
    parsed: ParsedHeader,
    config: H3TransformerConfig,
    task: H3TransformerTask,
    expected_format: H3ComfyPrunedFormat,
    published_artifact: Option<H3ComfyPublishedArtifact>,
) -> InspectionResult<H3ComfyCheckpointCandidate> {
    config.validate().map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::ConfigMismatch,
            error.to_string(),
        )
    })?;
    let table = parsed.tensors.get("adaln_t_table").ok_or_else(|| {
        failure(
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
            "H3 Comfy pruned checkpoint is missing adaln_t_table",
        )
    })?;
    if table.dtype != "F32" || table.shape.len() != 2 || table.shape[0] < 2 || table.shape[1] == 0 {
        return Err(failure(
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
            "H3 Comfy adaln_t_table must be F32 [grid >= 2,basis > 0]",
        ));
    }
    if table.shape[1] >= config.time_embed_dim {
        return Err(failure(
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
            "H3 Comfy curve basis must be narrower than the full timestep embedding",
        ));
    }
    if parsed
        .tensors
        .keys()
        .any(|name| name.starts_with("time_embedder."))
    {
        return Err(failure(
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
            "H3 Comfy pruned checkpoint cannot also contain the full timestep embedder",
        ));
    }
    let mode = H3AdaLnMode::Curve {
        grid: table.shape[0],
        basis_dim: table.shape[1],
    };
    let config_json = parsed.metadata.get("config").ok_or_else(|| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            "H3 Comfy checkpoint is missing __metadata__.config",
        )
    })?;
    let config_value = parse_strict_json(config_json.as_bytes(), "H3 Comfy config metadata")?;
    let config_envelope: ConfigEnvelope = from_value(config_value, "H3 Comfy config metadata")?;
    config_envelope.transformer.validate(&config, mode)?;

    let quantization = match parsed.metadata.get("_quantization_metadata") {
        Some(raw) => {
            let value = parse_strict_json(raw.as_bytes(), "H3 Comfy quantization metadata")?;
            Some(from_value::<QuantizationMetadata>(
                value,
                "H3 Comfy quantization metadata",
            )?)
        }
        None => None,
    };
    let detected_format = detect_format(quantization.as_ref())?;
    if detected_format != expected_format {
        return Err(failure(
            H3ComfyCheckpointErrorCode::FormatMismatch,
            format!(
                "H3 Comfy header format {detected_format:?} does not match source authority {expected_format:?}"
            ),
        ));
    }
    let (expected, policy) =
        expected_schema(&config, mode, detected_format, quantization.as_ref())?;
    validate_tensor_schema(&parsed.tensors, &expected)?;

    let tensor_bytes = parsed.file_len - parsed.header_len - 8;
    let dense = detected_format == H3ComfyPrunedFormat::Bf16;
    let memory = H3ComfyMemoryAccounting {
        artifact_file_bytes: parsed.file_len,
        header_bytes: parsed.header_len + 8,
        encoded_tensor_bytes: tensor_bytes,
        host_mapped_file_bytes: dense.then_some(parsed.file_len),
        host_staging_bytes: dense.then_some(0),
        resident_device_weight_bytes: dense.then_some(tensor_bytes),
        dequantization_workspace_bytes: dense.then_some(0),
        streamed_weight_bytes: dense.then_some(0),
    };
    let accuracy_tier = match detected_format {
        H3ComfyPrunedFormat::Bf16 => H3ComfyAccuracyTier::SourceClaimedEquivalentPrunedBf16,
        H3ComfyPrunedFormat::Fp8Scaled => H3ComfyAccuracyTier::UnqualifiedPrunedFp8,
        H3ComfyPrunedFormat::Int8ConvRot => H3ComfyAccuracyTier::UnqualifiedPrunedInt8ConvRot,
    };
    let artifact = published_artifact.unwrap_or(match (task, detected_format) {
        (H3TransformerTask::T2VaFl2Va, H3ComfyPrunedFormat::Bf16) => {
            H3ComfyPublishedArtifact::Fl2VaPrunedBf16
        }
        (H3TransformerTask::T2VaFl2Va, H3ComfyPrunedFormat::Fp8Scaled) => {
            H3ComfyPublishedArtifact::Fl2VaPrunedFp8Scaled
        }
        (H3TransformerTask::T2VaFl2Va, H3ComfyPrunedFormat::Int8ConvRot) => {
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot
        }
        (H3TransformerTask::Ref2Va, H3ComfyPrunedFormat::Bf16) => {
            H3ComfyPublishedArtifact::Ref2VaPrunedBf16
        }
        (H3TransformerTask::Ref2Va, H3ComfyPrunedFormat::Fp8Scaled) => {
            H3ComfyPublishedArtifact::Ref2VaPrunedFp8Scaled
        }
        (H3TransformerTask::Ref2Va, H3ComfyPrunedFormat::Int8ConvRot) => {
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot
        }
    });
    Ok(H3ComfyCheckpointCandidate {
        artifact,
        source_repository_revision: H3_COMFY_ORG_SOURCE_REVISION.to_owned(),
        implementation_revision: H3_COMFYUI_SOURCE_REVISION.to_owned(),
        expected_content_sha256: artifact.content_sha256().to_owned(),
        header_identity_sha256: parsed.header_identity_sha256,
        tensor_count: parsed.tensors.len(),
        strategy: H3ComfyFrozenStrategyMetadata {
            stable_id: detected_format.stable_id().to_owned(),
            task,
            format: detected_format,
            adaln_mode: mode,
            interpolation: H3ComfyCurveInterpolation::ClampUnitThenAdjacentLinear,
            qkv_layout: H3QkvLayout::QkvMajor,
            dense_compute_precision: H3PrecisionProfile::OfficialMixedBf16F32,
            accuracy_tier,
            quantization_policy: policy,
            memory,
        },
    })
}

fn detect_format(
    quantization: Option<&QuantizationMetadata>,
) -> InspectionResult<H3ComfyPrunedFormat> {
    let Some(quantization) = quantization else {
        return Ok(H3ComfyPrunedFormat::Bf16);
    };
    if quantization.format_version != "1.0" || quantization.layers.is_empty() {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            "H3 Comfy quantization metadata must use non-empty format_version 1.0",
        ));
    }
    let formats = quantization
        .layers
        .values()
        .map(|layer| layer.format.as_str())
        .collect::<BTreeSet<_>>();
    match formats.iter().copied().collect::<Vec<_>>().as_slice() {
        ["float8_e4m3fn"] => Ok(H3ComfyPrunedFormat::Fp8Scaled),
        ["int8_tensorwise"] => Ok(H3ComfyPrunedFormat::Int8ConvRot),
        _ => Err(failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            format!("unsupported or mixed H3 Comfy quantization formats {formats:?}"),
        )),
    }
}

fn expected_schema(
    config: &H3TransformerConfig,
    mode: H3AdaLnMode,
    format: H3ComfyPrunedFormat,
    quantization: Option<&QuantizationMetadata>,
) -> InspectionResult<(BTreeMap<String, ExpectedTensor>, H3ComfyQuantizationPolicy)> {
    let base = expected_h3_weight_specs(config, mode, H3PrecisionProfile::OfficialMixedBf16F32)
        .map_err(|error| {
            failure(
                H3ComfyCheckpointErrorCode::ConfigMismatch,
                error.to_string(),
            )
        })?;
    let quantizable = base
        .iter()
        .filter(|(name, spec)| {
            spec.dtype == DType::BF16 && spec.shape.len() == 2 && name.ends_with(".weight")
        })
        .map(|(name, _)| name.trim_end_matches(".weight").to_owned())
        .collect::<BTreeSet<_>>();
    let quantized_layers = if format == H3ComfyPrunedFormat::Bf16 {
        if quantization.is_some() {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidMetadata,
                "H3 Comfy pruned BF16 checkpoint cannot carry quantization metadata",
            ));
        }
        BTreeSet::new()
    } else {
        let metadata = quantization.ok_or_else(|| {
            failure(
                H3ComfyCheckpointErrorCode::InvalidMetadata,
                "H3 Comfy quantized checkpoint is missing quantization metadata",
            )
        })?;
        let actual = metadata.layers.keys().cloned().collect::<BTreeSet<_>>();
        if actual != quantizable {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidMetadata,
                set_mismatch("quantized layer", &quantizable, &actual),
            ));
        }
        for (name, layer) in &metadata.layers {
            match format {
                H3ComfyPrunedFormat::Fp8Scaled => {
                    if layer.format != "float8_e4m3fn"
                        || layer.full_precision_matrix_mult != Some(false)
                        || layer.convrot.is_some()
                        || layer.convrot_groupsize.is_some()
                    {
                        return Err(failure(
                            H3ComfyCheckpointErrorCode::InvalidMetadata,
                            format!("invalid H3 FP8 metadata for layer {name:?}"),
                        ));
                    }
                }
                H3ComfyPrunedFormat::Int8ConvRot => {
                    if layer.format != "int8_tensorwise"
                        || layer.full_precision_matrix_mult.is_some()
                        || layer.convrot != Some(true)
                        || layer.convrot_groupsize != Some(CONVROT_GROUP_SIZE)
                    {
                        return Err(failure(
                            H3ComfyCheckpointErrorCode::InvalidMetadata,
                            format!(
                                "H3 INT8 layer {name:?} must declare tensorwise ConvRot group {CONVROT_GROUP_SIZE}"
                            ),
                        ));
                    }
                }
                H3ComfyPrunedFormat::Bf16 => unreachable!(),
            }
        }
        actual
    };

    let mut expected = BTreeMap::new();
    let mut protected = Vec::new();
    for (name, spec) in base {
        let layer = name.strip_suffix(".weight");
        let quantized = layer.is_some_and(|layer| quantized_layers.contains(layer));
        let dtype = if quantized {
            match format {
                H3ComfyPrunedFormat::Fp8Scaled => "F8_E4M3",
                H3ComfyPrunedFormat::Int8ConvRot => "I8",
                H3ComfyPrunedFormat::Bf16 => unreachable!(),
            }
        } else {
            if name.ends_with(".weight") {
                protected.push(name.clone());
            }
            candle_dtype_name(spec.dtype)?
        };
        expected.insert(
            name.clone(),
            ExpectedTensor {
                dtype,
                shape: spec.shape,
            },
        );
        if quantized {
            expected.insert(
                format!("{layer}.weight_scale", layer = layer.unwrap()),
                ExpectedTensor {
                    dtype: "F32",
                    shape: vec![],
                },
            );
            if format == H3ComfyPrunedFormat::Fp8Scaled {
                expected.insert(
                    format!("{layer}.input_scale", layer = layer.unwrap()),
                    ExpectedTensor {
                        dtype: "F32",
                        shape: vec![],
                    },
                );
            }
        }
    }
    let quantized_layers = quantized_layers.into_iter().collect::<Vec<_>>();
    let mut digest = Sha256::new();
    digest.update(format.stable_id().as_bytes());
    for layer in &quantized_layers {
        digest.update((layer.len() as u64).to_le_bytes());
        digest.update(layer.as_bytes());
        if let Some(metadata) = quantization.and_then(|metadata| metadata.layers.get(layer)) {
            let encoded = serde_json::to_vec(metadata).map_err(|error| {
                failure(
                    H3ComfyCheckpointErrorCode::InvalidMetadata,
                    error.to_string(),
                )
            })?;
            digest.update((encoded.len() as u64).to_le_bytes());
            digest.update(encoded);
        }
    }
    Ok((
        expected,
        H3ComfyQuantizationPolicy {
            quantized_layers,
            protected_weight_tensors: protected,
            policy_sha256: hex_digest(digest.finalize()),
        },
    ))
}

fn validate_tensor_schema(
    actual: &BTreeMap<String, HeaderTensor>,
    expected: &BTreeMap<String, ExpectedTensor>,
) -> InspectionResult<()> {
    let actual_keys = actual.keys().cloned().collect::<BTreeSet<_>>();
    let expected_keys = expected.keys().cloned().collect::<BTreeSet<_>>();
    if actual_keys != expected_keys {
        return Err(failure(
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
            set_mismatch("tensor", &expected_keys, &actual_keys),
        ));
    }
    for (name, expected) in expected {
        let actual = &actual[name];
        if actual.dtype != expected.dtype || actual.shape != expected.shape {
            return Err(failure(
                H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                format!(
                    "H3 Comfy tensor {name:?} expected {} {:?}, got {} {:?}",
                    expected.dtype, expected.shape, actual.dtype, actual.shape
                ),
            ));
        }
    }
    Ok(())
}

fn set_mismatch(label: &str, expected: &BTreeSet<String>, actual: &BTreeSet<String>) -> String {
    let missing = expected
        .difference(actual)
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = actual
        .difference(expected)
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    format!("H3 Comfy {label} set mismatch: missing={missing:?} unexpected={unexpected:?}")
}

fn candle_dtype_name(dtype: DType) -> InspectionResult<&'static str> {
    match dtype {
        DType::BF16 => Ok("BF16"),
        DType::F32 => Ok("F32"),
        other => Err(failure(
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
            format!("unsupported H3 Comfy base dtype {other:?}"),
        )),
    }
}

fn from_value<T: DeserializeOwned>(value: Value, context: &str) -> InspectionResult<T> {
    serde_json::from_value(value).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            format!("invalid {context}: {error}"),
        )
    })
}

fn parse_strict_json(bytes: &[u8], context: &str) -> InspectionResult<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictJsonValue::deserialize(&mut deserializer)
        .map_err(|error| {
            failure(
                H3ComfyCheckpointErrorCode::InvalidHeader,
                format!("invalid {context}: {error}"),
            )
        })?
        .0;
    deserializer.end().map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            format!("invalid trailing data in {context}: {error}"),
        )
    })?;
    Ok(value)
}

struct StrictJsonValue(Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Number::from_f64(value)
            .map(|number| StrictJsonValue(Value::Number(number)))
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictJsonValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJsonValue>()? {
            values.push(value.0);
        }
        Ok(StrictJsonValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some((key, value)) = object.next_entry::<String, StrictJsonValue>()? {
            if values.insert(key.clone(), value.0).is_some() {
                return Err(A::Error::custom(format!("duplicate JSON key {key:?}")));
            }
        }
        Ok(StrictJsonValue(Value::Object(values)))
    }
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use candle::{Device, Tensor};

    use super::*;
    use crate::minimax_h3::interpolate_h3_adaln_curve;

    fn tiny_config() -> H3TransformerConfig {
        H3TransformerConfig {
            hidden_size: 8,
            num_layers: 2,
            token_refiner_num_layers: 1,
            num_attention_heads: 2,
            attention_head_dim: 8,
            ffn_hidden_size: 12,
            video_latent_channels: 2,
            audio_latent_channels: 3,
            patch_size: [1, 2, 2],
            text_dim: 6,
            timestep_input_dim: 4,
            time_embed_hidden_size: 8,
            time_embed_dim: 4,
            rope_inv_freq_len: 1,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    fn config_metadata(config: &H3TransformerConfig, grid: usize, basis: usize) -> String {
        serde_json::json!({
            "transformer": {
                "image_model": "minimax_h3",
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "token_refiner_num_layers": config.token_refiner_num_layers,
                "num_attention_heads": config.num_attention_heads,
                "attention_head_dim": config.attention_head_dim,
                "ffn_hidden_size": config.ffn_hidden_size,
                "latents_dim": config.video_latent_channels,
                "audio_latents_dim": config.audio_latent_channels,
                "patch_size": config.patch_size,
                "text_dim": config.text_dim,
                "adaln_curve_grid": grid,
                "time_embed_dim": basis,
                "rope_inv_freq_len": config.rope_inv_freq_len,
                "norm_eps": config.norm_eps,
                "qk_norm_eps": config.qk_norm_eps,
                "final_norm_eps": config.final_norm_eps,
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0
            }
        })
        .to_string()
    }

    fn quant_metadata(
        config: &H3TransformerConfig,
        mode: H3AdaLnMode,
        format: H3ComfyPrunedFormat,
    ) -> String {
        let base = expected_h3_weight_specs(config, mode, H3PrecisionProfile::OfficialMixedBf16F32)
            .unwrap();
        let layers = base
            .iter()
            .filter(|(name, spec)| {
                spec.dtype == DType::BF16 && spec.shape.len() == 2 && name.ends_with(".weight")
            })
            .map(|(name, _)| {
                let metadata = match format {
                    H3ComfyPrunedFormat::Fp8Scaled => serde_json::json!({
                        "format": "float8_e4m3fn",
                        "full_precision_matrix_mult": false
                    }),
                    H3ComfyPrunedFormat::Int8ConvRot => serde_json::json!({
                        "format": "int8_tensorwise",
                        "convrot": true,
                        "convrot_groupsize": CONVROT_GROUP_SIZE
                    }),
                    H3ComfyPrunedFormat::Bf16 => unreachable!(),
                };
                (name.trim_end_matches(".weight").to_owned(), metadata)
            })
            .collect::<Map<_, _>>();
        serde_json::json!({"format_version": "1.0", "layers": layers}).to_string()
    }

    fn fixture_header(
        config: &H3TransformerConfig,
        mode: H3AdaLnMode,
        format: H3ComfyPrunedFormat,
    ) -> (Value, Vec<u8>) {
        let quantization =
            (format != H3ComfyPrunedFormat::Bf16).then(|| quant_metadata(config, mode, format));
        let quantization_parsed = quantization
            .as_ref()
            .map(|raw| serde_json::from_str::<QuantizationMetadata>(raw).unwrap());
        let (expected, _) =
            expected_schema(config, mode, format, quantization_parsed.as_ref()).unwrap();
        let mut offset = 0u64;
        let mut header = Map::new();
        let mut data = Vec::new();
        for (name, tensor) in expected {
            let elements = tensor.shape.iter().product::<usize>();
            let bytes = elements * dtype_size(tensor.dtype).unwrap() as usize;
            let end = offset + bytes as u64;
            header.insert(
                name,
                serde_json::json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [offset, end]
                }),
            );
            data.resize(data.len() + bytes, 0);
            offset = end;
        }
        let H3AdaLnMode::Curve { grid, basis_dim } = mode else {
            unreachable!()
        };
        let mut metadata = Map::new();
        metadata.insert(
            "config".into(),
            Value::String(config_metadata(config, grid, basis_dim)),
        );
        if let Some(quantization) = quantization {
            metadata.insert("_quantization_metadata".into(), Value::String(quantization));
        }
        header.insert("__metadata__".into(), Value::Object(metadata));
        (Value::Object(header), data)
    }

    fn write_fixture(header: &Value, data: &[u8], tag: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mold-h3-comfy-{tag}-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let encoded = serde_json::to_vec(header).unwrap();
        let mut file = File::create(&path).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.write_all(data).unwrap();
        path
    }

    fn inspect_fixture(
        header: &Value,
        data: &[u8],
        format: H3ComfyPrunedFormat,
    ) -> InspectionResult<H3ComfyCheckpointCandidate> {
        let path = write_fixture(header, data, "candidate");
        let parsed = read_safetensors_header(&path)?;
        let result = inspect_parsed_header(
            parsed,
            tiny_config(),
            H3TransformerTask::T2VaFl2Va,
            format,
            None,
        );
        let _ = std::fs::remove_file(path);
        result
    }

    #[test]
    fn public_artifact_authorities_pin_task_format_size_and_digest() {
        let artifacts = [
            H3ComfyPublishedArtifact::Fl2VaPrunedBf16,
            H3ComfyPublishedArtifact::Fl2VaPrunedFp8Scaled,
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
            H3ComfyPublishedArtifact::Ref2VaPrunedBf16,
            H3ComfyPublishedArtifact::Ref2VaPrunedFp8Scaled,
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot,
        ];
        assert_eq!(
            artifacts.iter().map(|item| item.file_bytes()).sum::<u64>(),
            164_308_618_800
        );
        assert!(artifacts
            .iter()
            .all(|item| item.content_sha256().len() == 64));
        assert_eq!(
            artifacts
                .iter()
                .filter(|item| item.task() == H3TransformerTask::Ref2Va)
                .count(),
            3
        );
    }

    #[test]
    fn provisional_released_schema_counts_are_frozen_without_artifact_headers() {
        let config = H3TransformerConfig::default();
        let mode = H3AdaLnMode::Curve {
            grid: 129,
            basis_dim: 64,
        };
        let (bf16, bf16_policy) =
            expected_schema(&config, mode, H3ComfyPrunedFormat::Bf16, None).unwrap();
        assert_eq!(bf16.len(), 532);
        assert!(bf16_policy.quantized_layers.is_empty());

        let fp8_raw = quant_metadata(&config, mode, H3ComfyPrunedFormat::Fp8Scaled);
        let fp8_metadata: QuantizationMetadata = serde_json::from_str(&fp8_raw).unwrap();
        let (fp8, fp8_policy) = expected_schema(
            &config,
            mode,
            H3ComfyPrunedFormat::Fp8Scaled,
            Some(&fp8_metadata),
        )
        .unwrap();
        assert_eq!(fp8_policy.quantized_layers.len(), 209);
        assert_eq!(fp8.len(), 950);

        let int8_raw = quant_metadata(&config, mode, H3ComfyPrunedFormat::Int8ConvRot);
        let int8_metadata: QuantizationMetadata = serde_json::from_str(&int8_raw).unwrap();
        let (int8, int8_policy) = expected_schema(
            &config,
            mode,
            H3ComfyPrunedFormat::Int8ConvRot,
            Some(&int8_metadata),
        )
        .unwrap();
        assert_eq!(int8_policy.quantized_layers.len(), 209);
        assert_eq!(int8.len(), 741);
    }

    #[test]
    fn synthetic_pruned_bf16_fp8_and_int8_headers_freeze_distinct_strategies() {
        let config = tiny_config();
        let mode = H3AdaLnMode::Curve {
            grid: 5,
            basis_dim: 3,
        };
        for format in [
            H3ComfyPrunedFormat::Bf16,
            H3ComfyPrunedFormat::Fp8Scaled,
            H3ComfyPrunedFormat::Int8ConvRot,
        ] {
            let (header, data) = fixture_header(&config, mode, format);
            let candidate = inspect_fixture(&header, &data, format).unwrap();
            assert_eq!(candidate.strategy.format, format);
            assert_eq!(candidate.strategy.adaln_mode, mode);
            assert_eq!(candidate.strategy.qkv_layout, H3QkvLayout::QkvMajor);
            assert_eq!(
                candidate.strategy.interpolation,
                H3ComfyCurveInterpolation::ClampUnitThenAdjacentLinear
            );
            if format == H3ComfyPrunedFormat::Bf16 {
                assert!(candidate
                    .strategy
                    .quantization_policy
                    .quantized_layers
                    .is_empty());
                assert!(candidate
                    .strategy
                    .memory
                    .resident_device_weight_bytes
                    .is_some());
            } else {
                assert!(!candidate
                    .strategy
                    .quantization_policy
                    .quantized_layers
                    .is_empty());
                assert_eq!(candidate.strategy.memory.resident_device_weight_bytes, None);
            }
        }
    }

    #[test]
    fn quantized_runtime_rejections_name_kernel_and_backend_requirements() {
        let config = tiny_config();
        let mode = H3AdaLnMode::Curve {
            grid: 5,
            basis_dim: 3,
        };
        let (header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Int8ConvRot);
        let candidate = inspect_fixture(&header, &data, H3ComfyPrunedFormat::Int8ConvRot).unwrap();
        let cuda = candidate
            .require_supported_runtime(H3ComfyRuntimeBackend::Cuda)
            .unwrap_err();
        assert!(cuda
            .requirements
            .contains(&H3ComfyRuntimeRequirement::H3Int8ConvRotKernelOrQualifiedDequantization));
        assert!(!cuda
            .requirements
            .contains(&H3ComfyRuntimeRequirement::QuantizedBackendQualification));
        let metal = candidate
            .require_supported_runtime(H3ComfyRuntimeBackend::Metal)
            .unwrap_err();
        assert!(metal
            .requirements
            .contains(&H3ComfyRuntimeRequirement::QuantizedBackendQualification));
    }

    #[test]
    fn mutated_schema_config_quant_metadata_and_task_fail_closed() {
        let config = tiny_config();
        let mode = H3AdaLnMode::Curve {
            grid: 5,
            basis_dim: 3,
        };
        let (mut header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Bf16);
        header
            .as_object_mut()
            .unwrap()
            .get_mut("blocks.0.norm1.weight")
            .unwrap()["dtype"] = Value::String("F16".into());
        assert_eq!(
            inspect_fixture(&header, &data, H3ComfyPrunedFormat::Bf16)
                .unwrap_err()
                .code,
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch
        );

        let (mut header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Bf16);
        let metadata = header["__metadata__"].as_object_mut().unwrap();
        let mut config_value: Value =
            serde_json::from_str(metadata["config"].as_str().unwrap()).unwrap();
        config_value["transformer"]["hidden_size"] = Value::from(9);
        metadata.insert("config".into(), Value::String(config_value.to_string()));
        assert_eq!(
            inspect_fixture(&header, &data, H3ComfyPrunedFormat::Bf16)
                .unwrap_err()
                .code,
            H3ComfyCheckpointErrorCode::ConfigMismatch
        );

        let (mut header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Int8ConvRot);
        let metadata = header["__metadata__"].as_object_mut().unwrap();
        let mut quant: Value =
            serde_json::from_str(metadata["_quantization_metadata"].as_str().unwrap()).unwrap();
        let first = quant["layers"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        first["convrot_groupsize"] = Value::from(128);
        metadata.insert(
            "_quantization_metadata".into(),
            Value::String(quant.to_string()),
        );
        assert_eq!(
            inspect_fixture(&header, &data, H3ComfyPrunedFormat::Int8ConvRot)
                .unwrap_err()
                .code,
            H3ComfyCheckpointErrorCode::InvalidMetadata
        );

        let path = write_fixture(&header, &data, "task-authority");
        let error = inspect_h3_comfy_published_header(
            &path,
            H3TransformerTask::Ref2Va,
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
        )
        .unwrap_err();
        assert_eq!(
            error.code,
            H3ComfyCheckpointErrorCode::TaskAuthorityMismatch
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn source_size_format_and_quantized_sidecar_shape_fail_closed() {
        let config = tiny_config();
        let mode = H3AdaLnMode::Curve {
            grid: 5,
            basis_dim: 3,
        };
        let (header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Bf16);
        assert_eq!(
            inspect_fixture(&header, &data, H3ComfyPrunedFormat::Fp8Scaled)
                .unwrap_err()
                .code,
            H3ComfyCheckpointErrorCode::FormatMismatch
        );
        let path = write_fixture(&header, &data, "source-size");
        assert_eq!(
            inspect_h3_comfy_published_header(
                &path,
                H3TransformerTask::T2VaFl2Va,
                H3ComfyPublishedArtifact::Fl2VaPrunedBf16,
            )
            .unwrap_err()
            .code,
            H3ComfyCheckpointErrorCode::SourceSizeMismatch
        );
        let _ = std::fs::remove_file(path);

        let (mut header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Fp8Scaled);
        let scale = header
            .as_object()
            .unwrap()
            .keys()
            .find(|name| name.ends_with(".weight_scale"))
            .unwrap()
            .clone();
        header[&scale]["shape"] = serde_json::json!([1]);
        assert_eq!(
            inspect_fixture(&header, &data, H3ComfyPrunedFormat::Fp8Scaled)
                .unwrap_err()
                .code,
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch
        );
    }

    #[test]
    fn missing_scale_extra_tensor_and_malformed_offsets_fail_closed() {
        let config = tiny_config();
        let mode = H3AdaLnMode::Curve {
            grid: 5,
            basis_dim: 3,
        };
        let (mut header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Fp8Scaled);
        let scale = header
            .as_object()
            .unwrap()
            .keys()
            .find(|name| name.ends_with(".weight_scale"))
            .unwrap()
            .clone();
        header.as_object_mut().unwrap().remove(&scale);
        assert_eq!(
            inspect_fixture(&header, &data, H3ComfyPrunedFormat::Fp8Scaled)
                .unwrap_err()
                .code,
            H3ComfyCheckpointErrorCode::InvalidHeader
        );

        let (mut header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Bf16);
        header.as_object_mut().unwrap().insert(
            "unexpected.weight".into(),
            serde_json::json!({"dtype":"F32","shape":[],"data_offsets":[data.len(), data.len()+4]}),
        );
        let mut extra_data = data.clone();
        extra_data.extend_from_slice(&[0; 4]);
        assert_eq!(
            inspect_fixture(&header, &extra_data, H3ComfyPrunedFormat::Bf16)
                .unwrap_err()
                .code,
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch
        );

        let (mut header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Bf16);
        let first = header
            .as_object_mut()
            .unwrap()
            .iter_mut()
            .find(|(name, _)| name.as_str() != "__metadata__")
            .unwrap()
            .1;
        first["data_offsets"][0] = Value::from(1);
        let path = write_fixture(&header, &data, "bad-offset");
        assert_eq!(
            read_safetensors_header(&path).unwrap_err().code,
            H3ComfyCheckpointErrorCode::InvalidHeader
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn duplicate_json_keys_are_rejected_recursively() {
        assert!(parse_strict_json(br#"{"a":1,"a":2}"#, "duplicate root").is_err());
        assert!(parse_strict_json(br#"{"a":{"b":1,"b":2}}"#, "duplicate nested").is_err());
    }

    #[test]
    fn curve_interpolation_matches_comfy_clamp_and_adjacent_lerp() {
        let table = Tensor::new(&[[0f32, 10.0], [2.0, 20.0], [4.0, 40.0]], &Device::Cpu).unwrap();
        let timesteps =
            Tensor::new(&[-1f32, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0], &Device::Cpu).unwrap();
        let output = interpolate_h3_adaln_curve(&table, &timesteps)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(
            output,
            vec![
                vec![0., 10.],
                vec![0., 10.],
                vec![1., 15.],
                vec![2., 20.],
                vec![3., 30.],
                vec![4., 40.],
                vec![4., 40.],
            ]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn curve_interpolation_cpu_cuda_parity() -> candle::Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let table_values = (0..35)
            .map(|index| index as f32 / 17.0 - 0.8)
            .collect::<Vec<_>>();
        let timestep_values = vec![-0.2f32, 0.0, 0.13, 0.5, 0.91, 1.0, 1.2];
        let cpu = interpolate_h3_adaln_curve(
            &Tensor::from_vec(table_values.clone(), (7, 5), &Device::Cpu)?,
            &Tensor::from_vec(timestep_values.clone(), 7, &Device::Cpu)?,
        )?;
        let gpu = interpolate_h3_adaln_curve(
            &Tensor::from_vec(table_values, (7, 5), &cuda)?,
            &Tensor::from_vec(timestep_values, 7, &cuda)?,
        )?
        .to_device(&Device::Cpu)?;
        let cpu = cpu.flatten_all()?.to_vec1::<f32>()?;
        let gpu = gpu.flatten_all()?.to_vec1::<f32>()?;
        let max_error = cpu
            .iter()
            .zip(gpu)
            .map(|(left, right)| (left - right).abs())
            .fold(0f32, f32::max);
        assert!(max_error <= 1e-6, "max interpolation error {max_error}");
        Ok(())
    }
}
