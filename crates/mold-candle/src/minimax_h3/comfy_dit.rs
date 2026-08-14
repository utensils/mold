//! Source-pinned inspection and opened-file streaming primitives for ComfyUI's
//! pruned MiniMax H3 DiT checkpoints.
//!
//! The schema is derived from ComfyUI `a464ac33588ae182f81a090d910cfbf21e255b73`:
//! `comfy/ldm/minimax/model.py`, `comfy/model_detection.py`,
//! `comfy/quant_ops.py`, `comfy/ops.py`, and `comfy/utils.py`, plus its exact
//! comfy-kitchen 0.2.26 dependency for ConvRot scale semantics. Public artifact
//! sizes and content digests are pinned to the Comfy-Org repository revision
//! below. Header inspection remains payload-free; the additive INT8 primitive
//! fully authenticates and retains one opened descriptor before loading
//! protected resident tensors or one CPU-packed main block at a time.
//!
//! Neither path registers an inference engine, capability, catalog entry, or
//! download. Header candidates therefore remain non-executable, and the
//! opened-file primitive remains a private implementation prerequisite until
//! the separate runtime/release qualification and activation work lands.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;
use std::fmt;
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use serde::de::{DeserializeOwned, Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use super::attention::{
    H3AttentionDType, H3AttentionDevice, H3AttentionModelContract, H3AttentionRuntimeAuthority,
};
use super::comfy_quant::{H3ComfyInt8ConvRotLinear, H3_COMFY_PORTABLE_ROW_CHUNK};
use super::dit::{
    expected_h3_weight_specs, H3AdaLnMode, H3ComfyInt8BlockMatrices, H3LoadedTransformerBlock,
    H3PrecisionProfile, H3QkvLayout, H3StreamedTransformer, H3StreamedTransformerIdentity,
    H3TransformerConfig, H3TransformerTask,
};

pub const H3_COMFYUI_SOURCE_REVISION: &str = "a464ac33588ae182f81a090d910cfbf21e255b73";
pub const H3_COMFY_ORG_SOURCE_REVISION: &str = "eb8a16107c595128b3a578f82d2ce2f75920c355";
pub const H3_COMFY_KITCHEN_REPOSITORY: &str = "Comfy-Org/comfy-kitchen";
pub const H3_COMFY_KITCHEN_VERSION: &str = "0.2.26";
pub const H3_COMFY_KITCHEN_SOURCE_REVISION: &str = "255a43879fe57bbcbecfdb273b46d772b00c5a90";
/// JSON header length shared by the published FL2VA and Ref2VA INT8 ConvRot
/// artifacts at [`H3_COMFY_ORG_SOURCE_REVISION`]. The safetensors length prefix
/// is not included.
pub const H3_COMFY_PUBLISHED_INT8_HEADER_LEN: u64 = 95_416;
/// SHA-256 of the eight-byte little-endian safetensors length prefix followed
/// by the exact published JSON header bytes.
pub const H3_COMFY_PUBLISHED_INT8_HEADER_SHA256: &str =
    "8232b3c428e54591c286b5eee3eaba840f03d2cd2cb359ec09dd2558974ddc74";
/// Tensor entries in the exact released header; there is no `__metadata__`
/// entry to subtract.
pub const H3_COMFY_PUBLISHED_INT8_TENSOR_COUNT: usize = 932;

const MAX_HEADER_BYTES: u64 = 8 * 1024 * 1024;
const MAX_TENSORS: usize = 4_096;
const MAX_TENSOR_KEY_BYTES: usize = 4_096;
const MAX_TENSOR_RANK: usize = 8;
const CONVROT_GROUP_SIZE: usize = 256;
const PUBLISHED_INT8_CURVE_GRID: usize = 1_025;
const PUBLISHED_INT8_CURVE_BASIS: usize = 8;
const PUBLISHED_INT8_COMFY_QUANT_BYTES: usize = 72;
const PUBLISHED_INT8_COMFY_QUANT_COUNT: usize = 200;
const PUBLISHED_INT8_COMFY_QUANT_PAYLOAD: &[u8; PUBLISHED_INT8_COMFY_QUANT_BYTES] =
    br#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#;
const FILE_READ_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const QUANTIZED_BLOCK_WEIGHT_SUFFIXES: &[&str] = &[
    "attn.qkv_proj.weight",
    "attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
];

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
    PrunedAdaLnNumericalQualification,
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
            H3ComfyRuntimeRequirement::PrunedAdaLnNumericalQualification,
            H3ComfyRuntimeRequirement::RuntimeFactoryActivation,
        ];
        match self.strategy.format {
            H3ComfyPrunedFormat::Bf16 => {}
            H3ComfyPrunedFormat::Fp8Scaled => {
                requirements
                    .push(H3ComfyRuntimeRequirement::H3ScaledFp8KernelOrQualifiedDequantization);
                requirements.push(H3ComfyRuntimeRequirement::QuantizedBackendQualification);
            }
            H3ComfyPrunedFormat::Int8ConvRot => {
                requirements
                    .push(H3ComfyRuntimeRequirement::H3Int8ConvRotKernelOrQualifiedDequantization);
                requirements.push(H3ComfyRuntimeRequirement::QuantizedBackendQualification);
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
    Cancelled,
    InvalidHeader,
    InvalidMetadata,
    ConfigMismatch,
    TaskAuthorityMismatch,
    SourceSizeMismatch,
    ContentDigestMismatch,
    FileIdentityChanged,
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

/// Cooperative cancellation checked before every bounded file read and every
/// main-block construction boundary.
pub trait H3ComfyInt8Cancellation: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct H3ComfyNeverCancel;

impl H3ComfyInt8Cancellation for H3ComfyNeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3ComfyInt8BlockMemory {
    /// Quantized weights and F32 scale sidecars retained on CPU for one block.
    pub encoded_host_bytes: u64,
    /// Largest temporary host buffer used by one exact tensor read.
    pub max_host_read_staging_bytes: u64,
    /// Largest dense F32 row chunk staged on the execution device.
    pub max_device_weight_staging_bytes: u64,
    /// Protected norms and FP32 Curve-AdaLN tensors resident for this block.
    pub protected_device_bytes: u64,
}

/// Exact pre-allocation record for one authenticated main block.
///
/// The content digest covers the block's complete encoded safetensors payload
/// in file order, including the authenticated quantization sidecars. The
/// memory record deliberately excludes those sidecars because they are policy
/// metadata and are never retained by the block loader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3ComfyOpenedBlockMemoryEvidence {
    pub index: u16,
    pub memory: H3ComfyInt8BlockMemory,
    pub content_sha256: String,
}

/// Exact pre-allocation memory evidence derived from the same retained file
/// descriptor that later supplies the resident core and streamed blocks.
///
/// `fixed_*` covers every non-main-block tensor that the resident transformer
/// loads. F16 curve projections are charged at their logical F32 residency.
/// The fifty block records are ordered and exhaustive for released H3.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3ComfyOpenedMemoryEvidence {
    pub verified_file_bytes: u64,
    pub header_bytes: u64,
    pub fixed_encoded_host_bytes: u64,
    pub fixed_protected_device_bytes: u64,
    pub fixed_max_host_read_staging_bytes: u64,
    pub fixed_max_device_weight_staging_bytes: u64,
    pub blocks: Vec<H3ComfyOpenedBlockMemoryEvidence>,
    pub identity_sha256: String,
}

/// An exact, already-opened Comfy INT8 artifact authority.
///
/// Header parsing, schema/policy validation, full-content hashing, resident
/// loads, and every streamed block read all use the same retained file
/// descriptor. This primitive is deliberately not registered with Mold's
/// factory or catalog.
pub struct H3ComfyOpenedInt8Checkpoint {
    candidate: H3ComfyCheckpointCandidate,
    config: H3TransformerConfig,
    source: Arc<H3ComfyOpenedSource>,
    checkpoint_identity_sha256: String,
    memory_evidence: H3ComfyOpenedMemoryEvidence,
}

impl fmt::Debug for H3ComfyOpenedInt8Checkpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("H3ComfyOpenedInt8Checkpoint")
            .field("artifact", &self.candidate.artifact)
            .field("task", &self.candidate.strategy.task)
            .field(
                "checkpoint_identity_sha256",
                &self.checkpoint_identity_sha256,
            )
            .finish_non_exhaustive()
    }
}

impl H3ComfyOpenedInt8Checkpoint {
    pub fn candidate(&self) -> &H3ComfyCheckpointCandidate {
        &self.candidate
    }

    pub fn checkpoint_identity_sha256(&self) -> &str {
        &self.checkpoint_identity_sha256
    }

    pub fn content_sha256(&self) -> &str {
        &self.source.content_sha256
    }

    pub fn source_path(&self) -> &Path {
        &self.source.path
    }

    pub fn revalidate(&self) -> InspectionResult<()> {
        self.source.revalidate_authority()
    }

    pub fn config_identity_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"mold.minimax-h3.comfy-transformer-config.v1\0");
        digest.update(
            serde_json::to_vec(&self.config)
                .expect("validated H3 transformer configuration is serializable"),
        );
        hex_digest(digest.finalize())
    }

    pub fn verified_file_bytes(&self) -> u64 {
        self.source.identity.len
    }

    /// Immutable host/device/staging facts established before any Candle
    /// model tensor is allocated.
    pub fn memory_evidence(&self) -> &H3ComfyOpenedMemoryEvidence {
        &self.memory_evidence
    }

    pub fn bytes_read_after_verification(&self) -> u64 {
        self.source.runtime_bytes_read.load(Ordering::Relaxed)
    }

    /// Load resident stages and retain only the opened file plus an opaque
    /// exact-block loader for the fifty main blocks.
    pub fn load(
        self,
        device: &Device,
    ) -> candle::Result<(H3StreamedTransformer, H3ComfyInt8BlockLoader)> {
        let contract = H3AttentionModelContract {
            heads: self.config.num_attention_heads,
            head_dim: self.config.attention_head_dim,
            dtype: H3AttentionDType::from_candle(
                self.candidate
                    .strategy
                    .dense_compute_precision
                    .compute_dtype(),
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?,
        };
        let attention = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::from_candle(device),
            contract,
        )
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
        self.load_with_attention_and_cancellation(device, attention, Arc::new(H3ComfyNeverCancel))
    }

    pub fn load_with_attention_and_cancellation(
        self,
        device: &Device,
        attention: H3AttentionRuntimeAuthority,
        cancellation: Arc<dyn H3ComfyInt8Cancellation>,
    ) -> candle::Result<(H3StreamedTransformer, H3ComfyInt8BlockLoader)> {
        cancellation_boundary(cancellation.as_ref())?;
        let backend = H3ComfyOpenedBackend {
            source: self.source.clone(),
            cancellation: cancellation.clone(),
        };
        let vb = VarBuilder::from_backend(
            Box::new(backend),
            self.candidate
                .strategy
                .dense_compute_precision
                .compute_dtype(),
            device.clone(),
        );
        let mode = self.candidate.strategy.adaln_mode;
        let precision = self.candidate.strategy.dense_compute_precision;
        let task = self.candidate.strategy.task;
        let transformer = H3StreamedTransformer::load_comfy_int8_resident(
            self.config.clone(),
            task,
            mode,
            precision,
            vb.clone(),
            attention,
            self.checkpoint_identity_sha256.clone(),
        )?;
        cancellation_boundary(cancellation.as_ref())?;
        let loader = H3ComfyInt8BlockLoader {
            config: self.config,
            mode,
            precision,
            source: self.source,
            vb,
            identity: transformer.streamed_identity(),
            cancellation,
        };
        Ok((transformer, loader))
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

/// Open and fully authenticate one published Comfy INT8 ConvRot checkpoint.
///
/// Unlike [`inspect_h3_comfy_published_header`], this reads the complete file
/// and returns the retained opened-file authority needed by the private model
/// primitive. It still does not make the model reachable from any Mold runtime
/// factory, capability, catalog, or download route.
pub fn open_h3_comfy_published_int8_checkpoint(
    path: &Path,
    requested_task: H3TransformerTask,
    artifact: H3ComfyPublishedArtifact,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> InspectionResult<H3ComfyOpenedInt8Checkpoint> {
    if artifact.format() != H3ComfyPrunedFormat::Int8ConvRot {
        return Err(failure(
            H3ComfyCheckpointErrorCode::FormatMismatch,
            "H3 opened INT8 runtime requires a published INT8 ConvRot artifact",
        ));
    }
    open_h3_comfy_int8_checkpoint(
        path,
        H3TransformerConfig::default(),
        requested_task,
        Some(artifact),
        Some((artifact.file_bytes(), artifact.content_sha256())),
        cancellation,
    )
}

fn open_h3_comfy_int8_checkpoint(
    path: &Path,
    config: H3TransformerConfig,
    requested_task: H3TransformerTask,
    published_artifact: Option<H3ComfyPublishedArtifact>,
    expected_content: Option<(u64, &str)>,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> InspectionResult<H3ComfyOpenedInt8Checkpoint> {
    if let Some(artifact) = published_artifact {
        if artifact.format() != H3ComfyPrunedFormat::Int8ConvRot {
            return Err(failure(
                H3ComfyCheckpointErrorCode::FormatMismatch,
                "H3 opened INT8 runtime requires an INT8 ConvRot published authority",
            ));
        }
        if requested_task != artifact.task() {
            return Err(failure(
                H3ComfyCheckpointErrorCode::TaskAuthorityMismatch,
                format!(
                    "H3 requested task {requested_task:?} does not match source authority {:?}",
                    artifact.task()
                ),
            ));
        }
    }
    cancellation_boundary_inspection(cancellation)?;
    let symlink_metadata = std::fs::symlink_metadata(path).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::Io,
            format!("failed to inspect H3 Comfy checkpoint: {error}"),
        )
    })?;
    if symlink_metadata.file_type().is_symlink() || !symlink_metadata.is_file() {
        return Err(failure(
            H3ComfyCheckpointErrorCode::FileIdentityChanged,
            "H3 Comfy execution authority must be opened from a regular non-symlink file",
        ));
    }
    #[cfg(unix)]
    if symlink_metadata.permissions().mode() & 0o022 != 0 {
        return Err(failure(
            H3ComfyCheckpointErrorCode::FileIdentityChanged,
            "H3 Comfy execution authority must not be group/other writable",
        ));
    }
    let canonical_path = std::fs::canonicalize(path).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::Io,
            format!("failed to canonicalize H3 Comfy checkpoint: {error}"),
        )
    })?;
    let mut file = File::open(&canonical_path).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::Io,
            format!("failed to open H3 Comfy checkpoint: {error}"),
        )
    })?;
    let requested_identity = H3OpenedFileIdentity::from_metadata(&symlink_metadata);
    let opened_metadata = file
        .metadata()
        .map_err(|error| failure(H3ComfyCheckpointErrorCode::Io, error.to_string()))?;
    let identity = H3OpenedFileIdentity::from_metadata(&opened_metadata);
    if identity != requested_identity {
        return Err(failure(
            H3ComfyCheckpointErrorCode::FileIdentityChanged,
            "H3 Comfy checkpoint changed while its descriptor was opened",
        ));
    }
    let parsed = read_safetensors_header_from_file(&mut file)?;
    if let Some((expected_bytes, _)) = expected_content {
        if parsed.file_len != expected_bytes {
            return Err(failure(
                H3ComfyCheckpointErrorCode::SourceSizeMismatch,
                format!(
                    "H3 Comfy artifact size {} does not match source-pinned {expected_bytes}",
                    parsed.file_len
                ),
            ));
        }
    }
    let candidate = inspect_parsed_header(
        parsed.clone(),
        config.clone(),
        requested_task,
        H3ComfyPrunedFormat::Int8ConvRot,
        published_artifact,
    )?;
    if published_artifact.is_some() {
        validate_published_int8_quantization_sidecars(
            &mut file,
            &parsed,
            &candidate.strategy.quantization_policy,
            cancellation,
        )?;
    }
    let flight = opened_hash_flight(&canonical_path);
    let hash_flight = match flight.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let H3ComfyOpenedHashes {
        content_sha256,
        block_content_sha256,
    } = match lookup_cached_opened_hashes(&canonical_path, &identity) {
        Some(hashes) => hashes,
        None => {
            let hashes = hash_open_file(&mut file, &parsed, config.num_layers, cancellation)?;
            store_cached_opened_hashes(&canonical_path, identity.clone(), hashes.clone());
            hashes
        }
    };
    drop(hash_flight);
    if let Some((_, expected_sha256)) = expected_content {
        if content_sha256 != expected_sha256 {
            return Err(failure(
                H3ComfyCheckpointErrorCode::ContentDigestMismatch,
                format!(
                    "H3 Comfy content digest {content_sha256} does not match source-pinned {expected_sha256}"
                ),
            ));
        }
    }
    let final_identity = H3OpenedFileIdentity::from_metadata(
        &file
            .metadata()
            .map_err(|error| failure(H3ComfyCheckpointErrorCode::Io, error.to_string()))?,
    );
    if final_identity != identity {
        return Err(failure(
            H3ComfyCheckpointErrorCode::FileIdentityChanged,
            "H3 Comfy checkpoint changed during content verification",
        ));
    }
    let checkpoint_identity_sha256 = comfy_checkpoint_identity(&candidate, &content_sha256);
    let f16_curve_projection_tensors = published_artifact
        .map(|_| published_int8_curve_projection_tensors(&config))
        .unwrap_or_default();
    let memory_evidence = calculate_opened_memory_evidence(
        &parsed,
        &f16_curve_projection_tensors,
        config.num_layers,
        &block_content_sha256,
    )?;
    Ok(H3ComfyOpenedInt8Checkpoint {
        candidate,
        config,
        source: Arc::new(H3ComfyOpenedSource {
            path: canonical_path,
            file: Mutex::new(file),
            header: parsed,
            identity,
            content_sha256,
            runtime_bytes_read: AtomicU64::new(0),
            f16_curve_projection_tensors,
            f16_curve_projection_upcast_loads: AtomicU64::new(0),
        }),
        checkpoint_identity_sha256,
        memory_evidence,
    })
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct H3OpenedFileIdentity {
    len: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

impl H3OpenedFileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                len: metadata.len(),
                device: metadata.dev(),
                inode: metadata.ino(),
                modified_seconds: metadata.mtime(),
                modified_nanoseconds: metadata.mtime_nsec(),
                changed_seconds: metadata.ctime(),
                changed_nanoseconds: metadata.ctime_nsec(),
            }
        }
        #[cfg(not(unix))]
        {
            Self {
                len: metadata.len(),
                modified: metadata.modified().ok(),
            }
        }
    }
}

struct H3ComfyOpenedSource {
    path: PathBuf,
    file: Mutex<File>,
    header: ParsedHeader,
    identity: H3OpenedFileIdentity,
    content_sha256: String,
    runtime_bytes_read: AtomicU64,
    f16_curve_projection_tensors: BTreeSet<String>,
    f16_curve_projection_upcast_loads: AtomicU64,
}

impl fmt::Debug for H3ComfyOpenedSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("H3ComfyOpenedSource")
            .field("file_bytes", &self.identity.len)
            .field("content_sha256", &self.content_sha256)
            .field("tensor_count", &self.header.tensors.len())
            .finish_non_exhaustive()
    }
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
    read_safetensors_header_from_file(&mut file)
}

fn read_safetensors_header_from_file(file: &mut File) -> InspectionResult<ParsedHeader> {
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::Io,
            format!("failed to seek H3 Comfy checkpoint: {error}"),
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
    let tensor_count = object.len() - usize::from(object.contains_key("__metadata__"));
    if tensor_count > MAX_TENSORS {
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

fn cancellation_boundary(cancellation: &dyn H3ComfyInt8Cancellation) -> candle::Result<()> {
    if cancellation.is_cancelled() {
        candle::bail!("MiniMax H3 Comfy INT8 checkpoint read was cancelled")
    }
    Ok(())
}

fn cancellation_boundary_inspection(
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> InspectionResult<()> {
    if cancellation.is_cancelled() {
        return Err(failure(
            H3ComfyCheckpointErrorCode::Cancelled,
            "MiniMax H3 Comfy INT8 checkpoint read was cancelled",
        ));
    }
    Ok(())
}

fn validate_published_int8_quantization_sidecars(
    file: &mut File,
    parsed: &ParsedHeader,
    policy: &H3ComfyQuantizationPolicy,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> InspectionResult<()> {
    if policy.quantized_layers.len() != PUBLISHED_INT8_COMFY_QUANT_COUNT {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            format!(
                "published H3 INT8 authority requires exactly {PUBLISHED_INT8_COMFY_QUANT_COUNT} quantized layers"
            ),
        ));
    }
    for layer in &policy.quantized_layers {
        cancellation_boundary_inspection(cancellation)?;
        let name = format!("{layer}.comfy_quant");
        let tensor = parsed.tensors.get(&name).ok_or_else(|| {
            failure(
                H3ComfyCheckpointErrorCode::InvalidMetadata,
                format!("published H3 INT8 quantization sidecar {name:?} is missing"),
            )
        })?;
        if tensor.dtype != "U8"
            || tensor.shape != [PUBLISHED_INT8_COMFY_QUANT_BYTES]
            || tensor.data_offsets[1] - tensor.data_offsets[0]
                != PUBLISHED_INT8_COMFY_QUANT_BYTES as u64
        {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidMetadata,
                format!(
                    "published H3 INT8 quantization sidecar {name:?} does not have the authenticated U8[72] storage"
                ),
            ));
        }
        let data_start = 8u64
            .checked_add(parsed.header_len)
            .and_then(|value| value.checked_add(tensor.data_offsets[0]))
            .ok_or_else(|| {
                failure(
                    H3ComfyCheckpointErrorCode::InvalidHeader,
                    format!("published H3 INT8 quantization sidecar {name:?} offset overflows"),
                )
            })?;
        file.seek(SeekFrom::Start(data_start)).map_err(|error| {
            failure(
                H3ComfyCheckpointErrorCode::Io,
                format!("failed to seek H3 Comfy quantization sidecar {name:?}: {error}"),
            )
        })?;
        let mut payload = [0u8; PUBLISHED_INT8_COMFY_QUANT_BYTES];
        file.read_exact(&mut payload).map_err(|error| {
            failure(
                H3ComfyCheckpointErrorCode::Io,
                format!("failed to read H3 Comfy quantization sidecar {name:?}: {error}"),
            )
        })?;
        if payload != *PUBLISHED_INT8_COMFY_QUANT_PAYLOAD {
            return Err(failure(
                H3ComfyCheckpointErrorCode::InvalidMetadata,
                format!(
                    "published H3 INT8 quantization sidecar {name:?} differs from the authenticated Comfy payload"
                ),
            ));
        }
    }
    cancellation_boundary_inspection(cancellation)
}

#[derive(Clone)]
struct H3ComfyOpenedHashes {
    content_sha256: String,
    block_content_sha256: Vec<String>,
}

/// Process-lifetime content-hash cache for opened Comfy checkpoints.
///
/// Every open fully re-hashed the ~21 GB checkpoint (whole-file plus
/// per-block digests) — and H3 admission opens the checkpoint once per
/// candidate device, so a single placement preview on a 4-GPU host paid four
/// full passes. Hashes are reused only while the file's complete opened
/// metadata identity is byte-for-byte unchanged — the same identity this
/// open already requires to stay stable across the read — and every other
/// check (header parse, quantization sidecars, identity stability before and
/// after) still runs on each open. Any identity drift re-hashes.
static OPENED_CONTENT_HASH_CACHE: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<PathBuf, (H3OpenedFileIdentity, H3ComfyOpenedHashes)>,
    >,
> = std::sync::OnceLock::new();

fn opened_content_hash_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<PathBuf, (H3OpenedFileIdentity, H3ComfyOpenedHashes)>,
> {
    OPENED_CONTENT_HASH_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn lookup_cached_opened_hashes(
    canonical_path: &Path,
    identity: &H3OpenedFileIdentity,
) -> Option<H3ComfyOpenedHashes> {
    let cache = opened_content_hash_cache().lock().ok()?;
    let (cached_identity, hashes) = cache.get(canonical_path)?;
    (cached_identity == identity).then(|| hashes.clone())
}

fn store_cached_opened_hashes(
    canonical_path: &Path,
    identity: H3OpenedFileIdentity,
    hashes: H3ComfyOpenedHashes,
) {
    if let Ok(mut cache) = opened_content_hash_cache().lock() {
        cache.insert(canonical_path.to_path_buf(), (identity, hashes));
    }
}

/// Per-canonical-path single-flight guard: concurrent opens that both miss the
/// content-hash cache must not duplicate a cold ~21 GB hash pass. The loser
/// blocks until the winner stores its hashes, then reuses them through the
/// ordinary cache lookup. Poisoning is ignored — a panicking winner leaves no
/// cache entry, so the next holder simply re-hashes.
static OPENED_HASH_FLIGHTS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<PathBuf, std::sync::Arc<std::sync::Mutex<()>>>>,
> = std::sync::OnceLock::new();

fn opened_hash_flight(canonical_path: &Path) -> std::sync::Arc<std::sync::Mutex<()>> {
    let flights =
        OPENED_HASH_FLIGHTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = match flights.lock() {
        Ok(map) => map,
        Err(poisoned) => poisoned.into_inner(),
    };
    map.entry(canonical_path.to_path_buf()).or_default().clone()
}

fn hash_open_file(
    file: &mut File,
    parsed: &ParsedHeader,
    block_count: usize,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> InspectionResult<H3ComfyOpenedHashes> {
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        failure(
            H3ComfyCheckpointErrorCode::Io,
            format!("failed to seek H3 Comfy checkpoint for hashing: {error}"),
        )
    })?;
    let mut digest = Sha256::new();
    let data_start = 8_u64.checked_add(parsed.header_len).ok_or_else(|| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            "H3 Comfy data offset overflows",
        )
    })?;
    let mut block_ranges = Vec::with_capacity(block_count);
    for index in 0..block_count {
        let prefix = format!("blocks.{index}.");
        let mut ranges = parsed
            .tensors
            .range(prefix.clone()..)
            .take_while(|(name, _)| name.starts_with(&prefix))
            .map(|(_, tensor)| {
                Ok([
                    data_start
                        .checked_add(tensor.data_offsets[0])
                        .ok_or_else(|| {
                            failure(
                                H3ComfyCheckpointErrorCode::InvalidHeader,
                                "H3 Comfy block start offset overflows",
                            )
                        })?,
                    data_start
                        .checked_add(tensor.data_offsets[1])
                        .ok_or_else(|| {
                            failure(
                                H3ComfyCheckpointErrorCode::InvalidHeader,
                                "H3 Comfy block end offset overflows",
                            )
                        })?,
                ])
            })
            .collect::<InspectionResult<Vec<_>>>()?;
        ranges.sort_by_key(|range| range[0]);
        if ranges.is_empty() {
            return Err(failure(
                H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                format!("H3 Comfy opened checkpoint has no tensors for block {index}"),
            ));
        }
        block_ranges.push(ranges);
    }
    let mut block_digests = (0..block_count).map(|_| Sha256::new()).collect::<Vec<_>>();
    let mut verified = 0u64;
    let mut buffer = vec![0u8; FILE_READ_CHUNK_BYTES];
    while verified < parsed.file_len {
        cancellation_boundary_inspection(cancellation)?;
        let remaining =
            usize::try_from((parsed.file_len - verified).min(FILE_READ_CHUNK_BYTES as u64))
                .map_err(|_| {
                    failure(
                        H3ComfyCheckpointErrorCode::Io,
                        "H3 Comfy hash read size does not fit this platform",
                    )
                })?;
        file.read_exact(&mut buffer[..remaining]).map_err(|error| {
            failure(
                H3ComfyCheckpointErrorCode::Io,
                format!("failed to hash H3 Comfy checkpoint: {error}"),
            )
        })?;
        digest.update(&buffer[..remaining]);
        let chunk_end = verified + remaining as u64;
        for (block_digest, ranges) in block_digests.iter_mut().zip(&block_ranges) {
            for [start, end] in ranges {
                let overlap_start = verified.max(*start);
                let overlap_end = chunk_end.min(*end);
                if overlap_start < overlap_end {
                    let start =
                        usize::try_from(overlap_start - verified).expect("chunk offset fits usize");
                    let end =
                        usize::try_from(overlap_end - verified).expect("chunk offset fits usize");
                    block_digest.update(&buffer[start..end]);
                }
            }
        }
        verified += remaining as u64;
    }
    cancellation_boundary_inspection(cancellation)?;
    Ok(H3ComfyOpenedHashes {
        content_sha256: hex_digest(digest.finalize()),
        block_content_sha256: block_digests
            .into_iter()
            .map(|digest| hex_digest(digest.finalize()))
            .collect(),
    })
}

fn comfy_checkpoint_identity(
    candidate: &H3ComfyCheckpointCandidate,
    content_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.comfy-int8-opened-checkpoint.v1\0");
    digest.update(candidate.strategy.stable_id.as_bytes());
    digest.update([candidate.strategy.task as u8]);
    digest.update(candidate.header_identity_sha256.as_bytes());
    digest.update(content_sha256.as_bytes());
    digest.update(
        candidate
            .strategy
            .quantization_policy
            .policy_sha256
            .as_bytes(),
    );
    hex_digest(digest.finalize())
}

impl H3ComfyOpenedSource {
    fn revalidate_authority(&self) -> InspectionResult<()> {
        let path_metadata = std::fs::symlink_metadata(&self.path).map_err(|error| {
            failure(
                H3ComfyCheckpointErrorCode::FileIdentityChanged,
                format!("opened H3 Comfy checkpoint path disappeared: {error}"),
            )
        })?;
        if path_metadata.file_type().is_symlink()
            || !path_metadata.file_type().is_file()
            || H3OpenedFileIdentity::from_metadata(&path_metadata) != self.identity
        {
            return Err(failure(
                H3ComfyCheckpointErrorCode::FileIdentityChanged,
                "opened H3 Comfy checkpoint path no longer names the authenticated file",
            ));
        }
        let file = self.file.lock().map_err(|_| {
            failure(
                H3ComfyCheckpointErrorCode::Io,
                "opened H3 Comfy checkpoint lock is poisoned",
            )
        })?;
        self.verify_identity(&file).map_err(|error| {
            failure(
                H3ComfyCheckpointErrorCode::FileIdentityChanged,
                error.to_string(),
            )
        })
    }

    fn verify_identity(&self, file: &File) -> candle::Result<()> {
        let current = file.metadata().map_err(|error| {
            candle::Error::Msg(format!(
                "failed to stat opened H3 Comfy checkpoint: {error}"
            ))
        })?;
        if H3OpenedFileIdentity::from_metadata(&current) != self.identity {
            candle::bail!("opened H3 Comfy checkpoint changed after full verification")
        }
        Ok(())
    }

    fn tensor_header(&self, name: &str) -> candle::Result<&HeaderTensor> {
        self.header.tensors.get(name).ok_or_else(|| {
            candle::Error::Msg(format!(
                "verified H3 Comfy checkpoint does not contain tensor {name:?}"
            ))
        })
    }

    fn read_tensor_bytes(
        &self,
        name: &str,
        cancellation: &dyn H3ComfyInt8Cancellation,
    ) -> candle::Result<Vec<u8>> {
        cancellation_boundary(cancellation)?;
        let tensor = self.tensor_header(name)?;
        let len = usize::try_from(tensor.data_offsets[1] - tensor.data_offsets[0])
            .map_err(|_| candle::Error::Msg(format!("H3 Comfy tensor {name:?} is too large")))?;
        let data_start = 8u64
            .checked_add(self.header.header_len)
            .and_then(|value| value.checked_add(tensor.data_offsets[0]))
            .ok_or_else(|| candle::Error::Msg("H3 Comfy tensor offset overflows".into()))?;
        let mut bytes = vec![0u8; len];
        let mut file = self
            .file
            .lock()
            .map_err(|_| candle::Error::Msg("H3 Comfy file lock was poisoned".into()))?;
        self.verify_identity(&file)?;
        file.seek(SeekFrom::Start(data_start)).map_err(|error| {
            candle::Error::Msg(format!("failed to seek H3 Comfy tensor {name:?}: {error}"))
        })?;
        let mut offset = 0usize;
        while offset < bytes.len() {
            cancellation_boundary(cancellation)?;
            let end = (offset + FILE_READ_CHUNK_BYTES).min(bytes.len());
            file.read_exact(&mut bytes[offset..end]).map_err(|error| {
                candle::Error::Msg(format!("failed to read H3 Comfy tensor {name:?}: {error}"))
            })?;
            self.runtime_bytes_read
                .fetch_add((end - offset) as u64, Ordering::Relaxed);
            offset = end;
        }
        self.verify_identity(&file)?;
        cancellation_boundary(cancellation)?;
        Ok(bytes)
    }

    fn load_dense_tensor(
        &self,
        name: &str,
        requested_dtype: DType,
        device: &Device,
        cancellation: &dyn H3ComfyInt8Cancellation,
    ) -> candle::Result<Tensor> {
        let header = self.tensor_header(name)?;
        let f16_curve_projection_upcast = header.dtype == "F16";
        let source_dtype = match header.dtype.as_str() {
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            "F16"
                if requested_dtype == DType::F32
                    && self.f16_curve_projection_tensors.contains(name) =>
            {
                DType::F16
            }
            "F16" => candle::bail!(
                "H3 Comfy F16 storage is authorized only for published curve projections loaded as logical F32; refused tensor {name:?} as {requested_dtype:?}"
            ),
            dtype => candle::bail!(
                "H3 Comfy dense backend refuses encoded dtype {dtype:?} for tensor {name:?}"
            ),
        };
        let bytes = self.read_tensor_bytes(name, cancellation)?;
        let tensor = Tensor::from_raw_buffer(&bytes, source_dtype, &header.shape, &Device::Cpu)?;
        let tensor = if tensor.dtype() == requested_dtype {
            tensor
        } else {
            tensor.to_dtype(requested_dtype)?
        };
        let tensor = tensor.to_device(device)?;
        if f16_curve_projection_upcast {
            self.f16_curve_projection_upcast_loads
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(tensor)
    }

    fn load_raw_int8(
        &self,
        name: &str,
        cancellation: &dyn H3ComfyInt8Cancellation,
    ) -> candle::Result<Tensor> {
        let header = self.tensor_header(name)?;
        if header.dtype != "I8" {
            candle::bail!("H3 Comfy tensor {name:?} is not exact I8 storage")
        }
        let bytes = self.read_tensor_bytes(name, cancellation)?;
        Tensor::from_raw_buffer(&bytes, DType::U8, &header.shape, &Device::Cpu)
    }

    fn load_f32_cpu(
        &self,
        name: &str,
        cancellation: &dyn H3ComfyInt8Cancellation,
    ) -> candle::Result<Tensor> {
        self.load_dense_tensor(name, DType::F32, &Device::Cpu, cancellation)
    }
}

struct H3ComfyOpenedBackend {
    source: Arc<H3ComfyOpenedSource>,
    cancellation: Arc<dyn H3ComfyInt8Cancellation>,
}

impl SimpleBackend for H3ComfyOpenedBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hints: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle::Result<Tensor> {
        let tensor =
            self.source
                .load_dense_tensor(name, dtype, device, self.cancellation.as_ref())?;
        if tensor.shape() != &shape {
            return Err(candle::Error::UnexpectedShape {
                msg: format!("H3 Comfy opened-file backend shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> candle::Result<Tensor> {
        self.source
            .load_dense_tensor(name, dtype, device, self.cancellation.as_ref())
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.source.header.tensors.contains_key(name)
    }
}

pub struct H3ComfyInt8BlockLoader {
    config: H3TransformerConfig,
    mode: H3AdaLnMode,
    precision: H3PrecisionProfile,
    source: Arc<H3ComfyOpenedSource>,
    vb: VarBuilder<'static>,
    identity: Arc<H3StreamedTransformerIdentity>,
    cancellation: Arc<dyn H3ComfyInt8Cancellation>,
}

impl fmt::Debug for H3ComfyInt8BlockLoader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("H3ComfyInt8BlockLoader")
            .field("task", &self.identity.task())
            .field("block_count", &self.identity.block_count())
            .field("live_blocks", &self.identity.live_block_count())
            .field("content_sha256", &self.source.content_sha256)
            .finish_non_exhaustive()
    }
}

impl H3ComfyInt8BlockLoader {
    pub fn task(&self) -> H3TransformerTask {
        self.identity.task()
    }

    pub fn block_count(&self) -> usize {
        self.identity.block_count()
    }

    pub fn live_block_count(&self) -> usize {
        self.identity.live_block_count()
    }

    pub fn content_sha256(&self) -> &str {
        &self.source.content_sha256
    }

    /// Exact schema, policy, and authenticated-content identity shared with
    /// the resident transformer created by the same opened checkpoint.
    pub fn checkpoint_identity_sha256(&self) -> &str {
        self.identity
            .checkpoint_identity_sha256()
            .expect("Comfy block loaders always carry a checkpoint identity")
    }

    /// True only when this loader and resident transformer came from the same
    /// opened checkpoint authority, not merely equal artifact bytes.
    pub fn is_exact_pair_for(&self, transformer: &H3StreamedTransformer) -> bool {
        transformer.shares_streamed_identity(&self.identity)
    }

    pub fn bytes_read(&self) -> u64 {
        self.source.runtime_bytes_read.load(Ordering::Relaxed)
    }

    pub fn f16_curve_projection_upcast_loads(&self) -> u64 {
        self.source
            .f16_curve_projection_upcast_loads
            .load(Ordering::Relaxed)
    }

    pub fn block_memory(&self, index: usize) -> candle::Result<H3ComfyInt8BlockMemory> {
        calculate_block_memory(
            &self.source.header,
            &self.source.f16_curve_projection_tensors,
            index,
            self.block_count(),
        )
    }

    /// Load exactly one indexed main block. A second live block is rejected so
    /// neither compressed host weights nor protected device tensors can grow
    /// with the fifty-block stack.
    pub fn load_block(&mut self, index: usize) -> candle::Result<H3LoadedTransformerBlock> {
        cancellation_boundary(self.cancellation.as_ref())?;
        if index >= self.block_count() {
            candle::bail!(
                "MiniMax H3 block index {index} is outside the exact block count {}",
                self.block_count()
            )
        }
        if self.live_block_count() != 0 {
            candle::bail!("MiniMax H3 Comfy INT8 loader permits exactly one live main block")
        }
        let prefix = format!("blocks.{index}");
        let linear = |suffix: &str| -> candle::Result<H3ComfyInt8ConvRotLinear> {
            let layer = format!("{prefix}.{suffix}");
            H3ComfyInt8ConvRotLinear::new(
                self.source
                    .load_raw_int8(&format!("{layer}.weight"), self.cancellation.as_ref())?,
                self.source
                    .load_f32_cpu(&format!("{layer}.weight_scale"), self.cancellation.as_ref())?,
            )
        };
        let matrices = H3ComfyInt8BlockMatrices {
            qkv: linear("attn.qkv_proj")?,
            out: linear("attn.out_proj")?,
            fc1: linear("mlp.fc1")?,
            fc2: linear("mlp.fc2")?,
        };
        cancellation_boundary(self.cancellation.as_ref())?;
        H3LoadedTransformerBlock::new_comfy_int8(
            index,
            self.identity.clone(),
            &self.config,
            self.mode,
            self.precision,
            matrices,
            self.vb.pp(prefix),
        )
    }
}

fn calculate_block_memory(
    header: &ParsedHeader,
    f16_curve_projection_tensors: &BTreeSet<String>,
    index: usize,
    block_count: usize,
) -> candle::Result<H3ComfyInt8BlockMemory> {
    if index >= block_count {
        candle::bail!(
            "MiniMax H3 block index {index} is outside the exact block count {block_count}"
        )
    }
    let prefix = format!("blocks.{index}.");
    let mut encoded_host_bytes = 0u64;
    let mut max_host_read_staging_bytes = 0u64;
    let mut max_device_weight_staging_bytes = 0u64;
    let mut protected_device_bytes = 0u64;
    for (name, tensor) in header
        .tensors
        .range(prefix.clone()..)
        .take_while(|(name, _)| name.starts_with(&prefix))
    {
        if name.ends_with(".comfy_quant") {
            continue;
        }
        let bytes = tensor.data_offsets[1] - tensor.data_offsets[0];
        max_host_read_staging_bytes = max_host_read_staging_bytes.max(bytes);
        let quantized = QUANTIZED_BLOCK_WEIGHT_SUFFIXES
            .iter()
            .any(|suffix| name == &format!("{prefix}{suffix}"));
        let scale = name.ends_with(".weight_scale");
        if quantized || scale {
            encoded_host_bytes = encoded_host_bytes.checked_add(bytes).ok_or_else(|| {
                candle::Error::Msg("H3 Comfy block host byte count overflows".into())
            })?;
            if quantized {
                let rows = tensor.shape[0].min(H3_COMFY_PORTABLE_ROW_CHUNK);
                let staging = rows
                    .checked_mul(tensor.shape[1])
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| {
                        candle::Error::Msg("H3 Comfy device staging bytes overflow".into())
                    })? as u64;
                max_device_weight_staging_bytes = max_device_weight_staging_bytes.max(staging);
            }
        } else {
            let device_bytes = if tensor.dtype == "F16"
                && f16_curve_projection_tensors.contains(name)
            {
                tensor
                    .shape
                    .iter()
                    .try_fold(1u64, |elements, dimension| {
                        elements.checked_mul(*dimension as u64)
                    })
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u64))
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "H3 Comfy protected F32 device byte count overflows".into(),
                        )
                    })?
            } else {
                bytes
            };
            protected_device_bytes = protected_device_bytes
                .checked_add(device_bytes)
                .ok_or_else(|| {
                    candle::Error::Msg("H3 Comfy protected block bytes overflow".into())
                })?;
        }
    }
    Ok(H3ComfyInt8BlockMemory {
        encoded_host_bytes,
        max_host_read_staging_bytes,
        max_device_weight_staging_bytes,
        protected_device_bytes,
    })
}

fn calculate_opened_memory_evidence(
    parsed: &ParsedHeader,
    f16_curve_projection_tensors: &BTreeSet<String>,
    block_count: usize,
    block_content_sha256: &[String],
) -> InspectionResult<H3ComfyOpenedMemoryEvidence> {
    if block_count == 0
        || block_content_sha256.len() != block_count
        || block_content_sha256.iter().any(|digest| {
            digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
    {
        return Err(failure(
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
            "H3 Comfy opened checkpoint block evidence is incomplete",
        ));
    }

    let mut fixed_encoded_host_bytes = 0_u64;
    let mut fixed_protected_device_bytes = 0_u64;
    let mut fixed_max_host_read_staging_bytes = 0_u64;
    let mut fixed_max_device_weight_staging_bytes = 0_u64;
    for (name, tensor) in &parsed.tensors {
        if name.starts_with("blocks.") || name.ends_with(".comfy_quant") {
            continue;
        }
        let encoded = tensor.data_offsets[1] - tensor.data_offsets[0];
        let logical_device = if tensor.dtype == "F16" && f16_curve_projection_tensors.contains(name)
        {
            tensor
                .shape
                .iter()
                .try_fold(1_u64, |elements, dimension| {
                    elements.checked_mul(*dimension as u64)
                })
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u64))
                .ok_or_else(|| {
                    failure(
                        H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                        "H3 Comfy fixed logical device bytes overflow",
                    )
                })?
        } else {
            encoded
        };
        fixed_encoded_host_bytes =
            fixed_encoded_host_bytes
                .checked_add(encoded)
                .ok_or_else(|| {
                    failure(
                        H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                        "H3 Comfy fixed host bytes overflow",
                    )
                })?;
        fixed_protected_device_bytes = fixed_protected_device_bytes
            .checked_add(logical_device)
            .ok_or_else(|| {
                failure(
                    H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                    "H3 Comfy fixed device bytes overflow",
                )
            })?;
        fixed_max_host_read_staging_bytes = fixed_max_host_read_staging_bytes.max(encoded);
        fixed_max_device_weight_staging_bytes =
            fixed_max_device_weight_staging_bytes.max(logical_device);
    }
    if fixed_encoded_host_bytes == 0
        || fixed_protected_device_bytes == 0
        || fixed_max_host_read_staging_bytes == 0
        || fixed_max_device_weight_staging_bytes == 0
    {
        return Err(failure(
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
            "H3 Comfy opened checkpoint has incomplete fixed memory evidence",
        ));
    }

    let blocks = (0..block_count)
        .map(|index| {
            let memory =
                calculate_block_memory(parsed, f16_curve_projection_tensors, index, block_count)
                    .map_err(|error| {
                        failure(
                            H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                            error.to_string(),
                        )
                    })?;
            if memory.encoded_host_bytes == 0
                || memory.protected_device_bytes == 0
                || memory.max_host_read_staging_bytes == 0
                || memory.max_device_weight_staging_bytes == 0
            {
                return Err(failure(
                    H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                    format!("H3 Comfy block {index} has incomplete memory evidence"),
                ));
            }
            Ok(H3ComfyOpenedBlockMemoryEvidence {
                index: u16::try_from(index).map_err(|_| {
                    failure(
                        H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                        "H3 Comfy block index exceeds u16",
                    )
                })?,
                memory,
                content_sha256: block_content_sha256[index].clone(),
            })
        })
        .collect::<InspectionResult<Vec<_>>>()?;
    let header_bytes = parsed.header_len.checked_add(8).ok_or_else(|| {
        failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            "H3 Comfy header byte charge overflows",
        )
    })?;
    let mut evidence = H3ComfyOpenedMemoryEvidence {
        verified_file_bytes: parsed.file_len,
        header_bytes,
        fixed_encoded_host_bytes,
        fixed_protected_device_bytes,
        fixed_max_host_read_staging_bytes,
        fixed_max_device_weight_staging_bytes,
        blocks,
        identity_sha256: String::new(),
    };
    evidence.identity_sha256 = opened_memory_evidence_identity(&evidence);
    Ok(evidence)
}

fn opened_memory_evidence_identity(evidence: &H3ComfyOpenedMemoryEvidence) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.comfy-opened-memory-evidence.v1\0");
    for value in [
        evidence.verified_file_bytes,
        evidence.header_bytes,
        evidence.fixed_encoded_host_bytes,
        evidence.fixed_protected_device_bytes,
        evidence.fixed_max_host_read_staging_bytes,
        evidence.fixed_max_device_weight_staging_bytes,
    ] {
        digest.update(value.to_le_bytes());
    }
    digest.update((evidence.blocks.len() as u64).to_le_bytes());
    for block in &evidence.blocks {
        digest.update(block.index.to_le_bytes());
        for value in [
            block.memory.encoded_host_bytes,
            block.memory.protected_device_bytes,
            block.memory.max_device_weight_staging_bytes,
            block.memory.max_host_read_staging_bytes,
        ] {
            digest.update(value.to_le_bytes());
        }
        digest.update(block.content_sha256.as_bytes());
    }
    hex_digest(digest.finalize())
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

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantizationMetadata {
    #[serde(default)]
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
    let published_int8_artifact =
        published_artifact.filter(|artifact| artifact.format() == H3ComfyPrunedFormat::Int8ConvRot);
    let (quantization, detected_format, embedded_quantization_tensors) = if let Some(artifact) =
        published_int8_artifact
    {
        validate_published_int8_header_authority(&parsed, artifact, mode)?;
        (
            Some(published_int8_quantization(&config, mode)?),
            H3ComfyPrunedFormat::Int8ConvRot,
            true,
        )
    } else if let Some(config_json) = parsed.metadata.get("config") {
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
        let detected = detect_format(quantization.as_ref())?;
        (quantization, detected, false)
    } else {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            "H3 Comfy checkpoint is missing __metadata__.config",
        ));
    };
    if detected_format != expected_format {
        return Err(failure(
            H3ComfyCheckpointErrorCode::FormatMismatch,
            format!(
                "H3 Comfy header format {detected_format:?} does not match source authority {expected_format:?}"
            ),
        ));
    }
    let (mut expected, policy) =
        expected_schema(&config, mode, detected_format, quantization.as_ref())?;
    if embedded_quantization_tensors {
        apply_published_int8_storage_overrides(&mut expected, &config)?;
        add_published_int8_quantization_tensors(&mut expected, &policy)?;
    }
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

fn validate_published_int8_header_authority(
    parsed: &ParsedHeader,
    artifact: H3ComfyPublishedArtifact,
    mode: H3AdaLnMode,
) -> InspectionResult<()> {
    if artifact.format() != H3ComfyPrunedFormat::Int8ConvRot {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            "only the source-pinned published INT8 ConvRot header may omit safetensors metadata",
        ));
    }
    if !parsed.metadata.is_empty() {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            "published H3 INT8 header authority requires the released empty metadata map",
        ));
    }
    if parsed.file_len != artifact.file_bytes() {
        return Err(failure(
            H3ComfyCheckpointErrorCode::SourceSizeMismatch,
            "published H3 INT8 artifact length differs from its source authority",
        ));
    }
    if parsed.header_len != H3_COMFY_PUBLISHED_INT8_HEADER_LEN
        || parsed.header_identity_sha256 != H3_COMFY_PUBLISHED_INT8_HEADER_SHA256
        || parsed.tensors.len() != H3_COMFY_PUBLISHED_INT8_TENSOR_COUNT
    {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidHeader,
            "published H3 INT8 header identity differs from the authenticated released object",
        ));
    }
    if mode
        != (H3AdaLnMode::Curve {
            grid: PUBLISHED_INT8_CURVE_GRID,
            basis_dim: PUBLISHED_INT8_CURVE_BASIS,
        })
    {
        return Err(failure(
            H3ComfyCheckpointErrorCode::ConfigMismatch,
            "published H3 INT8 curve shape differs from the authenticated released object",
        ));
    }
    Ok(())
}

fn published_int8_quantization(
    config: &H3TransformerConfig,
    mode: H3AdaLnMode,
) -> InspectionResult<QuantizationMetadata> {
    let base = expected_h3_weight_specs(config, mode, H3PrecisionProfile::OfficialMixedBf16F32)
        .map_err(|error| {
            failure(
                H3ComfyCheckpointErrorCode::ConfigMismatch,
                error.to_string(),
            )
        })?;
    let layers = published_quantized_layers(config, &base)?
        .into_iter()
        .map(|name| {
            (
                name,
                QuantizedLayerMetadata {
                    format: "int8_tensorwise".to_owned(),
                    full_precision_matrix_mult: None,
                    convrot: Some(true),
                    convrot_groupsize: Some(CONVROT_GROUP_SIZE),
                },
            )
        })
        .collect();
    Ok(QuantizationMetadata {
        format_version: "1.0".to_owned(),
        layers,
    })
}

fn add_published_int8_quantization_tensors(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    policy: &H3ComfyQuantizationPolicy,
) -> InspectionResult<()> {
    for layer in &policy.quantized_layers {
        let name = format!("{layer}.comfy_quant");
        if expected
            .insert(
                name.clone(),
                ExpectedTensor {
                    dtype: "U8",
                    shape: vec![PUBLISHED_INT8_COMFY_QUANT_BYTES],
                },
            )
            .is_some()
        {
            return Err(failure(
                H3ComfyCheckpointErrorCode::TensorSchemaMismatch,
                format!("published H3 INT8 quantization tensor {name:?} collides with a weight"),
            ));
        }
    }
    Ok(())
}

fn apply_published_int8_storage_overrides(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    config: &H3TransformerConfig,
) -> InspectionResult<()> {
    for name in published_int8_curve_projection_tensors(config) {
        let tensor = expected.get_mut(&name).ok_or_else(|| {
            failure(
                H3ComfyCheckpointErrorCode::ConfigMismatch,
                format!("published H3 INT8 storage override references missing tensor {name:?}"),
            )
        })?;
        if tensor.dtype != "F32" {
            return Err(failure(
                H3ComfyCheckpointErrorCode::ConfigMismatch,
                format!("published H3 INT8 storage override expected F32 source tensor {name:?}"),
            ));
        }
        // The released Comfy INT8 conversion stores the curve projections in
        // F16 even though the logical H3 curve computation remains an
        // FP32-sensitive island. A runtime must upcast these values; this is a
        // source-pinned storage fact, not permission to lower compute precision.
        tensor.dtype = "F16";
    }
    Ok(())
}

fn published_int8_curve_projection_tensors(config: &H3TransformerConfig) -> BTreeSet<String> {
    (0..config.num_layers)
        .flat_map(|index| {
            [
                format!("blocks.{index}.adaln_proj.linear.weight"),
                format!("blocks.{index}.adaln_proj.linear.bias"),
            ]
        })
        .chain([
            "final_layer.adaln_proj.linear.weight".to_owned(),
            "final_layer.adaln_proj.linear.bias".to_owned(),
        ])
        .collect()
}

fn detect_format(
    quantization: Option<&QuantizationMetadata>,
) -> InspectionResult<H3ComfyPrunedFormat> {
    let Some(quantization) = quantization else {
        return Ok(H3ComfyPrunedFormat::Bf16);
    };
    if (!quantization.format_version.is_empty() && quantization.format_version != "1.0")
        || quantization.layers.is_empty()
    {
        return Err(failure(
            H3ComfyCheckpointErrorCode::InvalidMetadata,
            "H3 Comfy quantization metadata must have layers and use format_version 1.0 when versioned",
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

fn published_quantized_layers(
    config: &H3TransformerConfig,
    base: &BTreeMap<String, super::dit::H3TensorSpec>,
) -> InspectionResult<BTreeSet<String>> {
    (0..config.num_layers)
        .flat_map(|index| {
            QUANTIZED_BLOCK_WEIGHT_SUFFIXES
                .iter()
                .map(move |suffix| format!("blocks.{index}.{suffix}"))
        })
        .map(|weight| {
            let spec = base.get(&weight).ok_or_else(|| {
                failure(
                    H3ComfyCheckpointErrorCode::ConfigMismatch,
                    format!("H3 Comfy quantization policy references missing weight {weight:?}"),
                )
            })?;
            if spec.dtype != DType::BF16 || spec.shape.len() != 2 {
                return Err(failure(
                    H3ComfyCheckpointErrorCode::ConfigMismatch,
                    format!("H3 Comfy quantization policy requires BF16 matrix {weight:?}"),
                ));
            }
            Ok(weight.trim_end_matches(".weight").to_owned())
        })
        .collect()
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
    // The published H3 conversions quantize exactly the four large matrix
    // weights in each of the 50 denoising blocks. Input projections, the text
    // token refiner, all norms/biases, the FP32 patch/output islands, and the
    // curve AdaLN projections remain protected from INT8. The source-pinned
    // published INT8 path later applies its observed F16 curve-projection
    // storage while preserving the logical FP32 compute contract. Deriving the
    // quantized set from "every BF16 matrix" would silently admit nine layers
    // that the published artifacts intentionally retain in BF16.
    let quantizable = published_quantized_layers(config, &base)?;
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
                        || layer.full_precision_matrix_mult == Some(true)
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
        let quantized_output_features = quantized.then(|| spec.shape[0]);
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
                    shape: match format {
                        H3ComfyPrunedFormat::Fp8Scaled => vec![],
                        // comfy-kitchen's ConvRot path requires per-output-row
                        // quantization and preserves the reduced axis.
                        H3ComfyPrunedFormat::Int8ConvRot => {
                            vec![quantized_output_features.unwrap(), 1]
                        }
                        H3ComfyPrunedFormat::Bf16 => unreachable!(),
                    },
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
            let mut canonical = metadata.clone();
            if canonical.full_precision_matrix_mult == Some(false) {
                canonical.full_precision_matrix_mult = None;
            }
            let encoded = serde_json::to_vec(&canonical).map_err(|error| {
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
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    use candle::{Device, Tensor};

    use super::*;
    use crate::minimax_h3::interpolate_h3_adaln_curve;

    #[test]
    fn opened_content_hash_cache_reuses_only_identical_identities() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.safetensors");
        std::fs::write(&path, b"opened checkpoint bytes").unwrap();
        let identity =
            H3OpenedFileIdentity::from_metadata(&std::fs::symlink_metadata(&path).unwrap());
        let hashes = H3ComfyOpenedHashes {
            content_sha256: "a".repeat(64),
            block_content_sha256: vec!["b".repeat(64)],
        };

        assert!(lookup_cached_opened_hashes(&path, &identity).is_none());
        store_cached_opened_hashes(&path, identity.clone(), hashes.clone());
        let hit = lookup_cached_opened_hashes(&path, &identity).unwrap();
        assert_eq!(hit.content_sha256, hashes.content_sha256);
        assert_eq!(hit.block_content_sha256, hashes.block_content_sha256);

        // Any metadata drift is a miss, so changed bytes are always re-hashed.
        std::fs::write(&path, b"replaced checkpoint bytes!!").unwrap();
        let changed =
            H3OpenedFileIdentity::from_metadata(&std::fs::symlink_metadata(&path).unwrap());
        assert!(lookup_cached_opened_hashes(&path, &changed).is_none());
    }

    #[test]
    fn opened_hash_flight_serializes_cold_fills_per_path() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("flight-checkpoint.safetensors");
        std::fs::write(&path, b"flight checkpoint bytes").unwrap();
        let other = directory.path().join("other-checkpoint.safetensors");

        assert!(std::sync::Arc::ptr_eq(
            &opened_hash_flight(&path),
            &opened_hash_flight(&path)
        ));
        assert!(!std::sync::Arc::ptr_eq(
            &opened_hash_flight(&path),
            &opened_hash_flight(&other)
        ));

        // A loser blocked on the winner's flight observes the winner's stored
        // hashes as soon as it acquires the lock — it never re-hashes.
        let identity =
            H3OpenedFileIdentity::from_metadata(&std::fs::symlink_metadata(&path).unwrap());
        let hashes = H3ComfyOpenedHashes {
            content_sha256: "e".repeat(64),
            block_content_sha256: vec!["f".repeat(64)],
        };
        let flight = opened_hash_flight(&path);
        let winner_guard = flight.lock().unwrap();
        let loser = std::thread::spawn({
            let flight = opened_hash_flight(&path);
            let path = path.clone();
            let identity = identity.clone();
            move || {
                let _guard = flight.lock().unwrap();
                lookup_cached_opened_hashes(&path, &identity)
            }
        });
        store_cached_opened_hashes(&path, identity, hashes.clone());
        drop(winner_guard);
        let observed = loser.join().unwrap().unwrap();
        assert_eq!(observed.content_sha256, hashes.content_sha256);
    }

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

    fn runtime_config() -> H3TransformerConfig {
        H3TransformerConfig {
            hidden_size: 256,
            num_layers: 2,
            token_refiner_num_layers: 1,
            num_attention_heads: 2,
            attention_head_dim: 128,
            ffn_hidden_size: 256,
            video_latent_channels: 64,
            audio_latent_channels: 256,
            patch_size: [1, 2, 2],
            text_dim: 256,
            timestep_input_dim: 64,
            time_embed_hidden_size: 256,
            time_embed_dim: 128,
            rope_inv_freq_len: 16,
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
        let layers = published_quantized_layers(config, &base)
            .unwrap()
            .into_iter()
            .map(|name| {
                let metadata = match format {
                    H3ComfyPrunedFormat::Fp8Scaled => serde_json::json!({
                        "format": "float8_e4m3fn"
                    }),
                    H3ComfyPrunedFormat::Int8ConvRot => serde_json::json!({
                        "format": "int8_tensorwise",
                        "convrot": true,
                        "convrot_groupsize": CONVROT_GROUP_SIZE
                    }),
                    H3ComfyPrunedFormat::Bf16 => unreachable!(),
                };
                (name, metadata)
            })
            .collect::<Map<_, _>>();
        serde_json::json!({"layers": layers}).to_string()
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

    fn published_int8_parsed_header(artifact: H3ComfyPublishedArtifact) -> ParsedHeader {
        assert_eq!(artifact.format(), H3ComfyPrunedFormat::Int8ConvRot);
        let config = H3TransformerConfig::default();
        let mode = H3AdaLnMode::Curve {
            grid: PUBLISHED_INT8_CURVE_GRID,
            basis_dim: PUBLISHED_INT8_CURVE_BASIS,
        };
        let quantization = published_int8_quantization(&config, mode).unwrap();
        let (mut expected, policy) = expected_schema(
            &config,
            mode,
            H3ComfyPrunedFormat::Int8ConvRot,
            Some(&quantization),
        )
        .unwrap();
        apply_published_int8_storage_overrides(&mut expected, &config).unwrap();
        add_published_int8_quantization_tensors(&mut expected, &policy).unwrap();
        let mut cursor = 0_u64;
        let tensors = expected
            .into_iter()
            .map(|(name, tensor)| {
                let elements = tensor.shape.iter().product::<usize>() as u64;
                let bytes = elements * dtype_size(tensor.dtype).unwrap();
                let offsets = [cursor, cursor + bytes];
                cursor += bytes;
                (
                    name,
                    HeaderTensor {
                        dtype: tensor.dtype.to_owned(),
                        shape: tensor.shape,
                        data_offsets: offsets,
                    },
                )
            })
            .collect();
        assert_eq!(
            cursor,
            artifact.file_bytes() - H3_COMFY_PUBLISHED_INT8_HEADER_LEN - 8
        );
        ParsedHeader {
            metadata: BTreeMap::new(),
            tensors,
            header_len: H3_COMFY_PUBLISHED_INT8_HEADER_LEN,
            file_len: artifact.file_bytes(),
            header_identity_sha256: H3_COMFY_PUBLISHED_INT8_HEADER_SHA256.to_owned(),
        }
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

    fn fill_f32_tensor(header: &Value, data: &mut [u8], name: &str, value: f32) {
        let tensor = &header[name];
        let start = tensor["data_offsets"][0].as_u64().unwrap() as usize;
        let end = tensor["data_offsets"][1].as_u64().unwrap() as usize;
        for bytes in data[start..end].chunks_exact_mut(4) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
    }

    fn runtime_fixture() -> (H3TransformerConfig, Value, Vec<u8>) {
        let config = runtime_config();
        let mode = H3AdaLnMode::Curve {
            grid: 5,
            basis_dim: 64,
        };
        let (header, mut data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Int8ConvRot);
        let scales = header
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| name.ends_with(".weight_scale"))
            .cloned()
            .collect::<Vec<_>>();
        for scale in scales {
            fill_f32_tensor(&header, &mut data, &scale, 1.0);
        }
        (config, header, data)
    }

    #[derive(Default)]
    struct ToggleCancellation(AtomicBool);

    impl H3ComfyInt8Cancellation for ToggleCancellation {
        fn is_cancelled(&self) -> bool {
            self.0.load(AtomicOrdering::Relaxed)
        }
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
    fn published_int8_headers_use_source_pinned_embedded_quantization() {
        for artifact in [
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot,
        ] {
            let candidate = inspect_parsed_header(
                published_int8_parsed_header(artifact),
                H3TransformerConfig::default(),
                artifact.task(),
                artifact.format(),
                Some(artifact),
            )
            .unwrap();

            assert_eq!(candidate.artifact, artifact);
            assert_eq!(candidate.strategy.task, artifact.task());
            assert_eq!(candidate.strategy.format, H3ComfyPrunedFormat::Int8ConvRot);
            assert_eq!(
                candidate.strategy.adaln_mode,
                H3AdaLnMode::Curve {
                    grid: PUBLISHED_INT8_CURVE_GRID,
                    basis_dim: PUBLISHED_INT8_CURVE_BASIS,
                }
            );
            assert_eq!(candidate.expected_content_sha256, artifact.content_sha256());
            assert_eq!(
                candidate.header_identity_sha256,
                H3_COMFY_PUBLISHED_INT8_HEADER_SHA256
            );
            assert_eq!(candidate.tensor_count, H3_COMFY_PUBLISHED_INT8_TENSOR_COUNT);
            assert_eq!(
                candidate
                    .strategy
                    .quantization_policy
                    .quantized_layers
                    .len(),
                200
            );
            assert_eq!(
                candidate.strategy.memory.header_bytes,
                H3_COMFY_PUBLISHED_INT8_HEADER_LEN + 8
            );
            assert_eq!(
                candidate.strategy.memory.encoded_tensor_bytes,
                artifact.file_bytes() - H3_COMFY_PUBLISHED_INT8_HEADER_LEN - 8
            );
            assert_eq!(candidate.strategy.memory.resident_device_weight_bytes, None);
        }
    }

    #[test]
    fn published_int8_header_identity_and_schema_mutations_fail_closed() {
        let artifact = H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot;
        let inspect = |parsed| {
            inspect_parsed_header(
                parsed,
                H3TransformerConfig::default(),
                artifact.task(),
                artifact.format(),
                Some(artifact),
            )
        };

        let mut parsed = published_int8_parsed_header(artifact);
        parsed.header_identity_sha256 = "0".repeat(64);
        assert_eq!(
            inspect(parsed).unwrap_err().code,
            H3ComfyCheckpointErrorCode::InvalidHeader
        );

        let mut parsed = published_int8_parsed_header(artifact);
        parsed.metadata.insert(
            "config".to_owned(),
            config_metadata(
                &H3TransformerConfig::default(),
                PUBLISHED_INT8_CURVE_GRID,
                PUBLISHED_INT8_CURVE_BASIS,
            ),
        );
        assert_eq!(
            inspect(parsed).unwrap_err().code,
            H3ComfyCheckpointErrorCode::InvalidMetadata
        );

        let mut parsed = published_int8_parsed_header(artifact);
        parsed.tensors.get_mut("adaln_t_table").unwrap().shape[0] = PUBLISHED_INT8_CURVE_GRID - 1;
        assert_eq!(
            inspect(parsed).unwrap_err().code,
            H3ComfyCheckpointErrorCode::ConfigMismatch
        );

        let mut parsed = published_int8_parsed_header(artifact);
        let sidecar = parsed
            .tensors
            .keys()
            .find(|name| name.ends_with(".comfy_quant"))
            .unwrap()
            .clone();
        parsed.tensors.get_mut(&sidecar).unwrap().shape = vec![71];
        assert_eq!(
            inspect(parsed).unwrap_err().code,
            H3ComfyCheckpointErrorCode::TensorSchemaMismatch
        );

        let mut parsed = published_int8_parsed_header(artifact);
        parsed.tensors.remove(&sidecar);
        assert_eq!(
            inspect(parsed).unwrap_err().code,
            H3ComfyCheckpointErrorCode::InvalidHeader
        );
    }

    #[test]
    fn missing_metadata_is_allowed_only_for_pinned_published_int8_artifacts() {
        let artifact = H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot;
        assert_eq!(
            inspect_parsed_header(
                published_int8_parsed_header(artifact),
                H3TransformerConfig::default(),
                artifact.task(),
                artifact.format(),
                None,
            )
            .unwrap_err()
            .code,
            H3ComfyCheckpointErrorCode::InvalidMetadata
        );

        assert_eq!(
            inspect_parsed_header(
                published_int8_parsed_header(artifact),
                H3TransformerConfig::default(),
                H3TransformerTask::T2VaFl2Va,
                H3ComfyPrunedFormat::Bf16,
                Some(H3ComfyPublishedArtifact::Fl2VaPrunedBf16),
            )
            .unwrap_err()
            .code,
            H3ComfyCheckpointErrorCode::InvalidMetadata
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
        assert_eq!(fp8_policy.quantized_layers.len(), 200);
        assert_eq!(fp8.len(), 932);
        let mut explicit_false_fp8 = fp8_metadata.clone();
        for layer in explicit_false_fp8.layers.values_mut() {
            layer.full_precision_matrix_mult = Some(false);
        }
        let (_, explicit_false_policy) = expected_schema(
            &config,
            mode,
            H3ComfyPrunedFormat::Fp8Scaled,
            Some(&explicit_false_fp8),
        )
        .unwrap();
        assert_eq!(
            fp8_policy.policy_sha256,
            explicit_false_policy.policy_sha256
        );
        let mut invalid_full_precision_fp8 = fp8_metadata.clone();
        invalid_full_precision_fp8
            .layers
            .values_mut()
            .next()
            .unwrap()
            .full_precision_matrix_mult = Some(true);
        assert_eq!(
            expected_schema(
                &config,
                mode,
                H3ComfyPrunedFormat::Fp8Scaled,
                Some(&invalid_full_precision_fp8),
            )
            .unwrap_err()
            .code,
            H3ComfyCheckpointErrorCode::InvalidMetadata
        );

        let int8_raw = quant_metadata(&config, mode, H3ComfyPrunedFormat::Int8ConvRot);
        let int8_metadata: QuantizationMetadata = serde_json::from_str(&int8_raw).unwrap();
        let (int8, int8_policy) = expected_schema(
            &config,
            mode,
            H3ComfyPrunedFormat::Int8ConvRot,
            Some(&int8_metadata),
        )
        .unwrap();
        assert_eq!(int8_policy.quantized_layers.len(), 200);
        assert_eq!(int8_policy.protected_weight_tensors.len(), 274);
        assert_eq!(int8.len(), 732);
        assert_eq!(
            int8["blocks.0.attn.qkv_proj.weight_scale"].shape,
            vec![21_504, 1]
        );
        assert_eq!(int8["blocks.0.mlp.fc2.weight_scale"].shape, vec![5_376, 1]);
        for protected in [
            "condition_proj.weight",
            "token_refiner.blocks.0.attn.qkv_proj.weight",
            "blocks.0.adaln_proj.linear.weight",
        ] {
            assert_eq!(int8[protected].dtype, bf16[protected].dtype);
            assert!(int8_policy
                .protected_weight_tensors
                .contains(&protected.to_owned()));
        }
        assert!(int8_policy.quantized_layers.iter().all(|layer| {
            layer
                .strip_prefix("blocks.")
                .and_then(|rest| rest.split_once('.'))
                .is_some_and(|(index, suffix)| {
                    index.parse::<usize>().is_ok_and(|index| index < 50)
                        && QUANTIZED_BLOCK_WEIGHT_SUFFIXES
                            .iter()
                            .any(|expected| suffix == expected.trim_end_matches(".weight"))
                })
        }));

        let encoded_bytes = |schema: &BTreeMap<String, ExpectedTensor>| {
            schema
                .values()
                .map(|tensor| {
                    tensor
                        .shape
                        .iter()
                        .map(|dimension| *dimension as u64)
                        .product::<u64>()
                        * dtype_size(tensor.dtype).unwrap()
                })
                .sum::<u64>()
        };
        let bf16_bytes = encoded_bytes(&bf16);
        let fp8_bytes = encoded_bytes(&fp8);
        let int8_bytes = encoded_bytes(&int8);
        assert_eq!(bf16_bytes - fp8_bytes, 19_267_582_400);
        assert_eq!(bf16_bytes - int8_bytes, 19_255_398_400);
        assert_eq!(
            (bf16_bytes - fp8_bytes)
                - (H3ComfyPublishedArtifact::Fl2VaPrunedBf16.file_bytes()
                    - H3ComfyPublishedArtifact::Fl2VaPrunedFp8Scaled.file_bytes()),
            63_832
        );
        assert_eq!(
            (bf16_bytes - int8_bytes)
                - (H3ComfyPublishedArtifact::Fl2VaPrunedBf16.file_bytes()
                    - H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot.file_bytes()),
            53_840
        );
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
        assert!(cuda
            .requirements
            .contains(&H3ComfyRuntimeRequirement::QuantizedBackendQualification));
        assert!(cuda
            .requirements
            .contains(&H3ComfyRuntimeRequirement::PrunedAdaLnNumericalQualification));
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

        let (mut header, data) = fixture_header(&config, mode, H3ComfyPrunedFormat::Fp8Scaled);
        let metadata = header["__metadata__"].as_object_mut().unwrap();
        let mut quant: Value =
            serde_json::from_str(metadata["_quantization_metadata"].as_str().unwrap()).unwrap();
        quant["format_version"] = Value::String("2.0".into());
        metadata.insert(
            "_quantization_metadata".into(),
            Value::String(quant.to_string()),
        );
        assert_eq!(
            inspect_fixture(&header, &data, H3ComfyPrunedFormat::Fp8Scaled)
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
    fn published_int8_sidecar_payloads_are_exact_and_cancellable() {
        let config = H3TransformerConfig::default();
        let mode = H3AdaLnMode::Curve {
            grid: PUBLISHED_INT8_CURVE_GRID,
            basis_dim: PUBLISHED_INT8_CURVE_BASIS,
        };
        let quantization = published_int8_quantization(&config, mode).unwrap();
        let (_, policy) = expected_schema(
            &config,
            mode,
            H3ComfyPrunedFormat::Int8ConvRot,
            Some(&quantization),
        )
        .unwrap();
        assert_eq!(
            policy.quantized_layers.len(),
            PUBLISHED_INT8_COMFY_QUANT_COUNT
        );

        let mut tensors = BTreeMap::new();
        let mut data = Vec::new();
        for layer in &policy.quantized_layers {
            let start = data.len() as u64;
            data.extend_from_slice(PUBLISHED_INT8_COMFY_QUANT_PAYLOAD);
            tensors.insert(
                format!("{layer}.comfy_quant"),
                HeaderTensor {
                    dtype: "U8".into(),
                    shape: vec![PUBLISHED_INT8_COMFY_QUANT_BYTES],
                    data_offsets: [start, data.len() as u64],
                },
            );
        }
        let path = std::env::temp_dir().join(format!(
            "mold-h3-comfy-sidecars-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut created = File::create(&path).unwrap();
        created.write_all(&0u64.to_le_bytes()).unwrap();
        created.write_all(&data).unwrap();
        drop(created);
        let parsed = ParsedHeader {
            metadata: BTreeMap::new(),
            tensors,
            header_len: 0,
            file_len: 8 + data.len() as u64,
            header_identity_sha256: String::new(),
        };
        let mut file = File::open(&path).unwrap();
        validate_published_int8_quantization_sidecars(
            &mut file,
            &parsed,
            &policy,
            &H3ComfyNeverCancel,
        )
        .unwrap();

        let mut mutable = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        mutable.seek(SeekFrom::Start(8 + 17)).unwrap();
        mutable.write_all(b"X").unwrap();
        mutable.sync_all().unwrap();
        drop(mutable);
        let mut file = File::open(&path).unwrap();
        assert_eq!(
            validate_published_int8_quantization_sidecars(
                &mut file,
                &parsed,
                &policy,
                &H3ComfyNeverCancel,
            )
            .unwrap_err()
            .code,
            H3ComfyCheckpointErrorCode::InvalidMetadata
        );

        let cancelled = ToggleCancellation(AtomicBool::new(true));
        let mut file = File::open(&path).unwrap();
        assert_eq!(
            validate_published_int8_quantization_sidecars(&mut file, &parsed, &policy, &cancelled,)
                .unwrap_err()
                .code,
            H3ComfyCheckpointErrorCode::Cancelled
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn published_f16_curve_storage_upcasts_only_authorized_logical_f32() {
        let name = "blocks.0.adaln_proj.linear.bias";
        let header = serde_json::json!({
            (name): {
                "dtype": "F16",
                "shape": [2],
                "data_offsets": [0, 4]
            }
        });
        let path = write_fixture(&header, &[0x00, 0x3c, 0x00, 0xc0], "f16-upcast");
        let parsed = read_safetensors_header(&path).unwrap();
        let make_source = |allowed: BTreeSet<String>| {
            let file = File::open(&path).unwrap();
            let identity = H3OpenedFileIdentity::from_metadata(&file.metadata().unwrap());
            H3ComfyOpenedSource {
                path: std::fs::canonicalize(&path).unwrap(),
                file: Mutex::new(file),
                header: parsed.clone(),
                identity,
                content_sha256: "fixture".into(),
                runtime_bytes_read: AtomicU64::new(0),
                f16_curve_projection_tensors: allowed,
                f16_curve_projection_upcast_loads: AtomicU64::new(0),
            }
        };
        let source = make_source(BTreeSet::from([name.to_owned()]));
        let tensor = source
            .load_dense_tensor(name, DType::F32, &Device::Cpu, &H3ComfyNeverCancel)
            .unwrap();
        assert_eq!(tensor.to_vec1::<f32>().unwrap(), vec![1.0, -2.0]);
        assert_eq!(
            source
                .f16_curve_projection_upcast_loads
                .load(Ordering::Relaxed),
            1
        );
        assert!(source
            .load_dense_tensor(name, DType::BF16, &Device::Cpu, &H3ComfyNeverCancel)
            .unwrap_err()
            .to_string()
            .contains("loaded as logical F32"));
        let unauthorized = make_source(BTreeSet::new());
        assert!(unauthorized
            .load_dense_tensor(name, DType::F32, &Device::Cpu, &H3ComfyNeverCancel)
            .unwrap_err()
            .to_string()
            .contains("authorized only for published curve projections"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn published_block_memory_excludes_sidecars_and_counts_logical_f32() {
        let tensors = BTreeMap::from([
            (
                "blocks.0.attn.qkv_proj.weight".into(),
                HeaderTensor {
                    dtype: "I8".into(),
                    shape: vec![1, 4],
                    data_offsets: [0, 4],
                },
            ),
            (
                "blocks.0.attn.qkv_proj.weight_scale".into(),
                HeaderTensor {
                    dtype: "F32".into(),
                    shape: vec![1, 1],
                    data_offsets: [4, 8],
                },
            ),
            (
                "blocks.0.attn.qkv_proj.comfy_quant".into(),
                HeaderTensor {
                    dtype: "U8".into(),
                    shape: vec![PUBLISHED_INT8_COMFY_QUANT_BYTES],
                    data_offsets: [8, 80],
                },
            ),
            (
                "blocks.0.adaln_proj.linear.weight".into(),
                HeaderTensor {
                    dtype: "F16".into(),
                    shape: vec![2],
                    data_offsets: [80, 84],
                },
            ),
            (
                "blocks.0.norm1.weight".into(),
                HeaderTensor {
                    dtype: "F32".into(),
                    shape: vec![1],
                    data_offsets: [84, 88],
                },
            ),
        ]);
        let parsed = ParsedHeader {
            metadata: BTreeMap::new(),
            tensors,
            header_len: 0,
            file_len: 96,
            header_identity_sha256: String::new(),
        };
        let allowed = BTreeSet::from(["blocks.0.adaln_proj.linear.weight".to_owned()]);
        let memory = calculate_block_memory(&parsed, &allowed, 0, 1).unwrap();
        assert_eq!(memory.encoded_host_bytes, 8);
        assert_eq!(memory.max_host_read_staging_bytes, 4);
        assert_eq!(memory.max_device_weight_staging_bytes, 16);
        assert_eq!(memory.protected_device_bytes, 12);
    }

    #[test]
    fn opened_memory_evidence_freezes_all_fifty_block_records() {
        let mut tensors = BTreeMap::from([(
            "x_embedder.weight".into(),
            HeaderTensor {
                dtype: "F32".into(),
                shape: vec![1],
                data_offsets: [0, 4],
            },
        )]);
        for index in 0..50 {
            let start = 4 + index as u64 * 12;
            tensors.insert(
                format!("blocks.{index}.attn.qkv_proj.weight"),
                HeaderTensor {
                    dtype: "I8".into(),
                    shape: vec![1, 4],
                    data_offsets: [start, start + 4],
                },
            );
            tensors.insert(
                format!("blocks.{index}.attn.qkv_proj.weight_scale"),
                HeaderTensor {
                    dtype: "F32".into(),
                    shape: vec![1, 1],
                    data_offsets: [start + 4, start + 8],
                },
            );
            tensors.insert(
                format!("blocks.{index}.norm1.weight"),
                HeaderTensor {
                    dtype: "F32".into(),
                    shape: vec![1],
                    data_offsets: [start + 8, start + 12],
                },
            );
        }
        let parsed = ParsedHeader {
            metadata: BTreeMap::new(),
            tensors,
            header_len: 128,
            file_len: 8 + 128 + 4 + 50 * 12,
            header_identity_sha256: "a".repeat(64),
        };
        let digests = (0..50)
            .map(|index| format!("{index:064x}"))
            .collect::<Vec<_>>();
        let evidence =
            calculate_opened_memory_evidence(&parsed, &BTreeSet::new(), 50, &digests).unwrap();
        assert_eq!(evidence.blocks.len(), 50);
        assert_eq!(evidence.blocks[49].index, 49);
        assert_eq!(evidence.blocks[49].content_sha256, digests[49]);
        assert_eq!(evidence.fixed_encoded_host_bytes, 4);
        assert_eq!(evidence.fixed_protected_device_bytes, 4);
        assert_eq!(evidence.identity_sha256.len(), 64);
    }

    #[test]
    fn opened_int8_runtime_streams_exactly_one_cpu_packed_block() -> candle::Result<()> {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<H3ComfyOpenedInt8Checkpoint>();
        assert_send_sync::<H3ComfyInt8BlockLoader>();
        assert_send_sync::<H3ComfyInt8BlockMemory>();
        let (config, header, data) = runtime_fixture();
        let path = write_fixture(&header, &data, "opened-runtime");
        let opened = open_h3_comfy_int8_checkpoint(
            &path,
            config,
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
        assert_eq!(opened.bytes_read_after_verification(), 0);
        assert_eq!(opened.checkpoint_identity_sha256().len(), 64);
        let opened_memory = opened.memory_evidence().clone();
        assert_eq!(
            opened_memory.verified_file_bytes,
            std::fs::metadata(&path).unwrap().len()
        );
        assert_eq!(opened_memory.blocks.len(), 2);
        assert_eq!(opened_memory.identity_sha256.len(), 64);
        assert!(opened_memory.fixed_encoded_host_bytes > 0);
        assert!(opened_memory.fixed_protected_device_bytes > 0);
        assert!(opened_memory.fixed_max_host_read_staging_bytes > 0);
        assert!(opened_memory.fixed_max_device_weight_staging_bytes > 0);
        for (index, block) in opened_memory.blocks.iter().enumerate() {
            assert_eq!(usize::from(block.index), index);
            assert_eq!(block.content_sha256.len(), 64);
        }
        let expected_identity = opened.checkpoint_identity_sha256().to_owned();
        let (transformer, mut loader) = opened.load(&Device::Cpu)?;
        assert!(loader.is_exact_pair_for(&transformer));
        assert_eq!(
            transformer.checkpoint_identity_sha256(),
            Some(expected_identity.as_str())
        );
        assert_eq!(loader.block_count(), 2);
        assert_eq!(loader.live_block_count(), 0);
        assert!(
            loader.bytes_read() > 0,
            "resident weights must be file-backed"
        );
        for block in &opened_memory.blocks {
            assert_eq!(loader.block_memory(usize::from(block.index))?, block.memory);
        }
        let memory = loader.block_memory(0)?;
        assert!(memory.encoded_host_bytes > 0);
        assert!(memory.max_host_read_staging_bytes > 0);
        assert_eq!(memory.max_device_weight_staging_bytes, 256 * 256 * 4);
        assert!(memory.protected_device_bytes > 0);

        let block0 = loader.load_block(0)?;
        assert_eq!(block0.index(), 0);
        assert_eq!(loader.live_block_count(), 1);
        assert!(loader.load_block(1).is_err());
        drop(block0);
        assert_eq!(loader.live_block_count(), 0);
        let block1 = loader.load_block(1)?;
        assert_eq!(block1.index(), 1);
        drop(block1);
        assert_eq!(loader.live_block_count(), 0);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn independently_opened_runtime_halves_are_not_an_exact_pair() -> candle::Result<()> {
        let (config, header, data) = runtime_fixture();
        let path = write_fixture(&header, &data, "opened-pair-identity");
        let first = open_h3_comfy_int8_checkpoint(
            &path,
            config.clone(),
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let second = open_h3_comfy_int8_checkpoint(
            &path,
            config,
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let (first_transformer, first_loader) = first.load(&Device::Cpu)?;
        let (second_transformer, second_loader) = second.load(&Device::Cpu)?;
        assert_eq!(
            first_loader.checkpoint_identity_sha256(),
            second_loader.checkpoint_identity_sha256()
        );
        assert!(first_loader.is_exact_pair_for(&first_transformer));
        assert!(second_loader.is_exact_pair_for(&second_transformer));
        assert!(!first_loader.is_exact_pair_for(&second_transformer));
        assert!(!second_loader.is_exact_pair_for(&first_transformer));
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn opened_identity_freezes_task_header_content_and_quantization_policy() {
        let (config, header, data) = runtime_fixture();
        let path = write_fixture(&header, &data, "opened-task-identity");
        let fl = open_h3_comfy_int8_checkpoint(
            &path,
            config.clone(),
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .unwrap();
        let same = open_h3_comfy_int8_checkpoint(
            &path,
            config.clone(),
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .unwrap();
        let reference = open_h3_comfy_int8_checkpoint(
            &path,
            config,
            H3TransformerTask::Ref2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .unwrap();
        assert_eq!(
            fl.checkpoint_identity_sha256(),
            same.checkpoint_identity_sha256()
        );
        assert_ne!(
            fl.checkpoint_identity_sha256(),
            reference.checkpoint_identity_sha256()
        );
        assert_eq!(
            fl.candidate().header_identity_sha256,
            reference.candidate().header_identity_sha256
        );
        assert_eq!(
            fl.candidate().strategy.quantization_policy.policy_sha256,
            reference
                .candidate()
                .strategy
                .quantization_policy
                .policy_sha256
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn opened_int8_runtime_rejects_digest_drift_cancellation_and_post_open_mutation() {
        let (config, header, data) = runtime_fixture();
        let path = write_fixture(&header, &data, "opened-authority");
        let bytes = std::fs::metadata(&path).unwrap().len();
        let wrong_digest = "00".repeat(32);
        let digest_error = open_h3_comfy_int8_checkpoint(
            &path,
            config.clone(),
            H3TransformerTask::T2VaFl2Va,
            None,
            Some((bytes, &wrong_digest)),
            &H3ComfyNeverCancel,
        )
        .unwrap_err();
        assert_eq!(
            digest_error.code,
            H3ComfyCheckpointErrorCode::ContentDigestMismatch
        );

        let cancelled = ToggleCancellation(AtomicBool::new(true));
        let cancel_error = open_h3_comfy_int8_checkpoint(
            &path,
            config.clone(),
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &cancelled,
        )
        .unwrap_err();
        assert_eq!(cancel_error.code, H3ComfyCheckpointErrorCode::Cancelled);

        let opened = open_h3_comfy_int8_checkpoint(
            &path,
            config,
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .unwrap();
        let mut mutable = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        mutable.seek(SeekFrom::End(-1)).unwrap();
        mutable.write_all(&[1]).unwrap();
        mutable.sync_all().unwrap();
        let error = opened.load(&Device::Cpu).unwrap_err().to_string();
        assert!(error.contains("changed after full verification"), "{error}");
        let _ = std::fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn opened_int8_runtime_rejects_group_writable_source() {
        let (config, header, data) = runtime_fixture();
        let path = write_fixture(&header, &data, "opened-permissions");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        let error = open_h3_comfy_int8_checkpoint(
            &path,
            config,
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .unwrap_err();
        assert_eq!(error.code, H3ComfyCheckpointErrorCode::FileIdentityChanged);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn runtime_cancellation_is_polled_before_block_tensor_reads() -> candle::Result<()> {
        let (config, header, data) = runtime_fixture();
        let path = write_fixture(&header, &data, "block-cancellation");
        let opened = open_h3_comfy_int8_checkpoint(
            &path,
            config.clone(),
            H3TransformerTask::T2VaFl2Va,
            None,
            None,
            &H3ComfyNeverCancel,
        )
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let cancellation = Arc::new(ToggleCancellation::default());
        let attention = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::Cpu,
            H3AttentionModelContract {
                heads: config.num_attention_heads,
                head_dim: config.attention_head_dim,
                dtype: H3AttentionDType::Bf16,
            },
        )
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let (_transformer, mut loader) = opened.load_with_attention_and_cancellation(
            &Device::Cpu,
            attention,
            cancellation.clone(),
        )?;
        cancellation.0.store(true, AtomicOrdering::Relaxed);
        let before = loader.bytes_read();
        let error = loader.load_block(0).unwrap_err().to_string();
        assert!(error.contains("cancelled"), "{error}");
        assert_eq!(loader.bytes_read(), before);
        assert_eq!(loader.live_block_count(), 0);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn duplicate_json_keys_are_rejected_recursively() {
        assert!(parse_strict_json(br#"{"a":1,"a":2}"#, "duplicate root").is_err());
        assert!(parse_strict_json(br#"{"a":{"b":1,"b":2}}"#, "duplicate nested").is_err());
    }

    #[test]
    fn tensor_count_bound_applies_without_metadata() {
        let mut header = Map::new();
        for index in 0..=MAX_TENSORS {
            header.insert(
                format!("empty.{index}"),
                serde_json::json!({"dtype":"U8","shape":[0],"data_offsets":[0,0]}),
            );
        }
        let path = write_fixture(&Value::Object(header), &[], "tensor-bound");
        assert_eq!(
            read_safetensors_header(&path).unwrap_err().code,
            H3ComfyCheckpointErrorCode::InvalidHeader
        );
        let _ = std::fs::remove_file(path);
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
