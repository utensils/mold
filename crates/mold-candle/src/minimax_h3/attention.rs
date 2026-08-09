//! Lossless, fail-closed attention authority for MiniMax H3.
//!
//! H3 packs every text, reference, generated-video, and generated-audio row
//! into one full non-causal self-attention document. Released shapes make an
//! `N x N` score tensor non-viable. This module therefore keeps dense F32
//! attention behind a small synthetic bound and gives the optional Candle
//! FlashAttention v2 primitive a separate, source- and hardware-qualified
//! authority.
//!
//! The optional kernel is deliberately not a shipping claim. The dedicated
//! `h3-flash-attn-rc` feature is reachable only from a synthetic qualification
//! binary; Mold's release, Nix, Docker, desktop, and AUR builds compile neither
//! it nor the global `flash-attn` feature. The runtime authority below records
//! that state and always fails its release-activation gate even in a release-
//! candidate build.

use std::error::Error as StdError;
use std::fmt;

use candle::{DType, Device, Tensor, D};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub const H3_DENSE_SYNTHETIC_MAX_SCORE_ELEMENTS: u64 = 4 * 1024 * 1024;
pub const H3_FLASH_ATTN_PACKAGE_VERSION: &str = "0.11.0";
pub const H3_FLASH_ATTN_CRATE_SHA256: &str =
    "e5f1e2f29f5123d7a627171209fdaeb4ee91d076aefeac588922aa9842e30c2c";
pub const H3_FLASH_ATTN_QUALIFIED_COMPUTE_CAPABILITY: (u16, u16) = (8, 9);
pub const H3_ATTENTION_RC_SCHEMA: &str = "mold.minimax-h3.attention-rc.v1";
pub const H3_ATTENTION_RC_FEATURE: &str = "h3-flash-attn-rc";
pub const H3_ATTENTION_RC_CLAIM_MARKER: &str = "mold.minimax-h3.attention-rc.kernel-compiled.v1";
pub const H3_ATTENTION_RELEASE_PROVENANCE_PREFIX: &str =
    "mold.minimax-h3.attention-release-provenance.v2:";

#[cfg(all(not(feature = "h3-flash-attn-rc"), not(feature = "flash-attn")))]
pub const H3_ATTENTION_RELEASE_PROVENANCE_MARKER: &str =
    "mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted";
#[cfg(all(feature = "h3-flash-attn-rc", not(feature = "flash-attn")))]
pub const H3_ATTENTION_RELEASE_PROVENANCE_MARKER: &str =
    "mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=omitted";
#[cfg(all(not(feature = "h3-flash-attn-rc"), feature = "flash-attn"))]
pub const H3_ATTENTION_RELEASE_PROVENANCE_MARKER: &str =
    "mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=compiled";
#[cfg(all(feature = "h3-flash-attn-rc", feature = "flash-attn"))]
pub const H3_ATTENTION_RELEASE_PROVENANCE_MARKER: &str =
    "mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=compiled";

const H3_RELEASE_HEADS: usize = 56;
const H3_RELEASE_HEAD_DIM: usize = 128;
const PLAN_SCHEMA_VERSION: u32 = 1;

/// Positive, compile-time provenance for release artifact inspection.
///
/// Exactly one marker is compiled from the same Cargo feature gates that make
/// the H3 DiT and global FlashAttention code reachable. Published binaries
/// deliberately retain this value and their verifier requires the omitted /
/// omitted state; absence of evidence therefore fails closed.
pub const fn h3_attention_release_provenance_marker() -> &'static str {
    H3_ATTENTION_RELEASE_PROVENANCE_MARKER
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionDType {
    Bf16,
    F16,
    F32,
}

impl H3AttentionDType {
    const fn stable_id(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    const fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    pub(crate) fn from_candle(dtype: DType) -> InspectionResult<Self> {
        match dtype {
            DType::BF16 => Ok(Self::Bf16),
            DType::F16 => Ok(Self::F16),
            DType::F32 => Ok(Self::F32),
            other => Err(failure(
                H3AttentionErrorCode::UnsupportedDType,
                format!("MiniMax H3 attention does not support {other:?}"),
                vec![H3AttentionRequirement::Bf16OrBoundedSyntheticF32],
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionDevice {
    Cpu,
    Metal,
    Cuda {
        compute_capability: Option<(u16, u16)>,
    },
}

impl H3AttentionDevice {
    pub fn from_candle(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Metal(_) => Self::Metal,
            Device::Cuda(cuda) => {
                #[cfg(feature = "cuda")]
                let compute_capability = cuda
                    .cuda_stream()
                    .context()
                    .compute_capability()
                    .ok()
                    .and_then(|(major, minor)| {
                        Some((u16::try_from(major).ok()?, u16::try_from(minor).ok()?))
                    });
                #[cfg(not(feature = "cuda"))]
                let compute_capability = {
                    let _ = cuda;
                    None
                };
                Self::Cuda { compute_capability }
            }
        }
    }

    fn stable_id(self) -> String {
        match self {
            Self::Cpu => "cpu".into(),
            Self::Metal => "metal".into(),
            Self::Cuda {
                compute_capability: Some((major, minor)),
            } => format!("cuda-sm{major}{minor}"),
            Self::Cuda {
                compute_capability: None,
            } => "cuda-unknown".into(),
        }
    }

    fn matches_candle(self, device: &Device) -> bool {
        match (self, device) {
            (Self::Cpu, Device::Cpu) | (Self::Metal, Device::Metal(_)) => true,
            (
                Self::Cuda {
                    compute_capability: expected,
                },
                Device::Cuda(_),
            ) => {
                expected.is_none()
                    || Self::from_candle(device)
                        == (Self::Cuda {
                            compute_capability: expected,
                        })
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionScope {
    TokenRefiner,
    PackedTransformer,
}

impl H3AttentionScope {
    const fn stable_id(self) -> &'static str {
        match self {
            Self::TokenRefiner => "token-refiner",
            Self::PackedTransformer => "packed-transformer",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionMask {
    FullNonCausalPacked,
}

impl H3AttentionMask {
    const fn stable_id(self) -> &'static str {
        match self {
            Self::FullNonCausalPacked => "full-noncausal-packed",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionBackend {
    BoundedDenseMath,
    FlashAttentionV2,
}

impl H3AttentionBackend {
    const fn stable_id(self) -> &'static str {
        match self {
            Self::BoundedDenseMath => "bounded-dense-math",
            Self::FlashAttentionV2 => "flash-attention-v2",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionKernel {
    CandleDenseF32V011,
    CandleFlashFwdHdim128Bf16Sm80V011,
}

impl H3AttentionKernel {
    const fn stable_id(self) -> &'static str {
        match self {
            Self::CandleDenseF32V011 => "candle-0.11.dense-f32",
            Self::CandleFlashFwdHdim128Bf16Sm80V011 => {
                "candle-flash-attn-0.11.0.flash-fwd-hdim128-bf16-sm80"
            }
        }
    }

    /// Canonical non-executable kernel identifier used by scheduler/factory
    /// evidence. Runtime identity still binds the complete typed tuple.
    pub const fn identity(self) -> &'static str {
        self.stable_id()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionActivation {
    SyntheticCorrectnessOnly,
    ReleaseCandidateQualificationOnly,
}

impl H3AttentionActivation {
    const fn stable_id(self) -> &'static str {
        match self {
            Self::SyntheticCorrectnessOnly => "synthetic-correctness-only",
            Self::ReleaseCandidateQualificationOnly => "release-candidate-qualification-only",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3AttentionModelContract {
    pub heads: usize,
    pub head_dim: usize,
    pub dtype: H3AttentionDType,
}

impl H3AttentionModelContract {
    pub const fn released_bf16() -> Self {
        Self {
            heads: H3_RELEASE_HEADS,
            head_dim: H3_RELEASE_HEAD_DIM,
            dtype: H3AttentionDType::Bf16,
        }
    }
}

/// Compile-time identity of the isolated H3 attention release candidate.
///
/// A normal Mold build serializes this as `kernel_compiled = false` and has no
/// claim marker. The marker exists only in an opt-in qualification binary and
/// is forbidden by the release artifact verifier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct H3AttentionReleaseCandidateBuild {
    pub schema: String,
    pub identity_sha256: String,
    pub cargo_feature: String,
    pub claim_marker: Option<String>,
    pub kernel_compiled: bool,
    pub compiled_compute_capability: Option<(u16, u16)>,
    pub package_version: String,
    pub package_sha256: String,
    pub kernel: H3AttentionKernel,
    pub mask: H3AttentionMask,
    pub model_contract: H3AttentionModelContract,
}

impl H3AttentionReleaseCandidateBuild {
    /// Inspect this exact compilation. `CUDA_COMPUTE_CAP` is captured by
    /// rustc, not inferred from the runtime GPU, so a cached or cross-targeted
    /// kernel cannot silently claim the device it happens to run on.
    pub fn current() -> Self {
        Self::from_build_inputs(
            cfg!(feature = "h3-flash-attn-rc"),
            option_env!("CUDA_COMPUTE_CAP"),
        )
    }

    fn from_build_inputs(kernel_compiled: bool, cuda_compute_cap: Option<&str>) -> Self {
        let compiled_compute_capability = cuda_compute_cap.and_then(parse_compute_capability);
        let claim_marker = kernel_compiled.then(|| H3_ATTENTION_RC_CLAIM_MARKER.to_string());
        let mut build = Self {
            schema: H3_ATTENTION_RC_SCHEMA.to_string(),
            identity_sha256: String::new(),
            cargo_feature: H3_ATTENTION_RC_FEATURE.to_string(),
            claim_marker,
            kernel_compiled,
            compiled_compute_capability,
            package_version: H3_FLASH_ATTN_PACKAGE_VERSION.to_string(),
            package_sha256: H3_FLASH_ATTN_CRATE_SHA256.to_string(),
            kernel: H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
            mask: H3AttentionMask::FullNonCausalPacked,
            model_contract: H3AttentionModelContract::released_bf16(),
        };
        build.identity_sha256 = release_candidate_build_identity(&build);
        build
    }

    /// Turn a compiled candidate into the exact runtime authority used by the
    /// synthetic probe. Missing kernels, absent/unsupported build targets, and
    /// build/device mismatches are all rejected before tensor allocation.
    pub fn qualify(
        &self,
        device: H3AttentionDevice,
    ) -> InspectionResult<H3AttentionRuntimeAuthority> {
        self.validate_identity()?;
        if !self.kernel_compiled {
            return Err(failure(
                H3AttentionErrorCode::KernelNotCompiled,
                "MiniMax H3 release-candidate FlashAttention kernel is not compiled",
                vec![H3AttentionRequirement::CompiledH3FlashAttentionV2],
            ));
        }
        let compiled_target = self.compiled_compute_capability.ok_or_else(|| {
            failure(
                H3AttentionErrorCode::UnqualifiedArchitecture,
                "MiniMax H3 release-candidate kernel has no parseable CUDA_COMPUTE_CAP build target",
                vec![H3AttentionRequirement::QualifiedCudaSm89],
            )
        })?;
        if compiled_target != H3_FLASH_ATTN_QUALIFIED_COMPUTE_CAPABILITY {
            return Err(failure(
                H3AttentionErrorCode::UnqualifiedArchitecture,
                format!(
                    "MiniMax H3 release-candidate kernel was compiled for SM{}{}, not qualified SM89",
                    compiled_target.0, compiled_target.1
                ),
                vec![H3AttentionRequirement::QualifiedCudaSm89],
            ));
        }
        if device
            != (H3AttentionDevice::Cuda {
                compute_capability: Some(compiled_target),
            })
        {
            return Err(failure(
                H3AttentionErrorCode::UnqualifiedArchitecture,
                format!(
                    "MiniMax H3 release-candidate kernel target SM{}{} does not match {}",
                    compiled_target.0,
                    compiled_target.1,
                    device.stable_id()
                ),
                vec![H3AttentionRequirement::QualifiedCudaSm89],
            ));
        }
        qualify_flash_attention_v2_with_availability(device, self.model_contract, true)
    }

    fn validate_identity(&self) -> InspectionResult<()> {
        if self.schema != H3_ATTENTION_RC_SCHEMA
            || self.cargo_feature != H3_ATTENTION_RC_FEATURE
            || self.package_version != H3_FLASH_ATTN_PACKAGE_VERSION
            || self.package_sha256 != H3_FLASH_ATTN_CRATE_SHA256
            || self.kernel != H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011
            || self.mask != H3AttentionMask::FullNonCausalPacked
            || self.model_contract != H3AttentionModelContract::released_bf16()
            || self.claim_marker.as_deref()
                != self.kernel_compiled.then_some(H3_ATTENTION_RC_CLAIM_MARKER)
            || self.identity_sha256 != release_candidate_build_identity(self)
        {
            return Err(failure(
                H3AttentionErrorCode::FrozenIdentityMismatch,
                "MiniMax H3 attention release-candidate build identity is inconsistent",
                vec![H3AttentionRequirement::UntamperedFrozenPlan],
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3AttentionWorkspace {
    pub input_output_tensor_bytes: u64,
    pub qkv_compute_bytes: u64,
    pub score_matrix_bytes: u64,
    pub probability_matrix_bytes: u64,
    pub output_compute_bytes: u64,
    pub softmax_lse_bytes: u64,
    pub peak_auxiliary_bytes_upper_bound: u64,
}

/// Prompt/media-free evidence for one synchronized attention dispatch.
///
/// Qualification explicitly synchronizes the device before and after the
/// kernel so `elapsed_micros` measures execution rather than only CUDA launch
/// latency. The normal DiT path remains asynchronous and allocation-free.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct H3AttentionTelemetry {
    pub schema: String,
    pub plan_identity_sha256: String,
    pub backend: H3AttentionBackend,
    pub kernel: H3AttentionKernel,
    pub activation: H3AttentionActivation,
    pub device: H3AttentionDevice,
    pub scope: H3AttentionScope,
    pub mask: H3AttentionMask,
    pub dtype: H3AttentionDType,
    pub batch_size: usize,
    pub sequence_rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub workspace: H3AttentionWorkspace,
    pub elapsed_micros: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3FrozenAttentionPlan {
    schema_version: u32,
    identity_sha256: String,
    scope: H3AttentionScope,
    backend: H3AttentionBackend,
    kernel: H3AttentionKernel,
    activation: H3AttentionActivation,
    device: H3AttentionDevice,
    dtype: H3AttentionDType,
    mask: H3AttentionMask,
    batch_size: usize,
    query_rows: usize,
    key_value_rows: usize,
    heads: usize,
    head_dim: usize,
    softmax_scale_f32_bits: u32,
    workspace: H3AttentionWorkspace,
}

impl H3FrozenAttentionPlan {
    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub const fn scope(&self) -> H3AttentionScope {
        self.scope
    }

    pub const fn backend(&self) -> H3AttentionBackend {
        self.backend
    }

    pub const fn kernel(&self) -> H3AttentionKernel {
        self.kernel
    }

    pub const fn activation(&self) -> H3AttentionActivation {
        self.activation
    }

    pub const fn device(&self) -> H3AttentionDevice {
        self.device
    }

    pub const fn dtype(&self) -> H3AttentionDType {
        self.dtype
    }

    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub const fn rows(&self) -> usize {
        self.query_rows
    }

    pub const fn heads(&self) -> usize {
        self.heads
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub const fn workspace(&self) -> &H3AttentionWorkspace {
        &self.workspace
    }

    fn validate_identity(&self) -> InspectionResult<()> {
        let expected = plan_identity(self);
        if self.schema_version != PLAN_SCHEMA_VERSION || self.identity_sha256 != expected {
            return Err(failure(
                H3AttentionErrorCode::FrozenIdentityMismatch,
                "MiniMax H3 attention plan identity does not match its frozen fields",
                vec![H3AttentionRequirement::UntamperedFrozenPlan],
            ));
        }
        if self.batch_size == 0
            || self.query_rows == 0
            || self.query_rows != self.key_value_rows
            || self.mask != H3AttentionMask::FullNonCausalPacked
        {
            return Err(failure(
                H3AttentionErrorCode::UnsupportedSemantics,
                "MiniMax H3 frozen attention must be non-empty full packed self-attention",
                vec![H3AttentionRequirement::FullNonCausalPackedSelfAttention],
            ));
        }
        let contract = H3AttentionModelContract {
            heads: self.heads,
            head_dim: self.head_dim,
            dtype: self.dtype,
        };
        validate_backend_contract(
            self.backend,
            self.kernel,
            self.activation,
            self.device,
            contract,
        )?;
        let expected_scale = (1.0f32 / (self.head_dim as f32).sqrt()).to_bits();
        if self.softmax_scale_f32_bits != expected_scale {
            return Err(failure(
                H3AttentionErrorCode::RuntimePlanMismatch,
                "MiniMax H3 frozen softmax scale does not match its head dimension",
                vec![H3AttentionRequirement::UntamperedFrozenPlan],
            ));
        }
        let score_elements = checked_product(
            "attention score elements",
            &[
                self.batch_size,
                self.heads,
                self.query_rows,
                self.key_value_rows,
            ],
        )?;
        if self.backend == H3AttentionBackend::BoundedDenseMath
            && score_elements > H3_DENSE_SYNTHETIC_MAX_SCORE_ELEMENTS
        {
            return Err(failure(
                H3AttentionErrorCode::DenseMathLimitExceeded,
                "MiniMax H3 frozen dense attention exceeds the synthetic score bound",
                vec![H3AttentionRequirement::CompiledH3FlashAttentionV2],
            ));
        }
        validate_flash_launch_shape(
            self.backend,
            self.batch_size,
            self.query_rows,
            self.heads,
            self.head_dim,
        )?;
        let expected_workspace = workspace_for(
            self.backend,
            self.dtype,
            self.batch_size,
            self.heads,
            self.query_rows,
            self.head_dim,
        )?;
        if self.workspace != expected_workspace {
            return Err(failure(
                H3AttentionErrorCode::RuntimePlanMismatch,
                "MiniMax H3 frozen workspace does not match its backend and shape",
                vec![H3AttentionRequirement::UntamperedFrozenPlan],
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3FrozenAttentionExecution {
    pub token_refiner: H3FrozenAttentionPlan,
    pub packed_transformer: H3FrozenAttentionPlan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3AttentionRuntimeAuthority {
    identity_sha256: String,
    backend: H3AttentionBackend,
    kernel: H3AttentionKernel,
    activation: H3AttentionActivation,
    device: H3AttentionDevice,
    contract: H3AttentionModelContract,
}

impl H3AttentionRuntimeAuthority {
    /// Compute the canonical identity for a typed H3 attention runtime tuple
    /// without constructing executable authority.
    ///
    /// Scheduler/factory admission uses this to bind the exact backend,
    /// kernel, activation, device, and released model contract independently
    /// from qualification evidence. Invalid or unsupported tuples fail before
    /// an identity is returned.
    pub fn expected_identity_for(
        backend: H3AttentionBackend,
        kernel: H3AttentionKernel,
        activation: H3AttentionActivation,
        device: H3AttentionDevice,
        contract: H3AttentionModelContract,
    ) -> InspectionResult<String> {
        validate_backend_contract(backend, kernel, activation, device, contract)?;
        Ok(runtime_identity_fields(
            backend, kernel, activation, device, contract,
        ))
    }

    pub fn bounded_synthetic_math(
        device: H3AttentionDevice,
        contract: H3AttentionModelContract,
    ) -> InspectionResult<Self> {
        validate_model_contract_shape(contract)?;
        if !matches!(
            contract.dtype,
            H3AttentionDType::Bf16 | H3AttentionDType::F32
        ) {
            return Err(failure(
                H3AttentionErrorCode::UnsupportedDType,
                "MiniMax H3 bounded math accepts BF16 or synthetic F32 inputs",
                vec![H3AttentionRequirement::Bf16OrBoundedSyntheticF32],
            ));
        }
        Ok(Self::new(
            H3AttentionBackend::BoundedDenseMath,
            H3AttentionKernel::CandleDenseF32V011,
            H3AttentionActivation::SyntheticCorrectnessOnly,
            device,
            contract,
        ))
    }

    /// Qualify the only currently measured H3 FlashAttention primitive.
    ///
    /// This is deliberately exact: BF16, 56 heads, head dimension 128, and
    /// SM89. Other Candle-supported shapes or architectures are not H3
    /// qualification evidence and are rejected until separately measured.
    pub fn qualify_flash_attention_v2(
        device: H3AttentionDevice,
        contract: H3AttentionModelContract,
    ) -> InspectionResult<Self> {
        if contract != H3AttentionModelContract::released_bf16() {
            return qualify_flash_attention_v2_with_availability(device, contract, true);
        }
        H3AttentionReleaseCandidateBuild::current().qualify(device)
    }

    fn new(
        backend: H3AttentionBackend,
        kernel: H3AttentionKernel,
        activation: H3AttentionActivation,
        device: H3AttentionDevice,
        contract: H3AttentionModelContract,
    ) -> Self {
        let mut authority = Self {
            identity_sha256: String::new(),
            backend,
            kernel,
            activation,
            device,
            contract,
        };
        authority.identity_sha256 = runtime_identity(&authority);
        authority
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub const fn backend(&self) -> H3AttentionBackend {
        self.backend
    }

    pub const fn kernel(&self) -> H3AttentionKernel {
        self.kernel
    }

    pub const fn activation(&self) -> H3AttentionActivation {
        self.activation
    }

    pub const fn device(&self) -> H3AttentionDevice {
        self.device
    }

    pub const fn contract(&self) -> H3AttentionModelContract {
        self.contract
    }

    pub fn verify_model(
        &self,
        contract: H3AttentionModelContract,
        device: &Device,
    ) -> InspectionResult<()> {
        self.validate_identity()?;
        if self.contract != contract {
            return Err(failure(
                H3AttentionErrorCode::RuntimePlanMismatch,
                format!(
                    "MiniMax H3 attention authority {:?} does not match model {:?}",
                    self.contract, contract
                ),
                vec![H3AttentionRequirement::ExactModelContract],
            ));
        }
        if !self.device.matches_candle(device) {
            return Err(failure(
                H3AttentionErrorCode::RuntimePlanMismatch,
                "MiniMax H3 attention authority targets a different device backend",
                vec![H3AttentionRequirement::ExactDeviceBackend],
            ));
        }
        Ok(())
    }

    pub fn freeze_execution(
        &self,
        batch_size: usize,
        token_refiner_rows: usize,
        packed_rows: usize,
    ) -> InspectionResult<H3FrozenAttentionExecution> {
        self.validate_identity()?;
        Ok(H3FrozenAttentionExecution {
            token_refiner: self.freeze_scope(
                H3AttentionScope::TokenRefiner,
                batch_size,
                token_refiner_rows,
            )?,
            packed_transformer: self.freeze_scope(
                H3AttentionScope::PackedTransformer,
                batch_size,
                packed_rows,
            )?,
        })
    }

    fn freeze_scope(
        &self,
        scope: H3AttentionScope,
        batch_size: usize,
        rows: usize,
    ) -> InspectionResult<H3FrozenAttentionPlan> {
        if batch_size == 0 || rows == 0 {
            return Err(failure(
                H3AttentionErrorCode::InvalidShape,
                "MiniMax H3 attention batch and row counts must be positive",
                vec![H3AttentionRequirement::NonEmptySelfAttention],
            ));
        }
        let score_elements = checked_product(
            "attention score elements",
            &[batch_size, self.contract.heads, rows, rows],
        )?;
        if self.backend == H3AttentionBackend::BoundedDenseMath
            && score_elements > H3_DENSE_SYNTHETIC_MAX_SCORE_ELEMENTS
        {
            return Err(failure(
                H3AttentionErrorCode::DenseMathLimitExceeded,
                format!(
                    "MiniMax H3 dense attention would materialize {score_elements} score elements; the synthetic bound is {H3_DENSE_SYNTHETIC_MAX_SCORE_ELEMENTS}"
                ),
                vec![
                    H3AttentionRequirement::CompiledH3FlashAttentionV2,
                    H3AttentionRequirement::QualifiedCudaSm89,
                    H3AttentionRequirement::ReleasedBf16HeadGeometry,
                ],
            ));
        }
        validate_flash_launch_shape(
            self.backend,
            batch_size,
            rows,
            self.contract.heads,
            self.contract.head_dim,
        )?;

        let workspace = workspace_for(
            self.backend,
            self.contract.dtype,
            batch_size,
            self.contract.heads,
            rows,
            self.contract.head_dim,
        )?;
        let mut plan = H3FrozenAttentionPlan {
            schema_version: PLAN_SCHEMA_VERSION,
            identity_sha256: String::new(),
            scope,
            backend: self.backend,
            kernel: self.kernel,
            activation: self.activation,
            device: self.device,
            dtype: self.contract.dtype,
            mask: H3AttentionMask::FullNonCausalPacked,
            batch_size,
            query_rows: rows,
            key_value_rows: rows,
            heads: self.contract.heads,
            head_dim: self.contract.head_dim,
            softmax_scale_f32_bits: (1.0f32 / (self.contract.head_dim as f32).sqrt()).to_bits(),
            workspace,
        };
        plan.identity_sha256 = plan_identity(&plan);
        Ok(plan)
    }

    pub fn require_release_activation(&self) -> Result<(), H3AttentionReleaseRejection> {
        Err(H3AttentionReleaseRejection {
            requirements: vec![
                H3AttentionReleaseRequirement::MiniMaxH3Authorization,
                H3AttentionReleaseRequirement::ReleaseArtifactIncludesKernel,
                H3AttentionReleaseRequirement::CrossBackendReproducibilityDecision,
                H3AttentionReleaseRequirement::RemainingCudaArchitectureQualification,
            ],
        })
    }

    fn validate_identity(&self) -> InspectionResult<()> {
        if self.identity_sha256 != runtime_identity(self) {
            return Err(failure(
                H3AttentionErrorCode::FrozenIdentityMismatch,
                "MiniMax H3 attention runtime identity does not match its frozen fields",
                vec![H3AttentionRequirement::UntamperedFrozenPlan],
            ));
        }
        validate_backend_contract(
            self.backend,
            self.kernel,
            self.activation,
            self.device,
            self.contract,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionRequirement {
    NonEmptySelfAttention,
    NativeBatchOne,
    Bf16OrBoundedSyntheticF32,
    ReleasedBf16HeadGeometry,
    ExactModelContract,
    ExactDeviceBackend,
    FullNonCausalPackedSelfAttention,
    QualifiedCudaSm89,
    CompiledH3FlashAttentionV2,
    FlashAttentionU32LaunchShape,
    UntamperedFrozenPlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionErrorCode {
    InvalidShape,
    UnsupportedDType,
    UnsupportedSemantics,
    UnqualifiedArchitecture,
    KernelNotCompiled,
    DenseMathLimitExceeded,
    RuntimePlanMismatch,
    FrozenIdentityMismatch,
    ArithmeticOverflow,
    KernelExecution,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3AttentionError {
    pub code: H3AttentionErrorCode,
    pub message: String,
    pub requirements: Vec<H3AttentionRequirement>,
}

impl fmt::Display for H3AttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)?;
        if !self.requirements.is_empty() {
            write!(formatter, "; requirements: {:?}", self.requirements)?;
        }
        Ok(())
    }
}

impl StdError for H3AttentionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3AttentionReleaseRequirement {
    MiniMaxH3Authorization,
    ReleaseArtifactIncludesKernel,
    CrossBackendReproducibilityDecision,
    RemainingCudaArchitectureQualification,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3AttentionReleaseRejection {
    pub requirements: Vec<H3AttentionReleaseRequirement>,
}

impl fmt::Display for H3AttentionReleaseRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MiniMax H3 attention is not release-activated; requirements: {:?}",
            self.requirements
        )
    }
}

impl StdError for H3AttentionReleaseRejection {}

type InspectionResult<T> = std::result::Result<T, H3AttentionError>;

fn failure(
    code: H3AttentionErrorCode,
    message: impl Into<String>,
    requirements: Vec<H3AttentionRequirement>,
) -> H3AttentionError {
    H3AttentionError {
        code,
        message: message.into(),
        requirements,
    }
}

fn validate_model_contract_shape(contract: H3AttentionModelContract) -> InspectionResult<()> {
    if contract.heads == 0 || contract.head_dim == 0 {
        return Err(failure(
            H3AttentionErrorCode::InvalidShape,
            "MiniMax H3 attention heads and head dimension must be positive",
            vec![H3AttentionRequirement::NonEmptySelfAttention],
        ));
    }
    Ok(())
}

fn validate_backend_contract(
    backend: H3AttentionBackend,
    kernel: H3AttentionKernel,
    activation: H3AttentionActivation,
    device: H3AttentionDevice,
    contract: H3AttentionModelContract,
) -> InspectionResult<()> {
    validate_model_contract_shape(contract)?;
    match backend {
        H3AttentionBackend::BoundedDenseMath => {
            if kernel != H3AttentionKernel::CandleDenseF32V011
                || activation != H3AttentionActivation::SyntheticCorrectnessOnly
            {
                return Err(failure(
                    H3AttentionErrorCode::RuntimePlanMismatch,
                    "MiniMax H3 bounded math has an unqualified kernel/activation tuple",
                    vec![H3AttentionRequirement::UntamperedFrozenPlan],
                ));
            }
            if !matches!(
                contract.dtype,
                H3AttentionDType::Bf16 | H3AttentionDType::F32
            ) {
                return Err(failure(
                    H3AttentionErrorCode::UnsupportedDType,
                    "MiniMax H3 bounded math accepts BF16 or synthetic F32 inputs",
                    vec![H3AttentionRequirement::Bf16OrBoundedSyntheticF32],
                ));
            }
        }
        H3AttentionBackend::FlashAttentionV2 => {
            if kernel != H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011
                || activation != H3AttentionActivation::ReleaseCandidateQualificationOnly
            {
                return Err(failure(
                    H3AttentionErrorCode::RuntimePlanMismatch,
                    "MiniMax H3 FlashAttention has an unqualified kernel/activation tuple",
                    vec![H3AttentionRequirement::UntamperedFrozenPlan],
                ));
            }
            if contract != H3AttentionModelContract::released_bf16() {
                return Err(failure(
                    if contract.dtype != H3AttentionDType::Bf16 {
                        H3AttentionErrorCode::UnsupportedDType
                    } else {
                        H3AttentionErrorCode::InvalidShape
                    },
                    format!(
                        "MiniMax H3 FlashAttention qualification requires {:?}, got {contract:?}",
                        H3AttentionModelContract::released_bf16()
                    ),
                    vec![H3AttentionRequirement::ReleasedBf16HeadGeometry],
                ));
            }
            if device
                != (H3AttentionDevice::Cuda {
                    compute_capability: Some(H3_FLASH_ATTN_QUALIFIED_COMPUTE_CAPABILITY),
                })
            {
                return Err(failure(
                    H3AttentionErrorCode::UnqualifiedArchitecture,
                    format!(
                        "MiniMax H3 FlashAttention is qualified only for CUDA SM89, got {}",
                        device.stable_id()
                    ),
                    vec![H3AttentionRequirement::QualifiedCudaSm89],
                ));
            }
        }
    }
    Ok(())
}

fn qualify_flash_attention_v2_with_availability(
    device: H3AttentionDevice,
    contract: H3AttentionModelContract,
    kernel_compiled: bool,
) -> InspectionResult<H3AttentionRuntimeAuthority> {
    validate_backend_contract(
        H3AttentionBackend::FlashAttentionV2,
        H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
        H3AttentionActivation::ReleaseCandidateQualificationOnly,
        device,
        contract,
    )?;
    if !kernel_compiled {
        return Err(failure(
            H3AttentionErrorCode::KernelNotCompiled,
            "MiniMax H3 FlashAttention kernel is not compiled into this binary",
            vec![H3AttentionRequirement::CompiledH3FlashAttentionV2],
        ));
    }
    Ok(H3AttentionRuntimeAuthority::new(
        H3AttentionBackend::FlashAttentionV2,
        H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
        H3AttentionActivation::ReleaseCandidateQualificationOnly,
        device,
        contract,
    ))
}

fn validate_flash_launch_shape(
    backend: H3AttentionBackend,
    batch_size: usize,
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> InspectionResult<()> {
    if backend != H3AttentionBackend::FlashAttentionV2 {
        return Ok(());
    }
    if batch_size != 1 {
        return Err(failure(
            H3AttentionErrorCode::InvalidShape,
            "MiniMax H3 release-candidate FlashAttention is qualified only for native batch 1",
            vec![H3AttentionRequirement::NativeBatchOne],
        ));
    }
    let batch_stride = checked_product("FlashAttention batch stride", &[rows, heads, head_dim])?;
    let total_rows = checked_product("FlashAttention total rows", &[batch_size, rows])?;
    let rounded_rows = (rows as u64)
        .checked_add(127)
        .map(|value| value / 128 * 128)
        .ok_or_else(|| {
            failure(
                H3AttentionErrorCode::ArithmeticOverflow,
                "MiniMax H3 FlashAttention rounded sequence length overflows",
                vec![H3AttentionRequirement::FlashAttentionU32LaunchShape],
            )
        })?;
    if batch_stride > u32::MAX as u64
        || total_rows > u32::MAX as u64
        || rounded_rows > u32::MAX as u64
    {
        return Err(failure(
            H3AttentionErrorCode::InvalidShape,
            "MiniMax H3 attention shape exceeds Candle FlashAttention's u32 launch contract",
            vec![H3AttentionRequirement::FlashAttentionU32LaunchShape],
        ));
    }
    Ok(())
}

fn validate_flash_tensor_strides(label: &str, strides: &[usize]) -> InspectionResult<()> {
    if strides.len() != 4 || strides.last() != Some(&1) {
        return Err(failure(
            H3AttentionErrorCode::RuntimePlanMismatch,
            format!(
                "MiniMax H3 FlashAttention {label} must be rank-4 BNHD with unit head-dimension stride, got {strides:?}"
            ),
            vec![H3AttentionRequirement::ExactModelContract],
        ));
    }
    if let Some((dimension, stride)) = strides
        .iter()
        .copied()
        .enumerate()
        .find(|(_, stride)| *stride > u32::MAX as usize)
    {
        return Err(failure(
            H3AttentionErrorCode::RuntimePlanMismatch,
            format!(
                "MiniMax H3 FlashAttention {label} stride {dimension} value {stride} exceeds Candle FlashAttention's u32 FFI contract"
            ),
            vec![H3AttentionRequirement::FlashAttentionU32LaunchShape],
        ));
    }
    Ok(())
}

fn checked_product(label: &str, values: &[usize]) -> InspectionResult<u64> {
    values.iter().try_fold(1u64, |product, value| {
        product.checked_mul(*value as u64).ok_or_else(|| {
            failure(
                H3AttentionErrorCode::ArithmeticOverflow,
                format!("MiniMax H3 {label} overflows"),
                Vec::new(),
            )
        })
    })
}

fn checked_sum(label: &str, values: &[u64]) -> InspectionResult<u64> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value).ok_or_else(|| {
            failure(
                H3AttentionErrorCode::ArithmeticOverflow,
                format!("MiniMax H3 {label} overflows"),
                Vec::new(),
            )
        })
    })
}

fn workspace_for(
    backend: H3AttentionBackend,
    dtype: H3AttentionDType,
    batch: usize,
    heads: usize,
    rows: usize,
    head_dim: usize,
) -> InspectionResult<H3AttentionWorkspace> {
    let vector_elements =
        checked_product("attention vector elements", &[batch, heads, rows, head_dim])?;
    let score_elements = checked_product("attention score elements", &[batch, heads, rows, rows])?;
    let input_output_tensor_bytes = vector_elements
        .checked_mul(dtype.byte_width())
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            failure(
                H3AttentionErrorCode::ArithmeticOverflow,
                "MiniMax H3 attention input/output byte accounting overflows",
                Vec::new(),
            )
        })?;
    match backend {
        H3AttentionBackend::BoundedDenseMath => {
            let one_vector_f32 = vector_elements.checked_mul(4).ok_or_else(|| {
                failure(
                    H3AttentionErrorCode::ArithmeticOverflow,
                    "MiniMax H3 dense vector byte accounting overflows",
                    Vec::new(),
                )
            })?;
            let qkv_compute_bytes = one_vector_f32.checked_mul(3).ok_or_else(|| {
                failure(
                    H3AttentionErrorCode::ArithmeticOverflow,
                    "MiniMax H3 dense QKV byte accounting overflows",
                    Vec::new(),
                )
            })?;
            let score_matrix_bytes = score_elements.checked_mul(4).ok_or_else(|| {
                failure(
                    H3AttentionErrorCode::ArithmeticOverflow,
                    "MiniMax H3 dense score byte accounting overflows",
                    Vec::new(),
                )
            })?;
            let probability_matrix_bytes = score_matrix_bytes;
            let output_compute_bytes = one_vector_f32;
            Ok(H3AttentionWorkspace {
                input_output_tensor_bytes,
                qkv_compute_bytes,
                score_matrix_bytes,
                probability_matrix_bytes,
                output_compute_bytes,
                softmax_lse_bytes: 0,
                peak_auxiliary_bytes_upper_bound: checked_sum(
                    "dense attention workspace",
                    &[
                        qkv_compute_bytes,
                        score_matrix_bytes,
                        probability_matrix_bytes,
                        output_compute_bytes,
                    ],
                )?,
            })
        }
        H3AttentionBackend::FlashAttentionV2 => {
            let softmax_lse_bytes =
                checked_product("FlashAttention LSE elements", &[batch, heads, rows])?
                    .checked_mul(4)
                    .ok_or_else(|| {
                        failure(
                            H3AttentionErrorCode::ArithmeticOverflow,
                            "MiniMax H3 FlashAttention LSE byte accounting overflows",
                            Vec::new(),
                        )
                    })?;
            Ok(H3AttentionWorkspace {
                input_output_tensor_bytes,
                qkv_compute_bytes: 0,
                score_matrix_bytes: 0,
                probability_matrix_bytes: 0,
                output_compute_bytes: 0,
                softmax_lse_bytes,
                peak_auxiliary_bytes_upper_bound: softmax_lse_bytes,
            })
        }
    }
}

fn parse_compute_capability(raw: &str) -> Option<(u16, u16)> {
    let raw = raw.trim();
    if raw.len() < 2 || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let (major, minor) = raw.split_at(raw.len() - 1);
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn release_candidate_build_identity(build: &H3AttentionReleaseCandidateBuild) -> String {
    let mut digest = Sha256::new();
    digest.update(b"minimax-h3-attention-release-candidate-build-v1");
    digest.update(build.schema.as_bytes());
    digest.update(build.cargo_feature.as_bytes());
    digest.update([u8::from(build.kernel_compiled)]);
    match build.compiled_compute_capability {
        Some((major, minor)) => {
            digest.update([1]);
            digest.update(major.to_le_bytes());
            digest.update(minor.to_le_bytes());
        }
        None => digest.update([0]),
    }
    if let Some(marker) = build.claim_marker.as_deref() {
        digest.update(marker.as_bytes());
    }
    digest.update(build.package_version.as_bytes());
    digest.update(build.package_sha256.as_bytes());
    digest.update(build.kernel.stable_id().as_bytes());
    digest.update(build.mask.stable_id().as_bytes());
    digest.update((build.model_contract.heads as u64).to_le_bytes());
    digest.update((build.model_contract.head_dim as u64).to_le_bytes());
    digest.update(build.model_contract.dtype.stable_id().as_bytes());
    hex_digest(digest.finalize())
}

fn runtime_identity(authority: &H3AttentionRuntimeAuthority) -> String {
    runtime_identity_fields(
        authority.backend,
        authority.kernel,
        authority.activation,
        authority.device,
        authority.contract,
    )
}

fn runtime_identity_fields(
    backend: H3AttentionBackend,
    kernel: H3AttentionKernel,
    activation: H3AttentionActivation,
    device: H3AttentionDevice,
    contract: H3AttentionModelContract,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"minimax-h3-attention-runtime-v1");
    digest.update(backend.stable_id().as_bytes());
    digest.update(kernel.stable_id().as_bytes());
    digest.update(activation.stable_id().as_bytes());
    digest.update(device.stable_id().as_bytes());
    digest.update((contract.heads as u64).to_le_bytes());
    digest.update((contract.head_dim as u64).to_le_bytes());
    digest.update(contract.dtype.stable_id().as_bytes());
    digest.update(H3_FLASH_ATTN_PACKAGE_VERSION.as_bytes());
    digest.update(H3_FLASH_ATTN_CRATE_SHA256.as_bytes());
    hex_digest(digest.finalize())
}

fn plan_identity(plan: &H3FrozenAttentionPlan) -> String {
    let mut digest = Sha256::new();
    digest.update(b"minimax-h3-attention-plan-v1");
    digest.update(plan.scope.stable_id().as_bytes());
    digest.update(plan.backend.stable_id().as_bytes());
    digest.update(plan.kernel.stable_id().as_bytes());
    digest.update(plan.activation.stable_id().as_bytes());
    digest.update(plan.device.stable_id().as_bytes());
    digest.update(plan.dtype.stable_id().as_bytes());
    digest.update(plan.mask.stable_id().as_bytes());
    for value in [
        plan.batch_size,
        plan.query_rows,
        plan.key_value_rows,
        plan.heads,
        plan.head_dim,
    ] {
        digest.update((value as u64).to_le_bytes());
    }
    digest.update(plan.softmax_scale_f32_bits.to_le_bytes());
    for value in [
        plan.workspace.input_output_tensor_bytes,
        plan.workspace.qkv_compute_bytes,
        plan.workspace.score_matrix_bytes,
        plan.workspace.probability_matrix_bytes,
        plan.workspace.output_compute_bytes,
        plan.workspace.softmax_lse_bytes,
        plan.workspace.peak_auxiliary_bytes_upper_bound,
    ] {
        digest.update(value.to_le_bytes());
    }
    digest.update(H3_FLASH_ATTN_PACKAGE_VERSION.as_bytes());
    digest.update(H3_FLASH_ATTN_CRATE_SHA256.as_bytes());
    hex_digest(digest.finalize())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Execute one already-frozen H3 attention plan over BNHD tensors.
///
/// The function revalidates every serialized field and tensor property before
/// dispatch. It never silently substitutes dense attention for a missing or
/// ineligible FlashAttention kernel.
pub fn execute_h3_attention(
    plan: &H3FrozenAttentionPlan,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> InspectionResult<Tensor> {
    plan.validate_identity()?;
    let q_shape = q
        .dims4()
        .map_err(|error| kernel_shape_error("query", error))?;
    let k_shape = k
        .dims4()
        .map_err(|error| kernel_shape_error("key", error))?;
    let v_shape = v
        .dims4()
        .map_err(|error| kernel_shape_error("value", error))?;
    let expected_q = (plan.batch_size, plan.query_rows, plan.heads, plan.head_dim);
    let expected_kv = (
        plan.batch_size,
        plan.key_value_rows,
        plan.heads,
        plan.head_dim,
    );
    if q_shape != expected_q || k_shape != expected_kv || v_shape != expected_kv {
        return Err(failure(
            H3AttentionErrorCode::RuntimePlanMismatch,
            format!(
                "MiniMax H3 attention tensor shapes q={q_shape:?} k={k_shape:?} v={v_shape:?} do not match q={expected_q:?} kv={expected_kv:?}"
            ),
            vec![H3AttentionRequirement::UntamperedFrozenPlan],
        ));
    }
    let actual_dtype = H3AttentionDType::from_candle(q.dtype())?;
    if actual_dtype != plan.dtype || k.dtype() != q.dtype() || v.dtype() != q.dtype() {
        return Err(failure(
            H3AttentionErrorCode::RuntimePlanMismatch,
            "MiniMax H3 Q/K/V dtype does not match the frozen attention plan",
            vec![H3AttentionRequirement::ExactModelContract],
        ));
    }
    if !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || !plan.device.matches_candle(q.device())
    {
        return Err(failure(
            H3AttentionErrorCode::RuntimePlanMismatch,
            "MiniMax H3 Q/K/V device does not match the frozen attention plan",
            vec![H3AttentionRequirement::ExactDeviceBackend],
        ));
    }
    if plan.mask != H3AttentionMask::FullNonCausalPacked {
        return Err(failure(
            H3AttentionErrorCode::UnsupportedSemantics,
            "MiniMax H3 requires full non-causal packed self-attention",
            vec![H3AttentionRequirement::FullNonCausalPackedSelfAttention],
        ));
    }
    let scale = f32::from_bits(plan.softmax_scale_f32_bits);
    match plan.kernel {
        H3AttentionKernel::CandleDenseF32V011 => dense_attention(q, k, v, scale),
        H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011 => {
            validate_flash_tensor_strides("query", q.stride())?;
            validate_flash_tensor_strides("key", k.stride())?;
            validate_flash_tensor_strides("value", v.stride())?;
            flash_attention(q, k, v, scale)
        }
    }
}

/// Execute and synchronously time one synthetic qualification dispatch.
///
/// This is intentionally a separate API from [`execute_h3_attention`]: a
/// future authorized production runtime must choose its own sampling and
/// synchronization policy rather than paying a device-wide barrier in every
/// transformer block.
pub fn execute_h3_attention_with_telemetry(
    plan: &H3FrozenAttentionPlan,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> InspectionResult<(Tensor, H3AttentionTelemetry)> {
    plan.validate_identity()?;
    q.device()
        .synchronize()
        .map_err(|error| kernel_error("pre-qualification synchronization", error))?;
    let started = Instant::now();
    let output = execute_h3_attention(plan, q, k, v)?;
    q.device()
        .synchronize()
        .map_err(|error| kernel_error("post-qualification synchronization", error))?;
    let elapsed_micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
    let telemetry = H3AttentionTelemetry {
        schema: H3_ATTENTION_RC_SCHEMA.to_string(),
        plan_identity_sha256: plan.identity_sha256.clone(),
        backend: plan.backend,
        kernel: plan.kernel,
        activation: plan.activation,
        device: plan.device,
        scope: plan.scope,
        mask: plan.mask,
        dtype: plan.dtype,
        batch_size: plan.batch_size,
        sequence_rows: plan.query_rows,
        heads: plan.heads,
        head_dim: plan.head_dim,
        workspace: plan.workspace.clone(),
        elapsed_micros,
    };
    Ok((output, telemetry))
}

fn kernel_shape_error(name: &str, error: candle::Error) -> H3AttentionError {
    failure(
        H3AttentionErrorCode::RuntimePlanMismatch,
        format!("MiniMax H3 {name} tensor must be rank 4 BNHD: {error}"),
        vec![H3AttentionRequirement::ExactModelContract],
    )
}

fn kernel_error(context: &str, error: candle::Error) -> H3AttentionError {
    failure(
        H3AttentionErrorCode::KernelExecution,
        format!("MiniMax H3 {context} failed: {error}"),
        Vec::new(),
    )
}

fn dense_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> InspectionResult<Tensor> {
    let output_dtype = q.dtype();
    let q = q
        .transpose(1, 2)
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.to_dtype(DType::F32))
        .map_err(|error| kernel_error("dense query preparation", error))?;
    let k = k
        .transpose(1, 2)
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.to_dtype(DType::F32))
        .map_err(|error| kernel_error("dense key preparation", error))?;
    let v = v
        .transpose(1, 2)
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.to_dtype(DType::F32))
        .map_err(|error| kernel_error("dense value preparation", error))?;
    let scores = q
        .matmul(
            &k.transpose(2, 3)
                .and_then(|tensor| tensor.contiguous())
                .map_err(|error| kernel_error("dense key transpose", error))?,
        )
        .and_then(|tensor| tensor.affine(f64::from(scale), 0.0))
        .map_err(|error| kernel_error("dense score", error))?;
    let probabilities = candle_nn::ops::softmax(&scores, D::Minus1)
        .map_err(|error| kernel_error("dense softmax", error))?;
    probabilities
        .matmul(&v)
        .and_then(|tensor| tensor.to_dtype(output_dtype))
        .and_then(|tensor| tensor.transpose(1, 2))
        .and_then(|tensor| tensor.contiguous())
        .map_err(|error| kernel_error("dense output", error))
}

#[cfg(feature = "h3-flash-attn-rc")]
fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> InspectionResult<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, scale, false)
        .map_err(|error| kernel_error("FlashAttention v2 forward", error))
}

#[cfg(not(feature = "h3-flash-attn-rc"))]
fn flash_attention(_: &Tensor, _: &Tensor, _: &Tensor, _: f32) -> InspectionResult<Tensor> {
    Err(failure(
        H3AttentionErrorCode::KernelNotCompiled,
        "MiniMax H3 frozen plan references FlashAttention, but this binary omitted the kernel",
        vec![H3AttentionRequirement::CompiledH3FlashAttentionV2],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_flash_candidate() -> H3AttentionRuntimeAuthority {
        qualify_flash_attention_v2_with_availability(
            H3AttentionDevice::Cuda {
                compute_capability: Some((8, 9)),
            },
            H3AttentionModelContract::released_bf16(),
            true,
        )
        .unwrap()
    }

    #[test]
    fn release_candidate_build_requires_kernel_target_and_matching_sm89_device() {
        let sm89 = H3AttentionDevice::Cuda {
            compute_capability: Some((8, 9)),
        };
        let omitted = H3AttentionReleaseCandidateBuild::from_build_inputs(false, None);
        assert_eq!(omitted.claim_marker, None);
        assert_eq!(
            omitted.qualify(sm89).unwrap_err().code,
            H3AttentionErrorCode::KernelNotCompiled
        );

        for raw_target in [None, Some("8.9"), Some("86"), Some("90"), Some("invalid")] {
            let build = H3AttentionReleaseCandidateBuild::from_build_inputs(true, raw_target);
            assert_eq!(
                build.qualify(sm89).unwrap_err().code,
                H3AttentionErrorCode::UnqualifiedArchitecture
            );
        }

        let exact = H3AttentionReleaseCandidateBuild::from_build_inputs(true, Some("89"));
        assert_eq!(exact.compiled_compute_capability, Some((8, 9)));
        assert_eq!(
            exact.claim_marker.as_deref(),
            Some(H3_ATTENTION_RC_CLAIM_MARKER)
        );
        assert_eq!(exact.identity_sha256.len(), 64);
        assert_eq!(
            exact
                .qualify(H3AttentionDevice::Cuda {
                    compute_capability: Some((9, 0)),
                })
                .unwrap_err()
                .code,
            H3AttentionErrorCode::UnqualifiedArchitecture
        );
        assert_eq!(
            exact.qualify(sm89).unwrap().activation(),
            H3AttentionActivation::ReleaseCandidateQualificationOnly
        );

        let mut forged = exact;
        forged.kernel_compiled = false;
        assert_eq!(
            forged.qualify(sm89).unwrap_err().code,
            H3AttentionErrorCode::FrozenIdentityMismatch
        );
    }

    #[test]
    #[cfg(not(feature = "h3-flash-attn-rc"))]
    fn ordinary_build_carries_no_h3_attention_release_candidate_claim() {
        let build = H3AttentionReleaseCandidateBuild::current();
        assert!(!build.kernel_compiled);
        assert!(build.claim_marker.is_none());
        assert_eq!(
            build
                .qualify(H3AttentionDevice::Cuda {
                    compute_capability: Some((8, 9)),
                })
                .unwrap_err()
                .code,
            H3AttentionErrorCode::KernelNotCompiled
        );
    }

    #[test]
    fn released_flash_authority_is_exact_and_never_release_activated() {
        let authority = exact_flash_candidate();
        assert_eq!(authority.backend(), H3AttentionBackend::FlashAttentionV2);
        assert_eq!(
            authority.kernel(),
            H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011
        );
        assert_eq!(
            authority.activation(),
            H3AttentionActivation::ReleaseCandidateQualificationOnly
        );
        assert_eq!(authority.identity_sha256().len(), 64);
        assert_eq!(
            authority.identity_sha256(),
            H3AttentionRuntimeAuthority::expected_identity_for(
                authority.backend(),
                authority.kernel(),
                authority.activation(),
                authority.device(),
                authority.contract(),
            )
            .unwrap()
        );
        assert_eq!(H3_FLASH_ATTN_PACKAGE_VERSION, "0.11.0");
        assert_eq!(H3_FLASH_ATTN_CRATE_SHA256.len(), 64);
        let rejection = authority.require_release_activation().unwrap_err();
        assert!(rejection
            .requirements
            .contains(&H3AttentionReleaseRequirement::MiniMaxH3Authorization));
        assert!(rejection
            .requirements
            .contains(&H3AttentionReleaseRequirement::CrossBackendReproducibilityDecision));
    }

    #[test]
    fn expected_runtime_identity_rejects_each_crossed_tuple_field() {
        let device = H3AttentionDevice::Cuda {
            compute_capability: Some((8, 9)),
        };
        let contract = H3AttentionModelContract::released_bf16();
        let identity = H3AttentionRuntimeAuthority::expected_identity_for(
            H3AttentionBackend::FlashAttentionV2,
            H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
            H3AttentionActivation::ReleaseCandidateQualificationOnly,
            device,
            contract,
        )
        .unwrap();
        assert_eq!(identity.len(), 64);

        for result in [
            H3AttentionRuntimeAuthority::expected_identity_for(
                H3AttentionBackend::BoundedDenseMath,
                H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
                H3AttentionActivation::ReleaseCandidateQualificationOnly,
                device,
                contract,
            ),
            H3AttentionRuntimeAuthority::expected_identity_for(
                H3AttentionBackend::FlashAttentionV2,
                H3AttentionKernel::CandleDenseF32V011,
                H3AttentionActivation::ReleaseCandidateQualificationOnly,
                device,
                contract,
            ),
            H3AttentionRuntimeAuthority::expected_identity_for(
                H3AttentionBackend::FlashAttentionV2,
                H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
                H3AttentionActivation::SyntheticCorrectnessOnly,
                device,
                contract,
            ),
            H3AttentionRuntimeAuthority::expected_identity_for(
                H3AttentionBackend::FlashAttentionV2,
                H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
                H3AttentionActivation::ReleaseCandidateQualificationOnly,
                H3AttentionDevice::Cuda {
                    compute_capability: Some((9, 0)),
                },
                contract,
            ),
            H3AttentionRuntimeAuthority::expected_identity_for(
                H3AttentionBackend::FlashAttentionV2,
                H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
                H3AttentionActivation::ReleaseCandidateQualificationOnly,
                device,
                H3AttentionModelContract {
                    heads: 55,
                    ..contract
                },
            ),
        ] {
            assert!(result.is_err());
        }
    }

    #[test]
    fn flash_contract_mutations_fail_closed_before_tensor_execution() {
        let device = H3AttentionDevice::Cuda {
            compute_capability: Some((8, 9)),
        };
        for contract in [
            H3AttentionModelContract {
                heads: 55,
                ..H3AttentionModelContract::released_bf16()
            },
            H3AttentionModelContract {
                head_dim: 64,
                ..H3AttentionModelContract::released_bf16()
            },
            H3AttentionModelContract {
                dtype: H3AttentionDType::F16,
                ..H3AttentionModelContract::released_bf16()
            },
        ] {
            assert!(qualify_flash_attention_v2_with_availability(device, contract, true).is_err());
        }
        for wrong_device in [
            H3AttentionDevice::Cpu,
            H3AttentionDevice::Metal,
            H3AttentionDevice::Cuda {
                compute_capability: None,
            },
            H3AttentionDevice::Cuda {
                compute_capability: Some((8, 6)),
            },
            H3AttentionDevice::Cuda {
                compute_capability: Some((9, 0)),
            },
        ] {
            assert_eq!(
                qualify_flash_attention_v2_with_availability(
                    wrong_device,
                    H3AttentionModelContract::released_bf16(),
                    true,
                )
                .unwrap_err()
                .code,
                H3AttentionErrorCode::UnqualifiedArchitecture
            );
        }
        assert_eq!(
            qualify_flash_attention_v2_with_availability(
                device,
                H3AttentionModelContract::released_bf16(),
                false,
            )
            .unwrap_err()
            .code,
            H3AttentionErrorCode::KernelNotCompiled
        );
    }

    #[test]
    fn dense_math_is_bounded_to_synthetic_score_shapes() {
        let authority = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::Cpu,
            H3AttentionModelContract::released_bf16(),
        )
        .unwrap();
        let small = authority.freeze_execution(1, 32, 64).unwrap();
        assert_eq!(
            small.packed_transformer.backend(),
            H3AttentionBackend::BoundedDenseMath
        );
        assert!(small.packed_transformer.workspace().score_matrix_bytes > 0);
        let error = authority.freeze_execution(1, 256, 37_296).unwrap_err();
        assert_eq!(error.code, H3AttentionErrorCode::DenseMathLimitExceeded);
        assert!(error
            .requirements
            .contains(&H3AttentionRequirement::CompiledH3FlashAttentionV2));
    }

    #[test]
    fn flash_workspace_is_linear_for_released_row_counts() {
        let authority = exact_flash_candidate();
        let short = authority.freeze_execution(1, 256, 37_296).unwrap();
        let long = authority.freeze_execution(1, 256, 107_856).unwrap();
        for plan in [&short.packed_transformer, &long.packed_transformer] {
            assert_eq!(plan.workspace().score_matrix_bytes, 0);
            assert_eq!(plan.workspace().probability_matrix_bytes, 0);
            assert_eq!(
                plan.workspace().softmax_lse_bytes,
                (plan.heads() * plan.rows() * 4) as u64
            );
            assert_eq!(
                plan.workspace().peak_auxiliary_bytes_upper_bound,
                plan.workspace().softmax_lse_bytes
            );
        }
        assert_eq!(
            short.packed_transformer.workspace().softmax_lse_bytes,
            8_354_304
        );
        assert_eq!(
            long.packed_transformer.workspace().softmax_lse_bytes,
            24_159_744
        );
        assert_ne!(
            short.packed_transformer.identity_sha256(),
            long.packed_transformer.identity_sha256()
        );
    }

    #[test]
    fn flash_release_candidate_rejects_non_native_batch() {
        let authority = exact_flash_candidate();
        for batch_size in [2, 8] {
            let error = authority
                .freeze_execution(batch_size, 256, 37_296)
                .unwrap_err();
            assert_eq!(error.code, H3AttentionErrorCode::InvalidShape);
            assert!(error
                .requirements
                .contains(&H3AttentionRequirement::NativeBatchOne));
        }

        let mut forged = authority
            .freeze_execution(1, 8, 8)
            .unwrap()
            .packed_transformer;
        forged.batch_size = 2;
        forged.identity_sha256 = plan_identity(&forged);
        let error = forged.validate_identity().unwrap_err();
        assert_eq!(error.code, H3AttentionErrorCode::InvalidShape);
        assert!(error
            .requirements
            .contains(&H3AttentionRequirement::NativeBatchOne));

        let dense = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::Cpu,
            H3AttentionModelContract::released_bf16(),
        )
        .unwrap();
        assert!(dense.freeze_execution(2, 8, 8).is_ok());
    }

    #[test]
    fn flash_tensor_strides_are_checked_before_u32_ffi_conversion() {
        assert!(validate_flash_tensor_strides("query", &[1_048_576, 8_192, 128, 1]).is_ok());

        let non_unit_head_stride =
            validate_flash_tensor_strides("query", &[16, 8, 2, 2]).unwrap_err();
        assert_eq!(
            non_unit_head_stride.code,
            H3AttentionErrorCode::RuntimePlanMismatch
        );
        assert!(non_unit_head_stride
            .requirements
            .contains(&H3AttentionRequirement::ExactModelContract));

        #[cfg(target_pointer_width = "64")]
        {
            let oversized = u32::MAX as usize + 1;
            for dimension in 0..3 {
                let mut strides = [1_048_576, 8_192, 128, 1];
                strides[dimension] = oversized;
                let error = validate_flash_tensor_strides("query", &strides).unwrap_err();
                assert_eq!(error.code, H3AttentionErrorCode::RuntimePlanMismatch);
                assert!(error
                    .requirements
                    .contains(&H3AttentionRequirement::FlashAttentionU32LaunchShape));
            }
            assert!(validate_flash_tensor_strides(
                "query",
                &[u32::MAX as usize, u32::MAX as usize, u32::MAX as usize, 1]
            )
            .is_ok());
        }
    }

    #[test]
    fn frozen_plan_mutation_and_tensor_shape_mismatch_are_rejected() {
        let authority = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::Cpu,
            H3AttentionModelContract {
                heads: 2,
                head_dim: 8,
                dtype: H3AttentionDType::F32,
            },
        )
        .unwrap();
        let plan = authority
            .freeze_execution(1, 3, 4)
            .unwrap()
            .packed_transformer;
        let mut shape_mutation = plan.clone();
        shape_mutation.query_rows += 1;
        let mut backend_mutation = plan.clone();
        backend_mutation.backend = H3AttentionBackend::FlashAttentionV2;
        let mut kernel_mutation = plan.clone();
        kernel_mutation.kernel = H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011;
        let mut dtype_mutation = plan.clone();
        dtype_mutation.dtype = H3AttentionDType::Bf16;
        let mut workspace_mutation = plan.clone();
        workspace_mutation.workspace.softmax_lse_bytes = 1;
        for mutation in [
            shape_mutation,
            backend_mutation,
            kernel_mutation,
            dtype_mutation,
            workspace_mutation,
        ] {
            assert_eq!(
                mutation.validate_identity().unwrap_err().code,
                H3AttentionErrorCode::FrozenIdentityMismatch
            );
        }

        let mut semantically_forged = plan.clone();
        semantically_forged.backend = H3AttentionBackend::FlashAttentionV2;
        semantically_forged.identity_sha256 = plan_identity(&semantically_forged);
        assert_eq!(
            semantically_forged.validate_identity().unwrap_err().code,
            H3AttentionErrorCode::RuntimePlanMismatch
        );

        let q = Tensor::zeros((1, 4, 2, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::zeros((1, 3, 2, 8), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            execute_h3_attention(&plan, &q, &k, &k).unwrap_err().code,
            H3AttentionErrorCode::RuntimePlanMismatch
        );
    }

    #[test]
    fn bounded_dense_execution_matches_direct_f32_reference() {
        let authority = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::Cpu,
            H3AttentionModelContract {
                heads: 2,
                head_dim: 8,
                dtype: H3AttentionDType::F32,
            },
        )
        .unwrap();
        let plan = authority
            .freeze_execution(1, 3, 5)
            .unwrap()
            .packed_transformer;
        let values = (0..80)
            .map(|index| index as f32 / 41.0 - 0.8)
            .collect::<Vec<_>>();
        let q = Tensor::from_vec(values.clone(), (1, 5, 2, 8), &Device::Cpu).unwrap();
        let k = (&q / 3.0).unwrap();
        let v = Tensor::from_vec(
            values.into_iter().rev().collect::<Vec<_>>(),
            (1, 5, 2, 8),
            &Device::Cpu,
        )
        .unwrap();
        let actual = execute_h3_attention(&plan, &q, &k, &v).unwrap();
        let scale = 1.0 / (8f64).sqrt();
        let qh = q.transpose(1, 2).unwrap().contiguous().unwrap();
        let kh = k.transpose(1, 2).unwrap().contiguous().unwrap();
        let vh = v.transpose(1, 2).unwrap().contiguous().unwrap();
        let expected = candle_nn::ops::softmax(
            &qh.matmul(&kh.transpose(2, 3).unwrap().contiguous().unwrap())
                .unwrap()
                .affine(scale, 0.0)
                .unwrap(),
            D::Minus1,
        )
        .unwrap()
        .matmul(&vh)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
        let max = (actual - expected)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max <= 1e-6, "dense max difference {max}");
    }

    #[test]
    fn qualification_telemetry_contains_only_frozen_geometry_and_timing() {
        let authority = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::Cpu,
            H3AttentionModelContract {
                heads: 2,
                head_dim: 8,
                dtype: H3AttentionDType::F32,
            },
        )
        .unwrap();
        let plan = authority
            .freeze_execution(1, 3, 5)
            .unwrap()
            .packed_transformer;
        let q = Tensor::zeros((1, 5, 2, 8), DType::F32, &Device::Cpu).unwrap();
        let (output, telemetry) = execute_h3_attention_with_telemetry(&plan, &q, &q, &q).unwrap();
        assert_eq!(output.dims(), &[1, 5, 2, 8]);
        assert_eq!(telemetry.schema, H3_ATTENTION_RC_SCHEMA);
        assert_eq!(telemetry.plan_identity_sha256, plan.identity_sha256());
        assert_eq!(telemetry.scope, H3AttentionScope::PackedTransformer);
        assert_eq!(telemetry.mask, H3AttentionMask::FullNonCausalPacked);
        assert_eq!(telemetry.sequence_rows, 5);
        assert_eq!(telemetry.heads, 2);
        assert_eq!(telemetry.head_dim, 8);
        assert_eq!(telemetry.workspace, plan.workspace().clone());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_keeps_only_the_bounded_synthetic_dense_path() -> candle::Result<()> {
        let metal = Device::new_metal(0)?;
        let authority = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::from_candle(&metal),
            H3AttentionModelContract {
                heads: 2,
                head_dim: 8,
                dtype: H3AttentionDType::F32,
            },
        )
        .unwrap();
        let plan = authority
            .freeze_execution(1, 3, 5)
            .unwrap()
            .packed_transformer;
        let q = Tensor::zeros((1, 5, 2, 8), DType::F32, &metal)?;
        let output = execute_h3_attention(&plan, &q, &q, &q).unwrap();
        assert_eq!(output.dims4()?, (1, 5, 2, 8));
        assert!(output.device().same_device(&metal));

        let product_error = authority.freeze_execution(1, 3, 37_296).unwrap_err();
        assert_eq!(
            product_error.code,
            H3AttentionErrorCode::DenseMathLimitExceeded
        );
        assert_eq!(
            qualify_flash_attention_v2_with_availability(
                H3AttentionDevice::Metal,
                H3AttentionModelContract::released_bf16(),
                true,
            )
            .unwrap_err()
            .code,
            H3AttentionErrorCode::UnqualifiedArchitecture
        );
        Ok(())
    }

    #[cfg(feature = "h3-flash-attn-rc")]
    #[test]
    fn flash_bf16_cpu_reference_cuda_parity() -> candle::Result<()> {
        fn deterministic_probe_values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
            (0..count)
                .map(|index| {
                    let mut value = (index as u64).wrapping_add(seed);
                    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    value ^= value >> 31;
                    (((value >> 48) as i32 - 32_768) as f32 / 65_536.0) * scale
                })
                .collect()
        }

        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let flash = H3AttentionRuntimeAuthority::qualify_flash_attention_v2(
            H3AttentionDevice::Cuda {
                compute_capability: Some((8, 9)),
            },
            H3AttentionModelContract::released_bf16(),
        )
        .unwrap();
        let dense = H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::Cpu,
            H3AttentionModelContract::released_bf16(),
        )
        .unwrap();
        let flash_plan = flash.freeze_execution(1, 7, 37).unwrap().packed_transformer;
        let dense_plan = dense.freeze_execution(1, 7, 37).unwrap().packed_transformer;
        let count = 37 * H3_RELEASE_HEADS * H3_RELEASE_HEAD_DIM;
        let q_cpu = Tensor::from_vec(
            deterministic_probe_values(count, 0x243f_6a88_85a3_08d3, 2.0),
            (1, 37, H3_RELEASE_HEADS, H3_RELEASE_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?;
        let k_cpu = Tensor::from_vec(
            deterministic_probe_values(count, 0x1319_8a2e_0370_7344, 2.0),
            (1, 37, H3_RELEASE_HEADS, H3_RELEASE_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?;
        let v_cpu = Tensor::from_vec(
            deterministic_probe_values(count, 0xa409_3822_299f_31d0, 1.0),
            (1, 37, H3_RELEASE_HEADS, H3_RELEASE_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?;
        let q_gpu = Tensor::from_vec(
            deterministic_probe_values(count, 0x243f_6a88_85a3_08d3, 2.0),
            (1, 37, H3_RELEASE_HEADS, H3_RELEASE_HEAD_DIM),
            &cuda,
        )?
        .to_dtype(DType::BF16)?;
        let k_gpu = Tensor::from_vec(
            deterministic_probe_values(count, 0x1319_8a2e_0370_7344, 2.0),
            (1, 37, H3_RELEASE_HEADS, H3_RELEASE_HEAD_DIM),
            &cuda,
        )?
        .to_dtype(DType::BF16)?;
        let v_gpu = Tensor::from_vec(
            deterministic_probe_values(count, 0xa409_3822_299f_31d0, 1.0),
            (1, 37, H3_RELEASE_HEADS, H3_RELEASE_HEAD_DIM),
            &cuda,
        )?
        .to_dtype(DType::BF16)?;
        let cpu_out = execute_h3_attention(&dense_plan, &q_cpu, &k_cpu, &v_cpu)
            .unwrap()
            .to_dtype(DType::F32)?;
        let swapped_out = execute_h3_attention(&dense_plan, &k_cpu, &q_cpu, &v_cpu)
            .unwrap()
            .to_dtype(DType::F32)?;
        let swap_max = (&cpu_out - swapped_out)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            swap_max > 0.02,
            "qualification vectors do not expose swapped Q/K wiring: {swap_max}"
        );
        let gpu_out = execute_h3_attention(&flash_plan, &q_gpu, &k_gpu, &v_gpu)
            .unwrap()
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?;
        let max = (&cpu_out - gpu_out)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(max <= 0.02, "FlashAttention BF16 max difference {max}");
        Ok(())
    }
}
