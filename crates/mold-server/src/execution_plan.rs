//! Request-aware, concrete execution plans used by scheduler admission.
//!
//! The scheduler never guesses component placement from a model name. This
//! module resolves the already-normalized request plus concrete model paths
//! into one immutable plan per eligible device. Plans are validated again on
//! the owner thread before model loading touches CUDA.

use mold_core::{
    Config, DevicePlacement, DeviceRef, GenerateRequest, GpuBackend, ModelPaths, OutputFormat,
};
use mold_scheduler::ExecutionEquivalenceFingerprint;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const MIB: u64 = 1024 * 1024;
const BASE_HOST_TRANSIENT: u64 = 256 * MIB;
const UNKNOWN_ARTIFACT_HOST_CHARGE: u64 = 64 * MIB;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum ComponentRole {
    Transformer,
    TransformerShard(u8),
    Vae,
    T5,
    T5Tokenizer,
    ClipL,
    ClipLTokenizer,
    ClipG,
    ClipGTokenizer,
    QwenShard(u16),
    GemmaShard(u16),
    GenericTextEncoderShard(u16),
    TextTokenizer,
    Lora(u16),
    SpatialUpscaler,
    TemporalUpscaler,
    Decoder,
    DistilledLora,
}

impl ComponentRole {
    fn is_text_encoder(&self) -> bool {
        matches!(
            self,
            Self::T5
                | Self::ClipL
                | Self::ClipG
                | Self::QwenShard(_)
                | Self::GemmaShard(_)
                | Self::GenericTextEncoderShard(_)
        )
    }

    fn is_host_only(&self) -> bool {
        matches!(
            self,
            Self::T5Tokenizer
                | Self::ClipLTokenizer
                | Self::ClipGTokenizer
                | Self::TextTokenizer
                | Self::Lora(_)
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedComponentConstraint {
    Auto,
    Cpu,
    Device(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedComponentPlacement {
    Cpu,
    Device(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum ComponentLoadStrategy {
    Resident,
    DropReload,
    ParkedCpu,
    StreamedBlocks,
    TiledVae,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum QuantizationVariant {
    Q2,
    Q3,
    Q4,
    Q5,
    Q6,
    Q8,
    Fp8,
    Nvfp4,
    ConvRotW4A4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum PlannedDType {
    Bf16,
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum AttentionBackend {
    Math,
    Flash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum OffloadMode {
    None,
    Block,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum DeterminismClass {
    CpuSeededCrossBackend,
    /// Reserved for engines whose initial noise or sampling is backend-local.
    /// No current production family selects this class.
    BackendSeeded,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ContentFingerprint(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum EquivalenceContentIdentity {
    Sha256(String),
    /// The artifact could not be read. This opaque process-local token makes
    /// the plan fail closed without treating a path or inode as content.
    Unknown {
        discriminator: String,
    },
}

impl EquivalenceContentIdentity {
    fn update_hash(&self, hash: &mut Sha256) {
        hash.update(
            serde_json::to_vec(self).expect("content fingerprint serialization is infallible"),
        );
    }
}

/// Stable architecture boundary for deterministic parent execution.
///
/// CUDA compute capability comes from the driver discovery record. Unknown
/// architecture is deliberately device-specific so two devices with missing
/// facts cannot be assumed equivalent. No display-name parsing is permitted.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum DeviceArchitectureClass {
    CudaComputeCapability {
        major: u16,
        minor: u16,
    },
    MetalDefault,
    Unknown {
        backend: GpuBackend,
        device_id: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum SemanticComponentPlacement {
    Cpu,
    AssignedDevice,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum EngineLoadStrategyClass {
    Eager,
    Sequential,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum AttentionKernelClass {
    Math,
    Flash,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionCodeIdentity {
    pub package_version: String,
    pub source_revision: Option<String>,
    pub scope: CodeIdentityScope,
    pub process_discriminator: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum CodeIdentityScope {
    ImmutableBuild,
    CurrentProcessOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum SemanticAttentionBackend {
    Math,
    Flash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum SemanticAttentionChunk {
    Auto,
    Off,
    Size(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum SemanticVaeTiling {
    Auto,
    Force,
    Off,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum SemanticVaeDType {
    Auto,
    Bf16,
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum RuntimeSemanticVariable {
    Attn,
    AttnChunk,
    CfgPlus,
    Device,
    Eager,
    FluxDeltaCache,
    FluxKeepTransformer,
    KeepTeRam,
    LongPrompts,
    LoraBypass,
    LtxDebugAltPrompt,
    LtxDebugCompareUncond,
    LtxDebugDisableAudioBranch,
    LtxDebugDisableCrossAttentionAdaLn,
    Ltx2DebugDisableTransformerGatedAttention,
    Ltx2DebugForceCpuPromptEncoder,
    Ltx2DebugLoadBlocks,
    Ltx2ForceEager,
    Ltx2ForceStreaming,
    Ltx2Fp8InputScaleMode,
    Ltx2Fp8WeightScaleMode,
    Ltx2GemmaDevice,
    Ltx2GemmaVariant,
    Ltx2VaeDecodeChunkFrames,
    Ltx2VaeDecodeContextFrames,
    Ltx2VaeForceFramewise,
    Ltx2VaeForceFullDecode,
    Nvfp4Backend,
    Offload,
    OffloadPrefetch,
    PinnedVramMaxGb,
    Qwen2TextEncoderMode,
    Qwen2Variant,
    Qwen3Variant,
    ReserveVramMb,
    T5Variant,
    VaeDtype,
    VaeTiled,
    WuerstchenDecoderGuidance,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum CanonicalRuntimeValue {
    Unset,
    Boolean(bool),
    Presence(bool),
    Unsigned(Option<u64>),
    FloatBits(Option<u64>),
    Text(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeSemanticSetting {
    pub variable: RuntimeSemanticVariable,
    pub value: CanonicalRuntimeValue,
}

/// Canonical engine construction and runtime semantics that are independent
/// of the exact model/path route retained by the enclosing descriptor.
///
/// `from_frozen` fully destructures `FrozenEngineConfig`; adding a field to
/// the source type is therefore a compile error until its equivalence effect
/// is consciously classified here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionSemanticConfig {
    pub family: String,
    pub is_schnell: Option<bool>,
    pub is_turbo: Option<bool>,
    pub scheduler: Option<mold_core::Scheduler>,
    pub t5_variant: Option<String>,
    pub qwen3_variant: Option<String>,
    pub qwen2_variant: Option<String>,
    pub qwen2_text_encoder_mode: Option<String>,
    pub ltx2_gemma_variant: Option<String>,
    pub attention_backend: SemanticAttentionBackend,
    pub attention_chunk: SemanticAttentionChunk,
    pub vae_tiling: SemanticVaeTiling,
    pub vae_dtype: SemanticVaeDType,
    pub runtime: Vec<RuntimeSemanticSetting>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum EffectiveComponentDType {
    NotApplicable,
    Bf16,
    F16,
    F32,
    QuantizedNative,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum ArtifactFormatUnknown {
    CacheMiss,
    Io,
    UnsupportedContainer,
    InvalidHeader,
    UnsupportedTensorDType,
    UnsupportedGgufTensorFormat,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ComponentStorageFormat {
    Known(mold_inference::artifact_format::ArtifactStorageFormat),
    Unknown {
        reason: ArtifactFormatUnknown,
        content_discriminator: EquivalenceContentIdentity,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EffectiveComponentPrecision {
    pub storage: ComponentStorageFormat,
    pub compute_dtype: EffectiveComponentDType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EquivalentComponentExecution {
    pub role: ComponentRole,
    pub content_fingerprint: EquivalenceContentIdentity,
    pub precision: EffectiveComponentPrecision,
    pub dtype: Option<PlannedDType>,
    pub quantization: Option<QuantizationVariant>,
    pub placement: SemanticComponentPlacement,
    pub load_strategy: ComponentLoadStrategy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EquivalentLoraExecution {
    pub content_fingerprint: EquivalenceContentIdentity,
    pub scale_bits: u64,
}

/// Exact process-platform representation of a runtime artifact path.
///
/// Legacy engine construction still consumes model IDs, filenames,
/// extensions, and path equality in several family-specific dispatchers.
/// Until those dispatchers are fully content-typed, equivalence must retain
/// this conservative route identity instead of claiming path independence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum RuntimePathIdentity {
    UnixBytes(Vec<u8>),
    WindowsWide(Vec<u16>),
    Portable(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeArtifactPathIdentity {
    pub role: ComponentRole,
    pub path: RuntimePathIdentity,
}

/// Canonical deterministic-output environment for a future server batch
/// parent. Device identity, ordinal, and capacity estimates are absent.
/// Runtime model/path identity remains explicit while legacy engines still
/// branch on those values.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionEnvironmentDescriptor {
    pub schema_version: u16,
    pub backend: GpuBackend,
    pub architecture: DeviceArchitectureClass,
    pub attention_kernel_class: AttentionKernelClass,
    pub code: ExecutionCodeIdentity,
    pub semantic_config: ExecutionSemanticConfig,
    pub runtime_model_id: String,
    pub runtime_artifact_paths: Vec<RuntimeArtifactPathIdentity>,
    pub model_family: String,
    pub model_fingerprint: String,
    pub components: Vec<EquivalentComponentExecution>,
    pub loras: Vec<EquivalentLoraExecution>,
    pub engine_load_strategy: EngineLoadStrategyClass,
    pub offload_mode: OffloadMode,
    pub output_format: OutputFormat,
    pub determinism_class: DeterminismClass,
}

fn runtime_path_identity(path: &Path) -> RuntimePathIdentity {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        RuntimePathIdentity::UnixBytes(path.as_os_str().as_bytes().to_vec())
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        RuntimePathIdentity::WindowsWide(path.as_os_str().encode_wide().collect())
    }
    #[cfg(not(any(unix, windows)))]
    {
        RuntimePathIdentity::Portable(path.to_string_lossy().into_owned())
    }
}

impl ExecutionEnvironmentDescriptor {
    pub fn fingerprint(&self) -> ExecutionEquivalenceFingerprint {
        let encoded = serde_json::to_vec(self)
            .expect("execution environment descriptor serialization is infallible");
        let mut hash = Sha256::new();
        hash.update(b"mold.execution-equivalence.v3\0");
        hash.update(encoded);
        ExecutionEquivalenceFingerprint::new(format!("{:x}", hash.finalize()))
    }
}

impl ExecutionSemanticConfig {
    pub fn from_frozen(frozen: &mold_inference::FrozenEngineConfig) -> Self {
        let mold_inference::FrozenEngineConfig {
            family,
            is_schnell,
            is_turbo,
            scheduler,
            t5_variant,
            qwen3_variant,
            qwen2_variant,
            qwen2_text_encoder_mode,
            ltx2_gemma_variant,
            // Selected artifacts are represented by component content/format
            // facts and the conservative exact runtime route in the enclosing
            // descriptor.
            selected_t5_path: _,
            selected_qwen3_paths: _,
            selected_qwen2_path: _,
            selected_gemma_paths: _,
            runtime_environment,
            attention_backend,
            attention_chunk,
            vae_tiling,
            vae_dtype,
        } = frozen;
        let runtime = mold_inference::runtime_env::ENGINE_SHAPING_VARIABLES
            .iter()
            .map(|name| runtime_semantic_setting(name, runtime_environment.value(name)))
            .collect();
        Self {
            family: family.clone(),
            is_schnell: *is_schnell,
            is_turbo: *is_turbo,
            scheduler: *scheduler,
            t5_variant: t5_variant.clone(),
            qwen3_variant: qwen3_variant.clone(),
            qwen2_variant: qwen2_variant.clone(),
            qwen2_text_encoder_mode: qwen2_text_encoder_mode.clone(),
            ltx2_gemma_variant: ltx2_gemma_variant.clone(),
            attention_backend: match attention_backend {
                mold_inference::attention::AttentionBackend::Math => SemanticAttentionBackend::Math,
                mold_inference::attention::AttentionBackend::Flash => {
                    SemanticAttentionBackend::Flash
                }
            },
            attention_chunk: match attention_chunk {
                mold_inference::attention::AttentionChunkPolicy::Auto => {
                    SemanticAttentionChunk::Auto
                }
                mold_inference::attention::AttentionChunkPolicy::Off => SemanticAttentionChunk::Off,
                mold_inference::attention::AttentionChunkPolicy::Size(size) => {
                    SemanticAttentionChunk::Size(*size as u64)
                }
            },
            vae_tiling: match vae_tiling {
                mold_inference::vae_tiling::TiledMode::Auto => SemanticVaeTiling::Auto,
                mold_inference::vae_tiling::TiledMode::Force => SemanticVaeTiling::Force,
                mold_inference::vae_tiling::TiledMode::Off => SemanticVaeTiling::Off,
            },
            vae_dtype: match vae_dtype {
                mold_inference::device::VaeDtypePolicy::Auto => SemanticVaeDType::Auto,
                mold_inference::device::VaeDtypePolicy::Bf16 => SemanticVaeDType::Bf16,
                mold_inference::device::VaeDtypePolicy::F16 => SemanticVaeDType::F16,
                mold_inference::device::VaeDtypePolicy::F32 => SemanticVaeDType::F32,
            },
            runtime,
        }
    }
}

fn runtime_semantic_setting(name: &str, value: Option<&str>) -> RuntimeSemanticSetting {
    let variable = match name {
        "MOLD_ATTN" => RuntimeSemanticVariable::Attn,
        "MOLD_ATTN_CHUNK" => RuntimeSemanticVariable::AttnChunk,
        "MOLD_CFG_PLUS" => RuntimeSemanticVariable::CfgPlus,
        "MOLD_DEVICE" => RuntimeSemanticVariable::Device,
        "MOLD_EAGER" => RuntimeSemanticVariable::Eager,
        "MOLD_FLUX_DELTA_CACHE" => RuntimeSemanticVariable::FluxDeltaCache,
        "MOLD_FLUX_KEEP_TRANSFORMER" => RuntimeSemanticVariable::FluxKeepTransformer,
        "MOLD_KEEP_TE_RAM" => RuntimeSemanticVariable::KeepTeRam,
        "MOLD_LONG_PROMPTS" => RuntimeSemanticVariable::LongPrompts,
        "MOLD_LORA_BYPASS" => RuntimeSemanticVariable::LoraBypass,
        "MOLD_LTX_DEBUG_ALT_PROMPT" => RuntimeSemanticVariable::LtxDebugAltPrompt,
        "MOLD_LTX_DEBUG_COMPARE_UNCOND" => RuntimeSemanticVariable::LtxDebugCompareUncond,
        "MOLD_LTX_DEBUG_DISABLE_AUDIO_BRANCH" => {
            RuntimeSemanticVariable::LtxDebugDisableAudioBranch
        }
        "MOLD_LTX_DEBUG_DISABLE_CROSS_ATTENTION_ADALN" => {
            RuntimeSemanticVariable::LtxDebugDisableCrossAttentionAdaLn
        }
        "MOLD_LTX2_DEBUG_DISABLE_TRANSFORMER_GATED_ATTENTION" => {
            RuntimeSemanticVariable::Ltx2DebugDisableTransformerGatedAttention
        }
        "MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER" => {
            RuntimeSemanticVariable::Ltx2DebugForceCpuPromptEncoder
        }
        "MOLD_LTX2_DEBUG_LOAD_BLOCKS" => RuntimeSemanticVariable::Ltx2DebugLoadBlocks,
        "MOLD_LTX2_FORCE_EAGER" => RuntimeSemanticVariable::Ltx2ForceEager,
        "MOLD_LTX2_FORCE_STREAMING" => RuntimeSemanticVariable::Ltx2ForceStreaming,
        "MOLD_LTX2_FP8_INPUT_SCALE_MODE" => RuntimeSemanticVariable::Ltx2Fp8InputScaleMode,
        "MOLD_LTX2_FP8_WEIGHT_SCALE_MODE" => RuntimeSemanticVariable::Ltx2Fp8WeightScaleMode,
        "MOLD_LTX2_GEMMA_DEVICE" => RuntimeSemanticVariable::Ltx2GemmaDevice,
        "MOLD_LTX2_GEMMA_VARIANT" => RuntimeSemanticVariable::Ltx2GemmaVariant,
        "MOLD_LTX2_VAE_DECODE_CHUNK_FRAMES" => RuntimeSemanticVariable::Ltx2VaeDecodeChunkFrames,
        "MOLD_LTX2_VAE_DECODE_CONTEXT_FRAMES" => {
            RuntimeSemanticVariable::Ltx2VaeDecodeContextFrames
        }
        "MOLD_LTX2_VAE_FORCE_FRAMEWISE" => RuntimeSemanticVariable::Ltx2VaeForceFramewise,
        "MOLD_LTX2_VAE_FORCE_FULL_DECODE" => RuntimeSemanticVariable::Ltx2VaeForceFullDecode,
        "MOLD_NVFP4_BACKEND" => RuntimeSemanticVariable::Nvfp4Backend,
        "MOLD_OFFLOAD" => RuntimeSemanticVariable::Offload,
        "MOLD_OFFLOAD_PREFETCH" => RuntimeSemanticVariable::OffloadPrefetch,
        "MOLD_PINNED_VRAM_MAX_GB" => RuntimeSemanticVariable::PinnedVramMaxGb,
        "MOLD_QWEN2_TEXT_ENCODER_MODE" => RuntimeSemanticVariable::Qwen2TextEncoderMode,
        "MOLD_QWEN2_VARIANT" => RuntimeSemanticVariable::Qwen2Variant,
        "MOLD_QWEN3_VARIANT" => RuntimeSemanticVariable::Qwen3Variant,
        "MOLD_RESERVE_VRAM_MB" => RuntimeSemanticVariable::ReserveVramMb,
        "MOLD_T5_VARIANT" => RuntimeSemanticVariable::T5Variant,
        "MOLD_VAE_DTYPE" => RuntimeSemanticVariable::VaeDtype,
        "MOLD_VAE_TILED" => RuntimeSemanticVariable::VaeTiled,
        "MOLD_WUERSTCHEN_DECODER_GUIDANCE" => RuntimeSemanticVariable::WuerstchenDecoderGuidance,
        // A new engine-shaping input must never silently collapse into an old
        // equivalence class.
        _ => panic!("unclassified frozen engine-shaping variable {name}"),
    };
    let value = match value {
        None => CanonicalRuntimeValue::Unset,
        Some(value)
            if matches!(
                variable,
                RuntimeSemanticVariable::LtxDebugCompareUncond
                    | RuntimeSemanticVariable::LtxDebugDisableAudioBranch
                    | RuntimeSemanticVariable::LtxDebugDisableCrossAttentionAdaLn
                    | RuntimeSemanticVariable::Ltx2DebugDisableTransformerGatedAttention
                    | RuntimeSemanticVariable::Ltx2DebugLoadBlocks
                    | RuntimeSemanticVariable::Ltx2ForceEager
                    | RuntimeSemanticVariable::Ltx2ForceStreaming
                    | RuntimeSemanticVariable::Ltx2VaeForceFramewise
                    | RuntimeSemanticVariable::Ltx2VaeForceFullDecode
                    | RuntimeSemanticVariable::Ltx2DebugForceCpuPromptEncoder
            ) =>
        {
            let _ = value;
            CanonicalRuntimeValue::Presence(true)
        }
        Some(value) if variable == RuntimeSemanticVariable::LtxDebugAltPrompt => {
            CanonicalRuntimeValue::Text(value.to_string())
        }
        Some(value)
            if matches!(
                variable,
                RuntimeSemanticVariable::Ltx2VaeDecodeChunkFrames
                    | RuntimeSemanticVariable::Ltx2VaeDecodeContextFrames
                    | RuntimeSemanticVariable::ReserveVramMb
            ) =>
        {
            CanonicalRuntimeValue::Unsigned(mold_inference::runtime_env::parse_u64(value))
        }
        Some(value)
            if matches!(
                variable,
                RuntimeSemanticVariable::PinnedVramMaxGb
                    | RuntimeSemanticVariable::WuerstchenDecoderGuidance
            ) =>
        {
            CanonicalRuntimeValue::FloatBits(
                mold_inference::runtime_env::parse_f64(value).map(f64::to_bits),
            )
        }
        Some(value) if variable == RuntimeSemanticVariable::CfgPlus => {
            CanonicalRuntimeValue::Boolean(matches!(value, "1" | "true" | "yes"))
        }
        Some(value) if variable == RuntimeSemanticVariable::LongPrompts => {
            CanonicalRuntimeValue::Boolean(value == "1")
        }
        // Do not invent normalization for a runtime parser we have not made
        // authoritative here. Exact text is conservative: it can cause false
        // negatives, but never a false-equivalent execution class.
        Some(value) => CanonicalRuntimeValue::Text(value.to_string()),
    };
    RuntimeSemanticSetting { variable, value }
}

fn execution_code_identity_for(package_version: &str, git_sha: &str) -> ExecutionCodeIdentity {
    let source_revision = (git_sha != "unknown").then(|| git_sha.to_string());
    let (scope, process_discriminator) = if source_revision.is_some() {
        (CodeIdentityScope::ImmutableBuild, None)
    } else {
        static PROCESS_DISCRIMINATOR: OnceLock<String> = OnceLock::new();
        let discriminator = PROCESS_DISCRIMINATOR
            .get_or_init(|| {
                let mut bytes = [0_u8; 16];
                getrandom::fill(&mut bytes)
                    .expect("execution equivalence requires process-unique entropy");
                bytes.iter().map(|byte| format!("{byte:02x}")).collect()
            })
            .clone();
        (CodeIdentityScope::CurrentProcessOnly, Some(discriminator))
    };
    ExecutionCodeIdentity {
        package_version: package_version.to_string(),
        source_revision,
        scope,
        process_discriminator,
    }
}

fn execution_code_identity() -> ExecutionCodeIdentity {
    execution_code_identity_for(
        mold_core::build_info::VERSION,
        mold_core::build_info::GIT_SHA,
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentExecutionPlan {
    pub role: ComponentRole,
    pub artifact_path: PathBuf,
    pub content_fingerprint: ContentFingerprint,
    pub dtype: Option<PlannedDType>,
    pub quantization: Option<QuantizationVariant>,
    pub placement: ResolvedComponentPlacement,
    pub load_strategy: ComponentLoadStrategy,
    pub predicted_vram_bytes: u64,
    pub predicted_host_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectivePlacement {
    pub components: BTreeMap<ComponentRole, ResolvedComponentConstraint>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementCapabilities {
    pub supports_text_encoder_cpu: bool,
    pub supports_vae_cpu: bool,
    pub supports_audio_components_cpu: bool,
    pub supports_block_offload: bool,
    pub supports_tiled_vae: bool,
    /// Static family batch contract available before an engine is loaded.
    /// Unknown families remain `None` and cannot opt into parent batching.
    pub batch_execution: Option<mold_inference::BatchExecutionCapability>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedExecutionPlan {
    pub device_id: String,
    pub device_ordinal: usize,
    /// Semantic family resolved from authoritative model metadata.
    pub model_family: String,
    pub model_fingerprint: String,
    pub effective_placement: EffectivePlacement,
    pub components: BTreeMap<ComponentRole, ComponentExecutionPlan>,
    /// Exact paths and factory inputs consumed after the lease grant. Worker
    /// dispatch must not re-resolve these from mutable config or environment.
    pub engine_paths: ModelPaths,
    pub engine_config: mold_inference::FrozenEngineConfig,
    /// Mutable inputs observed before dependency preparation. These remain
    /// separate from the materialized engine inputs because an auto encoder
    /// preference is intentionally replaced by one concrete per-device
    /// variant during preparation.
    pub admission_paths: ModelPaths,
    pub admission_engine_config: mold_inference::FrozenEngineConfig,
    /// Ordered effective request/default LoRA stack. Scale uses IEEE bits so
    /// cache identity and equality preserve every finite wire value exactly.
    pub effective_loras: Vec<PlannedLora>,
    pub attention_backend: AttentionBackend,
    pub engine_load_strategy: mold_inference::LoadStrategy,
    pub offload_mode: OffloadMode,
    pub predicted_vram_peak_bytes: u64,
    /// Exact effective capacity against which this candidate was admitted:
    /// current sampled free VRAM plus only owner-reclaimable resident bytes.
    pub admitted_available_vram_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub determinism_class: DeterminismClass,
    pub execution_environment: ExecutionEnvironmentDescriptor,
    pub execution_equivalence_fingerprint: ExecutionEquivalenceFingerprint,
    /// Exact, device-qualified worker/lease identity. This remains the
    /// authority for residency, grants, cache reconstruction, and provenance.
    pub execution_fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedLora {
    pub path: PathBuf,
    pub content_fingerprint: ContentFingerprint,
    pub scale_bits: u64,
}

impl PlannedLora {
    pub fn scale(&self) -> f64 {
        f64::from_bits(self.scale_bits)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceFact {
    pub id: String,
    pub ordinal: usize,
    pub backend: GpuBackend,
    pub compute_capability: Option<(u16, u16)>,
    pub available_vram_bytes: u64,
}

/// Concrete engine inputs produced by asynchronous dependency preparation.
///
/// The map is keyed by stable runtime device id because mixed-capacity hosts
/// can legitimately select different encoder variants for different GPUs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PreparedExecutionInputs {
    /// Hash of the immutable request/config authority that selected these
    /// concrete dependency variants. An empty value is reserved for synthetic
    /// tests that do not exercise production dependency preparation.
    pub authority_fingerprint: String,
    pub by_device: BTreeMap<String, PreparedDeviceExecutionInputs>,
    /// Request-eligible devices whose dependency materialization failed while
    /// at least one sibling succeeded. These omissions are retryable; they
    /// must not silently become the device set for the lifetime of the job.
    pub retryable_device_failures: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedDeviceExecutionInputs {
    pub engine_paths: ModelPaths,
    pub engine_config: mold_inference::FrozenEngineConfig,
    /// Free VRAM used to resolve an `auto` dependency choice.
    pub prepared_available_vram_bytes: u64,
    /// False for explicit variants, whose choice must not churn with
    /// telemetry.
    pub capacity_sensitive: bool,
}

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ExecutionPlanError {
    #[error("model '{model}' has no concrete local artifact paths")]
    MissingArtifacts { model: String },
    #[error("component {role:?} is pinned to CPU, but family '{family}' does not support that placement")]
    UnsupportedCpuPlacement { family: String, role: ComponentRole },
    #[error("block offload was requested, but family '{family}' does not support it")]
    UnsupportedOffload { family: String },
    #[error("component placement references unavailable device '{0}'")]
    UnavailableDevice(String),
    #[error("component placement spans multiple devices: {0}")]
    CrossDevicePlacement(String),
    #[error("no device has enough effective VRAM capacity for a safe execution plan")]
    InsufficientVram,
    #[error("execution plan was invalidated before CUDA work: {0}")]
    PlanInvalidated(String),
    #[error("prepared execution inputs are stale: {0}")]
    PreparedInputsStale(String),
}

pub fn capabilities_for_family(family: &str) -> PlacementCapabilities {
    // Keep this list capability-based and deliberately conservative. Each
    // `true` corresponds to an implemented engine path, not a scheduler
    // heuristic. Unknown/catalog families receive no automatic CPU/offload.
    match family {
        "flux" => PlacementCapabilities {
            supports_text_encoder_cpu: true,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            supports_block_offload: true,
            supports_tiled_vae: true,
            batch_execution: mold_inference::batch_execution_capability_for_family(family),
        },
        "flux2" | "flux.2" | "flux2-klein" => PlacementCapabilities {
            supports_text_encoder_cpu: true,
            supports_vae_cpu: true,
            supports_audio_components_cpu: false,
            supports_block_offload: true,
            supports_tiled_vae: true,
            batch_execution: mold_inference::batch_execution_capability_for_family(family),
        },
        "ltx2" | "ltx-2" | "ltx2.3" => PlacementCapabilities {
            // Gemma CPU execution is tested; the transformer and video VAE
            // remain on the leased CUDA device.
            supports_text_encoder_cpu: true,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            // The native runtime consumes MOLD_OFFLOAD as a forced full-
            // streaming policy even though its factory does not take the
            // generic FLUX block-offload boolean.
            supports_block_offload: true,
            supports_tiled_vae: true,
            batch_execution: mold_inference::batch_execution_capability_for_family(family),
        },
        "z-image" | "qwen-image" | "qwen-image-edit" | "sd3" | "sd3.5" => PlacementCapabilities {
            supports_text_encoder_cpu: false,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            supports_block_offload: true,
            supports_tiled_vae: matches!(family, "z-image" | "qwen-image" | "qwen-image-edit"),
            batch_execution: mold_inference::batch_execution_capability_for_family(family),
        },
        _ => PlacementCapabilities {
            supports_text_encoder_cpu: false,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            supports_block_offload: false,
            supports_tiled_vae: false,
            batch_execution: mold_inference::batch_execution_capability_for_family(family),
        },
    }
}

/// Resolve hard request/config placement before dependency preparation.
///
/// This is intentionally artifact-only: it filters irrelevant sibling GPUs
/// without consulting CUDA or performing downloads. Full memory admission
/// still happens after dependencies are concrete.
pub fn eligible_devices_for_request(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
) -> Result<Vec<DeviceFact>, ExecutionPlanError> {
    let paths = ModelPaths::resolve(&request.model, config).ok_or_else(|| {
        ExecutionPlanError::MissingArtifacts {
            model: request.model.clone(),
        }
    })?;
    let family = config
        .resolved_model_config(&request.model)
        .family
        .or_else(|| {
            mold_core::manifest::find_manifest(&request.model)
                .map(|manifest| manifest.family.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let capabilities = capabilities_for_family(&family);
    let engine_config = mold_inference::FrozenEngineConfig::resolve(&request.model, config);
    let loras = effective_loras(config, request);
    let artifacts = concrete_artifacts_for_family(&paths, &family, &loras, &engine_config);
    let normalized = config.effective_placement(&request.model, request.placement.as_ref());
    let effective = effective_constraints(&normalized, &artifacts);
    validate_cpu_constraints(&family, &capabilities, &effective)?;
    let hard = hard_device_ids(&effective, devices)?;
    if hard.len() > 1 {
        return Err(ExecutionPlanError::CrossDevicePlacement(
            hard.into_iter().collect::<Vec<_>>().join(", "),
        ));
    }
    let hard = hard.into_iter().next();
    Ok(devices
        .iter()
        .filter(|device| hard.as_ref().is_none_or(|hard| hard == &device.id))
        .cloned()
        .collect())
}

pub fn resolve_execution_plans(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    offload_requested: bool,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    resolve_execution_plans_with_policy(
        config,
        request,
        devices,
        offload_requested,
        None,
        EquivalenceFactPolicy::AllowBlockingWarmup,
    )
}

pub fn resolve_execution_plans_with_prepared(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    offload_requested: bool,
    prepared: Option<&PreparedExecutionInputs>,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    let fact_policy = if prepared.is_some() {
        EquivalenceFactPolicy::CacheOnly
    } else {
        EquivalenceFactPolicy::AllowBlockingWarmup
    };
    resolve_execution_plans_with_policy(
        config,
        request,
        devices,
        offload_requested,
        prepared,
        fact_policy,
    )
}

/// Resolve plans from the scheduler coordinator without waiting on artifact
/// reads, single-flight owners, or admission permits.
///
/// Dependency preparation warms normal generation facts asynchronously. Some
/// coordinator-owned work (notably durable chain stages) has no prepared-input
/// carrier, so the coordinator must fail closed to metadata-bound cache-miss
/// identities instead of performing blocking I/O itself.
pub(crate) fn resolve_execution_plans_for_coordinator(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    offload_requested: bool,
    prepared: Option<&PreparedExecutionInputs>,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    resolve_execution_plans_with_policy(
        config,
        request,
        devices,
        offload_requested,
        prepared,
        EquivalenceFactPolicy::CacheOnly,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EquivalenceFactPolicy {
    AllowBlockingWarmup,
    CacheOnly,
}

fn resolve_execution_plans_with_policy(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    offload_requested: bool,
    prepared: Option<&PreparedExecutionInputs>,
    fact_policy: EquivalenceFactPolicy,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    let paths = ModelPaths::resolve(&request.model, config).ok_or_else(|| {
        ExecutionPlanError::MissingArtifacts {
            model: request.model.clone(),
        }
    })?;
    let family = config
        .resolved_model_config(&request.model)
        .family
        .or_else(|| {
            mold_core::manifest::find_manifest(&request.model)
                .map(|manifest| manifest.family.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let capabilities = capabilities_for_family(&family);
    if offload_requested && !capabilities.supports_block_offload {
        return Err(ExecutionPlanError::UnsupportedOffload { family });
    }

    let normalized = config.effective_placement(&request.model, request.placement.as_ref());
    let effective_loras = effective_loras(config, request);
    let admission_engine_config =
        mold_inference::FrozenEngineConfig::resolve(&request.model, config);
    if let Some(prepared) = prepared {
        let current_authority =
            preparation_authority_fingerprint(config, request, &paths, &admission_engine_config);
        if !prepared.authority_fingerprint.is_empty()
            && prepared.authority_fingerprint != current_authority
        {
            return Err(ExecutionPlanError::PreparedInputsStale(
                "request or configuration changed after dependency preparation".into(),
            ));
        }
    }
    let admission_artifacts =
        concrete_artifacts_for_family(&paths, &family, &effective_loras, &admission_engine_config);
    let constraints = effective_constraints(&normalized, &admission_artifacts);
    validate_cpu_constraints(&family, &capabilities, &constraints)?;
    let hard_devices = hard_device_ids(&constraints, devices)?;
    if hard_devices.len() > 1 {
        return Err(ExecutionPlanError::CrossDevicePlacement(
            hard_devices.into_iter().collect::<Vec<_>>().join(", "),
        ));
    }
    let hard_device = hard_devices.into_iter().next();
    let candidates = devices
        .iter()
        .filter(|device| hard_device.as_ref().is_none_or(|hard| hard == &device.id))
        .filter_map(|device| {
            let inputs = match prepared {
                Some(prepared) => prepared.by_device.get(&device.id).map(|prepared| {
                    (
                        prepared.engine_paths.clone(),
                        prepared.engine_config.clone(),
                    )
                })?,
                None => (paths.clone(), admission_engine_config.clone()),
            };
            let artifacts =
                concrete_artifacts_for_family(&inputs.0, &family, &effective_loras, &inputs.1);
            let effective = effective_constraints(&normalized, &artifacts);
            if let Err(error) = validate_cpu_constraints(&family, &capabilities, &effective) {
                return Some(Err(error));
            }
            let context = PlanContext {
                model: &request.model,
                family: &family,
                capabilities: &capabilities,
                request,
                paths: &inputs.0,
                engine_config: &inputs.1,
                admission_paths: &paths,
                admission_engine_config: &admission_engine_config,
                effective_loras: &effective_loras,
                artifacts: &artifacts,
                effective: &effective,
                offload_requested,
                equivalence_cache_only: fact_policy == EquivalenceFactPolicy::CacheOnly,
            };
            build_plan(&context, device)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if candidates.is_empty() {
        return Err(ExecutionPlanError::InsufficientVram);
    }
    Ok(candidates)
}

pub(crate) fn preparation_authority_fingerprint(
    config: &Config,
    request: &GenerateRequest,
    paths: &ModelPaths,
    engine_config: &mold_inference::FrozenEngineConfig,
) -> String {
    let mut normalized_request = request.clone();
    // Local prompt expansion is part of the same dependency-preparation
    // transaction and intentionally changes only these fields.
    normalized_request.prompt.clear();
    normalized_request.original_prompt = None;
    normalized_request.expand = None;

    let mut hash = Sha256::new();
    hash.update(format!("{paths:?}").as_bytes());
    hash.update(format!("{engine_config:?}").as_bytes());
    hash.update(
        format!(
            "{:?}",
            config.effective_placement(&request.model, request.placement.as_ref())
        )
        .as_bytes(),
    );
    hash.update(format!("{:?}", effective_loras(config, request)).as_bytes());
    hash.update(
        serde_json::to_vec(&normalized_request)
            .expect("GenerateRequest serialization is infallible")
            .as_slice(),
    );
    format!("{:x}", hash.finalize())
}

/// Populate metadata-bound byte-identity and format-fact caches during
/// asynchronous dependency preparation.
///
/// Execution-plan resolution runs in the scheduler coordinator and must only
/// perform metadata checks plus cache lookups for normal prepared jobs. The
/// caller is responsible for invoking this function through
/// `tokio::task::spawn_blocking`, because hashing and header probing a
/// multi-gigabyte checkpoint are blocking file I/O and CPU work.
pub(crate) fn warm_execution_equivalence_cache(
    config: &Config,
    request: &GenerateRequest,
    prepared: &PreparedExecutionInputs,
) {
    let family = config
        .resolved_model_config(&request.model)
        .family
        .or_else(|| {
            mold_core::manifest::find_manifest(&request.model)
                .map(|manifest| manifest.family.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let loras = effective_loras(config, request);
    let mut paths = BTreeSet::new();
    for inputs in prepared.by_device.values() {
        paths.extend(
            concrete_artifacts_for_family(
                &inputs.engine_paths,
                &family,
                &loras,
                &inputs.engine_config,
            )
            .into_values(),
        );
    }
    for path in paths {
        let _ = artifact_facts_path_with_policy(&path, false);
    }
}

pub fn validate_before_cuda(
    plan: &ResolvedExecutionPlan,
    worker_device_id: &str,
    worker_ordinal: usize,
    config: &Config,
    request: &GenerateRequest,
) -> Result<(), ExecutionPlanError> {
    if plan.device_id != worker_device_id || plan.device_ordinal != worker_ordinal {
        return Err(ExecutionPlanError::PlanInvalidated(format!(
            "lease targets {worker_device_id}/gpu:{worker_ordinal}, plan targets {}/gpu:{}",
            plan.device_id, plan.device_ordinal
        )));
    }
    let model = request.model.as_str();
    let current_paths = ModelPaths::resolve(model, config).ok_or_else(|| {
        ExecutionPlanError::PlanInvalidated("model paths are no longer resolvable".into())
    })?;
    let current_loras = effective_loras(config, request);
    let current_engine_config = mold_inference::FrozenEngineConfig::resolve(model, config);
    if current_paths != plan.admission_paths
        || current_engine_config != plan.admission_engine_config
        || current_loras != plan.effective_loras
    {
        return Err(ExecutionPlanError::PlanInvalidated(
            "frozen engine paths, config, or LoRA stack changed after admission".into(),
        ));
    }
    for component in plan.components.values() {
        let current = fingerprint_path(&component.artifact_path);
        if current != component.content_fingerprint {
            return Err(ExecutionPlanError::PlanInvalidated(format!(
                "artifact '{}' changed after admission",
                component.artifact_path.display()
            )));
        }
        if let ResolvedComponentPlacement::Device(device_id) = &component.placement {
            if device_id != worker_device_id {
                return Err(ExecutionPlanError::PlanInvalidated(format!(
                    "component {:?} targets sibling device {device_id}",
                    component.role
                )));
            }
        }
    }
    Ok(())
}

/// Convert the selected concrete component plan back into the current engine
/// placement contract. This removes runtime Auto decisions: the engine sees
/// exactly CPU or the leased device for every exposed component knob.
pub fn materialized_placement(plan: &ResolvedExecutionPlan) -> DevicePlacement {
    let component_ref = |role: &ComponentRole| {
        plan.components
            .get(role)
            .map(|component| match &component.placement {
                ResolvedComponentPlacement::Cpu => DeviceRef::Cpu,
                ResolvedComponentPlacement::Device(id) => DeviceRef::device(id.clone()),
            })
            .unwrap_or(DeviceRef::Auto)
    };
    let role_ref = |predicate: &dyn Fn(&ComponentRole) -> bool| {
        plan.components
            .iter()
            .find(|(role, _)| predicate(role))
            .map(|(_, component)| match &component.placement {
                ResolvedComponentPlacement::Cpu => DeviceRef::Cpu,
                ResolvedComponentPlacement::Device(id) => DeviceRef::device(id.clone()),
            })
    };
    let text = role_ref(&|role| {
        matches!(
            role,
            ComponentRole::GemmaShard(_) | ComponentRole::GenericTextEncoderShard(_)
        )
    })
    .or_else(|| role_ref(&ComponentRole::is_text_encoder))
    .unwrap_or(DeviceRef::Auto);
    DevicePlacement {
        text_encoders: text.clone(),
        advanced: Some(mold_core::types::AdvancedPlacement {
            transformer: component_ref(&ComponentRole::Transformer),
            vae: component_ref(&ComponentRole::Vae),
            clip_l: role_ref(&|role| matches!(role, ComponentRole::ClipL)),
            clip_g: role_ref(&|role| matches!(role, ComponentRole::ClipG)),
            t5: role_ref(&|role| matches!(role, ComponentRole::T5)),
            qwen: role_ref(&|role| matches!(role, ComponentRole::QwenShard(_))),
        }),
    }
}

/// Apply the request-shaping portion of a selected plan. This freezes the
/// ordered default/request LoRA stack in the payload actually consumed by the
/// engine; later config edits cannot inject or reorder adapters.
pub fn materialize_request(plan: &ResolvedExecutionPlan, request: &mut GenerateRequest) {
    request.placement = Some(materialized_placement(plan));
    let loras = plan
        .effective_loras
        .iter()
        .map(|lora| mold_core::LoraWeight {
            path: lora.path.to_string_lossy().into_owned(),
            scale: lora.scale(),
        })
        .collect::<Vec<_>>();
    request.lora = None;
    request.loras = (!loras.is_empty()).then_some(loras);
}

fn concrete_artifacts_for_family(
    paths: &ModelPaths,
    family: &str,
    effective_loras: &[PlannedLora],
    engine_config: &mold_inference::FrozenEngineConfig,
) -> BTreeMap<ComponentRole, PathBuf> {
    let mut artifacts = BTreeMap::new();
    // LTX-2 audio VAE and vocoder tensors are namespaces inside the primary
    // transformer/checkpoint safetensors, not independently addressable
    // ModelPaths. They therefore share this artifact, device placement, and
    // content fingerprint. Do not invent duplicate component paths: the
    // runtime currently cannot place either audio namespace on CPU
    // (`supports_audio_components_cpu == false`).
    artifacts.insert(ComponentRole::Transformer, paths.transformer.clone());
    for (index, shard) in paths.transformer_shards.iter().enumerate() {
        artifacts.insert(ComponentRole::TransformerShard(index as u8), shard.clone());
    }
    artifacts.insert(ComponentRole::Vae, paths.vae.clone());
    if let Some(path) = engine_config
        .selected_t5_path
        .as_ref()
        .or(paths.t5_encoder.as_ref())
    {
        artifacts.insert(ComponentRole::T5, path.clone());
    }
    if let Some(path) = &paths.t5_tokenizer {
        artifacts.insert(ComponentRole::T5Tokenizer, path.clone());
    }
    if let Some(path) = &paths.clip_encoder {
        artifacts.insert(ComponentRole::ClipL, path.clone());
    }
    if let Some(path) = &paths.clip_tokenizer {
        artifacts.insert(ComponentRole::ClipLTokenizer, path.clone());
    }
    if let Some(path) = &paths.clip_encoder_2 {
        artifacts.insert(ComponentRole::ClipG, path.clone());
    }
    if let Some(path) = &paths.clip_tokenizer_2 {
        artifacts.insert(ComponentRole::ClipGTokenizer, path.clone());
    }
    let selected_text_paths = if !engine_config.selected_qwen3_paths.is_empty() {
        engine_config.selected_qwen3_paths.clone()
    } else if let Some(path) = engine_config.selected_qwen2_path.as_ref() {
        // Qwen-Image-Edit still consumes the native multimodal shards for
        // vision even when its language path is the selected GGUF.
        std::iter::once(path.clone())
            .chain(
                paths
                    .text_encoder_files
                    .iter()
                    .filter(|candidate| *candidate != path)
                    .cloned(),
            )
            .collect()
    } else if !engine_config.selected_gemma_paths.is_empty() {
        // `text_encoder_files` also carries the Gemma tokenizer anchor and
        // optional LTX-2.3 text projection. Keep those host artifacts beside
        // the exact selected Gemma weight files.
        engine_config
            .selected_gemma_paths
            .iter()
            .cloned()
            .chain(paths.text_encoder_files.iter().cloned())
            .collect()
    } else {
        paths.text_encoder_files.clone()
    };
    for (index, path) in selected_text_paths.iter().enumerate() {
        let index = index as u16;
        let role = match family {
            "ltx2" | "ltx-2" | "ltx2.3" => ComponentRole::GemmaShard(index),
            "qwen-image" | "qwen-image-edit" | "z-image" | "flux2" | "flux.2" | "flux2-klein" => {
                ComponentRole::QwenShard(index)
            }
            _ => ComponentRole::GenericTextEncoderShard(index),
        };
        artifacts.insert(role, path.clone());
    }
    if let Some(path) = &paths.text_tokenizer {
        artifacts.insert(ComponentRole::TextTokenizer, path.clone());
    }
    if let Some(path) = &paths.spatial_upscaler {
        artifacts.insert(ComponentRole::SpatialUpscaler, path.clone());
    }
    if let Some(path) = &paths.temporal_upscaler {
        artifacts.insert(ComponentRole::TemporalUpscaler, path.clone());
    }
    if let Some(path) = &paths.decoder {
        artifacts.insert(ComponentRole::Decoder, path.clone());
    }
    if let Some(path) = &paths.distilled_lora {
        artifacts.insert(ComponentRole::DistilledLora, path.clone());
    }
    for (index, lora) in effective_loras.iter().enumerate() {
        artifacts.insert(ComponentRole::Lora(index as u16), lora.path.clone());
    }
    artifacts
}

fn effective_loras(config: &Config, request: &GenerateRequest) -> Vec<PlannedLora> {
    const ZERO_SCALE_EPS: f64 = 1e-8;
    let requested = request
        .loras
        .as_ref()
        .filter(|stack| !stack.is_empty())
        .cloned()
        .or_else(|| request.lora.clone().map(|lora| vec![lora]))
        .or_else(|| {
            config
                .resolved_model_config(&request.model)
                .effective_lora()
                .map(|(path, scale)| mold_core::LoraWeight { path, scale })
                .map(|lora| vec![lora])
        })
        .unwrap_or_default();
    requested
        .into_iter()
        .filter(|lora| lora.scale.abs() > ZERO_SCALE_EPS)
        .map(|lora| {
            let path = PathBuf::from(lora.path);
            PlannedLora {
                content_fingerprint: fingerprint_path(&path),
                path,
                scale_bits: lora.scale.to_bits(),
            }
        })
        .collect()
}

fn effective_constraints(
    placement: &DevicePlacement,
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
) -> EffectivePlacement {
    let advanced = placement.advanced.as_ref();
    let mut components = BTreeMap::new();
    for role in artifacts.keys() {
        let requested = match role {
            ComponentRole::Transformer | ComponentRole::TransformerShard(_) => {
                advanced.map(|value| &value.transformer)
            }
            ComponentRole::Vae => advanced.map(|value| &value.vae),
            ComponentRole::T5 => advanced
                .and_then(|value| value.t5.as_ref())
                .or(Some(&placement.text_encoders)),
            ComponentRole::ClipL => advanced
                .and_then(|value| value.clip_l.as_ref())
                .or(Some(&placement.text_encoders)),
            ComponentRole::ClipG => advanced
                .and_then(|value| value.clip_g.as_ref())
                .or(Some(&placement.text_encoders)),
            ComponentRole::QwenShard(_) => advanced
                .and_then(|value| value.qwen.as_ref())
                .or(Some(&placement.text_encoders)),
            ComponentRole::GemmaShard(_) | ComponentRole::GenericTextEncoderShard(_) => {
                Some(&placement.text_encoders)
            }
            role if role.is_host_only() => Some(&DeviceRef::Cpu),
            _ => None,
        }
        .unwrap_or(&DeviceRef::Auto);
        components.insert(role.clone(), constraint_from_ref(requested));
    }
    EffectivePlacement { components }
}

fn constraint_from_ref(reference: &DeviceRef) -> ResolvedComponentConstraint {
    match reference {
        DeviceRef::Auto => ResolvedComponentConstraint::Auto,
        DeviceRef::Cpu => ResolvedComponentConstraint::Cpu,
        DeviceRef::Gpu { ordinal } => {
            ResolvedComponentConstraint::Device(format!("ordinal:{ordinal}"))
        }
        DeviceRef::Device { id } => ResolvedComponentConstraint::Device(id.clone()),
    }
}

fn validate_cpu_constraints(
    family: &str,
    capabilities: &PlacementCapabilities,
    placement: &EffectivePlacement,
) -> Result<(), ExecutionPlanError> {
    for (role, constraint) in &placement.components {
        if constraint != &ResolvedComponentConstraint::Cpu {
            continue;
        }
        let supported = match role {
            role if role.is_host_only() => true,
            role if role.is_text_encoder() => capabilities.supports_text_encoder_cpu,
            ComponentRole::Vae => capabilities.supports_vae_cpu,
            ComponentRole::Transformer | ComponentRole::TransformerShard(_) => {
                matches!(family, "flux2" | "flux.2" | "flux2-klein")
            }
            _ => false,
        };
        if !supported {
            return Err(ExecutionPlanError::UnsupportedCpuPlacement {
                family: family.to_string(),
                role: role.clone(),
            });
        }
    }
    Ok(())
}

fn hard_device_ids(
    placement: &EffectivePlacement,
    devices: &[DeviceFact],
) -> Result<BTreeSet<String>, ExecutionPlanError> {
    placement
        .components
        .values()
        .filter_map(|constraint| match constraint {
            ResolvedComponentConstraint::Device(id) => Some(id),
            _ => None,
        })
        .map(|id| {
            if let Some(ordinal) = id.strip_prefix("ordinal:") {
                let ordinal = ordinal
                    .parse::<usize>()
                    .map_err(|_| ExecutionPlanError::UnavailableDevice(id.clone()))?;
                devices
                    .iter()
                    .find(|device| device.ordinal == ordinal)
                    .map(|device| device.id.clone())
                    .ok_or_else(|| ExecutionPlanError::UnavailableDevice(id.clone()))
            } else {
                devices
                    .iter()
                    .find(|device| device.id == *id)
                    .map(|device| device.id.clone())
                    .ok_or_else(|| ExecutionPlanError::UnavailableDevice(id.clone()))
            }
        })
        .collect()
}

struct PlanContext<'a> {
    model: &'a str,
    family: &'a str,
    capabilities: &'a PlacementCapabilities,
    request: &'a GenerateRequest,
    paths: &'a ModelPaths,
    engine_config: &'a mold_inference::FrozenEngineConfig,
    admission_paths: &'a ModelPaths,
    admission_engine_config: &'a mold_inference::FrozenEngineConfig,
    effective_loras: &'a [PlannedLora],
    artifacts: &'a BTreeMap<ComponentRole, PathBuf>,
    effective: &'a EffectivePlacement,
    offload_requested: bool,
    /// Prepared production jobs have already hashed every artifact on the
    /// blocking pool. A cache miss must fail closed instead of reading model
    /// bytes on the coordinator thread.
    equivalence_cache_only: bool,
}

fn build_plan(
    context: &PlanContext<'_>,
    device: &DeviceFact,
) -> Option<Result<ResolvedExecutionPlan, ExecutionPlanError>> {
    // These compatibility tokens preserve candidate-v1 exact lease identity.
    // They are never used as precision authority or equivalence facts.
    let exact_quantization = exact_v1_compatibility_quantization(context.model, context.artifacts);
    let exact_dtype = exact_v1_compatibility_dtype(context.model);
    let hint = Some(crate::memory_preflight::ActivationHint::from_request(
        context.request,
        context.family,
    ));
    let request_has_lora = !context.effective_loras.is_empty();
    let gemma_placement = mold_inference::device::resolve_ltx2_gemma_device_override_from_values(
        context
            .engine_config
            .runtime_environment
            .value("MOLD_LTX2_GEMMA_DEVICE"),
        context
            .engine_config
            .runtime_environment
            .value("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER"),
        device.ordinal,
    );
    let gemma_competes = matches!(
        gemma_placement,
        Some(mold_inference::device::LtxGemmaPlacement::Gpu(ordinal))
            if ordinal == device.ordinal
    );
    let initial_memory = crate::memory_preflight::estimate_generation_memory_for_request(
        context.request,
        context.paths,
        hint,
        Some(device.available_vram_bytes),
        context.offload_requested,
        request_has_lora,
        gemma_competes,
    );
    // A process-wide offload preference is advisory for concrete formats
    // which cannot honor it (for example Flux.2 GGUF/NVFP4 or a LoRA merge).
    // The family capability gate above remains a typed error; this path-level
    // policy exactly matches what engine construction will consume.
    let auto_cpu_text =
        initial_memory.under_memory_pressure && context.capabilities.supports_text_encoder_cpu;

    let placements = context
        .artifacts
        .keys()
        .map(|role| {
            let constraint = context
                .effective
                .components
                .get(role)
                .cloned()
                .unwrap_or(ResolvedComponentConstraint::Auto);
            let cpu = role.is_host_only()
                || constraint == ResolvedComponentConstraint::Cpu
                || (auto_cpu_text && role.is_text_encoder());
            (role.clone(), cpu)
        })
        .collect::<BTreeMap<_, _>>();
    let transformer_on_cpu = placements
        .iter()
        .filter(|(role, _)| {
            matches!(
                role,
                ComponentRole::Transformer | ComponentRole::TransformerShard(_)
            )
        })
        .all(|(_, cpu)| *cpu);
    let gpu_paths = gpu_resident_paths(context.paths, &placements);
    let memory = crate::memory_preflight::estimate_generation_memory_for_request(
        context.request,
        &gpu_paths,
        hint,
        Some(device.available_vram_bytes),
        initial_memory.block_offload && !transformer_on_cpu,
        request_has_lora,
        gemma_competes,
    );
    if memory.fits_available_memory != Some(true) {
        return None;
    }

    let mut components = BTreeMap::new();
    let mut host_paths = BTreeSet::new();
    for (role, path) in context.artifacts {
        let place_cpu = placements.get(role).copied().unwrap_or(false);
        let bytes = artifact_size(path);
        let (placement, load_strategy, vram, host) = if place_cpu {
            host_paths.insert(path.clone());
            (
                ResolvedComponentPlacement::Cpu,
                ComponentLoadStrategy::ParkedCpu,
                0,
                bytes,
            )
        } else {
            let strategy = if memory.block_offload
                && matches!(
                    role,
                    ComponentRole::Transformer | ComponentRole::TransformerShard(_)
                ) {
                host_paths.insert(path.clone());
                ComponentLoadStrategy::StreamedBlocks
            } else if role.is_text_encoder() {
                ComponentLoadStrategy::DropReload
            } else {
                ComponentLoadStrategy::Resident
            };
            (
                ResolvedComponentPlacement::Device(device.id.clone()),
                strategy,
                bytes,
                0,
            )
        };
        components.insert(
            role.clone(),
            ComponentExecutionPlan {
                role: role.clone(),
                artifact_path: path.clone(),
                content_fingerprint: fingerprint_path(path),
                dtype: (!role.is_host_only()).then_some(exact_dtype).flatten(),
                quantization: (!role.is_host_only())
                    .then_some(exact_quantization)
                    .flatten(),
                placement,
                load_strategy,
                predicted_vram_bytes: vram,
                predicted_host_bytes: host,
            },
        );
    }

    let predicted_vram = memory.peak_memory_bytes;
    let predicted_host = host_paths.iter().fold(BASE_HOST_TRANSIENT, |total, path| {
        total.saturating_add(artifact_size(path))
    });
    let fingerprint = execution_fingerprint(
        context.model,
        device,
        context.effective,
        &components,
        context.engine_config,
        context.effective_loras,
        memory.block_offload,
    );
    let model_fingerprint = model_fingerprint(context.model, context.artifacts);
    let equivalence_model_fingerprint =
        equivalence_model_fingerprint(context.artifacts, context.equivalence_cache_only);
    let attention_backend = match context.engine_config.attention_backend {
        mold_inference::attention::AttentionBackend::Math => AttentionBackend::Math,
        mold_inference::attention::AttentionBackend::Flash => AttentionBackend::Flash,
    };
    let offload_mode = if memory.block_offload {
        OffloadMode::Block
    } else {
        OffloadMode::None
    };
    let determinism_class = DeterminismClass::CpuSeededCrossBackend;
    let execution_environment = execution_environment_descriptor(
        device,
        context.model,
        context.family,
        &equivalence_model_fingerprint,
        &components,
        context.effective_loras,
        context.engine_config,
        attention_backend,
        memory.load_strategy,
        offload_mode,
        context.request.resolved_output_format(),
        determinism_class,
        context.equivalence_cache_only,
    );
    let execution_equivalence_fingerprint = execution_environment.fingerprint();
    Some(Ok(ResolvedExecutionPlan {
        device_id: device.id.clone(),
        device_ordinal: device.ordinal,
        model_family: context.family.to_string(),
        model_fingerprint,
        effective_placement: context.effective.clone(),
        components,
        engine_paths: context.paths.clone(),
        engine_config: context.engine_config.clone(),
        admission_paths: context.admission_paths.clone(),
        admission_engine_config: context.admission_engine_config.clone(),
        effective_loras: context.effective_loras.to_vec(),
        attention_backend,
        engine_load_strategy: memory.load_strategy,
        offload_mode,
        predicted_vram_peak_bytes: predicted_vram,
        admitted_available_vram_bytes: device.available_vram_bytes,
        predicted_host_increment_bytes: predicted_host,
        determinism_class,
        execution_environment,
        execution_equivalence_fingerprint,
        execution_fingerprint: fingerprint,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execution_environment_descriptor(
    device: &DeviceFact,
    runtime_model_id: &str,
    model_family: &str,
    model_fingerprint: &str,
    components: &BTreeMap<ComponentRole, ComponentExecutionPlan>,
    effective_loras: &[PlannedLora],
    engine_config: &mold_inference::FrozenEngineConfig,
    attention_backend: AttentionBackend,
    engine_load_strategy: mold_inference::LoadStrategy,
    offload_mode: OffloadMode,
    output_format: OutputFormat,
    determinism_class: DeterminismClass,
    equivalence_cache_only: bool,
) -> ExecutionEnvironmentDescriptor {
    let architecture = match (device.backend, device.compute_capability) {
        (GpuBackend::Cuda, Some((major, minor))) => {
            DeviceArchitectureClass::CudaComputeCapability { major, minor }
        }
        (GpuBackend::Metal, _) => DeviceArchitectureClass::MetalDefault,
        (backend, None) => DeviceArchitectureClass::Unknown {
            backend,
            device_id: device.id.clone(),
        },
    };
    let single_file = components
        .get(&ComponentRole::Transformer)
        .zip(components.get(&ComponentRole::Vae))
        .is_some_and(|(transformer, vae)| transformer.artifact_path == vae.artifact_path);
    let runtime_artifact_paths = components
        .iter()
        .map(|(role, component)| RuntimeArtifactPathIdentity {
            role: role.clone(),
            path: runtime_path_identity(&component.artifact_path),
        })
        .collect();
    let components = components
        .iter()
        .map(|(role, component)| {
            let facts =
                artifact_facts_path_with_policy(&component.artifact_path, equivalence_cache_only);
            let content_fingerprint = facts.content.clone();
            let storage = component_storage_format(&facts);
            let precision = effective_component_precision(
                model_family,
                role,
                &component.placement,
                device.backend,
                engine_config.vae_dtype,
                single_file,
                storage,
            );
            let dtype = match precision.compute_dtype {
                EffectiveComponentDType::Bf16 => Some(PlannedDType::Bf16),
                EffectiveComponentDType::F16 => Some(PlannedDType::F16),
                EffectiveComponentDType::F32 => Some(PlannedDType::F32),
                EffectiveComponentDType::NotApplicable
                | EffectiveComponentDType::QuantizedNative
                | EffectiveComponentDType::Unknown => None,
            };
            let quantization = quantization_from_storage(&precision.storage);
            EquivalentComponentExecution {
                role: role.clone(),
                content_fingerprint,
                precision,
                dtype,
                quantization,
                placement: match &component.placement {
                    ResolvedComponentPlacement::Cpu => SemanticComponentPlacement::Cpu,
                    ResolvedComponentPlacement::Device(_) => {
                        SemanticComponentPlacement::AssignedDevice
                    }
                },
                load_strategy: component.load_strategy,
            }
        })
        .collect();
    let loras = effective_loras
        .iter()
        .map(|lora| EquivalentLoraExecution {
            content_fingerprint: equivalence_fingerprint_path_with_policy(
                &lora.path,
                equivalence_cache_only,
            ),
            scale_bits: lora.scale_bits,
        })
        .collect();
    ExecutionEnvironmentDescriptor {
        schema_version: 3,
        backend: device.backend,
        architecture,
        attention_kernel_class: match attention_backend {
            AttentionBackend::Math => AttentionKernelClass::Math,
            AttentionBackend::Flash => AttentionKernelClass::Flash,
        },
        code: execution_code_identity(),
        semantic_config: ExecutionSemanticConfig::from_frozen(engine_config),
        runtime_model_id: runtime_model_id.to_string(),
        runtime_artifact_paths,
        model_family: model_family.to_string(),
        model_fingerprint: model_fingerprint.to_string(),
        components,
        loras,
        engine_load_strategy: match engine_load_strategy {
            mold_inference::LoadStrategy::Eager => EngineLoadStrategyClass::Eager,
            mold_inference::LoadStrategy::Sequential => EngineLoadStrategyClass::Sequential,
        },
        offload_mode,
        output_format,
        determinism_class,
    }
}

fn gpu_resident_paths(
    paths: &ModelPaths,
    placements: &BTreeMap<ComponentRole, bool>,
) -> ModelPaths {
    let mut gpu = paths.clone();
    let on_cpu = |role: &ComponentRole| placements.get(role).copied().unwrap_or(false);

    if on_cpu(&ComponentRole::Transformer) {
        gpu.transformer = PathBuf::new();
    }
    if !gpu.transformer_shards.is_empty() {
        gpu.transformer_shards = gpu
            .transformer_shards
            .into_iter()
            .enumerate()
            .filter_map(|(index, path)| {
                (!on_cpu(&ComponentRole::TransformerShard(index as u8))).then_some(path)
            })
            .collect();
    }
    if on_cpu(&ComponentRole::Vae) {
        gpu.vae = PathBuf::new();
    }

    if on_cpu(&ComponentRole::T5) {
        gpu.t5_encoder = None;
    }
    if on_cpu(&ComponentRole::ClipL) {
        gpu.clip_encoder = None;
    }
    if on_cpu(&ComponentRole::ClipG) {
        gpu.clip_encoder_2 = None;
    }
    gpu.text_encoder_files = paths
        .text_encoder_files
        .iter()
        .enumerate()
        .filter_map(|(index, path)| {
            let index = index as u16;
            let on_cpu_for_family = [
                ComponentRole::QwenShard(index),
                ComponentRole::GemmaShard(index),
                ComponentRole::GenericTextEncoderShard(index),
            ]
            .iter()
            .any(&on_cpu);
            (!on_cpu_for_family).then_some(path.clone())
        })
        .collect();
    gpu
}

fn artifact_size(path: &Path) -> u64 {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(UNKNOWN_ARTIFACT_HOST_CHARGE)
}

fn component_storage_format(facts: &ArtifactFacts) -> ComponentStorageFormat {
    match &facts.format {
        ArtifactFormatFact::Known(format) => ComponentStorageFormat::Known(format.clone()),
        ArtifactFormatFact::ProbeFailure(reason) => {
            use mold_inference::artifact_format::ArtifactProbeFailure;
            let reason = match reason {
                ArtifactProbeFailure::Io => ArtifactFormatUnknown::Io,
                ArtifactProbeFailure::UnsupportedContainer => {
                    ArtifactFormatUnknown::UnsupportedContainer
                }
                ArtifactProbeFailure::InvalidHeader => ArtifactFormatUnknown::InvalidHeader,
                ArtifactProbeFailure::UnsupportedTensorDType => {
                    ArtifactFormatUnknown::UnsupportedTensorDType
                }
                ArtifactProbeFailure::UnsupportedGgufTensorFormat => {
                    ArtifactFormatUnknown::UnsupportedGgufTensorFormat
                }
            };
            ComponentStorageFormat::Unknown {
                reason,
                content_discriminator: facts.content.clone(),
            }
        }
        ArtifactFormatFact::CacheMiss => ComponentStorageFormat::Unknown {
            reason: ArtifactFormatUnknown::CacheMiss,
            content_discriminator: facts.content.clone(),
        },
    }
}

/// Candidate-v1 exact fingerprint compatibility only.
///
/// The rejected candidate encoded these filename/model tokens inside the
/// device/path-qualified lease fingerprint. Removing them would invalidate
/// grants reconstructed from that exact identity. They are deliberately
/// quarantined here and never exposed as component precision or equivalence
/// authority; authoritative facts come only from `component_storage_format`.
fn exact_v1_compatibility_quantization(
    model: &str,
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
) -> Option<QuantizationVariant> {
    let joined = format!(
        "{} {}",
        model.to_ascii_lowercase(),
        artifacts
            .iter()
            .filter(|(role, _)| !role.is_host_only())
            .map(|(_, path)| path.to_string_lossy().to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if joined.contains("q4") {
        Some(QuantizationVariant::Q4)
    } else if joined.contains("q8") {
        Some(QuantizationVariant::Q8)
    } else if joined.contains("fp8") {
        Some(QuantizationVariant::Fp8)
    } else {
        None
    }
}

/// Candidate-v1 exact fingerprint compatibility only; see
/// `exact_v1_compatibility_quantization`.
fn exact_v1_compatibility_dtype(model: &str) -> Option<PlannedDType> {
    let lower = model.to_ascii_lowercase();
    if lower.contains("bf16") {
        Some(PlannedDType::Bf16)
    } else if lower.contains("fp16") {
        Some(PlannedDType::F16)
    } else {
        None
    }
}

fn quantization_from_storage(storage: &ComponentStorageFormat) -> Option<QuantizationVariant> {
    use mold_inference::artifact_format::{
        ArtifactStorageFormat, GgufTensorFormat, SafetensorsEncoding, TensorDType,
    };
    match storage {
        ComponentStorageFormat::Known(ArtifactStorageFormat::Safetensors {
            encoding: SafetensorsEncoding::Nvfp4,
            ..
        }) => Some(QuantizationVariant::Nvfp4),
        ComponentStorageFormat::Known(ArtifactStorageFormat::Safetensors {
            encoding: SafetensorsEncoding::ConvRotW4A4,
            ..
        }) => Some(QuantizationVariant::ConvRotW4A4),
        ComponentStorageFormat::Known(ArtifactStorageFormat::Safetensors {
            tensor_dtypes, ..
        }) if tensor_dtypes
            .iter()
            .any(|dtype| matches!(dtype, TensorDType::F8E4M3 | TensorDType::F8E5M2)) =>
        {
            Some(QuantizationVariant::Fp8)
        }
        ComponentStorageFormat::Known(ArtifactStorageFormat::Gguf { tensor_formats }) => {
            let has =
                |predicate: fn(&GgufTensorFormat) -> bool| tensor_formats.iter().any(predicate);
            if has(|format| matches!(format, GgufTensorFormat::Q2K)) {
                Some(QuantizationVariant::Q2)
            } else if has(|format| matches!(format, GgufTensorFormat::Q3K)) {
                Some(QuantizationVariant::Q3)
            } else if has(|format| {
                matches!(
                    format,
                    GgufTensorFormat::Q4_0 | GgufTensorFormat::Q4_1 | GgufTensorFormat::Q4K
                )
            }) {
                Some(QuantizationVariant::Q4)
            } else if has(|format| {
                matches!(
                    format,
                    GgufTensorFormat::Q5_0 | GgufTensorFormat::Q5_1 | GgufTensorFormat::Q5K
                )
            }) {
                Some(QuantizationVariant::Q5)
            } else if has(|format| matches!(format, GgufTensorFormat::Q6K)) {
                Some(QuantizationVariant::Q6)
            } else if has(|format| {
                matches!(
                    format,
                    GgufTensorFormat::Q8_0 | GgufTensorFormat::Q8_1 | GgufTensorFormat::Q8K
                )
            }) {
                Some(QuantizationVariant::Q8)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn effective_component_precision(
    family: &str,
    role: &ComponentRole,
    placement: &ResolvedComponentPlacement,
    backend: GpuBackend,
    vae_dtype: mold_inference::device::VaeDtypePolicy,
    single_file: bool,
    storage: ComponentStorageFormat,
) -> EffectiveComponentPrecision {
    use mold_inference::artifact_format::{
        ArtifactStorageFormat, SafetensorsEncoding, TensorDType,
    };

    let quantized = quantization_from_storage(&storage).is_some();
    let compute_dtype = if role.is_host_only() {
        EffectiveComponentDType::NotApplicable
    } else if matches!(placement, ResolvedComponentPlacement::Cpu) {
        if quantized && !matches!(role, ComponentRole::Vae) {
            EffectiveComponentDType::QuantizedNative
        } else {
            EffectiveComponentDType::F32
        }
    } else if matches!(role, ComponentRole::Vae) {
        match vae_dtype {
            mold_inference::device::VaeDtypePolicy::Bf16 => EffectiveComponentDType::Bf16,
            mold_inference::device::VaeDtypePolicy::F16 => EffectiveComponentDType::F16,
            mold_inference::device::VaeDtypePolicy::F32 => EffectiveComponentDType::F32,
            mold_inference::device::VaeDtypePolicy::Auto => {
                if matches!(family, "qwen-image" | "qwen-image-edit")
                    || (family == "sdxl" && single_file)
                {
                    EffectiveComponentDType::F32
                } else if matches!(family, "sd15" | "sdxl" | "sd3" | "wuerstchen") {
                    EffectiveComponentDType::F16
                } else if backend == GpuBackend::Cuda {
                    EffectiveComponentDType::Bf16
                } else {
                    EffectiveComponentDType::F32
                }
            }
        }
    } else if matches!(
        &storage,
        ComponentStorageFormat::Known(ArtifactStorageFormat::Safetensors {
            encoding: SafetensorsEncoding::Nvfp4 | SafetensorsEncoding::ConvRotW4A4,
            ..
        })
    ) {
        if backend == GpuBackend::Cuda {
            EffectiveComponentDType::Bf16
        } else {
            EffectiveComponentDType::F32
        }
    } else if family == "flux"
        && backend == GpuBackend::Cuda
        && matches!(
            &storage,
            ComponentStorageFormat::Known(ArtifactStorageFormat::Safetensors {
                tensor_dtypes,
                ..
            }) if tensor_dtypes.contains(&TensorDType::F8E4M3)
        )
    {
        EffectiveComponentDType::F16
    } else if quantized {
        EffectiveComponentDType::QuantizedNative
    } else if matches!(family, "sd15" | "sdxl" | "sd3" | "wuerstchen") {
        EffectiveComponentDType::F16
    } else if backend == GpuBackend::Cuda {
        EffectiveComponentDType::Bf16
    } else {
        EffectiveComponentDType::F32
    };
    EffectiveComponentPrecision {
        storage,
        compute_dtype,
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ArtifactMetadataIdentity {
    len: u64,
    platform_identity: Vec<u64>,
}

fn artifact_metadata_identity(
    _path: &Path,
    metadata: &std::fs::Metadata,
) -> ArtifactMetadataIdentity {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ArtifactMetadataIdentity {
            len: metadata.len(),
            // inode/file-id plus ctime detects replacement and in-place
            // mutation even when a caller restores size and mtime.
            platform_identity: vec![
                metadata.dev(),
                metadata.ino(),
                metadata.ctime() as u64,
                metadata.ctime_nsec() as u64,
            ],
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            FileBasicInfo, GetFileInformationByHandleEx, FILE_BASIC_INFO,
        };
        let change_time = std::fs::File::open(_path)
            .ok()
            .and_then(|file| {
                let mut info = std::mem::MaybeUninit::<FILE_BASIC_INFO>::uninit();
                // SAFETY: `file` owns a valid handle for the duration of the
                // call and `info` is correctly sized/aligned for FileBasicInfo.
                let result = unsafe {
                    GetFileInformationByHandleEx(
                        file.as_raw_handle() as _,
                        FileBasicInfo,
                        info.as_mut_ptr().cast(),
                        std::mem::size_of::<FILE_BASIC_INFO>() as u32,
                    )
                };
                // SAFETY: Win32 initializes the entire FILE_BASIC_INFO on a
                // nonzero result.
                (result != 0).then(|| unsafe { info.assume_init().ChangeTime as u64 })
            })
            .unwrap_or(0);
        ArtifactMetadataIdentity {
            len: metadata.file_size(),
            // File ID catches replacement; NTFS ChangeTime catches an
            // in-place overwrite even if last-write time and size are
            // restored by the caller.
            platform_identity: vec![
                u64::from(metadata.volume_serial_number().unwrap_or(0)),
                metadata.file_index().unwrap_or(0),
                metadata.creation_time(),
                change_time,
            ],
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos() as u64);
        ArtifactMetadataIdentity {
            len: metadata.len(),
            platform_identity: vec![modified],
        }
    }
}

const ARTIFACT_FINGERPRINT_CACHE_CAPACITY: usize = 1024;
const ARTIFACT_FACT_CACHE_CAPACITY: usize = 1024;
const ARTIFACT_MAX_IN_FLIGHT_IDENTITIES: usize = 64;
const ARTIFACT_MAX_CONCURRENT_READS: usize = 2;
const ARTIFACT_UNSTABLE_RETRY_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(1);

struct BoundedPathCache<V> {
    capacity: usize,
    clock: u64,
    entries: BTreeMap<PathBuf, (V, u64)>,
}

impl<V> BoundedPathCache<V> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            clock: 0,
            entries: BTreeMap::new(),
        }
    }

    fn get_mut(&mut self, path: &Path) -> Option<&mut V> {
        self.clock = self.clock.wrapping_add(1);
        let (value, last_used) = self.entries.get_mut(path)?;
        *last_used = self.clock;
        Some(value)
    }

    fn insert(&mut self, path: PathBuf, value: V) {
        self.clock = self.clock.wrapping_add(1);
        self.entries.insert(path, (value, self.clock));
        while self.entries.len() > self.capacity {
            let victim = self
                .entries
                .iter()
                .min_by(
                    |(left_path, (_, left_used)), (right_path, (_, right_used))| {
                        left_used
                            .cmp(right_used)
                            .then_with(|| left_path.cmp(right_path))
                    },
                )
                .map(|(path, _)| path.clone())
                .expect("over-capacity cache has an eviction candidate");
            self.entries.remove(&victim);
        }
    }
}

type ArtifactFingerprintCache = BoundedPathCache<(ArtifactMetadataIdentity, ContentFingerprint)>;

fn artifact_fingerprint_cache() -> &'static Mutex<ArtifactFingerprintCache> {
    static CACHE: OnceLock<Mutex<ArtifactFingerprintCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(BoundedPathCache::with_capacity(
            ARTIFACT_FINGERPRINT_CACHE_CAPACITY,
        ))
    })
}

#[cfg(not(any(unix, windows)))]
fn hash_artifact_contents(path: &Path) -> std::io::Result<ContentFingerprint> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(ContentFingerprint(format!("{:x}", hash.finalize())))
}

fn fingerprint_path(path: &Path) -> ContentFingerprint {
    let Ok(before_metadata) = std::fs::metadata(path) else {
        let mut hash = Sha256::new();
        hash.update(path.as_os_str().as_encoded_bytes());
        hash.update(b"missing");
        return ContentFingerprint(format!("{:x}", hash.finalize()));
    };
    let before = artifact_metadata_identity(path, &before_metadata);
    {
        let mut cache = artifact_fingerprint_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, fingerprint)) = cache
            .get_mut(path)
            .filter(|(identity, _)| identity == &before)
        {
            return fingerprint.clone();
        }
    }

    // Unix inode+ctime and Windows creation/file metadata are immutable
    // replacement identities and avoid reading multi-gigabyte checkpoints.
    // Exotic platforms fall back to one cached content hash per metadata
    // identity.
    #[cfg(any(unix, windows))]
    let fingerprint = {
        let mut hash = Sha256::new();
        hash.update(path.as_os_str().as_encoded_bytes());
        hash.update(before.len.to_le_bytes());
        for value in &before.platform_identity {
            hash.update(value.to_le_bytes());
        }
        ContentFingerprint(format!("{:x}", hash.finalize()))
    };
    #[cfg(not(any(unix, windows)))]
    let fingerprint = hash_artifact_contents(path).unwrap_or_else(|error| {
        let mut hash = Sha256::new();
        hash.update(path.as_os_str().as_encoded_bytes());
        hash.update(error.kind().to_string().as_bytes());
        ContentFingerprint(format!("{:x}", hash.finalize()))
    });
    if std::fs::metadata(path)
        .ok()
        .map(|metadata| artifact_metadata_identity(path, &metadata))
        .as_ref()
        == Some(&before)
    {
        artifact_fingerprint_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(path.to_path_buf(), (before, fingerprint.clone()));
    }
    fingerprint
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ArtifactFormatFact {
    Known(mold_inference::artifact_format::ArtifactStorageFormat),
    ProbeFailure(mold_inference::artifact_format::ArtifactProbeFailure),
    CacheMiss,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArtifactFacts {
    content: EquivalenceContentIdentity,
    format: ArtifactFormatFact,
}

#[derive(Clone)]
struct CachedArtifactFacts {
    metadata: ArtifactMetadataIdentity,
    facts: ArtifactFacts,
    last_used: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ArtifactFactKey {
    path: PathBuf,
    metadata: ArtifactMetadataIdentity,
}

struct ArtifactFactFlight {
    result: Mutex<Option<ArtifactFacts>>,
    ready: std::sync::Condvar,
}

impl ArtifactFactFlight {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            ready: std::sync::Condvar::new(),
        }
    }

    fn wait(&self) -> ArtifactFacts {
        let mut result = self
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if let Some(facts) = result.as_ref() {
                return facts.clone();
            }
            result = self
                .ready
                .wait(result)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn publish(&self, facts: ArtifactFacts) {
        *self
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(facts);
        self.ready.notify_all();
    }
}

struct ArtifactFactCache {
    capacity: usize,
    clock: u64,
    entries: BTreeMap<PathBuf, CachedArtifactFacts>,
    in_flight: BTreeMap<ArtifactFactKey, std::sync::Arc<ArtifactFactFlight>>,
    unstable_until: BTreeMap<PathBuf, std::time::Instant>,
}

impl ArtifactFactCache {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            clock: 0,
            entries: BTreeMap::new(),
            in_flight: BTreeMap::new(),
            unstable_until: BTreeMap::new(),
        }
    }

    fn get(&mut self, path: &Path, metadata: &ArtifactMetadataIdentity) -> Option<ArtifactFacts> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(path)?;
        if &entry.metadata != metadata {
            return None;
        }
        entry.last_used = self.clock;
        Some(entry.facts.clone())
    }

    fn insert(&mut self, path: PathBuf, metadata: ArtifactMetadataIdentity, facts: ArtifactFacts) {
        self.clock = self.clock.wrapping_add(1);
        self.entries.insert(
            path,
            CachedArtifactFacts {
                metadata,
                facts,
                last_used: self.clock,
            },
        );
        while self.entries.len() > self.capacity {
            let victim = self
                .entries
                .iter()
                .min_by(|(left_path, left), (right_path, right)| {
                    left.last_used
                        .cmp(&right.last_used)
                        .then_with(|| left_path.cmp(right_path))
                })
                .map(|(path, _)| path.clone())
                .expect("over-capacity cache has an eviction candidate");
            self.entries.remove(&victim);
        }
    }

    fn unstable_backoff_active_at(&mut self, path: &Path, now: std::time::Instant) -> bool {
        let Some(deadline) = self.unstable_until.get(path).copied() else {
            return false;
        };
        if now < deadline {
            return true;
        }
        self.unstable_until.remove(path);
        false
    }

    fn mark_unstable_at(&mut self, path: PathBuf, now: std::time::Instant) {
        self.unstable_until.insert(
            path,
            now.checked_add(ARTIFACT_UNSTABLE_RETRY_COOLDOWN)
                .unwrap_or(now),
        );
        while self.unstable_until.len() > self.capacity {
            let victim = self
                .unstable_until
                .iter()
                .min_by(|(left_path, left), (right_path, right)| {
                    left.cmp(right).then_with(|| left_path.cmp(right_path))
                })
                .map(|(path, _)| path.clone())
                .expect("over-capacity unstable cache has an eviction candidate");
            self.unstable_until.remove(&victim);
        }
    }

    #[cfg(test)]
    fn remove_path(&mut self, path: &Path) {
        self.entries.remove(path);
        self.unstable_until.remove(path);
    }

    #[cfg(test)]
    fn insert_for_test(&mut self, path: PathBuf) {
        self.insert(
            path,
            ArtifactMetadataIdentity {
                len: 0,
                platform_identity: Vec::new(),
            },
            ArtifactFacts {
                content: EquivalenceContentIdentity::Sha256("test".into()),
                format: ArtifactFormatFact::CacheMiss,
            },
        );
    }

    #[cfg(test)]
    fn touch_for_test(&mut self, path: &Path) -> bool {
        let Some(metadata) = self.entries.get(path).map(|entry| entry.metadata.clone()) else {
            return false;
        };
        self.get(path, &metadata).is_some()
    }

    #[cfg(test)]
    fn entry_paths_for_test(&self) -> Vec<PathBuf> {
        self.entries.keys().cloned().collect()
    }
}

fn artifact_fact_cache() -> &'static Mutex<ArtifactFactCache> {
    static CACHE: OnceLock<Mutex<ArtifactFactCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(ArtifactFactCache::with_capacity(
            ARTIFACT_FACT_CACHE_CAPACITY,
        ))
    })
}

struct ArtifactReadLimiter {
    capacity: usize,
    active: Mutex<usize>,
    available: std::sync::Condvar,
}

impl ArtifactReadLimiter {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            active: Mutex::new(0),
            available: std::sync::Condvar::new(),
        }
    }

    fn acquire(&self) -> ArtifactReadPermit<'_> {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while *active >= self.capacity {
            active = self
                .available
                .wait(active)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        *active += 1;
        ArtifactReadPermit { limiter: self }
    }
}

struct ArtifactReadPermit<'a> {
    limiter: &'a ArtifactReadLimiter,
}

impl Drop for ArtifactReadPermit<'_> {
    fn drop(&mut self) {
        let mut active = self
            .limiter
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *active = active.saturating_sub(1);
        self.limiter.available.notify_one();
    }
}

fn artifact_read_limiter() -> &'static ArtifactReadLimiter {
    static LIMITER: OnceLock<ArtifactReadLimiter> = OnceLock::new();
    LIMITER.get_or_init(|| ArtifactReadLimiter::new(ARTIFACT_MAX_CONCURRENT_READS))
}

fn artifact_in_flight_limiter() -> &'static ArtifactReadLimiter {
    static LIMITER: OnceLock<ArtifactReadLimiter> = OnceLock::new();
    LIMITER.get_or_init(|| ArtifactReadLimiter::new(ARTIFACT_MAX_IN_FLIGHT_IDENTITIES))
}

#[cfg(test)]
fn artifact_read_counters(
) -> &'static Mutex<BTreeMap<PathBuf, std::sync::Arc<std::sync::atomic::AtomicUsize>>> {
    static COUNTERS: OnceLock<
        Mutex<BTreeMap<PathBuf, std::sync::Arc<std::sync::atomic::AtomicUsize>>>,
    > = OnceLock::new();
    COUNTERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[cfg(test)]
struct ArtifactReadCounterGuard {
    path: PathBuf,
}

#[cfg(test)]
impl Drop for ArtifactReadCounterGuard {
    fn drop(&mut self) {
        artifact_read_counters()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.path);
    }
}

#[cfg(test)]
fn register_artifact_read_counter(
    path: PathBuf,
    counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> ArtifactReadCounterGuard {
    artifact_read_counters()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(path.clone(), counter);
    ArtifactReadCounterGuard { path }
}

#[cfg(test)]
fn note_artifact_physical_read(path: &Path) {
    use std::sync::atomic::Ordering;
    if let Some(counter) = artifact_read_counters()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(path)
        .cloned()
    {
        counter.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(not(test))]
fn note_artifact_physical_read(_path: &Path) {}

#[cfg(test)]
fn equivalence_fingerprint_path(path: &Path) -> EquivalenceContentIdentity {
    artifact_facts_path_with_policy(path, false).content
}

fn equivalence_fingerprint_path_with_policy(
    path: &Path,
    cache_only: bool,
) -> EquivalenceContentIdentity {
    artifact_facts_path_with_policy(path, cache_only).content
}

struct ArtifactFactOwnerGuard {
    path: PathBuf,
    key: ArtifactFactKey,
    flight: std::sync::Arc<ArtifactFactFlight>,
    fallback: ArtifactFacts,
    _in_flight_permit: ArtifactReadPermit<'static>,
    completed: bool,
}

impl ArtifactFactOwnerGuard {
    fn finish(
        mut self,
        facts: ArtifactFacts,
        stable: bool,
        metadata: ArtifactMetadataIdentity,
    ) -> ArtifactFacts {
        {
            let mut cache = artifact_fact_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.in_flight.remove(&self.key);
            if stable {
                cache.unstable_until.remove(&self.path);
                cache.insert(self.path.clone(), metadata, facts.clone());
            } else {
                cache.mark_unstable_at(self.path.clone(), std::time::Instant::now());
            }
        }
        self.flight.publish(facts.clone());
        self.completed = true;
        facts
    }
}

impl Drop for ArtifactFactOwnerGuard {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        artifact_fact_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .in_flight
            .remove(&self.key);
        self.flight.publish(self.fallback.clone());
    }
}

fn artifact_facts_path_with_policy(path: &Path, cache_only: bool) -> ArtifactFacts {
    let Ok(before_metadata) = std::fs::metadata(path) else {
        return ArtifactFacts {
            content: unknown_equivalence_content(path, None),
            format: ArtifactFormatFact::ProbeFailure(
                mold_inference::artifact_format::ArtifactProbeFailure::Io,
            ),
        };
    };
    let before = artifact_metadata_identity(path, &before_metadata);
    let key = ArtifactFactKey {
        path: path.to_path_buf(),
        metadata: before.clone(),
    };
    let existing_flight = {
        let mut cache = artifact_fact_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(facts) = cache.get(path, &before) {
            return facts;
        }
        if cache.unstable_backoff_active_at(path, std::time::Instant::now()) {
            return ArtifactFacts {
                content: unknown_equivalence_content(path, Some(&before)),
                format: ArtifactFormatFact::CacheMiss,
            };
        }
        if cache_only {
            return ArtifactFacts {
                content: unknown_equivalence_content(path, Some(&before)),
                format: ArtifactFormatFact::CacheMiss,
            };
        }
        cache.in_flight.get(&key).cloned()
    };
    if let Some(flight) = existing_flight {
        return flight.wait();
    }

    // Construct the unwind fallback before publishing an owner entry so every
    // subsequently visible flight is protected by its RAII owner guard.
    let owner_fallback = ArtifactFacts {
        content: unknown_equivalence_content(path, Some(&before)),
        format: ArtifactFormatFact::CacheMiss,
    };
    // Bound unique metadata identities separately from physical reads. This
    // prevents a large admission burst from growing the single-flight table
    // without limit while still allowing same-identity callers to join.
    let in_flight_permit = artifact_in_flight_limiter().acquire();
    let flight = {
        let mut cache = artifact_fact_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(facts) = cache.get(path, &before) {
            return facts;
        }
        if cache.unstable_backoff_active_at(path, std::time::Instant::now()) {
            drop(cache);
            drop(in_flight_permit);
            return ArtifactFacts {
                content: unknown_equivalence_content(path, Some(&before)),
                format: ArtifactFormatFact::CacheMiss,
            };
        }
        if let Some(flight) = cache.in_flight.get(&key).cloned() {
            drop(cache);
            drop(in_flight_permit);
            return flight.wait();
        }
        let flight = std::sync::Arc::new(ArtifactFactFlight::new());
        cache
            .in_flight
            .insert(key.clone(), std::sync::Arc::clone(&flight));
        flight
    };
    let owner = ArtifactFactOwnerGuard {
        path: path.to_path_buf(),
        key,
        flight,
        fallback: owner_fallback,
        _in_flight_permit: in_flight_permit,
        completed: false,
    };
    let facts = {
        let _permit = artifact_read_limiter().acquire();
        note_artifact_physical_read(path);
        let content = hash_equivalence_artifact_contents(path)
            .unwrap_or_else(|_| unknown_equivalence_content(path, Some(&before)));
        let format = match mold_inference::artifact_format::probe(path) {
            Ok(format) => ArtifactFormatFact::Known(format),
            Err(error) => ArtifactFormatFact::ProbeFailure(error),
        };
        ArtifactFacts { content, format }
    };
    let stable = std::fs::metadata(path)
        .ok()
        .map(|metadata| artifact_metadata_identity(path, &metadata))
        .as_ref()
        == Some(&before);
    let published = if stable {
        facts
    } else {
        // A digest spanning concurrent mutation is not the identity of one
        // stable artifact. Fail closed and let asynchronous preparation hash
        // the new metadata identity.
        ArtifactFacts {
            content: unknown_equivalence_content(path, Some(&before)),
            format: ArtifactFormatFact::CacheMiss,
        }
    };
    owner.finish(published, stable, before)
}

fn hash_equivalence_artifact_contents(path: &Path) -> std::io::Result<EquivalenceContentIdentity> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(EquivalenceContentIdentity::Sha256(format!(
        "{:x}",
        hash.finalize()
    )))
}

fn unknown_equivalence_content(
    path: &Path,
    metadata: Option<&ArtifactMetadataIdentity>,
) -> EquivalenceContentIdentity {
    static UNKNOWN_SECRET: OnceLock<[u8; 32]> = OnceLock::new();
    let secret = UNKNOWN_SECRET.get_or_init(|| {
        let mut bytes = [0_u8; 32];
        getrandom::fill(&mut bytes)
            .expect("unknown artifact identity requires process-unique entropy");
        bytes
    });
    let mut hash = Sha256::new();
    hash.update(b"mold.unknown-artifact.v2\0");
    hash.update(secret);
    hash.update(path.as_os_str().as_encoded_bytes());
    match metadata {
        Some(metadata) => {
            hash.update(b"observed\0");
            hash.update(metadata.len.to_le_bytes());
            for value in &metadata.platform_identity {
                hash.update(value.to_le_bytes());
            }
        }
        None => hash.update(b"missing\0"),
    }
    let discriminator = format!("{:x}", hash.finalize());
    EquivalenceContentIdentity::Unknown { discriminator }
}

fn model_fingerprint(model: &str, artifacts: &BTreeMap<ComponentRole, PathBuf>) -> String {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    for (role, path) in artifacts {
        hash.update(format!("{role:?}").as_bytes());
        hash.update(fingerprint_path(path).0.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn equivalence_model_fingerprint(
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
    cache_only: bool,
) -> String {
    let mut hash = Sha256::new();
    for (role, path) in artifacts {
        hash.update(format!("{role:?}").as_bytes());
        equivalence_fingerprint_path_with_policy(path, cache_only).update_hash(&mut hash);
    }
    format!("{:x}", hash.finalize())
}

pub fn resolved_model_fingerprint(
    config: &Config,
    model: &str,
) -> Result<String, ExecutionPlanError> {
    if let Some(model_config) = config.models.get(model) {
        return frozen_model_fingerprint(model, model_config);
    }
    let paths =
        ModelPaths::resolve(model, config).ok_or_else(|| ExecutionPlanError::MissingArtifacts {
            model: model.to_string(),
        })?;
    let mut artifacts = BTreeMap::new();
    artifacts.insert(ComponentRole::Transformer, paths.transformer);
    for (index, shard) in paths.transformer_shards.into_iter().enumerate() {
        artifacts.insert(ComponentRole::TransformerShard(index as u8), shard);
    }
    artifacts.insert(ComponentRole::Vae, paths.vae);
    Ok(model_fingerprint(model, &artifacts))
}

/// Fingerprint the complete immutable input to engine construction for a
/// durable chain job. This is deliberately broader than placement planning:
/// tokenizers, default LoRA, and every serialized model default participate
/// even when they do not consume GPU memory.
pub fn frozen_model_fingerprint(
    model: &str,
    model_config: &mold_core::ModelConfig,
) -> Result<String, ExecutionPlanError> {
    let paths = ModelPaths::resolve_from_model_config_exact(model_config).ok_or_else(|| {
        ExecutionPlanError::MissingArtifacts {
            model: model.to_string(),
        }
    })?;
    let mut files = vec![
        ("transformer".to_string(), paths.transformer),
        ("vae".to_string(), paths.vae),
    ];
    files.extend(
        paths
            .transformer_shards
            .into_iter()
            .enumerate()
            .map(|(index, path)| (format!("transformer_shard:{index}"), path)),
    );
    macro_rules! optional_file {
        ($name:literal, $value:expr) => {
            if let Some(path) = $value {
                files.push(($name.to_string(), path));
            }
        };
    }
    optional_file!("spatial_upscaler", paths.spatial_upscaler);
    optional_file!("temporal_upscaler", paths.temporal_upscaler);
    optional_file!("distilled_lora", paths.distilled_lora);
    optional_file!("t5_encoder", paths.t5_encoder);
    optional_file!("clip_encoder", paths.clip_encoder);
    optional_file!("t5_tokenizer", paths.t5_tokenizer);
    optional_file!("clip_tokenizer", paths.clip_tokenizer);
    optional_file!("clip_encoder_2", paths.clip_encoder_2);
    optional_file!("clip_tokenizer_2", paths.clip_tokenizer_2);
    files.extend(
        paths
            .text_encoder_files
            .into_iter()
            .enumerate()
            .map(|(index, path)| (format!("text_encoder:{index}"), path)),
    );
    optional_file!("text_tokenizer", paths.text_tokenizer);
    optional_file!("decoder", paths.decoder);
    if let Some(path) = model_config.lora.as_deref() {
        files.push(("default_lora".to_string(), PathBuf::from(path)));
    }

    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    hash.update(
        serde_json::to_vec(model_config)
            .map_err(|error| ExecutionPlanError::PlanInvalidated(error.to_string()))?,
    );
    for (role, path) in files {
        hash.update(role.as_bytes());
        hash.update(fingerprint_path(&path).0.as_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Resolve every model companion once and freeze it as a canonical absolute
/// path set. The synthetic runtime key deliberately cannot match a catalog
/// ID or built-in manifest, so recovery never consults changed sidecars,
/// `MOLD_HOME`, or per-model environment fallbacks.
pub fn freeze_chain_model(
    config: &Config,
    model: &str,
) -> Result<mold_core::chain_job::FrozenChainModel, ExecutionPlanError> {
    let paths =
        ModelPaths::resolve(model, config).ok_or_else(|| ExecutionPlanError::MissingArtifacts {
            model: model.to_string(),
        })?;
    let canonical = |path: &std::path::Path| {
        std::fs::canonicalize(path)
            .map_err(|_| ExecutionPlanError::MissingArtifacts {
                model: model.to_string(),
            })
            .map(|path| path.to_string_lossy().into_owned())
    };
    let canonical_optional = |path: Option<&PathBuf>| path.map(|path| canonical(path)).transpose();
    let mut frozen = config.resolved_model_config(model);
    frozen.transformer = Some(canonical(&paths.transformer)?);
    frozen.transformer_shards = (!paths.transformer_shards.is_empty())
        .then(|| {
            paths
                .transformer_shards
                .iter()
                .map(|path| canonical(path))
                .collect()
        })
        .transpose()?;
    frozen.vae = Some(canonical(&paths.vae)?);
    frozen.spatial_upscaler = canonical_optional(paths.spatial_upscaler.as_ref())?;
    frozen.temporal_upscaler = canonical_optional(paths.temporal_upscaler.as_ref())?;
    frozen.distilled_lora = canonical_optional(paths.distilled_lora.as_ref())?;
    frozen.t5_encoder = canonical_optional(paths.t5_encoder.as_ref())?;
    frozen.clip_encoder = canonical_optional(paths.clip_encoder.as_ref())?;
    frozen.t5_tokenizer = canonical_optional(paths.t5_tokenizer.as_ref())?;
    frozen.clip_tokenizer = canonical_optional(paths.clip_tokenizer.as_ref())?;
    frozen.clip_encoder_2 = canonical_optional(paths.clip_encoder_2.as_ref())?;
    frozen.clip_tokenizer_2 = canonical_optional(paths.clip_tokenizer_2.as_ref())?;
    frozen.text_encoder_files = (!paths.text_encoder_files.is_empty())
        .then(|| {
            paths
                .text_encoder_files
                .iter()
                .map(|path| canonical(path))
                .collect()
        })
        .transpose()?;
    frozen.text_tokenizer = canonical_optional(paths.text_tokenizer.as_ref())?;
    frozen.decoder = canonical_optional(paths.decoder.as_ref())?;
    frozen.lora = frozen
        .lora
        .as_deref()
        .map(|path| canonical(Path::new(path)))
        .transpose()?;

    let model_fingerprint = frozen_model_fingerprint(model, &frozen)?;
    let runtime_model_id = format!("mold-frozen-chain:{model_fingerprint}");
    Ok(mold_core::chain_job::FrozenChainModel {
        runtime_model_id,
        config: frozen,
        model_fingerprint,
    })
}

fn execution_fingerprint(
    model: &str,
    device: &DeviceFact,
    effective: &EffectivePlacement,
    components: &BTreeMap<ComponentRole, ComponentExecutionPlan>,
    engine_config: &mold_inference::FrozenEngineConfig,
    effective_loras: &[PlannedLora],
    offload: bool,
) -> String {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    hash.update(device.id.as_bytes());
    hash.update(format!("{effective:?}").as_bytes());
    hash.update(format!("{components:?}").as_bytes());
    hash.update(format!("{engine_config:?}").as_bytes());
    hash.update(format!("{effective_loras:?}").as_bytes());
    hash.update([u8::from(offload)]);
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{AdvancedPlacement, ModelConfig};
    use tempfile::TempDir;

    #[test]
    fn production_family_planning_uses_the_static_batch_registry_before_load() {
        for entry in mold_inference::production_batch_capabilities() {
            assert_eq!(
                capabilities_for_family(entry.family).batch_execution,
                Some(entry.execution),
                "{}",
                entry.family
            );
            for alias in entry.aliases {
                assert_eq!(
                    capabilities_for_family(alias).batch_execution,
                    Some(entry.execution),
                    "{alias}"
                );
            }
        }
        assert_eq!(capabilities_for_family("unknown").batch_execution, None);
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    fn sparse_file(path: &Path, bytes: u64) {
        let file = std::fs::File::create(path).unwrap();
        file.set_len(bytes).unwrap();
        drop(file);
        // Admission tests use logical multi-GiB sparse zero files solely for
        // their metadata length. Seed their synthetic identity so unrelated
        // memory-planning tests do not spend minutes hashing hole ranges.
        let metadata = std::fs::metadata(path).unwrap();
        artifact_fact_cache().lock().unwrap().insert(
            path.to_path_buf(),
            artifact_metadata_identity(path, &metadata),
            ArtifactFacts {
                content: EquivalenceContentIdentity::Sha256(format!("{bytes:064x}")),
                format: ArtifactFormatFact::CacheMiss,
            },
        );
    }

    fn config(root: &Path, family: &str, persisted: Option<DevicePlacement>) -> Config {
        let mut config = Config::default();
        config.models.insert(
            "test:q4".into(),
            ModelConfig {
                transformer: Some(root.join("transformer-q4.gguf").display().to_string()),
                vae: Some(root.join("vae.safetensors").display().to_string()),
                t5_encoder: Some(root.join("t5.safetensors").display().to_string()),
                family: Some(family.into()),
                placement: persisted,
                ..ModelConfig::default()
            },
        );
        config
    }

    fn request(placement: Option<DevicePlacement>) -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"test:q4","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        request.placement = placement;
        request
    }

    fn sized_config(
        root: &Path,
        family: &str,
        transformer_gib: u64,
        vae_gib: u64,
        encoder_gib: u64,
    ) -> (Config, GenerateRequest) {
        let transformer = root.join(format!("{family}-transformer.safetensors"));
        let vae = root.join(format!("{family}-vae.safetensors"));
        let encoder = root.join(format!("{family}-encoder.safetensors"));
        sparse_file(&transformer, transformer_gib * GIB);
        sparse_file(&vae, vae_gib * GIB);
        sparse_file(&encoder, encoder_gib * GIB);
        let mut config = Config::default();
        config.models.insert(
            "case:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                t5_encoder: Some(encoder.display().to_string()),
                family: Some(family.to_string()),
                ..ModelConfig::default()
            },
        );
        let request = serde_json::from_str(
            r#"{"prompt":"x","model":"case:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        (config, request)
    }

    fn devices(free: &[u64]) -> Vec<DeviceFact> {
        free.iter()
            .enumerate()
            .map(|(ordinal, bytes)| DeviceFact {
                id: format!("cuda:{ordinal}"),
                ordinal,
                backend: GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: *bytes,
            })
            .collect()
    }

    #[test]
    fn request_placement_wins_wholly_over_persisted() {
        let root = TempDir::new().unwrap();
        let persisted = DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        };
        let requested = DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::device("cuda:1"),
                ..AdvancedPlacement::default()
            }),
        };
        let plans = resolve_execution_plans(
            &config(root.path(), "flux2", Some(persisted)),
            &request(Some(requested)),
            &devices(&[24 * GIB, 24 * GIB]),
            false,
        )
        .unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].device_id, "cuda:1");
        assert!(plans[0]
            .components
            .values()
            .all(|component| { component.placement != ResolvedComponentPlacement::Cpu }));
    }

    #[test]
    fn explicit_cpu_is_charged_to_host_ram() {
        let root = TempDir::new().unwrap();
        sparse_file(&root.path().join("t5.safetensors"), MIB);
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        };
        let plan = resolve_execution_plans(
            &config(root.path(), "flux2", None),
            &request(Some(placement)),
            &devices(&[24 * GIB]),
            false,
        )
        .unwrap()
        .remove(0);
        assert_eq!(
            plan.predicted_host_increment_bytes,
            BASE_HOST_TRANSIENT + MIB
        );
        assert!(plan.components.iter().any(|(role, component)| {
            role.is_text_encoder() && component.placement == ResolvedComponentPlacement::Cpu
        }));
    }

    #[test]
    fn hard_cpu_transformer_pin_is_credited_before_the_fit_gate() {
        let root = TempDir::new().unwrap();
        let (config, mut request) = sized_config(root.path(), "flux2", 32, 1, 2);
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::Cpu,
                ..AdvancedPlacement::default()
            }),
        });

        let plan = resolve_execution_plans(&config, &request, &devices(&[8 * GIB]), false)
            .expect("the CPU-resident transformer must be removed from GPU admission")
            .remove(0);

        assert_eq!(
            plan.components[&ComponentRole::Transformer].placement,
            ResolvedComponentPlacement::Cpu
        );
        assert!(plan.predicted_vram_peak_bytes < 8 * GIB);
        assert!(
            plan.predicted_host_increment_bytes >= BASE_HOST_TRANSIENT + 32 * GIB,
            "CPU-resident weights must be reserved against host RAM"
        );
    }

    #[test]
    fn block_offload_reserves_the_streamed_transformer_in_host_ram() {
        let root = TempDir::new().unwrap();
        let (config, request) = sized_config(root.path(), "flux", 24, 1, 4);
        let plan = resolve_execution_plans(&config, &request, &devices(&[12 * GIB]), false)
            .expect("large FLUX should fit through automatic block offload")
            .remove(0);

        assert_eq!(plan.offload_mode, OffloadMode::Block);
        assert!(
            plan.predicted_host_increment_bytes >= BASE_HOST_TRANSIENT + 24 * GIB,
            "streamed transformer weights remain resident in host memory"
        );
    }

    #[test]
    fn different_gpu_component_pins_are_rejected() {
        let root = TempDir::new().unwrap();
        let placement = DevicePlacement {
            text_encoders: DeviceRef::device("cuda:0"),
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::device("cuda:1"),
                ..AdvancedPlacement::default()
            }),
        };
        assert!(matches!(
            resolve_execution_plans(
                &config(root.path(), "flux2", None),
                &request(Some(placement)),
                &devices(&[24 * GIB, 24 * GIB]),
                false,
            ),
            Err(ExecutionPlanError::CrossDevicePlacement(_))
        ));
    }

    #[test]
    fn auto_cpu_exists_only_under_pressure_for_supported_family() {
        let root = TempDir::new().unwrap();
        sparse_file(&root.path().join("transformer-q4.gguf"), 4 * GIB);
        sparse_file(&root.path().join("vae.safetensors"), GIB / 2);
        sparse_file(&root.path().join("t5.safetensors"), 4 * GIB);
        let flux2_config = config(root.path(), "flux2", None);
        let roomy =
            resolve_execution_plans(&flux2_config, &request(None), &devices(&[24 * GIB]), false)
                .unwrap()
                .remove(0);
        assert!(roomy
            .components
            .values()
            .all(|component| component.placement != ResolvedComponentPlacement::Cpu));

        let pressured =
            resolve_execution_plans(&flux2_config, &request(None), &devices(&[9 * GIB]), false)
                .unwrap()
                .remove(0);
        assert!(pressured.components.iter().any(|(role, component)| {
            role.is_text_encoder() && component.placement == ResolvedComponentPlacement::Cpu
        }));

        let unsupported = config(root.path(), "unknown-family", None);
        assert_eq!(
            resolve_execution_plans(&unsupported, &request(None), &devices(&[3 * GIB]), false,),
            Err(ExecutionPlanError::InsufficientVram)
        );
    }

    #[test]
    fn unsupported_family_offload_is_rejected_but_supported_runtime_fallbacks_are_plannable() {
        let root = TempDir::new().unwrap();
        assert!(matches!(
            resolve_execution_plans(
                &config(root.path(), "sdxl", None),
                &request(None),
                &devices(&[24 * GIB]),
                true,
            ),
            Err(ExecutionPlanError::UnsupportedOffload { .. })
        ));
        let ltx2 = resolve_execution_plans(
            &config(root.path(), "ltx2", None),
            &request(None),
            &devices(&[24 * GIB]),
            true,
        )
        .expect("LTX-2 honors MOLD_OFFLOAD through its native streaming runtime");
        assert_eq!(ltx2[0].offload_mode, OffloadMode::Block);

        sparse_file(&root.path().join("transformer-q4.gguf"), 4 * GIB);
        sparse_file(&root.path().join("vae.safetensors"), GIB);
        sparse_file(&root.path().join("t5.safetensors"), GIB);
        let flux2 = resolve_execution_plans(
            &config(root.path(), "flux2", None),
            &request(None),
            &devices(&[24 * GIB]),
            true,
        )
        .expect("a global offload preference must not reject an incompatible quantized path");
        assert_eq!(flux2[0].offload_mode, OffloadMode::None);
    }

    #[test]
    fn candidate_count_scales_without_two_gpu_assumption() {
        let root = TempDir::new().unwrap();
        let config = config(root.path(), "flux2", None);
        for count in [1, 2, 8, 16, 64] {
            let facts = devices(&vec![24 * GIB; count]);
            let plans = resolve_execution_plans(&config, &request(None), &facts, false).unwrap();
            assert_eq!(plans.len(), count);
        }
        assert_eq!(
            resolve_execution_plans(&config, &request(None), &[], false),
            Err(ExecutionPlanError::InsufficientVram)
        );
    }

    #[test]
    fn each_candidate_uses_its_own_effective_vram_capacity() {
        let root = TempDir::new().unwrap();
        let (config, request) = sized_config(root.path(), "flux2", 12, 1, 2);
        let plans = resolve_execution_plans(
            &config,
            &request,
            &devices(&[12 * GIB, 24 * GIB, 48 * GIB]),
            false,
        )
        .unwrap();
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.device_ordinal)
                .collect::<Vec<_>>(),
            vec![1, 2],
            "a roomy sibling must not make the 12 GiB candidate admissible"
        );
        assert!(plans
            .iter()
            .all(|plan| plan.predicted_vram_peak_bytes <= 24 * GIB));
    }

    #[test]
    fn missing_free_vram_never_falls_back_to_total_capacity() {
        let root = TempDir::new().unwrap();
        let (config, request) = sized_config(root.path(), "sd15", 1, 1, 1);
        assert_eq!(
            resolve_execution_plans(&config, &request, &devices(&[0]), false),
            Err(ExecutionPlanError::InsufficientVram)
        );
    }

    #[test]
    fn request_aware_family_regressions_cover_12_24_and_48_gib() {
        let cases = [
            ("flux", 20, 1, 4, vec![0, 1, 2]),
            ("flux2", 12, 1, 2, vec![1, 2]),
            ("sd15", 2, 1, 1, vec![0, 1, 2]),
            ("sdxl", 6, 1, 2, vec![0, 1, 2]),
            ("sd3", 14, 1, 3, vec![1, 2]),
            ("qwen-image", 34, 1, 6, vec![2]),
            ("z-image", 14, 1, 3, vec![1, 2]),
            ("ltx-video", 8, 1, 2, vec![1, 2]),
            ("ltx2", 40, 1, 10, vec![0, 1, 2]),
        ];
        for (family, transformer, vae, encoder, expected_ordinals) in cases {
            let root = TempDir::new().unwrap();
            let (config, request) = sized_config(root.path(), family, transformer, vae, encoder);
            let plans = resolve_execution_plans(
                &config,
                &request,
                &devices(&[12 * GIB, 24 * GIB, 48 * GIB]),
                false,
            )
            .unwrap_or_else(|error| panic!("{family} planning failed: {error}"));
            assert_eq!(
                plans
                    .iter()
                    .map(|plan| plan.device_ordinal)
                    .collect::<Vec<_>>(),
                expected_ordinals,
                "{family} admission drifted at 12/24/48 GiB"
            );
            if family == "ltx2" {
                assert!(plans.iter().all(|plan| {
                    plan.engine_load_strategy == mold_inference::LoadStrategy::Eager
                        && plan.offload_mode == OffloadMode::None
                }));
            }
        }
    }

    #[test]
    fn missing_artifact_metadata_keeps_a_conservative_host_charge() {
        let missing = PathBuf::from("/definitely/missing/mold-artifact.safetensors");
        assert_eq!(artifact_size(&missing), 64 * MIB);
    }

    #[test]
    fn owner_boundary_rejects_changed_artifact_and_materializes_auto() {
        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), vec![0_u8; 1024]).unwrap();
        }
        let config = config(root.path(), "flux2", None);
        let request = request(None);
        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        let placement = materialized_placement(&plan);
        assert!(matches!(
            placement.advanced.unwrap().transformer,
            DeviceRef::Device { .. }
        ));
        validate_before_cuda(&plan, "cuda:0", 0, &config, &request).unwrap();

        std::fs::write(root.path().join("transformer-q4.gguf"), vec![1_u8; 2048]).unwrap();
        assert!(matches!(
            validate_before_cuda(&plan, "cuda:0", 0, &config, &request),
            Err(ExecutionPlanError::PlanInvalidated(_))
        ));
    }

    #[test]
    fn plan_freezes_tokenizers_factory_config_and_ordered_default_lora_stack() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer-q4.gguf",
            "vae.safetensors",
            "t5.safetensors",
            "t5-tokenizer.json",
            "first-lora.safetensors",
            "second-lora.safetensors",
        ] {
            std::fs::write(root.path().join(name), name.as_bytes()).unwrap();
        }
        let mut config = config(root.path(), "flux", None);
        let model = config.models.get_mut("test:q4").unwrap();
        model.t5_tokenizer = Some(root.path().join("t5-tokenizer.json").display().to_string());
        model.is_schnell = Some(true);
        model.lora = Some(
            root.path()
                .join("first-lora.safetensors")
                .display()
                .to_string(),
        );
        model.lora_scale = Some(0.75);
        config.t5_variant = Some("q4".into());

        let mut request = request(None);
        let default_plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        assert!(default_plan
            .components
            .contains_key(&ComponentRole::T5Tokenizer));
        assert_eq!(default_plan.effective_loras.len(), 1);
        assert_eq!(default_plan.effective_loras[0].scale(), 0.75);
        assert_eq!(default_plan.engine_config.is_schnell, Some(true));
        assert_eq!(default_plan.engine_config.t5_variant.as_deref(), Some("q4"));

        let flux2_transformer = root.path().join("flux2-transformer.safetensors");
        std::fs::write(&flux2_transformer, b"weights").unwrap();
        let model = config.models.get_mut("test:q4").unwrap();
        model.family = Some("flux2".into());
        model.transformer = Some(flux2_transformer.display().to_string());
        let default_lora_plan =
            resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), true)
                .unwrap()
                .remove(0);
        assert_eq!(
            default_lora_plan.offload_mode,
            OffloadMode::None,
            "model-default LoRA must participate in offload/memory admission"
        );

        request.loras = Some(vec![
            mold_core::LoraWeight {
                path: root
                    .path()
                    .join("second-lora.safetensors")
                    .display()
                    .to_string(),
                scale: -0.5,
            },
            mold_core::LoraWeight {
                path: root
                    .path()
                    .join("first-lora.safetensors")
                    .display()
                    .to_string(),
                scale: 1.25,
            },
        ]);
        let request_plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        assert_ne!(
            request_plan.execution_fingerprint,
            default_plan.execution_fingerprint
        );
        assert_eq!(
            request_plan
                .effective_loras
                .iter()
                .map(PlannedLora::scale)
                .collect::<Vec<_>>(),
            vec![-0.5, 1.25]
        );
        let mut materialized = request.clone();
        materialize_request(&request_plan, &mut materialized);
        assert!(materialized.lora.is_none());
        assert_eq!(
            materialized
                .loras
                .unwrap()
                .iter()
                .map(|lora| lora.scale)
                .collect::<Vec<_>>(),
            vec![-0.5, 1.25]
        );

        config.models.get_mut("test:q4").unwrap().is_schnell = Some(false);
        assert!(matches!(
            validate_before_cuda(&request_plan, "cuda:0", 0, &config, &request),
            Err(ExecutionPlanError::PlanInvalidated(_))
        ));
    }

    #[test]
    fn semantic_encoder_roles_preserve_sparse_and_mixed_topologies() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "clip-l.safetensors",
            "clip-g.safetensors",
            "t5.safetensors",
            "qwen-0.safetensors",
            "qwen-1.safetensors",
        ] {
            std::fs::write(root.path().join(name), b"x").unwrap();
        }
        let paths = ModelPaths {
            transformer: root.path().join("transformer.safetensors"),
            transformer_shards: Vec::new(),
            vae: root.path().join("vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(root.path().join("t5.safetensors")),
            clip_encoder: Some(root.path().join("clip-l.safetensors")),
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: Some(root.path().join("clip-g.safetensors")),
            clip_tokenizer_2: None,
            text_encoder_files: vec![
                root.path().join("qwen-0.safetensors"),
                root.path().join("qwen-1.safetensors"),
            ],
            text_tokenizer: None,
            decoder: None,
        };
        let frozen = mold_inference::FrozenEngineConfig::resolve("unused", &Config::default());
        let artifacts = concrete_artifacts_for_family(&paths, "qwen-image", &[], &frozen);
        assert!(artifacts.contains_key(&ComponentRole::T5));
        assert!(artifacts.contains_key(&ComponentRole::ClipL));
        assert!(artifacts.contains_key(&ComponentRole::ClipG));
        assert!(artifacts.contains_key(&ComponentRole::QwenShard(0)));
        assert!(artifacts.contains_key(&ComponentRole::QwenShard(1)));
        for family in ["flux2", "flux.2", "flux2-klein", "z-image"] {
            let family_artifacts = concrete_artifacts_for_family(&paths, family, &[], &frozen);
            assert!(
                family_artifacts.contains_key(&ComponentRole::QwenShard(0)),
                "{family} must retain semantic Qwen topology"
            );
        }

        let sparse_ltx = ModelPaths {
            t5_encoder: None,
            clip_encoder: None,
            clip_encoder_2: None,
            text_encoder_files: vec![root.path().join("qwen-0.safetensors")],
            ..paths
        };
        let ltx_artifacts = concrete_artifacts_for_family(&sparse_ltx, "ltx2", &[], &frozen);
        assert!(ltx_artifacts.contains_key(&ComponentRole::GemmaShard(0)));
        assert!(!ltx_artifacts.keys().any(|role| matches!(
            role,
            ComponentRole::T5 | ComponentRole::ClipL | ComponentRole::ClipG
        )));
    }

    #[test]
    fn materialized_placement_preserves_clip_only_and_mixed_encoder_choices() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "clip-l.safetensors",
            "t5.safetensors",
        ] {
            std::fs::write(root.path().join(name), b"x").unwrap();
        }

        let mut clip_only_config = Config::default();
        clip_only_config.models.insert(
            "clip-only:bf16".into(),
            ModelConfig {
                transformer: Some(
                    root.path()
                        .join("transformer.safetensors")
                        .display()
                        .to_string(),
                ),
                vae: Some(root.path().join("vae.safetensors").display().to_string()),
                clip_encoder: Some(root.path().join("clip-l.safetensors").display().to_string()),
                family: Some("sdxl".into()),
                ..ModelConfig::default()
            },
        );
        let clip_request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"clip-only:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let clip_plan = resolve_execution_plans(
            &clip_only_config,
            &clip_request,
            &devices(&[24 * GIB]),
            false,
        )
        .unwrap()
        .remove(0);
        let clip_materialized = materialized_placement(&clip_plan);
        assert!(matches!(
            clip_materialized.text_encoders,
            DeviceRef::Device { .. }
        ));
        assert!(matches!(
            clip_materialized.advanced.unwrap().clip_l,
            Some(DeviceRef::Device { .. })
        ));

        let mut mixed_config = config(root.path(), "flux", None);
        mixed_config.models.get_mut("test:q4").unwrap().clip_encoder =
            Some(root.path().join("clip-l.safetensors").display().to_string());
        let mixed_request = request(Some(DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                t5: Some(DeviceRef::Cpu),
                clip_l: Some(DeviceRef::device("cuda:0")),
                ..AdvancedPlacement::default()
            }),
        }));
        let mixed_plan =
            resolve_execution_plans(&mixed_config, &mixed_request, &devices(&[24 * GIB]), false)
                .unwrap()
                .remove(0);
        let mixed_materialized = materialized_placement(&mixed_plan);
        let advanced = mixed_materialized.advanced.unwrap();
        assert_eq!(advanced.t5, Some(DeviceRef::Cpu));
        assert!(matches!(advanced.clip_l, Some(DeviceRef::Device { .. })));
    }

    #[test]
    fn prepared_inputs_are_per_device_and_preserve_admission_fences() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer-q4.gguf",
            "vae.safetensors",
            "t5.safetensors",
            "selected-t5.safetensors",
        ] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let config = config(root.path(), "flux", None);
        let request = request(None);
        let admission_paths = ModelPaths::resolve(&request.model, &config).unwrap();
        let mut selected_paths = admission_paths.clone();
        let selected_t5 = root.path().join("selected-t5.safetensors");
        selected_paths.t5_encoder = Some(selected_t5.clone());
        let mut selected_config =
            mold_inference::FrozenEngineConfig::resolve(&request.model, &config);
        selected_config.t5_variant = Some("fp16".to_string());
        selected_config.selected_t5_path = Some(selected_t5.clone());
        let prepared = PreparedExecutionInputs {
            authority_fingerprint: preparation_authority_fingerprint(
                &config,
                &request,
                &admission_paths,
                &mold_inference::FrozenEngineConfig::resolve(&request.model, &config),
            ),
            by_device: BTreeMap::from([(
                "cuda:0".to_string(),
                PreparedDeviceExecutionInputs {
                    engine_paths: selected_paths,
                    engine_config: selected_config,
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
        };

        let plans = resolve_execution_plans_with_prepared(
            &config,
            &request,
            &devices(&[24 * GIB, 24 * GIB]),
            false,
            Some(&prepared),
        )
        .unwrap();
        assert_eq!(plans.len(), 1, "unprepared sibling must not be admitted");
        let plan = &plans[0];
        assert_eq!(plan.device_id, "cuda:0");
        assert_eq!(plan.engine_paths.t5_encoder.as_ref(), Some(&selected_t5));
        assert_eq!(plan.admission_paths, admission_paths);
        assert_eq!(
            plan.components
                .get(&ComponentRole::T5)
                .map(|component| &component.artifact_path),
            Some(&selected_t5)
        );
        assert!(
            plan.components
                .values()
                .filter(|component| {
                    matches!(component.placement, ResolvedComponentPlacement::Device(_))
                })
                .all(|component| component.predicted_vram_bytes > 0),
            "every concrete GPU component must expose a non-zero weight estimate"
        );
        validate_before_cuda(plan, "cuda:0", 0, &config, &request).unwrap();
    }

    #[test]
    fn prepared_inputs_are_rejected_when_config_or_request_authority_changes() {
        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let config = config(root.path(), "flux", None);
        let request = request(None);
        let paths = ModelPaths::resolve(&request.model, &config).unwrap();
        let engine_config = mold_inference::FrozenEngineConfig::resolve(&request.model, &config);
        let prepared = PreparedExecutionInputs {
            authority_fingerprint: preparation_authority_fingerprint(
                &config,
                &request,
                &paths,
                &engine_config,
            ),
            by_device: BTreeMap::from([(
                "cuda:0".to_string(),
                PreparedDeviceExecutionInputs {
                    engine_paths: paths.clone(),
                    engine_config,
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
        };

        let mut changed_config = config.clone();
        changed_config.t5_variant = Some("q3".to_string());
        assert!(matches!(
            resolve_execution_plans_with_prepared(
                &changed_config,
                &request,
                &devices(&[24 * GIB]),
                false,
                Some(&prepared),
            ),
            Err(ExecutionPlanError::PreparedInputsStale(_))
        ));

        let mut changed_request = request.clone();
        changed_request.width += 64;
        assert!(matches!(
            resolve_execution_plans_with_prepared(
                &config,
                &changed_request,
                &devices(&[24 * GIB]),
                false,
                Some(&prepared),
            ),
            Err(ExecutionPlanError::PreparedInputsStale(_))
        ));
    }

    #[test]
    fn same_size_in_place_overwrite_with_restored_mtime_changes_content_identity() {
        use filetime::{set_file_mtime, FileTime};

        let root = TempDir::new().unwrap();
        let path = root.path().join("weights.safetensors");
        std::fs::write(&path, b"aaaa").unwrap();
        let original_mtime = FileTime::from_last_modification_time(&path.metadata().unwrap());
        let before = fingerprint_path(&path);

        std::fs::write(&path, b"bbbb").unwrap();
        set_file_mtime(&path, original_mtime).unwrap();

        assert_ne!(before, fingerprint_path(&path));
    }

    #[test]
    fn runtime_semantic_parser_matches_cfg_plus_and_long_prompt_contracts() {
        assert_eq!(
            runtime_semantic_setting("MOLD_CFG_PLUS", Some("true")).value,
            CanonicalRuntimeValue::Boolean(true)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_CFG_PLUS", Some("yes")).value,
            CanonicalRuntimeValue::Boolean(true)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_CFG_PLUS", Some("TRUE")).value,
            CanonicalRuntimeValue::Boolean(false)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_CFG_PLUS", Some(" true ")).value,
            CanonicalRuntimeValue::Boolean(false)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_LONG_PROMPTS", Some("1")).value,
            CanonicalRuntimeValue::Boolean(true)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_LONG_PROMPTS", Some("true")).value,
            CanonicalRuntimeValue::Boolean(false)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_LONG_PROMPTS", Some("yes")).value,
            CanonicalRuntimeValue::Boolean(false)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_LTX2_VAE_DECODE_CHUNK_FRAMES", Some(" 4 ")).value,
            CanonicalRuntimeValue::Unsigned(mold_inference::runtime_env::parse_u64(" 4 "))
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_WUERSTCHEN_DECODER_GUIDANCE", Some(" 1.5 ")).value,
            CanonicalRuntimeValue::FloatBits(
                mold_inference::runtime_env::parse_f64(" 1.5 ").map(f64::to_bits)
            )
        );
    }

    #[test]
    fn legacy_ltx2_cpu_prompt_encoder_flag_is_presence_based() {
        for value in ["", "0", "false", "anything"] {
            assert_eq!(
                runtime_semantic_setting("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", Some(value),)
                    .value,
                CanonicalRuntimeValue::Presence(true)
            );
        }
        assert_eq!(
            runtime_semantic_setting("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", None).value,
            CanonicalRuntimeValue::Unset
        );
    }

    #[test]
    fn unknown_source_revision_is_equivalent_only_within_one_process() {
        let first = execution_code_identity_for("0.20.2", "unknown");
        let second = execution_code_identity_for("0.20.2", "unknown");
        assert_eq!(first, second);
        assert_eq!(first.scope, CodeIdentityScope::CurrentProcessOnly);
        assert!(first.source_revision.is_none());
        assert!(first.process_discriminator.is_some());

        let immutable = execution_code_identity_for("0.20.2", "0bacf81d");
        assert_eq!(immutable.scope, CodeIdentityScope::ImmutableBuild);
        assert_eq!(immutable.source_revision.as_deref(), Some("0bacf81d"));
        assert!(immutable.process_discriminator.is_none());
    }

    #[test]
    fn component_map_insertion_order_does_not_change_equivalence_identity() {
        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), name.as_bytes()).unwrap();
        }
        let config = config(root.path(), "flux2", None);
        let plan = resolve_execution_plans(&config, &request(None), &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        let mut reverse_inserted = BTreeMap::new();
        for (role, component) in plan.components.iter().rev() {
            reverse_inserted.insert(role.clone(), component.clone());
        }
        let rebuild = |components: &BTreeMap<ComponentRole, ComponentExecutionPlan>| {
            execution_environment_descriptor(
                &DeviceFact {
                    id: plan.device_id.clone(),
                    ordinal: plan.device_ordinal,
                    backend: plan.execution_environment.backend,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: plan.admitted_available_vram_bytes,
                },
                &plan.execution_environment.runtime_model_id,
                &plan.model_family,
                &plan.execution_environment.model_fingerprint,
                components,
                &plan.effective_loras,
                &plan.engine_config,
                plan.attention_backend,
                plan.engine_load_strategy,
                plan.offload_mode,
                plan.execution_environment.output_format,
                plan.determinism_class,
                false,
            )
        };

        assert_eq!(
            rebuild(&plan.components).fingerprint(),
            rebuild(&reverse_inserted).fingerprint()
        );
    }

    #[test]
    fn same_size_in_place_overwrite_changes_equivalence_content_identity() {
        use filetime::{set_file_mtime, FileTime};

        let root = TempDir::new().unwrap();
        let path = root.path().join("weights.safetensors");
        std::fs::write(&path, b"aaaa").unwrap();
        let original_mtime = FileTime::from_last_modification_time(&path.metadata().unwrap());
        let before = equivalence_fingerprint_path(&path);

        std::fs::write(&path, b"bbbb").unwrap();
        set_file_mtime(&path, original_mtime).unwrap();

        assert_ne!(before, equivalence_fingerprint_path(&path));
    }

    #[test]
    fn cache_only_equivalence_lookup_never_hashes_on_the_coordinator_path() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("weights.safetensors");
        std::fs::write(&path, b"current artifact bytes").unwrap();
        artifact_fact_cache().lock().unwrap().remove_path(&path);

        assert!(matches!(
            equivalence_fingerprint_path_with_policy(&path, true),
            EquivalenceContentIdentity::Unknown { .. }
        ));
        assert!(
            !artifact_fact_cache()
                .lock()
                .unwrap()
                .entries
                .contains_key(&path),
            "cache-only coordinator lookup must not perform byte hashing"
        );
        assert!(matches!(
            equivalence_fingerprint_path(&path),
            EquivalenceContentIdentity::Sha256(_)
        ));
    }

    #[test]
    fn coordinator_plan_resolution_never_reads_cold_artifact_facts() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), b"cold coordinator artifact").unwrap();
        }
        let config = config(root.path(), "flux2", None);
        let request = request(None);
        let paths = ModelPaths::resolve(&request.model, &config).unwrap();
        let frozen = mold_inference::FrozenEngineConfig::resolve(&request.model, &config);
        let reads = Arc::new(AtomicUsize::new(0));
        let _guards = concrete_artifacts_for_family(&paths, "flux2", &[], &frozen)
            .into_values()
            .map(|path| {
                artifact_fact_cache().lock().unwrap().remove_path(&path);
                register_artifact_read_counter(path, Arc::clone(&reads))
            })
            .collect::<Vec<_>>();

        resolve_execution_plans_for_coordinator(
            &config,
            &request,
            &devices(&[24 * GIB]),
            false,
            None,
        )
        .unwrap();

        assert_eq!(
            reads.load(Ordering::SeqCst),
            0,
            "coordinator resolution must never hash or probe artifact bytes"
        );
    }

    #[test]
    fn cache_miss_identity_changes_when_artifact_metadata_changes() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("changing.safetensors");
        std::fs::write(&path, b"aaaa").unwrap();
        let first_metadata = artifact_metadata_identity(&path, &path.metadata().unwrap());
        artifact_fact_cache().lock().unwrap().remove_path(&path);
        let first = artifact_facts_path_with_policy(&path, true).content;

        std::fs::write(&path, b"bbbb").unwrap();
        let second_metadata = artifact_metadata_identity(&path, &path.metadata().unwrap());
        assert_ne!(
            first_metadata, second_metadata,
            "fixture must observe a new metadata identity"
        );
        artifact_fact_cache().lock().unwrap().remove_path(&path);
        let second = artifact_facts_path_with_policy(&path, true).content;

        assert_ne!(
            first, second,
            "different byte states at one path must not share a degraded identity"
        );
    }

    #[test]
    fn prepared_descriptor_performs_zero_artifact_reads_after_warm() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), b"prepared-format-cache").unwrap();
        }
        let config = config(root.path(), "flux2", None);
        let request = request(None);
        let paths = ModelPaths::resolve(&request.model, &config).unwrap();
        let engine_config = mold_inference::FrozenEngineConfig::resolve(&request.model, &config);
        let prepared = PreparedExecutionInputs {
            authority_fingerprint: preparation_authority_fingerprint(
                &config,
                &request,
                &paths,
                &engine_config,
            ),
            by_device: BTreeMap::from([(
                "cuda:0".to_string(),
                PreparedDeviceExecutionInputs {
                    engine_paths: paths.clone(),
                    engine_config,
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
        };
        warm_execution_equivalence_cache(&config, &request, &prepared);

        let reads = Arc::new(AtomicUsize::new(0));
        let _guards = concrete_artifacts_for_family(
            &paths,
            "flux2",
            &[],
            &prepared.by_device["cuda:0"].engine_config,
        )
        .into_values()
        .map(|path| register_artifact_read_counter(path, Arc::clone(&reads)))
        .collect::<Vec<_>>();
        resolve_execution_plans_with_prepared(
            &config,
            &request,
            &devices(&[24 * GIB]),
            false,
            Some(&prepared),
        )
        .unwrap();
        assert_eq!(reads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn concurrent_artifact_fact_warmers_single_flight_one_physical_read() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};

        let root = TempDir::new().unwrap();
        let path = root.path().join("shared.safetensors");
        std::fs::write(&path, b"shared artifact bytes").unwrap();
        artifact_fact_cache().lock().unwrap().remove_path(&path);
        let reads = Arc::new(AtomicUsize::new(0));
        let _guard = register_artifact_read_counter(path.clone(), Arc::clone(&reads));
        let barrier = Arc::new(Barrier::new(8));
        let threads = (0..8)
            .map(|_| {
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    artifact_facts_path_with_policy(&path, false)
                })
            })
            .collect::<Vec<_>>();
        let facts = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert!(facts.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(reads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn artifact_fact_owner_unwind_removes_flight_and_releases_waiters_fail_closed() {
        use std::sync::mpsc;
        use std::time::Duration;

        let root = TempDir::new().unwrap();
        let path = root.path().join("owner-unwind.safetensors");
        std::fs::write(&path, b"owner unwind").unwrap();
        let metadata = artifact_metadata_identity(&path, &path.metadata().unwrap());
        let key = ArtifactFactKey {
            path: path.clone(),
            metadata,
        };
        let flight = std::sync::Arc::new(ArtifactFactFlight::new());
        artifact_fact_cache()
            .lock()
            .unwrap()
            .in_flight
            .insert(key.clone(), std::sync::Arc::clone(&flight));
        let (tx, rx) = mpsc::channel();
        let waiter = {
            let flight = std::sync::Arc::clone(&flight);
            std::thread::spawn(move || tx.send(flight.wait()).unwrap())
        };

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _owner = ArtifactFactOwnerGuard {
                path: path.clone(),
                key: key.clone(),
                flight,
                fallback: ArtifactFacts {
                    content: unknown_equivalence_content(&path, Some(&key.metadata)),
                    format: ArtifactFormatFact::CacheMiss,
                },
                _in_flight_permit: artifact_in_flight_limiter().acquire(),
                completed: false,
            };
            panic!("synthetic artifact probe owner abort");
        }));
        assert!(unwind.is_err());
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap().format,
            ArtifactFormatFact::CacheMiss
        );
        waiter.join().unwrap();
        assert!(
            !artifact_fact_cache()
                .lock()
                .unwrap()
                .in_flight
                .contains_key(&key),
            "an aborted owner must not strand its metadata identity"
        );
    }

    #[test]
    fn artifact_fact_cache_eviction_is_bounded_and_deterministic() {
        let mut cache = ArtifactFactCache::with_capacity(2);
        let first = PathBuf::from("/artifact/first");
        let second = PathBuf::from("/artifact/second");
        let third = PathBuf::from("/artifact/third");
        cache.insert_for_test(first.clone());
        cache.insert_for_test(second.clone());
        assert!(cache.touch_for_test(&first));
        cache.insert_for_test(third.clone());
        assert_eq!(cache.entry_paths_for_test(), vec![first, third]);
    }

    #[test]
    fn unstable_artifact_retry_cooldown_is_bounded_and_expires() {
        let mut cache = ArtifactFactCache::with_capacity(2);
        let now = std::time::Instant::now();
        let first = PathBuf::from("/artifact/first");
        let second = PathBuf::from("/artifact/second");
        let third = PathBuf::from("/artifact/third");

        cache.mark_unstable_at(first.clone(), now);
        cache.mark_unstable_at(second.clone(), now);
        assert!(cache.unstable_backoff_active_at(&first, now));
        cache.mark_unstable_at(third.clone(), now);

        assert_eq!(cache.unstable_until.len(), 2);
        assert!(
            !cache.unstable_until.contains_key(&first),
            "equal-deadline eviction must use path order deterministically"
        );
        assert!(cache.unstable_backoff_active_at(&second, now));
        assert!(
            !cache.unstable_backoff_active_at(&second, now + ARTIFACT_UNSTABLE_RETRY_COOLDOWN),
            "cooldown must not permanently suppress a stable retry"
        );
    }

    #[test]
    fn physical_artifact_read_limiter_never_exceeds_capacity() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{mpsc, Arc, Barrier};
        use std::time::Duration;

        let limiter = Arc::new(ArtifactReadLimiter::new(2));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(7));
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let threads = (0..6)
            .map(|_| {
                let limiter = Arc::clone(&limiter);
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);
                let start = Arc::clone(&start);
                let release_rx = Arc::clone(&release_rx);
                std::thread::spawn(move || {
                    start.wait();
                    let _permit = limiter.acquire();
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active.fetch_max(now, Ordering::SeqCst);
                    release_rx.lock().unwrap().recv().unwrap();
                    active.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
        for _ in 0..6 {
            release_tx.send(()).unwrap();
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn exact_execution_fingerprint_matches_rejected_candidate_contract() {
        let device = DeviceFact {
            id: "cuda:stable-device".into(),
            ordinal: 2,
            backend: GpuBackend::Cuda,
            compute_capability: Some((8, 6)),
            available_vram_bytes: 24 * GIB,
        };
        let effective = EffectivePlacement {
            components: BTreeMap::from([(
                ComponentRole::Transformer,
                ResolvedComponentConstraint::Auto,
            )]),
        };
        let components = BTreeMap::from([(
            ComponentRole::Transformer,
            ComponentExecutionPlan {
                role: ComponentRole::Transformer,
                artifact_path: PathBuf::from("/models/transformer-q4.gguf"),
                content_fingerprint: ContentFingerprint("exact-content".into()),
                dtype: None,
                quantization: Some(QuantizationVariant::Q4),
                placement: ResolvedComponentPlacement::Device("cuda:stable-device".into()),
                load_strategy: ComponentLoadStrategy::Resident,
                predicted_vram_bytes: 1024,
                predicted_host_bytes: 0,
            },
        )]);
        let engine_config = mold_inference::FrozenEngineConfig {
            family: "flux2".into(),
            is_schnell: Some(false),
            is_turbo: None,
            scheduler: None,
            t5_variant: Some("q4".into()),
            qwen3_variant: None,
            qwen2_variant: None,
            qwen2_text_encoder_mode: None,
            ltx2_gemma_variant: None,
            selected_t5_path: None,
            selected_qwen3_paths: Vec::new(),
            selected_qwen2_path: None,
            selected_gemma_paths: Vec::new(),
            runtime_environment: mold_inference::runtime_env::FrozenRuntimeEnvironment::default(),
            attention_backend: mold_inference::attention::AttentionBackend::Math,
            attention_chunk: mold_inference::attention::AttentionChunkPolicy::Auto,
            vae_tiling: mold_inference::vae_tiling::TiledMode::Auto,
            vae_dtype: mold_inference::device::VaeDtypePolicy::Auto,
        };
        let fingerprint = execution_fingerprint(
            "cv:opaque",
            &device,
            &effective,
            &components,
            &engine_config,
            &[],
            false,
        );
        assert_eq!(
            fingerprint, "6148b3759215b8e2082e7e0ac02d0b31bc5d29841e73d4625f1140d77bb200d8",
            "this is the exact path/device-qualified candidate 0bacf81d contract"
        );
    }

    #[test]
    fn execution_equivalence_v3_schema_and_hash_are_golden() {
        let content = EquivalenceContentIdentity::Sha256("00".repeat(32));
        let descriptor = ExecutionEnvironmentDescriptor {
            schema_version: 3,
            backend: GpuBackend::Cuda,
            architecture: DeviceArchitectureClass::CudaComputeCapability { major: 8, minor: 6 },
            attention_kernel_class: AttentionKernelClass::Math,
            code: ExecutionCodeIdentity {
                package_version: "0.20.2".into(),
                source_revision: Some("0bacf81d".into()),
                scope: CodeIdentityScope::ImmutableBuild,
                process_discriminator: None,
            },
            semantic_config: ExecutionSemanticConfig {
                family: "flux".into(),
                is_schnell: Some(false),
                is_turbo: None,
                scheduler: None,
                t5_variant: Some("q4".into()),
                qwen3_variant: None,
                qwen2_variant: None,
                qwen2_text_encoder_mode: None,
                ltx2_gemma_variant: None,
                attention_backend: SemanticAttentionBackend::Math,
                attention_chunk: SemanticAttentionChunk::Auto,
                vae_tiling: SemanticVaeTiling::Auto,
                vae_dtype: SemanticVaeDType::Auto,
                runtime: vec![RuntimeSemanticSetting {
                    variable: RuntimeSemanticVariable::CfgPlus,
                    value: CanonicalRuntimeValue::Boolean(false),
                }],
            },
            runtime_model_id: "flux-dev:bf16".into(),
            runtime_artifact_paths: vec![RuntimeArtifactPathIdentity {
                role: ComponentRole::Transformer,
                path: RuntimePathIdentity::Portable("/models/transformer.safetensors".to_string()),
            }],
            model_family: "flux".into(),
            model_fingerprint: "model-content".into(),
            components: vec![EquivalentComponentExecution {
                role: ComponentRole::Transformer,
                content_fingerprint: content.clone(),
                precision: EffectiveComponentPrecision {
                    storage: ComponentStorageFormat::Unknown {
                        reason: ArtifactFormatUnknown::UnsupportedContainer,
                        content_discriminator: content,
                    },
                    compute_dtype: EffectiveComponentDType::Bf16,
                },
                dtype: Some(PlannedDType::Bf16),
                quantization: None,
                placement: SemanticComponentPlacement::AssignedDevice,
                load_strategy: ComponentLoadStrategy::Resident,
            }],
            loras: Vec::new(),
            engine_load_strategy: EngineLoadStrategyClass::Eager,
            offload_mode: OffloadMode::None,
            output_format: OutputFormat::Png,
            determinism_class: DeterminismClass::CpuSeededCrossBackend,
        };
        let encoded = serde_json::to_string(&descriptor).unwrap();
        assert_eq!(
            encoded,
            r#"{"schema_version":3,"backend":"cuda","architecture":{"CudaComputeCapability":{"major":8,"minor":6}},"attention_kernel_class":"Math","code":{"package_version":"0.20.2","source_revision":"0bacf81d","scope":"ImmutableBuild","process_discriminator":null},"semantic_config":{"family":"flux","is_schnell":false,"is_turbo":null,"scheduler":null,"t5_variant":"q4","qwen3_variant":null,"qwen2_variant":null,"qwen2_text_encoder_mode":null,"ltx2_gemma_variant":null,"attention_backend":"Math","attention_chunk":"Auto","vae_tiling":"Auto","vae_dtype":"Auto","runtime":[{"variable":"CfgPlus","value":{"Boolean":false}}]},"runtime_model_id":"flux-dev:bf16","runtime_artifact_paths":[{"role":"Transformer","path":{"Portable":"/models/transformer.safetensors"}}],"model_family":"flux","model_fingerprint":"model-content","components":[{"role":"Transformer","content_fingerprint":{"Sha256":"0000000000000000000000000000000000000000000000000000000000000000"},"precision":{"storage":{"Unknown":{"reason":"UnsupportedContainer","content_discriminator":{"Sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}},"compute_dtype":"Bf16"},"dtype":"Bf16","quantization":null,"placement":"AssignedDevice","load_strategy":"Resident"}],"loras":[],"engine_load_strategy":"Eager","offload_mode":"None","output_format":"png","determinism_class":"CpuSeededCrossBackend"}"#
        );
        assert_eq!(
            descriptor.fingerprint().as_str(),
            "8a7ad345c90e7dde228c90ee4a219986ad705a0958ead00d143ead84bef72be7"
        );
    }

    #[test]
    fn frozen_chain_model_uses_canonical_companions_and_ignores_changed_config() {
        let root = TempDir::new().unwrap();
        let assets = root.path().join("assets");
        let nested = assets.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let names = [
            "transformer.safetensors",
            "shard.safetensors",
            "vae.safetensors",
            "spatial.safetensors",
            "temporal.safetensors",
            "distilled.safetensors",
            "t5.safetensors",
            "clip.safetensors",
            "clip2.safetensors",
            "t5-tokenizer.json",
            "clip-tokenizer.json",
            "clip2-tokenizer.json",
            "text_projection.safetensors",
            "tokenizer.json",
            "decoder.safetensors",
            "default-lora.safetensors",
        ];
        for name in names {
            std::fs::write(assets.join(name), name.as_bytes()).unwrap();
        }
        let relative_to_nested = |name: &str| nested.join("..").join(name);
        let model = "cv:freeze-fixture";
        let mut config = Config::default();
        config.models.insert(
            model.to_string(),
            mold_core::ModelConfig {
                transformer: Some(
                    relative_to_nested("transformer.safetensors")
                        .display()
                        .to_string(),
                ),
                transformer_shards: Some(vec![relative_to_nested("shard.safetensors")
                    .display()
                    .to_string()]),
                vae: Some(relative_to_nested("vae.safetensors").display().to_string()),
                spatial_upscaler: Some(
                    relative_to_nested("spatial.safetensors")
                        .display()
                        .to_string(),
                ),
                temporal_upscaler: Some(
                    relative_to_nested("temporal.safetensors")
                        .display()
                        .to_string(),
                ),
                distilled_lora: Some(
                    relative_to_nested("distilled.safetensors")
                        .display()
                        .to_string(),
                ),
                t5_encoder: Some(relative_to_nested("t5.safetensors").display().to_string()),
                clip_encoder: Some(relative_to_nested("clip.safetensors").display().to_string()),
                clip_encoder_2: Some(
                    relative_to_nested("clip2.safetensors")
                        .display()
                        .to_string(),
                ),
                t5_tokenizer: Some(
                    relative_to_nested("t5-tokenizer.json")
                        .display()
                        .to_string(),
                ),
                clip_tokenizer: Some(
                    relative_to_nested("clip-tokenizer.json")
                        .display()
                        .to_string(),
                ),
                clip_tokenizer_2: Some(
                    relative_to_nested("clip2-tokenizer.json")
                        .display()
                        .to_string(),
                ),
                text_encoder_files: Some(vec![relative_to_nested("text_projection.safetensors")
                    .display()
                    .to_string()]),
                text_tokenizer: Some(relative_to_nested("tokenizer.json").display().to_string()),
                decoder: Some(
                    relative_to_nested("decoder.safetensors")
                        .display()
                        .to_string(),
                ),
                lora: Some(
                    relative_to_nested("default-lora.safetensors")
                        .display()
                        .to_string(),
                ),
                family: Some("ltx2".to_string()),
                ..mold_core::ModelConfig::default()
            },
        );

        let frozen = freeze_chain_model(&config, model).unwrap();
        assert!(frozen.runtime_model_id.starts_with("mold-frozen-chain:"));
        for path in frozen.config.all_file_paths() {
            let path = PathBuf::from(path);
            assert!(path.is_absolute());
            assert!(!path.components().any(|part| part.as_os_str() == ".."));
            assert!(path.is_file());
        }
        assert!(Path::new(frozen.config.lora.as_deref().unwrap()).is_absolute());
        assert_eq!(
            frozen.config.text_encoder_files.as_ref().unwrap()[0],
            assets
                .join("text_projection.safetensors")
                .canonicalize()
                .unwrap()
                .display()
                .to_string()
        );
        let original_fingerprint = frozen.model_fingerprint.clone();
        std::fs::write(assets.join("t5-tokenizer.json"), b"changed-tokenizer").unwrap();
        assert_ne!(
            frozen_model_fingerprint(model, &frozen.config).unwrap(),
            original_fingerprint,
            "tokenizers are engine inputs and must invalidate the durable identity"
        );
        std::fs::write(assets.join("t5-tokenizer.json"), b"t5-tokenizer.json").unwrap();
        std::fs::write(
            assets.join("default-lora.safetensors"),
            b"changed-default-lora",
        )
        .unwrap();
        assert_ne!(
            frozen_model_fingerprint(model, &frozen.config).unwrap(),
            original_fingerprint,
            "the configured default LoRA must invalidate the durable identity"
        );
        std::fs::write(
            assets.join("default-lora.safetensors"),
            b"default-lora.safetensors",
        )
        .unwrap();
        let mut changed_engine_defaults = frozen.config.clone();
        changed_engine_defaults.lora_scale = Some(0.25);
        assert_ne!(
            frozen_model_fingerprint(model, &changed_engine_defaults).unwrap(),
            original_fingerprint,
            "serialized engine-shaping defaults must participate in identity"
        );

        let changed = root.path().join("changed.safetensors");
        std::fs::write(&changed, b"changed").unwrap();
        let mut changed_config = config;
        changed_config.models.insert(
            model.to_string(),
            mold_core::ModelConfig {
                transformer: Some(changed.display().to_string()),
                vae: Some(changed.display().to_string()),
                family: Some("ltx2".to_string()),
                ..mold_core::ModelConfig::default()
            },
        );
        changed_config.install_frozen_model_config(model, frozen.config.clone());
        let paths = ModelPaths::resolve(model, &changed_config).unwrap();
        assert_eq!(
            paths.transformer,
            assets
                .join("transformer.safetensors")
                .canonicalize()
                .unwrap()
        );
        assert!(changed_config.has_frozen_model_config(model));
    }
}
