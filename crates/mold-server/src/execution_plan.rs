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
#[cfg(not(any(unix, windows)))]
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

const MIB: u64 = 1024 * 1024;
const BASE_HOST_TRANSIENT: u64 = 256 * MIB;
const UNKNOWN_ARTIFACT_HOST_CHARGE: u64 = 64 * MIB;

/// Semantic position of an artifact consumed by one engine execution.
///
/// Indexed roles retain the native `Vec` index so building the keyed artifact
/// topology cannot truncate, alias, and replace an earlier component. Serde's
/// numeric JSON representation remains unchanged for every previously valid
/// index.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum ComponentRole {
    Transformer,
    TransformerShard(usize),
    /// The low-noise expert of a two-expert pair (Wan 2.2 A14B). It is a
    /// separate artifact rather than a shard: the plan has to fingerprint and
    /// pre-validate it in its own right, because it is not read until the
    /// schedule crosses the expert boundary — long after admission.
    LowNoiseTransformer,
    Vae,
    T5,
    T5Tokenizer,
    ClipL,
    ClipLTokenizer,
    ClipG,
    ClipGTokenizer,
    QwenShard(usize),
    GemmaShard(usize),
    GenericTextEncoderShard(usize),
    TextTokenizer,
    Lora(usize),
    SpatialUpscaler,
    TemporalUpscaler,
    /// Split LTX-2.5 checkpoint containing both audio VAE and vocoder.
    AudioVae,
    /// Split LTX-2.5 caption-conditioned duration predictor.
    DurationHead,
    Decoder,
    DistilledLora,
    /// The distill belonging to [`ComponentRole::LowNoiseTransformer`]. Each
    /// expert of a pair is distilled separately, so the two adapters are
    /// distinct artifacts, not one applied twice.
    LowNoiseDistilledLora,
    /// PuLID's identity adapter (IDFormer + cross-attention weights). Loads on
    /// the generation device beside the transformer it conditions.
    IdentityAdapter,
    /// The EVA02-CLIP-L-14-336 vision tower PuLID encodes the reference face
    /// with. CPU-only: the tower runs once at admission, on the host, and the
    /// generation device never sees it.
    IdentityVisionEncoder,
    /// InsightFace SCRFD face detector. ONNX, and CPU-only in milestone 1.
    FaceDetector,
    /// InsightFace ArcFace recognizer. ONNX, and CPU-only in milestone 1.
    FaceRecognizer,
    /// facexlib's BiSeNet face parser. Runs on the host beside the other two,
    /// masking the aligned crop before the vision tower sees it (#1225).
    FaceParser,
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

    /// [`Self::is_host_only`] for a sibling module's test.
    #[cfg(test)]
    pub(crate) fn is_host_only_for_test(&self) -> bool {
        self.is_host_only()
    }

    fn is_host_only(&self) -> bool {
        matches!(
            self,
            Self::T5Tokenizer
                | Self::ClipLTokenizer
                | Self::ClipGTokenizer
                | Self::TextTokenizer
                | Self::Lora(_)
                // The whole identity EXTRACTION runs on the host, at
                // admission, before the scheduler has leased a device (#1223):
                // both InsightFace ONNX graphs, and the EVA02-CLIP tower that
                // turns the aligned crop into the IDFormer's five hidden
                // states, and the BiSeNet parser that masks that crop.
                // None of the four ever touches the generation
                // device, so their bytes are host demand and never VRAM.
                //
                // `IdentityAdapter` is deliberately NOT in this list: its
                // twenty cross-attention modules are the one identity artifact
                // that IS resident on the generation device, for the whole
                // denoise. It is also the file the IDFormer weights live in,
                // which is why the extraction reads it on the host too — but
                // the residency that matters for placement is the device one.
                | Self::FaceDetector
                | Self::FaceRecognizer
                | Self::FaceParser
                | Self::IdentityVisionEncoder
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
    /// A read-only preview dependency whose immutable registry identity is
    /// known but whose bytes have not landed yet. This domain is never used
    /// by admission or worker leases.
    PendingPreview {
        repo: String,
        filename: String,
        bytes: u64,
    },
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

/// One equivalence class per entry in `mold_inference::runtime_env::
/// ENGINE_SHAPING_VARIABLES`. Adding a name to that list requires a matching
/// variant and `runtime_semantic_variable` arm here;
/// `every_engine_shaping_variable_has_a_semantic_class` enforces the pairing.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum RuntimeSemanticVariable {
    Attn,
    AttnChunk,
    CfgPlus,
    Device,
    Eager,
    FluxDeltaCache,
    FluxKeepTransformer,
    H3TurboAdapter,
    H3TurboTier,
    KeepTeRam,
    LongPrompts,
    LoraBypass,
    LtxDebugAltPrompt,
    LtxDebugCompareUncond,
    LtxDebugDisableCrossAttentionAdaLn,
    Ltx2DebugDisableTransformerGatedAttention,
    Ltx2DebugForceCpuPromptEncoder,
    Ltx2DebugLoadBlocks,
    Ltx2AttnF32,
    Ltx2ForceEager,
    Ltx2ForceStreaming,
    Ltx2Fp8InputScaleMode,
    Ltx2Fp8WeightScaleMode,
    Ltx2GemmaDevice,
    Ltx2GemmaVariant,
    Ltx2Int8,
    Ltx2QMatMul,
    Umt5Variant,
    Ltx2SpatialTile,
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
    QwenFp8Cache,
    QwenQMatMul,
    ReserveVramMb,
    T5Variant,
    VaeDtype,
    VaeTiled,
    WanForceDmmv,
    WanPrefetch,
    WanOffloadBlocks,
    WanStepCache,
    WanStepProfile,
    WuerstchenDecoderGuidance,
    ZimageQMatMul,
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
    /// Explicit Wan UMT5 encoder variant.
    ///
    /// A different encoder produces different embeddings and therefore a
    /// different render, so it belongs in the execution-equivalence identity.
    /// Skipped when absent so fingerprints for every family that has no UMT5
    /// stay exactly as they were.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub umt5_variant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h3_factory_authority_sha256: Option<String>,
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
    /// Registry-declared container/quantization facts for an artifact that a
    /// read-only preview has not materialized yet. This is deliberately a
    /// separate domain from probed bytes and is replaced at admission.
    PendingPreview {
        container: PendingArtifactContainer,
        artifact_identity: EquivalenceContentIdentity,
        /// `None` for an unquantized pending artifact. Serde renders `Some(q)`
        /// exactly as the pre-`Option` field did, so every previously emitted
        /// GGUF descriptor keeps its bytes.
        quantization: Option<QuantizationVariant>,
    },
    Unknown {
        reason: ArtifactFormatUnknown,
        content_discriminator: EquivalenceContentIdentity,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum PendingArtifactContainer {
    Gguf,
    Safetensors,
    /// A PyTorch pickle archive carried as a conversion input (the EVA-CLIP
    /// vision tower release), never handed to a tensor loader as-is.
    TorchArchive,
    Onnx,
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
    pub fn from_frozen(
        frozen: &mold_inference::FrozenEngineConfig,
    ) -> Result<Self, ExecutionPlanError> {
        let mold_inference::FrozenEngineConfig {
            family,
            artifact_root: _,
            is_schnell,
            is_turbo,
            scheduler,
            t5_variant,
            qwen3_variant,
            qwen2_variant,
            qwen2_text_encoder_mode,
            ltx2_gemma_variant,
            umt5_variant,
            // Selected artifacts are represented by component content/format
            // facts and the conservative exact runtime route in the enclosing
            // descriptor.
            selected_t5_path: _,
            selected_qwen3_paths: _,
            selected_qwen2_path: _,
            selected_gemma_paths: _,
            selected_umt5_path: _,
            // Identity assets are represented by their own component
            // content/format facts in the enclosing descriptor, exactly like
            // the selected encoder artifacts above.
            identity_assets: _,
            h3_factory_authority,
            runtime_environment,
            attention_backend,
            attention_chunk,
            vae_tiling,
            vae_dtype,
        } = frozen;
        let runtime = mold_inference::runtime_env::ENGINE_SHAPING_VARIABLES
            .iter()
            .map(|name| {
                runtime_semantic_setting(name, runtime_environment.value(name)).ok_or_else(|| {
                    ExecutionPlanError::UnclassifiedRuntimeVariable {
                        name: (*name).to_string(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            family: family.clone(),
            is_schnell: *is_schnell,
            is_turbo: *is_turbo,
            scheduler: *scheduler,
            t5_variant: t5_variant.clone(),
            qwen3_variant: qwen3_variant.clone(),
            qwen2_variant: qwen2_variant.clone(),
            qwen2_text_encoder_mode: qwen2_text_encoder_mode.clone(),
            ltx2_gemma_variant: ltx2_gemma_variant.clone(),
            umt5_variant: umt5_variant.clone(),
            h3_factory_authority_sha256: h3_factory_authority
                .as_ref()
                .map(|authority| authority.identity_sha256().to_string()),
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
        })
    }
}

/// Classifies an engine-shaping environment variable into its execution-
/// equivalence class, or `None` for a name this build does not know.
///
/// The list of names lives in `mold_inference::runtime_env::
/// ENGINE_SHAPING_VARIABLES`; every entry added there must gain an arm here.
/// `every_engine_shaping_variable_has_a_semantic_class` fails CI on a
/// mismatch, and at runtime an unclassified name fails the job with
/// `ExecutionPlanError::UnclassifiedRuntimeVariable` instead of panicking the
/// server (issue #685).
fn runtime_semantic_variable(name: &str) -> Option<RuntimeSemanticVariable> {
    let variable = match name {
        "MOLD_ATTN" => RuntimeSemanticVariable::Attn,
        "MOLD_ATTN_CHUNK" => RuntimeSemanticVariable::AttnChunk,
        "MOLD_CFG_PLUS" => RuntimeSemanticVariable::CfgPlus,
        "MOLD_DEVICE" => RuntimeSemanticVariable::Device,
        "MOLD_EAGER" => RuntimeSemanticVariable::Eager,
        "MOLD_FLUX_DELTA_CACHE" => RuntimeSemanticVariable::FluxDeltaCache,
        "MOLD_FLUX_KEEP_TRANSFORMER" => RuntimeSemanticVariable::FluxKeepTransformer,
        "MOLD_H3_TURBO_ADAPTER" => RuntimeSemanticVariable::H3TurboAdapter,
        "MOLD_H3_TURBO_TIER" => RuntimeSemanticVariable::H3TurboTier,
        "MOLD_KEEP_TE_RAM" => RuntimeSemanticVariable::KeepTeRam,
        "MOLD_LONG_PROMPTS" => RuntimeSemanticVariable::LongPrompts,
        "MOLD_LORA_BYPASS" => RuntimeSemanticVariable::LoraBypass,
        "MOLD_LTX_DEBUG_ALT_PROMPT" => RuntimeSemanticVariable::LtxDebugAltPrompt,
        "MOLD_LTX_DEBUG_COMPARE_UNCOND" => RuntimeSemanticVariable::LtxDebugCompareUncond,
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
        // #735: forces the F32 chunked LTX-2 attention path in place of the
        // BF16 dispatcher, which changes the rendered output — its own
        // execution-equivalence and timing class.
        "MOLD_LTX2_ATTN_F32" => RuntimeSemanticVariable::Ltx2AttnF32,
        "MOLD_LTX2_FORCE_EAGER" => RuntimeSemanticVariable::Ltx2ForceEager,
        "MOLD_LTX2_FORCE_STREAMING" => RuntimeSemanticVariable::Ltx2ForceStreaming,
        "MOLD_LTX2_FP8_INPUT_SCALE_MODE" => RuntimeSemanticVariable::Ltx2Fp8InputScaleMode,
        "MOLD_LTX2_FP8_WEIGHT_SCALE_MODE" => RuntimeSemanticVariable::Ltx2Fp8WeightScaleMode,
        "MOLD_LTX2_GEMMA_DEVICE" => RuntimeSemanticVariable::Ltx2GemmaDevice,
        "MOLD_LTX2_GEMMA_VARIANT" => RuntimeSemanticVariable::Ltx2GemmaVariant,
        // Swaps the LTX-2 INT8 ConvRot execution arm (W8A8 kernel vs
        // per-forward widening), which changes pixels, transient memory, and
        // step latency — its own execution-equivalence and timing class.
        "MOLD_LTX2_INT8" => RuntimeSemanticVariable::Ltx2Int8,
        // Swaps the LTX-2.5 GGUF linear arm (per-forward dequant vs candle's
        // QMatMul fast path), which changes numerics, transient memory, and
        // step latency — its own execution-equivalence and timing class.
        "MOLD_LTX2_QMATMUL" => RuntimeSemanticVariable::Ltx2QMatMul,
        "MOLD_UMT5_VARIANT" => RuntimeSemanticVariable::Umt5Variant,
        "MOLD_LTX2_SPATIAL_TILE" => RuntimeSemanticVariable::Ltx2SpatialTile,
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
        "MOLD_QWEN_FP8_CACHE" => RuntimeSemanticVariable::QwenFp8Cache,
        "MOLD_QWEN_QMATMUL" => RuntimeSemanticVariable::QwenQMatMul,
        "MOLD_RESERVE_VRAM_MB" => RuntimeSemanticVariable::ReserveVramMb,
        "MOLD_T5_VARIANT" => RuntimeSemanticVariable::T5Variant,
        "MOLD_VAE_DTYPE" => RuntimeSemanticVariable::VaeDtype,
        "MOLD_VAE_TILED" => RuntimeSemanticVariable::VaeTiled,
        // #775 diagnostics: forcing the quantized-matmul fallback changes
        // numerics/runtime/memory; the profiler's per-phase syncs change
        // runtime. Distinct classes keep their fingerprints and learned
        // timings out of normal buckets.
        "MOLD_WAN_FORCE_DMMV" => RuntimeSemanticVariable::WanForceDmmv,
        // #802 item 3: changes wall-clock only, but that is enough to keep it
        // out of another run's learned-timing bucket.
        "MOLD_WAN_PREFETCH" => RuntimeSemanticVariable::WanPrefetch,
        // #801: residual reuse changes the rendered output and the step
        // count that actually runs, so it is its own equivalence class.
        "MOLD_WAN_STEP_CACHE" => RuntimeSemanticVariable::WanStepCache,
        // #776 item 3: block parking changes device residency and step
        // latency, so it is its own timing class.
        "MOLD_WAN_OFFLOAD_BLOCKS" => RuntimeSemanticVariable::WanOffloadBlocks,
        "MOLD_WAN_STEP_PROFILE" => RuntimeSemanticVariable::WanStepProfile,
        "MOLD_WUERSTCHEN_DECODER_GUIDANCE" => RuntimeSemanticVariable::WuerstchenDecoderGuidance,
        // Swaps the Z-Image quantized linear implementation on CUDA, which
        // changes numerics, transient memory, and step latency — its own
        // execution-equivalence and timing class.
        "MOLD_ZIMAGE_QMATMUL" => RuntimeSemanticVariable::ZimageQMatMul,
        // A new engine-shaping input must never silently collapse into an old
        // equivalence class. `None` here surfaces as a per-job
        // `UnclassifiedRuntimeVariable` planning error and as a named CI
        // failure, never as a corrupted fingerprint or a server abort.
        _ => return None,
    };
    Some(variable)
}

fn runtime_semantic_setting(name: &str, value: Option<&str>) -> Option<RuntimeSemanticSetting> {
    let variable = runtime_semantic_variable(name)?;
    let value = match value {
        None => CanonicalRuntimeValue::Unset,
        Some(value)
            if matches!(
                variable,
                RuntimeSemanticVariable::LtxDebugCompareUncond
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
        // Both mirror the engine's own parsers exactly (`parse_qwen_qmatmul`
        // and `parse_qwen_fp8_cache`), so the canonical value is the decision
        // rather than the spelling.
        Some(value) if variable == RuntimeSemanticVariable::QwenQMatMul => {
            CanonicalRuntimeValue::Boolean(matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            ))
        }
        Some(value) if variable == RuntimeSemanticVariable::QwenFp8Cache => {
            CanonicalRuntimeValue::Boolean(value.trim() == "1")
        }
        // Mirrors the engine's `parse_ltx2_int8_arm` exactly: `dequant`
        // selects the widening arm, every other spelling is the default.
        Some(value) if variable == RuntimeSemanticVariable::Ltx2Int8 => {
            CanonicalRuntimeValue::Boolean(value.trim().eq_ignore_ascii_case("dequant"))
        }
        // Mirrors the engine's `parse_attention_f32_forced` exactly: the LTX-2
        // F32 attention control shares the family's truthy spellings.
        Some(value) if variable == RuntimeSemanticVariable::Ltx2AttnF32 => {
            CanonicalRuntimeValue::Boolean(matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            ))
        }
        // Mirrors the shared `parse_qmatmul_flag` the engine reads exactly.
        Some(value) if variable == RuntimeSemanticVariable::Ltx2QMatMul => {
            CanonicalRuntimeValue::Boolean(matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            ))
        }
        // Mirrors the engine's `parse_zimage_qmatmul` exactly.
        Some(value) if variable == RuntimeSemanticVariable::ZimageQMatMul => {
            CanonicalRuntimeValue::Boolean(matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            ))
        }
        // Do not invent normalization for a runtime parser we have not made
        // authoritative here. Exact text is conservative: it can cause false
        // negatives, but never a false-equivalent execution class.
        Some(value) => CanonicalRuntimeValue::Text(value.to_string()),
    };
    Some(RuntimeSemanticSetting { variable, value })
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
    /// Backend of the device this plan was resolved against. Admission
    /// arithmetic branches on it: CUDA gates VRAM and host RAM as the two
    /// separate pools they are, while Metal collapses both claims onto the
    /// one unified pool (#1038).
    pub device_backend: GpuBackend,
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
    /// Decayed observed high-water envelope for this exact estimate bucket.
    ///
    /// Scheduler admission already reserves `max(static, learned)` capacity
    /// (`mold_scheduler::estimates`), but the worker's pre-load recheck only
    /// looked at `predicted_vram_peak_bytes`. Carrying the envelope lets the
    /// worker recheck the larger of the two and reject an impossible load in
    /// milliseconds instead of after another two-minute load (#641). `0` means
    /// no learned evidence.
    pub learned_vram_envelope_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    /// Host bytes one request allocates on a device whose resident engine
    /// already holds this plan (`GpuWorker::holds_execution_fingerprint`).
    ///
    /// A CPU-parked encoder is retained for the engine's life — FLUX drops
    /// its T5 only when `on_gpu` (`flux/pipeline.rs`), so a CPU-placed T5 on
    /// CUDA never leaves host RAM — and a streamed block file's host copy is
    /// retained the same way. `MemAvailable`, the ledger's own input, already
    /// excludes those pages, so charging `predicted_host_increment_bytes` to a
    /// warm hit double-counts memory the engine is holding and parks the job
    /// on `insufficient_host_ram` forever while the GPU idles (hal9000,
    /// 2026-08-27: 9.85 GB of headroom against the 10.5 GB cold figure for a
    /// resident `flux-dev:q8`; both queued prints dispatched within a second
    /// of the model being unloaded by hand). Only per-request heaps recur:
    /// the base transient and LTX-2's CPU Gemma streaming peak, which is a
    /// forward-loop allocation.
    pub predicted_warm_host_increment_bytes: u64,
    pub determinism_class: DeterminismClass,
    pub execution_environment: ExecutionEnvironmentDescriptor,
    pub execution_equivalence_fingerprint: ExecutionEquivalenceFingerprint,
    /// Exact, device-qualified worker/lease identity. This remains the
    /// authority for residency, grants, cache reconstruction, and provenance.
    pub execution_fingerprint: String,
}

impl ResolvedExecutionPlan {
    /// Host-charged bytes that never coexist with the device peak.
    ///
    /// A CPU-parked text encoder is loaded, used, and dropped before the
    /// transformer is built on every family (`wan/pipeline.rs` documents the
    /// order as a VRAM requirement; LTX-2's runtime `take()`s its prompt
    /// encoder the same way), so its bytes and the denoise peak are two
    /// phases of one lifetime, not a sum. A CPU-pinned transformer or a
    /// streamed block file stays out of this figure: those genuinely coexist
    /// with the device peak.
    pub fn predicted_phase_disjoint_host_bytes(&self) -> u64 {
        self.components
            .values()
            .filter(|component| {
                matches!(component.load_strategy, ComponentLoadStrategy::ParkedCpu)
                    && component.role.is_text_encoder()
            })
            .map(|component| component.predicted_host_bytes)
            .fold(0u64, u64::saturating_add)
    }

    /// The demand admission must prove against the device pool.
    ///
    /// CUDA: the raw predicted VRAM peak — host RAM is a separate pool with
    /// its own gate. Metal: both claims land on one unified pool
    /// (`memory_preflight`'s worker gate already models it that way), so the
    /// demand is the larger of the two phases — the encoder phase (its
    /// CPU-parked bytes, which on unified memory are the same physical pages
    /// a device placement would use) and the denoise peak — plus whatever
    /// host charge genuinely coexists with the peak (#1038).
    pub fn admission_vram_demand_bytes(&self) -> u64 {
        match self.device_backend {
            GpuBackend::Metal => {
                let disjoint = self.predicted_phase_disjoint_host_bytes();
                let concurrent = self.predicted_host_increment_bytes.saturating_sub(disjoint);
                self.predicted_vram_peak_bytes
                    .max(disjoint)
                    .saturating_add(concurrent)
            }
            GpuBackend::Cuda => self.predicted_vram_peak_bytes,
        }
    }

    /// The demand admission must prove against host RAM headroom.
    ///
    /// Zero on Metal: the host claim is already folded into
    /// [`Self::admission_vram_demand_bytes`], and gating it a second time
    /// against a second sample of the same physical pool — minus a safety
    /// floor the device gate does not pay — is exactly the #1038
    /// double-count.
    pub fn admission_host_demand_bytes(&self) -> u64 {
        match self.device_backend {
            GpuBackend::Metal => 0,
            GpuBackend::Cuda => self.predicted_host_increment_bytes,
        }
    }

    /// The host demand a warm hit must prove — see
    /// [`Self::predicted_warm_host_increment_bytes`]. Zero on Metal for the
    /// same reason as [`Self::admission_host_demand_bytes`].
    pub fn admission_warm_host_demand_bytes(&self) -> u64 {
        match self.device_backend {
            GpuBackend::Metal => 0,
            GpuBackend::Cuda => self.predicted_warm_host_increment_bytes,
        }
    }
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

/// The one identity a batch parent's siblings share, filled by whichever of
/// them extracts first.
///
/// #1227 phase 2 moved extraction into each lease, so the "one embedding per
/// parent" guarantee that used to come from preparation ordering now has to
/// come from somewhere else. The per-photograph cache and its single flight
/// give it *within* a window: concurrent siblings compose once and take the
/// winner's tokens. But that cache is a bounded 16-entry LRU and an
/// accelerator, not an authority — sixteen distinct photographs extracted
/// between one wave of siblings and the next, or before a retry, evict the
/// parent's entry and a later child re-extracts, possibly on another GPU and
/// therefore with an embedding that differs at the measured device tolerance.
/// Exact sibling reuse is the contract; equal-within-tolerance is not equal.
///
/// So the batch plan owns this cell, every child clones the same `Arc`, and the
/// first child to extract pins its result for every later one — for the
/// lifetime of the batch, independent of what the LRU does in between.
///
/// Empty (and cheap) for every request that conditions on no face.
#[derive(Clone, Debug, Default)]
pub struct IdentityPin(std::sync::Arc<std::sync::OnceLock<PinnedIdentity>>);

/// What a parent pins: the identity AND the advisory that came with it.
///
/// The advisory is here rather than beside it because a later child skips the
/// resolver entirely on a pin hit, and post-lease preparation leaves
/// `identity_warning` empty — so a child that took the pin would silently drop
/// "several faces were found, the largest was used". The person who supplied a
/// group photograph needs that on every print of the batch, not just on
/// whichever sibling happened to extract.
#[derive(Clone, Debug, PartialEq)]
pub struct PinnedIdentity {
    pub embedding: mold_core::identity::FrozenIdentityEmbedding,
    pub warning: Option<String>,
}

impl IdentityPin {
    /// The pinned identity and advisory, if a sibling has already extracted.
    pub fn get(&self) -> Option<PinnedIdentity> {
        self.0.get().cloned()
    }

    /// Pin `value`, and return whatever is pinned afterwards.
    ///
    /// The return matters more than the call: a loser gets the WINNER's
    /// embedding back and must use that, discarding its own. Returning `()` and
    /// letting the caller keep its own value would leave two siblings holding
    /// two tolerance-different identities while looking like it had worked.
    pub fn pin(&self, value: PinnedIdentity) -> PinnedIdentity {
        let _ = self.0.set(value);
        self.0
            .get()
            .cloned()
            .expect("a cell that was just set holds a value")
    }
}

impl PartialEq for IdentityPin {
    fn eq(&self, other: &Self) -> bool {
        self.0.get() == other.0.get()
    }
}

/// One queued generation held back by memory that another render is using.
///
/// The question a park answers is about the RESOURCE, never about who happens
/// to be busy at the instant admission sampled: "could this machine ever run
/// this?". A shortfall the hardware can satisfy waits for the fleet to settle;
/// one it cannot is refused immediately, with both numbers. Deciding on the
/// busy set instead made the outcome a race — a host shortfall observed while
/// every worker was momentarily idle became a permanent hold, and a device
/// shortfall never waited at all.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CapacityPark {
    /// The typed shortfall's own sentence, retained so a park that outlives an
    /// idle grace is refused with its numbers rather than "never scheduled".
    pub reason: String,
    /// Devices whose settling could change the answer. Preparation re-runs
    /// once none of them is busy.
    pub retry_after_devices: BTreeSet<String>,
}

/// Concrete engine inputs produced by asynchronous dependency preparation.
///
/// The map is keyed by stable runtime device id because mixed-capacity hosts
/// can legitimately select different encoder variants for different GPUs.
#[derive(Clone, Debug, Default, PartialEq)]
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
    /// A memory shortfall this hardware could satisfy once the fleet settles.
    ///
    /// Preparation leaves the generation queued rather than refusing it, and
    /// retries once every named device is idle. Placement preview turns this
    /// marker into a non-authoritative answer so compatible clients take their
    /// established direct-queue fallback. This is distinct from an ordinary
    /// retryable device failure: polling admission while a long render still
    /// owns the memory would repeatedly hash the full artifact set without
    /// changing the answer.
    pub capacity_park: Option<CapacityPark>,
    /// Runtime-only catalog config synthesized from an installed sidecar.
    ///
    /// The scheduler applies this overlay when the current config lacks the
    /// opaque model id. Carrying it with prepared work makes preview,
    /// admission, and pre-CUDA validation independent of endpoint call order
    /// and of `refresh_config()` erasing in-memory catalog entries.
    pub model_config_overlay: Option<Arc<mold_core::ModelConfig>>,
    /// The face identity this parent request conditions on, extracted ONCE.
    ///
    /// It lives here rather than in `FrozenEngineConfig` for two reasons that
    /// both matter. The engine is cached across requests while the identity is
    /// per request, so putting it in the engine's construction config would
    /// rebuild the engine for every new face; and this struct is exactly what
    /// the batch parent clones into every child
    /// (the durable feeder → `PreparedExecutionInputs`),
    /// which is what makes "one extraction per parent, reused by every sibling
    /// on every device" structural instead of a convention.
    ///
    /// `None` covers every request that does not condition on a face, an
    /// explicit `id_weight` of 0, and a build without the `pulid` feature.
    pub identity_embedding: Option<mold_core::identity::FrozenIdentityEmbedding>,
    /// A caller-facing advisory the identity extraction produced — today only
    /// "several faces were found, the largest was used".
    ///
    /// It rides here beside the embedding, and therefore into every batch
    /// child that clones these inputs, because the person who handed mold a
    /// group photograph is the one who needs to know which face it picked. The
    /// worker copies it onto `GenerateResponse.request_warnings` at
    /// completion, which is what puts it on `x-mold-request-warning` for the
    /// JSON path and in the SSE complete event for the streaming one.
    pub identity_warning: Option<String>,
    /// The batch-lifetime authority for "one identity per parent". See
    /// [`IdentityPin`]; empty for every request that conditions on no face.
    pub identity_pin: IdentityPin,
    /// Payload-free authenticated authority for the private H3 ingress. This
    /// is only a transport seam; per-device admission evidence is attached by
    /// dependency preparation before generic execution planning may consume it.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) h3_private_ingress_grant: Option<crate::h3_private_bridge::H3PrivateIngressGrant>,
    /// One immutable inference-derived admission record per eligible device.
    /// These DTOs contain identities/capacities only; opened artifacts and
    /// executable authority remain inside the inference owner boundary.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) h3_private_admission_by_device:
        BTreeMap<String, mold_inference::H3PrivateFl2VaAdmissionEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedDeviceExecutionInputs {
    pub engine_paths: ModelPaths,
    pub engine_config: mold_inference::FrozenEngineConfig,
    /// Preview-only identities for known dependencies that admission will
    /// materialize at these exact paths. Production admission always leaves
    /// this empty and fingerprints the landed artifacts instead.
    pub pending_artifacts: BTreeMap<PathBuf, PendingArtifactIdentity>,
    /// Free VRAM used to resolve an `auto` dependency choice.
    pub prepared_available_vram_bytes: u64,
    /// False for explicit variants, whose choice must not churn with
    /// telemetry.
    pub capacity_sensitive: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingArtifactIdentity {
    pub kind: String,
    pub repo: String,
    pub filename: String,
    pub bytes: u64,
    pub install_model: Option<String>,
    pub licenses: Vec<mold_core::LicenseRefusal>,
    /// Registry-declared container of the artifact admission will land here.
    /// The preview must not claim GGUF for a `.safetensors`, `.pt`, or `.onnx`
    /// dependency it has never read.
    pub container: PendingArtifactContainer,
    /// `None` for an artifact the registry declares unquantized.
    pub quantization: Option<QuantizationVariant>,
}

impl PendingArtifactIdentity {
    pub fn as_download(&self) -> mold_core::PendingModelDownload {
        mold_core::PendingModelDownload {
            kind: self.kind.clone(),
            name: self.filename.clone(),
            repo: self.repo.clone(),
            bytes: self.bytes,
            install_model: self.install_model.clone(),
            licenses: self.licenses.clone(),
        }
    }

    fn exact_fingerprint(&self) -> ContentFingerprint {
        let mut hash = Sha256::new();
        hash.update(b"mold.pending-preview.exact.v1\0");
        hash.update(self.repo.as_bytes());
        hash.update(b"\0");
        hash.update(self.filename.as_bytes());
        hash.update(b"\0");
        hash.update(self.bytes.to_le_bytes());
        ContentFingerprint(format!("{:x}", hash.finalize()))
    }

    fn equivalence_identity(&self) -> EquivalenceContentIdentity {
        EquivalenceContentIdentity::PendingPreview {
            repo: self.repo.clone(),
            filename: self.filename.clone(),
            bytes: self.bytes,
        }
    }
}

pub(crate) const ENCODER_DEPENDENCY_HEADROOM_BYTES: u64 = 2_000_000_000;

impl PreparedExecutionInputs {
    pub(crate) fn pending_downloads_for_device(
        &self,
        device_id: &str,
    ) -> Vec<mold_core::PendingModelDownload> {
        let Some(device) = self.by_device.get(device_id) else {
            return Vec::new();
        };
        let mut downloads = device
            .pending_artifacts
            .values()
            .map(PendingArtifactIdentity::as_download)
            .collect::<Vec<_>>();
        downloads.sort_by(|left, right| {
            (
                left.kind.as_str(),
                left.repo.as_str(),
                left.name.as_str(),
                left.bytes,
            )
                .cmp(&(
                    right.kind.as_str(),
                    right.repo.as_str(),
                    right.name.as_str(),
                    right.bytes,
                ))
        });
        downloads.dedup();
        downloads
    }
}

fn prepared_config_overlay(
    config: &Config,
    request: &GenerateRequest,
    prepared: Option<&PreparedExecutionInputs>,
) -> Option<Config> {
    let model_config = prepared?.model_config_overlay.as_ref()?;
    if config.models.contains_key(&request.model) {
        return None;
    }
    let mut effective = config.clone();
    effective
        .models
        .insert(request.model.clone(), model_config.as_ref().clone());
    Some(effective)
}

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ExecutionPlanError {
    #[error("model activation rejected execution planning: {0}")]
    ModelActivation(String),
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
    #[error("no device has enough effective VRAM capacity for a safe execution plan: {reason}")]
    InsufficientVram {
        /// Per-device explanation, so a rejection names which device fell
        /// short by how much — and, for LTX-2, a shape that would fit.
        reason: String,
        /// Smallest predicted peak across the rejected devices. The scheduler
        /// compares this against device *total* VRAM: a peak no device could
        /// ever hold is terminal, anything else is transient pressure.
        required_peak_bytes: u64,
        /// Stable IDs of the devices that were actually considered for this
        /// request. Physical-impossibility classification must not borrow
        /// capacity from a sibling excluded by placement or preparation.
        eligible_device_ids: Vec<String>,
    },
    #[error("execution plan was invalidated before CUDA work: {0}")]
    PlanInvalidated(String),
    #[error("prepared execution inputs are stale: {0}")]
    PreparedInputsStale(String),
    #[error(
        "camera-motion preset '{alias}' does not resolve to an installed adapter for this model"
    )]
    UnresolvableLora { alias: String },
    #[error(
        "engine-shaping environment variable '{name}' has no execution-equivalence \
         classification; add a RuntimeSemanticVariable arm in mold-server's \
         execution_plan.rs (build defect, not a user error)"
    )]
    UnclassifiedRuntimeVariable { name: String },
}

pub fn capabilities_for_family(family: &str) -> PlacementCapabilities {
    // This is a server-facing projection of the inference factory's one
    // authoritative, weight-free family registry. Unknown catalog families
    // fail closed instead of gaining scheduler capabilities by name matching.
    if let Some(capability) = mold_inference::production_family_capability_for_family(family) {
        PlacementCapabilities {
            supports_text_encoder_cpu: capability.placement.text_encoder_cpu,
            supports_vae_cpu: capability.placement.vae_cpu,
            supports_audio_components_cpu: capability.placement.audio_components_cpu,
            supports_block_offload: capability.block_offload,
            supports_tiled_vae: capability.tiled_vae
                != mold_inference::TiledVaeCapability::Unsupported,
            batch_execution: Some(capability.execution),
        }
    } else {
        PlacementCapabilities {
            supports_text_encoder_cpu: false,
            supports_vae_cpu: false,
            supports_audio_components_cpu: false,
            supports_block_offload: false,
            supports_tiled_vae: false,
            batch_execution: None,
        }
    }
}

fn require_execution_plan_activation(model: &str, family: &str) -> Result<(), ExecutionPlanError> {
    if mold_core::minimax_h3::is_family(family)
        || mold_core::minimax_h3::capability_contract_for_model(model).is_some()
    {
        return Err(ExecutionPlanError::ModelActivation(
            crate::h3_admission::reject_normal_h3_admission(model, family).to_string(),
        ));
    }
    mold_core::require_model_activation(model, Some(family))
        .map_err(|error| ExecutionPlanError::ModelActivation(error.to_string()))
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
    let family = config
        .resolved_model_config(&request.model)
        .family
        .or_else(|| {
            mold_core::manifest::find_manifest(&request.model)
                .map(|manifest| manifest.family.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());
    require_execution_plan_activation(&request.model, &family)?;
    let paths = ModelPaths::resolve(&request.model, config).ok_or_else(|| {
        ExecutionPlanError::MissingArtifacts {
            model: request.model.clone(),
        }
    })?;
    let capabilities = capabilities_for_family(&family);
    let engine_config = mold_inference::FrozenEngineConfig::resolve(&request.model, config);
    if let Some(alias) = unresolvable_camera_control_alias(config, request) {
        return Err(ExecutionPlanError::UnresolvableLora { alias });
    }
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

/// Placement-only counterpart for an already authenticated private H3
/// ingress grant. It deliberately does not call model/artifact activation;
/// inference preflight owns that authority. The canonical private runtime
/// keeps Qwen on the host and requires transformer/VAE execution on one CUDA or Metal
/// owner, while preserving the generic explicit-device conflict semantics.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn eligible_devices_for_private_h3(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
) -> Result<Vec<DeviceFact>, ExecutionPlanError> {
    let normalized = config.effective_placement(&request.model, request.placement.as_ref());
    let artifacts = BTreeMap::from([
        (ComponentRole::Transformer, PathBuf::new()),
        (ComponentRole::Vae, PathBuf::new()),
        (ComponentRole::QwenShard(0), PathBuf::new()),
    ]);
    let effective = effective_constraints(&normalized, &artifacts);
    for role in [ComponentRole::Transformer, ComponentRole::Vae] {
        if effective.components.get(&role) == Some(&ResolvedComponentConstraint::Cpu) {
            return Err(ExecutionPlanError::UnsupportedCpuPlacement {
                family: mold_core::minimax_h3::FAMILY.to_string(),
                role,
            });
        }
    }
    if matches!(
        effective.components.get(&ComponentRole::QwenShard(0)),
        Some(ResolvedComponentConstraint::Device(_))
    ) {
        return Err(ExecutionPlanError::PlanInvalidated(
            "MiniMax H3 private Qwen placement is fixed to host CPU".into(),
        ));
    }
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
        None,
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
    resolve_execution_plans_for_coordinator_with_projection(
        config,
        request,
        devices,
        offload_requested,
        prepared,
        None,
    )
}

pub(crate) fn resolve_execution_plans_for_coordinator_with_projection(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    offload_requested: bool,
    prepared: Option<&PreparedExecutionInputs>,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    resolve_execution_plans_with_policy(
        config,
        request,
        devices,
        offload_requested,
        prepared,
        projection,
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
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
    fact_policy: EquivalenceFactPolicy,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if let Some(prepared) = prepared.filter(|prepared| prepared.h3_private_ingress_grant.is_some())
    {
        return resolve_private_h3_execution_plans(config, request, devices, prepared, projection);
    }
    let overlaid_config = prepared_config_overlay(config, request, prepared);
    let config = overlaid_config.as_ref().unwrap_or(config);
    let family = config
        .resolved_model_config(&request.model)
        .family
        .or_else(|| {
            mold_core::manifest::find_manifest(&request.model)
                .map(|manifest| manifest.family.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());
    require_execution_plan_activation(&request.model, &family)?;
    let paths = ModelPaths::resolve(&request.model, config).ok_or_else(|| {
        ExecutionPlanError::MissingArtifacts {
            model: request.model.clone(),
        }
    })?;
    let capabilities = capabilities_for_family(&family);
    if offload_requested && !capabilities.supports_block_offload {
        return Err(ExecutionPlanError::UnsupportedOffload { family });
    }

    let normalized = config.effective_placement(&request.model, request.placement.as_ref());
    if let Some(alias) = unresolvable_camera_control_alias(config, request) {
        return Err(ExecutionPlanError::UnresolvableLora { alias });
    }
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
    let mut rejections: Vec<DeviceInfeasibility> = Vec::new();
    let candidates = devices
        .iter()
        .filter(|device| hard_device.as_ref().is_none_or(|hard| hard == &device.id))
        .filter_map(|device| {
            let inputs = match prepared {
                Some(prepared) => prepared.by_device.get(&device.id).map(|prepared| {
                    (
                        prepared.engine_paths.clone(),
                        prepared.engine_config.clone(),
                        prepared.pending_artifacts.clone(),
                    )
                })?,
                None => (
                    paths.clone(),
                    admission_engine_config.clone(),
                    BTreeMap::new(),
                ),
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
                projection,
                paths: &inputs.0,
                engine_config: &inputs.1,
                admission_paths: &paths,
                admission_engine_config: &admission_engine_config,
                effective_loras: &effective_loras,
                artifacts: &artifacts,
                pending_artifacts: &inputs.2,
                effective: &effective,
                offload_requested,
                equivalence_cache_only: fact_policy == EquivalenceFactPolicy::CacheOnly,
            };
            build_plan(&context, device, &mut rejections)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if candidates.is_empty() {
        return Err(insufficient_vram_error(&rejections));
    }
    Ok(candidates)
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn resolve_private_h3_execution_plans(
    config: &Config,
    request: &GenerateRequest,
    devices: &[DeviceFact],
    prepared: &PreparedExecutionInputs,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> Result<Vec<ResolvedExecutionPlan>, ExecutionPlanError> {
    // The scheduler resolves the payload-free durable row; what the queue
    // media store holds for it is the projection. Dropping it here read every
    // FL2VA first-frame job as T2AV and refused it at resolve (#1423).
    let media = mold_core::minimax_h3::ResolvedMediaPresence {
        source_image: request.source_image.is_some()
            || projection.is_some_and(|projection| projection.source_image),
    };
    let grant = prepared.h3_private_ingress_grant.as_ref().ok_or_else(|| {
        ExecutionPlanError::PreparedInputsStale(
            "MiniMax H3 private planning lost its authenticated ingress grant".into(),
        )
    })?;
    grant
        .validate_bound_request(request)
        .map_err(ExecutionPlanError::PreparedInputsStale)?;
    if prepared
        .h3_private_admission_by_device
        .keys()
        .collect::<Vec<_>>()
        != prepared.by_device.keys().collect::<Vec<_>>()
    {
        return Err(ExecutionPlanError::PreparedInputsStale(
            "MiniMax H3 prepared device routes and admission evidence diverged".into(),
        ));
    }
    let eligible = eligible_devices_for_private_h3(config, request, devices)?;
    let host = crate::h3_admission::current_h3_host_memory();
    let available_host_headroom_bytes = host.headroom_bytes();
    let mut plans = Vec::new();
    let mut rejections = Vec::new();
    let mut host_shortfall: Option<String> = None;
    for device in eligible {
        let Some(evidence) = prepared.h3_private_admission_by_device.get(&device.id) else {
            continue;
        };
        let available_device_bytes = if device.backend == GpuBackend::Metal {
            mold_inference::device::metal_unified_capacity_with_safety_floor(
                device.available_vram_bytes,
            )
        } else {
            device.available_vram_bytes
        };
        // Ask the host-memory question directly, before the frozen evidence is
        // revalidated. The live headroom recheck is one of `validate_for`'s
        // conjuncts and its refusal is a single opaque sentence, so the
        // host-vs-device distinction used to be recovered by sniffing that
        // sentence for the substring "host" — which it never contains. Every
        // H3 job blocked on host RAM was therefore filed as a VRAM shortfall
        // it did not have and reported to the planner as zero candidates,
        // which is the untyped `no_schedulable_device` of #1272.
        let required_host_bytes = evidence.predicted_host_increment_bytes();
        if available_host_headroom_bytes < required_host_bytes {
            host_shortfall.get_or_insert_with(|| {
                h3_host_headroom_shortfall_reason(
                    &device.id,
                    required_host_bytes,
                    available_host_headroom_bytes,
                    host.reclaimable_zfs_arc_bytes,
                )
            });
            continue;
        }
        if let Err(error) = evidence.validate_for(
            request,
            media,
            &device.id,
            device.ordinal,
            device.compute_capability,
            available_device_bytes,
            available_host_headroom_bytes,
        ) {
            // The planner reports this as a VRAM block with only the two byte
            // counts; the sentence that says WHICH conjunct refused must reach
            // the log or the block is undiagnosable (#1423: a fitting plan
            // sat blocked for four hours with "required < headroom").
            tracing::warn!(
                target: "mold_server::execution_plan",
                device_id = %device.id,
                model = %request.model,
                error = %format!("{error:#}"),
                "private H3 admission evidence refused at resolve"
            );
            rejections.push(DeviceInfeasibility {
                device_id: device.id,
                predicted_peak_bytes: evidence.predicted_device_peak_bytes(),
                available_bytes: available_device_bytes,
                advice: Some(format!(
                    "private admission evidence no longer fits: {error:#}"
                )),
            });
            continue;
        }
        let inputs = prepared.by_device.get(&device.id).ok_or_else(|| {
            ExecutionPlanError::PreparedInputsStale(format!(
                "MiniMax H3 device '{}' lost its prepared engine inputs",
                device.id
            ))
        })?;
        let mut expected_engine_config =
            mold_inference::FrozenEngineConfig::resolve(&request.model, config);
        expected_engine_config.family = mold_core::minimax_h3::FAMILY.to_string();
        expected_engine_config.h3_factory_authority =
            Some(evidence.base_factory_authority().clone());
        expected_engine_config.attention_backend = evidence.attention().generic_backend;
        expected_engine_config.attention_chunk = evidence.attention().generic_chunk;
        if inputs.engine_config != expected_engine_config {
            return Err(ExecutionPlanError::PreparedInputsStale(format!(
                "MiniMax H3 engine semantics changed after admission for '{}'",
                device.id
            )));
        }
        let attention_backend = match inputs.engine_config.attention_backend {
            mold_inference::attention::AttentionBackend::Math => AttentionBackend::Math,
            mold_inference::attention::AttentionBackend::Flash => AttentionBackend::Flash,
        };
        let offload_mode = if evidence.base_factory_authority().block_offload() {
            OffloadMode::Block
        } else {
            OffloadMode::None
        };
        let determinism_class = DeterminismClass::CpuSeededCrossBackend;
        let components = BTreeMap::new();
        let effective_placement = EffectivePlacement {
            components: BTreeMap::new(),
        };
        let execution_environment = execution_environment_descriptor(
            &device,
            &request.model,
            mold_core::minimax_h3::FAMILY,
            evidence.component_set_identity_sha256(),
            &components,
            &[],
            &inputs.engine_config,
            attention_backend,
            mold_inference::LoadStrategy::Sequential,
            offload_mode,
            request.resolved_output_format(),
            determinism_class,
            true,
            &BTreeMap::new(),
        )?;
        let execution_equivalence_fingerprint = execution_environment.fingerprint();
        plans.push(ResolvedExecutionPlan {
            device_id: device.id,
            device_ordinal: device.ordinal,
            device_backend: device.backend,
            model_family: mold_core::minimax_h3::FAMILY.to_string(),
            model_fingerprint: evidence.component_set_identity_sha256().to_string(),
            effective_placement,
            components,
            engine_paths: inputs.engine_paths.clone(),
            engine_config: inputs.engine_config.clone(),
            admission_paths: inputs.engine_paths.clone(),
            admission_engine_config: inputs.engine_config.clone(),
            effective_loras: Vec::new(),
            attention_backend,
            engine_load_strategy: mold_inference::LoadStrategy::Sequential,
            offload_mode,
            predicted_vram_peak_bytes: evidence.predicted_device_peak_bytes(),
            admitted_available_vram_bytes: evidence.admitted_available_device_bytes(),
            learned_vram_envelope_bytes: 0,
            predicted_host_increment_bytes: evidence.predicted_host_increment_bytes(),
            // An H3 attempt is one-shot owner work with no warm engine to
            // credit, so its warm figure is its cold one.
            predicted_warm_host_increment_bytes: evidence.predicted_host_increment_bytes(),
            determinism_class,
            execution_environment,
            execution_equivalence_fingerprint,
            execution_fingerprint: evidence.execution_fingerprint().to_string(),
        });
    }
    if plans.is_empty() {
        if let Some(reason) = host_shortfall {
            // The frozen evidence was minted against a host sample that no
            // longer holds, so only a fresh admission can decide. Staleness is
            // what re-runs it; its own refusal then names required vs sampled.
            return Err(ExecutionPlanError::PreparedInputsStale(reason));
        }
        return Err(insufficient_vram_error(&rejections));
    }
    Ok(plans)
}

/// Name both numbers a host-headroom shortfall turns on.
///
/// A bare "capacity changed" sentence is indistinguishable from every other
/// staleness, and the whole point of #1218's honest refusals is that a user can
/// read the shortfall and act on it.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
///
/// `reclaimable_zfs_arc_bytes` is the evictable ZFS ARC the SAME sample
/// counted into `available_host_headroom_bytes` (#1439); a positive credit is
/// named so the figure already includes everything the kernel would drain.
pub(crate) fn h3_host_headroom_shortfall_reason(
    device_id: &str,
    required_host_bytes: u64,
    available_host_headroom_bytes: u64,
    reclaimable_zfs_arc_bytes: Option<u64>,
) -> String {
    let clause = match reclaimable_zfs_arc_bytes {
        Some(credit) if credit > 0 => {
            format!(" (the sample includes {credit} bytes of evictable ZFS ARC)")
        }
        _ => String::new(),
    };
    format!(
        "MiniMax H3 host-memory capacity changed after private admission: {device_id} needs \
         {required_host_bytes} host bytes but only {available_host_headroom_bytes} are available{clause}"
    )
}

/// LTX-2 rejections name a shape that does fit on this device, so the user is
/// not left to guess which of resolution or frame count to reduce (#641).
fn ltx2_shape_advice(context: &PlanContext<'_>, device: &DeviceFact) -> Option<String> {
    if context.family != "ltx2" {
        return None;
    }
    let facts = crate::ltx2_admission::checkpoint_facts_cached(&context.paths.transformer)?;
    crate::ltx2_admission::supported_shape_advice(
        &facts,
        crate::ltx2_admission::Ltx2ShapeHint::from_request_with_projection(
            context.request,
            context.projection,
        ),
        device.available_vram_bytes,
    )
}

/// Why one device could not host this request. Collected per candidate so the
/// rejection names the shortfall instead of the bare
/// "no device has enough effective VRAM capacity".
#[derive(Clone, Debug)]
pub(crate) struct DeviceInfeasibility {
    pub(crate) device_id: String,
    pub(crate) predicted_peak_bytes: u64,
    pub(crate) available_bytes: u64,
    /// Family-specific remediation, e.g. an LTX-2 shape that does fit.
    pub(crate) advice: Option<String>,
}

pub(crate) fn insufficient_vram_error(rejections: &[DeviceInfeasibility]) -> ExecutionPlanError {
    if rejections.is_empty() {
        return ExecutionPlanError::InsufficientVram {
            reason: "no request-eligible device produced a concrete execution plan".to_string(),
            required_peak_bytes: 0,
            eligible_device_ids: Vec::new(),
        };
    }
    let reason = rejections
        .iter()
        .map(|rejection| {
            let advice = rejection
                .advice
                .as_ref()
                .map(|advice| format!(" ({advice})"))
                .unwrap_or_default();
            format!(
                "{} needs ~{:.1} GB but only ~{:.1} GB is currently available for this request{advice}",
                rejection.device_id,
                rejection.predicted_peak_bytes as f64 / 1_000_000_000.0,
                rejection.available_bytes as f64 / 1_000_000_000.0,
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    ExecutionPlanError::InsufficientVram {
        reason,
        required_peak_bytes: rejections
            .iter()
            .map(|rejection| rejection.predicted_peak_bytes)
            .min()
            .unwrap_or(0),
        eligible_device_ids: rejections
            .iter()
            .map(|rejection| rejection.device_id.clone())
            .collect(),
    }
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
    // Bind ordered reference content and probed shape without ever feeding
    // inline bytes, request-scoped upload handles, or server-local paths into
    // the serialized preparation authority. The content digest deliberately
    // makes equivalent authorities converge while vector order remains part
    // of the frozen identity.
    let ordered_references = normalized_request.references.as_ref().map(|references| {
        references
            .iter()
            .enumerate()
            .map(|(index, reference)| reference.redacted_metadata(index))
            .collect::<Vec<_>>()
    });
    normalized_request.references = None;

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
    hash.update(b"\0ordered-generation-references-v1\0");
    hash.update(
        serde_json::to_vec(&ordered_references)
            .expect("redacted reference serialization is infallible")
            .as_slice(),
    );
    let family = config.resolved_model_config(&request.model).family;
    if let Some(authority) =
        crate::h3_admission::preparation_authority_bytes(request, family.as_deref())
    {
        // This marker lands before runtime activation. It prevents prepared
        // dependencies from surviving a change to H3 row accounting,
        // one-device policy, Qwen truncation, or the host-memory floor even
        // though runtime admission still rejects unqualified H3 execution.
        hash.update(b"\0minimax-h3-admission-authority-v1\0");
        hash.update(authority);
    }
    format!("{:x}", hash.finalize())
}

/// Populate metadata-bound byte-identity and format-fact caches during
/// asynchronous dependency preparation.
///
/// Execution-plan resolution runs in the scheduler coordinator and must only
/// perform metadata checks plus cache lookups for normal prepared jobs. The
/// caller is responsible for invoking this function through
/// `tokio::task::spawn_blocking`, because unverified artifacts still require
/// blocking content hashing and every artifact requires header probing.
/// Verified downloads reuse their attested digest marker instead of rereading
/// multi-gigabyte checkpoint bodies.
pub(crate) fn warm_execution_equivalence_cache(
    config: &Config,
    request: &GenerateRequest,
    prepared: &PreparedExecutionInputs,
    preparation_progress: Option<&crate::variant_dependencies::PreparationProgressSink>,
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
    let total_bytes = paths
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok().map(|metadata| metadata.len()))
        .fold(0_u64, u64::saturating_add);
    let mut completed_bytes = 0_u64;
    crate::variant_dependencies::publish_preparation_progress(
        preparation_progress,
        "Verifying model files",
        completed_bytes,
        total_bytes,
    );
    for path in paths {
        let artifact_bytes = std::fs::metadata(&path)
            .ok()
            .map(|metadata| metadata.len())
            .unwrap_or_default();
        let mut progress = |done: u64, _total: u64| {
            crate::variant_dependencies::publish_preparation_progress(
                preparation_progress,
                "Verifying model files",
                completed_bytes.saturating_add(done.min(artifact_bytes)),
                total_bytes,
            );
            Ok(())
        };
        let _ = artifact_facts_path_with_policy_and_progress(&path, false, Some(&mut progress));
        completed_bytes = completed_bytes.saturating_add(artifact_bytes);
        crate::variant_dependencies::publish_preparation_progress(
            preparation_progress,
            "Verifying model files",
            completed_bytes,
            total_bytes,
        );
    }
    // LTX-2 admission needs the checkpoint's per-block weight layout. Reading
    // the safetensors header is blocking work, so it is warmed here (already
    // on the blocking pool) and only ever read from cache by the coordinator.
    if family == "ltx2" {
        for inputs in prepared.by_device.values() {
            crate::ltx2_admission::warm_checkpoint_facts(&inputs.engine_paths.transformer);
        }
    }
    // Wan's admission model reads its token grid and per-token slope from the
    // checkpoint header, which is the same blocking work for the same reason.
    if family == "wan" {
        for inputs in prepared.by_device.values() {
            crate::wan_admission::warm_checkpoint_geometry(&inputs.engine_paths);
        }
    }
}

pub fn validate_before_cuda(
    plan: &ResolvedExecutionPlan,
    worker_device_id: &str,
    worker_ordinal: usize,
    config: &Config,
    request: &GenerateRequest,
    prepared: Option<&PreparedExecutionInputs>,
) -> Result<(), ExecutionPlanError> {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if let Some(prepared) = prepared.filter(|prepared| prepared.h3_private_ingress_grant.is_some())
    {
        return validate_private_h3_before_cuda(
            plan,
            worker_device_id,
            worker_ordinal,
            config,
            request,
            prepared,
        );
    }
    require_execution_plan_activation(&request.model, &plan.model_family)?;
    if plan.device_id != worker_device_id || plan.device_ordinal != worker_ordinal {
        return Err(ExecutionPlanError::PlanInvalidated(format!(
            "lease targets {worker_device_id}/gpu:{worker_ordinal}, plan targets {}/gpu:{}",
            plan.device_id, plan.device_ordinal
        )));
    }
    let overlaid_config = prepared_config_overlay(config, request, prepared);
    let config = overlaid_config.as_ref().unwrap_or(config);
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

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn validate_private_h3_before_cuda(
    plan: &ResolvedExecutionPlan,
    worker_device_id: &str,
    worker_ordinal: usize,
    config: &Config,
    request: &GenerateRequest,
    prepared: &PreparedExecutionInputs,
) -> Result<(), ExecutionPlanError> {
    if plan.device_id != worker_device_id || plan.device_ordinal != worker_ordinal {
        return Err(ExecutionPlanError::PlanInvalidated(format!(
            "lease targets {worker_device_id}/gpu:{worker_ordinal}, plan targets {}/gpu:{}",
            plan.device_id, plan.device_ordinal
        )));
    }
    let grant = prepared.h3_private_ingress_grant.as_ref().ok_or_else(|| {
        ExecutionPlanError::PlanInvalidated(
            "MiniMax H3 pre-CUDA validation lost its ingress grant".into(),
        )
    })?;
    grant
        .validate_bound_request(request)
        .map_err(ExecutionPlanError::PlanInvalidated)?;
    let evidence = prepared
        .h3_private_admission_by_device
        .get(worker_device_id)
        .ok_or_else(|| {
            ExecutionPlanError::PlanInvalidated(format!(
                "MiniMax H3 pre-CUDA validation has no evidence for '{worker_device_id}'"
            ))
        })?;
    // This coordinator/acceptance fence proves immutable request and plan
    // identity only. The final owner fence revalidates the evidence against
    // the worker's actual compute capability and an authoritative post-drop
    // device/host capacity sample immediately before private preparation;
    // the run path repeats the exact-peak physical check before allocation.
    evidence
        .validate_resolved_request(request)
        .map_err(|error| {
            ExecutionPlanError::PlanInvalidated(format!(
                "MiniMax H3 request changed after private admission: {error:#}"
            ))
        })?;
    let inputs = prepared.by_device.get(worker_device_id).ok_or_else(|| {
        ExecutionPlanError::PlanInvalidated(format!(
            "MiniMax H3 pre-CUDA validation has no prepared route for '{worker_device_id}'"
        ))
    })?;
    let mut expected_engine_config =
        mold_inference::FrozenEngineConfig::resolve(&request.model, config);
    expected_engine_config.family = mold_core::minimax_h3::FAMILY.to_string();
    expected_engine_config.h3_factory_authority = Some(evidence.base_factory_authority().clone());
    expected_engine_config.attention_backend = evidence.attention().generic_backend;
    expected_engine_config.attention_chunk = evidence.attention().generic_chunk;
    let exact = plan.model_family == mold_core::minimax_h3::FAMILY
        && plan.model_fingerprint == evidence.component_set_identity_sha256()
        && plan.components.is_empty()
        && plan.engine_paths == inputs.engine_paths
        && plan.engine_config == inputs.engine_config
        && plan.engine_config == expected_engine_config
        && plan.admission_paths == inputs.engine_paths
        && plan.admission_engine_config == inputs.engine_config
        && plan.effective_loras.is_empty()
        && plan.execution_fingerprint == evidence.execution_fingerprint()
        && plan.predicted_vram_peak_bytes == evidence.predicted_device_peak_bytes()
        && plan.admitted_available_vram_bytes == evidence.admitted_available_device_bytes()
        && plan.predicted_host_increment_bytes == evidence.predicted_host_increment_bytes()
        && plan.engine_config.h3_factory_authority.as_ref()
            == Some(evidence.base_factory_authority());
    if !exact {
        return Err(ExecutionPlanError::PlanInvalidated(
            "MiniMax H3 plan changed from immutable private admission evidence".into(),
        ));
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
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if plan.engine_config.h3_factory_authority.is_some()
        && mold_core::minimax_h3::is_family(&plan.model_family)
    {
        return;
    }
    request.placement = Some(materialized_placement(plan));
    let loras = plan
        .effective_loras
        .iter()
        .map(|lora| mold_core::LoraWeight {
            path: lora.path.to_string_lossy().into_owned(),
            scale: lora.scale(),

            expert: None,
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
        artifacts.insert(ComponentRole::TransformerShard(index), shard.clone());
    }
    // A two-expert pair reads its second half only after the schedule crosses
    // the expert boundary. Without its own role the plan would freeze one
    // expert and validate one expert, and a file replaced or deleted between
    // admission and the swap would change the render — or fail it — with the
    // plan still reporting valid.
    if let Some(path) = &paths.low_noise_transformer {
        artifacts.insert(ComponentRole::LowNoiseTransformer, path.clone());
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
        // the exact selected Gemma weight files — but never the weights of the
        // variant that was not selected. Chaining the whole list made a Q4
        // selection plan the GGUF *and* the five BF16 shards, so the quantized
        // fallback raised predicted host memory instead of lowering it.
        let selected = engine_config
            .selected_gemma_paths
            .iter()
            .collect::<BTreeSet<_>>();
        engine_config
            .selected_gemma_paths
            .iter()
            .cloned()
            .chain(
                paths
                    .text_encoder_files
                    .iter()
                    .filter(|candidate| {
                        !selected.contains(*candidate) && !is_gemma_weight_file(candidate)
                    })
                    .cloned(),
            )
            .collect()
    } else if let Some(path) = engine_config.selected_umt5_path.as_ref() {
        // Wan's selected UMT5 GGUF fully replaces the manifest's FP16 shard:
        // the tokenizer lives in `text_tokenizer`, not in this list.
        vec![path.clone()]
    } else {
        paths.text_encoder_files.clone()
    };
    for (index, path) in selected_text_paths.iter().enumerate() {
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
    if matches!(family, "ltx2" | "ltx-2" | "ltx2.3") {
        if let Some(split) = mold_core::ltx25_manifest::Ltx25ModelPaths::resolve_for_transformer_in(
            &engine_config.artifact_root,
            &paths.transformer,
        ) {
            artifacts.insert(ComponentRole::AudioVae, split.audio_vae);
            artifacts.insert(ComponentRole::DurationHead, split.duration_head);
        }
    }
    if let Some(path) = &paths.decoder {
        artifacts.insert(ComponentRole::Decoder, path.clone());
    }
    if let Some(path) = &paths.distilled_lora {
        artifacts.insert(ComponentRole::DistilledLora, path.clone());
    }
    if let Some(path) = &paths.low_noise_distilled_lora {
        artifacts.insert(ComponentRole::LowNoiseDistilledLora, path.clone());
    }
    for (index, lora) in effective_loras.iter().enumerate() {
        artifacts.insert(ComponentRole::Lora(index), lora.path.clone());
    }
    // Identity conditioning is frozen, not requested: `identity_assets` is
    // populated by dependency preparation only for a request that asks for a
    // face with a non-zero effective weight, so a plain render and an
    // `id_weight` 0 render carry no identity components at all.
    if let Some(identity) = &engine_config.identity_assets {
        artifacts.insert(ComponentRole::IdentityAdapter, identity.adapter.clone());
        artifacts.insert(
            ComponentRole::IdentityVisionEncoder,
            identity.vision_encoder_source.clone(),
        );
        artifacts.insert(ComponentRole::FaceDetector, identity.face_detector.clone());
        artifacts.insert(
            ComponentRole::FaceRecognizer,
            identity.face_recognizer.clone(),
        );
        artifacts.insert(
            ComponentRole::FaceParser,
            identity.face_parser_source.clone(),
        );
    }
    artifacts
}

/// The first `camera-control:<id>` alias in the request that does not resolve
/// to an installed adapter, if any.
///
/// `effective_loras` deliberately leaves such an alias in place as its own
/// "path" so plan fingerprints stay stable and self-consistent. That is fine
/// for hashing and cache comparison, but it must never reach a device plan:
/// a relative path named `camera-control:dolly-in` fingerprints as "missing"
/// and then agrees with itself everywhere, so the render proceeds with the
/// preset silently absent. Admission calls this first and refuses instead.
fn unresolvable_camera_control_alias(config: &Config, request: &GenerateRequest) -> Option<String> {
    effective_lora_requests(config, request)
        .into_iter()
        .find_map(|lora| {
            let id = lora.path.strip_prefix("camera-control:")?;
            resolved_camera_control_path(config, id)
                .is_none()
                .then(|| lora.path.clone())
        })
}

/// Resolve a `camera-control:<id>` alias to the adapter's on-disk path.
fn resolved_camera_control_path(config: &Config, id: &str) -> Option<PathBuf> {
    let preset = mold_core::ltx2_camera::resolve_camera_control_preset(id).ok()?;
    let manifest = mold_core::manifest::find_manifest(preset.download_model)?;
    let file = manifest.files.first()?;
    Some(
        config
            .resolved_models_dir()
            .join(mold_core::manifest::storage_path(manifest, file)),
    )
}

/// The LoRA stack a request actually asks for, before alias resolution:
/// explicit `loras`, else the legacy single `lora`, else the model config's
/// own default.
fn effective_lora_requests(
    config: &Config,
    request: &GenerateRequest,
) -> Vec<mold_core::LoraWeight> {
    request
        .loras
        .as_ref()
        .filter(|stack| !stack.is_empty())
        .cloned()
        .or_else(|| request.lora.clone().map(|lora| vec![lora]))
        .or_else(|| {
            config
                .resolved_model_config(&request.model)
                .effective_lora()
                .map(|(path, scale)| mold_core::LoraWeight {
                    path,
                    scale,
                    expert: None,
                })
                .map(|lora| vec![lora])
        })
        .unwrap_or_default()
}

fn effective_loras(config: &Config, request: &GenerateRequest) -> Vec<PlannedLora> {
    const ZERO_SCALE_EPS: f64 = 1e-8;
    effective_lora_requests(config, request)
        .into_iter()
        .filter(|lora| lora.scale.abs() > ZERO_SCALE_EPS)
        .map(|lora| {
            let path = lora
                .path
                .strip_prefix("camera-control:")
                .and_then(|id| resolved_camera_control_path(config, id))
                .unwrap_or_else(|| PathBuf::from(lora.path));
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
            ComponentRole::Transformer
            | ComponentRole::TransformerShard(_)
            | ComponentRole::LowNoiseTransformer => advanced.map(|value| &value.transformer),
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
            ComponentRole::Transformer
            | ComponentRole::TransformerShard(_)
            | ComponentRole::LowNoiseTransformer => {
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
    projection: Option<&'a crate::queue_media_store::QueueMediaProjection>,
    paths: &'a ModelPaths,
    engine_config: &'a mold_inference::FrozenEngineConfig,
    admission_paths: &'a ModelPaths,
    admission_engine_config: &'a mold_inference::FrozenEngineConfig,
    effective_loras: &'a [PlannedLora],
    artifacts: &'a BTreeMap<ComponentRole, PathBuf>,
    pending_artifacts: &'a BTreeMap<PathBuf, PendingArtifactIdentity>,
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
    rejections: &mut Vec<DeviceInfeasibility>,
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
    let wan_block_offload_policy = mold_inference::wan::block_offload::AdmissionPolicy::from_values(
        device.backend,
        context
            .engine_config
            .runtime_environment
            .value("MOLD_WAN_OFFLOAD_BLOCKS"),
        context
            .engine_config
            .runtime_environment
            .value("MOLD_OFFLOAD"),
    );
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
    // A CUDA OOM on this exact (model, shape, GPU) leaves a reduced grant
    // behind. Planning the retry against it is what makes the retry
    // conservative instead of an identical repeat of the failing plan (#641).
    let device_budget = crate::gpu_pool::reduced_vram_grant(
        context.model,
        &crate::gpu_pool::oom_shape_bucket_with_projection(context.request, context.projection),
        device.ordinal,
    )
    .map_or(device.available_vram_bytes, |grant| {
        grant.min(device.available_vram_bytes)
    });
    let recent_oom_reduced_budget = device_budget < device.available_vram_bytes;
    let initial_memory =
        crate::memory_preflight::estimate_generation_memory_for_request_with_projection(
            context.request,
            context.paths,
            hint,
            crate::memory_preflight::GenerationOffloadPolicy::new(
                context.offload_requested,
                wan_block_offload_policy,
            ),
            Some(device_budget),
            request_has_lora,
            gemma_competes,
            context.projection,
        );
    // A process-wide offload preference is advisory for concrete formats
    // which cannot honor it (for example Flux.2 GGUF/NVFP4 or a LoRA merge).
    // The family capability gate above remains a typed error; this path-level
    // policy exactly matches what engine construction will consume.
    let pending_encoder_bytes = context
        .artifacts
        .iter()
        .filter(|(role, path)| {
            role.is_text_encoder() && context.pending_artifacts.contains_key(*path)
        })
        .map(|(_, path)| context.pending_artifacts[path].bytes)
        .sum::<u64>();
    let pending_pushes_eager_over_budget = initial_memory
        .eager_peak_memory_bytes
        .saturating_add(pending_encoder_bytes)
        > device.available_vram_bytes.saturating_mul(9) / 10;
    let auto_cpu_text = context.capabilities.supports_text_encoder_cpu
        && (initial_memory.under_memory_pressure || pending_pushes_eager_over_budget);

    let mut placements = context
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
                || (auto_cpu_text
                    && constraint == ResolvedComponentConstraint::Auto
                    && role.is_text_encoder());
            (role.clone(), cpu)
        })
        .collect::<BTreeMap<_, _>>();
    let transformer_on_cpu = placements
        .iter()
        .filter(|(role, _)| {
            matches!(
                role,
                ComponentRole::Transformer
                    | ComponentRole::TransformerShard(_)
                    | ComponentRole::LowNoiseTransformer
            )
        })
        .all(|(_, cpu)| *cpu);
    let gpu_paths = gpu_resident_paths(context.paths, &placements);
    let mut memory =
        crate::memory_preflight::estimate_generation_memory_for_request_with_projection(
            context.request,
            &gpu_paths,
            hint,
            crate::memory_preflight::GenerationOffloadPolicy::new(
                initial_memory.block_offload && !transformer_on_cpu,
                wan_block_offload_policy,
            ),
            Some(device_budget),
            request_has_lora,
            gemma_competes,
            context.projection,
        );
    if memory.fits_available_memory != Some(true)
        && context.capabilities.supports_vae_cpu
        && context
            .effective
            .components
            .get(&ComponentRole::Vae)
            .is_none_or(|constraint| constraint == &ResolvedComponentConstraint::Auto)
        && placements.contains_key(&ComponentRole::Vae)
    {
        placements.insert(ComponentRole::Vae, true);
        let gpu_paths = gpu_resident_paths(context.paths, &placements);
        memory = crate::memory_preflight::estimate_generation_memory_for_request_with_projection(
            context.request,
            &gpu_paths,
            hint,
            crate::memory_preflight::GenerationOffloadPolicy::new(
                initial_memory.block_offload && !transformer_on_cpu,
                wan_block_offload_policy,
            ),
            Some(device_budget),
            request_has_lora,
            gemma_competes,
            context.projection,
        );
    }
    if memory.fits_available_memory != Some(true) {
        let mut advice = ltx2_shape_advice(context, device);
        if recent_oom_reduced_budget {
            let cooldown = "this request is temporarily limited after a recent CUDA OOM; retry after the cooldown or reduce the output size".to_string();
            advice = Some(match advice {
                Some(existing) => format!("{existing}; {cooldown}"),
                None => cooldown,
            });
        }
        rejections.push(DeviceInfeasibility {
            device_id: device.id.clone(),
            predicted_peak_bytes: memory.peak_memory_bytes,
            available_bytes: device_budget,
            advice,
        });
        return None;
    }
    // Missing preview dependencies have no filesystem metadata yet. Mirror
    // the resolved component placement: a pending encoder assigned to CPU
    // consumes host memory and does not need to fit the GPU; a device-assigned
    // encoder must fit with the same preparation headroom. Admission
    // recomputes the complete plan from landed file metadata.
    let pending_dependency_peak = context
        .artifacts
        .iter()
        .filter(|(role, path)| {
            context.pending_artifacts.contains_key(*path)
                && !placements.get(*role).copied().unwrap_or(false)
        })
        .map(|(_, path)| {
            let artifact = &context.pending_artifacts[path];
            artifact
                .bytes
                .saturating_add(ENCODER_DEPENDENCY_HEADROOM_BYTES)
        })
        .max()
        .unwrap_or(0);
    if pending_dependency_peak > device.available_vram_bytes {
        rejections.push(DeviceInfeasibility {
            device_id: device.id.clone(),
            predicted_peak_bytes: pending_dependency_peak,
            available_bytes: device.available_vram_bytes,
            advice: None,
        });
        return None;
    }

    let mut components = BTreeMap::new();
    let mut host_bytes_by_path: BTreeMap<PathBuf, u64> = BTreeMap::new();
    // The subset of `host_bytes_by_path` a request allocates again on a warm
    // hit: a streaming encoder's forward-loop heap, and every host-only
    // component — the identity extraction stack is built and released per
    // extraction, a LoRA is merged per request — none of which the resident
    // engine holds. A parked encoder's bytes stay resident in the engine and
    // are already absent from `MemAvailable`.
    let mut recurring_host_bytes_by_path: BTreeMap<PathBuf, u64> = BTreeMap::new();
    let gemma_anon_peak_anchor =
        ltx2_cpu_gemma_anon_peak_anchor(context.family, context.artifacts, &placements);
    for (role, path) in context.artifacts {
        let place_cpu = placements.get(role).copied().unwrap_or(false);
        let bytes = context
            .pending_artifacts
            .get(path)
            .map_or_else(|| artifact_size(path), |artifact| artifact.bytes);
        let (placement, load_strategy, vram, host) = if place_cpu {
            // A streaming CPU encoder never materializes its weights: the
            // shards stay a memory-mapped `VarBuilder` and each decoder layer
            // is built and dropped inside the forward loop. Those pages are
            // file-backed and reclaimable — the kernel evicts them under
            // pressure and `MemAvailable`, this ledger's own input, already
            // counts them as available — so reserving them as anonymous demand
            // asks for ~24 GB of free anonymous room to hold memory that can
            // never cause an OOM. On a host whose headroom lands just under
            // that sum it refuses every job while the GPU idles (#1108).
            // Only the encoder's real anonymous heap is irreclaimable, and
            // only the anchor shard carries it.
            let streams_from_mmap = ltx2_cpu_gemma_streams_from_mmap(context.family, role, path);
            let host = if streams_from_mmap {
                if gemma_anon_peak_anchor.as_ref() == Some(role) {
                    mold_inference::ltx2::cpu_gemma_streaming_anon_peak_bytes()
                } else {
                    0
                }
            } else {
                bytes
            };
            host_bytes_by_path.insert(path.clone(), host);
            if streams_from_mmap || role.is_host_only() {
                recurring_host_bytes_by_path.insert(path.clone(), host);
            }
            (
                ResolvedComponentPlacement::Cpu,
                ComponentLoadStrategy::ParkedCpu,
                0,
                host,
            )
        } else {
            let strategy = if memory.block_offload
                && matches!(
                    role,
                    ComponentRole::Transformer
                        | ComponentRole::TransformerShard(_)
                        | ComponentRole::LowNoiseTransformer
                ) {
                // Most streamed backends retain an anonymous host copy at the
                // artifact's stored precision. LTX-2 safetensors is the
                // exception: its ordinary and ConvRot loaders retain only a
                // reclaimable mmap and materialize one bounded block/weight.
                if !ltx2_transformer_streams_from_mmap(context.family, role, path) {
                    host_bytes_by_path.insert(path.clone(), bytes);
                }
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
                content_fingerprint: context
                    .pending_artifacts
                    .get(path)
                    .map(PendingArtifactIdentity::exact_fingerprint)
                    .unwrap_or_else(|| fingerprint_path(path)),
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

    let predicted_vram = memory.peak_memory_bytes.max(pending_dependency_peak);
    let predicted_host = host_bytes_by_path
        .values()
        .fold(BASE_HOST_TRANSIENT, |total, bytes| {
            total.saturating_add(*bytes)
        });
    let predicted_warm_host = recurring_host_bytes_by_path
        .values()
        .fold(BASE_HOST_TRANSIENT, |total, bytes| {
            total.saturating_add(*bytes)
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
    let model_fingerprint =
        model_fingerprint(context.model, context.artifacts, context.pending_artifacts);
    let equivalence_model_fingerprint = equivalence_model_fingerprint(
        context.artifacts,
        context.pending_artifacts,
        context.equivalence_cache_only,
    );
    let attention_backend = match context.engine_config.attention_backend {
        mold_inference::attention::AttentionBackend::Math => AttentionBackend::Math,
        mold_inference::attention::AttentionBackend::Flash => AttentionBackend::Flash,
    };
    // `wan_block_offload` joins the disposition here and nowhere else. The
    // component load strategy and the host-RAM reservation above stay on
    // `block_offload` alone, because wan parks a subset of one expert's blocks
    // rather than streaming a transformer from host RAM — charging it as
    // `StreamedBlocks` would reserve every A14B expert file at full size
    // (#776).
    let offload_mode = if memory.block_offload || memory.wan_block_offload {
        OffloadMode::Block
    } else {
        OffloadMode::None
    };
    let determinism_class = DeterminismClass::CpuSeededCrossBackend;
    let execution_environment = match execution_environment_descriptor(
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
        context.pending_artifacts,
    ) {
        Ok(environment) => environment,
        Err(error) => return Some(Err(error)),
    };
    let execution_equivalence_fingerprint = execution_environment.fingerprint();
    Some(Ok(ResolvedExecutionPlan {
        device_id: device.id.clone(),
        device_ordinal: device.ordinal,
        device_backend: device.backend,
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
        learned_vram_envelope_bytes: 0,
        predicted_host_increment_bytes: predicted_host,
        predicted_warm_host_increment_bytes: predicted_warm_host,
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
    pending_artifacts: &BTreeMap<PathBuf, PendingArtifactIdentity>,
) -> Result<ExecutionEnvironmentDescriptor, ExecutionPlanError> {
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
            let pending = pending_artifacts.get(&component.artifact_path);
            let content_fingerprint = pending
                .map(PendingArtifactIdentity::equivalence_identity)
                .unwrap_or_else(|| facts.content.clone());
            let storage = pending.map_or_else(
                || component_storage_format(&facts),
                |pending| ComponentStorageFormat::PendingPreview {
                    container: pending.container,
                    artifact_identity: pending.equivalence_identity(),
                    quantization: pending.quantization,
                },
            );
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
    Ok(ExecutionEnvironmentDescriptor {
        schema_version: 3,
        backend: device.backend,
        architecture,
        attention_kernel_class: match attention_backend {
            AttentionBackend::Math => AttentionKernelClass::Math,
            AttentionBackend::Flash => AttentionKernelClass::Flash,
        },
        code: execution_code_identity(),
        semantic_config: ExecutionSemanticConfig::from_frozen(engine_config)?,
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
    })
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
    if on_cpu(&ComponentRole::LowNoiseTransformer) {
        gpu.low_noise_transformer = None;
    }
    if !gpu.transformer_shards.is_empty() {
        gpu.transformer_shards = gpu
            .transformer_shards
            .into_iter()
            .enumerate()
            .filter_map(|(index, path)| {
                (!on_cpu(&ComponentRole::TransformerShard(index))).then_some(path)
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

/// The one CPU-placed Gemma shard that carries the streaming encoder's
/// anonymous-heap peak, if this plan has one.
///
/// A CPU-parked component is normally reserved at its file length, because
/// most engines map or copy those weights at their stored precision. LTX-2's
/// prompt encoder additionally allocates a bounded working set on top of its
/// mmap — see `mold_inference::ltx2::cpu_gemma_streaming_anon_peak_bytes` — and
/// that set belongs to the encoder rather than to any one shard, so it is
/// attributed to the lowest-ordered shard. Added per shard it would be charged
/// five times over on a real Gemma split.
/// Whether a Gemma text-encoder file holds *weights* rather than a companion
/// artifact.
///
/// Mirrors the patterns `variant_dependencies::materialize_gemma` selects on:
/// BF16 arrives as `model.safetensors` or `model-<i>-of-<n>.safetensors`
/// shards, Q4 as a single `.gguf`. Everything else in `text_encoder_files` —
/// the tokenizer anchor, the LTX-2.3 text projection — belongs to every variant
/// and must survive the filter. Every entry is given a `GemmaShard` role, so
/// the role alone cannot tell a streamed weight shard from a companion the
/// runtime materializes.
fn is_gemma_weight_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name.ends_with(".gguf")
        || name == "model.safetensors"
        || (name.starts_with("model-") && name.ends_with(".safetensors"))
        || (name.starts_with("gemma4-") && name.ends_with(".safetensors"))
}

/// Whether a CPU-placed component's weights stay a reclaimable file mapping
/// rather than becoming anonymous host demand.
///
/// True only for LTX-2's safetensors Gemma **weight shards**, which
/// `GemmaHiddenStateEncoder::load_from_assets` builds through `new_streaming`.
/// The Q4 GGUF variant keeps its own quantized residency in host RAM, and every
/// other family's CPU-parked encoder is copied to the host, so both are charged
/// their bytes. LTX-2.3's ~2.3 GB text projection shares the `GemmaShard` role
/// but is loaded separately into the retained `EmbeddingsProcessor` rather than
/// through the streaming layer loader, so it is real anonymous demand and must
/// keep being charged — exempting it would over-admit a host near its floor.
fn ltx2_cpu_gemma_streams_from_mmap(family: &str, role: &ComponentRole, path: &Path) -> bool {
    matches!(family, "ltx2" | "ltx-2" | "ltx2.3")
        && matches!(role, ComponentRole::GemmaShard(_))
        && is_gemma_weight_file(path)
        && mold_inference::ltx2::cpu_gemma_allocates_anon_peak(path)
}

/// Whether LTX-2 block streaming keeps the checkpoint out of anonymous host
/// memory, materializing one tensor at a time.
///
/// The ordinary safetensors and ConvRot backends retain a reclaimable mmap;
/// the GGUF backend seeks one tensor at a time through a buffered reader
/// (`Ltx2GgufBackend`), so its transient is one raw tensor payload (≤ 67 MB
/// at Q8_0). Charging the complete transformer again as concurrent host
/// residency is especially wrong on Metal, where that charge is folded back
/// into the same unified-memory gate. The real transient for every format is
/// bounded by `BASE_HOST_TRANSIENT`.
fn ltx2_transformer_streams_from_mmap(family: &str, role: &ComponentRole, path: &Path) -> bool {
    matches!(family, "ltx2" | "ltx-2" | "ltx2.3")
        && matches!(
            role,
            ComponentRole::Transformer
                | ComponentRole::TransformerShard(_)
                | ComponentRole::LowNoiseTransformer
        )
        && path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("safetensors") || extension.eq_ignore_ascii_case("gguf")
        })
}

fn ltx2_cpu_gemma_anon_peak_anchor(
    family: &str,
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
    placements: &BTreeMap<ComponentRole, bool>,
) -> Option<ComponentRole> {
    if !matches!(family, "ltx2" | "ltx-2" | "ltx2.3") {
        return None;
    }
    artifacts
        .iter()
        .find(|(role, path)| {
            matches!(role, ComponentRole::GemmaShard(_))
                && placements.get(*role).copied().unwrap_or(false)
                && is_gemma_weight_file(path)
                && mold_inference::ltx2::cpu_gemma_allocates_anon_peak(path)
        })
        .map(|(role, _)| role.clone())
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
        ComponentStorageFormat::PendingPreview { quantization, .. } => *quantization,
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

/// Replace an artifact's bytes so its metadata identity is guaranteed to
/// change, for tests.
///
/// `std::fs::write` truncates in place, keeping the inode, so a same-size
/// rewrite can only be distinguished by `ctime` — which advances on the
/// kernel's coarse clock (~98 ms granularity on ext4/tmpfs here). Two writes in
/// one tick therefore produce an identical identity and any fixture demanding a
/// change is a coin flip. Writing a sibling and renaming over the target
/// allocates a new inode, so the identity differs on the *inode* rather than on
/// a timestamp — deterministic no matter how fast the calls land. This is also
/// the shape a real re-download takes.
///
/// `std::fs::rename` replaces an existing destination on both Unix and Windows
/// (`MoveFileExW` / `SetFileInformationByHandle` with replace-if-exists), so no
/// remove-then-rename fallback is needed — and adding one would open a window
/// where the artifact is absent.
#[cfg(test)]
pub(crate) fn replace_artifact_bytes(path: &Path, contents: &[u8]) {
    let staging = path.with_extension(format!(
        "{}.replacing",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("tmp")
    ));
    std::fs::write(&staging, contents).expect("write replacement artifact");
    std::fs::rename(&staging, path).expect("rename replacement artifact into place");
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
            FileBasicInfo, GetFileInformationByHandle, GetFileInformationByHandleEx,
            BY_HANDLE_FILE_INFORMATION, FILE_BASIC_INFO,
        };

        // `Metadata::volume_serial_number` / `file_index` are the obvious
        // spelling and are NIGHTLY-ONLY (`windows_by_handle`, rust#63010), so
        // read the same two fields from `GetFileInformationByHandle`, which is
        // stable — the identical route `batch_transaction::windows_file_identity`
        // already takes. Both readings come off ONE handle: opening the path
        // twice would let a replacement between the two calls pair a file id
        // with another file's ChangeTime, which is exactly the substitution
        // this identity exists to catch.
        let (volume_serial, file_index, change_time) = std::fs::File::open(_path)
            .ok()
            .map(|file| {
                let handle = file.as_raw_handle() as _;
                let mut by_handle = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::uninit();
                // SAFETY: `file` owns a valid handle for the duration of the
                // call and `by_handle` is correctly sized writable storage.
                let identity =
                    unsafe { GetFileInformationByHandle(handle, by_handle.as_mut_ptr()) };
                let (volume_serial, file_index) = if identity != 0 {
                    // SAFETY: a successful call initialized every field.
                    let info = unsafe { by_handle.assume_init() };
                    (
                        info.dwVolumeSerialNumber,
                        (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
                    )
                } else {
                    (0, 0)
                };

                let mut basic = std::mem::MaybeUninit::<FILE_BASIC_INFO>::uninit();
                // SAFETY: same handle, and `basic` is correctly sized/aligned
                // for FileBasicInfo.
                let result = unsafe {
                    GetFileInformationByHandleEx(
                        handle,
                        FileBasicInfo,
                        basic.as_mut_ptr().cast(),
                        std::mem::size_of::<FILE_BASIC_INFO>() as u32,
                    )
                };
                // SAFETY: Win32 initializes the entire FILE_BASIC_INFO on a
                // nonzero result.
                let change_time = (result != 0)
                    .then(|| unsafe { basic.assume_init().ChangeTime as u64 })
                    .unwrap_or(0);

                (volume_serial, file_index, change_time)
            })
            .unwrap_or((0, 0, 0));

        ArtifactMetadataIdentity {
            len: metadata.file_size(),
            // File ID catches replacement; NTFS ChangeTime catches an
            // in-place overwrite even if last-write time and size are
            // restored by the caller.
            platform_identity: vec![
                u64::from(volume_serial),
                file_index,
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
    artifact_facts_path_with_policy_and_progress(path, cache_only, None)
}

fn artifact_facts_path_with_policy_and_progress(
    path: &Path,
    cache_only: bool,
    mut progress: Option<&mut dyn FnMut(u64, u64) -> anyhow::Result<()>>,
) -> ArtifactFacts {
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
        let content = hash_equivalence_artifact_contents_with_progress(path, |done, total| {
            progress
                .as_mut()
                .map_or(Ok(()), |callback| callback(done, total))
        })
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

#[cfg(test)]
fn hash_equivalence_artifact_contents(path: &Path) -> std::io::Result<EquivalenceContentIdentity> {
    hash_equivalence_artifact_contents_with_progress(path, |_, _| Ok(()))
        .map_err(std::io::Error::other)
}

fn hash_equivalence_artifact_contents_with_progress(
    path: &Path,
    progress: impl FnMut(u64, u64) -> anyhow::Result<()>,
) -> anyhow::Result<EquivalenceContentIdentity> {
    // Use the retained-descriptor verifier shared with downloads. It refuses
    // symlinks and path replacement, single-flights concurrent callers, and
    // persists an owner-private identity-bound attestation so an unchanged
    // legacy artifact is read only once across process restarts.
    mold_core::download::pinned_file_digest_with_progress(path, progress)
        .map(EquivalenceContentIdentity::Sha256)
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

fn model_fingerprint(
    model: &str,
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
    pending_artifacts: &BTreeMap<PathBuf, PendingArtifactIdentity>,
) -> String {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    for (role, path) in artifacts {
        hash.update(format!("{role:?}").as_bytes());
        let fingerprint = pending_artifacts
            .get(path)
            .map(PendingArtifactIdentity::exact_fingerprint)
            .unwrap_or_else(|| fingerprint_path(path));
        hash.update(fingerprint.0.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn equivalence_model_fingerprint(
    artifacts: &BTreeMap<ComponentRole, PathBuf>,
    pending_artifacts: &BTreeMap<PathBuf, PendingArtifactIdentity>,
    cache_only: bool,
) -> String {
    let mut hash = Sha256::new();
    for (role, path) in artifacts {
        hash.update(format!("{role:?}").as_bytes());
        pending_artifacts
            .get(path)
            .map(PendingArtifactIdentity::equivalence_identity)
            .unwrap_or_else(|| equivalence_fingerprint_path_with_policy(path, cache_only))
            .update_hash(&mut hash);
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
    Ok(resolved_paths_model_fingerprint(model, paths))
}

fn resolved_paths_model_fingerprint(model: &str, paths: ModelPaths) -> String {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(ComponentRole::Transformer, paths.transformer);
    if let Some(path) = paths.low_noise_transformer {
        artifacts.insert(ComponentRole::LowNoiseTransformer, path);
    }
    for (index, shard) in paths.transformer_shards.into_iter().enumerate() {
        artifacts.insert(ComponentRole::TransformerShard(index), shard);
    }
    if !paths.vae.as_os_str().is_empty() {
        artifacts.insert(ComponentRole::Vae, paths.vae);
    }
    model_fingerprint(model, &artifacts, &BTreeMap::new())
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
    let mut files = vec![("transformer".to_string(), paths.transformer)];
    if !paths.vae.as_os_str().is_empty() {
        files.push(("vae".to_string(), paths.vae));
    }
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
    optional_file!("low_noise_transformer", paths.low_noise_transformer);
    optional_file!("low_noise_distilled_lora", paths.low_noise_distilled_lora);
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
    freeze_chain_model_with_paths(config, model, paths)
}

/// Freeze a model from the exact path resolution used at request admission.
///
/// Callers resolving opaque installed-catalog IDs must not repeat resolution
/// from the base config: the effective overlay and these paths are one
/// immutable authority snapshot.
pub fn freeze_chain_model_with_paths(
    config: &Config,
    model: &str,
    paths: ModelPaths,
) -> Result<mold_core::chain_job::FrozenChainModel, ExecutionPlanError> {
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
    frozen.low_noise_transformer = canonical_optional(paths.low_noise_transformer.as_ref())?;
    frozen.vae = (!paths.vae.as_os_str().is_empty())
        .then(|| canonical(&paths.vae))
        .transpose()?;
    frozen.spatial_upscaler = canonical_optional(paths.spatial_upscaler.as_ref())?;
    frozen.temporal_upscaler = canonical_optional(paths.temporal_upscaler.as_ref())?;
    frozen.distilled_lora = canonical_optional(paths.distilled_lora.as_ref())?;
    frozen.low_noise_distilled_lora = canonical_optional(paths.low_noise_distilled_lora.as_ref())?;
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
    hash.update(format!("{:?}", ExecutionFingerprintEngineConfig(engine_config)).as_bytes());
    hash.update(format!("{effective_loras:?}").as_bytes());
    hash.update([u8::from(offload)]);
    format!("{:x}", hash.finalize())
}

/// Stable exact-fingerprint view of the engine configuration.
///
/// `artifact_root` is a storage trust boundary, not an execution input. The
/// concrete component paths are already part of the exact fingerprint, so
/// moving the same artifacts under a different configured root must not alter
/// execution identity. Keep the historical `FrozenEngineConfig` debug shape
/// for the remaining fields because this fingerprint is a persisted contract.
struct ExecutionFingerprintEngineConfig<'a>(&'a mold_inference::FrozenEngineConfig);

impl std::fmt::Debug for ExecutionFingerprintEngineConfig<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mold_inference::FrozenEngineConfig {
            family,
            artifact_root: _,
            is_schnell,
            is_turbo,
            scheduler,
            t5_variant,
            qwen3_variant,
            qwen2_variant,
            qwen2_text_encoder_mode,
            ltx2_gemma_variant,
            umt5_variant,
            selected_t5_path,
            selected_qwen3_paths,
            selected_qwen2_path,
            selected_gemma_paths,
            selected_umt5_path,
            identity_assets,
            h3_factory_authority,
            runtime_environment,
            attention_backend,
            attention_chunk,
            vae_tiling,
            vae_dtype,
        } = self.0;

        let mut debug = formatter.debug_struct("FrozenEngineConfig");
        debug
            .field("family", family)
            .field("is_schnell", is_schnell)
            .field("is_turbo", is_turbo)
            .field("scheduler", scheduler)
            .field("t5_variant", t5_variant)
            .field("qwen3_variant", qwen3_variant)
            .field("qwen2_variant", qwen2_variant)
            .field("qwen2_text_encoder_mode", qwen2_text_encoder_mode)
            .field("ltx2_gemma_variant", ltx2_gemma_variant)
            .field("selected_t5_path", selected_t5_path)
            .field("selected_qwen3_paths", selected_qwen3_paths)
            .field("selected_qwen2_path", selected_qwen2_path)
            .field("selected_gemma_paths", selected_gemma_paths);
        // Both UMT5 fields are emitted only when present, so every fingerprint
        // that predates the Wan quantized encoder keeps its exact bytes.
        if let Some(variant) = umt5_variant {
            debug.field("umt5_variant", variant);
        }
        if let Some(path) = selected_umt5_path {
            debug.field("selected_umt5_path", path);
        }
        // Emitted only when present, so every fingerprint that predates
        // identity conditioning keeps its exact bytes.
        if let Some(identity) = identity_assets {
            debug.field("identity_assets", identity);
        }
        if let Some(authority) = h3_factory_authority {
            debug.field("h3_factory_authority", &authority.identity_sha256());
        }
        debug
            .field("runtime_environment", runtime_environment)
            .field("attention_backend", attention_backend)
            .field("attention_chunk", attention_chunk)
            .field("vae_tiling", vae_tiling)
            .field("vae_dtype", vae_dtype)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LTX-2 transformer streaming holds no anonymous whole-artifact host
    /// copy for either container: safetensors backends stream from an mmap
    /// and the GGUF backend seeks one tensor at a time, so both charge only
    /// the base transient.
    #[test]
    fn ltx2_transformer_streaming_charges_no_anon_copy_for_safetensors_or_gguf() {
        let role = ComponentRole::Transformer;
        assert!(ltx2_transformer_streams_from_mmap(
            "ltx2",
            &role,
            Path::new("/models/ltx-2.5/transformer.safetensors"),
        ));
        assert!(ltx2_transformer_streams_from_mmap(
            "ltx2",
            &role,
            Path::new("/models/ltx-2.5/LTX-2.5-Distilled-Q4_K_M.gguf"),
        ));
        assert!(!ltx2_transformer_streams_from_mmap(
            "ltx2",
            &role,
            Path::new("/models/ltx-2.5/transformer.bin"),
        ));
        assert!(!ltx2_transformer_streams_from_mmap(
            "wan",
            &role,
            Path::new("/models/wan/transformer.gguf"),
        ));
    }
    use mold_core::{
        AdvancedPlacement, GenerationReference, GenerationReferenceAuthority,
        GenerationReferenceProvenance, ModelConfig,
    };
    use tempfile::TempDir;

    /// The H3 host shortfall names the evictable ZFS ARC its sample already
    /// counted (#1439), and stays byte-identical for every other host.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    #[test]
    fn h3_host_headroom_shortfall_reason_names_the_evictable_arc() {
        let plain =
            h3_host_headroom_shortfall_reason("cuda:0", 32_775_178_178, 26_200_000_000, None);
        assert_eq!(
            plain,
            "MiniMax H3 host-memory capacity changed after private admission: cuda:0 needs \
             32775178178 host bytes but only 26200000000 are available"
        );
        assert_eq!(
            h3_host_headroom_shortfall_reason("cuda:0", 32_775_178_178, 26_200_000_000, Some(0)),
            plain,
            "a cold ARC on a ZFS host reads like any other host"
        );
        assert_eq!(
            h3_host_headroom_shortfall_reason(
                "cuda:0",
                45_000_000_000,
                41_281_432_704,
                Some(15_081_432_704),
            ),
            "MiniMax H3 host-memory capacity changed after private admission: cuda:0 needs \
             45000000000 host bytes but only 41281432704 are available (the sample includes \
             15081432704 bytes of evictable ZFS ARC)"
        );
    }

    #[test]
    fn packed_gemma4_safetensors_are_streamed_weights() {
        assert!(is_gemma_weight_file(Path::new("gemma4-12b-it.safetensors")));
        assert!(!is_gemma_weight_file(Path::new("tokenizer.json")));
    }

    #[test]
    fn production_family_planning_uses_the_static_batch_registry_before_load() {
        for entry in mold_inference::production_batch_capabilities() {
            let expected_tiled = entry.tiled_vae != mold_inference::TiledVaeCapability::Unsupported;
            let projected = capabilities_for_family(entry.family);
            assert_eq!(
                projected,
                PlacementCapabilities {
                    supports_text_encoder_cpu: entry.placement.text_encoder_cpu,
                    supports_vae_cpu: entry.placement.vae_cpu,
                    supports_audio_components_cpu: entry.placement.audio_components_cpu,
                    supports_block_offload: entry.block_offload,
                    supports_tiled_vae: expected_tiled,
                    batch_execution: Some(entry.execution),
                },
                "{}",
                entry.family
            );
            assert_eq!(
                projected.batch_execution,
                Some(entry.execution),
                "{}",
                entry.family
            );
            for alias in entry.aliases {
                assert_eq!(capabilities_for_family(alias), projected, "{alias}");
            }
        }
        assert_eq!(capabilities_for_family("unknown").batch_execution, None);
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    fn sparse_file(path: &Path, bytes: u64) {
        let file = std::fs::File::create(path).unwrap();
        file.set_len(bytes).unwrap();
        drop(file);
        mold_core::download::write_sha256_marker(path, &format!("{bytes:064x}")).unwrap();
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

    #[test]
    fn h3_activation_gate_precedes_artifact_resolution_in_normal_planning() {
        let root = TempDir::new().unwrap();
        let config = config(root.path(), "minimax-h3", None);
        let error = resolve_execution_plans(&config, &request(None), &devices(&[24 * GIB]), false)
            .unwrap_err();

        assert!(matches!(error, ExecutionPlanError::ModelActivation(_)));
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[test]
    fn preparation_authority_binds_reference_order_without_transport_secrets() {
        let root = TempDir::new().unwrap();
        let config = config(root.path(), "minimax-h3", None);
        let paths = ModelPaths::resolve("test:q4", &config).unwrap();
        let engine_config = mold_inference::FrozenEngineConfig::resolve("test:q4", &config);
        let make_inline = |name: &str, data: Vec<u8>| GenerationReference::Image {
            media: GenerationReferenceAuthority::Inline { data },
            provenance: GenerationReferenceProvenance {
                name: Some(name.to_string()),
                sha256: None,
                crop: None,
            },
            mime_type: "image/png".to_string(),
            width: 640,
            height: 480,
        };

        let first = make_inline(" first.png ", vec![1, 2, 3, 4]);
        let second = make_inline("second.png", vec![5, 6, 7, 8]);
        let mut request = request(None);
        request.references = Some(vec![first.clone(), second.clone()]);
        let forward = preparation_authority_fingerprint(&config, &request, &paths, &engine_config);
        request.references = Some(vec![second, first.clone()]);
        let reversed = preparation_authority_fingerprint(&config, &request, &paths, &engine_config);
        assert_ne!(forward, reversed, "reference order is admission authority");

        let digest = first.content_sha256().unwrap().to_ascii_uppercase();
        request.references = Some(vec![GenerationReference::Image {
            media: GenerationReferenceAuthority::ServerPath {
                path: "/private/never-serialize-this.png".to_string(),
            },
            provenance: GenerationReferenceProvenance {
                name: Some("first.png".to_string()),
                sha256: Some(digest),
                crop: None,
            },
            mime_type: "image/png".to_string(),
            width: 640,
            height: 480,
        }]);
        let path_authority =
            preparation_authority_fingerprint(&config, &request, &paths, &engine_config);
        request.references = Some(vec![first]);
        let inline_authority =
            preparation_authority_fingerprint(&config, &request, &paths, &engine_config);
        assert_eq!(
            path_authority, inline_authority,
            "transport authority must converge on the same normalized content identity"
        );

        let projection = request
            .references
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, reference)| reference.redacted_metadata(index))
            .collect::<Vec<_>>();
        let json = serde_json::to_string(&projection).unwrap();
        assert!(!json.contains("private"));
        assert!(!json.contains("authority"));
        assert!(!json.contains("AQIDBA"));
    }

    fn indexed_paths(
        transformer_shards: Vec<PathBuf>,
        text_encoder_files: Vec<PathBuf>,
    ) -> ModelPaths {
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: PathBuf::from("/models/transformer.safetensors"),
            transformer_shards,
            vae: PathBuf::from("/models/vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files,
            text_tokenizer: None,
            decoder: None,
        }
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

    fn metal_devices(free: &[u64]) -> Vec<DeviceFact> {
        free.iter()
            .enumerate()
            .map(|(ordinal, bytes)| DeviceFact {
                id: format!("metal:{ordinal}"),
                ordinal,
                backend: GpuBackend::Metal,
                compute_capability: None,
                available_vram_bytes: *bytes,
            })
            .collect()
    }

    /// A CPU-parked text encoder is dropped before the transformer loads, so
    /// its host charge is phase-disjoint from the device peak (#1038).
    #[test]
    fn a_parked_text_encoder_is_a_phase_disjoint_host_charge() {
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
        assert_eq!(plan.device_backend, GpuBackend::Cuda);
        assert_eq!(plan.predicted_phase_disjoint_host_bytes(), MIB);
        // CUDA admission is unchanged: two pools, raw figures.
        assert_eq!(
            plan.admission_vram_demand_bytes(),
            plan.predicted_vram_peak_bytes
        );
        assert_eq!(
            plan.admission_host_demand_bytes(),
            plan.predicted_host_increment_bytes
        );
    }

    /// On Metal both claims land on one unified pool: the demand is the
    /// larger phase plus the genuinely concurrent host charge, and the host
    /// gate sees zero so the same bytes are not proven twice against a
    /// second sample of the same pool (#1038).
    #[test]
    fn metal_admission_folds_both_claims_onto_the_unified_pool() {
        let root = TempDir::new().unwrap();
        sparse_file(&root.path().join("t5.safetensors"), MIB);
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        };
        let plan = resolve_execution_plans(
            &config(root.path(), "flux2", None),
            &request(Some(placement)),
            &metal_devices(&[24 * GIB]),
            false,
        )
        .unwrap()
        .remove(0);
        assert_eq!(plan.device_backend, GpuBackend::Metal);
        assert_eq!(plan.predicted_phase_disjoint_host_bytes(), MIB);
        // The denoise peak dwarfs the 1 MiB encoder, so the unified demand
        // is the peak plus only the concurrent remainder (the base
        // transient), never peak + encoder.
        assert_eq!(
            plan.admission_vram_demand_bytes(),
            plan.predicted_vram_peak_bytes + BASE_HOST_TRANSIENT
        );
        assert_eq!(plan.admission_host_demand_bytes(), 0);
    }

    /// An encoder phase larger than the denoise peak bounds the unified
    /// demand: max semantics, not a sum — the wan shape from #1038, where a
    /// an 11.7 GB UMT5 charge and a 9.6 GB peak were both proven at once.
    #[test]
    fn metal_unified_demand_is_the_larger_phase_not_the_sum() {
        let root = TempDir::new().unwrap();
        sparse_file(&root.path().join("t5.safetensors"), 12 * GIB);
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        };
        let plan = resolve_execution_plans(
            &config(root.path(), "flux2", None),
            &request(Some(placement)),
            &metal_devices(&[40 * GIB]),
            false,
        )
        .unwrap()
        .remove(0);
        let disjoint = plan.predicted_phase_disjoint_host_bytes();
        assert_eq!(disjoint, 12 * GIB);
        assert!(disjoint > plan.predicted_vram_peak_bytes);
        assert_eq!(
            plan.admission_vram_demand_bytes(),
            disjoint + BASE_HOST_TRANSIENT
        );
    }

    /// A CPU-pinned transformer computes during the peak, so its host bytes
    /// stay a concurrent charge even on Metal — only drop-before-denoise
    /// text encoders are phase-disjoint.
    #[test]
    fn a_cpu_pinned_transformer_stays_a_concurrent_charge_on_metal() {
        let root = TempDir::new().unwrap();
        let (config, mut request) = sized_config(root.path(), "flux2", 32, 1, 2);
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::Cpu,
                ..AdvancedPlacement::default()
            }),
        });
        let plan = resolve_execution_plans(&config, &request, &metal_devices(&[8 * GIB]), false)
            .unwrap()
            .remove(0);
        assert_eq!(
            plan.components[&ComponentRole::Transformer].placement,
            ResolvedComponentPlacement::Cpu
        );
        assert!(
            plan.predicted_phase_disjoint_host_bytes() < 32 * GIB,
            "a CPU-pinned transformer must never count as phase-disjoint"
        );
        assert!(
            plan.admission_vram_demand_bytes() >= plan.predicted_vram_peak_bytes + 32 * GIB,
            "CPU-pinned transformer weights must stay in the unified demand"
        );
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

    /// hal9000, 2026-08-27: `flux-dev:q8` resident on an idle 4090 with its
    /// 9.79 GB T5 parked in host RAM (FLUX drops an encoder only when it is
    /// `on_gpu`), `MemAvailable` 19.9 GB, safety floor 10.07 GB → 9.85 GB of
    /// headroom. The planner charged the warm candidate the cold figure, so two
    /// queued prints of the very model already loaded sat on
    /// `insufficient_host_ram` until the model was unloaded by hand — after
    /// which the cold reload was admitted against 30 GB and paid the load again.
    #[test]
    fn a_warm_hit_charges_only_the_recurring_host_transient() {
        const HAL9000_T5_FP16_BYTES: u64 = 9_787_000_000;
        const HAL9000_HOST_HEADROOM_BYTES: u64 = 9_853_131_776;
        let root = TempDir::new().unwrap();
        sparse_file(&root.path().join("t5.safetensors"), HAL9000_T5_FP16_BYTES);
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
            BASE_HOST_TRANSIENT + HAL9000_T5_FP16_BYTES
        );
        assert_eq!(
            plan.predicted_warm_host_increment_bytes, BASE_HOST_TRANSIENT,
            "a parked encoder is already in the engine's RSS; a warm hit \
             reallocates nothing but the transient"
        );
        assert!(plan.admission_host_demand_bytes() > HAL9000_HOST_HEADROOM_BYTES);
        assert!(plan.admission_warm_host_demand_bytes() <= HAL9000_HOST_HEADROOM_BYTES);
    }

    /// Only what the resident engine HOLDS is credited on a warm hit. A
    /// host-only component — a LoRA merged per request, the identity
    /// extraction stack built and released per extraction — is a per-request
    /// transient whatever the engine's residency, so its bytes stay in the warm
    /// figure beside the base transient.
    #[test]
    fn a_warm_hit_keeps_charging_host_only_transients() {
        let root = TempDir::new().unwrap();
        sparse_file(&root.path().join("t5.safetensors"), 4 * GIB);
        let lora = root.path().join("style.safetensors");
        sparse_file(&lora, 300 * MIB);
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        };
        let mut request = request(Some(placement));
        request.loras = Some(vec![mold_core::LoraWeight {
            path: lora.display().to_string(),
            scale: 1.0,
            expert: None,
        }]);
        let plan = resolve_execution_plans(
            &config(root.path(), "flux2", None),
            &request,
            &devices(&[24 * GIB]),
            false,
        )
        .unwrap()
        .remove(0);

        let lora_component = &plan.components[&ComponentRole::Lora(0)];
        assert_eq!(lora_component.placement, ResolvedComponentPlacement::Cpu);
        assert_eq!(lora_component.predicted_host_bytes, 300 * MIB);
        assert_eq!(
            plan.predicted_host_increment_bytes,
            BASE_HOST_TRANSIENT + 4 * GIB + 300 * MIB
        );
        assert_eq!(
            plan.predicted_warm_host_increment_bytes,
            BASE_HOST_TRANSIENT + 300 * MIB,
            "the parked encoder is credited; the per-request LoRA is not"
        );
    }

    /// A CPU-placed LTX-2 Gemma costs its streaming heap and nothing else.
    ///
    /// The shards stay a memory-mapped `VarBuilder`, so their pages are
    /// file-backed and reclaimable: the kernel can evict them under pressure
    /// and `MemAvailable` — the ledger's own input — already counts them as
    /// available. Reserving them as though they were anonymous demand requires
    /// ~24 GB of free anonymous room for memory that can never cause an OOM,
    /// and on a host whose headroom sits just under that sum it refuses the
    /// job forever (#1108). Only the encoder's real anonymous allocations —
    /// the F32 embedding table, in-flight layers, and hidden states — are
    /// irreclaimable, and those are exactly the streaming heap.
    #[test]
    fn ltx2_cpu_gemma_is_charged_only_its_streaming_heap() {
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("ltx2-transformer.safetensors");
        let vae = root.path().join("ltx2-vae.safetensors");
        let gemma = root.path().join("model-00001-of-00001.safetensors");
        sparse_file(&transformer, 8 * GIB);
        sparse_file(&vae, GIB);
        sparse_file(&gemma, 4 * GIB);
        let mut config = Config::default();
        config.models.insert(
            "ltx2-case:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![gemma.display().to_string()]),
                family: Some("ltx2".to_string()),
                ..ModelConfig::default()
            },
        );
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"ltx2-case:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        });

        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .expect("a CPU-pinned Gemma keeps LTX-2 admissible")
            .remove(0);

        let streaming_heap = mold_inference::ltx2::cpu_gemma_streaming_anon_peak_bytes();
        let encoder = &plan.components[&ComponentRole::GemmaShard(0)];
        assert_eq!(encoder.placement, ResolvedComponentPlacement::Cpu);
        assert_eq!(
            encoder.predicted_host_bytes, streaming_heap,
            "only the streaming encoder's anonymous heap; the shard itself is a \
             reclaimable file mapping"
        );
        assert_eq!(
            plan.predicted_host_increment_bytes,
            BASE_HOST_TRANSIENT + streaming_heap
        );
        assert_eq!(
            plan.predicted_warm_host_increment_bytes,
            BASE_HOST_TRANSIENT + streaming_heap,
            "the streaming heap is a forward-loop allocation and recurs on a warm hit"
        );
    }

    /// LTX-2.3's text projection is materialized, so it keeps paying host RAM.
    ///
    /// Every `text_encoder_files` entry is given a `GemmaShard` role, so the
    /// role cannot distinguish a streamed weight shard from the ~2.3 GB
    /// projection the runtime loads separately into the retained
    /// `EmbeddingsProcessor`. Exempting it along with the mmap'd shards would
    /// charge it zero and over-admit a host sitting near its memory floor.
    #[test]
    fn a_cpu_placed_text_projection_is_still_charged_to_host_ram() {
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("ltx2-transformer.safetensors");
        let vae = root.path().join("ltx2-vae.safetensors");
        let gemma = root.path().join("model-00001-of-00001.safetensors");
        let projection = root.path().join("text_projection.safetensors");
        sparse_file(&transformer, 8 * GIB);
        sparse_file(&vae, GIB);
        sparse_file(&gemma, 4 * GIB);
        sparse_file(&projection, 2 * GIB);
        let mut config = Config::default();
        config.models.insert(
            "ltx2-case:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![
                    gemma.display().to_string(),
                    projection.display().to_string(),
                ]),
                family: Some("ltx2.3".to_string()),
                ..ModelConfig::default()
            },
        );
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"ltx2-case:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        });

        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .expect("a CPU-pinned LTX-2.3 encoder set keeps the model admissible")
            .remove(0);

        let streaming_heap = mold_inference::ltx2::cpu_gemma_streaming_anon_peak_bytes();
        assert_eq!(
            plan.predicted_host_increment_bytes,
            BASE_HOST_TRANSIENT + 2 * GIB + streaming_heap,
            "the projection is anonymous demand; only the weight shard is a mapping"
        );
    }

    /// hal9000 refusing every LTX-2 job on an idle 4090 (#1108).
    ///
    /// Its 24.37 GB Gemma plus the 6.59 GB streaming heap came to 30.96 GB
    /// against 30.78 GB of ledger headroom, so both queued jobs sat blocked on
    /// `insufficient_host_ram` while the GPU was at 0% and nothing would ever
    /// free the difference. Charging the mapping is what made a 180 MB gap
    /// fatal.
    #[test]
    fn a_cpu_gemma_fits_the_host_that_reported_1108() {
        const HAL9000_GEMMA_BYTES: u64 = 24_370_000_000;
        const HAL9000_HOST_HEADROOM_BYTES: u64 = 30_782_477_722;
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("ltx2-transformer.safetensors");
        let vae = root.path().join("ltx2-vae.safetensors");
        let gemma = root.path().join("model-00001-of-00001.safetensors");
        sparse_file(&transformer, 8 * GIB);
        sparse_file(&vae, GIB);
        sparse_file(&gemma, HAL9000_GEMMA_BYTES);
        let mut config = Config::default();
        config.models.insert(
            "ltx2-case:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![gemma.display().to_string()]),
                family: Some("ltx2".to_string()),
                ..ModelConfig::default()
            },
        );
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"ltx2-case:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        });

        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .expect("an idle 4090 must admit this plan")
            .remove(0);

        assert!(
            plan.predicted_host_increment_bytes < HAL9000_HOST_HEADROOM_BYTES,
            "charged {} against {HAL9000_HOST_HEADROOM_BYTES} of headroom — this host \
             would refuse every LTX-2 job with an idle GPU",
            plan.predicted_host_increment_bytes
        );
    }

    /// The streaming heap belongs to the encoder, not to each of its shards.
    ///
    /// Charging it per shard would multiply it by five on a real Gemma split
    /// and reintroduce the over-reservation this model exists to avoid.
    #[test]
    fn a_multi_shard_cpu_gemma_charges_its_streaming_heap_once() {
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("ltx2-transformer.safetensors");
        let vae = root.path().join("ltx2-vae.safetensors");
        sparse_file(&transformer, 8 * GIB);
        sparse_file(&vae, GIB);
        let shards = (1..=5)
            .map(|index| {
                let path = root
                    .path()
                    .join(format!("model-0000{index}-of-00005.safetensors"));
                sparse_file(&path, 4 * GIB);
                path
            })
            .collect::<Vec<_>>();
        let mut config = Config::default();
        config.models.insert(
            "ltx2-case:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(
                    shards
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect(),
                ),
                family: Some("ltx2".to_string()),
                ..ModelConfig::default()
            },
        );
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"ltx2-case:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        });

        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .expect("a CPU-pinned sharded Gemma keeps LTX-2 admissible")
            .remove(0);

        let streaming_heap = mold_inference::ltx2::cpu_gemma_streaming_anon_peak_bytes();
        assert_eq!(
            plan.predicted_host_increment_bytes,
            BASE_HOST_TRANSIENT + streaming_heap,
            "five reclaimable mappings and one streaming heap"
        );
        let carrying_the_heap = (0..5)
            .filter(|index| {
                plan.components[&ComponentRole::GemmaShard(*index)].predicted_host_bytes > 0
            })
            .count();
        assert_eq!(carrying_the_heap, 1, "exactly one shard anchors the heap");
    }

    /// The host that reported #1099 must still admit its own workload.
    ///
    /// 62 GiB of RAM leaves roughly 48.7 GB of ledger headroom after the
    /// canonical floor. A flat dtype widening charged ~49.4 GB for the 24.7 GB
    /// BF16 Gemma and turned a job that ran into permanent
    /// `InsufficientHostRam`.
    #[test]
    fn a_cpu_gemma_still_fits_the_host_that_reported_1099() {
        const REAL_GEMMA_BF16_BYTES: u64 = 24_700_000_000;
        const REPORTED_HOST_HEADROOM_BYTES: u64 = 48_700_000_000;
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("ltx2-transformer.safetensors");
        let vae = root.path().join("ltx2-vae.safetensors");
        let gemma = root.path().join("model-00001-of-00001.safetensors");
        sparse_file(&transformer, 8 * GIB);
        sparse_file(&vae, GIB);
        sparse_file(&gemma, REAL_GEMMA_BF16_BYTES);
        let mut config = Config::default();
        config.models.insert(
            "ltx2-case:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![gemma.display().to_string()]),
                family: Some("ltx2".to_string()),
                ..ModelConfig::default()
            },
        );
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"ltx2-case:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: None,
        });

        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .expect("the reporting host admitted this plan")
            .remove(0);

        assert!(
            plan.predicted_host_increment_bytes
                >= mold_inference::ltx2::cpu_gemma_streaming_anon_peak_bytes(),
            "the streaming heap is real and must be charged"
        );
        assert!(
            plan.predicted_host_increment_bytes < REAL_GEMMA_BF16_BYTES,
            "the shards are a reclaimable file mapping, not anonymous demand"
        );
        assert!(
            plan.predicted_host_increment_bytes < REPORTED_HOST_HEADROOM_BYTES,
            "charged {} GB against {} GB of headroom — this host would park forever",
            plan.predicted_host_increment_bytes / 1_000_000_000,
            REPORTED_HOST_HEADROOM_BYTES / 1_000_000_000,
        );
    }

    /// Widening is a property of the CPU placement, not of the family.
    #[test]
    fn ltx2_gemma_on_the_leased_device_is_not_widened() {
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("ltx2-transformer.safetensors");
        let vae = root.path().join("ltx2-vae.safetensors");
        let gemma = root.path().join("model-00001-of-00001.safetensors");
        sparse_file(&transformer, 8 * GIB);
        sparse_file(&vae, GIB);
        sparse_file(&gemma, 4 * GIB);
        let mut config = Config::default();
        config.models.insert(
            "ltx2-case:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![gemma.display().to_string()]),
                family: Some("ltx2".to_string()),
                ..ModelConfig::default()
            },
        );
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"ltx2-case:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();

        let plan = resolve_execution_plans(&config, &request, &devices(&[48 * GIB]), false)
            .expect("a roomy device keeps every component on the GPU")
            .remove(0);

        let encoder = &plan.components[&ComponentRole::GemmaShard(0)];
        assert_ne!(encoder.placement, ResolvedComponentPlacement::Cpu);
        assert_eq!(encoder.predicted_host_bytes, 0);
        assert_eq!(plan.predicted_host_increment_bytes, BASE_HOST_TRANSIENT);
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
    fn ltx2_safetensors_streaming_does_not_double_charge_its_mmap() {
        let root = TempDir::new().unwrap();
        let (config, request) = sized_config(root.path(), "ltx2", 8, 1, 1);
        let plan = resolve_execution_plans(&config, &request, &metal_devices(&[24 * GIB]), true)
            .expect("LTX-2 safetensors streaming must fit without charging its mmap twice")
            .remove(0);

        assert_eq!(plan.offload_mode, OffloadMode::Block);
        assert_eq!(
            plan.components[&ComponentRole::Transformer].load_strategy,
            ComponentLoadStrategy::StreamedBlocks
        );
        assert_eq!(
            plan.predicted_host_increment_bytes, BASE_HOST_TRANSIENT,
            "the file mapping is reclaimable; only the bounded anonymous transient remains"
        );
        assert_eq!(
            plan.admission_vram_demand_bytes(),
            plan.predicted_vram_peak_bytes + BASE_HOST_TRANSIENT
        );
    }

    #[test]
    fn flux2_dev_shards_reserve_the_full_streamed_transformer_in_host_ram() {
        let root = TempDir::new().unwrap();
        let transformer_shards = (0..7)
            .map(|index| {
                let path = root.path().join(format!("flux2-dev-{index}.safetensors"));
                sparse_file(&path, 9 * GIB);
                path
            })
            .collect::<Vec<_>>();
        let vae = root.path().join("flux2-dev-vae.safetensors");
        sparse_file(&vae, GIB);
        let encoder = root.path().join("flux2-dev-text.safetensors");
        sparse_file(&encoder, GIB);
        let mut config = Config::default();
        config.models.insert(
            "test-flux2-dev:bf16".to_string(),
            ModelConfig {
                transformer: Some(transformer_shards[0].display().to_string()),
                transformer_shards: Some(
                    transformer_shards
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect(),
                ),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![encoder.display().to_string()]),
                family: Some("flux2".to_string()),
                ..ModelConfig::default()
            },
        );
        let request = serde_json::from_str(
            r#"{"prompt":"x","model":"test-flux2-dev:bf16","width":1024,"height":1024,"steps":50,"guidance":4.0}"#,
        )
        .unwrap();

        let resident_plan =
            resolve_execution_plans(&config, &request, &devices(&[96 * GIB]), false)
                .expect("FLUX.2 Dev should stay resident on a 96 GB GPU")
                .remove(0);
        assert_eq!(resident_plan.offload_mode, OffloadMode::None);

        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .expect("FLUX.2 Dev should fit a 24 GB GPU through automatic block offload")
            .remove(0);

        assert_eq!(plan.offload_mode, OffloadMode::Block);
        assert_eq!(
            plan.engine_load_strategy,
            mold_inference::LoadStrategy::Sequential
        );
        assert!(
            plan.predicted_host_increment_bytes >= BASE_HOST_TRANSIENT + 63 * GIB,
            "all seven streamed transformer shards must remain charged to the host ledger"
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
        assert!(matches!(
            resolve_execution_plans(&unsupported, &request(None), &devices(&[3 * GIB]), false,),
            Err(ExecutionPlanError::InsufficientVram { .. })
        ));
    }

    #[test]
    fn auto_cpu_moves_flux2_vae_only_when_encoder_offload_is_insufficient() {
        let root = TempDir::new().unwrap();
        let (config, request) = sized_config(root.path(), "flux2", 8, 5, 2);

        let roomy = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        assert_ne!(
            roomy.components[&ComponentRole::Vae].placement,
            ResolvedComponentPlacement::Cpu,
            "GPU-first placement must keep the VAE resident when capacity is roomy"
        );

        let pressured = resolve_execution_plans(&config, &request, &devices(&[12 * GIB]), false)
            .expect("Flux.2 should park its VAE after text-only CPU placement still cannot fit")
            .remove(0);
        assert_eq!(
            pressured.components[&ComponentRole::Vae].placement,
            ResolvedComponentPlacement::Cpu
        );
        assert_eq!(
            materialized_placement(&pressured)
                .advanced
                .expect("resolved plans always materialize advanced placement")
                .vae,
            DeviceRef::Cpu,
            "the engine must receive the scheduler's concrete VAE placement"
        );
        assert!(
            pressured.predicted_host_increment_bytes >= BASE_HOST_TRANSIENT + 5 * GIB,
            "a parked VAE must be charged to host RAM admission"
        );
        assert_ne!(
            pressured.execution_fingerprint, roomy.execution_fingerprint,
            "placement changes must produce distinct lease identities"
        );
    }

    #[test]
    fn explicit_gpu_vae_pin_prevents_pressure_fallback() {
        let root = TempDir::new().unwrap();
        let (config, mut request) = sized_config(root.path(), "flux2", 8, 5, 2);
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                vae: DeviceRef::device("cuda:0"),
                ..AdvancedPlacement::default()
            }),
        });

        assert!(
            matches!(
                resolve_execution_plans(&config, &request, &devices(&[12 * GIB]), false),
                Err(ExecutionPlanError::InsufficientVram { .. })
            ),
            "automatic CPU fallback must never override an explicit VAE GPU pin"
        );
    }

    #[test]
    fn insufficient_vram_preserves_stable_and_ordinal_pin_eligibility() {
        let root = TempDir::new().unwrap();
        let (config, base_request) = sized_config(root.path(), "unknown-family", 10, 1, 1);
        let candidates = devices(&[8 * GIB, 24 * GIB]);

        assert!(
            resolve_execution_plans(&config, &base_request, &candidates, false).is_ok(),
            "Auto must retain the larger eligible sibling"
        );

        for pin in [DeviceRef::device("cuda:0"), DeviceRef::gpu(0)] {
            let mut request = base_request.clone();
            request.placement = Some(DevicePlacement {
                text_encoders: DeviceRef::Auto,
                advanced: Some(AdvancedPlacement {
                    transformer: pin,
                    ..AdvancedPlacement::default()
                }),
            });
            let error = resolve_execution_plans(&config, &request, &candidates, false)
                .expect_err("the 8 GiB pinned device cannot hold this request");
            assert!(matches!(
                error,
                ExecutionPlanError::InsufficientVram {
                    eligible_device_ids,
                    ..
                } if eligible_device_ids == ["cuda:0"]
            ));
        }
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
        assert!(matches!(
            resolve_execution_plans(&config, &request(None), &[], false),
            Err(ExecutionPlanError::InsufficientVram { .. })
        ));
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
        assert!(matches!(
            resolve_execution_plans(&config, &request, &devices(&[0]), false),
            Err(ExecutionPlanError::InsufficientVram { .. })
        ));
    }

    #[test]
    fn insufficient_vram_message_names_the_actual_request_budget() {
        let error = insufficient_vram_error(&[DeviceInfeasibility {
            device_id: "cuda:0".into(),
            predicted_peak_bytes: 16_600_000_000,
            available_bytes: 15_000_000_000,
            advice: Some("retry after the cooldown".into()),
        }]);

        assert_eq!(
            error.to_string(),
            "no device has enough effective VRAM capacity for a safe execution plan: cuda:0 needs ~16.6 GB but only ~15.0 GB is currently available for this request (retry after the cooldown)"
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
        validate_before_cuda(&plan, "cuda:0", 0, &config, &request, None).unwrap();

        std::fs::write(root.path().join("transformer-q4.gguf"), vec![1_u8; 2048]).unwrap();
        assert!(matches!(
            validate_before_cuda(&plan, "cuda:0", 0, &config, &request, None),
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

                expert: None,
            },
            mold_core::LoraWeight {
                path: root
                    .path()
                    .join("first-lora.safetensors")
                    .display()
                    .to_string(),
                scale: 1.25,

                expert: None,
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
            validate_before_cuda(&request_plan, "cuda:0", 0, &config, &request, None),
            Err(ExecutionPlanError::PlanInvalidated(_))
        ));
    }

    #[test]
    fn camera_control_alias_plans_the_verified_manifest_path() {
        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), name.as_bytes()).unwrap();
        }
        let mut config = config(root.path(), "ltx2", None);
        config.models_dir = root.path().display().to_string();
        let preset = mold_core::ltx2_camera::resolve_camera_control_preset("dolly-in").unwrap();
        let manifest = mold_core::manifest::find_manifest(preset.download_model).unwrap();
        let file = manifest.files.first().unwrap();
        let expected = root
            .path()
            .join(mold_core::manifest::storage_path(manifest, file));
        std::fs::create_dir_all(expected.parent().unwrap()).unwrap();
        std::fs::write(&expected, b"camera").unwrap();

        let mut request = request(None);
        request.loras = Some(vec![mold_core::LoraWeight {
            path: "camera-control:dolly-in".into(),
            scale: 1.0,

            expert: None,
        }]);
        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        assert_eq!(plan.effective_loras[0].path, expected);
        assert!(plan.components.contains_key(&ComponentRole::Lora(0)));
    }

    /// An alias that does not resolve must be refused, never planned as a
    /// literal path. `PathBuf::from("camera-control:no-such-move")` is a
    /// *relative path*: it fingerprints as "missing" and then agrees with
    /// itself through the plan fingerprint, the equivalence cache, and
    /// `validate_before_cuda`, so nothing ever fires and the render silently
    /// proceeds without the preset the user asked for.
    #[test]
    fn unresolvable_camera_control_alias_is_refused_not_planned_as_a_literal_path() {
        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), name.as_bytes()).unwrap();
        }
        let mut config = config(root.path(), "ltx2", None);
        config.models_dir = root.path().display().to_string();

        let mut request = request(None);
        request.loras = Some(vec![mold_core::LoraWeight {
            path: "camera-control:no-such-move".into(),
            scale: 1.0,

            expert: None,
        }]);

        let err = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .expect_err("an unresolvable camera-control alias must not produce a plan");
        assert!(
            matches!(&err, ExecutionPlanError::UnresolvableLora { alias }
                if alias == "camera-control:no-such-move"),
            "expected UnresolvableLora, got {err:?}"
        );

        assert!(
            matches!(
                eligible_devices_for_request(&config, &request, &devices(&[24 * GIB])),
                Err(ExecutionPlanError::UnresolvableLora { .. })
            ),
            "device eligibility must refuse the same alias"
        );
    }

    /// A plain filesystem path is not an alias and must keep passing through
    /// untouched — the refusal above is scoped to `camera-control:` only.
    #[test]
    fn a_plain_lora_path_is_never_treated_as_an_unresolvable_alias() {
        let root = TempDir::new().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), name.as_bytes()).unwrap();
        }
        let mut config = config(root.path(), "ltx2", None);
        config.models_dir = root.path().display().to_string();

        let lora_path = root.path().join("user.safetensors");
        std::fs::write(&lora_path, b"lora").unwrap();
        let mut request = request(None);
        request.loras = Some(vec![mold_core::LoraWeight {
            path: lora_path.display().to_string(),
            scale: 1.0,

            expert: None,
        }]);

        let plan = resolve_execution_plans(&config, &request, &devices(&[24 * GIB]), false)
            .unwrap()
            .remove(0);
        assert_eq!(plan.effective_loras[0].path, lora_path);
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
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
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
    fn ltx25_execution_topology_keeps_split_audio_and_duration_components() {
        let root = TempDir::new().unwrap();
        let split = mold_core::ltx25_manifest::Ltx25ModelPaths::resolve_in(
            root.path(),
            mold_core::ltx25_manifest::DISTILLED_INT8_CONV,
        )
        .unwrap();
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: split.transformer.clone(),
            transformer_shards: Vec::new(),
            vae: split.video_vae.clone(),
            spatial_upscaler: Some(split.spatial_upscaler.clone()),
            temporal_upscaler: Some(split.temporal_upscaler.clone()),
            distilled_lora: split.distilled_lora.clone(),
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![split.gemma.clone()],
            text_tokenizer: None,
            decoder: None,
        };
        let mut frozen = mold_inference::FrozenEngineConfig::resolve(
            mold_core::ltx25_manifest::DISTILLED_INT8_CONV,
            &Config::default(),
        );
        frozen.artifact_root = root.path().to_path_buf();
        let artifacts = concrete_artifacts_for_family(&paths, "ltx2", &[], &frozen);
        assert_eq!(
            artifacts.get(&ComponentRole::AudioVae),
            Some(&split.audio_vae)
        );
        assert_eq!(
            artifacts.get(&ComponentRole::DurationHead),
            Some(&split.duration_head)
        );
    }

    /// Selecting the Q4 Gemma must replace the BF16 shards, not join them.
    ///
    /// `text_encoder_files` carries every Gemma weight variant plus the
    /// tokenizer anchor and the optional LTX-2.3 text projection. Chaining the
    /// whole list onto the selection made a Q4 pick cost the GGUF *and* the
    /// BF16 shards, so `predicted_host_increment_bytes` rose from 24.7 GB to
    /// 32.8 GB — the quantized fallback made memory pressure worse and LTX-2
    /// stayed unadmittable on a 48 GB unified-memory Mac.
    #[test]
    fn a_selected_gemma_variant_replaces_the_other_variants_weights() {
        let root = tempfile::tempdir().unwrap();
        let at = |name: &str| root.path().join(name);
        let bf16 = (1..=5)
            .map(|index| at(&format!("model-0000{index}-of-00005.safetensors")))
            .collect::<Vec<_>>();
        let gguf = at("gemma-3-12b-it-q4_0.gguf");
        let tokenizer = at("tokenizer.model");
        let projection = at("ltx-2.3-text-projection.safetensors");
        for path in bf16
            .iter()
            .chain(std::iter::once(&gguf))
            .chain(std::iter::once(&tokenizer))
            .chain(std::iter::once(&projection))
        {
            std::fs::write(path, b"x").unwrap();
        }

        let mut text_encoder_files = bf16.clone();
        text_encoder_files.push(gguf.clone());
        text_encoder_files.push(tokenizer.clone());
        text_encoder_files.push(projection.clone());

        let transformer = at("transformer.safetensors");
        let vae = at("vae.safetensors");
        std::fs::write(&transformer, b"x").unwrap();
        std::fs::write(&vae, b"x").unwrap();
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: Vec::new(),
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files,
            text_tokenizer: None,
            decoder: None,
        };
        let mut frozen = mold_inference::FrozenEngineConfig::resolve("unused", &Config::default());
        frozen.ltx2_gemma_variant = Some("q4".to_string());
        frozen.selected_gemma_paths = vec![gguf.clone()];

        let artifacts = concrete_artifacts_for_family(&paths, "ltx2", &[], &frozen);
        let planned = artifacts
            .values()
            .collect::<std::collections::BTreeSet<_>>();

        assert!(
            planned.contains(&gguf),
            "selected Q4 weights must be planned"
        );
        assert!(
            planned.contains(&tokenizer) && planned.contains(&projection),
            "tokenizer anchor and text projection are companions, not weights"
        );
        for shard in &bf16 {
            assert!(
                !planned.contains(&shard),
                "unselected BF16 shard {} must not be planned beside the Q4 weights",
                shard.display()
            );
        }
    }

    /// Both halves of a two-expert pair must be in the plan's artifact set.
    ///
    /// The low-noise expert is not opened until the schedule crosses the expert
    /// boundary — long after admission — so if it is absent here the plan
    /// freezes and pre-validates one expert while the generation reads two. A
    /// file swapped or deleted in between then changes the render, or fails it,
    /// with the plan still reporting valid. Same for its distill: the two
    /// adapters are separately trained, so one standing in for both is the
    /// wrong model rather than a degraded one.
    #[test]
    fn a_two_expert_pair_registers_both_experts_and_both_distills() {
        let root = tempfile::tempdir().unwrap();
        let at = |name: &str| root.path().join(name);
        let paths = ModelPaths {
            low_noise_transformer: Some(at("low-noise.gguf")),
            low_noise_distilled_lora: Some(at("low-noise-distill.safetensors")),
            transformer: at("high-noise.gguf"),
            transformer_shards: Vec::new(),
            vae: at("vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: Some(at("high-noise-distill.safetensors")),
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![at("umt5.safetensors")],
            text_tokenizer: None,
            decoder: None,
        };
        let frozen = mold_inference::FrozenEngineConfig::resolve("unused", &Config::default());
        let artifacts = concrete_artifacts_for_family(&paths, "wan", &[], &frozen);

        assert_eq!(
            artifacts.get(&ComponentRole::Transformer),
            Some(&at("high-noise.gguf"))
        );
        assert_eq!(
            artifacts.get(&ComponentRole::LowNoiseTransformer),
            Some(&at("low-noise.gguf")),
            "the low-noise expert must be its own frozen artifact"
        );
        assert_eq!(
            artifacts.get(&ComponentRole::DistilledLora),
            Some(&at("high-noise-distill.safetensors"))
        );
        assert_eq!(
            artifacts.get(&ComponentRole::LowNoiseDistilledLora),
            Some(&at("low-noise-distill.safetensors")),
            "each expert's distill is a distinct artifact"
        );

        // The roles are distinct keys, so the two experts cannot collapse into
        // one fingerprint entry.
        assert_ne!(
            artifacts.get(&ComponentRole::Transformer),
            artifacts.get(&ComponentRole::LowNoiseTransformer)
        );

        // A single-expert checkpoint gains neither role.
        let single = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            ..paths
        };
        let artifacts = concrete_artifacts_for_family(&single, "wan", &[], &frozen);
        assert!(!artifacts.contains_key(&ComponentRole::LowNoiseTransformer));
        assert!(!artifacts.contains_key(&ComponentRole::LowNoiseDistilledLora));
    }

    #[test]
    fn transformer_artifact_roles_do_not_alias_past_u8_cardinality() {
        let transformer_shards = (0..=usize::from(u8::MAX) + 1)
            .map(|index| PathBuf::from(format!("/models/transformer-{index}.safetensors")))
            .collect::<Vec<_>>();
        let paths = indexed_paths(transformer_shards.clone(), Vec::new());
        let frozen = mold_inference::FrozenEngineConfig::resolve("unused", &Config::default());
        let artifacts = concrete_artifacts_for_family(&paths, "z-image", &[], &frozen);
        let shard_artifacts = artifacts
            .iter()
            .filter(|(role, _)| matches!(role, ComponentRole::TransformerShard(_)))
            .collect::<Vec<_>>();

        assert_eq!(shard_artifacts.len(), transformer_shards.len());
        assert_eq!(
            artifacts.get(&ComponentRole::TransformerShard(0)),
            Some(&transformer_shards[0]),
            "the first shard must not be replaced by shard 256"
        );
        assert_eq!(
            artifacts.get(&ComponentRole::TransformerShard(usize::from(u8::MAX) + 1)),
            transformer_shards.last(),
        );
    }

    #[test]
    fn resolved_paths_model_fingerprint_retains_shards_past_u8_cardinality() {
        let paths = |first: &str| {
            indexed_paths(
                (0..=usize::from(u8::MAX) + 1)
                    .map(|index| {
                        if index == 0 {
                            PathBuf::from(first)
                        } else {
                            PathBuf::from(format!("/models/transformer-{index}.safetensors"))
                        }
                    })
                    .collect(),
                Vec::new(),
            )
        };

        assert_ne!(
            resolved_paths_model_fingerprint("model", paths("/models/first-a.safetensors")),
            resolved_paths_model_fingerprint("model", paths("/models/first-b.safetensors")),
            "a shard aliased out of the artifact map must not disappear from model identity"
        );
    }

    #[test]
    fn resolved_paths_model_fingerprint_includes_low_noise_expert() {
        let paths = |low: &str| {
            let mut paths = indexed_paths(Vec::new(), Vec::new());
            paths.low_noise_transformer = Some(PathBuf::from(low));
            paths
        };

        assert_ne!(
            resolved_paths_model_fingerprint("model", paths("/models/low-a.gguf")),
            resolved_paths_model_fingerprint("model", paths("/models/low-b.gguf")),
            "the paired expert must participate in model identity"
        );
    }

    #[test]
    fn text_encoder_and_lora_roles_do_not_alias_past_u16_cardinality() {
        let role_count = usize::from(u16::MAX) + 2;
        let text_encoder_files = (0..role_count)
            .map(|index| PathBuf::from(format!("/models/text-{index}.safetensors")))
            .collect::<Vec<_>>();
        let paths = indexed_paths(Vec::new(), text_encoder_files.clone());
        let loras = (0..role_count)
            .map(|index| PlannedLora {
                path: PathBuf::from(format!("/models/lora-{index}.safetensors")),
                content_fingerprint: ContentFingerprint(index.to_string()),
                scale_bits: 1.0_f64.to_bits(),
            })
            .collect::<Vec<_>>();
        let frozen = mold_inference::FrozenEngineConfig::resolve("unused", &Config::default());
        let artifacts = concrete_artifacts_for_family(&paths, "qwen-image", &loras, &frozen);

        assert_eq!(
            artifacts
                .keys()
                .filter(|role| matches!(role, ComponentRole::QwenShard(_)))
                .count(),
            role_count
        );
        assert_eq!(
            artifacts
                .keys()
                .filter(|role| matches!(role, ComponentRole::Lora(_)))
                .count(),
            role_count
        );
        assert_eq!(
            artifacts.get(&ComponentRole::QwenShard(0)),
            Some(&text_encoder_files[0]),
            "text encoder 65536 must not replace text encoder zero"
        );
        assert_eq!(
            artifacts.get(&ComponentRole::QwenShard(usize::from(u16::MAX) + 1)),
            text_encoder_files.last(),
        );
        assert_eq!(
            artifacts.get(&ComponentRole::Lora(0)),
            Some(&loras[0].path),
            "LoRA 65536 must not replace LoRA zero"
        );
        assert_eq!(
            artifacts.get(&ComponentRole::Lora(usize::from(u16::MAX) + 1)),
            loras.last().map(|lora| &lora.path),
        );
    }

    #[test]
    fn indexed_component_roles_preserve_legacy_serialization_shape() {
        assert_eq!(
            serde_json::to_string(&ComponentRole::TransformerShard(u8::MAX.into())).unwrap(),
            r#"{"TransformerShard":255}"#
        );
        assert_eq!(
            serde_json::to_string(&ComponentRole::QwenShard(u16::MAX.into())).unwrap(),
            r#"{"QwenShard":65535}"#
        );
        assert_eq!(
            serde_json::to_string(&ComponentRole::TransformerShard(256)).unwrap(),
            r#"{"TransformerShard":256}"#
        );
        assert_eq!(
            serde_json::to_string(&ComponentRole::Lora(65_536)).unwrap(),
            r#"{"Lora":65536}"#
        );
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
            identity_warning: None,
            identity_embedding: None,
            identity_pin: Default::default(),
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
                    pending_artifacts: BTreeMap::new(),
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
            capacity_park: None,
            model_config_overlay: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_admission_by_device: BTreeMap::new(),
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
        validate_before_cuda(plan, "cuda:0", 0, &config, &request, None).unwrap();
    }

    #[test]
    fn catalog_overlay_keeps_planning_and_cuda_validation_stable_on_cold_config() {
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("catalog.safetensors");
        let vae = root.path().join("vae.safetensors");
        let encoder = root.path().join("qwen3.safetensors");
        let tokenizer = root.path().join("tokenizer.json");
        for path in [&transformer, &vae, &encoder, &tokenizer] {
            std::fs::write(path, b"catalog").unwrap();
        }
        let model_config = ModelConfig {
            transformer: Some(transformer.display().to_string()),
            vae: Some(vae.display().to_string()),
            text_encoder_files: Some(vec![encoder.display().to_string()]),
            text_tokenizer: Some(tokenizer.display().to_string()),
            family: Some("z-image".to_string()),
            ..Default::default()
        };
        let cold_config = Config::default();
        let mut effective_config = cold_config.clone();
        effective_config
            .models
            .insert("cv:123".to_string(), model_config.clone());
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"cv:123","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let paths = ModelPaths::resolve(&request.model, &effective_config).unwrap();
        let engine_config =
            mold_inference::FrozenEngineConfig::resolve(&request.model, &effective_config);
        let prepared = PreparedExecutionInputs {
            identity_warning: None,
            identity_embedding: None,
            identity_pin: Default::default(),
            authority_fingerprint: preparation_authority_fingerprint(
                &effective_config,
                &request,
                &paths,
                &engine_config,
            ),
            by_device: BTreeMap::from([(
                "cuda:0".to_string(),
                PreparedDeviceExecutionInputs {
                    engine_paths: paths.clone(),
                    engine_config: engine_config.clone(),
                    pending_artifacts: BTreeMap::new(),
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
            capacity_park: None,
            model_config_overlay: Some(Arc::new(model_config)),
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_admission_by_device: BTreeMap::new(),
        };

        assert!(matches!(
            resolve_execution_plans_with_prepared(
                &cold_config,
                &request,
                &devices(&[24 * GIB]),
                false,
                None,
            ),
            Err(ExecutionPlanError::MissingArtifacts { .. })
        ));
        let plan = resolve_execution_plans_with_prepared(
            &cold_config,
            &request,
            &devices(&[24 * GIB]),
            false,
            Some(&prepared),
        )
        .unwrap()
        .remove(0);
        assert_eq!(plan.admission_paths, paths);
        validate_before_cuda(&plan, "cuda:0", 0, &cold_config, &request, Some(&prepared)).unwrap();
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
            identity_warning: None,
            identity_embedding: None,
            identity_pin: Default::default(),
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
                    pending_artifacts: BTreeMap::new(),
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
            capacity_park: None,
            model_config_overlay: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_admission_by_device: BTreeMap::new(),
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

    /// A same-size in-place overwrite is invisible in `len`, and a caller can
    /// restore `mtime`, so the only thing that can catch it is `ctime` (plus
    /// the inode for replacement). This used to be asserted by overwriting a
    /// real file and demanding the fingerprint change — but `ctime` advances
    /// on the kernel's coarse clock, measured here at ~98 ms granularity, so
    /// both writes routinely landed in one tick and the identities were
    /// legitimately equal. The test was racing the clock rather than checking
    /// the contract.
    ///
    /// Pin the derivation instead: `ctime` and the inode participate, `mtime`
    /// is not an input at all. That is strictly stronger than the old
    /// assertion (it holds no matter how fast the two writes land) and it
    /// still fails if anyone swaps `ctime` for `mtime`.
    #[test]
    fn artifact_identity_is_derived_from_inode_and_ctime_never_mtime() {
        use filetime::{set_file_mtime, FileTime};

        let root = TempDir::new().unwrap();
        let path = root.path().join("weights.safetensors");
        std::fs::write(&path, b"aaaa").unwrap();

        let metadata = path.metadata().unwrap();
        let identity = artifact_metadata_identity(&path, &metadata);
        assert_eq!(identity.len, 4);

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(
                identity.platform_identity,
                vec![
                    metadata.dev(),
                    metadata.ino(),
                    metadata.ctime() as u64,
                    metadata.ctime_nsec() as u64,
                ],
                "identity must be inode + ctime so a same-size in-place \
                 overwrite cannot hide behind a restored mtime"
            );

            // Rewriting mtime to an arbitrary value must not move the identity
            // toward equality with anything: mtime is simply not consulted.
            let mtime_only = FileTime::from_unix_time(metadata.ctime() - 86_400, 0);
            set_file_mtime(&path, mtime_only).unwrap();
            let after = path.metadata().unwrap();
            assert_eq!(
                FileTime::from_last_modification_time(&after),
                mtime_only,
                "fixture must actually have moved mtime"
            );
            assert_eq!(
                artifact_metadata_identity(&path, &after).platform_identity[..2],
                identity.platform_identity[..2],
                "dev/inode are stable across an mtime-only change"
            );
        }
    }

    /// Replacement — the shape a re-download actually takes — must change the
    /// identity, and unlike an in-place overwrite this is deterministic: the
    /// inode differs regardless of how close the two operations land.
    #[test]
    fn artifact_identity_changes_when_the_file_is_replaced() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("weights.safetensors");
        std::fs::write(&path, b"aaaa").unwrap();
        let before = artifact_metadata_identity(&path, &path.metadata().unwrap());

        replace_artifact_bytes(&path, b"bbbb");

        assert_ne!(
            before,
            artifact_metadata_identity(&path, &path.metadata().unwrap()),
            "a replaced artifact must never reuse the previous identity"
        );
    }

    #[test]
    fn legacy_verified_digest_marker_cannot_override_current_artifact_bytes() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("weights.safetensors");
        std::fs::write(&path, b"already verified model bytes").unwrap();
        let forged = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        mold_core::download::write_sha256_marker(&path, forged).unwrap();
        let file = mold_core::secure_file::open_regular_file_no_follow(&path).unwrap();
        let expected = mold_core::secure_file::sha256_open_file(&file).unwrap();

        assert_eq!(
            hash_equivalence_artifact_contents(&path).unwrap(),
            EquivalenceContentIdentity::Sha256(expected),
            "placement preparation must authenticate current bytes through the durable pinned-digest authority"
        );
    }

    #[test]
    fn runtime_semantic_parser_matches_cfg_plus_and_long_prompt_contracts() {
        assert_eq!(
            runtime_semantic_setting("MOLD_CFG_PLUS", Some("true"))
                .unwrap()
                .value,
            CanonicalRuntimeValue::Boolean(true)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_CFG_PLUS", Some("yes"))
                .unwrap()
                .value,
            CanonicalRuntimeValue::Boolean(true)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_CFG_PLUS", Some("TRUE"))
                .unwrap()
                .value,
            CanonicalRuntimeValue::Boolean(false)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_CFG_PLUS", Some(" true "))
                .unwrap()
                .value,
            CanonicalRuntimeValue::Boolean(false)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_LONG_PROMPTS", Some("1"))
                .unwrap()
                .value,
            CanonicalRuntimeValue::Boolean(true)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_LONG_PROMPTS", Some("true"))
                .unwrap()
                .value,
            CanonicalRuntimeValue::Boolean(false)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_LONG_PROMPTS", Some("yes"))
                .unwrap()
                .value,
            CanonicalRuntimeValue::Boolean(false)
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_LTX2_VAE_DECODE_CHUNK_FRAMES", Some(" 4 "))
                .unwrap()
                .value,
            CanonicalRuntimeValue::Unsigned(mold_inference::runtime_env::parse_u64(" 4 "))
        );
        assert_eq!(
            runtime_semantic_setting("MOLD_WUERSTCHEN_DECODER_GUIDANCE", Some(" 1.5 "))
                .unwrap()
                .value,
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
                    .unwrap()
                    .value,
                CanonicalRuntimeValue::Presence(true)
            );
        }
        assert_eq!(
            runtime_semantic_setting("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", None)
                .unwrap()
                .value,
            CanonicalRuntimeValue::Unset
        );
    }

    /// The classification contract for issue #685: every name in
    /// `mold_inference::runtime_env::ENGINE_SHAPING_VARIABLES` must map to its
    /// own `RuntimeSemanticVariable`. A variable registered in mold-inference
    /// without a matching arm here used to compile, pass CI, and then panic on
    /// the first generation request that reached planning; this test turns
    /// that mismatch into a named CI failure at the point of the mistake.
    #[test]
    fn every_engine_shaping_variable_has_a_semantic_class() {
        let mut classes = BTreeSet::new();
        for name in mold_inference::runtime_env::ENGINE_SHAPING_VARIABLES {
            let variable = runtime_semantic_variable(name).unwrap_or_else(|| {
                panic!(
                    "engine-shaping variable {name} has no RuntimeSemanticVariable \
                     classification; add a match arm in runtime_semantic_variable \
                     (crates/mold-server/src/execution_plan.rs)"
                )
            });
            assert!(
                classes.insert(variable),
                "engine-shaping variable {name} collapsed into an equivalence class \
                 already claimed by another variable ({variable:?}); every name needs \
                 its own RuntimeSemanticVariable variant"
            );
            // Exercise the value-canonicalization arms too: a classified name
            // must accept any raw value without panicking.
            assert!(runtime_semantic_setting(name, Some("probe")).is_some());
        }
    }

    #[test]
    fn unknown_variable_names_are_unclassified() {
        assert_eq!(
            runtime_semantic_variable("MOLD_NOT_A_SHAPING_VARIABLE"),
            None
        );
        assert!(runtime_semantic_setting("MOLD_NOT_A_SHAPING_VARIABLE", Some("1")).is_none());
    }

    #[test]
    fn unclassified_runtime_variable_error_names_the_variable() {
        let error = ExecutionPlanError::UnclassifiedRuntimeVariable {
            name: "MOLD_X".to_string(),
        };
        let rendered = error.to_string();
        assert!(rendered.contains("MOLD_X"), "{rendered}");
        assert!(
            rendered.contains("RuntimeSemanticVariable"),
            "the error must point at the classifier to edit: {rendered}"
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
                &BTreeMap::new(),
            )
            .expect("rebuild descriptor classifies every frozen engine-shaping variable")
        };

        assert_eq!(
            rebuild(&plan.components).fingerprint(),
            rebuild(&reverse_inserted).fingerprint()
        );
    }

    /// The equivalence identity is a real content hash, so a same-size in-place
    /// overwrite with a restored mtime genuinely does change it. What made this
    /// racy was the fact cache in front of it: entries are keyed on the metadata
    /// identity, and when both writes land in one coarse `ctime` tick the second
    /// lookup is served the first write's cached hash. Evicting between the two
    /// snapshots keeps the assertion about the content identity — which is what
    /// the test is named for — instead of about cache-invalidation timing.
    #[test]
    fn same_size_in_place_overwrite_changes_equivalence_content_identity() {
        use filetime::{set_file_mtime, FileTime};

        let root = TempDir::new().unwrap();
        let path = root.path().join("weights.safetensors");
        std::fs::write(&path, b"aaaa").unwrap();
        let original_mtime = FileTime::from_last_modification_time(&path.metadata().unwrap());
        artifact_fact_cache().lock().unwrap().remove_path(&path);
        let before = equivalence_fingerprint_path(&path);

        std::fs::write(&path, b"bbbb").unwrap();
        set_file_mtime(&path, original_mtime).unwrap();
        artifact_fact_cache().lock().unwrap().remove_path(&path);

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

    /// The degraded (cache-only) identity is derived from observed metadata, so
    /// this test needs the two byte states to carry genuinely different
    /// metadata. An in-place same-size rewrite could not guarantee that — it
    /// depended on the two writes falling in different coarse `ctime` ticks —
    /// so the fixture's own precondition failed most of the time on a fast
    /// filesystem, before the assertion under test ran. Replacing the file
    /// changes the inode and is deterministic everywhere.
    #[test]
    fn cache_miss_identity_changes_when_artifact_metadata_changes() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("changing.safetensors");
        std::fs::write(&path, b"aaaa").unwrap();
        let first_metadata = artifact_metadata_identity(&path, &path.metadata().unwrap());
        artifact_fact_cache().lock().unwrap().remove_path(&path);
        let first = artifact_facts_path_with_policy(&path, true).content;

        replace_artifact_bytes(&path, b"bbbb");
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
            identity_warning: None,
            identity_embedding: None,
            identity_pin: Default::default(),
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
                    pending_artifacts: BTreeMap::new(),
                    prepared_available_vram_bytes: 24 * GIB,
                    capacity_sensitive: false,
                },
            )]),
            retryable_device_failures: BTreeMap::new(),
            capacity_park: None,
            model_config_overlay: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_admission_by_device: BTreeMap::new(),
        };
        let progress = crate::variant_dependencies::PreparationProgressSink::default();
        warm_execution_equivalence_cache(&config, &request, &prepared, Some(&progress));
        let progress = progress
            .snapshot()
            .expect("warm pass reports its byte progress");
        assert_eq!(progress.component, "Verifying model files");
        assert!(progress.bytes_total > 0);
        assert_eq!(progress.bytes_done, progress.bytes_total);

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
            artifact_root: PathBuf::from("/models"),
            is_schnell: Some(false),
            is_turbo: None,
            scheduler: None,
            t5_variant: Some("q4".into()),
            qwen3_variant: None,
            qwen2_variant: None,
            qwen2_text_encoder_mode: None,
            ltx2_gemma_variant: None,
            umt5_variant: None,
            selected_t5_path: None,
            selected_qwen3_paths: Vec::new(),
            selected_qwen2_path: None,
            selected_gemma_paths: Vec::new(),
            selected_umt5_path: None,
            identity_assets: None,
            h3_factory_authority: None,
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

        let mut relocated_config = engine_config.clone();
        relocated_config.artifact_root = PathBuf::from("/different-mold-home/models");
        assert_eq!(
            fingerprint,
            execution_fingerprint(
                "cv:opaque",
                &device,
                &effective,
                &components,
                &relocated_config,
                &[],
                false,
            ),
            "the storage trust root is not part of execution identity"
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
                umt5_variant: None,
                h3_factory_authority_sha256: None,
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
    fn frozen_ltx2_chain_model_preserves_absent_vae_through_exact_recovery() {
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("ltx2-transformer.safetensors");
        std::fs::write(&transformer, b"ltx2").unwrap();
        let model = "ltx-2-test:fp8";
        let mut config = Config::default();
        config.models.insert(
            model.to_string(),
            mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(String::new()),
                family: Some("ltx2".to_string()),
                ..mold_core::ModelConfig::default()
            },
        );
        let mut paths = indexed_paths(Vec::new(), Vec::new());
        paths.transformer = transformer.canonicalize().unwrap();
        paths.vae = PathBuf::new();

        let frozen = freeze_chain_model_with_paths(&config, model, paths).unwrap();

        assert_eq!(frozen.config.vae, None);
        let recovered = ModelPaths::resolve_from_model_config_exact(&frozen.config)
            .expect("LTX-2 frozen paths remain exactly resolvable without a standalone VAE");
        assert_eq!(recovered.transformer, transformer.canonicalize().unwrap());
        assert_eq!(recovered.vae, PathBuf::new());
    }

    #[test]
    fn frozen_wan_chain_model_preserves_both_experts_and_distilled_loras() {
        let root = TempDir::new().unwrap();
        let write = |name: &str| {
            let path = root.path().join(name);
            std::fs::write(&path, name.as_bytes()).unwrap();
            path
        };
        let high = write("high.gguf");
        let low = write("low.gguf");
        let vae = write("vae.safetensors");
        let high_lora = write("high-lora.safetensors");
        let low_lora = write("low-lora.safetensors");
        let model = "wan-pair-test";
        let config = Config::default();
        let mut paths = indexed_paths(Vec::new(), Vec::new());
        paths.transformer = high.canonicalize().unwrap();
        paths.low_noise_transformer = Some(low.canonicalize().unwrap());
        paths.vae = vae.canonicalize().unwrap();
        paths.distilled_lora = Some(high_lora.canonicalize().unwrap());
        paths.low_noise_distilled_lora = Some(low_lora.canonicalize().unwrap());

        let frozen = freeze_chain_model_with_paths(&config, model, paths.clone()).unwrap();
        let recovered = ModelPaths::resolve_from_model_config_exact(&frozen.config)
            .expect("the frozen Wan pair remains exactly resolvable");

        assert_eq!(recovered, paths);
        let original_fingerprint = frozen.model_fingerprint;
        replace_artifact_bytes(&low, b"changed-low");
        assert_ne!(
            frozen_model_fingerprint(model, &frozen.config).unwrap(),
            original_fingerprint,
            "replacing the low-noise expert must invalidate durable chain authority"
        );
        replace_artifact_bytes(&low, b"low.gguf");
        let restored_fingerprint = frozen_model_fingerprint(model, &frozen.config).unwrap();
        replace_artifact_bytes(&low_lora, b"changed-low-lora");
        assert_ne!(
            frozen_model_fingerprint(model, &frozen.config).unwrap(),
            restored_fingerprint,
            "replacing the low-noise distillation adapter must invalidate durable chain authority"
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
        // `b"changed-tokenizer"` is exactly as long as `b"t5-tokenizer.json"`,
        // so this rewrite used to be invisible unless the two writes landed in
        // different coarse `ctime` ticks. This test is about *which files*
        // participate in the durable identity, not about same-size detection
        // (which `artifact_identity_is_derived_from_inode_and_ctime_never_mtime`
        // now pins), so replace the file rather than racing the clock.
        replace_artifact_bytes(&assets.join("t5-tokenizer.json"), b"changed-tokenizer");
        assert_ne!(
            frozen_model_fingerprint(model, &frozen.config).unwrap(),
            original_fingerprint,
            "tokenizers are engine inputs and must invalidate the durable identity"
        );
        replace_artifact_bytes(&assets.join("t5-tokenizer.json"), b"t5-tokenizer.json");
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
