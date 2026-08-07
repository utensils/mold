//! Reusable MiniMax H3 model primitives.
//!
//! Nothing in this module registers a runnable inference engine. The H3
//! license gate and the family capability contract remain the authority for
//! whether callers may construct these primitives with real weights.

mod artifacts;
mod attention;
mod comfy_dit;
mod comfy_quant;
mod config;
mod dit;
mod loader;
mod model;
mod presentation;
mod processor;
mod qwen_quant;
mod text;
mod vision;
mod visual_condition;
mod visual_geometry;
mod visual_vae;
mod visual_weights;

pub mod audio;
pub mod audio_config;
pub mod audio_weights;

pub use audio::{
    AudioPosterior, AudioPosteriorSelection, AudioSoundtrackAssociation, AudioVae,
    AudioVaeCancellation, AudioVaePhase, NeverCancel, StereoLatents, StereoWaveform,
};
pub use audio_config::{
    AudioVaeConfig, MiniMaxH3AudioContract, LATENT_CHANNELS, LATENT_ROWS_PER_SECOND,
    SAMPLES_PER_LATENT, SAMPLE_RATE,
};
pub use audio_weights::{
    load_validated_audio_vae, validate_audio_safetensors, AudioTensorLayout, AudioTensorSpec,
};

pub use artifacts::{
    validate_checkpoint_keys, ArtifactError, ArtifactFingerprint, ArtifactRole,
    ArtifactVerificationProgress, CheckpointKeyReport, ConditionerArtifacts, FrozenArtifact,
};
pub use attention::{
    execute_h3_attention, execute_h3_attention_with_telemetry, H3AttentionActivation,
    H3AttentionBackend, H3AttentionDType, H3AttentionDevice, H3AttentionError,
    H3AttentionErrorCode, H3AttentionKernel, H3AttentionMask, H3AttentionModelContract,
    H3AttentionReleaseCandidateBuild, H3AttentionReleaseRejection, H3AttentionReleaseRequirement,
    H3AttentionRequirement, H3AttentionRuntimeAuthority, H3AttentionScope, H3AttentionTelemetry,
    H3AttentionWorkspace, H3FrozenAttentionExecution, H3FrozenAttentionPlan,
    H3_ATTENTION_RC_CLAIM_MARKER, H3_ATTENTION_RC_FEATURE, H3_ATTENTION_RC_SCHEMA,
    H3_DENSE_SYNTHETIC_MAX_SCORE_ELEMENTS, H3_FLASH_ATTN_CRATE_SHA256,
    H3_FLASH_ATTN_PACKAGE_VERSION, H3_FLASH_ATTN_QUALIFIED_COMPUTE_CAPABILITY,
};
pub use comfy_dit::{
    inspect_h3_comfy_published_header, H3ComfyAccuracyTier, H3ComfyCheckpointCandidate,
    H3ComfyCheckpointError, H3ComfyCheckpointErrorCode, H3ComfyCurveInterpolation,
    H3ComfyFrozenStrategyMetadata, H3ComfyMemoryAccounting, H3ComfyPrunedFormat,
    H3ComfyPublishedArtifact, H3ComfyQuantizationPolicy, H3ComfyRuntimeBackend,
    H3ComfyRuntimeRejection, H3ComfyRuntimeRequirement, H3_COMFYUI_SOURCE_REVISION,
    H3_COMFY_KITCHEN_REPOSITORY, H3_COMFY_KITCHEN_SOURCE_REVISION, H3_COMFY_KITCHEN_VERSION,
    H3_COMFY_ORG_SOURCE_REVISION,
};
pub use comfy_quant::{
    H3ComfyFp8ScaledLinear, H3ComfyInt8ConvRotLinear, H3ComfyNvfp4AwqLinear,
    H3_COMFY_CONVROT_GROUP_SIZE, H3_COMFY_NVFP4_BLOCK_SIZE, H3_COMFY_PORTABLE_ROW_CHUNK,
};
pub use config::{
    H3ConditionerConfig, H3TextConfig, H3VisionConfig, H3_BF16_PARAMETER_BYTES,
    H3_COMFY_INT8_CONVROT_PARAMETER_BYTES, H3_COMFY_INT8_CONVROT_SAFETENSORS_BYTES,
    H3_COMFY_SAFETENSORS_BYTES, H3_FULL_CHECKPOINT_BYTES, H3_FULL_LANGUAGE_LAYERS,
    H3_SELECTED_LANGUAGE_LAYERS,
};
pub use dit::{
    audit_h3_checkpoint, expected_h3_weight_specs, interpolate_h3_adaln_curve, pack_h3_audio,
    patchify_h3_video, reorder_h3_grouped_qkv, unpack_h3_audio, unpatchify_h3_video, H3AdaLnMode,
    H3AuditedCheckpoint, H3BlockProgress, H3BlockStack, H3CheckpointComponent,
    H3CheckpointInventory, H3CheckpointSource, H3ForwardInput, H3FrozenPackedLayout, H3LoadPlan,
    H3Modality, H3PackedLayout, H3PrecisionProfile, H3QkvLayout, H3TensorSpec, H3Transformer,
    H3TransformerConfig, H3TransformerOutput, H3TransformerTask, H3WeightAuditError,
    H3WeightAuditIssue, H3_MODALITY_COUNT,
};
pub use loader::{
    load_bf16_conditioner, load_prepared_bf16_conditioner, prepare_conditioner_assets,
    prepare_conditioner_assets_with_progress, validate_processor_assets, H3LoadError,
    LoadedH3Conditioner, PreparedH3ConditionerAssets,
};
pub use model::{
    ConditionerCheckpoint, H3ConditionerInput, H3DTypeProfile, H3Layer50Conditioner, H3VisionInput,
};
pub use presentation::{
    build_fl2va_presentation, build_ref2va_presentation, build_text_presentation, H3ModalityTag,
    H3Presentation, H3RawTokenizer, PresentationError, RefPresentation, RefPresentationKind,
    VideoPresentationBlock,
};
pub use processor::{
    create_mm_token_type_ids, pack_qwen_vision_u8, qwen_mrope_positions, sample_video_frames,
    GridThw, PackedVisionPatches, QwenMmTokenType, SampledVideo,
};
pub use qwen_quant::{
    expected_h3_qwen_int8_weight_only_schema, validate_h3_qwen_int8_weight_only_schema,
    H3QwenFp32Boundary, H3QwenInt8CheckpointMetadata, H3QwenInt8LayerMetadata,
    H3QwenInt8MemoryAccounting, H3QwenInt8PolicyError, H3QwenInt8WeightOnlyPolicy,
    H3QwenLinearExecution, H3QwenTensorDType, H3QwenTensorSpec,
    H3_QWEN_INT8_QUANTIZED_LINEAR_COUNT, H3_QWEN_INT8_WEIGHT_ONLY_STABLE_ID,
};
pub use visual_condition::{
    ConditionEncodeMode, SignedRgbPixels, Uint8RgbPixels, UnitRgbPixels, H3_IMAGENET_MEAN,
    H3_IMAGENET_STD, H3_LATENTS_MEAN, H3_LATENTS_STD,
};
pub use visual_geometry::{
    EndpointResizePlan, ReferenceImageResizePlan, SpatialTilePlan, TemporalDecodeChunk,
    TemporalDecodePlan, TemporalEncodeChunk, TemporalEncodePlan, VisualTemporalGeometry,
};
pub use visual_vae::{
    DecodeComputePolicy, DecodeSink, MiniMaxH3VisualVae, MiniMaxH3VisualVaeConfig,
    NoopVisualVaeObserver, VisualAttentionBackend, VisualVaeEvent, VisualVaeObserver,
    VisualVaePhase, WeightLayout,
};
pub use visual_weights::{
    comfy_tensor_transforms, component_fingerprint, component_fingerprint_with_observer,
    expected_comfy_weight_shapes, expected_diffusers_weight_shapes, inspect_safetensors_header,
    inspect_visual_vae_config, validate_comfy_weight_file, validate_diffusers_weight_index,
    ComfyTensorTransform, DiffusersWeightIndex, NoopVisualWeightReadObserver, SafetensorsHeader,
    SafetensorsTensorHeader, ValidatedVisualVaeWeights, VisualVaeComponentRole,
    VisualVaeWeightInspection, VisualWeightReadObserver, COMFY_VISUAL_VAE_FILENAME,
};
