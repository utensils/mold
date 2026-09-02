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
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
mod private_runtime_observation;
mod processor;
mod qwen_nvfp4;
mod special_tokens;
// The private-artifact runtime is intentionally not wired into a public
// factory until the separate compliance activation lands.
#[allow(dead_code)]
mod qwen_nvfp4_runtime;
mod qwen_quant;
mod text;
mod turbo_lora;
mod turbo_runtime;
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
    inspect_audio_vae_config_bytes, load_validated_audio_vae,
    load_validated_audio_vae_with_observer, validate_audio_safetensors, AudioTensorLayout,
    AudioTensorSpec, AudioVaeLoadObserver, NoopAudioVaeLoadObserver,
};

pub use artifacts::{
    validate_checkpoint_keys, ArtifactError, ArtifactFingerprint, ArtifactRole,
    ArtifactVerificationProgress, CheckpointKeyReport, ConditionerArtifacts, FrozenArtifact,
};
pub use attention::{
    execute_h3_attention, execute_h3_attention_with_telemetry,
    h3_attention_release_provenance_marker, h3_query_chunk_rows, H3AttentionActivation,
    H3AttentionBackend, H3AttentionDType, H3AttentionDevice, H3AttentionError,
    H3AttentionErrorCode, H3AttentionKernel, H3AttentionMask, H3AttentionModelContract,
    H3AttentionReleaseCandidateBuild, H3AttentionReleaseRejection, H3AttentionReleaseRequirement,
    H3AttentionRequirement, H3AttentionRuntimeAuthority, H3AttentionScope, H3AttentionTelemetry,
    H3AttentionWorkspace, H3FrozenAttentionExecution, H3FrozenAttentionPlan,
    H3_ATTENTION_RC_CLAIM_MARKER, H3_ATTENTION_RC_FEATURE, H3_ATTENTION_RC_SCHEMA,
    H3_ATTENTION_RELEASE_PROVENANCE_MARKER, H3_ATTENTION_RELEASE_PROVENANCE_PREFIX,
    H3_DENSE_SYNTHETIC_MAX_SCORE_ELEMENTS, H3_FLASH_ATTN_PACKAGE_VERSION,
    H3_FLASH_ATTN_QUALIFIED_COMPUTE_CAPABILITY, H3_FLASH_ATTN_SOURCE,
    H3_METAL_DENSE_CHUNK_MAX_SCORE_ELEMENTS, H3_METAL_DENSE_MAX_QUERY_CHUNK_ROWS,
};
pub use comfy_dit::{
    inspect_h3_comfy_published_header, open_h3_comfy_published_int8_checkpoint,
    H3ComfyAccuracyTier, H3ComfyCheckpointCandidate, H3ComfyCheckpointError,
    H3ComfyCheckpointErrorCode, H3ComfyCurveInterpolation, H3ComfyFrozenStrategyMetadata,
    H3ComfyInt8BlockLoader, H3ComfyInt8BlockMemory, H3ComfyInt8Cancellation,
    H3ComfyMemoryAccounting, H3ComfyNeverCancel, H3ComfyOpenedBlockMemoryEvidence,
    H3ComfyOpenedInt8Checkpoint, H3ComfyOpenedMemoryEvidence, H3ComfyPrunedFormat,
    H3ComfyPublishedArtifact, H3ComfyQuantizationPolicy, H3ComfyRuntimeBackend,
    H3ComfyRuntimeRejection, H3ComfyRuntimeRequirement, H3_COMFYUI_SOURCE_REVISION,
    H3_COMFY_KITCHEN_REPOSITORY, H3_COMFY_KITCHEN_SOURCE_REVISION, H3_COMFY_KITCHEN_VERSION,
    H3_COMFY_ORG_SOURCE_REVISION, H3_COMFY_PUBLISHED_INT8_HEADER_LEN,
    H3_COMFY_PUBLISHED_INT8_HEADER_SHA256, H3_COMFY_PUBLISHED_INT8_TENSOR_COUNT,
};
pub use comfy_quant::{
    h3_nvfp4_weight_staging_bytes, select_h3_int8_linear_kind, select_h3_nvfp4_linear_kind,
    H3ComfyFp8ScaledLinear, H3ComfyInt8ConvRotLinear, H3ComfyInt8TensorwiseEmbedding,
    H3ComfyNvfp4AwqLinear, H3Int8LinearKind, H3Nvfp4LinearKind, H3_COMFY_CONVROT_GROUP_SIZE,
    H3_COMFY_NVFP4_BLOCK_SIZE, H3_COMFY_PORTABLE_ROW_CHUNK,
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
    H3LoadedTransformerBlock, H3Modality, H3PackedLayout, H3PrecisionProfile, H3QkvLayout,
    H3StreamedTransformer, H3TensorSpec, H3Transformer, H3TransformerBlockLoader,
    H3TransformerConfig, H3TransformerOutput, H3TransformerStep, H3TransformerTask,
    H3WeightAuditError, H3WeightAuditIssue, H3_MODALITY_COUNT,
};
#[cfg(feature = "h3-private-uat")]
pub use dit::{
    capture_h3_private_transformer_primitives, H3PrivateTransformerCapture,
    H3PrivateTransformerCaptureInput,
};
pub use loader::{
    load_bf16_conditioner, load_prepared_bf16_conditioner, prepare_conditioner_assets,
    prepare_conditioner_assets_with_progress, validate_processor_assets, H3LoadError,
    LoadedH3Conditioner, PreparedH3ConditionerAssets,
};
#[cfg(feature = "h3-private-uat")]
pub use loader::{load_streamed_bf16_conditioner, LoadedH3StreamingConditioner};
#[cfg(feature = "h3-private-uat")]
pub use model::H3Layer50StreamingConditioner;
pub use model::{
    ConditionerCheckpoint, H3ConditionerInput, H3DTypeProfile, H3Layer50Conditioner, H3VisionInput,
};
pub use presentation::{
    build_fl2va_presentation, build_ref2va_presentation, build_text_presentation, H3ModalityTag,
    H3Presentation, H3RawTokenizer, PresentationError, RefPresentation, RefPresentationKind,
    VideoPresentationBlock,
};
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub use private_runtime_observation::{H3PrivateWorkspaceCapture, H3PrivateWorkspaceObservation};
pub use processor::{
    create_mm_token_type_ids, pack_qwen_vision_u8, qwen_mrope_positions, sample_video_frames,
    GridThw, PackedVisionPatches, QwenMmTokenType, SampledVideo,
};
pub use qwen_nvfp4::{
    expected_h3_qwen_nvfp4_awq_schema, inspect_h3_qwen_nvfp4_awq_header,
    validate_h3_qwen_nvfp4_awq_schema, H3QwenNvfp4AwqError, H3QwenNvfp4AwqExecution,
    H3QwenNvfp4AwqInspection, H3QwenNvfp4AwqMemoryAccounting, H3QwenNvfp4AwqPolicy,
    H3QwenNvfp4AwqSchema, H3QwenQuantMarker, H3_QWEN_NVFP4_AWQ_FILENAME,
    H3_QWEN_NVFP4_AWQ_FILE_BYTES, H3_QWEN_NVFP4_AWQ_HEADER_BYTES, H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES,
    H3_QWEN_NVFP4_AWQ_POLICY_SHA256, H3_QWEN_NVFP4_AWQ_SHA256, H3_QWEN_NVFP4_AWQ_STABLE_ID,
    H3_QWEN_NVFP4_AWQ_TENSOR_COUNT, H3_QWEN_NVFP4_LINEAR_COUNT,
    H3_QWEN_NVFP4_PRE_QUANT_SCALE_COUNT, H3_QWEN_QUANT_MARKER_COUNT,
};
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[doc(hidden)]
pub use qwen_nvfp4_runtime::{
    load_h3_qwen_nvfp4_conditioner_after_authorization,
    load_h3_qwen_nvfp4_conditioner_from_authority,
    open_h3_qwen_nvfp4_authority_after_authorization,
    released_h3_qwen_nvfp4_max_weight_staging_bytes, released_h3_qwen_nvfp4_output_tensor_bytes,
    released_h3_qwen_nvfp4_runtime_memory_facts,
    released_h3_qwen_nvfp4_runtime_memory_facts_for_placement, H3AuthenticatedQwenNvfp4Authority,
    H3QwenNvfp4LoadEvent, H3QwenNvfp4LoadObserver, H3QwenNvfp4RuntimeError,
    H3QwenNvfp4RuntimeMemoryFacts, H3QwenNvfp4RuntimePlacement, LoadedH3QwenNvfp4Conditioner,
    NoopH3QwenNvfp4LoadObserver,
};
pub use qwen_quant::{
    expected_h3_qwen_int8_weight_only_schema, validate_h3_qwen_int8_weight_only_schema,
    H3QwenFp32Boundary, H3QwenInt8CheckpointMetadata, H3QwenInt8LayerMetadata,
    H3QwenInt8MemoryAccounting, H3QwenInt8PolicyError, H3QwenInt8WeightOnlyPolicy,
    H3QwenLinearExecution, H3QwenTensorDType, H3QwenTensorSpec,
    H3_QWEN_INT8_QUANTIZED_LINEAR_COUNT, H3_QWEN_INT8_WEIGHT_ONLY_STABLE_ID,
};
pub use special_tokens::{
    register_extra_special_tokens, H3SpecialTokenError, H3_BASE_VOCABULARY_SIZE,
    H3_EXTRA_SPECIAL_TOKENS, H3_REGISTERED_ADDED_TOKEN_COUNT, H3_REGISTERED_MAX_TOKEN_ID,
    H3_REGISTERED_VOCABULARY_SIZE, H3_RELEASED_ADDED_TOKEN_COUNT,
    H3_RELEASED_ADDITIONAL_SPECIAL_TOKEN_COUNT, H3_RELEASED_MAX_TOKEN_ID,
    H3_RELEASED_VOCABULARY_SIZE,
};
pub use turbo_lora::{
    authenticate_h3_turbo_lora_adapter, inspect_h3_turbo_lora_adapter, H3TurboLoraContract,
    H3TurboLoraError, H3TurboLoraErrorCode, H3TurboLoraExpectation, H3TurboLoraInspection,
    H3TurboLoraModule, H3TurboLoraModuleKind, H3TurboLoraModuleScope, H3TurboLoraResult,
    H3TurboLoraShape, H3TurboLoraTensorRef, H3TurboLoraTier, H3_TURBO_LORA_ALPHA_DTYPE,
    H3_TURBO_LORA_DIRECTORY, H3_TURBO_LORA_DRBAPH_REPOSITORY, H3_TURBO_LORA_DRBAPH_SOURCE_REVISION,
    H3_TURBO_LORA_DYNAMIC_TENSOR_COUNT, H3_TURBO_LORA_FUSED_QKV_MULTIPLE, H3_TURBO_LORA_KEY_PREFIX,
    H3_TURBO_LORA_LIGHTX2V_REPOSITORY, H3_TURBO_LORA_LIGHTX2V_SOURCE_REVISION,
    H3_TURBO_LORA_MODULE_COUNT, H3_TURBO_LORA_PAYLOAD_BYTES, H3_TURBO_LORA_REPOSITORY,
    H3_TURBO_LORA_SOURCE_FORMAT, H3_TURBO_LORA_SOURCE_REVISION, H3_TURBO_LORA_TARGET_FORMAT,
    H3_TURBO_LORA_TENSOR_COUNT, H3_TURBO_LORA_TRAINING_RANK, H3_TURBO_LORA_WEIGHT_DTYPE,
};
pub use turbo_runtime::{H3TurboBlockDeltas, H3TurboLoraDelta, H3TurboLoraRuntime};
pub use visual_condition::{
    ConditionEncodeMode, SignedRgbPixels, Uint8RgbPixels, UnitRgbPixels, H3_IMAGENET_MEAN,
    H3_IMAGENET_STD, H3_LATENTS_MEAN, H3_LATENTS_STD,
};
pub use visual_geometry::{
    EndpointResizePlan, ReferenceImageResizePlan, SpatialTilePlan, TemporalDecodeChunk,
    TemporalDecodePlan, TemporalEncodeChunk, TemporalEncodePlan, VisualTemporalGeometry,
};
#[cfg(feature = "h3-private-uat")]
pub use visual_vae::VisualVaeCaptureEvidence;
pub use visual_vae::{
    DecodeComputePolicy, DecodeSink, MiniMaxH3VisualVae, MiniMaxH3VisualVaeConfig,
    NoopVisualVaeObserver, VisualAttentionBackend, VisualVaeEvent, VisualVaeObserver,
    VisualVaePhase, WeightLayout,
};
pub use visual_weights::{
    comfy_tensor_transforms, component_fingerprint, component_fingerprint_with_observer,
    expected_comfy_weight_shapes, expected_diffusers_weight_shapes,
    expected_visual_vae_parameter_bytes, inspect_safetensors_header, inspect_visual_vae_config,
    inspect_visual_vae_config_bytes, validate_comfy_weight_file,
    validate_comfy_weight_file_from_authenticated_staging_descriptor,
    validate_comfy_weight_file_from_opened_descriptor, validate_diffusers_weight_index,
    ComfyTensorTransform, DiffusersWeightIndex, NoopVisualWeightReadObserver, SafetensorsHeader,
    SafetensorsTensorHeader, ValidatedVisualVaeWeights, VisualVaeComponentRole,
    VisualVaeWeightInspection, VisualWeightReadObserver, COMFY_VISUAL_VAE_FILENAME,
};
