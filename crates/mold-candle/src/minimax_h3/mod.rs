//! Reusable MiniMax H3 model primitives.
//!
//! Nothing in this module registers a runnable inference engine. The H3
//! license gate and the family capability contract remain the authority for
//! whether callers may construct these primitives with real weights.

mod artifacts;
mod config;
mod loader;
mod model;
mod presentation;
mod processor;
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
pub use config::{
    H3ConditionerConfig, H3TextConfig, H3VisionConfig, H3_BF16_PARAMETER_BYTES,
    H3_COMFY_SAFETENSORS_BYTES, H3_FULL_CHECKPOINT_BYTES, H3_FULL_LANGUAGE_LAYERS,
    H3_SELECTED_LANGUAGE_LAYERS,
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
    H3Presentation, RefPresentation, RefPresentationKind,
};
pub use processor::{
    create_mm_token_type_ids, pack_qwen_vision_u8, qwen_mrope_positions, sample_video_frames,
    GridThw, PackedVisionPatches, QwenMmTokenType, SampledVideo,
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
