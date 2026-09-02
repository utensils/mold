pub mod build_info;
pub mod catalog;
pub mod catalog_wire;
pub mod chain;
pub mod chain_job;
pub mod chain_toml;
pub mod client;
pub mod config;
pub mod config_keys;
pub mod control;
pub mod cuda_distribution;
pub mod download;
pub mod durable_generation;
pub mod error;
pub mod expand;
pub mod expand_prompts;
pub mod format;
pub mod generation_profile;
pub mod gguf_probe;
pub mod glb_summary;
pub mod identity;
pub mod install_error;
pub mod lambda;
pub mod license_acceptance;
pub mod ltx2_camera;
pub use ltx2_camera::{Ltx2CameraControlAvailability, Ltx2CameraControlInfo};
pub mod ltx2_control;
pub mod ltx2_duration;
pub mod ltx2_preprocess;
pub mod ltx2_weight_index;
pub use ltx2_control::Ltx2ControlAdapterInfo;
pub mod ltx25_manifest;
pub mod ltx25_probe;
pub mod manifest;
pub mod media_paths;
pub mod minimax_h3;
pub mod model_policy;
pub mod organization;
pub mod print_title;
pub mod prompt_text;
pub mod prompting;
pub mod pulid_assets;
pub mod queue_progress;
pub mod queue_wait;
pub mod reference_upload;
pub mod removal;
pub mod request_media;
pub mod runpod;
pub mod safetensors_probe;
pub mod secure_file;
pub mod time;
pub mod types;
pub mod validation;
pub mod video_upscale;
pub mod wan_expert_marker;

#[cfg(test)]
mod config_test;
#[cfg(test)]
mod test_support;

pub use catalog::{build_model_catalog, qualify_catalog_generation_delivery};
pub use chain::{
    ChainFailure, ChainProgressEvent, ChainRequest, ChainResponse, ChainScript, ChainScriptChain,
    ChainStage, ChainValidationResponse, ChainValidationStage, LoraSpec, NamedRef,
    SseChainCompleteEvent, TransitionMode, VramEstimate, MAX_CHAIN_STAGES,
};
pub use client::MoldClient;
pub use config::{
    parse_device_ref_str, Config, DefaultModelResolution, DefaultModelSource, LoggingConfig,
    ModelConfig, ModelPaths,
};
pub use control::{
    classify_generate_error, classify_server_error, GenerateServerAction, ServerAvailability,
};
pub use error::{MoldError, Result as MoldResult};
pub use generation_profile::{
    generation_profile_default_output_format, generation_profile_for_manifest,
    generation_profile_for_manifest_with_defaults, materialize_generation_profile_output_default,
    off_bucket_resolution_warning, prompt_requirement_for_family,
    qualify_generation_profile_delivery, resolution_advisory, resolve_generation_profile,
    validate_dimensions_against_recipe, validate_mesh_against_recipe,
    validate_output_format_against_generation_profile, validate_request_against_generation_profile,
    validate_request_against_recipe, AspectGroup, ControlMode, FloatControl, FpsControl,
    GenerationCapabilitiesProfile, GenerationDefaultsProfile, GenerationDeliveryCapabilities,
    GenerationProfileInput, GenerationProfileSet, GenerationRecipeProfile, IntegerControl,
    MeshCapabilitiesProfile, ProfileProvenance, PromptCapabilitiesProfile, PromptRequirement,
    ProvenanceKind, RecipeSelector, ResolutionDomain, ResolutionPreset, ResolutionProfile,
    TemporalProfile, GENERATION_PROFILE_SCHEMA_VERSION,
};
pub use install_error::InstallError;
pub use media_paths::{configured_media_roots, parse_media_roots_env, resolve_server_media_path};
pub use model_policy::{
    is_exact_registered_manifest, is_pinned_unrunnable_minimax_h3_identity,
    model_access_capabilities, model_acquisition, model_acquisition_available, model_activation,
    model_activation_available, model_artifact_activation, require_model_acquisition,
    require_model_activation, require_model_artifact_activation,
    require_registered_manifest_activation, ActivationRefusal, ModelAccessCapabilities,
    ModelAccessRestriction, ModelActivation, ModelActivationError,
    MINIMAX_H3_AUTHORIZATION_ISSUE_URL, MINIMAX_H3_AUTHORIZATION_REQUIRED,
    MINIMAX_H3_LICENSE_SHA256, MINIMAX_H3_LICENSE_URL, MINIMAX_H3_RUNTIME_UNAVAILABLE,
};
pub use organization::{
    collection_slug, compose_client_tags, normalize_request_tags, normalize_tag_name,
    validate_collection_name, ComposedClientTags, MAX_COLLECTION_NAME_CHARS,
    MAX_COLLECTION_SLUG_CHARS, MAX_REQUEST_TAGS, MAX_TAG_CHARS,
};
pub use print_title::{
    default_output_filename_titled, download_file_name, strip_title_slug, title_slug,
    validate_print_title, DOWNLOAD_FALLBACK_STEM, DOWNLOAD_MODEL_SLUG_MAX_LEN,
    DOWNLOAD_NAME_SEPARATOR, PRINT_TITLE_MAX_CHARS, TITLE_SLUG_MAX_LEN, TITLE_SLUG_SEPARATOR,
};
pub use prompt_text::normalize_prompt_newlines;
pub use reference_upload::{ReferenceUploadLease, ReferenceUploadSource};
pub use types::GenerateRequest;
pub use types::Scheduler;
pub use types::*;
pub use validation::{
    clamp_to_megapixel_limit, dimension_alignment_for_family, dimension_alignment_for_model,
    dimension_warning, dimension_warning_composed, family_supports_lora, fit_to_model_dimensions,
    fit_to_model_dimensions_aligned, fit_to_target_area, fixed_fps_for_family,
    frame_offset_for_family, largest_ltx2_rung_within, ltx2_output_rung, ltx2_spatial_composition,
    materialize_request_organization, min_frames_for_family, prompt_required_for,
    prompt_required_with_conditioning, recommended_dimensions, recommended_dimensions_composed,
    require_generate_request_model_activation, validate_generate_request,
    validate_generate_request_fields, validate_generate_request_with_family,
    validate_generation_dimensions, validate_generation_dimensions_composed,
    validate_generation_dimensions_for_model, validate_request_organization,
    validate_resolved_generate_request_with_family, validate_upscale_request,
    wan_dimension_alignment, Ltx2OutputRung, Ltx2SpatialComposition, ReferenceForm,
    RequestOrganization, LORA_CAPABLE_FAMILIES, LTX2_OUTPUT_RUNGS,
};
pub use video_upscale::*;

pub use expand::{
    ApiExpander, ExpandConfig, ExpandResult, ExpandSettings, FamilyOverride, PromptExpander,
};
pub use expand_prompts::{build_batch_messages, build_single_messages, format_chatml};
