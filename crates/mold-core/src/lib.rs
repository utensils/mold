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
pub mod error;
pub mod expand;
pub mod expand_prompts;
pub mod format;
pub mod install_error;
pub mod lambda;
pub mod ltx2_camera;
pub use ltx2_camera::{Ltx2CameraControlAvailability, Ltx2CameraControlInfo};
pub mod ltx2_control;
pub use ltx2_control::Ltx2ControlAdapterInfo;
pub mod manifest;
pub mod media_paths;
pub mod removal;
pub mod runpod;
pub mod safetensors_probe;
pub mod time;
pub mod types;
pub mod validation;

#[cfg(test)]
mod config_test;
#[cfg(test)]
mod test_support;

pub use catalog::build_model_catalog;
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
pub use install_error::InstallError;
pub use media_paths::{configured_media_roots, parse_media_roots_env, resolve_server_media_path};
pub use types::GenerateRequest;
pub use types::Scheduler;
pub use types::*;
pub use validation::{
    clamp_to_megapixel_limit, dimension_alignment_for_family, dimension_warning,
    family_supports_lora, fit_to_model_dimensions, fit_to_target_area, prompt_required_for,
    prompt_required_with_conditioning, recommended_dimensions, validate_generate_request,
    validate_generate_request_with_family, validate_generation_dimensions,
    validate_upscale_request, LORA_CAPABLE_FAMILIES,
};

pub use expand::{
    ApiExpander, ExpandConfig, ExpandResult, ExpandSettings, FamilyOverride, PromptExpander,
};
pub use expand_prompts::{build_batch_messages, build_single_messages, format_chatml};
