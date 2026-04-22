pub mod build_info;
pub mod catalog;
pub mod chain;
pub mod chain_toml;
pub mod client;
pub mod config;
pub mod control;
pub mod download;
pub mod error;
pub mod expand;
pub mod expand_prompts;
pub mod manifest;
pub mod runpod;
pub mod types;
pub mod validation;

#[cfg(test)]
mod config_test;
#[cfg(test)]
mod test_support;

pub use catalog::build_model_catalog;
pub use chain::{
    ChainProgressEvent, ChainRequest, ChainResponse, ChainStage, SseChainCompleteEvent,
    MAX_CHAIN_STAGES,
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
pub use types::GenerateRequest;
pub use types::Scheduler;
pub use types::*;
pub use validation::{
    clamp_to_megapixel_limit, dimension_warning, fit_to_model_dimensions, fit_to_target_area,
    recommended_dimensions, validate_generate_request, validate_upscale_request,
};

pub use expand::{
    ApiExpander, ExpandConfig, ExpandResult, ExpandSettings, FamilyOverride, PromptExpander,
};
pub use expand_prompts::{build_batch_messages, build_single_messages, format_chatml};
