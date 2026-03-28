pub mod build_info;
pub mod catalog;
pub mod client;
pub mod config;
pub mod control;
pub mod download;
pub mod error;
pub mod expand;
pub mod expand_prompts;
pub mod manifest;
pub mod types;
pub mod validation;

#[cfg(test)]
mod config_test;
#[cfg(test)]
mod test_support;

pub use catalog::build_model_catalog;
pub use client::MoldClient;
pub use config::{Config, DefaultModelResolution, DefaultModelSource, ModelConfig, ModelPaths};
pub use control::{
    classify_generate_error, classify_server_error, GenerateServerAction, ServerAvailability,
};
pub use error::{MoldError, Result as MoldResult};
pub use types::GenerateRequest;
pub use types::Scheduler;
pub use types::*;
pub use validation::{
    clamp_to_megapixel_limit, fit_to_model_dimensions, validate_generate_request,
};

pub use expand::{ApiExpander, ExpandConfig, ExpandResult, ExpandSettings, PromptExpander};
pub use expand_prompts::{build_batch_messages, build_single_messages, format_chatml};
