pub mod catalog;
pub mod client;
pub mod config;
pub mod control;
pub mod download;
pub mod error;
pub mod manifest;
pub mod types;
pub mod validation;

#[cfg(test)]
mod config_test;
#[cfg(test)]
mod test_support;

pub use catalog::build_model_catalog;
pub use client::MoldClient;
pub use config::{Config, ModelConfig, ModelPaths};
pub use control::{
    classify_generate_error, classify_server_error, GenerateServerAction, ServerAvailability,
};
pub use error::{MoldError, Result as MoldResult};
pub use types::GenerateRequest;
pub use types::Scheduler;
pub use types::*;
pub use validation::{clamp_to_megapixel_limit, validate_generate_request};
