use anyhow::{bail, Result};
use mold_core::{Config, ModelPaths};

use crate::engine::{InferenceEngine, LoadStrategy};
use crate::flux::FluxEngine;
use crate::sdxl::SDXLEngine;
use crate::zimage::ZImageEngine;

/// Determine the model family from config or manifest, defaulting to "flux".
fn resolve_family(model_name: &str, config: &Config) -> String {
    // Check config first
    let model_cfg = config.model_config(model_name);
    if let Some(family) = model_cfg.family {
        return family;
    }
    // Check manifest
    if let Some(manifest) = mold_core::manifest::find_manifest(model_name) {
        return manifest.family;
    }
    // Default to flux for backward compatibility
    "flux".to_string()
}

/// Create an inference engine for the given model, auto-detecting the family.
///
/// Returns the appropriate engine (FluxEngine, SDXLEngine, or ZImageEngine) based on
/// the model's family from config or manifest.
///
/// `load_strategy` controls whether components are loaded eagerly (server) or
/// sequentially (CLI one-shot) to reduce peak memory.
pub fn create_engine(
    model_name: String,
    paths: ModelPaths,
    config: &Config,
    load_strategy: LoadStrategy,
) -> Result<Box<dyn InferenceEngine>> {
    let family = resolve_family(&model_name, config);
    let model_cfg = config.model_config(&model_name);

    match family.as_str() {
        "flux" => {
            let is_schnell = model_cfg.is_schnell;
            let t5_variant = std::env::var("MOLD_T5_VARIANT")
                .ok()
                .or_else(|| config.t5_variant.clone());
            Ok(Box::new(FluxEngine::new(
                model_name,
                paths,
                is_schnell,
                t5_variant,
                load_strategy,
            )))
        }
        "sdxl" => {
            let scheduler_name = model_cfg.scheduler.unwrap_or_else(|| "ddim".to_string());
            let is_turbo = scheduler_name == "euler_ancestral" || model_name.contains("turbo");
            Ok(Box::new(SDXLEngine::new(
                model_name,
                paths,
                scheduler_name,
                is_turbo,
                load_strategy,
            )))
        }
        "z-image" => {
            let qwen3_variant = std::env::var("MOLD_QWEN3_VARIANT")
                .ok()
                .or_else(|| config.qwen3_variant.clone());
            Ok(Box::new(ZImageEngine::new(
                model_name,
                paths,
                qwen3_variant,
                load_strategy,
            )))
        }
        other => bail!(
            "unknown model family '{}' for model '{}'. Supported: flux, sdxl, z-image",
            other,
            model_name
        ),
    }
}
