use anyhow::{bail, Result};
use mold_core::{Config, ModelPaths};

use crate::engine::{InferenceEngine, LoadStrategy};
use crate::flux::FluxEngine;
use crate::sd15::SD15Engine;
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
/// Returns the appropriate engine (FluxEngine, SD15Engine, SDXLEngine, or ZImageEngine)
/// based on the model's family from config or manifest.
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
        "sd15" | "sd1.5" | "stable-diffusion-1.5" => {
            let scheduler_name = model_cfg.scheduler.unwrap_or_else(|| "ddim".to_string());
            Ok(Box::new(SD15Engine::new(
                model_name,
                paths,
                scheduler_name,
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
            "unknown model family '{}' for model '{}'. Supported: flux, sd15, sdxl, z-image",
            other,
            model_name
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dummy_paths() -> ModelPaths {
        ModelPaths {
            transformer: PathBuf::from("/tmp/transformer"),
            transformer_shards: vec![],
            vae: PathBuf::from("/tmp/vae"),
            t5_encoder: Some(PathBuf::from("/tmp/t5")),
            clip_encoder: Some(PathBuf::from("/tmp/clip")),
            t5_tokenizer: Some(PathBuf::from("/tmp/t5_tok")),
            clip_tokenizer: Some(PathBuf::from("/tmp/clip_tok")),
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
        }
    }

    #[test]
    fn resolve_family_from_manifest_sd15() {
        let config = Config::default();
        assert_eq!(resolve_family("sd15:fp16", &config), "sd15");
        assert_eq!(resolve_family("dreamshaper-v8:fp16", &config), "sd15");
        assert_eq!(resolve_family("realistic-vision-v5:fp16", &config), "sd15");
    }

    #[test]
    fn resolve_family_from_manifest_sdxl() {
        let config = Config::default();
        assert_eq!(resolve_family("sdxl-base:fp16", &config), "sdxl");
    }

    #[test]
    fn resolve_family_unknown_defaults_to_flux() {
        let config = Config::default();
        assert_eq!(resolve_family("totally-unknown-model", &config), "flux");
    }

    #[test]
    fn resolve_family_config_overrides_manifest() {
        let mut config = Config::default();
        config.models.insert(
            "custom-model".to_string(),
            mold_core::config::ModelConfig {
                family: Some("sd15".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(resolve_family("custom-model", &config), "sd15");
    }

    #[test]
    fn create_engine_sd15() {
        let config = Config::default();
        let engine = create_engine(
            "sd15:fp16".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "sd15:fp16");
    }

    #[test]
    fn create_engine_sd15_family_alias() {
        // Config-based family alias "sd1.5" should work
        let mut config = Config::default();
        config.models.insert(
            "my-model".to_string(),
            mold_core::config::ModelConfig {
                family: Some("sd1.5".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-model".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-model");
    }

    #[test]
    fn create_engine_unknown_family_fails() {
        let mut config = Config::default();
        config.models.insert(
            "bad-model".to_string(),
            mold_core::config::ModelConfig {
                family: Some("nosuchfamily".to_string()),
                ..Default::default()
            },
        );
        let result = create_engine(
            "bad-model".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
        );
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("nosuchfamily"));
    }
}
