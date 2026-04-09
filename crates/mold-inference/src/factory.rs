use anyhow::{bail, Result};
use mold_core::{Config, ModelPaths, Scheduler};
use std::sync::{Arc, Mutex};

use crate::engine::{InferenceEngine, LoadStrategy};
use crate::flux::FluxEngine;
use crate::flux2::Flux2Engine;
use crate::ltx2::Ltx2Engine;
use crate::ltx_video::LtxVideoEngine;
use crate::qwen_image::QwenImageEngine;
use crate::sd15::SD15Engine;
use crate::sd3::SD3Engine;
use crate::sdxl::SDXLEngine;
use crate::shared_pool::SharedPool;
use crate::wuerstchen::WuerstchenEngine;
use crate::zimage::ZImageEngine;

/// Determine the model family from config or manifest, defaulting to "flux".
fn resolve_family(model_name: &str, config: &Config) -> String {
    let model_cfg = config.resolved_model_config(model_name);
    if let Some(family) = model_cfg.family {
        return family;
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
    offload: bool,
) -> Result<Box<dyn InferenceEngine>> {
    create_engine_with_pool(model_name, paths, config, load_strategy, offload, None)
}

/// Create an inference engine with an optional shared tokenizer pool.
///
/// When `shared_pool` is provided, engines can cache and reuse tokenizers across
/// model switches (e.g. all FLUX variants share the same T5/CLIP tokenizers).
pub fn create_engine_with_pool(
    model_name: String,
    paths: ModelPaths,
    config: &Config,
    load_strategy: LoadStrategy,
    offload: bool,
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
) -> Result<Box<dyn InferenceEngine>> {
    let family = resolve_family(&model_name, config);
    let model_cfg = config.resolved_model_config(&model_name);

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
                offload,
                shared_pool,
            )))
        }
        "sd15" | "sd1.5" | "stable-diffusion-1.5" => {
            let scheduler = model_cfg.scheduler.unwrap_or(Scheduler::Ddim);
            Ok(Box::new(SD15Engine::new(
                model_name,
                paths,
                scheduler,
                load_strategy,
            )))
        }
        "sdxl" => {
            let is_turbo = model_cfg
                .is_turbo
                .unwrap_or_else(|| model_name.contains("turbo"));
            let scheduler = model_cfg.scheduler.unwrap_or(if is_turbo {
                Scheduler::EulerAncestral
            } else {
                Scheduler::Ddim
            });
            Ok(Box::new(SDXLEngine::new(
                model_name,
                paths,
                scheduler,
                is_turbo,
                load_strategy,
            )))
        }
        "sd3" | "sd3.5" | "stable-diffusion-3" | "stable-diffusion-3.5" => {
            let is_turbo = model_cfg
                .is_turbo
                .unwrap_or_else(|| model_name.contains("turbo"));
            let is_medium = model_name.contains("medium");
            let t5_variant = std::env::var("MOLD_T5_VARIANT")
                .ok()
                .or_else(|| config.t5_variant.clone());
            Ok(Box::new(SD3Engine::new(
                model_name,
                paths,
                is_turbo,
                is_medium,
                t5_variant,
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
        "flux2" | "flux.2" | "flux2-klein" => {
            let qwen3_variant = std::env::var("MOLD_QWEN3_VARIANT")
                .ok()
                .or_else(|| config.qwen3_variant.clone());
            Ok(Box::new(Flux2Engine::new(
                model_name,
                paths,
                qwen3_variant,
                load_strategy,
            )))
        }
        "qwen-image" | "qwen_image" => Ok(Box::new(QwenImageEngine::new(
            model_name,
            paths,
            load_strategy,
            offload,
        ))),
        "qwen-image-edit" => Ok(Box::new(QwenImageEngine::new(
            model_name,
            paths,
            load_strategy,
            offload,
        ))),
        "ltx-video" | "ltx_video" => {
            let t5_variant = std::env::var("MOLD_T5_VARIANT")
                .ok()
                .or_else(|| config.t5_variant.clone());
            Ok(Box::new(LtxVideoEngine::new(
                model_name,
                paths,
                t5_variant,
                load_strategy,
                shared_pool,
            )))
        }
        "ltx2" | "ltx-2" => Ok(Box::new(Ltx2Engine::new(model_name, paths, load_strategy))),
        "wuerstchen" | "wuerstchen-v2" => Ok(Box::new(WuerstchenEngine::new(
            model_name,
            paths,
            load_strategy,
        ))),
        other => bail!(
            "unknown model family '{}' for model '{}'. Supported: flux, flux2, sd15, sd3, sdxl, z-image, qwen-image, qwen-image-edit, wuerstchen",
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
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(PathBuf::from("/tmp/t5")),
            clip_encoder: Some(PathBuf::from("/tmp/clip")),
            t5_tokenizer: Some(PathBuf::from("/tmp/t5_tok")),
            clip_tokenizer: Some(PathBuf::from("/tmp/clip_tok")),
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
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
            false,
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
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-model");
    }

    #[test]
    fn resolve_family_from_manifest_sd3() {
        let config = Config::default();
        assert_eq!(resolve_family("sd3.5-large:q8", &config), "sd3");
        assert_eq!(resolve_family("sd3.5-medium:q8", &config), "sd3");
        assert_eq!(resolve_family("sd3.5-large-turbo:q8", &config), "sd3");
    }

    #[test]
    fn resolve_family_from_manifest_flux2() {
        let config = Config::default();
        assert_eq!(resolve_family("flux2-klein:bf16", &config), "flux2");
    }

    #[test]
    fn create_engine_flux2() {
        let config = Config::default();
        let engine = create_engine(
            "flux2-klein:bf16".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "flux2-klein:bf16");
    }

    #[test]
    fn create_engine_flux2_family_alias() {
        // Config-based family alias "flux.2" should work
        let mut config = Config::default();
        config.models.insert(
            "my-flux2".to_string(),
            mold_core::config::ModelConfig {
                family: Some("flux.2".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-flux2".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-flux2");
    }

    #[test]
    fn resolve_family_from_manifest_qwen_image() {
        let config = Config::default();
        assert_eq!(resolve_family("qwen-image:bf16", &config), "qwen-image");
    }

    #[test]
    fn resolve_family_from_manifest_qwen_image_edit() {
        let config = Config::default();
        assert_eq!(
            resolve_family("qwen-image-edit-2511:q4", &config),
            "qwen-image-edit"
        );
    }

    #[test]
    fn create_engine_qwen_image() {
        let mut config = Config::default();
        config.models.insert(
            "my-qwen-image".to_string(),
            mold_core::config::ModelConfig {
                family: Some("qwen-image".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-qwen-image".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-qwen-image");
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
            false,
        );
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("nosuchfamily"));
        assert!(err.contains("qwen-image-edit"));
    }

    #[test]
    fn create_engine_qwen_image_edit_routes_to_qwen_engine() {
        let result = create_engine(
            "qwen-image-edit-2511:q4".to_string(),
            dummy_paths(),
            &Config::default(),
            LoadStrategy::Sequential,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn resolve_family_from_manifest_wuerstchen() {
        let config = Config::default();
        assert_eq!(resolve_family("wuerstchen-v2:fp16", &config), "wuerstchen");
    }

    #[test]
    fn create_engine_wuerstchen() {
        let mut config = Config::default();
        config.models.insert(
            "my-wuerstchen".to_string(),
            mold_core::config::ModelConfig {
                family: Some("wuerstchen".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-wuerstchen".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-wuerstchen");
    }

    #[test]
    fn create_engine_wuerstchen_family_alias() {
        let mut config = Config::default();
        config.models.insert(
            "my-wurst".to_string(),
            mold_core::config::ModelConfig {
                family: Some("wuerstchen-v2".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-wurst".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-wurst");
    }
}
