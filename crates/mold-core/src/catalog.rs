use crate::manifest::known_manifests;
use crate::{Config, ModelDefaults, ModelInfo, ModelInfoExtended};

/// Build the user-facing model catalog from the manifest registry plus local config.
pub fn build_model_catalog(
    config: &Config,
    loaded_model: Option<&str>,
    engine_is_loaded: bool,
) -> Vec<ModelInfoExtended> {
    let mut models = Vec::with_capacity(known_manifests().len() + config.models.len());

    for manifest in known_manifests() {
        let model_cfg = config.resolved_model_config(&manifest.name);

        models.push(ModelInfoExtended {
            downloaded: config.models.contains_key(&manifest.name),
            defaults: ModelDefaults {
                default_steps: model_cfg.effective_steps(config),
                default_guidance: model_cfg.effective_guidance(),
                default_width: model_cfg.effective_width(config),
                default_height: model_cfg.effective_height(config),
                description: model_cfg
                    .description
                    .unwrap_or_else(|| manifest.name.clone()),
            },
            info: ModelInfo {
                name: manifest.name.clone(),
                family: manifest.family.clone(),
                size_gb: manifest.model_size_gb(),
                is_loaded: loaded_model
                    .is_some_and(|name| engine_is_loaded && name == manifest.name),
                last_used: None,
                hf_repo: manifest
                    .files
                    .iter()
                    .find(|f| f.component == crate::manifest::ModelComponent::Transformer)
                    .map(|f| f.hf_repo.clone())
                    .unwrap_or_default(),
            },
        });
    }

    let mut config_only: Vec<_> = config
        .models
        .iter()
        .filter(|(name, _)| crate::manifest::find_manifest(name).is_none())
        .collect();
    config_only.sort_by(|(left, _), (right, _)| left.cmp(right));

    for (name, model_cfg) in config_only {
        let size_gb = model_cfg
            .all_file_paths()
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .map(|meta| meta.len() as f32 / 1_073_741_824.0)
            .sum::<f32>();

        models.push(ModelInfoExtended {
            downloaded: true,
            defaults: ModelDefaults {
                default_steps: model_cfg.effective_steps(config),
                default_guidance: model_cfg.effective_guidance(),
                default_width: model_cfg.effective_width(config),
                default_height: model_cfg.effective_height(config),
                description: model_cfg
                    .description
                    .clone()
                    .unwrap_or_else(|| name.clone()),
            },
            info: ModelInfo {
                name: name.clone(),
                family: model_cfg
                    .family
                    .clone()
                    .unwrap_or_else(|| "flux".to_string()),
                size_gb,
                is_loaded: loaded_model.is_some_and(|loaded| engine_is_loaded && loaded == name),
                last_used: None,
                hf_repo: String::new(),
            },
        });
    }

    models
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;
    use std::collections::HashMap;

    #[test]
    fn build_model_catalog_marks_downloaded_manifest_models() {
        let mut models = HashMap::new();
        models.insert(
            "flux-schnell:q8".to_string(),
            ModelConfig {
                transformer: Some("/tmp/transformer.gguf".to_string()),
                vae: Some("/tmp/ae.safetensors".to_string()),
                ..ModelConfig::default()
            },
        );
        let config = Config {
            models,
            ..Config::default()
        };

        let entry = build_model_catalog(&config, Some("flux-schnell:q8"), true)
            .into_iter()
            .find(|model| model.name == "flux-schnell:q8")
            .expect("manifest model should exist");

        assert!(entry.downloaded);
        assert!(entry.is_loaded);
        assert_eq!(entry.defaults.default_steps, 4);
    }

    #[test]
    fn build_model_catalog_keeps_config_only_models() {
        let mut models = HashMap::new();
        models.insert(
            "custom-model".to_string(),
            ModelConfig {
                family: Some("custom".to_string()),
                description: Some("Custom".to_string()),
                default_steps: Some(12),
                ..ModelConfig::default()
            },
        );
        let config = Config {
            models,
            ..Config::default()
        };

        let entry = build_model_catalog(&config, None, false)
            .into_iter()
            .find(|model| model.name == "custom-model")
            .expect("config-only model should exist");

        assert!(entry.downloaded);
        assert_eq!(entry.family, "custom");
        assert_eq!(entry.defaults.default_steps, 12);
    }
}
