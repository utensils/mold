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
        let downloaded = config.manifest_model_is_downloaded(&manifest.name);
        let (_, remaining_download_bytes) = crate::manifest::compute_download_size(manifest);
        let disk_usage_bytes = downloaded.then(|| {
            let (bytes, _gb) = model_cfg.disk_usage();
            bytes
        });

        models.push(ModelInfoExtended {
            downloaded,
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
            disk_usage_bytes,
            remaining_download_bytes: Some(remaining_download_bytes),
        });
    }

    let mut config_only: Vec<_> = config
        .models
        .iter()
        .filter(|(name, _)| crate::manifest::find_manifest(name).is_none())
        .collect();
    config_only.sort_by(|(left, _), (right, _)| left.cmp(right));

    for (name, model_cfg) in config_only {
        let (disk_usage_bytes, size_gb_f64) = model_cfg.disk_usage();
        let size_gb = size_gb_f64 as f32;

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
            disk_usage_bytes: Some(disk_usage_bytes),
            remaining_download_bytes: None,
        });
    }

    models
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{find_manifest, storage_path};
    use crate::test_support::ENV_LOCK;
    use crate::ModelConfig;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn test_models_dir(name: &str) -> PathBuf {
        let unique = format!(
            "mold-catalog-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    fn populate_manifest_files(root: &std::path::Path, model: &str) {
        let manifest = find_manifest(model).unwrap();
        for file in &manifest.files {
            let path = root.join(storage_path(manifest, file));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, b"test").unwrap();
        }
    }

    #[test]
    fn build_model_catalog_marks_downloaded_manifest_models() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("downloaded");
        populate_manifest_files(&models_dir, "flux-schnell:q8");
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let config = Config {
            ..Config::default()
        };

        let entry = build_model_catalog(&config, Some("flux-schnell:q8"), true)
            .into_iter()
            .find(|model| model.name == "flux-schnell:q8")
            .expect("manifest model should exist");

        assert!(entry.downloaded);
        assert!(entry.is_loaded);
        assert_eq!(entry.defaults.default_steps, 4);

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
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

    #[test]
    fn build_model_catalog_marks_manifest_models_available_when_override_dir_is_empty() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models_dir = test_models_dir("empty");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::env::set_var("MOLD_MODELS_DIR", &models_dir);

        let entry = build_model_catalog(&Config::default(), None, false)
            .into_iter()
            .find(|model| model.name == "flux-schnell:q8")
            .expect("manifest model should exist");

        assert!(!entry.downloaded);
        assert!(entry.remaining_download_bytes.is_some());

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(models_dir);
    }
}
