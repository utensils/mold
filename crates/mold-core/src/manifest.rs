use crate::config::ModelConfig;
use crate::ModelPaths;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelComponent {
    Transformer,
    Vae,
    T5Encoder,
    ClipEncoder,
    T5Tokenizer,
    ClipTokenizer,
}

#[derive(Debug, Clone)]
pub struct ModelFile {
    pub hf_repo: String,
    pub hf_filename: String,
    pub component: ModelComponent,
    pub size_bytes: u64,
    pub gated: bool,
}

#[derive(Debug, Clone)]
pub struct ManifestDefaults {
    pub steps: u32,
    pub guidance: f64,
    pub width: u32,
    pub height: u32,
    pub is_schnell: bool,
}

#[derive(Debug, Clone)]
pub struct ModelManifest {
    pub name: String,
    pub family: String,
    pub description: String,
    pub size_gb: f32,
    pub files: Vec<ModelFile>,
    pub defaults: ManifestDefaults,
}

impl ModelManifest {
    /// Convert downloaded paths into a `ModelConfig` suitable for saving to config.toml.
    pub fn to_model_config(&self, paths: &ModelPaths) -> ModelConfig {
        ModelConfig {
            transformer: Some(paths.transformer.to_string_lossy().to_string()),
            vae: Some(paths.vae.to_string_lossy().to_string()),
            t5_encoder: Some(paths.t5_encoder.to_string_lossy().to_string()),
            clip_encoder: Some(paths.clip_encoder.to_string_lossy().to_string()),
            t5_tokenizer: Some(paths.t5_tokenizer.to_string_lossy().to_string()),
            clip_tokenizer: Some(paths.clip_tokenizer.to_string_lossy().to_string()),
            default_steps: Some(self.defaults.steps),
            default_guidance: Some(self.defaults.guidance),
            default_width: Some(self.defaults.width),
            default_height: Some(self.defaults.height),
            is_schnell: Some(self.defaults.is_schnell),
            description: Some(self.description.clone()),
            family: Some(self.family.clone()),
        }
    }
}

/// Shared FLUX component files (VAE, T5, CLIP, tokenizers) — identical across all FLUX models.
fn shared_flux_files() -> Vec<ModelFile> {
    vec![
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.1-schnell".to_string(),
            hf_filename: "ae.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 335_000_000, // ~335MB
            gated: false,
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "t5xxl_fp16.safetensors".to_string(),
            component: ModelComponent::T5Encoder,
            size_bytes: 9_200_000_000, // ~9.2GB
            gated: false,
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "clip_l.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 246_000_000, // ~246MB
            gated: false,
        },
        ModelFile {
            hf_repo: "lmz/mt5-tokenizers".to_string(),
            hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
            component: ModelComponent::T5Tokenizer,
            size_bytes: 2_400_000, // ~2.4MB
            gated: false,
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 600_000, // ~600KB
            gated: false,
        },
    ]
}

/// All known downloadable model manifests.
pub fn known_manifests() -> Vec<ModelManifest> {
    vec![
        ModelManifest {
            name: "flux-schnell:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q8 — fast 4-step, general purpose".to_string(),
            size_gb: 12.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_000_000_000, // ~12GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: true,
            },
        },
        ModelManifest {
            name: "flux-dev:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q8 — full quality, 20+ steps".to_string(),
            size_gb: 12.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_000_000_000, // ~12GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
            },
        },
        ModelManifest {
            name: "flux-dev:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q4 — smaller/faster, good quality".to_string(),
            size_gb: 7.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_000_000_000, // ~7GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
            },
        },
        ModelManifest {
            name: "flux-dev:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q6 — best quality/size trade-off".to_string(),
            size_gb: 9.9,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_860_000_000, // ~9.86GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
            },
        },
        ModelManifest {
            name: "flux-schnell:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q4 — fast 4-step, smaller footprint".to_string(),
            size_gb: 7.5,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_510_000_000, // ~7.51GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: true,
            },
        },
        ModelManifest {
            name: "flux-schnell:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q6 — fast 4-step, best quality/size trade-off".to_string(),
            size_gb: 9.8,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_830_000_000, // ~9.83GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: true,
            },
        },
        ModelManifest {
            name: "flux-krea:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q8 — aesthetic photography fine-tune".to_string(),
            size_gb: 12.7,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_700_000_000, // ~12.7GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
            },
        },
        ModelManifest {
            name: "flux-krea:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q4 — aesthetic photography, smaller footprint"
                .to_string(),
            size_gb: 7.5,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_530_000_000, // ~7.53GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
            },
        },
        ModelManifest {
            name: "flux-krea:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q6 — aesthetic photography, best quality/size trade-off"
                .to_string(),
            size_gb: 9.9,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_860_000_000, // ~9.86GB
                    gated: false,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
            },
        },
    ]
}

/// Resolve a user-provided model name to its canonical `name:tag` form.
///
/// - `flux-schnell` → `flux-schnell:q8`
/// - `flux-dev:q4` → `flux-dev:q4` (unchanged)
/// - `flux-dev-q4` → `flux-dev:q4` (legacy format)
pub fn resolve_model_name(input: &str) -> String {
    // Already has a tag
    if input.contains(':') {
        return input.to_string();
    }
    // Legacy format: flux-dev-q4 -> flux-dev:q4
    if let Some((base, suffix)) = input.rsplit_once('-') {
        if suffix.starts_with('q')
            && suffix.len() <= 3
            && suffix[1..].chars().all(|c| c.is_ascii_digit())
        {
            return format!("{base}:{suffix}");
        }
    }
    // Default tag
    format!("{input}:q8")
}

/// Find a manifest by name, handling tag resolution and legacy names.
pub fn find_manifest(name: &str) -> Option<ModelManifest> {
    let canonical = resolve_model_name(name);
    known_manifests().into_iter().find(|m| m.name == canonical)
}

/// Check if a name resolves to a known model (manifest or config).
pub fn is_known_model(name: &str, config: &crate::Config) -> bool {
    let canonical = resolve_model_name(name);
    config.models.contains_key(name)
        || config.models.contains_key(&canonical)
        || find_manifest(&canonical).is_some()
}

/// All known model names (manifests + config), deduplicated and sorted.
pub fn all_model_names(config: &crate::Config) -> Vec<String> {
    let mut names: Vec<String> = known_manifests().iter().map(|m| m.name.clone()).collect();
    for key in config.models.keys() {
        if !names.contains(key) {
            names.push(key.clone());
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Size of shared FLUX components (VAE, T5, CLIP, tokenizers) in GB.
pub const SHARED_COMPONENTS_GB: f32 = 9.8;

/// Total size of all files in the manifest in bytes.
pub fn total_download_size(manifest: &ModelManifest) -> u64 {
    manifest.files.iter().map(|f| f.size_bytes).sum()
}

/// Convert a `ModelManifest` to a `ModelPaths` from resolved download paths.
pub fn paths_from_downloads(downloads: &[(ModelComponent, PathBuf)]) -> Option<ModelPaths> {
    let find = |c: ModelComponent| -> Option<PathBuf> {
        downloads
            .iter()
            .find(|(comp, _)| *comp == c)
            .map(|(_, p)| p.clone())
    };

    Some(ModelPaths {
        transformer: find(ModelComponent::Transformer)?,
        vae: find(ModelComponent::Vae)?,
        t5_encoder: find(ModelComponent::T5Encoder)?,
        clip_encoder: find(ModelComponent::ClipEncoder)?,
        t5_tokenizer: find(ModelComponent::T5Tokenizer)?,
        clip_tokenizer: find(ModelComponent::ClipTokenizer)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_name_with_tag() {
        assert_eq!(resolve_model_name("flux-dev:q4"), "flux-dev:q4");
        assert_eq!(resolve_model_name("flux-schnell:q8"), "flux-schnell:q8");
    }

    #[test]
    fn resolve_name_default_tag() {
        assert_eq!(resolve_model_name("flux-schnell"), "flux-schnell:q8");
        assert_eq!(resolve_model_name("flux-dev"), "flux-dev:q8");
    }

    #[test]
    fn resolve_name_legacy_format() {
        assert_eq!(resolve_model_name("flux-dev-q4"), "flux-dev:q4");
        assert_eq!(resolve_model_name("flux-dev-q8"), "flux-dev:q8");
    }

    #[test]
    fn find_known_manifests() {
        assert!(find_manifest("flux-schnell").is_some());
        assert!(find_manifest("flux-dev:q4").is_some());
        assert!(find_manifest("flux-dev-q4").is_some());
        assert!(find_manifest("flux-krea").is_some());
        assert!(find_manifest("flux-schnell:q6").is_some());
        assert!(find_manifest("flux-dev:q6").is_some());
        assert!(find_manifest("flux-krea:q4").is_some());
        assert!(find_manifest("flux-krea:q6").is_some());
        assert!(find_manifest("nonexistent").is_none());
    }

    #[test]
    fn flux_krea_resolves_to_q8() {
        let manifest = find_manifest("flux-krea").unwrap();
        assert_eq!(manifest.name, "flux-krea:q8");
        assert!(!manifest.defaults.is_schnell);
        assert_eq!(manifest.defaults.steps, 25);
    }

    #[test]
    fn flux_krea_legacy_format() {
        assert_eq!(resolve_model_name("flux-krea-q4"), "flux-krea:q4");
        assert_eq!(resolve_model_name("flux-krea-q6"), "flux-krea:q6");
    }

    #[test]
    fn known_manifests_count() {
        assert_eq!(known_manifests().len(), 9);
    }

    #[test]
    fn manifest_has_all_components() {
        for manifest in known_manifests() {
            let components: Vec<_> = manifest.files.iter().map(|f| f.component).collect();
            assert!(components.contains(&ModelComponent::Transformer));
            assert!(components.contains(&ModelComponent::Vae));
            assert!(components.contains(&ModelComponent::T5Encoder));
            assert!(components.contains(&ModelComponent::ClipEncoder));
            assert!(components.contains(&ModelComponent::T5Tokenizer));
            assert!(components.contains(&ModelComponent::ClipTokenizer));
        }
    }

    #[test]
    fn shared_files_are_not_gated() {
        for file in shared_flux_files() {
            assert!(!file.gated, "{} should not be gated", file.hf_filename);
        }
    }
}
