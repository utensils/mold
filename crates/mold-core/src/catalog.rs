use crate::manifest::{known_manifests, model_base_name, variant_quality_rank, visible_manifests};
use crate::{Config, ModelDefaults, ModelInfo, ModelInfoExtended, RecommendedDimensions};

/// The resolution contract `/api/models` advertises for one model.
pub struct ResolutionDefaults {
    pub max_pixels: Option<u64>,
    pub max_axis_pixels: Option<u32>,
    pub recommended_dimensions: Vec<RecommendedDimensions>,
    pub dimension_alignment: Option<u32>,
}

/// Build the advertised resolution contract for a specific model.
///
/// Per model, not per family: an LTX-2 checkpoint that ships the spatial
/// upsampler renders stage 1 halved and refines it over tiles, which is what
/// lets it hold an axis past the trained RoPE span. One that does not cannot,
/// and offering it the composed rungs would advertise a size it rejects.
pub fn resolution_defaults(model: &str, family: &str) -> ResolutionDefaults {
    // Route through the validator's own helpers rather than restating the
    // rules: `/api/models` is the single source clients read, so an
    // advertisement that disagrees with admission is a rejected request the
    // user was told would work.
    let composition = if family == "ltx2" {
        crate::validation::ltx2_spatial_composition(model, None)
    } else {
        crate::validation::Ltx2SpatialComposition::SinglePass
    };
    ResolutionDefaults {
        max_pixels: Some(crate::validation::max_pixels_for_family_composed(
            Some(family),
            composition,
        )),
        max_axis_pixels: crate::validation::max_axis_pixels_for_family_composed(
            Some(family),
            composition,
        ),
        recommended_dimensions: if family == "wan" {
            // Wan buckets are per checkpoint (480p-only 1.3B vs 704-grid
            // TI2V-5B); the family list would advertise sizes the selected
            // model does not support.
            crate::validation::wan_recommended_dimensions(model).to_vec()
        } else {
            crate::validation::recommended_dimensions_composed(family, composition)
        }
        .into_iter()
        .map(|(width, height)| RecommendedDimensions { width, height })
        .collect(),
        // Per model, like the buckets above: `wan22-ti2v-5b`'s 2.2 VAE needs
        // a 32px grid the rest of its family does not, and Studio source
        // fitting floors to exactly this advertised value.
        dimension_alignment: Some(crate::validation::dimension_alignment_for_model(
            model,
            Some(family),
        )),
    }
}

/// Build the user-facing model catalog from the manifest registry plus local config.
/// Hidden manifests are excluded from the catalog (CLI list, TUI model selector).
/// Whether a family's runtime can render sequence clips at all.
///
/// This is the same answer `mold-server`'s `sequence_support` gives; it lives
/// here too because `/api/models` must advertise it per model, and the picker
/// on every surface reads it instead of guessing from the checkpoint name.
fn chain_capable_family(family: &str) -> bool {
    matches!(family, "ltx2" | "ltx-video")
}

pub fn build_model_catalog(
    config: &Config,
    loaded_model: Option<&str>,
    engine_is_loaded: bool,
) -> Vec<ModelInfoExtended> {
    let mut models = Vec::with_capacity(known_manifests().len() + config.models.len());
    let artifact_root = config.resolved_models_dir();

    for manifest in visible_manifests().filter(|manifest| {
        crate::require_model_activation(&manifest.name, Some(&manifest.family)).is_ok()
            && manifest.files.iter().all(|file| {
                crate::require_model_activation(&file.hf_repo, Some(&manifest.family)).is_ok()
                    && crate::require_model_activation(&file.hf_filename, Some(&manifest.family))
                        .is_ok()
            })
    }) {
        let resolution = resolution_defaults(&manifest.name, &manifest.family);
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
                default_frames: model_cfg.effective_frames(),
                default_fps: model_cfg.effective_fps(),
                min_frames: crate::validation::min_frames_for_family(&manifest.family),
                max_frames: crate::validation::max_frames_for_family_at_fps(
                    &manifest.family,
                    model_cfg
                        .effective_fps()
                        .unwrap_or(crate::validation::LTX2_DEFAULT_FPS),
                ),
                max_runtime_seconds: crate::validation::max_runtime_seconds_for_family(
                    &manifest.family,
                ),
                max_frames_absolute: crate::validation::max_frames_absolute_for_family(
                    &manifest.family,
                ),
                frame_step: crate::validation::frame_step_for_family(&manifest.family),
                frame_offset: crate::validation::frame_offset_for_family(&manifest.family),
                // Strictly the engine-applied absence fallback (wan). NOT
                // `manifest.defaults.negative_prompt`: wuerstchen's manifest
                // carries a CLI-layer default the server never substitutes,
                // and advertising it would promise behavior an HTTP request
                // does not get.
                default_negative_prompt: crate::manifest::default_negative_prompt_for_family(
                    &manifest.family,
                )
                .map(str::to_string),
                max_pixels: resolution.max_pixels,
                max_axis_pixels: resolution.max_axis_pixels,
                recommended_dimensions: resolution.recommended_dimensions,
                dimension_alignment: resolution.dimension_alignment,
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
                // Sharded checkpoints (e.g. qwen-image:bf16) carry their
                // weights as TransformerShard files — without the fallback
                // they'd report no repo and lose their source mark and
                // model-page link in clients.
                hf_repo: manifest
                    .files
                    .iter()
                    .find(|f| f.component == crate::manifest::ModelComponent::Transformer)
                    .or_else(|| {
                        manifest.files.iter().find(|f| {
                            f.component == crate::manifest::ModelComponent::TransformerShard
                        })
                    })
                    .map(|f| f.hf_repo.clone())
                    .unwrap_or_default(),
            },
            disk_usage_bytes,
            remaining_download_bytes: Some(remaining_download_bytes),
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: Some(manifest.family == "ltx2"),
            supports_sequence: Some(chain_capable_family(&manifest.family)),
            extend_default_overlap_frames: Some(crate::validation::DEFAULT_EXTEND_OVERLAP_FRAMES),
            guidance_capabilities: Some(crate::GuidanceCapabilities::for_recipe(
                &manifest.family,
                &manifest.name,
                None,
            )),
            // Recorded where the manifest was built — the one place that
            // structurally knows the task (#772). Cold tiers advertise
            // correctly before any file exists.
            source_image: manifest.defaults.source_image,
        });
    }

    let mut config_only: Vec<_> = config
        .models
        .iter()
        .filter(|(name, model_cfg)| {
            crate::manifest::find_manifest(name).is_none()
                && crate::require_model_activation(name, model_cfg.family.as_deref()).is_ok()
                && model_cfg.all_file_paths().iter().all(|path| {
                    crate::require_model_artifact_activation(
                        std::path::Path::new(path),
                        Some(&artifact_root),
                        model_cfg.family.as_deref(),
                    )
                    .is_ok()
                })
        })
        .collect();
    config_only.sort_by_key(|(name, _)| *name);

    for (name, model_cfg) in config_only {
        let (disk_usage_bytes, size_gb_f64) = model_cfg.disk_usage();
        let size_gb = size_gb_f64 as f32;
        let family: String = model_cfg
            .family
            .clone()
            .unwrap_or_else(|| "flux".to_string());
        let resolution = resolution_defaults(name, &family);
        let is_ltx2 = family == "ltx2";
        let sequence_capable = chain_capable_family(&family);
        let guidance_identity = format!(
            "{} {}",
            name,
            model_cfg.description.as_deref().unwrap_or_default()
        );

        models.push(ModelInfoExtended {
            downloaded: true,
            defaults: ModelDefaults {
                default_steps: model_cfg.effective_steps(config),
                default_guidance: model_cfg.effective_guidance(),
                default_width: model_cfg.effective_width(config),
                default_height: model_cfg.effective_height(config),
                default_frames: model_cfg.effective_frames(),
                default_fps: model_cfg.effective_fps(),
                min_frames: crate::validation::min_frames_for_family(&family),
                max_frames: crate::validation::max_frames_for_family_at_fps(
                    &family,
                    model_cfg
                        .effective_fps()
                        .unwrap_or(crate::validation::LTX2_DEFAULT_FPS),
                ),
                max_runtime_seconds: crate::validation::max_runtime_seconds_for_family(&family),
                max_frames_absolute: crate::validation::max_frames_absolute_for_family(&family),
                frame_step: crate::validation::frame_step_for_family(&family),
                frame_offset: crate::validation::frame_offset_for_family(&family),
                default_negative_prompt: crate::manifest::default_negative_prompt_for_family(
                    &family,
                )
                .map(str::to_string),
                max_pixels: resolution.max_pixels,
                max_axis_pixels: resolution.max_axis_pixels,
                recommended_dimensions: resolution.recommended_dimensions,
                dimension_alignment: resolution.dimension_alignment,
                description: model_cfg
                    .description
                    .clone()
                    .unwrap_or_else(|| name.clone()),
            },
            info: ModelInfo {
                name: name.clone(),
                family: family.clone(),
                size_gb,
                is_loaded: loaded_model.is_some_and(|loaded| engine_is_loaded && loaded == name),
                last_used: None,
                hf_repo: String::new(),
            },
            disk_usage_bytes: Some(disk_usage_bytes),
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: Some(is_ltx2),
            supports_sequence: Some(sequence_capable),
            extend_default_overlap_frames: Some(crate::validation::DEFAULT_EXTEND_OVERLAP_FRAMES),
            guidance_capabilities: Some(crate::GuidanceCapabilities::for_recipe(
                &family,
                &guidance_identity,
                None,
            )),
            // Config-only models have local weights: the server's annotate
            // pass derives the contract from checkpoint headers, the same
            // classification the engine applies (#772).
            source_image: None,
        });
    }

    // Sort variants within each model family by quality (bf16 > fp16 > fp8 > q8 > q6 > q5 > q4 > q3).
    // Preserve the manifest-defined order for different base model names.
    sort_models_by_variant_quality(&mut models);

    models
}

/// Sort models so variants of the same base name are grouped together,
/// ordered by quality rank (best first). Different base names keep their
/// original relative order (stable sort).
fn sort_models_by_variant_quality(models: &mut [ModelInfoExtended]) {
    // Assign each base name a sequence number based on its first appearance.
    let mut base_order: Vec<String> = Vec::new();
    for m in models.iter() {
        let base = model_base_name(&m.name).to_string();
        if !base_order.contains(&base) {
            base_order.push(base);
        }
    }

    models.sort_by(|a, b| {
        let base_a = model_base_name(&a.name);
        let base_b = model_base_name(&b.name);
        let ord_a = base_order
            .iter()
            .position(|s| s == base_a)
            .unwrap_or(usize::MAX);
        let ord_b = base_order
            .iter()
            .position(|s| s == base_b)
            .unwrap_or(usize::MAX);

        ord_a
            .cmp(&ord_b)
            .then_with(|| variant_quality_rank(&a.name).cmp(&variant_quality_rank(&b.name)))
    });
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
            std::fs::write(&path, b"test").unwrap();
            // Stamp a `.sha256-verified` marker so post-B3 acceptance
            // logic recognises the fixture as installed (4-byte stub
            // would otherwise fail the size-match fallback).
            crate::download::write_sha256_marker(&path, "deadbeef").unwrap();
        }
    }

    /// Models advertise both their family frame contract and the exact
    /// runnable resolution buckets consumed by every Studio surface.
    #[test]
    fn build_model_catalog_emits_video_frame_defaults() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = Config::default();
        let catalog = build_model_catalog(&config, None, false);

        let ltx2 = catalog
            .iter()
            .find(|model| model.family == "ltx2")
            .expect("an ltx2 manifest model should exist");
        assert_eq!(ltx2.defaults.default_frames, Some(97));
        assert_eq!(ltx2.defaults.default_fps, Some(24));
        // The LTX-2 ceiling is a 20s duration, so the advertised scalar is
        // the frame count that budget buys at this model's own default fps —
        // snapped onto the 8n+1 grid so a client can submit it.
        assert_eq!(
            ltx2.defaults.max_frames,
            Some(crate::validation::ltx2_max_frames_on_grid_at_fps(24)),
            "temporal RoPE ceiling at the model's default fps",
        );
        assert_eq!(ltx2.defaults.max_runtime_seconds, Some(20));
        assert_eq!(
            ltx2.defaults.max_frames_absolute,
            Some(crate::validation::LTX2_MAX_FRAMES_ABSOLUTE)
        );
        assert_eq!(ltx2.defaults.frame_step, Some(8));
        // Every manifest LTX-2 checkpoint ships the spatial upsampler, so it
        // composes: stage 1 halved, one x2 rung, tiled stage-2 refinement.
        // Its advertised ceiling is the composed one, not the family's
        // single-pass value.
        assert_eq!(
            ltx2.defaults.max_pixels,
            Some(crate::validation::LTX2_COMPOSED_MAX_PIXELS),
            "a composing LTX-2 checkpoint advertises the composed ceiling"
        );
        assert_eq!(
            ltx2.defaults.max_axis_pixels,
            Some(crate::validation::LTX2_COMPOSED_MAX_AXIS_PIXELS),
            "the per-axis span is advertised separately from the pixel budget"
        );
        assert_eq!(ltx2.defaults.dimension_alignment, Some(32));
        assert!(
            ltx2.defaults
                .recommended_dimensions
                .iter()
                .any(|size| size.width == 1216 && size.height == 704),
            "LTX-2's default landscape bucket must be advertised to every client",
        );
        assert!(
            ltx2.defaults
                .recommended_dimensions
                .iter()
                .any(|size| size.width == 3840 && size.height == 2112),
            "a composing checkpoint must advertise the 4K UHD rung",
        );
        // Every advertised bucket has to be admissible for this exact model,
        // or the picker offers a size the server rejects.
        for size in &ltx2.defaults.recommended_dimensions {
            assert!(
                crate::validation::validate_generation_dimensions_composed(
                    size.width,
                    size.height,
                    Some("ltx2"),
                    crate::validation::ltx2_spatial_composition(&ltx2.info.name, None),
                )
                .is_ok(),
                "{}x{} is advertised for {} but not admissible",
                size.width,
                size.height,
                ltx2.info.name
            );
        }

        let ltx_video = catalog
            .iter()
            .find(|model| model.family == "ltx-video")
            .expect("an ltx-video manifest model should exist");
        assert_eq!(ltx_video.defaults.default_frames, Some(25));
        assert_eq!(ltx_video.defaults.default_fps, Some(30));
        assert_eq!(ltx_video.defaults.max_frames, Some(257));
        assert_eq!(ltx_video.defaults.frame_step, Some(8));

        let flux = catalog
            .iter()
            .find(|model| model.family == "flux")
            .expect("a flux manifest model should exist");
        assert_eq!(flux.defaults.default_frames, None);
        assert_eq!(flux.defaults.default_fps, None);
        assert_eq!(flux.defaults.max_frames, None);
        assert_eq!(flux.defaults.frame_step, None);
        assert_eq!(
            flux.defaults.max_pixels,
            Some(crate::validation::MAX_PIXELS)
        );
        assert_eq!(flux.defaults.dimension_alignment, Some(16));
        assert!(!flux.defaults.recommended_dimensions.is_empty());
    }

    /// `/api/models` is the single source Studio surfaces read the grid from:
    /// the 5B must advertise its 2.2 VAE's 32px grid while the 2.1-VAE
    /// checkpoints keep the family's 16, and every advertised bucket must sit
    /// on its own model's grid or source fitting floors to a canvas that
    /// admission rejects.
    #[test]
    fn model_catalog_advertises_wan_alignment_per_checkpoint() {
        let catalog = build_model_catalog(&Config::default(), None, false);
        let alignment = |name: &str| {
            catalog
                .iter()
                .find(|model| model.name == name)
                .unwrap_or_else(|| panic!("{name} should be in the catalog"))
                .defaults
                .dimension_alignment
        };
        assert_eq!(alignment("wan22-ti2v-5b:fp16"), Some(32));
        assert_eq!(alignment("wan21-t2v-1.3b:bf16"), Some(16));
        assert_eq!(alignment("wan22-t2v-a14b:q8"), Some(16));

        for model in catalog.iter().filter(|model| model.family == "wan") {
            let align = model
                .defaults
                .dimension_alignment
                .expect("wan rows always advertise a grid");
            for size in &model.defaults.recommended_dimensions {
                assert!(
                    size.width % align == 0 && size.height % align == 0,
                    "{}: advertised {}x{} is off its own {align}px grid",
                    model.name,
                    size.width,
                    size.height,
                );
            }
        }
    }

    /// The low-VRAM Wan tiers (#794) surface in the catalog with their
    /// manifests' own recipes — the Q4_K_M A14B pair keeps the Lightning
    /// 4-step contract and the TI2V-5B Q8_0 keeps the 5B's 121@24 — and the
    /// quality sort keeps them behind the higher-precision variants of the
    /// same base name (bare-name defaults are unchanged).
    #[test]
    fn wan_low_vram_tiers_surface_in_catalog_with_manifest_defaults() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let catalog = build_model_catalog(&Config::default(), None, false);
        let position = |name: &str| {
            catalog
                .iter()
                .position(|model| model.name == name)
                .unwrap_or_else(|| panic!("{name} must be in the catalog"))
        };

        let ti2v_q8 = &catalog[position("wan22-ti2v-5b:q8")];
        assert_eq!(ti2v_q8.family, "wan");
        assert_eq!(ti2v_q8.defaults.default_steps, 20);
        assert_eq!(ti2v_q8.defaults.default_guidance, 5.0);
        assert_eq!(ti2v_q8.defaults.default_frames, Some(121));
        assert_eq!(ti2v_q8.defaults.default_fps, Some(24));
        assert_eq!(ti2v_q8.info.hf_repo, "QuantStack/Wan2.2-TI2V-5B-GGUF");

        for base in ["wan22-t2v-a14b", "wan22-i2v-a14b"] {
            let q4 = &catalog[position(&format!("{base}:q4"))];
            assert_eq!(q4.family, "wan");
            assert_eq!(q4.defaults.default_steps, 4);
            assert_eq!(q4.defaults.default_guidance, 1.0);
            assert_eq!(q4.defaults.default_frames, Some(53));
            assert_eq!(q4.defaults.default_fps, Some(16));

            // Quality order within the base name: q8 > q5 > q4.
            assert!(position(&format!("{base}:q8")) < position(&format!("{base}:q5")));
            assert!(position(&format!("{base}:q5")) < position(&format!("{base}:q4")));
        }

        // fp16 stays the 5B's lead (and bare-name default) variant.
        assert!(position("wan22-ti2v-5b:fp16") < position("wan22-ti2v-5b:q8"));
    }

    #[test]
    fn model_catalog_advertises_default_ltx_guidance_recipe() {
        let catalog = build_model_catalog(&Config::default(), None, false);
        let capability = |name: &str| {
            catalog
                .iter()
                .find(|model| model.name == name)
                .and_then(|model| model.guidance_capabilities)
                .expect("current model rows advertise guidance capabilities")
        };
        assert_eq!(
            capability("ltx-2.3-22b-distilled:fp8"),
            crate::GuidanceCapabilities::FIXED_ONE,
        );
        assert_eq!(
            capability("ltx-2.3-22b-dev:fp8"),
            crate::GuidanceCapabilities::ADJUSTABLE_CFG,
        );
        assert_eq!(
            capability("ltx-video-0.9.8-13b-distilled:bf16"),
            crate::GuidanceCapabilities::FIXED_ONE,
        );
        assert_eq!(
            capability("ltx-video-0.9.8-13b-dev:bf16"),
            crate::GuidanceCapabilities::ADJUSTABLE_CFG,
        );
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
    fn sharded_checkpoints_report_their_transformer_shard_repo() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // qwen-image:bf16 has no single Transformer file — its weights are
        // TransformerShard entries. The repo must come from the shards, not
        // fall back to empty (which strips the source mark in clients).
        let entry = build_model_catalog(&Config::default(), None, false)
            .into_iter()
            .find(|model| model.name == "qwen-image:bf16")
            .expect("manifest model should exist");
        assert_eq!(entry.info.hf_repo, "Qwen/Qwen-Image");
    }

    /// Wan rows advertise the engine's tuned absence-fallback negative so
    /// clients can show, edit, and explicitly disable it; families whose
    /// engines apply no default keep the field absent on the wire.
    #[test]
    fn catalog_advertises_wan_default_negative_prompt_only() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models = build_model_catalog(&Config::default(), None, false);
        let wan = models
            .iter()
            .find(|model| model.family == "wan")
            .expect("wan manifests exist");
        assert_eq!(
            wan.defaults.default_negative_prompt.as_deref(),
            Some(crate::manifest::WAN_DEFAULT_NEGATIVE_PROMPT)
        );
        let flux = models
            .iter()
            .find(|model| model.family == "flux")
            .expect("flux manifests exist");
        assert_eq!(flux.defaults.default_negative_prompt, None);
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
    fn build_model_catalog_hides_compliance_gated_config_only_models() {
        let mut models = HashMap::new();
        models.insert(
            "private-checkpoint".to_string(),
            ModelConfig {
                family: Some("minimax-h3".to_string()),
                ..ModelConfig::default()
            },
        );
        models.insert(
            "ordinary-custom-model".to_string(),
            ModelConfig {
                family: Some("custom".to_string()),
                ..ModelConfig::default()
            },
        );
        models.insert(
            "disguised-checkpoint".to_string(),
            ModelConfig {
                family: Some("custom".to_string()),
                transformer: Some("/models/MiniMax-H3/transformer.safetensors".to_string()),
                ..ModelConfig::default()
            },
        );

        let catalog = build_model_catalog(
            &Config {
                models,
                ..Config::default()
            },
            None,
            false,
        );

        assert!(!catalog
            .iter()
            .any(|model| model.name == "private-checkpoint"));
        assert!(!catalog
            .iter()
            .any(|model| model.name == "disguised-checkpoint"));
        assert!(catalog
            .iter()
            .any(|model| model.name == "ordinary-custom-model"));
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

    #[test]
    fn sort_models_by_variant_quality_groups_and_orders() {
        use super::sort_models_by_variant_quality;

        fn stub(name: &str) -> ModelInfoExtended {
            ModelInfoExtended {
                info: ModelInfo {
                    name: name.to_string(),
                    family: "flux".to_string(),
                    size_gb: 0.0,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: String::new(),
                },
                defaults: ModelDefaults {
                    default_steps: 4,
                    default_guidance: 0.0,
                    default_width: 1024,
                    default_height: 1024,
                    description: String::new(),
                    ..Default::default()
                },
                downloaded: false,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
            }
        }

        let mut models = vec![
            stub("flux-schnell:q4"),
            stub("flux-dev:q4"),
            stub("flux-schnell:bf16"),
            stub("flux-dev:bf16"),
            stub("flux-schnell:q8"),
            stub("flux-dev:q8"),
        ];

        sort_models_by_variant_quality(&mut models);

        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        // flux-schnell appeared first, so all its variants come first (bf16 > q8 > q4)
        // then flux-dev variants (bf16 > q8 > q4)
        assert_eq!(
            names,
            vec![
                "flux-schnell:bf16",
                "flux-schnell:q8",
                "flux-schnell:q4",
                "flux-dev:bf16",
                "flux-dev:q8",
                "flux-dev:q4",
            ]
        );
    }

    #[test]
    fn catalog_contains_upscaler_models() {
        // The full catalog should still include upscaler models (for mold list / Models tab).
        let catalog = build_model_catalog(&Config::default(), None, false);
        assert!(
            catalog.iter().any(|m| m.is_upscaler()),
            "catalog should include upscaler models"
        );
    }

    #[test]
    fn generation_model_filter_excludes_upscalers() {
        // When filtering for generation models, upscalers must not appear.
        let catalog = build_model_catalog(&Config::default(), None, false);
        let generation: Vec<_> = catalog.iter().filter(|m| m.is_generation_model()).collect();

        assert!(
            !generation.is_empty(),
            "there should be generation models in the catalog"
        );
        for m in &generation {
            assert!(
                !m.is_upscaler(),
                "generation model filter should exclude upscaler '{}'",
                m.name
            );
            assert!(
                !m.is_utility(),
                "generation model filter should exclude utility model '{}'",
                m.name
            );
            assert!(
                !m.is_auxiliary(),
                "generation model filter should exclude auxiliary model '{}'",
                m.name
            );
        }
    }

    #[test]
    fn generation_model_filter_excludes_utility_models() {
        let catalog = build_model_catalog(&Config::default(), None, false);
        let utility_in_generation: Vec<_> = catalog
            .iter()
            .filter(|m| m.is_generation_model() && m.is_utility())
            .collect();
        assert!(
            utility_in_generation.is_empty(),
            "no utility models should pass is_generation_model(): {:?}",
            utility_in_generation
                .iter()
                .map(|m| &m.name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn upscaler_config_only_model_excluded_from_generation() {
        // A config-only model with upscaler family should not be a generation model.
        let mut models = HashMap::new();
        models.insert(
            "my-custom-upscaler".to_string(),
            ModelConfig {
                family: Some("upscaler".to_string()),
                description: Some("Custom upscaler".to_string()),
                ..ModelConfig::default()
            },
        );
        let config = Config {
            models,
            ..Config::default()
        };

        let catalog = build_model_catalog(&config, None, false);
        let entry = catalog
            .iter()
            .find(|m| m.name == "my-custom-upscaler")
            .expect("config-only upscaler should be in catalog");

        assert!(entry.is_upscaler());
        assert!(!entry.is_generation_model());
    }
}
