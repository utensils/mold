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

/// Detect a Civitai single-file SD1.5 / SDXL checkpoint from a `ModelPaths`.
///
/// The Civitai single-file convention is one `.safetensors` containing
/// UNet + VAE + CLIP-L (+ CLIP-G for SDXL); diffusers layouts ship each
/// component as a separate file under its own subdir. The two are
/// distinguishable purely by `paths.transformer == paths.vae` plus the
/// `.safetensors` extension — a duck-type that avoids a new
/// `single_file: Option<PathBuf>` field threaded through every caller.
///
/// Used by the SD1.5 + SDXL match arms to dispatch to `from_single_file`
/// instead of the diffusers-layout `new` constructor.
fn is_single_file(paths: &ModelPaths) -> bool {
    paths.transformer == paths.vae
        && paths
            .transformer
            .extension()
            .is_some_and(|ext| ext == "safetensors")
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
    gpu_ordinal: usize,
    offload: bool,
) -> Result<Box<dyn InferenceEngine>> {
    create_engine_with_pool(
        model_name,
        paths,
        config,
        load_strategy,
        gpu_ordinal,
        offload,
        None,
    )
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
    gpu_ordinal: usize,
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
                gpu_ordinal,
                offload,
                shared_pool,
            )))
        }
        "sd15" | "sd1.5" | "stable-diffusion-1.5" => {
            let scheduler = model_cfg.scheduler.unwrap_or(Scheduler::Ddim);
            if is_single_file(&paths) {
                // Companion-pulled CLIP-L tokenizer (phase 2.7). Civitai
                // single-file checkpoints never bundle the tokenizer.
                let clip_tokenizer = paths
                    .clip_tokenizer
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!(
                        "single-file SD1.5 dispatch requires a companion-pulled clip_tokenizer (phase 2.7)"
                    ))?;
                let single_file = paths.transformer.clone();
                Ok(Box::new(SD15Engine::from_single_file(
                    model_name,
                    single_file,
                    clip_tokenizer,
                    scheduler,
                    load_strategy,
                    gpu_ordinal,
                )?))
            } else {
                Ok(Box::new(SD15Engine::new(
                    model_name,
                    paths,
                    scheduler,
                    load_strategy,
                    gpu_ordinal,
                )))
            }
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
            if is_single_file(&paths) {
                // Companion-pulled CLIP-L + CLIP-G tokenizers (phase 2.7).
                let clip_l_tokenizer = paths.clip_tokenizer.clone().ok_or_else(|| {
                    anyhow::anyhow!(
                        "single-file SDXL dispatch requires a companion-pulled clip_tokenizer (CLIP-L, phase 2.7)"
                    )
                })?;
                let clip_g_tokenizer = paths.clip_tokenizer_2.clone().ok_or_else(|| {
                    anyhow::anyhow!(
                        "single-file SDXL dispatch requires a companion-pulled clip_tokenizer_2 (CLIP-G, phase 2.7)"
                    )
                })?;
                let single_file = paths.transformer.clone();
                Ok(Box::new(SDXLEngine::from_single_file(
                    model_name,
                    single_file,
                    clip_l_tokenizer,
                    clip_g_tokenizer,
                    scheduler,
                    is_turbo,
                    load_strategy,
                    gpu_ordinal,
                )?))
            } else {
                Ok(Box::new(SDXLEngine::new(
                    model_name,
                    paths,
                    scheduler,
                    is_turbo,
                    load_strategy,
                    gpu_ordinal,
                )))
            }
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
                gpu_ordinal,
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
                gpu_ordinal,
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
                gpu_ordinal,
            )))
        }
        "qwen-image" | "qwen_image" => Ok(Box::new(QwenImageEngine::new(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
            offload,
        ))),
        "qwen-image-edit" => Ok(Box::new(QwenImageEngine::new(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
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
                gpu_ordinal,
                shared_pool,
            )))
        }
        "ltx2" | "ltx-2" => Ok(Box::new(Ltx2Engine::new(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
        ))),
        "wuerstchen" | "wuerstchen-v2" => Ok(Box::new(WuerstchenEngine::new(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
        ))),
        other => bail!(
            "unknown model family '{}' for model '{}'. Supported: flux, flux2, ltx-video, ltx2, sd15, sd3, sdxl, z-image, qwen-image, qwen-image-edit, wuerstchen",
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
            0,
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
            0,
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
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "flux2-klein:bf16");
    }

    #[test]
    fn unknown_family_error_lists_ltx_families() {
        let mut config = Config::default();
        config.models.insert(
            "my-model".to_string(),
            mold_core::config::ModelConfig {
                family: Some("mystery".to_string()),
                ..Default::default()
            },
        );
        let err = create_engine(
            "my-model".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .err()
        .expect("unknown family should fail");
        let message = err.to_string();
        assert!(message.contains("ltx-video"));
        assert!(message.contains("ltx2"));
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
            0,
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
            0,
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
            0,
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
            0,
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
            0,
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
            0,
            false,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-wurst");
    }

    // ----- Phase 2.6 single-file routing -----

    fn single_file_paths(checkpoint: &std::path::Path) -> ModelPaths {
        // The factory routing decision keys off transformer == vae +
        // `.safetensors` extension — every other path is a stand-in for
        // companion-pulled tokenizer / encoder assets the factory does
        // not consult when picking the constructor.
        ModelPaths {
            transformer: checkpoint.to_path_buf(),
            transformer_shards: vec![],
            vae: checkpoint.to_path_buf(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: Some(PathBuf::from("/tmp/will-be-clobbered-clip-l")),
            t5_tokenizer: None,
            clip_tokenizer: Some(PathBuf::from("/tmp/clip-l-tokenizer.json")),
            clip_encoder_2: Some(PathBuf::from("/tmp/will-be-clobbered-clip-g")),
            clip_tokenizer_2: Some(PathBuf::from("/tmp/clip-g-tokenizer.json")),
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        }
    }

    #[test]
    fn is_single_file_true_when_transformer_eq_vae_safetensors() {
        let path = PathBuf::from("/tmp/some-civitai-checkpoint.safetensors");
        let paths = single_file_paths(&path);
        assert!(
            super::is_single_file(&paths),
            "transformer == vae + .safetensors must trigger single-file dispatch",
        );
    }

    #[test]
    fn is_single_file_false_when_transformer_neq_vae() {
        // Diffusers-layout shards have transformer + vae as distinct files.
        let mut paths = single_file_paths(&PathBuf::from("/tmp/x.safetensors"));
        paths.vae = PathBuf::from("/tmp/different-vae.safetensors");
        assert!(
            !super::is_single_file(&paths),
            "distinct transformer + vae paths must route to the diffusers `new` constructor",
        );
    }

    #[test]
    fn is_single_file_false_when_extension_not_safetensors() {
        // Diffusers shards with a `.bin` (legacy) or no-extension shard
        // must not be misclassified just because transformer == vae would
        // otherwise match (catches future ModelPaths defaults that point
        // both at the same placeholder dir).
        let mut paths = single_file_paths(&PathBuf::from("/tmp/x.bin"));
        paths.transformer = PathBuf::from("/tmp/x.bin");
        paths.vae = PathBuf::from("/tmp/x.bin");
        assert!(
            !super::is_single_file(&paths),
            "non-`.safetensors` extension must never route to single-file",
        );
    }

    /// Synthesise a minimal SD1.5-shaped single-file safetensors so the
    /// factory's `from_single_file` dispatch survives header parsing +
    /// `build_sd15_remap` validation. Tensor data is one zero F32 per key
    /// — no model weights are materialised.
    fn synth_sd15_for_factory(name: &str) -> PathBuf {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        let path = std::env::temp_dir().join(format!(
            "mold-factory-sd15-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let keys: &[&str] = &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
        ];
        let f32_zero = 0.0f32.to_le_bytes().to_vec();
        let buffers: Vec<Vec<u8>> = keys.iter().map(|_| f32_zero.clone()).collect();
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (key, buf) in keys.iter().zip(buffers.iter()) {
            tensors.insert(
                (*key).to_string(),
                TensorView::new(SafeDtype::F32, vec![1], buf).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, &path).unwrap();
        path
    }

    /// SDXL synthetic checkpoint — adds CLIP-G slabs (incl. the fused QKV
    /// row stack) so `build_sdxl_remap` exercises the `FusedSlice` path at
    /// dispatch time.
    fn synth_sdxl_for_factory(name: &str) -> PathBuf {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        let path = std::env::temp_dir().join(format!(
            "mold-factory-sdxl-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let keys: &[&str] = &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
            "conditioner.embedders.1.model.text_projection",
        ];
        let f32_zero = 0.0f32.to_le_bytes().to_vec();
        let buffers: Vec<Vec<u8>> = keys.iter().map(|_| f32_zero.clone()).collect();
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (key, buf) in keys.iter().zip(buffers.iter()) {
            tensors.insert(
                (*key).to_string(),
                TensorView::new(SafeDtype::F32, vec![1], buf).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, &path).unwrap();
        path
    }

    #[test]
    fn factory_sd15_single_file_dispatches_to_from_single_file() {
        let single_file = synth_sd15_for_factory("dispatch");
        let paths = single_file_paths(&single_file);

        let engine = create_engine(
            "sd15:fp16".to_string(),
            paths,
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        )
        .expect("single-file dispatch must succeed against a valid SD1.5 checkpoint");

        assert_eq!(engine.model_name(), "sd15:fp16");

        // `from_single_file` rebuilds `ModelPaths` so every component
        // resolves through the single file; the original
        // `clip_encoder = /tmp/will-be-clobbered-clip-l` we passed in
        // must be replaced by the checkpoint path. If the factory had
        // dispatched to `SD15Engine::new`, the input `clip_encoder`
        // would have survived verbatim.
        let resolved = engine
            .model_paths()
            .expect("SD15 engine must expose its ModelPaths via the trait");
        assert_eq!(
            resolved.clip_encoder.as_deref(),
            Some(single_file.as_path()),
            "single-file dispatch must materialise clip_encoder from the checkpoint",
        );

        let _ = std::fs::remove_file(single_file);
    }

    #[test]
    fn factory_sd15_diffusers_layout_routes_to_new_constructor() {
        // Distinct transformer + vae paths — the diffusers-layout signal.
        // `SD15Engine::new` does not validate path existence, so this
        // succeeds even with non-existent paths. If the factory had
        // misrouted to `from_single_file`, the missing-file check would
        // surface a `single-file checkpoint not found` error.
        let mut paths = dummy_paths();
        paths.transformer =
            PathBuf::from("/tmp/diffusers/unet/diffusion_pytorch_model.safetensors");
        paths.vae = PathBuf::from("/tmp/diffusers/vae/diffusion_pytorch_model.safetensors");

        let engine = create_engine(
            "sd15:fp16".to_string(),
            paths,
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        )
        .expect("diffusers-layout SD1.5 must route to `new` and not validate paths");

        // The original `clip_encoder` path must survive untouched —
        // proof we did **not** take the single-file branch.
        let resolved = engine.model_paths().expect("SD15 must expose paths");
        assert_eq!(
            resolved.clip_encoder.as_deref(),
            Some(std::path::Path::new("/tmp/clip")),
        );
    }

    #[test]
    fn factory_sd15_single_file_missing_checkpoint_surfaces_error() {
        // transformer == vae + `.safetensors`, but the file does not
        // exist. The single-file path errors at the existence check; the
        // diffusers `new` path would instead succeed (no I/O at construct
        // time). This test pins down which branch ran.
        let bogus = std::env::temp_dir().join(format!(
            "mold-factory-sd15-missing-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let paths = single_file_paths(&bogus);

        let result = create_engine(
            "sd15:fp16".to_string(),
            paths,
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        );

        let err = result
            .err()
            .expect("missing single-file checkpoint must surface as constructor error");
        assert!(
            err.to_string().contains("single-file checkpoint not found"),
            "expected single-file error, got: {err}",
        );
    }

    #[test]
    fn factory_sdxl_single_file_dispatches_to_from_single_file() {
        let single_file = synth_sdxl_for_factory("dispatch");
        let paths = single_file_paths(&single_file);

        let engine = create_engine(
            "sdxl-base:fp16".to_string(),
            paths,
            &Config::default(),
            LoadStrategy::Sequential,
            0,
            false,
        )
        .expect("single-file dispatch must succeed against a valid SDXL checkpoint");

        let resolved = engine
            .model_paths()
            .expect("SDXL engine must expose its ModelPaths via the trait");
        // SDXL adds the CLIP-G identity check on top of CLIP-L; both must
        // resolve to the single-file path after dispatch.
        assert_eq!(
            resolved.clip_encoder.as_deref(),
            Some(single_file.as_path()),
            "single-file dispatch must materialise clip_encoder (CLIP-L) from the checkpoint",
        );
        assert_eq!(
            resolved.clip_encoder_2.as_deref(),
            Some(single_file.as_path()),
            "single-file dispatch must materialise clip_encoder_2 (CLIP-G) from the checkpoint",
        );

        let _ = std::fs::remove_file(single_file);
    }

    #[test]
    fn factory_sdxl_single_file_threads_is_turbo_via_model_config() {
        // is_turbo is plumbed through model_cfg → from_single_file.
        // Field-level verification of the threaded value lives in
        // `sdxl/pipeline.rs::tests` where direct field access works; at
        // the factory boundary we only assert that an `is_turbo = true`
        // model config dispatches successfully (which proves the
        // constructor signature accepts the threaded value).
        let single_file = synth_sdxl_for_factory("turbo");
        let paths = single_file_paths(&single_file);

        let mut config = Config::default();
        config.models.insert(
            "sdxl-turbo:fp16".to_string(),
            mold_core::config::ModelConfig {
                family: Some("sdxl".to_string()),
                is_turbo: Some(true),
                ..Default::default()
            },
        );

        let engine = create_engine(
            "sdxl-turbo:fp16".to_string(),
            paths,
            &config,
            LoadStrategy::Sequential,
            0,
            false,
        )
        .expect("is_turbo = true must thread through to the single-file constructor");
        assert_eq!(engine.model_name(), "sdxl-turbo:fp16");

        let _ = std::fs::remove_file(single_file);
    }
}
