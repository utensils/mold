//! Bridge between the catalog DB and the run-command resolution path.
//!
//! `mold pull cv:<id>` deposits a single-file Civitai checkpoint at
//! `<models_dir>/cv-<id>/<family>/civitai/<id>/<file>.safetensors` and pulls
//! its canonical companions through the manifest path. But until this
//! module landed, `mold run cv:<id> "<prompt>"` errored with
//! `unknown model 'cv:<id>'` — `mold_core::manifest::is_known_model` only
//! consults the manifest registry + `config.toml [models]`, never the
//! catalog DB.
//!
//! This module bridges the gap **without** dragging the `mold-db` /
//! `mold-catalog` crates into `mold-core` (where they'd transitively land
//! in `mold-discord` and `mold-tui`, which the catalog crate explicitly
//! forbids). The flow:
//!
//! 1. `looks_like_catalog_id` — pure `cv:` / `hf:` shape check.
//! 2. `lookup_catalog_row` — hits the catalog DB (soft-fails to `Ok(None)`
//!    when the DB isn't openable, so non-catalog flows aren't blocked).
//! 3. `synthesize_model_config` — translates a `CatalogRow` into a
//!    `ModelConfig` with all paths populated. Reuses the manifest-side
//!    `ModelPaths::resolve` for companions so we don't reimplement the
//!    path-rendering logic that `pull_and_configure` already drives.
//! 4. `install_catalog_model_with_db` — orchestrator: shape → DB → synth →
//!    `config.models.insert(id, synth)`. Once installed, the existing
//!    `is_known_model` check accepts the input and the local engine
//!    factory dispatches via the standard `ModelPaths::resolve(id, config)`
//!    path.

use std::path::Path;

use anyhow::Result;
use mold_catalog::entry::{Bundling, CatalogEntry, Kind};
use mold_core::download::sanitize_recipe_id;
use mold_core::{Config, ModelConfig, ModelPaths};

/// True if `input` has the structural shape of a catalog ID
/// (`cv:<civitai-version-id>` or `hf:<author>/<name>`). Pure shape
/// check — does not consult any source.
pub fn looks_like_catalog_id(input: &str) -> bool {
    input.starts_with("cv:") || input.starts_with("hf:")
}

/// Live single-id lookup for a catalog ID. Routes by the prefix:
/// `cv:` → Civitai model-version API, `hf:` → HF detail+tree.
pub async fn lookup_catalog_entry_live(id: &str) -> Result<CatalogEntry> {
    let civitai_base =
        std::env::var("CIVITAI_BASE").unwrap_or_else(|_| "https://civitai.com".to_string());
    let hf_base = std::env::var("HF_BASE").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let civitai_token = std::env::var("CIVITAI_TOKEN").ok();
    let hf_token = std::env::var("HF_TOKEN").ok();

    if let Some(version_id) = id.strip_prefix("cv:") {
        Ok(mold_catalog::live::fetch_civitai_version(
            &civitai_base,
            version_id,
            civitai_token.as_deref(),
        )
        .await?)
    } else if let Some(repo_id) = id.strip_prefix("hf:") {
        Ok(mold_catalog::live::fetch_hf_repo(&hf_base, repo_id, hf_token.as_deref()).await?)
    } else {
        anyhow::bail!("not a catalog id: {id}")
    }
}

/// Synthesize a `ModelConfig` for a catalog entry, mirroring the on-disk
/// layout that `mold pull <id>` writes to.
///
/// For single-file Civitai checkpoints (Bundling::SingleFile) the
/// resulting `ModelConfig` sets `transformer = vae = primary .safetensors`.
/// That's the duck-type the inference factory uses to dispatch to the
/// `from_single_file` constructors (`is_single_file(paths) =
/// paths.transformer == paths.vae && extension == .safetensors`).
///
/// Companion paths (clip-l tokenizer for SD1.5, clip-l + clip-g
/// tokenizers for SDXL) come from `ModelPaths::resolve("<companion-name>",
/// config)` so we don't reimplement manifest path rendering. Companions
/// must already be on disk (the catalog pull flow guarantees this by
/// pulling them companion-first before the primary).
pub fn synthesize_model_config(
    entry: &CatalogEntry,
    models_dir: &Path,
    config: &Config,
) -> Result<ModelConfig> {
    let primary =
        entry.download_recipe.files.first().ok_or_else(|| {
            anyhow::anyhow!("catalog entry {} has empty download_recipe", entry.id.0)
        })?;

    // Reproduce the path computation that `fetch_recipe` wrote to disk:
    // `<models_dir>/<sanitized-id>/<rendered-dest>`.
    let sanitized = sanitize_recipe_id(entry.id.as_str());
    let (author, name) = match entry.source_id.split_once('/') {
        Some((a, n)) => (a, n),
        None => ("", entry.source_id.as_str()),
    };
    let rendered_dest =
        mold_catalog::entry::render_recipe_dest(&primary.dest, entry.family.as_str(), author, name);
    let primary_path = models_dir.join(&sanitized).join(&rendered_dest);
    let primary_str = primary_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("synthesized path is not valid UTF-8: {primary_path:?}"))?
        .to_string();

    let mut cfg = ModelConfig {
        family: Some(entry.family.as_str().to_string()),
        ..Default::default()
    };

    // Single-file vs. separated dispatch. Civitai checkpoints in this
    // catalog are always SingleFile (the manifest path covers separated
    // diffusers layouts under `[models]`).
    if matches!(entry.bundling, Bundling::SingleFile) {
        cfg.transformer = Some(primary_str.clone());
        cfg.vae = Some(primary_str);
    } else {
        anyhow::bail!(
            "catalog entry {} has bundling={:?} which is not yet wired into the run bridge \
             (single-file only)",
            entry.id.0,
            entry.bundling,
        );
    }

    populate_companion_paths(
        &mut cfg,
        entry.family,
        entry.sub_family.as_deref(),
        entry.kind,
        config,
    );

    Ok(cfg)
}

/// For each canonical companion this family declares, look up the
/// companion's resolved manifest paths and copy the relevant token /
/// encoder fields onto `cfg`. Best-effort: skip companions whose paths
/// aren't resolvable. The single-file engine dispatch surfaces a
/// precise error ("requires a companion-pulled clip_tokenizer") when
/// a *required* field is still None at engine-construction time.
fn populate_companion_paths(
    cfg: &mut ModelConfig,
    family: mold_catalog::families::Family,
    sub_family: Option<&str>,
    kind: Kind,
    config: &Config,
) {
    use mold_catalog::companions::companions_for;
    for companion in companions_for(family, sub_family, Bundling::SingleFile, kind) {
        if let Some(paths) = ModelPaths::resolve(&companion, config) {
            copy_companion_into_cfg(cfg, &companion, &paths);
        }
    }
}

fn copy_companion_into_cfg(cfg: &mut ModelConfig, companion_name: &str, paths: &ModelPaths) {
    let to_string = |p: &std::path::PathBuf| p.to_str().map(str::to_owned);
    match companion_name {
        // CLIP-L lives under one manifest. Its `transformer` field is the
        // weights, `clip_tokenizer` is the tokenizer JSON. Single-file
        // SD1.5 / SDXL only need the tokenizer (the encoder weights are
        // baked into the primary safetensors), but we copy both fields so
        // future engines that DO want the external encoder Just Work.
        "clip-l" => {
            cfg.clip_encoder = to_string(&paths.transformer);
            cfg.clip_tokenizer = paths.clip_tokenizer.as_ref().and_then(to_string);
        }
        "clip-g" => {
            cfg.clip_encoder_2 = to_string(&paths.transformer);
            // The clip-g companion manifest doesn't ship a tokenizer
            // entry — OpenCLIP's vocab is byte-identical to OpenAI's
            // CLIP-L tokenizer, and shipping both would make
            // `shared/companion/tokenizer.json` collide. Fall back to
            // clip-l's tokenizer (already populated above when companion
            // ordering puts clip-l first). The single-file SDXL engine
            // tokenises clip-l and clip-g prompts independently but uses
            // the same vocab/merges file for both, so this is correct.
            cfg.clip_tokenizer_2 = paths
                .clip_tokenizer
                .as_ref()
                .and_then(to_string)
                .or_else(|| cfg.clip_tokenizer.clone());
        }
        "sdxl-vae" | "sd-vae-ft-mse" | "flux-vae" => {
            // Single-file checkpoints embed VAE weights, so leave
            // `cfg.vae` alone (it points at the primary). Stash the
            // companion path on a free field for future external-VAE use.
            // For now this is a no-op.
        }
        "ltx-video-vae" => {
            // LTX-Video Civitai checkpoints are transformer-only. The VAE
            // companion is a separate file and must override cfg.vae so the
            // engine's load_vae() finds it (rather than trying to load the
            // VAE from the transformer safetensors).
            cfg.vae = to_string(&paths.transformer);
        }
        "flux2-vae" => {
            // Flux.2 single-file Civitai checkpoints are transformer-only —
            // the Klein VAE (~168 MB) lives in a separate companion file.
            // Override cfg.vae so the engine's load_vae() reads from the
            // companion rather than from the transformer safetensors.
            cfg.vae = to_string(&paths.transformer);
        }
        "flux2-te" | "flux2-te-9b" => {
            // Flux.2 text encoder (Qwen3 4B with 2 shards for Klein-4B,
            // Qwen3 8B with 4 shards for Klein-9B / FLUX.2-Dev) + Qwen3
            // tokenizer. Mirrors the z-image-te wiring.
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_string)
                .collect::<Vec<_>>()
                .into();
            cfg.text_tokenizer = paths.text_tokenizer.as_ref().and_then(to_string);
        }
        "t5-v1_1-xxl" => {
            cfg.t5_encoder = to_string(&paths.transformer);
            cfg.t5_tokenizer = paths.t5_tokenizer.as_ref().and_then(to_string);
        }
        "z-image-te" => {
            // Z-Image companions bring text-encoder shards.
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_string)
                .collect::<Vec<_>>()
                .into();
            cfg.text_tokenizer = paths.text_tokenizer.as_ref().and_then(to_string);
        }
        "ltx2-te" => {
            // Gemma 3 12B for LTX-2. The runtime (`gemma_root` in
            // `ltx2/assets.rs`) only needs the parent directory of the
            // first text-encoder file, so populating the vec is enough —
            // tokenizer files are tagged TextEncoder in the manifest and
            // ride along in the same directory.
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_string)
                .collect::<Vec<_>>()
                .into();
        }
        _ => {}
    }
}

/// Top-level installer: if `id` is a catalog ID, look it up via live
/// HF/Civitai and synthesize a `ModelConfig` into `config.models`
/// under the same key. Returns `true` when an entry was installed.
///
/// Caller takes `&mut Config` and re-uses the same instance through
/// the rest of the run flow so `ModelPaths::resolve(id, config)` finds
/// the synthesized entry.
pub async fn install_catalog_model_live(config: &mut Config, id: &str) -> Result<bool> {
    if !looks_like_catalog_id(id) {
        return Ok(false);
    }
    let entry = lookup_catalog_entry_live(id).await?;
    let models_dir = config.resolved_models_dir();
    let synth = synthesize_model_config(&entry, &models_dir, config)?;
    config.models.insert(id.to_string(), synth);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;
    use std::collections::HashMap;

    #[test]
    fn looks_like_catalog_id_accepts_civitai_and_hf_shapes() {
        assert!(looks_like_catalog_id("cv:1759168"));
        assert!(looks_like_catalog_id("hf:RunDiffusion/Juggernaut-XL-v9"));
    }

    #[test]
    fn looks_like_catalog_id_rejects_manifest_and_prompt_inputs() {
        assert!(!looks_like_catalog_id("flux-dev"));
        assert!(!looks_like_catalog_id("flux-dev:q4"));
        assert!(!looks_like_catalog_id("realistic-vision-v5:fp16"));
        assert!(!looks_like_catalog_id("a cat"));
        assert!(!looks_like_catalog_id(""));
    }

    /// `Config` builder that is fully explicit (no `..Config::default()`)
    /// so it doesn't read `MOLD_HOME` mid-construction and race with other
    /// env-mutating tests.
    fn explicit_config(models_dir: &str) -> Config {
        Config {
            config_version: 1,
            default_model: "flux2-klein".into(),
            models_dir: models_dir.to_string(),
            server_port: 7680,
            default_width: 1024,
            default_height: 1024,
            default_steps: 4,
            embed_metadata: true,
            t5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            default_negative_prompt: None,
            expand: mold_core::ExpandSettings::default(),
            logging: mold_core::LoggingConfig::default(),
            runpod: mold_core::runpod::RunPodSettings::default(),
            gpus: None,
            queue_size: None,
            models: HashMap::new(),
        }
    }

    use mold_catalog::entry::{
        CatalogId, DownloadRecipe, FamilyRole, FileFormat, LicenseFlags, Modality, RecipeFile,
        Source, TokenKind,
    };
    use mold_catalog::families::Family;

    fn juggernaut_entry() -> CatalogEntry {
        CatalogEntry {
            id: CatalogId::from("cv:1759168"),
            source: Source::Civitai,
            source_id: "1759168".into(),
            name: "Juggernaut XL Ragnarok".into(),
            author: Some("RunDiffusion".into()),
            family: Family::Sdxl,
            family_role: FamilyRole::Finetune,
            sub_family: None,
            modality: Modality::Image,
            kind: Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: Bundling::SingleFile,
            size_bytes: Some(6_938_040_788),
            download_count: 12_345,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["clip-l".into(), "clip-g".into(), "sdxl-vae".into()],
            download_recipe: DownloadRecipe {
                files: vec![RecipeFile {
                    url: "https://civitai.com/api/download/models/1759168".into(),
                    dest: "{family}/civitai/1759168/juggernautXL_ragnarokBy.safetensors".into(),
                    sha256: Some(
                        "DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF".into(),
                    ),
                    size_bytes: Some(6_938_040_788),
                }],
                needs_token: Some(TokenKind::Civitai),
            },
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
        }
    }

    /// Stub the manifest-side companion paths (clip-l, clip-g) into
    /// `config.models` so `populate_companion_paths` can resolve them
    /// without a real on-disk model layout.
    fn stub_companion_paths(config: &mut Config, models_dir: &str) {
        let clip_l_dir = format!("{models_dir}/clip-l");
        config.models.insert(
            "clip-l".into(),
            mold_core::ModelConfig {
                family: Some("clip-l".into()),
                transformer: Some(format!("{clip_l_dir}/model.safetensors")),
                vae: Some(format!("{clip_l_dir}/model.safetensors")),
                clip_tokenizer: Some(format!("{clip_l_dir}/tokenizer.json")),
                ..Default::default()
            },
        );
        let clip_g_dir = format!("{models_dir}/clip-g");
        config.models.insert(
            "clip-g".into(),
            mold_core::ModelConfig {
                family: Some("clip-g".into()),
                transformer: Some(format!("{clip_g_dir}/open_clip_model.safetensors")),
                vae: Some(format!("{clip_g_dir}/open_clip_model.safetensors")),
                clip_tokenizer: Some(format!("{clip_g_dir}/tokenizer.json")),
                ..Default::default()
            },
        );
    }

    fn clear_models_dir_env() {
        for key in [
            "MOLD_TRANSFORMER_PATH",
            "MOLD_VAE_PATH",
            "MOLD_CLIP_PATH",
            "MOLD_CLIP_TOKENIZER_PATH",
            "MOLD_CLIP2_PATH",
            "MOLD_CLIP2_TOKENIZER_PATH",
            "MOLD_T5_PATH",
            "MOLD_T5_TOKENIZER_PATH",
            "MOLD_TEXT_TOKENIZER_PATH",
            "MOLD_DECODER_PATH",
            "MOLD_SPATIAL_UPSCALER_PATH",
            "MOLD_TEMPORAL_UPSCALER_PATH",
            "MOLD_DISTILLED_LORA_PATH",
        ] {
            std::env::remove_var(key);
        }
    }

    /// `stub_companion_paths` above stubs both clip-l and clip-g with
    /// their own `tokenizer.json`. Real on-disk state has clip-g without
    /// a tokenizer entry on its companion manifest — clip-g's text
    /// encoder uses the same vocab/merges as clip-l, so there's no
    /// second `tokenizer.json` to ship. Stub that asymmetry to verify
    /// the bridge's clip-l → clip-g fallback.
    fn stub_companion_paths_no_clip_g_tokenizer(config: &mut Config, models_dir: &str) {
        let clip_l_dir = format!("{models_dir}/clip-l");
        config.models.insert(
            "clip-l".into(),
            mold_core::ModelConfig {
                family: Some("clip-l".into()),
                transformer: Some(format!("{clip_l_dir}/model.safetensors")),
                vae: Some(format!("{clip_l_dir}/model.safetensors")),
                clip_tokenizer: Some(format!("{clip_l_dir}/tokenizer.json")),
                ..Default::default()
            },
        );
        let clip_g_dir = format!("{models_dir}/clip-g");
        config.models.insert(
            "clip-g".into(),
            mold_core::ModelConfig {
                family: Some("clip-g".into()),
                transformer: Some(format!("{clip_g_dir}/open_clip_model.safetensors")),
                vae: Some(format!("{clip_g_dir}/open_clip_model.safetensors")),
                // Intentionally no clip_tokenizer — matches the real
                // clip-g companion manifest.
                ..Default::default()
            },
        );
    }

    #[test]
    fn sdxl_synth_falls_back_clip_g_tokenizer_to_clip_l() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let models_dir = "/tmp/mold-test-models";
        let mut config = explicit_config(models_dir);
        stub_companion_paths_no_clip_g_tokenizer(&mut config, models_dir);

        let entry = juggernaut_entry();
        let synth =
            synthesize_model_config(&entry, std::path::Path::new(models_dir), &config).unwrap();

        let expected_tokenizer = format!("{models_dir}/clip-l/tokenizer.json");
        assert_eq!(
            synth.clip_tokenizer.as_deref(),
            Some(expected_tokenizer.as_str()),
            "clip_tokenizer comes straight from clip-l's manifest path"
        );
        assert_eq!(
            synth.clip_tokenizer_2.as_deref(),
            Some(expected_tokenizer.as_str()),
            "clip_tokenizer_2 falls back to clip-l's tokenizer when clip-g has none"
        );
    }

    #[test]
    fn synthesize_model_config_for_sdxl_single_file_civitai_entry() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let models_dir = "/tmp/mold-test-models";
        let mut config = explicit_config(models_dir);
        stub_companion_paths(&mut config, models_dir);

        let entry = juggernaut_entry();
        let synth =
            synthesize_model_config(&entry, std::path::Path::new(models_dir), &config).unwrap();

        // family is propagated for the engine factory dispatch.
        assert_eq!(synth.family.as_deref(), Some("sdxl"));

        // Single-file duck-type: transformer == vae and points at the
        // recipe-rendered .safetensors under the sanitized cv-id directory.
        let expected = format!(
            "{models_dir}/cv-1759168/sdxl/civitai/1759168/juggernautXL_ragnarokBy.safetensors"
        );
        assert_eq!(synth.transformer.as_deref(), Some(expected.as_str()));
        assert_eq!(synth.vae.as_deref(), Some(expected.as_str()));

        // Companion tokenizers come from the manifest-side paths so the
        // single-file SDXL backend can find clip-l + clip-g tokenizers.
        assert_eq!(
            synth.clip_tokenizer.as_deref(),
            Some(format!("{models_dir}/clip-l/tokenizer.json").as_str())
        );
        assert_eq!(
            synth.clip_tokenizer_2.as_deref(),
            Some(format!("{models_dir}/clip-g/tokenizer.json").as_str())
        );
    }

    #[test]
    fn synthesize_model_config_rejects_separated_bundling() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let mut entry = juggernaut_entry();
        entry.bundling = Bundling::Separated;
        let config = explicit_config("/tmp/mold-test-models");
        let err = synthesize_model_config(
            &entry,
            std::path::Path::new("/tmp/mold-test-models"),
            &config,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("single-file"),
            "should explain the supported bundling, got: {err}",
        );
    }

    // ── Flux.2 catalog bridge ────────────────────────────────────────────

    fn flux2_recipe(version_id: &str, file_name: &str) -> DownloadRecipe {
        DownloadRecipe {
            files: vec![RecipeFile {
                url: format!("https://civitai.com/api/download/models/{version_id}"),
                dest: format!("{{family}}/civitai/{version_id}/{file_name}"),
                sha256: Some(
                    "DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF".into(),
                ),
                size_bytes: Some(12_345_678),
            }],
            needs_token: Some(TokenKind::Civitai),
        }
    }

    fn flux2_klein_9b_entry() -> CatalogEntry {
        CatalogEntry {
            id: CatalogId::from("cv:2759597"),
            source: Source::Civitai,
            source_id: "2759597".into(),
            name: "Miraclein NSFW [Flux2Klein]".into(),
            author: Some("someone".into()),
            family: Family::Flux2,
            family_role: FamilyRole::Finetune,
            sub_family: Some("klein-9b".into()),
            modality: Modality::Image,
            kind: Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: Bundling::SingleFile,
            size_bytes: Some(12_345_678),
            download_count: 0,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["flux2-te-9b".into(), "flux2-vae".into()],
            download_recipe: flux2_recipe("2759597", "miraclein.safetensors"),
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
        }
    }

    fn flux2_klein_4b_entry() -> CatalogEntry {
        let mut entry = flux2_klein_9b_entry();
        entry.id = CatalogId::from("cv:2612554");
        entry.source_id = "2612554".into();
        entry.name = "Flux.2 Klein 4B finetune".into();
        entry.sub_family = Some("klein-4b".into());
        entry.companions = vec!["flux2-te".into(), "flux2-vae".into()];
        entry.download_recipe = flux2_recipe("2612554", "klein4b.safetensors");
        entry
    }

    /// Stub the manifest-side companion paths for the Flux.2 9B encoder
    /// (4 shards) + tokenizer + Klein VAE under `config.models` so the
    /// bridge can resolve them without touching disk.
    fn stub_flux2_9b_companion_paths(config: &mut Config, models_dir: &str) {
        let te_dir = format!("{models_dir}/flux2-te-9b");
        config.models.insert(
            "flux2-te-9b".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!(
                    "{te_dir}/text_encoder/model-00001-of-00004.safetensors"
                )),
                vae: Some(String::new()),
                text_encoder_files: Some(vec![
                    format!("{te_dir}/text_encoder/model-00001-of-00004.safetensors"),
                    format!("{te_dir}/text_encoder/model-00002-of-00004.safetensors"),
                    format!("{te_dir}/text_encoder/model-00003-of-00004.safetensors"),
                    format!("{te_dir}/text_encoder/model-00004-of-00004.safetensors"),
                ]),
                text_tokenizer: Some(format!("{te_dir}/tokenizer/tokenizer.json")),
                ..Default::default()
            },
        );
        let vae_dir = format!("{models_dir}/flux2-vae");
        config.models.insert(
            "flux2-vae".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!("{vae_dir}/vae/diffusion_pytorch_model.safetensors")),
                vae: Some(String::new()),
                ..Default::default()
            },
        );
    }

    fn stub_flux2_4b_companion_paths(config: &mut Config, models_dir: &str) {
        let te_dir = format!("{models_dir}/flux2-te");
        config.models.insert(
            "flux2-te".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!(
                    "{te_dir}/text_encoder/model-00001-of-00002.safetensors"
                )),
                vae: Some(String::new()),
                text_encoder_files: Some(vec![
                    format!("{te_dir}/text_encoder/model-00001-of-00002.safetensors"),
                    format!("{te_dir}/text_encoder/model-00002-of-00002.safetensors"),
                ]),
                text_tokenizer: Some(format!("{te_dir}/tokenizer/tokenizer.json")),
                ..Default::default()
            },
        );
        let vae_dir = format!("{models_dir}/flux2-vae");
        config.models.insert(
            "flux2-vae".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!("{vae_dir}/vae/diffusion_pytorch_model.safetensors")),
                vae: Some(String::new()),
                ..Default::default()
            },
        );
    }

    #[test]
    fn flux2_klein_9b_synth_populates_qwen3_8b_encoder_and_klein_vae() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let models_dir = "/tmp/mold-test-models";
        let mut config = explicit_config(models_dir);
        stub_flux2_9b_companion_paths(&mut config, models_dir);

        let entry = flux2_klein_9b_entry();
        let synth =
            synthesize_model_config(&entry, std::path::Path::new(models_dir), &config).unwrap();

        assert_eq!(synth.family.as_deref(), Some("flux2"));

        // Transformer points at the recipe-rendered .safetensors.
        let expected_transformer =
            format!("{models_dir}/cv-2759597/flux2/civitai/2759597/miraclein.safetensors");
        assert_eq!(
            synth.transformer.as_deref(),
            Some(expected_transformer.as_str())
        );

        // VAE is overridden to the Klein VAE companion (NOT the
        // transformer file — Flux.2 single-file checkpoints don't bundle a
        // VAE).
        let expected_vae =
            format!("{models_dir}/flux2-vae/vae/diffusion_pytorch_model.safetensors");
        assert_eq!(synth.vae.as_deref(), Some(expected_vae.as_str()));

        // Qwen3 8B encoder shards (4 files) populated from the gated
        // Klein-9B companion.
        let shards = synth
            .text_encoder_files
            .as_ref()
            .expect("Klein-9B needs text_encoder_files set");
        assert_eq!(shards.len(), 4, "Klein-9B uses 4 Qwen3-8B shards");
        assert!(shards[0].ends_with("model-00001-of-00004.safetensors"));

        // Tokenizer populated — this is what the original error
        // (`text tokenizer path required for Flux.2 models`) was about.
        let tokenizer = synth
            .text_tokenizer
            .as_deref()
            .expect("text_tokenizer must be populated for Flux.2");
        assert!(tokenizer.ends_with("tokenizer/tokenizer.json"));
    }

    #[test]
    fn flux2_klein_4b_synth_populates_qwen3_4b_encoder() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let models_dir = "/tmp/mold-test-models";
        let mut config = explicit_config(models_dir);
        stub_flux2_4b_companion_paths(&mut config, models_dir);

        let entry = flux2_klein_4b_entry();
        let synth =
            synthesize_model_config(&entry, std::path::Path::new(models_dir), &config).unwrap();

        // Klein-4B uses the 2-shard Qwen3-4B encoder (not the 4-shard 8B).
        let shards = synth
            .text_encoder_files
            .as_ref()
            .expect("Klein-4B needs text_encoder_files set");
        assert_eq!(shards.len(), 2, "Klein-4B uses 2 Qwen3-4B shards");
        assert!(synth.text_tokenizer.is_some());
        // Klein-specific VAE companion still overrides cfg.vae.
        let expected_vae =
            format!("{models_dir}/flux2-vae/vae/diffusion_pytorch_model.safetensors");
        assert_eq!(synth.vae.as_deref(), Some(expected_vae.as_str()));
    }
}
