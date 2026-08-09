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
use mold_catalog::entry::CatalogEntry;
use mold_core::{Config, ModelConfig};

/// True if `input` has the structural shape of a catalog ID
/// (`cv:<civitai-version-id>` or `hf:<author>/<name>`). Pure shape
/// check — does not consult any source.
pub fn looks_like_catalog_id(input: &str) -> bool {
    mold_catalog::resolve::looks_like_catalog_id(input)
}

/// Live single-id lookup for a catalog ID via the shared cv:/hf: dispatcher.
/// Bases honor `CIVITAI_BASE` / `HF_BASE` env for test overrides; tokens come
/// from `CIVITAI_TOKEN` / `HF_TOKEN`.
pub async fn lookup_catalog_entry_live(id: &str) -> Result<CatalogEntry> {
    mold_core::require_model_activation(id, None)?;
    let civitai_base =
        std::env::var("CIVITAI_BASE").unwrap_or_else(|_| "https://civitai.com".to_string());
    let hf_base = std::env::var("HF_BASE").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let civitai_token = std::env::var("CIVITAI_TOKEN").ok();
    let hf_token = std::env::var("HF_TOKEN").ok();

    let entry = mold_catalog::live::fetch_entry_by_id(
        id,
        &civitai_base,
        &hf_base,
        civitai_token.as_deref(),
        hf_token.as_deref(),
    )
    .await?;
    mold_catalog::entry::require_catalog_entry_activation(&entry)?;
    Ok(entry)
}

/// Resolve a pure catalog intent into a `ModelConfig` using the CLI's policy:
/// resolution runs right after the download completes, so a missing companion
/// is logged and skipped (the engine-load surface reports any genuinely
/// missing field) and the primary-present precheck is left off.
fn resolve_intent(
    model_name: &str,
    intent: &mold_catalog::synthesis::CatalogModelIntent,
    config: &Config,
) -> Result<ModelConfig> {
    mold_catalog::resolve::resolve_intent_to_model_config(
        model_name,
        intent,
        config,
        mold_catalog::resolve::ResolveOptions {
            missing_companions: mold_catalog::resolve::MissingCompanionPolicy::WarnAndSkip,
            require_primary_present: false,
        },
    )
    .map_err(anyhow::Error::from)
}

/// Synthesize a `ModelConfig` for a catalog entry, mirroring the on-disk
/// layout that `mold pull <id>` writes to. Pure intent synthesis
/// (`mold_catalog::synthesis`) followed by the shared disk-aware resolver.
pub fn synthesize_model_config(
    entry: &CatalogEntry,
    models_dir: &Path,
    config: &Config,
) -> Result<ModelConfig> {
    mold_catalog::entry::require_catalog_entry_activation(entry)?;
    let intent = mold_catalog::synthesis::synthesize_intent(entry, models_dir)?;
    resolve_intent(entry.id.as_str(), &intent, config)
}

/// Unified catalog-ID injection for the run/generate entry points.
///
/// If `id` is a catalog ID (`cv:*` / `hf:*`), resolve it — sidecar-first
/// (already-installed checkpoint, no network) then live HF/Civitai — and
/// synthesize a `ModelConfig` into `config.models` under the same key so the
/// downstream `ModelPaths::resolve(id, config)` finds it. Returns `Ok(true)`
/// when an entry was installed, `Ok(false)` for a non-catalog id (a no-op, so
/// callers can invoke it unconditionally).
///
/// Caller takes `&mut Config` and re-uses the same instance through the rest
/// of the run flow.
pub async fn ensure_catalog_model(config: &mut Config, id: &str) -> Result<bool> {
    if !looks_like_catalog_id(id) {
        return Ok(false);
    }
    if install_catalog_model_from_installed_sidecar(config, id)? {
        return Ok(true);
    }
    let entry = lookup_catalog_entry_live(id).await?;
    let models_dir = config.resolved_models_dir();
    let synth = synthesize_model_config(&entry, &models_dir, config)?;
    config.models.insert(id.to_string(), synth);
    Ok(true)
}

/// Resolve and enforce model activation before a cloud command performs any
/// provider operation. Catalog IDs are resolved through the same sidecar/live
/// bridge as local generation so an opaque `cv:` or `hf:` identifier cannot
/// hide restricted metadata. Configured families and paths are checked both
/// before and after catalog synthesis; the first pass keeps an already-known
/// restricted config from triggering even a catalog lookup.
pub async fn require_cloud_model_activation(config: &mut Config, id: &str) -> Result<()> {
    mold_core::require_model_activation(id, None)?;

    let canonical = mold_core::manifest::resolve_model_name(id);
    mold_core::require_model_activation(&canonical, None)?;
    require_configured_model_activation(config, id)?;
    if canonical != id {
        require_configured_model_activation(config, &canonical)?;
    }
    require_manifest_model_activation(&canonical)?;

    if looks_like_catalog_id(id) {
        ensure_catalog_model(config, id).await?;
        require_configured_model_activation(config, id)?;
    }

    Ok(())
}

fn require_manifest_model_activation(id: &str) -> Result<()> {
    let Some(manifest) = mold_core::manifest::find_manifest(id) else {
        return Ok(());
    };
    mold_core::require_model_activation(&manifest.name, Some(&manifest.family))?;
    for file in &manifest.files {
        mold_core::require_model_activation(&file.hf_repo, Some(&manifest.family))?;
        mold_core::require_model_activation(&file.hf_filename, Some(&manifest.family))?;
    }
    Ok(())
}

fn require_configured_model_activation(config: &Config, id: &str) -> Result<()> {
    let Some(model) = config.lookup_model_config(id) else {
        return Ok(());
    };
    let family = model.family.as_deref();
    mold_core::require_model_activation(id, family)?;
    let models_root = config.resolved_models_dir();
    for path in model.all_file_paths() {
        mold_core::require_model_artifact_activation(Path::new(&path), Some(&models_root), family)?;
    }
    Ok(())
}

/// Fail-closed local authority for a model identity without performing a live
/// catalog lookup or downloading anything.
pub(crate) fn require_known_model_activation(config: &Config, id: &str) -> Result<()> {
    mold_core::require_model_activation(id, None)?;
    let canonical = mold_core::manifest::resolve_model_name(id);
    mold_core::require_model_activation(&canonical, None)?;
    require_configured_model_activation(config, id)?;
    if canonical != id {
        require_configured_model_activation(config, &canonical)?;
    }
    require_manifest_model_activation(&canonical)?;

    let models_root = config.resolved_models_dir();
    mold_catalog::sidecar::require_installed_sidecar_activation(&models_root, id)?;
    let family = config
        .lookup_model_config(id)
        .and_then(|model| model.family)
        .or_else(|| {
            mold_core::manifest::find_manifest(&canonical).map(|manifest| manifest.family.clone())
        });
    if let Some(paths) = mold_core::ModelPaths::resolve(id, config) {
        for path in paths.all_file_paths() {
            mold_core::require_model_artifact_activation(
                path,
                Some(&models_root),
                family.as_deref(),
            )?;
        }
    }
    Ok(())
}

pub fn install_catalog_model_from_installed_sidecar(config: &mut Config, id: &str) -> Result<bool> {
    if !looks_like_catalog_id(id) {
        return Ok(false);
    }
    mold_catalog::sidecar::require_installed_sidecar_activation(&config.resolved_models_dir(), id)?;
    let Some(synth) = synthesize_model_config_from_installed_sidecar(config, id)? else {
        return Ok(false);
    };
    mold_core::require_model_activation(id, synth.family.as_deref())?;
    config.models.insert(id.to_string(), synth);
    Ok(true)
}

fn synthesize_model_config_from_installed_sidecar(
    config: &Config,
    id: &str,
) -> Result<Option<ModelConfig>> {
    let models_dir = config.resolved_models_dir();
    let Some(intent) = mold_catalog::resolve::installed_intent_from_sidecar(&models_dir, id) else {
        return Ok(None);
    };
    resolve_intent(id, &intent, config).map(Some)
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

    #[test]
    fn resolved_catalog_entry_cannot_hide_h3_behind_an_opaque_id() {
        let mut entry = juggernaut_entry();
        entry.id = CatalogId::from("cv:42");
        entry.source_id = "42".into();
        entry.name = "MiniMax H3 FL2VA".into();

        let error = mold_catalog::entry::require_catalog_entry_activation(&entry).unwrap_err();
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
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
            media_roots: None,
            default_negative_prompt: None,
            expand: mold_core::ExpandSettings::default(),
            scheduler: Default::default(),
            logging: mold_core::LoggingConfig::default(),
            runpod: mold_core::runpod::RunPodSettings::default(),
            lambda: mold_core::lambda::LambdaSettings::default(),
            gpus: None,
            queue_size: None,
            models: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn cloud_activation_rejects_configured_family_and_path_without_catalog_io() {
        let mut family_config = explicit_config("/tmp/mold-cloud-policy-family");
        family_config.models.insert(
            "renamed-model".into(),
            ModelConfig {
                family: Some("minimax-h3".into()),
                transformer: Some("/models/renamed/weights.safetensors".into()),
                ..Default::default()
            },
        );
        let error = require_cloud_model_activation(&mut family_config, "renamed-model")
            .await
            .expect_err("configured H3 family must be rejected");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));

        let mut path_config = explicit_config("/tmp/mold-cloud-policy-path");
        path_config.models.insert(
            "renamed-model".into(),
            ModelConfig {
                family: Some("custom".into()),
                transformer: Some("/models/MiniMax-H3/weights.safetensors".into()),
                ..Default::default()
            },
        );
        let error = require_cloud_model_activation(&mut path_config, "renamed-model")
            .await
            .expect_err("configured H3 path must be rejected");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[tokio::test]
    async fn cloud_activation_keeps_h3_lookalikes_available() {
        let mut config = explicit_config("/tmp/mold-cloud-policy-lookalike");
        config.models.insert(
            "renamed-model".into(),
            ModelConfig {
                family: Some("custom".into()),
                transformer: Some("/models/minimax-h30/weights.safetensors".into()),
                ..Default::default()
            },
        );
        require_cloud_model_activation(&mut config, "renamed-model")
            .await
            .expect("H3 lookalike must remain available");
    }

    use mold_catalog::entry::{
        Bundling, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags, Modality,
        RecipeFile, RecipeFileRole, Source, TokenKind,
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
                    role: None,
                }],
                needs_token: Some(TokenKind::Civitai),
            },
            supported: true,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
            page_url: None,
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

    #[test]
    fn zimage_recipe_text_encoder_wins_and_shared_companion_vae_is_used() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let _home_guard = pin_mold_home(std::path::Path::new("/tmp/mold-test-models"));

        let models_dir = "/tmp/mold-test-models";
        let mut config = explicit_config(models_dir);
        let te_dir = format!("{models_dir}/z-image-te");
        config.models.insert(
            "z-image-te".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!(
                    "{te_dir}/text_encoder/model-00001-of-00003.safetensors"
                )),
                vae: Some(format!("{te_dir}/vae/diffusion_pytorch_model.safetensors")),
                text_encoder_files: Some(vec![
                    format!("{te_dir}/text_encoder/model-00001-of-00003.safetensors"),
                    format!("{te_dir}/text_encoder/model-00002-of-00003.safetensors"),
                    format!("{te_dir}/text_encoder/model-00003-of-00003.safetensors"),
                ]),
                text_tokenizer: Some(format!("{te_dir}/tokenizer/tokenizer.json")),
                ..Default::default()
            },
        );
        let entry = CatalogEntry {
            id: CatalogId::from("cv:2442439"),
            source: Source::Civitai,
            source_id: "2442439".into(),
            name: "Z Image Turbo".into(),
            author: Some("z".into()),
            family: Family::ZImage,
            family_role: FamilyRole::Finetune,
            sub_family: None,
            modality: Modality::Image,
            kind: Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: Bundling::SingleFile,
            size_bytes: Some(12_021_353_906),
            download_count: 0,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["z-image-te".into()],
            download_recipe: DownloadRecipe {
                files: vec![
                    RecipeFile {
                        url: "https://civitai.example/model".into(),
                        dest: "{family}/civitai/2442439/zImageTurbo_turbo.safetensors".into(),
                        sha256: None,
                        size_bytes: Some(12_021_353_906),
                        role: None,
                    },
                    RecipeFile {
                        url: "https://civitai.example/text".into(),
                        dest: "{family}/civitai/2442439/zImageTurbo_turbo_txt.safetensors".into(),
                        sha256: None,
                        size_bytes: Some(8_044_982_048),
                        role: Some(RecipeFileRole::TextEncoder),
                    },
                ],
                needs_token: Some(TokenKind::Civitai),
            },
            supported: true,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
            page_url: None,
        };

        let synth =
            synthesize_model_config(&entry, std::path::Path::new(models_dir), &config).unwrap();

        let expected_transformer = format!(
            "{models_dir}/cv-2442439/z-image/civitai/2442439/zImageTurbo_turbo.safetensors"
        );
        let expected_vae = format!("{te_dir}/vae/diffusion_pytorch_model.safetensors");
        let expected_tokenizer = format!("{te_dir}/tokenizer/tokenizer.json");
        assert_eq!(
            synth.transformer.as_deref(),
            Some(expected_transformer.as_str())
        );
        assert_eq!(synth.vae.as_deref(), Some(expected_vae.as_str()));
        let expected_text_encoder = format!(
            "{models_dir}/cv-2442439/z-image/civitai/2442439/zImageTurbo_turbo_txt.safetensors"
        );
        let expected_text_encoder_files = vec![expected_text_encoder];
        assert_eq!(
            synth.text_encoder_files.as_deref(),
            Some(expected_text_encoder_files.as_slice())
        );
        assert_eq!(
            synth.text_tokenizer.as_deref(),
            Some(expected_tokenizer.as_str())
        );
    }

    /// Pin MOLD_HOME to a path that contains no real `.hf-cache/` so
    /// `ModelPaths::resolve` can't find an unrelated installed companion
    /// from the dev machine and inject it into the synthesized config.
    /// Returns a guard whose Drop restores the previous value.
    struct MoldHomeGuard {
        prev: Option<String>,
    }

    impl Drop for MoldHomeGuard {
        fn drop(&mut self) {
            unsafe {
                match self.prev.take() {
                    Some(v) => std::env::set_var("MOLD_HOME", v),
                    None => std::env::remove_var("MOLD_HOME"),
                }
            }
        }
    }

    fn pin_mold_home(path: &std::path::Path) -> MoldHomeGuard {
        let prev = std::env::var("MOLD_HOME").ok();
        unsafe {
            std::env::set_var("MOLD_HOME", path.to_string_lossy().as_ref());
        }
        MoldHomeGuard { prev }
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
        let _home_guard = pin_mold_home(std::path::Path::new("/tmp/mold-test-models"));

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
        let _home_guard = pin_mold_home(std::path::Path::new("/tmp/mold-test-models"));

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
    fn installed_checkpoint_sidecar_synthesizes_config_without_live_lookup() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let dir = tempfile::tempdir().unwrap();
        let _home_guard = pin_mold_home(dir.path());
        let models_dir = dir.path().to_str().unwrap();
        let mut config = explicit_config(models_dir);
        stub_companion_paths(&mut config, models_dir);

        let sidecar_dir = dir.path().join("cv-1075446");
        let primary_rel = "sdxl/civitai/1075446/realism.safetensors";
        let primary_path = sidecar_dir.join(primary_rel);
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        std::fs::write(&primary_path, b"ok").unwrap();
        mold_catalog::sidecar::write_sidecar(
            &sidecar_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &mold_catalog::sidecar::CatalogSidecar {
                schema: mold_catalog::sidecar::SIDECAR_SCHEMA,
                id: "cv:1075446".into(),
                source: "civitai".into(),
                source_id: "1075446".into(),
                name: "Realism By Stable Yogi".into(),
                author: Some("Stable_Yogi".into()),
                family: "sdxl".into(),
                family_role: "finetune".into(),
                sub_family: None,
                kind: "checkpoint".into(),
                modality: "image".into(),
                nsfw: None,
                description: None,
                tags: Vec::new(),
                license: None,
                page_url: None,
                thumbnail_url: None,
                size_bytes: Some(2),
                supported: true,
                trained_words: Vec::new(),
                primary_filename_rel: primary_rel.into(),
                primary_size_bytes: None,
                low_noise_filename_rel: None,
                low_noise_size_bytes: None,
                written_at: 0,
            },
        )
        .unwrap();

        let synth = synthesize_model_config_from_installed_sidecar(&config, "cv:1075446").unwrap();
        let synth = synth.expect("installed sidecar should synthesize without live catalog");

        assert_eq!(synth.family.as_deref(), Some("sdxl"));
        assert_eq!(synth.transformer.as_deref(), primary_path.to_str());
        assert_eq!(synth.vae.as_deref(), primary_path.to_str());
        assert_eq!(
            synth.clip_tokenizer.as_deref(),
            Some(format!("{models_dir}/clip-l/tokenizer.json").as_str())
        );
    }

    #[test]
    fn synthesize_model_config_sets_transformer_for_single_file() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        // Single-file SDXL entry with no companions installed: under the
        // CLI's WarnAndSkip policy the missing companions are logged and
        // skipped, and the synthesized config still points transformer at
        // the recipe path (the CLI no longer bails on separated bundling —
        // that rejection is now solely `synthesize_intent`'s job).
        let models_dir = "/tmp/mold-test-models";
        let entry = juggernaut_entry();
        let config = explicit_config(models_dir);
        let synth =
            synthesize_model_config(&entry, std::path::Path::new(models_dir), &config).unwrap();
        let expected = format!(
            "{models_dir}/cv-1759168/sdxl/civitai/1759168/juggernautXL_ragnarokBy.safetensors"
        );
        assert_eq!(synth.transformer.as_deref(), Some(expected.as_str()));
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
                role: None,
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
            supported: true,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
            page_url: None,
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
        let _home_guard = pin_mold_home(std::path::Path::new("/tmp/mold-test-models"));

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
        let _home_guard = pin_mold_home(std::path::Path::new("/tmp/mold-test-models"));

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

    // ── FLUX catalog bridge: VAE-bundle probe + flux-vae companion wiring ──

    fn flux_recipe(version_id: &str, file_name: &str) -> DownloadRecipe {
        DownloadRecipe {
            files: vec![RecipeFile {
                url: format!("https://civitai.com/api/download/models/{version_id}"),
                dest: format!("{{family}}/civitai/{version_id}/{file_name}"),
                sha256: Some(
                    "DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF".into(),
                ),
                size_bytes: Some(12_000_000_000),
                role: None,
            }],
            needs_token: Some(TokenKind::Civitai),
        }
    }

    fn flux_unet_only_entry(version_id: &str, file_name: &str) -> CatalogEntry {
        CatalogEntry {
            id: CatalogId::from(format!("cv:{version_id}")),
            source: Source::Civitai,
            source_id: version_id.to_string(),
            name: "Real Horny Pro V3 (transformer-only)".into(),
            author: Some("someone".into()),
            family: Family::Flux,
            family_role: FamilyRole::Finetune,
            sub_family: None,
            modality: Modality::Image,
            kind: Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: Bundling::SingleFile,
            size_bytes: Some(12_000_000_000),
            download_count: 0,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["t5-v1_1-xxl".into(), "clip-l".into(), "flux-vae".into()],
            download_recipe: flux_recipe(version_id, file_name),
            supported: true,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
            page_url: None,
        }
    }

    /// Stub the flux-vae + t5 + clip-l companion paths so
    /// `populate_companion_paths` can resolve them without disk I/O.
    fn stub_flux_companion_paths(config: &mut Config, models_dir: &str) {
        let clip_l_dir = format!("{models_dir}/clip-l");
        config.models.insert(
            "clip-l".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!("{clip_l_dir}/model.safetensors")),
                vae: Some(format!("{clip_l_dir}/model.safetensors")),
                clip_tokenizer: Some(format!("{clip_l_dir}/tokenizer.json")),
                ..Default::default()
            },
        );
        let t5_dir = format!("{models_dir}/t5-v1_1-xxl");
        config.models.insert(
            "t5-v1_1-xxl".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!("{t5_dir}/t5xxl_fp16.safetensors")),
                vae: Some(format!("{t5_dir}/t5xxl_fp16.safetensors")),
                t5_tokenizer: Some(format!("{t5_dir}/tokenizer.json")),
                ..Default::default()
            },
        );
        let vae_dir = format!("{models_dir}/flux-vae");
        config.models.insert(
            "flux-vae".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!("{vae_dir}/ae.safetensors")),
                vae: Some(String::new()),
                ..Default::default()
            },
        );
    }

    /// Write a synthetic safetensors at `path` whose header advertises the
    /// given keys (each as a 1-element F32 tensor whose data offsets land
    /// at `[0, 4]`). The 4-byte tensor blob is shared by every key — the
    /// safetensors format only reads `data_offsets[1]` bytes past the
    /// header, and our header probe never reads tensor data at all. Every
    /// fixture parses cleanly through a real safetensors reader too.
    fn write_safetensors_with_keys(path: &std::path::Path, keys: &[&str]) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        for key in keys {
            header.insert(
                (*key).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }),
            );
        }
        let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).expect("create fixture");
        f.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_json).unwrap();
        f.write_all(&[0u8; 4]).unwrap(); // F32 zero — shared by every key
    }

    #[test]
    fn synthesize_catalog_config_uses_flux_vae_companion_when_bundle_lacks_vae() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        // Build the on-disk transformer-only fixture exactly where
        // `synthesize_model_config` will probe it.
        let dir = tempfile::tempdir().unwrap();
        let _home_guard = pin_mold_home(dir.path());
        let models_dir = dir.path().to_str().unwrap();
        let primary_path = std::path::PathBuf::from(format!(
            "{models_dir}/cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors"
        ));
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        // Transformer-only: NO `encoder.conv_in.*` / `first_stage_model.*` /
        // `vae.*` keys — the cv:994561 case.
        write_safetensors_with_keys(
            &primary_path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );

        let mut config = explicit_config(models_dir);
        stub_flux_companion_paths(&mut config, models_dir);

        let entry = flux_unet_only_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let synth =
            synthesize_model_config(&entry, std::path::Path::new(models_dir), &config).unwrap();

        // cfg.transformer points at the primary file — unchanged from
        // the legacy bundled-VAE path.
        assert_eq!(synth.transformer.as_deref(), primary_path.to_str());
        // cfg.vae is the flux-vae companion — NOT the primary checkpoint.
        let expected_vae = format!("{models_dir}/flux-vae/ae.safetensors");
        assert_eq!(
            synth.vae.as_deref(),
            Some(expected_vae.as_str()),
            "flux-vae companion must populate cfg.vae for transformer-only checkpoints"
        );
    }

    #[test]
    fn synthesize_catalog_config_uses_bundled_vae_when_present() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let dir = tempfile::tempdir().unwrap();
        let _home_guard = pin_mold_home(dir.path());
        let models_dir = dir.path().to_str().unwrap();
        let primary_path = std::path::PathBuf::from(format!(
            "{models_dir}/cv-101010/flux/civitai/101010/flux_full.safetensors"
        ));
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        // Bundled-VAE fixture — A1111 prefix.
        write_safetensors_with_keys(
            &primary_path,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
                "first_stage_model.encoder.conv_in.weight",
            ],
        );

        let mut config = explicit_config(models_dir);
        stub_flux_companion_paths(&mut config, models_dir);

        let entry = flux_unet_only_entry("101010", "flux_full.safetensors");
        let synth =
            synthesize_model_config(&entry, std::path::Path::new(models_dir), &config).unwrap();

        // Both fields point at the primary checkpoint — the bundled VAE wins
        // and the flux-vae companion is a no-op.
        assert_eq!(synth.transformer.as_deref(), primary_path.to_str());
        assert_eq!(
            synth.vae.as_deref(),
            primary_path.to_str(),
            "bundled-VAE checkpoint must keep cfg.vae == primary; flux-vae companion is a no-op"
        );
    }

    // ── ensure_catalog_model unified entry point ─────────────────────────

    #[tokio::test]
    async fn ensure_catalog_model_is_noop_for_non_catalog_id() {
        // A bare manifest name (or a prompt) is not a catalog id — this reads
        // no env and touches no network, so it needs no ENV_LOCK.
        let mut config = explicit_config("/tmp/mold-test-models");
        let before = config.models.len();
        let installed = ensure_catalog_model(&mut config, "flux-dev").await.unwrap();
        assert!(
            !installed,
            "non-catalog id must be a no-op returning Ok(false)"
        );
        assert_eq!(config.models.len(), before);
    }

    // The ENV_LOCK guard is held across the `.await` to keep MOLD_HOME pinned
    // for the duration of the (sidecar-only, no real I/O) resolution; the tokio
    // test is single-threaded so this cannot deadlock.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn ensure_catalog_model_installs_from_installed_sidecar_without_network() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let dir = tempfile::tempdir().unwrap();
        let _home_guard = pin_mold_home(dir.path());
        let models_dir = dir.path().to_str().unwrap();
        let mut config = explicit_config(models_dir);
        stub_companion_paths(&mut config, models_dir);

        let sidecar_dir = dir.path().join("cv-1075446");
        let primary_rel = "sdxl/civitai/1075446/realism.safetensors";
        let primary_path = sidecar_dir.join(primary_rel);
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        std::fs::write(&primary_path, b"ok").unwrap();
        mold_catalog::sidecar::write_sidecar(
            &sidecar_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &mold_catalog::sidecar::CatalogSidecar {
                schema: mold_catalog::sidecar::SIDECAR_SCHEMA,
                id: "cv:1075446".into(),
                source: "civitai".into(),
                source_id: "1075446".into(),
                name: "Realism By Stable Yogi".into(),
                author: Some("Stable_Yogi".into()),
                family: "sdxl".into(),
                family_role: "finetune".into(),
                sub_family: None,
                kind: "checkpoint".into(),
                modality: "image".into(),
                nsfw: None,
                description: None,
                tags: Vec::new(),
                license: None,
                page_url: None,
                thumbnail_url: None,
                size_bytes: Some(2),
                supported: true,
                trained_words: Vec::new(),
                primary_filename_rel: primary_rel.into(),
                primary_size_bytes: None,
                low_noise_filename_rel: None,
                low_noise_size_bytes: None,
                written_at: 0,
            },
        )
        .unwrap();

        // Sidecar-first resolution succeeds without any live HF/Civitai call.
        let installed = ensure_catalog_model(&mut config, "cv:1075446")
            .await
            .unwrap();
        assert!(installed, "installed sidecar must resolve to Ok(true)");
        let synth = config
            .models
            .get("cv:1075446")
            .expect("catalog id must be injected into config.models");
        assert_eq!(synth.family.as_deref(), Some("sdxl"));
        assert_eq!(synth.transformer.as_deref(), primary_path.to_str());
    }
}
