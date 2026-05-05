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
use mold_core::download::sanitize_recipe_id;
use mold_core::{Config, ModelConfig, ModelPaths};
use mold_db::catalog::CatalogRow;
use mold_db::MetadataDb;

/// True if `input` has the structural shape of a catalog ID
/// (`cv:<civitai-version-id>` or `hf:<author>/<name>`). Pure shape
/// check — does not consult the catalog DB.
pub fn looks_like_catalog_id(input: &str) -> bool {
    input.starts_with("cv:") || input.starts_with("hf:")
}

/// Look up a catalog ID in the DB. Returns `Ok(None)` when the input
/// isn't a catalog ID OR the row doesn't exist; bubbles real DB errors.
pub fn lookup_catalog_row(db: &MetadataDb, id: &str) -> Result<Option<CatalogRow>> {
    if !looks_like_catalog_id(id) {
        return Ok(None);
    }
    db.catalog_get(id)
}

/// Synthesize a `ModelConfig` for a catalog row, mirroring the on-disk
/// layout that `mold pull <id>` writes to.
///
/// For single-file Civitai checkpoints (Bundling::SingleFile) the
/// resulting `ModelConfig` sets `transformer = vae = primary .safetensors`.
/// That's the duck-type the inference factory uses to dispatch to the
/// `from_single_file` constructors shipped in #271 (`is_single_file(paths)
/// = paths.transformer == paths.vae && extension == .safetensors`).
///
/// Companion paths (clip-l tokenizer for SD1.5, clip-l + clip-g
/// tokenizers for SDXL) come from `ModelPaths::resolve("<companion-name>",
/// config)` so we don't reimplement manifest path rendering. Companions
/// must already be on disk (the catalog pull flow guarantees this by
/// pulling them companion-first before the primary).
pub fn synthesize_model_config(
    row: &CatalogRow,
    models_dir: &Path,
    config: &Config,
) -> Result<ModelConfig> {
    let recipe: mold_catalog::entry::DownloadRecipe = serde_json::from_str(&row.download_recipe)
        .map_err(|e| {
            anyhow::anyhow!("catalog row {} has malformed download_recipe: {e}", row.id)
        })?;

    let primary = recipe
        .files
        .first()
        .ok_or_else(|| anyhow::anyhow!("catalog row {} has empty download_recipe", row.id))?;

    // Reproduce the path computation that `fetch_recipe` wrote to disk:
    // `<models_dir>/<sanitized-id>/<rendered-dest>`.
    let sanitized = sanitize_recipe_id(&row.id);
    let (author, name) = match row.source_id.split_once('/') {
        Some((a, n)) => (a, n),
        None => ("", row.source_id.as_str()),
    };
    let rendered_dest =
        mold_catalog::entry::render_recipe_dest(&primary.dest, &row.family, author, name);
    let primary_path = models_dir.join(&sanitized).join(&rendered_dest);
    let primary_str = primary_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("synthesized path is not valid UTF-8: {primary_path:?}"))?
        .to_string();

    let mut cfg = ModelConfig {
        family: Some(row.family.clone()),
        ..Default::default()
    };

    // Single-file vs. separated dispatch. Civitai checkpoints in this
    // catalog are always SingleFile (the manifest path covers separated
    // diffusers layouts under `[models]`).
    let is_single_file = row.bundling == "single-file";
    if is_single_file {
        cfg.transformer = Some(primary_str.clone());
        cfg.vae = Some(primary_str);
    } else {
        cfg.transformer = Some(primary_str);
        // Separated layouts must declare their VAE via the recipe; for
        // phase 2.5 we only validate single-file. Surface a clear error
        // rather than silently producing a broken config.
        anyhow::bail!(
            "catalog row {} has bundling={:?} which is not yet wired into the run bridge \
             (phase 2.5 supports single-file only)",
            row.id,
            row.bundling,
        );
    }

    // Pull companion paths from the manifest path (each companion lives
    // under its canonical manifest name and is populated when the catalog
    // pull ran). The single-file SDXL/SD1.5 backends only need tokenizers;
    // the encoder weights are bundled in the primary safetensors itself.
    populate_companion_paths(&mut cfg, &row.family, row.sub_family.as_deref(), config)?;

    Ok(cfg)
}

/// For each canonical companion this family declares, look up the
/// companion's resolved manifest paths and copy the relevant token /
/// encoder fields onto `cfg`. Errors out if a required companion is
/// missing, naming the companion so the user knows what to pull.
fn populate_companion_paths(
    cfg: &mut ModelConfig,
    family: &str,
    sub_family: Option<&str>,
    config: &Config,
) -> Result<()> {
    use mold_catalog::companions::companions_for;
    use mold_catalog::entry::Bundling;
    use mold_catalog::families::Family;

    let fam = match family {
        "sd15" => Family::Sd15,
        "sdxl" => Family::Sdxl,
        "flux" => Family::Flux,
        "flux2" => Family::Flux2,
        "z-image" => Family::ZImage,
        "ltx-video" => Family::LtxVideo,
        "ltx2" => Family::Ltx2,
        "qwen-image" => Family::QwenImage,
        "wuerstchen" => Family::Wuerstchen,
        other => anyhow::bail!("catalog family {other:?} not supported by the run bridge"),
    };

    // Best-effort: skip companions whose paths aren't resolvable yet. The
    // single-file engine dispatch surfaces a precise error ("requires a
    // companion-pulled clip_tokenizer") when a *required* field is still
    // None at engine-construction time, so we don't need to second-guess
    // here. This matters because the SDXL/SD1.5 VAE companions are pulled
    // for future external-VAE support but aren't actually needed by the
    // current single-file dispatch (the VAE is embedded in the primary).
    for companion in companions_for(fam, sub_family, Bundling::SingleFile) {
        if let Some(paths) = ModelPaths::resolve(&companion, config) {
            copy_companion_into_cfg(cfg, &companion, &paths);
        }
    }
    Ok(())
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
        _ => {}
    }
}

/// Top-level installer: if `id` is a known catalog row, synthesize a
/// `ModelConfig` and insert it into `config.models` under the same key.
/// Returns `true` when an entry was installed.
///
/// Caller takes `&mut Config` and re-uses the same instance through the
/// rest of the run flow so `ModelPaths::resolve(id, config)` finds the
/// synthesized entry.
pub fn install_catalog_model_with_db(
    db: &MetadataDb,
    config: &mut Config,
    id: &str,
) -> Result<bool> {
    let Some(row) = lookup_catalog_row(db, id)? else {
        return Ok(false);
    };
    let models_dir = config.resolved_models_dir();
    let synth = synthesize_model_config(&row, &models_dir, config)?;
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

    /// Realistic shape of the JSON `download_recipe` column the catalog
    /// scanner writes for a single-file Civitai SDXL row. The `dest`
    /// template ships literal `{family}` because the catalog is
    /// authored once per-row, then rendered per-pull.
    fn juggernaut_recipe_json() -> &'static str {
        r#"{
  "files": [
    {
      "url": "https://civitai.com/api/download/models/1759168",
      "dest": "{family}/civitai/1759168/juggernautXL_ragnarokBy.safetensors",
      "sha256": "DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF",
      "size_bytes": 6938040788
    }
  ],
  "needs_token": "civitai"
}"#
    }

    fn juggernaut_row() -> mold_db::catalog::CatalogRow {
        mold_db::catalog::CatalogRow {
            id: "cv:1759168".into(),
            source: "civitai".into(),
            source_id: "1759168".into(),
            name: "Juggernaut XL Ragnarok".into(),
            author: Some("RunDiffusion".into()),
            family: "sdxl".into(),
            family_role: "finetune".into(),
            sub_family: None,
            modality: "image".into(),
            kind: "checkpoint".into(),
            file_format: "safetensors".into(),
            bundling: "single-file".into(),
            size_bytes: Some(6_938_040_788),
            download_count: 12_345,
            rating: None,
            likes: 0,
            nsfw: 0,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: None,
            tags: Some("[]".into()),
            companions: Some(r#"["clip-l","clip-g","sdxl-vae"]"#.into()),
            download_recipe: juggernaut_recipe_json().into(),
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
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

        let row = juggernaut_row();
        let synth =
            synthesize_model_config(&row, std::path::Path::new(models_dir), &config).unwrap();

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
    fn synthesize_model_config_for_sdxl_single_file_civitai_row() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let models_dir = "/tmp/mold-test-models";
        let mut config = explicit_config(models_dir);
        stub_companion_paths(&mut config, models_dir);

        let row = juggernaut_row();
        let synth =
            synthesize_model_config(&row, std::path::Path::new(models_dir), &config).unwrap();

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
    fn synthesize_model_config_rejects_unknown_family() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let mut row = juggernaut_row();
        row.family = "made-up-family".into();
        let config = explicit_config("/tmp/mold-test-models");
        let err =
            synthesize_model_config(&row, std::path::Path::new("/tmp/mold-test-models"), &config)
                .unwrap_err();
        assert!(
            err.to_string().contains("made-up-family"),
            "should name the offending family, got: {err}",
        );
    }

    #[test]
    fn synthesize_model_config_rejects_separated_bundling() {
        // The bridge currently only handles single-file (phase 2.5 scope).
        // A separated catalog row should error rather than producing a
        // broken config that points its `vae` at the transformer file.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let mut row = juggernaut_row();
        row.bundling = "separated".into();
        let config = explicit_config("/tmp/mold-test-models");
        let err =
            synthesize_model_config(&row, std::path::Path::new("/tmp/mold-test-models"), &config)
                .unwrap_err();
        assert!(
            err.to_string().contains("single-file"),
            "should explain the supported bundling, got: {err}",
        );
    }

    #[test]
    fn lookup_catalog_row_returns_none_for_non_catalog_input() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        assert!(lookup_catalog_row(&db, "flux-dev:q4").unwrap().is_none());
        assert!(lookup_catalog_row(&db, "a cat").unwrap().is_none());
    }

    #[test]
    fn lookup_catalog_row_finds_row_when_present() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        db.catalog_upsert("sdxl", &[juggernaut_row()]).unwrap();
        let found = lookup_catalog_row(&db, "cv:1759168").unwrap().unwrap();
        assert_eq!(found.id, "cv:1759168");
        assert_eq!(found.family, "sdxl");
    }

    #[test]
    fn install_catalog_model_with_db_inserts_synthesized_entry() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();

        let models_dir = "/tmp/mold-test-models";
        let mut config = explicit_config(models_dir);
        stub_companion_paths(&mut config, models_dir);

        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        db.catalog_upsert("sdxl", &[juggernaut_row()]).unwrap();

        let installed = install_catalog_model_with_db(&db, &mut config, "cv:1759168").unwrap();
        assert!(installed, "row was in the catalog DB");

        let entry = config
            .models
            .get("cv:1759168")
            .expect("config.models has the synthesized entry");
        assert_eq!(entry.family.as_deref(), Some("sdxl"));
        assert_eq!(
            entry.transformer.as_deref(),
            Some(
                format!(
                    "{models_dir}/cv-1759168/sdxl/civitai/1759168/juggernautXL_ragnarokBy.safetensors"
                )
                .as_str()
            ),
        );
    }

    #[test]
    fn install_catalog_model_with_db_returns_false_for_unknown_id() {
        let mut config = explicit_config("/tmp/mold-test-models");
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let installed =
            install_catalog_model_with_db(&db, &mut config, "cv:does-not-exist").unwrap();
        assert!(!installed);
        assert!(!config.models.contains_key("cv:does-not-exist"));
    }

    // ── Flux.2 catalog bridge ────────────────────────────────────────────

    fn flux2_recipe_json(version_id: &str, file_name: &str) -> String {
        format!(
            r#"{{
  "files": [
    {{
      "url": "https://civitai.com/api/download/models/{version_id}",
      "dest": "{{family}}/civitai/{version_id}/{file_name}",
      "sha256": "DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF",
      "size_bytes": 12345678
    }}
  ],
  "needs_token": "civitai"
}}"#
        )
    }

    fn flux2_klein_9b_row() -> mold_db::catalog::CatalogRow {
        mold_db::catalog::CatalogRow {
            id: "cv:2759597".into(),
            source: "civitai".into(),
            source_id: "2759597".into(),
            name: "Miraclein NSFW [Flux2Klein]".into(),
            author: Some("someone".into()),
            family: "flux2".into(),
            family_role: "finetune".into(),
            sub_family: Some("klein-9b".into()),
            modality: "image".into(),
            kind: "checkpoint".into(),
            file_format: "safetensors".into(),
            bundling: "single-file".into(),
            size_bytes: Some(12_345_678),
            download_count: 0,
            rating: None,
            likes: 0,
            nsfw: 0,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: None,
            tags: Some("[]".into()),
            companions: Some(r#"["flux2-te-9b","flux2-vae"]"#.into()),
            download_recipe: flux2_recipe_json("2759597", "miraclein.safetensors"),
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
        }
    }

    fn flux2_klein_4b_row() -> mold_db::catalog::CatalogRow {
        let mut row = flux2_klein_9b_row();
        row.id = "cv:2612554".into();
        row.source_id = "2612554".into();
        row.name = "Flux.2 Klein 4B finetune".into();
        row.sub_family = Some("klein-4b".into());
        row.companions = Some(r#"["flux2-te","flux2-vae"]"#.into());
        row.download_recipe = flux2_recipe_json("2612554", "klein4b.safetensors");
        row
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

        let row = flux2_klein_9b_row();
        let synth =
            synthesize_model_config(&row, std::path::Path::new(models_dir), &config).unwrap();

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

        let row = flux2_klein_4b_row();
        let synth =
            synthesize_model_config(&row, std::path::Path::new(models_dir), &config).unwrap();

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
