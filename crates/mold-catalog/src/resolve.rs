//! Canonical disk-aware catalog resolver.
//!
//! Turns a pure [`CatalogModelIntent`](crate::synthesis::CatalogModelIntent)
//! into a runtime `mold_core::ModelConfig`, running every disk-dependent
//! decision here: the FLUX bundled-VAE header probe, companion path lookup,
//! primary-file completeness. This is the single source of truth previously
//! duplicated (and drifted) between `mold-cli`'s `catalog_bridge.rs` and
//! `mold-server`'s `model_manager.rs`.
//!
//! Two callers, two policies. The **server** resolves at engine-load time and
//! wants a hard error the moment any required companion is missing
//! ([`MissingCompanionPolicy::Fail`] + `require_primary_present: true`) so its
//! retry path can distinguish "still downloading" from "broken config". The
//! **CLI** resolves right after `pull_and_configure` and prefers to
//! log-and-continue ([`MissingCompanionPolicy::WarnAndSkip`] +
//! `require_primary_present: false`), letting the engine-load surface report
//! any genuinely missing field with a precise message.

use std::path::Path;

use mold_core::{Config, ModelConfig, ModelPaths};

use crate::entry::{Bundling, Kind};
use crate::families::Family;
use crate::synthesis::{family_bundles_vae_unconditionally, CatalogModelIntent, CompanionIntent};

/// True if `input` has the structural shape of a catalog ID
/// (`cv:<civitai-version-id>` or `hf:<author>/<name>`). Pure shape check —
/// does not consult any source.
pub fn looks_like_catalog_id(input: &str) -> bool {
    input.starts_with("cv:") || input.starts_with("hf:")
}

/// What to do when a companion's manifest paths can't be resolved on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingCompanionPolicy {
    /// Return a typed [`ResolveError`] naming the first missing required
    /// companion. Used by the server so its engine-load retry path can act.
    Fail,
    /// Log a `tracing::warn!` per skipped companion and keep going. Used by
    /// the CLI, whose pull flow already ran; the engine-load surface picks
    /// up any real missing field.
    WarnAndSkip,
}

/// Knobs controlling how strict [`resolve_intent_to_model_config`] is.
#[derive(Debug, Clone, Copy)]
pub struct ResolveOptions {
    pub missing_companions: MissingCompanionPolicy,
    /// When true, resolution fails early with [`ResolveError::PrimaryFileMissing`]
    /// unless the primary checkpoint is present (and, for `cv:*`, complete
    /// per its sidecar). The server relies on this ordering; the CLI leaves
    /// it off because its resolve runs after the download completes.
    pub require_primary_present: bool,
}

/// Errors from the disk-aware resolution stage that turns a pure
/// [`CatalogModelIntent`] into a runtime `ModelConfig`.
#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    /// A required companion's manifest entry isn't in the user's
    /// `Config.models` map. This is a real config bug — the user hasn't
    /// installed (or has a broken manifest entry for) a base model the
    /// catalog entry depends on.
    #[error(
        "catalog model '{model}' is missing required component '{companion}'.\n\
         The manifest entry for '{companion}' is not present in your config — \
         install the matching base model first."
    )]
    CompanionConfigMissing {
        model: String,
        companion: &'static str,
    },

    /// A required companion's file isn't on disk yet. Transient — caller
    /// hasn't finished `mold pull`-ing this entry.
    #[error(
        "catalog model '{model}' is missing required component '{companion}'.\n\
         The companion file isn't on disk yet. Run `mold pull {model}` to fetch \
         the primary checkpoint AND its companion files."
    )]
    CompanionFileMissing {
        model: String,
        companion: &'static str,
    },

    /// The catalog primary checkpoint is absent or only partially
    /// downloaded. Transient; repairable by `mold pull <catalog-id>`.
    #[error(
        "catalog model '{model}' is missing its primary checkpoint.\n\
         Run `mold pull {model}` to fetch or repair the primary checkpoint."
    )]
    PrimaryFileMissing { model: String },

    /// The bundled-VAE probe couldn't be run (e.g. malformed
    /// safetensors header).
    #[error("probe of checkpoint {path} for bundled VAE failed: {source}")]
    VaeProbeFailed {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The intent carries a family slug that doesn't map to a known
    /// [`Family`]. Should never fire for intents produced by
    /// `synthesize_intent` / [`installed_intent_from_sidecar`] (both only
    /// emit known slugs); the typed error replaces a former silent
    /// fall-through so a future stale slug surfaces loudly instead of being
    /// mis-treated as FLUX.
    #[error("catalog model '{model}' has an unknown family slug '{slug}'")]
    UnknownFamily { model: String, slug: String },
}

/// Disk-aware resolution: turn a pure [`CatalogModelIntent`] into a
/// `ModelConfig` ready for engine load.
pub fn resolve_intent_to_model_config(
    model_name: &str,
    intent: &CatalogModelIntent,
    config: &Config,
    options: ResolveOptions,
) -> Result<ModelConfig, ResolveError> {
    if options.require_primary_present && !catalog_primary_is_complete(model_name, intent, config) {
        return Err(ResolveError::PrimaryFileMissing {
            model: model_name.to_string(),
        });
    }

    let primary_str = intent
        .primary_recipe_path
        .to_str()
        .expect("synthesize_intent guarantees UTF-8 paths")
        .to_string();

    let mut cfg = ModelConfig {
        family: Some(intent.family.clone()),
        ..Default::default()
    };
    apply_catalog_runtime_defaults(&mut cfg, intent);
    cfg.transformer = Some(primary_str.clone());

    // FLUX and LTX-2 both have mixed bundling — peek the safetensors header
    // to decide whether the primary carries a VAE or a companion must fill it.
    let family = family_for_slug(model_name, &intent.family)?;
    let probe_says_bundled =
        if matches!(family, Family::Flux | Family::Ltx2) && intent.primary_recipe_path.exists() {
            Some(
                mold_core::safetensors_probe::single_file_bundles_vae(&intent.primary_recipe_path)
                    .map_err(|e| ResolveError::VaeProbeFailed {
                        path: intent.primary_recipe_path.display().to_string(),
                        source: e,
                    })?,
            )
        } else if matches!(family, Family::Flux | Family::Ltx2) {
            // Primary not on disk yet — defer. The required family/version VAE
            // companion must populate cfg.vae below.
            None
        } else if family_bundles_vae_unconditionally(family) {
            // SDXL / SD1.5 always bundle the VAE.
            Some(true)
        } else {
            // Flux.2 / Z-Image / LTX-Video / Wan always need a separate VAE
            // companion.
            Some(false)
        };
    if probe_says_bundled == Some(true) {
        cfg.vae = Some(primary_str);
    }
    if let Some(vae_recipe_path) = &intent.vae_recipe_path {
        cfg.vae = Some(
            vae_recipe_path
                .to_str()
                .expect("synthesize_intent guarantees UTF-8 paths")
                .to_string(),
        );
    }
    if !intent.text_encoder_recipe_paths.is_empty() {
        cfg.text_encoder_files = Some(
            intent
                .text_encoder_recipe_paths
                .iter()
                .map(|path| {
                    path.to_str()
                        .expect("synthesize_intent guarantees UTF-8 paths")
                        .to_string()
                })
                .collect(),
        );
    }
    // probe_says_bundled == Some(false) ⇒ leave cfg.vae None for the
    // VAE-companion arm in copy_companion_into_model_config to fill. The
    // None case (FLUX, file not yet on disk) is the same.

    let mut missing_required: Option<&'static str> = None;
    for companion in &intent.companions {
        let static_name =
            known_companion_static(&companion.name).expect("companion name in registry");
        match ModelPaths::resolve(static_name, config) {
            Some(paths) => copy_companion_into_model_config(&mut cfg, static_name, &paths),
            None => match options.missing_companions {
                MissingCompanionPolicy::Fail => {
                    if companion.required && missing_required.is_none() {
                        missing_required = Some(static_name);
                    }
                }
                MissingCompanionPolicy::WarnAndSkip => {
                    tracing::warn!(
                        target: "catalog.resolve",
                        model = %model_name,
                        companion = static_name,
                        "skipping catalog companion that could not be resolved to on-disk paths"
                    );
                }
            },
        }
    }

    if let Some(missing) = missing_required {
        return Err(classify_missing_companion(model_name, missing, config));
    }

    Ok(cfg)
}

/// Populate the `ModelConfig` fields a single companion contributes.
///
/// The arm set is the union of the historical CLI + server copies, with the
/// server's grouping as canonical.
pub fn copy_companion_into_model_config(
    cfg: &mut ModelConfig,
    companion: &str,
    paths: &ModelPaths,
) {
    let to_str = |p: &std::path::PathBuf| p.to_str().map(str::to_owned);
    match companion {
        "clip-l" => {
            cfg.clip_encoder = to_str(&paths.transformer);
            cfg.clip_tokenizer = paths.clip_tokenizer.as_ref().and_then(to_str);
        }
        "clip-g" => {
            cfg.clip_encoder_2 = to_str(&paths.transformer);
            cfg.clip_tokenizer_2 = paths
                .clip_tokenizer
                .as_ref()
                .and_then(to_str)
                .or_else(|| cfg.clip_tokenizer.clone());
        }
        "sdxl-vae" | "sd-vae-ft-mse" => {}
        // Civitai FLUX fine-tune convention is split: some bundle the VAE in
        // the same .safetensors as the transformer, some are transformer-only.
        // The probe above clears cfg.vae for the transformer-only case; here
        // we populate it from the companion's resolved path. When the bundle
        // DID include a VAE, cfg.vae is already set to the primary checkpoint
        // and we leave it alone — the guarded arm doesn't fire and the
        // catch-all (no-op) handles it.
        "flux-vae" if cfg.vae.is_none() => {
            cfg.vae = to_str(&paths.transformer);
        }
        "ltx-video-vae" | "ltx2-vae" | "ltx2.3-vae" | "flux2-vae" => {
            cfg.vae = to_str(&paths.transformer);
        }
        // Both Wan VAE companions declare their single file as a Transformer
        // component, the same shape the LTX VAE companions use, so the path
        // arrives in `paths.transformer`. Which of the two got pushed is
        // decided in `companions_for`; by here it is already the right one.
        "wan21-vae" | "wan22-vae" => {
            cfg.vae = to_str(&paths.transformer);
        }
        "t5-v1_1-xxl" => {
            cfg.t5_encoder = to_str(&paths.transformer);
            cfg.t5_tokenizer = paths.t5_tokenizer.as_ref().and_then(to_str);
        }
        "z-image-te" | "flux2-te" | "flux2-te-9b" => {
            if companion == "z-image-te" && cfg.vae.is_none() {
                cfg.vae = to_str(&paths.vae);
            }
            if companion != "z-image-te" || cfg.text_encoder_files.is_none() {
                cfg.text_encoder_files = paths
                    .text_encoder_files
                    .iter()
                    .filter_map(to_str)
                    .collect::<Vec<_>>()
                    .into();
            }
            cfg.text_tokenizer = paths.text_tokenizer.as_ref().and_then(to_str);
        }
        "ltx2-te" => {
            // Gemma 3 12B for LTX-2. The runtime calls `gemma_root` which
            // takes the parent directory of `text_encoder_files[0]`, so
            // populating that vec is sufficient — the manifest tags every
            // Gemma file (including tokenizer files) as TextEncoder so the
            // tokenizer rides along.
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_str)
                .collect::<Vec<_>>()
                .into();
        }
        "ltx2.3-text-projection" => {
            let mut files = cfg.text_encoder_files.take().unwrap_or_default();
            if let Some(path) = to_str(&paths.transformer) {
                if !files.contains(&path) {
                    files.push(path);
                }
            }
            cfg.text_encoder_files = Some(files);
        }
        // UMT5-XXL. `WanEngine` reads `text_encoder_files` first and falls
        // back to `t5_encoder`, and reads `text_tokenizer` before
        // `t5_tokenizer`; populating the generic pair is what the engine
        // actually looks for, and keeps the T5 slots free for families that
        // genuinely use T5.
        "wan-umt5" => {
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_str)
                .collect::<Vec<_>>()
                .into();
            cfg.text_tokenizer = paths.text_tokenizer.as_ref().and_then(to_str);
        }
        "qwen-image-runtime" => {
            cfg.vae = to_str(&paths.vae);
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_str)
                .collect::<Vec<_>>()
                .into();
            cfg.text_tokenizer = paths.text_tokenizer.as_ref().and_then(to_str);
        }
        "wuerstchen-runtime" => {
            cfg.vae = to_str(&paths.vae);
            cfg.decoder = paths.decoder.as_ref().and_then(to_str);
            cfg.clip_encoder = paths.clip_encoder.as_ref().and_then(to_str);
            cfg.clip_tokenizer = paths.clip_tokenizer.as_ref().and_then(to_str);
            cfg.clip_encoder_2 = paths.clip_encoder_2.as_ref().and_then(to_str);
            cfg.clip_tokenizer_2 = paths.clip_tokenizer_2.as_ref().and_then(to_str);
        }
        _ => {}
    }
}

/// Overlay per-family runtime defaults (dimensions, steps, guidance, schnell
/// flag) onto a freshly-built `ModelConfig`, without clobbering values the
/// caller already set.
pub fn apply_catalog_runtime_defaults(cfg: &mut ModelConfig, intent: &CatalogModelIntent) {
    let defaults =
        crate::defaults::runtime_defaults_for_family(&intent.family, intent.sub_family.as_deref());
    cfg.default_width.get_or_insert(defaults.width);
    cfg.default_height.get_or_insert(defaults.height);
    cfg.default_steps.get_or_insert(defaults.steps);
    cfg.default_guidance.get_or_insert(defaults.guidance);
    if let Some(is_schnell) = defaults.is_schnell {
        cfg.is_schnell.get_or_insert(is_schnell);
    }
}

/// Synthesize a [`CatalogModelIntent`] from an installed per-install sidecar,
/// so an already-downloaded `cv:*` checkpoint resolves without a live lookup.
/// Returns `None` when no matching, loadable checkpoint sidecar is present.
pub fn installed_intent_from_sidecar(
    models_dir: &Path,
    model_name: &str,
) -> Option<CatalogModelIntent> {
    let (sidecar_dir, sidecar) = crate::sidecar::walk_sidecars(models_dir)
        .into_iter()
        .find(|(_, sidecar)| sidecar.id == model_name)?;
    if sidecar.kind != "checkpoint" {
        return None;
    }
    if crate::sidecar::primary_looks_like_auxiliary(&sidecar) {
        return None;
    }
    let primary_recipe_path = crate::sidecar::primary_path_if_present(&sidecar_dir, &sidecar)?;
    let family = Family::from_str(&sidecar.family).ok()?;
    let companions = crate::companions::companions_for(
        family,
        sidecar.sub_family.as_deref(),
        Bundling::SingleFile,
        Kind::Checkpoint,
    )
    .into_iter()
    .map(|name| CompanionIntent {
        name,
        required: true,
    })
    .collect();

    Some(CatalogModelIntent {
        family: sidecar.family,
        sub_family: sidecar.sub_family,
        primary_recipe_path,
        vae_recipe_path: None,
        text_encoder_recipe_paths: Vec::new(),
        companions,
        bundling: Bundling::SingleFile,
    })
}

/// All companions emitted by `companions_for` are statically known names
/// from the registry. Pinning resolution to the same `&'static str` lets
/// error messages carry it without an allocation and guarantees the match in
/// [`copy_companion_into_model_config`] never sees a typo.
/// Hand-maintained mirror of [`crate::companions::COMPANIONS`], needed
/// because the resolver wants a `&'static str` it can hand to
/// `ModelPaths::resolve`. `resolve_intent_to_model_config` `expect`s on
/// this, so a registry entry missing here is a panic at install time, not
/// a degraded result — `every_registered_companion_has_a_static_name`
/// keeps the two in step.
fn known_companion_static(name: &str) -> Option<&'static str> {
    Some(match name {
        "clip-l" => "clip-l",
        "clip-g" => "clip-g",
        "sdxl-vae" => "sdxl-vae",
        "sd-vae-ft-mse" => "sd-vae-ft-mse",
        "flux-vae" => "flux-vae",
        "flux2-te" => "flux2-te",
        "flux2-te-9b" => "flux2-te-9b",
        "flux2-vae" => "flux2-vae",
        "z-image-te" => "z-image-te",
        "ltx-video-vae" => "ltx-video-vae",
        "ltx2-vae" => "ltx2-vae",
        "ltx2.3-vae" => "ltx2.3-vae",
        "ltx2.3-text-projection" => "ltx2.3-text-projection",
        "ltx2-te" => "ltx2-te",
        "wan-umt5" => "wan-umt5",
        "wan21-vae" => "wan21-vae",
        "wan22-vae" => "wan22-vae",
        "qwen-image-runtime" => "qwen-image-runtime",
        "wuerstchen-runtime" => "wuerstchen-runtime",
        "t5-v1_1-xxl" => "t5-v1_1-xxl",
        _ => return None,
    })
}

fn family_for_slug(model_name: &str, slug: &str) -> Result<Family, ResolveError> {
    Family::from_str(slug).map_err(|_| ResolveError::UnknownFamily {
        model: model_name.to_string(),
        slug: slug.to_string(),
    })
}

fn catalog_primary_is_complete(
    model_name: &str,
    intent: &CatalogModelIntent,
    config: &Config,
) -> bool {
    if !intent.primary_recipe_path.is_file() {
        return false;
    }
    if model_name.starts_with("cv:") {
        let models_dir = config.resolved_models_dir();
        let sc_path = crate::sidecar::civitai_sidecar_path(&models_dir, model_name);
        if let Ok(sidecar) = crate::sidecar::read_sidecar(&sc_path) {
            let Some(sidecar_dir) = sc_path.parent() else {
                return false;
            };
            return crate::sidecar::primary_path_if_present(sidecar_dir, &sidecar).is_some();
        }
    }
    true
}

/// Decide whether a missing companion is "config bug" vs "file not yet
/// downloaded": absent from `config.models` ⇒ config bug; present but its
/// paths point at a missing file ⇒ not pulled yet.
fn classify_missing_companion(
    model_name: &str,
    companion: &'static str,
    config: &Config,
) -> ResolveError {
    if !config.models.contains_key(companion) {
        ResolveError::CompanionConfigMissing {
            model: model_name.to_string(),
            companion,
        }
    } else {
        ResolveError::CompanionFileMissing {
            model: model_name.to_string(),
            companion,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::{
        CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, LicenseFlags, Modality,
        RecipeFile, RecipeFileRole, Source, TokenKind,
    };
    use crate::synthesis::synthesize_intent;
    use std::path::PathBuf;

    /// Resolver tests touch `MOLD_HOME` / `MOLD_*_PATH` env, which
    /// `ModelPaths::resolve` reads. Serialize them so parallel `cargo test`
    /// doesn't cross-contaminate.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const FAIL: ResolveOptions = ResolveOptions {
        missing_companions: MissingCompanionPolicy::Fail,
        require_primary_present: true,
    };
    const WARN: ResolveOptions = ResolveOptions {
        missing_companions: MissingCompanionPolicy::WarnAndSkip,
        require_primary_present: false,
    };

    /// `resolve_intent_to_model_config` calls `.expect("companion name in
    /// registry")` on this lookup, so a companion that `companions_for` can
    /// push but this table does not know is not a missing feature — it is a
    /// panic on the install path. Nothing enforced that the two agreed until
    /// this test; adding the Wan companions is what surfaced the gap.
    #[test]
    fn every_registered_companion_has_a_static_name() {
        for companion in crate::companions::COMPANIONS {
            assert!(
                known_companion_static(companion.canonical_name).is_some(),
                "`{}` is in COMPANIONS but missing from known_companion_static — \
                 resolving an entry that needs it would panic",
                companion.canonical_name
            );
        }
    }

    /// Pin MOLD_HOME to a path with no real `.hf-cache/` so
    /// `ModelPaths::resolve` can't discover an unrelated installed companion
    /// from the dev machine and inject it into the synthesized config.
    struct MoldHomeGuard {
        prev: Option<String>,
    }
    impl Drop for MoldHomeGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }
    fn pin_mold_home(path: &Path) -> MoldHomeGuard {
        let prev = std::env::var("MOLD_HOME").ok();
        std::env::set_var("MOLD_HOME", path.to_string_lossy().as_ref());
        MoldHomeGuard { prev }
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

    fn config_in(models_dir: &Path) -> Config {
        Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        }
    }

    fn empty_paths() -> ModelPaths {
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: PathBuf::new(),
            transformer_shards: Vec::new(),
            vae: PathBuf::new(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        }
    }

    fn write_safetensors_with_keys(path: &Path, keys: &[&str]) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        for key in keys {
            header.insert(
                (*key).to_string(),
                serde_json::json!({ "dtype": "F32", "shape": [1], "data_offsets": [0, 4] }),
            );
        }
        let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).expect("create fixture");
        f.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_json).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
    }

    fn base_entry() -> CatalogEntry {
        CatalogEntry {
            id: CatalogId::from("cv:1"),
            source: Source::Civitai,
            source_id: "1".into(),
            name: "Test".into(),
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
            companions: vec!["clip-l".into(), "clip-g".into(), "sdxl-vae".into()],
            download_recipe: DownloadRecipe {
                files: vec![RecipeFile {
                    url: "https://civitai.com/api/download/models/1".into(),
                    dest: "{family}/civitai/1/model.safetensors".into(),
                    sha256: None,
                    size_bytes: Some(12_000_000_000),
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

    fn juggernaut_entry() -> CatalogEntry {
        let mut e = base_entry();
        e.id = CatalogId::from("cv:1759168");
        e.source_id = "1759168".into();
        e.name = "Juggernaut XL Ragnarok".into();
        e.family = Family::Sdxl;
        e.companions = vec!["clip-l".into(), "clip-g".into(), "sdxl-vae".into()];
        e.download_recipe.files[0].dest =
            "{family}/civitai/1759168/juggernautXL_ragnarokBy.safetensors".into();
        e
    }

    fn flux_unet_only_entry(version_id: &str, file_name: &str) -> CatalogEntry {
        let mut e = base_entry();
        e.id = CatalogId::from(format!("cv:{version_id}"));
        e.source_id = version_id.to_string();
        e.name = "FLUX Unet-only fine-tune".into();
        e.family = Family::Flux;
        e.companions = vec!["t5-v1_1-xxl".into(), "clip-l".into(), "flux-vae".into()];
        e.download_recipe.files[0].url =
            format!("https://civitai.com/api/download/models/{version_id}");
        e.download_recipe.files[0].dest = format!("{{family}}/civitai/{version_id}/{file_name}");
        e
    }

    /// Stub clip-l + clip-g companion manifest entries pointing at
    /// `{models_dir}/{name}/…`, each with its own tokenizer.
    fn stub_sdxl_companions(config: &mut Config, models_dir: &str) {
        let clip_l_dir = format!("{models_dir}/clip-l");
        config.models.insert(
            "clip-l".into(),
            ModelConfig {
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
            ModelConfig {
                family: Some("clip-g".into()),
                transformer: Some(format!("{clip_g_dir}/open_clip_model.safetensors")),
                vae: Some(format!("{clip_g_dir}/open_clip_model.safetensors")),
                clip_tokenizer: Some(format!("{clip_g_dir}/tokenizer.json")),
                ..Default::default()
            },
        );
        // sdxl-vae is a required SDXL companion. Its copy arm is a no-op
        // (SDXL bundles the VAE), but it must be resolvable so the Fail
        // policy's required-companion check passes.
        let vae_dir = format!("{models_dir}/sdxl-vae");
        config.models.insert(
            "sdxl-vae".into(),
            ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!("{vae_dir}/sdxl_vae.safetensors")),
                vae: Some(format!("{vae_dir}/sdxl_vae.safetensors")),
                ..Default::default()
            },
        );
    }

    /// Stub the FLUX companions (flux-vae, clip-l, t5) at on-disk files under
    /// `models_dir`, optionally creating the flux-vae file.
    fn stub_flux_companions(config: &mut Config, models_dir: &Path, flux_vae_present: bool) {
        let vae_path = models_dir.join("flux-vae/ae.safetensors");
        std::fs::create_dir_all(vae_path.parent().unwrap()).unwrap();
        if flux_vae_present {
            std::fs::File::create(&vae_path).unwrap();
        }
        config.models.insert(
            "flux-vae".into(),
            ModelConfig {
                family: Some("companion".into()),
                transformer: Some(vae_path.to_string_lossy().into_owned()),
                vae: Some(vae_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        let clip_path = models_dir.join("clip-l/model.safetensors");
        std::fs::create_dir_all(clip_path.parent().unwrap()).unwrap();
        std::fs::File::create(&clip_path).unwrap();
        config.models.insert(
            "clip-l".into(),
            ModelConfig {
                family: Some("companion".into()),
                transformer: Some(clip_path.to_string_lossy().into_owned()),
                vae: Some(clip_path.to_string_lossy().into_owned()),
                clip_tokenizer: Some(format!("{}/clip-l/tokenizer.json", models_dir.display())),
                ..Default::default()
            },
        );
        let t5_path = models_dir.join("t5-v1_1-xxl/t5xxl_fp16.safetensors");
        std::fs::create_dir_all(t5_path.parent().unwrap()).unwrap();
        std::fs::File::create(&t5_path).unwrap();
        config.models.insert(
            "t5-v1_1-xxl".into(),
            ModelConfig {
                family: Some("companion".into()),
                transformer: Some(t5_path.to_string_lossy().into_owned()),
                vae: Some(t5_path.to_string_lossy().into_owned()),
                t5_tokenizer: Some(format!(
                    "{}/t5-v1_1-xxl/tokenizer.json",
                    models_dir.display()
                )),
                ..Default::default()
            },
        );
    }

    // ── shape check (no env) ──────────────────────────────────────────────

    #[test]
    fn looks_like_catalog_id_accepts_civitai_and_hf_shapes() {
        assert!(looks_like_catalog_id("cv:1759168"));
        assert!(looks_like_catalog_id("hf:RunDiffusion/Juggernaut-XL-v9"));
    }

    #[test]
    fn looks_like_catalog_id_rejects_manifest_and_prompt_inputs() {
        assert!(!looks_like_catalog_id("flux-dev"));
        assert!(!looks_like_catalog_id("flux-dev:q4"));
        assert!(!looks_like_catalog_id("a cat"));
        assert!(!looks_like_catalog_id(""));
    }

    // ── copy_companion_into_model_config unit tests (no env) ───────────────

    #[test]
    fn copy_companion_flux_vae_populates_when_unset() {
        let mut cfg = ModelConfig {
            family: Some("flux".into()),
            transformer: Some("/m/cv-x/flux/civitai/x/unet.safetensors".into()),
            vae: None,
            ..Default::default()
        };
        let mut paths = empty_paths();
        paths.transformer = PathBuf::from("/m/flux-vae/ae.safetensors");
        paths.vae = PathBuf::from("/m/flux-vae/ae.safetensors");
        copy_companion_into_model_config(&mut cfg, "flux-vae", &paths);
        assert_eq!(cfg.vae.as_deref(), Some("/m/flux-vae/ae.safetensors"));
    }

    #[test]
    fn copy_companion_flux_vae_does_not_override_bundled() {
        let mut cfg = ModelConfig {
            family: Some("flux".into()),
            transformer: Some("/m/cv-x/flux/civitai/x/full.safetensors".into()),
            vae: Some("/m/cv-x/flux/civitai/x/full.safetensors".into()),
            ..Default::default()
        };
        let mut paths = empty_paths();
        paths.transformer = PathBuf::from("/m/flux-vae/ae.safetensors");
        paths.vae = PathBuf::from("/m/flux-vae/ae.safetensors");
        copy_companion_into_model_config(&mut cfg, "flux-vae", &paths);
        assert_eq!(
            cfg.vae.as_deref(),
            Some("/m/cv-x/flux/civitai/x/full.safetensors"),
            "bundled cfg.vae must survive the flux-vae companion pass"
        );
    }

    #[test]
    fn copy_companion_zimage_populates_vae_and_text_encoder() {
        let mut paths = empty_paths();
        paths.transformer =
            "/models/shared/z-image/text_encoder/model-00001-of-00003.safetensors".into();
        paths.vae = "/models/shared/z-image/vae/diffusion_pytorch_model.safetensors".into();
        paths.text_encoder_files = vec![
            "/models/shared/z-image/text_encoder/model-00001-of-00003.safetensors".into(),
            "/models/shared/z-image/text_encoder/model-00002-of-00003.safetensors".into(),
            "/models/shared/z-image/text_encoder/model-00003-of-00003.safetensors".into(),
        ];
        paths.text_tokenizer = Some("/models/shared/z-image/tokenizer/tokenizer.json".into());
        let mut cfg = ModelConfig::default();
        copy_companion_into_model_config(&mut cfg, "z-image-te", &paths);
        assert_eq!(
            cfg.vae.as_deref(),
            Some("/models/shared/z-image/vae/diffusion_pytorch_model.safetensors")
        );
        assert_eq!(cfg.text_encoder_files.as_ref().unwrap().len(), 3);
        assert_eq!(
            cfg.text_tokenizer.as_deref(),
            Some("/models/shared/z-image/tokenizer/tokenizer.json")
        );
    }

    // ── resolve_intent_to_model_config (env-locked) ────────────────────────

    #[test]
    fn resolves_sdxl_single_file_bundled_vae_and_clip_tokenizers() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let mut config = config_in(models_dir);
        stub_sdxl_companions(&mut config, models_dir.to_str().unwrap());

        let entry = juggernaut_entry();
        let primary =
            models_dir.join("cv-1759168/sdxl/civitai/1759168/juggernautXL_ragnarokBy.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        std::fs::write(&primary, b"ok").unwrap();

        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_model_config("cv:1759168", &intent, &config, FAIL).unwrap();

        assert_eq!(cfg.family.as_deref(), Some("sdxl"));
        // SDXL always bundles the VAE → transformer == vae == primary.
        assert_eq!(cfg.transformer.as_deref(), primary.to_str());
        assert_eq!(cfg.vae.as_deref(), primary.to_str());
        assert_eq!(
            cfg.clip_tokenizer.as_deref(),
            Some(format!("{}/clip-l/tokenizer.json", models_dir.display()).as_str())
        );
        assert_eq!(
            cfg.clip_tokenizer_2.as_deref(),
            Some(format!("{}/clip-g/tokenizer.json", models_dir.display()).as_str())
        );
    }

    #[test]
    fn sdxl_clip_g_tokenizer_falls_back_to_clip_l() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let mut config = config_in(models_dir);
        // clip-g without a tokenizer of its own (matches the real manifest).
        stub_sdxl_companions(&mut config, models_dir.to_str().unwrap());
        config.models.get_mut("clip-g").unwrap().clip_tokenizer = None;

        let entry = juggernaut_entry();
        let primary =
            models_dir.join("cv-1759168/sdxl/civitai/1759168/juggernautXL_ragnarokBy.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        std::fs::write(&primary, b"ok").unwrap();
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_model_config("cv:1759168", &intent, &config, FAIL).unwrap();

        let expected = format!("{}/clip-l/tokenizer.json", models_dir.display());
        assert_eq!(cfg.clip_tokenizer.as_deref(), Some(expected.as_str()));
        assert_eq!(
            cfg.clip_tokenizer_2.as_deref(),
            Some(expected.as_str()),
            "clip_tokenizer_2 falls back to clip-l when clip-g has none"
        );
    }

    #[test]
    fn flux_transformer_only_picks_flux_vae_companion() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let primary = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );
        let mut config = config_in(models_dir);
        stub_flux_companions(&mut config, models_dir, true);

        let entry = flux_unet_only_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_model_config("cv:994561", &intent, &config, FAIL).unwrap();

        assert_eq!(cfg.transformer.as_deref(), primary.to_str());
        assert_eq!(
            cfg.vae.as_deref(),
            models_dir.join("flux-vae/ae.safetensors").to_str()
        );
    }

    #[test]
    fn flux_bundled_vae_keeps_primary() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let primary = models_dir.join("cv-101010/flux/civitai/101010/flux_full.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
                "first_stage_model.encoder.conv_in.weight",
            ],
        );
        let mut config = config_in(models_dir);
        stub_flux_companions(&mut config, models_dir, true);

        let entry = flux_unet_only_entry("101010", "flux_full.safetensors");
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_model_config("cv:101010", &intent, &config, FAIL).unwrap();

        assert_eq!(cfg.transformer.as_deref(), primary.to_str());
        assert_eq!(
            cfg.vae.as_deref(),
            primary.to_str(),
            "bundled-VAE checkpoint keeps cfg.vae == primary; flux-vae companion is a no-op"
        );
    }

    #[test]
    fn flux_schnell_subfamily_selects_schnell_config() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let primary = models_dir
            .join("cv-1153358/flux/civitai/1153358/agfluxSchnell_realistic23.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
                "model.diffusion_model.img_in.weight",
            ],
        );
        let mut config = config_in(models_dir);
        stub_flux_companions(&mut config, models_dir, true);

        let mut entry = flux_unet_only_entry("1153358", "agfluxSchnell_realistic23.safetensors");
        entry.sub_family = Some("flux1-s".into());
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_model_config("cv:1153358", &intent, &config, FAIL).unwrap();
        assert_eq!(cfg.is_schnell, Some(true));
    }

    #[test]
    fn flux_dev_subfamily_applies_dev_defaults() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let primary =
            models_dir.join("cv-2319074/flux/civitai/2319074/jibMixFlux_v12SRPO.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );
        let mut config = config_in(models_dir);
        stub_flux_companions(&mut config, models_dir, true);

        let mut entry = flux_unet_only_entry("2319074", "jibMixFlux_v12SRPO.safetensors");
        entry.sub_family = Some("flux1-d".into());
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_model_config("cv:2319074", &intent, &config, FAIL).unwrap();
        assert_eq!(cfg.is_schnell, Some(false));
        assert_eq!(cfg.default_steps, Some(25));
        assert_eq!(cfg.default_guidance, Some(3.5));
    }

    #[test]
    fn qwen_runtime_companion_paths_populated() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let primary =
            models_dir.join("cv-2110043/qwen-image/civitai/2110043/qwenImage_fp8.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        std::fs::File::create(&primary).unwrap();

        let runtime_dir = models_dir.join("qwen-image-runtime");
        let vae = runtime_dir.join("vae/diffusion_pytorch_model.safetensors");
        let te = runtime_dir.join("text_encoder/model-00001-of-00004.safetensors");
        let tok = runtime_dir.join("tokenizer/tokenizer.json");
        for p in [&vae, &te, &tok] {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::File::create(p).unwrap();
        }
        let mut config = config_in(models_dir);
        config.models.insert(
            "qwen-image-runtime".into(),
            ModelConfig {
                family: Some("companion".into()),
                transformer: Some(vae.to_string_lossy().into_owned()),
                vae: Some(vae.to_string_lossy().into_owned()),
                text_encoder_files: Some(vec![te.to_string_lossy().into_owned()]),
                text_tokenizer: Some(tok.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );

        let mut entry = flux_unet_only_entry("2110043", "qwenImage_fp8.safetensors");
        entry.family = Family::QwenImage;
        entry.companions = vec!["qwen-image-runtime".into()];
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_model_config("cv:2110043", &intent, &config, FAIL).unwrap();

        assert_eq!(cfg.transformer.as_deref(), primary.to_str());
        assert_eq!(cfg.vae.as_deref(), vae.to_str());
        assert_eq!(
            cfg.text_encoder_files.as_deref(),
            Some(vec![te.to_string_lossy().into_owned()].as_slice())
        );
        assert_eq!(cfg.text_tokenizer.as_deref(), tok.to_str());
    }

    #[test]
    fn wuerstchen_runtime_companion_paths_populated() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let primary = models_dir.join(
            "hf-example/wuerstchen-prior/wuerstchen/example/wuerstchen-prior/prior.safetensors",
        );
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        std::fs::File::create(&primary).unwrap();

        let runtime_dir = models_dir.join("wuerstchen-runtime");
        let decoder = runtime_dir.join("decoder/diffusion_pytorch_model.safetensors");
        let vae = runtime_dir.join("vqgan/diffusion_pytorch_model.safetensors");
        let clip = runtime_dir.join("text_encoder/model.safetensors");
        let clip_tok = runtime_dir.join("tokenizer/tokenizer.json");
        let clip_g = runtime_dir.join("prior/text_encoder/model.safetensors");
        let clip_g_tok = runtime_dir.join("prior/tokenizer/tokenizer.json");
        for p in [&decoder, &vae, &clip, &clip_tok, &clip_g, &clip_g_tok] {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::File::create(p).unwrap();
        }
        let mut config = config_in(models_dir);
        config.models.insert(
            "wuerstchen-runtime".into(),
            ModelConfig {
                family: Some("companion".into()),
                transformer: Some(decoder.to_string_lossy().into_owned()),
                decoder: Some(decoder.to_string_lossy().into_owned()),
                vae: Some(vae.to_string_lossy().into_owned()),
                clip_encoder: Some(clip.to_string_lossy().into_owned()),
                clip_tokenizer: Some(clip_tok.to_string_lossy().into_owned()),
                clip_encoder_2: Some(clip_g.to_string_lossy().into_owned()),
                clip_tokenizer_2: Some(clip_g_tok.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );

        let mut entry = flux_unet_only_entry("unused", "prior.safetensors");
        entry.id = CatalogId::from("hf:example/wuerstchen-prior");
        entry.source = Source::Hf;
        entry.source_id = "example/wuerstchen-prior".into();
        entry.family = Family::Wuerstchen;
        entry.companions = vec!["wuerstchen-runtime".into()];
        entry.download_recipe.files[0].dest = "{family}/{author}/{name}/prior.safetensors".into();
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg =
            resolve_intent_to_model_config("hf:example/wuerstchen-prior", &intent, &config, FAIL)
                .unwrap();

        assert_eq!(cfg.transformer.as_deref(), primary.to_str());
        assert_eq!(cfg.decoder.as_deref(), decoder.to_str());
        assert_eq!(cfg.vae.as_deref(), vae.to_str());
        assert_eq!(cfg.clip_encoder.as_deref(), clip.to_str());
        assert_eq!(cfg.clip_encoder_2.as_deref(), clip_g.to_str());
    }

    #[test]
    fn zimage_recipe_text_encoder_wins_and_shared_vae_used() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let te_dir = models_dir.join("z-image-te");
        let mut config = config_in(models_dir);
        config.models.insert(
            "z-image-te".into(),
            ModelConfig {
                family: Some("companion".into()),
                transformer: Some(
                    te_dir
                        .join("text_encoder/model-00001-of-00003.safetensors")
                        .to_string_lossy()
                        .into_owned(),
                ),
                vae: Some(
                    te_dir
                        .join("vae/diffusion_pytorch_model.safetensors")
                        .to_string_lossy()
                        .into_owned(),
                ),
                text_encoder_files: Some(vec![te_dir
                    .join("text_encoder/model-00001-of-00003.safetensors")
                    .to_string_lossy()
                    .into_owned()]),
                text_tokenizer: Some(
                    te_dir
                        .join("tokenizer/tokenizer.json")
                        .to_string_lossy()
                        .into_owned(),
                ),
                ..Default::default()
            },
        );

        let mut entry = flux_unet_only_entry("2442439", "zImageTurbo_turbo.safetensors");
        entry.family = Family::ZImage;
        entry.companions = vec!["z-image-te".into()];
        entry.download_recipe.files = vec![
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
        ];
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        std::fs::create_dir_all(intent.primary_recipe_path.parent().unwrap()).unwrap();
        std::fs::write(&intent.primary_recipe_path, b"primary").unwrap();
        let cfg = resolve_intent_to_model_config("cv:2442439", &intent, &config, FAIL).unwrap();

        let recipe_te =
            models_dir.join("cv-2442439/z-image/civitai/2442439/zImageTurbo_turbo_txt.safetensors");
        let shared_vae = te_dir.join("vae/diffusion_pytorch_model.safetensors");
        assert_eq!(cfg.vae.as_deref(), shared_vae.to_str());
        assert_eq!(
            cfg.text_encoder_files.as_deref(),
            Some(vec![recipe_te.to_string_lossy().into_owned()].as_slice()),
            "recipe-provided text encoder must beat the shared companion shards"
        );
    }

    #[test]
    fn missing_required_companion_errors_under_fail_policy() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let primary = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );
        // Empty config — none of t5/clip-l/flux-vae installed.
        let config = config_in(models_dir);

        let entry = flux_unet_only_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let err = resolve_intent_to_model_config("cv:994561", &intent, &config, FAIL).unwrap_err();
        assert!(matches!(err, ResolveError::CompanionConfigMissing { .. }));
        let msg = err.to_string();
        assert!(msg.contains("t5-v1_1-xxl") || msg.contains("clip-l") || msg.contains("flux-vae"));
    }

    #[test]
    fn missing_companion_under_warn_policy_still_sets_transformer() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        // No primary on disk, no companions installed — WarnAndSkip +
        // require_primary_present:false must still produce a config with the
        // transformer path set (the CLI's post-pull behavior).
        let config = config_in(models_dir);
        let entry = flux_unet_only_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_model_config("cv:994561", &intent, &config, WARN).unwrap();
        assert_eq!(
            cfg.transformer.as_deref(),
            intent.primary_recipe_path.to_str()
        );
    }

    #[test]
    fn primary_missing_errors_under_require_primary_present() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let mut config = config_in(models_dir);
        stub_flux_companions(&mut config, models_dir, true);

        let entry = flux_unet_only_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let intent = synthesize_intent(&entry, models_dir).unwrap();

        // First pass — primary not on disk.
        let err = resolve_intent_to_model_config("cv:994561", &intent, &config, FAIL).unwrap_err();
        assert!(matches!(err, ResolveError::PrimaryFileMissing { .. }));

        // File arrives as a transformer-only checkpoint; second pass resolves.
        std::fs::create_dir_all(intent.primary_recipe_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &intent.primary_recipe_path,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );
        let cfg = resolve_intent_to_model_config("cv:994561", &intent, &config, FAIL).unwrap();
        assert_eq!(
            cfg.vae.as_deref(),
            models_dir.join("flux-vae/ae.safetensors").to_str()
        );
    }

    #[test]
    fn rejects_truncated_sidecar_primary() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let primary = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );
        let entry = flux_unet_only_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let mut sidecar = crate::sidecar::sidecar_from_entry(
            &entry,
            "flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors".into(),
        );
        sidecar.size_bytes = Some(primary.metadata().unwrap().len() + 1);
        let sc_path = crate::sidecar::civitai_sidecar_path(models_dir, "cv:994561");
        crate::sidecar::write_sidecar(&sc_path, &sidecar).unwrap();

        let mut config = config_in(models_dir);
        stub_flux_companions(&mut config, models_dir, true);
        let intent = synthesize_intent(&entry, models_dir).unwrap();
        let err = resolve_intent_to_model_config("cv:994561", &intent, &config, FAIL).unwrap_err();
        assert!(matches!(err, ResolveError::PrimaryFileMissing { .. }));
    }

    // ── installed_intent_from_sidecar ──────────────────────────────────────

    fn write_checkpoint_sidecar(
        models_dir: &Path,
        id: &str,
        dir_name: &str,
        family: &str,
        sub_family: Option<&str>,
        primary_rel: &str,
        supported: bool,
    ) {
        let install_dir = models_dir.join(dir_name);
        let primary = install_dir.join(primary_rel);
        std::fs::create_dir_all(primary.parent().unwrap()).unwrap();
        std::fs::write(&primary, b"fake").unwrap();
        let sidecar = crate::sidecar::CatalogSidecar {
            schema: crate::sidecar::SIDECAR_SCHEMA,
            id: id.to_string(),
            source: "civitai".to_string(),
            source_id: id.trim_start_matches("cv:").to_string(),
            name: "X".to_string(),
            author: None,
            family: family.to_string(),
            family_role: "finetune".to_string(),
            sub_family: sub_family.map(str::to_string),
            kind: "checkpoint".to_string(),
            modality: "image".to_string(),
            nsfw: None,
            description: None,
            tags: Vec::new(),
            license: None,
            page_url: None,
            thumbnail_url: None,
            size_bytes: Some(4),
            supported,
            trained_words: Vec::new(),
            primary_filename_rel: primary_rel.to_string(),
            written_at: 0,
        };
        crate::sidecar::write_sidecar(
            &install_dir.join(crate::sidecar::SIDECAR_FILENAME),
            &sidecar,
        )
        .unwrap();
    }

    #[test]
    fn installed_intent_resolves_downloaded_ltx2_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint_sidecar(
            dir.path(),
            "cv:2752735",
            "cv-2752735",
            "ltx2",
            Some("v2.3"),
            "ltx2/civitai/2752735/ltx23_full.safetensors",
            true,
        );
        let intent = installed_intent_from_sidecar(dir.path(), "cv:2752735")
            .expect("installed sidecar should synthesize an intent");
        assert_eq!(intent.family, "ltx2");
        assert!(intent.companions.iter().any(|c| c.name == "ltx2-te"));
        assert!(intent.companions.iter().any(|c| c.name == "ltx2.3-vae"));
        assert!(intent
            .companions
            .iter()
            .any(|c| c.name == "ltx2.3-text-projection"));
    }

    #[test]
    fn installed_intent_rejects_text_encoder_primary() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint_sidecar(
            dir.path(),
            "cv:2442439",
            "cv-2442439",
            "z-image",
            Some("turbo"),
            "z-image/civitai/2442439/zImageTurbo_turbo_txt.safetensors",
            true,
        );
        assert!(installed_intent_from_sidecar(dir.path(), "cv:2442439").is_none());
    }

    #[test]
    fn installed_intent_rejects_te_suffix_primary() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint_sidecar(
            dir.path(),
            "cv:2597527",
            "cv-2597527",
            "flux2",
            Some("klein-9b"),
            "flux2/civitai/2597527/qwen38BFluxKlein9BTE_38b.safetensors",
            true,
        );
        assert!(installed_intent_from_sidecar(dir.path(), "cv:2597527").is_none());
    }

    #[test]
    fn installed_checkpoint_sidecar_resolves_config_without_live_lookup() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_models_dir_env();
        let dir = tempfile::tempdir().unwrap();
        let _home = pin_mold_home(dir.path());
        let models_dir = dir.path();
        let mut config = config_in(models_dir);
        stub_sdxl_companions(&mut config, models_dir.to_str().unwrap());

        write_checkpoint_sidecar(
            models_dir,
            "cv:1075446",
            "cv-1075446",
            "sdxl",
            None,
            "sdxl/civitai/1075446/realism.safetensors",
            true,
        );
        let intent = installed_intent_from_sidecar(models_dir, "cv:1075446")
            .expect("installed checkpoint sidecar resolves");
        let cfg = resolve_intent_to_model_config("cv:1075446", &intent, &config, WARN).unwrap();
        assert_eq!(cfg.family.as_deref(), Some("sdxl"));
        let primary = models_dir.join("cv-1075446/sdxl/civitai/1075446/realism.safetensors");
        assert_eq!(cfg.transformer.as_deref(), primary.to_str());
        assert_eq!(cfg.vae.as_deref(), primary.to_str());
    }
}
