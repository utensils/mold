//! Sync between `mold-core::Config` (TOML bootstrap file) and the
//! DB-backed user-preference surface. Phase 3 of the SQLite settings
//! migration (issue #265): user-facing slices of config.toml move into
//! the `settings` + `model_prefs` tables; paths/logging/credentials stay
//! in the TOML.
//!
//! Public entry points:
//!
//! - [`migrate_config_toml_to_db`] — one-shot, idempotent import gated by
//!   [`CONFIG_MIGRATED_FROM_TOML`](crate::settings::CONFIG_MIGRATED_FROM_TOML).
//! - [`hydrate_config_from_db`] — overlay DB values onto an already-loaded
//!   `Config` so downstream consumers read the authoritative values
//!   without knowing about the DB.
//! - [`save_expand_to_db`] / [`save_generate_globals_to_db`] — used by the
//!   CLI `mold config set` dispatcher for expand.* / generate.* keys.

use anyhow::{Context, Result};
use mold_core::config::{Config, ModelConfig};
use mold_core::expand::ExpandSettings;
use std::path::Path;
use std::sync::OnceLock;

use crate::db::MetadataDb;
use crate::model_prefs::ModelPrefs;
use crate::settings::{self as keys, Settings};

/// Save the user-facing expand settings into the `settings` table.
/// Families are stored as a JSON blob since it's a user-edited map.
pub fn save_expand_to_db(db: &MetadataDb, expand: &ExpandSettings) -> Result<()> {
    let s = Settings::new(db);
    s.set_bool(keys::EXPAND_ENABLED, expand.enabled)?;
    s.set_str(keys::EXPAND_BACKEND, &expand.backend)?;
    s.set_str(keys::EXPAND_MODEL, &expand.model)?;
    s.set_str(keys::EXPAND_API_MODEL, &expand.api_model)?;
    s.set_float(keys::EXPAND_TEMPERATURE, expand.temperature)?;
    s.set_float(keys::EXPAND_TOP_P, expand.top_p)?;
    s.set_int(keys::EXPAND_MAX_TOKENS, expand.max_tokens as i64)?;
    s.set_bool(keys::EXPAND_THINKING, expand.thinking)?;
    match &expand.system_prompt {
        Some(v) => s.set_str(keys::EXPAND_SYSTEM_PROMPT, v)?,
        None => {
            s.delete(keys::EXPAND_SYSTEM_PROMPT)?;
        }
    }
    match &expand.batch_prompt {
        Some(v) => s.set_str(keys::EXPAND_BATCH_PROMPT, v)?,
        None => {
            s.delete(keys::EXPAND_BATCH_PROMPT)?;
        }
    }
    if expand.families.is_empty() {
        s.delete(keys::EXPAND_FAMILIES_JSON)?;
    } else {
        s.set_json(keys::EXPAND_FAMILIES_JSON, &expand.families)?;
    }
    Ok(())
}

/// Overlay any stored expand settings onto `expand`, filling in only the
/// fields that have rows. Returns true if at least one row applied.
///
/// Caller is responsible for applying env-var overrides *after* this (the
/// same `MOLD_EXPAND_*` layer that `with_env_overrides()` uses for TOML).
pub fn hydrate_expand_from_db(db: &MetadataDb, expand: &mut ExpandSettings) -> Result<bool> {
    let s = Settings::new(db);
    let mut applied = false;
    if let Some(v) = s.get_bool(keys::EXPAND_ENABLED)? {
        expand.enabled = v;
        applied = true;
    }
    if let Some(v) = s.get_str(keys::EXPAND_BACKEND)? {
        expand.backend = v;
        applied = true;
    }
    if let Some(v) = s.get_str(keys::EXPAND_MODEL)? {
        expand.model = v;
        applied = true;
    }
    if let Some(v) = s.get_str(keys::EXPAND_API_MODEL)? {
        expand.api_model = v;
        applied = true;
    }
    if let Some(v) = s.get_float(keys::EXPAND_TEMPERATURE)? {
        expand.temperature = v;
        applied = true;
    }
    if let Some(v) = s.get_float(keys::EXPAND_TOP_P)? {
        expand.top_p = v;
        applied = true;
    }
    if let Some(v) = s.get_int(keys::EXPAND_MAX_TOKENS)? {
        expand.max_tokens = v as u32;
        applied = true;
    }
    if let Some(v) = s.get_bool(keys::EXPAND_THINKING)? {
        expand.thinking = v;
        applied = true;
    }
    if let Some(v) = s.get_str(keys::EXPAND_SYSTEM_PROMPT)? {
        expand.system_prompt = Some(v);
        applied = true;
    }
    if let Some(v) = s.get_str(keys::EXPAND_BATCH_PROMPT)? {
        expand.batch_prompt = Some(v);
        applied = true;
    }
    if let Some(families) = s.get_json(keys::EXPAND_FAMILIES_JSON)? {
        expand.families = families;
        applied = true;
    }
    Ok(applied)
}

/// Persist the global generation defaults from `Config` into `settings`.
///
/// Optional string fields (`default_negative_prompt`, `t5_variant`,
/// `qwen3_variant`) are always written: `None` becomes an empty-string
/// row so [`hydrate_generate_globals_from_db`] can distinguish "user
/// cleared this" from "never touched via DB". If we deleted the row
/// instead, the pre-migration TOML value would silently resurrect on
/// every load.
pub fn save_generate_globals_to_db(db: &MetadataDb, cfg: &Config) -> Result<()> {
    let s = Settings::new(db);
    s.set_int(keys::GENERATE_DEFAULT_WIDTH, cfg.default_width as i64)?;
    s.set_int(keys::GENERATE_DEFAULT_HEIGHT, cfg.default_height as i64)?;
    s.set_int(keys::GENERATE_DEFAULT_STEPS, cfg.default_steps as i64)?;
    s.set_bool(keys::GENERATE_EMBED_METADATA, cfg.embed_metadata)?;
    s.set_str(
        keys::GENERATE_DEFAULT_NEGATIVE_PROMPT,
        cfg.default_negative_prompt.as_deref().unwrap_or(""),
    )?;
    s.set_str(
        keys::GENERATE_T5_VARIANT,
        cfg.t5_variant.as_deref().unwrap_or(""),
    )?;
    s.set_str(
        keys::GENERATE_QWEN3_VARIANT,
        cfg.qwen3_variant.as_deref().unwrap_or(""),
    )?;
    Ok(())
}

/// Overlay any stored generation globals onto `cfg` in place. Only fields
/// present in the DB are touched. Returns true if at least one row applied.
///
/// Empty-string values on optional fields are treated as explicit clears:
/// `cfg.default_negative_prompt = None`, etc. The TOML-backed value is
/// replaced rather than preserved, which is the semantics a user expects
/// after `mold config set default_negative_prompt none`.
pub fn hydrate_generate_globals_from_db(db: &MetadataDb, cfg: &mut Config) -> Result<bool> {
    let s = Settings::new(db);
    let mut applied = false;
    if let Some(v) = s.get_int(keys::GENERATE_DEFAULT_WIDTH)? {
        cfg.default_width = v as u32;
        applied = true;
    }
    if let Some(v) = s.get_int(keys::GENERATE_DEFAULT_HEIGHT)? {
        cfg.default_height = v as u32;
        applied = true;
    }
    if let Some(v) = s.get_int(keys::GENERATE_DEFAULT_STEPS)? {
        cfg.default_steps = v as u32;
        applied = true;
    }
    if let Some(v) = s.get_bool(keys::GENERATE_EMBED_METADATA)? {
        cfg.embed_metadata = v;
        applied = true;
    }
    if let Some(v) = s.get_str(keys::GENERATE_DEFAULT_NEGATIVE_PROMPT)? {
        cfg.default_negative_prompt = if v.is_empty() { None } else { Some(v) };
        applied = true;
    }
    if let Some(v) = s.get_str(keys::GENERATE_T5_VARIANT)? {
        cfg.t5_variant = if v.is_empty() { None } else { Some(v) };
        applied = true;
    }
    if let Some(v) = s.get_str(keys::GENERATE_QWEN3_VARIANT)? {
        cfg.qwen3_variant = if v.is_empty() { None } else { Some(v) };
        applied = true;
    }
    Ok(applied)
}

/// Snapshot the user-editable per-model generation defaults out of a
/// `ModelConfig` into a `ModelPrefs` row (the path fields stay in TOML).
fn model_prefs_from_config(mc: &ModelConfig) -> ModelPrefs {
    ModelPrefs {
        width: mc.default_width,
        height: mc.default_height,
        steps: mc.default_steps,
        guidance: mc.default_guidance,
        scheduler: mc.scheduler.map(|s| s.to_string()),
        lora_path: mc.lora.clone(),
        lora_scale: mc.lora_scale,
        frames: mc.default_frames,
        fps: mc.default_fps,
        last_negative: mc.negative_prompt.clone(),
        ..Default::default()
    }
}

/// Apply a `ModelPrefs` row onto a `ModelConfig`, preserving any field the
/// row doesn't set. Path fields on `ModelConfig` are never touched.
pub fn apply_prefs_to_model_config(prefs: &ModelPrefs, mc: &mut ModelConfig) {
    if let Some(v) = prefs.width {
        mc.default_width = Some(v);
    }
    if let Some(v) = prefs.height {
        mc.default_height = Some(v);
    }
    if let Some(v) = prefs.steps {
        mc.default_steps = Some(v);
    }
    if let Some(v) = prefs.guidance {
        mc.default_guidance = Some(v);
    }
    if let Some(ref v) = prefs.scheduler {
        if let Some(parsed) = parse_scheduler(v) {
            mc.scheduler = Some(parsed);
        }
    }
    if let Some(ref v) = prefs.lora_path {
        mc.lora = Some(v.clone());
    }
    if let Some(v) = prefs.lora_scale {
        mc.lora_scale = Some(v);
    }
    if let Some(v) = prefs.frames {
        mc.default_frames = Some(v);
    }
    if let Some(v) = prefs.fps {
        mc.default_fps = Some(v);
    }
    if let Some(ref v) = prefs.last_negative {
        mc.negative_prompt = Some(v.clone());
    }
}

fn parse_scheduler(s: &str) -> Option<mold_core::Scheduler> {
    s.parse().ok()
}

/// One-shot import of user-preference slices out of `config.toml` into the
/// DB. Idempotent — gated by the [`CONFIG_MIGRATED_FROM_TOML`] sentinel.
///
/// Returns `Ok(true)` when the import actually ran (first call on an
/// un-migrated DB), `Ok(false)` when the sentinel was already set. The
/// sentinel is written inside the same `with_conn` scope to keep the
/// post-condition tight: on the rare failure mode where the last DB write
/// aborts, the next launch re-runs cleanly.
///
/// [`CONFIG_MIGRATED_FROM_TOML`]: crate::settings::CONFIG_MIGRATED_FROM_TOML
pub fn migrate_config_toml_to_db(db: &MetadataDb, cfg: &Config) -> Result<bool> {
    let s = Settings::new(db);
    if s.get_bool(keys::CONFIG_MIGRATED_FROM_TOML)? == Some(true) {
        return Ok(false);
    }

    // Global expand + generate.
    save_expand_to_db(db, &cfg.expand)?;
    save_generate_globals_to_db(db, cfg)?;

    // Per-model user prefs. Skip rows that carry nothing but paths — we
    // want a ModelPrefs row only when the user has set at least one
    // generation default.
    for (name, mc) in cfg.models.iter() {
        let prefs = model_prefs_from_config(mc);
        if prefs == ModelPrefs::default() {
            continue;
        }
        let canonical = mold_core::manifest::resolve_model_name(name);
        prefs.save(db, &canonical)?;
    }

    // Legacy `$MOLD_HOME/last-model` sidecar → settings. Use the sidecar
    // reader directly — the public `read_last_model()` now routes through
    // the DB hook, which would return an empty result pre-migration.
    if let Some(last) = Config::read_last_model_from_sidecar() {
        s.set_str(keys::TUI_LAST_MODEL, &last)?;
    }

    s.set_bool(keys::CONFIG_MIGRATED_FROM_TOML, true)?;
    Ok(true)
}

/// Move the original `config.toml` to `config.toml.migrated` and write a
/// stripped bootstrap-only TOML in its place. Safe to call repeatedly —
/// it's a no-op when `config_path` doesn't exist.
///
/// Used by the post-load hook after [`migrate_config_toml_to_db`] returns
/// `Ok(true)` so the TOML stops carrying stale user-preference fields
/// that the DB is now authoritative for. The `.migrated` sibling is the
/// one-release downgrade safety net; cleanup is tracked by
/// [`cleanup_stale_backups`].
pub fn rewrite_stripped_config_toml(cfg: &Config, config_path: &Path) -> Result<()> {
    if !config_path.exists() {
        return Ok(());
    }
    let backup_path = sibling_migrated_path(config_path);
    std::fs::rename(config_path, &backup_path).with_context(|| {
        format!(
            "renaming {} → {}",
            config_path.display(),
            backup_path.display()
        )
    })?;
    cfg.save_bootstrap_only_to(config_path).with_context(|| {
        format!(
            "writing stripped bootstrap config to {}",
            config_path.display()
        )
    })?;
    Ok(())
}

fn sibling_migrated_path(p: &Path) -> std::path::PathBuf {
    // `config.toml` → `config.toml.migrated`; works for any filename.
    let mut s = p.as_os_str().to_os_string();
    s.push(".migrated");
    std::path::PathBuf::from(s)
}

/// Scan a mold home directory for stale `.migrated` backups left by the
/// one-shot legacy-state imports (`tui-session.json.migrated`,
/// `prompt-history.jsonl.migrated`, `config.toml.migrated`). Returns the
/// list of paths that still exist.
///
/// This is the *detection* half of the cleanup. Actual deletion is
/// deferred one release per the original `.migrated` safety-net contract
/// — call [`cleanup_stale_backups`] once we're ready to ship that.
pub fn detect_stale_backups(mold_dir: &Path, config_dir: &Path) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    for p in [
        mold_dir.join("tui-session.json.migrated"),
        mold_dir.join("prompt-history.jsonl.migrated"),
        config_dir.join("config.toml.migrated"),
    ] {
        if p.exists() {
            paths.push(p);
        }
    }
    paths
}

/// Delete every `.migrated` backup found by [`detect_stale_backups`].
/// Writes the [`BACKUPS_CLEANED_AT_V6`](crate::settings::BACKUPS_CLEANED_AT_V6)
/// sentinel so subsequent launches skip the scan.
///
/// ### Release timing
///
/// Issue #265 introduces this helper but intentionally does NOT wire it
/// into the hook: the `.migrated` renames landed only one release ago, so
/// we keep them as a downgrade safety net for one more version. Flip the
/// `cleanup_stale_backups_on_launch` call in `global_hook` when #265 has
/// shipped in a stable release.
pub fn cleanup_stale_backups(db: &MetadataDb, mold_dir: &Path, config_dir: &Path) -> Result<usize> {
    let s = Settings::for_profile(db, keys::DEFAULT_PROFILE);
    if s.get_bool(keys::BACKUPS_CLEANED_AT_V6)? == Some(true) {
        return Ok(0);
    }
    let mut dropped = 0usize;
    for p in detect_stale_backups(mold_dir, config_dir) {
        match std::fs::remove_file(&p) {
            Ok(()) => {
                tracing::info!(path = %p.display(), "removed stale .migrated backup");
                dropped += 1;
            }
            Err(e) => {
                tracing::warn!(path = %p.display(), "failed to remove stale backup: {e}")
            }
        }
    }
    s.set_bool(keys::BACKUPS_CLEANED_AT_V6, true)?;
    Ok(dropped)
}

/// Overlay the DB view onto `cfg` in place — call this right after
/// `Config::load_or_default()` so downstream consumers see authoritative
/// values without knowing about the DB layer. Env-var overrides should be
/// re-applied to `cfg.expand` by the caller (`with_env_overrides()`).
///
/// Iterates `model_prefs` in two passes: first updates existing
/// `cfg.models` entries (preserves path fields on ModelConfig), then
/// materializes a default `ModelConfig` for any model that has a DB row
/// but no TOML entry — e.g. a manifest-discovered model, or one
/// configured only via `mold config set models.<name>.default_steps`.
/// Without the second pass those rows are invisible to
/// `resolved_model_config()` and the user's saved defaults silently fall
/// back to manifest values.
pub fn hydrate_config_from_db(db: &MetadataDb, cfg: &mut Config) -> Result<()> {
    hydrate_expand_from_db(db, &mut cfg.expand)?;
    hydrate_generate_globals_from_db(db, cfg)?;

    // Pass 1: overlay prefs for each model already in cfg.models. Keyed
    // on the canonical resolution so `flux-dev` and `flux-dev:q4` hit
    // the same DB row consistently.
    let existing_canonicals: std::collections::HashSet<String> = cfg
        .models
        .keys()
        .map(|k| mold_core::manifest::resolve_model_name(k))
        .collect();
    for (name, mc) in cfg.models.iter_mut() {
        let canonical = mold_core::manifest::resolve_model_name(name);
        if let Some(prefs) = ModelPrefs::load(db, &canonical)? {
            apply_prefs_to_model_config(&prefs, mc);
        }
    }

    // Pass 2: materialize entries for DB-only rows. Required so a user
    // who set per-model generation defaults via `mold config set` (or
    // via the TUI on a manifest-only model) finds their values applied
    // on the next load.
    for (model_key, prefs) in ModelPrefs::list(db)? {
        if existing_canonicals.contains(&model_key) {
            continue;
        }
        let mut mc = ModelConfig::default();
        apply_prefs_to_model_config(&prefs, &mut mc);
        cfg.models.insert(model_key, mc);
    }

    Ok(())
}

/// Migrate (once) + hydrate `cfg` using an explicit DB handle. Exposed so
/// embedders and tests can drive the hook body without installing the
/// process-wide `Config::load_or_default()` hook.
///
/// Migration idempotency is enforced by the DB-side
/// [`CONFIG_MIGRATED_FROM_TOML`](crate::settings::CONFIG_MIGRATED_FROM_TOML)
/// sentinel, so calling this on every load is safe but wasteful — the
/// [`install_config_post_load_hook`] installer adds a per-process
/// `OnceLock` fast-path on top.
pub fn run_post_load_hook(db: &MetadataDb, cfg: &mut Config) {
    match migrate_config_toml_to_db(db, cfg) {
        Ok(true) => rewrite_config_toml_on_disk(cfg),
        Ok(false) => {}
        Err(e) => tracing::warn!("config → DB migration skipped: {e:#}"),
    }
    if let Err(e) = hydrate_config_from_db(db, cfg) {
        tracing::warn!("config DB hydration skipped: {e:#}");
    }
}

fn rewrite_config_toml_on_disk(cfg: &Config) {
    let Some(path) = Config::config_path() else {
        return;
    };
    if let Err(e) = rewrite_stripped_config_toml(cfg, &path) {
        tracing::warn!(
            "stripped config.toml rewrite skipped for {}: {e:#}",
            path.display()
        );
    }
}

/// Register the process-wide `Config::load_or_default()` post-load hook
/// that migrates the TOML-held user preferences into the DB (once) and
/// overlays DB values on every subsequent load. Uses [`crate::global_db`]
/// — safe to call from any binary main(); idempotent on repeated calls.
///
/// Call this once from each binary's `main()` before anything invokes
/// `Config::load_or_default()`. `mold-cli`, `mold-server`, and
/// `mold-discord` all route through this entry point so they share the
/// same SQLite-backed view.
pub fn install_config_post_load_hook() {
    mold_core::config::install_post_load_hook(global_hook);
    mold_core::config::install_read_last_model_hook(db_read_last_model);
}

fn db_read_last_model() -> Option<String> {
    let db = crate::global_db()?;
    Settings::new(db)
        .get_str(keys::TUI_LAST_MODEL)
        .ok()
        .flatten()
        .filter(|s| !s.is_empty())
}

fn global_hook(cfg: &mut Config) {
    // Per-process fast path: the DB sentinel is the source of truth for
    // migration, but re-checking it on every config load is wasteful.
    static MIGRATION_DONE: OnceLock<()> = OnceLock::new();
    let Some(db) = crate::global_db() else {
        return;
    };
    if MIGRATION_DONE.set(()).is_ok() {
        match migrate_config_toml_to_db(db, cfg) {
            Ok(true) => rewrite_config_toml_on_disk(cfg),
            Ok(false) => {}
            Err(e) => tracing::warn!("config → DB migration skipped: {e:#}"),
        }
    }
    if let Err(e) = hydrate_config_from_db(db, cfg) {
        tracing::warn!("config DB hydration skipped: {e:#}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::config::ModelConfig;
    use mold_core::expand::FamilyOverride;
    use std::collections::HashMap;

    fn db() -> MetadataDb {
        MetadataDb::open_in_memory().unwrap()
    }

    #[test]
    fn expand_roundtrip_via_db() {
        let db = db();
        let mut families = HashMap::new();
        families.insert(
            "sd15".into(),
            FamilyOverride {
                word_limit: Some(40),
                style_notes: Some("keywords only".into()),
            },
        );
        let src = ExpandSettings {
            enabled: true,
            backend: "http://localhost:11434".into(),
            model: "qwen3-expand-small:q8".into(),
            api_model: "qwen2.5:3b".into(),
            temperature: 0.5,
            top_p: 0.8,
            max_tokens: 500,
            thinking: true,
            system_prompt: Some("sys {WORD_LIMIT} {MODEL_NOTES}".into()),
            batch_prompt: Some("batch {N} {WORD_LIMIT} {MODEL_NOTES}".into()),
            families,
        };
        save_expand_to_db(&db, &src).unwrap();

        let mut dst = ExpandSettings::default();
        let applied = hydrate_expand_from_db(&db, &mut dst).unwrap();
        assert!(applied);
        assert_eq!(dst.enabled, src.enabled);
        assert_eq!(dst.backend, src.backend);
        assert_eq!(dst.model, src.model);
        assert_eq!(dst.api_model, src.api_model);
        assert!((dst.temperature - src.temperature).abs() < 1e-9);
        assert!((dst.top_p - src.top_p).abs() < 1e-9);
        assert_eq!(dst.max_tokens, src.max_tokens);
        assert_eq!(dst.thinking, src.thinking);
        assert_eq!(dst.system_prompt, src.system_prompt);
        assert_eq!(dst.batch_prompt, src.batch_prompt);
        assert_eq!(dst.families.len(), 1);
        let sd15 = dst.families.get("sd15").unwrap();
        assert_eq!(sd15.word_limit, Some(40));
    }

    #[test]
    fn hydrate_expand_leaves_defaults_for_missing_keys() {
        let db = db();
        let mut dst = ExpandSettings::default();
        let applied = hydrate_expand_from_db(&db, &mut dst).unwrap();
        assert!(!applied);
        let defaults = ExpandSettings::default();
        assert_eq!(dst.enabled, defaults.enabled);
        assert_eq!(dst.backend, defaults.backend);
        assert_eq!(dst.model, defaults.model);
        assert_eq!(dst.temperature, defaults.temperature);
        assert!(dst.families.is_empty());
    }

    #[test]
    fn generate_globals_roundtrip_via_db() {
        let db = db();
        let cfg = Config {
            default_width: 1024,
            default_height: 1024,
            default_steps: 30,
            embed_metadata: false,
            default_negative_prompt: Some("ugly".into()),
            t5_variant: Some("q8".into()),
            qwen3_variant: Some("bf16".into()),
            ..Config::default()
        };
        save_generate_globals_to_db(&db, &cfg).unwrap();

        let mut target = Config::default();
        assert_ne!(target.default_width, 1024);
        let applied = hydrate_generate_globals_from_db(&db, &mut target).unwrap();
        assert!(applied);
        assert_eq!(target.default_width, 1024);
        assert_eq!(target.default_height, 1024);
        assert_eq!(target.default_steps, 30);
        assert!(!target.embed_metadata);
        assert_eq!(target.default_negative_prompt.as_deref(), Some("ugly"));
        assert_eq!(target.t5_variant.as_deref(), Some("q8"));
        assert_eq!(target.qwen3_variant.as_deref(), Some("bf16"));
    }

    #[test]
    fn migration_is_idempotent() {
        let db = db();
        let cfg = Config::default();
        assert!(migrate_config_toml_to_db(&db, &cfg).unwrap());
        assert!(!migrate_config_toml_to_db(&db, &cfg).unwrap());
    }

    #[test]
    fn migration_writes_expand_and_generate_and_per_model() {
        let db = db();
        let mut cfg = Config {
            default_width: 1536,
            default_steps: 25,
            ..Config::default()
        };
        cfg.expand.enabled = true;
        cfg.expand.temperature = 1.1;

        cfg.models.insert(
            "flux-dev:q4".into(),
            ModelConfig {
                default_steps: Some(20),
                default_guidance: Some(3.5),
                default_width: Some(1024),
                default_height: Some(1024),
                lora: Some("/path/lora.safetensors".into()),
                lora_scale: Some(0.8),
                negative_prompt: Some("blurry".into()),
                ..Default::default()
            },
        );

        assert!(migrate_config_toml_to_db(&db, &cfg).unwrap());

        // Expand landed.
        let mut expand = ExpandSettings::default();
        hydrate_expand_from_db(&db, &mut expand).unwrap();
        assert!(expand.enabled);
        assert!((expand.temperature - 1.1).abs() < 1e-9);

        // Generate globals landed.
        let mut tc = Config::default();
        hydrate_generate_globals_from_db(&db, &mut tc).unwrap();
        assert_eq!(tc.default_width, 1536);
        assert_eq!(tc.default_steps, 25);

        // Per-model row landed.
        let prefs = ModelPrefs::load(&db, "flux-dev:q4").unwrap().unwrap();
        assert_eq!(prefs.width, Some(1024));
        assert_eq!(prefs.steps, Some(20));
        assert_eq!(prefs.guidance, Some(3.5));
        assert_eq!(prefs.lora_path.as_deref(), Some("/path/lora.safetensors"));
        assert_eq!(prefs.lora_scale, Some(0.8));
        assert_eq!(prefs.last_negative.as_deref(), Some("blurry"));
    }

    #[test]
    fn migration_skips_path_only_model_rows() {
        let db = db();
        let mut cfg = Config::default();
        // Path fields only — no user-set defaults.
        cfg.models.insert(
            "flux-dev:q4".into(),
            ModelConfig {
                transformer: Some("/models/flux/transformer.gguf".into()),
                vae: Some("/models/flux/vae.safetensors".into()),
                ..Default::default()
            },
        );
        assert!(migrate_config_toml_to_db(&db, &cfg).unwrap());
        assert!(ModelPrefs::load(&db, "flux-dev:q4").unwrap().is_none());
    }

    #[test]
    fn hydrate_config_applies_per_model_prefs() {
        let db = db();
        ModelPrefs {
            width: Some(2048),
            height: Some(2048),
            steps: Some(15),
            guidance: Some(4.0),
            lora_path: Some("/foo.safetensors".into()),
            lora_scale: Some(1.2),
            ..Default::default()
        }
        .save(&db, "flux-dev:q4")
        .unwrap();

        let mut cfg = Config::default();
        cfg.models
            .insert("flux-dev:q4".into(), ModelConfig::default());
        hydrate_config_from_db(&db, &mut cfg).unwrap();

        let mc = cfg.models.get("flux-dev:q4").unwrap();
        assert_eq!(mc.default_width, Some(2048));
        assert_eq!(mc.default_steps, Some(15));
        assert_eq!(mc.default_guidance, Some(4.0));
        assert_eq!(mc.lora.as_deref(), Some("/foo.safetensors"));
        assert_eq!(mc.lora_scale, Some(1.2));
    }

    // ── Codex peer-review regressions (issue #265 round-2 fixes) ─────

    /// Codex P2: clearing an optional global like `default_negative_prompt`
    /// used to delete the DB row, letting the pre-migration TOML value
    /// silently resurrect on the next hydrate. The empty-string sentinel
    /// keeps the clear durable.
    #[test]
    fn hydrate_preserves_explicit_clear_for_optional_globals() {
        let db = db();
        // Simulate an earlier save that set a real value …
        let cfg_with_value = Config {
            default_negative_prompt: Some("ugly".into()),
            t5_variant: Some("q8".into()),
            qwen3_variant: Some("bf16".into()),
            ..Config::default()
        };
        save_generate_globals_to_db(&db, &cfg_with_value).unwrap();

        // … then the user clears each optional global.
        let cfg_cleared = Config {
            default_negative_prompt: None,
            t5_variant: None,
            qwen3_variant: None,
            ..Config::default()
        };
        save_generate_globals_to_db(&db, &cfg_cleared).unwrap();

        // Simulate a fresh load: TOML-shaped Config carries the *old*
        // values (TOML wasn't edited). Hydrate must wipe them to None
        // because the DB is now authoritative for these fields.
        let mut loaded = Config {
            default_negative_prompt: Some("stale-from-toml".into()),
            t5_variant: Some("q6".into()),
            qwen3_variant: Some("q8".into()),
            ..Config::default()
        };
        hydrate_generate_globals_from_db(&db, &mut loaded).unwrap();
        assert_eq!(loaded.default_negative_prompt, None);
        assert_eq!(loaded.t5_variant, None);
        assert_eq!(loaded.qwen3_variant, None);
    }

    /// Codex P2: a `model_prefs` row for a model with no `[models.*]`
    /// TOML entry was invisible — hydrate iterated only `cfg.models`.
    /// Pass 2 materializes a default `ModelConfig` for DB-only rows.
    #[test]
    fn hydrate_materializes_db_only_per_model_rows() {
        let db = db();
        ModelPrefs {
            width: Some(1024),
            steps: Some(25),
            guidance: Some(3.5),
            ..Default::default()
        }
        .save(&db, "flux-dev:q4")
        .unwrap();

        // Config has no matching model entry.
        let mut cfg = Config::default();
        assert!(!cfg.models.contains_key("flux-dev:q4"));

        hydrate_config_from_db(&db, &mut cfg).unwrap();

        // Pass 2 added the entry.
        let mc = cfg
            .models
            .get("flux-dev:q4")
            .expect("DB row should materialize");
        assert_eq!(mc.default_width, Some(1024));
        assert_eq!(mc.default_steps, Some(25));
        assert_eq!(mc.default_guidance, Some(3.5));
    }

    /// Codex P2: when both TOML and DB have the same canonical model,
    /// pass-1 wins and pass-2 doesn't double-insert — the path fields on
    /// the TOML-backed `ModelConfig` must survive.
    #[test]
    fn hydrate_pass_two_does_not_clobber_toml_model_path_fields() {
        let db = db();
        ModelPrefs {
            width: Some(768),
            ..Default::default()
        }
        .save(&db, "flux-dev:q4")
        .unwrap();

        let mut cfg = Config::default();
        cfg.models.insert(
            "flux-dev:q4".into(),
            ModelConfig {
                transformer: Some("/from/toml.safetensors".into()),
                ..Default::default()
            },
        );

        hydrate_config_from_db(&db, &mut cfg).unwrap();
        assert_eq!(cfg.models.len(), 1);
        let mc = cfg.models.get("flux-dev:q4").unwrap();
        assert_eq!(mc.default_width, Some(768));
        assert_eq!(mc.transformer.as_deref(), Some("/from/toml.safetensors"));
    }

    /// Codex P2: scheduler strings written by `mold config set` (canonical
    /// Display form, e.g. "euler-ancestral") must round-trip via
    /// `Scheduler::FromStr` so the TUI sees the same choice.
    #[test]
    fn scheduler_display_form_round_trips_via_from_str() {
        use mold_core::Scheduler;
        assert_eq!(
            Scheduler::EulerAncestral.to_string().parse::<Scheduler>(),
            Ok(Scheduler::EulerAncestral)
        );
        assert_eq!(
            Scheduler::UniPc.to_string().parse::<Scheduler>(),
            Ok(Scheduler::UniPc)
        );
        assert_eq!(
            Scheduler::Ddim.to_string().parse::<Scheduler>(),
            Ok(Scheduler::Ddim)
        );
    }

    /// Item 2 (post-#265): `run_post_load_hook` must drive both migrate
    /// and hydrate when given an explicit DB handle, so standalone
    /// binaries (`mold-server`, `mold-discord`) can reuse the same
    /// behaviour without linking against `mold-cli`.
    #[test]
    fn run_post_load_hook_migrates_and_hydrates() {
        let db = db();

        // First call with an "existing" TOML-shaped cfg: migration imports
        // cfg values into the DB, hydrate reads them back.
        let mut cfg = Config {
            default_width: 1536,
            default_steps: 33,
            ..Config::default()
        };
        cfg.expand.enabled = true;
        super::run_post_load_hook(&db, &mut cfg);

        assert_eq!(
            Settings::new(&db)
                .get_bool(keys::CONFIG_MIGRATED_FROM_TOML)
                .unwrap(),
            Some(true)
        );
        assert_eq!(cfg.default_width, 1536);

        // Simulate a user tweak via the DB (e.g. `mold config set`).
        Settings::new(&db)
            .set_int(keys::GENERATE_DEFAULT_WIDTH, 2112)
            .unwrap();

        // Second call on a fresh cfg: migration short-circuits (sentinel
        // set), hydrate pulls the new DB value in.
        let mut cfg2 = Config {
            default_width: 256,
            ..Config::default()
        };
        super::run_post_load_hook(&db, &mut cfg2);
        assert_eq!(cfg2.default_width, 2112);
    }

    /// Item 4 (post-#265): detect (don't delete) stale `.migrated`
    /// backups so future releases can flip to cleanup. Files returned
    /// match the three sidecars that land during the v5 → v6 upgrade.
    #[test]
    fn detect_stale_backups_lists_every_migrated_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let mold = dir.path().join("mold-home");
        let cfg = dir.path().join("config");
        std::fs::create_dir_all(&mold).unwrap();
        std::fs::create_dir_all(&cfg).unwrap();
        std::fs::write(mold.join("tui-session.json.migrated"), "{}").unwrap();
        std::fs::write(mold.join("prompt-history.jsonl.migrated"), "").unwrap();
        std::fs::write(cfg.join("config.toml.migrated"), "# old").unwrap();

        let found = super::detect_stale_backups(&mold, &cfg);
        assert_eq!(found.len(), 3);
    }

    /// Item 4: with no backups present, detection returns an empty list
    /// (no false positives).
    #[test]
    fn detect_stale_backups_is_empty_when_nothing_stale() {
        let dir = tempfile::tempdir().unwrap();
        assert!(super::detect_stale_backups(dir.path(), dir.path()).is_empty());
    }

    /// Item 4: `cleanup_stale_backups` removes every detected file and
    /// writes the idempotency sentinel. A second call is a no-op.
    #[test]
    fn cleanup_stale_backups_removes_files_once() {
        let dir = tempfile::tempdir().unwrap();
        let mold = dir.path().join("mold-home");
        std::fs::create_dir_all(&mold).unwrap();
        std::fs::write(mold.join("tui-session.json.migrated"), "{}").unwrap();
        std::fs::write(mold.join("prompt-history.jsonl.migrated"), "").unwrap();

        let db = db();
        let dropped = super::cleanup_stale_backups(&db, &mold, &mold).unwrap();
        assert_eq!(dropped, 2);
        assert!(!mold.join("tui-session.json.migrated").exists());
        assert!(!mold.join("prompt-history.jsonl.migrated").exists());
        assert_eq!(
            Settings::for_profile(&db, keys::DEFAULT_PROFILE)
                .get_bool(keys::BACKUPS_CLEANED_AT_V6)
                .unwrap(),
            Some(true)
        );
        // Second call is a no-op — sentinel guards.
        std::fs::write(mold.join("prompt-history.jsonl.migrated"), "").unwrap();
        let dropped2 = super::cleanup_stale_backups(&db, &mold, &mold).unwrap();
        assert_eq!(dropped2, 0);
        assert!(mold.join("prompt-history.jsonl.migrated").exists());
    }

    /// Item 1 (post-#265): the stripped-rewrite helper must back up the
    /// original `config.toml` to `.migrated` and write a new file that
    /// retains bootstrap fields but drops every key that now lives in the
    /// DB (expand, generate globals, per-model generation defaults, lora,
    /// scheduler, negative_prompt).
    #[test]
    fn rewrite_stripped_config_toml_strips_user_prefs_and_backs_up() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let original = r#"config_version = 1
default_model = "flux-dev:q4"
models_dir = "/opt/models"
server_port = 7680
default_width = 1536
default_height = 1024
default_steps = 30
embed_metadata = true
default_negative_prompt = "ugly"
t5_variant = "q8"
qwen3_variant = "bf16"

[expand]
enabled = true
temperature = 0.9

[logging]
level = "info"

[runpod]

[models."flux-dev:q4"]
transformer = "/opt/flux.gguf"
vae = "/opt/vae.safetensors"
default_steps = 20
default_guidance = 3.5
default_width = 1024
default_height = 1024
negative_prompt = "blurry"
lora = "/opt/lora.safetensors"
lora_scale = 0.8
scheduler = "euler-ancestral"
"#;
        std::fs::write(&config_path, original).unwrap();
        let cfg: Config = toml::from_str(original).unwrap();

        super::rewrite_stripped_config_toml(&cfg, &config_path).unwrap();

        // Backup mirrors the original untouched.
        let backup = std::fs::read_to_string(config_path.with_extension("toml.migrated"))
            .expect(".migrated backup should exist");
        assert!(backup.contains("default_width = 1536"));
        assert!(backup.contains("[expand]"));
        assert!(backup.contains("lora = \"/opt/lora.safetensors\""));
        assert!(backup.contains("scheduler = \"euler-ancestral\""));

        // Stripped file keeps the bootstrap surface.
        let stripped = std::fs::read_to_string(&config_path).unwrap();
        assert!(
            stripped.starts_with("# mold config"),
            "banner comment missing:\n{stripped}"
        );
        assert!(stripped.contains("default_model = \"flux-dev:q4\""));
        assert!(stripped.contains("models_dir = \"/opt/models\""));
        assert!(stripped.contains("server_port = 7680"));
        assert!(stripped.contains("[logging]"));
        assert!(stripped.contains("transformer = \"/opt/flux.gguf\""));
        assert!(stripped.contains("vae = \"/opt/vae.safetensors\""));

        // …and drops every user-preference field. Assertions use ` = ` /
        // ` =` suffixes because the banner comment at the top mentions
        // some of these key names in prose.
        let body = stripped
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !body.contains("default_width"),
            "global default_width must be stripped: {body}"
        );
        assert!(!body.contains("default_height"));
        assert!(!body.contains("default_steps"));
        assert!(!body.contains("embed_metadata"));
        assert!(!body.contains("default_negative_prompt"));
        assert!(!body.contains("t5_variant"));
        assert!(!body.contains("qwen3_variant"));
        assert!(!body.contains("[expand]"));
        assert!(
            !body.contains("lora = "),
            "per-model lora must be stripped: {body}"
        );
        assert!(!body.contains("lora_scale"));
        assert!(!body.contains("scheduler"));
        assert!(!body.contains("negative_prompt"));
        assert!(!body.contains("default_guidance"));
    }

    /// Item 1: calling the rewrite when no `config.toml` exists is a
    /// harmless no-op (e.g. a user running mold for the first time with
    /// MOLD_* env vars and no TOML).
    #[test]
    fn rewrite_stripped_config_toml_is_noop_without_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nothing.toml");
        let cfg = Config::default();
        super::rewrite_stripped_config_toml(&cfg, &path).unwrap();
        assert!(!path.exists());
        assert!(!path.with_extension("toml.migrated").exists());
    }

    /// Codex P2: the pre-#265 TUI wrote `{s:?}`.to_lowercase() (e.g.
    /// "eulerancestral") into the scheduler column. Existing DBs must
    /// still parse so a user's prior choice isn't lost on upgrade.
    #[test]
    fn scheduler_legacy_debug_form_still_parses() {
        use mold_core::Scheduler;
        assert_eq!(
            "eulerancestral".parse::<Scheduler>(),
            Ok(Scheduler::EulerAncestral)
        );
    }
}
