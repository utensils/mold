use anyhow::{anyhow, bail, Result};
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::config::Config;
// Re-exported for the tests below (`use super::*`), which construct
// per-model configs with explicit schedulers.
#[cfg(test)]
use mold_core::Scheduler;

use crate::theme;
use crate::AlreadyReported;

// ── Key registry / typed get-set / surface classification ──────────
//
// Moved to `mold_core::config_keys` so the server's `/api/config` surface
// shares the exact same registry, parsing, env-override detection, and
// DB-vs-TOML routing. Glob-imported so this module's `run_*` verbs (and
// the tests below via `use super::*`) keep their original names.
pub(crate) use mold_core::config_keys::*;

// ── Command implementations ─────────────────────────────────────────

pub fn run_list(json: bool) -> Result<()> {
    let config = Config::load_or_default();

    if json {
        return run_list_json(&config);
    }

    let mut current_section = "";

    for info in ALL_KEYS {
        if info.section != current_section {
            if !current_section.is_empty() {
                println!();
            }
            println!("{}", format!("[{}]", info.section).bold());
            current_section = info.section;
        }

        let value = get_static_value(&config, info.key)
            .map(|v| v.display())
            .unwrap_or_else(|_| "?".into());

        let badge = surface_badge(info.key);
        let env_note = if let Some((var, _)) = env_override_for(info.key) {
            format!("  (env: {})", var).dimmed().to_string()
        } else {
            String::new()
        };

        println!("  {} {:<26} = {}{}", badge, info.key, value, env_note);
    }

    // Per-model sections
    for model_name in config.models.keys() {
        println!();
        println!("{}", format!("[Model: {model_name}]").bold());
        for (field, _) in MODEL_FIELDS {
            let full_key = format!("models.{model_name}.{field}");
            let value = get_model_value(&config, &full_key)
                .map(|v| v.display())
                .unwrap_or_else(|_| "?".into());
            let badge = surface_badge(&full_key);
            println!("  {} {:<26} = {}", badge, full_key, value);
        }
    }

    Ok(())
}

/// Return a 6-char right-padded coloured surface badge for display in
/// `run_list`. `[env]` takes priority over `[db]` / `[file]` — an
/// environment override is the authoritative runtime value.
fn surface_badge(key: &str) -> String {
    if env_override_for(key).is_some() {
        return "[env] ".yellow().to_string();
    }
    let surface = model_field_surface(key).unwrap_or_else(|| surface_for_key(key));
    match surface {
        Surface::Db => "[db]  ".cyan().to_string(),
        Surface::File => "[file]".dimmed().to_string(),
    }
}

fn run_list_json(config: &Config) -> Result<()> {
    let mut map = serde_json::Map::new();

    // Under JSON each key becomes `{ "value": ..., "surface": "db|file|env" }`
    // so scripts can still tell where a value was read from. Raw values are
    // kept identical for backwards compat with pre-#265 consumers.
    for info in ALL_KEYS {
        if let Ok(val) = get_static_value(config, info.key) {
            map.insert(
                info.key.to_string(),
                surface_annotated_value(info.key, val.to_json()),
            );
        }
    }

    for model_name in config.models.keys() {
        for (field, _) in MODEL_FIELDS {
            let full_key = format!("models.{model_name}.{field}");
            if let Ok(val) = get_model_value(config, &full_key) {
                map.insert(
                    full_key.clone(),
                    surface_annotated_value(&full_key, val.to_json()),
                );
            }
        }
    }

    let json = serde_json::to_string_pretty(&serde_json::Value::Object(map))?;
    println!("{json}");
    Ok(())
}

fn surface_annotated_value(key: &str, value: serde_json::Value) -> serde_json::Value {
    let surface = if env_override_for(key).is_some() {
        "env"
    } else {
        model_field_surface(key)
            .unwrap_or_else(|| surface_for_key(key))
            .as_str()
    };
    serde_json::json!({ "value": value, "surface": surface })
}

pub fn run_get(key: &str, raw: bool) -> Result<()> {
    let config = Config::load_or_default();

    let value = match get_value(&config, key) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{} {e}", theme::icon_fail());
            return Err(AlreadyReported.into());
        }
    };

    if raw {
        println!("{}", value.raw());
        return Ok(());
    }

    println!("{} {} = {}", theme::icon_ok(), key.bold(), value.display());

    // Show env override info
    if let Some((var, env_val)) = env_override_for(key) {
        println!(
            "  {} overridden by {} = {}",
            theme::icon_bullet(),
            var.dimmed(),
            env_val.dimmed()
        );
    }

    Ok(())
}

pub fn run_set(key: &str, value: &str) -> Result<()> {
    // Validate the key exists before loading fresh config
    if find_static_key(key).is_none() && !key.starts_with("models.") {
        eprintln!("{} {}", theme::icon_fail(), unknown_key_error(key));
        return Err(AlreadyReported.into());
    }

    // For model keys, validate the field name
    if key.starts_with("models.") {
        if let Err(e) = parse_model_key(key) {
            eprintln!("{} {e}", theme::icon_fail());
            return Err(AlreadyReported.into());
        }
    }

    let mut config = Config::load_or_default();

    if let Err(e) = set_value(&mut config, key, value) {
        eprintln!("{} {e}", theme::icon_fail());
        return Err(AlreadyReported.into());
    }

    // Route persistence by surface: user-preference slices (expand.*,
    // generation defaults, per-model user prefs) go to the DB after
    // issue #265. Path/bootstrap/credential keys still land in the TOML.
    let persisted_surface = persist_value(&config, key)?;

    let display_value = get_value(&config, key)
        .map(|v| v.display())
        .unwrap_or_else(|_| value.to_string());

    println!(
        "{} Set {} = {} [{}]",
        theme::icon_done(),
        key.bold(),
        display_value,
        persisted_surface.as_str().dimmed(),
    );
    if key.starts_with("scheduler.") {
        println!(
            "  {} Restart the server to apply scheduler timing changes.",
            theme::icon_bullet()
        );
    }

    // Warn about env override
    if let Some((var, env_val)) = env_override_for(key) {
        eprintln!(
            "{} {} = {} will override this setting at runtime",
            theme::prefix_warning(),
            var.bold(),
            env_val,
        );
    }

    Ok(())
}

/// Persist the post-mutation `config` for this key. Returns the surface
/// that actually took the write.
fn persist_value(config: &Config, key: &str) -> Result<Surface> {
    // Per-model fields split by field name.
    if key.starts_with("models.") {
        if let Some(Surface::Db) = model_field_surface(key) {
            persist_model_field_to_db(config, key)?;
            return Ok(Surface::Db);
        }
        // Path fields still live in TOML. Use the bootstrap-only writer
        // so we don't re-serialize DB-owned fields (expand, generate
        // defaults, per-model lora/scheduler) that the hydrate hook
        // already mixed into `config` at load time.
        config.save_bootstrap_only()?;
        return Ok(Surface::File);
    }

    match surface_for_key(key) {
        Surface::Db => {
            persist_key_to_db(config, key)?;
            Ok(Surface::Db)
        }
        Surface::File => {
            // Same concern — see comment above.
            config.save_bootstrap_only()?;
            Ok(Surface::File)
        }
    }
}

fn persist_key_to_db(config: &Config, key: &str) -> Result<()> {
    let Some(db) = crate::metadata_db::handle() else {
        bail!(
            "settings DB is unavailable; key '{key}' requires MOLD_DB_DISABLE=0 \
             and a writable MOLD_HOME/mold.db"
        );
    };
    // Shared with the server's PUT /api/config/:key.
    mold_db::config_sync::persist_config_key(db, config, key)
}

fn persist_model_field_to_db(config: &Config, key: &str) -> Result<()> {
    let Some(db) = crate::metadata_db::handle() else {
        bail!(
            "settings DB is unavailable; key '{key}' requires MOLD_DB_DISABLE=0 \
             and a writable MOLD_HOME/mold.db"
        );
    };
    // Shared with the server's PUT /api/config/:key — resolves the model
    // name to canonical form and load-merge-saves the model_prefs row.
    mold_db::config_sync::persist_model_field(db, config, key)
}

/// `mold config where <key>` — print which surface owns the key so users
/// can disambiguate without reading source. Answers: "did my `mold config
/// set expand.enabled true` go to config.toml, mold.db, or is an env var
/// overriding it at runtime?"
pub fn run_where(key: &str) -> Result<()> {
    // Validate that this is a known key.
    if find_static_key(key).is_none() && !key.starts_with("models.") {
        eprintln!("{} {}", theme::icon_fail(), unknown_key_error(key));
        return Err(AlreadyReported.into());
    }

    let surface = if key.starts_with("models.") {
        if let Err(e) = parse_model_key(key) {
            eprintln!("{} {e}", theme::icon_fail());
            return Err(AlreadyReported.into());
        }
        model_field_surface(key).unwrap_or(Surface::File)
    } else {
        surface_for_key(key)
    };

    println!("{} {} = {}", theme::icon_ok(), key.bold(), surface.as_str());

    // Callout for env-var override at runtime — even DB/TOML values lose
    // to these, so it matters for disambiguation.
    if let Some((var, env_val)) = env_override_for(key) {
        println!(
            "  {} env {} = {} overrides at runtime",
            theme::icon_bullet(),
            var.dimmed(),
            env_val.dimmed()
        );
    }

    Ok(())
}

/// `mold config reset <key>` / `mold config reset --all` — drop a
/// DB-backed key (or every DB row) so the next read falls back to
/// `config.toml` / env / compiled default. Scoped to the active profile.
pub fn run_reset(key: Option<&str>, all: bool, yes: bool) -> Result<()> {
    let Some(db) = crate::metadata_db::handle() else {
        bail!(
            "settings DB is unavailable; reset requires MOLD_DB_DISABLE=0 \
             and a writable MOLD_HOME/mold.db"
        );
    };
    let profile = mold_db::resolve_active_profile(db);

    if all {
        if !yes && !confirm_reset_all(&profile)? {
            println!("{} aborted", theme::icon_info());
            return Ok(());
        }
        let dropped_settings = reset_user_setting_rows(db, &profile)?;
        let dropped_model_rows = mold_db::ModelPrefs::delete_all_in(db, &profile)?;
        println!(
            "{} reset profile {}: dropped {} setting(s) and {} model_prefs row(s)",
            theme::icon_ok(),
            profile.bold(),
            dropped_settings,
            dropped_model_rows
        );
        println!(
            "  {} Restart the server if scheduler timing settings were reset.",
            theme::icon_bullet()
        );
        return Ok(());
    }

    let Some(key) = key else {
        // Clap's required_unless_present should have caught this.
        bail!("missing key (or pass --all)");
    };

    // Validate key is known.
    if find_static_key(key).is_none() && !key.starts_with("models.") {
        eprintln!("{} {}", theme::icon_fail(), unknown_key_error(key));
        return Err(AlreadyReported.into());
    }

    // TOML-only keys are rejected — reset is a DB-surface operation.
    let surface = if key.starts_with("models.") {
        parse_model_key(key)?;
        model_field_surface(key).unwrap_or(Surface::File)
    } else {
        surface_for_key(key)
    };
    if surface != Surface::Db {
        eprintln!(
            "{} '{}' is stored in config.toml — edit the file or use `mold config set`",
            theme::icon_fail(),
            key
        );
        return Err(AlreadyReported.into());
    }

    if key.starts_with("models.") {
        reset_model_field(db, &profile, key)?;
    } else {
        reset_global_key(db, &profile, key)?;
    }

    println!(
        "{} reset {} (profile {})",
        theme::icon_ok(),
        key.bold(),
        profile.dimmed()
    );
    if key.starts_with("scheduler.") {
        println!(
            "  {} Restart the server to apply scheduler timing changes.",
            theme::icon_bullet()
        );
    }
    Ok(())
}

/// Keys the reset-all path must NOT delete from the settings table.
/// These are internal bookkeeping rows — removing them causes the next
/// launch to rerun one-shot migrations, which would re-import from the
/// already-stripped TOML and overwrite the original `.migrated` backup.
/// `profile.active` is also preserved so `--profile` selection survives
/// a purge.
const RESET_ALL_SENTINEL_KEYS: &[&str] = &[
    mold_db::settings::CONFIG_MIGRATED_FROM_TOML,
    mold_db::settings::TUI_MIGRATED_FROM_JSON,
    mold_db::settings::BACKUPS_CLEANED_AT_V6,
    mold_db::settings::ACTIVE_PROFILE,
];

/// Drop every user-facing settings row under `profile` while preserving
/// the migration/bookkeeping keys listed in
/// [`RESET_ALL_SENTINEL_KEYS`]. Returns the count of rows actually
/// deleted so the CLI can report it back to the user.
fn reset_user_setting_rows(db: &mold_db::MetadataDb, profile: &str) -> Result<usize> {
    let s = mold_db::Settings::for_profile(db, profile);
    let mut dropped = 0usize;
    for (k, _, _) in s.list_all()? {
        if RESET_ALL_SENTINEL_KEYS.contains(&k.as_str()) {
            continue;
        }
        if s.delete(&k)? {
            dropped += 1;
        }
    }
    Ok(dropped)
}

// Shared with the server's DELETE /api/config/:key — see
// `mold_db::config_sync` for the implementations.
fn reset_global_key(db: &mold_db::MetadataDb, profile: &str, key: &str) -> Result<()> {
    mold_db::config_sync::reset_global_key(db, profile, key)
}

fn reset_model_field(db: &mold_db::MetadataDb, profile: &str, key: &str) -> Result<()> {
    mold_db::config_sync::reset_model_field(db, profile, key)
}

fn confirm_reset_all(profile: &str) -> Result<bool> {
    use std::io::{self, Write};
    print!("Drop every DB-backed config row for profile '{profile}'? [y/N] ");
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

pub fn run_path() -> Result<()> {
    match Config::config_path() {
        Some(path) => {
            let exists = path.exists();
            if exists {
                println!("{} {}", theme::icon_ok(), path.display());
            } else {
                println!(
                    "{} {} (not yet created)",
                    theme::icon_info(),
                    path.display()
                );
            }
        }
        None => {
            eprintln!(
                "{} Cannot determine config path (HOME not set)",
                theme::icon_fail()
            );
            return Err(AlreadyReported.into());
        }
    }
    Ok(())
}

pub fn run_edit() -> Result<()> {
    let path = Config::config_path()
        .ok_or_else(|| anyhow!("cannot determine config path (HOME not set)"))?;

    // Create default config if it doesn't exist
    if !path.exists() {
        let config = Config::load_or_default();
        // Post-#265: the TOML is a bootstrap-only surface. Generate
        // defaults + expand live in the DB; writing a fresh file must
        // not seed stripped fields.
        config.save_bootstrap_only()?;
        println!(
            "{} Created default config at {}",
            theme::icon_info(),
            path.display()
        );
    }

    let editor = std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| "vi".to_string());

    // Split to handle EDITOR="code --wait" without shell injection risk
    let parts: Vec<&str> = editor.split_whitespace().collect();
    let (prog, args) = parts
        .split_first()
        .ok_or_else(|| anyhow!("EDITOR is empty"))?;
    let status = std::process::Command::new(prog)
        .args(args)
        .arg(&path)
        .status()?;

    if !status.success() {
        eprintln!("{} Editor exited with {}", theme::icon_fail(), status);
        return Err(AlreadyReported.into());
    }

    Ok(())
}

// ── Shell completion ────────────────────────────────────────────────

/// Return completion candidates for config key names.
pub fn complete_config_key() -> Vec<CompletionCandidate> {
    let mut candidates: Vec<CompletionCandidate> = ALL_KEYS
        .iter()
        .map(|k| CompletionCandidate::new(k.key))
        .collect();

    // Add per-model keys from the current config
    if let Some(config) = Config::config_path().and_then(|p| {
        if p.exists() {
            Some(Config::load_or_default())
        } else {
            None
        }
    }) {
        for model_name in config.models.keys() {
            for (field, _) in MODEL_FIELDS {
                candidates.push(CompletionCandidate::new(format!(
                    "models.{model_name}.{field}"
                )));
            }
        }
    }

    candidates
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;
    use std::collections::HashMap;

    fn test_config() -> Config {
        Config {
            config_version: 1,
            default_model: "flux2-klein".to_string(),
            models_dir: "~/.mold/models".to_string(),
            server_port: 7680,
            default_width: 768,
            default_height: 768,
            default_steps: 4,
            embed_metadata: true,
            t5_variant: None,
            umt5_variant: None,
            qwen3_variant: None,
            output_dir: None,
            media_roots: None,
            default_negative_prompt: None,
            expand: mold_core::ExpandSettings::default(),
            scheduler: Default::default(),
            gallery: Default::default(),
            queue: Default::default(),
            generate: Default::default(),
            logging: mold_core::LoggingConfig::default(),
            runpod: mold_core::runpod::RunPodSettings::default(),
            lambda: mold_core::lambda::LambdaSettings::default(),
            gpus: None,
            queue_size: None,
            models: HashMap::new(),
        }
    }

    fn test_config_with_model() -> Config {
        let mut config = test_config();
        config.models.insert(
            "test-model:q8".to_string(),
            mold_core::config::ModelConfig {
                default_steps: Some(20),
                default_guidance: Some(3.5),
                default_width: Some(1024),
                default_height: Some(1024),
                scheduler: Some(Scheduler::EulerAncestral),
                negative_prompt: Some("blurry".into()),
                lora: Some("/path/to/lora.safetensors".into()),
                lora_scale: Some(0.8),
                ..Default::default()
            },
        );
        config
    }

    // ── Key lookup tests ────────────────────────────────

    #[test]
    fn known_static_key_resolves() {
        for info in ALL_KEYS {
            assert!(
                find_static_key(info.key).is_some(),
                "key '{}' not found",
                info.key
            );
        }
    }

    #[test]
    fn unknown_key_errors() {
        let config = test_config();
        assert!(get_value(&config, "foo.bar").is_err());
        assert!(get_value(&config, "nonexistent").is_err());
    }

    #[test]
    fn model_key_parses_correctly() {
        let (model, field, _) = parse_model_key("models.flux-dev:q4.default_steps").unwrap();
        assert_eq!(model, "flux-dev:q4");
        assert_eq!(field, "default_steps");
    }

    #[test]
    fn model_key_unknown_field_errors() {
        assert!(parse_model_key("models.flux-dev:q4.bogus_field").is_err());
    }

    #[test]
    fn all_keys_count() {
        // 12 General + 8 Expand + 3 Scheduler + 1 Gallery + 1 Queue +
        // 1 Generate + 4 Logging + 8 RunPod + 9 Lambda static keys.
        // General gained `umt5_variant` with the Wan quantized encoder (#778);
        // Gallery gained `trash_retention_days` with the Library trash;
        // Generate gained `auto_tag_title` with creation-time filing;
        // Queue gained `held_retention_days` with durable held-row retention.
        assert_eq!(ALL_KEYS.len(), 47);
    }

    #[test]
    fn runpod_keys_registered() {
        for key in [
            "runpod.api_key",
            "runpod.default_gpu",
            "runpod.default_datacenter",
            "runpod.default_network_volume_id",
            "runpod.auto_teardown",
            "runpod.auto_teardown_idle_mins",
            "runpod.cost_alert_usd",
            "runpod.endpoint",
        ] {
            assert!(
                find_static_key(key).is_some(),
                "key {key} not registered in ALL_KEYS",
            );
        }
    }

    #[test]
    fn lambda_keys_registered() {
        for key in [
            "lambda.api_key",
            "lambda.endpoint",
            "lambda.image_repository",
            "lambda.ssh_key_name",
            "lambda.ssh_private_key_path",
            "lambda.filesystem_prefix",
            "lambda.filesystem_mount_path",
            "lambda.confirm_hourly_usd",
            "lambda.local_port",
        ] {
            assert!(
                find_static_key(key).is_some(),
                "key {key} not registered in ALL_KEYS",
            );
        }
    }

    #[test]
    fn lambda_api_key_roundtrip_is_redacted() {
        let mut config = test_config();
        set_value(&mut config, "lambda.api_key", "my-secret-key").unwrap();
        assert_eq!(config.lambda.api_key.as_deref(), Some("my-secret-key"));
        let v = get_value(&config, "lambda.api_key").unwrap();
        assert_eq!(v.raw(), "<set>");
        set_value(&mut config, "lambda.api_key", "none").unwrap();
        assert!(config.lambda.api_key.is_none());
    }

    #[test]
    fn runpod_api_key_roundtrip() {
        let mut config = test_config();
        set_value(&mut config, "runpod.api_key", "my-secret-key").unwrap();
        assert_eq!(config.runpod.api_key.as_deref(), Some("my-secret-key"));
        // Getter redacts the actual value.
        let v = get_value(&config, "runpod.api_key").unwrap();
        assert_eq!(v.raw(), "<set>");
        // "none" clears it.
        set_value(&mut config, "runpod.api_key", "none").unwrap();
        assert!(config.runpod.api_key.is_none());
    }

    #[test]
    fn runpod_auto_teardown_bool() {
        let mut config = test_config();
        set_value(&mut config, "runpod.auto_teardown", "true").unwrap();
        assert!(config.runpod.auto_teardown);
        set_value(&mut config, "runpod.auto_teardown", "off").unwrap();
        assert!(!config.runpod.auto_teardown);
    }

    #[test]
    fn runpod_cost_alert_bounds() {
        let mut config = test_config();
        set_value(&mut config, "runpod.cost_alert_usd", "2.5").unwrap();
        assert_eq!(config.runpod.cost_alert_usd, 2.5);
        assert!(set_value(&mut config, "runpod.cost_alert_usd", "-1").is_err());
    }

    // ── Value get tests ─────────────────────────────────

    #[test]
    fn get_default_model() {
        let config = test_config();
        let val = get_value(&config, "default_model").unwrap();
        assert_eq!(val.raw(), "flux2-klein");
    }

    #[test]
    fn get_server_port() {
        let config = test_config();
        let val = get_value(&config, "server_port").unwrap();
        assert_eq!(val.raw(), "7680");
    }

    #[test]
    fn get_embed_metadata() {
        let config = test_config();
        let val = get_value(&config, "embed_metadata").unwrap();
        assert_eq!(val.raw(), "true");
    }

    #[test]
    fn get_output_dir_unset() {
        let config = test_config();
        let val = get_value(&config, "output_dir").unwrap();
        assert_eq!(val.display(), "(not set)");
        assert_eq!(val.raw(), "");
    }

    #[test]
    fn get_output_dir_set() {
        let mut config = test_config();
        config.output_dir = Some("/tmp/output".into());
        let val = get_value(&config, "output_dir").unwrap();
        assert_eq!(val.raw(), "/tmp/output");
    }

    #[test]
    fn get_expand_temperature() {
        let config = test_config();
        let val = get_value(&config, "expand.temperature").unwrap();
        assert_eq!(val.raw(), "0.7");
    }

    #[test]
    fn get_expand_top_p() {
        let config = test_config();
        let val = get_value(&config, "expand.top_p").unwrap();
        assert_eq!(val.raw(), "0.9");
    }

    #[test]
    fn get_logging_level() {
        let config = test_config();
        let val = get_value(&config, "logging.level").unwrap();
        assert_eq!(val.raw(), "info");
    }

    #[test]
    fn get_logging_max_days() {
        let config = test_config();
        let val = get_value(&config, "logging.max_days").unwrap();
        assert_eq!(val.raw(), "7");
    }

    #[test]
    fn get_t5_variant_default() {
        let config = test_config();
        let val = get_value(&config, "t5_variant").unwrap();
        assert_eq!(val.raw(), "auto");
    }

    #[test]
    fn get_model_field() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.default_steps").unwrap();
        assert_eq!(val.raw(), "20");
    }

    #[test]
    fn get_model_field_guidance() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.default_guidance").unwrap();
        assert_eq!(val.raw(), "3.5");
    }

    #[test]
    fn get_model_field_scheduler() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.scheduler").unwrap();
        assert_eq!(val.raw(), "euler-ancestral");
    }

    #[test]
    fn get_model_field_lora() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.lora").unwrap();
        assert_eq!(val.raw(), "/path/to/lora.safetensors");
    }

    #[test]
    fn get_model_field_lora_scale() {
        let config = test_config_with_model();
        let val = get_value(&config, "models.test-model:q8.lora_scale").unwrap();
        assert_eq!(val.raw(), "0.8");
    }

    #[test]
    fn get_model_field_missing_model() {
        let config = test_config();
        assert!(get_value(&config, "models.nonexistent:q8.default_steps").is_err());
    }

    // ── Value set tests ─────────────────────────────────

    #[test]
    fn set_string_field() {
        let mut config = test_config();
        set_value(&mut config, "default_model", "sd15:fp16").unwrap();
        assert_eq!(config.default_model, "sd15:fp16");
    }

    #[test]
    fn set_bool_true() {
        let mut config = test_config();
        config.expand.enabled = false;
        set_value(&mut config, "expand.enabled", "true").unwrap();
        assert!(config.expand.enabled);
    }

    #[test]
    fn set_bool_false() {
        let mut config = test_config();
        set_value(&mut config, "embed_metadata", "false").unwrap();
        assert!(!config.embed_metadata);
    }

    #[test]
    fn set_bool_on() {
        let mut config = test_config();
        config.expand.enabled = false;
        set_value(&mut config, "expand.enabled", "on").unwrap();
        assert!(config.expand.enabled);
    }

    #[test]
    fn set_bool_off() {
        let mut config = test_config();
        set_value(&mut config, "embed_metadata", "off").unwrap();
        assert!(!config.embed_metadata);
    }

    #[test]
    fn set_bool_invalid() {
        let mut config = test_config();
        assert!(set_value(&mut config, "embed_metadata", "maybe").is_err());
    }

    #[test]
    fn set_u16_valid() {
        let mut config = test_config();
        set_value(&mut config, "server_port", "8080").unwrap();
        assert_eq!(config.server_port, 8080);
    }

    #[test]
    fn set_u16_out_of_range() {
        let mut config = test_config();
        assert!(set_value(&mut config, "server_port", "0").is_err());
        assert!(set_value(&mut config, "server_port", "99999").is_err());
    }

    #[test]
    fn set_u32_valid() {
        let mut config = test_config();
        set_value(&mut config, "default_steps", "50").unwrap();
        assert_eq!(config.default_steps, 50);
    }

    #[test]
    fn set_u32_out_of_range() {
        let mut config = test_config();
        assert!(set_value(&mut config, "default_steps", "0").is_err());
        assert!(set_value(&mut config, "default_steps", "9999").is_err());
    }

    #[test]
    fn set_f64_valid() {
        let mut config = test_config();
        set_value(&mut config, "expand.temperature", "1.5").unwrap();
        assert!((config.expand.temperature - 1.5).abs() < 0.001);
    }

    #[test]
    fn set_f64_out_of_range() {
        let mut config = test_config();
        assert!(set_value(&mut config, "expand.temperature", "3.0").is_err());
        assert!(set_value(&mut config, "expand.temperature", "-1.0").is_err());
    }

    #[test]
    fn set_optional_none_clears() {
        let mut config = test_config();
        config.output_dir = Some("/tmp".into());
        set_value(&mut config, "output_dir", "none").unwrap();
        assert!(config.output_dir.is_none());
    }

    #[test]
    fn set_optional_value() {
        let mut config = test_config();
        set_value(&mut config, "output_dir", "/tmp/images").unwrap();
        assert_eq!(config.output_dir, Some("/tmp/images".into()));
    }

    #[test]
    fn set_model_field_existing() {
        let mut config = test_config_with_model();
        set_value(&mut config, "models.test-model:q8.default_steps", "30").unwrap();
        assert_eq!(
            config.models.get("test-model:q8").unwrap().default_steps,
            Some(30)
        );
    }

    #[test]
    fn set_model_field_creates_entry() {
        let mut config = test_config();
        assert!(!config.models.contains_key("new-model:q4"));
        set_value(&mut config, "models.new-model:q4.default_steps", "25").unwrap();
        assert_eq!(
            config.models.get("new-model:q4").unwrap().default_steps,
            Some(25)
        );
    }

    #[test]
    fn set_model_scheduler() {
        let mut config = test_config_with_model();
        set_value(&mut config, "models.test-model:q8.scheduler", "uni-pc").unwrap();
        assert_eq!(
            config.models.get("test-model:q8").unwrap().scheduler,
            Some(Scheduler::UniPc)
        );
    }

    #[test]
    fn set_model_scheduler_none() {
        let mut config = test_config_with_model();
        set_value(&mut config, "models.test-model:q8.scheduler", "none").unwrap();
        assert!(config
            .models
            .get("test-model:q8")
            .unwrap()
            .scheduler
            .is_none());
    }

    #[test]
    fn set_model_scheduler_invalid() {
        let mut config = test_config_with_model();
        assert!(set_value(&mut config, "models.test-model:q8.scheduler", "invalid").is_err());
    }

    #[test]
    fn set_model_lora_scale() {
        let mut config = test_config_with_model();
        set_value(&mut config, "models.test-model:q8.lora_scale", "1.5").unwrap();
        assert!(
            (config
                .models
                .get("test-model:q8")
                .unwrap()
                .lora_scale
                .unwrap()
                - 1.5)
                .abs()
                < 0.001
        );
    }

    // ── Validation tests ────────────────────────────────

    #[test]
    fn validate_port_range() {
        assert!(parse_u16("1", 1, 65535, "port").is_ok());
        assert!(parse_u16("65535", 1, 65535, "port").is_ok());
        assert!(parse_u16("0", 1, 65535, "port").is_err());
    }

    #[test]
    fn validate_bool_values() {
        assert!(parse_bool("true", "k").is_ok());
        assert!(parse_bool("false", "k").is_ok());
        assert!(parse_bool("on", "k").is_ok());
        assert!(parse_bool("off", "k").is_ok());
        assert!(parse_bool("1", "k").is_ok());
        assert!(parse_bool("0", "k").is_ok());
        assert!(parse_bool("yes", "k").is_ok());
        assert!(parse_bool("no", "k").is_ok());
        assert!(parse_bool("maybe", "k").is_err());
    }

    #[test]
    fn validate_log_level_enum() {
        let mut config = test_config();
        set_value(&mut config, "logging.level", "debug").unwrap();
        assert_eq!(config.logging.level, "debug");
        assert!(set_value(&mut config, "logging.level", "verbose").is_err());
    }

    #[test]
    fn validate_t5_variant_enum() {
        let mut config = test_config();
        set_value(&mut config, "t5_variant", "q8").unwrap();
        assert_eq!(config.t5_variant, Some("q8".into()));
        set_value(&mut config, "t5_variant", "auto").unwrap();
        assert!(config.t5_variant.is_none()); // "auto" = None
        assert!(set_value(&mut config, "t5_variant", "invalid").is_err());
    }

    #[test]
    fn validate_qwen3_variant_enum() {
        let mut config = test_config();
        set_value(&mut config, "qwen3_variant", "bf16").unwrap();
        assert_eq!(config.qwen3_variant, Some("bf16".into()));
        assert!(set_value(&mut config, "qwen3_variant", "fp16").is_err()); // not valid for qwen3
    }

    #[test]
    fn validate_temperature_range() {
        assert!(parse_f64("0.0", 0.0, 2.0, "t").is_ok());
        assert!(parse_f64("2.0", 0.0, 2.0, "t").is_ok());
        assert!(parse_f64("2.1", 0.0, 2.0, "t").is_err());
        assert!(parse_f64("-0.1", 0.0, 2.0, "t").is_err());
    }

    // ── Env override tests ──────────────────────────────

    #[test]
    fn env_override_absent() {
        // Most env vars won't be set in test environment
        assert!(env_override_for("server_port").is_none()); // no env var mapped
        assert!(env_override_for("logging.level").is_none());
    }

    // ── Completion tests ────────────────────────────────

    #[test]
    fn complete_returns_candidates() {
        let candidates = complete_config_key();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn complete_includes_static_keys() {
        let candidates = complete_config_key();
        let keys: Vec<String> = candidates
            .iter()
            .map(|c| c.get_value().to_string_lossy().to_string())
            .collect();
        assert!(keys.contains(&"expand.backend".to_string()));
        assert!(keys.contains(&"server_port".to_string()));
        assert!(keys.contains(&"logging.level".to_string()));
    }

    // ── ConfigValue formatting tests ────────────────────

    #[test]
    fn config_value_display_none() {
        assert_eq!(ConfigValue::None.display(), "(not set)");
        assert_eq!(ConfigValue::None.raw(), "");
    }

    // ── Surface classification tests ────────────────────

    #[test]
    fn surface_for_tui_and_expand_and_generate_is_db() {
        assert_eq!(surface_for_key("tui.theme"), Surface::Db);
        assert_eq!(surface_for_key("expand.enabled"), Surface::Db);
        assert_eq!(surface_for_key("expand.backend"), Surface::Db);
        assert_eq!(surface_for_key("generate.default_width"), Surface::Db);
        assert_eq!(
            surface_for_key("model_prefs.flux-dev:q4.width"),
            Surface::Db
        );
    }

    #[test]
    fn surface_for_flat_generation_globals_is_db() {
        assert_eq!(surface_for_key("default_width"), Surface::Db);
        assert_eq!(surface_for_key("default_height"), Surface::Db);
        assert_eq!(surface_for_key("default_steps"), Surface::Db);
        assert_eq!(surface_for_key("embed_metadata"), Surface::Db);
        assert_eq!(surface_for_key("default_negative_prompt"), Surface::Db);
        assert_eq!(surface_for_key("t5_variant"), Surface::Db);
        assert_eq!(surface_for_key("qwen3_variant"), Surface::Db);
    }

    #[test]
    fn surface_for_bootstrap_keys_is_file() {
        assert_eq!(surface_for_key("default_model"), Surface::File);
        assert_eq!(surface_for_key("models_dir"), Surface::File);
        assert_eq!(surface_for_key("output_dir"), Surface::File);
        assert_eq!(surface_for_key("server_port"), Surface::File);
        assert_eq!(surface_for_key("logging.level"), Surface::File);
        assert_eq!(surface_for_key("runpod.api_key"), Surface::File);
    }

    #[test]
    fn model_field_surface_splits_db_vs_path() {
        // User preference fields → DB.
        for field in [
            "default_steps",
            "default_guidance",
            "default_width",
            "default_height",
            "scheduler",
            "negative_prompt",
            "lora",
            "lora_scale",
        ] {
            let key = format!("models.flux-dev:q4.{field}");
            assert_eq!(model_field_surface(&key), Some(Surface::Db), "key={key}");
        }
        // Path fields stay in TOML.
        assert_eq!(
            model_field_surface("models.flux-dev:q4.transformer"),
            Some(Surface::File)
        );
        // Non-model keys return None.
        assert_eq!(model_field_surface("expand.enabled"), None);
    }

    #[test]
    fn config_value_json_types() {
        assert_eq!(ConfigValue::Bool(true).to_json(), serde_json::json!(true));
        assert_eq!(ConfigValue::U32(42).to_json(), serde_json::json!(42));
        assert_eq!(
            ConfigValue::String("hello".into()).to_json(),
            serde_json::json!("hello")
        );
        assert!(ConfigValue::None.to_json().is_null());
    }

    // ── Integration tests ───────────────────────────────

    /// Codex P2 regression: clearing a per-model negative prompt via
    /// `mold config set models.<name>.negative_prompt none` must null
    /// out the stored `last_negative` row, not leave it untouched.
    /// Exercises the unconditional assignment in
    /// `persist_model_field_to_db`.
    #[test]
    fn model_config_sync_clear_nulls_last_negative() {
        // Pre-seed prefs with a negative prompt, then simulate the
        // "user cleared it via CLI" path by mutating the in-memory
        // ModelConfig and re-snapshotting.
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        mold_db::ModelPrefs {
            last_negative: Some("stale".into()),
            width: Some(1024),
            ..Default::default()
        }
        .save(&db, "flux-dev:q4")
        .unwrap();

        // Mimic persist_model_field_to_db's load-merge-save with a
        // cleared negative_prompt on ModelConfig.
        let mc = mold_core::config::ModelConfig {
            negative_prompt: None,
            default_width: Some(1024),
            ..Default::default()
        };
        let mut prefs = mold_db::ModelPrefs::load(&db, "flux-dev:q4")
            .unwrap()
            .unwrap_or_default();
        prefs.width = mc.default_width;
        prefs.last_negative = mc.negative_prompt.clone();
        prefs.save(&db, "flux-dev:q4").unwrap();

        let loaded = mold_db::ModelPrefs::load(&db, "flux-dev:q4")
            .unwrap()
            .unwrap();
        assert_eq!(loaded.last_negative, None, "clear must null the row");
        assert_eq!(loaded.width, Some(1024), "unrelated fields preserved");
    }

    #[test]
    fn set_persists_to_disk() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = std::env::temp_dir().join(format!(
            "mold-config-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        let prev_home = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", &tmp) };

        let mut config = Config::load_or_default();
        set_value(&mut config, "server_port", "9999").unwrap();
        config.save().unwrap();

        let config_path = tmp.join("config.toml");
        let written = std::fs::read_to_string(&config_path).unwrap();
        assert!(written.contains("server_port = 9999"));

        // Cleanup
        match prev_home {
            Some(v) => unsafe { std::env::set_var("MOLD_HOME", v) },
            None => unsafe { std::env::remove_var("MOLD_HOME") },
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Full round-trip for every global field ──────────

    #[test]
    fn roundtrip_all_global_fields() {
        let mut config = test_config();

        // String fields
        set_value(&mut config, "default_model", "sd15:fp16").unwrap();
        assert_eq!(
            get_value(&config, "default_model").unwrap().raw(),
            "sd15:fp16"
        );

        set_value(&mut config, "models_dir", "/new/path").unwrap();
        assert_eq!(get_value(&config, "models_dir").unwrap().raw(), "/new/path");

        // Optional string
        set_value(&mut config, "output_dir", "/out").unwrap();
        assert_eq!(get_value(&config, "output_dir").unwrap().raw(), "/out");
        set_value(&mut config, "output_dir", "none").unwrap();
        assert_eq!(get_value(&config, "output_dir").unwrap().raw(), "");

        set_value(&mut config, "default_negative_prompt", "ugly").unwrap();
        assert_eq!(
            get_value(&config, "default_negative_prompt").unwrap().raw(),
            "ugly"
        );

        // Numeric
        set_value(&mut config, "server_port", "9090").unwrap();
        assert_eq!(get_value(&config, "server_port").unwrap().raw(), "9090");

        set_value(&mut config, "default_width", "1024").unwrap();
        assert_eq!(get_value(&config, "default_width").unwrap().raw(), "1024");

        set_value(&mut config, "default_height", "512").unwrap();
        assert_eq!(get_value(&config, "default_height").unwrap().raw(), "512");

        set_value(&mut config, "default_steps", "50").unwrap();
        assert_eq!(get_value(&config, "default_steps").unwrap().raw(), "50");

        // Bool
        set_value(&mut config, "embed_metadata", "false").unwrap();
        assert_eq!(get_value(&config, "embed_metadata").unwrap().raw(), "false");

        // Variant enums
        set_value(&mut config, "t5_variant", "q4").unwrap();
        assert_eq!(get_value(&config, "t5_variant").unwrap().raw(), "q4");

        set_value(&mut config, "qwen3_variant", "q8").unwrap();
        assert_eq!(get_value(&config, "qwen3_variant").unwrap().raw(), "q8");
    }

    #[test]
    fn roundtrip_all_expand_fields() {
        let mut config = test_config();

        set_value(&mut config, "expand.enabled", "true").unwrap();
        assert_eq!(get_value(&config, "expand.enabled").unwrap().raw(), "true");

        set_value(&mut config, "expand.backend", "http://localhost:11434").unwrap();
        assert_eq!(
            get_value(&config, "expand.backend").unwrap().raw(),
            "http://localhost:11434"
        );

        set_value(&mut config, "expand.model", "qwen3:q4").unwrap();
        assert_eq!(
            get_value(&config, "expand.model").unwrap().raw(),
            "qwen3:q4"
        );

        set_value(&mut config, "expand.api_model", "gpt-4o").unwrap();
        assert_eq!(
            get_value(&config, "expand.api_model").unwrap().raw(),
            "gpt-4o"
        );

        set_value(&mut config, "expand.temperature", "0.5").unwrap();
        assert_eq!(
            get_value(&config, "expand.temperature").unwrap().raw(),
            "0.5"
        );

        set_value(&mut config, "expand.top_p", "0.8").unwrap();
        assert_eq!(get_value(&config, "expand.top_p").unwrap().raw(), "0.8");

        set_value(&mut config, "expand.max_tokens", "500").unwrap();
        assert_eq!(
            get_value(&config, "expand.max_tokens").unwrap().raw(),
            "500"
        );

        set_value(&mut config, "expand.thinking", "true").unwrap();
        assert_eq!(get_value(&config, "expand.thinking").unwrap().raw(), "true");
    }

    #[test]
    fn roundtrip_all_logging_fields() {
        let mut config = test_config();

        set_value(&mut config, "logging.level", "debug").unwrap();
        assert_eq!(get_value(&config, "logging.level").unwrap().raw(), "debug");

        set_value(&mut config, "logging.file", "true").unwrap();
        assert_eq!(get_value(&config, "logging.file").unwrap().raw(), "true");

        set_value(&mut config, "logging.dir", "/var/log/mold").unwrap();
        assert_eq!(
            get_value(&config, "logging.dir").unwrap().raw(),
            "/var/log/mold"
        );

        set_value(&mut config, "logging.max_days", "30").unwrap();
        assert_eq!(get_value(&config, "logging.max_days").unwrap().raw(), "30");
    }

    #[test]
    fn roundtrip_all_model_fields() {
        let mut config = test_config_with_model();
        let prefix = "models.test-model:q8";

        set_value(&mut config, &format!("{prefix}.default_steps"), "30").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.default_steps"))
                .unwrap()
                .raw(),
            "30"
        );

        set_value(&mut config, &format!("{prefix}.default_guidance"), "7.5").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.default_guidance"))
                .unwrap()
                .raw(),
            "7.5"
        );

        set_value(&mut config, &format!("{prefix}.default_width"), "512").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.default_width"))
                .unwrap()
                .raw(),
            "512"
        );

        set_value(&mut config, &format!("{prefix}.default_height"), "512").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.default_height"))
                .unwrap()
                .raw(),
            "512"
        );

        set_value(&mut config, &format!("{prefix}.scheduler"), "ddim").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.scheduler"))
                .unwrap()
                .raw(),
            "ddim"
        );

        set_value(
            &mut config,
            &format!("{prefix}.negative_prompt"),
            "bad quality",
        )
        .unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.negative_prompt"))
                .unwrap()
                .raw(),
            "bad quality"
        );

        set_value(
            &mut config,
            &format!("{prefix}.lora"),
            "/new/lora.safetensors",
        )
        .unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.lora")).unwrap().raw(),
            "/new/lora.safetensors"
        );

        set_value(&mut config, &format!("{prefix}.lora_scale"), "1.2").unwrap();
        assert_eq!(
            get_value(&config, &format!("{prefix}.lora_scale"))
                .unwrap()
                .raw(),
            "1.2"
        );
    }

    /// Codex review (P2): after `config.toml` is stripped to the
    /// bootstrap-only surface, a subsequent `mold config set server_port
    /// 9999` must keep the TOML bootstrap-only. `Config::save()` would
    /// re-serialize the hydrated view and resurrect `[expand]` /
    /// generation defaults / per-model lora rows that now belong to the
    /// DB. Persistence for file-backed writes goes through
    /// `save_bootstrap_only_to`.
    #[test]
    fn file_surface_set_keeps_toml_bootstrap_only() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("config.toml");

        // Simulate a post-migration state: the TOML is bootstrap-only.
        let cfg = mold_core::Config {
            server_port: 7680,
            ..mold_core::Config::default()
        };
        cfg.save_bootstrap_only_to(&toml_path).unwrap();
        let pre = std::fs::read_to_string(&toml_path).unwrap();
        assert!(!pre
            .lines()
            .filter(|l| !l.starts_with('#'))
            .any(|l| l.contains("default_width") || l.contains("[expand]")));

        // Imagine the user then runs `mold config set server_port 8080`.
        // The hydrated runtime cfg carries expand + generate defaults,
        // but the file-surface writer must NOT push them into the TOML.
        let mut hydrated = mold_core::Config {
            server_port: 8080,
            default_width: 2048, // pretend hydrated from DB
            ..mold_core::Config::default()
        };
        hydrated.expand.enabled = true; // pretend hydrated from DB
                                        // Correct persist path for file-surface keys:
        hydrated.save_bootstrap_only_to(&toml_path).unwrap();

        let post = std::fs::read_to_string(&toml_path).unwrap();
        let body = post
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(body.contains("server_port = 8080"));
        assert!(
            !body.contains("default_width"),
            "stripped fields must not return: {body}"
        );
        assert!(!body.contains("[expand]"));
    }

    /// Codex review (P2): `mold config reset --all` must NOT delete the
    /// migration bookkeeping rows (`config.migrated_from_toml`,
    /// `tui.migrated_from_json`, `migration.backups_cleaned_at_v6`,
    /// `profile.active`). Otherwise the next launch thinks migration
    /// hasn't run, re-imports the (already-stripped) TOML, and the
    /// rewrite clobbers the original `.migrated` backup.
    #[test]
    fn reset_all_skips_migration_sentinel_keys() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let s = mold_db::Settings::for_profile(&db, "default");
        // Seed real user prefs + migration sentinels.
        s.set_str(mold_db::settings::TUI_THEME, "mocha").unwrap();
        s.set_bool(mold_db::settings::EXPAND_ENABLED, true).unwrap();
        s.set_bool(mold_db::settings::CONFIG_MIGRATED_FROM_TOML, true)
            .unwrap();
        s.set_bool(mold_db::settings::TUI_MIGRATED_FROM_JSON, true)
            .unwrap();
        s.set_bool(mold_db::settings::BACKUPS_CLEANED_AT_V6, false)
            .unwrap();

        let dropped = super::reset_user_setting_rows(&db, "default").unwrap();
        // User rows gone.
        assert_eq!(dropped, 2);
        assert!(s.get_str(mold_db::settings::TUI_THEME).unwrap().is_none());
        assert!(s
            .get_bool(mold_db::settings::EXPAND_ENABLED)
            .unwrap()
            .is_none());
        // Sentinels preserved — next launch must not re-migrate.
        assert_eq!(
            s.get_bool(mold_db::settings::CONFIG_MIGRATED_FROM_TOML)
                .unwrap(),
            Some(true)
        );
        assert_eq!(
            s.get_bool(mold_db::settings::TUI_MIGRATED_FROM_JSON)
                .unwrap(),
            Some(true)
        );
        assert_eq!(
            s.get_bool(mold_db::settings::BACKUPS_CLEANED_AT_V6)
                .unwrap(),
            Some(false)
        );
    }

    /// Item 7: reset a per-model DB field — the row stays, only the
    /// targeted field is nulled. Leaves other fields (width, scheduler,
    /// last_prompt) intact.
    #[test]
    fn reset_model_field_nulls_only_target_field() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        mold_db::ModelPrefs {
            width: Some(1024),
            steps: Some(20),
            guidance: Some(3.5),
            lora_path: Some("/opt/lora.safetensors".into()),
            lora_scale: Some(0.8),
            last_negative: Some("blurry".into()),
            last_prompt: Some("a cat".into()),
            ..Default::default()
        }
        .save_in(&db, "default", "flux-dev:q4")
        .unwrap();

        super::reset_model_field(&db, "default", "models.flux-dev:q4.lora").unwrap();

        let loaded = mold_db::ModelPrefs::load_in(&db, "default", "flux-dev:q4")
            .unwrap()
            .unwrap();
        // Targeted field dropped.
        assert_eq!(loaded.lora_path, None);
        // Unrelated fields survived.
        assert_eq!(loaded.width, Some(1024));
        assert_eq!(loaded.steps, Some(20));
        assert_eq!(loaded.lora_scale, Some(0.8));
        assert_eq!(loaded.last_negative.as_deref(), Some("blurry"));
        assert_eq!(loaded.last_prompt.as_deref(), Some("a cat"));
    }

    /// Item 7: reset a global DB key drops the row; hydrate falls back to
    /// the TOML / default value on the next load.
    #[test]
    fn reset_global_key_drops_settings_row() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let s = mold_db::Settings::for_profile(&db, "default");
        s.set_int(mold_db::settings::GENERATE_DEFAULT_WIDTH, 2048)
            .unwrap();
        assert_eq!(
            s.get_int(mold_db::settings::GENERATE_DEFAULT_WIDTH)
                .unwrap(),
            Some(2048)
        );

        super::reset_global_key(&db, "default", "default_width").unwrap();

        assert!(s
            .get_int(mold_db::settings::GENERATE_DEFAULT_WIDTH)
            .unwrap()
            .is_none());
    }

    /// Item 7: reset on an unknown per-model row is a no-op success, not
    /// a panic — matches `mold config reset` called on a model the user
    /// never set any DB prefs for.
    #[test]
    fn reset_model_field_noop_when_no_row() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        super::reset_model_field(&db, "default", "models.sdxl:fp16.default_steps").unwrap();
        assert!(mold_db::ModelPrefs::load_in(&db, "default", "sdxl:fp16")
            .unwrap()
            .is_none());
    }

    /// Item 6: the JSON form of `run_list` annotates each key with its
    /// surface. Structure is `{ "value": ..., "surface": "db|file|env" }`
    /// per key, so scripts and the TUI can tell which source owns a row.
    #[test]
    fn surface_annotated_value_tags_env_db_and_file() {
        // env takes top priority — simulate by temporarily setting the
        // matching MOLD_* env var for a known key.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("MOLD_DEFAULT_MODEL").ok();
        unsafe { std::env::set_var("MOLD_DEFAULT_MODEL", "forced-by-env") };
        let annotated =
            super::surface_annotated_value("default_model", serde_json::json!("flux2-klein:q8"));
        assert_eq!(annotated["surface"], "env");
        match prev {
            Some(v) => unsafe { std::env::set_var("MOLD_DEFAULT_MODEL", v) },
            None => unsafe { std::env::remove_var("MOLD_DEFAULT_MODEL") },
        }

        // DB surface: expand.enabled lives in the settings table.
        let annotated = super::surface_annotated_value("expand.enabled", serde_json::json!(false));
        assert_eq!(annotated["surface"], "db");

        // File surface: models_dir stays in TOML.
        let annotated =
            super::surface_annotated_value("models_dir", serde_json::json!("/opt/models"));
        assert_eq!(annotated["surface"], "file");

        // Per-model paths stay in TOML even though generation defaults don't.
        let annotated = super::surface_annotated_value(
            "models.flux-dev:q4.transformer",
            serde_json::json!("/opt/flux.gguf"),
        );
        assert_eq!(annotated["surface"], "file");
        let annotated = super::surface_annotated_value(
            "models.flux-dev:q4.default_steps",
            serde_json::json!(20),
        );
        assert_eq!(annotated["surface"], "db");
    }
}
