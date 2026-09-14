//! Typed key/value settings API backed by the `settings` table (v3 schema).
//!
//! This is where the user-facing slice of `Config` lives — expand settings,
//! global generation defaults, the last-used model — after the move off
//! `config.toml` and the legacy JSON session sidecar.
//!
//! Callers get a tiny typed surface — `get_str`, `set_bool`, `set_json<T>`,
//! etc. — plus a set of namespaced key constants so we don't sprinkle raw
//! strings through the codebase.

use anyhow::{Context, Result};
use mold_core::config::SchedulerSettings;
use rusqlite::{params, OptionalExtension};
use serde::{de::DeserializeOwned, Serialize};

use crate::db::MetadataDb;

// ------------------------------------------------------------------
// Namespaced keys. Grouped by surface so grep finds related state.
// ------------------------------------------------------------------

// Expand — the `[expand]` section of config.toml after it moves here.
pub const EXPAND_ENABLED: &str = "expand.enabled";
pub const EXPAND_TEMPERATURE: &str = "expand.temperature";
pub const EXPAND_TOP_P: &str = "expand.top_p";
pub const EXPAND_MAX_TOKENS: &str = "expand.max_tokens";
pub const EXPAND_THINKING: &str = "expand.thinking";
pub const EXPAND_SYSTEM_PROMPT: &str = "expand.system_prompt";
pub const EXPAND_BATCH_PROMPT: &str = "expand.batch_prompt";
pub const EXPAND_BACKEND: &str = "expand.backend";
pub const EXPAND_MODEL: &str = "expand.model";
pub const EXPAND_API_MODEL: &str = "expand.api_model";
pub const EXPAND_FAMILIES_JSON: &str = "expand.families_json";

// Generate — global generation defaults previously on `Config`.
pub const GENERATE_DEFAULT_WIDTH: &str = "generate.default_width";
pub const GENERATE_DEFAULT_HEIGHT: &str = "generate.default_height";
pub const GENERATE_DEFAULT_STEPS: &str = "generate.default_steps";
pub const GENERATE_DEFAULT_NEGATIVE_PROMPT: &str = "generate.default_negative_prompt";
pub const GENERATE_EMBED_METADATA: &str = "generate.embed_metadata";
pub const GENERATE_T5_VARIANT: &str = "generate.t5_variant";
pub const GENERATE_QWEN3_VARIANT: &str = "generate.qwen3_variant";
/// The model the last run used. Written after every generation and read as
/// tier 4 of default-model resolution, so `mold run` with no model named
/// resumes where the user left off. Migration v38 renamed it here from the
/// retired terminal app's namespace.
pub const GENERATE_LAST_MODEL: &str = "generate.last_model";

// Scheduler — profile-scoped behavior, never machine device identity.
pub const SCHEDULER_REPLAN_DEBOUNCE_MS: &str = "scheduler.replan_debounce_ms";
pub const SCHEDULER_REPLAN_MAX_DELAY_MS: &str = "scheduler.replan_max_delay_ms";
pub const SCHEDULER_WARM_WAIT_MAX_MS: &str = "scheduler.warm_wait_max_ms";

// Chain jobs — durable chain-job retention settings.
pub const CHAIN_JOBS_ARTIFACT_TTL_DAYS: &str = "chain.jobs_artifact_ttl_days";
pub const CHAIN_JOBS_ARTIFACT_TTL_DEFAULT: i64 = 7;

// Gallery — profile-scoped library behaviour. Unlike the chain TTL above,
// this key is registered in `mold_core::config_keys::ALL_KEYS` (section
// Gallery) so `mold config` and `/api/config` can reach it; the trash
// sweeper reads it fresh from here on every pass.
pub const GALLERY_TRASH_RETENTION_DAYS: &str =
    mold_core::config_keys::GALLERY_TRASH_RETENTION_DAYS_KEY;
pub const GALLERY_AUTHORITY_LOG: &str = mold_core::config_keys::GALLERY_AUTHORITY_LOG_KEY;
/// Days a trashed print is retained before purge; `0` keeps it forever.
pub const GALLERY_TRASH_RETENTION_DEFAULT: i64 = 30;

/// Days a HELD durable queue row is retained before the queue retention
/// sweeper purges it and releases its encrypted request media; `0` keeps
/// held rows forever. Registered in `mold_core::config_keys::ALL_KEYS`
/// (section Queue) and read fresh by the sweeper on every pass.
pub const QUEUE_HELD_RETENTION_DAYS: &str = mold_core::config_keys::QUEUE_HELD_RETENTION_DAYS_KEY;

/// Whether a titled print is also tagged with its title slug by the client
/// that submits it. Registered in `mold_core::config_keys::ALL_KEYS`
/// (section Generate). Unlike the older flat `generate.*` rows below, the
/// user-facing key and the DB key are the same string.
pub const GENERATE_AUTO_TAG_TITLE: &str = mold_core::config_keys::GENERATE_AUTO_TAG_TITLE_KEY;

// Config — migration sentinel for the one-shot `config.toml → DB` pass.
pub const CONFIG_MIGRATED_FROM_TOML: &str = "config.migrated_from_toml";

// Profile — meta setting that records which profile to activate at
// startup when `MOLD_PROFILE` isn't set. Always lives under profile
// `"default"` so we have a bootstrap-safe read.
pub const ACTIVE_PROFILE: &str = "profile.active";

// Migration bookkeeping — set to true once we've wiped the `.migrated`
// backups from the v5 legacy-state imports. Gates the cleanup pass so it
// runs at most once per process lifetime.
pub const BACKUPS_CLEANED_AT_V6: &str = "migration.backups_cleaned_at_v6";

/// The canonical "no profile specified" value. Every v6-migrated row
/// lands here, and reads without an explicit profile fall back to it.
pub const DEFAULT_PROFILE: &str = "default";

/// Scalar type tag stored alongside every row so readers can sanity-check
/// before parsing. Also prevents accidental cross-type writes (e.g.
/// `set_bool` over a key someone else is reading as JSON).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    String,
    Bool,
    Int,
    Float,
    Json,
}

impl ValueType {
    fn as_str(self) -> &'static str {
        match self {
            ValueType::String => "string",
            ValueType::Bool => "bool",
            ValueType::Int => "int",
            ValueType::Float => "float",
            ValueType::Json => "json",
        }
    }
}

/// Best-effort recorder for the last-used model, driven from the process-wide
/// DB handle. Returns silently when the DB is disabled or unavailable so
/// callers can wire this in without wrapping every invocation in an
/// `if let Some(db) = …`.
///
/// Replaces the legacy `Config::write_last_model()` sidecar write after
/// issue #265 moved the last-used model into the `settings` table.
pub fn record_last_model(model: &str) {
    let Some(db) = crate::global_db() else {
        return;
    };
    if let Err(e) = Settings::new(db).record_last_model(model) {
        tracing::warn!("settings.generate.last_model write failed: {e:#}");
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Typed view onto the `settings` table scoped to a single profile.
/// Cheap to construct — it's just a borrowed handle over the DB
/// connection plus the profile name.
///
/// Every row in v6+ is keyed on `(profile, key)`. [`Settings::new`]
/// resolves the active profile via env + meta-setting; tests and
/// profile-crossing code (e.g. `mold config --profile dev set …`)
/// should reach for [`Settings::for_profile`] to stay explicit.
pub struct Settings<'a> {
    db: &'a MetadataDb,
    profile: String,
}

impl<'a> Settings<'a> {
    /// View onto the active profile — resolved in priority order: the
    /// `MOLD_PROFILE` env var → the `profile.active` setting row under
    /// profile `"default"` → `"default"`.
    pub fn new(db: &'a MetadataDb) -> Self {
        Self {
            profile: resolve_active_profile(db),
            db,
        }
    }

    /// View onto an explicit profile. Use for cross-profile tooling or
    /// for the bootstrap read of `profile.active` itself.
    pub fn for_profile(db: &'a MetadataDb, profile: impl Into<String>) -> Self {
        Self {
            db,
            profile: profile.into(),
        }
    }

    /// The profile this view is scoped to.
    pub fn profile(&self) -> &str {
        &self.profile
    }

    // ---- setters -------------------------------------------------

    pub fn set_str(&self, key: &str, value: &str) -> Result<()> {
        self.upsert(key, value, ValueType::String)
    }

    pub fn set_bool(&self, key: &str, value: bool) -> Result<()> {
        self.upsert(key, if value { "1" } else { "0" }, ValueType::Bool)
    }

    pub fn set_int(&self, key: &str, value: i64) -> Result<()> {
        self.upsert(key, &value.to_string(), ValueType::Int)
    }

    pub fn set_float(&self, key: &str, value: f64) -> Result<()> {
        self.upsert(key, &value.to_string(), ValueType::Float)
    }

    pub fn set_json<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let s = serde_json::to_string(value)
            .with_context(|| format!("serializing settings value for key {key}"))?;
        self.upsert(key, &s, ValueType::Json)
    }

    /// Persist the scheduler's cross-validated timing tuple as one SQLite
    /// transaction. No reader can observe only part of the tuple.
    pub(crate) fn set_scheduler_timings_atomic(&self, scheduler: SchedulerSettings) -> Result<()> {
        let scheduler = scheduler.validate()?;
        let ts = now_ms();
        self.db.transact_immediate(|conn| {
            for (key, value) in [
                (
                    SCHEDULER_REPLAN_DEBOUNCE_MS,
                    i64::from(scheduler.replan_debounce_ms),
                ),
                (
                    SCHEDULER_REPLAN_MAX_DELAY_MS,
                    i64::from(scheduler.replan_max_delay_ms),
                ),
                (
                    SCHEDULER_WARM_WAIT_MAX_MS,
                    i64::from(scheduler.warm_wait_max_ms),
                ),
            ] {
                upsert_with_conn(
                    conn,
                    &self.profile,
                    key,
                    &value.to_string(),
                    ValueType::Int,
                    ts,
                )?;
            }
            Ok(())
        })
    }

    /// Read the scheduler tuple under one connection lock, so a concurrent
    /// atomic writer cannot produce a hybrid old/new snapshot.
    pub(crate) fn scheduler_timings(
        &self,
        base: SchedulerSettings,
    ) -> Result<(SchedulerSettings, bool)> {
        self.db
            .with_conn(|conn| scheduler_timings_with_conn(conn, &self.profile, base, None))
    }

    /// Validate the compiled default for `key` against all surviving timing
    /// rows and delete it in the same transaction. This serializes reset with
    /// both local and cross-process scheduler writers.
    pub(crate) fn reset_scheduler_timing_atomic(&self, key: &str) -> Result<bool> {
        anyhow::ensure!(
            matches!(
                key,
                SCHEDULER_REPLAN_DEBOUNCE_MS
                    | SCHEDULER_REPLAN_MAX_DELAY_MS
                    | SCHEDULER_WARM_WAIT_MAX_MS
            ),
            "unsupported scheduler timing key: {key}"
        );
        self.db.transact_immediate(|conn| {
            scheduler_timings_with_conn(
                conn,
                &self.profile,
                SchedulerSettings::default(),
                Some(key),
            )?;
            let changed = conn.execute(
                "DELETE FROM settings WHERE profile = ?1 AND key = ?2",
                params![&self.profile, key],
            )?;
            Ok(changed > 0)
        })
    }

    // ---- getters -------------------------------------------------

    pub fn get_str(&self, key: &str) -> Result<Option<String>> {
        self.db.with_conn(|conn| {
            let v: Option<String> = conn
                .query_row(
                    "SELECT value FROM settings WHERE profile = ?1 AND key = ?2",
                    params![&self.profile, key],
                    |r| r.get(0),
                )
                .ok();
            Ok(v)
        })
    }

    pub fn get_bool(&self, key: &str) -> Result<Option<bool>> {
        match self.get_str(key)? {
            None => Ok(None),
            Some(v) => Ok(Some(matches!(v.as_str(), "1" | "true" | "yes"))),
        }
    }

    pub fn get_int(&self, key: &str) -> Result<Option<i64>> {
        match self.get_str(key)? {
            None => Ok(None),
            Some(v) => Ok(v.parse::<i64>().ok()),
        }
    }

    pub fn get_float(&self, key: &str) -> Result<Option<f64>> {
        match self.get_str(key)? {
            None => Ok(None),
            Some(v) => Ok(v.parse::<f64>().ok()),
        }
    }

    pub fn get_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match self.get_str(key)? {
            None => Ok(None),
            Some(v) => Ok(serde_json::from_str(&v).ok()),
        }
    }

    // ---- management ---------------------------------------------

    pub fn delete(&self, key: &str) -> Result<bool> {
        self.db.with_conn(|conn| {
            let n = conn.execute(
                "DELETE FROM settings WHERE profile = ?1 AND key = ?2",
                params![&self.profile, key],
            )?;
            Ok(n > 0)
        })
    }

    /// Dump the current profile's rows — primarily for `mold config list`.
    pub fn list_all(&self) -> Result<Vec<(String, String, String)>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT key, value, value_type FROM settings
                 WHERE profile = ?1 ORDER BY key",
            )?;
            let mut rows = stmt.query(params![&self.profile])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push((row.get(0)?, row.get(1)?, row.get(2)?));
            }
            Ok(out)
        })
    }

    /// Record the last-used model for resume-on-launch. Writes to the
    /// [`GENERATE_LAST_MODEL`] row — the single source of truth after issue
    /// #265 retired the `$MOLD_HOME/last-model` sidecar.
    pub fn record_last_model(&self, model: &str) -> Result<()> {
        self.set_str(GENERATE_LAST_MODEL, model)
    }

    fn upsert(&self, key: &str, value: &str, ty: ValueType) -> Result<()> {
        let ts = now_ms();
        self.db.with_conn(|conn| {
            upsert_with_conn(conn, &self.profile, key, value, ty, ts)?;
            Ok(())
        })
    }
}

fn upsert_with_conn(
    conn: &rusqlite::Connection,
    profile: &str,
    key: &str,
    value: &str,
    ty: ValueType,
    timestamp_ms: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO settings (profile, key, value, value_type, updated_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(profile, key) DO UPDATE SET
            value = excluded.value,
            value_type = excluded.value_type,
            updated_at_ms = excluded.updated_at_ms",
        params![profile, key, value, ty.as_str(), timestamp_ms],
    )?;
    Ok(())
}

fn scheduler_timings_with_conn(
    conn: &rusqlite::Connection,
    profile: &str,
    mut candidate: SchedulerSettings,
    skipped_key: Option<&str>,
) -> Result<(SchedulerSettings, bool)> {
    let mut applied = false;
    for key in [
        SCHEDULER_REPLAN_DEBOUNCE_MS,
        SCHEDULER_REPLAN_MAX_DELAY_MS,
        SCHEDULER_WARM_WAIT_MAX_MS,
    ] {
        if skipped_key == Some(key) {
            continue;
        }
        let value: Option<String> = conn
            .query_row(
                "SELECT value FROM settings WHERE profile = ?1 AND key = ?2",
                params![profile, key],
                |row| row.get(0),
            )
            .optional()?;
        let Some(value) = value else {
            continue;
        };
        let value = value
            .parse::<i64>()
            .with_context(|| format!("{key} must be an integer"))?;
        let value = u32::try_from(value)
            .with_context(|| format!("{key} must be a non-negative integer"))?;
        match key {
            SCHEDULER_REPLAN_DEBOUNCE_MS => candidate.replan_debounce_ms = value,
            SCHEDULER_REPLAN_MAX_DELAY_MS => candidate.replan_max_delay_ms = value,
            SCHEDULER_WARM_WAIT_MAX_MS => candidate.warm_wait_max_ms = value,
            _ => unreachable!("scheduler settings key list is exhaustive"),
        }
        applied = true;
    }
    Ok((candidate.validate()?, applied))
}

/// Every profile with at least one settings row, plus [`DEFAULT_PROFILE`]
/// (always listed even when empty). Sorted ascending.
pub fn list_profiles(db: &MetadataDb) -> Result<Vec<String>> {
    let mut profiles: Vec<String> = db.with_conn(|conn| {
        let mut stmt = conn.prepare("SELECT DISTINCT profile FROM settings ORDER BY profile")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    })?;
    if !profiles.iter().any(|p| p == DEFAULT_PROFILE) {
        profiles.push(DEFAULT_PROFILE.to_string());
        profiles.sort();
    }
    Ok(profiles)
}

/// Serializes every test that mutates `MOLD_PROFILE`; the variable is
/// process-global.
#[cfg(test)]
pub(crate) fn profile_env_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// Persist the active profile as the `profile.active` meta-row (always
/// stored under [`DEFAULT_PROFILE`] so the bootstrap read can find it).
/// Note `MOLD_PROFILE` still wins over this at runtime.
pub fn set_active_profile(db: &MetadataDb, name: &str) -> Result<()> {
    Settings::for_profile(db, DEFAULT_PROFILE).set_str(ACTIVE_PROFILE, name)
}

/// Resolve the active profile for this process. Priority:
/// 1. `MOLD_PROFILE` env var (if set and non-empty)
/// 2. The `profile.active` setting row under profile `"default"`
/// 3. `"default"`
///
/// Reads always go through the default profile for step 2 so the meta
/// setting itself has a bootstrap-safe location.
pub fn resolve_active_profile(db: &MetadataDb) -> String {
    if let Ok(v) = std::env::var("MOLD_PROFILE") {
        let v = v.trim();
        if !v.is_empty() {
            return v.to_string();
        }
    }
    let default_view = Settings::for_profile(db, DEFAULT_PROFILE);
    if let Ok(Some(v)) = default_view.get_str(ACTIVE_PROFILE) {
        if !v.is_empty() {
            return v;
        }
    }
    DEFAULT_PROFILE.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn db() -> MetadataDb {
        MetadataDb::open_in_memory().unwrap()
    }

    #[test]
    fn string_roundtrip() {
        let db = db();
        let s = Settings::new(&db);
        s.set_str(EXPAND_MODEL, "dracula").unwrap();
        assert_eq!(s.get_str(EXPAND_MODEL).unwrap().as_deref(), Some("dracula"));
    }

    #[test]
    fn missing_key_returns_none() {
        let db = db();
        let s = Settings::new(&db);
        assert!(s.get_str("not.set").unwrap().is_none());
        assert!(s.get_bool("not.set").unwrap().is_none());
        assert!(s.get_int("not.set").unwrap().is_none());
        assert!(s.get_float("not.set").unwrap().is_none());
    }

    #[test]
    fn bool_roundtrip() {
        let db = db();
        let s = Settings::new(&db);
        s.set_bool(EXPAND_THINKING, true).unwrap();
        assert_eq!(s.get_bool(EXPAND_THINKING).unwrap(), Some(true));
        s.set_bool(EXPAND_THINKING, false).unwrap();
        assert_eq!(s.get_bool(EXPAND_THINKING).unwrap(), Some(false));
    }

    #[test]
    fn int_and_float_roundtrip() {
        let db = db();
        let s = Settings::new(&db);
        s.set_int(GENERATE_DEFAULT_WIDTH, 1024).unwrap();
        assert_eq!(s.get_int(GENERATE_DEFAULT_WIDTH).unwrap(), Some(1024));
        s.set_float(EXPAND_TEMPERATURE, 0.85).unwrap();
        assert!((s.get_float(EXPAND_TEMPERATURE).unwrap().unwrap() - 0.85).abs() < 1e-9);
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Sample {
        a: u32,
        b: Vec<String>,
    }

    #[test]
    fn json_roundtrip() {
        let db = db();
        let s = Settings::new(&db);
        let sample = Sample {
            a: 42,
            b: vec!["one".into(), "two".into()],
        };
        s.set_json("sample", &sample).unwrap();
        let got: Option<Sample> = s.get_json("sample").unwrap();
        assert_eq!(got, Some(sample));
    }

    #[test]
    fn setter_overwrites_existing_key() {
        let db = db();
        let s = Settings::new(&db);
        s.set_str(EXPAND_MODEL, "mocha").unwrap();
        s.set_str(EXPAND_MODEL, "latte").unwrap();
        assert_eq!(s.get_str(EXPAND_MODEL).unwrap().as_deref(), Some("latte"));
    }

    #[test]
    fn delete_returns_true_when_removed_false_otherwise() {
        let db = db();
        let s = Settings::new(&db);
        s.set_str(EXPAND_MODEL, "nord").unwrap();
        assert!(s.delete(EXPAND_MODEL).unwrap());
        assert!(!s.delete(EXPAND_MODEL).unwrap());
        assert!(s.get_str(EXPAND_MODEL).unwrap().is_none());
    }

    /// Item 3 (post-#265): `record_last_model` must land on the
    /// `GENERATE_LAST_MODEL` row so resume-on-launch reads the DB-backed
    /// value, not the retired `last-model` sidecar.
    #[test]
    fn record_last_model_lands_on_generate_last_model_row() {
        let db = db();
        let s = Settings::new(&db);
        s.record_last_model("flux-dev:q4").unwrap();
        assert_eq!(
            s.get_str(GENERATE_LAST_MODEL).unwrap().as_deref(),
            Some("flux-dev:q4")
        );
        // Overwrite path mirrors the normal write flow.
        s.record_last_model("qwen-image:q6").unwrap();
        assert_eq!(
            s.get_str(GENERATE_LAST_MODEL).unwrap().as_deref(),
            Some("qwen-image:q6")
        );
    }

    /// Item 5: Settings scoped to different profiles must not see each
    /// other's rows. `default` and `dev` can both carry `expand.model`
    /// without clobbering.
    #[test]
    fn settings_isolate_across_profiles() {
        let db = db();
        let default = Settings::for_profile(&db, DEFAULT_PROFILE);
        let dev = Settings::for_profile(&db, "dev");
        default.set_str(EXPAND_MODEL, "mocha").unwrap();
        dev.set_str(EXPAND_MODEL, "nord").unwrap();
        assert_eq!(
            default.get_str(EXPAND_MODEL).unwrap().as_deref(),
            Some("mocha")
        );
        assert_eq!(dev.get_str(EXPAND_MODEL).unwrap().as_deref(), Some("nord"));
        // Deleting from one profile does not affect the other.
        assert!(dev.delete(EXPAND_MODEL).unwrap());
        assert_eq!(
            default.get_str(EXPAND_MODEL).unwrap().as_deref(),
            Some("mocha")
        );
        assert!(dev.get_str(EXPAND_MODEL).unwrap().is_none());
    }

    /// Item 5: `list_all` only returns rows for the current profile.
    #[test]
    fn list_all_is_scoped_to_profile() {
        let db = db();
        Settings::for_profile(&db, DEFAULT_PROFILE)
            .set_str(EXPAND_MODEL, "mocha")
            .unwrap();
        Settings::for_profile(&db, "dev")
            .set_str(EXPAND_MODEL, "nord")
            .unwrap();
        let default_rows = Settings::for_profile(&db, DEFAULT_PROFILE)
            .list_all()
            .unwrap();
        assert_eq!(default_rows.len(), 1);
        assert_eq!(default_rows[0].1, "mocha");
    }

    /// Item 5: `resolve_active_profile` honours the env var, then the
    /// DB meta-row, then falls back to `"default"`. Env var takes top
    /// priority so a CLI `--profile` flag can force a profile without
    /// touching the DB.
    #[test]
    fn resolve_active_profile_priority_env_then_setting_then_default() {
        // Coordinate env-var mutation with other tests that read MOLD_PROFILE.
        let _g = crate::settings::profile_env_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var("MOLD_PROFILE").ok();
        let db = db();
        // 1. Env var wins outright.
        std::env::set_var("MOLD_PROFILE", "env-wins");
        assert_eq!(resolve_active_profile(&db), "env-wins");
        std::env::remove_var("MOLD_PROFILE");

        // 2. Env var unset — falls back to the meta row.
        Settings::for_profile(&db, DEFAULT_PROFILE)
            .set_str(ACTIVE_PROFILE, "stored-active")
            .unwrap();
        assert_eq!(resolve_active_profile(&db), "stored-active");

        // 3. Both missing — defaults to "default".
        Settings::for_profile(&db, DEFAULT_PROFILE)
            .delete(ACTIVE_PROFILE)
            .unwrap();
        assert_eq!(resolve_active_profile(&db), DEFAULT_PROFILE);

        if let Some(p) = prior {
            std::env::set_var("MOLD_PROFILE", p);
        }
    }

    #[test]
    fn list_all_returns_rows_sorted_by_key() {
        let db = db();
        let s = Settings::new(&db);
        s.set_str("z.key", "z").unwrap();
        s.set_str("a.key", "a").unwrap();
        s.set_str("m.key", "m").unwrap();
        let all = s.list_all().unwrap();
        let keys: Vec<_> = all.iter().map(|(k, _, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["a.key", "m.key", "z.key"]);
    }
}
