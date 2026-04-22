//! Typed key/value settings API backed by the `settings` table (v3 schema).
//!
//! This is where TUI preferences (theme, last model, view mode) and the
//! user-facing slice of `Config` (expand settings, global generation
//! defaults) live after the move off `tui-session.json` / `config.toml`.
//!
//! Callers get a tiny typed surface — `get_str`, `set_bool`, `set_json<T>`,
//! etc. — plus a set of namespaced key constants so we don't sprinkle raw
//! strings through the codebase.

use anyhow::{Context, Result};
use rusqlite::params;
use serde::{de::DeserializeOwned, Serialize};

use crate::db::MetadataDb;

// ------------------------------------------------------------------
// Namespaced keys. Grouped by surface so grep finds related state.
// ------------------------------------------------------------------

// TUI — persisted UI state the user expects to survive restarts.
pub const TUI_THEME: &str = "tui.theme";
pub const TUI_LAST_MODEL: &str = "tui.last_model";
pub const TUI_LAST_PROMPT: &str = "tui.last_prompt";
pub const TUI_LAST_NEGATIVE: &str = "tui.last_negative";
pub const TUI_NEGATIVE_COLLAPSED: &str = "tui.negative_collapsed";
pub const TUI_VIEW_MODE: &str = "tui.view_mode";
pub const TUI_GALLERY_COLUMNS: &str = "tui.gallery_columns";
/// Sentinel marking that the legacy `tui-session.json` + `prompt-history.jsonl`
/// import has already run. Idempotent on subsequent launches.
pub const TUI_MIGRATED_FROM_JSON: &str = "tui.migrated_from_json";

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
/// issue #265 moved `tui.last_model` into the `settings` table.
pub fn record_last_model(model: &str) {
    let Some(db) = crate::global_db() else {
        return;
    };
    if let Err(e) = Settings::new(db).record_last_model(model) {
        tracing::warn!("settings.tui.last_model write failed: {e:#}");
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

    /// Record the last-used model for TUI/CLI resume-on-launch. Writes
    /// to the [`TUI_LAST_MODEL`] row — the single source of truth after
    /// issue #265 retired the `$MOLD_HOME/last-model` sidecar.
    pub fn record_last_model(&self, model: &str) -> Result<()> {
        self.set_str(TUI_LAST_MODEL, model)
    }

    fn upsert(&self, key: &str, value: &str, ty: ValueType) -> Result<()> {
        let ts = now_ms();
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO settings (profile, key, value, value_type, updated_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(profile, key) DO UPDATE SET
                    value = excluded.value,
                    value_type = excluded.value_type,
                    updated_at_ms = excluded.updated_at_ms",
                params![&self.profile, key, value, ty.as_str(), ts],
            )?;
            Ok(())
        })
    }
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
        s.set_str(TUI_THEME, "dracula").unwrap();
        assert_eq!(s.get_str(TUI_THEME).unwrap().as_deref(), Some("dracula"));
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
        s.set_bool(TUI_NEGATIVE_COLLAPSED, true).unwrap();
        assert_eq!(s.get_bool(TUI_NEGATIVE_COLLAPSED).unwrap(), Some(true));
        s.set_bool(TUI_NEGATIVE_COLLAPSED, false).unwrap();
        assert_eq!(s.get_bool(TUI_NEGATIVE_COLLAPSED).unwrap(), Some(false));
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
        s.set_str(TUI_THEME, "mocha").unwrap();
        s.set_str(TUI_THEME, "latte").unwrap();
        assert_eq!(s.get_str(TUI_THEME).unwrap().as_deref(), Some("latte"));
    }

    #[test]
    fn delete_returns_true_when_removed_false_otherwise() {
        let db = db();
        let s = Settings::new(&db);
        s.set_str(TUI_THEME, "nord").unwrap();
        assert!(s.delete(TUI_THEME).unwrap());
        assert!(!s.delete(TUI_THEME).unwrap());
        assert!(s.get_str(TUI_THEME).unwrap().is_none());
    }

    /// Item 3 (post-#265): `record_last_model` must land on the
    /// `TUI_LAST_MODEL` row so the TUI's resume-on-launch logic reads
    /// the DB-backed value, not the retired `last-model` sidecar.
    #[test]
    fn record_last_model_lands_on_tui_last_model_row() {
        let db = db();
        let s = Settings::new(&db);
        s.record_last_model("flux-dev:q4").unwrap();
        assert_eq!(
            s.get_str(TUI_LAST_MODEL).unwrap().as_deref(),
            Some("flux-dev:q4")
        );
        // Overwrite path mirrors the normal write flow.
        s.record_last_model("qwen-image:q6").unwrap();
        assert_eq!(
            s.get_str(TUI_LAST_MODEL).unwrap().as_deref(),
            Some("qwen-image:q6")
        );
    }

    /// Item 5: Settings scoped to different profiles must not see each
    /// other's rows. `default` and `dev` can both carry `tui.theme`
    /// without clobbering.
    #[test]
    fn settings_isolate_across_profiles() {
        let db = db();
        let default = Settings::for_profile(&db, DEFAULT_PROFILE);
        let dev = Settings::for_profile(&db, "dev");
        default.set_str(TUI_THEME, "mocha").unwrap();
        dev.set_str(TUI_THEME, "nord").unwrap();
        assert_eq!(
            default.get_str(TUI_THEME).unwrap().as_deref(),
            Some("mocha")
        );
        assert_eq!(dev.get_str(TUI_THEME).unwrap().as_deref(), Some("nord"));
        // Deleting from one profile does not affect the other.
        assert!(dev.delete(TUI_THEME).unwrap());
        assert_eq!(
            default.get_str(TUI_THEME).unwrap().as_deref(),
            Some("mocha")
        );
        assert!(dev.get_str(TUI_THEME).unwrap().is_none());
    }

    /// Item 5: `list_all` only returns rows for the current profile.
    #[test]
    fn list_all_is_scoped_to_profile() {
        let db = db();
        Settings::for_profile(&db, DEFAULT_PROFILE)
            .set_str(TUI_THEME, "mocha")
            .unwrap();
        Settings::for_profile(&db, "dev")
            .set_str(TUI_THEME, "nord")
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
        let _g = profile_env_lock().lock().unwrap_or_else(|e| e.into_inner());
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

    fn profile_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
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
