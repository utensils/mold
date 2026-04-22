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

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Typed view onto the `settings` table. Cheap to construct — it's just a
/// borrowed handle over the DB connection.
pub struct Settings<'a> {
    db: &'a MetadataDb,
}

impl<'a> Settings<'a> {
    pub fn new(db: &'a MetadataDb) -> Self {
        Self { db }
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
                    "SELECT value FROM settings WHERE key = ?1",
                    params![key],
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
            let n = conn.execute("DELETE FROM settings WHERE key = ?1", params![key])?;
            Ok(n > 0)
        })
    }

    /// Dump the full table — primarily for `mold config list` output.
    pub fn list_all(&self) -> Result<Vec<(String, String, String)>> {
        self.db.with_conn(|conn| {
            let mut stmt =
                conn.prepare("SELECT key, value, value_type FROM settings ORDER BY key")?;
            let mut rows = stmt.query([])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push((row.get(0)?, row.get(1)?, row.get(2)?));
            }
            Ok(out)
        })
    }

    fn upsert(&self, key: &str, value: &str, ty: ValueType) -> Result<()> {
        let ts = now_ms();
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO settings (key, value, value_type, updated_at_ms)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    value_type = excluded.value_type,
                    updated_at_ms = excluded.updated_at_ms",
                params![key, value, ty.as_str(), ts],
            )?;
            Ok(())
        })
    }
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
