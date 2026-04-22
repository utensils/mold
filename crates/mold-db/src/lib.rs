//! SQLite-backed metadata store for `mold`.
//!
//! The database lives at `MOLD_HOME/mold.db` (override with `MOLD_DB_PATH`).
//! Both the CLI's local generation path and the server write a [`GenerationRecord`]
//! per saved file, and the server queries the DB to build `/api/gallery`.
//!
//! Set `MOLD_DB_DISABLE=1` to opt out entirely (CLI and server fall back to
//! the previous filesystem-only behavior).

pub mod config_sync;
mod db;
mod metadata_io;
mod migrations;
mod model_prefs;
mod path;
mod prompt_history;
mod reconcile;
mod record;
pub mod settings;

pub use db::MetadataDb;
pub use migrations::SCHEMA_VERSION;
pub use model_prefs::ModelPrefs;
pub use path::{canonical_dir, canonical_dir_string};
pub use prompt_history::{HistoryEntry, PromptHistory};
pub use reconcile::ReconcileStats;
pub use record::{GenerationRecord, RecordSource};
pub use settings::{Settings, ValueType};

use std::path::PathBuf;

/// Default file name for the metadata database inside `MOLD_HOME`.
pub const DEFAULT_DB_FILENAME: &str = "mold.db";

/// Resolve the metadata DB path:
/// 1. `MOLD_DB_PATH` env var if set and non-empty
/// 2. `<MOLD_HOME>/mold.db` from [`mold_core::Config::mold_dir`]
/// 3. `None` if `MOLD_HOME` cannot be resolved
pub fn default_db_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MOLD_DB_PATH") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    mold_core::Config::mold_dir().map(|d| d.join(DEFAULT_DB_FILENAME))
}

/// Whether the metadata DB is disabled via `MOLD_DB_DISABLE=1`.
///
/// Accepts `1`, `true`, `yes` (case-insensitive). When disabled, all callers
/// that use [`open_default`] receive `Ok(None)` and must skip persistence.
pub fn is_disabled() -> bool {
    std::env::var("MOLD_DB_DISABLE")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Open the default metadata DB, creating parent directories and applying
/// migrations as needed. Returns `Ok(None)` if disabled or the path cannot
/// be resolved. Use [`MetadataDb::open`] for an explicit path.
pub fn open_default() -> anyhow::Result<Option<MetadataDb>> {
    if is_disabled() {
        return Ok(None);
    }
    let Some(path) = default_db_path() else {
        return Ok(None);
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    Ok(Some(MetadataDb::open(&path)?))
}
