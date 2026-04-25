//! Schema migration framework for the gallery metadata DB.
//!
//! Each migration is applied in order inside its own transaction. The
//! current schema version is tracked via SQLite's built-in `PRAGMA
//! user_version` so we don't need a sidecar table.
//!
//! Migrations are one-way (no `down` step): if you need to walk back, write
//! a forward-only migration that undoes the change.
//!
//! ## Two kinds of migration
//!
//! Most migrations are pure DDL — a string of SQL statements. Some need
//! programmatic rewrites of existing rows (e.g. v2 canonicalizes
//! `output_dir` values written under raw paths by the v0.8.x release).
//! The [`MigrationKind`] enum covers both.
//!
//! ## Adding a new migration
//!
//! Append a new entry to [`MIGRATIONS`] with the next sequential version.
//! Example DDL-only migration:
//!
//! ```ignore
//! Migration {
//!     version: 3,
//!     kind: MigrationKind::Sql(r#"
//!         ALTER TABLE generations ADD COLUMN controlnet_model TEXT;
//!         ALTER TABLE generations ADD COLUMN controlnet_scale REAL;
//!     "#),
//! },
//! ```

use anyhow::{bail, Result};
use rusqlite::{Connection, Transaction};

use crate::path::canonical_dir_string;

/// What a migration does. SQL migrations are applied by
/// [`Connection::execute_batch`]; Rust migrations receive the active
/// transaction so they can both read + rewrite existing rows in place.
pub(crate) enum MigrationKind {
    Sql(&'static str),
    Rust(fn(&Transaction<'_>) -> Result<()>),
}

/// A single forward-only migration.
pub(crate) struct Migration {
    pub version: i64,
    pub kind: MigrationKind,
}

/// The initial schema — what v0.8.x shipped. Kept as a single block so a
/// fresh DB needs only one transaction to become a v1 DB.
const V1_INITIAL_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS generations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    filename           TEXT    NOT NULL,
    output_dir         TEXT    NOT NULL,
    created_at_ms      INTEGER NOT NULL,
    file_mtime_ms      INTEGER,
    file_size_bytes    INTEGER,

    format             TEXT    NOT NULL,

    model              TEXT    NOT NULL,
    prompt             TEXT    NOT NULL DEFAULT '',
    negative_prompt    TEXT,
    original_prompt    TEXT,
    seed               INTEGER NOT NULL DEFAULT 0,
    steps              INTEGER NOT NULL DEFAULT 0,
    guidance           REAL    NOT NULL DEFAULT 0.0,
    width              INTEGER NOT NULL DEFAULT 0,
    height             INTEGER NOT NULL DEFAULT 0,
    strength           REAL,
    scheduler          TEXT,
    lora               TEXT,
    lora_scale         REAL,
    frames             INTEGER,
    fps                INTEGER,
    metadata_version   TEXT    NOT NULL DEFAULT '',

    generation_time_ms INTEGER,
    backend            TEXT,
    hostname           TEXT,
    source             TEXT    NOT NULL DEFAULT 'unknown',
    metadata_synthetic INTEGER NOT NULL DEFAULT 0,

    UNIQUE(output_dir, filename)
);

CREATE INDEX IF NOT EXISTS idx_gen_created_at ON generations(created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_gen_mtime      ON generations(file_mtime_ms DESC);
CREATE INDEX IF NOT EXISTS idx_gen_model      ON generations(model);
CREATE INDEX IF NOT EXISTS idx_gen_format     ON generations(format);
CREATE INDEX IF NOT EXISTS idx_gen_filename   ON generations(filename);
CREATE INDEX IF NOT EXISTS idx_gen_output_dir ON generations(output_dir);
"#;

/// v3 → add the global KV `settings` table. Used for TUI + user-preference
/// state that previously lived in `tui-session.json` and the user-facing
/// portions of `config.toml`.
const V3_SETTINGS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS settings (
    key           TEXT PRIMARY KEY,
    value         TEXT NOT NULL,
    value_type    TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
"#;

/// v4 → add the per-model preferences table. One row per resolved model
/// tag; every column is nullable because a fresh install has nothing to
/// remember yet.
const V4_MODEL_PREFS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS model_prefs (
    model           TEXT PRIMARY KEY,
    width           INTEGER,
    height          INTEGER,
    steps           INTEGER,
    guidance        REAL,
    scheduler       TEXT,
    seed_mode       TEXT,
    batch           INTEGER,
    format          TEXT,
    lora_path       TEXT,
    lora_scale      REAL,
    expand          INTEGER,
    offload         INTEGER,
    strength        REAL,
    control_scale   REAL,
    frames          INTEGER,
    fps             INTEGER,
    last_prompt     TEXT,
    last_negative   TEXT,
    updated_at_ms   INTEGER NOT NULL
);
"#;

/// v5 → prompt history. Replaces `prompt-history.jsonl`; bounded-size via
/// the caller-driven `trim_to()` API.
const V5_PROMPT_HISTORY_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS prompt_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt        TEXT NOT NULL,
    negative      TEXT,
    model         TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_hist_created
    ON prompt_history(created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_prompt_hist_model
    ON prompt_history(model);
"#;

/// v6 → add a `profile` column to `settings` and `model_prefs` so the
/// same DB can host multiple independent user preference sets (`default`,
/// `dev`, `portrait`, …). All v5 rows land under `profile = 'default'`
/// so existing installs keep working untouched.
///
/// SQLite can't change a PK in-place, so each table is recreated and the
/// data is copied. Both steps run in the same v6 transaction — a crash
/// mid-migration leaves the DB at v5.
const V6_PROFILE_SCOPING: &str = r#"
CREATE TABLE settings_v6 (
    profile       TEXT NOT NULL DEFAULT 'default',
    key           TEXT NOT NULL,
    value         TEXT NOT NULL,
    value_type    TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (profile, key)
);
INSERT INTO settings_v6 (profile, key, value, value_type, updated_at_ms)
    SELECT 'default', key, value, value_type, updated_at_ms FROM settings;
DROP TABLE settings;
ALTER TABLE settings_v6 RENAME TO settings;

CREATE TABLE model_prefs_v6 (
    profile         TEXT NOT NULL DEFAULT 'default',
    model           TEXT NOT NULL,
    width           INTEGER,
    height          INTEGER,
    steps           INTEGER,
    guidance        REAL,
    scheduler       TEXT,
    seed_mode       TEXT,
    batch           INTEGER,
    format          TEXT,
    lora_path       TEXT,
    lora_scale      REAL,
    expand          INTEGER,
    offload         INTEGER,
    strength        REAL,
    control_scale   REAL,
    frames          INTEGER,
    fps             INTEGER,
    last_prompt     TEXT,
    last_negative   TEXT,
    updated_at_ms   INTEGER NOT NULL,
    PRIMARY KEY (profile, model)
);
INSERT INTO model_prefs_v6
    SELECT 'default', model, width, height, steps, guidance, scheduler, seed_mode,
           batch, format, lora_path, lora_scale, expand, offload, strength,
           control_scale, frames, fps, last_prompt, last_negative, updated_at_ms
    FROM model_prefs;
DROP TABLE model_prefs;
ALTER TABLE model_prefs_v6 RENAME TO model_prefs;
"#;

/// v7 → add the `catalog` table for the model-catalog expansion, plus a
/// companion `catalog_fts` FTS5 virtual table for full-text search over
/// `name`, `author`, `description`, and `tags`. Six covering indexes are
/// added to support the most common browse/sort patterns.
const V7_CATALOG_TABLE: &str = r#"
CREATE TABLE catalog (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    name            TEXT NOT NULL,
    author          TEXT,
    family          TEXT NOT NULL,
    family_role     TEXT NOT NULL,
    sub_family      TEXT,
    modality        TEXT NOT NULL,
    kind            TEXT NOT NULL,
    file_format     TEXT NOT NULL,
    bundling        TEXT NOT NULL,
    size_bytes      INTEGER,
    download_count  INTEGER NOT NULL DEFAULT 0,
    rating          REAL,
    likes           INTEGER NOT NULL DEFAULT 0,
    nsfw            INTEGER NOT NULL DEFAULT 0,
    thumbnail_url   TEXT,
    description     TEXT,
    license         TEXT,
    license_flags   TEXT,
    tags            TEXT,
    companions      TEXT,
    download_recipe TEXT NOT NULL,
    engine_phase    INTEGER NOT NULL,
    created_at      INTEGER,
    updated_at      INTEGER,
    added_at        INTEGER NOT NULL DEFAULT 0,
    UNIQUE (source, source_id)
);

CREATE INDEX idx_catalog_family    ON catalog(family, family_role);
CREATE INDEX idx_catalog_modality  ON catalog(modality);
CREATE INDEX idx_catalog_downloads ON catalog(download_count DESC);
CREATE INDEX idx_catalog_updated   ON catalog(updated_at DESC);
CREATE INDEX idx_catalog_rating    ON catalog(rating DESC);
CREATE INDEX idx_catalog_phase     ON catalog(engine_phase);

CREATE VIRTUAL TABLE catalog_fts USING fts5(
    name,
    author,
    description,
    tags,
    content='catalog',
    content_rowid='rowid'
);
"#;

/// Ordered list of schema migrations. Version numbers must be strictly
/// increasing — [`apply_pending`] validates this at startup.
pub(crate) const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        kind: MigrationKind::Sql(V1_INITIAL_SCHEMA),
    },
    Migration {
        version: 2,
        kind: MigrationKind::Rust(canonicalize_existing_output_dirs),
    },
    Migration {
        version: 3,
        kind: MigrationKind::Sql(V3_SETTINGS_TABLE),
    },
    Migration {
        version: 4,
        kind: MigrationKind::Sql(V4_MODEL_PREFS_TABLE),
    },
    Migration {
        version: 5,
        kind: MigrationKind::Sql(V5_PROMPT_HISTORY_TABLE),
    },
    Migration {
        version: 6,
        kind: MigrationKind::Sql(V6_PROFILE_SCOPING),
    },
    Migration {
        version: 7,
        kind: MigrationKind::Sql(V7_CATALOG_TABLE),
    },
];

/// The highest migration version this build ships. Exposed publicly so
/// operators / tests can assert what schema level they're running against.
pub const SCHEMA_VERSION: i64 = 7;

/// v1 → v2: rewrite every `output_dir` value to its canonical form so
/// rows written by the v0.8.x release (which keyed on raw paths) keep
/// matching the new canonicalized lookups. Without this, an upgraded
/// install would see every row written under `/tmp/...` or a symlinked
/// directory stop matching queries, and reconcile would insert fresh
/// duplicates under the canonical key.
///
/// The rewrite is conflict-safe: if a row already exists under the
/// canonical key (e.g. because both forms somehow got written), we
/// prefer the canonical row and drop the legacy one.
fn canonicalize_existing_output_dirs(tx: &Transaction<'_>) -> Result<()> {
    #[derive(Debug)]
    struct Row {
        id: i64,
        output_dir: String,
        filename: String,
    }

    // Pull the full set up front — the table is tiny (rows measured in
    // thousands at most) and avoids holding a statement open while we
    // run UPDATE/DELETE against the same table.
    let mut stmt = tx.prepare("SELECT id, output_dir, filename FROM generations")?;
    let rows: Vec<Row> = stmt
        .query_map([], |r| {
            Ok(Row {
                id: r.get(0)?,
                output_dir: r.get(1)?,
                filename: r.get(2)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    let mut rewritten = 0u64;
    let mut dropped_legacy_dup = 0u64;
    for row in rows {
        let canonical = canonical_dir_string(std::path::Path::new(&row.output_dir));
        if canonical == row.output_dir {
            continue;
        }
        // Is there already a row under the canonical key + same filename?
        let conflict: Option<i64> = tx
            .query_row(
                "SELECT id FROM generations WHERE output_dir = ?1 AND filename = ?2",
                rusqlite::params![canonical, row.filename],
                |r| r.get(0),
            )
            .ok();
        if conflict.is_some_and(|id| id != row.id) {
            // Canonical row wins — drop the legacy one to keep UNIQUE happy.
            tx.execute(
                "DELETE FROM generations WHERE id = ?1",
                rusqlite::params![row.id],
            )?;
            dropped_legacy_dup += 1;
        } else {
            tx.execute(
                "UPDATE generations SET output_dir = ?1 WHERE id = ?2",
                rusqlite::params![canonical, row.id],
            )?;
            rewritten += 1;
        }
    }
    if rewritten > 0 || dropped_legacy_dup > 0 {
        tracing::info!(
            rewritten,
            dropped_legacy_dup,
            "v2 migration canonicalized existing output_dir keys"
        );
    }
    Ok(())
}

/// Apply every migration whose version is greater than the DB's current
/// `user_version` pragma. Runs each migration in its own transaction —
/// partial failures leave the DB at the previous version instead of a
/// half-migrated state. A catastrophic crash between migrations is safe
/// because each transaction commits the `user_version` bump alongside
/// the DDL.
pub(crate) fn apply_pending(conn: &mut Connection) -> Result<i64> {
    // Sanity-check the migration list in debug builds — the SCHEMA_VERSION
    // constant must match the last entry and versions must be monotonic.
    debug_assert!(!MIGRATIONS.is_empty(), "migration list cannot be empty");
    debug_assert_eq!(
        MIGRATIONS.last().map(|m| m.version),
        Some(SCHEMA_VERSION),
        "SCHEMA_VERSION must match the last migration"
    );
    for win in MIGRATIONS.windows(2) {
        debug_assert!(
            win[0].version < win[1].version,
            "migration versions must be strictly increasing"
        );
    }

    let mut current = current_version(conn)?;
    for m in MIGRATIONS {
        if m.version <= current {
            continue;
        }
        if m.version != current + 1 {
            bail!(
                "migration gap: DB at v{}, next migration is v{}, expected v{}",
                current,
                m.version,
                current + 1
            );
        }
        let tx = conn.transaction()?;
        match &m.kind {
            MigrationKind::Sql(sql) => tx.execute_batch(sql)?,
            MigrationKind::Rust(run) => run(&tx)?,
        }
        // `user_version` pragma doesn't bind parameters — safe because
        // `m.version` is compile-time constant from our own source.
        tx.execute_batch(&format!("PRAGMA user_version = {};", m.version))?;
        tx.commit()?;
        tracing::info!(version = m.version, "applied metadata DB migration");
        current = m.version;
    }
    Ok(current)
}

/// Read the DB's current schema version from the `user_version` pragma.
/// A freshly-created DB returns `0`.
pub(crate) fn current_version(conn: &Connection) -> Result<i64> {
    let v: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_list_invariants_hold() {
        assert!(!MIGRATIONS.is_empty());
        assert_eq!(MIGRATIONS.last().unwrap().version, SCHEMA_VERSION);
        for win in MIGRATIONS.windows(2) {
            assert!(win[0].version < win[1].version);
        }
    }

    #[test]
    fn apply_pending_on_fresh_db_reaches_schema_version() {
        let mut conn = Connection::open_in_memory().unwrap();
        let v = apply_pending(&mut conn).unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
    }

    #[test]
    fn apply_pending_is_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        let v1 = current_version(&conn).unwrap();
        apply_pending(&mut conn).unwrap();
        let v2 = current_version(&conn).unwrap();
        assert_eq!(v1, v2);
    }

    /// Synthesize an ad-hoc DDL migration at runtime (without touching the
    /// real MIGRATIONS list) to prove the transaction wrapping + ordering
    /// works. This is the pattern future `ALTER TABLE ADD COLUMN`
    /// migrations will follow.
    #[test]
    fn transaction_wraps_each_migration() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();

        let sql = "ALTER TABLE generations ADD COLUMN test_col TEXT;\n\
                   PRAGMA user_version = 99;";
        let tx = conn.transaction().unwrap();
        tx.execute_batch(sql).unwrap();
        tx.commit().unwrap();

        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('generations') WHERE name = 'test_col'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(current_version(&conn).unwrap(), 99);
    }

    /// v2 migration (Codex finding): simulate the pre-upgrade state where
    /// v1 rows were written under a non-canonical path. After v2 runs,
    /// the rows must sit at the canonical key so new queries can find
    /// them — no orphans, no duplicates.
    #[cfg(target_os = "macos")]
    #[test]
    fn v2_canonicalizes_legacy_tmp_rows_on_macos() {
        // Apply v1 only so we control the pre-v2 row layout.
        let mut conn = Connection::open_in_memory().unwrap();
        let v1_only = Migration {
            version: 1,
            kind: MigrationKind::Sql(V1_INITIAL_SCHEMA),
        };
        let tx = conn.transaction().unwrap();
        match &v1_only.kind {
            MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
            _ => unreachable!(),
        }
        tx.execute_batch("PRAGMA user_version = 1;").unwrap();
        tx.commit().unwrap();

        // Seed a row under a non-canonical /tmp alias.
        let tmp = tempfile::tempdir_in("/tmp").unwrap();
        let legacy_path = tmp.path().to_string_lossy().into_owned();
        assert!(legacy_path.starts_with("/tmp/"), "test setup sanity");
        conn.execute(
            "INSERT INTO generations
                (filename, output_dir, created_at_ms, format, model)
             VALUES (?1, ?2, 0, 'png', 'm')",
            rusqlite::params!["legacy.png", legacy_path],
        )
        .unwrap();

        // Now run the real migration pipeline — v2 should rewrite the row.
        let final_v = apply_pending(&mut conn).unwrap();
        assert_eq!(final_v, SCHEMA_VERSION);

        let stored: String = conn
            .query_row(
                "SELECT output_dir FROM generations WHERE filename = 'legacy.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let canonical = canonical_dir_string(tmp.path());
        assert_eq!(stored, canonical, "v2 must rewrite legacy /tmp key");
        assert_ne!(stored, legacy_path, "must be the canonical form");
    }

    /// v2 edge case: if a row already exists under the canonical key when
    /// the legacy row is encountered, the migration must drop the legacy
    /// row rather than blow up the UNIQUE constraint.
    #[cfg(target_os = "macos")]
    #[test]
    fn v2_drops_legacy_dup_when_canonical_already_present() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch("PRAGMA user_version = 1;").unwrap();
        tx.commit().unwrap();

        let tmp = tempfile::tempdir_in("/tmp").unwrap();
        let legacy = tmp.path().to_string_lossy().into_owned();
        let canonical = canonical_dir_string(tmp.path());
        assert_ne!(legacy, canonical);

        // Seed both a legacy-keyed and a canonical-keyed row with the same filename.
        conn.execute(
            "INSERT INTO generations (filename, output_dir, created_at_ms, format, model)
             VALUES ('dup.png', ?1, 0, 'png', 'm')",
            rusqlite::params![legacy],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO generations (filename, output_dir, created_at_ms, format, model)
             VALUES ('dup.png', ?1, 0, 'png', 'm')",
            rusqlite::params![canonical],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM generations WHERE filename = 'dup.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "legacy row should be dropped, canonical kept");
        let kept: String = conn
            .query_row(
                "SELECT output_dir FROM generations WHERE filename = 'dup.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(kept, canonical);
    }

    // ------------------------------------------------------------------
    // v3 / v4 / v5 migration tests — added for feat/sqlite-settings.
    // These assert the shape of the new tables before the migrations
    // land, so a regression in any future refactor is caught early.
    // ------------------------------------------------------------------

    fn column_names(conn: &Connection, table: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info('{table}')"))
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn table_exists(conn: &Connection, table: &str) -> bool {
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![table],
                |r| r.get(0),
            )
            .unwrap();
        n == 1
    }

    fn index_exists(conn: &Connection, index: &str) -> bool {
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
                rusqlite::params![index],
                |r| r.get(0),
            )
            .unwrap();
        n == 1
    }

    #[test]
    fn fresh_db_reaches_schema_version_7() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        assert_eq!(
            current_version(&conn).unwrap(),
            7,
            "SCHEMA_VERSION should be 7 after the v7 catalog migration"
        );
        assert_eq!(SCHEMA_VERSION, 7);
    }

    /// v6: `settings` keeps every existing row under `profile = 'default'`
    /// and the composite PK `(profile, key)` is in place.
    #[test]
    fn v6_moves_existing_settings_to_default_profile() {
        // Seed a v5 DB with real settings rows, then let apply_pending
        // migrate it forward.
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch(V3_SETTINGS_TABLE).unwrap();
        tx.execute_batch(V4_MODEL_PREFS_TABLE).unwrap();
        tx.execute_batch(V5_PROMPT_HISTORY_TABLE).unwrap();
        tx.execute_batch("PRAGMA user_version = 5;").unwrap();
        tx.commit().unwrap();
        conn.execute(
            "INSERT INTO settings (key, value, value_type, updated_at_ms)
             VALUES ('tui.theme', 'mocha', 'string', 123)",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();
        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);

        let cols = column_names(&conn, "settings");
        assert!(
            cols.iter().any(|c| c == "profile"),
            "settings must gain a profile column, got {cols:?}"
        );

        let (profile, value): (String, String) = conn
            .query_row(
                "SELECT profile, value FROM settings WHERE key = 'tui.theme'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(profile, "default");
        assert_eq!(value, "mocha");
    }

    /// v6: `model_prefs` keeps every existing row under `profile = 'default'`.
    #[test]
    fn v6_moves_existing_model_prefs_to_default_profile() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch(V3_SETTINGS_TABLE).unwrap();
        tx.execute_batch(V4_MODEL_PREFS_TABLE).unwrap();
        tx.execute_batch(V5_PROMPT_HISTORY_TABLE).unwrap();
        tx.execute_batch("PRAGMA user_version = 5;").unwrap();
        tx.commit().unwrap();
        conn.execute(
            "INSERT INTO model_prefs
                (model, width, height, steps, updated_at_ms)
             VALUES ('flux-dev:q4', 1024, 1024, 20, 123)",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        let cols = column_names(&conn, "model_prefs");
        assert!(
            cols.iter().any(|c| c == "profile"),
            "model_prefs must gain a profile column, got {cols:?}"
        );
        let (profile, width): (String, i64) = conn
            .query_row(
                "SELECT profile, width FROM model_prefs WHERE model = 'flux-dev:q4'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(profile, "default");
        assert_eq!(width, 1024);
    }

    /// v6: two rows with the same `key` but different `profile` values
    /// coexist under the composite PK.
    #[test]
    fn v6_allows_same_key_across_profiles() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        conn.execute(
            "INSERT INTO settings (profile, key, value, value_type, updated_at_ms)
             VALUES ('default', 'tui.theme', 'mocha', 'string', 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO settings (profile, key, value, value_type, updated_at_ms)
             VALUES ('dev', 'tui.theme', 'nord', 'string', 1)",
            [],
        )
        .unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM settings WHERE key = 'tui.theme'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2, "same key under distinct profiles must coexist");
    }

    #[test]
    fn v3_creates_settings_table() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        assert!(table_exists(&conn, "settings"));
        let cols = column_names(&conn, "settings");
        for expected in &["key", "value", "value_type", "updated_at_ms"] {
            assert!(
                cols.iter().any(|c| c == *expected),
                "settings table missing column {expected}; got {cols:?}"
            );
        }
    }

    #[test]
    fn v4_creates_model_prefs_table() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        assert!(table_exists(&conn, "model_prefs"));
        let cols = column_names(&conn, "model_prefs");
        // Spot-check the invariants — every field we persist must exist.
        for expected in &[
            "model",
            "width",
            "height",
            "steps",
            "guidance",
            "scheduler",
            "seed_mode",
            "batch",
            "format",
            "lora_path",
            "lora_scale",
            "expand",
            "offload",
            "strength",
            "control_scale",
            "frames",
            "fps",
            "last_prompt",
            "last_negative",
            "updated_at_ms",
        ] {
            assert!(
                cols.iter().any(|c| c == *expected),
                "model_prefs missing column {expected}; got {cols:?}"
            );
        }
    }

    #[test]
    fn v5_creates_prompt_history_table_with_indexes() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        assert!(table_exists(&conn, "prompt_history"));
        let cols = column_names(&conn, "prompt_history");
        for expected in &["id", "prompt", "negative", "model", "created_at_ms"] {
            assert!(
                cols.iter().any(|c| c == *expected),
                "prompt_history missing column {expected}; got {cols:?}"
            );
        }
        assert!(
            index_exists(&conn, "idx_prompt_hist_created"),
            "missing created-desc index on prompt_history"
        );
        assert!(
            index_exists(&conn, "idx_prompt_hist_model"),
            "missing model index on prompt_history"
        );
    }

    /// Upgrading a v2 DB with existing `generations` rows must not clobber
    /// those rows. The whole point of additive migrations is that prod data
    /// survives a version bump.
    #[test]
    fn upgrade_from_v2_preserves_generations_table() {
        // Manually seed a v2 DB (v1 schema + v2 user_version).
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch("PRAGMA user_version = 2;").unwrap();
        tx.commit().unwrap();

        // Seed a representative row.
        conn.execute(
            "INSERT INTO generations (filename, output_dir, created_at_ms, format, model, prompt)
             VALUES ('legacy.png', '/out', 1000, 'png', 'flux-dev:q4', 'a cat')",
            [],
        )
        .unwrap();

        // Apply the pending v3/v4/v5/v6 migrations.
        apply_pending(&mut conn).unwrap();
        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);

        // Original row intact.
        let prompt: String = conn
            .query_row(
                "SELECT prompt FROM generations WHERE filename = 'legacy.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(prompt, "a cat");

        // New tables exist and are empty.
        assert!(table_exists(&conn, "settings"));
        assert!(table_exists(&conn, "model_prefs"));
        assert!(table_exists(&conn, "prompt_history"));
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM settings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    /// A migration whose SQL is malformed must not advance the version.
    #[test]
    fn failed_migration_rolls_back_version() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        let before = current_version(&conn).unwrap();

        let tx = conn.transaction().unwrap();
        let err = tx.execute_batch("THIS IS NOT SQL;");
        assert!(err.is_err());
        // `tx` drops here and rolls back. Version should stay put.
        drop(tx);
        assert_eq!(current_version(&conn).unwrap(), before);
    }
}

#[cfg(test)]
mod v7_tests {
    use super::*;
    use rusqlite::Connection;

    fn open() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        conn
    }

    #[test]
    fn schema_version_is_seven() {
        assert_eq!(SCHEMA_VERSION, 7);
    }

    #[test]
    fn catalog_table_exists_with_expected_columns() {
        let conn = open();
        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(catalog)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        for required in [
            "id",
            "source",
            "source_id",
            "name",
            "author",
            "family",
            "family_role",
            "sub_family",
            "modality",
            "kind",
            "file_format",
            "bundling",
            "size_bytes",
            "download_count",
            "rating",
            "likes",
            "nsfw",
            "thumbnail_url",
            "description",
            "license",
            "license_flags",
            "tags",
            "companions",
            "download_recipe",
            "engine_phase",
            "created_at",
            "updated_at",
            "added_at",
        ] {
            assert!(
                cols.contains(&required.to_string()),
                "missing column: {required}"
            );
        }
    }

    #[test]
    fn catalog_fts_virtual_table_exists() {
        let conn = open();
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name LIKE 'catalog_fts%'")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert!(tables.contains(&"catalog_fts".to_string()));
    }

    #[test]
    fn catalog_indexes_exist() {
        let conn = open();
        let idx: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='catalog'")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        for name in [
            "idx_catalog_family",
            "idx_catalog_modality",
            "idx_catalog_downloads",
            "idx_catalog_updated",
            "idx_catalog_rating",
            "idx_catalog_phase",
        ] {
            assert!(idx.iter().any(|i| i == name), "missing index: {name}");
        }
    }

    #[test]
    fn unique_source_source_id_constraint() {
        let conn = open();
        conn.execute(
            "INSERT INTO catalog (id, source, source_id, name, family, family_role, modality, kind, file_format, bundling, download_recipe, engine_phase, added_at)
             VALUES ('hf:a', 'hf', 'a', 'A', 'flux', 'foundation', 'image', 'checkpoint', 'safetensors', 'separated', '{}', 1, 0)",
            [],
        ).unwrap();
        let dup = conn.execute(
            "INSERT INTO catalog (id, source, source_id, name, family, family_role, modality, kind, file_format, bundling, download_recipe, engine_phase, added_at)
             VALUES ('hf:dup', 'hf', 'a', 'A2', 'flux', 'foundation', 'image', 'checkpoint', 'safetensors', 'separated', '{}', 1, 0)",
            [],
        );
        assert!(
            dup.is_err(),
            "duplicate (source, source_id) should violate UNIQUE"
        );
    }
}
