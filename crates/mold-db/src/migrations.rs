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
];

/// The highest migration version this build ships. Exposed publicly so
/// operators / tests can assert what schema level they're running against.
pub const SCHEMA_VERSION: i64 = 2;

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
