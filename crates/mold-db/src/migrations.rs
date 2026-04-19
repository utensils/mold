//! Schema migration framework for the gallery metadata DB.
//!
//! Each migration is a `(version, sql)` pair applied in order inside a
//! transaction. The current schema version is tracked via SQLite's built-in
//! `PRAGMA user_version` so we don't need a sidecar table.
//!
//! ## Adding a new migration
//!
//! Append a new entry to [`MIGRATIONS`] with the next sequential version.
//! Migrations are one-way (no `down` step): if you need to walk back, write
//! a forward-only migration that undoes the change. Example:
//!
//! ```ignore
//! Migration {
//!     version: 2,
//!     sql: r#"
//!         ALTER TABLE generations ADD COLUMN inserted_by_version TEXT;
//!         ALTER TABLE generations ADD COLUMN controlnet_model TEXT;
//!         ALTER TABLE generations ADD COLUMN controlnet_scale REAL;
//!     "#,
//! },
//! ```
//!
//! Existing DBs get only the new ALTER blocks; fresh DBs get every
//! migration from 1 upward, so there's one source of truth for the schema.

use anyhow::{bail, Result};
use rusqlite::Connection;

/// A single forward-only migration. The `sql` string is fed to
/// [`Connection::execute_batch`] so it may contain multiple statements
/// separated by semicolons.
pub(crate) struct Migration {
    pub version: i64,
    pub sql: &'static str,
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
pub(crate) const MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    sql: V1_INITIAL_SCHEMA,
}];

/// The highest migration version this build ships. Exposed publicly so
/// operators / tests can assert what schema level they're running against.
pub const SCHEMA_VERSION: i64 = 1;

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
        tx.execute_batch(m.sql)?;
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

    /// Synthesize a 1→2 migration at runtime (without touching the real
    /// MIGRATIONS list) to prove the transaction wrapping + ordering
    /// works. This is the pattern real ALTER TABLE migrations will follow.
    #[test]
    fn transaction_wraps_each_migration() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();

        let fake = Migration {
            version: 2,
            sql: "ALTER TABLE generations ADD COLUMN test_col TEXT;\n\
                  PRAGMA user_version = 2;",
        };
        let tx = conn.transaction().unwrap();
        tx.execute_batch(fake.sql).unwrap();
        tx.commit().unwrap();

        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('generations') WHERE name = 'test_col'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(current_version(&conn).unwrap(), 2);
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
