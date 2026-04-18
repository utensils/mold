use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use mold_core::{OutputFormat, OutputMetadata, Scheduler};
use rusqlite::{params, Connection, OptionalExtension};

use crate::record::{GenerationRecord, RecordSource};

/// Current schema version. Bump and add a migration block in [`migrate`]
/// whenever the table layout changes.
const SCHEMA_VERSION: i64 = 1;

/// Stat snapshot returned by [`MetadataDb::snapshot_paths`] — one entry per
/// row, used by reconciliation to diff DB ↔ disk. Defined as a named struct
/// to keep callsites readable (and to satisfy clippy::type_complexity).
pub(crate) struct PathSnapshot {
    pub output_dir: String,
    pub filename: String,
    pub file_mtime_ms: Option<i64>,
    pub file_size_bytes: Option<i64>,
}

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

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

/// Thread-safe handle to the SQLite metadata DB.
///
/// The connection is wrapped in a `Mutex` because `rusqlite::Connection` is
/// `!Sync`. Operations are short and run inside `spawn_blocking` from async
/// callers, so contention is rare.
pub struct MetadataDb {
    conn: Mutex<Connection>,
    path: PathBuf,
}

impl std::fmt::Debug for MetadataDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetadataDb")
            .field("path", &self.path)
            .finish()
    }
}

impl MetadataDb {
    /// Open (or create) the SQLite database at `path` and apply migrations.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("opening metadata DB at {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        conn.pragma_update(None, "synchronous", "NORMAL").ok();
        conn.pragma_update(None, "foreign_keys", "ON").ok();
        let db = Self {
            conn: Mutex::new(conn),
            path: path.to_path_buf(),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Open an in-memory database — used by unit tests.
    #[doc(hidden)]
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self {
            conn: Mutex::new(conn),
            path: PathBuf::from(":memory:"),
        };
        db.migrate()?;
        Ok(db)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        conn.execute_batch(SCHEMA_V1)?;
        let current: Option<i64> = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |r| r.get(0))
            .optional()?
            .flatten();
        if current.unwrap_or(0) < SCHEMA_VERSION {
            conn.execute(
                "INSERT OR IGNORE INTO schema_version(version) VALUES (?1)",
                params![SCHEMA_VERSION],
            )?;
        }
        Ok(())
    }

    /// Insert or update a row keyed by `(output_dir, filename)`.
    /// Returns the row's primary-key id.
    pub fn upsert(&self, rec: &GenerationRecord) -> Result<i64> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        upsert_with_conn(&conn, rec)
    }

    /// Look up a row by its output directory + filename pair.
    pub fn get(&self, output_dir: &Path, filename: &str) -> Result<Option<GenerationRecord>> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, filename, output_dir, created_at_ms, file_mtime_ms, file_size_bytes,
                    format, model, prompt, negative_prompt, original_prompt, seed, steps,
                    guidance, width, height, strength, scheduler, lora, lora_scale, frames,
                    fps, metadata_version, generation_time_ms, backend, hostname, source,
                    metadata_synthetic
             FROM generations
             WHERE output_dir = ?1 AND filename = ?2",
        )?;
        let mut rows = stmt.query(params![output_dir.to_string_lossy().as_ref(), filename])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_record(row)?))
        } else {
            Ok(None)
        }
    }

    /// List rows for a specific `output_dir` (or all dirs when `None`),
    /// ordered newest-first by `file_mtime_ms` (falling back to `created_at_ms`).
    pub fn list(&self, output_dir: Option<&Path>) -> Result<Vec<GenerationRecord>> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let order_clause = "ORDER BY COALESCE(file_mtime_ms, created_at_ms) DESC";
        let select = "SELECT id, filename, output_dir, created_at_ms, file_mtime_ms, \
            file_size_bytes, format, model, prompt, negative_prompt, original_prompt, seed, \
            steps, guidance, width, height, strength, scheduler, lora, lora_scale, frames, \
            fps, metadata_version, generation_time_ms, backend, hostname, source, \
            metadata_synthetic FROM generations";
        let mut out = Vec::new();
        if let Some(dir) = output_dir {
            let mut stmt =
                conn.prepare(&format!("{select} WHERE output_dir = ?1 {order_clause}"))?;
            let mut rows = stmt.query(params![dir.to_string_lossy().as_ref()])?;
            while let Some(row) = rows.next()? {
                out.push(row_to_record(row)?);
            }
        } else {
            let mut stmt = conn.prepare(&format!("{select} {order_clause}"))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                out.push(row_to_record(row)?);
            }
        }
        Ok(out)
    }

    /// Remove a row by its `(output_dir, filename)` pair. Returns true if a
    /// row was deleted.
    pub fn delete(&self, output_dir: &Path, filename: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let n = conn.execute(
            "DELETE FROM generations WHERE output_dir = ?1 AND filename = ?2",
            params![output_dir.to_string_lossy().as_ref(), filename],
        )?;
        Ok(n > 0)
    }

    /// Total row count — used by tests and by the reconcile path's "kept" tally.
    pub fn count(&self) -> Result<i64> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM generations", [], |r| r.get(0))?;
        Ok(n)
    }

    /// Snapshot every `(output_dir, filename, file_mtime_ms, file_size_bytes)`
    /// pair so reconciliation can detect adds/removes/updates without holding
    /// the connection across long disk walks.
    pub(crate) fn snapshot_paths(&self) -> Result<Vec<PathSnapshot>> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT output_dir, filename, file_mtime_ms, file_size_bytes FROM generations",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PathSnapshot {
                output_dir: r.get::<_, String>(0)?,
                filename: r.get::<_, String>(1)?,
                file_mtime_ms: r.get::<_, Option<i64>>(2)?,
                file_size_bytes: r.get::<_, Option<i64>>(3)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Run `f` inside a single transaction. Useful for batched reconcile work.
    pub(crate) fn transact<R>(&self, f: impl FnOnce(&Connection) -> Result<R>) -> Result<R> {
        let mut conn = self.conn.lock().expect("metadata db mutex poisoned");
        let tx = conn.transaction()?;
        let r = f(&tx)?;
        tx.commit()?;
        Ok(r)
    }
}

/// Internal helper that takes an already-locked connection — lets callers
/// batch many upserts inside one transaction.
pub(crate) fn upsert_with_conn(conn: &Connection, rec: &GenerationRecord) -> Result<i64> {
    let scheduler_str = rec
        .metadata
        .scheduler
        .as_ref()
        .map(scheduler_to_str)
        .map(str::to_string);
    conn.execute(
        "INSERT INTO generations (
            filename, output_dir, created_at_ms, file_mtime_ms, file_size_bytes, format,
            model, prompt, negative_prompt, original_prompt, seed, steps, guidance,
            width, height, strength, scheduler, lora, lora_scale, frames, fps,
            metadata_version, generation_time_ms, backend, hostname, source, metadata_synthetic
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
            ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
         )
         ON CONFLICT(output_dir, filename) DO UPDATE SET
            created_at_ms = excluded.created_at_ms,
            file_mtime_ms = excluded.file_mtime_ms,
            file_size_bytes = excluded.file_size_bytes,
            format = excluded.format,
            model = excluded.model,
            prompt = excluded.prompt,
            negative_prompt = excluded.negative_prompt,
            original_prompt = excluded.original_prompt,
            seed = excluded.seed,
            steps = excluded.steps,
            guidance = excluded.guidance,
            width = excluded.width,
            height = excluded.height,
            strength = excluded.strength,
            scheduler = excluded.scheduler,
            lora = excluded.lora,
            lora_scale = excluded.lora_scale,
            frames = excluded.frames,
            fps = excluded.fps,
            metadata_version = excluded.metadata_version,
            generation_time_ms = excluded.generation_time_ms,
            backend = excluded.backend,
            hostname = excluded.hostname,
            source = excluded.source,
            metadata_synthetic = excluded.metadata_synthetic",
        params![
            rec.filename,
            rec.output_dir,
            rec.created_at_ms,
            rec.file_mtime_ms,
            rec.file_size_bytes,
            format_to_str(rec.format),
            rec.metadata.model,
            rec.metadata.prompt,
            rec.metadata.negative_prompt,
            rec.metadata.original_prompt,
            rec.metadata.seed as i64,
            rec.metadata.steps as i64,
            rec.metadata.guidance,
            rec.metadata.width as i64,
            rec.metadata.height as i64,
            rec.metadata.strength,
            scheduler_str,
            rec.metadata.lora,
            rec.metadata.lora_scale,
            rec.metadata.frames.map(|n| n as i64),
            rec.metadata.fps.map(|n| n as i64),
            rec.metadata.version,
            rec.generation_time_ms,
            rec.backend,
            rec.hostname,
            rec.source.as_str(),
            rec.metadata_synthetic as i64,
        ],
    )?;
    let id = conn.query_row(
        "SELECT id FROM generations WHERE output_dir = ?1 AND filename = ?2",
        params![rec.output_dir, rec.filename],
        |r| r.get::<_, i64>(0),
    )?;
    Ok(id)
}

pub(crate) fn delete_with_conn(
    conn: &Connection,
    output_dir: &str,
    filename: &str,
) -> Result<bool> {
    let n = conn.execute(
        "DELETE FROM generations WHERE output_dir = ?1 AND filename = ?2",
        params![output_dir, filename],
    )?;
    Ok(n > 0)
}

fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<GenerationRecord> {
    let format_s: String = row.get(6)?;
    let format = format_from_str(&format_s).unwrap_or(OutputFormat::Png);
    let scheduler_s: Option<String> = row.get(17)?;
    let scheduler = scheduler_s.as_deref().and_then(scheduler_from_str);
    let metadata = OutputMetadata {
        model: row.get(7)?,
        prompt: row.get(8)?,
        negative_prompt: row.get(9)?,
        original_prompt: row.get(10)?,
        seed: row.get::<_, i64>(11)? as u64,
        steps: row.get::<_, i64>(12)? as u32,
        guidance: row.get(13)?,
        width: row.get::<_, i64>(14)? as u32,
        height: row.get::<_, i64>(15)? as u32,
        strength: row.get(16)?,
        scheduler,
        lora: row.get(18)?,
        lora_scale: row.get(19)?,
        frames: row.get::<_, Option<i64>>(20)?.map(|n| n as u32),
        fps: row.get::<_, Option<i64>>(21)?.map(|n| n as u32),
        version: row.get(22)?,
    };
    let source_s: String = row.get(26)?;
    let synthetic_i: i64 = row.get(27)?;
    Ok(GenerationRecord {
        id: Some(row.get(0)?),
        filename: row.get(1)?,
        output_dir: row.get(2)?,
        created_at_ms: row.get(3)?,
        file_mtime_ms: row.get(4)?,
        file_size_bytes: row.get(5)?,
        format,
        metadata,
        generation_time_ms: row.get(23)?,
        backend: row.get(24)?,
        hostname: row.get(25)?,
        source: RecordSource::parse(&source_s),
        metadata_synthetic: synthetic_i != 0,
    })
}

fn format_to_str(f: OutputFormat) -> &'static str {
    match f {
        OutputFormat::Png => "png",
        OutputFormat::Jpeg => "jpeg",
        OutputFormat::Gif => "gif",
        OutputFormat::Apng => "apng",
        OutputFormat::Webp => "webp",
        OutputFormat::Mp4 => "mp4",
    }
}

fn format_from_str(s: &str) -> Option<OutputFormat> {
    Some(match s {
        "png" => OutputFormat::Png,
        "jpeg" | "jpg" => OutputFormat::Jpeg,
        "gif" => OutputFormat::Gif,
        "apng" => OutputFormat::Apng,
        "webp" => OutputFormat::Webp,
        "mp4" => OutputFormat::Mp4,
        _ => return None,
    })
}

fn scheduler_to_str(s: &Scheduler) -> &'static str {
    match s {
        Scheduler::Ddim => "ddim",
        Scheduler::EulerAncestral => "euler-ancestral",
        Scheduler::UniPc => "uni-pc",
    }
}

fn scheduler_from_str(s: &str) -> Option<Scheduler> {
    Some(match s {
        "ddim" => Scheduler::Ddim,
        "euler-ancestral" => Scheduler::EulerAncestral,
        "uni-pc" | "unipc" => Scheduler::UniPc,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::GenerationRecord;
    use mold_core::OutputMetadata;
    use std::path::Path;

    fn meta() -> OutputMetadata {
        OutputMetadata {
            prompt: "a cat".into(),
            negative_prompt: Some("blurry".into()),
            original_prompt: None,
            model: "flux-dev:q4".into(),
            seed: 42,
            steps: 20,
            guidance: 4.0,
            width: 1024,
            height: 1024,
            strength: Some(0.8),
            scheduler: Some(Scheduler::Ddim),
            lora: Some("style.safetensors".into()),
            lora_scale: Some(1.0),
            frames: None,
            fps: None,
            version: "0.8.1".into(),
        }
    }

    fn rec() -> GenerationRecord {
        GenerationRecord {
            id: None,
            filename: "mold-flux-dev-q4-1.png".into(),
            output_dir: "/tmp/out".into(),
            created_at_ms: 1_000,
            file_mtime_ms: Some(2_000),
            file_size_bytes: Some(123_456),
            format: OutputFormat::Png,
            metadata: meta(),
            generation_time_ms: Some(3_500),
            backend: Some("cuda".into()),
            hostname: Some("hal9000".into()),
            source: RecordSource::Server,
            metadata_synthetic: false,
        }
    }

    #[test]
    fn open_in_memory_then_upsert_round_trips() {
        let db = MetadataDb::open_in_memory().unwrap();
        let id = db.upsert(&rec()).unwrap();
        assert!(id > 0);
        let got = db
            .get(Path::new("/tmp/out"), "mold-flux-dev-q4-1.png")
            .unwrap()
            .expect("row should exist");
        assert_eq!(got.metadata.prompt, "a cat");
        assert_eq!(got.metadata.seed, 42);
        assert_eq!(got.format, OutputFormat::Png);
        assert_eq!(got.source, RecordSource::Server);
    }

    #[test]
    fn upsert_replaces_existing_row() {
        let db = MetadataDb::open_in_memory().unwrap();
        let id1 = db.upsert(&rec()).unwrap();
        let mut updated = rec();
        updated.metadata.prompt = "a different cat".into();
        let id2 = db.upsert(&updated).unwrap();
        assert_eq!(id1, id2, "upsert should keep the same primary key");
        let got = db
            .get(Path::new("/tmp/out"), "mold-flux-dev-q4-1.png")
            .unwrap()
            .unwrap();
        assert_eq!(got.metadata.prompt, "a different cat");
    }

    #[test]
    fn delete_removes_row_and_returns_true() {
        let db = MetadataDb::open_in_memory().unwrap();
        db.upsert(&rec()).unwrap();
        assert!(db
            .delete(Path::new("/tmp/out"), "mold-flux-dev-q4-1.png")
            .unwrap());
        assert!(!db
            .delete(Path::new("/tmp/out"), "mold-flux-dev-q4-1.png")
            .unwrap());
        assert_eq!(db.count().unwrap(), 0);
    }

    #[test]
    fn list_orders_newest_first() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut a = rec();
        a.filename = "a.png".into();
        a.file_mtime_ms = Some(100);
        let mut b = rec();
        b.filename = "b.png".into();
        b.file_mtime_ms = Some(500);
        let mut c = rec();
        c.filename = "c.png".into();
        c.file_mtime_ms = Some(300);
        for r in [&a, &b, &c] {
            db.upsert(r).unwrap();
        }
        let listed = db.list(Some(Path::new("/tmp/out"))).unwrap();
        let names: Vec<_> = listed.iter().map(|r| r.filename.as_str()).collect();
        assert_eq!(names, vec!["b.png", "c.png", "a.png"]);
    }

    #[test]
    fn list_filters_by_output_dir() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut a = rec();
        a.filename = "a.png".into();
        a.output_dir = "/dir/a".into();
        let mut b = rec();
        b.filename = "b.png".into();
        b.output_dir = "/dir/b".into();
        db.upsert(&a).unwrap();
        db.upsert(&b).unwrap();
        let only_a = db.list(Some(Path::new("/dir/a"))).unwrap();
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].filename, "a.png");
        let all = db.list(None).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn round_trips_format_and_scheduler() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut r = rec();
        r.format = OutputFormat::Mp4;
        r.metadata.scheduler = Some(Scheduler::EulerAncestral);
        db.upsert(&r).unwrap();
        let got = db
            .get(Path::new("/tmp/out"), "mold-flux-dev-q4-1.png")
            .unwrap()
            .unwrap();
        assert_eq!(got.format, OutputFormat::Mp4);
        assert_eq!(got.metadata.scheduler, Some(Scheduler::EulerAncestral));
    }

    #[test]
    fn migrate_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let _db1 = MetadataDb::open(&path).unwrap();
        let db2 = MetadataDb::open(&path).unwrap();
        db2.upsert(&rec()).unwrap();
        assert_eq!(db2.count().unwrap(), 1);
    }
}
