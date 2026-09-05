use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use mold_core::{OutputFormat, OutputMetadata, Scheduler};
use rusqlite::backup::{Backup, StepResult};
use rusqlite::{params, Connection, ErrorCode, OptionalExtension};

use crate::migrations;
use crate::path::canonical_dir_string;
use crate::record::{GenerationRecord, RecordSource};

/// How long one metadata operation waits for SQLite lock contention.
///
/// Durable-queue retry scheduling reuses this exact window: retrying a failed
/// persistence pass more frequently than the connection itself waits would
/// amplify pressure on the same database without improving recovery latency.
pub const METADATA_DB_BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(5000);

/// Stat snapshot returned by [`MetadataDb::snapshot_paths`] — one entry per
/// row, used by reconciliation to diff DB ↔ disk. Defined as a named struct
/// to keep callsites readable (and to satisfy clippy::type_complexity).
pub(crate) struct PathSnapshot {
    pub output_dir: String,
    pub filename: String,
    pub file_mtime_ms: Option<i64>,
    pub file_size_bytes: Option<i64>,
    pub trashed_at_ms: Option<i64>,
}

/// Which rows a gallery listing should return with respect to the trash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrashFilter {
    All,
    LiveOnly,
    TrashedOnly,
}

/// The column projection [`row_to_record`] expects, in order. Every
/// `SELECT` that feeds `row_to_record` must use exactly this list.
pub(crate) const GENERATION_SELECT_COLUMNS: &str = "SELECT id, filename, output_dir, \
    created_at_ms, file_mtime_ms, file_size_bytes, format, model, prompt, negative_prompt, \
    original_prompt, seed, steps, guidance, width, height, strength, scheduler, lora, \
    lora_scale, frames, fps, metadata_version, generation_time_ms, backend, hostname, \
    source, metadata_synthetic, metadata_json, title, favorite, trashed_at_ms";

/// Configure connection-level pragmas at open time. Pragmas that fail to
/// apply are logged at warn level — concurrent-writer performance degrades
/// but correctness is preserved (SQLite falls back to rollback journal).
/// Verifies `journal_mode=WAL` actually took by reading it back.
fn configure_pragmas(conn: &Connection, path: &Path) {
    if let Err(e) = conn.pragma_update(None, "journal_mode", "WAL") {
        tracing::warn!(
            error = %e,
            path = %path.display(),
            "metadata DB journal_mode=WAL pragma failed — falling back to rollback journal"
        );
    }
    // Read the mode back. Some filesystems (tmpfs, certain network mounts)
    // silently accept the pragma but leave the mode unchanged.
    if let Ok(mode) = query_pragma_string(conn, "journal_mode") {
        if !mode.eq_ignore_ascii_case("wal") {
            tracing::warn!(
                mode = %mode,
                path = %path.display(),
                "metadata DB is not in WAL mode — concurrent writers will block longer"
            );
        }
    }
    if let Err(e) = conn.pragma_update(None, "synchronous", "NORMAL") {
        tracing::warn!(error = %e, "metadata DB synchronous pragma failed");
    }
    // Wait out short lock contention instead of surfacing SQLITE_BUSY.
    // Multiple processes share this file (TUI + `mold serve` + CLI), and
    // most read paths treat an error as "no value" — without a timeout a
    // reader that lands on a checkpoint or schema lock silently loses
    // settings it would have found a few milliseconds later.
    if let Err(e) = conn.busy_timeout(METADATA_DB_BUSY_TIMEOUT) {
        tracing::warn!(error = %e, "metadata DB busy_timeout failed");
    }
    if let Err(e) = conn.pragma_update(None, "foreign_keys", "ON") {
        tracing::warn!(error = %e, "metadata DB foreign_keys pragma failed");
    }
}

/// Read a PRAGMA whose value is a single TEXT cell.
fn query_pragma_string(conn: &Connection, name: &str) -> Result<String> {
    // Pragma name is compile-time constant from our own code; never
    // user-controlled. Safe to inline.
    let v: String = conn.query_row(&format!("PRAGMA {name}"), [], |r| r.get(0))?;
    Ok(v)
}

fn verify_integrity(conn: &Connection) -> Result<()> {
    let mut check = conn.prepare("PRAGMA quick_check(1)")?;
    let messages = check.query_map([], |row| row.get::<_, String>(0))?;
    for message in messages {
        let message = message?;
        if !message.eq_ignore_ascii_case("ok") {
            bail!("metadata DB quick_check reported corruption: {message}");
        }
    }
    Ok(())
}

fn open_connection(path: &Path) -> Result<Connection> {
    let mut conn = Connection::open(path)
        .with_context(|| format!("opening metadata DB at {}", path.display()))?;
    configure_pragmas(&conn, path);
    migrations::apply_pending(&mut conn)
        .with_context(|| format!("applying migrations to metadata DB at {}", path.display()))?;

    // Opening SQLite itself does not necessarily touch every B-tree. A
    // corrupt gallery index can otherwise survive startup and fail only on
    // the first ordered query. quick_check is intentionally paid once per
    // open so corruption is discovered before the handle is published.
    verify_integrity(&conn)?;
    Ok(conn)
}

fn has_sqlite_error_code(error: &anyhow::Error, expected: ErrorCode) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<rusqlite::Error>()
            .is_some_and(|error| {
                matches!(error, rusqlite::Error::SqliteFailure(code, _) if code.code == expected)
            })
    })
}

fn is_corruption_error(error: &anyhow::Error) -> bool {
    has_sqlite_error_code(error, ErrorCode::DatabaseCorrupt)
        || has_sqlite_error_code(error, ErrorCode::NotADatabase)
        || error.chain().any(|cause| {
            cause
                .to_string()
                .starts_with("metadata DB quick_check reported corruption:")
        })
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let filename = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    path.with_file_name(format!("{filename}{suffix}"))
}

fn acquire_recovery_lock(path: &Path) -> Result<std::fs::File> {
    let lock_path = sidecar_path(path, ".recovery.lock");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("opening metadata DB recovery lock {}", lock_path.display()))?;
    fs2::FileExt::lock_exclusive(&lock)
        .with_context(|| format!("locking metadata DB recovery lock {}", lock_path.display()))?;
    Ok(lock)
}

fn quarantine_corrupt_files(path: &Path) -> Result<PathBuf> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let filename = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "mold.db".into());
    let mut attempt = 0u32;
    let quarantine = loop {
        let suffix = if attempt == 0 {
            String::new()
        } else {
            format!("-{attempt}")
        };
        let candidate = path.with_file_name(format!("{filename}.corrupt-{timestamp}{suffix}"));
        if !candidate.exists() {
            break candidate;
        }
        attempt += 1;
    };

    // Copy rather than rename: replacing a WAL database's inode while another
    // Mold process has it open is unsafe. The online backup below rewrites the
    // live database through SQLite's locking protocol; these copies only retain
    // the pre-recovery bytes for operator inspection or salvage.
    std::fs::copy(path, &quarantine).with_context(|| {
        format!(
            "copying corrupt metadata DB {} to {}",
            path.display(),
            quarantine.display()
        )
    })?;
    for suffix in ["-wal", "-shm"] {
        let source = sidecar_path(path, suffix);
        if source.exists() {
            let target = sidecar_path(&quarantine, suffix);
            match std::fs::copy(&source, &target) {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "copying corrupt metadata DB sidecar {} to {}",
                            source.display(),
                            target.display()
                        )
                    });
                }
            }
        }
    }
    Ok(quarantine)
}

fn rebuild_schema_in_place(conn: &mut Connection, path: &Path) -> Result<()> {
    let mut source = Connection::open_in_memory()
        .context("creating fresh schema source for metadata DB recovery")?;
    migrations::apply_pending(&mut source)
        .context("applying migrations to fresh metadata DB recovery source")?;

    // SQLite's online backup API writes the replacement through the existing
    // destination connection, so its normal locks coordinate with every other
    // process that already has mold.db open. An incomplete backup rolls back.
    let backup = Backup::new(&source, conn).context("starting metadata DB schema rebuild")?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match backup
            .step(100)
            .context("writing rebuilt metadata DB schema")?
        {
            StepResult::Done => break,
            StepResult::More => {}
            StepResult::Busy | StepResult::Locked => {
                if std::time::Instant::now() >= deadline {
                    bail!("metadata DB remained busy for 5 seconds during schema rebuild");
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            _ => unreachable!("rusqlite added an unknown backup step result"),
        }
    }
    drop(backup);
    configure_pragmas(conn, path);
    verify_integrity(conn).context("checking rebuilt metadata DB")?;
    Ok(())
}

/// Thread-safe handle to the SQLite metadata DB.
///
/// The connection is wrapped in a `Mutex` because `rusqlite::Connection` is
/// `!Sync`. Operations are short and run inside `spawn_blocking` from async
/// callers, so contention is rare.
pub struct MetadataDb {
    conn: Mutex<Connection>,
    path: PathBuf,
    recovery_epoch: AtomicU64,
}

impl std::fmt::Debug for MetadataDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetadataDb")
            .field("path", &self.path)
            .finish()
    }
}

impl MetadataDb {
    /// Open (or create) the SQLite database at `path`, enable WAL journal
    /// mode, and apply pending schema migrations.
    ///
    /// WAL mode is verified after the pragma: if it didn't actually take
    /// (e.g. on a filesystem that rejects WAL, like certain network mounts)
    /// a warning is logged so operators know concurrent writers will be
    /// slower. The open still succeeds — functionality is preserved, just
    /// with the default rollback journal.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = match open_connection(path) {
            Ok(conn) => conn,
            Err(error) if path.exists() && is_corruption_error(&error) => {
                // Serialize recovery across the server, CLI, TUI, and Discord
                // processes. Recheck after taking the lock: another process
                // may already have replaced the corrupt inode while we waited.
                let _recovery_lock = acquire_recovery_lock(path)?;
                match open_connection(path) {
                    Ok(conn) => conn,
                    Err(rechecked) if path.exists() && is_corruption_error(&rechecked) => {
                        let quarantine = quarantine_corrupt_files(path)?;
                        tracing::error!(
                            error = %error,
                            db = %path.display(),
                            quarantine = %quarantine.display(),
                            "metadata DB corruption detected at open — quarantined database and rebuilding schema"
                        );
                        if has_sqlite_error_code(&rechecked, ErrorCode::NotADatabase) {
                            // The backup API cannot open an arbitrary non-SQLite
                            // destination. No usable SQLite handle can exist for
                            // this file, so truncating it after retaining the raw
                            // quarantine copy is safe.
                            std::fs::OpenOptions::new()
                                .write(true)
                                .truncate(true)
                                .open(path)
                                .with_context(|| {
                                    format!("resetting non-database file {}", path.display())
                                })?;
                            for suffix in ["-wal", "-shm"] {
                                match std::fs::remove_file(sidecar_path(path, suffix)) {
                                    Ok(()) => {}
                                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                                    Err(error) => return Err(error.into()),
                                }
                            }
                            open_connection(path)?
                        } else {
                            let mut rebuilt = Connection::open(path).with_context(|| {
                                format!(
                                    "opening {} for in-place metadata DB rebuild",
                                    path.display()
                                )
                            })?;
                            rebuilt
                                .busy_timeout(std::time::Duration::from_secs(5))
                                .context("setting metadata DB recovery busy timeout")?;
                            rebuild_schema_in_place(&mut rebuilt, path).with_context(|| {
                                format!(
                                    "rebuilding metadata DB after quarantining {}",
                                    quarantine.display()
                                )
                            })?;
                            rebuilt
                        }
                    }
                    Err(rechecked) => return Err(rechecked),
                }
            }
            Err(error) => return Err(error),
        };
        Ok(Self {
            conn: Mutex::new(conn),
            path: path.to_path_buf(),
            recovery_epoch: AtomicU64::new(0),
        })
    }

    /// Open an in-memory database — used by unit tests. WAL mode is
    /// silently skipped for `:memory:` since SQLite only supports it on
    /// file-backed databases.
    #[doc(hidden)]
    pub fn open_in_memory() -> Result<Self> {
        let mut conn = Connection::open_in_memory()?;
        // `synchronous = NORMAL` is safe for in-memory; journal_mode stays
        // at `memory` which is correct for the `:memory:` backend.
        let _ = conn.pragma_update(None, "synchronous", "NORMAL");
        let _ = conn.pragma_update(None, "foreign_keys", "ON");
        migrations::apply_pending(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            path: PathBuf::from(":memory:"),
            recovery_epoch: AtomicU64::new(0),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Current applied schema version (see [`migrations::SCHEMA_VERSION`]).
    pub fn schema_version(&self) -> Result<i64> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        migrations::current_version(&conn)
    }

    /// Insert or update a row keyed by `(output_dir, filename)`.
    /// Returns the row's primary-key id.
    pub fn upsert(&self, rec: &GenerationRecord) -> Result<i64> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        upsert_with_conn(&conn, rec)
    }

    /// [`Self::upsert`], additionally reporting the creation-time filing
    /// that was seeded onto a freshly inserted row.
    ///
    /// The seeding happens on every `upsert` — it is part of what an insert
    /// means, so no publication path can forget it. This variant exists only
    /// so the server can emit `gallery_updated` (and
    /// `gallery_collections_changed`, when the seed had to create the
    /// collection) without re-reading the row it just wrote.
    pub fn upsert_reporting_organization(
        &self,
        rec: &GenerationRecord,
    ) -> Result<(i64, crate::organization::SeededOrganization)> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        upsert_with_conn_reporting_organization(&conn, rec)
    }

    /// Insert or update a set of gallery rows in one SQLite transaction.
    ///
    /// Batch publication uses this after every final file has been linked
    /// into place while the gallery publication writer gate is held. Either
    /// every child becomes visible to DB-backed gallery listings or none do.
    pub fn upsert_batch(&self, records: &[GenerationRecord]) -> Result<Vec<i64>> {
        self.transact(|conn| {
            records
                .iter()
                .map(|record| upsert_with_conn(conn, record))
                .collect()
        })
    }

    /// Look up a row by its output directory + filename pair.
    pub fn get(&self, output_dir: &Path, filename: &str) -> Result<Option<GenerationRecord>> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, filename, output_dir, created_at_ms, file_mtime_ms, file_size_bytes,
                    format, model, prompt, negative_prompt, original_prompt, seed, steps,
                    guidance, width, height, strength, scheduler, lora, lora_scale, frames,
                    fps, metadata_version, generation_time_ms, backend, hostname, source,
                    metadata_synthetic, metadata_json, title, favorite, trashed_at_ms
             FROM generations
             WHERE output_dir = ?1 AND filename = ?2",
        )?;
        // Normalize output_dir so `/tmp/foo` and `/private/tmp/foo`
        // (macOS) or a symlinked path all resolve to the same stored key.
        let dir_key = canonical_dir_string(output_dir);
        let mut rows = stmt.query(params![dir_key, filename])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_record(row)?))
        } else {
            Ok(None)
        }
    }

    /// List rows for a specific `output_dir` (or all dirs when `None`),
    /// ordered newest-first by `file_mtime_ms` (falling back to `created_at_ms`).
    ///
    /// Returns EVERY row, trashed ones included — callers that only want
    /// the live library filter on [`GenerationRecord::trashed_at_ms`] or use
    /// [`Self::list_live`] / [`Self::list_trashed`]. Keeping the full view
    /// here means the reconcile and TUI paths that predate the trash keep
    /// observing the whole table.
    pub fn list(&self, output_dir: Option<&Path>) -> Result<Vec<GenerationRecord>> {
        self.list_filtered(output_dir, TrashFilter::All)
    }

    /// Like [`Self::list`] but only rows that are NOT in the trash
    /// (`trashed_at_ms IS NULL`).
    pub fn list_live(&self, output_dir: Option<&Path>) -> Result<Vec<GenerationRecord>> {
        self.list_filtered(output_dir, TrashFilter::LiveOnly)
    }

    /// Like [`Self::list`] but only rows that ARE in the trash
    /// (`trashed_at_ms IS NOT NULL`).
    pub fn list_trashed(&self, output_dir: Option<&Path>) -> Result<Vec<GenerationRecord>> {
        self.list_filtered(output_dir, TrashFilter::TrashedOnly)
    }

    fn list_filtered(
        &self,
        output_dir: Option<&Path>,
        filter: TrashFilter,
    ) -> Result<Vec<GenerationRecord>> {
        let epoch = self.recovery_epoch.load(Ordering::Acquire);
        match self.list_once(output_dir, filter) {
            Ok(rows) => Ok(rows),
            Err(error) if is_corruption_error(&error) => {
                self.rebuild_after_corruption(epoch, &error)?;
                if let Some(dir) = output_dir {
                    let stats = self.reconcile(dir).with_context(|| {
                        format!(
                            "reconciling gallery after rebuilding corrupt metadata DB at {}",
                            self.path.display()
                        )
                    })?;
                    tracing::error!(
                        db = %self.path.display(),
                        imported = stats.imported,
                        updated = stats.updated,
                        removed = stats.removed,
                        kept = stats.kept,
                        "metadata DB rebuilt from gallery after query-time corruption"
                    );
                }
                self.list_once(output_dir, filter)
                    .context("retrying gallery query after metadata DB rebuild")
            }
            Err(error) => Err(error),
        }
    }

    fn list_once(
        &self,
        output_dir: Option<&Path>,
        filter: TrashFilter,
    ) -> Result<Vec<GenerationRecord>> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let order_clause = "ORDER BY COALESCE(file_mtime_ms, created_at_ms) DESC";
        let select = format!("{GENERATION_SELECT_COLUMNS} FROM generations");
        let trash_clause = match filter {
            TrashFilter::All => "",
            TrashFilter::LiveOnly => "trashed_at_ms IS NULL",
            TrashFilter::TrashedOnly => "trashed_at_ms IS NOT NULL",
        };
        let mut out = Vec::new();
        if let Some(dir) = output_dir {
            let dir_key = canonical_dir_string(dir);
            let sql = if trash_clause.is_empty() {
                format!("{select} WHERE output_dir = ?1 {order_clause}")
            } else {
                format!("{select} WHERE output_dir = ?1 AND {trash_clause} {order_clause}")
            };
            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query(params![dir_key])?;
            while let Some(row) = rows.next()? {
                out.push(row_to_record(row)?);
            }
        } else {
            let sql = if trash_clause.is_empty() {
                format!("{select} {order_clause}")
            } else {
                format!("{select} WHERE {trash_clause} {order_clause}")
            };
            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                out.push(row_to_record(row)?);
            }
        }
        Ok(out)
    }

    fn rebuild_after_corruption(&self, observed_epoch: u64, error: &anyhow::Error) -> Result<()> {
        let mut conn = self.conn.lock().expect("metadata db mutex poisoned");
        if self.recovery_epoch.load(Ordering::Acquire) != observed_epoch {
            return Ok(());
        }
        let _recovery_lock = acquire_recovery_lock(&self.path)?;
        // A different process may have completed recovery while this process
        // waited for the lock. Never overwrite that healthy replacement.
        match open_connection(&self.path) {
            Ok(reopened) => {
                *conn = reopened;
                self.recovery_epoch.fetch_add(1, Ordering::Release);
                tracing::warn!(
                    db = %self.path.display(),
                    "metadata DB reopened after concurrent corruption recovery"
                );
                return Ok(());
            }
            Err(rechecked) if is_corruption_error(&rechecked) => {}
            Err(rechecked) => {
                return Err(rechecked)
                    .context("reopening metadata DB before query-time corruption recovery");
            }
        }

        let quarantine = quarantine_corrupt_files(&self.path)?;
        tracing::error!(
            error = %error,
            db = %self.path.display(),
            quarantine = %quarantine.display(),
            "metadata DB corruption detected during query — quarantined database and rebuilding schema"
        );
        rebuild_schema_in_place(&mut conn, &self.path).with_context(|| {
            format!(
                "rebuilding metadata DB after quarantining {}",
                quarantine.display()
            )
        })?;
        self.recovery_epoch.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Remove a row by its `(output_dir, filename)` pair. Returns true if a
    /// row was deleted.
    pub fn delete(&self, output_dir: &Path, filename: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let dir_key = canonical_dir_string(output_dir);
        let n = conn.execute(
            "DELETE FROM generations WHERE output_dir = ?1 AND filename = ?2",
            params![dir_key, filename],
        )?;
        Ok(n > 0)
    }

    /// What the gallery takes on disk, live and trashed, from the rows' own
    /// recorded sizes — one aggregate query, never a directory walk. A row
    /// with no recorded size counts as a print and adds no bytes.
    pub fn storage_totals(&self) -> Result<mold_core::GalleryStorage> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT trashed_at_ms IS NOT NULL, COUNT(*), COALESCE(SUM(MAX(file_size_bytes, 0)), 0)
             FROM generations GROUP BY trashed_at_ms IS NOT NULL",
        )?;
        let mut totals = mold_core::GalleryStorage::default();
        for row in stmt.query_map([], |r| {
            Ok((
                r.get::<_, bool>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })? {
            let (trashed, prints, bytes) = row?;
            let prints = prints.max(0) as u64;
            let bytes = bytes.max(0) as u64;
            if trashed {
                totals.trash_prints = prints;
                totals.trash_bytes = bytes;
            } else {
                totals.prints = prints;
                totals.bytes = bytes;
            }
        }
        Ok(totals)
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
            "SELECT output_dir, filename, file_mtime_ms, file_size_bytes, trashed_at_ms
             FROM generations",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PathSnapshot {
                output_dir: r.get::<_, String>(0)?,
                filename: r.get::<_, String>(1)?,
                file_mtime_ms: r.get::<_, Option<i64>>(2)?,
                file_size_bytes: r.get::<_, Option<i64>>(3)?,
                trashed_at_ms: r.get::<_, Option<i64>>(4)?,
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

    /// Run `f` inside a single IMMEDIATE transaction with a caller-typed
    /// error. Used by the organization module so its typed `NotFound` /
    /// `Conflict` / `Invalid` outcomes survive the transaction boundary
    /// instead of being flattened into `anyhow`.
    pub(crate) fn transact_typed<R, E>(
        &self,
        f: impl FnOnce(&Connection) -> std::result::Result<R, E>,
    ) -> std::result::Result<R, E>
    where
        E: From<rusqlite::Error>,
    {
        let mut conn = self.conn.lock().expect("metadata db mutex poisoned");
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let result = f(&tx)?;
        tx.commit()?;
        Ok(result)
    }

    /// Read-only twin of [`Self::transact_typed`]: run `f` against the
    /// locked connection with a caller-typed error.
    pub(crate) fn with_conn_typed<R, E>(
        &self,
        f: impl FnOnce(&Connection) -> std::result::Result<R, E>,
    ) -> std::result::Result<R, E> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        f(&conn)
    }

    /// Run `f` in an IMMEDIATE transaction. This takes SQLite's writer
    /// reservation before any reads, for read-validate-write invariants that
    /// must serialize with other process-local or external writers.
    pub(crate) fn transact_immediate<R>(
        &self,
        f: impl FnOnce(&Connection) -> Result<R>,
    ) -> Result<R> {
        let mut conn = self.conn.lock().expect("metadata db mutex poisoned");
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let result = f(&tx)?;
        tx.commit()?;
        Ok(result)
    }

    /// Run `f` against the locked connection. Exposed to sibling modules
    /// (settings, model_prefs, prompt_history) so they don't need to
    /// re-implement the mutex dance for every read/write.
    pub fn with_conn<R>(&self, f: impl FnOnce(&Connection) -> Result<R>) -> Result<R> {
        let conn = self.conn.lock().expect("metadata db mutex poisoned");
        f(&conn)
    }
}

/// Internal helper that takes an already-locked connection — lets callers
/// batch many upserts inside one transaction.
///
/// Canonicalizes `rec.output_dir` before persisting so the `UNIQUE(output_dir,
/// filename)` constraint matches rows written by different callers for the
/// same underlying file (e.g. the CLI canonicalizes its saved path, the
/// server used the raw `config.effective_output_dir()`). The canonical form
/// is what actually lands in the DB — callers get consistent keys for free.
pub(crate) fn upsert_with_conn(conn: &Connection, rec: &GenerationRecord) -> Result<i64> {
    upsert_with_conn_reporting_organization(conn, rec).map(|(id, _)| id)
}

/// [`upsert_with_conn`], additionally reporting the creation-time filing
/// seeded onto a freshly inserted row.
pub(crate) fn upsert_with_conn_reporting_organization(
    conn: &Connection,
    rec: &GenerationRecord,
) -> Result<(i64, crate::organization::SeededOrganization)> {
    let dir_key = canonical_dir_string(Path::new(&rec.output_dir));
    let scheduler_str = rec
        .metadata
        .scheduler
        .as_ref()
        .map(scheduler_to_str)
        .map(str::to_string);
    let metadata_json = serde_json::to_string(&rec.metadata)?;
    // Whether this is an insert has to be known BEFORE the upsert: it is
    // what makes the creation-time filing seed-once. Tags and collection
    // membership are user-owned the moment the print exists, exactly like
    // `title`, so a reconcile refresh or a re-publication must not resurrect
    // a tag the user removed.
    let is_insert = conn
        .query_row(
            "SELECT 1 FROM generations WHERE output_dir = ?1 AND filename = ?2",
            params![dir_key, rec.filename],
            |_| Ok(()),
        )
        .optional()?
        .is_none();
    // `title`, `favorite`, and `trashed_at_ms` are user-owned: `title` is
    // seeded on insert and kept on conflict (an existing title always wins
    // over the incoming one), while `favorite` / `trashed_at_ms` are only
    // ever written by the organization and trash modules. A reconcile refresh
    // or a re-publication must never reset them.
    conn.execute(
        "INSERT INTO generations (
            filename, output_dir, created_at_ms, file_mtime_ms, file_size_bytes, format,
            model, prompt, negative_prompt, original_prompt, seed, steps, guidance,
            width, height, strength, scheduler, lora, lora_scale, frames, fps,
            metadata_version, generation_time_ms, backend, hostname, source, metadata_synthetic,
            metadata_json, title, favorite, trashed_at_ms,
            queue_job_id, queue_job_metadata_state
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
            ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33
         )
         ON CONFLICT(output_dir, filename) DO UPDATE SET
            title = COALESCE(generations.title, excluded.title),
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
            metadata_synthetic = excluded.metadata_synthetic,
            metadata_json = excluded.metadata_json,
            queue_job_id = excluded.queue_job_id,
            queue_job_metadata_state = excluded.queue_job_metadata_state",
        params![
            rec.filename,
            dir_key,
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
            metadata_json,
            rec.title,
            rec.favorite as i64,
            rec.trashed_at_ms,
            rec.metadata.job_id,
            1_i64,
        ],
    )?;
    // An older binary updating `metadata_json` cannot supply the v25
    // projection, so the migration trigger marks it unknown. On the current
    // writer's conflict-update path that same trigger runs after the UPSERT;
    // repair the projection immediately while the DB mutex/transaction still
    // fences readers. Inserts already carry these values, and this idempotent
    // update keeps both paths identical.
    conn.execute(
        "UPDATE generations
            SET queue_job_id = ?1, queue_job_metadata_state = 1
          WHERE output_dir = ?2 AND filename = ?3",
        params![rec.metadata.job_id, dir_key, rec.filename],
    )?;
    let id = conn.query_row(
        "SELECT id FROM generations WHERE output_dir = ?1 AND filename = ?2",
        params![dir_key, rec.filename],
        |r| r.get::<_, i64>(0),
    )?;
    // Seed the creation-time filing from the embedded metadata, on the
    // insert branch only. Doing it here rather than at each publication site
    // means every path — server queue, per-GPU worker, chain runner, CLI,
    // TUI, and reconcile-from-disk for a file that lost its row — files the
    // print identically, and none of them can forget to.
    let seeded = if is_insert {
        crate::organization::seed_creation_organization(
            conn,
            id,
            &rec.metadata,
            mold_core::time::now_epoch_ms(),
        )
        .map_err(|e| anyhow::anyhow!("seeding creation-time organization failed: {e}"))?
    } else {
        crate::organization::SeededOrganization::default()
    };
    Ok((id, seeded))
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

pub(crate) fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<GenerationRecord> {
    let format_s: String = row.get(6)?;
    let filename: String = row.get(1)?;
    // A stored string this build's enum doesn't know must not become `Png` —
    // that mislabels the row's media kind for every consumer (a `.wav` row
    // rendered as an image, a `.mp4` row served as `image/png`). Fall back to
    // the filename extension, which is the same evidence the reconcile walk
    // uses, and only then to the historical default.
    let format = format_from_str(&format_s)
        .or_else(|| crate::metadata_io::format_from_path(std::path::Path::new(&filename)))
        .unwrap_or(OutputFormat::Png);
    let scheduler_s: Option<String> = row.get(17)?;
    let scheduler = scheduler_s.as_deref().and_then(scheduler_from_str);
    let legacy_metadata = OutputMetadata {
        video_only: None,
        attention_path: None,
        int8_arm: None,
        collection: None,
        tags: None,
        title: None,
        generation_time_ms: None,
        source_fit: None,
        guidance_overrides: None,
        sample_shift: None,
        distill_strength_high: None,
        distill_strength_low: None,
        job_id: None,
        model: row.get(7)?,
        prompt: row.get(8)?,
        negative_prompt: row.get(9)?,
        original_prompt: row.get(10)?,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        output_mode: None,
        seed: row.get::<_, i64>(11)? as u64,
        steps: row.get::<_, i64>(12)? as u32,
        guidance: row.get(13)?,
        width: row.get::<_, i64>(14)? as u32,
        height: row.get::<_, i64>(15)? as u32,
        generation_width: None,
        mesh: None,
        generation_height: None,
        strength: row.get(16)?,
        source_image_name: None,
        source_image_sha256: None,
        edit_image_sha256s: None,
        references: None,
        keyframes: None,
        scheduler,
        output_format: Some(format),
        cfg_plus: None,
        lora: row.get(18)?,
        lora_scale: row.get(19)?,
        loras: None,
        control_model: None,
        control_scale: None,
        upscale_model: None,
        gif_preview: None,
        enable_audio: None,
        audio_file_path: None,
        source_video_path: None,
        extend_video_path: None,
        extend_overlap_frames: None,
        pipeline: None,
        pipeline_requested: None,
        duration_prediction_requested: None,
        pipeline_provenance_sha256: None,
        source_preprocessing: None,
        ic_lora_control: None,
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        frames: row.get::<_, Option<i64>>(20)?.map(|n| n as u32),
        fps: row.get::<_, Option<i64>>(21)?.map(|n| n as u32),
        chain_job_id: None,
        chain: None,
        version: row.get(22)?,
        id_image_name: None,
        id_image_sha256: None,
        id_weight: None,
        id_start_step: None,
        id_image_names: None,
        id_image_sha256s: None,
        true_cfg: None,
        cfg_start_step: None,
    };
    let source_s: String = row.get(26)?;
    let synthetic_i: i64 = row.get(27)?;
    let metadata_json: Option<String> = row.get(28)?;
    let title: Option<String> = row.get(29)?;
    let favorite_i: i64 = row.get(30)?;
    let trashed_at_ms: Option<i64> = row.get(31)?;
    let metadata = metadata_json
        .as_deref()
        .and_then(|json| serde_json::from_str::<OutputMetadata>(json).ok())
        .unwrap_or(legacy_metadata);
    Ok(GenerationRecord {
        id: Some(row.get(0)?),
        filename,
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
        title,
        favorite: favorite_i != 0,
        trashed_at_ms,
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
        OutputFormat::Wav => "wav",
        OutputFormat::Glb => "glb",
        OutputFormat::Obj => "obj",
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
        "wav" => OutputFormat::Wav,
        "glb" => OutputFormat::Glb,
        "obj" => OutputFormat::Obj,
        _ => return None,
    })
}

fn scheduler_to_str(s: &Scheduler) -> &'static str {
    match s {
        Scheduler::Ddim => "ddim",
        Scheduler::EulerAncestral => "euler-ancestral",
        Scheduler::UniPc => "uni-pc",
        Scheduler::EdmDpmPp2m => "edm-dpm-pp-2m",
        Scheduler::Euler => "euler",
        Scheduler::DpmPp => "dpm-pp",
    }
}

fn scheduler_from_str(s: &str) -> Option<Scheduler> {
    Some(match s {
        "ddim" => Scheduler::Ddim,
        "euler-ancestral" => Scheduler::EulerAncestral,
        "uni-pc" | "unipc" => Scheduler::UniPc,
        "edm-dpm-pp-2m" => Scheduler::EdmDpmPp2m,
        "euler" => Scheduler::Euler,
        "dpm-pp" | "dpm++" | "dpmpp" => Scheduler::DpmPp,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_timeout_pragma_is_set_on_open() {
        // Regression: without a busy timeout, concurrent access from a
        // second process (TUI + `mold serve` sharing mold.db) surfaces
        // SQLITE_BUSY instantly, and most read paths swallow the error
        // as "no value" — observed as settings silently reading back
        // empty under contention.
        let tmp = std::env::temp_dir().join(format!(
            "mold-db-busy-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let db = MetadataDb::open(&tmp.join("mold.db")).unwrap();
        let timeout: i64 = db
            .conn
            .lock()
            .unwrap()
            .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
            .unwrap();
        assert_eq!(timeout, 5000, "busy_timeout must be pinned at 5000ms");
        let _ = std::fs::remove_dir_all(&tmp);
    }
    use crate::record::GenerationRecord;
    use mold_core::OutputMetadata;
    use std::path::Path;

    fn meta() -> OutputMetadata {
        OutputMetadata {
            video_only: None,
            attention_path: None,
            int8_arm: None,
            collection: None,
            tags: None,
            title: None,
            generation_time_ms: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "a cat".into(),
            negative_prompt: Some("blurry".into()),
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            model: "flux-dev:q4".into(),
            seed: 42,
            steps: 20,
            guidance: 4.0,
            width: 1024,
            height: 1024,
            generation_width: Some(1024),
            mesh: None,
            generation_height: Some(1024),
            strength: Some(0.8),
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: Some(Scheduler::Ddim),
            output_format: Some(OutputFormat::Png),
            cfg_plus: None,
            lora: Some("style.safetensors".into()),
            lora_scale: Some(1.0),
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            duration_prediction_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            frames: None,
            fps: None,
            chain_job_id: None,
            chain: None,
            version: "0.8.1".into(),
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
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
            title: None,
            favorite: false,
            trashed_at_ms: None,
        }
    }

    #[test]
    fn upsert_batch_commits_every_record_in_one_transaction() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut first = rec();
        first.filename = "first.png".into();
        let mut second = rec();
        second.filename = "second.png".into();

        let ids = db.upsert_batch(&[first, second]).unwrap();

        assert_eq!(ids.len(), 2);
        assert_eq!(db.count().unwrap(), 2);
    }

    #[test]
    fn upsert_batch_rolls_back_every_record_when_one_insert_fails() {
        let db = MetadataDb::open_in_memory().unwrap();
        db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TRIGGER reject_second
                 BEFORE INSERT ON generations
                 WHEN NEW.filename = 'second.png'
                 BEGIN SELECT RAISE(ABORT, 'injected batch failure'); END;",
            )?;
            Ok(())
        })
        .unwrap();
        let mut first = rec();
        first.filename = "first.png".into();
        let mut second = rec();
        second.filename = "second.png".into();

        assert!(db.upsert_batch(&[first, second]).is_err());
        assert_eq!(db.count().unwrap(), 0);
    }

    /// The Storage card's "pictures take N GB": live and trashed rows summed
    /// separately from their own recorded sizes; a row that never recorded
    /// one still counts as a print.
    #[test]
    fn storage_totals_sum_live_and_trashed_rows_separately() {
        let db = MetadataDb::open_in_memory().unwrap();
        assert_eq!(
            db.storage_totals().unwrap(),
            mold_core::GalleryStorage::default()
        );

        let mut a = rec();
        a.file_size_bytes = Some(1_000);
        db.upsert(&a).unwrap();
        let mut b = rec();
        b.filename = "b.png".into();
        b.file_size_bytes = Some(2_500);
        db.upsert(&b).unwrap();
        let mut c = rec();
        c.filename = "c.png".into();
        c.file_size_bytes = None;
        db.upsert(&c).unwrap();
        assert_eq!(
            db.storage_totals().unwrap(),
            mold_core::GalleryStorage {
                prints: 3,
                bytes: 3_500,
                trash_prints: 0,
                trash_bytes: 0,
            }
        );

        assert!(db
            .mark_trashed(Path::new("/tmp/out"), "b.png", 5_000)
            .unwrap());
        assert_eq!(
            db.storage_totals().unwrap(),
            mold_core::GalleryStorage {
                prints: 2,
                bytes: 1_000,
                trash_prints: 1,
                trash_bytes: 2_500,
            }
        );
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
    fn current_upsert_keeps_the_queue_job_projection_exact() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut record = rec();
        record.metadata.job_id = Some("first-job".into());
        db.upsert(&record).unwrap();
        let projection = |record: &GenerationRecord| {
            db.with_conn(|conn| {
                conn.query_row(
                    "SELECT queue_job_id, queue_job_metadata_state
                       FROM generations
                      WHERE output_dir = ?1 AND filename = ?2",
                    params![
                        canonical_dir_string(Path::new(&record.output_dir)),
                        record.filename
                    ],
                    |row| Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?)),
                )
                .map_err(Into::into)
            })
            .unwrap()
        };
        assert_eq!(projection(&record), (Some("first-job".into()), 1));

        record.metadata.job_id = Some("replacement-job".into());
        db.upsert(&record).unwrap();
        assert_eq!(
            projection(&record),
            (Some("replacement-job".into()), 1),
            "the old-writer dirty trigger must not leave a current UPSERT unknown"
        );
    }

    #[test]
    fn upsert_round_trips_full_metadata_json_for_recreate() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut rec = rec();
        rec.metadata.loras = Some(vec![mold_core::LoraWeight {
            path: "/loras/style.safetensors".into(),
            scale: 0.6,

            expert: None,
        }]);
        rec.metadata.output_format = Some(OutputFormat::Png);
        rec.metadata.cfg_plus = Some(true);
        rec.metadata.control_model = Some("controlnet-canny-sd15".into());
        rec.metadata.control_scale = Some(0.8);
        rec.metadata.batch_id = Some("prepared-batch-1".into());
        rec.metadata.batch_index = Some(2);
        rec.metadata.batch_count = Some(3);
        rec.metadata.references = Some(vec![
            mold_core::GenerationReferenceMetadata {
                kind: mold_core::GenerationReferenceKind::Video,
                index: 1,
                name: Some("opening.mp4".into()),
                sha256: "a".repeat(64),
                mime_type: "video/mp4".into(),
                width: Some(1920),
                height: Some(1080),
                frame_count: Some(120),
                duration_ms: Some(4_000),
                fps: Some(24.0),
                has_audio: true,
                audio_duration_ms: Some(4_000),
                audio_sample_count: Some(192_000),
                audio_sample_rate: Some(48_000),
                audio_channels: Some(2),
                sample_rate: None,
                channels: None,
                sample_count: None,
                prepared_shape: Some(mold_core::minimax_h3::GenerationReferencePreparedShape {
                    version: mold_core::minimax_h3::REFERENCE_PREPROCESS_VERSION,
                    normalized_width: Some(1344),
                    normalized_height: Some(768),
                    normalized_video_frames: Some(96),
                    video_frames: Some(90),
                    qwen_video_frames: Some(8),
                    audio_samples_per_channel: Some(128_000),
                    visual_rows: 27_216,
                    audio_rows: 320,
                }),
                crop: None,
            },
            mold_core::GenerationReferenceMetadata {
                kind: mold_core::GenerationReferenceKind::Audio,
                index: 2,
                name: Some("voice.wav".into()),
                sha256: "b".repeat(64),
                mime_type: "audio/wav".into(),
                width: None,
                height: None,
                frame_count: None,
                duration_ms: Some(3_000),
                fps: None,
                has_audio: false,
                audio_duration_ms: None,
                audio_sample_count: None,
                audio_sample_rate: None,
                audio_channels: None,
                sample_rate: Some(32_000),
                channels: Some(2),
                sample_count: Some(96_000),
                prepared_shape: Some(mold_core::minimax_h3::GenerationReferencePreparedShape {
                    version: mold_core::minimax_h3::REFERENCE_PREPROCESS_VERSION,
                    normalized_width: None,
                    normalized_height: None,
                    normalized_video_frames: None,
                    video_frames: None,
                    qwen_video_frames: None,
                    audio_samples_per_channel: Some(96_000),
                    visual_rows: 0,
                    audio_rows: 240,
                }),
                crop: None,
            },
        ]);

        db.upsert(&rec).unwrap();
        let got = db
            .get(Path::new("/tmp/out"), "mold-flux-dev-q4-1.png")
            .unwrap()
            .expect("row should exist");

        assert_eq!(got.metadata.output_format, Some(OutputFormat::Png));
        assert_eq!(got.metadata.cfg_plus, Some(true));
        assert_eq!(
            got.metadata.loras.as_ref().unwrap()[0].path,
            "/loras/style.safetensors"
        );
        assert_eq!(
            got.metadata.control_model.as_deref(),
            Some("controlnet-canny-sd15")
        );
        assert_eq!(got.metadata.control_scale, Some(0.8));
        assert_eq!(got.metadata.batch_id.as_deref(), Some("prepared-batch-1"));
        assert_eq!(got.metadata.batch_index, Some(2));
        assert_eq!(got.metadata.batch_count, Some(3));
        let references = got.metadata.references.expect("ordered references");
        assert_eq!(references.len(), 2);
        assert_eq!(references[0].index, 1);
        assert_eq!(
            references[0].kind,
            mold_core::GenerationReferenceKind::Video
        );
        assert!(references[0].has_audio);
        assert_eq!(
            references[0]
                .prepared_shape
                .as_ref()
                .map(|shape| shape.visual_rows),
            Some(27_216)
        );
        assert_eq!(references[1].index, 2);
        assert_eq!(
            references[1].kind,
            mold_core::GenerationReferenceKind::Audio
        );
        assert_eq!(references[1].sha256, "b".repeat(64));
        assert_eq!(
            references[1]
                .prepared_shape
                .as_ref()
                .map(|shape| shape.audio_rows),
            Some(240)
        );
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

    /// The user-owned columns survive a conflicting upsert (reconcile
    /// refresh, re-publication): `title` keeps the existing value, and
    /// `favorite` / `trashed_at_ms` are never written from the incoming row.
    #[test]
    fn upsert_preserves_title_favorite_and_trashed_on_conflict() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut first = rec();
        first.title = Some("Owl study".into());
        db.upsert(&first).unwrap();
        db.set_favorite(Path::new("/tmp/out"), &first.filename, true)
            .unwrap();
        assert!(db
            .mark_trashed(Path::new("/tmp/out"), &first.filename, 77)
            .unwrap());

        // A backfill refresh carries neither a title nor the flags.
        let mut refresh = rec();
        refresh.metadata.prompt = "refreshed".into();
        refresh.title = Some("Reconciled title that must lose".into());
        db.upsert(&refresh).unwrap();

        let got = db
            .get(Path::new("/tmp/out"), &first.filename)
            .unwrap()
            .unwrap();
        assert_eq!(got.metadata.prompt, "refreshed");
        assert_eq!(got.title.as_deref(), Some("Owl study"));
        assert!(got.favorite);
        assert_eq!(got.trashed_at_ms, Some(77));
    }

    /// A row inserted without a title takes the title from the first upsert
    /// that carries one (the COALESCE arm), since `NULL` means "never set".
    #[test]
    fn upsert_seeds_title_when_existing_row_has_none() {
        let db = MetadataDb::open_in_memory().unwrap();
        db.upsert(&rec()).unwrap();
        let mut titled = rec();
        titled.title = Some("Late title".into());
        db.upsert(&titled).unwrap();
        let got = db
            .get(Path::new("/tmp/out"), &titled.filename)
            .unwrap()
            .unwrap();
        assert_eq!(got.title.as_deref(), Some("Late title"));
    }

    #[test]
    fn insert_seeds_title_and_round_trips_organization_columns() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut r = rec();
        r.title = Some("Seeded".into());
        r.favorite = true;
        r.trashed_at_ms = Some(5);
        db.upsert(&r).unwrap();
        let got = db.get(Path::new("/tmp/out"), &r.filename).unwrap().unwrap();
        assert_eq!(got.title.as_deref(), Some("Seeded"));
        assert!(got.favorite);
        assert_eq!(got.trashed_at_ms, Some(5));
    }

    // ── creation-time filing (tags / collection) ────────────────────────

    /// A print that arrives already tagged and filed lands with its tags
    /// attached and its collection created, resolved by slug.
    #[test]
    fn insert_seeds_creation_time_tags_and_creates_the_collection() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut r = rec();
        r.metadata.tags = Some(vec!["smurfs".into(), "village".into()]);
        r.metadata.collection = Some("Smurf Village".into());

        let (_, seeded) = db.upsert_reporting_organization(&r).unwrap();
        assert_eq!(
            seeded.tags,
            vec!["smurfs".to_string(), "village".to_string()]
        );
        assert_eq!(seeded.collection.as_deref(), Some("Smurf Village"));
        assert!(seeded.created_collection);

        let org = db
            .print_organization(Path::new("/tmp/out"), &r.filename)
            .unwrap()
            .unwrap();
        assert_eq!(org.tags, vec!["smurfs".to_string(), "village".to_string()]);
        let collections = db.list_collections().unwrap();
        assert_eq!(collections.len(), 1);
        assert_eq!(collections[0].name, "Smurf Village");
        assert_eq!(collections[0].slug, "smurf-village");
        assert_eq!(collections[0].count, 1);
    }

    /// A second print filed under the same name joins the EXISTING
    /// collection — the slug is the identity, which is what lets one name
    /// mean one collection across a fleet.
    #[test]
    fn seeding_reuses_an_existing_collection_by_slug() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut first = rec();
        first.filename = "first.png".into();
        first.metadata.collection = Some("Smurf Village".into());
        let (_, seeded) = db.upsert_reporting_organization(&first).unwrap();
        assert!(seeded.created_collection);

        let mut second = rec();
        second.filename = "second.png".into();
        // Different spelling, same slug.
        second.metadata.collection = Some("  smurf   VILLAGE  ".into());
        let (_, seeded) = db.upsert_reporting_organization(&second).unwrap();
        assert!(
            !seeded.created_collection,
            "the second print must join the existing collection, not make a new one"
        );

        let collections = db.list_collections().unwrap();
        assert_eq!(collections.len(), 1, "{collections:?}");
        // The original display name survives — a later print does not rename
        // the collection it joins.
        assert_eq!(collections[0].name, "Smurf Village");
        assert_eq!(collections[0].count, 2);
    }

    /// Organization is user-owned the moment the print exists. A reconcile
    /// refresh or a re-publication carrying the original metadata must NOT
    /// resurrect a tag the user removed, nor re-file a print they took out
    /// of a collection.
    #[test]
    fn seeding_is_once_and_a_later_upsert_never_re_applies_it() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut r = rec();
        r.metadata.tags = Some(vec!["smurfs".into()]);
        r.metadata.collection = Some("Smurf Village".into());
        db.upsert(&r).unwrap();

        // The user removes the tag and empties the collection.
        db.replace_tags(Path::new("/tmp/out"), &r.filename, &[])
            .unwrap();
        let collection_id = db.list_collections().unwrap()[0].id.clone();
        db.collection_remove(&collection_id, Path::new("/tmp/out"), &[r.filename.clone()])
            .unwrap();

        // A reconcile refresh re-upserts the very same record.
        let (_, seeded) = db.upsert_reporting_organization(&r).unwrap();
        assert!(
            seeded.is_empty(),
            "a conflicting upsert must seed nothing: {seeded:?}"
        );
        let org = db
            .print_organization(Path::new("/tmp/out"), &r.filename)
            .unwrap()
            .unwrap();
        assert!(org.tags.is_empty(), "{org:?}");
        assert_eq!(
            db.collection_filenames(&collection_id).unwrap(),
            Vec::<String>::new()
        );
    }

    /// Embedded metadata is bytes on disk, so a hand-edited or
    /// foreign-written chunk must not smuggle an invalid tag or a slugless
    /// collection past the rules — and must never fail the publication of a
    /// render that already succeeded.
    #[test]
    fn seeding_drops_invalid_values_instead_of_failing_the_publication() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut r = rec();
        r.metadata.tags = Some(vec![
            "good".into(),
            "bad\u{0}".into(),
            "x".repeat(mold_core::MAX_TAG_CHARS + 1),
            "  ".into(),
            "GOOD".into(),
        ]);
        r.metadata.collection = Some("日本語".into());

        let (_, seeded) = db.upsert_reporting_organization(&r).unwrap();
        assert_eq!(seeded.tags, vec!["good".to_string()]);
        assert_eq!(seeded.collection, None);
        assert!(!seeded.created_collection);
        assert!(db.list_collections().unwrap().is_empty());
        let got = db.get(Path::new("/tmp/out"), &r.filename).unwrap();
        assert!(got.is_some(), "the print still publishes");
    }

    /// The point of materializing the filing at admission: what a print's
    /// embedded metadata records and what its row actually holds must be the
    /// same values, byte for byte. Before materialization a raw-spelling HTTP
    /// client stamped `[" Smurfs ", "smurfs"]` into provenance while the row
    /// seeded one `Smurfs`, so Reuse restored duplicates.
    #[test]
    fn embedded_metadata_and_the_seeded_row_agree_after_materialization() {
        let db = MetadataDb::open_in_memory().unwrap();

        let mut request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        // Exactly what a curl caller might send.
        request.tags = Some(vec![
            "  Smurfs  ".into(),
            "smurfs".into(),
            "".into(),
            " village  green ".into(),
        ]);
        request.collection = Some(mold_core::CollectionRef::by_name("  Smurf   Village  "));
        mold_core::validation::materialize_request_organization(&mut request).unwrap();

        let metadata = mold_core::OutputMetadata::from_generate_request(&request, 42, None, "test");
        let mut rec = rec();
        rec.metadata = metadata.clone();
        db.upsert(&rec).unwrap();

        let org = db
            .print_organization(Path::new("/tmp/out"), &rec.filename)
            .unwrap()
            .unwrap();
        assert_eq!(
            org.tags,
            metadata.tags.clone().unwrap(),
            "the row holds exactly the tags provenance claims"
        );
        let collections = db.list_collections().unwrap();
        assert_eq!(collections.len(), 1);
        assert_eq!(
            collections[0].name,
            metadata.collection.clone().unwrap(),
            "and exactly the collection name provenance claims"
        );
        // And those values are the canonical ones, not the raw spellings.
        assert_eq!(
            org.tags,
            vec!["Smurfs".to_string(), "village green".to_string()]
        );
        assert_eq!(collections[0].name, "Smurf Village");
    }

    /// The overwhelmingly common case: an unfiled print costs nothing and
    /// reports nothing.
    #[test]
    fn an_unfiled_print_seeds_nothing() {
        let db = MetadataDb::open_in_memory().unwrap();
        let (_, seeded) = db.upsert_reporting_organization(&rec()).unwrap();
        assert!(seeded.is_empty());
        assert_eq!(seeded, crate::organization::SeededOrganization::default());
        assert!(db.list_tags().unwrap().is_empty());
        assert!(db.list_collections().unwrap().is_empty());
    }

    #[test]
    fn list_live_excludes_trashed_rows_and_list_keeps_everything() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut live = rec();
        live.filename = "live.png".into();
        let mut trashed = rec();
        trashed.filename = "trashed.png".into();
        db.upsert(&live).unwrap();
        db.upsert(&trashed).unwrap();
        assert!(db
            .mark_trashed(Path::new("/tmp/out"), "trashed.png", 123)
            .unwrap());

        let dir = Path::new("/tmp/out");
        let all: Vec<_> = db
            .list(Some(dir))
            .unwrap()
            .into_iter()
            .map(|r| r.filename)
            .collect();
        assert_eq!(all.len(), 2, "list() returns everything: {all:?}");
        let live_only: Vec<_> = db
            .list_live(Some(dir))
            .unwrap()
            .into_iter()
            .map(|r| r.filename)
            .collect();
        assert_eq!(live_only, vec!["live.png"]);
        let trashed_only: Vec<_> = db
            .list_trashed(Some(dir))
            .unwrap()
            .into_iter()
            .map(|r| r.filename)
            .collect();
        assert_eq!(trashed_only, vec!["trashed.png"]);
        // The unscoped variants filter the same way.
        assert_eq!(db.list_live(None).unwrap().len(), 1);
        assert_eq!(db.list_trashed(None).unwrap().len(), 1);
        assert_eq!(db.list(None).unwrap().len(), 2);
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

    /// Fix 1 regression guard: two callers handing in different string
    /// forms of the same directory (e.g. macOS `/tmp` vs `/private/tmp`,
    /// or a symlink) must collapse to the same row. Before the fix, they
    /// produced two rows and the gallery showed duplicates.
    #[cfg(target_os = "macos")]
    #[test]
    fn upsert_and_get_collapse_macos_tmp_symlink_aliases() {
        let db = MetadataDb::open_in_memory().unwrap();
        // Create a real dir under /tmp so both `/tmp/...` and
        // `/private/tmp/...` resolve. tempdir_in("/tmp") forces the
        // non-canonical-first form.
        let tmp = tempfile::tempdir_in("/tmp").unwrap();
        let via_tmp = tmp.path().to_path_buf();
        let via_private = Path::new("/private").join(via_tmp.strip_prefix("/").unwrap_or(&via_tmp));
        assert!(via_private.exists(), "test setup sanity");

        // Upsert via one alias, query via the other.
        let mut r = rec();
        r.filename = "dup.png".into();
        r.output_dir = via_tmp.to_string_lossy().into_owned();
        db.upsert(&r).unwrap();

        let via_other = db.get(&via_private, "dup.png").unwrap();
        assert!(
            via_other.is_some(),
            "get via /private/tmp must find the row written via /tmp"
        );

        // And a second upsert via the *other* alias must hit the same row.
        r.output_dir = via_private.to_string_lossy().into_owned();
        r.metadata.prompt = "updated via alias".into();
        db.upsert(&r).unwrap();
        assert_eq!(
            db.count().unwrap(),
            1,
            "UNIQUE constraint must see the two aliases as one key"
        );
        let got = db.get(&via_tmp, "dup.png").unwrap().unwrap();
        assert_eq!(got.metadata.prompt, "updated via alias");
    }

    #[test]
    fn schema_version_matches_build_constant() {
        let db = MetadataDb::open_in_memory().unwrap();
        assert_eq!(db.schema_version().unwrap(), crate::SCHEMA_VERSION);
    }

    /// Fix 2 regression guard: opening a file-backed DB should land in WAL
    /// mode on every supported filesystem. tempdir gives us a writable
    /// path on the host OS.
    #[test]
    fn open_file_backed_db_lands_in_wal_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        let conn = db.conn.lock().unwrap();
        let mode: String = conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert!(
            mode.eq_ignore_ascii_case("wal"),
            "expected WAL journal mode, got {mode:?}"
        );
    }

    /// Fix 1 + 2 + 3 integration guard: a fresh file-backed DB applies the
    /// v1 migration, lands at `SCHEMA_VERSION`, and round-trips a row
    /// keyed via the canonical path helper.
    #[test]
    fn fresh_file_db_applies_v1_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("mold.db");
        let db = MetadataDb::open(&db_path).unwrap();
        assert_eq!(db.schema_version().unwrap(), crate::SCHEMA_VERSION);

        // Write under the tempdir itself so canonicalization has something
        // to resolve. The round-trip proves upsert + get both see the
        // same canonical key.
        let mut r = rec();
        r.output_dir = dir.path().to_string_lossy().into_owned();
        db.upsert(&r).unwrap();
        assert!(db.get(dir.path(), &r.filename).unwrap().is_some());
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

    fn quarantined_files(dir: &Path) -> Vec<std::path::PathBuf> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name().is_some_and(|name| {
                    let name = name.to_string_lossy();
                    name.starts_with("mold.db.corrupt-")
                        && !name.ends_with("-wal")
                        && !name.ends_with("-shm")
                })
            })
            .collect();
        paths.sort();
        paths
    }

    #[test]
    fn open_quarantines_malformed_database_and_rebuilds_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        std::fs::write(&path, b"this is not a sqlite database").unwrap();

        let db = MetadataDb::open(&path).expect("corruption should self-heal at open");

        assert_eq!(db.schema_version().unwrap(), crate::SCHEMA_VERSION);
        assert!(
            path.exists(),
            "a fresh database should replace the corrupt file"
        );
        assert_eq!(
            quarantined_files(dir.path()).len(),
            1,
            "the malformed database should be retained for operator inspection"
        );
    }

    #[test]
    fn quarantine_keeps_wal_and_shm_with_corrupt_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        std::fs::write(&path, b"db").unwrap();
        std::fs::write(sidecar_path(&path, "-wal"), b"wal").unwrap();
        std::fs::write(sidecar_path(&path, "-shm"), b"shm").unwrap();

        let quarantine = quarantine_corrupt_files(&path).unwrap();

        assert_eq!(std::fs::read(&quarantine).unwrap(), b"db");
        assert_eq!(
            std::fs::read(sidecar_path(&quarantine, "-wal")).unwrap(),
            b"wal"
        );
        assert_eq!(
            std::fs::read(sidecar_path(&quarantine, "-shm")).unwrap(),
            b"shm"
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"db");
        assert_eq!(std::fs::read(sidecar_path(&path, "-wal")).unwrap(), b"wal");
        assert_eq!(std::fs::read(sidecar_path(&path, "-shm")).unwrap(), b"shm");
    }

    #[test]
    fn concurrent_opens_quarantine_corruption_only_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        std::fs::write(&path, b"not a sqlite database").unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(5));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    MetadataDb::open(&path).map(|db| db.schema_version().unwrap())
                })
            })
            .collect();
        barrier.wait();
        for handle in handles {
            assert_eq!(handle.join().unwrap().unwrap(), crate::SCHEMA_VERSION);
        }
        assert_eq!(
            quarantined_files(dir.path()).len(),
            1,
            "recovery lock must prevent a fresh replacement from being quarantined"
        );
    }

    #[test]
    fn recovery_keeps_an_existing_second_handle_usable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let first = MetadataDb::open(&path).unwrap();
        let second = MetadataDb::open(&path).unwrap();
        {
            let conn = first.conn.lock().unwrap();
            conn.execute_batch(
                "PRAGMA writable_schema = ON;
                 UPDATE sqlite_schema SET rootpage = 2147483647
                   WHERE name = 'idx_gen_mtime';
                 PRAGMA writable_schema = OFF;
                 PRAGMA schema_version = 9999;",
            )
            .unwrap();
        }

        first.list(Some(dir.path())).unwrap();
        second
            .list(Some(dir.path()))
            .expect("an already-open connection must observe the in-place rebuild");
        assert_eq!(quarantined_files(dir.path()).len(), 1);
    }

    #[test]
    fn failed_recovery_never_turns_handle_into_ephemeral_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        {
            let conn = db.conn.lock().unwrap();
            conn.execute_batch(
                "PRAGMA writable_schema = ON;
                 UPDATE sqlite_schema SET rootpage = 2147483647
                   WHERE name = 'idx_gen_mtime';
                 PRAGMA writable_schema = OFF;
                 PRAGMA schema_version = 9999;",
            )
            .unwrap();
        }
        for suffix in ["-wal", "-shm", ".recovery.lock"] {
            let _ = std::fs::remove_file(sidecar_path(&path, suffix));
        }
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(dir.path()).unwrap();

        assert!(db.list(None).is_err(), "recovery setup should fail");
        assert!(
            db.list(None).is_err(),
            "later queries must keep failing instead of using an in-memory placeholder"
        );
    }

    #[test]
    fn list_quarantines_corrupt_index_and_reconciles_gallery() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let image_path = dir.path().join("recovered.png");
        let image = image::ImageBuffer::from_fn(64u32, 64u32, |x, y| {
            if (x + y) % 2 == 0 {
                image::Rgb([255u8, 32, 16])
            } else {
                image::Rgb([16u8, 160, 255])
            }
        });
        image.save(&image_path).unwrap();

        let db = MetadataDb::open(&path).unwrap();
        db.reconcile(dir.path()).unwrap();
        assert_eq!(db.list(Some(dir.path())).unwrap().len(), 1);

        // Repoint only the ordering index at a page beyond EOF. The table
        // remains readable, mirroring the production corruption signature.
        {
            let conn = db.conn.lock().unwrap();
            conn.execute_batch(
                "PRAGMA writable_schema = ON;
                 UPDATE sqlite_schema SET rootpage = 2147483647
                   WHERE name = 'idx_gen_mtime';
                 PRAGMA writable_schema = OFF;
                 PRAGMA schema_version = 9999;",
            )
            .unwrap();
        }

        let rows = db
            .list(Some(dir.path()))
            .expect("gallery listing should quarantine, reconcile, and retry");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].filename, "recovered.png");
        assert_eq!(rows[0].source, RecordSource::Backfill);
        assert_eq!(quarantined_files(dir.path()).len(), 1);
    }
}
