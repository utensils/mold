//! Sync the metadata DB to the on-disk gallery directory.
//!
//! Two passes:
//!   1. Walk `output_dir`. For files the DB doesn't know about, insert a
//!      [`crate::GenerationRecord`] using embedded metadata when present,
//!      synthesizing the rest from the filename otherwise.
//!   2. Iterate every DB row scoped to `output_dir`. Drop rows whose file
//!      is missing on disk so deletes that happened outside the running
//!      server / CLI (manual `rm`, file manager, etc.) are reflected.
//!
//! This is intended to run once at server startup as a `tokio::spawn_blocking`
//! background task — it never blocks the request path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;

use crate::db::{delete_with_conn, upsert_with_conn, MetadataDb};
use crate::metadata_io::{format_from_path, read_embedded, synthesize_from_filename};
use crate::record::{GenerationRecord, RecordSource};

/// Counters returned by [`MetadataDb::reconcile`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconcileStats {
    /// Files that already had matching DB rows with the same mtime/size.
    pub kept: u64,
    /// Files that were on disk but not yet in the DB — added.
    pub imported: u64,
    /// DB rows whose mtime/size diverged from disk — refreshed in place.
    pub updated: u64,
    /// DB rows whose underlying file is missing — removed from the DB.
    pub removed: u64,
    /// Files we walked past because the extension isn't a gallery format.
    pub skipped_unrelated: u64,
}

impl MetadataDb {
    /// Walk `output_dir` and align the DB with what's on disk. Inserts
    /// new files, refreshes mtime/size for changed files, and drops rows
    /// whose backing file disappeared.
    pub fn reconcile(&self, output_dir: &Path) -> Result<ReconcileStats> {
        let mut stats = ReconcileStats::default();
        if !output_dir.is_dir() {
            // Nothing to walk. Still purge rows scoped to this dir if any
            // exist — typically means the user pointed MOLD_OUTPUT_DIR
            // somewhere else, so historical rows from another mount stick
            // around. Don't second-guess them.
            return Ok(stats);
        }

        // Snapshot existing rows for this dir up front so we can process
        // disk + DB diffs in a single transaction afterward.
        let existing = self.snapshot_paths()?;
        let dir_str = output_dir.to_string_lossy().into_owned();
        let mut existing_for_dir: HashMap<String, (Option<i64>, Option<i64>)> = existing
            .into_iter()
            .filter(|s| s.output_dir == dir_str)
            .map(|s| (s.filename, (s.file_mtime_ms, s.file_size_bytes)))
            .collect();

        let mut to_upsert: Vec<GenerationRecord> = Vec::new();
        let mut seen_filenames: Vec<String> = Vec::new();

        for entry in walkdir::WalkDir::new(output_dir)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(format) = format_from_path(path) else {
                stats.skipped_unrelated += 1;
                continue;
            };
            let filename = match path.file_name().and_then(|f| f.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };

            let fs_meta = entry.metadata().ok();
            let (mtime_ms, size_bytes) = stat_to_pair(fs_meta.as_ref());

            seen_filenames.push(filename.clone());

            // Decide: insert / refresh / keep.
            match existing_for_dir.remove(&filename) {
                Some((row_mt, row_sz)) if row_mt == mtime_ms && row_sz == size_bytes => {
                    stats.kept += 1;
                }
                Some(_) => {
                    // Stat changed. Re-read embedded metadata in case the
                    // file was rewritten with new params.
                    let rec = build_backfill_record(
                        output_dir, &filename, format, path, mtime_ms, size_bytes,
                    );
                    to_upsert.push(rec);
                    stats.updated += 1;
                }
                None => {
                    let rec = build_backfill_record(
                        output_dir, &filename, format, path, mtime_ms, size_bytes,
                    );
                    to_upsert.push(rec);
                    stats.imported += 1;
                }
            }
        }

        // Anything still in `existing_for_dir` was on disk last time we ran
        // but is gone now → drop it.
        let to_remove: Vec<String> = existing_for_dir.keys().cloned().collect();
        stats.removed = to_remove.len() as u64;

        if to_upsert.is_empty() && to_remove.is_empty() {
            return Ok(stats);
        }

        let dir_owned = dir_str.clone();
        self.transact(|conn| {
            for rec in &to_upsert {
                upsert_with_conn(conn, rec)?;
            }
            for filename in &to_remove {
                delete_with_conn(conn, &dir_owned, filename)?;
            }
            Ok(())
        })?;

        Ok(stats)
    }
}

fn stat_to_pair(meta: Option<&std::fs::Metadata>) -> (Option<i64>, Option<i64>) {
    let Some(m) = meta else {
        return (None, None);
    };
    let size = Some(m.len() as i64);
    let mtime = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64);
    (mtime, size)
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn build_backfill_record(
    output_dir: &Path,
    filename: &str,
    format: mold_core::OutputFormat,
    path: &Path,
    mtime_ms: Option<i64>,
    size_bytes: Option<i64>,
) -> GenerationRecord {
    let timestamp_secs = mtime_ms.map(|ms| ms / 1000).unwrap_or(0) as u64;
    let (metadata, synthetic) = match read_embedded(path, format) {
        Some(m) => (m, false),
        None => (synthesize_from_filename(filename, timestamp_secs), true),
    };
    let mut owned_dir = PathBuf::new();
    owned_dir.push(output_dir);
    GenerationRecord {
        id: None,
        filename: filename.to_string(),
        output_dir: owned_dir.to_string_lossy().into_owned(),
        created_at_ms: now_ms(),
        file_mtime_ms: mtime_ms,
        file_size_bytes: size_bytes,
        format,
        metadata,
        generation_time_ms: None,
        backend: None,
        hostname: None,
        source: RecordSource::Backfill,
        metadata_synthetic: synthetic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(p: &Path, bytes: &[u8]) {
        std::fs::write(p, bytes).unwrap();
    }

    #[test]
    fn reconcile_imports_unknown_files() {
        let tmp = tempfile::tempdir().unwrap();
        write(&tmp.path().join("mold-flux-dev-1.png"), b"abc");
        write(&tmp.path().join("mold-flux-dev-2.mp4"), b"defghi");
        write(&tmp.path().join("ignored.txt"), b"x");

        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.imported, 2);
        assert_eq!(stats.removed, 0);
        assert_eq!(stats.skipped_unrelated, 1);
        assert_eq!(db.count().unwrap(), 2);
    }

    #[test]
    fn reconcile_removes_rows_for_missing_files() {
        let tmp = tempfile::tempdir().unwrap();
        write(&tmp.path().join("a.png"), b"abc");
        write(&tmp.path().join("b.png"), b"def");

        let db = MetadataDb::open_in_memory().unwrap();
        let s1 = db.reconcile(tmp.path()).unwrap();
        assert_eq!(s1.imported, 2);

        std::fs::remove_file(tmp.path().join("a.png")).unwrap();
        let s2 = db.reconcile(tmp.path()).unwrap();
        assert_eq!(s2.removed, 1);
        assert_eq!(s2.kept, 1);
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn reconcile_refreshes_changed_size() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.png");
        write(&p, b"abc");

        let db = MetadataDb::open_in_memory().unwrap();
        let _ = db.reconcile(tmp.path()).unwrap();
        let before = db.get(tmp.path(), "x.png").unwrap().unwrap();
        assert_eq!(before.file_size_bytes, Some(3));

        // Rewrite with different content.
        write(&p, b"different bytes");
        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.updated, 1, "size change should refresh the row");
        let after = db.get(tmp.path(), "x.png").unwrap().unwrap();
        assert_eq!(after.file_size_bytes, Some(15));
    }

    #[test]
    fn reconcile_is_noop_for_missing_dir() {
        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db
            .reconcile(Path::new("/definitely/not/a/dir/here"))
            .unwrap();
        assert_eq!(stats, ReconcileStats::default());
    }
}
