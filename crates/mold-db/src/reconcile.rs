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

use anyhow::Result;

use crate::db::{delete_with_conn, upsert_with_conn, MetadataDb};
use crate::path::canonical_dir_string;
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
    /// Files that had a recognized extension but failed the size/header/
    /// solid-black guard rails — same set the server's filesystem walk
    /// already hides from the gallery.
    pub skipped_invalid: u64,
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
        // disk + DB diffs in a single transaction afterward. Use the
        // canonical form of `output_dir` — that's what upserts store, so
        // the snapshot filter must match.
        let existing = self.snapshot_paths()?;
        let dir_str = canonical_dir_string(output_dir);
        let mut existing_for_dir: HashMap<String, (Option<i64>, Option<i64>)> = existing
            .into_iter()
            .filter(|s| s.output_dir == dir_str)
            .map(|s| (s.filename, (s.file_mtime_ms, s.file_size_bytes)))
            .collect();

        let mut to_upsert: Vec<GenerationRecord> = Vec::new();

        for item in crate::scan::scan_output_dir(output_dir) {
            let file = match item {
                crate::scan::ScanItem::SkippedUnrelated => {
                    stats.skipped_unrelated += 1;
                    continue;
                }
                // Invalid files leave any existing row in the map; the
                // end-of-loop cleanup deletes everything still there
                // (covers both "file gone" and "file became invalid").
                crate::scan::ScanItem::SkippedInvalid => {
                    stats.skipped_invalid += 1;
                    continue;
                }
                crate::scan::ScanItem::Valid(file) => file,
            };

            // Decide: insert / refresh / keep. Metadata is parsed only
            // for new/changed files — the walker deliberately doesn't.
            match existing_for_dir.remove(&file.filename) {
                Some((row_mt, row_sz)) if row_mt == file.mtime_ms && row_sz == file.size_bytes => {
                    stats.kept += 1;
                }
                Some(_) => {
                    // Stat changed. Re-read embedded metadata in case the
                    // file was rewritten with new params.
                    let rec = build_backfill_record(
                        output_dir,
                        &file.filename,
                        file.format,
                        &file.path,
                        file.mtime_ms,
                        file.size_bytes,
                    );
                    to_upsert.push(rec);
                    stats.updated += 1;
                }
                None => {
                    let rec = build_backfill_record(
                        output_dir,
                        &file.filename,
                        file.format,
                        &file.path,
                        file.mtime_ms,
                        file.size_bytes,
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

fn now_ms() -> i64 {
    mold_core::time::now_epoch_ms()
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
    let (metadata, synthetic) =
        crate::metadata_io::read_or_synthesize(path, format, filename, timestamp_secs);
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
    use image::{ImageBuffer, Rgb};

    fn write(p: &Path, bytes: &[u8]) {
        std::fs::write(p, bytes).unwrap();
    }

    /// Synthesize a valid (non-black) raster PNG large enough to clear the
    /// 256-byte size floor and the solid-black sampler. 64x64 of varying
    /// pixel values weighs comfortably more than the 8 KB suspect-size
    /// ceiling so it survives every guard.
    fn write_valid_png(path: &Path) {
        let img = ImageBuffer::from_fn(64u32, 64u32, |x, y| {
            // High-contrast checkerboard so the file compresses well above
            // the suspect-size ceiling and never reads as solid black.
            if ((x / 8) + (y / 8)) % 2 == 0 {
                Rgb([255u8, 64, 32])
            } else {
                Rgb([16u8, 200, 240])
            }
        });
        img.save(path).unwrap();
    }

    /// Synthesize a valid MP4 stub with an `ftyp` box at offset 4 + enough
    /// padding to clear the 4096-byte floor.
    fn write_valid_mp4(path: &Path) {
        let mut bytes = Vec::with_capacity(8192);
        bytes.extend_from_slice(&[0u8, 0, 0, 0x20]); // box size
        bytes.extend_from_slice(b"ftyp");
        bytes.extend_from_slice(b"isom"); // major brand
        bytes.extend_from_slice(&[0u8, 0, 0, 1]); // minor version
        bytes.extend_from_slice(b"isomavc1mp41"); // compat brands
        bytes.resize(8192, 0);
        std::fs::write(path, &bytes).unwrap();
    }

    /// A GIF whose comment extension carries `mold:parameters` must
    /// import with real metadata (`metadata_synthetic == false`) — GIF
    /// comment parsing used to live only in the TUI, so reconcile
    /// synthesized rows for GIFs that carried full metadata.
    #[test]
    fn reconcile_recovers_embedded_gif_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let gif_path = tmp.path().join("mold-ltx-video-1700000000000.gif");

        // Encode a real, decodable GIF...
        let img = ImageBuffer::from_fn(64u32, 64u32, |x, y| {
            if ((x / 8) + (y / 8)) % 2 == 0 {
                Rgb([255u8, 64, 32])
            } else {
                Rgb([16u8, 200, 240])
            }
        });
        img.save(&gif_path).unwrap();

        // ...then splice a `mold:parameters` comment extension in front of
        // the trailer byte, the same shape mold's GIF writer produces.
        let mut bytes = std::fs::read(&gif_path).unwrap();
        let trailer = bytes.iter().rposition(|&b| b == 0x3B).unwrap();
        let comment = format!(
            "mold:parameters {}",
            r#"{"prompt":"a gif owl","model":"ltx-video","seed":7,"steps":30,"guidance":3.0,"width":64,"height":64,"version":"test"}"#
        );
        let mut ext = vec![0x21, 0xFE];
        for chunk in comment.as_bytes().chunks(255) {
            ext.push(chunk.len() as u8);
            ext.extend_from_slice(chunk);
        }
        ext.push(0);
        bytes.splice(trailer..trailer, ext);
        std::fs::write(&gif_path, &bytes).unwrap();

        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.imported, 1, "{stats:?}");

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].metadata_synthetic);
        assert_eq!(rows[0].metadata.prompt, "a gif owl");
        assert_eq!(rows[0].metadata.seed, 7);
    }

    #[test]
    fn reconcile_imports_unknown_files() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("mold-flux-dev-1.png"));
        write_valid_mp4(&tmp.path().join("mold-flux-dev-2.mp4"));
        write(&tmp.path().join("ignored.txt"), b"x");

        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.imported, 2);
        assert_eq!(stats.removed, 0);
        assert_eq!(stats.skipped_unrelated, 1);
        assert_eq!(stats.skipped_invalid, 0);
        assert_eq!(db.count().unwrap(), 2);
    }

    #[test]
    fn reconcile_removes_rows_for_missing_files() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("a.png"));
        write_valid_png(&tmp.path().join("b.png"));

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
        write_valid_png(&p);
        let original_size = std::fs::metadata(&p).unwrap().len() as i64;

        let db = MetadataDb::open_in_memory().unwrap();
        let _ = db.reconcile(tmp.path()).unwrap();
        let before = db.get(tmp.path(), "x.png").unwrap().unwrap();
        assert_eq!(before.file_size_bytes, Some(original_size));

        // Rewrite with different content but still valid.
        let other = tmp.path().join("other.png");
        write_valid_png(&other);
        // Ensure the new content has a different size to trigger refresh.
        let mut bytes = std::fs::read(&other).unwrap();
        bytes.extend_from_slice(b"trailing-junk-padding-to-change-size-but-not-validity");
        std::fs::write(&p, &bytes).unwrap();
        std::fs::remove_file(&other).unwrap();

        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.updated, 1, "size change should refresh the row");
        let after = db.get(tmp.path(), "x.png").unwrap().unwrap();
        assert_ne!(after.file_size_bytes, before.file_size_bytes);
    }

    #[test]
    fn reconcile_is_noop_for_missing_dir() {
        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db
            .reconcile(Path::new("/definitely/not/a/dir/here"))
            .unwrap();
        assert_eq!(stats, ReconcileStats::default());
    }

    /// Codex P2 finding 1: pre-existing truncated outputs from before the
    /// DB existed must NOT enter the gallery via reconcile. Mirrors the
    /// size/header/solid-black guard from `scan_gallery_dir`.
    #[test]
    fn reconcile_skips_invalid_files() {
        let tmp = tempfile::tempdir().unwrap();
        // Below the 256 B floor — looks like a PNG by name only.
        write(&tmp.path().join("tiny.png"), b"x");
        // Has the size but isn't really a PNG (random bytes).
        write(&tmp.path().join("bogus.png"), &vec![0u8; 1024]);
        // MP4 missing the ftyp box.
        write(&tmp.path().join("not-real.mp4"), &vec![0u8; 8192]);
        // One genuinely valid file so we know reconcile still imported.
        write_valid_png(&tmp.path().join("real.png"));

        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(
            stats.imported, 1,
            "only the well-formed PNG should be imported"
        );
        assert_eq!(stats.skipped_invalid, 3);
        assert_eq!(db.count().unwrap(), 1);
    }

    /// Codex P2 finding 2: synthetic backfill rows for files without
    /// embedded `mold:parameters` must still carry the file's real raster
    /// dimensions so the gallery card aspect ratio is correct.
    #[test]
    fn reconcile_synthetic_records_carry_real_dimensions() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("legacy.png");
        write_valid_png(&p);

        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.imported, 1);
        let row = db.get(tmp.path(), "legacy.png").unwrap().unwrap();
        assert!(row.metadata_synthetic, "no embedded chunk → synthetic");
        assert_eq!(row.metadata.width, 64, "width should come from header");
        assert_eq!(row.metadata.height, 64, "height should come from header");
    }

    /// A file that started valid but was later truncated should be
    /// dropped from the DB on the next reconcile pass — keeping `/api/gallery`
    /// in sync with what the legacy filesystem walk would have shown.
    #[test]
    fn reconcile_drops_rows_when_file_becomes_invalid() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("doomed.png");
        write_valid_png(&p);

        let db = MetadataDb::open_in_memory().unwrap();
        let s1 = db.reconcile(tmp.path()).unwrap();
        assert_eq!(s1.imported, 1);

        // Truncate to under the size floor.
        std::fs::write(&p, b"x").unwrap();
        let s2 = db.reconcile(tmp.path()).unwrap();
        assert_eq!(s2.skipped_invalid, 1);
        assert_eq!(s2.removed, 1);
        assert_eq!(db.count().unwrap(), 0);
    }
}
