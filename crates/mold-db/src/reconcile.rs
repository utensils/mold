//! Sync the metadata DB to the on-disk gallery directory.
//!
//! Three passes:
//!   1. Walk `output_dir` (depth 1 — `.trash/` is a directory, so the live
//!      walk never sees its contents). For files the DB doesn't know about,
//!      insert a [`crate::GenerationRecord`] using embedded metadata when
//!      present, synthesizing the rest from the filename otherwise. A row
//!      flagged trashed whose file is back in the live dir is un-trashed.
//!   2. Iterate every DB row scoped to `output_dir` that the live walk did
//!      not see. A trashed row is kept while `<dir>/.trash/<filename>` still
//!      exists and dropped otherwise. A live row whose file is gone but
//!      whose bytes sit in `.trash/` is re-flagged trashed (the tombstone's
//!      timestamp wins; DB-loss recovery). Everything else is dropped so
//!      deletes that happened outside the running server / CLI (manual
//!      `rm`, file manager, etc.) are reflected.
//!   3. Walk `<dir>/.trash/`. A valid gallery file with a tombstone and no
//!      row is imported as a trashed row (source `Backfill`, metadata from
//!      the tombstone or synthesized) and its tags / collection memberships
//!      are restored from the tombstone. Files without a tombstone are left
//!      alone — there is no provenance to import.
//!
//! This is intended to run once at server startup as a `tokio::spawn_blocking`
//! background task — it never blocks the request path.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::db::{delete_with_conn, upsert_with_conn, MetadataDb};
use crate::path::canonical_dir_string;
use crate::record::{GenerationRecord, RecordSource};
use crate::trash::{read_tombstone, tombstone_path, trash_dir, Tombstone};

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
    /// Trashed rows whose bytes are still in `.trash/` — kept as trashed.
    /// Also counts live rows that were re-flagged trashed because their
    /// file turned up in `.trash/` (DB-loss recovery).
    pub trashed_kept: u64,
    /// Tombstoned files in `.trash/` that had no row — imported as trashed.
    pub trashed_imported: u64,
    /// Trashed rows whose file reappeared in the live dir — un-trashed.
    pub trashed_restored: u64,
}

/// One DB row as the reconcile pass sees it.
#[derive(Debug, Clone, Copy)]
struct RowFacts {
    mtime_ms: Option<i64>,
    size_bytes: Option<i64>,
    trashed_at_ms: Option<i64>,
}

/// Row-level edits reconcile applies inside its single transaction,
/// besides upserts and deletes.
enum RowEdit {
    /// Set `trashed_at_ms` (live row whose bytes were found in `.trash/`).
    MarkTrashed {
        filename: String,
        trashed_at_ms: i64,
    },
    /// Clear `trashed_at_ms` (trashed row whose bytes are live again).
    MarkRestored { filename: String },
}

/// A trashed file with no row: import it and restore its organization.
struct TrashImport {
    record: GenerationRecord,
    tags: Vec<String>,
    collection_slugs: Vec<String>,
}

impl MetadataDb {
    /// Walk `output_dir` and align the DB with what's on disk. Inserts
    /// new files, refreshes mtime/size for changed files, drops rows
    /// whose backing file disappeared, and keeps the trash index honest
    /// (see the module docs).
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
        let mut existing_for_dir: HashMap<String, RowFacts> = existing
            .into_iter()
            .filter(|s| s.output_dir == dir_str)
            .map(|s| {
                (
                    s.filename,
                    RowFacts {
                        mtime_ms: s.file_mtime_ms,
                        size_bytes: s.file_size_bytes,
                        trashed_at_ms: s.trashed_at_ms,
                    },
                )
            })
            .collect();
        let all_row_filenames: HashSet<String> = existing_for_dir.keys().cloned().collect();

        let mut to_upsert: Vec<GenerationRecord> = Vec::new();
        let mut edits: Vec<RowEdit> = Vec::new();

        // ---- pass 1: live files ---------------------------------------
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
                Some(row) => {
                    if row.trashed_at_ms.is_some() {
                        // The bytes are live again (manual restore). The
                        // row's trash flag is stale — clear it.
                        edits.push(RowEdit::MarkRestored {
                            filename: file.filename.clone(),
                        });
                        stats.trashed_restored += 1;
                    }
                    if row.mtime_ms == file.mtime_ms && row.size_bytes == file.size_bytes {
                        stats.kept += 1;
                    } else {
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

        // ---- pass 2: rows the live walk did not see ---------------------
        // Trashed rows are kept while their bytes sit in `.trash/`; live
        // rows whose bytes moved there are re-flagged; everything else is
        // gone → drop it.
        let trash = trash_dir(output_dir);
        let mut to_remove: Vec<String> = Vec::new();
        for (filename, row) in existing_for_dir.drain() {
            let trashed_path = trash.join(&filename);
            if !trashed_path.is_file() {
                to_remove.push(filename);
                continue;
            }
            if row.trashed_at_ms.is_some() {
                stats.trashed_kept += 1;
                continue;
            }
            let trashed_at_ms = read_tombstone(&tombstone_path(&trash, &filename))
                .ok()
                .map(|t| t.trashed_at_ms)
                .or_else(|| {
                    std::fs::metadata(&trashed_path)
                        .ok()
                        .and_then(|m| crate::scan::stat_to_pair(Some(&m)).0)
                })
                .unwrap_or_else(now_ms);
            edits.push(RowEdit::MarkTrashed {
                filename,
                trashed_at_ms,
            });
            stats.trashed_kept += 1;
        }
        stats.removed = to_remove.len() as u64;

        // ---- pass 3: tombstoned files with no row -----------------------
        let mut trash_imports: Vec<TrashImport> = Vec::new();
        if trash.is_dir() {
            for item in crate::scan::scan_output_dir(&trash) {
                let crate::scan::ScanItem::Valid(file) = item else {
                    // Tombstones themselves are "unrelated"; invalid media
                    // in the trash is not worth a counter.
                    continue;
                };
                if all_row_filenames.contains(&file.filename) {
                    continue;
                }
                let Ok(tombstone) = read_tombstone(&tombstone_path(&trash, &file.filename)) else {
                    continue;
                };
                trash_imports.push(build_trash_import(
                    output_dir,
                    &file.filename,
                    file.format,
                    &file.path,
                    file.mtime_ms,
                    file.size_bytes,
                    tombstone,
                ));
                stats.trashed_imported += 1;
            }
        }

        if to_upsert.is_empty()
            && to_remove.is_empty()
            && edits.is_empty()
            && trash_imports.is_empty()
        {
            return Ok(stats);
        }

        let dir_owned = dir_str.clone();
        let now = now_ms();
        self.transact(|conn| {
            for rec in &to_upsert {
                upsert_with_conn(conn, rec)?;
            }
            for filename in &to_remove {
                delete_with_conn(conn, &dir_owned, filename)?;
            }
            for edit in &edits {
                match edit {
                    RowEdit::MarkTrashed {
                        filename,
                        trashed_at_ms,
                    } => {
                        conn.execute(
                            "UPDATE generations SET trashed_at_ms = ?3
                             WHERE output_dir = ?1 AND filename = ?2",
                            rusqlite::params![dir_owned, filename, trashed_at_ms],
                        )?;
                    }
                    RowEdit::MarkRestored { filename } => {
                        conn.execute(
                            "UPDATE generations SET trashed_at_ms = NULL
                             WHERE output_dir = ?1 AND filename = ?2",
                            rusqlite::params![dir_owned, filename],
                        )?;
                    }
                }
            }
            for import in &trash_imports {
                let id = upsert_with_conn(conn, &import.record)?;
                crate::organization::attach_tags(conn, id, &import.tags, now)?;
                for slug in &import.collection_slugs {
                    // Only re-home into collections that still exist — a
                    // slug alone cannot invent a display name.
                    if let Some(cid) = crate::organization::collection_id_for_slug(conn, slug)? {
                        crate::organization::collection_add_ids(conn, &cid, &[id], now)?;
                    }
                }
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
        title: None,
        favorite: false,
        trashed_at_ms: None,
    }
}

/// Build the row for a tombstoned file: the tombstone's metadata when it
/// parses, else whatever the bytes / filename yield; the tombstone's title,
/// favorite, and trash timestamp; tags + collection slugs carried along for
/// the transaction to apply.
fn build_trash_import(
    output_dir: &Path,
    filename: &str,
    format: mold_core::OutputFormat,
    path: &Path,
    mtime_ms: Option<i64>,
    size_bytes: Option<i64>,
    tombstone: Tombstone,
) -> TrashImport {
    let mut record =
        build_backfill_record(output_dir, filename, format, path, mtime_ms, size_bytes);
    if let Some(meta) = tombstone
        .metadata_json
        .as_deref()
        .and_then(|json| serde_json::from_str::<mold_core::OutputMetadata>(json).ok())
    {
        record.metadata = meta;
        record.metadata_synthetic = false;
    }
    record.title = tombstone.title;
    record.favorite = tombstone.favorite;
    record.trashed_at_ms = Some(tombstone.trashed_at_ms);
    // Tombstones are normally written from normalized state, but they are
    // plain JSON on disk; re-normalize so a hand-edited sidecar cannot
    // smuggle an invalid tag past the rules, dropping only the bad ones.
    let tags = tombstone
        .tags
        .iter()
        .filter_map(|t| crate::organization::normalize_tag_name(t).ok().flatten())
        .collect();
    TrashImport {
        record,
        tags,
        collection_slugs: tombstone.collections,
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

    // ---- trash awareness ---------------------------------------------------

    fn move_to_trash(dir: &Path, filename: &str) -> PathBuf {
        let trash = trash_dir(dir);
        std::fs::create_dir_all(&trash).unwrap();
        let dest = trash.join(filename);
        std::fs::rename(dir.join(filename), &dest).unwrap();
        dest
    }

    fn tombstone_for(dir: &Path, filename: &str, trashed_at_ms: i64) -> Tombstone {
        Tombstone {
            version: crate::trash::TOMBSTONE_VERSION,
            filename: filename.to_string(),
            trashed_at_ms,
            original_dir: canonical_dir_string(dir),
            title: None,
            favorite: false,
            tags: Vec::new(),
            collections: Vec::new(),
            metadata_json: None,
        }
    }

    /// A trashed row whose bytes sit in `.trash/` survives reconcile; the
    /// depth-1 live walk never sees the trash directory's contents.
    #[test]
    fn reconcile_keeps_trashed_rows_whose_bytes_are_in_trash() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("a.png"));
        write_valid_png(&tmp.path().join("b.png"));
        let db = MetadataDb::open_in_memory().unwrap();
        assert_eq!(db.reconcile(tmp.path()).unwrap().imported, 2);

        move_to_trash(tmp.path(), "a.png");
        crate::trash::write_tombstone(
            &trash_dir(tmp.path()),
            &tombstone_for(tmp.path(), "a.png", 4242),
        )
        .unwrap();
        assert!(db.mark_trashed(tmp.path(), "a.png", 4242).unwrap());

        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.trashed_kept, 1, "{stats:?}");
        assert_eq!(stats.removed, 0, "{stats:?}");
        assert_eq!(stats.kept, 1, "{stats:?}");
        assert_eq!(stats.imported, 0, "{stats:?}");
        assert_eq!(stats.trashed_imported, 0, "{stats:?}");
        let row = db.get(tmp.path(), "a.png").unwrap().unwrap();
        assert_eq!(row.trashed_at_ms, Some(4242));
        assert_eq!(db.list_live(Some(tmp.path())).unwrap().len(), 1);
        assert_eq!(db.list_trashed(Some(tmp.path())).unwrap().len(), 1);
        // Running it again is stable.
        let again = db.reconcile(tmp.path()).unwrap();
        assert_eq!(again.trashed_kept, 1);
        assert_eq!(db.count().unwrap(), 2);
    }

    /// A trashed row whose bytes are gone from both the live dir and
    /// `.trash/` is dropped like any other missing file.
    #[test]
    fn reconcile_drops_trashed_rows_missing_from_both_places() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("a.png"));
        let db = MetadataDb::open_in_memory().unwrap();
        db.reconcile(tmp.path()).unwrap();
        assert!(db.mark_trashed(tmp.path(), "a.png", 1).unwrap());
        std::fs::remove_file(tmp.path().join("a.png")).unwrap();

        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.removed, 1, "{stats:?}");
        assert_eq!(stats.trashed_kept, 0, "{stats:?}");
        assert_eq!(db.count().unwrap(), 0);
    }

    /// DB-loss recovery: the row is live (e.g. restored from a snapshot
    /// taken before the trash op) but the bytes + tombstone sit in
    /// `.trash/`. The row is re-flagged with the tombstone's timestamp
    /// instead of being deleted.
    #[test]
    fn reconcile_reflags_live_row_whose_file_moved_to_trash_from_tombstone() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("a.png"));
        let db = MetadataDb::open_in_memory().unwrap();
        db.reconcile(tmp.path()).unwrap();
        db.set_title(tmp.path(), "a.png", Some("Keep me")).unwrap();

        move_to_trash(tmp.path(), "a.png");
        crate::trash::write_tombstone(
            &trash_dir(tmp.path()),
            &tombstone_for(tmp.path(), "a.png", 9_001),
        )
        .unwrap();

        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.trashed_kept, 1, "{stats:?}");
        assert_eq!(stats.removed, 0, "{stats:?}");
        let row = db.get(tmp.path(), "a.png").unwrap().unwrap();
        assert_eq!(row.trashed_at_ms, Some(9_001));
        assert_eq!(
            row.title.as_deref(),
            Some("Keep me"),
            "row was kept, not rebuilt"
        );
    }

    /// Same recovery without a tombstone: the bytes in `.trash/` are still
    /// evidence of a trash op, so the row is flagged (timestamp falls back
    /// to the file's mtime) rather than dropped.
    #[test]
    fn reconcile_reflags_live_row_from_trash_bytes_without_tombstone() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("a.png"));
        let db = MetadataDb::open_in_memory().unwrap();
        db.reconcile(tmp.path()).unwrap();
        move_to_trash(tmp.path(), "a.png");

        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.trashed_kept, 1, "{stats:?}");
        assert_eq!(stats.removed, 0, "{stats:?}");
        let row = db.get(tmp.path(), "a.png").unwrap().unwrap();
        assert!(row.trashed_at_ms.is_some_and(|t| t > 1_600_000_000_000));
    }

    /// A tombstoned file in `.trash/` with no row (fresh DB) is imported as
    /// a trashed Backfill row with its organization restored; a file in
    /// `.trash/` without a tombstone is left alone.
    #[test]
    fn reconcile_imports_tombstoned_trash_files_without_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let trash = trash_dir(tmp.path());
        std::fs::create_dir_all(&trash).unwrap();
        write_valid_png(&trash.join("mold-flux-dev-1700000000000.png"));
        write_valid_png(&trash.join("orphan.png"));
        write_valid_png(&tmp.path().join("live.png"));

        let db = MetadataDb::open_in_memory().unwrap();
        let shelf = db.create_collection("Shelf", None).unwrap();
        let meta = crate::metadata_io::synthesize_from_filename("mold-flux-dev-1.png", 0);
        let mut meta = meta;
        meta.prompt = "from the tombstone".into();
        let mut tomb = tombstone_for(tmp.path(), "mold-flux-dev-1700000000000.png", 555);
        tomb.title = Some("Tombstoned owl".into());
        tomb.favorite = true;
        tomb.tags = vec!["Birds".into(), "night".into()];
        tomb.collections = vec!["shelf".into(), "no-such-collection".into()];
        tomb.metadata_json = Some(serde_json::to_string(&meta).unwrap());
        crate::trash::write_tombstone(&trash, &tomb).unwrap();

        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(
            stats.imported, 1,
            "only live.png is a live import: {stats:?}"
        );
        assert_eq!(stats.trashed_imported, 1, "{stats:?}");
        assert_eq!(stats.removed, 0, "{stats:?}");
        assert_eq!(
            db.count().unwrap(),
            2,
            "orphan.png has no tombstone → not imported"
        );

        let row = db
            .get(tmp.path(), "mold-flux-dev-1700000000000.png")
            .unwrap()
            .unwrap();
        assert_eq!(row.source, RecordSource::Backfill);
        assert_eq!(row.trashed_at_ms, Some(555));
        assert_eq!(row.title.as_deref(), Some("Tombstoned owl"));
        assert!(row.favorite);
        assert_eq!(row.metadata.prompt, "from the tombstone");
        assert!(!row.metadata_synthetic);
        let org = db
            .print_organization(tmp.path(), "mold-flux-dev-1700000000000.png")
            .unwrap()
            .unwrap();
        assert_eq!(org.tags, vec!["Birds", "night"]);
        assert_eq!(org.collections, vec![shelf.id.clone()]);
        assert_eq!(
            db.list_collections().unwrap().len(),
            1,
            "slugs never invent collections"
        );

        let live: Vec<_> = db
            .list_live(Some(tmp.path()))
            .unwrap()
            .into_iter()
            .map(|r| r.filename)
            .collect();
        assert_eq!(live, vec!["live.png"], "trash contents are never live");

        // Stable on a second pass.
        let again = db.reconcile(tmp.path()).unwrap();
        assert_eq!(again.trashed_imported, 0);
        assert_eq!(again.trashed_kept, 1);
        assert_eq!(db.count().unwrap(), 2);
    }

    /// A tombstone whose metadata_json is missing or unparsable still
    /// imports — metadata is synthesized from the trashed bytes.
    #[test]
    fn reconcile_trash_import_synthesizes_metadata_when_tombstone_has_none() {
        let tmp = tempfile::tempdir().unwrap();
        let trash = trash_dir(tmp.path());
        std::fs::create_dir_all(&trash).unwrap();
        write_valid_png(&trash.join("mold-zimage-1700000000000.png"));
        let mut tomb = tombstone_for(tmp.path(), "mold-zimage-1700000000000.png", 7);
        tomb.metadata_json = Some("{not json".into());
        crate::trash::write_tombstone(&trash, &tomb).unwrap();

        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.trashed_imported, 1, "{stats:?}");
        let row = db
            .get(tmp.path(), "mold-zimage-1700000000000.png")
            .unwrap()
            .unwrap();
        assert!(row.metadata_synthetic);
        assert_eq!(row.metadata.model, "zimage");
        assert_eq!(row.metadata.width, 64);
        assert_eq!(row.trashed_at_ms, Some(7));
    }

    /// A trashed row whose bytes are back in the live dir (manual restore
    /// with a file manager) is un-trashed rather than hidden forever.
    #[test]
    fn reconcile_untrashes_row_whose_file_is_live_again() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("a.png"));
        let db = MetadataDb::open_in_memory().unwrap();
        db.reconcile(tmp.path()).unwrap();
        assert!(db.mark_trashed(tmp.path(), "a.png", 1).unwrap());

        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.trashed_restored, 1, "{stats:?}");
        assert_eq!(stats.kept, 1, "{stats:?}");
        assert!(db.get(tmp.path(), "a.png").unwrap().unwrap().is_live());
    }

    /// The live walk is depth 1: nothing under `.trash/` (or any other
    /// subdirectory) is imported as a live print, tombstone or not.
    #[test]
    fn reconcile_live_walk_never_imports_trash_or_subdirectories() {
        let tmp = tempfile::tempdir().unwrap();
        let trash = trash_dir(tmp.path());
        std::fs::create_dir_all(&trash).unwrap();
        write_valid_png(&trash.join("no-tombstone.png"));
        let nested = tmp.path().join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        write_valid_png(&nested.join("deep.png"));

        let db = MetadataDb::open_in_memory().unwrap();
        let stats = db.reconcile(tmp.path()).unwrap();
        assert_eq!(stats.imported, 0, "{stats:?}");
        assert_eq!(stats.trashed_imported, 0, "{stats:?}");
        assert_eq!(db.count().unwrap(), 0);
    }
}
