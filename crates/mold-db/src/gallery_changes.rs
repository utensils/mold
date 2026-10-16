//! The gallery change log (schema v43): a per-directory DIRTY SET of gallery
//! filenames, each carrying the sequence number of its latest change.
//!
//! `gallery_changes(output_dir, filename, seq, changed_at_ms)` holds ONE row
//! per name — the latest change wins — and `gallery_feed_meta` holds the
//! single counter row (`epoch`, `next_seq`, `min_seq`,
//! `effective_retention_days`). Every writer is caught by SQLite triggers
//! installed in migration v43, so the server, the CLI's `record_local_save`,
//! reconcile, trash, and organization edits all mark changes without knowing
//! the log exists:
//!
//! - `generations` INSERT / DELETE, and UPDATE guarded on the
//!   [`GENERATIONS_VISIBLE_COLUMNS`] (a no-op `upsert_with_conn` always takes
//!   `ON CONFLICT DO UPDATE`; without the guard every refresh would mark).
//!   A rename of `filename`/`output_dir` marks both the old and the new name.
//! - `generation_tags` / `collection_items` INSERT / DELETE / UPDATE mark the
//!   parent print while its row still exists (a cascade from a print delete
//!   is already marked by the `generations` DELETE trigger).
//! - `tags` UPDATE OF `name` (case-sensitive) fans out to every member.
//! - `settings` writes of `gallery.trash_retention_days` fan out to every
//!   trashed row, because `purge_at` is derived from it.
//!
//! There is deliberately NO trigger on `generation_assets` /
//! `generation_asset_scans`: assets are indexed lazily while a listing is
//! read, so a trigger there would make feed reads write changes and a
//! snapshot would never settle. Collection renames need none either: rows
//! carry collection IDS, never names.
//!
//! There is no `removed` column. Whether a dirty name is a removal is decided
//! when the feed is read, because the committed-archive overlay can keep a
//! name visible after its DB row is gone.
//!
//! Sequence numbers are global across directories and strictly increasing;
//! SQLite's single writer makes them visible in commit order. `epoch` is
//! minted by the migration, so a recreated database (corruption recovery
//! rebuilds the schema through migrations) always answers a new one.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use rusqlite::{params, Connection};

use crate::path::canonical_dir_string;
use crate::MetadataDb;

/// `generations` columns whose change is visible in a gallery row. The v43
/// UPDATE trigger compares exactly these (`NEW.c IS NOT OLD.c`).
///
/// Adding a column to `generations` makes
/// `every_generations_column_is_classified` fail until it is listed here or
/// in [`GENERATIONS_EXCLUDED_COLUMNS`]. A new VISIBLE column also needs a
/// migration that drops and recreates `gallery_changes_generations_update`
/// with the new guard (`installed_update_guard_covers_every_visible_column`
/// enforces it).
pub const GENERATIONS_VISIBLE_COLUMNS: &[&str] = &[
    "filename",
    "output_dir",
    "created_at_ms",
    "file_mtime_ms",
    "file_size_bytes",
    "format",
    "model",
    "prompt",
    "negative_prompt",
    "original_prompt",
    "seed",
    "steps",
    "guidance",
    "width",
    "height",
    "strength",
    "scheduler",
    "lora",
    "lora_scale",
    "frames",
    "fps",
    "metadata_version",
    "generation_time_ms",
    "backend",
    "hostname",
    "source",
    "metadata_synthetic",
    "metadata_json",
    "title",
    "favorite",
    "trashed_at_ms",
];

/// `generations` columns that never change a gallery row: the row id, and
/// the v25 queue-job projection that `generations_queue_job_projection_dirty`
/// and every upsert rewrite.
pub const GENERATIONS_EXCLUDED_COLUMNS: &[&str] =
    &["id", "queue_job_id", "queue_job_metadata_state"];

/// Tombstone horizon used when trash retention is `0` (keep forever): a
/// permanently-deleted print's dirty row must still be collected eventually,
/// and a client offline that long simply resets.
pub const DEFAULT_TOMBSTONE_HORIZON_DAYS: u32 = 30;

/// Where the feed stands right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedState {
    /// Minted per database. A different epoch means the client's cursor
    /// belongs to another database (recreated or replaced) and must reset.
    pub epoch: String,
    /// A cursor below this has missed a collected tombstone and must reset.
    pub min_seq: i64,
    /// The highest sequence ever assigned (global across directories; it
    /// never decreases, so a cursor above it came from a newer database).
    pub max_seq: i64,
}

/// What [`gc_tombstones`] collected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GcOutcome {
    pub deleted: u64,
    /// `min_seq` after the collection.
    pub min_seq: i64,
}

/// The tombstone GC cutoff: dirty rows last changed before this are
/// eligible. `retention_days == 0` uses [`DEFAULT_TOMBSTONE_HORIZON_DAYS`].
pub fn tombstone_cutoff_ms(retention_days: u32, now_ms: i64) -> i64 {
    let days = if retention_days == 0 {
        DEFAULT_TOMBSTONE_HORIZON_DAYS
    } else {
        retention_days
    };
    now_ms.saturating_sub(i64::from(days) * 86_400_000)
}

/// The feed's epoch, collection floor, and highest assigned sequence.
pub fn feed_state(db: &MetadataDb) -> Result<FeedState> {
    db.with_conn(|conn| {
        Ok(conn.query_row(
            "SELECT epoch, min_seq, next_seq - 1 FROM gallery_feed_meta WHERE singleton = 1",
            [],
            |r| {
                Ok(FeedState {
                    epoch: r.get(0)?,
                    min_seq: r.get(1)?,
                    max_seq: r.get(2)?,
                })
            },
        )?)
    })
}

/// Dirty names in `output_dir` changed after `since_seq`, oldest change
/// first, at most `limit`. Keyset paging: pass the last returned seq as the
/// next `since_seq`. Seqs are unique, so a page boundary never splits one.
pub fn changes_since(
    db: &MetadataDb,
    output_dir: &Path,
    since_seq: i64,
    limit: usize,
) -> Result<Vec<(String, i64)>> {
    let dir_key = canonical_dir_string(output_dir);
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT filename, seq FROM gallery_changes
             WHERE output_dir = ?1 AND seq > ?2
             ORDER BY seq ASC LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![dir_key, since_seq, limit], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    })
}

/// Mark `filenames` in `output_dir` dirty with fresh seqs. This is the
/// explicit marker for changes no trigger can see (the committed-archive
/// projector). Duplicates are marked once. Returns how many were marked.
pub fn touch<S: AsRef<str>>(db: &MetadataDb, output_dir: &Path, filenames: &[S]) -> Result<u64> {
    let dir_key = canonical_dir_string(output_dir);
    db.transact_immediate(|conn| mark_names(conn, &dir_key, filenames))
}

/// Re-mark every name the caller can see — the fallback when the archive
/// projector cannot prove which names changed (a generation missing from its
/// log). Same semantics as [`touch`], kept separate so the O(n) call sites
/// are greppable.
pub fn touch_all<S: AsRef<str>>(
    db: &MetadataDb,
    output_dir: &Path,
    filenames: &[S],
) -> Result<u64> {
    touch(db, output_dir, filenames)
}

/// Collect tombstones: dirty rows in `output_dir` whose name no longer has a
/// `generations` row, is not in `keep`, and was last changed before
/// `older_than_ms`. In the SAME transaction `min_seq` is raised to the
/// highest collected seq, so a cursor that might have missed one resets.
///
/// `keep` is REQUIRED to hold every name still visible from the committed
/// archive: an archive-only print has no DB row, so without it the GC would
/// drop the dirty row of a print that still exists.
pub fn gc_tombstones(
    db: &MetadataDb,
    output_dir: &Path,
    keep: &HashSet<String>,
    older_than_ms: i64,
) -> Result<GcOutcome> {
    let dir_key = canonical_dir_string(output_dir);
    db.transact_immediate(|conn| {
        let candidates: Vec<(String, i64)> = {
            let mut stmt = conn.prepare(
                "SELECT c.filename, c.seq FROM gallery_changes c
                 WHERE c.output_dir = ?1 AND c.changed_at_ms < ?2
                   AND NOT EXISTS (
                       SELECT 1 FROM generations g
                        WHERE g.output_dir = c.output_dir AND g.filename = c.filename
                   )",
            )?;
            let rows = stmt
                .query_map(params![dir_key, older_than_ms], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        let mut deleted = 0u64;
        let mut max_deleted: Option<i64> = None;
        for (filename, seq) in candidates {
            if keep.contains(&filename) {
                continue;
            }
            deleted += conn.execute(
                "DELETE FROM gallery_changes WHERE output_dir = ?1 AND filename = ?2 AND seq = ?3",
                params![dir_key, filename, seq],
            )? as u64;
            max_deleted = Some(max_deleted.map_or(seq, |m| m.max(seq)));
        }
        if let Some(seq) = max_deleted {
            conn.execute(
                "UPDATE gallery_feed_meta SET min_seq = MAX(min_seq, ?1) WHERE singleton = 1",
                params![seq],
            )?;
        }
        let min_seq = conn.query_row(
            "SELECT min_seq FROM gallery_feed_meta WHERE singleton = 1",
            [],
            |r| r.get(0),
        )?;
        Ok(GcOutcome { deleted, min_seq })
    })
}

/// Record the trash retention actually in force (env override included).
/// When it differs from the stored one every trashed row is marked, because
/// each one's `purge_at` moved. The `settings` trigger already covers DB
/// writes; this covers `MOLD_GALLERY_TRASH_RETENTION_DAYS` changing across a
/// restart. Returns whether the stored value changed.
pub fn record_effective_retention(db: &MetadataDb, days: u32) -> Result<bool> {
    db.transact_immediate(|conn| {
        let stored: Option<i64> = conn.query_row(
            "SELECT effective_retention_days FROM gallery_feed_meta WHERE singleton = 1",
            [],
            |r| r.get(0),
        )?;
        if stored == Some(i64::from(days)) {
            return Ok(false);
        }
        conn.execute(
            "UPDATE gallery_feed_meta SET effective_retention_days = ?1 WHERE singleton = 1",
            params![i64::from(days)],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO gallery_changes (output_dir, filename, seq, changed_at_ms)
             SELECT g.output_dir, g.filename,
                    (SELECT next_seq FROM gallery_feed_meta WHERE singleton = 1)
                      + ROW_NUMBER() OVER (ORDER BY g.id) - 1,
                    CAST(strftime('%s','now') AS INTEGER) * 1000
               FROM generations g WHERE g.trashed_at_ms IS NOT NULL",
            [],
        )?;
        conn.execute(
            "UPDATE gallery_feed_meta
                SET next_seq = next_seq
                    + (SELECT COUNT(*) FROM generations WHERE trashed_at_ms IS NOT NULL)
              WHERE singleton = 1",
            [],
        )?;
        Ok(true)
    })
}

fn mark_names<S: AsRef<str>>(conn: &Connection, dir_key: &str, filenames: &[S]) -> Result<u64> {
    let mut seen = HashSet::new();
    let mut marked = 0u64;
    for name in filenames {
        let name = name.as_ref();
        if !seen.insert(name) {
            continue;
        }
        conn.execute(
            "INSERT OR REPLACE INTO gallery_changes (output_dir, filename, seq, changed_at_ms)
             VALUES (?1, ?2, (SELECT next_seq FROM gallery_feed_meta WHERE singleton = 1),
                     CAST(strftime('%s','now') AS INTEGER) * 1000)",
            params![dir_key, name],
        )?;
        conn.execute(
            "UPDATE gallery_feed_meta SET next_seq = next_seq + 1 WHERE singleton = 1",
            [],
        )?;
        marked += 1;
    }
    Ok(marked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{GenerationRecord, RecordSource};
    use crate::Settings;
    use mold_core::{OutputFormat, OutputMetadata};

    const DIR: &str = "/tmp/gallery-changes-tests";

    fn dir() -> &'static Path {
        Path::new(DIR)
    }

    fn meta() -> OutputMetadata {
        let req: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"an owl","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        OutputMetadata::from_generate_request(&req, 1, None, "test")
    }

    fn record(name: &str) -> GenerationRecord {
        GenerationRecord::from_save(
            dir(),
            name,
            OutputFormat::Png,
            meta(),
            RecordSource::Server,
            1,
        )
    }

    fn seed(db: &MetadataDb, names: &[&str]) {
        for name in names {
            db.upsert(&record(name)).unwrap();
        }
    }

    fn all(db: &MetadataDb) -> Vec<(String, i64)> {
        changes_since(db, dir(), 0, 10_000).unwrap()
    }

    fn seq_of(db: &MetadataDb, name: &str) -> Option<i64> {
        all(db).into_iter().find(|(n, _)| n == name).map(|(_, s)| s)
    }

    fn max_seq(db: &MetadataDb) -> i64 {
        feed_state(db).unwrap().max_seq
    }

    /// Run `f` and report whether it assigned any new seq.
    fn marks(db: &MetadataDb, f: impl FnOnce()) -> bool {
        let before = max_seq(db);
        f();
        max_seq(db) != before
    }

    fn names_after(db: &MetadataDb, since: i64) -> Vec<String> {
        let mut names: Vec<String> = changes_since(db, dir(), since, 10_000)
            .unwrap()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        names.sort();
        names
    }

    #[test]
    fn upgrade_backfills_every_existing_row_with_a_distinct_seq() {
        use crate::migrations::{apply_pending, MigrationKind, MIGRATIONS};
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON").unwrap();
        for migration in MIGRATIONS.iter().filter(|m| m.version <= 42) {
            let tx = conn.transaction().unwrap();
            match migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(f) => f(&tx).unwrap(),
            }
            tx.execute_batch(&format!("PRAGMA user_version = {};", migration.version))
                .unwrap();
            tx.commit().unwrap();
        }
        for (dir, name) in [("/a", "1.png"), ("/a", "2.png"), ("/b", "3.png")] {
            conn.execute(
                "INSERT INTO generations (filename, output_dir, created_at_ms, format, model)
                 VALUES (?1, ?2, 1, 'png', 'flux-dev:q4')",
                params![name, dir],
            )
            .unwrap();
        }
        apply_pending(&mut conn).unwrap();
        let rows: Vec<(String, String, i64)> = conn
            .prepare("SELECT output_dir, filename, seq FROM gallery_changes ORDER BY seq")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            rows,
            vec![
                ("/a".into(), "1.png".into(), 1),
                ("/a".into(), "2.png".into(), 2),
                ("/b".into(), "3.png".into(), 3),
            ]
        );
        let (next, min): (i64, i64) = conn
            .query_row(
                "SELECT next_seq, min_seq FROM gallery_feed_meta WHERE singleton = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!((next, min), (4, 0));
    }

    #[test]
    fn insert_marks_the_name() {
        let db = MetadataDb::open_in_memory().unwrap();
        assert!(all(&db).is_empty());
        seed(&db, &["a.png"]);
        assert_eq!(names_after(&db, 0), vec!["a.png"]);
    }

    #[test]
    fn no_op_upsert_does_not_mark() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        assert!(!marks(&db, || {
            db.upsert(&record("a.png")).unwrap();
        }));
    }

    #[test]
    fn visible_field_update_marks() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        let before = max_seq(&db);
        let mut changed = record("a.png");
        changed.file_size_bytes = Some(999);
        db.upsert(&changed).unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png"]);
    }

    #[test]
    fn queue_job_projection_only_update_does_not_mark() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        assert!(!marks(&db, || {
            db.with_conn(|conn| {
                conn.execute(
                    "UPDATE generations SET queue_job_id = 'job-x', queue_job_metadata_state = 0
                     WHERE filename = 'a.png'",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        }));
    }

    #[test]
    fn organize_title_and_favorite_mark() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png"]);
        let before = max_seq(&db);
        db.set_title(dir(), "a.png", Some("Owl")).unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png"]);
        let before = max_seq(&db);
        db.set_favorite(dir(), "b.png", true).unwrap();
        assert_eq!(names_after(&db, before), vec!["b.png"]);
        // Re-applying the same value is not a change.
        assert!(!marks(&db, || db
            .set_favorite(dir(), "b.png", true)
            .unwrap()));
    }

    #[test]
    fn tag_add_and_remove_mark_the_print() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png"]);
        let before = max_seq(&db);
        db.add_tags(dir(), "a.png", &["owls".into()]).unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png"]);
        let before = max_seq(&db);
        db.remove_tags(dir(), "a.png", &["owls".into()]).unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png"]);
    }

    #[test]
    fn tag_rename_fans_out_to_every_member_and_case_counts() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png", "c.png"]);
        db.add_tags(dir(), "a.png", &["owls".into()]).unwrap();
        db.add_tags(dir(), "b.png", &["owls".into()]).unwrap();
        let before = max_seq(&db);
        db.rename_tag("owls", "birds").unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png", "b.png"]);
        let before = max_seq(&db);
        db.rename_tag("birds", "Birds").unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png", "b.png"]);
        assert_eq!(max_seq(&db), before + 2, "one distinct seq per member");
        // A delete of the tag cascades to members, which marks them too.
        let before = max_seq(&db);
        db.delete_tag("Birds").unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png", "b.png"]);
    }

    #[test]
    fn collection_item_add_and_remove_mark_the_print() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png"]);
        let collection = db.create_collection("Birds", None).unwrap();
        let before = max_seq(&db);
        db.collection_add(&collection.id, dir(), &["a.png".into()])
            .unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png"]);
        // A rename of the collection changes no print row (rows carry ids).
        assert!(!marks(&db, || {
            db.update_collection(&collection.id, Some("Raptors"), None, None, None)
                .unwrap();
        }));
        let before = max_seq(&db);
        db.collection_remove(&collection.id, dir(), &["a.png".into()])
            .unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png"]);
    }

    #[test]
    fn trash_and_restore_mark() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        let before = max_seq(&db);
        assert!(db.mark_trashed(dir(), "a.png", 5_000).unwrap());
        assert_eq!(names_after(&db, before), vec!["a.png"]);
        let before = max_seq(&db);
        assert!(db.mark_restored(dir(), "a.png").unwrap());
        assert_eq!(names_after(&db, before), vec!["a.png"]);
    }

    #[test]
    fn permanent_delete_leaves_a_tombstone_and_cascades_cleanly() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        db.add_tags(dir(), "a.png", &["owls".into(), "night".into()])
            .unwrap();
        let collection = db.create_collection("Birds", None).unwrap();
        db.collection_add(&collection.id, dir(), &["a.png".into()])
            .unwrap();
        let before = max_seq(&db);
        assert!(db.delete(dir(), "a.png").unwrap());
        assert_eq!(db.count().unwrap(), 0);
        assert_eq!(names_after(&db, before), vec!["a.png"]);
        assert_eq!(
            max_seq(&db),
            before + 1,
            "cascaded child deletes must not re-mark a print that is gone"
        );
    }

    #[test]
    fn rename_of_filename_marks_old_and_new() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        let before = max_seq(&db);
        db.with_conn(|conn| {
            conn.execute(
                "UPDATE generations SET filename = 'b.png' WHERE filename = 'a.png'",
                [],
            )?;
            Ok(())
        })
        .unwrap();
        assert_eq!(names_after(&db, before), vec!["a.png", "b.png"]);
    }

    #[test]
    fn reconcile_of_unchanged_files_marks_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        for name in ["a.png", "b.png"] {
            // A checkerboard: tiny or solid images are skipped as suspect.
            image::RgbImage::from_fn(64, 64, |x, y| {
                if ((x / 8) + (y / 8)) % 2 == 0 {
                    image::Rgb([255, 64, 32])
                } else {
                    image::Rgb([16, 200, 240])
                }
            })
            .save(tmp.path().join(name))
            .unwrap();
        }
        let db = MetadataDb::open_in_memory().unwrap();
        db.reconcile(tmp.path()).unwrap();
        let first = changes_since(&db, tmp.path(), 0, 100).unwrap();
        assert_eq!(first.len(), 2);
        assert!(!marks(&db, || {
            db.reconcile(tmp.path()).unwrap();
        }));
    }

    #[test]
    fn retention_setting_change_marks_trashed_rows_only() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["live.png", "trashed.png"]);
        db.mark_trashed(dir(), "trashed.png", 1).unwrap();
        let settings = Settings::new(&db);
        let key = mold_core::config_keys::GALLERY_TRASH_RETENTION_DAYS_KEY;
        let before = max_seq(&db);
        settings.set_int(key, 7).unwrap();
        assert_eq!(names_after(&db, before), vec!["trashed.png"]);
        let before = max_seq(&db);
        settings.set_int(key, 9).unwrap();
        assert_eq!(names_after(&db, before), vec!["trashed.png"]);
        // Same value again: not a change.
        assert!(!marks(&db, || settings.set_int(key, 9).unwrap()));
        let before = max_seq(&db);
        settings.delete(key).unwrap();
        assert_eq!(names_after(&db, before), vec!["trashed.png"]);
        // Other keys never mark.
        assert!(!marks(&db, || settings
            .set_str("generate.last_model", "flux-dev:q4")
            .unwrap()));
    }

    #[test]
    fn effective_retention_marks_trashed_rows_when_it_moves() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["live.png", "trashed.png"]);
        db.mark_trashed(dir(), "trashed.png", 1).unwrap();
        let before = max_seq(&db);
        assert!(record_effective_retention(&db, 30).unwrap());
        assert_eq!(names_after(&db, before), vec!["trashed.png"]);
        assert!(!marks(&db, || {
            assert!(!record_effective_retention(&db, 30).unwrap());
        }));
        let before = max_seq(&db);
        assert!(record_effective_retention(&db, 1).unwrap());
        assert_eq!(names_after(&db, before), vec!["trashed.png"]);
    }

    #[test]
    fn epoch_differs_after_recreate() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mold.db");
        let first = feed_state(&MetadataDb::open(&path).unwrap()).unwrap().epoch;
        assert_eq!(first.len(), 32);
        // Reopening keeps the epoch.
        assert_eq!(
            feed_state(&MetadataDb::open(&path).unwrap()).unwrap().epoch,
            first
        );
        // Corrupt the file: open-time recovery quarantines and recreates it.
        for suffix in ["-wal", "-shm"] {
            let _ = std::fs::remove_file(tmp.path().join(format!("mold.db{suffix}")));
        }
        std::fs::write(&path, b"this is not a sqlite database").unwrap();
        let recovered = MetadataDb::open(&path).unwrap();
        assert_ne!(feed_state(&recovered).unwrap().epoch, first);
        // In-memory databases mint their own too.
        let a = feed_state(&MetadataDb::open_in_memory().unwrap()).unwrap();
        let b = feed_state(&MetadataDb::open_in_memory().unwrap()).unwrap();
        assert_ne!(a.epoch, b.epoch);
    }

    #[test]
    fn epoch_differs_after_query_time_rebuild() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        let first = feed_state(&db).unwrap().epoch;
        db.with_conn(|conn| {
            conn.execute_batch(
                "PRAGMA writable_schema = ON;
                 UPDATE sqlite_schema SET rootpage = 2147483647
                   WHERE name = 'idx_gen_mtime';
                 PRAGMA writable_schema = OFF;
                 PRAGMA schema_version = 9999;",
            )?;
            Ok(())
        })
        .unwrap();
        db.list(Some(tmp.path())).unwrap();
        assert_ne!(feed_state(&db).unwrap().epoch, first);
    }

    #[test]
    fn seqs_are_unique_and_strictly_increasing() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png", "c.png"]);
        db.set_title(dir(), "a.png", Some("x")).unwrap();
        touch(&db, dir(), &["d.png", "d.png", "e.png"]).unwrap();
        let rows = all(&db);
        let seqs: Vec<i64> = rows.iter().map(|(_, s)| *s).collect();
        assert!(seqs.windows(2).all(|w| w[0] < w[1]), "{rows:?}");
        assert_eq!(rows.last().unwrap().0, "e.png");
        assert_eq!(*seqs.last().unwrap(), max_seq(&db));
        // `a.png` moved to the end of the log ahead of the touches.
        assert!(seq_of(&db, "a.png").unwrap() > seq_of(&db, "c.png").unwrap());
    }

    #[test]
    fn changes_since_pages_by_seq() {
        let db = MetadataDb::open_in_memory().unwrap();
        let names: Vec<String> = (0..7).map(|i| format!("{i}.png")).collect();
        touch_all(&db, dir(), &names).unwrap();
        // Another directory never leaks in.
        touch(&db, Path::new("/tmp/other-dir"), &["x.png"]).unwrap();
        let mut since = 0;
        let mut seen = Vec::new();
        loop {
            let page = changes_since(&db, dir(), since, 3).unwrap();
            if page.is_empty() {
                break;
            }
            assert!(page.len() <= 3);
            since = page.last().unwrap().1;
            seen.extend(page.into_iter().map(|(n, _)| n));
        }
        assert_eq!(seen, names);
    }

    #[test]
    fn gc_collects_old_tombstones_raises_min_seq_and_respects_keep() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["live.png", "gone.png", "archive.png"]);
        db.delete(dir(), "gone.png").unwrap();
        db.delete(dir(), "archive.png").unwrap();
        let gone_seq = seq_of(&db, "gone.png").unwrap();
        let archive_seq = seq_of(&db, "archive.png").unwrap();
        assert!(archive_seq > gone_seq);
        let keep: HashSet<String> = ["archive.png".to_string()].into();

        // Nothing is old enough yet.
        let none = gc_tombstones(&db, dir(), &keep, 0).unwrap();
        assert_eq!(none.deleted, 0);
        assert_eq!(feed_state(&db).unwrap().min_seq, 0);

        let far_future = i64::MAX;
        let out = gc_tombstones(&db, dir(), &keep, far_future).unwrap();
        assert_eq!(out.deleted, 1);
        assert_eq!(out.min_seq, gone_seq);
        assert_eq!(feed_state(&db).unwrap().min_seq, gone_seq);
        let mut left = names_after(&db, 0);
        left.sort();
        assert_eq!(left, vec!["archive.png", "live.png"]);

        // Dropping the keep collects the archive name; min_seq only rises.
        let out = gc_tombstones(&db, dir(), &HashSet::new(), far_future).unwrap();
        assert_eq!(out.deleted, 1);
        assert_eq!(out.min_seq, archive_seq);
        let out = gc_tombstones(&db, dir(), &HashSet::new(), far_future).unwrap();
        assert_eq!(
            out,
            GcOutcome {
                deleted: 0,
                min_seq: archive_seq
            }
        );
        assert_eq!(names_after(&db, 0), vec!["live.png"]);
    }

    #[test]
    fn gc_and_min_seq_commit_atomically() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["gone.png"]);
        db.delete(dir(), "gone.png").unwrap();
        // Make the min_seq update fail: the delete must roll back with it.
        db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TRIGGER fail_min_seq BEFORE UPDATE OF min_seq ON gallery_feed_meta
                 BEGIN SELECT RAISE(ABORT, 'boom'); END;",
            )?;
            Ok(())
        })
        .unwrap();
        assert!(gc_tombstones(&db, dir(), &HashSet::new(), i64::MAX).is_err());
        assert_eq!(names_after(&db, 0), vec!["gone.png"]);
        assert_eq!(feed_state(&db).unwrap().min_seq, 0);
    }

    #[test]
    fn tombstone_cutoff_uses_retention_or_the_default_horizon() {
        let now = 100 * 86_400_000;
        assert_eq!(tombstone_cutoff_ms(7, now), 93 * 86_400_000);
        assert_eq!(tombstone_cutoff_ms(0, now), 70 * 86_400_000);
    }

    #[test]
    fn changes_since_uses_the_dir_seq_index() {
        let db = MetadataDb::open_in_memory().unwrap();
        let plan: Vec<String> = db
            .with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "EXPLAIN QUERY PLAN SELECT filename, seq FROM gallery_changes
                     WHERE output_dir = ?1 AND seq > ?2 ORDER BY seq ASC LIMIT ?3",
                )?;
                let rows = stmt
                    .query_map(params!["/out", 0_i64, 10_i64], |r| r.get::<_, String>(3))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows)
            })
            .unwrap();
        let joined = plan.join(" | ");
        assert!(
            joined.contains("idx_gallery_changes_dir_seq"),
            "changes_since must use the (output_dir, seq) index, got: {joined}"
        );
        assert!(!joined.contains("TEMP B-TREE"), "got: {joined}");
    }

    fn generations_columns(db: &MetadataDb) -> Vec<String> {
        db.with_conn(|conn| {
            let mut stmt = conn.prepare("PRAGMA table_info(generations)")?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(1))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
        .unwrap()
    }

    #[test]
    fn every_generations_column_is_classified() {
        let db = MetadataDb::open_in_memory().unwrap();
        let columns = generations_columns(&db);
        for column in &columns {
            let visible = GENERATIONS_VISIBLE_COLUMNS.contains(&column.as_str());
            let excluded = GENERATIONS_EXCLUDED_COLUMNS.contains(&column.as_str());
            assert!(
                visible ^ excluded,
                "generations.{column} must be in exactly one of GENERATIONS_VISIBLE_COLUMNS \
                 or GENERATIONS_EXCLUDED_COLUMNS (gallery_changes.rs) — decide whether a \
                 change to it is visible in a gallery row"
            );
        }
        for listed in GENERATIONS_VISIBLE_COLUMNS
            .iter()
            .chain(GENERATIONS_EXCLUDED_COLUMNS)
        {
            assert!(
                columns.iter().any(|c| c == listed),
                "{listed} is classified but is not a generations column"
            );
        }
    }

    #[test]
    fn installed_update_guard_covers_every_visible_column() {
        let db = MetadataDb::open_in_memory().unwrap();
        let sql: String = db
            .with_conn(|conn| {
                Ok(conn.query_row(
                    "SELECT sql FROM sqlite_schema
                     WHERE type = 'trigger' AND name = 'gallery_changes_generations_update'",
                    [],
                    |r| r.get(0),
                )?)
            })
            .unwrap();
        for column in GENERATIONS_VISIBLE_COLUMNS {
            assert!(
                sql.contains(&format!("NEW.{column} IS NOT OLD.{column}")),
                "the installed UPDATE guard does not compare {column}; recreate the trigger \
                 in a new migration"
            );
        }
        for column in GENERATIONS_EXCLUDED_COLUMNS {
            assert!(
                !sql.contains(&format!("NEW.{column} ")),
                "{column} must not be guarded"
            );
        }
    }
}
