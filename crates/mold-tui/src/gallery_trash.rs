//! Local "move to trash" for the TUI's disk-backed Library entries.
//!
//! A print the TUI read straight from the output directory has no server
//! to route its delete through, so the TUI performs the same steps the
//! server's `gallery_trash::trash_print_blocking` performs for its own
//! gallery, minus the publication gate it does not have:
//!
//! 1. make sure the print has a `generations` row (reconcile normally
//!    guarantees this; a file that landed between scans is imported from
//!    its embedded metadata exactly as the server does),
//! 2. rename `<output_dir>/<filename>` to `<output_dir>/.trash/<filename>`,
//! 3. write the tombstone next to it (`mold_db::trash::build_tombstone` +
//!    `write_tombstone`), and
//! 4. flag the row with `MetadataDb::mark_trashed`.
//!
//! The server, the desktop, and reconcile all agree on that layout, so a
//! print trashed here shows up in every other surface's Trash view and is
//! swept by the server's retention sweeper like any other.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use mold_db::MetadataDb;

/// Move the live print at `path` into its gallery's `.trash/` directory
/// and record it as trashed in `db`.
///
/// Idempotent for a print already flagged trashed (returns `Ok`). Fails
/// — leaving the live file untouched — when the path has no parent or
/// file name, the row cannot be established, or the rename itself fails.
pub(crate) fn trash_local_print(db: &MetadataDb, path: &Path, now_ms: i64) -> Result<()> {
    let output_dir = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .ok_or_else(|| anyhow!("{} has no gallery directory", path.display()))?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("{} has no usable file name", path.display()))?;

    let row = db
        .get(output_dir, filename)
        .with_context(|| format!("reading the gallery row for {filename}"))?;
    if row.as_ref().is_some_and(|r| r.trashed_at_ms.is_some()) {
        return Ok(());
    }
    if row.is_none() {
        if !path.is_file() {
            return Err(anyhow!("{} is not a gallery file", path.display()));
        }
        ensure_row_for_live_file(db, output_dir, filename, path)?;
    }

    let trash_dir = mold_db::trash::trash_dir(output_dir);
    if path.is_file() {
        std::fs::create_dir_all(&trash_dir)
            .with_context(|| format!("creating {}", trash_dir.display()))?;
        let trash_path = trash_dir.join(filename);
        std::fs::rename(path, &trash_path).with_context(|| {
            format!(
                "moving {} to the gallery trash at {}",
                path.display(),
                trash_path.display()
            )
        })?;
    }

    if let Some(tombstone) = db
        .build_tombstone(output_dir, filename, now_ms)
        .context("building the trash tombstone")?
    {
        mold_db::trash::write_tombstone(&trash_dir, &tombstone)
            .context("writing the trash tombstone")?;
    }
    db.mark_trashed(output_dir, filename, now_ms)
        .context("flagging the print as trashed")?;
    Ok(())
}

/// Import a live file that has no DB row yet so the trash index has a row
/// to flag — the TUI twin of the server's `ensure_row_for_live_file`.
fn ensure_row_for_live_file(
    db: &MetadataDb,
    output_dir: &Path,
    filename: &str,
    live_path: &Path,
) -> Result<()> {
    let format = mold_db::metadata_io::format_from_path(Path::new(filename))
        .ok_or_else(|| anyhow!("{filename} is not a gallery format"))?;
    let file_meta = std::fs::metadata(live_path).ok();
    let (mtime_ms, size_bytes) = file_meta
        .as_ref()
        .map(|m| {
            let size = Some(m.len() as i64);
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as i64);
            (mtime, size)
        })
        .unwrap_or((None, None));
    let timestamp_secs = mtime_ms.map(|ms| (ms / 1000).max(0)).unwrap_or(0) as u64;
    let (metadata, synthetic) =
        mold_db::metadata_io::read_or_synthesize(live_path, format, filename, timestamp_secs);
    let mut record = mold_db::GenerationRecord::from_save(
        output_dir,
        filename,
        format,
        metadata,
        mold_db::RecordSource::Backfill,
        mold_core::time::now_epoch_ms(),
    );
    record.file_mtime_ms = mtime_ms;
    record.file_size_bytes = size_bytes;
    record.metadata_synthetic = synthetic;
    db.upsert(&record)
        .with_context(|| format!("recording {filename} in the metadata DB before trashing it"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PNG that passes the shared gallery validity guards (size floor,
    /// decodable header, not solid black): noisy enough not to compress
    /// below the floor.
    fn write_png(path: &Path) {
        let img = image::RgbaImage::from_fn(64, 64, |x, y| {
            image::Rgba([
                (x * 37 % 251) as u8,
                (y * 91 % 241) as u8,
                ((x ^ y) * 13 % 233) as u8,
                255,
            ])
        });
        image::DynamicImage::ImageRgba8(img).save(path).unwrap();
    }

    fn open_db(dir: &Path) -> MetadataDb {
        MetadataDb::open(&dir.join("mold.db")).unwrap()
    }

    #[test]
    fn trash_moves_file_writes_tombstone_and_flags_row() {
        let tmp = tempfile::tempdir().unwrap();
        let gallery = tmp.path().join("output");
        std::fs::create_dir_all(&gallery).unwrap();
        let live = gallery.join("mold-test-0001.png");
        write_png(&live);
        let db = open_db(tmp.path());
        db.reconcile(&gallery).unwrap();
        assert!(db.get(&gallery, "mold-test-0001.png").unwrap().is_some());

        trash_local_print(&db, &live, 1_700_000_000_000).unwrap();

        assert!(!live.exists(), "live bytes move out of the gallery");
        let trash_dir = mold_db::trash::trash_dir(&gallery);
        assert!(trash_dir.join("mold-test-0001.png").is_file());
        let tombstone = mold_db::trash::read_tombstone(&mold_db::trash::tombstone_path(
            &trash_dir,
            "mold-test-0001.png",
        ))
        .unwrap();
        assert_eq!(tombstone.filename, "mold-test-0001.png");
        assert_eq!(tombstone.trashed_at_ms, 1_700_000_000_000);
        let row = db.get(&gallery, "mold-test-0001.png").unwrap().unwrap();
        assert_eq!(row.trashed_at_ms, Some(1_700_000_000_000));
        assert!(
            db.list_live(Some(&gallery)).unwrap().is_empty(),
            "a trashed print leaves the live listing"
        );
        assert_eq!(db.list_trashed(Some(&gallery)).unwrap().len(), 1);
    }

    #[test]
    fn trash_imports_a_row_for_a_file_reconcile_has_not_seen() {
        let tmp = tempfile::tempdir().unwrap();
        let gallery = tmp.path().join("output");
        std::fs::create_dir_all(&gallery).unwrap();
        let live = gallery.join("mold-fresh-0002.png");
        write_png(&live);
        let db = open_db(tmp.path());
        assert!(db.get(&gallery, "mold-fresh-0002.png").unwrap().is_none());

        trash_local_print(&db, &live, 42).unwrap();

        let row = db.get(&gallery, "mold-fresh-0002.png").unwrap().unwrap();
        assert_eq!(row.trashed_at_ms, Some(42));
        assert!(mold_db::trash::trash_dir(&gallery)
            .join("mold-fresh-0002.png")
            .is_file());
    }

    #[test]
    fn trash_is_idempotent_for_an_already_trashed_print() {
        let tmp = tempfile::tempdir().unwrap();
        let gallery = tmp.path().join("output");
        std::fs::create_dir_all(&gallery).unwrap();
        let live = gallery.join("mold-twice-0003.png");
        write_png(&live);
        let db = open_db(tmp.path());
        trash_local_print(&db, &live, 7).unwrap();
        trash_local_print(&db, &live, 8).unwrap();
        let row = db.get(&gallery, "mold-twice-0003.png").unwrap().unwrap();
        assert_eq!(row.trashed_at_ms, Some(7), "second call is a no-op");
    }

    #[test]
    fn trash_refuses_a_path_that_is_not_a_gallery_file() {
        let tmp = tempfile::tempdir().unwrap();
        let db = open_db(tmp.path());
        let missing = tmp.path().join("output").join("nope.png");
        let err = trash_local_print(&db, &missing, 1).unwrap_err();
        assert!(err.to_string().contains("not a gallery file"), "{err:#}");
    }
}
