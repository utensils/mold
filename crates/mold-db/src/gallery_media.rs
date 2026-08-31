//! Repairable projection of gallery-authoritative retained source media.

use anyhow::{ensure, Result};
use rusqlite::params;

use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GalleryMediaBinding {
    pub output_dir: String,
    pub filename: String,
    pub pin_id: String,
    pub media_set_id: String,
    pub owner_uuid: String,
    pub job_id: String,
}

/// Replace the full projection for one exact gallery filename. Callers must
/// derive `bindings` from a committed authority snapshot, never provenance.
pub fn replace_for_item(
    db: &MetadataDb,
    output_dir: &str,
    filename: &str,
    bindings: &[GalleryMediaBinding],
) -> Result<()> {
    db.with_conn(|conn| {
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM gallery_media_bindings WHERE output_dir = ?1 AND filename = ?2",
            params![output_dir, filename],
        )?;
        for binding in bindings {
            ensure!(
                binding.output_dir == output_dir,
                "gallery media output directory mismatch"
            );
            ensure!(
                binding.filename == filename,
                "gallery media filename mismatch"
            );
            ensure!(
                binding.pin_id.len() == 64
                    && binding.pin_id.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "gallery media pin id must be a hexadecimal digest"
            );
            tx.execute(
                "INSERT INTO gallery_media_sets (media_set_id, owner_uuid, job_id)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(media_set_id, owner_uuid, job_id) DO NOTHING",
                params![binding.media_set_id, binding.owner_uuid, binding.job_id],
            )?;
            tx.execute(
                "INSERT INTO gallery_media_bindings
                    (output_dir, filename, pin_id, media_set_id, owner_uuid, job_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    output_dir,
                    filename,
                    binding.pin_id,
                    binding.media_set_id,
                    binding.owner_uuid,
                    binding.job_id
                ],
            )?;
        }
        tx.execute(
            "DELETE FROM gallery_media_sets
              WHERE NOT EXISTS (
                    SELECT 1 FROM gallery_media_bindings AS binding
                     WHERE binding.media_set_id = gallery_media_sets.media_set_id
                       AND binding.owner_uuid = gallery_media_sets.owner_uuid
                       AND binding.job_id = gallery_media_sets.job_id
              )",
            [],
        )?;
        tx.commit()?;
        Ok(())
    })
}

pub fn list_for_item(
    db: &MetadataDb,
    output_dir: &str,
    filename: &str,
) -> Result<Vec<GalleryMediaBinding>> {
    db.with_conn(|conn| {
        let mut statement = conn.prepare(
            "SELECT binding.output_dir, binding.filename, binding.pin_id,
                    binding.media_set_id, media.owner_uuid, media.job_id
               FROM gallery_media_bindings AS binding
               JOIN gallery_media_sets AS media
                 ON media.media_set_id = binding.media_set_id
                AND media.owner_uuid = binding.owner_uuid
                AND media.job_id = binding.job_id
              WHERE binding.output_dir = ?1 AND binding.filename = ?2
              ORDER BY binding.pin_id, binding.media_set_id",
        )?;
        let rows = statement.query_map(params![output_dir, filename], |row| {
            Ok(GalleryMediaBinding {
                output_dir: row.get(0)?,
                filename: row.get(1)?,
                pin_id: row.get(2)?,
                media_set_id: row.get(3)?,
                owner_uuid: row.get(4)?,
                job_id: row.get(5)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

/// Replace one output directory after a complete committed-authority scan.
pub fn replace_directory(
    db: &MetadataDb,
    output_dir: &str,
    bindings: &[GalleryMediaBinding],
) -> Result<()> {
    db.with_conn(|conn| {
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM gallery_media_bindings WHERE output_dir = ?1",
            params![output_dir],
        )?;
        for binding in bindings {
            ensure!(
                binding.output_dir == output_dir,
                "gallery media output directory mismatch"
            );
            ensure!(
                binding.pin_id.len() == 64
                    && binding.pin_id.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "gallery media pin id must be a hexadecimal digest"
            );
            tx.execute(
                "INSERT INTO gallery_media_sets (media_set_id, owner_uuid, job_id)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(media_set_id, owner_uuid, job_id) DO NOTHING",
                params![binding.media_set_id, binding.owner_uuid, binding.job_id],
            )?;
            tx.execute(
                "INSERT INTO gallery_media_bindings
                    (output_dir, filename, pin_id, media_set_id, owner_uuid, job_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    output_dir,
                    binding.filename,
                    binding.pin_id,
                    binding.media_set_id,
                    binding.owner_uuid,
                    binding.job_id
                ],
            )?;
        }
        tx.execute(
            "DELETE FROM gallery_media_sets WHERE NOT EXISTS (
                SELECT 1 FROM gallery_media_bindings AS binding
                WHERE binding.media_set_id = gallery_media_sets.media_set_id
                  AND binding.owner_uuid = gallery_media_sets.owner_uuid
                  AND binding.job_id = gallery_media_sets.job_id
            )",
            [],
        )?;
        tx.commit()?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_preserves_many_to_many_and_collects_unreferenced_sets() {
        let dir = tempfile::tempdir().unwrap();
        let db = MetadataDb::open(&dir.path().join("gallery.db")).unwrap();
        let binding = |filename: &str, set: &str, digit: char| GalleryMediaBinding {
            output_dir: "/gallery".into(),
            filename: filename.into(),
            pin_id: digit.to_string().repeat(64),
            media_set_id: set.into(),
            owner_uuid: "owner".into(),
            job_id: "job".into(),
        };
        replace_for_item(&db, "/gallery", "a.png", &[binding("a.png", "set", 'a')]).unwrap();
        replace_for_item(&db, "/gallery", "b.png", &[binding("b.png", "set", 'b')]).unwrap();
        replace_for_item(&db, "/gallery", "a.png", &[]).unwrap();
        assert!(list_for_item(&db, "/gallery", "a.png").unwrap().is_empty());
        assert_eq!(list_for_item(&db, "/gallery", "b.png").unwrap().len(), 1);
        replace_for_item(&db, "/gallery", "b.png", &[]).unwrap();
        let set_count: i64 = db
            .with_conn(|conn| {
                conn.query_row("SELECT COUNT(*) FROM gallery_media_sets", [], |row| {
                    row.get(0)
                })
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(set_count, 0);
    }

    #[test]
    fn identical_set_ids_from_different_jobs_never_alias() {
        let db = MetadataDb::open_in_memory().unwrap();
        let binding = |filename: &str, owner: &str, job: &str, digit: char| GalleryMediaBinding {
            output_dir: "/gallery".into(),
            filename: filename.into(),
            pin_id: digit.to_string().repeat(64),
            media_set_id: "same-random-set-id".into(),
            owner_uuid: owner.into(),
            job_id: job.into(),
        };
        replace_for_item(
            &db,
            "/gallery",
            "a.png",
            &[binding("a.png", "owner-a", "job-a", 'a')],
        )
        .unwrap();
        replace_for_item(
            &db,
            "/gallery",
            "b.png",
            &[binding("b.png", "owner-b", "job-b", 'b')],
        )
        .unwrap();

        let sets: i64 = db
            .with_conn(|conn| {
                conn.query_row("SELECT COUNT(*) FROM gallery_media_sets", [], |row| {
                    row.get(0)
                })
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(sets, 2);
        assert_eq!(
            list_for_item(&db, "/gallery", "a.png").unwrap()[0].job_id,
            "job-a"
        );
        assert_eq!(
            list_for_item(&db, "/gallery", "b.png").unwrap()[0].job_id,
            "job-b"
        );

        let shared_pin = 'c';
        replace_for_item(
            &db,
            "/gallery",
            "multi.png",
            &[
                binding("multi.png", "owner-a", "job-a", shared_pin),
                binding("multi.png", "owner-b", "job-b", shared_pin),
            ],
        )
        .unwrap();
        assert_eq!(
            list_for_item(&db, "/gallery", "multi.png").unwrap().len(),
            2,
            "one gallery identity may retain multiple exact source-media sets"
        );
    }
}
