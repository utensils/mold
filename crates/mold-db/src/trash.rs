//! Gallery trash: the DB side of "move to `<output_dir>/.trash/`".
//!
//! A trashed print keeps its `generations` row and its `(output_dir,
//! filename)` identity; only `trashed_at_ms` is set and the bytes move to
//! [`trash_dir`]`/<filename>`. Next to the bytes sits a tombstone
//! (`<filename>.trash.json`, see [`Tombstone`]) carrying everything needed to
//! rebuild the row — and its organization — if `mold.db` is ever lost:
//! reconcile's trash pass imports tombstoned files that have no row.
//!
//! File moves are the caller's job (the server does them under its
//! publication gate; the desktop's offline path does them directly). This
//! module owns the row flag, the retention arithmetic, and the tombstone
//! format.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::path::canonical_dir_string;
use crate::record::GenerationRecord;
use crate::MetadataDb;

/// Name of the per-gallery trash directory.
pub const TRASH_DIR_NAME: &str = ".trash";
/// Suffix appended to a trashed filename for its tombstone.
pub const TOMBSTONE_SUFFIX: &str = ".trash.json";
/// Current tombstone format version.
pub const TOMBSTONE_VERSION: u32 = 1;
/// Milliseconds in a day — the retention unit.
pub const DAY_MS: i64 = 24 * 60 * 60 * 1000;

/// `<output_dir>/.trash`.
pub fn trash_dir(output_dir: &Path) -> PathBuf {
    output_dir.join(TRASH_DIR_NAME)
}

/// `<trash_dir>/<filename>.trash.json`.
pub fn tombstone_path(trash_dir: &Path, filename: &str) -> PathBuf {
    trash_dir.join(format!("{filename}{TOMBSTONE_SUFFIX}"))
}

/// True when `name` is a tombstone file name.
pub fn is_tombstone_filename(name: &str) -> bool {
    name.len() > TOMBSTONE_SUFFIX.len() && name.ends_with(TOMBSTONE_SUFFIX)
}

/// When a trashed print will be purged, or `None` when `retention_days == 0`
/// (keep forever).
pub fn purge_at_ms(trashed_at_ms: i64, retention_days: u32) -> Option<i64> {
    if retention_days == 0 {
        return None;
    }
    Some(trashed_at_ms.saturating_add(i64::from(retention_days).saturating_mul(DAY_MS)))
}

/// Sidecar written next to a trashed file so the row can be rebuilt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tombstone {
    /// [`TOMBSTONE_VERSION`].
    pub version: u32,
    pub filename: String,
    pub trashed_at_ms: i64,
    /// The gallery directory the file lived in (the row's `output_dir`).
    pub original_dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub favorite: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Collection SLUGS (not ids) so a rebuilt DB or another host can
    /// re-home the print by name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collections: Vec<String>,
    /// The row's serialized `OutputMetadata`, when it was available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata_json: Option<String>,
}

/// Write `tombstone` under `trash_dir` (created if needed). The write is
/// atomic: a temp file is renamed into place so a crash never leaves a
/// half-written sidecar that reconcile would then refuse to parse.
pub fn write_tombstone(trash_dir: &Path, tombstone: &Tombstone) -> Result<PathBuf> {
    std::fs::create_dir_all(trash_dir)
        .with_context(|| format!("creating trash dir {}", trash_dir.display()))?;
    let final_path = tombstone_path(trash_dir, &tombstone.filename);
    let tmp_path = trash_dir.join(format!(
        ".{}.{}.tmp",
        tombstone.filename,
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(tombstone).context("serializing tombstone")?;
    std::fs::write(&tmp_path, bytes)
        .with_context(|| format!("writing tombstone {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "moving tombstone {} into place at {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    Ok(final_path)
}

/// Parse a tombstone. Unknown future versions still parse when the fields
/// are compatible; callers decide what to trust.
pub fn read_tombstone(path: &Path) -> Result<Tombstone> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading tombstone {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing tombstone {}", path.display()))
}

/// Remove a tombstone, tolerating its absence.
pub fn remove_tombstone(trash_dir: &Path, filename: &str) -> Result<()> {
    match std::fs::remove_file(tombstone_path(trash_dir, filename)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("removing tombstone for {filename}")),
    }
}

impl MetadataDb {
    /// Flag a print as trashed at `now_ms`. Returns `false` when no row
    /// matches. Re-flagging an already-trashed row updates the timestamp.
    pub fn mark_trashed(&self, dir: &Path, filename: &str, now_ms: i64) -> Result<bool> {
        let dir_key = canonical_dir_string(dir);
        self.with_conn(|conn| {
            let n = conn.execute(
                "UPDATE generations SET trashed_at_ms = ?3
                 WHERE output_dir = ?1 AND filename = ?2",
                params![dir_key, filename, now_ms],
            )?;
            Ok(n > 0)
        })
    }

    /// Clear a print's trashed flag. Returns `false` when no row matches.
    pub fn mark_restored(&self, dir: &Path, filename: &str) -> Result<bool> {
        let dir_key = canonical_dir_string(dir);
        self.with_conn(|conn| {
            let n = conn.execute(
                "UPDATE generations SET trashed_at_ms = NULL
                 WHERE output_dir = ?1 AND filename = ?2",
                params![dir_key, filename],
            )?;
            Ok(n > 0)
        })
    }

    /// Every trashed row in `dir`, newest-first. Alias of
    /// [`Self::list_trashed`] scoped to one directory.
    pub fn list_trashed_rows(&self, dir: &Path) -> Result<Vec<GenerationRecord>> {
        self.list_trashed(Some(dir))
    }

    /// Trashed rows in `dir` whose retention has elapsed at `now_ms`.
    /// `retention_days == 0` means keep forever — always empty.
    pub fn expired_trashed(
        &self,
        dir: &Path,
        retention_days: u32,
        now_ms: i64,
    ) -> Result<Vec<GenerationRecord>> {
        if retention_days == 0 {
            return Ok(Vec::new());
        }
        Ok(self
            .list_trashed(Some(dir))?
            .into_iter()
            .filter(|rec| {
                rec.trashed_at_ms
                    .and_then(|t| purge_at_ms(t, retention_days))
                    .is_some_and(|purge_at| purge_at <= now_ms)
            })
            .collect())
    }

    /// Build the tombstone for a print from its row plus organization
    /// state. `None` when the print has no row. `trashed_at_ms` is supplied
    /// by the caller so the tombstone and [`Self::mark_trashed`] agree.
    pub fn build_tombstone(
        &self,
        dir: &Path,
        filename: &str,
        trashed_at_ms: i64,
    ) -> Result<Option<Tombstone>> {
        let Some(rec) = self.get(dir, filename)? else {
            return Ok(None);
        };
        let org = self.print_organization(dir, filename)?.unwrap_or_default();
        let collections = self
            .collection_slugs_for_print(dir, filename)
            .unwrap_or_default();
        let metadata_json = serde_json::to_string(&rec.metadata).ok();
        Ok(Some(Tombstone {
            version: TOMBSTONE_VERSION,
            filename: filename.to_string(),
            trashed_at_ms,
            original_dir: rec.output_dir,
            title: rec.title.or(org.title),
            favorite: rec.favorite || org.favorite,
            tags: org.tags,
            collections,
            metadata_json,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::RecordSource;
    use mold_core::{OutputFormat, OutputMetadata};

    const DIR: &str = "/tmp/trash-tests";

    fn meta() -> OutputMetadata {
        let req: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a trashed owl","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        OutputMetadata::from_generate_request(&req, 9, None, "test")
    }

    fn seed(db: &MetadataDb, names: &[&str]) {
        for name in names {
            let rec = GenerationRecord::from_save(
                Path::new(DIR),
                *name,
                OutputFormat::Png,
                meta(),
                RecordSource::Server,
                1,
            );
            db.upsert(&rec).unwrap();
        }
    }

    #[test]
    fn trash_paths_are_derived_from_the_output_dir() {
        let dir = Path::new("/gallery");
        assert_eq!(trash_dir(dir), PathBuf::from("/gallery/.trash"));
        assert_eq!(
            tombstone_path(&trash_dir(dir), "a.png"),
            PathBuf::from("/gallery/.trash/a.png.trash.json")
        );
        assert!(is_tombstone_filename("a.png.trash.json"));
        assert!(!is_tombstone_filename("a.png"));
        assert!(!is_tombstone_filename(".trash.json"));
    }

    #[test]
    fn purge_at_is_retention_days_after_trashing_and_none_for_forever() {
        assert_eq!(purge_at_ms(1_000, 0), None);
        assert_eq!(purge_at_ms(1_000, 1), Some(1_000 + DAY_MS));
        assert_eq!(purge_at_ms(0, 30), Some(30 * DAY_MS));
        assert_eq!(purge_at_ms(i64::MAX - 1, 3650), Some(i64::MAX));
    }

    #[test]
    fn mark_trashed_and_restored_flip_the_flag_and_report_missing_rows() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        assert!(db.mark_trashed(Path::new(DIR), "a.png", 500).unwrap());
        assert_eq!(
            db.get(Path::new(DIR), "a.png")
                .unwrap()
                .unwrap()
                .trashed_at_ms,
            Some(500)
        );
        assert_eq!(db.list_trashed_rows(Path::new(DIR)).unwrap().len(), 1);
        assert!(db.list_live(Some(Path::new(DIR))).unwrap().is_empty());
        assert!(db.mark_restored(Path::new(DIR), "a.png").unwrap());
        assert!(db.get(Path::new(DIR), "a.png").unwrap().unwrap().is_live());
        assert!(!db.mark_trashed(Path::new(DIR), "ghost.png", 1).unwrap());
        assert!(!db.mark_restored(Path::new(DIR), "ghost.png").unwrap());
    }

    #[test]
    fn expired_trashed_honors_retention_and_keep_forever() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["old.png", "new.png", "live.png"]);
        let dir = Path::new(DIR);
        db.mark_trashed(dir, "old.png", 0).unwrap();
        db.mark_trashed(dir, "new.png", 29 * DAY_MS).unwrap();
        let now = 30 * DAY_MS + 1;
        let expired: Vec<_> = db
            .expired_trashed(dir, 30, now)
            .unwrap()
            .into_iter()
            .map(|r| r.filename)
            .collect();
        assert_eq!(expired, vec!["old.png"]);
        assert!(db.expired_trashed(dir, 0, now).unwrap().is_empty());
        let all: Vec<_> = db
            .expired_trashed(dir, 1, 40 * DAY_MS)
            .unwrap()
            .into_iter()
            .map(|r| r.filename)
            .collect();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&"old.png".to_string()) && all.contains(&"new.png".to_string()));
        // Exactly at the boundary counts as expired.
        assert_eq!(db.expired_trashed(dir, 30, 30 * DAY_MS).unwrap().len(), 1);
        assert!(db
            .expired_trashed(dir, 30, 30 * DAY_MS - 1)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn tombstone_round_trips_through_disk_and_tolerates_missing_optional_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let trash = trash_dir(tmp.path());
        let tomb = Tombstone {
            version: TOMBSTONE_VERSION,
            filename: "a.png".into(),
            trashed_at_ms: 123,
            original_dir: tmp.path().to_string_lossy().into_owned(),
            title: Some("Owl".into()),
            favorite: true,
            tags: vec!["birds".into()],
            collections: vec!["shelf".into()],
            metadata_json: Some("{}".into()),
        };
        let path = write_tombstone(&trash, &tomb).unwrap();
        assert_eq!(path, tombstone_path(&trash, "a.png"));
        assert_eq!(read_tombstone(&path).unwrap(), tomb);
        // No stray temp file.
        let leftovers: Vec<_> = std::fs::read_dir(&trash)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(leftovers, vec!["a.png.trash.json"]);

        let minimal = r#"{"version":1,"filename":"b.png","trashed_at_ms":5,"original_dir":"/g"}"#;
        let minimal_path = trash.join("b.png.trash.json");
        std::fs::write(&minimal_path, minimal).unwrap();
        let parsed = read_tombstone(&minimal_path).unwrap();
        assert_eq!(parsed.title, None);
        assert!(!parsed.favorite);
        assert!(parsed.tags.is_empty());
        assert!(parsed.collections.is_empty());
        assert_eq!(parsed.metadata_json, None);

        remove_tombstone(&trash, "b.png").unwrap();
        assert!(!minimal_path.exists());
        remove_tombstone(&trash, "b.png").unwrap();
    }

    #[test]
    fn build_tombstone_captures_row_and_organization() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        let dir = Path::new(DIR);
        db.set_title(dir, "a.png", Some("Owl")).unwrap();
        db.set_favorite(dir, "a.png", true).unwrap();
        db.add_tags(dir, "a.png", &["zeta".into(), "Alpha".into()])
            .unwrap();
        let c = db.create_collection("Night Owls", None).unwrap();
        db.collection_add(&c.id, dir, &["a.png".into()]).unwrap();

        let tomb = db.build_tombstone(dir, "a.png", 777).unwrap().unwrap();
        assert_eq!(tomb.version, TOMBSTONE_VERSION);
        assert_eq!(tomb.filename, "a.png");
        assert_eq!(tomb.trashed_at_ms, 777);
        assert_eq!(tomb.original_dir, canonical_dir_string(dir));
        assert_eq!(tomb.title.as_deref(), Some("Owl"));
        assert!(tomb.favorite);
        assert_eq!(tomb.tags, vec!["Alpha", "zeta"]);
        assert_eq!(tomb.collections, vec!["night-owls"]);
        let meta: OutputMetadata =
            serde_json::from_str(tomb.metadata_json.as_deref().unwrap()).unwrap();
        assert_eq!(meta.prompt, "a trashed owl");
        assert!(db.build_tombstone(dir, "ghost.png", 1).unwrap().is_none());
    }
}
