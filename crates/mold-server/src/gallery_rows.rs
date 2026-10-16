//! Wire rows for `GET /api/gallery`, either for a whole directory or for a
//! caller-named set of prints.
//!
//! The full listing and [`build_rows`] answer the same per-name question —
//! "what does the listing say about this print" — and must agree exactly:
//! `?filename=` is a read of ONE element of the listing, not a second
//! policy. The per-name rules are the full listing's, in its order:
//!
//! - `?view=trash`: only a trashed DB row, rendered from the row itself.
//! - `?view=library`: a DB row is dropped while the committed archive
//!   holds the name retired or quarantined; otherwise the live archive
//!   record wins over the row. An archive entry with no DB row is listed
//!   unless quarantined (`CommittedArchiveIndex::overlay_db_gallery`).
//! - Organization is folded in by filename afterwards, and a print the DB
//!   says is trashed never appears in the library, even one the archive
//!   re-added (`gallery_organization::enrich_library_listing`).
//! - Asset inventories attach from the DB row; an archive-only print has
//!   none. Every row is stamped with the poster revision.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use mold_core::GalleryImage;
use mold_db::{GenerationRecord, MetadataDb};

use crate::batch_transaction::CommittedArchiveIndex;
use crate::gallery_organization::{apply_organization, enrich_library_listing};
use crate::routes::GalleryView;

#[cfg(test)]
thread_local! {
    /// How many whole-directory listings this thread has built. Lets a test
    /// prove a single-print read never takes the full path.
    pub(crate) static FULL_LISTING_BUILDS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// The listing for `GET /api/gallery`, answered from the metadata DB.
///
/// `only` narrows it to one print through [`build_rows`] without building
/// the rest of the directory. `Ok(None)` means the DB has not indexed this
/// directory at all yet, and the caller falls back to a filesystem scan —
/// for the library view only, exactly as the full listing always has.
/// `archive` is loaded lazily, and only for the library view.
pub(crate) fn listing_images(
    db: &MetadataDb,
    archive: impl FnOnce() -> anyhow::Result<Arc<CommittedArchiveIndex>>,
    dir: &Path,
    view: GalleryView,
    retention_days: u32,
    only: Option<&str>,
) -> anyhow::Result<Option<Vec<GalleryImage>>> {
    let Some(name) = only else {
        return full_listing(db, archive, dir, view, retention_days);
    };
    let names = [name.to_string()];
    let archive = match view {
        GalleryView::Trash => None,
        GalleryView::Library => {
            if !db.has_rows_for_dir(dir)? {
                return Ok(None);
            }
            Some(archive()?)
        }
    };
    let built = build_rows(db, archive.as_deref(), dir, &names, view, retention_days)?;
    Ok(Some(built.into_values().flatten().collect()))
}

/// Every print in `dir`, in listing order. The established whole-directory
/// path; [`build_rows`] is its per-name twin.
fn full_listing(
    db: &MetadataDb,
    archive: impl FnOnce() -> anyhow::Result<Arc<CommittedArchiveIndex>>,
    dir: &Path,
    view: GalleryView,
    retention_days: u32,
) -> anyhow::Result<Option<Vec<GalleryImage>>> {
    #[cfg(test)]
    FULL_LISTING_BUILDS.with(|count| count.set(count.get() + 1));
    let organization = db
        .organization_for_dir(dir)
        .map_err(|e| anyhow::anyhow!("gallery organization query failed: {e}"))?;
    match view {
        GalleryView::Trash => {
            let rows = db.list_trashed(Some(dir))?;
            let mut images = rows
                .iter()
                .map(|row| {
                    let mut image = row.to_gallery_image();
                    apply_organization(&mut image, organization.get(&row.filename), retention_days);
                    image
                })
                .collect::<Vec<_>>();
            crate::generation_assets::attach_to_images(db, dir, &rows, &mut images);
            images.sort_by_key(|image| {
                std::cmp::Reverse((image.trashed_at.unwrap_or(0), image.timestamp))
            });
            Ok(Some(images))
        }
        GalleryView::Library => {
            let rows = db.list(Some(dir))?;
            if rows.is_empty() {
                return Ok(None);
            }
            let archive = archive()?;
            let images = archive.overlay_db_gallery(&rows);
            let mut images = enrich_library_listing(images, &organization, retention_days);
            crate::generation_assets::attach_to_images(db, dir, &rows, &mut images);
            Ok(Some(images))
        }
    }
}

/// The listing's row for each of `names`, or `None` where the listing would
/// not show that print. Reads only the named rows and their organization;
/// never the whole directory. `archive` is consulted for the library view
/// only (pass `None` for trash).
pub(crate) fn build_rows(
    db: &MetadataDb,
    archive: Option<&CommittedArchiveIndex>,
    dir: &Path,
    names: &[String],
    view: GalleryView,
    retention_days: u32,
) -> anyhow::Result<BTreeMap<String, Option<GalleryImage>>> {
    let rows = db.rows_by_filenames(dir, names)?;
    build_rows_from(db, archive, dir, names, rows, view, retention_days)
}

fn build_rows_from(
    db: &MetadataDb,
    archive: Option<&CommittedArchiveIndex>,
    dir: &Path,
    names: &[String],
    rows: std::collections::HashMap<String, GenerationRecord>,
    view: GalleryView,
    retention_days: u32,
) -> anyhow::Result<BTreeMap<String, Option<GalleryImage>>> {
    let empty = CommittedArchiveIndex::default();
    let archive = archive.unwrap_or(&empty);
    let organization = db
        .organization_for_filenames(dir, names)
        .map_err(|e| anyhow::anyhow!("gallery organization query failed: {e}"))?;
    let mut built = BTreeMap::new();
    let mut listed_rows = Vec::new();
    let mut images = Vec::new();
    for name in names {
        if built.contains_key(name) {
            continue;
        }
        let row = rows.get(name);
        let image = match view {
            GalleryView::Trash => row
                .filter(|row| row.trashed_at_ms.is_some())
                .map(GenerationRecord::to_gallery_image),
            GalleryView::Library => library_image(archive, name, row),
        };
        let Some(mut image) = image else {
            built.insert(name.clone(), None);
            continue;
        };
        let print = organization.get(name);
        if view == GalleryView::Library && print.is_some_and(|p| p.trashed_at_ms.is_some()) {
            built.insert(name.clone(), None);
            continue;
        }
        apply_organization(&mut image, print, retention_days);
        if let Some(row) = row {
            listed_rows.push(row.clone());
        }
        built.insert(name.clone(), None);
        images.push(image);
    }
    crate::generation_assets::attach_to_images(db, dir, &listed_rows, &mut images);
    for mut image in images {
        crate::thumbnails::stamp_poster_revision(&mut image);
        built.insert(image.filename.clone(), Some(image));
    }
    Ok(built)
}

/// `overlay_db_gallery`'s answer for one name.
fn library_image(
    archive: &CommittedArchiveIndex,
    name: &str,
    row: Option<&GenerationRecord>,
) -> Option<GalleryImage> {
    match row {
        Some(row) => {
            if archive.is_retired(name) || archive.is_quarantined(name) {
                return None;
            }
            Some(
                archive
                    .get(name)
                    .map(|entry| entry.record.to_gallery_image())
                    .unwrap_or_else(|| row.to_gallery_image()),
            )
        }
        None => archive
            .entries
            .get(name)
            .filter(|_| !archive.is_quarantined(name))
            .map(|entry| entry.record.to_gallery_image()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_transaction::{ArchivedChildIdentity, CommittedArchiveEntry};
    use mold_core::OutputFormat;
    use mold_db::RecordSource;

    const RETENTION: u32 = 30;

    fn record(dir: &Path, name: &str, created_at_ms: i64) -> GenerationRecord {
        let mut record = GenerationRecord::from_save(
            dir,
            name,
            OutputFormat::Png,
            mold_db::metadata_io::synthesize_from_filename(name, created_at_ms as u64),
            RecordSource::Server,
            created_at_ms,
        );
        record.file_mtime_ms = Some(created_at_ms);
        record.file_size_bytes = Some(100);
        record
    }

    fn entry(record: GenerationRecord) -> CommittedArchiveEntry {
        CommittedArchiveEntry {
            identity: ArchivedChildIdentity {
                parent_id: "parent".into(),
                attempt_generation: 1,
                child_index: 0,
                final_name: record.filename.clone(),
                checksum_sha256: "0".repeat(64),
                size_bytes: 100,
            },
            record,
            facts: None,
            retained_media: Vec::new(),
        }
    }

    fn asset(id: &str) -> mold_db::generation_assets::GenerationAssetRecord {
        mold_db::generation_assets::GenerationAssetRecord {
            asset: mold_core::GenerationAsset {
                asset_id: id.into(),
                role: id.into(),
                display_name: format!("chair-{id}.png"),
                media_type: "image/png".into(),
                size_bytes: 12,
                sha256: "a".repeat(64),
                width: Some(8),
                height: Some(8),
            },
            storage: mold_db::generation_assets::AssetStorage::EmbeddedGlb { role: id.into() },
        }
    }

    /// One gallery holding every per-name case the listing distinguishes.
    struct Fixture {
        _tmp: tempfile::TempDir,
        dir: std::path::PathBuf,
        db: MetadataDb,
        archive: Arc<CommittedArchiveIndex>,
        collection_id: String,
    }

    const NAMES: [&str; 8] = [
        "plain.png",
        "organized.png",
        "archived.png",
        "archive-only.png",
        "retired.png",
        "quarantined.png",
        "trashed.png",
        "missing.png",
    ];

    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let db = MetadataDb::open_in_memory().unwrap();
        for (offset, name) in [
            "plain.png",
            "organized.png",
            "archived.png",
            "retired.png",
            "quarantined.png",
            "trashed.png",
        ]
        .into_iter()
        .enumerate()
        {
            db.upsert(&record(&dir, name, 1_000_000 + offset as i64 * 1_000))
                .unwrap();
        }
        let s = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        db.set_title(&dir, "organized.png", Some("A title"))
            .unwrap();
        db.set_favorite(&dir, "organized.png", true).unwrap();
        db.add_tags(&dir, "organized.png", &s(&["zeta", "Alpha"]))
            .unwrap();
        let collection = db.create_collection("Shelf", None).unwrap();
        db.collection_add(&collection.id, &dir, &s(&["organized.png", "trashed.png"]))
            .unwrap();
        db.add_tags(&dir, "trashed.png", &s(&["binned"])).unwrap();
        assert!(db.mark_trashed(&dir, "trashed.png", 5_000_000).unwrap());

        // Assets for the organized row, recorded as a current scan so the
        // attach path reads them rather than re-indexing a PNG.
        let organized = db.get(&dir, "organized.png").unwrap().unwrap();
        mold_db::generation_assets::replace_for_generation(
            &db,
            organized.id.unwrap(),
            &format!("{}:{}", organized.file_mtime_ms.unwrap(), 100),
            &[asset("base_color"), asset("normal")],
        )
        .unwrap();

        let mut archive = CommittedArchiveIndex::default();
        // The archive's record differs from the row, so "archive wins" is
        // observable.
        let mut archived = record(&dir, "archived.png", 9_000_000);
        archived.metadata.prompt = "from the archive".into();
        archive
            .entries
            .insert("archived.png".into(), entry(archived));
        archive.entries.insert(
            "archive-only.png".into(),
            entry(record(&dir, "archive-only.png", 2_000_000)),
        );
        archive.retired_names.insert("retired.png".into());
        archive.quarantined_names.insert("quarantined.png".into());
        archive.entries.insert(
            "quarantined.png".into(),
            entry(record(&dir, "quarantined.png", 3_000_000)),
        );
        Fixture {
            _tmp: tmp,
            dir,
            db,
            archive: Arc::new(archive),
            collection_id: collection.id,
        }
    }

    fn full(fx: &Fixture, view: GalleryView) -> Vec<serde_json::Value> {
        let mut images = listing_images(
            &fx.db,
            || Ok(fx.archive.clone()),
            &fx.dir,
            view,
            RETENTION,
            None,
        )
        .unwrap()
        .expect("the directory is indexed");
        // The response exit stamps every listing row.
        crate::thumbnails::stamp_poster_revision_all(&mut images);
        images
            .iter()
            .map(|image| serde_json::to_value(image).unwrap())
            .collect()
    }

    fn one(fx: &Fixture, view: GalleryView, name: &str) -> Vec<serde_json::Value> {
        listing_images(
            &fx.db,
            || Ok(fx.archive.clone()),
            &fx.dir,
            view,
            RETENTION,
            Some(name),
        )
        .unwrap()
        .expect("the directory is indexed")
        .iter()
        .map(|image| serde_json::to_value(image).unwrap())
        .collect()
    }

    fn assert_parity(fx: &Fixture, view: GalleryView) -> Vec<serde_json::Value> {
        let listing = full(fx, view);
        for name in NAMES {
            let expected = listing
                .iter()
                .filter(|image| image["filename"] == name)
                .cloned()
                .collect::<Vec<_>>();
            assert_eq!(
                one(fx, view, name),
                expected,
                "{view:?} ?filename={name} must equal its element of the full listing"
            );
        }
        listing
    }

    #[test]
    fn a_single_library_read_equals_its_element_of_the_full_listing() {
        let fx = fixture();
        let listing = assert_parity(&fx, GalleryView::Library);
        let listed = listing
            .iter()
            .map(|image| image["filename"].as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        // The fixture really covers every case: kept, archive-only, and the
        // three exclusions.
        assert_eq!(
            listed,
            [
                "archive-only.png",
                "archived.png",
                "organized.png",
                "plain.png"
            ]
            .into_iter()
            .collect()
        );
        let organized = one(&fx, GalleryView::Library, "organized.png").remove(0);
        assert_eq!(organized["title"], "A title");
        assert_eq!(organized["favorite"], true);
        assert_eq!(organized["tags"], serde_json::json!(["Alpha", "zeta"]));
        assert_eq!(
            organized["collections"],
            serde_json::json!([fx.collection_id])
        );
        assert_eq!(organized["assets"].as_array().unwrap().len(), 2);
        let archived = one(&fx, GalleryView::Library, "archived.png").remove(0);
        assert_eq!(archived["metadata"]["prompt"], "from the archive");
    }

    #[test]
    fn a_single_trash_read_equals_its_element_of_the_full_listing() {
        let fx = fixture();
        let listing = assert_parity(&fx, GalleryView::Trash);
        assert_eq!(listing.len(), 1);
        let trashed = one(&fx, GalleryView::Trash, "trashed.png").remove(0);
        assert_eq!(trashed["trashed_at"], 5_000);
        assert!(trashed["purge_at"].is_u64());
        assert_eq!(trashed["tags"], serde_json::json!(["binned"]));
        assert_eq!(
            trashed["collections"],
            serde_json::json!([fx.collection_id])
        );
    }

    #[test]
    fn a_single_print_read_never_builds_the_whole_directory() {
        let fx = fixture();
        let before = FULL_LISTING_BUILDS.with(std::cell::Cell::get);
        for view in [GalleryView::Library, GalleryView::Trash] {
            for name in NAMES {
                one(&fx, view, name);
            }
        }
        assert_eq!(
            FULL_LISTING_BUILDS.with(std::cell::Cell::get),
            before,
            "?filename= must not take the full-directory path"
        );
        full(&fx, GalleryView::Library);
        assert_eq!(
            FULL_LISTING_BUILDS.with(std::cell::Cell::get),
            before + 1,
            "the counter observes the full path"
        );
    }

    #[test]
    fn a_trash_read_never_loads_the_archive() {
        let fx = fixture();
        let images = listing_images(
            &fx.db,
            || panic!("trash rows come from the DB alone"),
            &fx.dir,
            GalleryView::Trash,
            RETENTION,
            Some("trashed.png"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(images.len(), 1);
    }

    #[test]
    fn an_unindexed_directory_falls_back_for_a_single_read_too() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let archive = Arc::new(CommittedArchiveIndex::default());
        for only in [None, Some("anything.png")] {
            assert!(listing_images(
                &db,
                || Ok(archive.clone()),
                tmp.path(),
                GalleryView::Library,
                RETENTION,
                only,
            )
            .unwrap()
            .is_none());
        }
    }

    #[test]
    fn build_rows_answers_every_requested_name() {
        let fx = fixture();
        let names = NAMES.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        let rows = build_rows(
            &fx.db,
            Some(&fx.archive),
            &fx.dir,
            &names,
            GalleryView::Library,
            RETENTION,
        )
        .unwrap();
        assert_eq!(rows.len(), NAMES.len());
        for (name, expected) in [
            ("plain.png", true),
            ("organized.png", true),
            ("archived.png", true),
            ("archive-only.png", true),
            ("retired.png", false),
            ("quarantined.png", false),
            ("trashed.png", false),
            ("missing.png", false),
        ] {
            assert_eq!(rows[name].is_some(), expected, "{name}");
        }
    }
}
