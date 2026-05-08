//! One-shot backfill: drops `mold-catalog.json` next to every
//! pre-existing Civitai install whose recipe row is still in the
//! catalog DB. Runs on server boot so users upgrading from the
//! bulk-scrape era don't lose their installed LoRAs from the picker
//! the moment the SPA switches to `/api/catalog/installed`.
//!
//! The walk is bounded to the immediate `cv-*` subdirs of `models_dir`
//! and reads only the catalog DB — no network calls, no recipe fetch.
//! Failure on a single row logs and continues; the picker degrades to
//! "fewer rows" rather than blocking startup.

use std::path::Path;
use std::sync::Arc;

use mold_catalog::sidecar::{sidecar_from_row, write_sidecar, SIDECAR_FILENAME};
use mold_db::MetadataDb;

/// Returns the number of sidecars written. `0` is the steady state on
/// every boot after the first.
pub fn backfill(models_dir: &Path, catalog_db: &MetadataDb) -> usize {
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        // Missing models_dir is fine — nothing to backfill yet.
        return 0;
    };

    let mut written = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        // Only `cv-*` dirs use the recipe layout this backfill targets.
        // HF entries land under `{family}/{author}/{name}/...` and do
        // not yet emit sidecars (see `sidecar` module-doc).
        let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !dir_name.starts_with("cv-") {
            continue;
        }
        let sc_path = path.join(SIDECAR_FILENAME);
        if sc_path.exists() {
            continue;
        }
        // Reverse the sanitization in `sanitize_recipe_id` to recover
        // the original catalog id (`cv:8001` from `cv-8001`).
        let catalog_id = format!("cv:{}", &dir_name["cv-".len()..]);

        let row = match catalog_db.catalog_get(&catalog_id) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(e) => {
                tracing::warn!(
                    target: "catalog.sidecar.backfill",
                    catalog_id = %catalog_id,
                    error = %e,
                    "DB lookup failed during backfill; skipping",
                );
                continue;
            }
        };

        // Find a primary file under the cv-<id> subdir. The recipe is
        // stored as JSON in the row; we render the first file's dest
        // template the same way the install handler does.
        let recipe: mold_catalog::entry::DownloadRecipe =
            match serde_json::from_str(&row.download_recipe) {
                Ok(r) => r,
                Err(_) => continue,
            };
        let Some(first) = recipe.files.first() else {
            continue;
        };
        let (author, name) = match row.source_id.split_once('/') {
            Some((a, n)) => (a, n),
            None => ("", row.source_id.as_str()),
        };
        let primary_filename_rel =
            mold_catalog::entry::render_recipe_dest(&first.dest, &row.family, author, name);

        let sidecar = sidecar_from_row(&row, primary_filename_rel);
        match write_sidecar(&sc_path, &sidecar) {
            Ok(()) => {
                written += 1;
            }
            Err(e) => {
                tracing::warn!(
                    target: "catalog.sidecar.backfill",
                    catalog_id = %catalog_id,
                    error = %e,
                    "sidecar write failed during backfill",
                );
            }
        }
    }

    if written > 0 {
        tracing::info!(
            target: "catalog.sidecar.backfill",
            count = written,
            "backfilled sidecars for installed catalog rows",
        );
    }
    written
}

/// Run [`backfill`] on a tokio task so server startup isn't blocked on
/// the filesystem walk.
pub fn spawn(models_dir: std::path::PathBuf, catalog_db: Arc<MetadataDb>) {
    tokio::task::spawn_blocking(move || {
        backfill(&models_dir, &catalog_db);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_catalog::sidecar::{read_sidecar, SIDECAR_FILENAME};

    fn seed_row(db: &MetadataDb) {
        db.catalog_upsert(
            "flux",
            &[mold_db::catalog::CatalogRow {
                id: "cv:8001".into(),
                source: "civitai".into(),
                source_id: "8001".into(),
                name: "Backfill Sample".into(),
                author: Some("alice".into()),
                family: "flux".into(),
                family_role: "finetune".into(),
                sub_family: Some("flux1-d".into()),
                modality: "image".into(),
                kind: "lora".into(),
                file_format: "safetensors".into(),
                bundling: "single-file".into(),
                size_bytes: Some(1),
                download_count: 1,
                rating: None,
                likes: 0,
                nsfw: 0,
                thumbnail_url: None,
                description: None,
                license: None,
                license_flags: None,
                tags: Some("[]".into()),
                companions: Some("[]".into()),
                download_recipe: r#"{
                    "files": [
                        {
                            "url": "https://example/x.safetensors",
                            "dest": "{family}/civitai/8001/x.safetensors",
                            "sha256": null,
                            "size_bytes": null
                        }
                    ],
                    "needs_token": "civitai"
                }"#
                .into(),
                engine_phase: 3,
                created_at: None,
                updated_at: None,
                added_at: 0,
                trained_words: r#"["trigger-A"]"#.into(),
            }],
        )
        .unwrap();
    }

    #[test]
    fn writes_sidecar_for_existing_install_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("cv-8001")).unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        seed_row(&db);

        let written = backfill(tmp.path(), &db);
        assert_eq!(written, 1);

        let sc = read_sidecar(&tmp.path().join("cv-8001").join(SIDECAR_FILENAME)).unwrap();
        assert_eq!(sc.id, "cv:8001");
        assert_eq!(sc.kind, "lora");
        assert_eq!(sc.trained_words, vec!["trigger-A"]);
        assert_eq!(sc.primary_filename_rel, "flux/civitai/8001/x.safetensors");
    }

    #[test]
    fn skips_dirs_with_existing_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("cv-8001");
        std::fs::create_dir_all(&dir).unwrap();
        // Pre-existing sidecar (unchanged content). Backfill must not
        // overwrite — existing sidecars may have been written by the
        // install handler with up-to-date data.
        std::fs::write(dir.join(SIDECAR_FILENAME), b"{ \"existing\": true }").unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        seed_row(&db);

        let written = backfill(tmp.path(), &db);
        assert_eq!(written, 0);
    }

    #[test]
    fn skips_non_cv_subdirs() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("flux/civitai/8001")).unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        seed_row(&db);

        let written = backfill(tmp.path(), &db);
        assert_eq!(
            written, 0,
            "backfill must only target cv-* subdirs at the top level"
        );
    }

    #[test]
    fn skips_when_db_has_no_matching_row() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("cv-9999")).unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        // No seed for cv:9999 — the post-deprecation case where the
        // user installed via live search and the DB never had the row.

        let written = backfill(tmp.path(), &db);
        assert_eq!(written, 0);
    }
}
