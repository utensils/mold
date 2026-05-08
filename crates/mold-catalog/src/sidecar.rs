//! Per-install sidecar metadata. Replaces the bulk-scrape catalog DB on
//! the LoRA-picker read path: every `cv:<id>` install drops a small
//! `mold-catalog.json` next to the primary file, and the picker walks
//! `models_dir` to enumerate installed entries.
//!
//! Why a sidecar (and not a DB table or filesystem-only inference):
//!
//! - **Trigger words** (`trainedWords`) live only in the upstream API
//!   response. Without persisting them at install time, the picker would
//!   either lose them or have to re-fetch live per render.
//! - **Self-describing**: a sidecar travels with the model file, so a
//!   user copying a Civitai download to another mold install gets the
//!   trigger words for free.
//! - **No DB migration**: the catalog DB is on the deprecation path.
//!   Adding a sidecar avoids growing the schema right before its
//!   removal in a follow-up release.
//!
//! Only Civitai sidecars are emitted today. HF entries flow through the
//! manifest registry path which has a different layout, and HF-hosted
//! LoRAs are rare enough to defer.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::entry::CatalogEntry;

pub const SIDECAR_FILENAME: &str = "mold-catalog.json";
pub const SIDECAR_SCHEMA: u32 = 1;

/// On-disk shape of a sidecar. Field names match the wire JSON exactly,
/// so the SPA can read either the full `CatalogEntryWire` or a sidecar
/// without an extra translation layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogSidecar {
    pub schema: u32,
    pub id: String,
    pub source: String,
    pub source_id: String,
    pub name: String,
    pub author: Option<String>,
    pub family: String,
    pub family_role: String,
    pub sub_family: Option<String>,
    pub kind: String,
    pub modality: String,
    pub thumbnail_url: Option<String>,
    pub size_bytes: Option<u64>,
    pub engine_phase: u8,
    /// Trigger phrases for LoRA entries — what the picker renders as
    /// click-to-insert chips. Empty for non-LoRA entries.
    #[serde(default)]
    pub trained_words: Vec<String>,
    /// Filename of the primary safetensors / GGUF file, relative to
    /// the sidecar's own directory. The picker resolves the absolute
    /// path via `sidecar_dir.join(primary_filename_rel)` so the layout
    /// stays portable across `MOLD_HOME` moves.
    pub primary_filename_rel: String,
    pub written_at: i64,
}

/// Build a sidecar from a normalized [`CatalogEntry`] plus the resolved
/// primary file dest (relative to the sidecar's directory). Civitai
/// recipes always have at least one file, but we accept a 0-length
/// `primary_filename_rel` and let the caller decide whether that's
/// fatal — the picker simply won't show the row.
pub fn sidecar_from_entry(entry: &CatalogEntry, primary_filename_rel: String) -> CatalogSidecar {
    CatalogSidecar {
        schema: SIDECAR_SCHEMA,
        id: entry.id.0.clone(),
        source: serde_kebab(&entry.source),
        source_id: entry.source_id.clone(),
        name: entry.name.clone(),
        author: entry.author.clone(),
        family: entry.family.as_str().to_string(),
        family_role: serde_kebab(&entry.family_role),
        sub_family: entry.sub_family.clone(),
        kind: serde_kebab(&entry.kind),
        modality: serde_kebab(&entry.modality),
        thumbnail_url: entry.thumbnail_url.clone(),
        size_bytes: entry.size_bytes,
        engine_phase: entry.engine_phase,
        trained_words: entry.trained_words.clone(),
        primary_filename_rel,
        written_at: chrono_now_unix(),
    }
}

fn serde_kebab<T: Serialize>(value: &T) -> String {
    // The catalog enums all serialize as bare kebab-case strings, so a
    // round-trip through serde_json gives us the canonical form without
    // hand-coding a `match` per enum.
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn chrono_now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Where a Civitai sidecar lives, given the catalog id (e.g. `cv:8001`).
/// The dir is the sanitized recipe subdir under `models_dir`, mirroring
/// the layout `mold_core::download::fetch_recipe` uses.
pub fn civitai_sidecar_path(models_dir: &Path, catalog_id: &str) -> PathBuf {
    let sanitized = mold_core::download::sanitize_recipe_id(catalog_id);
    models_dir.join(sanitized).join(SIDECAR_FILENAME)
}

/// Persist a sidecar atomically: write to a `*.tmp` then rename. On
/// rename failure the temp file is left in place — a subsequent install
/// will overwrite it.
pub fn write_sidecar(path: &Path, sidecar: &CatalogSidecar) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(sidecar)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

pub fn read_sidecar(path: &Path) -> io::Result<CatalogSidecar> {
    let raw = fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Walk `models_dir` for every `mold-catalog.json` and return parsed
/// sidecars paired with their containing directory. Errors on a single
/// sidecar (parse failure, IO error) are logged and skipped — the
/// picker degrades to "fewer rows" rather than failing the whole list.
///
/// Walks at most two levels deep: `{models_dir}/{sanitized_id}/mold-catalog.json`
/// is the canonical layout. Going deeper risks pulling in stale files
/// from `.pulling` partials or unrelated subtrees.
pub fn walk_sidecars(models_dir: &Path) -> Vec<(PathBuf, CatalogSidecar)> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(models_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let sidecar_path = path.join(SIDECAR_FILENAME);
        if !sidecar_path.is_file() {
            continue;
        }
        match read_sidecar(&sidecar_path) {
            Ok(sidecar) => out.push((path, sidecar)),
            Err(e) => {
                tracing::warn!(
                    target: "catalog.sidecar",
                    path = %sidecar_path.display(),
                    error = %e,
                    "failed to parse sidecar; skipping",
                );
            }
        }
    }
    out
}

/// Returns the absolute path to a sidecar's primary file, when that
/// file exists on disk. `None` indicates the sidecar is stale — the
/// caller should treat the row as not installed.
pub fn primary_path_if_present(sidecar_dir: &Path, sidecar: &CatalogSidecar) -> Option<PathBuf> {
    let abs = sidecar_dir.join(&sidecar.primary_filename_rel);
    abs.is_file().then_some(abs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::{
        Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind,
        LicenseFlags, Modality, Source, TokenKind,
    };
    use crate::families::Family;
    use std::fs;

    fn fixture_entry() -> CatalogEntry {
        CatalogEntry {
            id: CatalogId::from("cv:8001"),
            source: Source::Civitai,
            source_id: "8001".into(),
            name: "Test LoRA".into(),
            author: Some("alice".into()),
            family: Family::Flux,
            family_role: FamilyRole::Finetune,
            sub_family: Some("flux1-d".into()),
            modality: Modality::Image,
            kind: Kind::Lora,
            file_format: FileFormat::Safetensors,
            bundling: Bundling::SingleFile,
            size_bytes: Some(150_000_000),
            download_count: 999,
            rating: Some(4.5),
            likes: 12,
            nsfw: false,
            thumbnail_url: Some("https://example/preview.png".into()),
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec![],
            download_recipe: DownloadRecipe {
                files: vec![],
                needs_token: Some(TokenKind::Civitai),
            },
            engine_phase: 3,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec!["mold trigger".into(), "extra".into()],
        }
    }

    #[test]
    fn sidecar_from_entry_preserves_trigger_words() {
        let entry = fixture_entry();
        let sc = sidecar_from_entry(&entry, "flux/civitai/8001/test.safetensors".into());
        assert_eq!(sc.id, "cv:8001");
        assert_eq!(sc.source, "civitai");
        assert_eq!(sc.kind, "lora");
        assert_eq!(sc.family, "flux");
        assert_eq!(sc.trained_words, vec!["mold trigger", "extra"]);
        assert_eq!(
            sc.primary_filename_rel,
            "flux/civitai/8001/test.safetensors"
        );
    }

    #[test]
    fn write_then_read_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let entry = fixture_entry();
        let sc = sidecar_from_entry(&entry, "test.safetensors".into());
        let path = tmp.path().join(SIDECAR_FILENAME);
        write_sidecar(&path, &sc).unwrap();
        let read = read_sidecar(&path).unwrap();
        assert_eq!(read, sc);
    }

    #[test]
    fn walk_skips_dirs_without_sidecar_and_unreadable_files() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path();

        // Dir with valid sidecar.
        let good = models_dir.join("cv-8001");
        fs::create_dir_all(&good).unwrap();
        let entry = fixture_entry();
        let sc = sidecar_from_entry(&entry, "test.safetensors".into());
        write_sidecar(&good.join(SIDECAR_FILENAME), &sc).unwrap();

        // Dir without sidecar — must NOT appear.
        fs::create_dir_all(models_dir.join("noise")).unwrap();

        // Dir with corrupt sidecar — logged, skipped.
        let bad = models_dir.join("cv-bad");
        fs::create_dir_all(&bad).unwrap();
        fs::write(bad.join(SIDECAR_FILENAME), b"{ not valid json").unwrap();

        let found = walk_sidecars(models_dir);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1.id, "cv:8001");
        assert_eq!(found[0].0, good);
    }

    #[test]
    fn primary_path_returns_some_only_when_file_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let mut sc = sidecar_from_entry(&fixture_entry(), "x.safetensors".into());
        sc.primary_filename_rel = "x.safetensors".into();

        // No file yet → None.
        assert!(primary_path_if_present(tmp.path(), &sc).is_none());

        // Touch the file → Some.
        fs::write(tmp.path().join("x.safetensors"), b"data").unwrap();
        let abs = primary_path_if_present(tmp.path(), &sc).unwrap();
        assert_eq!(abs, tmp.path().join("x.safetensors"));
    }

    #[test]
    fn civitai_sidecar_path_uses_sanitized_recipe_subdir() {
        let path = civitai_sidecar_path(Path::new("/models"), "cv:8001");
        assert_eq!(path, Path::new("/models/cv-8001/mold-catalog.json"));
    }
}
