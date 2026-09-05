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
    /// `None` means an older sidecar did not record the classification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nsfw: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_url: Option<String>,
    pub thumbnail_url: Option<String>,
    pub size_bytes: Option<u64>,
    #[serde(default = "default_true")]
    pub supported: bool,
    /// Trigger phrases for LoRA entries — what the picker renders as
    /// click-to-insert chips. Empty for non-LoRA entries.
    #[serde(default)]
    pub trained_words: Vec<String>,
    /// Filename of the primary safetensors / GGUF file, relative to
    /// the sidecar's own directory. The picker resolves the absolute
    /// path via `sidecar_dir.join(primary_filename_rel)` so the layout
    /// stays portable across `MOLD_HOME` moves.
    pub primary_filename_rel: String,
    /// Expected size of the primary file alone. `size_bytes` above is the
    /// whole entry (for a two-expert pair that is the sum of both
    /// experts), so pair sidecars record the primary's own size here for
    /// the truncation check. Absent on single-file sidecars, where
    /// `size_bytes` IS the primary size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_size_bytes: Option<u64>,
    /// The low-noise expert of a two-expert checkpoint pair (Wan 2.2
    /// A14B), relative to the sidecar's directory like
    /// `primary_filename_rel` (which is then the high-noise expert).
    /// When declared, the install is complete only when BOTH files are
    /// present — a half-pair is never reported as installed. Absent on
    /// every single-transformer install and on pre-pairing sidecars.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_noise_filename_rel: Option<String>,
    /// Expected size of the low-noise expert file, for the same
    /// truncation check `size_bytes` provides the primary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_noise_size_bytes: Option<u64>,
    pub written_at: i64,
}

/// Apply the core activation policy before locally persisted catalog metadata
/// can re-enter inventory or engine resolution without a live lookup.
pub fn require_sidecar_activation(
    sidecar: &CatalogSidecar,
) -> Result<(), mold_core::ModelActivationError> {
    let family = Some(sidecar.family.as_str());
    for identity in [
        sidecar.id.as_str(),
        sidecar.source_id.as_str(),
        sidecar.name.as_str(),
    ] {
        mold_core::require_model_activation(identity, family)?;
    }
    if let Some(sub_family) = sidecar.sub_family.as_deref() {
        mold_core::require_model_activation(sub_family, family)?;
    }
    if let Some(page_url) = sidecar.page_url.as_deref() {
        mold_core::require_model_activation(page_url, family)?;
    }
    mold_core::require_model_activation(&sidecar.primary_filename_rel, family)?;
    if let Some(low_rel) = sidecar.low_noise_filename_rel.as_deref() {
        mold_core::require_model_activation(low_rel, family)?;
    }
    Ok(())
}

/// Apply activation policy to persisted sidecar metadata and its concrete
/// primary artifact without treating the configured models root as identity.
///
/// The sidecar directory is part of the artifact identity even when the
/// portable `primary_filename_rel` itself is neutral. This closes the case
/// where an opaque sidecar points at `MiniMax-H3/weights.safetensors`, while
/// still allowing an ordinary artifact when the operator's `models_dir` (or
/// an ancestor such as `MOLD_HOME`) happens to carry that name.
pub fn require_sidecar_artifact_activation(
    models_dir: &Path,
    sidecar_dir: &Path,
    sidecar: &CatalogSidecar,
) -> Result<(), mold_core::ModelActivationError> {
    require_sidecar_activation(sidecar)?;
    if let Some(primary) = primary_path(sidecar_dir, sidecar) {
        mold_core::require_model_artifact_activation(
            &primary,
            Some(models_dir),
            Some(sidecar.family.as_str()),
        )?;
    }
    if let Some(low) = low_noise_path(sidecar_dir, sidecar) {
        mold_core::require_model_artifact_activation(
            &low,
            Some(models_dir),
            Some(sidecar.family.as_str()),
        )?;
    }
    Ok(())
}

/// Reject a matching installed sidecar before callers decide whether to use it
/// or fall back to a live lookup. This prevents persisted metadata from being
/// treated as a cache miss that silently triggers network access.
pub fn require_installed_sidecar_activation(
    models_dir: &Path,
    catalog_id: &str,
) -> Result<(), mold_core::ModelActivationError> {
    if let Some((sidecar_dir, sidecar)) = walk_sidecars(models_dir)
        .into_iter()
        .find(|(_, sidecar)| sidecar.id == catalog_id)
    {
        require_sidecar_artifact_activation(models_dir, &sidecar_dir, &sidecar)?;
    }
    Ok(())
}

fn default_true() -> bool {
    true
}

/// Build a sidecar from a normalized [`CatalogEntry`] plus the resolved
/// primary file dest (relative to the sidecar's directory). Civitai
/// recipes always have at least one file, but we accept a 0-length
/// `primary_filename_rel` and let the caller decide whether that's
/// fatal — the picker simply won't show the row.
pub fn sidecar_from_entry(entry: &CatalogEntry, primary_filename_rel: String) -> CatalogSidecar {
    // A two-expert (Wan 2.2 A14B) recipe records its low-noise half so the
    // installed sidecar can resolve the pair without a live lookup, and so
    // completeness checks can refuse a half-pair. Derived here rather than
    // passed in so every sidecar writer (CLI pull, server download route,
    // server intent install) stays in agreement by construction.
    let (author, name) = match entry.source_id.split_once('/') {
        Some((author, name)) => (author, name),
        None => ("", entry.source_id.as_str()),
    };
    let low_noise = entry
        .download_recipe
        .files
        .iter()
        .find(|file| file.role == Some(crate::entry::RecipeFileRole::LowNoiseTransformer));
    let primary_size_bytes = low_noise.is_some().then(|| {
        entry
            .download_recipe
            .files
            .iter()
            .find(|file| file.role.is_none())
            .and_then(|file| file.size_bytes)
    });
    CatalogSidecar {
        primary_size_bytes: primary_size_bytes.flatten(),
        low_noise_filename_rel: low_noise.map(|file| {
            crate::entry::render_recipe_dest(&file.dest, entry.family.as_str(), author, name)
        }),
        low_noise_size_bytes: low_noise.and_then(|file| file.size_bytes),
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
        nsfw: Some(entry.nsfw),
        description: entry.description.clone(),
        tags: entry.tags.clone(),
        license: entry.license.clone(),
        page_url: entry.page_url.clone(),
        thumbnail_url: entry.thumbnail_url.clone(),
        size_bytes: entry.size_bytes,
        supported: entry.supported,
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

/// True when a sidecar tagged `kind = "checkpoint"` actually points its
/// primary file at an auxiliary asset (a text encoder / TE shard) rather
/// than a real diffusion checkpoint. Stale or mislabeled sidecars like
/// this must not be surfaced as loadable checkpoints.
///
/// The pattern set is the superset of the historical CLI + server guards:
/// diffusers `text_encoder/` directories, the `te_` shard-name convention,
/// and the `_txt.` / `-txt.` filename suffixes some Civitai TE files use.
pub fn primary_looks_like_auxiliary(sidecar: &CatalogSidecar) -> bool {
    let rel = sidecar.primary_filename_rel.to_ascii_lowercase();
    rel.contains("/text_encoder/")
        || rel.contains("text_encoder")
        || rel.contains("te_")
        || rel.contains("_txt.")
        || rel.contains("-txt.")
}

/// Resolve the primary path recorded by a sidecar without consulting disk.
///
/// Sidecars are local metadata, not an authority to escape their own install
/// directory. Accept only non-empty, strictly normal relative components so
/// every consumer (inventory, placement preview, admission, and worker load)
/// shares the same containment boundary.
pub fn primary_path(sidecar_dir: &Path, sidecar: &CatalogSidecar) -> Option<PathBuf> {
    let relative = Path::new(&sidecar.primary_filename_rel);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return None;
    }
    Some(sidecar_dir.join(relative))
}

/// Resolve the low-noise expert path a pair sidecar declares, with the
/// same containment rules as [`primary_path`]. `None` for single-file
/// sidecars and for declarations that would escape the install directory.
pub fn low_noise_path(sidecar_dir: &Path, sidecar: &CatalogSidecar) -> Option<PathBuf> {
    let relative = Path::new(sidecar.low_noise_filename_rel.as_deref()?);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return None;
    }
    Some(sidecar_dir.join(relative))
}

/// One contained-and-complete file check: exists, resolves inside the
/// install directory, and is verified by sha marker or declared size.
fn contained_complete_file(
    sidecar_dir: &Path,
    abs: PathBuf,
    expected_size: Option<u64>,
) -> Option<PathBuf> {
    if !abs.is_file() {
        return None;
    }
    let canonical_dir = std::fs::canonicalize(sidecar_dir).ok()?;
    let canonical = std::fs::canonicalize(&abs).ok()?;
    if !canonical.starts_with(&canonical_dir) {
        return None;
    }
    if mold_core::download::has_sha256_marker(&abs) {
        return Some(abs);
    }
    if let Some(expected) = expected_size {
        let actual = abs.metadata().ok()?.len();
        if actual != expected {
            return None;
        }
    }
    Some(abs)
}

/// Returns the absolute path to a sidecar's primary file, when the install
/// is complete enough to trust. `None` indicates the sidecar is stale or a
/// download is unfinished — the caller should treat the row as not
/// installed.
///
/// For a two-expert pair sidecar (Wan 2.2 A14B) "complete" means BOTH
/// declared experts: a present high-noise file with a missing low-noise
/// counterpart is a half-pair that must never be reported installed —
/// re-running the download resumes just the missing half.
pub fn primary_path_if_present(sidecar_dir: &Path, sidecar: &CatalogSidecar) -> Option<PathBuf> {
    if install_pull_in_progress(sidecar_dir, sidecar) {
        return None;
    }
    if sidecar.low_noise_filename_rel.is_some() {
        low_noise_file_if_complete(sidecar_dir, sidecar)?;
    }
    primary_file_if_complete(sidecar_dir, sidecar)
}

/// True while a pull for this install is in flight — the files on disk are
/// not authoritative yet.
pub fn install_pull_in_progress(sidecar_dir: &Path, sidecar: &CatalogSidecar) -> bool {
    sidecar_dir.parent().is_some_and(|models_dir| {
        mold_core::download::pulling_marker_path_in(models_dir, &sidecar.id).exists()
    })
}

/// Per-expert presence: whether the sidecar's primary file itself is
/// contained and complete, ignoring the low-noise counterpart and any
/// in-flight pull. Component diagnostics use this so a pair install with
/// one half missing names the truly missing file; whole-install
/// completeness remains [`primary_path_if_present`].
pub fn primary_file_if_complete(sidecar_dir: &Path, sidecar: &CatalogSidecar) -> Option<PathBuf> {
    let abs = primary_path(sidecar_dir, sidecar)?;
    // `size_bytes` is the whole entry; a pair sidecar records the
    // primary's own size separately.
    let expected = sidecar.primary_size_bytes.or(sidecar.size_bytes);
    contained_complete_file(sidecar_dir, abs, expected)
}

/// Per-expert presence for the declared low-noise counterpart, with the
/// same rules as [`primary_file_if_complete`]. `None` for single-file
/// sidecars.
pub fn low_noise_file_if_complete(sidecar_dir: &Path, sidecar: &CatalogSidecar) -> Option<PathBuf> {
    let abs = low_noise_path(sidecar_dir, sidecar)?;
    contained_complete_file(sidecar_dir, abs, sidecar.low_noise_size_bytes)
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
            supported: true,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec!["mold trigger".into(), "extra".into()],
            page_url: None,
        }
    }

    #[test]
    fn sidecar_from_entry_preserves_presentation_metadata() {
        let mut entry = fixture_entry();
        entry.nsfw = true;
        entry.description = Some("A cinematic portrait adapter.".into());
        entry.tags = vec!["portrait".into(), "cinematic".into()];
        entry.license = Some("CreativeML Open RAIL-M".into());
        entry.page_url = Some("https://civitai.com/models/8001".into());
        let sc = sidecar_from_entry(&entry, "flux/civitai/8001/test.safetensors".into());
        assert_eq!(sc.id, "cv:8001");
        assert_eq!(sc.source, "civitai");
        assert_eq!(sc.kind, "lora");
        assert_eq!(sc.family, "flux");
        assert_eq!(sc.nsfw, Some(true));
        assert_eq!(
            sc.description.as_deref(),
            Some("A cinematic portrait adapter.")
        );
        assert_eq!(sc.tags, vec!["portrait", "cinematic"]);
        assert_eq!(sc.license.as_deref(), Some("CreativeML Open RAIL-M"));
        assert_eq!(
            sc.page_url.as_deref(),
            Some("https://civitai.com/models/8001")
        );
        assert_eq!(sc.trained_words, vec!["mold trigger", "extra"]);
        assert_eq!(
            sc.primary_filename_rel,
            "flux/civitai/8001/test.safetensors"
        );
    }

    #[test]
    fn sidecar_preserves_qwen_image_edit_as_its_own_runtime_family() {
        let mut entry = fixture_entry();
        entry.family = Family::QwenImageEdit;
        entry.name = "QwenImageEdit2511 community".into();
        let sidecar = sidecar_from_entry(
            &entry,
            "qwen-image-edit/civitai/8001/model.safetensors".into(),
        );

        assert_eq!(sidecar.family, "qwen-image-edit");
        assert_eq!(
            Family::from_str(&sidecar.family).unwrap(),
            Family::QwenImageEdit
        );
    }

    #[test]
    fn installed_h3_sidecar_is_rejected_as_metadata_not_treated_as_a_cache_miss() {
        let root = tempfile::tempdir().unwrap();
        let mut entry = fixture_entry();
        entry.id = CatalogId::from("cv:42");
        entry.source_id = "42".into();
        entry.name = "MiniMax H3 Ref2VA".into();
        let sidecar = sidecar_from_entry(&entry, "weights.safetensors".into());
        let path = civitai_sidecar_path(root.path(), entry.id.as_str());
        write_sidecar(&path, &sidecar).unwrap();

        let error = require_installed_sidecar_activation(root.path(), entry.id.as_str())
            .expect_err("persisted H3 metadata must fail closed");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[test]
    fn neutral_sidecar_with_h3_primary_filename_is_rejected() {
        let mut sidecar = sidecar_from_entry(&fixture_entry(), "MiniMaxH3.safetensors".into());
        sidecar.id = "cv:42".into();
        sidecar.source_id = "42".into();
        sidecar.name = "renamed checkpoint".into();
        sidecar.family = "custom".into();

        let error = require_sidecar_activation(&sidecar).unwrap_err();
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[test]
    fn sidecar_artifact_policy_ignores_named_root_but_rejects_nested_h3_directory() {
        let root = tempfile::tempdir().unwrap();
        let models_dir = root.path().join("mold-uat/minimax-h3/models");
        let ordinary_dir = models_dir.join("cv-42");
        let h3_dir = models_dir.join("MiniMax-H3");
        let mut sidecar = sidecar_from_entry(&fixture_entry(), "weights.safetensors".into());
        sidecar.id = "cv:42".into();
        sidecar.source_id = "42".into();
        sidecar.name = "renamed checkpoint".into();
        sidecar.family = "custom".into();

        require_sidecar_artifact_activation(&models_dir, &ordinary_dir, &sidecar)
            .expect("the configured root is storage placement, not model identity");
        let error = require_sidecar_artifact_activation(&models_dir, &h3_dir, &sidecar)
            .expect_err("a nested H3 artifact directory must fail closed");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[cfg(unix)]
    #[test]
    fn primary_path_if_present_rejects_symlink_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let install = tmp.path().join("install");
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&install).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("model.safetensors"), b"outside").unwrap();
        std::os::unix::fs::symlink(&outside, install.join("linked")).unwrap();
        let sc = sidecar_from_entry(&fixture_entry(), "linked/model.safetensors".to_string());

        assert_eq!(
            primary_path(&install, &sc),
            Some(install.join("linked/model.safetensors"))
        );
        assert!(primary_path_if_present(&install, &sc).is_none());
    }

    #[test]
    fn older_sidecars_leave_new_presentation_metadata_unknown() {
        let entry = fixture_entry();
        let sidecar = sidecar_from_entry(&entry, "test.safetensors".into());
        let mut value = serde_json::to_value(sidecar).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("nsfw");
        object.remove("description");
        object.remove("tags");
        object.remove("license");
        object.remove("page_url");

        let read: CatalogSidecar = serde_json::from_value(value).unwrap();
        assert_eq!(read.nsfw, None);
        assert_eq!(read.description, None);
        assert!(read.tags.is_empty());
        assert_eq!(read.license, None);
        assert_eq!(read.page_url, None);
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
    fn primary_path_returns_some_only_when_file_exists_at_declared_size() {
        let tmp = tempfile::tempdir().unwrap();
        let mut sc = sidecar_from_entry(&fixture_entry(), "x.safetensors".into());
        sc.primary_filename_rel = "x.safetensors".into();
        sc.size_bytes = Some(4);

        // No file yet → None.
        assert!(primary_path_if_present(tmp.path(), &sc).is_none());

        // Wrong size → still None.
        fs::write(tmp.path().join("x.safetensors"), b"dat").unwrap();
        assert!(primary_path_if_present(tmp.path(), &sc).is_none());

        // Declared size → Some.
        fs::write(tmp.path().join("x.safetensors"), b"data").unwrap();
        let abs = primary_path_if_present(tmp.path(), &sc).unwrap();
        assert_eq!(abs, tmp.path().join("x.safetensors"));
    }

    #[test]
    fn primary_path_rejects_absolute_parent_and_non_normal_components() {
        let tmp = tempfile::tempdir().unwrap();
        let mut sc = sidecar_from_entry(&fixture_entry(), "x.safetensors".into());
        for invalid in [
            "",
            "/tmp/escape.safetensors",
            "../escape.safetensors",
            "nested/../escape.safetensors",
            "./x.safetensors",
        ] {
            sc.primary_filename_rel = invalid.into();
            assert!(
                primary_path(tmp.path(), &sc).is_none(),
                "accepted invalid sidecar primary path {invalid:?}"
            );
            assert!(primary_path_if_present(tmp.path(), &sc).is_none());
        }

        sc.primary_filename_rel = "nested/x.safetensors".into();
        assert_eq!(
            primary_path(tmp.path(), &sc),
            Some(tmp.path().join("nested/x.safetensors"))
        );
    }

    #[test]
    fn primary_path_accepts_marker_when_declared_size_is_stale() {
        let tmp = tempfile::tempdir().unwrap();
        let mut sc = sidecar_from_entry(&fixture_entry(), "x.safetensors".into());
        sc.primary_filename_rel = "x.safetensors".into();
        sc.size_bytes = Some(4);
        let primary = tmp.path().join("x.safetensors");
        fs::write(&primary, b"larger than catalog size").unwrap();
        mold_core::download::write_sha256_marker(&primary, "deadbeef").unwrap();

        let abs = primary_path_if_present(tmp.path(), &sc).unwrap();
        assert_eq!(abs, primary);
    }

    #[test]
    fn primary_path_returns_none_when_pulling_marker_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar_dir = tmp.path().join("cv-8001");
        fs::create_dir_all(&sidecar_dir).unwrap();
        let mut sc = sidecar_from_entry(&fixture_entry(), "x.safetensors".into());
        sc.primary_filename_rel = "x.safetensors".into();
        sc.size_bytes = Some(4);
        fs::write(sidecar_dir.join("x.safetensors"), b"data").unwrap();
        fs::write(
            mold_core::download::pulling_marker_path_in(tmp.path(), &sc.id),
            b"pulling",
        )
        .unwrap();

        assert!(primary_path_if_present(&sidecar_dir, &sc).is_none());
    }

    #[test]
    fn civitai_sidecar_path_uses_sanitized_recipe_subdir() {
        let path = civitai_sidecar_path(Path::new("/models"), "cv:8001");
        assert_eq!(path, Path::new("/models/cv-8001/mold-catalog.json"));
    }
}
