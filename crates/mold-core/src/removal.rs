//! Shared model-removal core used by `mold rm` (CLI) and the server's
//! `DELETE /api/models/:model` endpoint.
//!
//! Several models share components (T5/CLIP/Qwen encoders, VAEs) under the
//! models dir — this is why manifest SIZE (model weights only) differs from
//! FETCH (total download including shared deps). Removal therefore
//! ref-counts every file path across all installed models and only deletes
//! paths exclusively owned by the model being removed.
//!
//! The split is plan → execute:
//! - [`plan_removal`] classifies the model's files as unique vs shared
//!   (pure given a `Config`, no filesystem mutation).
//! - [`execute_removal`] deletes the unique clean paths plus their hf-hub
//!   cache blobs/snapshot symlinks (space is only freed when the last
//!   hardlink goes), returning what was removed, bytes freed, and any
//!   non-fatal warnings for the caller to surface.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::PathBuf;

use crate::manifest::known_manifests;
use crate::Config;

/// Get file size, returning 0 if the file doesn't exist or can't be read.
pub fn file_size(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Build a map of file_path -> list of model names that reference it.
pub fn build_ref_counts(config: &Config) -> HashMap<String, Vec<String>> {
    let mut refs: HashMap<String, Vec<String>> = HashMap::new();
    for (model_name, model_config) in &config.models {
        for path in model_config.all_file_paths() {
            refs.entry(path).or_default().push(model_name.clone());
        }
    }
    // Include manifest-backed downloaded models that have no config entry.
    // Without this, shared components (VAE, encoders) referenced by another
    // manifest-only install would be deleted when removing a model.
    for manifest in known_manifests() {
        if config.models.contains_key(&manifest.name) {
            continue; // already counted above
        }
        if config.manifest_model_is_downloaded(&manifest.name) {
            for path in config.model_config(&manifest.name).all_file_paths() {
                refs.entry(path).or_default().push(manifest.name.clone());
            }
        }
    }
    refs
}

/// True when `canonical` is removable: it has an explicit config entry or
/// its manifest files are fully present on disk.
pub fn is_model_installed(config: &Config, canonical: &str) -> bool {
    config.models.contains_key(canonical) || config.manifest_model_is_downloaded(canonical)
}

/// Classification of one model's files into exclusively-owned vs shared.
#[derive(Debug, Clone)]
pub struct RemovalPlan {
    /// Canonical model name the plan was built for.
    pub model: String,
    /// `(path, size_bytes)` for files referenced only by this model.
    pub unique_files: Vec<(String, u64)>,
    /// `(path, other_referencing_models)` for files that must be kept.
    pub shared_files: Vec<(String, Vec<String>)>,
    /// Transformer path (used to clean up the model's now-empty directory).
    pub transformer: Option<String>,
}

impl RemovalPlan {
    /// Sum of the unique files' sizes at plan time. Used for the CLI
    /// confirmation prompt and as the freed-bytes fallback when a model
    /// has no hf-cache blobs.
    pub fn total_unique_bytes(&self) -> u64 {
        self.unique_files.iter().map(|(_, size)| size).sum()
    }
}

/// Classify `canonical`'s files as unique (deletable) or shared (kept)
/// by ref-counting paths across every installed model.
pub fn plan_removal(config: &Config, canonical: &str) -> RemovalPlan {
    let ref_counts = build_ref_counts(config);

    // Get the model config — either from config.models or resolved from
    // the manifest registry for manifest-backed models without a config entry.
    let model_config = if let Some(cfg) = config.models.get(canonical) {
        cfg.clone()
    } else {
        config.resolved_model_config(canonical)
    };

    let mut unique_files: Vec<(String, u64)> = Vec::new();
    let mut shared_files: Vec<(String, Vec<String>)> = Vec::new();

    for path in &model_config.all_file_paths() {
        let refs = ref_counts.get(path).cloned().unwrap_or_default();
        let other_refs: Vec<String> = refs.into_iter().filter(|name| name != canonical).collect();

        if other_refs.is_empty() {
            unique_files.push((path.clone(), file_size(path)));
        } else {
            shared_files.push((path.clone(), other_refs));
        }
    }

    RemovalPlan {
        model: canonical.to_string(),
        unique_files,
        shared_files,
        transformer: model_config.transformer.clone(),
    }
}

/// Collect hf-hub cache blob paths for a model's files.
///
/// When `mold pull` downloads files, the hf-hub crate stores blobs at:
///   `<models_dir>/.hf-cache/models--<org>--<repo>/blobs/<sha256>`
/// and creates snapshot symlinks pointing to them. The clean storage paths
/// are then hardlinked from these blobs. This function walks the hf-cache
/// snapshot dirs looking for files whose names match the manifest's
/// `hf_filename` entries, then resolves their symlinks to find the blob paths.
///
/// Only files whose clean paths appear in `unique_clean_paths` are collected —
/// shared components (VAE, T5, CLIP) still referenced by other models must
/// NOT have their blobs deleted.
pub fn collect_hf_cache_blob_paths(
    config: &Config,
    model_name: &str,
    unique_clean_paths: &[(String, u64)],
) -> Vec<PathBuf> {
    let manifest = match crate::manifest::find_manifest(model_name) {
        Some(m) => m,
        None => return Vec::new(),
    };

    let models_dir = config.resolved_models_dir();
    let cache_dir = models_dir.join(".hf-cache");
    if !cache_dir.is_dir() {
        return Vec::new();
    }

    let unique_set: HashSet<String> = unique_clean_paths.iter().map(|(p, _)| p.clone()).collect();

    let mut blobs = Vec::new();

    for file in &manifest.files {
        // Only collect blobs for files whose clean paths are being deleted.
        let clean_path = models_dir
            .join(crate::manifest::storage_path(manifest, file))
            .to_string_lossy()
            .to_string();
        if !unique_set.contains(&clean_path) {
            continue;
        }
        // hf-hub stores repos as models--<org>--<repo>
        let repo_dir_name = format!("models--{}", file.hf_repo.replace('/', "--"));
        let repo_dir = cache_dir.join(&repo_dir_name);
        if !repo_dir.is_dir() {
            continue;
        }

        // Walk snapshots/<rev>/ looking for the filename
        let snapshots_dir = repo_dir.join("snapshots");
        if !snapshots_dir.is_dir() {
            continue;
        }

        // Use the full relative hf_filename (e.g. "text_encoder/model.safetensors")
        // because hf-hub preserves nested paths in snapshot directories.
        if let Ok(revisions) = std::fs::read_dir(&snapshots_dir) {
            for rev in revisions.flatten() {
                let snap_file = rev.path().join(&file.hf_filename);
                // The snapshot entry is a symlink to ../../blobs/<sha>.
                // Resolve it to get the blob path.
                if snap_file.symlink_metadata().is_ok() {
                    if let Ok(blob) = snap_file.canonicalize() {
                        blobs.push(blob);
                    }
                    // Also collect the symlink itself for cleanup
                    blobs.push(snap_file);
                }
            }
        }
    }

    blobs
}

/// Result of [`execute_removal`].
#[derive(Debug, Clone, Default)]
pub struct RemovalOutcome {
    /// Clean storage paths that were actually deleted.
    pub removed: Vec<String>,
    /// Bytes freed on disk. Counted at hf-cache blob removal when blobs
    /// exist (the clean path is a hardlink, so unlinking it alone frees
    /// nothing); otherwise the sum of the unique files' sizes.
    pub freed_bytes: u64,
    /// Non-fatal problems (missing files, failed unlinks). The CLI prints
    /// these with its warning prefix; the server logs them.
    pub warnings: Vec<String>,
}

/// Delete the plan's unique clean paths and their hf-cache counterparts,
/// then clean up the model's now-empty directory. Never touches shared
/// files. Errors are collected as warnings — removal is best-effort per
/// file, matching `mold rm`.
pub fn execute_removal(config: &Config, plan: &RemovalPlan) -> RemovalOutcome {
    let mut outcome = RemovalOutcome::default();

    // When `mold pull` downloads a file, it hardlinks the hf-hub cache
    // blob to a clean storage path under models_dir. Deleting only the
    // clean path leaves the cache blob on disk — the inode's link count
    // doesn't drop to zero, so `du` still reports the same usage.
    //
    // We use the manifest to locate the corresponding hf-cache blob
    // paths and delete those too.
    let hf_cache_blobs = collect_hf_cache_blob_paths(config, &plan.model, &plan.unique_files);

    for (path, _size) in &plan.unique_files {
        match std::fs::remove_file(path) {
            Ok(()) => outcome.removed.push(path.clone()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                outcome.warnings.push(format!("{path} already deleted"));
            }
            Err(e) => {
                outcome
                    .warnings
                    .push(format!("failed to delete {path}: {e}"));
            }
        }
    }

    // Delete hf-cache blobs that were hardlinked to the clean paths.
    // Space is only actually freed when the last hardlink (the blob) is removed,
    // so we count freed bytes here rather than at clean-path deletion.
    for blob_path in &hf_cache_blobs {
        if blob_path.exists() {
            let size = file_size(&blob_path.to_string_lossy());
            match std::fs::remove_file(blob_path) {
                Ok(()) => outcome.freed_bytes += size,
                Err(e) => {
                    outcome.warnings.push(format!(
                        "failed to clean up cache file {}: {e}",
                        blob_path.display()
                    ));
                }
            }
        }
    }

    // For non-manifest models (no hf-cache blobs), the clean path deletion
    // itself freed the space, so use the pre-computed total.
    if hf_cache_blobs.is_empty() {
        outcome.freed_bytes = plan.total_unique_bytes();
    }

    // Clean up empty model-specific directories left behind.
    if let Some(ref t) = plan.transformer {
        if let Some(parent) = std::path::Path::new(t).parent() {
            let _ = std::fs::remove_dir(parent); // only succeeds if empty
        }
    }

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;

    fn tmp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mold-removal-{}-{}",
            label,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn two_model_config(tmp: &std::path::Path) -> Config {
        let shared_vae = tmp.join("shared-vae.safetensors");
        let mut config = Config::default();
        config.models.insert(
            "model-a".into(),
            ModelConfig {
                transformer: Some(tmp.join("unique-a.gguf").to_string_lossy().into_owned()),
                vae: Some(shared_vae.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        config.models.insert(
            "model-b".into(),
            ModelConfig {
                transformer: Some(tmp.join("unique-b.gguf").to_string_lossy().into_owned()),
                vae: Some(shared_vae.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        config
    }

    #[test]
    fn build_ref_counts_counts_shared_files_across_models() {
        let tmp = tmp_dir("refs");
        let config = two_model_config(&tmp);
        let refs = build_ref_counts(&config);
        let shared = tmp.join("shared-vae.safetensors");
        assert_eq!(refs[&shared.to_string_lossy().into_owned()].len(), 2);
        assert_eq!(
            refs[&tmp.join("unique-a.gguf").to_string_lossy().into_owned()].len(),
            1
        );
    }

    #[test]
    fn plan_classifies_unique_and_shared_files() {
        let tmp = tmp_dir("plan");
        let config = two_model_config(&tmp);

        let plan = plan_removal(&config, "model-a");
        assert_eq!(plan.model, "model-a");
        assert_eq!(plan.unique_files.len(), 1, "only the transformer is unique");
        assert!(plan.unique_files[0].0.ends_with("unique-a.gguf"));
        assert_eq!(plan.shared_files.len(), 1, "the VAE is shared");
        assert!(plan.shared_files[0].0.ends_with("shared-vae.safetensors"));
        assert_eq!(plan.shared_files[0].1, vec!["model-b".to_string()]);
    }

    #[test]
    fn execute_removal_deletes_unique_files_and_keeps_shared() {
        let tmp = tmp_dir("exec");
        std::fs::create_dir_all(&tmp).unwrap();
        let unique_a = tmp.join("unique-a.gguf");
        let unique_b = tmp.join("unique-b.gguf");
        let shared_vae = tmp.join("shared-vae.safetensors");
        std::fs::write(&unique_a, b"transformer-a").unwrap(); // 13 bytes
        std::fs::write(&unique_b, b"transformer-b").unwrap();
        std::fs::write(&shared_vae, b"vae").unwrap();

        let config = two_model_config(&tmp);
        let plan = plan_removal(&config, "model-a");
        let outcome = execute_removal(&config, &plan);

        assert!(!unique_a.exists(), "exclusive file must be deleted");
        assert!(shared_vae.exists(), "shared file must survive");
        assert!(unique_b.exists(), "other model's file must survive");
        assert_eq!(
            outcome.removed,
            vec![unique_a.to_string_lossy().into_owned()]
        );
        assert_eq!(outcome.freed_bytes, 13, "freed = unique file size");
        assert!(outcome.warnings.is_empty(), "got: {:?}", outcome.warnings);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn execute_removal_reports_missing_files_as_warnings() {
        let tmp = tmp_dir("missing");
        // Files never created — every unique path is already gone.
        let config = two_model_config(&tmp);
        let plan = plan_removal(&config, "model-a");
        let outcome = execute_removal(&config, &plan);

        assert!(outcome.removed.is_empty());
        assert_eq!(outcome.warnings.len(), 1);
        assert!(
            outcome.warnings[0].ends_with("already deleted"),
            "got: {:?}",
            outcome.warnings
        );
    }
}
