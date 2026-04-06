use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::manifest::resolve_model_name;
use mold_core::Config;

use super::cleanup::{
    build_ref_counts, clean_hf_cache, clean_orphaned_shared_files, clean_stale_pull_markers,
    file_size,
};
#[cfg(test)]
use super::cleanup::{remove_empty_dirs_recursive, remove_orphaned_files_recursive, HfCacheIndex};
use crate::theme;
use crate::ui::format_bytes;
use crate::AlreadyReported;

/// Provide completions for installed model names (config + manifest-backed).
pub fn complete_installed_model_name() -> Vec<CompletionCandidate> {
    let config = Config::load_or_default();
    let mut names: HashSet<String> = config.models.keys().cloned().collect();
    for manifest in mold_core::manifest::known_manifests() {
        if config.manifest_model_is_downloaded(&manifest.name) {
            names.insert(manifest.name.clone());
        }
    }
    names.into_iter().map(CompletionCandidate::new).collect()
}

/// Collect hf-hub cache blob paths for a model's files.
///
/// When `mold pull` downloads files, the hf-hub crate stores blobs at:
///   `<models_dir>/.hf-cache/models--<org>--<repo>/blobs/<sha256>`
/// and creates snapshot symlinks pointing to them. The clean storage paths
/// are then hardlinked from these blobs. This function walks the hf-cache
/// snapshot dirs looking for files whose names match the manifest's
/// `hf_filename` entries, then resolves their symlinks to find the blob paths.
pub(super) fn collect_hf_cache_blob_paths(
    model_name: &str,
    unique_clean_paths: &[(String, u64)],
) -> Vec<PathBuf> {
    let manifest = match mold_core::manifest::find_manifest(model_name) {
        Some(m) => m,
        None => return Vec::new(),
    };

    let config = Config::load_or_default();
    let models_dir = config.resolved_models_dir();
    let cache_dir = models_dir.join(".hf-cache");
    if !cache_dir.is_dir() {
        return Vec::new();
    }

    // Build a set of unique clean paths to restrict which manifest files we
    // collect cache blobs for. Shared components (VAE, T5, CLIP) that are
    // still referenced by other models must NOT have their blobs deleted.
    let unique_set: HashSet<String> = unique_clean_paths.iter().map(|(p, _)| p.clone()).collect();

    let mut blobs = Vec::new();

    for file in &manifest.files {
        // Only collect blobs for files whose clean paths are being deleted.
        let clean_path = models_dir
            .join(mold_core::manifest::storage_path(manifest, file))
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

pub async fn run(models: &[String], force: bool) -> Result<()> {
    let mut config = Config::load_or_default();
    let mut any_error = false;

    for model_arg in models {
        let canonical = resolve_model_name(model_arg);

        // Check if the model is installed (config entry or manifest-backed download)
        let in_config = config.models.contains_key(&canonical);
        let manifest_downloaded = config.manifest_model_is_downloaded(&canonical);
        if !in_config && !manifest_downloaded {
            eprintln!(
                "{} {} is not installed",
                theme::prefix_error(),
                canonical.bold()
            );
            any_error = true;
            continue;
        }

        // Build reference counts across all installed models
        let ref_counts = build_ref_counts(&config);

        // Get the model config — either from config.models or resolved from
        // the manifest registry for manifest-backed models without a config entry.
        let model_config = if let Some(cfg) = config.models.get(&canonical) {
            cfg.clone()
        } else {
            config.resolved_model_config(&canonical)
        };
        let all_paths = model_config.all_file_paths();

        // Classify files as unique (only this model) or shared
        let mut unique_files: Vec<(String, u64)> = Vec::new();
        let mut shared_files: Vec<(String, Vec<String>)> = Vec::new();

        for path in &all_paths {
            let refs = ref_counts.get(path).cloned().unwrap_or_default();
            let other_refs: Vec<String> =
                refs.into_iter().filter(|name| name != &canonical).collect();

            if other_refs.is_empty() {
                unique_files.push((path.clone(), file_size(path)));
            } else {
                shared_files.push((path.clone(), other_refs));
            }
        }

        let total_freed: u64 = unique_files.iter().map(|(_, size)| size).sum();

        // Display summary
        println!("{}", canonical.bold());
        for (path, size) in &unique_files {
            let filename = std::path::Path::new(path)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            println!(
                "  {}   {} ({})",
                "Delete:".red(),
                filename,
                format_bytes(*size)
            );
        }
        for (path, refs) in &shared_files {
            let filename = std::path::Path::new(path)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            println!(
                "  {}   {} (used by {})",
                "Shared:".yellow(),
                filename,
                refs.join(", ")
            );
        }

        if unique_files.is_empty() && shared_files.is_empty() {
            println!("  {}", "(no files configured)".dimmed());
        }

        println!();

        // Confirm
        if !force {
            print!(
                "Remove {}? This will free {}. [y/N] ",
                canonical.bold(),
                format_bytes(total_freed)
            );
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Skipped {}", canonical);
                continue;
            }
        }

        // Delete unique files and their hf-cache counterparts.
        //
        // When `mold pull` downloads a file, it hardlinks the hf-hub cache
        // blob to a clean storage path under models_dir. Deleting only the
        // clean path leaves the cache blob on disk — the inode's link count
        // doesn't drop to zero, so `du` still reports the same usage.
        //
        // We use the manifest to locate the corresponding hf-cache blob
        // paths and delete those too.
        let mut freed: u64 = 0;
        let hf_cache_blobs = collect_hf_cache_blob_paths(&canonical, &unique_files);

        for (path, _size) in &unique_files {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    eprintln!("{} {} already deleted", theme::prefix_warning(), path);
                }
                Err(e) => {
                    eprintln!(
                        "{} failed to delete {}: {}",
                        theme::prefix_warning(),
                        path,
                        e
                    );
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
                    Ok(()) => freed += size,
                    Err(e) => {
                        eprintln!(
                            "{} failed to clean up cache file {}: {}",
                            theme::prefix_warning(),
                            blob_path.display(),
                            e
                        );
                    }
                }
            }
        }

        // For non-manifest models (no hf-cache blobs), the clean path deletion
        // itself freed the space, so use the pre-computed total.
        if hf_cache_blobs.is_empty() {
            freed = total_freed;
        }

        // Clean up empty model-specific directories left behind.
        if let Some(ref t) = model_config.transformer {
            if let Some(parent) = std::path::Path::new(t).parent() {
                let _ = std::fs::remove_dir(parent); // only succeeds if empty
            }
        }

        // Remove from config and clean up pull marker
        config.remove_model(&canonical);
        mold_core::download::remove_pulling_marker(&canonical);

        // Handle default_model update
        if config.default_model == canonical {
            // Check config entries first, then manifest-backed installed models.
            let new_default = config
                .models
                .keys()
                .min()
                .cloned()
                .or_else(|| {
                    mold_core::manifest::known_manifests()
                        .iter()
                        .filter(|m| !m.is_utility() && !m.is_upscaler())
                        .filter(|m| m.name != canonical)
                        .find(|m| config.manifest_model_is_downloaded(&m.name))
                        .map(|m| m.name.clone())
                })
                .unwrap_or_else(|| "flux2-klein".to_string());
            let has_remaining = !config.models.is_empty()
                || mold_core::manifest::known_manifests().iter().any(|m| {
                    !m.is_utility()
                        && !m.is_upscaler()
                        && m.name != canonical
                        && config.manifest_model_is_downloaded(&m.name)
                });
            if !has_remaining {
                eprintln!(
                    "{} default model reset to {}",
                    theme::prefix_note(),
                    "flux2-klein".bold()
                );
            } else {
                eprintln!(
                    "{} default model changed to {}",
                    theme::prefix_note(),
                    new_default.bold()
                );
            }
            config.default_model = new_default;
        }

        config.save()?;

        println!(
            "Removed {} (freed {})",
            canonical.bold(),
            format_bytes(freed)
        );
    }

    // Remove abandoned download markers before scanning for live references or
    // cleaning hf-cache transient files.
    clean_stale_pull_markers(&config);

    // Clean up orphaned shared files and empty shared directories.
    clean_orphaned_shared_files(&config);

    // Clean up dangling hf-cache entries (symlinks to deleted blobs,
    // transient lock/partial files in inactive repos, empty snapshot dirs,
    // empty repo dirs) once after all removals.
    clean_hf_cache(&config);

    if any_error {
        return Err(AlreadyReported.into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use super::*;
    use mold_core::ModelConfig;

    #[test]
    fn all_file_paths_collects_all() {
        let cfg = ModelConfig {
            transformer: Some("/a/transformer.gguf".into()),
            vae: Some("/a/vae.safetensors".into()),
            t5_encoder: Some("/a/t5.safetensors".into()),
            clip_encoder: Some("/a/clip.safetensors".into()),
            t5_tokenizer: Some("/a/t5.tokenizer.json".into()),
            clip_tokenizer: Some("/a/clip.tokenizer.json".into()),
            transformer_shards: Some(vec!["/a/shard1.safetensors".into()]),
            text_encoder_files: Some(vec!["/a/enc1.safetensors".into()]),
            text_tokenizer: Some("/a/text.tokenizer.json".into()),
            ..Default::default()
        };
        let paths = cfg.all_file_paths();
        assert_eq!(paths.len(), 9);
    }

    #[test]
    fn ref_counts_shared_file() {
        let mut config = Config::default();
        config.models.insert(
            "model-a".into(),
            ModelConfig {
                transformer: Some("/unique-a.gguf".into()),
                vae: Some("/shared-vae.safetensors".into()),
                ..Default::default()
            },
        );
        config.models.insert(
            "model-b".into(),
            ModelConfig {
                transformer: Some("/unique-b.gguf".into()),
                vae: Some("/shared-vae.safetensors".into()),
                ..Default::default()
            },
        );

        let refs = build_ref_counts(&config);
        assert_eq!(refs["/shared-vae.safetensors"].len(), 2);
        assert_eq!(refs["/unique-a.gguf"].len(), 1);
        assert_eq!(refs["/unique-b.gguf"].len(), 1);
    }

    #[test]
    fn complete_returns_installed_only() {
        // This just verifies it doesn't panic — actual model list depends on user config
        let candidates = complete_installed_model_name();
        // May be empty if no models installed; that's fine
        assert!(candidates.len() < 1000);
    }

    #[cfg(unix)]
    #[test]
    fn hf_cache_blob_cleanup_via_symlink_resolution() {
        // Simulate the hf-cache layout: blob + snapshot symlink + clean hardlink.
        // Verify that collect_hf_cache_blob_paths-style logic resolves correctly
        // and that deleting both the clean path and blob frees all data.
        let tmp = std::env::temp_dir().join(format!(
            "mold-rm-cache-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let blobs_dir = tmp.join("blobs");
        let snap_dir = tmp.join("snapshots").join("abc123");
        let clean_dir = tmp.join("clean");

        std::fs::create_dir_all(&blobs_dir).unwrap();
        std::fs::create_dir_all(&snap_dir).unwrap();
        std::fs::create_dir_all(&clean_dir).unwrap();

        // 1. Create blob (hf-cache stores actual data here)
        let blob = blobs_dir.join("deadbeef1234");
        std::fs::write(&blob, b"model weights data").unwrap();

        // 2. Snapshot symlink: snapshots/abc123/model.safetensors → ../../blobs/deadbeef1234
        let snap_link = snap_dir.join("model.safetensors");
        std::os::unix::fs::symlink(&blob, &snap_link).unwrap();

        // 3. Clean path: hardlinked from the blob (what mold pull does)
        let clean_path = clean_dir.join("model.safetensors");
        std::fs::hard_link(&blob, &clean_path).unwrap();

        // Resolve the blob via the snapshot symlink (this is what collect_hf_cache_blob_paths does)
        let resolved_blob = snap_link.canonicalize().unwrap();
        assert_eq!(resolved_blob, blob.canonicalize().unwrap());

        // Delete the clean path (what mold rm does for the config entry)
        std::fs::remove_file(&clean_path).unwrap();
        assert!(!clean_path.exists());
        // Blob still exists (hardlink from hf-cache)
        assert!(blob.exists());

        // Delete the blob (what the fix does via hf-cache cleanup)
        std::fs::remove_file(&resolved_blob).unwrap();
        assert!(!blob.exists(), "blob should be deleted after cleanup");

        // Delete the dangling symlink
        std::fs::remove_file(&snap_link).unwrap();
        assert!(!snap_link.exists());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn make_tmp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mold-rm-{}-{}",
            label,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn orphaned_gguf_cache_files_deleted_when_unreferenced() {
        // Simulate shared/t5-gguf/ with an orphaned GGUF encoder.
        let tmp = make_tmp_dir("orphan");
        let shared = tmp.join("shared");
        let t5_gguf_dir = shared.join("t5-gguf");
        std::fs::create_dir_all(&t5_gguf_dir).unwrap();

        // Orphaned file — not referenced by any config entry
        let orphan = t5_gguf_dir.join("t5-v1_1-xxl-encoder-Q8_0.gguf");
        std::fs::write(&orphan, b"orphaned encoder data").unwrap();

        let referenced_set: HashSet<String> = HashSet::new();

        // Run the orphan removal logic on the scoped cache dir
        let cfg = Config::default();
        let hf_index = HfCacheIndex::build(&cfg);
        let (count, bytes) =
            remove_orphaned_files_recursive(&t5_gguf_dir, &referenced_set, &hf_index);
        assert_eq!(count, 1, "should delete exactly 1 orphaned file");
        assert_eq!(bytes, 21, "should report correct bytes freed"); // b"orphaned encoder data" = 21 bytes

        assert!(!orphan.exists(), "orphaned T5 GGUF should be deleted");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn shared_cleanup_removes_unreferenced_manifest_shared_dirs() {
        let tmp = make_tmp_dir("shared-scope");
        let shared = tmp.join("shared");
        let flux_dir = shared.join("flux");
        std::fs::create_dir_all(&flux_dir).unwrap();

        let flux_vae = flux_dir.join("ae.safetensors");
        std::fs::write(&flux_vae, b"vae data").unwrap();

        let referenced: HashSet<String> = HashSet::new();
        let cfg = Config::default();
        let hf_index = HfCacheIndex::build(&cfg);
        remove_orphaned_files_recursive(&shared, &referenced, &hf_index);
        remove_empty_dirs_recursive(&shared);

        assert!(
            !flux_vae.exists(),
            "unreferenced file in shared/flux/ should be deleted"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn empty_shared_dirs_cleaned_after_all_files_removed() {
        let tmp = make_tmp_dir("empty-dirs");
        let shared = tmp.join("shared");
        let t5_dir = shared.join("t5-gguf");
        let qwen3_dir = shared.join("qwen3-gguf");
        let nested = shared.join("flux").join("subdir");
        std::fs::create_dir_all(&t5_dir).unwrap();
        std::fs::create_dir_all(&qwen3_dir).unwrap();
        std::fs::create_dir_all(&nested).unwrap();

        // All empty — simulate post-deletion state
        let referenced: HashSet<String> = HashSet::new();
        let cfg = Config::default();
        let hf_index = HfCacheIndex::build(&cfg);
        remove_orphaned_files_recursive(&shared, &referenced, &hf_index);
        remove_empty_dirs_recursive(&shared);
        let _ = std::fs::remove_dir(&shared);

        assert!(!t5_dir.exists(), "empty t5-gguf should be removed");
        assert!(!qwen3_dir.exists(), "empty qwen3-gguf should be removed");
        assert!(!nested.exists(), "nested empty dir should be removed");
        assert!(
            !shared.exists(),
            "shared/ itself should be removed when empty"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn orphan_cleanup_with_multiple_orphaned_files() {
        // Multiple orphaned files across different shared subdirectories
        let tmp = make_tmp_dir("multi-orphan");
        let shared = tmp.join("shared");
        let t5_dir = shared.join("t5-gguf");
        let qwen3_dir = shared.join("qwen3-gguf");
        std::fs::create_dir_all(&t5_dir).unwrap();
        std::fs::create_dir_all(&qwen3_dir).unwrap();

        let orphan_t5 = t5_dir.join("t5-v1_1-xxl-encoder-Q4_0.gguf");
        let orphan_qwen = qwen3_dir.join("Qwen_3_4b-Q8_0.gguf");
        std::fs::write(&orphan_t5, b"t5 data").unwrap();
        std::fs::write(&orphan_qwen, b"qwen data").unwrap();

        let referenced: HashSet<String> = HashSet::new();
        let cfg = Config::default();
        let hf_index = HfCacheIndex::build(&cfg);
        let (count, bytes) = remove_orphaned_files_recursive(&shared, &referenced, &hf_index);
        assert_eq!(count, 2, "should delete both orphaned files");
        assert_eq!(bytes, 16, "should report correct total bytes"); // "t5 data"(7) + "qwen data"(9)
        remove_empty_dirs_recursive(&shared);
        let _ = std::fs::remove_dir(&shared);

        assert!(!orphan_t5.exists(), "orphaned T5 should be deleted");
        assert!(!orphan_qwen.exists(), "orphaned Qwen3 should be deleted");
        assert!(
            !shared.exists(),
            "shared/ should be removed when all orphans deleted"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn orphan_cleanup_preserves_files_shared_across_models() {
        // File referenced by model-b should survive even when model-a is removed
        let tmp = make_tmp_dir("shared-survive");
        let shared = tmp.join("shared");
        let flux_dir = shared.join("flux");
        std::fs::create_dir_all(&flux_dir).unwrap();

        let shared_vae = flux_dir.join("ae.safetensors");
        let shared_t5 = flux_dir.join("t5xxl_fp16.safetensors");
        std::fs::write(&shared_vae, b"vae").unwrap();
        std::fs::write(&shared_t5, b"t5").unwrap();

        // model-b still references both files
        let referenced: HashSet<String> = [
            shared_vae.to_string_lossy().to_string(),
            shared_t5.to_string_lossy().to_string(),
        ]
        .into_iter()
        .collect();

        let cfg = Config::default();
        let hf_index = HfCacheIndex::build(&cfg);
        let (count, bytes) = remove_orphaned_files_recursive(&shared, &referenced, &hf_index);
        assert_eq!(count, 0, "no files should be deleted");
        assert_eq!(bytes, 0, "no bytes should be freed");
        remove_empty_dirs_recursive(&shared);

        assert!(shared_vae.exists(), "shared VAE should survive");
        assert!(shared_t5.exists(), "shared T5 should survive");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clean_orphaned_shared_files_noop_when_no_shared_dir() {
        // Should not panic when shared/ doesn't exist
        let config = Config::default();
        // This should be a no-op, not a crash
        clean_orphaned_shared_files(&config);
    }

    #[test]
    fn stale_pull_marker_is_removed_when_old() {
        let tmp = make_tmp_dir("stale-marker");
        let marker_dir = tmp.join("flux-dev-q6");
        std::fs::create_dir_all(&marker_dir).unwrap();
        let marker = marker_dir.join(".pulling");
        std::fs::write(&marker, b"flux-dev:q6").unwrap();

        let old = filetime::FileTime::from_unix_time(1, 0);
        filetime::set_file_mtime(&marker, old).unwrap();

        let mut cfg = Config::default();
        cfg.models_dir = tmp.to_string_lossy().to_string();

        clean_stale_pull_markers(&cfg);

        assert!(!marker.exists(), "stale pull marker should be removed");
        assert!(
            !marker_dir.exists(),
            "empty model dir should be removed too"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn hf_cache_cleanup_removes_inactive_repo_transient_files() {
        let tmp = make_tmp_dir("inactive-repo");
        let blobs_dir = tmp
            .join(".hf-cache")
            .join("models--example--orphaned-repo")
            .join("blobs");
        std::fs::create_dir_all(&blobs_dir).unwrap();

        let lock = blobs_dir.join("deadbeef.lock");
        let part = blobs_dir.join("deadbeef.sync.part");
        std::fs::write(&lock, b"").unwrap();
        std::fs::write(&part, b"partial").unwrap();

        let mut cfg = Config::default();
        cfg.models_dir = tmp.to_string_lossy().to_string();

        clean_hf_cache(&cfg);

        assert!(!lock.exists(), "inactive repo lock file should be removed");
        assert!(
            !part.exists(),
            "inactive repo sync.part file should be removed"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clean_orphaned_shared_files_preserves_manifest_discovered_shared_files() {
        let tmp = make_tmp_dir("manifest-shared");
        let manifest = mold_core::manifest::find_manifest("flux-schnell:q8").unwrap();

        for file in &manifest.files {
            let path = tmp.join(mold_core::manifest::storage_path(manifest, file));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, b"x").unwrap();
        }

        let orphan = tmp.join("shared/flux/orphan.bin");
        std::fs::write(&orphan, b"orphan").unwrap();

        let mut cfg = Config::default();
        cfg.models_dir = tmp.to_string_lossy().to_string();

        clean_orphaned_shared_files(&cfg);

        let shared_vae = tmp.join("shared/flux/ae.safetensors");
        let shared_t5 = tmp.join("shared/flux/t5xxl_fp16.safetensors");
        let transformer = tmp.join("flux-schnell-q8/flux1-schnell-Q8_0.gguf");

        assert!(
            transformer.exists(),
            "manifest model should still appear downloaded"
        );
        assert!(
            shared_vae.exists(),
            "shared file for manifest-discovered model should survive"
        );
        assert!(
            shared_t5.exists(),
            "shared encoder for manifest-discovered model should survive"
        );
        assert!(
            !orphan.exists(),
            "unreferenced sibling orphan should be removed"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn orphaned_shared_file_cleanup_removes_hf_blob_and_snapshot_symlink() {
        let tmp = make_tmp_dir("shared-hf");
        let shared_flux = tmp.join("shared").join("flux");
        let repo = tmp
            .join(".hf-cache")
            .join("models--comfyanonymous--flux_text_encoders");
        let blobs_dir = repo.join("blobs");
        let snapshot_dir = repo.join("snapshots").join("rev1");

        std::fs::create_dir_all(&shared_flux).unwrap();
        std::fs::create_dir_all(&blobs_dir).unwrap();
        std::fs::create_dir_all(&snapshot_dir).unwrap();

        let blob = blobs_dir.join("deadbeef");
        std::fs::write(&blob, b"shared t5 weights").unwrap();

        let snapshot_path = snapshot_dir.join("t5xxl_fp16.safetensors");
        std::os::unix::fs::symlink(&blob, &snapshot_path).unwrap();

        let clean_path = shared_flux.join("t5xxl_fp16.safetensors");
        std::fs::hard_link(&blob, &clean_path).unwrap();

        let mut cfg = Config::default();
        cfg.models_dir = tmp.to_string_lossy().to_string();

        let referenced: HashSet<String> = HashSet::new();
        let hf_index = HfCacheIndex::build(&cfg);
        let (count, _bytes) = remove_orphaned_files_recursive(&shared_flux, &referenced, &hf_index);

        assert_eq!(count, 1, "should count the orphaned shared file once");
        assert!(!clean_path.exists(), "clean shared path should be removed");
        assert!(!blob.exists(), "hf-cache blob should be removed");
        assert!(
            !snapshot_path.exists(),
            "snapshot symlink should be removed alongside the blob"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // --- Regression tests for issue #190 ---

    #[test]
    fn complete_includes_manifest_backed_installed_models() {
        // Regression: complete_installed_model_name() previously only iterated
        // config.models.keys(), missing manifest-backed downloaded models.
        // After the fix, it should also include manifest-backed models detected
        // via manifest_model_is_downloaded().
        use crate::test_support::ENV_LOCK;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = make_tmp_dir("complete-manifest");

        // Populate manifest files for flux-schnell:q8 on disk
        let manifest = mold_core::manifest::find_manifest("flux-schnell:q8").unwrap();
        for file in &manifest.files {
            let path = tmp.join(mold_core::manifest::storage_path(manifest, file));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, b"test").unwrap();
        }
        std::env::set_var("MOLD_MODELS_DIR", &tmp);

        let candidates = complete_installed_model_name();
        let names: Vec<String> = candidates
            .iter()
            .map(|c| c.get_value().to_string_lossy().to_string())
            .collect();

        assert!(
            names.contains(&"flux-schnell:q8".to_string()),
            "manifest-backed model should appear in completions, got: {names:?}"
        );

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn rm_recognises_manifest_backed_model_as_installed() {
        // Regression: `mold rm` only checked config.models.contains_key(),
        // so manifest-backed models that were pulled but not in config
        // were reported as "not installed".
        let tmp = make_tmp_dir("rm-manifest");
        let manifest = mold_core::manifest::find_manifest("flux-schnell:q8").unwrap();
        for file in &manifest.files {
            let path = tmp.join(mold_core::manifest::storage_path(manifest, file));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, b"test").unwrap();
        }

        let config = Config {
            models_dir: tmp.to_string_lossy().to_string(),
            ..Config::default()
        };

        // The model is not in config.models — only discoverable via manifest
        assert!(!config.models.contains_key("flux-schnell:q8"));
        assert!(
            config.manifest_model_is_downloaded("flux-schnell:q8"),
            "manifest-backed model should be detected as downloaded"
        );

        // Verify resolved_model_config returns usable paths
        let resolved = config.resolved_model_config("flux-schnell:q8");
        assert!(
            resolved.transformer.is_some(),
            "resolved config should have transformer path"
        );
        let paths = resolved.all_file_paths();
        assert!(
            !paths.is_empty(),
            "resolved config should produce file paths for deletion"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn ref_counts_include_manifest_backed_models() {
        // Regression: build_ref_counts() only counted config.models entries,
        // so shared components of manifest-backed models were treated as unique
        // and would be deleted when removing a sibling model.
        use crate::test_support::ENV_LOCK;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = make_tmp_dir("ref-counts-manifest");

        // Populate files for two FLUX models that share VAE/T5/CLIP
        for model in &["flux-schnell:q8", "flux-dev:q8"] {
            let manifest = mold_core::manifest::find_manifest(model).unwrap();
            for file in &manifest.files {
                let path = tmp.join(mold_core::manifest::storage_path(manifest, file));
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).unwrap();
                }
                std::fs::write(&path, b"test").unwrap();
            }
        }
        std::env::set_var("MOLD_MODELS_DIR", &tmp);

        let config = Config::default();
        // Neither model has a config entry
        assert!(!config.models.contains_key("flux-schnell:q8"));
        assert!(!config.models.contains_key("flux-dev:q8"));

        let refs = build_ref_counts(&config);

        // Shared VAE should be referenced by both models
        let schnell_cfg = config.resolved_model_config("flux-schnell:q8");
        if let Some(vae_path) = &schnell_cfg.vae {
            let vae_refs = refs.get(vae_path).expect("shared VAE should have refs");
            assert!(
                vae_refs.len() >= 2,
                "shared VAE should be referenced by at least 2 models, got: {vae_refs:?}"
            );
        }

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
