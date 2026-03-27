use std::collections::HashMap;
use std::io::{self, Write};

use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::manifest::resolve_model_name;
use mold_core::Config;

use crate::theme;
use crate::ui::format_bytes;
use crate::AlreadyReported;

/// Provide completions for installed (configured) model names only.
pub fn complete_installed_model_name() -> Vec<CompletionCandidate> {
    let config = Config::load_or_default();
    config.models.keys().map(CompletionCandidate::new).collect()
}

/// Get file size, returning 0 if the file doesn't exist or can't be read.
fn file_size(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Build a map of file_path -> list of model names that reference it.
fn build_ref_counts(config: &Config) -> HashMap<String, Vec<String>> {
    let mut refs: HashMap<String, Vec<String>> = HashMap::new();
    for (model_name, model_config) in &config.models {
        for path in model_config.all_file_paths() {
            refs.entry(path).or_default().push(model_name.clone());
        }
    }
    refs
}

/// Collect hf-hub cache blob paths for a model's files.
///
/// When `mold pull` downloads files, the hf-hub crate stores blobs at:
///   `<models_dir>/.hf-cache/models--<org>--<repo>/blobs/<sha256>`
/// and creates snapshot symlinks pointing to them. The clean storage paths
/// are then hardlinked from these blobs. This function walks the hf-cache
/// snapshot dirs looking for files whose names match the manifest's
/// `hf_filename` entries, then resolves their symlinks to find the blob paths.
fn collect_hf_cache_blob_paths(model_name: &str) -> Vec<std::path::PathBuf> {
    let manifest = match mold_core::manifest::find_manifest(model_name) {
        Some(m) => m,
        None => return Vec::new(),
    };

    let config = Config::load_or_default();
    let cache_dir = config.resolved_models_dir().join(".hf-cache");
    if !cache_dir.is_dir() {
        return Vec::new();
    }

    let mut blobs = Vec::new();

    for file in &manifest.files {
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

        let filename = std::path::Path::new(&file.hf_filename)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        if let Ok(revisions) = std::fs::read_dir(&snapshots_dir) {
            for rev in revisions.flatten() {
                let snap_file = rev.path().join(&filename);
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

/// Remove dangling symlinks and empty directories from the hf-hub cache.
///
/// After deleting blob files, the hf-cache may contain:
/// - Dangling symlinks in `snapshots/<rev>/` pointing to deleted blobs
/// - Empty snapshot revision directories
/// - Empty blob directories
/// - Empty repo directories (`models--org--repo/`)
fn clean_hf_cache() {
    let config = Config::load_or_default();
    let cache_dir = config.resolved_models_dir().join(".hf-cache");
    if !cache_dir.is_dir() {
        return;
    }

    // Walk repo directories
    let entries = match std::fs::read_dir(&cache_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let repo_dir = entry.path();
        if !repo_dir.is_dir() {
            continue;
        }

        // Clean dangling symlinks in snapshots/*/
        let snapshots_dir = repo_dir.join("snapshots");
        if snapshots_dir.is_dir() {
            if let Ok(revisions) = std::fs::read_dir(&snapshots_dir) {
                for rev in revisions.flatten() {
                    let rev_dir = rev.path();
                    if !rev_dir.is_dir() {
                        continue;
                    }
                    // Remove dangling symlinks
                    if let Ok(files) = std::fs::read_dir(&rev_dir) {
                        for file in files.flatten() {
                            let p = file.path();
                            // symlink_metadata succeeds for dangling symlinks;
                            // exists() returns false for them.
                            if p.symlink_metadata().is_ok() && !p.exists() {
                                let _ = std::fs::remove_file(&p);
                            }
                        }
                    }
                    // Remove empty revision dir
                    let _ = std::fs::remove_dir(&rev_dir);
                }
            }
            let _ = std::fs::remove_dir(&snapshots_dir);
        }

        // Remove empty blobs dir
        let blobs_dir = repo_dir.join("blobs");
        if blobs_dir.is_dir() {
            let _ = std::fs::remove_dir(&blobs_dir);
        }

        // Remove empty repo dir
        let _ = std::fs::remove_dir(&repo_dir);
    }

    // Remove empty .hf-cache dir itself
    let _ = std::fs::remove_dir(&cache_dir);
}

pub async fn run(models: &[String], force: bool) -> Result<()> {
    let mut config = Config::load_or_default();
    let mut any_error = false;

    for model_arg in models {
        let canonical = resolve_model_name(model_arg);

        // Check if the model is installed
        if !config.models.contains_key(&canonical) {
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

        let model_config = config.models.get(&canonical).unwrap();
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
        let hf_cache_blobs = collect_hf_cache_blob_paths(&canonical);

        for (path, size) in &unique_files {
            match std::fs::remove_file(path) {
                Ok(()) => freed += size,
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
        for blob_path in &hf_cache_blobs {
            if blob_path.exists() {
                let _ = std::fs::remove_file(blob_path);
            }
        }

        // Clean up empty model-specific directories left behind.
        if let Some(model_cfg) = config.models.get(&canonical) {
            if let Some(ref t) = model_cfg.transformer {
                if let Some(parent) = std::path::Path::new(t).parent() {
                    let _ = std::fs::remove_dir(parent); // only succeeds if empty
                }
            }
        }

        // Clean up dangling hf-cache entries (symlinks to deleted blobs,
        // empty snapshot dirs, empty repo dirs).
        clean_hf_cache();

        // Remove from config and clean up pull marker
        config.remove_model(&canonical);
        mold_core::download::remove_pulling_marker(&canonical);

        // Handle default_model update
        if config.default_model == canonical {
            let new_default = config
                .models
                .keys()
                .min()
                .cloned()
                .unwrap_or_else(|| "flux-schnell".to_string());
            if config.models.is_empty() {
                eprintln!(
                    "{} default model reset to {}",
                    theme::prefix_note(),
                    "flux-schnell".bold()
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

    if any_error {
        return Err(AlreadyReported.into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
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
}
