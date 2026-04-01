use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::manifest::{known_manifests, resolve_model_name};
use mold_core::Config;

use crate::theme;
use crate::ui::format_bytes;
use crate::AlreadyReported;

const STALE_PULL_MAX_AGE: Duration = Duration::from_secs(6 * 60 * 60);

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
fn collect_hf_cache_blob_paths(
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

/// Collect all file paths still referenced by remaining models.
///
/// This includes:
/// - explicit config entries
/// - manifest-discovered local installs
/// - manifest paths for models with an active `.pulling` marker
///
/// The latter two matter because manifest-backed models can exist without a
/// config entry, and incomplete pulls should prevent us from racing cleanup
/// against files that are still being written.
fn collect_referenced_paths(config: &Config) -> HashSet<String> {
    let mut referenced: HashSet<String> = config
        .models
        .values()
        .flat_map(|mc| mc.all_file_paths())
        .collect();

    let models_dir = config.resolved_models_dir();

    for manifest in known_manifests() {
        if mold_core::download::has_pulling_marker(&manifest.name) {
            for file in &manifest.files {
                referenced.insert(
                    models_dir
                        .join(mold_core::manifest::storage_path(manifest, file))
                        .to_string_lossy()
                        .to_string(),
                );
            }
            continue;
        }

        if config.manifest_model_is_downloaded(&manifest.name) {
            referenced.extend(config.model_config(&manifest.name).all_file_paths());
        }
    }

    referenced
}

/// Pre-built index mapping inode identity `(dev, ino)` to hf-cache paths
/// (blobs and snapshot symlinks) that should be removed when the corresponding
/// clean path is deleted.
///
/// Building this once avoids re-walking the entire `.hf-cache` tree for every
/// orphaned shared file.
struct HfCacheIndex {
    /// Maps `(dev, ino)` to the list of cache paths (blobs + snapshot symlinks)
    /// sharing that inode.
    #[cfg(unix)]
    by_inode: HashMap<(u64, u64), Vec<PathBuf>>,
    #[cfg(not(unix))]
    by_canonical: HashMap<PathBuf, Vec<PathBuf>>,
}

impl HfCacheIndex {
    fn build(config: &Config) -> Self {
        #[cfg(unix)]
        let mut by_inode: HashMap<(u64, u64), Vec<PathBuf>> = HashMap::new();
        #[cfg(not(unix))]
        let mut by_canonical: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();

        let models_dir = config.resolved_models_dir();
        let cache_dir = models_dir.join(".hf-cache");
        if !cache_dir.is_dir() {
            #[cfg(unix)]
            return Self { by_inode };
            #[cfg(not(unix))]
            return Self { by_canonical };
        }

        let repos = match std::fs::read_dir(&cache_dir) {
            Ok(entries) => entries,
            Err(_) => {
                #[cfg(unix)]
                return Self { by_inode };
                #[cfg(not(unix))]
                return Self { by_canonical };
            }
        };

        for repo in repos.flatten() {
            let snapshots_dir = repo.path().join("snapshots");
            if !snapshots_dir.is_dir() {
                continue;
            }
            let revisions = match std::fs::read_dir(&snapshots_dir) {
                Ok(entries) => entries,
                Err(_) => continue,
            };
            for rev in revisions.flatten() {
                let rev_dir = rev.path();
                if !rev_dir.is_dir() {
                    continue;
                }
                let files = match walk_files_recursive(&rev_dir) {
                    Ok(files) => files,
                    Err(_) => continue,
                };
                for snapshot_path in files {
                    let Ok(blob) = snapshot_path.canonicalize() else {
                        continue;
                    };

                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        if let Ok(meta) = std::fs::metadata(&blob) {
                            let key = (meta.dev(), meta.ino());
                            let entry = by_inode.entry(key).or_default();
                            if !entry.contains(&blob) {
                                entry.push(blob.clone());
                            }
                            entry.push(snapshot_path);
                        }
                    }

                    #[cfg(not(unix))]
                    {
                        let entry = by_canonical.entry(blob.clone()).or_default();
                        if !entry.contains(&blob) {
                            entry.push(blob.clone());
                        }
                        entry.push(snapshot_path);
                    }
                }
            }
        }

        #[cfg(unix)]
        return Self { by_inode };
        #[cfg(not(unix))]
        return Self { by_canonical };
    }

    /// Look up hf-cache paths (blobs + snapshot symlinks) for a given clean path.
    fn lookup(&self, clean_path: &Path) -> Vec<PathBuf> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let Ok(meta) = std::fs::metadata(clean_path) else {
                return Vec::new();
            };
            let key = (meta.dev(), meta.ino());
            self.by_inode.get(&key).cloned().unwrap_or_default()
        }

        #[cfg(not(unix))]
        {
            let canonical = clean_path
                .canonicalize()
                .unwrap_or_else(|_| clean_path.to_path_buf());
            self.by_canonical
                .get(&canonical)
                .cloned()
                .unwrap_or_default()
        }
    }
}

fn walk_files_recursive(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() && !path.is_symlink() {
            files.extend(walk_files_recursive(&path)?);
        } else if path.is_file() || path.is_symlink() {
            files.push(path);
        }
    }
    Ok(files)
}

fn is_path_older_than(path: &Path, age: Duration) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    let Ok(modified) = meta.modified() else {
        return false;
    };
    let Ok(elapsed) = modified.elapsed() else {
        return false;
    };
    elapsed >= age
}

fn pull_marker_path_for_model(config: &Config, model_name: &str) -> PathBuf {
    config
        .resolved_models_dir()
        .join(mold_core::download::pulling_marker_rel_path(model_name))
}

fn is_stale_pull_marker(config: &Config, model_name: &str) -> bool {
    let marker = pull_marker_path_for_model(config, model_name);
    marker.exists() && is_path_older_than(&marker, STALE_PULL_MAX_AGE)
}

fn active_pull_models(config: &Config) -> HashSet<String> {
    known_manifests()
        .iter()
        .filter_map(|manifest| {
            let name = &manifest.name;
            let marker_exists = mold_core::download::has_pulling_marker(name);
            (marker_exists && !is_stale_pull_marker(config, name)).then(|| name.clone())
        })
        .collect()
}

fn active_pull_repos(config: &Config) -> HashSet<String> {
    let active_models = active_pull_models(config);
    let mut repos = HashSet::new();

    for manifest in known_manifests() {
        if active_models.contains(&manifest.name) {
            repos.extend(manifest.files.iter().map(|file| file.hf_repo.clone()));
        }
    }

    repos
}

fn clean_stale_pull_markers(config: &Config) {
    let mut removed = 0u64;

    for manifest in known_manifests() {
        if !is_stale_pull_marker(config, &manifest.name) {
            continue;
        }

        let marker = pull_marker_path_for_model(config, &manifest.name);
        match std::fs::remove_file(&marker) {
            Ok(()) => {
                removed += 1;
                if let Some(parent) = marker.parent() {
                    let _ = std::fs::remove_dir(parent);
                }
            }
            Err(e) => {
                eprintln!(
                    "{} failed to remove stale pull marker {}: {}",
                    theme::prefix_warning(),
                    marker.display(),
                    e
                );
            }
        }
    }

    if removed > 0 {
        eprintln!(
            "{} removed {} stale pull marker{}",
            theme::prefix_note(),
            removed,
            if removed == 1 { "" } else { "s" }
        );
    }
}

/// Remove orphaned files from `shared/` that are not referenced by any
/// remaining model, then clean up empty directories under `shared/`.
///
/// This covers both runtime caches such as `shared/t5-gguf/` and normal
/// manifest-managed shared directories such as `shared/flux/` or `shared/sd15/`.
///
fn clean_orphaned_shared_files(config: &Config) {
    let models_dir = config.resolved_models_dir();
    let shared_dir = models_dir.join("shared");
    if !shared_dir.is_dir() {
        return;
    }

    let referenced = collect_referenced_paths(config);
    let hf_index = HfCacheIndex::build(config);

    let mut total_count = 0u64;
    let mut total_bytes = 0u64;
    let (count, bytes) = remove_orphaned_files_recursive(&shared_dir, &referenced, &hf_index);
    total_count += count;
    total_bytes += bytes;

    if total_count > 0 {
        eprintln!(
            "{} cleaned up {} orphaned shared file{} (freed {})",
            theme::prefix_note(),
            total_count,
            if total_count == 1 { "" } else { "s" },
            format_bytes(total_bytes),
        );
    }

    // Clean up empty directories bottom-up under shared/.
    remove_empty_dirs_recursive(&shared_dir);

    // Remove shared/ itself if empty.
    let _ = std::fs::remove_dir(&shared_dir);
}

/// Recursively remove files under `dir` that are not in `referenced`.
/// Returns `(files_deleted, bytes_freed)`.
fn remove_orphaned_files_recursive(
    dir: &Path,
    referenced: &HashSet<String>,
    hf_index: &HfCacheIndex,
) -> (u64, u64) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return (0, 0),
    };
    let mut count = 0u64;
    let mut bytes = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_symlink() && path.is_dir() {
            let (c, b) = remove_orphaned_files_recursive(&path, referenced, hf_index);
            count += c;
            bytes += b;
        } else if path.is_file() {
            let path_str = path.to_string_lossy().to_string();
            if !referenced.contains(&path_str) {
                let mut removed_any = false;
                let mut removed_bytes = 0u64;
                let cache_targets = hf_index.lookup(&path);
                let mut counted_inodes = HashSet::new();

                for target in &cache_targets {
                    if target.exists() {
                        let size = reclaimed_disk_bytes(target, &mut counted_inodes);
                        match std::fs::remove_file(target) {
                            Ok(()) => {
                                removed_any = true;
                                removed_bytes += size;
                            }
                            Err(e) => {
                                eprintln!(
                                    "{} failed to clean up cache file {}: {}",
                                    theme::prefix_warning(),
                                    target.display(),
                                    e
                                );
                            }
                        }
                    }
                }

                if path.exists() {
                    let clean_size = reclaimed_disk_bytes(&path, &mut counted_inodes);
                    match std::fs::remove_file(&path) {
                        Ok(()) => {
                            removed_any = true;
                            removed_bytes += clean_size;
                        }
                        Err(e) => {
                            eprintln!(
                                "{} failed to clean up orphaned file {}: {}",
                                theme::prefix_warning(),
                                path.display(),
                                e
                            );
                        }
                    }
                }

                if removed_any {
                    count += 1;
                    bytes += removed_bytes;
                }
            }
        }
    }
    (count, bytes)
}

fn reclaimed_disk_bytes(path: &Path, counted_inodes: &mut HashSet<(u64, u64)>) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };

    if meta.file_type().is_symlink() {
        return 0;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let inode = (meta.dev(), meta.ino());
        if !counted_inodes.insert(inode) {
            return 0;
        }
    }

    #[cfg(not(unix))]
    {
        let _ = counted_inodes;
    }

    meta.len()
}

/// Recursively remove empty directories under `dir` (bottom-up).
fn remove_empty_dirs_recursive(dir: &std::path::Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            remove_empty_dirs_recursive(&path);
            // Try removing — only succeeds if empty.
            let _ = std::fs::remove_dir(&path);
        }
    }
}

/// Remove dangling symlinks and empty directories from the hf-hub cache.
///
/// After deleting blob files, the hf-cache may contain:
/// - Dangling symlinks in `snapshots/<rev>/` pointing to deleted blobs
/// - Empty snapshot revision directories
/// - Empty blob directories
/// - Empty repo directories (`models--org--repo/`)
fn clean_hf_cache(config: &Config) {
    let cache_dir = config.resolved_models_dir().join(".hf-cache");
    if !cache_dir.is_dir() {
        return;
    }

    let active_repos = active_pull_repos(config);

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
        let repo_name = repo_dir
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|name| name.strip_prefix("models--"))
            .map(|rest| rest.replace("--", "/"));
        let repo_is_active = repo_name
            .as_ref()
            .is_some_and(|repo| active_repos.contains(repo));

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
            if !repo_is_active {
                if let Ok(files) = std::fs::read_dir(&blobs_dir) {
                    for file in files.flatten() {
                        let path = file.path();
                        let name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or_default();
                        let is_transient = name.ends_with(".lock") || name.ends_with(".sync.part");
                        if is_transient {
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
            }
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
        if let Some(model_cfg) = config.models.get(&canonical) {
            if let Some(ref t) = model_cfg.transformer {
                if let Some(parent) = std::path::Path::new(t).parent() {
                    let _ = std::fs::remove_dir(parent); // only succeeds if empty
                }
            }
        }

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
                .unwrap_or_else(|| "flux2-klein".to_string());
            if config.models.is_empty() {
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
}
