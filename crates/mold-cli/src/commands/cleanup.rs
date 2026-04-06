//! Shared cleanup utilities used by both `rm` and `clean` commands.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use mold_core::manifest::known_manifests;
use mold_core::Config;

use crate::theme;
use crate::ui::format_bytes;

pub const STALE_PULL_MAX_AGE: Duration = Duration::from_secs(6 * 60 * 60);

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
pub fn collect_referenced_paths(config: &Config) -> HashSet<String> {
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

pub fn walk_files_recursive(dir: &Path) -> io::Result<Vec<PathBuf>> {
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

pub fn is_path_older_than(path: &Path, age: Duration) -> bool {
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

pub fn pull_marker_path_for_model(config: &Config, model_name: &str) -> PathBuf {
    config
        .resolved_models_dir()
        .join(mold_core::download::pulling_marker_rel_path(model_name))
}

pub fn is_stale_pull_marker(config: &Config, model_name: &str) -> bool {
    let marker = pull_marker_path_for_model(config, model_name);
    marker.exists() && is_path_older_than(&marker, STALE_PULL_MAX_AGE)
}

pub fn active_pull_models(config: &Config) -> HashSet<String> {
    known_manifests()
        .iter()
        .filter_map(|manifest| {
            let name = &manifest.name;
            let marker_exists = mold_core::download::has_pulling_marker(name);
            (marker_exists && !is_stale_pull_marker(config, name)).then(|| name.clone())
        })
        .collect()
}

pub fn active_pull_repos(config: &Config) -> HashSet<String> {
    let active_models = active_pull_models(config);
    let mut repos = HashSet::new();

    for manifest in known_manifests() {
        if active_models.contains(&manifest.name) {
            repos.extend(manifest.files.iter().map(|file| file.hf_repo.clone()));
        }
    }

    repos
}

pub fn clean_stale_pull_markers(config: &Config) -> u64 {
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

    removed
}

/// Find stale pull markers without removing them. Returns (count, model_names).
pub fn find_stale_pull_markers(config: &Config) -> Vec<String> {
    known_manifests()
        .iter()
        .filter(|manifest| is_stale_pull_marker(config, &manifest.name))
        .map(|manifest| manifest.name.clone())
        .collect()
}

/// Pre-built index mapping inode identity `(dev, ino)` to hf-cache paths
/// (blobs and snapshot symlinks) that should be removed when the corresponding
/// clean path is deleted.
///
/// Building this once avoids re-walking the entire `.hf-cache` tree for every
/// orphaned shared file.
pub struct HfCacheIndex {
    #[cfg(unix)]
    by_inode: HashMap<(u64, u64), Vec<PathBuf>>,
    #[cfg(not(unix))]
    by_canonical: HashMap<PathBuf, Vec<PathBuf>>,
}

impl HfCacheIndex {
    pub fn build(config: &Config) -> Self {
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
    pub fn lookup(&self, clean_path: &Path) -> Vec<PathBuf> {
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

pub fn reclaimed_disk_bytes(path: &Path, counted_inodes: &mut HashSet<(u64, u64)>) -> u64 {
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

/// Recursively remove files under `dir` that are not in `referenced`.
/// Returns `(files_deleted, bytes_freed)`.
pub fn remove_orphaned_files_recursive(
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

/// Find orphaned files under `dir` without removing them.
/// Returns `(count, total_bytes)`.
pub fn find_orphaned_files_recursive(dir: &Path, referenced: &HashSet<String>) -> (u64, u64) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return (0, 0),
    };
    let mut count = 0u64;
    let mut bytes = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_symlink() && path.is_dir() {
            let (c, b) = find_orphaned_files_recursive(&path, referenced);
            count += c;
            bytes += b;
        } else if path.is_file() {
            let path_str = path.to_string_lossy().to_string();
            if !referenced.contains(&path_str) {
                count += 1;
                bytes += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    (count, bytes)
}

/// Recursively remove empty directories under `dir` (bottom-up).
pub fn remove_empty_dirs_recursive(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            remove_empty_dirs_recursive(&path);
            let _ = std::fs::remove_dir(&path);
        }
    }
}

/// Remove orphaned files from `shared/` that are not referenced by any
/// remaining model, then clean up empty directories under `shared/`.
pub fn clean_orphaned_shared_files(config: &Config) -> (u64, u64) {
    let models_dir = config.resolved_models_dir();
    let shared_dir = models_dir.join("shared");
    if !shared_dir.is_dir() {
        return (0, 0);
    }

    let referenced = collect_referenced_paths(config);
    let hf_index = HfCacheIndex::build(config);

    let (count, bytes) = remove_orphaned_files_recursive(&shared_dir, &referenced, &hf_index);

    if count > 0 {
        eprintln!(
            "{} cleaned up {} orphaned shared file{} (freed {})",
            theme::prefix_note(),
            count,
            if count == 1 { "" } else { "s" },
            format_bytes(bytes),
        );
    }

    remove_empty_dirs_recursive(&shared_dir);
    let _ = std::fs::remove_dir(&shared_dir);

    (count, bytes)
}

/// Find orphaned shared files without removing them.
pub fn find_orphaned_shared_files(config: &Config) -> (u64, u64) {
    let models_dir = config.resolved_models_dir();
    let shared_dir = models_dir.join("shared");
    if !shared_dir.is_dir() {
        return (0, 0);
    }

    let referenced = collect_referenced_paths(config);
    find_orphaned_files_recursive(&shared_dir, &referenced)
}

/// Remove dangling symlinks and empty directories from the hf-hub cache.
pub fn clean_hf_cache(config: &Config) {
    let cache_dir = config.resolved_models_dir().join(".hf-cache");
    if !cache_dir.is_dir() {
        return;
    }

    let active_repos = active_pull_repos(config);

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
                    if let Ok(files) = std::fs::read_dir(&rev_dir) {
                        for file in files.flatten() {
                            let p = file.path();
                            if p.symlink_metadata().is_ok() && !p.exists() {
                                let _ = std::fs::remove_file(&p);
                            }
                        }
                    }
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

        let _ = std::fs::remove_dir(&repo_dir);
    }

    let _ = std::fs::remove_dir(&cache_dir);
}

/// Count transient hf-cache files (locks, partial downloads, dangling symlinks).
pub fn count_hf_cache_transients(config: &Config) -> (u64, u64) {
    let cache_dir = config.resolved_models_dir().join(".hf-cache");
    if !cache_dir.is_dir() {
        return (0, 0);
    }

    let active_repos = active_pull_repos(config);
    let mut count = 0u64;
    let mut bytes = 0u64;

    let entries = match std::fs::read_dir(&cache_dir) {
        Ok(e) => e,
        Err(_) => return (0, 0),
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

        // Dangling symlinks in snapshots
        let snapshots_dir = repo_dir.join("snapshots");
        if snapshots_dir.is_dir() {
            if let Ok(revisions) = std::fs::read_dir(&snapshots_dir) {
                for rev in revisions.flatten() {
                    let rev_dir = rev.path();
                    if !rev_dir.is_dir() {
                        continue;
                    }
                    if let Ok(files) = std::fs::read_dir(&rev_dir) {
                        for file in files.flatten() {
                            let p = file.path();
                            if p.symlink_metadata().is_ok() && !p.exists() {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        // Transient files in blobs
        let blobs_dir = repo_dir.join("blobs");
        if blobs_dir.is_dir() && !repo_is_active {
            if let Ok(files) = std::fs::read_dir(&blobs_dir) {
                for file in files.flatten() {
                    let path = file.path();
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or_default();
                    let is_transient = name.ends_with(".lock") || name.ends_with(".sync.part");
                    if is_transient {
                        count += 1;
                        bytes += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    }
                }
            }
        }
    }

    (count, bytes)
}
