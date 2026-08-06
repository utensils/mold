use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MoldHomeInfo {
    pub path: String,
    pub source: &'static str,
    pub exists: bool,
}

pub fn info() -> anyhow::Result<MoldHomeInfo> {
    if std::env::var_os("MOLD_HOME").is_none() && mold_core::Config::read_saved_mold_dir().is_err()
    {
        return Ok(MoldHomeInfo {
            path: String::new(),
            source: "invalid",
            exists: false,
        });
    }
    let path = mold_core::Config::mold_dir()
        .ok_or_else(|| anyhow!("could not resolve the Mold home directory"))?;
    let source = if std::env::var_os("MOLD_HOME").is_some() {
        "environment"
    } else if mold_core::Config::saved_mold_dir().is_some() {
        "saved"
    } else {
        "default"
    };
    Ok(MoldHomeInfo {
        exists: path.is_dir(),
        path: path.to_string_lossy().into_owned(),
        source,
    })
}

pub fn prepare_destination(raw: &str) -> anyhow::Result<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("Choose a Mold home directory.");
    }
    if trimmed.contains('\0') {
        bail!("The directory contains an invalid null character.");
    }

    let expanded = if trimmed == "~" {
        dirs_home().ok_or_else(|| anyhow!("could not resolve your home directory"))?
    } else if let Some(rest) = trimmed
        .strip_prefix("~/")
        .or_else(|| trimmed.strip_prefix("~\\"))
    {
        dirs_home()
            .ok_or_else(|| anyhow!("could not resolve your home directory"))?
            .join(rest)
    } else {
        PathBuf::from(trimmed)
    };
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .context("could not resolve the current directory")?
            .join(expanded)
    };
    Ok(clean_path(&absolute))
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| {
            mold_core::Config::mold_dir().and_then(|path| path.parent().map(Path::to_path_buf))
        })
}

fn clean_path(path: &Path) -> PathBuf {
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                clean.pop();
            }
            other => clean.push(other.as_os_str()),
        }
    }
    clean
}

pub fn validate_or_migrate(source: &Path, destination: &Path, migrate: bool) -> anyhow::Result<()> {
    preflight(source, destination, migrate)?;
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("choose a directory with a writable parent"))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("could not create {}", parent.display()))?;

    // Re-resolve after creating the parent. This closes the gap where a
    // symlinked ancestor could place migration staging inside the source.
    reject_overlapping_roots(source, destination)?;

    if migrate {
        migrate_atomically(source, destination)
    } else {
        std::fs::create_dir_all(destination)
            .with_context(|| format!("could not create {}", destination.display()))?;
        verify_writable(destination)
    }
}

/// Read-only checks that can run while the current server still owns the
/// gallery and database. They catch ordinary form errors before shutdown.
pub fn preflight(source: &Path, destination: &Path, migrate: bool) -> anyhow::Result<()> {
    reject_overlapping_roots(source, destination)?;
    if migrate && !source.is_dir() {
        bail!(
            "The current Mold home at {} is unavailable. Reconnect that drive, or use the new folder without copying.",
            source.display()
        );
    }
    if migrate && destination.exists() && std::fs::read_dir(destination)?.next().is_some() {
        bail!(
            "{} is not empty. Choose an empty folder for migration, or use it without copying.",
            destination.display()
        );
    }
    Ok(())
}

fn reject_overlapping_roots(source: &Path, destination: &Path) -> anyhow::Result<()> {
    let source_identity = path_identity(source)?;
    let destination_identity = path_identity(destination)?;
    if source_identity == destination_identity {
        bail!("That is already the active Mold home directory.");
    }
    if destination_identity.starts_with(&source_identity)
        || source_identity.starts_with(&destination_identity)
    {
        bail!("The old and new Mold home directories cannot contain one another.");
    }
    Ok(())
}

fn path_identity(path: &Path) -> anyhow::Result<PathBuf> {
    if let Ok(identity) = path.canonicalize() {
        return Ok(identity);
    }
    let mut missing = Vec::new();
    let mut existing = path;
    while !existing.exists() {
        let name = existing
            .file_name()
            .ok_or_else(|| anyhow!("could not resolve {}", path.display()))?;
        missing.push(name.to_os_string());
        existing = existing
            .parent()
            .ok_or_else(|| anyhow!("could not resolve {}", path.display()))?;
    }
    let mut identity = existing
        .canonicalize()
        .with_context(|| format!("could not resolve {}", existing.display()))?;
    for component in missing.iter().rev() {
        identity.push(component);
    }
    Ok(identity)
}

pub fn persist_destination(destination: &Path) -> anyhow::Result<()> {
    mold_core::Config::save_mold_dir(destination)
        .with_context(|| format!("could not save {} as the Mold home", destination.display()))
}

fn migrate_atomically(source: &Path, destination: &Path) -> anyhow::Result<()> {
    let parent = destination.parent().expect("validated parent");
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("mold-home");
    let staging = parent.join(format!(".{name}.mold-migration-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&staging).with_context(|| {
        format!(
            "could not create migration staging directory in {}",
            parent.display()
        )
    })?;

    let copied = copy_tree(source, &staging).and_then(|()| verify_writable(&staging));
    if let Err(error) = copied {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    if destination.exists() {
        if let Err(error) = std::fs::remove_dir(destination) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(error).with_context(|| {
                format!(
                    "could not replace the empty directory {}",
                    destination.display()
                )
            });
        }
    }
    if let Err(error) = std::fs::rename(&staging, destination) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error)
            .with_context(|| format!("could not finish migration into {}", destination.display()));
    }
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> anyhow::Result<()> {
    if !source.exists() {
        return Ok(());
    }
    for entry in
        std::fs::read_dir(source).with_context(|| format!("could not read {}", source.display()))?
    {
        let entry = entry?;
        let from = entry.path();
        let to = destination.join(entry.file_name());
        let metadata = std::fs::symlink_metadata(&from)?;
        if metadata.file_type().is_symlink() {
            copy_symlink(&from, &to)?;
        } else if metadata.is_dir() {
            std::fs::create_dir(&to)?;
            copy_tree(&from, &to)?;
        } else if metadata.is_file() {
            std::fs::copy(&from, &to)
                .with_context(|| format!("could not copy {}", from.display()))?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn copy_symlink(source: &Path, destination: &Path) -> anyhow::Result<()> {
    std::os::unix::fs::symlink(std::fs::read_link(source)?, destination)?;
    Ok(())
}

#[cfg(windows)]
fn copy_symlink(source: &Path, destination: &Path) -> anyhow::Result<()> {
    let target = std::fs::read_link(source)?;
    if source.is_dir() {
        std::os::windows::fs::symlink_dir(target, destination)?;
    } else {
        std::os::windows::fs::symlink_file(target, destination)?;
    }
    Ok(())
}

fn verify_writable(directory: &Path) -> anyhow::Result<()> {
    let probe = directory.join(format!(".mold-write-test-{}", uuid::Uuid::new_v4()));
    std::fs::write(&probe, b"mold")
        .with_context(|| format!("{} is not writable", directory.display()))?;
    std::fs::remove_file(probe)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_relative_paths_become_clean_absolute_paths() {
        let path = prepare_destination("somewhere/../mold").unwrap();
        assert!(path.is_absolute());
        assert!(path.ends_with("mold"));
        assert!(!path.to_string_lossy().contains(".."));
    }

    #[test]
    fn migration_copies_the_tree_and_preserves_the_source() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("old");
        let destination = root.path().join("new");
        std::fs::create_dir_all(source.join("models/flux")).unwrap();
        std::fs::write(source.join("config.toml"), "models_dir = 'models'").unwrap();
        std::fs::write(source.join("models/flux/model.bin"), b"weights").unwrap();

        validate_or_migrate(&source, &destination, true).unwrap();

        assert_eq!(
            std::fs::read(destination.join("models/flux/model.bin")).unwrap(),
            b"weights"
        );
        assert!(source.join("models/flux/model.bin").exists());
    }

    #[test]
    fn migration_refuses_a_nonempty_destination_without_touching_it() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("old");
        let destination = root.path().join("existing");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&destination).unwrap();
        std::fs::write(destination.join("keep"), b"safe").unwrap();

        let error = validate_or_migrate(&source, &destination, true).unwrap_err();
        assert!(error.to_string().contains("not empty"));
        assert_eq!(std::fs::read(destination.join("keep")).unwrap(), b"safe");
    }

    #[test]
    fn migration_refuses_nested_roots() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("old");
        let destination = source.join("new");
        assert!(validate_or_migrate(&source, &destination, true)
            .unwrap_err()
            .to_string()
            .contains("contain one another"));
    }

    #[cfg(unix)]
    #[test]
    fn migration_refuses_a_symlinked_parent_inside_the_source() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("old");
        let nested = source.join("nested");
        let alias = root.path().join("alias");
        std::fs::create_dir_all(&nested).unwrap();
        std::os::unix::fs::symlink(&nested, &alias).unwrap();

        let error = validate_or_migrate(&source, &alias.join("new"), true).unwrap_err();
        assert!(error.to_string().contains("contain one another"));
        assert!(!nested.join("new").exists());
    }

    #[test]
    fn migration_refuses_an_unavailable_source() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("missing");
        let destination = root.path().join("new");
        let error = validate_or_migrate(&source, &destination, true).unwrap_err();
        assert!(error.to_string().contains("unavailable"));
        assert!(!destination.exists());
    }
}
