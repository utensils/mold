//! Race-resistant local file reads for user-selected conditioning media.
//!
//! Callers retain the returned descriptor for hashing, probing, and upload so
//! a later path or symlink replacement cannot substitute different bytes.

use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};

pub(crate) fn absolute_lexical_path(path: &Path) -> Result<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve the current directory")?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                ensure!(
                    normalized.pop(),
                    "reference path escapes the filesystem root"
                );
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    ensure!(
        normalized.is_absolute(),
        "reference path did not resolve absolutely"
    );
    Ok(normalized)
}

#[cfg(unix)]
fn open_directory_without_symlinks(path: &Path) -> Result<File> {
    #[cfg(target_os = "macos")]
    let resolved_system_alias;
    #[cfg(target_os = "macos")]
    let path = {
        // macOS exposes these immutable root entries as aliases into /private.
        // Resolve only those fixed aliases; every user-controlled component
        // below them is still opened with O_NOFOLLOW | O_DIRECTORY.
        let alias = ["var", "tmp", "etc"].into_iter().find_map(|name| {
            path.strip_prefix(Path::new("/").join(name))
                .ok()
                .map(|suffix| (name, suffix))
        });
        if let Some((name, suffix)) = alias {
            resolved_system_alias = Path::new("/private").join(name).join(suffix);
            resolved_system_alias.as_path()
        } else {
            path
        }
    };

    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    ensure!(path.is_absolute(), "reference parent is not absolute");
    let mut current = File::open("/")?;
    for component in path.components() {
        let Component::Normal(name) = component else {
            ensure!(
                matches!(component, Component::RootDir),
                "reference parent contains an invalid component"
            );
            continue;
        };
        let name = CString::new(name.as_bytes()).context("reference parent contains NUL")?;
        // SAFETY: `current` owns a valid directory descriptor, `name` is
        // NUL-terminated, and a successful descriptor is transferred once.
        let fd = unsafe {
            libc::openat(
                current.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error())
                .context("failed to open a no-follow reference parent");
        }
        // SAFETY: `fd` is a fresh owned descriptor returned by `openat`.
        current = unsafe { File::from_raw_fd(fd) };
    }
    Ok(current)
}

/// Open an existing regular file without following a symlink in the filename
/// or any parent component.
#[cfg(unix)]
pub fn open_regular_file_no_follow(path: &Path) -> Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    let path = absolute_lexical_path(path)?;
    let parent_path = path.parent().context("reference path has no parent")?;
    let name = path.file_name().context("reference path has no filename")?;
    let name = CString::new(name.as_bytes()).context("reference filename contains NUL")?;
    let parent = open_directory_without_symlinks(parent_path)?;
    // SAFETY: `parent` owns a valid directory descriptor, `name` is
    // NUL-terminated, and a successful descriptor is transferred once.
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error()).context("failed to open reference no-follow");
    }
    // SAFETY: `fd` is a fresh owned descriptor returned by `openat`.
    let file = unsafe { File::from_raw_fd(fd) };
    ensure!(
        file.metadata()?.is_file(),
        "reference is not a regular file"
    );
    Ok(file)
}

#[cfg(not(unix))]
pub fn open_regular_file_no_follow(path: &Path) -> Result<File> {
    let path = absolute_lexical_path(path)?;
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        let metadata = std::fs::symlink_metadata(&current)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "reference path contains a symlink"
        );
    }
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "reference is not a regular file"
    );
    Ok(file)
}

/// Hash an already-open file without changing the caller's cursor.
pub fn sha256_open_file(file: &File) -> Result<String> {
    let mut file = file.try_clone().context("failed to clone reference file")?;
    let original_position = file
        .stream_position()
        .context("failed to read reference cursor")?;
    file.seek(SeekFrom::Start(0))
        .context("failed to rewind reference file")?;
    let hash_result = (|| -> std::io::Result<String> {
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        Ok(format!("{:x}", digest.finalize()))
    })();
    let restore_result = file.seek(SeekFrom::Start(original_position));
    match (hash_result, restore_result) {
        (Ok(digest), Ok(_)) => Ok(digest),
        (Err(error), _) => Err(error).context("failed to hash reference file"),
        (Ok(_), Err(error)) => Err(error).context("failed to restore reference cursor"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opened_file_hash_stays_bound_after_path_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reference.bin");
        std::fs::write(&path, b"original").unwrap();
        let mut file = open_regular_file_no_follow(&path).unwrap();
        file.seek(SeekFrom::Start(3)).unwrap();
        std::fs::rename(&path, dir.path().join("original.bin")).unwrap();
        std::fs::write(&path, b"replacement").unwrap();
        assert_eq!(
            sha256_open_file(&file).unwrap(),
            format!("{:x}", Sha256::digest(b"original"))
        );
        assert_eq!(file.stream_position().unwrap(), 3);
    }

    #[cfg(unix)]
    #[test]
    fn no_follow_open_rejects_symlink_filename_and_parent() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let target = real.join("reference.bin");
        std::fs::write(&target, b"bytes").unwrap();
        let file_link = dir.path().join("file-link");
        symlink(&target, &file_link).unwrap();
        assert!(open_regular_file_no_follow(&file_link).is_err());
        let parent_link = dir.path().join("parent-link");
        symlink(&real, &parent_link).unwrap();
        assert!(open_regular_file_no_follow(&parent_link.join("reference.bin")).is_err());
    }
}
