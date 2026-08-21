//! Descriptor-bound directory operations.
//!
//! `mold_core::secure_file` binds a *file* to a descriptor so a later path swap
//! cannot substitute different bytes. This is the same idea for the surrounding
//! *directory*, and it exists because a pathname-based `rename` is not safe in
//! a model root.
//!
//! `CLAUDE.md`'s model-storage invariant makes shared, group-writable model
//! directories legitimate — a collaborative `0664`/`0775` umask must keep
//! working. Renaming an entry in a directory requires write permission on
//! **that directory**, not on the entry, so another member of the group can
//! rename our private 0o700 staging directory away and drop their own at the
//! same name. Everything that then resolves that pathname — including the
//! `rename` that publishes an artifact we already hashed — operates on their
//! bytes, and we return our digest for their file.
//!
//! A retained directory descriptor refers to the inode, not the name. Once
//! [`Dir`] is open, `openat`/`renameat`/`unlinkat` through it reach the
//! directory we opened even after the name has been stolen, so the hash and the
//! publish provably observe the same file.
//!
//! Unix only in substance. On other platforms the helpers degrade to their
//! pathname equivalents: the guarantee is weaker, but mold's model roots are a
//! Unix concern and the alternative is not building at all.
//!
//! This belongs beside `mold_core::secure_file` and should move there the
//! moment a second crate needs it; it lives here while PuLID is its only
//! consumer.

// Consumed only by `eva_clip_convert`, which is itself unreachable until the
// FLUX integration lands. See that module's note.
#![allow(dead_code)]

use anyhow::{Context, Result};
use std::fs::File;
use std::path::{Path, PathBuf};

/// An open directory, plus the path it was opened at **for error messages
/// only**.
///
/// Never re-resolve `display_path` to do work: it is precisely the thing this
/// type exists to stop trusting.
pub(crate) struct Dir {
    #[cfg(unix)]
    file: File,
    display_path: PathBuf,
}

impl Dir {
    pub(crate) fn display_path(&self) -> &Path {
        &self.display_path
    }
}

#[cfg(unix)]
mod imp {
    use super::*;
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd, RawFd};
    use std::os::unix::ffi::OsStrExt;

    fn c_name(name: &str) -> Result<CString> {
        CString::new(name).with_context(|| format!("{name} contains NUL"))
    }

    fn last_error<T>(what: String) -> Result<T> {
        Err(std::io::Error::last_os_error()).context(what)
    }

    impl Dir {
        pub(crate) fn raw_fd(&self) -> RawFd {
            self.file.as_raw_fd()
        }

        /// Open an existing directory, refusing a symlink at the final
        /// component. Parent components resolve normally — the caller is
        /// expected to have reached this path through `mold_core::secure_file`'s
        /// no-follow walk, or to own it already.
        pub(crate) fn open(path: &Path) -> Result<Self> {
            let c_path = CString::new(path.as_os_str().as_bytes())
                .with_context(|| format!("{} contains NUL", path.display()))?;
            // SAFETY: `c_path` is NUL-terminated and the returned descriptor is
            // transferred exactly once.
            let fd = unsafe {
                libc::open(
                    c_path.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                )
            };
            if fd < 0 {
                return last_error(format!("failed to open {} as a directory", path.display()));
            }
            // SAFETY: `fd` is a fresh owned descriptor.
            let file = unsafe { File::from_raw_fd(fd) };
            Ok(Self {
                file,
                display_path: path.to_path_buf(),
            })
        }

        /// `mkdirat` then `openat`, then prove the result is ours.
        ///
        /// `mkdirat` is exclusive, so it fails outright if the name is taken.
        /// The window between it and the `openat` is closed by the checks that
        /// follow: an attacker who removed our directory and created their own
        /// would own theirs, so the uid and mode comparison rejects it.
        pub(crate) fn create_subdir(&self, name: &str, mode: u32) -> Result<Self> {
            let c = c_name(name)?;
            // SAFETY: `self` owns a valid directory descriptor and `c` is
            // NUL-terminated.
            if unsafe { libc::mkdirat(self.raw_fd(), c.as_ptr(), mode as libc::mode_t) } < 0 {
                return last_error(format!(
                    "failed to create {name} in {}",
                    self.display_path.display()
                ));
            }
            // SAFETY: as above; the descriptor is transferred exactly once.
            let fd = unsafe {
                libc::openat(
                    self.raw_fd(),
                    c.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                )
            };
            if fd < 0 {
                return last_error(format!("failed to open the {name} staging directory"));
            }
            // SAFETY: `fd` is a fresh owned descriptor.
            let file = unsafe { File::from_raw_fd(fd) };
            let opened = Self {
                file,
                display_path: self.display_path.join(name),
            };
            opened.ensure_owned_with_mode(mode)?;
            Ok(opened)
        }

        /// Reject a directory that is not ours at exactly `mode`.
        fn ensure_owned_with_mode(&self, mode: u32) -> Result<()> {
            use std::os::unix::fs::MetadataExt;
            let metadata = self
                .file
                .metadata()
                .with_context(|| format!("failed to stat {}", self.display_path.display()))?;
            anyhow::ensure!(
                metadata.is_dir(),
                "{} is not a directory",
                self.display_path.display()
            );
            // SAFETY: `geteuid` takes no arguments and cannot fail.
            let euid = unsafe { libc::geteuid() };
            anyhow::ensure!(
                metadata.uid() == euid,
                "{} is owned by uid {} rather than {euid}",
                self.display_path.display(),
                metadata.uid()
            );
            anyhow::ensure!(
                metadata.mode() & 0o777 == mode,
                "{} has mode {:o} rather than {mode:o}",
                self.display_path.display(),
                metadata.mode() & 0o777
            );
            Ok(())
        }

        /// Create a file inside this directory that must not already exist.
        pub(crate) fn create_file(&self, name: &str, mode: u32) -> Result<File> {
            let c = c_name(name)?;
            // SAFETY: `self` owns a valid directory descriptor, `c` is
            // NUL-terminated, and the descriptor is transferred exactly once.
            let fd = unsafe {
                libc::openat(
                    self.raw_fd(),
                    c.as_ptr(),
                    libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC,
                    mode as libc::c_uint,
                )
            };
            if fd < 0 {
                return last_error(format!(
                    "failed to create {name} in {}",
                    self.display_path.display()
                ));
            }
            // SAFETY: `fd` is a fresh owned descriptor.
            Ok(unsafe { File::from_raw_fd(fd) })
        }

        /// Open an existing file inside this directory, refusing a symlink.
        pub(crate) fn open_file(&self, name: &str) -> Result<File> {
            let c = c_name(name)?;
            // SAFETY: as above.
            let fd = unsafe {
                libc::openat(
                    self.raw_fd(),
                    c.as_ptr(),
                    libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                )
            };
            if fd < 0 {
                return last_error(format!(
                    "failed to open {name} in {}",
                    self.display_path.display()
                ));
            }
            // SAFETY: `fd` is a fresh owned descriptor.
            let file = unsafe { File::from_raw_fd(fd) };
            anyhow::ensure!(
                file.metadata()?.is_file(),
                "{name} in {} is not a regular file",
                self.display_path.display()
            );
            Ok(file)
        }

        /// The path to hand an API that insists on one.
        ///
        /// Only legitimate for a directory no other user can reach, and the
        /// result must still be re-opened through the descriptor before it is
        /// trusted. See `eva_clip_convert::write_atomically`.
        pub(crate) fn unsafe_path_for(&self, name: &str) -> PathBuf {
            self.display_path.join(name)
        }

        /// `renameat`: both endpoints are descriptor-identified, so neither
        /// directory can be swapped out from under the publish.
        pub(crate) fn rename_into(
            &self,
            name: &str,
            target: &Dir,
            target_name: &str,
        ) -> Result<()> {
            let from = c_name(name)?;
            let to = c_name(target_name)?;
            // SAFETY: both descriptors are valid directories and both names are
            // NUL-terminated.
            if unsafe { libc::renameat(self.raw_fd(), from.as_ptr(), target.raw_fd(), to.as_ptr()) }
                < 0
            {
                return last_error(format!(
                    "failed to publish {name} as {target_name} in {}",
                    target.display_path.display()
                ));
            }
            Ok(())
        }

        pub(crate) fn remove_file(&self, name: &str) -> Result<()> {
            let c = c_name(name)?;
            // SAFETY: `self` owns a valid directory descriptor.
            if unsafe { libc::unlinkat(self.raw_fd(), c.as_ptr(), 0) } < 0 {
                return last_error(format!("failed to remove {name}"));
            }
            Ok(())
        }

        pub(crate) fn remove_subdir(&self, name: &str) -> Result<()> {
            let c = c_name(name)?;
            // SAFETY: as above.
            if unsafe { libc::unlinkat(self.raw_fd(), c.as_ptr(), libc::AT_REMOVEDIR) } < 0 {
                return last_error(format!("failed to remove the {name} directory"));
            }
            Ok(())
        }

        /// Flush this directory's entries so a rename survives a crash.
        pub(crate) fn sync(&self) {
            let _ = self.file.sync_all();
        }
    }

    /// `(device, inode)` of an open file.
    pub(crate) fn identity(file: &File) -> Result<(u64, u64)> {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata().context("failed to stat a file")?;
        Ok((metadata.dev(), metadata.ino()))
    }
}

#[cfg(not(unix))]
mod imp {
    use super::*;

    impl Dir {
        pub(crate) fn open(path: &Path) -> Result<Self> {
            anyhow::ensure!(path.is_dir(), "{} is not a directory", path.display());
            Ok(Self {
                display_path: path.to_path_buf(),
            })
        }

        pub(crate) fn create_subdir(&self, name: &str, _mode: u32) -> Result<Self> {
            let path = self.display_path.join(name);
            std::fs::create_dir(&path)
                .with_context(|| format!("failed to create {}", path.display()))?;
            Ok(Self { display_path: path })
        }

        pub(crate) fn create_file(&self, name: &str, _mode: u32) -> Result<File> {
            let path = self.display_path.join(name);
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .with_context(|| format!("failed to create {}", path.display()))
        }

        pub(crate) fn open_file(&self, name: &str) -> Result<File> {
            let path = self.display_path.join(name);
            File::open(&path).with_context(|| format!("failed to open {}", path.display()))
        }

        pub(crate) fn unsafe_path_for(&self, name: &str) -> PathBuf {
            self.display_path.join(name)
        }

        pub(crate) fn rename_into(
            &self,
            name: &str,
            target: &Dir,
            target_name: &str,
        ) -> Result<()> {
            std::fs::rename(
                self.display_path.join(name),
                target.display_path.join(target_name),
            )
            .context("failed to publish")
        }

        pub(crate) fn remove_file(&self, name: &str) -> Result<()> {
            std::fs::remove_file(self.display_path.join(name)).context("failed to remove")
        }

        pub(crate) fn remove_subdir(&self, name: &str) -> Result<()> {
            std::fs::remove_dir(self.display_path.join(name)).context("failed to remove")
        }

        pub(crate) fn sync(&self) {}
    }

    pub(crate) fn identity(file: &File) -> Result<(u64, u64)> {
        let length = file.metadata().context("failed to stat a file")?.len();
        Ok((length, length))
    }
}

pub(crate) use imp::identity;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn a_subdirectory_is_created_exclusively_and_owned() {
        let root = tempfile::tempdir().unwrap();
        let dir = Dir::open(root.path()).unwrap();
        let staging = dir.create_subdir("staging", 0o700).unwrap();
        assert!(staging.display_path().is_dir());
        // Exclusive: a second attempt at the same name fails.
        assert!(dir.create_subdir("staging", 0o700).is_err());
    }

    #[test]
    fn files_are_created_exclusively() {
        let root = tempfile::tempdir().unwrap();
        let dir = Dir::open(root.path()).unwrap();
        let mut file = dir.create_file("payload", 0o600).unwrap();
        file.write_all(b"hello").unwrap();
        drop(file);
        assert!(dir.create_file("payload", 0o600).is_err());
    }

    /// The point of the whole module: work continues to reach the directory we
    /// opened after its NAME has been taken over by something else.
    #[test]
    #[cfg(unix)]
    fn a_stolen_directory_name_does_not_redirect_the_descriptor() {
        let root = tempfile::tempdir().unwrap();
        let parent = Dir::open(root.path()).unwrap();
        let staging = parent.create_subdir("staging", 0o700).unwrap();
        let mut file = staging.create_file("payload", 0o600).unwrap();
        file.write_all(b"ours").unwrap();
        drop(file);

        // Someone with write access to the parent renames our directory away
        // and drops their own at the same name, with their own payload.
        std::fs::rename(root.path().join("staging"), root.path().join("stolen")).unwrap();
        std::fs::create_dir(root.path().join("staging")).unwrap();
        std::fs::write(root.path().join("staging/payload"), b"theirs").unwrap();

        // Reads through the retained descriptor still see our bytes.
        let mut contents = String::new();
        std::io::Read::read_to_string(&mut staging.open_file("payload").unwrap(), &mut contents)
            .unwrap();
        assert_eq!(contents, "ours");

        // ...and so does the publish.
        staging
            .rename_into("payload", &parent, "published")
            .unwrap();
        assert_eq!(
            std::fs::read(root.path().join("published")).unwrap(),
            b"ours"
        );
        // The substitute is untouched and was never published.
        assert_eq!(
            std::fs::read(root.path().join("staging/payload")).unwrap(),
            b"theirs"
        );
    }

    #[test]
    #[cfg(unix)]
    fn opening_a_symlinked_directory_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let real = root.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = root.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(Dir::open(&link).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn opening_a_symlinked_file_inside_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let dir = Dir::open(root.path()).unwrap();
        std::fs::write(root.path().join("victim"), b"secret").unwrap();
        std::os::unix::fs::symlink(root.path().join("victim"), root.path().join("link")).unwrap();
        assert!(dir.open_file("link").is_err());
    }

    #[test]
    fn identities_distinguish_files() {
        let root = tempfile::tempdir().unwrap();
        let dir = Dir::open(root.path()).unwrap();
        let first = dir.create_file("a", 0o600).unwrap();
        let second = dir.create_file("b", 0o600).unwrap();
        assert_ne!(identity(&first).unwrap(), identity(&second).unwrap());
        assert_eq!(
            identity(&first).unwrap(),
            identity(&dir.open_file("a").unwrap()).unwrap()
        );
    }
}
