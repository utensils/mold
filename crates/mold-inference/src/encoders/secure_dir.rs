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

// Consumed only by `pickle_convert`, which is itself unreachable until the
// FLUX integration lands. See that module's note.
#![allow(dead_code)]

use anyhow::{Context, Result};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Why a directory is unsafe to stage private work in.
///
/// Separated from its prose so the policy can be tested without matching on
/// error strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnprotectedParent {
    /// The path is absent, or is not a directory.
    NotADirectory,
    /// Not sticky, and owned by someone else — so `chmod` and `chown` are
    /// theirs to use and no mode we observe is stable.
    NotOwned { uid: u32, euid: u32 },
    /// Sticky, but owned by an untrusted third party. Sticky lets the
    /// DIRECTORY's owner rename entries too, so this one is only safe when
    /// that owner is us or root.
    StickyButForeignOwned { uid: u32, euid: u32 },
    /// Owned by us, but group- or world-writable without the sticky bit: any
    /// of those users can rename our entries out from under us.
    RenamableByOthers { mode: u32 },
}

impl std::fmt::Display for UnprotectedParent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotADirectory => write!(f, "not a directory"),
            Self::NotOwned { uid, euid } => {
                write!(f, "not sticky and owned by uid {uid} rather than {euid}")
            }
            Self::StickyButForeignOwned { uid, euid } => write!(
                f,
                "sticky but owned by uid {uid} rather than {euid} or root, and a sticky \
                 directory's owner may rename entries inside it"
            ),
            Self::RenamableByOthers { mode } => write!(
                f,
                "mode {mode:o} is writable by group or other without the sticky bit, so \
                 another user could rename a staging directory away"
            ),
        }
    }
}

/// Root, which is trusted by definition: it can already replace anything we
/// own, so requiring protection from it is meaningless.
const ROOT_UID: u32 = 0;

/// Could a user we do not trust rename an entry inside a directory with this
/// owner and mode?
///
/// Pure so the root-owned case is testable — a test cannot `chown` a fixture
/// to root, and Linux's `/tmp` is exactly that case.
///
/// Renaming or unlinking an entry needs write permission on the CONTAINING
/// directory, not on the entry, so a `0o700` staging directory inside a
/// group-writable parent can still be renamed away and replaced wholesale.
/// That is the hole that put the pickle parser in front of attacker bytes.
///
/// Two shapes are safe, and the STICKY test has to come first:
///
/// - **Sticky**, owned by us or by root. The sticky bit restricts rename and
///   unlink to each entry's own owner, the directory's owner, or root — so
///   other unprivileged users cannot touch our entries no matter how open the
///   write bits are. This is what makes Linux's root-owned `1777` `/tmp`
///   usable, and checking ownership before stickiness (as this did originally)
///   rejects it and breaks first-use conversion on any headless Linux box with
///   neither `XDG_RUNTIME_DIR` nor `TMPDIR` set.
///
///   The owner still matters, just less: sticky lets the DIRECTORY's owner
///   rename entries too, so a sticky directory belonging to some other
///   unprivileged user is not safe. Us or root, and nobody else.
///
/// - **Not sticky**, owned by us, and not writable by group or other. `0o700`
///   and `0o755` both qualify; `$XDG_RUNTIME_DIR` and macOS's per-user
///   `$TMPDIR` are `0o700`.
fn parent_policy(
    dir_uid: u32,
    mode: u32,
    current_uid: u32,
) -> std::result::Result<(), UnprotectedParent> {
    const STICKY: u32 = 0o1000;
    const OTHERS_WRITE: u32 = 0o022;

    if mode & STICKY != 0 {
        return if dir_uid == current_uid || dir_uid == ROOT_UID {
            Ok(())
        } else {
            Err(UnprotectedParent::StickyButForeignOwned {
                uid: dir_uid,
                euid: current_uid,
            })
        };
    }
    if dir_uid != current_uid {
        return Err(UnprotectedParent::NotOwned {
            uid: dir_uid,
            euid: current_uid,
        });
    }
    if mode & OTHERS_WRITE != 0 {
        return Err(UnprotectedParent::RenamableByOthers {
            mode: mode & 0o7777,
        });
    }
    Ok(())
}

/// Can a user we do not trust rename an entry inside `path`?
///
/// See [`parent_policy`] for the rule; this only supplies the owner and mode.
#[cfg(unix)]
pub(crate) fn parent_protects_entries(path: &Path) -> std::result::Result<(), UnprotectedParent> {
    use std::os::unix::fs::MetadataExt;
    let Ok(metadata) = std::fs::metadata(path) else {
        return Err(UnprotectedParent::NotADirectory);
    };
    if !metadata.is_dir() {
        return Err(UnprotectedParent::NotADirectory);
    }
    // SAFETY: `geteuid` takes no arguments and cannot fail.
    let euid = unsafe { libc::geteuid() };
    parent_policy(metadata.uid(), metadata.mode(), euid)
}

/// Windows answers only the "is it a directory" half.
///
/// The unix rule is expressed in owner-and-mode terms (sticky bit, group/other
/// write) that have no direct NTFS equivalent — the same question there is a
/// DACL walk for `FILE_ADD_FILE`/`DELETE` grants to principals other than the
/// current user. Until that exists, this admits any directory, so the
/// staging-hijack defence documented in [`parent_policy`] is NOT in force on
/// Windows: the private copy and the `renameat` between retained descriptors
/// still hold, but a model root deliberately shared with another principal is
/// not detected as one. `%LOCALAPPDATA%\Temp` and `%APPDATA%` — where the
/// staging root actually lands — are per-user by default, which is why this is
/// a documented gap rather than an open door.
#[cfg(not(unix))]
pub(crate) fn parent_protects_entries(path: &Path) -> std::result::Result<(), UnprotectedParent> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(UnprotectedParent::NotADirectory)
    }
}

/// An open directory, plus the path it was opened at **for error messages
/// only**.
///
/// Never re-resolve `display_path` to do work: it is precisely the thing this
/// type exists to stop trusting.
/// Create mode for a file that will be PUBLISHED into the model root.
///
/// `0o666` is not a permission grant: `openat`'s mode argument is masked by the
/// process umask, so this is exactly "whatever this process creates files as".
/// A collaborative `0o002` umask therefore yields `0o664` and a shared model
/// root stays readable by the group that prepared it, which is what
/// `CLAUDE.md`'s model-storage invariant requires of runnable weights — an
/// explicit `0o600` here survives the publish rename and leaves every other
/// user unable to read the artifact, reconverting it on every load.
pub(crate) const PUBLISHED_FILE_MODE: u32 = 0o666;

/// Create mode for a file that is NEVER published: the transient private copy
/// of a source checkpoint, staged in a world-writable tmp root. Owner-only is
/// right here for the same reason it is wrong above — nothing else has any
/// business reading it, and it never becomes a runnable artifact.
pub(crate) const PRIVATE_FILE_MODE: u32 = 0o600;

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
        /// trusted. See `pickle_convert::write_atomically`.
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

    const US: u32 = 501;
    const ROOT: u32 = 0;
    const MALLORY: u32 = 1234;

    /// Linux's `/tmp` is root-owned `1777`, and on a headless box with neither
    /// `XDG_RUNTIME_DIR` nor `TMPDIR` set it is the ONLY candidate. Checking
    /// ownership before stickiness rejected it and broke first-use conversion
    /// outright, so this case is pinned on its own.
    #[test]
    fn root_owned_sticky_tmp_is_accepted() {
        assert_eq!(parent_policy(ROOT, 0o1777, US), Ok(()));
        assert_eq!(parent_policy(ROOT, 0o1755, US), Ok(()));
    }

    /// The real thing, not a fixture: a sticky system `/tmp` must be accepted.
    ///
    /// macOS's `/private/tmp` and Linux's `/tmp` are both root-owned `1777`,
    /// so this is the reported failure exactly — and it is what
    /// `std::env::temp_dir()` returns when `TMPDIR` is unset, which is the
    /// normal state of a headless server.
    #[test]
    #[cfg(unix)]
    fn a_sticky_system_tmp_is_accepted() {
        use std::os::unix::fs::MetadataExt;
        let Ok(metadata) = std::fs::metadata("/tmp") else {
            return; // No /tmp at all; nothing to assert.
        };
        if metadata.mode() & 0o1000 == 0 {
            return; // Not sticky on this box, so not the case under test.
        }
        assert_eq!(
            parent_protects_entries(Path::new("/tmp")),
            Ok(()),
            "/tmp is sticky (mode {:o}, uid {}) and must be usable",
            metadata.mode() & 0o7777,
            metadata.uid()
        );
    }

    /// The whole matrix, as a pure function — a test cannot `chown` a fixture
    /// to root or to a stranger, so the filesystem test below cannot reach
    /// these rows.
    #[test]
    fn the_parent_policy_matrix_is_pinned() {
        // Sticky, owned by us or root: safe however open the write bits are,
        // because sticky restricts rename to each entry's own owner.
        for mode in [0o1777_u32, 0o1770, 0o1700, 0o1755] {
            assert_eq!(parent_policy(US, mode, US), Ok(()), "{mode:o} owned by us");
            assert_eq!(
                parent_policy(ROOT, mode, US),
                Ok(()),
                "{mode:o} owned by root"
            );
        }

        // Sticky but owned by another unprivileged user: NOT safe. Sticky lets
        // the directory's owner rename entries too, so Mallory could still
        // swap ours.
        assert_eq!(
            parent_policy(MALLORY, 0o1777, US),
            Err(UnprotectedParent::StickyButForeignOwned {
                uid: MALLORY,
                euid: US
            })
        );

        // Not sticky, owned by us, nobody else can write: safe.
        for mode in [0o700_u32, 0o755, 0o750, 0o500] {
            assert_eq!(parent_policy(US, mode, US), Ok(()), "{mode:o}");
        }

        // Not sticky, owned by us, group- or world-writable: the original
        // hole.
        for mode in [0o770_u32, 0o777, 0o707, 0o702] {
            assert_eq!(
                parent_policy(US, mode, US),
                Err(UnprotectedParent::RenamableByOthers { mode }),
                "{mode:o}"
            );
        }

        // Not sticky and not ours: refused whatever the mode says, because the
        // owner can change it at will.
        for mode in [0o700_u32, 0o755, 0o777] {
            assert_eq!(
                parent_policy(ROOT, mode, US),
                Err(UnprotectedParent::NotOwned {
                    uid: ROOT,
                    euid: US
                }),
                "{mode:o}"
            );
            assert_eq!(
                parent_policy(MALLORY, mode, US),
                Err(UnprotectedParent::NotOwned {
                    uid: MALLORY,
                    euid: US
                }),
                "{mode:o}"
            );
        }

        // Running as root ourselves: root-owned is our own.
        assert_eq!(parent_policy(ROOT, 0o700, ROOT), Ok(()));
    }

    /// Every rejection has to say which candidate failed and why, because the
    /// failure users hit is "conversion refuses and I do not know where to
    /// point TMPDIR".
    #[test]
    fn every_rejection_explains_itself() {
        for reason in [
            UnprotectedParent::NotADirectory,
            UnprotectedParent::NotOwned { uid: 0, euid: 501 },
            UnprotectedParent::StickyButForeignOwned {
                uid: 1234,
                euid: 501,
            },
            UnprotectedParent::RenamableByOthers { mode: 0o777 },
        ] {
            let rendered = reason.to_string();
            assert!(!rendered.is_empty(), "{reason:?} renders nothing");
            assert!(
                !rendered.contains("  "),
                "{reason:?} has a line-continuation artefact: {rendered}"
            );
        }
    }

    /// The same policy over real fixtures. Limited to directories we own —
    /// `the_parent_policy_matrix_is_pinned` covers root-owned and
    /// foreign-owned, which a test cannot create.
    #[test]
    #[cfg(unix)]
    fn the_parent_policy_accepts_only_unrenamable_directories() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let make = |name: &str, mode: u32| {
            let path = root.path().join(name);
            std::fs::create_dir(&path).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            path
        };

        // Owned by us and not writable by anyone else.
        assert_eq!(parent_protects_entries(&make("private", 0o700)), Ok(()));
        assert_eq!(parent_protects_entries(&make("readable", 0o755)), Ok(()));
        // Sticky: others may write, but only an entry's owner may rename it.
        assert_eq!(parent_protects_entries(&make("sticky", 0o1777)), Ok(()));
        assert_eq!(
            parent_protects_entries(&make("sticky_group", 0o1770)),
            Ok(())
        );

        // Group- or world-writable without the sticky bit is the shape that
        // lets another member rename our staging directory away.
        for (name, mode) in [("group", 0o770_u32), ("world", 0o777), ("other", 0o707)] {
            let error = parent_protects_entries(&make(name, mode)).unwrap_err();
            assert!(
                matches!(error, UnprotectedParent::RenamableByOthers { .. }),
                "{name} ({mode:o}) was accepted: {error:?}"
            );
        }

        assert_eq!(
            parent_protects_entries(&root.path().join("absent")),
            Err(UnprotectedParent::NotADirectory)
        );
        let file = root.path().join("file");
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(
            parent_protects_entries(&file),
            Err(UnprotectedParent::NotADirectory)
        );
    }

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
