//! One durable-rename fence for the whole server.
//!
//! Publishing an artifact is `write temp` → `rename` → **sync the directory**.
//! The last step is what makes the rename survive a power loss: without it the
//! renamed entry can still be lost while the file's own data is on disk, which
//! for the gallery and the chain journal means a print that exists as bytes and
//! not as a name.
//!
//! There were four private copies of this helper (`chain_execution`,
//! `chain_job_runner`, `batch_parent`, `batch_transaction`), each `#[cfg(unix)]`
//! with a `#[cfg(not(unix))] Ok(())` twin — so on Windows every one of them
//! silently did nothing and the durability argument above simply did not hold.
//! Windows *can* express it: a directory opened with
//! `FILE_FLAG_BACKUP_SEMANTICS` accepts `FlushFileBuffers`, which is what
//! `File::sync_all` calls, and NTFS honours it for directory metadata. That is
//! the same guarantee the unix arm asks for, so both platforms now make the
//! same promise rather than one of them quietly making none.

use std::path::Path;

/// Flush a directory's own metadata so entries renamed into it are durable.
#[cfg(unix)]
pub(crate) fn sync_directory(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

#[cfg(windows)]
pub(crate) fn sync_directory(path: &Path) -> std::io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;

    // Two Windows requirements, and missing either one is ERROR_ACCESS_DENIED
    // rather than a no-op. `FILE_FLAG_BACKUP_SEMANTICS` is the documented flag
    // that makes `CreateFileW` return a directory handle at all; and
    // `FlushFileBuffers` — which is what `sync_all` calls — requires the handle
    // to carry GENERIC_WRITE, so a read-only directory handle opens fine and
    // then fails on the flush. Measured on NTFS: read-only flushes with os
    // error 5, write and read+write both succeed.
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?
        .sync_all()
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn sync_directory(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the helper is that it SUCCEEDS on a directory. A platform
    /// whose arm still fails would otherwise turn every publication into an
    /// error at the last step, which is worse than the no-op it replaced.
    #[test]
    fn syncing_a_real_directory_succeeds_on_this_platform() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("entry"), b"published").expect("write");
        sync_directory(dir.path()).expect("directory sync");
    }

    #[test]
    fn syncing_a_missing_directory_reports_the_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(sync_directory(&dir.path().join("absent")).is_err());
    }
}
