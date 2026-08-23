//! One durable-rename fence for the whole server.
//!
//! Publishing an artifact is `write temp` → `rename` → **sync the directory**.
//! On unix that last step is what makes the rename survive a power loss:
//! without it the renamed entry can still be lost while the file's own data is
//! on disk, which for the gallery and the chain journal means a print that
//! exists as bytes and not as a name.
//!
//! There were five private copies of this helper (`chain_execution`,
//! `chain_job_runner`, `batch_parent`, `batch_transaction`, and
//! `gallery_authority`). Four were `#[cfg(unix)]` with a `Ok(())` twin, so on
//! Windows they silently did nothing; the fifth opened the directory with a
//! plain `File::open`, which Windows answers with ERROR_ACCESS_DENIED, so it
//! failed startup recovery outright.
//!
//! **What Windows actually promises here is weaker than unix, and it is
//! spelled out rather than implied.** `FlushFileBuffers` — what `sync_all`
//! calls — is documented for file and volume handles; Microsoft does not
//! document it as committing a directory's entries, and `std::fs::rename` does
//! not pass `MOVEFILE_WRITE_THROUGH`. So this is a best-effort flush on
//! Windows, not the crash-consistency guarantee the unix arm buys. Do not
//! restate it as equivalence.

use std::path::Path;

/// Flush a directory's own metadata so entries renamed into it are durable.
///
/// Unix behaviour is byte-identical to the four `#[cfg(unix)]` copies this
/// replaced: every error propagates, and callers that want to tolerate a
/// filesystem which cannot fsync a directory keep doing so through
/// `batch_transaction::directory_sync_is_unsupported`.
#[cfg(unix)]
pub(crate) fn sync_directory(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

#[cfg(windows)]
pub(crate) fn sync_directory(path: &Path) -> std::io::Result<()> {
    match windows_flush_directory(path) {
        Ok(()) => Ok(()),
        // A directory flush is a durability *upgrade* on an artifact whose
        // bytes are already written and whose rename has already landed.
        // Refusing the whole publication because the filesystem or the ACL
        // will not grant the flush trades a weaker guarantee for a lost print
        // — and worse, `queue.rs` holds the queue row on a failed
        // publication and `dispatch_attempts` is capped at 2, so a gallery on
        // exFAT, an SD card, or a network share would burn both attempts and
        // strand the job forever. `gallery_authority` calls this during
        // startup recovery, so the same refusal would stop the server booting.
        Err(error) if windows_directory_flush_unavailable(&error) => {
            tracing::warn!(
                directory = %path.display(),
                %error,
                "this filesystem does not permit flushing a directory; the entry is published \
                 with best-effort directory durability"
            );
            Ok(())
        }
        Err(error) => Err(error),
    }
}

#[cfg(windows)]
fn windows_flush_directory(path: &Path) -> std::io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;

    // Two Windows requirements, and missing either one is ERROR_ACCESS_DENIED
    // rather than a no-op. `FILE_FLAG_BACKUP_SEMANTICS` is the documented flag
    // that makes `CreateFileW` return a directory handle at all; and
    // `FlushFileBuffers` requires the handle to carry GENERIC_WRITE, so a
    // read-only directory handle opens fine and then fails on the flush.
    // Measured on NTFS: read-only flushes with os error 5, write-only and
    // read+write both succeed — so ask for write alone and leave read out of
    // the access mask an ACL could refuse.
    std::fs::OpenOptions::new()
        .write(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?
        .sync_all()
}

/// Distinguish "this filesystem/ACL will not flush a directory" from a real
/// I/O failure. Deliberately Windows-only: the unix arm keeps propagating
/// every error exactly as it did before this module existed.
#[cfg(windows)]
fn windows_directory_flush_unavailable(error: &std::io::Error) -> bool {
    use windows_sys::Win32::Foundation::{
        ERROR_ACCESS_DENIED, ERROR_INVALID_FUNCTION, ERROR_INVALID_PARAMETER, ERROR_NOT_SUPPORTED,
        ERROR_WRITE_PROTECT,
    };

    if error.kind() == std::io::ErrorKind::Unsupported {
        return true;
    }
    error.raw_os_error().is_some_and(|code| {
        let code = code as u32;
        // FAT/exFAT and many SMB servers answer INVALID_FUNCTION or
        // NOT_SUPPORTED; a read-only volume answers WRITE_PROTECT; a
        // restrictive ACL or a read-only attribute answers ACCESS_DENIED.
        code == ERROR_INVALID_FUNCTION
            || code == ERROR_ACCESS_DENIED
            || code == ERROR_NOT_SUPPORTED
            || code == ERROR_INVALID_PARAMETER
            || code == ERROR_WRITE_PROTECT
    })
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

    /// A refusal to flush must never fail the publication whose bytes and
    /// rename already landed — see the comment on the Windows arm. A missing
    /// directory (checked above) stays an error, so the two are distinguished.
    #[cfg(windows)]
    #[test]
    fn a_directory_that_refuses_the_flush_is_tolerated_not_fatal() {
        for code in [1i32, 5, 50, 87, 19] {
            let error = std::io::Error::from_raw_os_error(code);
            assert!(
                windows_directory_flush_unavailable(&error),
                "os error {code} should be tolerated"
            );
        }
        // A genuinely broken device is not a "cannot flush" answer.
        assert!(!windows_directory_flush_unavailable(
            &std::io::Error::from_raw_os_error(1117) // ERROR_IO_DEVICE
        ));
    }
}
