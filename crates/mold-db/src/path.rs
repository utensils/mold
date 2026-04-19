//! Shared path normalization so every caller keys rows the same way.
//!
//! Two code paths used to build DB keys from `output_dir`:
//!   - The CLI's `record_local_save` canonicalizes via `std::fs::canonicalize`,
//!     which resolves symlinks and rewrites `/tmp/...` to `/private/tmp/...`
//!     on macOS.
//!   - The server's `queue::save_*` and `routes::list_gallery` used the raw
//!     `config.effective_output_dir()` value, which does neither.
//!
//! With the same `MOLD_HOME/mold.db` open in both processes, the same file
//! on disk produced two rows under different `output_dir` keys and
//! appeared twice in the gallery. Normalizing in one place makes the
//! divergence impossible.

use std::path::{Path, PathBuf};

/// Normalize a directory path used as a DB key. Best-effort: returns the
/// canonical path when `std::fs::canonicalize` succeeds, and falls back to
/// the input otherwise so tests and not-yet-existing dirs still work.
///
/// Canonicalization here resolves symlinks, absolutizes relative paths,
/// and collapses OS-specific prefixes (e.g. macOS `/tmp` → `/private/tmp`).
/// Called by every `MetadataDb` method that takes `output_dir`, plus
/// `GenerationRecord::from_save`, so callers never have to remember.
pub fn canonical_dir(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Stringify a canonicalized directory for storage. Wraps [`canonical_dir`]
/// and hands back a `String` since the unique index is `TEXT`-typed.
pub fn canonical_dir_string(path: &Path) -> String {
    canonical_dir(path).to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_dir_resolves_existing_path() {
        let tmp = tempfile::tempdir().unwrap();
        let canon = canonical_dir(tmp.path());
        // The canonical form may differ from the input (e.g. /tmp vs
        // /private/tmp on macOS) but it must exist and it must round-trip.
        assert!(canon.exists());
        let twice = canonical_dir(&canon);
        assert_eq!(canon, twice, "canonicalization must be idempotent");
    }

    #[test]
    fn canonical_dir_falls_back_for_missing_path() {
        let p = Path::new("/definitely/not/a/dir/here");
        assert_eq!(canonical_dir(p), p.to_path_buf());
    }

    #[test]
    fn canonical_dir_string_matches_canonical_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            canonical_dir_string(tmp.path()),
            canonical_dir(tmp.path()).to_string_lossy().into_owned(),
        );
    }

    /// The whole reason this helper exists: on macOS, `/tmp` and
    /// `/private/tmp` are the same directory via a symlink, but the string
    /// forms differ. Both inputs must produce the same normalized key.
    #[cfg(target_os = "macos")]
    #[test]
    fn canonical_dir_resolves_macos_tmp_symlink() {
        let tmp = tempfile::tempdir_in("/tmp").unwrap();
        let via_tmp = tmp.path().to_path_buf();
        let via_private_tmp =
            Path::new("/private").join(via_tmp.strip_prefix("/").unwrap_or(&via_tmp));
        assert!(via_private_tmp.exists(), "test setup sanity");
        assert_eq!(
            canonical_dir(&via_tmp),
            canonical_dir(&via_private_tmp),
            "/tmp and /private/tmp must normalize to the same key"
        );
    }
}
