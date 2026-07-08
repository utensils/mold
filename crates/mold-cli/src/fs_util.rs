//! Small filesystem helpers shared across CLI commands.

use std::path::Path;

/// Count files and total bytes in a directory (recursive, no symlink following).
/// Avoids double-counting symlinked HF cache blobs.
pub(crate) fn dir_stats(path: &Path) -> (u64, u64) {
    let mut files = 0u64;
    let mut bytes = 0u64;
    for entry in walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .flatten()
    {
        if entry.file_type().is_file() {
            files += 1;
            bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    (files, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_stats_counts_nested_files_without_following_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), [0u8; 10]).unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("b.bin"), [0u8; 32]).unwrap();

        // A symlinked directory full of data must not be double-counted.
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("blob.bin"), [0u8; 1000]).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), dir.path().join("linked")).unwrap();

        let (files, bytes) = dir_stats(dir.path());
        assert_eq!(files, 2);
        assert_eq!(bytes, 42);
    }
}
