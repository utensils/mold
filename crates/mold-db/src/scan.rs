//! Shared output-directory walker for gallery consumers.
//!
//! One home for the "walk the output dir, filter to gallery formats,
//! apply the size/header/solid-black validity guards" pass that
//! reconcile, the server's filesystem gallery scan, and the TUI's
//! fallback walk each hand-rolled. Deliberately two-level: the walker
//! yields file facts only — metadata parsing stays a separate leaf
//! ([`crate::metadata_io::read_or_synthesize`]) so reconcile keeps its
//! parse-only-new/changed-files property.

use std::path::{Path, PathBuf};

use mold_core::OutputFormat;

use crate::metadata_io::{format_from_path, is_valid_gallery_file};

/// A gallery-format file that passed the validity guards.
pub struct ScannedFile {
    pub path: PathBuf,
    pub filename: String,
    pub format: OutputFormat,
    /// File mtime in epoch ms; `None` when stat failed.
    pub mtime_ms: Option<i64>,
    /// File size in bytes; `None` when stat failed.
    pub size_bytes: Option<i64>,
}

impl ScannedFile {
    /// Mtime as epoch seconds (0 when stat failed) — the value filename
    /// synthesis and gallery sorting key on.
    pub fn timestamp_secs(&self) -> u64 {
        self.mtime_ms.map(|ms| ms / 1000).unwrap_or(0).max(0) as u64
    }

    /// Size as u64 (0 when stat failed).
    pub fn size_u64(&self) -> u64 {
        self.size_bytes.unwrap_or(0).max(0) as u64
    }
}

/// One walk observation. Skips are reported (not silently dropped) so
/// consumers like reconcile can keep their per-category counters.
pub enum ScanItem {
    Valid(ScannedFile),
    /// Extension isn't a gallery format.
    SkippedUnrelated,
    /// Recognized extension but failed the size/header/solid-black guards.
    SkippedInvalid,
}

/// Walk `dir` (files only, depth 1) and classify every entry. Metadata
/// is NOT parsed here — callers decide when to pay for that.
pub fn scan_output_dir(dir: &Path) -> impl Iterator<Item = ScanItem> {
    walkdir::WalkDir::new(dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let Some(format) = format_from_path(path) else {
                return Some(ScanItem::SkippedUnrelated);
            };
            // Non-UTF8 filenames can't round-trip through the DB or the
            // HTTP API; skip them silently like every consumer did.
            let filename = path.file_name().and_then(|f| f.to_str())?.to_string();
            let (mtime_ms, size_bytes) = stat_to_pair(entry.metadata().ok().as_ref());
            let raw_size = size_bytes.unwrap_or(0).max(0) as u64;
            if !is_valid_gallery_file(path, format, raw_size) {
                return Some(ScanItem::SkippedInvalid);
            }
            Some(ScanItem::Valid(ScannedFile {
                path: path.to_path_buf(),
                filename,
                format,
                mtime_ms,
                size_bytes,
            }))
        })
}

pub(crate) fn stat_to_pair(meta: Option<&std::fs::Metadata>) -> (Option<i64>, Option<i64>) {
    let Some(m) = meta else {
        return (None, None);
    };
    let size = Some(m.len() as i64);
    let mtime = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64);
    (mtime, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn write_valid_png(path: &Path) {
        let img = ImageBuffer::from_fn(64u32, 64u32, |x, y| {
            if ((x / 8) + (y / 8)) % 2 == 0 {
                Rgb([255u8, 64, 32])
            } else {
                Rgb([16u8, 200, 240])
            }
        });
        img.save(path).unwrap();
    }

    fn write_valid_mp4(path: &Path) {
        let mut bytes = Vec::with_capacity(8192);
        bytes.extend_from_slice(&[0u8, 0, 0, 0x20]);
        bytes.extend_from_slice(b"ftyp");
        bytes.extend_from_slice(b"isomavc1mp41....");
        bytes.resize(8192, 0);
        std::fs::write(path, &bytes).unwrap();
    }

    /// Ported from the shape of the server's
    /// `scan_gallery_dir_filters_invalid_and_keeps_valid` test: valid
    /// files survive; truncated stubs, undecodable rasters, and
    /// ftyp-less MP4s are reported as invalid; foreign extensions as
    /// unrelated.
    #[test]
    fn scan_classifies_valid_invalid_and_unrelated() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("good.png"));
        write_valid_mp4(&tmp.path().join("good.mp4"));
        std::fs::write(tmp.path().join("tiny.png"), b"x").unwrap();
        std::fs::write(tmp.path().join("bogus.png"), vec![0xAAu8; 4096]).unwrap();
        std::fs::write(tmp.path().join("noftyp.mp4"), vec![0u8; 8192]).unwrap();
        std::fs::write(tmp.path().join("notes.txt"), b"hello").unwrap();

        let mut valid = Vec::new();
        let mut invalid = 0;
        let mut unrelated = 0;
        for item in scan_output_dir(tmp.path()) {
            match item {
                ScanItem::Valid(f) => valid.push(f.filename),
                ScanItem::SkippedInvalid => invalid += 1,
                ScanItem::SkippedUnrelated => unrelated += 1,
            }
        }
        valid.sort();
        assert_eq!(valid, vec!["good.mp4", "good.png"]);
        assert_eq!(invalid, 3);
        assert_eq!(unrelated, 1);
    }

    #[test]
    fn scanned_file_carries_stat_facts() {
        let tmp = tempfile::tempdir().unwrap();
        write_valid_png(&tmp.path().join("good.png"));

        let f = scan_output_dir(tmp.path())
            .find_map(|i| match i {
                ScanItem::Valid(f) => Some(f),
                _ => None,
            })
            .unwrap();
        assert_eq!(f.format, OutputFormat::Png);
        assert!(f.size_u64() > 256);
        assert!(f.timestamp_secs() > 1_600_000_000);
        assert!(f.mtime_ms.is_some());
    }

    #[test]
    fn scan_of_missing_dir_yields_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope");
        assert_eq!(scan_output_dir(&missing).count(), 0);
    }
}
