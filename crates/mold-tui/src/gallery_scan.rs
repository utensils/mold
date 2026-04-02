use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::app::GalleryEntry;

/// Returns the default gallery directory: ~/.mold/gallery/
pub fn default_gallery_dir() -> PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("gallery")
}

/// Scan for mold-generated PNG images in the output directory.
/// Only includes PNGs with valid `mold:parameters` metadata.
/// Returns entries sorted newest-first by modification time.
pub fn scan_images() -> Vec<GalleryEntry> {
    let config = mold_core::Config::load_or_default();
    let output_dir = config
        .resolved_output_dir()
        .unwrap_or_else(default_gallery_dir);

    if !output_dir.is_dir() {
        return Vec::new();
    }

    let mut entries = Vec::new();
    let walker = walkdir::WalkDir::new(&output_dir).max_depth(1).into_iter();
    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path().to_path_buf();
        if !path.is_file() {
            continue;
        }

        // Only PNG files
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        if ext.as_deref() != Some("png") {
            continue;
        }

        let timestamp = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Only include files with valid mold metadata
        if let Some(gallery_entry) = read_png_metadata(&path, timestamp) {
            entries.push(gallery_entry);
        }
    }

    // Sort by modification time, newest first
    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    entries
}

/// Try to read OutputMetadata from a PNG file's text chunks.
fn read_png_metadata(path: &Path, timestamp: u64) -> Option<GalleryEntry> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let reader = decoder.read_info().ok()?;
    let info = reader.info();

    // Check tEXt chunks (Latin-1)
    for chunk in &info.uncompressed_latin1_text {
        if chunk.keyword == "mold:parameters" {
            if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(&chunk.text) {
                return Some(GalleryEntry {
                    path: path.to_path_buf(),
                    metadata: meta,
                    generation_time_ms: None,
                    timestamp,
                });
            }
        }
    }

    // Check iTXt chunks (UTF-8)
    for chunk in &info.utf8_text {
        if chunk.keyword == "mold:parameters" {
            let text = chunk.get_text().ok()?;
            if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(&text) {
                return Some(GalleryEntry {
                    path: path.to_path_buf(),
                    metadata: meta,
                    generation_time_ms: None,
                    timestamp,
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_gallery_dir_contains_gallery() {
        let dir = default_gallery_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains("gallery"),
            "default_gallery_dir should contain 'gallery': {dir_str}"
        );
    }

    #[test]
    fn default_gallery_dir_under_mold() {
        let dir = default_gallery_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains(".mold") || dir_str.contains("mold"),
            "default_gallery_dir should be under mold dir: {dir_str}"
        );
    }

    #[test]
    fn scan_images_returns_empty_for_nonexistent_dir() {
        // With no output dir configured and default not existing, should return empty
        let entries = scan_images();
        // This test just verifies it doesn't panic
        let _ = entries;
    }
}
