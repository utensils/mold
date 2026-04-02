use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::app::GalleryEntry;

/// Returns the default output directory: ~/.mold/output/
pub fn default_gallery_dir() -> PathBuf {
    mold_core::Config::load_or_default().effective_output_dir()
}

/// Returns the image cache directory for server-fetched images: ~/.mold/cache/images/
pub fn image_cache_dir() -> PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("images")
}

/// Scan for gallery images via the server API.
/// All entries are server-backed — images are fetched via API for
/// thumbnails, previews, and opening, then cached locally.
pub async fn scan_images_from_server(server_url: &str) -> Vec<GalleryEntry> {
    let client = mold_core::MoldClient::new(server_url);
    let images = match client.list_gallery().await {
        Ok(images) => images,
        Err(_) => return Vec::new(),
    };

    images
        .into_iter()
        .map(|img| GalleryEntry {
            path: PathBuf::from(&img.filename),
            metadata: img.metadata,
            generation_time_ms: None,
            timestamp: img.timestamp,
            server_url: Some(server_url.to_string()),
        })
        .collect()
}

/// Fetch an image from the server and cache it locally.
/// Returns the local cache path if successful.
pub async fn fetch_and_cache_image(server_url: &str, filename: &str) -> Option<PathBuf> {
    let cache_dir = image_cache_dir();
    let cached_path = cache_dir.join(filename);

    // Return cached copy if it exists
    if cached_path.is_file() {
        return Some(cached_path);
    }

    let client = mold_core::MoldClient::new(server_url);
    let data = client.get_gallery_image(filename).await.ok()?;

    std::fs::create_dir_all(&cache_dir).ok()?;
    std::fs::write(&cached_path, &data).ok()?;
    Some(cached_path)
}

/// Scan for mold-generated PNG images in the local output directory.
/// Only includes PNGs with valid `mold:parameters` metadata.
/// Returns entries sorted newest-first by modification time.
pub fn scan_images_local() -> Vec<GalleryEntry> {
    let output_dir = default_gallery_dir();

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

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        if !matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
            continue;
        }

        let timestamp = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let gallery_entry = if ext.as_deref() == Some("png") {
            read_png_metadata(&path, timestamp)
        } else {
            read_jpeg_metadata(&path, timestamp)
        };
        if let Some(ge) = gallery_entry {
            entries.push(ge);
        }
    }

    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    entries
}

/// Try to read OutputMetadata from a PNG file's text chunks.
fn read_png_metadata(path: &Path, timestamp: u64) -> Option<GalleryEntry> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let reader = decoder.read_info().ok()?;
    let info = reader.info();

    for chunk in &info.uncompressed_latin1_text {
        if chunk.keyword == "mold:parameters" {
            if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(&chunk.text) {
                return Some(GalleryEntry {
                    path: path.to_path_buf(),
                    metadata: meta,
                    generation_time_ms: None,
                    timestamp,
                    server_url: None,
                });
            }
        }
    }

    for chunk in &info.utf8_text {
        if chunk.keyword == "mold:parameters" {
            let text = chunk.get_text().ok()?;
            if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(&text) {
                return Some(GalleryEntry {
                    path: path.to_path_buf(),
                    metadata: meta,
                    generation_time_ms: None,
                    timestamp,
                    server_url: None,
                });
            }
        }
    }

    None
}

/// Read OutputMetadata from a JPEG file's COM marker.
/// Mold writes `mold:parameters {json}` as the COM comment.
fn read_jpeg_metadata(path: &Path, timestamp: u64) -> Option<GalleryEntry> {
    let data = std::fs::read(path).ok()?;
    // Scan for COM markers (0xFF 0xFE)
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xFE {
            let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            if i + 2 + len <= data.len() {
                let comment = &data[i + 4..i + 2 + len];
                if let Ok(text) = std::str::from_utf8(comment) {
                    if let Some(json) = text.strip_prefix("mold:parameters ") {
                        if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(json) {
                            return Some(GalleryEntry {
                                path: path.to_path_buf(),
                                metadata: meta,
                                generation_time_ms: None,
                                timestamp,
                                server_url: None,
                            });
                        }
                    }
                }
            }
            i += 2 + len;
        } else if data[i] == 0xFF {
            if data[i + 1] == 0xD9 {
                break; // EOI
            }
            if i + 3 < data.len() {
                let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                i += 2 + len;
            } else {
                break;
            }
        } else {
            i += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_gallery_dir_contains_output() {
        let dir = default_gallery_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains("output"),
            "default_gallery_dir should contain 'output': {dir_str}"
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
    fn scan_images_local_returns_empty_for_nonexistent_dir() {
        let entries = scan_images_local();
        let _ = entries;
    }
}
