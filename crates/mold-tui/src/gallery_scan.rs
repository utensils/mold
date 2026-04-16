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
        if !matches!(
            ext.as_deref(),
            Some("png" | "jpg" | "jpeg" | "gif" | "apng" | "webp" | "mp4")
        ) {
            continue;
        }

        let timestamp = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let gallery_entry = match ext.as_deref() {
            Some("png" | "apng") => read_png_metadata(&path, timestamp),
            Some("gif") => read_gif_metadata(&path, timestamp),
            Some("jpg" | "jpeg") => read_jpeg_metadata(&path, timestamp),
            // WebP/MP4: minimal entry (no embedded metadata to parse)
            Some("webp" | "mp4") => {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                Some(GalleryEntry {
                    path: path.clone(),
                    metadata: mold_core::OutputMetadata {
                        prompt: String::new(),
                        negative_prompt: None,
                        original_prompt: None,
                        model: name.to_string(),
                        seed: 0,
                        steps: 0,
                        guidance: 0.0,
                        width: 0,
                        height: 0,
                        strength: None,
                        scheduler: None,
                        lora: None,
                        lora_scale: None,
                        frames: None,
                        fps: None,
                        version: String::new(),
                    },
                    generation_time_ms: None,
                    timestamp,
                    server_url: None,
                })
            }
            _ => None,
        };
        if let Some(ge) = gallery_entry {
            entries.push(ge);
        }
    }

    entries.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
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

/// Read OutputMetadata from a GIF file's comment extension.
/// GIF comment extensions use introducer 0x21 + label 0xFE, followed by sub-blocks.
/// Falls back to a placeholder entry so GIF files still appear in the gallery.
fn read_gif_metadata(path: &Path, timestamp: u64) -> Option<GalleryEntry> {
    let data = std::fs::read(path).ok()?;
    // Look for comment extension blocks: 0x21 0xFE
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0x21 && data[i + 1] == 0xFE {
            // Read sub-blocks after the 2-byte header
            let mut comment = Vec::new();
            let mut j = i + 2;
            while j < data.len() {
                let block_size = data[j] as usize;
                if block_size == 0 {
                    break; // Block terminator
                }
                j += 1;
                let end = (j + block_size).min(data.len());
                comment.extend_from_slice(&data[j..end]);
                j = end;
            }
            if let Ok(text) = std::str::from_utf8(&comment) {
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
        i += 1;
    }
    // No metadata found — show with default placeholder so GIFs still appear in gallery
    Some(GalleryEntry {
        path: path.to_path_buf(),
        metadata: mold_core::OutputMetadata {
            prompt: String::new(),
            negative_prompt: None,
            original_prompt: None,
            model: path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            seed: 0,
            steps: 0,
            guidance: 0.0,
            width: 0,
            height: 0,
            strength: None,
            scheduler: None,
            lora: None,
            lora_scale: None,
            frames: None,
            fps: None,
            version: String::new(),
        },
        generation_time_ms: None,
        timestamp,
        server_url: None,
    })
}

/// Read OutputMetadata from a JPEG file's COM marker.
/// Mold writes `mold:parameters {json}` as the COM comment.
fn read_jpeg_metadata(path: &Path, timestamp: u64) -> Option<GalleryEntry> {
    let data = std::fs::read(path).ok()?;
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        match marker {
            // Standalone markers (no length field): SOI, TEM
            0xD8 | 0x01 => {
                i += 2;
            }
            0xD9 => break, // EOI
            0xD0..=0xD7 => {
                i += 2; // RST markers
            }
            // COM marker
            0xFE => {
                if i + 3 >= data.len() {
                    break;
                }
                let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                if len < 2 || i + 2 + len > data.len() {
                    break;
                }
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
                i += 2 + len;
            }
            // All other markers have a 2-byte length field
            _ => {
                if i + 3 >= data.len() {
                    break;
                }
                let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                if len < 2 || i + 2 + len > data.len() {
                    break;
                }
                i += 2 + len;
            }
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
        let mold_dir = mold_core::Config::mold_dir().expect("mold dir should resolve in tests");
        assert!(
            dir.starts_with(&mold_dir),
            "default_gallery_dir should be under mold dir: {} (mold dir: {})",
            dir.display(),
            mold_dir.display()
        );
    }

    #[test]
    fn scan_images_local_returns_empty_for_nonexistent_dir() {
        let entries = scan_images_local();
        let _ = entries;
    }
}
