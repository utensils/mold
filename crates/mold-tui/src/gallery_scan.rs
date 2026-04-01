use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::app::GalleryEntry;

/// Scan for images in the current directory and optionally the output directory.
/// Returns entries sorted newest-first by modification time.
pub fn scan_images() -> Vec<GalleryEntry> {
    let mut entries = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Scan current working directory
    if let Ok(cwd) = std::env::current_dir() {
        scan_directory(&cwd, &mut entries, &mut seen);
    }

    // Scan MOLD_OUTPUT_DIR if configured
    let config = mold_core::Config::load_or_default();
    if let Some(output_dir) = config.resolved_output_dir() {
        if output_dir.is_dir() {
            scan_directory(&output_dir, &mut entries, &mut seen);
        }
    }

    // Sort by modification time, newest first
    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    entries
}

fn scan_directory(
    dir: &Path,
    entries: &mut Vec<GalleryEntry>,
    seen: &mut std::collections::HashSet<PathBuf>,
) {
    let walker = walkdir::WalkDir::new(dir).max_depth(1).into_iter();
    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path().to_path_buf();
        if !path.is_file() {
            continue;
        }

        // Only image files
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        if !matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
            continue;
        }

        // Deduplicate across directories
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if !seen.insert(canonical) {
            continue;
        }

        let timestamp = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Try to read mold metadata from PNG
        if ext.as_deref() == Some("png") {
            if let Some(gallery_entry) = read_png_metadata(&path, timestamp) {
                entries.push(gallery_entry);
                continue;
            }
        }

        // Fall back to parsing the filename for mold-generated images
        if let Some(gallery_entry) = parse_mold_filename(&path, timestamp) {
            entries.push(gallery_entry);
            continue;
        }

        // Generic image (no mold metadata) — still include for img2img selection
        let filename = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        entries.push(GalleryEntry {
            path,
            prompt_preview: filename,
            model: String::new(),
            generation_time_ms: None,
            seed: None,
            width: 0,
            height: 0,
            timestamp,
        });
    }
}

/// Try to read OutputMetadata from a PNG file's tEXt chunks.
fn read_png_metadata(path: &Path, timestamp: u64) -> Option<GalleryEntry> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let reader = decoder.read_info().ok()?;
    let info = reader.info();

    // Look for the mold:parameters JSON chunk
    for chunk in &info.uncompressed_latin1_text {
        if chunk.keyword == "mold:parameters" {
            if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(&chunk.text) {
                let prompt_preview = if meta.prompt.len() > 60 {
                    format!("{}...", &meta.prompt[..57])
                } else {
                    meta.prompt.clone()
                };
                return Some(GalleryEntry {
                    path: path.to_path_buf(),
                    prompt_preview,
                    model: meta.model,
                    generation_time_ms: None,
                    seed: Some(meta.seed),
                    width: meta.width,
                    height: meta.height,
                    timestamp,
                });
            }
        }
    }

    // Also check iTXt chunks (mold uses ITXt for unicode prompts)
    for chunk in &info.utf8_text {
        if chunk.keyword == "mold:parameters" {
            let text = match chunk.get_text() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(&text) {
                let prompt_preview = if meta.prompt.len() > 60 {
                    format!("{}...", &meta.prompt[..57])
                } else {
                    meta.prompt.clone()
                };
                return Some(GalleryEntry {
                    path: path.to_path_buf(),
                    prompt_preview,
                    model: meta.model,
                    generation_time_ms: None,
                    seed: Some(meta.seed),
                    width: meta.width,
                    height: meta.height,
                    timestamp,
                });
            }
        }
    }

    None
}

/// Try to extract model name and timestamp from the mold filename pattern.
/// Pattern: `mold-{model}-{timestamp}[-{index}].{ext}`
fn parse_mold_filename(path: &Path, timestamp: u64) -> Option<GalleryEntry> {
    let stem = path.file_stem()?.to_str()?;
    if !stem.starts_with("mold-") {
        return None;
    }

    let rest = &stem[5..]; // strip "mold-"

    // The timestamp is the last numeric segment before any batch index
    // e.g., "flux-dev-q8-1700000000" or "flux-dev-q8-1700000000-2"
    let parts: Vec<&str> = rest.rsplitn(2, '-').collect();
    let model = if parts.len() == 2 {
        // Check if the last part is all digits (timestamp or batch index)
        if parts[0].chars().all(|c| c.is_ascii_digit()) {
            // Check if remaining also ends with a timestamp
            let inner_parts: Vec<&str> = parts[1].rsplitn(2, '-').collect();
            if inner_parts.len() == 2
                && inner_parts[0].len() >= 10
                && inner_parts[0].chars().all(|c| c.is_ascii_digit())
            {
                // This was model-timestamp-index
                inner_parts[1].to_string()
            } else {
                // This was model-timestamp
                parts[1].to_string()
            }
        } else {
            rest.to_string()
        }
    } else {
        rest.to_string()
    };

    Some(GalleryEntry {
        path: path.to_path_buf(),
        prompt_preview: stem.to_string(),
        model,
        generation_time_ms: None,
        seed: None,
        width: 0,
        height: 0,
        timestamp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mold_filename_basic() {
        let path = PathBuf::from("/tmp/mold-flux-dev-q8-1700000000.png");
        let entry = parse_mold_filename(&path, 100).unwrap();
        assert_eq!(entry.model, "flux-dev-q8");
        assert_eq!(entry.prompt_preview, "mold-flux-dev-q8-1700000000");
    }

    #[test]
    fn parse_mold_filename_batch() {
        let path = PathBuf::from("/tmp/mold-flux-dev-q8-1700000000-2.png");
        let entry = parse_mold_filename(&path, 100).unwrap();
        assert_eq!(entry.model, "flux-dev-q8");
    }

    #[test]
    fn parse_non_mold_file_returns_none() {
        let path = PathBuf::from("/tmp/photo.png");
        assert!(parse_mold_filename(&path, 100).is_none());
    }
}
