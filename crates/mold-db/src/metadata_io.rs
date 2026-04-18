//! Read embedded `mold:parameters` metadata from existing files on disk.
//!
//! Used by [`crate::reconcile`] to backfill rows for files that were
//! generated before the database existed (or while the DB was disabled).
//! Mirrors the parsers in `crates/mold-server/src/routes.rs` so backfilled
//! rows are indistinguishable from rows written at generation time when
//! the file already had embedded metadata.

use mold_core::{OutputFormat, OutputMetadata};
use std::path::Path;

/// Format inferred from a file extension. `None` for anything we don't
/// store in the gallery (everything outside the [`OutputFormat`] set).
pub fn format_from_path(path: &Path) -> Option<OutputFormat> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    match ext.as_deref() {
        Some("png") => Some(OutputFormat::Png),
        Some("jpg") | Some("jpeg") => Some(OutputFormat::Jpeg),
        Some("gif") => Some(OutputFormat::Gif),
        Some("apng") => Some(OutputFormat::Apng),
        Some("webp") => Some(OutputFormat::Webp),
        Some("mp4") => Some(OutputFormat::Mp4),
        _ => None,
    }
}

/// Try to extract embedded metadata from a file. Returns `None` for files
/// without a `mold:parameters` chunk/marker (or for formats we don't embed
/// metadata into, like mp4/gif).
pub fn read_embedded(path: &Path, format: OutputFormat) -> Option<OutputMetadata> {
    match format {
        OutputFormat::Png | OutputFormat::Apng => read_png_metadata(path),
        OutputFormat::Jpeg => read_jpeg_metadata(path),
        OutputFormat::Gif | OutputFormat::Webp | OutputFormat::Mp4 => None,
    }
}

/// Build a best-effort `OutputMetadata` from a filename like
/// `mold-<model>-<unix>[-<idx>].<ext>`.
pub fn synthesize_from_filename(filename: &str, timestamp_secs: u64) -> OutputMetadata {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let model = stem
        .strip_prefix("mold-")
        .and_then(|rest| {
            let mut parts: Vec<&str> = rest.split('-').collect();
            while parts
                .last()
                .map(|p| p.chars().all(|c| c.is_ascii_digit()))
                .unwrap_or(false)
                && parts.len() > 1
            {
                parts.pop();
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("-"))
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    OutputMetadata {
        prompt: String::new(),
        negative_prompt: None,
        original_prompt: None,
        model,
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
        version: format!("synthesized@{timestamp_secs}"),
    }
}

fn read_png_metadata(path: &Path) -> Option<OutputMetadata> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let reader = decoder.read_info().ok()?;
    let info = reader.info();

    for chunk in &info.uncompressed_latin1_text {
        if chunk.keyword == "mold:parameters" {
            if let Ok(meta) = serde_json::from_str::<OutputMetadata>(&chunk.text) {
                return Some(meta);
            }
        }
    }
    for chunk in &info.utf8_text {
        if chunk.keyword == "mold:parameters" {
            if let Ok(text) = chunk.get_text() {
                if let Ok(meta) = serde_json::from_str::<OutputMetadata>(&text) {
                    return Some(meta);
                }
            }
        }
    }
    None
}

fn read_jpeg_metadata(path: &Path) -> Option<OutputMetadata> {
    let data = std::fs::read(path).ok()?;
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        match marker {
            0xD8 | 0x01 => {
                i += 2;
            }
            0xD9 => break,
            0xD0..=0xD7 => {
                i += 2;
            }
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
                        if let Ok(meta) = serde_json::from_str::<OutputMetadata>(json) {
                            return Some(meta);
                        }
                    }
                }
                i += 2 + len;
            }
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
    fn format_inference_handles_known_extensions() {
        assert_eq!(
            format_from_path(Path::new("a.png")),
            Some(OutputFormat::Png)
        );
        assert_eq!(
            format_from_path(Path::new("b.JPG")),
            Some(OutputFormat::Jpeg)
        );
        assert_eq!(
            format_from_path(Path::new("c.mp4")),
            Some(OutputFormat::Mp4)
        );
        assert_eq!(
            format_from_path(Path::new("d.webp")),
            Some(OutputFormat::Webp)
        );
        assert_eq!(format_from_path(Path::new("e.txt")), None);
        assert_eq!(format_from_path(Path::new("noext")), None);
    }

    #[test]
    fn synthesize_recovers_model_name() {
        let m = synthesize_from_filename("mold-flux-dev-q4-1700000000.png", 1700000000);
        assert_eq!(m.model, "flux-dev-q4");
        assert!(m.prompt.is_empty());
        assert!(m.version.starts_with("synthesized@"));
    }

    #[test]
    fn synthesize_handles_batch_indexed_filename() {
        let m = synthesize_from_filename("mold-sdxl-1234567890-3.png", 1234567890);
        assert_eq!(m.model, "sdxl");
    }

    #[test]
    fn synthesize_falls_back_to_unknown_for_garbage() {
        let m = synthesize_from_filename("not-a-mold-file.png", 0);
        assert_eq!(m.model, "unknown");
    }

    #[test]
    fn read_embedded_returns_none_for_video_formats() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("v.mp4");
        std::fs::write(&p, b"fake").unwrap();
        assert!(read_embedded(&p, OutputFormat::Mp4).is_none());
    }
}
