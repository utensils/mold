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

/// Minimum on-disk size below which a file is treated as a corrupt/aborted
/// output. Mirrors the thresholds in `crates/mold-server/src/routes.rs` so
/// reconcile and the legacy filesystem walk filter the same set of files.
pub fn min_valid_size(format: OutputFormat) -> u64 {
    match format {
        OutputFormat::Png | OutputFormat::Apng | OutputFormat::Jpeg | OutputFormat::Webp => 256,
        OutputFormat::Gif => 128,
        OutputFormat::Mp4 => 4096,
    }
}

/// Header-only "does this decode as an image?" probe. Returns `(width, height)`
/// on success — used both to validate and to fill aspect-ratio metadata for
/// synthetic backfill rows.
pub fn image_header_dims(path: &Path) -> Option<(u32, u32)> {
    image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// ISO-BMFF `ftyp` box check at offset 4 — same MP4 sniff used by the
/// server's gallery scan.
pub fn has_ftyp_box(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 12];
    if f.read_exact(&mut buf).is_err() {
        return false;
    }
    &buf[4..8] == b"ftyp"
}

/// Heuristic "this is a solid black image" detector — the same NaN/aborted
/// generation guard the server already uses. Only inspects raster files
/// below a per-format suspect-size ceiling so we never decode real outputs.
pub fn is_probably_solid_black(path: &Path, format: OutputFormat, size_bytes: u64) -> bool {
    const SAMPLE_DIM: u32 = 16;
    const CHANNEL_CEILING: u8 = 16;

    let suspect_threshold: u64 = match format {
        OutputFormat::Png | OutputFormat::Apng => 8 * 1024,
        OutputFormat::Jpeg => 4 * 1024,
        OutputFormat::Gif | OutputFormat::Webp => 4 * 1024,
        OutputFormat::Mp4 => return false,
    };
    if size_bytes > suspect_threshold {
        return false;
    }

    let Ok(img) = image::open(path) else {
        return false;
    };
    let thumb = img.thumbnail(SAMPLE_DIM, SAMPLE_DIM).to_rgb8();
    let mut max_channel: u8 = 0;
    for pixel in thumb.pixels() {
        let m = pixel.0[0].max(pixel.0[1]).max(pixel.0[2]);
        if m > max_channel {
            max_channel = m;
        }
        if max_channel > CHANNEL_CEILING {
            return false;
        }
    }
    max_channel <= CHANNEL_CEILING
}

/// Combined gallery-validity check used by reconcile to skip files the
/// existing server-side scanner would have hidden. Returns `false` for
/// truncated/aborted/solid-black outputs.
pub fn is_valid_gallery_file(path: &Path, format: OutputFormat, size_bytes: u64) -> bool {
    if size_bytes < min_valid_size(format) {
        return false;
    }
    let header_ok = match format {
        OutputFormat::Mp4 => has_ftyp_box(path),
        _ => image_header_dims(path).is_some(),
    };
    if !header_ok {
        return false;
    }
    if !matches!(format, OutputFormat::Mp4) && is_probably_solid_black(path, format, size_bytes) {
        return false;
    }
    true
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
