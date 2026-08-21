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
        Some("wav") => Some(OutputFormat::Wav),
        _ => None,
    }
}

/// Try to extract embedded metadata from a file. Returns `None` for files
/// without a `mold:parameters` chunk/marker/comment (or for formats we
/// don't embed metadata into, like webp/mp4).
pub fn read_embedded(path: &Path, format: OutputFormat) -> Option<OutputMetadata> {
    match format {
        OutputFormat::Png | OutputFormat::Apng => read_png_metadata(path),
        OutputFormat::Jpeg => read_jpeg_metadata(path),
        OutputFormat::Gif => read_gif_metadata(path),
        OutputFormat::Webp | OutputFormat::Mp4 | OutputFormat::Wav => None,
    }
}

/// Embedded metadata if present, else a best-effort synthetic row built
/// from the filename. Returns `(metadata, synthetic)`. The synthetic path
/// backfills real raster dimensions from the image header (non-MP4) so
/// gallery cards keep the correct aspect ratio.
///
/// This is the one metadata leaf every gallery consumer maps from —
/// server scan → `GalleryImage`, reconcile → `GenerationRecord`, TUI
/// fallback walk → its own entry type.
pub fn read_or_synthesize(
    path: &Path,
    format: OutputFormat,
    filename: &str,
    timestamp_secs: u64,
) -> (OutputMetadata, bool) {
    match read_embedded(path, format) {
        Some(m) => (m, false),
        None => {
            let mut meta = synthesize_from_filename(filename, timestamp_secs);
            if !matches!(format, OutputFormat::Mp4 | OutputFormat::Wav) {
                if let Some((w, h)) = image_header_dims(path) {
                    meta.width = w;
                    meta.height = h;
                }
            }
            (meta, true)
        }
    }
}

/// Build a best-effort `OutputMetadata` from a filename like
/// `mold-<model>-<unix>[-<idx>][~<slug>].<ext>`.
///
/// A trailing `~<slug>` (the lossy title slug appended by
/// `mold_core::default_output_filename_titled`) is stripped before the
/// model is recovered. The slug is never promoted back into `title` — it is
/// lowercase ASCII with punctuation collapsed, so the authoritative title
/// can only come from embedded metadata or the gallery row.
pub fn synthesize_from_filename(filename: &str, timestamp_secs: u64) -> OutputMetadata {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let stem = mold_core::strip_title_slug(stem);
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
        collection: None,
        tags: None,
        title: None,
        source_fit: None,
        guidance_overrides: None,
        sample_shift: None,
        distill_strength_high: None,
        distill_strength_low: None,
        job_id: None,
        prompt: String::new(),
        negative_prompt: None,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        output_mode: None,
        model,
        seed: 0,
        steps: 0,
        guidance: 0.0,
        width: 0,
        height: 0,
        generation_width: None,
        generation_height: None,
        strength: None,
        source_image_name: None,
        source_image_sha256: None,
        edit_image_sha256s: None,
        references: None,
        keyframes: None,
        scheduler: None,
        output_format: None,
        cfg_plus: None,
        lora: None,
        lora_scale: None,
        loras: None,
        control_model: None,
        control_scale: None,
        upscale_model: None,
        gif_preview: None,
        enable_audio: None,
        audio_file_path: None,
        source_video_path: None,
        extend_video_path: None,
        extend_overlap_frames: None,
        pipeline: None,
        pipeline_provenance_sha256: None,
        source_preprocessing: None,
        ic_lora_control: None,
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        frames: None,
        fps: None,
        chain_job_id: None,
        chain: None,
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
        // 44-byte canonical RIFF/WAVE header plus a token amount of PCM. A
        // header-only file decodes as a zero-length track everywhere.
        OutputFormat::Wav => 1024,
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

/// `RIFF....WAVE` sniff — the audio counterpart of [`has_ftyp_box`], so a
/// truncated or mislabelled `.wav` never reaches the gallery.
pub fn has_riff_wave_header(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 12];
    if f.read_exact(&mut buf).is_err() {
        return false;
    }
    &buf[0..4] == b"RIFF" && &buf[8..12] == b"WAVE"
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
        OutputFormat::Mp4 | OutputFormat::Wav => return false,
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
        OutputFormat::Wav => has_riff_wave_header(path),
        _ => image_header_dims(path).is_some(),
    };
    if !header_ok {
        return false;
    }
    if !matches!(format, OutputFormat::Mp4 | OutputFormat::Wav)
        && is_probably_solid_black(path, format, size_bytes)
    {
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

/// Read metadata from a GIF comment extension (`0x21 0xFE` + length-
/// prefixed sub-blocks) carrying `mold:parameters {json}`. Parser
/// upstreamed from the TUI's gallery scanner so every consumer can
/// recover GIF metadata.
fn read_gif_metadata(path: &Path) -> Option<OutputMetadata> {
    let data = std::fs::read(path).ok()?;
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0x21 && data[i + 1] == 0xFE {
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
                    if let Ok(meta) = serde_json::from_str::<OutputMetadata>(json) {
                        return Some(meta);
                    }
                }
            }
        }
        i += 1;
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

    /// Every writer (CLI and server) now stamps millisecond timestamps
    /// into default filenames — 13-digit groups must strip like 10-digit
    /// ones did.
    #[test]
    fn synthesize_handles_millisecond_timestamps() {
        let m = synthesize_from_filename("mold-flux-dev-q4-1700000000000.png", 1700000000);
        assert_eq!(m.model, "flux-dev-q4");
        let batch = synthesize_from_filename("mold-sdxl-1700000000000-2.png", 1700000000);
        assert_eq!(batch.model, "sdxl");
    }

    /// Titled prints append `~<slug>` to the stem; the slug must be stripped
    /// before the model is recovered and must never be promoted to a title.
    #[test]
    fn synthesize_strips_trailing_title_slug() {
        let m = synthesize_from_filename("mold-flux-dev-q4-1700000000000~smurf-04.png", 1700000000);
        assert_eq!(m.model, "flux-dev-q4");
        assert_eq!(m.title, None);
        let batch = synthesize_from_filename("mold-sdxl-1700000000000-2~river.png", 1700000000);
        assert_eq!(batch.model, "sdxl");
        assert_eq!(batch.title, None);
        // A slug that is itself all digits must not be mistaken for the
        // timestamp / batch index.
        let digits = synthesize_from_filename("mold-sdxl-1700000000000~2024.png", 1700000000);
        assert_eq!(digits.model, "sdxl");
        // Untitled filenames are unaffected.
        let plain = synthesize_from_filename("mold-flux-dev-q4-1700000000000.png", 1700000000);
        assert_eq!(plain.model, "flux-dev-q4");
    }

    /// The gallery-validity guard is about bytes, not names: a `~` in the
    /// stem must not exclude a titled print from reconcile.
    #[test]
    fn titled_filename_passes_gallery_validity_guard() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir
            .path()
            .join("mold-flux-dev-q4-1700000000000~smurf-04.png");
        let img = image::RgbImage::from_fn(64, 64, |x, y| {
            image::Rgb([(x * 4) as u8, (y * 4) as u8, ((x ^ y) * 4) as u8])
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        std::fs::write(&p, buf.into_inner()).unwrap();
        let size = std::fs::metadata(&p).unwrap().len();
        assert_eq!(format_from_path(&p), Some(OutputFormat::Png));
        assert!(is_valid_gallery_file(&p, OutputFormat::Png, size));
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

    fn gif_with_comment(comment: &str) -> Vec<u8> {
        let mut data = b"GIF89a".to_vec();
        // 1x1 logical screen descriptor, no global color table.
        data.extend_from_slice(&[1, 0, 1, 0, 0, 0, 0]);
        // Comment extension: 0x21 0xFE + length-prefixed sub-blocks + 0x00.
        data.extend_from_slice(&[0x21, 0xFE]);
        for chunk in comment.as_bytes().chunks(255) {
            data.push(chunk.len() as u8);
            data.extend_from_slice(chunk);
        }
        data.push(0);
        data.push(0x3B); // trailer
        data
    }

    fn minimal_metadata_json() -> String {
        r#"{"prompt":"a gif owl","model":"ltx-video","seed":7,"steps":30,"guidance":3.0,"width":64,"height":64,"version":"test"}"#.to_string()
    }

    /// GIF comment-extension metadata must round-trip through
    /// read_embedded — previously only the TUI could parse it, so
    /// reconcile and the server gallery synthesized rows for GIFs that
    /// actually carried full metadata.
    #[test]
    fn read_embedded_recovers_gif_comment_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("clip.gif");
        let comment = format!("mold:parameters {}", minimal_metadata_json());
        std::fs::write(&p, gif_with_comment(&comment)).unwrap();

        let meta = read_embedded(&p, OutputFormat::Gif).expect("gif metadata");
        assert_eq!(meta.prompt, "a gif owl");
        assert_eq!(meta.seed, 7);
    }

    #[test]
    fn read_embedded_returns_none_for_gif_without_comment() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("plain.gif");
        std::fs::write(&p, b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x3B").unwrap();
        assert!(read_embedded(&p, OutputFormat::Gif).is_none());
    }

    fn tiny_png(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(width, height, image::Rgb([200, 10, 10]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Synthetic path backfills real raster dims from the header so
    /// gallery cards keep the right aspect ratio.
    #[test]
    fn read_or_synthesize_backfills_dims_for_unmarked_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("mold-flux-dev-q4-1700000000000.png");
        std::fs::write(&p, tiny_png(64, 48)).unwrap();

        let (meta, synthetic) = read_or_synthesize(
            &p,
            OutputFormat::Png,
            "mold-flux-dev-q4-1700000000000.png",
            1_700_000_000,
        );
        assert!(synthetic);
        assert_eq!(meta.model, "flux-dev-q4");
        assert_eq!((meta.width, meta.height), (64, 48));
    }

    #[test]
    fn read_or_synthesize_prefers_embedded_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("clip.gif");
        let comment = format!("mold:parameters {}", minimal_metadata_json());
        std::fs::write(&p, gif_with_comment(&comment)).unwrap();

        let (meta, synthetic) = read_or_synthesize(&p, OutputFormat::Gif, "clip.gif", 0);
        assert!(!synthetic);
        assert_eq!(meta.prompt, "a gif owl");
    }

    #[test]
    fn read_or_synthesize_skips_dims_probe_for_mp4() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("mold-ltx2-1700000000000.mp4");
        std::fs::write(&p, b"not really an mp4").unwrap();

        let (meta, synthetic) =
            read_or_synthesize(&p, OutputFormat::Mp4, "mold-ltx2-1700000000000.mp4", 0);
        assert!(synthetic);
        assert_eq!((meta.width, meta.height), (0, 0));
    }

    // ── Gallery validity guards (ported from mold-server/routes.rs when
    // the duplicated guard implementations were consolidated here) ──────

    #[test]
    fn min_valid_size_thresholds_are_sensible() {
        assert!(min_valid_size(OutputFormat::Png) >= 128);
        assert!(min_valid_size(OutputFormat::Jpeg) >= 128);
        assert!(min_valid_size(OutputFormat::Apng) >= 128);
        assert!(min_valid_size(OutputFormat::Webp) >= 128);
        assert!(min_valid_size(OutputFormat::Gif) <= 512);
        assert!(min_valid_size(OutputFormat::Mp4) >= 1024);
    }

    #[test]
    fn has_ftyp_box_accepts_real_header_and_rejects_garbage() {
        let td = tempfile::tempdir().unwrap();

        let mut real = Vec::new();
        real.extend_from_slice(&[0x00, 0x00, 0x00, 0x20]);
        real.extend_from_slice(b"ftyp");
        real.extend_from_slice(b"isom\x00\x00\x02\x00isomiso2mp41");
        let real_path = td.path().join("real.mp4");
        std::fs::write(&real_path, &real).unwrap();
        assert!(has_ftyp_box(&real_path));

        let fake_path = td.path().join("fake.mp4");
        std::fs::write(&fake_path, b"this is not an mp4 file at all").unwrap();
        assert!(!has_ftyp_box(&fake_path));

        let trunc_path = td.path().join("truncated.mp4");
        std::fs::write(&trunc_path, b"\x00\x00\x00\x20").unwrap();
        assert!(!has_ftyp_box(&trunc_path));

        assert!(!has_ftyp_box(&td.path().join("nope.mp4")));
    }

    #[test]
    fn image_header_dims_returns_real_dimensions() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("valid.png");
        std::fs::write(&p, tiny_png(42, 24)).unwrap();
        assert_eq!(image_header_dims(&p), Some((42, 24)));

        let stub = td.path().join("stub.png");
        std::fs::write(&stub, b"\x89PNG\r\n\x1a\n").unwrap();
        assert!(image_header_dims(&stub).is_none());

        let text = td.path().join("text.png");
        std::fs::write(&text, b"hello world, not a png").unwrap();
        assert!(image_header_dims(&text).is_none());
    }

    /// A `.wav` that never reaches [`format_from_path`] is dropped before the
    /// gallery, the DB, or reconcile ever sees it — the whole audio artifact
    /// would silently vanish. Pin the admission chain end to end.
    #[test]
    fn wav_outputs_are_admitted_to_the_gallery() {
        assert_eq!(
            format_from_path(Path::new("mold-ltx2-1.wav")),
            Some(OutputFormat::Wav)
        );
        assert_eq!(
            format_from_path(Path::new("mold-ltx2-1.WAV")),
            Some(OutputFormat::Wav)
        );

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("take.wav");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&2048u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.resize(2048, 0);
        std::fs::write(&path, &bytes).unwrap();

        assert!(has_riff_wave_header(&path));
        assert!(is_valid_gallery_file(
            &path,
            OutputFormat::Wav,
            bytes.len() as u64
        ));
        // Audio never goes through the solid-black raster guard: there is no
        // raster, and `image::open` would just fail.
        assert!(!is_probably_solid_black(
            &path,
            OutputFormat::Wav,
            bytes.len() as u64
        ));
        // No embedded-metadata format for WAV — reconcile synthesizes instead.
        assert!(read_embedded(&path, OutputFormat::Wav).is_none());
    }

    #[test]
    fn truncated_or_mislabelled_wav_is_rejected() {
        let td = tempfile::tempdir().unwrap();

        let short = td.path().join("short.wav");
        std::fs::write(&short, b"RIFF\x00\x00\x00\x00WAVE").unwrap();
        assert!(
            !is_valid_gallery_file(&short, OutputFormat::Wav, 12),
            "a header-only wav decodes as a zero-length track"
        );

        let wrong = td.path().join("wrong.wav");
        std::fs::write(&wrong, vec![0u8; 4096]).unwrap();
        assert!(!has_riff_wave_header(&wrong));
        assert!(!is_valid_gallery_file(&wrong, OutputFormat::Wav, 4096));
    }

    #[test]
    fn probably_solid_black_ignores_large_files() {
        // Files above the per-format suspect size are trusted without a
        // full decode. Verify we bail out on size alone.
        let td = tempfile::tempdir().unwrap();
        let big_path = td.path().join("big.png");
        std::fs::write(&big_path, vec![0u8; 20 * 1024]).unwrap();
        assert!(!is_probably_solid_black(
            &big_path,
            OutputFormat::Png,
            20 * 1024,
        ));
    }
}
