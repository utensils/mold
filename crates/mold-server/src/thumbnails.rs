//! Gallery thumbnail rendering and the server-side thumbnail cache layout.
//!
//! One implementation serves two consumers: `routes.rs`'s
//! `GET /api/gallery/thumbnail/:filename` (and its startup warmup), and the
//! desktop app's offline "This device" tiles, which render from the output
//! dir in-process while the embedded server is Off. Both must agree on the
//! cache path (`<MOLD_HOME>/cache/thumbnails/<sha256(filename:version)>.png`)
//! so a tile the server warmed is a free hit for the desktop and vice versa.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The historical tile edge — every existing cache file is this size.
pub const DEFAULT_MAX_DIM: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThumbFormat {
    Png,
    Jpeg,
}

impl ThumbFormat {
    pub fn content_type(self) -> &'static str {
        match self {
            ThumbFormat::Png => "image/png",
            ThumbFormat::Jpeg => "image/jpeg",
        }
    }
}

pub struct RenderedThumbnail {
    pub bytes: Vec<u8>,
    pub content_type: &'static str,
}

/// Server-side thumbnail cache directory.
pub fn server_thumbnail_dir() -> PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("thumbnails")
}

/// The cache identity of a source file: modification time in nanoseconds
/// plus length. Distinct from the wire `media_version` (`mtime_ms:size`),
/// which clients key their own caches on.
pub fn file_media_version(metadata: &std::fs::Metadata) -> String {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    format!("{modified}-{}", metadata.len())
}

/// Where the 256 px PNG tile for (`filename`, `media_version`) lives.
pub fn versioned_thumbnail_path(thumb_dir: &Path, filename: &str, media_version: &str) -> PathBuf {
    let key = Sha256::digest(format!("{filename}:{media_version}").as_bytes());
    thumb_dir.join(format!("{key:x}.png"))
}

/// Downscale to `max_dim` on the longer edge. Sources several times larger
/// than the target take the fast box filter; a near-size source keeps the
/// triangle filter so a 512 px tile from a 1024 px print stays crisp.
pub fn downscale(img: &image::DynamicImage, max_dim: u32) -> image::DynamicImage {
    let (w, h) = (img.width(), img.height());
    if w <= max_dim && h <= max_dim {
        return img.clone();
    }
    if w >= max_dim * 4 && h >= max_dim * 4 {
        return img.thumbnail(max_dim, max_dim);
    }
    img.resize(max_dim, max_dim, image::imageops::FilterType::Triangle)
}

/// Encode one thumbnail. JPEG flattens alpha onto black; a source with real
/// transparency is kept as PNG regardless of the requested format.
pub fn encode(img: &image::DynamicImage, format: ThumbFormat) -> anyhow::Result<RenderedThumbnail> {
    let mut buf = std::io::Cursor::new(Vec::new());
    match format {
        ThumbFormat::Png => {
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            Ok(RenderedThumbnail {
                bytes: buf.into_inner(),
                content_type: "image/png",
            })
        }
        ThumbFormat::Jpeg if img.color().has_alpha() => {
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            Ok(RenderedThumbnail {
                bytes: buf.into_inner(),
                content_type: "image/png",
            })
        }
        ThumbFormat::Jpeg => {
            let rgb = img.to_rgb8();
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85);
            encoder.encode_image(&rgb)?;
            Ok(RenderedThumbnail {
                bytes: buf.into_inner(),
                content_type: "image/jpeg",
            })
        }
    }
}

/// Decode a raster source with allocation limits, so a hostile file cannot
/// balloon the process.
pub fn open_raster(source: &Path) -> anyhow::Result<image::DynamicImage> {
    let mut reader = image::ImageReader::open(source)?.with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16_384);
    limits.max_image_height = Some(16_384);
    limits.max_alloc = Some(1024 * 1024 * 1024);
    reader.limits(limits);
    Ok(reader.decode()?)
}

/// A still image's thumbnail.
pub fn render_raster_thumbnail(
    source: &Path,
    max_dim: u32,
    format: ThumbFormat,
) -> anyhow::Result<RenderedThumbnail> {
    let img = open_raster(source)?;
    encode(&downscale(&img, max_dim), format)
}

/// A video's poster: the FIRST frame only, decoded through openh264. Older
/// code decoded the whole clip into memory to take frame 0.
pub fn render_video_thumbnail(
    source: &Path,
    max_dim: u32,
    format: ThumbFormat,
) -> anyhow::Result<RenderedThumbnail> {
    let frame = mold_inference::ltx2::media::extract_first_frame(source)?;
    let img = image::DynamicImage::ImageRgb8(frame);
    encode(&downscale(&img, max_dim), format)
}

pub fn is_video_filename(filename: &str) -> bool {
    filename.to_ascii_lowercase().ends_with(".mp4")
}

pub fn is_audio_filename(filename: &str) -> bool {
    filename.to_ascii_lowercase().ends_with(".wav")
}

/// Dispatch on the filename: video poster, or raster decode. Audio has no
/// pixels to read (its waveform tile is written at save time) and is refused
/// here so callers reach for the sidecar or the placeholder.
pub fn render_thumbnail(
    source: &Path,
    filename: &str,
    max_dim: u32,
    format: ThumbFormat,
) -> anyhow::Result<RenderedThumbnail> {
    if is_audio_filename(filename) {
        anyhow::bail!("audio prints have no raster thumbnail to render");
    }
    if is_video_filename(filename) {
        render_video_thumbnail(source, max_dim, format)
    } else {
        render_raster_thumbnail(source, max_dim, format)
    }
}

pub const AUDIO_PLACEHOLDER_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"><defs><linearGradient id="a" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1e293b"/><stop offset="1" stop-color="#0f172a"/></linearGradient></defs><rect width="256" height="256" fill="url(#a)"/><g fill="rgba(226,232,240,0.85)"><rect x="52" y="112" width="8" height="32" rx="4"/><rect x="72" y="92" width="8" height="72" rx="4"/><rect x="92" y="68" width="8" height="120" rx="4"/><rect x="112" y="100" width="8" height="56" rx="4"/><rect x="132" y="76" width="8" height="104" rx="4"/><rect x="152" y="104" width="8" height="48" rx="4"/><rect x="172" y="86" width="8" height="84" rx="4"/><rect x="192" y="116" width="8" height="24" rx="4"/></g></svg>"##;

pub const VIDEO_PLACEHOLDER_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1e293b"/><stop offset="1" stop-color="#0f172a"/></linearGradient></defs><rect width="256" height="256" fill="url(#g)"/><circle cx="128" cy="128" r="52" fill="rgba(255,255,255,0.08)"/><polygon points="112,100 112,156 160,128" fill="rgba(226,232,240,0.85)"/></svg>"##;

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32, alpha: bool) -> image::DynamicImage {
        if alpha {
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
                image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, if x < w / 2 { 255 } else { 0 }])
            }))
        } else {
            image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
            }))
        }
    }

    #[test]
    fn downscale_bounds_the_longer_edge_and_keeps_small_sources() {
        let big = downscale(&gradient(2048, 1024, false), 256);
        assert_eq!((big.width(), big.height()), (256, 128));
        let near = downscale(&gradient(600, 900, false), 512);
        assert_eq!((near.width(), near.height()), (341, 512));
        let small = downscale(&gradient(100, 50, false), 256);
        assert_eq!((small.width(), small.height()), (100, 50));
    }

    #[test]
    fn jpeg_is_requested_but_alpha_sources_stay_png() {
        let opaque = encode(&gradient(64, 64, false), ThumbFormat::Jpeg).unwrap();
        assert_eq!(opaque.content_type, "image/jpeg");
        assert!(opaque.bytes.starts_with(&[0xFF, 0xD8, 0xFF]));
        let alpha = encode(&gradient(64, 64, true), ThumbFormat::Jpeg).unwrap();
        assert_eq!(alpha.content_type, "image/png");
        assert!(alpha.bytes.starts_with(&[0x89, b'P', b'N', b'G']));
        let png = encode(&gradient(64, 64, false), ThumbFormat::Png).unwrap();
        assert_eq!(png.content_type, "image/png");
    }

    #[test]
    fn raster_render_round_trips_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("print.png");
        gradient(1024, 768, false).save(&source).unwrap();
        let rendered = render_thumbnail(&source, "print.png", 256, ThumbFormat::Png).unwrap();
        let decoded = image::load_from_memory(&rendered.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (256, 192));
        let retina = render_thumbnail(&source, "print.png", 512, ThumbFormat::Jpeg).unwrap();
        let decoded = image::load_from_memory(&retina.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (512, 384));
        assert_eq!(retina.content_type, "image/jpeg");
    }

    #[test]
    fn audio_is_refused_rather_than_decoded() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("clip.wav");
        std::fs::write(&source, b"RIFF....WAVE").unwrap();
        assert!(render_thumbnail(&source, "clip.wav", 256, ThumbFormat::Png).is_err());
    }

    #[test]
    fn versioned_path_is_stable_and_version_sensitive() {
        let dir = Path::new("/cache");
        let first = versioned_thumbnail_path(dir, "cat.png", "1-100");
        let second = versioned_thumbnail_path(dir, "cat.png", "2-100");
        assert_ne!(first, second);
        assert_eq!(first, versioned_thumbnail_path(dir, "cat.png", "1-100"));
        assert!(first.to_string_lossy().ends_with(".png"));
    }
}
