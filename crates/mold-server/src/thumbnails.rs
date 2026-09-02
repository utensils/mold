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

pub fn is_mesh_filename(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    lower.ends_with(".glb") || lower.ends_with(".obj")
}

/// Dispatch on the filename: video poster, or raster decode. Audio and mesh
/// have no pixels to read and are refused here, so callers reach for the
/// sidecar, [`ensure_mesh_poster`], or the placeholder.
///
/// Refusing is what keeps the raster decode from ever being handed a glTF
/// buffer or a RIFF header. `image::open` on either would fail anyway, but it
/// would fail AFTER reading the whole file, and the route's failure fallback
/// would then serve the source bytes to an `<img>` tag.
pub fn render_thumbnail(
    source: &Path,
    filename: &str,
    max_dim: u32,
    format: ThumbFormat,
) -> anyhow::Result<RenderedThumbnail> {
    if is_audio_filename(filename) {
        anyhow::bail!("audio prints have no raster thumbnail to render");
    }
    if is_mesh_filename(filename) {
        anyhow::bail!("mesh prints have no raster thumbnail to render");
    }
    if is_video_filename(filename) {
        render_video_thumbnail(source, max_dim, format)
    } else {
        render_raster_thumbnail(source, max_dim, format)
    }
}

/// The edge an on-demand mesh poster is drawn at.
///
/// Deliberately the figure the H3 engine already renders at save time
/// (`hunyuan3d::engine::POSTER_SIZE`), so a tile derived from a mirrored GLB
/// is indistinguishable from one written by the machine that generated it.
pub const MESH_POSTER_SIZE: u32 = 512;

/// Render the gallery poster for a stored `.glb`.
///
/// The poster is derivable from the file, not a by-product of generation:
/// the mesh IS the picture. Writing it only at save time left every mesh a
/// client had merely MIRRORED — imported over `PUT /api/gallery/import`, or
/// copied into the output directory — showing the generic wireframe cube
/// forever, because nothing downstream could ever produce one.
///
/// Bounded on both axes the rasterizer can blow up on: the requested edge is
/// clamped to the poster module's own `MAX_POSTER_SIZE` (its G-buffers cost
/// ~29 bytes per supersampled pixel), and `read_glb` refuses anything that is
/// not a triangle mesh before a single pixel is shaded.
pub fn render_mesh_poster(source: &Path) -> anyhow::Result<Vec<u8>> {
    render_mesh_poster_sized(source, MESH_POSTER_SIZE)
}

/// [`render_mesh_poster`] at an explicit edge, clamped to the rasterizer's
/// own ceiling rather than refused, because this is a cache tile and not a
/// user-facing export the way `/api/gallery/export` is.
pub fn render_mesh_poster_sized(source: &Path, size: u32) -> anyhow::Result<Vec<u8>> {
    use mold_inference::hunyuan3d::poster::MAX_POSTER_SIZE;
    let size = size.clamp(1, MAX_POSTER_SIZE);
    let bytes = std::fs::read(source)
        .map_err(|error| anyhow::anyhow!("cannot read the mesh {}: {error}", source.display()))?;
    let mesh = mold_inference::hunyuan3d::glb::read_glb(&bytes)
        .map_err(|error| anyhow::anyhow!("cannot read the mesh: {error}"))?;
    mold_inference::hunyuan3d::poster::render_poster(&mesh, size)
}

/// Write one tile atomically (temp + rename) so a concurrent reader never
/// sees a half-written PNG.
pub fn write_bytes_atomically(dest: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    if let Err(error) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error.into());
    }
    Ok(())
}

/// Write a mesh poster to BOTH sidecar names.
///
/// The server route reads `<file>.png` and the TUI reads `<file>.thumb.png`,
/// so a writer that guesses one leaves the other surface on the placeholder —
/// see `mold_core::media_paths::mesh_poster_thumbnail_paths`. A failure on
/// either name is reported; the first is kept so a caller can log one cause.
pub fn write_mesh_poster_sidecars(
    thumb_dir: &Path,
    filename: &str,
    png: &[u8],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(thumb_dir)?;
    let mut first_error: Option<anyhow::Error> = None;
    for dest in mold_core::media_paths::mesh_poster_thumbnail_paths(thumb_dir, filename) {
        if let Err(error) = write_bytes_atomically(&dest, png) {
            first_error.get_or_insert(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// The poster bytes for a stored mesh: the sidecar when one is already
/// there, otherwise a fresh render written back to both sidecar names.
///
/// The write is best effort. A read-only or full cache directory must still
/// yield a tile for this request — only a mesh that cannot be READ or RENDERED
/// is a placeholder.
pub fn ensure_mesh_poster(
    source: &Path,
    thumb_dir: &Path,
    filename: &str,
) -> anyhow::Result<Vec<u8>> {
    let sidecar = thumb_dir.join(format!("{filename}.png"));
    if let Ok(bytes) = std::fs::read(&sidecar) {
        if sniff_content_type(&bytes).is_some() {
            return Ok(bytes);
        }
    }
    let png = render_mesh_poster(source)?;
    if let Err(error) = write_mesh_poster_sidecars(thumb_dir, filename, &png) {
        tracing::warn!(
            file = %filename,
            error = %format!("{error:#}"),
            "mesh poster rendered but its sidecars could not be written"
        );
    }
    Ok(png)
}

/// One requested rendition of a tile. The default (256 px PNG) is the shape
/// every cache file written before `?size`/`?fmt` existed has, and it keeps
/// its historical path and ETag so older clients, the TUI, and the desktop's
/// shared-cache lookup stay byte-for-byte compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThumbnailVariant {
    pub max_dim: u32,
    pub format: ThumbFormat,
}

impl ThumbnailVariant {
    pub const DEFAULT: ThumbnailVariant = ThumbnailVariant {
        max_dim: DEFAULT_MAX_DIM,
        format: ThumbFormat::Png,
    };
    pub const SIZES: [u32; 2] = [256, 512];

    pub fn is_default(self) -> bool {
        self == Self::DEFAULT
    }

    /// Resolve the query; absent parameters mean the default rendition, and
    /// anything outside the two tiers / two formats is a validation error.
    pub fn from_query(size: Option<u32>, fmt: Option<&str>) -> Result<Self, String> {
        let max_dim = match size {
            None => DEFAULT_MAX_DIM,
            Some(px) if Self::SIZES.contains(&px) => px,
            Some(px) => return Err(format!("unsupported thumbnail size {px}; use 256 or 512")),
        };
        let format = match fmt.map(str::to_ascii_lowercase).as_deref() {
            None | Some("png") => ThumbFormat::Png,
            Some("jpeg") | Some("jpg") => ThumbFormat::Jpeg,
            Some(other) => {
                return Err(format!(
                    "unsupported thumbnail format {other}; use png or jpeg"
                ))
            }
        };
        Ok(Self { max_dim, format })
    }

    /// Every rendition a file may have on disk, for the sweeper.
    pub fn all() -> impl Iterator<Item = ThumbnailVariant> {
        Self::SIZES.into_iter().flat_map(|max_dim| {
            [ThumbFormat::Png, ThumbFormat::Jpeg]
                .into_iter()
                .map(move |format| ThumbnailVariant { max_dim, format })
        })
    }

    fn extension(self) -> &'static str {
        match self.format {
            ThumbFormat::Png => "png",
            ThumbFormat::Jpeg => "jpg",
        }
    }

    /// Cache path for this rendition: the historical `<hash>.png` for the
    /// default, `<hash>-<size>.<ext>` otherwise.
    pub fn cache_path(self, thumb_dir: &Path, filename: &str, media_version: &str) -> PathBuf {
        if self.is_default() {
            return versioned_thumbnail_path(thumb_dir, filename, media_version);
        }
        let key = Sha256::digest(format!("{filename}:{media_version}").as_bytes());
        thumb_dir.join(format!("{key:x}-{}.{}", self.max_dim, self.extension()))
    }

    /// `256-png` / `512-jpg`: the `x-mold-thumbnail-rendition` header value.
    pub fn rendition_label(self) -> String {
        format!("{}-{}", self.max_dim, self.extension())
    }

    /// ETag suffix: empty for the default so its tag is unchanged.
    pub fn etag_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("-{}-{}", self.max_dim, self.extension())
        }
    }
}

/// The formats a cache file may legitimately hold. A JPEG-requested tile of
/// a transparent source is stored as PNG under the `.jpg` name, so the
/// content type comes from the bytes, never the extension.
pub fn sniff_content_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else {
        None
    }
}

/// Whether a cache filename is one this module minted (`<sha256>.png` or
/// `<sha256>-<size>.<ext>`), as opposed to the TUI's `<name>.thumb.png`, an
/// audio waveform `<name>.png`, or anything else sharing the directory.
fn is_versioned_cache_name(name: &str) -> bool {
    let Some((stem, ext)) = name.rsplit_once('.') else {
        return false;
    };
    if !matches!(ext, "png" | "jpg") {
        return false;
    }
    let hash = stem.split('-').next().unwrap_or_default();
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Delete versioned tiles that no live or trashed print can address any
/// more. Trash purge cannot compute the versioned names after the file is
/// gone (they hash its mtime and size), so without this every purged or
/// re-rendered print left its tiles behind forever. Only files this module
/// minted are candidates, and only once they are older than `min_age`, so a
/// tile being written for a print that just landed is never swept.
pub fn sweep_orphans(
    output_dir: &Path,
    thumb_dir: &Path,
    min_age: std::time::Duration,
) -> std::io::Result<usize> {
    let mut expected = std::collections::HashSet::new();
    let trash_dir = mold_db::trash::trash_dir(output_dir);
    for dir in [output_dir, trash_dir.as_path()] {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if !metadata.is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let version = file_media_version(&metadata);
            for variant in ThumbnailVariant::all() {
                if let Some(file) = variant.cache_path(thumb_dir, &name, &version).file_name() {
                    expected.insert(file.to_string_lossy().into_owned());
                }
            }
        }
    }
    let now = std::time::SystemTime::now();
    let mut removed = 0;
    for entry in std::fs::read_dir(thumb_dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !is_versioned_cache_name(&name) || expected.contains(&name) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let old_enough = metadata
            .modified()
            .ok()
            .and_then(|m| now.duration_since(m).ok())
            .is_some_and(|age| age >= min_age);
        if old_enough && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

pub const AUDIO_PLACEHOLDER_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"><defs><linearGradient id="a" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1e293b"/><stop offset="1" stop-color="#0f172a"/></linearGradient></defs><rect width="256" height="256" fill="url(#a)"/><g fill="rgba(226,232,240,0.85)"><rect x="52" y="112" width="8" height="32" rx="4"/><rect x="72" y="92" width="8" height="72" rx="4"/><rect x="92" y="68" width="8" height="120" rx="4"/><rect x="112" y="100" width="8" height="56" rx="4"/><rect x="132" y="76" width="8" height="104" rx="4"/><rect x="152" y="104" width="8" height="48" rx="4"/><rect x="172" y="86" width="8" height="84" rx="4"/><rect x="192" y="116" width="8" height="24" rx="4"/></g></svg>"##;

pub const MESH_PLACEHOLDER_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"><defs><linearGradient id="m" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1e293b"/><stop offset="1" stop-color="#0f172a"/></linearGradient></defs><rect width="256" height="256" fill="url(#m)"/><g fill="none" stroke="rgba(226,232,240,0.85)" stroke-width="6" stroke-linejoin="round"><polygon points="128,58 196,98 196,168 128,208 60,168 60,98"/><polyline points="60,98 128,138 196,98"/><line x1="128" y1="138" x2="128" y2="208"/></g></svg>"##;

pub const VIDEO_PLACEHOLDER_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1e293b"/><stop offset="1" stop-color="#0f172a"/></linearGradient></defs><rect width="256" height="256" fill="url(#g)"/><circle cx="128" cy="128" r="52" fill="rgba(255,255,255,0.08)"/><polygon points="112,100 112,156 160,128" fill="rgba(226,232,240,0.85)"/></svg>"##;

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32, alpha: bool) -> image::DynamicImage {
        if alpha {
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
                image::Rgba([
                    (x % 256) as u8,
                    (y % 256) as u8,
                    128,
                    if x < w / 2 { 255 } else { 0 },
                ])
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

    /// A two-triangle quad, the same shape the gallery export tests use.
    /// Small enough to render in milliseconds, real enough to exercise the
    /// glTF reader and the rasterizer rather than a stub.
    fn glb_fixture() -> Vec<u8> {
        let mesh = mold_inference::hunyuan3d::mesh::Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            normals: Some(vec![[0.0, 0.0, 1.0]; 4]),
            uvs: None,
            vertex_colors: None,
        };
        mold_inference::hunyuan3d::glb::write_glb(
            &mesh,
            &mold_inference::hunyuan3d::glb::GlbMaterial::default(),
            None,
        )
        .expect("write the GLB fixture")
    }

    #[test]
    fn a_stored_glb_renders_its_own_poster() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("mold-hunyuan3d-1.glb");
        std::fs::write(&source, glb_fixture()).unwrap();
        let png = render_mesh_poster(&source).expect("render the poster");
        assert_eq!(sniff_content_type(&png), Some("image/png"));
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!(
            (decoded.width(), decoded.height()),
            (MESH_POSTER_SIZE, MESH_POSTER_SIZE)
        );
    }

    #[test]
    fn an_unreadable_mesh_is_an_error_not_a_blank_tile() {
        let dir = tempfile::tempdir().unwrap();
        let corrupt = dir.path().join("broken.glb");
        std::fs::write(&corrupt, b"not a glTF container at all").unwrap();
        assert!(render_mesh_poster(&corrupt).is_err());
        assert!(render_mesh_poster(&dir.path().join("absent.glb")).is_err());
    }

    /// The route reads `<file>.png` and the TUI reads `<file>.thumb.png`, so
    /// one render has to land under both names or one surface keeps the
    /// placeholder for a print that has a perfectly good tile.
    #[test]
    fn ensuring_a_poster_writes_both_sidecar_names() {
        let output = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let name = "mold-hunyuan3d-fp16-1788357387469.glb";
        let source = output.path().join(name);
        std::fs::write(&source, glb_fixture()).unwrap();

        let png = ensure_mesh_poster(&source, cache.path(), name).expect("render on demand");
        assert_eq!(sniff_content_type(&png), Some("image/png"));
        let sidecars = mold_core::media_paths::mesh_poster_thumbnail_paths(cache.path(), name);
        for sidecar in &sidecars {
            assert!(sidecar.is_file(), "{} was not written", sidecar.display());
            assert_eq!(std::fs::read(sidecar).unwrap(), png);
        }
        assert!(
            sidecars[0] != sidecars[1],
            "the two consumers must not share one path"
        );

        // A second call is served from the sidecar rather than re-rendered:
        // planting different bytes under the server's name is what comes back.
        std::fs::write(&sidecars[0], b"\x89PNG\r\n\x1a\nplanted").unwrap();
        let second = ensure_mesh_poster(&source, cache.path(), name).expect("sidecar hit");
        assert_eq!(second, b"\x89PNG\r\n\x1a\nplanted");
    }

    /// A corrupt sidecar is not a tile. Reading it back as one served an
    /// `<img>` tag whatever happened to be on disk, so it is re-rendered.
    #[test]
    fn a_corrupt_sidecar_is_re_rendered_rather_than_served() {
        let output = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let name = "quad.glb";
        let source = output.path().join(name);
        std::fs::write(&source, glb_fixture()).unwrap();
        let sidecar = cache.path().join(format!("{name}.png"));
        std::fs::write(&sidecar, b"truncated garbage").unwrap();
        let png = ensure_mesh_poster(&source, cache.path(), name).expect("re-render");
        assert_eq!(sniff_content_type(&png), Some("image/png"));
        assert_eq!(std::fs::read(&sidecar).unwrap(), png);
    }

    #[test]
    fn variant_query_defaults_and_rejects_unknown_values() {
        assert!(ThumbnailVariant::from_query(None, None)
            .unwrap()
            .is_default());
        assert!(ThumbnailVariant::from_query(Some(256), Some("png"))
            .unwrap()
            .is_default());
        let retina = ThumbnailVariant::from_query(Some(512), Some("jpeg")).unwrap();
        assert_eq!(retina.max_dim, 512);
        assert_eq!(retina.format, ThumbFormat::Jpeg);
        assert_eq!(
            ThumbnailVariant::from_query(Some(512), Some("JPG")).unwrap(),
            retina
        );
        assert!(ThumbnailVariant::from_query(Some(300), None).is_err());
        assert!(ThumbnailVariant::from_query(None, Some("webp")).is_err());
    }

    #[test]
    fn default_variant_keeps_the_historical_path_and_etag() {
        let dir = Path::new("/cache");
        let default = ThumbnailVariant::DEFAULT.cache_path(dir, "cat.png", "1-100");
        assert_eq!(default, versioned_thumbnail_path(dir, "cat.png", "1-100"));
        assert_eq!(ThumbnailVariant::DEFAULT.etag_suffix(), "");
        let retina = ThumbnailVariant {
            max_dim: 512,
            format: ThumbFormat::Jpeg,
        };
        let path = retina.cache_path(dir, "cat.png", "1-100");
        assert!(path.to_string_lossy().ends_with("-512.jpg"));
        assert_ne!(path, default);
        assert_eq!(retina.etag_suffix(), "-512-jpg");
        assert_eq!(retina.rendition_label(), "512-jpg");
        assert_eq!(ThumbnailVariant::DEFAULT.rendition_label(), "256-png");
        assert!(is_versioned_cache_name(
            path.file_name().unwrap().to_str().unwrap()
        ));
        assert!(is_versioned_cache_name(
            default.file_name().unwrap().to_str().unwrap()
        ));
        assert!(!is_versioned_cache_name("mold-flux-1.png.thumb.png"));
        assert!(!is_versioned_cache_name("clip.wav.png"));
    }

    #[test]
    fn sweep_removes_only_orphaned_versioned_tiles() {
        let output = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let live = output.path().join("live.png");
        gradient(32, 32, false).save(&live).unwrap();
        let version = file_media_version(&std::fs::metadata(&live).unwrap());
        let kept = ThumbnailVariant::DEFAULT.cache_path(cache.path(), "live.png", &version);
        let kept_retina = ThumbnailVariant {
            max_dim: 512,
            format: ThumbFormat::Jpeg,
        }
        .cache_path(cache.path(), "live.png", &version);
        let orphan = ThumbnailVariant::DEFAULT.cache_path(cache.path(), "gone.png", "1-1");
        let fresh_orphan = ThumbnailVariant::DEFAULT.cache_path(cache.path(), "new.png", "2-2");
        let foreign = cache.path().join("live.png.thumb.png");
        let waveform = cache.path().join("clip.wav.png");
        for path in [
            &kept,
            &kept_retina,
            &orphan,
            &fresh_orphan,
            &foreign,
            &waveform,
        ] {
            std::fs::write(path, b"x").unwrap();
        }
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(48 * 3600);
        for path in [&kept, &orphan, &foreign, &waveform] {
            filetime::set_file_mtime(path, filetime::FileTime::from_system_time(old)).unwrap();
        }
        let removed = sweep_orphans(
            output.path(),
            cache.path(),
            std::time::Duration::from_secs(24 * 3600),
        )
        .unwrap();
        assert_eq!(removed, 1);
        assert!(!orphan.exists(), "an orphaned versioned tile is swept");
        assert!(
            kept.exists() && kept_retina.exists(),
            "live tiles of every variant stay"
        );
        assert!(
            fresh_orphan.exists(),
            "a tile younger than the grace period stays"
        );
        assert!(
            foreign.exists() && waveform.exists(),
            "other layouts are never touched"
        );
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
