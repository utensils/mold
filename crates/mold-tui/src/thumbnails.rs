use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::Result;

const THUMBNAIL_MAX_DIM: u32 = 256;

/// Returns the thumbnail cache directory: ~/.mold/cache/thumbnails/
pub fn thumbnail_dir() -> PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("thumbnails")
}

/// Compute the thumbnail path for a given source image path.
pub fn thumbnail_path(source: &Path) -> PathBuf {
    let canonical = source
        .canonicalize()
        .unwrap_or_else(|_| source.to_path_buf());
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    canonical.hash(&mut hasher);
    let hash = hasher.finish();
    thumbnail_dir().join(format!("{hash:016x}.png"))
}

/// Check if a thumbnail exists for the given source.
pub fn thumbnail_exists(source: &Path) -> bool {
    thumbnail_path(source).exists()
}

/// Generate a thumbnail for the given source image.
/// Resizes to fit within 256x256 maintaining aspect ratio, saves as PNG.
pub fn generate_thumbnail(source: &Path) -> Result<PathBuf> {
    let img = image::open(source)?;
    let thumb = img.thumbnail(THUMBNAIL_MAX_DIM, THUMBNAIL_MAX_DIM);

    let thumb_path = thumbnail_path(source);
    if let Some(parent) = thumb_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    thumb.save(&thumb_path)?;
    Ok(thumb_path)
}

/// Generate a thumbnail from raw image bytes.
/// The `key` is used to compute the cache path (typically the filename).
pub fn generate_thumbnail_from_bytes(data: &[u8], key: &Path) -> Result<PathBuf> {
    let img = image::load_from_memory(data)?;
    let thumb = img.thumbnail(THUMBNAIL_MAX_DIM, THUMBNAIL_MAX_DIM);

    let thumb_path = thumbnail_path(key);
    if let Some(parent) = thumb_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    thumb.save(&thumb_path)?;
    Ok(thumb_path)
}

/// Generate a thumbnail from a cached image file, keyed by a different path.
/// Used when the source was fetched from a server and cached locally.
pub fn generate_thumbnail_from_cached(source: &Path, key: &Path) -> Result<PathBuf> {
    let img = image::open(source)?;
    let thumb = img.thumbnail(THUMBNAIL_MAX_DIM, THUMBNAIL_MAX_DIM);

    let thumb_path = thumbnail_path(key);
    if let Some(parent) = thumb_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    thumb.save(&thumb_path)?;
    Ok(thumb_path)
}

/// Ensure thumbnails exist for all given image paths.
/// Generates missing thumbnails. Returns the number generated.
pub fn ensure_thumbnails(paths: &[PathBuf]) -> usize {
    let mut generated = 0;
    for path in paths {
        if !thumbnail_exists(path) && generate_thumbnail(path).is_ok() {
            generated += 1;
        }
    }
    generated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thumbnail_path_is_deterministic() {
        let p = Path::new("/tmp/test-image.png");
        let a = thumbnail_path(p);
        let b = thumbnail_path(p);
        assert_eq!(a, b);
    }

    #[test]
    fn thumbnail_path_differs_for_different_inputs() {
        let a = thumbnail_path(Path::new("/tmp/a.png"));
        let b = thumbnail_path(Path::new("/tmp/b.png"));
        assert_ne!(a, b);
    }

    #[test]
    fn thumbnail_dir_under_mold_dir() {
        let dir = thumbnail_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains("thumbnails"),
            "thumbnail_dir should contain 'thumbnails': {dir_str}"
        );
        assert!(
            dir_str.contains("cache"),
            "thumbnail_dir should contain 'cache': {dir_str}"
        );
    }

    #[test]
    fn thumbnail_path_has_png_extension() {
        let p = thumbnail_path(Path::new("/some/image.png"));
        assert_eq!(p.extension().and_then(|e| e.to_str()), Some("png"));
    }

    #[test]
    fn generate_thumbnail_creates_smaller_file() {
        // Create a test image in a temp directory
        let tmp = std::env::temp_dir().join("mold-thumb-test");
        std::fs::create_dir_all(&tmp).unwrap();
        let source = tmp.join("test-source.png");

        // Create a 512x512 red image
        let img = image::RgbImage::from_fn(512, 512, |_, _| image::Rgb([255, 0, 0]));
        img.save(&source).unwrap();

        let thumb_path = generate_thumbnail(&source).unwrap();
        assert!(thumb_path.exists());

        let source_size = std::fs::metadata(&source).unwrap().len();
        let thumb_size = std::fs::metadata(&thumb_path).unwrap().len();
        assert!(
            thumb_size < source_size,
            "thumbnail ({thumb_size}) should be smaller than source ({source_size})"
        );

        // Verify dimensions
        let thumb_img = image::open(&thumb_path).unwrap();
        assert!(thumb_img.width() <= THUMBNAIL_MAX_DIM);
        assert!(thumb_img.height() <= THUMBNAIL_MAX_DIM);

        // Cleanup
        std::fs::remove_file(&source).ok();
        std::fs::remove_file(&thumb_path).ok();
        std::fs::remove_dir_all(&tmp).ok();
    }
}
