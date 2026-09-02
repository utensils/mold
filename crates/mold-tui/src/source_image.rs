//! Source-image entry for the Create form's Source row.
//!
//! The row is the only way a terminal user can hand a conditioning image to
//! an img2img, image-to-video, or image-to-3-D recipe, so the typed path is
//! checked at *entry* time — `~` expanded, the file confirmed to exist and be
//! a regular file, the container confirmed to be one the server decodes — and
//! a failure stays in the picker rather than arriving as a late server error
//! after the queue slot is paid for. The bytes themselves are read once, at
//! dispatch, by `backend::build_request`.

use std::path::{Path, PathBuf};

/// Longest path the picker accepts; matches the identity picker's ceiling.
pub(crate) const SOURCE_PATH_MAX_BYTES: usize = 4096;

/// Containers the server's source-image decoder accepts.
const ACCEPTED_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

/// Expand a leading `~` or `~/` to the home directory. Any other path is
/// returned unchanged; without a resolvable home the tilde is left alone
/// and the existence check names it.
pub(crate) fn expand_home(input: &str) -> PathBuf {
    if input == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = input.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(input)
}

/// Check the typed path and return the expanded form the request will read.
///
/// Errors are the row's inline text: the missing file, the non-file, or the
/// container, each named so the fix is obvious.
pub(crate) fn validate_source_image_path(input: &str) -> Result<String, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Source image needs a file path".to_string());
    }
    let path = expand_home(trimmed);
    let metadata = std::fs::metadata(&path)
        .map_err(|_| format!("Source image not found: {}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("Source image is not a file: {}", path.display()));
    }
    if !has_accepted_extension(&path) {
        return Err(
            "Source image must be a PNG, JPEG, or WebP file (.png, .jpg, .jpeg, .webp)".to_string(),
        );
    }
    Ok(path.to_string_lossy().into_owned())
}

fn has_accepted_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .is_some_and(|ext| ACCEPTED_EXTENSIONS.contains(&ext.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilde_expands_to_home_and_other_paths_pass_through() {
        let home = dirs::home_dir().expect("a home directory in the test environment");
        assert_eq!(expand_home("~"), home);
        assert_eq!(expand_home("~/pics/cat.png"), home.join("pics/cat.png"));
        assert_eq!(expand_home("/tmp/cat.png"), PathBuf::from("/tmp/cat.png"));
        assert_eq!(expand_home("~cat.png"), PathBuf::from("~cat.png"));
    }

    #[test]
    fn validation_names_the_missing_file_the_directory_and_the_container() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            validate_source_image_path("   "),
            Err("Source image needs a file path".into())
        );
        let missing = dir.path().join("nope.png");
        let error = validate_source_image_path(&missing.to_string_lossy()).unwrap_err();
        assert!(error.starts_with("Source image not found"), "{error}");
        let error = validate_source_image_path(&dir.path().to_string_lossy()).unwrap_err();
        assert!(error.starts_with("Source image is not a file"), "{error}");
        let text = dir.path().join("notes.txt");
        std::fs::write(&text, b"x").unwrap();
        let error = validate_source_image_path(&text.to_string_lossy()).unwrap_err();
        assert!(error.contains("PNG, JPEG, or WebP"), "{error}");
    }

    #[test]
    fn an_existing_raster_is_accepted_with_its_expanded_path() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["cat.png", "cat.JPG", "cat.jpeg", "cat.webp"] {
            let file = dir.path().join(name);
            std::fs::write(&file, b"bytes").unwrap();
            let padded = format!("  {}  ", file.to_string_lossy());
            assert_eq!(
                validate_source_image_path(&padded),
                Ok(file.to_string_lossy().into_owned()),
                "{name}"
            );
        }
    }
}
