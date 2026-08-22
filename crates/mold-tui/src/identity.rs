//! Face-identity (PuLID) authoring for the terminal surface.
//!
//! The Create form's Advanced ▸ Identity section collects a local photo path
//! plus the two knobs `mold_core::identity` owns. Everything semantic —
//! the accepted containers, the encoded-size ceiling, the header-declared
//! pixel bounds, the `id_weight` range, the `id_start_step` rule, and the
//! model-gate refusal — comes from that module; this file only opens the
//! file safely and turns a failure into a line the row can render.
//!
//! The path is validated at *entry* time rather than at dispatch: an
//! unreadable, symlinked, non-image, or oversized file is refused while the
//! picker is still open, so the form never carries a photo that would earn a
//! late rejection after the queue slot is paid for.

use std::path::Path;

use mold_core::identity;

/// The photo is read once at entry to bounds-check it and once again at
/// dispatch. Both reads stop at the encoded ceiling `mold_core` publishes, so
/// a huge file is never fully buffered just to be refused.
fn read_bounded(path: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;

    let file = mold_core::secure_file::open_regular_file_no_follow(path)
        .map_err(|error| format!("Identity photo could not be opened: {error}"))?;
    let limit = identity::ID_IMAGE_LIMITS.max_encoded_bytes;
    let mut bytes = Vec::new();
    // One byte past the limit is enough to prove the file is over it.
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Identity photo could not be read: {error}"))?;
    if bytes.len() > limit {
        return Err(format!(
            "Identity photo exceeds the {} byte (16 MiB) limit",
            limit
        ));
    }
    Ok(bytes)
}

/// Open, bounds-check, and return the identity photo at `path`.
///
/// Refuses anything `identity::validate_id_image_bytes` refuses, and — before
/// that — anything that is not a regular file reached without traversing a
/// symlink. The bytes are returned so the entry path and the dispatch path
/// share one implementation.
pub(crate) fn load_identity_image(path: &str) -> Result<Vec<u8>, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("Identity photo needs a file path".to_string());
    }
    let bytes = read_bounded(Path::new(trimmed))?;
    identity::validate_id_image_bytes(&bytes)?;
    Ok(bytes)
}

/// Provenance label shipped as `GenerateRequest.id_image_name` — the picked
/// file's basename, never the full local path.
pub(crate) fn identity_image_name(path: &str) -> Option<String> {
    Path::new(path.trim())
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .filter(|name| !name.is_empty())
}

/// One-line identity provenance for the Library panels, built from the saved
/// metadata rather than from a request. `None` when the print carried no
/// identity reference at all.
///
/// The digest is shortened to twelve hex characters: enough to match a photo
/// against a local stash, short enough to sit in the narrow Details panel.
pub(crate) fn metadata_summary(metadata: &mold_core::OutputMetadata) -> Option<String> {
    provenance_summary(
        metadata.id_image_name.as_deref(),
        metadata.id_image_sha256.as_deref(),
        metadata.id_weight,
        metadata.id_start_step,
    )
}

/// The field-level half of [`metadata_summary`], so the wording is testable
/// without materializing a whole `OutputMetadata`.
///
/// Any one present field proves the print carried identity conditioning —
/// the server records all four together, but an older or partial record must
/// still render rather than silently drop the provenance. Absent knobs fall
/// back to `mold_core::identity`'s published defaults, which is what actually
/// rendered.
pub(crate) fn provenance_summary(
    name: Option<&str>,
    sha256: Option<&str>,
    weight: Option<f64>,
    start_step: Option<u32>,
) -> Option<String> {
    let name = name.map(str::trim).filter(|value| !value.is_empty());
    let sha256 = sha256.map(str::trim).filter(|value| !value.is_empty());
    if name.is_none() && sha256.is_none() && weight.is_none() && start_step.is_none() {
        return None;
    }
    let mut parts = Vec::new();
    if let Some(name) = name {
        parts.push(name.to_string());
    }
    if let Some(sha256) = sha256 {
        parts.push(sha256.chars().take(12).collect::<String>());
    }
    parts.push(format!(
        "str {:.2}",
        weight.unwrap_or(identity::ID_WEIGHT_DEFAULT)
    ));
    parts.push(format!(
        "from step {}",
        start_step.unwrap_or(identity::ID_START_STEP_DEFAULT)
    ));
    Some(parts.join(" \u{00b7} "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn png_1x1() -> Vec<u8> {
        vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ]
    }

    fn write_file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
        let path = dir.path().join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn a_real_png_loads_and_names_itself_by_basename() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "face.png", &png_1x1());
        assert_eq!(load_identity_image(&path).unwrap(), png_1x1());
        assert_eq!(identity_image_name(&path).as_deref(), Some("face.png"));
    }

    #[test]
    fn a_non_image_is_refused_with_the_core_wording() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "notes.txt", b"not an image");
        let error = load_identity_image(&path).unwrap_err();
        assert_eq!(
            error,
            identity::validate_id_image_bytes(b"not an image").unwrap_err(),
            "the row must show mold-core's own refusal, never a restatement"
        );
    }

    #[test]
    fn an_empty_path_and_a_missing_file_are_both_refused() {
        assert!(load_identity_image("   ").is_err());
        assert!(load_identity_image("/nonexistent/face.png").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_photo_is_refused_before_it_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let real = write_file(&dir, "face.png", &png_1x1());
        let link = dir.path().join("link.png");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let error = load_identity_image(&link.to_string_lossy()).unwrap_err();
        assert!(
            error.starts_with("Identity photo could not be opened"),
            "symlinks must never be followed: {error}"
        );
    }

    #[test]
    fn a_directory_is_not_a_photo() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_identity_image(&dir.path().to_string_lossy()).is_err());
    }

    #[test]
    fn an_oversized_file_is_refused_on_its_size_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = png_1x1();
        bytes.resize(identity::ID_IMAGE_LIMITS.max_encoded_bytes + 1, 0);
        let path = write_file(&dir, "huge.png", &bytes);
        let error = load_identity_image(&path).unwrap_err();
        assert!(error.contains("16 MiB"), "{error}");
    }

    #[test]
    fn provenance_names_the_photo_digest_weight_and_start_step() {
        let summary = provenance_summary(
            Some("face.png"),
            Some("0123456789abcdef0123456789abcdef"),
            Some(1.25),
            Some(3),
        )
        .unwrap();
        assert_eq!(
            summary,
            "face.png \u{00b7} 0123456789ab \u{00b7} str 1.25 \u{00b7} from step 3"
        );
    }

    #[test]
    fn provenance_is_absent_for_a_print_with_no_identity() {
        assert_eq!(provenance_summary(None, None, None, None), None);
        assert_eq!(provenance_summary(Some("  "), Some(""), None, None), None);
    }

    #[test]
    fn provenance_falls_back_to_the_core_defaults() {
        let summary = provenance_summary(Some("face.png"), None, None, None).unwrap();
        assert_eq!(
            summary,
            format!(
                "face.png \u{00b7} str {:.2} \u{00b7} from step {}",
                identity::ID_WEIGHT_DEFAULT,
                identity::ID_START_STEP_DEFAULT
            )
        );
    }
}
