//! `--id-image` file handling for `mold run`.
//!
//! The one place the CLI turns a user-supplied path into request bytes. It is
//! its own module because reading it is not a `std::fs::read`: an identity
//! photograph is a biometric input that goes on to drive two ONNX graphs and a
//! 24-block vision tower, so the file is opened under the same discipline
//! every other model-adjacent read in mold uses, and the request contract's
//! bounded-decode limits are applied to it BEFORE any request bytes exist.
//!
//! Nothing here re-implements policy. `mold_core::identity` owns the limits,
//! the defaults, and the qualified-model list; the server re-validates
//! everything on arrival. What this module owns is the *file*: which one was
//! opened, that it was a regular file reached without traversing a symlink,
//! that the bytes came from that exact descriptor, and that it was small
//! enough to be worth reading at all.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mold_core::identity::ID_IMAGE_LIMITS;

/// The identity flags exactly as clap parsed them.
///
/// `--id-image` is repeatable, so the path field is a `Vec` even in the
/// overwhelmingly common one-photograph case; [`IdentityOptions`] is where the
/// two wire shapes are decided.
#[derive(Debug, Clone, Default)]
pub struct IdentityArgs {
    pub id_images: Vec<PathBuf>,
    pub id_weight: Option<f64>,
    pub id_start_step: Option<u32>,
    pub true_cfg: Option<f64>,
    pub cfg_start_step: Option<u32>,
}

/// The same flags, resolved: bytes read, provenance derived.
///
/// One type for both the remote and the forced-local path, which is what makes
/// the two requests identical by construction rather than by review.
///
/// The singular and plural request fields are mutually exclusive by contract
/// (`mold_core::identity::IDENTITY_IMAGE_FORM_CONFLICT`), and this is the one
/// place the CLI chooses between them: exactly one `--id-image` produces the
/// singular form, so a one-photograph `mold run` puts the same bytes on the
/// same field it always has and an older server still understands it.
#[derive(Debug, Clone, Default)]
pub struct IdentityOptions {
    pub id_image: Option<Vec<u8>>,
    /// The file's own name, never the client path. Provenance the gallery and
    /// the embedded metadata record.
    pub id_image_name: Option<String>,
    pub id_images: Option<Vec<Vec<u8>>>,
    pub id_image_names: Option<Vec<String>>,
    pub id_weight: Option<f64>,
    pub id_start_step: Option<u32>,
    pub true_cfg: Option<f64>,
    pub cfg_start_step: Option<u32>,
}

impl IdentityArgs {
    /// Whether the user asked for identity conditioning in any way, including
    /// the incomplete forms. Mirrors
    /// `mold_core::identity::request_mentions_identity` on the pre-request
    /// side, where the fields are still a path rather than bytes.
    #[cfg(test)]
    pub fn mentions_identity(&self) -> bool {
        !self.id_images.is_empty() || self.id_weight.is_some() || self.id_start_step.is_some()
    }

    /// Read the reference photographs and produce the request fields.
    ///
    /// A knob without an image is refused here rather than at the server, so
    /// the user finds out before a 16 MB upload rather than after one — the
    /// wording is deliberately the same either way. The count and the whole-set
    /// budgets come from `mold_core::identity`, so a fifth `--id-image` is
    /// refused before the fifth file is even opened.
    pub fn resolve(self) -> Result<IdentityOptions> {
        if self.id_images.is_empty() {
            if self.id_weight.is_some()
                || self.id_start_step.is_some()
                || self.true_cfg.is_some()
                || self.cfg_start_step.is_some()
            {
                anyhow::bail!(
                    "--id-weight, --id-start-step, --true-cfg and --cfg-start-step require \
                     --id-image; there is no identity to condition on without a reference \
                     photograph"
                );
            }
            return Ok(IdentityOptions::default());
        }
        if self.id_images.len() > mold_core::identity::ID_IMAGES_MAX {
            anyhow::bail!(
                "--id-image was given {} times; at most {} reference photographs are accepted",
                self.id_images.len(),
                mold_core::identity::ID_IMAGES_MAX
            );
        }

        let mut bytes = Vec::with_capacity(self.id_images.len());
        let mut names = Vec::with_capacity(self.id_images.len());
        for path in &self.id_images {
            let (payload, name) = read_id_image(path)?;
            bytes.push(payload);
            names.push(name);
        }
        // The whole-set budgets, applied before anything is uploaded.
        let borrowed: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
        mold_core::identity::validate_id_images(&borrowed)
            .map_err(|reason| anyhow::anyhow!("invalid --id-image set: {reason}"))?;

        let mut options = IdentityOptions {
            id_weight: self.id_weight,
            id_start_step: self.id_start_step,
            true_cfg: self.true_cfg,
            cfg_start_step: self.cfg_start_step,
            ..IdentityOptions::default()
        };
        if bytes.len() == 1 {
            options.id_image = bytes.pop();
            options.id_image_name = names.pop();
        } else {
            options.id_images = Some(bytes);
            options.id_image_names = Some(names);
        }
        Ok(options)
    }
}

/// Open, bound, and read one identity image.
///
/// Four properties, in this order, and the order is the point:
///
/// 1. **Regular file reached without following a symlink.** Every parent
///    component is opened `O_NOFOLLOW | O_DIRECTORY` and the file itself
///    `O_NOFOLLOW`, then proven to be a regular file — so `--id-image` cannot
///    be pointed at a link into somewhere else, at a directory, or at a fifo
///    that would block forever.
/// 2. **Bounded before allocating.** The descriptor's own size is checked
///    against the contract's encoded-byte limit before a buffer is reserved,
///    so an enormous file is refused rather than read.
/// 3. **Read from the retained descriptor.** The bytes come from the handle
///    that was checked, never from a second open of the pathname — the name
///    can be replaced between the two, the descriptor cannot.
/// 4. **Validated against the contract's bounded-decode limits.** Magic bytes
///    and header-declared dimensions only; no decoder sees the payload here.
///
/// Returns the bytes and the file's own name, which becomes `id_image_name`.
/// The client's directory layout is never sent.
pub fn read_id_image(path: &Path) -> Result<(Vec<u8>, String)> {
    let mut file = mold_core::secure_file::open_regular_file_no_follow(path)
        .with_context(|| format!("failed to open --id-image '{}'", path.display()))?;

    let length = file
        .metadata()
        .with_context(|| format!("failed to stat --id-image '{}'", path.display()))?
        .len();
    if length == 0 {
        anyhow::bail!("--id-image '{}' is empty", path.display());
    }
    if length > ID_IMAGE_LIMITS.max_encoded_bytes as u64 {
        anyhow::bail!(
            "--id-image '{}' is {length} bytes, which exceeds the {} byte (16 MiB) limit",
            path.display(),
            ID_IMAGE_LIMITS.max_encoded_bytes
        );
    }

    let mut bytes = Vec::with_capacity(length as usize);
    file.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read --id-image '{}'", path.display()))?;

    mold_core::identity::validate_id_image_bytes(&bytes)
        .map_err(|reason| anyhow::anyhow!("invalid --id-image '{}': {reason}", path.display()))?;

    Ok((bytes, display_name(path)))
}

/// The file's own name, with any directory component discarded.
fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "id-image".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A genuine 1x1 RGBA PNG — the same fixture `mold_core::identity` pins
    /// its header parser against.
    fn png_1x1() -> Vec<u8> {
        vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ]
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn a_real_png_is_read_and_named_by_its_file_name_only() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("photos");
        std::fs::create_dir(&nested).unwrap();
        let path = write(&nested, "portrait.png", &png_1x1());

        let (bytes, name) = read_id_image(&path).expect("a real PNG reads");
        assert_eq!(bytes, png_1x1());
        assert_eq!(
            name, "portrait.png",
            "the client's directory layout is never sent"
        );
    }

    #[test]
    fn a_directory_is_not_an_image() {
        let dir = tempfile::tempdir().unwrap();
        let error = read_id_image(dir.path()).unwrap_err();
        assert!(format!("{error:#}").contains("--id-image"), "{error:#}");
    }

    #[test]
    fn a_missing_file_names_the_flag_and_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let error = read_id_image(&dir.path().join("absent.png")).unwrap_err();
        let rendered = format!("{error:#}");
        assert!(rendered.contains("--id-image"), "{rendered}");
        assert!(rendered.contains("absent.png"), "{rendered}");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_is_refused_rather_than_followed_into_somewhere_else() {
        let dir = tempfile::tempdir().unwrap();
        let real = write(dir.path(), "real.png", &png_1x1());
        let link = dir.path().join("link.png");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let error = read_id_image(&link).unwrap_err();
        assert!(format!("{error:#}").contains("link.png"), "{error:#}");
        // The target itself is still readable — only the link is refused.
        assert!(read_id_image(&real).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_parent_directory_is_refused_too() {
        let dir = tempfile::tempdir().unwrap();
        let real_dir = dir.path().join("real");
        std::fs::create_dir(&real_dir).unwrap();
        write(&real_dir, "portrait.png", &png_1x1());
        let linked_dir = dir.path().join("linked");
        std::os::unix::fs::symlink(&real_dir, &linked_dir).unwrap();

        assert!(read_id_image(&linked_dir.join("portrait.png")).is_err());
    }

    #[test]
    fn an_empty_file_is_refused_before_the_contract_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "empty.png", b"");
        let error = read_id_image(&path).unwrap_err();
        assert!(format!("{error:#}").contains("is empty"), "{error:#}");
    }

    /// The size bound is applied to the DESCRIPTOR, before a buffer is
    /// reserved — a 17 MiB file must never be read into memory to discover it
    /// was too big.
    #[test]
    fn an_oversized_file_is_refused_on_its_size_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = png_1x1();
        bytes.resize(ID_IMAGE_LIMITS.max_encoded_bytes + 1, 0);
        let path = write(dir.path(), "huge.png", &bytes);

        let error = read_id_image(&path).unwrap_err();
        let rendered = format!("{error:#}");
        assert!(rendered.contains("16 MiB"), "{rendered}");
        assert!(rendered.contains("huge.png"), "{rendered}");
    }

    #[test]
    fn a_non_image_is_refused_by_the_shared_contract_validator() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "notes.txt", b"this is not a photograph");
        let error = read_id_image(&path).unwrap_err();
        let rendered = format!("{error:#}");
        assert!(rendered.contains("PNG or JPEG"), "{rendered}");
        assert!(rendered.contains("notes.txt"), "{rendered}");
    }

    #[test]
    fn resolving_without_an_image_is_inert() {
        let resolved = IdentityArgs::default().resolve().expect("nothing to do");
        assert!(resolved.id_image.is_none());
        assert!(resolved.id_image_name.is_none());
        assert!(resolved.id_images.is_none());
        assert!(resolved.id_image_names.is_none());
        assert!(resolved.id_weight.is_none());
        assert!(resolved.id_start_step.is_none());
        assert!(resolved.true_cfg.is_none());
        assert!(resolved.cfg_start_step.is_none());
    }

    #[test]
    fn a_knob_without_an_image_is_refused_before_the_upload() {
        for args in [
            IdentityArgs {
                id_weight: Some(0.8),
                ..IdentityArgs::default()
            },
            IdentityArgs {
                id_start_step: Some(2),
                ..IdentityArgs::default()
            },
            IdentityArgs {
                true_cfg: Some(2.0),
                ..IdentityArgs::default()
            },
        ] {
            let error = args.resolve().unwrap_err();
            assert!(
                format!("{error:#}").contains("require --id-image"),
                "{error:#}"
            );
        }
    }

    #[test]
    fn resolving_carries_every_field_through() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "face.png", &png_1x1());
        let resolved = IdentityArgs {
            id_images: vec![path],
            id_weight: Some(0.85),
            id_start_step: Some(3),
            true_cfg: Some(2.0),
            cfg_start_step: Some(1),
        }
        .resolve()
        .expect("a real PNG resolves");

        assert_eq!(resolved.id_image.as_deref(), Some(png_1x1().as_slice()));
        assert_eq!(resolved.id_image_name.as_deref(), Some("face.png"));
        assert_eq!(resolved.id_weight, Some(0.85));
        assert_eq!(resolved.id_start_step, Some(3));
        assert_eq!(resolved.true_cfg, Some(2.0));
        assert_eq!(resolved.cfg_start_step, Some(1));
    }

    /// Exactly one `--id-image` must produce the SINGULAR wire field. The two
    /// shapes are mutually exclusive at admission, and a one-photograph run
    /// that silently switched shapes would stop working against a server that
    /// predates `id_images`.
    #[test]
    fn one_photograph_takes_the_singular_wire_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "face.png", &png_1x1());
        let resolved = IdentityArgs {
            id_images: vec![path],
            ..IdentityArgs::default()
        }
        .resolve()
        .expect("a real PNG resolves");
        assert!(resolved.id_image.is_some());
        assert_eq!(resolved.id_image_name.as_deref(), Some("face.png"));
        assert!(resolved.id_images.is_none());
        assert!(resolved.id_image_names.is_none());
    }

    /// Several photographs take the plural shape, in the order they were
    /// given, with one name per photograph — which is what admission requires.
    #[test]
    fn several_photographs_take_the_plural_wire_shape_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let one = write(dir.path(), "one.png", &png_1x1());
        let two = write(dir.path(), "two.png", &png_1x1());
        let resolved = IdentityArgs {
            id_images: vec![one, two],
            ..IdentityArgs::default()
        }
        .resolve()
        .expect("two real PNGs resolve");

        assert!(resolved.id_image.is_none());
        assert!(resolved.id_image_name.is_none());
        assert_eq!(resolved.id_images.as_ref().map(Vec::len), Some(2));
        assert_eq!(
            resolved.id_image_names.as_deref(),
            Some(["one.png".to_string(), "two.png".to_string()].as_slice())
        );
    }

    /// The count bound is the contract's, applied before the extra files are
    /// even opened.
    #[test]
    fn too_many_photographs_are_refused_before_the_upload() {
        let dir = tempfile::tempdir().unwrap();
        let paths: Vec<PathBuf> = (0..mold_core::identity::ID_IMAGES_MAX + 1)
            .map(|index| write(dir.path(), &format!("face{index}.png"), &png_1x1()))
            .collect();
        let error = IdentityArgs {
            id_images: paths,
            ..IdentityArgs::default()
        }
        .resolve()
        .unwrap_err();
        let rendered = format!("{error:#}");
        assert!(rendered.contains("at most"), "{rendered}");
        assert!(
            rendered.contains(&mold_core::identity::ID_IMAGES_MAX.to_string()),
            "{rendered}"
        );
    }

    #[test]
    fn mentions_identity_covers_the_incomplete_forms() {
        assert!(!IdentityArgs::default().mentions_identity());
        assert!(IdentityArgs {
            id_weight: Some(1.0),
            ..IdentityArgs::default()
        }
        .mentions_identity());
        assert!(IdentityArgs {
            id_start_step: Some(0),
            ..IdentityArgs::default()
        }
        .mentions_identity());
    }
}
