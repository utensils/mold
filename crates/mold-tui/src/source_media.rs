//! Restoring a print's retained source image on Reuse settings.
//!
//! A durable host keeps the conditioning media a print was rendered from
//! (CLAUDE.md "Durable gallery source media") and is the only authority on
//! what it retained, so a reuse ALWAYS asks — `OutputMetadata` records the
//! source image's name, never whether the host still has the bytes. An
//! `available` answer with a `source_image` member is downloaded into the
//! TUI's cache and becomes the Source row's file, so the bytes ride the next
//! request exactly as a hand-picked path would. Every unavailable state
//! keeps the row's "attach again" marker and is named on the timeline.

use std::path::{Path, PathBuf};

use mold_core::{RetainedSourceMediaAvailability, RetainedSourceMediaMember};

/// The member role the Source row restores. Other roles (masks, identity
/// photos, keyframes, audio) have no row to land on in the TUI.
const SOURCE_IMAGE_ROLE: &str = "source_image";

/// How a reuse's source-image restore settled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SourceRestore {
    /// The bytes are on disk at this path; the Source row can name it.
    Restored(PathBuf),
    /// The host answered, but not with the bytes. The word is the timeline's.
    Unavailable(&'static str),
    /// The host retained media for the print, none of it a source image.
    NoSourceMember,
    /// The request itself failed (network, a refused download).
    Failed(String),
}

/// Timeline wording for each unavailable state, one short clause that
/// names the host's reason and, where the user can act, the action.
pub(crate) fn availability_wording(availability: RetainedSourceMediaAvailability) -> &'static str {
    match availability {
        RetainedSourceMediaAvailability::Available => "available",
        RetainedSourceMediaAvailability::UnavailableLegacy => {
            "source not retained on this host (older print); attach it again"
        }
        RetainedSourceMediaAvailability::UnavailableMissingOrCorrupt => {
            "retained source is missing or damaged on this host; attach it again"
        }
        RetainedSourceMediaAvailability::UnavailableAuth => {
            "this host needs its API key before it releases private source media"
        }
    }
}

/// `<mold_dir>/cache/source-media/` — beside the thumbnail cache.
pub(crate) fn cache_dir() -> PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("source-media")
}

/// Where a restored member lands: `<cache>/<print filename>/<file name>`
/// (the file name from [`restored_file_name`]), so two prints conditioned
/// on files of the same name never collide and the file keeps the name the
/// row shows. Both segments are reduced to a plain file name first.
pub(crate) fn cache_path(print_filename: &str, display_name: &str) -> PathBuf {
    cache_dir()
        .join(file_name_only(print_filename))
        .join(file_name_only(display_name))
}

fn file_name_only(name: &str) -> String {
    safe_file_name(name).unwrap_or_else(|| "member".to_string())
}

/// Reduce `name` to one plain file-name segment: the last path component,
/// with separators and control characters replaced. `None` when nothing
/// usable is left (empty, `.`, `..`, whitespace only).
fn safe_file_name(name: &str) -> Option<String> {
    let base = Path::new(name)
        .file_name()
        .map(|part| part.to_string_lossy().to_string())
        .unwrap_or_default();
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_control() || c == '/' || c == '\\' {
                '_'
            } else {
                c
            }
        })
        .collect();
    if cleaned.trim().is_empty() || cleaned == "." || cleaned == ".." {
        None
    } else {
        Some(cleaned)
    }
}

/// The file name a restored source image is cached under and shown as on
/// the Source row.
///
/// The print's recorded name (`OutputMetadata.source_image_name`, the name
/// the reuse already shows as "attach again") comes first: it is the name
/// the user attached. The host's `display_name` is only a fallback because
/// it degrades to `<role>-<n>` (`source_image-1`) whenever the pin kept no
/// usable name, and the opaque member id is the last resort. A recorded
/// name without an extension borrows the member's, so the file still opens
/// by type. Every candidate is reduced to a safe file name, and one that
/// reduces to nothing falls through to the next.
pub(crate) fn restored_file_name(
    recorded_name: Option<&str>,
    member: &RetainedSourceMediaMember,
) -> String {
    let display = safe_file_name(&member.display_name);
    if let Some(recorded) = recorded_name.and_then(safe_file_name) {
        if Path::new(&recorded).extension().is_some() {
            return recorded;
        }
        return match display
            .as_deref()
            .and_then(|name| Path::new(name).extension())
            .and_then(|ext| ext.to_str())
        {
            Some(ext) => format!("{recorded}.{ext}"),
            None => recorded,
        };
    }
    display
        .or_else(|| safe_file_name(&member.member_id))
        .unwrap_or_else(|| "member".to_string())
}

/// Ask `server_url` what it retained for `print_filename` and bring the
/// source image back into the cache, named by [`restored_file_name`] from
/// the print's `recorded_name`.
pub(crate) async fn restore_source_image(
    server_url: &str,
    host_id: &str,
    print_filename: &str,
    recorded_name: Option<&str>,
) -> SourceRestore {
    let api_key = crate::hosts::api_key_for(host_id);
    let client = crate::hosts::client_for(server_url, api_key.as_deref());
    let inventory = match client.gallery_source_media(print_filename).await {
        Ok(inventory) => inventory,
        Err(error) => return SourceRestore::Failed(error.to_string()),
    };
    if inventory.availability != RetainedSourceMediaAvailability::Available {
        return SourceRestore::Unavailable(availability_wording(inventory.availability));
    }
    let Some(member) = inventory
        .members
        .iter()
        .find(|member| member.role == SOURCE_IMAGE_ROLE)
    else {
        return SourceRestore::NoSourceMember;
    };
    let bytes = match client
        .download_gallery_source_media_member(print_filename, &member.member_id)
        .await
    {
        Ok(bytes) => bytes,
        Err(error) => return SourceRestore::Failed(error.to_string()),
    };
    let target = cache_path(print_filename, &restored_file_name(recorded_name, member));
    let write = async {
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&target, &bytes).await
    };
    match write.await {
        Ok(()) => SourceRestore::Restored(target),
        Err(error) => SourceRestore::Failed(format!(
            "could not write {} to the cache: {error}",
            target.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_path_keeps_the_print_and_member_names_as_plain_file_names() {
        let path = cache_path("mold-chair.glb", "armchair.png");
        assert!(path.ends_with("source-media/mold-chair.glb/armchair.png"));
        // Traversal and separators in either name are neutralized.
        let hostile = cache_path("../../etc", "../passwd/../x.png");
        assert!(hostile.ends_with("source-media/etc/x.png"), "{hostile:?}");
        let empty = cache_path("", "..");
        assert!(empty.ends_with("source-media/member/member"), "{empty:?}");
    }

    fn member(id: &str, display_name: &str) -> RetainedSourceMediaMember {
        RetainedSourceMediaMember {
            member_id: id.into(),
            role: SOURCE_IMAGE_ROLE.into(),
            display_name: display_name.into(),
            size_bytes: 8,
        }
    }

    /// The print's recorded file name wins; the host's display name is the
    /// fallback (it is `<role>-<n>` whenever the pin kept no usable name),
    /// and the opaque id is the last resort.
    #[test]
    fn restored_file_name_prefers_the_recorded_name_then_display_then_id() {
        assert_eq!(
            restored_file_name(
                Some("armchair-cutout.png"),
                &member("src-1", "source_image-1")
            ),
            "armchair-cutout.png"
        );
        assert_eq!(
            restored_file_name(None, &member("src-1", "chair.webp")),
            "chair.webp"
        );
        assert_eq!(restored_file_name(None, &member("src-1", "")), "src-1");
        assert_eq!(
            restored_file_name(Some("  "), &member("src-1", "   ")),
            "src-1"
        );
        // A recorded name without an extension borrows the member's.
        assert_eq!(
            restored_file_name(Some("armchair"), &member("src-1", "source_image-1.png")),
            "armchair.png"
        );
        assert_eq!(
            restored_file_name(Some("armchair"), &member("src-1", "source_image-1")),
            "armchair"
        );
    }

    /// Whatever name wins, it is reduced to one plain file-name segment.
    #[test]
    fn restored_file_name_is_reduced_to_a_safe_file_name() {
        assert_eq!(
            restored_file_name(Some("../../etc/passwd\\x.png"), &member("src-1", "a.png")),
            "passwd_x.png"
        );
        // A candidate that reduces to nothing falls through to the next.
        assert_eq!(
            restored_file_name(Some(".."), &member("src-1", "chair.png")),
            "chair.png"
        );
        assert_eq!(
            restored_file_name(Some("evil\nname.png"), &member("src-1", "a.png")),
            "evil_name.png"
        );
        assert_eq!(restored_file_name(None, &member("../id", "../..")), "id");
        let path = cache_path(
            "chair.glb",
            &restored_file_name(
                Some("/tmp/armchair-cutout.png"),
                &member("src-1", "source_image-1"),
            ),
        );
        assert!(
            path.ends_with("source-media/chair.glb/armchair-cutout.png"),
            "{path:?}"
        );
    }

    #[test]
    fn every_unavailable_state_has_its_own_clause() {
        use RetainedSourceMediaAvailability::*;
        assert!(availability_wording(UnavailableLegacy).contains("not retained"));
        assert!(availability_wording(UnavailableMissingOrCorrupt).contains("missing or damaged"));
        assert!(availability_wording(UnavailableAuth).contains("API key"));
    }
}
