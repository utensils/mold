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

use mold_core::RetainedSourceMediaAvailability;

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

/// Where a restored member lands: `<cache>/<print filename>/<display name>`,
/// so two prints conditioned on files of the same name never collide and
/// the file keeps the name the row shows. Both segments are reduced to a
/// plain file name first; the host's display name is already sanitized, and
/// this makes sure of it locally.
pub(crate) fn cache_path(print_filename: &str, display_name: &str) -> PathBuf {
    cache_dir()
        .join(file_name_only(print_filename))
        .join(file_name_only(display_name))
}

fn file_name_only(name: &str) -> String {
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
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        "member".to_string()
    } else {
        cleaned
    }
}

/// Ask `server_url` what it retained for `print_filename` and bring the
/// source image back into the cache.
pub(crate) async fn restore_source_image(
    server_url: &str,
    host_id: &str,
    print_filename: &str,
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
    let target = cache_path(print_filename, &member.display_name);
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

    #[test]
    fn every_unavailable_state_has_its_own_clause() {
        use RetainedSourceMediaAvailability::*;
        assert!(availability_wording(UnavailableLegacy).contains("not retained"));
        assert!(availability_wording(UnavailableMissingOrCorrupt).contains("missing or damaged"));
        assert!(availability_wording(UnavailableAuth).contains("API key"));
    }
}
