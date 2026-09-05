//! mold's supported family taxonomy. The string forms are load-bearing
//! — they match `crates/mold-core/src/manifest.rs` `ModelManifest.family`
//! values and the `family` column of the new `catalog` SQLite table.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    Flux,
    Flux2,
    Sd15,
    Sdxl,
    Sd3,
    ZImage,
    LtxVideo,
    Ltx2,
    Wan,
    MinimaxH3,
    QwenImage,
    QwenImageEdit,
    Wuerstchen,
    Hunyuan3d,
}

pub const ALL_FAMILIES: &[Family] = &[
    Family::Flux,
    Family::Flux2,
    Family::Sd15,
    Family::Sdxl,
    Family::Sd3,
    Family::ZImage,
    Family::LtxVideo,
    Family::Ltx2,
    Family::Wan,
    Family::MinimaxH3,
    Family::QwenImage,
    Family::QwenImageEdit,
    Family::Wuerstchen,
    Family::Hunyuan3d,
];

/// Families permitted in ordinary acquisition surfaces for this build.
///
/// `ALL_FAMILIES` remains the exhaustive internal taxonomy so classification
/// and exhaustive matches know about contract-only families. Public catalog
/// endpoints must consume this filtered iterator instead: its predicate is a
/// direct view of mold-core's acquisition authority and does not imply runtime
/// availability.
pub fn active_families() -> impl Iterator<Item = Family> {
    ALL_FAMILIES.iter().copied().filter(|family| {
        mold_core::model_acquisition_available(family.as_str(), Some(family.as_str()))
    })
}

#[derive(Debug, thiserror::Error)]
#[error("unknown family: {0:?}")]
pub struct UnknownFamily(pub String);

impl Family {
    /// Stable string form used in the SQLite `family` column and in the
    /// existing manifest.rs `ModelManifest.family`. **Do not change** —
    /// this is a load-bearing identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Family::Flux => "flux",
            Family::Flux2 => "flux2",
            Family::Sd15 => "sd15",
            Family::Sdxl => "sdxl",
            Family::Sd3 => "sd3",
            Family::ZImage => "z-image",
            Family::LtxVideo => "ltx-video",
            Family::Ltx2 => "ltx2",
            Family::Wan => "wan",
            Family::MinimaxH3 => "minimax-h3",
            Family::QwenImage => "qwen-image",
            Family::QwenImageEdit => "qwen-image-edit",
            Family::Wuerstchen => "wuerstchen",
            Family::Hunyuan3d => "hunyuan3d",
        }
    }

    /// Whether this family generates video rather than images.
    ///
    /// The modality decision is made at four separate points — HF repo
    /// normalization, Civitai version normalization, the companion row
    /// builder, and the HF search-hit path — and every one of them was a
    /// `match` with an `_ => Modality::Image` catch-all. That shape means a
    /// newly added video family compiles cleanly while being classified as an
    /// image model at any site the author missed: the catalog row then renders
    /// with no duration controls and its requests are built with no frame
    /// count. One predicate is what keeps the four in step.
    pub fn is_video(&self) -> bool {
        match self {
            Family::LtxVideo | Family::Ltx2 | Family::Wan | Family::MinimaxH3 => true,
            Family::Flux
            | Family::Flux2
            | Family::Sd15
            | Family::Sdxl
            | Family::Sd3
            | Family::ZImage
            | Family::QwenImage
            | Family::QwenImageEdit
            | Family::Wuerstchen
            | Family::Hunyuan3d => false,
        }
    }

    /// Whether this family generates a 3-D mesh rather than raster frames.
    ///
    /// The third media class, and deliberately a SEPARATE predicate from
    /// [`Self::is_video`] rather than a widening of it: a mesh has no frames,
    /// no fps and no canvas, so every duration and resolution control must
    /// read `false` here exactly as it does for an image family — while every
    /// "this is an image" assumption downstream must read `true` here and
    /// stop. Exhaustive on purpose, for the same reason `is_video` is: a new
    /// family must state which side it falls on.
    pub fn is_mesh(&self) -> bool {
        match self {
            Family::Hunyuan3d => true,
            Family::Flux
            | Family::Flux2
            | Family::Sd15
            | Family::Sdxl
            | Family::Sd3
            | Family::ZImage
            | Family::LtxVideo
            | Family::Ltx2
            | Family::Wan
            | Family::MinimaxH3
            | Family::QwenImage
            | Family::QwenImageEdit
            | Family::Wuerstchen => false,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, UnknownFamily> {
        Ok(match s {
            "flux" => Family::Flux,
            "flux2" => Family::Flux2,
            "sd15" => Family::Sd15,
            "sdxl" => Family::Sdxl,
            "sd3" | "sd3.5" | "stable-diffusion-3" | "stable-diffusion-3.5" => Family::Sd3,
            "z-image" => Family::ZImage,
            "ltx-video" => Family::LtxVideo,
            "ltx2" => Family::Ltx2,
            "wan" => Family::Wan,
            "minimax-h3" | "minimax_h3" | "minimaxh3" => Family::MinimaxH3,
            "qwen-image" => Family::QwenImage,
            "qwen-image-edit" | "qwen_image_edit" => Family::QwenImageEdit,
            "wuerstchen" => Family::Wuerstchen,
            "hunyuan3d" | "hunyuan-3d" => Family::Hunyuan3d,
            other => return Err(UnknownFamily(other.to_string())),
        })
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_h3_aliases_are_canonical_video_taxonomy() {
        for alias in ["minimax-h3", "minimax_h3", "minimaxh3"] {
            let family = Family::from_str(alias).unwrap();
            assert_eq!(family, Family::MinimaxH3);
            assert_eq!(family.as_str(), "minimax-h3");
            assert!(family.is_video());
        }
    }

    #[test]
    fn authorized_h3_is_visible_for_acquisition() {
        let families = active_families().collect::<Vec<_>>();
        assert!(families.contains(&Family::MinimaxH3));
        assert!(families.contains(&Family::Flux));
    }

    #[test]
    fn sd3_aliases_use_the_manifest_family_slug() {
        for alias in ["sd3", "sd3.5", "stable-diffusion-3", "stable-diffusion-3.5"] {
            let family = Family::from_str(alias).unwrap();
            assert_eq!(family, Family::Sd3);
            assert_eq!(family.as_str(), "sd3");
            assert!(!family.is_video());
        }
        assert!(active_families().any(|family| family == Family::Sd3));
    }

    #[test]
    fn qwen_image_edit_is_a_distinct_raster_family() {
        for alias in ["qwen-image-edit", "qwen_image_edit"] {
            let family = Family::from_str(alias).unwrap();
            assert_eq!(family, Family::QwenImageEdit);
            assert_eq!(family.as_str(), "qwen-image-edit");
            assert!(!family.is_video());
            assert!(!family.is_mesh());
        }
        assert!(active_families().any(|family| family == Family::QwenImageEdit));
    }
}
