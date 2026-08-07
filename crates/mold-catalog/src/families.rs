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
    ZImage,
    LtxVideo,
    Ltx2,
    Wan,
    MinimaxH3,
    QwenImage,
    Wuerstchen,
}

pub const ALL_FAMILIES: &[Family] = &[
    Family::Flux,
    Family::Flux2,
    Family::Sd15,
    Family::Sdxl,
    Family::ZImage,
    Family::LtxVideo,
    Family::Ltx2,
    Family::Wan,
    Family::MinimaxH3,
    Family::QwenImage,
    Family::Wuerstchen,
];

/// Families permitted in ordinary discovery surfaces for this build.
///
/// `ALL_FAMILIES` remains the exhaustive internal taxonomy so classification
/// and exhaustive matches know about contract-only families. Public catalog
/// endpoints must consume this filtered iterator instead: its predicate is a
/// direct view of mold-core's single activation authority.
pub fn active_families() -> impl Iterator<Item = Family> {
    ALL_FAMILIES.iter().copied().filter(|family| {
        mold_core::model_activation_available(family.as_str(), Some(family.as_str()))
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
            Family::ZImage => "z-image",
            Family::LtxVideo => "ltx-video",
            Family::Ltx2 => "ltx2",
            Family::Wan => "wan",
            Family::MinimaxH3 => "minimax-h3",
            Family::QwenImage => "qwen-image",
            Family::Wuerstchen => "wuerstchen",
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
            | Family::ZImage
            | Family::QwenImage
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
            "z-image" => Family::ZImage,
            "ltx-video" => Family::LtxVideo,
            "ltx2" => Family::Ltx2,
            "wan" => Family::Wan,
            "minimax-h3" | "minimax_h3" | "minimaxh3" => Family::MinimaxH3,
            "qwen-image" => Family::QwenImage,
            "wuerstchen" => Family::Wuerstchen,
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
    fn compliance_gated_families_stay_out_of_public_discovery() {
        let families = active_families().collect::<Vec<_>>();
        assert!(!families.contains(&Family::MinimaxH3));
        assert!(families.contains(&Family::Flux));
    }
}
