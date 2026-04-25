//! mold's supported family taxonomy. The string forms are load-bearing
//! — they match `crates/mold-core/src/manifest.rs` `ModelManifest.family`
//! values and the `family` column of the new `catalog` SQLite table.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    Flux,
    Flux2,
    Sd15,
    Sdxl,
    ZImage,
    LtxVideo,
    Ltx2,
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
    Family::QwenImage,
    Family::Wuerstchen,
];

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
            Family::QwenImage => "qwen-image",
            Family::Wuerstchen => "wuerstchen",
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
