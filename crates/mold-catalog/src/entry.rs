//! Core catalog entry types. These serialize as the on-disk shard format
//! AND as the wire format for `/api/catalog/*`. Changing field names or
//! kebab-case forms is a wire-protocol break.

use serde::{Deserialize, Serialize};

use crate::families::Family;

pub type CompanionRef = String;

/// `"hf:author/repo"` for HF entries, `"cv:<modelVersionId>"` for Civitai.
/// Stored as a `String`-newtype for type safety at API boundaries; the
/// inner `String` is what hits SQLite as the `id` column primary key.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CatalogId(pub String);

impl<S: Into<String>> From<S> for CatalogId {
    fn from(s: S) -> Self {
        Self(s.into())
    }
}

impl CatalogId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    Hf,
    Civitai,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FamilyRole {
    Foundation,
    Finetune,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Modality {
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    Checkpoint,
    Lora,
    Vae,
    TextEncoder,
    ControlNet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FileFormat {
    Safetensors,
    Gguf,
    Diffusers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Bundling {
    Separated,
    SingleFile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TokenKind {
    Hf,
    Civitai,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LicenseFlags {
    pub commercial: Option<bool>,
    pub derivatives: Option<bool>,
    pub different_license: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeFile {
    pub url: String,
    pub dest: String,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DownloadRecipe {
    pub files: Vec<RecipeFile>,
    pub needs_token: Option<TokenKind>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: CatalogId,
    pub source: Source,
    pub source_id: String,
    pub name: String,
    pub author: Option<String>,
    pub family: Family,
    pub family_role: FamilyRole,
    pub sub_family: Option<String>,
    pub modality: Modality,
    pub kind: Kind,
    pub file_format: FileFormat,
    pub bundling: Bundling,
    pub size_bytes: Option<u64>,
    pub download_count: u64,
    pub rating: Option<f32>,
    pub likes: u64,
    pub nsfw: bool,
    pub thumbnail_url: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub license_flags: LicenseFlags,
    pub tags: Vec<String>,
    pub companions: Vec<CompanionRef>,
    pub download_recipe: DownloadRecipe,
    pub engine_phase: u8,
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
    pub added_at: i64,
}

/// On-disk shard format. One file per family in `data/catalog/`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shard {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub family: String,
    pub generated_at: String,
    pub scanner_version: String,
    pub entries: Vec<CatalogEntry>,
}

pub const SHARD_SCHEMA: &str = "mold.catalog.v1";
