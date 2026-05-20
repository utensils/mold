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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<RecipeFileRole>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecipeFileRole {
    Vae,
    TextEncoder,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DownloadRecipe {
    pub files: Vec<RecipeFile>,
    pub needs_token: Option<TokenKind>,
}

/// Render a recipe `dest` template by substituting `{family}`, `{author}`,
/// and `{name}` with concrete values. Templates ship literal in the
/// catalog (`"{family}/civitai/{id}/{filename}"` for Civitai;
/// `"{family}/{author}/{name}/{path}"` for HF) so consumers don't have
/// to bake the per-row context into normaliser output. Without this
/// step, `mold pull cv:<id>` would deposit files under
/// `~/.mold/models/cv-<id>/{family}/civitai/...` — i.e. with a literal
/// `{family}` directory name — and `mold run cv:<id>` could never find
/// the safetensors.
///
/// `author` and `name` are only meaningful for HF entries; pass empty
/// strings for Civitai (the template never references them, so the
/// substitution is a no-op).
pub fn render_recipe_dest(template: &str, family: &str, author: &str, name: &str) -> String {
    template
        .replace("{family}", family)
        .replace("{author}", author)
        .replace("{name}", name)
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
    /// Trigger phrases (Civitai `trainedWords`) advertised by the entry.
    /// Empty for HF and for non-LoRA Civitai entries. Surface in the UI
    /// as click-to-insert chips next to the LoRA picker.
    ///
    /// `serde(default)` means existing on-disk shards (written before v8)
    /// keep deserializing without needing a migration pass on the JSON.
    #[serde(default)]
    pub trained_words: Vec<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_recipe_dest_substitutes_family_for_civitai() {
        let template = "{family}/civitai/1759168/juggernautXL_ragnarokBy.safetensors";
        let rendered = render_recipe_dest(template, "sdxl", "", "");
        assert_eq!(
            rendered,
            "sdxl/civitai/1759168/juggernautXL_ragnarokBy.safetensors"
        );
    }

    #[test]
    fn render_recipe_dest_substitutes_all_three_for_hf() {
        let template = "{family}/{author}/{name}/sd_xl_base_1.0.safetensors";
        let rendered = render_recipe_dest(template, "sdxl", "stabilityai", "stable-diffusion-xl");
        assert_eq!(
            rendered,
            "sdxl/stabilityai/stable-diffusion-xl/sd_xl_base_1.0.safetensors"
        );
    }

    #[test]
    fn render_recipe_dest_passes_through_when_no_placeholders() {
        let template = "static/path/to/file.safetensors";
        assert_eq!(
            render_recipe_dest(template, "sdxl", "", ""),
            "static/path/to/file.safetensors"
        );
    }
}
