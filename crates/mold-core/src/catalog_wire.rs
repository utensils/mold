//! Wire types for `GET /api/catalog/installed`.
//!
//! One struct serves both sides: `mold-server` serializes it and the
//! client in this crate deserializes it, so the payload shape can no
//! longer drift silently between the two. Every field is always
//! serialized (no `skip_serializing_if`) — the SPA expects explicit
//! `null`s so a single TypeScript interface covers this endpoint and
//! `/api/catalog/search`. Deserialization stays tolerant of older
//! servers via `#[serde(default)]` on everything the client can
//! reasonably backfill.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstalledCatalogResponse {
    #[serde(default)]
    pub entries: Vec<InstalledCatalogEntry>,
    #[serde(default)]
    pub page: i64,
    #[serde(default)]
    pub page_size: i64,
    #[serde(default)]
    pub total: i64,
}

/// One installed catalog entry, shaped to stay wire-uniform with the
/// entries `/api/catalog/search` returns. Fields the sidecar doesn't
/// carry (ratings, tags, timestamps…) serialize as empty/`null`
/// placeholders on purpose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledCatalogEntry {
    pub id: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub source_id: String,
    pub name: String,
    #[serde(default)]
    pub author: Option<String>,
    pub family: String,
    #[serde(default)]
    pub family_role: String,
    #[serde(default)]
    pub sub_family: Option<String>,
    #[serde(default)]
    pub modality: String,
    pub kind: String,
    #[serde(default)]
    pub file_format: String,
    #[serde(default)]
    pub bundling: String,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub download_count: u64,
    #[serde(default)]
    pub rating: Option<f64>,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub nsfw: bool,
    #[serde(default)]
    pub thumbnail_url: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub license_flags: Option<serde_json::Value>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub companions: Vec<String>,
    #[serde(default)]
    pub companion_details: Vec<serde_json::Value>,
    #[serde(default)]
    pub download_recipe: DownloadRecipeWire,
    #[serde(default)]
    pub engine_phase: u8,
    #[serde(default)]
    pub installed: bool,
    #[serde(default)]
    pub primary_path: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub added_at: i64,
    #[serde(default)]
    pub trained_words: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DownloadRecipeWire {
    #[serde(default)]
    pub files: Vec<serde_json::Value>,
    #[serde(default)]
    pub needs_token: Option<bool>,
}
