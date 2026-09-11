//! Wire types for `GET /api/catalog/installed` and `GET /api/catalog/search`.
//!
//! One struct serves both sides: `mold-server` serializes it and the
//! client in this crate deserializes it, so the payload shape can no
//! longer drift silently between the two. Every field is always
//! serialized (no `skip_serializing_if`) — the SPA expects explicit
//! `null`s so a single TypeScript interface covers both endpoints.
//! Deserialization stays tolerant of older servers via `#[serde(default)]`
//! on everything the client can reasonably backfill.
//!
//! The INSTALLED endpoint builds these structs directly, so the compiler
//! keeps it honest. The SEARCH endpoint does not: it serializes an ad-hoc
//! `serde_json::json!` from `live_entry_to_wire`, and nothing but a test can
//! make the two agree. They did not — the sidecar sends `null` for the
//! recipe's token kind and for both timestamps, so `needs_token: bool` and
//! `created_at: String` parsed fine there while every live search failed to
//! decode against a real Civitai row. Fields the client does not display are
//! deliberately typed loosely (`serde_json::Value`) rather than mirrored, and
//! `mold-server`'s `a_live_search_page_deserializes_into_the_type_the_client_reads`
//! pins the live half against the real handler.

use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

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
    /// `None` means the legacy sidecar did not record a safety classification.
    /// Keep it distinct from an explicit `Some(false)` on the JSON wire.
    #[serde(default)]
    pub nsfw: Option<bool>,
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
    #[serde(default = "default_true")]
    pub supported: bool,
    #[serde(default)]
    pub installed: bool,
    #[serde(default)]
    pub primary_path: Option<String>,
    /// Upstream publication time, in EPOCH SECONDS — an integer on both
    /// wires, never a formatted string.
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub updated_at: Option<i64>,
    #[serde(default)]
    pub added_at: i64,
    #[serde(default)]
    pub trained_words: Vec<String>,
    #[serde(default)]
    pub page_url: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DownloadRecipeWire {
    #[serde(default)]
    pub files: Vec<serde_json::Value>,
    /// Which credential the download needs: `"hf"`, `"civitai"`, or absent.
    ///
    /// A token KIND, not a yes/no — `mold_catalog::entry::TokenKind` on the
    /// serializing side. Reading it as a `bool` is what made a live search
    /// fail with `invalid type: string "civitai", expected a boolean`.
    #[serde(default)]
    pub needs_token: Option<String>,
}

/// Query for `GET /api/catalog/search`.
///
/// Every field is optional because the server owns each default (page 1,
/// page size 20, `downloads` sort, NSFW included) and a client that repeated
/// them here would be a second authority on what an omitted parameter means.
/// `sort` stays a `String` rather than an enum so a value this build has
/// never heard of still reaches the host, which answers 422 with the list it
/// accepts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CatalogSearchQuery {
    pub q: Option<String>,
    pub family: Option<String>,
    pub kind: Option<String>,
    pub source: Option<String>,
    pub sort: Option<String>,
    pub page: Option<u32>,
    pub page_size: Option<u32>,
    pub include_nsfw: Option<bool>,
}

impl CatalogSearchQuery {
    /// The query parameters to put on the wire, omitting every field the
    /// caller left unset so the host applies its own defaults.
    pub fn query_pairs(&self) -> Vec<(&'static str, String)> {
        let mut pairs: Vec<(&'static str, String)> = Vec::new();
        let mut push_text = |key: &'static str, value: &Option<String>| {
            if let Some(text) = value.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                pairs.push((key, text.to_string()));
            }
        };
        push_text("q", &self.q);
        push_text("family", &self.family);
        push_text("kind", &self.kind);
        push_text("source", &self.source);
        push_text("sort", &self.sort);
        if let Some(page) = self.page {
            pairs.push(("page", page.to_string()));
        }
        if let Some(page_size) = self.page_size {
            pairs.push(("page_size", page_size.to_string()));
        }
        if let Some(include_nsfw) = self.include_nsfw {
            pairs.push(("include_nsfw", include_nsfw.to_string()));
        }
        pairs
    }
}

/// One live catalog hit.
///
/// `GET /api/catalog/search` and `GET /api/catalog/installed` are
/// wire-uniform by construction (see [`InstalledCatalogEntry`]'s own note),
/// so one struct deserializes both rather than two that can drift apart.
pub type CatalogSearchEntry = InstalledCatalogEntry;

/// One page of `GET /api/catalog/search`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogSearchPage {
    #[serde(default)]
    pub entries: Vec<CatalogSearchEntry>,
    #[serde(default)]
    pub page: i64,
    #[serde(default)]
    pub page_size: i64,
    #[serde(default)]
    pub total: i64,
    /// Provider-scoped failures from a merged search. The healthy provider's
    /// rows are still in `entries`, so these are warnings to report beside
    /// the results rather than an error that replaces them.
    #[serde(default)]
    pub provider_errors: Vec<CatalogProviderErrorWire>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogProviderErrorWire {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub retry_after_seconds: Option<u64>,
}

/// `GET /api/catalog/families` — the static taxonomy, with no per-family
/// counts. Present so a client can offer `--family` values without guessing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogFamiliesResponse {
    #[serde(default)]
    pub families: Vec<CatalogFamilyWire>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFamilyWire {
    pub family: String,
}

#[cfg(test)]
mod tests {
    use super::{CatalogSearchPage, CatalogSearchQuery, InstalledCatalogEntry};

    #[test]
    fn a_search_query_puts_only_the_fields_the_caller_set_on_the_wire() {
        assert!(CatalogSearchQuery::default().query_pairs().is_empty());

        let pairs = CatalogSearchQuery {
            q: Some("  flux  ".into()),
            family: Some(String::new()),
            sort: Some("recent".into()),
            page: Some(2),
            include_nsfw: Some(false),
            ..CatalogSearchQuery::default()
        }
        .query_pairs();
        assert_eq!(
            pairs,
            vec![
                ("q", "flux".to_string()),
                ("sort", "recent".to_string()),
                ("page", "2".to_string()),
                ("include_nsfw", "false".to_string()),
            ]
        );
    }

    /// One real `GET /api/catalog/search` row, captured from a live host.
    ///
    /// Field TYPES are the point, not the values: `added_at` is an integer,
    /// `created_at` and `updated_at` are null where this row has none but are
    /// integers where it does, `license_flags` is an OBJECT, `rating` and
    /// `author` are null, `nsfw` is a bare boolean, and
    /// `download_recipe.needs_token` is the string `"civitai"`. Every one of
    /// those was a live search that would not decode.
    #[test]
    fn a_real_live_search_row_from_a_host_deserializes() {
        let page: CatalogSearchPage = serde_json::from_value(serde_json::json!({
            "entries": [{
                "added_at": 1701098351i64,
                "author": null,
                "bundling": "single-file",
                "companion_details": [],
                "companions": [],
                "created_at": null,
                "description": "A long description of the adapter.",
                "download_count": 2637,
                "download_recipe": {
                    "files": [{
                        "url": "https://civitai.com/api/download/models/691639",
                        "dest": "model.safetensors",
                        "sha256": null,
                        "size_bytes": 167938890
                    }],
                    "needs_token": "civitai"
                },
                "family": "flux",
                "family_role": "finetune",
                "file_format": "safetensors",
                "id": "cv:691639",
                "installed": false,
                "kind": "lora",
                "license": null,
                "license_flags": {
                    "commercial": true,
                    "derivatives": false,
                    "different_license": null
                },
                "likes": 412,
                "modality": "image",
                "name": "Flux Skin Texture",
                "nsfw": false,
                "page_url": "https://civitai.com/models/1?modelVersionId=691639",
                "primary_path": null,
                "rating": null,
                "size_bytes": 167938890i64,
                "source": "civitai",
                "source_id": "691639",
                "sub_family": "dev",
                "supported": true,
                "tags": ["skin", "realism"],
                "thumbnail_url": "https://example.invalid/thumb.jpg",
                "trained_words": ["realskin"],
                "updated_at": 1701098400i64
            }],
            "page": 1,
            "page_size": 5,
            "provider_errors": [],
            "total": 1
        }))
        .expect("a real live row deserializes");

        assert_eq!(page.total, 1);
        assert_eq!(page.page_size, 5);
        assert!(page.provider_errors.is_empty());
        let row = &page.entries[0];
        assert_eq!(row.id, "cv:691639");
        assert_eq!(row.source, "civitai");
        assert_eq!(row.name, "Flux Skin Texture");
        assert_eq!(row.author, None);
        assert_eq!(row.family, "flux");
        assert_eq!(row.kind, "lora");
        assert_eq!(row.size_bytes, Some(167_938_890));
        assert_eq!(row.download_count, 2637);
        assert_eq!(row.nsfw, Some(false));
        assert!(row.supported);
        assert!(!row.installed);
        assert_eq!(row.created_at, None);
        assert_eq!(row.updated_at, Some(1_701_098_400));
        assert_eq!(row.added_at, 1_701_098_351);
        assert_eq!(row.download_recipe.needs_token.as_deref(), Some("civitai"));
        assert!(row
            .license_flags
            .as_ref()
            .is_some_and(|flags| flags.is_object()));
        assert_eq!(
            row.page_url.as_deref().map(|url| url.contains("civitai")),
            Some(true)
        );
    }

    /// A page from an older host that knew none of the additive fields still
    /// parses: every one of them backfills rather than refusing the page.
    #[test]
    fn a_sparse_page_from_an_older_host_still_parses() {
        let page: CatalogSearchPage = serde_json::from_value(serde_json::json!({
            "entries": [{
                "id": "hf:black-forest-labs/FLUX.1-dev",
                "name": "FLUX.1 [dev]",
                "family": "flux",
                "kind": "checkpoint"
            }],
            "page": 1,
            "page_size": 20,
            "total": 1
        }))
        .expect("an older host's page deserializes");
        assert_eq!(page.entries[0].id, "hf:black-forest-labs/FLUX.1-dev");
        assert!(page.entries[0].supported, "absent `supported` means yes");
        assert!(page.provider_errors.is_empty());
    }

    #[test]
    fn installed_catalog_entry_round_trips_unknown_nsfw_as_null() {
        let entry: InstalledCatalogEntry = serde_json::from_value(serde_json::json!({
            "id": "cv:1",
            "name": "Legacy install",
            "family": "flux",
            "kind": "lora",
            "nsfw": null
        }))
        .expect("null is a valid unknown safety classification");

        let wire = serde_json::to_value(entry).unwrap();
        assert_eq!(wire["nsfw"], serde_json::Value::Null);
    }
}
