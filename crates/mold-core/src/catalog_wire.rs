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
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
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
    #[serde(default)]
    pub needs_token: Option<bool>,
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
    use super::{CatalogSearchQuery, InstalledCatalogEntry};

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
