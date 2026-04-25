//! Scanner orchestrator. Coordinates the HF + Civitai stages, applies the
//! filter, and hands off to the sink. Per-family failure isolation: one
//! family hitting a rate limit or auth error does not abort the run —
//! see `ScanReport::per_family`.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::entry::CatalogEntry;
use crate::families::Family;

#[derive(Clone, Debug)]
pub struct ScanOptions {
    pub families: Vec<Family>,
    /// Civitai base-model entries below this download count are dropped at
    /// scanner time. Default 100; lower with `--min-downloads 0` for
    /// completeness, raise to thin the catalog.
    pub min_downloads: u64,
    pub include_nsfw: bool,
    pub hf_token: Option<String>,
    pub civitai_token: Option<String>,
    pub per_family_cap: Option<usize>,
    pub request_timeout: Duration,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            families: crate::families::ALL_FAMILIES.to_vec(),
            min_downloads: 100,
            include_nsfw: false,
            hf_token: None,
            civitai_token: None,
            per_family_cap: None,
            request_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScanReport {
    pub per_family: BTreeMap<Family, FamilyScanOutcome>,
    pub total_entries: usize,
}

#[derive(Clone, Debug)]
pub enum FamilyScanOutcome {
    Ok { entries: usize },
    RateLimited { partial: usize },
    AuthRequired,
    NetworkError { message: String },
}

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("network: {0}")]
    Network(#[from] reqwest::Error),
    #[error("decode: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("auth required for {host}")]
    AuthRequired { host: &'static str },
    #[error("rate limited by {host}")]
    RateLimited { host: &'static str },
}

/// Results of one family's scan: HF entries + Civitai entries (mixed in
/// the same vec; `Source` discriminates them downstream).
pub type FamilyScanResult = Result<Vec<CatalogEntry>, ScanError>;
