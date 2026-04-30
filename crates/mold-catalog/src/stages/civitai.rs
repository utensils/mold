//! Civitai stage. Walks `baseModels=` page-by-page; the base URL is
//! parameterized so tests can point at wiremock. Production passes
//! `"https://civitai.com"`.
//!
//! Hard-rule: only safetensors files are surfaced. Civitai's legacy `.pt`
//! ("PickleTensor") format is dropped at the scanner — arbitrary-code
//! execution risk on deserialization is not worth catalog completeness.

use serde::Deserialize;
use std::time::Duration;

use crate::entry::CatalogEntry;
use crate::normalizer::{from_civitai, CivitaiItem};
use crate::scanner::{ScanError, ScanOptions};

#[derive(Clone, Debug, Deserialize)]
struct CivitaiResponse {
    #[serde(default)]
    items: Vec<CivitaiItem>,
    #[serde(default)]
    metadata: CivitaiPaging,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct CivitaiPaging {
    #[serde(default, rename = "totalPages")]
    total_pages: Option<u32>,
    #[serde(default, rename = "nextPage")]
    next_page: Option<String>,
}

/// Scan one or more Civitai base-model buckets. Returns whatever entries
/// were collected before any unrecoverable error, paired with `Some(err)`
/// when the page walk had to bail (rate limit, auth, network, decode).
///
/// The partial-success contract is load-bearing: in production the
/// scanner walks ~10 pages successfully, then Civitai's per-IP cooldown
/// kicks in around page 11 and the throttle layer surfaces a
/// `RateLimited` error after exponential backoff. Returning the
/// already-collected entries (instead of `Err`-shaped early-return that
/// drops them) is the difference between "100 SDXL Civitai rows
/// catalogued" and "0 SDXL Civitai rows catalogued and the orchestrator
/// reports `RateLimited { partial: 0 }`".
///
/// Once the first persistent error fires we stop walking entirely:
/// remaining base_models share the same per-IP cooldown so every
/// subsequent request would 429 too.
pub async fn scan(
    base: &str,
    options: &ScanOptions,
    base_models: &[&str],
) -> (Vec<CatalogEntry>, Option<ScanError>) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(options.request_timeout.as_secs()))
        .build()
    {
        Ok(c) => c,
        Err(e) => return (Vec::new(), Some(ScanError::Network(e))),
    };

    let mut entries = Vec::new();
    'outer: for base_model in base_models {
        let mut page = 1u32;
        loop {
            let url = format!(
                "{base}/api/v1/models?baseModels={bm}&types=Checkpoint&sort=Most+Downloaded&limit=100&page={page}",
                bm = urlencoding::encode(base_model),
            );
            let resp = match http_get(&client, options, &url).await {
                Ok(r) => r,
                Err(e) => {
                    // Per-IP rate limit / auth / network failure is
                    // global to civitai.com — break the *outer* loop so
                    // the partial bucket is returned alongside the error.
                    return (entries, Some(e));
                }
            };
            let parsed: CivitaiResponse = match serde_json::from_str(&resp) {
                Ok(p) => p,
                Err(e) => {
                    return (entries, Some(ScanError::Decode(e)));
                }
            };
            if parsed.items.is_empty() {
                break;
            }
            for item in parsed.items {
                if let Some(e) = from_civitai(item) {
                    entries.push(e);
                    if let Some(cap) = options.per_family_cap {
                        if entries.len() >= cap {
                            return (entries, None);
                        }
                    }
                }
            }
            if parsed.metadata.next_page.is_none()
                && parsed
                    .metadata
                    .total_pages
                    .map(|t| page >= t)
                    .unwrap_or(true)
            {
                continue 'outer;
            }
            page += 1;
        }
    }
    (entries, None)
}

async fn http_get(
    client: &reqwest::Client,
    options: &ScanOptions,
    url: &str,
) -> Result<String, ScanError> {
    let mut builder = client.get(url);
    if let Some(t) = options.civitai_token.as_deref() {
        builder = builder.bearer_auth(t);
    }
    let req = builder.build()?;
    let outcome = crate::stages::throttle::polite_send(
        client,
        req,
        options.civitai_request_delay,
        options.default_429_backoff,
        options.max_429_retries,
        "civitai.com",
    )
    .await?;
    Ok(outcome.body)
}
