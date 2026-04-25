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

pub async fn scan(
    base: &str,
    options: &ScanOptions,
    base_models: &[&str],
) -> Result<Vec<CatalogEntry>, ScanError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(options.request_timeout.as_secs()))
        .build()?;

    let mut entries = Vec::new();
    for base_model in base_models {
        let mut page = 1u32;
        loop {
            let url = format!(
                "{base}/api/v1/models?baseModels={bm}&types=Checkpoint&sort=Most+Downloaded&limit=100&page={page}",
                bm = urlencoding::encode(base_model),
            );
            let resp = http_get(&client, options, &url).await?;
            let parsed: CivitaiResponse = serde_json::from_str(&resp)?;
            if parsed.items.is_empty() {
                break;
            }
            for item in parsed.items {
                if let Some(e) = from_civitai(item) {
                    entries.push(e);
                    if let Some(cap) = options.per_family_cap {
                        if entries.len() >= cap {
                            return Ok(entries);
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
                break;
            }
            page += 1;
        }
    }
    Ok(entries)
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
