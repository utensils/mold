//! Hugging Face stage. Walks `seeds_for(family)` and follows the
//! `base_model:` finetune graph one page at a time. The base URL is
//! parameterized so tests can point at wiremock; production passes
//! `"https://huggingface.co"`.

use serde::Deserialize;
use std::time::Duration;

use crate::entry::{CatalogEntry, FamilyRole};
use crate::families::Family;
use crate::normalizer::{from_hf, HfDetail, HfTreeEntry};
use crate::scanner::{ScanError, ScanOptions};

#[derive(Clone, Debug, Deserialize)]
struct HfModelStub {
    id: String,
}

pub async fn scan_family(
    base: &str,
    options: &ScanOptions,
    family: Family,
    seeds: &[&str],
) -> Result<Vec<CatalogEntry>, ScanError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(options.request_timeout.as_secs()))
        .build()?;

    let mut entries = Vec::new();
    for seed in seeds {
        if let Some(e) =
            fetch_repo_as_entry(&client, base, options, seed, family, FamilyRole::Foundation)
                .await?
        {
            entries.push(e);
        }
        let mut page = 1u32;
        loop {
            let url = format!(
                "{base}/api/models?filter=base_model:{seed}&sort=downloads&direction=-1&limit=100&page={page}",
            );
            let resp = http_get(&client, options, &url).await?;
            let stubs: Vec<HfModelStub> = serde_json::from_str(&resp)?;
            if stubs.is_empty() {
                break;
            }
            for stub in stubs {
                if let Some(e) = fetch_repo_as_entry(
                    &client,
                    base,
                    options,
                    &stub.id,
                    family,
                    FamilyRole::Finetune,
                )
                .await?
                {
                    entries.push(e);
                }
                if let Some(cap) = options.per_family_cap {
                    if entries.len() >= cap {
                        return Ok(entries);
                    }
                }
            }
            page += 1;
        }
    }
    Ok(entries)
}

async fn fetch_repo_as_entry(
    client: &reqwest::Client,
    base: &str,
    options: &ScanOptions,
    repo: &str,
    family: Family,
    role: FamilyRole,
) -> Result<Option<CatalogEntry>, ScanError> {
    let detail_url = format!("{base}/api/models/{repo}");
    let detail_body = http_get(client, options, &detail_url).await?;
    let detail: HfDetail = serde_json::from_str(&detail_body)?;

    let tree_url = format!("{base}/api/models/{repo}/tree/main");
    let tree_body = http_get(client, options, &tree_url).await?;
    let tree: Vec<HfTreeEntry> = serde_json::from_str(&tree_body)?;

    match from_hf(detail, tree, family, role) {
        Ok(e) => Ok(Some(e)),
        Err(_) => Ok(None),
    }
}

async fn http_get(
    client: &reqwest::Client,
    options: &ScanOptions,
    url: &str,
) -> Result<String, ScanError> {
    let mut req = client.get(url);
    if let Some(t) = options.hf_token.as_deref() {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await?;
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(ScanError::AuthRequired {
            host: "huggingface.co",
        });
    }
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(ScanError::RateLimited {
            host: "huggingface.co",
        });
    }
    Ok(resp.text().await?)
}
