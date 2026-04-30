//! Hugging Face stage. Walks `seeds_for(family)` and follows the
//! `base_model:` finetune graph one page at a time. The base URL is
//! parameterized so tests can point at wiremock; production passes
//! `"https://huggingface.co"`.

use serde::Deserialize;
use std::time::{Duration, Instant};

use crate::entry::{CatalogEntry, FamilyRole};
use crate::families::Family;
use crate::normalizer::{from_hf, HfDetail, HfTreeEntry};
use crate::scanner::{update_progress, ProgressHandle, ScanError, ScanOptions};

#[derive(Clone, Debug, Deserialize)]
struct HfModelStub {
    id: String,
}

pub async fn scan_family(
    base: &str,
    options: &ScanOptions,
    family: Family,
    seeds: &[&str],
    progress: Option<&ProgressHandle>,
) -> Result<Vec<CatalogEntry>, ScanError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(options.request_timeout.as_secs()))
        .build()?;

    let family_start = Instant::now();
    let deadline = options.max_family_wallclock.map(|cap| family_start + cap);
    let deadline_reached = |label: &str, now: Instant| -> bool {
        if let Some(d) = deadline {
            if now >= d {
                tracing::warn!(
                    family = %family,
                    cap_ms = options
                        .max_family_wallclock
                        .map(|c| c.as_millis() as u64)
                        .unwrap_or(0),
                    elapsed_ms = family_start.elapsed().as_millis() as u64,
                    "scan_family: {} cap reached, returning partial entries",
                    label,
                );
                return true;
            }
        }
        false
    };
    let mut entries = Vec::new();
    for seed in seeds {
        if deadline_reached("pre-seed", Instant::now()) {
            return Ok(entries);
        }
        // Stamp the seed and reset the per-seed page counter so the SPA
        // can render "seed N: page K" instead of a single static label
        // for the whole HF stage.
        update_progress(progress, |p| {
            p.current_seed = Some((*seed).to_string());
            p.pages_done = 0;
        });
        let seed_start = Instant::now();
        let entries_before_seed = entries.len();
        tracing::info!(family = %family, seed = %seed, "scan_family: seed start");
        if let Some(e) =
            fetch_repo_as_entry(&client, base, options, seed, family, FamilyRole::Foundation)
                .await?
        {
            entries.push(e);
            update_progress(progress, |p| p.entries_so_far += 1);
        }
        let mut page = 1u32;
        let mut pages_for_seed: u32 = 0;
        loop {
            if deadline_reached("pre-page", Instant::now()) {
                return Ok(entries);
            }
            let url = format!(
                "{base}/api/models?filter=base_model:{seed}&sort=downloads&direction=-1&limit=100&page={page}",
            );
            let resp = http_get(&client, options, &url).await?;
            let stubs: Vec<HfModelStub> = serde_json::from_str(&resp)?;
            tracing::debug!(
                family = %family,
                seed = %seed,
                page = page,
                stubs = stubs.len(),
                "scan_family: page fetched",
            );
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
                    update_progress(progress, |p| p.entries_so_far += 1);
                }
                if let Some(cap) = options.per_family_cap {
                    if entries.len() >= cap {
                        tracing::info!(
                            family = %family,
                            cap = cap,
                            entries = entries.len(),
                            "scan_family: per_family_cap reached, stopping early",
                        );
                        return Ok(entries);
                    }
                }
                // Mid-page deadline check too — a slow per-stub walk can
                // exceed the cap by more than one request without this.
                if deadline_reached("mid-page", Instant::now()) {
                    return Ok(entries);
                }
            }
            // Increment after the page is fully processed so the snapshot
            // taken when the *next* page list-request fires reflects "N
            // pages already walked".
            update_progress(progress, |p| p.pages_done += 1);
            pages_for_seed += 1;
            page += 1;
        }
        tracing::info!(
            family = %family,
            seed = %seed,
            entries = entries.len() - entries_before_seed,
            pages = pages_for_seed,
            elapsed_ms = seed_start.elapsed().as_millis() as u64,
            "scan_family: seed done",
        );
    }
    tracing::info!(
        family = %family,
        seeds = seeds.len(),
        total_entries = entries.len(),
        elapsed_ms = family_start.elapsed().as_millis() as u64,
        "scan_family: family done",
    );
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
    let mut builder = client.get(url);
    if let Some(t) = options.hf_token.as_deref() {
        builder = builder.bearer_auth(t);
    }
    let req = builder.build()?;
    let outcome = crate::stages::throttle::polite_send(
        client,
        req,
        options.hf_request_delay,
        options.default_429_backoff,
        options.max_429_retries,
        "huggingface.co",
    )
    .await?;
    Ok(outcome.body)
}
