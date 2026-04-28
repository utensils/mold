//! Scanner orchestrator. Coordinates the HF + Civitai stages, applies the
//! filter, and hands off to the sink. Per-family failure isolation: one
//! family hitting a rate limit or auth error does not abort the run —
//! see `ScanReport::per_family`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::civitai_map::map_base_model;
use crate::entry::CatalogEntry;
use crate::families::Family;
use crate::{filter, hf_seeds, stages};

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
    /// Sleep before every HF request. Defaults to 250 ms (~4 req/s), well
    /// under the documented unauthenticated `/api/models` ceiling. A full
    /// `mold catalog refresh` makes thousands of requests; without this
    /// throttle the scanner fires them as fast as the network returns,
    /// which is what triggers HTTP 429s. Tests set this to `Duration::ZERO`.
    pub hf_request_delay: Duration,
    /// Sleep before every Civitai request. Defaults to 1.5 s (~40 req/min)
    /// — Civitai's anonymous limit is closer to ~50/min and we want a
    /// safety margin. Tests set this to `Duration::ZERO`.
    pub civitai_request_delay: Duration,
    /// Number of times to retry after an HTTP 429 before giving up on a
    /// request. Defaults to 3. Tests set this to 0 so they don't pay
    /// retry latency on intentional 429 fixtures.
    pub max_429_retries: u8,
    /// Backoff used when a 429 response carries no `Retry-After` header.
    /// Doubled on each subsequent retry (5 s → 10 s → 20 s by default).
    pub default_429_backoff: Duration,
    /// Defensive cap on per-family wall-clock walk time. When `Some`,
    /// the HF stage of any single family will gracefully bail (returning
    /// the partial entry bucket collected so far) once `now >= start +
    /// cap`. Default `None` preserves the unbounded behaviour for
    /// callers that explicitly want a complete scan.
    ///
    /// Belt-and-braces against the "scanner appears hung" symptom — even
    /// a true tokio-future-wakeup bug is bounded by this cap if it's
    /// configured.
    pub max_family_wallclock: Option<Duration>,
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
            hf_request_delay: Duration::from_millis(250),
            civitai_request_delay: Duration::from_millis(1500),
            max_429_retries: 3,
            default_429_backoff: Duration::from_secs(5),
            max_family_wallclock: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScanReport {
    pub per_family: BTreeMap<Family, FamilyScanOutcome>,
    pub total_entries: usize,
}

/// Live progress snapshot, updated by `run_scan_with_progress` as it walks
/// each family. Lock briefly with `lock()`, copy out, drop the guard — the
/// scanner contends for the same lock when it advances.
///
/// The HF stage of a large family (SDXL has thousands of fine-tunes) takes
/// hours to walk at the politeness throttle. Without per-seed and per-page
/// fields, the SPA shows `families_done:0` for the entire walk, which is
/// indistinguishable from a hang. The extra fields here let the UI render
/// "Scanning sdxl seed 1: page 17 (1700 finetunes so far)".
#[derive(Clone, Debug, Default)]
pub struct ScanProgress {
    pub families_total: usize,
    pub families_done: usize,
    pub current_family: Option<Family>,
    /// "hf" or "civitai" — set as the scanner enters that stage of the
    /// current family. `None` between families.
    pub current_stage: Option<&'static str>,
    /// Seed currently being walked within the HF stage. Resets to `None`
    /// at the end of the HF stage and between families.
    pub current_seed: Option<String>,
    /// Pages walked within `current_seed`. Resets to 0 when the seed
    /// changes.
    pub pages_done: usize,
    /// Entries collected within the current family. Resets to 0 between
    /// families. Cumulative across both seeds and stages within a family.
    pub entries_so_far: usize,
}

pub type ProgressHandle = Arc<Mutex<ScanProgress>>;

pub(crate) fn update_progress<F: FnOnce(&mut ScanProgress)>(handle: Option<&ProgressHandle>, f: F) {
    if let Some(h) = handle {
        if let Ok(mut p) = h.lock() {
            f(&mut p);
        }
    }
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

/// Orchestrate one full scan across HF + Civitai for the families in
/// `options.families`. Each family runs as an independent task; errors
/// are isolated per family and surfaced via `ScanReport.per_family`.
///
/// `hf_base` and `civitai_base` allow tests to point the stages at
/// wiremock; production callers pass `"https://huggingface.co"` and
/// `"https://civitai.com"`.
pub async fn run_scan(hf_base: &str, civitai_base: &str, options: &ScanOptions) -> ScanReport {
    run_scan_with_progress(hf_base, civitai_base, options, None).await
}

/// Same as [`run_scan`] but writes intra-scan progress into a shared
/// `ScanProgress`. The server hands the same handle to its REST status
/// endpoint so clients can render "Scanning sdxl… (3/9)" with a
/// determinate progress bar instead of a blind spinner.
pub async fn run_scan_with_progress(
    hf_base: &str,
    civitai_base: &str,
    options: &ScanOptions,
    progress: Option<ProgressHandle>,
) -> ScanReport {
    let mut report = ScanReport::default();
    update_progress(progress.as_ref(), |p| {
        p.families_total = options.families.len();
        p.families_done = 0;
        p.current_family = None;
        p.current_stage = None;
    });

    for &family in &options.families {
        update_progress(progress.as_ref(), |p| {
            p.current_family = Some(family);
            p.current_stage = Some("hf");
            // Per-family resets: the seed/page/entry counters track the
            // *current* family only.
            p.current_seed = None;
            p.pages_done = 0;
            p.entries_so_far = 0;
        });
        let mut bucket: Vec<crate::entry::CatalogEntry> = Vec::new();
        let mut auth_required = false;
        let mut rate_limited = false;
        let mut network_error: Option<String> = None;

        // ── HF stage ──────────────────────────────────────────────────
        // Call scan_family once per seed so that a transient error on one
        // seed does not discard entries already fetched from earlier seeds.
        for seed in hf_seeds::seeds_for(family) {
            match stages::hf::scan_family(
                hf_base,
                options,
                family,
                std::slice::from_ref(seed),
                progress.as_ref(),
            )
            .await
            {
                Ok(entries) => bucket.extend(entries),
                Err(ScanError::AuthRequired { .. }) => {
                    auth_required = true;
                }
                Err(ScanError::RateLimited { .. }) => {
                    rate_limited = true;
                }
                Err(e) => {
                    network_error = Some(e.to_string());
                }
            }
        }

        // ── Civitai stage ─────────────────────────────────────────────
        let cv_keys: Vec<&'static str> = crate::civitai_map::CIVITAI_BASE_MODELS
            .iter()
            .copied()
            .filter(|k| matches!(map_base_model(k), Some((f, _, _)) if f == family))
            .collect();
        if !cv_keys.is_empty() {
            update_progress(progress.as_ref(), |p| {
                p.current_stage = Some("civitai");
                // The per-seed counter is HF-only; clear it so the SPA
                // doesn't show a stale seed name during the Civitai stage.
                p.current_seed = None;
            });
            match stages::civitai::scan(civitai_base, options, &cv_keys).await {
                Ok(entries) => bucket.extend(entries),
                Err(ScanError::AuthRequired { .. }) => auth_required = true,
                Err(ScanError::RateLimited { .. }) => rate_limited = true,
                Err(e) => {
                    network_error.get_or_insert(e.to_string());
                }
            };
        }

        let kept = filter::apply(bucket, options);
        report.total_entries += kept.len();

        let outcome = if !kept.is_empty() {
            // At least one entry survived the filter — report success even if
            // later seeds encountered transient errors.
            FamilyScanOutcome::Ok {
                entries: kept.len(),
            }
        } else if rate_limited {
            FamilyScanOutcome::RateLimited { partial: 0 }
        } else if auth_required {
            FamilyScanOutcome::AuthRequired
        } else if let Some(msg) = network_error {
            FamilyScanOutcome::NetworkError { message: msg }
        } else {
            FamilyScanOutcome::Ok { entries: 0 }
        };
        report.per_family.insert(family, outcome);
        update_progress(progress.as_ref(), |p| {
            p.families_done += 1;
            p.current_stage = None;
        });
    }

    update_progress(progress.as_ref(), |p| {
        p.current_family = None;
        p.current_stage = None;
    });

    report
}
