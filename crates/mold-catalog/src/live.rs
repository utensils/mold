//! Live catalog search — in-process proxy that replaces the bulk-scrape
//! model on the read path.
//!
//! Why this exists: the [`scanner`](crate::scanner) walks Civitai
//! page-by-page, hits per-IP rate limits around page 11, and persists
//! 1000+ rows the user never sees. What users actually do is type a
//! query and pick from ~20 results, so we proxy each query directly
//! against `/api/v1/models` with a small TTL cache to absorb duplicate
//! calls (SPA re-mounts, page-flips back to the same query, etc.).
//!
//! The cache is bounded — past `max_entries`, the oldest entry is
//! evicted on each insert. There is no LRU heat-tracking because the
//! TTL is short (5 minutes by default) and the working set is small.
//!
//! HF is intentionally out-of-scope for this module: HF rarely
//! rate-limits and the existing seed-walker keeps working until a
//! follow-up release deprecates it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::civitai_map::{map_base_model, CIVITAI_BASE_MODELS};
use crate::entry::{CatalogEntry, Kind, Source};
use crate::families::Family;
use crate::normalizer::{from_civitai, CivitaiItem};

/// Request shape for [`search`]. Hash-equal opts hit the same cache key,
/// so two callers asking the same question see identical results until
/// the TTL expires.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LiveSearchOpts {
    /// Free-text query forwarded as Civitai `query=`. `None` falls back
    /// to the default sort ("most downloaded") for the family/kind.
    pub q: Option<String>,
    /// Constrain to one mold family — translated to a multi-valued
    /// Civitai `baseModels=` parameter via [`CIVITAI_BASE_MODELS`].
    pub family: Option<Family>,
    /// `Lora` / `Checkpoint` etc. — translated to Civitai `types=`. The
    /// `Vae` / `TextEncoder` / `ControlNet` variants don't yet have a
    /// natural Civitai representation; they fall through to all types.
    pub kind: Option<Kind>,
    /// `None` (or `Some(Civitai)`) → query Civitai. Reserved for HF
    /// support in a follow-up; the variant is preserved for forward
    /// compatibility but the HF branch is currently a no-op.
    pub source: Option<Source>,
    pub page: u32,
    pub page_size: u32,
    /// When false, NSFW rows are filtered out post-fetch. Civitai's
    /// `nsfw=false` query param is not 100% reliable on its own.
    pub include_nsfw: bool,
    /// Optional tokens forwarded as `Authorization: Bearer …` to the
    /// upstream when present.
    pub civitai_token: Option<String>,
    pub hf_token: Option<String>,
}

impl Default for LiveSearchOpts {
    fn default() -> Self {
        Self {
            q: None,
            family: None,
            kind: None,
            source: None,
            page: 1,
            page_size: 20,
            include_nsfw: false,
            civitai_token: None,
            hf_token: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LiveSearchError {
    #[error("network: {0}")]
    Network(#[from] reqwest::Error),
    #[error("decode: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("upstream {host}: HTTP {status} {body}")]
    Upstream {
        host: &'static str,
        status: u16,
        body: String,
    },
}

/// Bounded TTL cache. Holds whole result vectors keyed on the
/// caller-visible [`LiveSearchOpts`] so equal opts hit the same entry.
#[derive(Clone)]
pub struct LiveCache {
    inner: Arc<Mutex<CacheState>>,
    ttl: Duration,
    max_entries: usize,
}

struct CacheState {
    map: HashMap<LiveSearchOpts, CacheValue>,
    /// Insertion order — oldest at the front. Used to evict on overflow.
    order: Vec<LiveSearchOpts>,
}

struct CacheValue {
    stored_at: Instant,
    entries: Vec<CatalogEntry>,
}

impl LiveCache {
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheState {
                map: HashMap::new(),
                order: Vec::new(),
            })),
            ttl,
            max_entries: max_entries.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("cache mutex").map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: &LiveSearchOpts) -> Option<Vec<CatalogEntry>> {
        let now = Instant::now();
        let mut state = self.inner.lock().expect("cache mutex");
        if let Some(v) = state.map.get(key) {
            if now.duration_since(v.stored_at) <= self.ttl {
                return Some(v.entries.clone());
            }
        }
        // Expired or missing — drop any stale entry so the next pass
        // doesn't re-check it.
        if state.map.remove(key).is_some() {
            state.order.retain(|k| k != key);
        }
        None
    }

    fn put(&self, key: LiveSearchOpts, entries: Vec<CatalogEntry>) {
        let mut state = self.inner.lock().expect("cache mutex");
        if state.map.contains_key(&key) {
            state.order.retain(|k| k != &key);
        }
        state.order.push(key.clone());
        state.map.insert(
            key,
            CacheValue {
                stored_at: Instant::now(),
                entries,
            },
        );
        while state.order.len() > self.max_entries {
            let evict = state.order.remove(0);
            state.map.remove(&evict);
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct CivitaiResponse {
    #[serde(default)]
    items: Vec<CivitaiItem>,
}

/// One-shot live query. Returns a normalized [`CatalogEntry`] vector
/// suitable for direct serialization into the `/api/catalog` wire
/// format. Errors are surfaced as [`LiveSearchError`] so the caller
/// can choose how to render them.
pub async fn search(
    civitai_base: &str,
    _hf_base: &str,
    cache: &LiveCache,
    opts: &LiveSearchOpts,
) -> Result<Vec<CatalogEntry>, LiveSearchError> {
    if let Some(hit) = cache.get(opts) {
        return Ok(hit);
    }

    // Civitai is the only live source for now (see module doc-comment).
    // Skip when the caller pinned `source=hf`.
    let entries = if matches!(opts.source, Some(Source::Hf)) {
        Vec::new()
    } else {
        civitai_search(civitai_base, opts).await?
    };

    cache.put(opts.clone(), entries.clone());
    Ok(entries)
}

async fn civitai_search(
    base: &str,
    opts: &LiveSearchOpts,
) -> Result<Vec<CatalogEntry>, LiveSearchError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    let mut url = reqwest::Url::parse(&format!("{base}/api/v1/models"))
        .expect("civitai base URL must be valid");
    let trimmed_q = opts
        .q
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("limit", &opts.page_size.clamp(1, 100).to_string());
        // Civitai rejects `page=` when `query=` is present, returning
        // 400 "Cannot use page param with query search. Use cursor-based
        // pagination." So we omit page when querying — first-page
        // results are all the LoRA picker / search box surface; deep
        // cursor pagination is a follow-up.
        if trimmed_q.is_none() {
            q.append_pair("page", &opts.page.max(1).to_string());
        }
        q.append_pair("sort", "Most Downloaded");
        if let Some(query) = trimmed_q.as_deref() {
            q.append_pair("query", query);
        }
        match opts.kind {
            Some(Kind::Lora) => {
                q.append_pair("types", "LORA");
            }
            Some(Kind::Checkpoint) => {
                q.append_pair("types", "Checkpoint");
            }
            // Vae / TextEncoder / ControlNet aren't searchable on Civitai
            // (or are not yet first-class in mold), so we fall through to
            // the default — both Checkpoints and LoRAs.
            _ => {
                q.append_pair("types", "Checkpoint");
                q.append_pair("types", "LORA");
            }
        }
        // Civitai quirk: when `query=` is combined with `baseModels=`,
        // the first cursor window comes back empty (followed by many
        // empty cursor steps). Verified directly against civitai.com:
        // `query=anime&baseModels=Flux.1+D&types=LORA` returns 0 items
        // even though each filter independently returns plenty. So
        // when the user types a query we drop the upstream
        // `baseModels=` filter and rely on the local family check
        // below — the response is broader but cards land on screen.
        if trimmed_q.is_none() {
            if let Some(family) = opts.family {
                for bm in CIVITAI_BASE_MODELS
                    .iter()
                    .filter(|bm| matches!(map_base_model(bm), Some((f, _, _)) if f == family))
                {
                    q.append_pair("baseModels", bm);
                }
            }
        }
        if !opts.include_nsfw {
            q.append_pair("nsfw", "false");
        }
    }

    tracing::debug!(target: "catalog.live", url = %url, "civitai request");
    let mut req = client.get(url);
    if let Some(token) = opts.civitai_token.as_deref() {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await?;
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        return Err(LiveSearchError::Upstream {
            host: "civitai.com",
            status: status.as_u16(),
            body: body.chars().take(400).collect(),
        });
    }
    let parsed: CivitaiResponse = serde_json::from_str(&body)?;

    let mut out = Vec::with_capacity(parsed.items.len());
    for item in parsed.items {
        let nsfw_item = item.nsfw;
        if let Some(entry) = from_civitai(item) {
            // Civitai's nsfw filter is best-effort — drop again locally so
            // a row whose top-level `nsfw=false` but version is gated
            // doesn't leak through. The from_civitai output preserves the
            // top-level flag in `entry.nsfw`.
            if !opts.include_nsfw && (nsfw_item || entry.nsfw) {
                continue;
            }
            // Family filter is upstream via baseModels=, but normalize
            // again locally since Civitai will sometimes return rows for
            // a slightly different baseModel string than what we asked
            // for (e.g. "Flux.1 D" vs "Flux.1 Dev" in older entries).
            if let Some(family) = opts.family {
                if entry.family != family {
                    continue;
                }
            }
            // Same idea for kind: Civitai's `types=` does the upstream
            // filtering, but the response sometimes mixes (a LoCon row
            // typed "LORA" upstream is still surfaced under types=LORA).
            // Trusting the per-row Kind avoids that drift.
            if let Some(kind) = opts.kind {
                if entry.kind != kind {
                    continue;
                }
            }
            out.push(entry);
        }
    }
    Ok(out)
}
