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
use crate::entry::{CatalogEntry, FamilyRole, Kind, Source};
use crate::families::Family;
use crate::normalizer::{
    from_civitai, from_hf, CivitaiItem, CivitaiVersion, HfDetail, HfTreeEntry,
};

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
            include_nsfw: true,
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
///
/// Source routing:
/// - `Some(Civitai)` → Civitai only
/// - `Some(Hf)`      → HF only
/// - `None`          → both (concatenated, Civitai first)
pub async fn search(
    civitai_base: &str,
    hf_base: &str,
    cache: &LiveCache,
    opts: &LiveSearchOpts,
) -> Result<Vec<CatalogEntry>, LiveSearchError> {
    if let Some(hit) = cache.get(opts) {
        return Ok(hit);
    }

    let mut entries = Vec::new();
    if !matches!(opts.source, Some(Source::Hf)) {
        entries.extend(civitai_search(civitai_base, opts).await?);
    }
    if !matches!(opts.source, Some(Source::Civitai)) {
        // HF errors must not nuke Civitai results. Log and continue.
        match hf_search(hf_base, opts).await {
            Ok(rows) => entries.extend(rows),
            Err(e) => {
                tracing::warn!(target: "catalog.live", error = %e, "hf search failed");
            }
        }
    }

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

// ── HF live search ──────────────────────────────────────────────────────────

/// HF `/api/models?search=…` summary row. Lean compared to the per-repo
/// detail; `tree` is fetched lazily on download (`fetch_hf_repo`).
#[derive(Clone, Debug, Deserialize)]
struct HfSearchHit {
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, rename = "pipeline_tag")]
    pipeline_tag: Option<String>,
    #[serde(default, rename = "createdAt")]
    created_at: Option<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
}

async fn hf_search(
    base: &str,
    opts: &LiveSearchOpts,
) -> Result<Vec<CatalogEntry>, LiveSearchError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    let mut url =
        reqwest::Url::parse(&format!("{base}/api/models")).expect("hf base URL must be valid");
    let trimmed_q = opts.q.as_deref().map(str::trim).filter(|s| !s.is_empty());
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("limit", &opts.page_size.clamp(1, 100).to_string());
        q.append_pair("sort", "downloads");
        q.append_pair("direction", "-1");
        if let Some(query) = trimmed_q {
            q.append_pair("search", query);
        }
        // Pin to diffusers / text-to-image / image-to-video so the search
        // doesn't drown in unrelated NLP repos. HF's `filter` accepts
        // multiple values; tag-style filters are AND across types,
        // so we use OR-equivalent `pipeline_tag` filters by listing each.
        match opts.kind {
            Some(Kind::Lora) => {
                q.append_pair("filter", "lora");
            }
            _ => {
                q.append_pair("filter", "diffusers");
            }
        }
    }

    tracing::debug!(target: "catalog.live", url = %url, "hf search request");
    let mut req = client.get(url);
    if let Some(token) = opts.hf_token.as_deref() {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await?;
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        return Err(LiveSearchError::Upstream {
            host: "huggingface.co",
            status: status.as_u16(),
            body: body.chars().take(400).collect(),
        });
    }
    let hits: Vec<HfSearchHit> = serde_json::from_str(&body)?;

    let mut out = Vec::with_capacity(hits.len());
    for hit in hits {
        let Some((family, family_role)) =
            family_from_hf(&hit.id, &hit.tags, hit.pipeline_tag.as_deref())
        else {
            continue;
        };
        if let Some(want) = opts.family {
            if family != want {
                continue;
            }
        }
        let entry = hf_summary_to_entry(hit, family, family_role);
        // Kind filter applied after entry construction since HF tags aren't
        // a perfect signal upstream.
        if let Some(want) = opts.kind {
            if entry.kind != want {
                continue;
            }
        }
        out.push(entry);
    }
    Ok(out)
}

/// Build a recipe-less `CatalogEntry` from an HF search hit. The recipe
/// is left empty: the search/list view doesn't need it, and download
/// flow re-fetches detail+tree via `fetch_hf_repo` to compute file URLs.
fn hf_summary_to_entry(hit: HfSearchHit, family: Family, family_role: FamilyRole) -> CatalogEntry {
    use crate::entry::{Bundling, CatalogId, DownloadRecipe, FileFormat, LicenseFlags, Modality};

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let modality = match family {
        Family::LtxVideo | Family::Ltx2 => Modality::Video,
        _ => Modality::Image,
    };
    // Heuristic kind: "lora" tag wins, otherwise checkpoint.
    let kind = if hit
        .tags
        .iter()
        .any(|t| t.eq_ignore_ascii_case("lora") || t.eq_ignore_ascii_case("adapter"))
    {
        Kind::Lora
    } else {
        Kind::Checkpoint
    };
    let name = hit.id.split('/').next_back().unwrap_or(&hit.id).to_string();

    CatalogEntry {
        id: CatalogId::from(format!("hf:{}", hit.id)),
        source: Source::Hf,
        source_id: hit.id.clone(),
        name,
        author: hit.author,
        family,
        family_role,
        sub_family: None,
        modality,
        kind,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: None,
        download_count: hit.downloads,
        rating: None,
        likes: hit.likes,
        nsfw: false,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: hit.tags,
        companions: Vec::new(),
        // Recipe is deliberately empty — call `fetch_hf_repo` at download
        // time to materialize file URLs from the tree.
        download_recipe: DownloadRecipe {
            files: Vec::new(),
            needs_token: None,
        },
        engine_phase: crate::civitai_map::engine_phase_for(family, Bundling::Separated),
        created_at: parse_iso(hit.created_at.as_deref()),
        updated_at: parse_iso(hit.last_modified.as_deref()),
        added_at: now,
        trained_words: Vec::new(),
    }
}

fn parse_iso(s: Option<&str>) -> Option<i64> {
    s.and_then(|s| {
        time::OffsetDateTime::parse(s, &time::format_description::well_known::Iso8601::DEFAULT)
            .ok()
            .map(|dt| dt.unix_timestamp())
    })
}

/// Best-effort family inference from HF metadata. Returns `None` for
/// repos that don't map to a supported mold family — the caller drops
/// those silently rather than surfacing a row the user can't generate
/// from. Order matters: more-specific patterns (`flux.2`, `ltx-video-2`)
/// must precede their less-specific siblings (`flux`, `ltx-video`).
pub fn family_from_hf(
    repo_id: &str,
    tags: &[String],
    _pipeline_tag: Option<&str>,
) -> Option<(Family, FamilyRole)> {
    let id_lower = repo_id.to_ascii_lowercase();
    let role_for = |id: &str, family: Family| -> FamilyRole {
        let seeds = crate::hf_seeds::seeds_for(family);
        if seeds.iter().any(|seed| seed.eq_ignore_ascii_case(id)) {
            FamilyRole::Foundation
        } else {
            FamilyRole::Finetune
        }
    };

    // Repo-id substring match — the load-bearing path. HF doesn't
    // expose a canonical "model family" tag, so id-substring + curated
    // seed list is the cleanest signal.
    let family = if id_lower.contains("flux.2") || id_lower.contains("flux-2") {
        Family::Flux2
    } else if id_lower.contains("flux.1")
        || id_lower.contains("flux-1")
        || id_lower.contains("/flux")
    {
        Family::Flux
    } else if id_lower.contains("ltx-video-2")
        || id_lower.contains("ltx-2")
        || id_lower.contains("ltx2")
    {
        Family::Ltx2
    } else if id_lower.contains("ltx-video") {
        Family::LtxVideo
    } else if id_lower.contains("z-image") || id_lower.contains("zimage") {
        Family::ZImage
    } else if id_lower.contains("qwen-image") || id_lower.contains("qwen_image") {
        Family::QwenImage
    } else if id_lower.contains("wuerstchen") {
        Family::Wuerstchen
    } else if id_lower.contains("stable-diffusion-xl") || id_lower.contains("sdxl") {
        Family::Sdxl
    } else if id_lower.contains("stable-diffusion-v1")
        || id_lower.contains("sd-v1")
        || id_lower.contains("/sd1")
    {
        Family::Sd15
    } else {
        // Tag-based fallback for anything that doesn't match the id heuristic.
        let mut matched: Option<Family> = None;
        for tag in tags {
            let t = tag.to_ascii_lowercase();
            matched = match t.as_str() {
                "flux" => Some(Family::Flux),
                "flux.2" | "flux2" => Some(Family::Flux2),
                "stable-diffusion-xl" | "sdxl" => Some(Family::Sdxl),
                "stable-diffusion" => Some(Family::Sd15),
                _ => continue,
            };
            break;
        }
        matched?
    };

    Some((family, role_for(repo_id, family)))
}

// ── Single-id lookups ───────────────────────────────────────────────────────

/// Civitai's `/api/v1/model-versions/{id}` response. Differs from the
/// search shape: each version carries a top-level `model` field with
/// the parent's metadata, instead of being nested inside it.
#[derive(Clone, Debug, Deserialize)]
struct CivitaiVersionDetail {
    #[serde(flatten)]
    version: CivitaiVersion,
    model: CivitaiVersionModel,
}

#[derive(Clone, Debug, Deserialize)]
struct CivitaiVersionModel {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    nsfw: bool,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    creator: Option<crate::normalizer::CivitaiCreator>,
    #[serde(default)]
    stats: Option<crate::normalizer::CivitaiStats>,
}

/// Fetch and normalize a single Civitai model-version. Stats/creator
/// are lossy when the upstream omits the embedded `model` block — the
/// resulting entry still has a usable recipe, which is what download
/// flow needs.
pub async fn fetch_civitai_version(
    civitai_base: &str,
    version_id: &str,
    civitai_token: Option<&str>,
) -> Result<CatalogEntry, LiveSearchError> {
    let url = format!("{civitai_base}/api/v1/model-versions/{version_id}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;
    tracing::debug!(target: "catalog.live", url = %url, "civitai version fetch");
    let mut req = client.get(&url);
    if let Some(t) = civitai_token {
        req = req.bearer_auth(t);
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
    let detail: CivitaiVersionDetail = serde_json::from_str(&body)?;

    let item = CivitaiItem {
        id: 0,
        name: detail.model.name,
        kind: detail.model.kind,
        nsfw: detail.model.nsfw,
        creator: detail.model.creator,
        stats: detail.model.stats,
        tags: detail.model.tags,
        model_versions: vec![detail.version],
    };
    from_civitai(item).ok_or_else(|| LiveSearchError::Upstream {
        host: "civitai.com",
        status: 422,
        body: format!(
            "model-version {version_id} did not normalize (unsupported kind, missing safetensors, or unknown baseModel)"
        ),
    })
}

/// HF detail+tree for a single repo, normalized to a [`CatalogEntry`]
/// with a fully-resolved recipe. Used by both `hf:<repo>` resolution
/// and the on-click download flow when the search summary's recipe
/// was empty.
pub async fn fetch_hf_repo(
    hf_base: &str,
    repo_id: &str,
    hf_token: Option<&str>,
) -> Result<CatalogEntry, LiveSearchError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let detail_url = format!("{hf_base}/api/models/{repo_id}");
    let tree_url = format!("{hf_base}/api/models/{repo_id}/tree/main?recursive=1");

    let mut detail_req = client.get(&detail_url);
    let mut tree_req = client.get(&tree_url);
    if let Some(t) = hf_token {
        detail_req = detail_req.bearer_auth(t);
        tree_req = tree_req.bearer_auth(t);
    }

    let (detail_resp, tree_resp) = tokio::try_join!(detail_req.send(), tree_req.send())?;

    let detail_status = detail_resp.status();
    let detail_body = detail_resp.text().await?;
    if !detail_status.is_success() {
        return Err(LiveSearchError::Upstream {
            host: "huggingface.co",
            status: detail_status.as_u16(),
            body: detail_body.chars().take(400).collect(),
        });
    }
    let detail: HfDetail = serde_json::from_str(&detail_body)?;

    let tree_status = tree_resp.status();
    let tree_body = tree_resp.text().await?;
    if !tree_status.is_success() {
        return Err(LiveSearchError::Upstream {
            host: "huggingface.co",
            status: tree_status.as_u16(),
            body: tree_body.chars().take(400).collect(),
        });
    }
    let tree: Vec<HfTreeEntry> = serde_json::from_str(&tree_body)?;

    let (family, family_role) =
        family_from_hf(&detail.id, &detail.tags, detail.pipeline_tag.as_deref()).ok_or_else(
            || LiveSearchError::Upstream {
                host: "huggingface.co",
                status: 422,
                body: format!("repo {repo_id} doesn't map to a supported mold family"),
            },
        )?;

    from_hf(detail, tree, family, family_role).map_err(|e| LiveSearchError::Upstream {
        host: "huggingface.co",
        status: 422,
        body: e.to_string(),
    })
}
