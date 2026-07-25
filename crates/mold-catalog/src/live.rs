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
//! The cache is bounded and per-source — emitted Civitai pages, emitted
//! HF pages, and in-flight Civitai cursor chains each live in their own
//! TTL map. Past `max_entries`, the oldest entry is evicted on each
//! insert. There is no LRU heat-tracking because the TTL is short
//! (5 minutes by default) and the working set is small.
//!
//! Civitai's `/api/v1/models` is cursor-paginated (`page=` is ignored),
//! so deep pages are served by walking a per-query cursor chain forward
//! one `page_size` window at a time, buffering locally-filtered leftover
//! rows so every emitted page is full-length. Both upstreams are fetched
//! concurrently in merged mode.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::civitai_map::{map_base_model, CIVITAI_BASE_MODELS};
use crate::companions::{Companion, COMPANIONS};
use crate::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags,
    Modality, RecipeFile, Source,
};
use crate::families::Family;
use crate::normalizer::{
    from_civitai, from_civitai_search_entries, from_hf, CivitaiItem, CivitaiVersion, HfDetail,
    HfTreeEntry, HF_RAW,
};

/// Result ordering requested from the upstream catalogs. Each variant
/// maps to a native upstream sort so ordering happens server-side at the
/// source — Civitai: `Most Downloaded` / `Newest` / `Highest Rated`;
/// Hugging Face: `downloads` / `lastModified` / `likes` (all descending
/// via `direction=-1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CatalogSort {
    #[default]
    Downloads,
    Recent,
    Rating,
}

impl CatalogSort {
    /// Wire vocabulary accepted by `GET /api/catalog/search`'s `sort=`
    /// parameter, in the order clients should present it. Also what
    /// `GET /api/capabilities` advertises for feature detection.
    pub const WIRE_VALUES: [&'static str; 3] = ["downloads", "recent", "rating"];

    /// Parse the caller-facing wire value. `None` for anything outside
    /// [`Self::WIRE_VALUES`] — callers decide whether that's a 422.
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "downloads" => Some(Self::Downloads),
            "recent" => Some(Self::Recent),
            "rating" => Some(Self::Rating),
            _ => None,
        }
    }

    /// Civitai `/api/v1/models` `sort=` value.
    pub fn civitai_value(self) -> &'static str {
        match self {
            Self::Downloads => "Most Downloaded",
            Self::Recent => "Newest",
            Self::Rating => "Highest Rated",
        }
    }

    /// Hugging Face `/api/models` `sort=` value (paired with
    /// `direction=-1` for descending order).
    pub fn hf_value(self) -> &'static str {
        match self {
            Self::Downloads => "downloads",
            Self::Recent => "lastModified",
            Self::Rating => "likes",
        }
    }
}

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
    /// `Lora` / `Checkpoint` etc. — translated to upstream filters where
    /// possible. `Clip` / `Tokenizer` are local companion-component rows;
    /// VAE and text-encoder components come from the same local registry
    /// and can also coexist with upstream rows where an upstream has them.
    pub kind: Option<Kind>,
    /// `None` queries Civitai + HF + local component rows, while an
    /// explicit source restricts to that source family.
    pub source: Option<Source>,
    pub page: u32,
    pub page_size: u32,
    /// When false, NSFW rows are filtered out post-fetch. Civitai's
    /// `nsfw=false` query param is not 100% reliable on its own.
    pub include_nsfw: bool,
    /// Result ordering, forwarded to each upstream's native sort
    /// parameter. Part of the cache key — two sorts of the same query
    /// must never share a cached result vector.
    pub sort: CatalogSort,
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
            sort: CatalogSort::default(),
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

/// Bounded FIFO map with a per-entry TTL. Expired entries are dropped on
/// lookup; past `max_entries`, the oldest insertion is evicted. There is
/// no LRU heat-tracking because the TTL is short and the working set is
/// small.
struct TtlMap<K, V> {
    map: HashMap<K, TtlValue<V>>,
    /// Insertion order — oldest at the front. Used to evict on overflow.
    order: Vec<K>,
    ttl: Duration,
    max_entries: usize,
}

struct TtlValue<V> {
    stored_at: Instant,
    value: V,
}

impl<K: Clone + Eq + std::hash::Hash, V: Clone> TtlMap<K, V> {
    fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: Vec::new(),
            ttl,
            max_entries: max_entries.max(1),
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        let now = Instant::now();
        if let Some(v) = self.map.get(key) {
            if now.duration_since(v.stored_at) <= self.ttl {
                return Some(v.value.clone());
            }
        }
        // Expired or missing — drop any stale entry so the next pass
        // doesn't re-check it.
        if self.map.remove(key).is_some() {
            self.order.retain(|k| k != key);
        }
        None
    }

    fn put(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            self.order.retain(|k| k != &key);
        }
        self.order.push(key.clone());
        self.map.insert(
            key,
            TtlValue {
                stored_at: Instant::now(),
                value,
            },
        );
        while self.order.len() > self.max_entries {
            let evict = self.order.remove(0);
            self.map.remove(&evict);
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

/// Per-source key for one *chain* of Civitai cursor windows (or, with
/// [`SourcePageKey`], one emitted page of either upstream). Any change to
/// query/filter/sort/token/page-size lands on a different key, which is
/// what resets a cursor chain when the user changes filters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SourceChainKey {
    q: Option<String>,
    family: Option<Family>,
    kind: Option<Kind>,
    include_nsfw: bool,
    sort: CatalogSort,
    token: Option<String>,
    /// Per-source page size (post split across sources), not the
    /// caller-visible one.
    page_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SourcePageKey {
    chain: SourceChainKey,
    page: u32,
}

fn chain_key(opts: &LiveSearchOpts, token: Option<&String>) -> SourceChainKey {
    SourceChainKey {
        q: opts.q.clone(),
        family: opts.family,
        kind: opts.kind,
        include_nsfw: opts.include_nsfw,
        sort: opts.sort,
        token: token.cloned(),
        page_size: opts.page_size,
    }
}

/// Bounded TTL cache backing [`search_page`]. Three private maps: emitted
/// Civitai pages, emitted HF pages, and in-flight Civitai cursor chains.
/// Caching per source (rather than per merged result) lets a page fetched
/// during a forward cursor walk satisfy later requests for that page, and
/// keeps a Civitai chain's progress across merged/soloed queries.
#[derive(Clone)]
pub struct LiveCache {
    civitai_pages: Arc<Mutex<TtlMap<SourcePageKey, LiveSearchResult>>>,
    hf_pages: Arc<Mutex<TtlMap<SourcePageKey, LiveSearchResult>>>,
    civitai_chains: Arc<Mutex<TtlMap<SourceChainKey, CivitaiChain>>>,
}

impl LiveCache {
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            civitai_pages: Arc::new(Mutex::new(TtlMap::new(ttl, max_entries))),
            hf_pages: Arc::new(Mutex::new(TtlMap::new(ttl, max_entries))),
            civitai_chains: Arc::new(Mutex::new(TtlMap::new(ttl, max_entries))),
        }
    }

    pub fn len(&self) -> usize {
        self.civitai_pages.lock().expect("cache mutex").len()
            + self.hf_pages.lock().expect("cache mutex").len()
            + self.civitai_chains.lock().expect("cache mutex").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_civitai_page(&self, key: &SourcePageKey) -> Option<LiveSearchResult> {
        self.civitai_pages.lock().expect("cache mutex").get(key)
    }

    fn put_civitai_page(&self, key: SourcePageKey, result: LiveSearchResult) {
        self.civitai_pages
            .lock()
            .expect("cache mutex")
            .put(key, result);
    }

    fn get_hf_page(&self, key: &SourcePageKey) -> Option<LiveSearchResult> {
        self.hf_pages.lock().expect("cache mutex").get(key)
    }

    fn put_hf_page(&self, key: SourcePageKey, result: LiveSearchResult) {
        self.hf_pages.lock().expect("cache mutex").put(key, result);
    }

    fn get_chain(&self, key: &SourceChainKey) -> Option<CivitaiChain> {
        self.civitai_chains.lock().expect("cache mutex").get(key)
    }

    fn put_chain(&self, key: SourceChainKey, chain: CivitaiChain) {
        self.civitai_chains
            .lock()
            .expect("cache mutex")
            .put(key, chain);
    }
}

/// One Civitai cursor chain: where the next window fetch resumes and what
/// filtered rows are buffered ahead of the next emitted page. Cursors only
/// move forward — a request for an earlier, uncached page restarts the
/// chain and re-walks (re-populating the page cache along the way).
#[derive(Clone, Debug)]
struct CivitaiChain {
    /// Cursor for the next window fetch; `None` before the first fetch
    /// and after exhaustion.
    next_cursor: Option<String>,
    /// Upstream signalled the end of the feed (no `nextCursor` or an
    /// empty window).
    exhausted: bool,
    /// Next page (1-based) this chain will emit into the page cache.
    next_page: u32,
    /// Filtered rows fetched but not yet emitted — keeps every emitted
    /// page full-length even when local filters drop upstream rows.
    leftover: Vec<CatalogEntry>,
    /// Filtered rows already emitted into cached pages.
    emitted: usize,
    /// Largest `totalItems` the upstream has reported (0 in cursor mode).
    reported_total: usize,
}

impl CivitaiChain {
    fn new() -> Self {
        Self {
            next_cursor: None,
            exhausted: false,
            next_page: 1,
            leftover: Vec::new(),
            emitted: 0,
            reported_total: 0,
        }
    }

    /// Complete-feed estimate: the reported total when the upstream gives
    /// one, otherwise the rows seen so far — plus one phantom row while
    /// the cursor is still open so `hasMore` stays true until the feed is
    /// genuinely exhausted. Exact at exhaustion.
    fn total(&self) -> usize {
        self.reported_total
            .max(self.emitted + self.leftover.len() + usize::from(!self.exhausted))
    }
}

#[derive(Clone, Debug, Deserialize)]
struct CivitaiResponse {
    #[serde(default)]
    items: Vec<CivitaiItem>,
    #[serde(default)]
    metadata: CivitaiMetadata,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct CivitaiMetadata {
    /// Absent on cursor-mode responses; 0 then, and [`CivitaiChain::total`]
    /// supplies the observed floor instead.
    #[serde(default, rename = "totalItems")]
    total_items: usize,
    /// Cursor for the next window. Documented as string-or-number, so
    /// deserialize leniently.
    #[serde(default, rename = "nextCursor", deserialize_with = "de_cursor")]
    next_cursor: Option<String>,
}

fn de_cursor<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| match value {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }))
}

#[derive(Clone, Debug)]
pub struct LiveSearchResult {
    pub entries: Vec<CatalogEntry>,
    /// Complete filtered row count when the upstream exposes it; otherwise a
    /// conservative lower bound that keeps pagination available.
    pub total: usize,
}

/// Backwards-compatible one-shot query for Rust consumers that only need the
/// rows. The server uses [`search_page`] so it can preserve total-count
/// metadata on the HTTP wire.
pub async fn search(
    civitai_base: &str,
    hf_base: &str,
    cache: &LiveCache,
    opts: &LiveSearchOpts,
) -> Result<Vec<CatalogEntry>, LiveSearchError> {
    Ok(search_page(civitai_base, hf_base, cache, opts)
        .await?
        .entries)
}

/// One-shot live query with pagination metadata. Returns normalized
/// [`CatalogEntry`] rows suitable for direct serialization into the
/// `/api/catalog/search` wire format.
///
/// Source routing:
/// - `Some(Civitai)` → Civitai only
/// - `Some(Hf)`      → HF only
/// - `None`          → both (concatenated, Civitai first)
pub async fn search_page(
    civitai_base: &str,
    hf_base: &str,
    cache: &LiveCache,
    opts: &LiveSearchOpts,
) -> Result<LiveSearchResult, LiveSearchError> {
    let local_components = companion_component_entries(opts);
    if matches!(opts.kind, Some(Kind::Clip | Kind::Tokenizer)) {
        return Ok(paginate_local(local_components, opts));
    }
    let source_count =
        if opts.source.is_none() { 2 } else { 1 } + u32::from(!local_components.is_empty());
    let mut upstream_opts = opts.clone();
    upstream_opts.page_size = opts.page_size.div_ceil(source_count);

    let want_civitai = !matches!(opts.source, Some(Source::Hf));
    let want_hf = !matches!(opts.source, Some(Source::Civitai));
    let civitai_fut = async {
        if want_civitai {
            Some(civitai_search_paged(civitai_base, cache, &upstream_opts).await)
        } else {
            None
        }
    };
    let hf_fut = async {
        if want_hf {
            Some(hf_search_paged(hf_base, cache, &upstream_opts).await)
        } else {
            None
        }
    };
    let (civitai_result, hf_result) = tokio::join!(civitai_fut, hf_fut);

    let mut entries = Vec::new();
    let mut total = 0;
    if let Some(page) = civitai_result.transpose()? {
        total += page.total;
        entries.extend(page.entries);
    }
    if let Some(result) = hf_result {
        // Authentication errors must surface so the server can retry with its
        // next credential candidate; other combined-search failures stay soft.
        match result {
            Ok(page) => {
                total += page.total;
                entries.extend(page.entries);
            }
            Err(
                e @ LiveSearchError::Upstream {
                    status: 401 | 403, ..
                },
            ) => return Err(e),
            Err(e) if matches!(opts.source, Some(Source::Hf)) => return Err(e),
            Err(e) => {
                tracing::warn!(target: "catalog.live", error = %e, "hf search failed");
            }
        }
    }
    if !local_components.is_empty() {
        let local_page = paginate_local(local_components, &upstream_opts);
        total += local_page.total;
        entries.extend(local_page.entries);
    }
    entries.truncate(opts.page_size as usize);

    Ok(LiveSearchResult { entries, total })
}

fn paginate_local(entries: Vec<CatalogEntry>, opts: &LiveSearchOpts) -> LiveSearchResult {
    let total = entries.len();
    let start = page_offset(opts);
    let entries = entries
        .into_iter()
        .skip(start)
        .take(opts.page_size as usize)
        .collect();
    LiveSearchResult { entries, total }
}

fn page_offset(opts: &LiveSearchOpts) -> usize {
    let offset = u64::from(opts.page.saturating_sub(1)).saturating_mul(u64::from(opts.page_size));
    usize::try_from(offset).unwrap_or(usize::MAX)
}

pub fn companion_entry_for_id(id: &str) -> Option<CatalogEntry> {
    let rest = id.strip_prefix("hf:companion/")?;
    let (name, tokenizer) = match rest.strip_suffix("/tokenizer") {
        Some(name) => (name, true),
        None => (rest, false),
    };
    let companion = COMPANIONS.iter().find(|c| c.canonical_name == name)?;
    if tokenizer && !companion_has_tokenizer(companion) {
        return None;
    }
    let family = companion.family_scope.first().copied()?;
    Some(companion_to_entry(companion, family, tokenizer))
}

pub fn companion_name_for_catalog_id(id: &str) -> Option<&'static str> {
    let rest = id.strip_prefix("hf:companion/")?;
    let name = rest.strip_suffix("/tokenizer").unwrap_or(rest);
    COMPANIONS
        .iter()
        .find(|c| c.canonical_name == name)
        .map(|c| c.canonical_name)
}

fn companion_component_entries(opts: &LiveSearchOpts) -> Vec<CatalogEntry> {
    if matches!(opts.source, Some(Source::Civitai)) {
        return Vec::new();
    }
    if opts.kind.is_none() && !query_mentions_component(opts.q.as_deref()) {
        return Vec::new();
    }

    let mut entries = Vec::new();
    for companion in COMPANIONS {
        let Some(family) = companion_family_for_filter(companion, opts.family) else {
            continue;
        };

        let include_bundle = match opts.kind {
            Some(Kind::Clip) => companion_kind(companion) == Kind::Clip,
            Some(Kind::TextEncoder) => companion_kind(companion) == Kind::TextEncoder,
            Some(Kind::Vae) => companion_kind(companion) == Kind::Vae,
            Some(Kind::Tokenizer) => false,
            Some(Kind::Checkpoint | Kind::Lora | Kind::ControlNet) => false,
            None => true,
        };
        if include_bundle {
            let entry = companion_to_entry(companion, family, false);
            if companion_matches_query(&entry, opts.q.as_deref()) {
                entries.push(entry);
            }
        }

        if companion_has_tokenizer(companion) && matches!(opts.kind, None | Some(Kind::Tokenizer)) {
            let entry = companion_to_entry(companion, family, true);
            if companion_matches_query(&entry, opts.q.as_deref()) {
                entries.push(entry);
            }
        }
    }
    entries
}

fn query_mentions_component(q: Option<&str>) -> bool {
    let Some(q) = q.map(str::trim).filter(|s| !s.is_empty()) else {
        return false;
    };
    let q = q.to_ascii_lowercase();
    [
        "clip",
        "tokenizer",
        "vae",
        "text encoder",
        "text-encoder",
        "t5",
        "qwen",
        "gemma",
    ]
    .iter()
    .any(|needle| q.contains(needle))
}

fn companion_family_for_filter(companion: &Companion, filter: Option<Family>) -> Option<Family> {
    match filter {
        Some(family) if companion.family_scope.contains(&family) => Some(family),
        Some(_) => None,
        None => companion.family_scope.first().copied(),
    }
}

fn companion_kind(companion: &Companion) -> Kind {
    if companion.canonical_name.starts_with("clip-") {
        Kind::Clip
    } else {
        companion.kind
    }
}

fn companion_has_tokenizer(companion: &Companion) -> bool {
    companion
        .files
        .iter()
        .any(|path| path.to_ascii_lowercase().contains("tokenizer"))
}

fn companion_matches_query(entry: &CatalogEntry, q: Option<&str>) -> bool {
    let Some(q) = q.map(str::trim).filter(|s| !s.is_empty()) else {
        return true;
    };
    let q = q.to_ascii_lowercase();
    entry.name.to_ascii_lowercase().contains(&q)
        || entry.source_id.to_ascii_lowercase().contains(&q)
        || entry
            .description
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .contains(&q)
        || entry
            .tags
            .iter()
            .any(|tag| tag.to_ascii_lowercase().contains(&q))
}

fn companion_to_entry(companion: &Companion, family: Family, tokenizer: bool) -> CatalogEntry {
    let kind = if tokenizer {
        Kind::Tokenizer
    } else {
        companion_kind(companion)
    };
    let name = if tokenizer {
        format!("{} tokenizer", companion.canonical_name)
    } else {
        companion.canonical_name.to_string()
    };
    let id = if tokenizer {
        format!("hf:companion/{}/tokenizer", companion.canonical_name)
    } else {
        format!("hf:companion/{}", companion.canonical_name)
    };
    let files = companion
        .files
        .iter()
        .map(|file| RecipeFile {
            url: format!("{HF_RAW}/{}/resolve/main/{file}", companion.repo),
            dest: format!("{{family}}/{}/{file}", companion.canonical_name),
            sha256: None,
            size_bytes: None,
            role: None,
        })
        .collect::<Vec<_>>();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    CatalogEntry {
        id: CatalogId::from(id),
        source: Source::Hf,
        source_id: companion.repo.to_string(),
        name,
        author: companion.repo.split('/').next().map(str::to_string),
        family,
        family_role: FamilyRole::Foundation,
        sub_family: None,
        modality: match family {
            Family::LtxVideo | Family::Ltx2 => Modality::Video,
            _ => Modality::Image,
        },
        kind,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: Some(companion.size_bytes),
        download_count: 0,
        rating: None,
        likes: 0,
        nsfw: false,
        thumbnail_url: None,
        description: Some(format!(
            "{} component bundle from {}",
            companion.canonical_name, companion.repo
        )),
        license: None,
        license_flags: LicenseFlags::default(),
        tags: companion
            .family_scope
            .iter()
            .map(|family| family.as_str().to_string())
            .chain(std::iter::once(kind_tag(kind).to_string()))
            .collect(),
        companions: Vec::new(),
        download_recipe: DownloadRecipe {
            files,
            needs_token: None,
        },
        supported: true,
        created_at: None,
        updated_at: None,
        added_at: now,
        trained_words: Vec::new(),
        // Component bundles are curated slices of a real HF repo — link
        // to that repo, same as any other HF-sourced row.
        page_url: Some(format!("{HF_RAW}/{}", companion.repo)),
    }
}

fn kind_tag(kind: Kind) -> &'static str {
    match kind {
        Kind::Checkpoint => "checkpoint",
        Kind::Lora => "lora",
        Kind::Vae => "vae",
        Kind::TextEncoder => "text-encoder",
        Kind::Tokenizer => "tokenizer",
        Kind::Clip => "clip",
        Kind::ControlNet => "control-net",
    }
}

/// Cap on upstream window fetches a single [`civitai_search_paged`] call
/// may issue. A deep forward walk (cache expired, client re-requests page
/// N) costs about one fetch per page plus leftover top-ups; past this
/// budget the request returns short and the persisted chain resumes on
/// the next call.
const MAX_WINDOW_FETCHES: usize = 12;

/// One fetched-and-filtered Civitai cursor window.
struct CivitaiWindow {
    /// Rows that survived the local nsfw/family/kind re-filtering.
    entries: Vec<CatalogEntry>,
    /// Raw upstream item count, pre-filtering — an empty raw window means
    /// the feed is done regardless of what filtering kept.
    raw_count: usize,
    next_cursor: Option<String>,
    total_items: usize,
}

/// Serve one page of Civitai results via true cursor pagination.
///
/// Page-cache hit fast path; otherwise the per-chain cursor state is
/// advanced window-by-window, buffering filtered leftovers so every
/// emitted page is a full `page_size` (short mid-feed pages kill infinite
/// scroll). Pages emitted during the walk are cached individually, so a
/// deep jump re-populates every page on the way. Cursors cannot rewind:
/// a request for an earlier, uncached page restarts the chain.
async fn civitai_search_paged(
    base: &str,
    cache: &LiveCache,
    opts: &LiveSearchOpts,
) -> Result<LiveSearchResult, LiveSearchError> {
    let key = chain_key(opts, opts.civitai_token.as_ref());
    let page_key = SourcePageKey {
        chain: key.clone(),
        page: opts.page,
    };
    if let Some(hit) = cache.get_civitai_page(&page_key) {
        return Ok(hit);
    }

    // Clone the chain out and put it back after the walk — the cache
    // mutex must never be held across an await.
    let mut chain = cache.get_chain(&key).unwrap_or_else(CivitaiChain::new);
    if opts.page < chain.next_page {
        chain = CivitaiChain::new();
    }

    let page_size = opts.page_size.clamp(1, 100) as usize;
    let mut fetches = 0usize;
    let mut result = None;
    while chain.next_page <= opts.page {
        while chain.leftover.len() < page_size && !chain.exhausted && fetches < MAX_WINDOW_FETCHES {
            let window = civitai_fetch_window(base, opts, chain.next_cursor.as_deref()).await?;
            fetches += 1;
            chain.reported_total = chain.reported_total.max(window.total_items);
            chain.leftover.extend(window.entries);
            if window.raw_count == 0 || window.next_cursor.is_none() {
                chain.exhausted = true;
                chain.next_cursor = None;
            } else {
                chain.next_cursor = window.next_cursor;
            }
        }
        if !chain.exhausted && chain.leftover.len() < page_size {
            // Fetch budget spent mid-walk. When the shortfall lands on the
            // requested page, emit the buffered rows rather than a false
            // empty page — clients treat an empty page as exhaustion and
            // would skip them. Intermediate pages are never emitted short:
            // the walk resumes from the stored cursor on the next request
            // and `total()` keeps advertising more rows either way.
            if chain.next_page == opts.page && !chain.leftover.is_empty() {
                let entries: Vec<CatalogEntry> = chain.leftover.drain(..).collect();
                chain.emitted += entries.len();
                let page_result = LiveSearchResult {
                    entries,
                    total: chain.total(),
                };
                cache.put_civitai_page(
                    SourcePageKey {
                        chain: key.clone(),
                        page: chain.next_page,
                    },
                    page_result.clone(),
                );
                chain.next_page += 1;
                result = Some(page_result);
            }
            break;
        }
        let take = page_size.min(chain.leftover.len());
        let entries: Vec<CatalogEntry> = chain.leftover.drain(..take).collect();
        chain.emitted += entries.len();
        let page_result = LiveSearchResult {
            entries,
            total: chain.total(),
        };
        cache.put_civitai_page(
            SourcePageKey {
                chain: key.clone(),
                page: chain.next_page,
            },
            page_result.clone(),
        );
        if chain.next_page == opts.page {
            result = Some(page_result);
        }
        chain.next_page += 1;
        if chain.exhausted && chain.leftover.is_empty() && result.is_none() {
            // Every remaining page is empty — answer directly instead of
            // caching arbitrarily deep empty pages one by one.
            result = Some(LiveSearchResult {
                entries: Vec::new(),
                total: chain.total(),
            });
            break;
        }
    }
    let total = chain.total();
    cache.put_chain(key, chain);
    Ok(result.unwrap_or(LiveSearchResult {
        entries: Vec::new(),
        total,
    }))
}

/// Fetch one Civitai cursor window: `limit = page_size`, `cursor=` when
/// resuming, never `page=` (ignored upstream, rejected alongside
/// `query=`). Rows are re-filtered locally for nsfw/family/kind drift.
async fn civitai_fetch_window(
    base: &str,
    opts: &LiveSearchOpts,
    cursor: Option<&str>,
) -> Result<CivitaiWindow, LiveSearchError> {
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
        let limit = opts.page_size.clamp(1, 100);
        q.append_pair("limit", &limit.to_string());
        if let Some(cursor) = cursor {
            q.append_pair("cursor", cursor);
        }
        q.append_pair("sort", opts.sort.civitai_value());
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
            Some(Kind::ControlNet) => {
                q.append_pair("types", "Controlnet");
            }
            // Vae / TextEncoder aren't searchable on Civitai (or are not
            // yet first-class in mold), so we fall through to the default —
            // both Checkpoints and LoRAs.
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

    let metadata = parsed.metadata;
    let raw_count = parsed.items.len();
    let mut out = Vec::new();
    for item in parsed.items {
        let nsfw_item = item.nsfw;
        for entry in from_civitai_search_entries(item) {
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
    Ok(CivitaiWindow {
        entries: out,
        raw_count,
        next_cursor: metadata.next_cursor,
        total_items: metadata.total_items,
    })
}

// ── HF live search ──────────────────────────────────────────────────────────

/// Per-page cache wrapper around [`hf_search`]. HF keeps the offset
/// window strategy (real offsets, clamped at 1000) — only the caching is
/// per (query, page) like the Civitai side.
async fn hf_search_paged(
    base: &str,
    cache: &LiveCache,
    opts: &LiveSearchOpts,
) -> Result<LiveSearchResult, LiveSearchError> {
    let key = SourcePageKey {
        chain: chain_key(opts, opts.hf_token.as_ref()),
        page: opts.page,
    };
    if let Some(hit) = cache.get_hf_page(&key) {
        return Ok(hit);
    }
    let result = hf_search(base, opts).await?;
    cache.put_hf_page(key, result.clone());
    Ok(result)
}

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

async fn hf_search(base: &str, opts: &LiveSearchOpts) -> Result<LiveSearchResult, LiveSearchError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    let mut url =
        reqwest::Url::parse(&format!("{base}/api/models")).expect("hf base URL must be valid");
    let trimmed_q = opts.q.as_deref().map(str::trim).filter(|s| !s.is_empty());
    {
        let mut q = url.query_pairs_mut();
        let window = opts.page.saturating_mul(opts.page_size).clamp(1, 1000);
        q.append_pair("limit", &window.to_string());
        q.append_pair("sort", opts.sort.hf_value());
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
    let fetched = hits.len();

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
    let filtered = out.len();
    let start = page_offset(opts);
    let entries = out
        .into_iter()
        .skip(start)
        .take(opts.page_size as usize)
        .collect::<Vec<_>>();
    let requested = opts.page.saturating_mul(opts.page_size) as usize;
    let total = if fetched < requested {
        filtered
    } else {
        start
            .saturating_add(entries.len())
            .saturating_add(usize::from(!entries.is_empty()))
    };
    Ok(LiveSearchResult { entries, total })
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
        supported: crate::civitai_map::supported_for(family, Bundling::Separated, kind),
        created_at: parse_iso(hit.created_at.as_deref()),
        updated_at: parse_iso(hit.last_modified.as_deref()),
        added_at: now,
        trained_words: Vec::new(),
        page_url: Some(format!("{HF_RAW}/{}", hit.id)),
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
    /// Parent model id — present on the live `/api/v1/model-versions/{id}`
    /// response but absent from older cached bodies and fixtures, so it
    /// must stay optional. When present it lets the normalized entry
    /// carry a model page URL.
    #[serde(default, rename = "modelId")]
    model_id: Option<u64>,
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
        // `from_civitai` composes the model page URL from this id and
        // skips it when 0 — the sentinel for "upstream omitted modelId".
        id: detail.model_id.unwrap_or(0),
        name: detail.model.name,
        kind: detail.model.kind,
        nsfw: detail.model.nsfw,
        description: None,
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

/// Fetch a single catalog entry by its `cv:` / `hf:` id, dispatching to
/// the Civitai model-version or HF detail+tree fetcher as appropriate.
///
/// Deduplicates the near-identical prefix dispatch that previously lived
/// in both `mold-cli`'s catalog bridge and `mold-server`'s model manager.
/// Bases and tokens are passed explicitly so each caller keeps control of
/// its own upstream endpoints (the server pins a test-overridable Civitai
/// base; the CLI honors `CIVITAI_BASE` / `HF_BASE` env).
pub async fn fetch_entry_by_id(
    id: &str,
    civitai_base: &str,
    hf_base: &str,
    civitai_token: Option<&str>,
    hf_token: Option<&str>,
) -> Result<CatalogEntry, LiveSearchError> {
    if let Some(version_id) = id.strip_prefix("cv:") {
        fetch_civitai_version(civitai_base, version_id, civitai_token).await
    } else if let Some(repo_id) = id.strip_prefix("hf:") {
        fetch_hf_repo(hf_base, repo_id, hf_token).await
    } else {
        Err(LiveSearchError::Upstream {
            host: "catalog",
            status: 400,
            body: format!("'{id}' is not a catalog id (expected a cv: or hf: prefix)"),
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path, query_param, query_param_is_missing};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_cache() -> LiveCache {
        LiveCache::new(Duration::from_secs(300), 256)
    }

    fn civitai_opts(page: u32, page_size: u32) -> LiveSearchOpts {
        LiveSearchOpts {
            page,
            page_size,
            source: Some(Source::Civitai),
            ..Default::default()
        }
    }

    fn civitai_item(version_id: u64, nsfw: bool) -> serde_json::Value {
        serde_json::json!({
            "id": 100_000 + version_id,
            "name": format!("Model {version_id}"),
            "type": "Checkpoint",
            "nsfw": nsfw,
            "creator": { "username": "alice" },
            "stats": { "downloadCount": 100 },
            "tags": [],
            "modelVersions": [{
                "id": version_id,
                "name": "v1",
                "baseModel": "Flux.1 D",
                "baseModelType": "Standard",
                "files": [{
                    "id": version_id, "name": "x.safetensors", "sizeKB": 100,
                    "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                    "downloadUrl": "https://civitai.example/x.safetensors",
                    "hashes": { "SHA256": "deadbeef" }
                }],
                "images": []
            }]
        })
    }

    fn civitai_body(
        version_ids: std::ops::RangeInclusive<u64>,
        next_cursor: Option<&str>,
    ) -> serde_json::Value {
        let items = version_ids
            .map(|id| civitai_item(id, false))
            .collect::<Vec<_>>();
        let metadata = match next_cursor {
            Some(cursor) => serde_json::json!({ "nextCursor": cursor }),
            None => serde_json::json!({}),
        };
        serde_json::json!({ "items": items, "metadata": metadata })
    }

    fn hf_hit(index: u64) -> serde_json::Value {
        serde_json::json!({
            "id": format!("black-forest-labs/FLUX.1-dev-{index}"),
            "tags": ["diffusers", "flux"],
            "pipeline_tag": "text-to-image"
        })
    }

    fn ids(result: &LiveSearchResult) -> Vec<&str> {
        result
            .entries
            .iter()
            .map(|entry| entry.id.0.as_str())
            .collect()
    }

    async fn request_queries(
        server: &MockServer,
    ) -> Vec<std::collections::HashMap<String, String>> {
        server
            .received_requests()
            .await
            .expect("requests")
            .iter()
            .map(|request| {
                request
                    .url
                    .query_pairs()
                    .map(|(k, v)| (k.into_owned(), v.into_owned()))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn catalog_sort_maps_to_each_upstream_vocabulary() {
        let cases = [
            (CatalogSort::Downloads, "Most Downloaded", "downloads"),
            (CatalogSort::Recent, "Newest", "lastModified"),
            (CatalogSort::Rating, "Highest Rated", "likes"),
        ];
        for (sort, civitai, hf) in cases {
            assert_eq!(sort.civitai_value(), civitai);
            assert_eq!(sort.hf_value(), hf);
        }
    }

    #[test]
    fn catalog_sort_wire_round_trip() {
        assert_eq!(
            CatalogSort::from_wire("downloads"),
            Some(CatalogSort::Downloads)
        );
        assert_eq!(CatalogSort::from_wire("recent"), Some(CatalogSort::Recent));
        assert_eq!(CatalogSort::from_wire("rating"), Some(CatalogSort::Rating));
        // "name" has no upstream mapping and is deliberately NOT accepted;
        // the server turns it into a 422 rather than silently ignoring it.
        assert_eq!(CatalogSort::from_wire("name"), None);
        assert_eq!(CatalogSort::from_wire("Most Downloaded"), None);
        for value in CatalogSort::WIRE_VALUES {
            assert!(CatalogSort::from_wire(value).is_some(), "{value}");
        }
    }

    #[test]
    fn catalog_sort_defaults_to_downloads() {
        assert_eq!(CatalogSort::default(), CatalogSort::Downloads);
        assert_eq!(LiveSearchOpts::default().sort, CatalogSort::Downloads);
    }

    #[test]
    fn page_offset_is_safe_for_zero_and_maximum_caller_values() {
        assert_eq!(page_offset(&LiveSearchOpts::default()), 0);
        assert_eq!(
            page_offset(&LiveSearchOpts {
                page: 0,
                page_size: u32::MAX,
                ..Default::default()
            }),
            0
        );

        let expected = u64::from(u32::MAX - 1).saturating_mul(u64::from(u32::MAX));
        assert_eq!(
            page_offset(&LiveSearchOpts {
                page: u32::MAX,
                page_size: u32::MAX,
                ..Default::default()
            }),
            usize::try_from(expected).unwrap_or(usize::MAX)
        );
    }

    #[tokio::test]
    async fn hf_search_fetches_a_window_and_returns_the_requested_page() {
        let server = MockServer::start().await;
        let hits = (1..=4)
            .map(|index| {
                serde_json::json!({
                    "id": format!("black-forest-labs/FLUX.1-dev-{index}"),
                    "tags": ["diffusers", "flux"],
                    "pipeline_tag": "text-to-image"
                })
            })
            .collect::<Vec<_>>();
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .and(query_param("limit", "4"))
            .respond_with(ResponseTemplate::new(200).set_body_json(hits))
            .mount(&server)
            .await;

        let result = hf_search(
            &server.uri(),
            &LiveSearchOpts {
                page: 2,
                page_size: 2,
                source: Some(Source::Hf),
                ..Default::default()
            },
        )
        .await
        .expect("HF page");

        assert_eq!(result.entries.len(), 2);
        assert!(result.entries[0].id.0.contains("dev-3"));
        assert!(result.entries[1].id.0.contains("dev-4"));
        assert_eq!(result.total, 5, "a full window must advertise another page");
    }

    /// Real Civitai ignores `page=` entirely — `/api/v1/models` is
    /// cursor-paginated. The proxy fetches one `page_size` window at a
    /// time (never `limit = page × page_size`) and buffers leftovers, so
    /// a deep page rides the cursor chain instead of widening the limit.
    #[tokio::test]
    async fn civitai_search_paged_requests_single_page_windows() {
        let server = MockServer::start().await;
        // Like the live API: one window (here over-full) with cursor-mode
        // metadata and no totals.
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(8001..=8004, Some("1|2|3"))),
            )
            .mount(&server)
            .await;

        let result = civitai_search_paged(&server.uri(), &test_cache(), &civitai_opts(2, 2))
            .await
            .expect("civitai page");

        assert_eq!(result.entries.len(), 2, "page 2 must hold fresh rows");
        assert_eq!(ids(&result), ["cv:8003", "cv:8004"]);
        assert_eq!(
            result.total, 5,
            "an open cursor must advertise another page"
        );

        let queries = request_queries(&server).await;
        assert_eq!(queries.len(), 1, "leftovers must cover page 2 (no refetch)");
        assert_eq!(
            queries[0].get("limit").map(String::as_str),
            Some("2"),
            "limit must be one page, not the page × page_size window"
        );
        assert!(
            !queries[0].contains_key("page"),
            "page= is ignored upstream and must not be sent"
        );
    }

    #[tokio::test]
    async fn civitai_pagination_advances_with_the_stored_cursor() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param_is_missing("cursor"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(101..=102, Some("c1"))),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param("cursor", "c1"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(103..=104, Some("c2"))),
            )
            .expect(1)
            .mount(&server)
            .await;

        let cache = test_cache();
        let page1 = civitai_search_paged(&server.uri(), &cache, &civitai_opts(1, 2))
            .await
            .expect("page 1");
        assert_eq!(ids(&page1), ["cv:101", "cv:102"]);

        let page2 = civitai_search_paged(&server.uri(), &cache, &civitai_opts(2, 2))
            .await
            .expect("page 2");
        assert_eq!(ids(&page2), ["cv:103", "cv:104"]);

        let queries = request_queries(&server).await;
        assert_eq!(queries.len(), 2);
        for query in &queries {
            assert!(!query.contains_key("page"), "page= must never be sent");
            assert_eq!(query.get("limit").map(String::as_str), Some("2"));
        }
        assert_eq!(
            queries[1].get("cursor").map(String::as_str),
            Some("c1"),
            "page 2 must ride the cursor from page 1's response"
        );
    }

    #[tokio::test]
    async fn civitai_page_cache_serves_repeats_without_upstream_calls() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(101..=102, Some("c1"))),
            )
            .expect(1)
            .mount(&server)
            .await;

        let cache = test_cache();
        let first = civitai_search_paged(&server.uri(), &cache, &civitai_opts(1, 2))
            .await
            .expect("first");
        let second = civitai_search_paged(&server.uri(), &cache, &civitai_opts(1, 2))
            .await
            .expect("second");

        assert_eq!(ids(&first), ids(&second));
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            1,
            "the repeat must be a page-cache hit"
        );
    }

    #[tokio::test]
    async fn civitai_deep_jump_walks_forward_and_caches_intermediate_pages() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param_is_missing("cursor"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(101..=101, Some("c1"))),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param("cursor", "c1"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(102..=102, Some("c2"))),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param("cursor", "c2"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(103..=103, Some("c3"))),
            )
            .mount(&server)
            .await;

        let cache = test_cache();
        let page3 = civitai_search_paged(&server.uri(), &cache, &civitai_opts(3, 1))
            .await
            .expect("page 3");
        assert_eq!(ids(&page3), ["cv:103"]);
        assert_eq!(server.received_requests().await.expect("requests").len(), 3);

        // The forward walk populated pages 1..3 — page 2 must now be a
        // cache hit even though it was never requested directly.
        let page2 = civitai_search_paged(&server.uri(), &cache, &civitai_opts(2, 1))
            .await
            .expect("page 2");
        assert_eq!(ids(&page2), ["cv:102"]);
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            3,
            "page 2 must come from the walk's page cache"
        );
    }

    #[tokio::test]
    async fn civitai_exhausted_feed_reports_exact_total_and_serves_empty_pages() {
        let server = MockServer::start().await;
        // No nextCursor → the feed ends after these two rows.
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(civitai_body(101..=102, None)))
            .mount(&server)
            .await;

        let cache = test_cache();
        let page1 = civitai_search_paged(&server.uri(), &cache, &civitai_opts(1, 2))
            .await
            .expect("page 1");
        assert_eq!(ids(&page1), ["cv:101", "cv:102"]);
        assert_eq!(page1.total, 2, "exhausted feed must report the exact total");

        let page3 = civitai_search_paged(&server.uri(), &cache, &civitai_opts(3, 2))
            .await
            .expect("page 3");
        assert!(page3.entries.is_empty());
        assert_eq!(page3.total, 2);
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            1,
            "pages past exhaustion must not refetch"
        );
    }

    /// Locally-filtered rows (NSFW here) must not shorten emitted pages —
    /// the chain buffers leftovers and fetches another window until the
    /// page is full, otherwise infinite scroll stalls on a short page.
    #[tokio::test]
    async fn civitai_filtered_rows_do_not_shorten_pages() {
        let server = MockServer::start().await;
        let mut window1 = civitai_body(101..=101, Some("c1"));
        window1["items"]
            .as_array_mut()
            .expect("items")
            .push(civitai_item(999, true));
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param_is_missing("cursor"))
            .respond_with(ResponseTemplate::new(200).set_body_json(window1))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param("cursor", "c1"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(102..=103, Some("c2"))),
            )
            .mount(&server)
            .await;

        let opts = LiveSearchOpts {
            include_nsfw: false,
            ..civitai_opts(1, 2)
        };
        let page1 = civitai_search_paged(&server.uri(), &test_cache(), &opts)
            .await
            .expect("page 1");

        assert_eq!(
            ids(&page1),
            ["cv:101", "cv:102"],
            "the filtered row must be backfilled from the next window"
        );
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            2,
            "a second window fetch must top the page up"
        );
    }

    /// When the per-request window budget runs out before the requested
    /// page fills, the rows already buffered must still be served —
    /// clients treat an empty page as exhaustion and would permanently
    /// skip them.
    #[tokio::test]
    async fn civitai_budget_exhaustion_emits_partial_page_not_empty() {
        let server = MockServer::start().await;
        // Every window yields one row and keeps the cursor open, so a
        // page_size-20 request drains the whole fetch budget.
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(300..=300, Some("c"))),
            )
            .mount(&server)
            .await;

        let cache = test_cache();
        let opts = civitai_opts(1, 20);
        let page1 = civitai_search_paged(&server.uri(), &cache, &opts)
            .await
            .expect("page 1");

        assert_eq!(
            page1.entries.len(),
            MAX_WINDOW_FETCHES,
            "buffered rows must be emitted when the budget expires"
        );
        assert!(
            page1.total > page1.entries.len(),
            "total must keep advertising more rows while the cursor is open"
        );
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            MAX_WINDOW_FETCHES,
            "the walk stops at the fetch budget"
        );

        let again = civitai_search_paged(&server.uri(), &cache, &opts)
            .await
            .expect("page 1 again");
        assert_eq!(again.entries.len(), MAX_WINDOW_FETCHES);
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            MAX_WINDOW_FETCHES,
            "the partial page must be served from the page cache"
        );
    }

    /// The regression this module exists to fix: Civitai caps `limit=` at
    /// 100, so the old `limit = page × page_size` window could never
    /// surface rows past 100. Cursor pagination has no such ceiling.
    #[tokio::test]
    async fn civitai_pages_beyond_row_100_are_reachable() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param_is_missing("cursor"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(1001..=1050, Some("c1"))),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param("cursor", "c1"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(1051..=1100, Some("c2"))),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param("cursor", "c2"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(1101..=1150, Some("c3"))),
            )
            .mount(&server)
            .await;

        let cache = test_cache();
        for page in 1..=2 {
            civitai_search_paged(&server.uri(), &cache, &civitai_opts(page, 50))
                .await
                .expect("shallow page");
        }
        let page3 = civitai_search_paged(&server.uri(), &cache, &civitai_opts(3, 50))
            .await
            .expect("page 3");

        assert_eq!(page3.entries.len(), 50);
        assert_eq!(
            page3.entries[0].id.0, "cv:1101",
            "row 101 must be reachable"
        );
        assert_eq!(page3.entries[49].id.0, "cv:1150");
        let queries = request_queries(&server).await;
        assert!(queries
            .iter()
            .all(|q| q.get("limit").map(String::as_str) == Some("50")));
    }

    #[tokio::test]
    async fn civitai_sort_change_starts_a_fresh_chain() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param("sort", "Most Downloaded"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(101..=102, Some("c1"))),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .and(query_param("sort", "Highest Rated"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(201..=202, Some("r1"))),
            )
            .expect(1)
            .mount(&server)
            .await;

        let cache = test_cache();
        let downloads = civitai_search_paged(&server.uri(), &cache, &civitai_opts(1, 2))
            .await
            .expect("downloads sort");
        assert_eq!(ids(&downloads), ["cv:101", "cv:102"]);

        let rated = civitai_search_paged(
            &server.uri(),
            &cache,
            &LiveSearchOpts {
                sort: CatalogSort::Rating,
                ..civitai_opts(1, 2)
            },
        )
        .await
        .expect("rating sort");
        assert_eq!(
            ids(&rated),
            ["cv:201", "cv:202"],
            "a sort change must key a fresh chain, not replay the old one"
        );
    }

    #[tokio::test]
    async fn merged_search_fetches_civitai_and_hf_in_parallel() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(civitai_body(101..=102, Some("c1")))
                    .set_delay(Duration::from_millis(300)),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(vec![hf_hit(1), hf_hit(2)])
                    .set_delay(Duration::from_millis(300)),
            )
            .mount(&server)
            .await;

        let started = std::time::Instant::now();
        let result = search_page(
            &server.uri(),
            &server.uri(),
            &test_cache(),
            &LiveSearchOpts {
                page_size: 4,
                ..Default::default()
            },
        )
        .await
        .expect("merged page");
        let elapsed = started.elapsed();

        assert!(
            result.entries.iter().any(|e| e.id.0.starts_with("cv:")),
            "civitai rows present"
        );
        assert!(
            result.entries.iter().any(|e| e.id.0.starts_with("hf:")),
            "hf rows present"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "two 300ms upstreams must overlap, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn merged_search_reuses_per_source_page_caches() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(101..=102, Some("c1"))),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vec![hf_hit(1), hf_hit(2)]))
            .expect(1)
            .mount(&server)
            .await;

        let cache = test_cache();
        let opts = LiveSearchOpts {
            page_size: 4,
            ..Default::default()
        };
        let first = search_page(&server.uri(), &server.uri(), &cache, &opts)
            .await
            .expect("first");
        let second = search_page(&server.uri(), &server.uri(), &cache, &opts)
            .await
            .expect("second");

        assert_eq!(ids(&first), ids(&second));
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            2,
            "the repeat must hit both per-source page caches"
        );
    }

    #[tokio::test]
    async fn merged_search_propagates_hf_auth_errors() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(101..=102, Some("c1"))),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .respond_with(ResponseTemplate::new(401).set_body_string("bad token"))
            .mount(&server)
            .await;

        let err = search_page(
            &server.uri(),
            &server.uri(),
            &test_cache(),
            &LiveSearchOpts::default(),
        )
        .await
        .expect_err("401 must propagate for the credential-retry loop");
        match err {
            LiveSearchError::Upstream { host, status, .. } => {
                assert_eq!(host, "huggingface.co");
                assert_eq!(status, 401);
            }
            other => panic!("expected Upstream 401, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn merged_search_soft_fails_hf_server_errors() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(civitai_body(101..=102, Some("c1"))),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .respond_with(ResponseTemplate::new(500).set_body_string("hf down"))
            .mount(&server)
            .await;

        let result = search_page(
            &server.uri(),
            &server.uri(),
            &test_cache(),
            &LiveSearchOpts::default(),
        )
        .await
        .expect("merged search must survive an HF 5xx");
        assert!(result.entries.iter().all(|e| e.id.0.starts_with("cv:")));
        assert!(!result.entries.is_empty());
    }

    #[tokio::test]
    async fn merged_search_propagates_civitai_errors() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(ResponseTemplate::new(500).set_body_string("civitai down"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vec![hf_hit(1)]))
            .mount(&server)
            .await;

        let err = search_page(
            &server.uri(),
            &server.uri(),
            &test_cache(),
            &LiveSearchOpts::default(),
        )
        .await
        .expect_err("civitai errors must propagate from the parallel path");
        match err {
            LiveSearchError::Upstream { host, status, .. } => {
                assert_eq!(host, "civitai.com");
                assert_eq!(status, 500);
            }
            other => panic!("expected Upstream 500, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn hf_page_cache_serves_repeats_without_upstream_calls() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vec![hf_hit(1), hf_hit(2)]))
            .expect(1)
            .mount(&server)
            .await;

        let cache = test_cache();
        let opts = LiveSearchOpts {
            source: Some(Source::Hf),
            page_size: 2,
            ..Default::default()
        };
        let first = search_page("http://127.0.0.1:1", &server.uri(), &cache, &opts)
            .await
            .expect("first");
        let second = search_page("http://127.0.0.1:1", &server.uri(), &cache, &opts)
            .await
            .expect("second");

        assert_eq!(ids(&first), ids(&second));
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            1,
            "the repeat must be an HF page-cache hit"
        );
    }

    const VERSION_DETAIL_BODY: &str = r#"{
        "id": 8001,
        "name": "v1",
        "description": "<p>Version notes &amp; guidance.</p>",
        "baseModel": "Flux.1 D",
        "baseModelType": "Standard",
        "files": [],
        "images": [],
        "model": { "name": "Test", "type": "LORA", "nsfw": false, "tags": [] }
    }"#;

    #[test]
    fn civitai_version_detail_deserializes_with_model_id() {
        let body = format!(
            "{}{}",
            r#"{ "modelId": 9001,"#,
            VERSION_DETAIL_BODY.trim_start().trim_start_matches('{')
        );
        let detail: CivitaiVersionDetail = serde_json::from_str(&body).expect("with modelId");
        assert_eq!(detail.model_id, Some(9001));
        assert_eq!(detail.version.id, 8001);
        assert_eq!(
            detail.version.description.as_deref(),
            Some("<p>Version notes &amp; guidance.</p>")
        );
    }

    #[test]
    fn civitai_version_detail_deserializes_without_model_id() {
        // Older cached bodies / test fixtures omit `modelId`; the field
        // must default to None instead of failing the whole fetch.
        let detail: CivitaiVersionDetail =
            serde_json::from_str(VERSION_DETAIL_BODY).expect("without modelId");
        assert_eq!(detail.model_id, None);
        assert_eq!(detail.version.id, 8001);
    }

    #[tokio::test]
    async fn fetch_entry_by_id_rejects_non_catalog_shape() {
        // The cv:/hf: dispatch itself is covered end-to-end by the server's
        // wiremock-backed catalog integration tests; here we only pin the
        // defensive fallback for a mis-shaped id.
        let err = fetch_entry_by_id(
            "flux-dev",
            "http://127.0.0.1:0",
            "http://127.0.0.1:0",
            None,
            None,
        )
        .await
        .unwrap_err();
        match err {
            LiveSearchError::Upstream { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Upstream 400 for non-catalog id, got {other:?}"),
        }
    }
}
