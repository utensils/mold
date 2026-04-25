//! Catalog REST + scan-queue surface. Mirrors the single-writer pattern
//! used by `crate::downloads::DownloadQueue`: one scan at a time per
//! server, status polled by id.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use mold_catalog::scanner::{run_scan, ScanOptions, ScanReport};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CatalogScanStatus {
    Pending,
    Running,
    Done {
        total_entries: usize,
        per_family: BTreeMap<String, String>,
    },
    Failed {
        message: String,
    },
}

#[async_trait]
pub trait ScanDriver: Send + Sync + 'static {
    async fn run(&self, opts: ScanOptions) -> ScanReport;
}

/// Production driver. Hits the real Hugging Face + Civitai endpoints.
pub struct LiveScanDriver;

#[async_trait]
impl ScanDriver for LiveScanDriver {
    async fn run(&self, opts: ScanOptions) -> ScanReport {
        run_scan("https://huggingface.co", "https://civitai.com", &opts).await
    }
}

#[derive(Clone)]
pub struct CatalogScanQueue {
    inner: Arc<Mutex<Inner>>,
    notify: Arc<tokio::sync::Notify>,
}

struct Inner {
    pending: Option<(String, ScanOptions)>,
    statuses: BTreeMap<String, CatalogScanStatus>,
    active: Option<String>,
}

impl Default for CatalogScanQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl CatalogScanQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                pending: None,
                statuses: BTreeMap::new(),
                active: None,
            })),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub async fn enqueue(&self, opts: ScanOptions) -> Result<String, String> {
        let mut inner = self.inner.lock().await;
        if inner.active.is_some() || inner.pending.is_some() {
            return Err("a catalog refresh is already in progress".into());
        }
        let id = Uuid::new_v4().to_string();
        inner.pending = Some((id.clone(), opts));
        inner
            .statuses
            .insert(id.clone(), CatalogScanStatus::Pending);
        drop(inner);
        self.notify.notify_one();
        Ok(id)
    }

    pub async fn status(&self, id: &str) -> Option<CatalogScanStatus> {
        self.inner.lock().await.statuses.get(id).cloned()
    }

    /// Returns the in-flight scan (active or pending) so clients other
    /// than the one that submitted it can still observe it. Used by the
    /// web UI to disable the "Refresh catalog" button when a scan
    /// initiated by another browser tab or the CLI is already running.
    pub async fn active(&self) -> Option<(String, CatalogScanStatus)> {
        let inner = self.inner.lock().await;
        let id = inner
            .active
            .clone()
            .or_else(|| inner.pending.as_ref().map(|(id, _)| id.clone()))?;
        let status = inner.statuses.get(&id).cloned()?;
        Some((id, status))
    }

    pub async fn drive(self, driver: Arc<dyn ScanDriver>) {
        loop {
            self.notify.notified().await;
            let job = {
                let mut inner = self.inner.lock().await;
                let job = inner.pending.take();
                if let Some((id, _)) = &job {
                    inner.active = Some(id.clone());
                    inner
                        .statuses
                        .insert(id.clone(), CatalogScanStatus::Running);
                }
                job
            };
            let Some((id, opts)) = job else { continue };
            let report = driver.run(opts).await;
            let mut summary = BTreeMap::new();
            for (fam, outcome) in &report.per_family {
                summary.insert(fam.as_str().to_string(), format!("{:?}", outcome));
            }
            let mut inner = self.inner.lock().await;
            inner.active = None;
            inner.statuses.insert(
                id.clone(),
                CatalogScanStatus::Done {
                    total_entries: report.total_entries,
                    per_family: summary,
                },
            );
        }
    }
}

// ── REST handlers ────────────────────────────────────────────────────────────

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use mold_db::catalog::{ListParams, SortBy};

#[derive(Debug, serde::Deserialize)]
pub struct ListQuery {
    pub family: Option<String>,
    pub family_role: Option<String>,
    pub modality: Option<String>,
    pub source: Option<String>,
    pub sub_family: Option<String>,
    pub q: Option<String>,
    pub include_nsfw: Option<bool>,
    pub max_engine_phase: Option<u8>,
    pub sort: Option<String>,
    pub page: Option<i64>,
    pub page_size: Option<i64>,
    /// Legacy alias accepted as `limit` (maps to `page_size` on the first page).
    pub limit: Option<i64>,
    /// Legacy alias accepted as `offset` (treated as absolute row offset).
    pub offset: Option<i64>,
}

pub async fn list_catalog(
    State(state): State<crate::state::AppState>,
    Query(q): Query<ListQuery>,
) -> impl IntoResponse {
    let page_size = q.page_size.or(q.limit).unwrap_or(48).clamp(1, 200);
    let page = q.page.unwrap_or(1).max(1);
    // When a raw `offset` is supplied (legacy) use it directly; otherwise
    // derive from `page`.
    let offset = q.offset.unwrap_or_else(|| (page - 1) * page_size);
    let params = ListParams {
        family: q.family,
        family_role: q.family_role,
        modality: q.modality,
        source: q.source,
        sub_family: q.sub_family,
        q: q.q,
        include_nsfw: q.include_nsfw.unwrap_or(false),
        max_engine_phase: q.max_engine_phase,
        sort: match q.sort.as_deref() {
            Some("rating") => SortBy::Rating,
            Some("recent") => SortBy::Recent,
            Some("name") => SortBy::Name,
            _ => SortBy::Downloads,
        },
        limit: page_size,
        offset,
    };
    match state.catalog_db.catalog_list(&params) {
        Ok(rows) => Json(serde_json::json!({
            "entries": rows.into_iter().map(catalog_row_to_wire).collect::<Vec<_>>(),
            "page": page,
            "page_size": page_size,
        }))
        .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn get_catalog_entry(
    State(state): State<crate::state::AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.catalog_db.catalog_get(&id) {
        Ok(Some(row)) => Json(catalog_row_to_wire(row)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "not found").into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// POST dispatcher for `/api/catalog/*id` — routes sub-actions based on the
/// trailing path segment.  Currently only `/download` is handled; everything
/// else returns 404.
pub async fn post_catalog_dispatch(
    State(state): State<crate::state::AppState>,
    Path(rest): Path<String>,
) -> impl IntoResponse {
    if let Some(id) = rest.strip_suffix("/download") {
        post_catalog_download(State(state), Path(id.to_string()))
            .await
            .into_response()
    } else {
        (StatusCode::NOT_FOUND, "unknown catalog action").into_response()
    }
}

pub async fn list_families(State(state): State<crate::state::AppState>) -> impl IntoResponse {
    match state.catalog_db.catalog_family_counts() {
        Ok(rows) => Json(serde_json::json!({ "families": rows.into_iter().map(|fc| {
            serde_json::json!({
                "family": fc.family,
                "foundation": fc.foundation,
                "finetune": fc.finetune,
            })
        }).collect::<Vec<_>>() }))
        .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn catalog_row_to_wire(r: mold_db::catalog::CatalogRow) -> serde_json::Value {
    serde_json::json!({
        "id": r.id,
        "source": r.source,
        "source_id": r.source_id,
        "name": r.name,
        "author": r.author,
        "family": r.family,
        "family_role": r.family_role,
        "sub_family": r.sub_family,
        "modality": r.modality,
        "kind": r.kind,
        "file_format": r.file_format,
        "bundling": r.bundling,
        "size_bytes": r.size_bytes,
        "download_count": r.download_count,
        "rating": r.rating,
        "likes": r.likes,
        "nsfw": r.nsfw != 0,
        "thumbnail_url": r.thumbnail_url,
        "description": r.description,
        "license": r.license,
        "license_flags": r.license_flags.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
        "tags": r.tags.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
        "companions": r.companions.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
        "download_recipe": serde_json::from_str::<serde_json::Value>(&r.download_recipe).unwrap_or(serde_json::json!({})),
        "engine_phase": r.engine_phase,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "added_at": r.added_at,
    })
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct RefreshBody {
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub min_downloads: Option<u64>,
    #[serde(default)]
    pub no_nsfw: Option<bool>,
    #[serde(default)]
    pub include_nsfw: Option<bool>,
    #[serde(default)]
    pub hf_token: Option<String>,
    #[serde(default)]
    pub civitai_token: Option<String>,
}

pub async fn post_refresh(
    State(state): State<crate::state::AppState>,
    body: Option<Json<RefreshBody>>,
) -> impl IntoResponse {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let mut opts = ScanOptions::default();
    if let Some(family) = body.family.as_deref() {
        if let Ok(f) = mold_catalog::families::Family::from_str(family) {
            opts.families = vec![f];
        } else {
            return (StatusCode::BAD_REQUEST, format!("unknown family: {family}")).into_response();
        }
    }
    if let Some(m) = body.min_downloads {
        opts.min_downloads = m;
    }
    if let Some(true) = body.no_nsfw {
        opts.include_nsfw = false;
    } else if let Some(true) = body.include_nsfw {
        opts.include_nsfw = true;
    }
    opts.hf_token = body.hf_token.or_else(|| std::env::var("HF_TOKEN").ok());
    opts.civitai_token = body
        .civitai_token
        .or_else(|| std::env::var("CIVITAI_TOKEN").ok());

    match state.catalog_scan.enqueue(opts).await {
        Ok(id) => (StatusCode::ACCEPTED, Json(serde_json::json!({ "id": id }))).into_response(),
        Err(msg) => (StatusCode::CONFLICT, msg).into_response(),
    }
}

pub async fn get_refresh_status(
    State(state): State<crate::state::AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.catalog_scan.status(&id).await {
        Some(status) => Json(status).into_response(),
        None => (StatusCode::NOT_FOUND, "unknown scan id").into_response(),
    }
}

/// `GET /api/catalog/refresh` — returns the in-flight scan if any. The
/// web UI calls this on mount to disable the refresh button when a scan
/// initiated by another tab or the CLI is already running.
pub async fn get_active_refresh(State(state): State<crate::state::AppState>) -> impl IntoResponse {
    match state.catalog_scan.active().await {
        Some((id, status)) => {
            Json(serde_json::json!({ "active": { "id": id, "status": status } })).into_response()
        }
        None => Json(serde_json::json!({ "active": null })).into_response(),
    }
}

pub async fn post_catalog_download(
    State(state): State<crate::state::AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let row = match state.catalog_db.catalog_get(&id) {
        Ok(Some(r)) => r,
        Ok(None) => return (StatusCode::NOT_FOUND, "unknown catalog id").into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    if row.engine_phase >= 2 {
        return (
            StatusCode::CONFLICT,
            format!(
                "engine_phase {} not yet supported by this build — see release notes",
                row.engine_phase
            ),
        )
            .into_response();
    }

    // Phase 1 reuses the existing `DownloadQueue` for HF entries that map
    // 1:1 onto a manifest model name. For Civitai entries, the recipe's
    // first file URL is used to build a stub manifest at runtime — that
    // path is implemented in phase 2 alongside companion auto-pull. For
    // now, return a placeholder list of pending job ids.
    let mut job_ids: Vec<String> = Vec::new();
    if row.source == "hf" {
        // Best-effort: use the source_id (e.g. "black-forest-labs/FLUX.1-dev")
        // to look up an existing manifest model name; if not found, surface
        // the catalog-id directly to the queue.
        let model = match mold_core::manifest::find_manifest(&row.source_id) {
            Some(m) => m.name.clone(),
            None => row.source_id.clone(),
        };
        match state.downloads.enqueue(model).await {
            Ok((jid, _, _)) => job_ids.push(jid),
            // Phase 2 will wire up catalog-native downloads; for entries that
            // aren't in the static manifest yet, accept the request and return
            // an empty job list — the caller can poll the catalog entry status.
            Err(crate::downloads::EnqueueError::UnknownModel(_)) => {}
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        }
    } else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "civitai catalog download is implemented in phase 2 (single-file loaders)",
        )
            .into_response();
    }

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_ids": job_ids })),
    )
        .into_response()
}

#[cfg(test)]
#[path = "catalog_api_test.rs"]
mod catalog_api_test;
