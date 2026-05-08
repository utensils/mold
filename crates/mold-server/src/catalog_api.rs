//! Catalog REST + scan-queue surface. Mirrors the single-writer pattern
//! used by `crate::downloads::DownloadQueue`: one scan at a time per
//! server, status polled by id.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use mold_catalog::scanner::{
    run_scan_with_progress, ProgressHandle, ScanOptions, ScanProgress, ScanReport,
};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CatalogScanStatus {
    Pending,
    Running {
        families_total: usize,
        families_done: usize,
        current_family: Option<String>,
        current_stage: Option<String>,
        /// HF-stage seed currently being walked. `None` when the stage
        /// is idle, between families, or in the Civitai stage.
        current_seed: Option<String>,
        /// Pages walked within `current_seed`. Resets to 0 each seed.
        pages_done: usize,
        /// Entries collected for `current_family` so far. Cumulative
        /// across both stages within a family; resets between families.
        entries_so_far: usize,
        /// Wall-clock ms since epoch — lets the UI render an elapsed
        /// timer that's correct even for clients attaching mid-scan.
        started_at_ms: u64,
    },
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
    /// Scans must write into the supplied progress handle as they walk
    /// each family — the queue's `status()` reads from the same handle
    /// so polling clients see the bar advance.
    async fn run(&self, opts: ScanOptions, progress: ProgressHandle) -> ScanReport;
}

/// Production driver. Hits the real Hugging Face + Civitai endpoints.
pub struct LiveScanDriver;

#[async_trait]
impl ScanDriver for LiveScanDriver {
    async fn run(&self, opts: ScanOptions, progress: ProgressHandle) -> ScanReport {
        run_scan_with_progress(
            "https://huggingface.co",
            "https://civitai.com",
            &opts,
            Some(progress),
        )
        .await
    }
}

#[derive(Clone)]
pub struct CatalogScanQueue {
    inner: Arc<Mutex<Inner>>,
    notify: Arc<tokio::sync::Notify>,
}

/// In-flight scan bookkeeping. The progress handle is shared with the
/// scanner so `status()` can read live counters without the driver
/// having to push updates back into `Inner`.
struct InFlight {
    id: String,
    started_at_ms: u64,
    progress: ProgressHandle,
}

struct Inner {
    pending: Option<(String, ScanOptions)>,
    /// Holds Pending/Done/Failed snapshots only — Running is derived
    /// live from `in_flight` so families_done/current_family stay fresh.
    statuses: BTreeMap<String, CatalogScanStatus>,
    in_flight: Option<InFlight>,
}

impl Default for CatalogScanQueue {
    fn default() -> Self {
        Self::new()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn running_from(snapshot: &ScanProgress, started_at_ms: u64) -> CatalogScanStatus {
    CatalogScanStatus::Running {
        families_total: snapshot.families_total,
        families_done: snapshot.families_done,
        current_family: snapshot.current_family.map(|f| f.as_str().to_string()),
        current_stage: snapshot.current_stage.map(str::to_string),
        current_seed: snapshot.current_seed.clone(),
        pages_done: snapshot.pages_done,
        entries_so_far: snapshot.entries_so_far,
        started_at_ms,
    }
}

impl CatalogScanQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                pending: None,
                statuses: BTreeMap::new(),
                in_flight: None,
            })),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub async fn enqueue(&self, opts: ScanOptions) -> Result<String, String> {
        let mut inner = self.inner.lock().await;
        if inner.in_flight.is_some() || inner.pending.is_some() {
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
        let inner = self.inner.lock().await;
        if let Some(flight) = &inner.in_flight {
            if flight.id == id {
                let snapshot = flight
                    .progress
                    .lock()
                    .map(|g| g.clone())
                    .unwrap_or_default();
                return Some(running_from(&snapshot, flight.started_at_ms));
            }
        }
        inner.statuses.get(id).cloned()
    }

    /// Returns the in-flight scan (active or pending) so clients other
    /// than the one that submitted it can still observe it. Used by the
    /// web UI to disable the "Refresh catalog" button when a scan
    /// initiated by another browser tab or the CLI is already running.
    pub async fn active(&self) -> Option<(String, CatalogScanStatus)> {
        let inner = self.inner.lock().await;
        if let Some(flight) = &inner.in_flight {
            let snapshot = flight
                .progress
                .lock()
                .map(|g| g.clone())
                .unwrap_or_default();
            return Some((
                flight.id.clone(),
                running_from(&snapshot, flight.started_at_ms),
            ));
        }
        if let Some((id, _)) = &inner.pending {
            let status = inner
                .statuses
                .get(id)
                .cloned()
                .unwrap_or(CatalogScanStatus::Pending);
            return Some((id.clone(), status));
        }
        None
    }

    /// Background driver loop. Pops one queued scan at a time, asks
    /// `driver` to run it, then writes the resulting entries through
    /// `mold_catalog::sink::upsert_family` so the catalog DB reflects
    /// the scan. `catalog_db` is the sink target — typically
    /// `state.catalog_db`.
    pub async fn drive(self, driver: Arc<dyn ScanDriver>, catalog_db: Arc<mold_db::MetadataDb>) {
        loop {
            self.notify.notified().await;
            let job = {
                let mut inner = self.inner.lock().await;
                let job = inner.pending.take();
                if let Some((id, _)) = &job {
                    let progress: ProgressHandle =
                        Arc::new(std::sync::Mutex::new(ScanProgress::default()));
                    inner.in_flight = Some(InFlight {
                        id: id.clone(),
                        started_at_ms: now_ms(),
                        progress,
                    });
                }
                job
            };
            let Some((id, opts)) = job else { continue };
            let progress = {
                let inner = self.inner.lock().await;
                inner
                    .in_flight
                    .as_ref()
                    .expect("in_flight set above")
                    .progress
                    .clone()
            };
            let report = driver.run(opts, progress).await;

            // Phase H follow-up: persist scanned entries. The orchestrator
            // gathered them per family in `report.entries`; without this
            // sink call the HTTP path would count entries and drop them on
            // the floor (see commit history for the catalog-scanner
            // observability work).
            for (fam, entries) in &report.entries {
                let fam_copy = *fam;
                let entries_copy = entries.clone();
                let db = catalog_db.clone();
                let res = tokio::task::spawn_blocking(move || {
                    db.with_conn(|conn| {
                        Ok(mold_catalog::sink::upsert_family(
                            conn,
                            fam_copy,
                            &entries_copy,
                        )?)
                    })
                })
                .await;
                match res {
                    Ok(Ok(())) => {
                        tracing::info!(
                            target: "catalog",
                            family = %fam.as_str(),
                            rows = entries.len(),
                            "persisted scan entries",
                        );
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(
                            target: "catalog",
                            family = %fam.as_str(),
                            error = %e,
                            "persist failed",
                        );
                    }
                    Err(join_err) => {
                        tracing::warn!(
                            target: "catalog",
                            family = %fam.as_str(),
                            error = %join_err,
                            "persist task panicked",
                        );
                    }
                }
            }

            let mut summary = BTreeMap::new();
            for (fam, outcome) in &report.per_family {
                summary.insert(fam.as_str().to_string(), format!("{:?}", outcome));
            }
            let mut inner = self.inner.lock().await;
            inner.in_flight = None;
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
    pub kind: Option<String>,
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
        kind: q.kind,
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
    let cfg_guard = state.config.read().await;
    let models_dir = cfg_guard.resolved_models_dir();
    // `total` is the unpaginated count for the same WHERE clauses — the SPA
    // uses it to stop infinite-scrolling once `entries.length >= total`.
    let total = match state.catalog_db.catalog_count(&params) {
        Ok(n) => n,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    match state.catalog_db.catalog_list(&params) {
        Ok(rows) => Json(serde_json::json!({
            "entries": rows
                .into_iter()
                .map(|r| catalog_row_to_wire(r, &models_dir, &cfg_guard))
                .collect::<Vec<_>>(),
            "page": page,
            "page_size": page_size,
            "total": total,
        }))
        .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn get_catalog_entry(
    State(state): State<crate::state::AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let cfg_guard = state.config.read().await;
    let models_dir = cfg_guard.resolved_models_dir();
    match state.catalog_db.catalog_get(&id) {
        Ok(Some(row)) => Json(catalog_row_to_wire(row, &models_dir, &cfg_guard)).into_response(),
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
    use mold_catalog::families::{Family, ALL_FAMILIES};
    use std::collections::HashMap;
    let db_rows = match state.catalog_db.catalog_family_counts() {
        Ok(rows) => rows,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    // Surface every supported family — even ones with zero DB rows — so the
    // sidebar can drill into a family that the live-search backend will
    // populate on demand. Without this, families whose DB shard is empty
    // (e.g. FLUX after a partially-rate-limited refresh) silently disappear
    // from the UI even though `/api/catalog/search?family=flux` returns
    // results.
    let by_name: HashMap<&str, &mold_db::catalog::FamilyCount> =
        db_rows.iter().map(|r| (r.family.as_str(), r)).collect();
    let merged: Vec<serde_json::Value> = ALL_FAMILIES
        .iter()
        .map(|f: &Family| {
            let key = f.as_str();
            let (foundation, finetune) = by_name
                .get(key)
                .map(|r| (r.foundation, r.finetune))
                .unwrap_or((0, 0));
            serde_json::json!({
                "family": key,
                "foundation": foundation,
                "finetune": finetune,
            })
        })
        .collect();
    Json(serde_json::json!({ "families": merged })).into_response()
}

fn catalog_row_to_wire(
    r: mold_db::catalog::CatalogRow,
    models_dir: &std::path::Path,
    config: &mold_core::Config,
) -> serde_json::Value {
    let (installed, primary_path) = match r.source.as_str() {
        // HF rows install via the manifest path. The catalog stores the
        // HF repo path in `source_id` (e.g. "black-forest-labs/FLUX.1-dev")
        // — mold's manifest registry is keyed on canonical names like
        // "flux-dev:q8" instead, so we use the reverse lookup
        // `find_manifest_by_hf_repo` to bridge them. When no manifest in
        // this build matches the HF repo, the entry can't be installed
        // (no canonical name to point disk-state at), so report false.
        "hf" => match mold_core::manifest::find_manifest_by_hf_repo(&r.source_id) {
            Some(manifest) => (config.manifest_model_is_downloaded(&manifest.name), None),
            None => (false, None),
        },
        // Civitai rows go through the recipe path. Parse the stored recipe
        // JSON, translate to RecipeFetchFile borrowed slice, and delegate
        // to mold_core::download::catalog_entry_installed (which uses the
        // shared recipe_file_is_placed predicate so the wire field can't
        // disagree with the fetcher's idempotency check).
        "civitai" => {
            match serde_json::from_str::<mold_catalog::entry::DownloadRecipe>(&r.download_recipe) {
                Ok(recipe) => {
                    let (author, name) = match r.source_id.split_once('/') {
                        Some((a, n)) => (a, n),
                        None => ("", r.source_id.as_str()),
                    };
                    let rendered_dests: Vec<String> = recipe
                        .files
                        .iter()
                        .map(|f| {
                            mold_catalog::entry::render_recipe_dest(
                                &f.dest, &r.family, author, name,
                            )
                        })
                        .collect();
                    let files: Vec<mold_core::download::RecipeFetchFile<'_>> = recipe
                        .files
                        .iter()
                        .zip(rendered_dests.iter())
                        .map(|(f, dest)| mold_core::download::RecipeFetchFile {
                            url: f.url.as_str(),
                            dest: dest.as_str(),
                            sha256: f.sha256.as_deref(),
                            size_bytes: f.size_bytes,
                        })
                        .collect();
                    let is_installed =
                        mold_core::download::catalog_entry_installed(models_dir, &r.id, &files);
                    // For installed Civitai entries expose the absolute path of
                    // the primary (first) recipe file so the web UI can pass it
                    // directly as `lora.path` in a generate request.
                    // Path layout: {models_dir}/{sanitized_id}/{rendered_dest}
                    let path = if is_installed {
                        let sanitized = mold_core::download::sanitize_recipe_id(&r.id);
                        let subdir = models_dir.join(&sanitized);
                        rendered_dests
                            .first()
                            .map(|dest| subdir.join(dest).to_string_lossy().into_owned())
                    } else {
                        None
                    };
                    (is_installed, path)
                }
                Err(e) => {
                    tracing::warn!(
                        target: "catalog",
                        catalog_id = %r.id,
                        error = %e,
                        "failed to parse stored download_recipe; reporting installed=false",
                    );
                    (false, None)
                }
            }
        }
        _ => (false, None),
    };
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
        "installed": installed,
        "primary_path": primary_path,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "added_at": r.added_at,
        // Trigger phrases for Civitai LoRAs; the picker renders these as
        // click-to-insert chips next to the LoRA selection. JSON-decoded
        // here so the client doesn't need a second pass.
        "trained_words": serde_json::from_str::<serde_json::Value>(&r.trained_words)
            .unwrap_or(serde_json::json!([])),
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
    /// Defensive cap on per-family wall-clock walk time in seconds.
    /// Mirrors the CLI's `--max-family-wallclock-secs`.
    #[serde(default)]
    pub max_family_wallclock_secs: Option<u64>,
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
    opts.max_family_wallclock = body
        .max_family_wallclock_secs
        .map(std::time::Duration::from_secs);

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

/// One queued companion download surfaced in the
/// `POST /api/catalog/:id/download` response.
#[derive(Clone, Debug, serde::Serialize)]
pub struct CompanionJob {
    /// Canonical companion name from the catalog scanner
    /// (`mold_catalog::companions::COMPANIONS`). Same string the
    /// `companions` array on the catalog row carries.
    pub name: String,
    /// Job id that polls against `GET /api/downloads`.
    pub job_id: String,
}

/// Enqueue any companions referenced by `companions_json` that are not
/// already fully present under `models_dir`. Returns the per-companion job
/// ids in the order they appear in the catalog row.
///
/// `companions_json` is the raw value from `CatalogRow::companions` — a
/// JSON-encoded `Vec<String>` of canonical companion names. Phase-1 HF
/// diffusers entries pass `None` here because they don't strip components.
///
/// On-disk presence + name resolution are delegated to
/// `mold_core::download::missing_companions_from_json`, which the CLI also
/// consumes for `mold pull cv:<id>`'s companion-first ordering.
///
/// `DownloadQueue::enqueue` is idempotent against canonical manifest
/// names: re-enqueuing a companion that's mid-flight returns the existing
/// job id so concurrent download requests won't double-pull. That lets
/// the on-disk presence check fall back on "missing → enqueue" without a
/// race condition.
pub(crate) async fn enqueue_missing_companions(
    companions_json: Option<&str>,
    models_dir: &std::path::Path,
    queue: &crate::downloads::DownloadQueue,
    catalog_id: Option<&str>,
) -> Vec<CompanionJob> {
    let manifests = mold_core::download::missing_companions_from_json(companions_json, models_dir);
    let mut jobs = Vec::with_capacity(manifests.len());
    for manifest in manifests {
        // When called for a catalog download, the companion job is
        // registered in the catalog group so `CatalogReady` waits for
        // it. Outside that context (no catalog id), fall back to the
        // ungrouped enqueue — companions invoked from elsewhere don't
        // synchronise on a group event.
        let result = match catalog_id {
            Some(id) => queue.enqueue_in_group(manifest.name.clone(), id).await,
            None => queue.enqueue(manifest.name.clone()).await,
        };
        match result {
            Ok((job_id, _, _)) => jobs.push(CompanionJob {
                name: manifest.name.clone(),
                job_id,
            }),
            Err(e) => {
                tracing::warn!(
                    companion = %manifest.name,
                    error = %e,
                    "companion enqueue failed",
                );
            }
        }
    }
    jobs
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

    if row.engine_phase >= 6 {
        return (
            StatusCode::CONFLICT,
            format!(
                "engine_phase {} not yet supported by this build — see release notes",
                row.engine_phase
            ),
        )
            .into_response();
    }

    // Phase 2: companions auto-pull before the primary entry. Civitai
    // single-file checkpoints commonly strip their text encoders + VAE,
    // so the catalog scanner records `companions: ["clip-l", ...]` on
    // those entries. Each canonical companion has a hidden synthetic
    // manifest in `mold-core` so the queue can enqueue them by name.
    let models_dir = state.config.read().await.resolved_models_dir();
    let companion_jobs = enqueue_missing_companions(
        row.companions.as_deref(),
        &models_dir,
        &state.downloads,
        Some(&row.id),
    )
    .await;

    // Primary entry. HF rows map onto a manifest model name (best-effort —
    // diffusers entries with a recognised source_id flow through; the rest
    // surface a `null` primary while companion downloads still run). `cv:`
    // rows go through the recipe path: parse `download_recipe`, resolve
    // `CIVITAI_TOKEN` if `needs_token: Civitai`, and call
    // `DownloadQueue::enqueue_recipe`. The job id surfaces in
    // `primary_job_id` so the SPA's downloads drawer can poll it like any
    // other job.
    let primary_job_id: Option<String> = if row.source == "hf" {
        let model = match mold_core::manifest::find_manifest(&row.source_id) {
            Some(m) => m.name.clone(),
            None => row.source_id.clone(),
        };
        match state.downloads.enqueue_in_group(model, &row.id).await {
            Ok((jid, _, _)) => Some(jid),
            Err(crate::downloads::EnqueueError::UnknownModel(_)) => None,
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        }
    } else if row.source == "civitai" {
        let recipe: mold_catalog::entry::DownloadRecipe =
            match serde_json::from_str(&row.download_recipe) {
                Ok(r) => r,
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("catalog row {} has malformed download_recipe: {e}", row.id),
                    )
                        .into_response();
                }
            };
        let auth = match recipe.needs_token {
            Some(mold_catalog::entry::TokenKind::Civitai) => {
                match mold_core::download::civitai_auth_or_error(&row.id) {
                    Ok(a) => a,
                    Err(e) => {
                        return (StatusCode::UNAUTHORIZED, e.to_string()).into_response();
                    }
                }
            }
            _ => mold_core::download::RecipeAuth::None,
        };
        let (author, name) = match row.source_id.split_once('/') {
            Some((a, n)) => (a.to_string(), n.to_string()),
            None => (String::new(), row.source_id.clone()),
        };
        let files: Vec<crate::downloads::OwnedRecipeFile> = recipe
            .files
            .into_iter()
            .map(|f| crate::downloads::OwnedRecipeFile {
                url: f.url,
                dest: mold_catalog::entry::render_recipe_dest(&f.dest, &row.family, &author, &name),
                sha256: f.sha256,
                size_bytes: f.size_bytes,
            })
            .collect();
        // Sidecar write is best-effort: it lets the LoRA picker see the
        // entry as soon as the file lands without re-querying the live
        // catalog. Failure is not fatal — install proceeds either way,
        // and the next install of the same id (or a startup backfill)
        // would heal the missing sidecar.
        if let Some(primary) = files.first() {
            let sidecar = mold_catalog::sidecar::sidecar_from_row(&row, primary.dest.clone());
            let sc_path = mold_catalog::sidecar::civitai_sidecar_path(&models_dir, &row.id);
            if let Err(e) = mold_catalog::sidecar::write_sidecar(&sc_path, &sidecar) {
                tracing::warn!(
                    target: "catalog.sidecar",
                    catalog_id = %row.id,
                    error = %e,
                    "sidecar write failed; picker will need a backfill or reinstall to surface this row",
                );
            }
        }
        let payload = crate::downloads::RecipePayload {
            catalog_id: row.id.clone(),
            files,
            auth,
        };
        match state
            .downloads
            .enqueue_recipe_in_group(payload, &row.id)
            .await
        {
            Ok((jid, _, _)) => Some(jid),
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        }
    } else {
        None
    };

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "primary_job_id": primary_job_id,
            "companion_jobs": companion_jobs,
        })),
    )
        .into_response()
}

// ── Live search + installed (replaces the bulk-scrape DB) ──────────────────

#[derive(Debug, serde::Deserialize)]
pub struct LiveSearchQuery {
    pub q: Option<String>,
    pub family: Option<String>,
    pub kind: Option<String>,
    pub source: Option<String>,
    pub include_nsfw: Option<bool>,
    pub page: Option<u32>,
    pub page_size: Option<u32>,
}

/// `GET /api/catalog/search` — live proxy to upstream catalog APIs with
/// a 5-min in-process cache. Replaces `/api/catalog` on the read path
/// for the SPA's catalog tab. The bulk-scrape DB is retained until a
/// follow-up release.
pub async fn live_search_catalog(
    State(state): State<crate::state::AppState>,
    Query(q): Query<LiveSearchQuery>,
) -> impl IntoResponse {
    let family = match q.family.as_deref().filter(|s| !s.is_empty()) {
        Some(s) => match mold_catalog::families::Family::from_str(s) {
            Ok(f) => Some(f),
            Err(_) => {
                return (StatusCode::BAD_REQUEST, format!("unknown family: {s}")).into_response()
            }
        },
        None => None,
    };
    let kind = match q.kind.as_deref().filter(|s| !s.is_empty()) {
        Some(s) => match parse_kind(s) {
            Some(k) => Some(k),
            None => return (StatusCode::BAD_REQUEST, format!("unknown kind: {s}")).into_response(),
        },
        None => None,
    };
    let source = match q.source.as_deref().filter(|s| !s.is_empty()) {
        Some("hf") => Some(mold_catalog::entry::Source::Hf),
        Some("civitai") => Some(mold_catalog::entry::Source::Civitai),
        Some(other) => {
            return (StatusCode::BAD_REQUEST, format!("unknown source: {other}")).into_response()
        }
        None => None,
    };

    let opts = mold_catalog::live::LiveSearchOpts {
        q: q.q.clone(),
        family,
        kind,
        source,
        page: q.page.unwrap_or(1).max(1),
        page_size: q.page_size.unwrap_or(20).clamp(1, 100),
        include_nsfw: q.include_nsfw.unwrap_or(false),
        civitai_token: std::env::var("CIVITAI_TOKEN").ok(),
        hf_token: std::env::var("HF_TOKEN").ok(),
    };

    let cfg = state.config.read().await;
    let models_dir = cfg.resolved_models_dir();
    drop(cfg);

    let entries = match mold_catalog::live::search(
        state.catalog_live_civitai_base.as_str(),
        "https://huggingface.co",
        &state.catalog_live_cache,
        &opts,
    )
    .await
    {
        Ok(es) => es,
        Err(e) => {
            tracing::warn!(target: "catalog.live", error = %e, "live search failed");
            return (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response();
        }
    };

    let wire: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| live_entry_to_wire(e, &models_dir))
        .collect();
    let total = wire.len() as i64;
    Json(serde_json::json!({
        "entries": wire,
        "page": opts.page,
        "page_size": opts.page_size,
        "total": total,
    }))
    .into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct InstalledQuery {
    pub kind: Option<String>,
    pub family: Option<String>,
}

/// `GET /api/catalog/installed` — enumerate installed catalog entries
/// from the per-install sidecar files under `models_dir`. The LoRA
/// picker uses this as its primary data source.
pub async fn list_installed_catalog(
    State(state): State<crate::state::AppState>,
    Query(q): Query<InstalledQuery>,
) -> impl IntoResponse {
    let kind_filter = q.kind.as_deref().map(|s| s.to_string());
    let family_filter = q.family.as_deref().map(|s| s.to_string());

    let cfg = state.config.read().await;
    let models_dir = cfg.resolved_models_dir();
    drop(cfg);

    let walked = mold_catalog::sidecar::walk_sidecars(&models_dir);
    let mut wire = Vec::with_capacity(walked.len());
    for (dir, sidecar) in walked {
        if let Some(k) = kind_filter.as_deref() {
            if sidecar.kind != k {
                continue;
            }
        }
        if let Some(f) = family_filter.as_deref() {
            if sidecar.family != f {
                continue;
            }
        }
        let abs = mold_catalog::sidecar::primary_path_if_present(&dir, &sidecar);
        let installed = abs.is_some();
        let primary_path = abs.map(|p| p.to_string_lossy().into_owned());
        wire.push(sidecar_to_wire(sidecar, installed, primary_path));
    }
    let total = wire.len() as i64;
    Json(serde_json::json!({
        "entries": wire,
        "page": 1,
        "page_size": total,
        "total": total,
    }))
    .into_response()
}

fn parse_kind(s: &str) -> Option<mold_catalog::entry::Kind> {
    use mold_catalog::entry::Kind::*;
    Some(match s {
        "checkpoint" => Checkpoint,
        "lora" => Lora,
        "vae" => Vae,
        "text-encoder" => TextEncoder,
        "control-net" => ControlNet,
        _ => return None,
    })
}

fn live_entry_to_wire(
    entry: &mold_catalog::entry::CatalogEntry,
    models_dir: &std::path::Path,
) -> serde_json::Value {
    // Installed-detection for live rows hangs off the sidecar layout: a
    // sidecar at `{models_dir}/{sanitized}/mold-catalog.json` whose
    // primary file actually exists on disk → installed=true. We
    // intentionally do NOT re-walk the recipe here: that path is a
    // legacy fallback for rows the catalog DB returned, and live rows
    // by definition pre-date their sidecar (which gets written at
    // download time).
    let (installed, primary_path) = if matches!(entry.source, mold_catalog::entry::Source::Civitai)
    {
        let sc_path = mold_catalog::sidecar::civitai_sidecar_path(models_dir, entry.id.as_str());
        match mold_catalog::sidecar::read_sidecar(&sc_path) {
            Ok(sidecar) => match sc_path.parent() {
                Some(parent) => {
                    match mold_catalog::sidecar::primary_path_if_present(parent, &sidecar) {
                        Some(abs) => (true, Some(abs.to_string_lossy().into_owned())),
                        None => (false, None),
                    }
                }
                None => (false, None),
            },
            Err(_) => (false, None),
        }
    } else {
        (false, None)
    };
    serde_json::json!({
        "id": entry.id.as_str(),
        "source": entry.source,
        "source_id": entry.source_id,
        "name": entry.name,
        "author": entry.author,
        "family": entry.family.as_str(),
        "family_role": entry.family_role,
        "sub_family": entry.sub_family,
        "modality": entry.modality,
        "kind": entry.kind,
        "file_format": entry.file_format,
        "bundling": entry.bundling,
        "size_bytes": entry.size_bytes,
        "download_count": entry.download_count,
        "rating": entry.rating,
        "likes": entry.likes,
        "nsfw": entry.nsfw,
        "thumbnail_url": entry.thumbnail_url,
        "description": entry.description,
        "license": entry.license,
        "license_flags": entry.license_flags,
        "tags": entry.tags,
        "companions": entry.companions,
        "download_recipe": entry.download_recipe,
        "engine_phase": entry.engine_phase,
        "installed": installed,
        "primary_path": primary_path,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "added_at": entry.added_at,
        "trained_words": entry.trained_words,
    })
}

fn sidecar_to_wire(
    sc: mold_catalog::sidecar::CatalogSidecar,
    installed: bool,
    primary_path: Option<String>,
) -> serde_json::Value {
    // The sidecar carries fewer fields than a full CatalogEntryWire on
    // purpose — the picker only needs id/name/family/kind/trained_words/
    // primary_path. Empty defaults are returned for fields the SPA
    // ignores in this code path; this keeps the wire shape uniform with
    // `/api/catalog/search` so a single TypeScript interface works for
    // both.
    serde_json::json!({
        "id": sc.id,
        "source": sc.source,
        "source_id": sc.source_id,
        "name": sc.name,
        "author": sc.author,
        "family": sc.family,
        "family_role": sc.family_role,
        "sub_family": sc.sub_family,
        "modality": sc.modality,
        "kind": sc.kind,
        "file_format": "safetensors",
        "bundling": "single-file",
        "size_bytes": sc.size_bytes,
        "download_count": 0,
        "rating": null,
        "likes": 0,
        "nsfw": false,
        "thumbnail_url": sc.thumbnail_url,
        "description": null,
        "license": null,
        "license_flags": null,
        "tags": [],
        "companions": [],
        "download_recipe": { "files": [], "needs_token": null },
        "engine_phase": sc.engine_phase,
        "installed": installed,
        "primary_path": primary_path,
        "created_at": null,
        "updated_at": null,
        "added_at": sc.written_at,
        "trained_words": sc.trained_words,
    })
}

#[cfg(test)]
#[path = "catalog_api_test.rs"]
mod catalog_api_test;

#[cfg(test)]
#[path = "catalog_live_test.rs"]
mod catalog_live_test;
