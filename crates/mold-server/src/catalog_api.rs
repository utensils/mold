//! Catalog REST surface. Live HF + Civitai proxy with a 5-min in-process
//! cache; the bulk-scrape DB and scanner are gone.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};

pub async fn get_catalog_entry(
    State(state): State<crate::state::AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let cfg_guard = state.config.read().await;
    let models_dir = cfg_guard.resolved_models_dir();
    drop(cfg_guard);

    let entry = if let Some(entry) = mold_catalog::live::companion_entry_for_id(&id) {
        entry
    } else if let Some(version_id) = id.strip_prefix("cv:") {
        match mold_catalog::live::fetch_civitai_version(
            state.catalog_live_civitai_base.as_str(),
            version_id,
            std::env::var("CIVITAI_TOKEN").ok().as_deref(),
        )
        .await
        {
            Ok(e) => e,
            Err(mold_catalog::live::LiveSearchError::Upstream { status: 404, .. }) => {
                return (StatusCode::NOT_FOUND, "not found").into_response();
            }
            Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
        }
    } else if let Some(repo_id) = id.strip_prefix("hf:") {
        match mold_catalog::live::fetch_hf_repo(
            "https://huggingface.co",
            repo_id,
            std::env::var("HF_TOKEN").ok().as_deref(),
        )
        .await
        {
            Ok(e) => e,
            Err(mold_catalog::live::LiveSearchError::Upstream { status: 404, .. }) => {
                return (StatusCode::NOT_FOUND, "not found").into_response();
            }
            Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            "id must be `cv:` or `hf:` prefixed",
        )
            .into_response();
    };

    Json(live_entry_to_wire(&entry, &models_dir)).into_response()
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

fn rendered_primary_recipe_dest(
    entry: &mold_catalog::entry::CatalogEntry,
    author: &str,
    name: &str,
) -> Option<String> {
    entry
        .download_recipe
        .files
        .iter()
        .find(|file| file.role.is_none())
        .or_else(|| entry.download_recipe.files.first())
        .map(|file| {
            mold_catalog::entry::render_recipe_dest(&file.dest, entry.family.as_str(), author, name)
        })
}

pub(crate) async fn enqueue_catalog_primary_repair(
    state: &crate::state::AppState,
    id: &str,
) -> Result<Option<String>, (StatusCode, String)> {
    let models_dir = state.config.read().await.resolved_models_dir();
    if let Some(version_id) = id.strip_prefix("cv:") {
        let sc_path = mold_catalog::sidecar::civitai_sidecar_path(&models_dir, id);
        if let Ok(sidecar) = mold_catalog::sidecar::read_sidecar(&sc_path) {
            if let Some(parent) = sc_path.parent() {
                if mold_catalog::sidecar::primary_path_if_present(parent, &sidecar).is_some() {
                    return Ok(None);
                }
            }
        }

        let entry = mold_catalog::live::fetch_civitai_version(
            state.catalog_live_civitai_base.as_str(),
            version_id,
            std::env::var("CIVITAI_TOKEN").ok().as_deref(),
        )
        .await
        .map_err(|e| match e {
            mold_catalog::live::LiveSearchError::Upstream { status: 404, .. } => (
                StatusCode::NOT_FOUND,
                format!("{id}: upstream returned 404"),
            ),
            other => (StatusCode::BAD_GATEWAY, format!("upstream: {other}")),
        })?;

        let auth = match entry.download_recipe.needs_token {
            Some(mold_catalog::entry::TokenKind::Civitai) => {
                mold_core::download::civitai_auth_or_error(id)
                    .map_err(|e| (StatusCode::UNAUTHORIZED, e.to_string()))?
            }
            _ => mold_core::download::RecipeAuth::None,
        };
        let (author, name) = match entry.source_id.split_once('/') {
            Some((a, n)) => (a.to_string(), n.to_string()),
            None => (String::new(), entry.source_id.clone()),
        };
        let files: Vec<crate::downloads::OwnedRecipeFile> = entry
            .download_recipe
            .files
            .iter()
            .map(|f| crate::downloads::OwnedRecipeFile {
                url: f.url.clone(),
                dest: mold_catalog::entry::render_recipe_dest(
                    &f.dest,
                    entry.family.as_str(),
                    &author,
                    &name,
                ),
                sha256: f.sha256.clone(),
                size_bytes: f.size_bytes,
            })
            .collect();
        if let Some(primary_dest) = rendered_primary_recipe_dest(&entry, &author, &name) {
            let sidecar = mold_catalog::sidecar::sidecar_from_entry(&entry, primary_dest);
            if let Err(e) = mold_catalog::sidecar::write_sidecar(&sc_path, &sidecar) {
                tracing::warn!(
                    target: "catalog.sidecar",
                    catalog_id = %id,
                    error = %e,
                    "sidecar write failed during primary repair",
                );
            }
        }
        let payload = crate::downloads::RecipePayload {
            catalog_id: id.to_string(),
            files,
            auth,
        };
        let (job_id, _, _) = state
            .downloads
            .enqueue_recipe_in_group(payload, id)
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        return Ok(Some(job_id));
    }

    if let Some(repo_id) = id.strip_prefix("hf:") {
        let entry = mold_catalog::live::fetch_hf_repo(
            "https://huggingface.co",
            repo_id,
            std::env::var("HF_TOKEN").ok().as_deref(),
        )
        .await
        .map_err(|e| match e {
            mold_catalog::live::LiveSearchError::Upstream { status: 404, .. } => (
                StatusCode::NOT_FOUND,
                format!("{id}: upstream returned 404"),
            ),
            other => (StatusCode::BAD_GATEWAY, format!("upstream: {other}")),
        })?;
        let model = match mold_core::manifest::find_manifest(&entry.source_id) {
            Some(m) => m.name.clone(),
            None => entry.source_id.clone(),
        };
        let (job_id, _, _) = state
            .downloads
            .enqueue_in_group(model, id)
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        return Ok(Some(job_id));
    }

    Ok(None)
}

pub async fn list_families(State(_state): State<crate::state::AppState>) -> impl IntoResponse {
    // Live search hits one family per request, so the sidebar just gets
    // the static taxonomy. No per-family counts — that line is gone from
    // the SPA too.
    use mold_catalog::families::{Family, ALL_FAMILIES};
    let merged: Vec<serde_json::Value> = ALL_FAMILIES
        .iter()
        .map(|f: &Family| serde_json::json!({ "family": f.as_str() }))
        .collect();
    Json(serde_json::json!({ "families": merged })).into_response()
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

/// Enqueue any companions in `companion_names` that are not already
/// fully present under `models_dir`. Returns the per-companion job ids
/// in the order they appear in the catalog entry.
///
/// On-disk presence + name resolution are delegated to
/// `mold_core::download::missing_companions`, which the CLI also
/// consumes for `mold pull cv:<id>`'s companion-first ordering.
///
/// `DownloadQueue::enqueue` is idempotent against canonical manifest
/// names: re-enqueuing a companion that's mid-flight returns the existing
/// job id so concurrent download requests won't double-pull. That lets
/// the on-disk presence check fall back on "missing → enqueue" without a
/// race condition.
pub(crate) async fn enqueue_missing_companions(
    companion_names: &[String],
    models_dir: &std::path::Path,
    queue: &crate::downloads::DownloadQueue,
    catalog_id: Option<&str>,
) -> Vec<CompanionJob> {
    let manifests = mold_core::download::missing_companions(companion_names, models_dir);
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
    // Live single-id lookup — replaces the bulk-scrape DB read.
    if let Some(companion_name) = mold_catalog::live::companion_name_for_catalog_id(&id) {
        let models_dir = state.config.read().await.resolved_models_dir();
        let companion_jobs = enqueue_missing_companions(
            &[companion_name.to_string()],
            &models_dir,
            &state.downloads,
            Some(&id),
        )
        .await;
        let primary_job_id = if companion_jobs.is_empty() {
            None
        } else {
            Some(companion_jobs[0].job_id.clone())
        };
        return (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "primary_job_id": primary_job_id,
                "companion_jobs": [],
            })),
        )
            .into_response();
    }

    let entry = if let Some(version_id) = id.strip_prefix("cv:") {
        match mold_catalog::live::fetch_civitai_version(
            state.catalog_live_civitai_base.as_str(),
            version_id,
            std::env::var("CIVITAI_TOKEN").ok().as_deref(),
        )
        .await
        {
            Ok(e) => e,
            Err(mold_catalog::live::LiveSearchError::Upstream { status: 404, .. }) => {
                return (StatusCode::NOT_FOUND, "unknown catalog id").into_response();
            }
            Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
        }
    } else if let Some(repo_id) = id.strip_prefix("hf:") {
        match mold_catalog::live::fetch_hf_repo(
            "https://huggingface.co",
            repo_id,
            std::env::var("HF_TOKEN").ok().as_deref(),
        )
        .await
        {
            Ok(e) => e,
            Err(mold_catalog::live::LiveSearchError::Upstream { status: 404, .. }) => {
                return (StatusCode::NOT_FOUND, "unknown catalog id").into_response();
            }
            Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response(),
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            "id must be `cv:` or `hf:` prefixed",
        )
            .into_response();
    };

    if entry.engine_phase >= 6 {
        return (
            StatusCode::CONFLICT,
            format!(
                "engine_phase {} not yet supported by this build — see release notes",
                entry.engine_phase
            ),
        )
            .into_response();
    }

    // Phase 2: companions auto-pull before the primary entry. Civitai
    // single-file checkpoints commonly strip their text encoders + VAE,
    // so the catalog records `companions: ["clip-l", ...]` on those
    // entries. Each canonical companion has a hidden synthetic manifest
    // in `mold-core` so the queue can enqueue them by name.
    let models_dir = state.config.read().await.resolved_models_dir();
    let entry_id = entry.id.as_str().to_string();
    let companion_jobs = enqueue_missing_companions(
        &entry.companions,
        &models_dir,
        &state.downloads,
        Some(&entry_id),
    )
    .await;

    // Primary entry. HF rows map onto a manifest model name (best-effort —
    // diffusers entries with a recognised source_id flow through; the rest
    // surface a `null` primary while companion downloads still run). `cv:`
    // rows go through the recipe path: resolve `CIVITAI_TOKEN` if
    // `needs_token: Civitai`, and call `DownloadQueue::enqueue_recipe`.
    use mold_catalog::entry::Source;
    let primary_job_id: Option<String> = match entry.source {
        Source::Hf => {
            let model = match mold_core::manifest::find_manifest(&entry.source_id) {
                Some(m) => m.name.clone(),
                None => entry.source_id.clone(),
            };
            match state.downloads.enqueue_in_group(model, &entry_id).await {
                Ok((jid, _, _)) => Some(jid),
                Err(crate::downloads::EnqueueError::UnknownModel(_)) => None,
                Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
            }
        }
        Source::Civitai => {
            let auth = match entry.download_recipe.needs_token {
                Some(mold_catalog::entry::TokenKind::Civitai) => {
                    match mold_core::download::civitai_auth_or_error(&entry_id) {
                        Ok(a) => a,
                        Err(e) => {
                            return (StatusCode::UNAUTHORIZED, e.to_string()).into_response();
                        }
                    }
                }
                _ => mold_core::download::RecipeAuth::None,
            };
            let (author, name) = match entry.source_id.split_once('/') {
                Some((a, n)) => (a.to_string(), n.to_string()),
                None => (String::new(), entry.source_id.clone()),
            };
            let files: Vec<crate::downloads::OwnedRecipeFile> = entry
                .download_recipe
                .files
                .iter()
                .map(|f| crate::downloads::OwnedRecipeFile {
                    url: f.url.clone(),
                    dest: mold_catalog::entry::render_recipe_dest(
                        &f.dest,
                        entry.family.as_str(),
                        &author,
                        &name,
                    ),
                    sha256: f.sha256.clone(),
                    size_bytes: f.size_bytes,
                })
                .collect();
            // Sidecar write is best-effort: it lets the LoRA picker see
            // the entry as soon as the file lands without re-querying
            // the live catalog. Failure is not fatal.
            if let Some(primary_dest) = rendered_primary_recipe_dest(&entry, &author, &name) {
                let sidecar = mold_catalog::sidecar::sidecar_from_entry(&entry, primary_dest);
                let sc_path = mold_catalog::sidecar::civitai_sidecar_path(&models_dir, &entry_id);
                if let Err(e) = mold_catalog::sidecar::write_sidecar(&sc_path, &sidecar) {
                    tracing::warn!(
                        target: "catalog.sidecar",
                        catalog_id = %entry_id,
                        error = %e,
                        "sidecar write failed; picker will need a reinstall to surface this row",
                    );
                }
            }
            let payload = crate::downloads::RecipePayload {
                catalog_id: entry_id.clone(),
                files,
                auth,
            };
            match state
                .downloads
                .enqueue_recipe_in_group(payload, &entry_id)
                .await
            {
                Ok((jid, _, _)) => Some(jid),
                Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
            }
        }
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
        include_nsfw: q.include_nsfw.unwrap_or(true),
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

#[derive(Debug, serde::Deserialize)]
pub struct ListLorasQuery {
    pub model: Option<String>,
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
    let compatible_lora_families = match kind_filter.as_deref() {
        Some("lora") => family_filter.as_deref().map(compatible_lora_families),
        _ => None,
    };

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
            let family_matches = compatible_lora_families
                .as_ref()
                .is_some_and(|families| families.contains(&sidecar.family))
                || sidecar.family == f;
            if !family_matches {
                continue;
            }
        }
        let abs = mold_catalog::sidecar::primary_path_if_present(&dir, &sidecar);
        let installed = abs.is_some();
        let primary_path = abs.map(|p| p.to_string_lossy().into_owned());
        wire.push(sidecar_to_wire(sidecar, installed, primary_path));
    }
    let total = wire.len() as i64;
    Json(mold_core::catalog_wire::InstalledCatalogResponse {
        entries: wire,
        page: 1,
        page_size: total,
        total,
    })
    .into_response()
}

/// `GET /api/loras` — list installed LoRA adapters, optionally filtered
/// by model compatibility. This is a small MCP-friendly view over the
/// catalog sidecars used by the web picker.
#[utoipa::path(
    get,
    path = "/api/loras",
    tag = "models",
    params(
        ("model" = Option<String>, Query, description = "Optional model name used to filter LoRAs by compatible family")
    ),
    responses(
        (status = 200, description = "Installed LoRAs", body = Vec<mold_core::LoraInfo>),
        (status = 400, description = "Unknown model")
    )
)]
pub async fn list_loras(
    State(state): State<crate::state::AppState>,
    Query(q): Query<ListLorasQuery>,
) -> Result<Json<Vec<mold_core::LoraInfo>>, crate::routes::ApiError> {
    let family_filter = match q.model.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(model) => {
            let family = lora_family_for_model(&state, model).await.ok_or_else(|| {
                crate::routes::ApiError::unknown_model(format!(
                    "unknown model '{model}'; cannot resolve compatible LoRAs"
                ))
            })?;
            if !mold_core::family_supports_lora(&family) {
                return Ok(Json(Vec::new()));
            }
            Some(compatible_lora_families(&family))
        }
        None => None,
    };

    let cfg = state.config.read().await;
    let models_dir = cfg.resolved_models_dir();
    drop(cfg);

    let mut loras = mold_catalog::sidecar::walk_sidecars(&models_dir)
        .into_iter()
        .filter_map(|(dir, sidecar)| {
            if sidecar.kind != "lora" {
                return None;
            }
            if let Some(families) = family_filter.as_ref() {
                if !families.contains(&sidecar.family) {
                    return None;
                }
            }
            sidecar_to_lora_info(&dir, sidecar)
        })
        .collect::<Vec<_>>();

    loras.sort_by(|a, b| {
        b.added_at
            .cmp(&a.added_at)
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(Json(loras))
}

fn compatible_lora_families(family: &str) -> Vec<String> {
    match family {
        "qwen-image-edit" | "qwen_image_edit" => {
            vec!["qwen-image".to_string(), "qwen-image-edit".to_string()]
        }
        other => vec![other.to_string()],
    }
}

async fn lora_family_for_model(state: &crate::state::AppState, model: &str) -> Option<String> {
    let canonical = mold_core::manifest::resolve_model_name(model);
    if let Some(manifest) = mold_core::manifest::find_manifest(&canonical) {
        return Some(manifest.family.clone());
    }
    if let Some(manifest) = mold_core::manifest::find_manifest(model) {
        return Some(manifest.family.clone());
    }

    {
        let intents = state.catalog_intents.read().await;
        if let Some(intent) = intents.get(model).or_else(|| intents.get(&canonical)) {
            return Some(intent.family.clone());
        }
    }

    let config = state.config.read().await;
    let configured = config
        .models
        .get(model)
        .or_else(|| config.models.get(&canonical))
        .and_then(|m| m.family.clone());
    if configured.is_some() {
        return configured;
    }
    let models_dir = config.resolved_models_dir();
    drop(config);

    if model.starts_with("cv:") || model.starts_with("hf:") {
        for (_, sidecar) in mold_catalog::sidecar::walk_sidecars(&models_dir) {
            if sidecar.id == model {
                return Some(sidecar.family);
            }
        }
    }

    None
}

fn sidecar_to_lora_info(
    dir: &std::path::Path,
    sidecar: mold_catalog::sidecar::CatalogSidecar,
) -> Option<mold_core::LoraInfo> {
    let path = mold_catalog::sidecar::primary_path_if_present(dir, &sidecar)?;
    Some(mold_core::LoraInfo {
        id: sidecar.id,
        name: sidecar.name,
        family: sidecar.family,
        author: sidecar.author,
        path: path.to_string_lossy().into_owned(),
        trained_words: sidecar.trained_words,
        size_bytes: sidecar.size_bytes,
        thumbnail_url: sidecar.thumbnail_url,
        added_at: sidecar.written_at,
    })
}

fn parse_kind(s: &str) -> Option<mold_catalog::entry::Kind> {
    use mold_catalog::entry::Kind::*;
    Some(match s {
        "checkpoint" => Checkpoint,
        "lora" => Lora,
        "vae" => Vae,
        "text-encoder" => TextEncoder,
        "tokenizer" => Tokenizer,
        "clip" => Clip,
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
    let (installed, primary_path) = if let Some(companion_name) =
        mold_catalog::live::companion_name_for_catalog_id(entry.id.as_str())
    {
        let installed = mold_core::manifest::find_manifest(companion_name)
            .map(|manifest| mold_core::download::companion_present_on_disk(models_dir, manifest))
            .unwrap_or(false);
        (installed, None)
    } else if matches!(entry.source, mold_catalog::entry::Source::Civitai) {
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
    let companion_details = entry
        .companions
        .iter()
        .filter_map(|name| {
            mold_catalog::companions::COMPANIONS
                .iter()
                .find(|companion| companion.canonical_name == name)
                .map(|companion| {
                    serde_json::json!({
                        "name": companion.canonical_name,
                        "kind": companion.kind,
                        "repo": companion.repo,
                        "size_bytes": companion.size_bytes,
                    })
                })
        })
        .collect::<Vec<_>>();

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
        "companion_details": companion_details,
        "download_recipe": entry.download_recipe,
        "engine_phase": entry.engine_phase,
        "installed": installed,
        "primary_path": primary_path,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "added_at": entry.added_at,
        "trained_words": entry.trained_words,
        "page_url": entry.page_url,
    })
}

pub(crate) fn sidecar_to_wire(
    sc: mold_catalog::sidecar::CatalogSidecar,
    installed: bool,
    primary_path: Option<String>,
) -> mold_core::catalog_wire::InstalledCatalogEntry {
    // The sidecar carries fewer fields than a full CatalogEntryWire on
    // purpose — the picker only needs id/name/family/kind/trained_words/
    // primary_path. Empty defaults are returned for fields the SPA
    // ignores in this code path; the shared wire struct keeps the shape
    // uniform with `/api/catalog/search` so a single TypeScript
    // interface works for both, and keeps the client deserializer in
    // `mold_core::client` from drifting.
    mold_core::catalog_wire::InstalledCatalogEntry {
        id: sc.id,
        source: sc.source,
        source_id: sc.source_id,
        name: sc.name,
        author: sc.author,
        family: sc.family,
        family_role: sc.family_role,
        sub_family: sc.sub_family,
        modality: sc.modality,
        kind: sc.kind,
        file_format: "safetensors".into(),
        bundling: "single-file".into(),
        size_bytes: sc.size_bytes,
        download_count: 0,
        rating: None,
        likes: 0,
        nsfw: false,
        thumbnail_url: sc.thumbnail_url,
        description: None,
        license: None,
        license_flags: None,
        tags: Vec::new(),
        companions: Vec::new(),
        companion_details: Vec::new(),
        download_recipe: mold_core::catalog_wire::DownloadRecipeWire::default(),
        engine_phase: sc.engine_phase,
        installed,
        primary_path,
        created_at: None,
        updated_at: None,
        added_at: sc.written_at,
        trained_words: sc.trained_words,
    }
}

#[cfg(test)]
#[path = "catalog_live_test.rs"]
mod catalog_live_test;
