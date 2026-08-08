//! Bounded-parallel download queue wrapping `mold_core::download::pull_model_with_callback`.
//!
//! One long-running driver task pulls jobs off a `VecDeque`, spawns a pull
//! task per job, forwards `DownloadProgressEvent` → `DownloadEvent` on a
//! broadcast channel, and cleans up on completion or cancellation.
//!
//! This module owns the download queue and progress contract. Do
//! not take a lock on resources (Agent B) or placement (Agent C) from here.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex as StdMutex};

use mold_core::download::RecipeAuth;
use mold_core::types::{DownloadEvent, DownloadJob, DownloadsListing, JobStatus};
use tokio::sync::{broadcast, Mutex as AsyncMutex, Notify};
use tokio_util::sync::CancellationToken;

/// Owned counterpart of [`mold_core::download::RecipeFetchFile`]. The queue
/// stores files until the driver task picks the job up, so we can't carry
/// borrowed lifetimes here.
#[derive(Clone, Debug)]
pub struct OwnedRecipeFile {
    pub url: String,
    pub dest: String,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
}

/// Everything the recipe driver needs to fetch a `cv:<id>` primary
/// safetensors. Stored on the queue separately from `DownloadJob` so the
/// public `DownloadJob` type (broadcast / `/api/downloads` listing) stays
/// unchanged; only the driver dispatch reads it.
#[derive(Clone, Debug)]
pub struct RecipePayload {
    /// Catalog id (e.g. `cv:618692`); used as both the dedup key and the
    /// per-recipe subdir name (sanitized via `replace(':', "-")`).
    pub catalog_id: String,
    pub files: Vec<OwnedRecipeFile>,
    pub auth: RecipeAuth,
}

/// Capacity of the broadcast channel. Slow subscribers see the oldest events
/// lag (we accept this — the SPA will also re-fetch `/api/downloads` on reconnect).
pub const EVENT_BUFFER: usize = 256;

/// Max history entries retained for the drawer's "Recent" section.
pub const HISTORY_CAP: usize = 20;

/// Two simultaneous model pulls make good use of fast remote hosts without
/// multiplying disk pressure for large checkpoints without bound.
pub const DOWNLOAD_CONCURRENCY: usize = 2;

/// Active download handle.
pub struct ActiveHandle {
    pub job: DownloadJob,
    pub abort: CancellationToken,
}

/// Membership ledger for a catalog entry's downloads. Tracks every job
/// (primary + companions) that belongs to a single user-visible catalog
/// id (`cv:1759168`, an HF entry, etc.) so the queue knows when the *whole*
/// entry is ready — not just when one of its jobs finished.
///
/// Why this exists: `catalog_api::post_catalog_download` enqueues companions
/// (CLIP-L/G, VAE) before the primary checkpoint. Pre-C, the SPA refreshed
/// its model list on the primary's `JobDone` event, which sometimes fired
/// before companions were on disk — that's the "model sometimes doesn't
/// show up after download" bug. Now the SPA listens for `CatalogReady`,
/// emitted exactly once when the last job in the group settles.
#[derive(Debug, Clone)]
struct CatalogGroup {
    catalog_id: String,
    /// Every job that must settle before the group is ready.
    job_ids: std::collections::HashSet<String>,
    /// Final status keyed by job id. Group is ready when
    /// `outcomes.len() == job_ids.len()`.
    outcomes: HashMap<String, JobStatus>,
}

pub struct DownloadQueue {
    active: AsyncMutex<HashMap<String, ActiveHandle>>,
    queued: StdMutex<VecDeque<DownloadJob>>,
    history: StdMutex<VecDeque<DownloadJob>>,
    events: broadcast::Sender<DownloadEvent>,
    /// Wakes the driver task when new work arrives.
    notify: Notify,
    /// Recipe payloads indexed by job id. Populated by `enqueue_recipe`,
    /// drained by the driver task before running the job. Manifest jobs
    /// have no entry here and fall through to the `PullDriver` path.
    recipe_payloads: StdMutex<HashMap<String, RecipePayload>>,
    /// Request-scoped HF fallback tokens for manifest jobs initiated by a
    /// trusted remote desktop client. Tokens live only until the job settles
    /// (or is cancelled while queued) and never appear in the public download
    /// listing or event stream.
    hf_fallback_tokens: StdMutex<HashMap<String, String>>,
    /// Catalog group bookkeeping. Keyed by catalog id; cleared when the
    /// group emits `CatalogReady`. Memory-only — server restart resets
    /// the ledger, which is fine because in-flight downloads were lost
    /// too and the next `manifest_files_exist` poll will report any
    /// half-done install as incomplete.
    groups: StdMutex<HashMap<String, CatalogGroup>>,
}

#[derive(Debug, thiserror::Error)]
pub enum EnqueueError {
    #[error(transparent)]
    ModelActivation(#[from] mold_core::ModelActivationError),
    #[error("unknown model '{0}'. Run 'mold list' to see available models.")]
    UnknownModel(String),
    #[error("download queue lock poisoned")]
    LockPoisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// Already in the queue or currently active — no new job created.
    AlreadyPresent,
    /// Created a new job at the given 0-based position (0 = will start next).
    Created,
}

fn require_manifest_enqueue_activation(
    manifest: &mold_core::manifest::ModelManifest,
) -> Result<(), EnqueueError> {
    mold_core::require_model_activation(&manifest.name, Some(&manifest.family))?;
    for file in &manifest.files {
        mold_core::require_model_activation(&file.hf_repo, Some(&manifest.family))?;
        mold_core::require_model_activation(&file.hf_filename, Some(&manifest.family))?;
    }
    Ok(())
}

fn require_recipe_enqueue_activation(payload: &RecipePayload) -> Result<(), EnqueueError> {
    mold_core::require_model_activation(&payload.catalog_id, None)?;
    for file in &payload.files {
        mold_core::require_model_activation(&file.url, None)?;
        mold_core::require_model_activation(&file.dest, None)?;
    }
    Ok(())
}

impl DownloadQueue {
    pub fn new() -> Arc<Self> {
        let (events, _rx) = broadcast::channel(EVENT_BUFFER);
        Arc::new(Self {
            active: AsyncMutex::new(HashMap::new()),
            queued: StdMutex::new(VecDeque::new()),
            history: StdMutex::new(VecDeque::new()),
            events,
            notify: Notify::new(),
            recipe_payloads: StdMutex::new(HashMap::new()),
            hf_fallback_tokens: StdMutex::new(HashMap::new()),
            groups: StdMutex::new(HashMap::new()),
        })
    }

    /// Testing helper — same as `new()` but returns a non-`Arc` for
    /// direct `await` use. Callers that need the driver running must wrap in Arc.
    #[cfg(test)]
    pub fn new_for_test() -> Arc<Self> {
        Self::new()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DownloadEvent> {
        self.events.subscribe()
    }

    /// Returns the current active/queued/history snapshot.
    pub async fn listing(&self) -> DownloadsListing {
        let mut active_jobs: Vec<DownloadJob> = self
            .active
            .lock()
            .await
            .values()
            .map(|handle| handle.job.clone())
            .collect();
        active_jobs.sort_by_key(|job| (job.started_at, job.id.clone()));
        let active = active_jobs.first().cloned();
        let queued: Vec<DownloadJob> = self
            .queued
            .lock()
            .expect("queued lock poisoned")
            .iter()
            .cloned()
            .collect();
        let history: Vec<DownloadJob> = self
            .history
            .lock()
            .expect("history lock poisoned")
            .iter()
            .cloned()
            .collect();
        DownloadsListing {
            active_jobs,
            active,
            queued,
            history,
        }
    }

    /// Enqueue a model for download. Returns `(job_id, position, outcome)`.
    /// Position 0 means the job will run immediately after the driver wakes;
    /// N means N jobs are ahead. Duplicate enqueue returns the existing id.
    pub async fn enqueue(
        &self,
        model: String,
    ) -> Result<(String, usize, EnqueueOutcome), EnqueueError> {
        self.enqueue_with_hf_fallback(model, None).await
    }

    pub async fn enqueue_with_hf_fallback(
        &self,
        model: String,
        hf_fallback_token: Option<String>,
    ) -> Result<(String, usize, EnqueueOutcome), EnqueueError> {
        // Check the caller-supplied identity before resolution so a gated
        // model that has no registered manifest still receives the stable
        // policy error rather than being flattened into UnknownModel.
        mold_core::require_model_activation(&model, None)?;

        // Manifest validation up front so the caller gets a real 400 instead of a
        // background failure.
        let canonical = mold_core::manifest::resolve_model_name(&model);
        let manifest = mold_core::manifest::find_manifest(&canonical)
            .ok_or(EnqueueError::UnknownModel(model))?;
        self.enqueue_resolved_manifest(canonical, manifest, hf_fallback_token)
            .await
    }

    async fn enqueue_resolved_manifest(
        &self,
        canonical: String,
        manifest: &mold_core::manifest::ModelManifest,
        hf_fallback_token: Option<String>,
    ) -> Result<(String, usize, EnqueueOutcome), EnqueueError> {
        // The queue is an independent authority boundary: validate the full
        // resolved manifest, including repos, before locking or mutating any
        // queue state. This closes opaque/renamed-id bypasses even when a
        // higher-level catalog caller misses its own preflight.
        require_manifest_enqueue_activation(manifest)?;

        // Check for active/queued duplicate.
        {
            let active = self.active.lock().await;
            if let Some(handle) = active.values().find(|handle| handle.job.model == canonical) {
                self.store_hf_fallback_token(&handle.job.id, hf_fallback_token.as_deref())?;
                return Ok((handle.job.id.clone(), 0, EnqueueOutcome::AlreadyPresent));
            }
        }
        {
            let queued = self.queued.lock().map_err(|_| EnqueueError::LockPoisoned)?;
            if let Some((idx, existing)) = queued
                .iter()
                .enumerate()
                .find(|(_, j)| j.model == canonical)
            {
                self.store_hf_fallback_token(&existing.id, hf_fallback_token.as_deref())?;
                return Ok((existing.id.clone(), idx + 1, EnqueueOutcome::AlreadyPresent));
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let job = DownloadJob {
            id: id.clone(),
            model: canonical.clone(),
            catalog_id: None,
            status: JobStatus::Queued,
            files_done: 0,
            files_total: 0,
            bytes_done: 0,
            bytes_total: 0,
            current_file: None,
            started_at: None,
            completed_at: None,
            error: None,
        };

        let position = {
            let mut queued = self.queued.lock().map_err(|_| EnqueueError::LockPoisoned)?;
            queued.push_back(job);
            queued.len() // position shown to the user is 1-based in the drawer
        };
        self.store_hf_fallback_token(&id, hf_fallback_token.as_deref())?;
        let _ = self.events.send(DownloadEvent::Enqueued {
            id: id.clone(),
            model: canonical,
            position,
        });
        self.notify.notify_one();
        Ok((id, position, EnqueueOutcome::Created))
    }

    /// Enqueue a recipe-driven download (Civitai single-file checkpoint).
    /// Returns `(job_id, position, outcome)`. Dedupes on `payload.catalog_id`
    /// — re-enqueuing the same recipe returns the existing job id so the
    /// SPA's idempotent retry doesn't double-pull.
    ///
    /// Recipe jobs use the catalog id (e.g. `cv:618692`) as `DownloadJob.model`
    /// so the drawer / API listing show something meaningful. The driver task
    /// looks up the recipe payload by job id and dispatches to the recipe
    /// driver instead of the manifest driver.
    pub async fn enqueue_recipe(
        &self,
        payload: RecipePayload,
    ) -> Result<(String, usize, EnqueueOutcome), EnqueueError> {
        require_recipe_enqueue_activation(&payload)?;
        if payload.catalog_id.trim().is_empty() {
            return Err(EnqueueError::UnknownModel(payload.catalog_id));
        }

        // Dedup: active or queued recipe with the same catalog id?
        {
            let active = self.active.lock().await;
            if let Some(handle) = active
                .values()
                .find(|handle| handle.job.model == payload.catalog_id)
            {
                return Ok((handle.job.id.clone(), 0, EnqueueOutcome::AlreadyPresent));
            }
        }
        {
            let queued = self.queued.lock().map_err(|_| EnqueueError::LockPoisoned)?;
            if let Some((idx, existing)) = queued
                .iter()
                .enumerate()
                .find(|(_, j)| j.model == payload.catalog_id)
            {
                return Ok((existing.id.clone(), idx + 1, EnqueueOutcome::AlreadyPresent));
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let job = DownloadJob {
            id: id.clone(),
            model: payload.catalog_id.clone(),
            catalog_id: Some(payload.catalog_id.clone()),
            status: JobStatus::Queued,
            files_done: 0,
            files_total: payload.files.len(),
            bytes_done: 0,
            bytes_total: payload.files.iter().filter_map(|f| f.size_bytes).sum(),
            current_file: None,
            started_at: None,
            completed_at: None,
            error: None,
        };

        // Stash the recipe payload before pushing the job so the driver
        // can never observe a queued job whose payload hasn't arrived yet.
        {
            let mut payloads = self
                .recipe_payloads
                .lock()
                .map_err(|_| EnqueueError::LockPoisoned)?;
            payloads.insert(id.clone(), payload.clone());
        }

        let position = {
            let mut queued = self.queued.lock().map_err(|_| EnqueueError::LockPoisoned)?;
            queued.push_back(job);
            queued.len()
        };
        let _ = self.events.send(DownloadEvent::Enqueued {
            id: id.clone(),
            model: payload.catalog_id,
            position,
        });
        self.notify.notify_one();
        Ok((id, position, EnqueueOutcome::Created))
    }

    /// Enqueue a manifest model and bind the resulting job to a catalog
    /// group. Same return shape as [`Self::enqueue`]; an `AlreadyPresent`
    /// outcome still registers the existing job into the group, so a
    /// catalog-driven enqueue that races with a manual `mold pull` of the
    /// same companion still gets its membership tracked correctly.
    pub async fn enqueue_in_group(
        &self,
        model: String,
        catalog_id: &str,
    ) -> Result<(String, usize, EnqueueOutcome), EnqueueError> {
        let (id, pos, outcome) = self.enqueue(model).await?;
        self.register_in_group(catalog_id, &id);
        Ok((id, pos, outcome))
    }

    pub async fn enqueue_in_group_with_hf_fallback(
        &self,
        model: String,
        catalog_id: &str,
        hf_fallback_token: Option<String>,
    ) -> Result<(String, usize, EnqueueOutcome), EnqueueError> {
        let (id, pos, outcome) = self
            .enqueue_with_hf_fallback(model, hf_fallback_token)
            .await?;
        self.register_in_group(catalog_id, &id);
        Ok((id, pos, outcome))
    }

    /// Recipe-pull variant of [`Self::enqueue_in_group`]. Catalog id is
    /// taken from the payload, so callers don't pass it twice.
    pub async fn enqueue_recipe_in_group(
        &self,
        payload: RecipePayload,
        catalog_id: &str,
    ) -> Result<(String, usize, EnqueueOutcome), EnqueueError> {
        let (id, pos, outcome) = self.enqueue_recipe(payload).await?;
        self.register_in_group(catalog_id, &id);
        Ok((id, pos, outcome))
    }

    fn register_in_group(&self, catalog_id: &str, job_id: &str) {
        self.tag_job_with_catalog_id(catalog_id, job_id);
        let mut groups = match self.groups.lock() {
            Ok(g) => g,
            Err(_) => return, // poisoned — best effort, the broadcast still works
        };
        let entry = groups
            .entry(catalog_id.to_string())
            .or_insert_with(|| CatalogGroup {
                catalog_id: catalog_id.to_string(),
                job_ids: std::collections::HashSet::new(),
                outcomes: HashMap::new(),
            });
        entry.job_ids.insert(job_id.to_string());
    }

    fn tag_job_with_catalog_id(&self, catalog_id: &str, job_id: &str) {
        if let Ok(mut queued) = self.queued.lock() {
            if let Some(job) = queued.iter_mut().find(|job| job.id == job_id) {
                job.catalog_id = Some(catalog_id.to_string());
                return;
            }
        }
        if let Ok(mut history) = self.history.lock() {
            if let Some(job) = history.iter_mut().find(|job| job.id == job_id) {
                job.catalog_id = Some(catalog_id.to_string());
                return;
            }
        }
        if let Ok(mut active) = self.active.try_lock() {
            if let Some(handle) = active.get_mut(job_id) {
                handle.job.catalog_id = Some(catalog_id.to_string());
            }
        }
    }

    /// Record a job's terminal status against any catalog group it belongs
    /// to. When the group's outcome map fills (`outcomes.len() ==
    /// job_ids.len()`), emit `CatalogReady { ok: all_completed }` and
    /// drop the group from the ledger. Idempotent: a job not in any
    /// group is a no-op.
    ///
    /// Called from `try_pull_with_retry` after each terminal arm
    /// (Completed / Failed / Cancelled).
    pub(crate) fn settle_for_group(&self, job_id: &str, status: JobStatus) {
        // Two-phase to avoid emitting under the lock.
        let to_emit = {
            let mut groups = match self.groups.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            let mut emit_for: Option<(String, bool)> = None;
            // O(groups). In practice catalogs queue 2–4 jobs each and
            // there's rarely more than a handful of in-flight catalog
            // entries — keeping the index direct rather than building a
            // job_id → catalog_id reverse map.
            for group in groups.values_mut() {
                if !group.job_ids.contains(job_id) {
                    continue;
                }
                group.outcomes.insert(job_id.to_string(), status);
                if group.outcomes.len() == group.job_ids.len() {
                    let ok = group
                        .outcomes
                        .values()
                        .all(|s| matches!(s, JobStatus::Completed));
                    emit_for = Some((group.catalog_id.clone(), ok));
                }
                break; // a job belongs to at most one group
            }
            if let Some((id, _)) = &emit_for {
                groups.remove(id);
            }
            emit_for
        };
        if let Some((id, ok)) = to_emit {
            self.emit(DownloadEvent::CatalogReady { id, ok });
        }
    }

    /// Take ownership of the recipe payload registered for a job, if any.
    /// The driver task calls this before running a job to determine which
    /// driver to dispatch.
    pub(crate) fn take_recipe_payload(&self, job_id: &str) -> Option<RecipePayload> {
        self.recipe_payloads
            .lock()
            .ok()
            .and_then(|mut p| p.remove(job_id))
    }

    fn store_hf_fallback_token(
        &self,
        job_id: &str,
        token: Option<&str>,
    ) -> Result<(), EnqueueError> {
        let Some(token) = token.map(str::trim).filter(|token| !token.is_empty()) else {
            return Ok(());
        };
        self.hf_fallback_tokens
            .lock()
            .map_err(|_| EnqueueError::LockPoisoned)?
            .insert(job_id.to_string(), token.to_string());
        Ok(())
    }

    pub(crate) fn hf_fallback_token(&self, job_id: &str) -> Option<String> {
        self.hf_fallback_tokens
            .lock()
            .ok()
            .and_then(|tokens| tokens.get(job_id).cloned())
    }

    fn forget_hf_fallback_token(&self, job_id: &str) {
        if let Ok(mut tokens) = self.hf_fallback_tokens.lock() {
            tokens.remove(job_id);
        }
    }

    /// Cancel an in-flight or queued download. Returns `true` if a job was
    /// found and cancelled; `false` if the id is unknown.
    pub async fn cancel(&self, id: &str) -> bool {
        // Check queued first — cheap.
        {
            let mut queued = self.queued.lock().expect("queued lock poisoned");
            if let Some(pos) = queued.iter().position(|j| j.id == id) {
                let mut job = queued.remove(pos).expect("queued position must exist");
                drop(queued);
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(now_ms());
                self.recipe_payloads
                    .lock()
                    .expect("recipe payload lock poisoned")
                    .remove(id);
                self.hf_fallback_tokens
                    .lock()
                    .expect("HF fallback token lock poisoned")
                    .remove(id);
                self.push_history(job);
                let _ = self
                    .events
                    .send(DownloadEvent::JobCancelled { id: id.to_string() });
                let _ = self
                    .events
                    .send(DownloadEvent::Dequeued { id: id.to_string() });
                self.settle_for_group(id, JobStatus::Cancelled);
                return true;
            }
        }
        // Active?
        let active = self.active.lock().await;
        if let Some(handle) = active.get(id) {
            handle.abort.cancel();
            // Driver task will observe the cancel, emit JobCancelled, and clear the job.
            return true;
        }
        // Not found.
        false
    }

    /// Called by the driver task to push a finished job into history and keep
    /// the most recent 20.
    pub(crate) fn push_history(&self, job: DownloadJob) {
        let mut history = self.history.lock().expect("history lock poisoned");
        if history.len() >= HISTORY_CAP {
            history.pop_front();
        }
        history.push_back(job);
    }

    /// Wait until either new work is notified or `shutdown` fires.
    pub(crate) async fn wait_for_work(&self, shutdown: &CancellationToken) {
        tokio::select! {
            _ = self.notify.notified() => {},
            _ = shutdown.cancelled() => {},
        }
    }

    pub(crate) fn take_next_queued(&self) -> Option<DownloadJob> {
        self.queued
            .lock()
            .expect("queued lock poisoned")
            .pop_front()
    }

    pub(crate) async fn set_active(&self, handle: ActiveHandle) {
        self.active
            .lock()
            .await
            .insert(handle.job.id.clone(), handle);
    }

    pub(crate) async fn clear_active(&self, id: &str) -> Option<ActiveHandle> {
        self.active.lock().await.remove(id)
    }

    pub(crate) async fn with_active<F, R>(&self, id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&mut DownloadJob) -> R,
    {
        let mut active = self.active.lock().await;
        active.get_mut(id).map(|a| f(&mut a.job))
    }

    async fn cancel_all_active(&self) {
        for handle in self.active.lock().await.values() {
            handle.abort.cancel();
        }
    }

    pub(crate) fn emit(&self, event: DownloadEvent) {
        let _ = self.events.send(event);
    }
}

// ── PullDriver trait + real & test implementations ──────────────────────────

/// Late-readable request credential for an active manifest pull. The queue
/// keeps the token out of public job state while allowing a duplicate desktop
/// request to attach a valid credential even after the pull attempt began.
pub type HfFallbackTokenProvider = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// Trait that hides the HuggingFace pull behind something the tests can fake.
///
/// The real implementation in `HfPullDriver` calls
/// `mold_core::download::pull_and_configure_with_callback`. Tests inject a stub.
#[async_trait::async_trait]
pub trait PullDriver: Send + Sync + 'static {
    async fn pull(
        &self,
        model: &str,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String>;

    /// Pull with a credential provider that can be updated while the request
    /// is active. Existing third-party drivers retain source compatibility via
    /// this default adapter; the production HF driver overrides it.
    async fn pull_with_token_provider(
        &self,
        model: &str,
        hf_fallback_token: HfFallbackTokenProvider,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        let _ = hf_fallback_token;
        self.pull(model, on_progress, cancel).await
    }
}

/// Trait that hides the recipe-driven (Civitai) pull. Parallel to
/// `PullDriver` so the existing 6 manifest-driver implementations don't
/// need to change. The driver task picks between them based on whether
/// the queue stored a recipe payload for the job id.
#[async_trait::async_trait]
pub trait RecipePullDriver: Send + Sync + 'static {
    async fn pull(
        &self,
        payload: RecipePayload,
        models_dir: std::path::PathBuf,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String>;
}

/// Production recipe driver — calls `mold_core::download::fetch_recipe`.
pub struct CivitaiRecipeDriver;

#[async_trait::async_trait]
impl RecipePullDriver for CivitaiRecipeDriver {
    async fn pull(
        &self,
        payload: RecipePayload,
        models_dir: std::path::PathBuf,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        let on_progress: mold_core::download::DownloadProgressCallback =
            std::sync::Arc::from(on_progress);
        let opts = mold_core::download::PullOptions::default();
        let files: Vec<mold_core::download::RecipeFetchFile<'_>> = payload
            .files
            .iter()
            .map(|f| mold_core::download::RecipeFetchFile {
                url: f.url.as_str(),
                dest: f.dest.as_str(),
                sha256: f.sha256.as_deref(),
                size_bytes: f.size_bytes,
            })
            .collect();
        tokio::select! {
            res = mold_core::download::fetch_recipe(
                &payload.catalog_id,
                &files,
                payload.auth.clone(),
                &models_dir,
                Some(on_progress),
                &opts,
            ) => res.map(|_| ()).map_err(|e| e.to_string()),
            _ = cancel.cancelled() => Err("cancelled".into()),
        }
    }
}

/// Production driver — wraps the real HF pull.
pub struct HfPullDriver;

#[async_trait::async_trait]
impl PullDriver for HfPullDriver {
    async fn pull(
        &self,
        model: &str,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        self.pull_with_token_provider(model, Arc::new(|| None), on_progress, cancel)
            .await
    }

    async fn pull_with_token_provider(
        &self,
        model: &str,
        hf_fallback_token: HfFallbackTokenProvider,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        let on_progress: mold_core::download::DownloadProgressCallback =
            std::sync::Arc::from(on_progress);
        let model = model.to_string();
        let pull = async {
            let primary = mold_core::download::pull_and_configure_with_callback(
                &model,
                on_progress.clone(),
                &Default::default(),
            )
            .await;
            match primary {
                Err(
                    mold_core::download::DownloadError::Unauthorized { .. }
                    | mold_core::download::DownloadError::GatedModel { .. },
                ) => {
                    if let Some(token) = hf_fallback_token() {
                        mold_core::download::pull_and_configure_with_callback_and_hf_token(
                            &model,
                            on_progress,
                            &Default::default(),
                            Some(&token),
                        )
                        .await
                    } else {
                        primary
                    }
                }
                result => result,
            }
        };
        // Race the pull against cancellation. pull_and_configure_with_callback
        // is not cancel-aware internally, but dropping its future on cancel
        // aborts the underlying tokio-based HTTP calls.
        tokio::select! {
            res = pull => {
                res.map(|_| ()).map_err(|e| e.to_string())
            }
            _ = cancel.cancelled() => {
                // Terminal cleanup (markers + partials) happens in
                // `try_pull_with_retry` so it covers both cancel and failure
                // paths uniformly. Just surface the cancel error here.
                Err("cancelled".into())
            }
        }
    }
}

/// Delete partial files under `<models_dir>/<sanitized>/` for a cancelled or
/// failed pull. Preserves any file that has a sibling `<file>.sha256-verified`
/// marker (and the marker itself) — those represent fully-downloaded,
/// integrity-checked content that should survive a retry/cancel cycle. Also
/// leaves the HF cache under `~/.cache/huggingface` intact so retry is cheap.
fn cleanup_partials_for_model(model: &str) {
    use mold_core::manifest::resolve_model_name;
    let canonical = resolve_model_name(model);
    let sanitized = canonical.replace(':', "-");
    // Remove `.pulling` marker first so `has_pulling_marker` returns false.
    mold_core::download::remove_pulling_marker(&canonical);
    #[cfg(test)]
    test_hooks::record_cleanup(&canonical);
    let target = mold_core::Config::load_or_default()
        .resolved_models_dir()
        .join(&sanitized);
    cleanup_partials_in_dir(&target);
}

/// Test seam: lets the test module observe which models `cleanup_partials_for_model`
/// has been called with, without having to mutate process-wide `MOLD_MODELS_DIR`
/// (which races against other tests in the same binary).
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::sync::Mutex;
    static CLEANUPS: Mutex<Vec<String>> = Mutex::new(Vec::new());

    pub(crate) fn record_cleanup(model: &str) {
        if let Ok(mut v) = CLEANUPS.lock() {
            v.push(model.to_string());
        }
    }

    /// Drain and return every model name `cleanup_partials_for_model` has
    /// recorded since the last call. Tests should snapshot this before and
    /// after their exercise to get a stable comparison.
    pub fn drain_cleanups() -> Vec<String> {
        CLEANUPS
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default()
    }
}

/// Marker suffix written after a successful SHA-256 verification.
/// A file `foo.safetensors` is "verified" when `foo.safetensors.sha256-verified`
/// exists next to it. Sourced from `mold-core` so the constant has a single
/// definition shared across the pull writer and cleanup reader.
use mold_core::download::SHA256_VERIFIED_SUFFIX;

/// Walk `dir` and delete every regular file that does NOT have a sibling
/// `<file>.sha256-verified` marker. Files that are themselves `*.sha256-verified`
/// markers are preserved. Best-effort — I/O errors are swallowed because cleanup
/// runs on terminal paths where we can't recover anyway.
///
/// If `dir` does not exist or is not a directory, this is a no-op. Subdirectories
/// are descended into and pruned if they end up empty.
///
/// **Safety:** symlinks (file or directory) are always skipped. Deleting follows
/// `entry.path().symlink_metadata()` rather than `entry.file_type()` so a
/// symlink planted inside the models dir can never cause recursion or deletion
/// outside the tree. Before removing a file we also canonicalize the path and
/// assert it's still under `dir`'s canonical root as a belt-and-braces check.
pub(crate) fn cleanup_partials_in_dir(dir: &std::path::Path) {
    // Resolve the intended root once. If this fails the dir doesn't exist or
    // isn't accessible — treat as no-op.
    let root = match dir.canonicalize() {
        Ok(p) => p,
        Err(_) => return,
    };
    cleanup_partials_in_dir_under_root(dir, &root);
}

fn cleanup_partials_in_dir_under_root(dir: &std::path::Path, root: &std::path::Path) {
    if !dir.is_dir() {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();

        // Use symlink_metadata (does NOT follow symlinks) to detect symlinks
        // before we decide whether to recurse or delete. A symlink
        // (file or directory) is always skipped — we don't chase
        // hand-placed pointers out of the models dir.
        let meta = match path.symlink_metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let file_type = meta.file_type();
        if file_type.is_symlink() {
            tracing::warn!(
                path = %path.display(),
                "cleanup_partials: skipping symlink (not following)",
            );
            continue;
        }
        if file_type.is_dir() {
            cleanup_partials_in_dir_under_root(&path, root);
            // Prune the directory if it's now empty.
            let _ = std::fs::remove_dir(&path);
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        // Keep marker files themselves.
        if name.ends_with(SHA256_VERIFIED_SUFFIX) {
            continue;
        }
        // Keep any file that has a sibling `<file>.sha256-verified` marker.
        let marker = path.with_file_name(format!("{name}{SHA256_VERIFIED_SUFFIX}"));
        if marker.exists() {
            continue;
        }
        // Belt-and-braces: canonicalize the target and refuse to delete
        // anything that escapes the models-dir root. symlink_metadata already
        // prevents symlink traversal, but this catches pathological cases
        // (e.g. mount binds) before we hand the path to std::fs::remove_file.
        match path.canonicalize() {
            Ok(canon) => {
                if !canon.starts_with(root) {
                    tracing::warn!(
                        path = %path.display(),
                        canon = %canon.display(),
                        root = %root.display(),
                        "cleanup_partials: refusing to delete file outside models dir",
                    );
                    continue;
                }
            }
            Err(_) => continue,
        }
        let _ = std::fs::remove_file(&path);
    }
}

/// Spawn the bounded-parallel driver. Returns the JoinHandle; drop or await
/// to stop the driver (combined with `shutdown.cancel()`).
pub fn spawn_driver(
    queue: Arc<DownloadQueue>,
    driver: Arc<dyn PullDriver>,
    recipe_driver: Arc<dyn RecipePullDriver>,
    models_dir: std::path::PathBuf,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        tracing::info!("download queue driver started");
        let mut running = tokio::task::JoinSet::new();
        loop {
            if shutdown.is_cancelled() {
                queue.cancel_all_active().await;
                while running.join_next().await.is_some() {}
                break;
            }
            while running.len() < DOWNLOAD_CONCURRENCY {
                let Some(job) = queue.take_next_queued() else {
                    break;
                };
                let queue = queue.clone();
                let driver = driver.clone();
                let recipe_driver = recipe_driver.clone();
                let models_dir = models_dir.clone();
                running.spawn(async move {
                    run_one_job(&queue, &driver, &recipe_driver, &models_dir, job).await;
                });
            }

            if running.is_empty() {
                queue.wait_for_work(&shutdown).await;
            } else {
                tokio::select! {
                    _ = queue.notify.notified() => {},
                    _ = running.join_next() => {},
                    _ = shutdown.cancelled() => {},
                }
            }
        }
        tracing::info!("download queue driver exiting");
    })
}

async fn run_one_job(
    queue: &Arc<DownloadQueue>,
    driver: &Arc<dyn PullDriver>,
    recipe_driver: &Arc<dyn RecipePullDriver>,
    models_dir: &std::path::Path,
    mut job: DownloadJob,
) {
    job.status = JobStatus::Active;
    job.started_at = Some(now_ms());
    let cancel = CancellationToken::new();
    let handle_job = job.clone();

    // Pull the recipe payload off the queue once, before the retry loop.
    // Subsequent retries clone from this owned copy.
    let recipe_payload = queue.take_recipe_payload(&job.id);
    let queue_for_token = queue.clone();
    let token_job_id = job.id.clone();
    let hf_fallback_token: HfFallbackTokenProvider =
        Arc::new(move || queue_for_token.hf_fallback_token(&token_job_id));

    // Install the job as active. The pull runs inline in this function so we
    // don't need a separate task handle — cancellation flows through `abort`.
    let active_handle = ActiveHandle {
        job: handle_job,
        abort: cancel.clone(),
    };
    queue.set_active(active_handle).await;
    // The queue owns the lifecycle transition, so publish it immediately.
    // Waiting for the pull driver's first FileStart left clients showing
    // "queued" throughout repository resolution, cache verification, and
    // request setup even though this job already occupied an active slot.
    // A second Started frame supplies the real totals once file metadata is
    // available; reducers treat it as an idempotent active-job update.
    queue.emit(DownloadEvent::Started {
        id: job.id.clone(),
        files_total: 0,
        bytes_total: 0,
    });

    let _ = try_pull_with_retry(
        queue,
        driver,
        recipe_driver,
        models_dir,
        recipe_payload.as_ref(),
        hf_fallback_token,
        &mut job,
        cancel.clone(),
    )
    .await;
    queue.forget_hf_fallback_token(&job.id);

    // Move the finished job into history.
    let final_job = queue
        .with_active(&job.id, |a| a.clone())
        .await
        .unwrap_or_else(|| job.clone());
    queue.clear_active(&job.id).await;
    queue.push_history(final_job);
}

#[allow(clippy::too_many_arguments)]
async fn try_pull_with_retry(
    queue: &Arc<DownloadQueue>,
    driver: &Arc<dyn PullDriver>,
    recipe_driver: &Arc<dyn RecipePullDriver>,
    models_dir: &std::path::Path,
    recipe_payload: Option<&RecipePayload>,
    hf_fallback_token: HfFallbackTokenProvider,
    job: &mut DownloadJob,
    cancel: CancellationToken,
) -> Result<(), ()> {
    // Initial attempt + optional single retry.
    for attempt in 0..=1u8 {
        let result = run_single_attempt(
            queue,
            driver,
            recipe_driver,
            models_dir,
            recipe_payload,
            hf_fallback_token.clone(),
            job,
            cancel.clone(),
        )
        .await;
        match result {
            Ok(()) => {
                job.status = JobStatus::Completed;
                job.completed_at = Some(now_ms());
                // Reflect into the active-job snapshot so history includes the final state.
                queue
                    .with_active(&job.id, |a| {
                        *a = job.clone();
                    })
                    .await;
                queue.emit(DownloadEvent::JobDone {
                    id: job.id.clone(),
                    model: job.model.clone(),
                });
                queue.settle_for_group(&job.id, JobStatus::Completed);
                return Ok(());
            }
            Err(AttemptError::Cancelled) => {
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(now_ms());
                queue
                    .with_active(&job.id, |a| {
                        *a = job.clone();
                    })
                    .await;
                cleanup_partials_for_model(&job.model);
                queue.emit(DownloadEvent::JobCancelled { id: job.id.clone() });
                queue.settle_for_group(&job.id, JobStatus::Cancelled);
                return Err(());
            }
            Err(AttemptError::Failed(msg)) => {
                if attempt == 0 {
                    tracing::warn!(model = %job.model, "pull attempt 1 failed: {msg} — retrying in 5s");
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {},
                        _ = cancel.cancelled() => {
                            job.status = JobStatus::Cancelled;
                            job.completed_at = Some(now_ms());
                            queue
                                .with_active(&job.id, |a| {
                                    *a = job.clone();
                                })
                                .await;
                            cleanup_partials_for_model(&job.model);
                            queue.emit(DownloadEvent::JobCancelled { id: job.id.clone() });
                            queue.settle_for_group(&job.id, JobStatus::Cancelled);
                            return Err(());
                        }
                    }
                    continue;
                }
                job.status = JobStatus::Failed;
                job.error = Some(msg.clone());
                job.completed_at = Some(now_ms());
                queue
                    .with_active(&job.id, |a| {
                        *a = job.clone();
                    })
                    .await;
                cleanup_partials_for_model(&job.model);
                queue.emit(DownloadEvent::JobFailed {
                    id: job.id.clone(),
                    error: msg,
                });
                queue.settle_for_group(&job.id, JobStatus::Failed);
                return Err(());
            }
        }
    }
    Err(())
}

enum AttemptError {
    Cancelled,
    Failed(String),
}

#[allow(clippy::too_many_arguments)]
async fn run_single_attempt(
    queue: &Arc<DownloadQueue>,
    driver: &Arc<dyn PullDriver>,
    recipe_driver: &Arc<dyn RecipePullDriver>,
    models_dir: &std::path::Path,
    recipe_payload: Option<&RecipePayload>,
    hf_fallback_token: HfFallbackTokenProvider,
    job: &mut DownloadJob,
    cancel: CancellationToken,
) -> Result<(), AttemptError> {
    // Pipe progress events through an mpsc so we can process them serially,
    // maintain ordering, and guarantee they drain before JobDone fires.
    let (tx, mut rx) =
        tokio::sync::mpsc::unbounded_channel::<mold_core::download::DownloadProgressEvent>();
    let on_progress = Box::new(move |evt: mold_core::download::DownloadProgressEvent| {
        let _ = tx.send(evt);
    });

    let queue_for_drain = queue.clone();
    let job_id = job.id.clone();
    let drain_handle = tokio::spawn(async move {
        while let Some(evt) = rx.recv().await {
            translate_event(&queue_for_drain, &job_id, evt).await;
        }
    });

    let result = match recipe_payload {
        Some(payload) => {
            recipe_driver
                .pull(
                    payload.clone(),
                    models_dir.to_path_buf(),
                    on_progress,
                    cancel.clone(),
                )
                .await
        }
        None => {
            driver
                .pull_with_token_provider(
                    &job.model,
                    hf_fallback_token,
                    on_progress,
                    cancel.clone(),
                )
                .await
        }
    };
    // Wait for the drain task to finish (sender dropped when `pull` returned).
    let _ = drain_handle.await;

    match result {
        Ok(()) => Ok(()),
        Err(msg) if cancel.is_cancelled() => {
            let _ = msg;
            Err(AttemptError::Cancelled)
        }
        Err(msg) => Err(AttemptError::Failed(msg)),
    }
}

async fn translate_event(
    queue: &Arc<DownloadQueue>,
    job_id: &str,
    evt: mold_core::download::DownloadProgressEvent,
) {
    use mold_core::download::DownloadProgressEvent as P;
    let queue = queue.clone();
    let id = job_id.to_string();
    {
        match evt {
            P::FileStart {
                total_files,
                batch_bytes_total,
                filename,
                file_index,
                ..
            } => {
                let emitted_started = queue
                    .with_active(&id, |j| {
                        if j.files_total == 0 {
                            j.files_total = total_files;
                            j.bytes_total = batch_bytes_total;
                            true
                        } else {
                            false
                        }
                    })
                    .await
                    .unwrap_or(false);
                if emitted_started {
                    queue.emit(DownloadEvent::Started {
                        id: id.clone(),
                        files_total: total_files,
                        bytes_total: batch_bytes_total,
                    });
                }
                let _ = file_index;
                queue
                    .with_active(&id, |j| {
                        j.current_file = Some(filename.clone());
                    })
                    .await;
            }
            P::FileProgress {
                filename,
                bytes_downloaded,
                batch_bytes_downloaded,
                batch_bytes_total,
                file_index,
                bytes_total,
                ..
            } => {
                let _ = (bytes_downloaded, bytes_total, file_index);
                // Read the current `files_done` counter from the active job
                // so the emitted Progress event carries truth, not a zero
                // placeholder. The client reducer assigns `files_done`
                // directly from this field, so emitting 0 mid-batch would
                // reset the drawer's "X of Y files" counter and cause a
                // visible flicker until the next FileDone event.
                let files_done = queue
                    .with_active(&id, |j| {
                        j.bytes_done = batch_bytes_downloaded;
                        if j.bytes_total == 0 {
                            j.bytes_total = batch_bytes_total;
                        }
                        j.current_file = Some(filename.clone());
                        j.files_done
                    })
                    .await
                    .unwrap_or(0);
                queue.emit(DownloadEvent::Progress {
                    id: id.clone(),
                    files_done,
                    bytes_done: batch_bytes_downloaded,
                    current_file: Some(filename),
                });
            }
            P::FileDone {
                filename,
                file_index,
                total_files,
                batch_bytes_downloaded,
                batch_bytes_total,
                ..
            } => {
                let _ = batch_bytes_total;
                queue
                    .with_active(&id, |j| {
                        j.files_done = file_index + 1;
                        j.files_total = total_files;
                        j.bytes_done = batch_bytes_downloaded;
                    })
                    .await;
                queue.emit(DownloadEvent::FileDone {
                    id: id.clone(),
                    filename,
                });
            }
            P::Status { message } => {
                // Status-only preflight work is still live progress. Mirror
                // the current counters into a Progress delta so connected
                // clients see verification/resolution text without waiting
                // for a later snapshot or the first transferred byte.
                let counters = queue
                    .with_active(&id, |j| {
                        j.current_file = Some(message.clone());
                        (j.files_done, j.bytes_done)
                    })
                    .await;
                if let Some((files_done, bytes_done)) = counters {
                    queue.emit(DownloadEvent::Progress {
                        id: id.clone(),
                        files_done,
                        bytes_done,
                        current_file: Some(message),
                    });
                }
            }
        }
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "downloads_test.rs"]
mod tests;
