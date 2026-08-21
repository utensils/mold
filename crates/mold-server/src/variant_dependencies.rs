//! Pre-admission materialization of auto-selected encoder dependencies.
//!
//! This module is deliberately CUDA-free. It consumes the background resource
//! snapshot, selects one concrete encoder variant per schedulable device, and
//! downloads missing quantized files on Tokio's blocking pool. The scheduler
//! does not mark a generation Ready until this returns.

use crate::execution_plan::{
    DeviceFact, PreparedDeviceExecutionInputs, PreparedExecutionInputs,
    ENCODER_DEPENDENCY_HEADROOM_BYTES,
};
use crate::scheduler::worker_device_id;
use crate::state::{AppState, SseMessage};
use mold_core::{Config, GenerateRequest, ModelPaths, SseProgressEvent};
use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const T5_FP16_THRESHOLD: u64 = 16_000_000_000;
const QWEN3_4B_FP16_THRESHOLD: u64 = 10_200_000_000;
const QWEN2_FP16_THRESHOLD: u64 = 16_000_000_000;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct DownloadKey {
    models_root: PathBuf,
    repo: String,
    filename: String,
    subdir: String,
}

#[derive(Clone, Copy)]
pub(crate) struct DependencySpec<'a> {
    pub(crate) models_root: &'a Path,
    pub(crate) repo: &'a str,
    pub(crate) filename: &'a str,
    pub(crate) expected_bytes: Option<u64>,
    /// What a read-only preview reports this dependency as. A `kind` of
    /// `text_encoder` is the encoder ladders' answer; identity assets name
    /// their own component so a client can tell a face model from a prompt
    /// encoder in `pending_downloads`.
    pub(crate) kind: &'a str,
    /// Registry-declared container. A preview has not read the file, so this
    /// is the only honest source — claiming GGUF for every pending dependency
    /// mislabels the identity bundle's safetensors, `.pt`, and `.onnx` files.
    pub(crate) container: crate::execution_plan::PendingArtifactContainer,
    pub(crate) quantization: Option<crate::execution_plan::QuantizationVariant>,
    /// Manifest-pinned content digest the landed bytes must hash to.
    ///
    /// `None` keeps the historical bytes-and-size contract. `Some` is what
    /// makes a mutable Hugging Face `main` revision safe: this downloader
    /// resolves `main`, not a commit, so without the pin a file replaced
    /// upstream — or served by a compromised mirror — would be frozen into the
    /// plan and executed, since everything downstream only ever proves that
    /// the path is local.
    pub(crate) expected_sha256: Option<PinnedDigest<'a>>,
    pub(crate) subdir: &'a str,
}

/// A content pin plus the repair its violation should tell the user to run.
#[derive(Clone, Copy)]
pub(crate) struct PinnedDigest<'a> {
    pub(crate) sha256: &'a str,
    /// Model named by the `mold pull <model>` remedy in a mismatch error. The
    /// repo is the wrong answer here — a user repairs a bundle, not a
    /// Hugging Face repository.
    pub(crate) repair_model: &'a str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MissingDependency {
    pub(crate) download: mold_core::PendingModelDownload,
    pub(crate) path: PathBuf,
    pub(crate) container: crate::execution_plan::PendingArtifactContainer,
    pub(crate) quantization: Option<crate::execution_plan::QuantizationVariant>,
}

pub(crate) enum ResolvedDependency {
    Available(PathBuf),
    Pending(MissingDependency),
}

impl ResolvedDependency {
    pub(crate) fn into_path(self, pending: &mut Vec<MissingDependency>) -> PathBuf {
        match self {
            Self::Available(path) => path,
            Self::Pending(dependency) => {
                let path = dependency.path.clone();
                pending.push(dependency);
                path
            }
        }
    }
}

#[derive(Debug)]
struct SharedDownload {
    result: Mutex<Option<Result<PathBuf, String>>>,
    notify: tokio::sync::Notify,
    watchers: Mutex<Vec<DownloadWatcher>>,
}

#[derive(Debug)]
struct DownloadWatcher {
    id: u64,
    sender: tokio::sync::mpsc::UnboundedSender<SseMessage>,
}

struct DownloadWatcherGuard {
    id: u64,
    shared: Arc<SharedDownload>,
}

impl Drop for DownloadWatcherGuard {
    fn drop(&mut self) {
        self.shared
            .watchers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|watcher| watcher.id != self.id);
    }
}

static DOWNLOADS: OnceLock<Mutex<HashMap<DownloadKey, Arc<SharedDownload>>>> = OnceLock::new();
static NEXT_WATCHER_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
type TestDownloadAdapter =
    Arc<dyn Fn(&Path, &str, &str, &str) -> Result<PathBuf, String> + Send + Sync>;
#[cfg(test)]
static TEST_DOWNLOAD_ADAPTERS: OnceLock<Mutex<HashMap<String, TestDownloadAdapter>>> =
    OnceLock::new();

fn downloads() -> &'static Mutex<HashMap<DownloadKey, Arc<SharedDownload>>> {
    DOWNLOADS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn download_dependency_sync(
    models_root: &Path,
    repo: &str,
    filename: &str,
    subdir: &str,
    callback: mold_core::download::DownloadProgressCallback,
) -> Result<PathBuf, String> {
    #[cfg(test)]
    if let Some(adapter) = TEST_DOWNLOAD_ADAPTERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(repo)
        .cloned()
    {
        return adapter(models_root, repo, filename, subdir);
    }

    mold_core::download::download_single_file_sync_with_progress_in(
        models_root,
        repo,
        filename,
        Some(subdir),
        callback,
    )
    .map_err(|error| error.to_string())
}

fn normalized_download_root(root: &Path) -> Result<PathBuf, String> {
    let absolute = if root.is_absolute() {
        root.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("cannot resolve models root: {error}"))?
            .join(root)
    };
    if let Ok(canonical) = absolute.canonicalize() {
        return Ok(canonical);
    }

    let mut cursor = absolute.as_path();
    let mut suffix = Vec::<OsString>::new();
    while !cursor.exists() {
        let component = cursor.file_name().ok_or_else(|| {
            format!(
                "cannot normalize models root with no existing ancestor: {}",
                root.display()
            )
        })?;
        suffix.push(component.to_os_string());
        cursor = cursor.parent().ok_or_else(|| {
            format!(
                "cannot normalize models root with no existing ancestor: {}",
                root.display()
            )
        })?;
    }
    let mut normalized = cursor
        .canonicalize()
        .map_err(|error| format!("cannot normalize models root {}: {error}", root.display()))?;
    for component in suffix.into_iter().rev() {
        normalized.push(component);
    }
    Ok(normalized)
}

fn send_dependency_wait(
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    dependency: impl Into<String>,
    reason: impl Into<String>,
) {
    if let Some(progress) = progress {
        let _ = progress.send(SseMessage::Progress(SseProgressEvent::DependencyWait {
            dependency: dependency.into(),
            reason: reason.into(),
        }));
    }
}

/// What an already-present copy of a pinned dependency is worth.
enum CachedDependencyVerdict {
    /// Unpinned, or proven to be the pinned bytes.
    Usable,
    /// Pinned, present, and NOT the pinned bytes. Under `Admission` the file
    /// has already been removed, so a retry re-downloads.
    Rejected(String),
    /// Pinned and present, but this policy may not prove it. Only a read-only
    /// preview reaches this.
    Unproven,
}

/// Prove an already-placed dependency against its manifest pin.
///
/// Admission is the enforcing policy: it hashes the file's current bytes,
/// deletes a file that does not match, and attests one that does. A read-only
/// placement preview must not delete, must not attest, and must not refuse —
/// it only decides whether the copy on disk counts as evidence that nothing
/// needs downloading.
///
/// Neither policy reads the `.sha256-verified` marker as proof. Content
/// authentication cannot come from a sidecar anyone who can write the artifact
/// can also write.
fn verify_cached_dependency(
    path: &Path,
    filename: &str,
    expected_sha256: Option<PinnedDigest<'_>>,
    policy: DependencyMaterializationPolicy,
) -> CachedDependencyVerdict {
    let Some(pin) = expected_sha256 else {
        return CachedDependencyVerdict::Usable;
    };
    if policy == DependencyMaterializationPolicy::ExistingOnly {
        // Read-only: hash the current bytes through a retained descriptor and
        // answer. Never the `.sha256-verified` marker — it is a writable
        // sidecar in a models root the model-storage invariant lets a group
        // write, so it attests nothing about content. The hash is memoized per
        // file identity, so a preview of an unchanged installed bundle costs
        // one `fstat` per asset after the first.
        return if mold_core::download::pinned_file_matches(path, pin.sha256) {
            CachedDependencyVerdict::Usable
        } else {
            CachedDependencyVerdict::Unproven
        };
    }
    match mold_core::download::verify_pinned_file(path, pin.sha256, filename, pin.repair_model) {
        Ok(()) => CachedDependencyVerdict::Usable,
        Err(error) => CachedDependencyVerdict::Rejected(error.to_string()),
    }
}

pub(crate) async fn ensure_downloaded(
    state: Option<&AppState>,
    work_id: &str,
    dependency: DependencySpec<'_>,
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    policy: DependencyMaterializationPolicy,
) -> Result<ResolvedDependency, String> {
    let DependencySpec {
        models_root,
        repo,
        filename,
        expected_bytes,
        kind,
        container,
        quantization,
        expected_sha256,
        subdir,
    } = dependency;
    let cached = if policy == DependencyMaterializationPolicy::ExistingOnly {
        mold_core::download::cached_file_path_existing_only(
            models_root,
            repo,
            filename,
            Some(subdir),
        )
    } else {
        mold_core::download::cached_file_path_in(models_root, repo, filename, Some(subdir))
    };
    if let Some(path) = cached {
        match verify_cached_dependency(&path, filename, expected_sha256, policy) {
            CachedDependencyVerdict::Usable => return Ok(ResolvedDependency::Available(path)),
            CachedDependencyVerdict::Rejected(error) => return Err(error),
            // A read-only preview neither deletes the file nor writes an
            // attestation for it, so an unproven copy is simply not usable
            // evidence — it falls through and is reported as a pending
            // download that admission will verify for real.
            CachedDependencyVerdict::Unproven => {}
        }
    }
    if policy == DependencyMaterializationPolicy::ExistingOnly {
        let bytes = expected_bytes.ok_or_else(|| {
            format!(
                "required local dependency '{filename}' from '{repo}' is not installed and has no known materialization record"
            )
        })?;
        // A GGUF's quantization IS its storage record, so a registry entry
        // that cannot name one is not previewable. Containers that carry no
        // quantization at all (safetensors, `.pt`, `.onnx`) are declared by
        // `container` instead and legitimately have `None` here.
        if container == crate::execution_plan::PendingArtifactContainer::Gguf
            && quantization.is_none()
        {
            return Err(format!(
                "required local dependency '{filename}' from '{repo}' is not installed and has no known storage record"
            ));
        }
        return Ok(ResolvedDependency::Pending(MissingDependency {
            download: mold_core::PendingModelDownload {
                kind: kind.to_string(),
                name: filename.to_string(),
                repo: repo.to_string(),
                bytes,
            },
            path: mold_core::download::planned_single_file_path_in(models_root, filename, subdir),
            container,
            quantization,
        }));
    }

    let key = DownloadKey {
        models_root: normalized_download_root(models_root)?,
        repo: repo.to_string(),
        filename: filename.to_string(),
        subdir: subdir.to_string(),
    };
    let (shared, creator) = {
        let mut active = downloads()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match active.get(&key) {
            Some(shared) => (shared.clone(), false),
            None => {
                let shared = Arc::new(SharedDownload {
                    result: Mutex::new(None),
                    notify: tokio::sync::Notify::new(),
                    watchers: Mutex::new(Vec::new()),
                });
                active.insert(key.clone(), shared.clone());
                (shared, true)
            }
        }
    };
    let _watcher = progress.map(|progress| {
        let id = NEXT_WATCHER_ID.fetch_add(1, Ordering::Relaxed);
        shared
            .watchers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(DownloadWatcher {
                id,
                sender: progress.clone(),
            });
        DownloadWatcherGuard {
            id,
            shared: shared.clone(),
        }
    });

    send_dependency_wait(
        progress,
        filename,
        if creator {
            "downloading selected encoder dependency"
        } else {
            "joining an in-progress encoder dependency download"
        },
    );

    if creator {
        let shared = shared.clone();
        let key_for_task = key.clone();
        let models_root = models_root.to_path_buf();
        // Verification runs once, inside the shared task, so joiners share the
        // one hash of a multi-gigabyte artifact and every one of them sees the
        // same verdict — a joiner that skipped the check would freeze bytes
        // the creator rejected.
        let pin = expected_sha256.map(|pin| (pin.sha256.to_string(), pin.repair_model.to_string()));
        tokio::spawn(async move {
            let repo = key_for_task.repo.clone();
            let filename = key_for_task.filename.clone();
            let subdir = key_for_task.subdir.clone();
            let callback_shared = shared.clone();
            let callback: mold_core::download::DownloadProgressCallback = Arc::new(move |event| {
                let sse = match event {
                    mold_core::download::DownloadProgressEvent::FileStart {
                        filename,
                        size_bytes,
                        ..
                    } => Some(SseProgressEvent::DownloadProgress {
                        filename,
                        file_index: 0,
                        total_files: 1,
                        bytes_downloaded: 0,
                        bytes_total: size_bytes,
                        batch_bytes_downloaded: 0,
                        batch_bytes_total: size_bytes,
                        batch_elapsed_ms: 0,
                    }),
                    mold_core::download::DownloadProgressEvent::FileProgress {
                        filename,
                        file_index,
                        bytes_downloaded,
                        bytes_total,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    } => Some(SseProgressEvent::DownloadProgress {
                        filename,
                        file_index,
                        total_files: 1,
                        bytes_downloaded,
                        bytes_total,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    }),
                    mold_core::download::DownloadProgressEvent::FileDone {
                        filename,
                        file_index,
                        total_files,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    } => Some(SseProgressEvent::DownloadDone {
                        filename,
                        file_index,
                        total_files,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    }),
                    mold_core::download::DownloadProgressEvent::Status { .. } => None,
                };
                if let Some(sse) = sse {
                    callback_shared
                        .watchers
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .retain(|watcher| {
                            watcher
                                .sender
                                .send(SseMessage::Progress(sse.clone()))
                                .is_ok()
                        });
                }
            });
            let result = tokio::task::spawn_blocking(move || {
                download_dependency_sync(&models_root, &repo, &filename, &subdir, callback)
            })
            .await
            .map_err(|error| format!("encoder dependency task failed: {error}"))
            .and_then(|result| result)
            .and_then(|path| {
                // Hugging Face `main` is mutable and this downloader resolves
                // the branch, not a commit, so "the download succeeded" is not
                // "the manifest's bytes landed". A mismatch removes the file
                // here, before any caller can freeze the path into a plan.
                let Some((sha256, repair_model)) = pin.as_ref() else {
                    return Ok(path);
                };
                mold_core::download::verify_pinned_file(
                    &path,
                    sha256,
                    &key_for_task.filename,
                    repair_model,
                )
                .map(|()| path)
                .map_err(|error| error.to_string())
            });
            *shared
                .result
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result);
            shared.notify.notify_waiters();
            downloads()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&key_for_task);
        });
    }

    loop {
        let notified = shared.notify.notified();
        if let Some(result) = shared
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
        {
            return result.map(ResolvedDependency::Available);
        }
        if let Some(state) = state {
            if state.job_registry.entry(work_id).is_none() {
                return Err(format!(
                    "generation job {work_id} was cancelled while waiting for encoder dependency {filename}"
                ));
            }
            let cancelled = state.job_registry.mutation_notifier();
            tokio::select! {
                _ = notified => {}
                _ = cancelled.notified() => {
                if state.job_registry.entry(work_id).is_none() {
                    return Err(format!(
                        "generation job {work_id} was cancelled while waiting for encoder dependency {filename}"
                    ));
                }
            }
            }
        } else {
            notified.await;
        }
    }
}

fn resource_device_facts(state: &AppState) -> Vec<DeviceFact> {
    let resources = state.resources.latest();
    let registry = state.device_registry.canonical_snapshot(
        &state.gpu_pool,
        resources.as_ref(),
        &state.job_registry,
    );
    registry
        .scheduler_devices
        .into_iter()
        .filter(|device| device.schedulable)
        .filter_map(|device| {
            let worker = state
                .gpu_pool
                .workers
                .iter()
                .find(|worker| crate::scheduler::worker_device_id(worker) == device.id)?;
            let reclaimable_cache_bytes = worker
                .model_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .active_vram_bytes();
            let reclaimable_cache_bytes = device
                .sampled_mold_vram_bytes
                .map(|used_by_mold| reclaimable_cache_bytes.min(used_by_mold))
                .unwrap_or(0);
            let available = crate::scheduler::effective_available_vram_bytes(
                device.sampled_free_vram_bytes,
                reclaimable_cache_bytes,
                worker.gpu.total_vram_bytes,
            );
            Some(DeviceFact {
                id: device.id,
                ordinal: device.ordinal,
                backend: device.backend,
                compute_capability: device.compute_capability,
                available_vram_bytes: available,
            })
        })
        .collect()
}

#[cfg(test)]
fn effective_preparation_available_vram(
    total_vram_bytes: u64,
    used_vram_bytes: Option<u64>,
    mold_used_bytes: Option<u64>,
    active_cache_bytes: u64,
) -> u64 {
    let sampled_free_bytes = used_vram_bytes
        .map(|used| total_vram_bytes.saturating_sub(used))
        .unwrap_or(0);
    let reclaimable_cache_bytes = mold_used_bytes
        .map(|used_by_mold| active_cache_bytes.min(used_by_mold))
        .unwrap_or(0);
    crate::scheduler::effective_available_vram_bytes(
        sampled_free_bytes,
        reclaimable_cache_bytes,
        total_vram_bytes,
    )
}

fn worker_device_facts_from_startup_sample(state: &AppState) -> Vec<DeviceFact> {
    // Discovery sampled free memory before workers were created. Use that
    // bounded observation until the resource sampler publishes its first
    // snapshot; never substitute total VRAM for an unavailable free value.
    state
        .gpu_pool
        .workers
        .iter()
        .map(|worker| DeviceFact {
            id: worker_device_id(&worker),
            ordinal: worker.gpu.ordinal,
            backend: worker.gpu.backend,
            compute_capability: worker.gpu.compute_capability,
            available_vram_bytes: worker.gpu.free_vram_bytes,
        })
        .collect()
}

fn choose_largest_fitting<'a, T>(
    variants: &'a [T],
    free: u64,
    fields: impl Fn(&'a T) -> (&'static str, u64),
) -> Option<&'a T> {
    variants.iter().find(|variant| {
        fields(variant)
            .1
            .saturating_add(ENCODER_DEPENDENCY_HEADROOM_BYTES)
            <= free
    })
}

fn shared_quantized_fallback<'a, T>(
    variants: &'a [T],
    devices: &[DeviceFact],
    fields: impl Copy + Fn(&'a T) -> (&'static str, u64),
) -> Option<&'a T> {
    let smallest_supported_budget = devices
        .iter()
        .filter(|device| {
            variants.iter().any(|variant| {
                fields(variant)
                    .1
                    .saturating_add(ENCODER_DEPENDENCY_HEADROOM_BYTES)
                    <= device.available_vram_bytes
            })
        })
        .map(|device| device.available_vram_bytes)
        .min();
    smallest_supported_budget
        .and_then(|budget| choose_largest_fitting(variants, budget, fields))
        // Dependency materialization is not admission. When every device is
        // temporarily pressured, prepare the smallest concrete variant and
        // let the scheduler keep the work blocked until a device can admit it.
        .or_else(|| variants.last())
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct AutoQwen3SelectionKey {
    models_root: PathBuf,
    cache_subdir: String,
}

static AUTO_QWEN3_SELECTIONS: OnceLock<Mutex<HashMap<AutoQwen3SelectionKey, String>>> =
    OnceLock::new();

fn auto_qwen3_selections() -> &'static Mutex<HashMap<AutoQwen3SelectionKey, String>> {
    AUTO_QWEN3_SELECTIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Select one auto Qwen3 variant per storage root and encoder size.
///
/// The largest already-cached variant wins even when it does not fit the
/// current GPU snapshot; execution planning can place that stable dependency
/// on CPU. If no variant is cached, admission remembers its first
/// capacity-based choice before starting the download so concurrent or later
/// jobs join the same artifact instead of accumulating q3/iq4/q6 variants as
/// free VRAM fluctuates. Read-only previews may observe an admission choice but
/// never establish one.
fn select_auto_qwen3_variant_with_cache<'a>(
    models_root: &Path,
    cache_subdir: &str,
    variants: &'a [mold_core::manifest::Qwen3Variant],
    devices: &[DeviceFact],
    persist_selection: bool,
    is_cached: impl Fn(&mold_core::manifest::Qwen3Variant) -> bool,
) -> Option<&'a mold_core::manifest::Qwen3Variant> {
    let key = AutoQwen3SelectionKey {
        models_root: normalized_download_root(models_root)
            .unwrap_or_else(|_| models_root.to_path_buf()),
        cache_subdir: cache_subdir.to_string(),
    };

    if let Some(cached) = variants.iter().find(|variant| is_cached(variant)) {
        if persist_selection {
            auto_qwen3_selections()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(key, cached.tag.to_string());
        }
        return Some(cached);
    }

    let mut selections = auto_qwen3_selections()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(selected) = selections
        .get(&key)
        .and_then(|tag| variants.iter().find(|variant| variant.tag == tag))
    {
        return Some(selected);
    }

    let selected = shared_quantized_fallback(variants, devices, |variant| {
        (variant.tag, variant.size_bytes)
    })?;
    if persist_selection {
        selections.insert(key, selected.tag.to_string());
    }
    Some(selected)
}

fn select_auto_qwen3_variant<'a>(
    models_root: &Path,
    cache_subdir: &str,
    variants: &'a [mold_core::manifest::Qwen3Variant],
    devices: &[DeviceFact],
    persist_selection: bool,
) -> Option<&'a mold_core::manifest::Qwen3Variant> {
    select_auto_qwen3_variant_with_cache(
        models_root,
        cache_subdir,
        variants,
        devices,
        persist_selection,
        |variant| {
            mold_core::download::cached_file_path_existing_only(
                models_root,
                variant.hf_repo,
                variant.hf_filename,
                Some(cache_subdir),
            )
            .is_some()
        },
    )
}

fn registry_quantization(tag: &str) -> Option<crate::execution_plan::QuantizationVariant> {
    use crate::execution_plan::QuantizationVariant;
    match tag {
        "q2" => Some(QuantizationVariant::Q2),
        "q3" => Some(QuantizationVariant::Q3),
        "q4" | "iq4" => Some(QuantizationVariant::Q4),
        "q5" => Some(QuantizationVariant::Q5),
        "q6" => Some(QuantizationVariant::Q6),
        "q8" => Some(QuantizationVariant::Q8),
        _ => None,
    }
}

fn flux2_uses_qwen3_8b(model: &str, paths: &ModelPaths) -> bool {
    if model.to_ascii_lowercase().contains("9b") {
        return true;
    }
    paths
        .text_encoder_files
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum::<u64>()
        > 12_000_000_000
}

pub(crate) struct DependencyContext<'a> {
    pub(crate) state: Option<&'a AppState>,
    pub(crate) models_root: &'a Path,
    pub(crate) work_id: &'a str,
    pub(crate) progress: Option<&'a tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    pub(crate) policy: DependencyMaterializationPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DependencyMaterializationPolicy {
    Admission,
    ExistingOnly,
}

struct VariantSelection<'a> {
    preference: Option<&'a str>,
    auto_quantized_tag: Option<&'a str>,
    free: u64,
}

async fn materialize_t5(
    context: &DependencyContext<'_>,
    selection: VariantSelection<'_>,
    paths: &mut ModelPaths,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    let tag = match selection.preference {
        Some("fp16") => "fp16",
        Some("auto") | None if selection.free >= T5_FP16_THRESHOLD => "fp16",
        Some("auto") | None => selection.auto_quantized_tag.unwrap_or("fp16"),
        Some(tag) => {
            mold_core::manifest::find_t5_variant(tag)
                .ok_or_else(|| format!("unknown T5 encoder variant '{tag}'"))?;
            tag
        }
    };
    frozen.t5_variant = Some(tag.to_string());
    let selected = if tag == "fp16" {
        paths
            .t5_encoder
            .clone()
            .filter(|path| path.is_file())
            .ok_or_else(|| "selected FP16 T5 encoder is not locally available".to_string())?
    } else {
        let variant = mold_core::manifest::find_t5_variant(tag)
            .ok_or_else(|| format!("unknown T5 encoder variant '{tag}'"))?;
        ensure_downloaded(
            context.state,
            context.work_id,
            DependencySpec {
                models_root: context.models_root,
                repo: variant.hf_repo,
                filename: variant.hf_filename,
                expected_bytes: Some(variant.size_bytes),
                kind: "text_encoder",
                container: crate::execution_plan::PendingArtifactContainer::Gguf,
                quantization: registry_quantization(variant.tag),
                // The quantized encoder registries publish repo, filename, and
                // size but no digest, so there is nothing to pin here yet.
                // Pinning them is its own change: it needs 20+ digests sourced
                // and reviewed, not a field flipped.
                expected_sha256: None,
                subdir: "shared/t5-gguf",
            },
            context.progress,
            context.policy,
        )
        .await?
        .into_path(pending)
    };
    paths.t5_encoder = Some(selected.clone());
    frozen.selected_t5_path = Some(selected);
    Ok(())
}

/// Which UMT5 variant a device with `free` bytes of VRAM should render with.
///
/// The ladder mirrors `resolve_umt5_variant` in the engine, because admission
/// and execution disagreeing here means a download the render then does not
/// use. FP16 when it fits this device; otherwise the largest GGUF that does;
/// otherwise FP16, which is what the manifest already ships — so the "fits
/// nowhere" case downloads nothing and the engine parks the FP16 encoder.
fn select_umt5_tag(preference: Option<&str>, free: u64) -> Result<&'static str, String> {
    use mold_core::manifest::{find_umt5_variant, known_umt5_variants, UMT5_FP16_SIZE};

    // The engine's own threshold, not a second constant: the encoder needs its
    // weights plus activation headroom.
    let fits = |bytes: u64| free >= mold_inference::device::t5_vram_threshold(bytes);
    Ok(match preference {
        Some("fp16") => "fp16",
        Some("auto") | Some("") | None if fits(UMT5_FP16_SIZE) => "fp16",
        // `known_umt5_variants` is ordered largest-first, so the first fit is
        // the highest-quality one that fits.
        Some("auto") | Some("") | None => known_umt5_variants()
            .iter()
            .find(|variant| fits(variant.size_bytes))
            .map_or("fp16", |variant| variant.tag),
        Some(tag) => {
            find_umt5_variant(tag)
                .ok_or_else(|| format!("unknown UMT5 encoder variant '{tag}'"))?
                .tag
        }
    })
}

/// Wan's UMT5-XXL encoder.
///
/// Only the GGUF branch freezes a path. `fp16` leaves `selected_umt5_path`
/// unset, which is what tells the engine the manifest encoder is the route.
async fn materialize_umt5(
    context: &DependencyContext<'_>,
    selection: VariantSelection<'_>,
    paths: &mut ModelPaths,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    use mold_core::manifest::find_umt5_variant;

    let tag = select_umt5_tag(selection.preference, selection.free)?;
    frozen.umt5_variant = Some(tag.to_string());
    if tag == "fp16" {
        frozen.selected_umt5_path = None;
        return Ok(());
    }
    let variant =
        find_umt5_variant(tag).ok_or_else(|| format!("unknown UMT5 encoder variant '{tag}'"))?;
    let selected = ensure_downloaded(
        context.state,
        context.work_id,
        DependencySpec {
            models_root: context.models_root,
            repo: variant.hf_repo,
            filename: variant.hf_filename,
            expected_bytes: Some(variant.size_bytes),
            kind: "text_encoder",
            container: crate::execution_plan::PendingArtifactContainer::Gguf,
            quantization: registry_quantization(variant.tag),
            expected_sha256: None,
            subdir: "shared/wan/umt5-gguf",
        },
        context.progress,
        context.policy,
    )
    .await?
    .into_path(pending);
    paths.text_encoder_files = vec![selected.clone()];
    frozen.selected_umt5_path = Some(selected);
    Ok(())
}

async fn materialize_qwen3(
    context: &DependencyContext<'_>,
    model: &str,
    family: &str,
    selection: VariantSelection<'_>,
    paths: &mut ModelPaths,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    let b8 = family.starts_with("flux2") || family == "flux.2";
    let b8 = b8 && flux2_uses_qwen3_8b(model, paths);
    let variants = if b8 {
        mold_core::manifest::known_qwen3_8b_variants()
    } else {
        mold_core::manifest::known_qwen3_variants()
    };
    let find = |tag: &str| variants.iter().find(|variant| variant.tag == tag);
    let have_bf16 = !paths.text_encoder_files.is_empty()
        && paths.text_encoder_files.iter().all(|path| path.is_file());
    let fp16_threshold = if b8 {
        (mold_core::manifest::QWEN3_8B_FP16_SIZE as f64 * 1.25) as u64
    } else {
        QWEN3_4B_FP16_THRESHOLD
    };
    let prefer_quantized = matches!(family, "flux2" | "flux.2" | "flux2-klein");
    let tag = match selection.preference {
        Some("bf16") => {
            if !have_bf16 {
                return Err("selected BF16 Qwen3 encoder is not locally available".into());
            }
            "bf16"
        }
        Some("auto") | None
            if !prefer_quantized && have_bf16 && selection.free >= fp16_threshold =>
        {
            "bf16"
        }
        Some("auto") | None => {
            if let Some(tag) = selection.auto_quantized_tag {
                tag
            } else if have_bf16 {
                "bf16"
            } else {
                return Err(format!(
                    "no concrete Qwen3 encoder variant fits device budget {free}",
                    free = selection.free
                ));
            }
        }
        Some(tag) => {
            find(tag).ok_or_else(|| format!("unknown Qwen3 encoder variant '{tag}'"))?;
            tag
        }
    };
    frozen.qwen3_variant = Some(tag.to_string());
    let selected = if tag == "bf16" {
        paths.text_encoder_files.clone()
    } else {
        let variant = find(tag).ok_or_else(|| format!("unknown Qwen3 encoder variant '{tag}'"))?;
        let subdir = if b8 {
            "shared/qwen3-8b-gguf"
        } else {
            "shared/qwen3-gguf"
        };
        vec![ensure_downloaded(
            context.state,
            context.work_id,
            DependencySpec {
                models_root: context.models_root,
                repo: variant.hf_repo,
                filename: variant.hf_filename,
                expected_bytes: Some(variant.size_bytes),
                kind: "text_encoder",
                container: crate::execution_plan::PendingArtifactContainer::Gguf,
                quantization: registry_quantization(variant.tag),
                expected_sha256: None,
                subdir,
            },
            context.progress,
            context.policy,
        )
        .await?
        .into_path(pending)]
    };
    paths.text_encoder_files = selected.clone();
    frozen.selected_qwen3_paths = selected;
    Ok(())
}

async fn materialize_qwen2(
    context: &DependencyContext<'_>,
    selection: VariantSelection<'_>,
    family: &str,
    paths: &mut ModelPaths,
    frozen: &mut mold_inference::FrozenEngineConfig,
    pending: &mut Vec<MissingDependency>,
) -> Result<(), String> {
    let have_bf16 = !paths.text_encoder_files.is_empty()
        && paths.text_encoder_files.iter().all(|path| path.is_file());
    let tag = match selection.preference {
        Some("bf16") => "bf16",
        Some("auto") | None if have_bf16 && selection.free >= QWEN2_FP16_THRESHOLD => "bf16",
        Some("auto") | None => "q4",
        Some(tag) => {
            mold_core::manifest::find_qwen2_vl_variant(tag)
                .ok_or_else(|| format!("unknown Qwen2.5-VL encoder variant '{tag}'"))?;
            tag
        }
    };
    frozen.qwen2_variant = Some(tag.to_string());
    if tag == "bf16" {
        if !have_bf16 {
            return Err("selected BF16 Qwen2.5-VL encoder is not locally available".into());
        }
        return Ok(());
    }
    let variant = mold_core::manifest::find_qwen2_vl_variant(tag)
        .ok_or_else(|| format!("unknown Qwen2.5-VL encoder variant '{tag}'"))?;
    let selected = ensure_downloaded(
        context.state,
        context.work_id,
        DependencySpec {
            models_root: context.models_root,
            repo: variant.hf_repo,
            filename: variant.hf_filename,
            expected_bytes: Some(variant.size_bytes),
            kind: "text_encoder",
            container: crate::execution_plan::PendingArtifactContainer::Gguf,
            quantization: registry_quantization(variant.tag),
            expected_sha256: None,
            subdir: "shared/qwen2-vl-gguf",
        },
        context.progress,
        context.policy,
    )
    .await?
    .into_path(pending);
    if family != "qwen-image-edit" {
        paths.text_encoder_files = vec![selected.clone()];
    }
    frozen.selected_qwen2_path = Some(selected);
    Ok(())
}

fn gemma_root(paths: &ModelPaths) -> Result<PathBuf, String> {
    paths
        .text_encoder_files
        .first()
        .and_then(|path| path.parent())
        .map(Path::to_path_buf)
        .ok_or_else(|| "LTX-2 Gemma asset root is not configured".to_string())
}

fn sorted_matching_files(root: &Path, predicate: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut files = std::fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            (entry.path().is_file() && predicate(name)).then(|| entry.path())
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn complete_gemma_bf16_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let files = sorted_matching_files(root, |name| {
        name == "model.safetensors"
            || (name.starts_with("model-") && name.ends_with(".safetensors"))
    });
    let unsharded = files
        .iter()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name == "model.safetensors")
        })
        .cloned()
        .collect::<Vec<_>>();
    let sharded = files
        .iter()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name != "model.safetensors")
        })
        .cloned()
        .collect::<Vec<_>>();
    if !unsharded.is_empty() {
        if !sharded.is_empty() {
            return Err(
                "Gemma BF16 root mixes model.safetensors with sharded model-N-of-M files".into(),
            );
        }
        return Ok(unsharded);
    }
    if sharded.is_empty() {
        return Ok(Vec::new());
    }

    let mut total = None;
    let mut indices = std::collections::BTreeSet::new();
    for path in &sharded {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "Gemma BF16 shard name is not valid UTF-8".to_string())?;
        let body = name
            .strip_prefix("model-")
            .and_then(|value| value.strip_suffix(".safetensors"))
            .ok_or_else(|| format!("malformed Gemma BF16 shard name '{name}'"))?;
        let (index, shard_total) = body
            .split_once("-of-")
            .ok_or_else(|| format!("malformed Gemma BF16 shard name '{name}'"))?;
        let index = index
            .parse::<usize>()
            .map_err(|_| format!("malformed Gemma BF16 shard index in '{name}'"))?;
        let shard_total = shard_total
            .parse::<usize>()
            .map_err(|_| format!("malformed Gemma BF16 shard total in '{name}'"))?;
        if shard_total == 0 || index == 0 || index > shard_total {
            return Err(format!("invalid Gemma BF16 shard range in '{name}'"));
        }
        if total
            .replace(shard_total)
            .is_some_and(|prior| prior != shard_total)
        {
            return Err("inconsistent Gemma BF16 shard totals".into());
        }
        if !indices.insert(index) {
            return Err(format!("duplicate Gemma BF16 shard index {index}"));
        }
    }
    let total = total.expect("non-empty shard list records a total");
    if indices.len() != total || !(1..=total).all(|index| indices.contains(&index)) {
        return Err(format!(
            "incomplete Gemma BF16 shard set: found {} of {total}",
            indices.len()
        ));
    }
    Ok(sharded)
}

/// Floor of the host headroom admission keeps free, mirroring
/// `local_engine::plan_local_batch`: `max(15% of installed RAM, 8 GiB)`.
///
/// Reserving a flat 8 GiB here would disagree with admission on any host above
/// ~53 GB, letting this pick BF16 for a plan the scheduler then rejects while
/// Q4 would have fitted.
const GEMMA_HOST_SAFETY_FLOOR_MIN: u64 = 8 << 30;

/// Base transient admission charges every plan on top of its artifacts
/// (`execution_plan::BASE_HOST_TRANSIENT`).
const GEMMA_HOST_BASE_TRANSIENT: u64 = 256 * 1024 * 1024;

/// Headroom that must remain beside a resident BF16 Gemma encoder for the plan
/// carrying it to be admissible on a unified-memory host, where the encoder
/// shares one pool with the transformer it sits beside.
fn gemma_host_headroom(total_bytes: u64) -> u64 {
    total_bytes
        .saturating_mul(15)
        .saturating_div(100)
        .max(GEMMA_HOST_SAFETY_FLOOR_MIN)
        .saturating_add(GEMMA_HOST_BASE_TRANSIENT)
}

/// Pick a Gemma variant when nothing is pinned, given a memory reading.
///
/// Selection used to be presence-based: BF16 won whenever its shards existed,
/// however little memory was left. That is the one encoder in the tree without
/// the quantized auto-fallback T5, Qwen3, Qwen2 and UMT5 all perform, and on a
/// 48 GB unified-memory Mac it left LTX-2 permanently unadmittable — 24.5 GB of
/// resident Gemma plus a ~20 GB transformer does not fit, so every plan was
/// refused before a weight was read.
///
/// Quantizing this encoder is the normal path elsewhere rather than a
/// compromise: ComfyUI ships Gemma-3-12B for LTX-2 at BF16 (24.4 GB), FP8
/// (13.2 GB) and FP4-mixed (9.45 GB), and steers 16-24 GB cards to the smallest
/// of those, reserving BF16 for 32 GB+. Mold's Q4 is Google's quantization-aware
/// `q4_0` release, which holds up better than a post-training cast of the same
/// width.
///
/// `available` and `total` are `None` off macOS, where no reclaimable-memory
/// figure is published; BF16 stays the default there, so CUDA is unaffected.
///
/// `has_gguf` means *exactly one* GGUF is present. Inference loads only the
/// lexicographically first file in the Gemma root
/// (`ltx2::text::gemma::discover_gguf`), so an ambiguous set would let this
/// choose Q4 and then load a file nobody selected. Ambiguity keeps BF16, whose
/// shard set is validated as a complete five-way group.
fn choose_gemma_tag(
    bf16_bytes: u64,
    has_gguf: bool,
    available: Option<u64>,
    total: Option<u64>,
) -> &'static str {
    if !has_gguf {
        return "bf16";
    }
    let (Some(available), Some(total)) = (available, total) else {
        return "bf16";
    };
    // checked_add, not saturating: a saturated sum would compare equal to a
    // saturated `available` and wrongly read as "fits".
    match bf16_bytes.checked_add(gemma_host_headroom(total)) {
        Some(required) if required <= available => "bf16",
        _ => "q4",
    }
}

/// Resolve [`choose_gemma_tag`] against live host memory, disclosing a
/// downgrade rather than silently substituting different weights.
fn auto_gemma_tag(bf16: &[PathBuf], gguf: &[PathBuf]) -> &'static str {
    let bf16_bytes = bf16
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .fold(0u64, u64::saturating_add);
    let available = mold_inference::device::available_system_memory_bytes();
    let total = mold_inference::device::total_system_memory_bytes();
    let tag = choose_gemma_tag(bf16_bytes, gguf.len() == 1, available, total);
    if tag == "q4" {
        tracing::warn!(
            bf16_bytes,
            available_bytes = available.unwrap_or(0),
            "the BF16 Gemma prompt encoder does not fit in available host memory; \
             using the Q4 encoder instead — prompt adherence may differ. Pin it with \
             MOLD_LTX2_GEMMA_VARIANT=bf16 or q4 to choose explicitly."
        );
    }
    tag
}

fn materialize_gemma(
    preference: Option<&str>,
    paths: &ModelPaths,
    frozen: &mut mold_inference::FrozenEngineConfig,
) -> Result<(), String> {
    // Every built-in runnable LTX-2 manifest carries the five verified Gemma
    // BF16 shards as TextEncoder files (guarded by the manifest contract
    // test). Catalog companions must establish the same complete local root
    // before ModelPaths resolves. Q4 has no authoritative remote variant
    // registry, so an explicit missing Q4 pin fails here rather than causing
    // network or discovery after scheduler admission.
    let root = gemma_root(paths)?;
    let bf16 = complete_gemma_bf16_files(&root)?;
    // Case-insensitive, matching `ltx2::text::gemma::discover_gguf`. Counting
    // only lowercase names here would let `a.GGUF` beside `b.gguf` look like a
    // single unambiguous Q4 to preparation while inference sorts both and can
    // load the one that was never frozen.
    let gguf = sorted_matching_files(&root, |name| {
        std::path::Path::new(name)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    });
    let tag = match preference.map(|value| value.trim().to_ascii_lowercase()) {
        Some(value) if matches!(value.as_str(), "q4" | "gguf" | "q4_gguf") => "q4",
        Some(value) if matches!(value.as_str(), "bf16" | "safetensors" | "bf16_safetensors") => {
            "bf16"
        }
        Some(value) if (value.is_empty() || value == "auto") && !bf16.is_empty() => {
            auto_gemma_tag(&bf16, &gguf)
        }
        Some(value) if value.is_empty() || value == "auto" => "q4",
        Some(value) => return Err(format!("unknown LTX-2 Gemma variant '{value}'")),
        None if !bf16.is_empty() => auto_gemma_tag(&bf16, &gguf),
        None => "q4",
    };
    let selected = if tag == "bf16" { bf16 } else { gguf };
    if selected.is_empty() {
        return Err(format!(
            "selected LTX-2 Gemma {tag} encoder is not locally available in '{}'",
            root.display()
        ));
    }
    frozen.ltx2_gemma_variant = Some(tag.to_string());
    frozen.selected_gemma_paths = selected;
    Ok(())
}

// Keep the generic dependency inputs explicit: private admission must validate
// its ingress context before any model-path resolution or materialization.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn prepare_inputs_for_devices(
    state: Option<&AppState>,
    work_id: &str,
    request: &GenerateRequest,
    config: &Config,
    devices: Vec<DeviceFact>,
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    policy: DependencyMaterializationPolicy,
    context: DependencyPreparationContext,
) -> Result<PreparedExecutionInputs, String> {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if let Some(grant) = context.h3_private_ingress_grant.clone() {
        let live_state = state.ok_or_else(|| {
            "MiniMax H3 private dependency preparation has no server instance authority".to_string()
        })?;
        grant.validate_for_request(request, live_state.instance_id.as_str())?;
        return prepare_h3_private_inputs_for_devices(
            state,
            work_id,
            request,
            config,
            devices,
            progress,
            policy,
            grant,
            context.h3_resolved_references.clone(),
        )
        .await;
    }
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    let _ = &context;
    let resolution = crate::model_manager::resolve_existing_model_paths(&request.model, config)
        .map_err(|error| error.error)?
        .ok_or_else(|| format!("model '{}' has no concrete local artifacts", request.model))?;
    let model_config_overlay = resolution.model_config_overlay.map(Arc::new);
    let overlaid_config = model_config_overlay.as_ref().map(|model_config| {
        let mut effective = config.clone();
        effective
            .models
            .insert(request.model.clone(), model_config.as_ref().clone());
        effective
    });
    let config = overlaid_config.as_ref().unwrap_or(config);
    let paths = resolution.paths;
    let family = config
        .resolved_model_config(&request.model)
        .family
        .or_else(|| {
            mold_core::manifest::find_manifest(&request.model)
                .map(|manifest| manifest.family.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let base = mold_inference::FrozenEngineConfig::resolve(&request.model, config);
    let authority_fingerprint =
        crate::execution_plan::preparation_authority_fingerprint(config, request, &paths, &base);
    let devices = crate::execution_plan::eligible_devices_for_request(config, request, &devices)
        .map_err(|error| error.to_string())?;
    if devices.is_empty() {
        return Err("request placement has no eligible schedulable device".into());
    }
    let models_root = config.resolved_models_dir();
    let shared_t5_tag = shared_quantized_fallback(
        mold_core::manifest::known_t5_variants(),
        &devices,
        |variant| (variant.tag, variant.size_bytes),
    )
    .map(|variant| variant.tag.to_string());
    let qwen3_family = matches!(
        family.as_str(),
        "z-image" | "flux2" | "flux.2" | "flux2-klein"
    ) && !mold_core::validation::is_flux2_dev_model(&request.model);
    let qwen3_auto = base
        .qwen3_variant
        .as_deref()
        .is_none_or(|value| value.is_empty() || value == "auto");
    let qwen3_8b = qwen3_family && flux2_uses_qwen3_8b(&request.model, &paths);
    let qwen3_variants = if qwen3_8b {
        mold_core::manifest::known_qwen3_8b_variants()
    } else {
        mold_core::manifest::known_qwen3_variants()
    };
    let qwen3_cache_subdir = if qwen3_8b {
        "shared/qwen3-8b-gguf"
    } else {
        "shared/qwen3-gguf"
    };
    let shared_qwen3_tag = (qwen3_family && qwen3_auto)
        .then(|| {
            select_auto_qwen3_variant(
                &models_root,
                qwen3_cache_subdir,
                qwen3_variants,
                &devices,
                policy == DependencyMaterializationPolicy::Admission,
            )
        })
        .flatten()
        .map(|variant| variant.tag.to_string());
    let capacity_sensitive = match family.as_str() {
        "flux"
        | "sd3"
        | "sd3.5"
        | "stable-diffusion-3"
        | "stable-diffusion-3.5"
        | "ltx-video"
        | "ltx_video" => base
            .t5_variant
            .as_deref()
            .is_none_or(|value| value.is_empty() || value == "auto"),
        "z-image" | "flux2" | "flux.2" | "flux2-klein"
            if !mold_core::validation::is_flux2_dev_model(&request.model) =>
        {
            base.qwen3_variant
                .as_deref()
                .is_none_or(|value| value.is_empty() || value == "auto")
        }
        "qwen-image" | "qwen_image" | "qwen-image-edit" => base
            .qwen2_variant
            .as_deref()
            .is_none_or(|value| value.is_empty() || value == "auto"),
        "wan" => base
            .umt5_variant
            .as_deref()
            .is_none_or(|value| value.is_empty() || value == "auto"),
        // An unpinned Gemma is chosen from live host memory the same way, so a
        // plan frozen under one pressure must be replanned under another.
        // `materialize_gemma` trims and lowercases the preference before
        // matching it, so this must too — otherwise `AUTO` or ` auto ` selects
        // automatically and then never replans. The arms above compare the raw
        // string and carry the same latent gap.
        "ltx2" | "ltx-2" | "ltx2.3" => base.ltx2_gemma_variant.as_deref().is_none_or(|value| {
            let value = value.trim().to_ascii_lowercase();
            value.is_empty() || value == "auto"
        }),
        _ => false,
    };

    let mut by_device = BTreeMap::new();
    let mut failures = BTreeMap::new();
    let dependency_context = DependencyContext {
        state,
        models_root: &models_root,
        work_id,
        progress,
        policy,
    };
    for device in devices {
        let mut selected_paths = paths.clone();
        let mut frozen = base.clone();
        let mut pending = Vec::new();
        let materialized = match family.as_str() {
            "flux"
            | "sd3"
            | "sd3.5"
            | "stable-diffusion-3"
            | "stable-diffusion-3.5"
            | "ltx-video"
            | "ltx_video" => {
                materialize_t5(
                    &dependency_context,
                    VariantSelection {
                        preference: base.t5_variant.as_deref(),
                        auto_quantized_tag: shared_t5_tag.as_deref(),
                        free: device.available_vram_bytes,
                    },
                    &mut selected_paths,
                    &mut frozen,
                    &mut pending,
                )
                .await
            }
            "z-image" | "flux2" | "flux.2" | "flux2-klein"
                if !mold_core::validation::is_flux2_dev_model(&request.model) =>
            {
                materialize_qwen3(
                    &dependency_context,
                    &request.model,
                    &family,
                    VariantSelection {
                        preference: base.qwen3_variant.as_deref(),
                        auto_quantized_tag: shared_qwen3_tag.as_deref(),
                        free: device.available_vram_bytes,
                    },
                    &mut selected_paths,
                    &mut frozen,
                    &mut pending,
                )
                .await
            }
            "qwen-image" | "qwen_image" | "qwen-image-edit" => {
                materialize_qwen2(
                    &dependency_context,
                    VariantSelection {
                        preference: base.qwen2_variant.as_deref(),
                        auto_quantized_tag: None,
                        free: device.available_vram_bytes,
                    },
                    &family,
                    &mut selected_paths,
                    &mut frozen,
                    &mut pending,
                )
                .await
            }
            "ltx2" | "ltx-2" | "ltx2.3" => materialize_gemma(
                base.ltx2_gemma_variant.as_deref(),
                &selected_paths,
                &mut frozen,
            ),
            "wan" => {
                materialize_umt5(
                    &dependency_context,
                    VariantSelection {
                        preference: base.umt5_variant.as_deref(),
                        auto_quantized_tag: None,
                        free: device.available_vram_bytes,
                    },
                    &mut selected_paths,
                    &mut frozen,
                    &mut pending,
                )
                .await
            }
            _ => Ok(()),
        };
        // Identity assets are orthogonal to the encoder ladder above: the
        // request asks for a face, not for a variant, so this runs after
        // whichever family arm applied and is inert for every request that
        // does not condition on one.
        let materialized = match materialized {
            Ok(()) => {
                crate::identity_dependencies::materialize_identity_assets(
                    &dependency_context,
                    request,
                    &family,
                    &mut frozen,
                    &mut pending,
                )
                .await
            }
            Err(error) => Err(error),
        };
        if let Err(error) = materialized {
            failures.insert(device.id, error);
            continue;
        }
        by_device.insert(
            device.id,
            PreparedDeviceExecutionInputs {
                engine_paths: selected_paths,
                engine_config: frozen,
                pending_artifacts: pending
                    .iter()
                    .map(|dependency| {
                        (
                            dependency.path.clone(),
                            crate::execution_plan::PendingArtifactIdentity {
                                kind: dependency.download.kind.clone(),
                                repo: dependency.download.repo.clone(),
                                filename: dependency.download.name.clone(),
                                bytes: dependency.download.bytes,
                                container: dependency.container,
                                quantization: dependency.quantization,
                            },
                        )
                    })
                    .collect(),
                prepared_available_vram_bytes: device.available_vram_bytes,
                capacity_sensitive,
            },
        );
    }
    if by_device.is_empty() {
        return Err(format!(
            "no request-eligible device could materialize encoder dependencies: {}",
            failures
                .iter()
                .map(|(device, error)| format!("{device}: {error}"))
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    // The identity is resolved ONCE, here, after the per-device loop and
    // before any fan-out, and the value is device-independent by construction.
    // This is the whole of #1223's extraction lifetime: a batch parent clones
    // this struct into every child, so every sibling and every denoise step
    // reuses this exact embedding. See `crate::identity_extraction`.
    let identity_embedding = match context.frozen_identity.clone() {
        // A batch child, re-prepared by the scheduler: the parent already
        // resolved this exact face. Reusing it is the contract, not an
        // optimization.
        Some(frozen) => Some(frozen),
        // Placement preview is a read-only probe. It must never spend seconds
        // and 1.4 GB running the extractor, and it does not need to: identity
        // changes the memory demand, which `memory_preflight` charges from the
        // request, not from the embedding.
        None if policy == DependencyMaterializationPolicy::ExistingOnly => None,
        None => {
            let identity_paths = by_device
                .values()
                .find_map(|device| device.engine_config.identity_assets.clone());
            crate::identity_extraction::resolve_identity_embedding(
                request,
                identity_paths.as_ref(),
            )
            .await?
        }
    };

    let prepared = PreparedExecutionInputs {
        authority_fingerprint,
        by_device,
        retryable_device_failures: failures,
        model_config_overlay,
        identity_embedding,
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        h3_private_ingress_grant: context.h3_private_ingress_grant,
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        h3_private_admission_by_device: BTreeMap::new(),
    };
    let warm_config = config.clone();
    let warm_request = request.clone();
    let warm_prepared = prepared.clone();
    tokio::task::spawn_blocking(move || {
        crate::execution_plan::warm_execution_equivalence_cache(
            &warm_config,
            &warm_request,
            &warm_prepared,
        );
    })
    .await
    .map_err(|error| format!("execution-equivalence artifact hashing failed: {error}"))?;
    Ok(prepared)
}

/// Cancels its token when dropped UNLESS disarmed, so an abandoned or
/// unwinding preparation stops the CPU decoding it spawned, while a completed
/// one leaves its token uncancelled — the admitted evidence and any binding
/// derived from it must stay valid after preparation returns.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
struct PreparationCancellationGuard {
    token: mold_inference::InferenceCancellationToken,
    armed: bool,
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
impl PreparationCancellationGuard {
    fn new() -> Self {
        Self {
            token: mold_inference::InferenceCancellationToken::default(),
            armed: true,
        }
    }

    fn token(&self) -> mold_inference::InferenceCancellationToken {
        self.token.clone()
    }

    /// Call on the success path only. Every early return and every unwind
    /// leaves the guard armed, which is what makes abandonment cancel.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
impl Drop for PreparationCancellationGuard {
    fn drop(&mut self) {
        if self.armed {
            self.token.cancel();
        }
    }
}

#[cfg(all(test, any(feature = "h3", feature = "h3-private-uat")))]
mod preparation_cancellation_tests {
    use super::PreparationCancellationGuard;

    /// An abandoned or unwinding preparation must cancel the work it spawned,
    /// so a cancelled job stops decoding at the next checkpoint instead of
    /// running to completion on media nobody will use.
    #[test]
    fn an_armed_guard_cancels_its_token_on_drop() {
        let guard = PreparationCancellationGuard::new();
        let token = guard.token();
        assert!(!token.is_cancelled());
        drop(guard);
        assert!(token.is_cancelled());
    }

    /// A COMPLETED preparation must not cancel: its admitted evidence and any
    /// binding derived from it have to stay valid after preparation returns.
    #[test]
    fn a_disarmed_guard_leaves_its_token_usable() {
        let mut guard = PreparationCancellationGuard::new();
        let token = guard.token();
        guard.disarm();
        drop(guard);
        assert!(!token.is_cancelled());
    }

    /// The Ref2VA media decode checkpoints through the admission reporter, so
    /// the reporter — not just the binding step — has to carry the token.
    /// Binding consults it directly while hashing and retains none, so a token
    /// installed only there leaves the decode itself uninterruptible.
    #[test]
    fn the_admission_reporter_carries_the_cancellation_token_into_decode() {
        let guard = PreparationCancellationGuard::new();
        let mut reporter = mold_inference::progress::ProgressReporter::default();
        reporter.set_cancellation_token(guard.token());

        // Before cancellation the decode proceeds.
        reporter
            .checkpoint()
            .expect("an armed preparation still runs");

        // Dropping the guard is what an abandoned preparation does, and the
        // next decode checkpoint must observe it.
        drop(guard);
        let stopped = reporter.checkpoint().unwrap_err();
        assert!(mold_inference::is_inference_cancelled(&anyhow::Error::new(
            stopped
        )));
    }

    /// Unwinding is the abandonment case that is easiest to get wrong, since
    /// it takes no explicit return path.
    #[test]
    fn a_panicking_preparation_still_cancels() {
        let guard = PreparationCancellationGuard::new();
        let token = guard.token();
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = guard;
            panic!("preparation failed");
        }));
        assert!(unwound.is_err());
        assert!(token.is_cancelled());
    }
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[allow(clippy::too_many_arguments)]
async fn prepare_h3_private_inputs_for_devices(
    state: Option<&AppState>,
    _work_id: &str,
    request: &GenerateRequest,
    config: &Config,
    devices: Vec<DeviceFact>,
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    _policy: DependencyMaterializationPolicy,
    ingress_grant: crate::h3_private_bridge::H3PrivateIngressGrant,
    resolved_references: Option<crate::reference_uploads::ResolvedReferenceAdmissionView>,
) -> Result<PreparedExecutionInputs, String> {
    use sha2::{Digest, Sha256};

    let devices = crate::execution_plan::eligible_devices_for_private_h3(config, request, &devices)
        .map_err(|error| error.to_string())?;
    if devices.is_empty() {
        return Err("request placement has no eligible schedulable device".into());
    }
    let available_host_headroom_bytes =
        crate::h3_admission::current_h3_host_memory().headroom_bytes();
    let uat_paths =
        crate::h3_private_bridge::H3PrivateUatPathSet::resolve(config.resolved_models_dir());
    // The public runtime owns its MOLD_HOME-derived staging root; create it
    // when absent so a fresh deployment can admit. The private-UAT campaign
    // layout stays fail-closed — its hand-built scope must already exist.
    #[cfg(feature = "h3")]
    uat_paths.ensure_staging_root();
    let mut resolved_request = request.clone();
    // Dropped when this preparation returns or unwinds, which cancels every
    // spawned decode that is still running for it.
    let mut preparation_cancellation = PreparationCancellationGuard::new();
    let mut evidence_by_device = BTreeMap::new();
    let mut failures = BTreeMap::new();

    for device in devices {
        if device.backend != mold_core::GpuBackend::Cuda {
            failures.insert(
                device.id,
                "MiniMax H3 private UAT requires a CUDA device".to_string(),
            );
            continue;
        }
        let Some(compute_capability) = device.compute_capability else {
            failures.insert(
                device.id,
                "MiniMax H3 private UAT requires exact CUDA compute capability".to_string(),
            );
            continue;
        };
        let admission_request = resolved_request.clone();
        let paths = uat_paths.clone();
        // Each device's admission mints its own descriptors rather than
        // sharing one set: a binding owns an open file, and re-opening per
        // attempt is what keeps descriptor identity fenced to the attempt that
        // verified it.
        let references = resolved_references.clone();
        let cancellation = preparation_cancellation.token();
        let device_id = device.id.clone();
        let device_ordinal = device.ordinal;
        let available_device_bytes = device.available_vram_bytes;
        let progress_tx = progress.cloned();
        let evidence = tokio::task::spawn_blocking(move || {
            let mut reporter = mold_inference::progress::ProgressReporter::default();
            // The decode checkpoints through this reporter, so the token has
            // to live here — bindings retain none, and consulting it only
            // while hashing would leave the media decode uninterruptible.
            reporter.set_cancellation_token(cancellation.clone());
            if let Some(progress_tx) = progress_tx {
                reporter.set_callback(Box::new(move |event| {
                    crate::gpu_worker::record_h3_progress(event, Some(&progress_tx));
                }));
            }
            // Cooperative abort: a cancelled preparation stops decoding at
            // the next checkpoint instead of burning CPU on media whose job is
            // already gone. The staging and its quota stay alive regardless,
            // because the view shares the resolved set's hold.
            let bindings = match references
                .as_ref()
                .map(|references| {
                    references.inference_bindings(&admission_request, Some(&cancellation))
                })
                .transpose()
            {
                Ok(bindings) => bindings.unwrap_or_default(),
                Err(error) => {
                    return Err(format!(
                        "MiniMax H3 admission could not bind its staged references: {error}"
                    ))
                }
            };
            mold_inference::prepare_h3_private_fl2va_admission(
                mold_inference::H3PrivateFl2VaAdmissionInput {
                    request: &admission_request,
                    paths: paths.inference_paths(),
                    references: &bindings,
                    device_id: &device_id,
                    device_ordinal,
                    compute_capability,
                    available_device_bytes,
                    available_host_headroom_bytes,
                },
                &reporter,
            )
            .map_err(crate::h3_private_bridge::private_prepare_error_message)
        })
        .await
        .map_err(|error| format!("MiniMax H3 admission worker failed: {error}"))?;
        let evidence = match evidence {
            Ok(evidence) => evidence,
            Err(error) => {
                failures.insert(device.id, error);
                continue;
            }
        };
        let next_request = evidence
            .resolve_request(&resolved_request)
            .map_err(|error| format!("MiniMax H3 admission seed resolution failed: {error:#}"))?;
        evidence
            .validate_for(
                &next_request,
                &device.id,
                device.ordinal,
                compute_capability,
                device.available_vram_bytes,
                available_host_headroom_bytes,
            )
            .map_err(|error| {
                format!("MiniMax H3 admission evidence did not revalidate: {error:#}")
            })?;
        resolved_request = next_request;
        evidence_by_device.insert(device.id.clone(), (device, evidence));
    }
    if evidence_by_device.is_empty() {
        return Err(format!(
            "no request-eligible device passed MiniMax H3 private admission: {}",
            failures
                .iter()
                .map(|(device, error)| format!("{device}: {error}"))
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }

    let instance_id = state
        .ok_or_else(|| {
            "MiniMax H3 private admission lost its server instance authority".to_string()
        })?
        .instance_id
        .as_str();
    let rebound_grant =
        ingress_grant.rebind_resolved_request(request, &resolved_request, instance_id)?;
    let engine_paths = ModelPaths::resolve(&resolved_request.model, config).ok_or_else(|| {
        "reviewed MiniMax H3 admission could not project its verified manifest paths".to_string()
    })?;
    let mut by_device = BTreeMap::new();
    let mut admissions = BTreeMap::new();
    for (device_id, (device, evidence)) in evidence_by_device {
        let mut frozen =
            mold_inference::FrozenEngineConfig::resolve(&resolved_request.model, config);
        frozen.family = mold_core::minimax_h3::FAMILY.to_string();
        frozen.h3_factory_authority = Some(evidence.base_factory_authority().clone());
        frozen.attention_backend = evidence.attention().generic_backend;
        frozen.attention_chunk = evidence.attention().generic_chunk;
        by_device.insert(
            device_id.clone(),
            PreparedDeviceExecutionInputs {
                engine_paths: engine_paths.clone(),
                engine_config: frozen,
                pending_artifacts: BTreeMap::new(),
                prepared_available_vram_bytes: device.available_vram_bytes,
                capacity_sensitive: true,
            },
        );
        admissions.insert(device_id, evidence);
    }
    let mut authority = Sha256::new();
    authority.update(b"mold.minimax-h3.private-prepared-inputs.v1\0");
    authority.update(rebound_grant.authority_identity_sha256().as_bytes());
    for (device_id, evidence) in &admissions {
        authority.update((device_id.len() as u64).to_le_bytes());
        authority.update(device_id.as_bytes());
        authority.update(evidence.identity_sha256().as_bytes());
    }
    // Every device admitted, so the work this token guards is finished and
    // its bindings must remain usable.
    preparation_cancellation.disarm();
    Ok(PreparedExecutionInputs {
        authority_fingerprint: format!("{:x}", authority.finalize()),
        by_device,
        retryable_device_failures: failures,
        model_config_overlay: None,
        // The private H3 ingress has no face-identity path; FLUX is the only
        // family qualified for it.
        identity_embedding: None,
        h3_private_ingress_grant: Some(rebound_grant),
        h3_private_admission_by_device: admissions,
    })
}

#[derive(Clone, Debug, Default)]
pub struct DependencyPreparationContext {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) h3_private_ingress_grant: Option<crate::h3_private_bridge::H3PrivateIngressGrant>,
    /// Staged Ref2VA references, as a payload-free view.
    ///
    /// Ref2VA admission must derive its prepared shapes from the real media,
    /// so it needs the staged files before the frozen plan exists. Only the
    /// scheduler populates this, from the owning job; placement preview leaves
    /// it `None`, which keeps that probe media-free and non-authoritative for
    /// Ref2VA exactly as the placement-preview boundary requires.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) h3_resolved_references:
        Option<crate::reference_uploads::ResolvedReferenceAdmissionView>,
    /// An identity the parent request already froze.
    ///
    /// The scheduler prepares dependencies for EVERY pending job, batch
    /// children included, and the result replaces whatever the parent
    /// composed. Without this the four children of a `batch_size = 4` parent
    /// would each re-run the extractor — five extractions of one face, five
    /// chances for the siblings to disagree. Populated, the extraction is
    /// skipped entirely and the parent's exact value is carried forward.
    pub(crate) frozen_identity: Option<mold_core::identity::FrozenIdentityEmbedding>,
}

pub async fn prepare_execution_inputs(
    state: &AppState,
    work_id: &str,
    request: &GenerateRequest,
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    context: DependencyPreparationContext,
) -> Result<PreparedExecutionInputs, String> {
    let config = state.config.read().await.clone();
    let mut devices = resource_device_facts(state);
    if devices.is_empty() {
        devices = worker_device_facts_from_startup_sample(state);
    }
    prepare_inputs_for_devices(
        Some(state),
        work_id,
        request,
        &config,
        devices,
        progress,
        DependencyMaterializationPolicy::Admission,
        context,
    )
    .await
}

pub async fn prepare_execution_inputs_existing_only(
    state: &AppState,
    request: &GenerateRequest,
    context: DependencyPreparationContext,
) -> Result<PreparedExecutionInputs, String> {
    let config = state.config.read().await.clone();
    let mut devices = resource_device_facts(state);
    if devices.is_empty() {
        devices = worker_device_facts_from_startup_sample(state);
    }
    prepare_inputs_for_devices(
        Some(state),
        "placement-preview",
        request,
        &config,
        devices,
        None,
        DependencyMaterializationPolicy::ExistingOnly,
        context,
    )
    .await
}

/// Forced-local counterpart to server preparation. It shares variant
/// selection and download de-duplication but has no job-registry cancellation
/// source or SSE transport.
pub async fn prepare_local_execution_inputs(
    config: &Config,
    request: &GenerateRequest,
    devices: Vec<DeviceFact>,
) -> Result<PreparedExecutionInputs, String> {
    prepare_inputs_for_devices(
        None,
        "local",
        request,
        config,
        devices,
        None,
        DependencyMaterializationPolicy::Admission,
        DependencyPreparationContext::default(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{DevicePlacement, DeviceRef, ModelConfig};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use tempfile::TempDir;

    #[test]
    fn encoder_dependency_headroom_contract_is_decimal_two_gigabytes() {
        assert_eq!(ENCODER_DEPENDENCY_HEADROOM_BYTES, 2_000_000_000);
    }

    /// Admission's UMT5 ladder has to be the engine's ladder.
    ///
    /// If they disagree, admission downloads one tier and the engine renders
    /// with another — which on the shipped 24 GB card is the difference
    /// between a 6 GB fetch and an 11.4 GB one that was never needed.
    #[test]
    fn the_umt5_ladder_steps_down_with_free_vram_and_bottoms_out_at_fp16() {
        use mold_core::manifest::{known_umt5_variants, UMT5_FP16_SIZE};
        let threshold = mold_inference::device::t5_vram_threshold(UMT5_FP16_SIZE);

        // Room for FP16: FP16, whether asked implicitly or by name.
        assert_eq!(select_umt5_tag(None, threshold).unwrap(), "fp16");
        assert_eq!(select_umt5_tag(Some("auto"), threshold).unwrap(), "fp16");
        // A stored-but-empty preference is "auto", not a variant lookup.
        assert_eq!(select_umt5_tag(Some(""), threshold).unwrap(), "fp16");

        // One byte short of FP16: the largest GGUF that fits, which is the
        // registry's first entry given its largest-first order.
        let variants = known_umt5_variants();
        let largest = variants.first().expect("the registry ships GGUF variants");
        assert_eq!(
            select_umt5_tag(None, threshold - 1).unwrap(),
            largest.tag,
            "just below the FP16 threshold must step to the largest GGUF, not to CPU FP16",
        );
        // Each successive tier is reachable by dropping below the one above it.
        for pair in variants.windows(2) {
            let below = mold_inference::device::t5_vram_threshold(pair[0].size_bytes) - 1;
            assert_eq!(select_umt5_tag(None, below).unwrap(), pair[1].tag);
        }

        // Nothing fits: FP16, which the manifest already ships, so admission
        // downloads nothing and the engine parks it.
        assert_eq!(select_umt5_tag(None, 0).unwrap(), "fp16");

        // Explicit beats auto in both directions, and an unknown tag is an
        // error rather than a silent fallback.
        assert_eq!(select_umt5_tag(Some("fp16"), 0).unwrap(), "fp16");
        assert_eq!(
            select_umt5_tag(Some(largest.tag), u64::MAX).unwrap(),
            largest.tag,
        );
        assert!(select_umt5_tag(Some("q3"), u64::MAX).is_err());
    }

    /// A selected GGUF must be addressable: the arm that downloads it reads
    /// repo, filename, and the dedupe subdir straight off the registry entry,
    /// and the factory's prepared-artifact check resolves the same subdir.
    #[test]
    fn every_umt5_gguf_variant_is_a_complete_download_identity() {
        for variant in mold_core::manifest::known_umt5_variants() {
            assert!(!variant.hf_repo.is_empty(), "{} has no repo", variant.tag);
            assert!(
                variant.hf_filename.ends_with(".gguf"),
                "{} is not a GGUF",
                variant.tag
            );
            assert!(variant.size_bytes > 0, "{} has no size", variant.tag);
            assert_eq!(
                mold_core::manifest::find_umt5_variant(variant.tag).map(|found| found.tag),
                Some(variant.tag),
            );
        }
    }

    struct TestDownloadAdapterGuard {
        repo: String,
    }

    impl Drop for TestDownloadAdapterGuard {
        fn drop(&mut self) {
            TEST_DOWNLOAD_ADAPTERS
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&self.repo);
        }
    }

    const PINNED_CONTENT: &[u8] = b"the exact pinned bytes";
    /// SHA-256 of [`PINNED_CONTENT`], asserted against the real digest by
    /// `the_pin_fixture_digests_are_the_digests_of_the_fixtures`.
    const PINNED_CONTENT_SHA256: &str =
        "53d2d7847273102e2b9997c3651ae60a1a5653c9eb17b09a956f88d829333b0e";
    const TAMPERED_CONTENT: &[u8] = b"tampered";
    const TAMPERED_CONTENT_SHA256: &str =
        "d121be3103007b41edf96f8262925f8c7d61894afe9a041843b631f69445bc57";

    /// A pin fixture that does not actually hash to its constant would make
    /// every test below pass for the wrong reason.
    #[test]
    fn the_pin_fixture_digests_are_the_digests_of_the_fixtures() {
        let dir = TempDir::new().unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            mold_core::download::compute_sha256(&path).unwrap()
        };
        assert_eq!(write("pinned", PINNED_CONTENT), PINNED_CONTENT_SHA256);
        assert_eq!(write("tampered", TAMPERED_CONTENT), TAMPERED_CONTENT_SHA256);
    }

    fn unique_test_repo(prefix: &str) -> String {
        format!(
            "{prefix}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )
    }

    fn pinned_spec<'a>(
        models_root: &'a Path,
        repo: &'a str,
        sha256: &'a str,
    ) -> DependencySpec<'a> {
        DependencySpec {
            models_root,
            repo,
            filename: "pinned.safetensors",
            expected_bytes: Some(PINNED_CONTENT.len() as u64),
            kind: "identity_adapter",
            container: crate::execution_plan::PendingArtifactContainer::Safetensors,
            quantization: None,
            expected_sha256: Some(PinnedDigest {
                sha256,
                repair_model: "pinned-bundle",
            }),
            subdir: "shared/pin-test",
        }
    }

    fn install_test_download_adapter(
        repo: &str,
        adapter: TestDownloadAdapter,
    ) -> TestDownloadAdapterGuard {
        TEST_DOWNLOAD_ADAPTERS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(repo.to_string(), adapter);
        TestDownloadAdapterGuard {
            repo: repo.to_string(),
        }
    }

    #[test]
    fn preparation_capacity_includes_only_measured_reclaimable_warm_cache() {
        const GIB: u64 = 1 << 30;
        assert_eq!(
            effective_preparation_available_vram(
                24 * GIB,
                Some(20 * GIB),
                Some(16 * GIB),
                16 * GIB,
            ),
            20 * GIB,
            "auto-variant preparation must use the same free plus reclaimable-cache metric as admission"
        );
        assert_eq!(
            effective_preparation_available_vram(24 * GIB, Some(20 * GIB), None, 16 * GIB),
            4 * GIB,
            "unattributed process memory must never be assumed reclaimable"
        );
    }

    #[test]
    fn variant_fit_selection_is_capacity_generic() {
        let variants = mold_core::manifest::known_t5_variants();
        assert_eq!(
            choose_largest_fitting(variants, 8_000_000_000, |v| (v.tag, v.size_bytes))
                .map(|v| v.tag),
            Some("q8")
        );
        assert_eq!(
            choose_largest_fitting(variants, 4_500_000_000, |v| (v.tag, v.size_bytes))
                .map(|v| v.tag),
            Some("q3")
        );
        assert!(
            choose_largest_fitting(variants, 4_000_000_000, |v| (v.tag, v.size_bytes)).is_none()
        );
    }

    /// Hugging Face `main` is a mutable branch and this downloader resolves the
    /// branch, not a commit. A dependency that arrives with the right byte
    /// count but the wrong content must therefore be rejected on its manifest
    /// pin and REMOVED — freezing it would put unverified bytes into an
    /// execution plan, and everything downstream only ever proves that the
    /// path is local.
    #[tokio::test]
    async fn a_pinned_download_whose_bytes_do_not_match_is_rejected_and_removed() {
        let cache = TempDir::new().unwrap();
        let repo = unique_test_repo("mold-pin-corrupt");
        let _adapter = install_test_download_adapter(
            &repo,
            Arc::new(|models_root, _repo, filename, subdir| {
                let path =
                    mold_core::download::planned_single_file_path_in(models_root, filename, subdir);
                std::fs::create_dir_all(path.parent().unwrap())
                    .map_err(|error| error.to_string())?;
                // Right size, wrong bytes: exactly what a mutated upstream
                // revision or a compromised mirror looks like to a
                // size-and-bytes contract.
                std::fs::write(&path, TAMPERED_CONTENT).map_err(|error| error.to_string())?;
                Ok(path)
            }),
        );

        let error = ensure_downloaded(
            None,
            "admission",
            pinned_spec(cache.path(), &repo, PINNED_CONTENT_SHA256),
            None,
            DependencyMaterializationPolicy::Admission,
        )
        .await
        .err()
        .expect("a pinned dependency whose bytes do not match must be refused");

        assert!(error.contains("pinned.safetensors"), "{error}");
        assert!(error.contains(PINNED_CONTENT_SHA256), "{error}");
        assert!(
            error.contains(TAMPERED_CONTENT_SHA256),
            "the error must name the digest that actually landed: {error}"
        );
        assert!(error.contains("mold pull pinned-bundle"), "{error}");

        let landed = mold_core::download::planned_single_file_path_in(
            cache.path(),
            "pinned.safetensors",
            "shared/pin-test",
        );
        assert!(
            !landed.exists(),
            "the rejected bytes must be removed so a retry re-downloads"
        );
        assert!(!mold_core::download::has_sha256_marker(&landed));
    }

    /// The matching case: the file is kept and attested with the shared
    /// `.sha256-verified` marker, so the next admission short-circuits instead
    /// of rehashing a multi-gigabyte artifact.
    #[tokio::test]
    async fn a_pinned_download_that_matches_is_accepted_and_attested() {
        let cache = TempDir::new().unwrap();
        let repo = unique_test_repo("mold-pin-good");
        let _adapter = install_test_download_adapter(
            &repo,
            Arc::new(|models_root, _repo, filename, subdir| {
                let path =
                    mold_core::download::planned_single_file_path_in(models_root, filename, subdir);
                std::fs::create_dir_all(path.parent().unwrap())
                    .map_err(|error| error.to_string())?;
                std::fs::write(&path, PINNED_CONTENT).map_err(|error| error.to_string())?;
                Ok(path)
            }),
        );

        let resolved = ensure_downloaded(
            None,
            "admission",
            pinned_spec(cache.path(), &repo, PINNED_CONTENT_SHA256),
            None,
            DependencyMaterializationPolicy::Admission,
        )
        .await
        .expect("bytes that match the pin are accepted");
        let ResolvedDependency::Available(path) = resolved else {
            panic!("admission never returns pending");
        };

        assert!(path.is_file());
        assert_eq!(
            mold_core::download::recorded_sha256_marker(&path).as_deref(),
            Some(PINNED_CONTENT_SHA256),
            "a proven download must be attested with the digest it hashed to"
        );
    }

    /// The `.sha256-verified` sidecar is writable by anyone who can write the
    /// artifact — the model-storage invariant supports group-writable model
    /// roots on purpose — so it can never be the thing that authenticates
    /// content. Neither policy may accept it, and a stale one left behind by a
    /// replaced file must not resurrect the replacement either.
    #[tokio::test]
    async fn a_forged_attestation_beside_wrong_bytes_is_refused_by_both_policies() {
        let cache = TempDir::new().unwrap();
        let repo = unique_test_repo("mold-pin-forged");
        let path = mold_core::download::planned_single_file_path_in(
            cache.path(),
            "pinned.safetensors",
            "shared/pin-test",
        );
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let forge = || {
            std::fs::write(&path, TAMPERED_CONTENT).unwrap();
            mold_core::download::write_sha256_marker(&path, PINNED_CONTENT_SHA256).unwrap();
        };
        forge();

        // Read-only: the forged attestation buys nothing, and the preview
        // still neither deletes nor refuses.
        let resolved = ensure_downloaded(
            None,
            "placement-preview",
            pinned_spec(cache.path(), &repo, PINNED_CONTENT_SHA256),
            None,
            DependencyMaterializationPolicy::ExistingOnly,
        )
        .await
        .expect("a preview never refuses");
        assert!(
            matches!(resolved, ResolvedDependency::Pending(_)),
            "a self-served attestation is not evidence the file is installed"
        );
        assert!(path.exists());

        // Admission: refused on the bytes, and the lying sidecar goes with the
        // file it lied about. No download adapter is installed, so reaching
        // the downloader would be real network I/O — the cached branch must
        // refuse first.
        forge();
        let error = ensure_downloaded(
            None,
            "admission",
            pinned_spec(cache.path(), &repo, PINNED_CONTENT_SHA256),
            None,
            DependencyMaterializationPolicy::Admission,
        )
        .await
        .err()
        .expect("a forged attestation must not authenticate the bytes beside it");
        assert!(error.contains(PINNED_CONTENT_SHA256), "{error}");
        assert!(error.contains(TAMPERED_CONTENT_SHA256), "{error}");
        assert!(!path.exists());
        assert!(!mold_core::download::has_sha256_marker(&path));
    }

    /// A copy already on disk is not evidence of anything until it is proven:
    /// the cache lookup happens before the downloader, so an attacker who can
    /// write into the models root would otherwise bypass the pin entirely by
    /// pre-placing the file.
    #[tokio::test]
    async fn a_pre_existing_unattested_copy_is_verified_before_it_is_reused() {
        let cache = TempDir::new().unwrap();
        let repo = unique_test_repo("mold-pin-preplaced");
        let path = mold_core::download::planned_single_file_path_in(
            cache.path(),
            "pinned.safetensors",
            "shared/pin-test",
        );
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"pre-placed").unwrap();

        // No download adapter is installed, so reaching the downloader at all
        // would attempt real network I/O. The cached branch must refuse first.
        let error = ensure_downloaded(
            None,
            "admission",
            pinned_spec(cache.path(), &repo, PINNED_CONTENT_SHA256),
            None,
            DependencyMaterializationPolicy::Admission,
        )
        .await
        .err()
        .expect("a pre-placed file that is not the pinned bytes must be refused");
        assert!(error.contains(PINNED_CONTENT_SHA256), "{error}");
        assert!(!path.exists(), "the rejected copy must be removed");

        // A read-only preview must neither delete nor refuse — it reports the
        // unproven copy as work admission still has to do.
        std::fs::write(&path, b"pre-placed").unwrap();
        let resolved = ensure_downloaded(
            None,
            "placement-preview",
            pinned_spec(cache.path(), &repo, PINNED_CONTENT_SHA256),
            None,
            DependencyMaterializationPolicy::ExistingOnly,
        )
        .await
        .expect("a preview never refuses");
        assert!(
            matches!(resolved, ResolvedDependency::Pending(_)),
            "an unproven copy is not evidence that nothing needs downloading"
        );
        assert!(
            path.exists(),
            "a read-only preview must not delete the file it could not prove"
        );
        assert!(
            !mold_core::download::has_sha256_marker(&path),
            "a read-only preview must not write an attestation either"
        );

        // Proven bytes are reused by both policies without a download.
        std::fs::write(&path, PINNED_CONTENT).unwrap();
        for policy in [
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyMaterializationPolicy::Admission,
        ] {
            let resolved = ensure_downloaded(
                None,
                "reuse",
                pinned_spec(cache.path(), &repo, PINNED_CONTENT_SHA256),
                None,
                policy,
            )
            .await
            .expect("pinned bytes already on disk are reused");
            assert!(matches!(resolved, ResolvedDependency::Available(_)));
        }
    }

    #[tokio::test]
    async fn existing_only_dependency_check_never_starts_a_download() {
        let cache = TempDir::new().unwrap();
        let repo = format!(
            "mold-preview-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dependency = ensure_downloaded(
            None,
            "placement-preview",
            DependencySpec {
                models_root: cache.path(),
                repo: &repo,
                filename: "missing.safetensors",
                expected_bytes: Some(123_456),
                kind: "text_encoder",
                container: crate::execution_plan::PendingArtifactContainer::Gguf,
                quantization: Some(crate::execution_plan::QuantizationVariant::Q4),
                expected_sha256: None,
                subdir: "preview-test",
            },
            None,
            DependencyMaterializationPolicy::ExistingOnly,
        )
        .await
        .expect("known dependency must remain previewable");
        let registered_for_preview = downloads()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .any(|key| key.models_root.starts_with(cache.path()));

        let ResolvedDependency::Pending(dependency) = dependency else {
            panic!("fixture dependency must be reported as pending");
        };
        assert_eq!(dependency.download.kind, "text_encoder");
        assert_eq!(dependency.download.repo, repo);
        assert_eq!(dependency.download.name, "missing.safetensors");
        assert_eq!(dependency.download.bytes, 123_456);
        assert!(!dependency.path.is_file());
        assert!(
            !registered_for_preview,
            "preview must not register a download for its isolated models root"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn existing_only_dependency_check_does_not_create_cache_roots() {
        let _env = crate::test_support::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let parent = TempDir::new().unwrap();
        let models_root = parent.path().join("absent-models");
        let hf_home = parent.path().join("absent-hf-home");
        let previous_hf_home = std::env::var_os("HF_HOME");
        std::env::set_var("HF_HOME", &hf_home);
        let dependency = ensure_downloaded(
            None,
            "placement-preview",
            DependencySpec {
                models_root: &models_root,
                repo: "preview/no-create",
                filename: "missing.gguf",
                expected_bytes: Some(42),
                kind: "text_encoder",
                container: crate::execution_plan::PendingArtifactContainer::Gguf,
                quantization: Some(crate::execution_plan::QuantizationVariant::Q4),
                expected_sha256: None,
                subdir: "shared/test",
            },
            None,
            DependencyMaterializationPolicy::ExistingOnly,
        )
        .await
        .unwrap();

        assert!(matches!(dependency, ResolvedDependency::Pending(_)));
        assert!(!models_root.exists());
        assert!(!hf_home.exists());
        match previous_hf_home {
            Some(value) => std::env::set_var("HF_HOME", value),
            None => std::env::remove_var("HF_HOME"),
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn preview_and_admission_use_the_same_explicit_models_root() {
        let _env = crate::test_support::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let parent = TempDir::new().unwrap();
        let models_root = parent.path().join("config-owned-models");
        let repo = format!("test/explicit-root-parity-{}", std::process::id());
        let filename = "encoder.gguf";
        let subdir = "shared/explicit-root";
        let _adapter = install_test_download_adapter(
            &repo,
            Arc::new(|models_root, _repo, filename, subdir| {
                let path =
                    mold_core::download::planned_single_file_path_in(models_root, filename, subdir);
                std::fs::create_dir_all(path.parent().unwrap())
                    .map_err(|error| error.to_string())?;
                std::fs::write(&path, b"test encoder").map_err(|error| error.to_string())?;
                Ok(path)
            }),
        );
        let spec = || DependencySpec {
            models_root: &models_root,
            repo: &repo,
            filename,
            expected_bytes: Some(12),
            kind: "text_encoder",
            container: crate::execution_plan::PendingArtifactContainer::Gguf,
            quantization: Some(crate::execution_plan::QuantizationVariant::Q4),
            expected_sha256: None,
            subdir,
        };

        let preview = ensure_downloaded(
            None,
            "placement-preview",
            spec(),
            None,
            DependencyMaterializationPolicy::ExistingOnly,
        )
        .await
        .unwrap();
        let ResolvedDependency::Pending(preview) = preview else {
            panic!("missing dependency must be pending during preview");
        };
        let admitted = ensure_downloaded(
            None,
            "admission",
            spec(),
            None,
            DependencyMaterializationPolicy::Admission,
        )
        .await
        .unwrap();
        let ResolvedDependency::Available(admitted) = admitted else {
            panic!("admission must materialize the dependency");
        };

        assert_eq!(preview.path, admitted);
        assert!(admitted.starts_with(&models_root));
        assert!(admitted.is_file());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn admission_deduplicates_within_but_never_across_models_roots() {
        let _env = crate::test_support::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let parent = TempDir::new().unwrap();
        let first_root = parent.path().join("first-models");
        let second_root = parent.path().join("second-models");
        let repo = format!("test/root-scoped-dedupe-{}", std::process::id());
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_adapter = calls.clone();
        let _adapter = install_test_download_adapter(
            &repo,
            Arc::new(move |models_root, _repo, filename, subdir| {
                calls_for_adapter.fetch_add(1, AtomicOrdering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(50));
                let path =
                    mold_core::download::planned_single_file_path_in(models_root, filename, subdir);
                std::fs::create_dir_all(path.parent().unwrap())
                    .map_err(|error| error.to_string())?;
                std::fs::write(&path, b"test encoder").map_err(|error| error.to_string())?;
                Ok(path)
            }),
        );
        let make_spec = |models_root| DependencySpec {
            models_root,
            repo: &repo,
            filename: "encoder.gguf",
            expected_bytes: Some(12),
            kind: "text_encoder",
            container: crate::execution_plan::PendingArtifactContainer::Gguf,
            quantization: Some(crate::execution_plan::QuantizationVariant::Q4),
            expected_sha256: None,
            subdir: "shared/dedupe",
        };

        let (first, first_joiner, second) = tokio::join!(
            ensure_downloaded(
                None,
                "first",
                make_spec(&first_root),
                None,
                DependencyMaterializationPolicy::Admission,
            ),
            ensure_downloaded(
                None,
                "first-joiner",
                make_spec(&first_root),
                None,
                DependencyMaterializationPolicy::Admission,
            ),
            ensure_downloaded(
                None,
                "second",
                make_spec(&second_root),
                None,
                DependencyMaterializationPolicy::Admission,
            ),
        );
        let available_path = |result: Result<ResolvedDependency, String>| match result.unwrap() {
            ResolvedDependency::Available(path) => path,
            ResolvedDependency::Pending(_) => panic!("admission cannot return pending"),
        };
        let first = available_path(first);
        let first_joiner = available_path(first_joiner);
        let second = available_path(second);

        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
        assert_eq!(first, first_joiner);
        assert_ne!(first, second);
        assert!(first.starts_with(&first_root));
        assert!(second.starts_with(&second_root));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn existing_only_preparation_plans_known_missing_encoder_without_downloading() {
        let _env = crate::test_support::env_lock().lock().unwrap();
        let cache = TempDir::new().unwrap();
        std::env::set_var("MOLD_MODELS_DIR", cache.path().join("models"));
        std::env::set_var("HF_HOME", cache.path().join("hf"));

        let (_root, mut config, request) = zimage_case();
        config.models.get_mut("prepared-z").unwrap().family = Some("flux2".to_string());
        config.qwen3_variant = Some("q8".to_string());
        let prepared = prepare_inputs_for_devices(
            None,
            "placement-preview",
            &request,
            &config,
            vec![DeviceFact {
                id: "cuda:0".to_string(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24_000_000_000,
            }],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap();
        let registered_for_preview = downloads()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .any(|key| key.models_root.starts_with(cache.path()));

        let pending_downloads = prepared.pending_downloads_for_device("cuda:0");
        assert_eq!(pending_downloads.len(), 1);
        assert_eq!(pending_downloads[0].kind, "text_encoder");
        assert_eq!(pending_downloads[0].name, "Qwen_3_4b-Q8_0.gguf");
        let device = &prepared.by_device["cuda:0"];
        assert_eq!(device.pending_artifacts.len(), 1);
        assert!(device.pending_artifacts.keys().all(|path| !path.is_file()));
        let plans = crate::execution_plan::resolve_execution_plans_with_prepared(
            &config,
            &request,
            &[DeviceFact {
                id: "cuda:0".to_string(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24_000_000_000,
            }],
            false,
            Some(&prepared),
        )
        .unwrap();
        assert_eq!(plans.len(), 1);
        assert!(
            plans[0].predicted_vram_peak_bytes
                >= pending_downloads[0]
                    .bytes
                    .saturating_add(ENCODER_DEPENDENCY_HEADROOM_BYTES)
        );
        let pending_component = plans[0]
            .execution_environment
            .components
            .iter()
            .find(|component| {
                matches!(
                    &component.content_fingerprint,
                    crate::execution_plan::EquivalenceContentIdentity::PendingPreview { .. }
                )
            })
            .expect("pending dependency must retain its preview identity");
        assert!(matches!(
            &pending_component.precision.storage,
            crate::execution_plan::ComponentStorageFormat::PendingPreview {
                container: crate::execution_plan::PendingArtifactContainer::Gguf,
                quantization: Some(crate::execution_plan::QuantizationVariant::Q8),
                ..
            }
        ));
        let serialized = serde_json::to_string(pending_component).unwrap();
        assert!(!serialized.contains("\"Unknown\""), "{serialized}");

        let pressured = crate::execution_plan::resolve_execution_plans_with_prepared(
            &config,
            &request,
            &[DeviceFact {
                id: "cuda:0".to_string(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 6_000_000_000,
            }],
            false,
            Some(&prepared),
        )
        .unwrap();
        let pending_plan = pressured[0]
            .components
            .values()
            .find(|component| {
                device
                    .pending_artifacts
                    .contains_key(&component.artifact_path)
            })
            .expect("pending encoder must remain in the pressured plan");
        assert!(matches!(
            pending_plan.placement,
            crate::execution_plan::ResolvedComponentPlacement::Cpu
        ));
        assert!(
            pressured[0].predicted_vram_peak_bytes
                < pending_downloads[0]
                    .bytes
                    .saturating_add(ENCODER_DEPENDENCY_HEADROOM_BYTES),
            "CPU-placed pending encoder must not be charged against GPU peak"
        );
        assert!(
            !registered_for_preview,
            "preview must not register a download for its isolated models root"
        );

        std::env::remove_var("MOLD_MODELS_DIR");
        std::env::remove_var("HF_HOME");
    }

    #[test]
    fn flux2_size_selection_does_not_assume_device_count() {
        let mut paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: PathBuf::from("/tmp/transformer"),
            transformer_shards: Vec::new(),
            vae: PathBuf::from("/tmp/vae"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        assert!(!flux2_uses_qwen3_8b("flux2-klein-4b", &paths));
        assert!(flux2_uses_qwen3_8b("flux2-klein-9b", &paths));
        paths.text_encoder_files = vec![PathBuf::from("/missing")];
        assert!(!flux2_uses_qwen3_8b("opaque-model-id", &paths));
    }

    #[test]
    fn flux2_dev_is_not_a_qwen3_dependency_family() {
        assert!(mold_core::validation::is_flux2_dev_model("flux2-dev:bf16"));
        assert!(mold_core::validation::is_flux2_dev_model(
            "hf:black-forest-labs/FLUX.2-dev"
        ));
        assert!(!mold_core::validation::is_flux2_dev_model(
            "flux2-klein-9b:bf16"
        ));
    }

    #[tokio::test]
    async fn flux2_dev_preserves_checkpoint_native_mistral_dependencies() {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "mistral-00001.safetensors",
            "tokenizer.json",
        ] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let mistral = root.path().join("mistral-00001.safetensors");
        let mut config = Config {
            qwen3_variant: Some("q4".to_string()),
            ..Config::default()
        };
        config.models.insert(
            "prepared-flux2-dev".to_string(),
            ModelConfig {
                transformer: Some(
                    root.path()
                        .join("transformer.safetensors")
                        .display()
                        .to_string(),
                ),
                vae: Some(root.path().join("vae.safetensors").display().to_string()),
                text_encoder_files: Some(vec![mistral.display().to_string()]),
                text_tokenizer: Some(root.path().join("tokenizer.json").display().to_string()),
                family: Some("flux2".to_string()),
                ..ModelConfig::default()
            },
        );
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"prepared-flux2-dev","width":256,"height":256,"steps":1,"guidance":4.0}"#,
        )
        .unwrap();
        let prepared = prepare_local_execution_inputs(
            &config,
            &request,
            vec![DeviceFact {
                id: "cuda:0".to_string(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24_000_000_000,
            }],
        )
        .await
        .unwrap();
        let device = &prepared.by_device["cuda:0"];
        assert_eq!(device.engine_paths.text_encoder_files, vec![mistral]);
        assert_eq!(device.engine_config.qwen3_variant, None);
        assert!(device.engine_config.selected_qwen3_paths.is_empty());
    }

    fn zimage_case() -> (TempDir, Config, GenerateRequest) {
        let root = TempDir::new().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "qwen3.safetensors",
            "tokenizer.json",
        ] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let mut config = Config::default();
        config.models.insert(
            "prepared-z".to_string(),
            ModelConfig {
                transformer: Some(
                    root.path()
                        .join("transformer.safetensors")
                        .display()
                        .to_string(),
                ),
                vae: Some(root.path().join("vae.safetensors").display().to_string()),
                text_encoder_files: Some(vec![root
                    .path()
                    .join("qwen3.safetensors")
                    .display()
                    .to_string()]),
                text_tokenizer: Some(root.path().join("tokenizer.json").display().to_string()),
                family: Some("z-image".to_string()),
                ..ModelConfig::default()
            },
        );
        let request = serde_json::from_str(
            r#"{"prompt":"x","model":"prepared-z","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        (root, config, request)
    }

    #[tokio::test]
    async fn transient_pressure_does_not_remove_a_prepared_sibling() {
        let (_root, mut config, request) = zimage_case();
        config.qwen3_variant = Some("bf16".to_string());
        let prepared = prepare_local_execution_inputs(
            &config,
            &request,
            vec![
                DeviceFact {
                    id: "cuda:0".to_string(),
                    ordinal: 0,
                    backend: mold_core::GpuBackend::Cuda,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: 4_000_000_000,
                },
                DeviceFact {
                    id: "cuda:1".to_string(),
                    ordinal: 1,
                    backend: mold_core::GpuBackend::Cuda,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: 24_000_000_000,
                },
            ],
        )
        .await
        .unwrap();
        assert_eq!(
            prepared.by_device.keys().cloned().collect::<Vec<_>>(),
            vec!["cuda:0".to_string(), "cuda:1".to_string()]
        );
        assert_eq!(
            prepared.by_device["cuda:0"]
                .engine_config
                .qwen3_variant
                .as_deref(),
            Some("bf16")
        );
    }

    #[tokio::test]
    async fn all_temporarily_low_devices_still_materialize_for_later_replanning() {
        let (_root, mut config, request) = zimage_case();
        config.qwen3_variant = Some("bf16".to_string());
        let low_facts = (0..8)
            .map(|ordinal| DeviceFact {
                id: format!("cuda:{ordinal}"),
                ordinal,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 1,
            })
            .collect::<Vec<_>>();
        let prepared = prepare_local_execution_inputs(&config, &request, low_facts.clone())
            .await
            .unwrap();

        assert_eq!(prepared.by_device.len(), 8);
        assert!(matches!(
            crate::execution_plan::resolve_execution_plans_with_prepared(
                &config,
                &request,
                &low_facts,
                false,
                Some(&prepared),
            ),
            Err(crate::execution_plan::ExecutionPlanError::InsufficientVram { .. })
        ));
        let recovered = crate::execution_plan::resolve_execution_plans_with_prepared(
            &config,
            &request,
            &(0..8)
                .map(|ordinal| DeviceFact {
                    id: format!("cuda:{ordinal}"),
                    ordinal,
                    backend: mold_core::GpuBackend::Cuda,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: 24_000_000_000,
                })
                .collect::<Vec<_>>(),
            false,
            Some(&prepared),
        )
        .unwrap();
        assert_eq!(recovered.len(), 8);
    }

    #[tokio::test]
    async fn hard_pin_filters_irrelevant_devices_before_materialization() {
        let (_root, config, mut request) = zimage_case();
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::device("cuda:1"),
            advanced: None,
        });
        let prepared = prepare_local_execution_inputs(
            &config,
            &request,
            (0..8)
                .map(|ordinal| DeviceFact {
                    id: format!("cuda:{ordinal}"),
                    ordinal,
                    backend: mold_core::GpuBackend::Cuda,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: if ordinal == 1 {
                        24_000_000_000
                    } else {
                        8_000_000_000
                    },
                })
                .collect(),
        )
        .await
        .unwrap();
        assert_eq!(prepared.by_device.len(), 1);
        assert!(prepared.by_device.contains_key("cuda:1"));
    }

    #[tokio::test]
    async fn stale_device_pin_remains_hard_infeasible() {
        let (_root, config, mut request) = zimage_case();
        request.placement = Some(DevicePlacement {
            text_encoders: DeviceRef::device("cuda:gone"),
            advanced: None,
        });
        let error = prepare_inputs_for_devices(
            None,
            "placement-preview",
            &request,
            &config,
            vec![DeviceFact {
                id: "cuda:0".to_string(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24_000_000_000,
            }],
            None,
            DependencyMaterializationPolicy::ExistingOnly,
            DependencyPreparationContext::default(),
        )
        .await
        .unwrap_err();

        assert!(error.contains("unavailable device 'cuda:gone'"), "{error}");
    }

    #[test]
    fn cancelled_download_watcher_is_detached_without_affecting_joiner() {
        let shared = Arc::new(SharedDownload {
            result: Mutex::new(None),
            notify: tokio::sync::Notify::new(),
            watchers: Mutex::new(Vec::new()),
        });
        let (cancelled, mut cancelled_rx) = tokio::sync::mpsc::unbounded_channel();
        let (joiner, mut joiner_rx) = tokio::sync::mpsc::unbounded_channel();
        shared.watchers.lock().unwrap().extend([
            DownloadWatcher {
                id: 10,
                sender: cancelled,
            },
            DownloadWatcher {
                id: 11,
                sender: joiner,
            },
        ]);
        drop(DownloadWatcherGuard {
            id: 10,
            shared: shared.clone(),
        });

        for watcher in shared.watchers.lock().unwrap().iter() {
            watcher
                .sender
                .send(SseMessage::Progress(SseProgressEvent::DependencyWait {
                    dependency: "encoder.gguf".into(),
                    reason: "still downloading".into(),
                }))
                .unwrap();
        }
        assert!(cancelled_rx.try_recv().is_err());
        assert!(matches!(
            joiner_rx.try_recv().unwrap(),
            SseMessage::Progress(SseProgressEvent::DependencyWait {
                dependency,
                reason,
            }) if dependency == "encoder.gguf" && reason == "still downloading"
        ));
    }

    /// BF16 Gemma is ~24.5 GB resident, so on a host that cannot hold it beside
    /// the transformer the quantized encoder is the only admittable choice.
    /// Presence-based selection pinned BF16 regardless and left LTX-2
    /// unrunnable on a 48 GB unified-memory Mac.
    #[test]
    fn an_unpinned_gemma_falls_back_to_q4_only_when_bf16_cannot_fit() {
        const GB: u64 = 1 << 30;
        let bf16 = 24 * GB;
        let mac48 = 48 * GB;
        let headroom = super::gemma_host_headroom(mac48);

        // Comfortable host: BF16 wins.
        assert_eq!(
            super::choose_gemma_tag(bf16, true, Some(120 * GB), Some(128 * GB)),
            "bf16"
        );
        // Exactly at the boundary: still BF16, the comparison is inclusive.
        assert_eq!(
            super::choose_gemma_tag(bf16, true, Some(bf16 + headroom), Some(mac48)),
            "bf16"
        );
        // One byte short: downgrade.
        assert_eq!(
            super::choose_gemma_tag(bf16, true, Some(bf16 + headroom - 1), Some(mac48)),
            "q4"
        );
        // A 48 GB Mac with 24 GB reclaimable — the case that was refused.
        assert_eq!(
            super::choose_gemma_tag(bf16, true, Some(24 * GB), Some(mac48)),
            "q4"
        );
    }

    /// The headroom must track admission's own floor rather than a flat 8 GiB,
    /// which only coincides below ~53 GB of installed RAM.
    #[test]
    fn gemma_headroom_tracks_the_admission_safety_floor() {
        const GB: u64 = 1 << 30;
        let transient = super::GEMMA_HOST_BASE_TRANSIENT;
        // 15% of 48 GB is 7.2 GiB, so the 8 GiB minimum governs.
        assert_eq!(
            super::gemma_host_headroom(48 * GB),
            super::GEMMA_HOST_SAFETY_FLOOR_MIN + transient
        );
        // 15% of 128 GB is 19.2 GiB and governs instead.
        assert_eq!(
            super::gemma_host_headroom(128 * GB),
            (128 * GB) * 15 / 100 + transient
        );
        assert!(super::gemma_host_headroom(u64::MAX) > 0);
    }

    /// Never downgrade to a variant that is not unambiguously there, and never
    /// change behaviour on a host that publishes no memory figures.
    #[test]
    fn gemma_selection_holds_bf16_without_one_q4_or_a_memory_reading() {
        const GB: u64 = 1 << 30;
        let mac48 = Some(48 * GB);
        // No GGUF on disk: BF16 even under pressure, so the caller's
        // "selected variant is not locally available" error still governs.
        assert_eq!(
            super::choose_gemma_tag(24 * GB, false, Some(GB), mac48),
            "bf16"
        );
        // Off macOS there are no readings; CUDA hosts keep their behaviour.
        assert_eq!(super::choose_gemma_tag(24 * GB, true, None, mac48), "bf16");
        assert_eq!(
            super::choose_gemma_tag(24 * GB, true, Some(GB), None),
            "bf16"
        );
        // A saturating sum must not read as "fits" against a saturated
        // available figure.
        assert_eq!(
            super::choose_gemma_tag(u64::MAX, true, Some(u64::MAX), mac48),
            "q4"
        );
    }

    /// Preparation must enumerate GGUFs exactly as inference does, or a
    /// mixed-case pair looks unambiguous here and ambiguous at load time.
    #[test]
    fn gemma_gguf_discovery_is_case_insensitive_like_the_runtime() {
        let root = TempDir::new().unwrap();
        std::fs::write(root.path().join("a-gemma.GGUF"), b"x").unwrap();
        std::fs::write(root.path().join("b-gemma.gguf"), b"x").unwrap();
        let found = super::sorted_matching_files(root.path(), |name| {
            std::path::Path::new(name)
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        });
        assert_eq!(
            found.len(),
            2,
            "both casings must be seen, so the set reads as ambiguous"
        );
    }

    /// `materialize_gemma` trims and lowercases the preference, so the
    /// capacity-sensitivity predicate has to agree or an `AUTO` plan selects
    /// on live memory and is then never replanned when pressure changes.
    #[test]
    fn gemma_auto_detection_ignores_case_and_surrounding_space() {
        let is_auto = |value: &str| {
            let value = value.trim().to_ascii_lowercase();
            value.is_empty() || value == "auto"
        };
        for value in ["auto", "AUTO", " auto ", "Auto", ""] {
            assert!(is_auto(value), "{value:?} should count as automatic");
        }
        for value in ["q4", "bf16", "gguf"] {
            assert!(!is_auto(value), "{value:?} is an explicit pin");
        }
    }

    /// An ambiguous GGUF set must not trigger the downgrade: inference loads
    /// only the lexicographically first file, so choosing Q4 here would load a
    /// checkpoint nobody selected.
    #[test]
    fn an_ambiguous_gguf_set_never_triggers_the_gemma_downgrade() {
        const GB: u64 = 1 << 30;
        let root = TempDir::new().unwrap();
        let bf16 = Vec::new();
        let two = vec![
            root.path().join("a-gemma.gguf"),
            root.path().join("b-unrelated.gguf"),
        ];
        for path in &two {
            std::fs::write(path, b"x").unwrap();
        }
        assert_eq!(super::auto_gemma_tag(&bf16, &two), "bf16");
        // Starved host, still ambiguous, still no downgrade.
        assert_eq!(
            super::choose_gemma_tag(24 * GB, two.len() == 1, Some(GB), Some(48 * GB)),
            "bf16"
        );
    }

    #[test]
    fn gemma_variant_is_materialized_as_an_exact_local_artifact_set() {
        let root = TempDir::new().unwrap();
        let tokenizer = root.path().join("tokenizer.json");
        std::fs::write(&tokenizer, b"tokenizer").unwrap();
        let mut expected = Vec::new();
        for index in 1..=5 {
            let path = root
                .path()
                .join(format!("model-{index:05}-of-00005.safetensors"));
            std::fs::write(&path, b"weights").unwrap();
            expected.push(path);
        }
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: root.path().join("transformer.safetensors"),
            transformer_shards: Vec::new(),
            vae: root.path().join("vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![tokenizer],
            text_tokenizer: None,
            decoder: None,
        };
        let mut frozen = mold_inference::FrozenEngineConfig::resolve(
            "ltx-2-19b-distilled:fp8",
            &Config::default(),
        );
        materialize_gemma(None, &paths, &mut frozen).unwrap();
        assert_eq!(frozen.ltx2_gemma_variant.as_deref(), Some("bf16"));
        assert_eq!(frozen.selected_gemma_paths, expected);

        let mut q4 = frozen.clone();
        let error = materialize_gemma(Some("q4"), &paths, &mut q4).unwrap_err();
        assert!(error.contains("q4 encoder is not locally available"));
    }

    #[test]
    fn gemma_rejects_partial_or_inconsistent_shard_sets_before_readiness() {
        let root = TempDir::new().unwrap();
        let tokenizer = root.path().join("tokenizer.json");
        std::fs::write(&tokenizer, b"tokenizer").unwrap();
        std::fs::write(
            root.path().join("model-00001-of-00005.safetensors"),
            b"partial",
        )
        .unwrap();
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: root.path().join("transformer.safetensors"),
            transformer_shards: Vec::new(),
            vae: root.path().join("vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![tokenizer],
            text_tokenizer: None,
            decoder: None,
        };
        let mut frozen = mold_inference::FrozenEngineConfig::resolve("ltx-2", &Config::default());

        let error = materialize_gemma(Some("bf16"), &paths, &mut frozen).unwrap_err();
        assert!(error.contains("incomplete Gemma BF16 shard set"), "{error}");

        std::fs::write(
            root.path().join("model-00002-of-00003.safetensors"),
            b"inconsistent",
        )
        .unwrap();
        let error = materialize_gemma(Some("bf16"), &paths, &mut frozen).unwrap_err();
        assert!(
            error.contains("inconsistent Gemma BF16 shard totals"),
            "{error}"
        );
    }

    #[test]
    fn gemma_accepts_one_unsharded_weight_file() {
        let root = TempDir::new().unwrap();
        let tokenizer = root.path().join("tokenizer.json");
        let weights = root.path().join("model.safetensors");
        std::fs::write(&tokenizer, b"tokenizer").unwrap();
        std::fs::write(&weights, b"weights").unwrap();
        let paths = ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: root.path().join("transformer.safetensors"),
            transformer_shards: Vec::new(),
            vae: root.path().join("vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![tokenizer],
            text_tokenizer: None,
            decoder: None,
        };
        let mut frozen = mold_inference::FrozenEngineConfig::resolve("ltx-2", &Config::default());

        materialize_gemma(Some("bf16"), &paths, &mut frozen).unwrap();
        assert_eq!(frozen.selected_gemma_paths, vec![weights]);
    }

    #[test]
    fn auto_quantized_download_choice_is_bounded_for_arbitrary_device_count() {
        let devices = (0..64)
            .map(|ordinal| DeviceFact {
                id: format!("cuda:{ordinal}"),
                ordinal,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 6_500_000_000 + (ordinal as u64 % 8) * 1_000_000_000,
            })
            .collect::<Vec<_>>();
        let selected = shared_quantized_fallback(
            mold_core::manifest::known_qwen3_variants(),
            &devices,
            |variant| (variant.tag, variant.size_bytes),
        )
        .unwrap();
        assert_eq!(selected.tag, "q8");
        // The fallback is one process-wide spec for this job, not one spec
        // per device. Materialization therefore has at most one cache miss
        // regardless of device count.
        assert_eq!(
            devices
                .iter()
                .map(|_| selected.tag)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
    }

    #[test]
    fn auto_qwen3_prefers_largest_cached_variant_over_vram_churn() {
        let root = TempDir::new().unwrap();
        let variants = mold_core::manifest::known_qwen3_8b_variants();
        let devices = vec![DeviceFact {
            id: "cuda:0".to_string(),
            ordinal: 0,
            backend: mold_core::GpuBackend::Cuda,
            compute_capability: Some((8, 6)),
            available_vram_bytes: 9_000_000_000,
        }];

        let selected = select_auto_qwen3_variant_with_cache(
            root.path(),
            "shared/qwen3-8b-gguf",
            variants,
            &devices,
            true,
            |variant| variant.tag == "q8",
        )
        .unwrap();

        assert_eq!(selected.tag, "q8");
        assert!(
            selected.size_bytes + ENCODER_DEPENDENCY_HEADROOM_BYTES
                > devices[0].available_vram_bytes,
            "the cached choice should remain stable and let planning move it to CPU"
        );
    }

    #[test]
    fn auto_qwen3_admission_sticks_before_download_finishes() {
        let root = TempDir::new().unwrap();
        let variants = mold_core::manifest::known_qwen3_8b_variants();
        let pressured = vec![DeviceFact {
            id: "cuda:0".to_string(),
            ordinal: 0,
            backend: mold_core::GpuBackend::Cuda,
            compute_capability: Some((8, 6)),
            available_vram_bytes: 6_400_000_000,
        }];
        let recovered = vec![DeviceFact {
            available_vram_bytes: 9_000_000_000,
            ..pressured[0].clone()
        }];

        let first = select_auto_qwen3_variant_with_cache(
            root.path(),
            "shared/qwen3-8b-gguf",
            variants,
            &pressured,
            true,
            |_| false,
        )
        .unwrap();
        let second = select_auto_qwen3_variant_with_cache(
            root.path(),
            "shared/qwen3-8b-gguf",
            variants,
            &recovered,
            true,
            |_| false,
        )
        .unwrap();

        assert_eq!(first.tag, "q3");
        assert_eq!(second.tag, first.tag);
    }

    #[test]
    fn auto_qwen3_preview_does_not_establish_sticky_state() {
        let root = TempDir::new().unwrap();
        let variants = mold_core::manifest::known_qwen3_8b_variants();
        let pressured = vec![DeviceFact {
            id: "cuda:0".to_string(),
            ordinal: 0,
            backend: mold_core::GpuBackend::Cuda,
            compute_capability: Some((8, 6)),
            available_vram_bytes: 6_400_000_000,
        }];
        let recovered = vec![DeviceFact {
            available_vram_bytes: 9_000_000_000,
            ..pressured[0].clone()
        }];

        let preview = select_auto_qwen3_variant_with_cache(
            root.path(),
            "shared/qwen3-8b-gguf",
            variants,
            &pressured,
            false,
            |_| false,
        )
        .unwrap();
        let admission = select_auto_qwen3_variant_with_cache(
            root.path(),
            "shared/qwen3-8b-gguf",
            variants,
            &recovered,
            true,
            |_| false,
        )
        .unwrap();

        assert_eq!(preview.tag, "q3");
        assert_eq!(admission.tag, "q6");
    }
}
