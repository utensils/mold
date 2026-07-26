//! Pre-admission materialization of auto-selected encoder dependencies.
//!
//! This module is deliberately CUDA-free. It consumes the background resource
//! snapshot, selects one concrete encoder variant per schedulable device, and
//! downloads missing quantized files on Tokio's blocking pool. The scheduler
//! does not mark a generation Ready until this returns.

use crate::execution_plan::{DeviceFact, PreparedDeviceExecutionInputs, PreparedExecutionInputs};
use crate::scheduler::worker_device_id;
use crate::state::{AppState, SseMessage};
use mold_core::{Config, GenerateRequest, ModelPaths, SseProgressEvent};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const ENCODER_HEADROOM: u64 = 2_000_000_000;
const T5_FP16_THRESHOLD: u64 = 16_000_000_000;
const QWEN3_4B_FP16_THRESHOLD: u64 = 10_200_000_000;
const QWEN2_FP16_THRESHOLD: u64 = 16_000_000_000;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct DownloadKey {
    repo: String,
    filename: String,
    subdir: String,
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

fn downloads() -> &'static Mutex<HashMap<DownloadKey, Arc<SharedDownload>>> {
    DOWNLOADS.get_or_init(|| Mutex::new(HashMap::new()))
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

async fn ensure_downloaded(
    state: Option<&AppState>,
    work_id: &str,
    repo: &str,
    filename: &str,
    subdir: &str,
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<PathBuf, String> {
    if let Some(path) = mold_core::download::cached_file_path(repo, filename, Some(subdir)) {
        return Ok(path);
    }

    let key = DownloadKey {
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
                mold_core::download::download_single_file_sync_with_progress(
                    &repo,
                    &filename,
                    Some(&subdir),
                    callback,
                )
                .map_err(|error| error.to_string())
            })
            .await
            .map_err(|error| format!("encoder dependency task failed: {error}"))
            .and_then(|result| result);
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
            return result;
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
    let registry =
        state
            .device_registry
            .snapshot(&state.gpu_pool, resources.as_ref(), &state.job_registry);
    registry
        .devices
        .into_iter()
        .filter(|device| device.schedulable)
        .filter_map(|device| {
            let ordinal = device.ordinal?;
            let total = device.memory.total_bytes?;
            let available = device
                .memory
                .used_bytes
                .map(|used| total.saturating_sub(used))
                .or_else(|| {
                    state
                        .gpu_pool
                        .workers
                        .iter()
                        .find(|worker| worker.gpu.ordinal == ordinal)
                        .map(|worker| worker.gpu.free_vram_bytes)
                })
                .unwrap_or(0);
            Some(DeviceFact {
                id: device.id,
                ordinal,
                available_vram_bytes: available,
            })
        })
        .collect()
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
            id: worker_device_id(worker),
            ordinal: worker.gpu.ordinal,
            available_vram_bytes: worker.gpu.free_vram_bytes,
        })
        .collect()
}

fn choose_largest_fitting<'a, T>(
    variants: &'a [T],
    free: u64,
    fields: impl Fn(&'a T) -> (&'static str, u64),
) -> Option<&'a T> {
    variants
        .iter()
        .find(|variant| fields(variant).1.saturating_add(ENCODER_HEADROOM) <= free)
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
                fields(variant).1.saturating_add(ENCODER_HEADROOM) <= device.available_vram_bytes
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

struct DependencyContext<'a> {
    state: Option<&'a AppState>,
    work_id: &'a str,
    progress: Option<&'a tokio::sync::mpsc::UnboundedSender<SseMessage>>,
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
            variant.hf_repo,
            variant.hf_filename,
            "shared/t5-gguf",
            context.progress,
        )
        .await?
    };
    paths.t5_encoder = Some(selected.clone());
    frozen.selected_t5_path = Some(selected);
    Ok(())
}

async fn materialize_qwen3(
    context: &DependencyContext<'_>,
    model: &str,
    family: &str,
    selection: VariantSelection<'_>,
    paths: &mut ModelPaths,
    frozen: &mut mold_inference::FrozenEngineConfig,
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
        vec![
            ensure_downloaded(
                context.state,
                context.work_id,
                variant.hf_repo,
                variant.hf_filename,
                subdir,
                context.progress,
            )
            .await?,
        ]
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
        variant.hf_repo,
        variant.hf_filename,
        "shared/qwen2-vl-gguf",
        context.progress,
    )
    .await?;
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
    let gguf = sorted_matching_files(&root, |name| name.ends_with(".gguf"));
    let tag = match preference.map(|value| value.trim().to_ascii_lowercase()) {
        Some(value) if matches!(value.as_str(), "q4" | "gguf" | "q4_gguf") => "q4",
        Some(value) if matches!(value.as_str(), "bf16" | "safetensors" | "bf16_safetensors") => {
            "bf16"
        }
        Some(value) if (value.is_empty() || value == "auto") && !bf16.is_empty() => "bf16",
        Some(value) if value.is_empty() || value == "auto" => "q4",
        Some(value) => return Err(format!("unknown LTX-2 Gemma variant '{value}'")),
        None if !bf16.is_empty() => "bf16",
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

async fn prepare_inputs_for_devices(
    state: Option<&AppState>,
    work_id: &str,
    request: &GenerateRequest,
    config: &Config,
    devices: Vec<DeviceFact>,
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<PreparedExecutionInputs, String> {
    let paths = ModelPaths::resolve(&request.model, config)
        .ok_or_else(|| format!("model '{}' has no concrete local artifacts", request.model))?;
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
    let shared_t5_tag = shared_quantized_fallback(
        mold_core::manifest::known_t5_variants(),
        &devices,
        |variant| (variant.tag, variant.size_bytes),
    )
    .map(|variant| variant.tag.to_string());
    let qwen3_8b = flux2_uses_qwen3_8b(&request.model, &paths);
    let qwen3_variants = if qwen3_8b {
        mold_core::manifest::known_qwen3_8b_variants()
    } else {
        mold_core::manifest::known_qwen3_variants()
    };
    let shared_qwen3_tag = shared_quantized_fallback(qwen3_variants, &devices, |variant| {
        (variant.tag, variant.size_bytes)
    })
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
        "z-image" | "flux2" | "flux.2" | "flux2-klein" => base
            .qwen3_variant
            .as_deref()
            .is_none_or(|value| value.is_empty() || value == "auto"),
        "qwen-image" | "qwen_image" | "qwen-image-edit" => base
            .qwen2_variant
            .as_deref()
            .is_none_or(|value| value.is_empty() || value == "auto"),
        _ => false,
    };

    let mut by_device = BTreeMap::new();
    let mut failures = BTreeMap::new();
    let dependency_context = DependencyContext {
        state,
        work_id,
        progress,
    };
    for device in devices {
        let mut selected_paths = paths.clone();
        let mut frozen = base.clone();
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
                )
                .await
            }
            "z-image" | "flux2" | "flux.2" | "flux2-klein" => {
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
                )
                .await
            }
            "ltx2" | "ltx-2" | "ltx2.3" => materialize_gemma(
                base.ltx2_gemma_variant.as_deref(),
                &selected_paths,
                &mut frozen,
            ),
            _ => Ok(()),
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
    Ok(PreparedExecutionInputs {
        authority_fingerprint,
        by_device,
        retryable_device_failures: failures,
    })
}

pub async fn prepare_execution_inputs(
    state: &AppState,
    work_id: &str,
    request: &GenerateRequest,
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<PreparedExecutionInputs, String> {
    let config = state.config.read().await.clone();
    let mut devices = resource_device_facts(state);
    if devices.is_empty() {
        devices = worker_device_facts_from_startup_sample(state);
    }
    prepare_inputs_for_devices(Some(state), work_id, request, &config, devices, progress).await
}

/// Forced-local counterpart to server preparation. It shares variant
/// selection and download de-duplication but has no job-registry cancellation
/// source or SSE transport.
pub async fn prepare_local_execution_inputs(
    config: &Config,
    request: &GenerateRequest,
    devices: Vec<DeviceFact>,
) -> Result<PreparedExecutionInputs, String> {
    prepare_inputs_for_devices(None, "local", request, config, devices, None).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{DevicePlacement, DeviceRef, ModelConfig};
    use tempfile::TempDir;

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

    #[test]
    fn flux2_size_selection_does_not_assume_device_count() {
        let mut paths = ModelPaths {
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
                    available_vram_bytes: 4_000_000_000,
                },
                DeviceFact {
                    id: "cuda:1".to_string(),
                    ordinal: 1,
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
            Err(crate::execution_plan::ExecutionPlanError::InsufficientVram)
        ));
        let recovered = crate::execution_plan::resolve_execution_plans_with_prepared(
            &config,
            &request,
            &(0..8)
                .map(|ordinal| DeviceFact {
                    id: format!("cuda:{ordinal}"),
                    ordinal,
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
}
