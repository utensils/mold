use std::sync::Arc;

use mold_core::{
    build_model_catalog, GenerateRequest, ModelDefaults, ModelInfo, ModelInfoExtended, ModelPaths,
};
use mold_inference::device::{activation_bytes, activation_family_for, ActivationFamily};

use crate::model_cache::ModelResidency;
use crate::{routes::ApiError, state::AppState};

pub(crate) type EngineProgressCallback = Arc<dyn Fn(mold_inference::ProgressEvent) + Send + Sync>;

/// Per-request shape hint passed into [`preflight_memory_guard`] so the
/// activation budget can scale with resolution / dtype / arch. `None`
/// degrades to the previous fixed-headroom approximation (the
/// `MEMORY_BUDGET_HEADROOM` baked into `estimate_peak_memory`'s 2 GB
/// constant), which keeps behavior identical for callers that don't yet
/// have a request in scope (e.g. admin-API model loads with no resolution
/// context).
///
/// Public because `gpu_worker::ensure_model_ready_sync` and
/// `gpu_worker::run_chain_blocking` (both `pub`) take it as a parameter.
#[derive(Debug, Clone, Copy)]
pub struct ActivationHint {
    /// Image-space width.
    pub width: u32,
    /// Image-space height.
    pub height: u32,
    /// CFG-doubled forwards typically pass `2`; non-CFG passes `1`.
    pub batch: u32,
    /// Bytes per element (`2` for bf16/fp16, `4` for f32).
    pub dtype_bytes: u32,
    /// Architecture family — drives the per-arch factor in
    /// `mold_inference::device::activation_bytes`.
    pub family: ActivationFamily,
}

impl ActivationHint {
    /// Build a hint from a [`GenerateRequest`] and the manifest family slug
    /// (e.g. `"flux"`, `"sdxl"`). The family slug is what
    /// [`activation_family_for`] expects — when the caller doesn't have a
    /// strong family signal (catalog ID without an installed manifest, etc.)
    /// passing the empty string falls back to `ActivationFamily::FluxDit`.
    pub fn from_request(req: &GenerateRequest, family_slug: &str) -> Self {
        // CFG-doubled forwards: SDXL/SD3 batch=2 when guidance ≈/> 1.0; FLUX,
        // Z-Image, Flux.2 are guidance-distilled and run a single forward.
        let family = activation_family_for(family_slug);
        let batch = match family {
            ActivationFamily::SdxlUnet | ActivationFamily::Sd3Mmdit if req.guidance > 1.0 => 2,
            _ => 1,
        };
        Self {
            width: req.width,
            height: req.height,
            batch,
            // Server-side preflight assumes bf16/fp16 activations — every
            // diffusion family in this repo runs in bf16/fp16 on GPU.
            dtype_bytes: 2,
            family,
        }
    }

    /// Compute the activation budget bytes from this hint.
    pub fn budget_bytes(&self) -> u64 {
        activation_bytes(
            self.width,
            self.height,
            self.batch,
            self.dtype_bytes,
            self.family,
        )
    }
}

// ── MPS memory guard ────────────────────────────────────────────────────────

/// Pure logic for the server memory guard, factored out for testing.
///
/// Hard-fails if peak > 90% of available (model won't fit even with page reclamation).
/// Warns if peak > 80% of available (tight but feasible).
///
/// `suggestion` is appended to the rejection message so call sites can surface
/// arch-specific remediation (e.g. reduce `--frames` / `--width` for LTX-Video).
fn check_model_memory_budget(
    model_name: &str,
    peak_bytes: u64,
    available_bytes: u64,
    suggestion: &str,
) -> Result<(), ApiError> {
    let hard_limit = available_bytes * 9 / 10; // 90%
    if peak_bytes > hard_limit {
        return Err(ApiError::insufficient_memory(format!(
            "model '{}' estimated peak ~{:.1} GB exceeds the per-load budget cap ~{:.1} GB \
             (90% of {:.1} GB free, with 2 GB activation headroom built into peak estimate; \
             encoders are dropped before denoise). {}",
            model_name,
            peak_bytes as f64 / 1_000_000_000.0,
            hard_limit as f64 / 1_000_000_000.0,
            available_bytes as f64 / 1_000_000_000.0,
            suggestion,
        )));
    }

    let warn_limit = available_bytes * 8 / 10; // 80%
    if peak_bytes > warn_limit {
        tracing::warn!(
            model = %model_name,
            peak_gb = format_args!("{:.1}", peak_bytes as f64 / 1_000_000_000.0),
            available_gb = format_args!("{:.1}", available_bytes as f64 / 1_000_000_000.0),
            "model is close to memory limit — may trigger page reclamation"
        );
    }

    Ok(())
}

/// Build the suggestion text appended to preflight rejection messages.
/// For LTX-Video (non-streaming full-weight load) the dominant knob is
/// reducing `frames` or `width`/`height`; for other families a quantized
/// variant or `--offload` (FLUX) is the typical escape.
fn rejection_suggestion(hint: Option<ActivationHint>) -> &'static str {
    match hint.map(|h| h.family) {
        Some(ActivationFamily::LtxVideo) => {
            "Try reducing --frames or --width/--height, use a quantized variant \
             (e.g. ':q8'), or close other GPU apps."
        }
        _ => {
            "Try a smaller variant (e.g. ':q8' / ':q5'), enable --offload (FLUX), \
             or close other GPU apps."
        }
    }
}

const FLUX2_KLEIN_9B_BF16_MIN_VRAM: u64 = 32_000_000_000;
const FLUX2_KLEIN_9B_TRANSFORMER_FLOOR: u64 = 16_000_000_000;
const FLUX2_KLEIN_9B_TEXT_ENCODER_FLOOR: u64 = 14_000_000_000;

fn transformer_file_size(paths: &ModelPaths) -> u64 {
    let file_size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    if !paths.transformer_shards.is_empty() {
        paths.transformer_shards.iter().map(|p| file_size(p)).sum()
    } else {
        file_size(&paths.transformer)
    }
}

fn is_gguf_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

fn is_flux2_klein_9b_bf16_load(
    model_name: &str,
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
) -> bool {
    let lower = model_name.to_lowercase();
    let family_matches = hint
        .map(|h| h.family == ActivationFamily::Flux2Dit)
        .unwrap_or_else(|| lower.contains("flux2"));
    if !family_matches || is_gguf_path(&paths.transformer) {
        return false;
    }

    let name_matches_9b = lower.contains("klein-9b") || lower.contains("9b");
    let name_marks_full_precision = lower.contains(":bf16") || lower.contains(":fp16");
    let transformer_size = transformer_file_size(paths);
    let text_encoder_size: u64 = paths
        .text_encoder_files
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let size_matches_9b = transformer_size >= FLUX2_KLEIN_9B_TRANSFORMER_FLOOR
        && text_encoder_size >= FLUX2_KLEIN_9B_TEXT_ENCODER_FLOOR;

    (name_matches_9b && name_marks_full_precision) || size_matches_9b
}

/// Pure inner: given an `available_bytes` budget and the active model's
/// reclaimable VRAM, decide whether the new model fits. Adding
/// `active_vram_bytes` to `available_bytes` accounts for the currently-loaded
/// model that will be unloaded before the new one loads — without this, a
/// swap of two near-equal-size models would be falsely rejected even though
/// the swap is feasible.
///
/// Peak is estimated under `LoadStrategy::Sequential` because every diffusion
/// family in this repo (FLUX, SD3, Z-Image, Flux.2, Qwen-Image, LTX) drops
/// text encoders from GPU after encoding before the transformer denoises.
/// The Eager sum (`transformer + vae + all_encoders`) overcounts by the
/// encoder weight on every load — enough to false-reject a quantized FLUX on
/// a 24 GB card even when the swap would actually fit.
///
/// `hint` adds a resolution-scaled activation budget on top of the
/// component-size peak so a 2048² generation isn't under-budgeted. When
/// `None` the inner peak retains the existing 2 GB
/// `MEMORY_BUDGET_HEADROOM` constant from `estimate_peak_memory` and no
/// extra is added — equivalent to the pre-Tier-2.3 behavior.
pub(crate) fn preflight_memory_guard_with_available(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    // Streaming-transformer families (LTX-Video / LTX-2) load only a couple
    // of transformer blocks onto GPU at a time via `new_streaming` — the
    // file-size-based estimate (which assumes the whole transformer becomes
    // GPU-resident) over-counts by ~40+ GB for the 22B LTX-2 preset and
    // false-rejects on 24 GB cards. When the hint marks the family as
    // streaming, we replace the file-size transformer component with a
    // generous fixed cap that covers `streaming_prefetch_count` blocks
    // plus the always-resident top-level weights (proj_in / proj_out /
    // time_embed / caption_projection / scale_shift_table / norms).
    let streaming = hint.is_some_and(|h| h.family.streaming_transformer());
    let peak = if streaming {
        // LTX-2 also pays for a Gemma 3 12B prompt encoder. When the resolver
        // determines the encoder will run on CPU (`MOLD_LTX2_GEMMA_DEVICE=cpu`
        // or auto-placement on a card without enough free VRAM) the encoder
        // phase doesn't compete with the transformer phase for GPU at all —
        // collapse encoder_total out of the peak so the preflight admits the
        // load. This must stay in lockstep with `resolve_prompt_encoder_device`
        // in `mold-inference::ltx2::pipeline` or the preflight will admit
        // loads that subsequently OOM (or reject loads that would have fit).
        let gemma_will_be_cpu = ltx2_encoder_phase_will_be_cpu(0);
        streaming_transformer_peak(paths, gemma_will_be_cpu)
    } else {
        mold_inference::device::estimate_peak_memory(
            paths,
            mold_inference::LoadStrategy::Sequential,
        )
    };
    // Add the per-request activation budget on top of the file-size peak.
    // The 2 GB `MEMORY_BUDGET_HEADROOM` already inside `estimate_peak_memory`
    // is a generic "kernels + small state" constant that doesn't scale; the
    // hint is the resolution/dtype/arch-aware delta on top.
    let activation = hint.map(|h| h.budget_bytes()).unwrap_or(0);
    let peak_with_activation = peak.saturating_add(activation);
    let effective_available = available_bytes.saturating_add(active_vram_bytes);
    let suggestion = rejection_suggestion(hint);

    if is_flux2_klein_9b_bf16_load(model_name, paths, hint)
        && effective_available < FLUX2_KLEIN_9B_BF16_MIN_VRAM
    {
        return Err(ApiError::insufficient_memory(format!(
            "model '{}' requires a 32 GB GPU for the Flux.2 Klein-9B BF16 transformer \
             ({:.1} GB effective budget detected). Use a quantized Klein-9B variant \
             (e.g. ':q8' / ':q4') or run on a larger card.",
            model_name,
            effective_available as f64 / 1_000_000_000.0,
        )));
    }

    check_model_memory_budget(
        model_name,
        peak_with_activation,
        effective_available,
        suggestion,
    )
}

/// Peak GPU residency for streaming-transformer families. Mirrors the
/// Sequential strategy in `device::estimate_peak_memory` but replaces the
/// `transformer_size + vae_size` term with a `STREAMING_TRANSFORMER_CAP`
/// that bounds "block-streaming overhead, fully-resident top-level weights,
/// and VAE."
///
/// When `gemma_on_cpu` is true, encoder_total is dropped from the max because
/// the prompt encoder won't compete for VRAM at all — it lives in system RAM
/// and pipes its conditioning across to the transformer GPU at encode time.
/// When false, the encoder phase still pays full encoder_total (text encoders
/// load whole; the runtime drops them before denoise but during the encode
/// phase they're co-resident with allocations made earlier in the request).
///
/// The cap is conservative: at 22B BF16 with `streaming_prefetch_count=2`,
/// two blocks ≈ 1.83 GB + non-block fragments ≈ 200 MB + VAE ≈ 200 MB
/// ≈ 2.3 GB. The 6 GB cap leaves room for activation workspace, OS
/// fragmentation, and future LTX presets without revisiting this file.
fn streaming_transformer_peak(paths: &ModelPaths, gemma_on_cpu: bool) -> u64 {
    const STREAMING_TRANSFORMER_CAP: u64 = 6_000_000_000; // 6 GB
    const HEADROOM: u64 = 2_000_000_000; // 2 GB, mirrors device::MEMORY_BUDGET_HEADROOM

    let file_size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let t5_size = paths.t5_encoder.as_ref().map(|p| file_size(p)).unwrap_or(0);
    let clip_size = paths
        .clip_encoder
        .as_ref()
        .map(|p| file_size(p))
        .unwrap_or(0);
    let clip2_size = paths
        .clip_encoder_2
        .as_ref()
        .map(|p| file_size(p))
        .unwrap_or(0);
    let text_encoder_size: u64 = paths.text_encoder_files.iter().map(|p| file_size(p)).sum();
    let encoder_total = if gemma_on_cpu {
        0
    } else {
        t5_size + clip_size + clip2_size + text_encoder_size
    };

    let inference_phase = STREAMING_TRANSFORMER_CAP;
    std::cmp::max(encoder_total, inference_phase) + HEADROOM
}

/// Resolve, for the preflight, whether the LTX-2 Gemma 3 12B prompt encoder
/// is going to land on CPU. This is a thin wrapper around the inference-side
/// `resolve_ltx2_gemma_placement` so the preflight and the load path reach
/// the same conclusion under the same env state and the same observation of
/// free VRAM.
fn ltx2_encoder_phase_will_be_cpu(gpu_ordinal: usize) -> bool {
    matches!(
        mold_inference::device::resolve_ltx2_gemma_placement(gpu_ordinal),
        mold_inference::device::LtxGemmaPlacement::Cpu,
    )
}

/// Check whether estimated peak memory fits before committing to a model load.
///
/// Budgeting strategy on CUDA:
/// - **No active model on this GPU** — the new load lands in whatever is
///   currently free, so use `free_vram_bytes(gpu_ordinal)`.
/// - **Active model present** — the call site unloads it and runs
///   `cuDevicePrimaryCtxReset_v2`, which releases *every* allocation on the
///   device (transformer, leftover activation buffers, fragmentation in the
///   caching pool). The realistic post-reclaim budget is total VRAM, not
///   `free + recorded active_vram`. Using the latter under-counts whatever
///   the cache forgot to track (notably the encoder churn during the
///   previous generation) and produces false rejections.
///
/// On macOS (unified memory) we keep the additive `available + active_vram`
/// budget because Metal has no equivalent device-wide context reset; tensors
/// freed during `unload()` simply return to the system page cache.
/// On other platforms with no memory query available, the guard is a no-op.
pub(crate) fn preflight_memory_guard(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    // CUDA branch: when an active model will be reclaimed via primary-context
    // reset, the post-reclaim budget is the device total, not free+active.
    #[cfg(feature = "cuda")]
    {
        if active_vram_bytes > 0 {
            if let Some(total) = mold_inference::device::total_vram_bytes(gpu_ordinal) {
                return preflight_memory_guard_with_available(model_name, paths, 0, total, hint);
            }
        }
        // Ghost-VRAM case: no active model in our cache, but the device
        // reports `free` significantly below `total` because cuBLAS / cuDNN /
        // kernel modules from a previous load are still squatting on
        // workspace allocations. Reclaim the primary context — we have
        // nothing live to lose — and re-query before deciding. After reclaim,
        // re-query through `usable_free_vram_bytes` so the OS reserve
        // (T2-B) is respected on the post-reclaim reading too.
        if let (Some(free), Some(total)) = (
            mold_inference::device::free_vram_bytes(gpu_ordinal),
            mold_inference::device::total_vram_bytes(gpu_ordinal),
        ) {
            const GHOST_VRAM_THRESHOLD: u64 = 1_500_000_000; // 1.5 GB
            if total.saturating_sub(free) > GHOST_VRAM_THRESHOLD {
                tracing::info!(
                    gpu = gpu_ordinal,
                    free_gb = format_args!("{:.1}", free as f64 / 1e9),
                    total_gb = format_args!("{:.1}", total as f64 / 1e9),
                    "no active model on this GPU but VRAM is held — reclaiming primary context",
                );
                mold_inference::device::reclaim_gpu_memory(gpu_ordinal);
            }
            let effective_free = mold_inference::device::usable_free_vram_bytes(gpu_ordinal)
                .unwrap_or_else(|| {
                    free.saturating_sub(mold_inference::device::reserved_vram_bytes())
                });
            return preflight_memory_guard_with_available(
                model_name,
                paths,
                active_vram_bytes,
                effective_free,
                hint,
            );
        }
        // Fallback if total_vram is unavailable: still go through the
        // reserve-adjusted reading.
        if let Some(free) = mold_inference::device::usable_free_vram_bytes(gpu_ordinal) {
            return preflight_memory_guard_with_available(
                model_name,
                paths,
                active_vram_bytes,
                free,
                hint,
            );
        }
    }

    // macOS unified memory: query system memory and add reclaimable footprint.
    if let Some(available) = mold_inference::device::available_system_memory_bytes() {
        if available > 0 {
            return preflight_memory_guard_with_available(
                model_name,
                paths,
                active_vram_bytes,
                available,
                hint,
            );
        }
    }

    // No memory info available on this platform — skip the guard.
    Ok(())
}

/// Effective memory budget to use when deciding whether a server engine can
/// stay eager-loaded or should degrade to load-use-drop sequential mode.
///
/// This mirrors the budget shape in [`preflight_memory_guard`]: CUDA swaps can
/// reclaim the whole primary context when an active model exists, while Metal
/// uses unified system memory and adds the active footprint as reclaimable.
pub(crate) fn effective_load_available_bytes(
    active_vram_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
) -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        if active_vram_bytes > 0 {
            if let Some(total) = mold_inference::device::total_vram_bytes(gpu_ordinal) {
                return Some(total);
            }
        }
        if let Some(free) = mold_inference::device::usable_free_vram_bytes(gpu_ordinal) {
            return Some(free);
        }
    }

    mold_inference::device::available_system_memory_bytes()
        .filter(|available| *available > 0)
        .map(|available| available.saturating_add(active_vram_bytes))
}

/// Choose the server load strategy for the current memory budget.
///
/// The server normally prefers eager engines so the active model stays hot.
/// When eager residency would exceed the same 90% cap used by preflight but
/// the model fits under sequential load-use-drop, degrade to Sequential. This
/// keeps preflight and the actual load path consistent: a model admitted only
/// because text encoders can be dropped should not then OOM during eager
/// startup before it gets a chance to generate.
pub(crate) fn select_server_load_strategy_for_budget(
    paths: &ModelPaths,
    available_bytes: Option<u64>,
    hint: Option<ActivationHint>,
) -> mold_inference::LoadStrategy {
    let Some(available_bytes) = available_bytes.filter(|v| *v > 0) else {
        return mold_inference::LoadStrategy::Eager;
    };

    let activation = hint.map(|h| h.budget_bytes()).unwrap_or(0);
    let eager_peak =
        mold_inference::device::estimate_peak_memory(paths, mold_inference::LoadStrategy::Eager)
            .saturating_add(activation);
    let sequential_peak = mold_inference::device::estimate_peak_memory(
        paths,
        mold_inference::LoadStrategy::Sequential,
    )
    .saturating_add(activation);
    let hard_limit = available_bytes.saturating_mul(9) / 10;

    if eager_peak > hard_limit && sequential_peak <= hard_limit {
        mold_inference::LoadStrategy::Sequential
    } else {
        mold_inference::LoadStrategy::Eager
    }
}
pub(crate) type DownloadProgressCallback =
    Arc<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>;

pub(crate) enum PullStatus {
    AlreadyAvailable,
    Pulled,
}

pub(crate) async fn refresh_config(state: &AppState) -> mold_core::Config {
    let fresh = {
        let current = state.config.read().await;
        current.reload_from_disk_preserving_runtime()
    };

    let mut config = state.config.write().await;
    *config = fresh.clone();
    fresh
}

pub(crate) async fn list_models(state: &AppState) -> Vec<ModelInfoExtended> {
    let config = refresh_config(state).await;
    let models_dir = config.resolved_models_dir();

    // Multi-GPU mode: derive "loaded" state from the worker pool so /api/models
    // reflects the actual engine cache, not the legacy single-GPU snapshot.
    if state.gpu_pool.worker_count() > 0 {
        let loaded_models = loaded_models_across_pool(state);
        let primary = loaded_models.first().cloned();
        let mut catalog = build_model_catalog(&config, primary.as_deref(), primary.is_some());
        // Mark every GPU-resident model as loaded (not just the primary).
        for entry in catalog.iter_mut() {
            if loaded_models.contains(&entry.info.name) {
                entry.info.is_loaded = true;
            }
        }
        catalog.extend(installed_catalog_models(
            state,
            &config,
            &models_dir,
            primary.as_deref(),
            primary.is_some(),
        ));
        return catalog;
    }

    let snapshot = state.model_cache.lock().await.snapshot();
    let mut catalog =
        build_model_catalog(&config, snapshot.model_name.as_deref(), snapshot.is_loaded);
    catalog.extend(installed_catalog_models(
        state,
        &config,
        &models_dir,
        snapshot.model_name.as_deref(),
        snapshot.is_loaded,
    ));
    catalog
}

fn loaded_models_across_pool(state: &AppState) -> Vec<String> {
    let mut names = Vec::new();
    for worker in &state.gpu_pool.workers {
        // Prefer the active-generation model (cache entry is taken out during
        // inflight generation), else whatever is GPU-resident.
        let active = worker
            .active_generation
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|g| g.model.clone()));
        let loaded = active.or_else(|| {
            let cache = worker.model_cache.lock().ok()?;
            cache.active_model().map(|s| s.to_string())
        });
        if let Some(name) = loaded {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

// ── Catalog bridge ───────────────────────────────────────────────────────────
//
// Mirrors the logic in `mold-cli/src/catalog_bridge.rs` so the server can
// resolve `cv:*` model IDs (Civitai single-file checkpoints downloaded via
// the catalog web UI) without the binary crate as an intermediary.

fn looks_like_catalog_id(id: &str) -> bool {
    id.starts_with("cv:") || id.starts_with("hf:")
}

/// Best-effort family lookup for a catalog (`cv:*` / `hf:*`) model name,
/// used to feed `validate_generate_request_with_family` so catalog IDs
/// get the same family-gated feature checks as manifest-resident models.
///
/// Reads from the intent cache first (populated by `install_catalog_model`
/// before the engine-load resolution runs), then falls back to
/// `state.config.models` for back-compat with callers that snapshotted
/// the resolved `ModelConfig`.
pub(crate) async fn catalog_family_for(state: &AppState, model_name: &str) -> Option<String> {
    if !looks_like_catalog_id(model_name) {
        return None;
    }
    {
        let intents = state.catalog_intents.read().await;
        if let Some(intent) = intents.get(model_name) {
            return Some(intent.family.clone());
        }
    }
    let config = state.config.read().await;
    config.models.get(model_name).and_then(|m| m.family.clone())
}

/// Resolve the family slug for any model name — checks the static manifest
/// first (covers `flux-dev:q8` and friends), then falls back to the catalog
/// config entry (covers `cv:*` / `hf:*`). Used by the activation-budget
/// preflight to dispatch to the right [`ActivationFamily`].
pub(crate) async fn family_for_model(state: &AppState, model_name: &str) -> Option<String> {
    if let Some(manifest) = mold_core::manifest::find_manifest(model_name) {
        return Some(manifest.family.clone());
    }
    catalog_family_for(state, model_name).await
}

/// Sync variant of [`family_for_model`] for the GPU-worker hot path which
/// runs inside `spawn_blocking` and only has a `&Config` snapshot to work
/// with. Falls through to manifest-then-config in the same order.
pub(crate) fn family_for_model_sync(
    model_name: &str,
    config: &mold_core::Config,
) -> Option<String> {
    if let Some(manifest) = mold_core::manifest::find_manifest(model_name) {
        return Some(manifest.family.clone());
    }
    config.models.get(model_name).and_then(|m| m.family.clone())
}

/// Sync variant of [`activation_hint_for_request`] for the GPU-worker hot
/// path. Returns `None` when the family slug can't be resolved.
pub(crate) fn activation_hint_for_request_sync(
    config: &mold_core::Config,
    req: &GenerateRequest,
) -> Option<ActivationHint> {
    let family = family_for_model_sync(&req.model, config)?;
    Some(ActivationHint::from_request(req, &family))
}

/// Build an [`ActivationHint`] for the given request using the resolved
/// model family (manifest or catalog). Returns `None` when the family slug
/// can't be resolved — the preflight then falls back to the size-only peak.
pub(crate) async fn activation_hint_for_request(
    state: &AppState,
    req: &GenerateRequest,
) -> Option<ActivationHint> {
    let family = family_for_model(state, &req.model).await?;
    Some(ActivationHint::from_request(req, &family))
}

fn copy_catalog_companion(cfg: &mut mold_core::ModelConfig, companion: &str, paths: &ModelPaths) {
    let to_str = |p: &std::path::PathBuf| p.to_str().map(str::to_owned);
    match companion {
        "clip-l" => {
            cfg.clip_encoder = to_str(&paths.transformer);
            cfg.clip_tokenizer = paths.clip_tokenizer.as_ref().and_then(to_str);
        }
        "clip-g" => {
            cfg.clip_encoder_2 = to_str(&paths.transformer);
            cfg.clip_tokenizer_2 = paths
                .clip_tokenizer
                .as_ref()
                .and_then(to_str)
                .or_else(|| cfg.clip_tokenizer.clone());
        }
        "sdxl-vae" | "sd-vae-ft-mse" => {}
        // Civitai FLUX fine-tune convention is split: some bundle the
        // VAE in the same .safetensors as the transformer, some are
        // transformer-only. `synthesize_catalog_config` peeks the
        // header and clears cfg.vae for the transformer-only case;
        // here we populate it from the companion's resolved path.
        // When the bundle DID include a VAE, cfg.vae is already set
        // to the primary checkpoint and we leave it alone — the
        // bundled VAE wins, so the guarded arm doesn't fire and the
        // catch-all (no-op) handles it.
        "flux-vae" if cfg.vae.is_none() => {
            cfg.vae = to_str(&paths.transformer);
        }
        "ltx-video-vae" | "flux2-vae" => {
            cfg.vae = to_str(&paths.transformer);
        }
        "t5-v1_1-xxl" => {
            cfg.t5_encoder = to_str(&paths.transformer);
            cfg.t5_tokenizer = paths.t5_tokenizer.as_ref().and_then(to_str);
        }
        "z-image-te" | "flux2-te" | "flux2-te-9b" => {
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_str)
                .collect::<Vec<_>>()
                .into();
            cfg.text_tokenizer = paths.text_tokenizer.as_ref().and_then(to_str);
        }
        "ltx2-te" => {
            // Gemma 3 12B for LTX-2. The runtime calls `gemma_root` which
            // takes the parent directory of `text_encoder_files[0]`, so
            // populating that vec is sufficient — we don't need a separate
            // `text_tokenizer` field (the manifest tags every Gemma file
            // including `tokenizer.json` / `tokenizer.model` as
            // ModelComponent::TextEncoder, so the tokenizer rides along).
            cfg.text_encoder_files = paths
                .text_encoder_files
                .iter()
                .filter_map(to_str)
                .collect::<Vec<_>>()
                .into();
        }
        _ => {}
    }
}

/// Errors from the disk-aware resolution stage that turns a pure
/// `CatalogModelIntent` into a runtime `ModelConfig`.
///
/// Intent synthesis is pure; everything that depends on actual disk state
/// — the FLUX VAE-bundle probe, companion path lookup, file existence —
/// runs here and surfaces precise errors instead of silent partial state.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ResolveError {
    /// A required companion's manifest entry isn't in the user's
    /// `Config.models` map. This is a real config bug — the user hasn't
    /// installed (or has a broken manifest entry for) a base model the
    /// catalog entry depends on.
    #[error(
        "catalog model '{model}' is missing required component '{companion}'.\n\
         The manifest entry for '{companion}' is not present in your config — \
         install the matching base model first."
    )]
    CompanionConfigMissing {
        model: String,
        companion: &'static str,
    },

    /// A required companion's file isn't on disk yet. Transient — caller
    /// hasn't finished `mold pull`-ing this entry. Tells the user
    /// specifically which component is missing rather than the generic
    /// "missing required components" pre-fix message.
    #[error(
        "catalog model '{model}' is missing required component '{companion}'.\n\
         The companion file isn't on disk yet. Run `mold pull {model}` to fetch \
         the primary checkpoint AND its companion files."
    )]
    CompanionFileMissing {
        model: String,
        companion: &'static str,
    },

    /// The FLUX bundled-VAE probe couldn't be run (e.g. malformed
    /// safetensors header). Surfaces the underlying inference-loader
    /// error so the user can act.
    #[error("probe of FLUX checkpoint {path} for bundled VAE failed: {source}")]
    FluxProbeFailed {
        path: String,
        #[source]
        source: anyhow::Error,
    },
}

/// All companions emitted by `companions_for` are statically known
/// names from the registry. We pin the resolution to the same
/// `&'static str` so error messages can carry it without an allocation
/// and the match in `copy_catalog_companion` never sees a typo.
fn known_companion_static(name: &str) -> Option<&'static str> {
    // The set is closed: any name in `companions_for` outputs maps here.
    // Adding a new companion to the registry without a match arm here is
    // a compile-time invitation to update both sides.
    Some(match name {
        "clip-l" => "clip-l",
        "clip-g" => "clip-g",
        "sdxl-vae" => "sdxl-vae",
        "sd-vae-ft-mse" => "sd-vae-ft-mse",
        "flux-vae" => "flux-vae",
        "flux2-te" => "flux2-te",
        "flux2-te-9b" => "flux2-te-9b",
        "flux2-vae" => "flux2-vae",
        "z-image-te" => "z-image-te",
        "ltx-video-vae" => "ltx-video-vae",
        "ltx2-te" => "ltx2-te",
        "t5-v1_1-xxl" => "t5-v1_1-xxl",
        _ => return None,
    })
}

/// Disk-aware resolution: turn a pure `CatalogModelIntent` into a
/// `ModelConfig` ready for engine load.
///
/// Runs the FLUX bundled-VAE probe (file must be on disk) and walks the
/// required-companion list, populating the `ModelConfig` fields each
/// companion contributes. Returns a [`ResolveError`] when a required
/// companion can't be resolved — never silently returns a partial config.
pub(crate) fn resolve_intent_to_paths(
    model_name: &str,
    intent: &mold_catalog::synthesis::CatalogModelIntent,
    config: &mold_core::Config,
) -> Result<mold_core::ModelConfig, ResolveError> {
    let primary_str = intent
        .primary_recipe_path
        .to_str()
        .expect("synthesize_intent guarantees UTF-8 paths")
        .to_string();

    let mut cfg = mold_core::ModelConfig {
        family: Some(intent.family.clone()),
        ..Default::default()
    };
    cfg.transformer = Some(primary_str.clone());

    // FLUX is the only family with mixed bundling — peek the safetensors
    // header to decide whether the primary file carries the VAE or
    // whether the flux-vae companion has to populate cfg.vae.
    let is_flux = intent.family == "flux";
    let probe_says_bundled = if is_flux && intent.primary_recipe_path.exists() {
        Some(
            mold_inference::loader::flux_single_file_bundles_vae(&intent.primary_recipe_path)
                .map_err(|e| ResolveError::FluxProbeFailed {
                    path: intent.primary_recipe_path.display().to_string(),
                    source: e.into(),
                })?,
        )
    } else if is_flux {
        // Primary not on disk yet — defer. The flux-vae companion (which
        // is in the required list for FLUX) must populate cfg.vae below.
        None
    } else if mold_catalog::synthesis::family_bundles_vae_unconditionally(flux_family_for_slug(
        &intent.family,
    )) {
        // SDXL / SD1.5 always bundle the VAE.
        Some(true)
    } else {
        // Flux.2 / LTX-Video / LTX-2 always need a separate VAE companion.
        Some(false)
    };
    if probe_says_bundled == Some(true) {
        cfg.vae = Some(primary_str);
    }
    // probe_says_bundled == Some(false) ⇒ leave cfg.vae None for the
    // VAE-companion arm in copy_catalog_companion to fill. None case
    // (FLUX, file not yet on disk) is the same: the companion fills
    // cfg.vae if its own file is present, otherwise we surface
    // CompanionFileMissing below.

    let mut missing_required: Option<&'static str> = None;

    for companion in &intent.companions {
        let static_name =
            known_companion_static(&companion.name).expect("companion name in registry");
        match ModelPaths::resolve(static_name, config) {
            Some(paths) => copy_catalog_companion(&mut cfg, static_name, &paths),
            None => {
                if companion.required {
                    // Pin the first missing required companion; we keep
                    // looping so the entire missing set could be logged
                    // in future iterations, but the first one is what the
                    // user-facing error reports.
                    if missing_required.is_none() {
                        missing_required = Some(static_name);
                    }
                }
            }
        }
    }

    if let Some(missing) = missing_required {
        return Err(classify_missing_companion(model_name, missing, config));
    }

    Ok(cfg)
}

fn flux_family_for_slug(slug: &str) -> mold_catalog::families::Family {
    use mold_catalog::families::Family;
    match slug {
        "flux" => Family::Flux,
        "flux2" => Family::Flux2,
        "sd15" => Family::Sd15,
        "sdxl" => Family::Sdxl,
        "z-image" => Family::ZImage,
        "ltx-video" => Family::LtxVideo,
        "ltx2" => Family::Ltx2,
        "qwen-image" => Family::QwenImage,
        "wuerstchen" => Family::Wuerstchen,
        // Defensive — synthesize_intent only emits known slugs.
        _ => Family::Flux,
    }
}

/// Decide whether a missing companion is "config bug" vs "file not yet
/// downloaded". The catalog bridge writes companion entries into
/// `config.models` lazily — if they aren't there at all, that's a
/// CompanionConfigMissing. If the entry is there but the file the
/// manifest paths point at doesn't exist, that's CompanionFileMissing
/// (the user didn't `mold pull` yet).
fn classify_missing_companion(
    model_name: &str,
    companion: &'static str,
    config: &mold_core::Config,
) -> ResolveError {
    if !config.models.contains_key(companion) {
        ResolveError::CompanionConfigMissing {
            model: model_name.to_string(),
            companion,
        }
    } else {
        ResolveError::CompanionFileMissing {
            model: model_name.to_string(),
            companion,
        }
    }
}

/// Backwards-compatible wrapper: pure synthesis + disk-aware resolution
/// in one step. Used only by the in-tree tests that pre-date the
/// intent / lazy-resolve split — production paths call
/// `synthesize_intent` + `resolve_intent_to_paths` separately so the
/// intent can be cached and resolution re-runs as files land.
#[cfg(test)]
fn synthesize_catalog_config(
    entry: &mold_catalog::entry::CatalogEntry,
    models_dir: &std::path::Path,
    config: &mold_core::Config,
) -> anyhow::Result<mold_core::ModelConfig> {
    let intent = mold_catalog::synthesis::synthesize_intent(entry, models_dir)?;
    Ok(resolve_intent_to_paths(entry.id.as_str(), &intent, config)?)
}

/// If `model_name` is a catalog ID and not yet in the intent cache, hit
/// live HF/Civitai for the entry and synthesize its
/// [`CatalogModelIntent`]. Returns `Ok(())` when a fresh entry was
/// installed or one was already cached. Returns a typed [`InstallError`]
/// so the caller can map per-variant to a user-facing HTTP status.
///
/// This function is *pure synthesis* — no disk reads. The intent stays
/// valid across download events; resolution into a `ModelConfig` is run
/// lazily by [`resolve_intent_to_paths`] at engine-load time.
pub(crate) async fn install_catalog_model(
    state: &AppState,
    model_name: &str,
) -> Result<(), mold_core::InstallError> {
    if !looks_like_catalog_id(model_name) {
        // Caller should have shape-checked first; treat it as a no-op
        // success rather than fabricating a custom error variant.
        return Ok(());
    }

    // Already installed from a prior request — keep the cached intent.
    {
        let intents = state.catalog_intents.read().await;
        if intents.contains_key(model_name) {
            return Ok(());
        }
    }

    // Live single-id lookup. Tokens are picked up from env so
    // unauthenticated browsing still works.
    let civitai_base = state.catalog_live_civitai_base.as_str();
    let entry = if let Some(version_id) = model_name.strip_prefix("cv:") {
        mold_catalog::live::fetch_civitai_version(
            civitai_base,
            version_id,
            std::env::var("CIVITAI_TOKEN").ok().as_deref(),
        )
        .await
        .map_err(|e| live_error_to_install_error(model_name, &e))?
    } else if let Some(repo_id) = model_name.strip_prefix("hf:") {
        mold_catalog::live::fetch_hf_repo(
            "https://huggingface.co",
            repo_id,
            std::env::var("HF_TOKEN").ok().as_deref(),
        )
        .await
        .map_err(|e| live_error_to_install_error(model_name, &e))?
    } else {
        return Ok(());
    };

    let models_dir = state.config.read().await.resolved_models_dir();
    let intent = mold_catalog::synthesis::synthesize_intent(&entry, &models_dir).map_err(|e| {
        mold_core::InstallError::RecipeMalformed(format!("synthesize intent for {model_name}: {e}"))
    })?;

    let mut intents = state.catalog_intents.write().await;
    intents.insert(model_name.to_string(), intent);
    Ok(())
}

/// Translate a `LiveSearchError` into the user-facing `InstallError`
/// shape. The `Upstream` variant carries the HTTP status from the
/// upstream's response body — 404 is "not found", anything else is
/// "malformed recipe" (the live API answered, but the response wasn't
/// what mold expects).
fn live_error_to_install_error(
    model_name: &str,
    err: &mold_catalog::live::LiveSearchError,
) -> mold_core::InstallError {
    use mold_catalog::live::LiveSearchError;
    match err {
        LiveSearchError::Network(e) => {
            mold_core::InstallError::Network(format!("{model_name}: {e}"))
        }
        LiveSearchError::Decode(e) => mold_core::InstallError::RecipeMalformed(format!(
            "{model_name}: decode upstream payload: {e}"
        )),
        LiveSearchError::Upstream { status, body, .. } if *status == 404 => {
            mold_core::InstallError::NotFound(format!(
                "{model_name}: upstream returned 404 ({})",
                truncate_body(body)
            ))
        }
        LiveSearchError::Upstream { status, body, .. } => {
            mold_core::InstallError::RecipeMalformed(format!(
                "{model_name}: upstream HTTP {status}: {}",
                truncate_body(body)
            ))
        }
    }
}

fn truncate_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.len() > 160 {
        format!("{}…", &trimmed[..160])
    } else {
        trimmed.to_string()
    }
}

/// Translate `InstallError` into the user-facing `ApiError`. Each variant
/// maps to a distinct HTTP status so clients can tell "Civitai is down"
/// from "you typed a bad ID".
pub(crate) fn install_error_to_api_error(err: &mold_core::InstallError) -> ApiError {
    use mold_core::InstallError;
    match err {
        InstallError::Network(msg) => {
            // 502 Bad Gateway via internal_with_status — the catalog
            // upstream is unreachable, not a mold internal failure.
            ApiError::internal_with_status(
                format!("network unreachable: {msg}"),
                axum::http::StatusCode::BAD_GATEWAY,
            )
        }
        InstallError::NotFound(msg) => ApiError::not_found(msg.to_string()),
        InstallError::RecipeMalformed(msg) => ApiError::internal(msg.to_string()),
    }
}

/// Translate `ResolveError` into the user-facing `ApiError`. Both
/// variants surface as 404 (the model isn't loadable yet) but with
/// distinct, specific messages so the user can act.
pub(crate) fn resolve_error_to_api_error(err: &ResolveError) -> ApiError {
    ApiError::not_found(err.to_string())
}

/// Return `ModelInfoExtended` entries for every installed catalog
/// checkpoint discovered via per-install sidecars. Replaces the
/// bulk-scrape DB query that used to back this surface.
fn installed_catalog_models(
    _state: &AppState,
    config: &mold_core::Config,
    models_dir: &std::path::Path,
    loaded_model: Option<&str>,
    engine_is_loaded: bool,
) -> Vec<ModelInfoExtended> {
    let walked = mold_catalog::sidecar::walk_sidecars(models_dir);
    let mut out = Vec::new();
    for (sidecar_dir, sidecar) in walked {
        if sidecar.kind != "checkpoint" {
            continue;
        }
        // Skip sidecars whose primary file isn't actually present —
        // partial pulls / aborted downloads.
        if mold_catalog::sidecar::primary_path_if_present(&sidecar_dir, &sidecar).is_none() {
            continue;
        }
        // Skip entries already covered by a manifest.
        if mold_core::manifest::find_manifest_by_hf_repo(&sidecar.source_id).is_some() {
            continue;
        }

        let size_gb = sidecar
            .size_bytes
            .map(|b| b as f32 / 1_000_000_000.0)
            .unwrap_or(0.0);

        let (w, h, steps, guidance) = mold_core::manifest::visible_manifests()
            .find(|m| m.family == sidecar.family)
            .map(|m| {
                let cfg = config.resolved_model_config(&m.name);
                (
                    cfg.effective_width(config),
                    cfg.effective_height(config),
                    cfg.effective_steps(config),
                    cfg.effective_guidance(),
                )
            })
            .unwrap_or_else(|| match sidecar.family.as_str() {
                "ltx-video" | "ltx2" => (768, 512, 25, 3.5),
                "sdxl" => (1024, 1024, 20, 7.5),
                "sd15" => (512, 512, 20, 7.5),
                "flux" => (1024, 1024, 20, 3.5),
                "flux2" => (512, 512, 4, 0.0),
                _ => (1024, 1024, 20, 3.5),
            });

        let description = match &sidecar.author {
            Some(a) if !a.is_empty() => format!("{} by {a}", sidecar.name),
            _ => sidecar.name.clone(),
        };

        out.push(ModelInfoExtended {
            downloaded: true,
            defaults: ModelDefaults {
                default_width: w,
                default_height: h,
                default_steps: steps,
                default_guidance: guidance,
                description,
            },
            info: ModelInfo {
                name: sidecar.id.clone(),
                family: sidecar.family.clone(),
                size_gb,
                is_loaded: loaded_model.is_some_and(|n| engine_is_loaded && n == sidecar.id),
                last_used: None,
                hf_repo: String::new(),
            },
            disk_usage_bytes: sidecar.size_bytes,
            remaining_download_bytes: Some(0),
        });
    }
    out
}

/// Check whether a model is available — either already in the cache or
/// has resolvable paths on disk. Returns `Some(paths)` if the model needs
/// to be created from scratch, `None` if already in the cache.
pub(crate) async fn check_model_available(
    state: &AppState,
    model_name: &str,
) -> Result<Option<ModelPaths>, ApiError> {
    // Check the model cache first.
    {
        let cache = state.model_cache.lock().await;
        if cache.contains(model_name) {
            return Ok(None);
        }
    }

    let paths = {
        let config = state.config.read().await;
        if config.manifest_model_needs_download(model_name) {
            None
        } else {
            ModelPaths::resolve(model_name, &config)
        }
    };
    if let Some(paths) = paths {
        return Ok(Some(paths));
    }

    {
        let current = state.config.read().await.clone();
        let fresh_config = current.reload_from_disk_preserving_runtime();
        let needs_download = fresh_config.manifest_model_needs_download(model_name);
        let paths = if needs_download {
            None
        } else {
            ModelPaths::resolve(model_name, &fresh_config)
        };
        {
            let mut config = state.config.write().await;
            *config = fresh_config;
        }
        if let Some(paths) = paths {
            return Ok(Some(paths));
        }
    }

    // Catalog bridge: install (live lookup → cache intent), then resolve the
    // intent against fresh disk state on every request. Lazy resolution
    // means a request that fires while files are still downloading no
    // longer seals a wrong `cfg.vae` into the config — the next request
    // re-resolves once files land.
    if looks_like_catalog_id(model_name) {
        if let Err(install_err) = install_catalog_model(state, model_name).await {
            return Err(install_error_to_api_error(&install_err));
        }
        let intents = state.catalog_intents.read().await;
        let intent = intents
            .get(model_name)
            .ok_or_else(|| {
                ApiError::not_found(format!(
                    "catalog model '{model_name}' is not installed. Download it from \
                     the catalog first."
                ))
            })?
            .clone();
        drop(intents);

        let resolved = {
            let config = state.config.read().await;
            resolve_intent_to_paths(model_name, &intent, &config)
        };
        match resolved {
            Ok(model_cfg) => {
                // Cache the resolved ModelConfig back into config.models so
                // downstream `resolved_model_config` / family lookups still
                // work as before.
                {
                    let mut config = state.config.write().await;
                    config.models.insert(model_name.to_string(), model_cfg);
                }
                let config = state.config.read().await;
                if let Some(paths) = ModelPaths::resolve(model_name, &config) {
                    return Ok(Some(paths));
                }
                // ModelPaths::resolve failed despite successful synthesis —
                // shouldn't happen because resolve_intent_to_paths checks
                // companions, but surface a precise diagnostic if it does.
                return Err(ApiError::not_found(format!(
                    "catalog model '{model_name}' resolved to a config that ModelPaths \
                     could not turn into runtime paths — internal mismatch, please file an issue."
                )));
            }
            Err(e) => return Err(resolve_error_to_api_error(&e)),
        }
    }

    if mold_core::manifest::find_manifest(model_name).is_some() {
        return Err(ApiError::not_found(format!(
            "model '{model_name}' is not downloaded. Run: mold pull {model_name}"
        )));
    }
    Err(ApiError::unknown_model(format!(
        "unknown model '{model_name}'. Run 'mold list' to see available models."
    )))
}

/// Ensure the requested model is loaded on GPU and ready for inference.
///
/// Checks the model cache: if already loaded, just touches the LRU order.
/// If cached but unloaded, reloads it. If not in cache, creates a new engine.
///
/// `hint` carries the per-request resolution / family used by the activation
/// budget. `None` falls back to the previous fixed-headroom approximation —
/// admin-API loads with no resolution context (cache prewarm, etc.) take
/// this path.
pub(crate) async fn ensure_model_ready(
    state: &AppState,
    model_name: &str,
    progress: Option<EngineProgressCallback>,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    let _guard = state.model_load_lock.lock().await;

    // Fast path: model is in cache and loaded.
    {
        let mut cache = state.model_cache.lock().await;
        // Grab active model's VRAM before mutable borrow via get_mut.
        let active_vram = cache.active_vram_bytes();
        if let Some(entry) = cache.get_mut(model_name) {
            if entry.residency == ModelResidency::Gpu {
                // Already loaded — just set up progress callback.
                if let Some(callback) = progress.clone() {
                    entry.engine.set_on_progress(Box::new(move |event| {
                        callback(event);
                    }));
                } else {
                    entry.engine.clear_on_progress();
                }
                return Ok(());
            }

            // Cached but not on GPU (Parked) — need to reload.
            // MPS memory guard: check before unloading the active model.
            // Include the active model's footprint as reclaimable memory.
            let cached_paths = entry.engine.model_paths().cloned();
            if let Some(paths) = cached_paths.as_ref() {
                preflight_memory_guard(model_name, paths, active_vram, 0, hint)?;
            }
            let load_strategy = cached_paths
                .as_ref()
                .map(|paths| {
                    select_server_load_strategy_for_budget(
                        paths,
                        effective_load_available_bytes(active_vram, 0),
                        hint,
                    )
                })
                .unwrap_or(mold_inference::LoadStrategy::Eager);
            if load_strategy == mold_inference::LoadStrategy::Sequential {
                tracing::info!(
                    model = %model_name,
                    "server load strategy degraded to sequential to fit memory budget"
                );
            }

            // Parked engines retain tokenizers/caches for faster reload.
            // First unload the currently active model (if any) to free VRAM.
            if let Some(active_name) = cache.unload_active() {
                #[cfg(feature = "metrics")]
                crate::metrics::clear_model_loaded(&active_name);
                tracing::info!(
                    from = %active_name,
                    to = %model_name,
                    "unloaded active model to reload cached model"
                );
                // Legacy no-worker path only: hardcoded ordinal 0 is safe here
                // because `state.model_load_lock` (taken above) is the only
                // lock protecting GPU 0's primary context on this path — the
                // GpuPool path uses `worker.model_load_lock` and
                // `reclaim_gpu_memory(worker.gpu.ordinal)` via `gpu_worker`.
                mold_inference::reclaim_gpu_memory(0);
            }

            // Take the engine out of cache to load in spawn_blocking. Using
            // `take()` (not `remove()`) keeps the model name in the cache's
            // `in_flight` set so concurrent `check_model_available` calls
            // still see it as logically cached during the load window.
            let cached = cache.take(model_name).ok_or_else(|| {
                ApiError::internal(format!("cache race: model '{model_name}' vanished"))
            })?;
            drop(cache);

            let mut engine = cached.engine;
            if load_strategy == mold_inference::LoadStrategy::Sequential {
                let Some(paths) = cached_paths else {
                    let evicted = {
                        let mut cache = state.model_cache.lock().await;
                        cache.insert(engine, 0)
                    };
                    drop(evicted);
                    return Err(ApiError::internal(format!(
                        "cached engine for '{model_name}' does not expose model paths"
                    )));
                };
                let config = state.config.read().await;
                let offload = std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");
                match mold_inference::create_engine_with_pool(
                    model_name.to_string(),
                    paths,
                    &config,
                    load_strategy,
                    0,
                    offload,
                    Some(state.shared_pool.clone()),
                ) {
                    Ok(new_engine) => {
                        drop(config);
                        drop(engine);
                        engine = new_engine;
                    }
                    Err(e) => {
                        drop(config);
                        let evicted = {
                            let mut cache = state.model_cache.lock().await;
                            cache.insert(engine, 0)
                        };
                        drop(evicted);
                        return Err(ApiError::internal(format!(
                            "failed to recreate cached engine for '{model_name}': {e}"
                        )));
                    }
                }
            }

            if let Some(callback) = progress.clone() {
                engine.set_on_progress(Box::new(move |event| {
                    callback(event);
                }));
            } else {
                engine.clear_on_progress();
            }

            let model_log = model_name.to_string();
            #[cfg(feature = "metrics")]
            let load_start = std::time::Instant::now();
            // Sample VRAM baseline before load so we can record the new
            // model's per-load delta rather than the device-global usage.
            let vram_baseline = mold_inference::device::vram_in_use_bytes(0);
            let join_result = tokio::task::spawn_blocking(move || {
                tracing::info!(model = %model_log, "reloading cached engine...");
                if let Err(e) = engine.load() {
                    tracing::error!("model reload failed: {e:#}");
                    return Err((
                        ApiError::internal(format!("model reload error: {e}")),
                        engine,
                    ));
                }
                Ok(engine)
            })
            .await;

            match join_result {
                Ok(Ok(loaded_engine)) => {
                    #[cfg(feature = "metrics")]
                    {
                        let duration = load_start.elapsed().as_secs_f64();
                        crate::metrics::record_model_load(model_name, duration);
                        crate::metrics::set_model_loaded(model_name);
                        let vram_est = mold_inference::device::vram_in_use_bytes(0);
                        crate::metrics::record_gpu_memory(vram_est);
                    }
                    let vram = mold_inference::device::vram_load_delta(0, vram_baseline);
                    // Insert under the cache lock, but drop the evicted engine
                    // OUTSIDE the lock — `cuMemFree` and safetensor unmap during
                    // `Box<dyn ...>` drop can block other cache users for hundreds
                    // of milliseconds. `insert` clears the in_flight marker.
                    let evicted = {
                        let mut cache = state.model_cache.lock().await;
                        cache.insert(loaded_engine, vram)
                    };
                    drop(evicted);
                }
                Ok(Err((api_err, unloaded_engine))) => {
                    // Put it back as unloaded so cache isn't corrupted. The
                    // unloaded engine has no GPU resources to free, but for
                    // consistency with the file's invariant — never drop an
                    // engine while holding the cache lock — bind and drop
                    // outside.
                    let evicted = {
                        let mut cache = state.model_cache.lock().await;
                        cache.insert(unloaded_engine, 0)
                    };
                    drop(evicted);
                    return Err(api_err);
                }
                Err(join_err) => {
                    // The blocking task aborted (panic that escaped the
                    // closure, runtime shutdown, etc.). Engine is gone —
                    // we can't restore it, so clear the in_flight marker
                    // explicitly. Without this, the model name leaks
                    // forever in `in_flight`: every subsequent
                    // `ensure_model_ready` fast-paths through
                    // `cache.contains()` (which still says true), then
                    // `cache.take()` returns None, and generation fails
                    // with "no engine available after model readiness
                    // check" indefinitely for this model.
                    {
                        let mut cache = state.model_cache.lock().await;
                        cache.clear_in_flight(model_name);
                    }
                    return Err(ApiError::internal(format!(
                        "model reload task failed: {join_err}"
                    )));
                }
            }
            return Ok(());
        }
    }

    // Not in cache — check if model is available on disk.
    match check_model_available(state, model_name).await? {
        Some(paths) => create_and_load_engine(state, model_name, paths, progress, hint).await,
        None => Ok(()),
    }
}

pub(crate) async fn pull_model(
    state: &AppState,
    model: &str,
    progress: Option<DownloadProgressCallback>,
) -> Result<PullStatus, ApiError> {
    if mold_core::manifest::find_manifest(&mold_core::manifest::resolve_model_name(model)).is_none()
    {
        return Err(ApiError::unknown_model(format!(
            "unknown model '{model}'. Run 'mold list' to see available models."
        )));
    }

    let _guard = state.pull_lock.lock().await;

    {
        let config = refresh_config(state).await;
        if config.manifest_model_is_downloaded(model) {
            return Ok(PullStatus::AlreadyAvailable);
        }
    }

    tracing::info!(model = %model, "pulling model via API");

    let opts = mold_core::download::PullOptions::default();
    let new_config = match progress {
        Some(callback) => {
            mold_core::download::pull_and_configure_with_callback(model, callback, &opts)
                .await
                .map(|(config, _)| config)
        }
        None => mold_core::download::pull_and_configure(model, &opts)
            .await
            .map(|(config, _)| config),
    }
    .map_err(|e| {
        tracing::error!("pull failed for {}: {e}", model);
        ApiError::internal(format!("failed to pull model '{}': {e}", model))
    })?;

    {
        let mut config = state.config.write().await;
        *config = new_config;
    }

    tracing::info!(model = %model, "pull complete");
    Ok(PullStatus::Pulled)
}

/// Unload the active model from GPU. The engine remains in the cache (unloaded)
/// so it can be reloaded quickly on the next request.
pub(crate) async fn unload_model(state: &AppState) -> String {
    // Always clear the cached upscaler engine to free GPU memory,
    // regardless of whether a diffusion model is loaded.
    // Use try_lock() to avoid blocking the async runtime if an upscale
    // is in progress (the spawn_blocking thread holds this lock).
    if let Ok(mut upscaler) = state.upscaler_cache.try_lock() {
        if upscaler.is_some() {
            *upscaler = None;
            tracing::info!("upscaler cache cleared");
        }
    }

    let mut cache = state.model_cache.lock().await;
    match cache.unload_active() {
        Some(name) => {
            #[cfg(feature = "metrics")]
            {
                crate::metrics::clear_model_loaded(&name);
                crate::metrics::record_gpu_memory(0);
            }
            drop(cache);
            // Legacy no-worker path only: hardcoded ordinal 0 is safe here
            // because `state.model_load_lock` (taken above) is the only
            // lock protecting GPU 0's primary context on this path — the
            // GpuPool path uses `worker.model_load_lock` and
            // `reclaim_gpu_memory(worker.gpu.ordinal)` via `gpu_worker`.
            mold_inference::reclaim_gpu_memory(0);
            tracing::info!(model = %name, "model unloaded via API");
            format!("unloaded {name}")
        }
        None => "no model loaded".to_string(),
    }
}

async fn create_and_load_engine(
    state: &AppState,
    model_name: &str,
    paths: ModelPaths,
    progress: Option<EngineProgressCallback>,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    // MPS memory guard: reject before unloading current model so it stays operational.
    // Include the active model's footprint as reclaimable memory.
    let active_vram = {
        let cache = state.model_cache.lock().await;
        cache.active_vram_bytes()
    };
    preflight_memory_guard(model_name, &paths, active_vram, 0, hint)?;
    let load_strategy = select_server_load_strategy_for_budget(
        &paths,
        effective_load_available_bytes(active_vram, 0),
        hint,
    );
    if load_strategy == mold_inference::LoadStrategy::Sequential {
        tracing::info!(
            model = %model_name,
            "server load strategy degraded to sequential to fit memory budget"
        );
    }

    // Unload the current active model to free GPU memory.
    // Only reclaim GPU memory if there was an active model — calling
    // reclaim_gpu_memory() (CUDA primary context reset) when nothing was
    // loaded is unnecessary and may misbehave on some driver versions.
    let had_active = {
        let mut cache = state.model_cache.lock().await;
        let result = cache.unload_active();
        if let Some(ref name) = result {
            #[cfg(feature = "metrics")]
            crate::metrics::clear_model_loaded(name);
            tracing::info!(
                from = %name,
                to = %model_name,
                "unloading active model before loading new one"
            );
        }
        result.is_some()
    };
    if had_active {
        // Legacy no-worker path only: hardcoded ordinal 0 is safe here
        // because `state.model_load_lock` (taken above) is the only
        // lock protecting GPU 0's primary context on this path — the
        // GpuPool path uses `worker.model_load_lock` and
        // `reclaim_gpu_memory(worker.gpu.ordinal)` via `gpu_worker`.
        mold_inference::reclaim_gpu_memory(0);
    }

    let config = state.config.read().await;
    let offload = std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");
    let mut new_engine = mold_inference::create_engine_with_pool(
        model_name.to_string(),
        paths,
        &config,
        load_strategy,
        0,
        offload,
        Some(state.shared_pool.clone()),
    )
    .map_err(|e| ApiError::internal(format!("failed to create engine for '{model_name}': {e}")))?;
    drop(config);

    if let Some(callback) = progress {
        new_engine.set_on_progress(Box::new(move |event| {
            callback(event);
        }));
    } else {
        new_engine.clear_on_progress();
    }

    let model_log = model_name.to_string();
    #[cfg(feature = "metrics")]
    let load_start = std::time::Instant::now();
    // Sample VRAM baseline before load so we can record the new model's
    // per-load delta rather than the device-global usage.
    let vram_baseline = mold_inference::device::vram_in_use_bytes(0);
    new_engine = tokio::task::spawn_blocking(move || {
        tracing::info!(model = %model_log, "loading model...");
        new_engine.load().map_err(|e| {
            tracing::error!("model load failed: {e:#}");
            ApiError::internal(format!("model load error: {e}"))
        })?;
        Ok::<_, ApiError>(new_engine)
    })
    .await
    .map_err(|e| ApiError::internal(format!("model load task failed: {e}")))??;

    #[cfg(feature = "metrics")]
    {
        let duration = load_start.elapsed().as_secs_f64();
        crate::metrics::record_model_load(model_name, duration);
        crate::metrics::set_model_loaded(model_name);
    }

    let vram = mold_inference::device::vram_load_delta(0, vram_baseline);
    #[cfg(feature = "metrics")]
    crate::metrics::record_gpu_memory(mold_inference::device::vram_in_use_bytes(0));

    // Insert under the cache lock, but drop the evicted engine OUTSIDE the
    // lock — `cuMemFree` and safetensor unmap during `Box<dyn ...>` drop can
    // block other cache users for hundreds of milliseconds.
    let evicted = {
        let mut cache = state.model_cache.lock().await;
        cache.insert(new_engine, vram)
    };
    drop(evicted);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const GB: u64 = 1_000_000_000;

    /// RAII guard for `MOLD_LTX2_GEMMA_DEVICE` (and the deprecated alias).
    /// Drop restores the prior values so adjacent tests don't see stale
    /// state. Cargo's parallel runner is serialized via a static mutex
    /// because env vars are process-global.
    struct Ltx2GemmaEnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        prior_main: Option<std::ffi::OsString>,
        prior_legacy: Option<std::ffi::OsString>,
    }

    impl Drop for Ltx2GemmaEnvGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("MOLD_LTX2_GEMMA_DEVICE");
                std::env::remove_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
                if let Some(v) = self.prior_main.take() {
                    std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", v);
                }
                if let Some(v) = self.prior_legacy.take() {
                    std::env::set_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", v);
                }
            }
        }
    }

    fn ltx2_gemma_env_guard(value: &str) -> Ltx2GemmaEnvGuard {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prior_main = std::env::var_os("MOLD_LTX2_GEMMA_DEVICE");
        let prior_legacy = std::env::var_os("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
        unsafe {
            std::env::remove_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
            std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", value);
        }
        Ltx2GemmaEnvGuard {
            _lock: lock,
            prior_main,
            prior_legacy,
        }
    }

    /// Build a `ModelPaths` whose `transformer` and `vae` files exist on disk
    /// with a combined size of `total_bytes`. `estimate_peak_memory()` reads
    /// file sizes via `std::fs::metadata`, so the on-disk footprint is what
    /// drives the composition under test.
    fn test_paths_with_total_size(total_bytes: u64) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let transformer = dir.path().join("transformer.safetensors");
        let vae = dir.path().join("vae.safetensors");
        // Split half-and-half between the two required files so that the
        // sum equals `total_bytes`. `set_len` creates a sparse file, which
        // is fast and reports the requested size via `metadata().len()`.
        let half = total_bytes / 2;
        let rest = total_bytes - half;
        let f1 = std::fs::File::create(&transformer).expect("create transformer");
        f1.set_len(half).expect("set transformer len");
        let f2 = std::fs::File::create(&vae).expect("create vae");
        f2.set_len(rest).expect("set vae len");

        let paths = ModelPaths {
            transformer,
            transformer_shards: Vec::new(),
            vae,
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
        (dir, paths)
    }

    /// Sanity: confirm that `estimate_peak_memory` actually sees the on-disk
    /// sizes we set via `test_paths_with_total_size`. The peak includes a
    /// fixed headroom term added by the inference crate, so we just verify
    /// the lower bound matches our component total.
    #[test]
    fn test_paths_helper_sets_file_sizes() {
        let (_dir, paths) = test_paths_with_total_size(10 * GB);
        let peak = mold_inference::device::estimate_peak_memory(
            &paths,
            mold_inference::LoadStrategy::Eager,
        );
        assert!(
            peak >= 10 * GB,
            "expected peak >= 10 GB component sum, got {peak}"
        );
        // The transformer and vae paths must point to real files.
        assert!(PathBuf::from(&paths.transformer).exists());
        assert!(PathBuf::from(&paths.vae).exists());
    }

    /// Composition test: peak fits in `(available + active)` but not in
    /// `available` alone — the inner guard must let the swap proceed.
    #[test]
    fn preflight_uses_active_vram_as_reclaimable() {
        // 10 GB on disk → peak ≈ 10 GB + headroom.
        // Available 8 GB alone is insufficient; add 10 GB active VRAM →
        // 18 GB effective, comfortably above peak.
        let (_dir, paths) = test_paths_with_total_size(10 * GB);
        let result =
            preflight_memory_guard_with_available("swap-test", &paths, 10 * GB, 8 * GB, None);
        assert!(
            result.is_ok(),
            "expected swap to succeed with reclaimable VRAM, got {result:?}"
        );
    }

    /// Composition test: peak exceeds even the post-swap budget — the inner
    /// guard must reject.
    #[test]
    fn preflight_rejects_when_peak_exceeds_effective_available() {
        // 20 GB on disk → peak ≥ 20 GB. Available 8 GB + 5 GB active =
        // 13 GB effective < peak → reject.
        let (_dir, paths) = test_paths_with_total_size(20 * GB);
        let result = preflight_memory_guard_with_available("too-big", &paths, 5 * GB, 8 * GB, None);
        assert!(
            result.is_err(),
            "expected oversized model to be rejected, got Ok"
        );
    }

    #[test]
    fn memory_guard_ok_when_plenty_of_memory() {
        assert!(check_model_memory_budget("test-model", 5 * GB, 20 * GB, "").is_ok());
    }

    #[test]
    fn memory_guard_rejects_over_90pct() {
        let result = check_model_memory_budget("flux-dev:bf16", 19 * GB, 20 * GB, "Try --offload.");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, "INSUFFICIENT_MEMORY");
        assert!(err.error.contains("flux-dev:bf16"));
        assert!(err.error.contains("budget cap"));
    }

    #[test]
    fn memory_guard_ok_at_90pct_boundary() {
        // 18 GB peak, 20 GB available → 90% exactly → should pass
        assert!(check_model_memory_budget("test", 18 * GB, 20 * GB, "").is_ok());
    }

    #[test]
    fn memory_guard_ok_in_warn_zone() {
        // 17 GB peak, 20 GB available → 85% → passes but would warn
        assert!(check_model_memory_budget("test", 17 * GB, 20 * GB, "").is_ok());
    }

    #[test]
    fn memory_guard_ok_below_warn_zone() {
        // 15 GB peak, 20 GB available → 75% → no warn, no error
        assert!(check_model_memory_budget("test", 15 * GB, 20 * GB, "").is_ok());
    }

    #[test]
    fn memory_guard_rejects_tiny_available() {
        // Model larger than total available
        let result = check_model_memory_budget("huge-model", 30 * GB, 16 * GB, "");
        assert!(result.is_err());
    }

    /// CUDA branch math: when free VRAM is small but the active model can be
    /// reclaimed, `effective_available = free + active_vram` should let the
    /// new model load. Without the additive term, a swap of two near-
    /// equal-size models on a fully-loaded GPU would always be rejected.
    #[test]
    fn memory_guard_swap_uses_active_vram_as_reclaimable() {
        // Free VRAM is only 2 GB but the currently-loaded model occupies
        // 18 GB which becomes available on swap → effective = 20 GB.
        // A 15 GB peak model fits comfortably (<= 90% of 20 GB).
        let free_vram = 2 * GB;
        let active_vram = 18 * GB;
        let effective = free_vram + active_vram;
        assert!(check_model_memory_budget("swap-target", 15 * GB, effective, "").is_ok());
    }

    /// Verifies the swap math also rejects when the swap is genuinely
    /// infeasible — peak exceeds even the post-swap budget.
    #[test]
    fn memory_guard_swap_still_rejects_when_oversized() {
        // Free 1 GB + active 8 GB = 9 GB effective. A 15 GB model can't fit
        // even after the active model is unloaded.
        let free_vram = GB;
        let active_vram = 8 * GB;
        let effective = free_vram + active_vram;
        assert!(check_model_memory_budget("too-large", 15 * GB, effective, "").is_err());
    }

    /// Build a `ModelPaths` whose components include text encoders, mirroring a
    /// real FLUX-family layout. Used to verify the Sequential-strategy peak
    /// estimate doesn't sum the encoder onto the transformer (the bug fix).
    fn flux_shaped_paths_with_sizes(
        transformer_gb: u64,
        vae_gb: u64,
        t5_gb: u64,
        clip_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("transformer.safetensors", transformer_gb);
        let vae = mk("vae.safetensors", vae_gb);
        let t5 = mk("t5.safetensors", t5_gb);
        let clip = mk("clip.safetensors", clip_gb);
        let paths = ModelPaths {
            transformer,
            transformer_shards: Vec::new(),
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(t5),
            clip_encoder: Some(clip),
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    /// Regression: a quantized FLUX-shaped model should fit on a 24 GB card
    /// when the sibling model is unloaded and the context reset, even though
    /// the Eager (sum) peak would have been ~24 GB and tripped the 90 %
    /// hard limit.
    ///
    /// Concrete shape: FLUX-dev:q8 → transformer ≈ 12 GB, VAE ≈ 0.3 GB,
    /// T5 ≈ 9.5 GB, CLIP ≈ 0.25 GB. Eager peak = 12+0.3+9.5+0.25+2 ≈ 24 GB
    /// (rejects at 90 % of 24 GB = 21.6). Sequential peak =
    /// max(9.75, 12.3) + 2 ≈ 14.3 GB (passes comfortably).
    #[test]
    fn preflight_passes_for_quantized_flux_on_24gb_card_with_swap() {
        // Use whole-GB sizes to match the helper's u64 parameters; the
        // composition is realistic enough to exercise Eager-vs-Sequential
        // divergence (encoder + transformer both > headroom).
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        // Free 4 GB on a 24 GB card with an 18 GB sibling about to be
        // reclaimed → effective_available passed in by the outer guard
        // is total_vram = 24 GB on CUDA, but we test the inner directly with
        // the Sequential strategy in mind.
        let result = preflight_memory_guard_with_available("flux-dev:q8", &paths, 0, 24 * GB, None);
        assert!(
            result.is_ok(),
            "quantized FLUX must fit on a 24 GB card under the Sequential \
             peak estimate (drop-and-reload encoders), got {result:?}"
        );
    }

    /// Companion: under the *old* Eager-strategy math the same model would
    /// have been rejected. Verifying explicitly so a regression that flips
    /// the strategy back gets caught.
    #[test]
    fn eager_strategy_would_have_rejected_quantized_flux_on_24gb() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        let eager_peak = mold_inference::device::estimate_peak_memory(
            &paths,
            mold_inference::LoadStrategy::Eager,
        );
        // 12 + 1 + 10 + 1 + 2 GB headroom = 26 GB → above 90 % of 24 GB.
        let hard_limit = (24 * GB) * 9 / 10;
        assert!(
            eager_peak > hard_limit,
            "Eager peak ({eager_peak}) should exceed hard limit ({hard_limit}) — \
            this is the false-rejection the Sequential switch fixes"
        );
    }

    #[test]
    fn server_load_strategy_degrades_when_only_sequential_fits() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        let strategy = select_server_load_strategy_for_budget(&paths, Some(24 * GB), None);

        assert_eq!(
            strategy,
            mold_inference::LoadStrategy::Sequential,
            "server load should match the sequential preflight assumption instead \
             of eager-loading a model whose summed components exceed the budget"
        );
    }

    #[test]
    fn server_load_strategy_stays_eager_when_eager_fits() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(8, 1, 2, 1);
        let strategy = select_server_load_strategy_for_budget(&paths, Some(24 * GB), None);

        assert_eq!(strategy, mold_inference::LoadStrategy::Eager);
    }

    #[test]
    fn server_load_strategy_stays_eager_when_no_budget_available() {
        let (_dir, paths) = flux_shaped_paths_with_sizes(12, 1, 10, 1);
        let strategy = select_server_load_strategy_for_budget(&paths, None, None);

        assert_eq!(strategy, mold_inference::LoadStrategy::Eager);
    }

    /// Tier 2.3: the server-side preflight must consume the
    /// resolution-scaled activation budget. A model that fits at 768²
    /// (where the activation budget is the 256 MB floor) must be rejected
    /// at 2048² (where the budget grows past 1 GB) on the same card.
    #[test]
    fn preflight_memory_guard_accepts_resolution_for_activation_budget() {
        // Shape: 23 GB transformer, 1 GB VAE, 9 GB T5, 1 GB CLIP. Sequential
        // peak = max(10, 24) + 2 GB headroom = 26 GB. On a 30 GB card the
        // 90 % hard limit is 27 GB:
        //   * 768²:  26 + 0.256 (floor) = 26.256 ≤ 27  → accept
        //   * 2048²: 26 + 1.09          = 27.09  > 27  → reject
        // Without the activation hint both would land at 26 GB and accept.
        let (_dir, paths) = flux_shaped_paths_with_sizes(23, 1, 9, 1);

        let hint_768 = ActivationHint {
            width: 768,
            height: 768,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };
        let hint_2048 = ActivationHint {
            width: 2048,
            height: 2048,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };

        let card_total = 30 * GB;
        let result_768 = preflight_memory_guard_with_available(
            "flux-dev",
            &paths,
            0,
            card_total,
            Some(hint_768),
        );
        let result_2048 = preflight_memory_guard_with_available(
            "flux-dev",
            &paths,
            0,
            card_total,
            Some(hint_2048),
        );

        assert!(
            result_768.is_ok(),
            "768² FLUX should fit on 30 GB (small activation budget), got {result_768:?}"
        );
        assert!(
            result_2048.is_err(),
            "2048² FLUX must be rejected on 30 GB (large activation budget pushes \
            peak past 90 % cap), got {result_2048:?}"
        );
    }

    fn flux2_klein9b_bf16_paths() -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let shard_a = mk("diffusion_pytorch_model-00001-of-00002.safetensors", 10);
        let shard_b = mk("diffusion_pytorch_model-00002-of-00002.safetensors", 8);
        let vae = mk("flux2-vae.safetensors", 1);
        let te_a = mk("text_encoder-00001-of-00004.safetensors", 5);
        let te_b = mk("text_encoder-00002-of-00004.safetensors", 5);
        let te_c = mk("text_encoder-00003-of-00004.safetensors", 5);
        let te_d = mk("text_encoder-00004-of-00004.safetensors", 1);
        let paths = ModelPaths {
            transformer: shard_a.clone(),
            transformer_shards: vec![shard_a, shard_b],
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![te_a, te_b, te_c, te_d],
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    #[test]
    fn preflight_rejects_flux2_klein9b_bf16_on_24gb_card() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let result = preflight_memory_guard_with_available(
            "flux2-klein-9b:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
        );

        assert!(
            result.is_err(),
            "Klein-9B BF16 must not be admitted on a 24 GB card; \
             the plain file-size sequential estimate undercounts its load-time peak, got {result:?}"
        );
    }

    #[test]
    fn preflight_allows_flux2_klein9b_bf16_on_32gb_card() {
        let (_dir, paths) = flux2_klein9b_bf16_paths();
        let hint = ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Flux2Dit,
        };

        let result = preflight_memory_guard_with_available(
            "flux2-klein-9b:bf16",
            &paths,
            0,
            32 * GB,
            Some(hint),
        );

        assert!(
            result.is_ok(),
            "Klein-9B BF16 should be allowed once the card budget reaches 32 GB, got {result:?}"
        );
    }

    /// Build LTX-2-shaped paths: a single 46 GB single-file checkpoint
    /// (transformer == vae) and a 25 GB Gemma TE in `text_encoder_files`.
    /// Mirrors cv:2752735 on disk.
    fn ltx2_shaped_paths_with_sizes(
        transformer_gb: u64,
        gemma_te_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("ltx2_full.safetensors", transformer_gb);
        let gemma = mk("gemma_te.safetensors", gemma_te_gb);
        // LTX-2 catalog bridge sets vae == transformer (single-file
        // convention). The peak estimator detects this and avoids
        // double-counting.
        let paths = ModelPaths {
            transformer: transformer.clone(),
            transformer_shards: Vec::new(),
            vae: transformer,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![gemma],
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    /// LTX-2 22B (cv:2752735) on a 24 GB 3090 must NOT be falsely rejected
    /// by the file-size-based preflight. The transformer streams blocks
    /// (`Ltx2AvTransformer3DModel::new_streaming`); only ~2 GB of weights
    /// are co-resident at peak. The activation hint marks the family as
    /// `Ltx2Video`, which routes through `streaming_transformer_peak`.
    #[test]
    fn preflight_accepts_ltx2_22b_on_24gb_card_via_streaming_peak() {
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 0);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Ltx2Video,
        };
        let result =
            preflight_memory_guard_with_available("cv:2752735", &paths, 0, 24 * GB, Some(hint));
        assert!(
            result.is_ok(),
            "22B LTX-2 must fit on a 24 GB card under streaming-aware peak \
             (only ~2 blocks co-resident; runtime handles its own memory), \
             got {result:?}",
        );
    }

    /// Without the streaming hint the same paths land on the file-size
    /// peak and reject — pinning the previous behavior so a regression
    /// that flips the hint plumbing back is caught.
    #[test]
    fn preflight_rejects_ltx2_22b_when_hint_marks_non_streaming() {
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 0);
        // Use FluxDit family — same shape, no streaming flag — so the
        // preflight falls through to the file-size estimator and rejects
        // at 90 % of 24 GB.
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::FluxDit,
        };
        let result =
            preflight_memory_guard_with_available("cv:2752735", &paths, 0, 24 * GB, Some(hint));
        assert!(
            result.is_err(),
            "without the LTX-2 streaming hint the file-size peak must reject \
             a 46 GB transformer on a 24 GB card — this anchors the regression \
             that landed before the streaming-aware path",
        );
    }

    /// Encoder phase for LTX-2 still pays full encoder_total when the user
    /// pins the placement to GPU (`MOLD_LTX2_GEMMA_DEVICE=gpu`). With a
    /// 25 GB Gemma TE on a 24 GB card the encoder phase trips the 90 % cap
    /// even when the transformer is streamed — a real OOM the user should
    /// see from the runtime, but the preflight captures it up-front.
    #[test]
    fn preflight_rejects_ltx2_when_encoder_phase_exceeds_card() {
        let _guard = ltx2_gemma_env_guard("gpu");
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 25);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Ltx2Video,
        };
        let result =
            preflight_memory_guard_with_available("cv:2752735", &paths, 0, 24 * GB, Some(hint));
        assert!(
            result.is_err(),
            "25 GB Gemma TE alone exceeds 90 %% of 24 GB during the encoder \
             phase — preflight must surface this even when the transformer \
             is streamed, got {result:?}",
        );
    }

    /// `MOLD_LTX2_GEMMA_DEVICE=cpu` shifts the Gemma TE to system RAM. The
    /// encoder phase no longer competes for VRAM; the streaming-aware peak
    /// collapses to "transformer streaming cap + activation + headroom" and
    /// the same 25 GB Gemma + 46 GB transformer paths admit on a 24 GB card.
    /// This is the load-bearing behavior on a single 3090 running cv:2752735.
    #[test]
    fn preflight_admits_ltx2_22b_with_25gb_gemma_when_resolver_picks_cpu() {
        let _guard = ltx2_gemma_env_guard("cpu");
        let (_dir, paths) = ltx2_shaped_paths_with_sizes(46, 25);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::Ltx2Video,
        };
        let result =
            preflight_memory_guard_with_available("cv:2752735", &paths, 0, 24 * GB, Some(hint));
        assert!(
            result.is_ok(),
            "with MOLD_LTX2_GEMMA_DEVICE=cpu the encoder phase should not \
             count against GPU VRAM, so cv:2752735 must admit on 24 GB even \
             with a 25 GB Gemma TE, got {result:?}",
        );
    }

    /// `ActivationHint::from_request` picks the right family + batch for a
    /// real-shaped GenerateRequest.
    #[test]
    fn activation_hint_from_request_classifies_correctly() {
        let mut req = GenerateRequest {
            prompt: "test".into(),
            negative_prompt: None,
            model: "flux-dev:bf16".into(),
            width: 1024,
            height: 1024,
            steps: 20,
            guidance: 3.5,
            seed: None,
            batch_size: 1,
            output_format: Default::default(),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            edit_images: None,
            strength: 1.0,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        };

        // FLUX is guidance-distilled → batch=1 even with guidance > 1.
        let hint_flux = ActivationHint::from_request(&req, "flux");
        assert_eq!(hint_flux.family, ActivationFamily::FluxDit);
        assert_eq!(hint_flux.batch, 1);

        // SDXL with CFG → batch=2.
        let hint_sdxl = ActivationHint::from_request(&req, "sdxl");
        assert_eq!(hint_sdxl.family, ActivationFamily::SdxlUnet);
        assert_eq!(hint_sdxl.batch, 2);

        // SDXL with no CFG (LCM/Turbo) → batch=1.
        req.guidance = 1.0;
        let hint_sdxl_lcm = ActivationHint::from_request(&req, "sdxl");
        assert_eq!(hint_sdxl_lcm.batch, 1);

        // Unknown family slug falls through to FluxDit.
        let hint_unknown = ActivationHint::from_request(&req, "totally-bogus");
        assert_eq!(hint_unknown.family, ActivationFamily::FluxDit);
    }

    // ── FLUX catalog bridge: VAE-bundle probe + flux-vae companion wiring ──

    /// Direct-unit test of `copy_catalog_companion("flux-vae", _)`: when
    /// `cfg.vae` is unset (transformer-only checkpoint) the handler must
    /// populate it from the companion's resolved `paths.transformer`.
    /// The same handler must NOT clobber a bundled-VAE cfg.vae.
    #[test]
    fn copy_catalog_companion_flux_vae_populates_when_unset() {
        let mut cfg = mold_core::ModelConfig {
            family: Some("flux".into()),
            transformer: Some("/m/cv-x/flux/civitai/x/unet.safetensors".into()),
            vae: None,
            ..Default::default()
        };
        let paths = ModelPaths {
            transformer: PathBuf::from("/m/flux-vae/ae.safetensors"),
            transformer_shards: Vec::new(),
            vae: PathBuf::from("/m/flux-vae/ae.safetensors"),
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
        copy_catalog_companion(&mut cfg, "flux-vae", &paths);
        assert_eq!(
            cfg.vae.as_deref(),
            Some("/m/flux-vae/ae.safetensors"),
            "transformer-only cfg.vae must be populated from the flux-vae companion path"
        );
    }

    #[test]
    fn copy_catalog_companion_flux_vae_does_not_override_bundled() {
        let mut cfg = mold_core::ModelConfig {
            family: Some("flux".into()),
            transformer: Some("/m/cv-x/flux/civitai/x/full.safetensors".into()),
            vae: Some("/m/cv-x/flux/civitai/x/full.safetensors".into()),
            ..Default::default()
        };
        let paths = ModelPaths {
            transformer: PathBuf::from("/m/flux-vae/ae.safetensors"),
            transformer_shards: Vec::new(),
            vae: PathBuf::from("/m/flux-vae/ae.safetensors"),
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
        copy_catalog_companion(&mut cfg, "flux-vae", &paths);
        assert_eq!(
            cfg.vae.as_deref(),
            Some("/m/cv-x/flux/civitai/x/full.safetensors"),
            "bundled cfg.vae must survive the flux-vae companion pass"
        );
    }

    /// Synthesize a minimal safetensors at `path` that lists the given
    /// keys in the JSON header (each as a 1-element F32 tensor sharing the
    /// same 4-byte zero blob). Sufficient for the header-peek probe; no
    /// dep on the `safetensors` crate.
    fn write_safetensors_with_keys(path: &std::path::Path, keys: &[&str]) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        for key in keys {
            header.insert(
                (*key).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }),
            );
        }
        let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).expect("create fixture");
        f.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_json).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
    }

    fn flux_unet_only_catalog_entry(
        version_id: &str,
        file_name: &str,
    ) -> mold_catalog::entry::CatalogEntry {
        use mold_catalog::entry::{
            CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, LicenseFlags,
            Modality, RecipeFile, Source, TokenKind,
        };
        use mold_catalog::families::Family;

        CatalogEntry {
            id: CatalogId::from(format!("cv:{version_id}")),
            source: Source::Civitai,
            source_id: version_id.to_string(),
            name: "FLUX Unet-only fine-tune".into(),
            author: Some("someone".into()),
            family: Family::Flux,
            family_role: FamilyRole::Finetune,
            sub_family: None,
            modality: Modality::Image,
            kind: mold_catalog::entry::Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: mold_catalog::entry::Bundling::SingleFile,
            size_bytes: Some(12_000_000_000),
            download_count: 0,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["t5-v1_1-xxl".into(), "clip-l".into(), "flux-vae".into()],
            download_recipe: DownloadRecipe {
                files: vec![RecipeFile {
                    url: format!("https://civitai.com/api/download/models/{version_id}"),
                    dest: format!("{{family}}/civitai/{version_id}/{file_name}"),
                    sha256: Some("DEAD".repeat(16)),
                    size_bytes: Some(12_000_000_000),
                }],
                needs_token: Some(TokenKind::Civitai),
            },
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
        }
    }

    /// Transformer-only FLUX checkpoint: synth must defer cfg.vae and the
    /// flux-vae companion path must populate it (when present in config).
    ///
    /// Sets MOLD_HOME to the tempdir so `ModelPaths::resolve` doesn't walk
    /// the developer's real `~/.mold/models/.hf-cache/` (which on this
    /// machine has a real flux-vae shipped from a manifest install).
    #[test]
    fn synthesize_catalog_config_uses_flux_vae_companion_when_bundle_lacks_vae() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        // SAFETY: env mutation is racy with other tests; this is the only
        // test in this module that touches MOLD_HOME and the suite serializes
        // env-var tests by combining them into one #[test].
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

        // Place a transformer-only safetensors fixture at the path that
        // the recipe-rendering will produce for `cv:994561`.
        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );

        // Stub flux-vae's manifest config so `ModelPaths::resolve("flux-vae", _)`
        // returns a path WITHOUT walking the dev machine's real HF cache.
        let vae_path = models_dir.join("flux-vae/ae.safetensors");
        std::fs::create_dir_all(vae_path.parent().unwrap()).unwrap();
        std::fs::File::create(&vae_path).unwrap();
        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        config.models.insert(
            "flux-vae".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(vae_path.to_string_lossy().into_owned()),
                vae: Some(vae_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        // Stub clip-l + t5 so their absence doesn't blow up the loop —
        // they're declared in `companions_for(Family::Flux, _)` and
        // unresolved entries log a warning rather than failing, but we set
        // them anyway to verify they don't interfere.
        config.models.insert(
            "clip-l".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!("{}/clip-l/model.safetensors", models_dir.display())),
                vae: Some(format!("{}/clip-l/model.safetensors", models_dir.display())),
                ..Default::default()
            },
        );
        config.models.insert(
            "t5-v1_1-xxl".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!(
                    "{}/t5-v1_1-xxl/t5xxl_fp16.safetensors",
                    models_dir.display()
                )),
                vae: Some(format!(
                    "{}/t5-v1_1-xxl/t5xxl_fp16.safetensors",
                    models_dir.display()
                )),
                ..Default::default()
            },
        );

        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let synth = synthesize_catalog_config(&entry, models_dir, &config).unwrap();

        // cfg.transformer points at the recipe-rendered primary path.
        assert_eq!(
            synth.transformer.as_deref(),
            primary_path.to_str(),
            "transformer must point at the on-disk primary checkpoint"
        );
        // cfg.vae was populated from the flux-vae companion's resolved
        // path — NOT from the transformer-only primary.
        assert_eq!(
            synth.vae.as_deref(),
            vae_path.to_str(),
            "flux-vae companion must populate cfg.vae for transformer-only checkpoints",
        );

        // Restore env for sibling tests.
        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    /// Bundled-VAE FLUX checkpoint: synth must keep cfg.vae == primary;
    /// flux-vae companion is a no-op.
    #[test]
    fn synthesize_catalog_config_uses_bundled_vae_when_present() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        // SAFETY: env mutation is racy with other tests; this is the
        // only test in this module that touches MOLD_HOME for the
        // bundled-VAE flow. Pair with the transformer-only sibling.
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

        let primary_path = models_dir.join("cv-101010/flux/civitai/101010/flux_full.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        // Bundled-VAE: A1111 prefix on the encoder.
        write_safetensors_with_keys(
            &primary_path,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
                "first_stage_model.encoder.conv_in.weight",
            ],
        );

        let vae_path = models_dir.join("flux-vae/ae.safetensors");
        std::fs::create_dir_all(vae_path.parent().unwrap()).unwrap();
        std::fs::File::create(&vae_path).unwrap();
        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        // Stub all FLUX companions so resolve_intent_to_paths' hard
        // companion check passes — even with flux-vae present we want to
        // assert cfg.vae stays at the primary.
        config.models.insert(
            "flux-vae".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(vae_path.to_string_lossy().into_owned()),
                vae: Some(vae_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        config.models.insert(
            "clip-l".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!("{}/clip-l/model.safetensors", models_dir.display())),
                vae: Some(format!("{}/clip-l/model.safetensors", models_dir.display())),
                ..Default::default()
            },
        );
        config.models.insert(
            "t5-v1_1-xxl".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(format!(
                    "{}/t5-v1_1-xxl/t5xxl_fp16.safetensors",
                    models_dir.display()
                )),
                vae: Some(format!(
                    "{}/t5-v1_1-xxl/t5xxl_fp16.safetensors",
                    models_dir.display()
                )),
                ..Default::default()
            },
        );

        let entry = flux_unet_only_catalog_entry("101010", "flux_full.safetensors");
        let synth = synthesize_catalog_config(&entry, models_dir, &config).unwrap();

        assert_eq!(synth.transformer.as_deref(), primary_path.to_str());
        assert_eq!(
            synth.vae.as_deref(),
            primary_path.to_str(),
            "bundled-VAE checkpoint must keep cfg.vae == primary; flux-vae companion is a no-op",
        );

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    // ── Lazy intent / disk-aware resolution regression tests ───────────────

    /// Stub the FLUX companion manifest entries (flux-vae, clip-l, t5)
    /// pointing at on-disk files inside `models_dir`. Mirrors what the
    /// real manifest installer would have done.
    fn stub_flux_companion_paths_in_dir(
        config: &mut mold_core::Config,
        models_dir: &std::path::Path,
        flux_vae_present: bool,
    ) {
        let vae_path = models_dir.join("flux-vae/ae.safetensors");
        std::fs::create_dir_all(vae_path.parent().unwrap()).unwrap();
        if flux_vae_present {
            std::fs::File::create(&vae_path).unwrap();
        }
        config.models.insert(
            "flux-vae".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(vae_path.to_string_lossy().into_owned()),
                vae: Some(vae_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        let clip_path = models_dir.join("clip-l/model.safetensors");
        std::fs::create_dir_all(clip_path.parent().unwrap()).unwrap();
        std::fs::File::create(&clip_path).unwrap();
        config.models.insert(
            "clip-l".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(clip_path.to_string_lossy().into_owned()),
                vae: Some(clip_path.to_string_lossy().into_owned()),
                clip_tokenizer: Some(format!("{}/clip-l/tokenizer.json", models_dir.display())),
                ..Default::default()
            },
        );
        let t5_path = models_dir.join("t5-v1_1-xxl/t5xxl_fp16.safetensors");
        std::fs::create_dir_all(t5_path.parent().unwrap()).unwrap();
        std::fs::File::create(&t5_path).unwrap();
        config.models.insert(
            "t5-v1_1-xxl".into(),
            mold_core::ModelConfig {
                family: Some("companion".into()),
                transformer: Some(t5_path.to_string_lossy().into_owned()),
                vae: Some(t5_path.to_string_lossy().into_owned()),
                t5_tokenizer: Some(format!(
                    "{}/t5-v1_1-xxl/tokenizer.json",
                    models_dir.display()
                )),
                ..Default::default()
            },
        );
    }

    /// Task 3 (Step 1): synthesis must produce the same intent regardless
    /// of whether the primary file is on disk yet. The disk-dependent
    /// part is moved to `resolve_intent_to_paths`, which is only called
    /// at engine-load time.
    #[test]
    fn synthesis_intent_is_consistent_before_and_after_download() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");

        let intent_absent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();

        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );

        let intent_present =
            mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();

        assert_eq!(
            intent_absent, intent_present,
            "intent synthesis must be pure — independent of disk state"
        );
    }

    /// Task 3 (Step 2 + Step 3): with the primary present as a
    /// transformer-only checkpoint, resolution picks the flux-vae
    /// companion for cfg.vae rather than the primary checkpoint.
    #[test]
    fn resolve_intent_picks_flux_vae_companion_when_primary_is_transformer_only() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
            ],
        );

        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        stub_flux_companion_paths_in_dir(&mut config, models_dir, true);

        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let cfg = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap();

        let vae_path = models_dir.join("flux-vae/ae.safetensors");
        assert_eq!(cfg.vae.as_deref(), vae_path.to_str());
        assert_eq!(cfg.transformer.as_deref(), primary_path.to_str());

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    /// Task 5 (Step 1): resolution surfaces the *specific* missing
    /// companion name rather than a generic "missing required components"
    /// blob. The CompanionConfigMissing variant fires when the manifest
    /// entry isn't in the user's Config.models — a real config bug.
    #[test]
    fn resolve_intent_returns_error_naming_missing_required_companion() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();

        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );

        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

        // Empty config — none of t5, clip-l, flux-vae are installed.
        let config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };

        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();
        let err = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap_err();

        let msg = err.to_string();
        // The first missing companion in the FLUX list is t5-v1_1-xxl;
        // we just need the error to name *some* specific companion.
        assert!(
            msg.contains("t5-v1_1-xxl") || msg.contains("clip-l") || msg.contains("flux-vae"),
            "error must name a specific missing companion, got: {msg}"
        );
        assert!(matches!(err, ResolveError::CompanionConfigMissing { .. }));

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    /// Task 6: with the lazy intent / resolve flow, a second request
    /// after the file lands succeeds where the first might have raced.
    /// This exercises the no-stale-config invariant directly: both calls
    /// run the full intent + resolve pipeline; the second one sees the
    /// fresh disk state.
    #[test]
    fn cv_id_resolves_when_files_arrive_after_initial_request() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let _saved = std::env::var("MOLD_HOME").ok();
        unsafe { std::env::set_var("MOLD_HOME", models_dir.to_string_lossy().as_ref()) };

        let entry =
            flux_unet_only_catalog_entry("994561", "realHornyProV3_realHornyProV3Unet.safetensors");
        let mut config = mold_core::Config {
            models_dir: models_dir.to_string_lossy().into_owned(),
            ..Default::default()
        };
        // Companions are present; primary file initially missing.
        stub_flux_companion_paths_in_dir(&mut config, models_dir, true);

        let intent = mold_catalog::synthesis::synthesize_intent(&entry, models_dir).unwrap();

        // First resolve — primary not yet on disk. The flux-vae companion
        // populates cfg.vae; the FLUX probe is short-circuited because
        // the file doesn't exist. cfg.transformer points at the expected
        // future location (intent-derived).
        let cfg_first = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap();
        let primary_path = models_dir
            .join("cv-994561/flux/civitai/994561/realHornyProV3_realHornyProV3Unet.safetensors");
        let vae_path = models_dir.join("flux-vae/ae.safetensors");
        assert_eq!(cfg_first.transformer.as_deref(), primary_path.to_str());
        assert_eq!(cfg_first.vae.as_deref(), vae_path.to_str());

        // File arrives mid-flight as a transformer-only checkpoint.
        std::fs::create_dir_all(primary_path.parent().unwrap()).unwrap();
        write_safetensors_with_keys(
            &primary_path,
            &["double_blocks.0.img_attn.proj.weight", "img_in.weight"],
        );

        // Second resolve sees the file: probe runs, declares "no bundled
        // VAE", flux-vae companion still wins. Same result, but this time
        // through the fully-armed disk-aware path.
        let cfg_second = resolve_intent_to_paths("cv:994561", &intent, &config).unwrap();
        assert_eq!(cfg_second.transformer.as_deref(), primary_path.to_str());
        assert_eq!(cfg_second.vae.as_deref(), vae_path.to_str());

        unsafe {
            match _saved {
                Some(v) => std::env::set_var("MOLD_HOME", v),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    // ── InstallError translation tests (Task 4) ────────────────────────────

    #[test]
    fn live_error_to_install_error_maps_404_to_not_found() {
        let upstream = mold_catalog::live::LiveSearchError::Upstream {
            host: "civitai.com",
            status: 404,
            body: "{\"error\": \"not found\"}".to_string(),
        };
        let mapped = live_error_to_install_error("cv:42", &upstream);
        assert!(matches!(mapped, mold_core::InstallError::NotFound(_)));
    }

    #[test]
    fn live_error_to_install_error_maps_5xx_to_recipe_malformed() {
        let upstream = mold_catalog::live::LiveSearchError::Upstream {
            host: "civitai.com",
            status: 500,
            body: "internal".into(),
        };
        let mapped = live_error_to_install_error("cv:42", &upstream);
        assert!(matches!(
            mapped,
            mold_core::InstallError::RecipeMalformed(_)
        ));
    }

    #[test]
    fn install_error_to_api_error_maps_network_to_502() {
        let err = mold_core::InstallError::Network("dns: civitai.com".into());
        let api = install_error_to_api_error(&err);
        // API error doesn't expose .status() publicly, but .code is
        // "INTERNAL_ERROR" with the BAD_GATEWAY status flag (see
        // ApiError::internal_with_status). The user-visible message must
        // carry "network unreachable" so they know what happened.
        assert!(api.error.contains("network unreachable"));
    }

    #[test]
    fn install_error_to_api_error_maps_not_found_to_404() {
        let err = mold_core::InstallError::NotFound("cv:99999999".into());
        let api = install_error_to_api_error(&err);
        assert_eq!(api.code, "MODEL_NOT_FOUND");
    }

    // ── preflight error message budget-cap correctness ───────────────────────

    /// The rejection fraction used by `check_model_memory_budget`. Pinned so a
    /// future change to the factor forces a matching update to the error message
    /// (and this test).
    const BUDGET_FRACTION_NUMERATOR: u64 = 9;
    const BUDGET_FRACTION_DENOMINATOR: u64 = 10;

    /// Compute the budget cap the way `check_model_memory_budget` does.
    fn expected_budget_cap(available: u64) -> u64 {
        available * BUDGET_FRACTION_NUMERATOR / BUDGET_FRACTION_DENOMINATOR
    }

    /// The rejection error message must display the budget cap (the number that
    /// was actually compared against peak), not the raw available VRAM. This is
    /// the root-cause regression test for the "24.4 GB exceeds 25.3 GB" bug.
    #[test]
    fn preflight_error_message_states_correct_budget_cap() {
        // Mirrors the user's reported values: peak=24.4 GB, free=25.3 GB.
        // Cap = 25.3 × 0.9 = 22.77 GB. Peak (24.4) > cap (22.77) → reject.
        let peak: u64 = 24_400_000_000;
        let available: u64 = 25_300_000_000;
        let cap = expected_budget_cap(available);

        // Sanity: the test scenario is actually a rejection.
        assert!(
            peak > cap,
            "test invariant: peak ({peak}) must exceed cap ({cap})"
        );

        let result = check_model_memory_budget(
            "qwen-image:q8",
            peak,
            available,
            "Try a smaller variant (e.g. ':q5' / ':q4'), enable --offload (FLUX), or close other GPU apps.",
        );
        assert!(result.is_err(), "expected rejection, got Ok");

        let err = result.unwrap_err();
        let msg = &err.error;

        // The message must contain the cap, not just the raw available.
        let cap_gb = cap as f64 / 1_000_000_000.0;
        let cap_str = format!("{cap_gb:.1}");
        assert!(
            msg.contains("budget cap"),
            "error must mention 'budget cap', got: {msg}"
        );
        assert!(
            msg.contains(&cap_str),
            "error must contain the cap value ~{cap_str} GB, got: {msg}"
        );

        // The message must NOT imply that peak < available (the original bug).
        // If the message says "exceeds X GB" where X > peak, the user will be confused.
        // We detect this by checking there's no bare available_gb with no "cap" context
        // that would make the inequality look false.
        let available_gb = available as f64 / 1_000_000_000.0;
        let available_str = format!("{available_gb:.1}");
        // available_gb should appear only as the input to the cap formula, not as the
        // comparison target. Presence of "budget cap" already anchors correct phrasing.
        let _ = available_str; // checked indirectly via the "budget cap" assertion above
    }

    /// The "exceeds" target printed in the rejection message must always be
    /// strictly less than the printed peak. A table of (peak_gb, available_gb)
    /// rejection scenarios verifies no phrasing inverts the inequality.
    #[test]
    fn preflight_error_message_does_not_imply_peak_less_than_available() {
        let scenarios: &[(f64, f64)] = &[
            // (peak_gb, available_gb) — all must trigger rejection
            (24.4, 25.3), // user-reported case
            (19.0, 20.0), // 19 > 90% of 20 = 18
            (10.0, 10.5), // just over 90%
            (30.0, 32.0), // 30 > 90% of 32 = 28.8
            (9.1, 10.0),  // 9.1 > 90% of 10 = 9
        ];
        for &(peak_gb, available_gb) in scenarios {
            let peak = (peak_gb * 1_000_000_000.0) as u64;
            let available = (available_gb * 1_000_000_000.0) as u64;
            let cap = expected_budget_cap(available);

            // Only test rejection scenarios.
            if peak <= cap {
                continue;
            }

            let result =
                check_model_memory_budget("test-model", peak, available, "Try a smaller variant.");
            assert!(
                result.is_err(),
                "expected rejection for peak={peak_gb} available={available_gb}, got Ok"
            );

            let msg = result.unwrap_err().error;

            // The comparison target in the message ("budget cap") must be < peak.
            // We verify this by asserting the cap value appears in the message.
            let cap_gb = cap as f64 / 1_000_000_000.0;
            let cap_str = format!("{cap_gb:.1}");
            assert!(
                msg.contains("budget cap"),
                "scenario peak={peak_gb} available={available_gb}: \
                 message must say 'budget cap', got: {msg}"
            );
            assert!(
                msg.contains(&cap_str),
                "scenario peak={peak_gb} available={available_gb}: \
                 message must include cap={cap_str}, got: {msg}"
            );
        }
    }

    // ── LTX-Video preflight regression (Part 1) ──────────────────────────────

    /// Build LTX-Video-shaped paths: separate transformer (13B BF16 ≈ 26 GB),
    /// VAE (~0.5 GB), and T5 encoder (~9.5 GB). Mirrors the on-disk layout of
    /// `ltx-video-0.9.8-13b-dev:bf16` pulled via `mold pull`.
    fn ltx_video_13b_paths(
        transformer_gb: u64,
        vae_gb: u64,
        t5_gb: u64,
    ) -> (tempfile::TempDir, ModelPaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mk = |name: &str, sz: u64| {
            let p = dir.path().join(name);
            let f = std::fs::File::create(&p).unwrap();
            f.set_len(sz * GB).unwrap();
            p
        };
        let transformer = mk("ltx-video-0.9.8-13b-dev_fp16.safetensors", transformer_gb);
        let vae = mk("ltx-video-vae.safetensors", vae_gb);
        let t5 = mk("t5xxl_fp16.safetensors", t5_gb);
        let paths = ModelPaths {
            transformer,
            transformer_shards: Vec::new(),
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(t5),
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        };
        (dir, paths)
    }

    /// Regression: `ltx-video-0.9.8-13b-dev:bf16` at 768×512×25 on a 24 GB
    /// card MUST be rejected by the preflight. The transformer is ~26 GB BF16
    /// (non-streaming, loaded whole), which alone exceeds 24 GB VRAM. Before
    /// the fix, `activation_family_for("ltx-video")` returned `Ltx2Video`
    /// which triggered the streaming cap path (6 GB) and let the load through,
    /// causing a hard OOM during `load_transformer`.
    #[test]
    fn preflight_rejects_ltx_video_13b_at_768x512x25_on_24gb_card() {
        // 26 GB transformer + 0.5 GB VAE + 9.5 GB T5 (mirrors real disk layout).
        // Sequential peak = max(T5=9.5, transformer+VAE=26.5) + 2 GB headroom
        //                 = 26.5 + 2 = 28.5 GB > 90% of 24 GB (21.6 GB) → reject.
        let (_dir, paths) = ltx_video_13b_paths(26, 1, 10);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::LtxVideo,
        };
        let result = preflight_memory_guard_with_available(
            "ltx-video-0.9.8-13b-dev:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
        );
        assert!(
            result.is_err(),
            "13B LTX-Video BF16 (26 GB transformer) must be rejected on a 24 GB card — \
             the transformer is not streamed and its full weight must be counted, \
             got {result:?}",
        );
        let err = result.unwrap_err();
        // Message must surface the LTX-Video-specific mitigation hints.
        assert!(
            err.error.contains("frames") || err.error.contains("width"),
            "rejection message must suggest reducing frames or resolution, got: {}",
            err.error,
        );
    }

    /// Golden: the preflight estimate for LTX-Video 13B BF16 must be within
    /// ~3 GB of the expected peak (Sequential: max(T5, transformer+VAE) +
    /// 2 GB headroom = max(9.5, 26.5) + 2 = 28.5 GB). We accept ±3 GB to
    /// account for rounding in file sizes; the key invariant is that it's
    /// never so low that a 24 GB card incorrectly admits the load.
    #[test]
    fn preflight_estimate_for_ltx_video_13b_within_expected_range() {
        let (_dir, paths) = ltx_video_13b_paths(26, 1, 10);
        // Sequential peak = max(T5=10, transformer+VAE=27) + 2 headroom = 29 GB.
        // (We use 10 GB for T5 to keep nice round numbers; real T5 is ~9.5 GB.)
        let expected_gb = 29u64;
        let peak = mold_inference::device::estimate_peak_memory(
            &paths,
            mold_inference::LoadStrategy::Sequential,
        );
        let peak_gb = peak / GB;
        assert!(
            peak_gb >= expected_gb.saturating_sub(3),
            "peak estimate ({peak_gb} GB) is unexpectedly low — LTX-Video 13B BF16 \
             sequential estimate should be ≥ {} GB",
            expected_gb.saturating_sub(3),
        );
        assert!(
            peak_gb <= expected_gb + 3,
            "peak estimate ({peak_gb} GB) is unexpectedly high for 26+1+10 GB layout \
             — should be ≤ {} GB",
            expected_gb + 3,
        );
    }

    /// `activation_family_for("ltx-video")` must return `LtxVideo` (non-streaming),
    /// not `Ltx2Video`. Before the fix both slugs returned `Ltx2Video`, which
    /// caused the preflight to apply the streaming cap and admit an OOM load.
    #[test]
    fn activation_family_for_ltx_video_is_non_streaming() {
        let family = mold_inference::device::activation_family_for("ltx-video");
        assert_eq!(
            family,
            ActivationFamily::LtxVideo,
            "ltx-video slug must map to LtxVideo (non-streaming, full-weight load)"
        );
        // Verify it does NOT trigger the streaming cap path.
        assert!(
            !family.streaming_transformer(),
            "LtxVideo must NOT be treated as a streaming transformer — \
             it loads the entire weight file into VRAM at generate time"
        );
        // Verify ltx2 still maps to the streaming variant.
        assert!(
            mold_inference::device::activation_family_for("ltx2").streaming_transformer(),
            "ltx2 must still map to the streaming family"
        );
    }

    /// The rejection message for an OOM-at-preflight LTX-Video load must
    /// mention reducing frames or resolution (not `--offload`, which is a
    /// FLUX-specific flag and not applicable to LTX-Video).
    #[test]
    fn preflight_rejection_message_for_ltx_video_suggests_frames_or_resolution() {
        let (_dir, paths) = ltx_video_13b_paths(26, 1, 10);
        let hint = ActivationHint {
            width: 768,
            height: 512,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::LtxVideo,
        };
        let result = preflight_memory_guard_with_available(
            "ltx-video-0.9.8-13b-dev:bf16",
            &paths,
            0,
            24 * GB,
            Some(hint),
        );
        let err = result.expect_err("must reject");
        assert!(
            err.error.contains("frames") || err.error.contains("width"),
            "LTX-Video rejection message must suggest reducing frames or \
             width/height (not --offload), got: {}",
            err.error,
        );
        // Must NOT suggest --offload (that's a FLUX flag, not applicable here).
        assert!(
            !err.error.contains("--offload"),
            "LTX-Video rejection must not mention --offload (FLUX-only flag), \
             got: {}",
            err.error,
        );
    }
}
