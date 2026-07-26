use std::path::Path;

use mold_core::{GenerateRequest, ModelPaths};
use mold_inference::device::{activation_bytes, activation_family_for, ActivationFamily};

use crate::routes::ApiError;

fn transformer_path_lower(paths: &ModelPaths) -> String {
    paths.transformer.to_string_lossy().to_ascii_lowercase()
}

fn transformer_path_looks_flux2(path: &str) -> bool {
    path.contains("/flux2/") || path.contains("flux2")
}

fn transformer_path_looks_ltx2(path: &str) -> bool {
    path.contains("/ltx2/") || path.contains("ltx2")
}

fn transformer_path_looks_zimage(path: &str) -> bool {
    path.contains("/z-image/") || path.contains("zimage")
}

fn transformer_path_is_gguf(paths: &ModelPaths) -> bool {
    paths
        .transformer
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

fn model_component_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn transformer_component_size(paths: &ModelPaths) -> u64 {
    if paths.transformer_shards.is_empty() {
        model_component_size(&paths.transformer)
    } else {
        paths
            .transformer_shards
            .iter()
            .map(|path| model_component_size(path))
            .sum()
    }
}

fn large_flux_bf16_should_auto_offload(paths: &ModelPaths, hint: Option<ActivationHint>) -> bool {
    const LARGE_FLUX_BF16_TRANSFORMER_BYTES: u64 = 20_000_000_000;

    if !hint.is_some_and(|h| h.family == ActivationFamily::FluxDit)
        || transformer_path_is_gguf(paths)
    {
        return false;
    }

    let transformer_path = transformer_path_lower(paths);
    if transformer_path_looks_flux2(&transformer_path)
        || transformer_path_looks_zimage(&transformer_path)
        || transformer_path_looks_ltx2(&transformer_path)
        || transformer_path.contains("nvfp4")
    {
        return false;
    }

    transformer_component_size(paths) >= LARGE_FLUX_BF16_TRANSFORMER_BYTES
}

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
pub(crate) fn check_model_memory_budget(
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
/// reducing `frames` or `width`/`height`; for image families, resolution and
/// batch size are usually the first levers because activation and VAE
/// workspace can dominate the checkpoint size.
pub(crate) fn rejection_suggestion(hint: Option<ActivationHint>) -> &'static str {
    match hint.map(|h| h.family) {
        Some(ActivationFamily::LtxVideo) => {
            "Try reducing --frames or --width/--height, use a quantized variant \
             (e.g. ':q8'), or close other GPU apps."
        }
        _ => {
            "Try lowering --width/--height, reduce --batch, use a smaller/quantized \
             variant if available, enable --offload for FLUX, or close other GPU apps."
        }
    }
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
#[cfg(test)]
pub(crate) fn preflight_memory_guard_with_available(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    preflight_memory_guard_with_available_on_gpu(
        model_name,
        paths,
        active_vram_bytes,
        available_bytes,
        0,
        hint,
    )
}

pub(crate) fn preflight_memory_guard_with_available_on_gpu(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
    gpu_ordinal: usize,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    let forced_offload = matches!(
        mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
        Some("1") | Some("true") | Some("yes")
    );
    let gemma_competes = ltx2_encoder_phase_competes_with_transformer_gpu(gpu_ordinal);
    preflight_memory_guard_with_available_and_policy(
        model_name,
        paths,
        active_vram_bytes,
        available_bytes,
        hint,
        forced_offload,
        gemma_competes,
    )
}

pub(crate) fn preflight_memory_guard_with_available_and_policy(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
    hint: Option<ActivationHint>,
    forced_offload: bool,
    gemma_competes: bool,
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
    let transformer_path = transformer_path_lower(paths);
    let streaming = hint
        .map(|h| h.family.streaming_transformer())
        .unwrap_or_else(|| transformer_path_looks_ltx2(&transformer_path));
    let flux_offload = (hint.is_some_and(|h| h.family == ActivationFamily::FluxDit)
        && forced_offload)
        || large_flux_bf16_should_auto_offload(paths, hint);
    let qwen_family = hint.is_some_and(|h| h.family == ActivationFamily::QwenImageDit);
    let qwen_quantized = qwen_family
        && paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    let peak = base_peak_memory_for_paths(
        paths,
        hint,
        streaming,
        flux_offload,
        qwen_quantized,
        gemma_competes,
    );
    // Add the per-request activation budget on top of the file-size peak.
    // The 2 GB `MEMORY_BUDGET_HEADROOM` already inside `estimate_peak_memory`
    // is a generic "kernels + small state" constant that doesn't scale; the
    // hint is the resolution/dtype/arch-aware delta on top.
    let activation = activation_memory_for_estimate(hint, qwen_quantized);
    let peak_with_activation = peak.saturating_add(activation);
    let effective_available = available_bytes.saturating_add(active_vram_bytes);
    // Qwen-Image runs phase-sequential on BOTH runtimes — GGUF and BF16 drop
    // the text encoder before the transformer loads (encode → drop TE →
    // denoise → VAE) — so the flat 90% cap double-penalizes a peak estimate
    // that already carries 2 GB of headroom plus the activation budget.
    // Accept whenever the estimated peak simply fits in free VRAM (a 41 GB
    // BF16 qwen on a 46 GB card was rejected with ~5 GB of real slack).
    if qwen_family && peak_with_activation <= effective_available {
        return Ok(());
    }
    let suggestion = rejection_suggestion(hint);

    check_model_memory_budget(
        model_name,
        peak_with_activation,
        effective_available,
        suggestion,
    )
}

fn base_peak_memory_for_paths(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    streaming: bool,
    flux_offload: bool,
    qwen_quantized: bool,
    gemma_competes: bool,
) -> u64 {
    if streaming {
        // LTX-2 also pays for a Gemma 3 12B prompt encoder. Auto placement
        // may try GPU first, but the runtime catches prompt-encoder CUDA OOMs
        // and retries on CPU before loading the streamed transformer. Preflight
        // must not reject that recoverable path. Only an explicit same-GPU pin
        // (`MOLD_LTX2_GEMMA_DEVICE=gpu`) is counted against this GPU because
        // the runtime will surface that OOM instead of rewriting the request.
        return streaming_transformer_peak(paths, gemma_competes);
    } else if flux_offload {
        return streaming_transformer_peak(paths, false);
    } else if hint.is_some_and(|h| h.family == ActivationFamily::Sd3Mmdit) {
        return sd3_sequential_peak(paths);
    } else if qwen_quantized {
        return qwen_image_quantized_sequential_peak(paths, hint);
    }

    mold_inference::device::estimate_peak_memory(paths, mold_inference::LoadStrategy::Sequential)
}

fn activation_memory_for_estimate(hint: Option<ActivationHint>, qwen_quantized: bool) -> u64 {
    if qwen_quantized {
        0
    } else {
        hint.map(|h| h.budget_bytes()).unwrap_or(0)
    }
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
fn streaming_transformer_peak(
    paths: &ModelPaths,
    gemma_competes_with_transformer_gpu: bool,
) -> u64 {
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
    let encoder_total = if gemma_competes_with_transformer_gpu {
        t5_size + clip_size + clip2_size + text_encoder_size
    } else {
        0
    };

    let inference_phase = STREAMING_TRANSFORMER_CAP;
    std::cmp::max(encoder_total, inference_phase) + HEADROOM
}

/// Peak GPU residency for SD3's staged sequential runtime. SD3 loads the
/// triple text encoder, drops it, optionally VAE-encodes the source image,
/// drops VAE, loads MMDiT for denoise, drops it, then loads VAE again for
/// decode. GGUF SD3 models use the monolithic Stability safetensors file as
/// the VAE source, but only VAE tensors are materialized by the runtime.
fn sd3_sequential_peak(paths: &ModelPaths) -> u64 {
    const SD3_VAE_RESIDENCY_CAP: u64 = 1_000_000_000; // VAE portion is ~300 MB; keep slack.
    const HEADROOM: u64 = 2_000_000_000; // mirrors device::MEMORY_BUDGET_HEADROOM

    let file_size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let transformer_size = if !paths.transformer_shards.is_empty() {
        paths.transformer_shards.iter().map(|p| file_size(p)).sum()
    } else {
        file_size(&paths.transformer)
    };
    let vae_size = file_size(&paths.vae).min(SD3_VAE_RESIDENCY_CAP);
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
    let encoder_total = t5_size + clip_size + clip2_size + text_encoder_size;

    transformer_size.max(vae_size).max(encoder_total) + HEADROOM
}

/// Peak GPU residency for Qwen-Image GGUF under its low-memory sequential
/// runtime. The quantized CUDA path disables CFG batching under pressure, so
/// the transformer phase is the quantized transformer plus a single-forward
/// activation reserve. Text encoder and VAE run in separate phases.
fn qwen_image_quantized_sequential_peak(paths: &ModelPaths, hint: Option<ActivationHint>) -> u64 {
    const QWEN_GGUF_PHASE_HEADROOM: u64 = 128_000_000;

    let file_size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let transformer_size = if !paths.transformer_shards.is_empty() {
        paths.transformer_shards.iter().map(|p| file_size(p)).sum()
    } else {
        file_size(&paths.transformer)
    };
    let text_encoder_size: u64 = paths.text_encoder_files.iter().map(|p| file_size(p)).sum();
    let vae_size = file_size(&paths.vae);
    let activation = hint
        .map(|h| {
            mold_inference::device::activation_bytes(
                h.width,
                h.height,
                1,
                h.dtype_bytes,
                ActivationFamily::QwenImageDit,
            )
        })
        .unwrap_or(0);

    transformer_size
        .saturating_add(activation)
        .saturating_add(QWEN_GGUF_PHASE_HEADROOM)
        .max(text_encoder_size)
        .max(vae_size)
}

/// Whether preflight should count the LTX-2 Gemma prompt encoder against the
/// transformer's GPU budget. Auto placement can recover from CUDA OOM by
/// retrying the prompt path on CPU; explicit same-GPU placement cannot.
pub(crate) fn ltx2_encoder_phase_competes_with_transformer_gpu(gpu_ordinal: usize) -> bool {
    ltx2_encoder_phase_competes_with_transformer_gpu_from_values(
        mold_inference::runtime_env::value("MOLD_LTX2_GEMMA_DEVICE").as_deref(),
        mold_inference::runtime_env::value("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER").as_deref(),
        gpu_ordinal,
    )
}

fn ltx2_encoder_phase_competes_with_transformer_gpu_from_values(
    primary: Option<&str>,
    legacy_force_cpu: Option<&str>,
    gpu_ordinal: usize,
) -> bool {
    matches!(
        mold_inference::device::resolve_ltx2_gemma_device_override_from_values(
            primary,
            legacy_force_cpu,
            gpu_ordinal,
        ),
        Some(mold_inference::device::LtxGemmaPlacement::Gpu(ordinal)) if ordinal == gpu_ordinal
    )
}

/// Check whether estimated peak memory fits before committing to a model load.
///
/// CUDA uses the current reserve-adjusted free reading plus only the active
/// model footprint that the caller is about to drop. Driver workspaces,
/// allocator fragmentation, retained live handles, and external allocations
/// are deliberately not promoted back to total capacity.
///
/// Callers perform a second guard with an actual post-drop sample before
/// allocating the replacement model. The first pass preserves the old model
/// when the request is obviously infeasible; the second catches an optimistic
/// recorded footprint or unrecovered "ghost" VRAM.
///
/// On macOS (unified memory) the same additive `available + active_vram`
/// budget applies because tensors freed during `unload()` return to the
/// system page cache.
/// On other platforms with no memory query available, the guard is a no-op.
pub(crate) fn preflight_memory_guard(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    #[cfg(feature = "cuda")]
    {
        let effective_free = authoritative_cuda_available(
            mold_inference::device::usable_free_vram_bytes_result(gpu_ordinal),
        )?;
        preflight_memory_guard_with_available_on_gpu(
            model_name,
            paths,
            active_vram_bytes,
            effective_free,
            gpu_ordinal,
            hint,
        )
    }

    #[cfg(not(feature = "cuda"))]
    {
        // macOS unified memory: query system memory and add reclaimable footprint.
        if let Some(available) = mold_inference::device::available_system_memory_bytes() {
            if available > 0 {
                return preflight_memory_guard_with_available_on_gpu(
                    model_name,
                    paths,
                    active_vram_bytes,
                    available,
                    gpu_ordinal,
                    hint,
                );
            }
        }

        // No memory info available on this platform — skip the guard.
        Ok(())
    }
}

/// Re-check a load against the driver's actual free-memory reading after the
/// previous engine and all of its device-backed state have been dropped.
///
/// This is the authoritative swap gate. It intentionally passes no
/// reclaimable active footprint: anything the driver still reports as used is
/// unavailable pressure, regardless of whether Mold expected the drop to
/// release it.
pub(crate) fn preflight_memory_guard_after_drop(
    model_name: &str,
    paths: &ModelPaths,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    #[cfg(feature = "cuda")]
    {
        let available = authoritative_cuda_available(
            mold_inference::device::post_drop_free_vram_bytes(gpu_ordinal),
        )?;
        preflight_memory_guard_with_available_on_gpu(
            model_name,
            paths,
            0,
            available,
            gpu_ordinal,
            hint,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        // Metal's unified-memory admission already used available system
        // memory plus the active engine's reclaimable footprint in the first
        // guard. A second instantaneous sample after `unload()` can lag page
        // reclamation and falsely reject a swap.
        let _ = (model_name, paths, gpu_ordinal, hint);
        Ok(())
    }
}

/// Effective memory budget to use when deciding whether a server engine can
/// stay eager-loaded or should degrade to load-use-drop sequential mode.
///
/// This mirrors the budget shape in [`preflight_memory_guard`]: current free
/// memory plus the explicitly tracked active footprint that will be dropped.
pub(crate) fn effective_load_available_bytes(
    active_vram_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
) -> Result<Option<u64>, ApiError> {
    #[cfg(feature = "cuda")]
    {
        let free = authoritative_cuda_available(
            mold_inference::device::usable_free_vram_bytes_result(gpu_ordinal),
        )?;
        Ok(Some(free.saturating_add(active_vram_bytes)))
    }

    #[cfg(not(feature = "cuda"))]
    Ok(mold_inference::device::available_system_memory_bytes()
        .filter(|available| *available > 0)
        .map(|available| available.saturating_add(active_vram_bytes)))
}

#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
fn authoritative_cuda_available(
    sample: Result<u64, mold_inference::device::DeviceMemoryError>,
) -> Result<u64, ApiError> {
    sample.map_err(|error| {
        if error.is_fatal_cuda() {
            ApiError::internal(error.to_string())
        } else {
            ApiError::insufficient_memory(format!(
                "GPU memory admission blocked because current free VRAM could not be measured: {error}"
            ))
        }
    })
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
    let transformer_is_gguf = transformer_path_is_gguf(paths);

    if hint.is_some_and(|h| h.family == ActivationFamily::ZImageDit) && !transformer_is_gguf {
        return mold_inference::LoadStrategy::Sequential;
    }
    if transformer_is_gguf
        && hint.is_some_and(|h| {
            matches!(
                h.family,
                ActivationFamily::Sd3Mmdit | ActivationFamily::ZImageDit
            )
        })
    {
        return mold_inference::LoadStrategy::Eager;
    }
    let qwen_quantized =
        hint.is_some_and(|h| h.family == ActivationFamily::QwenImageDit) && transformer_is_gguf;

    let Some(available_bytes) = available_bytes.filter(|v| *v > 0) else {
        return mold_inference::LoadStrategy::Eager;
    };

    if qwen_quantized {
        let peak = qwen_image_quantized_sequential_peak(paths, hint);
        if peak <= available_bytes {
            return mold_inference::LoadStrategy::Sequential;
        }
    }

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

    // Paired with the qwen_family admission bypass in the preflight guard:
    // a Qwen-Image load admitted because its phase-sequential peak fits FREE
    // VRAM (100%, not 90%) must actually load Sequential — Eager co-resides
    // transformer + text encoder + VAE, which is exactly what the admission
    // assumed would NOT happen. Without this branch, a BF16 qwen in the
    // 90–100%-of-free band was admitted and then handed the Eager strategy.
    let qwen_family = hint.is_some_and(|h| h.family == ActivationFamily::QwenImageDit);
    if qwen_family && eager_peak > hard_limit && sequential_peak <= available_bytes {
        return mold_inference::LoadStrategy::Sequential;
    }

    if eager_peak > hard_limit && sequential_peak <= hard_limit {
        mold_inference::LoadStrategy::Sequential
    } else {
        mold_inference::LoadStrategy::Eager
    }
}

pub(crate) fn select_server_load_strategy_for_device(
    paths: &ModelPaths,
    available_bytes: Option<u64>,
    device_total_bytes: Option<u64>,
    hint: Option<ActivationHint>,
) -> mold_inference::LoadStrategy {
    let capped_available = match (
        available_bytes.filter(|available| *available > 0),
        device_total_bytes.filter(|total| *total > 0),
    ) {
        (Some(available), Some(total)) => Some(available.min(total)),
        (available, None) => available,
        (None, Some(_)) => None,
    };

    select_server_load_strategy_for_budget(paths, capped_available, hint)
}

pub(crate) fn server_offload_enabled_for_paths_with_request(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
    forced_offload: bool,
) -> bool {
    let transformer_path = transformer_path_lower(paths);
    let transformer_looks_flux2 = transformer_path_looks_flux2(&transformer_path);
    let transformer_looks_zimage = transformer_path_looks_zimage(&transformer_path);
    let transformer_looks_nvfp4 = transformer_path.contains("nvfp4");

    if request_has_lora
        && (transformer_looks_flux2
            || transformer_looks_zimage
            || hint.is_some_and(|h| {
                matches!(
                    h.family,
                    ActivationFamily::Flux2Dit | ActivationFamily::ZImageDit
                )
            }))
    {
        return false;
    }

    let transformer_is_gguf = transformer_path_is_gguf(paths);

    if transformer_looks_nvfp4
        && (transformer_looks_flux2 || hint.is_some_and(|h| h.family == ActivationFamily::Flux2Dit))
    {
        return false;
    }

    if transformer_is_gguf
        && hint.is_some_and(|h| {
            matches!(
                h.family,
                ActivationFamily::Sd3Mmdit
                    | ActivationFamily::ZImageDit
                    | ActivationFamily::Flux2Dit
            )
        })
    {
        return false;
    }

    forced_offload || large_flux_bf16_should_auto_offload(paths, hint)
}

pub(crate) fn server_offload_enabled_for_paths(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
) -> bool {
    let forced_offload = matches!(
        mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
        Some("1") | Some("true") | Some("yes")
    );
    server_offload_enabled_for_paths_with_request(paths, hint, request_has_lora, forced_offload)
}

pub(crate) fn request_requires_fresh_engine_for_offload_policy(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
) -> bool {
    let forced_offload = matches!(
        mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
        Some("1") | Some("true") | Some("yes")
    );
    request_requires_fresh_engine_for_offload_policy_with_request(
        paths,
        hint,
        request_has_lora,
        forced_offload,
    )
}

pub(crate) fn request_requires_fresh_engine_for_offload_policy_with_request(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
    forced_offload: bool,
) -> bool {
    request_has_lora
        && server_offload_enabled_for_paths_with_request(paths, hint, false, forced_offload)
        && !server_offload_enabled_for_paths_with_request(paths, hint, true, forced_offload)
}

pub(crate) struct GenerationMemoryBudget {
    pub(crate) peak_memory_bytes: u64,
    pub(crate) activation_memory_bytes: u64,
    pub(crate) available_memory_bytes: Option<u64>,
    pub(crate) load_strategy: mold_inference::LoadStrategy,
    pub(crate) block_offload: bool,
    pub(crate) under_memory_pressure: bool,
    pub(crate) fits_available_memory: Option<bool>,
}

/// Resolve one generation's memory/load policy against an explicit sampled
/// free-memory budget.
///
/// This is deliberately pure: scheduler candidates pass their own
/// `DeviceFact::available_vram_bytes`, while legacy diagnostics may pass
/// `None` when no authoritative sample exists. It never queries device zero
/// and never substitutes total VRAM for missing free VRAM.
pub(crate) fn estimate_generation_memory_for_request(
    req: &GenerateRequest,
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    available_memory_bytes: Option<u64>,
    forced_offload: bool,
    request_has_lora: bool,
    gemma_competes: bool,
) -> GenerationMemoryBudget {
    let transformer_path = transformer_path_lower(paths);
    let streaming = hint
        .map(|h| h.family.streaming_transformer())
        .unwrap_or_else(|| transformer_path_looks_ltx2(&transformer_path));
    let block_offload = server_offload_enabled_for_paths_with_request(
        paths,
        hint,
        request_has_lora,
        forced_offload,
    );
    let flux_offload = hint.is_some_and(|h| h.family == ActivationFamily::FluxDit) && block_offload;
    let qwen_quantized = hint.is_some_and(|h| h.family == ActivationFamily::QwenImageDit)
        && transformer_path_is_gguf(paths);
    let base_peak = base_peak_memory_for_paths(
        paths,
        hint,
        streaming,
        flux_offload,
        qwen_quantized,
        gemma_competes,
    );
    let activation = request_sensitive_activation_memory(req, hint, qwen_quantized);
    let peak = base_peak.saturating_add(activation);
    let available_memory_bytes = available_memory_bytes.filter(|available| *available > 0);
    let load_strategy = select_server_load_strategy_for_budget(paths, available_memory_bytes, hint);
    let eager_peak =
        mold_inference::device::estimate_peak_memory(paths, mold_inference::LoadStrategy::Eager)
            .saturating_add(activation);
    let under_memory_pressure = available_memory_bytes
        .is_some_and(|available| eager_peak > available.saturating_mul(9) / 10);
    let qwen_family = hint.is_some_and(|h| h.family == ActivationFamily::QwenImageDit);
    let fits_available_memory = available_memory_bytes.map(|available| {
        if qwen_family {
            peak <= available
        } else {
            peak <= available.saturating_mul(9) / 10
        }
    });

    GenerationMemoryBudget {
        peak_memory_bytes: peak,
        activation_memory_bytes: activation,
        available_memory_bytes,
        load_strategy,
        block_offload,
        under_memory_pressure,
        fits_available_memory,
    }
}

fn request_sensitive_activation_memory(
    req: &GenerateRequest,
    hint: Option<ActivationHint>,
    qwen_quantized: bool,
) -> u64 {
    let base = activation_memory_for_estimate(hint, qwen_quantized);
    let batch = u64::from(req.batch_size.max(1));
    let video_frames = u64::from(req.frames.unwrap_or(1).max(1));
    let video_factor = if hint.is_some_and(|h| h.family.streaming_transformer()) {
        // Video runtimes denoise multiple latent frames but do not keep every
        // frame's full activation workspace resident at once. Scale
        // sublinearly so longer clips still move the estimate without
        // turning it into a file-size guess.
        video_frames.div_ceil(25).max(1)
    } else {
        1
    };
    let cfg_factor = if req.guidance > 1.0 && req.negative_prompt.is_some() {
        2
    } else {
        1
    };

    let mut activation = base
        .saturating_mul(batch)
        .saturating_mul(video_factor)
        .saturating_mul(cfg_factor);

    let pixel_bytes = u64::from(req.width)
        .saturating_mul(u64::from(req.height))
        .saturating_mul(4);
    if req.source_image.is_some()
        || req
            .edit_images
            .as_ref()
            .is_some_and(|images| !images.is_empty())
    {
        activation = activation.saturating_add(pixel_bytes.saturating_mul(batch));
    }
    if req.mask_image.is_some() {
        activation = activation.saturating_add(pixel_bytes / 2);
    }
    if req.control_image.is_some() || req.control_model.as_deref().is_some_and(|m| !m.is_empty()) {
        activation = activation.saturating_add(pixel_bytes.saturating_mul(2));
    }
    if req.upscale_model.as_deref().is_some_and(|m| !m.is_empty()) {
        activation = activation.saturating_add(pixel_bytes.saturating_mul(4));
    }
    let lora_count = req
        .loras
        .as_ref()
        .map(|loras| loras.len())
        .unwrap_or_else(|| usize::from(req.lora.is_some())) as u64;
    activation.saturating_add(lora_count.saturating_mul(128 * 1024 * 1024))
}

#[cfg(test)]
mod fail_closed_tests {
    use super::*;

    #[test]
    fn explicit_gemma_gpu_policy_tracks_the_assigned_worker_ordinal() {
        assert!(ltx2_encoder_phase_competes_with_transformer_gpu_from_values(Some("gpu"), None, 1));
        assert!(ltx2_encoder_phase_competes_with_transformer_gpu_from_values(Some("gpu"), None, 7));
        assert!(
            !ltx2_encoder_phase_competes_with_transformer_gpu_from_values(Some("cpu"), None, 1)
        );
    }

    #[test]
    fn unavailable_cuda_sample_blocks_admission_with_typed_api_error() {
        let error = authoritative_cuda_available(Err(
            mold_inference::device::DeviceMemoryError::Unavailable {
                operation: "free VRAM query",
                message: "injected unavailable sample".to_string(),
            },
        ))
        .unwrap_err();

        assert_eq!(error.code, "INSUFFICIENT_MEMORY");
        assert!(
            error.error.contains("admission blocked"),
            "got: {}",
            error.error
        );
    }

    #[test]
    fn fatal_cuda_sample_is_not_downgraded_to_memory_pressure() {
        let error = authoritative_cuda_available(Err(
            mold_inference::device::DeviceMemoryError::FatalCuda {
                operation: "device synchronize",
                message: "CUDA_ERROR_ILLEGAL_ADDRESS".to_string(),
            },
        ))
        .unwrap_err();

        assert_eq!(error.code, "INTERNAL_ERROR");
        assert!(error.error.contains("fatal CUDA error"));
    }
}
