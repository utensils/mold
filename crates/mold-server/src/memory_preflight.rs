use std::path::Path;

use mold_core::{GenerateRequest, ModelPaths};
use mold_inference::device::{activation_bytes, activation_family_for, ActivationFamily};
use mold_inference::engine::cfg_active;

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

/// Whether a large BF16 FLUX.1 checkpoint takes the block-streaming path.
///
/// Mirrors [`large_flux2_bf16_should_auto_offload`] exactly, including its
/// two-step shape: `available_bytes: None` answers the size question alone —
/// today's answer, byte for byte — and `Some(available)` narrows it by asking
/// whether the checkpoint plus this render's workspace actually exceeds 90 %
/// of the card. Without the second step a 23.8 GB BF16 dev streams its blocks
/// on a 46 GB L40S that has room for the whole thing, at the documented 3–5x
/// penalty, for nothing.
fn large_flux_bf16_should_auto_offload(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    available_bytes: Option<u64>,
    activation_bytes: u64,
) -> bool {
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

    if transformer_component_size(paths) < LARGE_FLUX_BF16_TRANSFORMER_BYTES {
        return false;
    }

    available_bytes
        .filter(|bytes| *bytes > 0)
        .is_none_or(|available| {
            let resident_peak = mold_inference::device::estimate_peak_memory(
                paths,
                mold_inference::LoadStrategy::Sequential,
            )
            .saturating_add(activation_bytes);
            resident_peak > available.saturating_mul(9) / 10
        })
}

/// Extra resident bytes an fp8 FLUX.2 transformer costs when the engine
/// widens it once at load.
///
/// Name-gated the way the `nvfp4` arm beside it is: an estimate runs on the
/// coordinator and must not read a 9 GB header to answer. `None` availability
/// means the widen gate cannot be resolved, and an unresolvable gate charges
/// nothing — today's answer, byte for byte.
fn flux2_fp8_widen_extra_bytes(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    available_bytes: Option<u64>,
) -> u64 {
    let transformer_path = transformer_path_lower(paths);
    let is_flux2 = transformer_path_looks_flux2(&transformer_path)
        || hint.is_some_and(|hint| hint.family == ActivationFamily::Flux2Dit);
    if !is_flux2 || !transformer_path.contains("fp8") || transformer_path_is_gguf(paths) {
        return 0;
    }
    let Some(available) = available_bytes.filter(|bytes| *bytes > 0) else {
        return 0;
    };
    // The checkpoint, not its file length: the engine sizes the gate from the
    // resolved `Flux2Config`, and a `fp8mixed` file is materially larger than
    // its parameter count. One function answers for both.
    //
    // The model name is the path's stem here rather than the request's model
    // id, because that is what this estimator has — and it feeds only the
    // fallback arm of the resolver, which the header probe pre-empts for
    // every safetensors checkpoint this gate accepts.
    let model_name = paths
        .transformer
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    mold_inference::flux2_fp8_widen_extra_resident_bytes_for_checkpoint(
        &paths.transformer,
        model_name,
        available,
    )
}

/// The FLUX.2 denoise geometry this checkpoint will run at, or `None` when the
/// build recognises no variant in it and the estimate keeps its previous
/// pixel-area answer.
///
/// Resolved through the engine's own `Flux2Config` probe, beside
/// `flux2_fp8_widen_extra_bytes`, which resolves the widen gate the same way
/// and for the same reason.
fn flux2_activation_geometry(
    model_name: &str,
    paths: &ModelPaths,
) -> Option<mold_inference::device::Flux2ActivationGeometry> {
    mold_inference::flux2_activation_geometry_for_checkpoint(&paths.transformer, model_name)
}

/// Why this FLUX.2 checkpoint cannot stream its blocks, or `None` when it can.
///
/// A thin pass-through to `mold_inference::flux2_block_offload_unsupported_reason`
/// so admission, the offload gate and the refusal text all read the engine's
/// own decision. Non-FLUX.2 callers never reach it.
pub(crate) fn flux2_block_offload_unsupported_reason(
    paths: &ModelPaths,
    request_has_lora: bool,
) -> Option<&'static str> {
    mold_inference::flux2_block_offload_unsupported_reason(&paths.transformer, request_has_lora)
}

fn large_flux2_bf16_should_auto_offload(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    available_bytes: Option<u64>,
    activation_bytes: u64,
) -> bool {
    const LARGE_FLUX2_BF16_TRANSFORMER_BYTES: u64 = 20_000_000_000;
    let eligible = hint.is_some_and(|h| h.family == ActivationFamily::Flux2Dit)
        && !transformer_path_is_gguf(paths)
        && !transformer_path_lower(paths).contains("nvfp4")
        && transformer_component_size(paths) >= LARGE_FLUX2_BF16_TRANSFORMER_BYTES;
    if !eligible {
        return false;
    }

    available_bytes
        .filter(|bytes| *bytes > 0)
        .is_none_or(|available| {
            let resident_peak = mold_inference::device::estimate_peak_memory(
                paths,
                mold_inference::LoadStrategy::Sequential,
            )
            .saturating_add(activation_bytes);
            resident_peak > available.saturating_mul(9) / 10
        })
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
        // CFG-doubled forwards: SDXL/SD3 batch=2 only while CFG is active;
        // FLUX, Z-Image, and Flux.2 run a single forward.
        let family = activation_family_for(family_slug);
        let batch = match family {
            ActivationFamily::SdxlUnet | ActivationFamily::Sd3Mmdit if cfg_active(req.guidance) => {
                2
            }
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

/// Identifies a dispatch-time rejection of an already-admitted plan.
///
/// Both planned-budget rechecks end with this sentence, and the worker's
/// failure classifier keys on it rather than on prose so that rewording a
/// message cannot silently start counting memory pressure as worker ill
/// health.
pub(crate) const ADMISSION_PRESSURE_MARKER: &str =
    "memory pressure changed after scheduler admission";

/// Revalidate one already-admitted execution plan against a fresh physical
/// memory sample without replacing the plan's memory model at dispatch.
///
/// Scheduler admission has already applied the family-specific safety policy
/// (including the ordinary 90% cap) to `predicted_peak_bytes`. The worker's
/// post-grant responsibility is narrower: fail closed if new external or
/// unrecovered pressure means that exact frozen peak no longer physically
/// fits. Reapplying the legacy path-based estimator here can silently erase
/// CPU placement or block-offload authority and reject a plan that the worker
/// is required to execute unchanged.
pub(crate) fn check_planned_memory_budget(
    model_name: &str,
    predicted_peak_bytes: u64,
    available_bytes: u64,
    physical_capacity_bytes: Option<u64>,
    suggestion: &str,
) -> Result<(), ApiError> {
    if predicted_peak_bytes > available_bytes {
        // Naming a cause is only useful if it is the cause. When the peak is
        // above the DEVICE'S OWN capacity, nothing was contended and nothing
        // will be released — the 24 GB simulation in the 2026-09-11 audit read
        // "memory pressure changed after scheduler admission" on a card with
        // nothing else on it at all. Say what is true: this shape does not fit
        // this device.
        if physical_capacity_bytes.is_some_and(|capacity| predicted_peak_bytes > capacity) {
            return Err(ApiError::insufficient_memory(format!(
                "model '{}' execution plan peak ~{:.1} GB exceeds this GPU's ~{:.1} GB usable \
                 capacity, so no amount of waiting will make it fit. {}",
                model_name,
                predicted_peak_bytes as f64 / 1_000_000_000.0,
                available_bytes as f64 / 1_000_000_000.0,
                suggestion,
            )));
        }
        return Err(ApiError::insufficient_memory(format!(
            "model '{}' frozen execution plan peak ~{:.1} GB no longer fits the current ~{:.1} GB \
             physical memory budget; {}. Retry after other GPU work releases memory. {}",
            model_name,
            predicted_peak_bytes as f64 / 1_000_000_000.0,
            available_bytes as f64 / 1_000_000_000.0,
            ADMISSION_PRESSURE_MARKER,
            suggestion,
        )));
    }

    let warn_limit = available_bytes * 8 / 10;
    if predicted_peak_bytes > warn_limit {
        tracing::warn!(
            model = %model_name,
            planned_peak_gb = format_args!("{:.1}", predicted_peak_bytes as f64 / 1_000_000_000.0),
            available_gb = format_args!("{:.1}", available_bytes as f64 / 1_000_000_000.0),
            "admitted execution plan is close to the current physical memory limit"
        );
    }

    Ok(())
}

/// Revalidate an already-admitted plan's host-RAM increment against the same
/// ledger that granted its lease.
///
/// The host counterpart of [`check_planned_memory_budget`], and it obeys the
/// same discipline: it rechecks the exact frozen increment and never recomputes
/// a demand of its own. `available_host_headroom_bytes` must come from
/// `HostMemoryLedger`, not from an independent floor computation — admission
/// and dispatch disagreeing is what oscillates work through the dispatch-replan
/// budget. Callers with no ledger evidence retain the grant instead of calling
/// this with a guessed headroom.
///
/// `reclaimable_zfs_arc_bytes` is the evictable ZFS ARC the SAME ledger sample
/// counted into that headroom (#1439); a positive credit is named in the
/// refusal so the figure a user reads already includes it.
pub(crate) fn check_planned_host_budget(
    model_name: &str,
    predicted_host_increment_bytes: u64,
    available_host_headroom_bytes: u64,
    reclaimable_zfs_arc_bytes: Option<u64>,
) -> Result<(), ApiError> {
    if predicted_host_increment_bytes > available_host_headroom_bytes {
        let clause = match reclaimable_zfs_arc_bytes {
            Some(credit) if credit > 0 => format!(
                " (including ~{:.1} GB evictable ZFS ARC)",
                credit as f64 / 1_000_000_000.0
            ),
            _ => String::new(),
        };
        return Err(ApiError::insufficient_memory(format!(
            "model '{model_name}' frozen host-memory increment ~{:.1} GB no longer fits the current ~{:.1} GB host-memory headroom after the canonical safety floor{clause}; {ADMISSION_PRESSURE_MARKER}",
            predicted_host_increment_bytes as f64 / 1_000_000_000.0,
            available_host_headroom_bytes as f64 / 1_000_000_000.0,
        )));
    }
    Ok(())
}

fn check_planned_memory_budget_with_resident(
    model_name: &str,
    predicted_peak_bytes: u64,
    free_bytes: u64,
    resident_vram_bytes: u64,
    physical_capacity_bytes: Option<u64>,
    suggestion: &str,
) -> Result<(), ApiError> {
    check_planned_memory_budget(
        model_name,
        predicted_peak_bytes,
        free_bytes.saturating_add(resident_vram_bytes),
        physical_capacity_bytes,
        suggestion,
    )
}

/// Build the suggestion text appended to preflight rejection messages.
/// For LTX-Video (non-streaming full-weight load) the dominant knob is
/// reducing `frames` or `width`/`height`; for image families, resolution and
/// batch size are usually the first levers because activation and VAE
/// workspace can dominate the checkpoint size.
pub(crate) fn rejection_suggestion(hint: Option<ActivationHint>) -> &'static str {
    match hint.map(|h| h.family) {
        // Every full-weight video family, not just LTX-Video. Wan reaching the
        // generic arm was actively misleading: it recommends `--batch` (wan
        // renders one clip at a time regardless) and `--offload` (a FLUX flag
        // with no wan path), while omitting `--frames`, which is the single
        // most effective lever because the transformer's activation cost
        // scales with the token count and tokens scale with frames.
        Some(family) if family.is_full_weight_video() => {
            "Try reducing --frames or --width/--height, use a quantized variant \
             (e.g. ':q8'), or close other GPU apps."
        }
        _ => {
            "Try lowering --width/--height, reduce --batch, use a smaller/quantized \
             variant if available, enable --offload for FLUX, or close other GPU apps."
        }
    }
}

/// [`rejection_suggestion`] with the resolved model name in hand, so a video
/// rejection can name a tier the user is not already on.
///
/// `(e.g. ':q8')` is circular advice for someone whose request just failed
/// *on* `:q8` — the next lever down is `:q5`, and below that the shape itself.
pub(crate) fn rejection_suggestion_for_model(
    hint: Option<ActivationHint>,
    model_name: &str,
) -> String {
    let video = hint.is_some_and(|h| h.family.is_full_weight_video());
    if !video {
        return rejection_suggestion(hint).to_string();
    }
    // Only name a tier when the model's own tag proves the ladder exists.
    //
    // An untagged name, an opaque `cv:`/`hf:` id, or a family that ships no
    // quantized tier at all must not be told to try `:q8`: every shipped
    // LTX-Video checkpoint is `:bf16`-only, so naming a quantization there
    // sends the user after a variant that does not exist.
    let next_tier = match model_name.rsplit_once(':').map(|(_, tag)| tag) {
        Some("q8") => Some("':q5'"),
        Some("q5") => Some("':q4'"),
        // Already the smallest tier that ships: the levers left are the shape
        // and the card.
        Some("q4") => {
            return "Try reducing --frames or --width/--height, or close other GPU apps — \
                    this is already the smallest quantized tier."
                .to_string();
        }
        _ => None,
    };
    match next_tier {
        Some(next) => format!(
            "Try reducing --frames or --width/--height, use a smaller quantized \
             variant (e.g. {next}), or close other GPU apps."
        ),
        None => "Try reducing --frames or --width/--height, use a quantized variant if \
                 one is published for this model, or close other GPU apps."
            .to_string(),
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

#[cfg(test)]
pub(crate) fn preflight_memory_guard_with_available_on_gpu(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
    gpu_ordinal: usize,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    preflight_memory_guard_with_available_on_gpu_for_request(
        model_name,
        paths,
        active_vram_bytes,
        available_bytes,
        gpu_ordinal,
        hint,
        false,
    )
}

fn preflight_memory_guard_with_available_on_gpu_for_request(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
    gpu_ordinal: usize,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
) -> Result<(), ApiError> {
    let forced_offload = matches!(
        mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
        Some("1") | Some("true") | Some("yes")
    );
    let gemma_competes = ltx2_encoder_phase_competes_with_transformer_gpu(gpu_ordinal);
    preflight_memory_guard_with_available_and_policy_for_request(
        model_name,
        paths,
        active_vram_bytes,
        available_bytes,
        hint,
        LegacyPreflightPolicy {
            forced_offload,
            gemma_competes,
            request_has_lora,
        },
    )
}

#[cfg(test)]
pub(crate) fn preflight_memory_guard_with_available_and_policy(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
    hint: Option<ActivationHint>,
    forced_offload: bool,
    gemma_competes: bool,
) -> Result<(), ApiError> {
    preflight_memory_guard_with_available_and_policy_for_request(
        model_name,
        paths,
        active_vram_bytes,
        available_bytes,
        hint,
        LegacyPreflightPolicy {
            forced_offload,
            gemma_competes,
            request_has_lora: false,
        },
    )
}

#[derive(Clone, Copy)]
struct LegacyPreflightPolicy {
    forced_offload: bool,
    gemma_competes: bool,
    request_has_lora: bool,
}

fn preflight_memory_guard_with_available_and_policy_for_request(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    available_bytes: u64,
    hint: Option<ActivationHint>,
    policy: LegacyPreflightPolicy,
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
    let effective_available = available_bytes.saturating_add(active_vram_bytes);
    let conservative_flux_offload = server_offload_enabled_for_paths_with_request(
        paths,
        hint,
        policy.request_has_lora,
        policy.forced_offload,
    );
    // Both families' size predicates are narrowed by the measured card in the
    // same shape: the conservative answer above is a size question, and a card
    // with room for the whole checkpoint must not stream its blocks.
    let flux_offload = if conservative_flux_offload && !policy.forced_offload {
        let activation = activation_memory_for_estimate(hint, false);
        if large_flux2_bf16_should_auto_offload(paths, hint, None, 0) {
            large_flux2_bf16_should_auto_offload(paths, hint, Some(effective_available), activation)
        } else if large_flux_bf16_should_auto_offload(paths, hint, None, 0) {
            large_flux_bf16_should_auto_offload(paths, hint, Some(effective_available), activation)
        } else {
            conservative_flux_offload
        }
    } else {
        conservative_flux_offload
    };
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
        policy.gemma_competes,
        streamed_text_encoder_device_charge(model_name, paths),
    );
    // Add the per-request activation budget on top of the file-size peak.
    // The 2 GB `MEMORY_BUDGET_HEADROOM` already inside `estimate_peak_memory`
    // is a generic "kernels + small state" constant that doesn't scale; the
    // hint is the resolution/dtype/arch-aware delta on top.
    let activation = activation_memory_for_estimate(hint, qwen_quantized);
    let peak_with_activation = peak.saturating_add(activation);
    // Qwen-Image runs phase-sequential on BOTH runtimes — GGUF and BF16 drop
    // the text encoder before the transformer loads (encode → drop TE →
    // denoise → VAE) — so the flat 90% cap double-penalizes a peak estimate
    // that already carries 2 GB of headroom plus the activation budget.
    // Accept whenever the estimated peak simply fits in free VRAM (a 41 GB
    // BF16 qwen on a 46 GB card was rejected with ~5 GB of real slack).
    if qwen_family && peak_with_activation <= effective_available {
        return Ok(());
    }
    let suggestion = rejection_suggestion_for_model(hint, model_name);

    check_model_memory_budget(
        model_name,
        peak_with_activation,
        effective_available,
        &suggestion,
    )
}

/// The device charge for a FLUX.2 [dev] Mistral3 encoder, or `None` when the
/// model's encoder phase is priced from its files.
///
/// One derivation, read by both the generic peak and the eager peak, because a
/// planner that agreed with itself on only one of the two would still declare
/// memory pressure and park the encoder on the CPU — the 78.8 s encode the
/// 2026-09-11 audit measured with the GPU at 0 % SM.
///
/// BF16 is the dtype FLUX.2 runs a GPU-placed encoder at
/// (`flux2/pipeline.rs`'s `gpu_dtype`); F16 would price identically.
fn streamed_text_encoder_device_charge(model_name: &str, paths: &ModelPaths) -> Option<u64> {
    mold_inference::flux2::text_encoder_residency::mistral3_admission_charge_for_gpu(
        model_name, paths,
    )
    .map(|charge| charge.device_peak)
}

fn base_peak_memory_for_paths(
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    streaming: bool,
    flux_offload: bool,
    qwen_quantized: bool,
    gemma_competes: bool,
    encoder_override: Option<u64>,
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

    mold_inference::device::estimate_peak_memory_with_encoder_override(
        paths,
        mold_inference::LoadStrategy::Sequential,
        encoder_override,
    )
}

/// The `MEMORY_BUDGET_HEADROOM` baked into `estimate_peak_memory`, removed
/// from the CUDA Wan fit only.
///
/// It is a stand-in for activation and fragmentation the generic estimate does
/// not model. The request-aware wan path models both — `wan_admission`'s term
/// is fitted to whole-process peaks measured on real hardware — so charging
/// both there counts fragmentation twice, which pushed the shipped 53-frame
/// A14B default to ~26.9 GB against ~25.2 GB usable and refused a render that
/// measures 23,975 MiB and completes.
///
/// Metal retains it after #1059's live margin run showed the denoise phase
/// consuming it. Deliberately NOT subtracted inside
/// `base_peak_memory_for_paths`: that
/// function also serves the model-load guard, which has no request, prices
/// activations with the generic estimate, and still needs the headroom.
const WAN_REQUEST_AWARE_HEADROOM_BYTES: u64 = 2_000_000_000;

/// Extra allocator variation above the generic 2 GB runtime allowance for a
/// Wan denoise on Metal.
///
/// The #1059 margin run isolated the 1.3B transformer/VAE phase at
/// 8,184,725,504 allocated Metal bytes. The CUDA-calibrated request-aware
/// formula originally predicted 6.2 GB because it removed the generic 2 GB
/// allowance. Later activation calibration lowered that request estimate far
/// enough that the former 256 MiB margin fell below the retained observation.
/// Keep a further 768 MiB so admission remains above the measured peak with
/// roughly the original 256 MiB of allocator variation still available.
const WAN_METAL_ALLOCATOR_MARGIN_BYTES: u64 = 768 * 1024 * 1024;

/// Device memory a face-identity render needs beside the checkpoint it
/// conditions.
///
/// **Two** terms, not the five the whole PuLID stack weighs, because #1223
/// moved the extraction to the host at admission. What is actually resident on
/// the generation device for the duration of a denoise is:
///
/// | Term | Bytes | Where it comes from |
/// | --- | --- | --- |
/// | PuLID cross-attention adapter, 20 modules at FLUX.1's geometry, f16/bf16 | 839,270,400 | `flux::pulid::PulidAdapter::resident_bytes`, pinned below |
/// | Cross-attention activation headroom | 410,729,600 | `[1, 4096, 3072]` + `[1, 4096, 2048]` working tensors at 1024x1024, with margin |
/// | **Total** | **1,250,000,000** | |
///
/// Everything else in the stack is CPU work that has finished before the job
/// is dispatched: the SCRFD detector (17 MB) and the ArcFace recognizer
/// (261 MB) evaluate through `candle-onnx`, which materializes on
/// `Device::Cpu` and refuses anything else, and the EVA02-CLIP-L-14-336 tower
/// (609 MB derived, from the 856 MB `.pt`) plus the IDFormer run beside them.
/// All four are charged as HOST bytes by their component roles in
/// `execution_plan::ComponentRole::is_host_only`, and the measured host peak of
/// that phase is `mold_inference::identity::extraction::EXTRACTION_HOST_PEAK_BYTES`.
///
/// This replaces #1220's declared 2.3 GB placeholder, which charged the vision
/// tower and the IDFormer to VRAM on the assumption that the extractor would
/// run on the generation device. It does not, and over-charging VRAM by ~1 GB
/// parks renders a card could actually run.
///
/// The adapter term is the f16/bf16 figure, which is what every GPU render
/// loads: `PulidAdapter::load` takes the transformer's working dtype and FLUX's
/// is BF16 on CUDA and on Metal. An f32 adapter is 1,678,540,800 bytes, but the
/// only path that builds one is CPU inference, which this gate does not govern.
pub(crate) const IDENTITY_VRAM_OVERHEAD_BYTES: u64 = 1_250_000_000;

/// Device memory an SDXL face-identity render needs beside the checkpoint it
/// conditions.
///
/// The SDXL adapter is a different shape from FLUX's and is derived the same
/// way — from the checkpoint's own header, not from an analogy:
///
/// | Term | Bytes | Where it comes from |
/// | --- | --- | --- |
/// | `id_adapter_attn_layers.*`, 70 x (`id_to_k` + `id_to_v`), f16/bf16 | 681,574,400 | pinned below against `sdxl::pulid::SdxlPulidAdapter::resident_bytes` |
/// | Cross-attention activation headroom | 168,425,600 | see the arithmetic below |
/// | **Total** | **850,000,000** | |
///
/// The weight term is exact. Each of the 70 UNet cross-attentions carries two
/// bias-free `[hidden_size, 2048]` linears, and the layer table is
/// `10 x 640 + 60 x 1280` (`testdata/pulid_sdxl/attn_layer_map.json`):
/// `2 x 2048 x (10 x 640 + 60 x 1280) = 340,787,200` elements, `x 2` bytes at the
/// engine's f16/bf16 compute dtype. The checkpoint's OTHER half —
/// `id_adapter.*`, the 151,398,400-element IDFormer — is deliberately NOT in
/// this figure: it belongs to the extraction phase, which is charged by
/// [`IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES`] and has been released before a
/// single denoise step runs.
///
/// The activation term is generous by construction. The largest identity
/// branch is a 640-wide layer at 1024x1024 — `[2, 4096, 640]` under the CFG
/// batch, whose query, id-attention output, and combined result are three such
/// tensors plus a `[2, 10, 4096, 32]` score matrix and two `[2, 32, 640]`
/// projections: ~37 MB at bf16, and only one layer's worth is live at a time.
/// Charging ~168 MB leaves better than 4x for allocator caching across the 70
/// injections rather than budgeting to the arithmetic exactly.
///
/// Smaller than FLUX's 1.25 GB because the adapter is smaller and SDXL's
/// attention runs at a quarter the token count, not because anything was
/// trimmed.
pub(crate) const IDENTITY_SDXL_VRAM_OVERHEAD_BYTES: u64 = 850_000_000;

/// The adapter half of [`IDENTITY_SDXL_VRAM_OVERHEAD_BYTES`], pinned against
/// the engine's own arithmetic by
/// `sdxl_identity_overhead_matches_the_adapters_own_resident_arithmetic`. The
/// documented decomposition of the budget rather than a second input to it, so
/// only that test reads it.
#[cfg(test)]
pub(crate) const IDENTITY_SDXL_ADAPTER_BF16_BYTES: u64 = 681_574_400;

/// The device overhead one family's resident adapter costs.
///
/// One switch so the estimate cannot charge FLUX's 1.25 GB for an SDXL render
/// (which parks cards that could run it) or SDXL's 850 MB for a FLUX one
/// (which admits a render with 400 MB nowhere to go).
pub(crate) fn identity_adapter_overhead_bytes(family: mold_core::identity::IdentityFamily) -> u64 {
    match family {
        mold_core::identity::IdentityFamily::Flux => IDENTITY_VRAM_OVERHEAD_BYTES,
        mold_core::identity::IdentityFamily::Sdxl => IDENTITY_SDXL_VRAM_OVERHEAD_BYTES,
    }
}

/// Device memory an SD1.5 IP-Adapter render needs beside the checkpoint.
///
/// | Term | Bytes | Where it comes from |
/// | --- | --- | --- |
/// | 16 x (`to_k_ip` + `to_v_ip`) + `image_proj`, f16/bf16 | 44,639,232 | pinned below against `ip_adapter::IpAdapter::resident_bytes` |
/// | Cross-attention injection headroom | 45,360,768 | see the arithmetic below |
/// | **Total** | **90,000,000** | |
///
/// The weight term is exact and is the checkpoint's own inventory. Each of the
/// 16 SD1.5 cross-attentions carries two bias-free `[hidden_size, 768]`
/// linears, and the layer table is `5 x 320 + 5 x 640 + 6 x 1280 = 12,480`
/// (`testdata/pulid_sdxl/attn_layer_map_sd15.json`): `2 x 768 x 12,480 =
/// 19,169,280` elements. `image_proj` adds
/// `4 x 768 x 1024 + 4 x 768 + 2 x 768 = 3,150,336` — its
/// `Linear(clip_dim, tokens * ctx_dim)` matrix, that linear's bias, and the
/// output `LayerNorm`'s weight and bias. `x 2` bytes at the engine's f16/bf16
/// compute dtype gives 44,639,232, against a 44,642,768-byte file: the
/// difference is the safetensors header.
///
/// The activation term is generous by construction, and deliberately larger
/// than the weights it serves — which for this family is the honest ordering,
/// because a 44 MB adapter injects into activations an order of magnitude
/// bigger than itself. The largest injection is a 320-wide layer at 768x768:
/// `[2, 9216, 320]` under the CFG batch, of which the image branch's delta,
/// its transposed intermediate, and the combined result are three, plus a
/// `[2, 8, 9216, 4]` score matrix — ~35 MB at bf16, and only one layer's worth
/// is live at a time.
pub(crate) const IP_ADAPTER_SD15_VRAM_OVERHEAD_BYTES: u64 = 90_000_000;

/// The adapter half of [`IP_ADAPTER_SD15_VRAM_OVERHEAD_BYTES`], pinned against
/// the engine's own arithmetic by
/// `ip_adapter_overhead_matches_the_adapters_own_resident_arithmetic`. The
/// documented decomposition of the budget rather than a second input to it, so
/// only that test reads it.
#[cfg(test)]
pub(crate) const IP_ADAPTER_SD15_ADAPTER_BF16_BYTES: u64 = 44_639_232;

/// Device memory an SDXL IP-Adapter render needs beside the checkpoint.
///
/// | Term | Bytes | Where it comes from |
/// | --- | --- | --- |
/// | 70 x (`to_k_ip` + `to_v_ip`) + `image_proj`, f16/bf16 | 698,376,192 | pinned below against `ip_adapter::IpAdapter::resident_bytes` |
/// | Cross-attention injection headroom | 131,623,808 | see the arithmetic below |
/// | **Total** | **830,000,000** | |
///
/// Derived the same way as the SD1.5 term and from the same table, one family
/// over. Each of the 70 SDXL cross-attentions carries two bias-free
/// `[hidden_size, 2048]` linears over `10 x 640 + 60 x 1280 = 83,200`
/// (`testdata/pulid_sdxl/attn_layer_map.json`): `2 x 2048 x 83,200 =
/// 340,787,200` elements, plus `image_proj`'s
/// `4 x 2048 x 1024 + 4 x 2048 + 2 x 2048 = 8,400,896`. `x 2` bytes gives
/// 698,376,192, against a 698,391,064-byte file.
///
/// That the attention half is byte-identical to
/// `IDENTITY_SDXL_ADAPTER_BF16_BYTES` is a real coincidence, not a shared
/// derivation: PuLID's `id_to_k`/`id_to_v` are `[hidden_size, ID_TOKEN_DIM]`
/// with `ID_TOKEN_DIM = 2048`, and SDXL's `cross_attention_dim` is also 2048.
/// The two are charged from their own arithmetic precisely so a change to
/// either width cannot silently move the other.
///
/// The activation term follows the identity one's shape: the largest injection
/// is a 640-wide layer at 1024x1024 — `[2, 4096, 640]` under the CFG batch,
/// three such tensors plus a `[2, 10, 4096, 4]` score matrix, ~32 MB at bf16 —
/// and ~131 MB leaves better than 4x for allocator caching across the 70
/// injections rather than budgeting to the arithmetic exactly.
pub(crate) const IP_ADAPTER_SDXL_VRAM_OVERHEAD_BYTES: u64 = 830_000_000;

/// The adapter half of [`IP_ADAPTER_SDXL_VRAM_OVERHEAD_BYTES`], pinned the same
/// way and read by the same test.
#[cfg(test)]
pub(crate) const IP_ADAPTER_SDXL_ADAPTER_BF16_BYTES: u64 = 698_376_192;

/// The device overhead one family's resident image-prompt adapter costs.
///
/// One switch so the estimate cannot charge SDXL's 830 MB for an SD1.5 render
/// (which parks 8 GB cards that could run it comfortably) or SD1.5's 90 MB for
/// an SDXL one (which admits a render with 740 MB nowhere to go). The same
/// reason [`identity_adapter_overhead_bytes`] is a switch.
pub(crate) fn ip_adapter_overhead_bytes(
    family: mold_core::ip_adapter_assets::ImagePromptFamily,
) -> u64 {
    match family {
        mold_core::ip_adapter_assets::ImagePromptFamily::Sd15 => {
            IP_ADAPTER_SD15_VRAM_OVERHEAD_BYTES
        }
        mold_core::ip_adapter_assets::ImagePromptFamily::Sdxl => {
            IP_ADAPTER_SDXL_VRAM_OVERHEAD_BYTES
        }
    }
}

/// Parameters in the OpenCLIP ViT-H/14 vision tower, from its own published
/// architecture rather than from the file size.
///
/// `hidden 1280, layers 32, heads 16, intermediate 5120, patch 14, image 224,
/// projection 1024` (`h94/IP-Adapter`'s `models/image_encoder/config.json`,
/// the file this bundle carries for exactly this reason):
///
/// * embeddings `1280 + 1280 x 3 x 14 x 14 + 257 x 1280 = 1,082,880`
/// * `pre_layrnorm` `2 x 1280 = 2,560`
/// * per encoder layer `4 x (1280² + 1280) + 2 x (2 x 1280) + (1280 x 5120 +
///   5120) + (5120 x 1280 + 1280) = 19,677,440`, times 32
/// * `post_layernorm` `2,560`
/// * bias-free `visual_projection` `1280 x 1024 = 1,310,720`
///
/// `x 4` bytes is 2,528,307,200 against the 2,528,373,448-byte f32 file — the
/// difference is the safetensors header, which is how this is checked rather
/// than asserted.
#[cfg(test)]
pub(crate) const IP_ADAPTER_VISION_TOWER_PARAMETERS: u64 = 632_076_800;

/// Device memory the reference-picture ENCODE peaks at, beside
/// [`ip_adapter_overhead_bytes`].
///
/// **These bytes are a PEAK, not residency.** The tower follows the crate's
/// drop-and-reload rule — `encoders/openclip_vision.rs`'s "build it, encode,
/// drop it" — so it is constructed, runs one 224x224 forward per reference
/// picture, and is released BEFORE the denoise loop allocates a single step's
/// arena. Nothing here is held for the render. It is charged as its own named
/// term rather than folded into the adapter's for exactly that reason: every
/// reference render pays the adapter for the whole denoise, and this is a
/// strictly earlier, strictly disjoint phase.
///
/// It is nevertheless charged ADDITIVELY rather than as a maximum against the
/// denoise peak, for the same reason
/// [`IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES`] is: the engine cache is warm
/// across requests, so the transformer may well already be resident when this
/// phase runs, and a term that assumed the two peaks were mutually exclusive
/// would admit a render with nowhere to put the tower.
///
/// The figure is `IP_ADAPTER_VISION_TOWER_PARAMETERS` at **f16** —
/// 1,264,153,600 bytes — plus ~136 MB for the forward's working set (a
/// `[1, 257, 1280]` hidden stream, a `[1, 16, 257, 257]` score matrix, a
/// `[1, 257, 5120]` MLP intermediate) and the staging the `VarBuilder` copies
/// through as it widens the file.
///
/// f16 is not a guess about the caller: it is what the caller does. The tower
/// takes the ENGINE's dtype (`sd_reference::SdReferenceState::encode` hands
/// `resolve`'s `dtype` straight to `OpenClipVisionTower`), and both SD engines
/// resolve that to `DType::F16` on every GPU (`sd15/pipeline.rs`,
/// `sdxl/pipeline.rs`) — this is a VRAM term, so the CPU f32 case is not the
/// one being charged. An earlier version charged the checkpoint's own f32
/// width, which is the width on DISK; that over-charged every reference render
/// by ~1.24 GB and would have parked renders a 12 GB card can run, which is
/// the mistake the identity terms' history records twice.
pub(crate) const IP_ADAPTER_VISION_TOWER_VRAM_PEAK_BYTES: u64 = 1_400_000_000;

/// Whether this request will actually condition on a reference picture.
///
/// Weight zero is completely inert — no assets are planned, nothing is
/// downloaded, no tower is built, and no memory is charged — so the predicate
/// is the effective weight, never the mere presence of `edit_images`. It is
/// [`crate::ip_adapter_dependencies`]'s own predicate, so the bundle that is
/// downloaded and the memory that is charged can never disagree.
#[cfg(test)]
pub(crate) fn request_charges_ip_adapter_overhead(req: &GenerateRequest) -> bool {
    request_charges_ip_adapter_overhead_with_projection(req, None)
}

pub(crate) fn request_charges_ip_adapter_overhead_with_projection(
    req: &GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> bool {
    crate::ip_adapter_dependencies::request_needs_ip_adapter_assets_with_projection(req, projection)
}

/// The image-prompt family whose overhead this request is charged, or `None`
/// when it conditions on no reference picture — or does, on a family that has
/// no image-prompt adapter at all (FLUX.2 and Qwen-Image-Edit both carry
/// `edit_images`, and neither loads any of this).
///
/// The manifest is the authority, exactly as it is for identity: a built-in
/// checkpoint's own family answers, and the hint is the fallback for the
/// live-catalog `cv:`/`hf:` ids that have no manifest. Where this differs from
/// identity is that the hint ALONE cannot answer — `activation_family_for`
/// maps `sd15` and `sdxl` onto the one `ActivationFamily::SdxlUnet`, so the
/// two bases that need different adapters are indistinguishable in it. The
/// second discriminator is the checkpoint's own composition: SDXL conditions
/// on TWO text encoders and SD1.5 on one, which is a structural fact about the
/// checkpoint rather than a guess from its name.
///
/// Both halves of that second encoder are asked, and that is not belt and
/// braces. The `paths` this estimate receives are `gpu_resident_paths`', which
/// clears `clip_encoder_2` whenever CLIP-G is parked on the CPU — reading the
/// encoder alone would silently reclassify exactly those SDXL renders as
/// SD1.5 and under-charge them by 740 MB. The tokenizer survives that
/// stripping (it is host-only by role and is never a GPU resident), and an
/// SDXL checkpoint cannot run without both, so the pair answers on every
/// placement.
fn ip_adapter_overhead_family_with_projection_and_hint(
    req: &GenerateRequest,
    paths: &ModelPaths,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
    hint: Option<ActivationHint>,
) -> Option<mold_core::ip_adapter_assets::ImagePromptFamily> {
    use mold_core::ip_adapter_assets::ImagePromptFamily;
    if !request_charges_ip_adapter_overhead_with_projection(req, projection) {
        return None;
    }
    if let Some(manifest) = mold_core::manifest::find_manifest(&req.model) {
        return ImagePromptFamily::from_generation_family(&manifest.family);
    }
    if hint.is_some_and(|hint| hint.family == ActivationFamily::SdxlUnet) {
        return Some(
            if paths.clip_encoder_2.is_some() || paths.clip_tokenizer_2.is_some() {
                ImagePromptFamily::Sdxl
            } else {
                ImagePromptFamily::Sd15
            },
        );
    }
    None
}

/// Device memory the face-identity EXTRACTION peaks at, beside
/// [`IDENTITY_VRAM_OVERHEAD_BYTES`].
///
/// #1220 charged the whole PuLID stack to VRAM on the assumption the extractor
/// would run on the generation device; #1223 removed it because the extractor
/// ran on the host at admission. #1227 phase 2 moved it back onto the render's
/// own leased device — so the charge returns, but as its OWN named term rather
/// than folded into the adapter's, because the two answer different questions:
/// every identity render pays the adapter for the whole denoise, and this is a
/// strictly earlier, strictly disjoint phase that is released before the
/// adapter loads.
///
/// It is charged ADDITIVELY rather than as a maximum against the adapter term,
/// and that is deliberate. The engine cache is warm across requests, so the
/// transformer may well already be resident when this phase runs; a term that
/// assumed the peak phases were mutually exclusive would admit a render with
/// nowhere to put this one. The figure itself is
/// [`mold_core::identity::EXTRACTION_DEVICE_PEAK_BYTES`], whose doc carries the
/// per-artifact derivation, and `pulid_device_parity.rs`'s
/// `the_measured_device_peak_is_within_ten_percent_of_the_charged_term` is the
/// live check on it.
///
/// **Metal charges this once and nothing else.** Unified memory means these
/// bytes ARE host bytes, and mold's standing rule is that Metal reserves no
/// host RAM separately — its host claim rides the unified device gate. On CUDA
/// the host side is only the private authenticated copy the `VarBuilder` reads
/// from, which the identity artifacts' own `is_host_only` component roles
/// already charge from their pinned sizes.
pub(crate) const IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES: u64 =
    mold_core::identity::EXTRACTION_DEVICE_PEAK_BYTES;

/// The adapter half of [`IDENTITY_VRAM_OVERHEAD_BYTES`], pinned against the
/// engine's own arithmetic by
/// `identity_overhead_matches_the_adapters_own_resident_arithmetic`. It is the
/// documented decomposition of the budget rather than a second input to it,
/// so only that test reads it.
#[cfg(test)]
pub(crate) const IDENTITY_ADAPTER_BF16_BYTES: u64 = 839_270_400;

/// Whether this request will actually condition on a face.
///
/// Weight zero is completely inert — no assets are planned, nothing is
/// downloaded, and no memory is charged — so the predicate is the effective
/// weight, never the mere presence of the fields.
#[cfg(test)]
pub(crate) fn request_charges_identity_overhead(req: &GenerateRequest) -> bool {
    request_charges_identity_overhead_with_projection(req, None)
}

pub(crate) fn request_charges_identity_overhead_with_projection(
    req: &GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> bool {
    (mold_core::identity::request_mentions_identity(req)
        || projection.is_some_and(|projection| projection.identity_present))
        && mold_core::identity::effective_id_weight(req) > 0.0
}

/// The identity family whose overhead this request is charged, or `None` when
/// it conditions on no face at all.
///
/// Both halves are required: a request may name identity fields on a model
/// that is not qualified — admission refuses it, but the estimate runs first
/// and must not invent an adapter for a checkpoint that has none.
#[cfg(test)]
pub(crate) fn identity_overhead_family(
    req: &GenerateRequest,
) -> Option<mold_core::identity::IdentityFamily> {
    identity_overhead_family_with_projection(req, None)
}

#[cfg(test)]
pub(crate) fn identity_overhead_family_with_projection(
    req: &GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> Option<mold_core::identity::IdentityFamily> {
    identity_overhead_family_with_projection_and_hint(req, projection, None)
}

fn identity_overhead_family_with_projection_and_hint(
    req: &GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
    hint: Option<ActivationHint>,
) -> Option<mold_core::identity::IdentityFamily> {
    let family_hint = hint.and_then(|hint| match hint.family {
        ActivationFamily::FluxDit => Some("flux"),
        ActivationFamily::SdxlUnet => Some("sdxl"),
        _ => None,
    });
    request_charges_identity_overhead_with_projection(req, projection)
        .then(|| mold_core::identity::identity_family_with_hint(&req.model, family_hint))
        .flatten()
}

/// Device memory a PuLID true-CFG render needs beside
/// [`IDENTITY_VRAM_OVERHEAD_BYTES`].
///
/// True CFG runs TWO transformer forwards per step instead of one, but they run
/// in sequence, so the peak does not double — the second forward reuses the
/// first's activation arena. What is genuinely additional, and resident for the
/// whole denoise, is:
///
/// | Term | Bytes | Where it comes from |
/// | --- | --- | --- |
/// | Negative T5 conditioning, `[1, 512, 4096]` bf16 | 4,194,304 | a second `prepare()` (`PuLID/app_flux.py:111`) |
/// | Negative pooled CLIP vector and its packed ids | ~50,000 | same |
/// | Unconditional identity context, `[1, 32, 2048]` bf16 | 131,072 | `PuLID/pulid/pipeline_flux.py:188-192` |
/// | The conditional prediction, held while the negative forward runs | 524,288 | `[1, 4096, 64]` bf16 at 1024x1024 |
/// | Cross-attention working headroom for the second injection pass | ~145,000,000 | the `[1, 4096, 3072]` + `[1, 4096, 2048]` pair, not reused between passes |
/// | **Total** | **150,000,000** | |
///
/// It is charged as its own term rather than folded into the identity overhead
/// because the two answer different requests: every identity render pays the
/// adapter, and only a true-CFG one pays this. Charging a request on the plain
/// identity estimate would admit a render whose second pass has nowhere to go.
///
/// The cost that is NOT memory is time: a true-CFG render is close to twice the
/// denoise wall clock, which the scheduler's learned phase timings observe
/// directly rather than predict from this constant.
pub(crate) const TRUE_CFG_VRAM_OVERHEAD_BYTES: u64 = 150_000_000;

/// Whether this request will actually run the true-CFG negative branch.
///
/// Delegates to the request contract so admission and the engine cannot
/// disagree about which requests are true-CFG requests — an inert 1.0 scale and
/// a zero identity weight both answer `false` here exactly as they do there.
#[cfg(test)]
pub(crate) fn request_charges_true_cfg_overhead(req: &GenerateRequest) -> bool {
    request_charges_true_cfg_overhead_with_projection(req, None)
}

pub(crate) fn request_charges_true_cfg_overhead_with_projection(
    req: &GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> bool {
    mold_core::identity::request_uses_true_cfg_with_identity_presence(
        req,
        mold_core::identity::request_carries_identity_photo(req)
            || projection.is_some_and(|projection| projection.identity_present),
    )
}

fn activation_memory_for_estimate(hint: Option<ActivationHint>, qwen_quantized: bool) -> u64 {
    if qwen_quantized {
        0
    } else {
        hint.map(|h| h.budget_bytes()).unwrap_or(0)
    }
}

/// Peak GPU residency for streaming-transformer families when the checkpoint's
/// own weight layout is not available.
///
/// Mirrors the Sequential strategy in `device::estimate_peak_memory` but
/// replaces the `transformer_size + vae_size` term with a
/// `STREAMING_TRANSFORMER_CAP` that bounds "block-streaming overhead,
/// fully-resident top-level weights, and VAE."
///
/// When the encoder does not compete with the transformer GPU, `encoder_total`
/// is dropped from the max because the prompt encoder lives in system RAM or
/// streams one layer at a time and pipes its conditioning to the transformer
/// GPU. This covers both LTX-2's CPU Gemma recovery and FLUX.2 Dev's streamed
/// Mistral3 prefix. When it does compete, the encoder phase pays the full file
/// total because those encoders load whole before being dropped for denoise.
///
/// This cap is deliberately shape-blind and is now only the fallback: it was
/// the sole LTX-2 estimate until #641 showed it under-counts the real peak by
/// more than 10x (the 19B FP8 preset carries 2.1 GB of non-block transformer
/// weights and a 2.4 GB VAE, not the ~200 MB each this cap assumed, and the
/// engine's adaptive planner then keeps ~19 GB of blocks resident on top).
/// [`crate::ltx2_admission`] owns the real model whenever the checkpoint's
/// header has been read.
fn streaming_transformer_peak(
    paths: &ModelPaths,
    gemma_competes_with_transformer_gpu: bool,
) -> u64 {
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

    let inference_phase = mold_inference::device::STREAMING_TRANSFORMER_CAP_BYTES;
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
/// On Metal, the shared policy bounds reclaim credit by native allocations,
/// the working-set capacity and live host headroom. A fresh post-drop sample
/// verifies the released buffers before a replacement is loaded.
/// On other platforms with no memory query available, the guard is a no-op.
pub(crate) fn preflight_memory_guard_for_request(
    model_name: &str,
    paths: &ModelPaths,
    active_vram_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
) -> Result<(), ApiError> {
    #[cfg(feature = "cuda")]
    {
        let effective_free = authoritative_cuda_available(
            mold_inference::device::usable_free_vram_bytes_result(gpu_ordinal),
        )?;
        preflight_memory_guard_with_available_on_gpu_for_request(
            model_name,
            paths,
            active_vram_bytes,
            effective_free,
            gpu_ordinal,
            hint,
            request_has_lora,
        )
    }

    #[cfg(not(feature = "cuda"))]
    {
        if let Some(available) = metal_available_after_reclaim(gpu_ordinal, active_vram_bytes)? {
            return preflight_memory_guard_with_available_on_gpu_for_request(
                model_name,
                paths,
                0,
                available,
                gpu_ordinal,
                hint,
                request_has_lora,
            );
        }
        // macOS unified memory: query system memory and add reclaimable footprint.
        if let Some(available) = mold_inference::device::available_system_memory_bytes() {
            if available > 0 {
                return preflight_memory_guard_with_available_on_gpu_for_request(
                    model_name,
                    paths,
                    active_vram_bytes,
                    available,
                    gpu_ordinal,
                    hint,
                    request_has_lora,
                );
            }
        }

        // No memory info available on this platform — skip the guard.
        Ok(())
    }
}

/// Recheck a frozen scheduler plan before dropping the currently resident
/// engine. Only the exact planned peak is authoritative; `active_vram_bytes`
/// is reclaimable because the caller will unload that engine before loading
/// the plan.
pub(crate) fn preflight_planned_memory_guard(
    model_name: &str,
    predicted_peak_bytes: u64,
    active_vram_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    #[cfg(feature = "cuda")]
    {
        let free = authoritative_cuda_available(
            mold_inference::device::usable_free_vram_bytes_result(gpu_ordinal),
        )?;
        check_planned_memory_budget_with_resident(
            model_name,
            predicted_peak_bytes,
            free,
            active_vram_bytes,
            mold_inference::device::total_vram_bytes(gpu_ordinal)
                .map(|total| total.saturating_sub(mold_inference::device::reserved_vram_bytes())),
            &rejection_suggestion_for_model(hint, model_name),
        )
    }

    #[cfg(not(feature = "cuda"))]
    {
        if let Some(available) = metal_available_after_reclaim(gpu_ordinal, active_vram_bytes)? {
            return check_planned_memory_budget_with_resident(
                model_name,
                predicted_peak_bytes,
                available,
                0,
                None,
                rejection_suggestion(hint),
            );
        }
        if let Some(available) = mold_inference::device::available_system_memory_bytes()
            .filter(|available| *available > 0)
        {
            return check_planned_memory_budget_with_resident(
                model_name,
                predicted_peak_bytes,
                available,
                active_vram_bytes,
                None,
                rejection_suggestion(hint),
            );
        }
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
#[cfg(all(test, not(any(feature = "cuda", feature = "metal"))))]
pub(crate) fn preflight_memory_guard_after_drop(
    model_name: &str,
    paths: &ModelPaths,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    preflight_memory_guard_after_drop_for_request(model_name, paths, gpu_ordinal, hint, false)
}

pub(crate) fn preflight_memory_guard_after_drop_for_request(
    model_name: &str,
    paths: &ModelPaths,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
) -> Result<(), ApiError> {
    #[cfg(feature = "cuda")]
    {
        let available = authoritative_cuda_available(
            mold_inference::device::post_drop_free_vram_bytes(gpu_ordinal),
        )?;
        preflight_memory_guard_with_available_on_gpu_for_request(
            model_name,
            paths,
            0,
            available,
            gpu_ordinal,
            hint,
            request_has_lora,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        mold_inference::device::release_pooled_metal_memory(gpu_ordinal);
        if let Some(available) = metal_available_after_reclaim(gpu_ordinal, 0)? {
            return preflight_memory_guard_with_available_on_gpu_for_request(
                model_name,
                paths,
                0,
                available,
                gpu_ordinal,
                hint,
                request_has_lora,
            );
        }
        // Preserve the no-backend behavior when no native Metal policy exists.
        let _ = (model_name, paths, gpu_ordinal, hint, request_has_lora);
        Ok(())
    }
}

/// Authoritative post-drop recheck for a frozen scheduler plan.
///
/// Both CUDA and Metal prove the exact peak against a fresh sample; Metal
/// releases the buffer pool first and never credits the old engine twice.
pub(crate) fn preflight_planned_memory_guard_after_drop(
    model_name: &str,
    predicted_peak_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] gpu_ordinal: usize,
    hint: Option<ActivationHint>,
) -> Result<(), ApiError> {
    #[cfg(feature = "cuda")]
    {
        let available = authoritative_cuda_available(
            mold_inference::device::post_drop_free_vram_bytes(gpu_ordinal),
        )?;
        check_planned_memory_budget(
            model_name,
            predicted_peak_bytes,
            available,
            mold_inference::device::total_vram_bytes(gpu_ordinal)
                .map(|total| total.saturating_sub(mold_inference::device::reserved_vram_bytes())),
            rejection_suggestion(hint),
        )
    }

    #[cfg(not(feature = "cuda"))]
    {
        mold_inference::device::release_pooled_metal_memory(gpu_ordinal);
        if let Some(available) = metal_available_after_reclaim(gpu_ordinal, 0)? {
            return check_planned_memory_budget(
                model_name,
                predicted_peak_bytes,
                available,
                None,
                rejection_suggestion(hint),
            );
        }
        let _ = (model_name, predicted_peak_bytes, gpu_ordinal, hint);
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
    {
        if let Some(available) = metal_available_after_reclaim(gpu_ordinal, active_vram_bytes)? {
            return Ok(Some(available));
        }
        Ok(mold_inference::device::available_system_memory_bytes()
            .filter(|available| *available > 0)
            .map(|available| available.saturating_add(active_vram_bytes)))
    }
}

/// Credit only measured, reclaimable Metal allocations, and never exceed the
/// current total policy. A failed supported probe must not use the RAM fallback.
#[cfg(not(feature = "cuda"))]
fn metal_available_after_reclaim(
    ordinal: usize,
    reclaimable: u64,
) -> Result<Option<u64>, ApiError> {
    metal_available_from_sample(
        mold_inference::metal_memory::snapshot(ordinal).as_ref(),
        reclaimable,
    )
}

#[cfg(not(feature = "cuda"))]
fn metal_available_from_sample(
    sample: Option<&mold_core::metal_memory::MetalMemorySnapshot>,
    reclaimable: u64,
) -> Result<Option<u64>, ApiError> {
    let Some(sample) = sample else {
        return Ok(None);
    };
    if sample.allocation_headroom_bytes.is_none() {
        return Err(ApiError::insufficient_memory(format!(
            "Metal memory admission unavailable: {}",
            sample
                .error
                .as_deref()
                .unwrap_or("could not measure the working-set budget"),
        )));
    }
    Ok(Some(sample.with_reclaimable(reclaimable)))
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

    if large_flux2_bf16_should_auto_offload(
        paths,
        hint,
        available_bytes,
        activation_memory_for_estimate(hint, false),
    ) {
        return mold_inference::LoadStrategy::Sequential;
    }

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
    // The fp8 widen holds two bytes per parameter where the file-length
    // estimate charges one, and this is a LOAD-STRATEGY decision — the whole
    // point of which is whether the eager set co-resides.
    let widen = flux2_fp8_widen_extra_bytes(paths, hint, Some(available_bytes));
    let eager_peak =
        mold_inference::device::estimate_peak_memory(paths, mold_inference::LoadStrategy::Eager)
            .saturating_add(activation)
            .saturating_add(widen);
    let sequential_peak = mold_inference::device::estimate_peak_memory(
        paths,
        mold_inference::LoadStrategy::Sequential,
    )
    .saturating_add(activation)
    .saturating_add(widen);
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

/// Apply request-specific engine constraints after the general memory policy.
///
/// Flux.2 and Z-Image merge LoRAs while constructing the transformer and use
/// their load-use-drop generation paths for adapted requests. Eagerly loading
/// an unadapted transformer first leaves it resident when `generate()` begins,
/// so the subsequent LoRA transformer build either doubles the peak or fails
/// immediately on 24 GB cards.
///
/// Flux.2 requests that carry CONDITIONING IMAGES also require the sequential
/// path — a source image or ordered `edit_images` references, which is the
/// same constraint for the same reason. Either is encoded in a VAE-only phase
/// that drops the VAE before the transformer loads
/// (`flux2/pipeline.rs::uses_sequential_generate_path`); eager mode keeps both
/// resident and can OOM with Klein-9B BF16 on a 24 GB card, and worse,
/// `generate_inner` would unload the eagerly loaded transformer anyway. The
/// execution plan must make these runtime constraints authoritative.
///
/// What this does NOT decide is transformer RESIDENCY. Both engines keep or
/// drop the transformer around VAE decode on their own budget
/// (`mold_inference::device::still_transformer_residency`), sampled against
/// the card at the moment of the decode rather than against the planner's
/// estimate, so an Eager strategy no longer implies a transformer that
/// survives the render and a Sequential one no longer implies a reload.
pub(crate) fn request_aware_load_strategy(
    strategy: mold_inference::LoadStrategy,
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    request_has_lora: bool,
    request_has_conditioning_images: bool,
) -> mold_inference::LoadStrategy {
    let transformer_path = transformer_path_lower(paths);
    let flux2 = transformer_path_looks_flux2(&transformer_path)
        || hint.is_some_and(|hint| hint.family == ActivationFamily::Flux2Dit);
    let zimage = transformer_path_looks_zimage(&transformer_path)
        || hint.is_some_and(|hint| hint.family == ActivationFamily::ZImageDit);
    if (request_has_lora && (flux2 || zimage)) || (request_has_conditioning_images && flux2) {
        mold_inference::LoadStrategy::Sequential
    } else {
        strategy
    }
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

    // FLUX.2 asks the ENGINE whether this checkpoint can stream at all, rather
    // than keeping a second copy of the format predicate here. The two answers
    // disagreeing is how a plan gets admitted that the loader cannot honour.
    if (transformer_looks_flux2 || hint.is_some_and(|h| h.family == ActivationFamily::Flux2Dit))
        && flux2_block_offload_unsupported_reason(paths, request_has_lora).is_some()
    {
        return false;
    }

    if transformer_is_gguf
        && hint.is_some_and(|h| {
            matches!(
                h.family,
                ActivationFamily::Sd3Mmdit | ActivationFamily::ZImageDit
            )
        })
    {
        return false;
    }
    let _ = transformer_looks_nvfp4;

    forced_offload
        || large_flux_bf16_should_auto_offload(paths, hint, None, 0)
        || large_flux2_bf16_should_auto_offload(paths, hint, None, 0)
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
    /// Wan will park trailing transformer blocks for this render.
    ///
    /// Deliberately separate from [`Self::block_offload`], which means "stream
    /// this transformer's blocks from host RAM" and drives both the component
    /// load strategy and the plan's host-RAM reservation. Wan's parking is
    /// neither: it keeps one expert's non-parked blocks resident, holds only
    /// the parked subset on the host, and never streams the whole file — so
    /// folding it into that flag would reserve every A14B expert at full size
    /// (22-31 GB for the pair, of which the runtime holds one expert's tail)
    /// and could reject a render whose real working set fits. This names the
    /// disposition for the plan and nothing else.
    pub(crate) wan_block_offload: bool,
    pub(crate) under_memory_pressure: bool,
    pub(crate) eager_peak_memory_bytes: u64,
    /// The figure [`Self::fits_available_memory`] compared the peak against:
    /// the whole budget for the families whose estimate is measured, and 90 %
    /// of it for every family whose estimate is a heuristic. A refusal must
    /// print THIS, not the budget it derives from.
    pub(crate) admissible_ceiling_bytes: Option<u64>,
    pub(crate) fits_available_memory: Option<bool>,
}

#[derive(Clone, Copy)]
pub(crate) struct GenerationOffloadPolicy {
    forced: bool,
    wan: mold_inference::wan::block_offload::AdmissionPolicy,
    metal: bool,
}

impl GenerationOffloadPolicy {
    pub(crate) const fn new(
        forced: bool,
        wan: mold_inference::wan::block_offload::AdmissionPolicy,
        metal: bool,
    ) -> Self {
        Self { forced, wan, metal }
    }
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
    offload_policy: GenerationOffloadPolicy,
    available_memory_bytes: Option<u64>,
    request_has_lora: bool,
    gemma_competes: bool,
) -> GenerationMemoryBudget {
    estimate_generation_memory_for_request_with_projection(
        req,
        paths,
        hint,
        offload_policy,
        available_memory_bytes,
        request_has_lora,
        gemma_competes,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_generation_memory_for_request_with_projection(
    req: &GenerateRequest,
    paths: &ModelPaths,
    hint: Option<ActivationHint>,
    offload_policy: GenerationOffloadPolicy,
    available_memory_bytes: Option<u64>,
    request_has_lora: bool,
    gemma_competes: bool,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> GenerationMemoryBudget {
    let transformer_path = transformer_path_lower(paths);
    let streaming = hint
        .map(|h| h.family.streaming_transformer())
        .unwrap_or_else(|| transformer_path_looks_ltx2(&transformer_path));
    // Derived ONCE and handed to both the generic peak and the eager peak
    // below: `under_memory_pressure` and the planner's auto-park decision are
    // computed from `eager_peak`, so re-pricing only the sequential arm would
    // leave FLUX.2 [dev]'s encoder parked on the CPU anyway.
    let streamed_encoder_charge = streamed_text_encoder_device_charge(&req.model, paths);
    let qwen_quantized = hint.is_some_and(|h| h.family == ActivationFamily::QwenImageDit)
        && transformer_path_is_gguf(paths);
    // Wan's token grid and per-token slope are properties of the checkpoint,
    // so price against its own header when one is reachable rather than the
    // conservative A14B fallback.
    let wan_geometry = hint
        .filter(|h| h.family == ActivationFamily::WanVideo)
        .and_then(|_| crate::wan_admission::checkpoint_geometry_cached(paths));
    // Derived here, beside the geometry, because this is where `paths` is in
    // scope. A distill adapter refuses the step cache, so it decides whether
    // the cache's retained tensors are charged (#1482).
    let wan_distilled = crate::wan_admission::wan_distill_is_active(paths);
    // Derived here for the same reason wan's geometry is: this is where
    // `paths` is in scope, and the denoise budget is a function of the
    // transformer's own width rather than of the canvas.
    let flux2_geometry = hint
        .filter(|h| h.family == ActivationFamily::Flux2Dit)
        .and_then(|_| flux2_activation_geometry(&req.model, paths));
    let activation = request_sensitive_activation_memory_with_wan_geometry(
        req,
        hint,
        qwen_quantized,
        wan_geometry,
        wan_distilled,
        flux2_geometry,
        projection,
    );
    // The SAME request without the FLUX.2 denoise model, and it answers a
    // different question: `eager_peak` below asks whether every component can
    // be resident AT ONCE, which is the encoder-placement decision. A FLUX.2
    // text encoder is `DropReload` — it is gone before the first denoise step
    // — so its co-residency is a question about WEIGHTS, and folding a 3 GB
    // denoise working set into it auto-parks the Mistral3 conditioner onto the
    // CPU of an idle 46 GB card, which is the 78.8 s F32 encode this campaign
    // exists to remove. Identical to `activation` for every other family, so
    // no other placement decision moves.
    let co_residency_activation = if flux2_geometry.is_some() {
        request_sensitive_activation_memory_with_wan_geometry(
            req,
            hint,
            qwen_quantized,
            wan_geometry,
            wan_distilled,
            None,
            projection,
        )
    } else {
        activation
    };
    let conservative_block_offload = server_offload_enabled_for_paths_with_request(
        paths,
        hint,
        request_has_lora,
        offload_policy.forced,
    );
    // Both families' size predicates are narrowed by the measured card, in
    // the same shape and for the same reason: a 46 GB L40S that can hold a
    // 23.8 GB BF16 checkpoint whole must not stream its blocks at the
    // documented 3-5x penalty.
    let block_offload = if conservative_block_offload && !offload_policy.forced && !request_has_lora
    {
        if large_flux2_bf16_should_auto_offload(paths, hint, None, 0) {
            large_flux2_bf16_should_auto_offload(paths, hint, available_memory_bytes, activation)
        } else if large_flux_bf16_should_auto_offload(paths, hint, None, 0) {
            large_flux_bf16_should_auto_offload(paths, hint, available_memory_bytes, activation)
        } else {
            conservative_block_offload
        }
    } else {
        conservative_block_offload
    };
    let flux_offload = hint.is_some_and(|h| {
        matches!(
            h.family,
            ActivationFamily::FluxDit | ActivationFamily::Flux2Dit
        )
    }) && block_offload;
    let available_memory_bytes = available_memory_bytes.filter(|available| *available > 0);
    // Weight bytes the wan arm below discounted because parking can free them.
    // Kept so the plan can re-add them and ask the engine's own question — will
    // this render park? — rather than inferring it from the discounted peak.
    let mut wan_offload_relief = 0u64;
    // LTX-2: when the checkpoint's weight layout has already been read, the
    // adaptive-residency model in `ltx2_admission` is authoritative — it counts
    // the non-block transformer weights, the bundled VAE, the resident block
    // set the engine will actually keep, and a fragmentation margin. The flat
    // streaming cap below is only the cold-cache fallback.
    let ltx2 = streaming
        .then(|| {
            crate::ltx2_admission::Ltx2ShapeHint::from_request_with_projection(req, projection)
        })
        .zip(available_memory_bytes)
        .and_then(|(shape, available)| {
            crate::ltx2_admission::checkpoint_facts_cached(&paths.transformer)
                .map(|facts| (facts, shape, available))
        });
    let (peak, activation) = match &ltx2 {
        Some((facts, shape, available)) => {
            let activation = facts.activation_bytes(*shape);
            let estimate = crate::ltx2_admission::ltx2_peak_estimate(facts, activation, *available);
            (estimate.peak_bytes, activation)
        }
        None => {
            let base_peak = base_peak_memory_for_paths(
                paths,
                hint,
                streaming,
                flux_offload,
                qwen_quantized,
                gemma_competes,
                streamed_encoder_charge,
            );
            let wan = hint.is_some_and(|h| h.family == ActivationFamily::WanVideo);
            if wan {
                // Block offload (#776 item 3) can park trailing transformer
                // blocks in host RAM, so a shape that does not fit resident
                // may still be feasible. Charging the full weight term would
                // refuse it before the engine ever got the chance.
                wan_offload_relief = wan_block_offload_relief_for_policy(
                    paths,
                    offload_policy.wan,
                    wan_geometry.map(|geometry| geometry.num_layers as usize),
                );
            }
            // Carry the wan checkpoint geometry in here too. Wan is never
            // streaming, so this arm is the only one it takes — recomputing
            // without the geometry silently discarded the header read and
            // priced every wan request with the A14B fallback.
            let activation = request_sensitive_activation_memory_with_wan_geometry(
                req,
                hint,
                qwen_quantized,
                wan_geometry,
                wan_distilled,
                flux2_geometry,
                projection,
            );
            let peak = if wan && offload_policy.metal {
                // Wan's sequential Metal engine drops its text encoder before
                // it loads the transformer and VAE. Price those phases
                // separately: adding denoise activation to the generic
                // max(weights) estimate makes a GPU-resident encoder appear to
                // overlap the denoise and creates a discontinuous false OOM at
                // the encoder placement threshold.
                let (encoder_weights, inference_weights) =
                    mold_inference::device::estimate_sequential_phase_weights(paths);
                let encoder_is_gguf = paths
                    .t5_encoder
                    .iter()
                    .chain(paths.text_encoder_files.iter())
                    .any(|path| {
                        path.extension()
                            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                    });
                let encoder_threshold = if encoder_is_gguf {
                    mold_inference::device::t5_metal_gguf_vram_threshold(encoder_weights)
                } else {
                    encoder_weights.saturating_add(WAN_REQUEST_AWARE_HEADROOM_BYTES)
                };
                // The runtime makes this same fit decision immediately before
                // encoding. Below the threshold UMT5 runs on CPU and consumes
                // no Metal budget; at or above it the complete encoder phase
                // must be covered by the plan.
                let encoder_peak = if available_memory_bytes
                    .is_none_or(|available| available >= encoder_threshold)
                {
                    encoder_threshold
                } else {
                    0
                };
                let denoise_peak = inference_weights
                    .saturating_sub(wan_offload_relief)
                    .saturating_add(activation)
                    // #1059 measured 8,184,725,504 Metal bytes in this phase:
                    // the CUDA-calibrated fit still needs the generic runtime
                    // allowance on Metal, plus a small allocator margin.
                    .saturating_add(WAN_REQUEST_AWARE_HEADROOM_BYTES)
                    .saturating_add(WAN_METAL_ALLOCATOR_MARGIN_BYTES);
                encoder_peak.max(denoise_peak)
            } else if wan {
                // Preserve the calibrated CUDA Wan estimate exactly: its
                // measured activation fit already includes allocator runtime.
                base_peak
                    .saturating_sub(WAN_REQUEST_AWARE_HEADROOM_BYTES)
                    .saturating_sub(wan_offload_relief)
                    .saturating_add(activation)
            } else {
                base_peak.saturating_add(activation)
            };
            (peak, activation)
        }
    };
    // Identity conditioning adds resident weights and activations to whichever
    // arm produced the peak. Charged from the request rather than from a path,
    // because the assets are not part of the checkpoint's `ModelPaths`.
    // The adapter term follows the FAMILY, because the adapter does. The
    // extraction term does not: the detector, recognizer, parser, and tower are
    // shared, and only the IDFormer's prefix differs.
    let peak = match identity_overhead_family_with_projection_and_hint(req, projection, hint) {
        Some(family) => peak
            .saturating_add(identity_adapter_overhead_bytes(family))
            .saturating_add(IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES),
        None => peak,
    };
    // Image prompting adds the same two kinds of term identity does, and is
    // charged from the request for the same reason: the assets are not part of
    // the checkpoint's `ModelPaths`. The adapter term follows the FAMILY,
    // because the adapter does; the tower term does not, because SD1.5 and
    // SDXL condition on the very same ViT-H/14 file.
    let peak =
        match ip_adapter_overhead_family_with_projection_and_hint(req, paths, projection, hint) {
            Some(family) => peak
                .saturating_add(ip_adapter_overhead_bytes(family))
                .saturating_add(IP_ADAPTER_VISION_TOWER_VRAM_PEAK_BYTES),
            None => peak,
        };
    // A true-CFG render's second forward per step is additional resident
    // conditioning and a second cross-attention pass. Charged separately from
    // the identity overhead because only a request that engages the branch pays
    // it — never let one be admitted on the plain estimate.
    let peak = if request_charges_true_cfg_overhead_with_projection(req, projection) {
        peak.saturating_add(TRUE_CFG_VRAM_OVERHEAD_BYTES)
    } else {
        peak
    };
    let load_strategy = request_aware_load_strategy(
        select_server_load_strategy_for_budget(paths, available_memory_bytes, hint),
        paths,
        hint,
        request_has_lora,
        // References route exactly like a source image, and the projection is
        // the durable-queue answer for a request whose bytes were already
        // handed off — a reference-only Klein job that read only the inline
        // field would be planned Eager and then unloaded.
        req.source_image.is_some()
            || req
                .edit_images
                .as_ref()
                .is_some_and(|images| !images.is_empty())
            || projection.is_some_and(|projection| {
                // `edit_image_count`, not the `edit_images` dimension slots:
                // the slots are capped at the flux2 ceiling while the count is
                // the unbounded truth, and presence is the only question here.
                projection.source_image || projection.edit_image_count > 0
            }),
    );
    // Charged HERE, before `eager_peak` is used, because `under_memory_pressure`
    // reads it and `should_auto_park_text_encoder` reads that — which IS the
    // encoder-placement decision this charge exists for. Adding it only to the
    // final peak left the one consumer that matters reading the old number.
    let fp8_widen_bytes = flux2_fp8_widen_extra_bytes(paths, hint, available_memory_bytes);
    let eager_peak = mold_inference::device::estimate_peak_memory_with_encoder_override(
        paths,
        mold_inference::LoadStrategy::Eager,
        streamed_encoder_charge,
    )
    .saturating_add(co_residency_activation)
    .saturating_add(fp8_widen_bytes);
    let under_memory_pressure = available_memory_bytes
        .is_some_and(|available| eager_peak > available.saturating_mul(9) / 10);
    let qwen_family = hint.is_some_and(|h| h.family == ActivationFamily::QwenImageDit);
    // Wan joins Qwen-Image on the un-derated cap, for the same reason and now
    // with the evidence to back it.
    //
    // The 90% cap is a proxy for "the estimate is a heuristic, so leave room".
    // Wan's is no longer a heuristic: it is a measured per-token slope plus a
    // measured flat term, validated against four points on real hardware, and
    // the family is phase-sequential — the UMT5 encoder is dropped before the
    // transformer denoises, so the peak this predicts is the whole peak.
    //
    // Derating it a second time is not conservatism, it is a wrong answer: the
    // shipped 53-frame A14B default measures 23,975 MiB on a 24 GB card, so a
    // 90% cap refuses the tier's own default at the shape it was chosen for.
    // Admission still refuses 81 frames, which is the shape that actually
    // OOM'd.
    let wan_family = hint.is_some_and(|h| h.family == ActivationFamily::WanVideo);
    // Name wan's block offload in the plan (#776 item 3's acceptance criterion).
    //
    // Every other family reaches `OffloadMode::Block` through the *request*
    // flag above, which the wan factory arm does not read: wan decides its own
    // residency at load time from the render's activation budget. So a wan
    // render that parks half its blocks reported `OffloadMode::None`, and the
    // execution descriptor fingerprinted two materially different executions
    // the same. Asking `will_park` — the engine's own predicate, against the
    // peak with the relief added back, which is what the engine sees before it
    // parks anything — is what makes the plan describe what actually runs.
    // Restricted to checkpoints that *can* park: fp8 and plain safetensors have
    // no byte round trip, so claiming the disposition for them would be a
    // promise the engine cannot keep.
    // Derived, never OR'd with the generic flag: `MOLD_OFFLOAD=1` sets that
    // flag for every family, but `MOLD_WAN_OFFLOAD_BLOCKS=0` still turns wan's
    // parking off, and an fp8 checkpoint cannot park at all. OR-ing would put
    // both cases in the plan as offloads the engine will not perform.
    // `will_park` is the engine's own predicate and already resolves both
    // variables, so asking it is what keeps the two in step.
    //
    // This is a *prediction* from the admission snapshot, and the engine asks
    // the same question again against a fresh free-VRAM reading once the
    // weights are resident — which is where the park count comes from, since
    // that needs the checkpoint's own block size. So the two can disagree in
    // one narrow window: free VRAM moving between admission and load. The
    // engine's answer is the one that must win there, because it is the
    // reading taken against the memory the denoise will actually run in, and
    // freezing "do not park" from a staler snapshot trades a mislabelled
    // fingerprint for an OOM. Naming the disposition is what this field is
    // for; owning the count is not.
    let wan_block_offload = wan_family
        && wan_transformer_can_park(paths)
        && available_memory_bytes.is_some_and(|available| {
            offload_policy
                .wan
                .will_park(peak.saturating_add(wan_offload_relief), available)
        });
    // `MOLD_OFFLOAD=1` sets the generic streaming flag for every family, but
    // no wan arm streams a transformer from host RAM — the factory does not
    // even read the flag. Leaving it set made the plan reserve the whole
    // checkpoint on the host (both A14B experts) and label it `StreamedBlocks`
    // for an execution that never happens, including on fp8, which cannot park
    // at all. Wan's disposition is `wan_block_offload` and nothing else.
    let block_offload = block_offload && !wan_family;
    // An fp8 FLUX.2 checkpoint that the engine widens once at load holds TWO
    // bytes per parameter, not one, and until this charge existed every
    // server-side estimate priced the file. On a 32 GB card a klein-9B fp8
    // planned as 9.08 GB, resided as 18.16 GB, and the encoder-variant
    // selector then found no room for Qwen3 and fell back to a Q8 GGUF or to
    // the CPU — the F32 encode this campaign exists to remove. The widen gate
    // and the encoder selector could not see each other; now the planner sees
    // both.
    let peak = peak.saturating_add(fp8_widen_bytes);
    // ONE expression for the ceiling and for the verdict, so the number a
    // refusal prints is by construction the number the decision used. Printing
    // the raw budget instead is how a 43.00 GB plan refused against a 42.05 GB
    // ceiling came out as "still 0.0 GB short … 46.72 GB available".
    let admissible_ceiling_bytes = available_memory_bytes.map(|available| {
        if qwen_family || wan_family {
            available
        } else {
            available.saturating_mul(9) / 10
        }
    });
    let fits_available_memory = admissible_ceiling_bytes.map(|ceiling| peak <= ceiling);

    GenerationMemoryBudget {
        peak_memory_bytes: peak,
        activation_memory_bytes: activation,
        available_memory_bytes,
        load_strategy,
        block_offload,
        wan_block_offload,
        under_memory_pressure,
        eager_peak_memory_bytes: eager_peak,
        fits_available_memory,
        admissible_ceiling_bytes,
    }
}

#[cfg(test)]
fn request_sensitive_activation_memory(
    req: &GenerateRequest,
    hint: Option<ActivationHint>,
    qwen_quantized: bool,
) -> u64 {
    request_sensitive_activation_memory_with_wan_geometry(
        req,
        hint,
        qwen_quantized,
        None,
        false,
        None,
        None,
    )
}

/// As [`request_sensitive_activation_memory`], with the wan checkpoint's real
/// geometry when the caller has read its header.
///
/// Wan is the one family whose activation cost cannot be recovered from the
/// request alone: the token grid depends on which VAE generation the
/// checkpoint pairs with, and the per-token slope on its width. Callers
/// without a path pass `None` and get the A14B shape, which is the largest
/// shipped tier and therefore the conservative choice for admission.
/// Weight bytes block offload can be relied on to free for a wan render.
///
/// Zero unless the transformer is a GGUF checkpoint: parking is a raw-byte
/// round trip through `QTensor::data`, which the plain and fp8 weight sources
/// have no equivalent of, so promising relief for them would admit a shape the
/// engine cannot then fit.
///
/// The fraction is measured and deliberately smaller than what parking
/// achieved at its best - see `mold_inference::wan::block_offload`.
fn wan_block_offload_relief(paths: &ModelPaths) -> u64 {
    if !wan_transformer_can_park(paths) {
        return 0;
    }
    let bytes = std::fs::metadata(&paths.transformer)
        .map(|m| m.len())
        .unwrap_or(0);
    mold_inference::wan::block_offload::max_block_offload_relief_bytes(bytes)
}

fn wan_block_offload_relief_for_policy(
    paths: &ModelPaths,
    policy: mold_inference::wan::block_offload::AdmissionPolicy,
    total_blocks: Option<usize>,
) -> u64 {
    if policy.supports_max_relief(total_blocks) {
        wan_block_offload_relief(paths)
    } else {
        0
    }
}

/// Whether this checkpoint's weights can park at all.
///
/// Parking is a raw-byte round trip through `QTensor::data`, so it exists only
/// for GGUF. The fp8 and plain safetensors sources have no equivalent, which is
/// why their envelopes are what fits resident.
fn wan_transformer_can_park(paths: &ModelPaths) -> bool {
    paths
        .transformer
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

#[allow(clippy::too_many_arguments)]
fn request_sensitive_activation_memory_with_wan_geometry(
    req: &GenerateRequest,
    hint: Option<ActivationHint>,
    qwen_quantized: bool,
    wan_geometry: Option<mold_inference::device::WanActivationGeometry>,
    wan_distilled: bool,
    flux2_geometry: Option<mold_inference::device::Flux2ActivationGeometry>,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> u64 {
    let batch = u64::from(req.batch_size.max(1));
    // Wan prices its own CFG: `wan::pipeline::needs_cfg_pass` keys on guidance
    // alone, and an absent negative is filled engine-side with the tuned
    // default, so the `negative_prompt.is_some()` gate below is wrong for it.
    // The two forwards are also sequential, so CFG is a bounded additive term
    // rather than a multiplier — see `crate::wan_admission`.
    let wan = hint.is_some_and(|h| h.family == ActivationFamily::WanVideo);
    let cfg_factor = if !wan && cfg_active(req.guidance) && req.negative_prompt.is_some() {
        2
    } else {
        1
    };
    // LTX-2's activation working set is a function of transformer tokens, not
    // of image pixel area: the pixel-area heuristic scaled by latent frames
    // under-counted the 1024x1024 x 97 frame stage-2 shape by several GB
    // (#641). Price it from the same token model admission uses.
    let base = if wan {
        // Wan owns its own token model, including its own CFG treatment, so
        // it bypasses the pixel-area estimate and the `cfg_factor` above. It
        // does NOT bypass the per-request tail below: an I2V request carries a
        // source image and the distill tiers ship two adapters, both of which
        // are real resident memory.
        crate::wan_admission::wan_activation_bytes(
            crate::wan_admission::WanShapeHint::from_request_with_projection(req, projection),
            wan_geometry.unwrap_or_else(mold_inference::device::WanActivationGeometry::a14b),
            wan_distilled,
        )
    } else if hint.is_some_and(|h| h.family == ActivationFamily::Hunyuan3dShape) {
        // A mesh has no canvas, so the pixel-area estimate would price every
        // 3-D render identically no matter what was asked for. The peak is
        // whichever of the image conditioner, the shape DiT and one decode
        // chunk is largest — see `crate::hunyuan3d_admission`.
        crate::hunyuan3d_admission::activation_peak_bytes(
            crate::hunyuan3d_admission::Hunyuan3dShape::from_request(req),
        )
    } else if let Some(geometry) =
        flux2_geometry.filter(|_| hint.is_some_and(|h| h.family == ActivationFamily::Flux2Dit))
    {
        // FLUX.2 prices its denoise from the transformer's own geometry, not
        // from the canvas: the family's peak site is a single block's fused
        // `3h + 2 x mlp` projection, which at [dev]'s 6144-wide stream is
        // 165,888 elements per joint token — a working set the FLUX.1 area
        // factor above under-charges by an order of magnitude (#1707). The
        // reference tail below scales these same bytes, so the base is the
        // no-reference sequence.
        mold_inference::device::flux2_denoise_activation_bytes_for_canvas(
            geometry,
            req.width,
            req.height,
            hint.map_or(1, |h| h.batch),
            hint.map_or(2, |h| h.dtype_bytes),
            mold_inference::device::flux_effective_attention_backend(),
        )
    } else if hint.is_some_and(|h| h.family.streaming_transformer()) {
        // Cold-cache fallback: no checkpoint header has been read here, so the
        // AdaLN width is unknown and falls back to the six-component default.
        // The authoritative path in `estimate_generation_memory_for_request`
        // passes the checkpoint's real width.
        crate::ltx2_admission::ltx2_activation_bytes(
            crate::ltx2_admission::Ltx2ShapeHint::from_request_with_projection(req, projection),
            None,
        )
    } else {
        activation_memory_for_estimate(hint, qwen_quantized)
    };

    let mut activation = base.saturating_mul(batch).saturating_mul(cfg_factor);

    if !wan && hint.is_some_and(|h| h.family == ActivationFamily::Flux2Dit) {
        let request_images = req.edit_images.as_ref().filter(|images| !images.is_empty());
        let projected_images = projection
            .map(|projection| projection.edit_images.as_slice())
            .filter(|images| !images.is_empty());
        let image_count =
            request_images.map_or_else(|| projected_images.map_or(0, <[_]>::len), Vec::len);
        if image_count > 0 {
            let target_pixels = u64::from(req.width)
                .saturating_mul(u64::from(req.height))
                .max(1);
            let per_image_cap = if image_count == 1 {
                mold_core::validation::FLUX2_SINGLE_REFERENCE_MAX_PIXELS
            } else {
                mold_core::validation::FLUX2_MULTI_REFERENCE_MAX_PIXELS
            };
            // Reference bytes are already part of the finalized request, so
            // plan against their real dimensions. Falling back to the full
            // preprocessing cap on an unreadable header remains fail-closed.
            let reference_pixels = if let Some(images) = request_images {
                images.iter().fold(0u64, |total, bytes| {
                    let pixels = image::ImageReader::new(std::io::Cursor::new(bytes))
                        .with_guessed_format()
                        .ok()
                        .and_then(|reader| reader.into_dimensions().ok())
                        .map(|(width, height)| {
                            u64::from(width)
                                .saturating_mul(u64::from(height))
                                .min(per_image_cap)
                        })
                        .unwrap_or(per_image_cap);
                    total.saturating_add(pixels)
                })
            } else {
                projected_images
                    .into_iter()
                    .flatten()
                    .fold(0u64, |total, dimensions| {
                        use crate::queue_media_store::ProjectedImageDimensions;
                        let pixels = match dimensions {
                            ProjectedImageDimensions::Known { width, height } => u64::from(*width)
                                .saturating_mul(u64::from(*height))
                                .min(per_image_cap),
                            ProjectedImageDimensions::UnreadableHeader => per_image_cap,
                        };
                        total.saturating_add(pixels)
                    })
            };
            // The engine's own answer, so admission plans against exactly the
            // bytes `flux2/pipeline.rs` then reserves — there the ratio is
            // taken over real packed TOKEN counts, here over pixels, which is
            // the same number because both sides patchify at the same rate.
            activation = mold_inference::device::flux2_reference_scaled_activation_bytes(
                activation,
                target_pixels,
                reference_pixels,
            );
        }
    }

    let pixel_bytes = u64::from(req.width)
        .saturating_mul(u64::from(req.height))
        .saturating_mul(4);
    if req.source_image.is_some()
        || req
            .edit_images
            .as_ref()
            .is_some_and(|images| !images.is_empty())
        || projection
            .is_some_and(|projection| projection.source_image || !projection.edit_images.is_empty())
    {
        activation = activation.saturating_add(pixel_bytes.saturating_mul(batch));
    }
    if req.mask_image.is_some() || projection.is_some_and(|projection| projection.mask_image) {
        activation = activation.saturating_add(pixel_bytes / 2);
    }
    if req.control_image.is_some()
        || projection.is_some_and(|projection| projection.control_image)
        || req.control_model.as_deref().is_some_and(|m| !m.is_empty())
    {
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
    use base64::Engine as _;
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
    use mold_inference::wan::block_offload::AdmissionPolicy;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    fn offload(wan: AdmissionPolicy) -> GenerationOffloadPolicy {
        GenerationOffloadPolicy::new(false, wan, false)
    }

    fn metal_offload(wan: AdmissionPolicy) -> GenerationOffloadPolicy {
        GenerationOffloadPolicy::new(false, wan, true)
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = ImageBuffer::from_pixel(width, height, Rgb([1u8, 2, 3]));
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    #[test]
    fn authenticated_projection_matches_hydrated_media_memory_facts() {
        use crate::queue_media_store::{ProjectedImageDimensions, QueueMediaProjection};

        let image = png(320, 240);
        let mut hydrated: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a portrait",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 768,
            "steps": 20,
            "guidance": 3.5,
            "id_image": base64::engine::general_purpose::STANDARD.encode(&image),
            "id_weight": 0.8,
            "true_cfg": 2.0,
            "source_image": base64::engine::general_purpose::STANDARD.encode(&image),
            "edit_images": [base64::engine::general_purpose::STANDARD.encode(&image)],
            "mask_image": base64::engine::general_purpose::STANDARD.encode(&image),
            "control_image": base64::engine::general_purpose::STANDARD.encode(&image)
        }))
        .unwrap();
        let projection = QueueMediaProjection {
            source_image: true,
            identity_present: true,
            identity_photograph_count: 1,
            edit_image_count: 1,
            edit_images: vec![ProjectedImageDimensions::Known {
                width: 320,
                height: 240,
            }],
            mask_image: true,
            control_image: true,
            ..QueueMediaProjection::default()
        };
        let mut sanitized = hydrated.clone();
        sanitized.source_image = None;
        sanitized.id_image = None;
        sanitized.edit_images = None;
        sanitized.mask_image = None;
        sanitized.control_image = None;

        let hint = Some(hint(ActivationFamily::Flux2Dit));
        assert_eq!(
            request_sensitive_activation_memory_with_wan_geometry(
                &hydrated, hint, false, None, false, None, None,
            ),
            request_sensitive_activation_memory_with_wan_geometry(
                &sanitized,
                hint,
                false,
                None,
                false,
                None,
                Some(&projection),
            )
        );
        assert_eq!(
            identity_overhead_family_with_projection(&hydrated, None),
            identity_overhead_family_with_projection(&sanitized, Some(&projection)),
        );
        assert_eq!(
            request_charges_true_cfg_overhead_with_projection(&hydrated, None),
            request_charges_true_cfg_overhead_with_projection(&sanitized, Some(&projection)),
        );

        // An unreadable header uses the same fail-closed preprocessing cap as
        // an unreadable hydrated image.
        hydrated.edit_images = Some(vec![b"not-an-image".to_vec()]);
        let mut unreadable = projection;
        unreadable.edit_images = vec![ProjectedImageDimensions::UnreadableHeader];
        assert_eq!(
            request_sensitive_activation_memory_with_wan_geometry(
                &hydrated, hint, false, None, false, None, None,
            ),
            request_sensitive_activation_memory_with_wan_geometry(
                &sanitized,
                hint,
                false,
                None,
                false,
                None,
                Some(&unreadable),
            )
        );
    }

    #[test]
    fn frozen_flux_offload_plan_recheck_accepts_fresh_and_parked_budgets() {
        let fresh_available = 24_500_000_000;
        let parked_free = 1_500_000_000u64;
        let reclaimable_active = 23_000_000_000u64;
        let frozen_offload_peak = 8_272_629_760;
        let legacy_resident_peak = 26_100_000_000;

        assert!(
            check_model_memory_budget("cv:2925935", legacy_resident_peak, fresh_available, "")
                .is_err(),
            "the path-based resident estimate reproduces the false rejection"
        );
        assert!(
            check_planned_memory_budget(
                "cv:2925935",
                frozen_offload_peak,
                fresh_available,
                None,
                rejection_suggestion(None),
            )
            .is_ok(),
            "a fresh worker must retain the scheduler's admitted block-offload peak"
        );
        assert!(
            check_planned_memory_budget(
                "cv:2925935",
                frozen_offload_peak,
                parked_free.saturating_add(reclaimable_active),
                None,
                rejection_suggestion(None),
            )
            .is_ok(),
            "a parked reload must count the active engine that is about to be dropped"
        );
    }

    #[test]
    fn frozen_plan_recheck_fails_closed_when_physical_memory_drops_below_peak() {
        let error = check_planned_memory_budget(
            "planned",
            8_300_000_000,
            8_200_000_000,
            // A card with capacity to spare: the plan lost memory to something
            // else, which is the one case the pressure wording is true of.
            Some(24_000_000_000),
            rejection_suggestion(None),
        )
        .expect_err("new pressure after admission must reject the frozen plan");

        assert!(error.error.contains("frozen execution plan peak"));
        assert!(error
            .error
            .contains("memory pressure changed after scheduler admission"));
    }

    #[test]
    fn frozen_hot_cache_plan_counts_reused_resident_footprint_but_not_stale_capacity() {
        let resident_vram = 6_000_000_000;
        let activation_and_workspace = 2_300_000_000;
        let frozen_peak = resident_vram + activation_and_workspace;

        assert!(check_planned_memory_budget_with_resident(
            "hot-cache",
            frozen_peak,
            activation_and_workspace,
            resident_vram,
            None,
            rejection_suggestion(None),
        )
        .is_ok());
        assert!(check_planned_memory_budget_with_resident(
            "hot-cache",
            frozen_peak,
            activation_and_workspace - 1,
            resident_vram,
            None,
            rejection_suggestion(None),
        )
        .is_err());
    }

    fn hint(family: ActivationFamily) -> ActivationHint {
        ActivationHint {
            width: 1024,
            height: 1024,
            batch: 1,
            dtype_bytes: 2,
            family,
        }
    }

    fn paths(transformer: &str) -> ModelPaths {
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: PathBuf::from(transformer),
            transformer_shards: Vec::new(),
            vae: PathBuf::from("/models/vae.safetensors"),
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
        }
    }

    fn write_sparse_wan_1_3b_header(path: &Path, target_size: u64) {
        use std::io::{Seek, SeekFrom, Write};

        let mut shapes = vec![
            (
                "patch_embedding.weight".to_string(),
                vec![1536, 16, 1, 2, 2],
            ),
            ("blocks.0.ffn.0.weight".to_string(), vec![8960, 1536]),
            ("text_embedding.0.weight".to_string(), vec![1536, 4096]),
            ("time_embedding.0.weight".to_string(), vec![1536, 256]),
            ("head.head.weight".to_string(), vec![64, 1536]),
            ("blocks.0.self_attn.q.weight".to_string(), vec![1536, 1536]),
        ];
        for layer in 0..30 {
            shapes.push((format!("blocks.{layer}.modulation"), vec![1, 6, 1536]));
        }

        let probe_bytes = shapes
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>() as u64)
            .sum::<u64>();
        let mut dummy_bytes = target_size - probe_bytes - 4096;
        let (header, data_bytes) = loop {
            let mut offset = 0u64;
            let mut entries = serde_json::Map::new();
            for (name, shape) in &shapes {
                let bytes = shape.iter().product::<usize>() as u64;
                entries.insert(
                    name.clone(),
                    serde_json::json!({
                        "dtype": "U8",
                        "shape": shape,
                        "data_offsets": [offset, offset + bytes]
                    }),
                );
                offset += bytes;
            }
            entries.insert(
                "qualification.padding".to_string(),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": [dummy_bytes],
                    "data_offsets": [offset, offset + dummy_bytes]
                }),
            );
            let mut header = serde_json::to_vec(&serde_json::Value::Object(entries)).unwrap();
            header.resize(header.len().next_multiple_of(8), b' ');
            let revised = target_size - 8 - header.len() as u64 - probe_bytes;
            if revised == dummy_bytes {
                break (header, probe_bytes + dummy_bytes);
            }
            dummy_bytes = revised;
        };
        assert_eq!(8 + header.len() as u64 + data_bytes, target_size);
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.seek(SeekFrom::Start(target_size - 1)).unwrap();
        file.write_all(&[0]).unwrap();
    }

    #[test]
    fn forced_flux2_offload_never_applies_streaming_admission_to_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let transformer = dir.path().join("opaque.gguf");
        std::fs::File::create(&transformer)
            .unwrap()
            .set_len(30_000_000_000)
            .unwrap();
        let model_paths = paths(transformer.to_str().unwrap());

        assert!(preflight_memory_guard_with_available_and_policy(
            "opaque-flux2-gguf",
            &model_paths,
            0,
            10_000_000_000,
            Some(hint(ActivationFamily::Flux2Dit)),
            true,
            false,
        )
        .is_err());
    }

    /// plato's `flux2-dev:q8` install, byte for byte: the Q8_0 GGUF, the shared
    /// FLUX.2 VAE and the BF16 Mistral3 conditioner.
    fn flux2_dev_paths(dir: &Path, transformer_name: &str, transformer_bytes: u64) -> ModelPaths {
        let transformer = dir.join(transformer_name);
        std::fs::File::create(&transformer)
            .unwrap()
            .set_len(transformer_bytes)
            .unwrap();
        let vae = dir.join("flux2-vae.safetensors");
        std::fs::File::create(&vae)
            .unwrap()
            .set_len(336_213_556)
            .unwrap();
        let text_encoder = dir.join("mistral_3_small_flux2_bf16.safetensors");
        std::fs::File::create(&text_encoder)
            .unwrap()
            .set_len(35_584_897_447)
            .unwrap();
        let mut model_paths = paths(transformer.to_str().unwrap());
        model_paths.transformer = transformer;
        model_paths.vae = vae;
        model_paths.text_encoder_files = vec![text_encoder];
        model_paths
    }

    fn flux2_dev_request(edit_images: Option<Vec<String>>) -> GenerateRequest {
        let mut value = serde_json::json!({
            "prompt": "a photorealistic portrait of an astronaut on a beach at golden hour",
            "model": "flux2-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 50,
            "batch_size": 1
        });
        if let Some(images) = edit_images {
            value["edit_images"] = serde_json::json!(images);
        }
        serde_json::from_value(value).unwrap()
    }

    fn flux2_budget(
        req: &GenerateRequest,
        model_paths: &ModelPaths,
        available: u64,
    ) -> GenerationMemoryBudget {
        estimate_generation_memory_for_request(
            req,
            model_paths,
            Some(hint(ActivationFamily::Flux2Dit)),
            offload(AdmissionPolicy::Disabled),
            Some(available),
            false,
            false,
        )
    }

    /// plato (4x L40S, #1707). `flux2-dev:q8` at 1024x1024 / batch 1 measured a
    /// **38,359,007,232 byte** process VRAM high water on the BF16 GGUF path
    /// (`scheduler_estimates`, 2026-09-12 02:23:48Z) and **42,515,562,496** on
    /// the F32 one production still runs. Admission planned ~37.6 GB for both,
    /// because the activation term was FLUX.1's pixel-area factor — 273 MB for
    /// a working set three quantizations independently measure at ~3.0 GB.
    ///
    /// The plan must cover the measured peak and still fit the card, because
    /// nine prints at this exact shape completed on this machine.
    #[test]
    fn flux2_dev_q8_is_planned_above_the_peak_plato_measured() {
        const PLATO_L40S_AVAILABLE_BYTES: u64 = 46_100_000_000;
        const MEASURED_HIGH_WATER_BYTES: u64 = 38_359_007_232;
        let dir = tempfile::tempdir().unwrap();
        let model_paths = flux2_dev_paths(dir.path(), "flux2-dev-Q8_0.gguf", 35_002_602_464);
        let budget = flux2_budget(
            &flux2_dev_request(None),
            &model_paths,
            PLATO_L40S_AVAILABLE_BYTES,
        );

        assert!(
            budget.peak_memory_bytes >= MEASURED_HIGH_WATER_BYTES,
            "planned {} must cover the {} plato measured",
            budget.peak_memory_bytes,
            MEASURED_HIGH_WATER_BYTES
        );
        assert_eq!(
            budget.fits_available_memory,
            Some(true),
            "the nine completed prints at this shape must stay admissible (planned {})",
            budget.peak_memory_bytes
        );
        // And on the arm plato itself ran, which is the arm the 3.0 GB denoise
        // charge was fitted to: 40.34 GB planned against a 46.1 GB card.
        let flash = platos_flash_arm_peak(budget.peak_memory_bytes, 0);
        assert!(
            flash >= MEASURED_HIGH_WATER_BYTES,
            "the flash-arm plan {flash} must cover the {MEASURED_HIGH_WATER_BYTES} plato measured"
        );
        assert!(
            flash <= PLATO_L40S_AVAILABLE_BYTES * 9 / 10,
            "the flash-arm plan {flash} must stay admissible on plato's own card"
        );
    }

    /// The same tier and canvas with ONE reference image is the shape that
    /// OOM'd: its failed attempts recorded a 46,554,677,248 byte high water on
    /// a card with ~46.1 GB usable. References are appended to the packed
    /// sequence, so the denoise working set scales with them — admission has to
    /// refuse before the load rather than two minutes into the denoise.
    #[test]
    fn flux2_dev_q8_with_a_reference_no_longer_fits_platos_card() {
        const PLATO_L40S_AVAILABLE_BYTES: u64 = 46_100_000_000;
        let dir = tempfile::tempdir().unwrap();
        let model_paths = flux2_dev_paths(dir.path(), "flux2-dev-Q8_0.gguf", 35_002_602_464);
        let reference = base64::engine::general_purpose::STANDARD.encode(png(1024, 1024));
        let budget = flux2_budget(
            &flux2_dev_request(Some(vec![reference])),
            &model_paths,
            PLATO_L40S_AVAILABLE_BYTES,
        );

        assert_eq!(
            budget.fits_available_memory,
            Some(false),
            "a reference render whose measured high water was 46.55 GB must be \
             refused on a 46.1 GB card, not admitted (planned {})",
            budget.peak_memory_bytes
        );
        // The refusal is what the flash arm — plato's own — produces too:
        // 43.34 GB against a 41.49 GB admissible ceiling.
        let flash = platos_flash_arm_peak(budget.peak_memory_bytes, 1);
        assert!(
            flash > PLATO_L40S_AVAILABLE_BYTES * 9 / 10,
            "the flash-arm plan {flash} must still be refused on plato's card"
        );
    }

    /// The plan plato's own build produced, derived from the plan THIS build
    /// produced.
    ///
    /// Every figure these fixtures pin was measured on plato, whose log for
    /// those renders reports `fast_still_default=Flash`; the planner asks
    /// `flux_effective_attention_backend()`, so plato planned on the FLASH
    /// arm. A test binary built without the flash kernels resolves to math,
    /// which charges one score pair more for the same stream. On this path the
    /// peak is `base_peak + activation` and neither tier's residency decision
    /// moves between the arms, so the difference between the two plans is
    /// exactly the difference between the two activation figures. Returns the
    /// plan unchanged on a flash-compiled build.
    fn platos_flash_arm_peak(peak: u64, references: u64) -> u64 {
        use mold_inference::attention::AttentionBackend;
        use mold_inference::device::{
            flux2_denoise_activation_bytes_for_canvas, flux2_reference_scaled_activation_bytes,
            flux_effective_attention_backend, Flux2ActivationGeometry,
        };
        const PIXELS: u64 = 1024 * 1024;
        let arm = |backend| {
            let base = flux2_denoise_activation_bytes_for_canvas(
                Flux2ActivationGeometry::dev(),
                1024,
                1024,
                1,
                2,
                backend,
            );
            flux2_reference_scaled_activation_bytes(base, PIXELS, references * PIXELS)
        };
        peak - (arm(flux_effective_attention_backend()) - arm(AttentionBackend::Flash))
    }

    /// The fp8 tier renders at this shape today and must keep doing so. Its own
    /// measured high water on the BF16 path is 39,701,184,512 bytes — higher
    /// than q8's, because the per-forward widen holds a working-dtype copy of
    /// one linear beside the stream — and that difference is absorbed by
    /// `MEMORY_BUDGET_HEADROOM` rather than charged a second time.
    #[test]
    fn flux2_dev_fp8_keeps_its_admission_decision_at_platos_shape() {
        const PLATO_L40S_AVAILABLE_BYTES: u64 = 46_100_000_000;
        const MEASURED_HIGH_WATER_BYTES: u64 = 39_701_184_512;
        let dir = tempfile::tempdir().unwrap();
        let model_paths =
            flux2_dev_paths(dir.path(), "flux2_dev_fp8mixed.safetensors", 35_455_599_592);
        let mut req = flux2_dev_request(None);
        req.model = "flux2-dev:fp8".to_string();
        let budget = flux2_budget(&req, &model_paths, PLATO_L40S_AVAILABLE_BYTES);

        assert!(
            budget.peak_memory_bytes >= MEASURED_HIGH_WATER_BYTES,
            "planned {} must cover the {} plato measured",
            budget.peak_memory_bytes,
            MEASURED_HIGH_WATER_BYTES
        );
        assert_eq!(budget.fits_available_memory, Some(true));
        // The decision that must not move is the one plato's own arm makes:
        // 40.79 GB planned, resident, admitted.
        let flash = platos_flash_arm_peak(budget.peak_memory_bytes, 0);
        assert!(
            flash >= MEASURED_HIGH_WATER_BYTES,
            "the flash-arm plan {flash} must cover the {MEASURED_HIGH_WATER_BYTES} plato measured"
        );
        assert!(
            flash <= PLATO_L40S_AVAILABLE_BYTES * 9 / 10,
            "the flash-arm plan {flash} must keep the fp8 tier admissible"
        );
        assert!(
            !budget.block_offload,
            "the fp8 tier must stay resident on this card, not stream its blocks"
        );
    }

    /// The correction is the FLUX.2 family's alone. A FLUX.1 render at the same
    /// canvas keeps the pixel-area number it has always been planned with.
    #[test]
    fn the_flux2_denoise_charge_never_reaches_flux1() {
        let dir = tempfile::tempdir().unwrap();
        let transformer = dir.path().join("flux1-dev-Q8_0.gguf");
        std::fs::File::create(&transformer)
            .unwrap()
            .set_len(12_000_000_000)
            .unwrap();
        let mut model_paths = paths(transformer.to_str().unwrap());
        model_paths.transformer = transformer;
        let req: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "batch_size": 1
        }))
        .unwrap();
        let budget = estimate_generation_memory_for_request(
            &req,
            &model_paths,
            Some(hint(ActivationFamily::FluxDit)),
            offload(AdmissionPolicy::Disabled),
            Some(24_000_000_000),
            false,
            false,
        );
        assert_eq!(
            budget.activation_memory_bytes,
            mold_inference::device::activation_bytes(1024, 1024, 1, 2, ActivationFamily::FluxDit)
        );
    }

    /// A 24 GB card, the shape the 2026-09-11 audit simulated with
    /// `MOLD_RESERVE_VRAM_MB=22000`.
    ///
    /// The audit's finding was that both FLUX.2 [dev] tiers were refused BY THE
    /// LOADER with "memory pressure changed after scheduler admission" on a
    /// card nothing else was using. Two things were wrong: the scheduler and
    /// the loader were reading different budgets, and nothing said that a GGUF
    /// dev tier has no streamed path at all.
    ///
    /// What is true, and is now pinned: an fp8/BF16 dev tier does not fit
    /// resident and STREAMS, so it is admitted; a GGUF tier cannot stream, so
    /// it is refused — with the engine's own reason, at admission.
    #[test]
    fn on_a_24gb_card_a_streamable_dev_tier_is_admitted_and_a_gguf_one_is_not() {
        const CARD_24GB_USABLE_BYTES: u64 = 23_600_000_000;
        let dir = tempfile::tempdir().unwrap();

        let streamable =
            flux2_dev_paths(dir.path(), "flux2_dev_fp8mixed.safetensors", 35_455_599_592);
        let mut req = flux2_dev_request(None);
        req.model = "flux2-dev:fp8".to_string();
        let budget = flux2_budget(&req, &streamable, CARD_24GB_USABLE_BYTES);
        assert!(
            budget.block_offload,
            "a dev tier that cannot be resident on 24 GB must stream its blocks"
        );
        assert_eq!(
            budget.fits_available_memory,
            Some(true),
            "the streamed working set fits a 24 GB card (planned {})",
            budget.peak_memory_bytes
        );
        assert!(
            flux2_block_offload_unsupported_reason(&streamable, false).is_none(),
            "the engine streams this checkpoint, so the planner must not claim otherwise"
        );

        let quantized_dir = tempfile::tempdir().unwrap();
        let gguf = flux2_dev_paths(
            quantized_dir.path(),
            "flux2-dev-Q4_K_M.gguf",
            19_959_731_168,
        );
        let mut req = flux2_dev_request(None);
        req.model = "flux2-dev:q4".to_string();
        let budget = flux2_budget(&req, &gguf, CARD_24GB_USABLE_BYTES);
        assert!(
            !budget.block_offload,
            "the FLUX.2 engine has no block-streaming path for GGUF; admission \
             must never promise one"
        );
        assert_eq!(
            budget.fits_available_memory,
            Some(false),
            "20.3 GB of resident weights plus a 3.0 GB denoise cannot fit a 24 GB \
             card, and saying so at admission is the whole point (planned {})",
            budget.peak_memory_bytes
        );
        let reason = flux2_block_offload_unsupported_reason(&gguf, false)
            .expect("a GGUF dev tier must carry the engine's own refusal reason");
        assert!(
            reason.contains("GGUF"),
            "the refusal must name the format, not blame memory pressure: {reason}"
        );
    }

    /// The card each FLUX.2 [dev] GGUF tier actually needs, which is what the
    /// manifest description and `website/models/flux2.md` now say.
    ///
    /// A GGUF tier has no block-streaming path, so every byte of it is
    /// resident and the planner adds the ~3.0 GB denoise working set #1707
    /// measured, the VAE, and the 2 GB budget headroom on top of the
    /// checkpoint. `flux2-dev:q4` was documented as running on a 24 GB GPU
    /// and `:q6` as fitting a 32 GB one on the strength of the checkpoint
    /// size alone; neither is admitted there. The figures below are the
    /// planner's own, so a doc claim and the refusal can never disagree
    /// again.
    #[test]
    fn the_dev_gguf_tiers_need_the_card_the_docs_now_name() {
        // usable bytes after the 90% cap and the default reserve, for the
        // card class named in each doc row.
        const CARD_24GB_USABLE_BYTES: u64 = 23_600_000_000;
        const CARD_32GB_USABLE_BYTES: u64 = 31_500_000_000;
        const CARD_40GB_USABLE_BYTES: u64 = 39_500_000_000;
        const CARD_48GB_USABLE_BYTES: u64 = 47_500_000_000;

        let plan = |file: &str, bytes: u64, model: &str, available: u64| {
            let dir = tempfile::tempdir().unwrap();
            let model_paths = flux2_dev_paths(dir.path(), file, bytes);
            let mut req = flux2_dev_request(None);
            req.model = model.to_string();
            let budget = flux2_budget(&req, &model_paths, available);
            assert!(
                !budget.block_offload,
                "{model} is GGUF: the engine has no streamed path for it"
            );
            (budget.fits_available_memory, budget.peak_memory_bytes)
        };

        // q4 — the "runs on a 24 GB GPU" claim. It does not.
        let (fits_24, q4_peak) = plan(
            "flux2-dev-Q4_K_M.gguf",
            19_959_731_168,
            "flux2-dev:q4",
            CARD_24GB_USABLE_BYTES,
        );
        assert_eq!(
            fits_24,
            Some(false),
            "flux2-dev:q4 plans {q4_peak} bytes, which a 24 GB card cannot hold"
        );
        assert!(
            q4_peak > 24_000_000_000,
            "the doc says ~25 GB; the planner says {q4_peak}"
        );
        let (fits_32, _) = plan(
            "flux2-dev-Q4_K_M.gguf",
            19_959_731_168,
            "flux2-dev:q4",
            CARD_32GB_USABLE_BYTES,
        );
        assert_eq!(
            fits_32,
            Some(true),
            "a 32 GB card is the smallest the docs may name for q4"
        );

        // q6 — the "fits a 32 GB GPU with room for activations" claim. The
        // activations are exactly what it has no room for.
        let (fits_32, q6_peak) = plan(
            "flux2-dev-Q6_K.gguf",
            27_396_232_160,
            "flux2-dev:q6",
            CARD_32GB_USABLE_BYTES,
        );
        assert_eq!(
            fits_32,
            Some(false),
            "flux2-dev:q6 plans {q6_peak} bytes, which a 32 GB card cannot hold"
        );
        let (fits_40, _) = plan(
            "flux2-dev-Q6_K.gguf",
            27_396_232_160,
            "flux2-dev:q6",
            CARD_40GB_USABLE_BYTES,
        );
        assert_eq!(
            fits_40,
            Some(true),
            "a 40 GB card is the smallest the docs may name for q6"
        );

        // q8 — the tier the campaign measured on plato's 46 GB L40S.
        let (fits_40, q8_peak) = plan(
            "flux2-dev-Q8_0.gguf",
            35_002_602_464,
            "flux2-dev:q8",
            CARD_40GB_USABLE_BYTES,
        );
        assert_eq!(
            fits_40,
            Some(false),
            "flux2-dev:q8 plans {q8_peak} bytes, past a 40 GB card"
        );
        let (fits_48, _) = plan(
            "flux2-dev-Q8_0.gguf",
            35_002_602_464,
            "flux2-dev:q8",
            CARD_48GB_USABLE_BYTES,
        );
        assert_eq!(fits_48, Some(true), "q8 is a 46/48 GB-class tier");
    }

    /// One budget, two readers. Admission planned against the raw free sample
    /// while every pre-load gate reads `free - reserved_vram_bytes()`, so a
    /// plan admitted inside the reserve was refused at load.
    #[test]
    fn admission_and_the_loader_read_the_same_reserve_adjusted_budget() {
        use crate::scheduler::schedulable_available_vram_bytes_with_reserve;
        const TOTAL: u64 = 48_000_000_000;
        const FREE: u64 = 46_500_000_000;
        let budget = |reclaimable, reserve| {
            schedulable_available_vram_bytes_with_reserve(
                FREE,
                reclaimable,
                None,
                false,
                TOTAL,
                reserve,
            )
        };

        // The Linux default. A plan sized to the raw sample sits inside the
        // reserve the loader will subtract.
        assert_eq!(budget(0, 400_000_000), 46_100_000_000);
        // The audit's 24 GB simulation: 22,000 MB reserved, and the two
        // budgets differed by all of it.
        assert_eq!(budget(0, 22_000_000_000), 24_500_000_000);
        // Reclaimable cache is still credited, and the total is still the cap
        // before the reserve comes off.
        assert_eq!(budget(40_000_000_000, 400_000_000), TOTAL - 400_000_000);
        // A reserve larger than what is free leaves nothing, never an
        // underflow.
        assert_eq!(
            schedulable_available_vram_bytes_with_reserve(
                1_000_000,
                0,
                None,
                false,
                TOTAL,
                400_000_000
            ),
            0
        );
    }

    #[test]
    fn flux2_and_zimage_loras_force_sequential_engine_plans() {
        for family in [ActivationFamily::Flux2Dit, ActivationFamily::ZImageDit] {
            assert_eq!(
                request_aware_load_strategy(
                    mold_inference::LoadStrategy::Eager,
                    &paths("/models/opaque/model.safetensors"),
                    Some(hint(family)),
                    true,
                    false,
                ),
                mold_inference::LoadStrategy::Sequential
            );
        }
        assert_eq!(
            request_aware_load_strategy(
                mold_inference::LoadStrategy::Eager,
                &paths("/models/opaque/model.safetensors"),
                Some(hint(ActivationFamily::Flux2Dit)),
                false,
                false,
            ),
            mold_inference::LoadStrategy::Eager
        );
        assert_eq!(
            request_aware_load_strategy(
                mold_inference::LoadStrategy::Eager,
                &paths("/models/opaque/model.safetensors"),
                Some(hint(ActivationFamily::FluxDit)),
                true,
                false,
            ),
            mold_inference::LoadStrategy::Eager
        );
    }

    #[test]
    fn flux2_and_zimage_lora_paths_force_sequential_without_family_hint() {
        for transformer in [
            "/models/cv-opaque/flux2-klein.safetensors",
            "/models/cv-opaque/z-image/model.safetensors",
        ] {
            assert_eq!(
                request_aware_load_strategy(
                    mold_inference::LoadStrategy::Eager,
                    &paths(transformer),
                    None,
                    true,
                    false,
                ),
                mold_inference::LoadStrategy::Sequential
            );
        }
    }

    /// The paths as they are actually installed, because that is what the
    /// matcher reads. A Klein GGUF tier's FILE is `flux-2-klein-4b-Q8_0.gguf`
    /// — with a hyphen, so the family token lives only in the model
    /// directory — and a plan this misses is an Eager engine for a LoRA
    /// request, which is what the flux2-klein + LoRA row of the perf UAT ran
    /// into on a 46 GB card.
    #[test]
    fn installed_klein_tier_paths_force_sequential_for_lora_requests() {
        for transformer in [
            "/storage/mold/models/flux2-klein-q8/flux-2-klein-4b-Q8_0.gguf",
            "/storage/mold/models/flux2-klein-9b-fp8/transformer.safetensors",
            "/storage/mold/models/flux2-klein-base-bf16/transformer.safetensors",
            "/storage/mold/models/flux2-dev-q8/flux-2-dev-Q8_0.gguf",
        ] {
            assert_eq!(
                request_aware_load_strategy(
                    mold_inference::LoadStrategy::Eager,
                    &paths(transformer),
                    None,
                    true,
                    false,
                ),
                mold_inference::LoadStrategy::Sequential,
                "{transformer} carries a LoRA and must be planned sequentially"
            );
        }
    }

    #[test]
    fn flux2_source_images_force_sequential_engine_plans() {
        assert_eq!(
            request_aware_load_strategy(
                mold_inference::LoadStrategy::Eager,
                &paths("/models/cv-opaque/model.safetensors"),
                Some(hint(ActivationFamily::Flux2Dit)),
                false,
                true,
            ),
            mold_inference::LoadStrategy::Sequential
        );
        assert_eq!(
            request_aware_load_strategy(
                mold_inference::LoadStrategy::Eager,
                &paths("/models/cv-opaque/model.safetensors"),
                Some(hint(ActivationFamily::FluxDit)),
                false,
                true,
            ),
            mold_inference::LoadStrategy::Eager,
            "source images must not change unrelated family load policies"
        );
    }

    /// Klein's reference protocol takes the same VAE-only encode phase a
    /// source image does, so it takes the same plan. An Eager plan here is
    /// worse than merely wasteful: the engine routes sequentially anyway and
    /// `generate_inner` unloads the transformer the eager plan just loaded.
    #[test]
    fn klein_reference_images_force_sequential_engine_plans() {
        assert_eq!(
            request_aware_load_strategy(
                mold_inference::LoadStrategy::Eager,
                &paths("/models/cv-opaque/model.safetensors"),
                Some(hint(ActivationFamily::Flux2Dit)),
                false,
                true,
            ),
            mold_inference::LoadStrategy::Sequential
        );
        assert_eq!(
            request_aware_load_strategy(
                mold_inference::LoadStrategy::Eager,
                &paths("/models/cv-opaque/model.safetensors"),
                Some(hint(ActivationFamily::FluxDit)),
                false,
                true,
            ),
            mold_inference::LoadStrategy::Eager,
            "conditioning images must not change unrelated family load policies"
        );
    }

    /// The request's inline bytes and the durable queue's projection are two
    /// answers to the same question. A job whose media was already handed off
    /// carries its references only in the projection, and planning THAT one
    /// Eager is the half a source-image-only predicate would have missed.
    #[test]
    fn generation_memory_budget_carries_reference_images_into_flux2_load_policy() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "put sunglasses on the person",
            "model": "cv:test-klein-9b",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 1.0,
            "batch_size": 1
        }))
        .unwrap();
        let model_paths = paths("/models/cv-opaque/model.safetensors");
        let activation = Some(hint(ActivationFamily::Flux2Dit));

        let plain = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(64_000_000_000),
            false,
            false,
        );
        assert_eq!(plain.load_strategy, mold_inference::LoadStrategy::Eager);

        request.edit_images = Some(vec![vec![0x89, 0x50, 0x4e, 0x47]]);
        let inline = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(64_000_000_000),
            false,
            false,
        );
        assert_eq!(
            inline.load_strategy,
            mold_inference::LoadStrategy::Sequential
        );

        // An empty vector is not a reference request.
        request.edit_images = Some(Vec::new());
        let empty = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(64_000_000_000),
            false,
            false,
        );
        assert_eq!(empty.load_strategy, mold_inference::LoadStrategy::Eager);

        request.edit_images = None;
        let projection = crate::queue_media_store::QueueMediaProjection {
            edit_image_count: 1,
            edit_images: vec![crate::queue_media_store::ProjectedImageDimensions::Known {
                width: 1024,
                height: 1024,
            }],
            ..Default::default()
        };
        let projected = estimate_generation_memory_for_request_with_projection(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(64_000_000_000),
            false,
            false,
            Some(&projection),
        );
        assert_eq!(
            projected.load_strategy,
            mold_inference::LoadStrategy::Sequential,
            "a durable job whose references live only in the projection plans the same way"
        );
    }

    #[test]
    fn generation_memory_budget_carries_source_image_into_flux2_load_policy() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "portrait",
            "model": "cv:test-klein-9b",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 1.0,
            "batch_size": 1
        }))
        .unwrap();
        let model_paths = paths("/models/cv-opaque/model.safetensors");
        let activation = Some(hint(ActivationFamily::Flux2Dit));

        let plain = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(64_000_000_000),
            false,
            false,
        );
        assert_eq!(plain.load_strategy, mold_inference::LoadStrategy::Eager);

        request.source_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        let source = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(64_000_000_000),
            false,
            false,
        );
        assert_eq!(
            source.load_strategy,
            mold_inference::LoadStrategy::Sequential
        );
    }

    /// A face-identity render carries a resident cross-attention adapter the
    /// checkpoint's `ModelPaths` cannot describe, so the estimate has to charge
    /// it from the request. Weight zero applies no identity at all and must
    /// cost exactly nothing — an inert knob that still reserved 1.25 GB would
    /// refuse renders that fit.
    #[test]
    fn identity_conditioning_charges_exactly_its_named_overhead_and_weight_zero_charges_nothing() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "portrait",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 3.5,
            "batch_size": 1
        }))
        .unwrap();
        let model_paths = paths("/models/flux-dev/flux1-dev-Q8_0.gguf");
        let activation = Some(hint(ActivationFamily::FluxDit));
        let estimate = |request: &GenerateRequest| {
            estimate_generation_memory_for_request(
                request,
                &model_paths,
                activation,
                offload(AdmissionPolicy::Disabled),
                Some(24_000_000_000),
                false,
                false,
            )
            .peak_memory_bytes
        };

        let plain = estimate(&request);

        request.id_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        assert!(!request_charges_identity_overhead(&{
            let mut zero = request.clone();
            zero.id_weight = Some(0.0);
            zero
        }));
        assert_eq!(
            estimate(&{
                let mut zero = request.clone();
                zero.id_weight = Some(0.0);
                zero
            }),
            plain,
            "id_weight 0 applies no identity, so it must add no memory demand"
        );

        assert!(request_charges_identity_overhead(&request));
        // TWO named terms since #1227 phase 2: the adapter's residency for the
        // whole denoise, and the extraction phase that now runs on this same
        // leased device before the model loads. Summed rather than maxed —
        // see `IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES` for why a warm engine
        // cache makes the disjointness argument unavailable to admission.
        let identity_terms = IDENTITY_VRAM_OVERHEAD_BYTES + IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES;
        assert_eq!(
            estimate(&request),
            plain + identity_terms,
            "an identity render must be charged exactly the named overheads"
        );

        let mut weighted = request.clone();
        weighted.id_weight = Some(0.85);
        assert_eq!(estimate(&weighted), plain + identity_terms);
    }

    /// A true-CFG render runs a second forward per step over a second
    /// conditioning pair. It is charged its own named overhead ON TOP of the
    /// identity one, and — like `id_weight: 0` — an inert scale must cost
    /// exactly nothing.
    #[test]
    fn true_cfg_charges_its_own_overhead_and_an_inert_scale_charges_nothing() {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "portrait",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 3.5,
            "batch_size": 1
        }))
        .unwrap();
        let model_paths = paths("/models/flux-dev/flux1-dev-Q8_0.gguf");
        let activation = Some(hint(ActivationFamily::FluxDit));
        let estimate = |request: &GenerateRequest| {
            estimate_generation_memory_for_request(
                request,
                &model_paths,
                activation,
                offload(AdmissionPolicy::Disabled),
                Some(24_000_000_000),
                false,
                false,
            )
            .peak_memory_bytes
        };

        request.id_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        let identity_only = estimate(&request);
        assert!(!request_charges_true_cfg_overhead(&request));

        let mut inert = request.clone();
        inert.true_cfg = Some(1.0);
        assert!(!request_charges_true_cfg_overhead(&inert));
        assert_eq!(
            estimate(&inert),
            identity_only,
            "an inert scale runs the distilled path, so it must add no demand"
        );

        let mut branched = request.clone();
        branched.true_cfg = Some(2.5);
        assert!(request_charges_true_cfg_overhead(&branched));
        assert_eq!(
            estimate(&branched),
            identity_only + TRUE_CFG_VRAM_OVERHEAD_BYTES,
            "a true-CFG render must never be admitted on the plain identity estimate"
        );

        // A zero weight renders the plain print, so neither term is charged.
        let mut zero = branched.clone();
        zero.id_weight = Some(0.0);
        assert!(!request_charges_true_cfg_overhead(&zero));
        assert_eq!(
            estimate(&zero),
            identity_only - IDENTITY_VRAM_OVERHEAD_BYTES - IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES
        );
    }

    /// Wan never reaches `OffloadMode::Block` through the request-controlled
    /// streaming flag — the factory arm does not read it — so the plan has to
    /// derive the disposition from the engine's own predicate. It did not, and
    /// a render that parked half its blocks reported no offload at all and
    /// fingerprinted identically to one that parked nothing (#776).
    #[test]
    fn a_wan_render_that_will_park_names_block_offload_in_the_plan() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("Wan2.2-T2V-A14B-HighNoise-Q5_K_M.gguf");
        std::fs::File::create(&gguf)
            .unwrap()
            .set_len(10_790_416_896)
            .unwrap();
        let mut model_paths = paths("/unused/transformer.gguf");
        model_paths.transformer = gguf.clone();
        let activation = Some(hint(ActivationFamily::WanVideo));

        let request = |frames: u32| -> GenerateRequest {
            serde_json::from_value(serde_json::json!({
                "prompt": "a red sports car on a coast road",
                "model": "wan22-t2v-a14b:q5",
                "width": 832,
                "height": 480,
                "frames": frames,
                "steps": 4,
                "guidance": 1.0,
                "batch_size": 1
            }))
            .unwrap()
        };
        // Exercise pressure relative to the selected kernel's resident peak.
        // FlashAttention fits 81 frames in the old fixed 24 GB test budget.
        let device_budget = estimate_generation_memory_for_request(
            &request(81),
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            None,
            false,
            false,
        )
        .peak_memory_bytes
        .saturating_sub(1024 * 1024 * 1024);
        let budget = |req: &GenerateRequest, paths: &ModelPaths| {
            estimate_generation_memory_for_request(
                req,
                paths,
                activation,
                offload(AdmissionPolicy::Automatic),
                Some(device_budget),
                false,
                false,
            )
        };

        // At this budget 81 frames cannot stay resident; the plan must name parking.
        let parks = budget(&request(81), &model_paths);
        assert!(
            parks.wan_block_offload,
            "an 81-frame A14B render parks blocks, so the plan must name it"
        );
        // ...but only in the disposition. The generic flag additionally makes
        // `build_plan` reserve every transformer file in host RAM as
        // `StreamedBlocks`, which for the A14B pair is both experts at full
        // size — 22-31 GB against a runtime that holds one expert's parked
        // tail, and enough to reject a render whose real working set fits.
        assert!(
            !parks.block_offload,
            "wan parks a subset of one expert; it must not be charged as a \
             streamed transformer"
        );

        // Metal has no automatic parking plan because the runtime cannot take
        // the CUDA-ordinal free-VRAM reading it needs. Admission must retain
        // the resident weight term there instead of crediting CUDA-only
        // relief (#1060).
        let metal = estimate_generation_memory_for_request(
            &request(81),
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(device_budget),
            false,
            false,
        );
        assert_eq!(
            metal.peak_memory_bytes,
            parks
                .peak_memory_bytes
                .saturating_add(wan_block_offload_relief(&model_paths))
        );
        assert_eq!(metal.fits_available_memory, Some(false));
        assert!(!metal.wan_block_offload);

        let finite_override = estimate_generation_memory_for_request(
            &request(81),
            &model_paths,
            activation,
            offload(AdmissionPolicy::ForcedFinite(1)),
            Some(device_budget),
            false,
            false,
        );
        assert_eq!(finite_override.peak_memory_bytes, metal.peak_memory_bytes);
        assert_eq!(finite_override.fits_available_memory, Some(false));
        assert!(finite_override.wan_block_offload);

        let full_override = estimate_generation_memory_for_request(
            &request(81),
            &model_paths,
            activation,
            offload(AdmissionPolicy::ForcedFinite(40)),
            Some(device_budget),
            false,
            false,
        );
        assert_eq!(full_override.peak_memory_bytes, metal.peak_memory_bytes);
        assert_eq!(full_override.fits_available_memory, Some(false));
        assert!(full_override.wan_block_offload);
        let exact_relief = wan_block_offload_relief(&model_paths);
        assert_eq!(
            wan_block_offload_relief_for_policy(
                &model_paths,
                AdmissionPolicy::ForcedFinite(40),
                Some(40),
            ),
            exact_relief
        );
        assert_eq!(
            wan_block_offload_relief_for_policy(
                &model_paths,
                AdmissionPolicy::ForcedFinite(999),
                Some(40),
            ),
            exact_relief
        );
        assert_eq!(
            wan_block_offload_relief_for_policy(
                &model_paths,
                AdmissionPolicy::ForcedFinite(40),
                None,
            ),
            0
        );

        // A short clip fits with the weights resident and must keep exactly the
        // execution it had before any of this existed.
        let fits = budget(&request(17), &model_paths);
        assert!(
            !fits.wan_block_offload,
            "a render that fits resident must not claim an offload it will not do"
        );

        // The same shape on a checkpoint that cannot park: fp8 has no raw-byte
        // round trip, so claiming the disposition would be a promise the engine
        // cannot keep — and its envelope is what fits resident.
        let fp8 = dir
            .path()
            .join("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors");
        std::fs::File::create(&fp8)
            .unwrap()
            .set_len(14_300_000_000)
            .unwrap();
        let mut fp8_paths = model_paths.clone();
        fp8_paths.transformer = fp8;
        let fp8_budget = budget(&request(81), &fp8_paths);
        assert!(
            !fp8_budget.wan_block_offload && !fp8_budget.block_offload,
            "fp8 cannot park, so no shape may report block offload for it"
        );
    }

    /// At 6.5 GB the selected Q8 encoder cannot fit its 8.04 GB GPU threshold,
    /// so the prepared plan parks it on CPU and the device estimate contains
    /// only the 1.3B transformer, VAE, and denoise activation. The #1059 run
    /// reached 8,184,725,504 Metal bytes in that phase after admission had
    /// predicted 6.2 GB: Metal still needs the generic runtime allowance the
    /// CUDA-calibrated Wan path removes.
    #[test]
    fn wan_metal_keeps_runtime_headroom_after_the_encoder_parks() {
        let dir = tempfile::tempdir().unwrap();
        let transformer = dir.path().join("wan2.1_t2v_1.3B_bf16.safetensors");
        let vae = dir.path().join("wan_2.1_vae.safetensors");
        let umt5 = dir.path().join("umt5-xxl-encoder-Q8_0.gguf");
        write_sparse_wan_1_3b_header(&transformer, 2_838_104_528);
        std::fs::File::create(&vae)
            .unwrap()
            .set_len(253_815_318)
            .unwrap();
        std::fs::File::create(&umt5)
            .unwrap()
            .set_len(6_043_068_256)
            .unwrap();
        let mut model_paths = paths(transformer.to_str().unwrap());
        model_paths.vae = vae;
        model_paths.text_encoder_files = vec![umt5.clone()];
        assert_eq!(
            crate::wan_admission::warm_checkpoint_geometry(&model_paths),
            Some(mold_inference::device::WanActivationGeometry::t2v_1_3b())
        );
        assert!(
            mold_inference::device::t5_vram_threshold(6_043_068_256) > 6_500_000_000,
            "the incident boundary parks this prepared encoder on CPU"
        );
        let mut denoise_paths = model_paths.clone();
        denoise_paths.text_encoder_files.clear();
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a fox walking through snow",
            "model": "wan21-t2v-1.3b:bf16",
            "width": 512,
            "height": 288,
            "frames": 17,
            "steps": 30,
            "guidance": 6.0,
            "batch_size": 1
        }))
        .unwrap();

        let old_boundary = estimate_generation_memory_for_request(
            &request,
            &denoise_paths,
            Some(ActivationHint::from_request(&request, "wan")),
            metal_offload(mold_inference::wan::block_offload::AdmissionPolicy::Disabled),
            Some(6_500_000_000),
            false,
            false,
        );
        assert_eq!(old_boundary.fits_available_memory, Some(false));
        assert!(
            old_boundary.peak_memory_bytes > 8_184_725_504,
            "estimated {} bytes",
            old_boundary.peak_memory_bytes
        );

        let whole_request = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            Some(ActivationHint::from_request(&request, "wan")),
            metal_offload(mold_inference::wan::block_offload::AdmissionPolicy::Disabled),
            Some(6_500_000_000),
            false,
            false,
        );
        assert_eq!(whole_request.fits_available_memory, Some(false));
        assert_eq!(
            whole_request.peak_memory_bytes, old_boundary.peak_memory_bytes,
            "the encoder phase must not overlap the larger denoise phase"
        );

        let gpu_encoder = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            Some(ActivationHint::from_request(&request, "wan")),
            metal_offload(mold_inference::wan::block_offload::AdmissionPolicy::Disabled),
            Some(14_000_000_000),
            false,
            false,
        );
        assert_eq!(
            gpu_encoder.peak_memory_bytes,
            mold_inference::device::t5_metal_gguf_vram_threshold(6_043_068_256),
            "once UMT5 promotes to Metal, admission must cover its measured phase"
        );

        let cuda = estimate_generation_memory_for_request(
            &request,
            &denoise_paths,
            Some(ActivationHint::from_request(&request, "wan")),
            offload(mold_inference::wan::block_offload::AdmissionPolicy::Disabled),
            Some(6_500_000_000),
            false,
            false,
        );
        assert_eq!(
            old_boundary.peak_memory_bytes - cuda.peak_memory_bytes,
            WAN_REQUEST_AWARE_HEADROOM_BYTES + WAN_METAL_ALLOCATOR_MARGIN_BYTES,
            "the live correction must be Metal-only; CUDA keeps its calibrated estimate"
        );
    }

    #[test]
    fn large_flux2_bf16_auto_offloads_only_when_residency_does_not_fit() {
        let dir = tempfile::tempdir().unwrap();
        let mut model_paths = paths("/unused/first-shard.safetensors");
        model_paths.transformer_shards = (0..7)
            .map(|index| {
                let path = dir.path().join(format!("transformer-{index}.safetensors"));
                std::fs::File::create(&path)
                    .unwrap()
                    .set_len(9_000_000_000)
                    .unwrap();
                path
            })
            .collect();
        model_paths.transformer = model_paths.transformer_shards[0].clone();
        let activation = Some(hint(ActivationFamily::Flux2Dit));
        assert!(large_flux2_bf16_should_auto_offload(
            &model_paths,
            activation,
            Some(24_000_000_000),
            256_000_000,
        ));

        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "portrait",
            "model": "flux2-dev:bf16",
            "width": 1024,
            "height": 1024,
            "steps": 50,
            "guidance": 4.0,
            "batch_size": 1
        }))
        .unwrap();
        let plain = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(24_000_000_000),
            false,
            false,
        );
        assert!(plain.block_offload);
        assert_eq!(
            plain.load_strategy,
            mold_inference::LoadStrategy::Sequential
        );
        assert!(plain.peak_memory_bytes < 24_000_000_000);

        let resident_capable = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(96_000_000_000),
            false,
            false,
        );
        assert!(
            !resident_capable.block_offload,
            "a 96 GB GPU should keep FLUX.2 Dev resident instead of paying the streaming penalty"
        );

        request.edit_images = Some(vec![png(256, 256), png(256, 256)]);
        let with_references = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(24_000_000_000),
            false,
            false,
        );
        assert!(with_references.activation_memory_bytes > plain.activation_memory_bytes);
        assert!(with_references.fits_available_memory == Some(true));

        request.edit_images = Some(vec![vec![1], vec![2]]);
        let unreadable_references = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(24_000_000_000),
            false,
            false,
        );
        assert!(
            unreadable_references.activation_memory_bytes > with_references.activation_memory_bytes,
            "unreadable reference headers must retain the cap-based fail-closed estimate"
        );
    }

    /// The widen decision must be visible to the planner, because it is the
    /// planner that decides whether Qwen3 gets a GPU slot.
    ///
    /// On a 32 GB card a klein-9B fp8 checkpoint resolves `AtLoad` and holds
    /// two bytes per parameter. Priced from its file, the plan believed there
    /// was room for a bf16 Qwen3 that in reality could not fit — and the
    /// encoder-variant selector silently fell back to a Q8 GGUF or to the
    /// CPU, which is the F32 encode this campaign exists to remove.
    #[test]
    fn an_fp8_flux2_checkpoint_charges_the_widened_bytes_when_it_will_widen() {
        let dir = tempfile::tempdir().unwrap();
        let make = |name: &str, len: u64| {
            let path = dir.path().join(name);
            std::fs::File::create(&path).unwrap().set_len(len).unwrap();
            path
        };
        let activation = Some(hint(ActivationFamily::Flux2Dit));
        let with = |transformer: std::path::PathBuf| {
            let mut model_paths = paths("/unused/transformer.safetensors");
            model_paths.transformer = transformer;
            model_paths
        };

        // A 32 GB card affords three copies at load, so the engine widens and
        // the estimate must charge the second resident copy.
        let compact = with(make("flux2-klein-9b-fp8.safetensors", 9_079_000_000));
        let charged = flux2_fp8_widen_extra_bytes(&compact, activation, Some(33_600_000_000));
        assert!(
            charged > 0,
            "a card that will widen must be planned against the widened residency"
        );

        // THE DRIFT THIS FIXES: the charge is derived from the resolved
        // `Flux2Config`, exactly as the engine derives its gate — so a
        // `fp8mixed` checkpoint, whose FILE is materially larger because its
        // attention stays BF16, gets the same answer. When the server sized
        // this from the file it could say `PerForward` and charge nothing
        // where the engine said `AtLoad` and widened.
        let mixed = with(make("flux2-klein-9b-fp8mixed.safetensors", 14_000_000_000));
        assert_eq!(
            flux2_fp8_widen_extra_bytes(&mixed, activation, Some(33_600_000_000)),
            charged,
            "the same architecture must charge the same widen whatever its file weighs"
        );

        // A 24 GiB card cannot afford three copies, so nothing is added and
        // the estimate is what it was before this charge existed.
        assert_eq!(
            flux2_fp8_widen_extra_bytes(&compact, activation, Some(24 * 1024 * 1024 * 1024)),
            0,
            "a card that keeps the per-forward arm holds one copy, as it always did"
        );

        // An unresolvable gate charges nothing rather than guessing.
        assert_eq!(flux2_fp8_widen_extra_bytes(&compact, activation, None), 0);
        assert_eq!(
            flux2_fp8_widen_extra_bytes(&compact, activation, Some(0)),
            0
        );

        // And the gates: only fp8, only flux2, never a GGUF.
        let bf16 = with(make("flux2-klein-9b-bf16.safetensors", 9_079_000_000));
        assert_eq!(
            flux2_fp8_widen_extra_bytes(&bf16, activation, Some(33_600_000_000)),
            0,
            "a bf16 checkpoint is never widened"
        );
        let gguf = with(make("flux2-klein-9b-fp8.gguf", 9_079_000_000));
        assert_eq!(
            flux2_fp8_widen_extra_bytes(&gguf, activation, Some(33_600_000_000)),
            0,
            "a GGUF takes the quantized path, not the FP8 widen"
        );
        assert_eq!(
            flux2_fp8_widen_extra_bytes(
                &compact,
                Some(hint(ActivationFamily::FluxDit)),
                Some(33_600_000_000)
            ),
            charged,
            "the path name still identifies the family when the hint disagrees"
        );
    }

    /// The charge reaches the estimator whose output drives encoder
    /// placement.
    ///
    /// `under_memory_pressure` is computed from `eager_peak`, and
    /// `should_auto_park_text_encoder` reads that — so a charge added only to
    /// the final peak left the one consumer it was written for reading the
    /// old number.
    #[test]
    fn the_widen_reaches_the_eager_peak_that_decides_encoder_placement() {
        let dir = tempfile::tempdir().unwrap();
        let transformer = dir.path().join("flux2-klein-9b-fp8.safetensors");
        std::fs::File::create(&transformer)
            .unwrap()
            .set_len(9_079_000_000)
            .unwrap();
        let mut model_paths = paths("/unused/transformer.safetensors");
        model_paths.transformer = transformer;
        let activation = Some(hint(ActivationFamily::Flux2Dit));

        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "portrait",
            "model": "flux2-klein-9b:fp8",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 1.0,
            "batch_size": 1
        }))
        .unwrap();

        let widen = flux2_fp8_widen_extra_bytes(&model_paths, activation, Some(33_600_000_000));
        assert!(widen > 0, "the fixture must be a card that widens");

        let budget = estimate_generation_memory_for_request(
            &request,
            &model_paths,
            activation,
            offload(AdmissionPolicy::Disabled),
            Some(33_600_000_000),
            false,
            false,
        );
        let bare = mold_inference::device::estimate_peak_memory(
            &model_paths,
            mold_inference::LoadStrategy::Eager,
        );
        assert!(
            budget.eager_peak_memory_bytes >= bare.saturating_add(widen),
            "eager_peak {} must carry the {widen}-byte widen over the {bare}-byte weights",
            budget.eager_peak_memory_bytes
        );
    }

    /// FLUX.1's predicate gains its flux2 sibling's second step. A 23.8 GB
    /// BF16 dev streams its blocks on a 24 GB card, as it always has, and
    /// stops streaming them on a card with room for the whole thing — which
    /// it never did before, because the size test had no availability arm at
    /// all.
    #[test]
    fn large_flux_bf16_auto_offloads_only_when_residency_does_not_fit() {
        let dir = tempfile::tempdir().unwrap();
        let transformer = dir.path().join("flux1-dev.safetensors");
        std::fs::File::create(&transformer)
            .unwrap()
            .set_len(23_800_000_000)
            .unwrap();
        let mut model_paths = paths("/unused/transformer.safetensors");
        model_paths.transformer = transformer;
        let activation = Some(hint(ActivationFamily::FluxDit));

        assert!(
            large_flux_bf16_should_auto_offload(&model_paths, activation, None, 0),
            "the size question alone is today's answer, byte for byte"
        );
        assert!(
            large_flux_bf16_should_auto_offload(
                &model_paths,
                activation,
                Some(24_000_000_000),
                256_000_000
            ),
            "a 24 GB card cannot hold a 23.8 GB BF16 checkpoint and its workspace"
        );
        assert!(
            !large_flux_bf16_should_auto_offload(
                &model_paths,
                activation,
                Some(96_000_000_000),
                256_000_000
            ),
            "a 96 GB card must not pay the 3-5x streaming penalty for a checkpoint it holds"
        );

        // The eligibility gates are untouched: a GGUF of the same size never
        // enters, and neither does another family's checkpoint.
        let gguf = dir.path().join("flux1-dev-Q8_0.gguf");
        std::fs::File::create(&gguf)
            .unwrap()
            .set_len(23_800_000_000)
            .unwrap();
        let mut gguf_paths = model_paths.clone();
        gguf_paths.transformer = gguf;
        assert!(!large_flux_bf16_should_auto_offload(
            &gguf_paths,
            activation,
            None,
            0
        ));
        assert!(!large_flux_bf16_should_auto_offload(
            &model_paths,
            Some(hint(ActivationFamily::Flux2Dit)),
            None,
            0
        ));
    }

    #[test]
    fn explicit_gemma_gpu_policy_tracks_the_assigned_worker_ordinal() {
        assert!(ltx2_encoder_phase_competes_with_transformer_gpu_from_values(Some("gpu"), None, 1));
        assert!(ltx2_encoder_phase_competes_with_transformer_gpu_from_values(Some("gpu"), None, 7));
        assert!(
            !ltx2_encoder_phase_competes_with_transformer_gpu_from_values(Some("cpu"), None, 1)
        );
    }

    /// The #641 incident request: `ltx-2-19b-distilled:fp8` image-to-video at
    /// 1024x1024 x 97 frames.
    fn ltx2_incident_request() -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{
                "prompt": "Bring this image to life",
                "model": "ltx-2-19b-distilled:fp8",
                "width": 1024,
                "height": 1024,
                "steps": 8,
                "guidance": 3.0,
                "frames": 97
            }"#,
        )
        .unwrap();
        request.source_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        request
    }

    /// Updated for #641: the LTX-2 activation budget was a pixel-area estimate
    /// multiplied by the live latent-frame count, which under-counted the
    /// two-stage 1024x1024 shape. It is now the token model that
    /// `ltx2_admission` and the engine share.
    #[test]
    fn ltx2_activation_budget_is_token_based() {
        let request = ltx2_incident_request();
        let hint = ActivationHint::from_request(&request, "ltx2");

        // The token budget, plus the existing decoded source-frame buffer that
        // every conditioned request pays regardless of family.
        let source_frame_bytes = 1024u64 * 1024 * 4;
        assert_eq!(
            request_sensitive_activation_memory(&request, Some(hint), false),
            crate::ltx2_admission::ltx2_activation_budget_bytes(1024, 1024, 97, true, None)
                + source_frame_bytes,
        );
        // 97 frames produce 13 live latent frames; a 49-frame clip produces 7.
        let shorter = GenerateRequest {
            frames: Some(49),
            ..request.clone()
        };
        assert!(
            request_sensitive_activation_memory(&shorter, Some(hint), false)
                < request_sensitive_activation_memory(&request, Some(hint), false),
            "the budget must still scale with the temporally-compressed frame count"
        );
    }

    /// LTX-2 paths for the admission tests: a header-only checkpoint whose
    /// weight layout matches the measured `ltx-2-19b-distilled:fp8` preset.
    fn ltx2_paths(dir: &Path) -> ModelPaths {
        let checkpoint = dir.join("ltx2").join("ltx-2-19b-distilled-fp8.safetensors");
        std::fs::create_dir_all(checkpoint.parent().unwrap()).unwrap();
        crate::ltx2_admission::test_support::write_header_only_checkpoint(
            &checkpoint,
            &crate::ltx2_admission::test_support::ltx2_19b_fp8_facts(),
        );
        crate::ltx2_admission::warm_checkpoint_facts(&checkpoint)
            .expect("the fixture checkpoint header must parse");
        ModelPaths {
            transformer: checkpoint,
            ..paths("/models/unused.safetensors")
        }
    }

    /// RTX 4090 free reading from the #641 incident host.
    const RTX_4090_AVAILABLE: u64 = 25_757_220_864;

    #[test]
    fn ltx2_peak_includes_resident_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let request = ltx2_incident_request();
        let hint = ActivationHint::from_request(&request, "ltx2");

        let budget = estimate_generation_memory_for_request(
            &request,
            &ltx2_paths(dir.path()),
            Some(hint),
            offload(AdmissionPolicy::Disabled),
            Some(RTX_4090_AVAILABLE),
            false,
            false,
        );

        // Before #641 this returned 11,548,381,184 — a flat 6 GB streaming cap
        // plus 2 GB headroom plus a pixel-area activation guess, with none of
        // the 20.9 GB of transformer blocks the engine keeps resident.
        assert!(
            budget.peak_memory_bytes > 20_000_000_000,
            "predicted peak {} must account for resident transformer blocks",
            budget.peak_memory_bytes
        );
    }

    /// Issue #641's primary acceptance criterion. The incident shape must be
    /// admitted on a 24 GB card and run by streaming most of the transformer;
    /// the honest estimate exists to bound the plan, not to refuse the
    /// workload adaptive offload is built for.
    #[test]
    fn ltx2_1024x1024x97_is_admissible_on_24gb() {
        let dir = tempfile::tempdir().unwrap();
        let request = ltx2_incident_request();
        let hint = ActivationHint::from_request(&request, "ltx2");

        let budget = estimate_generation_memory_for_request(
            &request,
            &ltx2_paths(dir.path()),
            Some(hint),
            offload(AdmissionPolicy::Disabled),
            Some(RTX_4090_AVAILABLE),
            false,
            false,
        );

        assert_eq!(
            budget.fits_available_memory,
            Some(true),
            "1024x1024 x 97 frames must be admitted on a 24 GB card by streaming \
             blocks, not refused; predicted peak {}",
            budget.peak_memory_bytes
        );
        assert!(
            budget.peak_memory_bytes <= RTX_4090_AVAILABLE,
            "the predicted peak {} must fit the card it was planned against",
            budget.peak_memory_bytes
        );
    }

    #[test]
    fn ltx2_shorter_clip_still_passes_admission_on_24gb() {
        let dir = tempfile::tempdir().unwrap();
        let request = GenerateRequest {
            frames: Some(49),
            ..ltx2_incident_request()
        };
        let hint = ActivationHint::from_request(&request, "ltx2");

        let budget = estimate_generation_memory_for_request(
            &request,
            &ltx2_paths(dir.path()),
            Some(hint),
            offload(AdmissionPolicy::Disabled),
            Some(RTX_4090_AVAILABLE),
            false,
            false,
        );

        assert_eq!(budget.fits_available_memory, Some(true));
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

    /// The charged overhead is a MEASUREMENT, and this is what keeps it one.
    ///
    /// The adapter half is recomputed from `flux::pulid`'s own geometry — the
    /// same arithmetic `PulidAdapter::resident_bytes` performs — so a change to
    /// the adapter's shape breaks this rather than silently making the budget
    /// wrong. The remainder is the cross-attention activation headroom, which
    /// is bounded rather than pinned: it must be enough for 1024x1024 and small
    /// enough not to become a second adapter.
    #[test]
    fn identity_overhead_matches_the_adapters_own_resident_arithmetic() {
        // `PulidAdapterConfig::default()`: dim 3072, dim_head 128, heads 16,
        // kv_dim 2048; twenty modules for FLUX.1's 19 double + 38 single
        // blocks.
        const DIM: u64 = 3072;
        const INNER: u64 = 128 * 16;
        const KV: u64 = 2048;
        const MODULES: u64 = 20;
        const F16: u64 = 2;
        let per_module = 2 * KV + 2 * DIM + INNER * DIM + 2 * INNER * KV + DIM * INNER;
        let adapter = per_module * MODULES * F16;

        assert_eq!(
            adapter, IDENTITY_ADAPTER_BF16_BYTES,
            "the adapter term must be the adapter's own resident arithmetic"
        );
        assert!(
            IDENTITY_VRAM_OVERHEAD_BYTES > adapter,
            "the budget must leave room for the injections themselves"
        );

        // 1024x1024 is 4096 image tokens. One injection's working set is the
        // normalized image stream plus the query/output projections.
        const TOKENS: u64 = 4096;
        let one_injection = (TOKENS * DIM + TOKENS * INNER) * F16;
        let activations = IDENTITY_VRAM_OVERHEAD_BYTES - adapter;
        assert!(
            activations >= 4 * one_injection,
            "activation headroom {activations} must cover a 1024x1024 injection's working set"
        );
        assert!(
            activations < adapter,
            "activation headroom {activations} must not exceed the weights it serves"
        );

        // The old declared placeholder charged the CPU-side extraction to the
        // device. It must not creep back.
        const {
            assert!(
                IDENTITY_VRAM_OVERHEAD_BYTES < 2_300_000_000,
                "the extraction runs on the host and must not be charged as VRAM"
            )
        };
    }

    /// The SDXL charge is the SDXL adapter's own arithmetic, derived from the
    /// UNet's cross-attention layout rather than analogized from FLUX's.
    ///
    /// The engine computes the same figure from
    /// `plan_attn_layers(sdxl_unet_layout())`, and
    /// `mold_inference::sdxl::pulid`'s
    /// `the_sdxl_adapters_resident_bytes_are_the_layer_tables_own_arithmetic`
    /// pins the layer table itself against the checkpoint. This end holds the
    /// budget honest without linking the engine.
    #[test]
    fn sdxl_identity_overhead_matches_the_adapters_own_resident_arithmetic() {
        // `testdata/pulid_sdxl/attn_layer_map.json`: 70 attn2 modules, of
        // which 10 are 640-wide (down_blocks.1 and up_blocks.1) and 60 are
        // 1280-wide (down_blocks.2, up_blocks.0, and the mid block). Each
        // carries a bias-free `id_to_k` and `id_to_v` of
        // `[hidden_size, 2048]`.
        const CROSS: u64 = 2048;
        const NARROW: u64 = 640;
        const WIDE: u64 = 1280;
        const NARROW_LAYERS: u64 = 10;
        const WIDE_LAYERS: u64 = 60;
        const F16: u64 = 2;
        let elements = 2 * CROSS * (NARROW_LAYERS * NARROW + WIDE_LAYERS * WIDE);
        assert_eq!(elements, 340_787_200);
        let adapter = elements * F16;

        assert_eq!(
            adapter, IDENTITY_SDXL_ADAPTER_BF16_BYTES,
            "the adapter term must be the adapter's own resident arithmetic"
        );
        assert!(
            IDENTITY_SDXL_VRAM_OVERHEAD_BYTES > adapter,
            "the budget must leave room for the injections themselves"
        );

        // The largest identity branch is a 640-wide layer at 1024x1024 —
        // 64x64 tokens after one downsample, doubled by the CFG batch. Its
        // working set is the query, the attention output, and the combined
        // result, plus a `[2, 10, 4096, 32]` score matrix.
        const BATCH: u64 = 2;
        const TOKENS: u64 = 64 * 64;
        const HEADS: u64 = 10;
        const ID_TOKENS: u64 = 32;
        let one_injection =
            (3 * BATCH * TOKENS * NARROW + BATCH * HEADS * TOKENS * ID_TOKENS) * F16;
        let activations = IDENTITY_SDXL_VRAM_OVERHEAD_BYTES - adapter;
        assert!(
            activations >= 4 * one_injection,
            "activation headroom {activations} must cover a 1024x1024 injection's working set \
             ({one_injection}) several times over"
        );
        assert!(
            activations < adapter,
            "activation headroom {activations} must not exceed the weights it serves"
        );

        // SDXL's adapter is genuinely smaller than FLUX's and its attention
        // runs at a quarter the token count, so the charge must be lower —
        // charging FLUX's figure would park cards that can run this.
        const { assert!(IDENTITY_SDXL_VRAM_OVERHEAD_BYTES < IDENTITY_VRAM_OVERHEAD_BYTES) };
        assert_eq!(
            identity_adapter_overhead_bytes(mold_core::identity::IdentityFamily::Sdxl),
            IDENTITY_SDXL_VRAM_OVERHEAD_BYTES
        );
        assert_eq!(
            identity_adapter_overhead_bytes(mold_core::identity::IdentityFamily::Flux),
            IDENTITY_VRAM_OVERHEAD_BYTES
        );
    }

    /// The IP-Adapter charge is each adapter's own arithmetic, derived from
    /// the UNet cross-attention layout the checkpoint was trained against
    /// rather than from the published file size.
    ///
    /// The engine computes the same figure from
    /// `ip_adapter::IpAdapter::resident_bytes`, over
    /// `plan_attn_layers(...)`; `mold_inference::sdxl::pulid`'s
    /// `the_sd15_layer_table_matches_upstreams_own_processor_enumeration` and
    /// `the_sdxl_layer_table_...` pin the layer tables themselves against
    /// upstream's own enumeration of a real diffusers UNet. This end holds the
    /// budgets honest without linking the engine.
    ///
    /// The published file sizes are the cross-check: an arithmetic that is a
    /// safetensors header away from the bytes upstream actually ships is
    /// arithmetic about the right file.
    #[test]
    fn ip_adapter_overhead_matches_the_adapters_own_resident_arithmetic() {
        const F16: u64 = 2;
        // `image_proj` is `Linear(clip_dim, tokens * ctx_dim)` plus that
        // linear's bias plus the output `LayerNorm`'s weight and bias.
        const TOKENS: u64 = 4;
        const CLIP: u64 = 1024;
        let projection = |context: u64| TOKENS * context * CLIP + TOKENS * context + 2 * context;

        // `testdata/pulid_sdxl/attn_layer_map_sd15.json`: 16 attn2 modules —
        // 5 x 320, 5 x 640, 6 x 1280 — each with a bias-free `to_k_ip` and
        // `to_v_ip` of `[hidden_size, 768]`.
        const SD15_CONTEXT: u64 = 768;
        let sd15_hidden = 5 * 320 + 5 * 640 + 6 * 1280;
        assert_eq!(sd15_hidden, 12_480);
        let sd15 = (2 * SD15_CONTEXT * sd15_hidden + projection(SD15_CONTEXT)) * F16;
        assert_eq!(
            sd15, IP_ADAPTER_SD15_ADAPTER_BF16_BYTES,
            "the SD1.5 term must be the adapter's own resident arithmetic"
        );
        // `ip-adapter_sd15.safetensors` as published, less its header.
        assert!(
            (44_642_768 - sd15) < 8_192,
            "the arithmetic must land a header away from the published file"
        );

        // `testdata/pulid_sdxl/attn_layer_map.json`: 70 attn2 modules —
        // 10 x 640 and 60 x 1280 — at `[hidden_size, 2048]`.
        const SDXL_CONTEXT: u64 = 2048;
        let sdxl_hidden = 10 * 640 + 60 * 1280;
        assert_eq!(sdxl_hidden, 83_200);
        let sdxl = (2 * SDXL_CONTEXT * sdxl_hidden + projection(SDXL_CONTEXT)) * F16;
        assert_eq!(
            sdxl, IP_ADAPTER_SDXL_ADAPTER_BF16_BYTES,
            "the SDXL term must be the adapter's own resident arithmetic"
        );
        assert!(
            (698_391_064 - sdxl) < 32_768,
            "the arithmetic must land a header away from the published file"
        );

        // Each budget leaves room for the injections the weights serve.
        assert!(IP_ADAPTER_SD15_VRAM_OVERHEAD_BYTES > sd15);
        assert!(IP_ADAPTER_SDXL_VRAM_OVERHEAD_BYTES > sdxl);

        // The largest SDXL injection: a 640-wide layer at 1024x1024 is 64x64
        // tokens after one downsample, doubled by the CFG batch. Its working
        // set is the image branch's delta, its transposed intermediate, and
        // the combined result, plus a `[2, 10, 4096, 4]` score matrix.
        const BATCH: u64 = 2;
        const SDXL_TOKENS: u64 = 64 * 64;
        const SDXL_NARROW: u64 = 640;
        const SDXL_HEADS: u64 = 10;
        let sdxl_injection = (3 * BATCH * SDXL_TOKENS * SDXL_NARROW
            + BATCH * SDXL_HEADS * SDXL_TOKENS * TOKENS)
            * F16;
        let sdxl_headroom = IP_ADAPTER_SDXL_VRAM_OVERHEAD_BYTES - sdxl;
        assert!(
            sdxl_headroom >= 4 * sdxl_injection,
            "SDXL headroom {sdxl_headroom} must cover a 1024x1024 injection's working set"
        );

        // The largest SD1.5 injection: a 320-wide layer at 768x768 is 96x96
        // tokens, doubled by the CFG batch. Unlike SDXL's, this legitimately
        // EXCEEDS the 44 MB of weights it serves, which is why the SD1.5
        // budget deliberately does not carry the identity charge's
        // "headroom must not exceed the weights" bound.
        const SD15_TOKENS: u64 = 96 * 96;
        const SD15_NARROW: u64 = 320;
        const SD15_HEADS: u64 = 8;
        let sd15_injection = (3 * BATCH * SD15_TOKENS * SD15_NARROW
            + BATCH * SD15_HEADS * SD15_TOKENS * TOKENS)
            * F16;
        let sd15_headroom = IP_ADAPTER_SD15_VRAM_OVERHEAD_BYTES - sd15;
        assert!(
            sd15_headroom >= sd15_injection,
            "SD1.5 headroom {sd15_headroom} must cover a 768x768 injection's working set"
        );

        // The two families must never be charged the same number: 830 MB on an
        // SD1.5 render parks 8 GB cards, and 90 MB on an SDXL one admits a
        // render with 740 MB nowhere to go.
        assert_ne!(
            IP_ADAPTER_SD15_VRAM_OVERHEAD_BYTES,
            IP_ADAPTER_SDXL_VRAM_OVERHEAD_BYTES
        );
        assert_eq!(
            ip_adapter_overhead_bytes(mold_core::ip_adapter_assets::ImagePromptFamily::Sd15),
            IP_ADAPTER_SD15_VRAM_OVERHEAD_BYTES
        );
        assert_eq!(
            ip_adapter_overhead_bytes(mold_core::ip_adapter_assets::ImagePromptFamily::Sdxl),
            IP_ADAPTER_SDXL_VRAM_OVERHEAD_BYTES
        );
    }

    /// The tower term is the published architecture's own parameter count at
    /// the checkpoint's f32 width, and it is a TRANSIENT — released before the
    /// denoise loop — so it must never grow into a resident-sized allowance.
    #[test]
    fn the_vision_tower_term_is_the_towers_own_parameter_count() {
        // `models/image_encoder/config.json`: hidden 1280, 32 layers, 16
        // heads, intermediate 5120, patch 14, image 224, projection 1024.
        const HIDDEN: u64 = 1280;
        const LAYERS: u64 = 32;
        const INTERMEDIATE: u64 = 5120;
        const PATCHES: u64 = (224 / 14) * (224 / 14) + 1;
        const PROJECTION: u64 = 1024;
        let embeddings = HIDDEN + HIDDEN * 3 * 14 * 14 + PATCHES * HIDDEN;
        let per_layer = 4 * (HIDDEN * HIDDEN + HIDDEN)
            + 2 * (2 * HIDDEN)
            + (HIDDEN * INTERMEDIATE + INTERMEDIATE)
            + (INTERMEDIATE * HIDDEN + HIDDEN);
        let parameters =
            embeddings + 2 * HIDDEN + LAYERS * per_layer + 2 * HIDDEN + HIDDEN * PROJECTION;
        assert_eq!(parameters, IP_ADAPTER_VISION_TOWER_PARAMETERS);
        // `models/image_encoder/model.safetensors` as published, less its
        // header: the architecture and the file are the same tower.
        let f32_bytes = parameters * 4;
        assert!(
            (2_528_373_448 - f32_bytes) < 131_072,
            "the parameter count must land a header away from the published f32 file"
        );

        // The CHARGE is against the width the tower is actually built at,
        // which is the engine's f16 — not the file's f32.
        let f16_bytes = parameters * 2;
        assert!(
            IP_ADAPTER_VISION_TOWER_VRAM_PEAK_BYTES > f16_bytes,
            "the charge must cover the weights themselves"
        );
        assert!(
            IP_ADAPTER_VISION_TOWER_VRAM_PEAK_BYTES < f32_bytes,
            "charging the file's f32 width over-charges every reference render \
             by more than a gigabyte"
        );
        let working_set = IP_ADAPTER_VISION_TOWER_VRAM_PEAK_BYTES - f16_bytes;
        // One 224x224 forward at batch 1: the hidden stream, the score matrix,
        // and the MLP intermediate, at f32.
        let forward = (PATCHES * HIDDEN + 16 * PATCHES * PATCHES + PATCHES * INTERMEDIATE) * 4;
        assert!(
            working_set >= 4 * forward,
            "the tower allowance {working_set} must cover its own forward"
        );
        assert!(
            working_set < f32_bytes / 8,
            "the allowance is for a forward, not a second copy of the tower"
        );
    }

    /// The IP-Adapter charge follows the family of the requested model, and the
    /// two bases that share one [`ActivationFamily`] are still told apart.
    ///
    /// This is the case identity does not have: `activation_family_for` maps
    /// both `sd15` and `sdxl` onto `SdxlUnet`, so a checkpoint with no built-in
    /// manifest is separated by its own composition — SDXL's second text
    /// encoder — rather than by a hint that cannot answer.
    #[test]
    fn the_ip_adapter_charge_follows_the_family_of_the_requested_model() {
        use mold_core::ip_adapter_assets::ImagePromptFamily;

        let base: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a lighthouse",
            "model": "sdxl-base:fp16",
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "guidance": 7.5,
        }))
        .unwrap();
        let sdxl_paths = ModelPaths {
            clip_encoder_2: Some(PathBuf::from("/models/clip_g.safetensors")),
            ..paths("/models/sdxl.safetensors")
        };
        let sd15_paths = paths("/models/sd15.safetensors");
        let family = |req: &GenerateRequest, paths: &ModelPaths| {
            ip_adapter_overhead_family_with_projection_and_hint(
                req,
                paths,
                None,
                Some(hint(ActivationFamily::SdxlUnet)),
            )
        };

        // No reference attached: nothing is charged, on either base.
        assert_eq!(family(&base, &sdxl_paths), None);
        assert!(!request_charges_ip_adapter_overhead(&base));

        let mut sdxl = base.clone();
        sdxl.edit_images = Some(vec![vec![0x89, 0x50, 0x4e, 0x47]]);
        assert!(request_charges_ip_adapter_overhead(&sdxl));
        assert_eq!(
            family(&sdxl, &sdxl_paths),
            Some(ImagePromptFamily::Sdxl),
            "a built-in manifest's own family is the authority"
        );

        // Weight zero is inert, exactly as `id_weight` 0 is.
        let mut zero = sdxl.clone();
        zero.reference_weight = Some(0.0);
        assert_eq!(family(&zero, &sdxl_paths), None);
        assert!(!request_charges_ip_adapter_overhead(&zero));

        // An SD1.5 checkpoint charges the SD1.5 adapter.
        let mut sd15 = sdxl.clone();
        sd15.model = "sd15:fp16".to_string();
        assert_eq!(family(&sd15, &sd15_paths), Some(ImagePromptFamily::Sd15));

        // A live-catalog id has no manifest, so the checkpoint's own
        // composition decides — and it must decide DIFFERENTLY for the two.
        let mut catalog = sdxl.clone();
        catalog.model = "cv:123456".to_string();
        assert_eq!(family(&catalog, &sdxl_paths), Some(ImagePromptFamily::Sdxl));
        assert_eq!(family(&catalog, &sd15_paths), Some(ImagePromptFamily::Sd15));

        // And it must still decide SDXL once CLIP-G is parked on the CPU.
        // `gpu_resident_paths` clears `clip_encoder_2` for exactly that
        // placement, which is how reading the encoder alone would under-charge
        // a whole class of SDXL renders by 740 MB.
        let parked_clip_g = ModelPaths {
            clip_encoder_2: None,
            clip_tokenizer_2: Some(PathBuf::from("/models/clip_g_tokenizer.json")),
            ..paths("/models/sdxl.safetensors")
        };
        assert_eq!(
            family(&catalog, &parked_clip_g),
            Some(ImagePromptFamily::Sdxl)
        );

        // A family with no image-prompt adapter charges nothing, even carrying
        // references: FLUX.2 and Qwen-Image-Edit condition on them natively.
        let mut flux2 = sdxl.clone();
        flux2.model = "flux2-klein:q8".to_string();
        assert_eq!(
            ip_adapter_overhead_family_with_projection_and_hint(
                &flux2,
                &paths("/models/flux2.safetensors"),
                None,
                Some(hint(ActivationFamily::Flux2Dit)),
            ),
            None
        );
    }

    /// The whole charge, through the estimate the scheduler actually reads: a
    /// reference render is charged its adapter AND its tower, and a request
    /// that attaches nothing is charged neither.
    #[test]
    fn the_estimate_charges_the_adapter_and_the_tower_and_only_for_a_reference() {
        let plain: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a lighthouse",
            "model": "sdxl-base:fp16",
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "guidance": 7.5,
        }))
        .unwrap();
        let mut referenced = plain.clone();
        referenced.edit_images = Some(vec![vec![0x89, 0x50, 0x4e, 0x47]]);
        let mut zero = referenced.clone();
        zero.reference_weight = Some(0.0);

        let sdxl_paths = ModelPaths {
            clip_encoder_2: Some(PathBuf::from("/models/clip_g.safetensors")),
            ..paths("/models/sdxl.safetensors")
        };
        let estimate = |req: &GenerateRequest| {
            estimate_generation_memory_for_request(
                req,
                &sdxl_paths,
                Some(hint(ActivationFamily::SdxlUnet)),
                GenerationOffloadPolicy::new(
                    false,
                    mold_inference::wan::block_offload::AdmissionPolicy::Disabled,
                    false,
                ),
                None,
                false,
                false,
            )
            .peak_memory_bytes
        };

        // The comparison is against the ZERO-weight render, not the plain one,
        // and that is the point of the test rather than a convenience. Both
        // carry an attached picture, so the pre-existing decoded-pixel term
        // (`width x height x 4 x batch`, charged for any source or reference
        // image) is identical on both sides and what remains is exactly this
        // slice's charge.
        assert_eq!(
            estimate(&referenced) - estimate(&zero),
            IP_ADAPTER_SDXL_VRAM_OVERHEAD_BYTES + IP_ADAPTER_VISION_TOWER_VRAM_PEAK_BYTES,
            "a reference render pays exactly its adapter plus its tower"
        );
        // And a zero-weight render pays NONE of the bundle: what separates it
        // from the plain render is that one decoded-pixel buffer and nothing
        // else. An adapter charge hiding in here would be at least 90 MB.
        assert_eq!(
            estimate(&zero) - estimate(&plain),
            u64::from(plain.width) * u64::from(plain.height) * 4,
            "reference_weight 0 must charge nothing but the attached picture itself"
        );
    }

    /// The estimate charges the family's OWN adapter, and a request naming
    /// identity fields on an unqualified checkpoint charges no adapter at all
    /// — admission refuses it, but the estimate runs first and must not
    /// invent one.
    #[test]
    fn the_identity_charge_follows_the_family_of_the_requested_model() {
        let mut sdxl: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a portrait",
            "model": "sdxl-base:fp16",
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "guidance": 7.5,
        }))
        .unwrap();
        assert_eq!(identity_overhead_family(&sdxl), None);

        sdxl.id_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        assert_eq!(
            identity_overhead_family(&sdxl),
            Some(mold_core::identity::IdentityFamily::Sdxl)
        );

        let mut flux = sdxl.clone();
        flux.model = "flux-dev:q8".to_string();
        assert_eq!(
            identity_overhead_family(&flux),
            Some(mold_core::identity::IdentityFamily::Flux)
        );

        let mut catalog = flux.clone();
        catalog.model = "cv:123".to_string();
        assert_eq!(identity_overhead_family(&catalog), None);
        assert_eq!(
            identity_overhead_family_with_projection_and_hint(
                &catalog,
                None,
                Some(hint(ActivationFamily::FluxDit))
            ),
            Some(mold_core::identity::IdentityFamily::Flux)
        );
        assert_eq!(
            identity_overhead_family_with_projection_and_hint(
                &catalog,
                None,
                Some(hint(ActivationFamily::SdxlUnet))
            ),
            Some(mold_core::identity::IdentityFamily::Sdxl)
        );

        let mut turbo = sdxl.clone();
        turbo.model = "sdxl-turbo:fp16".to_string();
        assert!(request_charges_identity_overhead(&turbo));
        assert_eq!(
            identity_overhead_family(&turbo),
            None,
            "an unqualified checkpoint has no adapter to charge"
        );

        let mut zero = sdxl.clone();
        zero.id_weight = Some(0.0);
        assert_eq!(identity_overhead_family(&zero), None);
    }

    /// Every artifact the extraction touches is host demand, and the adapter —
    /// the only device-resident one — is not.
    #[test]
    fn the_extraction_artifacts_are_host_only_and_the_adapter_is_not() {
        use crate::execution_plan::ComponentRole;
        for role in [
            ComponentRole::FaceDetector,
            ComponentRole::FaceRecognizer,
            ComponentRole::IdentityVisionEncoder,
        ] {
            assert!(
                role.is_host_only_for_test(),
                "{role:?} runs on the host at admission"
            );
        }
        assert!(
            !ComponentRole::IdentityAdapter.is_host_only_for_test(),
            "the cross-attention adapter is resident on the generation device"
        );
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod metal_policy_tests {
    use super::*;
    use mold_core::metal_memory::{MetalMemorySnapshot, MetalWiredLimit};

    #[test]
    fn metal_memory_warm_admission_credits_weights_once_and_rechecks_changed_policy() {
        let gib = 1 << 30;
        let mut sample = MetalMemorySnapshot {
            wired_limit: MetalWiredLimit::Automatic,
            physical_bytes: Some(48 * gib),
            available_host_bytes: Some(32 * gib),
            recommended_bytes: Some(37 * gib),
            allocated_bytes: Some(25 * gib),
            effective_capacity_bytes: None,
            allocation_headroom_bytes: None,
            error: None,
        }
        .resolve();
        // Same native delta recorded by the loader: 25 GiB after - 1 GiB before.
        let resident = 24 * gib;
        let admits = |sample: &MetalMemorySnapshot, credit| {
            let available = metal_available_from_sample(Some(sample), credit)?.unwrap();
            check_planned_memory_budget_with_resident("warm", 30 * gib, available, 0, None, "")
        };
        assert!(admits(&sample, resident).is_ok());
        assert!(
            admits(&sample, 0).is_err(),
            "missing loader attribution double-charges weights"
        );
        sample.wired_limit = MetalWiredLimit::Explicit { mib: 16 * 1024 };
        sample = sample.resolve();
        assert!(
            admits(&sample, resident).is_err(),
            "resident credit must not defeat a lowered policy"
        );
        sample.wired_limit = MetalWiredLimit::Unavailable;
        sample.error = Some("permission denied".into());
        sample = sample.resolve();
        assert!(admits(&sample, resident)
            .unwrap_err()
            .error
            .contains("permission denied"));
        assert_eq!(metal_available_from_sample(None, resident).unwrap(), None);
    }
}

#[cfg(test)]
mod streamed_text_encoder_tests {
    use super::*;
    use crate::execution_plan::sparse_admission_test_file;
    use mold_inference::wan::block_offload::AdmissionPolicy;
    use std::path::PathBuf;
    use tempfile::TempDir;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn paths(transformer: &Path, vae: &Path, encoders: &[&Path]) -> ModelPaths {
        ModelPaths {
            transformer: transformer.to_path_buf(),
            transformer_shards: Vec::new(),
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            vae: vae.to_path_buf(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: encoders.iter().map(|p| PathBuf::from(*p)).collect(),
            text_tokenizer: None,
            decoder: None,
        }
    }

    fn budget(model: &str, paths: &ModelPaths, available: u64) -> GenerationMemoryBudget {
        let request: GenerateRequest = serde_json::from_str(&format!(
            r#"{{"prompt":"x","model":"{model}","width":1024,"height":1024,"steps":20,"guidance":4.0}}"#
        ))
        .unwrap();
        let hint = Some(ActivationHint::from_request(&request, "flux2"));
        estimate_generation_memory_for_request(
            &request,
            paths,
            hint,
            GenerationOffloadPolicy::new(false, AdmissionPolicy::Disabled, false),
            Some(available),
            false,
            false,
        )
    }

    /// The planner charges FLUX.2 [dev]'s Mistral3 encoder what it STREAMS,
    /// never what it weighs on disk.
    ///
    /// `eager_peak_memory_bytes` is what `under_memory_pressure` and the
    /// placement planner's auto-park decision both read. With the encoder's
    /// 36 GB file in it, a completely idle 46 GB L40S looked over-subscribed,
    /// the encoder was parked on the CPU, FLUX.2 selected F32 there, and the
    /// 2026-09-11 audit measured a 78.8 s prompt encode with the GPU at 0 % SM
    /// for its entire duration.
    #[test]
    fn flux2_dev_eager_peak_charges_the_streamed_prefix_not_the_encoder_file() {
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("flux2-dev-Q8_0.gguf");
        let vae = root.path().join("flux2-vae.safetensors");
        let encoder = root.path().join("mistral_3_small_flux2_bf16.safetensors");
        sparse_admission_test_file(&transformer, 33 * GIB);
        sparse_admission_test_file(&vae, GIB / 2);
        sparse_admission_test_file(&encoder, 36 * GIB);
        let model_paths = paths(&transformer, &vae, &[&encoder]);

        let charge =
            mold_inference::flux2::text_encoder_residency::mistral3_admission_charge_for_gpu(
                "flux2-dev:q8",
                &model_paths,
            )
            .expect("a dev Mistral3 encoder streams");
        let file_priced = mold_inference::device::estimate_peak_memory(
            &model_paths,
            mold_inference::LoadStrategy::Eager,
        );
        let streamed = mold_inference::device::estimate_peak_memory_with_encoder_override(
            &model_paths,
            mold_inference::LoadStrategy::Eager,
            Some(charge.device_peak),
        );

        let answer = budget("flux2-dev:q8", &model_paths, 46 * GIB);
        // The generic per-request workspace, NOT `activation_memory_bytes`:
        // since #1707 that field carries FLUX.2's denoise working set, which
        // is spent after this `DropReload` encoder is already gone. The
        // co-residency question the eager peak asks is about weights.
        let co_residency_activation =
            mold_inference::device::activation_bytes(1024, 1024, 1, 2, ActivationFamily::Flux2Dit);
        assert_eq!(
            answer.eager_peak_memory_bytes,
            streamed.saturating_add(co_residency_activation),
            "the eager peak must price the streamed encoder, not its shards"
        );
        assert!(
            answer.activation_memory_bytes > co_residency_activation,
            "the denoise budget is the larger, separate figure the peak carries"
        );
        assert!(
            file_priced - streamed > 30 * GIB,
            "the file-priced estimate carries the ~36 GB over-charge this fixes"
        );
        assert!(
            !answer.under_memory_pressure,
            "an idle 46 GB card is not under pressure for a streamed encoder"
        );
    }

    /// The predicate is FLUX.2 [dev]'s alone. Klein conditions on a Qwen3
    /// encoder the engine materializes, so it keeps its file-size charge and
    /// the estimate must be byte-for-byte what it always was.
    #[test]
    fn a_klein_9b_bf16_encoder_keeps_its_file_size_charge() {
        let root = TempDir::new().unwrap();
        let transformer = root.path().join("flux2-klein-9b-bf16.safetensors");
        let vae = root.path().join("flux2-vae.safetensors");
        let encoder = root.path().join("qwen3-4b-bf16.safetensors");
        sparse_admission_test_file(&transformer, 18 * GIB);
        sparse_admission_test_file(&vae, GIB / 2);
        sparse_admission_test_file(&encoder, 8 * GIB);
        let model_paths = paths(&transformer, &vae, &[&encoder]);

        assert!(
            mold_inference::flux2::text_encoder_residency::mistral3_admission_charge_for_gpu(
                "flux2-klein-9b:bf16",
                &model_paths,
            )
            .is_none(),
            "Klein has no streamed Mistral3 prefix"
        );

        let answer = budget("flux2-klein-9b:bf16", &model_paths, 46 * GIB);
        assert_eq!(
            answer.eager_peak_memory_bytes,
            mold_inference::device::estimate_peak_memory(
                &model_paths,
                mold_inference::LoadStrategy::Eager,
            )
            // The co-residency workspace, not the denoise one — see
            // `flux2_dev_eager_peak_charges_the_streamed_prefix_not_the_encoder_file`.
            .saturating_add(mold_inference::device::activation_bytes(
                1024,
                1024,
                1,
                2,
                ActivationFamily::Flux2Dit,
            ))
        );
    }
}
