//! Wan admission memory model.
//!
//! The generic activation estimate prices a render as pixel area times a
//! per-family constant. For a video family that is blind to the axis that
//! dominates: an 832x480 wan request is priced near the floor whether it asks
//! for one frame or 257. The server therefore admits wan jobs that are
//! guaranteed to OOM mid-denoise, after the user has already paid for the UMT5
//! encode and the expert load, and cannot refuse an 81-frame 720p request on a
//! 24 GB card.
//!
//! This module prices the shape the engine will actually run, from the
//! checkpoint's own header. `mold_inference::device` owns the calibrated
//! per-token model so admission and the engine cannot drift; this side owns
//! reading the geometry out of the checkpoint and turning a request into a
//! shape.
//!
//! Two things measurement changed relative to the issue that asked for this:
//!
//! * **CFG is not a multiplier.** `wan::pipeline` runs the conditional forward
//!   and then the unconditional one *sequentially*, combining two velocity
//!   tensors, so the activation working set is one forward's worth. Measured
//!   on an RTX 4090 (1.3B, 81f/832x480, guidance 1.0 vs 5.0): 11,858 MiB vs
//!   12,370 MiB, +512 MiB — against a working set of ~2 GB at that shape.
//!   Pricing CFG at 2x would reject shapes that run.
//! * **The generic estimate's CFG flag is wrong for wan regardless.**
//!   `request_sensitive_activation_memory` gates on `negative_prompt.is_some()`
//!   but `wan::pipeline::needs_cfg_pass` keys purely on `guidance > 1.0`, and
//!   an absent negative is filled engine-side with the tuned default.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use mold_core::ModelPaths;
use mold_inference::device::{WanActivationGeometry, WanVaeGeneration};

/// Request shape driving the wan activation budget. `ActivationHint` stays as
/// it is — it is constructed by name across the server and carries no
/// video-specific fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WanShapeHint {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    /// Whether the schedule runs a second, unconditional forward.
    ///
    /// Keyed on guidance alone, mirroring `wan::pipeline::needs_cfg_pass`. The
    /// request's negative prompt is deliberately not consulted: wan fills an
    /// absent one engine-side with the checkpoint's tuned default, and an
    /// explicit empty string is still a real uncond forward.
    pub(crate) cfg: bool,
    /// Whether this request drives per-token timesteps.
    ///
    /// TI2V's latent inpaint gives frame 0 its own timestep, which turns every
    /// block's modulation table from one broadcast row into a full
    /// `[1, T, 6, dim]` F32 tensor. `WanImageConditioning::LatentInpaint` is
    /// built only in the source-image branch
    /// (`crates/mold-inference/src/wan/pipeline.rs`), so a plain T2V request on
    /// a TI2V checkpoint runs the scalar path and must not be charged for it.
    pub(crate) latent_inpaint: bool,
}

/// Frames the engine renders when the request omits them.
///
/// `wan::pipeline` fills an absent `frames` from the VAE generation's own
/// default (`WanVaeGeneration::default_timing`) — 81 for 2.1, 121 for 2.2 —
/// so pricing an omitted count as a single latent frame would estimate a
/// one-frame still for a job that renders a full clip. This module's premise
/// is pricing the shape the engine will actually run.
fn default_frames(vae: WanVaeGeneration) -> u32 {
    match vae {
        WanVaeGeneration::V21 => 81,
        WanVaeGeneration::V22 => 121,
    }
}

impl WanShapeHint {
    pub(crate) fn from_request(req: &mold_core::GenerateRequest) -> Self {
        Self {
            width: req.width,
            height: req.height,
            // Resolved against the checkpoint in `wan_activation_bytes`, which
            // is where the VAE generation is known; 0 marks "request omitted".
            frames: req.frames.unwrap_or(0),
            cfg: req.guidance > 1.0,
            latent_inpaint: req.source_image.is_some()
                || req
                    .keyframes
                    .as_ref()
                    .is_some_and(|keyframes| !keyframes.is_empty()),
        }
    }
}

/// Peak activation bytes for one wan request against one checkpoint.
/// The measured calibration behind this estimate. The constants themselves
/// now live in `mold_inference::device` so the engine's block-offload policy
/// reads the same numbers; the fit that produced them is recorded here.
///
/// Extra device memory a CFG step holds beyond a single forward.
///
/// The two forwards are sequential, so this is not the working set doubling.
/// Re-measured after #776 item 2 sliced the gated residual, which is what the
/// old figures were mostly charging for — the uncond pass used to leave
/// full-clip F32 transients alive, and now it cannot. Same protocol, one
/// render with and one without the uncond pass:
///
/// | Tier | CFG off | CFG on | Delta |
/// | --- | ---: | ---: | ---: |
/// | 1.3B bf16, 81f/832x480 | 12,114 MiB | 12,370 MiB | +256 MiB |
/// | A14B q5, 53f/832x480 | 21,354 MiB | 21,322 MiB | -32 MiB |
///
/// Was +512 and +945 MiB respectively. A14B's delta is now negative, i.e. gone
/// into the noise, so the policy of carrying the larger of the two now carries
/// the *small* model's figure — which is the safe direction, since it
/// over-charges the tier that sits nowhere near a 24 GB ceiling and charges the
/// tier that does exactly what it was measured to cost.
/// Token-independent denoise workspace: RoPE tables held across the loop, the
/// latent and noise tensors, the text embeddings, the VAE decode working set,
/// and allocator/cuDNN scratch.
///
/// This was 0, with the whole cost folded into the measured slope. That is only
/// self-consistent while the slope is fitted at the shape it is used at: a
/// difference-based fit cancels the intercept, so applying it against total
/// tokens silently re-charges the intercept once per token. The old 2.26 slope
/// was steep enough to hide it; re-fitting to the real per-token cost (2.14)
/// exposed it as a ~2.3 GB shortfall that would have made admission accept an
/// 81-frame A14B render and OOM mid-denoise.
///
/// Fitted, not guessed: the CFG-off A14B pair gives 0.37607 MiB/token, so
/// extrapolating the 17-frame point back to zero tokens leaves a 13,141 MiB
/// intercept, of which 10,840 MiB is the UMT5 weight floor this module already
/// charges. The remainder is here.
/// Measured multiplier on the derived per-token tensor sum.
///
/// `device::wan_activation_budget_bytes` counts the tensors `WanBlock::forward`
/// visibly materializes; a real forward holds about 2.26x that. The residual is
/// intermediates candle keeps live that a hand-count of the obvious buffers
/// misses — the LayerNorm's internal F32 copies, RoPE application, the q/k RMS
/// norms over the full inner dimension, and expression temporaries that are not
/// freed until the enclosing statement ends.
///
/// The derived formula supplies the *shape* of the cost (it scales with `dim`,
/// `ffn_dim`, heads, and tokens); this supplies the *scale*, from hardware.
/// Naming it as one measured factor is honest about which half is which —
/// folding it into the flat term instead would make the slope silently wrong
/// and only look right at the shape it was fitted to, which is the defect this
/// calibration replaced.
///
/// Derived from two renders differing only in frame count, so every fixed cost
/// — weights, the encoder pool, allocator scratch — cancels.
///
/// Re-fitted for #776 item 2, which sliced the gated residual's F32 arithmetic
/// so it no longer scales with the clip. That changed both halves: the derived
/// per-token sum fell from 225,280 B to 184,320 B (three per-token F32 buffers
/// became two BF16 ones in `device::wan_activation_budget_bytes`), and the
/// measured per-token cost fell with it. Same protocol and same hardware as the
/// original fit — RTX 4090, `wan22-t2v-a14b:q5`, 832x480, CFG off, sampling
/// `nvidia-smi` through the run:
///
/// | Frames | Tokens | Peak |
/// | -----: | -----: | ---: |
/// | 17 | 7,800 | 16,074 MiB |
/// | 53 | 21,840 | 21,354 MiB |
///
/// 5,280 MiB over 14,040 tokens is 394,336 B/token against the derived
/// 184,320, i.e. 2.14. Both points are denoise-dominated rather than pinned by
/// the UMT5 pool, which was checked directly: forcing the encoder to CPU moves
/// the 17-frame peak by 32 MiB (16,074 -> 16,042).
///
/// For reference the pre-slicing fit was 509,052 B/token against 225,280, i.e.
/// 2.26 — so the ratio barely moved while the absolute per-token cost fell 23%.
/// That is the expected shape of the result: slicing removed buffers the
/// derived formula was already counting, rather than changing how much candle
/// holds beyond what a hand-count sees.
pub(crate) fn wan_activation_bytes(shape: WanShapeHint, geometry: WanActivationGeometry) -> u64 {
    // The checkpoint says whether per-token timesteps are *possible*; the
    // request says whether they happen.
    let geometry = WanActivationGeometry {
        per_token_timesteps: shape.latent_inpaint && matches!(geometry.vae, WanVaeGeneration::V22),
        ..geometry
    };
    let frames = if shape.frames == 0 {
        default_frames(geometry.vae)
    } else {
        shape.frames
    };
    // Delegates to the shared authority in `mold_inference::device` so the
    // engine's block-offload policy and this estimate cannot drift apart; the
    // constants and their fit are documented there and above.
    mold_inference::device::wan_calibrated_activation_bytes(
        shape.width,
        shape.height,
        frames,
        geometry,
        shape.cfg,
    )
}

/// Shared UMT5-XXL fp16 encoder.
///
/// Sequential weight peak is `max(encoder, transformer + vae)` — the encoder
/// is dropped before the transformer denoises — and for every wan tier the
/// encoder is the larger term, so it is the floor every render pays.
#[cfg(test)]
const WAN_UMT5_FP16_BYTES: u64 = 11_366_399_385;

/// One A14B expert at Q5_K_M, the tier the measured anchor was taken on.
///
/// VRAM is the max over the pair, not the sum — `WanExperts` drops the
/// resident expert before loading its partner
/// (`crates/mold-inference/src/wan/experts.rs:18-21`). Disk is the sum, which
/// is why a file-size estimator that adds both would reject a 53-frame render
/// that demonstrably runs.
#[cfg(test)]
const WAN_A14B_Q5_EXPERT_BYTES: u64 = 10_790_416_896;

/// Geometry for a checkpoint, delegated to the engine's own probe.
///
/// `wan::pipeline::activation_geometry` reads through `header_shapes`, which
/// handles GGUF and safetensors alike. A second parser here read only
/// safetensors, so every GGUF tier — `wan22-{t2v,i2v}-a14b:{q4,q5,q8}` and
/// `wan22-ti2v-5b:q8` — silently fell back to the A14B shape. For
/// `wan22-ti2v-5b:q8` that is a 4x token over-count (dim 3072 / 32 px per
/// token axis priced as dim 5120 / 16 px), which would make the tier that
/// exists for 8-12 GB cards unadmittable anywhere.
///
/// Reads every file the DiT spans, not just the primary: a diffusers export
/// splits the transformer without regard for the probe set, so the
/// output-channel probe can live alone in a later shard. Missing it makes the
/// geometry unknown and falls the estimate back to the conservative A14B
/// shape — which for the sharded 5B is a refusal at ~67 GB against ~24.8 GB
/// usable, for a checkpoint that actually renders.
pub(crate) fn wan_geometry_from_header(paths: &ModelPaths) -> Option<WanActivationGeometry> {
    mold_inference::wan::pipeline::activation_geometry_across(
        &mold_inference::wan::pipeline::transformer_files(paths),
    )
}

/// The file whose identity keys the geometry cache: the first of the DiT's
/// own files that exists. Sharded checkpoints have no single "the"
/// transformer, and keying on a path that is not on disk would never cache.
fn cache_identity(paths: &ModelPaths) -> PathBuf {
    mold_inference::wan::pipeline::transformer_files(paths)
        .into_iter()
        .next()
        .unwrap_or_else(|| paths.transformer.clone())
}

/// Identity of a checkpoint file, so a re-pull at the same path invalidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheKey {
    len: u64,
    modified_ns: u128,
}

fn cache_key(path: &Path) -> Option<CacheKey> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(CacheKey {
        len: metadata.len(),
        modified_ns: metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|since| since.as_nanos())
            .unwrap_or(0),
    })
}

type FactsCache = RwLock<HashMap<PathBuf, (CacheKey, WanActivationGeometry)>>;

fn facts_cache() -> &'static FactsCache {
    static CACHE: OnceLock<FactsCache> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// Cache-only lookup: no filesystem body read, no parse.
///
/// Safe on the scheduler coordinator thread, which is why the estimate path
/// calls this and never [`warm_checkpoint_geometry`]. A miss falls back to the
/// conservative A14B shape for that one estimate rather than blocking the
/// coordinator on a GGUF header parse.
pub(crate) fn checkpoint_geometry_cached(paths: &ModelPaths) -> Option<WanActivationGeometry> {
    let identity = cache_identity(paths);
    let key = cache_key(&identity)?;
    let cache = facts_cache().read().ok()?;
    cache
        .get(&identity)
        .filter(|(cached, _)| cached == &key)
        .map(|(_, geometry)| *geometry)
}

/// Parse and cache one checkpoint's geometry.
///
/// Blocking: reads and parses the checkpoint header. Call from
/// `spawn_blocking` or a worker thread, never from the coordinator. A negative
/// result is deliberately not cached — a placement preview can run against a
/// checkpoint whose bytes have not landed yet, and pinning that path to the
/// fallback for the process lifetime would outlive the download.
pub(crate) fn warm_checkpoint_geometry(paths: &ModelPaths) -> Option<WanActivationGeometry> {
    let identity = cache_identity(paths);
    let key = cache_key(&identity)?;
    if let Some(hit) = checkpoint_geometry_cached(paths) {
        return Some(hit);
    }
    let geometry = wan_geometry_from_header(paths)?;
    if let Ok(mut cache) = facts_cache().write() {
        cache.insert(identity, (key, geometry));
    }
    Some(geometry)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(width: u32, height: u32, frames: u32, guidance: f64) -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a red fox running through snow",
            "model": "wan22-t2v-a14b:q5",
            "width": width,
            "height": height,
            "frames": frames,
            "guidance": guidance,
            "steps": 20,
            "seed": 42,
        }))
        .expect("synthetic wan request")
    }

    #[test]
    fn shape_hint_reads_frames_and_keys_cfg_on_guidance_alone() {
        let hint = WanShapeHint::from_request(&request(832, 480, 53, 3.5));
        assert_eq!(hint.frames, 53);
        assert!(hint.cfg);

        // The tuned default negative is applied engine-side, so an absent
        // negative still runs the uncond forward. Gating on
        // `negative_prompt.is_some()` — which the generic estimate does — would
        // price this as a single forward.
        let mut absent_negative = request(832, 480, 53, 3.5);
        absent_negative.negative_prompt = None;
        assert!(WanShapeHint::from_request(&absent_negative).cfg);

        // Guidance <= 1 is the Lightning recipe: one forward per step.
        assert!(!WanShapeHint::from_request(&request(832, 480, 53, 1.0)).cfg);
    }

    #[test]
    fn frames_move_the_estimate() {
        let geometry = WanActivationGeometry::a14b();
        let short = wan_activation_bytes(
            WanShapeHint::from_request(&request(832, 480, 17, 1.0)),
            geometry,
        );
        let long = wan_activation_bytes(
            WanShapeHint::from_request(&request(832, 480, 81, 1.0)),
            geometry,
        );
        assert!(
            long > short,
            "the whole point of the bespoke model: {short} -> {long}"
        );
    }

    #[test]
    fn cfg_adds_a_bounded_term_rather_than_doubling() {
        // Measured: +512 MiB on the 1.3B reference shape, against a working
        // set of ~2 GB. A multiplier would reject shapes that demonstrably run.
        let geometry = WanActivationGeometry::a14b();
        let without = wan_activation_bytes(
            WanShapeHint::from_request(&request(832, 480, 53, 1.0)),
            geometry,
        );
        let with = wan_activation_bytes(
            WanShapeHint::from_request(&request(832, 480, 53, 3.5)),
            geometry,
        );
        assert_eq!(
            with - without,
            mold_inference::device::WAN_CFG_RESIDENT_BYTES
        );
        assert!(
            with < without * 2,
            "CFG must not be priced as a doubling: {without} -> {with}"
        );
    }

    /// Every measured point, not just the one the constants were fitted to.
    ///
    /// The acceptance criterion for #774 is ~15% against the recorded anchor,
    /// but one point cannot distinguish a steep slope with a small flat term
    /// from a shallow slope with a large one — an earlier revision of this
    /// model fitted three free constants to a single anchor, matched it to
    /// 1%, and over-predicted an independent checkpoint by 26%.
    ///
    /// Four points, all RTX 4090:
    /// * `wan22-t2v-a14b:q5` 17f/832x480 CFG off — 16,214 MiB (measured here)
    /// * `wan22-t2v-a14b:q5` 53f/832x480 CFG off — 23,030 MiB (measured here)
    /// * `wan22-t2v-a14b:q5` 53f/832x480 CFG on  — 23,975 MiB (manifest.rs)
    /// * `wan22-ti2v-5b:q8` 121f/1280x704 CFG on — 18,460 MiB (manifest.rs)
    ///
    /// The first three set the slope, the flat term, and the CFG term. The
    /// fourth is a different width, a different VAE generation, and a
    /// different quantization — it validates rather than calibrates, so its
    /// error is the honest measure of how far this generalizes.
    #[test]
    fn every_measured_point_lands_within_fifteen_percent() {
        const A14B_Q5_VAE_BYTES: u64 = 500_000_000;
        const TI2V_5B_Q8_TRANSFORMER_BYTES: u64 = 5_400_179_040;

        struct Point {
            label: &'static str,
            weights: u64,
            geometry: WanActivationGeometry,
            width: u32,
            height: u32,
            frames: u32,
            guidance: f64,
            measured_mib: u64,
        }

        let points = [
            Point {
                label: "A14B q5 17f CFG off",
                weights: WAN_A14B_Q5_EXPERT_BYTES + A14B_Q5_VAE_BYTES,
                geometry: WanActivationGeometry::a14b(),
                width: 832,
                height: 480,
                frames: 17,
                guidance: 1.0,
                measured_mib: 16_074,
            },
            Point {
                label: "A14B q5 53f CFG off",
                weights: WAN_A14B_Q5_EXPERT_BYTES + A14B_Q5_VAE_BYTES,
                geometry: WanActivationGeometry::a14b(),
                width: 832,
                height: 480,
                frames: 53,
                guidance: 1.0,
                measured_mib: 21_354,
            },
            Point {
                label: "A14B q5 53f CFG on",
                weights: WAN_A14B_Q5_EXPERT_BYTES + A14B_Q5_VAE_BYTES,
                geometry: WanActivationGeometry::a14b(),
                width: 832,
                height: 480,
                frames: 53,
                guidance: 3.5,
                measured_mib: 21_322,
            },
            Point {
                label: "TI2V-5B q8 121f CFG on",
                weights: TI2V_5B_Q8_TRANSFORMER_BYTES + A14B_Q5_VAE_BYTES,
                geometry: WanActivationGeometry::ti2v_5b(),
                width: 1280,
                height: 704,
                frames: 121,
                guidance: 5.0,
                measured_mib: 18_794,
            },
        ];

        for point in points {
            let measured = point.measured_mib * 1024 * 1024;
            // The real admission arithmetic: the Sequential weight peak (which
            // for wan is the encoder floor) plus this module's activation
            // term. An earlier revision of this test summed the transformer
            // and the activation instead, which is not what any code path
            // computes — it under-stated the prediction by the encoder and let
            // a shape that admission actually refused look admissible here.
            let predicted = point.weights.max(WAN_UMT5_FP16_BYTES)
                + wan_activation_bytes(
                    WanShapeHint::from_request(&request(
                        point.width,
                        point.height,
                        point.frames,
                        point.guidance,
                    )),
                    point.geometry,
                );
            let low = measured * 85 / 100;
            let high = measured * 115 / 100;
            assert!(
                (low..=high).contains(&predicted),
                "{}: predicted {predicted} is outside 15% of the measured {measured}",
                point.label,
            );
        }
    }

    /// The whole point of #774: 53 frames ran on the 4090 and 81 frames OOM'd
    /// mid-denoise, so the model must admit the first and refuse the second.
    ///
    /// Both are the same resolution and the same tier — frames is the only
    /// axis separating them, and the pixel-area estimate this replaces priced
    /// them identically.
    ///
    /// The budget here is the whole card, not 90% of it: wan takes the
    /// un-derated cap alongside Qwen-Image because it is phase-sequential and
    /// its estimate is measured rather than heuristic. Derating a measured
    /// peak would refuse the tier's own shipped default — 53 frames measures
    /// 23,975 MiB, which is 95% of a 24 GB card and ran.
    #[test]
    fn eighty_one_frames_is_refused_where_fifty_three_is_admitted_on_24gb() {
        const CARD_BYTES: u64 = 24_564 * 1024 * 1024;
        const A14B_Q5_VAE_BYTES: u64 = 500_000_000;
        let geometry = WanActivationGeometry::a14b();
        let peak = |frames: u32| {
            (WAN_A14B_Q5_EXPERT_BYTES + A14B_Q5_VAE_BYTES).max(WAN_UMT5_FP16_BYTES)
                + wan_activation_bytes(
                    WanShapeHint::from_request(&request(832, 480, frames, 3.5)),
                    geometry,
                )
        };
        assert!(
            peak(53) <= CARD_BYTES,
            "53f is the shipped default and demonstrably runs; admission must not refuse it: \
             {} > {CARD_BYTES}",
            peak(53)
        );
        assert!(
            peak(81) > CARD_BYTES,
            "81f OOM'd on this card and must be refused before the encode and expert load: \
             {} <= {CARD_BYTES}",
            peak(81)
        );
    }

    /// `(e.g. ':q8')` is circular for a request that just failed on `:q8`.
    #[test]
    fn rejection_advice_names_a_tier_the_user_is_not_already_on() {
        use crate::memory_preflight::rejection_suggestion_for_model;
        use mold_inference::device::ActivationFamily;

        let wan = crate::memory_preflight::ActivationHint {
            width: 832,
            height: 480,
            batch: 1,
            dtype_bytes: 2,
            family: ActivationFamily::WanVideo,
        };

        let q8 = rejection_suggestion_for_model(Some(wan), "wan22-t2v-a14b:q8");
        assert!(q8.contains("':q5'"), "{q8}");
        assert!(
            !q8.contains("':q8'"),
            "must not suggest the failing tier: {q8}"
        );

        let q5 = rejection_suggestion_for_model(Some(wan), "wan22-t2v-a14b:q5");
        assert!(q5.contains("':q4'"), "{q5}");

        // Nothing smaller ships: say so instead of naming a tier that is not
        // there.
        let q4 = rejection_suggestion_for_model(Some(wan), "wan22-t2v-a14b:q4");
        assert!(q4.contains("smallest quantized tier"), "{q4}");
        assert!(!q4.contains("':q"), "{q4}");

        // Every video rejection leads with frames, the dominant lever.
        for advice in [&q8, &q5, &q4] {
            assert!(advice.contains("--frames"), "{advice}");
        }

        // An unquantized tier is not told to try a specific tag — the advice
        // cannot know one is published for that checkpoint.
        let fp16 = rejection_suggestion_for_model(Some(wan), "wan22-ti2v-5b:fp16");
        assert!(!fp16.contains("':q"), "{fp16}");
        assert!(fp16.contains("if one is published"), "{fp16}");

        // Every shipped LTX-Video tier is `:bf16` — naming ':q8' there sends
        // the user after a variant that does not exist.
        let ltx = crate::memory_preflight::ActivationHint {
            family: ActivationFamily::LtxVideo,
            ..wan
        };
        let ltx_advice = rejection_suggestion_for_model(Some(ltx), "ltx-video-0.9.6:bf16");
        assert!(!ltx_advice.contains("':q"), "{ltx_advice}");

        // An opaque catalog id must not have a repo path read as a tier.
        let opaque = rejection_suggestion_for_model(Some(wan), "hf:Wan-AI/Wan2.2-T2V-A14B");
        assert!(!opaque.contains("':q"), "{opaque}");
    }

    #[test]
    fn an_omitted_frame_count_is_priced_as_the_clip_the_engine_renders() {
        let mut omitted: mold_core::GenerateRequest = request(832, 480, 53, 3.5);
        omitted.frames = None;
        let hint = WanShapeHint::from_request(&omitted);

        // 2.1 fills 81, 2.2 fills 121 — not one latent frame.
        let a14b = wan_activation_bytes(hint, WanActivationGeometry::a14b());
        let explicit_81 = wan_activation_bytes(
            WanShapeHint::from_request(&request(832, 480, 81, 3.5)),
            WanActivationGeometry::a14b(),
        );
        assert_eq!(a14b, explicit_81);

        let ti2v = wan_activation_bytes(hint, WanActivationGeometry::ti2v_5b());
        let explicit_121 = wan_activation_bytes(
            WanShapeHint::from_request(&request(832, 480, 121, 3.5)),
            WanActivationGeometry::ti2v_5b(),
        );
        assert_eq!(ti2v, explicit_121);

        // A single-frame still is far cheaper, which is what the old
        // `unwrap_or(1)` priced a full clip at.
        let single = wan_activation_bytes(
            WanShapeHint::from_request(&request(832, 480, 1, 3.5)),
            WanActivationGeometry::a14b(),
        );
        assert!(single < a14b);
    }

    /// A `ModelPaths` naming only the DiT files, which is all the geometry
    /// probe reads.
    fn dit_paths(transformer: &Path, shards: &[PathBuf]) -> ModelPaths {
        ModelPaths {
            transformer: transformer.to_path_buf(),
            transformer_shards: shards.to_vec(),
            low_noise_transformer: None,
            vae: PathBuf::new(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            low_noise_distilled_lora: None,
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

    #[test]
    fn geometry_is_rejected_rather_than_guessed_for_a_non_wan_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-wan.safetensors");
        std::fs::write(&path, b"not a safetensors file at all").unwrap();
        assert!(wan_geometry_from_header(&dit_paths(&path, &[])).is_none());
    }

    /// A sharded checkpoint's geometry probe must read every shard.
    ///
    /// The published `Wan2.2-TI2V-5B-Turbo-Diffusers` splits the DiT so the
    /// output-channel probe (`proj_out.weight`) sits alone in an 89 MB second
    /// shard. Reading only `paths.transformer` leaves the geometry unknown,
    /// and the estimate falls back to the conservative A14B shape — which
    /// refused that checkpoint at ~67 GB against ~24.8 GB usable.
    #[test]
    fn the_geometry_probe_reads_every_shard_not_just_the_primary() {
        let dir = tempfile::tempdir().unwrap();
        let primary = dir.path().join("shard-1.safetensors");
        let second = dir.path().join("shard-2.safetensors");
        std::fs::write(&primary, b"x").unwrap();
        std::fs::write(&second, b"y").unwrap();

        let files = mold_inference::wan::pipeline::transformer_files(&dit_paths(
            &primary,
            &[primary.clone(), second.clone()],
        ));
        assert_eq!(
            files,
            vec![primary.clone(), second.clone()],
            "both shards must reach the probe, and the primary must not repeat",
        );

        // A path that is not on disk is not silently dropped to an empty set:
        // the caller still gets something to fail by name on.
        let missing = dir.path().join("absent.safetensors");
        assert_eq!(
            mold_inference::wan::pipeline::transformer_files(&dit_paths(&missing, &[])),
            vec![missing],
        );
    }
}
