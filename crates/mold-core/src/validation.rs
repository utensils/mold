use crate::{
    GenerateRequest, KeyframeCondition, LoraWeight, Ltx2GuidanceOverrides, Ltx2PipelineMode,
    Ltx2SpatialUpscale, OutputFormat, UpscaleRequest,
};

/// Maximum total pixels allowed (~1.8 megapixels). Qwen-Image trains at ~1.6MP
/// (1328x1328), other models at ≤1MP. Headroom for non-square aspect ratios.
pub const MAX_PIXELS: u64 = 1_800_000;
/// LTX-2's own ceiling: upstream's shipped `LTX_2_3_HQ_PARAMS` renders
/// 1920x1088 (stage 1 at 960x544, refined x2), which is 2,088,960 px. The
/// flat 1.8 MP limit made mold unable to express the reference
/// implementation's own top-end preset.
pub const LTX2_MAX_PIXELS: u64 = 1_920 * 1_088;
/// Per-axis span, independent of the pixel budget.
///
/// The checkpoints ship `positional_embedding_max_pos = [20, 2048, 2048]` and
/// RoPE normalizes pixel positions by it, so an axis past 2048 lands outside
/// the trained [-1, 1] range with no error raised. 3200x512 is only 1.64 MP
/// and still out of distribution. Going beyond this needs tiled stage-2
/// refinement with renormalized positions, not a larger single denoise —
/// see [`LTX2_COMPOSED_MAX_AXIS_PIXELS`].
pub const LTX2_MAX_AXIS_PIXELS: u32 = 2_048;

/// Per-axis ceiling for a render that *composes* its output: stage 1 at half
/// the target, one x2 spatial rung, then a tiled stage-2 refinement.
///
/// This is `2 * LTX2_MAX_AXIS_PIXELS` and the factor is not a safety margin —
/// it is exactly where the composition stops working. Stage 1 renders the
/// target halved (`derive_stage1_render_shape`), so a 4096 px target puts
/// stage 1 at 2048 px, the last shape still inside the trained span. Stage 2
/// is tiled, and a tile is always brought back inside that span, so stage 2
/// itself imposes no ceiling. mold applies at most one spatial rung, so there
/// is no second halving to rescue a wider target: past 4096 px, stage 1 is out
/// of distribution and no amount of tiling downstream repairs it.
pub const LTX2_COMPOSED_MAX_AXIS_PIXELS: u32 = 2 * LTX2_MAX_AXIS_PIXELS;

/// Total-pixel ceiling for a composed LTX-2 render (`4096 x 2176`, 8.9 MP).
///
/// Like the single-pass budget this is a resource guard rather than a model
/// limit: it is the widest axis the composition can hold paired with a
/// 4K-class height. It is deliberately *above* the top of
/// [`LTX2_OUTPUT_RUNGS`], which stops at what the bundled H.264 encoder can
/// write — generation and delivery have different ceilings, and conflating
/// them would refuse shapes that render correctly to a non-MP4 target.
pub const LTX2_COMPOSED_MAX_PIXELS: u64 = 4_096 * 2_176;
pub const MAX_INLINE_AUDIO_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_INLINE_SOURCE_VIDEO_BYTES: usize = 64 * 1024 * 1024;
pub const FLUX2_DEV_MAX_REFERENCE_IMAGES: usize = 4;
/// BFL's pixel cap for a single FLUX.2 Dev reference. The upstream value is
/// intentionally 2024 squared, not 2048 squared.
pub const FLUX2_DEV_SINGLE_REFERENCE_MAX_PIXELS: u64 = 2_024 * 2_024;
/// BFL's per-image pixel cap when a FLUX.2 Dev request has multiple references.
pub const FLUX2_DEV_MULTI_REFERENCE_MAX_PIXELS: u64 = 1_024 * 1_024;
pub const LORA_CAPABLE_FAMILIES: &[&str] = &[
    "flux",
    "flux2",
    "ltx2",
    "sd15",
    "sd3",
    "sdxl",
    "qwen-image",
    "qwen-image-edit",
    "wan",
    "z-image",
];

pub fn family_supports_lora(family: &str) -> bool {
    LORA_CAPABLE_FAMILIES.contains(&family)
}

/// Temporal RoPE budget for LTX-2 / LTX-2.3, **in seconds of video runtime**.
///
/// The checkpoints ship `pos_embed_max_pos = 20`, and both upstream `ltx_core`
/// and mold's own RoPE path convert the temporal axis to *seconds* before
/// normalizing by it: `ltx2/model/rope.rs`'s `scale_video_time_to_seconds`
/// divides the pixel-frame coordinate by the request's fps. So `20` bounds
/// twenty seconds of runtime, not twenty latent frames — which is exactly the
/// ~20 s single-generation duration Lightricks advertises for LTX-2.3.
pub const LTX2_MAX_RUNTIME_SECONDS: u32 = 20;

/// fps assumed for LTX-2 when a caller must name a frame ceiling without a
/// request in hand (the `/api/models` scalar fallback, and requests that leave
/// `fps` unset for the server to fill in). Matches the manifest default.
pub const LTX2_DEFAULT_FPS: u32 = 24;

/// Absolute pixel-frame ceiling for LTX-2 regardless of fps.
///
/// This is a resource guard, not a model limit: the seconds budget alone would
/// admit 2404 frames at the maximum allowed 120 fps, which no current GPU can
/// denoise in one pass. 604 is `LTX2_MAX_RUNTIME_SECONDS` at 30 fps — the point
/// where a practical frame budget meets the model's real duration budget.
pub const LTX2_MAX_FRAMES_ABSOLUTE: u32 = LTX2_MAX_RUNTIME_SECONDS * 30 + 4;

/// Global frame ceiling for video families that do not publish their own
/// duration budget (currently `ltx-video`).
pub const MAX_FRAMES_GLOBAL: u32 = 257;

/// Default pixel-frame overlap for `extend_video` on LTX-2 — and the fallback
/// for a family whose carryover cannot be resolved — matching the chain
/// motion-tail default so an extend seam and a sequence seam behave the same.
/// 17 pixel frames is three LTX-2 latent frames under the VAE's 8x causal
/// temporal compression. Resolve it through
/// [`default_extend_overlap_frames_for_family`] rather than reading it
/// directly: it is not the answer for every extend-capable family.
pub const DEFAULT_EXTEND_OVERLAP_FRAMES: u32 = 17;

/// Pixel-frame overlap an `extend_video` request gets when it names none.
///
/// The default is a property of the family's *carryover*, never a global
/// scalar. Wan's continuation is seeded with one frame and its engine refuses
/// any other overlap, so advertising LTX-2's 17 handed wan clients a value
/// that clears wan's `4k+1` grid check at admission and then fails inside the
/// engine, after the model load had already been paid for (#783).
pub fn default_extend_overlap_frames_for_family(family: Option<&str>) -> u32 {
    match family {
        Some("wan") => WAN_HANDOFF_DUPLICATED_FRAMES,
        _ => DEFAULT_EXTEND_OVERLAP_FRAMES,
    }
}

/// Write the family's own carryover into a continuation that named no overlap.
///
/// This is a mutation rather than a read at the point of use because of
/// *provenance*. [`crate::OutputMetadata::from_generate_request`] records what
/// rendered, and it holds no family — it resolves one through the manifest,
/// which an installed `cv:` / `hf:` wan checkpoint does not have. A wan
/// continuation that ran with one carryover frame was therefore saved as
/// having used LTX-2's 17 (#783). Server admission and the forced-local CLI
/// path both know the resolved family, so both fill the field in before
/// anything reads it — the same seam `materialize_default_negative_prompt`
/// uses for the wan uncond.
///
/// An explicit value is authoritative and passes through untouched, and a
/// non-extend request is never given one: a bare `extend_overlap_frames` is a
/// validation error, not a default.
pub fn materialize_extend_overlap_frames(req: &mut GenerateRequest, family: Option<&str>) {
    if req.is_extend() && req.extend_overlap_frames.is_none() {
        req.extend_overlap_frames = Some(default_extend_overlap_frames_for_family(family));
    }
}

/// The motion-tail overlap a chain seam actually renders with.
///
/// The tail is a property of the family's carryover and, for wan, of the
/// selected *checkpoint's* conditioning contract — never of what the caller
/// asked for. Wan has no latent motion tail: its seam re-renders exactly the
/// one frame the continuation was seeded with, and only an image-conditioned
/// checkpoint can be seeded at all. LTX-Video has no img2vid path, so its
/// Smooth boundaries collapse to clean concatenation.
///
/// This lives here, beside [`WAN_HANDOFF_DUPLICATED_FRAMES`], because the
/// server had been the only caller that normalized it (#936). The forced-local
/// `--script` path, `mold chain validate`, and `--dry-run` all ran the
/// family-generic `ChainRequest::normalise` and passed the requested tail
/// through untouched — and `17 % 4 == 1`, so LTX-2's default clears wan's own
/// `4k+1` grid check and then discards sixteen good frames at every Smooth
/// seam, with correct-looking validation output (#783).
///
/// `source_image` is the resolved contract — probed from the checkpoint's own
/// headers where possible, the manifest as the cold fallback. `None` is
/// "unknown", which takes the conservative path rather than assuming a
/// handoff. Families with a real latent window keep what the caller asked for.
pub fn chain_motion_tail_frames_for_family(
    family: &str,
    source_image: Option<crate::SourceImageCapability>,
    requested: u32,
) -> u32 {
    match family {
        "wan" => {
            let carries_context = source_image.is_some_and(|capability| {
                matches!(
                    capability,
                    crate::SourceImageCapability::Required | crate::SourceImageCapability::Optional
                )
            });
            if carries_context {
                WAN_HANDOFF_DUPLICATED_FRAMES
            } else {
                0
            }
        }
        "ltx-video" => 0,
        _ => requested,
    }
}

/// Inline `extend_video` payloads share the source-video body budget.
pub const MAX_INLINE_EXTEND_VIDEO_BYTES: usize = MAX_INLINE_SOURCE_VIDEO_BYTES;

/// Upper bound for a requested STG block index. The deepest LTX-2 transformer
/// mold runs has 48 layers; the ceiling is loose on purpose because the exact
/// depth is a property of the resolved checkpoint, which validation does not
/// have. The engine rejects an index the loaded transformer does not have.
pub const MAX_STG_BLOCK_INDEX: u32 = 64;

/// Maximum number of simultaneously perturbed STG blocks. Every extra block
/// deepens the perturbed pass; upstream configurations use one or two.
pub const MAX_STG_BLOCKS: usize = 8;

/// Largest pixel-frame count whose final RoPE token still lands inside the
/// LTX-2 temporal budget at `fps`.
///
/// After the causal first-frame fix, latent frame `k` spans pixel bounds
/// `[8k - 7, 8k + 1]`, so the midpoint the RoPE grid actually sees is
/// `(8k - 3) / fps` seconds. `F` pixel frames on the `8n + 1` grid put the last
/// latent at `k = (F - 1) / 8`, giving a midpoint of `(F - 4) / fps`. Requiring
/// that to stay within the budget yields `F <= seconds * fps + 4`.
pub fn ltx2_max_frames_at_fps(fps: u32) -> u32 {
    LTX2_MAX_RUNTIME_SECONDS
        .saturating_mul(fps.max(1))
        .saturating_add(4)
        .min(LTX2_MAX_FRAMES_ABSOLUTE)
}

/// [`ltx2_max_frames_at_fps`] snapped down onto the `8n+1` grid the validator
/// actually enforces.
///
/// The raw cap is not requestable: `20 * 24 + 4 = 484` and `483 % 8 == 3`, so a
/// client that clamps a slider to the advertised maximum and submits gets a
/// 422. At 48 fps the absolute guard bites first — 964 clamps to 604, which is
/// equally off-grid — so this matters at every rate, not just the default.
pub fn ltx2_max_frames_on_grid_at_fps(fps: u32) -> u32 {
    snap_frames_to_8k1(ltx2_max_frames_at_fps(fps))
}

/// Spatial alignment a two-stage LTX-2 render needs. Stage 1 renders at half
/// the requested size, so both axes must survive the halving and still land on
/// the VAE's 32-pixel latent grid. Mirrors upstream `assert_resolution`'s
/// `divisor = 64 if is_two_stage else 32`
/// (`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:326`).
pub const LTX2_TWO_STAGE_ALIGNMENT: u32 = 64;

/// The VAE's causal temporal compression factor: latent frame 0 covers one
/// pixel frame and every later latent frame covers eight, so a renderable
/// pixel-frame count is always `8k + 1`.
pub const LTX2_TEMPORAL_SCALE: u32 = 8;

/// Round `frames` **down** onto the `8k + 1` grid LTX-2 can actually render.
///
/// Mirrors upstream `_snap_frames_to_8k1`
/// (`packages/ltx-pipelines/src/ltx_pipelines/lipdub.py:46-49`). Rounding down
/// matters for lip-dub: the reference clip's frame count is whatever the
/// camera produced, and rounding *up* would ask for frames the reference does
/// not have.
pub fn snap_frames_to_8k1(frames: u32) -> u32 {
    if frames <= 1 {
        return 1;
    }
    frames - ((frames - 1) % LTX2_TEMPORAL_SCALE)
}

/// Wan's causal video VAE compresses time by 4: latent frame 0 covers one
/// pixel frame and every later latent frame covers four, so a renderable
/// pixel-frame count is always `4k + 1` (upstream enforces the same grid in
/// `Wan2.1/generate.py`).
pub const WAN_TEMPORAL_SCALE: u32 = 4;

/// The pixel frames a wan continuation duplicates from the clip before it.
///
/// Wan has no latent motion tail. Its handoff is last-frame *image*
/// conditioning: the continuation is seeded with the previous clip's final
/// frame, so it re-renders exactly that one frame and the stitch trims exactly
/// one. This is deliberately not LTX-2's 17 — that number is the pixel window
/// its VAE turns into three latent slots of carryover, which wan has no
/// equivalent of, and copying it would discard sixteen good frames per seam.
///
/// It lives in `mold-core` because it is the value the whole stack derives
/// from — admission, `/api/models`, the CLI's chain planner, and the engine
/// gate that enforces it (`mold_inference::wan::pipeline` re-exports this).
pub const WAN_HANDOFF_DUPLICATED_FRAMES: u32 = 1;

/// Smallest clip `wan22-ti2v-5b` first/last-frame conditioning accepts. TI2V
/// pins both endpoints in latent space, where the 2.2 VAE's 4x temporal
/// stride turns a 5-frame pixel clip into two latent frames — both anchored,
/// nothing left to denoise. Nine pixel frames (three latent frames) is the
/// smallest `4k + 1` clip with an interior. Admission, the CLI, Discord, and
/// the studio surfaces (`studio/lib/sourceImageCapability.ts`) all enforce
/// this same floor before dispatch; the shared fixture
/// `tests/fixtures/wan/surface-parity-v1.json` pins them together (#806).
pub const WAN_TI2V_FLF_MIN_FRAMES: u32 = 9;

/// The frame count and rate a lip-dub render must use, plus anything the
/// caller asked for that the reference video overrode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LipDubTiming {
    /// Reference frame count snapped down onto the `8k + 1` grid.
    pub frames: u32,
    /// The reference clip's own frame rate.
    pub fps: u32,
    /// Human-readable notes about requested values that were replaced.
    /// Empty when the caller asked for exactly what the reference provides.
    pub warnings: Vec<String>,
}

/// What a probe of the lip-dub reference clip reports.
///
/// A struct rather than positional arguments so every property the pipeline
/// depends on is supplied by name, and so adding one is a compile error at
/// every call site — which is what stops the server and forced-local paths
/// validating different things.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LipDubReference {
    pub frames: u32,
    pub fps: u32,
    /// Whether the clip carries a decodable audio stream. Lip dub imitates the
    /// reference speaker's voice, so a silent reference cannot drive one.
    pub has_audio: bool,
}

/// Resolve a lip-dub render's parameters from its reference video, rejecting a
/// reference that cannot drive one.
///
/// Lip-dub re-voices an existing clip, so the output must land on the
/// reference's own timeline: upstream reads both the frame count and the frame
/// rate straight off the reference stream
/// (`packages/ltx-pipelines/src/ltx_pipelines/lipdub.py:190-192`) rather than
/// taking them from the caller. mold keeps the same rule, but says so out loud
/// when a client asked for something else — a silently retimed dub looks fine
/// and is out of sync.
/// Every precondition lives here rather than at the call sites, so a request
/// that cannot succeed is refused before it is validated, scheduled, and
/// granted VRAM — not several minutes later when the audio VAE has nothing to
/// encode.
pub fn resolve_lip_dub_timing(
    reference: LipDubReference,
    requested_frames: Option<u32>,
    requested_fps: Option<u32>,
) -> Result<LipDubTiming, String> {
    let LipDubReference {
        frames: reference_frames,
        fps: reference_fps,
        has_audio,
    } = reference;
    if reference_fps == 0 {
        return Err("lip-dub reference video reports a frame rate of 0".to_string());
    }
    // Upstream raises on a reference with no audio stream
    // (`lipdub.py:166-170`). Catching it at the request boundary rather than at
    // decode time is the difference between a 422 and a queued job that dies
    // after loading a 22B checkpoint.
    if !has_audio {
        return Err(
            "lip-dub reference video has no audio track; the pipeline re-voices existing \
             speech, so the reference must contain some"
                .to_string(),
        );
    }
    let frames = snap_frames_to_8k1(reference_frames);
    if frames < 9 {
        return Err(format!(
            "lip-dub reference video is too short: {reference_frames} frames snap down to \
             {frames}, and the pipeline needs at least 9"
        ));
    }
    let mut warnings = Vec::new();
    if requested_frames.is_some_and(|requested| requested != frames) {
        warnings.push(format!(
            "lip-dub takes its length from the reference video: rendering {frames} frames \
             instead of the requested {}",
            requested_frames.unwrap_or_default()
        ));
    } else if requested_frames.is_none() && frames != reference_frames {
        warnings.push(format!(
            "lip-dub snapped the reference video's {reference_frames} frames down to {frames} \
             (LTX-2 renders 8k+1 frames)"
        ));
    }
    if requested_fps.is_some_and(|requested| requested != reference_fps) {
        warnings.push(format!(
            "lip-dub takes its frame rate from the reference video: rendering at \
             {reference_fps} fps instead of the requested {}",
            requested_fps.unwrap_or_default()
        ));
    }
    Ok(LipDubTiming {
        frames,
        fps: reference_fps,
        warnings,
    })
}

/// Per-family single-request frame ceiling at `fps` — the value `/api/models`
/// advertises as `max_frames`. Must stay in agreement with
/// `validate_generate_request`'s rejections, which consume this helper.
///
/// LTX-2's ceiling is a duration, so it moves with fps; every other video
/// family reports the flat global ceiling.
pub fn max_frames_for_family_at_fps(family: &str, fps: u32) -> Option<u32> {
    match family {
        // Advertise the value a client can actually submit. The raw duration
        // ceiling sits off the `8n+1` grid at every fps, so a slider clamped
        // to it produced a 422.
        "ltx2" => Some(ltx2_max_frames_on_grid_at_fps(fps)),
        "ltx-video" => Some(MAX_FRAMES_GLOBAL),
        // Wan's temporal RoPE is indexed by latent frame against a 1024-entry
        // table, so duration is nowhere near the binding limit — memory is.
        // The flat global ceiling is the resource guard, and 257 sits on the
        // `4k+1` grid, so the advertised maximum is itself submittable.
        "wan" => Some(MAX_FRAMES_GLOBAL),
        family if crate::minimax_h3::is_family(family) => Some(crate::minimax_h3::MAX_FRAMES),
        _ => None,
    }
}

/// `max_frames_for_family_at_fps` at each family's default fps, for callers
/// that have no per-model fps to hand.
pub fn max_frames_for_family(family: &str) -> Option<u32> {
    max_frames_for_family_at_fps(family, LTX2_DEFAULT_FPS)
}

/// Minimum requestable frame count for families that impose one above the
/// generic single-frame floor. `None` retains the historical minimum of one.
pub fn min_frames_for_family(family: &str) -> Option<u32> {
    crate::minimax_h3::is_family(family).then_some(crate::minimax_h3::MIN_FRAMES)
}

/// A family's mandatory frame rate, when the checkpoint does not support
/// arbitrary FPS. `None` means callers may choose any otherwise-valid rate.
pub fn fixed_fps_for_family(family: &str) -> Option<u32> {
    crate::minimax_h3::is_family(family).then_some(crate::minimax_h3::FIXED_FPS)
}

/// Single-request runtime ceiling in seconds for families whose real limit is
/// a duration. `None` means the family's ceiling is a plain frame count.
pub fn max_runtime_seconds_for_family(family: &str) -> Option<u32> {
    match family {
        "ltx2" => Some(LTX2_MAX_RUNTIME_SECONDS),
        family if crate::minimax_h3::is_family(family) => {
            Some(crate::minimax_h3::MAX_DURATION_SECONDS)
        }
        _ => None,
    }
}

/// fps-independent frame guard, paired with `max_runtime_seconds_for_family`.
pub fn max_frames_absolute_for_family(family: &str) -> Option<u32> {
    match family {
        "ltx2" => Some(LTX2_MAX_FRAMES_ABSOLUTE),
        family if crate::minimax_h3::is_family(family) => Some(crate::minimax_h3::MAX_FRAMES),
        _ => None,
    }
}

/// Step of the frame-count grid for a family. Pair with
/// [`frame_offset_for_family`]; valid counts are `k * step + offset`.
pub fn frame_step_for_family(family: &str) -> Option<u32> {
    match family {
        "ltx2" | "ltx-video" => Some(LTX2_TEMPORAL_SCALE),
        "wan" => Some(WAN_TEMPORAL_SCALE),
        family if crate::minimax_h3::is_family(family) => Some(crate::minimax_h3::FRAME_STEP),
        _ => None,
    }
}

/// Offset of the frame-count grid. Existing video families use 1; MiniMax H3
/// uses 5 (`17n+5`). `None` means the family has no temporal grid.
pub fn frame_offset_for_family(family: &str) -> Option<u32> {
    frame_step_for_family(family).map(|_| {
        if crate::minimax_h3::is_family(family) {
            crate::minimax_h3::FRAME_OFFSET
        } else {
            1
        }
    })
}

/// Validate family-specific temporal constraints that sit above the generic
/// non-zero FPS/frame checks. Public admission calls this only after the model
/// activation gate; keeping it factored lets the authority be tested without
/// introducing a test-only authorization bypass.
fn validate_family_video_timing_constraints(
    frames: Option<u32>,
    fps: Option<u32>,
    family: Option<&str>,
) -> Result<(), String> {
    if let (Some(family), Some(fps)) = (family, fps) {
        if let Some(fixed_fps) = fixed_fps_for_family(family) {
            if fps != fixed_fps {
                return Err(format!("{family} requires {fixed_fps} fps; received {fps}"));
            }
        }
    }
    if let (Some(family), Some(frames)) = (family, frames) {
        if let Some(min_frames) = min_frames_for_family(family) {
            if frames < min_frames {
                return Err(format!(
                    "frames ({frames}) must be >= {min_frames} for {family}"
                ));
            }
        }
    }
    Ok(())
}

fn megapixel_limit_label_for(limit: u64) -> String {
    format!("{:.1}MP", limit as f64 / 1_000_000.0)
}

/// How much spatial work a resolved LTX-2 render splits into.
///
/// This is the only thing that decides whether an axis past the trained RoPE
/// span is renderable, so it is resolved once — from the model and the
/// requested pipeline — rather than inferred separately by each surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ltx2SpatialComposition {
    /// One un-tiled denoise at the requested shape. The trained span is a hard
    /// ceiling: there is nothing downstream to renormalize positions.
    #[default]
    SinglePass,
    /// Stage 1 at half the target, one x2 spatial rung, then a stage-2
    /// refinement over latent tiles each brought back inside the trained span.
    TiledTwoStage,
}

/// Whether a checkpoint ships the spatial upsampler the composition needs.
///
/// Mirrors `Ltx2Pipeline::select_pipeline`: without the upsampler asset every
/// LTX-2 request falls back to a plain one-stage denoise, whatever pipeline was
/// asked for. Single-file catalog checkpoints (`cv:` / `hf:`) have no manifest
/// and therefore no upsampler, which is the conservative answer.
///
/// Component paths supplied through `config.toml` are deliberately not
/// consulted: the validator has no config, and guessing "yes" here would admit
/// a shape the engine then renders out of distribution.
fn model_has_spatial_upsampler(model: &str) -> bool {
    let canonical = crate::manifest::resolve_model_name(model);
    crate::manifest::find_manifest(&canonical).is_some_and(|manifest| {
        manifest
            .files
            .iter()
            .any(|file| file.component == crate::manifest::ModelComponent::SpatialUpscaler)
    })
}

/// Resolve the spatial composition a request will actually run.
///
/// `pipeline` is the request's explicit `ltx2.pipeline`, or `None` to let the
/// engine choose. Either way the answer requires a spatial upsampler on disk,
/// because that is what `select_pipeline` requires before it will pick a
/// refining pipeline at all.
///
/// Prefer [`ltx2_spatial_composition_for_request`] when the request is in
/// hand: with `pipeline: None` this assumes the engine's *default* choice, and
/// several request fields override that default before it is reached.
pub fn ltx2_spatial_composition(
    model: &str,
    pipeline: Option<Ltx2PipelineMode>,
) -> Ltx2SpatialComposition {
    if !model_has_spatial_upsampler(model) {
        return Ltx2SpatialComposition::SinglePass;
    }
    let refines = match pipeline {
        Some(mode) => mode.refines_spatially(),
        // `select_pipeline`'s default for a checkpoint that has the upsampler
        // is `Distilled` or `TwoStage`; both refine.
        None => true,
    };
    if refines {
        Ltx2SpatialComposition::TiledTwoStage
    } else {
        Ltx2SpatialComposition::SinglePass
    }
}

/// The pipeline `select_pipeline` will resolve for a request that names none.
///
/// Mirrors `Ltx2Pipeline::select_pipeline`'s implicit branch order
/// (`ltx2/pipeline.rs:377-388`). Only the *conditioning* selectors are
/// mirrored: the checkpoint-name fallback below them chooses between
/// `Distilled` and `TwoStage`, which both refine, so it cannot change this
/// answer. `retake_range` can and does — retake denoises once.
fn ltx2_implicit_pipeline(req: &GenerateRequest) -> Option<Ltx2PipelineMode> {
    if req.retake_range.is_some() {
        return Some(Ltx2PipelineMode::Retake);
    }
    if req.audio_file.is_some() || req.audio_file_path.is_some() {
        return Some(Ltx2PipelineMode::A2Vid);
    }
    if req.keyframes.as_ref().is_some_and(|items| items.len() > 1) {
        return Some(Ltx2PipelineMode::Keyframe);
    }
    if req.source_video.is_some() || req.source_video_path.is_some() {
        return Some(Ltx2PipelineMode::IcLora);
    }
    None
}

/// [`ltx2_spatial_composition`] resolved from the whole request.
///
/// An explicit `pipeline` wins; otherwise the request's own conditioning
/// decides, exactly as the engine's `select_pipeline` does. Without this a
/// retake — which denoises once — would be admitted at the composed ceiling
/// and only refused by the engine's backstop, minutes later.
pub fn ltx2_spatial_composition_for_request(req: &GenerateRequest) -> Ltx2SpatialComposition {
    ltx2_spatial_composition(
        &req.model,
        req.pipeline.or_else(|| ltx2_implicit_pipeline(req)),
    )
}

/// Total-pixel ceiling for a generation family, assuming no composition.
///
/// Callers that know the resolved model should use
/// [`max_pixels_for_family_composed`]; this is the conservative answer for the
/// ones that only have a family string.
pub fn max_pixels_for_family(family: Option<&str>) -> u64 {
    max_pixels_for_family_composed(family, Ltx2SpatialComposition::SinglePass)
}

/// Upstream Qwen-Image-Edit normalizes each edit image to a 1024x1024 pixel
/// area while preserving its aspect ratio before VAE conditioning. Keep this
/// separate from the output canvas ceiling: edit inputs and generated images
/// are independent memory domains.
pub const QWEN_IMAGE_EDIT_SOURCE_MAX_PIXELS: u64 = 1024 * 1024;

/// Composition-aware counterpart to [`max_pixels_for_family`].
pub fn max_pixels_for_family_composed(
    family: Option<&str>,
    composition: Ltx2SpatialComposition,
) -> u64 {
    match (family, composition) {
        (Some("ltx2"), Ltx2SpatialComposition::TiledTwoStage) => LTX2_COMPOSED_MAX_PIXELS,
        (Some("ltx2"), Ltx2SpatialComposition::SinglePass) => LTX2_MAX_PIXELS,
        (Some(family), _) if crate::minimax_h3::is_family(family) => crate::minimax_h3::MAX_PIXELS,
        _ => MAX_PIXELS,
    }
}

/// Per-axis ceiling for a generation family, where one exists.
pub fn max_axis_pixels_for_family(family: Option<&str>) -> Option<u32> {
    max_axis_pixels_for_family_composed(family, Ltx2SpatialComposition::SinglePass)
}

/// Composition-aware counterpart to [`max_axis_pixels_for_family`].
pub fn max_axis_pixels_for_family_composed(
    family: Option<&str>,
    composition: Ltx2SpatialComposition,
) -> Option<u32> {
    match (family, composition) {
        (Some("ltx2"), Ltx2SpatialComposition::TiledTwoStage) => {
            Some(LTX2_COMPOSED_MAX_AXIS_PIXELS)
        }
        (Some("ltx2"), Ltx2SpatialComposition::SinglePass) => Some(LTX2_MAX_AXIS_PIXELS),
        _ => None,
    }
}

/// Required pixel grid for a generation family.
///
/// LTX video VAEs compress spatial dimensions by 32. Every other current
/// family uses the shared 16px generation grid.
pub fn dimension_alignment_for_family(family: Option<&str>) -> u32 {
    if matches!(family, Some("ltx-video" | "ltx2"))
        || family.is_some_and(crate::minimax_h3::is_family)
    {
        32
    } else {
        16
    }
}

/// Model-aware counterpart to [`dimension_alignment_for_family`].
///
/// Most families have one grid, but Wan's is per checkpoint:
/// `wan22-ti2v-5b`'s 2.2 VAE compresses 16x spatially and its DiT patches the
/// latent 2x2, putting it on a 32 px grid while the 2.1-VAE checkpoints keep
/// the family's 16 (see [`wan_dimension_alignment`]). `family_hint` mirrors
/// [`validate_generate_request_with_family`]: pass the catalog-resolved family
/// for `cv:` / `hf:` ids; manifest models resolve without it.
pub fn dimension_alignment_for_model(model: &str, family_hint: Option<&str>) -> u32 {
    let family = resolved_family(model, family_hint);
    if family == Some("wan") {
        return wan_dimension_alignment(model);
    }
    dimension_alignment_for_family(family)
}

/// Validate explicit generation dimensions without rewriting them.
///
/// This is the shared admission boundary for one-shot and chain requests.
/// Clients may project a source image onto this contract, but the server must
/// reject invalid dimensions rather than silently changing the requested
/// canvas.
pub fn validate_generation_dimensions(
    width: u32,
    height: u32,
    family: Option<&str>,
) -> Result<(), String> {
    validate_generation_dimensions_composed(
        width,
        height,
        family,
        Ltx2SpatialComposition::SinglePass,
    )
}

/// Composition-aware counterpart to [`validate_generation_dimensions`].
///
/// Callers that have resolved the model — the HTTP generate and chain paths,
/// and the CLI — pass the real composition so a two-stage LTX-2 render can be
/// admitted past the trained span. Callers that only have a family string keep
/// the conservative single-pass ceiling.
pub fn validate_generation_dimensions_composed(
    width: u32,
    height: u32,
    family: Option<&str>,
    composition: Ltx2SpatialComposition,
) -> Result<(), String> {
    validate_generation_dimensions_with_alignment(
        width,
        height,
        family,
        composition,
        dimension_alignment_for_family(family),
    )
}

/// Model-aware sibling of [`validate_generation_dimensions_composed`].
///
/// Same contract, but the pixel grid comes from
/// [`dimension_alignment_for_model`], so a per-checkpoint grid — currently
/// `wan22-ti2v-5b`'s 32 — is enforced at admission instead of after the model
/// has loaded. Callers that cannot name a model keep the family-only
/// validator, whose answer is deliberately unchanged.
pub fn validate_generation_dimensions_for_model(
    model: &str,
    width: u32,
    height: u32,
    family: Option<&str>,
    composition: Ltx2SpatialComposition,
) -> Result<(), String> {
    validate_generation_dimensions_with_alignment(
        width,
        height,
        family,
        composition,
        dimension_alignment_for_model(model, family),
    )
}

fn validate_generation_dimensions_with_alignment(
    width: u32,
    height: u32,
    family: Option<&str>,
    composition: Ltx2SpatialComposition,
    alignment: u32,
) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("width and height must be > 0".to_string());
    }

    if !width.is_multiple_of(alignment) || !height.is_multiple_of(alignment) {
        let family_label = family
            .filter(|value| !value.is_empty())
            .map(|value| format!(" for {value} models"))
            .unwrap_or_default();
        return Err(format!(
            "width ({width}) and height ({height}) must be multiples of {alignment}{family_label}"
        ));
    }

    if let Some(axis_limit) = max_axis_pixels_for_family_composed(family, composition) {
        let longest = width.max(height);
        if longest > axis_limit {
            // Two different failures wear the same shape here, and telling
            // them apart is the whole difference between an actionable error
            // and a dead end. Past the composed ceiling nothing helps but a
            // smaller output; past the trained span with a single-pass model,
            // a checkpoint that ships the spatial upsampler does.
            let mut remedy = String::new();
            if composition == Ltx2SpatialComposition::SinglePass
                && longest <= LTX2_COMPOSED_MAX_AXIS_PIXELS
            {
                remedy.push_str(
                    " This checkpoint renders in one pass; reaching that size needs a checkpoint \
                     that ships the spatial upsampler, which renders stage 1 at half size and \
                     refines it over tiles.",
                );
            }
            if let Some(rung) = largest_ltx2_rung_within(axis_limit) {
                remedy.push_str(&format!(
                    " The largest output this render reaches is {} ({}x{}).",
                    rung.label, rung.width, rung.height
                ));
            }
            return Err(format!(
                "{width}x{height} has a {longest}px axis, beyond the {axis_limit}px span this \
                 render can hold — positions past it are out of distribution. Render at or below \
                 {axis_limit}px on the long edge.{remedy}"
            ));
        }
    }

    let limit = max_pixels_for_family_composed(family, composition);
    let pixels = width as u64 * height as u64;
    if pixels > limit {
        return Err(format!(
            "{width}x{height} = {:.2} megapixels exceeds the {} limit (VAE VRAM constraint)",
            pixels as f64 / 1_000_000.0,
            megapixel_limit_label_for(limit)
        ));
    }

    Ok(())
}

/// One rung of the LTX-2 output ladder.
///
/// A rung is an output shape plus the composition that reaches it. Every entry
/// is 64-aligned so stage 1 — the target halved — still lands on the VAE's
/// 32 px latent grid, which is upstream's own `divisor = 64 if is_two_stage`
/// rule (`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:326`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ltx2OutputRung {
    /// Stable identifier, safe to persist and to match on.
    pub id: &'static str,
    /// Human-readable name for pickers and errors.
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
}

impl Ltx2OutputRung {
    /// Shape stage 1 renders at under one x2 spatial rung.
    ///
    /// Mirrors `derive_stage1_render_shape` / `latent_grid_downsample`: the
    /// target's latent grid is halved with ceiling division, then expanded
    /// back to pixels. `advertised_rungs_match_the_engines_own_arithmetic` in
    /// `mold-inference` pins this to the engine's own arithmetic — this crate
    /// cannot see it, and a rung that names a stage-1 shape the engine does not
    /// render is worse than naming none.
    pub const fn stage1_shape(&self) -> (u32, u32) {
        (
            ltx2_stage1_axis_for(self.width, Some(Ltx2SpatialUpscale::X2)),
            ltx2_stage1_axis_for(self.height, Some(Ltx2SpatialUpscale::X2)),
        )
    }

    /// Whether this rung needs the tiled stage-2 refinement, i.e. whether it
    /// has an axis past the span a single denoise can hold.
    pub const fn requires_tiled_stage2(&self) -> bool {
        self.width > LTX2_MAX_AXIS_PIXELS || self.height > LTX2_MAX_AXIS_PIXELS
    }

    /// Spatial tiles stage 2 splits into, as `(columns, rows)`.
    ///
    /// Mirrors `plan_stage2_tiling`: an axis inside the trained span stays
    /// whole, and an oversized one is split into the fewest tiles whose every
    /// tile fits. Pinned to the engine by
    /// `advertised_rungs_match_the_engines_own_arithmetic`.
    pub const fn stage2_tiles(&self) -> (u32, u32) {
        (
            ltx2_axis_tile_count(self.width),
            ltx2_axis_tile_count(self.height),
        )
    }
}

/// Stage-1 extent for one axis under a spatial rung.
///
/// Mirrors `latent_grid_downsample` in `ltx2/model/upsampler.rs`, including
/// its x1.5 case: the rational upsampler emits `floor((3 * latent + 1) / 2)`
/// cells, so stage 1 needs `ceil((2 * target_latent - 1) / 3)` to cover the
/// requested lattice. An absent rung means stage 1 renders the target itself.
pub const fn ltx2_stage1_axis_for(target: u32, upscale: Option<Ltx2SpatialUpscale>) -> u32 {
    let grid = LTX2_SPATIAL_LATENT_STRIDE;
    let Some(upscale) = upscale else {
        return if target < grid { grid } else { target };
    };
    let target_latent = if target < grid {
        1
    } else {
        target.div_ceil(grid)
    };
    let stage1_latent = match upscale {
        Ltx2SpatialUpscale::X2 => target_latent.div_ceil(2),
        Ltx2SpatialUpscale::X1_5 => target_latent
            .saturating_mul(2)
            .saturating_sub(1)
            .div_ceil(3),
    };
    if stage1_latent == 0 {
        grid
    } else {
        stage1_latent * grid
    }
}

/// Largest output axis whose stage 1 still lands inside the trained span
/// under `upscale`.
///
/// x2 halves, so it reaches `2 * span`. x1.5 only divides by 1.5, so it stops
/// at 3072px — asking for 4K with `--spatial-upscale x1.5` puts stage 1 at
/// 2560px, exactly the out-of-distribution render the ceiling exists to
/// prevent.
pub fn ltx2_composed_axis_ceiling(upscale: Option<Ltx2SpatialUpscale>) -> u32 {
    match upscale {
        None | Some(Ltx2SpatialUpscale::X2) => LTX2_COMPOSED_MAX_AXIS_PIXELS,
        Some(Ltx2SpatialUpscale::X1_5) => {
            // Walk the 32px grid rather than inverting the rational
            // downsample in closed form; the loop is bounded by the composed
            // ceiling and runs once per admission.
            let mut ceiling = LTX2_MAX_AXIS_PIXELS;
            while ceiling < LTX2_COMPOSED_MAX_AXIS_PIXELS
                && ltx2_stage1_axis_for(ceiling + LTX2_SPATIAL_LATENT_STRIDE, upscale)
                    <= LTX2_MAX_AXIS_PIXELS
            {
                ceiling += LTX2_SPATIAL_LATENT_STRIDE;
            }
            ceiling
        }
    }
}

/// Refuse a composed render whose *stage 1* leaves the trained span.
///
/// [`LTX2_COMPOSED_MAX_AXIS_PIXELS`] is shorthand for this check under the
/// default x2 rung. A request that names x1.5 instead needs the real one:
/// a 3840px output renders stage 1 at 2560px, and nothing downstream repairs
/// that — stage 2 tiles the *refinement*, never stage 1.
pub fn validate_ltx2_stage1_span(
    width: u32,
    height: u32,
    upscale: Option<Ltx2SpatialUpscale>,
) -> Result<(), String> {
    // An absent rung on a refining pipeline is not "no rung": the runtime
    // applies an implicit x2 so stage 1 renders halved anyway
    // (`ltx2/runtime.rs`'s `implicit_x2_shape`). Reading `None` literally here
    // would refuse every composed render at once.
    let effective = upscale.unwrap_or(Ltx2SpatialUpscale::X2);
    let stage1 = (
        ltx2_stage1_axis_for(width, Some(effective)),
        ltx2_stage1_axis_for(height, Some(effective)),
    );
    let longest = stage1.0.max(stage1.1);
    if longest <= LTX2_MAX_AXIS_PIXELS {
        return Ok(());
    }
    let rung = match effective {
        Ltx2SpatialUpscale::X1_5 => "x1.5",
        Ltx2SpatialUpscale::X2 => "x2",
    };
    let ceiling = ltx2_composed_axis_ceiling(upscale);
    Err(format!(
        "{width}x{height} with {rung} spatial upscale renders stage 1 at {}x{}, whose {longest}px \
         axis is past the {}px span these checkpoints were trained on. The rung sets the ceiling: \
         it reaches {ceiling}px on the long edge. Use a x2 upscale, or render at or below \
         {ceiling}px.",
        stage1.0, stage1.1, LTX2_MAX_AXIS_PIXELS,
    ))
}

/// Number of stage-2 tiles one axis is split into.
const fn ltx2_axis_tile_count(target: u32) -> u32 {
    if target <= LTX2_MAX_AXIS_PIXELS {
        return 1;
    }
    let count = target.div_ceil(LTX2_MAX_AXIS_PIXELS);
    if count < 2 {
        2
    } else {
        count
    }
}

/// The LTX video VAE's spatial compression factor.
pub const LTX2_SPATIAL_LATENT_STRIDE: u32 = 32;

/// The LTX-2 output ladder, smallest rung first.
///
/// Every rung above 1080p is reached by composition, not by a bigger denoise:
/// stage 1 renders the halved shape, one x2 spatial rung upsamples it, and
/// stage 2 refines the result over tiles.
///
/// **The ladder stops at 4K UHD because of the encoder, not the model.**
/// `LTX2_COMPOSED_MAX_AXIS_PIXELS` (4096) is where a single halving stops
/// landing stage 1 inside the trained span, and generation is admitted that
/// far — but the bundled OpenH264 encoder refuses anything past 3840x2160
/// ("Encoder max resolution 3840x2160 horizontal or 2160x3840 vertical"), and
/// MP4 is this family's default container. A rung wider than 3840, or the
/// 3840x2176 that rounding 2160 *up* onto the /64 grid would give, generates
/// fine and then fails at save time — after the whole render. So the rung is
/// 3840x2112, rounding 2160 *down*, which is upstream's own CENTER_CROP
/// alignment and the largest UHD-class shape mold can actually deliver.
///
/// VRAM: see `website/models/ltx2.md`. The numbers live in prose because the
/// only published figures are upstream's, they are for a different pipeline
/// (HDR IC-LoRA, 161 frames, 22B), and pinning them here would read as mold's
/// own measured requirement.
pub const LTX2_OUTPUT_RUNGS: &[Ltx2OutputRung] = &[
    Ltx2OutputRung {
        id: "720p",
        label: "720p HD",
        width: 1_280,
        height: 704,
    },
    Ltx2OutputRung {
        id: "1080p",
        label: "1080p Full HD",
        width: 1_920,
        height: 1_088,
    },
    Ltx2OutputRung {
        id: "1440p",
        label: "1440p QHD",
        width: 2_560,
        height: 1_408,
    },
    Ltx2OutputRung {
        id: "4k-uhd",
        label: "4K UHD",
        width: 3_840,
        height: 2_112,
    },
];

/// The rung an output shape lands on, in either orientation.
///
/// Portrait is the same rung as its landscape transpose: the composition and
/// the cost are identical, and a picker that called 2176x3840 an unnamed shape
/// would be lying about both.
pub fn ltx2_output_rung(width: u32, height: u32) -> Option<&'static Ltx2OutputRung> {
    LTX2_OUTPUT_RUNGS.iter().find(|rung| {
        (rung.width == width && rung.height == height)
            || (rung.width == height && rung.height == width)
    })
}

/// The largest rung whose long edge fits `axis_limit`.
///
/// This is what makes an over-size rejection actionable: naming the ceiling in
/// pixels tells the user what they cannot have, naming the rung tells them
/// what they can.
pub fn largest_ltx2_rung_within(axis_limit: u32) -> Option<&'static Ltx2OutputRung> {
    LTX2_OUTPUT_RUNGS
        .iter()
        .rfind(|rung| rung.width.max(rung.height) <= axis_limit)
}

fn mib_label(bytes: usize) -> String {
    format!("{:.0} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// Clamp dimensions to fit within the megapixel limit, preserving aspect ratio.
/// Both dimensions are rounded down to multiples of 16.
/// Returns the original dimensions unchanged if already within limits.
pub fn clamp_to_megapixel_limit(w: u32, h: u32) -> (u32, u32) {
    clamp_to_family_pixel_limit(w, h, None)
}

/// Family-aware counterpart to [`clamp_to_megapixel_limit`].
///
/// Both the ceiling and the rounding grid come from the family. Clamping an
/// LTX-2 source projection with the shared 1.8 MP limit and a /16 grid would
/// shrink a canvas the validator would have accepted, and could land off the
/// /32 grid it requires — a silent downgrade followed by a rejection.
pub fn clamp_to_family_pixel_limit(w: u32, h: u32, family: Option<&str>) -> (u32, u32) {
    clamp_dims_to(
        w,
        h,
        max_pixels_for_family(family),
        dimension_alignment_for_family(family),
        max_axis_pixels_for_family(family),
    )
}

fn clamp_dims_to(w: u32, h: u32, limit: u64, align: u32, axis_limit: Option<u32>) -> (u32, u32) {
    let pixels = w as u64 * h as u64;
    let within_axis = axis_limit.is_none_or(|axis| w.max(h) <= axis);
    if pixels <= limit && within_axis {
        return (w, h);
    }

    let mut scale = if pixels > limit {
        (limit as f64 / pixels as f64).sqrt()
    } else {
        1.0
    };
    if let Some(axis) = axis_limit {
        let longest = w.max(h) as f64;
        if longest * scale > axis as f64 {
            scale = axis as f64 / longest;
        }
    }

    let new_w = ((w as f64 * scale) as u32 / align) * align;
    let new_h = ((h as f64 * scale) as u32 / align) * align;
    // Ensure we don't produce zero dimensions
    (new_w.max(align), new_h.max(align))
}

/// Fit source image dimensions into a model's native resolution bounding box,
/// preserving aspect ratio.
///
/// The model's default width/height define the bounding box. The source image's
/// aspect ratio is preserved:
/// - If the source is wider than the model bounds, width is set to `model_w` and
///   height is scaled proportionally.
/// - If the source is taller, height is set to `model_h` and width is scaled.
/// - If the source fits entirely within model bounds (same aspect ratio as the
///   model), the model's native dimensions are used as the output. For sources
///   with a different aspect ratio, the output fills the limiting axis at model
///   scale while keeping the other axis within bounds.
///
/// Output is rounded to 16px alignment and clamped to the megapixel limit.
///
/// This is the family-only compatibility path: 16 is the shared generation
/// grid, but not every checkpoint's. Callers that know the model (or its
/// advertised `dimension_alignment`) should use
/// [`fit_to_model_dimensions_aligned`] with
/// [`dimension_alignment_for_model`]'s answer so a 32-grid checkpoint like
/// `wan22-ti2v-5b` receives a canvas its VAE can encode.
pub fn fit_to_model_dimensions(src_w: u32, src_h: u32, model_w: u32, model_h: u32) -> (u32, u32) {
    fit_to_model_dimensions_aligned(src_w, src_h, model_w, model_h, 16)
}

/// Alignment-aware counterpart to [`fit_to_model_dimensions`]: identical
/// aspect-preserving fit, but both axes are floored to the caller-supplied
/// grid — the resolved model's alignment, not the family-wide 16.
pub fn fit_to_model_dimensions_aligned(
    src_w: u32,
    src_h: u32,
    model_w: u32,
    model_h: u32,
    align: u32,
) -> (u32, u32) {
    let align = align.max(1);
    let src_ratio = src_w as f64 / src_h as f64;
    let model_ratio = model_w as f64 / model_h as f64;

    let (w, h) = if src_ratio > model_ratio {
        // Source is wider: width-limited
        (model_w as f64, model_w as f64 / src_ratio)
    } else {
        // Source is taller or same: height-limited
        (model_h as f64 * src_ratio, model_h as f64)
    };

    let w = ((w as u32) / align * align).max(align);
    let h = ((h as u32) / align * align).max(align);
    clamp_dims_to(w, h, MAX_PIXELS, align, None)
}

/// Resize dimensions toward a target pixel area while preserving aspect ratio.
///
/// The result is rounded to the requested alignment and clamped to the shared
/// megapixel safety limit.
pub fn fit_to_target_area(src_w: u32, src_h: u32, target_area: u32, align: u32) -> (u32, u32) {
    let src_w = src_w.max(1);
    let src_h = src_h.max(1);
    let align = align.max(1);
    let scale = (f64::from(target_area) / (f64::from(src_w) * f64::from(src_h))).sqrt();
    let width = ((f64::from(src_w) * scale) / f64::from(align)).round() as u32 * align;
    let height = ((f64::from(src_h) * scale) / f64::from(align)).round() as u32 * align;
    clamp_to_megapixel_limit(width.max(align), height.max(align))
}

/// Check whether `data` starts with a recognized image format magic bytes (PNG or JPEG).
fn is_valid_image_format(data: &[u8]) -> bool {
    let is_png = data.len() >= 4 && data[..4] == [0x89, 0x50, 0x4E, 0x47];
    let is_jpeg = data.len() >= 2 && data[..2] == [0xFF, 0xD8];
    is_png || is_jpeg
}

fn model_family(model_name: &str) -> Option<&str> {
    crate::manifest::find_manifest(model_name)
        .map(|m| m.family.as_str())
        .or_else(|| {
            if model_name.starts_with("qwen-image-edit") {
                Some("qwen-image-edit")
            } else if model_name.starts_with("qwen-image") {
                Some("qwen-image")
            } else {
                None
            }
        })
}

/// Resolve a model's family for validation, preferring an explicit hint when
/// provided. The hint lets callers (e.g. the HTTP server) pass through a family
/// that the manifest layer can't see — most notably catalog IDs like
/// `cv:2781713` whose family is recorded in the catalog DB rather than the
/// hardcoded manifest. When `family_hint` is `None` (or an empty string), the
/// manifest fallback runs as before.
fn resolved_family<'a>(model_name: &'a str, family_hint: Option<&'a str>) -> Option<&'a str> {
    family_hint
        .filter(|h| !h.is_empty())
        .or_else(|| model_family(model_name))
}

/// Whether `req` must carry a non-empty prompt.
///
/// Video families whose text encoder pads to a fixed-width context (LTX-2's
/// Gemma connector replaces every padded position with learned register
/// embeddings, so `""` is a trained context rather than a degenerate one)
/// accept an empty prompt as long as the request carries visual conditioning
/// to continue: a source image, keyframes, a source video, or an extend. Pure
/// text-to-video and every image family keep the prompt required.
///
/// Note this buys no VRAM — the Gemma context is a fixed-size tensor whose
/// footprint is independent of the token count — and an unprompted clip tends
/// toward near-static micro-motion. Callers should surface that as guidance
/// rather than synthesising a placeholder prompt.
///
/// `family_hint` mirrors [`validate_generate_request_with_family`]: pass the
/// catalog-resolved family for `cv:` / `hf:` model IDs, whose family the
/// manifest cannot see.
pub fn prompt_required_for(req: &GenerateRequest, family_hint: Option<&str>) -> bool {
    prompt_required_with_conditioning(
        resolved_family(&req.model, family_hint),
        has_visual_conditioning(req),
    )
}

/// Whether a request carries visual conditioning — a source image, keyframes,
/// a source video (inline or server-local path), or an extend.
///
/// This is the single definition of "conditioned" for the whole request path.
/// Beyond the optional-prompt rule it also separates OOM cooldown buckets, and
/// those must agree: two requests with different conditioning have different
/// VRAM profiles and must never share a cooldown or a reduced memory grant.
pub fn has_visual_conditioning(req: &GenerateRequest) -> bool {
    req.source_image.is_some()
        || req.keyframes.as_ref().is_some_and(|k| !k.is_empty())
        || req.source_video.is_some()
        || req.source_video_path.is_some()
        || req.is_extend()
}

/// Lower-level form of [`prompt_required_for`] for callers that have not yet
/// assembled a [`GenerateRequest`] — the CLI, TUI and Discord front-ends build
/// the request only after the prompt is resolved. `has_visual_conditioning` is
/// true when the request will carry a source image, keyframes, a source video,
/// or an extend.
pub fn prompt_required_with_conditioning(
    family: Option<&str>,
    has_visual_conditioning: bool,
) -> bool {
    !(matches!(family, Some("ltx2" | "ltx-video")) && has_visual_conditioning)
}

fn validate_lora_weight(lora: &LoraWeight, field_name: &str) -> Result<(), String> {
    if lora.scale < 0.0 || lora.scale > 2.0 {
        return Err(format!(
            "{field_name} scale ({}) must be in range [0.0, 2.0]",
            lora.scale
        ));
    }
    if !lora.path.ends_with(".safetensors") && !lora.path.starts_with("camera-control:") {
        return Err(format!(
            "{field_name} file must be a .safetensors file or camera-control preset"
        ));
    }
    Ok(())
}

/// Refuse an explicit `expert` on a model that has no experts to route to.
///
/// Only the Wan 2.2 A14B pair has a high/low split. Silently ignoring the
/// field on a single-expert checkpoint would let a user believe an adapter was
/// bound to half the schedule when it was applied to all of it.
fn require_expert_routable_model(
    lora: &LoraWeight,
    model: &str,
    family: Option<&str>,
) -> Result<(), String> {
    let Some(expert) = lora.expert else {
        return Ok(());
    };
    let expert = match expert {
        crate::LoraExpert::High => "high",
        crate::LoraExpert::Low => "low",
    };
    if family != Some("wan") {
        return Err(format!(
            "lora expert ('{expert}') applies to the Wan 2.2 A14B expert pair; \
             {} is not a Wan model",
            model
        ));
    }
    // An opaque catalog id cannot be classified by name, and the engine reads
    // the real pair from the checkpoint, so those are left to it rather than
    // guessed at here.
    let opaque = model.starts_with("cv:") || model.starts_with("hf:");
    if !opaque && !model.to_ascii_lowercase().contains("a14b") {
        return Err(format!(
            "lora expert ('{expert}') needs the Wan 2.2 A14B two-expert pair; \
             {model} is a single-expert checkpoint — drop the expert field to \
             apply the adapter to it"
        ));
    }
    Ok(())
}

fn validate_keyframes(
    keyframes: &[KeyframeCondition],
    frames: Option<u32>,
    family: Option<&str>,
) -> Result<(), String> {
    match family {
        Some("ltx2") => {}
        // Wan accepts exactly the first/last endpoint pair (#779) — the
        // family has no mid-clip keyframe path, so every other layout is
        // named at admission rather than after the model loads.
        Some("wan") => {
            if keyframes.len() != 2 {
                return Err(format!(
                    "Wan supports exactly two keyframes — the first and last pixel frames — \
                     got {}",
                    keyframes.len()
                ));
            }
            // Without an explicit clip length the closing anchor cannot be
            // checked here, and the engine would resolve its own default and
            // reject a mismatched endpoint only after the model loads —
            // defeating admission-time validation.
            let Some(frames) = frames else {
                return Err(
                    "Wan first/last-frame keyframes require an explicit frames count — the \
                     closing keyframe must anchor the clip's final frame"
                        .to_string(),
                );
            };
            // A single-frame clip has coincident endpoints; the generic
            // duplicate-frame check below would also refuse it, but with a
            // message that doesn't say why. Name the real problem instead.
            if frames < 2 {
                return Err(
                    "Wan first/last-frame keyframes need a multi-frame clip — frames=1 \
                     renders a single still, which has no distinct last frame"
                        .to_string(),
                );
            }
            let last = frames.saturating_sub(1);
            if keyframes[0].frame != 0 || keyframes[1].frame != last {
                return Err(format!(
                    "Wan first/last-frame keyframes must anchor frames 0 and {last} (the \
                     clip's endpoints), got frames {} and {}",
                    keyframes[0].frame, keyframes[1].frame
                ));
            }
        }
        None => {
            return Err(
                "unknown model family; keyframes are only supported for LTX-2 / LTX-2.3 and \
                 Wan models"
                    .to_string(),
            );
        }
        _ => {
            return Err(
                "keyframes are only supported for LTX-2 / LTX-2.3 and Wan models".to_string(),
            );
        }
    }
    if keyframes.is_empty() {
        return Err("keyframes must not be empty".to_string());
    }

    let mut seen = std::collections::BTreeSet::new();
    for keyframe in keyframes {
        if !is_valid_image_format(&keyframe.image) {
            return Err("keyframes must contain only PNG or JPEG images".to_string());
        }
        if let Some(total_frames) = frames {
            if keyframe.frame >= total_frames {
                return Err(format!(
                    "keyframe frame ({}) must be less than frames ({total_frames})",
                    keyframe.frame
                ));
            }
        }
        if !seen.insert(keyframe.frame) {
            return Err(format!("duplicate keyframe frame: {}", keyframe.frame));
        }
    }

    Ok(())
}

/// Bounds-check the LTX-2 multimodal guider overrides.
///
/// These are advanced quality/motion knobs, so the ranges are deliberately
/// generous — the job here is to reject values that cannot mean anything
/// (NaN, negatives, block indices no checkpoint has) before a request reaches
/// the queue, not to police taste. The engine re-checks `stg_blocks` against
/// the resolved checkpoint's transformer depth, which validation cannot know.
fn validate_guidance_overrides(overrides: &Ltx2GuidanceOverrides) -> Result<(), String> {
    if overrides.is_empty() {
        return Err(
            "guidance_overrides must set at least one field; omit it to keep pipeline defaults"
                .to_string(),
        );
    }
    let bounded = |value: Option<f64>, name: &str, max: f64| -> Result<(), String> {
        match value {
            Some(value) if !value.is_finite() => Err(format!("{name} must be a finite number")),
            Some(value) if !(0.0..=max).contains(&value) => {
                Err(format!("{name} ({value}) must be between 0.0 and {max}"))
            }
            _ => Ok(()),
        }
    };
    bounded(
        overrides.stg_scale,
        "guidance_overrides.stg_scale",
        Ltx2GuidanceOverrides::MAX_SCALE,
    )?;
    bounded(
        overrides.modality_scale,
        "guidance_overrides.modality_scale",
        Ltx2GuidanceOverrides::MAX_SCALE,
    )?;
    // Rescale is an interpolation factor between the guided prediction and
    // its std-matched form, so anything outside 0..=1 is meaningless.
    bounded(
        overrides.rescale_scale,
        "guidance_overrides.rescale_scale",
        1.0,
    )?;
    if let Some(skip_step) = overrides.skip_step {
        if skip_step > Ltx2GuidanceOverrides::MAX_SKIP_STEP {
            return Err(format!(
                "guidance_overrides.skip_step ({skip_step}) must be <= {}",
                Ltx2GuidanceOverrides::MAX_SKIP_STEP
            ));
        }
    }
    if let Some(blocks) = &overrides.stg_blocks {
        if blocks.is_empty() {
            return Err(
                "guidance_overrides.stg_blocks must not be empty; omit it to keep the pipeline default block"
                    .to_string(),
            );
        }
        if blocks.len() > MAX_STG_BLOCKS {
            return Err(format!(
                "guidance_overrides.stg_blocks lists {} blocks; at most {MAX_STG_BLOCKS} are supported",
                blocks.len()
            ));
        }
        for (index, block) in blocks.iter().enumerate() {
            if *block >= MAX_STG_BLOCK_INDEX {
                return Err(format!(
                    "guidance_overrides.stg_blocks[{index}] ({block}) exceeds the deepest supported transformer block ({})",
                    MAX_STG_BLOCK_INDEX - 1
                ));
            }
            if blocks[..index].contains(block) {
                return Err(format!(
                    "guidance_overrides.stg_blocks[{index}] ({block}) is listed more than once"
                ));
            }
        }
    }
    Ok(())
}

/// Admission rules for `extend_video` / `extend_video_path`.
///
/// Extend reuses the chain motion-tail machinery, so it inherits the same two
/// hard constraints: the overlap has to land on the family's own temporal grid
/// to re-encode cleanly — `8k+1` for the LTX-2 VAE's causal grid, `4k+1` for
/// wan, resolved through [`frame_step_for_family`] rather than assumed — and it
/// has to be strictly shorter than the rendered clip or the continuation
/// contributes no new frames at all.
fn validate_extend(req: &GenerateRequest, family: Option<&str>) -> Result<(), String> {
    if let Some(video) = &req.extend_video {
        require_extend_capable_family(family, "extend_video")?;
        if req.extend_video_path.is_some() {
            return Err("extend_video_path cannot be combined with extend_video".to_string());
        }
        if video.is_empty() {
            return Err("extend_video must not be empty".to_string());
        }
        validate_inline_media_size(video, "extend_video", MAX_INLINE_EXTEND_VIDEO_BYTES)?;
    }
    if let Some(path) = &req.extend_video_path {
        require_extend_capable_family(family, "extend_video_path")?;
        if path.trim().is_empty() {
            return Err("extend_video_path must not be empty".to_string());
        }
    }

    if !req.is_extend() {
        if req.extend_overlap_frames.is_some() {
            return Err(
                "extend_overlap_frames requires extend_video or extend_video_path".to_string(),
            );
        }
        return Ok(());
    }

    // Extend continues one clip's motion; a reference video conditions a fresh
    // render. Accepting both would leave two competing sources of truth for
    // what the first frames should look like.
    if req.source_video.is_some() || req.source_video_path.is_some() {
        return Err(
            "extend_video cannot be combined with source_video; extend continues an existing \
             clip, while source_video is reference conditioning for a fresh render"
                .to_string(),
        );
    }
    if req.source_image.is_some() {
        return Err(
            "extend_video cannot be combined with source_image; the continuation's first frames \
             are pinned by the source video's tail"
                .to_string(),
        );
    }
    if req.keyframes.is_some() {
        return Err("extend_video cannot be combined with keyframes".to_string());
    }

    let overlap = req.effective_extend_overlap_frames_for_family(family);
    if overlap == 0 {
        return Err(
            "extend_overlap_frames must be >= 1 so the continuation has motion context".to_string(),
        );
    }
    // The carryover frames re-encode through the family's own video VAE, so the
    // overlap has to sit on that VAE's temporal grid — 8x causal for LTX-2,
    // 4x for wan. A hardcoded 8 rejected every valid wan overlap.
    let step = family.and_then(frame_step_for_family).unwrap_or(8);
    if overlap % step != 1 {
        let examples: Vec<String> = (0..4).map(|k| (k * step + 1).to_string()).collect();
        return Err(format!(
            "extend_overlap_frames ({overlap}) must be {step}k+1 ({}, …) so the carryover \
             frames re-encode cleanly through this family's video VAE temporal grid",
            examples.join(", "),
        ));
    }
    if let Some(frames) = req.frames {
        if overlap >= frames {
            return Err(format!(
                "extend_overlap_frames ({overlap}) must be strictly less than frames ({frames}) \
                 so the continuation adds at least one new frame"
            ));
        }
    }
    Ok(())
}

/// Families whose engines can continue an existing clip.
///
/// LTX-2 re-encodes the tail as a latent motion carryover. Wan has no latent
/// motion tail, so it continues the way its chain seam does — the source
/// clip's final frame becomes the continuation's image conditioning — which
/// only an image-conditioned checkpoint can accept. The per-model
/// `supports_extend` field is what narrows the family to those checkpoints;
/// this gate only rejects families with no continuation path at all.
///
/// Public because the CLI preflight has to ask it *before* the source-image
/// contract gate does: an extend now counts as carrying source frames, so a
/// continuation aimed at a text-to-video-only family would otherwise be
/// refused for "does not accept a source image or keyframes" — wording for a
/// request that supplied neither (#783).
pub fn require_extend_capable_family(
    family: Option<&str>,
    feature_name: &str,
) -> Result<(), String> {
    match family {
        Some("ltx2") | Some("wan") => Ok(()),
        None => Err(format!(
            "unknown model family; {feature_name} is only supported for LTX-2 / LTX-2.3 and Wan models"
        )),
        _ => Err(format!(
            "{feature_name} is only supported for LTX-2 / LTX-2.3 and Wan models"
        )),
    }
}

fn require_ltx2_family(family: Option<&str>, feature_name: &str) -> Result<(), String> {
    match family {
        Some("ltx2") => Ok(()),
        None => Err(format!(
            "unknown model family; {feature_name} is only supported for LTX-2 / LTX-2.3 models"
        )),
        _ => Err(format!(
            "{feature_name} is only supported for LTX-2 / LTX-2.3 models"
        )),
    }
}

/// LoRA support is available for FLUX, Flux.2, LTX-2, SD1.5, SD3, SDXL,
/// Qwen-Image (and qwen-image-edit), Wan, and Z-Image — `mold-inference`'s
/// per-family `lora.rs` modules are the engine paths that know how to merge
/// low-rank adapters into the base weights. Surfacing the gate at validation
/// produces a clear 400 instead of an opaque inference-layer panic when a
/// user picks an unsupported model family + a LoRA.
fn require_lora_capable_family(family: Option<&str>) -> Result<(), String> {
    match family {
        Some(family) if family_supports_lora(family) => Ok(()),
        Some(other) => Err(format!(
            "LoRA is currently supported for FLUX, Flux.2, LTX-2, SD1.5, SD3, SDXL, Qwen-Image, Wan, and Z-Image models; got family {other:?}"
        )),
        None => Err(
            "LoRA requires a known model family — pick a FLUX, Flux.2, LTX-2, SD1.5, SD3, SDXL, Qwen-Image, Wan, or Z-Image model first"
                .to_string(),
        ),
    }
}

fn require_controlnet_capable_family(family: Option<&str>) -> Result<(), String> {
    match family {
        Some("sd15" | "sd1.5" | "stable-diffusion-1.5") => Ok(()),
        Some(other) => Err(format!(
            "ControlNet generation is currently supported for SD1.5 models; got family {other:?}"
        )),
        None => Err(
            "ControlNet generation requires a known model family — pick an SD1.5 model first"
                .to_string(),
        ),
    }
}

fn validate_inline_media_size(
    bytes: &[u8],
    field_name: &str,
    max_bytes: usize,
) -> Result<(), String> {
    if bytes.len() > max_bytes {
        return Err(format!(
            "{field_name} exceeds the {} inline request limit (got {:.1} MiB)",
            mib_label(max_bytes),
            bytes.len() as f64 / (1024.0 * 1024.0)
        ));
    }
    Ok(())
}

/// Validate a generate request. Returns `Ok(())` if valid, or an error message.
/// Shared between the HTTP server and local CLI inference paths.
///
/// For models whose family can't be derived from the manifest (catalog IDs
/// like `cv:2781713`), use [`validate_generate_request_with_family`] and pass
/// the resolved family from the catalog DB; otherwise the family-gated
/// features (audio, keyframes, retake, …) will fail with
/// `unknown model family` even on legitimate LTX-2 catalog checkpoints.
pub fn validate_generate_request(req: &GenerateRequest) -> Result<(), String> {
    validate_generate_request_with_family(req, None)
}

/// Enforce model access for every model/artifact identity carried directly by
/// a generation request.
///
/// This is deliberately separate from shape/feature validation: callers that
/// own an artifact root must run it before downloads, queue registration, or
/// other admission mutations. The base model's configured/default artifacts
/// remain the caller's responsibility because they do not travel in the
/// request itself.
pub fn require_generate_request_model_activation(
    req: &GenerateRequest,
    artifact_root: Option<&std::path::Path>,
    family_hint: Option<&str>,
) -> Result<(), crate::ModelActivationError> {
    crate::require_model_activation(&req.model, family_hint)?;
    for identity in [req.control_model.as_deref(), req.upscale_model.as_deref()]
        .into_iter()
        .flatten()
    {
        crate::require_model_activation(identity, None)?;
    }
    for lora in req.lora.iter().chain(req.loras.iter().flatten()) {
        crate::require_model_artifact_activation(
            std::path::Path::new(&lora.path),
            artifact_root,
            None,
        )?;
    }
    Ok(())
}

/// Whether a request carries the source frames the per-checkpoint
/// source-image contract (#772) is asked about — the `has_source` argument of
/// [`source_image_contract_violation`].
///
/// Three inputs carry them. A `source_image` is the obvious one; first/last
/// frame `keyframes` carry them too (#779); and so does an extend, whose first
/// frames come from the tail of the clip it continues (#783). Extend is the
/// non-obvious member: [`validate_generate_request`] forbids pairing
/// `extend_video` with `source_image` or keyframes, so an extend request
/// provably has neither of the other two — and a gate that counted only those
/// saw every continuation as source-less. That refused every Wan I2V extend
/// with "this Wan I2V checkpoint needs a source image", the exact contract
/// that makes the checkpoint extend-capable, while letting a text-to-video
/// extend through to die in the engine after the load was paid for.
pub fn request_carries_source_frames(req: &GenerateRequest) -> bool {
    req.source_image.is_some()
        || req.keyframes.as_ref().is_some_and(|k| !k.is_empty())
        || req.is_extend()
}

/// The one wording for a source-image contract violation (#772), shared by
/// server admission, the CLI preflight, and the Discord preflight so the
/// rejection reads identically wherever it lands.
///
/// `family` selects the family-aware phrasing — Wan keeps its checkpoint-swap
/// suggestions, every other family (plain LTX-Video today) gets wording that
/// names the actual model instead of mislabeling it as Wan. `has_source`
/// counts first/last-frame keyframes as well as a source image (#779): both
/// carry source frames, so either satisfies a required contract and either is
/// refused by a text-to-video-only checkpoint. A `None` capability enforces
/// nothing — the engine remains the authority.
pub fn source_image_contract_violation(
    family: Option<&str>,
    model: &str,
    capability: Option<crate::types::SourceImageCapability>,
    has_source: bool,
) -> Option<String> {
    use crate::types::SourceImageCapability;
    let wan = family == Some("wan");
    match capability {
        Some(SourceImageCapability::Unsupported) if has_source => Some(if wan {
            "this Wan checkpoint is text-to-video only and does not accept a source image \
             or keyframes — remove them, or pick an I2V-capable checkpoint such as \
             wan22-ti2v-5b or wan22-i2v-a14b"
                .to_string()
        } else {
            format!(
                "{model} is text-to-video only and does not accept a source image — its \
                 engine has no image-to-video path; remove the image, or pick an \
                 image-capable checkpoint such as an LTX-2 model"
            )
        }),
        Some(SourceImageCapability::Required) if !has_source => Some(if wan {
            "this Wan I2V checkpoint needs a source image; supply one, or pick a \
             text-to-video checkpoint such as wan22-t2v-a14b"
                .to_string()
        } else {
            format!("{model} needs a source image; supply one")
        }),
        _ => None,
    }
}

/// Variant of [`validate_generate_request`] that accepts an explicit family
/// hint. The hint takes precedence over the manifest lookup, letting the HTTP
/// server feed in the catalog-resolved family for `cv:` / `hf:` model IDs.
/// Normalized creation-time filing resolved from a request's `tags` /
/// `collection` fields.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestOrganization {
    /// Tags in first-seen order, empties dropped and case-insensitive
    /// duplicates collapsed. Empty when the request filed under none.
    pub tags: Vec<String>,
    /// The collection reference exactly as it must be resolved at
    /// publication: `Ok(name)` for a create-by-name, `Err(id)` for an exact
    /// row this host still has to look up. `None` when no collection was
    /// requested.
    pub collection: Option<Result<String, String>>,
}

/// Validate the creation-time filing a request carries.
///
/// Refuses (422 at admission) an invalid tag, more than
/// [`crate::MAX_REQUEST_TAGS`] distinct tags, an invalid collection name, and
/// a `CollectionRef` with neither `id` nor `name` set. Everything else — an
/// empty tag, a repeated tag — is normalized away silently, because neither
/// is a mistake the user can see.
///
/// A `name` always wins over an `id` when both are present: the name is what
/// the print's embedded provenance will record, so resolving the id would
/// risk filing under one collection and recording another.
pub fn validate_request_organization(
    tags: Option<&[String]>,
    collection: Option<&crate::CollectionRef>,
) -> Result<RequestOrganization, String> {
    let tags = match tags {
        Some(raw) => crate::normalize_request_tags(raw)?,
        None => Vec::new(),
    };
    let collection = match collection {
        None => None,
        Some(reference) => {
            if reference.is_unset() {
                return Err(
                    "collection must set either 'name' (resolved by slug, created when absent) \
                     or 'id' (an existing collection on the generating host)"
                        .to_string(),
                );
            }
            match reference
                .name
                .as_deref()
                .map(str::trim)
                .filter(|name| !name.is_empty())
            {
                Some(name) => Some(Ok(crate::validate_collection_name(name)?.0)),
                None => Some(Err(reference
                    .id
                    .as_deref()
                    .map(str::trim)
                    .unwrap_or_default()
                    .to_string())),
            }
        }
    };
    Ok(RequestOrganization { tags, collection })
}

/// Rewrite a request's creation-time filing into the exact form that will be
/// applied, so saved provenance records what actually happened.
///
/// [`validate_request_organization`] already computes the normalized tags and
/// collection name, but checking is not enough: `OutputMetadata` is built
/// from the REQUEST while the gallery row is seeded through a path that
/// re-normalizes. Leave the raw spellings in place and the two disagree — a
/// direct HTTP client sending `[" Smurfs ", "smurfs"]` stamps both into the
/// embedded metadata while the row holds one `Smurfs`, and Reuse restores the
/// duplicates. First-party clients normalize before they send, so this is
/// invisible until someone drives the API with curl, which is exactly when a
/// provenance bug is hardest to explain.
///
/// Same discipline as [`materialize_extend_overlap_frames`]: the admitted
/// request is the record, so it must carry the effective value rather than
/// the requested one. Idempotent, and absent stays absent — a tag list that
/// normalizes away entirely becomes `None`, never `Some([])`, so an unfiled
/// print's provenance is unchanged.
///
/// A `{id}` collection reference is left alone: resolving it needs the host's
/// own collection table, which happens at admission in the server.
pub fn materialize_request_organization(req: &mut GenerateRequest) -> Result<(), String> {
    let organization = validate_request_organization(req.tags.as_deref(), req.collection.as_ref())?;

    if req.tags.is_some() {
        req.tags = (!organization.tags.is_empty()).then_some(organization.tags);
    }
    if let Some(Ok(name)) = organization.collection {
        // Keep any id the caller sent beside the canonical name — the server
        // reports the id it could not resolve, and the name is what the row
        // records.
        let id = req
            .collection
            .as_ref()
            .and_then(|reference| reference.id.clone());
        req.collection = Some(crate::CollectionRef {
            id,
            name: Some(name),
        });
    }
    Ok(())
}

pub fn validate_generate_request_with_family(
    req: &GenerateRequest,
    family_hint: Option<&str>,
) -> Result<(), String> {
    crate::require_model_activation(&req.model, family_hint).map_err(|error| error.to_string())?;
    validate_generate_request_after_activation(req, family_hint)
}

/// Validate the exact MiniMax H3 private-UAT request partition after the
/// server has issued its authenticated ingress grant.
///
/// This feature-gated helper does not grant authorization or activate a model;
/// it only prevents the already-authorized private route from re-entering the
/// public compliance gate before applying the same field validation.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub fn validate_h3_private_uat_request(req: &GenerateRequest) -> Result<(), String> {
    if !crate::minimax_h3::is_reviewed_compact_model(&req.model) {
        return Err(
            "private MiniMax H3 validation requires an exact reviewed task model".to_string(),
        );
    }
    validate_generate_request_after_activation(req, Some(crate::minimax_h3::FAMILY))
}

/// Shape/feature validation after the caller has passed the model-activation
/// authority. Kept private so tests can prove the future authorized H3 path
/// without exposing a compliance-gate bypass to production callers.
fn validate_generate_request_after_activation(
    req: &GenerateRequest,
    family_hint: Option<&str>,
) -> Result<(), String> {
    let family = resolved_family(&req.model, family_hint);

    if req.references.is_some() && !family.is_some_and(crate::minimax_h3::is_family) {
        return Err(
            "references is only supported by MiniMax H3 Ref2VA; other families retain their existing source/edit fields"
                .to_string(),
        );
    }

    // Creation-time filing is validated before anything expensive: a bad tag
    // or an unset collection ref is a client mistake, and paying for a model
    // load to discover it is the wrong trade.
    validate_request_organization(req.tags.as_deref(), req.collection.as_ref())?;

    if req.prompt.trim().is_empty() && prompt_required_for(req, family_hint) {
        return Err("prompt must not be empty".to_string());
    }
    // Resolve the composition from the request itself. A model that ships the
    // spatial upsampler renders stage 1 at half size and refines it over
    // tiles, which is the only way an axis past the trained RoPE span is in
    // distribution; anything else keeps the single-pass ceiling.
    let composition = if family == Some("ltx2") {
        ltx2_spatial_composition_for_request(req)
    } else {
        Ltx2SpatialComposition::SinglePass
    };
    let audio_only =
        family == Some("ltx2") && req.pipeline.is_some_and(Ltx2PipelineMode::is_audio_only);
    if !audio_only {
        validate_generation_dimensions_for_model(
            &req.model,
            req.width,
            req.height,
            family,
            composition,
        )?;
    }
    validate_family_video_timing_constraints(req.frames, req.fps, family)?;
    if composition == Ltx2SpatialComposition::TiledTwoStage {
        // The composed ceiling above is the x2 rung's. A request that names a
        // different rung reaches a different stage-1 shape, and only stage 1's
        // own span decides whether it is in distribution.
        validate_ltx2_stage1_span(req.width, req.height, req.spatial_upscale)?;
    }
    if req.steps == 0 {
        return Err("steps must be >= 1".to_string());
    }
    if req.steps > 100 {
        return Err(format!("steps ({}) must be <= 100", req.steps));
    }
    // Face-identity conditioning is its own contract; `crate::identity` owns
    // every rule so this validator does not grow a second authority.
    crate::identity::validate_identity_conditioning(req)?;
    if req.batch_size == 0 {
        return Err("batch_size must be >= 1".to_string());
    }
    // The shared inference/planning contract intentionally has no generic
    // upper limit. Live atomic HTTP delivery has a separate server-advertised
    // materialization bound because its durable manifest and response are
    // still O(batch_size).
    if req.guidance < 0.0 {
        return Err(format!("guidance ({}) must be >= 0.0", req.guidance));
    }
    if req.guidance > 100.0 {
        return Err(format!("guidance ({}) must be <= 100.0", req.guidance));
    }
    if req.prompt.len() > 77_000 {
        return Err(format!(
            "prompt length ({} bytes) exceeds the 77,000-byte limit",
            req.prompt.len()
        ));
    }
    if let Some(ref neg) = req.negative_prompt {
        if neg.len() > 77_000 {
            return Err(format!(
                "negative_prompt length ({} bytes) exceeds the 77,000-byte limit",
                neg.len()
            ));
        }
    }
    if family.is_some_and(crate::minimax_h3::is_family) {
        let task = crate::minimax_h3::task_for_model(&req.model).ok_or_else(|| {
            "MiniMax H3 requests must resolve an explicit FL2VA or Ref2VA task partition"
                .to_string()
        })?;
        if req.mask_image.is_some() {
            return Err("MiniMax H3 does not support mask_image".to_string());
        }
        if req.control_image.is_some() || req.control_model.is_some() {
            return Err("MiniMax H3 does not support ControlNet inputs".to_string());
        }
        if req.cfg_plus.is_some() {
            return Err("MiniMax H3 does not support cfg_plus".to_string());
        }
        if req.scheduler.is_some() {
            return Err(
                "MiniMax H3 uses its dedicated synchronized dual-shift schedule; scheduler overrides are unsupported"
                    .to_string(),
            );
        }
        if req.lora.is_some() || req.loras.is_some() {
            return Err("MiniMax H3 does not support LoRA".to_string());
        }
        if req.upscale_model.is_some() {
            return Err("MiniMax H3 does not support post-generation image upscaling".to_string());
        }
        if req.pipeline.is_some()
            || req.ic_lora_control.is_some()
            || req.retake_range.is_some()
            || req.spatial_upscale.is_some()
            || req.temporal_upscale.is_some()
            || req.guidance_overrides.is_some()
            || req.hdr_exr_dir.is_some()
            || req.hdr_exr_full_float
        {
            return Err("MiniMax H3 does not accept LTX-2 pipeline controls".to_string());
        }
        if req
            .source_image
            .as_deref()
            .is_some_and(|image| !is_valid_image_format(image))
        {
            return Err("source_image must be a PNG or JPEG image".to_string());
        }
        if req.source_image.is_some()
            && (!req.strength.is_finite() || !(0.0..=1.0).contains(&req.strength))
        {
            return Err(format!(
                "strength ({}) must be a finite value in range [0.0, 1.0] when source_image is provided",
                req.strength
            ));
        }
        if req
            .edit_images
            .as_ref()
            .is_some_and(|images| images.iter().any(|image| !is_valid_image_format(image)))
        {
            return Err("edit_images must contain only PNG or JPEG images".to_string());
        }
        if req.edit_images.as_ref().is_some_and(Vec::is_empty) {
            return Err("edit_images must not be empty when provided".to_string());
        }
        if req.keyframes.as_ref().is_some_and(|keyframes| {
            keyframes
                .iter()
                .any(|keyframe| !is_valid_image_format(&keyframe.image))
        }) {
            return Err("keyframes must contain only PNG or JPEG images".to_string());
        }
        if req.keyframes.as_ref().is_some_and(Vec::is_empty) {
            return Err("keyframes must not be empty when provided".to_string());
        }
        if req.extend_overlap_frames.is_some() {
            return Err(
                "extend_overlap_frames requires extend_video or extend_video_path, which MiniMax H3 does not support"
                    .to_string(),
            );
        }
        crate::minimax_h3::validate_request_contract(req, task)
            .map(|_| ())
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let flux2_dev = is_flux2_dev_model(&req.model);
    if family == Some("qwen-image-edit") {
        if req.edit_images.as_ref().is_none_or(Vec::is_empty) {
            return Err(
                "Qwen Image Edit needs at least one image. Add a Target image and try again."
                    .to_string(),
            );
        }
        if req.batch_size != 1 {
            return Err("qwen-image-edit only supports batch_size = 1".to_string());
        }
        if req.source_image.is_some() {
            return Err("qwen-image-edit uses edit_images instead of source_image".to_string());
        }
        if req.mask_image.is_some() {
            return Err("qwen-image-edit does not support mask_image".to_string());
        }
        if req.control_image.is_some() || req.control_model.is_some() {
            return Err("qwen-image-edit does not support ControlNet inputs".to_string());
        }
        if let Some(ref images) = req.edit_images {
            for image in images {
                if !is_valid_image_format(image) {
                    return Err("edit_images must contain only PNG or JPEG images".to_string());
                }
            }
        }
    } else if flux2_dev {
        if req.batch_size != 1
            && req
                .edit_images
                .as_ref()
                .is_some_and(|images| !images.is_empty())
        {
            return Err("flux2-dev reference editing only supports batch_size = 1".to_string());
        }
        if req.source_image.is_some() {
            return Err("flux2-dev uses edit_images instead of source_image".to_string());
        }
        if req.mask_image.is_some() {
            return Err("flux2-dev does not support mask_image".to_string());
        }
        if req.control_image.is_some() || req.control_model.is_some() {
            return Err("flux2-dev does not support ControlNet inputs".to_string());
        }
        if req.lora.is_some() || req.loras.as_ref().is_some_and(|loras| !loras.is_empty()) {
            return Err("flux2-dev does not support LoRA".to_string());
        }
        if let Some(images) = &req.edit_images {
            if images.len() > FLUX2_DEV_MAX_REFERENCE_IMAGES {
                return Err(format!(
                    "flux2-dev supports at most {FLUX2_DEV_MAX_REFERENCE_IMAGES} ordered reference images"
                ));
            }
            if images.iter().any(|image| !is_valid_image_format(image)) {
                return Err("edit_images must contain only PNG or JPEG images".to_string());
            }
        }
    } else if req.edit_images.is_some() {
        return Err(
            "edit_images are only supported for qwen-image-edit and flux2-dev models".to_string(),
        );
    }
    // img2img validation
    if let Some(ref img) = req.source_image {
        if req.strength < 0.0 || req.strength > 1.0 {
            return Err(format!(
                "strength ({}) must be in range [0.0, 1.0] when source_image is provided",
                req.strength
            ));
        }
        if !is_valid_image_format(img) {
            return Err("source_image must be a PNG or JPEG image".to_string());
        }
    }
    // ControlNet validation
    if let Some(ref ctrl) = req.control_image {
        require_controlnet_capable_family(family)?;
        if req.control_model.is_none() {
            return Err("control_image requires control_model to also be provided".to_string());
        }
        if !is_valid_image_format(ctrl) {
            return Err("control_image must be a PNG or JPEG image".to_string());
        }
        if req.control_scale < 0.0 {
            return Err(format!(
                "control_scale ({}) must be >= 0.0",
                req.control_scale
            ));
        }
    }
    if req.control_model.is_some() && req.control_image.is_none() {
        require_controlnet_capable_family(family)?;
        return Err("control_model requires control_image to also be provided".to_string());
    }
    // Inpainting validation
    if let Some(ref mask) = req.mask_image {
        if req.source_image.is_none() {
            return Err("mask_image requires source_image to also be provided".to_string());
        }
        if !is_valid_image_format(mask) {
            return Err("mask_image must be a PNG or JPEG image".to_string());
        }
    }
    // LoRA validation (format checks only — path existence is checked at the
    // inference layer, since in remote mode the path refers to the server filesystem).
    if let Some(ref lora) = req.lora {
        require_lora_capable_family(family)?;
        validate_lora_weight(lora, "lora")?;
        require_expert_routable_model(lora, &req.model, family)?;
    }
    if let Some(ref loras) = req.loras {
        if loras.is_empty() {
            return Err("loras must not be empty when provided".to_string());
        }
        require_lora_capable_family(family)?;
        for lora in loras {
            validate_lora_weight(lora, "loras")?;
            require_expert_routable_model(lora, &req.model, family)?;
        }
    }
    if let Some(fps) = req.fps {
        if fps == 0 {
            return Err("fps must be >= 1".to_string());
        }
        if fps > 120 {
            return Err(format!("fps ({fps}) must be <= 120"));
        }
    }
    // Video frame validation
    if let Some(frames) = req.frames {
        if frames == 0 {
            return Err("frames must be >= 1".to_string());
        }
        if let Some(step) = family.and_then(frame_step_for_family) {
            let offset = family.and_then(frame_offset_for_family).unwrap_or(1);
            if frames < offset || !(frames - offset).is_multiple_of(step) {
                return Err(format!(
                    "frames ({frames}) must be {step}n+{offset} for this model family (e.g. {}, {}, {}, …)",
                    step + offset,
                    2 * step + offset,
                    3 * step + offset,
                ));
            }
        }
        // LTX-2's ceiling is a duration (see `LTX2_MAX_RUNTIME_SECONDS`), so it
        // is derived per request from fps instead of the flat global ceiling.
        if matches!(family, Some("ltx2")) {
            let fps = req.fps.unwrap_or(LTX2_DEFAULT_FPS).max(1);
            // `derive_stage1_render_shape` halves BOTH the frame count and the
            // fps for `--temporal-upscale x2`, so stage 1 renders the same
            // runtime at half the frame rate. Mirror that: temporal upscaling
            // buys temporal resolution, never extra duration.
            let (stage1_frames, stage1_fps) = match req.temporal_upscale {
                Some(crate::Ltx2TemporalUpscale::X2) => {
                    (frames.saturating_sub(1) / 2 + 1, (fps / 2).max(1))
                }
                None => (frames, fps),
            };
            let stage1_cap = ltx2_max_frames_at_fps(stage1_fps);
            if stage1_frames > stage1_cap {
                // Quote a frame count the user can actually submit. The raw
                // duration ceiling is off the 8n+1 grid, so naming it sends
                // them straight into a second rejection.
                let delivered_cap = match req.temporal_upscale {
                    Some(crate::Ltx2TemporalUpscale::X2) => (stage1_cap - 1) * 2 + 1,
                    None => stage1_cap,
                };
                let delivered_cap = if delivered_cap > 1 {
                    delivered_cap - ((delivered_cap - 1) % 8)
                } else {
                    delivered_cap
                };
                return Err(format!(
                    "frames ({frames}) exceeds the LTX-2 / LTX-2.3 temporal RoPE budget of \
                     {LTX2_MAX_RUNTIME_SECONDS}s: at {fps} fps the ceiling is {delivered_cap} frames. \
                     Raise --fps, lower --frames, or render the shot as a multi-clip sequence"
                ));
            }
        } else {
            let max_frames = family
                .and_then(|family| {
                    max_frames_for_family_at_fps(family, req.fps.unwrap_or(LTX2_DEFAULT_FPS).max(1))
                })
                .unwrap_or(MAX_FRAMES_GLOBAL);
            if frames > max_frames {
                return Err(format!("frames ({frames}) must be <= {max_frames}"));
            }
        }
    }
    if let Some(keyframes) = &req.keyframes {
        validate_keyframes(keyframes, req.frames, family)?;
        // TI2V pins endpoints in latent space, where the 2.2 VAE's 4x
        // temporal stride turns a 5-frame pixel clip into two latent frames —
        // both anchored, nothing left to denoise. The engine refuses that
        // degenerate grid after the 10 GB load; admission must agree first.
        // 9 pixel frames (three latent frames) is the smallest 4k+1 clip with
        // an interior. Opaque cv:/hf: installs keep the engine's own check.
        if family == Some("wan")
            && keyframes.len() == 2
            && crate::manifest::resolve_model_name(&req.model).starts_with("wan22-ti2v-5b")
            && req
                .frames
                .is_some_and(|frames| frames < WAN_TI2V_FLF_MIN_FRAMES)
        {
            return Err(
                "wan22-ti2v-5b first/last-frame conditioning needs at least 9 frames — \
                 shorter clips leave no latent frames to denoise between the pinned endpoints"
                    .to_string(),
            );
        }
    }
    if let Some(audio) = &req.audio_file {
        require_ltx2_family(family, "audio_file")?;
        if req.audio_file_path.is_some() {
            return Err("audio_file_path cannot be combined with audio_file".to_string());
        }
        if audio.is_empty() {
            return Err("audio_file must not be empty".to_string());
        }
        validate_inline_media_size(audio, "audio_file", MAX_INLINE_AUDIO_BYTES)?;
    }
    if let Some(path) = &req.audio_file_path {
        require_ltx2_family(family, "audio_file_path")?;
        if path.trim().is_empty() {
            return Err("audio_file_path must not be empty".to_string());
        }
    }
    if let Some(video) = &req.source_video {
        require_ltx2_family(family, "source_video")?;
        if req.source_video_path.is_some() {
            return Err("source_video_path cannot be combined with source_video".to_string());
        }
        if video.is_empty() {
            return Err("source_video must not be empty".to_string());
        }
        validate_inline_media_size(video, "source_video", MAX_INLINE_SOURCE_VIDEO_BYTES)?;
    }
    if let Some(path) = &req.source_video_path {
        require_ltx2_family(family, "source_video_path")?;
        if path.trim().is_empty() {
            return Err("source_video_path must not be empty".to_string());
        }
    }
    validate_extend(req, family)?;
    // Only enforce the LTX-2 family gate when audio is actually requested
    // (`Some(true)`). The web form serializes its tri-state checkbox as
    // `Some(false)` when the user has explicitly turned audio off — which
    // must NOT trip a family error for video-only families, since the user
    // didn't ask for audio at all.
    if req.enable_audio == Some(true) {
        require_ltx2_family(family, "enable_audio")?;
    }
    if req.retake_range.is_some() {
        require_ltx2_family(family, "retake_range")?;
    }
    if req.spatial_upscale.is_some() {
        require_ltx2_family(family, "spatial_upscale")?;
    }
    if req.temporal_upscale.is_some() {
        require_ltx2_family(family, "temporal_upscale")?;
    }
    if req.pipeline.is_some() {
        require_ltx2_family(family, "pipeline")?;
    }
    if let Some(overrides) = &req.guidance_overrides {
        require_ltx2_family(family, "guidance_overrides")?;
        validate_guidance_overrides(overrides)?;
        // Cross-modal guidance needs both modalities resident. An audio-only
        // run has no video branch for `modality_scale` to act on, so a
        // non-1.0 value cannot be honoured — reject it instead of accepting
        // a number that would silently do nothing.
        if req.pipeline.is_some_and(Ltx2PipelineMode::is_audio_only) {
            if let Some(modality_scale) = overrides.modality_scale {
                if (modality_scale - 1.0).abs() > f64::EPSILON {
                    return Err(
                        "guidance_overrides.modality_scale must be 1.0 for pipeline=t2a: \
                         audio-only generation has no video modality to guide against"
                            .to_string(),
                    );
                }
            }
        }
    }
    if let Some(dir) = req.hdr_exr_dir.as_deref() {
        require_ltx2_family(family, "hdr_exr_dir")?;
        if dir.trim().is_empty() {
            return Err("hdr_exr_dir must not be empty".to_string());
        }
        // Today this is also unreachable transitively (hdr needs the ic-lora
        // control, which needs source_video, which extend forbids), but the
        // engine's extend path re-renders through the chain-stage machinery
        // where a per-clip EXR sequence would misalign with the stitched
        // timeline — say it directly instead of leaning on that implication
        // chain staying intact.
        if req.extend_video.is_some() || req.extend_video_path.is_some() {
            return Err("hdr_exr_dir cannot be combined with extend_video".to_string());
        }
        // The adapter is what makes the render HDR. Without it the decode
        // would apply a LogC3 inverse to an ordinary SDR signal and write a
        // wrongly-graded EXR that looks deliberate.
        // Through the shared normalizer, not a raw compare: every other
        // consumer accepts `HDR` and `hdr_`, so a bare `trim()` here would
        // reject spellings the rest of the stack resolves fine.
        if req
            .ic_lora_control
            .as_deref()
            .map(crate::ltx2_control::normalize_control_id)
            .as_deref()
            != Some("hdr")
        {
            return Err(
                "hdr_exr_dir requires ic_lora_control=hdr — EXR output is only meaningful for \
                 the HDR adapter's LogC3 signal"
                    .to_string(),
            );
        }
    } else if req.hdr_exr_full_float {
        return Err("hdr_exr_full_float requires hdr_exr_dir".to_string());
    }

    if let Some(control) = req.ic_lora_control.as_deref() {
        require_ltx2_family(family, "ic_lora_control")?;
        if control.trim().is_empty() {
            return Err("ic_lora_control must not be empty".to_string());
        }
        // Most control adapters drive the generic in-context pipeline. The
        // lip-dub adapter has its own pipeline (frozen stage-2 audio, an
        // appended audio reference, the LoRA on both stages), so it is the one
        // control whose required pipeline is not `ic-lora`.
        let required_pipeline = crate::ltx2_control::pipeline_for_control_id(control);
        if req.pipeline != Some(required_pipeline) {
            return Err(format!(
                "ic_lora_control '{}' requires pipeline={required_pipeline}",
                crate::ltx2_control::normalize_control_id(control)
            ));
        }
        if req.source_video.is_none() && req.source_video_path.is_none() {
            return Err("ic_lora_control requires source_video or source_video_path".to_string());
        }
        let user_loras = usize::from(req.lora.is_some()) + req.loras.as_ref().map_or(0, Vec::len);
        if user_loras + 1 > 4 {
            return Err(
                "ic_lora_control plus custom LoRAs exceeds the four-LoRA stack limit".to_string(),
            );
        }
    }

    // Wan renders video, with one deliberate exception: a single-frame render
    // is a still (#798) — upstream's own `t2i-14B` task is the same weights at
    // `frame_num=1` — so png/jpeg are admitted exactly when `frames == 1`.
    // Every other image format request would otherwise reach the engine and
    // fail after the model loads instead of at admission (Wan has no audio
    // path, so `wav` is refused here too).
    if family == Some("wan") {
        match (req.resolved_output_format(), req.frames) {
            (
                OutputFormat::Gif | OutputFormat::Apng | OutputFormat::Webp | OutputFormat::Mp4,
                _,
            ) => {}
            (OutputFormat::Png | OutputFormat::Jpeg, Some(1)) => {}
            _ => return Err("Wan outputs must use mp4, gif, apng, or webp".to_string()),
        }

        // First/last-frame conditioning (#779): the first frame comes from
        // either `source_image` or `keyframes[0]`, never both — an ambiguous
        // mix is refused at admission with the engine's own wording.
        if req.source_image.is_some() && req.keyframes.as_ref().is_some_and(|k| !k.is_empty()) {
            return Err(
                "Wan takes the first frame from either source_image or keyframes[0], not both \
                 — for a first/last-frame render, put both endpoints in keyframes"
                    .to_string(),
            );
        }
    }

    // The scheduler slot is shared by two disjoint solver families (#795):
    // wan's flow solvers are rejected off-family and the UNet schedulers are
    // rejected for wan — at admission, not after the model loads.
    match req.scheduler {
        Some(crate::Scheduler::Euler | crate::Scheduler::DpmPp) if family != Some("wan") => {
            return Err(format!(
                "scheduler '{}' is a Wan sample solver and is only supported for wan models",
                req.scheduler.expect("matched Some")
            ));
        }
        Some(crate::Scheduler::Ddim | crate::Scheduler::EulerAncestral)
            if family == Some("wan") =>
        {
            return Err(format!(
                "Wan supports the uni-pc, euler, and dpm-pp sample solvers; '{}' is a UNet \
                 scheduler",
                req.scheduler.expect("matched Some")
            ));
        }
        _ => {}
    }

    // Wan flow shift (#782): rejected, not ignored, off-family — a silently
    // inert quality knob looks like the knob failing.
    if let Some(shift) = req.sample_shift {
        if family != Some("wan") {
            return Err(
                "sample_shift is a Wan flow-matching control and is not supported for this model"
                    .to_string(),
            );
        }
        if !shift.is_finite() || shift <= 0.0 {
            return Err(format!(
                "sample_shift must be finite and positive, got {shift}"
            ));
        }
    }

    // The wan fp8-scaled A14B tier refuses every adapter stack (#777): an fp8
    // merge would re-round each targeted weight to three mantissa bits, and
    // the loader fails closed — but only after the UMT5 encode. Name it at
    // admission instead. Opaque cv:/hf: installs keep the engine's check.
    if family == Some("wan")
        && (req.lora.is_some() || req.loras.as_ref().is_some_and(|list| !list.is_empty()))
    {
        let canonical = crate::manifest::resolve_model_name(&req.model);
        if canonical.ends_with(":fp8") && canonical.contains("a14b") {
            return Err(format!(
                "{canonical} is fp8-scaled and refuses LoRA stacks — merging would re-round \
                 every targeted weight to three mantissa bits. Use the :q5/:q8 GGUF or bf16 \
                 tier for adapters"
            ));
        }
    }

    // Wan Lightning distill strengths (#795): wan only, within the accepted
    // band. Whether the model actually ships a distill in the addressed slot
    // is the engine's check — it knows the resolved component paths.
    for (label, value) in [
        ("high", req.distill_strength_high),
        ("low", req.distill_strength_low),
    ] {
        if let Some(strength) = value {
            if family != Some("wan") {
                return Err(format!(
                    "distill_strength_{label} is a Wan Lightning control and is not supported \
                     for this model"
                ));
            }
            if !strength.is_finite() || strength <= 0.0 || strength > 4.0 {
                return Err(format!(
                    "distill_strength_{label} must be in (0, 4], got {strength}"
                ));
            }
        }
    }

    if family == Some("ltx2") {
        let audio_only = req.pipeline.is_some_and(Ltx2PipelineMode::is_audio_only);
        match (req.resolved_output_format(), audio_only) {
            (OutputFormat::Wav, true) => {}
            (OutputFormat::Wav, false) => {
                return Err("wav output requires pipeline=t2a".to_string());
            }
            (_, true) => {
                return Err("pipeline=t2a renders audio only; set output_format=wav".to_string());
            }
            (
                OutputFormat::Gif | OutputFormat::Apng | OutputFormat::Webp | OutputFormat::Mp4,
                false,
            ) => {}
            (_, false) => return Err("LTX-2 outputs must use mp4, gif, apng, or webp".to_string()),
        }

        if req.enable_audio == Some(true)
            && !audio_only
            && req.resolved_output_format() != OutputFormat::Mp4
        {
            return Err("audio-enabled LTX-2 outputs must use mp4 format".to_string());
        }
        if req.enable_audio == Some(false) && audio_only {
            return Err("pipeline=t2a cannot be combined with enable_audio=false".to_string());
        }

        if req.retake_range.is_some()
            && req.source_video.is_none()
            && req.source_video_path.is_none()
        {
            return Err(
                "retake_range requires source_video or source_video_path to also be provided"
                    .to_string(),
            );
        }

        if let Some(range) = &req.retake_range {
            if !(range.start_seconds.is_finite() && range.end_seconds.is_finite()) {
                return Err("retake_range values must be finite numbers".to_string());
            }
            if range.start_seconds < 0.0 {
                return Err("retake_range start_seconds must be >= 0.0".to_string());
            }
            if range.end_seconds <= range.start_seconds {
                return Err(
                    "retake_range end_seconds must be greater than start_seconds".to_string(),
                );
            }
        }

        if let Some(pipeline) = req.pipeline {
            match pipeline {
                Ltx2PipelineMode::A2Vid => {
                    if req.audio_file.is_none() && req.audio_file_path.is_none() {
                        return Err(
                            "pipeline=a2-vid requires audio_file or audio_file_path".to_string()
                        );
                    }
                }
                Ltx2PipelineMode::Retake => {
                    if req.source_video.is_none() && req.source_video_path.is_none() {
                        return Err("pipeline=retake requires source_video or source_video_path"
                            .to_string());
                    }
                    if req.retake_range.is_none() {
                        return Err("pipeline=retake requires retake_range".to_string());
                    }
                }
                Ltx2PipelineMode::Keyframe => {
                    let keyframe_count = req.keyframes.as_ref().map_or(0, Vec::len);
                    if keyframe_count < 2 {
                        return Err("pipeline=keyframe requires at least 2 keyframes".to_string());
                    }
                }
                Ltx2PipelineMode::IcLora => {
                    if req.source_video.is_none() && req.source_video_path.is_none() {
                        return Err(
                            "pipeline=ic-lora requires source_video or source_video_path"
                                .to_string(),
                        );
                    }
                    if req.ic_lora_control.is_none()
                        && req.lora.is_none()
                        && req.loras.as_ref().is_none_or(Vec::is_empty)
                    {
                        return Err("pipeline=ic-lora requires at least one LoRA".to_string());
                    }
                }
                Ltx2PipelineMode::LipDub => {
                    if req.source_video.is_none() && req.source_video_path.is_none() {
                        return Err(
                            "pipeline=lip-dub requires source_video or source_video_path (the \
                             clip being re-voiced)"
                                .to_string(),
                        );
                    }
                    if req.ic_lora_control.is_none()
                        && req.lora.is_none()
                        && req.loras.as_ref().is_none_or(Vec::is_empty)
                    {
                        return Err("pipeline=lip-dub requires the lip-dub IC-LoRA; pass \
                             ic_lora_control=lipdub"
                            .to_string());
                    }
                    // Upstream asserts a two-stage resolution before doing any
                    // work (`assert_resolution(..., is_two_stage=True)` in
                    // `utils/helpers.py:321-332`). Lip dub is always two-stage
                    // — stage 1 renders at half size — so an odd multiple of 32
                    // would leave stage 1 off the latent grid.
                    if !req.width.is_multiple_of(LTX2_TWO_STAGE_ALIGNMENT)
                        || !req.height.is_multiple_of(LTX2_TWO_STAGE_ALIGNMENT)
                    {
                        return Err(format!(
                            "pipeline=lip-dub renders in two stages, so width and height must be \
                             multiples of {LTX2_TWO_STAGE_ALIGNMENT}; got {}x{}",
                            req.width, req.height
                        ));
                    }
                    if req.retake_range.is_some() {
                        return Err(
                            "pipeline=lip-dub cannot be combined with retake_range".to_string()
                        );
                    }
                    if req
                        .keyframes
                        .as_ref()
                        .is_some_and(|items| !items.is_empty())
                    {
                        return Err(
                            "pipeline=lip-dub cannot be combined with keyframes".to_string()
                        );
                    }
                    // Both would change the output shape out from under the
                    // reference clip, whose resolution and length the dub has
                    // to match. Upstream's pipeline composes with neither.
                    if req.spatial_upscale.is_some() || req.temporal_upscale.is_some() {
                        return Err(
                            "pipeline=lip-dub cannot be combined with spatial_upscale or \
                             temporal_upscale; the render must match the reference video"
                                .to_string(),
                        );
                    }
                }
                Ltx2PipelineMode::T2a => {
                    // Text-to-audio has no video modality at all: there is no
                    // frame to condition on and no cross-modal path for a
                    // reference to reach. Reject conditioning outright rather
                    // than silently ignoring inputs the caller paid to upload.
                    for (present, field) in [
                        (req.source_image.is_some(), "source_image"),
                        (req.source_video.is_some(), "source_video"),
                        (req.source_video_path.is_some(), "source_video_path"),
                        (req.audio_file.is_some(), "audio_file"),
                        (req.audio_file_path.is_some(), "audio_file_path"),
                        (req.is_extend(), "extend_video"),
                        (
                            req.keyframes.as_ref().is_some_and(|k| !k.is_empty()),
                            "keyframes",
                        ),
                        (req.retake_range.is_some(), "retake_range"),
                        (req.spatial_upscale.is_some(), "spatial_upscale"),
                        (req.temporal_upscale.is_some(), "temporal_upscale"),
                        (req.upscale_model.is_some(), "upscale_model"),
                    ] {
                        if present {
                            return Err(format!(
                                "pipeline=t2a generates audio only and cannot be combined with {field}"
                            ));
                        }
                    }
                }
                Ltx2PipelineMode::OneStage
                | Ltx2PipelineMode::TwoStage
                | Ltx2PipelineMode::TwoStageHq
                | Ltx2PipelineMode::Distilled => {}
            }
        }
    }

    Ok(())
}

/// Whether a stable name or catalog ID denotes the first-party FLUX.2 Dev
/// architecture rather than a Klein checkpoint.
pub fn is_flux2_dev_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("flux2-dev") || model.contains("flux.2-dev")
}

/// Validate an upscale request. Returns `Ok(())` if valid, or an error message.
pub fn validate_upscale_request(req: &UpscaleRequest) -> Result<(), String> {
    if req.model.trim().is_empty() {
        return Err("upscale model must not be empty".to_string());
    }
    if req.image.is_empty() {
        return Err("upscale image must not be empty".to_string());
    }
    if !is_valid_image_format(&req.image) {
        return Err("upscale image must be a PNG or JPEG image".to_string());
    }
    if let Some(tile_size) = req.tile_size {
        if tile_size != 0 && tile_size < 64 {
            return Err(format!(
                "tile_size ({tile_size}) must be 0 (disabled) or >= 64"
            ));
        }
    }
    Ok(())
}

// ── Dimension recommendations ───────────────────────────────────────────────

/// Per-checkpoint recommended buckets for the Wan family.
///
/// The family-wide list unions buckets no single checkpoint supports —
/// `wan21-t2v-1.3b` is 480p-only, and `wan22-ti2v-5b`'s native pair is
/// 1280x704 on its 2.2 VAE's 32px grid — so `/api/models` resolves the
/// advertisement per model. The family list remains the fallback for
/// checkpoints this build has no manifest for (catalog `cv:`/`hf:` ids).
pub fn wan_recommended_dimensions(model: &str) -> &'static [(u32, u32)] {
    crate::generation_profile::presets_for_identity(model, "wan", None)
}

/// Per-checkpoint dimension grid for the Wan family.
///
/// `wan22-ti2v-5b`'s 2.2 VAE compresses 16x spatially and its DiT patches the
/// latent 2x2, so its pixel grid is 32 — the engine enforces exactly this
/// product after loading (`wan/pipeline.rs`), and admission must agree so an
/// off-grid canvas never queues a 10 GB load it cannot survive. The 2.1-VAE
/// checkpoints (1.3B, A14B: 8x stride x 2x2 patch) keep the family's 16.
/// Mirrors [`wan_recommended_dimensions`]: variant tags and legacy dash names
/// resolve through the manifest first, and unknown `cv:`/`hf:` installs keep
/// the family fallback — deriving the grid from a sidecar-described VAE
/// component is deliberately follow-up work.
pub fn wan_dimension_alignment(model: &str) -> u32 {
    let canonical = crate::manifest::resolve_model_name(model);
    if canonical.starts_with("wan22-ti2v-5b") {
        return 32;
    }
    dimension_alignment_for_family(Some("wan"))
}

/// Return the list of recommended (width, height) pairs for a model family.
///
/// Returns an empty slice for unknown families, utility models (e.g. `qwen3-expand`),
/// and conditioning models (e.g. ControlNet).
pub fn recommended_dimensions(family: &str) -> &'static [(u32, u32)] {
    crate::generation_profile::family_presets(family)
}

/// Composition-aware counterpart to [`recommended_dimensions`].
///
/// `/api/models` advertises this per model so a checkpoint that cannot compose
/// never offers a rung it cannot render. Returns an owned list because the
/// composed ladder is the base list plus the composed rungs.
pub fn recommended_dimensions_composed(
    family: &str,
    composition: Ltx2SpatialComposition,
) -> Vec<(u32, u32)> {
    let base = recommended_dimensions(family);
    if family != "ltx2" || composition != Ltx2SpatialComposition::TiledTwoStage {
        return base.to_vec();
    }
    // Derived from the ladder rather than restated beside it. A rung that
    // needs tiling is exactly a rung a single-pass checkpoint cannot render,
    // which is exactly the set to withhold from one.
    let mut out = base.to_vec();
    for rung in LTX2_OUTPUT_RUNGS
        .iter()
        .filter(|rung| rung.requires_tiled_stage2())
    {
        out.push((rung.width, rung.height));
        out.push((rung.height, rung.width));
    }
    out
}

/// Check if the requested dimensions match any recommended resolution for the model family.
///
/// Returns `None` if the dimensions are recommended or the family has no recommendation list.
/// Returns `Some(warning_message)` with suggested alternatives otherwise.
pub fn dimension_warning(width: u32, height: u32, family: &str) -> Option<String> {
    dimension_warning_composed(width, height, family, Ltx2SpatialComposition::SinglePass)
}

/// Composition-aware counterpart to [`dimension_warning`].
///
/// A composing LTX-2 checkpoint has more buckets than its family fallback, so
/// the single-pass list would call 3840x2176 unrecommended on the very
/// checkpoint the rung was added for.
pub fn dimension_warning_composed(
    width: u32,
    height: u32,
    family: &str,
    composition: Ltx2SpatialComposition,
) -> Option<String> {
    let dims = recommended_dimensions_composed(family, composition);
    if dims.is_empty() {
        return None;
    }
    if dims.contains(&(width, height)) {
        return None;
    }
    // Build a compact list of suggested alternatives (show up to 4)
    let suggestions: Vec<String> = dims
        .iter()
        .take(4)
        .map(|(w, h)| format!("{w}x{h}"))
        .collect();
    let more = if dims.len() > 4 {
        format!(", ... ({} total)", dims.len())
    } else {
        String::new()
    };
    Some(format!(
        "{width}x{height} is not a recommended resolution for {family} models. \
         Suggested: {}{}",
        suggestions.join(", "),
        more,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputFormat;

    /// Only the A14B pair has experts; silently ignoring the field elsewhere
    /// would let a user believe an adapter was bound to half the schedule when
    /// it was applied to all of it.
    #[test]
    fn an_expert_bound_lora_needs_a_two_expert_checkpoint() {
        let lora = |expert| LoraWeight {
            path: "/loras/high_noise_model.safetensors".to_string(),
            scale: 1.0,
            expert,
        };

        // The A14B pair accepts it.
        assert!(require_expert_routable_model(
            &lora(Some(crate::LoraExpert::High)),
            "wan22-t2v-a14b:q5",
            Some("wan"),
        )
        .is_ok());

        // Single-expert wan checkpoints name the problem and the remedy.
        for model in ["wan21-t2v-1.3b:bf16", "wan22-ti2v-5b:fp16"] {
            let error = require_expert_routable_model(
                &lora(Some(crate::LoraExpert::Low)),
                model,
                Some("wan"),
            )
            .unwrap_err();
            assert!(error.contains("single-expert"), "{error}");
            assert!(error.contains("drop the expert field"), "{error}");
        }

        // A non-wan family has no experts at all.
        let error = require_expert_routable_model(
            &lora(Some(crate::LoraExpert::High)),
            "flux-dev:q8",
            Some("flux"),
        )
        .unwrap_err();
        assert!(error.contains("not a Wan model"), "{error}");

        // An opaque catalog id cannot be classified by name; the engine reads
        // the real pair from the checkpoint, so admission does not guess.
        assert!(require_expert_routable_model(
            &lora(Some(crate::LoraExpert::High)),
            "cv:123456",
            Some("wan"),
        )
        .is_ok());

        // Absent stays absent — the historical apply-to-both path.
        assert!(require_expert_routable_model(&lora(None), "flux-dev:q8", Some("flux")).is_ok());
    }

    /// Upstream's own shipped LTX-2.3 HQ default is 1920x1088
    /// (`LTX_2_3_HQ_PARAMS`: stage 1 at 960x544, refined x2). That is
    /// 2,088,960 px, so the flat 1.8 MP ceiling made mold unable to express
    /// the reference implementation's own top-end preset.
    #[test]
    fn ltx2_admits_upstreams_shipped_1080p_shape() {
        assert!(validate_generation_dimensions(1920, 1088, Some("ltx2")).is_ok());
        assert!(validate_generation_dimensions(1088, 1920, Some("ltx2")).is_ok());
    }

    #[test]
    fn non_ltx2_families_keep_the_default_ceiling() {
        for family in [Some("flux"), Some("ltx-video"), Some("sdxl"), None] {
            let err = validate_generation_dimensions(1920, 1088, family)
                .expect_err("only LTX-2 gets the raised ceiling");
            assert!(
                err.contains("1.8MP"),
                "{family:?} must still report the default limit, got: {err}"
            );
        }
    }

    /// Independent of the pixel budget. The checkpoints ship
    /// `positional_embedding_max_pos = [20, 2048, 2048]` and normalize pixel
    /// positions by it, so an axis past 2048 is out of distribution even when
    /// the frame is small: 3200x512 is only 1.64 MP but its width position is
    /// 1.5625, far outside the trained [-1, 1].
    #[test]
    fn ltx2_rejects_an_axis_beyond_the_rope_span() {
        let err = validate_generation_dimensions(3200, 512, Some("ltx2"))
            .expect_err("an over-wide axis must be rejected on its own merits");
        assert!(
            err.contains("2048"),
            "the error must name the axis limit, got: {err}"
        );
        // The transpose is equally out of distribution.
        assert!(validate_generation_dimensions(512, 3200, Some("ltx2")).is_err());
        // Exactly at the span is in distribution, when the pixel budget also
        // allows it: 2048x992 is 2.03 MP, 2048x1024 would be 2.10 MP and is
        // rejected on pixels instead. The two limits are independent.
        assert!(validate_generation_dimensions(2048, 992, Some("ltx2")).is_ok());
        assert!(validate_generation_dimensions(2048, 1024, Some("ltx2"))
            .expect_err("over the pixel budget")
            .contains("megapixels"));
    }

    #[test]
    fn ltx2_recommended_dimensions_are_grid_aligned_and_inside_the_family_ceiling() {
        for &(width, height) in recommended_dimensions("ltx2") {
            assert!(
                validate_generation_dimensions(width, height, Some("ltx2")).is_ok(),
                "advertised preset {width}x{height} must be admissible"
            );
        }
    }

    /// The whole point of gating the raised ceiling on the composition is that
    /// nothing renderable today changes. Every shape the single-pass validator
    /// used to accept or reject must still get the same answer, at exactly the
    /// same boundary — a ceiling that leaked into the default path would admit
    /// out-of-distribution renders on one-stage checkpoints.
    #[test]
    fn single_pass_admission_is_byte_for_byte_unchanged() {
        // Accepted before, accepted now.
        for &(width, height) in &[
            (768u32, 512u32),
            (1216, 704),
            (1920, 1088),
            (1088, 1920),
            (2048, 992),
        ] {
            assert!(
                validate_generation_dimensions(width, height, Some("ltx2")).is_ok(),
                "{width}x{height} was admissible before the composed ceiling"
            );
        }
        // Rejected before, rejected now — on the same limit each time.
        assert!(validate_generation_dimensions(2048, 1024, Some("ltx2"))
            .expect_err("2.10 MP is over the single-pass pixel budget")
            .contains("megapixels"));
        assert!(validate_generation_dimensions(3200, 512, Some("ltx2"))
            .expect_err("a 3200px axis is past the trained span")
            .contains("2048"));
        assert!(validate_generation_dimensions(2080, 512, Some("ltx2")).is_err());
    }

    /// The threshold is the trained span itself, not a rounded-down neighbour.
    /// One latent cell either side of 2048 is the difference between the shape
    /// upstream ships and a shape no checkpoint has seen.
    #[test]
    fn the_axis_threshold_fires_exactly_at_the_trained_span() {
        // 2048 is the last in-distribution axis for a single pass; 2080 is the
        // next 32-aligned value and is the first rejected one.
        assert!(validate_generation_dimensions(2048, 512, Some("ltx2")).is_ok());
        assert!(validate_generation_dimensions(2080, 512, Some("ltx2")).is_err());

        // A composed render admits both, and its own ceiling behaves the same
        // way one cell either side of 4096.
        let composed = Ltx2SpatialComposition::TiledTwoStage;
        assert!(validate_generation_dimensions_composed(2080, 512, Some("ltx2"), composed).is_ok());
        assert!(
            validate_generation_dimensions_composed(4096, 2176, Some("ltx2"), composed).is_ok()
        );
        assert!(
            validate_generation_dimensions_composed(4128, 2176, Some("ltx2"), composed).is_err(),
            "past 4096 the halved stage-1 shape is itself out of distribution"
        );
    }

    /// The composed ceiling is `2 * trained span` for a reason that has to
    /// stay true: stage 1 renders the target halved, so 4096 is the widest
    /// target whose stage 1 still lands inside the span.
    #[test]
    fn the_composed_ceiling_is_where_stage_one_leaves_the_trained_span() {
        assert_eq!(LTX2_COMPOSED_MAX_AXIS_PIXELS, 2 * LTX2_MAX_AXIS_PIXELS);
        let widest = Ltx2OutputRung {
            id: "test",
            label: "test",
            width: LTX2_COMPOSED_MAX_AXIS_PIXELS,
            height: 2_176,
        };
        assert_eq!(widest.stage1_shape().0, LTX2_MAX_AXIS_PIXELS);

        // One rung wider and stage 1 is already out of distribution, which no
        // amount of stage-2 tiling repairs.
        let too_wide = Ltx2OutputRung {
            id: "test",
            label: "test",
            width: LTX2_COMPOSED_MAX_AXIS_PIXELS + 64,
            height: 2_176,
        };
        assert!(too_wide.stage1_shape().0 > LTX2_MAX_AXIS_PIXELS);
    }

    /// A checkpoint reaches the composed ceiling only if it can actually
    /// compose: it ships the spatial upsampler *and* runs a refining pipeline.
    #[test]
    fn the_composed_ceiling_requires_a_checkpoint_that_can_compose() {
        // Manifest LTX-2 checkpoints all ship the upsampler.
        assert_eq!(
            ltx2_spatial_composition("ltx-2-19b-distilled:fp8", None),
            Ltx2SpatialComposition::TiledTwoStage
        );
        // A single-file catalog checkpoint has no manifest and no upsampler.
        assert_eq!(
            ltx2_spatial_composition("cv:3143864", None),
            Ltx2SpatialComposition::SinglePass
        );
        // An explicit non-refining pipeline denoises the requested shape once,
        // however capable the checkpoint is.
        for mode in [
            Ltx2PipelineMode::OneStage,
            Ltx2PipelineMode::Retake,
            Ltx2PipelineMode::LipDub,
        ] {
            assert_eq!(
                ltx2_spatial_composition("ltx-2-19b-distilled:fp8", Some(mode)),
                Ltx2SpatialComposition::SinglePass,
                "{mode} denoises once and cannot hold an oversized axis"
            );
        }
        for mode in Ltx2PipelineMode::ALL
            .iter()
            .filter(|m| m.refines_spatially())
        {
            assert_eq!(
                ltx2_spatial_composition("ltx-2-19b-distilled:fp8", Some(*mode)),
                Ltx2SpatialComposition::TiledTwoStage,
                "{mode} refines a halved stage 1 and can hold one"
            );
        }
    }

    /// End-to-end through the request validator: the same 4K request is
    /// admitted on a composing checkpoint and refused on a one-stage one, and
    /// the refusal says what would make it work.
    #[test]
    fn a_4k_request_is_admitted_only_where_the_composition_exists() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.width = 3_840;
        req.height = 2_176;
        req.frames = Some(25);
        req.fps = Some(24);
        req.output_format = Some(OutputFormat::Mp4);
        validate_generate_request_with_family(&req, Some("ltx2"))
            .expect("a composing checkpoint reaches 4K UHD");

        req.model = "cv:3143864".to_string();
        let err = validate_generate_request_with_family(&req, Some("ltx2"))
            .expect_err("a one-stage checkpoint cannot");
        assert!(
            err.contains("3840") && err.contains("spatial upsampler"),
            "the refusal must name the axis and the way out, got: {err}"
        );

        // Explicitly asking for a one-stage render is refused on the same
        // grounds, even on the composing checkpoint.
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.pipeline = Some(Ltx2PipelineMode::OneStage);
        assert!(validate_generate_request_with_family(&req, Some("ltx2")).is_err());
    }

    /// Every advertised rung has to be admissible under the composition that
    /// reaches it — and the composed-only ones have to be refused without it,
    /// or a one-stage checkpoint would be offered a size it cannot render.
    #[test]
    fn every_composed_rung_is_admissible_exactly_under_composition() {
        let two_stage = Ltx2SpatialComposition::TiledTwoStage;
        for (width, height) in recommended_dimensions_composed("ltx2", two_stage) {
            assert!(
                validate_generation_dimensions_composed(width, height, Some("ltx2"), two_stage)
                    .is_ok(),
                "advertised composed preset {width}x{height} must be admissible"
            );
        }
        for rung in LTX2_OUTPUT_RUNGS {
            let (width, height) = (rung.width, rung.height);
            assert!(
                width.is_multiple_of(LTX2_TWO_STAGE_ALIGNMENT)
                    && height.is_multiple_of(LTX2_TWO_STAGE_ALIGNMENT),
                "{width}x{height} must survive halving onto the 32px latent grid"
            );
            if !rung.requires_tiled_stage2() {
                continue;
            }
            for shape in [(width, height), (height, width)] {
                assert!(
                    validate_generation_dimensions(shape.0, shape.1, Some("ltx2")).is_err(),
                    "{}x{} must not be offered to a single-pass checkpoint",
                    shape.0,
                    shape.1
                );
                assert!(
                    recommended_dimensions_composed("ltx2", two_stage).contains(&shape),
                    "{}x{} must be advertised to a composing checkpoint",
                    shape.0,
                    shape.1
                );
            }
        }
        // A single-pass model's advertised list is exactly the old one.
        assert_eq!(
            recommended_dimensions_composed("ltx2", Ltx2SpatialComposition::SinglePass),
            recommended_dimensions("ltx2").to_vec()
        );
    }

    /// The ladder's arithmetic: each rung's stage-1 shape is the target halved
    /// onto the latent grid, and its tile counts are the fewest tiles that
    /// bring every axis back inside the trained span.
    #[test]
    fn rung_composition_arithmetic_is_exact() {
        struct ExpectedRung {
            id: &'static str,
            stage1: (u32, u32),
            /// `(columns, rows)`.
            tiles: (u32, u32),
            tiled: bool,
        }
        let expected = [
            ExpectedRung {
                id: "720p",
                stage1: (640, 352),
                tiles: (1, 1),
                tiled: false,
            },
            ExpectedRung {
                id: "1080p",
                stage1: (960, 544),
                tiles: (1, 1),
                tiled: false,
            },
            ExpectedRung {
                id: "1440p",
                stage1: (1_280, 704),
                tiles: (2, 1),
                tiled: true,
            },
            ExpectedRung {
                id: "4k-uhd",
                stage1: (1_920, 1_056),
                tiles: (2, 2),
                tiled: true,
            },
        ];
        assert_eq!(LTX2_OUTPUT_RUNGS.len(), expected.len());
        for (
            rung,
            ExpectedRung {
                id,
                stage1,
                tiles,
                tiled,
            },
        ) in LTX2_OUTPUT_RUNGS.iter().zip(&expected)
        {
            let (id, stage1, tiles, tiled) = (*id, *stage1, *tiles, *tiled);
            assert_eq!(rung.id, id);
            assert_eq!(rung.stage1_shape(), stage1, "{id} stage-1 shape");
            assert_eq!(rung.stage2_tiles(), tiles, "{id} stage-2 tile counts");
            assert_eq!(rung.requires_tiled_stage2(), tiled, "{id} tiling need");
            // A rung is only meaningful if its own advertised shape is
            // admissible under the composition that reaches it.
            assert!(validate_generation_dimensions_composed(
                rung.width,
                rung.height,
                Some("ltx2"),
                Ltx2SpatialComposition::TiledTwoStage,
            )
            .is_ok());
        }
    }

    /// The composed ceiling is the **x2 rung's**. x1.5 divides by 1.5, so the
    /// same 4K output leaves stage 1 at 2560px — out of distribution, with
    /// nothing downstream to repair it, because stage 2 tiles the refinement
    /// and never stage 1.
    #[test]
    fn a_smaller_spatial_rung_lowers_the_ceiling_it_can_reach() {
        assert_eq!(
            ltx2_composed_axis_ceiling(Some(Ltx2SpatialUpscale::X2)),
            LTX2_COMPOSED_MAX_AXIS_PIXELS
        );
        assert_eq!(
            ltx2_composed_axis_ceiling(None),
            LTX2_COMPOSED_MAX_AXIS_PIXELS
        );
        assert_eq!(
            ltx2_composed_axis_ceiling(Some(Ltx2SpatialUpscale::X1_5)),
            3_072
        );

        // Every ceiling is exactly the largest target its rung can hold, and
        // one grid step past it is not.
        for upscale in [Some(Ltx2SpatialUpscale::X2), Some(Ltx2SpatialUpscale::X1_5)] {
            let ceiling = ltx2_composed_axis_ceiling(upscale);
            assert!(
                ltx2_stage1_axis_for(ceiling, upscale) <= LTX2_MAX_AXIS_PIXELS,
                "{upscale:?} must reach its own ceiling"
            );
            assert!(
                ltx2_stage1_axis_for(ceiling + LTX2_SPATIAL_LATENT_STRIDE, upscale)
                    > LTX2_MAX_AXIS_PIXELS,
                "{upscale:?} must not reach one grid step past it"
            );
        }

        // 4K on x1.5 is refused, and the refusal names the shape stage 1 would
        // have rendered rather than restating the output size.
        let err = validate_ltx2_stage1_span(3_840, 2_176, Some(Ltx2SpatialUpscale::X1_5))
            .expect_err("x1.5 cannot halve 3840 back inside the span");
        assert!(err.contains("2560") && err.contains("3072"), "got: {err}");
        // The same output on x2 is fine.
        assert!(validate_ltx2_stage1_span(3_840, 2_176, Some(Ltx2SpatialUpscale::X2)).is_ok());
        // And x1.5 is fine at its own ceiling.
        assert!(validate_ltx2_stage1_span(3_072, 1_728, Some(Ltx2SpatialUpscale::X1_5)).is_ok());
    }

    /// The whole request decides the pipeline, not just the `pipeline` field.
    /// `select_pipeline` routes a retake before it ever considers the
    /// checkpoint's upsampler, and a retake denoises once.
    #[test]
    fn an_implicit_retake_is_admitted_as_single_pass() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.width = 3_840;
        req.height = 2_176;
        req.frames = Some(25);
        req.fps = Some(24);
        req.output_format = Some(OutputFormat::Mp4);
        // No explicit pipeline: the composing default admits 4K.
        assert_eq!(
            ltx2_spatial_composition_for_request(&req),
            Ltx2SpatialComposition::TiledTwoStage
        );
        validate_generate_request_with_family(&req, Some("ltx2")).expect("4K composes");

        // Adding a retake range changes what the engine will run, so it has to
        // change what admission allows — otherwise this is refused minutes
        // later by the engine backstop instead of at the request boundary.
        req.retake_range = Some(crate::TimeRange {
            start_seconds: 0.0,
            end_seconds: 0.5,
        });
        req.source_video_path = Some("/tmp/clip.mp4".to_string());
        assert_eq!(
            ltx2_spatial_composition_for_request(&req),
            Ltx2SpatialComposition::SinglePass
        );
        let err = validate_generate_request_with_family(&req, Some("ltx2"))
            .expect_err("a retake denoises once and cannot hold a 3840px axis");
        assert!(err.contains("3840"), "got: {err}");
    }

    /// The other implicit selectors all resolve to refining pipelines, so they
    /// must not narrow the ceiling.
    #[test]
    fn implicit_refining_pipelines_keep_the_composed_ceiling() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.width = 3_840;
        req.height = 2_176;
        req.frames = Some(25);
        req.fps = Some(24);
        req.output_format = Some(OutputFormat::Mp4);

        let mut with_audio = req.clone();
        with_audio.audio_file_path = Some("/tmp/voice.wav".to_string());
        assert_eq!(
            ltx2_spatial_composition_for_request(&with_audio),
            Ltx2SpatialComposition::TiledTwoStage
        );

        let mut with_source = req.clone();
        with_source.source_video_path = Some("/tmp/clip.mp4".to_string());
        assert_eq!(
            ltx2_spatial_composition_for_request(&with_source),
            Ltx2SpatialComposition::TiledTwoStage
        );

        // An explicit pipeline still wins over every implicit selector.
        let mut explicit = with_source.clone();
        explicit.pipeline = Some(Ltx2PipelineMode::OneStage);
        assert_eq!(
            ltx2_spatial_composition_for_request(&explicit),
            Ltx2SpatialComposition::SinglePass
        );
    }

    /// A rung is the same rung in either orientation — the composition and its
    /// cost are identical under transposition.
    #[test]
    fn rungs_resolve_in_either_orientation() {
        assert_eq!(ltx2_output_rung(3_840, 2_112).map(|r| r.id), Some("4k-uhd"));
        assert_eq!(ltx2_output_rung(2_112, 3_840).map(|r| r.id), Some("4k-uhd"));
        assert_eq!(ltx2_output_rung(1_920, 1_088).map(|r| r.id), Some("1080p"));
        assert_eq!(ltx2_output_rung(1_234, 567), None);
    }

    /// An over-size rejection has to say what the user *can* have. The pixel
    /// ceiling alone says only what they cannot.
    #[test]
    fn an_oversize_rejection_names_the_largest_reachable_rung() {
        assert_eq!(
            largest_ltx2_rung_within(LTX2_MAX_AXIS_PIXELS).map(|rung| rung.id),
            Some("1080p"),
        );
        assert_eq!(
            largest_ltx2_rung_within(LTX2_COMPOSED_MAX_AXIS_PIXELS).map(|rung| rung.id),
            Some("4k-uhd"),
        );
        assert_eq!(largest_ltx2_rung_within(64), None);

        let err = validate_generation_dimensions(3_840, 2_112, Some("ltx2"))
            .expect_err("a single-pass render cannot reach 4K");
        assert!(err.contains("spatial upsampler"), "got: {err}");
        assert!(err.contains("1080p Full HD (1920x1088)"), "got: {err}");

        let err = validate_generation_dimensions_composed(
            4_160,
            2_176,
            Some("ltx2"),
            Ltx2SpatialComposition::TiledTwoStage,
        )
        .expect_err("past the composed ceiling");
        assert!(
            !err.contains("spatial upsampler"),
            "a composing render is already using it, got: {err}"
        );
        assert!(err.contains("4K UHD (3840x2112)"), "got: {err}");
    }

    /// The issue's named 9:16 shape.
    #[test]
    fn ltx2_offers_portrait_presets() {
        let presets = recommended_dimensions("ltx2");
        assert!(
            presets.contains(&(704, 1216)),
            "704x1216 portrait must be advertised, got: {presets:?}"
        );
        assert!(
            presets.iter().any(|(w, h)| h > w && w * h > 1_000_000),
            "a high-resolution portrait preset must be advertised, got: {presets:?}"
        );
    }

    /// The advertised cap must be requestable. A client that clamps to it and
    /// submits should not get a 422 for being off the `8n+1` grid.
    #[test]
    fn ltx2_grid_snapped_cap_is_actually_requestable() {
        for fps in [6, 12, 24, 30, 48, 60, 120] {
            let cap = ltx2_max_frames_on_grid_at_fps(fps);
            assert_eq!(
                (cap - 1) % 8,
                0,
                "the advertised cap at {fps} fps must sit on the 8n+1 grid"
            );
            assert!(cap <= ltx2_max_frames_at_fps(fps));

            let mut req = valid_req();
            req.model = "ltx-2-19b-distilled:fp8".to_string();
            req.width = 768;
            req.height = 512;
            req.output_format = Some(OutputFormat::Mp4);
            req.frames = Some(cap);
            req.fps = Some(fps);
            validate_generate_request_with_family(&req, Some("ltx2")).unwrap_or_else(|err| {
                panic!("the advertised cap {cap} at {fps} fps must validate, got: {err}")
            });
        }
        // The raw ceilings are off-grid in both directions, which is the bug.
        assert_eq!(ltx2_max_frames_at_fps(24), 484);
        assert_eq!(ltx2_max_frames_on_grid_at_fps(24), 481);
        assert_eq!(ltx2_max_frames_at_fps(48), LTX2_MAX_FRAMES_ABSOLUTE);
        assert_eq!(ltx2_max_frames_on_grid_at_fps(48), 601);
    }

    /// EXR output is only meaningful for the HDR adapter's LogC3 signal.
    /// Applying the inverse to an ordinary SDR render would write a
    /// wrongly-graded file that looks deliberate — worse than a rejection.
    #[test]
    fn exr_output_requires_the_hdr_adapter() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.hdr_exr_dir = Some("/tmp/shot_exr".to_string());

        let err = validate_generate_request_with_family(&req, Some("ltx2"))
            .expect_err("EXR without the HDR adapter must be rejected");
        assert!(err.contains("ic_lora_control=hdr"), "got: {err}");

        // With the adapter (and the pipeline it forces) it validates.
        req.ic_lora_control = Some("hdr".to_string());
        req.pipeline = Some(Ltx2PipelineMode::IcLora);
        req.source_video_path = Some("/tmp/reference.mp4".to_string());
        req.loras = Some(vec![LoraWeight {
            path: "/models/hdr.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        validate_generate_request_with_family(&req, Some("ltx2"))
            .expect("the HDR adapter makes EXR output valid");
    }

    /// The extend path re-renders through the chain-stage machinery, where a
    /// per-clip EXR sequence would misalign with the stitched timeline. The
    /// rejection must be direct, not an accident of the ic-lora ⇒
    /// source_video ⇒ extend-exclusive implication chain.
    #[test]
    fn exr_output_rejects_extend_directly() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.hdr_exr_dir = Some("/tmp/shot_exr".to_string());
        req.extend_video_path = Some("/tmp/base.mp4".to_string());

        let err = validate_generate_request_with_family(&req, Some("ltx2"))
            .expect_err("EXR + extend must be rejected");
        assert!(err.contains("extend_video"), "got: {err}");
    }

    #[test]
    fn exr_options_are_rejected_for_non_ltx2_families() {
        let mut req = valid_req();
        req.hdr_exr_dir = Some("/tmp/shot_exr".to_string());
        assert!(validate_generate_request_with_family(&req, Some("flux")).is_err());
    }

    #[test]
    fn exr_precision_without_an_output_directory_is_rejected() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.hdr_exr_full_float = true;
        let err = validate_generate_request_with_family(&req, Some("ltx2"))
            .expect_err("a precision knob with nothing to write is a mistake");
        assert!(err.contains("hdr_exr_dir"), "got: {err}");
    }

    /// Every other consumer resolves control ids through
    /// `normalize_control_id`, so this gate must accept the same spellings —
    /// otherwise `--ic-lora-control HDR` succeeds everywhere except here.
    #[test]
    fn exr_accepts_any_spelling_the_control_registry_accepts() {
        // Case and surrounding whitespace only. A trailing `_` is *not* an
        // alias: the normalizer maps `_` to `-`, so `hdr_` becomes `hdr-`,
        // which is not a registered id anywhere in the stack.
        for spelling in ["hdr", "HDR", " Hdr ", "\tHDR\n"] {
            let mut req = valid_req();
            req.model = "ltx-2.3-22b-distilled:fp8".to_string();
            req.output_format = Some(OutputFormat::Mp4);
            req.source_video_path = Some("/tmp/reference.mp4".to_string());
            req.pipeline = Some(Ltx2PipelineMode::IcLora);
            req.ic_lora_control = Some(spelling.to_string());
            req.hdr_exr_dir = Some("/tmp/shot_exr".to_string());
            let result = validate_generate_request_with_family(&req, Some("ltx2"));
            assert!(
                result.is_ok(),
                "spelling {spelling:?} must be accepted, got: {result:?}"
            );
        }
    }

    #[test]
    fn exr_still_rejects_a_different_control() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.source_video_path = Some("/tmp/reference.mp4".to_string());
        req.pipeline = Some(Ltx2PipelineMode::IcLora);
        req.ic_lora_control = Some("union".to_string());
        req.hdr_exr_dir = Some("/tmp/shot_exr".to_string());
        let err = validate_generate_request_with_family(&req, Some("ltx2"))
            .expect_err("only the HDR adapter produces a LogC3 signal");
        assert!(err.contains("ic_lora_control=hdr"), "got: {err}");
    }

    /// The gallery artifact is the tonemapped video, so the sidecar's location
    /// is only discoverable from saved metadata.
    #[test]
    fn saved_metadata_records_where_the_exr_sequence_went() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.ic_lora_control = Some("hdr".to_string());
        req.hdr_exr_dir = Some("/tmp/shot_exr".to_string());
        req.hdr_exr_full_float = true;

        let metadata = crate::OutputMetadata::from_generate_request(&req, 7, None, "test");
        assert_eq!(metadata.hdr_exr_dir.as_deref(), Some("/tmp/shot_exr"));
        assert!(metadata.hdr_exr_full_float);

        let round_tripped: crate::OutputMetadata =
            serde_json::from_str(&serde_json::to_string(&metadata).unwrap()).unwrap();
        assert_eq!(round_tripped.hdr_exr_dir.as_deref(), Some("/tmp/shot_exr"));
        assert!(round_tripped.hdr_exr_full_float);
    }

    /// An ordinary render must not gain the fields, so existing rows and
    /// older readers see exactly the JSON they saw before.
    #[test]
    fn a_non_hdr_render_serializes_no_exr_fields() {
        let metadata = crate::OutputMetadata::from_generate_request(&valid_req(), 7, None, "test");
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(!json.contains("hdr_exr"), "got: {json}");
    }

    fn valid_req() -> GenerateRequest {
        GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            prompt: "a red apple".to_string(),
            negative_prompt: None,
            model: "test-model".to_string(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 0.0,
            seed: Some(42),
            batch_size: 1,
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    #[test]
    fn generation_rejects_compliance_gated_model_identity_before_other_validation() {
        let mut req = valid_req();
        req.model = "hf:MiniMaxAI/MiniMax-H3".to_string();
        req.prompt.clear();

        let error = validate_generate_request_with_family(&req, None).unwrap_err();
        assert!(error.contains(crate::MINIMAX_H3_AUTHORIZATION_REQUIRED));
        assert!(!error.contains(&req.model));
    }

    #[test]
    fn generation_rejects_opaque_catalog_id_with_compliance_gated_family() {
        let mut req = valid_req();
        req.model = "cv:42".to_string();

        let error = validate_generate_request_with_family(&req, Some("minimax-h3")).unwrap_err();
        assert!(error.contains(crate::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    fn valid_h3_request(model: &str) -> GenerateRequest {
        let mut req = valid_req();
        req.model = model.to_string();
        req.width = crate::minimax_h3::DEFAULT_WIDTH;
        req.height = crate::minimax_h3::DEFAULT_HEIGHT;
        req.steps = crate::minimax_h3::DEFAULT_STEPS;
        req.frames = Some(crate::minimax_h3::MIN_FRAMES);
        req.fps = Some(crate::minimax_h3::FIXED_FPS);
        req.output_format = Some(OutputFormat::Mp4);
        req.enable_audio = Some(true);
        // H3 has no denoise-strength control. The wire field is non-optional
        // for legacy families, so activated H3 callers must send its neutral
        // value rather than inheriting the generic img2img default.
        req.strength = 1.0;
        req
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    #[test]
    fn private_h3_validation_bypasses_only_activation_for_exact_reviewed_models() {
        let req = valid_h3_request(crate::minimax_h3::FL2VA_COMFY);
        let public_error =
            validate_generate_request_with_family(&req, Some(crate::minimax_h3::FAMILY))
                .unwrap_err();
        assert!(public_error.contains(crate::MINIMAX_H3_AUTHORIZATION_REQUIRED));
        validate_h3_private_uat_request(&req).unwrap();

        let mut official = req;
        official.model = crate::minimax_h3::FL2VA_OFFICIAL.to_string();
        assert!(validate_h3_private_uat_request(&official)
            .unwrap_err()
            .contains("exact reviewed task model"));
    }

    #[test]
    fn h3_post_activation_fl2va_accepts_first_and_last_boundary_frames() {
        let mut req = valid_h3_request(crate::minimax_h3::FL2VA_COMFY);
        req.source_image = Some(png_bytes());
        req.keyframes = Some(vec![crate::KeyframeCondition {
            frame: crate::minimax_h3::MIN_FRAMES - 1,
            image: jpeg_bytes(),
            name: None,
        }]);

        assert!(
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY),)
                .is_ok()
        );
    }

    #[test]
    fn h3_post_activation_ref2va_accepts_image_references() {
        let mut req = valid_h3_request(crate::minimax_h3::REF2VA_COMFY);
        req.references = Some(vec![crate::GenerationReference::Image {
            media: crate::GenerationReferenceAuthority::Inline { data: png_bytes() },
            provenance: crate::GenerationReferenceProvenance {
                name: Some("reference.png".to_string()),
                sha256: None,
            },
            mime_type: "image/png".to_string(),
            width: 1920,
            height: 1080,
        }]);

        assert!(
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY),)
                .is_ok()
        );
    }

    #[test]
    fn h3_post_activation_rejects_non_boundary_keyframes() {
        let mut req = valid_h3_request(crate::minimax_h3::FL2VA_COMFY);
        req.keyframes = Some(vec![crate::KeyframeCondition {
            frame: 17,
            image: png_bytes(),
            name: None,
        }]);

        let error =
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY))
                .unwrap_err();
        assert!(
            error.contains("only frame 0 or final frame"),
            "got: {error}"
        );
    }

    #[test]
    fn h3_post_activation_rejects_generic_scheduler_and_lora_overrides() {
        let mut req = valid_h3_request(crate::minimax_h3::FL2VA_COMFY);
        req.scheduler = Some(crate::Scheduler::UniPc);
        let error =
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY))
                .unwrap_err();
        assert!(error.contains("scheduler overrides"), "got: {error}");

        req.scheduler = None;
        req.lora = Some(crate::LoraWeight {
            path: "/tmp/adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        let error =
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY))
                .unwrap_err();
        assert!(error.contains("does not support LoRA"), "got: {error}");

        req.lora = None;
        req.loras = Some(Vec::new());
        let error =
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY))
                .unwrap_err();
        assert!(error.contains("does not support LoRA"), "got: {error}");
    }

    #[test]
    fn h3_post_activation_preserves_source_and_extend_invariants() {
        for strength in [-1.0, 1.01, f64::NAN] {
            let mut req = valid_h3_request(crate::minimax_h3::FL2VA_COMFY);
            req.source_image = Some(png_bytes());
            req.strength = strength;
            let error =
                validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY))
                    .unwrap_err();
            assert!(error.contains("finite value in range"), "got: {error}");
        }

        let mut req = valid_h3_request(crate::minimax_h3::FL2VA_COMFY);
        req.extend_overlap_frames = Some(9);
        let error =
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY))
                .unwrap_err();
        assert!(
            error.contains("extend_overlap_frames requires extend_video"),
            "got: {error}"
        );
    }

    #[test]
    fn h3_post_activation_rejects_empty_conditioning_collections() {
        let mut req = valid_h3_request(crate::minimax_h3::FL2VA_COMFY);
        req.edit_images = Some(Vec::new());
        let error =
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY))
                .unwrap_err();
        assert!(
            error.contains("edit_images must not be empty"),
            "got: {error}"
        );

        req.edit_images = None;
        req.keyframes = Some(Vec::new());
        let error =
            validate_generate_request_after_activation(&req, Some(crate::minimax_h3::FAMILY))
                .unwrap_err();
        assert!(
            error.contains("keyframes must not be empty"),
            "got: {error}"
        );
    }

    #[test]
    fn generation_model_preflight_gates_nested_identities_and_artifacts() {
        let root = std::path::Path::new("/Volumes/ExternalStorage/mold-uat/minimax-h3/models");

        let mut req = valid_req();
        req.control_model = Some("MiniMax-H3-FL2VA".to_string());
        assert!(require_generate_request_model_activation(&req, Some(root), Some("flux")).is_err());

        req.control_model = None;
        req.upscale_model = Some("hf:MiniMaxAI/MiniMax-H3".to_string());
        assert!(require_generate_request_model_activation(&req, Some(root), Some("flux")).is_err());

        req.upscale_model = None;
        req.lora = Some(crate::LoraWeight {
            path: root
                .join("custom/MiniMax-H3/adapter.safetensors")
                .to_string_lossy()
                .into_owned(),
            scale: 1.0,

            expert: None,
        });
        assert!(require_generate_request_model_activation(&req, Some(root), Some("flux")).is_err());

        req.lora.as_mut().unwrap().path = root
            .join("flux/ordinary-adapter.safetensors")
            .to_string_lossy()
            .into_owned();
        assert!(require_generate_request_model_activation(&req, Some(root), Some("flux")).is_ok());
    }

    /// Minimal valid PNG header bytes for testing.
    fn png_bytes() -> Vec<u8> {
        vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    }

    /// Minimal valid JPEG header bytes for testing.
    fn jpeg_bytes() -> Vec<u8> {
        vec![0xFF, 0xD8, 0xFF, 0xE0]
    }

    // ── clamp_to_megapixel_limit tests ──────────────────────────────────────

    #[test]
    fn clamp_noop_within_limit() {
        assert_eq!(super::clamp_to_megapixel_limit(1024, 1024), (1024, 1024));
    }

    #[test]
    fn clamp_noop_qwen_image_native_resolution() {
        // Qwen-Image trains at 1328x1328 (~1.76MP), must fit within MAX_PIXELS
        assert_eq!(super::clamp_to_megapixel_limit(1328, 1328), (1328, 1328));
    }

    #[test]
    fn clamp_noop_qwen_image_landscape() {
        // Qwen-Image 16:9 training resolution (1664x928 = ~1.54MP)
        assert_eq!(super::clamp_to_megapixel_limit(1664, 928), (1664, 928));
    }

    #[test]
    fn clamp_downscales_oversized() {
        let (w, h) = super::clamp_to_megapixel_limit(1888, 1168);
        assert!(w % 16 == 0 && h % 16 == 0, "must be multiples of 16");
        let pixels = w as u64 * h as u64;
        assert!(
            pixels <= super::MAX_PIXELS,
            "must be within limit: {pixels}"
        );
        // Aspect ratio roughly preserved
        let orig_ratio = 1888.0 / 1168.0;
        let new_ratio = w as f64 / h as f64;
        assert!(
            (orig_ratio - new_ratio).abs() < 0.05,
            "aspect ratio drift too large"
        );
    }

    #[test]
    fn clamp_large_square() {
        let (w, h) = super::clamp_to_megapixel_limit(2048, 2048);
        assert!(w % 16 == 0 && h % 16 == 0);
        assert!(w as u64 * h as u64 <= super::MAX_PIXELS);
    }

    #[test]
    fn clamp_extreme_aspect_ratio() {
        let (w, h) = super::clamp_to_megapixel_limit(4096, 256);
        assert!(w % 16 == 0 && h % 16 == 0);
        assert!(w as u64 * h as u64 <= super::MAX_PIXELS);
        assert!(w > h, "should remain landscape");
    }

    // ── normalise_output_format tests ────────────────────────────────────────

    /// The shared Wan surface-parity fixture (#806) pins the family policy the
    /// server default and the CLI's client-side default both derive from, plus
    /// the frame grid and the TI2V first/last-frame floor every surface
    /// enforces before dispatch. Editing one side without the other fails here.
    #[test]
    fn wan_surface_parity_fixture_pins_the_core_contracts() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/wan/surface-parity-v1.json"
        )))
        .expect("fixture parses");

        // Container policy: exactly the fixture's family set defaults to MP4.
        let mp4_families: Vec<&str> = fixture["container_default"]["mp4_default_families"]
            .as_array()
            .expect("mp4_default_families")
            .iter()
            .map(|value| value.as_str().expect("family string"))
            .collect();
        for family in &mp4_families {
            assert!(
                crate::family_output_defaults_to_mp4(family),
                "{family} must default to mp4"
            );
        }
        for family in ["flux", "sdxl", "qwen-image", "z-image", ""] {
            assert!(
                !crate::family_output_defaults_to_mp4(family),
                "{family:?} must not default to mp4"
            );
        }

        // Server normalisation: unset wan multi-frame → mp4; frames == 1 → png.
        let mut req = valid_req();
        req.model = "wan22-t2v-a14b:q8".to_string();
        req.frames = Some(81);
        req.output_format = None;
        req.normalise_output_format(Some(fixture["family"].as_str().unwrap()));
        assert_eq!(
            format!("{:?}", req.resolved_output_format()).to_lowercase(),
            fixture["container_default"]["unset_multi_frame"]
                .as_str()
                .unwrap()
        );
        let mut still = valid_req();
        still.model = "wan22-t2v-a14b:q8".to_string();
        still.frames = Some(1);
        still.output_format = None;
        still.normalise_output_format(Some("wan"));
        assert_eq!(
            format!("{:?}", still.resolved_output_format()).to_lowercase(),
            fixture["container_default"]["wan_single_frame"]
                .as_str()
                .unwrap()
        );

        // Frame grid.
        assert_eq!(
            u64::from(WAN_TEMPORAL_SCALE),
            fixture["frame_grid"]["step"].as_u64().unwrap()
        );
        assert_eq!(
            u64::from(frame_offset_for_family("wan").expect("wan has a grid")),
            fixture["frame_grid"]["offset"].as_u64().unwrap()
        );

        // TI2V first/last-frame floor.
        assert_eq!(
            u64::from(WAN_TI2V_FLF_MIN_FRAMES),
            fixture["first_last_frame"]["ti2v_min_frames"]
                .as_u64()
                .unwrap()
        );
        assert!(fixture["first_last_frame"]["ti2v_model_prefix"]
            .as_str()
            .unwrap()
            .starts_with("wan22-ti2v-5b"));
    }

    #[test]
    fn normalise_output_format_unset_for_ltx2_picks_mp4() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = None;
        req.normalise_output_format(Some("ltx2"));
        assert_eq!(
            req.resolved_output_format(),
            OutputFormat::Mp4,
            "ltx2 with no explicit format should default to mp4"
        );
    }

    #[test]
    fn normalise_output_format_unset_for_ltx2_with_audio_picks_mp4() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = None;
        req.enable_audio = Some(true);
        req.normalise_output_format(Some("ltx2"));
        assert_eq!(
            req.resolved_output_format(),
            OutputFormat::Mp4,
            "ltx2 with audio and no explicit format should default to mp4"
        );
    }

    #[test]
    fn normalise_output_format_unset_for_ltx_video_picks_mp4() {
        let mut req = valid_req();
        req.model = "ltx-video:fp16".to_string();
        req.output_format = None;
        req.normalise_output_format(Some("ltx-video"));
        assert_eq!(
            req.resolved_output_format(),
            OutputFormat::Mp4,
            "ltx-video with no explicit format should default to mp4"
        );
    }

    #[test]
    fn normalise_output_format_unset_for_flux_picks_png() {
        let mut req = valid_req();
        req.model = "flux-schnell:q8".to_string();
        req.output_format = None;
        req.normalise_output_format(Some("flux"));
        assert_eq!(
            req.resolved_output_format(),
            OutputFormat::Png,
            "flux with no explicit format should default to png"
        );
    }

    #[test]
    fn normalise_output_format_explicit_png_for_ltx2_remains_png_and_validation_rejects_it() {
        // When the user explicitly requests PNG for an ltx2 model, normalise
        // must leave it as-is so validation can reject it with a clear error.
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Png);
        req.normalise_output_format(Some("ltx2"));
        // normalise must not touch an explicit value
        assert_eq!(req.output_format, Some(OutputFormat::Png));
        // and validation must still reject explicit PNG on ltx2
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("LTX-2 outputs must use"),
            "expected validation error for explicit png on ltx2, got: {err}"
        );
    }

    // ── validate_generate_request tests ──────────────────────────────────────

    #[test]
    fn valid_request_passes() {
        assert!(validate_generate_request(&valid_req()).is_ok());
    }

    #[test]
    fn ltx2_audio_requires_mp4() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Gif);
        req.enable_audio = Some(true);
        assert!(validate_generate_request(&req).unwrap_err().contains("mp4"));
    }

    /// A T2A request produces a WAV and only a WAV. Both directions of the
    /// pairing are enforced: `t2a` without `wav` would encode frames that
    /// don't exist, and `wav` without `t2a` would ask a video pipeline for a
    /// container it never writes.
    #[test]
    fn ltx2_t2a_requires_wav_output_and_wav_requires_t2a() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-dev:fp8".to_string();
        req.pipeline = Some(Ltx2PipelineMode::T2a);
        req.output_format = Some(OutputFormat::Wav);
        req.width = 0;
        req.height = 0;
        assert!(validate_generate_request(&req).is_ok());

        req.output_format = Some(OutputFormat::Mp4);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("audio only"), "got: {err}");

        req.pipeline = None;
        req.output_format = Some(OutputFormat::Wav);
        req.width = 1024;
        req.height = 1024;
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("pipeline=t2a"), "got: {err}");
    }

    #[test]
    fn ltx2_t2a_is_dimensionless_and_ignores_a_legacy_raster_canvas() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-dev:fp8".to_string();
        req.pipeline = Some(Ltx2PipelineMode::T2a);
        req.output_format = Some(OutputFormat::Wav);
        req.width = 0;
        req.height = 0;
        validate_generate_request(&req).unwrap();

        // One-release compatibility: older clients still serialize their
        // inactive raster fields. They remain irrelevant to T2A admission.
        req.width = 1024;
        req.height = 576;
        validate_generate_request(&req).unwrap();
    }

    #[test]
    fn ltx2_t2a_rejects_every_conditioning_input() {
        let base = || {
            let mut req = valid_req();
            req.model = "ltx-2.3-22b-dev:fp8".to_string();
            req.pipeline = Some(Ltx2PipelineMode::T2a);
            req.output_format = Some(OutputFormat::Wav);
            req.width = 0;
            req.height = 0;
            req
        };

        let mut with_image = base();
        with_image.source_image = Some(vec![1, 2, 3]);
        assert!(validate_generate_request(&with_image)
            .unwrap_err()
            .contains("source_image"));

        let mut with_audio = base();
        with_audio.audio_file_path = Some("/srv/voice.wav".to_string());
        assert!(validate_generate_request(&with_audio)
            .unwrap_err()
            .contains("audio_file_path"));

        let mut with_upscale = base();
        with_upscale.spatial_upscale = Some(crate::Ltx2SpatialUpscale::X2);
        assert!(validate_generate_request(&with_upscale)
            .unwrap_err()
            .contains("spatial_upscale"));

        let mut with_post_upscale = base();
        with_post_upscale.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        assert!(validate_generate_request(&with_post_upscale)
            .unwrap_err()
            .contains("upscale_model"));
    }

    /// ControlNet is refused for `t2a` by the family gate, not by the
    /// pipeline's own conditioning list. A ControlNet pair requires an SD1.5
    /// family and `pipeline` requires `ltx2`, so the two can never both be
    /// satisfied — the audio-only runtime cannot be reached with a control
    /// model loaded. Pinned here because the t2a rejection list reads as
    /// though it were the only guard, and a future refactor that relaxed
    /// `require_controlnet_capable_family` would silently open that door.
    #[test]
    fn ltx2_t2a_cannot_carry_controlnet_inputs() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-dev:fp8".to_string();
        req.pipeline = Some(Ltx2PipelineMode::T2a);
        req.output_format = Some(OutputFormat::Wav);
        req.width = 0;
        req.height = 0;
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        req.control_scale = 0.8;

        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("ControlNet"), "got: {err}");

        // And the mirror case: a control model without an image is refused on
        // the same family grounds rather than reaching the audio pipeline.
        req.control_image = None;
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("ControlNet"), "got: {err}");
    }

    #[test]
    fn ltx2_t2a_rejects_enable_audio_false() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-dev:fp8".to_string();
        req.pipeline = Some(Ltx2PipelineMode::T2a);
        req.output_format = Some(OutputFormat::Wav);
        req.width = 0;
        req.height = 0;
        req.enable_audio = Some(false);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("enable_audio=false"), "got: {err}");
    }

    /// `modality_scale` steers the audio↔video cross-attention. Audio-only has
    /// no video branch, so a non-1.0 value cannot be honoured — reject it
    /// rather than accept a number that silently does nothing.
    #[test]
    fn ltx2_t2a_rejects_non_unit_modality_scale_override() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-dev:fp8".to_string();
        req.pipeline = Some(Ltx2PipelineMode::T2a);
        req.output_format = Some(OutputFormat::Wav);
        req.width = 0;
        req.height = 0;
        req.guidance_overrides = Some(crate::Ltx2GuidanceOverrides {
            modality_scale: Some(3.0),
            ..Default::default()
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("modality_scale"), "got: {err}");

        req.guidance_overrides = Some(crate::Ltx2GuidanceOverrides {
            modality_scale: Some(1.0),
            ..Default::default()
        });
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn ltx2_retake_requires_source_video() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.retake_range = Some(crate::TimeRange {
            start_seconds: 0.0,
            end_seconds: 1.0,
        });
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("source_video"));
    }

    #[test]
    fn ltx2_audio_file_rejects_inline_payloads_above_limit() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.audio_file = Some(vec![0; MAX_INLINE_AUDIO_BYTES + 1]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("audio_file exceeds"), "got: {err}");
        assert!(err.contains("64 MiB"), "got: {err}");
    }

    #[test]
    fn ltx2_source_video_rejects_inline_payloads_above_limit() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.source_video = Some(vec![0; MAX_INLINE_SOURCE_VIDEO_BYTES + 1]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("source_video exceeds"), "got: {err}");
        assert!(err.contains("64 MiB"), "got: {err}");
    }

    #[test]
    fn ltx2_audio_file_path_is_family_gated_and_preserves_inline_limit() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.audio_file_path = Some("/srv/mold-media/voice.wav".to_string());
        assert!(validate_generate_request(&req).is_ok());

        req.audio_file = Some(vec![0; MAX_INLINE_AUDIO_BYTES + 1]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("audio_file_path cannot be combined"),
            "got: {err}"
        );

        let mut wrong_family = valid_req();
        wrong_family.model = "flux-schnell:q8".to_string();
        wrong_family.audio_file_path = Some("/srv/mold-media/voice.wav".to_string());
        let err = validate_generate_request(&wrong_family).unwrap_err();
        assert!(
            err.contains("audio_file_path is only supported"),
            "got: {err}"
        );
    }

    #[test]
    fn ltx2_source_video_path_satisfies_retake_requirements() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.source_video_path = Some("/srv/mold-media/clip.mp4".to_string());
        req.retake_range = Some(crate::TimeRange {
            start_seconds: 0.0,
            end_seconds: 1.0,
        });

        assert!(validate_generate_request(&req).is_ok());

        req.source_video = Some(vec![0; MAX_INLINE_SOURCE_VIDEO_BYTES + 1]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("source_video_path cannot be combined"),
            "got: {err}"
        );
    }

    #[test]
    fn ltx2_keyframe_pipeline_requires_multiple_keyframes() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.pipeline = Some(crate::Ltx2PipelineMode::Keyframe);
        req.frames = Some(17);
        req.keyframes = Some(vec![crate::KeyframeCondition {
            frame: 0,
            image: png_bytes(),
            name: None,
        }]);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("at least 2 keyframes"));
    }

    #[test]
    fn keyframes_on_unknown_family_report_unknown_model_family() {
        let mut req = valid_req();
        req.model = "private-ltx2-style-model".to_string();
        req.frames = Some(17);
        req.keyframes = Some(vec![
            crate::KeyframeCondition {
                frame: 0,
                image: png_bytes(),
                name: None,
            },
            crate::KeyframeCondition {
                frame: 16,
                image: png_bytes(),
                name: None,
            },
        ]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("unknown model family"), "got: {err}");
    }

    fn ltx2_req_with_overrides(overrides: Ltx2GuidanceOverrides) -> GenerateRequest {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.frames = Some(17);
        req.guidance_overrides = Some(overrides);
        req
    }

    #[test]
    fn ltx2_guidance_overrides_accept_upstream_ranges() {
        validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            stg_scale: Some(1.5),
            stg_blocks: Some(vec![28, 29]),
            rescale_scale: Some(0.7),
            modality_scale: Some(3.0),
            skip_step: Some(2),
        }))
        .unwrap();
    }

    #[test]
    fn ltx2_guidance_overrides_are_family_gated() {
        let mut req = valid_req();
        req.guidance_overrides = Some(Ltx2GuidanceOverrides {
            stg_scale: Some(1.0),
            ..Ltx2GuidanceOverrides::default()
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("guidance_overrides"), "got: {err}");
        assert!(err.contains("LTX-2"), "got: {err}");
    }

    #[test]
    fn ltx2_guidance_overrides_reject_empty_objects() {
        let err =
            validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides::default()))
                .unwrap_err();
        assert!(err.contains("at least one field"), "got: {err}");
    }

    #[test]
    fn ltx2_guidance_overrides_reject_out_of_range_scales() {
        let err = validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            stg_scale: Some(-0.5),
            ..Ltx2GuidanceOverrides::default()
        }))
        .unwrap_err();
        assert!(err.contains("stg_scale"), "got: {err}");

        let err = validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            stg_scale: Some(f64::NAN),
            ..Ltx2GuidanceOverrides::default()
        }))
        .unwrap_err();
        assert!(err.contains("finite"), "got: {err}");

        // Rescale is an interpolation factor, so its ceiling is 1.0 even
        // though the other scales accept much larger values.
        let err = validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            rescale_scale: Some(1.5),
            ..Ltx2GuidanceOverrides::default()
        }))
        .unwrap_err();
        assert!(err.contains("rescale_scale"), "got: {err}");
        validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            modality_scale: Some(1.5),
            ..Ltx2GuidanceOverrides::default()
        }))
        .unwrap();
    }

    #[test]
    fn ltx2_guidance_overrides_reject_unusable_stg_blocks() {
        let err = validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            stg_blocks: Some(Vec::new()),
            ..Ltx2GuidanceOverrides::default()
        }))
        .unwrap_err();
        assert!(err.contains("must not be empty"), "got: {err}");

        let err = validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            stg_blocks: Some(vec![MAX_STG_BLOCK_INDEX]),
            ..Ltx2GuidanceOverrides::default()
        }))
        .unwrap_err();
        assert!(err.contains("deepest supported"), "got: {err}");

        let err = validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            stg_blocks: Some(vec![29, 29]),
            ..Ltx2GuidanceOverrides::default()
        }))
        .unwrap_err();
        assert!(err.contains("more than once"), "got: {err}");
    }

    #[test]
    fn ltx2_guidance_overrides_bound_the_skip_stride() {
        let err = validate_generate_request(&ltx2_req_with_overrides(Ltx2GuidanceOverrides {
            skip_step: Some(Ltx2GuidanceOverrides::MAX_SKIP_STEP + 1),
            ..Ltx2GuidanceOverrides::default()
        }))
        .unwrap_err();
        assert!(err.contains("skip_step"), "got: {err}");
    }

    #[test]
    fn enable_audio_some_false_does_not_trip_family_check() {
        // Web form serializes the audio toggle as `Some(false)` whenever the
        // checkbox is explicitly off. That must be a no-op for any family
        // (including the unknown-family case used by catalog `cv:*` IDs)
        // since the user did not ask for audio.
        let mut req = valid_req();
        req.model = "cv:2781713".to_string();
        req.enable_audio = Some(false);
        // No family hint provided — exercises the unknown-family branch.
        validate_generate_request(&req).unwrap();
    }

    #[test]
    fn enable_audio_some_true_with_family_hint_passes_for_catalog_ltx2() {
        // The HTTP server resolves `cv:*` IDs against the catalog DB and
        // passes the family through as a hint. With the LTX-2 hint, audio
        // is allowed even though the manifest layer has no entry for the
        // catalog ID.
        let mut req = valid_req();
        req.model = "cv:2781713".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.enable_audio = Some(true);
        validate_generate_request_with_family(&req, Some("ltx2")).unwrap();
    }

    #[test]
    fn enable_audio_some_true_without_hint_still_errors_on_unknown_family() {
        // No hint, no manifest entry — the family gate still fires so the
        // user gets a clear 400 instead of an opaque inference-layer error.
        let mut req = valid_req();
        req.model = "cv:2781713".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.enable_audio = Some(true);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("unknown model family"), "got: {err}");
        assert!(err.contains("enable_audio"), "got: {err}");
    }

    #[test]
    fn family_hint_overrides_manifest_lookup() {
        // Even when the manifest would resolve the model name to a different
        // family, the explicit hint wins. This lets the server pass the
        // catalog-resolved family through unconditionally.
        let mut req = valid_req();
        req.model = "private-name".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.enable_audio = Some(true);
        validate_generate_request_with_family(&req, Some("ltx2")).unwrap();
    }

    #[test]
    fn ltx2_allows_temporal_upscale_request() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.temporal_upscale = Some(crate::Ltx2TemporalUpscale::X2);
        validate_generate_request(&req).unwrap();
    }

    #[test]
    fn ltx2_allows_x1_5_spatial_upscale_request() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.spatial_upscale = Some(crate::Ltx2SpatialUpscale::X1_5);
        validate_generate_request(&req).unwrap();
    }

    #[test]
    fn empty_prompt_rejected() {
        // The default `valid_req()` is a text-to-image request with no visual
        // conditioning, so the prompt stays mandatory.
        let mut req = valid_req();
        req.prompt = "   ".to_string();
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("prompt"));
    }

    /// Baseline LTX-2 video request with no visual conditioning attached.
    fn ltx2_video_req() -> GenerateRequest {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(24);
        req.frames = Some(97);
        req
    }

    #[test]
    fn empty_prompt_allowed_for_ltx2_with_source_image() {
        let mut req = ltx2_video_req();
        req.prompt = String::new();
        req.source_image = Some(png_bytes());
        validate_generate_request(&req).unwrap();

        // Whitespace-only is the same case as empty.
        req.prompt = "  \n ".to_string();
        validate_generate_request(&req).unwrap();

        // Catalog IDs only resolve to `ltx2` through the family hint.
        let mut catalog = req.clone();
        catalog.model = "cv:2781713".to_string();
        assert!(validate_generate_request(&catalog).is_err());
        validate_generate_request_with_family(&catalog, Some("ltx2")).unwrap();
    }

    #[test]
    fn empty_prompt_allowed_for_ltx2_keyframes_video_and_extend() {
        let mut keyframed = ltx2_video_req();
        keyframed.prompt = String::new();
        keyframed.keyframes = Some(vec![KeyframeCondition {
            frame: 0,
            image: png_bytes(),
            name: None,
        }]);
        validate_generate_request(&keyframed).unwrap();

        let mut from_video = ltx2_video_req();
        from_video.prompt = String::new();
        from_video.source_video = Some(vec![0, 0, 0, 0x20, b'f', b't', b'y', b'p']);
        validate_generate_request(&from_video).unwrap();

        // The server validates before `resolve_server_local_media_paths`, so
        // the `*_path` variants must count as conditioning too.
        let mut from_video_path = ltx2_video_req();
        from_video_path.prompt = String::new();
        from_video_path.source_video_path = Some("/srv/clips/shot.mp4".to_string());
        validate_generate_request(&from_video_path).unwrap();

        let mut extended = extend_req();
        extended.prompt = String::new();
        validate_generate_request(&extended).unwrap();

        let mut extended_path = ltx2_video_req();
        extended_path.prompt = String::new();
        extended_path.extend_video_path = Some("/srv/clips/shot.mp4".to_string());
        validate_generate_request(&extended_path).unwrap();
    }

    #[test]
    fn empty_prompt_allowed_for_ltx_video_with_source_image() {
        let mut req = valid_req();
        req.model = "ltx-video-0.9.8-2b-distilled:bf16".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.prompt = String::new();
        req.source_image = Some(png_bytes());
        validate_generate_request(&req).unwrap();
    }

    #[test]
    fn empty_prompt_still_rejected_for_ltx2_text_to_video() {
        let mut req = ltx2_video_req();
        req.prompt = String::new();
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("prompt"));
    }

    #[test]
    fn empty_prompt_still_rejected_for_flux_and_sd() {
        // Image families keep the prompt required even with a source image —
        // an empty img2img prompt is not a trained context there.
        for model in [
            "flux-dev:q8",
            "sd15:fp16",
            "sdxl:fp16",
            "z-image-turbo:bf16",
        ] {
            let mut req = valid_req();
            req.model = model.to_string();
            req.prompt = String::new();
            req.source_image = Some(png_bytes());
            assert!(
                validate_generate_request(&req)
                    .unwrap_err()
                    .contains("prompt"),
                "{model} must still require a prompt"
            );
        }
    }

    #[test]
    fn prompt_required_predicate_matches_validation() {
        let mut req = ltx2_video_req();
        assert!(super::prompt_required_for(&req, None));
        req.source_image = Some(png_bytes());
        assert!(!super::prompt_required_for(&req, None));

        // Unknown family (catalog ID without a hint) stays required.
        let mut catalog = req.clone();
        catalog.model = "hf:Lightricks/LTX-2".to_string();
        assert!(super::prompt_required_for(&catalog, None));
        assert!(!super::prompt_required_for(&catalog, Some("ltx2")));
    }

    #[test]
    fn prompt_length_limit_still_enforced_without_a_prompt_requirement() {
        let mut req = ltx2_video_req();
        req.source_image = Some(png_bytes());
        req.prompt = "a".repeat(77_001);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("77,000"));
    }

    #[test]
    fn zero_dimensions_rejected() {
        let mut req = valid_req();
        req.width = 0;
        assert!(validate_generate_request(&req).is_err());
        req.width = 1024;
        req.height = 0;
        assert!(validate_generate_request(&req).is_err());
    }

    #[test]
    fn dimensions_must_be_multiple_of_16() {
        let mut req = valid_req();
        req.width = 513; // not multiple of 16
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("multiples of 16"));
    }

    #[test]
    fn ltx2_dimensions_must_be_multiple_of_32() {
        let mut req = valid_req();
        req.width = 1008; // multiple of 16, but not 32
        req.height = 704;

        let error = validate_generate_request_with_family(&req, Some("ltx2"))
            .expect_err("LTX-2 must reject a 16px-only canvas");

        assert!(error.contains("multiples of 32"), "{error}");
        assert!(error.contains("ltx2"), "{error}");
    }

    #[test]
    fn ltx2_accepts_custom_32_aligned_dimensions() {
        let mut req = valid_req();
        req.width = 1056;
        req.height = 736;
        req.output_format = Some(OutputFormat::Mp4);

        assert!(validate_generate_request_with_family(&req, Some("ltx2")).is_ok());
    }

    #[test]
    fn valid_non_square_dimensions() {
        let mut req = valid_req();
        req.width = 512;
        req.height = 768;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn oversized_image_rejected() {
        let mut req = valid_req();
        req.width = 1408;
        req.height = 1408; // ~1.98MP > 1.8MP limit
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("megapixels"));
    }

    #[test]
    fn oversized_image_error_reports_current_megapixel_limit() {
        let mut req = valid_req();
        req.width = 1408;
        req.height = 1408;
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("1.8MP"), "got: {err}");
    }

    #[test]
    fn zero_steps_rejected() {
        let mut req = valid_req();
        req.steps = 0;
        assert!(validate_generate_request(&req).is_err());
    }

    #[test]
    fn excessive_steps_rejected() {
        let mut req = valid_req();
        req.steps = 101;
        assert!(validate_generate_request(&req).is_err());
    }

    #[test]
    fn valid_step_counts() {
        for steps in [1, 4, 20, 28, 50, 100] {
            let mut req = valid_req();
            req.steps = steps;
            assert!(
                validate_generate_request(&req).is_ok(),
                "steps={steps} should be valid"
            );
        }
    }

    #[test]
    fn ltx2_frames_must_still_follow_8n_plus_1() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.frames = Some(10);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("8n+1"), "got: {err}");
        // The message derives its examples from the family's own step so it
        // stays correct now that more than one grid exists (LTX 8, Wan 4).
        assert!(err.contains("9, 17, 25"), "got: {err}");
    }

    fn extend_req() -> GenerateRequest {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(24);
        req.frames = Some(97);
        req.extend_video = Some(vec![0, 0, 0, 0x20, b'f', b't', b'y', b'p']);
        req
    }

    #[test]
    fn extend_accepts_a_video_with_the_default_overlap() {
        let req = extend_req();
        assert!(validate_generate_request(&req).is_ok());
        assert!(req.is_extend());
        assert_eq!(
            req.effective_extend_overlap_frames(),
            DEFAULT_EXTEND_OVERLAP_FRAMES
        );
        // 97 rendered frames minus the 17-frame overlap that reproduces the
        // source tail = 80 genuinely new frames appended.
        assert_eq!(req.extend_new_frames(), Some(80));
    }

    /// An extend carries its source frames in the clip it continues (#783).
    ///
    /// `validate_extend` forbids pairing `extend_video` with `source_image` or
    /// keyframes, so a gate that counted only those two saw *every*
    /// continuation as source-less — and refused every Wan I2V extend with
    /// "this Wan I2V checkpoint needs a source image", the very contract that
    /// makes the checkpoint extend-capable in the first place.
    #[test]
    fn extend_carries_the_source_frames_the_contract_gate_looks_for() {
        use crate::types::SourceImageCapability;

        let mut req = extend_req();
        req.model = "wan22-i2v-a14b:q8".to_string();
        req.width = 832;
        req.height = 480;
        req.fps = Some(16);
        req.frames = Some(49);
        assert!(req.source_image.is_none() && req.keyframes.is_none());
        assert!(request_carries_source_frames(&req));

        // Required + extend is satisfied: admission must let it through.
        assert_eq!(
            source_image_contract_violation(
                Some("wan"),
                &req.model,
                Some(SourceImageCapability::Required),
                request_carries_source_frames(&req),
            ),
            None
        );
        // …and a text-to-video checkpoint is refused at admission instead of
        // dying in the engine after the UMT5 encode and expert load are paid.
        assert!(source_image_contract_violation(
            Some("wan"),
            "wan22-t2v-a14b:q8",
            Some(SourceImageCapability::Unsupported),
            request_carries_source_frames(&req),
        )
        .is_some());

        // An ordinary render still carries nothing.
        let plain = valid_req();
        assert!(!request_carries_source_frames(&plain));
    }

    /// The overlap default is a property of the family's carryover, not a
    /// global scalar (#783): wan's handoff is one frame and its engine refuses
    /// anything else, so advertising LTX-2's 17 handed every wan client a
    /// value that clears wan's `4k+1` grid check and then fails in the engine.
    #[test]
    fn extend_overlap_default_follows_the_familys_own_carryover() {
        assert_eq!(
            default_extend_overlap_frames_for_family(Some("wan")),
            WAN_HANDOFF_DUPLICATED_FRAMES
        );
        assert_eq!(WAN_HANDOFF_DUPLICATED_FRAMES, 1);
        assert_eq!(
            default_extend_overlap_frames_for_family(Some("ltx2")),
            DEFAULT_EXTEND_OVERLAP_FRAMES
        );
        // An unresolved family keeps the historical scalar.
        assert_eq!(
            default_extend_overlap_frames_for_family(None),
            DEFAULT_EXTEND_OVERLAP_FRAMES
        );

        let mut req = extend_req();
        req.model = "wan22-ti2v-5b:fp16".to_string();
        req.width = 704;
        req.height = 384;
        req.fps = Some(24);
        req.frames = Some(49);
        assert_eq!(
            req.effective_extend_overlap_frames_for_family(Some("wan")),
            WAN_HANDOFF_DUPLICATED_FRAMES
        );
        // With no hint the request resolves its own family from the manifest,
        // so metadata provenance records what the engine actually applied.
        assert_eq!(
            req.effective_extend_overlap_frames(),
            WAN_HANDOFF_DUPLICATED_FRAMES
        );
        assert_eq!(req.extend_new_frames(), Some(48));
        assert!(validate_generate_request(&req).is_ok());

        // An explicit value is never overridden — validation still owns it.
        req.extend_overlap_frames = Some(9);
        assert_eq!(
            req.effective_extend_overlap_frames_for_family(Some("wan")),
            9
        );
    }

    /// A chain seam's carryover is the family's, never the caller's (#783).
    ///
    /// The server has normalized this since #936, but only the server: a
    /// forced-local `--script` run, `mold chain validate`, and `--dry-run`
    /// all called the family-generic `normalise()` and passed the requested
    /// tail straight through. `17 % 4 == 1`, so LTX-2's default clears wan's
    /// own `4k+1` grid check and then silently discards sixteen good frames
    /// at every Smooth seam. One authority so the two cannot drift.
    #[test]
    fn chain_motion_tail_follows_the_checkpoints_carryover_not_the_request() {
        use crate::SourceImageCapability::{Optional, Required, Unsupported};

        // Wan's seam re-renders exactly the one frame it was seeded with, and
        // only an image-conditioned checkpoint can be seeded at all.
        for capability in [Required, Optional] {
            assert_eq!(
                chain_motion_tail_frames_for_family("wan", Some(capability), 17),
                WAN_HANDOFF_DUPLICATED_FRAMES,
                "{capability:?} carries context, so the tail is one frame"
            );
        }
        assert_eq!(
            chain_motion_tail_frames_for_family("wan", Some(Unsupported), 17),
            0,
            "a text-to-video checkpoint has no channel to be seeded through"
        );
        // Unclassified is "unknown", never an assumed handoff.
        assert_eq!(chain_motion_tail_frames_for_family("wan", None, 17), 0);

        // An already-correct request is left exactly as it is.
        assert_eq!(
            chain_motion_tail_frames_for_family("wan", Some(Required), 1),
            WAN_HANDOFF_DUPLICATED_FRAMES
        );

        // LTX-Video has no img2vid path, so its Smooth seams concatenate.
        assert_eq!(
            chain_motion_tail_frames_for_family("ltx-video", None, 17),
            0
        );

        // Every other family keeps what the caller asked for — LTX-2's tail
        // is a real latent window and the request owns it.
        assert_eq!(chain_motion_tail_frames_for_family("ltx2", None, 17), 17);
        assert_eq!(chain_motion_tail_frames_for_family("ltx2", None, 9), 9);
        assert_eq!(chain_motion_tail_frames_for_family("", None, 17), 17);
    }

    /// Saved provenance has to name the overlap that actually rendered.
    ///
    /// `OutputMetadata::from_generate_request` holds no family and resolves
    /// one through the manifest, which an installed `cv:` / `hf:` wan
    /// checkpoint has none of — so a continuation that ran with wan's single
    /// carryover frame was recorded as having used LTX-2's 17 (#783).
    /// Admission and the forced-local CLI both know the resolved family and
    /// materialize it before metadata is built.
    #[test]
    fn materializing_the_overlap_makes_saved_provenance_match_the_render() {
        let installed_wan = || {
            let mut req = extend_req();
            // An installed catalog id: `find_manifest` cannot classify it, so
            // the family-blind fallback is the wrong 17.
            req.model = "cv:2041121".to_string();
            req.width = 832;
            req.height = 480;
            req.fps = Some(16);
            req.frames = Some(49);
            req
        };

        let unmaterialized = installed_wan();
        assert_eq!(
            unmaterialized.effective_extend_overlap_frames(),
            DEFAULT_EXTEND_OVERLAP_FRAMES,
            "the family-blind fallback is exactly what makes materialization necessary"
        );

        let mut req = installed_wan();
        materialize_extend_overlap_frames(&mut req, Some("wan"));
        assert_eq!(
            req.extend_overlap_frames,
            Some(WAN_HANDOFF_DUPLICATED_FRAMES)
        );
        let metadata = crate::OutputMetadata::from_generate_request(&req, 7, None, "test");
        assert_eq!(
            metadata.extend_overlap_frames,
            Some(WAN_HANDOFF_DUPLICATED_FRAMES),
            "recorded provenance must be the overlap the engine applied"
        );
        // Net-new frames are derived from the same field, so the recorded
        // clip length stops disagreeing with the file too.
        assert_eq!(req.extend_new_frames(), Some(48));

        // An explicit value is authoritative.
        let mut explicit = installed_wan();
        explicit.extend_overlap_frames = Some(5);
        materialize_extend_overlap_frames(&mut explicit, Some("wan"));
        assert_eq!(explicit.extend_overlap_frames, Some(5));

        // LTX-2 keeps 17, and an ordinary render is never handed a bare
        // overlap — that is a validation error, not a default.
        let mut ltx2 = extend_req();
        materialize_extend_overlap_frames(&mut ltx2, Some("ltx2"));
        assert_eq!(
            ltx2.extend_overlap_frames,
            Some(DEFAULT_EXTEND_OVERLAP_FRAMES)
        );
        let mut plain = valid_req();
        materialize_extend_overlap_frames(&mut plain, Some("wan"));
        assert_eq!(plain.extend_overlap_frames, None);
    }

    #[test]
    fn extend_is_limited_to_families_with_a_continuation_path() {
        let mut req = extend_req();
        req.model = "ltx-video-0.9.6-distilled:bf16".to_string();
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("extend_video"), "got: {err}");
    }

    /// Wan continues a clip too (#783), but on its own VAE grid.
    ///
    /// The overlap re-encodes through the family's video VAE, and wan's
    /// compresses time by 4 where LTX-2's compresses by 8 — a hardcoded 8
    /// rejected every valid wan overlap.
    #[test]
    fn wan_extend_uses_wans_own_temporal_grid() {
        let wan_req = |overlap: Option<u32>| {
            let mut req = extend_req();
            req.model = "wan22-ti2v-5b:fp16".to_string();
            req.width = 704;
            req.height = 384;
            req.fps = Some(24);
            req.frames = Some(49);
            req.extend_overlap_frames = overlap;
            req
        };

        // Wan carries exactly one frame — the seed — and 1 is on both grids.
        assert!(validate_generate_request(&wan_req(Some(1))).is_ok());
        // 5 and 13 are on 4k+1 but not on 8k+1: the old rule refused them.
        for overlap in [5u32, 9, 13] {
            assert!(
                validate_generate_request(&wan_req(Some(overlap))).is_ok(),
                "{overlap} is on wan's 4k+1 grid",
            );
        }
        // Off wan's grid, and the message must name wan's step, not LTX-2's.
        let err = validate_generate_request(&wan_req(Some(4))).unwrap_err();
        assert!(err.contains("4k+1"), "got: {err}");
        assert!(!err.contains("8k+1"), "got: {err}");
    }

    #[test]
    fn extend_rejects_both_inline_bytes_and_a_path() {
        let mut req = extend_req();
        req.extend_video_path = Some("/srv/mold/clip.mp4".to_string());
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("cannot be combined"), "got: {err}");
    }

    #[test]
    fn extend_rejects_empty_payloads() {
        let mut req = extend_req();
        req.extend_video = Some(Vec::new());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("must not be empty"));

        let mut req = extend_req();
        req.extend_video = None;
        req.extend_video_path = Some("   ".to_string());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("must not be empty"));
    }

    /// The overlap re-encodes through the VAE's 8x causal temporal grid, so an
    /// off-grid value would not map onto whole latent slots.
    #[test]
    fn extend_overlap_must_sit_on_the_latent_grid() {
        let mut req = extend_req();
        req.extend_overlap_frames = Some(12);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("8k+1"), "got: {err}");

        for overlap in [1u32, 9, 17, 25] {
            let mut req = extend_req();
            req.extend_overlap_frames = Some(overlap);
            assert!(
                validate_generate_request(&req).is_ok(),
                "{overlap} is on the 8k+1 grid",
            );
        }
    }

    /// An overlap at or above the clip length means every rendered frame
    /// reproduces the source and the continuation adds nothing.
    #[test]
    fn extend_overlap_must_leave_room_for_new_frames() {
        let mut req = extend_req();
        req.frames = Some(25);
        req.extend_overlap_frames = Some(25);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("strictly less than"), "got: {err}");

        req.extend_overlap_frames = Some(17);
        assert!(validate_generate_request(&req).is_ok());
        assert_eq!(req.extend_new_frames(), Some(8));
    }

    #[test]
    fn extend_overlap_requires_a_video_to_extend() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.frames = Some(97);
        req.extend_overlap_frames = Some(17);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("requires extend_video"), "got: {err}");
    }

    /// Extend continues one clip's motion; the other conditioning inputs each
    /// claim authority over the same opening frames.
    #[test]
    fn extend_rejects_competing_conditioning_inputs() {
        let mut req = extend_req();
        req.source_video = Some(vec![1, 2, 3]);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("source_video"));

        let mut req = extend_req();
        req.source_image = Some(png_bytes());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("source_image"));

        let mut req = extend_req();
        req.keyframes = Some(vec![KeyframeCondition {
            frame: 0,
            image: png_bytes(),
            name: None,
        }]);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("keyframes"));
    }

    /// An extend clip is an ordinary render, so it is bound by the same
    /// duration budget as any other single request.
    #[test]
    fn extend_respects_the_temporal_budget() {
        let mut req = extend_req();
        req.frames = Some(481);
        assert!(validate_generate_request(&req).is_ok());

        req.frames = Some(489);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("RoPE"), "got: {err}");
    }

    /// Extend provenance must reach saved metadata, and must not appear on
    /// ordinary renders where it would read as a continuation that never was.
    #[test]
    fn extend_provenance_reaches_output_metadata() {
        let mut req = extend_req();
        req.extend_video = None;
        req.extend_video_path = Some("/srv/mold/clip.mp4".to_string());
        req.extend_overlap_frames = Some(25);
        let metadata = crate::OutputMetadata::from_generate_request(&req, 7, None, "test");
        assert_eq!(
            metadata.extend_video_path.as_deref(),
            Some("/srv/mold/clip.mp4")
        );
        assert_eq!(metadata.extend_overlap_frames, Some(25));

        let plain = crate::OutputMetadata::from_generate_request(&valid_req(), 7, None, "test");
        assert_eq!(plain.extend_video_path, None);
        assert_eq!(plain.extend_overlap_frames, None);
    }

    /// The RoPE temporal axis is expressed in *seconds* (`rope.rs`'s
    /// `scale_video_time_to_seconds` divides the pixel-frame coordinate by fps
    /// before `max_pos` normalization), so the ceiling is a duration and must
    /// scale with fps rather than sit at a fixed frame count.
    #[test]
    fn ltx2_frame_ceiling_tracks_fps() {
        // Latent k spans pixel [8k-7, 8k+1] after the causal fix, so F frames
        // put the last RoPE midpoint at (F-4)/fps seconds: F = 20*fps + 4.
        assert_eq!(ltx2_max_frames_at_fps(24), 484);
        assert_eq!(ltx2_max_frames_at_fps(25), 504);
        assert_eq!(ltx2_max_frames_at_fps(12), 244);
        assert_eq!(ltx2_max_frames_at_fps(8), 164);
        // Low fps is *tighter* than the old flat 153: 6 fps only buys 20s of
        // runtime, which the previous constant silently over-admitted.
        assert_eq!(ltx2_max_frames_at_fps(6), 124);
        // The absolute resource guard binds before the seconds budget does at
        // high frame rates.
        assert_eq!(ltx2_max_frames_at_fps(60), LTX2_MAX_FRAMES_ABSOLUTE);
        assert_eq!(ltx2_max_frames_at_fps(120), LTX2_MAX_FRAMES_ABSOLUTE);
        // fps=0 is rejected elsewhere; the helper must not divide by zero.
        assert_eq!(ltx2_max_frames_at_fps(0), ltx2_max_frames_at_fps(1));
    }

    /// The helper values are what /api/models advertises as max_frames /
    /// frame_step; they must agree with what the validator enforces so the
    /// wire contract can't drift from the actual rejection rules.
    #[test]
    fn frame_constraint_helpers_match_validator_behavior() {
        // Advertised values are grid-snapped so a client that clamps to them
        // can actually submit; the raw duration ceiling is off the 8n+1 grid.
        assert_eq!(
            max_frames_for_family("ltx2"),
            Some(ltx2_max_frames_on_grid_at_fps(LTX2_DEFAULT_FPS))
        );
        assert_eq!(max_frames_for_family_at_fps("ltx2", 12), Some(241));
        assert_eq!(max_frames_for_family_at_fps("ltx-video", 12), Some(257));
        assert_eq!(max_frames_for_family("ltx-video"), Some(257));
        assert_eq!(max_frames_for_family("flux"), None);
        assert_eq!(max_frames_for_family("sdxl"), None);
        assert_eq!(frame_step_for_family("ltx2"), Some(8));
        assert_eq!(frame_step_for_family("ltx-video"), Some(8));
        assert_eq!(frame_step_for_family("flux"), None);
        assert_eq!(min_frames_for_family("flux"), None);
        assert_eq!(fixed_fps_for_family("flux"), None);

        // One grid step past the advertised ltx-video cap must be rejected,
        // and the rejection must quote the same cap the wire advertises.
        let cap = max_frames_for_family("ltx-video").unwrap();
        let mut req = valid_req();
        req.model = "ltx-video-0.9.6-distilled:bf16".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.frames = Some(cap + 8); // stays on the 8n+1 grid so only the cap trips
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains(&cap.to_string()), "got: {err}");

        // Same agreement for the ltx2 ceiling, which is fps-dependent, so the
        // request has to name the fps the helper was asked about.
        let cap = max_frames_for_family_at_fps("ltx2", 12).unwrap();
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(12);
        req.frames = Some(249); // first 8n+1 value past the 244-frame cap
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains(&cap.to_string()), "got: {err}");
    }

    #[test]
    fn h3_post_activation_timing_authority_rejects_short_or_retimed_requests() {
        assert_eq!(
            min_frames_for_family(crate::minimax_h3::FAMILY),
            Some(crate::minimax_h3::MIN_FRAMES)
        );
        assert_eq!(
            fixed_fps_for_family(crate::minimax_h3::FAMILY),
            Some(crate::minimax_h3::FIXED_FPS)
        );
        assert_eq!(
            max_frames_for_family(crate::minimax_h3::FAMILY),
            Some(crate::minimax_h3::MAX_FRAMES)
        );
        assert_eq!(
            max_runtime_seconds_for_family(crate::minimax_h3::FAMILY),
            Some(crate::minimax_h3::MAX_DURATION_SECONDS)
        );
        assert_eq!(
            max_frames_absolute_for_family(crate::minimax_h3::FAMILY),
            Some(crate::minimax_h3::MAX_FRAMES)
        );

        let short = validate_family_video_timing_constraints(
            Some(crate::minimax_h3::FRAME_OFFSET),
            Some(crate::minimax_h3::FIXED_FPS),
            Some(crate::minimax_h3::FAMILY),
        )
        .unwrap_err();
        assert!(short.contains("124"), "got: {short}");

        let retimed = validate_family_video_timing_constraints(
            Some(crate::minimax_h3::MIN_FRAMES),
            Some(23),
            Some(crate::minimax_h3::FAMILY),
        )
        .unwrap_err();
        assert!(retimed.contains("24 fps"), "got: {retimed}");

        assert!(validate_family_video_timing_constraints(
            Some(crate::minimax_h3::MIN_FRAMES),
            Some(crate::minimax_h3::FIXED_FPS),
            Some(crate::minimax_h3::FAMILY),
        )
        .is_ok());
    }

    /// Wan advertises a flat frame guard on the `4k+1` grid; the advertised
    /// values must agree with what the validator enforces (same drift-proofing
    /// as the LTX contract above).
    #[test]
    fn wan_frame_contract_helpers() {
        assert_eq!(frame_step_for_family("wan"), Some(WAN_TEMPORAL_SCALE));
        assert_eq!(
            max_frames_for_family_at_fps("wan", 16),
            Some(MAX_FRAMES_GLOBAL)
        );
        assert_eq!(max_frames_for_family("wan"), Some(MAX_FRAMES_GLOBAL));
        // Wan's ceiling is a flat resource guard, not a duration budget.
        assert_eq!(max_runtime_seconds_for_family("wan"), None);
        assert_eq!(max_frames_absolute_for_family("wan"), None);
        // The advertised maximum must itself sit on the 4k+1 grid so a client
        // that clamps a slider to it can actually submit.
        assert_eq!((MAX_FRAMES_GLOBAL - 1) % WAN_TEMPORAL_SCALE, 0);
    }

    #[test]
    fn wan_frames_grid_and_cap_enforced() {
        // On-grid counts inside the guard are accepted (81 is Wan's default).
        let mut req = valid_req();
        req.model = "wan22-ti2v-5b:fp16".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(24);
        req.frames = Some(81);
        assert!(validate_generate_request(&req).is_ok());

        // Off the 4k+1 grid is rejected, and the error names Wan's own step —
        // not the LTX 8.
        req.frames = Some(80);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("4n+1"), "got: {err}");

        // One grid step past the flat guard is rejected with the same cap the
        // wire advertises.
        let cap = max_frames_for_family("wan").unwrap();
        req.frames = Some(cap + WAN_TEMPORAL_SCALE);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains(&cap.to_string()), "got: {err}");
    }

    /// Wan is a video family in every consuming authority: the expansion
    /// task resolver (twin: `studio/lib/expandTask.ts`) and the output-format
    /// default + gate. A miss in either renders wan as an image model.
    #[test]
    fn wan_routes_through_the_video_authorities() {
        use crate::ExpandTask;
        assert_eq!(ExpandTask::for_family("wan"), ExpandTask::TextToVideo);
        assert_eq!(
            ExpandTask::for_conditioning("wan", None, true, false, false, 0, false, None),
            ExpandTask::ImageToVideo
        );
        assert_eq!(
            ExpandTask::for_conditioning("wan", None, false, false, false, 0, false, None),
            ExpandTask::TextToVideo
        );

        let mut req = valid_req();
        req.model = "wan22-ti2v-5b:fp16".to_string();
        req.output_format = None;
        req.normalise_output_format(Some("wan"));
        assert_eq!(req.resolved_output_format(), OutputFormat::Mp4);

        req.output_format = Some(OutputFormat::Png);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("mp4"), "got: {err}");
    }

    /// #798: a single-frame Wan render is a still. png/jpeg are admitted at
    /// exactly `frames == 1`, default to png there, and classify as
    /// image-style prompt work — while `frames > 1` (or unset, which defaults
    /// to a full clip) keeps the video-only contract and its current error.
    /// Twin: `studio/lib/expandTask.ts`.
    #[test]
    fn wan_single_frame_is_a_still() {
        use crate::ExpandTask;

        let mut req = valid_req();
        req.model = "wan22-t2v-a14b:q5".to_string();
        req.frames = Some(1);

        // Unset format at frames=1 normalises to a still, and both image
        // formats pass admission.
        req.output_format = None;
        req.normalise_output_format(Some("wan"));
        assert_eq!(req.resolved_output_format(), OutputFormat::Png);
        assert!(validate_generate_request(&req).is_ok());
        req.output_format = Some(OutputFormat::Jpeg);
        assert!(validate_generate_request(&req).is_ok());
        // Video formats stay allowed at frames=1 — permitting stills must not
        // revoke the existing contract.
        req.output_format = Some(OutputFormat::Mp4);
        assert!(validate_generate_request(&req).is_ok());
        // Audio never becomes admissible through the still gate.
        req.output_format = Some(OutputFormat::Wav);
        assert!(validate_generate_request(&req).is_err());

        // frames > 1 and frames unset keep refusing image formats with the
        // current message.
        for frames in [Some(5), None] {
            req.frames = frames;
            req.output_format = Some(OutputFormat::Png);
            let err = validate_generate_request(&req).unwrap_err();
            assert!(err.contains("mp4, gif, apng, or webp"), "got: {err}");
            req.output_format = None;
            req.normalise_output_format(Some("wan"));
            assert_eq!(req.resolved_output_format(), OutputFormat::Mp4);
        }

        // The expansion task follows: a frames=1 wan request is image-style
        // prompt work, not chronological shot direction.
        assert_eq!(
            ExpandTask::for_conditioning("wan", None, false, false, false, 0, false, Some(1)),
            ExpandTask::TextToImage
        );
        // …unless a source image conditions it: source authority survives the
        // still contract (codex review).
        assert_eq!(
            ExpandTask::for_conditioning("wan", None, true, false, false, 0, false, Some(1)),
            ExpandTask::ImageToVideo
        );
        assert_eq!(
            ExpandTask::for_conditioning("wan", None, false, false, false, 0, false, Some(81)),
            ExpandTask::TextToVideo
        );
        // LTX keeps its video classification even at one frame — the still
        // contract is wan's.
        assert_eq!(
            ExpandTask::for_conditioning("ltx2", None, false, false, false, 0, false, Some(1)),
            ExpandTask::TextToVideo
        );
        let mut still_req = valid_req();
        still_req.model = "wan22-t2v-a14b:q5".to_string();
        still_req.frames = Some(1);
        assert_eq!(
            ExpandTask::for_generation("wan", &still_req),
            ExpandTask::TextToImage
        );
    }

    /// #782 / #795: the wan recipe knobs are admitted for wan and rejected —
    /// never ignored — everywhere else, and the shared scheduler slot's two
    /// solver families stay disjoint at admission.
    #[test]
    fn wan_recipe_knobs_gate_by_family() {
        use crate::Scheduler;

        // sample_shift: wan takes finite positive values only.
        let mut wan = valid_req();
        wan.model = "wan22-t2v-a14b:q8".to_string();
        wan.output_format = Some(OutputFormat::Mp4);
        wan.sample_shift = Some(12.0);
        assert!(validate_generate_request(&wan).is_ok());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            wan.sample_shift = Some(bad);
            assert!(
                validate_generate_request(&wan).is_err(),
                "shift {bad} must be rejected"
            );
        }
        wan.sample_shift = None;

        // The wan solvers ride the scheduler slot; UNet schedulers are
        // refused for wan and the wan solvers are refused off-family.
        for solver in [Scheduler::UniPc, Scheduler::Euler, Scheduler::DpmPp] {
            wan.scheduler = Some(solver);
            assert!(
                validate_generate_request(&wan).is_ok(),
                "wan must accept {solver}"
            );
        }
        for unet in [Scheduler::Ddim, Scheduler::EulerAncestral] {
            wan.scheduler = Some(unet);
            let err = validate_generate_request(&wan).unwrap_err();
            assert!(err.contains("UNet scheduler"), "got: {err}");
        }
        wan.scheduler = None;

        // The fp8-scaled tier refuses LoRA stacks at admission — its loader
        // fails closed, but only after the UMT5 encode. GGUF tiers accept.
        wan.model = "wan22-t2v-a14b:fp8".to_string();
        wan.loras = Some(vec![crate::LoraWeight {
            path: "distill.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        let err = validate_generate_request(&wan).unwrap_err();
        assert!(err.contains("fp8-scaled"), "got: {err}");
        wan.model = "wan22-t2v-a14b:q8".to_string();
        assert!(validate_generate_request(&wan).is_ok());
        wan.loras = None;

        // Distill strengths: the community band is accepted, typos are not.
        wan.distill_strength_high = Some(1.8);
        wan.distill_strength_low = Some(1.0);
        assert!(validate_generate_request(&wan).is_ok());
        wan.distill_strength_high = Some(4.5);
        assert!(validate_generate_request(&wan).is_err());
        wan.distill_strength_high = Some(0.0);
        assert!(validate_generate_request(&wan).is_err());

        // Every knob is rejected, not ignored, for a non-wan family.
        let mut flux = valid_req();
        flux.sample_shift = Some(5.0);
        let err = validate_generate_request(&flux).unwrap_err();
        assert!(err.contains("sample_shift"), "got: {err}");
        flux.sample_shift = None;
        flux.distill_strength_high = Some(1.5);
        let err = validate_generate_request(&flux).unwrap_err();
        assert!(err.contains("distill_strength_high"), "got: {err}");
        flux.distill_strength_high = None;
        flux.scheduler = Some(Scheduler::Euler);
        let err = validate_generate_request(&flux).unwrap_err();
        assert!(err.contains("Wan sample solver"), "got: {err}");
        // The UNet schedulers keep working off-family.
        flux.scheduler = Some(Scheduler::Ddim);
        assert!(validate_generate_request(&flux).is_ok());
    }

    /// #779: wan admits exactly the first/last endpoint keyframe pair, and
    /// classifies it as boundary-preserving prompt work — parity twin:
    /// `studio/lib/expandTask.ts`.
    #[test]
    fn wan_keyframes_admit_only_the_endpoint_pair() {
        use crate::{ExpandTask, KeyframeCondition};
        let keyframe = |frame: u32| KeyframeCondition {
            frame,
            // A real PNG header so the image-format check passes.
            image: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
            name: None,
        };

        let mut req = valid_req();
        req.model = "wan22-i2v-a14b:q5".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.frames = Some(33);
        req.keyframes = Some(vec![keyframe(0), keyframe(32)]);
        assert!(validate_generate_request(&req).is_ok());

        // Wrong count, wrong anchors, and the ambiguous source+keyframes mix
        // are named at admission.
        req.keyframes = Some(vec![keyframe(0)]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("exactly two keyframes"), "got: {err}");
        req.keyframes = Some(vec![keyframe(0), keyframe(7)]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("frames 0 and 32"), "got: {err}");
        req.keyframes = Some(vec![keyframe(0), keyframe(32)]);
        req.source_image = Some(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("not both"), "got: {err}");
        req.source_image = None;

        // Without an explicit frames count the closing anchor is uncheckable
        // here, and the engine would reject a mismatch only after loading.
        req.frames = None;
        req.keyframes = Some(vec![keyframe(0), keyframe(32)]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("explicit frames count"), "got: {err}");

        // frames=1 renders a still — its endpoints coincide, so the pair is
        // refused by name rather than as a generic duplicate frame.
        req.frames = Some(1);
        req.keyframes = Some(vec![keyframe(0), keyframe(0)]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("multi-frame clip"), "got: {err}");
        req.frames = Some(33);
        req.keyframes = Some(vec![keyframe(0), keyframe(32)]);

        // TI2V pins endpoints in latent space: a 5-frame pixel clip is two
        // latent frames, both anchored — refused before the 10 GB load. The
        // A14B channel-concat path has no such floor (checked above at 33).
        let mut ti2v = valid_req();
        ti2v.model = "wan22-ti2v-5b:fp16".to_string();
        ti2v.output_format = Some(OutputFormat::Mp4);
        ti2v.width = 1280;
        ti2v.height = 704;
        ti2v.frames = Some(5);
        ti2v.keyframes = Some(vec![keyframe(0), keyframe(4)]);
        let err = validate_generate_request(&ti2v).unwrap_err();
        assert!(err.contains("at least 9 frames"), "got: {err}");
        ti2v.frames = Some(9);
        ti2v.keyframes = Some(vec![keyframe(0), keyframe(8)]);
        assert!(validate_generate_request(&ti2v).is_ok());

        // The expansion task treats the pair as boundary anchors, exactly as
        // LTX-2 keyframes classify.
        assert_eq!(
            ExpandTask::for_generation("wan", &req),
            ExpandTask::KeyframeInterpolation
        );

        // Non-video families keep rejecting keyframes outright.
        let mut flux = valid_req();
        flux.keyframes = Some(vec![keyframe(0), keyframe(8)]);
        assert!(validate_generate_request(&flux).is_err());
    }

    /// Buckets are advertised per checkpoint: the 480p-only 1.3B and the
    /// 704-grid TI2V-5B must not inherit each other's sizes, while unknown
    /// wan checkpoints keep the family fallback.
    #[test]
    fn wan_recommended_dimensions_are_per_checkpoint() {
        assert_eq!(
            wan_recommended_dimensions("wan21-t2v-1.3b"),
            &[(832, 480), (480, 832)]
        );
        assert_eq!(
            wan_recommended_dimensions("wan22-ti2v-5b:fp16"),
            &[(1280, 704), (704, 1280)]
        );
        assert_eq!(
            wan_recommended_dimensions("cv:someone/some-wan-finetune"),
            recommended_dimensions("wan")
        );
        for model in ["wan21-t2v-1.3b", "wan22-ti2v-5b"] {
            for (w, h) in wan_recommended_dimensions(model) {
                assert!(
                    validate_generation_dimensions(*w, *h, Some("wan")).is_ok(),
                    "{model}: advertised {w}x{h} must pass the validator"
                );
                assert!(
                    validate_generation_dimensions_for_model(
                        model,
                        *w,
                        *h,
                        Some("wan"),
                        Ltx2SpatialComposition::SinglePass,
                    )
                    .is_ok(),
                    "{model}: advertised {w}x{h} must pass its own model-aware validator"
                );
            }
        }
    }

    /// The grid is per checkpoint too: `wan22-ti2v-5b`'s 2.2 VAE compresses
    /// 16x spatially and its DiT patches the latent 2x2, so its pixel grid is
    /// 32 while the 2.1-VAE checkpoints keep the family's 16.
    #[test]
    fn wan_dimension_alignment_is_per_checkpoint() {
        assert_eq!(wan_dimension_alignment("wan22-ti2v-5b"), 32);
        assert_eq!(wan_dimension_alignment("wan22-ti2v-5b:fp16"), 32);
        // Variant tags and the legacy dash form must match like
        // `wan_recommended_dimensions` — a `:q8` install of the same
        // checkpoint has the same VAE.
        assert_eq!(wan_dimension_alignment("wan22-ti2v-5b:q8"), 32);
        assert_eq!(wan_dimension_alignment("wan22-ti2v-5b-fp16"), 32);
        assert_eq!(wan_dimension_alignment("wan21-t2v-1.3b"), 16);
        assert_eq!(wan_dimension_alignment("wan22-t2v-a14b:q5"), 16);
        assert_eq!(wan_dimension_alignment("wan22-i2v-a14b:q8"), 16);
        // Unknown catalog installs keep the family fallback; deriving the
        // grid from a sidecar-described VAE is follow-up work.
        assert_eq!(wan_dimension_alignment("cv:someone/some-wan-finetune"), 16);
    }

    #[test]
    fn dimension_alignment_for_model_dispatches_wan_checkpoints() {
        assert_eq!(
            dimension_alignment_for_model("wan22-ti2v-5b", Some("wan")),
            32
        );
        // Manifest models resolve their family without a hint.
        assert_eq!(
            dimension_alignment_for_model("wan22-ti2v-5b:fp16", None),
            32
        );
        assert_eq!(
            dimension_alignment_for_model("wan21-t2v-1.3b", Some("wan")),
            16
        );
        assert_eq!(
            dimension_alignment_for_model("cv:someone/some-wan-finetune", Some("wan")),
            16
        );
        // Every other family keeps its family-wide answer.
        assert_eq!(
            dimension_alignment_for_model("ltx-2-19b-distilled:fp8", Some("ltx2")),
            32
        );
        assert_eq!(
            dimension_alignment_for_model("flux-dev:q4", Some("flux")),
            16
        );
    }

    /// 1280x720 is on the family's 16 px grid but off the 5B's 32 px grid.
    /// Admission must reject it before a 10 GB model load, with the engine's
    /// own number; the same canvas stays valid for the 2.1-VAE checkpoints.
    #[test]
    fn wan22_ti2v_5b_off_grid_dimensions_rejected_at_admission() {
        let mut req = valid_req();
        req.model = "wan22-ti2v-5b".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.width = 1280;
        req.height = 720;
        let err = validate_generate_request_with_family(&req, Some("wan")).unwrap_err();
        assert!(err.contains("multiples of 32"), "got: {err}");

        req.width = 704;
        req.height = 1280;
        validate_generate_request_with_family(&req, Some("wan"))
            .expect("the 5B's native portrait bucket is on its 32px grid");

        req.model = "wan21-t2v-1.3b".to_string();
        req.width = 1280;
        req.height = 720;
        validate_generate_request_with_family(&req, Some("wan"))
            .expect("the 2.1-VAE checkpoints keep the family's 16px grid");
    }

    #[test]
    fn validate_generation_dimensions_for_model_uses_the_checkpoint_grid() {
        let err = validate_generation_dimensions_for_model(
            "wan22-ti2v-5b",
            1280,
            720,
            Some("wan"),
            Ltx2SpatialComposition::SinglePass,
        )
        .unwrap_err();
        assert!(err.contains("multiples of 32"), "got: {err}");
        validate_generation_dimensions_for_model(
            "wan22-ti2v-5b",
            1280,
            704,
            Some("wan"),
            Ltx2SpatialComposition::SinglePass,
        )
        .expect("1280x704 sits on the 32px grid");
        // The family-only validator deliberately keeps the compatible 16px
        // answer for callers that cannot name a model.
        assert!(validate_generation_dimensions(1280, 720, Some("wan")).is_ok());
    }

    #[test]
    fn wan_recommended_dimensions_fit_their_own_contracts() {
        let dims = recommended_dimensions("wan");
        assert!(!dims.is_empty());
        for (w, h) in dims {
            assert!(
                w.is_multiple_of(16) && h.is_multiple_of(16),
                "{w}x{h} must sit on the family's 16px grid"
            );
            assert!(
                u64::from(*w) * u64::from(*h) <= MAX_PIXELS,
                "{w}x{h} must fit the generic pixel budget"
            );
            assert!(
                validate_generation_dimensions(*w, *h, Some("wan")).is_ok(),
                "{w}x{h} must pass the validator it is advertised against"
            );
        }
    }

    #[test]
    fn ltx2_frames_at_rope_budget_accepted() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(24);
        // 481 = 20s at 24 fps on the 8n+1 grid (484 is the exact ceiling).
        req.frames = Some(481);
        assert!(validate_generate_request(&req).is_ok());
    }

    /// The old flat 153 was a floor, not a ceiling, at the default frame rate:
    /// LTX-2.3 advertises ~20s single-shot generation and the checkpoint budget
    /// agrees. Frame counts that used to be rejected out of hand must pass.
    #[test]
    fn ltx2_frames_over_the_old_flat_cap_are_accepted_within_the_duration_budget() {
        for frames in [161u32, 193, 257, 401] {
            let mut req = valid_req();
            req.model = "ltx-2-19b-distilled:fp8".to_string();
            req.output_format = Some(OutputFormat::Mp4);
            req.fps = Some(24);
            req.frames = Some(frames);
            assert!(
                validate_generate_request(&req).is_ok(),
                "{frames} frames at 24 fps is {:.1}s, inside the {LTX2_MAX_RUNTIME_SECONDS}s budget",
                frames as f64 / 24.0,
            );
        }
    }

    #[test]
    fn ltx2_frames_over_rope_budget_rejected() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(24);
        req.frames = Some(489); // 20.2s at 24 fps — one grid step past the budget
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("489"), "got: {err}");
        // The quoted ceiling is grid-snapped so it is directly usable: 484 is
        // the exact budget but 483 % 8 == 3, so retrying at 484 would fail again.
        assert!(err.contains("481"), "got: {err}");
        assert!(err.contains("RoPE"), "got: {err}");
    }

    /// The same frame count can be inside or outside the budget depending on
    /// fps — this is the whole point of deriving the ceiling instead of fixing
    /// it. 193 frames is 8s at 24 fps but 32s at 6 fps.
    #[test]
    fn ltx2_frame_budget_is_a_duration_not_a_frame_count() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.frames = Some(193);

        req.fps = Some(24);
        assert!(validate_generate_request(&req).is_ok());

        req.fps = Some(6);
        let err = validate_generate_request(&req).unwrap_err();
        // 20s at 6 fps is 124 frames; 121 is that budget on the 8n+1 grid.
        assert!(err.contains("121"), "got: {err}");
    }

    #[test]
    fn ltx2_absolute_frame_guard_binds_above_thirty_fps() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(120);
        req.frames = Some(609); // first 8n+1 value past the 604-frame guard
        let err = validate_generate_request(&req).unwrap_err();
        // The absolute guard binds here, quoted on the 8n+1 grid (604 -> 601).
        assert!(
            err.contains(&ltx2_max_frames_on_grid_at_fps(120).to_string()),
            "got: {err}"
        );
        assert_eq!(ltx2_max_frames_on_grid_at_fps(120), 601);
    }

    #[test]
    fn ltx_video_family_is_not_subject_to_the_ltx2_rope_cap() {
        let mut req = valid_req();
        req.model = "ltx-video-0.9.6-distilled:bf16".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.frames = Some(161);
        assert!(validate_generate_request(&req).is_ok());
    }

    /// `ltx-video` keeps the flat global ceiling; only `ltx2` publishes a
    /// duration budget, so the two families must not share a cap.
    #[test]
    fn ltx_video_keeps_the_flat_global_ceiling() {
        let mut req = valid_req();
        req.model = "ltx-video-0.9.6-distilled:bf16".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(30);
        req.frames = Some(MAX_FRAMES_GLOBAL + 8);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains(&MAX_FRAMES_GLOBAL.to_string()), "got: {err}");
    }

    /// `derive_stage1_render_shape` halves the frame count *and* the fps, so an
    /// x2 temporal upscale renders the same runtime — it never buys duration.
    #[test]
    fn ltx2_temporal_upscale_x2_does_not_extend_the_duration_budget() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(24);
        req.temporal_upscale = Some(crate::Ltx2TemporalUpscale::X2);

        // stage 1 = (481-1)/2+1 = 241 frames at 12 fps, ceiling 244 → fits.
        req.frames = Some(481);
        assert!(validate_generate_request(&req).is_ok());

        // 20.4s of runtime is over budget with or without temporal upscaling.
        req.frames = Some(497);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("RoPE"), "got: {err}");
    }

    #[test]
    fn non_ltx_models_do_not_apply_the_ltx_frame_grid_rule() {
        let mut req = valid_req();
        req.frames = Some(10);
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn zero_batch_rejected() {
        let mut req = valid_req();
        req.batch_size = 0;
        assert!(validate_generate_request(&req).is_err());
    }

    #[test]
    fn large_batch_accepted() {
        let mut req = valid_req();
        req.batch_size = 100;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn negative_guidance_rejected() {
        let mut req = valid_req();
        req.guidance = -1.0;
        assert!(validate_generate_request(&req).is_err());
    }

    #[test]
    fn zero_guidance_valid() {
        let mut req = valid_req();
        req.guidance = 0.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn high_guidance_valid() {
        let mut req = valid_req();
        req.guidance = 20.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn guidance_over_100_rejected() {
        let mut req = valid_req();
        req.guidance = 100.1;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("guidance"));
    }

    #[test]
    fn guidance_at_100_valid() {
        let mut req = valid_req();
        req.guidance = 100.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn prompt_too_long_rejected() {
        let mut req = valid_req();
        req.prompt = "x".repeat(77_001);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("77,000"));
    }

    #[test]
    fn prompt_at_limit_valid() {
        let mut req = valid_req();
        req.prompt = "x".repeat(77_000);
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn negative_prompt_too_long_rejected() {
        let mut req = valid_req();
        req.negative_prompt = Some("x".repeat(77_001));
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("negative_prompt"));
    }

    #[test]
    fn negative_prompt_at_limit_valid() {
        let mut req = valid_req();
        req.negative_prompt = Some("x".repeat(77_000));
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn negative_prompt_none_valid() {
        let req = valid_req();
        assert!(req.negative_prompt.is_none());
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn negative_prompt_empty_valid() {
        let mut req = valid_req();
        req.negative_prompt = Some(String::new());
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn seed_is_optional() {
        let mut req = valid_req();
        req.seed = None;
        assert!(validate_generate_request(&req).is_ok());
    }

    // ── img2img validation tests ────────────────────────────────────────────

    #[test]
    fn img2img_strength_zero_accepted() {
        let mut req = valid_req();
        req.source_image = Some(png_bytes());
        req.strength = 0.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn img2img_strength_negative_rejected() {
        let mut req = valid_req();
        req.source_image = Some(png_bytes());
        req.strength = -0.1;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("strength"));
    }

    #[test]
    fn img2img_strength_one_accepted() {
        let mut req = valid_req();
        req.source_image = Some(png_bytes());
        req.strength = 1.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn img2img_strength_half_accepted() {
        let mut req = valid_req();
        req.source_image = Some(png_bytes());
        req.strength = 0.5;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn img2img_invalid_magic_bytes_rejected() {
        let mut req = valid_req();
        req.source_image = Some(vec![0x00, 0x01, 0x02, 0x03]);
        req.strength = 0.75;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("PNG or JPEG"));
    }

    #[test]
    fn img2img_jpeg_accepted() {
        let mut req = valid_req();
        req.source_image = Some(jpeg_bytes());
        req.strength = 0.75;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn img2img_no_source_image_skips_strength_check() {
        let mut req = valid_req();
        req.source_image = None;
        req.strength = 0.0; // Would fail if source_image present, but should pass without
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn qwen_image_edit_requires_edit_images() {
        let mut req = valid_req();
        req.model = "qwen-image-edit:q4".to_string();
        let err = validate_generate_request(&req).unwrap_err();
        assert_eq!(
            err,
            "Qwen Image Edit needs at least one image. Add a Target image and try again."
        );
    }

    #[test]
    fn qwen_image_edit_rejects_batch_size_above_one() {
        let mut req = valid_req();
        req.model = "qwen-image-edit:q4".to_string();
        req.edit_images = Some(vec![png_bytes()]);
        req.batch_size = 2;
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("batch_size = 1"), "got: {err}");
    }

    #[test]
    fn qwen_image_edit_accepts_edit_images() {
        let mut req = valid_req();
        req.model = "qwen-image-edit:q4".to_string();
        req.edit_images = Some(vec![png_bytes()]);
        req.guidance = 4.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn flux2_dev_accepts_text_only_and_ordered_references() {
        let mut req = valid_req();
        req.model = "flux2-dev:bf16".to_string();
        req.guidance = 4.0;
        assert!(validate_generate_request(&req).is_ok());

        req.edit_images = Some(vec![png_bytes(), jpeg_bytes()]);
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn flux2_dev_catalog_id_accepts_references_but_rejects_img2img_fields() {
        let mut req = valid_req();
        req.model = "hf:black-forest-labs/FLUX.2-dev".to_string();
        req.edit_images = Some(vec![png_bytes()]);
        assert!(validate_generate_request_with_family(&req, Some("flux2")).is_ok());

        req.source_image = Some(png_bytes());
        let error = validate_generate_request_with_family(&req, Some("flux2")).unwrap_err();
        assert!(error.contains("edit_images instead of source_image"));
    }

    #[test]
    fn flux2_dev_bounds_reference_count_and_rejects_lora() {
        let mut req = valid_req();
        req.model = "flux2-dev:bf16".to_string();
        req.edit_images = Some(vec![png_bytes(); FLUX2_DEV_MAX_REFERENCE_IMAGES + 1]);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("at most"));

        req.edit_images = None;
        req.lora = Some(LoraWeight {
            path: "adapter.safetensors".into(),
            scale: 1.0,

            expert: None,
        });
        assert_eq!(
            validate_generate_request(&req).unwrap_err(),
            "flux2-dev does not support LoRA"
        );
    }

    #[test]
    fn qwen_image_edit_rejects_source_image_field() {
        let mut req = valid_req();
        req.model = "qwen-image-edit:q4".to_string();
        req.edit_images = Some(vec![png_bytes()]);
        req.source_image = Some(png_bytes());
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("edit_images instead of source_image"),
            "got: {err}"
        );
    }

    #[test]
    fn non_edit_models_reject_edit_images() {
        let mut req = valid_req();
        req.model = "flux-schnell:q8".to_string();
        req.edit_images = Some(vec![png_bytes()]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("only supported for qwen-image-edit"),
            "got: {err}"
        );
    }

    #[test]
    fn non_edit_models_reject_edit_images_before_format_validation() {
        let mut req = valid_req();
        req.model = "flux-schnell:q8".to_string();
        req.edit_images = Some(vec![b"not-an-image".to_vec()]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("only supported for qwen-image-edit"),
            "got: {err}"
        );
    }

    // ── ControlNet validation tests ────────────────────────────────────────

    #[test]
    fn controlnet_valid_request() {
        let mut req = valid_req();
        req.model = "dreamshaper-v8:fp16".to_string();
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        req.control_scale = 0.8;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn controlnet_image_without_model_rejected() {
        let mut req = valid_req();
        req.model = "dreamshaper-v8:fp16".to_string();
        req.control_image = Some(png_bytes());
        req.control_model = None;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("control_model"));
    }

    #[test]
    fn controlnet_model_without_image_rejected() {
        let mut req = valid_req();
        req.model = "dreamshaper-v8:fp16".to_string();
        req.control_image = None;
        req.control_model = Some("controlnet-canny-sd15".to_string());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("control_image"));
    }

    #[test]
    fn controlnet_invalid_image_rejected() {
        let mut req = valid_req();
        req.model = "dreamshaper-v8:fp16".to_string();
        req.control_image = Some(vec![0x00, 0x01, 0x02, 0x03]);
        req.control_model = Some("controlnet-canny-sd15".to_string());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("PNG or JPEG"));
    }

    #[test]
    fn controlnet_negative_scale_rejected() {
        let mut req = valid_req();
        req.model = "dreamshaper-v8:fp16".to_string();
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        req.control_scale = -0.1;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("control_scale"));
    }

    #[test]
    fn controlnet_zero_scale_accepted() {
        let mut req = valid_req();
        req.model = "dreamshaper-v8:fp16".to_string();
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        req.control_scale = 0.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn controlnet_high_scale_accepted() {
        let mut req = valid_req();
        req.model = "dreamshaper-v8:fp16".to_string();
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        req.control_scale = 2.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn controlnet_jpeg_accepted() {
        let mut req = valid_req();
        req.model = "dreamshaper-v8:fp16".to_string();
        req.control_image = Some(jpeg_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn controlnet_rejected_for_non_sd15_family() {
        let mut req = valid_req();
        req.model = "sdxl:fp16".to_string();
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());

        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("SD1.5"), "got: {err}");
    }
    // ── Inpainting validation tests ───────────────────────────────────────

    #[test]
    fn mask_without_source_image_rejected() {
        let mut req = valid_req();
        req.mask_image = Some(png_bytes());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("mask_image requires source_image"));
    }

    #[test]
    fn mask_with_source_image_accepted() {
        let mut req = valid_req();
        req.source_image = Some(png_bytes());
        req.mask_image = Some(png_bytes());
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn mask_jpeg_accepted() {
        let mut req = valid_req();
        req.source_image = Some(png_bytes());
        req.mask_image = Some(jpeg_bytes());
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn mask_invalid_bytes_rejected() {
        let mut req = valid_req();
        req.source_image = Some(png_bytes());
        req.mask_image = Some(vec![0x00, 0x01, 0x02, 0x03]);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("mask_image must be a PNG or JPEG"));
    }

    #[test]
    fn no_mask_no_source_passes() {
        let req = valid_req();
        assert!(validate_generate_request(&req).is_ok());
    }

    // ── fit_to_model_dimensions tests ────────────────────────────────────

    #[test]
    fn fit_same_aspect_downscale() {
        // 1024x1024 source -> 512x512 SD1.5 model
        assert_eq!(fit_to_model_dimensions(1024, 1024, 512, 512), (512, 512));
    }

    #[test]
    fn fit_wide_source_downscale() {
        // 1920x1080 source -> 512x512 SD1.5 model
        // width-limited: w=512, h=512/1.778=287.9 -> 288 (16px aligned)
        assert_eq!(fit_to_model_dimensions(1920, 1080, 512, 512), (512, 288));
    }

    #[test]
    fn fit_small_source_upscale_to_model_native() {
        // 512x512 source -> 1024x1024 FLUX model (upscale to native)
        assert_eq!(fit_to_model_dimensions(512, 512, 1024, 1024), (1024, 1024));
    }

    #[test]
    fn fit_portrait_source() {
        // 768x1024 source -> 512x512 model
        // height-limited: h=512, w=512*0.75=384
        assert_eq!(fit_to_model_dimensions(768, 1024, 512, 512), (384, 512));
    }

    #[test]
    fn fit_identity() {
        assert_eq!(
            fit_to_model_dimensions(1024, 1024, 1024, 1024),
            (1024, 1024)
        );
    }

    #[test]
    fn fit_extreme_landscape() {
        // 3840x720 -> 1024x1024 model
        // width-limited: w=1024, h=1024/5.333=192
        assert_eq!(fit_to_model_dimensions(3840, 720, 1024, 1024), (1024, 192));
    }

    #[test]
    fn fit_non_square_model_bounds() {
        // 1920x1080 -> 1024x768 model
        // src_ratio=1.778, model_ratio=1.333, width-limited: w=1024, h=1024/1.778=575.8 -> 576
        assert_eq!(fit_to_model_dimensions(1920, 1080, 1024, 768), (1024, 576));
    }

    #[test]
    fn fit_dimensions_are_16px_aligned() {
        let (w, h) = fit_to_model_dimensions(1000, 600, 512, 512);
        assert!(w % 16 == 0, "width {w} must be 16px aligned");
        assert!(h % 16 == 0, "height {h} must be 16px aligned");
    }

    #[test]
    fn fit_within_megapixel_limit() {
        let (w, h) = fit_to_model_dimensions(4096, 4096, 2048, 2048);
        let pixels = w as u64 * h as u64;
        assert!(
            pixels <= MAX_PIXELS,
            "{}x{} = {} pixels exceeds limit",
            w,
            h,
            pixels
        );
    }

    #[test]
    fn fit_tiny_source_gets_model_native() {
        // 64x64 source -> 1024x1024 model
        assert_eq!(fit_to_model_dimensions(64, 64, 1024, 1024), (1024, 1024));
    }

    #[test]
    fn fit_to_model_dimensions_aligned_rounds_to_the_models_grid() {
        // 1617x1000 into the 5B's 1280x704 canvas is height-limited:
        // h=704, w=704*1.617=1138.4 — which floors differently per grid.
        assert_eq!(
            fit_to_model_dimensions_aligned(1617, 1000, 1280, 704, 32),
            (1120, 704)
        );
        assert_eq!(
            fit_to_model_dimensions_aligned(1617, 1000, 1280, 704, 16),
            (1136, 704)
        );
        // The family-only helper stays the /16 compatibility path.
        assert_eq!(fit_to_model_dimensions(1617, 1000, 1280, 704), (1136, 704));
    }

    #[test]
    fn fit_to_target_area_preserves_ratio_and_alignment() {
        let (w, h) = fit_to_target_area(1600, 900, 1024 * 1024, 16);
        assert_eq!((w, h), (1360, 768));
    }

    // ── LoRA validation tests ──────────────────────────────────────────────

    /// Build a FLUX-model request — the only family that supports LoRAs
    /// today. Tests that exercise LoRA value-validation (scale, extension)
    /// must use a LoRA-capable family or they fail on the upstream
    /// family-gate before the value check can trip.
    fn valid_flux_req() -> GenerateRequest {
        GenerateRequest {
            model: "flux-dev".to_string(),
            ..valid_req()
        }
    }

    const IDENTITY_PNG_1X1: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89,
    ];

    /// True on every build: identity validation is delegated to
    /// `crate::identity` and an ordinary request is untouched by it.
    #[test]
    fn generate_request_validation_leaves_ordinary_requests_alone() {
        let mut plain = valid_req();
        plain.model = "flux-dev:q8".to_string();
        validate_generate_request_with_family(&plain, None).expect("no identity fields");
        plain.steps = 0;
        assert!(validate_generate_request_with_family(&plain, None)
            .unwrap_err()
            .contains("steps must be >= 1"));
    }

    /// The shared validator delegates to `crate::identity` — proving the
    /// wiring here, not restating the rules, which are table-driven there.
    ///
    /// Written for both values of `IDENTITY_RUNTIME_READY`: while the runtime
    /// adapter is pending, the delegation is proven by the pending refusal
    /// arriving through this entry point instead.
    #[cfg(feature = "pulid")]
    #[test]
    fn generate_request_validation_enforces_identity_conditioning() {
        let mut req = valid_req();
        req.model = "flux-dev".to_string();
        req.id_image = Some(IDENTITY_PNG_1X1.to_vec());

        if !crate::identity::IDENTITY_RUNTIME_READY {
            assert_eq!(
                validate_generate_request_with_family(&req, None).unwrap_err(),
                crate::identity::IDENTITY_RUNTIME_PENDING
            );
            let mut bare = valid_req();
            bare.model = "flux-dev:q8".to_string();
            bare.id_weight = Some(1.5);
            assert_eq!(
                validate_generate_request_with_family(&bare, None).unwrap_err(),
                crate::identity::IDENTITY_RUNTIME_PENDING
            );
            return;
        }

        validate_generate_request_with_family(&req, None)
            .expect("flux-dev resolves to the qualified flux-dev:q8");

        req.model = "flux-dev:bf16".to_string();
        let error = validate_generate_request_with_family(&req, None).unwrap_err();
        assert!(error.contains("flux-dev:q4"), "{error}");
        assert!(error.contains("flux-dev:q8"), "{error}");

        // A knob without the reference is refused, never silently ignored.
        let mut bare = valid_req();
        bare.model = "flux-dev:q8".to_string();
        bare.id_weight = Some(1.5);
        let error = validate_generate_request_with_family(&bare, None).unwrap_err();
        assert!(
            error.contains("id_image (or id_images) is required"),
            "{error}"
        );
    }

    /// Whatever the feature, a build that cannot execute identity refuses the
    /// request through the shared validator rather than rendering a print
    /// that silently has no face in it.
    #[test]
    fn generate_request_validation_never_admits_identity_it_cannot_execute() {
        if crate::identity::identity_runtime_available() {
            return;
        }
        let mut req = valid_req();
        req.model = "flux-dev:q8".to_string();
        req.id_image = Some(IDENTITY_PNG_1X1.to_vec());
        let error = validate_generate_request_with_family(&req, None).unwrap_err();
        assert!(
            error == crate::identity::IDENTITY_BUILD_UNSUPPORTED
                || error == crate::identity::IDENTITY_RUNTIME_PENDING,
            "unexpected refusal: {error}"
        );
    }

    /// Without the feature the shared validator refuses the request and names
    /// the missing build support, so no print renders silently face-less.
    #[cfg(not(feature = "pulid"))]
    #[test]
    fn generate_request_validation_refuses_identity_without_the_adapter() {
        let mut req = valid_req();
        req.model = "flux-dev:q8".to_string();
        req.id_image = Some(IDENTITY_PNG_1X1.to_vec());
        let error = validate_generate_request_with_family(&req, None).unwrap_err();
        assert_eq!(error, crate::identity::IDENTITY_BUILD_UNSUPPORTED);

        // Same refusal for a bare knob — the build, not the field, is why.
        let mut bare = valid_req();
        bare.model = "flux-dev:q8".to_string();
        bare.id_start_step = Some(2);
        assert_eq!(
            validate_generate_request_with_family(&bare, None).unwrap_err(),
            crate::identity::IDENTITY_BUILD_UNSUPPORTED
        );
    }

    #[test]
    fn lora_none_valid() {
        let req = valid_req();
        assert!(req.lora.is_none());
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn lora_scale_too_low_rejected() {
        let mut req = valid_flux_req();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: -0.1,

            expert: None,
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("lora scale"),
            "expected lora scale error: {err}"
        );
    }

    #[test]
    fn lora_scale_too_high_rejected() {
        let mut req = valid_flux_req();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 2.1,

            expert: None,
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("lora scale"),
            "expected lora scale error: {err}"
        );
    }

    #[test]
    fn lora_scale_boundary_valid() {
        for scale in [0.0, 1.0, 2.0] {
            let mut req = valid_flux_req();
            req.lora = Some(crate::LoraWeight {
                path: "adapter.safetensors".to_string(),
                scale,

                expert: None,
            });
            assert!(
                validate_generate_request(&req).is_ok(),
                "scale={scale} should be valid"
            );
        }
    }

    #[test]
    fn lora_path_not_found_passes_validation() {
        // Path existence is checked at the inference layer, not validation,
        // so remote LoRA paths (server-side files) work correctly.
        let mut req = valid_flux_req();
        req.lora = Some(crate::LoraWeight {
            path: "/nonexistent/path/adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn lora_wrong_extension_rejected() {
        let mut req = valid_flux_req();
        req.lora = Some(crate::LoraWeight {
            path: "/some/path/adapter.bin".to_string(),
            scale: 1.0,

            expert: None,
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("safetensors"),
            "expected safetensors error: {err}"
        );
    }

    fn valid_sdxl_req() -> GenerateRequest {
        // Pick a real manifest-known SDXL name so `model_family` resolves to
        // `sdxl`. The test surface mirrors `valid_flux_req` / `valid_ltx2_req`.
        GenerateRequest {
            model: "sdxl-base:fp16".to_string(),
            ..valid_req()
        }
    }

    /// SDXL gained LoRA support in Wave 1 of the LoRA-all-families work —
    /// `mold-inference::sdxl::lora` wraps the UNet `VarBuilder` with an
    /// `SdxlLoraBackend` that merges `W' = W + scale·(B @ A)` on the fly.
    /// The validator must now accept LoRAs on SDXL.
    #[test]
    fn lora_on_sdxl_accepted() {
        let mut req = valid_sdxl_req();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        assert!(
            validate_generate_request(&req).is_ok(),
            "SDXL + LoRA must pass validation now that sdxl/lora.rs is live"
        );
    }

    #[test]
    fn loras_plural_on_sdxl_accepted() {
        let mut req = valid_sdxl_req();
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "a.safetensors".to_string(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "b.safetensors".to_string(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "SDXL + plural LoRAs (multi-LoRA stack) must pass validation"
        );
    }

    /// Wan gained LoRA support with the A14B Lightning distills (#747) and
    /// community `.diff`/`.diff_b` deltas (#781) — `wan/lora.rs` merges pairs
    /// and deltas on both the safetensors and GGUF weight paths — but this
    /// gate was never updated, so the server 400'd every explicit wan LoRA
    /// request while the engine loaded the same files happily.
    #[test]
    fn lora_on_wan_accepted() {
        let mut req = valid_req();
        req.model = "wan21-t2v-1.3b".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(16);
        req.frames = Some(33);
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        assert!(
            validate_generate_request(&req).is_ok(),
            "Wan + LoRA must pass validation now that wan/lora.rs is live"
        );

        req.lora = None;
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "a.safetensors".to_string(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "b.safetensors".to_string(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "Wan + plural LoRAs (multi-LoRA stack) must pass validation"
        );
    }

    #[test]
    fn loras_plural_on_flux_valid() {
        // Multi-LoRA is supported on FLUX. The validator must not block
        // the plural form just because the singular form already gates.
        let mut req = valid_flux_req();
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "a.safetensors".into(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(validate_generate_request(&req).is_ok());
    }

    fn valid_ltx2_req() -> GenerateRequest {
        GenerateRequest {
            model: "ltx-2-19b-distilled:fp8".to_string(),
            output_format: Some(OutputFormat::Mp4),
            ..valid_req()
        }
    }

    #[test]
    fn lora_on_ltx2_accepted() {
        // LTX-2 has a full LoRA engine path (ltx2/lora.rs) — the validator
        // must not block it.
        let mut req = valid_ltx2_req();
        req.lora = Some(crate::LoraWeight {
            path: "LTX2.3_Crisp_Enhance.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        assert!(
            validate_generate_request(&req).is_ok(),
            "LTX-2 + LoRA must pass validation"
        );
    }

    #[test]
    fn loras_plural_on_ltx2_accepted() {
        // The loras-plural path routes through the same gate; confirm LTX-2
        // passes there too.
        let mut req = valid_ltx2_req();
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "a.safetensors".into(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "LTX-2 + loras plural must pass validation"
        );
    }

    fn valid_zimage_req() -> GenerateRequest {
        GenerateRequest {
            model: "z-image-turbo:bf16".to_string(),
            ..valid_req()
        }
    }

    fn valid_sd3_req() -> GenerateRequest {
        GenerateRequest {
            model: "sd3.5-large".to_string(),
            ..valid_req()
        }
    }

    #[test]
    fn lora_on_sd3_accepted() {
        // SD3.5 has a full LoRA engine path (sd3/lora.rs) — the validator
        // must not block it.
        let mut req = valid_sd3_req();
        req.lora = Some(crate::LoraWeight {
            path: "sd35_style.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        assert!(
            validate_generate_request(&req).is_ok(),
            "SD3 + LoRA must pass validation: {:?}",
            validate_generate_request(&req)
        );
    }

    #[test]
    fn loras_plural_on_sd3_accepted() {
        let mut req = valid_sd3_req();
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "a.safetensors".into(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "SD3 + loras plural must pass validation"
        );
    }

    #[test]
    fn lora_rejection_message_lists_sd3() {
        // The rejection message must enumerate every supported family.
        // wuerstchen has no LoRA path so the request is rejected; the message
        // must include SD3 in the supported list.
        let mut req = valid_req();
        req.model = "wuerstchen-c".to_string();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.to_lowercase().contains("sd3"),
            "rejection message must list SD3 alongside FLUX/LTX-2: {err}"
        );
    }

    #[test]
    fn lora_on_zimage_accepted() {
        // Z-Image grew a LoRA engine path (zimage/lora.rs) — the validator
        // must let it through.
        let mut req = valid_zimage_req();
        req.lora = Some(crate::LoraWeight {
            path: "NSFW_master_ZIT_000017532.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        assert!(
            validate_generate_request(&req).is_ok(),
            "Z-Image + LoRA must pass validation"
        );
    }

    #[test]
    fn loras_plural_on_zimage_accepted() {
        let mut req = valid_zimage_req();
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "a.safetensors".into(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "Z-Image + loras plural must pass validation"
        );
    }

    #[test]
    fn lora_on_flux2_accepted() {
        // Flux.2 has a full LoRA engine path (flux2/lora.rs) — the validator
        // must not block it. The validator only sees the family resolved
        // from the model name; Flux.2 LoRAs from Civitai (cv:2682864 and
        // siblings) reach this code via the `family_hint` carried by the
        // catalog, but a stable model name like `flux2-klein` works the
        // same way.
        let mut req = valid_req();
        req.model = "flux2-klein".to_string();
        req.lora = Some(crate::LoraWeight {
            path: "DarkKlein9b.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        assert!(
            validate_generate_request(&req).is_ok(),
            "Flux.2 + LoRA must pass validation"
        );
    }

    #[test]
    fn loras_plural_on_flux2_accepted() {
        // The plural loras stack must also pass on Flux.2.
        let mut req = valid_req();
        req.model = "flux2-klein-9b".to_string();
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "lora-a.safetensors".into(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "lora-b.safetensors".into(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "Flux.2 + loras plural must pass validation"
        );
    }

    #[test]
    fn lora_on_unsupported_family_lists_sdxl_in_message() {
        // SD3 / Qwen-Image still lack a LoRA engine path. The validator must
        // reject and the message must enumerate every supported family so
        // the user knows what to pick instead.
        let mut req = valid_req();
        req.model = "wuerstchen-c".to_string();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.to_lowercase().contains("flux"),
            "error must mention FLUX: {err}"
        );
        assert!(
            err.to_lowercase().contains("flux.2") || err.to_lowercase().contains("flux2"),
            "error must mention Flux.2: {err}"
        );
        assert!(
            err.to_lowercase().contains("ltx-2") || err.to_lowercase().contains("ltx2"),
            "error must mention LTX-2: {err}"
        );
        assert!(
            err.to_lowercase().contains("sdxl"),
            "error must mention SDXL: {err}"
        );
        assert!(
            err.to_lowercase().contains("qwen-image"),
            "error must mention Qwen-Image: {err}"
        );
    }

    /// Qwen-Image gained LoRA support in feat/lora-all-families. The
    /// validator must let `qwen-image` through.
    #[test]
    fn lora_on_qwen_image_accepted() {
        let mut req = valid_req();
        req.model = "qwen-image-2512".to_string();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        assert!(
            validate_generate_request(&req).is_ok(),
            "Qwen-Image + LoRA must pass validation",
        );
    }

    /// `qwen-image-edit` shares the LoRA family gate with `qwen-image`.
    /// The edit family also requires a target image separately, so this
    /// test exercises just the LoRA gate by inspecting the rejection
    /// message: it must NOT mention LoRA when the only non-LoRA failure
    /// is the missing target image.
    #[test]
    fn lora_on_qwen_image_edit_passes_lora_gate() {
        let mut req = valid_req();
        req.model = "qwen-image-edit-2511:q4".to_string();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        // The request fails on its target-image requirement, but the
        // LoRA gate is permissive.
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            !err.to_lowercase().contains("lora"),
            "LoRA gate must not reject qwen-image-edit; remaining failure should be on the target image: {err}",
        );
        assert!(
            err.contains("Add a Target image"),
            "expected the only failure to be the target-image requirement: {err}",
        );
    }

    #[test]
    fn loras_plural_on_qwen_image_accepted() {
        let mut req = valid_req();
        req.model = "qwen-image-2512".to_string();
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "a.safetensors".into(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "Qwen-Image + multi-LoRA must pass validation",
        );
    }

    #[test]
    fn lora_on_unknown_family_still_rejected() {
        // family: None (no manifest match) must still produce an error.
        let mut req = valid_req();
        req.model = "some-unknown-model-xyz".to_string();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            !err.is_empty(),
            "unknown family with LoRA must produce an error: {err}"
        );
    }

    /// SD1.5 LoRA support landed in `crates/mold-inference/src/sd15/lora.rs` —
    /// the validator must accept it, just like FLUX and LTX-2.
    #[test]
    fn lora_on_sd15_accepted() {
        let mut req = valid_req();
        req.model = "sd15:fp16".to_string();
        req.width = 512;
        req.height = 512;
        req.guidance = 7.0;
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 0.8,

            expert: None,
        });
        assert!(
            validate_generate_request(&req).is_ok(),
            "SD1.5 + LoRA must pass validation"
        );
    }

    /// The plural `loras` form must accept SD1.5 too — the gate must apply
    /// uniformly to both shapes.
    #[test]
    fn loras_plural_on_sd15_accepted() {
        let mut req = valid_req();
        req.model = "sd15:fp16".to_string();
        req.width = 512;
        req.height = 512;
        req.guidance = 7.0;
        req.loras = Some(vec![
            crate::LoraWeight {
                path: "a.safetensors".into(),
                scale: 0.8,

                expert: None,
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,

                expert: None,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "SD1.5 + loras plural must pass validation"
        );
    }

    /// The rejection message lists every supported family; SDXL still isn't
    /// supported, so a SDXL request with a LoRA should mention SD1.5 in the
    /// list of available alternatives.
    #[test]
    fn lora_on_sdxl_message_now_lists_sd15() {
        let mut req = valid_req();
        req.model = "sdxl".to_string();
        req.lora = Some(crate::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        });
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.to_lowercase().contains("sd1.5")
                || err.to_lowercase().contains("sd15")
                || err.to_lowercase().contains("sd 1.5"),
            "error must list SD1.5 as a supported family: {err}"
        );
    }

    // ── dimension_warning tests ────────────────────────────────────────────

    #[test]
    fn dimension_warning_matching_returns_none() {
        assert!(dimension_warning(1024, 1024, "flux").is_none());
        assert!(dimension_warning(512, 512, "sd15").is_none());
        assert!(dimension_warning(1024, 1024, "sdxl").is_none());
        assert!(dimension_warning(1024, 1024, "wuerstchen").is_none());
    }

    #[test]
    fn dimension_warning_non_matching_returns_some() {
        let warning = dimension_warning(256, 256, "flux");
        assert!(warning.is_some());
        let msg = warning.unwrap();
        assert!(msg.contains("256x256"), "should mention requested dims");
        assert!(msg.contains("flux"), "should mention model family");
        assert!(msg.contains("Suggested"), "should include suggestions");
    }

    #[test]
    fn dimension_warning_unknown_family_returns_none() {
        assert!(dimension_warning(256, 256, "unknown-model").is_none());
    }

    #[test]
    fn dimension_warning_empty_family_returns_none() {
        assert!(dimension_warning(512, 512, "").is_none());
    }

    #[test]
    fn dimension_warning_sd15_at_1024_warns() {
        let warning = dimension_warning(1024, 1024, "sd15");
        assert!(warning.is_some(), "SD1.5 at 1024x1024 should warn");
        assert!(warning.unwrap().contains("512x512"));
    }

    #[test]
    fn dimension_warning_sdxl_buckets_accepted() {
        for (w, h) in recommended_dimensions("sdxl") {
            assert!(
                dimension_warning(*w, *h, "sdxl").is_none(),
                "SDXL bucket {w}x{h} should not warn"
            );
        }
    }

    #[test]
    fn dimension_warning_qwen_image_uses_upstream_aspect_presets() {
        assert_eq!(recommended_dimensions("qwen-image").len(), 7);
        assert_eq!(dimension_warning(1328, 1328, "qwen-image"), None);
        assert_eq!(dimension_warning(1664, 928, "qwen-image"), None);
        assert_eq!(dimension_warning(928, 1664, "qwen-image"), None);
        assert!(dimension_warning(512, 512, "qwen-image").is_some());
    }

    #[test]
    fn dimension_warning_qwen_image_edit_reuses_qwen_dimensions() {
        assert_eq!(
            recommended_dimensions("qwen-image-edit"),
            recommended_dimensions("qwen-image")
        );
        assert_eq!(dimension_warning(1328, 1328, "qwen-image-edit"), None);
    }

    #[test]
    fn dimension_warning_flux2_uses_flux_dims() {
        assert_eq!(
            recommended_dimensions("flux2"),
            recommended_dimensions("flux"),
            "flux2 should share FLUX dimensions"
        );
    }

    #[test]
    fn every_family_native_in_recommendations() {
        // Each family with a qualified recommendation set includes its native
        // resolution. Z-Image and Qwen both expose their qualified upstream
        // aspect sets through the same shared profile registry.
        let families = &[
            ("sd15", 512, 512),
            ("sdxl", 1024, 1024),
            ("sd3", 1024, 1024),
            ("flux", 1024, 1024),
            ("flux2", 1024, 1024),
            ("wuerstchen", 1024, 1024),
            ("ltx-video", 768, 512),
            ("minimax-h3", 1344, 768),
            ("z-image", 1024, 1024),
            ("qwen-image", 1328, 1328),
            ("qwen-image-edit", 1328, 1328),
        ];
        for (family, w, h) in families {
            let dims = recommended_dimensions(family);
            assert!(
                dims.contains(&(*w, *h)),
                "{family} native {w}x{h} missing from recommended list"
            );
        }
    }

    #[test]
    fn h3_recommendations_are_the_official_product_ratios_on_the_oracle_canvas() {
        assert_eq!(
            recommended_dimensions(crate::minimax_h3::FAMILY),
            &[
                (1536, 672),
                (1344, 768),
                (1024, 768),
                (768, 768),
                (768, 1024),
                (768, 1344),
            ]
        );
    }

    #[test]
    fn dimension_warning_message_format() {
        let msg = dimension_warning(800, 600, "sd15").unwrap();
        assert!(msg.contains("800x600"));
        assert!(msg.contains("sd15"));
        assert!(msg.contains("Suggested:"));
        // Should list known alternatives
        assert!(msg.contains("512x512"));
    }

    #[test]
    fn dimension_warning_truncates_long_lists() {
        // SDXL has 9 buckets but warning should show at most 4 + "N total"
        let msg = dimension_warning(800, 600, "sdxl").unwrap();
        assert!(msg.contains("total"), "long lists should show total count");
    }

    // ── validate_upscale_request tests ────────────────────────────────────

    fn valid_upscale_req() -> crate::UpscaleRequest {
        crate::UpscaleRequest {
            model: "real-esrgan-x4plus:fp16".to_string(),
            image: png_bytes(),
            output_format: crate::OutputFormat::Png,
            tile_size: None,
            metadata: None,
        }
    }

    #[test]
    fn upscale_valid_request_passes() {
        assert!(validate_upscale_request(&valid_upscale_req()).is_ok());
    }

    #[test]
    fn upscale_empty_model_rejected() {
        let mut req = valid_upscale_req();
        req.model = "  ".to_string();
        assert!(validate_upscale_request(&req)
            .unwrap_err()
            .contains("model"));
    }

    #[test]
    fn upscale_empty_image_rejected() {
        let mut req = valid_upscale_req();
        req.image = vec![];
        assert!(validate_upscale_request(&req)
            .unwrap_err()
            .contains("empty"));
    }

    #[test]
    fn upscale_invalid_image_format_rejected() {
        let mut req = valid_upscale_req();
        req.image = vec![0x00, 0x01, 0x02, 0x03];
        assert!(validate_upscale_request(&req)
            .unwrap_err()
            .contains("PNG or JPEG"));
    }

    #[test]
    fn upscale_jpeg_accepted() {
        let mut req = valid_upscale_req();
        req.image = jpeg_bytes();
        assert!(validate_upscale_request(&req).is_ok());
    }

    #[test]
    fn upscale_tile_size_too_small_rejected() {
        let mut req = valid_upscale_req();
        req.tile_size = Some(32);
        assert!(validate_upscale_request(&req)
            .unwrap_err()
            .contains("tile_size"));
    }

    #[test]
    fn upscale_tile_size_zero_accepted() {
        let mut req = valid_upscale_req();
        req.tile_size = Some(0);
        assert!(validate_upscale_request(&req).is_ok());
    }

    #[test]
    fn upscale_tile_size_64_accepted() {
        let mut req = valid_upscale_req();
        req.tile_size = Some(64);
        assert!(validate_upscale_request(&req).is_ok());
    }

    #[test]
    fn upscale_tile_size_none_accepted() {
        let req = valid_upscale_req();
        assert!(validate_upscale_request(&req).is_ok());
    }

    #[test]
    fn built_in_ic_lora_control_requires_video_pipeline_and_reserves_a_stack_slot() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = Some(crate::OutputFormat::Mp4);
        req.frames = Some(97);
        req.ic_lora_control = Some("union".to_string());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("pipeline=ic-lora"));

        req.pipeline = Some(crate::Ltx2PipelineMode::IcLora);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("source_video"));
        req.source_video_path = Some("/guides/canny.mp4".to_string());
        assert!(validate_generate_request(&req).is_ok());

        req.loras = Some(
            (0..4)
                .map(|index| crate::LoraWeight {
                    path: format!("/loras/{index}.safetensors"),
                    scale: 1.0,

                    expert: None,
                })
                .collect(),
        );
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("four-LoRA"));
    }

    // ── lip-dub ─────────────────────────────────────────────────────────────

    fn lip_dub_req() -> GenerateRequest {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.width = 1216;
        req.height = 704;
        req.pipeline = Some(Ltx2PipelineMode::LipDub);
        req.ic_lora_control = Some("lipdub".to_string());
        req.source_video_path = Some("/clips/speaker.mp4".to_string());
        req
    }

    #[test]
    fn snap_frames_to_8k1_rounds_down_never_up() {
        // Exactly on the grid stays put.
        for on_grid in [1, 9, 17, 97, 121, 481] {
            assert_eq!(super::snap_frames_to_8k1(on_grid), on_grid);
        }
        // Everything between two grid points falls back to the lower one, so a
        // dub never asks for frames the reference video does not have.
        assert_eq!(super::snap_frames_to_8k1(2), 1);
        assert_eq!(super::snap_frames_to_8k1(8), 1);
        assert_eq!(super::snap_frames_to_8k1(16), 9);
        assert_eq!(super::snap_frames_to_8k1(96), 89);
        assert_eq!(super::snap_frames_to_8k1(100), 97);
        assert_eq!(super::snap_frames_to_8k1(0), 1);
        // The advertised LTX-2 ceiling is derived from the same snap.
        assert_eq!(super::ltx2_max_frames_on_grid_at_fps(24), 481);
    }

    /// A reference clip that could drive a dub, unless a test says otherwise.
    fn lip_dub_reference(frames: u32, fps: u32) -> super::LipDubReference {
        super::LipDubReference {
            frames,
            fps,
            has_audio: true,
        }
    }

    #[test]
    fn lip_dub_timing_comes_from_the_reference_video() {
        let timing = super::resolve_lip_dub_timing(lip_dub_reference(120, 25), None, None).unwrap();
        assert_eq!(timing.frames, 113);
        assert_eq!(timing.fps, 25);
        assert_eq!(timing.warnings.len(), 1, "{:?}", timing.warnings);
        assert!(timing.warnings[0].contains("113"));

        // Already on the grid at the requested values: nothing to say.
        let timing =
            super::resolve_lip_dub_timing(lip_dub_reference(97, 24), Some(97), Some(24)).unwrap();
        assert_eq!((timing.frames, timing.fps), (97, 24));
        assert!(timing.warnings.is_empty());
    }

    #[test]
    fn lip_dub_timing_overrides_and_reports_conflicting_requests() {
        let timing =
            super::resolve_lip_dub_timing(lip_dub_reference(97, 24), Some(241), Some(30)).unwrap();
        assert_eq!((timing.frames, timing.fps), (97, 24));
        assert_eq!(timing.warnings.len(), 2, "{:?}", timing.warnings);
        assert!(timing.warnings[0].contains("241") && timing.warnings[0].contains("97"));
        assert!(timing.warnings[1].contains("30") && timing.warnings[1].contains("24"));
    }

    #[test]
    fn lip_dub_timing_rejects_unusable_references() {
        assert!(
            super::resolve_lip_dub_timing(lip_dub_reference(97, 0), None, None)
                .unwrap_err()
                .contains("frame rate")
        );
        assert!(
            super::resolve_lip_dub_timing(lip_dub_reference(8, 24), None, None)
                .unwrap_err()
                .contains("too short")
        );
        // A silent reference is refused here, at the request boundary, rather
        // than minutes later when the audio VAE has nothing to encode.
        let silent = super::LipDubReference {
            has_audio: false,
            ..lip_dub_reference(97, 24)
        };
        assert!(super::resolve_lip_dub_timing(silent, None, None)
            .unwrap_err()
            .contains("no audio track"));
    }

    #[test]
    fn lip_dub_requires_a_reference_video_and_the_adapter() {
        let mut req = lip_dub_req();
        req.source_video_path = None;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("source_video"));

        let mut req = lip_dub_req();
        req.ic_lora_control = None;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("ic_lora_control=lipdub"));

        assert!(validate_generate_request(&lip_dub_req()).is_ok());
    }

    #[test]
    fn lip_dub_rejects_dimensions_that_are_not_multiples_of_64() {
        // 1216x704 is fine; 1216x736 is a multiple of 32 but not of 64, which
        // is exactly the case a one-stage-only check would let through.
        let mut req = lip_dub_req();
        req.height = 736;
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("multiples of 64"), "{err}");

        let mut req = lip_dub_req();
        req.width = 1184;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("multiples of 64"));
    }

    #[test]
    fn lip_dub_control_id_routes_to_the_lip_dub_pipeline_not_ic_lora() {
        use crate::ltx2_control::pipeline_for_control_id;
        assert_eq!(pipeline_for_control_id("lipdub"), Ltx2PipelineMode::LipDub);
        assert_eq!(pipeline_for_control_id("LipDub"), Ltx2PipelineMode::LipDub);
        assert_eq!(pipeline_for_control_id("union"), Ltx2PipelineMode::IcLora);

        // Asking for the lip-dub adapter on the generic in-context pipeline is
        // a mistake worth naming: the weights would load and the wrong graph
        // would run.
        let mut req = lip_dub_req();
        req.pipeline = Some(Ltx2PipelineMode::IcLora);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("requires pipeline=lip-dub"));

        let mut req = lip_dub_req();
        req.ic_lora_control = Some("union".to_string());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("requires pipeline=ic-lora"));
    }

    #[test]
    fn lip_dub_rejects_conflicting_conditioning_modes() {
        let mut req = lip_dub_req();
        req.retake_range = Some(crate::TimeRange {
            start_seconds: 0.0,
            end_seconds: 1.0,
        });
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("retake_range"));

        let mut req = lip_dub_req();
        req.keyframes = Some(vec![KeyframeCondition {
            frame: 0,
            image: png_bytes(),
            name: None,
        }]);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("keyframes"));

        // Upscaling would change the output shape out from under the clip the
        // dub has to line up with.
        let mut req = lip_dub_req();
        req.spatial_upscale = Some(crate::Ltx2SpatialUpscale::X2);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("spatial_upscale"));

        let mut req = lip_dub_req();
        req.temporal_upscale = Some(crate::Ltx2TemporalUpscale::X2);
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("temporal_upscale"));
    }

    // ── creation-time filing (tags / collection) ────────────────────────

    #[test]
    fn organization_validation_normalizes_tags_and_a_collection_name() {
        let org = validate_request_organization(
            Some(&[
                "  Smurfs  ".into(),
                "smurfs".into(),
                "".into(),
                "village".into(),
            ]),
            Some(&crate::CollectionRef::by_name("  Smurf   Village ")),
        )
        .unwrap();
        assert_eq!(org.tags, vec!["Smurfs".to_string(), "village".to_string()]);
        assert_eq!(org.collection, Some(Ok("Smurf Village".to_string())));
    }

    #[test]
    fn organization_validation_keeps_an_id_reference_unresolved() {
        let org =
            validate_request_organization(None, Some(&crate::CollectionRef::by_id("  col-1  ")))
                .unwrap();
        assert!(org.tags.is_empty());
        assert_eq!(org.collection, Some(Err("col-1".to_string())));
    }

    /// A name and an id together resolve as the name: the name is what the
    /// print's embedded provenance records, so honouring the id could file
    /// under one collection and record another.
    #[test]
    fn organization_validation_prefers_the_name_when_both_are_present() {
        let org = validate_request_organization(
            None,
            Some(&crate::CollectionRef {
                id: Some("col-1".into()),
                name: Some("Smurf Village".into()),
            }),
        )
        .unwrap();
        assert_eq!(org.collection, Some(Ok("Smurf Village".to_string())));
    }

    #[test]
    fn organization_validation_refuses_a_collection_ref_with_neither_field() {
        for reference in [
            crate::CollectionRef::default(),
            crate::CollectionRef {
                id: Some("   ".into()),
                name: Some("".into()),
            },
        ] {
            let err = validate_request_organization(None, Some(&reference)).unwrap_err();
            assert!(err.contains("name"), "{err}");
            assert!(err.contains("id"), "{err}");
        }
    }

    #[test]
    fn organization_validation_refuses_invalid_tags_and_collection_names() {
        assert!(validate_request_organization(Some(&["ok\u{1b}[0m".into()]), None).is_err());
        assert!(
            validate_request_organization(Some(&["x".repeat(crate::MAX_TAG_CHARS + 1)]), None)
                .is_err()
        );
        let over: Vec<String> = (0..crate::MAX_REQUEST_TAGS + 1)
            .map(|i| format!("t{i}"))
            .collect();
        assert!(validate_request_organization(Some(&over), None).is_err());
        // A name with no ASCII alphanumeric has no slug to merge on.
        assert!(validate_request_organization(
            None,
            Some(&crate::CollectionRef::by_name("日本語"))
        )
        .is_err());
    }

    /// Admission must not merely *check* the filing — it must write the
    /// normalized form back into the request, because the request is what
    /// `OutputMetadata::from_generate_request` embeds while the DB row seeds
    /// through a re-normalizing path. A raw-spelling HTTP client (curl, a
    /// script) would otherwise stamp `[" Smurfs ", "smurfs"]` into provenance
    /// while the row holds `["Smurfs"]`, and Reuse would restore the
    /// duplicates.
    #[test]
    fn materializing_organization_rewrites_the_request_to_what_will_apply() {
        let mut req = valid_req();
        req.tags = Some(vec![
            "  Smurfs  ".into(),
            "smurfs".into(),
            "".into(),
            " village  green ".into(),
        ]);
        req.collection = Some(crate::CollectionRef::by_name("  Smurf   Village  "));

        materialize_request_organization(&mut req).unwrap();

        assert_eq!(
            req.tags.as_deref(),
            Some(["Smurfs".to_string(), "village green".to_string()].as_slice()),
            "the request now carries exactly the tags the row will hold"
        );
        assert_eq!(
            req.collection,
            Some(crate::CollectionRef::by_name("Smurf Village")),
            "and the collection name the row will resolve by slug"
        );

        // The whole point: embedded provenance and the applied row agree.
        let metadata = crate::OutputMetadata::from_generate_request(&req, 7, None, "test");
        assert_eq!(
            metadata.tags.as_deref(),
            Some(["Smurfs".to_string(), "village green".to_string()].as_slice())
        );
        assert_eq!(metadata.collection.as_deref(), Some("Smurf Village"));
    }

    /// An already-canonical request is left byte-identical — materialization
    /// must not churn what every first-party client already sends.
    #[test]
    fn materializing_organization_leaves_a_canonical_request_untouched() {
        let mut req = valid_req();
        req.tags = Some(vec!["Smurfs".into(), "village green".into()]);
        req.collection = Some(crate::CollectionRef::by_name("Smurf Village"));
        let before = req.clone();

        materialize_request_organization(&mut req).unwrap();
        assert_eq!(req.tags, before.tags);
        assert_eq!(req.collection, before.collection);
    }

    /// An unfiled request gains nothing — never an empty list that would
    /// stamp `"tags": []` into every print's provenance.
    #[test]
    fn materializing_organization_leaves_an_unfiled_request_absent() {
        let mut req = valid_req();
        materialize_request_organization(&mut req).unwrap();
        assert_eq!(req.tags, None);
        assert_eq!(req.collection, None);

        // A list that normalizes away entirely collapses to absent, not `[]`.
        req.tags = Some(vec!["".into(), "   ".into()]);
        materialize_request_organization(&mut req).unwrap();
        assert_eq!(req.tags, None);
    }

    /// An `{id}` reference is left for the server to resolve against its own
    /// collection table — materialization normalizes names, it does not
    /// invent one.
    #[test]
    fn materializing_organization_preserves_an_unresolved_id_reference() {
        let mut req = valid_req();
        req.collection = Some(crate::CollectionRef::by_id("col-1"));
        materialize_request_organization(&mut req).unwrap();
        assert_eq!(req.collection, Some(crate::CollectionRef::by_id("col-1")));
    }

    #[test]
    fn materializing_organization_refuses_what_validation_refuses() {
        let mut req = valid_req();
        req.tags = Some(vec!["nul\0".into()]);
        assert!(materialize_request_organization(&mut req).is_err());
    }

    #[test]
    fn organization_validation_accepts_absence() {
        let org = validate_request_organization(None, None).unwrap();
        assert_eq!(org, RequestOrganization::default());
        assert_eq!(
            validate_request_organization(Some(&[]), None).unwrap(),
            RequestOrganization::default()
        );
    }

    /// Admission runs the filing check, so a bad tag is refused before any
    /// model work is paid for.
    #[test]
    fn generate_request_validation_refuses_bad_filing() {
        let mut req = valid_req();
        req.tags = Some(vec!["nul\0".into()]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("control characters"), "{err}");

        let mut req = valid_req();
        req.collection = Some(crate::CollectionRef::default());
        assert!(validate_generate_request(&req).is_err());

        // Valid filing passes through untouched.
        let mut req = valid_req();
        req.tags = Some(vec!["smurfs".into()]);
        req.collection = Some(crate::CollectionRef::by_name("Smurf Village"));
        assert!(validate_generate_request(&req).is_ok());
    }
}
