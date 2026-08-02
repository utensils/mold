use crate::{
    GenerateRequest, KeyframeCondition, LoraWeight, Ltx2GuidanceOverrides, Ltx2PipelineMode,
    OutputFormat, UpscaleRequest,
};

/// Maximum total pixels allowed (~1.8 megapixels). Qwen-Image trains at ~1.6MP
/// (1328x1328), other models at ≤1MP. Headroom for non-square aspect ratios.
pub const MAX_PIXELS: u64 = 1_800_000;
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

/// Default pixel-frame overlap for `extend_video`, matching the chain
/// motion-tail default so an extend seam and a sequence seam behave the same.
/// 17 pixel frames is three LTX-2 latent frames under the VAE's 8x causal
/// temporal compression.
pub const DEFAULT_EXTEND_OVERLAP_FRAMES: u32 = 17;

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

/// Per-family single-request frame ceiling at `fps` — the value `/api/models`
/// advertises as `max_frames`. Must stay in agreement with
/// `validate_generate_request`'s rejections, which consume this helper.
///
/// LTX-2's ceiling is a duration, so it moves with fps; every other video
/// family reports the flat global ceiling.
pub fn max_frames_for_family_at_fps(family: &str, fps: u32) -> Option<u32> {
    match family {
        "ltx2" => Some(ltx2_max_frames_at_fps(fps)),
        "ltx-video" => Some(MAX_FRAMES_GLOBAL),
        _ => None,
    }
}

/// `max_frames_for_family_at_fps` at each family's default fps, for callers
/// that have no per-model fps to hand.
pub fn max_frames_for_family(family: &str) -> Option<u32> {
    max_frames_for_family_at_fps(family, LTX2_DEFAULT_FPS)
}

/// Single-request runtime ceiling in seconds for families whose real limit is
/// a duration. `None` means the family's ceiling is a plain frame count.
pub fn max_runtime_seconds_for_family(family: &str) -> Option<u32> {
    (family == "ltx2").then_some(LTX2_MAX_RUNTIME_SECONDS)
}

/// fps-independent frame guard, paired with `max_runtime_seconds_for_family`.
pub fn max_frames_absolute_for_family(family: &str) -> Option<u32> {
    (family == "ltx2").then_some(LTX2_MAX_FRAMES_ABSOLUTE)
}

/// Frame-count grid for a family: valid counts are `k * step + 1`. The value
/// `/api/models` advertises as `frame_step`; the validator consumes it.
pub fn frame_step_for_family(family: &str) -> Option<u32> {
    matches!(family, "ltx2" | "ltx-video").then_some(8)
}

fn megapixel_limit_label() -> String {
    format!("{:.1}MP", MAX_PIXELS as f64 / 1_000_000.0)
}

/// Required pixel grid for a generation family.
///
/// LTX video VAEs compress spatial dimensions by 32. Every other current
/// family uses the shared 16px generation grid.
pub fn dimension_alignment_for_family(family: Option<&str>) -> u32 {
    if matches!(family, Some("ltx-video" | "ltx2")) {
        32
    } else {
        16
    }
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
    if width == 0 || height == 0 {
        return Err("width and height must be > 0".to_string());
    }

    let alignment = dimension_alignment_for_family(family);
    if !width.is_multiple_of(alignment) || !height.is_multiple_of(alignment) {
        let family_label = family
            .filter(|value| !value.is_empty())
            .map(|value| format!(" for {value} models"))
            .unwrap_or_default();
        return Err(format!(
            "width ({width}) and height ({height}) must be multiples of {alignment}{family_label}"
        ));
    }

    let pixels = width as u64 * height as u64;
    if pixels > MAX_PIXELS {
        return Err(format!(
            "{width}x{height} = {:.2} megapixels exceeds the {} limit (VAE VRAM constraint)",
            pixels as f64 / 1_000_000.0,
            megapixel_limit_label()
        ));
    }

    Ok(())
}

fn mib_label(bytes: usize) -> String {
    format!("{:.0} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// Clamp dimensions to fit within the megapixel limit, preserving aspect ratio.
/// Both dimensions are rounded down to multiples of 16.
/// Returns the original dimensions unchanged if already within limits.
pub fn clamp_to_megapixel_limit(w: u32, h: u32) -> (u32, u32) {
    let pixels = w as u64 * h as u64;
    if pixels <= MAX_PIXELS {
        return (w, h);
    }
    let scale = (MAX_PIXELS as f64 / pixels as f64).sqrt();
    let new_w = ((w as f64 * scale) as u32 / 16) * 16;
    let new_h = ((h as f64 * scale) as u32 / 16) * 16;
    // Ensure we don't produce zero dimensions
    (new_w.max(16), new_h.max(16))
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
pub fn fit_to_model_dimensions(src_w: u32, src_h: u32, model_w: u32, model_h: u32) -> (u32, u32) {
    let src_ratio = src_w as f64 / src_h as f64;
    let model_ratio = model_w as f64 / model_h as f64;

    let (w, h) = if src_ratio > model_ratio {
        // Source is wider: width-limited
        (model_w as f64, model_w as f64 / src_ratio)
    } else {
        // Source is taller or same: height-limited
        (model_h as f64 * src_ratio, model_h as f64)
    };

    let w = ((w as u32) / 16 * 16).max(16);
    let h = ((h as u32) / 16 * 16).max(16);
    clamp_to_megapixel_limit(w, h)
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

fn validate_keyframes(
    keyframes: &[KeyframeCondition],
    frames: Option<u32>,
    family: Option<&str>,
) -> Result<(), String> {
    match family {
        Some("ltx2") => {}
        None => {
            return Err(
                "unknown model family; keyframes are only supported for LTX-2 / LTX-2.3 models"
                    .to_string(),
            );
        }
        _ => {
            return Err("keyframes are only supported for LTX-2 / LTX-2.3 models".to_string());
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
/// hard constraints: the overlap has to land on the LTX-2 VAE's `8k+1` causal
/// temporal grid to re-encode cleanly, and it has to be strictly shorter than
/// the rendered clip or the continuation contributes no new frames at all.
fn validate_extend(req: &GenerateRequest, family: Option<&str>) -> Result<(), String> {
    if let Some(video) = &req.extend_video {
        require_ltx2_family(family, "extend_video")?;
        if req.extend_video_path.is_some() {
            return Err("extend_video_path cannot be combined with extend_video".to_string());
        }
        if video.is_empty() {
            return Err("extend_video must not be empty".to_string());
        }
        validate_inline_media_size(video, "extend_video", MAX_INLINE_EXTEND_VIDEO_BYTES)?;
    }
    if let Some(path) = &req.extend_video_path {
        require_ltx2_family(family, "extend_video_path")?;
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

    let overlap = req.effective_extend_overlap_frames();
    if overlap == 0 {
        return Err(
            "extend_overlap_frames must be >= 1 so the continuation has motion context".to_string(),
        );
    }
    if overlap % 8 != 1 {
        return Err(format!(
            "extend_overlap_frames ({overlap}) must be 8k+1 (1, 9, 17, 25, …) so the carryover \
             frames re-encode cleanly through the LTX-2 video VAE's 8x causal grid"
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
/// Qwen-Image (and qwen-image-edit), and Z-Image — `mold-inference`'s
/// per-family `lora.rs` modules are the engine paths that know how to merge
/// low-rank adapters into the base weights. Surfacing the gate at validation
/// produces a clear 400 instead of an opaque inference-layer panic when a
/// user picks an unsupported model family + a LoRA.
fn require_lora_capable_family(family: Option<&str>) -> Result<(), String> {
    match family {
        Some(family) if family_supports_lora(family) => Ok(()),
        Some(other) => Err(format!(
            "LoRA is currently supported for FLUX, Flux.2, LTX-2, SD1.5, SD3, SDXL, Qwen-Image, and Z-Image models; got family {other:?}"
        )),
        None => Err(
            "LoRA requires a known model family — pick a FLUX, Flux.2, LTX-2, SD1.5, SD3, SDXL, Qwen-Image, or Z-Image model first"
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

/// Variant of [`validate_generate_request`] that accepts an explicit family
/// hint. The hint takes precedence over the manifest lookup, letting the HTTP
/// server feed in the catalog-resolved family for `cv:` / `hf:` model IDs.
pub fn validate_generate_request_with_family(
    req: &GenerateRequest,
    family_hint: Option<&str>,
) -> Result<(), String> {
    let family = resolved_family(&req.model, family_hint);

    if req.prompt.trim().is_empty() && prompt_required_for(req, family_hint) {
        return Err("prompt must not be empty".to_string());
    }
    validate_generation_dimensions(req.width, req.height, family)?;
    if req.steps == 0 {
        return Err("steps must be >= 1".to_string());
    }
    if req.steps > 100 {
        return Err(format!("steps ({}) must be <= 100", req.steps));
    }
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
    }
    if let Some(ref loras) = req.loras {
        if loras.is_empty() {
            return Err("loras must not be empty when provided".to_string());
        }
        require_lora_capable_family(family)?;
        for lora in loras {
            validate_lora_weight(lora, "loras")?;
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
            if frames > 1 && (frames - 1) % step != 0 {
                return Err(format!(
                    "frames ({frames}) must be {step}n+1 for current LTX-Video / LTX-2 models (e.g. 9, 17, 25, 33, 41, 49, …)"
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
                let delivered_cap = match req.temporal_upscale {
                    Some(crate::Ltx2TemporalUpscale::X2) => (stage1_cap - 1) * 2 + 1,
                    None => stage1_cap,
                };
                return Err(format!(
                    "frames ({frames}) exceeds the LTX-2 / LTX-2.3 temporal RoPE budget of \
                     {LTX2_MAX_RUNTIME_SECONDS}s: at {fps} fps the ceiling is {delivered_cap} frames. \
                     Raise --fps, lower --frames, or render the shot as a multi-clip sequence"
                ));
            }
        } else if frames > MAX_FRAMES_GLOBAL {
            return Err(format!("frames ({frames}) must be <= {MAX_FRAMES_GLOBAL}"));
        }
    }
    if let Some(keyframes) = &req.keyframes {
        validate_keyframes(keyframes, req.frames, family)?;
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
    }
    if let Some(control) = req.ic_lora_control.as_deref() {
        require_ltx2_family(family, "ic_lora_control")?;
        if control.trim().is_empty() {
            return Err("ic_lora_control must not be empty".to_string());
        }
        if req.pipeline != Some(Ltx2PipelineMode::IcLora) {
            return Err("ic_lora_control requires pipeline=ic-lora".to_string());
        }
        if req.source_video.is_none() && req.source_video_path.is_none() {
            return Err("ic_lora_control requires source_video".to_string());
        }
        let user_loras = usize::from(req.lora.is_some()) + req.loras.as_ref().map_or(0, Vec::len);
        if user_loras + 1 > 4 {
            return Err(
                "ic_lora_control plus custom LoRAs exceeds the four-LoRA stack limit".to_string(),
            );
        }
    }

    if family == Some("ltx2") {
        match req.resolved_output_format() {
            OutputFormat::Gif | OutputFormat::Apng | OutputFormat::Webp | OutputFormat::Mp4 => {}
            _ => return Err("LTX-2 outputs must use mp4, gif, apng, or webp".to_string()),
        }

        if req.enable_audio == Some(true) && req.resolved_output_format() != OutputFormat::Mp4 {
            return Err("audio-enabled LTX-2 outputs must use mp4 format".to_string());
        }

        if req.retake_range.is_some()
            && req.source_video.is_none()
            && req.source_video_path.is_none()
        {
            return Err("retake_range requires source_video to also be provided".to_string());
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
                        return Err("pipeline=a2vid requires audio_file".to_string());
                    }
                }
                Ltx2PipelineMode::Retake => {
                    if req.source_video.is_none() && req.source_video_path.is_none() {
                        return Err("pipeline=retake requires source_video".to_string());
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
                        return Err("pipeline=ic-lora requires source_video".to_string());
                    }
                    if req.ic_lora_control.is_none()
                        && req.lora.is_none()
                        && req.loras.as_ref().is_none_or(Vec::is_empty)
                    {
                        return Err("pipeline=ic-lora requires at least one LoRA".to_string());
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

/// Recommended (width, height) pairs for SD1.5 models (native 512x512).
const SD15_DIMS: &[(u32, u32)] = &[(512, 512), (512, 768), (768, 512), (384, 512), (512, 384)];

/// Official SDXL training buckets from Stability AI (native 1024x1024).
const SDXL_DIMS: &[(u32, u32)] = &[
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
];

/// Recommended dimensions for SD3.5 models (native 1024x1024).
const SD3_DIMS: &[(u32, u32)] = &[
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
];

/// Recommended dimensions for FLUX models (native 1024x1024).
const FLUX_DIMS: &[(u32, u32)] = &[
    (1024, 1024),
    (1024, 768),
    (768, 1024),
    (1024, 576),
    (576, 1024),
    (768, 768),
];

/// Recommended dimensions for Z-Image models (native 1024x1024).
const ZIMAGE_DIMS: &[(u32, u32)] = &[(1024, 1024), (1024, 768), (768, 1024)];

/// Recommended dimensions for Qwen-Image models (native 1328x1328, ~1.76MP max).
/// Supports dynamic resolution — any dims divisible by 16 within the megapixel budget work,
/// but these are the standard aspect-ratio buckets.
const QWEN_IMAGE_DIMS: &[(u32, u32)] = &[
    (1328, 1328), // 1:1 (native)
    (1024, 1024), // 1:1
    (1152, 896),  // 9:7
    (896, 1152),  // 7:9
    (1216, 832),  // 19:13
    (832, 1216),  // 13:19
    (1344, 768),  // 7:4
    (768, 1344),  // 4:7
    (1664, 928),  // ~16:9
    (928, 1664),  // ~9:16
    (768, 768),   // 1:1 (small)
    (512, 512),   // 1:1 (small, fast)
];

/// Recommended dimensions for Wuerstchen models (native 1024x1024).
const WUERSTCHEN_DIMS: &[(u32, u32)] = &[(1024, 1024)];

/// Recommended dimensions for LTX Video models (native 768x512).
/// LTX Video requires dimensions divisible by 32 (patchification).
const LTX_VIDEO_DIMS: &[(u32, u32)] = &[
    (704, 480),  // 22:15 (compact sample bucket)
    (768, 512),  // 3:2 (native)
    (512, 512),  // 1:1
    (1024, 576), // 16:9
    (1216, 704), // 16:9 (LTX-2 19B/22B default)
    (576, 1024), // 9:16
    (768, 768),  // 1:1
    (512, 768),  // 2:3
];

/// Return the list of recommended (width, height) pairs for a model family.
///
/// Returns an empty slice for unknown families, utility models (e.g. `qwen3-expand`),
/// and conditioning models (e.g. ControlNet).
pub fn recommended_dimensions(family: &str) -> &'static [(u32, u32)] {
    match family {
        "sd15" => SD15_DIMS,
        "sdxl" => SDXL_DIMS,
        "sd3" => SD3_DIMS,
        "flux" => FLUX_DIMS,
        "flux2" => FLUX_DIMS,
        "z-image" => ZIMAGE_DIMS,
        "qwen-image" => QWEN_IMAGE_DIMS,
        "qwen-image-edit" => QWEN_IMAGE_DIMS,
        "wuerstchen" => WUERSTCHEN_DIMS,
        "ltx-video" | "ltx2" => LTX_VIDEO_DIMS,
        _ => &[],
    }
}

/// Check if the requested dimensions match any recommended resolution for the model family.
///
/// Returns `None` if the dimensions are recommended or the family has no recommendation list.
/// Returns `Some(warning_message)` with suggested alternatives otherwise.
pub fn dimension_warning(width: u32, height: u32, family: &str) -> Option<String> {
    let dims = recommended_dimensions(family);
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

    fn valid_req() -> GenerateRequest {
        GenerateRequest {
            guidance_overrides: None,
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
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
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
        }
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
            },
            crate::KeyframeCondition {
                frame: 16,
                image: png_bytes(),
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
        assert!(err.contains("LTX-Video / LTX-2"), "got: {err}");
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

    #[test]
    fn extend_is_ltx2_only() {
        let mut req = extend_req();
        req.model = "ltx-video-0.9.6-distilled:bf16".to_string();
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("extend_video"), "got: {err}");
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
        assert_eq!(
            max_frames_for_family("ltx2"),
            Some(ltx2_max_frames_at_fps(LTX2_DEFAULT_FPS))
        );
        assert_eq!(max_frames_for_family_at_fps("ltx2", 12), Some(244));
        assert_eq!(max_frames_for_family_at_fps("ltx-video", 12), Some(257));
        assert_eq!(max_frames_for_family("ltx-video"), Some(257));
        assert_eq!(max_frames_for_family("flux"), None);
        assert_eq!(max_frames_for_family("sdxl"), None);
        assert_eq!(frame_step_for_family("ltx2"), Some(8));
        assert_eq!(frame_step_for_family("ltx-video"), Some(8));
        assert_eq!(frame_step_for_family("flux"), None);

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
        assert!(err.contains("484"), "got: {err}");
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
        assert!(err.contains("124"), "got: {err}");
    }

    #[test]
    fn ltx2_absolute_frame_guard_binds_above_thirty_fps() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = Some(OutputFormat::Mp4);
        req.fps = Some(120);
        req.frames = Some(609); // first 8n+1 value past the 604-frame guard
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains(&LTX2_MAX_FRAMES_ABSOLUTE.to_string()),
            "got: {err}"
        );
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
        });
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn lora_wrong_extension_rejected() {
        let mut req = valid_flux_req();
        req.lora = Some(crate::LoraWeight {
            path: "/some/path/adapter.bin".to_string(),
            scale: 1.0,
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
            },
            crate::LoraWeight {
                path: "b.safetensors".to_string(),
                scale: 0.4,
            },
        ]);
        assert!(
            validate_generate_request(&req).is_ok(),
            "SDXL + plural LoRAs (multi-LoRA stack) must pass validation"
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
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,
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
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,
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
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,
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
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,
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
            },
            crate::LoraWeight {
                path: "lora-b.safetensors".into(),
                scale: 0.4,
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
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,
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
            },
            crate::LoraWeight {
                path: "b.safetensors".into(),
                scale: 0.4,
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
    fn dimension_warning_qwen_image_has_native_resolution() {
        let dims = recommended_dimensions("qwen-image");
        assert!(
            dims.contains(&(1328, 1328)),
            "must include native 1328x1328"
        );
        assert!(dims.contains(&(512, 512)), "must include 512x512");
        assert!(dims.contains(&(1024, 1024)), "must include 1024x1024");
        assert_eq!(dimension_warning(1328, 1328, "qwen-image"), None);
        assert_eq!(dimension_warning(512, 512, "qwen-image"), None);
    }

    #[test]
    fn dimension_warning_qwen_image_edit_reuses_qwen_dimensions() {
        assert_eq!(
            recommended_dimensions("qwen-image-edit"),
            recommended_dimensions("qwen-image")
        );
        assert_eq!(dimension_warning(1024, 1024, "qwen-image-edit"), None);
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
        // Each family's native resolution (from ManifestDefaults) should appear
        // in its recommended list.
        let families = &[
            ("sd15", 512, 512),
            ("sdxl", 1024, 1024),
            ("sd3", 1024, 1024),
            ("flux", 1024, 1024),
            ("flux2", 1024, 1024),
            ("z-image", 1024, 1024),
            ("qwen-image", 1024, 1024),
            ("qwen-image-edit", 1024, 1024),
            ("wuerstchen", 1024, 1024),
            ("ltx-video", 768, 512),
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
                })
                .collect(),
        );
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("four-LoRA"));
    }
}
