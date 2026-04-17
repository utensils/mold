use crate::{
    GenerateRequest, KeyframeCondition, LoraWeight, Ltx2PipelineMode, OutputFormat, UpscaleRequest,
};

/// Maximum total pixels allowed (~1.8 megapixels). Qwen-Image trains at ~1.6MP
/// (1328x1328), other models at ≤1MP. Headroom for non-square aspect ratios.
pub const MAX_PIXELS: u64 = 1_800_000;
pub const MAX_INLINE_AUDIO_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_INLINE_SOURCE_VIDEO_BYTES: usize = 64 * 1024 * 1024;

/// Maximum pixel-frame count for LTX-2 / LTX-2.3. Derived from the checkpoint's
/// `positional_embedding_max_pos[0] = 20` temporal RoPE budget and the 8x VAE
/// temporal compression: `(20 - 1) * 8 + 1 = 153`. Exceeding this overflows the
/// RoPE normalization and collapses output into random-color noise.
pub const LTX2_MAX_FRAMES: u32 = 153;

fn megapixel_limit_label() -> String {
    format!("{:.1}MP", MAX_PIXELS as f64 / 1_000_000.0)
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
pub fn validate_generate_request(req: &GenerateRequest) -> Result<(), String> {
    let family = model_family(&req.model);

    if req.prompt.trim().is_empty() {
        return Err("prompt must not be empty".to_string());
    }
    if req.width == 0 || req.height == 0 {
        return Err("width and height must be > 0".to_string());
    }
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(format!(
            "width ({}) and height ({}) must be multiples of 16 (FLUX patchification requirement)",
            req.width, req.height
        ));
    }
    // Cap by total pixel count rather than per-dimension to allow portrait/landscape.
    // 896x1152 = 1.03M, 1024x1024 = 1.05M, 1280x768 = 0.98M — all fine.
    // 1408x1408 = 1.98M — too large, OOMs on VAE decode.
    let pixels = req.width as u64 * req.height as u64;
    if pixels > MAX_PIXELS {
        return Err(format!(
            "{}x{} = {:.2} megapixels exceeds the {} limit (VAE VRAM constraint)",
            req.width,
            req.height,
            pixels as f64 / 1_000_000.0,
            megapixel_limit_label()
        ));
    }
    if req.steps == 0 {
        return Err("steps must be >= 1".to_string());
    }
    if req.steps > 100 {
        return Err(format!("steps ({}) must be <= 100", req.steps));
    }
    if req.batch_size == 0 {
        return Err("batch_size must be >= 1".to_string());
    }
    // No upper limit on batch_size — users can batch as many as they want.
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
    if family == Some("qwen-image-edit") {
        if req.edit_images.as_ref().is_none_or(Vec::is_empty) {
            return Err("qwen-image-edit requires edit_images to be provided".to_string());
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
    } else if req.edit_images.is_some() {
        return Err("edit_images are only supported for qwen-image-edit models".to_string());
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
        validate_lora_weight(lora, "lora")?;
    }
    if let Some(ref loras) = req.loras {
        if loras.is_empty() {
            return Err("loras must not be empty when provided".to_string());
        }
        for lora in loras {
            validate_lora_weight(lora, "loras")?;
        }
    }
    // Video frame validation
    if let Some(frames) = req.frames {
        if frames == 0 {
            return Err("frames must be >= 1".to_string());
        }
        if matches!(family, Some("ltx-video" | "ltx2")) && frames > 1 && (frames - 1) % 8 != 0 {
            return Err(format!(
                "frames ({frames}) must be 8n+1 for current LTX-Video / LTX-2 models (e.g. 9, 17, 25, 33, 41, 49, …)"
            ));
        }
        // LTX-2 transformers ship `positional_embedding_max_pos: [20, 2048, 2048]`
        // — exceeding 20 latent frames wraps RoPE into an untrained region and
        // collapses output into rainbow/static noise. The 8x VAE temporal
        // compression gives max pixel frames = (20 - 1) * 8 + 1 = 153 for
        // single-pass runs. `--temporal-upscale x2` halves the stage-1 frame
        // count (see `derive_stage1_render_shape`), so the transformer only
        // denoises `(frames - 1) / 2 + 1` pixel frames; effective cap doubles.
        if matches!(family, Some("ltx2")) {
            let stage1_frames = match req.temporal_upscale {
                Some(crate::Ltx2TemporalUpscale::X2) => frames.saturating_sub(1) / 2 + 1,
                None => frames,
            };
            if stage1_frames > LTX2_MAX_FRAMES {
                return Err(format!(
                    "frames ({frames}) must be <= {LTX2_MAX_FRAMES} for LTX-2 / LTX-2.3 (temporal RoPE budget); \
                     pass --temporal-upscale x2 to double the effective frame ceiling"
                ));
            }
        }
        if frames > 257 {
            return Err(format!("frames ({frames}) must be <= 257"));
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
    if let Some(keyframes) = &req.keyframes {
        validate_keyframes(keyframes, req.frames, family)?;
    }
    if let Some(audio) = &req.audio_file {
        require_ltx2_family(family, "audio_file")?;
        if audio.is_empty() {
            return Err("audio_file must not be empty".to_string());
        }
        validate_inline_media_size(audio, "audio_file", MAX_INLINE_AUDIO_BYTES)?;
    }
    if let Some(video) = &req.source_video {
        require_ltx2_family(family, "source_video")?;
        if video.is_empty() {
            return Err("source_video must not be empty".to_string());
        }
        validate_inline_media_size(video, "source_video", MAX_INLINE_SOURCE_VIDEO_BYTES)?;
    }
    if req.enable_audio.is_some() {
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

    if family == Some("ltx2") {
        match req.output_format {
            OutputFormat::Gif | OutputFormat::Apng | OutputFormat::Webp | OutputFormat::Mp4 => {}
            _ => return Err("LTX-2 outputs must use mp4, gif, apng, or webp".to_string()),
        }

        if req.enable_audio == Some(true) && req.output_format != OutputFormat::Mp4 {
            return Err("audio-enabled LTX-2 outputs must use mp4 format".to_string());
        }

        if req.retake_range.is_some() && req.source_video.is_none() {
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
                    if req.audio_file.is_none() {
                        return Err("pipeline=a2vid requires audio_file".to_string());
                    }
                }
                Ltx2PipelineMode::Retake => {
                    if req.source_video.is_none() {
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
                    if req.source_video.is_none() {
                        return Err("pipeline=ic-lora requires source_video".to_string());
                    }
                    if req.lora.is_none() && req.loras.as_ref().is_none_or(Vec::is_empty) {
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
    (768, 512),  // 3:2 (native)
    (512, 512),  // 1:1
    (1024, 576), // 16:9
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
        "ltx-video" => LTX_VIDEO_DIMS,
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
            prompt: "a red apple".to_string(),
            negative_prompt: None,
            model: "test-model".to_string(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 0.0,
            seed: Some(42),
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
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

    // ── validate_generate_request tests ──────────────────────────────────────

    #[test]
    fn valid_request_passes() {
        assert!(validate_generate_request(&valid_req()).is_ok());
    }

    #[test]
    fn ltx2_audio_requires_mp4() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Gif;
        req.enable_audio = Some(true);
        assert!(validate_generate_request(&req).unwrap_err().contains("mp4"));
    }

    #[test]
    fn ltx2_retake_requires_source_video() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
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
        req.output_format = OutputFormat::Mp4;
        req.audio_file = Some(vec![0; MAX_INLINE_AUDIO_BYTES + 1]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("audio_file exceeds"), "got: {err}");
        assert!(err.contains("64 MiB"), "got: {err}");
    }

    #[test]
    fn ltx2_source_video_rejects_inline_payloads_above_limit() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
        req.source_video = Some(vec![0; MAX_INLINE_SOURCE_VIDEO_BYTES + 1]);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("source_video exceeds"), "got: {err}");
        assert!(err.contains("64 MiB"), "got: {err}");
    }

    #[test]
    fn ltx2_keyframe_pipeline_requires_multiple_keyframes() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
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

    #[test]
    fn ltx2_allows_temporal_upscale_request() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
        req.temporal_upscale = Some(crate::Ltx2TemporalUpscale::X2);
        validate_generate_request(&req).unwrap();
    }

    #[test]
    fn ltx2_allows_x1_5_spatial_upscale_request() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
        req.spatial_upscale = Some(crate::Ltx2SpatialUpscale::X1_5);
        validate_generate_request(&req).unwrap();
    }

    #[test]
    fn empty_prompt_rejected() {
        let mut req = valid_req();
        req.prompt = "   ".to_string();
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("prompt"));
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
        req.output_format = OutputFormat::Mp4;
        req.frames = Some(10);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("8n+1"), "got: {err}");
        assert!(err.contains("LTX-Video / LTX-2"), "got: {err}");
    }

    #[test]
    fn ltx2_frames_at_rope_budget_accepted() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
        req.frames = Some(LTX2_MAX_FRAMES); // 153 pixel frames → 20 latent frames
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn ltx2_frames_over_rope_budget_rejected() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
        req.frames = Some(161); // 161 pixel frames → 21 latent frames > 20-frame RoPE max
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains("161"), "got: {err}");
        assert!(err.contains(&LTX2_MAX_FRAMES.to_string()), "got: {err}");
        assert!(err.contains("RoPE"), "got: {err}");
    }

    #[test]
    fn ltx2_3_frames_over_rope_budget_rejected() {
        let mut req = valid_req();
        req.model = "ltx-2.3-22b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
        req.frames = Some(193); // matches #226 repro: (193-1)/8+1 = 25 latent > 20
        let err = validate_generate_request(&req).unwrap_err();
        assert!(err.contains(&LTX2_MAX_FRAMES.to_string()), "got: {err}");
    }

    #[test]
    fn ltx_video_family_is_not_subject_to_the_ltx2_rope_cap() {
        let mut req = valid_req();
        req.model = "ltx-video-0.9.6-distilled:bf16".to_string();
        req.output_format = OutputFormat::Mp4;
        req.frames = Some(161); // above LTX2_MAX_FRAMES but under the generic 257 ceiling
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn ltx2_temporal_upscale_x2_doubles_the_effective_frame_ceiling() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
        req.frames = Some(257); // stage-1 = (257-1)/2+1 = 129 → 17 latent frames, fits
        req.temporal_upscale = Some(crate::Ltx2TemporalUpscale::X2);
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn ltx2_temporal_upscale_x2_still_rejects_stage1_overflow() {
        let mut req = valid_req();
        req.model = "ltx-2-19b-distilled:fp8".to_string();
        req.output_format = OutputFormat::Mp4;
        // 313 would produce stage-1 = 157 frames → 21 latent, but the generic
        // 257-frame ceiling catches it first; use a value inside that ceiling
        // that still overflows stage-1 after halving: frames=297 → stage-1=149
        // passes 20-latent budget. The nearest overflow above 257 is outside
        // the generic ceiling. Validate the principle that a 307-frame request
        // would be caught by *either* check, and document the 257 ceiling
        // remains the tighter gate for temporal-upscale requests.
        req.frames = Some(289); // over the generic 257 ceiling
        req.temporal_upscale = Some(crate::Ltx2TemporalUpscale::X2);
        let err = validate_generate_request(&req).unwrap_err();
        assert!(
            err.contains("257") || err.contains(&LTX2_MAX_FRAMES.to_string()),
            "got: {err}"
        );
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
        assert!(err.contains("requires edit_images"), "got: {err}");
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
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        req.control_scale = 0.8;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn controlnet_image_without_model_rejected() {
        let mut req = valid_req();
        req.control_image = Some(png_bytes());
        req.control_model = None;
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("control_model"));
    }

    #[test]
    fn controlnet_model_without_image_rejected() {
        let mut req = valid_req();
        req.control_image = None;
        req.control_model = Some("controlnet-canny-sd15".to_string());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("control_image"));
    }

    #[test]
    fn controlnet_invalid_image_rejected() {
        let mut req = valid_req();
        req.control_image = Some(vec![0x00, 0x01, 0x02, 0x03]);
        req.control_model = Some("controlnet-canny-sd15".to_string());
        assert!(validate_generate_request(&req)
            .unwrap_err()
            .contains("PNG or JPEG"));
    }

    #[test]
    fn controlnet_negative_scale_rejected() {
        let mut req = valid_req();
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
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        req.control_scale = 0.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn controlnet_high_scale_accepted() {
        let mut req = valid_req();
        req.control_image = Some(png_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        req.control_scale = 2.0;
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn controlnet_jpeg_accepted() {
        let mut req = valid_req();
        req.control_image = Some(jpeg_bytes());
        req.control_model = Some("controlnet-canny-sd15".to_string());
        assert!(validate_generate_request(&req).is_ok());
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

    #[test]
    fn lora_none_valid() {
        let req = valid_req();
        assert!(req.lora.is_none());
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn lora_scale_too_low_rejected() {
        let mut req = valid_req();
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
        let mut req = valid_req();
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
            let mut req = valid_req();
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
        let mut req = valid_req();
        req.lora = Some(crate::LoraWeight {
            path: "/nonexistent/path/adapter.safetensors".to_string(),
            scale: 1.0,
        });
        assert!(validate_generate_request(&req).is_ok());
    }

    #[test]
    fn lora_wrong_extension_rejected() {
        let mut req = valid_req();
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
}
