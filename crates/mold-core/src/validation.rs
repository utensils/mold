use crate::{GenerateRequest, UpscaleRequest};

/// Maximum total pixels allowed (~1.8 megapixels). Qwen-Image trains at ~1.6MP
/// (1328x1328), other models at ≤1MP. Headroom for non-square aspect ratios.
pub const MAX_PIXELS: u64 = 1_800_000;

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

/// Check whether `data` starts with a recognized image format magic bytes (PNG or JPEG).
fn is_valid_image_format(data: &[u8]) -> bool {
    let is_png = data.len() >= 4 && data[..4] == [0x89, 0x50, 0x4E, 0x47];
    let is_jpeg = data.len() >= 2 && data[..2] == [0xFF, 0xD8];
    is_png || is_jpeg
}

/// Validate a generate request. Returns `Ok(())` if valid, or an error message.
/// Shared between the HTTP server and local CLI inference paths.
pub fn validate_generate_request(req: &GenerateRequest) -> Result<(), String> {
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
    // Cap by total pixel count (~1.1M) rather than per-dimension to allow portrait/landscape.
    // 896x1152 = 1.03M, 1024x1024 = 1.05M, 1280x768 = 0.98M — all fine.
    // 1280x1280 = 1.64M — too large, OOMs on VAE decode.
    let pixels = req.width as u64 * req.height as u64;
    if pixels > MAX_PIXELS {
        return Err(format!(
            "{}x{} = {} megapixels exceeds the ~1.1MP limit (VAE VRAM constraint)",
            req.width,
            req.height,
            pixels as f64 / 1_000_000.0
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
    // img2img validation
    if let Some(ref img) = req.source_image {
        if req.strength <= 0.0 || req.strength > 1.0 {
            return Err(format!(
                "strength ({}) must be in range (0.0, 1.0] when source_image is provided",
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
        if lora.scale < 0.0 || lora.scale > 2.0 {
            return Err(format!(
                "lora scale ({}) must be in range [0.0, 2.0]",
                lora.scale
            ));
        }
        if !lora.path.ends_with(".safetensors") {
            return Err("lora file must be a .safetensors file".to_string());
        }
    }
    // Video frame validation
    if let Some(frames) = req.frames {
        if frames == 0 {
            return Err("frames must be >= 1".to_string());
        }
        // LTX Video requires frames = 8n+1 (9, 17, 25, 33, …)
        if frames > 1 && (frames - 1) % 8 != 0 {
            return Err(format!(
                "frames ({frames}) must be 8n+1 (e.g. 9, 17, 25, 33, 41, 49, …)"
            ));
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
    fn img2img_strength_zero_rejected() {
        let mut req = valid_req();
        req.source_image = Some(png_bytes());
        req.strength = 0.0;
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
