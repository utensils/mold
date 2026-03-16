use crate::GenerateRequest;

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
    if pixels > 1_100_000 {
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
    Ok(())
}
