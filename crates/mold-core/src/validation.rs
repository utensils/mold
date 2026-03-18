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
    if req.batch_size == 0 {
        return Err("batch_size must be >= 1".to_string());
    }
    if req.batch_size > 16 {
        return Err(format!("batch_size ({}) must be <= 16", req.batch_size));
    }
    if req.guidance < 0.0 {
        return Err(format!("guidance ({}) must be >= 0.0", req.guidance));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputFormat;

    fn valid_req() -> GenerateRequest {
        GenerateRequest {
            prompt: "a red apple".to_string(),
            model: "test-model".to_string(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 0.0,
            seed: Some(42),
            batch_size: 1,
            output_format: OutputFormat::Png,
        }
    }

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
        req.width = 1280;
        req.height = 1280; // 1.64MP > 1.1MP limit
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
    fn excessive_batch_rejected() {
        let mut req = valid_req();
        req.batch_size = 17;
        assert!(validate_generate_request(&req).is_err());
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
    fn seed_is_optional() {
        let mut req = valid_req();
        req.seed = None;
        assert!(validate_generate_request(&req).is_ok());
    }
}
