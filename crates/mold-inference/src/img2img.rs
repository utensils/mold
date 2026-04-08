/// Convert an img2img strength into the number of denoising steps to keep.
///
/// This matches the scheduler semantics used by diffusers img2img pipelines:
/// `floor(total_steps * strength)`, clamped to `total_steps`.
pub(crate) fn img2img_effective_steps(total_steps: usize, strength: f64) -> usize {
    ((total_steps as f64 * strength).floor() as usize).min(total_steps)
}

/// Convert an img2img strength into the schedule start index.
///
/// `0` means full txt2img-style denoising, `total_steps` means preserve the
/// encoded image without any denoising steps.
pub(crate) fn img2img_start_index(total_steps: usize, strength: f64) -> usize {
    total_steps.saturating_sub(img2img_effective_steps(total_steps, strength))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_strength_keeps_all_steps() {
        assert_eq!(img2img_effective_steps(20, 1.0), 20);
        assert_eq!(img2img_start_index(20, 1.0), 0);
    }

    #[test]
    fn low_strength_keeps_tail_steps() {
        assert_eq!(img2img_effective_steps(20, 0.1), 2);
        assert_eq!(img2img_start_index(20, 0.1), 18);
    }

    #[test]
    fn floor_semantics_match_reference_behavior() {
        assert_eq!(img2img_effective_steps(20, 0.3), 6);
        assert_eq!(img2img_start_index(20, 0.3), 14);
        assert_eq!(img2img_effective_steps(20, 0.75), 15);
        assert_eq!(img2img_start_index(20, 0.75), 5);
    }

    #[test]
    fn tiny_strength_can_skip_all_steps() {
        assert_eq!(img2img_effective_steps(20, 0.01), 0);
        assert_eq!(img2img_start_index(20, 0.01), 20);
    }
}
