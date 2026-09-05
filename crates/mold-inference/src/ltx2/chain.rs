//! LTX-2 VAE-ratio helpers for chain carryover.

use crate::ltx2::model::shapes::SpatioTemporalScaleFactors;

/// Number of latent frames corresponding to `pixel_frames` pixel frames
/// under the LTX-2 VAE's 8× causal temporal compression. `1` for
/// `1..=8` pixel frames, `2` for `9..=16`, etc. Matches
/// `VideoLatentShape::from_pixel_shape`.
///
/// Panics if `pixel_frames == 0` — a zero-frame tail is nonsensical and
/// would under-flow the formula. Callers must validate upstream.
pub fn tail_latent_frame_count(pixel_frames: u32) -> usize {
    assert!(
        pixel_frames > 0,
        "tail_latent_frame_count: pixel_frames must be > 0",
    );
    let scale = SpatioTemporalScaleFactors::default().time;
    ((pixel_frames as usize - 1) / scale) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_latent_frame_count_matches_vae_formula() {
        // Single-frame tail and up to 8 pixel frames fit in 1 latent frame
        // (LTX-2 VAE uses causal first frame + 8× temporal compression).
        for px in [1u32, 2, 4, 8] {
            assert_eq!(tail_latent_frame_count(px), 1, "{px} pixel frames");
        }
        // 9..=16 span 2 latent frames, 17..=24 span 3, etc.
        assert_eq!(tail_latent_frame_count(9), 2);
        assert_eq!(tail_latent_frame_count(16), 2);
        assert_eq!(tail_latent_frame_count(17), 3);
        assert_eq!(tail_latent_frame_count(24), 3);
        // Full-clip tail (97 frames) → 13 latent frames, matching
        // VideoLatentShape::from_pixel_shape under the same VAE ratio.
        assert_eq!(tail_latent_frame_count(97), 13);
    }

    #[test]
    #[should_panic(expected = "pixel_frames must be > 0")]
    fn tail_latent_frame_count_rejects_zero() {
        tail_latent_frame_count(0);
    }
}
