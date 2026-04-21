//! LTX-2 chain carryover primitives.
//!
//! Server-side chained video generation stitches multiple per-clip renders
//! into a single output. To avoid a VAE decode → RGB → VAE encode round-trip
//! between clips (which loses information and doubles VAE cost), the tail of
//! each clip is carried across as latent-space tokens and threaded into the
//! next clip's conditioning directly.
//!
//! This module owns the data types and shape math for that handoff. The
//! orchestrator and the `Ltx2Engine::generate_with_carryover` entry point
//! land in sibling commits.
//!
//! See `tasks/render-chain-v1-plan.md` Phase 1.1 for context.

use anyhow::{anyhow, Context, Result};
use candle_core::Tensor;
use image::RgbImage;

use crate::ltx2::model::shapes::SpatioTemporalScaleFactors;

/// Opaque carryover payload handed from one chain stage to the next.
///
/// Holds the final VAE latents of the emitting stage's motion tail, not the
/// decoded pixels — so the receiving stage can patchify the tokens directly
/// into its conditioning without a VAE re-encode.
#[derive(Debug, Clone)]
pub struct ChainTail {
    /// Number of *pixel* frames this tail represents (not latent frames).
    /// Clients of [`ChainTail`] work in pixel-frame units because that's
    /// what users think in; the latent-frame count is derived from this
    /// plus the LTX-2 VAE's 8× causal temporal ratio.
    pub frames: u32,

    /// Latent tokens for the tail.
    ///
    /// Shape: `[batch=1, channels=128, tail_latent_frames, H/32, W/32]`
    /// where `tail_latent_frames = tail_latent_frame_count(self.frames)`.
    ///
    /// Dtype is whatever the denoise loop produced — typically `F32`.
    /// Device is the engine's active device (GPU or CPU); the orchestrator
    /// is responsible for ensuring the next stage runs on the same device.
    pub latents: Tensor,

    /// The last decoded pixel frame of the emitting stage. Kept for
    /// debugging, progress UIs that want a thumbnail of the handoff point,
    /// and as a fallback rendering target if latent carryover ever needs
    /// to be disabled at runtime.
    pub last_rgb_frame: RgbImage,
}

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

/// Slice the last `tail_latent_frame_count(pixel_frames)` frames off the
/// time axis of a rank-5 video-latents tensor shaped
/// `[B, C, T, H, W]`.
///
/// The returned tensor is a view/narrow on the input (no copy on candle's
/// current backends) so callers who intend to hand it to a separate engine
/// invocation — which may drop this engine's state and rebuild it — should
/// `.contiguous()` or `.copy()` the result before the original owner goes
/// out of scope.
///
/// Errors if the tensor is not rank-5 or the requested tail exceeds the
/// available time axis — the latter would mean the orchestrator asked for
/// more tail than the stage produced, which indicates a caller bug.
pub fn extract_tail_latents(final_latents: &Tensor, pixel_frames: u32) -> Result<Tensor> {
    let dims = final_latents.dims();
    if dims.len() != 5 {
        return Err(anyhow!(
            "extract_tail_latents: expected rank-5 tensor [B, C, T, H, W], got shape {:?}",
            dims,
        ));
    }
    let time = dims[2];
    let tail = tail_latent_frame_count(pixel_frames);
    if tail > time {
        return Err(anyhow!(
            "extract_tail_latents: tail requests {} latent frames but the stage emitted only {} \
             (pixel_frames={}, tensor shape={:?})",
            tail,
            time,
            pixel_frames,
            dims,
        ));
    }
    let start = time - tail;
    final_latents
        .narrow(2, start, tail)
        .with_context(|| format!("narrow last {tail} latent frames off time axis"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

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

    #[test]
    fn extract_tail_narrows_last_latent_frame_for_4_pixel_frame_tail() {
        // Build a synthetic [1, 2, 3, 1, 1] where channel 0 is the latent-
        // frame index and channel 1 is a sentinel (42, 43, 44) so we can
        // see which frames the narrow returns.
        let data = vec![
            // frame 0
            0.0f32, 42.0, // frame 1
            1.0, 43.0, // frame 2
            2.0, 44.0,
        ];
        // Arrange [B=1, C=2, T=3, H=1, W=1]. `Tensor::from_vec` fills in
        // row-major order — the permute below puts channels on axis 1.
        let raw = Tensor::from_vec(data, (1, 3, 2, 1, 1), &Device::Cpu).expect("build raw tensor");
        // Reshape [1, T, C, H, W] → [1, C, T, H, W]
        let latents = raw
            .permute([0, 2, 1, 3, 4])
            .expect("permute to [B, C, T, H, W]");
        assert_eq!(latents.dims(), &[1, 2, 3, 1, 1]);

        // tail_latent_frame_count(4) = 1 → take the last latent frame only.
        let tail = extract_tail_latents(&latents, 4).expect("extract");
        assert_eq!(tail.dims(), &[1, 2, 1, 1, 1]);
        let values = tail.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            values,
            vec![2.0, 44.0],
            "tail must be the last latent frame (index 2) across all channels",
        );
    }

    #[test]
    fn extract_tail_narrows_two_frames_for_9_pixel_frame_tail() {
        // Simple rank-5 zero tensor with T=3; narrowing the last 2 frames
        // out of 3 is enough to verify the shape without wrestling with
        // permutations again.
        let latents = Tensor::zeros((1, 1, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let tail = extract_tail_latents(&latents, 9).expect("extract");
        assert_eq!(tail.dims(), &[1, 1, 2, 2, 2]);
    }

    #[test]
    fn extract_tail_rejects_rank_4_tensor() {
        let bad = Tensor::zeros((1, 128, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let err = extract_tail_latents(&bad, 4).expect_err("rank 4 must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("rank-5") && msg.contains("T, H, W"),
            "error must identify the rank mismatch, got: {msg}",
        );
    }

    #[test]
    fn extract_tail_rejects_oversize_request() {
        // Tensor has 1 latent frame; asking for a 9-pixel-frame tail needs 2.
        let latents = Tensor::zeros((1, 128, 1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let err = extract_tail_latents(&latents, 9).expect_err("oversize tail must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("requests 2") && msg.contains("only 1"),
            "error must name the latent-frame mismatch, got: {msg}",
        );
    }
}
