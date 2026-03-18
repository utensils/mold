//! Flux.2 Klein-4B sampling utilities.
//!
//! Similar to FLUX.1 sampling but adapted for:
//! - 128 input channels (latent_channels=32, patchified)
//! - 4D positional IDs (vs FLUX.1's 3D)
//! - No pooled text vector (Klein uses only timestep conditioning)

#[cfg(test)]
use candle_core::Device;
use candle_core::{DType, Result, Tensor};

/// Generate initial noise for Flux.2 Klein.
///
/// The VAE produces latents with 32 channels and built-in 2x2 patchifying,
/// so the transformer sees 128-channel tokens. The noise shape matches
/// the VAE latent space before patchifying:
/// `(batch, latent_channels=32, height/8, width/8)`
///
/// After patchifying in [`Flux2State::new`], this becomes the 128-channel
/// sequence the transformer operates on.
#[cfg(test)]
pub fn get_noise(
    num_samples: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    // Flux.2 VAE has 3 downsample stages (factor 8) with patch_size=[2,2].
    // Latent spatial dims = pixel_dims / 8.
    let latent_h = height.div_ceil(8);
    let latent_w = width.div_ceil(8);
    Tensor::randn(0f32, 1., (num_samples, 32, latent_h, latent_w), device)
}

/// Sampling state for Flux.2 Klein-4B.
///
/// Prepares image tokens, positional IDs, text embeddings, and the
/// conditioning vector for the transformer's denoising loop.
#[derive(Debug, Clone)]
pub struct Flux2State {
    /// Patchified image tokens: (B, seq_len, 128)
    pub img: Tensor,
    /// Image positional IDs: (B, seq_len, 4)
    pub img_ids: Tensor,
    /// Text encoder hidden states: (B, txt_len, 7680)
    pub txt: Tensor,
    /// Text positional IDs: (B, txt_len, 4) — zeros for text tokens
    pub txt_ids: Tensor,
    /// Conditioning vector: (B, vec_dim) — zeros for Klein (no pooled text)
    pub vec: Tensor,
}

impl Flux2State {
    /// Build sampling state from text embeddings and noise.
    ///
    /// - `txt_emb`: (1, txt_len, 7680) from Qwen3 encoder (stacked 3x hidden states)
    /// - `img`: (B, 32, H/8, W/8) noise tensor from [`get_noise`]
    ///
    /// The image is patchified with 2x2 patches, producing:
    /// - `img`: (B, (H/8/2)*(W/8/2), 32*4=128) tokens
    /// - `img_ids`: (B, seq_len, 4) with [0, row, col, 0] layout
    pub fn new(txt_emb: &Tensor, img: &Tensor) -> Result<Self> {
        let dtype = img.dtype();
        let (bs, c, h, w) = img.dims4()?;
        let dev = img.device();

        // Patchify: reshape (B, C, H, 2, W, 2) -> (B, H*W, C*4)
        let img = img.reshape((bs, c, h / 2, 2, w / 2, 2))?;
        let img = img.permute((0, 2, 4, 1, 3, 5))?;
        let img = img.reshape((bs, h / 2 * w / 2, c * 4))?;

        // Build 4D image position IDs: [channel_idx=0, row, col, extra=0]
        let ph = h / 2;
        let pw = w / 2;
        let img_ids = Tensor::stack(
            &[
                // Axis 0: constant 0 (channel index placeholder)
                Tensor::full(0u32, (ph, pw), dev)?,
                // Axis 1: row index
                Tensor::arange(0u32, ph as u32, dev)?
                    .reshape(((), 1))?
                    .broadcast_as((ph, pw))?,
                // Axis 2: column index
                Tensor::arange(0u32, pw as u32, dev)?
                    .reshape((1, ()))?
                    .broadcast_as((ph, pw))?,
                // Axis 3: constant 0 (extra axis for 4D RoPE)
                Tensor::full(0u32, (ph, pw), dev)?,
            ],
            2,
        )?
        .to_dtype(dtype)?;
        let img_ids = img_ids.reshape((1, ph * pw, 4))?;
        let img_ids = img_ids.repeat((bs, 1, 1))?;

        // Text tokens
        let txt = txt_emb.repeat(bs)?;
        let txt_ids = Tensor::zeros((bs, txt.dim(1)?, 4), dtype, dev)?;

        // Klein has no pooled text vector — use a zero vector
        // The transformer's vector_in is None so this won't be used,
        // but we keep a minimal tensor for API compatibility.
        let vec = Tensor::zeros((bs, 1), dtype, dev)?;

        Ok(Self {
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
        })
    }
}

/// Compute the Flux.2 flow-matching timestep schedule.
///
/// Uses `compute_empirical_mu` (from BFL's official flux2 code) to calculate
/// a resolution-and-step-dependent mu, then applies `generalized_time_snr_shift`.
/// This is NOT the same as FLUX.1's simple linear interpolation schedule.
///
/// For 1024x1024 at 4 steps: produces [1.0, 0.967, 0.908, 0.767, 0.0].
pub fn get_schedule(num_steps: usize, image_seq_len: usize) -> Vec<f64> {
    let mu = compute_empirical_mu(image_seq_len, num_steps);
    let timesteps: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .rev()
        .collect();
    timesteps
        .into_iter()
        .map(|t| generalized_time_snr_shift(t, mu, 1.0))
        .collect()
}

/// BFL's empirical mu computation for Flux.2 timestep scheduling.
///
/// A piecewise linear function of both image sequence length and step count,
/// calibrated with empirical coefficients. For `image_seq_len > 4300`, only
/// the 200-step calibration line is used (step count becomes irrelevant).
fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let (a1, b1) = (8.738_095_24e-05, 1.898_333_33);
    let (a2, b2) = (0.000_169_27, 0.456_666_66);
    let seq = image_seq_len as f64;

    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }

    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

/// Generalized SNR time shift: `exp(mu) / (exp(mu) + (1/t - 1)^sigma)`.
///
/// Compresses timesteps toward 1.0 (more denoising in later steps).
/// With sigma=1.0, this simplifies to `exp(mu) / (exp(mu) + 1/t - 1)`.
fn generalized_time_snr_shift(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let e = mu.exp();
    e / (e + (1.0 / t - 1.0).powf(sigma))
}

/// Unpack transformer output back to spatial format.
///
/// Reverses the patchifying: (B, H/2*W/2, C*4) -> (B, C, H, W)
/// where H, W are the latent spatial dimensions (pixel_dims / 8).
pub fn unpack(xs: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _h_w, c_ph_pw) = xs.dims3()?;
    let latent_h = height.div_ceil(8);
    let latent_w = width.div_ceil(8);
    let ph = latent_h / 2;
    let pw = latent_w / 2;
    xs.reshape((b, ph, pw, c_ph_pw / 4, 2, 2))?
        .permute((0, 3, 1, 4, 2, 5))?
        .reshape((b, c_ph_pw / 4, ph * 2, pw * 2))
}

/// Convert a raw DType (BF16/F32/F16) tensor to the appropriate type for
/// the denoising state. Quantized models need F32 state tensors.
///
/// Not used for Klein (BF16 only), but kept for future GGUF support.
#[allow(dead_code)]
pub fn to_state_dtype(t: &Tensor, is_quantized: bool) -> Result<Tensor> {
    if is_quantized {
        t.to_dtype(DType::F32)
    } else {
        Ok(t.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_noise_shape() {
        let t = get_noise(1, 1024, 1024, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 32, 128, 128]);
    }

    #[test]
    fn schedule_endpoints() {
        let s = get_schedule(4, 4096);
        assert_eq!(s.len(), 5); // num_steps + 1
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!((s[4] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn schedule_1024x1024_4steps_matches_bfl() {
        // BFL reference values for 1024x1024 (seq_len=4096) at 4 steps
        let s = get_schedule(4, 4096);
        assert!((s[0] - 1.0).abs() < 1e-4, "t0={}", s[0]);
        assert!((s[1] - 0.9674).abs() < 0.005, "t1={}", s[1]);
        assert!((s[2] - 0.9081).abs() < 0.005, "t2={}", s[2]);
        assert!((s[3] - 0.7672).abs() < 0.005, "t3={}", s[3]);
        assert!((s[4] - 0.0).abs() < 1e-4, "t4={}", s[4]);
    }

    #[test]
    fn schedule_is_not_linear() {
        let s = get_schedule(4, 4096);
        // With empirical mu shift, intermediate values are compressed toward 1.0
        // A linear schedule would have [1.0, 0.75, 0.5, 0.25, 0.0]
        assert!(s[1] > 0.9, "t1={} should be > 0.9 (shifted)", s[1]);
        assert!(s[2] > 0.85, "t2={} should be > 0.85 (shifted)", s[2]);
    }

    #[test]
    fn empirical_mu_increases_with_resolution() {
        let mu_small = compute_empirical_mu(256, 4);
        let mu_large = compute_empirical_mu(4096, 4);
        assert!(mu_large > mu_small, "larger images should have higher mu");
    }

    #[test]
    fn unpack_roundtrips_with_patchify() {
        let dev = Device::Cpu;
        let img = Tensor::randn(0f32, 1., (1, 32, 128, 128), &dev).unwrap();
        // Patchify: (1, 32, 64, 2, 64, 2) -> (1, 64*64, 128)
        let patched = img
            .reshape((1, 32, 64, 2, 64, 2))
            .unwrap()
            .permute((0, 2, 4, 1, 3, 5))
            .unwrap()
            .reshape((1, 64 * 64, 128))
            .unwrap();
        let recovered = unpack(&patched, 1024, 1024).unwrap();
        assert_eq!(recovered.dims(), &[1, 32, 128, 128]);
    }

    #[test]
    fn flux2_state_builds_correct_shapes() {
        let dev = Device::Cpu;
        let txt = Tensor::randn(0f32, 1., (1, 50, 7680), &dev).unwrap();
        let img = Tensor::randn(0f32, 1., (1, 32, 128, 128), &dev).unwrap();
        let state = Flux2State::new(&txt, &img).unwrap();
        assert_eq!(state.img.dims(), &[1, 64 * 64, 128]); // patchified
        assert_eq!(state.img_ids.dims(), &[1, 64 * 64, 4]); // 4D IDs
        assert_eq!(state.txt.dims(), &[1, 50, 7680]);
        assert_eq!(state.txt_ids.dims(), &[1, 50, 4]);
    }
}
