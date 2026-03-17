//! Flux.2 Klein-4B sampling utilities.
//!
//! Similar to FLUX.1 sampling but adapted for:
//! - 128 input channels (latent_channels=32, patchified)
//! - 4D positional IDs (vs FLUX.1's 3D)
//! - No pooled text vector (Klein uses only timestep conditioning)

use candle_core::{DType, Device, Result, Tensor};

/// Generate initial noise for Flux.2 Klein.
///
/// The VAE produces latents with 32 channels and built-in 2x2 patchifying,
/// so the transformer sees 128-channel tokens. The noise shape matches
/// the VAE latent space before patchifying:
/// `(batch, latent_channels=32, height/8, width/8)`
///
/// After patchifying in [`Flux2State::new`], this becomes the 128-channel
/// sequence the transformer operates on.
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

/// Compute the flow-matching timestep schedule.
///
/// Same as FLUX.1: linearly spaced from 1.0 to 0.0 with optional time-shift.
/// For Klein (distilled), use `shift=None` (no time-shift, like Schnell).
pub fn get_schedule(num_steps: usize, shift: Option<(usize, f64, f64)>) -> Vec<f64> {
    let timesteps: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .rev()
        .collect();
    match shift {
        None => timesteps,
        Some((image_seq_len, y1, y2)) => {
            let (x1, x2) = (256., 4096.);
            let m = (y2 - y1) / (x2 - x1);
            let b = y1 - m * x1;
            let mu = m * image_seq_len as f64 + b;
            timesteps
                .into_iter()
                .map(|v| time_shift(mu, 1., v))
                .collect()
        }
    }
}

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1. / t - 1.).powf(sigma))
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
