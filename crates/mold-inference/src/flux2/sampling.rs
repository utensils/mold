//! FLUX.2 sampling utilities shared by Klein and Dev checkpoints.
//!
//! Similar to FLUX.1 sampling but adapted for:
//! - 128 input channels (latent_channels=32, patchified)
//! - 4D positional IDs (vs FLUX.1's 3D)
//! - No pooled text vector (Dev adds guidance conditioning separately)

use anyhow::Context as _;
use candle_core::{Result, Tensor};

/// Sampling state for FLUX.2 Klein and Dev checkpoints.
///
/// Prepares image tokens, positional IDs, text embeddings, and the
/// conditioning vector for the transformer's denoising loop.
#[derive(Debug, Clone)]
pub struct Flux2State {
    /// Patchified image tokens: (B, seq_len, 128)
    pub img: Tensor,
    /// Image positional IDs: (B, seq_len, 4)
    pub img_ids: Tensor,
    /// Text encoder hidden states: (B, txt_len, context_dim)
    pub txt: Tensor,
    /// Text positional IDs: (B, txt_len, 4) — `[0, 0, 0, token_index]`
    pub txt_ids: Tensor,
    /// Conditioning vector: (B, vec_dim) — retained as zeros because FLUX.2
    /// has no pooled text input.
    pub vec: Tensor,
}

impl Flux2State {
    /// Build sampling state from text embeddings and noise.
    ///
    /// - `txt_emb`: (1, txt_len, context_dim) from three stacked encoder states
    /// - `img`: (B, 32, H/8, W/8) noise tensor
    ///
    /// The image is patchified with 2x2 patches, producing:
    /// - `img`: (B, (H/8/2)*(W/8/2), 32*4=128) tokens
    /// - `img_ids`: (B, seq_len, 4) with [0, row, col, 0] layout
    pub fn new(txt_emb: &Tensor, img: &Tensor) -> Result<Self> {
        let dtype = img.dtype();
        let (bs, _, _, _) = img.dims4()?;
        let dev = img.device();
        let (img, img_ids) = pack_image_tokens(img, 0)?;

        let (txt, txt_ids) = text_conditioning(txt_emb, bs, dev)?;

        // FLUX.2 has no pooled text vector — use a zero vector.
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

/// Batch one encoder output into transformer text tokens and their four-axis
/// position ids.
///
/// A classifier-free-guided render needs exactly this pair for its negative
/// prompt and nothing else — no second latent, no second id grid, since both
/// branches denoise the SAME image tokens. Building it through `Flux2State`
/// instead would demand the rank-4 latent the positive state was already
/// built from, which by then has been packed to rank 3.
///
/// This is the ONE place FLUX.2 text position ids are built — [dev] and
/// [klein], positive branch and unconditional branch alike — so the table
/// cannot drift between tiers or between the two halves of a guided step.
pub fn text_conditioning(
    txt_emb: &Tensor,
    batch_size: usize,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let txt = txt_emb.repeat(batch_size)?;
    let txt_ids = text_position_ids(txt.dim(1)?, batch_size, device)?;
    Ok((txt, txt_ids))
}

/// The four-axis position table FLUX.2 gives its text tokens.
///
/// Axis 3 is a RUNNING TOKEN INDEX and axes 0-2 (time, height, width) are
/// zero. Three upstreams agree and mold matched none of them until #WP13:
///
/// - BFL `flux2/src/flux2/sampling.py:93-104` — `prc_txt` builds
///   `cartesian_prod(t=arange(1), h=arange(1), w=arange(1), l=arange(L))`,
///   whose only varying column is the last.
/// - diffusers `pipelines/flux2/pipeline_flux2_klein.py:264-282` —
///   `_prepare_text_ids` repeats that cartesian product per batch row.
/// - ComfyUI `comfy/model_detection.py:278` sets `txt_ids_dims = [3]` for
///   every `flux2` checkpoint, and `comfy/ldm/flux/model.py:400-406` writes
///   `linspace(0, L-1)` into exactly those axes of an otherwise zero table.
///
/// mold carried FLUX.1's convention instead — an all-zero table — which gives
/// every text token the same RoPE position, so the transformer could not tell
/// the first word of the prompt from the last. That is a conditioning bug on
/// EVERY FLUX.2 render, [dev] and [klein] both.
///
/// The axis choice is not arbitrary: [`pack_image_tokens`] mirrors BFL's
/// `prc_img` (`sampling.py:141-151`), whose cartesian product is
/// `(t, h, w, l=arange(1))` — image tokens pin axis 3 to zero and vary axes
/// 0-2, so text and image tables are disjoint by construction and
/// `EmbedNd`'s four 32-wide RoPE axes each carry exactly one meaning.
///
/// Ids stay F32 for the same reason [`pack_image_tokens`] does: BF16 cannot
/// represent every integer above 256, and a padded FLUX.2 [klein] prompt is
/// 512 tokens long.
pub fn text_position_ids(
    len: usize,
    batch_size: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let zeros = Tensor::zeros((len, 1), candle_core::DType::F32, device)?;
    let index = Tensor::arange(0u32, len as u32, device)?
        .to_dtype(candle_core::DType::F32)?
        .reshape((len, 1))?;
    Tensor::cat(&[&zeros, &zeros, &zeros, &index], 1)?
        .reshape((1, len, 4))?
        .repeat((batch_size, 1, 1))
}

/// Patchify one FLUX.2 VAE latent and create its four-axis position IDs.
/// Target noise uses time coordinate 0; ordered reference images use 10, 20,
/// ... so the transformer can distinguish their token groups exactly as in
/// the upstream BFL implementation.
pub fn pack_image_tokens(img: &Tensor, time_coordinate: u32) -> Result<(Tensor, Tensor)> {
    let (batch, channels, height, width) = img.dims4()?;
    let device = img.device();
    let ph = height / 2;
    let pw = width / 2;
    let tokens = img
        .reshape((batch, channels, ph, 2, pw, 2))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((batch, ph * pw, channels * 4))?;
    let ids = Tensor::stack(
        &[
            Tensor::full(time_coordinate, (ph, pw), device)?,
            Tensor::arange(0u32, ph as u32, device)?
                .reshape(((), 1))?
                .broadcast_as((ph, pw))?,
            Tensor::arange(0u32, pw as u32, device)?
                .reshape((1, ()))?
                .broadcast_as((ph, pw))?,
            Tensor::full(0u32, (ph, pw), device)?,
        ],
        2,
    )?
    // BFL keeps coordinate IDs exact until RoPE evaluates them in float32.
    // BF16 cannot represent every integer above 256, which is reachable by
    // valid wide reference images.
    .to_dtype(candle_core::DType::F32)?
    .reshape((1, ph * pw, 4))?
    .repeat((batch, 1, 1))?;
    Ok((tokens, ids))
}

/// Pack an ORDERED group of VAE-encoded reference latents into the single
/// token/id pair the transformer appends after the noisy target tokens.
///
/// Reference `i` (zero-based) takes time coordinate `10 * (i + 1)`, and the
/// groups concatenate along the sequence axis in the order they were given.
/// The scale of 10 is first-party BFL: `flux2/src/flux2/sampling.py:53` sets
/// `scale = 10` and `default_prep` (`:226`) numbers the references from it;
/// diffusers mirrors it in `_prepare_image_ids`
/// (`pipeline_flux2_klein.py:318-366`, `t = scale + scale * i` at `:352`), and
/// ComfyUI's `comfy/model_detection.py:242-256` sets `ref_index_scale = 10.0`
/// for EVERY flux2 checkpoint — which is why distilled Klein, Klein base, and
/// [dev] all speak this protocol with the same checkpoint layout.
///
/// Each group restarts its own row/column grid, so the coordinate that keeps
/// two references apart is the time plane and nothing else. That is also why
/// this is a function rather than a loop inside the pipeline: the ordering is
/// the contract, and it is unit-testable without weights.
pub fn pack_reference_group(latents: &[Tensor]) -> anyhow::Result<(Tensor, Tensor)> {
    if latents.is_empty() {
        anyhow::bail!("pack_reference_group requires at least one reference latent");
    }
    let mut token_groups = Vec::with_capacity(latents.len());
    let mut id_groups = Vec::with_capacity(latents.len());
    for (index, latent) in latents.iter().enumerate() {
        let time_coordinate = u32::try_from(index + 1)
            .context("too many FLUX.2 reference images")?
            .checked_mul(10)
            .context("FLUX.2 reference time coordinate overflow")?;
        let (tokens, ids) = pack_image_tokens(latent, time_coordinate)?;
        token_groups.push(tokens);
        id_groups.push(ids);
    }
    let tokens = Tensor::cat(&token_groups.iter().collect::<Vec<_>>(), 1)?;
    let ids = Tensor::cat(&id_groups.iter().collect::<Vec<_>>(), 1)?;
    Ok((tokens, ids))
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp};

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

    /// Ordered references are told apart by their TIME plane and nothing
    /// else: reference 1 sits at t=10, reference 2 at t=20
    /// (`flux2/src/flux2/sampling.py:53` `scale = 10`, `default_prep` at
    /// `:226`; diffusers `pipeline_flux2_klein.py:352`), each restarts its own
    /// row/column grid, and axis 3 stays 0 for every image token. Two
    /// differently shaped latents pin all of it at once, because a group that
    /// carried the target's grid instead would still concatenate cleanly.
    #[test]
    fn ordered_klein_references_occupy_successive_time_planes() {
        let dev = Device::Cpu;
        let first = Tensor::randn(0f32, 1., (1, 32, 8, 6), &dev).unwrap();
        let second = Tensor::randn(0f32, 1., (1, 32, 4, 4), &dev).unwrap();
        let (tokens, ids) = pack_reference_group(&[first, second]).unwrap();

        // (8/2)*(6/2) = 12 tokens, then (4/2)*(4/2) = 4.
        assert_eq!(tokens.dims(), &[1, 12 + 4, 128]);
        assert_eq!(ids.dims(), &[1, 12 + 4, 4]);

        let ids = ids.i(0).unwrap().to_vec2::<f32>().unwrap();
        for row in ids.iter().take(12) {
            assert_eq!(row[0], 10.0, "the first reference occupies time plane 10");
        }
        for row in ids.iter().skip(12) {
            assert_eq!(row[0], 20.0, "the second reference occupies time plane 20");
        }
        // Row/column coordinates restart per group rather than continuing the
        // previous one's grid.
        assert_eq!((ids[0][1], ids[0][2]), (0.0, 0.0));
        assert_eq!((ids[11][1], ids[11][2]), (3.0, 2.0));
        assert_eq!((ids[12][1], ids[12][2]), (0.0, 0.0));
        assert_eq!((ids[15][1], ids[15][2]), (1.0, 1.0));
        // Axis 3 is unused by the image branch and must stay zero.
        assert!(ids.iter().all(|row| row[3] == 0.0));

        // A single reference still lands on t=10, never on the target's 0.
        let (_, solo) =
            pack_reference_group(&[Tensor::randn(0f32, 1., (1, 32, 4, 4), &dev).unwrap()]).unwrap();
        assert!(solo
            .i(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .all(|row| row[0] == 10.0));

        assert!(pack_reference_group(&[]).is_err());
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

    /// The unconditional branch is built AFTER the positive state, when the
    /// only latent in scope is the packed rank-3 one. Building it through
    /// `Flux2State::new` would demand the rank-4 latent that no longer exists
    /// there and fail every CFG render with "unexpected rank, expected: 4".
    #[test]
    fn negative_conditioning_matches_the_positive_state_without_a_latent() {
        let dev = Device::Cpu;
        let txt = Tensor::randn(0f32, 1., (1, 50, 7680), &dev).unwrap();
        let img = Tensor::randn(0f32, 1., (1, 32, 128, 128), &dev).unwrap();
        let state = Flux2State::new(&txt, &img).unwrap();

        let negative = Tensor::randn(0f32, 1., (1, 50, 7680), &dev).unwrap();
        let (neg_txt, neg_txt_ids) =
            text_conditioning(&negative, state.img.dim(0).unwrap(), state.img.device()).unwrap();

        assert_eq!(neg_txt.dims(), state.txt.dims());
        assert_eq!(neg_txt_ids.dims(), state.txt_ids.dims());
        // And it is the NEGATIVE embedding, not a second copy of the positive.
        let neg_first = neg_txt.i((0, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
        let expected = negative.i((0, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(neg_first, expected);
    }

    #[test]
    fn reference_tokens_carry_ordered_time_coordinate() {
        let dev = Device::Cpu;
        let latent = Tensor::zeros((1, 32, 8, 6), DType::F32, &dev).unwrap();
        let (tokens, ids) = pack_image_tokens(&latent, 20).unwrap();
        assert_eq!(tokens.dims(), &[1, 12, 128]);
        assert_eq!(ids.dims(), &[1, 12, 4]);
        let ids = ids.to_vec3::<f32>().unwrap();
        assert!(ids[0].iter().all(|id| id[0] == 20.0 && id[3] == 0.0));
    }

    /// BFL's `prc_txt` (`flux2/sampling.py:93-104`) is
    /// `cartesian_prod(arange(1), arange(1), arange(1), arange(L))`, so for
    /// `L = 3` the table is exactly the literal below: axes 0-2 (t, h, w) are
    /// pinned to zero and axis 3 (`l`) is a running token index. diffusers'
    /// `_prepare_text_ids` (`pipeline_flux2_klein.py:264-282`) builds the same
    /// four columns, and ComfyUI writes `linspace(0, L-1)` into the axes named
    /// by `txt_ids_dims = [3]` (`comfy/model_detection.py:278`,
    /// `comfy/ldm/flux/model.py:400-406`).
    ///
    /// mold used to hand the transformer an all-zero table, which is FLUX.1's
    /// convention: every text token then shared one RoPE position.
    #[test]
    fn flux2_text_ids_are_a_running_index_on_axis_three() {
        let dev = Device::Cpu;
        let txt = Tensor::randn(0f32, 1., (1, 3, 7680), &dev).unwrap();
        let (_, ids) = text_conditioning(&txt, 1, &dev).unwrap();
        assert_eq!(ids.dims(), &[1, 3, 4]);
        assert_eq!(
            ids.i(0).unwrap().to_vec2::<f32>().unwrap(),
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
                vec![0.0, 0.0, 0.0, 2.0],
            ],
            "axis 3 must be the running token index, axes 0-2 zero"
        );
    }

    /// The axis the text index lands on is the one `pack_image_tokens` leaves
    /// free. Image ids are `[t, h, w, 0]` (BFL `prc_img`,
    /// `flux2/sampling.py:141-151`) and text ids are `[0, 0, 0, l]`, so the two
    /// tables never collide on a shared axis — which is the whole reason the
    /// fourth axis exists in a model whose images are 2-D.
    #[test]
    fn text_and_image_ids_occupy_disjoint_axes() {
        let dev = Device::Cpu;
        let latent = Tensor::zeros((1, 32, 4, 6), DType::F32, &dev).unwrap();
        let (_, img_ids) = pack_image_tokens(&latent, 0).unwrap();
        let img_rows = img_ids.i(0).unwrap().to_vec2::<f32>().unwrap();
        assert!(
            img_rows.iter().all(|row| row[3] == 0.0),
            "image ids must leave axis 3 free for text"
        );

        let txt = Tensor::randn(0f32, 1., (1, 5, 7680), &dev).unwrap();
        let (_, txt_ids) = text_conditioning(&txt, 1, &dev).unwrap();
        let txt_rows = txt_ids.i(0).unwrap().to_vec2::<f32>().unwrap();
        assert!(
            txt_rows
                .iter()
                .all(|row| row[0] == 0.0 && row[1] == 0.0 && row[2] == 0.0),
            "text ids must leave the time/height/width axes to the image"
        );
    }

    /// FLUX.2 [dev] and [klein] differ only in the width of the conditioning
    /// their encoders produce — Mistral3's `3 * 5120` against Qwen3-4B's
    /// `3 * 2560`. The position ids are a property of the SEQUENCE, not of the
    /// encoder, and both tiers reach the transformer through
    /// `Flux2State::new` -> `text_conditioning`. One builder, one table.
    #[test]
    fn dev_and_klein_share_one_txt_id_builder() {
        let dev = Device::Cpu;
        let img = Tensor::randn(0f32, 1., (1, 32, 16, 16), &dev).unwrap();

        // [klein]: Qwen3 layers 9/18/27 stacked -> 3 * 2560.
        let klein = Tensor::randn(0f32, 1., (1, 7, 7680), &dev).unwrap();
        let klein_ids = Flux2State::new(&klein, &img).unwrap().txt_ids;
        // [dev]: Mistral3 layers 10/20/30 stacked -> 3 * 5120.
        let dev_emb = Tensor::randn(0f32, 1., (1, 7, 15360), &dev).unwrap();
        let dev_ids = Flux2State::new(&dev_emb, &img).unwrap().txt_ids;

        assert_eq!(
            klein_ids.to_vec3::<f32>().unwrap(),
            dev_ids.to_vec3::<f32>().unwrap(),
            "both tiers must get the same position table for the same length"
        );
        // And it is the running index, not FLUX.1's zeros.
        let (_, standalone) = text_conditioning(&klein, 1, &dev).unwrap();
        assert_eq!(
            klein_ids.to_vec3::<f32>().unwrap(),
            standalone.to_vec3::<f32>().unwrap(),
            "the CFG branch's builder and the state's builder must be one"
        );
    }

    #[test]
    fn wide_reference_position_ids_remain_exact_above_bf16_integer_range() {
        let dev = Device::Cpu;
        let latent = Tensor::zeros((1, 32, 2, 520), DType::BF16, &dev).unwrap();
        let (_, ids) = pack_image_tokens(&latent, 10).unwrap();
        assert_eq!(ids.dtype(), DType::F32);
        let ids = ids.to_vec3::<f32>().unwrap();
        assert_eq!(ids[0][257][2], 257.0);
        assert_eq!(ids[0][259][2], 259.0);
    }
}
