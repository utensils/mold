//! A6 precision study: is the fork's fused LayerNorm kernel accurate enough on
//! Qwen Image 2.1's real residual stream to replace the hand-written F32
//! adaLN?
//!
//! `candle_nn::ops::layer_norm` on CUDA (`candle-kernels/src/reduce.cu`
//! `layernorm`) accumulates `sum(x)` and `sum(x^2)` in F32 in ONE pass and
//! takes `var = E[x^2] - mean^2`. That formula cancels catastrophically when
//! a row's `|mean|` is large against its standard deviation, and the 2.1
//! residual stream grows with depth. The transformer's legacy path is a
//! two-pass F32 LayerNorm (`LayerNormNoParams`) followed by a BF16
//! `* (1 + scale)`. This study measures both against an f64 two-pass
//! reference at every block's two adaLN sites.

use super::cuda::{bench_mode, env_or, install_mode, transformer_paths, DEFAULT_PROMPT};
use super::*;
use crate::progress::ProgressReporter;
use crate::qwen_image21::{encode_t2i_prompts, QWEN_IMAGE_21_LATENT_CHANNELS};

/// Error statistics of one normalization candidate against the f64 reference.
#[derive(Debug, Default, Clone, Copy)]
struct NormError {
    /// `max |candidate - reference| / max |reference|`.
    max_rel: f64,
    /// `rms(candidate - reference) / rms(reference)`.
    rel_rms: f64,
}

/// Compare `candidate` (row-major `[rows, D]`) against the f64 two-pass
/// LayerNorm of `input` scaled by `alpha` (`[D]`). Also returns the worst
/// row's `|mean| / std`, the quantity one-pass variance is sensitive to.
fn adaln_error(input: &[f32], alpha: &[f32], candidate: &[f32], eps: f64) -> (NormError, f64) {
    let dim = alpha.len();
    let (mut max_err, mut max_ref, mut err_sq, mut ref_sq) = (0f64, 0f64, 0f64, 0f64);
    let mut worst_mean_over_std = 0f64;
    for (row, out) in input.chunks_exact(dim).zip(candidate.chunks_exact(dim)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / dim as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / dim as f64;
        let inv = 1.0 / (var + eps).sqrt();
        worst_mean_over_std =
            worst_mean_over_std.max(mean.abs() / var.sqrt().max(f64::MIN_POSITIVE));
        for ((&x, &a), &c) in row.iter().zip(alpha).zip(out) {
            let reference = (f64::from(x) - mean) * inv * f64::from(a);
            let err = f64::from(c) - reference;
            max_err = max_err.max(err.abs());
            max_ref = max_ref.max(reference.abs());
            err_sq += err * err;
            ref_sq += reference * reference;
        }
    }
    (
        NormError {
            max_rel: max_err / max_ref.max(f64::MIN_POSITIVE),
            rel_rms: (err_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt(),
        },
        worst_mean_over_std,
    )
}

fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

struct Site {
    hand: NormError,
    fused: NormError,
    mean_over_std: f64,
    absmax: f32,
}

/// Measure one adaLN site both ways: the legacy hand F32 LayerNorm followed by
/// the BF16 `* (1 + scale)`, and the fused `layer_norm(x, alpha = 1 + scale,
/// beta = 0)` kernel.
fn measure_site(x: &Tensor, scale: &Tensor, norm: &LayerNormNoParams, eps: f64) -> Result<Site> {
    let dim = x.dim(D::Minus1)?;
    let x2 = x.reshape(((), dim))?.contiguous()?;
    let alpha = (scale.reshape(dim)? + 1.0)?.contiguous()?;
    let hand = norm.forward(&x2)?.broadcast_mul(&alpha)?;
    let beta = Tensor::zeros(dim, x2.dtype(), x2.device())?;
    let fused = candle_nn::ops::layer_norm(&x2, &alpha, &beta, eps as f32)?;
    let input = to_f32_vec(&x2)?;
    let alpha = to_f32_vec(&alpha)?;
    let (hand, mean_over_std) = adaln_error(&input, &alpha, &to_f32_vec(&hand)?, eps);
    let (fused, _) = adaln_error(&input, &alpha, &to_f32_vec(&fused)?, eps);
    let absmax = input.iter().fold(0f32, |m, v| m.max(v.abs()));
    Ok(Site {
        hand,
        fused,
        mean_over_std,
        absmax,
    })
}

/// Run the legacy forward on the real checkpoint (1024², three timesteps) and
/// measure every block's norm1 and norm2 inputs, text and target rows
/// separately (the prefix takes the t=0 modulation row). The design gate for
/// adopting the fused kernel is a worst max relative error <= 1e-2.
#[test]
#[ignore = "requires installed Qwen Image 2.1 weights and an idle, exclusive CUDA GPU"]
fn official_cuda_adaln_precision_study() -> Result<()> {
    let root = PathBuf::from(std::env::var("QWEN_IMAGE21_MODEL_ROOT")?);
    let output = PathBuf::from(std::env::var("QWEN_IMAGE21_BENCH_OUTPUT")?);
    std::fs::create_dir_all(&output)?;
    let prompt =
        std::env::var("QWEN_IMAGE21_BENCH_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let seed: u64 = env_or("QWEN_IMAGE21_BENCH_SEED", 210001)?;
    let device = Device::new_cuda(0)?;
    let dtype = crate::engine::gpu_dtype(&device);
    let progress = ProgressReporter::default();
    let shared = root.join("shared/qwen-image21");
    let text_paths = (1..=4)
        .map(|i| shared.join(format!("text_encoder/model-{i:05}-of-00004.safetensors")))
        .collect::<Vec<_>>();
    let conditioning = {
        let mut encoder = crate::encoders::qwen3::Qwen3Encoder::load_bf16(
            &text_paths,
            &shared.join("processor/tokenizer.json"),
            &device,
            dtype,
            &crate::encoders::qwen3_bf16::Qwen3BF16Config::qwen3_image_21_text_encoder(),
            &progress,
        )?;
        encode_t2i_prompts(&mut encoder, std::slice::from_ref(&prompt))?
            .to_device_dtype(&device, dtype)?
    };
    let mut transformer = QwenImage21Transformer::load(
        &transformer_paths(&root, "bf16")?,
        &device,
        dtype,
        &progress,
    )?;
    install_mode(&mut transformer, &bench_mode("legacy")?)?;
    let eps = transformer.cfg.eps;
    let inner = transformer.cfg.inner_dim();
    let (lh, lw) = (64usize, 64usize);
    let text_len = conditioning.sequence_length();
    let target_len = lh * lw;
    let noise = crate::engine::seeded_randn(
        seed,
        &[1, target_len, QWEN_IMAGE_21_LATENT_CHANNELS],
        &device,
        dtype,
    )?;
    let text = conditioning.embeddings.to_dtype(dtype)?;
    let (rope_cos, rope_sin) = transformer.t2i_rope(text_len, lh, lw, dtype, &device)?;
    let mut sites = Vec::new();
    let mut worst = (0f64, String::new());
    let mut worst_hand = 0f64;
    let mut worst_ratio = 0f64;
    for timestep in [1.0f64, 0.5, 0.05] {
        // Magnitude-matched stand-in for the sampler's latent at this sigma.
        let latents = (&noise * timestep)?;
        let mut hidden = Tensor::cat(
            &[
                &transformer.txt_in.forward(&text)?,
                &transformer.img_in.forward(&latents)?,
            ],
            1,
        )?;
        let temb = transformer
            .time_text_embed
            .forward(&[timestep, 0.0], dtype, &device)?;
        let modulation = transformer
            .modulation
            .forward(&candle_nn::Activation::Silu.forward(&temb)?)?;
        let real = modulation.narrow(0, 0, 1)?;
        let zero = modulation.narrow(0, 1, 1)?;
        let per_token = Tensor::cat(
            &[
                &zero.unsqueeze(1)?.broadcast_as((1, text_len, 4 * inner))?,
                &real
                    .unsqueeze(1)?
                    .broadcast_as((1, target_len, 4 * inner))?,
            ],
            1,
        )?;
        for (index, block) in transformer.blocks.iter().enumerate() {
            for (site, offset) in [("norm1", 0usize), ("norm2", 2 * inner)] {
                let (x, norm) = if site == "norm1" {
                    (hidden.clone(), &block.norm1)
                } else {
                    // Recompute the attention half to reach norm2's input.
                    let mod1 = per_token.narrow(D::Minus1, 0, 2 * inner)?;
                    let (normalized, gate) =
                        TransformerBlock::modulate(block.norm1.forward(&hidden)?, &mod1)?;
                    let attn = block.attn.forward_t2i(
                        &normalized,
                        &rope_cos,
                        &rope_sin,
                        &conditioning.valid_tokens,
                        LayerCache::Disabled,
                    )?;
                    (
                        (&hidden + gate.tanh()?.broadcast_mul(&attn)?)?,
                        &block.norm2,
                    )
                };
                for (segment, start, len, row) in [
                    ("text", 0, text_len, &zero),
                    ("target", text_len, target_len, &real),
                ] {
                    let scale = row.narrow(D::Minus1, offset, inner)?;
                    let measured = measure_site(&x.narrow(1, start, len)?, &scale, norm, eps)?;
                    let name = format!("t{timestep}/block{index}/{site}/{segment}");
                    if measured.fused.max_rel > worst.0 {
                        worst = (measured.fused.max_rel, name.clone());
                    }
                    worst_hand = worst_hand.max(measured.hand.max_rel);
                    worst_ratio = worst_ratio.max(measured.mean_over_std);
                    sites.push(json!({
                        "site": name,
                        "input_absmax": measured.absmax,
                        "worst_row_mean_over_std": measured.mean_over_std,
                        "hand_max_rel": measured.hand.max_rel,
                        "hand_rel_rms": measured.hand.rel_rms,
                        "fused_max_rel": measured.fused.max_rel,
                        "fused_rel_rms": measured.fused.rel_rms,
                    }));
                }
            }
            hidden = block.forward_t2i(
                &hidden,
                &per_token,
                &rope_cos,
                &rope_sin,
                &conditioning.valid_tokens,
                LayerCache::Disabled,
            )?;
        }
        eprintln!("adaln t={timestep}: worst fused so far {worst:?}, hand {worst_hand:.3e}");
    }
    let receipt = json!({
        "prompt": prompt,
        "seed": seed,
        "canvas": "1024x1024",
        "kernel": "candle-kernels reduce.cu layernorm: one-pass F32 E[x^2]-mean^2",
        "gate_max_rel": 1e-2,
        "worst_fused_max_rel": worst.0,
        "worst_fused_site": worst.1,
        "worst_hand_max_rel": worst_hand,
        "worst_row_mean_over_std": worst_ratio,
        "passes": worst.0 <= 1e-2,
        "sites": sites,
    });
    std::fs::write(
        output.join("receipt-adaln-precision.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    eprintln!(
        "adaln worst fused max_rel {:.3e} at {} (hand {:.3e}, worst |mean|/std {:.2})",
        worst.0, worst.1, worst_hand, worst_ratio
    );
    Ok(())
}

#[test]
fn adaln_error_is_zero_for_an_exact_candidate() {
    let input = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
    let alpha = [1.0f32, 2.0, 0.5, 1.0];
    let mut exact = Vec::new();
    for row in input.chunks_exact(4) {
        let mean = row.iter().sum::<f32>() / 4.0;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        for (x, a) in row.iter().zip(alpha) {
            exact.push((x - mean) / (var + 1e-6).sqrt() * a);
        }
    }
    let (error, ratio) = adaln_error(&input, &alpha, &exact, 1e-6);
    assert!(error.max_rel < 1e-6 && error.rel_rms < 1e-6);
    // Row one: mean 2.5, variance 1.25.
    assert!((ratio - 2.5 / 1.25f64.sqrt()).abs() < 1e-9);
}
