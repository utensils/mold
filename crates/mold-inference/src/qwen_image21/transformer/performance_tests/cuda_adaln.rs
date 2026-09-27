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

/// The design gate for adopting the fused kernel: its worst max relative
/// error against the f64 reference at any measured site.
const FUSED_ADALN_GATE_MAX_REL: f64 = 1e-2;

/// Which rows of the joint sequence a measurement covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowKind {
    Text,
    Condition,
    Target,
}

impl RowKind {
    fn label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Condition => "condition",
            Self::Target => "target",
        }
    }
}

/// Joint positions of every row of `kind` in `layout`.
fn rows_of(
    layout: &crate::qwen_image21::layout::QwenImage21JointLayout,
    kind: RowKind,
) -> Vec<u32> {
    use crate::qwen_image21::layout::SegmentKind;
    layout
        .segments()
        .iter()
        .filter(|segment| {
            matches!(
                (segment.kind, kind),
                (SegmentKind::Text, RowKind::Text)
                    | (SegmentKind::ConditionImage { .. }, RowKind::Condition)
                    | (SegmentKind::Target, RowKind::Target)
            )
        })
        .flat_map(|segment| (segment.start as u32)..(segment.end() as u32))
        .collect()
}

/// One measured layout: a text-to-image prompt, or an image-conditioned one
/// whose condition rows are real VAE latents.
struct StudyCase {
    name: &'static str,
    text: Tensor,
    cond_latents: Option<Tensor>,
    layout: crate::qwen_image21::layout::QwenImage21JointLayout,
}

/// Running worst values across every measured site.
#[derive(Default)]
struct Worst {
    fused: (f64, String),
    hand: f64,
    ratio: f64,
    fused_by_rows: std::collections::BTreeMap<&'static str, f64>,
}

/// Run the legacy forward over `case` at three timesteps and measure every
/// block's norm1 and norm2 inputs, per row kind (the prefix — text and
/// condition rows — takes the t=0 modulation row; upstream's
/// `target_token_mask`, `transformer_qwenimage21.py`).
fn study_case(
    transformer: &QwenImage21Transformer,
    case: &StudyCase,
    noise: &Tensor,
    sites: &mut Vec<serde_json::Value>,
    worst: &mut Worst,
) -> Result<()> {
    let (device, dtype) = (noise.device(), noise.dtype());
    let eps = transformer.cfg.eps;
    let inner = transformer.cfg.inner_dim();
    let layout = &case.layout;
    let prefix_len = layout.prefix_len();
    let target_len = layout.target_tokens();
    // The tables the legacy path itself builds for this layout.
    let request = crate::qwen_image21::exec_path::Qwen21RequestShape {
        has_v032_bytes: layout.condition_tokens() == 0,
    };
    let (rope_cos, rope_sin) = crate::qwen_image21::layout::QwenImage21JointLayout::rope_tables(
        layout.rope(),
        transformer.cfg.axes_dims_rope,
        transformer.exec.rope_angles(request),
        transformer.rope_table_dtype(dtype, layout),
        device,
    )?;
    let plan = layout.attention_plan(false);
    let mut kinds = Vec::new();
    for kind in [RowKind::Text, RowKind::Condition, RowKind::Target] {
        let rows = rows_of(layout, kind);
        if !rows.is_empty() {
            let count = rows.len();
            kinds.push((kind, Tensor::from_vec(rows, count, device)?));
        }
    }
    for timestep in [1.0f64, 0.5, 0.05] {
        // Magnitude-matched stand-in for the sampler's latent at this sigma.
        let latents = (noise * timestep)?;
        let mut hidden =
            transformer.assemble_joint(&case.text, case.cond_latents.as_ref(), &latents, layout)?;
        let temb = transformer
            .time_text_embed
            .forward(&[timestep, 0.0], dtype, device)?;
        let modulation = transformer
            .modulation
            .forward(&candle_nn::Activation::Silu.forward(&temb)?)?;
        let real = modulation.narrow(0, 0, 1)?;
        let zero = modulation.narrow(0, 1, 1)?;
        let per_token = Tensor::cat(
            &[
                &zero
                    .unsqueeze(1)?
                    .broadcast_as((1, prefix_len, 4 * inner))?,
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
                    let attn = block.attn.forward_block_causal(
                        &normalized,
                        &rope_cos,
                        &rope_sin,
                        &plan,
                        prefix_len,
                        LayerCache::Disabled,
                    )?;
                    (
                        (&hidden + gate.tanh()?.broadcast_mul(&attn)?)?,
                        &block.norm2,
                    )
                };
                for (kind, rows) in &kinds {
                    let row = if *kind == RowKind::Target {
                        &real
                    } else {
                        &zero
                    };
                    let scale = row.narrow(D::Minus1, offset, inner)?;
                    let measured = measure_site(&x.index_select(rows, 1)?, &scale, norm, eps)?;
                    let name = format!(
                        "{}/t{timestep}/block{index}/{site}/{}",
                        case.name,
                        kind.label()
                    );
                    if measured.fused.max_rel > worst.fused.0 {
                        worst.fused = (measured.fused.max_rel, name.clone());
                    }
                    let by_rows = worst.fused_by_rows.entry(kind.label()).or_default();
                    *by_rows = by_rows.max(measured.fused.max_rel);
                    worst.hand = worst.hand.max(measured.hand.max_rel);
                    worst.ratio = worst.ratio.max(measured.mean_over_std);
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
            hidden = block.forward_block_causal(
                &hidden,
                &BlockModulation::PerToken(per_token.clone()),
                &rope_cos,
                &rope_sin,
                &plan,
                prefix_len,
                LayerCache::Disabled,
            )?;
        }
        eprintln!(
            "adaln {} t={timestep}: worst fused so far {:?}, hand {:.3e}",
            case.name, worst.fused, worst.hand
        );
    }
    Ok(())
}

fn required_path(name: &str, what: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} must name {what}")))
}

/// Run the legacy forward on the real checkpoint (1024² target, three
/// timesteps) over a text-to-image prompt AND an image-conditioned one (the
/// P8 prompt's captured Qwen3-VL embeddings and its reference's captured VAE
/// latents), and measure every block's norm1 and norm2 inputs for text,
/// condition-image and target rows separately. Fails unless the fused
/// kernel's worst max relative error is within [`FUSED_ADALN_GATE_MAX_REL`].
#[test]
#[ignore = "requires installed Qwen Image 2.1 weights, QWEN_IMAGE21_FIXTURES and an idle, exclusive CUDA GPU"]
fn official_cuda_adaln_precision_study() -> Result<()> {
    let root = required_path("QWEN_IMAGE21_MODEL_ROOT", "a mold models dir");
    let output = required_path("QWEN_IMAGE21_BENCH_OUTPUT", "the receipt directory");
    let fixtures = required_path("QWEN_IMAGE21_FIXTURES", "the large-capture directory");
    std::fs::create_dir_all(&output)?;
    let prompt =
        std::env::var("QWEN_IMAGE21_BENCH_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let seed: u64 = env_or("QWEN_IMAGE21_BENCH_SEED", 210001)?;
    let device = Device::new_cuda(0).expect("the adaLN precision study needs a CUDA device");
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
    install_mode(&mut transformer, &bench_mode("legacy")?);
    let (lh, lw) = (64usize, 64usize);
    let noise = crate::engine::seeded_randn(
        seed,
        &[1, lh * lw, QWEN_IMAGE_21_LATENT_CHANNELS],
        &device,
        dtype,
    )?;
    let t2i = StudyCase {
        name: "t2i",
        text: conditioning.embeddings.to_dtype(dtype)?,
        cond_latents: None,
        layout: crate::qwen_image21::layout::QwenImage21JointLayout::text_to_image(
            &conditioning.valid_tokens,
            (lh, lw),
        )?,
    };
    // The P8 edit prompt (one 832x1248 reference, 52x78 latents), from the
    // upstream captures so the condition rows are the real ones.
    // `load_capture` reads the captures' BOOL masks, which candle cannot.
    let embeds = crate::qwen_image21::parity_tests::load_capture(
        &fixtures.join("p3_p8_pos_bf16.safetensors"),
    );
    let slots: Vec<bool> = embeds["image_pad_mask"]
        .flatten_all()?
        .to_dtype(DType::U8)?
        .to_vec1::<u8>()?
        .into_iter()
        .map(|value| value != 0)
        .collect();
    let encoded = crate::qwen_image21::parity_tests::load_capture(
        &fixtures.join("p4_vae_encode_fp32.safetensors"),
    );
    let referenced = StudyCase {
        name: "reference",
        text: embeds["prompt_embeds"]
            .to_device(&device)?
            .to_dtype(dtype)?,
        cond_latents: Some(
            encoded["opaque_packed"]
                .to_device(&device)?
                .to_dtype(dtype)?,
        ),
        layout: crate::qwen_image21::layout::QwenImage21JointLayout::build(
            &slots,
            &[vec![true; slots.len()]],
            &[(52, 78)],
            (lh, lw),
        )?,
    };
    let mut sites = Vec::new();
    let mut worst = Worst::default();
    for case in [&t2i, &referenced] {
        study_case(&transformer, case, &noise, &mut sites, &mut worst)?;
    }
    let passes = worst.fused.0 <= FUSED_ADALN_GATE_MAX_REL;
    let receipt = json!({
        "prompt": prompt,
        "reference_case": "p3_p8_pos_bf16 embeddings + p4_vae_encode_fp32 opaque_packed (52x78)",
        "seed": seed,
        "canvas": "1024x1024",
        "kernel": "candle-kernels reduce.cu layernorm: one-pass F32 E[x^2]-mean^2",
        "gate_max_rel": FUSED_ADALN_GATE_MAX_REL,
        "worst_fused_max_rel": worst.fused.0,
        "worst_fused_site": worst.fused.1,
        "worst_fused_max_rel_by_rows": worst.fused_by_rows,
        "worst_hand_max_rel": worst.hand,
        "worst_row_mean_over_std": worst.ratio,
        "passes": passes,
        "sites": sites,
    });
    std::fs::write(
        output.join("receipt-adaln-precision.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    eprintln!(
        "adaln worst fused max_rel {:.3e} at {} (by rows {:?}; hand {:.3e}, worst |mean|/std {:.2})",
        worst.fused.0, worst.fused.1, worst.fused_by_rows, worst.hand, worst.ratio
    );
    assert!(
        worst.fused_by_rows.contains_key("condition"),
        "the study measured no condition-image rows"
    );
    assert!(
        passes,
        "fused adaLN worst max_rel {:.3e} at {} exceeds the {FUSED_ADALN_GATE_MAX_REL:e} gate",
        worst.fused.0, worst.fused.1
    );
    Ok(())
}

#[test]
fn rows_of_partitions_a_reference_layout_by_kind() {
    let slots = [false, false, false, true, true, false, false, false];
    let layout = crate::qwen_image21::layout::QwenImage21JointLayout::build(
        &slots,
        &[vec![true; 8]],
        &[(2, 4)],
        (4, 4),
    )
    .unwrap();
    assert_eq!(rows_of(&layout, RowKind::Text), vec![0, 1, 2, 11, 12, 13]);
    assert_eq!(
        rows_of(&layout, RowKind::Condition),
        (3..11).collect::<Vec<u32>>()
    );
    assert_eq!(
        rows_of(&layout, RowKind::Target),
        (14..30).collect::<Vec<u32>>()
    );
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
