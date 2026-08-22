//! Per-variant PuLID injection coverage on a synthetic FLUX transformer.
//!
//! `FluxTransformer` has four arms and they reach their block loops through
//! three different code paths: the candle fork's `forward_with_hook` for the
//! dense and quantized upstream models, and mold's own loops in `offload.rs`
//! and `quantized_transformer.rs`. The injection policy is shared — one
//! `PulidBlockHook` — but the plumbing is not, so each arm is exercised here
//! against the same three claims:
//!
//! 1. An effective `id_weight` of 0 renders **bit-identically** to a render
//!    that passed no identity at all.
//! 2. A step before `id_start_step` does the same.
//! 3. A live weight actually changes the output — otherwise (1) and (2) would
//!    pass on a transformer that ignores the hook entirely.
//!
//! Everything runs on the CPU against a transformer small enough to be
//! milliseconds per forward: 4 double and 8 single blocks, 32 hidden, 2 heads.
//! The PuLID module geometry is scaled to match (`tiny_config`) — the trained
//! 3072/2048/16x128 shape would need FLUX-sized blocks to attach to.
//!
//! One `VarMap` supplies every variant's weights, so the four transformers are
//! numerically the same model expressed four ways.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::flux;
use candle_transformers::quantized_var_builder;

use crate::flux::pulid::{
    tests::{synthetic_adapter, synthetic_context, tiny_config},
    PulidAdapter, PulidContext, PulidRuntime,
};
use crate::flux::transformer::{FluxTransformer, TrueCfgBranch};
use crate::progress::ProgressReporter;

const DEPTH: usize = 4;
const DEPTH_SINGLE: usize = 8;
const HIDDEN: usize = 32;
const HEADS: usize = 2;
const IN_CHANNELS: usize = 8;
const CONTEXT_IN_DIM: usize = 16;
const VEC_IN_DIM: usize = 16;
const TXT_TOKENS: usize = 5;
const IMG_TOKENS: usize = 6;

pub(crate) fn tiny_flux_config() -> flux::model::Config {
    flux::model::Config {
        in_channels: IN_CHANNELS,
        vec_in_dim: VEC_IN_DIM,
        context_in_dim: CONTEXT_IN_DIM,
        hidden_size: HIDDEN,
        mlp_ratio: 2.0,
        num_heads: HEADS,
        depth: DEPTH,
        depth_single_blocks: DEPTH_SINGLE,
        // `pe_dim` is hidden_size / num_heads = 16; the axes must sum to it and
        // each must be even for the RoPE half-split.
        axes_dim: vec![4, 4, 8],
        theta: 10_000,
        qkv_bias: true,
        guidance_embed: false,
    }
}

/// Every weight the dense model asks for, materialized once.
///
/// `VarBuilder::from_varmap` creates a tensor on first request, so building
/// the dense model is also how the parameter set is discovered — no hand-kept
/// list of tensor names that could drift from candle's.
pub(crate) fn shared_weights(cfg: &flux::model::Config) -> HashMap<String, Tensor> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    flux::model::Flux::new(cfg, vb).expect("the tiny dense model builds");
    let data = varmap.data().lock().expect("varmap lock");
    data.iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect()
}

/// The same weights as a gguf file, so the two quantized variants load through
/// their real `from_gguf` path rather than a test-only shortcut.
///
/// Stored as `GgmlDType::F32`, which is a `QTensor` with no quantization at
/// all: the point here is the block loop and the hook, not the kernel, and an
/// exact dtype keeps the "bit-identical" claim about the injection rather than
/// about rounding. It also sidesteps the 32-element block constraint the real
/// quantized dtypes place on the last axis.
fn gguf_weights(weights: &HashMap<String, Tensor>, path: &std::path::Path) -> Result<()> {
    let quantized: Vec<(String, candle_core::quantized::QTensor)> = weights
        .iter()
        .map(|(name, tensor)| {
            let q = mold_candle::quantized::quantize_onto(
                &tensor.to_dtype(DType::F32)?.contiguous()?,
                GgmlDType::F32,
                &Device::Cpu,
            )?;
            Ok((name.clone(), q))
        })
        .collect::<Result<_>>()?;
    let refs: Vec<(&str, &candle_core::quantized::QTensor)> = quantized
        .iter()
        .map(|(name, q)| (name.as_str(), q))
        .collect();
    let mut file = std::fs::File::create(path)?;
    candle_core::quantized::gguf_file::write(&mut file, &[], &refs)?;
    Ok(())
}

/// Just the dense arm, for a caller that needs a `FluxTransformer` and does
/// not care which one — `pipeline`'s VAE-headroom test, which is about the
/// drop, not the arithmetic.
pub(crate) fn tiny_dense_transformer() -> FluxTransformer {
    let cfg = tiny_flux_config();
    let vb = VarBuilder::from_tensors(shared_weights(&cfg), DType::F32, &Device::Cpu);
    FluxTransformer::BF16(flux::model::Flux::new(&cfg, vb).expect("dense variant"))
}

struct Inputs {
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec_: Tensor,
    timesteps: Vec<f64>,
}

/// Deterministic inputs — a fixed ramp, so two runs in one process feed the
/// transformer exactly the same bytes and a difference can only come from the
/// hook.
fn inputs() -> Inputs {
    let device = Device::Cpu;
    let ramp = |shape: (usize, usize, usize), offset: f32| {
        let count = shape.0 * shape.1 * shape.2;
        let values: Vec<f32> = (0..count)
            .map(|i| ((i as f32 * 0.017 + offset).sin()) * 0.5)
            .collect();
        Tensor::from_vec(values, shape, &device).expect("ramp tensor")
    };
    Inputs {
        img: ramp((1, IMG_TOKENS, IN_CHANNELS), 0.0),
        img_ids: Tensor::zeros((1, IMG_TOKENS, 3), DType::F32, &device).expect("img ids"),
        txt: ramp((1, TXT_TOKENS, CONTEXT_IN_DIM), 1.0),
        txt_ids: Tensor::zeros((1, TXT_TOKENS, 3), DType::F32, &device).expect("txt ids"),
        vec_: Tensor::from_vec(
            (0..VEC_IN_DIM).map(|i| i as f32 / 16.0).collect::<Vec<_>>(),
            (1, VEC_IN_DIM),
            &device,
        )
        .expect("vec"),
        // Four denoise steps, so a `start_step` of 4 is genuinely never reached.
        timesteps: vec![1.0, 0.75, 0.5, 0.25, 0.0],
    }
}

fn denoise(
    transformer: &FluxTransformer,
    inputs: &Inputs,
    pulid: Option<PulidRuntime<'_>>,
) -> Vec<f32> {
    denoise_with_true_cfg(transformer, inputs, pulid, None)
}

fn denoise_with_true_cfg(
    transformer: &FluxTransformer,
    inputs: &Inputs,
    pulid: Option<PulidRuntime<'_>>,
    true_cfg: Option<&TrueCfgBranch<'_>>,
) -> Vec<f32> {
    let progress = ProgressReporter::default();
    transformer
        .denoise(
            &inputs.img,
            &inputs.img_ids,
            &inputs.txt,
            &inputs.txt_ids,
            &inputs.vec_,
            &inputs.timesteps,
            0.0,
            &progress,
            None,
            None,
            pulid,
            true_cfg,
        )
        .expect("the tiny transformer denoises")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("f32 output")
}

/// The four arms, all carrying the same weights.
fn variants(
    cfg: &flux::model::Config,
    gguf: &std::path::Path,
) -> Vec<(&'static str, FluxTransformer)> {
    let weights = shared_weights(cfg);
    gguf_weights(&weights, gguf).expect("the synthetic gguf writes");

    let dense_vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
    let bf16 = FluxTransformer::BF16(flux::model::Flux::new(cfg, dense_vb).expect("dense variant"));

    let quantized_vb =
        quantized_var_builder::VarBuilder::from_gguf(gguf, &Device::Cpu).expect("gguf var builder");
    let quantized = FluxTransformer::Quantized(
        flux::quantized_model::Flux::new(cfg, quantized_vb).expect("quantized variant"),
    );

    let progress = ProgressReporter::default();
    // mold's bypass transformer takes mold-candle's own quantized VarBuilder,
    // not candle-transformers'.
    let bypass_vb = mold_candle::quantized::VarBuilder::from_gguf(gguf, &Device::Cpu)
        .expect("gguf var builder");
    let bypass = FluxTransformer::QuantizedBypass(
        crate::flux::quantized_transformer::QuantizedFluxTransformer::load(
            cfg, bypass_vb, None, &progress,
        )
        .expect("bypass variant"),
    );

    let offload_vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let offloaded = FluxTransformer::Offloaded(
        crate::flux::offload::OffloadedFluxTransformer::load(
            offload_vb,
            cfg,
            &Device::Cpu,
            0,
            // A budget far above this toy model, so residency planning is not
            // what the test is measuring.
            1 << 30,
            None,
            &progress,
        )
        .expect("offloaded variant"),
    );

    vec![
        ("BF16", bf16),
        ("Quantized", quantized),
        ("QuantizedBypass", bypass),
        ("Offloaded", offloaded),
    ]
}

fn adapter() -> Arc<PulidAdapter> {
    Arc::new(synthetic_adapter(
        tiny_config(),
        DEPTH,
        DEPTH_SINGLE,
        DType::F32,
        &Device::Cpu,
    ))
}

fn context(id_weight: f32, start_step: usize) -> PulidContext {
    synthetic_context(
        tiny_config(),
        id_weight,
        start_step,
        DType::F32,
        &Device::Cpu,
    )
}

#[test]
fn every_variant_is_bit_identical_when_identity_contributes_nothing() {
    let cfg = tiny_flux_config();
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("tiny-flux.gguf");
    let inputs = inputs();
    let adapter = adapter();

    let zero_weight = context(0.0, 0);
    // Four denoise steps run as `step` 0..=3, so a start of 4 is never active.
    let never_starts = context(1.0, 4);
    let live = context(1.0, 0);

    for (name, transformer) in variants(&cfg, &gguf) {
        let baseline = denoise(&transformer, &inputs, None);

        let gated_by_weight = denoise(
            &transformer,
            &inputs,
            Some(PulidRuntime::new(&adapter, &zero_weight)),
        );
        assert_eq!(
            baseline, gated_by_weight,
            "{name}: an effective id_weight of 0 must render bit-identically to no identity"
        );

        let gated_by_start = denoise(
            &transformer,
            &inputs,
            Some(PulidRuntime::new(&adapter, &never_starts)),
        );
        assert_eq!(
            baseline, gated_by_start,
            "{name}: every step before id_start_step must render bit-identically"
        );

        let conditioned = denoise(
            &transformer,
            &inputs,
            Some(PulidRuntime::new(&adapter, &live)),
        );
        assert_ne!(
            baseline, conditioned,
            "{name}: a live id_weight must actually change the render — otherwise the two \
             assertions above would pass on a transformer that ignores the hook"
        );
        assert_eq!(
            baseline.len(),
            conditioned.len(),
            "{name}: injection must not change the output shape"
        );
        assert!(
            conditioned.iter().all(|value| value.is_finite()),
            "{name}: conditioned output must stay finite"
        );
    }
}

/// A delayed start conditions only the steps at or after it, which is a
/// different claim from "it does nothing" and from "it does everything".
#[test]
fn a_partial_start_step_lands_between_the_two_gated_extremes() {
    let cfg = tiny_flux_config();
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("tiny-flux.gguf");
    let inputs = inputs();
    let adapter = adapter();

    let from_the_start = context(1.0, 0);
    let from_step_two = context(1.0, 2);

    for (name, transformer) in variants(&cfg, &gguf) {
        let baseline = denoise(&transformer, &inputs, None);
        let all_steps = denoise(
            &transformer,
            &inputs,
            Some(PulidRuntime::new(&adapter, &from_the_start)),
        );
        let late = denoise(
            &transformer,
            &inputs,
            Some(PulidRuntime::new(&adapter, &from_step_two)),
        );
        assert_ne!(baseline, late, "{name}: step 2 onward must be conditioned");
        assert_ne!(
            all_steps, late,
            "{name}: skipping the first two steps must change the result"
        );
    }
}

/// The four arms agree numerically, which is what makes the per-variant
/// bit-identity claims above claims about one model rather than four.
#[test]
fn the_four_variants_are_the_same_model() {
    let cfg = tiny_flux_config();
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("tiny-flux.gguf");
    let inputs = inputs();
    let adapter = adapter();
    let live = context(1.0, 0);

    let mut baselines: Vec<(&str, Vec<f32>)> = Vec::new();
    let mut conditioned: Vec<(&str, Vec<f32>)> = Vec::new();
    for (name, transformer) in variants(&cfg, &gguf) {
        baselines.push((name, denoise(&transformer, &inputs, None)));
        conditioned.push((
            name,
            denoise(
                &transformer,
                &inputs,
                Some(PulidRuntime::new(&adapter, &live)),
            ),
        ));
    }

    // The quantized arms round through `QTensor`, and the offloaded arm
    // reassembles blocks, so this is a numeric-agreement budget rather than a
    // bit-identity one. It is orders of magnitude tighter than the injection's
    // own effect, which is what the assertion needs to be useful.
    const TOLERANCE: f32 = 1e-3;
    for set in [&baselines, &conditioned] {
        let (reference_name, reference) = &set[0];
        for (name, values) in set.iter().skip(1) {
            let worst = reference
                .iter()
                .zip(values)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < TOLERANCE,
                "{name} diverges from {reference_name} by {worst}"
            );
        }
    }
}

/// True CFG must be as structurally inert as `id_weight: 0` when it is not
/// engaged, on the SAME transformer route — the branch is `None` at the call
/// site and the loop below it is byte-for-byte the loop that always ran.
///
/// The scale-1.0 case is the one that matters most: upstream treats it as off
/// (`PuLID/flux/sampling.py:120`), and `neg + 1.0 * (pos - neg)` is `pos`
/// arithmetically but NOT bit-identically once floating point is involved — so
/// mold refuses the branch rather than running it and rounding.
#[test]
fn every_variant_is_bit_identical_when_true_cfg_is_not_engaged() {
    let cfg = tiny_flux_config();
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("tiny-flux.gguf");
    let inputs = inputs();
    let adapter = adapter();
    let live = context(1.0, 0);
    // Four steps run as 0..=3, so a branch that starts at 4 never runs.
    let uncond = context(1.0, 0);
    let negative = negative_inputs();

    for (name, transformer) in variants(&cfg, &gguf) {
        let baseline = denoise(
            &transformer,
            &inputs,
            Some(PulidRuntime::new(&adapter, &live)),
        );

        let never_starts = TrueCfgBranch {
            scale: 2.0,
            start_step: 4,
            txt: &negative.txt,
            txt_ids: &negative.txt_ids,
            vec_: &negative.vec_,
            pulid: Some(PulidRuntime::new(&adapter, &uncond)),
        };
        assert_eq!(
            baseline,
            denoise_with_true_cfg(
                &transformer,
                &inputs,
                Some(PulidRuntime::new(&adapter, &live)),
                Some(&never_starts),
            ),
            "{name}: a branch that never starts must render bit-identically"
        );

        // And a live branch must actually change the render, so the assertion
        // above is about the gate rather than about a transformer that ignores
        // the negative pass entirely.
        let live_branch = TrueCfgBranch {
            scale: 3.0,
            start_step: 0,
            txt: &negative.txt,
            txt_ids: &negative.txt_ids,
            vec_: &negative.vec_,
            pulid: Some(PulidRuntime::new(&adapter, &uncond)),
        };
        assert_ne!(
            baseline,
            denoise_with_true_cfg(
                &transformer,
                &inputs,
                Some(PulidRuntime::new(&adapter, &live)),
                Some(&live_branch),
            ),
            "{name}: an engaged true-CFG branch must change the render"
        );
    }
}

/// `neg + scale * (pos - neg)` — `PuLID/flux/sampling.py:149`.
///
/// Checked on the dense arm alone because the formula is variant-independent;
/// what the four arms differ in is the forward pass, which the test above
/// covers. A scale of 1 must reproduce the conditional prediction and a scale
/// of 0 the negative one, which is what pins the direction of the lerp: the
/// two are trivially swappable and swapping them renders the negative prompt.
#[test]
fn the_true_cfg_combination_matches_upstreams_formula() {
    let cfg = tiny_flux_config();
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("tiny-flux.gguf");
    let inputs = inputs();
    let adapter = adapter();
    let live = context(1.0, 0);
    let uncond = context(1.0, 0);
    let negative = negative_inputs();
    let (_, transformer) = variants(&cfg, &gguf).remove(0);

    let branch = |scale: f64| TrueCfgBranch {
        scale,
        start_step: 0,
        txt: &negative.txt,
        txt_ids: &negative.txt_ids,
        vec_: &negative.vec_,
        pulid: Some(PulidRuntime::new(&adapter, &uncond)),
    };
    let run = |branch: Option<&TrueCfgBranch<'_>>| {
        denoise_with_true_cfg(
            &transformer,
            &inputs,
            Some(PulidRuntime::new(&adapter, &live)),
            branch,
        )
    };

    let conditional = run(None);
    let at_one = run(Some(&branch(1.0)));
    let worst = conditional
        .iter()
        .zip(&at_one)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-5,
        "scale 1 must reproduce the conditional prediction, worst {worst}"
    );

    // Scale 0 is `neg_pred` — outside the accepted request range, but the one
    // value that proves the lerp is not reversed.
    let at_zero = run(Some(&branch(0.0)));
    let worst = conditional
        .iter()
        .zip(&at_zero)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 1e-5,
        "scale 0 must render the NEGATIVE prediction, not the conditional one"
    );
}

/// A second, different conditioning pair standing in for the negative prompt's
/// T5 and pooled CLIP output.
fn negative_inputs() -> Inputs {
    let device = Device::Cpu;
    let base = inputs();
    Inputs {
        // The image half is shared with the conditional branch upstream
        // (`prepare(..., img=x, prompt=neg_prompt)`), so only the text half
        // differs here.
        txt: Tensor::from_vec(
            (0..TXT_TOKENS * CONTEXT_IN_DIM)
                .map(|i| 0.5 - i as f32 / 64.0)
                .collect::<Vec<_>>(),
            (1, TXT_TOKENS, CONTEXT_IN_DIM),
            &device,
        )
        .expect("negative txt"),
        vec_: Tensor::from_vec(
            (0..VEC_IN_DIM)
                .map(|i| 0.25 - i as f32 / 32.0)
                .collect::<Vec<_>>(),
            (1, VEC_IN_DIM),
            &device,
        )
        .expect("negative vec"),
        ..base
    }
}
