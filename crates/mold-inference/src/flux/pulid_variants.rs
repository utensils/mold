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
use crate::flux::transformer::FluxTransformer;
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

fn tiny_flux_config() -> flux::model::Config {
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
fn shared_weights(cfg: &flux::model::Config) -> HashMap<String, Tensor> {
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
