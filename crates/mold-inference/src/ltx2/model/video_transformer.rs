#![allow(clippy::too_many_arguments)]

// LTX-2 video transformer — adapted from Mold's LTX Video model.
// This keeps the proven video-only denoiser structure but patches the positional
// embedding path and config surface to match the native LTX-2 checkpoints.
//
// That LTX Video model was ported from FerrisMind/candle-video (Apache 2.0,
// Copyright 2025 FerrisMind, https://github.com/FerrisMind/candle-video), so
// portions of this file — the PixArt-Alpha timestep/caption embeddings, the
// attention-mask bias, AdaLN gating, and block wiring — remain that code under
// the Apache License, Version 2.0. See THIRD_PARTY_NOTICES.md at the repo root.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn as nn;
use nn::{Module, VarBuilder};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::adaptive_offload::AdaptiveResidencyPlan;
use crate::ltx2::guidance::{BatchedPerturbationConfig, PerturbationType};
use crate::ltx2::lora::{LinearLoraAdapter, Ltx2LoraRegistry};

use super::rope::LtxRopeType;

pub(crate) fn ltx2_block_debug_enabled() -> bool {
    std::env::var_os("MOLD_LTX2_DEBUG_BLOCKS").is_some()
}

fn ltx2_block_detail_target() -> Option<usize> {
    std::env::var("MOLD_LTX2_DEBUG_BLOCK_DETAIL")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn ltx2_block_detail_enabled(index: usize) -> bool {
    ltx2_block_detail_target() == Some(index)
}

pub(crate) fn ltx2_load_debug_enabled() -> bool {
    crate::runtime_env::value("MOLD_LTX2_DEBUG_LOAD_BLOCKS").is_some()
}

pub(crate) fn tensor_debug_stats(xs: &Tensor) -> Result<(f32, f32, f32)> {
    let flat = xs.flatten_all()?.to_dtype(DType::F32)?;
    let mean = flat.mean_all()?.to_scalar::<f32>()?;
    let abs_mean = flat.abs()?.mean_all()?.to_scalar::<f32>()?;
    let abs_max = flat.abs()?.max_all()?.to_scalar::<f32>()?;
    Ok((mean, abs_mean, abs_max))
}

pub(crate) fn log_detail_tensor(index: usize, label: &str, xs: &Tensor) -> Result<()> {
    if !ltx2_block_detail_enabled(index) {
        return Ok(());
    }
    let (mean, abs_mean, abs_max) = tensor_debug_stats(xs)?;
    eprintln!(
        "[ltx2-block-detail] block={index} {label}(mean={mean:.6}, abs_mean={abs_mean:.6}, abs_max={abs_max:.6})"
    );
    Ok(())
}

fn should_synchronize_streaming_layer(
    index: usize,
    total_layers: usize,
    prefetch_count: usize,
) -> bool {
    let interval = prefetch_count.max(1);
    let layer_num = index + 1;
    layer_num.is_multiple_of(interval) || layer_num == total_layers
}

// ---------------------------------------------------------------------------
// Output wrapper
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Ltx2VideoTransformer3DModelConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[allow(dead_code)]
    pub patch_size: usize,
    #[allow(dead_code)]
    pub patch_size_t: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub cross_attention_dim: usize,
    pub num_layers: usize,
    pub qk_norm: String,
    #[allow(dead_code)]
    pub norm_elementwise_affine: bool,
    pub norm_eps: f64,
    pub caption_channels: usize,
    pub caption_projection_in_transformer: bool,
    pub attention_bias: bool,
    pub attention_out_bias: bool,
    pub positional_embedding_theta: f64,
    pub positional_embedding_max_pos: Vec<usize>,
    pub use_middle_indices_grid: bool,
    pub rope_type: LtxRopeType,
    pub double_precision_rope: bool,
    pub audio_num_attention_heads: usize,
    pub audio_attention_head_dim: usize,
    pub audio_in_channels: usize,
    pub audio_out_channels: usize,
    pub audio_cross_attention_dim: usize,
    pub audio_positional_embedding_max_pos: Vec<usize>,
    pub apply_gated_attention: bool,
    pub av_ca_timestep_scale_multiplier: f64,
    pub cross_attention_adaln: bool,
    pub streaming_prefetch_count: usize,
}

impl Default for Ltx2VideoTransformer3DModelConfig {
    fn default() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: 32,
            attention_head_dim: 128,
            cross_attention_dim: 4096,
            num_layers: 48,
            qk_norm: "rms_norm".to_string(),
            norm_elementwise_affine: false,
            norm_eps: 1e-6,
            caption_channels: 3840,
            caption_projection_in_transformer: true,
            attention_bias: true,
            attention_out_bias: true,
            positional_embedding_theta: 10_000.0,
            positional_embedding_max_pos: vec![20, 2048, 2048],
            use_middle_indices_grid: true,
            rope_type: LtxRopeType::Split,
            double_precision_rope: false,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_cross_attention_dim: 2048,
            audio_positional_embedding_max_pos: vec![20],
            apply_gated_attention: false,
            av_ca_timestep_scale_multiplier: 1000.0,
            cross_attention_adaln: false,
            streaming_prefetch_count: 1,
        }
    }
}

impl Ltx2VideoTransformer3DModelConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

// ---------------------------------------------------------------------------
// LayerNorm without affine parameters (elementwise_affine=False)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let last_dim = xs_f32.dim(D::Minus1)?;
        let mean = (xs_f32.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let xc = xs_f32.broadcast_sub(&mean)?;
        let var = (xc.sqr()?.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let denom = (var + self.eps)?.sqrt()?;
        xc.broadcast_div(&denom)?.to_dtype(dtype)
    }
}

// ---------------------------------------------------------------------------
// RMSNorm with optional affine weight
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, elementwise_affine: bool, vb: VarBuilder) -> Result<Self> {
        let weight = if elementwise_affine {
            Some(vb.get(dim, "weight")?)
        } else {
            None
        };
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let dim = xs_f32.dim(D::Minus1)? as f64;
        let ms = xs_f32
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .affine(1.0 / dim, 0.0)?;
        let denom = ms.affine(1.0, self.eps)?.sqrt()?;
        let ys_f32 = xs_f32.broadcast_div(&denom)?;
        let mut ys = ys_f32.to_dtype(dtype)?;
        if let Some(w) = &self.weight {
            let rank = ys.rank();
            let mut shape = vec![1usize; rank];
            shape[rank - 1] = w.dims1()?;
            let w = w.reshape(shape)?;
            ys = ys.broadcast_mul(&w)?;
        }
        Ok(ys)
    }
}

// ---------------------------------------------------------------------------
// GELU (approximate) — F32 upcast for numerical stability
// ---------------------------------------------------------------------------

/// Tanh-approximate GELU evaluated in F32.
///
/// `Tensor::gelu` is candle's fused single-kernel form of the same polynomial
/// `0.5·v·(1 + tanh(sqrt(2/π)·(v + 0.044715·v³)))`. The hand-rolled expression
/// this replaced bound eight live full-size F32 buffers; at LTX-2 stage-2 shapes
/// (13,312 tokens × 16,384 feed-forward hidden) that alone peaked near 7 GB.
pub fn gelu_approximate(x: &Tensor) -> Result<Tensor> {
    x.to_dtype(DType::F32)?.gelu()?.to_dtype(x.dtype())
}

// ---------------------------------------------------------------------------
// FP8-aware linear
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) enum LtxLinear {
    Standard {
        linear: nn::Linear,
        adapters: Vec<LinearLoraAdapter>,
    },
    Fp8 {
        weight: Tensor,
        weight_scale: Option<Tensor>,
        input_scale: Option<Tensor>,
        bias: Option<Tensor>,
        adapters: Vec<LinearLoraAdapter>,
    },
    Nvfp4Streaming {
        packed: Tensor,
        block_scales: Tensor,
        tensor_scale: f32,
        out_dim: usize,
        #[allow(dead_code)]
        in_dim: usize,
        bias: Option<Tensor>,
        adapters: Vec<LinearLoraAdapter>,
        cache: Arc<OnceLock<Tensor>>,
    },
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Nvfp4LinearCache {
    entries: Arc<Mutex<HashMap<String, Arc<OnceLock<Tensor>>>>>,
}

impl Nvfp4LinearCache {
    fn entry(&self, key: &str) -> Arc<OnceLock<Tensor>> {
        self.entries
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8InputScaleMode {
    Skip,
    EmulateDivide,
    EmulateMultiply,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8WeightScaleMode {
    Skip,
    Apply,
}

impl LtxLinear {
    #[cfg(test)]
    fn load(
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
        vb: VarBuilder,
        adapters: Vec<LinearLoraAdapter>,
    ) -> Result<Self> {
        Self::load_with_nvfp4_cache(in_dim, out_dim, has_bias, vb, adapters, None, None)
    }

    pub(crate) fn load_with_nvfp4_cache(
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
        vb: VarBuilder,
        adapters: Vec<LinearLoraAdapter>,
        nvfp4_cache: Option<&Nvfp4LinearCache>,
        cache_key: Option<&str>,
    ) -> Result<Self> {
        if vb.contains_tensor("weight.nvfp4_packed") {
            let cpu = Device::Cpu;
            let packed = vb
                .get_unchecked_dtype("weight.nvfp4_packed", DType::U8)?
                .to_device(&cpu)?;
            let block_scales = vb
                .get_unchecked_dtype("weight.nvfp4_block_scales", DType::F8E4M3)?
                .to_device(&cpu)?;
            let tensor_scale = vb
                .get_unchecked_dtype("weight.nvfp4_tensor_scale", DType::F32)?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;

            let packed_dims = packed.dims();
            if packed_dims.len() != 2 {
                candle_core::bail!(
                    "LTX-2 NVFP4 packed weight must be rank 2, got {:?}",
                    packed_dims,
                );
            }
            let n = packed_dims[0];
            let k = packed_dims[1] * 2;
            if n != out_dim || k != in_dim {
                candle_core::bail!(
                    "LTX-2 NVFP4 shape mismatch: checkpoint weight [{n}, {k}], module expected [{out_dim}, {in_dim}]",
                );
            }
            let scale_dims = block_scales.dims();
            let expected_scale_rows = n.div_ceil(128) * 128;
            let expected_scale_cols = (k / crate::nvfp4::NVFP4_BLOCK_SIZE).div_ceil(4) * 4;
            if scale_dims != [expected_scale_rows, expected_scale_cols] {
                candle_core::bail!(
                    "LTX-2 NVFP4 block-scale shape mismatch: checkpoint {:?}, expected [{expected_scale_rows}, {expected_scale_cols}] for packed [{n}, {}]",
                    scale_dims,
                    packed_dims[1],
                );
            }

            let bias = if has_bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            let cache = nvfp4_cache
                .zip(cache_key)
                .map(|(cache, key)| cache.entry(key))
                .unwrap_or_else(|| Arc::new(OnceLock::new()));

            return Ok(Self::Nvfp4Streaming {
                packed,
                block_scales,
                tensor_scale,
                out_dim,
                in_dim,
                bias,
                adapters,
                cache,
            });
        }

        let weight = vb.get((out_dim, in_dim), "weight")?;
        let weight_scale = if vb.contains_tensor("weight_scale") {
            Some(vb.get((), "weight_scale")?)
        } else {
            None
        };
        let input_scale = if vb.contains_tensor("input_scale") {
            Some(vb.get((), "input_scale")?)
        } else {
            None
        };
        let bias = if has_bias {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        if weight.dtype() == DType::F8E4M3 {
            Ok(Self::Fp8 {
                weight,
                weight_scale,
                input_scale,
                bias,
                adapters,
            })
        } else {
            Ok(Self::Standard {
                linear: nn::Linear::new(weight, bias),
                adapters,
            })
        }
    }
}

fn adapter_to_runtime_dtype(tensor: &Tensor, xs: &Tensor, runtime_dtype: DType) -> Result<Tensor> {
    if tensor.device().same_device(xs.device()) {
        tensor.to_dtype(runtime_dtype)
    } else {
        tensor.to_device(xs.device())?.to_dtype(runtime_dtype)
    }
}

fn lora_linear_forward(
    xs: &Tensor,
    a: &Tensor,
    b: &Tensor,
    runtime_dtype: DType,
) -> Result<Tensor> {
    let a = adapter_to_runtime_dtype(a, xs, runtime_dtype)?;
    let b = adapter_to_runtime_dtype(b, xs, runtime_dtype)?;
    let a_t = a.t()?;
    let b_t = b.t()?;
    match *xs.dims() {
        [batch0, batch1, tokens, hidden] => xs
            .reshape((batch0 * batch1 * tokens, hidden))?
            .matmul(&a_t)?
            .matmul(&b_t)?
            .reshape((batch0, batch1, tokens, ())),
        [batch, tokens, hidden] => xs
            .reshape((batch * tokens, hidden))?
            .matmul(&a_t)?
            .matmul(&b_t)?
            .reshape((batch, tokens, ())),
        _ => xs.matmul(&a_t)?.matmul(&b_t),
    }
}

fn apply_linear_loras(
    base: Tensor,
    xs: &Tensor,
    adapters: &[LinearLoraAdapter],
    runtime_dtype: DType,
) -> Result<Tensor> {
    if adapters.is_empty() {
        return Ok(base);
    }
    let mut out = if base.dtype() == runtime_dtype {
        base
    } else {
        base.to_dtype(runtime_dtype)?
    };
    let xs = if xs.dtype() == runtime_dtype {
        xs.clone()
    } else {
        xs.to_dtype(runtime_dtype)?
    };
    for adapter in adapters {
        let delta = lora_linear_forward(&xs, &adapter.a, &adapter.b, runtime_dtype)?;
        out = out.broadcast_add(&delta.affine(adapter.scale, 0.0)?)?;
    }
    Ok(out)
}

pub(crate) fn lora_adapters_for(
    registry: Option<&Ltx2LoraRegistry>,
    key: &str,
) -> Vec<LinearLoraAdapter> {
    registry
        .map(|registry| registry.adapters_for(key))
        .unwrap_or_default()
}

fn dequantize_fp8_weight_for_runtime(
    weight: &Tensor,
    weight_scale: Option<&Tensor>,
    runtime_dtype: DType,
    runtime_device: &Device,
) -> Result<Tensor> {
    // Candle stores F8E4M3 tensors on Metal but cannot cast them there yet, and
    // its CPU cast converts one byte at a time — a stack sample of an LTX-2
    // Metal denoise put 47% of the worker thread inside `F8E4M3::to_f32` and
    // 0.1% in the matmul it feeds. `fp8_widen` does the same conversion through
    // a 256-entry table, in parallel, and is bit-identical by construction.
    //
    // Bridge only the active output-row chunk, so the surrounding streaming
    // block retains its compact FP8 storage and never expands the complete
    // transformer.
    let widened = if weight.dtype() == DType::F8E4M3
        && (runtime_device.is_metal() || !weight.device().same_device(runtime_device))
    {
        crate::ltx2::fp8_widen::widen_f8e4m3(weight, runtime_dtype, runtime_device)?
    } else {
        None
    };
    let mut dequantized = match widened {
        Some(dequantized) => dequantized,
        None if weight.device().same_device(runtime_device) => weight.to_dtype(runtime_dtype)?,
        None if weight.dtype() == DType::F8E4M3 => weight
            .to_device(&Device::Cpu)?
            .to_dtype(runtime_dtype)?
            .to_device(runtime_device)?,
        None => weight.to_device(runtime_device)?.to_dtype(runtime_dtype)?,
    };
    if let Some(scale) = weight_scale {
        let scale = if scale.device().same_device(runtime_device) {
            scale.to_dtype(runtime_dtype)?
        } else {
            scale.to_device(runtime_device)?.to_dtype(runtime_dtype)?
        };
        dequantized = dequantized.broadcast_mul(&scale)?;
    }
    Ok(dequantized)
}

/// Writes one output-dimension chunk into a lazily allocated destination.
///
/// Collecting every chunk in a `Vec` and then `cat`ing them holds the whole
/// output twice; writing into a single preallocated buffer keeps the peak at
/// one output plus the chunk being produced.
fn write_output_chunk(
    dst: &mut Option<Tensor>,
    chunk: &Tensor,
    out_dim: usize,
    offset: usize,
) -> Result<()> {
    if dst.is_none() {
        let mut shape = chunk.dims().to_vec();
        match shape.last_mut() {
            Some(last) => *last = out_dim,
            None => candle_core::bail!("chunked linear output must have at least one dimension"),
        }
        *dst = Some(Tensor::zeros(shape, chunk.dtype(), chunk.device())?);
    }
    let dst = dst
        .as_ref()
        .expect("chunked linear destination allocated above");
    dst.slice_set(&chunk.contiguous()?, D::Minus1, offset)
}

fn finish_output_chunks(dst: Option<Tensor>) -> Result<Tensor> {
    dst.ok_or_else(|| candle_core::Error::msg("chunked linear produced no output chunks"))
}

fn fp8_linear_output_chunk_size(weight: &Tensor) -> Result<usize> {
    let out_dim = weight.dim(0)?;
    if !weight.device().is_cuda() && !weight.device().is_metal() {
        return Ok(out_dim);
    }
    Ok(if out_dim >= 16_384 {
        1_024
    } else if out_dim >= 8_192 {
        1_536
    } else if out_dim >= 4_096 {
        2_048.min(out_dim)
    } else {
        out_dim
    })
}

fn fp8_linear_forward_chunked(
    xs: &Tensor,
    weight: &Tensor,
    weight_scale: Option<&Tensor>,
    runtime_dtype: DType,
    chunk_size: usize,
) -> Result<Tensor> {
    let out_dim = weight.dim(0)?;
    if chunk_size >= out_dim {
        let weight =
            dequantize_fp8_weight_for_runtime(weight, weight_scale, runtime_dtype, xs.device())?;
        let weight_t = weight.t()?;
        return match *xs.dims() {
            [batch0, batch1, tokens, hidden] => xs
                .reshape((batch0 * batch1 * tokens, hidden))?
                .matmul(&weight_t)?
                .reshape((batch0, batch1, tokens, ())),
            [batch, tokens, hidden] => xs
                .reshape((batch * tokens, hidden))?
                .matmul(&weight_t)?
                .reshape((batch, tokens, ())),
            _ => xs.matmul(&weight_t),
        };
    }

    let mut dst = None;
    match *xs.dims() {
        [batch0, batch1, tokens, hidden] => {
            let xs_flat = xs.reshape((batch0 * batch1 * tokens, hidden))?;
            let mut offset = 0;
            while offset < out_dim {
                let rows = chunk_size.min(out_dim - offset);
                let weight_chunk = weight.narrow(0, offset, rows)?.contiguous()?;
                let weight_chunk = dequantize_fp8_weight_for_runtime(
                    &weight_chunk,
                    weight_scale,
                    runtime_dtype,
                    xs.device(),
                )?;
                let chunk = xs_flat
                    .matmul(&weight_chunk.t()?)?
                    .reshape((batch0, batch1, tokens, rows))?;
                write_output_chunk(&mut dst, &chunk, out_dim, offset)?;
                offset += rows;
            }
        }
        [batch, tokens, hidden] => {
            let xs_flat = xs.reshape((batch * tokens, hidden))?;
            let mut offset = 0;
            while offset < out_dim {
                let rows = chunk_size.min(out_dim - offset);
                let weight_chunk = weight.narrow(0, offset, rows)?.contiguous()?;
                let weight_chunk = dequantize_fp8_weight_for_runtime(
                    &weight_chunk,
                    weight_scale,
                    runtime_dtype,
                    xs.device(),
                )?;
                let chunk = xs_flat
                    .matmul(&weight_chunk.t()?)?
                    .reshape((batch, tokens, rows))?;
                write_output_chunk(&mut dst, &chunk, out_dim, offset)?;
                offset += rows;
            }
        }
        _ => {
            let mut offset = 0;
            while offset < out_dim {
                let rows = chunk_size.min(out_dim - offset);
                let weight_chunk = weight.narrow(0, offset, rows)?.contiguous()?;
                let weight_chunk = dequantize_fp8_weight_for_runtime(
                    &weight_chunk,
                    weight_scale,
                    runtime_dtype,
                    xs.device(),
                )?;
                let chunk = xs.matmul(&weight_chunk.t()?)?;
                write_output_chunk(&mut dst, &chunk, out_dim, offset)?;
                offset += rows;
            }
        }
    }
    finish_output_chunks(dst)
}

fn nvfp4_linear_output_chunk_size(out_dim: usize, device: &Device) -> usize {
    if !device.is_cuda() {
        return out_dim;
    }
    if out_dim >= 16_384 {
        1_024
    } else if out_dim >= 8_192 {
        1_536
    } else if out_dim >= 4_096 {
        2_048.min(out_dim)
    } else {
        out_dim
    }
}

fn nvfp4_linear_forward_chunked(
    xs: &Tensor,
    bf16_weight_cpu: &Tensor,
    runtime_dtype: DType,
    chunk_size: usize,
) -> Result<Tensor> {
    let out_dim = bf16_weight_cpu.dim(0)?;
    if chunk_size >= out_dim {
        let weight = bf16_weight_cpu
            .to_device(xs.device())?
            .to_dtype(runtime_dtype)?;
        let weight_t = weight.t()?;
        return match *xs.dims() {
            [batch0, batch1, tokens, hidden] => xs
                .reshape((batch0 * batch1 * tokens, hidden))?
                .matmul(&weight_t)?
                .reshape((batch0, batch1, tokens, ())),
            [batch, tokens, hidden] => xs
                .reshape((batch * tokens, hidden))?
                .matmul(&weight_t)?
                .reshape((batch, tokens, ())),
            _ => xs.matmul(&weight_t),
        };
    }

    let mut dst = None;
    match *xs.dims() {
        [batch0, batch1, tokens, hidden] => {
            let xs_flat = xs.reshape((batch0 * batch1 * tokens, hidden))?;
            let mut offset = 0;
            while offset < out_dim {
                let rows = chunk_size.min(out_dim - offset);
                let weight_chunk = bf16_weight_cpu
                    .narrow(0, offset, rows)?
                    .contiguous()?
                    .to_device(xs.device())?
                    .to_dtype(runtime_dtype)?;
                let chunk = xs_flat
                    .matmul(&weight_chunk.t()?)?
                    .reshape((batch0, batch1, tokens, rows))?;
                write_output_chunk(&mut dst, &chunk, out_dim, offset)?;
                offset += rows;
            }
        }
        [batch, tokens, hidden] => {
            let xs_flat = xs.reshape((batch * tokens, hidden))?;
            let mut offset = 0;
            while offset < out_dim {
                let rows = chunk_size.min(out_dim - offset);
                let weight_chunk = bf16_weight_cpu
                    .narrow(0, offset, rows)?
                    .contiguous()?
                    .to_device(xs.device())?
                    .to_dtype(runtime_dtype)?;
                let chunk = xs_flat
                    .matmul(&weight_chunk.t()?)?
                    .reshape((batch, tokens, rows))?;
                write_output_chunk(&mut dst, &chunk, out_dim, offset)?;
                offset += rows;
            }
        }
        _ => {
            let mut offset = 0;
            while offset < out_dim {
                let rows = chunk_size.min(out_dim - offset);
                let weight_chunk = bf16_weight_cpu
                    .narrow(0, offset, rows)?
                    .contiguous()?
                    .to_device(xs.device())?
                    .to_dtype(runtime_dtype)?;
                let chunk = xs.matmul(&weight_chunk.t()?)?;
                write_output_chunk(&mut dst, &chunk, out_dim, offset)?;
                offset += rows;
            }
        }
    }
    finish_output_chunks(dst)
}

impl Module for LtxLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard { linear, adapters } => {
                let base = linear.forward(xs)?;
                let dtype = base.dtype();
                apply_linear_loras(base, xs, adapters, dtype)
            }
            Self::Fp8 {
                weight,
                weight_scale,
                input_scale,
                bias,
                adapters,
            } => {
                let dtype = match xs.dtype() {
                    DType::F8E4M3 => DType::BF16,
                    other => other,
                };
                let xs = if xs.dtype() == dtype {
                    xs.clone()
                } else {
                    xs.to_dtype(dtype)?
                };
                // Public LTX-2 `fp8-cast` checkpoints can still carry
                // `input_scale` tensors from the export path, but cast mode does
                // not quantize activations at runtime. Keep scaled-mm-style
                // emulation behind an explicit override for debugging and future
                // Hopper-only work.
                let xs = match input_scale {
                    Some(scale) => emulate_static_fp8_input_quantization(&xs, scale, dtype)?,
                    None => xs,
                };
                let chunk_size = fp8_linear_output_chunk_size(weight)?;
                let out = fp8_linear_forward_chunked(
                    &xs,
                    weight,
                    match fp8_weight_scale_mode() {
                        Fp8WeightScaleMode::Skip => None,
                        Fp8WeightScaleMode::Apply => weight_scale.as_ref(),
                    },
                    dtype,
                    chunk_size,
                )?;
                let out = match bias {
                    Some(bias) => out.broadcast_add(&bias.to_dtype(dtype)?),
                    None => Ok(out),
                }?;
                apply_linear_loras(out, &xs, adapters, dtype)
            }
            Self::Nvfp4Streaming {
                packed,
                block_scales,
                tensor_scale,
                out_dim,
                bias,
                adapters,
                cache,
                ..
            } => {
                let _backend = crate::nvfp4::resolve_nvfp4_backend(xs.device())?;
                let dtype = match xs.dtype() {
                    DType::F8E4M3 => DType::BF16,
                    other => other,
                };
                let xs = if xs.dtype() == dtype {
                    xs.clone()
                } else {
                    xs.to_dtype(dtype)?
                };
                let bf16_weight_cpu = match cache.get() {
                    Some(tensor) => tensor,
                    None => {
                        let dequanted = crate::nvfp4::dequant_nvfp4_to_bf16_cpu(
                            packed,
                            block_scales,
                            *tensor_scale,
                        )?;
                        let _ = cache.set(dequanted);
                        cache.get().expect("cache populated above")
                    }
                };
                let chunk_size = nvfp4_linear_output_chunk_size(*out_dim, xs.device());
                let out = nvfp4_linear_forward_chunked(&xs, bf16_weight_cpu, dtype, chunk_size)?;
                let out = match bias {
                    Some(bias) => {
                        let bias = if bias.device().same_device(out.device()) {
                            bias.to_dtype(dtype)?
                        } else {
                            bias.to_device(out.device())?.to_dtype(dtype)?
                        };
                        out.broadcast_add(&bias)
                    }
                    None => Ok(out),
                }?;
                apply_linear_loras(out, &xs, adapters, dtype)
            }
        }
    }
}

fn emulate_static_fp8_input_quantization(
    xs: &Tensor,
    input_scale: &Tensor,
    compute_dtype: DType,
) -> Result<Tensor> {
    let scale_mode = match crate::runtime_env::value("MOLD_LTX2_FP8_INPUT_SCALE_MODE").as_deref() {
        Some("divide") | Some("emulate") => Fp8InputScaleMode::EmulateDivide,
        Some("multiply") => Fp8InputScaleMode::EmulateMultiply,
        Some("skip") | None => Fp8InputScaleMode::Skip,
        Some(_) => Fp8InputScaleMode::Skip,
    };
    let scale = input_scale.to_dtype(compute_dtype)?;
    match scale_mode {
        Fp8InputScaleMode::Skip => Ok(xs.clone()),
        Fp8InputScaleMode::EmulateMultiply => xs
            .broadcast_mul(&scale)?
            .to_dtype(DType::F8E4M3)?
            .to_dtype(compute_dtype)?
            .broadcast_mul(&scale),
        Fp8InputScaleMode::EmulateDivide => xs
            .broadcast_div(&scale)?
            .to_dtype(DType::F8E4M3)?
            .to_dtype(compute_dtype)?
            .broadcast_mul(&scale),
    }
}

fn fp8_weight_scale_mode() -> Fp8WeightScaleMode {
    match crate::runtime_env::value("MOLD_LTX2_FP8_WEIGHT_SCALE_MODE").as_deref() {
        Some("apply") | Some("scaled-mm") => Fp8WeightScaleMode::Apply,
        Some("skip") => Fp8WeightScaleMode::Skip,
        None | Some(_) => Fp8WeightScaleMode::Apply,
    }
}

// ---------------------------------------------------------------------------
// GeluProjection (Linear + GELU approximate)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct GeluProjection {
    proj: LtxLinear,
}

impl GeluProjection {
    fn new(
        dim_in: usize,
        dim_out: usize,
        vb: VarBuilder,
        lora_registry: Option<&Ltx2LoraRegistry>,
        lora_key: &str,
        nvfp4_cache: Option<&Nvfp4LinearCache>,
    ) -> Result<Self> {
        let proj = LtxLinear::load_with_nvfp4_cache(
            dim_in,
            dim_out,
            true,
            vb.pp("proj"),
            lora_adapters_for(lora_registry, lora_key),
            nvfp4_cache,
            Some(lora_key),
        )?;
        Ok(Self { proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(xs)?;
        gelu_approximate(&x)
    }
}

impl Module for GeluProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// FeedForward (GELU projection + linear)
// ---------------------------------------------------------------------------

/// Token-axis chunk size for `FeedForward`.
///
/// The GELU projection widens the hidden dimension 4×, so an unchunked pass at
/// LTX-2 stage-2 shapes holds a `[1, 13312, 16384]` activation (plus its F32
/// GELU staging) live at once. Chunking the token axis makes the working set
/// O(chunk) instead of O(tokens) at any resolution, and composes with the
/// out-dimension chunking the quantized linears already do.
const FEED_FORWARD_TOKEN_CHUNK: usize = 2_048;

#[derive(Clone, Debug)]
pub struct FeedForward {
    net_0: GeluProjection,
    net_2: LtxLinear,
}

impl FeedForward {
    pub fn new(
        dim: usize,
        vb: VarBuilder,
        lora_registry: Option<&Ltx2LoraRegistry>,
        lora_key_prefix: &str,
        nvfp4_cache: Option<&Nvfp4LinearCache>,
    ) -> Result<Self> {
        let hidden = dim * 4;
        let net_0 = GeluProjection::new(
            dim,
            hidden,
            vb.pp("net.0"),
            lora_registry,
            &format!("{lora_key_prefix}.net.0.proj"),
            nvfp4_cache,
        )?;
        let net_2 = LtxLinear::load_with_nvfp4_cache(
            hidden,
            dim,
            true,
            vb.pp("net.2"),
            lora_adapters_for(lora_registry, &format!("{lora_key_prefix}.net.2")),
            nvfp4_cache,
            Some(&format!("{lora_key_prefix}.net.2")),
        )?;
        Ok(Self { net_0, net_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_token_chunked(xs, FEED_FORWARD_TOKEN_CHUNK)
    }

    fn forward_token_chunked(&self, xs: &Tensor, chunk_size: usize) -> Result<Tensor> {
        if chunk_size == 0 || xs.rank() != 3 || xs.dim(1)? <= chunk_size {
            let x = self.net_0.forward(xs)?;
            return self.net_2.forward(&x);
        }

        let (batch, tokens, _) = xs.dims3()?;
        let mut out = None;
        let mut offset = 0;
        while offset < tokens {
            let rows = chunk_size.min(tokens - offset);
            let chunk = xs.narrow(1, offset, rows)?.contiguous()?;
            let chunk = self.net_2.forward(&self.net_0.forward(&chunk)?)?;
            if out.is_none() {
                let dim = chunk.dim(D::Minus1)?;
                out = Some(Tensor::zeros(
                    (batch, tokens, dim),
                    chunk.dtype(),
                    chunk.device(),
                )?);
            }
            let dst = out
                .as_ref()
                .expect("feed-forward destination allocated above");
            dst.slice_set(&chunk.contiguous()?, 1, offset)?;
            offset += rows;
        }
        out.ok_or_else(|| candle_core::Error::msg("feed-forward produced no token chunks"))
    }
}

// ---------------------------------------------------------------------------
// PixArtAlphaTextProjection
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PixArtAlphaTextProjection {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl PixArtAlphaTextProjection {
    #[allow(dead_code)]
    pub fn new(in_features: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_out_features(in_features, hidden_size, hidden_size, vb)
    }

    pub fn new_with_out_features(
        in_features: usize,
        hidden_size: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_1 = nn::linear(in_features, hidden_size, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(hidden_size, out_features, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = gelu_approximate(&x)?;
        self.linear_2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// TimestepEmbedding (two linear layers + SiLU)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TimestepEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl TimestepEmbedding {
    pub fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = x.silu()?;
        self.linear_2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// PixArtAlphaCombinedTimestepSizeEmbeddings
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PixArtAlphaCombinedTimestepSizeEmbeddings {
    timestep_embedder: TimestepEmbedding,
    inv_freq_cache: Arc<TimestepEmbeddingInvFreqCache>,
}

impl PixArtAlphaCombinedTimestepSizeEmbeddings {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep_embedder =
            TimestepEmbedding::new(256, embedding_dim, vb.pp("timestep_embedder"))?;
        Ok(Self {
            timestep_embedder,
            inv_freq_cache: Arc::new(TimestepEmbeddingInvFreqCache::default()),
        })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let timesteps_proj =
            get_timestep_embedding_cached(timestep, 256, true, &self.inv_freq_cache)?;
        self.timestep_embedder.forward(&timesteps_proj)
    }
}

// ---------------------------------------------------------------------------
// AdaLayerNormSingle
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AdaLayerNormSingle {
    emb: PixArtAlphaCombinedTimestepSizeEmbeddings,
    linear: nn::Linear,
}

impl AdaLayerNormSingle {
    #[allow(dead_code)]
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_coefficient(dim, 6, vb)
    }

    pub fn new_with_coefficient(
        dim: usize,
        embedding_coefficient: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let emb = PixArtAlphaCombinedTimestepSizeEmbeddings::new(dim, vb.pp("emb"))?;
        let linear = nn::linear(dim, embedding_coefficient * dim, vb.pp("linear"))?;
        Ok(Self { emb, linear })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<(Tensor, Tensor)> {
        let embedded_timestep = self.emb.forward(timestep)?;
        let x = embedded_timestep.silu()?;
        let x = self.linear.forward(&x)?;
        Ok((x, embedded_timestep))
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal timestep embedding (DDPM-style)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TimestepEmbeddingInvFreqCacheKey {
    embedding_dim: usize,
    device: String,
}

impl TimestepEmbeddingInvFreqCacheKey {
    pub(crate) fn new(embedding_dim: usize, device: &Device) -> Self {
        Self {
            embedding_dim,
            device: tensor_device_cache_key(device),
        }
    }
}

pub(crate) type TimestepEmbeddingInvFreqCache =
    Mutex<HashMap<TimestepEmbeddingInvFreqCacheKey, Tensor>>;

fn tensor_device_cache_key(device: &Device) -> String {
    format!("{device:?}")
}

pub(crate) fn cached_timestep_embedding_inv_freq(
    cache: &TimestepEmbeddingInvFreqCache,
    key: TimestepEmbeddingInvFreqCacheKey,
    device: &Device,
    half: usize,
) -> Result<(Tensor, bool)> {
    if let Some(tensor) = cache
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&key)
        .cloned()
    {
        return Ok((tensor, true));
    }

    let inv_freq: Vec<_> = (0..half)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / (half as f32)))
        .collect();
    let tensor = Tensor::new(inv_freq.as_slice(), device)?.to_dtype(DType::F32)?;
    cache
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .insert(key, tensor.clone());
    Ok((tensor, false))
}

fn get_timestep_embedding_cached(
    timesteps: &Tensor,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
    inv_freq_cache: &TimestepEmbeddingInvFreqCache,
) -> Result<Tensor> {
    get_timestep_embedding_with_inv_freq(
        timesteps,
        embedding_dim,
        flip_sin_to_cos,
        |device, half| {
            let key = TimestepEmbeddingInvFreqCacheKey::new(embedding_dim, device);
            cached_timestep_embedding_inv_freq(inv_freq_cache, key, device, half)
                .map(|(tensor, _)| tensor)
        },
    )
}

fn get_timestep_embedding_with_inv_freq<F>(
    timesteps: &Tensor,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
    inv_freq: F,
) -> Result<Tensor>
where
    F: FnOnce(&Device, usize) -> Result<Tensor>,
{
    let device = timesteps.device();
    let original_dtype = timesteps.dtype();
    let dtype = DType::F32;

    let n = timesteps.dim(0)?;
    let half = embedding_dim / 2;

    let t = timesteps.to_dtype(dtype)?;
    let t = t.unsqueeze(1)?;

    let inv_freq = inv_freq(device, half)?.to_dtype(dtype)?;
    let freqs = t.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let sin = freqs.sin()?;
    let cos = freqs.cos()?;

    let emb = if flip_sin_to_cos {
        Tensor::cat(&[cos, sin], D::Minus1)?
    } else {
        Tensor::cat(&[sin, cos], D::Minus1)?
    };

    if embedding_dim % 2 == 1 {
        let pad = Tensor::zeros((n, 1), dtype, device)?;
        Tensor::cat(&[emb, pad], D::Minus1)?.to_dtype(original_dtype)
    } else {
        emb.to_dtype(original_dtype)
    }
}

// ---------------------------------------------------------------------------
// Rotary position embedding helpers — F32 upcast for stability
// ---------------------------------------------------------------------------

fn apply_interleaved_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let cos = cos.to_dtype(DType::F32)?;
    let sin = sin.to_dtype(DType::F32)?;

    let (b, s, c) = x_f32.dims3()?;
    if c % 2 != 0 {
        candle_core::bail!("apply_rotary_emb expects last dim even, got {c}");
    }
    let half = c / 2;

    let x2 = x_f32.reshape((b, s, half, 2))?;
    let x_real = x2.i((.., .., .., 0))?;
    let x_imag = x2.i((.., .., .., 1))?;

    let x_rot = Tensor::stack(&[x_imag.neg()?, x_real.clone()], D::Minus1)?.reshape((b, s, c))?;

    let out = x_f32
        .broadcast_mul(&cos)?
        .broadcast_add(&x_rot.broadcast_mul(&sin)?)?;
    out.to_dtype(dtype)
}

fn apply_split_rotary_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let dtype = x.dtype();
    let (batch, seq, inner_dim) = x.dims3()?;
    if inner_dim != heads * head_dim {
        candle_core::bail!(
            "split rotary input dimension mismatch: expected {}, got {}",
            heads * head_dim,
            inner_dim
        );
    }
    if !head_dim.is_multiple_of(2) {
        candle_core::bail!("split rotary requires an even head_dim, got {head_dim}");
    }

    // Scope the F32 upcast so it drops once the head-major copy exists; shadowing
    // it kept both full-size buffers alive for the whole call. The F32 staging
    // itself is deliberate — `double_precision_rope` depends on it.
    let x = {
        let x_f32 = x.to_dtype(DType::F32)?;
        x_f32
            .reshape((batch, seq, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, heads, seq, 2, head_dim / 2))?
    };
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(3)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(3)?;

    let rotated = {
        let first = x.i((.., .., .., 0..1, ..))?;
        let second = x.i((.., .., .., 1..2, ..))?;
        let first_out = first
            .broadcast_mul(&cos)?
            .broadcast_sub(&second.broadcast_mul(&sin)?)?;
        let second_out = second
            .broadcast_mul(&cos)?
            .broadcast_add(&first.broadcast_mul(&sin)?)?;
        Tensor::cat(&[first_out, second_out], 3)?
    };
    drop(x);

    rotated
        .reshape((batch, heads, seq, head_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, seq, inner_dim))?
        .to_dtype(dtype)
}

pub fn apply_rotary_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rope_type: LtxRopeType,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    match rope_type {
        LtxRopeType::Interleaved => apply_interleaved_rotary_emb(x, cos, sin),
        LtxRopeType::Split => apply_split_rotary_emb(x, cos, sin, heads, head_dim),
    }
}

// ---------------------------------------------------------------------------
// LTX-2 rotary embedding — supports both interleaved and split RoPE.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Ltx2VideoRotaryPosEmbed {
    dim: usize,
    theta: f64,
    max_pos: Vec<usize>,
    use_middle_indices_grid: bool,
    num_attention_heads: usize,
    rope_type: LtxRopeType,
    double_precision_rope: bool,
    base_indices_cache: Arc<Mutex<HashMap<Ltx2RotaryBaseIndicesCacheKey, Tensor>>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Ltx2RotaryBaseIndicesCacheKey {
    position_dims: usize,
    device: String,
}

impl Ltx2VideoRotaryPosEmbed {
    pub fn new(
        dim: usize,
        theta: f64,
        max_pos: Vec<usize>,
        use_middle_indices_grid: bool,
        num_attention_heads: usize,
        rope_type: LtxRopeType,
        double_precision_rope: bool,
    ) -> Self {
        Self {
            dim,
            theta,
            max_pos,
            use_middle_indices_grid,
            num_attention_heads,
            rope_type,
            double_precision_rope,
            base_indices_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn fractional_positions(&self, indices_grid: &Tensor) -> Result<Tensor> {
        let (batch, pos_dims, seq, _bounds) = indices_grid.dims4()?;
        if pos_dims != self.max_pos.len() {
            candle_core::bail!(
                "rotary position dims mismatch: expected {}, got {}",
                self.max_pos.len(),
                pos_dims
            );
        }
        let grid = if self.use_middle_indices_grid {
            let starts = indices_grid.narrow(3, 0, 1)?;
            let ends = indices_grid.narrow(3, 1, 1)?;
            starts.broadcast_add(&ends)?.affine(0.5, 0.0)?
        } else {
            indices_grid.narrow(3, 0, 1)?
        }
        .squeeze(3)?
        .to_dtype(DType::F32)?;

        let mut normalized = Vec::with_capacity(pos_dims);
        for (dim, max_pos) in self.max_pos.iter().enumerate() {
            normalized.push(grid.i((.., dim, ..))?.affine(1.0 / *max_pos as f64, 0.0)?);
        }
        Tensor::stack(&normalized, 2)?.reshape((batch, seq, pos_dims))
    }

    fn base_indices_cached(&self, device: &Device, position_dims: usize) -> Result<(Tensor, bool)> {
        let steps = self.dim / (2 * position_dims);
        if steps == 0 {
            candle_core::bail!(
                "rotary dimension {} is too small for {} positional dims",
                self.dim,
                position_dims
            );
        }
        let key = Ltx2RotaryBaseIndicesCacheKey {
            position_dims,
            device: tensor_device_cache_key(device),
        };
        if let Some(tensor) = self
            .base_indices_cache
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok((tensor, true));
        }
        if steps == 1 {
            let tensor = Tensor::zeros((1,), DType::F32, device)?;
            self.base_indices_cache
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .insert(key, tensor.clone());
            Ok((tensor, false))
        } else {
            let denom = (steps - 1) as f64;
            let values: Vec<f32> = (0..steps)
                .map(|index| {
                    let ratio = index as f64 / denom;
                    let power = if self.double_precision_rope {
                        self.theta.powf(ratio)
                    } else {
                        (self.theta as f32).powf(ratio as f32) as f64
                    };
                    (power * std::f64::consts::PI / 2.0) as f32
                })
                .collect();
            let tensor = Tensor::from_vec(values, (steps,), device)?;
            self.base_indices_cache
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .insert(key, tensor.clone());
            Ok((tensor, false))
        }
    }

    fn base_indices(&self, device: &Device, position_dims: usize) -> Result<Tensor> {
        self.base_indices_cached(device, position_dims)
            .map(|(tensor, _)| tensor)
    }

    fn repeat_interleave_2(freqs: &Tensor) -> Result<Tensor> {
        let freq_unsq = freqs.unsqueeze(D::Minus1)?;
        let duplicated = Tensor::cat(&[freq_unsq.clone(), freq_unsq], D::Minus1)?;
        let shape = freqs.dims();
        let mut new_shape = shape[..shape.len() - 1].to_vec();
        new_shape.push(shape[shape.len() - 1] * 2);
        duplicated.reshape(new_shape)
    }

    #[allow(dead_code)]
    pub fn forward(&self, hidden_states: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        self.forward_for_dtype(hidden_states.device(), hidden_states.dtype(), positions)
    }

    pub fn forward_for_dtype(
        &self,
        device: &Device,
        dtype: DType,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let position_dims = positions.dim(1)?;
        let indices = self.base_indices(device, position_dims)?;
        let fractional = self.fractional_positions(positions)?;
        let scaled = fractional.unsqueeze(D::Minus1)?.affine(2.0, -1.0)?;
        let freqs = indices
            .reshape((1, 1, 1, indices.dim(0)?))?
            .broadcast_mul(&scaled)?
            .transpose(2, 3)?
            .contiguous()?
            .flatten_from(2)?;

        match self.rope_type {
            LtxRopeType::Interleaved => {
                let mut cos = Self::repeat_interleave_2(&freqs.cos()?)?;
                let mut sin = Self::repeat_interleave_2(&freqs.sin()?)?;
                let rem = self.dim % (2 * position_dims);
                if rem != 0 {
                    let (batch, seq, _) = cos.dims3()?;
                    let cos_pad = Tensor::ones((batch, seq, rem), DType::F32, device)?;
                    let sin_pad = Tensor::zeros((batch, seq, rem), DType::F32, device)?;
                    cos = Tensor::cat(&[cos_pad, cos], D::Minus1)?;
                    sin = Tensor::cat(&[sin_pad, sin], D::Minus1)?;
                }
                Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
            }
            LtxRopeType::Split => {
                let expected = self.dim / 2;
                let current = freqs.dim(D::Minus1)?;
                let pad_size = expected.saturating_sub(current);
                let mut cos = freqs.cos()?;
                let mut sin = freqs.sin()?;
                if pad_size != 0 {
                    let (batch, seq, _) = cos.dims3()?;
                    let cos_pad = Tensor::ones((batch, seq, pad_size), DType::F32, device)?;
                    let sin_pad = Tensor::zeros((batch, seq, pad_size), DType::F32, device)?;
                    cos = Tensor::cat(&[cos_pad, cos], D::Minus1)?;
                    sin = Tensor::cat(&[sin_pad, sin], D::Minus1)?;
                }
                let (batch, seq, _) = cos.dims3()?;
                let cos = cos
                    .reshape((
                        batch,
                        seq,
                        self.num_attention_heads,
                        expected / self.num_attention_heads,
                    ))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let sin = sin
                    .reshape((
                        batch,
                        seq,
                        self.num_attention_heads,
                        expected / self.num_attention_heads,
                    ))?
                    .transpose(1, 2)?
                    .contiguous()?;
                Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LtxAttention — multi-head attention with RoPE + QK RMSNorm
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct LtxAttention {
    heads: usize,
    head_dim: usize,
    inner_dim: usize,
    inner_kv_dim: usize,
    cross_attention_dim: usize,
    rope_type: LtxRopeType,
    norm_eps: f64,

    norm_q: RmsNorm,
    norm_k: RmsNorm,

    to_q: LtxLinear,
    to_k: LtxLinear,
    to_v: LtxLinear,

    to_out: LtxLinear,
    to_gate_logits: Option<LtxLinear>,
    dropout: nn::Dropout,
}

impl LtxAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query_dim: usize,
        heads: usize,
        kv_heads: usize,
        dim_head: usize,
        dropout: f64,
        bias: bool,
        cross_attention_dim: Option<usize>,
        out_bias: bool,
        qk_norm: &str,
        rope_type: LtxRopeType,
        apply_gated_attention: bool,
        vb: VarBuilder,
        lora_registry: Option<&Ltx2LoraRegistry>,
        lora_key_prefix: &str,
        nvfp4_cache: Option<&Nvfp4LinearCache>,
    ) -> Result<Self> {
        if qk_norm != "rms_norm_across_heads" && qk_norm != "rms_norm" {
            candle_core::bail!(
                "Only 'rms_norm' and 'rms_norm_across_heads' are supported as qk_norm."
            );
        }

        let inner_dim = dim_head * heads;
        let inner_kv_dim = dim_head * kv_heads;
        let cross_attention_dim = cross_attention_dim.unwrap_or(query_dim);

        let norm_eps = 1e-6;
        let norm_q = RmsNorm::new(inner_dim, norm_eps, true, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(inner_kv_dim, norm_eps, true, vb.pp("norm_k"))?;

        let to_q_key = format!("{lora_key_prefix}.to_q");
        let to_q = LtxLinear::load_with_nvfp4_cache(
            query_dim,
            inner_dim,
            bias,
            vb.pp("to_q"),
            lora_adapters_for(lora_registry, &to_q_key),
            nvfp4_cache,
            Some(&to_q_key),
        )?;
        let to_k_key = format!("{lora_key_prefix}.to_k");
        let to_k = LtxLinear::load_with_nvfp4_cache(
            cross_attention_dim,
            inner_kv_dim,
            bias,
            vb.pp("to_k"),
            lora_adapters_for(lora_registry, &to_k_key),
            nvfp4_cache,
            Some(&to_k_key),
        )?;
        let to_v_key = format!("{lora_key_prefix}.to_v");
        let to_v = LtxLinear::load_with_nvfp4_cache(
            cross_attention_dim,
            inner_kv_dim,
            bias,
            vb.pp("to_v"),
            lora_adapters_for(lora_registry, &to_v_key),
            nvfp4_cache,
            Some(&to_v_key),
        )?;

        let to_out_key = format!("{lora_key_prefix}.to_out.0");
        let to_out = LtxLinear::load_with_nvfp4_cache(
            inner_dim,
            query_dim,
            out_bias,
            vb.pp("to_out").pp("0"),
            lora_adapters_for(lora_registry, &to_out_key),
            nvfp4_cache,
            Some(&to_out_key),
        )?;
        let to_gate_logits = apply_gated_attention
            .then(|| {
                let to_gate_logits_key = format!("{lora_key_prefix}.to_gate_logits");
                LtxLinear::load_with_nvfp4_cache(
                    query_dim,
                    heads,
                    true,
                    vb.pp("to_gate_logits"),
                    lora_adapters_for(lora_registry, &to_gate_logits_key),
                    nvfp4_cache,
                    Some(&to_gate_logits_key),
                )
            })
            .transpose()?;
        let dropout = nn::Dropout::new(dropout as f32);

        Ok(Self {
            heads,
            head_dim: dim_head,
            inner_dim,
            inner_kv_dim,
            cross_attention_dim,
            rope_type,
            norm_eps,
            norm_q,
            norm_k,
            to_q,
            to_k,
            to_v,
            to_out,
            to_gate_logits,
            dropout,
        })
    }

    fn prepare_attention_mask(
        &self,
        attention_mask: &Tensor,
        q_len: usize,
        k_len: usize,
    ) -> Result<Tensor> {
        match attention_mask.rank() {
            2 => {
                let (b, kk) = attention_mask.dims2()?;
                if kk != k_len {
                    candle_core::bail!(
                        "Expected attention_mask [B,k_len]=[{},{}], got [{},{}]",
                        b,
                        k_len,
                        b,
                        kk
                    );
                }
                let mask_f = attention_mask.to_dtype(DType::F32)?;
                let mask = ((mask_f.affine(-1.0, 1.0))?.affine(-10000.0, 0.0))?;
                let m = mask.unsqueeze(1)?.unsqueeze(1)?;
                m.broadcast_as((b, self.heads, q_len, k_len))?.contiguous()
            }
            3 => {
                let (b, one, kk) = attention_mask.dims3()?;
                if one != 1 || kk != k_len {
                    candle_core::bail!(
                        "Expected attention_mask [B,1,k_len]=[{},1,{}], got [{},{},{}]",
                        b,
                        k_len,
                        b,
                        one,
                        kk
                    );
                }
                let m = attention_mask.unsqueeze(2)?;
                m.broadcast_as((b, self.heads, q_len, k_len))?.contiguous()
            }
            4 => Ok(attention_mask.clone()),
            other => candle_core::bail!("Unsupported attention_mask rank {other}"),
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
        key_rotary_emb: Option<(&Tensor, &Tensor)>,
        perturbation_mask: Option<&Tensor>,
        all_perturbed: bool,
    ) -> Result<Tensor> {
        let (b, q_len, _) = hidden_states.dims3()?;
        let is_self_attention = encoder_hidden_states.is_none();
        let enc = encoder_hidden_states.unwrap_or(hidden_states);
        let (_, k_len, _) = enc.dims3()?;

        let attn_mask = if let Some(mask) = attention_mask {
            Some(self.prepare_attention_mask(mask, q_len, k_len)?)
        } else {
            None
        };

        let v = self.to_v.forward(enc)?;
        let v = v.reshape((b, k_len, self.heads, self.head_dim))?;

        let dtype = hidden_states.dtype();
        // The head-major passthrough copy is only read by the perturbation paths;
        // materializing it eagerly cost a full transposing copy of `v` on every
        // call, including the entire unperturbed distilled path.
        let out = if all_perturbed {
            v.transpose(1, 2)?.contiguous()?
        } else {
            let mut q = self.to_q.forward(hidden_states)?;
            let mut k = self.to_k.forward(enc)?;

            q = self.norm_q.forward(&q)?;
            k = self.norm_k.forward(&k)?;

            if let Some((cos, sin)) = image_rotary_emb {
                if is_self_attention {
                    q = apply_rotary_emb(&q, cos, sin, self.rope_type, self.heads, self.head_dim)?;
                    k = apply_rotary_emb(&k, cos, sin, self.rope_type, self.heads, self.head_dim)?;
                } else if let Some((k_cos, k_sin)) = key_rotary_emb {
                    q = apply_rotary_emb(&q, cos, sin, self.rope_type, self.heads, self.head_dim)?;
                    k = apply_rotary_emb(
                        &k,
                        k_cos,
                        k_sin,
                        self.rope_type,
                        self.heads,
                        self.head_dim,
                    )?;
                }
            }

            let q = q.reshape((b, q_len, self.heads, self.head_dim))?;
            let k = k.reshape((b, k_len, self.heads, self.head_dim))?;

            let scale = 1f32 / (self.head_dim as f32).sqrt();
            let device = hidden_states.device();
            let mut out = if device.is_metal() {
                // Candle's Metal SDPA reads Q/K/V strides directly. Force
                // contiguous head-major inputs until upstream's non-contiguous
                // correctness fix is part of our pinned release.
                metal_fused_attention(
                    &q.transpose(1, 2)?,
                    &k.transpose(1, 2)?,
                    &v.transpose(1, 2)?,
                    attn_mask.as_ref(),
                    scale,
                )?
            } else {
                let out_f32 = if should_chunk_attention(b, self.heads, q_len, k_len, dtype, device)
                {
                    let attn_mask_f32 = attn_mask
                        .as_ref()
                        .map(|mask| mask.to_dtype(DType::F32))
                        .transpose()?;
                    let q_t = q.transpose(1, 2)?;
                    let k_t = k.transpose(1, 2)?;
                    let v_t = v.transpose(1, 2)?;
                    let key_chunk = attention_key_chunk_size(k_len);
                    chunked_attention(
                        &q_t,
                        &k_t,
                        &v_t,
                        attn_mask_f32.as_ref(),
                        scale,
                        attention_query_chunk_size(b, self.heads, q_len, key_chunk),
                        key_chunk,
                    )?
                } else {
                    let attn_mask_f32 = attn_mask
                        .as_ref()
                        .map(|mask| mask.to_dtype(DType::F32))
                        .transpose()?;
                    let q_f32 = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
                    let k_f32 = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
                    let v_f32 = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
                    full_attention(&q_f32, &k_f32, &v_f32, attn_mask_f32.as_ref(), scale)?
                };
                out_f32.to_dtype(dtype)?
            };
            if let Some(mask) = perturbation_mask {
                let mask = if mask.rank() == out.rank() {
                    mask.clone()
                } else {
                    let mut shape = vec![mask.dim(0)?];
                    shape.extend(std::iter::repeat_n(1usize, out.rank().saturating_sub(1)));
                    mask.reshape(shape)?
                };
                let mask = if mask.dtype() == out.dtype() {
                    mask
                } else {
                    mask.to_dtype(out.dtype())?
                };
                let one_minus_mask = Tensor::ones_like(&mask)?.broadcast_sub(&mask)?;
                let value_passthrough = v.transpose(1, 2)?.contiguous()?;
                out = out
                    .broadcast_mul(&mask)?
                    .broadcast_add(&value_passthrough.broadcast_mul(&one_minus_mask)?)?;
            }
            out
        };

        let mut out = out.transpose(1, 2)?.contiguous()?;
        if let Some(to_gate_logits) = &self.to_gate_logits {
            let gates = to_gate_logits.forward(hidden_states)?;
            let gates = nn::ops::sigmoid(&gates)?.affine(2.0, 0.0)?;
            let gates = if gates.dtype() == out.dtype() {
                gates
            } else {
                gates.to_dtype(out.dtype())?
            };
            out = out.broadcast_mul(&gates.unsqueeze(D::Minus1)?)?;
        }
        let out = out.reshape((b, q_len, self.inner_dim))?;

        let out = self.to_out.forward(&out)?;
        self.dropout.forward(&out, false)
    }
}

const CUDA_ATTENTION_CHUNK_THRESHOLD_BYTES: u64 = 1_048_576 * 4;
const CPU_ATTENTION_CHUNK_THRESHOLD_BYTES: u64 = CUDA_ATTENTION_CHUNK_THRESHOLD_BYTES * 4;

fn attention_work_bytes(
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    dtype: DType,
) -> u64 {
    (batch.max(1) as u64)
        .saturating_mul(heads.max(1) as u64)
        .saturating_mul(q_len as u64)
        .saturating_mul(k_len as u64)
        .saturating_mul(crate::device::dtype_bytes(dtype) as u64)
}

fn attention_chunk_threshold_bytes(device: &Device) -> u64 {
    if device.is_cuda() {
        CUDA_ATTENTION_CHUNK_THRESHOLD_BYTES
    } else {
        CPU_ATTENTION_CHUNK_THRESHOLD_BYTES
    }
}

fn should_chunk_attention(
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    dtype: DType,
    device: &Device,
) -> bool {
    attention_work_bytes(batch, heads, q_len, k_len, dtype)
        > attention_chunk_threshold_bytes(device)
}

/// Target F32 bytes for one `query_chunk × key_chunk` attention tile.
const ATTENTION_TILE_TARGET_BYTES: u64 = 128 * 1024 * 1024;
const MIN_ATTENTION_QUERY_CHUNK: usize = 32;
const MAX_ATTENTION_QUERY_CHUNK: usize = 2_048;

/// Query-chunk size derived from a tile byte budget rather than from `q_len`.
///
/// The previous 32/64/128 step function shrank the tile precisely where the
/// sequence is longest: at LTX-2 stage-2 shapes (13,312 queries, 32 heads) it
/// produced 416 × 13 = 5,408 tiny inner iterations per self-attention, leaving
/// almost all of the GPU idle. Sizing from bytes keeps the same working set
/// while giving each matmul enough work to saturate the device.
fn attention_query_chunk_size(
    batch: usize,
    heads: usize,
    q_len: usize,
    key_chunk_size: usize,
) -> usize {
    let bytes_per_query = (batch.max(1) as u64)
        .saturating_mul(heads.max(1) as u64)
        .saturating_mul(key_chunk_size.max(1) as u64)
        .saturating_mul(std::mem::size_of::<f32>() as u64);
    let chunk = (ATTENTION_TILE_TARGET_BYTES / bytes_per_query.max(1)) as usize;
    chunk
        .clamp(MIN_ATTENTION_QUERY_CHUNK, MAX_ATTENTION_QUERY_CHUNK)
        .min(q_len.max(1))
}

fn attention_key_chunk_size(k_len: usize) -> usize {
    if k_len >= 8_192 {
        1_024
    } else if k_len >= 4_096 {
        2_048
    } else {
        k_len
    }
}

fn full_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    let att = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
    let att = (att * (scale as f64))?;

    let att = match attn_mask {
        Some(mask) => att.broadcast_add(mask)?,
        None => att,
    };

    let (b_sz, h_sz, q_l, k_l) = att.dims4()?;
    let att = att.reshape((b_sz * h_sz * q_l, k_l))?;
    let att = nn::ops::softmax(&att, D::Minus1)?;
    let att = att.reshape((b_sz, h_sz, q_l, k_l))?;

    att.matmul(v)
}

/// Non-materializing attention for Apple unified memory.
///
/// LTX-2 stage-two self-attention reaches roughly 14k latent tokens. Building
/// the complete score matrix is both slower and large enough to pressure the
/// Metal driver allocator. Candle's fused SDPA mirrors the official LTX-2 MPS
/// path while preserving the existing additive-mask semantics.
fn metal_fused_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let mask = attn_mask
        .map(|mask| mask.to_dtype(q.dtype())?.contiguous())
        .transpose()?;
    nn::ops::sdpa(&q, &k, &v, mask.as_ref(), false, scale, 1.0)
}

fn chunked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    scale: f32,
    query_chunk_size: usize,
    key_chunk_size: usize,
) -> Result<Tensor> {
    // q/k/v stay in their incoming dtype here; each tile is narrowed first and
    // upcast to F32 inside the loop. Upcasting whole tensors up front cost four
    // full-size F32 buffers (q, k, v, and the transposed k) that exist only to
    // be read one tile at a time — 872 MB at the 13,312-token stage-2 shape.
    // BF16 -> F32 is an exact widening and the matmuls are still F32, so tiling
    // the cast is bit-identical, not a numerics trade.
    let q = q.clone();
    let k = k.clone();
    let v = v.clone();
    let attn_mask = attn_mask
        .map(|mask| {
            if mask.dtype() == DType::F32 {
                Ok(mask.clone())
            } else {
                mask.to_dtype(DType::F32)
            }
        })
        .transpose()?;
    let q_len = q.dim(2)?;
    let k_len = k.dim(2)?;
    let value_dim = v.dim(3)?;
    let mut out: Option<Tensor> = None;
    let mut q_offset = 0;
    while q_offset < q_len {
        let q_chunk_len = query_chunk_size.min(q_len - q_offset);
        let q_chunk = q
            .narrow(2, q_offset, q_chunk_len)?
            .contiguous()?
            .to_dtype(DType::F32)?;
        let (b_sz, h_sz, _, _) = q_chunk.dims4()?;
        let mut running_max =
            Tensor::full(f32::NEG_INFINITY, (b_sz, h_sz, q_chunk_len, 1), q.device())?;
        let mut running_denom =
            Tensor::zeros((b_sz, h_sz, q_chunk_len, 1), DType::F32, q.device())?;
        let mut running_out =
            Tensor::zeros((b_sz, h_sz, q_chunk_len, value_dim), DType::F32, q.device())?;

        let mut k_offset = 0;
        while k_offset < k_len {
            let k_chunk_len = key_chunk_size.min(k_len - k_offset);
            // Transpose the key tile rather than the whole key tensor: the
            // full `[b, h, d, k_len]` contiguous copy was the single largest
            // staging buffer here.
            let k_chunk = k
                .narrow(2, k_offset, k_chunk_len)?
                .transpose(D::Minus1, D::Minus2)?
                .contiguous()?
                .to_dtype(DType::F32)?;
            let v_chunk = v
                .narrow(2, k_offset, k_chunk_len)?
                .contiguous()?
                .to_dtype(DType::F32)?;

            let mut att = q_chunk.matmul(&k_chunk)?;
            att = (att * (scale as f64))?;
            if let Some(mask) = attn_mask.as_ref() {
                let mask = mask
                    .narrow(2, q_offset, q_chunk_len)?
                    .narrow(3, k_offset, k_chunk_len)?
                    .contiguous()?;
                att = att.broadcast_add(&mask)?;
            }

            let chunk_max = att.max_keepdim(D::Minus1)?;
            let next_max = running_max.maximum(&chunk_max)?;
            let prev_scale = running_max.broadcast_sub(&next_max)?.exp()?;
            // Reassign rather than shadow: shadowing keeps every intermediate
            // score tile alive for the rest of the iteration, tripling the peak.
            att = att.broadcast_sub(&next_max)?;
            att = att.exp()?;
            let chunk_denom = att.sum_keepdim(D::Minus1)?;
            let chunk_out = att.matmul(&v_chunk)?;
            drop(att);

            running_denom = running_denom
                .broadcast_mul(&prev_scale)?
                .broadcast_add(&chunk_denom)?;
            running_out = running_out
                .broadcast_mul(&prev_scale)?
                .broadcast_add(&chunk_out)?;
            running_max = next_max;
            k_offset += k_chunk_len;
        }

        let chunk = running_out.broadcast_div(&running_denom)?;
        if out.is_none() {
            out = Some(Tensor::zeros(
                (b_sz, h_sz, q_len, value_dim),
                DType::F32,
                q.device(),
            )?);
        }
        let dst = out
            .as_ref()
            .expect("chunked attention destination allocated above");
        dst.slice_set(&chunk.contiguous()?, 2, q_offset)?;
        q_offset += q_chunk_len;
    }
    out.ok_or_else(|| candle_core::Error::msg("chunked attention produced no query chunks"))
}

// ---------------------------------------------------------------------------
// LtxVideoTransformerBlock
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LtxVideoTransformerBlock {
    norm1: RmsNorm,
    attn1: LtxAttention,
    norm2: RmsNorm,
    attn2: LtxAttention,
    norm3: RmsNorm,
    ff: FeedForward,
    scale_shift_table: Tensor,
}

#[allow(dead_code)]
impl LtxVideoTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_attention_heads: usize,
        attention_head_dim: usize,
        cross_attention_dim: usize,
        qk_norm: &str,
        rope_type: LtxRopeType,
        attention_bias: bool,
        attention_out_bias: bool,
        eps: f64,
        elementwise_affine: bool,
        apply_gated_attention: bool,
        vb: VarBuilder,
        lora_registry: Option<&Ltx2LoraRegistry>,
        block_key: &str,
        nvfp4_cache: Option<&Nvfp4LinearCache>,
    ) -> Result<Self> {
        let norm1 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm1"))?;
        let attn1 = LtxAttention::new(
            dim,
            num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            0.0,
            attention_bias,
            None,
            attention_out_bias,
            qk_norm,
            rope_type,
            apply_gated_attention,
            vb.pp("attn1"),
            lora_registry,
            &format!("{block_key}.attn1"),
            nvfp4_cache,
        )?;
        let norm2 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm2"))?;
        let attn2 = LtxAttention::new(
            dim,
            num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            0.0,
            attention_bias,
            Some(cross_attention_dim),
            attention_out_bias,
            qk_norm,
            rope_type,
            apply_gated_attention,
            vb.pp("attn2"),
            lora_registry,
            &format!("{block_key}.attn2"),
            nvfp4_cache,
        )?;
        let norm3 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm3"))?;

        let ff = FeedForward::new(
            dim,
            vb.pp("ff"),
            lora_registry,
            &format!("{block_key}.ff"),
            nvfp4_cache,
        )?;
        let scale_shift_table = vb.get((6, dim), "scale_shift_table")?;

        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
            scale_shift_table,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b = hidden_states.dim(0)?;
        let norm_hidden = self.norm1.forward(hidden_states)?;

        let (b_temb, temb_last) = temb.dims2()?;
        if b_temb != b {
            candle_core::bail!(
                "temb batch size {} mismatch hidden_states batch size {}",
                b_temb,
                b
            );
        }

        if temb_last % 6 != 0 {
            candle_core::bail!("temb last dim must be divisible by 6, got {temb_last}");
        }
        let dim = temb_last / 6;
        let t = 1;
        let temb_reshaped = temb.reshape((b, t, 6, dim))?;

        let table = self
            .scale_shift_table
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, t, 6, dim))?;
        let ada = table.broadcast_add(&temb_reshaped)?;

        let shift_msa = ada.i((.., .., 0, ..))?;
        let scale_msa = ada.i((.., .., 1, ..))?;
        let gate_msa = ada.i((.., .., 2, ..))?;
        let shift_mlp = ada.i((.., .., 3, ..))?;
        let scale_mlp = ada.i((.., .., 4, ..))?;
        let gate_mlp = ada.i((.., .., 5, ..))?;

        let norm_hidden = {
            let one = Tensor::ones_like(&scale_msa)?;
            let s = one.broadcast_add(&scale_msa)?;
            let s = if s.dim(1)? == 1 {
                s.broadcast_as((b, hidden_states.dim(1)?, s.dim(2)?))?
            } else {
                s
            };
            let sh = if shift_msa.dim(1)? == 1 {
                shift_msa.broadcast_as((b, hidden_states.dim(1)?, shift_msa.dim(2)?))?
            } else {
                shift_msa
            };
            norm_hidden.broadcast_mul(&s)?.broadcast_add(&sh)?
        };

        let attn1 = self.attn1.forward(
            &norm_hidden,
            None,
            None,
            image_rotary_emb,
            None,
            None,
            false,
        )?;
        let gate_msa = if gate_msa.dim(1)? == 1 {
            gate_msa.broadcast_as((b, hidden_states.dim(1)?, gate_msa.dim(2)?))?
        } else {
            gate_msa
        };
        let mut hs = hidden_states.broadcast_add(&attn1.broadcast_mul(&gate_msa)?)?;

        let norm2 = self.norm2.forward(&hs)?;

        let attn2 = self.attn2.forward(
            &norm2,
            Some(encoder_hidden_states),
            encoder_attention_mask,
            None,
            None,
            None,
            false,
        )?;
        hs = hs.broadcast_add(&attn2)?;

        let norm3 = self.norm3.forward(&hs)?;
        let norm3 = {
            let one = Tensor::ones_like(&scale_mlp)?;
            let s = one.broadcast_add(&scale_mlp)?;
            let s = if s.dim(1)? == 1 {
                s.broadcast_as((b, hs.dim(1)?, s.dim(2)?))?
            } else {
                s
            };
            let sh = if shift_mlp.dim(1)? == 1 {
                shift_mlp.broadcast_as((b, hs.dim(1)?, shift_mlp.dim(2)?))?
            } else {
                shift_mlp
            };
            norm3.broadcast_mul(&s)?.broadcast_add(&sh)?
        };
        let ff = self.ff.forward(&norm3)?;
        let gate_mlp = if gate_mlp.dim(1)? == 1 {
            gate_mlp.broadcast_as((b, hs.dim(1)?, gate_mlp.dim(2)?))?
        } else {
            gate_mlp
        };
        hs = hs.broadcast_add(&ff.broadcast_mul(&gate_mlp)?)?;

        Ok(hs)
    }
}

// ---------------------------------------------------------------------------
// LTX-2 video transformer — top-level model
// ---------------------------------------------------------------------------

#[allow(dead_code)]
enum TransformerBlockSource {
    Eager(Vec<LtxVideoTransformerBlock>),
    Streaming(VarBuilder<'static>),
}

#[allow(dead_code)]
pub struct Ltx2VideoTransformer3DModel {
    proj_in: nn::Linear,
    scale_shift_table: Tensor,
    time_embed: AdaLayerNormSingle,
    caption_projection: PixArtAlphaTextProjection,
    rope: Ltx2VideoRotaryPosEmbed,
    transformer_blocks: TransformerBlockSource,
    norm_out: LayerNormNoParams,
    proj_out: LtxLinear,
    nvfp4_cache: Nvfp4LinearCache,
    config: Ltx2VideoTransformer3DModelConfig,
    skip_block_list: Vec<usize>,
    /// Timestep scaling factor (1000.0 for LTX Video, matching Python's
    /// `timestep_scale_multiplier`). Applied to the input timestep before
    /// computing the sinusoidal embedding.
    timestep_scale_multiplier: f64,
}

#[allow(dead_code)]
impl Ltx2VideoTransformer3DModel {
    pub fn new(config: &Ltx2VideoTransformer3DModelConfig, vb: VarBuilder) -> Result<Self> {
        let out_channels = if config.out_channels == 0 {
            config.in_channels
        } else {
            config.out_channels
        };
        let inner_dim = config.num_attention_heads * config.attention_head_dim;
        let nvfp4_cache = Nvfp4LinearCache::default();

        let proj_in = nn::linear(config.in_channels, inner_dim, vb.pp("proj_in"))?;
        let scale_shift_table = vb.get((2, inner_dim), "scale_shift_table")?;

        let time_embed = AdaLayerNormSingle::new(inner_dim, vb.pp("time_embed"))?;
        let caption_projection = PixArtAlphaTextProjection::new(
            config.caption_channels,
            inner_dim,
            vb.pp("caption_projection"),
        )?;

        let rope = Ltx2VideoRotaryPosEmbed::new(
            inner_dim,
            config.positional_embedding_theta,
            config.positional_embedding_max_pos.clone(),
            config.use_middle_indices_grid,
            config.num_attention_heads,
            config.rope_type,
            config.double_precision_rope,
        );

        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            transformer_blocks.push(LtxVideoTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                config.cross_attention_dim,
                &config.qk_norm,
                config.rope_type,
                config.attention_bias,
                config.attention_out_bias,
                config.norm_eps,
                config.norm_elementwise_affine,
                config.apply_gated_attention,
                vb.pp("transformer_blocks").pp(layer_idx.to_string()),
                None,
                &format!("diffusion_model.transformer_blocks.{layer_idx}"),
                Some(&nvfp4_cache),
            )?);
        }

        let norm_out = LayerNormNoParams::new(1e-6);
        let proj_out = LtxLinear::load_with_nvfp4_cache(
            inner_dim,
            out_channels,
            true,
            vb.pp("proj_out"),
            vec![],
            Some(&nvfp4_cache),
            Some("diffusion_model.proj_out"),
        )?;

        Ok(Self {
            proj_in,
            scale_shift_table,
            time_embed,
            caption_projection,
            rope,
            transformer_blocks: TransformerBlockSource::Eager(transformer_blocks),
            norm_out,
            proj_out,
            nvfp4_cache,
            config: config.clone(),
            skip_block_list: Vec::new(),
            timestep_scale_multiplier: 1000.0,
        })
    }

    pub fn new_streaming(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder<'static>,
    ) -> Result<Self> {
        let out_channels = if config.out_channels == 0 {
            config.in_channels
        } else {
            config.out_channels
        };
        let inner_dim = config.num_attention_heads * config.attention_head_dim;
        let nvfp4_cache = Nvfp4LinearCache::default();

        let proj_in = nn::linear(config.in_channels, inner_dim, vb.pp("proj_in"))?;
        let scale_shift_table = vb.get((2, inner_dim), "scale_shift_table")?;

        let time_embed = AdaLayerNormSingle::new(inner_dim, vb.pp("time_embed"))?;
        let caption_projection = PixArtAlphaTextProjection::new(
            config.caption_channels,
            inner_dim,
            vb.pp("caption_projection"),
        )?;

        let rope = Ltx2VideoRotaryPosEmbed::new(
            inner_dim,
            config.positional_embedding_theta,
            config.positional_embedding_max_pos.clone(),
            config.use_middle_indices_grid,
            config.num_attention_heads,
            config.rope_type,
            config.double_precision_rope,
        );

        let norm_out = LayerNormNoParams::new(1e-6);
        let proj_out = LtxLinear::load_with_nvfp4_cache(
            inner_dim,
            out_channels,
            true,
            vb.pp("proj_out"),
            vec![],
            Some(&nvfp4_cache),
            Some("diffusion_model.proj_out"),
        )?;

        Ok(Self {
            proj_in,
            scale_shift_table,
            time_embed,
            caption_projection,
            rope,
            transformer_blocks: TransformerBlockSource::Streaming(vb.pp("transformer_blocks")),
            norm_out,
            proj_out,
            nvfp4_cache,
            config: config.clone(),
            skip_block_list: Vec::new(),
            timestep_scale_multiplier: 1000.0,
        })
    }

    pub fn config(&self) -> &Ltx2VideoTransformer3DModelConfig {
        &self.config
    }

    pub fn set_skip_block_list(&mut self, list: Vec<usize>) {
        self.skip_block_list = list;
    }

    fn streaming_block(
        &self,
        blocks_vb: VarBuilder<'static>,
        index: usize,
    ) -> Result<LtxVideoTransformerBlock> {
        let inner_dim = self.config.num_attention_heads * self.config.attention_head_dim;
        LtxVideoTransformerBlock::new(
            inner_dim,
            self.config.num_attention_heads,
            self.config.attention_head_dim,
            self.config.cross_attention_dim,
            &self.config.qk_norm,
            self.config.rope_type,
            self.config.attention_bias,
            self.config.attention_out_bias,
            self.config.norm_eps,
            self.config.norm_elementwise_affine,
            self.config.apply_gated_attention,
            blocks_vb.pp(index.to_string()),
            None,
            &format!("diffusion_model.transformer_blocks.{index}"),
            Some(&self.nvfp4_cache),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_block(
        &self,
        index: usize,
        block: &LtxVideoTransformerBlock,
        hidden_states: Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.skip_block_list.contains(&index) {
            return Ok(hidden_states);
        }

        let original_hidden_states = if skip_layer_mask.is_some() {
            Some(hidden_states.clone())
        } else {
            None
        };

        let mut hidden_states = block.forward(
            &hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            encoder_attention_mask,
        )?;

        if let (Some(mask), Some(orig)) = (skip_layer_mask, original_hidden_states) {
            let m = mask.narrow(0, index, 1)?.flatten_all()?;
            let batch = hidden_states.dim(0)?;
            let m = m.reshape((batch, 1, 1))?.to_dtype(hidden_states.dtype())?;
            let one_minus_m = m.affine(-1.0, 1.0)?;
            hidden_states = hidden_states
                .broadcast_mul(&one_minus_m)?
                .broadcast_add(&orig.broadcast_mul(&m)?)?;
        }

        Ok(hidden_states)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        _num_frames: usize,
        _height: usize,
        _width: usize,
        _rope_interpolation_scale: Option<(f64, f64, f64)>,
        video_coords: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let weight_dtype = self.proj_in.weight().dtype();
        let compute_dtype = match weight_dtype {
            DType::F8E4M3 => DType::BF16,
            _ => weight_dtype,
        };
        let hidden_states = hidden_states.to_dtype(compute_dtype)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(compute_dtype)?;

        let hidden_states = self.proj_in.forward(&hidden_states)?;

        // Scale timestep by multiplier (1000.0) before computing sinusoidal embedding.
        // The model receives raw sigma (0-1) and needs to map to embedding range (0-1000).
        let timestep = timestep
            .flatten_all()?
            .to_dtype(compute_dtype)?
            .affine(self.timestep_scale_multiplier, 0.0)?;

        let (temb, embedded_timestep) = self.time_embed.forward(&timestep)?;

        let encoder_hidden_states = self.caption_projection.forward(&encoder_hidden_states)?;

        let encoder_attention_mask = if let Some(mask) = encoder_attention_mask {
            if mask.rank() == 2 {
                let mask_f = mask.to_dtype(hidden_states.dtype())?;
                let bias = (mask_f.affine(-1.0, 1.0)? * (-10000.0))?;
                Some(bias.unsqueeze(1)?)
            } else {
                Some(mask.clone())
            }
        } else {
            None
        };
        let encoder_attention_mask = encoder_attention_mask.as_ref();

        let video_coords = video_coords.ok_or_else(|| {
            candle_core::Error::msg("LTX-2 video transformer requires explicit positional bounds")
        })?;
        let (cos, sin) = self.rope.forward(&hidden_states, video_coords)?;

        let mut hidden_states = hidden_states;
        let image_rotary_emb = Some((&cos, &sin));

        match &self.transformer_blocks {
            TransformerBlockSource::Eager(blocks) => {
                for (index, block) in blocks.iter().enumerate() {
                    if ltx2_block_debug_enabled() {
                        eprintln!("[ltx2-block-debug] enter block={index}");
                    }
                    hidden_states = self.apply_block(
                        index,
                        block,
                        hidden_states,
                        &encoder_hidden_states,
                        &temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        skip_layer_mask,
                    )?;
                }
            }
            TransformerBlockSource::Streaming(blocks_vb) => {
                for index in 0..self.config.num_layers {
                    if ltx2_block_debug_enabled() {
                        eprintln!("[ltx2-block-debug] enter block={index}");
                    }
                    let block = self.streaming_block(blocks_vb.clone(), index)?;
                    hidden_states = self.apply_block(
                        index,
                        &block,
                        hidden_states,
                        &encoder_hidden_states,
                        &temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        skip_layer_mask,
                    )?;
                    if hidden_states.device().is_cuda()
                        && should_synchronize_streaming_layer(
                            index,
                            self.config.num_layers,
                            self.config.streaming_prefetch_count,
                        )
                    {
                        hidden_states.device().synchronize()?;
                    }
                    drop(block);
                }
            }
        }

        // Final modulation
        let b = hidden_states.dim(0)?;
        let inner_dim = hidden_states.dim(2)?;

        let table = self.scale_shift_table.to_dtype(embedded_timestep.dtype())?;
        let table = table.unsqueeze(0)?.unsqueeze(0)?;
        let emb = embedded_timestep.unsqueeze(1)?.unsqueeze(2)?;
        let scale_shift = table.broadcast_add(&emb)?;

        let shift = scale_shift.i((.., .., 0, ..))?;
        let scale = scale_shift.i((.., .., 1, ..))?;

        let mut hidden_states = self.norm_out.forward(&hidden_states)?;

        let one = Tensor::ones_like(&scale)?;
        let ss = one.broadcast_add(&scale)?;

        let s_dim = hidden_states.dim(1)?;
        let ss = ss.broadcast_as((b, s_dim, inner_dim))?;
        let sh = shift.broadcast_as((b, s_dim, inner_dim))?;

        hidden_states = hidden_states.broadcast_mul(&ss)?.broadcast_add(&sh)?;

        self.proj_out.forward(&hidden_states)
    }
}

pub(crate) fn rms_norm_tensor(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs_f32 = xs.to_dtype(DType::F32)?;
    let dim = xs_f32.dim(D::Minus1)? as f64;
    let ms = xs_f32
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .affine(1.0 / dim, 0.0)?;
    let denom = ms.affine(1.0, eps)?.sqrt()?;
    xs_f32.broadcast_div(&denom)?.to_dtype(dtype)
}

/// One AdaLN modulation component: `scale_shift_table[index] + timestep[.., index]`.
///
/// `timestep` is `[batch, tokens, count * dim]` and the table broadcasts over the
/// token axis, so summing the whole `[batch, tokens, count, dim]` block only to
/// narrow a few components out of it costs a full extra activation (~650 MB at
/// LTX-2 stage-2 shapes, twice per block). Slicing the component first keeps each
/// result at `[batch, tokens, dim]`.
pub(crate) fn ada_component(
    scale_shift_table: &Tensor,
    timestep: &Tensor,
    index: usize,
) -> Result<Tensor> {
    let dim = scale_shift_table.dim(1)?;
    let width = timestep.dim(D::Minus1)?;
    if !width.is_multiple_of(dim) || (index + 1) * dim > width {
        candle_core::bail!(
            "AdaLN component {index} is out of range for timestep width {width} and dim {dim}"
        );
    }
    let entry = scale_shift_table
        .narrow(0, index, 1)?
        .reshape((1, 1, dim))?
        .to_dtype(timestep.dtype())?;
    timestep
        .narrow(D::Minus1, index * dim, dim)?
        .broadcast_add(&entry)
}

fn broadcast_to_tokens(values: &Tensor, tokens: usize) -> Result<Tensor> {
    if values.dim(1)? == 1 {
        values.broadcast_as((values.dim(0)?, tokens, values.dim(2)?))
    } else {
        Ok(values.clone())
    }
}

pub(crate) fn modulate_tokens(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    // Fold the identity into `scale` before broadcasting: `Tensor::ones_like` on
    // a token-broadcast view materializes a full-size activation for nothing.
    let scale = if scale.dtype() == x.dtype() {
        scale.clone()
    } else {
        scale.to_dtype(x.dtype())?
    };
    let shift = if shift.dtype() == x.dtype() {
        shift.clone()
    } else {
        shift.to_dtype(x.dtype())?
    };
    x.broadcast_mul(&scale.affine(1.0, 1.0)?)?
        .broadcast_add(&shift)
}

pub(crate) fn gate_tokens(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let gate = broadcast_to_tokens(gate, x.dim(1)?)?;
    let gate = if gate.dtype() == x.dtype() {
        gate
    } else {
        gate.to_dtype(x.dtype())?
    };
    x.broadcast_mul(&gate)
}

#[derive(Clone, Debug)]
pub(crate) struct LtxPreparedModality {
    pub(crate) x: Tensor,
    pub(crate) context: Tensor,
    pub(crate) context_mask: Option<Tensor>,
    pub(crate) self_attention_mask: Option<Tensor>,
    pub(crate) timesteps: Tensor,
    pub(crate) embedded_timestep: Tensor,
    pub(crate) rope: (Tensor, Tensor),
    pub(crate) cross_rope: Option<(Tensor, Tensor)>,
    pub(crate) cross_scale_shift_timestep: Option<Tensor>,
    pub(crate) cross_gate_timestep: Option<Tensor>,
    pub(crate) prompt_timestep: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub(crate) struct LtxPreparedModalityStatic {
    pub(crate) context: Tensor,
    pub(crate) context_mask: Option<Tensor>,
    pub(crate) self_attention_mask: Option<Tensor>,
    pub(crate) rope: (Tensor, Tensor),
    pub(crate) cross_rope: Option<(Tensor, Tensor)>,
}

#[derive(Clone, Debug)]
pub(crate) struct LtxPreparedStaticInputs {
    video: LtxPreparedModalityStatic,
    audio: Option<LtxPreparedModalityStatic>,
}

#[derive(Clone, Debug)]
pub(crate) struct LtxAvTransformerBlock {
    video_attn1: LtxAttention,
    video_attn2: LtxAttention,
    video_ff: FeedForward,
    video_scale_shift_table: Tensor,
    audio_attn1: LtxAttention,
    audio_attn2: LtxAttention,
    audio_ff: FeedForward,
    audio_scale_shift_table: Tensor,
    audio_to_video_attn: LtxAttention,
    video_to_audio_attn: LtxAttention,
    scale_shift_table_a2v_ca_audio: Tensor,
    scale_shift_table_a2v_ca_video: Tensor,
    norm_eps: f64,
    cross_attention_adaln: bool,
    prompt_scale_shift_table: Option<Tensor>,
    audio_prompt_scale_shift_table: Option<Tensor>,
}

impl LtxAvTransformerBlock {
    pub(crate) fn new(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder,
        lora_registry: Option<&Ltx2LoraRegistry>,
        block_key: &str,
        nvfp4_cache: Option<&Nvfp4LinearCache>,
    ) -> Result<Self> {
        let video_dim = config.inner_dim();
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;

        let video_attn1 = LtxAttention::new(
            video_dim,
            config.num_attention_heads,
            config.num_attention_heads,
            config.attention_head_dim,
            0.0,
            config.attention_bias,
            None,
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            config.apply_gated_attention,
            vb.pp("attn1"),
            lora_registry,
            &format!("{block_key}.attn1"),
            nvfp4_cache,
        )?;
        let video_attn2 = LtxAttention::new(
            video_dim,
            config.num_attention_heads,
            config.num_attention_heads,
            config.attention_head_dim,
            0.0,
            config.attention_bias,
            Some(config.cross_attention_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            config.apply_gated_attention,
            vb.pp("attn2"),
            lora_registry,
            &format!("{block_key}.attn2"),
            nvfp4_cache,
        )?;
        let video_ff = FeedForward::new(
            video_dim,
            vb.pp("ff"),
            lora_registry,
            &format!("{block_key}.ff"),
            nvfp4_cache,
        )?;
        let video_scale_shift_table = vb.get(
            (if config.cross_attention_adaln { 9 } else { 6 }, video_dim),
            "scale_shift_table",
        )?;

        let audio_attn1 = LtxAttention::new(
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            None,
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            config.apply_gated_attention,
            vb.pp("audio_attn1"),
            lora_registry,
            &format!("{block_key}.audio_attn1"),
            nvfp4_cache,
        )?;
        let audio_attn2 = LtxAttention::new(
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            Some(config.audio_cross_attention_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            config.apply_gated_attention,
            vb.pp("audio_attn2"),
            lora_registry,
            &format!("{block_key}.audio_attn2"),
            nvfp4_cache,
        )?;
        let audio_ff = FeedForward::new(
            audio_dim,
            vb.pp("audio_ff"),
            lora_registry,
            &format!("{block_key}.audio_ff"),
            nvfp4_cache,
        )?;
        let audio_scale_shift_table = vb.get(
            (if config.cross_attention_adaln { 9 } else { 6 }, audio_dim),
            "audio_scale_shift_table",
        )?;

        let audio_to_video_attn = LtxAttention::new(
            video_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            Some(audio_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            config.apply_gated_attention,
            vb.pp("audio_to_video_attn"),
            lora_registry,
            &format!("{block_key}.audio_to_video_attn"),
            nvfp4_cache,
        )?;
        let video_to_audio_attn = LtxAttention::new(
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            Some(video_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            config.apply_gated_attention,
            vb.pp("video_to_audio_attn"),
            lora_registry,
            &format!("{block_key}.video_to_audio_attn"),
            nvfp4_cache,
        )?;
        let scale_shift_table_a2v_ca_audio =
            vb.get((5, audio_dim), "scale_shift_table_a2v_ca_audio")?;
        let scale_shift_table_a2v_ca_video =
            vb.get((5, video_dim), "scale_shift_table_a2v_ca_video")?;

        let prompt_scale_shift_table = if config.cross_attention_adaln {
            Some(vb.get((2, video_dim), "prompt_scale_shift_table")?)
        } else {
            None
        };
        let audio_prompt_scale_shift_table = if config.cross_attention_adaln {
            Some(vb.get((2, audio_dim), "audio_prompt_scale_shift_table")?)
        } else {
            None
        };

        Ok(Self {
            video_attn1,
            video_attn2,
            video_ff,
            video_scale_shift_table,
            audio_attn1,
            audio_attn2,
            audio_ff,
            audio_scale_shift_table,
            audio_to_video_attn,
            video_to_audio_attn,
            scale_shift_table_a2v_ca_audio,
            scale_shift_table_a2v_ca_video,
            norm_eps: config.norm_eps,
            cross_attention_adaln: config.cross_attention_adaln,
            prompt_scale_shift_table,
            audio_prompt_scale_shift_table,
        })
    }

    fn get_ada_triplet(
        &self,
        scale_shift_table: &Tensor,
        timestep: &Tensor,
        start_index: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((
            ada_component(scale_shift_table, timestep, start_index)?,
            ada_component(scale_shift_table, timestep, start_index + 1)?,
            ada_component(scale_shift_table, timestep, start_index + 2)?,
        ))
    }

    fn get_cross_ada_values(
        &self,
        scale_shift_table: &Tensor,
        scale_shift_timestep: &Tensor,
        gate_timestep: &Tensor,
        start_index: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let scale_shift_table_head = scale_shift_table.i((0..4, ..))?;
        let gate_table = scale_shift_table.i((4..5, ..))?;
        Ok((
            ada_component(&scale_shift_table_head, scale_shift_timestep, start_index)?,
            ada_component(
                &scale_shift_table_head,
                scale_shift_timestep,
                start_index + 1,
            )?,
            ada_component(&gate_table, gate_timestep, 0)?,
        ))
    }

    fn apply_text_cross_attention(
        &self,
        x: &Tensor,
        context: &Tensor,
        attn: &LtxAttention,
        scale_shift_table: &Tensor,
        prompt_scale_shift_table: Option<&Tensor>,
        timestep: &Tensor,
        prompt_timestep: Option<&Tensor>,
        context_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.cross_attention_adaln {
            let prompt_scale_shift_table = prompt_scale_shift_table.ok_or_else(|| {
                candle_core::Error::msg("cross-attention AdaLN requires prompt scale-shift weights")
            })?;
            let prompt_timestep = prompt_timestep.ok_or_else(|| {
                candle_core::Error::msg("cross-attention AdaLN requires prompt timestep embeddings")
            })?;
            let (shift_q, scale_q, gate) = self.get_ada_triplet(scale_shift_table, timestep, 6)?;
            let attn_input =
                modulate_tokens(&rms_norm_tensor(x, self.norm_eps)?, &scale_q, &shift_q)?;
            let shift_kv = ada_component(prompt_scale_shift_table, prompt_timestep, 0)?;
            let scale_kv = ada_component(prompt_scale_shift_table, prompt_timestep, 1)?;
            let context = modulate_tokens(context, &scale_kv, &shift_kv)?;
            return gate_tokens(
                &attn.forward(
                    &attn_input,
                    Some(&context),
                    context_mask,
                    None,
                    None,
                    None,
                    false,
                )?,
                &gate,
            );
        }
        attn.forward(
            &rms_norm_tensor(x, self.norm_eps)?,
            Some(context),
            context_mask,
            None,
            None,
            None,
            false,
        )
    }

    pub(crate) fn forward(
        &self,
        index: usize,
        video: Option<&LtxPreparedModality>,
        audio: Option<&LtxPreparedModality>,
        perturbations: &BatchedPerturbationConfig,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if video.is_none() && audio.is_none() {
            candle_core::bail!("AV transformer block requires at least one modality");
        }

        let mut vx = None;
        if let Some(video) = video {
            let (v_shift_msa, v_scale_msa, v_gate_msa) =
                self.get_ada_triplet(&self.video_scale_shift_table, &video.timesteps, 0)?;
            let v_self_input = modulate_tokens(
                &rms_norm_tensor(&video.x, self.norm_eps)?,
                &v_scale_msa,
                &v_shift_msa,
            )?;
            let all_video_self_perturbed =
                perturbations.all_in_batch(PerturbationType::SkipVideoSelfAttention, index);
            let video_self_mask = if all_video_self_perturbed
                || !perturbations.any_in_batch(PerturbationType::SkipVideoSelfAttention, index)
            {
                None
            } else {
                Some(perturbations.mask_like(
                    PerturbationType::SkipVideoSelfAttention,
                    index,
                    &v_self_input,
                )?)
            };
            let v_self = gate_tokens(
                &self.video_attn1.forward(
                    &v_self_input,
                    None,
                    video.self_attention_mask.as_ref(),
                    Some((&video.rope.0, &video.rope.1)),
                    None,
                    video_self_mask.as_ref(),
                    all_video_self_perturbed,
                )?,
                &v_gate_msa,
            )?;
            log_detail_tensor(index, "video_input", &video.x)?;
            log_detail_tensor(index, "video_timesteps", &video.timesteps)?;
            log_detail_tensor(index, "video_embedded_timestep", &video.embedded_timestep)?;
            log_detail_tensor(index, "video_context", &video.context)?;
            log_detail_tensor(index, "video_self_input", &v_self_input)?;
            log_detail_tensor(index, "video_self_out", &v_self)?;
            let mut current_vx = video.x.broadcast_add(&v_self)?;
            log_detail_tensor(index, "video_after_self", &current_vx)?;
            let v_text_cross = self.apply_text_cross_attention(
                &current_vx,
                &video.context,
                &self.video_attn2,
                &self.video_scale_shift_table,
                self.prompt_scale_shift_table.as_ref(),
                &video.timesteps,
                video.prompt_timestep.as_ref(),
                video.context_mask.as_ref(),
            )?;
            log_detail_tensor(index, "video_text_cross_out", &v_text_cross)?;
            current_vx = current_vx.broadcast_add(&v_text_cross)?;
            log_detail_tensor(index, "video_after_text_cross", &current_vx)?;
            vx = Some(current_vx);
        }

        let mut ax = None;
        if let Some(audio) = audio {
            let (a_shift_msa, a_scale_msa, a_gate_msa) =
                self.get_ada_triplet(&self.audio_scale_shift_table, &audio.timesteps, 0)?;
            let a_self_input = modulate_tokens(
                &rms_norm_tensor(&audio.x, self.norm_eps)?,
                &a_scale_msa,
                &a_shift_msa,
            )?;
            let all_audio_self_perturbed =
                perturbations.all_in_batch(PerturbationType::SkipAudioSelfAttention, index);
            let audio_self_mask = if all_audio_self_perturbed
                || !perturbations.any_in_batch(PerturbationType::SkipAudioSelfAttention, index)
            {
                None
            } else {
                Some(perturbations.mask_like(
                    PerturbationType::SkipAudioSelfAttention,
                    index,
                    &a_self_input,
                )?)
            };
            let a_self = gate_tokens(
                &self.audio_attn1.forward(
                    &a_self_input,
                    None,
                    audio.self_attention_mask.as_ref(),
                    Some((&audio.rope.0, &audio.rope.1)),
                    None,
                    audio_self_mask.as_ref(),
                    all_audio_self_perturbed,
                )?,
                &a_gate_msa,
            )?;
            log_detail_tensor(index, "audio_input", &audio.x)?;
            log_detail_tensor(index, "audio_timesteps", &audio.timesteps)?;
            log_detail_tensor(index, "audio_embedded_timestep", &audio.embedded_timestep)?;
            log_detail_tensor(index, "audio_context", &audio.context)?;
            log_detail_tensor(index, "audio_self_input", &a_self_input)?;
            log_detail_tensor(index, "audio_self_out", &a_self)?;
            let mut current_ax = audio.x.broadcast_add(&a_self)?;
            log_detail_tensor(index, "audio_after_self", &current_ax)?;
            let a_text_cross = self.apply_text_cross_attention(
                &current_ax,
                &audio.context,
                &self.audio_attn2,
                &self.audio_scale_shift_table,
                self.audio_prompt_scale_shift_table.as_ref(),
                &audio.timesteps,
                audio.prompt_timestep.as_ref(),
                audio.context_mask.as_ref(),
            )?;
            log_detail_tensor(index, "audio_text_cross_out", &a_text_cross)?;
            current_ax = current_ax.broadcast_add(&a_text_cross)?;
            log_detail_tensor(index, "audio_after_text_cross", &current_ax)?;
            ax = Some(current_ax);
        }

        if let (Some(video), Some(audio), Some(vx_before_cross), Some(ax_before_cross)) =
            (video, audio, vx.as_ref(), ax.as_ref())
        {
            let vx_norm3 = rms_norm_tensor(vx_before_cross, self.norm_eps)?;
            let ax_norm3 = rms_norm_tensor(ax_before_cross, self.norm_eps)?;

            let video_cross_scale_shift_timestep =
                video.cross_scale_shift_timestep.as_ref().ok_or_else(|| {
                    candle_core::Error::msg(
                        "video cross scale-shift timestep missing for AV transformer",
                    )
                })?;
            let video_cross_gate_timestep =
                video.cross_gate_timestep.as_ref().ok_or_else(|| {
                    candle_core::Error::msg("video cross gate timestep missing for AV transformer")
                })?;
            let audio_cross_scale_shift_timestep =
                audio.cross_scale_shift_timestep.as_ref().ok_or_else(|| {
                    candle_core::Error::msg(
                        "audio cross scale-shift timestep missing for AV transformer",
                    )
                })?;
            let audio_cross_gate_timestep =
                audio.cross_gate_timestep.as_ref().ok_or_else(|| {
                    candle_core::Error::msg("audio cross gate timestep missing for AV transformer")
                })?;
            let video_cross_rope = video.cross_rope.as_ref().ok_or_else(|| {
                candle_core::Error::msg(
                    "video cross positional embeddings missing for AV transformer",
                )
            })?;
            let audio_cross_rope = audio.cross_rope.as_ref().ok_or_else(|| {
                candle_core::Error::msg(
                    "audio cross positional embeddings missing for AV transformer",
                )
            })?;

            let (v_ca_scale, v_ca_shift, v_gate) = self.get_cross_ada_values(
                &self.scale_shift_table_a2v_ca_video,
                video_cross_scale_shift_timestep,
                video_cross_gate_timestep,
                0,
            )?;
            let (a_ca_scale, a_ca_shift, _) = self.get_cross_ada_values(
                &self.scale_shift_table_a2v_ca_audio,
                audio_cross_scale_shift_timestep,
                audio_cross_gate_timestep,
                0,
            )?;
            let vx_scaled = modulate_tokens(&vx_norm3, &v_ca_scale, &v_ca_shift)?;
            let ax_scaled = modulate_tokens(&ax_norm3, &a_ca_scale, &a_ca_shift)?;
            if !perturbations.all_in_batch(PerturbationType::SkipA2VCrossAttention, index) {
                let a2v_mask = perturbations.mask_like(
                    PerturbationType::SkipA2VCrossAttention,
                    index,
                    vx_before_cross,
                )?;
                let a2v = gate_tokens(
                    &self.audio_to_video_attn.forward(
                        &vx_scaled,
                        Some(&ax_scaled),
                        None,
                        Some((&video_cross_rope.0, &video_cross_rope.1)),
                        Some((&audio_cross_rope.0, &audio_cross_rope.1)),
                        None,
                        false,
                    )?,
                    &v_gate,
                )?;
                let a2v_mask = if a2v_mask.dtype() == a2v.dtype() {
                    a2v_mask
                } else {
                    a2v_mask.to_dtype(a2v.dtype())?
                };
                let a2v = a2v.broadcast_mul(&a2v_mask)?;
                log_detail_tensor(index, "video_av_cross_out", &a2v)?;
                let current_vx = vx_before_cross.broadcast_add(&a2v)?;
                log_detail_tensor(index, "video_after_av_cross", &current_vx)?;
                vx = Some(current_vx);
            }

            let (a_ca_scale, a_ca_shift, a_gate) = self.get_cross_ada_values(
                &self.scale_shift_table_a2v_ca_audio,
                audio_cross_scale_shift_timestep,
                audio_cross_gate_timestep,
                2,
            )?;
            let (v_ca_scale, v_ca_shift, _) = self.get_cross_ada_values(
                &self.scale_shift_table_a2v_ca_video,
                video_cross_scale_shift_timestep,
                video_cross_gate_timestep,
                2,
            )?;
            let ax_scaled = modulate_tokens(&ax_norm3, &a_ca_scale, &a_ca_shift)?;
            let vx_scaled = modulate_tokens(&vx_norm3, &v_ca_scale, &v_ca_shift)?;
            if !perturbations.all_in_batch(PerturbationType::SkipV2ACrossAttention, index) {
                let v2a_mask = perturbations.mask_like(
                    PerturbationType::SkipV2ACrossAttention,
                    index,
                    ax_before_cross,
                )?;
                let v2a = gate_tokens(
                    &self.video_to_audio_attn.forward(
                        &ax_scaled,
                        Some(&vx_scaled),
                        None,
                        Some((&audio_cross_rope.0, &audio_cross_rope.1)),
                        Some((&video_cross_rope.0, &video_cross_rope.1)),
                        None,
                        false,
                    )?,
                    &a_gate,
                )?;
                let v2a_mask = if v2a_mask.dtype() == v2a.dtype() {
                    v2a_mask
                } else {
                    v2a_mask.to_dtype(v2a.dtype())?
                };
                let v2a = v2a.broadcast_mul(&v2a_mask)?;
                log_detail_tensor(index, "audio_av_cross_out", &v2a)?;
                let current_ax = ax_before_cross.broadcast_add(&v2a)?;
                log_detail_tensor(index, "audio_after_av_cross", &current_ax)?;
                ax = Some(current_ax);
            }
        }

        if let (Some(video), Some(vx_before_ff)) = (video, vx.as_ref()) {
            let (v_shift_mlp, v_scale_mlp, v_gate_mlp) =
                self.get_ada_triplet(&self.video_scale_shift_table, &video.timesteps, 3)?;
            let v_ff_input = modulate_tokens(
                &rms_norm_tensor(vx_before_ff, self.norm_eps)?,
                &v_scale_mlp,
                &v_shift_mlp,
            )?;
            let v_ff = gate_tokens(&self.video_ff.forward(&v_ff_input)?, &v_gate_mlp)?;
            log_detail_tensor(index, "video_ff_input", &v_ff_input)?;
            log_detail_tensor(index, "video_ff_out", &v_ff)?;
            let current_vx = vx_before_ff.broadcast_add(&v_ff)?;
            log_detail_tensor(index, "video_after_ff", &current_vx)?;
            vx = Some(current_vx);
        }

        if let (Some(audio), Some(ax_before_ff)) = (audio, ax.as_ref()) {
            let (a_shift_mlp, a_scale_mlp, a_gate_mlp) =
                self.get_ada_triplet(&self.audio_scale_shift_table, &audio.timesteps, 3)?;
            let a_ff_input = modulate_tokens(
                &rms_norm_tensor(ax_before_ff, self.norm_eps)?,
                &a_scale_mlp,
                &a_shift_mlp,
            )?;
            let a_ff = gate_tokens(&self.audio_ff.forward(&a_ff_input)?, &a_gate_mlp)?;
            log_detail_tensor(index, "audio_ff_input", &a_ff_input)?;
            log_detail_tensor(index, "audio_ff_out", &a_ff)?;
            let current_ax = ax_before_ff.broadcast_add(&a_ff)?;
            log_detail_tensor(index, "audio_after_ff", &current_ax)?;
            ax = Some(current_ax);
        }

        Ok((vx, ax))
    }
}

enum AvTransformerBlockSource {
    Eager(Vec<LtxAvTransformerBlock>),
    Streaming(VarBuilder<'static>),
    Adaptive {
        resident_blocks: Vec<Option<LtxAvTransformerBlock>>,
        streamed_vb: VarBuilder<'static>,
        plan: AdaptiveResidencyPlan,
    },
}

pub struct Ltx2AvTransformer3DModel {
    patchify_proj: nn::Linear,
    adaln_single: AdaLayerNormSingle,
    prompt_adaln_single: Option<AdaLayerNormSingle>,
    caption_projection: Option<PixArtAlphaTextProjection>,
    scale_shift_table: Tensor,
    norm_out: LayerNormNoParams,
    proj_out: LtxLinear,
    audio_patchify_proj: nn::Linear,
    audio_adaln_single: AdaLayerNormSingle,
    audio_prompt_adaln_single: Option<AdaLayerNormSingle>,
    audio_caption_projection: Option<PixArtAlphaTextProjection>,
    audio_scale_shift_table: Tensor,
    audio_norm_out: LayerNormNoParams,
    audio_proj_out: LtxLinear,
    av_ca_video_scale_shift_adaln_single: AdaLayerNormSingle,
    av_ca_audio_scale_shift_adaln_single: AdaLayerNormSingle,
    av_ca_a2v_gate_adaln_single: AdaLayerNormSingle,
    av_ca_v2a_gate_adaln_single: AdaLayerNormSingle,
    video_rope: Ltx2VideoRotaryPosEmbed,
    audio_rope: Ltx2VideoRotaryPosEmbed,
    cross_rope: Ltx2VideoRotaryPosEmbed,
    transformer_blocks: AvTransformerBlockSource,
    lora_registry: Option<Arc<Ltx2LoraRegistry>>,
    nvfp4_cache: Nvfp4LinearCache,
    config: Ltx2VideoTransformer3DModelConfig,
}

impl Ltx2AvTransformer3DModel {
    pub fn new(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder,
        lora_registry: Option<Arc<Ltx2LoraRegistry>>,
    ) -> Result<Self> {
        let video_dim = config.inner_dim();
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;
        let nvfp4_cache = Nvfp4LinearCache::default();
        let adaln_embedding_coefficient = if config.cross_attention_adaln { 9 } else { 6 };
        let cross_max = config
            .positional_embedding_max_pos
            .first()
            .copied()
            .unwrap_or(20)
            .max(
                config
                    .audio_positional_embedding_max_pos
                    .first()
                    .copied()
                    .unwrap_or(20),
            );
        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            transformer_blocks.push(LtxAvTransformerBlock::new(
                config,
                vb.pp("transformer_blocks").pp(layer_idx.to_string()),
                lora_registry.as_deref(),
                &format!("diffusion_model.transformer_blocks.{layer_idx}"),
                Some(&nvfp4_cache),
            )?);
            if ltx2_load_debug_enabled() {
                eprintln!(
                    "[ltx2-load] eager_av_block={}/{}",
                    layer_idx + 1,
                    config.num_layers
                );
            }
        }

        Ok(Self {
            patchify_proj: nn::linear(config.in_channels, video_dim, vb.pp("patchify_proj"))?,
            adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                adaln_embedding_coefficient,
                vb.pp("adaln_single"),
            )?,
            prompt_adaln_single: if config.cross_attention_adaln {
                Some(AdaLayerNormSingle::new_with_coefficient(
                    video_dim,
                    2,
                    vb.pp("prompt_adaln_single"),
                )?)
            } else {
                None
            },
            caption_projection: if config.caption_projection_in_transformer {
                Some(PixArtAlphaTextProjection::new_with_out_features(
                    config.caption_channels,
                    video_dim,
                    video_dim,
                    vb.pp("caption_projection"),
                )?)
            } else {
                None
            },
            scale_shift_table: vb.get((2, video_dim), "scale_shift_table")?,
            norm_out: LayerNormNoParams::new(config.norm_eps),
            proj_out: LtxLinear::load_with_nvfp4_cache(
                video_dim,
                config.out_channels,
                true,
                vb.pp("proj_out"),
                lora_adapters_for(lora_registry.as_deref(), "diffusion_model.proj_out"),
                Some(&nvfp4_cache),
                Some("diffusion_model.proj_out"),
            )?,
            audio_patchify_proj: nn::linear(
                config.audio_in_channels,
                audio_dim,
                vb.pp("audio_patchify_proj"),
            )?,
            audio_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                adaln_embedding_coefficient,
                vb.pp("audio_adaln_single"),
            )?,
            audio_prompt_adaln_single: if config.cross_attention_adaln {
                Some(AdaLayerNormSingle::new_with_coefficient(
                    audio_dim,
                    2,
                    vb.pp("audio_prompt_adaln_single"),
                )?)
            } else {
                None
            },
            audio_caption_projection: if config.caption_projection_in_transformer {
                Some(PixArtAlphaTextProjection::new_with_out_features(
                    config.caption_channels,
                    audio_dim,
                    audio_dim,
                    vb.pp("audio_caption_projection"),
                )?)
            } else {
                None
            },
            audio_scale_shift_table: vb.get((2, audio_dim), "audio_scale_shift_table")?,
            audio_norm_out: LayerNormNoParams::new(config.norm_eps),
            audio_proj_out: LtxLinear::load_with_nvfp4_cache(
                audio_dim,
                config.audio_out_channels,
                true,
                vb.pp("audio_proj_out"),
                lora_adapters_for(lora_registry.as_deref(), "diffusion_model.audio_proj_out"),
                Some(&nvfp4_cache),
                Some("diffusion_model.audio_proj_out"),
            )?,
            av_ca_video_scale_shift_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                4,
                vb.pp("av_ca_video_scale_shift_adaln_single"),
            )?,
            av_ca_audio_scale_shift_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                4,
                vb.pp("av_ca_audio_scale_shift_adaln_single"),
            )?,
            av_ca_a2v_gate_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                1,
                vb.pp("av_ca_a2v_gate_adaln_single"),
            )?,
            av_ca_v2a_gate_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                1,
                vb.pp("av_ca_v2a_gate_adaln_single"),
            )?,
            video_rope: Ltx2VideoRotaryPosEmbed::new(
                video_dim,
                config.positional_embedding_theta,
                config.positional_embedding_max_pos.clone(),
                config.use_middle_indices_grid,
                config.num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            audio_rope: Ltx2VideoRotaryPosEmbed::new(
                audio_dim,
                config.positional_embedding_theta,
                config.audio_positional_embedding_max_pos.clone(),
                config.use_middle_indices_grid,
                config.audio_num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            cross_rope: Ltx2VideoRotaryPosEmbed::new(
                config.audio_cross_attention_dim,
                config.positional_embedding_theta,
                vec![cross_max],
                true,
                config.audio_num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            transformer_blocks: AvTransformerBlockSource::Eager(transformer_blocks),
            lora_registry,
            nvfp4_cache,
            config: config.clone(),
        })
    }

    pub fn new_streaming(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder<'static>,
        lora_registry: Option<Arc<Ltx2LoraRegistry>>,
    ) -> Result<Self> {
        let nvfp4_cache = Nvfp4LinearCache::default();
        let transformer_blocks = AvTransformerBlockSource::Streaming(vb.pp("transformer_blocks"));
        Self::new_with_block_source(config, vb, lora_registry, transformer_blocks, nvfp4_cache)
    }

    pub(crate) fn new_adaptive(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder<'static>,
        lora_registry: Option<Arc<Ltx2LoraRegistry>>,
        plan: AdaptiveResidencyPlan,
    ) -> Result<Self> {
        let nvfp4_cache = Nvfp4LinearCache::default();
        if plan.resident.len() != config.num_layers {
            candle_core::bail!(
                "LTX-2 adaptive residency plan has {} layers, expected {}",
                plan.resident.len(),
                config.num_layers
            );
        }

        let blocks_vb = vb.pp("transformer_blocks");
        let mut resident_blocks = Vec::with_capacity(config.num_layers);
        for (index, is_resident) in plan.resident.iter().copied().enumerate() {
            if is_resident {
                resident_blocks.push(Some(LtxAvTransformerBlock::new(
                    config,
                    blocks_vb.pp(index.to_string()),
                    lora_registry.as_deref(),
                    &format!("diffusion_model.transformer_blocks.{index}"),
                    Some(&nvfp4_cache),
                )?));
            } else {
                resident_blocks.push(None);
            }
        }

        let transformer_blocks = AvTransformerBlockSource::Adaptive {
            resident_blocks,
            streamed_vb: blocks_vb,
            plan,
        };
        Self::new_with_block_source(config, vb, lora_registry, transformer_blocks, nvfp4_cache)
    }

    fn new_with_block_source(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder<'static>,
        lora_registry: Option<Arc<Ltx2LoraRegistry>>,
        transformer_blocks: AvTransformerBlockSource,
        nvfp4_cache: Nvfp4LinearCache,
    ) -> Result<Self> {
        let video_dim = config.inner_dim();
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;
        let adaln_embedding_coefficient = if config.cross_attention_adaln { 9 } else { 6 };
        let cross_max = config
            .positional_embedding_max_pos
            .first()
            .copied()
            .unwrap_or(20)
            .max(
                config
                    .audio_positional_embedding_max_pos
                    .first()
                    .copied()
                    .unwrap_or(20),
            );

        Ok(Self {
            patchify_proj: nn::linear(config.in_channels, video_dim, vb.pp("patchify_proj"))?,
            adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                adaln_embedding_coefficient,
                vb.pp("adaln_single"),
            )?,
            prompt_adaln_single: if config.cross_attention_adaln {
                Some(AdaLayerNormSingle::new_with_coefficient(
                    video_dim,
                    2,
                    vb.pp("prompt_adaln_single"),
                )?)
            } else {
                None
            },
            caption_projection: if config.caption_projection_in_transformer {
                Some(PixArtAlphaTextProjection::new_with_out_features(
                    config.caption_channels,
                    video_dim,
                    video_dim,
                    vb.pp("caption_projection"),
                )?)
            } else {
                None
            },
            scale_shift_table: vb.get((2, video_dim), "scale_shift_table")?,
            norm_out: LayerNormNoParams::new(config.norm_eps),
            proj_out: LtxLinear::load_with_nvfp4_cache(
                video_dim,
                config.out_channels,
                true,
                vb.pp("proj_out"),
                lora_adapters_for(lora_registry.as_deref(), "diffusion_model.proj_out"),
                Some(&nvfp4_cache),
                Some("diffusion_model.proj_out"),
            )?,
            audio_patchify_proj: nn::linear(
                config.audio_in_channels,
                audio_dim,
                vb.pp("audio_patchify_proj"),
            )?,
            audio_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                adaln_embedding_coefficient,
                vb.pp("audio_adaln_single"),
            )?,
            audio_prompt_adaln_single: if config.cross_attention_adaln {
                Some(AdaLayerNormSingle::new_with_coefficient(
                    audio_dim,
                    2,
                    vb.pp("audio_prompt_adaln_single"),
                )?)
            } else {
                None
            },
            audio_caption_projection: if config.caption_projection_in_transformer {
                Some(PixArtAlphaTextProjection::new_with_out_features(
                    config.caption_channels,
                    audio_dim,
                    audio_dim,
                    vb.pp("audio_caption_projection"),
                )?)
            } else {
                None
            },
            audio_scale_shift_table: vb.get((2, audio_dim), "audio_scale_shift_table")?,
            audio_norm_out: LayerNormNoParams::new(config.norm_eps),
            audio_proj_out: LtxLinear::load_with_nvfp4_cache(
                audio_dim,
                config.audio_out_channels,
                true,
                vb.pp("audio_proj_out"),
                lora_adapters_for(lora_registry.as_deref(), "diffusion_model.audio_proj_out"),
                Some(&nvfp4_cache),
                Some("diffusion_model.audio_proj_out"),
            )?,
            av_ca_video_scale_shift_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                4,
                vb.pp("av_ca_video_scale_shift_adaln_single"),
            )?,
            av_ca_audio_scale_shift_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                4,
                vb.pp("av_ca_audio_scale_shift_adaln_single"),
            )?,
            av_ca_a2v_gate_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                1,
                vb.pp("av_ca_a2v_gate_adaln_single"),
            )?,
            av_ca_v2a_gate_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                1,
                vb.pp("av_ca_v2a_gate_adaln_single"),
            )?,
            video_rope: Ltx2VideoRotaryPosEmbed::new(
                video_dim,
                config.positional_embedding_theta,
                config.positional_embedding_max_pos.clone(),
                config.use_middle_indices_grid,
                config.num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            audio_rope: Ltx2VideoRotaryPosEmbed::new(
                audio_dim,
                config.positional_embedding_theta,
                config.audio_positional_embedding_max_pos.clone(),
                config.use_middle_indices_grid,
                config.audio_num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            cross_rope: Ltx2VideoRotaryPosEmbed::new(
                config.audio_cross_attention_dim,
                config.positional_embedding_theta,
                vec![cross_max],
                true,
                config.audio_num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            transformer_blocks,
            lora_registry,
            nvfp4_cache,
            config: config.clone(),
        })
    }

    #[cfg(test)]
    pub(crate) fn adaptive_residency_counts(&self) -> Option<(usize, usize)> {
        match &self.transformer_blocks {
            AvTransformerBlockSource::Adaptive {
                resident_blocks, ..
            } => {
                let resident_count = resident_blocks
                    .iter()
                    .filter(|block| block.is_some())
                    .count();
                Some((
                    resident_count,
                    resident_blocks.len().saturating_sub(resident_count),
                ))
            }
            _ => None,
        }
    }

    fn streaming_block(
        &self,
        blocks_vb: VarBuilder<'static>,
        index: usize,
    ) -> Result<LtxAvTransformerBlock> {
        LtxAvTransformerBlock::new(
            &self.config,
            blocks_vb.pp(index.to_string()),
            self.lora_registry.as_deref(),
            &format!("diffusion_model.transformer_blocks.{index}"),
            Some(&self.nvfp4_cache),
        )
    }

    pub(crate) fn prepare_context_mask(
        mask: Option<&Tensor>,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        match mask {
            Some(mask) if mask.rank() == 2 => Ok(Some(
                (mask.to_dtype(dtype)?.affine(-1.0, 1.0)? * (-10000.0))?.unsqueeze(1)?,
            )),
            Some(mask) => Ok(Some(mask.clone())),
            None => Ok(None),
        }
    }

    pub(crate) fn prepare_self_attention_mask(
        mask: Option<&Tensor>,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        match mask {
            Some(mask) => {
                let mask_f32 = mask.to_dtype(DType::F32)?;
                let positive = mask_f32.gt(&mask_f32.zeros_like()?)?;
                let log_bias = mask_f32.clamp(f32::MIN_POSITIVE, f32::INFINITY)?.log()?;
                let neg_inf =
                    Tensor::full(f32::NEG_INFINITY, mask_f32.shape().dims(), mask.device())?;
                let bias = positive
                    .where_cond(&log_bias, &neg_inf)?
                    .to_dtype(dtype)?
                    .unsqueeze(1)?;
                Ok(Some(bias))
            }
            None => Ok(None),
        }
    }

    fn temporal_cross_positions(&self, positions: &Tensor, expected_dims: usize) -> Result<Tensor> {
        let dims = positions.dim(1)?;
        if dims == expected_dims {
            return Ok(positions.clone());
        }
        if dims < 1 {
            candle_core::bail!("expected at least one positional dimension, got {dims}");
        }
        positions.i((.., 0..1, .., ..))
    }

    pub(crate) fn reshape_adaln_output(output: &Tensor, batch: usize) -> Result<Tensor> {
        let (rows, dim) = output.dims2()?;
        if rows % batch != 0 {
            candle_core::bail!(
                "AdaLN output row count {rows} is not divisible by batch size {batch}"
            );
        }
        output.reshape((batch, rows / batch, dim))
    }

    fn prepare_modality(
        &self,
        latent: &Tensor,
        timesteps: &Tensor,
        sigma: &Tensor,
        patchify_proj: &nn::Linear,
        adaln_single: &AdaLayerNormSingle,
        prompt_adaln_single: Option<&AdaLayerNormSingle>,
        static_inputs: &LtxPreparedModalityStatic,
    ) -> Result<LtxPreparedModality> {
        let x = patchify_proj.forward(latent)?;
        let (timesteps, embedded_timestep) = adaln_single.forward(&timesteps.flatten_all()?)?;
        let batch = x.dim(0)?;
        let timesteps = Self::reshape_adaln_output(&timesteps, batch)?;
        let embedded_timestep = Self::reshape_adaln_output(&embedded_timestep, batch)?;
        let prompt_timestep = if let Some(prompt_adaln_single) = prompt_adaln_single {
            let (prompt_timestep, _) = prompt_adaln_single.forward(&sigma.flatten_all()?)?;
            Some(Self::reshape_adaln_output(&prompt_timestep, batch)?)
        } else {
            None
        };
        Ok(LtxPreparedModality {
            x,
            context: static_inputs.context.clone(),
            context_mask: static_inputs.context_mask.clone(),
            self_attention_mask: static_inputs.self_attention_mask.clone(),
            timesteps,
            embedded_timestep,
            rope: static_inputs.rope.clone(),
            cross_rope: static_inputs.cross_rope.clone(),
            cross_scale_shift_timestep: None,
            cross_gate_timestep: None,
            prompt_timestep,
        })
    }

    fn prepare_modality_static(
        &self,
        context: &Tensor,
        context_mask: Option<&Tensor>,
        self_attention_mask: Option<&Tensor>,
        positions: &Tensor,
        caption_projection: Option<&PixArtAlphaTextProjection>,
        rope: &Ltx2VideoRotaryPosEmbed,
        cross_positions: Option<&Tensor>,
        compute_dtype: DType,
    ) -> Result<LtxPreparedModalityStatic> {
        let context = context.to_dtype(compute_dtype)?;
        let context = if let Some(caption_projection) = caption_projection {
            caption_projection.forward(&context)?
        } else {
            context
        };
        let rope = rope.forward_for_dtype(context.device(), compute_dtype, positions)?;
        let cross_rope = cross_positions
            .map(|cross_positions| {
                self.cross_rope
                    .forward_for_dtype(context.device(), compute_dtype, cross_positions)
            })
            .transpose()?;
        Ok(LtxPreparedModalityStatic {
            context,
            context_mask: Self::prepare_context_mask(context_mask, compute_dtype)?,
            self_attention_mask: Self::prepare_self_attention_mask(
                self_attention_mask,
                compute_dtype,
            )?,
            rope,
            cross_rope,
        })
    }

    pub(crate) fn prepare_static_inputs(
        &self,
        video_encoder_hidden_states: &Tensor,
        audio_encoder_hidden_states: Option<&Tensor>,
        video_encoder_attention_mask: Option<&Tensor>,
        audio_encoder_attention_mask: Option<&Tensor>,
        video_self_attention_mask: Option<&Tensor>,
        audio_self_attention_mask: Option<&Tensor>,
        video_positions: &Tensor,
        audio_positions: Option<&Tensor>,
    ) -> Result<LtxPreparedStaticInputs> {
        let compute_dtype = match self.patchify_proj.weight().dtype() {
            DType::F8E4M3 => DType::BF16,
            other => other,
        };
        let video_cross_positions = audio_positions
            .map(|_| self.temporal_cross_positions(video_positions, 1))
            .transpose()?;
        let audio_cross_positions = audio_positions
            .map(|positions| self.temporal_cross_positions(positions, 1))
            .transpose()?;
        let video = self.prepare_modality_static(
            video_encoder_hidden_states,
            video_encoder_attention_mask,
            video_self_attention_mask,
            video_positions,
            self.caption_projection.as_ref(),
            &self.video_rope,
            video_cross_positions.as_ref(),
            compute_dtype,
        )?;
        let audio = match (audio_encoder_hidden_states, audio_positions) {
            (Some(audio_encoder_hidden_states), Some(audio_positions)) => Some(
                self.prepare_modality_static(
                    audio_encoder_hidden_states,
                    audio_encoder_attention_mask,
                    audio_self_attention_mask,
                    audio_positions,
                    self.audio_caption_projection.as_ref(),
                    &self.audio_rope,
                    audio_cross_positions.as_ref(),
                    compute_dtype,
                )?,
            ),
            (None, None) => None,
            _ => candle_core::bail!(
                "audio hidden states and positions must be provided together when preparing static inputs"
            ),
        };
        Ok(LtxPreparedStaticInputs { video, audio })
    }

    fn prepare_cross_attention_timestep(
        adaln: &AdaLayerNormSingle,
        timestep: &Tensor,
        scale: f64,
        batch: usize,
    ) -> Result<Tensor> {
        let (output, _) = adaln.forward(&timestep.flatten_all()?.affine(scale, 0.0)?)?;
        Self::reshape_adaln_output(&output, batch)
    }

    pub(crate) fn process_output(
        scale_shift_table: &Tensor,
        norm_out: &LayerNormNoParams,
        proj_out: &LtxLinear,
        x: &Tensor,
        embedded_timestep: &Tensor,
    ) -> Result<Tensor> {
        let table = scale_shift_table
            .to_dtype(embedded_timestep.dtype())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let scale_shift = table.broadcast_add(&embedded_timestep.unsqueeze(2)?)?;
        let shift = scale_shift.i((.., .., 0, ..))?;
        let scale = scale_shift.i((.., .., 1, ..))?;
        let x = norm_out.forward(x)?;
        // Same identity fold as `modulate_tokens`: add 1.0 to the small
        // per-sample scale before it is broadcast across the token axis.
        let scale = if scale.dtype() == x.dtype() {
            scale
        } else {
            scale.to_dtype(x.dtype())?
        };
        let shift = if shift.dtype() == x.dtype() {
            shift
        } else {
            shift.to_dtype(x.dtype())?
        };
        proj_out.forward(
            &x.broadcast_mul(&scale.affine(1.0, 1.0)?)?
                .broadcast_add(&shift)?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_static_inputs(
        &self,
        video_hidden_states: &Tensor,
        audio_hidden_states: Option<&Tensor>,
        video_sigma: &Tensor,
        video_timestep: &Tensor,
        audio_sigma: Option<&Tensor>,
        audio_timestep: Option<&Tensor>,
        static_inputs: &LtxPreparedStaticInputs,
        perturbations: Option<&BatchedPerturbationConfig>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let compute_dtype = match self.patchify_proj.weight().dtype() {
            DType::F8E4M3 => DType::BF16,
            other => other,
        };
        let video_sigma = video_sigma.to_dtype(compute_dtype)?.affine(1000.0, 0.0)?;
        let video_timestep = video_timestep
            .to_dtype(compute_dtype)?
            .affine(1000.0, 0.0)?;
        let audio_sigma = audio_sigma
            .map(|sigma| sigma.to_dtype(compute_dtype)?.affine(1000.0, 0.0))
            .transpose()?;
        let audio_timestep = audio_timestep
            .map(|timestep| timestep.to_dtype(compute_dtype)?.affine(1000.0, 0.0))
            .transpose()?;
        let mut video = self.prepare_modality(
            &video_hidden_states.to_dtype(compute_dtype)?,
            &video_timestep,
            &video_sigma,
            &self.patchify_proj,
            &self.adaln_single,
            self.prompt_adaln_single.as_ref(),
            &static_inputs.video,
        )?;
        let audio_sigma_ref = audio_sigma.as_ref();
        let audio_timestep_ref = audio_timestep.as_ref();
        let mut audio = match (
            audio_hidden_states,
            static_inputs.audio.as_ref(),
            audio_sigma_ref,
            audio_timestep_ref,
        ) {
            (
                Some(audio_hidden_states),
                Some(audio_static_inputs),
                Some(audio_sigma),
                Some(audio_timestep),
            ) => Some(self.prepare_modality(
                &audio_hidden_states.to_dtype(compute_dtype)?,
                audio_timestep,
                audio_sigma,
                &self.audio_patchify_proj,
                &self.audio_adaln_single,
                self.audio_prompt_adaln_single.as_ref(),
                audio_static_inputs,
            )?),
            (None, None, None, None) => None,
            _ => candle_core::bail!(
                "audio hidden states, static inputs, sigma, and timesteps must be provided together"
            ),
        };
        let batch = video.x.dim(0)?;
        let perturbations = perturbations
            .cloned()
            .unwrap_or_else(|| BatchedPerturbationConfig::empty(batch));
        let av_scale = self.config.av_ca_timestep_scale_multiplier / 1000.0;
        if let Some(audio) = audio.as_mut() {
            video.cross_scale_shift_timestep = Some(Self::prepare_cross_attention_timestep(
                &self.av_ca_video_scale_shift_adaln_single,
                audio_sigma.as_ref().expect("audio sigma already validated"),
                1.0,
                batch,
            )?);
            audio.cross_scale_shift_timestep = Some(Self::prepare_cross_attention_timestep(
                &self.av_ca_audio_scale_shift_adaln_single,
                &video_sigma,
                1.0,
                batch,
            )?);
            video.cross_gate_timestep = Some(Self::prepare_cross_attention_timestep(
                &self.av_ca_a2v_gate_adaln_single,
                audio_sigma.as_ref().expect("audio sigma already validated"),
                av_scale,
                batch,
            )?);
            audio.cross_gate_timestep = Some(Self::prepare_cross_attention_timestep(
                &self.av_ca_v2a_gate_adaln_single,
                &video_sigma,
                av_scale,
                batch,
            )?);
        }

        match &self.transformer_blocks {
            AvTransformerBlockSource::Eager(blocks) => {
                for (index, block) in blocks.iter().enumerate() {
                    if ltx2_block_debug_enabled() {
                        eprintln!("[ltx2-block-debug] enter block={index}");
                    }
                    let (vx, ax) =
                        block.forward(index, Some(&video), audio.as_ref(), &perturbations)?;
                    video.x = vx.ok_or_else(|| {
                        candle_core::Error::msg("video branch unexpectedly returned no output")
                    })?;
                    if let (Some(audio), Some(ax)) = (audio.as_mut(), ax) {
                        audio.x = ax;
                    }
                    if ltx2_block_debug_enabled() {
                        let (v_mean, v_abs_mean, v_abs_max) = tensor_debug_stats(&video.x)?;
                        if let Some(audio) = audio.as_ref() {
                            let (a_mean, a_abs_mean, a_abs_max) = tensor_debug_stats(&audio.x)?;
                            eprintln!(
                                "[ltx2-block-debug] block={index} video(mean={v_mean:.6}, abs_mean={v_abs_mean:.6}, abs_max={v_abs_max:.6}) audio(mean={a_mean:.6}, abs_mean={a_abs_mean:.6}, abs_max={a_abs_max:.6})"
                            );
                        } else {
                            eprintln!(
                                "[ltx2-block-debug] block={index} video(mean={v_mean:.6}, abs_mean={v_abs_mean:.6}, abs_max={v_abs_max:.6})"
                            );
                        }
                    }
                }
            }
            AvTransformerBlockSource::Streaming(blocks_vb) => {
                for index in 0..self.config.num_layers {
                    if ltx2_block_debug_enabled() {
                        eprintln!("[ltx2-block-debug] enter block={index}");
                    }
                    let block = self.streaming_block(blocks_vb.clone(), index)?;
                    let (vx, ax) =
                        block.forward(index, Some(&video), audio.as_ref(), &perturbations)?;
                    video.x = vx.ok_or_else(|| {
                        candle_core::Error::msg("video branch unexpectedly returned no output")
                    })?;
                    if let (Some(audio), Some(ax)) = (audio.as_mut(), ax) {
                        audio.x = ax;
                    }
                    if ltx2_block_debug_enabled() {
                        let (v_mean, v_abs_mean, v_abs_max) = tensor_debug_stats(&video.x)?;
                        if let Some(audio) = audio.as_ref() {
                            let (a_mean, a_abs_mean, a_abs_max) = tensor_debug_stats(&audio.x)?;
                            eprintln!(
                                "[ltx2-block-debug] block={index} video(mean={v_mean:.6}, abs_mean={v_abs_mean:.6}, abs_max={v_abs_max:.6}) audio(mean={a_mean:.6}, abs_mean={a_abs_mean:.6}, abs_max={a_abs_max:.6})"
                            );
                        } else {
                            eprintln!(
                                "[ltx2-block-debug] block={index} video(mean={v_mean:.6}, abs_mean={v_abs_mean:.6}, abs_max={v_abs_max:.6})"
                            );
                        }
                    }
                    if video.x.device().is_cuda()
                        && should_synchronize_streaming_layer(
                            index,
                            self.config.num_layers,
                            self.config.streaming_prefetch_count,
                        )
                    {
                        video.x.device().synchronize()?;
                    }
                }
            }
            AvTransformerBlockSource::Adaptive {
                resident_blocks,
                streamed_vb,
                plan,
            } => {
                debug_assert_eq!(plan.resident.len(), resident_blocks.len());
                for index in 0..self.config.num_layers {
                    if ltx2_block_debug_enabled() {
                        eprintln!("[ltx2-block-debug] enter block={index}");
                    }
                    let should_reside = plan.resident.get(index).copied().unwrap_or(false);
                    let (vx, ax) = if should_reside {
                        let block = resident_blocks
                            .get(index)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| {
                                candle_core::Error::msg(format!(
                                    "missing resident LTX-2 block {index} from adaptive plan"
                                ))
                            })?;
                        block.forward(index, Some(&video), audio.as_ref(), &perturbations)?
                    } else {
                        let block = self.streaming_block(streamed_vb.clone(), index)?;
                        block.forward(index, Some(&video), audio.as_ref(), &perturbations)?
                    };
                    video.x = vx.ok_or_else(|| {
                        candle_core::Error::msg("video branch unexpectedly returned no output")
                    })?;
                    if let (Some(audio), Some(ax)) = (audio.as_mut(), ax) {
                        audio.x = ax;
                    }
                    if ltx2_block_debug_enabled() {
                        let (v_mean, v_abs_mean, v_abs_max) = tensor_debug_stats(&video.x)?;
                        if let Some(audio) = audio.as_ref() {
                            let (a_mean, a_abs_mean, a_abs_max) = tensor_debug_stats(&audio.x)?;
                            eprintln!(
                                "[ltx2-block-debug] block={index} video(mean={v_mean:.6}, abs_mean={v_abs_mean:.6}, abs_max={v_abs_max:.6}) audio(mean={a_mean:.6}, abs_mean={a_abs_mean:.6}, abs_max={a_abs_max:.6})"
                            );
                        } else {
                            eprintln!(
                                "[ltx2-block-debug] block={index} video(mean={v_mean:.6}, abs_mean={v_abs_mean:.6}, abs_max={v_abs_max:.6})"
                            );
                        }
                    }
                    if video.x.device().is_cuda()
                        && should_synchronize_streaming_layer(
                            index,
                            self.config.num_layers,
                            self.config.streaming_prefetch_count,
                        )
                    {
                        video.x.device().synchronize()?;
                    }
                }
            }
        }

        let video = Self::process_output(
            &self.scale_shift_table,
            &self.norm_out,
            &self.proj_out,
            &video.x,
            &video.embedded_timestep,
        )?;
        let audio = match audio {
            Some(audio) => Some(Self::process_output(
                &self.audio_scale_shift_table,
                &self.audio_norm_out,
                &self.audio_proj_out,
                &audio.x,
                &audio.embedded_timestep,
            )?),
            None => None,
        };

        Ok((video, audio))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn forward(
        &self,
        video_hidden_states: &Tensor,
        audio_hidden_states: Option<&Tensor>,
        video_encoder_hidden_states: &Tensor,
        audio_encoder_hidden_states: Option<&Tensor>,
        video_sigma: &Tensor,
        video_timestep: &Tensor,
        audio_sigma: Option<&Tensor>,
        audio_timestep: Option<&Tensor>,
        video_encoder_attention_mask: Option<&Tensor>,
        audio_encoder_attention_mask: Option<&Tensor>,
        video_self_attention_mask: Option<&Tensor>,
        audio_self_attention_mask: Option<&Tensor>,
        video_positions: &Tensor,
        audio_positions: Option<&Tensor>,
        perturbations: Option<&BatchedPerturbationConfig>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let static_inputs = self.prepare_static_inputs(
            video_encoder_hidden_states,
            audio_encoder_hidden_states,
            video_encoder_attention_mask,
            audio_encoder_attention_mask,
            video_self_attention_mask,
            audio_self_attention_mask,
            video_positions,
            audio_positions,
        )?;
        self.forward_with_static_inputs(
            video_hidden_states,
            audio_hidden_states,
            video_sigma,
            video_timestep,
            audio_sigma,
            audio_timestep,
            &static_inputs,
            perturbations,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Linear, Module, VarBuilder};

    use super::{
        cached_timestep_embedding_inv_freq, emulate_static_fp8_input_quantization, gate_tokens,
        modulate_tokens, FeedForward, LayerNormNoParams, LinearLoraAdapter,
        Ltx2AvTransformer3DModel, Ltx2VideoRotaryPosEmbed, Ltx2VideoTransformer3DModelConfig,
        LtxAttention, LtxLinear, LtxRopeType, Nvfp4LinearCache, TimestepEmbeddingInvFreqCache,
        TimestepEmbeddingInvFreqCacheKey,
    };

    fn fp8_input_scale_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn fp8_weight_scale_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            // SAFETY: these tests serialize access to the process-wide env var
            // through `fp8_input_scale_env_lock`.
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            // SAFETY: these tests serialize access to the process-wide env var
            // through `fp8_input_scale_env_lock`.
            unsafe { std::env::remove_var(key) };
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => {
                    // SAFETY: these tests serialize access to the process-wide env var
                    // through `fp8_input_scale_env_lock`.
                    unsafe { std::env::set_var(self.key, previous) };
                }
                None => {
                    // SAFETY: these tests serialize access to the process-wide env var
                    // through `fp8_input_scale_env_lock`.
                    unsafe { std::env::remove_var(self.key) };
                }
            }
        }
    }

    fn attention_var_builder(dim: usize) -> VarBuilder<'static> {
        attention_var_builder_with_gate(dim, None)
    }

    fn attention_var_builder_with_gate(dim: usize, gate_bias: Option<f32>) -> VarBuilder<'static> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        let mut identity = vec![0.0f32; dim * dim];
        for idx in 0..dim {
            identity[idx * dim + idx] = 1.0;
        }
        for name in ["to_q", "to_k", "to_v", "to_out.0"] {
            tensors.insert(
                format!("{name}.weight"),
                Tensor::from_vec(identity.clone(), (dim, dim), &device).unwrap(),
            );
            tensors.insert(
                format!("{name}.bias"),
                Tensor::zeros(dim, DType::F32, &device).unwrap(),
            );
        }
        tensors.insert(
            "norm_q.weight".to_string(),
            Tensor::ones(dim, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "norm_k.weight".to_string(),
            Tensor::ones(dim, DType::F32, &device).unwrap(),
        );
        if let Some(gate_bias) = gate_bias {
            tensors.insert(
                "to_gate_logits.weight".to_string(),
                Tensor::zeros((1, dim), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                "to_gate_logits.bias".to_string(),
                Tensor::full(gate_bias, 1, &device).unwrap(),
            );
        }
        VarBuilder::from_tensors(tensors, DType::F32, &device)
    }

    fn patterned_values(len: usize, offset: usize) -> Vec<f32> {
        (0..len)
            .map(|index| (((index + offset) % 19) as f32 - 9.0) / 16.0)
            .collect()
    }

    #[test]
    fn ltx2_timestep_inv_freq_cache_reuses_shape_device_entry() {
        let cache = TimestepEmbeddingInvFreqCache::default();
        let device = Device::Cpu;
        let key = TimestepEmbeddingInvFreqCacheKey::new(256, &device);

        let (first, first_hit) =
            cached_timestep_embedding_inv_freq(&cache, key.clone(), &device, 128).unwrap();
        let (second, second_hit) =
            cached_timestep_embedding_inv_freq(&cache, key, &device, 128).unwrap();

        assert!(!first_hit);
        assert!(second_hit);
        assert_eq!(first.dims1().unwrap(), 128);
        assert_eq!(second.dims1().unwrap(), 128);
        assert_eq!(second.dtype(), DType::F32);
        assert_eq!(format!("{:?}", second.device()), format!("{device:?}"));
    }

    #[test]
    fn ltx2_rotary_base_indices_cache_distinguishes_position_dims() {
        let device = Device::Cpu;
        let rope = Ltx2VideoRotaryPosEmbed::new(
            16,
            10_000.0,
            vec![20, 2048, 2048],
            true,
            2,
            LtxRopeType::Split,
            false,
        );

        let (first, first_hit) = rope.base_indices_cached(&device, 2).unwrap();
        let (second, second_hit) = rope.base_indices_cached(&device, 2).unwrap();
        let (third, third_hit) = rope.base_indices_cached(&device, 1).unwrap();

        assert!(!first_hit);
        assert!(second_hit);
        assert!(!third_hit);
        assert_eq!(first.dims1().unwrap(), 4);
        assert_eq!(second.dims1().unwrap(), 4);
        assert_eq!(third.dims1().unwrap(), 8);
        assert_eq!(format!("{:?}", second.device()), format!("{device:?}"));
    }

    fn insert_linear(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
        fp8: bool,
    ) {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            patterned_values(out_dim * in_dim, prefix.len()),
            (out_dim, in_dim),
            &device,
        )
        .unwrap();
        let weight = if fp8 {
            weight.to_dtype(DType::F8E4M3).unwrap()
        } else {
            weight
        };
        tensors.insert(format!("{prefix}.weight"), weight);
        tensors.insert(
            format!("{prefix}.bias"),
            Tensor::from_vec(
                patterned_values(out_dim, prefix.len() + 7),
                out_dim,
                &device,
            )
            .unwrap(),
        );
        if fp8 {
            tensors.insert(
                format!("{prefix}.input_scale"),
                Tensor::new(1.0f32, &device).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.weight_scale"),
                Tensor::new(1.0f32, &device).unwrap(),
            );
        }
    }

    fn insert_nvfp4_linear(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
    ) {
        assert!(
            in_dim.is_multiple_of(crate::nvfp4::NVFP4_BLOCK_SIZE),
            "NVFP4 test fixture input dim must be block-aligned"
        );
        let device = Device::Cpu;
        let packed = vec![0x22u8; out_dim * in_dim / 2];
        tensors.insert(
            format!("{prefix}.weight.nvfp4_packed"),
            Tensor::from_vec(packed, (out_dim, in_dim / 2), &device).unwrap(),
        );

        let scale_cols = in_dim / crate::nvfp4::NVFP4_BLOCK_SIZE;
        let unswizzled = vec![1.0f32; out_dim * scale_cols];
        let swizzled =
            crate::nvfp4::swizzle_block_scales(&unswizzled, out_dim, scale_cols).unwrap();
        tensors.insert(
            format!("{prefix}.weight.nvfp4_block_scales"),
            Tensor::from_vec(
                swizzled,
                (out_dim.div_ceil(128) * 128, scale_cols.div_ceil(4) * 4),
                &device,
            )
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap(),
        );
        tensors.insert(
            format!("{prefix}.weight.nvfp4_tensor_scale"),
            Tensor::new(1.0f32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.bias"),
            Tensor::from_vec(
                patterned_values(out_dim, prefix.len() + 7),
                out_dim,
                &device,
            )
            .unwrap(),
        );
    }

    fn insert_rms_norm(tensors: &mut HashMap<String, Tensor>, prefix: &str, dim: usize) {
        tensors.insert(
            format!("{prefix}.weight"),
            Tensor::ones(dim, DType::F32, &Device::Cpu).unwrap(),
        );
    }

    fn insert_matrix(
        tensors: &mut HashMap<String, Tensor>,
        name: &str,
        rows: usize,
        cols: usize,
        offset: usize,
    ) {
        tensors.insert(
            name.to_string(),
            Tensor::from_vec(
                patterned_values(rows * cols, offset),
                (rows, cols),
                &Device::Cpu,
            )
            .unwrap(),
        );
    }

    fn insert_adaln_single(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        coefficient: usize,
    ) {
        insert_linear(
            tensors,
            &format!("{prefix}.emb.timestep_embedder.linear_1"),
            dim,
            256,
            false,
        );
        insert_linear(
            tensors,
            &format!("{prefix}.emb.timestep_embedder.linear_2"),
            dim,
            dim,
            false,
        );
        insert_linear(
            tensors,
            &format!("{prefix}.linear"),
            coefficient * dim,
            dim,
            false,
        );
    }

    fn insert_text_projection(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
    ) {
        insert_linear(
            tensors,
            &format!("{prefix}.linear_1"),
            hidden_dim,
            in_dim,
            false,
        );
        insert_linear(
            tensors,
            &format!("{prefix}.linear_2"),
            out_dim,
            hidden_dim,
            false,
        );
    }

    fn insert_attention(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        query_dim: usize,
        context_dim: usize,
        heads: usize,
        dim_head: usize,
        apply_gated_attention: bool,
        fp8: bool,
    ) {
        let inner_dim = heads * dim_head;
        insert_rms_norm(tensors, &format!("{prefix}.norm_q"), inner_dim);
        insert_rms_norm(tensors, &format!("{prefix}.norm_k"), inner_dim);
        insert_linear(
            tensors,
            &format!("{prefix}.to_q"),
            inner_dim,
            query_dim,
            fp8,
        );
        insert_linear(
            tensors,
            &format!("{prefix}.to_k"),
            inner_dim,
            context_dim,
            fp8,
        );
        insert_linear(
            tensors,
            &format!("{prefix}.to_v"),
            inner_dim,
            context_dim,
            fp8,
        );
        insert_linear(
            tensors,
            &format!("{prefix}.to_out.0"),
            query_dim,
            inner_dim,
            fp8,
        );
        if apply_gated_attention {
            insert_linear(
                tensors,
                &format!("{prefix}.to_gate_logits"),
                heads,
                query_dim,
                fp8,
            );
        }
    }

    fn insert_feed_forward(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        fp8: bool,
    ) {
        insert_linear(tensors, &format!("{prefix}.net.0.proj"), dim * 4, dim, fp8);
        insert_linear(tensors, &format!("{prefix}.net.2"), dim, dim * 4, fp8);
    }

    pub(crate) fn insert_av_block(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        config: &Ltx2VideoTransformer3DModelConfig,
        fp8: bool,
    ) {
        let video_dim = config.inner_dim();
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;

        insert_attention(
            tensors,
            &format!("{prefix}.attn1"),
            video_dim,
            video_dim,
            config.num_attention_heads,
            config.attention_head_dim,
            config.apply_gated_attention,
            fp8,
        );
        insert_attention(
            tensors,
            &format!("{prefix}.attn2"),
            video_dim,
            config.cross_attention_dim,
            config.num_attention_heads,
            config.attention_head_dim,
            config.apply_gated_attention,
            fp8,
        );
        insert_feed_forward(tensors, &format!("{prefix}.ff"), video_dim, fp8);
        insert_matrix(
            tensors,
            &format!("{prefix}.scale_shift_table"),
            if config.cross_attention_adaln { 9 } else { 6 },
            video_dim,
            prefix.len(),
        );
        if config.cross_attention_adaln {
            insert_matrix(
                tensors,
                &format!("{prefix}.prompt_scale_shift_table"),
                2,
                video_dim,
                prefix.len() + 1,
            );
        }

        insert_attention(
            tensors,
            &format!("{prefix}.audio_attn1"),
            audio_dim,
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            config.apply_gated_attention,
            fp8,
        );
        insert_attention(
            tensors,
            &format!("{prefix}.audio_attn2"),
            audio_dim,
            config.audio_cross_attention_dim,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            config.apply_gated_attention,
            fp8,
        );
        insert_feed_forward(tensors, &format!("{prefix}.audio_ff"), audio_dim, fp8);
        insert_matrix(
            tensors,
            &format!("{prefix}.audio_scale_shift_table"),
            if config.cross_attention_adaln { 9 } else { 6 },
            audio_dim,
            prefix.len() + 3,
        );
        if config.cross_attention_adaln {
            insert_matrix(
                tensors,
                &format!("{prefix}.audio_prompt_scale_shift_table"),
                2,
                audio_dim,
                prefix.len() + 4,
            );
        }

        insert_attention(
            tensors,
            &format!("{prefix}.audio_to_video_attn"),
            video_dim,
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            config.apply_gated_attention,
            fp8,
        );
        insert_attention(
            tensors,
            &format!("{prefix}.video_to_audio_attn"),
            audio_dim,
            video_dim,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            config.apply_gated_attention,
            fp8,
        );
        insert_matrix(
            tensors,
            &format!("{prefix}.scale_shift_table_a2v_ca_audio"),
            5,
            audio_dim,
            prefix.len() + 5,
        );
        insert_matrix(
            tensors,
            &format!("{prefix}.scale_shift_table_a2v_ca_video"),
            5,
            video_dim,
            prefix.len() + 7,
        );
    }

    pub(crate) fn tiny_av_config() -> Ltx2VideoTransformer3DModelConfig {
        Ltx2VideoTransformer3DModelConfig {
            in_channels: 2,
            out_channels: 2,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: 1,
            attention_head_dim: 8,
            cross_attention_dim: 8,
            num_layers: 2,
            qk_norm: "rms_norm".to_string(),
            norm_elementwise_affine: false,
            norm_eps: 1e-6,
            caption_channels: 4,
            caption_projection_in_transformer: true,
            attention_bias: true,
            attention_out_bias: true,
            positional_embedding_theta: 10_000.0,
            positional_embedding_max_pos: vec![4, 4, 4],
            use_middle_indices_grid: true,
            rope_type: LtxRopeType::Split,
            double_precision_rope: true,
            audio_num_attention_heads: 1,
            audio_attention_head_dim: 8,
            audio_in_channels: 2,
            audio_out_channels: 2,
            audio_cross_attention_dim: 8,
            audio_positional_embedding_max_pos: vec![4],
            apply_gated_attention: false,
            av_ca_timestep_scale_multiplier: 1000.0,
            cross_attention_adaln: false,
            streaming_prefetch_count: 2,
        }
    }

    fn av_transformer_var_builder() -> VarBuilder<'static> {
        av_transformer_var_builder_with_options(tiny_av_config(), false)
    }

    pub(crate) fn av_transformer_var_builder_with_options(
        config: Ltx2VideoTransformer3DModelConfig,
        nvfp4_outputs: bool,
    ) -> VarBuilder<'static> {
        VarBuilder::from_tensors(
            av_transformer_tensors(config, nvfp4_outputs),
            DType::F32,
            &Device::Cpu,
        )
    }

    /// The raw weight map behind [`av_transformer_var_builder_with_options`].
    /// Exposed so tests can assert which subset of keys a loader touches.
    pub(crate) fn av_transformer_tensors(
        config: Ltx2VideoTransformer3DModelConfig,
        nvfp4_outputs: bool,
    ) -> HashMap<String, Tensor> {
        let video_dim = config.inner_dim();
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;
        let mut tensors = HashMap::new();

        insert_linear(
            &mut tensors,
            "patchify_proj",
            video_dim,
            config.in_channels,
            false,
        );
        insert_adaln_single(&mut tensors, "adaln_single", video_dim, 6);
        insert_text_projection(
            &mut tensors,
            "caption_projection",
            config.caption_channels,
            video_dim,
            video_dim,
        );
        insert_matrix(&mut tensors, "scale_shift_table", 2, video_dim, 11);
        if nvfp4_outputs {
            insert_nvfp4_linear(&mut tensors, "proj_out", config.out_channels, video_dim);
        } else {
            insert_linear(
                &mut tensors,
                "proj_out",
                config.out_channels,
                video_dim,
                false,
            );
        }

        insert_linear(
            &mut tensors,
            "audio_patchify_proj",
            audio_dim,
            config.audio_in_channels,
            false,
        );
        insert_adaln_single(&mut tensors, "audio_adaln_single", audio_dim, 6);
        insert_text_projection(
            &mut tensors,
            "audio_caption_projection",
            config.caption_channels,
            audio_dim,
            audio_dim,
        );
        insert_matrix(&mut tensors, "audio_scale_shift_table", 2, audio_dim, 13);
        if nvfp4_outputs {
            insert_nvfp4_linear(
                &mut tensors,
                "audio_proj_out",
                config.audio_out_channels,
                audio_dim,
            );
        } else {
            insert_linear(
                &mut tensors,
                "audio_proj_out",
                config.audio_out_channels,
                audio_dim,
                false,
            );
        }

        insert_adaln_single(
            &mut tensors,
            "av_ca_video_scale_shift_adaln_single",
            video_dim,
            4,
        );
        insert_adaln_single(
            &mut tensors,
            "av_ca_audio_scale_shift_adaln_single",
            audio_dim,
            4,
        );
        insert_adaln_single(&mut tensors, "av_ca_a2v_gate_adaln_single", video_dim, 1);
        insert_adaln_single(&mut tensors, "av_ca_v2a_gate_adaln_single", audio_dim, 1);

        insert_av_block(&mut tensors, "transformer_blocks.0", &config, false);
        insert_av_block(&mut tensors, "transformer_blocks.1", &config, true);

        tensors
    }

    pub(crate) fn assert_tensors_close(lhs: &Tensor, rhs: &Tensor, tolerance: f32) {
        let diff = lhs
            .to_dtype(DType::F32)
            .unwrap()
            .broadcast_sub(&rhs.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .fold(0.0f32, f32::max);
        assert!(diff <= tolerance, "max diff {diff} exceeds {tolerance}");
    }

    #[test]
    fn text_cross_attention_ignores_video_rope_without_key_rotary_inputs() {
        let attention = LtxAttention::new(
            4,
            1,
            1,
            4,
            0.0,
            true,
            Some(4),
            true,
            "rms_norm",
            LtxRopeType::Interleaved,
            false,
            attention_var_builder(4),
            None,
            "diffusion_model.transformer_blocks.0.attn2",
            None,
        )
        .unwrap();
        let hidden_states = Tensor::new(
            &[[[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let encoder_hidden_states = Tensor::new(
            &[[
                [0.5f32, 1.0, 1.5, 2.0],
                [2.0, 1.5, 1.0, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ]],
            &Device::Cpu,
        )
        .unwrap();
        let cos = Tensor::new(
            &[[[1.0f32, 0.5, -1.0, 0.25], [0.25, -1.0, 0.5, 1.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let sin = Tensor::new(
            &[[
                [0.0f32, 0.8660254, 0.0, -0.9689124],
                [0.9689124, 0.0, -0.8660254, 0.0],
            ]],
            &Device::Cpu,
        )
        .unwrap();

        let baseline = attention
            .forward(
                &hidden_states,
                Some(&encoder_hidden_states),
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();
        let with_video_rope = attention
            .forward(
                &hidden_states,
                Some(&encoder_hidden_states),
                None,
                Some((&cos, &sin)),
                None,
                None,
                false,
            )
            .unwrap();

        assert_eq!(
            baseline.to_vec3::<f32>().unwrap(),
            with_video_rope.to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn text_attention_zero_init_gates_preserve_output() {
        let hidden_states = Tensor::new(
            &[[[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let ungated = LtxAttention::new(
            4,
            1,
            1,
            4,
            0.0,
            true,
            None,
            true,
            "rms_norm",
            LtxRopeType::Interleaved,
            false,
            attention_var_builder(4),
            None,
            "diffusion_model.transformer_blocks.0.attn1",
            None,
        )
        .unwrap();
        let gated = LtxAttention::new(
            4,
            1,
            1,
            4,
            0.0,
            true,
            None,
            true,
            "rms_norm",
            LtxRopeType::Interleaved,
            true,
            attention_var_builder_with_gate(4, Some(0.0)),
            None,
            "diffusion_model.transformer_blocks.0.attn1",
            None,
        )
        .unwrap();

        let ungated_out = ungated
            .forward(&hidden_states, None, None, None, None, None, false)
            .unwrap();
        let gated_out = gated
            .forward(&hidden_states, None, None, None, None, None, false)
            .unwrap();
        assert_tensors_close(&ungated_out, &gated_out, 1e-5);
    }

    #[test]
    fn self_attention_partial_perturbation_mask_broadcasts_across_heads() {
        let hidden_states = Tensor::new(
            &[
                [[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                [[0.5f32, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let attention = LtxAttention::new(
            4,
            1,
            1,
            4,
            0.0,
            true,
            None,
            true,
            "rms_norm",
            LtxRopeType::Interleaved,
            false,
            attention_var_builder(4),
            None,
            "diffusion_model.transformer_blocks.0.attn1",
            None,
        )
        .unwrap();
        let perturbation_mask = Tensor::new(&[[[1.0f32]], [[0.0f32]]], &Device::Cpu).unwrap();

        let baseline = attention
            .forward(&hidden_states, None, None, None, None, None, false)
            .unwrap();
        let passthrough = attention
            .forward(&hidden_states, None, None, None, None, None, true)
            .unwrap();
        let blended = attention
            .forward(
                &hidden_states,
                None,
                None,
                None,
                None,
                Some(&perturbation_mask),
                false,
            )
            .unwrap();

        assert_tensors_close(
            &blended.narrow(0, 0, 1).unwrap(),
            &baseline.narrow(0, 0, 1).unwrap(),
            1e-5,
        );
        assert_tensors_close(
            &blended.narrow(0, 1, 1).unwrap(),
            &passthrough.narrow(0, 1, 1).unwrap(),
            1e-5,
        );
    }

    #[test]
    fn modulate_and_gate_tokens_cast_to_input_dtype() {
        let x = Tensor::new(&[[[1.0f32, 2.0], [3.0, 4.0]]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let scale = Tensor::new(&[[[0.5f32, -0.25]]], &Device::Cpu).unwrap();
        let shift = Tensor::new(&[[[1.0f32, -1.0]]], &Device::Cpu).unwrap();
        let gate = Tensor::new(&[[[0.25f32, 0.5]]], &Device::Cpu).unwrap();

        let modulated = modulate_tokens(&x, &scale, &shift).unwrap();
        let gated = gate_tokens(&modulated, &gate).unwrap();

        assert_eq!(modulated.dtype(), DType::BF16);
        assert_eq!(gated.dtype(), DType::BF16);
    }

    #[test]
    fn prepare_self_attention_mask_matches_upstream_log_bias_semantics() {
        let raw = Tensor::new(
            &[[[1.0f32, 0.5, 0.0], [0.25, 1.0, 0.125], [0.0, 0.0, 1.0]]],
            &Device::Cpu,
        )
        .unwrap();

        let mask = Ltx2AvTransformer3DModel::prepare_self_attention_mask(Some(&raw), DType::BF16)
            .unwrap()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let values = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(mask.dims4().unwrap(), (1, 1, 3, 3));
        assert!((values[0] - 0.0).abs() < 1e-6);
        assert!((values[1] - 0.5f32.ln()).abs() < 1e-2);
        assert!(values[2].is_infinite() && values[2].is_sign_negative());
        assert!((values[3] - 0.25f32.ln()).abs() < 1e-2);
        assert!((values[5] - 0.125f32.ln()).abs() < 1e-2);
        assert!(values[6].is_infinite() && values[6].is_sign_negative());
        assert!(values[7].is_infinite() && values[7].is_sign_negative());
        assert!((values[8] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_no_params_matches_f32_reference_for_bf16_inputs() {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(
            vec![
                -3.5f32, 0.25, 1.5, 7.0, 2.5, -1.25, 0.0, 4.5, 8.0, -2.0, 1.0, -6.5,
            ],
            (1, 3, 4),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let norm = LayerNormNoParams::new(1e-6);

        let actual = norm.forward(&xs).unwrap().to_dtype(DType::F32).unwrap();

        let xs_f32 = xs.to_dtype(DType::F32).unwrap();
        let last_dim = xs_f32.dim(candle_core::D::Minus1).unwrap();
        let mean =
            (xs_f32.sum_keepdim(candle_core::D::Minus1).unwrap() / (last_dim as f64)).unwrap();
        let centered = xs_f32.broadcast_sub(&mean).unwrap();
        let var = (centered
            .sqr()
            .unwrap()
            .sum_keepdim(candle_core::D::Minus1)
            .unwrap()
            / (last_dim as f64))
            .unwrap();
        let reference = centered
            .broadcast_div(&(var + 1e-6).unwrap().sqrt().unwrap())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let max_diff = actual
            .broadcast_sub(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "max diff {max_diff}");
    }

    #[test]
    fn fp8_linear_upcasts_weights_without_scaled_mm_quantization() {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![0.95f32, -0.41, 0.26, 0.73], (1, 2, 2), &device).unwrap();
        let weight = Tensor::from_vec(vec![0.5f32, -0.75, 1.25, 0.25], (2, 2), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let bias = Tensor::new(&[0.1f32, -0.2], &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();

        let linear = LtxLinear::Fp8 {
            weight: weight.clone(),
            weight_scale: None,
            input_scale: None,
            bias: Some(bias.clone()),
            adapters: vec![],
        };

        let out = linear.forward(&xs).unwrap().to_dtype(DType::F32).unwrap();

        let expected_w = weight.to_dtype(DType::F32).unwrap();
        let expected = xs
            .reshape((2, 2))
            .unwrap()
            .matmul(&expected_w.t().unwrap())
            .unwrap()
            .reshape((1, 2, 2))
            .unwrap()
            .broadcast_add(&bias.to_dtype(DType::F32).unwrap())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let actual = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_linear_load_preserves_float8_weights_for_runtime_cast_mode() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![0.5f32, -0.75, 1.25, 0.25], (2, 2), &device)
                .unwrap()
                .to_dtype(DType::F8E4M3)
                .unwrap(),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::new(&[0.1f32, -0.2], &device).unwrap(),
        );
        tensors.insert(
            "weight_scale".to_string(),
            Tensor::new(0.25f32, &device).unwrap(),
        );
        tensors.insert(
            "input_scale".to_string(),
            Tensor::new(0.125f32, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F8E4M3, &device);

        let linear = LtxLinear::load(2, 2, true, vb, vec![]).unwrap();

        match linear {
            LtxLinear::Fp8 {
                weight,
                weight_scale,
                input_scale,
                bias,
                adapters,
            } => {
                assert_eq!(weight.dtype(), DType::F8E4M3);
                assert!(weight_scale.is_some());
                assert!(input_scale.is_some());
                assert!(bias.is_some());
                assert!(adapters.is_empty());
            }
            LtxLinear::Standard { .. } | LtxLinear::Nvfp4Streaming { .. } => {
                panic!("expected fp8 linear")
            }
        }
    }

    #[test]
    fn ltx2_nvfp4_linear_loads_packed_sidecars_and_applies_tensor_scale() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        let mut packed = vec![0x22u8; 8];
        packed.extend(std::iter::repeat_n(0x44u8, 8));
        tensors.insert(
            "weight.nvfp4_packed".to_string(),
            Tensor::from_vec(packed, (2, 8), &device).unwrap(),
        );
        tensors.insert(
            "weight.nvfp4_block_scales".to_string(),
            {
                let swizzled = crate::nvfp4::swizzle_block_scales(&[1.0f32, 1.0], 2, 1).unwrap();
                Tensor::from_vec(swizzled, (128, 4), &device)
            }
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap(),
        );
        tensors.insert(
            "weight.nvfp4_tensor_scale".to_string(),
            Tensor::new(0.5f32, &device).unwrap(),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_vec(vec![0.25f32, -0.5], 2, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let linear = LtxLinear::load(16, 2, true, vb, vec![]).unwrap();
        let xs = Tensor::ones((1, 1, 16), DType::F32, &device).unwrap();
        let out = linear.forward(&xs).unwrap().to_dtype(DType::F32).unwrap();

        let actual = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual.len(), 2);
        assert!((actual[0] - 8.25).abs() < 1e-3, "{actual:?}");
        assert!((actual[1] - 15.5).abs() < 1e-3, "{actual:?}");
    }

    fn nvfp4_linear_var_builder() -> VarBuilder<'static> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        let mut packed = vec![0x22u8; 8];
        packed.extend(std::iter::repeat_n(0x44u8, 8));
        tensors.insert(
            "weight.nvfp4_packed".to_string(),
            Tensor::from_vec(packed, (2, 8), &device).unwrap(),
        );
        tensors.insert(
            "weight.nvfp4_block_scales".to_string(),
            {
                let swizzled = crate::nvfp4::swizzle_block_scales(&[1.0f32, 1.0], 2, 1).unwrap();
                Tensor::from_vec(swizzled, (128, 4), &device)
            }
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap(),
        );
        tensors.insert(
            "weight.nvfp4_tensor_scale".to_string(),
            Tensor::new(1.0f32, &device).unwrap(),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_vec(vec![0.25f32, -0.5], 2, &device).unwrap(),
        );
        VarBuilder::from_tensors(tensors, DType::F32, &device)
    }

    #[test]
    fn ltx2_nvfp4_linear_reuses_shared_cache_entry_for_same_key() {
        let cache = Nvfp4LinearCache::default();
        let first = LtxLinear::load_with_nvfp4_cache(
            16,
            2,
            true,
            nvfp4_linear_var_builder(),
            vec![],
            Some(&cache),
            Some("diffusion_model.transformer_blocks.0.attn1.to_q"),
        )
        .unwrap();
        let second = LtxLinear::load_with_nvfp4_cache(
            16,
            2,
            true,
            nvfp4_linear_var_builder(),
            vec![],
            Some(&cache),
            Some("diffusion_model.transformer_blocks.0.attn1.to_q"),
        )
        .unwrap();
        let third = LtxLinear::load_with_nvfp4_cache(
            16,
            2,
            true,
            nvfp4_linear_var_builder(),
            vec![],
            Some(&cache),
            Some("diffusion_model.transformer_blocks.1.attn1.to_q"),
        )
        .unwrap();

        let first_cache = match &first {
            LtxLinear::Nvfp4Streaming { cache, .. } => cache,
            other => panic!("expected NVFP4 linear, got {other:?}"),
        };
        let second_cache = match &second {
            LtxLinear::Nvfp4Streaming { cache, .. } => cache,
            other => panic!("expected NVFP4 linear, got {other:?}"),
        };
        let third_cache = match &third {
            LtxLinear::Nvfp4Streaming { cache, .. } => cache,
            other => panic!("expected NVFP4 linear, got {other:?}"),
        };

        assert!(std::sync::Arc::ptr_eq(first_cache, second_cache));
        assert!(!std::sync::Arc::ptr_eq(first_cache, third_cache));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn av_transformer_loads_nvfp4_top_level_output_projections() {
        let mut config = tiny_av_config();
        config.attention_head_dim = 16;
        config.audio_attention_head_dim = 16;
        let vb = av_transformer_var_builder_with_options(config.clone(), true);

        let model = Ltx2AvTransformer3DModel::new_streaming(&config, vb, None).unwrap();

        match &model.proj_out {
            LtxLinear::Nvfp4Streaming { out_dim, .. } => {
                assert_eq!(*out_dim, config.out_channels);
            }
            other => panic!("expected NVFP4 video proj_out, got {other:?}"),
        }
        match &model.audio_proj_out {
            LtxLinear::Nvfp4Streaming { out_dim, .. } => {
                assert_eq!(*out_dim, config.audio_out_channels);
            }
            other => panic!("expected NVFP4 audio_proj_out, got {other:?}"),
        }
    }

    #[test]
    fn fp8_weight_scale_can_be_skipped_for_debugging() {
        let _env_lock = fp8_weight_scale_env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set("MOLD_LTX2_FP8_WEIGHT_SCALE_MODE", "skip");
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![2.0f32, -4.0], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let weight_scale = Tensor::new(0.25f32, &device).unwrap();
        let dequantized = super::dequantize_fp8_weight_for_runtime(
            &weight,
            match super::fp8_weight_scale_mode() {
                super::Fp8WeightScaleMode::Skip => None,
                super::Fp8WeightScaleMode::Apply => Some(&weight_scale),
            },
            DType::F32,
            &device,
        )
        .unwrap();
        let expected = weight.to_dtype(DType::F32).unwrap();

        let actual = dequantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_weight_scale_is_applied_by_default() {
        let _env_lock = fp8_weight_scale_env_lock().lock().unwrap();
        let _guard = EnvVarGuard::unset("MOLD_LTX2_FP8_WEIGHT_SCALE_MODE");
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![2.0f32, -4.0], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let weight_scale = Tensor::new(0.25f32, &device).unwrap();
        let dequantized = super::dequantize_fp8_weight_for_runtime(
            &weight,
            match super::fp8_weight_scale_mode() {
                super::Fp8WeightScaleMode::Skip => None,
                super::Fp8WeightScaleMode::Apply => Some(&weight_scale),
            },
            DType::F32,
            &device,
        )
        .unwrap();
        let expected = weight
            .to_dtype(DType::F32)
            .unwrap()
            .broadcast_mul(&weight_scale)
            .unwrap();

        let actual = dequantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_linear_forward_can_skip_weight_scale_for_debugging() {
        let _env_lock = fp8_weight_scale_env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set("MOLD_LTX2_FP8_WEIGHT_SCALE_MODE", "skip");
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![0.42f32, -0.91, 1.37, -0.18], (1, 2, 2), &device).unwrap();
        let weight = Tensor::from_vec(vec![1.5f32, -0.75, 0.25, 2.0], (2, 2), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let weight_scale = Tensor::new(0.125f32, &device).unwrap();
        let linear = LtxLinear::Fp8 {
            weight: weight.clone(),
            weight_scale: Some(weight_scale),
            input_scale: None,
            bias: None,
            adapters: vec![],
        };

        let out = linear.forward(&xs).unwrap().to_dtype(DType::F32).unwrap();
        let expected = xs.reshape((2, 2)).unwrap();
        let expected = expected
            .matmul(&weight.to_dtype(DType::F32).unwrap().t().unwrap())
            .unwrap()
            .reshape((1, 2, 2))
            .unwrap();

        let actual = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_linear_forward_applies_weight_scale_by_default() {
        let _env_lock = fp8_weight_scale_env_lock().lock().unwrap();
        let _guard = EnvVarGuard::unset("MOLD_LTX2_FP8_WEIGHT_SCALE_MODE");
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![0.42f32, -0.91, 1.37, -0.18], (1, 2, 2), &device).unwrap();
        let weight = Tensor::from_vec(vec![1.5f32, -0.75, 0.25, 2.0], (2, 2), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let weight_scale = Tensor::new(0.125f32, &device).unwrap();
        let linear = LtxLinear::Fp8 {
            weight: weight.clone(),
            weight_scale: Some(weight_scale.clone()),
            input_scale: None,
            bias: None,
            adapters: vec![],
        };

        let out = linear.forward(&xs).unwrap().to_dtype(DType::F32).unwrap();
        let expected_weight = weight
            .to_dtype(DType::F32)
            .unwrap()
            .broadcast_mul(&weight_scale)
            .unwrap();
        let expected = xs.reshape((2, 2)).unwrap();
        let expected = expected
            .matmul(&expected_weight.t().unwrap())
            .unwrap()
            .reshape((1, 2, 2))
            .unwrap();

        let actual = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_linear_forward_chunked_matches_full_dequantized_matmul() {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(patterned_values(24, 29), (1, 2, 3, 4), &device).unwrap();
        let weight = Tensor::from_vec(patterned_values(24, 31), (6, 4), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let weight_scale = Tensor::new(0.25f32, &device).unwrap();

        let full = super::fp8_linear_forward_chunked(
            &xs,
            &weight,
            Some(&weight_scale),
            DType::F32,
            weight.dim(0).unwrap(),
        )
        .unwrap();
        let chunked =
            super::fp8_linear_forward_chunked(&xs, &weight, Some(&weight_scale), DType::F32, 2)
                .unwrap();

        assert_tensors_close(&full, &chunked, 1e-5);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fp8_linear_streams_cast_weights_through_cpu_on_metal() {
        let cpu = Device::Cpu;
        let metal = Device::new_metal(0).unwrap();
        let xs = Tensor::from_vec(vec![0.42f32, -0.91, 1.37, -0.18], (1, 2, 2), &metal)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weight_cpu = Tensor::from_vec(vec![1.5f32, -0.75, 0.25, 2.0], (2, 2), &cpu)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let weight = weight_cpu.to_device(&metal).unwrap();
        let weight_scale = Tensor::new(0.125f32, &metal).unwrap();
        let linear = LtxLinear::Fp8 {
            weight,
            weight_scale: Some(weight_scale),
            input_scale: None,
            bias: None,
            adapters: vec![],
        };

        let actual = linear
            .forward(&xs)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected_weight = weight_cpu
            .to_dtype(DType::F32)
            .unwrap()
            .affine(0.125, 0.0)
            .unwrap();
        let expected = xs
            .to_device(&cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((2, 2))
            .unwrap()
            .matmul(&expected_weight.t().unwrap())
            .unwrap()
            .reshape((1, 2, 2))
            .unwrap();

        assert_tensors_close(&actual, &expected, 2e-2);
    }

    #[test]
    fn standard_linear_applies_lora_adapters_without_merging_base_weight() {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![4.0f32, 5.0], (1, 1, 2), &device).unwrap();
        let weight = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let linear = LtxLinear::Standard {
            linear: Linear::new(weight, None),
            adapters: vec![LinearLoraAdapter {
                a: Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device).unwrap(),
                b: Tensor::from_vec(vec![2.0f32, 3.0], (2, 1), &device).unwrap(),
                scale: 0.5,
            }],
        };

        let out = linear.forward(&xs).unwrap().flatten_all().unwrap();
        assert_eq!(out.to_vec1::<f32>().unwrap(), vec![8.0, 11.0]);
    }

    #[test]
    fn fp8_linear_ignores_input_scale_by_default_in_fp8_cast_mode() {
        let _env_lock = fp8_input_scale_env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set("MOLD_LTX2_FP8_INPUT_SCALE_MODE", "skip");
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![0.42f32, -0.91, 1.37, -0.18], (1, 2, 2), &device).unwrap();
        let weight = Tensor::from_vec(vec![1.5f32, -0.75, 0.25, 2.0], (2, 2), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let input_scale = Tensor::new(0.125f32, &device).unwrap();
        let linear = LtxLinear::Fp8 {
            weight: weight.clone(),
            weight_scale: None,
            input_scale: Some(input_scale.clone()),
            bias: None,
            adapters: vec![],
        };

        let out = linear.forward(&xs).unwrap().to_dtype(DType::F32).unwrap();
        let expected = xs.reshape((2, 2)).unwrap();
        let expected = expected
            .matmul(&weight.to_dtype(DType::F32).unwrap().t().unwrap())
            .unwrap()
            .reshape((1, 2, 2))
            .unwrap();

        let actual = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_linear_can_emulate_input_scale_when_requested() {
        let _env_lock = fp8_input_scale_env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set("MOLD_LTX2_FP8_INPUT_SCALE_MODE", "emulate");
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![0.42f32, -0.91, 1.37, -0.18], (1, 2, 2), &device).unwrap();
        let weight = Tensor::from_vec(vec![1.5f32, -0.75, 0.25, 2.0], (2, 2), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let input_scale = Tensor::new(0.125f32, &device).unwrap();
        let linear = LtxLinear::Fp8 {
            weight: weight.clone(),
            weight_scale: None,
            input_scale: Some(input_scale.clone()),
            bias: None,
            adapters: vec![],
        };

        let out = linear.forward(&xs).unwrap().to_dtype(DType::F32).unwrap();
        let quantized_input = emulate_static_fp8_input_quantization(&xs, &input_scale, DType::F32)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        let expected = quantized_input
            .matmul(&weight.to_dtype(DType::F32).unwrap().t().unwrap())
            .unwrap()
            .reshape((1, 2, 2))
            .unwrap();

        let actual = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_weight_dequantization_can_apply_weight_scale_once_at_load_time() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![2.0f32, -4.0], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let weight_scale = Tensor::new(0.25f32, &device).unwrap();

        let dequantized = super::dequantize_fp8_weight_for_runtime(
            &weight,
            Some(&weight_scale),
            DType::F32,
            &device,
        )
        .unwrap();
        let expected = weight
            .to_dtype(DType::F32)
            .unwrap()
            .broadcast_mul(&weight_scale)
            .unwrap();

        assert_eq!(dequantized.dtype(), DType::F32);
        let actual = dequantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }

    #[test]
    fn streaming_layer_sync_uses_prefetch_interval_and_final_layer() {
        assert!(!super::should_synchronize_streaming_layer(0, 6, 2));
        assert!(super::should_synchronize_streaming_layer(1, 6, 2));
        assert!(!super::should_synchronize_streaming_layer(2, 6, 2));
        assert!(super::should_synchronize_streaming_layer(3, 6, 2));
        assert!(!super::should_synchronize_streaming_layer(4, 6, 2));
        assert!(super::should_synchronize_streaming_layer(5, 6, 2));
    }

    #[test]
    fn streaming_layer_sync_treats_zero_prefetch_as_every_layer() {
        assert!(super::should_synchronize_streaming_layer(0, 3, 0));
        assert!(super::should_synchronize_streaming_layer(1, 3, 0));
        assert!(super::should_synchronize_streaming_layer(2, 3, 0));
    }

    #[test]
    fn chunked_attention_matches_full_attention_without_mask() {
        let device = Device::Cpu;
        let q = Tensor::from_vec(patterned_values(40, 7), (1, 2, 5, 4), &device).unwrap();
        let k = Tensor::from_vec(patterned_values(40, 11), (1, 2, 5, 4), &device).unwrap();
        let v = Tensor::from_vec(patterned_values(40, 13), (1, 2, 5, 4), &device).unwrap();
        let scale = 1f32 / 2f32.sqrt();

        let full = super::full_attention(&q, &k, &v, None, scale).unwrap();
        let chunked = super::chunked_attention(&q, &k, &v, None, scale, 2, 3).unwrap();

        assert_tensors_close(&full, &chunked, 1e-5);
    }

    #[test]
    fn attention_chunk_policy_accounts_for_batch_and_heads() {
        let device = Device::Cpu;

        assert!(!super::should_chunk_attention(
            1,
            1,
            512,
            512,
            DType::F32,
            &device
        ));
        assert!(super::should_chunk_attention(
            8,
            8,
            512,
            512,
            DType::F32,
            &device
        ));
    }

    #[test]
    fn attention_chunk_policy_accounts_for_dtype_width() {
        let device = Device::Cpu;

        assert!(super::should_chunk_attention(
            4,
            4,
            600,
            500,
            DType::F32,
            &device
        ));
        assert!(!super::should_chunk_attention(
            4,
            4,
            600,
            500,
            DType::BF16,
            &device
        ));
    }

    #[test]
    fn attention_work_bytes_uses_saturating_math() {
        assert_eq!(
            super::attention_work_bytes(2, 3, 5, 7, DType::F32),
            2 * 3 * 5 * 7 * 4
        );
        assert_eq!(
            super::attention_work_bytes(usize::MAX, usize::MAX, usize::MAX, usize::MAX, DType::F32),
            u64::MAX
        );
    }

    #[test]
    fn chunked_attention_matches_full_attention_with_mask() {
        let device = Device::Cpu;
        let q = Tensor::from_vec(patterned_values(40, 17), (1, 2, 5, 4), &device).unwrap();
        let k = Tensor::from_vec(patterned_values(40, 19), (1, 2, 5, 4), &device).unwrap();
        let v = Tensor::from_vec(patterned_values(40, 23), (1, 2, 5, 4), &device).unwrap();
        let mut mask_values = vec![0.0f32; 2 * 5 * 5];
        for head in 0..2 {
            let base = head * 25;
            mask_values[base + 3] = f32::NEG_INFINITY;
            mask_values[base + 4] = f32::NEG_INFINITY;
        }
        let mask = Tensor::from_vec(mask_values, (1, 2, 5, 5), &device).unwrap();
        let scale = 1f32 / 2f32.sqrt();

        let full = super::full_attention(&q, &k, &v, Some(&mask), scale).unwrap();
        let chunked = super::chunked_attention(&q, &k, &v, Some(&mask), scale, 2, 3).unwrap();

        assert_tensors_close(&full, &chunked, 1e-5);
    }

    #[test]
    fn chunked_attention_matches_full_attention_with_bf16_inputs() {
        let device = Device::Cpu;
        let q = Tensor::from_vec(patterned_values(40, 29), (1, 2, 5, 4), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::from_vec(patterned_values(40, 31), (1, 2, 5, 4), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::from_vec(patterned_values(40, 37), (1, 2, 5, 4), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let scale = 1f32 / 2f32.sqrt();

        let full = super::full_attention(
            &q.to_dtype(DType::F32).unwrap(),
            &k.to_dtype(DType::F32).unwrap(),
            &v.to_dtype(DType::F32).unwrap(),
            None,
            scale,
        )
        .unwrap();
        let chunked = super::chunked_attention(&q, &k, &v, None, scale, 2, 3).unwrap();

        assert_tensors_close(&full, &chunked, 1e-3);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_fused_attention_matches_f32_reference_with_mask() {
        let device = Device::new_metal(0).unwrap();
        let shape = (1, 2, 5, 64);
        let q = Tensor::from_vec(patterned_values(640, 41), shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::from_vec(patterned_values(640, 43), shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::from_vec(patterned_values(640, 47), shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let mut mask_values = vec![0.0f32; 2 * 5 * 5];
        for head in 0..2 {
            mask_values[head * 25 + 4] = -10_000.0;
        }
        let mask = Tensor::from_vec(mask_values, (1, 2, 5, 5), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let scale = 1f32 / 8f32.sqrt();

        let fused = super::metal_fused_attention(&q, &k, &v, Some(&mask), scale).unwrap();
        let reference = super::full_attention(
            &q.to_dtype(DType::F32).unwrap(),
            &k.to_dtype(DType::F32).unwrap(),
            &v.to_dtype(DType::F32).unwrap(),
            Some(&mask.to_dtype(DType::F32).unwrap()),
            scale,
        )
        .unwrap();

        assert_eq!(fused.dtype(), DType::BF16);
        assert_tensors_close(&fused.to_dtype(DType::F32).unwrap(), &reference, 2e-2);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_fused_attention_matches_reference_at_production_cross_attention_shape() {
        // The toy-shape test above passes while renders on Metal ignore the
        // prompt; the production text cross-attention shape (long rectangular
        // q x k, head_dim 128) selects a different candle SDPA kernel, so
        // parity must be proven at that shape. Production passes
        // `context_mask=None` (the connector zero-packs padding), and candle's
        // Metal SDPA refuses a masked q_seq > k_seq call outright, so the
        // production-relevant case is the unmasked rectangular one.
        let device = Device::new_metal(0).unwrap();
        let (b, h, q_len, k_len, head_dim) = (1usize, 32usize, 3120usize, 256usize, 128usize);
        let q = Tensor::from_vec(
            patterned_values(b * h * q_len * head_dim, 41),
            (b, h, q_len, head_dim),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let k = Tensor::from_vec(
            patterned_values(b * h * k_len * head_dim, 43),
            (b, h, k_len, head_dim),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let v = Tensor::from_vec(
            patterned_values(b * h * k_len * head_dim, 47),
            (b, h, k_len, head_dim),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let scale = 1f32 / (head_dim as f32).sqrt();

        let fused = super::metal_fused_attention(&q, &k, &v, None, scale).unwrap();
        let reference = super::full_attention(
            &q.to_dtype(DType::F32).unwrap(),
            &k.to_dtype(DType::F32).unwrap(),
            &v.to_dtype(DType::F32).unwrap(),
            None,
            scale,
        )
        .unwrap();

        assert_tensors_close(&fused.to_dtype(DType::F32).unwrap(), &reference, 2e-2);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_fused_attention_matches_reference_at_production_self_attention_shape() {
        // Long unmasked self-attention at the stage-1 latent token count.
        let device = Device::new_metal(0).unwrap();
        let (b, h, q_len, head_dim) = (1usize, 32usize, 3120usize, 128usize);
        let q = Tensor::from_vec(
            patterned_values(b * h * q_len * head_dim, 41),
            (b, h, q_len, head_dim),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let k = Tensor::from_vec(
            patterned_values(b * h * q_len * head_dim, 43),
            (b, h, q_len, head_dim),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let v = Tensor::from_vec(
            patterned_values(b * h * q_len * head_dim, 47),
            (b, h, q_len, head_dim),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let scale = 1f32 / (head_dim as f32).sqrt();

        let fused = super::metal_fused_attention(&q, &k, &v, None, scale).unwrap();
        let reference = super::full_attention(
            &q.to_dtype(DType::F32).unwrap(),
            &k.to_dtype(DType::F32).unwrap(),
            &v.to_dtype(DType::F32).unwrap(),
            None,
            scale,
        )
        .unwrap();

        assert_tensors_close(&fused.to_dtype(DType::F32).unwrap(), &reference, 2e-2);
    }

    #[test]
    fn av_transformer_streaming_matches_eager_with_mixed_fp8_blocks() {
        let device = Device::Cpu;
        let config = tiny_av_config();
        let vb = av_transformer_var_builder();
        let eager = Ltx2AvTransformer3DModel::new(&config, vb.clone(), None).unwrap();
        let streaming = Ltx2AvTransformer3DModel::new_streaming(&config, vb, None).unwrap();
        let adaptive_plan = crate::adaptive_offload::AdaptiveResidencyPlan {
            resident: vec![true, false],
            resident_bytes: 10,
            streamed_bytes: 20,
            largest_streamed_block: 20,
            activation_budget: 0,
            runtime_headroom: 0,
            fixed_resident_bytes: 0,
        };
        let adaptive = Ltx2AvTransformer3DModel::new_adaptive(
            &config,
            av_transformer_var_builder(),
            None,
            adaptive_plan,
        )
        .unwrap();

        let video_hidden_states = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6],
            (1, 3, config.in_channels),
            &device,
        )
        .unwrap();
        let audio_hidden_states = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.5, -0.4],
            (1, 2, config.audio_in_channels),
            &device,
        )
        .unwrap();
        let video_encoder_hidden_states = Tensor::from_vec(
            patterned_values(16, 3),
            (1, 4, config.caption_channels),
            &device,
        )
        .unwrap();
        let audio_encoder_hidden_states = Tensor::from_vec(
            patterned_values(16, 9),
            (1, 4, config.caption_channels),
            &device,
        )
        .unwrap();
        let timestep = Tensor::new(&[0.75f32], &device).unwrap();
        let video_attention_mask = Tensor::new(&[[1u8, 1, 0, 0]], &device).unwrap();
        let audio_attention_mask = Tensor::new(&[[1u8, 0, 1, 0]], &device).unwrap();
        let video_positions = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0,
                2.0, 3.0,
            ],
            (1, 3, 3, 2),
            &device,
        )
        .unwrap();
        let audio_positions =
            Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 2.0], (1, 1, 2, 2), &device).unwrap();

        let (eager_video, eager_audio) = eager
            .forward(
                &video_hidden_states,
                Some(&audio_hidden_states),
                &video_encoder_hidden_states,
                Some(&audio_encoder_hidden_states),
                &timestep,
                &timestep,
                Some(&timestep),
                Some(&timestep),
                Some(&video_attention_mask),
                Some(&audio_attention_mask),
                None,
                None,
                &video_positions,
                Some(&audio_positions),
                None,
            )
            .unwrap();
        let (streaming_video, streaming_audio) = streaming
            .forward(
                &video_hidden_states,
                Some(&audio_hidden_states),
                &video_encoder_hidden_states,
                Some(&audio_encoder_hidden_states),
                &timestep,
                &timestep,
                Some(&timestep),
                Some(&timestep),
                Some(&video_attention_mask),
                Some(&audio_attention_mask),
                None,
                None,
                &video_positions,
                Some(&audio_positions),
                None,
            )
            .unwrap();
        let (adaptive_video, adaptive_audio) = adaptive
            .forward(
                &video_hidden_states,
                Some(&audio_hidden_states),
                &video_encoder_hidden_states,
                Some(&audio_encoder_hidden_states),
                &timestep,
                &timestep,
                Some(&timestep),
                Some(&timestep),
                Some(&video_attention_mask),
                Some(&audio_attention_mask),
                None,
                None,
                &video_positions,
                Some(&audio_positions),
                None,
            )
            .unwrap();

        assert_tensors_close(&eager_video, &streaming_video, 1e-4);
        let eager_audio = eager_audio.unwrap();
        assert_tensors_close(&eager_audio, &streaming_audio.unwrap(), 1e-4);
        assert_tensors_close(&eager_video, &adaptive_video, 1e-4);
        assert_tensors_close(&eager_audio, &adaptive_audio.unwrap(), 1e-4);
        assert_eq!(adaptive.adaptive_residency_counts(), Some((1, 1)));
    }

    #[test]
    fn av_transformer_forward_with_static_inputs_matches_full_forward() {
        let device = Device::Cpu;
        let config = tiny_av_config();
        let model =
            Ltx2AvTransformer3DModel::new(&config, av_transformer_var_builder(), None).unwrap();

        let video_hidden_states = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6],
            (1, 3, config.in_channels),
            &device,
        )
        .unwrap();
        let audio_hidden_states = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.5, -0.4],
            (1, 2, config.audio_in_channels),
            &device,
        )
        .unwrap();
        let video_encoder_hidden_states = Tensor::from_vec(
            patterned_values(16, 3),
            (1, 4, config.caption_channels),
            &device,
        )
        .unwrap();
        let audio_encoder_hidden_states = Tensor::from_vec(
            patterned_values(16, 9),
            (1, 4, config.caption_channels),
            &device,
        )
        .unwrap();
        let timestep = Tensor::new(&[0.75f32], &device).unwrap();
        let video_attention_mask = Tensor::new(&[[1u8, 1, 0, 0]], &device).unwrap();
        let audio_attention_mask = Tensor::new(&[[1u8, 0, 1, 0]], &device).unwrap();
        let video_positions = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0,
                2.0, 3.0,
            ],
            (1, 3, 3, 2),
            &device,
        )
        .unwrap();
        let audio_positions =
            Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 2.0], (1, 1, 2, 2), &device).unwrap();

        let static_inputs = model
            .prepare_static_inputs(
                &video_encoder_hidden_states,
                Some(&audio_encoder_hidden_states),
                Some(&video_attention_mask),
                Some(&audio_attention_mask),
                None,
                None,
                &video_positions,
                Some(&audio_positions),
            )
            .unwrap();
        let (full_video, full_audio) = model
            .forward(
                &video_hidden_states,
                Some(&audio_hidden_states),
                &video_encoder_hidden_states,
                Some(&audio_encoder_hidden_states),
                &timestep,
                &timestep,
                Some(&timestep),
                Some(&timestep),
                Some(&video_attention_mask),
                Some(&audio_attention_mask),
                None,
                None,
                &video_positions,
                Some(&audio_positions),
                None,
            )
            .unwrap();
        let (static_video, static_audio) = model
            .forward_with_static_inputs(
                &video_hidden_states,
                Some(&audio_hidden_states),
                &timestep,
                &timestep,
                Some(&timestep),
                Some(&timestep),
                &static_inputs,
                None,
            )
            .unwrap();

        assert_tensors_close(&full_video, &static_video, 1e-4);
        let full_audio = full_audio.unwrap();
        let static_audio = static_audio.unwrap();
        assert_eq!(static_video.dims(), video_hidden_states.dims());
        assert_eq!(static_audio.dims(), audio_hidden_states.dims());
        assert_tensors_close(&full_audio, &static_audio, 1e-4);
    }

    #[test]
    fn av_transformer_uniform_tokenwise_timesteps_match_scalar_sigma_path() {
        let device = Device::Cpu;
        let config = tiny_av_config();
        let model =
            Ltx2AvTransformer3DModel::new(&config, av_transformer_var_builder(), None).unwrap();

        let video_hidden_states = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6],
            (1, 3, config.in_channels),
            &device,
        )
        .unwrap();
        let audio_hidden_states = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.5, -0.4],
            (1, 2, config.audio_in_channels),
            &device,
        )
        .unwrap();
        let video_encoder_hidden_states = Tensor::from_vec(
            patterned_values(16, 3),
            (1, 4, config.caption_channels),
            &device,
        )
        .unwrap();
        let audio_encoder_hidden_states = Tensor::from_vec(
            patterned_values(16, 9),
            (1, 4, config.caption_channels),
            &device,
        )
        .unwrap();
        let sigma = Tensor::new(&[0.75f32], &device).unwrap();
        let video_timesteps = Tensor::new(&[[0.75f32, 0.75, 0.75]], &device).unwrap();
        let audio_timesteps = Tensor::new(&[[0.75f32, 0.75]], &device).unwrap();
        let video_attention_mask = Tensor::new(&[[1u8, 1, 0, 0]], &device).unwrap();
        let audio_attention_mask = Tensor::new(&[[1u8, 0, 1, 0]], &device).unwrap();
        let video_positions = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0,
                2.0, 3.0,
            ],
            (1, 3, 3, 2),
            &device,
        )
        .unwrap();
        let audio_positions =
            Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 2.0], (1, 1, 2, 2), &device).unwrap();

        let (scalar_video, scalar_audio) = model
            .forward(
                &video_hidden_states,
                Some(&audio_hidden_states),
                &video_encoder_hidden_states,
                Some(&audio_encoder_hidden_states),
                &sigma,
                &sigma,
                Some(&sigma),
                Some(&sigma),
                Some(&video_attention_mask),
                Some(&audio_attention_mask),
                None,
                None,
                &video_positions,
                Some(&audio_positions),
                None,
            )
            .unwrap();
        let (token_video, token_audio) = model
            .forward(
                &video_hidden_states,
                Some(&audio_hidden_states),
                &video_encoder_hidden_states,
                Some(&audio_encoder_hidden_states),
                &sigma,
                &video_timesteps,
                Some(&sigma),
                Some(&audio_timesteps),
                Some(&video_attention_mask),
                Some(&audio_attention_mask),
                None,
                None,
                &video_positions,
                Some(&audio_positions),
                None,
            )
            .unwrap();

        assert_tensors_close(&scalar_video, &token_video, 1e-4);
        assert_tensors_close(&scalar_audio.unwrap(), &token_audio.unwrap(), 1e-4);
    }

    #[test]
    fn gelu_approximate_matches_tanh_polynomial_reference() {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..512)
            .map(|index| (index as f32 - 256.0) / 64.0)
            .collect();
        let x = Tensor::from_vec(values, (1, 32, 16), &device).unwrap();

        // The hand-rolled expression this helper used to evaluate. Kept as the
        // permanent guard for the fused single-kernel form.
        let x_f32 = x.to_dtype(DType::F32).unwrap();
        let x_cube = x_f32.sqr().unwrap().broadcast_mul(&x_f32).unwrap();
        let inner = x_f32
            .broadcast_add(&x_cube.affine(0.044715, 0.0).unwrap())
            .unwrap();
        let scale = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
        let tanh_out = inner.affine(scale as f64, 0.0).unwrap().tanh().unwrap();
        let expected = x_f32
            .broadcast_mul(&tanh_out.affine(1.0, 1.0).unwrap())
            .unwrap()
            .affine(0.5, 0.0)
            .unwrap();

        let actual = super::gelu_approximate(&x).unwrap();

        assert_eq!(actual.dtype(), DType::F32);
        assert_tensors_close(&expected, &actual, 1e-6);
    }

    #[test]
    fn gelu_approximate_preserves_input_dtype() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(patterned_values(32, 3), (1, 8, 4), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let out = super::gelu_approximate(&x).unwrap();

        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.dims(), &[1, 8, 4]);
    }

    fn feed_forward_var_builder(dim: usize, device: &Device) -> VarBuilder<'static> {
        let hidden = dim * 4;
        let mut tensors = HashMap::new();
        tensors.insert(
            "net.0.proj.weight".to_string(),
            Tensor::from_vec(patterned_values(hidden * dim, 3), (hidden, dim), device).unwrap(),
        );
        tensors.insert(
            "net.0.proj.bias".to_string(),
            Tensor::from_vec(patterned_values(hidden, 5), hidden, device).unwrap(),
        );
        tensors.insert(
            "net.2.weight".to_string(),
            Tensor::from_vec(patterned_values(dim * hidden, 7), (dim, hidden), device).unwrap(),
        );
        tensors.insert(
            "net.2.bias".to_string(),
            Tensor::from_vec(patterned_values(dim, 11), dim, device).unwrap(),
        );
        VarBuilder::from_tensors(tensors, DType::F32, device)
    }

    #[test]
    fn feed_forward_token_chunking_matches_unchunked_output() {
        let device = Device::Cpu;
        let ff =
            FeedForward::new(4, feed_forward_var_builder(4, &device), None, "ff", None).unwrap();
        let xs = Tensor::from_vec(patterned_values(2 * 10 * 4, 13), (2, 10, 4), &device).unwrap();

        let unchunked = ff.forward_token_chunked(&xs, usize::MAX).unwrap();
        let chunked = ff.forward_token_chunked(&xs, 3).unwrap();

        assert_eq!(chunked.dims(), unchunked.dims());
        assert_tensors_close(&unchunked, &chunked, 1e-6);
    }

    #[test]
    fn feed_forward_leaves_small_token_counts_unchunked() {
        let device = Device::Cpu;
        let ff =
            FeedForward::new(4, feed_forward_var_builder(4, &device), None, "ff", None).unwrap();
        let xs = Tensor::from_vec(patterned_values(5 * 4, 17), (1, 5, 4), &device).unwrap();

        let via_forward = ff.forward(&xs).unwrap();
        let unchunked = ff.forward_token_chunked(&xs, usize::MAX).unwrap();

        assert_tensors_close(&via_forward, &unchunked, 0.0);
    }

    #[test]
    fn fp8_linear_forward_chunked_matches_full_for_rank3_and_rank2_inputs() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(patterned_values(24, 31), (6, 4), &device)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let weight_scale = Tensor::new(0.25f32, &device).unwrap();

        for dims in [vec![2usize, 3, 4], vec![5usize, 4]] {
            let len: usize = dims.iter().product();
            let xs = Tensor::from_vec(patterned_values(len, 41), dims.clone(), &device).unwrap();

            let full =
                super::fp8_linear_forward_chunked(&xs, &weight, Some(&weight_scale), DType::F32, 6)
                    .unwrap();
            let chunked =
                super::fp8_linear_forward_chunked(&xs, &weight, Some(&weight_scale), DType::F32, 4)
                    .unwrap();

            assert_eq!(chunked.dims(), full.dims());
            assert_tensors_close(&full, &chunked, 1e-5);
        }
    }

    #[test]
    fn nvfp4_linear_forward_chunked_matches_full_dequantized_matmul() {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(patterned_values(24, 43), (1, 2, 3, 4), &device).unwrap();
        let weight = Tensor::from_vec(patterned_values(24, 47), (6, 4), &device).unwrap();

        let full = super::nvfp4_linear_forward_chunked(&xs, &weight, DType::F32, 6).unwrap();
        let chunked = super::nvfp4_linear_forward_chunked(&xs, &weight, DType::F32, 2).unwrap();

        assert_eq!(chunked.dims(), full.dims());
        assert_tensors_close(&full, &chunked, 1e-5);
    }

    #[test]
    fn ada_component_matches_broadcast_table_reference() {
        let device = Device::Cpu;
        let (batch, tokens, count, dim) = (2usize, 3usize, 6usize, 4usize);
        let table =
            Tensor::from_vec(patterned_values(count * dim, 1), (count, dim), &device).unwrap();
        let timestep = Tensor::from_vec(
            patterned_values(batch * tokens * count * dim, 5),
            (batch, tokens, count * dim),
            &device,
        )
        .unwrap();

        let reference = table
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .broadcast_as((batch, tokens, count, dim))
            .unwrap()
            .broadcast_add(&timestep.reshape((batch, tokens, count, dim)).unwrap())
            .unwrap();

        for index in 0..count {
            let expected = reference.narrow(2, index, 1).unwrap().squeeze(2).unwrap();
            let actual = super::ada_component(&table, &timestep, index).unwrap();

            assert_eq!(actual.dims(), &[batch, tokens, dim]);
            assert_tensors_close(&expected, &actual, 1e-6);
        }
    }

    #[test]
    fn ada_component_rejects_out_of_range_components() {
        let device = Device::Cpu;
        let table = Tensor::from_vec(patterned_values(8, 1), (2, 4), &device).unwrap();
        let timestep = Tensor::from_vec(patterned_values(24, 5), (1, 3, 8), &device).unwrap();

        assert!(super::ada_component(&table, &timestep, 2).is_err());
    }

    #[test]
    fn modulate_tokens_matches_one_plus_scale_reference() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(patterned_values(2 * 5 * 3, 19), (2, 5, 3), &device).unwrap();
        let scale = Tensor::from_vec(patterned_values(2 * 3, 23), (2, 1, 3), &device).unwrap();
        let shift = Tensor::from_vec(patterned_values(2 * 3, 29), (2, 1, 3), &device).unwrap();

        let scale_broadcast = scale.broadcast_as((2usize, 5, 3)).unwrap();
        let expected = x
            .broadcast_mul(
                &Tensor::ones_like(&scale_broadcast)
                    .unwrap()
                    .broadcast_add(&scale_broadcast)
                    .unwrap(),
            )
            .unwrap()
            .broadcast_add(&shift.broadcast_as((2usize, 5, 3)).unwrap())
            .unwrap();

        let actual = modulate_tokens(&x, &scale, &shift).unwrap();

        assert_eq!(actual.dims(), &[2, 5, 3]);
        assert_tensors_close(&expected, &actual, 1e-6);
    }

    #[test]
    fn split_rotary_is_identity_for_unit_cosine() {
        let device = Device::Cpu;
        let (batch, seq, heads, head_dim) = (1usize, 3usize, 2usize, 4usize);
        let x = Tensor::from_vec(
            patterned_values(batch * seq * heads * head_dim, 53),
            (batch, seq, heads * head_dim),
            &device,
        )
        .unwrap();
        let cos = Tensor::ones((batch, heads, seq, head_dim / 2), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((batch, heads, seq, head_dim / 2), DType::F32, &device).unwrap();

        let out = super::apply_split_rotary_emb(&x, &cos, &sin, heads, head_dim).unwrap();

        assert_eq!(out.dims(), &[batch, seq, heads * head_dim]);
        assert_tensors_close(&x, &out, 1e-6);
    }

    #[test]
    fn split_rotary_swaps_halves_for_quarter_turn() {
        let device = Device::Cpu;
        let (batch, seq, heads, head_dim) = (1usize, 2usize, 2usize, 4usize);
        let x = Tensor::from_vec(
            patterned_values(batch * seq * heads * head_dim, 59),
            (batch, seq, heads * head_dim),
            &device,
        )
        .unwrap();
        let cos = Tensor::zeros((batch, heads, seq, head_dim / 2), DType::F32, &device).unwrap();
        let sin = Tensor::ones((batch, heads, seq, head_dim / 2), DType::F32, &device).unwrap();

        let out = super::apply_split_rotary_emb(&x, &cos, &sin, heads, head_dim).unwrap();

        // cos = 0, sin = 1 rotates each head's (first, second) half pair to
        // (-second, first).
        let heads_view = x.reshape((batch, seq, heads, head_dim)).unwrap();
        let first = heads_view.narrow(3, 0, head_dim / 2).unwrap();
        let second = heads_view.narrow(3, head_dim / 2, head_dim / 2).unwrap();
        let expected = Tensor::cat(&[&second.neg().unwrap(), &first], 3)
            .unwrap()
            .reshape((batch, seq, heads * head_dim))
            .unwrap();

        assert_tensors_close(&expected, &out, 1e-6);
    }

    #[test]
    fn attention_query_chunk_size_targets_a_tile_byte_budget() {
        let heads = 32usize;
        let key_chunk = 1_024usize;

        let chunk = super::attention_query_chunk_size(1, heads, 13_312, key_chunk);

        assert_eq!(
            chunk,
            (super::ATTENTION_TILE_TARGET_BYTES / (heads * key_chunk * 4) as u64) as usize
        );
        assert!(
            chunk >= 512,
            "expected a wide query tile at stage-2 shapes, got {chunk}"
        );
    }

    #[test]
    fn attention_query_chunk_size_is_clamped_to_bounds_and_query_length() {
        assert_eq!(super::attention_query_chunk_size(1, 1, 8, 8), 8);
        assert_eq!(
            super::attention_query_chunk_size(64, 64, 65_536, 65_536),
            super::MIN_ATTENTION_QUERY_CHUNK
        );
        assert_eq!(
            super::attention_query_chunk_size(1, 1, 1_000_000, 1),
            super::MAX_ATTENTION_QUERY_CHUNK
        );
    }
}
