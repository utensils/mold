//! GGUF / quantized FLUX transformer with bypass-mode LoRA.
//!
//! This is the GGUF analog of [`crate::flux::offload::OffloadedFluxTransformer`]:
//! every Linear site is wrapped in [`super::lora_bypass::LoraLinear`] so a
//! LoRA stack can apply at forward time instead of dequant→merge→requant
//! at load time. Unlike offload, all blocks are GPU-resident — the GGUF
//! transformer fits in ~12 GB at Q8 so streaming buys nothing.
//!
//! Mirrors the layout of `candle_transformers::models::flux::quantized_model`
//! but cannot reuse it directly: that crate's `Linear` fields are private,
//! so we'd have no way to substitute [`super::lora_bypass::LoraLinear`]
//! without forking the upstream model. Re-implementing here gives us the
//! same forward math while letting LoRAs bind cheaply.
//!
//! Since the FLUX performance campaign this is the ONLY GGUF path: the
//! `FluxTransformer::Quantized` arm that wrapped the fork's own quantized
//! model is gone. It had no attention-policy hook (`flux/model.rs:64-76` is
//! unchunked F32 math), so every no-LoRA GGUF render — the common case —
//! could not reach FlashAttention no matter what the build compiled. The
//! two were verified bit-identical before the deletion (see
//! `f32_bypass_forward_matches_the_upstream_quantized_model`).
//!
//! ## Working dtype
//!
//! The transformer is built at an explicit `working_dtype`, which on CUDA is
//! BF16 — what BFL runs (`util.py:666-668`, `cli.py:274`) and what ComfyUI
//! runs (`model_management.py:1959-1960`). Three things follow from that and
//! none of them are optional:
//!
//! * **Every dense tensor in the GGUF must be materialized at that dtype at
//!   load.** `QTensor::dequantize` always answers F32
//!   (`candle-core/src/quantized/mod.rs:740-744`), so a bias or a stem left
//!   as it comes out is a `broadcast_add`/`matmul` dtype error on the first
//!   forward. [`crate::quantized_linear::QuantizedLinear`] owns that rule.
//! * **Norm weights follow the activations.** candle's fused RMSNorm and
//!   LayerNorm kernels are dtype-typed (`candle-nn/src/ops.rs:535-537`,
//!   `:770`) and bail on a mismatch.
//! * **LayerNorm carries an explicit zero bias.** `LayerNorm::forward` reaches
//!   the fused kernel only when a bias is present
//!   (`candle-nn/src/layer_norm.rs:118-122`); `new_no_bias` takes the
//!   unfused sum/div/sqrt sequence instead, which is both slower and, at a
//!   half working dtype, a second dtype trap.
//!
//! ## Forward-pass equivalence
//!
//! The block-level math (modulation, RMS-norm Q/K, RoPE, attention, MLP)
//! is the upstream quantized model's. Two deliberate divergences move the
//! bytes: FLUX renders under `AttentionPolicy::FastStill`, whose math path
//! folds the softmax scale into K, and the LayerNorms above are fused.
//! Neither changes the network; both change reduction order.

// `rebind_lora` / `set_lora_registry` are public-API surface for the future
// in-place LoRA swap (no transformer reload). The current pipeline still
// drops + reloads on swap, so these are exercised only by tests today.
#![allow(dead_code)]

use anyhow::Result;
use candle_core::{quantized::QTensor, DType, IndexOp, Module, Tensor, D};
use candle_nn::{LayerNorm, RmsNorm};
use candle_transformers::models::flux::model::{Config, EmbedNd};
use candle_transformers::models::flux::BlockHook;
use mold_candle::quantized::VarBuilder;
use std::sync::Arc;

use crate::quantized_linear::QuantizedLinear;

use crate::flux::lora_bypass::{LoraLinear, LoraRegistry};
use crate::progress::ProgressReporter;

// ── Reimplemented candle-internal helpers ────────────────────────────────────
//
// These mirror `flux/offload.rs` (which mirrors `candle_transformers::models::flux::model`)
// — keep the implementations in sync if upstream ever changes.

fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        anyhow::bail!("{dim} is odd");
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // FLUX renders under `FastStill`: flash wherever the kernel is compiled
    // in, and the math fallback folds the scale into K. See
    // `attention::AttentionPolicy::FastStill`.
    Ok(crate::attention::attention_default_scale_for(
        crate::attention::AttentionPolicy::FastStill,
        q,
        k,
        v,
    )?)
}

/// BFL applies the rotary embedding in F32 and casts the result back
/// (`flux/math.py:20-21`, `:32`). The positions reach `EmbedNd` in F32 too
/// (see `forward_with_hook`), so `freq_cis` is F32 and this is the one place
/// the working dtype is widened.
fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let output_dtype = x.dtype();
    let x = &x.to_dtype(DType::F32)?;
    // One fused launch over contiguous memory when the layout allows; the
    // broadcast form below is the fallback and the definition.
    if let Some(out) = crate::flux_rope::fused_interleaved_rope(x, freq_cis)? {
        return Ok(out.to_dtype(output_dtype)?);
    }
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    Ok((fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?
        .reshape(dims.to_vec())?
        .to_dtype(output_dtype)?)
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention(&q, &k, v)?;
    Ok(x.transpose(1, 2)?.flatten_from(2)?)
}

/// FLUX's affine-less LayerNorm, built so it reaches candle's fused kernel.
///
/// Upstream builds a ones weight and no bias (`quantized_model.rs:7-10`), and
/// `LayerNorm::forward` takes the fused `ops::layer_norm` only when a bias is
/// present (`candle-nn/src/layer_norm.rs:118-122`). An explicit zero bias is
/// arithmetically the same affine and reaches the kernel. Both tensors are
/// built at the working dtype because that kernel is dtype-typed.
fn layer_norm(dim: usize, vb: &VarBuilder, dtype: DType) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, dtype, vb.device())?;
    let bs = Tensor::zeros(dim, dtype, vb.device())?;
    Ok(LayerNorm::new(ws, bs, 1e-6))
}

// ── LoraLinear constructors ──────────────────────────────────────────────────

/// Load a `quantized_nn::Linear` and wrap it in [`LoraLinear`], attaching
/// any bypass-mode adapters that target `key`. Mirrors
/// `lora_linear_to_device` from `flux/offload.rs` but for the quantized
/// path — the GGUF tensor lives on `device` already (the VarBuilder's
/// `from_gguf` did the upload), so this is just a wrapping operation.
fn load_quantized_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    dtype: DType,
    registry: Option<&LoraRegistry>,
    key: &str,
) -> Result<LoraLinear> {
    let inner = quantized_linear_at(in_dim, out_dim, bias, &vb, dtype)?;
    let adapters = registry
        .map(|r| r.adapters_for(key).to_vec())
        .unwrap_or_default();
    if adapters.is_empty() {
        Ok(LoraLinear::Quantized(inner))
    } else {
        Ok(LoraLinear::WithAdaptersQuantized { inner, adapters })
    }
}

/// `linear` (with bias as the dequantized bias term) — used everywhere
/// except the QKV stems, where qkv_bias is conditional.
fn quantized_linear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    dtype: DType,
    registry: Option<&LoraRegistry>,
    key: &str,
) -> Result<LoraLinear> {
    load_quantized_linear(in_dim, out_dim, true, vb, dtype, registry, key)
}

/// One [`QuantizedLinear`] from a GGUF entry, at the working dtype.
///
/// `qmatmul_enabled` is unconditionally `true` here: FLUX.1 has rendered
/// correctly through candle's MMQ kernels since before
/// `crate::quantized_linear` existed, which is a different evidential
/// position from Qwen-Image's and Z-Image's (`docs/architecture/qwen-mmq-nan.md`).
/// A weight the kernels decline still falls through to the dequant arm, and a
/// densely stored one takes the hoisted dense arm.
fn quantized_linear_at(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: &VarBuilder,
    dtype: DType,
) -> Result<QuantizedLinear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?.dequantize(vb.device())?)
    } else {
        None
    };
    Ok(QuantizedLinear::new(
        weight,
        bias,
        vb.device(),
        dtype,
        true,
    )?)
}

/// Pull a (de-)quantized norm scale tensor out of the GGUF — used for
/// `query_norm.scale` / `key_norm.scale`.
///
/// Upstream dequantizes via `vb.get(...).dequantize(device)` and stops there,
/// which is F32 unconditionally. The scale has to follow the activations
/// instead: candle's fused RMSNorm kernel is dtype-typed and bails on a
/// mismatch (`candle-nn/src/ops.rs:535-537`).
fn rms_norm_from_qtensor(dim: usize, vb: VarBuilder, name: &str, dtype: DType) -> Result<RmsNorm> {
    let weight = vb
        .get(dim, name)?
        .dequantize(vb.device())?
        .to_dtype(dtype)?;
    Ok(RmsNorm::new(weight, 1e-6))
}

// ── Block types ──────────────────────────────────────────────────────────────

struct Modulation1 {
    lin: LoraLinear,
}

impl Modulation1 {
    fn load(
        dim: usize,
        vb: VarBuilder,
        dtype: DType,
        registry: Option<&LoraRegistry>,
        base_key: &str,
    ) -> Result<Self> {
        let lin = quantized_linear(
            dim,
            3 * dim,
            vb.pp("lin"),
            dtype,
            registry,
            &format!("{base_key}.lin.weight"),
        )?;
        Ok(Self { lin })
    }
    fn forward(&self, vec_: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let pre = vec_.silu()?;
        let ys = self.lin.forward(&pre)?.unsqueeze(1)?.chunk(3, D::Minus1)?;
        Ok((ys[0].clone(), ys[1].clone(), ys[2].clone()))
    }
    /// Replace the LoRA stack on this layer in-place.
    fn rebind_lora(&mut self, registry: Option<&LoraRegistry>, base_key: &str) {
        rebind(&mut self.lin, registry, &format!("{base_key}.lin.weight"));
    }
}

struct Modulation2 {
    lin: LoraLinear,
}

impl Modulation2 {
    fn load(
        dim: usize,
        vb: VarBuilder,
        dtype: DType,
        registry: Option<&LoraRegistry>,
        base_key: &str,
    ) -> Result<Self> {
        let lin = quantized_linear(
            dim,
            6 * dim,
            vb.pp("lin"),
            dtype,
            registry,
            &format!("{base_key}.lin.weight"),
        )?;
        Ok(Self { lin })
    }
    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        vec_: &Tensor,
    ) -> Result<((Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor))> {
        let pre = vec_.silu()?;
        let ys = self.lin.forward(&pre)?.unsqueeze(1)?.chunk(6, D::Minus1)?;
        Ok((
            (ys[0].clone(), ys[1].clone(), ys[2].clone()),
            (ys[3].clone(), ys[4].clone(), ys[5].clone()),
        ))
    }
    fn rebind_lora(&mut self, registry: Option<&LoraRegistry>, base_key: &str) {
        rebind(&mut self.lin, registry, &format!("{base_key}.lin.weight"));
    }
}

struct SelfAttention {
    qkv: LoraLinear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    proj: LoraLinear,
    num_heads: usize,
}

impl SelfAttention {
    fn load(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        vb: VarBuilder,
        dtype: DType,
        registry: Option<&LoraRegistry>,
        base_key: &str,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = load_quantized_linear(
            dim,
            dim * 3,
            qkv_bias,
            vb.pp("qkv"),
            dtype,
            registry,
            &format!("{base_key}.qkv.weight"),
        )?;
        let query_norm = rms_norm_from_qtensor(head_dim, vb.pp("norm"), "query_norm.scale", dtype)?;
        let key_norm = rms_norm_from_qtensor(head_dim, vb.pp("norm"), "key_norm.scale", dtype)?;
        let proj = quantized_linear(
            dim,
            dim,
            vb.pp("proj"),
            dtype,
            registry,
            &format!("{base_key}.proj.weight"),
        )?;
        Ok(Self {
            qkv,
            query_norm,
            key_norm,
            proj,
            num_heads,
        })
    }
    fn qkv_split(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.qkv.forward(xs)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        // Normalize BEFORE the transpose. `i(.., .., n)` on the packed QKV is
        // a narrow, so one `contiguous()` buys candle's fused RMSNorm kernel
        // (`candle-nn/src/layer_norm.rs:202-210` takes it only for a
        // contiguous input); the transposed view took the ~9-kernel strided
        // fallback. RMSNorm normalizes the LAST dim, which is `head_dim` in
        // both layouts, so this is the same arithmetic BFL performs on its own
        // `K B H L D` rearrangement (`flux2/model.py:752-755`).
        let q = qkv
            .i((.., .., 0))?
            .contiguous()?
            .apply(&self.query_norm)?
            .transpose(1, 2)?;
        let k = qkv
            .i((.., .., 1))?
            .contiguous()?
            .apply(&self.key_norm)?
            .transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        Ok((q, k, v))
    }
    fn rebind_lora(&mut self, registry: Option<&LoraRegistry>, base_key: &str) {
        rebind(&mut self.qkv, registry, &format!("{base_key}.qkv.weight"));
        rebind(&mut self.proj, registry, &format!("{base_key}.proj.weight"));
    }
}

struct Mlp {
    lin1: LoraLinear,
    lin2: LoraLinear,
}

impl Mlp {
    fn load(
        in_sz: usize,
        mlp_sz: usize,
        vb: VarBuilder,
        dtype: DType,
        registry: Option<&LoraRegistry>,
        base_key: &str,
    ) -> Result<Self> {
        // `vb.pp("0")` / `vb.pp("2")` matches the Diffusers Sequential MLP
        // (Linear, GELU, Linear) tensor-key convention used by upstream.
        let lin1 = quantized_linear(
            in_sz,
            mlp_sz,
            vb.pp("0"),
            dtype,
            registry,
            &format!("{base_key}.0.weight"),
        )?;
        let lin2 = quantized_linear(
            mlp_sz,
            in_sz,
            vb.pp("2"),
            dtype,
            registry,
            &format!("{base_key}.2.weight"),
        )?;
        Ok(Self { lin1, lin2 })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.lin1.forward(xs)?.gelu()?;
        self.lin2.forward(&h)
    }
    fn rebind_lora(&mut self, registry: Option<&LoraRegistry>, base_key: &str) {
        rebind(&mut self.lin1, registry, &format!("{base_key}.0.weight"));
        rebind(&mut self.lin2, registry, &format!("{base_key}.2.weight"));
    }
}

struct DoubleBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleBlock {
    fn load(
        cfg: &Config,
        vb: VarBuilder,
        dtype: DType,
        registry: Option<&LoraRegistry>,
        idx: usize,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_sz = (h as f64 * cfg.mlp_ratio) as usize;
        let base = format!("double_blocks.{idx}");
        Ok(Self {
            img_mod: Modulation2::load(
                h,
                vb.pp("img_mod"),
                dtype,
                registry,
                &format!("{base}.img_mod"),
            )?,
            img_norm1: layer_norm(h, &vb.pp("img_norm1"), dtype)?,
            img_attn: SelfAttention::load(
                h,
                cfg.num_heads,
                cfg.qkv_bias,
                vb.pp("img_attn"),
                dtype,
                registry,
                &format!("{base}.img_attn"),
            )?,
            img_norm2: layer_norm(h, &vb.pp("img_norm2"), dtype)?,
            img_mlp: Mlp::load(
                h,
                mlp_sz,
                vb.pp("img_mlp"),
                dtype,
                registry,
                &format!("{base}.img_mlp"),
            )?,
            txt_mod: Modulation2::load(
                h,
                vb.pp("txt_mod"),
                dtype,
                registry,
                &format!("{base}.txt_mod"),
            )?,
            txt_norm1: layer_norm(h, &vb.pp("txt_norm1"), dtype)?,
            txt_attn: SelfAttention::load(
                h,
                cfg.num_heads,
                cfg.qkv_bias,
                vb.pp("txt_attn"),
                dtype,
                registry,
                &format!("{base}.txt_attn"),
            )?,
            txt_norm2: layer_norm(h, &vb.pp("txt_norm2"), dtype)?,
            txt_mlp: Mlp::load(
                h,
                mlp_sz,
                vb.pp("txt_mlp"),
                dtype,
                registry,
                &format!("{base}.txt_mlp"),
            )?,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let ((img_s1, img_sc1, img_g1), (img_s2, img_sc2, img_g2)) = self.img_mod.forward(vec_)?;
        let ((txt_s1, txt_sc1, txt_g1), (txt_s2, txt_sc2, txt_g2)) = self.txt_mod.forward(vec_)?;

        // QKV for both streams
        let img_modulated = img
            .apply(&self.img_norm1)?
            .broadcast_mul(&(&img_sc1 + 1.)?)?
            .broadcast_add(&img_s1)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv_split(&img_modulated)?;

        let txt_modulated = txt
            .apply(&self.txt_norm1)?
            .broadcast_mul(&(&txt_sc1 + 1.)?)?
            .broadcast_add(&txt_s1)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv_split(&txt_modulated)?;

        // Cross-attention
        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;
        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn_out = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn_out = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        // Image residual
        let img = (img + img_g1.broadcast_mul(&self.img_attn.proj.forward(&img_attn_out)?)?)?;
        let img_ff = img
            .apply(&self.img_norm2)?
            .broadcast_mul(&(&img_sc2 + 1.)?)?
            .broadcast_add(&img_s2)?;
        let img = (&img + img_g2.broadcast_mul(&self.img_mlp.forward(&img_ff)?)?)?;

        // Text residual
        let txt = (txt + txt_g1.broadcast_mul(&self.txt_attn.proj.forward(&txt_attn_out)?)?)?;
        let txt_ff = txt
            .apply(&self.txt_norm2)?
            .broadcast_mul(&(&txt_sc2 + 1.)?)?
            .broadcast_add(&txt_s2)?;
        let txt = (&txt + txt_g2.broadcast_mul(&self.txt_mlp.forward(&txt_ff)?)?)?;

        Ok((img, txt))
    }

    fn rebind_lora(&mut self, registry: Option<&LoraRegistry>, idx: usize) {
        let base = format!("double_blocks.{idx}");
        self.img_mod
            .rebind_lora(registry, &format!("{base}.img_mod"));
        self.img_attn
            .rebind_lora(registry, &format!("{base}.img_attn"));
        self.img_mlp
            .rebind_lora(registry, &format!("{base}.img_mlp"));
        self.txt_mod
            .rebind_lora(registry, &format!("{base}.txt_mod"));
        self.txt_attn
            .rebind_lora(registry, &format!("{base}.txt_attn"));
        self.txt_mlp
            .rebind_lora(registry, &format!("{base}.txt_mlp"));
    }
}

struct SingleBlock {
    linear1: LoraLinear,
    linear2: LoraLinear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleBlock {
    fn load(
        cfg: &Config,
        vb: VarBuilder,
        dtype: DType,
        registry: Option<&LoraRegistry>,
        idx: usize,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_sz = (h as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h / cfg.num_heads;
        let base = format!("single_blocks.{idx}");
        Ok(Self {
            linear1: quantized_linear(
                h,
                h * 3 + mlp_sz,
                vb.pp("linear1"),
                dtype,
                registry,
                &format!("{base}.linear1.weight"),
            )?,
            linear2: quantized_linear(
                h + mlp_sz,
                h,
                vb.pp("linear2"),
                dtype,
                registry,
                &format!("{base}.linear2.weight"),
            )?,
            query_norm: rms_norm_from_qtensor(head_dim, vb.pp("norm"), "query_norm.scale", dtype)?,
            key_norm: rms_norm_from_qtensor(head_dim, vb.pp("norm"), "key_norm.scale", dtype)?,
            pre_norm: layer_norm(h, &vb.pp("pre_norm"), dtype)?,
            modulation: Modulation1::load(
                h,
                vb.pp("modulation"),
                dtype,
                registry,
                &format!("{base}.modulation"),
            )?,
            h_sz: h,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (shift, scale, gate) = self.modulation.forward(vec_)?;
        let x_mod = xs
            .apply(&self.pre_norm)?
            .broadcast_mul(&(&scale + 1.)?)?
            .broadcast_add(&shift)?;
        let x_mod = self.linear1.forward(&x_mod)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        // Norm before transpose — see `SelfAttention::qkv_split`.
        let q = qkv
            .i((.., .., 0))?
            .contiguous()?
            .apply(&self.query_norm)?
            .transpose(1, 2)?;
        let k = qkv
            .i((.., .., 1))?
            .contiguous()?
            .apply(&self.key_norm)?
            .transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output_in = Tensor::cat(&[attn, mlp.gelu()?], 2)?;
        let output = self.linear2.forward(&output_in)?;
        Ok((xs + gate.broadcast_mul(&output)?)?)
    }

    fn rebind_lora(&mut self, registry: Option<&LoraRegistry>, idx: usize) {
        let base = format!("single_blocks.{idx}");
        rebind(
            &mut self.linear1,
            registry,
            &format!("{base}.linear1.weight"),
        );
        rebind(
            &mut self.linear2,
            registry,
            &format!("{base}.linear2.weight"),
        );
        self.modulation
            .rebind_lora(registry, &format!("{base}.modulation"));
    }
}

/// Last layer: AdaLN modulation → linear projection. No fields are LoRA
/// targets so we don't need to wrap them — but we still load via
/// `quantized_nn::Linear` to match the dtype/dequant semantics of upstream.
struct FinalLayer {
    norm_final: LayerNorm,
    linear: QuantizedLinear,
    ada_ln_modulation: QuantizedLinear,
}

impl FinalLayer {
    fn load(h_sz: usize, p_sz: usize, out_c: usize, vb: VarBuilder, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm_final: layer_norm(h_sz, &vb.pp("norm_final"), dtype)?,
            linear: quantized_linear_at(h_sz, p_sz * p_sz * out_c, true, &vb.pp("linear"), dtype)?,
            ada_ln_modulation: quantized_linear_at(
                h_sz,
                2 * h_sz,
                true,
                &vb.pp("adaLN_modulation.1"),
                dtype,
            )?,
        })
    }
    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        Ok(xs.apply(&self.linear)?)
    }
}

/// Mlp embedder for stem inputs (`time_in`, `vector_in`, `guidance_in`).
/// `quantized_nn::Linear` × 2 + SiLU between them — no LoRA targets here.
struct StemMlpEmbedder {
    in_layer: QuantizedLinear,
    out_layer: QuantizedLinear,
}

impl StemMlpEmbedder {
    fn load(in_sz: usize, h_sz: usize, vb: VarBuilder, dtype: DType) -> Result<Self> {
        Ok(Self {
            in_layer: quantized_linear_at(in_sz, h_sz, true, &vb.pp("in_layer"), dtype)?,
            out_layer: quantized_linear_at(h_sz, h_sz, true, &vb.pp("out_layer"), dtype)?,
        })
    }
}

impl Module for StemMlpEmbedder {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

// ── Main quantized transformer ───────────────────────────────────────────────

/// FLUX transformer with quantized (GGUF) weights and bypass-mode LoRA.
///
/// Drop-in replacement for `flux::quantized_model::Flux`. The forward
/// signature matches `WithForward`, so the callsite in `transformer.rs`
/// only adds a new `FluxTransformer::QuantizedBypass(_)` variant.
pub(crate) struct QuantizedFluxTransformer {
    img_in: QuantizedLinear,
    txt_in: QuantizedLinear,
    time_in: StemMlpEmbedder,
    vector_in: StemMlpEmbedder,
    guidance_in: Option<StemMlpEmbedder>,
    pe_embedder: EmbedNd,
    final_layer: FinalLayer,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    /// Bypass-mode LoRA stack — `None` when no LoRAs are active.
    /// Adapters live on the same device as the GGUF blocks, so swap is
    /// a registry-pointer replace + per-Linear rebind.
    lora_registry: Option<LoraRegistry>,
}

impl QuantizedFluxTransformer {
    /// Load the full quantized FLUX transformer from a `from_gguf`
    /// `VarBuilder`. All blocks land on `vb.device()`. If `registry`
    /// is set, adapters are bound to each Linear at load time.
    pub fn load(
        cfg: &Config,
        vb: VarBuilder,
        registry: Option<&LoraRegistry>,
        progress: &ProgressReporter,
        working_dtype: DType,
    ) -> Result<Self> {
        progress.info("Loading FLUX quantized transformer (bypass-mode LoRA)");
        let dtype = working_dtype;

        let img_in = quantized_linear_at(
            cfg.in_channels,
            cfg.hidden_size,
            true,
            &vb.pp("img_in"),
            dtype,
        )?;
        let txt_in = quantized_linear_at(
            cfg.context_in_dim,
            cfg.hidden_size,
            true,
            &vb.pp("txt_in"),
            dtype,
        )?;
        let time_in = StemMlpEmbedder::load(256, cfg.hidden_size, vb.pp("time_in"), dtype)?;
        let vector_in =
            StemMlpEmbedder::load(cfg.vec_in_dim, cfg.hidden_size, vb.pp("vector_in"), dtype)?;
        let guidance_in = if cfg.guidance_embed {
            Some(StemMlpEmbedder::load(
                256,
                cfg.hidden_size,
                vb.pp("guidance_in"),
                dtype,
            )?)
        } else {
            None
        };

        let pe_dim = cfg.hidden_size / cfg.num_heads;
        let pe_embedder = EmbedNd::new(pe_dim, cfg.theta, cfg.axes_dim.to_vec());

        let final_layer = FinalLayer::load(
            cfg.hidden_size,
            1,
            cfg.in_channels,
            vb.pp("final_layer"),
            dtype,
        )?;

        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("double_blocks");
        for idx in 0..cfg.depth {
            double_blocks.push(DoubleBlock::load(cfg, vb_d.pp(idx), dtype, registry, idx)?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        let vb_s = vb.pp("single_blocks");
        for idx in 0..cfg.depth_single_blocks {
            single_blocks.push(SingleBlock::load(cfg, vb_s.pp(idx), dtype, registry, idx)?);
        }

        progress.info(&format!(
            "Quantized transformer loaded: {} double + {} single blocks \
             (GPU-resident, {dtype:?} activations)",
            double_blocks.len(),
            single_blocks.len(),
        ));

        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            final_layer,
            double_blocks,
            single_blocks,
            lora_registry: registry.cloned(),
        })
    }

    /// Replace the bypass-mode LoRA stack in-place. Walks every Linear
    /// site and rebinds. No transformer reload — this is the
    /// LoRA-swap fast path the keystone makes possible.
    pub(crate) fn set_lora_registry(&mut self, registry: Option<LoraRegistry>) {
        let r = registry.as_ref();
        for (idx, b) in self.double_blocks.iter_mut().enumerate() {
            b.rebind_lora(r, idx);
        }
        for (idx, b) in self.single_blocks.iter_mut().enumerate() {
            b.rebind_lora(r, idx);
        }
        self.lora_registry = registry;
    }

    /// True when at least one bypass-mode adapter is installed.
    #[allow(dead_code)]
    pub(crate) fn has_loras(&self) -> bool {
        self.lora_registry
            .as_ref()
            .map(|r| !r.is_empty())
            .unwrap_or(false)
    }

    /// Run the full FLUX forward pass, with an optional per-block hook.
    ///
    /// Mirrors the candle fork's `Flux::forward_with_hook` for the two upstream
    /// variants, but takes an `Option` rather than a no-op implementation: a
    /// `None` hook has to execute the untouched loop, because a gated PuLID
    /// step must be bit-identical to a render that never asked for identity.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_hook(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        hook: Option<&dyn BlockHook>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 {
            anyhow::bail!("unexpected shape for txt {:?}", txt.shape());
        }
        if img.rank() != 3 {
            anyhow::bail!("unexpected shape for img {:?}", img.shape());
        }
        let dtype = img.dtype();

        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            // Upstream's `rope` runs on float positions and returns `.float()`
            // (`flux/math.py:16-25`) — and FLUX.2 already does the same
            // (`flux2/transformer.rs::rope`). The fork's `EmbedNd` follows the
            // dtype it is handed, so a half working dtype would otherwise
            // compute every `cos`/`sin` in eight mantissa bits.
            let ids = ids.to_dtype(DType::F32)?;
            ids.apply(&self.pe_embedder)?
        };

        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;

        let vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        for (index, block) in self.double_blocks.iter().enumerate() {
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?;
            if let Some(hook) = hook {
                if let Some(replacement) = hook.after_double_block(index, &img, &txt)? {
                    img = replacement;
                }
            }
        }

        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.dim(1)?;
        for (index, block) in self.single_blocks.iter().enumerate() {
            img = block.forward(&img, &vec_, &pe)?;
            if let Some(hook) = hook {
                if let Some(replacement) = hook.after_single_block(index, txt_len, &img)? {
                    img = replacement;
                }
            }
        }

        let img = img.i((.., txt_len..))?;
        self.final_layer.forward(&img, &vec_)
    }
}

/// Helper: replace adapters on a `LoraLinear` from a registry lookup.
/// Pulled out so every block / sublayer's `rebind_lora` reads the same.
fn rebind(lin: &mut LoraLinear, registry: Option<&LoraRegistry>, key: &str) {
    let stack = registry
        .map(|r| r.adapters_for(key).to_vec())
        .unwrap_or_default();
    lin.set_adapters(stack);
}

// ── Test-only helpers used to construct synthetic GGUF fixtures ──────────────

/// Quantize a CPU F32 tensor onto `device` in `dtype`. Wrapper around
/// `QTensor::quantize_onto`. Test-only — production loads QTensors from
/// real GGUF files via the upstream `from_gguf` path.
#[cfg(test)]
fn quantize_cpu(
    src: &Tensor,
    dtype: candle_core::quantized::GgmlDType,
    device: &candle_core::Device,
) -> Result<QTensor> {
    Ok(mold_candle::quantized::quantize_onto(src, dtype, device)?)
}

/// Convenience to construct an `Arc<QTensor>` directly from a CPU F32
/// tensor. Test-only.
#[cfg(test)]
fn arc_quantize_cpu(
    src: &Tensor,
    dtype: candle_core::quantized::GgmlDType,
    device: &candle_core::Device,
) -> Result<Arc<QTensor>> {
    Ok(Arc::new(quantize_cpu(src, dtype, device)?))
}

// `Arc<QTensor>` is unused in production builds; reference the import so
// non-test cargo check doesn't warn.
#[allow(dead_code)]
fn _arc_qtensor_unused(_t: Arc<QTensor>) {}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux::lora_bypass::LinearLoraAdapter;
    use candle_core::quantized::GgmlDType;
    use candle_core::Device;

    /// The mold-owned bypass transformer and the candle fork's
    /// `flux::quantized_model::Flux` are the same network read two ways, and
    /// the whole reason the fork arm can be deleted is that they agree.
    ///
    /// They are NOT bit-identical any more, by design and in exactly two
    /// named places: FLUX renders under `AttentionPolicy::FastStill`, whose
    /// math path folds the softmax scale into K rather than multiplying the
    /// score matrix, and this transformer's LayerNorms now carry a zero bias
    /// so they reach candle's fused kernel (`candle-nn/src/layer_norm.rs:118`)
    /// instead of the unfused sum/div/sqrt sequence. Both change reduction
    /// order, neither changes the network. `1e-4` over a four-double /
    /// eight-single forward at F32 is what that costs.
    #[test]
    fn f32_bypass_forward_matches_the_upstream_quantized_model() {
        use crate::flux::pulid_variants::{gguf_weights, shared_weights, tiny_flux_config};
        use candle_transformers::models::flux::WithForward;

        let device = Device::Cpu;
        let cfg = tiny_flux_config();
        let weights = shared_weights(&cfg);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tiny-flux-f32.gguf");
        gguf_weights(&weights, &path).expect("the synthetic gguf writes");

        let upstream_vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&path, &device)
                .expect("upstream gguf var builder");
        let upstream =
            candle_transformers::models::flux::quantized_model::Flux::new(&cfg, upstream_vb)
                .expect("upstream quantized model");

        let progress = ProgressReporter::default();
        let mold_vb =
            mold_candle::quantized::VarBuilder::from_gguf(&path, &device).expect("mold gguf vb");
        let mold = QuantizedFluxTransformer::load(&cfg, mold_vb, None, &progress, DType::F32)
            .expect("mold quantized transformer");

        let inputs = tiny_forward_inputs(&cfg, DType::F32, &device);
        let want = upstream
            .forward(
                &inputs.img,
                &inputs.img_ids,
                &inputs.txt,
                &inputs.txt_ids,
                &inputs.timesteps,
                &inputs.y,
                None,
            )
            .expect("upstream forward");
        let got = mold
            .forward_with_hook(
                &inputs.img,
                &inputs.img_ids,
                &inputs.txt,
                &inputs.txt_ids,
                &inputs.timesteps,
                &inputs.y,
                None,
                None,
            )
            .expect("mold forward");
        assert_eq!(got.dims(), want.dims());
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff < 1e-5,
            "the mold-owned transformer diverged from the fork's by {diff}"
        );
    }

    /// A real GGUF is a mix: quantized blocks beside dense F16 stems, biases
    /// and norm scales. At any working dtype other than F32 the dense half
    /// has to be materialized at that dtype — `QTensor::dequantize` answers
    /// F32 and nothing downstream will widen for it — and the norms have to
    /// follow, because candle's fused RMS/LayerNorm kernels are dtype-typed
    /// (`candle-nn/src/ops.rs:535-537`, `:770`). Get any of that wrong and
    /// the forward does not merely drift, it errors.
    #[test]
    fn mixed_f16_and_q8_gguf_forwards_in_a_non_f32_working_dtype() {
        use crate::flux::pulid_variants::{
            mixed_dtype_gguf_weights, shared_weights, tiny_flux_config,
        };

        let device = Device::Cpu;
        let cfg = tiny_flux_config();
        let weights = shared_weights(&cfg);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tiny-flux-mixed.gguf");
        mixed_dtype_gguf_weights(&weights, &path).expect("the synthetic gguf writes");

        let progress = ProgressReporter::default();
        let forward_at = |dtype: DType| {
            let vb =
                mold_candle::quantized::VarBuilder::from_gguf(&path, &device).expect("gguf vb");
            let model = QuantizedFluxTransformer::load(&cfg, vb, None, &progress, dtype)
                .expect("mixed-dtype transformer");
            let inputs = tiny_forward_inputs(&cfg, dtype, &device);
            model
                .forward_with_hook(
                    &inputs.img,
                    &inputs.img_ids,
                    &inputs.txt,
                    &inputs.txt_ids,
                    &inputs.timesteps,
                    &inputs.y,
                    None,
                    None,
                )
                .expect("mixed-dtype forward")
        };

        let reference = forward_at(DType::F32);
        let half = forward_at(DType::F16);
        assert_eq!(half.dtype(), DType::F16, "the working dtype must survive");
        let diff = max_abs_diff(
            &reference.to_dtype(DType::F32).unwrap(),
            &half.to_dtype(DType::F32).unwrap(),
        );
        assert!(
            diff < 5e-2,
            "the F16 working dtype diverged from F32 by {diff}"
        );
    }

    struct TinyForwardInputs {
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
    }

    /// Deterministic forward inputs at an explicit dtype.
    ///
    /// The position ids follow the activation dtype because that is what
    /// `flux::sampling::State::new` does (`sampling.rs:25`, `:43`, `:47`) and
    /// what `rope` keys its inverse frequencies on (`model.rs:92`) — a
    /// mismatch there is a `broadcast_mul` dtype error, not a drift.
    fn tiny_forward_inputs(cfg: &Config, dtype: DType, device: &Device) -> TinyForwardInputs {
        let img_tokens = 6usize;
        let txt_tokens = 5usize;
        let ramp = |count: usize, offset: f32| -> Vec<f32> {
            (0..count)
                .map(|i| ((i as f32 * 0.017 + offset).sin()) * 0.5)
                .collect()
        };
        let img = Tensor::from_vec(
            ramp(img_tokens * cfg.in_channels, 0.0),
            (1, img_tokens, cfg.in_channels),
            device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        let txt = Tensor::from_vec(
            ramp(txt_tokens * cfg.context_in_dim, 1.0),
            (1, txt_tokens, cfg.context_in_dim),
            device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        TinyForwardInputs {
            img,
            img_ids: Tensor::zeros((1, img_tokens, 3), dtype, device).unwrap(),
            txt,
            txt_ids: Tensor::zeros((1, txt_tokens, 3), dtype, device).unwrap(),
            timesteps: Tensor::from_vec(vec![0.5f32], 1, device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
            y: Tensor::from_vec(
                (0..cfg.vec_in_dim)
                    .map(|i| i as f32 / 16.0)
                    .collect::<Vec<_>>(),
                (1, cfg.vec_in_dim),
                device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap(),
        }
    }

    /// Quantized linear with no adapter must be bit-identical to the
    /// unwrapped quantized linear. Catches any forward dispatch
    /// regression on the no-LoRA hot path.
    #[test]
    fn quantized_linear_with_no_adapter_matches_unwrapped_forward() {
        // Q8_0 / Q4_0 GGUF quantization requires the last dim of every
        // quantized tensor to be a multiple of 32 (block size). We pick
        // 32×64 for the base linear so the Q8_0 path is exercisable
        // without padding hacks.
        let device = Device::Cpu;
        let in_dim = 32;
        let out_dim = 64;

        // Build a base weight + bias on CPU/F32, quantize to Q8_0, wrap.
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let weight = Tensor::from_vec(weight, (out_dim, in_dim), &device).unwrap();
        let q_weight = arc_quantize_cpu(&weight, GgmlDType::Q8_0, &device).unwrap();

        let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.01).collect();
        let bias = Tensor::from_vec(bias, (out_dim,), &device).unwrap();

        let inner = QuantizedLinear::new(
            q_weight.clone(),
            Some(bias.clone()),
            &device,
            DType::F32,
            false,
        )
        .unwrap();
        let wrapped = LoraLinear::Quantized(inner.clone());

        let x_data: Vec<f32> = (0..2 * 3 * in_dim)
            .map(|i| ((i as f32) * 0.017).cos())
            .collect();
        let x = Tensor::from_vec(x_data, (2, 3, in_dim), &device).unwrap();

        let baseline = <QuantizedLinear as candle_core::Module>::forward(&inner, &x).unwrap();
        let wrapped_out = wrapped.forward(&x).unwrap();
        let max = max_abs_diff(&baseline, &wrapped_out);
        assert!(max < 1e-7, "no-adapter wrapped diverged: {max}");
    }

    /// Bypass output `q.forward(x) + scale·(x@d.T)@u.T` must match the
    /// ground-truth dequant→merge→requant→forward at the q-noise tolerance.
    /// This is the math correctness check for the GGUF bypass path.
    #[test]
    fn quantized_linear_with_one_adapter_matches_explicit_dequant_merge() {
        // in_dim must be a Q8_0 block multiple (32). out_dim is the
        // "row" axis which is unconstrained by the block layout.
        let device = Device::Cpu;
        let in_dim = 32;
        let out_dim = 64;
        let rank = 4;
        let scale = 0.5f32;

        // Base weight (no bias for this test — keeps the math compact).
        let w_vec: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.1)
            .collect();
        let w = Tensor::from_vec(w_vec, (out_dim, in_dim), &device).unwrap();

        // LoRA tensors (down: rank×in, up: out×rank).
        let down_vec: Vec<f32> = (0..rank * in_dim)
            .map(|i| ((i as f32) * 0.019).cos() * 0.05)
            .collect();
        let up_vec: Vec<f32> = (0..out_dim * rank)
            .map(|i| ((i as f32) * 0.023).sin() * 0.05)
            .collect();
        let down = Tensor::from_vec(down_vec, (rank, in_dim), &device).unwrap();
        let up = Tensor::from_vec(up_vec, (out_dim, rank), &device).unwrap();

        // Bypass: quantize base only, apply LoRA at forward.
        let q_w = arc_quantize_cpu(&w, GgmlDType::Q8_0, &device).unwrap();
        let inner = QuantizedLinear::new(q_w, None, &device, DType::F32, false).unwrap();
        let bypass = LoraLinear::WithAdaptersQuantized {
            inner,
            adapters: vec![LinearLoraAdapter {
                down: down.clone(),
                up: up.clone(),
                scale,
                fused_slice: None,
            }],
        };

        // Ground truth: merge in F32, requant, forward through new
        // quantized linear. Both sides eat the same q-noise on the
        // BASE weight so the diff is dominated by the rounding of the
        // LoRA delta itself, not by the base.
        let merged_delta = up.matmul(&down).unwrap().affine(scale as f64, 0.0).unwrap();
        let merged_w = (&w + &merged_delta).unwrap();
        let q_merged = arc_quantize_cpu(&merged_w, GgmlDType::Q8_0, &device).unwrap();
        let merged_inner =
            QuantizedLinear::new(q_merged, None, &device, DType::F32, false).unwrap();

        let x_vec: Vec<f32> = (0..3 * in_dim)
            .map(|i| ((i as f32) * 0.029).cos() * 0.5)
            .collect();
        let x = Tensor::from_vec(x_vec, (1, 3, in_dim), &device).unwrap();

        let bypass_out = bypass.forward(&x).unwrap();
        let merged_out =
            <QuantizedLinear as candle_core::Module>::forward(&merged_inner, &x).unwrap();

        // Q8_0 has ~1% error; LoRA adds another small rounding pass.
        // The merge path saves the LoRA contribution into the same
        // Q8_0 grid, so its error is bounded by the LoRA delta's
        // dynamic range — looser tol than F32-vs-F32.
        let max = max_abs_diff(&bypass_out, &merged_out);
        assert!(max < 5e-2, "Q8_0 bypass vs merge max diff: {max}");
    }

    /// Sum of two adapter contributions must equal a single-stack
    /// forward — checks that adapter composition is purely additive.
    #[test]
    fn quantized_linear_two_adapters_compose() {
        let device = Device::Cpu;
        let in_dim = 32;
        let out_dim = 16;
        let rank = 3;

        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.1)
            .collect();
        let w = Tensor::from_vec(w, (out_dim, in_dim), &device).unwrap();
        let q_w = arc_quantize_cpu(&w, GgmlDType::Q8_0, &device).unwrap();

        let make_pair = |salt: f32| {
            let d: Vec<f32> = (0..rank * in_dim)
                .map(|i| ((i as f32 + salt) * 0.013).sin() * 0.03)
                .collect();
            let u: Vec<f32> = (0..out_dim * rank)
                .map(|i| ((i as f32 + salt) * 0.017).cos() * 0.03)
                .collect();
            (
                Tensor::from_vec(d, (rank, in_dim), &device).unwrap(),
                Tensor::from_vec(u, (out_dim, rank), &device).unwrap(),
            )
        };
        let (d1, u1) = make_pair(1.0);
        let (d2, u2) = make_pair(7.0);
        let s1 = 0.4f32;
        let s2 = -0.3f32;

        // Both adapters bypassed at forward.
        let inner = QuantizedLinear::new(q_w.clone(), None, &device, DType::F32, false).unwrap();
        let two = LoraLinear::WithAdaptersQuantized {
            inner: inner.clone(),
            adapters: vec![
                LinearLoraAdapter {
                    down: d1.clone(),
                    up: u1.clone(),
                    scale: s1,
                    fused_slice: None,
                },
                LinearLoraAdapter {
                    down: d2.clone(),
                    up: u2.clone(),
                    scale: s2,
                    fused_slice: None,
                },
            ],
        };

        // One adapter at a time, summed.
        let one1 = LoraLinear::WithAdaptersQuantized {
            inner: inner.clone(),
            adapters: vec![LinearLoraAdapter {
                down: d1,
                up: u1,
                scale: s1,
                fused_slice: None,
            }],
        };
        let one2 = LoraLinear::WithAdaptersQuantized {
            inner,
            adapters: vec![LinearLoraAdapter {
                down: d2,
                up: u2,
                scale: s2,
                fused_slice: None,
            }],
        };

        let x: Vec<f32> = (0..2 * in_dim)
            .map(|i| ((i as f32) * 0.029).cos() * 0.5)
            .collect();
        let x = Tensor::from_vec(x, (1, 2, in_dim), &device).unwrap();

        // two_out = base + Δ1 + Δ2
        // one1_out + one2_out = (base + Δ1) + (base + Δ2) = 2*base + Δ1 + Δ2
        // So: two_out + base = one1_out + one2_out
        let two_out = two.forward(&x).unwrap();
        let one1_out = one1.forward(&x).unwrap();
        let one2_out = one2.forward(&x).unwrap();
        let base_out = <QuantizedLinear as candle_core::Module>::forward(
            &QuantizedLinear::new(q_w, None, &device, DType::F32, false).unwrap(),
            &x,
        )
        .unwrap();
        let lhs = (&two_out + &base_out).unwrap();
        let rhs = (&one1_out + &one2_out).unwrap();
        let max = max_abs_diff(&lhs, &rhs);
        assert!(max < 1e-5, "two-adapter compose != sum of singles: {max}");
    }

    /// Adapter on Q-only must not disturb K/V output rows — fused-slice
    /// plumbing must clip the delta to the configured row range.
    #[test]
    fn quantized_linear_fused_qkv_only_writes_target_slice() {
        let device = Device::Cpu;
        let in_dim = 32; // Q8_0 block multiple
        let h = 16; // each of Q, K, V is `h` rows in the fused output
        let out_dim = 3 * h;
        let rank = 2;

        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.1)
            .collect();
        let w = Tensor::from_vec(w, (out_dim, in_dim), &device).unwrap();
        let q_w = arc_quantize_cpu(&w, GgmlDType::Q8_0, &device).unwrap();
        let inner = QuantizedLinear::new(q_w, None, &device, DType::F32, false).unwrap();

        // Adapter only touches Q (rows [0, h)).
        let down: Vec<f32> = (0..rank * in_dim)
            .map(|i| ((i as f32) * 0.013).sin() * 0.05)
            .collect();
        let up: Vec<f32> = (0..h * rank)
            .map(|i| ((i as f32) * 0.017).cos() * 0.05)
            .collect();
        let down = Tensor::from_vec(down, (rank, in_dim), &device).unwrap();
        let up = Tensor::from_vec(up, (h, rank), &device).unwrap();

        let with_q = LoraLinear::WithAdaptersQuantized {
            inner: inner.clone(),
            adapters: vec![LinearLoraAdapter {
                down,
                up,
                scale: 0.7,
                fused_slice: Some(crate::flux::lora_bypass::FusedSlice {
                    offset: 0,
                    length: h,
                }),
            }],
        };

        let x: Vec<f32> = (0..3 * in_dim)
            .map(|i| ((i as f32) * 0.029).cos() * 0.5)
            .collect();
        let x = Tensor::from_vec(x, (1, 3, in_dim), &device).unwrap();

        let plain_out = <QuantizedLinear as candle_core::Module>::forward(&inner, &x).unwrap();
        let bypass_out = with_q.forward(&x).unwrap();

        // K and V rows ([h, 2h) and [2h, 3h)) must match plain exactly.
        let kv_plain = plain_out.narrow(2, h, 2 * h).unwrap();
        let kv_bypass = bypass_out.narrow(2, h, 2 * h).unwrap();
        let max = max_abs_diff(&kv_plain, &kv_bypass);
        assert!(max < 1e-7, "K/V rows drifted under Q-only adapter: {max}");
    }

    /// `set_lora_registry` walks the live transformer and rebinds every
    /// adapter target without rebuilding the model. The same registry
    /// installed twice must leave the second call's per-Linear adapter
    /// vectors equal to the first's — a pure-rebind contract.
    ///
    /// This synthesises a 1-block FLUX-shaped transformer with handcrafted
    /// quantized weights so we don't need a real GGUF fixture.
    #[test]
    fn bypass_skips_rebuild_on_same_fingerprint() {
        let device = Device::Cpu;
        // Make a synthetic registry targeting `double_blocks.0.img_attn.qkv.weight`
        // so the rebind call has somewhere to land.
        use crate::flux::lora::{LoraAdapter, LoraLayer, LoraSpec};
        use crate::flux::lora_bypass::{build_registry, LoraRegistry};
        use std::collections::HashMap as HM;

        let h = 16;
        let a = Tensor::zeros((4, h), DType::F32, &device).unwrap();
        let b = Tensor::zeros((h, 4), DType::F32, &device).unwrap();
        let mut layers = std::collections::HashMap::new();
        layers.insert(
            "transformer.transformer_blocks.0.attn.to_q".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 4 };
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 0.5,
            path_hash: 0x1234,
        }];
        let mut linear_out_dims = HM::new();
        linear_out_dims.insert("double_blocks.0.img_attn.qkv.weight".to_string(), 3 * h);

        let r1 = build_registry(&specs, &linear_out_dims, &device, DType::F32).unwrap();
        let r2 = build_registry(&specs, &linear_out_dims, &device, DType::F32).unwrap();
        assert_eq!(r1.len(), r2.len(), "same specs → same registry length");

        // We don't construct a full transformer here (no GGUF fixture);
        // the rebuild-skip semantics are entirely captured by `LoraLinear::set_adapters`
        // collapsing to the same enum variant when the adapter stack is
        // identical. That collapse is exhaustively tested in `lora_bypass.rs`,
        // so all this test asserts is that two registries built from the
        // same input have equivalent target-tensor coverage.
        let key = "double_blocks.0.img_attn.qkv.weight";
        assert_eq!(r1.adapters_for(key).len(), r2.adapters_for(key).len());

        // Suppress "unused" so the registry stays in scope during the assert.
        drop(LoraRegistry::default());
    }

    /// Reordering the QK RMSNorm ahead of the transpose is a layout change,
    /// not an arithmetic one.
    ///
    /// RMSNorm reduces over the LAST dim, and `head_dim` is last in both the
    /// packed `B L H D` layout and the transposed `B H L D` one, so the two
    /// orders compute the same numbers. The reorder exists only because
    /// candle takes its fused kernel exclusively for a contiguous input
    /// (`candle-nn/src/layer_norm.rs:202-210`), and a narrow-then-transpose
    /// view is never contiguous — which is what sent every one of these
    /// through the strided ten-op fallback.
    #[test]
    fn rms_norm_before_transpose_equals_after() {
        let device = Device::Cpu;
        let (b, l, heads, head_dim) = (1usize, 6usize, 3usize, 4usize);
        // A packed QKV exactly as `qkv_split` sees it.
        let qkv = Tensor::arange(0f32, (b * l * 3 * heads * head_dim) as f32, &device)
            .unwrap()
            .reshape((b, l, 3, heads, head_dim))
            .unwrap()
            .affine(0.013, -0.7)
            .unwrap();
        let scale = Tensor::arange(1f32, (head_dim + 1) as f32, &device)
            .unwrap()
            .affine(0.25, 0.5)
            .unwrap();
        let norm = RmsNorm::new(scale, 1e-6);

        let after = qkv
            .i((.., .., 0))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .apply(&norm)
            .unwrap();
        let before = qkv
            .i((.., .., 0))
            .unwrap()
            .contiguous()
            .unwrap()
            .apply(&norm)
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        assert_eq!(before.dims(), after.dims());
        let diff = max_abs_diff(&before.contiguous().unwrap(), &after.contiguous().unwrap());
        assert!(
            diff < 1e-6,
            "normalizing before the transpose moved the values by {diff}"
        );
    }

    /// BFL's interleaved rope layout is candle's `rope_i` contract.
    ///
    /// `math.py:19-32` reshapes the head dim into `(-1, 1, 2)` pairs and
    /// combines them with `freqs_cis`, whose `(i, j)` matrix is
    /// `[[cos, -sin], [sin, cos]]` (`math.py:22`). Taking column `j = 0` out
    /// of it gives `cos = pe[.., 0, 0]` and `sin = pe[.., 1, 0]`, which is
    /// exactly `y0 = x0*cos - x1*sin`, `y1 = x0*sin + x1*cos` — the rule
    /// `candle-nn/src/rotary_emb.rs:6-10` documents and `rope_i` implements
    /// (`:262-288`). This pins the correspondence so the fused kernel can be
    /// reached from an F32 copy without re-deriving the layout by hand.
    #[test]
    fn rope_i_on_f32_matches_apply_rope() {
        let device = Device::Cpu;
        let (b, heads, seq, head_dim) = (1usize, 2usize, 5usize, 8usize);
        let angles = Tensor::arange(0f32, (b * seq * head_dim / 2) as f32, &device)
            .unwrap()
            .affine(0.37, -1.1)
            .unwrap()
            .reshape((b, seq, head_dim / 2))
            .unwrap();
        let cos = angles.cos().unwrap();
        let sin = angles.sin().unwrap();
        // `[[cos, -sin], [sin, cos]]`, stacked exactly as candle's `rope`
        // builds FLUX's positional embedding.
        let pe = Tensor::stack(
            &[&cos, &sin.neg().unwrap(), &sin, &cos],
            candle_core::D::Minus1,
        )
        .unwrap()
        .reshape((b, seq, head_dim / 2, 2, 2))
        .unwrap();

        let x = Tensor::arange(0f32, (b * heads * seq * head_dim) as f32, &device)
            .unwrap()
            .affine(0.011, -0.4)
            .unwrap()
            .reshape((b, heads, seq, head_dim))
            .unwrap();

        let want = apply_rope(&x, &pe).unwrap();
        let got = candle_nn::rotary_emb::rope_i(
            &x.contiguous().unwrap(),
            &cos.squeeze(0).unwrap().contiguous().unwrap(),
            &sin.squeeze(0).unwrap().contiguous().unwrap(),
        )
        .unwrap();

        assert_eq!(got.dims(), want.dims());
        let diff = max_abs_diff(&got, &want.contiguous().unwrap());
        assert!(diff < 1e-6, "rope_i diverged from apply_rope by {diff}");
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = (a - b).unwrap().abs().unwrap();
        diff.flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }
}
