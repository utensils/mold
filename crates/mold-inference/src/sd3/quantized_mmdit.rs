//! Quantized (GGUF) MMDiT for SD3.5
//!
//! Mirrors `candle_transformers::models::mmdit::model::MMDiT` but uses quantized layer
//! types from `candle_transformers::quantized_nn`. The GGUF tensor naming from city96
//! quantizations preserves the BF16 naming convention, so tensor paths match directly.
//!
//! Supports both sd3_5_large (depth=38) and sd3_5_medium (depth=24) configs.

use anyhow::Result;
use candle_core::{DType, Module, Tensor, D};
use candle_nn::RmsNorm as CandleRmsNorm;
use candle_transformers::models::mmdit::model::Config as MMDiTConfig;
use candle_transformers::quantized_nn::{self, Linear};
use candle_transformers::quantized_var_builder::VarBuilder;

// ==================== LayerNormNoAffine ====================

struct LayerNormNoAffine {
    eps: f64,
}

impl LayerNormNoAffine {
    fn new(eps: f64) -> Self {
        Self { eps }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::LayerNorm::new_no_bias(Tensor::ones_like(x)?, self.eps).forward(x)
    }
}

// ==================== PatchEmbedder ====================

struct PatchEmbedder {
    proj_weight: Tensor,
    proj_bias: Tensor,
    patch_size: usize,
}

impl PatchEmbedder {
    fn new(
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let proj_vb = vb.pp("proj");
        let proj_weight = proj_vb
            .get((embed_dim, in_channels, patch_size, patch_size), "weight")?
            .dequantize(vb.device())?;
        let proj_bias = proj_vb.get(embed_dim, "bias")?.dequantize(vb.device())?;
        Ok(Self {
            proj_weight,
            proj_bias,
            patch_size,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.conv2d(&self.proj_weight, 0, self.patch_size, 1, 1)?;
        let x = x.broadcast_add(&self.proj_bias.reshape((1, (), 1, 1))?)?;
        let (b, c, h, w) = x.dims4()?;
        x.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

// ==================== PositionEmbedder ====================

struct PositionEmbedder {
    pos_embed: Tensor,
    patch_size: usize,
    pos_embed_max_size: usize,
}

impl PositionEmbedder {
    fn new(
        hidden_size: usize,
        patch_size: usize,
        pos_embed_max_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pos_embed = vb
            .get(
                (1, pos_embed_max_size * pos_embed_max_size, hidden_size),
                "pos_embed",
            )?
            .dequantize(vb.device())?;
        Ok(Self {
            pos_embed,
            patch_size,
            pos_embed_max_size,
        })
    }

    fn get_cropped_pos_embed(&self, h: usize, w: usize) -> candle_core::Result<Tensor> {
        let h = (h + 1) / self.patch_size;
        let w = (w + 1) / self.patch_size;

        if h > self.pos_embed_max_size || w > self.pos_embed_max_size {
            candle_core::bail!("Input size is too large for the position embedding");
        }

        let top = (self.pos_embed_max_size - h) / 2;
        let left = (self.pos_embed_max_size - w) / 2;

        let pos_embed =
            self.pos_embed
                .reshape((1, self.pos_embed_max_size, self.pos_embed_max_size, ()))?;
        let pos_embed = pos_embed.narrow(1, top, h)?.narrow(2, left, w)?;
        pos_embed.reshape((1, h * w, ()))
    }
}

// ==================== TimestepEmbedder ====================

struct TimestepEmbedder {
    mlp_0: Linear,
    mlp_2: Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    fn new(hidden_size: usize, frequency_embedding_size: usize, vb: VarBuilder) -> Result<Self> {
        let mlp_0 = quantized_nn::linear(frequency_embedding_size, hidden_size, vb.pp("mlp.0"))?;
        let mlp_2 = quantized_nn::linear(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        Ok(Self {
            mlp_0,
            mlp_2,
            frequency_embedding_size,
        })
    }

    fn timestep_embedding(t: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
        let half = dim / 2;
        let max_period: f64 = 10000.0;
        let freqs = Tensor::arange(0f32, half as f32, t.device())?
            .to_dtype(DType::F32)?
            .affine(-max_period.ln() / half as f64, 0.0)?
            .exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .matmul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
        // Keep F32 for quantized path (QMatMul dequantizes weights to F32)
        Ok(embedding)
    }

    fn forward(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let t_freq = Self::timestep_embedding(t, self.frequency_embedding_size)?;
        t_freq.apply(&self.mlp_0)?.silu()?.apply(&self.mlp_2)
    }
}

// ==================== VectorEmbedder ====================

struct VectorEmbedder {
    mlp_0: Linear,
    mlp_2: Linear,
}

impl VectorEmbedder {
    fn new(input_dim: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let mlp_0 = quantized_nn::linear(input_dim, hidden_size, vb.pp("mlp.0"))?;
        let mlp_2 = quantized_nn::linear(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        Ok(Self { mlp_0, mlp_2 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.apply(&self.mlp_0)?.silu()?.apply(&self.mlp_2)
    }
}

// ==================== Mlp ====================

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(in_features: usize, hidden_features: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = quantized_nn::linear(in_features, hidden_features, vb.pp("fc1"))?;
        let fc2 = quantized_nn::linear(hidden_features, in_features, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // GeluPytorchTanh activation
        // Ensure contiguous for quantized matmul
        x.contiguous()?
            .apply(&self.fc1)?
            .apply(&candle_nn::Activation::GeluPytorchTanh)?
            .contiguous()?
            .apply(&self.fc2)
    }
}

// ==================== AttnProjections ====================

struct AttnProjections {
    head_dim: usize,
    qkv: Linear,
    ln_k: Option<CandleRmsNorm>,
    ln_q: Option<CandleRmsNorm>,
    proj: Linear,
}

impl AttnProjections {
    fn new(dim: usize, num_heads: usize, has_qk_norm: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = quantized_nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = quantized_nn::linear(dim, dim, vb.pp("proj"))?;
        let (ln_k, ln_q) = if has_qk_norm {
            let ln_k_w = vb
                .pp("ln_k")
                .get(head_dim, "weight")?
                .dequantize(vb.device())?;
            let ln_q_w = vb
                .pp("ln_q")
                .get(head_dim, "weight")?
                .dequantize(vb.device())?;
            (
                Some(CandleRmsNorm::new(ln_k_w, 1e-6)),
                Some(CandleRmsNorm::new(ln_q_w, 1e-6)),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            head_dim,
            qkv,
            ln_k,
            ln_q,
            proj,
        })
    }

    fn pre_attention(&self, x: &Tensor) -> candle_core::Result<Qkv> {
        let qkv = self.qkv.forward(x)?;
        let Qkv { q, k, v } = split_qkv(&qkv, self.head_dim)?;
        let q = match self.ln_q.as_ref() {
            None => q,
            Some(l) => {
                let (b, t, h) = q.dims3()?;
                l.forward(&q.reshape((b, t, (), self.head_dim))?)?
                    .reshape((b, t, h))?
            }
        };
        let k = match self.ln_k.as_ref() {
            None => k,
            Some(l) => {
                let (b, t, h) = k.dims3()?;
                l.forward(&k.reshape((b, t, (), self.head_dim))?)?
                    .reshape((b, t, h))?
            }
        };
        Ok(Qkv { q, k, v })
    }

    fn post_attention(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Trace attention output before proj
        static PROJ_DIAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !PROJ_DIAG.swap(true, std::sync::atomic::Ordering::Relaxed)
            && std::env::var_os("MOLD_SD3_DEBUG").is_some()
        {
            let xf = x.to_dtype(DType::F32)?;
            let mn = xf.min_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
            let mx = xf.max_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
            eprintln!("[sd3-proj] attention output (before proj): [{mn:.4},{mx:.4}] shape={:?} dtype={:?}", x.shape(), x.dtype());
        }
        // Quantized matmul requires contiguous tensors
        self.proj.forward(&x.contiguous()?)
    }
}

// ==================== QkvOnlyAttnProjections ====================

struct QkvOnlyAttnProjections {
    qkv: Linear,
    head_dim: usize,
}

impl QkvOnlyAttnProjections {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = quantized_nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        Ok(Self { qkv, head_dim })
    }

    fn pre_attention(&self, x: &Tensor) -> candle_core::Result<Qkv> {
        let qkv = self.qkv.forward(x)?;
        split_qkv(&qkv, self.head_dim)
    }
}

// ==================== Qkv + helpers ====================

struct Qkv {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

fn split_qkv(qkv: &Tensor, head_dim: usize) -> candle_core::Result<Qkv> {
    let (batch_size, seq_len, _) = qkv.dims3()?;
    let qkv = qkv.reshape((batch_size, seq_len, 3, (), head_dim))?;
    let q = qkv.get_on_dim(2, 0)?.reshape((batch_size, seq_len, ()))?;
    let k = qkv.get_on_dim(2, 1)?.reshape((batch_size, seq_len, ()))?;
    let v = qkv.get_on_dim(2, 2)?;
    Ok(Qkv { q, k, v })
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> candle_core::Result<Tensor> {
    let shift = shift.unsqueeze(1)?;
    let scale = scale.unsqueeze(1)?;
    let scale_plus_one = scale.broadcast_add(&Tensor::ones_like(&scale)?)?;
    // Ensure contiguous for downstream quantized matmul operations
    shift
        .broadcast_add(&x.broadcast_mul(&scale_plus_one)?)?
        .contiguous()
}

/// Attention computation in BF16 — matches ComfyUI/sd.cpp approach.
///
/// ComfyUI dequantizes GGUF to F16 and runs attention in F16.
/// sd.cpp uses ggml's native F16 attention kernels.
/// Our quantized linears output F32, so we cast Q/K/V to BF16 for attention
/// (halving the [B*heads, seq, seq] matrix from ~5.5GB to ~2.75GB at 1024x1024)
/// then cast back to F32 for downstream quantized linear layers.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, num_heads: usize) -> candle_core::Result<Tensor> {
    let batch_size = q.dim(0)?;
    let seqlen = q.dim(1)?;
    let q = q.reshape((batch_size, seqlen, num_heads, ()))?;
    let k = k.reshape((batch_size, seqlen, num_heads, ()))?;
    let headdim = q.dim(D::Minus1)?;
    let softmax_scale = 1.0 / (headdim as f64).sqrt();

    // Cast to BF16 for attention (halves the massive [B*heads, seq, seq] matrix VRAM).
    // Every reshape/transpose is followed by .contiguous() to prevent CUDA garbage reads.
    let q = q.transpose(1, 2)?.contiguous()?.flatten_to(1)?.to_dtype(DType::F16)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?.flatten_to(1)?.to_dtype(DType::F16)?.contiguous()?;
    let v_orig_shape = v.dims().to_vec();
    let v = v.reshape((batch_size, seqlen, num_heads, ()))?;
    // Check V values BEFORE F16 cast
    if std::env::var_os("MOLD_SD3_DEBUG").is_some() {
        static V_DIAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !V_DIAG.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let vf = v.to_dtype(DType::F32)?;
            let mn = vf.min_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
            let mx = vf.max_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
            eprintln!("[sd3-attn] V before F16 cast: [{mn:.4},{mx:.4}] (F16 max=65504)");
        }
    }
    let v = v
        .transpose(1, 2)?
        .contiguous()?
        .flatten_to(1)?
        .to_dtype(DType::F16)?
        .contiguous()?;

    // Chunked attention: process query tokens in chunks to avoid OOM on the
    // massive [B*heads, full_seq, full_seq] attention matrix.
    // At 1024x1024 with batch=2: [76, 4429, 4429] F16 = ~3GB which OOMs on 24GB.
    // Chunk size 512: [76, 512, 4429] F16 = ~345MB per chunk — fits easily.
    let k_t = k.t()?.contiguous()?;
    let chunk_size = 512;
    let mut attn_chunks = Vec::new();
    let mut offset = 0;
    let mut _chunk_idx = 0;
    while offset < seqlen {
        let len = chunk_size.min(seqlen - offset);
        let q_chunk = q.narrow(1, offset, len)?.contiguous()?;
        let weights_chunk = (q_chunk.matmul(&k_t)? * softmax_scale)?;
        // Softmax in F32 to avoid candle F16 softmax CUDA issues, then back to F16
        let sm_chunk = candle_nn::ops::softmax_last_dim(&weights_chunk.to_dtype(DType::F32)?)?
            .to_dtype(DType::F16)?;
        // Trace softmax output on first chunk
        if _chunk_idx == 0 && std::env::var_os("MOLD_SD3_DEBUG").is_some() {
            static SM_DIAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SM_DIAG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let smf = sm_chunk.to_dtype(DType::F32)?;
                let mn = smf.min_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
                let mx = smf.max_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
                let sm = smf.sum_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
                eprintln!("[sd3-attn] softmax chunk0: [{mn:.6},{mx:.6}] sum={sm:.2}");
            }
        }
        let scores_chunk = sm_chunk.contiguous()?.matmul(&v)?;
        // Trace first few chunks on first call
        static CHUNK_DIAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !CHUNK_DIAG.load(std::sync::atomic::Ordering::Relaxed)
            && _chunk_idx < 3
            && std::env::var_os("MOLD_SD3_DEBUG").is_some()
        {
            let sf = scores_chunk.to_dtype(DType::F32)?;
            let mn = sf.min_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
            let mx = sf.max_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
            let wf = weights_chunk.to_dtype(DType::F32)?;
            let wmn = wf.min_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
            let wmx = wf.max_all()?.to_scalar::<f32>().unwrap_or(f32::NAN);
            eprintln!("[sd3-chunk] chunk {_chunk_idx} (offset={offset},len={len}): weights=[{wmn:.4},{wmx:.4}] scores=[{mn:.4},{mx:.4}]");
        }
        attn_chunks.push(scores_chunk);
        offset += len;
        _chunk_idx += 1;
    }
    let attn_scores = Tensor::cat(&attn_chunks, 1)?;

    // Cast back to F32, ensure contiguous for downstream quantized linear layers
    let head_dim = v_orig_shape.last().copied().unwrap_or(headdim);
    attn_scores
        .reshape((batch_size, num_heads, seqlen, head_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .reshape((batch_size, seqlen, ()))?
        .contiguous()
}

fn joint_attn(
    context_qkv: &Qkv,
    x_qkv: &Qkv,
    num_heads: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    let q = Tensor::cat(&[&context_qkv.q, &x_qkv.q], 1)?;
    let k = Tensor::cat(&[&context_qkv.k, &x_qkv.k], 1)?;
    let v = Tensor::cat(&[&context_qkv.v, &x_qkv.v], 1)?;

    let seqlen = q.dim(1)?;
    let attn = attention(&q, &k, &v, num_heads)?;
    let context_seqlen = context_qkv.q.dim(1)?;
    let context_attn = attn.narrow(1, 0, context_seqlen)?;
    let x_attn = attn.narrow(1, context_seqlen, seqlen - context_seqlen)?;
    Ok((context_attn, x_attn))
}

// ==================== DiTBlock (MMDiT standard) ====================

struct DiTBlock {
    norm1: LayerNormNoAffine,
    attn: AttnProjections,
    norm2: LayerNormNoAffine,
    mlp: Mlp,
    ada_ln_modulation_1: Linear,
}

impl DiTBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        has_qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = AttnProjections::new(hidden_size, num_heads, has_qk_norm, vb.pp("attn"))?;
        let norm2 = LayerNormNoAffine::new(1e-6);
        let mlp_ratio = 4;
        let mlp = Mlp::new(hidden_size, hidden_size * mlp_ratio, vb.pp("mlp"))?;
        let n_mods = 6;
        let ada_ln_modulation_1 = quantized_nn::linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
        )?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            ada_ln_modulation_1,
        })
    }

    fn pre_attention(
        &self,
        x: &Tensor,
        c: &Tensor,
    ) -> candle_core::Result<(Qkv, ModulateIntermediates)> {
        let modulation = c.silu()?.apply(&self.ada_ln_modulation_1)?;
        let chunks = modulation.chunk(6, D::Minus1)?;
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            chunks[0].clone(),
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
            chunks[4].clone(),
            chunks[5].clone(),
        );

        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        let qkv = self.attn.pre_attention(&modulated_x)?;

        Ok((
            qkv,
            ModulateIntermediates {
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            },
        ))
    }

    fn post_attention(
        &self,
        attn: &Tensor,
        x: &Tensor,
        mod_interm: &ModulateIntermediates,
    ) -> candle_core::Result<Tensor> {
        let attn_out = self.attn.post_attention(attn)?;

        // Trace post_attention on first call
        static PA_DIAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let do_diag = !PA_DIAG.swap(true, std::sync::atomic::Ordering::Relaxed)
            && std::env::var_os("MOLD_SD3_DEBUG").is_some();
        macro_rules! trace {
            ($name:expr, $t:expr) => {
                if do_diag {
                    let mn = $t.min_all().and_then(|v| v.to_dtype(DType::F32)?.to_scalar::<f32>()).unwrap_or(f32::NAN);
                    let mx = $t.max_all().and_then(|v| v.to_dtype(DType::F32)?.to_scalar::<f32>()).unwrap_or(f32::NAN);
                    eprintln!("[sd3-post] {}: [{mn:.4},{mx:.4}]", $name);
                }
            }
        }

        trace!("attn_proj", &attn_out);
        let gated_attn = attn_out.broadcast_mul(&mod_interm.gate_msa.unsqueeze(1)?)?;
        trace!("gated_attn", &gated_attn);
        let x = x.broadcast_add(&gated_attn)?;
        trace!("x+gated", &x);

        let norm_x = self.norm2.forward(&x)?;
        trace!("norm_x", &norm_x);
        let modulated_x = modulate(&norm_x, &mod_interm.shift_mlp, &mod_interm.scale_mlp)?;
        trace!("modulated", &modulated_x);
        let mlp_out = self.mlp.forward(&modulated_x)?;
        trace!("mlp_out", &mlp_out);
        let gated_mlp = mlp_out.broadcast_mul(&mod_interm.gate_mlp.unsqueeze(1)?)?;
        trace!("gated_mlp", &gated_mlp);
        let result = x.broadcast_add(&gated_mlp)?;
        trace!("final", &result);
        Ok(result)
    }
}

struct ModulateIntermediates {
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
}

// ==================== SelfAttnDiTBlock (MMDiT-X) ====================

struct SelfAttnDiTBlock {
    norm1: LayerNormNoAffine,
    attn: AttnProjections,
    attn2: AttnProjections,
    norm2: LayerNormNoAffine,
    mlp: Mlp,
    ada_ln_modulation_1: Linear,
}

struct SelfAttnModulateIntermediates {
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
    gate_msa2: Tensor,
}

impl SelfAttnDiTBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        has_qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = AttnProjections::new(hidden_size, num_heads, has_qk_norm, vb.pp("attn"))?;
        let attn2 = AttnProjections::new(hidden_size, num_heads, has_qk_norm, vb.pp("attn2"))?;
        let norm2 = LayerNormNoAffine::new(1e-6);
        let mlp_ratio = 4;
        let mlp = Mlp::new(hidden_size, hidden_size * mlp_ratio, vb.pp("mlp"))?;
        let n_mods = 9;
        let ada_ln_modulation_1 = quantized_nn::linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
        )?;
        Ok(Self {
            norm1,
            attn,
            attn2,
            norm2,
            mlp,
            ada_ln_modulation_1,
        })
    }

    fn pre_attention(
        &self,
        x: &Tensor,
        c: &Tensor,
    ) -> candle_core::Result<(Qkv, Qkv, SelfAttnModulateIntermediates)> {
        let modulation = c.silu()?.apply(&self.ada_ln_modulation_1)?;
        let chunks = modulation.chunk(9, D::Minus1)?;
        let (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = (
            chunks[0].clone(),
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
            chunks[4].clone(),
            chunks[5].clone(),
            chunks[6].clone(),
            chunks[7].clone(),
            chunks[8].clone(),
        );

        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        let qkv = self.attn.pre_attention(&modulated_x)?;

        let modulated_x2 = modulate(&norm_x, &shift_msa2, &scale_msa2)?;
        let qkv2 = self.attn2.pre_attention(&modulated_x2)?;

        Ok((
            qkv,
            qkv2,
            SelfAttnModulateIntermediates {
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                gate_msa2,
            },
        ))
    }

    fn post_attention(
        &self,
        attn: &Tensor,
        attn2: &Tensor,
        x: &Tensor,
        mod_interm: &SelfAttnModulateIntermediates,
    ) -> candle_core::Result<Tensor> {
        let attn_out = self.attn.post_attention(attn)?;
        let x = x.broadcast_add(&attn_out.broadcast_mul(&mod_interm.gate_msa.unsqueeze(1)?)?)?;
        let attn_out2 = self.attn2.post_attention(attn2)?;
        let x = x.broadcast_add(&attn_out2.broadcast_mul(&mod_interm.gate_msa2.unsqueeze(1)?)?)?;

        let norm_x = self.norm2.forward(&x)?;
        let modulated_x = modulate(&norm_x, &mod_interm.shift_mlp, &mod_interm.scale_mlp)?;
        let mlp_out = self.mlp.forward(&modulated_x)?;
        x.broadcast_add(&mlp_out.broadcast_mul(&mod_interm.gate_mlp.unsqueeze(1)?)?)
    }
}

// ==================== QkvOnlyDiTBlock (final joint block context) ====================

struct QkvOnlyDiTBlock {
    norm1: LayerNormNoAffine,
    attn: QkvOnlyAttnProjections,
    ada_ln_modulation_1: Linear,
}

impl QkvOnlyDiTBlock {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = QkvOnlyAttnProjections::new(hidden_size, num_heads, vb.pp("attn"))?;
        let n_mods = 2;
        let ada_ln_modulation_1 = quantized_nn::linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
        )?;
        Ok(Self {
            norm1,
            attn,
            ada_ln_modulation_1,
        })
    }

    fn pre_attention(&self, x: &Tensor, c: &Tensor) -> candle_core::Result<Qkv> {
        let modulation = c.silu()?.apply(&self.ada_ln_modulation_1)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (shift_msa, scale_msa) = (chunks[0].clone(), chunks[1].clone());
        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        self.attn.pre_attention(&modulated_x)
    }
}

// ==================== FinalLayer ====================

struct FinalLayer {
    norm_final: LayerNormNoAffine,
    linear: Linear,
    ada_ln_modulation_1: Linear,
}

impl FinalLayer {
    fn new(
        hidden_size: usize,
        patch_size: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm_final = LayerNormNoAffine::new(1e-6);
        let linear = quantized_nn::linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            vb.pp("linear"),
        )?;
        let ada_ln_modulation_1 =
            quantized_nn::linear(hidden_size, 2 * hidden_size, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation_1,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> candle_core::Result<Tensor> {
        let modulation = c.silu()?.apply(&self.ada_ln_modulation_1)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (shift, scale) = (chunks[0].clone(), chunks[1].clone());
        let norm_x = self.norm_final.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift, &scale)?;
        self.linear.forward(&modulated_x)
    }
}

// ==================== Unpatchifier ====================

struct Unpatchifier {
    patch_size: usize,
    out_channels: usize,
}

impl Unpatchifier {
    fn new(patch_size: usize, out_channels: usize) -> Self {
        Self {
            patch_size,
            out_channels,
        }
    }

    fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> candle_core::Result<Tensor> {
        let h = (h + 1) / self.patch_size;
        let w = (w + 1) / self.patch_size;
        let x = x.reshape((
            x.dim(0)?,
            h,
            w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        ))?;
        let x = x.permute((0, 5, 1, 3, 2, 4))?; // "nhwpqc->nchpwq"
        x.reshape((
            x.dim(0)?,
            self.out_channels,
            self.patch_size * h,
            self.patch_size * w,
        ))
    }
}

// ==================== JointBlock trait + implementations ====================

trait JointBlock {
    fn forward(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
        num_heads: usize,
    ) -> candle_core::Result<(Tensor, Tensor)>;
}

/// Standard MMDiT joint block (SD3.5 Large uses these exclusively).
struct MMDiTJointBlock {
    x_block: DiTBlock,
    context_block: DiTBlock,
}

impl MMDiTJointBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        has_qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let x_block = DiTBlock::new(hidden_size, num_heads, has_qk_norm, vb.pp("x_block"))?;
        let context_block =
            DiTBlock::new(hidden_size, num_heads, has_qk_norm, vb.pp("context_block"))?;
        Ok(Self {
            x_block,
            context_block,
        })
    }
}

impl JointBlock for MMDiTJointBlock {
    fn forward(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
        num_heads: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (context_qkv, context_interm) = self.context_block.pre_attention(context, c)?;
        let (x_qkv, x_interm) = self.x_block.pre_attention(x, c)?;

        // Trace block internals on first call
        static BLK_DIAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let do_diag = !BLK_DIAG.swap(true, std::sync::atomic::Ordering::Relaxed)
            && std::env::var_os("MOLD_SD3_DEBUG").is_some();
        if do_diag {
            let stats = |name: &str, t: &Tensor| -> String {
                let mn = t.min_all().and_then(|v| v.to_dtype(DType::F32)?.to_scalar::<f32>()).unwrap_or(f32::NAN);
                let mx = t.max_all().and_then(|v| v.to_dtype(DType::F32)?.to_scalar::<f32>()).unwrap_or(f32::NAN);
                format!("{name}=[{mn:.4},{mx:.4}]")
            };
            eprintln!("[sd3-blk0] pre_attn: {} {} {} {} gate_msa={} gate_mlp={}",
                stats("x_q", &x_qkv.q), stats("x_k", &x_qkv.k),
                stats("ctx_q", &context_qkv.q), stats("ctx_k", &context_qkv.k),
                stats("x_gate", &x_interm.gate_msa), stats("x_gmlp", &x_interm.gate_mlp));
        }

        let (context_attn, x_attn) = joint_attn(&context_qkv, &x_qkv, num_heads)?;

        if do_diag {
            let mn = |t: &Tensor| t.min_all().and_then(|v| v.to_dtype(DType::F32)?.to_scalar::<f32>()).unwrap_or(f32::NAN);
            let mx = |t: &Tensor| t.max_all().and_then(|v| v.to_dtype(DType::F32)?.to_scalar::<f32>()).unwrap_or(f32::NAN);
            eprintln!("[sd3-blk0] attn out: x=[{:.4},{:.4}] ctx=[{:.4},{:.4}]",
                mn(&x_attn), mx(&x_attn), mn(&context_attn), mx(&context_attn));
        }

        let context_out =
            self.context_block
                .post_attention(&context_attn, context, &context_interm)?;
        let x_out = self.x_block.post_attention(&x_attn, x, &x_interm)?;

        if do_diag {
            let mn = |t: &Tensor| t.min_all().and_then(|v| v.to_dtype(DType::F32)?.to_scalar::<f32>()).unwrap_or(f32::NAN);
            let mx = |t: &Tensor| t.max_all().and_then(|v| v.to_dtype(DType::F32)?.to_scalar::<f32>()).unwrap_or(f32::NAN);
            eprintln!("[sd3-blk0] post_attn: x=[{:.4},{:.4}] ctx=[{:.4},{:.4}]",
                mn(&x_out), mx(&x_out), mn(&context_out), mx(&context_out));
        }

        Ok((context_out, x_out))
    }
}

/// MMDiT-X joint block (SD3.5 Medium uses these with self-attention on x).
struct MMDiTXJointBlock {
    x_block: SelfAttnDiTBlock,
    context_block: DiTBlock,
}

impl MMDiTXJointBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        has_qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let x_block = SelfAttnDiTBlock::new(hidden_size, num_heads, has_qk_norm, vb.pp("x_block"))?;
        let context_block =
            DiTBlock::new(hidden_size, num_heads, has_qk_norm, vb.pp("context_block"))?;
        Ok(Self {
            x_block,
            context_block,
        })
    }
}

impl JointBlock for MMDiTXJointBlock {
    fn forward(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
        num_heads: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (context_qkv, context_interm) = self.context_block.pre_attention(context, c)?;
        let (x_qkv, x_qkv2, x_interm) = self.x_block.pre_attention(x, c)?;
        let (context_attn, x_attn) = joint_attn(&context_qkv, &x_qkv, num_heads)?;
        let x_attn2 = attention(&x_qkv2.q, &x_qkv2.k, &x_qkv2.v, num_heads)?;
        let context_out =
            self.context_block
                .post_attention(&context_attn, context, &context_interm)?;
        let x_out = self
            .x_block
            .post_attention(&x_attn, &x_attn2, x, &x_interm)?;
        Ok((context_out, x_out))
    }
}

// ==================== QuantizedMMDiT (main struct) ====================

/// Quantized MMDiT model for SD3.5 inference with GGUF weights.
pub(crate) struct QuantizedMMDiT {
    patch_embedder: PatchEmbedder,
    pos_embedder: PositionEmbedder,
    timestep_embedder: TimestepEmbedder,
    vector_embedder: VectorEmbedder,
    context_embedder: Linear,
    joint_blocks: Vec<Box<dyn JointBlock + Send + Sync>>,
    context_qkv_only_block: ContextQkvOnlyBlock,
    final_layer: FinalLayer,
    unpatchifier: Unpatchifier,
    num_heads: usize,
}

/// The last joint block where context only produces QKV (no MLP).
struct ContextQkvOnlyBlock {
    x_block: DiTBlock,
    context_block: QkvOnlyDiTBlock,
}

impl ContextQkvOnlyBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        has_qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let x_block = DiTBlock::new(hidden_size, num_heads, has_qk_norm, vb.pp("x_block"))?;
        let context_block = QkvOnlyDiTBlock::new(hidden_size, num_heads, vb.pp("context_block"))?;
        Ok(Self {
            x_block,
            context_block,
        })
    }
}

impl QuantizedMMDiT {
    /// Load a quantized MMDiT from GGUF weights.
    ///
    /// GGUF files from city96 use unprefixed tensor names (e.g. `x_embedder.proj.weight`),
    /// so `vb` should NOT have a `model.diffusion_model` prefix.
    pub fn new(cfg: &MMDiTConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.head_size * cfg.depth;
        let num_heads = cfg.depth;

        let patch_embedder = PatchEmbedder::new(
            cfg.patch_size,
            cfg.in_channels,
            hidden_size,
            vb.pp("x_embedder"),
        )?;
        let pos_embedder = PositionEmbedder::new(
            hidden_size,
            cfg.patch_size,
            cfg.pos_embed_max_size,
            vb.clone(),
        )?;
        let timestep_embedder = TimestepEmbedder::new(
            hidden_size,
            cfg.frequency_embedding_size,
            vb.pp("t_embedder"),
        )?;
        let vector_embedder =
            VectorEmbedder::new(cfg.adm_in_channels, hidden_size, vb.pp("y_embedder"))?;
        let context_embedder = quantized_nn::linear(
            cfg.context_embed_size,
            hidden_size,
            vb.pp("context_embedder"),
        )?;

        // Detect MMDiT vs MMDiT-X blocks by checking for attn2 weights
        let mut joint_blocks: Vec<Box<dyn JointBlock + Send + Sync>> =
            Vec::with_capacity(cfg.depth - 1);
        for i in 0..cfg.depth - 1 {
            let block_vb = vb.pp(format!("joint_blocks.{i}"));
            // Check if this block has attn2 (MMDiT-X) by trying to find the weight
            let has_attn2 = block_vb
                .pp("x_block")
                .pp("attn2")
                .pp("qkv")
                .get(1, "weight") // probe for existence
                .is_ok();
            // Check for QK norm by probing ln_k (head_dim shape)
            let head_dim = hidden_size / num_heads;
            let has_qk_norm = block_vb
                .pp("x_block")
                .pp("attn")
                .pp("ln_k")
                .get(head_dim, "weight")
                .is_ok();
            let block: Box<dyn JointBlock + Send + Sync> = if has_attn2 {
                Box::new(MMDiTXJointBlock::new(
                    hidden_size,
                    num_heads,
                    has_qk_norm,
                    block_vb,
                )?)
            } else {
                Box::new(MMDiTJointBlock::new(
                    hidden_size,
                    num_heads,
                    has_qk_norm,
                    block_vb,
                )?)
            };
            joint_blocks.push(block);
        }

        // Check for QK norm on the final block
        let final_block_vb = vb.pp(format!("joint_blocks.{}", cfg.depth - 1));
        let head_dim = hidden_size / num_heads;
        let final_has_qk_norm = final_block_vb
            .pp("x_block")
            .pp("attn")
            .pp("ln_k")
            .get(head_dim, "weight")
            .is_ok();

        let context_qkv_only_block =
            ContextQkvOnlyBlock::new(hidden_size, num_heads, final_has_qk_norm, final_block_vb)?;

        let final_layer = FinalLayer::new(
            hidden_size,
            cfg.patch_size,
            cfg.out_channels,
            vb.pp("final_layer"),
        )?;

        let unpatchifier = Unpatchifier::new(cfg.patch_size, cfg.out_channels);

        Ok(Self {
            patch_embedder,
            pos_embedder,
            timestep_embedder,
            vector_embedder,
            context_embedder,
            joint_blocks,
            context_qkv_only_block,
            final_layer,
            unpatchifier,
            num_heads,
        })
    }

    /// Forward pass through the quantized MMDiT.
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        // Quantized model operates in F32 (QMatMul dequantizes weights to F32)
        let x = &x.to_dtype(DType::F32)?;
        let t = &t.to_dtype(DType::F32)?;
        let y = &y.to_dtype(DType::F32)?;
        let context = &context.to_dtype(DType::F32)?;

        let h = x.dim(D::Minus2)?;
        let w = x.dim(D::Minus1)?;
        let cropped_pos_embed = self.pos_embedder.get_cropped_pos_embed(h, w)?;
        let x = self
            .patch_embedder
            .forward(x)?
            .broadcast_add(&cropped_pos_embed)?;
        let c = self.timestep_embedder.forward(t)?;
        let y = self.vector_embedder.forward(y)?;
        let c = (c + y)?;
        let context = self.context_embedder.forward(context)?;

        // Joint blocks
        let (mut context, mut x) = (context, x);
        for (i, joint_block) in self.joint_blocks.iter().enumerate() {
            if let Some(skip) = &skip_layers {
                if skip.contains(&i) {
                    continue;
                }
            }
            let result = joint_block.forward(&context, &x, &c, self.num_heads)?;
            context = result.0;
            x = result.1;
            // One-shot diagnostic for first forward pass
            if i == 0 && std::env::var_os("MOLD_SD3_DEBUG").is_some() {
                let xmin = x.min_all()?.to_dtype(DType::F32)?.to_scalar::<f32>().unwrap_or(f32::NAN);
                let xmax = x.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>().unwrap_or(f32::NAN);
                eprintln!("[sd3-debug] block 0 output: x=[{xmin:.4},{xmax:.4}]");
            }
        }

        // Final context QKV only block
        let context_qkv = self
            .context_qkv_only_block
            .context_block
            .pre_attention(&context, &c)?;
        let (x_qkv, x_interm) = self.context_qkv_only_block.x_block.pre_attention(&x, &c)?;
        let (_, x_attn) = joint_attn(&context_qkv, &x_qkv, self.num_heads)?;
        let x = self
            .context_qkv_only_block
            .x_block
            .post_attention(&x_attn, &x, &x_interm)?;

        let x = self.final_layer.forward(&x, &c)?;
        let x = self.unpatchifier.unpatchify(&x, h, w)?;
        Ok(x.narrow(2, 0, h)?.narrow(3, 0, w)?)
    }
}
