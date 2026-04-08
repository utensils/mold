//! Quantized (GGUF) Qwen-Image transformer with device-specific linear dispatch.
//!
//! **CUDA**: either keeps GGUF weights on GPU or loads them CPU-staged for
//! low-VRAM offload, dequantizing each linear layer to BF16 per forward call.
//! All computation stays in BF16 matching the model's training dtype.
//!
//! **Metal**: uses candle's `QMatMul`-backed `Linear` which avoids per-forward
//! full dequantization (faster on Metal). Computation in F32 since Metal's QMatMul
//! dequantizes to F32 internally.

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_transformers::models::z_image::transformer::apply_rotary_emb;
use candle_transformers::quantized_nn::Linear as QMatMulLinear;
use candle_transformers::quantized_var_builder::VarBuilder;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::transformer::{QwenImageConfig, MAX_PERIOD};

const FREQUENCY_EMBEDDING_SIZE: usize = 256;
pub(crate) const ROPE_CACHE_LEN: usize = 4096;

/// CUDA: BF16 (matches training dtype, halves activation memory vs F32).
/// Metal: F32 (QMatMul dequantizes to F32 internally on Metal).
fn working_dtype(device: &Device) -> DType {
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

fn debug_stage(stage: &str) {
    if std::env::var_os("MOLD_QWEN_DEBUG").is_some() {
        eprintln!("[qwen-quantized] {stage}");
    }
}

/// Device-dispatched quantized linear layer.
///
/// CUDA: dequantizes weight to BF16 per forward (temporary ~72MB peak).
/// Metal: uses QMatMul (weight stays quantized, dequant inside kernel).
enum QwenLinear {
    /// Per-forward BF16 dequantization — correct dtype for CUDA.
    Dequant {
        weight: Arc<QTensor>,
        bias: Option<Tensor>,
    },
    /// QMatMul-backed — avoids full dequant, faster on Metal.
    QMatMul(QMatMulLinear),
}

impl Module for QwenLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dequant { weight, bias } => {
                let dtype = working_dtype(x.device());
                let x = x.to_dtype(dtype)?;
                let w = if weight.device().is_cpu() && !x.device().is_cpu() {
                    weight
                        .dequantize(&Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_device(x.device())?
                } else {
                    weight.dequantize(x.device())?.to_dtype(dtype)?
                };
                let bias = bias
                    .as_ref()
                    .map(|b| b.to_device(x.device())?.to_dtype(dtype))
                    .transpose()?;
                candle_nn::Linear::new(w, bias).forward(&x)
            }
            Self::QMatMul(inner) => inner.forward(x),
        }
    }
}

fn qlinear(vb: &VarBuilder, name: &str) -> Result<QwenLinear> {
    let vb = vb.pp(name);
    let weight = vb.get_no_shape("weight")?;
    let dtype = working_dtype(vb.device());
    let bias = match vb.get_no_shape("bias") {
        Ok(b) => Some(b.dequantize(vb.device())?.to_dtype(dtype)?),
        Err(_) => None,
    };
    if vb.device().is_metal() {
        // F32 bias matches QMatMul's F32 output on Metal.
        Ok(QwenLinear::QMatMul(QMatMulLinear::from_arc(weight, bias)?))
    } else {
        // BF16 bias for CUDA dequant path.
        Ok(QwenLinear::Dequant { weight, bias })
    }
}

#[derive(Debug, Clone)]
struct DynamicRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl DynamicRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let weight = self.weight.to_device(xs.device())?.to_dtype(dtype)?;
        xs.broadcast_mul(&weight)
    }
}

/// Dequantize a small 1D weight vector for RmsNorm (norm weights are tiny).
fn dequant_rms_norm(vb: &VarBuilder, name: &str, eps: f64) -> Result<DynamicRmsNorm> {
    let dtype = working_dtype(vb.device());
    let weight = vb
        .pp(name)
        .get_no_shape("weight")?
        .dequantize(vb.device())?
        .to_dtype(dtype)?;
    Ok(DynamicRmsNorm { weight, eps })
}

#[derive(Debug, Clone)]
struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoParams {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            dtype => dtype,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RopeCacheDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RopeCacheKey {
    frame: usize,
    height: usize,
    width: usize,
    max_txt_seq_len: usize,
    device: RopeCacheDevice,
}

impl RopeCacheDevice {
    fn from_device(device: &Device) -> Self {
        if device.is_cuda() {
            Self::Cuda
        } else if device.is_metal() {
            Self::Metal
        } else {
            Self::Cpu
        }
    }
}

type RopeCacheValue = (Tensor, Tensor, Tensor, Tensor);

#[derive(Debug)]
pub(crate) struct QwenRopeEmbedder {
    axes_dims: Vec<usize>,
    axis_half_dims: Vec<usize>,
    axis_offsets: Vec<usize>,
    pos_cos: Tensor,
    pos_sin: Tensor,
    neg_cos: Tensor,
    neg_sin: Tensor,
    dtype: DType,
    cache: Mutex<HashMap<RopeCacheKey, RopeCacheValue>>,
}

impl Clone for QwenRopeEmbedder {
    fn clone(&self) -> Self {
        Self {
            axes_dims: self.axes_dims.clone(),
            axis_half_dims: self.axis_half_dims.clone(),
            axis_offsets: self.axis_offsets.clone(),
            pos_cos: self.pos_cos.clone(),
            pos_sin: self.pos_sin.clone(),
            neg_cos: self.neg_cos.clone(),
            neg_sin: self.neg_sin.clone(),
            dtype: self.dtype,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl QwenRopeEmbedder {
    pub(crate) fn new(
        theta: f64,
        axes_dims: Vec<usize>,
        cpu_device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut axis_half_dims = Vec::with_capacity(axes_dims.len());
        let mut axis_offsets = Vec::with_capacity(axes_dims.len());
        let mut running_offset = 0;
        for &dim in &axes_dims {
            if dim % 2 != 0 {
                candle_core::bail!("Qwen RoPE axis dim {dim} must be even");
            }
            axis_offsets.push(running_offset);
            let half_dim = dim / 2;
            axis_half_dims.push(half_dim);
            running_offset += half_dim;
        }

        let pos_index: Vec<f32> = (0..ROPE_CACHE_LEN).map(|i| i as f32).collect();
        let neg_index: Vec<f32> = (0..ROPE_CACHE_LEN)
            .rev()
            .map(|i| -(i as i32) as f32 - 1.0)
            .collect();

        let mut pos_cos_parts = Vec::with_capacity(axes_dims.len());
        let mut pos_sin_parts = Vec::with_capacity(axes_dims.len());
        let mut neg_cos_parts = Vec::with_capacity(axes_dims.len());
        let mut neg_sin_parts = Vec::with_capacity(axes_dims.len());
        for &dim in &axes_dims {
            let (pos_cos, pos_sin) = Self::rope_params(&pos_index, dim, theta, cpu_device)?;
            let (neg_cos, neg_sin) = Self::rope_params(&neg_index, dim, theta, cpu_device)?;
            pos_cos_parts.push(pos_cos);
            pos_sin_parts.push(pos_sin);
            neg_cos_parts.push(neg_cos);
            neg_sin_parts.push(neg_sin);
        }

        Ok(Self {
            axes_dims,
            axis_half_dims,
            axis_offsets,
            pos_cos: Tensor::cat(&pos_cos_parts, D::Minus1)?,
            pos_sin: Tensor::cat(&pos_sin_parts, D::Minus1)?,
            neg_cos: Tensor::cat(&neg_cos_parts, D::Minus1)?,
            neg_sin: Tensor::cat(&neg_sin_parts, D::Minus1)?,
            dtype,
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn rope_params(
        index: &[f32],
        dim: usize,
        theta: f64,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|k| 1.0 / (theta as f32).powf(k as f32 / dim as f32))
            .collect();
        let index = Tensor::from_vec(index.to_vec(), index.len(), device)?;
        let inv_freq = Tensor::from_vec(inv_freq, dim / 2, device)?;
        let freqs = index.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    fn axis_slice(&self, table: &Tensor, axis: usize) -> Result<Tensor> {
        table.narrow(1, self.axis_offsets[axis], self.axis_half_dims[axis])
    }

    fn leading_axis_freqs(&self, table: &Tensor, axis: usize, len: usize) -> Result<Tensor> {
        if len > ROPE_CACHE_LEN {
            candle_core::bail!("Qwen RoPE length {len} exceeds cache size {ROPE_CACHE_LEN}");
        }
        self.axis_slice(table, axis)?.narrow(0, 0, len)
    }

    fn centered_axis_freqs(
        &self,
        pos_table: &Tensor,
        neg_table: &Tensor,
        axis: usize,
        len: usize,
    ) -> Result<Tensor> {
        if len > ROPE_CACHE_LEN {
            candle_core::bail!("Qwen RoPE length {len} exceeds cache size {ROPE_CACHE_LEN}");
        }
        let pos_len = len / 2;
        let neg_len = len - pos_len;
        let pos_axis = self.axis_slice(pos_table, axis)?;
        let neg_axis = self.axis_slice(neg_table, axis)?;
        match (neg_len, pos_len) {
            (0, _) => pos_axis.narrow(0, 0, pos_len),
            (_, 0) => neg_axis.narrow(0, ROPE_CACHE_LEN - neg_len, neg_len),
            _ => Tensor::cat(
                &[
                    neg_axis.narrow(0, ROPE_CACHE_LEN - neg_len, neg_len)?,
                    pos_axis.narrow(0, 0, pos_len)?,
                ],
                0,
            ),
        }
    }

    fn to_target(&self, tensor: Tensor, device: &Device) -> Result<Tensor> {
        tensor.to_device(device)?.to_dtype(self.dtype)
    }

    pub(crate) fn forward(
        &self,
        frame: usize,
        height: usize,
        width: usize,
        max_txt_seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        if self.axes_dims.len() != 3 {
            candle_core::bail!(
                "Qwen RoPE expects exactly 3 axes, got {}",
                self.axes_dims.len()
            );
        }

        let key = RopeCacheKey {
            frame,
            height,
            width,
            max_txt_seq_len,
            device: RopeCacheDevice::from_device(device),
        };
        if let Some(cached) = self.cache.lock().unwrap().get(&key) {
            return Ok(cached.clone());
        }

        let frame_cos = self.leading_axis_freqs(&self.pos_cos, 0, frame)?;
        let frame_sin = self.leading_axis_freqs(&self.pos_sin, 0, frame)?;
        let height_cos = self.centered_axis_freqs(&self.pos_cos, &self.neg_cos, 1, height)?;
        let height_sin = self.centered_axis_freqs(&self.pos_sin, &self.neg_sin, 1, height)?;
        let width_cos = self.centered_axis_freqs(&self.pos_cos, &self.neg_cos, 2, width)?;
        let width_sin = self.centered_axis_freqs(&self.pos_sin, &self.neg_sin, 2, width)?;

        let frame_half = self.axis_half_dims[0];
        let height_half = self.axis_half_dims[1];
        let width_half = self.axis_half_dims[2];
        let total_half = frame_half + height_half + width_half;
        let seq_len = frame * height * width;

        let img_cos = Tensor::cat(
            &[
                frame_cos
                    .reshape((frame, 1, 1, frame_half))?
                    .expand((frame, height, width, frame_half))?,
                height_cos.reshape((1, height, 1, height_half))?.expand((
                    frame,
                    height,
                    width,
                    height_half,
                ))?,
                width_cos
                    .reshape((1, 1, width, width_half))?
                    .expand((frame, height, width, width_half))?,
            ],
            D::Minus1,
        )?
        .reshape((seq_len, total_half))?;
        let img_sin = Tensor::cat(
            &[
                frame_sin
                    .reshape((frame, 1, 1, frame_half))?
                    .expand((frame, height, width, frame_half))?,
                height_sin.reshape((1, height, 1, height_half))?.expand((
                    frame,
                    height,
                    width,
                    height_half,
                ))?,
                width_sin
                    .reshape((1, 1, width, width_half))?
                    .expand((frame, height, width, width_half))?,
            ],
            D::Minus1,
        )?
        .reshape((seq_len, total_half))?;

        let max_vid_index = (height / 2).max(width / 2);
        if max_vid_index + max_txt_seq_len > ROPE_CACHE_LEN {
            candle_core::bail!(
                "Qwen text RoPE slice [{}..{}) exceeds cache size {}",
                max_vid_index,
                max_vid_index + max_txt_seq_len,
                ROPE_CACHE_LEN
            );
        }
        let txt_cos = self.pos_cos.narrow(0, max_vid_index, max_txt_seq_len)?;
        let txt_sin = self.pos_sin.narrow(0, max_vid_index, max_txt_seq_len)?;

        let value = (
            self.to_target(img_cos, device)?,
            self.to_target(img_sin, device)?,
            self.to_target(txt_cos, device)?,
            self.to_target(txt_sin, device)?,
        );
        self.cache.lock().unwrap().insert(key, value.clone());
        Ok(value)
    }
}

struct TimestepProjEmbeddings {
    linear1: QwenLinear,
    linear2: QwenLinear,
}

impl TimestepProjEmbeddings {
    fn new(vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("time_text_embed").pp("timestep_embedder");
        Ok(Self {
            linear1: qlinear(&vb, "linear_1")?,
            linear2: qlinear(&vb, "linear_2")?,
        })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let half = FREQUENCY_EMBEDDING_SIZE / 2;
        let freqs = Tensor::arange(0u32, half as u32, t.device())?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        embedding.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

struct ApproximateGelu {
    proj: QwenLinear,
}

impl ApproximateGelu {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: qlinear(&vb, "proj")?,
        })
    }
}

impl Module for ApproximateGelu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.proj)?
            .apply(&candle_nn::Activation::GeluPytorchTanh)
    }
}

struct FeedForward {
    act: ApproximateGelu,
    out: QwenLinear,
}

impl FeedForward {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act: ApproximateGelu::new(vb.pp("net").pp("0"))?,
            out: qlinear(&vb.pp("net"), "2")?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.act.forward(x)?.apply(&self.out)
    }
}

struct QkNorm {
    norm_q: DynamicRmsNorm,
    norm_k: DynamicRmsNorm,
}

impl QkNorm {
    fn new(eps: f64, vb: &VarBuilder, q_name: &str, k_name: &str) -> Result<Self> {
        Ok(Self {
            norm_q: dequant_rms_norm(vb, q_name, eps)?,
            norm_k: dequant_rms_norm(vb, k_name, eps)?,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.norm_q.forward(q)?, self.norm_k.forward(k)?))
    }
}

struct JointAttention {
    to_q: QwenLinear,
    to_k: QwenLinear,
    to_v: QwenLinear,
    to_out: QwenLinear,
    add_q_proj: QwenLinear,
    add_k_proj: QwenLinear,
    add_v_proj: QwenLinear,
    add_out_proj: QwenLinear,
    qk_norm: QkNorm,
    added_qk_norm: QkNorm,
    n_heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            to_q: qlinear(&vb, "to_q")?,
            to_k: qlinear(&vb, "to_k")?,
            to_v: qlinear(&vb, "to_v")?,
            to_out: qlinear(&vb.pp("to_out"), "0")?,
            add_q_proj: qlinear(&vb, "add_q_proj")?,
            add_k_proj: qlinear(&vb, "add_k_proj")?,
            add_v_proj: qlinear(&vb, "add_v_proj")?,
            add_out_proj: qlinear(&vb, "to_add_out")?,
            qk_norm: QkNorm::new(1e-6, &vb, "norm_q", "norm_k")?,
            added_qk_norm: QkNorm::new(1e-6, &vb, "norm_added_q", "norm_added_k")?,
            n_heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img_hidden: &Tensor,
        txt_hidden: &Tensor,
        txt_mask: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        img_seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _, _) = img_hidden.dims3()?;
        let txt_seq_len = txt_hidden.dim(1)?;

        let q_img = img_hidden.apply(&self.to_q)?.reshape((
            batch,
            img_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let k_img = img_hidden.apply(&self.to_k)?.reshape((
            batch,
            img_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let v_img = img_hidden.apply(&self.to_v)?.reshape((
            batch,
            img_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;

        let q_txt = txt_hidden.apply(&self.add_q_proj)?.reshape((
            batch,
            txt_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let k_txt = txt_hidden.apply(&self.add_k_proj)?.reshape((
            batch,
            txt_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let v_txt = txt_hidden.apply(&self.add_v_proj)?.reshape((
            batch,
            txt_seq_len,
            self.n_heads,
            self.head_dim,
        ))?;

        let (q_img, k_img) = self.qk_norm.forward(&q_img, &k_img)?;
        let (q_txt, k_txt) = self.added_qk_norm.forward(&q_txt, &k_txt)?;

        let q_img = apply_rotary_emb(&q_img, img_cos, img_sin)?;
        let k_img = apply_rotary_emb(&k_img, img_cos, img_sin)?;
        let q_txt = apply_rotary_emb(&q_txt, txt_cos, txt_sin)?;
        let k_txt = apply_rotary_emb(&k_txt, txt_cos, txt_sin)?;

        let q = Tensor::cat(&[&q_txt, &q_img], 1)?;
        let k = Tensor::cat(&[&k_txt, &k_img], 1)?;
        let v = Tensor::cat(&[&v_txt, &v_img], 1)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let img_mask = Tensor::ones((batch, img_seq_len), DType::U8, img_hidden.device())?;
        let key_mask = Tensor::cat(&[txt_mask, &img_mask], 1)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let attn = self.attention_dispatch(&q, &k, &v, &key_mask, scale)?;

        let total_seq_len = img_seq_len + txt_seq_len;
        let attn = attn.transpose(1, 2)?.reshape((batch, total_seq_len, ()))?;
        // contiguous() required: narrow() creates non-contiguous strided views;
        // candle's matmul expects contiguous input on CUDA.
        let txt_attn = attn.narrow(1, 0, txt_seq_len)?.contiguous()?;
        let img_attn = attn.narrow(1, txt_seq_len, img_seq_len)?.contiguous()?;

        Ok((
            img_attn.apply(&self.to_out)?,
            txt_attn.apply(&self.add_out_proj)?,
        ))
    }

    fn attention_dispatch(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        key_mask: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        if q.device().is_metal() {
            let sdpa_mask = self.prepare_sdpa_mask(key_mask, q)?;
            candle_nn::ops::sdpa(q, k, v, Some(&sdpa_mask), false, scale as f32, 1.0)
        } else {
            self.attention_basic(q, k, v, key_mask, scale)
        }
    }

    fn attention_basic(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        key_mask: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let on_true = key_mask.zeros_like()?.to_dtype(attn_weights.dtype())?;
        let on_false = Tensor::new(f32::NEG_INFINITY, attn_weights.device())?
            .broadcast_as(key_mask.shape())?
            .to_dtype(attn_weights.dtype())?;
        let key_mask = key_mask.where_cond(&on_true, &on_false)?;
        attn_weights = attn_weights.broadcast_add(&key_mask)?;
        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_weights.matmul(v)
    }

    fn prepare_sdpa_mask(&self, key_mask: &Tensor, q: &Tensor) -> Result<Tensor> {
        let (batch, _, seq_len, _) = q.dims4()?;
        let key_len = key_mask.dim(3)?;
        let mask = key_mask.to_dtype(q.dtype())?;
        let mask = ((mask - 1.0)? * 1e9)?;
        mask.broadcast_as((batch, self.n_heads, seq_len, key_len))?
            .contiguous()
    }
}

struct QwenImageTransformerBlock {
    img_norm1: LayerNormNoParams,
    img_norm2: LayerNormNoParams,
    txt_norm1: LayerNormNoParams,
    txt_norm2: LayerNormNoParams,
    attn: JointAttention,
    img_mlp: FeedForward,
    txt_mlp: FeedForward,
    img_mod: QwenLinear,
    txt_mod: QwenLinear,
}

impl QwenImageTransformerBlock {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            img_norm1: LayerNormNoParams::new(1e-6),
            img_norm2: LayerNormNoParams::new(1e-6),
            txt_norm1: LayerNormNoParams::new(1e-6),
            txt_norm2: LayerNormNoParams::new(1e-6),
            attn: JointAttention::new(cfg, vb.pp("attn"))?,
            img_mlp: FeedForward::new(vb.pp("img_mlp"))?,
            txt_mlp: FeedForward::new(vb.pp("txt_mlp"))?,
            img_mod: qlinear(&vb.pp("img_mod"), "1")?,
            txt_mod: qlinear(&vb.pp("txt_mod"), "1")?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img_hidden: &Tensor,
        txt_hidden: &Tensor,
        txt_mask: &Tensor,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let img_seq_len = img_hidden.dim(1)?;
        let temb = temb.silu()?;
        let img_mod = temb.apply(&self.img_mod)?.unsqueeze(1)?;
        let txt_mod = temb.apply(&self.txt_mod)?.unsqueeze(1)?;
        let img_chunks = img_mod.chunk(6, D::Minus1)?;
        let txt_chunks = txt_mod.chunk(6, D::Minus1)?;
        let (
            img_shift_msa,
            img_scale_msa,
            img_gate_msa,
            img_shift_mlp,
            img_scale_mlp,
            img_gate_mlp,
        ) = (
            &img_chunks[0],
            &img_chunks[1],
            &img_chunks[2],
            &img_chunks[3],
            &img_chunks[4],
            &img_chunks[5],
        );
        let (
            txt_shift_msa,
            txt_scale_msa,
            txt_gate_msa,
            txt_shift_mlp,
            txt_scale_mlp,
            txt_gate_mlp,
        ) = (
            &txt_chunks[0],
            &txt_chunks[1],
            &txt_chunks[2],
            &txt_chunks[3],
            &txt_chunks[4],
            &txt_chunks[5],
        );

        let img_attn_in = self
            .img_norm1
            .forward(img_hidden)?
            .broadcast_mul(&(img_scale_msa + 1.0)?)?
            .broadcast_add(img_shift_msa)?;
        let txt_attn_in = self
            .txt_norm1
            .forward(txt_hidden)?
            .broadcast_mul(&(txt_scale_msa + 1.0)?)?
            .broadcast_add(txt_shift_msa)?;
        let (img_attn, txt_attn) = self.attn.forward(
            &img_attn_in,
            &txt_attn_in,
            txt_mask,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            img_seq_len,
        )?;

        let img_hidden = (img_hidden + img_gate_msa.broadcast_mul(&img_attn)?)?;
        let txt_hidden = (txt_hidden + txt_gate_msa.broadcast_mul(&txt_attn)?)?;

        let img_mlp_in = self
            .img_norm2
            .forward(&img_hidden)?
            .broadcast_mul(&(img_scale_mlp + 1.0)?)?
            .broadcast_add(img_shift_mlp)?;
        let txt_mlp_in = self
            .txt_norm2
            .forward(&txt_hidden)?
            .broadcast_mul(&(txt_scale_mlp + 1.0)?)?
            .broadcast_add(txt_shift_mlp)?;
        let img_ff = self.img_mlp.forward(&img_mlp_in)?;

        let img_hidden = (&img_hidden + img_gate_mlp.broadcast_mul(&img_ff)?)?;
        let txt_hidden =
            (&txt_hidden + txt_gate_mlp.broadcast_mul(&self.txt_mlp.forward(&txt_mlp_in)?)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

struct OutputLayer {
    norm_final: LayerNormNoParams,
    adaln_linear: QwenLinear,
    linear: QwenLinear,
}

impl OutputLayer {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_final: LayerNormNoParams::new(1e-6),
            adaln_linear: qlinear(&vb.pp("norm_out"), "linear")?,
            linear: qlinear(&vb, "proj_out")?,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let mod_params = temb.silu()?.apply(&self.adaln_linear)?;
        let chunks = mod_params.chunk(2, D::Minus1)?;
        let scale = chunks[0].unsqueeze(1)?;
        let shift = chunks[1].unsqueeze(1)?;
        let x = self
            .norm_final
            .forward(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        x.apply(&self.linear)
    }
}

pub(crate) struct QuantizedQwenImageTransformer2DModel {
    time_embed: TimestepProjEmbeddings,
    img_in: QwenLinear,
    txt_in: QwenLinear,
    txt_norm: DynamicRmsNorm,
    blocks: Vec<QwenImageTransformerBlock>,
    rope_embedder: QwenRopeEmbedder,
    output_layer: OutputLayer,
    cfg: QwenImageConfig,
    supports_cfg_batching: bool,
}

impl QuantizedQwenImageTransformer2DModel {
    pub fn new(
        cfg: &QwenImageConfig,
        vb: VarBuilder,
        device: &Device,
        supports_cfg_batching: bool,
    ) -> Result<Self> {
        let time_embed = TimestepProjEmbeddings::new(vb.clone())?;
        let img_in = qlinear(&vb, "img_in")?;
        let txt_in = qlinear(&vb, "txt_in")?;
        let txt_norm = dequant_rms_norm(&vb, "txt_norm", cfg.norm_eps)?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for i in 0..cfg.num_layers {
            blocks.push(QwenImageTransformerBlock::new(cfg, vb_blocks.pp(i))?);
        }

        // RoPE source tables stay on CPU; device-local views are cached by shape.
        let rope_dtype = working_dtype(device);
        let rope_embedder = QwenRopeEmbedder::new(
            10000.0,
            cfg.axes_dims_rope.clone(),
            &Device::Cpu,
            rope_dtype,
        )?;
        let output_layer = OutputLayer::new(vb)?;

        Ok(Self {
            time_embed,
            img_in,
            txt_in,
            txt_norm,
            blocks,
            rope_embedder,
            output_layer,
            cfg: cfg.clone(),
            supports_cfg_batching,
        })
    }

    pub fn supports_cfg_batching(&self) -> bool {
        self.supports_cfg_batching
    }

    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let out_dtype = x.dtype();
        let device = x.device();

        // CUDA: BF16 (matches training dtype, halves activation memory).
        // Metal/CPU: F32 (QMatMul dequantizes to F32 internally on Metal).
        let dtype = working_dtype(device);
        let x = x.to_dtype(dtype)?;
        let t = t.to_dtype(dtype)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(dtype)?;
        let encoder_attention_mask = encoder_attention_mask.to_device(device)?;
        debug_stage("inputs prepared");

        let (batch, channels, height, width) = x.dims4()?;
        let patch_size = self.cfg.patch_size;
        let temb = self.time_embed.forward(&t)?;
        debug_stage("time embedding");

        let height_patches = height / patch_size;
        let width_patches = width / patch_size;
        let x_packed = x
            .reshape((
                batch,
                channels,
                height_patches,
                patch_size,
                width_patches,
                patch_size,
            ))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((
                batch,
                height_patches * width_patches,
                channels * patch_size * patch_size,
            ))?
            .contiguous()?;

        let mut img = x_packed.apply(&self.img_in)?;
        debug_stage("image stem");
        let txt_normed = self.txt_norm.forward(&encoder_hidden_states)?;
        let mut txt = txt_normed.apply(&self.txt_in)?;
        debug_stage("text stem");

        let h_tokens = height / patch_size;
        let w_tokens = width / patch_size;
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_embedder
                .forward(1, h_tokens, w_tokens, txt_seq_len, device)?;
        debug_stage("rope");

        for (i, block) in self.blocks.iter().enumerate() {
            (img, txt) = block.forward(
                &img,
                &txt,
                &encoder_attention_mask,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
            )?;
            if i == 0 || i + 1 == self.blocks.len() {
                debug_stage(&format!("block {}", i + 1));
            }
        }

        let img_out = self.output_layer.forward(&img, &temb)?;
        debug_stage("output layer");
        let out_channels = self.cfg.out_channels;
        let x_out = img_out
            .reshape((
                batch,
                height_patches,
                width_patches,
                out_channels,
                patch_size,
                patch_size,
            ))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((batch, out_channels, height, width))?
            .contiguous()?;

        x_out.to_dtype(out_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::{RopeCacheDevice, RopeCacheKey};
    use candle_core::Device;

    #[test]
    fn rope_cache_device_detects_cpu() {
        assert!(matches!(
            RopeCacheDevice::from_device(&Device::Cpu),
            RopeCacheDevice::Cpu
        ));
    }

    #[test]
    fn rope_cache_key_includes_shape_and_device() {
        let a = RopeCacheKey {
            frame: 1,
            height: 64,
            width: 64,
            max_txt_seq_len: 512,
            device: RopeCacheDevice::Cpu,
        };
        let b = RopeCacheKey {
            frame: 1,
            height: 64,
            width: 64,
            max_txt_seq_len: 512,
            device: RopeCacheDevice::Metal,
        };
        assert_ne!(a, b);
    }
}
