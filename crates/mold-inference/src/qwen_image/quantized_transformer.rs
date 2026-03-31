//! GGUF Qwen-Image transformer with block-level CPU<->GPU offloading.
//!
//! The GGUF weights are dequantized to BF16 on CPU at load time. Stem and
//! output layers stay resident on the GPU, while transformer blocks remain on
//! CPU and are streamed to the GPU one at a time during forward.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Linear, RmsNorm};
use candle_transformers::models::z_image::transformer::apply_rotary_emb;
use candle_transformers::quantized_var_builder::VarBuilder;

use super::transformer::{QwenImageConfig, MAX_PERIOD};

const FREQUENCY_EMBEDDING_SIZE: usize = 256;
const ROPE_CACHE_LEN: usize = 4096;

fn dequant_tensor(vb: &VarBuilder, name: &str, dtype: DType, target: &Device) -> Result<Tensor> {
    vb.get_no_shape(name)?
        .dequantize(vb.device())?
        .to_dtype(dtype)?
        .to_device(target)
}

fn dequant_linear(vb: &VarBuilder, name: &str, dtype: DType, target: &Device) -> Result<Linear> {
    let weight = dequant_tensor(vb, &format!("{name}.weight"), dtype, target)?;
    let bias = match vb.get_no_shape(&format!("{name}.bias")) {
        Ok(bias) => Some(
            bias.dequantize(vb.device())?
                .to_dtype(dtype)?
                .to_device(target)?,
        ),
        Err(_) => None,
    };
    Ok(Linear::new(weight, bias))
}

fn linear_to_device(linear: &Linear, target: &Device) -> Result<Linear> {
    let weight = linear.weight().to_device(target)?;
    let bias = linear
        .bias()
        .map(|bias| bias.to_device(target))
        .transpose()?;
    Ok(Linear::new(weight, bias))
}

fn rms_norm_to_device(norm: &RmsNorm, eps: f64, target: &Device) -> Result<RmsNorm> {
    let inner = norm.clone().into_inner();
    Ok(RmsNorm::new(inner.weight().to_device(target)?, eps))
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

struct QwenRopeEmbedder {
    axes_dims: Vec<usize>,
    axis_half_dims: Vec<usize>,
    axis_offsets: Vec<usize>,
    pos_cos: Tensor,
    pos_sin: Tensor,
    neg_cos: Tensor,
    neg_sin: Tensor,
    dtype: DType,
}

impl QwenRopeEmbedder {
    fn new(theta: f64, axes_dims: Vec<usize>, cpu_device: &Device, dtype: DType) -> Result<Self> {
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

    fn forward(
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

        Ok((
            self.to_target(img_cos, device)?,
            self.to_target(img_sin, device)?,
            self.to_target(txt_cos, device)?,
            self.to_target(txt_sin, device)?,
        ))
    }
}

struct TimestepProjEmbeddings {
    linear1: Linear,
    linear2: Linear,
}

impl TimestepProjEmbeddings {
    fn new(inner_dim: usize, vb: VarBuilder, dtype: DType, target: &Device) -> Result<Self> {
        let vb = vb.pp("time_text_embed").pp("timestep_embedder");
        let _ = inner_dim;
        Ok(Self {
            linear1: dequant_linear(&vb, "linear_1", dtype, target)?,
            linear2: dequant_linear(&vb, "linear_2", dtype, target)?,
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
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?
            .to_dtype(self.linear1.weight().dtype())?;
        embedding.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

struct ApproximateGelu {
    proj: Linear,
}

impl ApproximateGelu {
    fn new(vb: VarBuilder, dtype: DType, target: &Device) -> Result<Self> {
        Ok(Self {
            proj: dequant_linear(&vb, "proj", dtype, target)?,
        })
    }

    fn to_device(&self, target: &Device) -> Result<Self> {
        Ok(Self {
            proj: linear_to_device(&self.proj, target)?,
        })
    }
}

impl Module for ApproximateGelu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.apply(&self.proj)?;
        x.broadcast_mul(&candle_nn::ops::sigmoid(&(x.clone() * 1.702)?)?)
    }
}

struct FeedForward {
    act: ApproximateGelu,
    out: Linear,
}

impl FeedForward {
    fn new(vb: VarBuilder, dtype: DType, target: &Device) -> Result<Self> {
        Ok(Self {
            act: ApproximateGelu::new(vb.pp("net").pp("0"), dtype, target)?,
            out: dequant_linear(&vb.pp("net"), "2", dtype, target)?,
        })
    }

    fn to_device(&self, target: &Device) -> Result<Self> {
        Ok(Self {
            act: self.act.to_device(target)?,
            out: linear_to_device(&self.out, target)?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.act.forward(x)?.apply(&self.out)
    }
}

struct QkNorm {
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    eps: f64,
}

impl QkNorm {
    fn new(
        head_dim: usize,
        eps: f64,
        vb: VarBuilder,
        q_name: &str,
        k_name: &str,
        dtype: DType,
        target: &Device,
    ) -> Result<Self> {
        let _ = head_dim;
        Ok(Self {
            norm_q: RmsNorm::new(
                dequant_tensor(&vb.pp(q_name), "weight", dtype, target)?,
                eps,
            ),
            norm_k: RmsNorm::new(
                dequant_tensor(&vb.pp(k_name), "weight", dtype, target)?,
                eps,
            ),
            eps,
        })
    }

    fn to_device(&self, target: &Device) -> Result<Self> {
        Ok(Self {
            norm_q: rms_norm_to_device(&self.norm_q, self.eps, target)?,
            norm_k: rms_norm_to_device(&self.norm_k, self.eps, target)?,
            eps: self.eps,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.norm_q.forward(q)?, self.norm_k.forward(k)?))
    }
}

struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    add_out_proj: Linear,
    qk_norm: QkNorm,
    added_qk_norm: QkNorm,
    n_heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder, dtype: DType, target: &Device) -> Result<Self> {
        let dim = cfg.inner_dim;
        let qkv_dim = cfg.num_attention_heads * cfg.attention_head_dim;
        let _ = qkv_dim;
        let _ = dim;
        Ok(Self {
            to_q: dequant_linear(&vb, "to_q", dtype, target)?,
            to_k: dequant_linear(&vb, "to_k", dtype, target)?,
            to_v: dequant_linear(&vb, "to_v", dtype, target)?,
            to_out: dequant_linear(&vb.pp("to_out"), "0", dtype, target)?,
            add_q_proj: dequant_linear(&vb, "add_q_proj", dtype, target)?,
            add_k_proj: dequant_linear(&vb, "add_k_proj", dtype, target)?,
            add_v_proj: dequant_linear(&vb, "add_v_proj", dtype, target)?,
            add_out_proj: dequant_linear(&vb, "to_add_out", dtype, target)?,
            qk_norm: QkNorm::new(
                cfg.attention_head_dim,
                1e-6,
                vb.clone(),
                "norm_q",
                "norm_k",
                dtype,
                target,
            )?,
            added_qk_norm: QkNorm::new(
                cfg.attention_head_dim,
                1e-6,
                vb.clone(),
                "norm_added_q",
                "norm_added_k",
                dtype,
                target,
            )?,
            n_heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
        })
    }

    fn to_device(&self, target: &Device) -> Result<Self> {
        Ok(Self {
            to_q: linear_to_device(&self.to_q, target)?,
            to_k: linear_to_device(&self.to_k, target)?,
            to_v: linear_to_device(&self.to_v, target)?,
            to_out: linear_to_device(&self.to_out, target)?,
            add_q_proj: linear_to_device(&self.add_q_proj, target)?,
            add_k_proj: linear_to_device(&self.add_k_proj, target)?,
            add_v_proj: linear_to_device(&self.add_v_proj, target)?,
            add_out_proj: linear_to_device(&self.add_out_proj, target)?,
            qk_norm: self.qk_norm.to_device(target)?,
            added_qk_norm: self.added_qk_norm.to_device(target)?,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
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
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let img_mask = Tensor::ones((batch, img_seq_len), DType::U8, img_hidden.device())?;
        let key_mask = Tensor::cat(&[txt_mask, &img_mask], 1)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let on_true = key_mask.zeros_like()?.to_dtype(attn_weights.dtype())?;
        let on_false = Tensor::new(f32::NEG_INFINITY, attn_weights.device())?
            .broadcast_as(key_mask.shape())?
            .to_dtype(attn_weights.dtype())?;
        let key_mask = key_mask.where_cond(&on_true, &on_false)?;
        attn_weights = attn_weights.broadcast_add(&key_mask)?;
        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn = attn_weights.matmul(&v)?;

        let total_seq_len = img_seq_len + txt_seq_len;
        let attn = attn.transpose(1, 2)?.reshape((batch, total_seq_len, ()))?;
        let txt_attn = attn.narrow(1, 0, txt_seq_len)?;
        let img_attn = attn.narrow(1, txt_seq_len, img_seq_len)?;

        Ok((
            img_attn.apply(&self.to_out)?,
            txt_attn.apply(&self.add_out_proj)?,
        ))
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
    img_mod: Linear,
    txt_mod: Linear,
}

impl QwenImageTransformerBlock {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder, dtype: DType, target: &Device) -> Result<Self> {
        let _ = cfg.inner_dim;
        Ok(Self {
            img_norm1: LayerNormNoParams::new(1e-6),
            img_norm2: LayerNormNoParams::new(1e-6),
            txt_norm1: LayerNormNoParams::new(1e-6),
            txt_norm2: LayerNormNoParams::new(1e-6),
            attn: JointAttention::new(cfg, vb.pp("attn"), dtype, target)?,
            img_mlp: FeedForward::new(vb.pp("img_mlp"), dtype, target)?,
            txt_mlp: FeedForward::new(vb.pp("txt_mlp"), dtype, target)?,
            img_mod: dequant_linear(&vb.pp("img_mod"), "1", dtype, target)?,
            txt_mod: dequant_linear(&vb.pp("txt_mod"), "1", dtype, target)?,
        })
    }

    fn to_device(&self, target: &Device) -> Result<Self> {
        Ok(Self {
            img_norm1: self.img_norm1.clone(),
            img_norm2: self.img_norm2.clone(),
            txt_norm1: self.txt_norm1.clone(),
            txt_norm2: self.txt_norm2.clone(),
            attn: self.attn.to_device(target)?,
            img_mlp: self.img_mlp.to_device(target)?,
            txt_mlp: self.txt_mlp.to_device(target)?,
            img_mod: linear_to_device(&self.img_mod, target)?,
            txt_mod: linear_to_device(&self.txt_mod, target)?,
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
        let img_hidden =
            (&img_hidden + img_gate_mlp.broadcast_mul(&self.img_mlp.forward(&img_mlp_in)?)?)?;
        let txt_hidden =
            (&txt_hidden + txt_gate_mlp.broadcast_mul(&self.txt_mlp.forward(&txt_mlp_in)?)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

struct OutputLayer {
    norm_final: LayerNormNoParams,
    adaln_linear: Linear,
    linear: Linear,
}

impl OutputLayer {
    fn new(
        inner_dim: usize,
        out_channels: usize,
        patch_size: usize,
        vb: VarBuilder,
        dtype: DType,
        target: &Device,
    ) -> Result<Self> {
        let _ = (inner_dim, out_channels, patch_size);
        Ok(Self {
            norm_final: LayerNormNoParams::new(1e-6),
            adaln_linear: dequant_linear(&vb.pp("norm_out"), "linear", dtype, target)?,
            linear: dequant_linear(&vb, "proj_out", dtype, target)?,
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
    img_in: Linear,
    txt_in: Linear,
    txt_norm: RmsNorm,
    blocks: Vec<QwenImageTransformerBlock>,
    rope_embedder: QwenRopeEmbedder,
    output_layer: OutputLayer,
    cfg: QwenImageConfig,
    gpu_device: Device,
    dtype: DType,
}

impl QuantizedQwenImageTransformer2DModel {
    pub fn new(
        cfg: &QwenImageConfig,
        vb: VarBuilder,
        cpu_device: &Device,
        gpu_device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let time_embed = TimestepProjEmbeddings::new(cfg.inner_dim, vb.clone(), dtype, gpu_device)?;
        let img_in = dequant_linear(&vb, "img_in", dtype, gpu_device)?;
        let txt_in = dequant_linear(&vb, "txt_in", dtype, gpu_device)?;
        let txt_norm = RmsNorm::new(
            dequant_tensor(&vb.pp("txt_norm"), "weight", dtype, gpu_device)?,
            cfg.norm_eps,
        );

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for i in 0..cfg.num_layers {
            blocks.push(QwenImageTransformerBlock::new(
                cfg,
                vb_blocks.pp(i),
                dtype,
                cpu_device,
            )?);
        }

        let rope_embedder =
            QwenRopeEmbedder::new(10000.0, cfg.axes_dims_rope.clone(), cpu_device, dtype)?;
        let output_layer = OutputLayer::new(
            cfg.inner_dim,
            cfg.out_channels,
            cfg.patch_size,
            vb,
            dtype,
            gpu_device,
        )?;

        Ok(Self {
            time_embed,
            img_in,
            txt_in,
            txt_norm,
            blocks,
            rope_embedder,
            output_layer,
            cfg: cfg.clone(),
            gpu_device: gpu_device.clone(),
            dtype,
        })
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
        let x = if x.dtype() == self.dtype {
            x.clone()
        } else {
            x.to_dtype(self.dtype)?
        };
        let t = if t.dtype() == self.dtype {
            t.clone()
        } else {
            t.to_dtype(self.dtype)?
        };
        let encoder_hidden_states = if encoder_hidden_states.dtype() == self.dtype {
            encoder_hidden_states.clone()
        } else {
            encoder_hidden_states.to_dtype(self.dtype)?
        };
        let encoder_attention_mask = encoder_attention_mask.to_device(device)?;

        let (batch, channels, height, width) = x.dims4()?;
        let patch_size = self.cfg.patch_size;
        let temb = self.time_embed.forward(&t)?;

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
        let txt_normed = self.txt_norm.forward(&encoder_hidden_states)?;
        let mut txt = txt_normed.apply(&self.txt_in)?;

        let h_tokens = height / patch_size;
        let w_tokens = width / patch_size;
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_embedder
                .forward(1, h_tokens, w_tokens, txt_seq_len, device)?;

        for block in &self.blocks {
            let gpu_block = block.to_device(&self.gpu_device)?;
            (img, txt) = gpu_block.forward(
                &img,
                &txt,
                &encoder_attention_mask,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
            )?;
            self.gpu_device.synchronize()?;
            drop(gpu_block);
        }

        let img_out = self.output_layer.forward(&img, &temb)?;
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
