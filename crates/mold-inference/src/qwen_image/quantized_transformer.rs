//! Quantized (GGUF) Qwen-Image transformer.

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::RmsNorm as CandleRmsNorm;
use candle_transformers::models::z_image::transformer::{
    apply_rotary_emb, create_coordinate_grid, RopeEmbedder,
};
use candle_transformers::quantized_nn::{self, Linear};
use candle_transformers::quantized_var_builder::VarBuilder;

use super::transformer::{QwenImageConfig, MAX_PERIOD};

const FREQUENCY_EMBEDDING_SIZE: usize = 256;

/// Apply a quantized linear layer and replace non-finite (NaN/inf) values with 0.
/// Candle's CUDA QMatMul produces sporadic NaN and inf in some output elements
/// when processing large-magnitude inputs. This wrapper prevents propagation.
fn linear_safe(linear: &Linear, x: &Tensor) -> Result<Tensor> {
    let out = linear.forward(x)?;
    // Detect non-finite: NaN (ne self) or inf (abs > finite max)
    let is_nan = out.ne(&out)?;
    let abs_out = out.abs()?;
    let threshold = Tensor::new(f32::MAX, out.device())?.broadcast_as(abs_out.shape())?;
    let is_inf = abs_out.gt(&threshold)?;
    // Combined mask: NaN or inf
    let bad = (is_nan.to_dtype(DType::U8)? + is_inf.to_dtype(DType::U8)?)?.gt(&Tensor::zeros_like(&is_nan)?.to_dtype(DType::U8)?)?;
    let zero = Tensor::zeros_like(&out)?;
    bad.where_cond(&zero, &out)
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
            d => d,
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

struct TimestepProjEmbeddings {
    linear1: Linear,
    linear2: Linear,
}

impl TimestepProjEmbeddings {
    fn new(inner_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: quantized_nn::linear(
                FREQUENCY_EMBEDDING_SIZE,
                inner_dim,
                vb.pp("time_text_embed")
                    .pp("timestep_embedder")
                    .pp("linear_1"),
            )?,
            linear2: quantized_nn::linear(
                inner_dim,
                inner_dim,
                vb.pp("time_text_embed")
                    .pp("timestep_embedder")
                    .pp("linear_2"),
            )?,
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
    proj: Linear,
}

impl ApproximateGelu {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: quantized_nn::linear(dim, hidden_dim, vb.pp("proj"))?,
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
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act: ApproximateGelu::new(dim, hidden_dim, vb.pp("net").pp("0"))?,
            out: quantized_nn::linear(hidden_dim, dim, vb.pp("net").pp("2"))?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.act.forward(x)?.apply(&self.out)
    }
}

struct QkNorm {
    norm_q: CandleRmsNorm,
    norm_k: CandleRmsNorm,
}

impl QkNorm {
    fn new(head_dim: usize, eps: f64, vb: VarBuilder, q_name: &str, k_name: &str) -> Result<Self> {
        let norm_q_w = vb
            .pp(q_name)
            .get(head_dim, "weight")?
            .dequantize(vb.device())?;
        let norm_k_w = vb
            .pp(k_name)
            .get(head_dim, "weight")?
            .dequantize(vb.device())?;
        Ok(Self {
            norm_q: CandleRmsNorm::new(norm_q_w, eps),
            norm_k: CandleRmsNorm::new(norm_k_w, eps),
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
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim;
        let qkv_dim = cfg.num_attention_heads * cfg.attention_head_dim;
        Ok(Self {
            to_q: quantized_nn::linear(dim, qkv_dim, vb.pp("to_q"))?,
            to_k: quantized_nn::linear(dim, qkv_dim, vb.pp("to_k"))?,
            to_v: quantized_nn::linear(dim, qkv_dim, vb.pp("to_v"))?,
            to_out: quantized_nn::linear(qkv_dim, dim, vb.pp("to_out").pp("0"))?,
            add_q_proj: quantized_nn::linear(dim, qkv_dim, vb.pp("add_q_proj"))?,
            add_k_proj: quantized_nn::linear(dim, qkv_dim, vb.pp("add_k_proj"))?,
            add_v_proj: quantized_nn::linear(dim, qkv_dim, vb.pp("add_v_proj"))?,
            add_out_proj: quantized_nn::linear(qkv_dim, dim, vb.pp("to_add_out"))?,
            qk_norm: QkNorm::new(cfg.attention_head_dim, 1e-6, vb.clone(), "norm_q", "norm_k")?,
            added_qk_norm: QkNorm::new(
                cfg.attention_head_dim,
                1e-6,
                vb.clone(),
                "norm_added_q",
                "norm_added_k",
            )?,
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
        let (b, _, _) = img_hidden.dims3()?;
        let txt_seq_len = txt_hidden.dim(1)?;

        let q_img = linear_safe(&self.to_q, img_hidden)?
            .reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let k_img = linear_safe(&self.to_k, img_hidden)?
            .reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let v_img = linear_safe(&self.to_v, img_hidden)?
            .reshape((b, img_seq_len, self.n_heads, self.head_dim))?;

        let q_txt = linear_safe(&self.add_q_proj, txt_hidden)?
            .reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;
        let k_txt = linear_safe(&self.add_k_proj, txt_hidden)?
            .reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;
        let v_txt = linear_safe(&self.add_v_proj, txt_hidden)?
            .reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;

        let (q_img, k_img) = self.qk_norm.forward(&q_img, &k_img)?;
        let (q_txt, k_txt) = self.added_qk_norm.forward(&q_txt, &k_txt)?;

        let q_img = apply_rotary_emb(&q_img, img_cos, img_sin)?;
        let k_img = apply_rotary_emb(&k_img, img_cos, img_sin)?;
        let q_txt = apply_rotary_emb(&q_txt, txt_cos, txt_sin)?;
        let k_txt = apply_rotary_emb(&k_txt, txt_cos, txt_sin)?;

        // Concatenate in [text, image] order (matches diffusers QwenDoubleStreamAttnProcessor2_0)
        let q = Tensor::cat(&[&q_txt, &q_img], 1)?;
        let k = Tensor::cat(&[&k_txt, &k_img], 1)?;
        let v = Tensor::cat(&[&v_txt, &v_img], 1)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Attention with key masking for text padding (matches diffusers)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let img_mask = Tensor::ones((b, img_seq_len), DType::U8, img_hidden.device())?;
        // Key mask order: [text, image] to match concatenation
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

        let total_seq = img_seq_len + txt_seq_len;
        let attn = attn.transpose(1, 2)?.reshape((b, total_seq, ()))?;
        // Split in [text, image] order
        let txt_attn = attn.narrow(1, 0, txt_seq_len)?;
        let img_attn = attn.narrow(1, txt_seq_len, img_seq_len)?;

        let txt_out = linear_safe(&self.add_out_proj, &txt_attn)?;

        Ok((linear_safe(&self.to_out, &img_attn)?, txt_out))
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
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim;
        let mlp_dim = dim * 4;
        Ok(Self {
            img_norm1: LayerNormNoParams::new(1e-6),
            img_norm2: LayerNormNoParams::new(1e-6),
            txt_norm1: LayerNormNoParams::new(1e-6),
            txt_norm2: LayerNormNoParams::new(1e-6),
            attn: JointAttention::new(cfg, vb.pp("attn"))?,
            img_mlp: FeedForward::new(dim, mlp_dim, vb.pp("img_mlp"))?,
            txt_mlp: FeedForward::new(dim, mlp_dim, vb.pp("txt_mlp"))?,
            img_mod: quantized_nn::linear(dim, 6 * dim, vb.pp("img_mod").pp("1"))?,
            txt_mod: quantized_nn::linear(dim, 6 * dim, vb.pp("txt_mod").pp("1"))?,
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
        let img_mod = linear_safe(&self.img_mod, &temb.silu()?)?.unsqueeze(1)?;
        let txt_mod = linear_safe(&self.txt_mod, &temb.silu()?)?.unsqueeze(1)?;
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
        // BF16 roundtrip on gated residuals — matches diffusers' BF16 precision.
        // QMatMul requires F32 but the model was trained in BF16; rounding after each
        // residual prevents activation growth that occurs in pure F32.
        let img_hidden = (img_hidden + img_gate_msa.broadcast_mul(&img_attn)?)?
            .to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        let txt_hidden = (txt_hidden + txt_gate_msa.broadcast_mul(&txt_attn)?)?
            .to_dtype(DType::BF16)?.to_dtype(DType::F32)?;

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
        let img_hidden = (img_hidden + img_gate_mlp.broadcast_mul(&self.img_mlp.forward(&img_mlp_in)?)?)?
            .to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        let txt_hidden = (txt_hidden + txt_gate_mlp.broadcast_mul(&self.txt_mlp.forward(&txt_mlp_in)?)?)?
            .to_dtype(DType::BF16)?.to_dtype(DType::F32)?;

        Ok((img_hidden, txt_hidden))
    }
}

struct OutputLayer {
    norm_final: LayerNormNoParams,
    adaln_linear: candle_nn::Linear,
    linear: candle_nn::Linear,
}

impl OutputLayer {
    fn new(
        inner_dim: usize,
        out_channels: usize,
        patch_size: usize,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let output_dim = patch_size * patch_size * out_channels;
        // Dequantize to standard F32 linear — these weights are BF16 in the GGUF
        // and candle's QMatMul produces NaN for certain F32 input × BF16 weight combos.
        let adaln_vb = vb.pp("norm_out").pp("linear");
        let adaln_w = adaln_vb.get((2 * inner_dim, inner_dim), "weight")?.dequantize(device)?;
        let adaln_b = adaln_vb.get(2 * inner_dim, "bias")?.dequantize(device)?;

        let proj_vb = vb.pp("proj_out");
        let proj_w = proj_vb.get((output_dim, inner_dim), "weight")?.dequantize(device)?;
        let proj_b = proj_vb.get(output_dim, "bias")?.dequantize(device)?;

        Ok(Self {
            norm_final: LayerNormNoParams::new(1e-6),
            adaln_linear: candle_nn::Linear::new(adaln_w, Some(adaln_b)),
            linear: candle_nn::Linear::new(proj_w, Some(proj_b)),
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let mod_params = temb.silu()?.apply(&self.adaln_linear)?;
        let chunks = mod_params.chunk(2, D::Minus1)?;
        // AdaLayerNormContinuous: scale = chunk[0], shift = chunk[1]
        // (opposite of block-level modulation which uses shift, scale, gate order)
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
    img_in: candle_nn::Linear,
    txt_in: Linear,
    txt_norm: quantized_nn::RmsNorm,
    blocks: Vec<QwenImageTransformerBlock>,
    rope_embedder: RopeEmbedder,
    output_layer: OutputLayer,
    cfg: QwenImageConfig,
}

impl QuantizedQwenImageTransformer2DModel {
    pub fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(QwenImageTransformerBlock::new(
                cfg,
                vb.pp("transformer_blocks").pp(i),
            )?);
        }
        // Dequantize img_in to standard F32 linear — its weight is BF16 in the GGUF
        // and candle's QMatMul produces NaN for certain F32 input ranges with BF16 weights.
        // img_in is small (64→3072 = 768KB) so the VRAM cost is negligible.
        let img_in_w = vb.pp("img_in").get((cfg.inner_dim, cfg.in_channels), "weight")?
            .dequantize(&device)?;
        let img_in_b = vb.pp("img_in").get(cfg.inner_dim, "bias")?
            .dequantize(&device)?;
        let img_in = candle_nn::Linear::new(img_in_w, Some(img_in_b));

        Ok(Self {
            time_embed: TimestepProjEmbeddings::new(cfg.inner_dim, vb.clone())?,
            img_in,
            txt_in: quantized_nn::linear(cfg.joint_attention_dim, cfg.inner_dim, vb.pp("txt_in"))?,
            txt_norm: quantized_nn::RmsNorm::new(
                cfg.joint_attention_dim,
                cfg.norm_eps,
                vb.pp("txt_norm"),
            )?,
            blocks,
            rope_embedder: RopeEmbedder::new(
                10000.0,
                cfg.axes_dims_rope.clone(),
                vec![2048, 2048, 2048],
                &device,
                DType::F32,
            )?,
            output_layer: OutputLayer::new(cfg.inner_dim, cfg.out_channels, cfg.patch_size, vb, &device)?,
            cfg: cfg.clone(),
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
        // QMatMul requires F32 input — cast here, BF16 roundtrip between blocks
        // prevents activation growth (matching diffusers' BF16 precision).
        let x = x.to_dtype(DType::F32)?;
        let t = t.to_dtype(DType::F32)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(DType::F32)?;
        let encoder_attention_mask = encoder_attention_mask.to_device(device)?;

        let (b, c, h, w) = x.dims4()?;
        let patch_size = self.cfg.patch_size;
        let temb = self.time_embed.forward(&t)?;

        // Pack latents: [B, C, H, W] → [B, (H/p)*(W/p), C*p*p]
        // Matches diffusers' _pack_latents — simple reshape, NOT Conv3d patchify
        let hp = h / patch_size;
        let wp = w / patch_size;
        let x_packed = x
            .reshape((b, c, hp, patch_size, wp, patch_size))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((b, hp * wp, c * patch_size * patch_size))?
            .contiguous()?;

        let mut img = x_packed.apply(&self.img_in)?;

        // Note: we do NOT mask padding text tokens here.
        let txt_normed = self.txt_norm.forward(&encoder_hidden_states)?;
        let mut txt = txt_normed.apply(&self.txt_in)?;

        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let img_pos_ids = create_coordinate_grid((1, h_tokens, w_tokens), (0, 0, 0), device)?;
        let (img_cos, img_sin) = self.rope_embedder.forward(&img_pos_ids)?;
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let txt_offset = h_tokens.max(w_tokens) as u32;
        let mut txt_coords = Vec::with_capacity(txt_seq_len * 3);
        for i in 0..txt_seq_len {
            let pos = txt_offset + i as u32;
            txt_coords.push(pos);
            txt_coords.push(pos);
            txt_coords.push(pos);
        }
        let txt_pos_ids = Tensor::from_vec(txt_coords, (txt_seq_len, 3), device)?;
        let (txt_cos, txt_sin) = self.rope_embedder.forward(&txt_pos_ids)?;

        for block in &self.blocks {
            let (new_img, new_txt) = block.forward(
                &img,
                &txt,
                &encoder_attention_mask,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
            )?;
            img = new_img;
            txt = new_txt;
        }

        let img_out = self.output_layer.forward(&img, &temb)?;

        // Unpack latents: [B, (H/p)*(W/p), out_channels*p*p] → [B, out_channels, H, W]
        // Matches diffusers' _unpack_latents — simple reshape, NOT Conv3d unpatchify
        let out_c = self.cfg.out_channels;
        let x_out = img_out
            .reshape((b, hp, wp, out_c, patch_size, patch_size))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((b, out_c, h, w))?
            .contiguous()?;
        x_out.to_dtype(out_dtype)
    }
}
