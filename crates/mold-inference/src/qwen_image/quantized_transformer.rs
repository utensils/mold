//! Quantized (GGUF) Qwen-Image transformer.

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::RmsNorm as CandleRmsNorm;
use candle_transformers::models::z_image::transformer::{
    apply_rotary_emb, create_coordinate_grid, patchify, unpatchify, RopeEmbedder,
};
use candle_transformers::quantized_nn::{self, Linear};
use candle_transformers::quantized_var_builder::VarBuilder;

use super::transformer::{QwenImageConfig, MAX_PERIOD};

const FREQUENCY_EMBEDDING_SIZE: usize = 256;

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
                vb.pp("time_text_embed").pp("timestep_embedder").pp("linear_1"),
            )?,
            linear2: quantized_nn::linear(
                inner_dim,
                inner_dim,
                vb.pp("time_text_embed").pp("timestep_embedder").pp("linear_2"),
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
        x.apply(&self.act)?.apply(&self.out)
    }
}

struct QkNorm {
    norm_q: CandleRmsNorm,
    norm_k: CandleRmsNorm,
}

impl QkNorm {
    fn new(head_dim: usize, eps: f64, vb: VarBuilder, q_name: &str, k_name: &str) -> Result<Self> {
        let norm_q_w = vb.pp(q_name).get(head_dim, "weight")?.dequantize(vb.device())?;
        let norm_k_w = vb.pp(k_name).get(head_dim, "weight")?.dequantize(vb.device())?;
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

    fn forward(
        &self,
        img_hidden: &Tensor,
        txt_hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        img_seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b, _, _) = img_hidden.dims3()?;
        let txt_seq_len = txt_hidden.dim(1)?;

        let q_img = img_hidden
            .apply(&self.to_q)?
            .reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let k_img = img_hidden
            .apply(&self.to_k)?
            .reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let v_img = img_hidden
            .apply(&self.to_v)?
            .reshape((b, img_seq_len, self.n_heads, self.head_dim))?;

        let q_txt = txt_hidden
            .apply(&self.add_q_proj)?
            .reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;
        let k_txt = txt_hidden
            .apply(&self.add_k_proj)?
            .reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;
        let v_txt = txt_hidden
            .apply(&self.add_v_proj)?
            .reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;

        let (q_img, k_img) = self.qk_norm.forward(&q_img, &k_img)?;
        let (q_txt, k_txt) = self.added_qk_norm.forward(&q_txt, &k_txt)?;

        let q_img = apply_rotary_emb(&q_img, cos, sin)?;
        let k_img = apply_rotary_emb(&k_img, cos, sin)?;

        let q = Tensor::cat(&[&q_txt, &q_img], 1)?;
        let k = Tensor::cat(&[&k_txt, &k_img], 1)?;
        let v = Tensor::cat(&[&v_txt, &v_img], 1)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn = attn_weights.matmul(&v)?;

        let total_seq = img_seq_len + txt_seq_len;
        let attn = attn.transpose(1, 2)?.reshape((b, total_seq, ()))?;
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

    fn forward(
        &self,
        img_hidden: &Tensor,
        txt_hidden: &Tensor,
        temb: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let img_seq_len = img_hidden.dim(1)?;
        let img_mod = temb.silu()?.apply(&self.img_mod)?.unsqueeze(1)?;
        let txt_mod = temb.silu()?.apply(&self.txt_mod)?.unsqueeze(1)?;
        let img_chunks = img_mod.chunk(6, D::Minus1)?;
        let txt_chunks = txt_mod.chunk(6, D::Minus1)?;
        let (img_shift_msa, img_scale_msa, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp) = (
            &img_chunks[0], &img_chunks[1], &img_chunks[2], &img_chunks[3], &img_chunks[4], &img_chunks[5],
        );
        let (txt_shift_msa, txt_scale_msa, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp) = (
            &txt_chunks[0], &txt_chunks[1], &txt_chunks[2], &txt_chunks[3], &txt_chunks[4], &txt_chunks[5],
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
        let (img_attn, txt_attn) =
            self.attn
                .forward(&img_attn_in, &txt_attn_in, cos, sin, img_seq_len)?;
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
        let img_hidden = (img_hidden + img_gate_mlp.broadcast_mul(&self.img_mlp.forward(&img_mlp_in)?)?)?;
        let txt_hidden = (txt_hidden + txt_gate_mlp.broadcast_mul(&self.txt_mlp.forward(&txt_mlp_in)?)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

struct OutputLayer {
    norm_final: LayerNormNoParams,
    adaln_linear: Linear,
    linear: Linear,
}

impl OutputLayer {
    fn new(inner_dim: usize, out_channels: usize, patch_size: usize, vb: VarBuilder) -> Result<Self> {
        let output_dim = patch_size * patch_size * out_channels;
        Ok(Self {
            norm_final: LayerNormNoParams::new(1e-6),
            adaln_linear: quantized_nn::linear(inner_dim, 2 * inner_dim, vb.pp("norm_out").pp("linear"))?,
            linear: quantized_nn::linear(inner_dim, output_dim, vb.pp("proj_out"))?,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let mod_params = temb.silu()?.apply(&self.adaln_linear)?;
        let chunks = mod_params.chunk(2, D::Minus1)?;
        let shift = chunks[0].unsqueeze(1)?;
        let scale = chunks[1].unsqueeze(1)?;
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
            blocks.push(QwenImageTransformerBlock::new(cfg, vb.pp("transformer_blocks").pp(i))?);
        }
        Ok(Self {
            time_embed: TimestepProjEmbeddings::new(cfg.inner_dim, vb.clone())?,
            img_in: quantized_nn::linear(cfg.in_channels, cfg.inner_dim, vb.pp("img_in"))?,
            txt_in: quantized_nn::linear(cfg.joint_attention_dim, cfg.inner_dim, vb.pp("txt_in"))?,
            txt_norm: quantized_nn::RmsNorm::new(cfg.joint_attention_dim, cfg.norm_eps, vb.pp("txt_norm"))?,
            blocks,
            rope_embedder: RopeEmbedder::new(
                10000.0,
                cfg.axes_dims_rope.clone(),
                vec![512, 512, 512],
                &device,
                DType::F32,
            )?,
            output_layer: OutputLayer::new(cfg.inner_dim, cfg.out_channels, cfg.patch_size, vb)?,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor, t: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        let out_dtype = x.dtype();
        let device = x.device();
        let x = x.to_dtype(DType::F32)?;
        let t = t.to_dtype(DType::F32)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(DType::F32)?;

        let (_b, _c, h, w) = x.dims4()?;
        let patch_size = self.cfg.patch_size;
        let temb = self.time_embed.forward(&t)?;

        let x_5d = x.unsqueeze(2)?;
        let (x_patches, orig_size) = patchify(&x_5d, patch_size, 1)?;
        let mut img = x_patches.apply(&self.img_in)?;
        let mut txt = self.txt_norm.forward(&encoder_hidden_states)?.apply(&self.txt_in)?;

        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let img_pos_ids = create_coordinate_grid((1, h_tokens, w_tokens), (0, 0, 0), device)?;
        let (img_cos, img_sin) = self.rope_embedder.forward(&img_pos_ids)?;

        for block in &self.blocks {
            let (new_img, new_txt) = block.forward(&img, &txt, &temb, &img_cos, &img_sin)?;
            img = new_img;
            txt = new_txt;
        }

        let img_out = self.output_layer.forward(&img, &temb)?;
        let x_out = unpatchify(&img_out, orig_size, patch_size, 1, self.cfg.out_channels)?;
        x_out.squeeze(2)?.to_dtype(out_dtype)
    }
}
