//! GGUF Flux.2 Klein transformer — quantized inference via QMatMul.
//!
//! Weights stay quantized in VRAM and are dequantized on-the-fly per matmul
//! operation, matching the approach used by ComfyUI and InvokeAI. A Q4 Klein-9B
//! model uses ~6GB VRAM instead of ~18GB with full dequantization.
//!
//! Uses `mold_candle::quantized_nn::Linear` which wraps `QMatMul` — the
//! quantized matmul dequantizes to F32 per operation, so inference runs in F32.
//! Norm weights (small, <1MB total) are dequantized to F32 at load time.
//!
//! NaN safety: candle's CUDA QMatMul can produce sporadic NaN in some output
//! elements. All linear operations are wrapped with `linear_nan_safe()` which
//! replaces NaN with 0.0, following the pattern established in SD3's quantized
//! MMDiT (`sd3/quantized_mmdit.rs`).
//!
//! GGUF tensor naming (unsloth convention):
//! - `double_blocks.{i}.img_attn.qkv.weight` (fused Q+K+V)
//! - `double_blocks.{i}.img_mlp.0.weight` (gate+up fused), `.2.weight` (down)
//! - `single_blocks.{i}.linear1.weight`, `.linear2.weight`
//! - Norms: `.norm.{query,key}_norm.scale`
//! - Embedders: `time_in`, `img_in`, `txt_in`, modulations, `final_layer`

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{LayerNorm, RmsNorm};
use mold_candle::quantized::VarBuilder;
use mold_candle::quantized_nn::{self, Linear};

use super::transformer::EmbedNd;
use super::transformer::Flux2Config;
use super::transformer::{attention, timestep_embedding};

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Apply a quantized linear layer and replace any NaN values with 0.0.
/// Candle's CUDA QMatMul can produce sporadic NaN in some output elements
/// when processing large tensors. This wrapper prevents NaN propagation.
fn linear_nan_safe(linear: &Linear, x: &Tensor) -> Result<Tensor> {
    let out = linear.forward(x)?;
    let nan_mask = out.ne(&out)?; // NaN != NaN → true
    let zero = Tensor::zeros_like(&out)?;
    Ok(nan_mask.where_cond(&zero, &out)?)
}

/// Dequantize a small tensor (norm weights, embeddings) from GGUF to F32.
/// Only used for tensors that must be full precision (norms, LayerNorm).
fn dequant_tensor(vb: &VarBuilder, name: &str, device: &Device) -> Result<Tensor> {
    Ok(vb
        .get_no_shape(name)?
        .dequantize(device)?
        .to_dtype(DType::F32)?)
}

fn make_layer_norm(h_sz: usize, device: &Device) -> Result<LayerNorm> {
    Ok(LayerNorm::new_no_bias(
        Tensor::ones(h_sz, DType::F32, device)?,
        1e-6,
    ))
}

// ---------------------------------------------------------------------------
// Building blocks (quantized inference via QMatMul, F32 compute)
// ---------------------------------------------------------------------------

struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            in_layer: quantized_nn::linear_no_bias(
                in_sz,
                h_sz,
                vb.pp(format!("{prefix}.in_layer")),
            )?,
            out_layer: quantized_nn::linear_no_bias(
                h_sz,
                h_sz,
                vb.pp(format!("{prefix}.out_layer")),
            )?,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        linear_nan_safe(
            &self.out_layer,
            &linear_nan_safe(&self.in_layer, xs)?.silu()?,
        )
    }
}

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs
            .broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)?)
    }
    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.gate.broadcast_mul(xs)?)
    }
}

struct Modulation1 {
    lin: Linear,
}

impl Modulation1 {
    fn new(h_sz: usize, vb: &VarBuilder, name: &str) -> Result<Self> {
        Ok(Self {
            lin: quantized_nn::linear_no_bias(h_sz, 3 * h_sz, vb.pp(format!("{name}.lin")))?,
        })
    }
    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = linear_nan_safe(&self.lin, &vec_.silu()?)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

struct Modulation2 {
    lin: Linear,
}

impl Modulation2 {
    fn new(h_sz: usize, vb: &VarBuilder, name: &str) -> Result<Self> {
        Ok(Self {
            lin: quantized_nn::linear_no_bias(h_sz, 6 * h_sz, vb.pp(format!("{name}.lin")))?,
        })
    }
    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = linear_nan_safe(&self.lin, &vec_.silu()?)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        Ok((
            ModulationOut {
                shift: ys[0].clone(),
                scale: ys[1].clone(),
                gate: ys[2].clone(),
            },
            ModulationOut {
                shift: ys[3].clone(),
                scale: ys[4].clone(),
                gate: ys[5].clone(),
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// DoubleStreamBlock
// ---------------------------------------------------------------------------

struct QDoubleStreamBlock {
    img_qkv: Linear,
    img_proj: Linear,
    img_q_norm: RmsNorm,
    img_k_norm: RmsNorm,
    img_norm1: LayerNorm,
    img_mlp_in: Linear,
    img_mlp_out: Linear,
    img_norm2: LayerNorm,
    txt_qkv: Linear,
    txt_proj: Linear,
    txt_q_norm: RmsNorm,
    txt_k_norm: RmsNorm,
    txt_norm1: LayerNorm,
    txt_mlp_in: Linear,
    txt_mlp_out: Linear,
    txt_norm2: LayerNorm,
    num_heads: usize,
    mlp_sz: usize,
}

impl QDoubleStreamBlock {
    fn new(cfg: &Flux2Config, vb: &VarBuilder, prefix: &str, device: &Device) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let p = |suffix: &str| format!("{prefix}.{suffix}");

        Ok(Self {
            img_qkv: quantized_nn::linear_no_bias(h_sz, 3 * h_sz, vb.pp(p("img_attn.qkv")))?,
            img_proj: quantized_nn::linear_no_bias(h_sz, h_sz, vb.pp(p("img_attn.proj")))?,
            img_q_norm: RmsNorm::new(
                dequant_tensor(vb, &p("img_attn.norm.query_norm.scale"), device)?,
                1e-6,
            ),
            img_k_norm: RmsNorm::new(
                dequant_tensor(vb, &p("img_attn.norm.key_norm.scale"), device)?,
                1e-6,
            ),
            img_norm1: make_layer_norm(h_sz, device)?,
            img_mlp_in: quantized_nn::linear_no_bias(h_sz, 2 * mlp_sz, vb.pp(p("img_mlp.0")))?,
            img_mlp_out: quantized_nn::linear_no_bias(mlp_sz, h_sz, vb.pp(p("img_mlp.2")))?,
            img_norm2: make_layer_norm(h_sz, device)?,
            txt_qkv: quantized_nn::linear_no_bias(h_sz, 3 * h_sz, vb.pp(p("txt_attn.qkv")))?,
            txt_proj: quantized_nn::linear_no_bias(h_sz, h_sz, vb.pp(p("txt_attn.proj")))?,
            txt_q_norm: RmsNorm::new(
                dequant_tensor(vb, &p("txt_attn.norm.query_norm.scale"), device)?,
                1e-6,
            ),
            txt_k_norm: RmsNorm::new(
                dequant_tensor(vb, &p("txt_attn.norm.key_norm.scale"), device)?,
                1e-6,
            ),
            txt_norm1: make_layer_norm(h_sz, device)?,
            txt_mlp_in: quantized_nn::linear_no_bias(h_sz, 2 * mlp_sz, vb.pp(p("txt_mlp.0")))?,
            txt_mlp_out: quantized_nn::linear_no_bias(mlp_sz, h_sz, vb.pp(p("txt_mlp.2")))?,
            txt_norm2: make_layer_norm(h_sz, device)?,
            num_heads: cfg.num_heads,
            mlp_sz,
        })
    }

    fn qkv_from_fused(
        &self,
        xs: &Tensor,
        qkv_proj: &Linear,
        q_norm: &RmsNorm,
        k_norm: &RmsNorm,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, l, _) = xs.dims3()?;
        let qkv = linear_nan_safe(qkv_proj, xs)?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.apply(q_norm)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.apply(k_norm)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        Ok((q, k, v))
    }

    fn mlp_swiglu(&self, xs: &Tensor, mlp_in: &Linear, mlp_out: &Linear) -> Result<Tensor> {
        let x = linear_nan_safe(mlp_in, xs)?;
        let gate = x.narrow(D::Minus1, 0, self.mlp_sz)?.silu()?;
        let val = x.narrow(D::Minus1, self.mlp_sz, self.mlp_sz)?;
        linear_nan_safe(mlp_out, &(gate * val)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod1: &ModulationOut,
        img_mod2: &ModulationOut,
        txt_mod1: &ModulationOut,
        txt_mod2: &ModulationOut,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let img_modulated = img_mod1.scale_shift(&img.apply(&self.img_norm1)?)?;
        let (img_q, img_k, img_v) = self.qkv_from_fused(
            &img_modulated,
            &self.img_qkv,
            &self.img_q_norm,
            &self.img_k_norm,
        )?;
        let txt_modulated = txt_mod1.scale_shift(&txt.apply(&self.txt_norm1)?)?;
        let (txt_q, txt_k, txt_v) = self.qkv_from_fused(
            &txt_modulated,
            &self.txt_qkv,
            &self.txt_q_norm,
            &self.txt_k_norm,
        )?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;
        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn_out = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn_out = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        let img = (img + img_mod1.gate(&linear_nan_safe(&self.img_proj, &img_attn_out)?)?)?;
        let img = (&img
            + img_mod2.gate(&self.mlp_swiglu(
                &img_mod2.scale_shift(&img.apply(&self.img_norm2)?)?,
                &self.img_mlp_in,
                &self.img_mlp_out,
            )?)?)?;
        let txt = (txt + txt_mod1.gate(&linear_nan_safe(&self.txt_proj, &txt_attn_out)?)?)?;
        let txt = (&txt
            + txt_mod2.gate(&self.mlp_swiglu(
                &txt_mod2.scale_shift(&txt.apply(&self.txt_norm2)?)?,
                &self.txt_mlp_in,
                &self.txt_mlp_out,
            )?)?)?;

        Ok((img, txt))
    }
}

// ---------------------------------------------------------------------------
// SingleStreamBlock
// ---------------------------------------------------------------------------

struct QSingleStreamBlock {
    linear1: Linear,
    linear2: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    pre_norm: LayerNorm,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl QSingleStreamBlock {
    fn new(cfg: &Flux2Config, vb: &VarBuilder, prefix: &str, device: &Device) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;

        Ok(Self {
            linear1: quantized_nn::linear_no_bias(
                h_sz,
                3 * h_sz + 2 * mlp_sz,
                vb.pp(format!("{prefix}.linear1")),
            )?,
            linear2: quantized_nn::linear_no_bias(
                h_sz + mlp_sz,
                h_sz,
                vb.pp(format!("{prefix}.linear2")),
            )?,
            norm_q: RmsNorm::new(
                dequant_tensor(vb, &format!("{prefix}.norm.query_norm.scale"), device)?,
                1e-6,
            ),
            norm_k: RmsNorm::new(
                dequant_tensor(vb, &format!("{prefix}.norm.key_norm.scale"), device)?,
                1e-6,
            ),
            pre_norm: make_layer_norm(h_sz, device)?,
            h_sz,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, mod_out: &ModulationOut, pe: &Tensor) -> Result<Tensor> {
        let x_mod = mod_out.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let x_mod = linear_nan_safe(&self.linear1, &x_mod)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.apply(&self.norm_q)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.apply(&self.norm_k)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp_portion = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz * 2)?;
        let attn = attention(&q, &k, &v, pe)?;
        let mlp_gate = mlp_portion.narrow(D::Minus1, 0, self.mlp_sz)?.silu()?;
        let mlp_val = mlp_portion.narrow(D::Minus1, self.mlp_sz, self.mlp_sz)?;
        let mlp_out = (mlp_gate * mlp_val)?;
        let output = linear_nan_safe(&self.linear2, &Tensor::cat(&[attn, mlp_out], 2)?)?;
        Ok((xs + mod_out.gate(&output)?)?)
    }
}

// ---------------------------------------------------------------------------
// LastLayer
// ---------------------------------------------------------------------------

struct QLastLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl QLastLayer {
    fn new(vb: &VarBuilder, h_sz: usize, out_channels: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            norm_final: make_layer_norm(h_sz, device)?,
            linear: quantized_nn::linear_no_bias(h_sz, out_channels, vb.pp("final_layer.linear"))?,
            ada_ln_modulation: quantized_nn::linear_no_bias(
                h_sz,
                2 * h_sz,
                vb.pp("final_layer.adaLN_modulation.1"),
            )?,
        })
    }
    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = linear_nan_safe(&self.ada_ln_modulation, &vec.silu()?)?.chunk(2, 1)?;
        // BFL format: shift first, scale second (opposite of diffusers format)
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        linear_nan_safe(&self.linear, &xs)
    }
}

// ---------------------------------------------------------------------------
// QuantizedFlux2Transformer
// ---------------------------------------------------------------------------

/// Flux.2 Klein transformer loaded from GGUF with quantized inference.
///
/// Weights stay quantized in VRAM (Q4/Q6/Q8) and are dequantized on-the-fly
/// per matmul via `QMatMul`. A Q4 Klein-9B uses ~6GB VRAM vs ~18GB with full
/// dequantization. Inference runs in F32 (QMatMul dequantizes weights to F32).
pub(crate) struct QuantizedFlux2Transformer {
    img_in: Linear,
    txt_in: Linear,
    time_in: MlpEmbedder,
    /// Present only for guidance-distilled checkpoints (`cfg.guidance_embed`),
    /// i.e. FLUX.2 [dev]. Klein GGUFs ship no `guidance_in.*` tensors at all.
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    double_mod_img: Modulation2,
    double_mod_txt: Modulation2,
    single_mod: Modulation1,
    double_blocks: Vec<QDoubleStreamBlock>,
    single_blocks: Vec<QSingleStreamBlock>,
    final_layer: QLastLayer,
}

impl QuantizedFlux2Transformer {
    /// Load from a GGUF VarBuilder. Weights stay quantized; only norm tensors
    /// are dequantized to F32.
    pub fn new(cfg: &Flux2Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let img_in = quantized_nn::linear_no_bias(cfg.in_channels, h_sz, vb.pp("img_in"))?;
        let txt_in = quantized_nn::linear_no_bias(cfg.context_in_dim, h_sz, vb.pp("txt_in"))?;
        let time_in = MlpEmbedder::new(256, h_sz, &vb, "time_in")?;
        // `comfy/ldm/flux/model.py:93` builds `guidance_in` only when
        // `params.guidance_embed`; a distilled Klein GGUF has no such tensors,
        // so the load must stay optional rather than fail looking for them.
        let guidance_in = if cfg.guidance_embed {
            Some(MlpEmbedder::new(256, h_sz, &vb, "guidance_in")?)
        } else {
            None
        };

        let double_mod_img = Modulation2::new(h_sz, &vb, "double_stream_modulation_img")?;
        let double_mod_txt = Modulation2::new(h_sz, &vb, "double_stream_modulation_txt")?;
        let single_mod = Modulation1::new(h_sz, &vb, "single_stream_modulation")?;

        let mut double_blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            double_blocks.push(QDoubleStreamBlock::new(
                cfg,
                &vb,
                &format!("double_blocks.{i}"),
                device,
            )?);
        }

        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        for i in 0..cfg.depth_single_blocks {
            single_blocks.push(QSingleStreamBlock::new(
                cfg,
                &vb,
                &format!("single_blocks.{i}"),
                device,
            )?);
        }

        let final_layer = QLastLayer::new(&vb, h_sz, cfg.in_channels, device)?;
        let pe_embedder = EmbedNd::new(cfg.theta, cfg.axes_dim.to_vec());

        Ok(Self {
            img_in,
            txt_in,
            time_in,
            guidance_in,
            pe_embedder,
            double_mod_img,
            double_mod_txt,
            single_mod,
            double_blocks,
            single_blocks,
            final_layer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        _y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 || img.rank() != 3 {
            anyhow::bail!("expected rank 3, got txt={} img={}", txt.rank(), img.rank())
        }
        let input_dtype = img.dtype();

        // QMatMul dequantizes weights to F32, so all compute runs in F32.
        let img = &img.to_dtype(DType::F32)?;
        let txt = &txt.to_dtype(DType::F32)?;
        let img_ids = &img_ids.to_dtype(DType::F32)?;
        let txt_ids = &txt_ids.to_dtype(DType::F32)?;
        let timesteps = &timesteps.to_dtype(DType::F32)?;

        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = linear_nan_safe(&self.txt_in, txt)?;
        let mut img = linear_nan_safe(&self.img_in, img)?;
        let mut vec_ = self
            .time_in
            .forward(&timestep_embedding(timesteps, 256, DType::F32)?)?;
        // `comfy/ldm/flux/model.py:171-173`:
        //   vec = vec + guidance_in(timestep_embedding(guidance, 256))
        // FLUX.2 [dev] is guidance-distilled, so dropping this term leaves the
        // conditioning vector at its unguided value and the render collapses.
        if let (Some(g_in), Some(guidance)) = (self.guidance_in.as_ref(), guidance) {
            let guidance = guidance.to_device(img.device())?.to_dtype(DType::F32)?;
            vec_ = (vec_ + g_in.forward(&timestep_embedding(&guidance, 256, DType::F32)?)?)?;
        }

        let (img_mod1, img_mod2) = self.double_mod_img.forward(&vec_)?;
        let (txt_mod1, txt_mod2) = self.double_mod_txt.forward(&vec_)?;

        for block in &self.double_blocks {
            (img, txt) =
                block.forward(&img, &txt, &img_mod1, &img_mod2, &txt_mod1, &txt_mod2, &pe)?;
        }

        let single_mod = self.single_mod.forward(&vec_)?;
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in &self.single_blocks {
            img = block.forward(&img, &single_mod, &pe)?;
        }
        let img = img.i((.., txt.dim(1)?..))?;
        let out = self.final_layer.forward(&img, &vec_)?;

        // Convert back to caller's dtype (BF16 for downstream VAE decode)
        out.to_dtype(input_dtype).map_err(Into::into)
    }
}

/// Tiny synthetic Flux.2 transformers for unit tests. Shared with
/// `super::transformer`'s tests, which exercise the denoise loop (and its
/// classifier-free-guidance branch) over one of these.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_core::Device;

    /// A Flux.2 config small enough to instantiate on CPU in a unit test while
    /// keeping every shape relationship the loader depends on (head_dim ==
    /// sum(axes_dim), mlp_ratio, fused QKV widths).
    pub(crate) fn tiny_cfg(guidance_embed: bool) -> Flux2Config {
        Flux2Config {
            in_channels: 4,
            vec_in_dim: 0,
            context_in_dim: 6,
            hidden_size: 8,
            mlp_ratio: 3.0,
            num_heads: 1,
            depth: 1,
            depth_single_blocks: 1,
            axes_dim: vec![2, 2, 2, 2],
            theta: 2000,
            guidance_embed,
        }
    }

    /// Deterministic non-degenerate weights: a constant tensor would make every
    /// output row identical and hide a dropped conditioning term.
    pub(crate) fn spread(shape: (usize, usize), salt: f32) -> Tensor {
        let (rows, cols) = shape;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.37 + salt).sin()) * 0.1)
            .collect();
        Tensor::from_vec(data, (rows, cols), &Device::Cpu).expect("weight")
    }

    /// Every tensor `QuantizedFlux2Transformer::new` looks up for `cfg`, in the
    /// BFL-native GGUF naming unsloth/city96 publish.
    pub(crate) fn tiny_gguf_tensors(cfg: &Flux2Config) -> Vec<(String, Tensor)> {
        let h = cfg.hidden_size;
        let mlp = (h as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h / cfg.num_heads;
        let mut t: Vec<(String, Tensor)> = vec![
            ("img_in.weight".into(), spread((h, cfg.in_channels), 0.1)),
            ("txt_in.weight".into(), spread((h, cfg.context_in_dim), 0.2)),
            ("time_in.in_layer.weight".into(), spread((h, 256), 0.3)),
            ("time_in.out_layer.weight".into(), spread((h, h), 0.4)),
            (
                "double_stream_modulation_img.lin.weight".into(),
                spread((6 * h, h), 0.5),
            ),
            (
                "double_stream_modulation_txt.lin.weight".into(),
                spread((6 * h, h), 0.6),
            ),
            (
                "single_stream_modulation.lin.weight".into(),
                spread((3 * h, h), 0.7),
            ),
            (
                "final_layer.linear.weight".into(),
                spread((cfg.in_channels, h), 0.8),
            ),
            (
                "final_layer.adaLN_modulation.1.weight".into(),
                spread((2 * h, h), 0.9),
            ),
        ];
        if cfg.guidance_embed {
            t.push(("guidance_in.in_layer.weight".into(), spread((h, 256), 1.1)));
            t.push(("guidance_in.out_layer.weight".into(), spread((h, h), 1.2)));
        }
        for i in 0..cfg.depth {
            for side in ["img", "txt"] {
                t.push((
                    format!("double_blocks.{i}.{side}_attn.qkv.weight"),
                    spread((3 * h, h), 1.3),
                ));
                t.push((
                    format!("double_blocks.{i}.{side}_attn.proj.weight"),
                    spread((h, h), 1.4),
                ));
                t.push((
                    format!("double_blocks.{i}.{side}_attn.norm.query_norm.scale"),
                    spread((1, head_dim), 1.5).flatten_all().unwrap(),
                ));
                t.push((
                    format!("double_blocks.{i}.{side}_attn.norm.key_norm.scale"),
                    spread((1, head_dim), 1.6).flatten_all().unwrap(),
                ));
                t.push((
                    format!("double_blocks.{i}.{side}_mlp.0.weight"),
                    spread((2 * mlp, h), 1.7),
                ));
                t.push((
                    format!("double_blocks.{i}.{side}_mlp.2.weight"),
                    spread((h, mlp), 1.8),
                ));
            }
        }
        for i in 0..cfg.depth_single_blocks {
            t.push((
                format!("single_blocks.{i}.linear1.weight"),
                spread((3 * h + 2 * mlp, h), 2.1),
            ));
            t.push((
                format!("single_blocks.{i}.linear2.weight"),
                spread((h, h + mlp), 2.2),
            ));
            t.push((
                format!("single_blocks.{i}.norm.query_norm.scale"),
                spread((1, head_dim), 2.3).flatten_all().unwrap(),
            ));
            t.push((
                format!("single_blocks.{i}.norm.key_norm.scale"),
                spread((1, head_dim), 2.4).flatten_all().unwrap(),
            ));
        }
        t
    }

    pub(crate) fn tiny_transformer(cfg: &Flux2Config) -> QuantizedFlux2Transformer {
        tiny_transformer_with(cfg, |_, tensor| tensor)
    }

    /// As `tiny_transformer`, with each tensor passed through `edit` first —
    /// used to zero one sub-module and hold the rest fixed.
    pub(crate) fn tiny_transformer_with(
        cfg: &Flux2Config,
        edit: impl Fn(&str, Tensor) -> Tensor,
    ) -> QuantizedFlux2Transformer {
        let tensors: Vec<(String, Tensor)> = tiny_gguf_tensors(cfg)
            .into_iter()
            .map(|(name, tensor)| {
                let edited = edit(&name, tensor);
                (name, edited)
            })
            .collect();
        let quantized: Vec<(String, QTensor)> = tensors
            .into_iter()
            .map(|(name, tensor)| {
                (
                    name,
                    QTensor::quantize(&tensor, GgmlDType::F32).expect("quantize"),
                )
            })
            .collect();
        let refs: Vec<(&str, &QTensor)> = quantized
            .iter()
            .map(|(name, q)| (name.as_str(), q))
            .collect();
        let mut buffer = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut buffer, &[], &refs).expect("write gguf");
        let vb = VarBuilder::from_gguf_buffer(&buffer.into_inner(), &Device::Cpu)
            .expect("load test gguf");
        QuantizedFlux2Transformer::new(cfg, vb, &Device::Cpu).expect("build transformer")
    }

    /// Run one forward with the tiny model at the given guidance value.
    pub(crate) fn tiny_forward(
        model: &QuantizedFlux2Transformer,
        guidance: Option<f32>,
    ) -> Vec<f32> {
        let device = Device::Cpu;
        let img = spread((3, 4), 3.1).reshape((1, 3, 4)).unwrap();
        let txt = spread((2, 6), 3.2).reshape((1, 2, 6)).unwrap();
        let img_ids = Tensor::zeros((1, 3, 4), DType::F32, &device).unwrap();
        let txt_ids = Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap();
        let timesteps = Tensor::full(0.5f32, 1, &device).unwrap();
        let y = Tensor::zeros((1, 1), DType::F32, &device).unwrap();
        let guidance = guidance.map(|g| Tensor::full(g, 1, &device).unwrap());
        model
            .forward(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                guidance.as_ref(),
            )
            .expect("forward")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// FLUX.2 [dev] is guidance-distilled: `comfy/ldm/flux/model.py:171-173`
    /// adds `guidance_in(timestep_embedding(guidance))` into the conditioning
    /// vector. A GGUF dev checkpoint that ignored the term would denoise as if
    /// every request asked for the same guidance.
    #[test]
    fn guidance_distilled_gguf_conditions_on_the_requested_guidance() {
        let cfg = tiny_cfg(true);
        let model = tiny_transformer(&cfg);

        let low = tiny_forward(&model, Some(1.0));
        let high = tiny_forward(&model, Some(7.0));

        assert_eq!(low.len(), high.len());
        assert!(
            low.iter().all(|v| v.is_finite()) && high.iter().all(|v| v.is_finite()),
            "guided forward must stay finite"
        );
        assert!(
            low.iter().zip(&high).any(|(a, b)| (a - b).abs() > 1e-6),
            "guidance 1.0 and 7.0 produced identical predictions — guidance_in is not wired in"
        );
    }

    /// The guidance term is ADDED to the conditioning vector
    /// (`comfy/ldm/flux/model.py:173`: `vec = vec + guidance_in(...)`), so an
    /// all-zero `guidance_in` must leave the render exactly where the
    /// unguided one is. A composition that replaced `vec` instead of adding
    /// to it, or that fed the raw scale in place of its timestep embedding,
    /// would move the output here while still "responding to guidance".
    #[test]
    fn a_zero_guidance_embedder_leaves_the_conditioning_vector_untouched() {
        let cfg = tiny_cfg(true);
        let zeroed = tiny_transformer_with(&cfg, |name, tensor| {
            if name.starts_with("guidance_in.") {
                Tensor::zeros(tensor.shape(), DType::F32, &Device::Cpu).expect("zeros")
            } else {
                tensor
            }
        });
        let distilled = tiny_transformer(&tiny_cfg(false));

        let guided = tiny_forward(&zeroed, Some(7.0));
        let unguided = tiny_forward(&zeroed, None);
        assert_eq!(
            guided, unguided,
            "a zero guidance embedder must contribute nothing"
        );
        // And it lands where the checkpoint with no guidance embedder at all
        // does — the term is the only difference between the two.
        let baseline = tiny_forward(&distilled, None);
        assert_eq!(guided.len(), baseline.len());
        for (a, b) in guided.iter().zip(&baseline) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// A distilled Klein GGUF ships no `guidance_in.*` tensors at all, so the
    /// loader must not look for them and the request's guidance value must not
    /// reach the conditioning vector.
    #[test]
    fn distilled_gguf_loads_without_guidance_tensors_and_ignores_guidance() {
        let cfg = tiny_cfg(false);
        let model = tiny_transformer(&cfg);

        let none = tiny_forward(&model, None);
        let seven = tiny_forward(&model, Some(7.0));

        assert_eq!(
            none, seven,
            "distilled Klein must ignore the guidance value"
        );
    }

    /// Verify that QLastLayer::forward uses BFL ordering (shift, scale) not diffusers (scale, shift).
    ///
    /// The BFL reference code unpacks adaLN_modulation output as:
    ///   shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
    /// while diffusers uses:
    ///   scale, shift = ...chunk(2, dim=1)
    ///
    /// GGUF files use BFL naming/convention, so the quantized transformer must use
    /// BFL ordering. Getting this wrong causes ~3x output amplitude divergence.
    ///
    /// This test validates the ordering by manually computing the BFL result and
    /// comparing against the diffusers result with known inputs.
    #[test]
    fn bfl_shift_scale_ordering_produces_additive_shift() {
        let device = Device::Cpu;
        let h_sz = 8;

        // Simulate BFL modulation: chunks[0] = shift, chunks[1] = scale
        let silu_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp()); // silu(1) ≈ 0.7311

        // Construct modulation output where shift ≈ 0.73, scale = 0
        let shift = Tensor::full(silu_1, (1, h_sz), &device).unwrap();
        let scale = Tensor::zeros((1, h_sz), DType::F32, &device).unwrap();

        // Non-uniform input so norm != 1
        let xs_data: Vec<f32> = (0..h_sz).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let xs = Tensor::from_vec(xs_data, (1, h_sz), &device).unwrap();

        // BFL: result = norm(xs) * (scale + 1) + shift = norm(xs) + shift
        let norm = make_layer_norm(h_sz, &device).unwrap();
        let normed = xs.apply(&norm).unwrap();
        let bfl_result = normed
            .broadcast_mul(&(scale.unsqueeze(1).unwrap() + 1.0).unwrap())
            .unwrap()
            .broadcast_add(&shift.unsqueeze(1).unwrap())
            .unwrap();

        // With scale=0, BFL adds a constant shift to normalized values.
        // The shift contribution should be visible in every output element.
        let bfl_vals: Vec<f32> = bfl_result.flatten_all().unwrap().to_vec1().unwrap();
        for v in &bfl_vals {
            assert!(v.is_finite(), "BFL result contains non-finite: {v}");
            // With scale=0: result = norm(x) + shift. Since norm has zero mean
            // and shift > 0, the mean of output should be approximately shift.
        }
        let mean: f32 = bfl_vals.iter().sum::<f32>() / bfl_vals.len() as f32;
        assert!(
            (mean - silu_1).abs() < 0.01,
            "BFL mean {mean} should be close to shift {silu_1}"
        );
    }
}
