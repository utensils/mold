//! GGUF Flux.2 Klein transformer — dequantize-on-CPU approach.
//!
//! Loads GGUF weights, dequantizes on CPU to BF16, moves to GPU, then uses
//! standard `candle_nn::Linear` for inference. This produces identical precision
//! to the BF16 safetensors path since all inference runs through BF16 matmul.
//!
//! GGUF tensor naming (unsloth convention):
//! - `double_blocks.{i}.img_attn.qkv.weight` (fused Q+K+V)
//! - `double_blocks.{i}.img_mlp.0.weight` (gate+up fused), `.2.weight` (down)
//! - `single_blocks.{i}.linear1.weight`, `.linear2.weight`
//! - Norms: `.norm.{query,key}_norm.scale`
//! - Embedders: `time_in`, `img_in`, `txt_in`, modulations, `final_layer`

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm};
use candle_transformers::quantized_var_builder::VarBuilder;

use super::transformer::EmbedNd;
use super::transformer::Flux2Config;
use super::transformer::{attention, timestep_embedding};

// ---------------------------------------------------------------------------
// Utility: dequantize GGUF → BF16 on CPU → move to GPU
// ---------------------------------------------------------------------------

fn dequant_linear(vb: &VarBuilder, name: &str, dtype: DType, device: &Device) -> Result<Linear> {
    let w = vb
        .get_no_shape(name)?
        .dequantize(vb.device())?
        .to_dtype(dtype)?
        .to_device(device)?;
    Ok(Linear::new(w, None))
}

fn dequant_tensor(vb: &VarBuilder, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
    Ok(vb
        .get_no_shape(name)?
        .dequantize(vb.device())?
        .to_dtype(dtype)?
        .to_device(device)?)
}

fn make_layer_norm(h_sz: usize, dtype: DType, device: &Device) -> Result<LayerNorm> {
    Ok(LayerNorm::new_no_bias(
        Tensor::ones(h_sz, dtype, device)?,
        1e-6,
    ))
}

// ---------------------------------------------------------------------------
// Building blocks (all use candle_nn::Linear, BF16 on GPU)
// ---------------------------------------------------------------------------

struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn new(vb: &VarBuilder, prefix: &str, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            in_layer: dequant_linear(vb, &format!("{prefix}.in_layer.weight"), dtype, device)?,
            out_layer: dequant_linear(vb, &format!("{prefix}.out_layer.weight"), dtype, device)?,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)?)
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
    fn new(vb: &VarBuilder, name: &str, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            lin: dequant_linear(vb, &format!("{name}.lin.weight"), dtype, device)?,
        })
    }
    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
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
    fn new(vb: &VarBuilder, name: &str, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            lin: dequant_linear(vb, &format!("{name}.lin.weight"), dtype, device)?,
        })
    }
    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
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
    fn new(
        cfg: &Flux2Config,
        vb: &VarBuilder,
        prefix: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let p = |suffix: &str| format!("{prefix}.{suffix}");

        Ok(Self {
            img_qkv: dequant_linear(vb, &p("img_attn.qkv.weight"), dtype, device)?,
            img_proj: dequant_linear(vb, &p("img_attn.proj.weight"), dtype, device)?,
            img_q_norm: RmsNorm::new(
                dequant_tensor(vb, &p("img_attn.norm.query_norm.scale"), dtype, device)?,
                1e-6,
            ),
            img_k_norm: RmsNorm::new(
                dequant_tensor(vb, &p("img_attn.norm.key_norm.scale"), dtype, device)?,
                1e-6,
            ),
            img_norm1: make_layer_norm(h_sz, dtype, device)?,
            img_mlp_in: dequant_linear(vb, &p("img_mlp.0.weight"), dtype, device)?,
            img_mlp_out: dequant_linear(vb, &p("img_mlp.2.weight"), dtype, device)?,
            img_norm2: make_layer_norm(h_sz, dtype, device)?,
            txt_qkv: dequant_linear(vb, &p("txt_attn.qkv.weight"), dtype, device)?,
            txt_proj: dequant_linear(vb, &p("txt_attn.proj.weight"), dtype, device)?,
            txt_q_norm: RmsNorm::new(
                dequant_tensor(vb, &p("txt_attn.norm.query_norm.scale"), dtype, device)?,
                1e-6,
            ),
            txt_k_norm: RmsNorm::new(
                dequant_tensor(vb, &p("txt_attn.norm.key_norm.scale"), dtype, device)?,
                1e-6,
            ),
            txt_norm1: make_layer_norm(h_sz, dtype, device)?,
            txt_mlp_in: dequant_linear(vb, &p("txt_mlp.0.weight"), dtype, device)?,
            txt_mlp_out: dequant_linear(vb, &p("txt_mlp.2.weight"), dtype, device)?,
            txt_norm2: make_layer_norm(h_sz, dtype, device)?,
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
        let qkv = xs.apply(qkv_proj)?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.apply(q_norm)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.apply(k_norm)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        Ok((q, k, v))
    }

    fn mlp_swiglu(&self, xs: &Tensor, mlp_in: &Linear, mlp_out: &Linear) -> Result<Tensor> {
        let x = xs.apply(mlp_in)?;
        let gate = x.narrow(D::Minus1, 0, self.mlp_sz)?.silu()?;
        let val = x.narrow(D::Minus1, self.mlp_sz, self.mlp_sz)?;
        (gate * val)?.apply(mlp_out).map_err(Into::into)
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

        let img = (img + img_mod1.gate(&img_attn_out.apply(&self.img_proj)?)?)?;
        let img = (&img
            + img_mod2.gate(&self.mlp_swiglu(
                &img_mod2.scale_shift(&img.apply(&self.img_norm2)?)?,
                &self.img_mlp_in,
                &self.img_mlp_out,
            )?)?)?;
        let txt = (txt + txt_mod1.gate(&txt_attn_out.apply(&self.txt_proj)?)?)?;
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
    fn new(
        cfg: &Flux2Config,
        vb: &VarBuilder,
        prefix: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;

        Ok(Self {
            linear1: dequant_linear(vb, &format!("{prefix}.linear1.weight"), dtype, device)?,
            linear2: dequant_linear(vb, &format!("{prefix}.linear2.weight"), dtype, device)?,
            norm_q: RmsNorm::new(
                dequant_tensor(
                    vb,
                    &format!("{prefix}.norm.query_norm.scale"),
                    dtype,
                    device,
                )?,
                1e-6,
            ),
            norm_k: RmsNorm::new(
                dequant_tensor(vb, &format!("{prefix}.norm.key_norm.scale"), dtype, device)?,
                1e-6,
            ),
            pre_norm: make_layer_norm(h_sz, dtype, device)?,
            h_sz,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, mod_out: &ModulationOut, pe: &Tensor) -> Result<Tensor> {
        let x_mod = mod_out.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let x_mod = x_mod.apply(&self.linear1)?;
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
        let output = Tensor::cat(&[attn, mlp_out], 2)?.apply(&self.linear2)?;
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
    fn new(vb: &VarBuilder, dtype: DType, h_sz: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            norm_final: make_layer_norm(h_sz, dtype, device)?,
            linear: dequant_linear(vb, "final_layer.linear.weight", dtype, device)?,
            ada_ln_modulation: dequant_linear(
                vb,
                "final_layer.adaLN_modulation.1.weight",
                dtype,
                device,
            )?,
        })
    }
    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        // BFL format: shift first, scale second (opposite of diffusers format)
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        xs.apply(&self.linear).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// QuantizedFlux2Transformer
// ---------------------------------------------------------------------------

/// Flux.2 Klein transformer loaded from GGUF.
///
/// Weights are dequantized on CPU to BF16, then moved to GPU. Inference uses
/// standard `candle_nn::Linear` (BF16 matmul), giving identical precision to
/// the BF16 safetensors path. GGUF advantage: smaller download (Q8: 4.3GB vs 7.7GB).
pub(crate) struct QuantizedFlux2Transformer {
    img_in: Linear,
    txt_in: Linear,
    time_in: MlpEmbedder,
    pe_embedder: EmbedNd,
    double_mod_img: Modulation2,
    double_mod_txt: Modulation2,
    single_mod: Modulation1,
    double_blocks: Vec<QDoubleStreamBlock>,
    single_blocks: Vec<QSingleStreamBlock>,
    final_layer: QLastLayer,
}

impl QuantizedFlux2Transformer {
    /// Load from a GGUF VarBuilder (on CPU). Dequantizes to `gpu_dtype` and moves to `device`.
    pub fn new(
        cfg: &Flux2Config,
        vb: VarBuilder,
        gpu_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let dtype = gpu_dtype;
        let img_in = dequant_linear(&vb, "img_in.weight", dtype, device)?;
        let txt_in = dequant_linear(&vb, "txt_in.weight", dtype, device)?;
        let time_in = MlpEmbedder::new(&vb, "time_in", dtype, device)?;

        let double_mod_img = Modulation2::new(&vb, "double_stream_modulation_img", dtype, device)?;
        let double_mod_txt = Modulation2::new(&vb, "double_stream_modulation_txt", dtype, device)?;
        let single_mod = Modulation1::new(&vb, "single_stream_modulation", dtype, device)?;

        let mut double_blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            double_blocks.push(QDoubleStreamBlock::new(
                cfg,
                &vb,
                &format!("double_blocks.{i}"),
                dtype,
                device,
            )?);
        }

        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        for i in 0..cfg.depth_single_blocks {
            single_blocks.push(QSingleStreamBlock::new(
                cfg,
                &vb,
                &format!("single_blocks.{i}"),
                dtype,
                device,
            )?);
        }

        let final_layer = QLastLayer::new(&vb, dtype, cfg.hidden_size, device)?;
        let pe_embedder = EmbedNd::new(cfg.theta, cfg.axes_dim.to_vec());

        Ok(Self {
            img_in,
            txt_in,
            time_in,
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
        _guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 || img.rank() != 3 {
            anyhow::bail!("expected rank 3, got txt={} img={}", txt.rank(), img.rank())
        }
        let dtype = img.dtype();

        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;
        let vec_ = self
            .time_in
            .forward(&timestep_embedding(timesteps, 256, dtype)?)?;

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
        self.final_layer.forward(&img, &vec_)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Verify that GGUF QLastLayer uses BFL ordering (shift, scale) not diffusers (scale, shift).
    ///
    /// The BFL reference code unpacks adaLN_modulation output as:
    ///   shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
    /// while diffusers uses:
    ///   scale, shift = ...chunk(2, dim=1)
    ///
    /// GGUF files use BFL naming/convention, so the quantized transformer must use
    /// BFL ordering. Getting this wrong causes ~3x output amplitude divergence.
    /// Verify QLastLayer uses BFL ordering (shift, scale) not diffusers (scale, shift).
    ///
    /// Constructs a layer where the first half of ada_ln output is nonzero (intended
    /// as shift in BFL) and the second half is zero. With non-uniform input, the two
    /// orderings produce numerically different results:
    ///   BFL:      result = norm(xs) * (0 + 1) + shift   = norm(xs) + shift
    ///   diffusers: result = norm(xs) * (shift + 1) + 0  = norm(xs) * (shift + 1)
    #[test]
    fn qlast_layer_uses_bfl_shift_scale_ordering() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let h_sz = 8;
        let out_c = 4;

        // ada_ln weight: first h_sz rows = identity (produces silu(vec)·1 ≈ 0.73),
        // second h_sz rows = zeros. So chunks[0] ≈ 0.73, chunks[1] = 0.
        let mut ada_data = vec![0f32; 2 * h_sz * h_sz];
        for i in 0..h_sz {
            ada_data[i * h_sz + i] = 1.0;
        }
        let ada_w = Tensor::from_vec(ada_data, (2 * h_sz, h_sz), &device).unwrap();

        let layer = QLastLayer {
            norm_final: make_layer_norm(h_sz, dtype, &device).unwrap(),
            linear: Linear::new(Tensor::ones((out_c, h_sz), dtype, &device).unwrap(), None),
            ada_ln_modulation: Linear::new(ada_w, None),
        };

        // Non-uniform input so norm(xs) != 1, making the two orderings distinguishable
        let xs_data: Vec<f32> = (0..2 * h_sz).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let xs = Tensor::from_vec(xs_data, (1, 2, h_sz), &device).unwrap();
        let vec = Tensor::ones((1, h_sz), dtype, &device).unwrap();

        let out = layer.forward(&xs, &vec).unwrap();
        let out_vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        // Compute expected BFL result manually:
        // shift ≈ silu(1.0) = 0.7311 per element, scale = 0
        // result_elem = norm(xs_elem) * (0 + 1) + 0.7311
        // With diffusers ordering it would be: norm(xs_elem) * (0.7311 + 1) + 0
        // These differ when norm(xs) != 1.
        let silu_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp()); // silu(1) ≈ 0.7311

        // For BFL: each output element = sum_over_h_sz(norm_elem + silu_1) = sum(norm) + h_sz * silu_1
        // For diffusers: each output element = sum_over_h_sz(norm_elem * (silu_1 + 1)) = sum(norm) * (silu_1 + 1)
        // These are only equal when sum(norm) = h_sz * silu_1 / silu_1 = h_sz, i.e. norm=1.
        // With our non-uniform input, norm != 1, so they differ.
        for v in &out_vals {
            assert!(v.is_finite(), "QLastLayer output contains non-finite: {v}");
        }

        // Verify we get the BFL result, not the diffusers result.
        // Pick the first output element (sum over h_sz of the first token's processed values).
        let first = out_vals[0];
        // The BFL result includes an additive shift term. With scale=0, the multiplication
        // factor is exactly 1.0, so the output is: sum(norm(xs[0])) + h_sz * silu_1.
        // The diffusers result would multiply by (silu_1 + 1) with no additive term.
        // The additive shift makes BFL output LARGER than pure multiplicative scaling
        // when norm values are small (our input starts at 0.1).
        let expected_shift_contribution = h_sz as f32 * silu_1;
        // If the shift is additive (BFL), subtracting it should leave just sum(norm).
        // If the shift is multiplicative (diffusers), the value structure is different.
        // Assert the output is consistent with additive shift (BFL ordering).
        assert!(
            (first - expected_shift_contribution).abs() < first.abs(),
            "Output {first} not consistent with BFL additive shift of {expected_shift_contribution}"
        );
    }
}
