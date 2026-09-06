//! Hunyuan3D 2.1 shape transformer.
//!
//! References: ComfyUI 15eb748b, comfy/ldm/hunyuan3dv2_1/hunyuandit.py,
//! and Tencent 82920d64, hy3dshape/hy3dshape/models/denoisers/hunyuandit.py.
//! The synthetic complete-forward fixture executes Tencent on CUDA.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, RmsNorm, VarBuilder};

use crate::attention::{attention_for, AttentionPolicy};

#[derive(Debug, Clone)]
pub struct Config {
    pub in_channels: usize,
    pub hidden_size: usize,
    pub context_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub num_moe_layers: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            in_channels: 64,
            hidden_size: 2048,
            context_dim: 1024,
            depth: 21,
            num_heads: 16,
            num_moe_layers: 6,
            num_experts: 8,
            top_k: 2,
        }
    }
}

/// Upstream Timesteps (:190-237), not FLUX's cosine-first, scaled embedding.
fn timestep_embedding(t: &Tensor, width: usize) -> Result<Tensor> {
    if width == 0 || !width.is_multiple_of(2) {
        candle_core::bail!("Hunyuan3D 2.1 timestep width must be positive and even")
    }
    let half = width / 2;
    let frequency = (Tensor::arange(0u32, half as u32, t.device())?.to_dtype(DType::F32)?
        * (-10000f64.ln() / half as f64))?
        .exp()?;
    let phase = t
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&frequency.unsqueeze(0)?)?;
    Tensor::cat(&[phase.sin()?, phase.cos()?], 1)
}

/// Upstream Attention.forward (:401-412) packs complete projections before
/// reshaping heads. It is deliberately NOT a conventional split-QKV attention.
fn pack_self_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, n, width) = q.dims3()?;
    if heads == 0 || !width.is_multiple_of(heads) {
        candle_core::bail!("Hunyuan3D 2.1 attention width must divide into heads")
    }
    let d = width / heads;
    let packed = Tensor::cat(&[q, k, v], D::Minus1)?.reshape((b, n, heads, 3 * d))?;
    Ok((
        packed.narrow(3, 0, d)?,
        packed.narrow(3, d, d)?,
        packed.narrow(3, 2 * d, d)?,
    ))
}

/// Build expert batches from the small routing matrix; hidden states and
/// expert weights stay on their original device. Selected softmax weights
/// retain their mass (:91-93); renormalizing the top two changes the model.
fn expert_routing(scores: &Tensor, top_k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
    let (tokens, experts) = scores.dims2()?;
    if top_k == 0 || top_k > experts || tokens > u32::MAX as usize {
        candle_core::bail!("invalid Hunyuan3D 2.1 expert routing dimensions")
    }
    let mut routing = vec![Vec::new(); experts];
    for (token, row) in scores
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .iter()
        .enumerate()
    {
        if row.iter().any(|value| !value.is_finite()) {
            candle_core::bail!("non-finite Hunyuan3D 2.1 router score")
        }
        let mut indices: Vec<usize> = (0..experts).collect();
        indices.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        for expert in indices.into_iter().take(top_k) {
            routing[expert].push((token as u32, row[expert]));
        }
    }
    Ok(routing)
}

struct FeedForward {
    first: Linear,
    second: Linear,
}

impl FeedForward {
    fn new(width: usize, expert: bool, vb: VarBuilder) -> Result<Self> {
        let (first, second) = if expert {
            ("net.0.proj", "net.2")
        } else {
            ("fc1", "fc2")
        };
        Ok(Self {
            first: candle_nn::linear(width, width * 4, vb.pp(first))?,
            second: candle_nn::linear(width * 4, width, vb.pp(second))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.second.forward(&self.first.forward(x)?.gelu_erf()?)
    }
}

struct Moe {
    gate: Linear,
    experts: Vec<FeedForward>,
    shared: FeedForward,
    top_k: usize,
}

impl Moe {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?,
            experts: (0..cfg.num_experts)
                .map(|index| {
                    FeedForward::new(cfg.hidden_size, true, vb.pp(format!("experts.{index}")))
                })
                .collect::<Result<_>>()?,
            shared: FeedForward::new(cfg.hidden_size, true, vb.pp("shared_experts"))?,
            top_k: cfg.top_k,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().clone();
        let flat = x.flatten_to(1)?;
        let scores = candle_nn::ops::softmax_last_dim(&self.gate.forward(&flat)?)?;
        let routing = expert_routing(&scores, self.top_k)?;
        let mut out = flat.zeros_like()?;
        for (expert, assignments) in self.experts.iter().zip(routing) {
            if assignments.is_empty() {
                continue;
            }
            let (indices, weights): (Vec<u32>, Vec<f32>) = assignments.into_iter().unzip();
            let count = indices.len();
            let indices = Tensor::from_vec(indices, count, x.device())?;
            let weights = Tensor::from_vec(weights, (count, 1), x.device())?.to_dtype(x.dtype())?;
            let values = expert
                .forward(&flat.index_select(&indices, 0)?)?
                .broadcast_mul(&weights)?;
            out = out.index_add(&indices, &values, 0)?;
        }
        out.reshape(shape)? + self.shared.forward(x)?
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    out: Linear,
    heads: usize,
}

impl Attention {
    fn new(width: usize, context: usize, heads: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: candle_nn::linear_no_bias(width, width, vb.pp("to_q"))?,
            k: candle_nn::linear_no_bias(context, width, vb.pp("to_k"))?,
            v: candle_nn::linear_no_bias(context, width, vb.pp("to_v"))?,
            q_norm: candle_nn::rms_norm(width / heads, eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(width / heads, eps, vb.pp("k_norm"))?,
            out: candle_nn::linear(width, width, vb.pp("out_proj"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let q = self.q.forward(x)?;
        let y = context.unwrap_or(x);
        let k = self.k.forward(y)?;
        let v = self.v.forward(y)?;
        let (b, n, width) = q.dims3()?;
        let d = width / self.heads;
        let (q, k, v) = if context.is_some() {
            // CrossAttention.forward (:331-341) has the same interleave for KV.
            let m = k.dim(1)?;
            let packed = Tensor::cat(&[k, v], 2)?.reshape((b, m, self.heads, 2 * d))?;
            (
                q.reshape((b, n, self.heads, d))?,
                packed.narrow(3, 0, d)?,
                packed.narrow(3, d, d)?,
            )
        } else {
            pack_self_attention(&q, &k, &v, self.heads)?
        };
        let q = self
            .q_norm
            .forward(&q.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_norm
            .forward(&k.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let attended = attention_for(
            AttentionPolicy::Image,
            &q,
            &k,
            &v,
            (1.0 / (d as f64).sqrt()) as f32,
        )
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        self.out
            .forward(&attended.transpose(1, 2)?.reshape((b, n, width))?)
    }
}

enum Mlp {
    Dense(FeedForward),
    Experts(Moe),
}

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    self_attention: Attention,
    cross_attention: Attention,
    mlp: Mlp,
    skip: Option<(Linear, LayerNorm)>,
}

impl Block {
    fn new(cfg: &Config, index: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let width = cfg.hidden_size;
        Ok(Self {
            norm1: candle_nn::layer_norm(width, eps, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(width, eps, vb.pp("norm2"))?,
            norm3: candle_nn::layer_norm(width, eps, vb.pp("norm3"))?,
            self_attention: Attention::new(width, width, cfg.num_heads, eps, vb.pp("attn1"))?,
            cross_attention: Attention::new(
                width,
                cfg.context_dim,
                cfg.num_heads,
                eps,
                vb.pp("attn2"),
            )?,
            mlp: if cfg.depth - index <= cfg.num_moe_layers {
                Mlp::Experts(Moe::new(cfg, vb.pp("moe"))?)
            } else {
                Mlp::Dense(FeedForward::new(width, false, vb.pp("mlp"))?)
            },
            skip: if index > cfg.depth / 2 {
                Some((
                    candle_nn::linear(width * 2, width, vb.pp("skip_linear"))?,
                    candle_nn::layer_norm(width, eps, vb.pp("skip_norm"))?,
                ))
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor, skip: Option<Tensor>) -> Result<Tensor> {
        let x = if let Some((linear, norm)) = &self.skip {
            let skip = skip.ok_or_else(|| {
                candle_core::Error::Msg("missing Hunyuan3D 2.1 skip tensor".into())
            })?;
            norm.forward(&linear.forward(&Tensor::cat(&[&skip, x], 2)?)?)?
        } else {
            x.clone()
        };
        let x = (&x
            + self
                .self_attention
                .forward(&self.norm1.forward(&x)?, None)?)?;
        let x = (&x
            + self
                .cross_attention
                .forward(&self.norm2.forward(&x)?, Some(context))?)?;
        let normalized = self.norm3.forward(&x)?;
        let mlp = match &self.mlp {
            Mlp::Dense(mlp) => mlp.forward(&normalized)?,
            Mlp::Experts(moe) => moe.forward(&normalized)?,
        };
        x + mlp
    }
}

pub struct Hunyuan3dDit21 {
    cfg: Config,
    input: Linear,
    time_in: Linear,
    time_out: Linear,
    blocks: Vec<Block>,
    final_norm: LayerNorm,
    final_linear: Linear,
}

impl Hunyuan3dDit21 {
    pub fn in_channels(&self) -> usize {
        self.cfg.in_channels
    }

    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if cfg.depth == 0
            || cfg.depth.is_multiple_of(2)
            || cfg.hidden_size == 0
            || cfg.num_heads == 0
            || !cfg.hidden_size.is_multiple_of(cfg.num_heads)
            || !cfg.hidden_size.is_multiple_of(2)
            || cfg.num_moe_layers > cfg.depth
            || cfg.top_k == 0
            || cfg.top_k > cfg.num_experts
        {
            candle_core::bail!("invalid Hunyuan3D 2.1 transformer configuration")
        }
        let width = cfg.hidden_size;
        let eps = if vb.dtype() == DType::F16 {
            1.0 / 65504.0
        } else {
            1e-6
        };
        Ok(Self {
            cfg: cfg.clone(),
            input: candle_nn::linear(cfg.in_channels, width, vb.pp("x_embedder"))?,
            time_in: candle_nn::linear(width, width * 4, vb.pp("t_embedder.mlp.0"))?,
            time_out: candle_nn::linear(width * 4, width, vb.pp("t_embedder.mlp.2"))?,
            blocks: (0..cfg.depth)
                .map(|i| Block::new(cfg, i, eps, vb.pp(format!("blocks.{i}"))))
                .collect::<Result<_>>()?,
            final_norm: candle_nn::layer_norm(width, eps, vb.pp("final_layer.norm_final"))?,
            final_linear: candle_nn::linear(width, cfg.in_channels, vb.pp("final_layer.linear"))?,
        })
    }

    /// The ordinary mold sampler invokes conditional and unconditional passes
    /// separately. There is consequently no Comfy-specific batched CFG-half swap.
    /// Input/output [B, C, L], context [B, T, D], sigma [B] in F32.
    pub fn forward(&self, x: &Tensor, sigma: &Tensor, context: &Tensor) -> Result<Tensor> {
        let time = timestep_embedding(
            &(sigma.to_dtype(DType::F32)?.neg()? + 1.0)?,
            self.cfg.hidden_size,
        )?
        .to_dtype(x.dtype())?;
        let time = self
            .time_out
            .forward(&self.time_in.forward(&time)?.gelu_erf()?)?
            .unsqueeze(1)?;
        let latent = self.input.forward(&x.transpose(1, 2)?)?;
        let mut combined = Tensor::cat(&[time, latent], 1)?;
        let mut skips = Vec::with_capacity(self.cfg.depth / 2);
        for (index, block) in self.blocks.iter().enumerate() {
            let skip = if index > self.cfg.depth / 2 {
                skips.pop()
            } else {
                None
            };
            combined = block.forward(&combined, context, skip)?;
            if index < self.cfg.depth / 2 {
                skips.push(combined.clone());
            }
        }
        let normalized = self.final_norm.forward(&combined)?;
        self.final_linear
            .forward(&normalized.narrow(1, 1, normalized.dim(1)? - 1)?)?
            .transpose(1, 2)?
            .neg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn complete_tiny_dit_matches_executable_tencent_cuda_fixture() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/transformer21.safetensors"),
            &device,
        )?;
        let cfg = Config {
            in_channels: 4,
            hidden_size: 32,
            context_dim: 16,
            depth: 3,
            num_heads: 2,
            num_moe_layers: 1,
            num_experts: 3,
            top_k: 2,
        };
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device);
        let model = Hunyuan3dDit21::new(&cfg, vb.pp("model"))?;
        let actual = model.forward(&tensors["input"], &tensors["sigma"], &tensors["context"])?;
        let errors = (actual - &tensors["expected"])?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(errors.iter().all(|error| error.is_finite()));
        let error = errors.into_iter().fold(0f32, f32::max);
        assert!(
            error < 0.00005,
            "Tencent fixture maximum absolute error: {error}"
        );
        Ok(())
    }

    /// Requires retained pretrained weights and the executable oracle export.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA, pretrained checkpoint and retained Tencent oracle tensors"]
    fn pretrained_dit_matches_tencent_cuda_oracle() -> anyhow::Result<()> {
        let checkpoint = std::env::var("MOLD_HUNYUAN3D21_CHECKPOINT")?;
        let fixture = std::env::var("MOLD_HUNYUAN3D21_ORACLE")?;
        let output = std::env::var("MOLD_HUNYUAN3D21_RESULT")?;
        anyhow::ensure!(
            !std::path::Path::new(&output).exists(),
            "retain earlier parity results"
        );
        let device = Device::new_cuda(0)?;
        let tensors = candle_core::safetensors::load(&fixture, &device)?;
        let vb = crate::weight_loader::load_safetensors_with_progress(
            &[checkpoint],
            DType::F32,
            &device,
            "2.1 qualification",
            &crate::progress::ProgressReporter::default(),
        )?;
        let model = Hunyuan3dDit21::new(&Config::default(), vb.pp("model"))?;
        let actual = model.forward(&tensors["input"], &tensors["sigma"], &tensors["context"])?;
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("actual".to_string(), actual.clone())]),
            output,
        )?;
        let values = (actual - &tensors["expected"])?
            .flatten_all()?
            .to_vec1::<f32>()?;
        anyhow::ensure!(values.iter().all(|x| x.is_finite()), "nonfinite prediction");
        let max = values.iter().map(|x| x.abs()).fold(0f32, f32::max);
        let rms =
            (values.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        eprintln!("pretrained Tencent CUDA parity: max_abs={max}, rms={rms}");
        anyhow::ensure!(
            max < 0.0001 && rms < 0.00001,
            "pretrained prediction diverges"
        );
        Ok(())
    }

    #[test]
    fn timestep_uses_sine_then_cosine_without_flux_scaling() -> candle_core::Result<()> {
        let time = Tensor::new(&[0f32, 1.], &Device::Cpu)?;
        let embedded = timestep_embedding(&time, 4)?.to_vec2::<f32>()?;
        assert_eq!(embedded[0], [0., 0., 1., 1.]);
        for (actual, expected) in
            embedded[1]
                .iter()
                .zip([1f32.sin(), 0.01f32.sin(), 1f32.cos(), 0.01f32.cos()])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn qkv_packing_preserves_upstream_head_interleave() -> candle_core::Result<()> {
        // The upstream concatenates complete Q/K/V before reshaping to heads.
        // Independently reshaping each projection gives DIFFERENT tensors.
        let q = Tensor::new(&[[[0f32, 1., 2., 3.]]], &Device::Cpu)?;
        let k = (&q + 4.)?;
        let v = (&q + 8.)?;
        let (q, k, v) = pack_self_attention(&q, &k, &v, 2)?;
        assert_eq!(q.flatten_all()?.to_vec1::<f32>()?, [0., 1., 6., 7.]);
        assert_eq!(k.flatten_all()?.to_vec1::<f32>()?, [2., 3., 8., 9.]);
        assert_eq!(v.flatten_all()?.to_vec1::<f32>()?, [4., 5., 10., 11.]);
        Ok(())
    }

    #[test]
    fn routing_keeps_softmax_mass_instead_of_renormalizing_top_two() -> candle_core::Result<()> {
        let scores = Tensor::new(
            &[[0.1f32, 0.2, 0.4, 0.3], [0.7, 0.1, 0.15, 0.05]],
            &Device::Cpu,
        )?;
        let routing = expert_routing(&scores, 2)?;
        assert_eq!(routing[0], vec![(1, 0.7)]);
        assert!(routing[1].is_empty());
        assert_eq!(routing[2], vec![(0, 0.4), (1, 0.15)]);
        assert_eq!(routing[3], vec![(0, 0.3)]);
        Ok(())
    }

    #[test]
    fn routing_refuses_non_finite_scores() -> candle_core::Result<()> {
        let scores = Tensor::new(&[[0.1f32, f32::NAN]], &Device::Cpu)?;
        assert!(expert_routing(&scores, 2).is_err());
        Ok(())
    }
}
