//! BF16 Z-Image transformer adaptive block offloading.
//!
//! Stem/final/RoPE layers stay on the runtime device. Refiner and main
//! transformer blocks use adaptive residency: the planner keeps the largest
//! safe subset GPU-resident and streams only the remaining CPU-resident blocks
//! one at a time for each forward pass.

use crate::adaptive_offload::{
    plan_adaptive_residency, AdaptiveResidencyPlan, ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM,
};
use crate::progress::ProgressReporter;
use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};
use candle_transformers::models::z_image::transformer::{
    apply_rotary_emb, create_coordinate_grid, patchify, unpatchify, Config, FinalLayer,
    RopeEmbedder, TimestepEmbedder, ADALN_EMBED_DIM, SEQ_MULTI_OF,
};

use super::transformer::{
    build_basic_unified_sequence, pad_position_ids_with_zeros, pad_token_sequence,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ZImageStreamingBlock {
    ContextRefiner(usize),
    NoiseRefiner(usize),
    Layer(usize),
}

pub(crate) fn zimage_streaming_block_plan(cfg: &Config) -> Vec<ZImageStreamingBlock> {
    let mut blocks = Vec::with_capacity(cfg.n_refiner_layers * 2 + cfg.n_layers);
    blocks.extend((0..cfg.n_refiner_layers).map(ZImageStreamingBlock::ContextRefiner));
    blocks.extend((0..cfg.n_refiner_layers).map(ZImageStreamingBlock::NoiseRefiner));
    blocks.extend((0..cfg.n_layers).map(ZImageStreamingBlock::Layer));
    blocks
}

fn linear_to_device(linear: &Linear, device: &Device) -> Result<Linear> {
    let weight = linear.weight().to_device(device)?;
    let bias = linear
        .bias()
        .map(|bias| bias.to_device(device))
        .transpose()?;
    Ok(Linear::new(weight, bias))
}

fn tensor_bytes(t: &Tensor) -> usize {
    t.elem_count() * t.dtype().size_in_bytes()
}

fn linear_bytes(linear: &Linear) -> usize {
    tensor_bytes(linear.weight()) + linear.bias().map(tensor_bytes).unwrap_or(0)
}

#[derive(Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }

    fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_device(device)?,
            eps: self.eps,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.dim(D::Minus1)?;
        let internal_dtype = match x.dtype() {
            candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
            dtype => dtype,
        };
        let x_internal = x.to_dtype(internal_dtype)?;
        let norm = (x_internal.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let x_normed = x_internal.broadcast_div(&(norm + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x.dtype())?.broadcast_mul(&self.weight)
    }
}

fn rms_norm_to_device(norm: &RmsNorm, device: &Device) -> Result<RmsNorm> {
    norm.to_device(device)
}

fn rms_norm_bytes(norm: &RmsNorm) -> usize {
    tensor_bytes(&norm.weight)
}

#[derive(Clone)]
struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(dim, hidden_dim, vb.pp("w1"))?,
            w2: linear_no_bias(hidden_dim, dim, vb.pp("w2"))?,
            w3: linear_no_bias(dim, hidden_dim, vb.pp("w3"))?,
        })
    }

    fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            w1: linear_to_device(&self.w1, device)?,
            w2: linear_to_device(&self.w2, device)?,
            w3: linear_to_device(&self.w3, device)?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = x.apply(&self.w1)?.silu()?;
        let x3 = x.apply(&self.w3)?;
        (x1 * x3)?.apply(&self.w2)
    }
}

fn feed_forward_bytes(feed_forward: &FeedForward) -> usize {
    linear_bytes(&feed_forward.w1) + linear_bytes(&feed_forward.w2) + linear_bytes(&feed_forward.w3)
}

#[derive(Clone)]
struct QkNorm {
    norm_q: RmsNorm,
    norm_k: RmsNorm,
}

impl QkNorm {
    fn new(head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_q: RmsNorm::new(head_dim, eps, vb.pp("norm_q"))?,
            norm_k: RmsNorm::new(head_dim, eps, vb.pp("norm_k"))?,
        })
    }

    fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            norm_q: rms_norm_to_device(&self.norm_q, device)?,
            norm_k: rms_norm_to_device(&self.norm_k, device)?,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.norm_q.forward(q)?, self.norm_k.forward(k)?))
    }
}

fn qk_norm_bytes(norm: &QkNorm) -> usize {
    rms_norm_bytes(&norm.norm_q) + rms_norm_bytes(&norm.norm_k)
}

#[derive(Clone)]
struct ZImageAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    qk_norm: Option<QkNorm>,
    n_heads: usize,
    head_dim: usize,
}

impl ZImageAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        Ok(Self {
            to_q: linear_no_bias(dim, cfg.n_heads * head_dim, vb.pp("to_q"))?,
            to_k: linear_no_bias(dim, cfg.n_kv_heads * head_dim, vb.pp("to_k"))?,
            to_v: linear_no_bias(dim, cfg.n_kv_heads * head_dim, vb.pp("to_v"))?,
            to_out: linear_no_bias(cfg.n_heads * head_dim, dim, vb.pp("to_out").pp("0"))?,
            qk_norm: cfg
                .qk_norm
                .then(|| QkNorm::new(head_dim, cfg.norm_eps, vb.clone()))
                .transpose()?,
            n_heads: cfg.n_heads,
            head_dim,
        })
    }

    fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            to_q: linear_to_device(&self.to_q, device)?,
            to_k: linear_to_device(&self.to_k, device)?,
            to_v: linear_to_device(&self.to_v, device)?,
            to_out: linear_to_device(&self.to_out, device)?,
            qk_norm: self
                .qk_norm
                .as_ref()
                .map(|norm| norm.to_device(device))
                .transpose()?,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        _attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        let q = hidden_states.apply(&self.to_q)?.reshape((
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let k = hidden_states.apply(&self.to_k)?.reshape((
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let v = hidden_states.apply(&self.to_v)?.reshape((
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let (q, k) = if let Some(norm) = &self.qk_norm {
            norm.forward(&q, &k)?
        } else {
            (q, k)
        };
        let q = apply_rotary_emb(&q, cos, sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = apply_rotary_emb(&k, cos, sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let context = crate::attention::attention_default_scale(&q, &k, &v)?;
        context
            .transpose(1, 2)?
            .reshape((batch, seq_len, ()))?
            .apply(&self.to_out)
    }
}

fn zimage_attention_bytes(attention: &ZImageAttention) -> usize {
    linear_bytes(&attention.to_q)
        + linear_bytes(&attention.to_k)
        + linear_bytes(&attention.to_v)
        + linear_bytes(&attention.to_out)
        + attention.qk_norm.as_ref().map(qk_norm_bytes).unwrap_or(0)
}

#[derive(Clone)]
struct ZImageTransformerBlock {
    attention: ZImageAttention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    adaln_modulation: Option<Linear>,
}

fn zimage_transformer_block_bytes(block: &ZImageTransformerBlock) -> usize {
    zimage_attention_bytes(&block.attention)
        + feed_forward_bytes(&block.feed_forward)
        + rms_norm_bytes(&block.attention_norm1)
        + rms_norm_bytes(&block.attention_norm2)
        + rms_norm_bytes(&block.ffn_norm1)
        + rms_norm_bytes(&block.ffn_norm2)
        + block
            .adaln_modulation
            .as_ref()
            .map(linear_bytes)
            .unwrap_or(0)
}

impl ZImageTransformerBlock {
    fn new(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim();
        Ok(Self {
            attention: ZImageAttention::new(cfg, vb.pp("attention"))?,
            feed_forward: FeedForward::new(dim, hidden_dim, vb.pp("feed_forward"))?,
            attention_norm1: RmsNorm::new(dim, cfg.norm_eps, vb.pp("attention_norm1"))?,
            attention_norm2: RmsNorm::new(dim, cfg.norm_eps, vb.pp("attention_norm2"))?,
            ffn_norm1: RmsNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm1"))?,
            ffn_norm2: RmsNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm2"))?,
            adaln_modulation: if modulation {
                let adaln_dim = dim.min(ADALN_EMBED_DIM);
                Some(linear(
                    adaln_dim,
                    4 * dim,
                    vb.pp("adaLN_modulation").pp("0"),
                )?)
            } else {
                None
            },
        })
    }

    fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            attention: self.attention.to_device(device)?,
            feed_forward: self.feed_forward.to_device(device)?,
            attention_norm1: rms_norm_to_device(&self.attention_norm1, device)?,
            attention_norm2: rms_norm_to_device(&self.attention_norm2, device)?,
            ffn_norm1: rms_norm_to_device(&self.ffn_norm1, device)?,
            ffn_norm2: rms_norm_to_device(&self.ffn_norm2, device)?,
            adaln_modulation: self
                .adaln_modulation
                .as_ref()
                .map(|linear| linear_to_device(linear, device))
                .transpose()?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(adaln) = &self.adaln_modulation {
            let adaln_input = adaln_input.expect("adaln_input required when modulation=true");
            let modulation = adaln_input.apply(adaln)?.unsqueeze(1)?;
            let chunks = modulation.chunk(4, D::Minus1)?;
            let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
                (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);
            let gate_msa = gate_msa.tanh()?;
            let gate_mlp = gate_mlp.tanh()?;
            let scale_msa = (scale_msa + 1.0)?;
            let scale_mlp = (scale_mlp + 1.0)?;

            let normed = self.attention_norm1.forward(x)?;
            let scaled = normed.broadcast_mul(&scale_msa)?;
            let attn_out = self.attention.forward(&scaled, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + gate_msa.broadcast_mul(&attn_out)?)?;

            let normed = self.ffn_norm1.forward(&x)?;
            let scaled = normed.broadcast_mul(&scale_mlp)?;
            let ffn_out = self.feed_forward.forward(&scaled)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            x + gate_mlp.broadcast_mul(&ffn_out)?
        } else {
            let normed = self.attention_norm1.forward(x)?;
            let attn_out = self.attention.forward(&normed, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + attn_out)?;
            let normed = self.ffn_norm1.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            x + ffn_out
        }
    }
}

enum ZImageBlockSlot {
    Resident(ZImageTransformerBlock),
    Streamed(ZImageTransformerBlock),
}

fn is_probable_cuda_oom(err: &candle_core::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("cuda_error_out_of_memory")
        || msg.contains("out of memory")
        || msg.contains("memory allocation")
}

fn materialize_zimage_block_slots(
    context_refiner: &[ZImageTransformerBlock],
    noise_refiner: &[ZImageTransformerBlock],
    layers: &[ZImageTransformerBlock],
    plan: &AdaptiveResidencyPlan,
    device: &Device,
) -> Result<(
    Vec<ZImageBlockSlot>,
    Vec<ZImageBlockSlot>,
    Vec<ZImageBlockSlot>,
)> {
    let mut offset = 0usize;
    let mut context_slots = Vec::with_capacity(context_refiner.len());
    for block in context_refiner {
        if plan.resident.get(offset).copied().unwrap_or(false) {
            context_slots.push(ZImageBlockSlot::Resident(block.to_device(device)?));
        } else {
            context_slots.push(ZImageBlockSlot::Streamed(block.clone()));
        }
        offset += 1;
    }

    let mut noise_slots = Vec::with_capacity(noise_refiner.len());
    for block in noise_refiner {
        if plan.resident.get(offset).copied().unwrap_or(false) {
            noise_slots.push(ZImageBlockSlot::Resident(block.to_device(device)?));
        } else {
            noise_slots.push(ZImageBlockSlot::Streamed(block.clone()));
        }
        offset += 1;
    }

    let mut layer_slots = Vec::with_capacity(layers.len());
    for block in layers {
        if plan.resident.get(offset).copied().unwrap_or(false) {
            layer_slots.push(ZImageBlockSlot::Resident(block.to_device(device)?));
        } else {
            layer_slots.push(ZImageBlockSlot::Streamed(block.clone()));
        }
        offset += 1;
    }

    Ok((context_slots, noise_slots, layer_slots))
}

pub(crate) struct OffloadedZImageTransformer {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: Linear,
    x_embedder: Linear,
    final_layer: FinalLayer,
    x_pad_token: Tensor,
    cap_pad_token: Tensor,
    noise_refiner: Vec<ZImageBlockSlot>,
    context_refiner: Vec<ZImageBlockSlot>,
    layers: Vec<ZImageBlockSlot>,
    rope_embedder: RopeEmbedder,
    cfg: Config,
    gpu_device: Device,
}

impl OffloadedZImageTransformer {
    pub(crate) fn new(
        cfg: &Config,
        gpu_vb: VarBuilder,
        cpu_vb: VarBuilder,
        gpu_ordinal: usize,
        activation_budget: u64,
        progress: &ProgressReporter,
    ) -> Result<Self> {
        let gpu_device = gpu_vb.device().clone();
        let dtype = gpu_vb.dtype();
        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);
        let t_embedder = TimestepEmbedder::new(adaln_dim, 1024, gpu_vb.pp("t_embedder"))?;
        let cap_embedder_norm = RmsNorm::new(
            cfg.cap_feat_dim,
            cfg.norm_eps,
            gpu_vb.pp("cap_embedder").pp("0"),
        )?;
        let cap_embedder_linear =
            linear(cfg.cap_feat_dim, cfg.dim, gpu_vb.pp("cap_embedder").pp("1"))?;
        let patch_dim = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.in_channels;
        let x_embedder = linear(patch_dim, cfg.dim, gpu_vb.pp("all_x_embedder").pp("2-1"))?;
        let out_channels = cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_f_patch_size[0]
            * cfg.in_channels;
        let final_layer = FinalLayer::new(
            cfg.dim,
            out_channels,
            gpu_vb.pp("all_final_layer").pp("2-1"),
        )?;
        let x_pad_token = gpu_vb.get((1, cfg.dim), "x_pad_token")?;
        let cap_pad_token = gpu_vb.get((1, cfg.dim), "cap_pad_token")?;

        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for block in zimage_streaming_block_plan(cfg) {
            match block {
                ZImageStreamingBlock::ContextRefiner(i) => {
                    context_refiner.push(ZImageTransformerBlock::new(
                        cfg,
                        false,
                        cpu_vb.pp("context_refiner").pp(i),
                    )?);
                }
                ZImageStreamingBlock::NoiseRefiner(i) => {
                    noise_refiner.push(ZImageTransformerBlock::new(
                        cfg,
                        true,
                        cpu_vb.pp("noise_refiner").pp(i),
                    )?);
                }
                ZImageStreamingBlock::Layer(i) => {
                    layers.push(ZImageTransformerBlock::new(
                        cfg,
                        true,
                        cpu_vb.pp("layers").pp(i),
                    )?);
                }
            }
        }

        let mut block_sizes =
            Vec::with_capacity(context_refiner.len() + noise_refiner.len() + layers.len());
        block_sizes.extend(context_refiner.iter().map(zimage_transformer_block_bytes));
        block_sizes.extend(noise_refiner.iter().map(zimage_transformer_block_bytes));
        block_sizes.extend(layers.iter().map(zimage_transformer_block_bytes));

        let free_vram = crate::device::usable_free_vram_bytes(gpu_ordinal).unwrap_or(0);
        let mut plan = plan_adaptive_residency(
            &block_sizes,
            free_vram,
            activation_budget,
            ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM,
        );

        let (context_refiner, noise_refiner, layers, plan) = loop {
            match materialize_zimage_block_slots(
                &context_refiner,
                &noise_refiner,
                &layers,
                &plan,
                &gpu_device,
            ) {
                Ok((context_slots, noise_slots, layer_slots)) => {
                    break (context_slots, noise_slots, layer_slots, plan);
                }
                Err(err)
                    if gpu_device.is_cuda()
                        && plan.resident_count() > 0
                        && is_probable_cuda_oom(&err) =>
                {
                    progress.info(&format!(
                        "Z-Image adaptive offload: resident allocation OOM at {} resident blocks; \
                         retrying with fewer resident blocks",
                        plan.resident_count()
                    ));
                    if let Err(sync_err) = gpu_device.synchronize() {
                        tracing::warn!(
                            "Z-Image adaptive offload: synchronize after OOM failed: {sync_err}"
                        );
                    }
                    if !plan.demote_largest_resident(&block_sizes) {
                        return Err(err);
                    }
                }
                Err(err) => return Err(err),
            }
        };

        progress.info(&format!(
            "Z-Image adaptive offload: {} resident / {} streamed blocks \
             (resident {:.2} GB, streamed {:.2} GB per denoise pass, reserve {:.2} GB)",
            plan.resident_count(),
            plan.streamed_count(),
            plan.resident_bytes as f64 / 1_000_000_000.0,
            plan.streamed_bytes as f64 / 1_000_000_000.0,
            plan.reserved_bytes() as f64 / 1_000_000_000.0,
        ));

        let rope_embedder = RopeEmbedder::new(
            cfg.rope_theta,
            cfg.axes_dims.clone(),
            cfg.axes_lens.clone(),
            &gpu_device,
            dtype,
        )?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            x_pad_token,
            cap_pad_token,
            context_refiner,
            noise_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
            gpu_device,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        _cap_mask: &Tensor,
    ) -> Result<Tensor> {
        let device = &self.gpu_device;
        let (_batch, _channels, frames, height, width) = x.dims5()?;
        let patch_size = self.cfg.all_patch_size[0];
        let frame_patch_size = self.cfg.all_f_patch_size[0];

        let t_scaled = (t.to_device(device)? * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?;
        let cap_feats = cap_feats.to_device(device)?;
        let cap = self.cap_embedder_norm.forward(&cap_feats)?;
        let cap = cap.apply(&self.cap_embedder_linear)?;
        let (mut cap, _) = pad_token_sequence(&cap, &self.cap_pad_token, SEQ_MULTI_OF)?;
        let padded_text_len = cap.dim(1)?;
        let cap_pos_ids = create_coordinate_grid((padded_text_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        let x = x.to_device(device)?;
        let (x_patches, orig_size) = patchify(&x, patch_size, frame_patch_size)?;
        let x = x_patches.apply(&self.x_embedder)?;
        let (mut image, image_pad_extra) = pad_token_sequence(&x, &self.x_pad_token, SEQ_MULTI_OF)?;
        let padded_image_seq_len = image.dim(1)?;
        let frame_tokens = frames / frame_patch_size;
        let height_tokens = height / patch_size;
        let width_tokens = width / patch_size;
        let image_pos_ids = create_coordinate_grid(
            (frame_tokens, height_tokens, width_tokens),
            (padded_text_len + 1, 0, 0),
            device,
        )?;
        let image_pos_ids = pad_position_ids_with_zeros(&image_pos_ids, image_pad_extra)?;
        let (image_cos, image_sin) = self.rope_embedder.forward(&image_pos_ids)?;

        for block in &self.context_refiner {
            match block {
                ZImageBlockSlot::Resident(block) => {
                    cap = block.forward(&cap, None, &cap_cos, &cap_sin, None)?;
                }
                ZImageBlockSlot::Streamed(block) => {
                    let block = block.to_device(device)?;
                    cap = block.forward(&cap, None, &cap_cos, &cap_sin, None)?;
                    device.synchronize()?;
                    drop(block);
                }
            }
        }
        for block in &self.noise_refiner {
            match block {
                ZImageBlockSlot::Resident(block) => {
                    image =
                        block.forward(&image, None, &image_cos, &image_sin, Some(&adaln_input))?;
                }
                ZImageBlockSlot::Streamed(block) => {
                    let block = block.to_device(device)?;
                    image =
                        block.forward(&image, None, &image_cos, &image_sin, Some(&adaln_input))?;
                    device.synchronize()?;
                    drop(block);
                }
            }
        }

        let (mut unified, unified_pos_ids) =
            build_basic_unified_sequence(&image, &cap, &image_pos_ids, &cap_pos_ids)?;
        let (unified_cos, unified_sin) = self.rope_embedder.forward(&unified_pos_ids)?;
        for block in &self.layers {
            match block {
                ZImageBlockSlot::Resident(block) => {
                    unified = block.forward(
                        &unified,
                        None,
                        &unified_cos,
                        &unified_sin,
                        Some(&adaln_input),
                    )?;
                }
                ZImageBlockSlot::Streamed(block) => {
                    let block = block.to_device(device)?;
                    unified = block.forward(
                        &unified,
                        None,
                        &unified_cos,
                        &unified_sin,
                        Some(&adaln_input),
                    )?;
                    device.synchronize()?;
                    drop(block);
                }
            }
        }

        let image = unified.narrow(1, 0, padded_image_seq_len)?;
        let image = self.final_layer.forward(&image, &adaln_input)?;
        unpatchify(
            &image,
            orig_size,
            patch_size,
            frame_patch_size,
            self.cfg.in_channels,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zimage_streaming_block_plan_preserves_reference_order() {
        let mut cfg = Config::z_image_turbo();
        cfg.n_refiner_layers = 2;
        cfg.n_layers = 3;

        assert_eq!(
            zimage_streaming_block_plan(&cfg),
            vec![
                ZImageStreamingBlock::ContextRefiner(0),
                ZImageStreamingBlock::ContextRefiner(1),
                ZImageStreamingBlock::NoiseRefiner(0),
                ZImageStreamingBlock::NoiseRefiner(1),
                ZImageStreamingBlock::Layer(0),
                ZImageStreamingBlock::Layer(1),
                ZImageStreamingBlock::Layer(2),
            ]
        );
    }
}
