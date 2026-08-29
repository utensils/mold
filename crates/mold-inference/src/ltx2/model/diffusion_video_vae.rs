//! LTX-2.5 neighborhood-attention diffusion video decoder.
//!
//! Stages 1-4 deterministically refine and pixel-shuffle the latent context.
//! Stage 5 performs the checkpoint's single x0 diffusion step against fixed
//! seed-zero noise. Tensor names match the official split checkpoint.

use anyhow::{ensure, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear_b, linear_no_bias, Linear, VarBuilder};

const HEAD_DIM: usize = 64;
const MLP_TOKEN_CHUNK: usize = 16_384;
const QKV_WORKSPACE_ELEMENTS: usize = 1 << 25;
const PATCH_SIZE: usize = 4;
const T_EMBED_DIM: usize = 384;
const ROPE_SPLIT: [usize; 3] = [16, 24, 24];

#[derive(Debug, Clone)]
struct LastDimRmsNorm {
    weight: Tensor,
}

impl LastDimRmsNorm {
    fn new(dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let normalized = xs
            .to_dtype(DType::F32)?
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, 1e-6)?
            .sqrt()?
            .recip()?
            .broadcast_mul(&xs.to_dtype(DType::F32)?)?;
        Ok(normalized
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)?)
    }
}

/// candle's `Linear` only matmuls rank <= 4 inputs, so every projection over
/// the decoder's rank-5 (batch, time, height, width, channels) grids runs on
/// a (batch, tokens, channels) flattening and is reshaped back.
fn linear_5d(layer: &Linear, xs: &Tensor) -> Result<Tensor> {
    let (batch, time, height, width, dim) = xs.dims5()?;
    let flat = xs.reshape((batch, time * height * width, dim))?;
    let projected = layer.forward(&flat)?;
    let out = projected.dim(D::Minus1)?;
    Ok(projected.reshape((batch, time, height, width, out))?)
}

#[derive(Debug, Clone)]
struct SwiGlu {
    up: Linear,
    gate: Linear,
    down: Linear,
}

impl SwiGlu {
    fn new(dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = dim * 4;
        Ok(Self {
            up: linear_no_bias(dim, hidden, vb.pp("w_up"))?,
            gate: linear_no_bias(dim, hidden, vb.pp("w_gate"))?,
            down: linear_no_bias(hidden, dim, vb.pp("w_down"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, time, height, width, dim) = xs.dims5()?;
        let tokens = time * height * width;
        // candle's `Linear` only matmuls rank <= 4 inputs, so every branch
        // runs on the (batch, tokens, dim) flattening of the rank-5 grid.
        let flat = xs.reshape((batch, tokens, dim))?;
        if tokens <= MLP_TOKEN_CHUNK {
            let gate = candle_nn::ops::silu(&self.gate.forward(&flat)?)?;
            return Ok(self
                .down
                .forward(&(gate * self.up.forward(&flat)?)?)?
                .reshape(xs.shape())?);
        }

        let mut output = Vec::with_capacity(tokens.div_ceil(MLP_TOKEN_CHUNK));
        for start in (0..tokens).step_by(MLP_TOKEN_CHUNK) {
            let length = MLP_TOKEN_CHUNK.min(tokens - start);
            let tile = flat.narrow(1, start, length)?;
            let gate = candle_nn::ops::silu(&self.gate.forward(&tile)?)?;
            output.push(self.down.forward(&(gate * self.up.forward(&tile)?)?)?);
        }
        let output = output.iter().collect::<Vec<_>>();
        Ok(Tensor::cat(&output, 1)?.reshape(xs.shape())?)
    }
}

#[derive(Debug, Clone)]
struct NeighborhoodAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: LastDimRmsNorm,
    k_norm: LastDimRmsNorm,
    heads: usize,
    kernel: [usize; 3],
}

impl NeighborhoodAttention {
    fn new(dim: usize, kernel: [usize; 3], vb: VarBuilder<'_>) -> Result<Self> {
        ensure!(
            dim.is_multiple_of(HEAD_DIM),
            "diffusion VAE attention width must divide by 64"
        );
        Ok(Self {
            qkv: linear_b(dim, dim * 3, true, vb.pp("qkv"))?,
            proj: linear_b(dim, dim, true, vb.pp("proj"))?,
            q_norm: LastDimRmsNorm::new(HEAD_DIM, vb.pp("q_norm"))?,
            k_norm: LastDimRmsNorm::new(HEAD_DIM, vb.pp("k_norm"))?,
            heads: dim / HEAD_DIM,
            kernel,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, time, height, width, dim) = xs.dims5()?;
        let time_chunk = (QKV_WORKSPACE_ELEMENTS / (height * width * dim).max(1)).max(1);
        let mut q_tiles = Vec::with_capacity(time.div_ceil(time_chunk));
        let mut k_tiles = Vec::with_capacity(q_tiles.capacity());
        let mut v_tiles = Vec::with_capacity(q_tiles.capacity());
        for start in (0..time).step_by(time_chunk) {
            let length = time_chunk.min(time - start);
            let qkv = linear_5d(&self.qkv, &xs.narrow(1, start, length)?)?;
            let shape = (batch, length, height, width, self.heads, HEAD_DIM);
            let q = qkv.narrow(D::Minus1, 0, dim)?.reshape(shape)?;
            let k = qkv.narrow(D::Minus1, dim, dim)?.reshape(shape)?;
            let v = qkv.narrow(D::Minus1, dim * 2, dim)?.reshape(shape)?;
            q_tiles.push(apply_absolute_rope(
                &self.q_norm.forward(&q)?,
                start,
                height,
                width,
            )?);
            k_tiles.push(apply_absolute_rope(
                &self.k_norm.forward(&k)?,
                start,
                height,
                width,
            )?);
            v_tiles.push(v);
        }
        let q_refs = q_tiles.iter().collect::<Vec<_>>();
        let k_refs = k_tiles.iter().collect::<Vec<_>>();
        let v_refs = v_tiles.iter().collect::<Vec<_>>();
        let q = Tensor::cat(&q_refs, 1)?;
        let k = Tensor::cat(&k_refs, 1)?;
        let v = Tensor::cat(&v_refs, 1)?;
        let attended = candle_core::neighborhood_attention::neighborhood_attention3d(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            self.kernel,
            (HEAD_DIM as f32).sqrt().recip(),
        )?;
        let attended = attended.reshape((batch, time, height, width, dim))?;
        let mut output = Vec::with_capacity(time.div_ceil(time_chunk));
        for start in (0..time).step_by(time_chunk) {
            let length = time_chunk.min(time - start);
            output.push(linear_5d(&self.proj, &attended.narrow(1, start, length)?)?);
        }
        let output = output.iter().collect::<Vec<_>>();
        Ok(Tensor::cat(&output, 1)?)
    }
}

#[derive(Debug, Clone)]
struct NaBlock {
    norm1: LastDimRmsNorm,
    attention: NeighborhoodAttention,
    norm2: LastDimRmsNorm,
    mlp: SwiGlu,
}

impl NaBlock {
    fn new(dim: usize, kernel: [usize; 3], vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            norm1: LastDimRmsNorm::new(dim, vb.pp("norm1"))?,
            attention: NeighborhoodAttention::new(dim, kernel, vb.pp("attn"))?,
            norm2: LastDimRmsNorm::new(dim, vb.pp("norm2"))?,
            mlp: SwiGlu::new(dim, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attention.forward(&self.norm1.forward(xs)?)?)?;
        Ok((&xs + self.mlp.forward(&self.norm2.forward(&xs)?)?)?)
    }
}

#[derive(Debug, Clone)]
struct PixelShuffleUpsample {
    proj: Linear,
    stride: [usize; 3],
    out_channels: usize,
}

impl PixelShuffleUpsample {
    fn new(
        channels: usize,
        stride: [usize; 3],
        reduction: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let volume: usize = stride.iter().product();
        let projected = volume * channels / reduction;
        Ok(Self {
            proj: linear_b(channels, projected, true, vb.pp("proj"))?,
            stride,
            out_channels: projected / volume,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, time, height, width, _) = xs.dims5()?;
        let [pt, ph, pw] = self.stride;
        let projected = linear_5d(&self.proj, xs)?.reshape(&[
            batch,
            time,
            height,
            width,
            self.out_channels,
            pt,
            ph,
            pw,
        ])?;
        let shuffled = projected.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?.reshape((
            batch,
            time * pt,
            height * ph,
            width * pw,
            self.out_channels,
        ))?;
        if pt == 2 {
            Ok(shuffled.narrow(1, 1, time * pt - 1)?)
        } else {
            Ok(shuffled)
        }
    }
}

#[derive(Debug, Clone)]
struct DeterministicStage {
    blocks: Vec<NaBlock>,
    upsample: PixelShuffleUpsample,
}

#[derive(Debug, Clone)]
struct DiffusionBlock {
    context_proj: Linear,
    scale_shift_table: Tensor,
    norm1: LastDimRmsNorm,
    attention: NeighborhoodAttention,
    norm2: LastDimRmsNorm,
    mlp: SwiGlu,
}

impl DiffusionBlock {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            context_proj: linear_b(256, 256, true, vb.pp("context_proj"))?,
            scale_shift_table: vb.get((7, 256), "scale_shift_table")?,
            norm1: LastDimRmsNorm::new(256, vb.pp("norm1"))?,
            attention: NeighborhoodAttention::new(256, [11, 11, 11], vb.pp("attn"))?,
            norm2: LastDimRmsNorm::new(256, vb.pp("norm2"))?,
            mlp: SwiGlu::new(256, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, context: &Tensor, modulation: &[Tensor]) -> Result<Tensor> {
        let table = self
            .scale_shift_table
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let parameter = |index: usize| -> Result<Tensor> {
            Ok(modulation[index].broadcast_add(&table.narrow(4, index, 1)?.squeeze(4)?)?)
        };
        let scale_msa = parameter(0)?;
        let shift_msa = parameter(1)?;
        let scale_mlp = parameter(3)?;
        let shift_mlp = parameter(4)?;
        let mut xs = (xs + linear_5d(&self.context_proj, context)?)?;
        let normed = modulate(&self.norm1.forward(&xs)?, &scale_msa, &shift_msa)?;
        xs = (&xs + self.attention.forward(&normed)?)?;
        let normed = modulate(&self.norm2.forward(&xs)?, &scale_mlp, &shift_mlp)?;
        Ok((&xs + self.mlp.forward(&normed)?)?)
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2DiffusionVideoDecoder {
    conv_in: Linear,
    deterministic: Vec<DeterministicStage>,
    timestep_0: Linear,
    timestep_2: Linear,
    conv_in_x_t: Linear,
    shared_adaln: Linear,
    diffusion: Vec<DiffusionBlock>,
    norm_out: LastDimRmsNorm,
    conv_out: Linear,
}

impl Ltx2DiffusionVideoDecoder {
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let channels = [2048, 1024, 512, 512, 256];
        let depths = [4, 6, 4, 2];
        let kernels = [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5]];
        let upsamples = [
            ([1, 2, 2], 2),
            ([2, 1, 1], 2),
            ([2, 2, 2], 1),
            ([2, 2, 2], 2),
        ];
        let mut deterministic = Vec::with_capacity(4);
        for stage in 0..4 {
            deterministic.push(DeterministicStage {
                blocks: (0..depths[stage])
                    .map(|block| {
                        NaBlock::new(
                            channels[stage],
                            kernels[stage],
                            vb.pp(format!("det_stages.{stage}.{block}")),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
                upsample: PixelShuffleUpsample::new(
                    channels[stage],
                    upsamples[stage].0,
                    upsamples[stage].1,
                    vb.pp(format!("upsamples.{stage}")),
                )?,
            });
        }
        Ok(Self {
            conv_in: linear_b(128, 2048, true, vb.pp("conv_in"))?,
            deterministic,
            timestep_0: linear_b(256, T_EMBED_DIM, true, vb.pp("t_embedder.mlp.0"))?,
            timestep_2: linear_b(T_EMBED_DIM, T_EMBED_DIM, true, vb.pp("t_embedder.mlp.2"))?,
            conv_in_x_t: linear_b(3 * PATCH_SIZE * PATCH_SIZE, 256, true, vb.pp("conv_in_x_t"))?,
            shared_adaln: linear_b(T_EMBED_DIM, 7 * 256, true, vb.pp("shared_adaln.proj"))?,
            diffusion: (0..8)
                .map(|block| DiffusionBlock::new(vb.pp(format!("diff_blocks.{block}"))))
                .collect::<Result<Vec<_>>>()?,
            norm_out: LastDimRmsNorm::new(256, vb.pp("norm_out"))?,
            conv_out: linear_b(256, 3 * PATCH_SIZE * PATCH_SIZE, true, vb.pp("conv_out"))?,
        })
    }

    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        let (_, _, latent_time, _, _) = latents.dims5()?;
        let trailing = latents.narrow(2, latent_time - 1, 1)?.broadcast_as((
            latents.dim(0)?,
            latents.dim(1)?,
            2,
            latents.dim(3)?,
            latents.dim(4)?,
        ))?;
        let padded = Tensor::cat(&[latents, &trailing], 2)?;
        let mut context = linear_5d(
            &self.conv_in,
            &padded.permute((0, 2, 3, 4, 1))?.contiguous()?,
        )?;
        for stage in &self.deterministic {
            for block in &stage.blocks {
                context = block.forward(&context)?;
            }
            context = stage.upsample.forward(&context)?;
        }
        let context_time = context.dim(1)?;
        context = context.narrow(1, 0, context_time - 16)?;

        let (batch, time, height, width, _) = context.dims5()?;
        latents.device().set_seed(0)?;
        let noise = Tensor::randn(
            0f32,
            1f32,
            (batch, 3, time, height * PATCH_SIZE, width * PATCH_SIZE),
            latents.device(),
        )?
        .to_dtype(latents.dtype())?;
        let patched = patchify_pixels(&noise, PATCH_SIZE)?.permute((0, 2, 3, 4, 1))?;
        let mut xs = linear_5d(&self.conv_in_x_t, &patched.contiguous()?)?;
        let timestep = timestep_embedding(1000.0, batch, latents.device(), latents.dtype())?;
        let timestep = self
            .timestep_2
            .forward(&candle_nn::ops::silu(&self.timestep_0.forward(&timestep)?)?)?;
        let adaln = self
            .shared_adaln
            .forward(&candle_nn::ops::silu(&timestep)?)?
            .reshape((batch, 7, 256))?;
        let modulation = (0..7)
            .map(|index| {
                adaln
                    .narrow(1, index, 1)?
                    .squeeze(1)?
                    .unsqueeze(1)?
                    .unsqueeze(1)?
                    .unsqueeze(1)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        for block in &self.diffusion {
            xs = block.forward(&xs, &context, &modulation)?;
        }
        let pixels =
            linear_5d(&self.conv_out, &self.norm_out.forward(&xs)?)?.permute((0, 4, 1, 2, 3))?;
        unpatchify_pixels(&pixels, PATCH_SIZE)
    }
}

fn modulate(xs: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    Ok(xs.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?)
}

fn apply_absolute_rope(
    xs: &Tensor,
    time_offset: usize,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let [batch, _, _, _, heads, dim] = xs.dims() else {
        anyhow::bail!("diffusion VAE RoPE expects rank-6 attention tensors")
    };
    let (batch, time, heads, dim) = (*batch, xs.dim(1)?, *heads, *dim);
    ensure!(dim == HEAD_DIM, "diffusion VAE RoPE expects 64-wide heads");
    let mut cos = Vec::with_capacity(time * height * width * (dim / 2));
    let mut sin = Vec::with_capacity(cos.capacity());
    for local_t in 0..time {
        let t = time_offset + local_t;
        for h in 0..height {
            for w in 0..width {
                for (axis, axis_dim) in [t, h, w].into_iter().zip(ROPE_SPLIT) {
                    for pair in 0..axis_dim / 2 {
                        let frequency = 10_000f64.powf(-((2 * pair) as f64) / axis_dim as f64);
                        let angle = axis as f64 * frequency;
                        cos.push(angle.cos() as f32);
                        sin.push(angle.sin() as f32);
                    }
                }
            }
        }
    }
    let table_shape = (1, time, height, width, 1, dim / 2);
    let cos = Tensor::from_vec(cos, table_shape, &Device::Cpu)?
        .to_device(xs.device())?
        .to_dtype(xs.dtype())?;
    let sin = Tensor::from_vec(sin, table_shape, &Device::Cpu)?
        .to_device(xs.device())?
        .to_dtype(xs.dtype())?;
    let pairs = xs.reshape(&[batch, time, height, width, heads, dim / 2, 2])?;
    let even = pairs.i((.., .., .., .., .., .., 0))?;
    let odd = pairs.i((.., .., .., .., .., .., 1))?;
    let rotated_even = (even.broadcast_mul(&cos)? - odd.broadcast_mul(&sin)?)?;
    let rotated_odd = (even.broadcast_mul(&sin)? + odd.broadcast_mul(&cos)?)?;
    Ok(Tensor::stack(&[rotated_even, rotated_odd], D::Minus1)?.reshape(xs.shape())?)
}

fn timestep_embedding(
    timestep: f32,
    batch: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let half = 128usize;
    let mut values = Vec::with_capacity(256);
    for pair in 0..half {
        let frequency = 10_000f64.powf(-(pair as f64) / half as f64);
        values.push((f64::from(timestep) * frequency).cos() as f32);
    }
    for pair in 0..half {
        let frequency = 10_000f64.powf(-(pair as f64) / half as f64);
        values.push((f64::from(timestep) * frequency).sin() as f32);
    }
    Ok(Tensor::from_vec(values, (1, 256), &Device::Cpu)?
        .broadcast_as((batch, 256))?
        .contiguous()?
        .to_device(device)?
        .to_dtype(dtype)?)
}

fn patchify_pixels(xs: &Tensor, patch: usize) -> Result<Tensor> {
    let (batch, channels, time, height, width) = xs.dims5()?;
    ensure!(
        height % patch == 0 && width % patch == 0,
        "pixel grid must divide by patch size"
    );
    Ok(xs
        .reshape(&[
            batch,
            channels,
            time,
            height / patch,
            patch,
            width / patch,
            patch,
        ])?
        .permute(vec![0, 1, 6, 4, 2, 3, 5])?
        .reshape((
            batch,
            channels * patch * patch,
            time,
            height / patch,
            width / patch,
        ))?)
}

fn unpatchify_pixels(xs: &Tensor, patch: usize) -> Result<Tensor> {
    let (batch, channels, time, height, width) = xs.dims5()?;
    ensure!(
        channels % (patch * patch) == 0,
        "patchified channels are invalid"
    );
    let output_channels = channels / (patch * patch);
    Ok(xs
        .reshape(&[batch, output_channels, patch, patch, time, height, width])?
        .permute(vec![0, 1, 4, 5, 3, 6, 2])?
        .reshape((batch, output_channels, time, height * patch, width * patch))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_patch_round_trip() {
        let xs = Tensor::arange(0f32, 3.0 * 2.0 * 8.0 * 8.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 3, 2, 8, 8))
            .unwrap();
        let patched = patchify_pixels(&xs, 4).unwrap();
        assert_eq!(patched.dims(), &[1, 48, 2, 2, 2]);
        assert_eq!(
            unpatchify_pixels(&patched, 4)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            xs.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn swiglu_loads_from_bias_free_checkpoint_weights() {
        // The shipped ltx-2.5 diffusion VAE stores its SwiGLU MLPs as
        // weight-only tensors (w_up/w_gate/w_down have no `.bias` entries),
        // so construction must not demand biases.
        let dim = 4usize;
        let hidden = dim * 4;
        let mut tensors = std::collections::HashMap::new();
        let dev = Device::Cpu;
        tensors.insert(
            "w_up.weight".to_string(),
            Tensor::zeros((hidden, dim), DType::F32, &dev).unwrap(),
        );
        tensors.insert(
            "w_gate.weight".to_string(),
            Tensor::zeros((hidden, dim), DType::F32, &dev).unwrap(),
        );
        tensors.insert(
            "w_down.weight".to_string(),
            Tensor::zeros((dim, hidden), DType::F32, &dev).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let mlp = SwiGlu::new(dim, vb).expect("bias-free SwiGLU weights must load");
        let xs = Tensor::zeros((1, 1, 2, 2, dim), DType::F32, &dev).unwrap();
        assert_eq!(mlp.forward(&xs).unwrap().dims(), &[1, 1, 2, 2, dim]);
    }

    #[test]
    fn neighborhood_attention_forwards_a_rank5_grid() {
        let dim = HEAD_DIM;
        let dev = Device::Cpu;
        let mut tensors = std::collections::HashMap::new();
        for name in ["qkv", "proj"] {
            let out = if name == "qkv" { dim * 3 } else { dim };
            tensors.insert(
                format!("{name}.weight"),
                Tensor::zeros((out, dim), DType::F32, &dev).unwrap(),
            );
            tensors.insert(
                format!("{name}.bias"),
                Tensor::zeros(out, DType::F32, &dev).unwrap(),
            );
        }
        for name in ["q_norm", "k_norm"] {
            tensors.insert(
                format!("{name}.weight"),
                Tensor::ones(HEAD_DIM, DType::F32, &dev).unwrap(),
            );
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let attn = NeighborhoodAttention::new(dim, [1, 3, 3], vb).unwrap();
        let xs = Tensor::zeros((1, 2, 4, 4, dim), DType::F32, &dev).unwrap();
        assert_eq!(attn.forward(&xs).unwrap().dims(), &[1, 2, 4, 4, dim]);
    }

    #[test]
    fn pixel_shuffle_upsample_forwards_a_rank5_grid() {
        let channels = 8usize;
        let stride = [1usize, 2, 2];
        let reduction = 2usize;
        let dev = Device::Cpu;
        let projected = stride.iter().product::<usize>() * channels / reduction;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "proj.weight".to_string(),
            Tensor::zeros((projected, channels), DType::F32, &dev).unwrap(),
        );
        tensors.insert(
            "proj.bias".to_string(),
            Tensor::zeros(projected, DType::F32, &dev).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let upsample = PixelShuffleUpsample::new(channels, stride, reduction, vb).unwrap();
        let xs = Tensor::zeros((1, 2, 4, 4, channels), DType::F32, &dev).unwrap();
        let out = upsample.forward(&xs).unwrap();
        assert_eq!(out.dims(), &[1, 2, 8, 8, channels / reduction]);
    }

    #[test]
    fn pixel_patch_packing_matches_upstream_width_then_height_order() {
        let xs = Tensor::from_vec(vec![0f32, 1., 2., 3.], (1, 1, 1, 2, 2), &Device::Cpu).unwrap();
        assert_eq!(
            patchify_pixels(&xs, 2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0., 2., 1., 3.]
        );
    }
}
