// Copyright 2024 The HuggingFace Team. All rights reserved.
// Ported and modified in Rust for mold: inference-only VAE blocks, explicit
// numerical policy and bounded attention. Apache-2.0; see THIRD_PARTY_NOTICES.md.
//! Diffusers VAE blocks with PyTorch's opmath precision for normalization and
//! SiLU. Architecture: diffusers v0.30.0 (8a79d8ec), models/resnet.py:267-377,
//! unets/unet_2d_blocks.py DownEncoderBlock2D/UpDecoderBlock2D/UNetMidBlock2D.
//! No training/dropout/time-embedding branches are needed by AutoencoderKL.
use super::VaeNumerics;
use candle::{DType, Module, Result, Tensor};
use candle_nn as nn;
use std::fmt::Debug;

pub(super) trait Block: Module + Debug + Send + Sync {}
impl<T: Module + Debug + Send + Sync> Block for T {}

#[derive(Debug)]
pub(super) struct LegacyMid(pub super::UNetMidBlock2D);
impl Module for LegacyMid {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs, None)
    }
}

#[derive(Debug)]
pub(super) enum Norm {
    Candle(nn::GroupNorm),
    Diffusers(crate::stable_diffusion::normalization::DiffusersGroupNorm),
}
impl Norm {
    pub(super) fn new(
        vb: nn::VarBuilder,
        groups: usize,
        channels: usize,
        numerics: VaeNumerics,
    ) -> Result<Self> {
        if numerics == VaeNumerics::Diffusers {
            Ok(Self::Diffusers(
                crate::stable_diffusion::normalization::DiffusersGroupNorm::new(
                    vb, groups, channels, 1e-6,
                )?,
            ))
        } else {
            Ok(Self::Candle(nn::group_norm(groups, channels, 1e-6, vb)?))
        }
    }
}
impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Candle(norm) => norm.forward(xs),
            Self::Diffusers(norm) => norm.forward(xs),
        }
    }
}
pub(super) fn silu(xs: &Tensor, numerics: VaeNumerics) -> Result<Tensor> {
    if numerics == VaeNumerics::Diffusers && matches!(xs.dtype(), DType::F16 | DType::BF16) {
        xs.to_dtype(DType::F32)?.silu()?.to_dtype(xs.dtype())
    } else {
        xs.silu()
    }
}
fn conv(
    vb: nn::VarBuilder,
    input: usize,
    output: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
) -> Result<nn::Conv2d> {
    nn::conv2d(
        input,
        output,
        kernel,
        nn::Conv2dConfig {
            padding,
            stride,
            ..Default::default()
        },
        vb,
    )
}

#[derive(Debug)]
struct Resnet {
    norm1: Norm,
    norm2: Norm,
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
    shortcut: Option<nn::Conv2d>,
}
impl Resnet {
    fn new(vb: nn::VarBuilder, input: usize, output: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            norm1: Norm::new(vb.pp("norm1"), groups, input, VaeNumerics::Diffusers)?,
            norm2: Norm::new(vb.pp("norm2"), groups, output, VaeNumerics::Diffusers)?,
            conv1: conv(vb.pp("conv1"), input, output, 3, 1, 1)?,
            conv2: conv(vb.pp("conv2"), output, output, 3, 1, 1)?,
            shortcut: if input != output {
                Some(conv(vb.pp("conv_shortcut"), input, output, 1, 0, 1)?)
            } else {
                None
            },
        })
    }
}
impl Module for Resnet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = match &self.shortcut {
            Some(conv) => conv.forward(xs)?,
            None => xs.clone(),
        };
        let xs = self
            .conv1
            .forward(&silu(&self.norm1.forward(xs)?, VaeNumerics::Diffusers)?)?;
        let xs = self
            .conv2
            .forward(&silu(&self.norm2.forward(&xs)?, VaeNumerics::Diffusers)?)?;
        xs + residual
    }
}

#[derive(Debug)]
pub(super) struct Down {
    resnets: Vec<Resnet>,
    downsample: Option<nn::Conv2d>,
}
impl Down {
    pub(super) fn new(
        vb: nn::VarBuilder,
        input: usize,
        output: usize,
        layers: usize,
        groups: usize,
        downsample: bool,
    ) -> Result<Self> {
        let resnets = (0..layers)
            .map(|i| {
                Resnet::new(
                    vb.pp(format!("resnets.{i}")),
                    if i == 0 { input } else { output },
                    output,
                    groups,
                )
            })
            .collect::<Result<_>>()?;
        let downsample = if downsample {
            Some(conv(vb.pp("downsamplers.0.conv"), output, output, 3, 0, 2)?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsample,
        })
    }
}
impl Module for Down {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for resnet in &self.resnets {
            xs = resnet.forward(&xs)?;
        }
        if let Some(conv) = &self.downsample {
            xs = conv.forward(&xs.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
pub(super) struct Up {
    resnets: Vec<Resnet>,
    upsample: Option<nn::Conv2d>,
}
impl Up {
    pub(super) fn new(
        vb: nn::VarBuilder,
        input: usize,
        output: usize,
        layers: usize,
        groups: usize,
        upsample: bool,
    ) -> Result<Self> {
        let resnets = (0..layers)
            .map(|i| {
                Resnet::new(
                    vb.pp(format!("resnets.{i}")),
                    if i == 0 { input } else { output },
                    output,
                    groups,
                )
            })
            .collect::<Result<_>>()?;
        let upsample = if upsample {
            Some(conv(vb.pp("upsamplers.0.conv"), output, output, 3, 1, 1)?)
        } else {
            None
        };
        Ok(Self { resnets, upsample })
    }
}
impl Module for Up {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for resnet in &self.resnets {
            xs = resnet.forward(&xs)?;
        }
        if let Some(conv) = &self.upsample {
            let (_, _, height, width) = xs.dims4()?;
            xs = conv.forward(&xs.upsample_nearest2d(height * 2, width * 2)?)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
pub(super) struct Mid {
    first: Resnet,
    last: Resnet,
    norm: Norm,
    query: AttentionLinear,
    key: AttentionLinear,
    value: AttentionLinear,
    output: AttentionLinear,
}
impl Mid {
    pub(super) fn new(vb: nn::VarBuilder, channels: usize, groups: usize) -> Result<Self> {
        let attention = vb.pp("attentions.0");
        // Published paint VAE uses Diffusers' pre-0.17 names. The oracle's
        // from_pretrained migrates these to the current Attention names.
        let (q, k, v, out) = if attention.contains_tensor("to_q.weight") {
            ("to_q", "to_k", "to_v", "to_out.0")
        } else {
            ("query", "key", "value", "proj_attn")
        };
        Ok(Self {
            first: Resnet::new(vb.pp("resnets.0"), channels, channels, groups)?,
            last: Resnet::new(vb.pp("resnets.1"), channels, channels, groups)?,
            norm: Norm::new(
                attention.pp("group_norm"),
                groups,
                channels,
                VaeNumerics::Diffusers,
            )?,
            query: attention_linear(attention.pp(q), channels)?,
            key: attention_linear(attention.pp(k), channels)?,
            value: attention_linear(attention.pp(v), channels)?,
            output: attention_linear(attention.pp(out), channels)?,
        })
    }
}

#[derive(Debug)]
struct AttentionLinear(nn::Linear);
impl Module for AttentionLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        crate::stable_diffusion::linear::forward(&self.0, xs)
    }
}
fn attention_linear(vb: nn::VarBuilder, channels: usize) -> Result<AttentionLinear> {
    let weight = match vb.get((channels, channels), "weight") {
        Ok(weight) => weight,
        Err(_) => vb
            .get((channels, channels, 1, 1), "weight")?
            .reshape((channels, channels))?,
    };
    Ok(AttentionLinear(nn::Linear::new(
        weight.to_dtype(DType::F32)?,
        Some(vb.get(channels, "bias")?.to_dtype(DType::F32)?),
    )))
}
impl Module for Mid {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.first.forward(xs)?;
        let (batch, channels, height, width) = xs.dims4()?;
        let hidden = self
            .norm
            .forward(&xs)?
            .reshape((batch, channels, height * width))?
            // Diffusers AttnProcessor2_0:2188 preserves this non-contiguous
            // view, so Q/K/V use matmul then a separate half bias addition.
            .transpose(1, 2)?;
        let query = self.query.forward(&hidden)?.to_dtype(DType::F32)?;
        let key = self
            .key
            .forward(&hidden)?
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?;
        let value = self.value.forward(&hidden)?.to_dtype(DType::F32)?;
        // One attention head in this VAE. Bound each score allocation to 64 MiB
        // without changing which keys any query can attend to.
        let tokens = height * width;
        let chunk = (16_777_216 / batch / tokens).max(1).min(tokens);
        let mut attended = Vec::new();
        for start in (0..tokens).step_by(chunk) {
            let count = chunk.min(tokens - start);
            let scores = (query.narrow(1, start, count)?.contiguous()?.matmul(&key)?
                / (channels as f64).sqrt())?;
            attended.push(nn::ops::softmax_last_dim(&scores)?.matmul(&value)?);
        }
        let hidden = Tensor::cat(&attended, 1)?.to_dtype(xs.dtype())?;
        let hidden = self
            .output
            .forward(&hidden)?
            .transpose(1, 2)?
            .reshape((batch, channels, height, width))?;
        self.last.forward(&(xs + hidden)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn noncontiguous_vae_attention_linear_rounds_product_before_bias() -> Result<()> {
        let device = candle::Device::Cpu;
        let x = Tensor::new(&[[[1f32 + 1. / 1024., 1. + 1. / 1024.], [0., 0.]]], &device)?
            .to_dtype(DType::F16)?
            .transpose(1, 2)?;
        let layer = AttentionLinear(nn::Linear::new(
            Tensor::new(&[[450.25f32, 0.]], &device)?,
            Some(Tensor::new(&[1f32 + 427. / 1024.], &device)?),
        ));
        assert_eq!(
            layer
                .forward(&x)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![452.25, 452.25]
        );
        Ok(())
    }
    #[test]
    fn unet_normalization_uses_its_own_epsilon() -> Result<()> {
        let device = candle::Device::Cpu;
        let fixture = candle::safetensors::load_buffer(
            include_bytes!("../../../../../tests/fixtures/hunyuan3d/paint-conv.safetensors"),
            &device,
        )?;
        let vb = nn::VarBuilder::from_tensors(fixture.clone(), DType::F16, &device).pp("norm.unet");
        let actual =
            crate::stable_diffusion::normalization::DiffusersGroupNorm::new(vb, 4, 32, 1e-5)?
                .forward(&fixture["norm.unet.input"])?;
        let error = (actual.to_dtype(DType::F32)?
            - fixture["norm.unet.expected"].to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
        assert!(error <= 0.002, "UNet epsilon error {error}");
        Ok(())
    }

    #[test]
    fn normalization_and_silu_match_pytorch_cuda_half_fixture() -> Result<()> {
        let device = candle::Device::Cpu;
        let fixture = candle::safetensors::load_buffer(
            include_bytes!("../../../../../tests/fixtures/hunyuan3d/paint-vae-opmath.safetensors"),
            &device,
        )?;
        let vb = nn::VarBuilder::from_tensors(fixture.clone(), DType::F16, &device);
        let normalized = Norm::new(vb, 4, 8, VaeNumerics::Diffusers)?.forward(&fixture["input"])?;
        let activated = silu(&fixture["input"], VaeNumerics::Diffusers)?;
        for (name, actual, tolerance) in [
            ("normalized", normalized, 0.002),
            ("activated", activated, 0.0005),
        ] {
            let error = (actual.to_dtype(DType::F32)? - fixture[name].to_dtype(DType::F32)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(error <= tolerance, "{name}: {error}");
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA for the PyTorch reduction-kernel fixture"]
    fn cuda_normalization_matches_pytorch_reduction_and_affine() -> Result<()> {
        let device = candle::Device::new_cuda(0)?;
        let fixture = candle::safetensors::load_buffer(
            include_bytes!("../../../../../tests/fixtures/hunyuan3d/paint-vae-opmath.safetensors"),
            &device,
        )?;
        let vb = nn::VarBuilder::from_tensors(fixture.clone(), DType::F16, &device);
        let norm = Norm::new(vb, 4, 8, VaeNumerics::Diffusers)?;
        for suffix in ["", "_large", "_spatial1"] {
            let actual = norm.forward(&fixture[&format!("input{suffix}")])?;
            let error = (actual.to_dtype(DType::F32)?
                - fixture[&format!("normalized{suffix}")].to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
            eprintln!("CUDA GroupNorm{suffix}: {error}");
            assert_eq!(error, 0., "CUDA normalization{suffix}");
        }
        let linear = AttentionLinear(nn::Linear::new(
            fixture["linear_weight"].to_dtype(DType::F32)?,
            Some(fixture["bias"].to_dtype(DType::F32)?),
        ));
        let actual = linear.forward(&fixture["linear_input"])?;
        let error = (actual.to_dtype(DType::F32)?
            - fixture["linear_output"].to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
        assert!(error <= 0.0005, "CUDA addmm opmath: {error}");
        Ok(())
    }
}
