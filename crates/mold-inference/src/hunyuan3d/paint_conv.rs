// Copyright 2024 The HuggingFace Team. All rights reserved.
// Rust adaptation for mold: inference-only fixed paint recipe and bounded resampling.
// Apache-2.0; see THIRD_PARTY_NOTICES.md and LICENSE-APACHE-2.0.
//! Diffusers 0.30 convolutional stages for the Hunyuan3D 2.1 paint UNet.
//! Reference: diffusers 8a79d8ec models/resnet.py:267-377,
//! models/downsampling.py:69-149 and models/upsampling.py:75-185.
use super::paint_attention::{linear, projected};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use mold_candle::stable_diffusion::normalization::DiffusersGroupNorm;

pub(super) fn silu(x: &Tensor) -> Result<Tensor> {
    x.to_dtype(DType::F32)?.silu()?.to_dtype(x.dtype())
}

pub(super) fn conv(
    vb: VarBuilder,
    input: usize,
    output: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
) -> Result<candle_nn::Conv2d> {
    // PyTorch's cuDNN path rounds convolution output before adding bias
    // (ATen/native/Convolution.cpp:1532-1536); keep Candle's same boundary.
    candle_nn::conv2d(
        input,
        output,
        kernel,
        candle_nn::Conv2dConfig {
            padding,
            stride,
            ..Default::default()
        },
        vb,
    )
}

pub struct PaintResnet {
    norm1: DiffusersGroupNorm,
    norm2: DiffusersGroupNorm,
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    time: candle_nn::Linear,
    shortcut: Option<candle_nn::Conv2d>,
    input: usize,
    output: usize,
    time_width: usize,
    dtype: DType,
}

impl PaintResnet {
    pub fn new(
        input: usize,
        output: usize,
        time_width: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if time_width == 0 || !matches!(vb.dtype(), DType::F16 | DType::F32) {
            candle_core::bail!("invalid paint residual block configuration")
        }
        Ok(Self {
            norm1: DiffusersGroupNorm::new(vb.pp("norm1"), groups, input, 1e-5)?,
            norm2: DiffusersGroupNorm::new(vb.pp("norm2"), groups, output, 1e-5)?,
            conv1: conv(vb.pp("conv1"), input, output, 3, 1, 1)?,
            conv2: conv(vb.pp("conv2"), output, output, 3, 1, 1)?,
            time: linear(vb.pp("time_emb_proj"), time_width, output, true)?,
            shortcut: if input != output {
                Some(conv(vb.pp("conv_shortcut"), input, output, 1, 0, 1)?)
            } else {
                None
            },
            input,
            output,
            time_width,
            dtype: vb.dtype(),
        })
    }

    pub fn forward(&self, x: &Tensor, time: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = x.dims4()?;
        if batch == 0
            || channels != self.input
            || height == 0
            || width == 0
            || time.dims() != [batch, self.time_width]
            || x.dtype() != self.dtype
            || time.dtype() != self.dtype
        {
            candle_core::bail!("invalid paint residual block input or timestep embedding")
        }
        let h = self.conv1.forward(&silu(&self.norm1.forward(x)?)?)?;
        let time = projected(&self.time, &silu(time)?)?.reshape((batch, self.output, 1, 1))?;
        let h = h.broadcast_add(&time)?;
        let h = self.conv2.forward(&silu(&self.norm2.forward(&h)?)?)?;
        let residual = match &self.shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        // Dropout is disabled, output_scale_factor is 1 in the pinned recipe.
        residual + h
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintResampleKind {
    Down,
    Up,
}

pub struct PaintResample {
    conv: candle_nn::Conv2d,
    channels: usize,
    dtype: DType,
    kind: PaintResampleKind,
}

impl PaintResample {
    pub fn new(channels: usize, kind: PaintResampleKind, vb: VarBuilder) -> Result<Self> {
        if channels == 0 || !matches!(vb.dtype(), DType::F16 | DType::F32) {
            candle_core::bail!("invalid paint resampling configuration")
        }
        Ok(Self {
            conv: conv(
                vb.pp("conv"),
                channels,
                channels,
                3,
                1,
                if kind == PaintResampleKind::Down {
                    2
                } else {
                    1
                },
            )?,
            channels,
            dtype: vb.dtype(),
            kind,
        })
    }

    pub fn forward(&self, x: &Tensor, output_size: Option<(usize, usize)>) -> Result<Tensor> {
        let (batch, channels, height, width) = x.dims4()?;
        if batch == 0
            || channels != self.channels
            || height == 0
            || width == 0
            || x.dtype() != self.dtype
        {
            candle_core::bail!("invalid paint resampling input")
        }
        if self.kind == PaintResampleKind::Down {
            if output_size.is_some() {
                candle_core::bail!("paint downsampling does not accept an output size")
            }
            return self.conv.forward(x);
        }
        let (height, width) = match output_size {
            Some(size) => size,
            None => (
                height.checked_mul(2).ok_or_else(|| {
                    candle_core::Error::Msg("paint upsample height overflow".into())
                })?,
                width.checked_mul(2).ok_or_else(|| {
                    candle_core::Error::Msg("paint upsample width overflow".into())
                })?,
            ),
        };
        if height == 0
            || width == 0
            || batch
                .checked_mul(channels)
                .and_then(|n| n.checked_mul(height))
                .and_then(|n| n.checked_mul(width))
                .is_none_or(|n| n > u32::MAX as usize)
        {
            candle_core::bail!("paint upsample output exceeds tensor allocation bound")
        }
        self.conv.forward(&x.upsample_nearest2d(height, width)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn paint_resnet_matches_diffusers() -> Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-conv.safetensors"),
            &Device::Cpu,
        )?;
        compare(&fixture, &Device::Cpu, false, None)
    }

    fn compare(
        fixture: &std::collections::HashMap<String, Tensor>,
        device: &Device,
        pretrained: bool,
        output: Option<&std::path::Path>,
    ) -> Result<()> {
        let (channels, time_width, groups) = if pretrained {
            (320, 1280, 32)
        } else {
            (32, 128, 4)
        };
        let labels: &[&str] = if pretrained {
            &["resnet", "down", "up"]
        } else {
            &["resnet", "shortcut", "down", "up"]
        };
        for &label in labels {
            for (name, dtype) in [("f32", DType::F32), ("f16", DType::F16)] {
                let get = |key: &str| &fixture[&format!("{label}.{name}.{key}")];
                let vb = VarBuilder::from_tensors(fixture.clone(), dtype, device)
                    .pp(format!("{label}.weights"));
                let actual = if label == "resnet" || label == "shortcut" {
                    PaintResnet::new(
                        channels,
                        if label == "shortcut" { 64 } else { channels },
                        time_width,
                        groups,
                        vb,
                    )?
                    .forward(get("input"), get("time"))?
                } else {
                    let kind = if label == "down" {
                        PaintResampleKind::Down
                    } else {
                        PaintResampleKind::Up
                    };
                    let sampler = PaintResample::new(get("input").dim(1)?, kind, vb)?;
                    if kind == PaintResampleKind::Up {
                        check(
                            &sampler.forward(get("input"), Some((17, 13)))?,
                            get("explicit_expected"),
                            &format!("{label}.{name}.explicit"),
                            dtype,
                            output,
                        )?;
                        assert!(sampler
                            .forward(get("input"), Some((usize::MAX, 3)))
                            .is_err());
                    } else {
                        assert!(sampler.forward(get("input"), Some((17, 13))).is_err());
                    }
                    sampler.forward(get("input"), None)?
                };
                check(
                    &actual,
                    get("expected"),
                    &format!("{label}.{name}"),
                    dtype,
                    output,
                )?;
            }
        }
        Ok(())
    }

    fn check(
        actual: &Tensor,
        expected: &Tensor,
        label: &str,
        dtype: DType,
        output: Option<&std::path::Path>,
    ) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        if let Some(output) = output {
            candle_core::safetensors::save(
                &std::collections::HashMap::from([("actual", actual.clone())]),
                output.join(format!("{label}.safetensors")),
            )?;
        }
        let difference = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
        let max = difference.abs()?.max_all()?.to_scalar::<f32>()?;
        let rms = difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
        let (max_bound, rms_bound) = if dtype == DType::F32 {
            (1e-4, 1e-5)
        } else {
            (0.02, 0.002)
        };
        eprintln!("paint convolution {label}: max {max}, RMS {rms}");
        assert!(
            max < max_bound && rms < rms_bound,
            "{label}: max {max}, RMS {rms}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires installed-weight oracle and NVIDIA CUDA"]
    fn pretrained_paint_convolution_matches_diffusers() -> Result<()> {
        let oracle = std::env::var("MOLD_PAINT_CONV_ORACLE").expect("retained oracle directory");
        let output = std::path::PathBuf::from(
            std::env::var("MOLD_PAINT_CONV_RESULT").expect("new output directory"),
        );
        std::fs::create_dir(&output)?;
        let device = Device::new_cuda(0)?;
        let fixture = candle_core::safetensors::load(
            std::path::Path::new(&oracle).join("paint-conv.safetensors"),
            &device,
        )?;
        compare(&fixture, &device, true, Some(&output))
    }
}
