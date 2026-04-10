use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, IndexOp, Module, Tensor};
use candle_nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct LatentUpsamplerConfig {
    #[serde(default = "default_in_channels")]
    in_channels: usize,
    #[serde(default = "default_mid_channels")]
    mid_channels: usize,
    #[serde(default = "default_num_blocks")]
    num_blocks_per_stage: usize,
    #[serde(default = "default_dims")]
    dims: usize,
    #[serde(default = "default_spatial_upsample")]
    spatial_upsample: bool,
    #[serde(default)]
    temporal_upsample: bool,
    #[serde(default = "default_spatial_scale")]
    spatial_scale: f32,
    #[serde(default)]
    rational_resampler: bool,
}

impl Default for LatentUpsamplerConfig {
    fn default() -> Self {
        Self {
            in_channels: default_in_channels(),
            mid_channels: default_mid_channels(),
            num_blocks_per_stage: default_num_blocks(),
            dims: default_dims(),
            spatial_upsample: default_spatial_upsample(),
            temporal_upsample: false,
            spatial_scale: default_spatial_scale(),
            rational_resampler: false,
        }
    }
}

fn default_in_channels() -> usize {
    128
}

fn default_mid_channels() -> usize {
    512
}

fn default_num_blocks() -> usize {
    4
}

fn default_dims() -> usize {
    3
}

fn default_spatial_upsample() -> bool {
    true
}

fn default_spatial_scale() -> f32 {
    2.0
}

#[derive(Clone, Debug)]
struct NonCausalConv3d {
    kt: usize,
    stride_t: usize,
    dil_t: usize,
    conv2d_slices: Vec<Conv2d>,
    bias: Tensor,
}

impl NonCausalConv3d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel;
        let (st, sh, sw) = stride;
        let (dt, dh, dw) = dilation;
        if sh != sw {
            bail!("spatial stride must match for latent upsampler");
        }
        if dh != dw {
            bail!("spatial dilation must match for latent upsampler");
        }

        let weight = vb.get((out_channels, in_channels / groups, kt, kh, kw), "weight")?;
        let bias = vb.get(out_channels, "bias")?;
        let padding = kh / 2;
        let mut conv2d_slices = Vec::with_capacity(kt);
        for ti in 0..kt {
            let slice = weight.i((.., .., ti, .., ..))?.contiguous()?;
            let cfg = Conv2dConfig {
                padding,
                stride: sh,
                dilation: dh,
                groups,
                ..Default::default()
            };
            conv2d_slices.push(Conv2d::new(slice, None, cfg));
        }

        Ok(Self {
            kt,
            stride_t: st,
            dil_t: dt,
            conv2d_slices,
            bias,
        })
    }

    fn pad_time(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, _t, h, w) = x.dims5()?;
        if self.kt <= 1 {
            return Ok(x.clone());
        }

        let left = (self.kt - 1) / 2;
        let right = (self.kt - 1) / 2;

        // Match upstream Conv3d(..., padding=1) semantics for the LTX-2
        // spatial upsampler: temporal padding is zero-filled, not replicated.
        let pad_left = if left == 0 {
            None
        } else {
            Some(Tensor::zeros((b, c, left, h, w), x.dtype(), x.device())?)
        };
        let pad_right = if right == 0 {
            None
        } else {
            Some(Tensor::zeros((b, c, right, h, w), x.dtype(), x.device())?)
        };

        match (pad_left, pad_right) {
            (None, None) => Ok(x.clone()),
            (Some(pl), None) => Ok(Tensor::cat(&[&pl, x], 2)?),
            (None, Some(pr)) => Ok(Tensor::cat(&[x, &pr], 2)?),
            (Some(pl), Some(pr)) => Ok(Tensor::cat(&[&pl, x, &pr], 2)?),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pad_time(x)?;
        let (_, _, t_pad, _, _) = x.dims5()?;
        let needed = (self.kt - 1) * self.dil_t + 1;
        if t_pad < needed {
            bail!("time dimension too small for non-causal conv3d");
        }
        let t_out = (t_pad - needed) / self.stride_t + 1;
        let mut outputs = Vec::with_capacity(t_out);

        for to in 0..t_out {
            let base_t = to * self.stride_t;
            let mut acc: Option<Tensor> = None;
            for ki in 0..self.kt {
                let ti = base_t + ki * self.dil_t;
                let xt = x.i((.., .., ti, .., ..))?;
                let yt = xt.apply(&self.conv2d_slices[ki])?;
                acc = Some(match acc {
                    None => yt,
                    Some(prev) => prev.add(&yt)?,
                });
            }
            outputs.push(acc.expect("kt >= 1").unsqueeze(2)?);
        }

        let y = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?;
        let bias = self.bias.reshape((1, self.bias.dims1()?, 1, 1, 1))?;
        Ok(y.broadcast_add(&bias)?)
    }
}

#[derive(Clone, Debug)]
struct ResBlock3d {
    conv1: NonCausalConv3d,
    norm1: GroupNorm,
    conv2: NonCausalConv3d,
    norm2: GroupNorm,
}

impl ResBlock3d {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: NonCausalConv3d::new(
                channels,
                channels,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                1,
                vb.pp("conv1"),
            )?,
            norm1: group_norm(32, channels, 1e-5, vb.pp("norm1"))?,
            conv2: NonCausalConv3d::new(
                channels,
                channels,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                1,
                vb.pp("conv2"),
            )?,
            norm2: group_norm(32, channels, 1e-5, vb.pp("norm2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.conv1.forward(x)?;
        let x = x.apply(&self.norm1)?.silu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.apply(&self.norm2)?;
        Ok((x + residual)?.silu()?)
    }
}

pub struct LatentUpsampler {
    config: LatentUpsamplerConfig,
    initial_conv: NonCausalConv3d,
    initial_norm: GroupNorm,
    res_blocks: Vec<ResBlock3d>,
    spatial_upsampler: SpatialUpsampler2d,
    post_upsample_res_blocks: Vec<ResBlock3d>,
    final_conv: NonCausalConv3d,
}

enum SpatialUpsampler2d {
    PixelShuffle {
        conv: Conv2d,
        scale: usize,
    },
    Rational {
        conv: Conv2d,
        scale: usize,
        blur_down: Conv2d,
    },
}

impl SpatialUpsampler2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::PixelShuffle { conv, scale } => {
                let x = conv.forward(x)?;
                Ok(candle_nn::ops::pixel_shuffle(&x, *scale)?)
            }
            Self::Rational {
                conv,
                scale,
                blur_down,
            } => {
                let x = conv.forward(x)?;
                let x = candle_nn::ops::pixel_shuffle(&x, *scale)?;
                Ok(blur_down.forward(&x)?)
            }
        }
    }
}

impl LatentUpsampler {
    pub fn load(path: &Path, dtype: DType, device: &candle_core::Device) -> Result<Self> {
        let config = Self::load_config(path)?;
        if config.dims != 3 || !config.spatial_upsample || config.temporal_upsample {
            bail!(
                "unsupported latent upsampler config: dims={}, spatial_upsample={}, temporal_upsample={}",
                config.dims,
                config.spatial_upsample,
                config.temporal_upsample
            );
        }

        // SAFETY: the safetensors file is immutable model data.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[path.to_path_buf()], dtype, device)? };
        Self::new(config, vb)
    }

    fn load_config(path: &Path) -> Result<LatentUpsamplerConfig> {
        let data = std::fs::read(path).with_context(|| {
            format!(
                "failed to read latent upsampler config from {}",
                path.display()
            )
        })?;
        let (_header_len, metadata) = safetensors::tensor::SafeTensors::read_metadata(&data)
            .with_context(|| {
                format!(
                    "failed to parse safetensors metadata from {}",
                    path.display()
                )
            })?;
        let tensors = safetensors::SafeTensors::deserialize(&data).with_context(|| {
            format!(
                "failed to parse latent upsampler tensors from {}",
                path.display()
            )
        })?;
        let metadata = metadata.metadata().as_ref();
        let mut config =
            if let Some(config_json) = metadata.and_then(|metadata| metadata.get("config")) {
                serde_json::from_str(config_json)?
            } else {
                LatentUpsamplerConfig::default()
            };

        if let Ok(tensor) = tensors.tensor("initial_conv.bias") {
            config.mid_channels = tensor.shape()[0];
        }
        if let Ok(tensor) = tensors.tensor("final_conv.bias") {
            config.in_channels = tensor.shape()[0];
        }
        if tensors.tensor("upsampler.conv.weight").is_ok() {
            config.rational_resampler = true;
        }

        let upsampler_weight = if let Ok(tensor) = tensors.tensor("upsampler.conv.weight") {
            Some(tensor)
        } else {
            tensors.tensor("upsampler.0.weight").ok()
        };
        if let Some(weight) = upsampler_weight {
            let ratio = weight.shape()[0] / config.mid_channels.max(1);
            config.spatial_scale = match ratio {
                4 => 2.0,
                9 => 1.5,
                16 => 4.0,
                _ => config.spatial_scale,
            };
        }

        let mut block_count = 0usize;
        loop {
            let key = format!("res_blocks.{block_count}.conv1.weight");
            if tensors.tensor(&key).is_ok() {
                block_count += 1;
            } else {
                break;
            }
        }
        if block_count > 0 {
            config.num_blocks_per_stage = block_count;
        }

        Ok(config)
    }

    fn new(config: LatentUpsamplerConfig, vb: VarBuilder) -> Result<Self> {
        let initial_conv = NonCausalConv3d::new(
            config.in_channels,
            config.mid_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            vb.pp("initial_conv"),
        )?;
        let initial_norm = group_norm(32, config.mid_channels, 1e-5, vb.pp("initial_norm"))?;

        let mut res_blocks = Vec::with_capacity(config.num_blocks_per_stage);
        for i in 0..config.num_blocks_per_stage {
            res_blocks.push(ResBlock3d::new(
                config.mid_channels,
                vb.pp("res_blocks").pp(i.to_string()),
            )?);
        }

        let spatial_upsampler = if config.rational_resampler {
            let scale = match config.spatial_scale {
                scale if (scale - 1.5).abs() < f32::EPSILON => 3,
                scale if (scale - 2.0).abs() < f32::EPSILON => 2,
                scale if (scale - 4.0).abs() < f32::EPSILON => 4,
                other => bail!("unsupported rational latent upsampler scale: {other}"),
            };
            let den = match scale {
                3 => 2,
                2 | 4 => 1,
                _ => unreachable!("validated scale above"),
            };
            let conv = conv2d(
                config.mid_channels,
                scale * scale * config.mid_channels,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("upsampler").pp("conv"),
            )?;
            let blur_kernel = vb
                .pp("upsampler")
                .pp("blur_down")
                .get((1, 1, 5, 5), "kernel")?
                .repeat((config.mid_channels, 1, 1, 1))?;
            let blur_down = Conv2d::new(
                blur_kernel,
                None,
                Conv2dConfig {
                    padding: 2,
                    stride: den,
                    groups: config.mid_channels,
                    ..Default::default()
                },
            );
            SpatialUpsampler2d::Rational {
                conv,
                scale,
                blur_down,
            }
        } else {
            let conv = conv2d(
                config.mid_channels,
                4 * config.mid_channels,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("upsampler.0"),
            )?;
            SpatialUpsampler2d::PixelShuffle { conv, scale: 2 }
        };

        let mut post_upsample_res_blocks = Vec::with_capacity(config.num_blocks_per_stage);
        for i in 0..config.num_blocks_per_stage {
            post_upsample_res_blocks.push(ResBlock3d::new(
                config.mid_channels,
                vb.pp("post_upsample_res_blocks").pp(i.to_string()),
            )?);
        }

        let final_conv = NonCausalConv3d::new(
            config.mid_channels,
            config.in_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            vb.pp("final_conv"),
        )?;

        Ok(Self {
            config,
            initial_conv,
            initial_norm,
            res_blocks,
            spatial_upsampler,
            post_upsample_res_blocks,
            final_conv,
        })
    }

    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let (b, _c, f, h, w) = latent.dims5()?;
        let mut x = self.initial_conv.forward(latent)?;
        x = x.apply(&self.initial_norm)?.silu()?;

        for block in &self.res_blocks {
            x = block.forward(&x)?;
        }

        // Spatial-only upsampling: flatten frames into the batch, apply the
        // 2D conv + pixel shuffle branch, then restore [B, C, F, H, W].
        let x2 = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * f, self.config.mid_channels, h, w))?;
        let x2 = self.spatial_upsampler.forward(&x2)?;
        let (_bf, c2, h2, w2) = x2.dims4()?;
        let mut x = x2.reshape((b, f, c2, h2, w2))?.permute((0, 2, 1, 3, 4))?;

        for block in &self.post_upsample_res_blocks {
            x = block.forward(&x)?;
        }

        self.final_conv.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    use super::NonCausalConv3d;

    #[test]
    fn upsampler_conv3d_matches_zero_padded_temporal_convolution() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 4.0], (1, 1, 3, 1, 1), &device).unwrap(),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_vec(vec![0.0f32], 1, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, candle_core::DType::F32, &device);
        let conv = NonCausalConv3d::new(1, 1, (3, 1, 1), (1, 1, 1), (1, 1, 1), 1, vb).unwrap();

        let input = Tensor::from_vec(vec![10.0f32, 20.0], (1, 1, 2, 1, 1), &device).unwrap();
        let output = conv.forward(&input).unwrap();

        let actual = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual, vec![100.0, 50.0]);
    }
}
