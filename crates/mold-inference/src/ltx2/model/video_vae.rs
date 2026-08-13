#![allow(clippy::large_enum_variant)]

use candle_core::{bail, DType, IndexOp, Result, Tensor};
use candle_nn::{group_norm, ops, Conv2d, Conv2dConfig, GroupNorm, VarBuilder};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ltx2::tiling::{split_by_size, SpatialDecodeTiling};

fn cat_dim(xs: &[Tensor], dim: usize) -> Result<Tensor> {
    let refs = xs.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, dim)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    ops::silu(x)
}

#[derive(Debug, Clone)]
pub struct PerChannelRmsNorm {
    eps: f64,
    channel_dim: usize,
}

impl PerChannelRmsNorm {
    pub fn new(channel_dim: usize, eps: f64) -> Self {
        Self { eps, channel_dim }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean_sq =
            crate::ltx2::metal_reduce::mean_keepdim_stable(&x_f32.sqr()?, self.channel_dim)?;
        let rms = mean_sq.affine(1.0, self.eps)?.sqrt()?;
        x_f32.broadcast_div(&rms)?.to_dtype(dtype)
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Conv3dLikeConfig {
    stride_t: usize,
    stride_h: usize,
    dil_t: usize,
    groups: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpatialPaddingMode {
    Zeros,
    Reflect,
}

#[derive(Debug, Clone)]
pub struct Ltx2VideoCausalConv3d {
    kt: usize,
    #[allow(dead_code)]
    is_causal_default: bool,
    cfg: Conv3dLikeConfig,
    spatial_pad_h: usize,
    spatial_pad_w: usize,
    spatial_padding_mode: SpatialPaddingMode,
    conv2d_slices: Vec<Conv2d>,
    bias: Option<Tensor>,
}

impl Ltx2VideoCausalConv3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        is_causal_default: bool,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel;
        let (st, sh, sw) = stride;
        let (dt, dh, dw) = dilation;
        if sh != sw {
            bail!("LTX-2 VAE expects symmetric spatial stride, got ({sh}, {sw})");
        }
        if dh != dw {
            bail!("LTX-2 VAE expects symmetric spatial dilation, got ({dh}, {dw})");
        }

        let conv_vb = vb.pp("conv");
        let weight = conv_vb.get((out_channels, in_channels / groups, kt, kh, kw), "weight")?;
        let bias = conv_vb.get(out_channels, "bias").ok();

        let mut conv2d_slices = Vec::with_capacity(kt);
        for ti in 0..kt {
            let weight2d = weight.i((.., .., ti, .., ..))?.contiguous()?;
            let cfg = Conv2dConfig {
                stride: sh,
                dilation: dh,
                groups,
                ..Default::default()
            };
            conv2d_slices.push(Conv2d::new(weight2d, None, cfg));
        }

        Ok(Self {
            kt,
            is_causal_default,
            cfg: Conv3dLikeConfig {
                stride_t: st,
                stride_h: sh,
                dil_t: dt,
                groups,
            },
            spatial_pad_h: kh / 2,
            spatial_pad_w: kw / 2,
            spatial_padding_mode,
            conv2d_slices,
            bias,
        })
    }

    fn pad_time_replicate(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let (_, _, t, _, _) = x.dims5()?;
        if self.kt <= 1 {
            return Ok(x.clone());
        }
        if causal {
            let first = x.i((.., .., 0, .., ..))?.unsqueeze(2)?;
            let left = first.repeat((1, 1, self.kt - 1, 1, 1))?;
            cat_dim(&[left, x.clone()], 2)
        } else {
            let pad = (self.kt - 1) / 2;
            let first = x.i((.., .., 0, .., ..))?.unsqueeze(2)?;
            let last = x.i((.., .., t - 1, .., ..))?.unsqueeze(2)?;
            let left = first.repeat((1, 1, pad, 1, 1))?;
            let right = last.repeat((1, 1, pad, 1, 1))?;
            cat_dim(&[left, x.clone(), right], 2)
        }
    }

    fn reflect_pad_4d(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        if self.spatial_pad_h != 0 && h <= self.spatial_pad_h {
            bail!(
                "reflect padding requires height > pad, got height={} pad={}",
                h,
                self.spatial_pad_h
            );
        }
        if self.spatial_pad_w != 0 && w <= self.spatial_pad_w {
            bail!(
                "reflect padding requires width > pad, got width={} pad={}",
                w,
                self.spatial_pad_w
            );
        }

        let mut padded = x.clone();
        if self.spatial_pad_w != 0 {
            let left = padded
                .i((.., .., .., 1..(self.spatial_pad_w + 1)))?
                .contiguous()?
                .flip(&[3])?;
            let right = padded
                .i((.., .., .., (w - self.spatial_pad_w - 1)..(w - 1)))?
                .contiguous()?
                .flip(&[3])?;
            padded = Tensor::cat(&[left, padded, right], 3)?;
        }
        if self.spatial_pad_h != 0 {
            let top = padded
                .i((.., .., 1..(self.spatial_pad_h + 1), ..))?
                .contiguous()?
                .flip(&[2])?;
            let bottom = padded
                .i((.., .., (h - self.spatial_pad_h - 1)..(h - 1), ..))?
                .contiguous()?
                .flip(&[2])?;
            padded = Tensor::cat(&[top, padded, bottom], 2)?;
        }
        Ok(padded)
    }

    fn pad_spatial(&self, x: &Tensor) -> Result<Tensor> {
        if self.spatial_pad_h == 0 && self.spatial_pad_w == 0 {
            return Ok(x.clone());
        }
        match self.spatial_padding_mode {
            SpatialPaddingMode::Zeros => x
                .pad_with_zeros(3, self.spatial_pad_w, self.spatial_pad_w)?
                .pad_with_zeros(2, self.spatial_pad_h, self.spatial_pad_h),
            SpatialPaddingMode::Reflect => self.reflect_pad_4d(x),
        }
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let x = self.pad_time_replicate(x, causal)?;
        let (_, _, t_pad, _, _) = x.dims5()?;
        let needed = (self.kt - 1) * self.cfg.dil_t + 1;
        if t_pad < needed {
            bail!("time dimension too small after padding: {t_pad} < {needed}");
        }
        let t_out = (t_pad - needed) / self.cfg.stride_t + 1;

        let mut ys = Vec::with_capacity(t_out);
        for t_out_idx in 0..t_out {
            let base_t = t_out_idx * self.cfg.stride_t;
            let mut acc: Option<Tensor> = None;
            for ki in 0..self.kt {
                let ti = base_t + ki * self.cfg.dil_t;
                let xt = x.i((.., .., ti, .., ..))?;
                let yt = self.pad_spatial(&xt)?.apply(&self.conv2d_slices[ki])?;
                acc = Some(match acc {
                    None => yt,
                    Some(prev) => prev.add(&yt)?,
                });
            }
            ys.push(acc.expect("temporal kernel must accumulate").unsqueeze(2)?);
        }

        let y = cat_dim(&ys, 2)?;
        match &self.bias {
            Some(bias) => {
                let bias = bias.reshape((1, bias.dims1()?, 1, 1, 1))?;
                y.broadcast_add(&bias)
            }
            None => Ok(y),
        }
    }

    #[allow(dead_code)]
    pub fn forward_default(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x, self.is_causal_default)
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2VideoDownsampler3d {
    stride: (usize, usize, usize),
    group_size: usize,
    conv: Ltx2VideoCausalConv3d,
}

impl Ltx2VideoDownsampler3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize, usize),
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (st, sh, sw) = stride;
        let stride_product = st * sh * sw;
        let group_size = (in_channels * stride_product) / out_channels;
        let conv = Ltx2VideoCausalConv3d::new(
            in_channels,
            out_channels / stride_product,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            spatial_padding_mode,
            vb.pp("conv"),
        )?;
        Ok(Self {
            stride,
            group_size,
            conv,
        })
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let (st, sh, sw) = self.stride;
        let (b, c, _t, _h, _w) = x.dims5()?;
        let padded = if st > 1 {
            let first = x.i((.., .., 0, .., ..))?.unsqueeze(2)?;
            let prefix = first.repeat((1, 1, st - 1, 1, 1))?;
            Tensor::cat(&[&prefix, x], 2)?
        } else {
            x.clone()
        };
        let (_, _, t_pad, h_pad, w_pad) = padded.dims5()?;
        let t_new = t_pad / st;
        let h_new = h_pad / sh;
        let w_new = w_pad / sw;

        let residual = padded
            .reshape(&[b, c, t_new, st, h_new, sh, w_new, sw])?
            .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?
            .reshape((b, c * st * sh * sw, t_new, h_new, w_new))?
            .reshape(&[
                b,
                c * st * sh * sw / self.group_size,
                self.group_size,
                t_new,
                h_new,
                w_new,
            ])?
            .mean(2)?;

        let hidden = self
            .conv
            .forward(&padded, causal)?
            .reshape(&[
                b,
                residual.dims5()?.1 / (st * sh * sw),
                t_new,
                st,
                h_new,
                sh,
                w_new,
                sw,
            ])?
            .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?
            .reshape((b, residual.dims5()?.1, t_new, h_new, w_new))?;

        hidden.add(&residual)
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2VideoUpsampler3d {
    stride_t: usize,
    stride_h: usize,
    stride_w: usize,
    residual: bool,
    out_channels_reduction_factor: usize,
    conv: Ltx2VideoCausalConv3d,
}

impl Ltx2VideoUpsampler3d {
    pub fn new(
        in_channels: usize,
        stride: (usize, usize, usize),
        residual: bool,
        out_channels_reduction_factor: usize,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (st, sh, sw) = stride;
        let out_channels = st * sh * sw * in_channels / out_channels_reduction_factor;
        let conv = Ltx2VideoCausalConv3d::new(
            in_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            spatial_padding_mode,
            vb.pp("conv"),
        )?;
        Ok(Self {
            stride_t: st,
            stride_h: sh,
            stride_w: sw,
            residual,
            out_channels_reduction_factor,
            conv,
        })
    }

    fn rearrange(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let st = self.stride_t;
        let sh = self.stride_h;
        let sw = self.stride_w;
        let c_out = c / (st * sh * sw);
        x.reshape(&[b, c_out, st, sh, sw, t, h, w])?
            .permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?
            .contiguous()?
            .reshape(&[b, c_out, t, st, h, sh, w * sw])?
            .reshape(&[b, c_out, t, st, h * sh, w * sw])?
            .reshape(&[b, c_out, t * st, h * sh, w * sw])
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let residual = if self.residual {
            let x_in = self.rearrange(x)?;
            let repeats = (self.stride_t * self.stride_h * self.stride_w)
                / self.out_channels_reduction_factor;
            let x_in = if repeats > 1 {
                x_in.repeat((1, repeats, 1, 1, 1))?
            } else {
                x_in
            };
            let x_in = if self.stride_t > 1 {
                x_in.i((.., .., 1.., .., ..))?
            } else {
                x_in
            };
            Some(x_in)
        } else {
            None
        };

        let hidden = self.rearrange(&self.conv.forward(x, causal)?)?;
        let hidden = if self.stride_t > 1 {
            hidden.i((.., .., 1.., .., ..))?
        } else {
            hidden
        };

        match residual {
            Some(residual) => hidden.add(&residual),
            None => Ok(hidden),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2VideoResnetBlock3d {
    norm1: PerChannelRmsNorm,
    conv1: Ltx2VideoCausalConv3d,
    norm2: PerChannelRmsNorm,
    conv2: Ltx2VideoCausalConv3d,
    norm3: Option<GroupNorm>,
    conv_shortcut: Option<Ltx2VideoCausalConv3d>,
    per_channel_scale1: Option<Tensor>,
    per_channel_scale2: Option<Tensor>,
}

impl Ltx2VideoResnetBlock3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        eps: f64,
        inject_noise: bool,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = PerChannelRmsNorm::new(1, eps);
        let conv1 = Ltx2VideoCausalConv3d::new(
            in_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            spatial_padding_mode,
            vb.pp("conv1"),
        )?;
        let norm2 = PerChannelRmsNorm::new(1, eps);
        let conv2 = Ltx2VideoCausalConv3d::new(
            out_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            spatial_padding_mode,
            vb.pp("conv2"),
        )?;

        let (norm3, conv_shortcut) = if in_channels != out_channels {
            let norm3 = group_norm(1, in_channels, eps, vb.pp("norm3")).ok();
            let conv_shortcut = Ltx2VideoCausalConv3d::new(
                in_channels,
                out_channels,
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                1,
                true,
                spatial_padding_mode,
                vb.pp("conv_shortcut"),
            )
            .ok();
            (norm3, conv_shortcut)
        } else {
            (None, None)
        };

        let per_channel_scale1 = if inject_noise {
            vb.get((in_channels, 1, 1), "per_channel_scale1")
                .or_else(|_| {
                    vb.pp("per_channel_scale1")
                        .get((in_channels, 1, 1), "weight")
                })
                .ok()
        } else {
            None
        };
        let per_channel_scale2 = if inject_noise {
            vb.get((out_channels, 1, 1), "per_channel_scale2")
                .or_else(|_| {
                    vb.pp("per_channel_scale2")
                        .get((out_channels, 1, 1), "weight")
                })
                .ok()
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            norm3,
            conv_shortcut,
            per_channel_scale1,
            per_channel_scale2,
        })
    }

    fn maybe_inject_noise(&self, x: Tensor, scale: Option<&Tensor>) -> Result<Tensor> {
        let Some(scale) = scale else {
            return Ok(x);
        };
        let (_, _, _, h, w) = x.dims5()?;
        let noise = Tensor::randn(0f32, 1f32, (h, w), x.device())?
            .to_dtype(x.dtype())?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let scale = scale.unsqueeze(0)?.unsqueeze(2)?;
        x.add(&noise.broadcast_mul(&scale)?)
    }

    pub fn forward(&self, inputs: &Tensor, causal: bool) -> Result<Tensor> {
        let mut hidden = self.norm1.forward(inputs)?;
        hidden = silu(&hidden)?;
        hidden = self.conv1.forward(&hidden, causal)?;
        hidden = self.maybe_inject_noise(hidden, self.per_channel_scale1.as_ref())?;

        hidden = self.norm2.forward(&hidden)?;
        hidden = silu(&hidden)?;
        hidden = self.conv2.forward(&hidden, causal)?;
        hidden = self.maybe_inject_noise(hidden, self.per_channel_scale2.as_ref())?;

        let mut residual = inputs.clone();
        if let Some(norm3) = &self.norm3 {
            residual = residual.apply(norm3)?;
        }
        if let Some(conv_shortcut) = &self.conv_shortcut {
            residual = conv_shortcut.forward(&residual, causal)?;
        }

        hidden.add(&residual)
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2VideoResBlockStack {
    res_blocks: Vec<Ltx2VideoResnetBlock3d>,
}

impl Ltx2VideoResBlockStack {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        inject_noise: bool,
        eps: f64,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut res_blocks = Vec::with_capacity(num_layers);
        let mut current_in = in_channels;
        for index in 0..num_layers {
            let current_out = out_channels;
            res_blocks.push(Ltx2VideoResnetBlock3d::new(
                current_in,
                current_out,
                eps,
                inject_noise,
                spatial_padding_mode,
                vb.pp(format!("res_blocks.{index}")),
            )?);
            current_in = current_out;
        }
        Ok(Self { res_blocks })
    }

    pub fn forward(&self, hidden_states: &Tensor, causal: bool) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for block in &self.res_blocks {
            hidden_states = block.forward(&hidden_states, causal)?;
        }
        Ok(hidden_states)
    }
}

#[derive(Debug, Clone)]
enum EncoderBlock {
    ResStack(Ltx2VideoResBlockStack),
    ProjectionRes(Ltx2VideoResnetBlock3d),
    Conv(Ltx2VideoCausalConv3d),
    Downsample(Ltx2VideoDownsampler3d),
}

impl EncoderBlock {
    fn forward(&self, hidden_states: &Tensor, causal: bool) -> Result<Tensor> {
        match self {
            Self::ResStack(block) => block.forward(hidden_states, causal),
            Self::ProjectionRes(block) => block.forward(hidden_states, causal),
            Self::Conv(block) => block.forward(hidden_states, causal),
            Self::Downsample(block) => block.forward(hidden_states, causal),
        }
    }
}

#[derive(Debug, Clone)]
enum DecoderBlock {
    ResStack(Ltx2VideoResBlockStack),
    ProjectionRes(Ltx2VideoResnetBlock3d),
    Upsample(Ltx2VideoUpsampler3d),
}

impl DecoderBlock {
    fn forward(&self, hidden_states: &Tensor, causal: bool) -> Result<Tensor> {
        match self {
            Self::ResStack(block) => block.forward(hidden_states, causal),
            Self::ProjectionRes(block) => block.forward(hidden_states, causal),
            Self::Upsample(block) => block.forward(hidden_states, causal),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VaeBlockConfig {
    pub name: String,
    pub num_layers: usize,
    pub multiplier: usize,
    pub residual: bool,
    pub inject_noise: bool,
}

impl VaeBlockConfig {
    pub fn res_x(num_layers: usize) -> Self {
        Self {
            name: "res_x".to_string(),
            num_layers,
            multiplier: 1,
            residual: false,
            inject_noise: false,
        }
    }

    pub fn res_x_with_noise(num_layers: usize, inject_noise: bool) -> Self {
        Self {
            inject_noise,
            ..Self::res_x(num_layers)
        }
    }

    pub fn compress(name: &str, multiplier: usize, residual: bool) -> Self {
        Self {
            name: name.to_string(),
            num_layers: 0,
            multiplier,
            residual,
            inject_noise: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum LatentLogVar {
    PerChannel,
    Uniform,
    Constant,
    None,
}

#[derive(Clone, Debug)]
pub struct AutoencoderKLLtx2VideoConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub encoder_blocks: Vec<VaeBlockConfig>,
    pub decoder_blocks: Vec<VaeBlockConfig>,
    pub patch_size: usize,
    pub resnet_eps: f64,
    pub scaling_factor: f64,
    pub latent_log_var: LatentLogVar,
    pub encoder_base_channels: usize,
    pub decoder_base_channels: usize,
    encoder_spatial_padding_mode: SpatialPaddingMode,
    decoder_spatial_padding_mode: SpatialPaddingMode,
    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,
    pub timestep_conditioning: bool,
    pub decoder_causal: bool,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl Default for AutoencoderKLLtx2VideoConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 128,
            encoder_blocks: vec![
                VaeBlockConfig::res_x(4),
                VaeBlockConfig::compress("compress_space_res", 2, false),
                VaeBlockConfig::res_x(6),
                VaeBlockConfig::compress("compress_time_res", 2, false),
                VaeBlockConfig::res_x(6),
                VaeBlockConfig::compress("compress_all_res", 2, false),
                VaeBlockConfig::res_x(2),
                VaeBlockConfig::compress("compress_all_res", 2, false),
                VaeBlockConfig::res_x(2),
            ],
            decoder_blocks: vec![
                VaeBlockConfig::res_x_with_noise(5, false),
                VaeBlockConfig::compress("compress_all", 2, true),
                VaeBlockConfig::res_x_with_noise(5, false),
                VaeBlockConfig::compress("compress_all", 2, true),
                VaeBlockConfig::res_x_with_noise(5, false),
                VaeBlockConfig::compress("compress_all", 2, true),
                VaeBlockConfig::res_x_with_noise(5, false),
            ],
            patch_size: 4,
            resnet_eps: 1e-6,
            scaling_factor: 1.0,
            latent_log_var: LatentLogVar::Uniform,
            encoder_base_channels: 128,
            decoder_base_channels: 128,
            encoder_spatial_padding_mode: SpatialPaddingMode::Zeros,
            decoder_spatial_padding_mode: SpatialPaddingMode::Reflect,
            spatial_compression_ratio: 32,
            temporal_compression_ratio: 8,
            timestep_conditioning: false,
            decoder_causal: false,
            latents_mean: vec![0.0; 128],
            latents_std: vec![1.0; 128],
        }
    }
}

impl AutoencoderKLLtx2VideoConfig {
    pub(crate) fn ltx2_22b() -> Self {
        Self {
            encoder_blocks: vec![
                VaeBlockConfig::res_x(4),
                VaeBlockConfig::compress("compress_space_res", 2, true),
                VaeBlockConfig::res_x(6),
                VaeBlockConfig::compress("compress_time_res", 2, true),
                VaeBlockConfig::res_x(4),
                VaeBlockConfig::compress("compress_all_res", 2, true),
                VaeBlockConfig::res_x(2),
                VaeBlockConfig::compress("compress_all_res", 1, true),
                VaeBlockConfig::res_x(2),
            ],
            decoder_blocks: vec![
                VaeBlockConfig::res_x_with_noise(4, false),
                VaeBlockConfig::compress("compress_space", 2, false),
                VaeBlockConfig::res_x_with_noise(6, false),
                VaeBlockConfig::compress("compress_time", 2, false),
                VaeBlockConfig::res_x_with_noise(4, false),
                VaeBlockConfig::compress("compress_all", 1, false),
                VaeBlockConfig::res_x_with_noise(2, false),
                VaeBlockConfig::compress("compress_all", 2, false),
                VaeBlockConfig::res_x_with_noise(2, false),
            ],
            decoder_spatial_padding_mode: SpatialPaddingMode::Zeros,
            ..Self::default()
        }
    }
}

fn encoder_stride(name: &str) -> Option<(usize, usize, usize)> {
    match name {
        "compress_time" | "compress_time_res" => Some((2, 1, 1)),
        "compress_space" | "compress_space_res" => Some((1, 2, 2)),
        "compress_all" | "compress_all_x_y" | "compress_all_res" => Some((2, 2, 2)),
        _ => None,
    }
}

fn build_encoder_block(
    cfg: &VaeBlockConfig,
    in_channels: usize,
    eps: f64,
    spatial_padding_mode: SpatialPaddingMode,
    vb: VarBuilder,
) -> Result<(EncoderBlock, usize)> {
    match cfg.name.as_str() {
        "res_x" => Ok((
            EncoderBlock::ResStack(Ltx2VideoResBlockStack::new(
                in_channels,
                in_channels,
                cfg.num_layers,
                cfg.inject_noise,
                eps,
                spatial_padding_mode,
                vb,
            )?),
            in_channels,
        )),
        "res_x_y" => {
            let out_channels = in_channels * cfg.multiplier.max(1);
            Ok((
                EncoderBlock::ProjectionRes(Ltx2VideoResnetBlock3d::new(
                    in_channels,
                    out_channels,
                    eps,
                    cfg.inject_noise,
                    spatial_padding_mode,
                    vb,
                )?),
                out_channels,
            ))
        }
        "compress_time" | "compress_space" | "compress_all" => {
            let stride = encoder_stride(&cfg.name).expect("checked above");
            Ok((
                EncoderBlock::Conv(Ltx2VideoCausalConv3d::new(
                    in_channels,
                    in_channels,
                    (3, 3, 3),
                    stride,
                    (1, 1, 1),
                    1,
                    true,
                    spatial_padding_mode,
                    vb,
                )?),
                in_channels,
            ))
        }
        "compress_all_x_y" => {
            let out_channels = in_channels * cfg.multiplier.max(1);
            Ok((
                EncoderBlock::Conv(Ltx2VideoCausalConv3d::new(
                    in_channels,
                    out_channels,
                    (3, 3, 3),
                    (2, 2, 2),
                    (1, 1, 1),
                    1,
                    true,
                    spatial_padding_mode,
                    vb,
                )?),
                out_channels,
            ))
        }
        "compress_time_res" | "compress_space_res" | "compress_all_res" => {
            let out_channels = in_channels * cfg.multiplier.max(1);
            let stride = encoder_stride(&cfg.name).expect("checked above");
            Ok((
                EncoderBlock::Downsample(Ltx2VideoDownsampler3d::new(
                    in_channels,
                    out_channels,
                    stride,
                    spatial_padding_mode,
                    vb,
                )?),
                out_channels,
            ))
        }
        other => bail!("unsupported LTX-2 VAE encoder block: {other}"),
    }
}

fn build_decoder_block(
    cfg: &VaeBlockConfig,
    in_channels: usize,
    eps: f64,
    spatial_padding_mode: SpatialPaddingMode,
    vb: VarBuilder,
) -> Result<(DecoderBlock, usize)> {
    match cfg.name.as_str() {
        "res_x" => Ok((
            DecoderBlock::ResStack(Ltx2VideoResBlockStack::new(
                in_channels,
                in_channels,
                cfg.num_layers,
                cfg.inject_noise,
                eps,
                spatial_padding_mode,
                vb,
            )?),
            in_channels,
        )),
        "res_x_y" => {
            let out_channels = in_channels / cfg.multiplier.max(1);
            Ok((
                DecoderBlock::ProjectionRes(Ltx2VideoResnetBlock3d::new(
                    in_channels,
                    out_channels,
                    eps,
                    cfg.inject_noise,
                    spatial_padding_mode,
                    vb,
                )?),
                out_channels,
            ))
        }
        "compress_time" | "compress_space" | "compress_all" => {
            let stride = encoder_stride(&cfg.name).expect("checked above");
            let out_channels = in_channels / cfg.multiplier.max(1);
            Ok((
                DecoderBlock::Upsample(Ltx2VideoUpsampler3d::new(
                    in_channels,
                    stride,
                    cfg.residual,
                    cfg.multiplier.max(1),
                    spatial_padding_mode,
                    vb,
                )?),
                out_channels,
            ))
        }
        other => bail!("unsupported LTX-2 VAE decoder block: {other}"),
    }
}

fn patchify_video(sample: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(sample.clone());
    }
    let (b, c, f, h, w) = sample.dims5()?;
    if f % patch_size_t != 0 || h % patch_size_hw != 0 || w % patch_size_hw != 0 {
        bail!("input not divisible by patch sizes");
    }
    let f_out = f / patch_size_t;
    let h_out = h / patch_size_hw;
    let w_out = w / patch_size_hw;
    sample
        .reshape(&[
            b,
            c,
            f_out,
            patch_size_t,
            h_out,
            patch_size_hw,
            w_out,
            patch_size_hw,
        ])?
        .permute(vec![0, 1, 3, 7, 5, 2, 4, 6])?
        .contiguous()?
        .reshape((
            b,
            c * patch_size_t * patch_size_hw * patch_size_hw,
            f_out,
            h_out,
            w_out,
        ))
}

fn unpatchify_video(sample: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(sample.clone());
    }
    let (b, c, f, h, w) = sample.dims5()?;
    let out_channels = c / (patch_size_t * patch_size_hw * patch_size_hw);
    sample
        .reshape(&[
            b,
            out_channels,
            patch_size_t,
            patch_size_hw,
            patch_size_hw,
            f,
            h,
            w,
        ])?
        .permute(vec![0, 1, 5, 2, 6, 4, 7, 3])?
        .contiguous()?
        .reshape(&[
            b,
            out_channels,
            f,
            patch_size_t,
            h,
            patch_size_hw,
            w * patch_size_hw,
        ])?
        .reshape(&[
            b,
            out_channels,
            f,
            patch_size_t,
            h * patch_size_hw,
            w * patch_size_hw,
        ])?
        .reshape((
            b,
            out_channels,
            f * patch_size_t,
            h * patch_size_hw,
            w * patch_size_hw,
        ))
}

#[derive(Clone, Debug)]
pub struct DecoderOutput {
    #[allow(dead_code)]
    pub sample: Tensor,
}

const DEFAULT_TEMPORAL_DECODE_CHUNK_LATENT_FRAMES: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
struct TemporalDecodeChunk {
    input_start: usize,
    input_end: usize,
    output_start: usize,
    output_end: usize,
}

fn temporal_output_frames_for_latents(latent_frames: usize, temporal_scale: usize) -> usize {
    if latent_frames == 0 {
        0
    } else {
        (latent_frames - 1)
            .saturating_mul(temporal_scale.max(1))
            .saturating_add(1)
    }
}

/// Reassemble decoded strips along `dim`, crossfading their overlaps.
///
/// Each entry is `(strip, left_overlap, right_overlap)` in pixels; a strip's
/// right overlap must equal the next strip's left overlap, which is what
/// `split_by_size` guarantees. Ramp weights are `i / (overlap + 1)`, the same
/// values `trapezoidal_mask` uses, so the pair sums to one everywhere and the
/// blend needs no normalization pass.
fn crossfade_strips(strips: &[(Tensor, usize, usize)], dim: usize) -> Result<Tensor> {
    let mut segments: Vec<Tensor> = Vec::with_capacity(strips.len() * 2);
    let mut pending_tail: Option<Tensor> = None;
    for (index, (strip, left, right)) in strips.iter().enumerate() {
        let length = strip.dim(dim)?;
        if left + right > length {
            bail!("spatial decode strip {index} is {length} px but ramps {left}+{right} px");
        }
        match pending_tail.take() {
            Some(previous) => {
                if previous.dim(dim)? != *left {
                    bail!(
                        "spatial decode strip {index} expects a {left} px overlap but the \
                         previous strip left {} px",
                        previous.dim(dim)?
                    );
                }
                segments.push(crossfade(&previous, &strip.narrow(dim, 0, *left)?, dim)?);
            }
            None if *left != 0 => {
                bail!("spatial decode strip {index} ramps in with nothing to ramp from")
            }
            None => {}
        }
        segments.push(strip.narrow(dim, *left, length - left - right)?);
        if *right > 0 {
            pending_tail = Some(strip.narrow(dim, length - right, *right)?);
        }
    }
    if pending_tail.is_some() {
        bail!("the last spatial decode strip ramps out with nothing to ramp into");
    }
    cat_dim(&segments, dim)
}

fn crossfade(previous: &Tensor, next: &Tensor, dim: usize) -> Result<Tensor> {
    let length = previous.dim(dim)?;
    let weights = (0..length)
        .map(|index| (index + 1) as f32 / (length + 1) as f32)
        .collect::<Vec<_>>();
    let mut shape = vec![1usize; previous.rank()];
    shape[dim] = length;
    let ramp = Tensor::from_vec(weights, shape, previous.device())?.to_dtype(previous.dtype())?;
    let inverse = ramp.affine(-1.0, 1.0)?;
    previous
        .broadcast_mul(&inverse)?
        .add(&next.broadcast_mul(&ramp)?)
}

fn positive_env_usize(key: &str) -> Option<usize> {
    crate::runtime_env::value(key)
        .and_then(|value| crate::runtime_env::parse_u64(&value))
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
}

#[derive(Debug, Clone)]
pub struct Ltx2VideoEncoder {
    patch_size: usize,
    conv_in: Ltx2VideoCausalConv3d,
    down_blocks: Vec<EncoderBlock>,
    norm_out: PerChannelRmsNorm,
    conv_out: Ltx2VideoCausalConv3d,
    latent_log_var: LatentLogVar,
}

impl Ltx2VideoEncoder {
    pub fn new(config: &AutoencoderKLLtx2VideoConfig, vb: VarBuilder) -> Result<Self> {
        if config.timestep_conditioning {
            bail!("timestep-conditioned LTX-2 VAE encoder is not implemented");
        }

        let conv_in = Ltx2VideoCausalConv3d::new(
            config.in_channels * config.patch_size * config.patch_size,
            config.encoder_base_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            config.encoder_spatial_padding_mode,
            vb.pp("conv_in"),
        )?;

        let mut down_blocks = Vec::with_capacity(config.encoder_blocks.len());
        let mut current_channels = config.encoder_base_channels;
        for (index, block_cfg) in config.encoder_blocks.iter().enumerate() {
            let (block, out_channels) = build_encoder_block(
                block_cfg,
                current_channels,
                config.resnet_eps,
                config.encoder_spatial_padding_mode,
                vb.pp(format!("down_blocks.{index}")),
            )?;
            down_blocks.push(block);
            current_channels = out_channels;
        }

        let conv_out_channels = match config.latent_log_var {
            LatentLogVar::PerChannel => config.latent_channels * 2,
            LatentLogVar::Uniform | LatentLogVar::Constant => config.latent_channels + 1,
            LatentLogVar::None => config.latent_channels,
        };
        let conv_out = Ltx2VideoCausalConv3d::new(
            current_channels,
            conv_out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            config.encoder_spatial_padding_mode,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            patch_size: config.patch_size,
            conv_in,
            down_blocks,
            norm_out: PerChannelRmsNorm::new(1, config.resnet_eps),
            conv_out,
            latent_log_var: config.latent_log_var,
        })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let mut hidden = patchify_video(sample, self.patch_size, 1)?;
        hidden = self.conv_in.forward(&hidden, true)?;
        for block in &self.down_blocks {
            hidden = block.forward(&hidden, true)?;
        }
        hidden = self.norm_out.forward(&hidden)?;
        hidden = silu(&hidden)?;
        hidden = self.conv_out.forward(&hidden, true)?;

        match self.latent_log_var {
            LatentLogVar::Uniform => {
                let channels = hidden.dim(1)?;
                let means = hidden.i((.., 0..channels - 1, .., .., ..))?;
                let logvar = hidden.i((.., channels - 1..channels, .., .., ..))?;
                let repeated_logvar = logvar.repeat((1, means.dim(1)?, 1, 1, 1))?;
                Tensor::cat(&[&means, &repeated_logvar], 1)
            }
            LatentLogVar::Constant => {
                let channels = hidden.dim(1)?;
                let means = hidden.i((.., 0..channels - 1, .., .., ..))?;
                let approx_ln_0 =
                    Tensor::full(-30f32, means.shape(), means.device())?.to_dtype(means.dtype())?;
                Tensor::cat(&[&means, &approx_ln_0], 1)
            }
            _ => Ok(hidden),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2VideoDecoder {
    patch_size: usize,
    conv_in: Ltx2VideoCausalConv3d,
    up_blocks: Vec<DecoderBlock>,
    norm_out: PerChannelRmsNorm,
    conv_out: Ltx2VideoCausalConv3d,
    causal: bool,
}

impl Ltx2VideoDecoder {
    pub fn new(config: &AutoencoderKLLtx2VideoConfig, vb: VarBuilder) -> Result<Self> {
        if config.timestep_conditioning {
            bail!("timestep-conditioned LTX-2 VAE decoder is not implemented");
        }

        let feature_scale = config
            .decoder_blocks
            .iter()
            .filter(|block| block.name.starts_with("compress_"))
            .map(|block| block.multiplier.max(1))
            .product::<usize>();
        let conv_in_channels = config.decoder_base_channels * feature_scale.max(1);
        let conv_in = Ltx2VideoCausalConv3d::new(
            config.latent_channels,
            conv_in_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            config.decoder_causal,
            config.decoder_spatial_padding_mode,
            vb.pp("conv_in"),
        )?;

        let mut up_blocks = Vec::with_capacity(config.decoder_blocks.len());
        let mut current_channels = conv_in_channels;
        for (index, block_cfg) in config.decoder_blocks.iter().rev().enumerate() {
            let (block, out_channels) = build_decoder_block(
                block_cfg,
                current_channels,
                config.resnet_eps,
                config.decoder_spatial_padding_mode,
                vb.pp(format!("up_blocks.{index}")),
            )?;
            up_blocks.push(block);
            current_channels = out_channels;
        }

        let conv_out = Ltx2VideoCausalConv3d::new(
            current_channels,
            config.out_channels * config.patch_size * config.patch_size,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            config.decoder_causal,
            config.decoder_spatial_padding_mode,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            patch_size: config.patch_size,
            conv_in,
            up_blocks,
            norm_out: PerChannelRmsNorm::new(1, config.resnet_eps),
            conv_out,
            causal: config.decoder_causal,
        })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let mut hidden = self.conv_in.forward(sample, self.causal)?;
        for block in &self.up_blocks {
            hidden = block.forward(&hidden, self.causal)?;
        }
        hidden = self.norm_out.forward(&hidden)?;
        hidden = silu(&hidden)?;
        hidden = self.conv_out.forward(&hidden, self.causal)?;
        unpatchify_video(&hidden, self.patch_size, 1)
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKLLtx2Video {
    encoder: Ltx2VideoEncoder,
    decoder: Ltx2VideoDecoder,
    latents_mean: Tensor,
    latents_std: Tensor,
    latent_stats_cache: Arc<Mutex<HashMap<VideoVaeLatentStatsCacheKey, (Tensor, Tensor)>>>,
    #[allow(dead_code)]
    scaling_factor: f64,
    #[allow(dead_code)]
    spatial_compression_ratio: usize,
    #[allow(dead_code)]
    temporal_compression_ratio: usize,
    config: AutoencoderKLLtx2VideoConfig,
    pub use_tiling: bool,
    pub use_framewise_decoding: bool,
    /// Split each frame into overlapping spatial tiles before decoding.
    ///
    /// Temporal chunking alone stops helping once a *single* frame is large:
    /// the decoder's activations scale with `H x W` too, so a 4K decode still
    /// peaks far above what a consumer card has. `None` decodes exactly as
    /// this VAE always has.
    pub(crate) spatial_decode_tiling: Option<SpatialDecodeTiling>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct VideoVaeLatentStatsCacheKey {
    channels: usize,
    dtype: String,
    device: String,
}

impl AutoencoderKLLtx2Video {
    pub fn new(config: AutoencoderKLLtx2VideoConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Ltx2VideoEncoder::new(&config, vb.pp("encoder"))?;
        let decoder = Ltx2VideoDecoder::new(&config, vb.pp("decoder"))?;
        let stats_vb = vb.pp("per_channel_statistics");
        let latents_mean = if stats_vb.contains_tensor("mean-of-means") {
            stats_vb.get(config.latent_channels, "mean-of-means")?
        } else {
            Tensor::new(config.latents_mean.as_slice(), vb.device())?.to_dtype(vb.dtype())?
        };
        let latents_std = if stats_vb.contains_tensor("std-of-means") {
            stats_vb.get(config.latent_channels, "std-of-means")?
        } else {
            Tensor::new(config.latents_std.as_slice(), vb.device())?.to_dtype(vb.dtype())?
        };

        Ok(Self {
            encoder,
            decoder,
            latents_mean,
            latents_std,
            latent_stats_cache: Arc::new(Mutex::new(HashMap::new())),
            scaling_factor: config.scaling_factor,
            spatial_compression_ratio: config.spatial_compression_ratio,
            temporal_compression_ratio: config.temporal_compression_ratio,
            config,
            use_tiling: false,
            use_framewise_decoding: false,
            spatial_decode_tiling: None,
        })
    }

    fn broadcast_latent_stats(&self, latents: &Tensor) -> Result<((Tensor, Tensor), bool)> {
        let channels = latents.dim(1)?;
        let key = VideoVaeLatentStatsCacheKey {
            channels,
            dtype: format!("{:?}", latents.dtype()),
            device: format!("{:?}", latents.device()),
        };
        if let Some((mean, std)) = self
            .latent_stats_cache
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(((mean, std), true));
        }

        let mean = self
            .latents_mean
            .reshape((1, channels, 1, 1, 1))?
            .to_device(latents.device())?;
        let mean = if mean.dtype() == latents.dtype() {
            mean
        } else {
            mean.to_dtype(latents.dtype())?
        };
        let std = self
            .latents_std
            .reshape((1, channels, 1, 1, 1))?
            .to_device(latents.device())?;
        let std = if std.dtype() == latents.dtype() {
            std
        } else {
            std.to_dtype(latents.dtype())?
        };
        self.latent_stats_cache
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .insert(key, (mean.clone(), std.clone()));
        Ok(((mean, std), false))
    }

    pub(crate) fn normalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let ((mean, std), _) = self.broadcast_latent_stats(latents)?;
        latents.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    pub(crate) fn denormalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let ((mean, std), _) = self.broadcast_latent_stats(latents)?;
        latents.broadcast_mul(&std)?.broadcast_add(&mean)
    }

    #[allow(dead_code)]
    pub fn latents_mean(&self) -> &Tensor {
        &self.latents_mean
    }

    #[allow(dead_code)]
    pub fn latents_std(&self) -> &Tensor {
        &self.latents_std
    }

    #[allow(dead_code)]
    pub fn scaling_factor(&self) -> f64 {
        self.scaling_factor
    }

    #[allow(dead_code)]
    pub fn spatial_compression_ratio(&self) -> usize {
        self.spatial_compression_ratio
    }

    #[allow(dead_code)]
    pub fn temporal_compression_ratio(&self) -> usize {
        self.temporal_compression_ratio
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &AutoencoderKLLtx2VideoConfig {
        &self.config
    }

    pub fn encode(&self, sample: &Tensor) -> Result<Tensor> {
        let encoded = self.encoder.forward(sample)?;
        match self.config.latent_log_var {
            LatentLogVar::None => self.normalize_latents(&encoded),
            _ => {
                let channels = encoded.dim(1)? / 2;
                let means = encoded.i((.., 0..channels, .., .., ..))?;
                self.normalize_latents(&means)
            }
        }
    }

    pub fn decode(
        &self,
        latents: &Tensor,
        _temb: Option<&Tensor>,
        return_dict: bool,
        _train: bool,
    ) -> Result<(Option<DecoderOutput>, Tensor)> {
        let latents = self.denormalize_latents(latents)?;
        let decoded = match self.spatial_decode_tiling {
            Some(tiling) => self.decode_spatial_tiles(&latents, tiling)?,
            None => self.decode_frames(&latents)?,
        };
        if return_dict {
            Ok((
                Some(DecoderOutput {
                    sample: decoded.clone(),
                }),
                decoded,
            ))
        } else {
            Ok((None, decoded))
        }
    }

    /// Decode a whole latent block, temporally chunked when configured.
    fn decode_frames(&self, latents: &Tensor) -> Result<Tensor> {
        if self.use_framewise_decoding || self.use_tiling {
            self.decode_temporal_chunks(latents)
        } else {
            self.decoder.forward(latents)
        }
    }

    /// Decode in overlapping spatial tiles and crossfade the seams.
    ///
    /// Tiles are cut with the same splitter stage-2 refinement uses, and the
    /// overlap is crossfaded with the same `i / (overlap + 1)` ramp as
    /// `trapezoidal_mask`, so the two sides sum to exactly one. The blend is
    /// done strip by strip rather than into one full-size accumulator: at the
    /// resolutions that need this at all, a padded full-frame buffer per tile
    /// would cost more than the decode it is saving.
    fn decode_spatial_tiles(
        &self,
        latents: &Tensor,
        tiling: SpatialDecodeTiling,
    ) -> Result<Tensor> {
        let (_batch, _channels, _frames, latent_height, latent_width) = latents.dims5()?;
        let scale = self.spatial_compression_ratio.max(1);
        let rows = split_by_size(latent_height, tiling.tile_cells, tiling.overlap_cells)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        let columns = split_by_size(latent_width, tiling.tile_cells, tiling.overlap_cells)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        if rows.len() <= 1 && columns.len() <= 1 {
            return self.decode_frames(latents);
        }

        let mut row_strips = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut column_strips = Vec::with_capacity(columns.len());
            for column in &columns {
                let tile = latents
                    .i((.., .., .., row.start..row.end, column.start..column.end))?
                    .contiguous()?;
                column_strips.push((
                    self.decode_frames(&tile)?,
                    column.left_ramp * scale,
                    column.right_ramp * scale,
                ));
            }
            row_strips.push((
                crossfade_strips(&column_strips, 4)?,
                row.left_ramp * scale,
                row.right_ramp * scale,
            ));
        }
        crossfade_strips(&row_strips, 3)
    }

    fn temporal_decode_chunk_latent_frames(&self) -> usize {
        positive_env_usize("MOLD_LTX2_VAE_DECODE_CHUNK_FRAMES")
            .unwrap_or(DEFAULT_TEMPORAL_DECODE_CHUNK_LATENT_FRAMES)
    }

    fn temporal_decode_context_latent_frames(&self) -> usize {
        positive_env_usize("MOLD_LTX2_VAE_DECODE_CONTEXT_FRAMES")
            .unwrap_or_else(|| self.decoder_temporal_context_latent_frames())
    }

    fn decoder_temporal_context_latent_frames(&self) -> usize {
        let mut causal_conv_count = 2usize; // decoder conv_in + conv_out.
        for block in &self.config.decoder_blocks {
            match block.name.as_str() {
                "res_x" | "res_x_y" => {
                    causal_conv_count =
                        causal_conv_count.saturating_add(block.num_layers.saturating_mul(2));
                }
                name if name.starts_with("compress_") => {
                    causal_conv_count = causal_conv_count.saturating_add(1);
                }
                _ => {}
            }
        }
        causal_conv_count.saturating_mul(2)
    }

    fn temporal_decode_chunk_plan(
        &self,
        latent_frames: usize,
        chunk_latent_frames: usize,
    ) -> Result<Vec<TemporalDecodeChunk>> {
        if chunk_latent_frames == 0 {
            bail!("LTX-2 VAE temporal decode chunk size must be greater than zero");
        }
        if latent_frames == 0 {
            return Ok(Vec::new());
        }

        let temporal_scale = self.temporal_compression_ratio.max(1);
        let context = self.temporal_decode_context_latent_frames();
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < latent_frames {
            let end = start.saturating_add(chunk_latent_frames).min(latent_frames);
            let input_start = start.saturating_sub(context);
            let input_end = end.saturating_add(context).min(latent_frames);
            let output_start = temporal_output_frames_for_latents(
                start.saturating_sub(input_start),
                temporal_scale,
            );
            let output_end =
                temporal_output_frames_for_latents(end.saturating_sub(input_start), temporal_scale);
            chunks.push(TemporalDecodeChunk {
                input_start,
                input_end,
                output_start,
                output_end,
            });
            start = end;
        }
        Ok(chunks)
    }

    fn decode_temporal_chunks(&self, latents: &Tensor) -> Result<Tensor> {
        let latent_frames = latents.dim(2)?;
        let chunks = self.temporal_decode_chunk_plan(
            latent_frames,
            self.temporal_decode_chunk_latent_frames(),
        )?;
        if chunks.len() <= 1 {
            return self.decoder.forward(latents);
        }

        let mut decoded_chunks = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let chunk_latents = latents.i((.., .., chunk.input_start..chunk.input_end, .., ..))?;
            let decoded = self.decoder.forward(&chunk_latents)?;
            let decoded_frames = decoded.dim(2)?;
            if chunk.output_end > decoded_frames {
                bail!(
                    "LTX-2 VAE temporal chunk output {}..{} exceeds decoded frame count {}",
                    chunk.output_start,
                    chunk.output_end,
                    decoded_frames
                );
            }
            decoded_chunks.push(decoded.i((
                ..,
                ..,
                chunk.output_start..chunk.output_end,
                ..,
                ..,
            ))?);
        }
        cat_dim(&decoded_chunks, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        patchify_video, unpatchify_video, AutoencoderKLLtx2Video, AutoencoderKLLtx2VideoConfig,
        Ltx2VideoCausalConv3d, Ltx2VideoDownsampler3d, Ltx2VideoResnetBlock3d,
        Ltx2VideoUpsampler3d, PerChannelRmsNorm, SpatialDecodeTiling, SpatialPaddingMode,
        VaeBlockConfig,
    };
    use candle_core::{DType, Device, Tensor};
    use candle_nn::group_norm;
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    /// Decode one fixed latent through the real VAE on Metal and on the CPU.
    ///
    /// A 512x512x9 distilled render produced healthy final latents (rms 1.393)
    /// and an all-NaN decode on Metal, which clamps to a black clip with no
    /// error. Forcing the decode to F32 did not change it, so the fault is not
    /// precision. This isolates the decoder from the 19B transformer: point
    /// `MOLD_TEST_LTX2_CHECKPOINT` at a combined LTX-2 checkpoint to run it.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_vae_decode_matches_the_cpu_reference() {
        let Some(path) = std::env::var_os("MOLD_TEST_LTX2_CHECKPOINT") else {
            return;
        };
        let decode_on = |device: Device, dtype: DType, spatial: usize| -> (f32, f32, usize) {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&std::path::PathBuf::from(&path)),
                    dtype,
                    &device,
                )
            }
            .unwrap()
            .pp("vae");
            let vae =
                AutoencoderKLLtx2Video::new(AutoencoderKLLtx2VideoConfig::default(), vb).unwrap();
            // Same values on both devices: seeded on the CPU, then moved.
            let count = 128.0 * 2.0 * (spatial * spatial) as f32;
            let latents = Tensor::arange(0f32, count, &Device::Cpu)
                .unwrap()
                .reshape((1usize, 128usize, 2usize, spatial, spatial))
                .unwrap()
                .affine(1.0 / 1024.0, -0.5)
                .unwrap()
                .to_device(&device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let (_, video) = vae.decode(&latents, None, false, false).unwrap();
            let flat = video.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
            let values = flat.to_vec1::<f32>().unwrap();
            let nan = values.iter().filter(|value| value.is_nan()).count();
            let finite = values.iter().filter(|value| value.is_finite());
            let count = finite.clone().count().max(1);
            let mean = finite.clone().sum::<f32>() / count as f32;
            let max = finite.fold(0f32, |acc, value| acc.max(value.abs()));
            (mean, max, nan)
        };

        // One load, several decode shapes: a size-dependent failure points at
        // a kernel dispatch limit rather than the maths.
        for spatial in [2usize, 4, 8, 16] {
            let (_, _, cpu_nan) = decode_on(Device::Cpu, DType::F32, spatial);
            let (_, _, metal_nan) = decode_on(Device::new_metal(0).unwrap(), DType::F32, spatial);
            println!("latent {spatial}x{spatial}: cpu_nan={cpu_nan} metal_nan={metal_nan}");
        }
    }

    fn insert_conv(
        tensors: &mut HashMap<String, Tensor>,
        path: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
    ) {
        tensors.insert(
            format!("{path}.conv.weight"),
            Tensor::zeros(
                (out_channels, in_channels, kernel.0, kernel.1, kernel.2),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{path}.conv.bias"),
            Tensor::zeros(out_channels, DType::F32, &Device::Cpu).unwrap(),
        );
    }

    fn insert_patterned_conv(
        tensors: &mut HashMap<String, Tensor>,
        path: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
    ) {
        let len = out_channels * in_channels * kernel.0 * kernel.1 * kernel.2;
        let weights = (0..len)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.0025)
            .collect::<Vec<_>>();
        tensors.insert(
            format!("{path}.conv.weight"),
            Tensor::from_vec(
                weights,
                (out_channels, in_channels, kernel.0, kernel.1, kernel.2),
                &Device::Cpu,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{path}.conv.bias"),
            Tensor::zeros(out_channels, DType::F32, &Device::Cpu).unwrap(),
        );
    }

    fn insert_res_stack(
        tensors: &mut HashMap<String, Tensor>,
        path: &str,
        channels: usize,
        num_layers: usize,
    ) {
        for index in 0..num_layers {
            insert_conv(
                tensors,
                &format!("{path}.res_blocks.{index}.conv1"),
                channels,
                channels,
                (3, 3, 3),
            );
            insert_conv(
                tensors,
                &format!("{path}.res_blocks.{index}.conv2"),
                channels,
                channels,
                (3, 3, 3),
            );
        }
    }

    fn insert_patterned_res_stack(
        tensors: &mut HashMap<String, Tensor>,
        path: &str,
        channels: usize,
        num_layers: usize,
    ) {
        for index in 0..num_layers {
            insert_patterned_conv(
                tensors,
                &format!("{path}.res_blocks.{index}.conv1"),
                channels,
                channels,
                (3, 3, 3),
            );
            insert_patterned_conv(
                tensors,
                &format!("{path}.res_blocks.{index}.conv2"),
                channels,
                channels,
                (3, 3, 3),
            );
        }
    }

    /// `PerChannelRmsNorm` is the decoder's last normalization and touches
    /// every output value, so a NaN here blackens the whole frame. It reduces
    /// with `mean_keepdim` over the channel dim of a 5-D tensor and then takes
    /// `sqrt`, so a reduction that returns a negative on Metal yields NaN.
    #[cfg(feature = "metal")]
    #[test]
    fn per_channel_rms_norm_agrees_between_metal_and_cpu() {
        let metal = Device::new_metal(0).unwrap();
        let shape = (1usize, 8usize, 3usize, 4usize, 4usize);
        let count: usize = shape.0 * shape.1 * shape.2 * shape.3 * shape.4;

        let build = |device: &Device| {
            Tensor::arange(0f32, count as f32, device)
                .unwrap()
                .reshape(shape)
                .unwrap()
                .affine(0.01, -0.5)
                .unwrap()
        };
        let cpu_input = build(&Device::Cpu);
        let metal_input = build(&metal);

        let norm = PerChannelRmsNorm::new(1, 1e-6);
        let cpu_out = norm.forward(&cpu_input).unwrap();
        let metal_out = norm.forward(&metal_input).unwrap();
        let metal_v = metal_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cpu_v = cpu_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nan = metal_v.iter().filter(|value| value.is_nan()).count();
        assert_eq!(
            nan, 0,
            "PerChannelRmsNorm produced {nan} NaN values on Metal"
        );
        for (cpu, metal) in cpu_v.iter().zip(metal_v.iter()) {
            assert!(
                (cpu - metal).abs() <= 1e-3 * cpu.abs().max(1.0),
                "rms norm disagrees: cpu {cpu} metal {metal}"
            );
        }
    }

    /// Which VAE primitive returns NaN on Metal but not on the CPU.
    ///
    /// The full decoder returns 100% NaN on Metal from healthy latents. The
    /// blocks below are the decoder's primitives, run with identical synthetic
    /// weights on both devices, so a failure names the op instead of the
    /// pipeline. Needs no checkpoint, so it runs in CI on any Metal host.
    #[cfg(feature = "metal")]
    #[test]
    fn vae_primitives_agree_between_metal_and_cpu() {
        let metal = Device::new_metal(0).unwrap();
        let nan_count = |tensor: &Tensor| -> usize {
            tensor
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .filter(|value| value.is_nan())
                .count()
        };

        // A causal conv on its own.
        for device in [Device::Cpu, metal.clone()] {
            let vb = {
                let mut tensors = HashMap::new();
                insert_conv(&mut tensors, "conv", 4, 4, (3, 3, 3));
                VarBuilder::from_tensors(tensors, DType::F32, &device)
            };
            let conv = Ltx2VideoCausalConv3d::new(
                4,
                4,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                1,
                true,
                SpatialPaddingMode::Zeros,
                vb.pp("conv"),
            )
            .unwrap();
            let input = Tensor::ones((1, 4, 3, 8, 8), DType::F32, &device).unwrap();
            let out = conv.forward(&input, true).unwrap();
            assert_eq!(
                nan_count(&out),
                0,
                "causal conv produced NaN on {:?}",
                out.device()
            );
        }

        // A resnet block, which adds the GroupNorm + SiLU path.
        for device in [Device::Cpu, metal.clone()] {
            let vb = {
                let mut tensors = HashMap::new();
                insert_conv(&mut tensors, "conv1", 4, 4, (3, 3, 3));
                insert_conv(&mut tensors, "conv2", 4, 4, (3, 3, 3));
                VarBuilder::from_tensors(tensors, DType::F32, &device)
            };
            let block =
                Ltx2VideoResnetBlock3d::new(4, 4, 1e-6, false, SpatialPaddingMode::Zeros, vb)
                    .unwrap();
            let input = Tensor::ones((1, 4, 3, 8, 8), DType::F32, &device).unwrap();
            let out = block.forward(&input, true).unwrap();
            assert_eq!(
                nan_count(&out),
                0,
                "resnet block produced NaN on {:?}",
                out.device()
            );
        }

        // A bare GroupNorm over a 5-D activation, reshaped the way the VAE does.
        for device in [Device::Cpu, metal.clone()] {
            let vb = VarBuilder::from_tensors(
                HashMap::from([
                    (
                        "weight".to_string(),
                        Tensor::ones(4, DType::F32, &device).unwrap(),
                    ),
                    (
                        "bias".to_string(),
                        Tensor::zeros(4, DType::F32, &device).unwrap(),
                    ),
                ]),
                DType::F32,
                &device,
            );
            let norm = group_norm(1, 4, 1e-6, vb).unwrap();
            let input = Tensor::ones((1, 4, 3, 8, 8), DType::F32, &device).unwrap();
            let out = candle_core::Module::forward(&norm, &input).unwrap();
            assert_eq!(
                nan_count(&out),
                0,
                "group norm produced NaN on {:?}",
                out.device()
            );
        }
    }

    fn tiny_autoencoder_var_builder(config: &AutoencoderKLLtx2VideoConfig) -> VarBuilder<'static> {
        let mut tensors = HashMap::new();

        insert_conv(
            &mut tensors,
            "encoder.conv_in",
            config.in_channels * config.patch_size * config.patch_size,
            config.encoder_base_channels,
            (3, 3, 3),
        );
        let mut encoder_channels = config.encoder_base_channels;
        for (index, block) in config.encoder_blocks.iter().enumerate() {
            match block.name.as_str() {
                "res_x" => {
                    insert_res_stack(
                        &mut tensors,
                        &format!("encoder.down_blocks.{index}"),
                        encoder_channels,
                        block.num_layers,
                    );
                }
                "compress_all_res" | "compress_space_res" | "compress_time_res" => {
                    let stride = match block.name.as_str() {
                        "compress_space_res" => (1, 2, 2),
                        "compress_time_res" => (2, 1, 1),
                        _ => (2, 2, 2),
                    };
                    let stride_product = stride.0 * stride.1 * stride.2;
                    insert_conv(
                        &mut tensors,
                        &format!("encoder.down_blocks.{index}.conv"),
                        encoder_channels,
                        encoder_channels * block.multiplier / stride_product,
                        (3, 3, 3),
                    );
                    encoder_channels *= block.multiplier;
                }
                other => panic!("unsupported test encoder block {other}"),
            }
        }
        insert_conv(
            &mut tensors,
            "encoder.conv_out",
            encoder_channels,
            config.latent_channels + 1,
            (3, 3, 3),
        );

        let feature_scale = config
            .decoder_blocks
            .iter()
            .filter(|block| block.name.starts_with("compress_"))
            .map(|block| block.multiplier.max(1))
            .product::<usize>();
        let mut decoder_channels = config.decoder_base_channels * feature_scale.max(1);
        insert_conv(
            &mut tensors,
            "decoder.conv_in",
            config.latent_channels,
            decoder_channels,
            (3, 3, 3),
        );
        for (index, block) in config.decoder_blocks.iter().rev().enumerate() {
            match block.name.as_str() {
                "res_x" => {
                    insert_res_stack(
                        &mut tensors,
                        &format!("decoder.up_blocks.{index}"),
                        decoder_channels,
                        block.num_layers,
                    );
                }
                "compress_all" => {
                    let stride_product = 8;
                    insert_conv(
                        &mut tensors,
                        &format!("decoder.up_blocks.{index}.conv"),
                        decoder_channels,
                        stride_product * decoder_channels / block.multiplier,
                        (3, 3, 3),
                    );
                    decoder_channels /= block.multiplier;
                }
                other => panic!("unsupported test decoder block {other}"),
            }
        }
        insert_conv(
            &mut tensors,
            "decoder.conv_out",
            decoder_channels,
            config.out_channels * config.patch_size * config.patch_size,
            (3, 3, 3),
        );

        tensors.insert(
            "per_channel_statistics.mean-of-means".to_string(),
            Tensor::zeros(config.latent_channels, DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "per_channel_statistics.std-of-means".to_string(),
            Tensor::ones(config.latent_channels, DType::F32, &Device::Cpu).unwrap(),
        );

        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    fn patterned_autoencoder_var_builder(
        config: &AutoencoderKLLtx2VideoConfig,
    ) -> VarBuilder<'static> {
        let mut tensors = HashMap::new();

        insert_conv(
            &mut tensors,
            "encoder.conv_in",
            config.in_channels * config.patch_size * config.patch_size,
            config.encoder_base_channels,
            (3, 3, 3),
        );
        let mut encoder_channels = config.encoder_base_channels;
        for (index, block) in config.encoder_blocks.iter().enumerate() {
            match block.name.as_str() {
                "res_x" => {
                    insert_res_stack(
                        &mut tensors,
                        &format!("encoder.down_blocks.{index}"),
                        encoder_channels,
                        block.num_layers,
                    );
                }
                "compress_all_res" | "compress_space_res" | "compress_time_res" => {
                    let stride = match block.name.as_str() {
                        "compress_space_res" => (1, 2, 2),
                        "compress_time_res" => (2, 1, 1),
                        _ => (2, 2, 2),
                    };
                    let stride_product = stride.0 * stride.1 * stride.2;
                    insert_conv(
                        &mut tensors,
                        &format!("encoder.down_blocks.{index}.conv"),
                        encoder_channels,
                        encoder_channels * block.multiplier / stride_product,
                        (3, 3, 3),
                    );
                    encoder_channels *= block.multiplier;
                }
                other => panic!("unsupported test encoder block {other}"),
            }
        }
        insert_conv(
            &mut tensors,
            "encoder.conv_out",
            encoder_channels,
            config.latent_channels + 1,
            (3, 3, 3),
        );

        let feature_scale = config
            .decoder_blocks
            .iter()
            .filter(|block| block.name.starts_with("compress_"))
            .map(|block| block.multiplier.max(1))
            .product::<usize>();
        let mut decoder_channels = config.decoder_base_channels * feature_scale.max(1);
        insert_patterned_conv(
            &mut tensors,
            "decoder.conv_in",
            config.latent_channels,
            decoder_channels,
            (3, 3, 3),
        );
        for (index, block) in config.decoder_blocks.iter().rev().enumerate() {
            match block.name.as_str() {
                "res_x" => {
                    insert_patterned_res_stack(
                        &mut tensors,
                        &format!("decoder.up_blocks.{index}"),
                        decoder_channels,
                        block.num_layers,
                    );
                }
                "compress_all" => {
                    let stride_product = 8;
                    insert_patterned_conv(
                        &mut tensors,
                        &format!("decoder.up_blocks.{index}.conv"),
                        decoder_channels,
                        stride_product * decoder_channels / block.multiplier,
                        (3, 3, 3),
                    );
                    decoder_channels /= block.multiplier;
                }
                other => panic!("unsupported test decoder block {other}"),
            }
        }
        insert_patterned_conv(
            &mut tensors,
            "decoder.conv_out",
            decoder_channels,
            config.out_channels * config.patch_size * config.patch_size,
            (3, 3, 3),
        );

        tensors.insert(
            "per_channel_statistics.mean-of-means".to_string(),
            Tensor::zeros(config.latent_channels, DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "per_channel_statistics.std-of-means".to_string(),
            Tensor::ones(config.latent_channels, DType::F32, &Device::Cpu).unwrap(),
        );

        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    fn assert_tensors_close(lhs: &Tensor, rhs: &Tensor, tolerance: f32) {
        assert_eq!(lhs.shape(), rhs.shape());
        let lhs = lhs
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rhs = rhs
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (idx, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (l - r).abs() <= tolerance,
                "tensor mismatch at {idx}: {l} vs {r}"
            );
        }
    }

    #[test]
    fn patchify_roundtrips_back_to_input_layout() {
        let device = Device::Cpu;
        let sample = Tensor::arange(0u32, 3 * 3 * 4 * 4, &device)
            .unwrap()
            .reshape((1, 3, 3, 4, 4))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let patched = patchify_video(&sample, 2, 1).unwrap();
        let roundtrip = unpatchify_video(&patched, 2, 1).unwrap();
        let lhs = sample.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rhs = roundtrip.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn residual_downsample_and_upsample_keep_expected_shapes() {
        let device = Device::Cpu;

        let down_vb = {
            let mut tensors = HashMap::new();
            insert_conv(&mut tensors, "conv", 4, 1, (3, 3, 3));
            VarBuilder::from_tensors(tensors, DType::F32, &device)
        };
        let down = Ltx2VideoDownsampler3d::new(4, 8, (2, 2, 2), SpatialPaddingMode::Zeros, down_vb)
            .unwrap();
        let hidden = Tensor::zeros((1, 4, 3, 4, 4), DType::F32, &device).unwrap();
        let downsampled = down.forward(&hidden, true).unwrap();
        assert_eq!(downsampled.dims5().unwrap(), (1, 8, 2, 2, 2));

        let up_vb = {
            let mut tensors = HashMap::new();
            insert_conv(&mut tensors, "conv", 8, 32, (3, 3, 3));
            VarBuilder::from_tensors(tensors, DType::F32, &device)
        };
        let up =
            Ltx2VideoUpsampler3d::new(8, (2, 2, 2), true, 2, SpatialPaddingMode::Reflect, up_vb)
                .unwrap();
        let upsampled = up.forward(&downsampled, false).unwrap();
        assert_eq!(upsampled.dims5().unwrap(), (1, 4, 3, 4, 4));
    }

    #[test]
    fn vae_norm_layers_respect_configured_epsilon() {
        let eps = 1e-6;
        let block_vb = {
            let mut tensors = HashMap::new();
            insert_conv(&mut tensors, "conv1", 4, 4, (3, 3, 3));
            insert_conv(&mut tensors, "conv2", 4, 4, (3, 3, 3));
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
        };
        let block =
            Ltx2VideoResnetBlock3d::new(4, 4, eps, false, SpatialPaddingMode::Zeros, block_vb)
                .unwrap();
        assert_eq!(block.norm1.eps, eps);
        assert_eq!(block.norm2.eps, eps);

        let config = AutoencoderKLLtx2VideoConfig {
            latent_channels: 4,
            encoder_base_channels: 4,
            decoder_base_channels: 4,
            encoder_blocks: vec![VaeBlockConfig::res_x(1)],
            decoder_blocks: vec![VaeBlockConfig::res_x(1)],
            resnet_eps: eps,
            latents_mean: vec![0.0; 4],
            latents_std: vec![1.0; 4],
            ..Default::default()
        };
        let vae =
            AutoencoderKLLtx2Video::new(config.clone(), tiny_autoencoder_var_builder(&config))
                .unwrap();
        assert_eq!(vae.encoder.norm_out.eps, eps);
        assert_eq!(vae.decoder.norm_out.eps, eps);
    }

    #[test]
    fn video_vae_latent_stats_cache_reuses_broadcast_tensors() {
        let config = AutoencoderKLLtx2VideoConfig {
            latent_channels: 4,
            encoder_base_channels: 4,
            decoder_base_channels: 4,
            encoder_blocks: vec![VaeBlockConfig::res_x(1)],
            decoder_blocks: vec![VaeBlockConfig::res_x(1)],
            latents_mean: vec![1.0, 2.0, 3.0, 4.0],
            latents_std: vec![2.0, 4.0, 8.0, 16.0],
            ..Default::default()
        };
        let vae =
            AutoencoderKLLtx2Video::new(config.clone(), tiny_autoencoder_var_builder(&config))
                .unwrap();
        let latents = Tensor::from_vec(
            vec![3.0f32, 10.0, 19.0, 36.0],
            (1, 4, 1, 1, 1),
            &Device::Cpu,
        )
        .unwrap();

        let ((mean, std), first_hit) = vae.broadcast_latent_stats(&latents).unwrap();
        let ((mean_again, std_again), second_hit) = vae.broadcast_latent_stats(&latents).unwrap();
        let normalized = vae.normalize_latents(&latents).unwrap();

        assert!(!first_hit);
        assert!(second_hit);
        assert_eq!(mean.dims5().unwrap(), (1, 4, 1, 1, 1));
        assert_eq!(std.dims5().unwrap(), (1, 4, 1, 1, 1));
        assert_eq!(format!("{:?}", mean_again.device()), "Cpu");
        assert_eq!(std_again.dtype(), DType::F32);
        assert_eq!(
            normalized.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![3.0, 10.0, 19.0, 36.0]
        );
    }

    #[test]
    fn autoencoder_decode_expands_latent_video_shape() {
        let config = AutoencoderKLLtx2VideoConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            encoder_blocks: vec![
                VaeBlockConfig::res_x(1),
                VaeBlockConfig::compress("compress_all_res", 2, false),
            ],
            decoder_blocks: vec![
                VaeBlockConfig::res_x(1),
                VaeBlockConfig::compress("compress_all", 2, true),
                VaeBlockConfig::res_x(1),
            ],
            patch_size: 2,
            resnet_eps: 1e-6,
            scaling_factor: 1.0,
            latent_log_var: super::LatentLogVar::Uniform,
            encoder_base_channels: 4,
            decoder_base_channels: 4,
            encoder_spatial_padding_mode: SpatialPaddingMode::Zeros,
            decoder_spatial_padding_mode: SpatialPaddingMode::Reflect,
            spatial_compression_ratio: 4,
            temporal_compression_ratio: 2,
            timestep_conditioning: false,
            decoder_causal: false,
            latents_mean: vec![0.0; 4],
            latents_std: vec![1.0; 4],
        };
        let vae =
            AutoencoderKLLtx2Video::new(config.clone(), tiny_autoencoder_var_builder(&config))
                .unwrap();
        let latents = Tensor::zeros((1, 4, 2, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let (_output, video) = vae.decode(&latents, None, false, false).unwrap();
        assert_eq!(video.dims5().unwrap(), (1, 3, 3, 8, 8));
    }

    #[test]
    fn autoencoder_framewise_decode_matches_full_decode_across_chunk_boundaries() {
        let config = AutoencoderKLLtx2VideoConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            encoder_blocks: vec![VaeBlockConfig::res_x(1)],
            decoder_blocks: vec![
                VaeBlockConfig::res_x(1),
                VaeBlockConfig::compress("compress_all", 2, true),
                VaeBlockConfig::res_x(1),
            ],
            patch_size: 1,
            resnet_eps: 1e-6,
            scaling_factor: 1.0,
            latent_log_var: super::LatentLogVar::Uniform,
            encoder_base_channels: 4,
            decoder_base_channels: 4,
            encoder_spatial_padding_mode: SpatialPaddingMode::Zeros,
            decoder_spatial_padding_mode: SpatialPaddingMode::Zeros,
            spatial_compression_ratio: 2,
            temporal_compression_ratio: 2,
            timestep_conditioning: false,
            decoder_causal: true,
            latents_mean: vec![0.0; 4],
            latents_std: vec![1.0; 4],
        };
        let full =
            AutoencoderKLLtx2Video::new(config.clone(), patterned_autoencoder_var_builder(&config))
                .unwrap();
        let mut chunked =
            AutoencoderKLLtx2Video::new(config.clone(), patterned_autoencoder_var_builder(&config))
                .unwrap();
        chunked.use_framewise_decoding = true;
        let values = (0..(4 * 9 * 3 * 3))
            .map(|idx| ((idx % 23) as f32 - 11.0) / 13.0)
            .collect::<Vec<_>>();
        let latents = Tensor::from_vec(values, (1, 4, 9, 3, 3), &Device::Cpu).unwrap();

        let chunks = chunked.temporal_decode_chunk_plan(9, 4).unwrap();
        assert_eq!(chunks.len(), 3);

        let (_full_output, full_video) = full.decode(&latents, None, false, false).unwrap();
        let (_chunked_output, chunked_video) =
            chunked.decode(&latents, None, false, false).unwrap();

        assert_tensors_close(&full_video, &chunked_video, 1e-4);
    }

    /// Temporal chunking bounds frames in flight; it does nothing about one
    /// enormous frame. A spatially tiled decode has to reproduce the whole
    /// decode, seams included — a crossfade that does not sum to one shows up
    /// as a bright or dark band, never as an error.
    ///
    /// The residual is not zero and cannot be: at a tile edge the decoder's
    /// convolutions pad instead of seeing the neighbouring pixels, which is
    /// precisely what the overlap exists to hide. So the property asserted
    /// here is convergence — widening the overlap must move the tiled decode
    /// toward the full one — which a mis-weighted blend would not satisfy.
    #[test]
    fn autoencoder_spatial_tiled_decode_converges_on_the_full_decode() {
        let config = AutoencoderKLLtx2VideoConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            encoder_blocks: vec![VaeBlockConfig::res_x(1)],
            decoder_blocks: vec![
                VaeBlockConfig::res_x(1),
                VaeBlockConfig::compress("compress_all", 2, true),
                VaeBlockConfig::res_x(1),
            ],
            patch_size: 1,
            resnet_eps: 1e-6,
            scaling_factor: 1.0,
            latent_log_var: super::LatentLogVar::Uniform,
            encoder_base_channels: 4,
            decoder_base_channels: 4,
            encoder_spatial_padding_mode: SpatialPaddingMode::Zeros,
            decoder_spatial_padding_mode: SpatialPaddingMode::Zeros,
            spatial_compression_ratio: 2,
            temporal_compression_ratio: 2,
            timestep_conditioning: false,
            decoder_causal: true,
            latents_mean: vec![0.0; 4],
            latents_std: vec![1.0; 4],
        };
        let values = (0..(4 * 3 * 10 * 14))
            .map(|idx| ((idx % 23) as f32 - 11.0) / 13.0)
            .collect::<Vec<_>>();
        let latents = Tensor::from_vec(values, (1, 4, 3, 10, 14), &Device::Cpu).unwrap();

        let full =
            AutoencoderKLLtx2Video::new(config.clone(), patterned_autoencoder_var_builder(&config))
                .unwrap();
        let (_full_output, full_video) = full.decode(&latents, None, false, false).unwrap();
        let range = full_video
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        let deviation = |tile_cells: usize, overlap_cells: usize| {
            let mut tiled = AutoencoderKLLtx2Video::new(
                config.clone(),
                patterned_autoencoder_var_builder(&config),
            )
            .unwrap();
            tiled.spatial_decode_tiling = Some(SpatialDecodeTiling {
                tile_cells,
                overlap_cells,
            });
            let (_output, video) = tiled.decode(&latents, None, false, false).unwrap();
            assert_eq!(full_video.dims5().unwrap(), video.dims5().unwrap());
            full_video
                .sub(&video)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        let narrow = deviation(6, 2);
        let wide = deviation(12, 8);
        // Quadrupling the overlap cut the seam by ~6.7x when this was written
        // (0.674 -> 0.100 against a signal peaking at 0.446). The bounds are
        // loose because this fixture is deliberately hostile — four random
        // channels through a zero-padded decoder is a far worse neighbour
        // approximation than a trained VAE — but a blend whose weights did not
        // sum to one would not converge at all.
        assert!(
            wide < narrow * 0.25,
            "a wider overlap must hide most of the seam, got {wide} against {narrow}"
        );
        assert!(
            wide < range * 0.25,
            "a {wide} deviation on a signal peaking at {range} is a seam, not a blend"
        );
    }

    /// The property the crossfade rests on, isolated from any decoder: two
    /// strips whose overlap is faded together reproduce the original.
    #[test]
    fn crossfading_strips_reassembles_the_original_signal() {
        let width = 12usize;
        let values = (0..width).map(|i| i as f32).collect::<Vec<_>>();
        let whole = Tensor::from_vec(values.clone(), (1, 1, 1, 1, width), &Device::Cpu).unwrap();
        let overlap = 4usize;
        let left = whole.narrow(4, 0, 8).unwrap();
        let right = whole.narrow(4, 4, 8).unwrap();

        let joined =
            super::crossfade_strips(&[(left, 0, overlap), (right, overlap, 0)], 4).unwrap();

        assert_eq!(
            joined.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            values
        );
    }

    #[test]
    fn crossfading_rejects_strips_whose_overlaps_disagree() {
        let strip = Tensor::zeros((1, 1, 1, 1, 8), DType::F32, &Device::Cpu).unwrap();
        // A lone strip needs no ramps at all.
        assert!(super::crossfade_strips(&[(strip.clone(), 0, 0)], 4).is_ok());
        // Ramping in with no predecessor, out with no successor, or into an
        // overlap of a different width all mean the split and the blend
        // disagree — which is exactly the silent-seam case.
        assert!(super::crossfade_strips(&[(strip.clone(), 2, 0)], 4).is_err());
        assert!(super::crossfade_strips(&[(strip.clone(), 0, 4)], 4).is_err());
        assert!(
            super::crossfade_strips(&[(strip.clone(), 0, 4), (strip.clone(), 2, 0)], 4).is_err()
        );
    }

    #[test]
    fn ltx2_22b_config_matches_embedded_checkpoint_layout() {
        let config = AutoencoderKLLtx2VideoConfig::ltx2_22b();
        assert_eq!(
            config
                .encoder_blocks
                .iter()
                .map(|block| (
                    block.name.as_str(),
                    block.num_layers,
                    block.multiplier,
                    block.residual
                ))
                .collect::<Vec<_>>(),
            vec![
                ("res_x", 4, 1, false),
                ("compress_space_res", 0, 2, true),
                ("res_x", 6, 1, false),
                ("compress_time_res", 0, 2, true),
                ("res_x", 4, 1, false),
                ("compress_all_res", 0, 2, true),
                ("res_x", 2, 1, false),
                ("compress_all_res", 0, 1, true),
                ("res_x", 2, 1, false),
            ]
        );
        assert_eq!(
            config
                .decoder_blocks
                .iter()
                .map(|block| {
                    (
                        block.name.as_str(),
                        block.num_layers,
                        block.multiplier,
                        block.residual,
                        block.inject_noise,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                ("res_x", 4, 1, false, false),
                ("compress_space", 0, 2, false, false),
                ("res_x", 6, 1, false, false),
                ("compress_time", 0, 2, false, false),
                ("res_x", 4, 1, false, false),
                ("compress_all", 0, 1, false, false),
                ("res_x", 2, 1, false, false),
                ("compress_all", 0, 2, false, false),
                ("res_x", 2, 1, false, false),
            ]
        );
        assert_eq!(
            config.encoder_spatial_padding_mode,
            SpatialPaddingMode::Zeros
        );
        assert_eq!(
            config.decoder_spatial_padding_mode,
            SpatialPaddingMode::Zeros
        );
    }
}
