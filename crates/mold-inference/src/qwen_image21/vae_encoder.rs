//! Image-specialized Qwen Image 2.1 VAE encoder.
//!
//! Ports the encoder half of diffusers `e0abab83b`
//! `models/autoencoders/autoencoder_kl_qwenimage21.py` (cited `V:`) for the
//! one-frame case the pipeline feeds it: `_encode` runs `encoder(x[:, :, :1])`
//! as its first (and only) chunk with an empty feature cache (`V:1259-1263`),
//! so every `QwenImage21CausalConv3d` is a plain 2-D convolution (`V:171-184`),
//! the `downsample3d` resamplers cache their output and SKIP `time_conv`
//! (`V:315-321`), and `quant_conv` follows (`V:1272`).
//!
//! The one place a single frame is not simply "2-D" is the residual shortcut:
//! `QwenImage21AvgDown3D` pads the time axis to a multiple of `factor_t`
//! BEFORE averaging (`V:57-60`), so with `factor_t = 2` it prepends a ZERO
//! frame and half of every temporally-downsampled block's shortcut channels
//! come out exactly zero. [`avg_down_shortcut`] reproduces that exactly.
//!
//! It is a separate struct from the decoder so an encode phase maps only
//! `encoder.*` and `quant_conv.*` (~0.29 GiB) out of the VAE file. Its
//! caller is the reference-conditioned encode phase of the pipeline.
#![allow(dead_code)]

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};
use std::path::Path;

use super::vae::{
    MidBlock2d, ResidualBlock2d, RmsNorm2d, DIM_MULT, LATENTS_MEAN, LATENTS_STD, NUM_RES_BLOCKS,
};
use super::QWEN_IMAGE_21_LATENT_CHANNELS;

/// `base_dim` of the checkpoint's `vae/config.json` (the decoder has its own
/// `decoder_base_dim` of 144).
pub(crate) const ENCODER_BASE_DIM: usize = 96;
/// `temperal_downsample` (`V:1001`); the last down block never downsamples.
const TEMPORAL_DOWNSAMPLE: [bool; 4] = [false, true, true, true];
/// RGBA in (`V:1137`, `in_channels=4`).
pub(crate) const VAE_IMAGE_CHANNELS: usize = 4;
/// Pixels per latent cell.
const SPATIAL_FACTOR: usize = 16;

/// Tensors the encode phase reads from the VAE file.
pub(crate) fn is_encoder_tensor(name: &str) -> bool {
    name.starts_with("encoder.") || name.starts_with("quant_conv.")
}

/// `QwenImage21AvgDown3D.forward` (`V:57-89`) on a single frame.
///
/// The frame axis is padded at the FRONT to a multiple of `factor_t`, the
/// `(t, sh, sw)` sub-grid is folded into channels in `c, t, sh, sw` order,
/// and consecutive `group_size` entries are averaged. For `factor_t = 2` the
/// padded frame is zero, so output channel `2c` is exactly zero and `2c + 1`
/// is channel `c`'s `factor_s²` average. The mean is taken in F32, as
/// PyTorch accumulates a reduced-precision `mean`.
pub(crate) fn avg_down_shortcut(
    xs: &Tensor,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
) -> Result<Tensor> {
    let (batch, channels, height, width) = xs.dims4()?;
    let factor = factor_t * factor_s * factor_s;
    anyhow::ensure!(
        factor_t >= 1
            && factor_s >= 1
            && (channels * factor).is_multiple_of(out_channels)
            && height.is_multiple_of(factor_s)
            && width.is_multiple_of(factor_s),
        "Qwen Image 2.1 AvgDown3D cannot fold {channels}x{height}x{width} by t{factor_t}/s{factor_s} into {out_channels}"
    );
    let group = channels * factor / out_channels;
    let dtype = xs.dtype();
    let frame = xs.to_dtype(DType::F32)?.unsqueeze(2)?;
    // `pad_t = (factor_t - T % factor_t) % factor_t` zero frames in FRONT.
    let frames = if factor_t > 1 {
        let zeros = Tensor::zeros(
            (batch, channels, factor_t - 1, height, width),
            DType::F32,
            xs.device(),
        )?;
        Tensor::cat(&[&zeros, &frame], 2)?
    } else {
        frame
    };
    let (h, w) = (height / factor_s, width / factor_s);
    // [B, C, ft, H/fs, fs, W/fs, fs] -> [B, C, ft, fs, fs, H/fs, W/fs].
    // Candle's tuple permute stops at six axes, so express the 7-D move as
    // pairwise swaps (the decoder's DupUp3D does the same).
    let folded = frames
        .reshape(&[batch, channels, factor_t, h, factor_s, w, factor_s])?
        .transpose(3, 4)?
        .transpose(4, 6)?
        .transpose(5, 6)?
        .contiguous()?
        .reshape((batch, out_channels, group, h, w))?;
    Ok(folded.mean(2)?.to_dtype(dtype)?)
}

/// `QwenImage21ResidualDownBlock` (`V:489-525`).
struct ResidualDownBlock2d {
    resnets: Vec<ResidualBlock2d>,
    downsampler: Option<Conv2d>,
    out_dim: usize,
    factor_t: usize,
    factor_s: usize,
}

impl ResidualDownBlock2d {
    fn new(
        in_dim: usize,
        out_dim: usize,
        temporal_downsample: bool,
        down_flag: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(NUM_RES_BLOCKS);
        let mut current = in_dim;
        for index in 0..NUM_RES_BLOCKS {
            resnets.push(ResidualBlock2d::new(
                current,
                out_dim,
                vb.pp("resnets").pp(index),
            )?);
            current = out_dim;
        }
        // `ZeroPad2d((0, 1, 0, 1))` then a stride-2, unpadded 3x3 conv
        // (`V:266-270`); the pad is applied in `forward`.
        let downsampler = down_flag
            .then(|| {
                conv2d(
                    out_dim,
                    out_dim,
                    3,
                    Conv2dConfig {
                        stride: 2,
                        ..Default::default()
                    },
                    vb.pp("downsampler").pp("resample").pp("1"),
                )
            })
            .transpose()?;
        Ok(Self {
            resnets,
            downsampler,
            out_dim,
            factor_t: if temporal_downsample { 2 } else { 1 },
            factor_s: if down_flag { 2 } else { 1 },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = xs.clone();
        for resnet in &self.resnets {
            hidden = resnet.forward(&hidden)?;
        }
        if let Some(downsampler) = &self.downsampler {
            // Right and bottom only.
            let padded = hidden.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?;
            hidden = downsampler.forward(&padded)?;
        }
        let shortcut = avg_down_shortcut(xs, self.out_dim, self.factor_t, self.factor_s)?;
        Ok((hidden + shortcut)?)
    }
}

/// `QwenImage21Encoder3d` with `is_residual=True` (`V:527-647`).
struct Encoder2d {
    conv_in: Conv2d,
    down_blocks: Vec<ResidualDownBlock2d>,
    mid_block: MidBlock2d,
    norm_out: RmsNorm2d,
    conv_out: Conv2d,
}

impl Encoder2d {
    fn new(base_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let padded = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut dims = vec![base_dim];
        dims.extend(DIM_MULT.iter().map(|mult| base_dim * mult));
        let last = DIM_MULT.len() - 1;
        let mut down_blocks = Vec::with_capacity(DIM_MULT.len());
        for index in 0..DIM_MULT.len() {
            down_blocks.push(ResidualDownBlock2d::new(
                dims[index],
                dims[index + 1],
                index != last && TEMPORAL_DOWNSAMPLE[index],
                index != last,
                vb.pp("down_blocks").pp(index),
            )?);
        }
        let out_dim = dims[DIM_MULT.len()];
        Ok(Self {
            conv_in: conv2d(VAE_IMAGE_CHANNELS, dims[0], 3, padded, vb.pp("conv_in"))?,
            down_blocks,
            mid_block: MidBlock2d::new(out_dim, vb.pp("mid_block"))?,
            norm_out: RmsNorm2d::feature(out_dim, vb.pp("norm_out"))?,
            // `z_dim * 2`: the posterior mean and log-variance.
            conv_out: conv2d(
                out_dim,
                2 * QWEN_IMAGE_21_LATENT_CHANNELS,
                3,
                padded,
                vb.pp("conv_out"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = self.conv_in.forward(xs)?;
        for block in &self.down_blocks {
            hidden = block.forward(&hidden)?;
        }
        let hidden = self.mid_block.forward(&hidden)?;
        self.conv_out
            .forward(&candle_nn::Activation::Silu.forward(&self.norm_out.forward(&hidden)?)?)
            .map_err(Into::into)
    }
}

/// The encode half of the Qwen Image 2.1 VAE.
pub(crate) struct QwenImage21VaeEncoder {
    encoder: Encoder2d,
    quant_conv: Conv2d,
    latents_mean: Tensor,
    latents_std: Tensor,
}

impl QwenImage21VaeEncoder {
    /// Map only the encoder half of the VAE file.
    pub(crate) fn load(
        path: &Path,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let vb = crate::weight_loader::load_safetensors_with_filtered_progress(
            &[path],
            dtype,
            device,
            "Qwen Image 2.1 VAE encoder",
            progress,
            is_encoder_tensor,
        )?;
        Self::from_var_builder(ENCODER_BASE_DIM, vb, device, dtype)
    }

    pub(crate) fn from_var_builder(
        base_dim: usize,
        vb: VarBuilder<'_>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let channels = 2 * QWEN_IMAGE_21_LATENT_CHANNELS;
        let statistic = |values: &[f64]| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                (1, QWEN_IMAGE_21_LATENT_CHANNELS, 1, 1),
                device,
            )?
            .to_dtype(dtype)?)
        };
        Ok(Self {
            encoder: Encoder2d::new(base_dim, vb.pp("encoder"))?,
            quant_conv: conv2d(
                channels,
                channels,
                1,
                Default::default(),
                vb.pp("quant_conv"),
            )?,
            latents_mean: statistic(&LATENTS_MEAN)?,
            latents_std: statistic(&LATENTS_STD)?,
        })
    }

    /// The posterior MODE `[B, 64, H/16, W/16]` of RGBA samples in `[-1, 1]`,
    /// before normalization: `retrieve_latents(..., sample_mode="argmax")`
    /// takes the first `z_dim` channels (`pipeline_qwenimage21.py:140-141`).
    pub(crate) fn encode_mode(&self, rgba: &Tensor) -> Result<Tensor> {
        let (_, channels, height, width) = rgba.dims4()?;
        anyhow::ensure!(
            channels == VAE_IMAGE_CHANNELS,
            "Qwen Image 2.1 VAE encodes RGBA ({VAE_IMAGE_CHANNELS} channels), got {channels}"
        );
        anyhow::ensure!(
            height > 0
                && width > 0
                && height.is_multiple_of(SPATIAL_FACTOR)
                && width.is_multiple_of(SPATIAL_FACTOR),
            "Qwen Image 2.1 VAE input {width}x{height} is not a multiple of {SPATIAL_FACTOR}"
        );
        let moments = self.quant_conv.forward(&self.encoder.forward(rgba)?)?;
        Ok(moments.narrow(1, 0, QWEN_IMAGE_21_LATENT_CHANNELS)?)
    }

    /// Encode RGBA `[B, 4, H, W]` in `[-1, 1]` into packed, normalized
    /// transformer latents `[B, (H/16)(W/16), 64]`:
    /// `(z - latents_mean) / latents_std` (`pipeline_qwenimage21.py:423-446`)
    /// then a plain raster flatten (`:410-412`, no 2x2 patchify).
    pub(crate) fn encode_packed(&self, rgba: &Tensor) -> Result<Tensor> {
        let mode = self.encode_mode(rgba)?;
        let (batch, channels, height, width) = mode.dims4()?;
        let normalized = mode
            .broadcast_sub(&self.latents_mean.to_dtype(mode.dtype())?)?
            .broadcast_div(&self.latents_std.to_dtype(mode.dtype())?)?;
        Ok(normalized
            .reshape((batch, channels, height * width))?
            .transpose(1, 2)?
            .contiguous()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// U7: the temporal factor prepends a zero frame, so channel `2c` is zero
    /// and `2c + 1` is channel `c`'s 2x2 average.
    #[test]
    fn avg_down_zero_frame_channel_pattern() {
        let xs = Tensor::arange(0f32, 32.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 4, 4))
            .unwrap();
        let out = avg_down_shortcut(&xs, 4, 2, 2).unwrap();
        assert_eq!(out.dims(), &[1, 4, 2, 2]);
        let rows = out.squeeze(0).unwrap().to_vec3::<f32>().unwrap();
        assert_eq!(rows[0], vec![vec![0.0; 2]; 2]);
        assert_eq!(rows[2], vec![vec![0.0; 2]; 2]);
        // Channel 0 is 0..16 as a 4x4 grid: top-left 2x2 = {0, 1, 4, 5}.
        assert_eq!(rows[1], vec![vec![2.5, 4.5], vec![10.5, 12.5]]);
        assert_eq!(rows[3], vec![vec![18.5, 20.5], vec![26.5, 28.5]]);
    }

    #[test]
    fn avg_down_without_time_is_spatial_group_mean() {
        let xs = Tensor::arange(0f32, 32.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 4, 4))
            .unwrap();
        // factor 4, 2 -> 2 channels: one group per channel = 2x2 average.
        let out = avg_down_shortcut(&xs, 2, 1, 2).unwrap();
        let rows = out.squeeze(0).unwrap().to_vec3::<f32>().unwrap();
        assert_eq!(rows[0], vec![vec![2.5, 4.5], vec![10.5, 12.5]]);
        // The last block (no downsampling, equal widths) is the identity.
        assert_eq!(
            values(&avg_down_shortcut(&xs, 2, 1, 1).unwrap()),
            values(&xs)
        );
        // Widening a 2-channel input to 4 channels with factor 1 is refused.
        assert!(avg_down_shortcut(&xs, 3, 1, 1).is_err());
    }

    fn tiny_encoder(base_dim: usize) -> QwenImage21VaeEncoder {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder =
            QwenImage21VaeEncoder::from_var_builder(base_dim, vb, &device, DType::F32).unwrap();
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        for (seed, name) in names.into_iter().enumerate() {
            let var = &data[name];
            let random =
                crate::engine::seeded_randn(seed as u64, var.dims(), &device, DType::F32).unwrap();
            var.set(&(random * 0.05).unwrap()).unwrap();
        }
        drop(data);
        encoder
    }

    #[test]
    fn tiny_encoder_packs_normalized_mode_in_raster_order() {
        let encoder = tiny_encoder(4);
        let rgba = crate::engine::seeded_randn(3, &[1, 4, 64, 48], &Device::Cpu, DType::F32)
            .unwrap()
            .clamp(-1f32, 1f32)
            .unwrap();
        let mode = encoder.encode_mode(&rgba).unwrap();
        assert_eq!(mode.dims(), &[1, 64, 4, 3]);
        let packed = encoder.encode_packed(&rgba).unwrap();
        assert_eq!(packed.dims(), &[1, 12, 64]);
        // Token (row 2, column 1) channel 5 is `(mode - mean) / std`.
        let expected = (mode
            .get(0)
            .unwrap()
            .get(5)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[2][1]
            - LATENTS_MEAN[5] as f32)
            / LATENTS_STD[5] as f32;
        let actual = packed.get(0).unwrap().to_vec2::<f32>().unwrap()[2 * 3 + 1][5];
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        assert!(values(&packed).iter().all(|value| value.is_finite()));
    }

    #[test]
    fn encoder_input_contract() {
        let encoder = tiny_encoder(4);
        let rgb = Tensor::zeros((1, 3, 32, 32), DType::F32, &Device::Cpu).unwrap();
        assert!(encoder.encode_packed(&rgb).is_err());
        let odd = Tensor::zeros((1, 4, 40, 32), DType::F32, &Device::Cpu).unwrap();
        assert!(encoder.encode_packed(&odd).is_err());
        assert!(is_encoder_tensor("encoder.conv_in.weight"));
        assert!(is_encoder_tensor("quant_conv.bias"));
        assert!(!is_encoder_tensor("decoder.conv_in.weight"));
        assert!(!is_encoder_tensor("post_quant_conv.weight"));
    }

    /// Relative max error of `actual` against an upstream capture with the
    /// singleton frame axis (`[B, C, 1, H, W]`).
    fn relative_error(actual: &Tensor, expected: &Tensor) -> f32 {
        let expected = if expected.rank() == 5 {
            expected.squeeze(2).unwrap()
        } else {
            expected.clone()
        };
        let error = (actual.to_dtype(DType::F32).unwrap() - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let peak = expected
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        error / peak.max(f32::MIN_POSITIVE)
    }

    fn parity_inputs() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let root = std::env::var_os("QWEN_IMAGE21_MODEL_ROOT")?;
        let fixtures = std::env::var_os("QWEN_IMAGE21_FIXTURES")?;
        Some((
            std::path::PathBuf::from(root)
                .join("shared/qwen-image21/vae/diffusion_pytorch_model.safetensors"),
            std::path::PathBuf::from(fixtures),
        ))
    }

    /// P4 (stages): every encoder stage of a 64x96 RGBA crop — including each
    /// AvgDown3D shortcut, the zero-frame case among them — against the fp32
    /// upstream capture. CPU, F32.
    #[test]
    #[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
    fn p4_encoder_stages_match_the_upstream_capture() {
        let Some((vae_path, fixtures)) = parity_inputs() else {
            return;
        };
        let device = Device::Cpu;
        let progress = crate::progress::ProgressReporter::default();
        let encoder =
            QwenImage21VaeEncoder::load(&vae_path, &device, DType::F32, &progress).unwrap();
        let captured = candle_core::safetensors::load(
            fixtures.join("p4_vae_encoder_internals_64x96_fp32.safetensors"),
            &device,
        )
        .unwrap();
        let check = |name: &str, actual: &Tensor| {
            let error = relative_error(actual, &captured[name]);
            eprintln!("{name}: relative max error {error:.3e}");
            assert!(error < 1e-4, "{name}: {error}");
        };
        let input = captured["input"].squeeze(2).unwrap();
        let mut hidden = encoder.encoder.conv_in.forward(&input).unwrap();
        check("encoder.conv_in", &hidden);
        for (index, block) in encoder.encoder.down_blocks.iter().enumerate() {
            let shortcut =
                avg_down_shortcut(&hidden, block.out_dim, block.factor_t, block.factor_s).unwrap();
            check(
                &format!("encoder.down_blocks.{index}.avg_shortcut"),
                &shortcut,
            );
            hidden = block.forward(&hidden).unwrap();
            check(&format!("encoder.down_blocks.{index}"), &hidden);
        }
        let hidden = encoder.encoder.mid_block.forward(&hidden).unwrap();
        check("encoder.mid_block", &hidden);
        let hidden = encoder.encoder.norm_out.forward(&hidden).unwrap();
        check("encoder.norm_out", &hidden);
        let hidden = encoder
            .encoder
            .conv_out
            .forward(&candle_nn::Activation::Silu.forward(&hidden).unwrap())
            .unwrap();
        check("encoder.conv_out", &hidden);
        check("quant_conv", &encoder.quant_conv.forward(&hidden).unwrap());
        check("mode", &encoder.encode_mode(&input).unwrap());
    }

    /// P4 (full): both captured references at their resized canvases — the
    /// opaque 1248x832 and the transparent 928x1152 — through `encode_mode`
    /// and `encode_packed`. CPU, F32.
    #[test]
    #[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
    fn p4_encode_packed_matches_the_upstream_capture() {
        let Some((vae_path, fixtures)) = parity_inputs() else {
            return;
        };
        let device = Device::Cpu;
        let progress = crate::progress::ProgressReporter::default();
        let encoder =
            QwenImage21VaeEncoder::load(&vae_path, &device, DType::F32, &progress).unwrap();
        let captured = candle_core::safetensors::load(
            fixtures.join("p4_vae_encode_fp32.safetensors"),
            &device,
        )
        .unwrap();
        for case in ["opaque", "rgba"] {
            let input = captured[&format!("{case}_input")].squeeze(2).unwrap();
            let mode = encoder.encode_mode(&input).unwrap();
            let error = relative_error(&mode, &captured[&format!("{case}_mode")]);
            eprintln!("{case} mode: relative max error {error:.3e}");
            assert!(error < 1e-4, "{case} mode {error}");
            let packed = encoder.encode_packed(&input).unwrap();
            let error = relative_error(&packed, &captured[&format!("{case}_packed")]);
            eprintln!("{case} packed: relative max error {error:.3e}");
            assert!(error < 1e-4, "{case} packed {error}");
        }
    }
}
