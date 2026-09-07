//! Pure Rust U²-Net background matting for Hunyuan3D conditioning.
//!
//! Architecture: U-2-Net `model/u2net.py` at
//! `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`. Pre/post-processing follows
//! rembg `U2netSession` at `030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709`.
//! The ONNX export has batch normalization folded into its 119 convolutions;
//! graph order and tensor shapes are checked while constructing this port.

use std::path::Path;

use anyhow::{bail, ensure, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Module};
use image::{Rgb, RgbImage, Rgba, RgbaImage};

use crate::{
    identity::{
        onnx_graph::{load_installed_onnx_model, PinnedArtifact},
        onnx_weights::WeightTape,
    },
    pillow_resize::{self, Filter},
};

pub const U2NET_INPUT_SIZE: u32 = 320;
pub const U2NET_ONNX_BYTES: u64 = 175_997_641;
pub const U2NET_ONNX_SHA256: &str =
    "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491";
pub const MAX_MATTING_AXIS: u32 = 16_384;
pub const MAX_MATTING_PIXELS: u64 = 64 * 1024 * 1024;

/// `auto` should preserve alpha only when it describes both a visible subject
/// and a non-opaque background. Fully transparent placeholders and fully
/// opaque images still need a prediction.
pub fn has_useful_alpha(image: &RgbaImage) -> bool {
    let needed = (image.len() / 4 / 1_000).max(1);
    let mut transparent = 0;
    let mut visible = 0;
    for pixel in image.pixels() {
        transparent += usize::from(pixel[3] < 250);
        visible += usize::from(pixel[3] > 5);
        if transparent >= needed && visible >= needed {
            return true;
        }
    }
    false
}

#[derive(Debug)]
struct RebnConv(Conv2d);

impl RebnConv {
    fn load(
        tape: &mut WeightTape<'_>,
        input: usize,
        output: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self(tape.next_conv_dilated(output, input, 3, 1, dilation)?))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(x)?.relu()?)
    }
}

#[derive(Debug)]
struct Rsu {
    input: RebnConv,
    encoders: Vec<RebnConv>,
    bottom: RebnConv,
    decoders: Vec<RebnConv>,
    pooled: bool,
}

impl Rsu {
    fn load(
        tape: &mut WeightTape<'_>,
        depth: usize,
        input_channels: usize,
        mid_channels: usize,
        output_channels: usize,
    ) -> Result<Self> {
        ensure!((4..=7).contains(&depth), "RSU depth must be 4 through 7");
        let input = RebnConv::load(tape, input_channels, output_channels, 1)?;
        let mut encoders = Vec::with_capacity(depth - 1);
        encoders.push(RebnConv::load(tape, output_channels, mid_channels, 1)?);
        for _ in 1..depth - 1 {
            encoders.push(RebnConv::load(tape, mid_channels, mid_channels, 1)?);
        }
        let bottom = RebnConv::load(tape, mid_channels, mid_channels, 2)?;
        let mut decoders = Vec::with_capacity(depth - 1);
        for index in 0..depth - 1 {
            let output = if index + 1 == depth - 1 {
                output_channels
            } else {
                mid_channels
            };
            decoders.push(RebnConv::load(tape, mid_channels * 2, output, 1)?);
        }
        Ok(Self {
            input,
            encoders,
            bottom,
            decoders,
            pooled: true,
        })
    }

    fn load_flat(
        tape: &mut WeightTape<'_>,
        input_channels: usize,
        mid_channels: usize,
        output_channels: usize,
    ) -> Result<Self> {
        let input = RebnConv::load(tape, input_channels, output_channels, 1)?;
        let encoders = vec![
            RebnConv::load(tape, output_channels, mid_channels, 1)?,
            RebnConv::load(tape, mid_channels, mid_channels, 2)?,
            RebnConv::load(tape, mid_channels, mid_channels, 4)?,
        ];
        let bottom = RebnConv::load(tape, mid_channels, mid_channels, 8)?;
        let decoders = vec![
            RebnConv::load(tape, mid_channels * 2, mid_channels, 4)?,
            RebnConv::load(tape, mid_channels * 2, mid_channels, 2)?,
            RebnConv::load(tape, mid_channels * 2, output_channels, 1)?,
        ];
        Ok(Self {
            input,
            encoders,
            bottom,
            decoders,
            pooled: false,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = self.input.forward(x)?;
        let mut value = residual.clone();
        let mut skips = Vec::with_capacity(self.encoders.len());
        for (index, encoder) in self.encoders.iter().enumerate() {
            value = encoder.forward(&value)?;
            skips.push(value.clone());
            if self.pooled && index + 1 < self.encoders.len() {
                value = ceil_max_pool_2x2(&value)?;
            }
        }
        value = self.bottom.forward(&value)?;
        for (index, (decoder, skip)) in self.decoders.iter().zip(skips.iter().rev()).enumerate() {
            value = decoder.forward(&Tensor::cat(&[&value, skip], 1)?)?;
            if self.pooled && index + 1 < self.decoders.len() {
                // Resize before the next decoder, exactly as `_upsample_like`.
                value = upsample_like(&value, &skips[skips.len() - 2 - index])?;
            }
        }
        Ok((value + residual)?)
    }
}

fn ceil_max_pool_2x2(x: &Tensor) -> Result<Tensor> {
    let (_, _, height, width) = x.dims4()?;
    let x = if height % 2 == 1 {
        x.pad_with_zeros(2, 0, 1)?
    } else {
        x.clone()
    };
    let x = if width % 2 == 1 {
        x.pad_with_zeros(3, 0, 1)?
    } else {
        x
    };
    Ok(x.max_pool2d_with_stride(2, 2)?)
}

fn upsample_like(source: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (_, _, height, width) = target.dims4()?;
    Ok(source.upsample_bilinear2d(height, width, false)?)
}

#[derive(Debug)]
pub struct U2Net {
    stages: Vec<Rsu>,
    sides: Vec<Conv2d>,
    output: Conv2d,
    device: Device,
}

impl U2Net {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let loaded = load_installed_onnx_model(
            path,
            Some(PinnedArtifact {
                size_bytes: U2NET_ONNX_BYTES,
                sha256: U2NET_ONNX_SHA256.into(),
            }),
        )?;
        let mut tape = WeightTape::new(&loaded.model, device)?;
        let stages = vec![
            Rsu::load(&mut tape, 7, 3, 32, 64)?,
            Rsu::load(&mut tape, 6, 64, 32, 128)?,
            Rsu::load(&mut tape, 5, 128, 64, 256)?,
            Rsu::load(&mut tape, 4, 256, 128, 512)?,
            Rsu::load_flat(&mut tape, 512, 256, 512)?,
            Rsu::load_flat(&mut tape, 512, 256, 512)?,
            Rsu::load_flat(&mut tape, 1024, 256, 512)?,
            Rsu::load(&mut tape, 4, 1024, 128, 256)?,
            Rsu::load(&mut tape, 5, 512, 64, 128)?,
            Rsu::load(&mut tape, 6, 256, 32, 64)?,
            Rsu::load(&mut tape, 7, 128, 16, 64)?,
        ];
        let sides = vec![
            tape.next_conv(1, 64, 3, 1)?,
            tape.next_conv(1, 64, 3, 1)?,
            tape.next_conv(1, 128, 3, 1)?,
            tape.next_conv(1, 256, 3, 1)?,
            tape.next_conv(1, 512, 3, 1)?,
            tape.next_conv(1, 512, 3, 1)?,
        ];
        let output = tape.next_conv(1, 6, 1, 1)?;
        tape.finish()?;
        Ok(Self {
            stages,
            sides,
            output,
            device: device.clone(),
        })
    }

    pub fn predict_320(&self, input: &Tensor) -> Result<Tensor> {
        ensure!(
            input.dims() == [1, 3, 320, 320],
            "U²-Net input must be [1, 3, 320, 320]"
        );
        let h1 = self.stages[0].forward(input)?;
        let h2 = self.stages[1].forward(&ceil_max_pool_2x2(&h1)?)?;
        let h3 = self.stages[2].forward(&ceil_max_pool_2x2(&h2)?)?;
        let h4 = self.stages[3].forward(&ceil_max_pool_2x2(&h3)?)?;
        let h5 = self.stages[4].forward(&ceil_max_pool_2x2(&h4)?)?;
        let h6 = self.stages[5].forward(&ceil_max_pool_2x2(&h5)?)?;
        let h5d = self.stages[6].forward(&Tensor::cat(&[&upsample_like(&h6, &h5)?, &h5], 1)?)?;
        let h4d = self.stages[7].forward(&Tensor::cat(&[&upsample_like(&h5d, &h4)?, &h4], 1)?)?;
        let h3d = self.stages[8].forward(&Tensor::cat(&[&upsample_like(&h4d, &h3)?, &h3], 1)?)?;
        let h2d = self.stages[9].forward(&Tensor::cat(&[&upsample_like(&h3d, &h2)?, &h2], 1)?)?;
        let h1d = self.stages[10].forward(&Tensor::cat(&[&upsample_like(&h2d, &h1)?, &h1], 1)?)?;
        let features = [&h1d, &h2d, &h3d, &h4d, &h5d, &h6];
        let mut sides = Vec::with_capacity(6);
        for (conv, feature) in self.sides.iter().zip(features) {
            let side = conv.forward(feature)?;
            sides.push(upsample_like(&side, &h1d)?);
        }
        let refs = sides.iter().collect::<Vec<_>>();
        Ok(candle_nn::ops::sigmoid(
            &self.output.forward(&Tensor::cat(&refs, 1)?)?,
        )?)
    }

    pub fn matte(&self, image: &RgbaImage) -> Result<RgbaImage> {
        validate_dimensions(image)?;
        let rgb = RgbImage::from_fn(image.width(), image.height(), |x, y| {
            let p = image.get_pixel(x, y);
            Rgb([p[0], p[1], p[2]])
        });
        let resized = pillow_resize::resize(
            &rgb,
            U2NET_INPUT_SIZE,
            U2NET_INPUT_SIZE,
            Filter::Lanczos,
            &mut || Ok(()),
        )?;
        let max = resized.as_raw().iter().copied().max().unwrap_or(0).max(1) as f32;
        let means = [0.485_f32, 0.456, 0.406];
        let stds = [0.229_f32, 0.224, 0.225];
        let mut input = vec![0_f32; 3 * 320 * 320];
        for (index, pixel) in resized.pixels().enumerate() {
            for channel in 0..3 {
                input[channel * 320 * 320 + index] =
                    (f32::from(pixel[channel]) / max - means[channel]) / stds[channel];
            }
        }
        let input =
            Tensor::from_vec(input, (1, 3, 320, 320), &self.device)?.to_dtype(DType::F32)?;
        let raw = self.predict_320(&input)?.to_device(&Device::Cpu)?;
        let raw = raw.flatten_all()?.to_vec1::<f32>()?;
        let min = raw.iter().copied().fold(f32::INFINITY, f32::min);
        let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        ensure!(
            min.is_finite() && max.is_finite(),
            "U²-Net produced a non-finite mask"
        );
        let range = max - min;
        let mask = if range <= f32::EPSILON {
            vec![0_u8; 320 * 320]
        } else {
            raw.into_iter()
                .map(|value| (((value - min) / range).clamp(0.0, 1.0) * 255.0) as u8)
                .collect()
        };
        let mask = RgbImage::from_fn(320, 320, |x, y| {
            let alpha = mask[(y * 320 + x) as usize];
            Rgb([alpha, alpha, alpha])
        });
        let mask = pillow_resize::resize(
            &mask,
            image.width(),
            image.height(),
            Filter::Lanczos,
            &mut || Ok(()),
        )?;
        Ok(RgbaImage::from_fn(image.width(), image.height(), |x, y| {
            let source = image.get_pixel(x, y);
            Rgba([source[0], source[1], source[2], mask.get_pixel(x, y)[0]])
        }))
    }
}

fn validate_dimensions(image: &RgbaImage) -> Result<()> {
    if image.width() == 0 || image.height() == 0 {
        bail!("matting input dimensions must be non-zero");
    }
    ensure!(
        image.width() <= MAX_MATTING_AXIS && image.height() <= MAX_MATTING_AXIS,
        "matting input dimensions exceed {MAX_MATTING_AXIS}"
    );
    ensure!(
        u64::from(image.width()) * u64::from(image.height()) <= MAX_MATTING_PIXELS,
        "matting input exceeds {MAX_MATTING_PIXELS} pixels"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_alpha_requires_visible_and_nonopaque_pixels() {
        assert!(!has_useful_alpha(&RgbaImage::from_pixel(
            2,
            2,
            Rgba([1, 2, 3, 255])
        )));
        assert!(!has_useful_alpha(&RgbaImage::from_pixel(
            2,
            2,
            Rgba([1, 2, 3, 0])
        )));
        let mut cutout = RgbaImage::from_pixel(2, 2, Rgba([1, 2, 3, 255]));
        cutout.put_pixel(0, 0, Rgba([1, 2, 3, 0]));
        assert!(has_useful_alpha(&cutout));

        let mut noisy = RgbaImage::from_pixel(100, 100, Rgba([1, 2, 3, 255]));
        noisy.put_pixel(0, 0, Rgba([1, 2, 3, 0]));
        assert!(!has_useful_alpha(&noisy));
    }

    #[test]
    fn matting_dimensions_are_bounded_before_allocation() {
        assert!(validate_dimensions(&RgbaImage::new(1, 1)).is_ok());
        assert!(validate_dimensions(&RgbaImage::new(MAX_MATTING_AXIS + 1, 1)).is_err());
        assert!(validate_dimensions(&RgbaImage::new(8193, 8193)).is_err());
    }
}
