//! SRVGGNetCompact — Compact Real-ESRGAN architecture.
//!
//! A simple linear chain of Conv2d + PReLU layers ending with pixel_shuffle.
//! Used by realesr-general-x4v3 and realesr-animevideov3 models.
//!
//! Architecture:
//! ```text
//! Conv2d(in_ch, nf, 3, pad=1)
//! [PReLU(nf) -> Conv2d(nf, nf, 3, pad=1)] x num_conv
//! PReLU(nf)
//! Conv2d(nf, out_ch * scale^2, 3, pad=1)
//! pixel_shuffle(scale)
//! ```

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, PReLU, VarBuilder};

/// A single Conv2d + PReLU block.
struct ConvPReLU {
    conv: Conv2d,
    prelu: PReLU,
}

impl ConvPReLU {
    fn new(conv: Conv2d, prelu: PReLU) -> Self {
        Self { conv, prelu }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        Ok(self.prelu.forward(&xs)?)
    }
}

pub struct SRVGGNetCompact {
    body_first: Conv2d,
    body_blocks: Vec<ConvPReLU>,
    body_last_prelu: PReLU,
    body_last_conv: Conv2d,
    scale: u32,
}

impl SRVGGNetCompact {
    /// Load from safetensors VarBuilder.
    ///
    /// Infers architecture parameters from the available weights:
    /// - num_feat from body.0.weight shape
    /// - num_conv from body layer count
    /// - scale from last conv output channels
    pub fn load(vb: &VarBuilder, num_feat: usize, num_conv: usize, scale: u32) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };

        // Body index layout:
        // 0: first conv (3 -> num_feat)
        // 1: first prelu
        // 2: conv, 3: prelu, 4: conv, 5: prelu, ...
        // last-1: final prelu
        // last: final conv (num_feat -> out_ch * scale^2)

        let body_first = candle_nn::conv2d(3, num_feat, 3, cfg, vb.pp("body.0"))?;

        let mut body_blocks = Vec::with_capacity(num_conv);
        for i in 0..num_conv {
            let conv_idx = 2 + i * 2;
            let prelu_idx = conv_idx - 1;
            let prelu = candle_nn::prelu(Some(num_feat), vb.pp(format!("body.{prelu_idx}")))?;
            let conv = candle_nn::conv2d(
                num_feat,
                num_feat,
                3,
                cfg,
                vb.pp(format!("body.{conv_idx}")),
            )?;
            body_blocks.push(ConvPReLU::new(conv, prelu));
        }

        let last_prelu_idx = 2 + num_conv * 2 - 1;
        let last_conv_idx = last_prelu_idx + 1;

        let body_last_prelu =
            candle_nn::prelu(Some(num_feat), vb.pp(format!("body.{last_prelu_idx}")))?;

        let out_channels = 3 * (scale as usize) * (scale as usize);
        let body_last_conv = candle_nn::conv2d(
            num_feat,
            out_channels,
            3,
            cfg,
            vb.pp(format!("body.{last_conv_idx}")),
        )?;

        Ok(Self {
            body_first,
            body_blocks,
            body_last_prelu,
            body_last_conv,
            scale,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out = self.body_first.forward(xs)?;
        for block in &self.body_blocks {
            out = block.forward(&out)?;
        }
        out = self.body_last_prelu.forward(&out)?;
        out = self.body_last_conv.forward(&out)?;
        out = candle_nn::ops::pixel_shuffle(&out, self.scale as usize)?;

        // Residual skip: add nearest-neighbor upsampled input to the learned branch.
        // This is critical — without it the output is dark/distorted.
        let (_, _, h, w) = xs.dims4()?;
        let upsampled = xs.upsample_nearest2d(h * self.scale as usize, w * self.scale as usize)?;
        let out = (out + upsampled)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn srvggnet_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let num_feat = 4;
        let num_conv = 2;
        let scale = 4u32;

        // Initialize all expected weights
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };

        // body.0: Conv2d(3, 4)
        let _ = candle_nn::conv2d(3, num_feat, 3, cfg, vb.pp("body.0")).unwrap();
        // body.1: PReLU(4), body.2: Conv2d(4,4)
        let _ = candle_nn::prelu(Some(num_feat), vb.pp("body.1")).unwrap();
        let _ = candle_nn::conv2d(num_feat, num_feat, 3, cfg, vb.pp("body.2")).unwrap();
        // body.3: PReLU(4), body.4: Conv2d(4,4)
        let _ = candle_nn::prelu(Some(num_feat), vb.pp("body.3")).unwrap();
        let _ = candle_nn::conv2d(num_feat, num_feat, 3, cfg, vb.pp("body.4")).unwrap();
        // body.5: PReLU(4), body.6: Conv2d(4, 3*16=48)
        let _ = candle_nn::prelu(Some(num_feat), vb.pp("body.5")).unwrap();
        let out_ch = 3 * (scale as usize) * (scale as usize);
        let _ = candle_nn::conv2d(num_feat, out_ch, 3, cfg, vb.pp("body.6")).unwrap();

        let vb2 = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = SRVGGNetCompact::load(&vb2, num_feat, num_conv, scale).unwrap();

        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), &device).unwrap();
        let output = model.forward(&input).unwrap();
        let dims = output.dims4().unwrap();
        assert_eq!(dims, (1, 3, 32, 32)); // 8*4 = 32
    }
}
