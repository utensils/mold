//! RRDBNet — Real-ESRGAN full architecture (Residual-in-Residual Dense Block Network).
//!
//! Used by RealESRGAN_x4plus, x2plus, and x4plus_anime_6B models.
//!
//! Architecture:
//! ```text
//! ResidualDenseBlock: 5x [Conv2d(in, gc, 3, pad=1) + LeakyReLU(0.2)] with dense connections
//! RRDB: 3x ResidualDenseBlock + residual scaling (0.2)
//! RRDBNet:
//!   conv_first(3, nf, 3, pad=1)
//!   body: N x RRDB + conv_body(nf, nf, 3, pad=1)
//!   conv_up1(nf, nf, 3, pad=1) after upsample_nearest2d(2x)
//!   [conv_up2(nf, nf, 3, pad=1) after upsample_nearest2d(2x)]  -- 4x only
//!   conv_hr(nf, nf, 3, pad=1) + LeakyReLU(0.2)
//!   conv_last(nf, 3, 3, pad=1)
//! ```

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

const LRELU_SLOPE: f64 = 0.2;
const RESIDUAL_SCALE: f64 = 0.2;

fn leaky_relu(xs: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::Activation::LeakyRelu(LRELU_SLOPE).forward(xs)?)
}

fn conv_cfg() -> Conv2dConfig {
    Conv2dConfig {
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
        ..Default::default()
    }
}

/// Residual Dense Block: 5 convolutions with dense (concatenation) connections.
struct ResidualDenseBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
}

impl ResidualDenseBlock {
    fn load(nf: usize, gc: usize, vb: &VarBuilder) -> Result<Self> {
        let cfg = conv_cfg();
        Ok(Self {
            conv1: candle_nn::conv2d(nf, gc, 3, cfg, vb.pp("conv1"))?,
            conv2: candle_nn::conv2d(nf + gc, gc, 3, cfg, vb.pp("conv2"))?,
            conv3: candle_nn::conv2d(nf + 2 * gc, gc, 3, cfg, vb.pp("conv3"))?,
            conv4: candle_nn::conv2d(nf + 3 * gc, gc, 3, cfg, vb.pp("conv4"))?,
            conv5: candle_nn::conv2d(nf + 4 * gc, nf, 3, cfg, vb.pp("conv5"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x1 = leaky_relu(&self.conv1.forward(xs)?)?;
        let x2 = leaky_relu(&self.conv2.forward(&Tensor::cat(&[xs, &x1], 1)?)?)?;
        let x3 = leaky_relu(&self.conv3.forward(&Tensor::cat(&[xs, &x1, &x2], 1)?)?)?;
        let x4 = leaky_relu(&self.conv4.forward(&Tensor::cat(&[xs, &x1, &x2, &x3], 1)?)?)?;
        let x5 = self
            .conv5
            .forward(&Tensor::cat(&[xs, &x1, &x2, &x3, &x4], 1)?)?;
        // Residual scaling
        let scaled = (x5 * RESIDUAL_SCALE)?;
        Ok((&scaled + xs)?)
    }
}

/// RRDB: 3 Residual Dense Blocks with residual scaling.
#[allow(clippy::upper_case_acronyms)]
struct RRDB {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl RRDB {
    fn load(nf: usize, gc: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            rdb1: ResidualDenseBlock::load(nf, gc, &vb.pp("rdb1"))?,
            rdb2: ResidualDenseBlock::load(nf, gc, &vb.pp("rdb2"))?,
            rdb3: ResidualDenseBlock::load(nf, gc, &vb.pp("rdb3"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.rdb1.forward(xs)?;
        let out = self.rdb2.forward(&out)?;
        let out = self.rdb3.forward(&out)?;
        let scaled = (out * RESIDUAL_SCALE)?;
        Ok((&scaled + xs)?)
    }
}

/// Full RRDBNet architecture.
pub struct RRDBNet {
    conv_first: Conv2d,
    body: Vec<RRDB>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Option<Conv2d>,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    scale: u32,
}

impl RRDBNet {
    pub fn load(
        vb: &VarBuilder,
        num_feat: usize,
        num_grow_ch: usize,
        num_block: usize,
        scale: u32,
    ) -> Result<Self> {
        let cfg = conv_cfg();

        let conv_first = candle_nn::conv2d(3, num_feat, 3, cfg, vb.pp("conv_first"))?;

        let mut body = Vec::with_capacity(num_block);
        for i in 0..num_block {
            body.push(RRDB::load(
                num_feat,
                num_grow_ch,
                &vb.pp(format!("body.{i}")),
            )?);
        }
        // conv_body may be "conv_body" (hlky/diffusers format) or
        // "body.{num_block}" (original Real-ESRGAN format).
        let conv_body =
            candle_nn::conv2d(num_feat, num_feat, 3, cfg, vb.pp("conv_body")).or_else(|_| {
                candle_nn::conv2d(
                    num_feat,
                    num_feat,
                    3,
                    cfg,
                    vb.pp(format!("body.{num_block}")),
                )
            })?;

        let conv_up1 = candle_nn::conv2d(num_feat, num_feat, 3, cfg, vb.pp("conv_up1"))?;
        let conv_up2 = if scale >= 4 {
            Some(candle_nn::conv2d(
                num_feat,
                num_feat,
                3,
                cfg,
                vb.pp("conv_up2"),
            )?)
        } else {
            None
        };
        let conv_hr = candle_nn::conv2d(num_feat, num_feat, 3, cfg, vb.pp("conv_hr"))?;
        let conv_last = candle_nn::conv2d(num_feat, 3, 3, cfg, vb.pp("conv_last"))?;

        Ok(Self {
            conv_first,
            body,
            conv_body,
            conv_up1,
            conv_up2,
            conv_hr,
            conv_last,
            scale,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let feat = self.conv_first.forward(xs)?;
        let mut body_feat = feat.clone();
        for rrdb in &self.body {
            body_feat = rrdb.forward(&body_feat)?;
        }
        body_feat = self.conv_body.forward(&body_feat)?;
        let feat = (feat + body_feat)?;

        // Upsample
        let (_, _, h, w) = feat.dims4()?;
        let feat = feat.upsample_nearest2d(h * 2, w * 2)?;
        let feat = leaky_relu(&self.conv_up1.forward(&feat)?)?;

        let feat = if let Some(ref conv_up2) = self.conv_up2 {
            let (_, _, h2, w2) = feat.dims4()?;
            let feat = feat.upsample_nearest2d(h2 * 2, w2 * 2)?;
            leaky_relu(&conv_up2.forward(&feat)?)?
        } else {
            feat
        };

        let out = leaky_relu(&self.conv_hr.forward(&feat)?)?;
        let out = self.conv_last.forward(&out)?;
        Ok(out)
    }

    #[allow(dead_code)]
    pub fn scale(&self) -> u32 {
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn build_test_rrdbnet(
        num_feat: usize,
        num_grow_ch: usize,
        num_block: usize,
        scale: u32,
    ) -> (VarMap, RRDBNet) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Initialize all weights by constructing the model
        let model = RRDBNet::load(&vb, num_feat, num_grow_ch, num_block, scale).unwrap();
        (varmap, model)
    }

    #[test]
    fn rrdbnet_x4_output_shape() {
        let (_varmap, model) = build_test_rrdbnet(8, 4, 1, 4);
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), &device).unwrap();
        let output = model.forward(&input).unwrap();
        let dims = output.dims4().unwrap();
        assert_eq!(dims, (1, 3, 32, 32)); // 8*4 = 32
    }

    #[test]
    fn rrdbnet_x2_output_shape() {
        let (_varmap, model) = build_test_rrdbnet(8, 4, 1, 2);
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 16, 16), &device).unwrap();
        let output = model.forward(&input).unwrap();
        let dims = output.dims4().unwrap();
        assert_eq!(dims, (1, 3, 32, 32)); // 16*2 = 32
    }
}
