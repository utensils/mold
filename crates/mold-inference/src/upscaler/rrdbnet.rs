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
use candle_core::{DType, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

const LRELU_SLOPE: f64 = 0.2;
const RESIDUAL_SCALE: f64 = 0.2;

fn leaky_relu(xs: &Tensor) -> Result<Tensor> {
    // Torch 2.5.1 ActivationLeakyReluKernel.cu:29-33 uses opmath_t:
    // the slope stays F32 even when the input/output are half. Candle's
    // half affine kernel instead rounds the scalar before multiplication.
    if xs.dtype() == DType::F16 {
        return Ok(candle_nn::Activation::LeakyRelu(LRELU_SLOPE)
            .forward(&xs.to_dtype(DType::F32)?)?
            .to_dtype(DType::F16)?);
    }
    Ok(candle_nn::Activation::LeakyRelu(LRELU_SLOPE).forward(xs)?)
}

fn residual_scale(xs: &Tensor) -> Result<Tensor> {
    // BasicSR v1.4.2 rrdbnet_arch.py:39,63: multiply, round to the
    // activation dtype, THEN add the residual (not a fused multiply-add).
    if xs.dtype() == DType::F16 {
        return Ok((xs.to_dtype(DType::F32)? * RESIDUAL_SCALE)?.to_dtype(DType::F16)?);
    }
    Ok((xs * RESIDUAL_SCALE)?)
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
        let scaled = residual_scale(&x5)?;
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
        let scaled = residual_scale(&out)?;
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

        // Tencent's first convolution also uses cuDNN. Its small RGB im2col
        // buffer falls below Candle's automatic performance threshold, but
        // those early rounding differences amplify through all 23 RRDBs.
        // An explicit algorithm bypasses that heuristic only when the caller
        // has enabled cuDNN; the normal im2col policy remains authoritative.
        let first_cfg = Conv2dConfig {
            cudnn_fwd_algo: Some(candle_core::conv::CudnnFwdAlgo::ImplicitPrecompGemm),
            ..cfg
        };
        let conv_first = candle_nn::conv2d(3, num_feat, 3, first_cfg, vb.pp("conv_first"))?;

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

    #[cfg(test)]
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_observer(xs, |_, _| Ok(()))
    }

    pub fn forward_with_checkpoint(
        &self,
        xs: &Tensor,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<Tensor> {
        checkpoint()?;
        self.forward_with_observer(xs, |_, _| checkpoint())
    }

    fn forward_with_observer(
        &self,
        xs: &Tensor,
        mut observe: impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let feat = self.conv_first.forward(xs)?;
        observe("conv_first", &feat)?;
        let mut body_feat = feat.clone();
        for (index, rrdb) in self.body.iter().enumerate() {
            body_feat = rrdb.forward(&body_feat)?;
            observe(&format!("body.{index}"), &body_feat)?;
        }
        body_feat = self.conv_body.forward(&body_feat)?;
        observe("conv_body", &body_feat)?;
        let feat = (feat + body_feat)?;

        // Upsample
        let (_, _, h, w) = feat.dims4()?;
        let feat = feat.upsample_nearest2d(h * 2, w * 2)?;
        let feat = self.conv_up1.forward(&feat)?;
        observe("conv_up1", &feat)?;
        let feat = leaky_relu(&feat)?;

        let feat = if let Some(ref conv_up2) = self.conv_up2 {
            let (_, _, h2, w2) = feat.dims4()?;
            let feat = feat.upsample_nearest2d(h2 * 2, w2 * 2)?;
            let feat = conv_up2.forward(&feat)?;
            observe("conv_up2", &feat)?;
            leaky_relu(&feat)?
        } else {
            feat
        };

        let out = self.conv_hr.forward(&feat)?;
        observe("conv_hr", &out)?;
        let out = leaky_relu(&out)?;
        let out = self.conv_last.forward(&out)?;
        observe("conv_last", &out)?;
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

    fn require_unmodified_oracle(metadata: &serde_json::Value) -> Result<()> {
        anyhow::ensure!(
            metadata.get("diagnostic_first_features").is_none()
                && !metadata["argv"]
                    .as_array()
                    .is_some_and(|args| args.iter().any(|arg| arg
                        .as_str()
                        .is_some_and(|value| value.starts_with("--diagnostic-")))),
            "diagnostic tensor substitutions cannot qualify the upscaler"
        );
        Ok(())
    }

    #[test]
    fn substituted_oracles_cannot_qualify_network_parity() {
        assert!(require_unmodified_oracle(&serde_json::json!({"argv": ["capture.py"]})).is_ok());
        assert!(
            require_unmodified_oracle(&serde_json::json!({"diagnostic_first_features": {}}))
                .is_err()
        );
        assert!(require_unmodified_oracle(&serde_json::json!({"argv": ["capture.py", "--diagnostic-first-features", "value.safetensors"]})).is_err());
    }

    #[test]
    fn rrdb_cancellation_at_every_boundary_preserves_reuse() -> Result<()> {
        let (_weights, model) = build_test_rrdbnet(4, 2, 2, 4);
        let input = Tensor::randn(0f32, 0.1, (1, 3, 2, 2), &Device::Cpu)?;
        let expected = model.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        let mut boundaries = 0;
        let actual = model.forward_with_checkpoint(&input, || {
            boundaries += 1;
            Ok(())
        })?;
        assert_eq!(actual.flatten_all()?.to_vec1::<f32>()?, expected);
        // Entry + first convolution + two RRDBs + body/up1/up2/hr/last.
        assert_eq!(boundaries, 9);
        for stop in 1..=boundaries {
            let token = crate::progress::InferenceCancellationToken::default();
            let mut progress = crate::progress::ProgressReporter::default();
            progress.set_cancellation_token(token.clone());
            let mut visited = 0;
            let error = model
                .forward_with_checkpoint(&input, || {
                    visited += 1;
                    if visited == stop {
                        token.cancel();
                    }
                    Ok(progress.checkpoint()?)
                })
                .unwrap_err();
            assert!(crate::progress::is_inference_cancelled(&error));
            assert_eq!(visited, stop);
            assert_eq!(
                model.forward(&input)?.flatten_all()?.to_vec1::<f32>()?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires CUDA, installed RRDB weights and retained Tencent oracle"]
    fn pretrained_paint_upscaler_matches_tencent() -> Result<()> {
        use std::{collections::HashMap, path::PathBuf};
        let fixture = PathBuf::from(std::env::var("MOLD_PAINT_UPSCALER_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_PAINT_UPSCALER_OUTPUT")?);
        let weights = PathBuf::from(std::env::var("MOLD_PAINT_UPSCALER_WEIGHTS")?);
        require_unmodified_oracle(&serde_json::from_slice(&std::fs::read(
            fixture.join("completed.json"),
        )?)?)?;
        std::fs::create_dir(&output)?;
        let _scope = crate::conv_policy::ConvScope::for_family("hunyuan3d");
        let device = Device::new_cuda(0)?;
        let oracle =
            candle_core::safetensors::load(fixture.join("stages.safetensors"), &Device::Cpu)?;
        let required = [
            "conv_first",
            "body.0",
            "body.11",
            "body.22",
            "conv_body",
            "conv_up1",
            "conv_up2",
            "conv_hr",
            "conv_last",
        ];
        for name in std::iter::once("input").chain(required) {
            anyhow::ensure!(oracle.contains_key(name), "missing oracle stage: {name}");
            anyhow::ensure!(
                oracle[name].dtype() == DType::F16,
                "oracle stage must be half: {name}"
            );
        }
        // SAFETY: the installed checkpoint is retained, immutable model storage.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F16, &device)? };
        let model = RRDBNet::load(&vb, 64, 32, 23, 4)?;
        let input = oracle["input"].to_device(&device)?;
        let mut comparisons = serde_json::Map::new();
        let mut failed = Vec::new();
        let start = std::time::Instant::now();
        let dispatch_before = candle_core::cudnn_policy::dispatch_count();
        let result = model.forward_with_observer(&input, |name, actual| {
            if name == "conv_first" && candle_core::cudnn_policy::is_enabled() {
                let dispatched = candle_core::cudnn_policy::dispatch_count() - dispatch_before;
                anyhow::ensure!(
                    dispatched == 1,
                    "first convolution did not execute on cuDNN"
                );
                eprintln!("first convolution cuDNN dispatches={dispatched}");
            }
            let Some(expected) = oracle.get(name) else {
                return Ok(());
            };
            anyhow::ensure!(actual.dims() == expected.dims(), "stage shape: {name}");
            let actual = actual.to_device(&Device::Cpu)?;
            candle_core::safetensors::save(
                &HashMap::from([("value".to_string(), actual.clone())]),
                output.join(format!("{name}.safetensors")),
            )?;
            let a = actual
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let b = expected
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut max = 0.0f64;
            let mut sum = 0.0f64;
            let mut nonfinite = 0usize;
            for (a, b) in a.iter().zip(&b) {
                let delta = f64::from(*a) - f64::from(*b);
                if !delta.is_finite() {
                    nonfinite += 1;
                }
                max = max.max(delta.abs());
                sum += delta * delta;
            }
            let rms = (sum / a.len() as f64).sqrt();
            eprintln!("{name}: max={max}, rms={rms}, nonfinite={nonfinite}");
            comparisons.insert(
                name.to_string(),
                serde_json::json!({"max":max,"rms":rms,"nonfinite":nonfinite}),
            );
            if max > 0.01 || nonfinite != 0 {
                failed.push(name.to_string());
            }
            Ok(())
        })?;
        anyhow::ensure!(
            comparisons.len() == required.len(),
            "not all oracle stages were compared"
        );
        let seconds = start.elapsed().as_secs_f64();
        let (_, _, height, width) = result.dims4()?;
        let values = result
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut rgb = Vec::with_capacity(values.len());
        for pixel in values.chunks_exact(3) {
            for channel in pixel.iter().rev() {
                rgb.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        let image = image::RgbImage::from_raw(width as u32, height as u32, rgb).unwrap();
        image.save(output.join("actual.png"))?;
        let expected = image::open(fixture.join("expected.png"))?.to_rgb8();
        anyhow::ensure!(
            image.dimensions() == expected.dimensions(),
            "image dimensions differ"
        );
        let max_byte = image
            .as_raw()
            .iter()
            .zip(expected.as_raw())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        comparisons.insert(
            "image".into(),
            serde_json::json!({"max_byte":max_byte,"seconds_with_observation":seconds}),
        );
        std::fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(&comparisons)?,
        )?;
        anyhow::ensure!(
            failed.is_empty() && max_byte <= 8,
            "upscaler parity failures: {failed:?}; image max byte={max_byte}"
        );
        Ok(())
    }

    #[test]
    #[allow(clippy::excessive_precision)] // Exact dyadic values from the half oracle.
    fn half_residual_and_activation_keep_float_scalar_precision() -> Result<()> {
        // Values selected from the exhaustive Torch CUDA scalar capture.
        let input = Tensor::new(&[-2.00390625f32, 3.140625], &Device::Cpu)?.to_dtype(DType::F16)?;
        assert_eq!(
            residual_scale(&input)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
            [-0.40087890625, 0.6279296875]
        );
        assert_eq!(
            leaky_relu(&input)?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            [-0.40087890625, 3.140625]
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires CUDA and retained Tencent upscaler scalar oracle"]
    fn half_scalar_operations_match_torch() -> Result<()> {
        let path = std::env::var("MOLD_PAINT_UPSCALER_SCALARS")?;
        let device = Device::new_cuda(0)?;
        let oracle = candle_core::safetensors::load(path, &device)?;
        let input = &oracle["input"];
        let mut failures = Vec::new();
        for (name, actual) in [
            ("scaled", residual_scale(input)?),
            ("leaky_relu", leaky_relu(input)?),
        ] {
            let actual = actual.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let expected = oracle[name].to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let count = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            eprintln!(
                "{name}: {count}/{} different finite half results",
                actual.len()
            );
            if count != 0 {
                failures.push((name, count));
            }
        }
        anyhow::ensure!(
            failures.is_empty(),
            "Torch FP16 scalar mismatches: {failures:?}"
        );
        Ok(())
    }

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

    #[test]
    fn rrdbnet_anime_6_blocks() {
        // Anime variant uses 6 blocks instead of 23
        let (_varmap, model) = build_test_rrdbnet(8, 4, 6, 4);
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), &device).unwrap();
        let output = model.forward(&input).unwrap();
        let dims = output.dims4().unwrap();
        assert_eq!(dims, (1, 3, 32, 32));
    }

    #[test]
    fn rrdbnet_output_has_3_channels() {
        let (_varmap, model) = build_test_rrdbnet(8, 4, 1, 4);
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 16, 12), &device).unwrap();
        let output = model.forward(&input).unwrap();
        let (b, c, h, w) = output.dims4().unwrap();
        assert_eq!(b, 1);
        assert_eq!(c, 3);
        assert_eq!(h, 64); // 16*4
        assert_eq!(w, 48); // 12*4
    }
}
