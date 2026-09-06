// Copyright 2024 The HuggingFace Team. All rights reserved.
// Apache-2.0; see THIRD_PARTY_NOTICES.md and LICENSE-APACHE-2.0.
//! Complete paint spatial UNet. References: diffusers 8a79d8ec
//! unet_2d_condition.py (forward), transformer_2d.py:480-527; Tencent
//! 82920d64 hy3dpaint/hunyuanpaintpbr/unet/modules.py:785-1100.
use super::paint_attention::{linear, projected, PaintRope};
use super::paint_block::{
    PaintBlockCondition, PaintBlockKind, PaintReferenceScale, PaintTransformerBlock,
};
use super::paint_conv::{conv, silu, PaintResample, PaintResampleKind, PaintResnet};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use mold_candle::stable_diffusion::normalization::DiffusersGroupNorm;
use std::collections::HashMap;

pub struct PaintUnetCondition<'a> {
    pub reference: &'a HashMap<String, Tensor>,
    pub dino: &'a Tensor,
    /// Keyed by all views' token count, as in Tencent's position pyramid.
    pub ropes: &'a HashMap<usize, PaintRope>,
    pub reference_scale: &'a Tensor,
}

struct Spatial {
    name: String,
    norm: DiffusersGroupNorm,
    input: candle_nn::Linear,
    output: candle_nn::Linear,
    block: PaintTransformerBlock,
}
impl Spatial {
    fn new(
        name: String,
        width: usize,
        heads: usize,
        kind: PaintBlockKind,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            name,
            norm: DiffusersGroupNorm::new(vb.pp("norm"), 32, width, 1e-6)?,
            input: linear(vb.pp("proj_in"), width, width, true)?,
            output: linear(vb.pp("proj_out"), width, width, true)?,
            block: PaintTransformerBlock::new(
                width,
                1024,
                heads,
                kind,
                vb.pp("transformer_blocks.0"),
            )?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        text: &Tensor,
        views: usize,
        condition: Option<&PaintUnetCondition<'_>>,
        cache: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let (batch, width, height, columns) = x.dims4()?;
        let hidden = self.norm.forward(x)?.permute((0, 2, 3, 1))?.reshape((
            batch,
            height * columns,
            width,
        ))?;
        let hidden = projected(&self.input, &hidden)?;
        let condition = condition
            .map(|c| -> Result<_> {
                Ok(PaintBlockCondition {
                    reference: c.reference.get(&self.name).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "missing paint reference cache {}",
                            self.name
                        ))
                    })?,
                    dino: c.dino,
                    rope: Some(c.ropes.get(&(views * height * columns)).ok_or_else(|| {
                        candle_core::Error::Msg("missing paint spatial RoPE".into())
                    })?),
                    reference_scale: PaintReferenceScale::PerBatch(c.reference_scale),
                    multiview_scale: 1.,
                })
            })
            .transpose()?;
        let result = self
            .block
            .forward(&hidden, text, views, condition.as_ref())?;
        if let Some(reference) = result.reference_cache {
            cache.insert(self.name.clone(), reference);
        }
        let hidden = projected(&self.output, &result.hidden)?
            .reshape((batch, height, columns, width))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        hidden + x
    }
}
struct Stage {
    residuals: Vec<PaintResnet>,
    attention: Vec<Spatial>,
    resample: Option<PaintResample>,
}

pub struct PaintUnet {
    input: candle_nn::Conv2d,
    time1: candle_nn::Linear,
    time2: candle_nn::Linear,
    down: Vec<Stage>,
    mid1: PaintResnet,
    mid_attention: Spatial,
    mid2: PaintResnet,
    up: Vec<Stage>,
    norm: DiffusersGroupNorm,
    output: candle_nn::Conv2d,
    width: usize,
    kind: PaintBlockKind,
    dtype: DType,
}

impl PaintUnet {
    pub fn new(kind: PaintBlockKind, vb: VarBuilder) -> Result<Self> {
        Self::with_widths(kind, [320, 640, 1280, 1280], [5, 10, 20, 20], vb)
    }
    fn with_widths(
        kind: PaintBlockKind,
        channels: [usize; 4],
        heads: [usize; 4],
        vb: VarBuilder,
    ) -> Result<Self> {
        let width = channels[0];
        let time = width * 4;
        let mut down = Vec::new();
        let mut previous = width;
        for i in 0..4 {
            let base = vb.pp(format!("down_blocks.{i}"));
            let mut residuals = Vec::new();
            let mut attention = Vec::new();
            for j in 0..2 {
                residuals.push(PaintResnet::new(
                    previous,
                    channels[i],
                    time,
                    32,
                    base.pp(format!("resnets.{j}")),
                )?);
                previous = channels[i];
                if i < 3 {
                    attention.push(Spatial::new(
                        format!("down_{i}_{j}_0"),
                        channels[i],
                        heads[i],
                        kind,
                        base.pp(format!("attentions.{j}")),
                    )?);
                }
            }
            down.push(Stage {
                residuals,
                attention,
                resample: if i < 3 {
                    Some(PaintResample::new(
                        channels[i],
                        PaintResampleKind::Down,
                        base.pp("downsamplers.0"),
                    )?)
                } else {
                    None
                },
            });
        }
        let mut up = Vec::new();
        for i in 0..4 {
            let level = 3 - i;
            let output = channels[level];
            let base = vb.pp(format!("up_blocks.{i}"));
            let mut residuals = Vec::new();
            let mut attention = Vec::new();
            for j in 0..3 {
                let skip = if j == 2 {
                    channels[level.saturating_sub(1)]
                } else {
                    output
                };
                residuals.push(PaintResnet::new(
                    previous + skip,
                    output,
                    time,
                    32,
                    base.pp(format!("resnets.{j}")),
                )?);
                previous = output;
                if i > 0 {
                    attention.push(Spatial::new(
                        format!("up_{i}_{j}_0"),
                        output,
                        heads[level],
                        kind,
                        base.pp(format!("attentions.{j}")),
                    )?);
                }
            }
            up.push(Stage {
                residuals,
                attention,
                resample: if i < 3 {
                    Some(PaintResample::new(
                        output,
                        PaintResampleKind::Up,
                        base.pp("upsamplers.0"),
                    )?)
                } else {
                    None
                },
            });
        }
        Ok(Self {
            input: conv(
                vb.pp("conv_in"),
                if kind == PaintBlockKind::Main { 12 } else { 4 },
                width,
                3,
                1,
                1,
            )?,
            time1: linear(vb.pp("time_embedding.linear_1"), width, time, true)?,
            time2: linear(vb.pp("time_embedding.linear_2"), time, time, true)?,
            down,
            mid1: PaintResnet::new(
                channels[3],
                channels[3],
                time,
                32,
                vb.pp("mid_block.resnets.0"),
            )?,
            mid_attention: Spatial::new(
                "mid_0_0".into(),
                channels[3],
                heads[3],
                kind,
                vb.pp("mid_block.attentions.0"),
            )?,
            mid2: PaintResnet::new(
                channels[3],
                channels[3],
                time,
                32,
                vb.pp("mid_block.resnets.1"),
            )?,
            up,
            norm: DiffusersGroupNorm::new(vb.pp("conv_norm_out"), 32, width, 1e-5)?,
            output: conv(vb.pp("conv_out"), width, 4, 3, 1, 1)?,
            width,
            kind,
            dtype: vb.dtype(),
        })
    }
    pub fn forward(
        &self,
        input: &Tensor,
        timestep: f64,
        text: &Tensor,
        views: usize,
        condition: Option<&PaintUnetCondition<'_>>,
    ) -> Result<(Tensor, HashMap<String, Tensor>)> {
        let (batch, channels, height, width) = input.dims4()?;
        let materials = if self.kind == PaintBlockKind::Main {
            2
        } else {
            1
        };
        let group = views
            .checked_mul(materials)
            .filter(|&n| n > 0)
            .ok_or_else(|| candle_core::Error::Msg("invalid paint UNet view count".into()))?;
        if batch == 0
            || !batch.is_multiple_of(group)
            || channels != if materials == 2 { 12 } else { 4 }
            || height == 0
            || width == 0
            || height % 8 != 0
            || width % 8 != 0
            || !timestep.is_finite()
            || input.dtype() != self.dtype
            || text.dims() != [batch, 77, 1024]
            || text.dtype() != self.dtype
            || condition.is_some() != (self.kind == PaintBlockKind::Main)
        {
            candle_core::bail!("invalid paint UNet input, timestep or conditioning")
        }
        let half = self.width / 2;
        let frequencies = Tensor::arange(0u32, half as u32, input.device())?
            .to_dtype(DType::F32)?
            .affine(-10000f64.ln(), 0.)?
            .affine(1. / half as f64, 0.)?
            .exp()?
            .affine(timestep, 0.)?;
        let time = Tensor::cat(&[frequencies.cos()?, frequencies.sin()?], 0)?
            .unsqueeze(0)?
            .repeat((batch, 1))?
            .to_dtype(self.dtype)?;
        let time = projected(&self.time2, &silu(&projected(&self.time1, &time)?)?)?;
        let mut x = self.input.forward(input)?;
        let mut skips = vec![x.clone()];
        let mut cache = HashMap::new();
        for stage in &self.down {
            for (j, residual) in stage.residuals.iter().enumerate() {
                x = residual.forward(&x, &time)?;
                if let Some(attention) = stage.attention.get(j) {
                    x = attention.forward(&x, text, views, condition, &mut cache)?;
                }
                skips.push(x.clone());
            }
            if let Some(down) = &stage.resample {
                x = down.forward(&x, None)?;
                skips.push(x.clone());
            }
        }
        x = self.mid1.forward(&x, &time)?;
        x = self
            .mid_attention
            .forward(&x, text, views, condition, &mut cache)?;
        x = self.mid2.forward(&x, &time)?;
        for stage in &self.up {
            for (j, residual) in stage.residuals.iter().enumerate() {
                let skip = skips
                    .pop()
                    .ok_or_else(|| candle_core::Error::Msg("paint UNet skip underflow".into()))?;
                x = residual.forward(&Tensor::cat(&[&x, &skip], 1)?, &time)?;
                if let Some(attention) = stage.attention.get(j) {
                    x = attention.forward(&x, text, views, condition, &mut cache)?;
                }
            }
            if let Some(up) = &stage.resample {
                x = up.forward(&x, None)?;
            }
        }
        if !skips.is_empty() {
            candle_core::bail!("paint UNet left unused skips")
        }
        Ok((self.output.forward(&silu(&self.norm.forward(&x)?)?)?, cache))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    #[ignore = "requires retained full Tencent UNet capture"]
    fn complete_paint_unet_matches_tencent() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("MOLD_PAINT_UNET_ORACLE").expect("retained oracle directory"),
        );
        let device = Device::new_cuda(0)?;
        let fixture = candle_core::safetensors::load(root.join("paint-unet.safetensors"), &device)?;
        let dtype = fixture["input.sample"].dtype();
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("paint-unet.json")).unwrap()).unwrap();
        let tiny = metadata["tiny"].as_bool().unwrap();
        let build = |vb: VarBuilder| -> Result<_> {
            let channels = if tiny {
                [32, 64, 128, 128]
            } else {
                [320, 640, 1280, 1280]
            };
            let heads = if tiny { [2, 4, 8, 8] } else { [5, 10, 20, 20] };
            let reference = PaintUnet::with_widths(
                PaintBlockKind::Reference,
                channels,
                heads,
                vb.pp("unet_dual"),
            )?;
            let main =
                PaintUnet::with_widths(PaintBlockKind::Main, channels, heads, vb.pp("unet"))?;
            let projector = super::super::paint_projector::PaintImageProjector::new(
                vb.pp("unet.image_proj_model_dino"),
            )?;
            let texts = ["albedo", "mr", "ref"]
                .map(|name| {
                    vb.pp("unet")
                        .get((77, 1024), &format!("learned_text_clip_{name}"))
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            Ok((reference, main, projector, texts))
        };
        let (reference, main, projector, texts) = if tiny {
            let weights = candle_core::safetensors::load(
                root.join("paint-unet-tiny-weights.safetensors"),
                &device,
            )?;
            let weights = weights
                .into_iter()
                .map(|(k, v)| Ok((k, v.to_dtype(dtype)?)))
                .collect::<Result<HashMap<_, _>>>()?;
            build(VarBuilder::from_tensors(weights, dtype, &device))?
        } else {
            let path =
                std::env::var("MOLD_PAINT_UNET_WEIGHTS").expect("installed paint checkpoint");
            super::super::paint_weights::load_pth_exact(
                std::path::Path::new(&path),
                dtype,
                &device,
                build,
            )
            .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?
        };
        let output = std::env::var("MOLD_PAINT_UNET_OUTPUT")
            .ok()
            .map(std::path::PathBuf::from);
        if let Some(output) = &output {
            std::fs::create_dir_all(output).unwrap();
        }
        let dims = fixture["input.sample"].dims();
        let (batch, views, size) = (dims[0], dims[2], dims[4]);
        let reference_views = fixture["input.reference"].dim(1)?;
        let reference_input =
            fixture["input.reference"].reshape((batch * reference_views, 4, size, size))?;
        let reference_text = texts[2]
            .unsqueeze(0)?
            .repeat((batch * reference_views, 1, 1))?;
        let (_, cache) =
            reference.forward(&reference_input, 0., &reference_text, reference_views, None)?;
        assert_eq!(cache.len(), 16);
        let compare = |label: &str, actual: &Tensor, expected: &Tensor| -> Result<bool> {
            if let Some(output) = &output {
                candle_core::safetensors::save(
                    &HashMap::from([("actual", actual.contiguous()?)]),
                    output.join(format!("{label}.safetensors")),
                )?;
            }
            let delta = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
            let maximum = delta.abs()?.max_all()?.to_scalar::<f32>()?;
            let rms = delta.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("{label}: max={maximum} rms={rms}");
            let (max_bound, rms_bound) = if dtype == DType::F16 {
                (0.02, 0.002)
            } else {
                (1e-4, 1e-5)
            };
            Ok(maximum < max_bound && rms < rms_bound)
        };
        let mut passed = true;
        let mut names = cache.keys().collect::<Vec<_>>();
        names.sort();
        for name in names {
            passed &= compare(
                name,
                &cache[name],
                &fixture[&format!("cache.reference.{name}")],
            )?;
        }
        let dino = projector.forward(&fixture["input.dino"])?;
        passed &= compare("dino", &dino, &fixture["cache.dino"])?;
        let positions =
            super::super::paint_positions::position_pyramid(&fixture["input.position_maps"], size)?;
        let mut ropes = HashMap::new();
        for grid in [size, size / 2, size / 4, size / 8] {
            let tokens = views * grid * grid;
            ropes.insert(
                tokens,
                PaintRope::new(&positions[&tokens], if tiny { 16 } else { 64 }, grid * 8, 2)?,
            );
        }
        let normal = fixture["input.normal"]
            .unsqueeze(1)?
            .repeat((1, 2, 1, 1, 1, 1))?;
        let position = fixture["input.position"]
            .unsqueeze(1)?
            .repeat((1, 2, 1, 1, 1, 1))?;
        let input = Tensor::cat(&[&fixture["input.sample"], &normal, &position], 3)?.reshape((
            batch * 2 * views,
            12,
            size,
            size,
        ))?;
        let text = fixture["input.text"]
            .unsqueeze(2)?
            .repeat((1, 1, views, 1, 1))?
            .reshape((batch * 2 * views, 77, 1024))?;
        let condition = PaintUnetCondition {
            reference: &cache,
            dino: &dino,
            ropes: &ropes,
            reference_scale: &fixture["input.reference_scale"],
        };
        for invalid_views in [0, usize::MAX, 1usize << (usize::BITS - 1)] {
            assert!(main
                .forward(&input, 500., &text, invalid_views, Some(&condition))
                .is_err());
        }
        for timestep in [500, 400] {
            let (actual, unused) =
                main.forward(&input, timestep as f64, &text, views, Some(&condition))?;
            assert!(unused.is_empty());
            passed &= compare(
                &format!("denoise.{timestep}"),
                &actual,
                &fixture[&format!("expected.{timestep}")],
            )?;
        }
        assert!(
            passed,
            "complete UNet parity gate failed; all outputs retained"
        );
        Ok(())
    }
}
