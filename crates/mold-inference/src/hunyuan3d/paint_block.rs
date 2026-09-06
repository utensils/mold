//! Complete Tencent Hunyuan3D 2.1 paint transformer block.
//! Reference: hy3dpaint/hunyuanpaintpbr/unet/modules.py:273-707,
//! revision 82920d643c0dc2f7bfd7255f45f62d386edfe60c. The pinned recipe uses
//! affine LayerNorm, GEGLU, no dropout at inference and all four paint branches.

use super::paint_attention::{linear, projected, PaintAttention, PaintAttentionKind, PaintRope};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintBlockKind {
    Main,
    Reference,
}

pub enum PaintReferenceScale<'a> {
    Scalar(f64),
    PerBatch(&'a Tensor),
}

pub struct PaintBlockCondition<'a> {
    pub reference: &'a Tensor,
    pub dino: &'a Tensor,
    pub rope: Option<&'a PaintRope>,
    pub reference_scale: PaintReferenceScale<'a>,
    pub multiview_scale: f64,
}

pub struct PaintBlockOutput {
    pub hidden: Tensor,
    /// The reference network caches norm1, before attention or residual updates.
    pub reference_cache: Option<Tensor>,
}

pub struct PaintTransformerBlock {
    kind: PaintBlockKind,
    width: usize,
    context: usize,
    dtype: DType,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
    norm3: candle_nn::LayerNorm,
    self_attention: PaintAttention,
    text_attention: PaintAttention,
    paint: Option<PaintBranches>,
    feed_in: candle_nn::Linear,
    feed_out: candle_nn::Linear,
}

struct PaintBranches {
    reference: PaintAttention,
    multiview: PaintAttention,
    dino: PaintAttention,
}

fn norm(vb: VarBuilder, width: usize) -> Result<candle_nn::LayerNorm> {
    Ok(candle_nn::LayerNorm::new(
        vb.get(width, "weight")?.to_dtype(DType::F32)?,
        vb.get(width, "bias")?.to_dtype(DType::F32)?,
        1e-5,
    ))
}

fn normalized(norm: &candle_nn::LayerNorm, x: &Tensor) -> Result<Tensor> {
    norm.forward(&x.to_dtype(DType::F32)?)?.to_dtype(x.dtype())
}

fn scaled(x: &Tensor, scale: f64) -> Result<Tensor> {
    if !scale.is_finite() {
        candle_core::bail!("paint attention scale must be finite")
    }
    x.to_dtype(DType::F32)?
        .affine(scale, 0.)?
        .to_dtype(x.dtype())
}

impl PaintTransformerBlock {
    pub fn new(
        width: usize,
        context: usize,
        heads: usize,
        kind: PaintBlockKind,
        vb: VarBuilder,
    ) -> Result<Self> {
        let base = vb.pp("transformer");
        // Attention construction validates dimensions before any larger FF allocation.
        let self_attention = PaintAttention::new(
            width,
            width,
            heads,
            if kind == PaintBlockKind::Main {
                PaintAttentionKind::MaterialSelf
            } else {
                PaintAttentionKind::Plain
            },
            base.pp("attn1"),
        )?;
        let text_attention = PaintAttention::new(
            width,
            context,
            heads,
            PaintAttentionKind::Plain,
            base.pp("attn2"),
        )?;
        let paint = if kind == PaintBlockKind::Main {
            Some(PaintBranches {
                reference: PaintAttention::new(
                    width,
                    width,
                    heads,
                    PaintAttentionKind::Reference,
                    vb.pp("attn_refview"),
                )?,
                multiview: PaintAttention::new(
                    width,
                    width,
                    heads,
                    PaintAttentionKind::Plain,
                    vb.pp("attn_multiview"),
                )?,
                dino: PaintAttention::new(
                    width,
                    context,
                    heads,
                    PaintAttentionKind::Plain,
                    vb.pp("attn_dino"),
                )?,
            })
        } else {
            None
        };
        Ok(Self {
            kind,
            width,
            context,
            dtype: vb.dtype(),
            norm1: norm(base.pp("norm1"), width)?,
            norm2: norm(base.pp("norm2"), width)?,
            norm3: norm(base.pp("norm3"), width)?,
            self_attention,
            text_attention,
            paint,
            feed_in: linear(base.pp("ff.net.0.proj"), width, width * 8, true)?,
            feed_out: linear(base.pp("ff.net.2"), width * 4, width, true)?,
        })
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        text: &Tensor,
        views: usize,
        condition: Option<&PaintBlockCondition<'_>>,
    ) -> Result<PaintBlockOutput> {
        let (rows, tokens, width) = hidden.dims3()?;
        let materials = if self.kind == PaintBlockKind::Main {
            2
        } else {
            1
        };
        let group = views
            .checked_mul(materials)
            .filter(|&n| n != 0)
            .ok_or_else(|| candle_core::Error::Msg("invalid paint view count".into()))?;
        let (text_rows, text_tokens, text_width) = text.dims3()?;
        if rows == 0
            || !rows.is_multiple_of(group)
            || tokens == 0
            || width != self.width
            || hidden.dtype() != self.dtype
            || text.dtype() != self.dtype
            || text_rows != rows
            || text_tokens == 0
            || text_width != self.context
            || self.paint.is_some() != condition.is_some()
        {
            candle_core::bail!("invalid paint block inputs or conditioning")
        }
        let batch = rows / group;
        if let Some(condition) = condition {
            if condition.reference.dims().len() != 3
                || condition.reference.dim(0)? != batch
                || condition.reference.dim(2)? != width
                || condition.dino.dims().len() != 3
                || condition.dino.dim(0)? != batch
                || condition.dino.dim(2)? != self.context
                || condition.reference.dtype() != self.dtype
                || condition.dino.dtype() != self.dtype
                || !condition.multiview_scale.is_finite()
            {
                candle_core::bail!("invalid paint reference or DINO conditioning")
            }
            match condition.reference_scale {
                PaintReferenceScale::Scalar(value) if !value.is_finite() => {
                    candle_core::bail!("invalid paint reference scale")
                }
                PaintReferenceScale::PerBatch(value)
                    if value.dims() != [batch] || value.dtype() != self.dtype =>
                {
                    candle_core::bail!("paint reference scale must match the batch and dtype")
                }
                PaintReferenceScale::PerBatch(value)
                    if value
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?
                        .iter()
                        .any(|x| !x.is_finite()) =>
                {
                    candle_core::bail!("paint reference scale must be finite")
                }
                _ => (),
            }
        }
        let norm1 = normalized(&self.norm1, hidden)?;
        let attention_input = if materials == 2 {
            norm1.reshape((batch, materials, views, tokens, width))?
        } else {
            norm1.clone()
        };
        let mut hidden = (hidden
            + self
                .self_attention
                .forward(&attention_input, None, None)?
                .reshape((rows, tokens, width))?)?;
        let reference_cache = if self.kind == PaintBlockKind::Reference {
            Some(norm1.reshape((batch, views * tokens, width))?)
        } else {
            None
        };
        if let (Some(branches), Some(condition)) = (&self.paint, condition) {
            // Both branches read the SAME original norm1, never the updated residual.
            let albedo = norm1
                .reshape((batch, 2, views * tokens, width))?
                .narrow(1, 0, 1)?
                .contiguous()?
                .reshape((batch, views * tokens, width))?;
            let reference = branches
                .reference
                .forward(&albedo, Some(condition.reference), None)?
                .reshape((rows, tokens, width))?;
            let reference = match condition.reference_scale {
                PaintReferenceScale::Scalar(scale) => scaled(&reference, scale)?,
                PaintReferenceScale::PerBatch(scale) => reference
                    .reshape((batch, group, tokens, width))?
                    .broadcast_mul(&scale.reshape((batch, 1, 1, 1))?)?
                    .reshape((rows, tokens, width))?,
            };
            hidden = (reference + hidden)?;
            if views > 1 {
                let multiview = norm1.reshape((batch * materials, views * tokens, width))?;
                let multiview = branches
                    .multiview
                    .forward(&multiview, Some(&multiview), condition.rope)?
                    .reshape((rows, tokens, width))?;
                hidden = (scaled(&multiview, condition.multiview_scale)? + hidden)?;
            }
        }
        let norm2 = normalized(&self.norm2, &hidden)?;
        hidden = (self.text_attention.forward(&norm2, Some(text), None)? + hidden)?;
        if let (Some(branches), Some(condition)) = (&self.paint, condition) {
            let context_tokens = condition.dino.dim(1)?;
            let dino = condition
                .dino
                .unsqueeze(1)?
                .broadcast_as((batch, group, context_tokens, self.context))?
                .contiguous()?
                .reshape((rows, context_tokens, self.context))?;
            // DINO uses the pre-text norm2 output, matching Tencent modules.py:663-678.
            hidden = (branches.dino.forward(&norm2, Some(&dino), None)? + hidden)?;
        }
        let feed = projected(&self.feed_in, &normalized(&self.norm3, &hidden)?)?;
        let inner = self.width * 4;
        let gate = feed
            .narrow(2, inner, inner)?
            .to_dtype(DType::F32)?
            .gelu_erf()?
            .to_dtype(self.dtype)?;
        let feed = (feed.narrow(2, 0, inner)? * gate)?;
        hidden = (projected(&self.feed_out, &feed)? + hidden)?;
        Ok(PaintBlockOutput {
            hidden,
            reference_cache,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn fixture(device: &Device) -> Result<std::collections::HashMap<String, Tensor>> {
        candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-block.safetensors"),
            device,
        )
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires retained installed-weight spatial trace"]
    fn captured_paint_spatial_stages_match_tencent() -> anyhow::Result<()> {
        use std::{collections::HashMap, path::PathBuf};
        let root = PathBuf::from(std::env::var("MOLD_PAINT_SPATIAL_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_PAINT_SPATIAL_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let device = Device::new_cuda(0)?;
        let tensors = candle_core::safetensors::load(root.join("paint-unet.safetensors"), &device)?;
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("paint-unet.json"))?)?;
        let mut passed = true;
        for site in ["up_1_2_0", "up_2_0_0", "up_2_1_0"] {
            for stage in ["groupnorm", "projection", "layernorm"] {
                let key = format!("trace.{site}.{stage}");
                let input = &tensors[&format!("{key}.input")];
                // Safetensors stores contiguous bytes; restore the actual
                // spatial projection's B,HW,C view before invoking Linear.
                let restored = if stage == "projection" {
                    input.transpose(1, 2)?.contiguous()?.transpose(1, 2)?
                } else {
                    input.clone()
                };
                let input = &restored;
                let expected_stride: Vec<usize> =
                    serde_json::from_value(metadata["trace_layouts"][&key]["stride"].clone())?;
                assert_eq!(input.stride(), expected_stride);
                let expected = &tensors[&format!("{key}.output")];
                let weights = HashMap::from([
                    ("weight".into(), tensors[&format!("{key}.weight")].clone()),
                    ("bias".into(), tensors[&format!("{key}.bias")].clone()),
                ]);
                let vb = VarBuilder::from_tensors(weights, input.dtype(), &device);
                let actual = match stage {
                    "groupnorm" => {
                        mold_candle::stable_diffusion::normalization::DiffusersGroupNorm::new(
                            vb,
                            32,
                            input.dim(1)?,
                            1e-6,
                        )?
                        .forward(input)?
                    }
                    "projection" => projected(
                        &linear(
                            vb,
                            input.dim(candle_core::D::Minus1)?,
                            expected.dim(candle_core::D::Minus1)?,
                            true,
                        )?,
                        input,
                    )?,
                    "layernorm" => {
                        normalized(&norm(vb, input.dim(candle_core::D::Minus1)?)?, input)?
                    }
                    _ => unreachable!(),
                };
                candle_core::safetensors::save(
                    &HashMap::from([("actual", actual.clone())]),
                    output.join(format!("{key}.safetensors")),
                )?;
                let delta = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
                let max = delta.abs()?.max_all()?.to_scalar::<f32>()?;
                let rms = delta.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                eprintln!("{key}: max={max} rms={rms}");
                // Tighter than the complete network's .02/.002 gate: replay
                // each layer on the oracle's exact input to localize drift.
                passed &= max < 0.002 && rms < 0.0003;
            }
        }
        anyhow::ensure!(passed, "spatial stage replay diverges; tensors retained");
        Ok(())
    }
    #[test]
    fn rejects_nonfinite_reference_scales_before_computation() -> Result<()> {
        let device = Device::Cpu;
        let tensors = fixture(&device)?;
        let model = PaintTransformerBlock::new(
            80,
            12,
            5,
            PaintBlockKind::Main,
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device).pp("main.weights"),
        )?;
        let get = |key: &str| &tensors[&format!("main.f32.views3.{key}")];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let scale = Tensor::new(&[bad, 1.0], &device)?;
            let condition = PaintBlockCondition {
                reference: get("reference"),
                dino: get("dino"),
                rope: None,
                reference_scale: PaintReferenceScale::PerBatch(&scale),
                multiview_scale: 0.7,
            };
            assert!(
                model
                    .forward(get("input"), get("encoder"), 3, Some(&condition))
                    .is_err(),
                "nonfinite per-batch scale must be rejected"
            );
        }
        Ok(())
    }

    #[test]
    fn complete_paint_block_matches_tencent() -> Result<()> {
        compare(&fixture(&Device::Cpu)?, &Device::Cpu, false, None)
    }

    fn compare(
        tensors: &std::collections::HashMap<String, Tensor>,
        device: &Device,
        pretrained: bool,
        output: Option<&std::path::Path>,
    ) -> Result<()> {
        let (width, context, head_dim, max_views) = if pretrained {
            (320, 1024, 64, 6)
        } else {
            (80, 12, 16, 3)
        };
        for (branch, kind) in [
            ("main", PaintBlockKind::Main),
            ("dual", PaintBlockKind::Reference),
        ] {
            for (name, dtype) in [("f32", DType::F32), ("f16", DType::F16)] {
                let model = PaintTransformerBlock::new(
                    width,
                    context,
                    width / head_dim,
                    kind,
                    VarBuilder::from_tensors(tensors.clone(), dtype, device)
                        .pp(format!("{branch}.weights")),
                )?;
                let mut cases = vec![(max_views, ""), (1, "")];
                if kind == PaintBlockKind::Main {
                    cases.push((max_views, ".cfg3"));
                }
                for (views, suffix) in cases {
                    let prefix = format!("{branch}.{name}.views{views}{suffix}");
                    let get = |key: &str| &tensors[&format!("{prefix}.{key}")];
                    let rope = PaintRope::new(get("positions"), head_dim, 512, 2)?;
                    let mut condition = PaintBlockCondition {
                        reference: get("reference"),
                        dino: get("dino"),
                        rope: Some(&rope),
                        reference_scale: PaintReferenceScale::PerBatch(get("ref_scale")),
                        multiview_scale: 0.7,
                    };
                    let result = model.forward(
                        get("input"),
                        get("encoder"),
                        views,
                        if kind == PaintBlockKind::Main {
                            Some(&condition)
                        } else {
                            None
                        },
                    )?;
                    if let Some(output) = output {
                        let mut actual =
                            std::collections::HashMap::from([("hidden", result.hidden.clone())]);
                        if let Some(cache) = &result.reference_cache {
                            actual.insert("reference_cache", cache.clone());
                        }
                        candle_core::safetensors::save(
                            &actual,
                            output.join(format!("{prefix}.safetensors")),
                        )?;
                    }
                    let (maximum, rms) = error(&result.hidden, get("expected"))?;
                    let (max_bound, rms_bound) = if dtype == DType::F32 {
                        (1e-4, 1e-5)
                    } else {
                        (0.02, 0.002)
                    };
                    eprintln!("paint block {prefix}: maximum {maximum}, RMS {rms}");
                    assert!(
                        maximum < max_bound && rms < rms_bound,
                        "{prefix}: max {maximum}, RMS {rms}"
                    );
                    if kind == PaintBlockKind::Reference {
                        let (maximum, rms) = error(
                            result.reference_cache.as_ref().expect("reference cache"),
                            get("cache_expected"),
                        )?;
                        assert!(
                            maximum < max_bound && rms < rms_bound,
                            "reference cache {prefix}: max {maximum}, RMS {rms}"
                        );
                    } else {
                        assert!(result.reference_cache.is_none());
                        for (scale_name, scale) in [
                            ("scalar", PaintReferenceScale::Scalar(0.6)),
                            (
                                "zero_cfg",
                                PaintReferenceScale::PerBatch(get("zero_cfg_scale")),
                            ),
                        ] {
                            condition.reference_scale = scale;
                            let result = model.forward(
                                get("input"),
                                get("encoder"),
                                views,
                                Some(&condition),
                            )?;
                            if let Some(output) = output {
                                candle_core::safetensors::save(
                                    &std::collections::HashMap::from([(
                                        "hidden",
                                        result.hidden.clone(),
                                    )]),
                                    output.join(format!("{prefix}.{scale_name}.safetensors")),
                                )?;
                            }
                            let (maximum, rms) =
                                error(&result.hidden, get(&format!("{scale_name}_expected")))?;
                            eprintln!(
                                "paint block {prefix} {scale_name}: maximum {maximum}, RMS {rms}"
                            );
                            assert!(
                                maximum < max_bound && rms < rms_bound,
                                "{prefix} {scale_name}: max {maximum}, RMS {rms}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn error(actual: &Tensor, expected: &Tensor) -> Result<(f32, f32)> {
        assert_eq!(actual.dims(), expected.dims());
        let difference = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
        Ok((
            difference.abs()?.max_all()?.to_scalar::<f32>()?,
            difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?,
        ))
    }

    #[test]
    fn reference_cache_precedes_text_conditioning_and_attention() -> Result<()> {
        let device = Device::Cpu;
        let tensors = fixture(&device)?;
        let model = PaintTransformerBlock::new(
            80,
            12,
            5,
            PaintBlockKind::Reference,
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device).pp("dual.weights"),
        )?;
        let get = |key: &str| &tensors[&format!("dual.f32.views3.{key}")];
        let original = model.forward(get("input"), get("encoder"), 3, None)?;
        let changed = model.forward(get("input"), &get("encoder").zeros_like()?, 3, None)?;
        assert!(
            error(&original.hidden, &changed.hidden)?.0 > 1e-4,
            "text must affect the block output"
        );
        assert_eq!(
            error(
                original.reference_cache.as_ref().unwrap(),
                changed.reference_cache.as_ref().unwrap()
            )?
            .0,
            0.
        );
        assert!(model
            .forward(get("input"), get("encoder"), 0, None)
            .is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires installed-weight Tencent capture and NVIDIA CUDA"]
    fn pretrained_paint_block_matches_tencent_on_cuda() -> Result<()> {
        let oracle = std::env::var("MOLD_PAINT_BLOCK_ORACLE").expect("retained oracle directory");
        let output = std::path::PathBuf::from(
            std::env::var("MOLD_PAINT_BLOCK_RESULT").expect("new output directory"),
        );
        std::fs::create_dir(&output)?;
        let device = Device::new_cuda(0)?;
        let tensors = candle_core::safetensors::load(
            std::path::Path::new(&oracle).join("paint-block.safetensors"),
            &device,
        )?;
        compare(&tensors, &device, true, Some(&output))
    }
}
