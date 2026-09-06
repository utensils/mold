//! Tencent Hunyuan3D 2.1 material, reference and position-aware paint attention.
//! Reference: hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py at
//! 82920d643c0dc2f7bfd7255f45f62d386edfe60c.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

/// The three checkpoint projection layouts. Text, DINO and multiview use Plain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintAttentionKind {
    Plain,
    MaterialSelf,
    Reference,
}

/// Cached float32 trigonometric tables, repeated in batch/material order.
pub struct PaintRope {
    cos: Tensor,
    sin: Tensor,
}

impl PaintRope {
    pub fn new(
        positions: &Tensor,
        width: usize,
        resolution: usize,
        materials: usize,
    ) -> Result<Self> {
        let (batch, tokens, xyz) = positions.dims3()?;
        if batch == 0
            || tokens == 0
            || xyz != 3
            || width == 0
            || !width.is_multiple_of(16)
            || width > 256
            || resolution == 0
            || resolution > 4096
            || !(1..=2).contains(&materials)
            || !matches!(positions.dtype(), DType::U32 | DType::I64)
        {
            candle_core::bail!("invalid paint rotary position shape, width or resolution")
        }
        let elements = batch
            .checked_mul(materials)
            .and_then(|n| n.checked_mul(tokens))
            .and_then(|n| n.checked_mul(width));
        if elements.is_none_or(|n| n > 64 * 1024 * 1024) {
            candle_core::bail!("paint rotary embedding exceeds allocation bound")
        }
        // Validate on the host before GPU gather; signed negatives must not wrap.
        let indices = positions
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        if indices
            .iter()
            .any(|&index| index < 0 || index >= resolution as i64)
        {
            candle_core::bail!("paint rotary position outside voxel grid")
        }
        let positions = positions
            .to_dtype(DType::U32)?
            .reshape((batch * tokens, 3))?;
        let mut cos = Vec::with_capacity(3);
        let mut sin = Vec::with_capacity(3);
        for (axis, dim) in [width / 8 * 3, width / 8 * 3, width / 8 * 2]
            .into_iter()
            .enumerate()
        {
            let mut c = Vec::with_capacity(resolution * dim);
            let mut s = Vec::with_capacity(resolution * dim);
            for position in 0..resolution {
                for pair in 0..dim / 2 {
                    let frequency = 1.0f32 / 10000.0f32.powf((2 * pair) as f32 / dim as f32);
                    let angle = position as f32 * frequency;
                    c.extend([angle.cos(); 2]);
                    s.extend([angle.sin(); 2]);
                }
            }
            let ids = positions.narrow(1, axis, 1)?.flatten_all()?.contiguous()?;
            cos.push(
                Tensor::from_vec(c, (resolution, dim), positions.device())?
                    .index_select(&ids, 0)?,
            );
            sin.push(
                Tensor::from_vec(s, (resolution, dim), positions.device())?
                    .index_select(&ids, 0)?,
            );
        }
        let repeat = |parts: &[Tensor]| -> Result<Tensor> {
            Tensor::cat(parts, 1)?
                .reshape((batch, 1, tokens, width))?
                .broadcast_as((batch, materials, tokens, width))?
                .contiguous()?
                .reshape((batch * materials, 1, tokens, width))
        };
        Ok(Self {
            cos: repeat(&cos)?,
            sin: repeat(&sin)?,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, heads, tokens, width) = x.dims4()?;
        if self.cos.dims() != [batch, 1, tokens, width]
            || !matches!(x.dtype(), DType::F16 | DType::F32)
        {
            candle_core::bail!("paint rotary embedding does not match attention query")
        }
        let pairs = x.reshape((batch, heads, tokens, width / 2, 2))?;
        let real = pairs.narrow(4, 0, 1)?;
        let imaginary = pairs.narrow(4, 1, 1)?.neg()?;
        let rotated = Tensor::cat(&[imaginary, real], 4)?
            .reshape(x.shape())?
            .to_dtype(DType::F32)?;
        (x.to_dtype(DType::F32)?.broadcast_mul(&self.cos)? + rotated.broadcast_mul(&self.sin)?)?
            .to_dtype(x.dtype())
    }
}

struct Projections {
    q: candle_nn::Linear,
    k: candle_nn::Linear,
    v: candle_nn::Linear,
    out: candle_nn::Linear,
}

pub(super) fn linear(
    vb: VarBuilder,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<candle_nn::Linear> {
    Ok(candle_nn::Linear::new(
        vb.get((output, input), "weight")?.to_dtype(DType::F32)?,
        if bias {
            Some(vb.get(output, "bias")?.to_dtype(DType::F32)?)
        } else {
            None
        },
    ))
}

pub(super) fn projected(layer: &candle_nn::Linear, input: &Tensor) -> Result<Tensor> {
    // Torch includes bias in float32 accumulation, then rounds the output once.
    layer
        .forward(&input.to_dtype(DType::F32)?)?
        .to_dtype(input.dtype())
}

impl Projections {
    fn new(vb: VarBuilder, query: usize, context: usize, suffix: &str) -> Result<Self> {
        Ok(Self {
            q: linear(vb.pp(format!("to_q{suffix}")), query, query, false)?,
            k: linear(vb.pp(format!("to_k{suffix}")), context, query, false)?,
            v: linear(vb.pp(format!("to_v{suffix}")), context, query, false)?,
            out: linear(vb.pp(format!("to_out{suffix}.0")), query, query, true)?,
        })
    }
}

pub struct PaintAttention {
    base: Projections,
    material: Option<Projections>,
    reference: Option<(candle_nn::Linear, candle_nn::Linear)>,
    kind: PaintAttentionKind,
    width: usize,
    context: usize,
    heads: usize,
    dtype: DType,
}

impl PaintAttention {
    pub fn new(
        width: usize,
        context: usize,
        heads: usize,
        kind: PaintAttentionKind,
        vb: VarBuilder,
    ) -> Result<Self> {
        if width == 0
            || context == 0
            || heads == 0
            || !width.is_multiple_of(heads)
            || width > 4096
            || context > 4096
            || !matches!(vb.dtype(), DType::F32 | DType::F16)
            || (kind != PaintAttentionKind::Plain && context != width)
        {
            candle_core::bail!("invalid paint attention dimensions or dtype")
        }
        let base = Projections::new(vb.clone(), width, context, "")?;
        let material = if kind == PaintAttentionKind::MaterialSelf {
            Some(Projections::new(vb.pp("processor"), width, context, "_mr")?)
        } else {
            None
        };
        let reference = if kind == PaintAttentionKind::Reference {
            Some((
                linear(vb.pp("processor.to_v_mr"), context, width, false)?,
                linear(vb.pp("processor.to_out_mr.0"), width, width, true)?,
            ))
        } else {
            None
        };
        Ok(Self {
            base,
            material,
            reference,
            kind,
            width,
            context,
            heads,
            dtype: vb.dtype(),
        })
    }

    /// MaterialSelf takes [batch,2,views,tokens,width]; other kinds take rank 3.
    /// Reference returns [batch,2,tokens,width]. No caller supplies a mask in
    /// the pinned paint recipe; spatial/group/QK normalization is absent.
    pub fn forward(
        &self,
        hidden: &Tensor,
        encoder: Option<&Tensor>,
        rope: Option<&PaintRope>,
    ) -> Result<Tensor> {
        if hidden.dtype() != self.dtype {
            candle_core::bail!("paint attention input dtype differs from checkpoint")
        }
        if let Some(material) = &self.material {
            let (batch, pbr, views, tokens, width) = hidden.dims5()?;
            if batch == 0
                || pbr != 2
                || views == 0
                || tokens == 0
                || width != self.width
                || encoder.is_some()
                || rope.is_some()
            {
                candle_core::bail!("invalid paint material attention input")
            }
            let mut outputs = Vec::with_capacity(2);
            for (index, projections) in [&self.base, material].into_iter().enumerate() {
                let input = hidden.narrow(1, index, 1)?.contiguous()?.reshape((
                    batch * views,
                    tokens,
                    width,
                ))?;
                outputs.push(
                    self.single(projections, &input, &input, None)?
                        .reshape((batch, 1, views, tokens, width))?,
                );
            }
            return Tensor::cat(&outputs, 1);
        }
        let (batch, tokens, width) = hidden.dims3()?;
        let encoder = encoder.unwrap_or(hidden);
        let (context_batch, context_tokens, context_width) = encoder.dims3()?;
        if batch == 0
            || tokens == 0
            || width != self.width
            || context_batch != batch
            || context_tokens == 0
            || context_width != self.context
            || encoder.dtype() != self.dtype
        {
            candle_core::bail!("invalid paint attention query or context")
        }
        if let Some((value_mr, out_mr)) = &self.reference {
            if rope.is_some() {
                candle_core::bail!("reference paint attention does not use rotary positions")
            }
            let q = self.heads(projected(&self.base.q, hidden)?)?;
            let k = self.heads(projected(&self.base.k, encoder)?)?;
            // Tencent concatenates BEFORE head reshape. Splitting V by material
            // first would change which value head each query head reads (815-839).
            let values = Tensor::cat(
                &[
                    projected(&self.base.v, encoder)?,
                    projected(value_mr, encoder)?,
                ],
                2,
            )?;
            let values = self.heads(values)?;
            let attended = attention(&q, &k, &values)?;
            let dim = self.width / self.heads;
            let mut results = Vec::with_capacity(2);
            for (index, out) in [&self.base.out, out_mr].into_iter().enumerate() {
                let value = attended
                    .narrow(3, index * dim, dim)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((batch, tokens, width))?;
                results.push(projected(out, &value)?.unsqueeze(1)?);
            }
            return Tensor::cat(&results, 1);
        }
        debug_assert_eq!(self.kind, PaintAttentionKind::Plain);
        self.single(&self.base, hidden, encoder, rope)
    }

    fn heads(&self, x: Tensor) -> Result<Tensor> {
        let (batch, tokens, width) = x.dims3()?;
        x.reshape((batch, tokens, self.heads, width / self.heads))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn single(
        &self,
        layers: &Projections,
        hidden: &Tensor,
        encoder: &Tensor,
        rope: Option<&PaintRope>,
    ) -> Result<Tensor> {
        let (batch, tokens, width) = hidden.dims3()?;
        let mut q = self.heads(projected(&layers.q, hidden)?)?;
        let mut k = self.heads(projected(&layers.k, encoder)?)?;
        let v = self.heads(projected(&layers.v, encoder)?)?;
        if let Some(rope) = rope {
            q = rope.apply(&q)?;
            k = rope.apply(&k)?;
        }
        let output = attention(&q, &k, &v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, tokens, width))?;
        projected(&layers.out, &output)
    }
}

/// Query chunking bounds each float32 score allocation to 64 MiB. Every chunk
/// still attends the entire key sequence; no view or material is dropped.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    attention_with_budget(q, k, v, 64 * 1024 * 1024)
}

fn attention_with_budget(q: &Tensor, k: &Tensor, v: &Tensor, bytes: usize) -> Result<Tensor> {
    let (batch, heads, queries, width) = q.dims4()?;
    let (kb, kh, keys, kw) = k.dims4()?;
    let (vb, vh, values, _) = v.dims4()?;
    if batch == 0
        || heads == 0
        || queries == 0
        || keys == 0
        || width == 0
        || (kb, kh, kw) != (batch, heads, width)
        || (vb, vh, values) != (batch, heads, keys)
        || q.dtype() != k.dtype()
        || q.dtype() != v.dtype()
    {
        candle_core::bail!("invalid paint attention Q/K/V")
    }
    let row_bytes = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(keys))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| {
            candle_core::Error::Msg("paint attention score dimensions overflow".into())
        })?;
    let chunk = (bytes / row_bytes).min(queries);
    if chunk == 0 {
        candle_core::bail!("paint attention key sequence exceeds score allocation bound")
    }
    let q = q.to_dtype(DType::F32)?;
    let kt = k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
    let vf = v.to_dtype(DType::F32)?;
    let mut result = Vec::with_capacity(queries.div_ceil(chunk));
    for start in (0..queries).step_by(chunk) {
        let scores = q
            .narrow(2, start, chunk.min(queries - start))?
            .contiguous()?
            .matmul(&kt)?
            .affine(1. / (width as f64).sqrt(), 0.)?;
        result.push(
            candle_nn::ops::softmax(&scores, D::Minus1)?
                .matmul(&vf)?
                .to_dtype(v.dtype())?,
        );
    }
    Tensor::cat(&result, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn fixture(device: &Device) -> Result<std::collections::HashMap<String, Tensor>> {
        candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-attention.safetensors"),
            device,
        )
    }

    #[test]
    fn paint_attention_matches_executable_tencent_processors() -> Result<()> {
        let tensors = fixture(&Device::Cpu)?;
        compare_processors(&tensors, &Device::Cpu, 16, None)?;
        Ok(())
    }

    fn compare_processors(
        tensors: &std::collections::HashMap<String, Tensor>,
        device: &Device,
        head_dim: usize,
        output: Option<&std::path::Path>,
    ) -> Result<()> {
        let width = tensors["self.weights.to_q.weight"].dim(0)?;
        let cross_width = tensors["cross.weights.to_k.weight"].dim(1)?;
        for label in ["self", "ref", "pose", "plain", "cross"] {
            let kind = match label {
                "self" => PaintAttentionKind::MaterialSelf,
                "ref" => PaintAttentionKind::Reference,
                _ => PaintAttentionKind::Plain,
            };
            for (name, dtype, tolerance) in [("f32", DType::F32, 1e-5), ("f16", DType::F16, 0.002)]
            {
                let vb = VarBuilder::from_tensors(tensors.clone(), dtype, device)
                    .pp(format!("{label}.weights"));
                let model = PaintAttention::new(
                    width,
                    if label == "cross" { cross_width } else { width },
                    width / head_dim,
                    kind,
                    vb,
                )?;
                let rope = if label == "pose" {
                    Some(PaintRope::new(&tensors["positions"], head_dim, 512, 1)?)
                } else {
                    None
                };
                let actual = model.forward(
                    &tensors[&format!("{label}.{name}.input")],
                    tensors.get(&format!("{label}.{name}.encoder")),
                    rope.as_ref(),
                )?;
                let expected = &tensors[&format!("{label}.{name}.expected")];
                assert_eq!(actual.dims(), expected.dims());
                if let Some(output) = output {
                    candle_core::safetensors::save(
                        &std::collections::HashMap::from([("actual", actual.clone())]),
                        output.join(format!("{label}-{name}.safetensors")),
                    )?;
                }
                let difference = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
                let maximum = difference.abs()?.max_all()?.to_scalar::<f32>()?;
                let rms = difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                let (tolerance, rms_bound) = if head_dim == 64 {
                    if dtype == DType::F32 {
                        (1e-4, 1e-5)
                    } else {
                        (0.01, 0.001)
                    }
                } else {
                    (tolerance, tolerance)
                };
                eprintln!("paint {label} {name} maximum error {maximum}, RMS {rms}");
                assert!(maximum < tolerance, "{label} {name} error {maximum}");
                assert!(rms < rms_bound, "{label} {name} RMS {rms}");
            }
        }
        Ok(())
    }

    #[test]
    fn rotary_positions_match_tencent_and_repeat_per_material() -> Result<()> {
        let tensors = fixture(&Device::Cpu)?;
        let rope = PaintRope::new(&tensors["positions"], 16, 512, 1)?;
        for (label, actual) in [("rope_cos", &rope.cos), ("rope_sin", &rope.sin)] {
            let delta = (actual.squeeze(1)? - &tensors[label])?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(delta < 1e-5, "{label}: {delta}");
        }
        for (label, bound) in [("f32", 1e-5), ("f16", 0.002)] {
            let output = rope.apply(&tensors[&format!("rope_{label}_input")])?;
            let delta = (output.to_dtype(DType::F32)?
                - tensors[&format!("rope_{label}_expected")].to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
            assert!(delta < bound, "{label}: {delta}");
        }
        let repeated = PaintRope::new(&tensors["positions"], 16, 512, 2)?;
        for batch in 0..2 {
            for material in 0..2 {
                let delta = (repeated.cos.narrow(0, batch * 2 + material, 1)?
                    - rope.cos.narrow(0, batch, 1)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
                assert_eq!(delta, 0.);
            }
        }
        let negative = Tensor::new(&[[[-1i64, 0, 0]]], &Device::Cpu)?;
        assert!(PaintRope::new(&negative, 16, 512, 1).is_err());
        assert!(PaintRope::new(&tensors["positions"], 8, 512, 1).is_err());
        assert!(PaintRope::new(&tensors["positions"], 16, 2, 1).is_err());
        Ok(())
    }

    #[test]
    fn rotary_allocation_limit_precedes_materializing_position_indices() -> Result<()> {
        let positions =
            Tensor::new(&[[[-1i64, 0, 0]]], &Device::Cpu)?.broadcast_as((1, 131073, 3))?;
        let error = PaintRope::new(&positions, 256, 512, 2)
            .err()
            .expect("oversized rotary table");
        assert!(error.to_string().contains("allocation bound"), "{error}");
        Ok(())
    }

    #[test]
    fn score_chunking_preserves_all_keys_and_rejects_an_unbounded_row() -> Result<()> {
        let fixture = fixture(&Device::Cpu)?;
        let q = &fixture["rope_f32_input"];
        let k = q.narrow(2, 0, 7)?.contiguous()?;
        let v = k.affine(0.7, 0.2)?;
        let one_row = 2 * 5 * 7 * 4;
        let whole = attention(q, &k, &v)?;
        let chunked = attention_with_budget(q, &k, &v, one_row * 3)?;
        let delta = (whole - chunked)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(delta < 1e-6, "chunked attention error {delta}");
        assert!(attention_with_budget(q, &k, &v, one_row - 1).is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires NVIDIA CUDA"]
    fn paint_attention_matches_tencent_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        compare_processors(&fixture(&device)?, &device, 16, None)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires installed-weight Tencent capture and NVIDIA CUDA"]
    fn pretrained_paint_attention_matches_tencent_on_cuda() -> Result<()> {
        let oracle =
            std::env::var("MOLD_PAINT_ATTENTION_ORACLE").expect("retained oracle directory");
        let output = std::path::PathBuf::from(
            std::env::var("MOLD_PAINT_ATTENTION_RESULT").expect("new output directory"),
        );
        std::fs::create_dir(&output)?;
        let device = Device::new_cuda(0)?;
        let tensors = candle_core::safetensors::load(
            std::path::Path::new(&oracle).join("paint-attention.safetensors"),
            &device,
        )?;
        compare_processors(&tensors, &device, 64, Some(&output))
    }
}
