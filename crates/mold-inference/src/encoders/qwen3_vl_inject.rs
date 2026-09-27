//! Qwen3-VL multimodal hooks shared by the BF16 and GGUF language models.
//!
//! Qwen Image 2.1 conditions on Qwen3-VL-8B-Instruct's language model with the
//! vision tower's rows spliced in. Three things differ from a text-only
//! forward, and all three are the language model's business rather than the
//! tower's, so both LM arms (`qwen3_bf16`, `qwen3_gguf`) apply them through
//! this one module:
//!
//! 1. **Row replacement.** The embedding rows at `<|image_pad|>` positions are
//!    replaced by the vision merger's output (transformers
//!    `modeling_qwen3_vl.py` `Qwen3VLModel.forward`, `masked_scatter`).
//! 2. **Interleaved MRoPE.** Each rotary frequency index takes its position
//!    from the T, H or W axis per `apply_interleaved_mrope` (`:299-314`):
//!    index `i` reads H when `i % 3 == 1 && i < 3·sections[1]`, W when
//!    `i % 3 == 2 && i < 3·sections[2]`, and T otherwise. With all three axes
//!    equal (text-only) this is ordinary 1-D RoPE.
//! 3. **DeepStack.** After decoder layer `k < deepstack.len()`, the k-th
//!    DeepStack feature is ADDED to the hidden rows at the visual positions
//!    (`:861-867`, `_deepstack_process` `:876-883`).
//!
//! The vision tower itself always runs from the BF16 shards whatever the LM
//! tier, so a GGUF LM receives exactly the rows a BF16 LM would.

// Reached through `GgufQwen3Encoder::forward_multimodal_final_pre_norm` and the
// BF16 twin, which the reference-image conditioning encoder drives.
#![allow(dead_code)]

use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};

/// Qwen3-VL-8B's MRoPE sections (`text_config.rope_scaling.mrope_section`,
/// and the GGUF's `qwen3vl.rope.dimension_sections = [24, 20, 20, 0]`).
pub(crate) const QWEN3_VL_MROPE_SECTIONS: [usize; 3] = [24, 20, 20];

/// The vision side of one multimodal forward.
#[derive(Clone, Debug)]
pub(crate) struct VisualInjection {
    /// Sequence positions of the `<|image_pad|>` tokens, ascending, one per
    /// merger row.
    pub positions: Vec<usize>,
    /// Merger output, `[positions.len(), hidden]`.
    pub embeds: Tensor,
    /// DeepStack features, one `[positions.len(), hidden]` per early layer.
    pub deepstack: Vec<Tensor>,
}

impl VisualInjection {
    pub(crate) fn validate(&self, sequence: usize, hidden: usize, layers: usize) -> Result<()> {
        let rows = self.positions.len();
        ensure!(
            self.positions.windows(2).all(|pair| pair[0] < pair[1]),
            "Qwen3-VL visual positions must be strictly ascending"
        );
        ensure!(
            self.positions.last().is_none_or(|last| *last < sequence),
            "Qwen3-VL visual position past the {sequence}-token sequence"
        );
        ensure!(
            self.embeds.dims() == [rows, hidden],
            "Qwen3-VL visual embeds are {:?}, expected [{rows}, {hidden}]",
            self.embeds.dims()
        );
        ensure!(
            self.deepstack.len() <= layers,
            "{} DeepStack features for a {layers}-layer language model",
            self.deepstack.len()
        );
        for (index, feature) in self.deepstack.iter().enumerate() {
            ensure!(
                feature.dims() == [rows, hidden],
                "Qwen3-VL DeepStack feature {index} is {:?}, expected [{rows}, {hidden}]",
                feature.dims()
            );
        }
        Ok(())
    }
}

/// Maximal runs of consecutive positions, as `(start, rows_before, len)`.
fn runs(positions: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut out: Vec<(usize, usize, usize)> = Vec::new();
    for (row, &position) in positions.iter().enumerate() {
        match out.last_mut() {
            Some((start, _, len)) if *start + *len == position => *len += 1,
            _ => out.push((position, row, 1)),
        }
    }
    out
}

/// Write (`add = false`) or add (`add = true`) `rows` into `hidden[0]` at
/// `positions`. `hidden` is `[1, L, H]`. Image pads are contiguous per image,
/// so this is one `slice_assign` per image, not per token.
fn place_rows(hidden: &Tensor, positions: &[usize], rows: &Tensor, add: bool) -> Result<Tensor> {
    let (batch, _, width) = hidden.dims3()?;
    ensure!(batch == 1, "Qwen3-VL multimodal forward is batch-1");
    let rows = rows.to_device(hidden.device())?.to_dtype(hidden.dtype())?;
    let mut hidden = hidden.clone();
    for (start, first_row, len) in runs(positions) {
        let block = rows.narrow(0, first_row, len)?.unsqueeze(0)?;
        let block = if add {
            (hidden.narrow(1, start, len)? + block)?
        } else {
            block
        };
        hidden = hidden.slice_assign(&[0..1, start..start + len, 0..width], &block)?;
    }
    Ok(hidden)
}

/// Replace the `<|image_pad|>` embedding rows with the merger output.
pub(crate) fn inject_visual_rows(hidden: &Tensor, visual: &VisualInjection) -> Result<Tensor> {
    place_rows(hidden, &visual.positions, &visual.embeds, false)
}

/// Apply DeepStack after decoder layer `layer`: a no-op past the last feature.
pub(crate) fn apply_deepstack(
    hidden: &Tensor,
    visual: &VisualInjection,
    layer: usize,
) -> Result<Tensor> {
    match visual.deepstack.get(layer) {
        Some(feature) => place_rows(hidden, &visual.positions, feature, true),
        None => Ok(hidden.clone()),
    }
}

/// The frequency index → axis map of interleaved MRoPE.
pub(crate) fn mrope_axis(index: usize, sections: [usize; 3]) -> usize {
    if index % 3 == 1 && index < sections[1] * 3 {
        1
    } else if index % 3 == 2 && index < sections[2] * 3 {
        2
    } else {
        0
    }
}

/// `(cos, sin)` for a batch-1 sequence, each `[L, head_dim / 2]` F32: the
/// frequency table evaluated at each index's own axis position.
///
/// The value at `(t, i)` is `cos/sin(position[axis(i)][t] · θ^(-2i/d))` —
/// arithmetically the gather from a 1-D table that transformers' interleave
/// performs, so equal axes reproduce the text-only table exactly.
pub(crate) fn mrope_cos_sin(
    mrope: &[Vec<u32>; 3],
    head_dim: usize,
    rope_theta: f64,
    sections: [usize; 3],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let sequence = mrope[0].len();
    ensure!(
        mrope.iter().all(|axis| axis.len() == sequence),
        "Qwen3-VL MRoPE axes differ in length"
    );
    let half = head_dim / 2;
    ensure!(
        sections.iter().sum::<usize>() == half,
        "MRoPE sections {sections:?} do not cover {half} frequencies"
    );
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let axes = (0..half)
        .map(|i| &mrope[mrope_axis(i, sections)])
        .collect::<Vec<_>>();
    let angles = (0..sequence)
        .flat_map(|t| {
            axes.iter()
                .zip(&inv_freq)
                .map(move |(axis, freq)| axis[t] as f32 * freq)
        })
        .collect::<Vec<_>>();
    let angles = Tensor::from_vec(angles, (sequence, half), device)?;
    Ok((angles.cos()?, angles.sin()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn the_axis_map_matches_transformers_interleave() {
        // transformers: freqs_t[..., slice(1, 60, 3)] = H; slice(2, 60, 3) = W.
        let sections = QWEN3_VL_MROPE_SECTIONS;
        let axes = (0..64).map(|i| mrope_axis(i, sections)).collect::<Vec<_>>();
        for (i, axis) in axes.iter().enumerate() {
            let expected = if (1..60).step_by(3).any(|h| h == i) {
                1
            } else if (2..60).step_by(3).any(|w| w == i) {
                2
            } else {
                0
            };
            assert_eq!(*axis, expected, "frequency {i}");
        }
        assert_eq!(axes.iter().filter(|a| **a == 0).count(), 24);
        assert_eq!(axes.iter().filter(|a| **a == 1).count(), 20);
        assert_eq!(axes.iter().filter(|a| **a == 2).count(), 20);
    }

    #[test]
    fn equal_axes_reduce_to_one_dimensional_rope() {
        let positions: Vec<u32> = (0..7).collect();
        let mrope = [positions.clone(), positions.clone(), positions];
        let (cos, sin) =
            mrope_cos_sin(&mrope, 128, 5e6, QWEN3_VL_MROPE_SECTIONS, &Device::Cpu).unwrap();
        let (cos_1d, sin_1d) = mrope_cos_sin(
            &[(0..7).collect(), vec![0; 7], vec![0; 7]],
            128,
            5e6,
            [64, 0, 0],
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            cos.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            cos_1d.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            sin.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            sin_1d.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn rows_are_replaced_then_deepstack_is_added_at_the_same_positions() {
        let hidden = Tensor::zeros((1, 6, 2), DType::F32, &Device::Cpu).unwrap();
        let visual = VisualInjection {
            positions: vec![1, 2, 4],
            embeds: Tensor::from_vec(vec![1f32, 1., 2., 2., 3., 3.], (3, 2), &Device::Cpu).unwrap(),
            deepstack: vec![Tensor::ones((3, 2), DType::F32, &Device::Cpu).unwrap()],
        };
        visual.validate(6, 2, 36).unwrap();
        let injected = inject_visual_rows(&hidden, &visual).unwrap();
        let stacked = apply_deepstack(&injected, &visual, 0).unwrap();
        let untouched = apply_deepstack(&stacked, &visual, 1).unwrap();
        let rows = untouched.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(
            rows,
            [[0., 0.], [2., 2.], [3., 3.], [0., 0.], [4., 4.], [0., 0.]]
        );
        assert_eq!(runs(&[1, 2, 4]), [(1, 0, 2), (4, 2, 1)]);
    }

    #[test]
    fn a_malformed_injection_is_refused() {
        let embeds = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        let bad_order = VisualInjection {
            positions: vec![3, 2],
            embeds: embeds.clone(),
            deepstack: vec![],
        };
        assert!(bad_order.validate(8, 4, 36).is_err());
        let past_end = VisualInjection {
            positions: vec![6, 8],
            embeds: embeds.clone(),
            deepstack: vec![],
        };
        assert!(past_end.validate(8, 4, 36).is_err());
        let bad_width = VisualInjection {
            positions: vec![1, 2],
            embeds,
            deepstack: vec![],
        };
        assert!(bad_width.validate(8, 5, 36).is_err());
    }
}
