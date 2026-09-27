//! Qwen Image 2.1's block-causal attention dispatch.
//!
//! [`SegmentDispatch::attend`] executes a [`BlockCausalPlan`] one segment at a
//! time, mirroring upstream's default `QwenImage21AttnProcessor`
//! (`transformer_qwenimage21.py:462-545` at `e0abab83b`): each segment's
//! queries attend to keys `[0, kv_len)`, text segments add a causal triangle
//! over their own keys, and padded keys are excluded. Every backend decision
//! lives in [`SegmentDispatch::attend_segment`]; that single function is the
//! seam a fused or flash implementation plugs into, and the math arms below
//! are the definition any such implementation must reproduce.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::layout::{AttentionSegment, BlockCausalPlan, SegmentMask};

/// Per-attention-module dispatch configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmentDispatch {
    /// Metal's fused SDPA for unbiased `Full` segments. Resolved once from
    /// `MOLD_ATTN` at construction (`metal_fast_path_enabled`).
    pub fused_target: bool,
    pub head_dim: usize,
}

impl SegmentDispatch {
    fn scale(&self) -> f32 {
        (1.0 / (self.head_dim as f64).sqrt()) as f32
    }

    /// Attend `q` `[B, H, Sq, D]` against `k`/`v` `[B, H, Skv, D]` under
    /// `plan`, returning `[B, H, Sq, D]` with the segments in query order.
    pub(crate) fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        plan: &BlockCausalPlan,
    ) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(plan.segments.len());
        for segment in &plan.segments {
            outputs.push(self.attend_segment(q, k, v, segment, plan)?);
        }
        if outputs.len() == 1 {
            return Ok(outputs.pop().expect("one segment"));
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        Ok(Tensor::cat(&refs, 2)?)
    }

    /// One segment of the plan. `q`, `k` and `v` are the whole tensors; the
    /// segment names the rows it reads.
    pub(crate) fn attend_segment(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        segment: &AttentionSegment,
        plan: &BlockCausalPlan,
    ) -> Result<Tensor> {
        // Candle Metal matmul requires the physical tensor extent to match the
        // view it receives, and a `narrow` retains the whole joint allocation,
        // so a partial view is made contiguous. A view that already IS the
        // whole tensor is passed through untouched.
        let q_seg = if segment.q_start == 0 && segment.q_len == q.dim(2)? {
            q.clone()
        } else {
            q.narrow(2, segment.q_start, segment.q_len)?.contiguous()?
        };
        let (k_seg, v_seg) = if segment.kv_len == k.dim(2)? {
            (k.clone(), v.clone())
        } else {
            (
                k.narrow(2, 0, segment.kv_len)?.contiguous()?,
                v.narrow(2, 0, segment.kv_len)?.contiguous()?,
            )
        };
        let batch = q.dim(0)?;
        match segment.mask {
            SegmentMask::CausalBottomRight => {
                let bias = causal_bottom_right_bias(
                    segment,
                    plan.key_valid_prefix(segment.kv_len).as_deref(),
                    batch,
                    q.dtype(),
                    q.device(),
                )?;
                Ok(crate::attention::attention_with_bias(
                    &q_seg,
                    &k_seg,
                    &v_seg,
                    self.scale(),
                    Some(&bias),
                )?)
            }
            SegmentMask::Full => match plan.key_valid_prefix(segment.kv_len) {
                Some(rows) => {
                    let bias = key_padding_bias(&rows, q.dtype(), q.device())?;
                    Ok(crate::attention::attention_with_bias(
                        &q_seg,
                        &k_seg,
                        &v_seg,
                        self.scale(),
                        Some(&bias),
                    )?)
                }
                None => self.full_unbiased(&q_seg, &k_seg, &v_seg),
            },
        }
    }

    /// An unmasked image block. Metal's fused SDPA handles the head widths it
    /// supports; everything else is the shared chunked math attention.
    pub(crate) fn full_unbiased(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let scale = self.scale();
        if self.fused_target
            && q.device().is_metal()
            && matches!(self.head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
        {
            return Ok(candle_nn::ops::sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                None,
                false,
                scale,
                1.0,
            )?);
        }
        Ok(crate::attention::attention_with_bias(q, k, v, scale, None)?)
    }
}

/// `[B, 1, q_len, kv_len]`: query row `i` sees key `j` iff
/// `j <= kv_len - q_len + i` and the key is not padding.
fn causal_bottom_right_bias(
    segment: &AttentionSegment,
    key_valid: Option<&[&[bool]]>,
    batch: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let (q_len, kv_len) = (segment.q_len, segment.kv_len);
    anyhow::ensure!(
        q_len > 0 && kv_len >= q_len,
        "Qwen Image 2.1 causal segment needs {q_len} queries within {kv_len} keys"
    );
    let offset = kv_len - q_len;
    let mut values = Vec::with_capacity(batch * q_len * kv_len);
    for row in 0..batch {
        let valid = key_valid.map(|rows| rows[row]);
        for query in 0..q_len {
            for key in 0..kv_len {
                let allowed = key <= offset + query && valid.is_none_or(|valid| valid[key]);
                values.push(if allowed { 0.0 } else { f32::NEG_INFINITY });
            }
        }
    }
    Ok(Tensor::from_vec(values, (batch, 1, q_len, kv_len), device)?.to_dtype(dtype)?)
}

/// `[B, 1, 1, kv_len]` additive key-padding mask.
fn key_padding_bias(rows: &[&[bool]], dtype: DType, device: &Device) -> Result<Tensor> {
    let kv_len = rows.first().map_or(0, |row| row.len());
    let values: Vec<f32> = rows
        .iter()
        .flat_map(|row| {
            row.iter()
                .map(|valid| if *valid { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    Ok(Tensor::from_vec(values, (rows.len(), 1, 1, kv_len), device)?.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::super::layout::QwenImage21JointLayout;
    use super::*;

    fn max_error(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// U4: executing the segment plan equals ONE dense attention under the
    /// full block-causal mask, uncached and cached, with and without padding.
    #[test]
    fn segment_plan_equals_dense_block_causal_attention() {
        let device = Device::Cpu;
        let slots = [false, false, false, true, true, false, false, false];
        for valid in [
            vec![vec![true; 8], vec![true; 8]],
            vec![
                vec![true, true, true, true, true, true, true, false],
                vec![true, true, true, true, true, true, false, false],
            ],
        ] {
            let layout = QwenImage21JointLayout::build(&slots, &valid, &[(2, 4)], (4, 4)).unwrap();
            let n = layout.total_len();
            let q = crate::engine::seeded_randn(7, &[2, 2, n, 8], &device, DType::F32).unwrap();
            let k = crate::engine::seeded_randn(8, &[2, 2, n, 8], &device, DType::F32).unwrap();
            let v = crate::engine::seeded_randn(9, &[2, 2, n, 8], &device, DType::F32).unwrap();
            let dispatch = SegmentDispatch {
                fused_target: false,
                head_dim: 8,
            };
            let dense = crate::attention::attention_with_bias(
                &q,
                &k,
                &v,
                dispatch.scale(),
                Some(&layout.dense_bias(2, &device).unwrap()),
            )
            .unwrap();
            let planned = dispatch
                .attend(&q, &k, &v, &layout.attention_plan(false))
                .unwrap();
            assert!(max_error(&planned, &dense) < 1e-5);

            // A cached step queries the target rows against every key.
            let prefix = layout.prefix_len();
            let target_q = q
                .narrow(2, prefix, n - prefix)
                .unwrap()
                .contiguous()
                .unwrap();
            let cached = dispatch
                .attend(&target_q, &k, &v, &layout.attention_plan(true))
                .unwrap();
            let dense_target = dense.narrow(2, prefix, n - prefix).unwrap();
            assert!(max_error(&cached, &dense_target) < 1e-5);
        }
    }

    #[test]
    fn causal_bias_is_bottom_right_aligned_and_masks_padding() {
        let segment = AttentionSegment {
            q_start: 3,
            q_len: 2,
            kv_len: 5,
            mask: SegmentMask::CausalBottomRight,
        };
        let valid: [&[bool]; 1] = [&[true, false, true, true, true]];
        let bias =
            causal_bottom_right_bias(&segment, Some(&valid), 1, DType::F32, &Device::Cpu).unwrap();
        let rows = bias
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let allowed: Vec<Vec<bool>> = rows
            .iter()
            .map(|row| row.iter().map(|value| *value == 0.0).collect())
            .collect();
        assert_eq!(
            allowed,
            vec![
                vec![true, false, true, true, false],
                vec![true, false, true, true, true]
            ]
        );
    }
}
