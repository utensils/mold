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
//!
//! Three backends, chosen by [`TargetAttention`] from the resolved
//! [`super::exec_path::Qwen21ExecPath`]:
//!
//! - `Legacy` — v0.32: every segment through `attention::attention_with_bias`
//!   (image-policy math, scale on the scores). The byte-identity reference.
//! - `MetalSdpa` — Metal's fused SDPA for unbiased `Full` segments; the rest
//!   is `Legacy`. Metal's shipped path, unchanged.
//! - `FastStill` — FlashAttention wherever the kernel is compiled in and the
//!   tensors are eligible (CUDA BF16/F16): a `CausalBottomRight` text segment
//!   is `flash_attn_windowed(.., None, Some(0))`, whose causal mask is
//!   bottom-right aligned exactly like the segment's; an unpadded `Full`
//!   segment is plain `flash_attn`; a padded `Full` segment (batched CFG with
//!   prompts of different lengths) packs each row's valid keys into ONE
//!   `flash_attn_varlen` call. A padded causal segment — only a batched
//!   prefill's text rows, run once per request — keeps the biased math path
//!   under the FastStill scale placement, as does every ineligible tensor.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::exec_path::TargetAttention;
use super::layout::{AttentionSegment, BlockCausalPlan, SegmentMask};
use crate::attention::AttentionPolicy;

/// Per-attention-module dispatch configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmentDispatch {
    /// The backend family, from the resolved execution path.
    pub attention: TargetAttention,
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
        if self.attention == TargetAttention::FastStill {
            return self.fast_still_segment(&q_seg, &k_seg, &v_seg, segment, plan);
        }
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

    /// The `FastStill` arm of [`Self::attend_segment`], on already-sliced
    /// segment tensors.
    fn fast_still_segment(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        segment: &AttentionSegment,
        plan: &BlockCausalPlan,
    ) -> Result<Tensor> {
        const POLICY: AttentionPolicy = AttentionPolicy::FastStill;
        let scale = self.scale();
        let padding = plan.key_valid_prefix(segment.kv_len);
        let flash = crate::attention::takes_flash(POLICY, q);
        match (segment.mask, padding) {
            (SegmentMask::CausalBottomRight, None) if flash => {
                Ok(crate::attention::flash_causal_bottom_right(q, k, v, scale)?)
            }
            (SegmentMask::Full, None) => Ok(crate::attention::attention_with_bias_for(
                POLICY, q, k, v, scale, None,
            )?),
            (SegmentMask::Full, Some(rows)) if flash => {
                let key_index: Vec<Vec<u32>> = rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(index, valid)| valid.then_some(index as u32))
                            .collect()
                    })
                    .collect();
                Ok(crate::attention::flash_varlen_keys(
                    q, k, v, &key_index, scale,
                )?)
            }
            (SegmentMask::Full, Some(rows)) => {
                let bias = key_padding_bias(&rows, q.dtype(), q.device())?;
                Ok(crate::attention::attention_with_bias_for(
                    POLICY,
                    q,
                    k,
                    v,
                    scale,
                    Some(&bias),
                )?)
            }
            (SegmentMask::CausalBottomRight, padding) => {
                let bias = causal_bottom_right_bias(
                    segment,
                    padding.as_deref(),
                    q.dim(0)?,
                    q.dtype(),
                    q.device(),
                )?;
                Ok(crate::attention::attention_with_bias_for(
                    POLICY,
                    q,
                    k,
                    v,
                    scale,
                    Some(&bias),
                )?)
            }
        }
    }

    /// An unmasked image block. Metal's fused SDPA handles the head widths it
    /// supports; everything else is the shared chunked math attention.
    pub(crate) fn full_unbiased(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let scale = self.scale();
        if self.attention == TargetAttention::MetalSdpa
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
            // Every backend family: on CPU `FastStill` is its math fallback
            // (scale folded into K), which must describe the same attention.
            for attention in [TargetAttention::Legacy, TargetAttention::FastStill] {
                let dispatch = SegmentDispatch {
                    attention,
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
                assert!(max_error(&planned, &dense) < 1e-5, "{attention:?}");

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
                assert!(max_error(&cached, &dense_target) < 1e-5, "{attention:?}");
            }
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

    /// The CUDA FastStill arms against the v0.32 math prefill, on a layout
    /// with every segment kind (a causal text run split by a condition
    /// image, the condition block, the target), uncached and cached, with
    /// and without padded keys. Flash accumulates in F32 over BF16 inputs,
    /// so it matches a math reference evaluated in F32 on the same BF16
    /// values to BF16 rounding, and it must actually be the flash kernels
    /// that ran. Ignored by default: it needs a CUDA device and panics without one.
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn fast_still_flash_segments_match_the_math_prefill_on_cuda() {
        let device = Device::new_cuda(0).expect("this test needs a CUDA device");
        let slots = [
            false, false, false, false, true, true, false, false, false, false,
        ];
        for valid in [
            vec![vec![true; 10], vec![true; 10]],
            vec![
                vec![true, true, true, true, true, true, true, true, true, false],
                vec![
                    true, true, true, true, true, true, true, false, false, false,
                ],
            ],
        ] {
            let layout =
                QwenImage21JointLayout::build(&slots, &valid, &[(2, 4)], (12, 16)).unwrap();
            let n = layout.total_len();
            let (heads, d) = (4, 128);
            let bf16 = |seed| {
                crate::engine::seeded_randn(seed, &[2, heads, n, d], &device, DType::BF16).unwrap()
            };
            let (q, k, v) = (bf16(31), bf16(32), bf16(33));
            let f32 = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
            assert!(crate::attention::takes_flash(
                crate::attention::AttentionPolicy::FastStill,
                &q
            ));
            let legacy = SegmentDispatch {
                attention: TargetAttention::Legacy,
                head_dim: d,
            };
            let fast = SegmentDispatch {
                attention: TargetAttention::FastStill,
                head_dim: d,
            };
            for cached in [false, true] {
                let plan = layout.attention_plan(cached);
                let queries = if cached {
                    let prefix = layout.prefix_len();
                    q.narrow(2, prefix, n - prefix)
                        .unwrap()
                        .contiguous()
                        .unwrap()
                } else {
                    q.clone()
                };
                let reference = legacy
                    .attend(&f32(&queries), &f32(&k), &f32(&v), &plan)
                    .unwrap();
                // Every full block (padded or not) and every unpadded causal
                // text run has a flash arm; only a causal run over padded
                // keys builds a math bias. `takes_flash` alone proves
                // nothing, so count the kernels that actually launched.
                let flash_segments = plan
                    .segments
                    .iter()
                    .filter(|segment| {
                        segment.mask == SegmentMask::Full
                            || plan.key_valid_prefix(segment.kv_len).is_none()
                    })
                    .count() as u64;
                assert!(flash_segments > 0);
                let dispatches = crate::attention::flash_dispatch_count();
                let actual = fast.attend(&queries, &k, &v, &plan).unwrap();
                assert_eq!(
                    crate::attention::flash_dispatch_count() - dispatches,
                    flash_segments,
                    "cached={cached}: a flash-eligible segment did not run FlashAttention"
                );
                assert_eq!(actual.dims(), reference.dims());
                assert_eq!(actual.dtype(), DType::BF16);
                let error = max_error(&f32(&actual), &reference);
                assert!(
                    error < 2e-2,
                    "cached={cached} padded={}: {error}",
                    valid[1].iter().any(|valid| !valid)
                );
            }
        }
    }
}
