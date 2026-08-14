//! The one joint-attention dispatcher for every Qwen-Image transformer.
//!
//! The BF16, quantized and offloaded engines used to carry three hand-rolled
//! copies of `QK^T → mask → softmax → V`, and they had drifted: the BF16 Metal
//! arm passed `None` to `candle_nn::ops::sdpa` while holding a real key mask
//! (so batched CFG over two different prompt lengths was silently unmasked),
//! and the offloaded arm never reached `crate::attention` at all, so
//! `MOLD_ATTN` / `MOLD_ATTN_CHUNK` did not apply to it.
//!
//! Both problems are structural, so there is now exactly one implementation:
//! Metal keeps candle's fused `sdpa` (the family's Metal correctness path),
//! and CUDA/CPU route through `crate::attention`.
//!
//! The bias itself is usually absent. Text conditioning arrives unpadded from
//! `Qwen2TextEncoder`, and the pipeline only pads when batched CFG puts two
//! different prompt lengths in one forward — mirroring diffusers'
//! `pipeline_qwenimage.py:265-266` (`if prompt_embeds_mask.all(): mask = None`).

use candle_core::{DType, Device, Result, Tensor, D};

/// Additive-bias floor for candle's fused Metal `sdpa`.
///
/// The math path takes a true `-inf` and lets `softmax` produce an exact zero
/// weight, but the fused kernel's online softmax turns `-inf` into `NaN`. The
/// quantized Metal arm has always used `(mask - 1) * 1e9` for exactly this
/// reason, so the floor keeps that behaviour bit-for-bit.
const SDPA_BIAS_FLOOR: f64 = -1e9;

/// `[text, image]` joint key bias, or `None` when the text stream is unpadded.
///
/// `txt_mask` is the u8 `[B, T]` mask; `None` means "no padding", which is the
/// common case and the one that lets attention take its fastest path. The
/// image half is never masked, so the returned bias is
/// `[B, 1, 1, T + img_seq_len]` with `0` on a live key and `-inf` on a pad.
pub(super) fn joint_key_bias(
    txt_mask: Option<&Tensor>,
    img_seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    let Some(txt_mask) = txt_mask else {
        return Ok(None);
    };
    let txt_mask = txt_mask.to_device(device)?;
    let batch = txt_mask.dim(0)?;
    let img_mask = Tensor::ones((batch, img_seq_len), txt_mask.dtype(), device)?;
    // Key order is [text, image] to match the Q/K/V concatenation.
    let key_mask = Tensor::cat(&[&txt_mask, &img_mask], 1)?
        .unsqueeze(1)?
        .unsqueeze(1)?;
    let on_true = key_mask.zeros_like()?.to_dtype(dtype)?;
    let on_false = Tensor::new(f32::NEG_INFINITY, device)?
        .broadcast_as(key_mask.shape())?
        .to_dtype(dtype)?;
    Ok(Some(key_mask.where_cond(&on_true, &on_false)?))
}

/// Joint attention for all three Qwen-Image transformers.
///
/// Metal keeps candle's fused `sdpa`; CUDA and CPU go through
/// `crate::attention`, so `MOLD_ATTN` and `MOLD_ATTN_CHUNK` apply uniformly.
pub(super) fn joint_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if q.device().is_metal() {
        let mask = bias.map(|bias| sdpa_bias(bias, q)).transpose()?;
        candle_nn::ops::sdpa(q, k, v, mask.as_ref(), false, scale as f32, 1.0)
    } else {
        crate::attention::attention_with_bias(q, k, v, scale as f32, bias)
    }
}

/// `[B, 1, 1, K]` additive bias expanded to the `[B, H, Q, K]` candle's Metal
/// `sdpa` wants, with `-inf` floored to [`SDPA_BIAS_FLOOR`].
fn sdpa_bias(bias: &Tensor, q: &Tensor) -> Result<Tensor> {
    let (batch, heads, seq_len, _) = q.dims4()?;
    let key_len = bias.dim(D::Minus1)?;
    bias.to_dtype(q.dtype())?
        .clamp(SDPA_BIAS_FLOOR, f64::MAX)?
        .broadcast_as((batch, heads, seq_len, key_len))?
        .contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_key_bias_is_none_without_padding() {
        let bias = joint_key_bias(None, 4, DType::F32, &Device::Cpu).unwrap();
        assert!(
            bias.is_none(),
            "unpadded text must produce no bias so attention takes its fast path"
        );
    }

    #[test]
    fn joint_key_bias_masks_only_text_padding() {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(vec![1u8, 1, 0], (1, 3), &device).unwrap();
        let bias = joint_key_bias(Some(&mask), 2, DType::F32, &device)
            .unwrap()
            .expect("a padded mask must produce a bias");

        assert_eq!(bias.dims(), &[1, 1, 1, 5]);
        let values = bias.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 0.0);
        assert!(values[2].is_infinite() && values[2] < 0.0);
        // The image half is never masked.
        assert_eq!(values[3], 0.0);
        assert_eq!(values[4], 0.0);
    }

    /// Batched CFG stacks the two prompts on the batch axis, so the bias has
    /// to stay per-row rather than collapsing to one shared mask.
    #[test]
    fn joint_key_bias_keeps_per_batch_rows() {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(vec![1u8, 1, 1, 1, 0, 0], (2, 3), &device).unwrap();
        let bias = joint_key_bias(Some(&mask), 1, DType::F32, &device)
            .unwrap()
            .unwrap();
        assert_eq!(bias.dims(), &[2, 1, 1, 4]);
        let values = bias.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values[..4].iter().all(|v| *v == 0.0));
        assert_eq!(values[4], 0.0);
        assert!(values[5].is_infinite());
        assert!(values[6].is_infinite());
        assert_eq!(values[7], 0.0);
    }

    /// Off Metal the dispatcher is `crate::attention`, so a `None` bias is the
    /// ordinary unmasked path and the masked path matches slicing.
    #[test]
    fn joint_attention_without_bias_is_plain_attention() {
        let device = Device::Cpu;
        let mk = || Tensor::randn(0.0_f32, 1.0_f32, (1, 2, 6, 8), &device).unwrap();
        let (q, k, v) = (mk(), mk(), mk());
        let scale = 1.0 / (8f64).sqrt();
        let got = joint_attention(&q, &k, &v, scale, None).unwrap();
        let want = crate::attention::attention(&q, &k, &v, scale as f32).unwrap();
        let diff = (got - want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0);
    }
}
