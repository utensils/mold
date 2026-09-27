//! Fused per-head query/key normalization and interleaved rotary embedding.
//!
//! Transformers that RMS-normalize each head of Q and K and then rotate them
//! with an interleaved (complex-pair) RoPE pay, in the composite form, one
//! pass for the norm, one to transpose `[B, S, H, D]` into `[B, H, S, D]`, one
//! to widen to F32, one to rotate and one to narrow back — five full passes
//! over a tensor that at a 2K canvas is 136 MB per projection per block.
//! [`rms_norm_rope_i`] does the same arithmetic in one read and one write on
//! CUDA, and IS that composite everywhere else.
//!
//! The definition, which the CUDA kernel reproduces element for element:
//!
//! ```text
//! n   = rms_norm(x, weight, eps)            // in x's dtype
//! out = rope_i(n.transpose(1, 2).to_f32(), cos, sin).to_dtype(x.dtype())
//! ```
//!
//! `cos`/`sin` are F32 `[S, D / 2]` tables shared by every batch row and
//! head; rotation is upstream's `apply_rotary_emb_qwen(.., use_real=False)`,
//! which rotates in float32 (diffusers `transformer_qwenimage21.py`).

use candle::{DType, Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda;

/// `x` `[B, S, H, D]` (the projection output, before any transpose),
/// `weight` `[D]` in `x`'s dtype, `cos`/`sin` F32 `[S, D / 2]`. Returns the
/// normalized, rotated heads as contiguous `[B, H, S, D]` in `x`'s dtype.
pub fn rms_norm_rope_i(
    x: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (_, seq, _, head_dim) = x.dims4()?;
    if weight.dims1()? != head_dim
        || cos.dims2()? != (seq, head_dim / 2)
        || sin.dims2()? != (seq, head_dim / 2)
        || head_dim % 2 != 0
    {
        candle::bail!(
            "rms_norm_rope_i: x {:?}, weight {:?}, cos {:?}, sin {:?}",
            x.shape(),
            weight.shape(),
            cos.shape(),
            sin.shape()
        );
    }
    if cos.dtype() != DType::F32 || sin.dtype() != DType::F32 || weight.dtype() != x.dtype() {
        candle::bail!("rms_norm_rope_i needs F32 tables and a weight in x's dtype");
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && cuda::supports(x.dtype(), head_dim) {
        return cuda::forward(x, weight, cos, sin, eps);
    }
    composite(x, weight, cos, sin, eps)
}

/// The definition: candle's own kernels, one pass each.
pub fn composite(
    x: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (batch, seq, heads, head_dim) = x.dims4()?;
    let normalized = candle_nn::ops::rms_norm(
        &x.contiguous()?.reshape((batch * seq * heads, head_dim))?,
        weight,
        eps,
    )?
    .reshape((batch, seq, heads, head_dim))?
    .transpose(1, 2)?
    .contiguous()?;
    candle_nn::rotary_emb::rope_i(
        &normalized.to_dtype(DType::F32)?,
        &cos.contiguous()?,
        &sin.contiguous()?,
    )?
    .to_dtype(x.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn inputs(device: &Device, dtype: DType) -> (Tensor, Tensor, Tensor, Tensor) {
        let (batch, seq, heads, head_dim) = (2, 37, 3, 128);
        let x = (Tensor::randn(0f32, 3.0, (batch, seq, heads, head_dim), device).unwrap())
            .to_dtype(dtype)
            .unwrap();
        let weight = Tensor::randn(1f32, 0.2, head_dim, device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let angles = Tensor::randn(0f32, 3.0, (seq, head_dim / 2), device).unwrap();
        (x, weight, angles.cos().unwrap(), angles.sin().unwrap())
    }

    /// On CPU the entry point is the composite itself.
    #[test]
    fn cpu_is_the_composite() {
        let (x, w, c, s) = inputs(&Device::Cpu, DType::F32);
        let fused = rms_norm_rope_i(&x, &w, &c, &s, 1e-6).unwrap();
        let reference = composite(&x, &w, &c, &s, 1e-6).unwrap();
        assert_eq!(fused.dims(), &[2, 3, 37, 128]);
        let diff = (fused - reference)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn shape_and_dtype_contracts_are_refused() {
        let (x, w, c, s) = inputs(&Device::Cpu, DType::F32);
        assert!(rms_norm_rope_i(&x, &w.narrow(0, 0, 64).unwrap(), &c, &s, 1e-6).is_err());
        assert!(rms_norm_rope_i(&x, &w, &c.to_dtype(DType::BF16).unwrap(), &s, 1e-6).is_err());
    }

    /// The CUDA kernel is the composite, bit for bit, in every dtype it
    /// takes. Skips without a CUDA device (CI has none).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_kernel_is_bitwise_the_composite() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let (x, w, c, s) = inputs(&device, dtype);
            let fused = cuda::forward(&x, &w, &c, &s, 1e-6).unwrap();
            let reference = composite(&x, &w, &c, &s, 1e-6).unwrap();
            assert_eq!(fused.dims(), reference.dims());
            assert_eq!(fused.dtype(), dtype);
            let bits = |t: &Tensor| {
                t.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let (got, want) = (bits(&fused), bits(&reference));
            let mismatches = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(mismatches, 0, "{dtype:?}: {mismatches} of {}", got.len());
        }
    }
}
