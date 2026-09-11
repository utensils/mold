//! The fused interleaved-RoPE path shared by FLUX.1 and FLUX.2.
//!
//! Both families build their positional embedding as BFL does
//! (`flux/math.py:19-25`, `flux2/model.py`): a `[b, seq, head_dim/2, 2, 2]`
//! tensor whose innermost matrix is `[[cos, -sin], [sin, cos]]`, applied to
//! head-dim pairs as
//!
//! ```text
//! y0 = x0 * cos - x1 * sin
//! y1 = x0 * sin + x1 * cos
//! ```
//!
//! That is verbatim the contract candle's `rotary_emb::rope_i` documents
//! (`candle-nn/src/rotary_emb.rs:6-10`) and implements (`:262-288`), so the
//! three strided broadcast kernels each `apply_rope` runs today can collapse
//! into one fused launch over contiguous memory. Column `j = 0` of the
//! rotation matrix carries both factors: `cos = pe[.., 0, 0]`,
//! `sin = pe[.., 1, 0]`.
//!
//! The kernel is stricter than the broadcast form it replaces, so this module
//! answers `None` rather than failing whenever the request does not fit it —
//! the caller then runs the original arithmetic and the render is unaffected.

use candle_core::{DType, Result, Tensor};

/// The dtypes `rope_i` dispatches (`rotary_emb.rs:84-95` on CPU,
/// `:165-175` on CUDA). Anything else falls back.
fn rope_i_supports(dtype: DType) -> bool {
    matches!(dtype, DType::BF16 | DType::F16 | DType::F32 | DType::F64)
}

/// Apply FLUX's rotary embedding through candle's fused interleaved kernel.
///
/// Returns `Ok(None)` when `x` / `freq_cis` are not the shape, batching or
/// dtype the kernel accepts; the caller must then take its own broadcast
/// implementation, which has no such constraints.
///
/// Only the batch-shared embedding (`freq_cis.dim(0) == 1`) is routed: that is
/// what both engines build, and it maps onto the kernel's 2-D `cos`/`sin`
/// form, whose per-batch stride is zero (`rotary_emb.rs:141-145`). A genuinely
/// per-batch embedding falls back rather than guessing at the 3-D layout.
pub(crate) fn fused_interleaved_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Option<Tensor>> {
    let pe_dims = freq_cis.dims();
    if pe_dims.len() != 5 || pe_dims[0] != 1 || pe_dims[3] != 2 || pe_dims[4] != 2 {
        return Ok(None);
    }
    let Ok((_b, _heads, seq_len, head_dim)) = x.dims4() else {
        return Ok(None);
    };
    if pe_dims[1] != seq_len || pe_dims[2] * 2 != head_dim {
        return Ok(None);
    }
    if x.dtype() != freq_cis.dtype() || !rope_i_supports(x.dtype()) {
        return Ok(None);
    }

    // `[[cos, -sin], [sin, cos]]` — column 0 is `[cos, sin]`.
    let cos = freq_cis
        .narrow(3, 0, 1)?
        .narrow(4, 0, 1)?
        .squeeze(4)?
        .squeeze(3)?
        .squeeze(0)?
        .contiguous()?;
    let sin = freq_cis
        .narrow(3, 1, 1)?
        .narrow(4, 0, 1)?
        .squeeze(4)?
        .squeeze(3)?
        .squeeze(0)?
        .contiguous()?;
    let xs = x.contiguous()?;
    Ok(Some(candle_nn::rotary_emb::rope_i(&xs, &cos, &sin)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    /// The broadcast form every FLUX engine ships, reproduced here so the
    /// fused path is compared against the arithmetic it replaces rather than
    /// against one engine's copy of it.
    fn broadcast_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
        let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;
        let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
        let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
        (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims)
    }

    fn tiny_pe(seq: usize, head_dim: usize, device: &Device) -> Tensor {
        let angles = Tensor::arange(0f32, (seq * head_dim / 2) as f32, device)
            .unwrap()
            .affine(0.37, -1.1)
            .unwrap()
            .reshape((1, seq, head_dim / 2))
            .unwrap();
        let cos = angles.cos().unwrap();
        let sin = angles.sin().unwrap();
        Tensor::stack(&[&cos, &sin.neg().unwrap(), &sin, &cos], D::Minus1)
            .unwrap()
            .reshape((1, seq, head_dim / 2, 2, 2))
            .unwrap()
    }

    fn tiny_x(b: usize, heads: usize, seq: usize, head_dim: usize, device: &Device) -> Tensor {
        Tensor::arange(0f32, (b * heads * seq * head_dim) as f32, device)
            .unwrap()
            .affine(0.011, -0.4)
            .unwrap()
            .reshape((b, heads, seq, head_dim))
            .unwrap()
    }

    #[test]
    fn the_fused_kernel_reproduces_the_broadcast_rope() {
        let device = Device::Cpu;
        let (seq, head_dim) = (5usize, 8usize);
        let pe = tiny_pe(seq, head_dim, &device);
        let x = tiny_x(1, 2, seq, head_dim, &device);

        let want = broadcast_rope(&x, &pe).unwrap();
        let got = fused_interleaved_rope(&x, &pe)
            .unwrap()
            .expect("the tiny case is exactly the layout the kernel accepts");

        assert_eq!(got.dims(), want.dims());
        let diff = (&got - &want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "the fused rope diverged by {diff}");
    }

    /// A batch-shared embedding rotates every batch row identically, which is
    /// what lets a batched CFG forward keep one `pe`.
    #[test]
    fn a_batched_input_keeps_the_shared_embedding() {
        let device = Device::Cpu;
        let (seq, head_dim) = (4usize, 8usize);
        let pe = tiny_pe(seq, head_dim, &device);
        let x = tiny_x(2, 3, seq, head_dim, &device);

        let want = broadcast_rope(&x, &pe).unwrap();
        let got = fused_interleaved_rope(&x, &pe).unwrap().expect("routed");
        let diff = (&got - &want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "the batched fused rope diverged by {diff}");
    }

    /// Every shape the kernel cannot take must fall back, never fail: a
    /// mismatched dtype, a rank the engine does not build, and a sequence
    /// shorter than the embedding.
    #[test]
    fn an_unroutable_request_falls_back_instead_of_failing() {
        let device = Device::Cpu;
        let (seq, head_dim) = (4usize, 8usize);
        let pe = tiny_pe(seq, head_dim, &device);
        let x = tiny_x(1, 2, seq, head_dim, &device);

        let half = x.to_dtype(DType::F16).unwrap();
        assert!(
            fused_interleaved_rope(&half, &pe).unwrap().is_none(),
            "a dtype the embedding does not share must fall back"
        );
        let short = tiny_x(1, 2, seq - 1, head_dim, &device);
        assert!(
            fused_interleaved_rope(&short, &pe).unwrap().is_none(),
            "a sequence the embedding does not cover must fall back"
        );
        let rank3 = x.flatten_from(2).unwrap();
        assert!(
            fused_interleaved_rope(&rank3, &pe).unwrap().is_none(),
            "a rank the kernel does not take must fall back"
        );
    }
}
