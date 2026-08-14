//! Wan's family-scoped compute-dtype policy.
//!
//! The general policy (`crate::device::gpu_dtype`) keeps Metal in F32 because
//! older image-diffusion paths accumulate visible BF16 error there. Wan takes
//! the same family-scoped exception LTX-2 does (`crate::ltx2::backend`): the
//! DiT and VAE run BF16 on Metal, with the numerically sensitive pieces
//! already in explicit F32 islands — the sampler's host-side f64 solver, the
//! F32 timestep tensor, `gated_residual`'s F32 accumulation, the VAE norms'
//! F32 interior, and `CastBoundary`'s F32 activations into every quantized
//! linear (which candle's Metal QMatMul asserts on, so the cast is
//! load-bearing there, not just a VRAM choice).
//!
//! F32-on-Metal would also double the resident set on unified memory — a
//! 10 GB fp16 TI2V-5B file becomes ~20 GB before activations — and the
//! quantized arm (`WanWeights::dtype`) already carries BF16 between ops on
//! every non-CPU device, so BF16 here makes the dense and quantized paths
//! agree instead of diverging by weight format.

use candle_core::{DType, Device};

/// Resolve Wan's compute dtype for the device the render selected.
pub(crate) fn compute_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        // candle's CPU backend has no BF16 matmul.
        DType::F32
    } else {
        DType::BF16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_computes_in_f32() {
        assert_eq!(compute_dtype(&Device::Cpu), DType::F32);
    }

    /// Metal gets the same BF16 the quantized arm already uses on non-CPU
    /// devices, matching the LTX-2 family-scoped Metal policy rather than the
    /// image-family F32 default.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_computes_in_bf16() {
        let metal = Device::new_metal(0).unwrap();
        assert_eq!(compute_dtype(&metal), DType::BF16);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_computes_in_bf16() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        assert_eq!(compute_dtype(&cuda), DType::BF16);
    }
}
