//! MiniMax H3's family-scoped compute-dtype policy.
//!
//! The general policy (`crate::device::gpu_dtype`) keeps Metal in F32 because
//! older image-diffusion paths accumulate visible BF16 error there. H3 takes
//! the same family-scoped exception LTX-2 and Wan do (`crate::ltx2::backend`,
//! `crate::wan::backend`): the DiT runs BF16 on Metal and CUDA alike, with the
//! numerically sensitive pieces already sitting in explicit F32 islands.
//!
//! H3's islands are not the same ones Wan has, and naming them is the whole
//! justification for the exception:
//!
//! - **Attention.** Both dense arms in `mold_candle::minimax_h3::attention`
//!   promote Q, K, and V to F32 before the score matmul and softmax, and cast
//!   back only at the end. The Metal chunked arm inherits that unchanged, so
//!   the precision of a chunked pass equals the precision of the unchunked one
//!   — chunking narrows a buffer, it does not lower a dtype.
//! - **Quantized linears.** The INT8 ConvRot portable arm — the only arm Metal
//!   can take — computes its per-row activation scale in F32, accumulates the
//!   signed matmul in F32, and applies both scales in F32 before casting to the
//!   output dtype. That is H3's equivalent of Wan's `CastBoundary`: the cast
//!   into the quantized op is load-bearing, not a VRAM choice, and the arm is
//!   written so no BF16 value reaches the accumulator.
//! - **The audio VAE.** It requires F32 outright (`require_f32`), so it is
//!   unaffected by this policy in either direction.
//!
//! F32 for the transformer would also roughly double the resident set on
//! unified memory, where the compact stack is already ~42.5 GB — the one
//! resource that makes Apple Silicon viable for H3 at all.
//!
//! CPU stays refused. This is a real capability limit, not a licence gate: the
//! CPU has no qualified H3 route, and candle's CPU backend has no BF16 matmul,
//! so the fallback would be an unqualified F32 path rather than the checkpoint
//! the manifests describe.

use anyhow::{bail, Result};
use candle_core::{DType, Device};

/// Resolve H3's compute dtype for the device the render selected.
///
/// This must agree with `H3PrecisionProfile::OfficialMixedBf16F32`, which is
/// what the checkpoint's own weights are stored in; the pinning test below is
/// the guard against the two drifting.
pub(crate) fn compute_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        // Unreachable through `ensure_supported`, and F32 rather than BF16
        // because candle's CPU backend has no BF16 matmul.
        DType::F32
    } else {
        DType::BF16
    }
}

/// Refuse a device H3 has no qualified route for, by name.
pub(crate) fn ensure_supported(device: &Device) -> Result<()> {
    if device.is_cpu() {
        bail!(
            "MiniMax H3 has no CPU execution route — the compact stack is qualified on CUDA and, \
             for correctness only, on Apple Metal. Select a GPU device."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_candle::minimax_h3::H3PrecisionProfile;

    /// The device policy must not be able to drift from the dtype the
    /// checkpoint's own weights are stored in.
    #[test]
    fn the_gpu_dtype_is_the_official_checkpoint_precision() {
        assert_eq!(
            H3PrecisionProfile::OfficialMixedBf16F32.compute_dtype(),
            DType::BF16
        );
    }

    #[test]
    fn cpu_is_refused_by_name_and_never_silently_downgraded() {
        let error = ensure_supported(&Device::Cpu).unwrap_err().to_string();
        assert!(error.contains("CPU"), "{error}");
        assert!(error.contains("Metal"), "{error}");
        assert_eq!(compute_dtype(&Device::Cpu), DType::F32);
    }

    /// Metal takes the family-scoped BF16 exception rather than the image
    /// family's F32-on-Metal default: F32 would roughly double a 42.5 GB
    /// resident stack on unified memory, and H3's sensitive arithmetic is
    /// already in explicit F32 islands.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_computes_in_bf16() {
        let metal = Device::new_metal(0).unwrap();
        assert_eq!(compute_dtype(&metal), DType::BF16);
        assert!(ensure_supported(&metal).is_ok());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_computes_in_bf16() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        assert_eq!(compute_dtype(&cuda), DType::BF16);
        assert!(ensure_supported(&cuda).is_ok());
    }
}
