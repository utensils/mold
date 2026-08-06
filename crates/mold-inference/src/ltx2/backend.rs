use anyhow::Result;
use candle_core::{DType, Device};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ltx2Backend {
    Cpu,
    Cuda,
    Metal,
}

impl Ltx2Backend {
    fn from_availability(force_cpu: bool, has_cuda: bool, has_metal: bool) -> Self {
        if force_cpu {
            Self::Cpu
        } else if has_cuda {
            Self::Cuda
        } else if has_metal {
            Self::Metal
        } else {
            Self::Cpu
        }
    }

    pub(crate) fn detect() -> Self {
        let force_cpu = crate::runtime_env::value("MOLD_DEVICE")
            .map(|value| value.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        Self::from_availability(
            force_cpu,
            candle_core::utils::cuda_is_available(),
            candle_core::utils::metal_is_available(),
        )
    }

    pub(crate) fn ensure_supported(self) -> Result<()> {
        Ok(())
    }

    pub(crate) const fn compute_dtype(self) -> DType {
        match self {
            Self::Cuda | Self::Metal => DType::BF16,
            Self::Cpu => DType::F32,
        }
    }
}

/// LTX-2 uses a family-specific Metal precision policy.
///
/// The general image-family policy keeps Metal in F32 because older diffusion
/// paths accumulate visible BF16 error. The official LTX-2 Apple path instead
/// runs the transformer in BF16 and keeps numerically sensitive components,
/// such as the vocoder, in explicit F32 islands. Applying that policy only
/// here avoids doubling a 22B transformer's working set on unified memory.
pub(crate) fn compute_dtype(device: &Device) -> DType {
    let backend = if device.is_cuda() {
        Ltx2Backend::Cuda
    } else if device.is_metal() {
        Ltx2Backend::Metal
    } else {
        Ltx2Backend::Cpu
    };
    backend.compute_dtype()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_detection_prefers_forced_cpu() {
        assert_eq!(
            Ltx2Backend::from_availability(true, true, true),
            Ltx2Backend::Cpu
        );
    }

    #[test]
    fn backend_detection_prefers_cuda_over_metal() {
        assert_eq!(
            Ltx2Backend::from_availability(false, true, true),
            Ltx2Backend::Cuda
        );
    }

    #[test]
    fn backend_detection_uses_metal_when_no_cuda_is_available() {
        assert_eq!(
            Ltx2Backend::from_availability(false, false, true),
            Ltx2Backend::Metal
        );
    }

    #[test]
    fn metal_backend_is_available_for_correctness_inference() {
        Ltx2Backend::Metal.ensure_supported().unwrap();
    }

    #[test]
    fn accelerator_backends_use_bf16_compute() {
        assert_eq!(Ltx2Backend::Cuda.compute_dtype(), candle_core::DType::BF16);
        assert_eq!(Ltx2Backend::Metal.compute_dtype(), candle_core::DType::BF16);
        assert_eq!(Ltx2Backend::Cpu.compute_dtype(), candle_core::DType::F32);
    }
}
