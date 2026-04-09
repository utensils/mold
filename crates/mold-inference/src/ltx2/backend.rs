use anyhow::{bail, Result};

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
        let force_cpu = std::env::var("MOLD_DEVICE")
            .map(|value| value.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        Self::from_availability(
            force_cpu,
            candle_core::utils::cuda_is_available(),
            candle_core::utils::metal_is_available(),
        )
    }

    pub(crate) fn ensure_supported(self) -> Result<()> {
        if self == Self::Metal {
            bail!(
                "LTX-2 / LTX-2.3 is not supported on Metal yet; use CUDA for real inference or set MOLD_DEVICE=cpu for correctness-only fallback"
            );
        }
        Ok(())
    }
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
    fn metal_backend_returns_clear_error() {
        let err = Ltx2Backend::Metal.ensure_supported().unwrap_err();
        assert!(err.to_string().contains("not supported on Metal"));
    }
}
