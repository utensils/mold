//! Shared quantized-linear dispatch for GGUF-backed engines.
//!
//! Qwen-Image discovered the contract (`docs/architecture/qwen-mmq-nan.md`),
//! Z-Image confirmed it independently, and LTX-2.5 is the third consumer, so
//! the pure halves live here rather than as a third copy: candle's CUDA
//! MMQ/MMVQ fast paths accept only some weights (`cuda_mmq_block_size` +
//! [`select_linear_kind`]) and only some activations per forward
//! ([`qmatmul_forward_supported`]), and every decline lands in
//! `dequantize_matmul`, which reads the activation as `f32` and therefore
//! errors on BF16. Each family keeps its own env name
//! (`MOLD_QWEN_QMATMUL` / `MOLD_ZIMAGE_QMATMUL` / `MOLD_LTX2_QMATMUL`) and
//! its own working-dtype policy; the tables and the forward rules are one.
//!
//! Default on CUDA is the dequant arm for every family that has asked so
//! far: candle's fast MMQ kernels returned non-finite values for both
//! Qwen-Image (100% NaN) and Z-Image (solid-black renders) — #1048 is the
//! open kernel investigation — so a new family must opt in per render via
//! its env flag rather than ship the fast path untested.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use mold_candle::quantized_nn::Linear as QMatMulLinear;

/// Device class a linear resolves to, so the arm decision stays a pure
/// function that can be exercised for CUDA without a CUDA device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinearDevice {
    Cuda,
    Metal,
    Other,
}

impl LinearDevice {
    pub(crate) fn of(device: &Device) -> Self {
        if device.is_cuda() {
            Self::Cuda
        } else if device.is_metal() {
            Self::Metal
        } else {
            Self::Other
        }
    }
}

/// Which implementation a quantized linear resolves to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QuantizedLinearKind {
    /// Weight stays quantized; candle's kernels consume it directly.
    QMatMul,
    /// Full per-forward dequantization to the working dtype.
    Dequant,
}

/// `qk` — the block quantization size candle's MMQ kernel requires the
/// activation's `k` (the weight's `in_features`) to be a multiple of
/// (`candle-core/src/quantized/fast_mmq.rs`, `qk_for` plus the `k % qk != 0`
/// decline). `None` for a GGML dtype `fast_mmq::supports` does not accept at
/// all, which is the same answer: anything the kernels decline falls through
/// to `dequantize_matmul`, which reads the activation as `f32` and so errors
/// on BF16 activations — those weights must keep the per-forward dequant arm.
///
/// This is candle's *weight-side* table only. Both fast paths also require a
/// contiguous, rank-2-or-3 activation, and `fast_mmvq` declines past 8 rows.
/// Every one of those declines lands in the same BF16-hostile fallback, so
/// each is gated somewhere — the weight-side ones here through
/// [`select_linear_kind`], the activation-shaped ones per forward through
/// [`qmatmul_forward_supported`].
pub(crate) fn cuda_mmq_block_size(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            Some(32)
        }
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => {
            Some(256)
        }
        _ => None,
    }
}

/// One parser for the per-family `MOLD_*_QMATMUL` flags: a truthy value opts
/// CUDA back into candle's quantized fast path. Unset — or a value we do not
/// understand — keeps the shipped default, so a typo degrades to the arm the
/// family qualified rather than the one under investigation. The falsey set
/// is spelled out because `false`/`off`/`no` are what a user reaching for a
/// kill switch actually types.
pub(crate) fn parse_qmatmul_flag(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

/// Every shape-fixed half of the linear-arm decision, as a pure function.
///
/// Metal has always used `QMatMul`. CUDA joins it only when every kernel
/// precondition a weight can settle holds: the weight's GGML dtype is one the
/// MMQ/MMVQ kernels accept, its row width is a multiple of that dtype's MMQ
/// block size, it is resident on the device the activations live on (a
/// CPU-staged weight would hit candle's `unreachable!` in the CUDA matmul),
/// candle's process-global `FORCE_DMMV` switch is not already routing every
/// quantized matmul into the fallback, and the family's escape hatch opted
/// in. Everything else — CPU included — dequantizes per forward.
///
/// `FORCE_DMMV` is a *process* switch that mold itself flips
/// (`MOLD_WAN_FORCE_DMMV=1`, from Wan's denoise loop) and never clears, so it
/// can also flip after this decision is made; [`qmatmul_forward_supported`]
/// is the per-forward half that catches an engine built before the flip.
pub(crate) fn select_linear_kind(
    device: LinearDevice,
    weight_dtype: GgmlDType,
    weight_in_features: usize,
    weight_on_target_device: bool,
    qmatmul_enabled: bool,
    force_dmmv: bool,
) -> QuantizedLinearKind {
    match device {
        LinearDevice::Metal => QuantizedLinearKind::QMatMul,
        LinearDevice::Cuda
            if qmatmul_enabled
                && !force_dmmv
                && weight_on_target_device
                && cuda_mmq_block_size(weight_dtype)
                    .is_some_and(|qk| weight_in_features.is_multiple_of(qk)) =>
        {
            QuantizedLinearKind::QMatMul
        }
        _ => QuantizedLinearKind::Dequant,
    }
}

/// The activation-shaped half of the same decision, asked once per forward.
///
/// Candle's CUDA fast paths decline an activation whose rank is neither 2 nor
/// 3 (`fast_mmq.rs` / `fast_mmvq.rs`, `match rhs_l.shape().dims()`), and skip
/// both kernels outright while `FORCE_DMMV` is set. Either decline lands in
/// `dequantize_matmul`, which reads the activation as `f32` and rejects BF16
/// — so a `QMatMul` arm degrades to a dequantized forward for that call
/// rather than handing candle a matmul it will refuse. Metal and CPU are
/// unaffected: neither reads `FORCE_DMMV`, and candle's CPU `QMatMul` takes
/// any rank ≥ 2.
pub(crate) fn qmatmul_forward_supported(
    device: LinearDevice,
    activation_rank: usize,
    force_dmmv: bool,
) -> bool {
    match device {
        LinearDevice::Cuda => !force_dmmv && matches!(activation_rank, 2 | 3),
        _ => true,
    }
}

/// Dequantize `weight` to `dtype` and apply it densely, staging through the
/// CPU when the weight is CPU-resident and the activation is not (a
/// CPU-staged `QTensor` meeting the CUDA matmul is an `unreachable!` panic in
/// candle). Shared by every family's `Dequant` arm and by a `QMatMul` layer
/// whose forward candle would decline.
pub(crate) fn dequant_forward(
    weight: &QTensor,
    bias: Option<&Tensor>,
    x: &Tensor,
    dtype: DType,
) -> Result<Tensor> {
    let x = if x.dtype() == dtype {
        x.clone()
    } else {
        x.to_dtype(dtype)?
    };
    let w = if weight.device().is_cpu() && !x.device().is_cpu() {
        weight
            .dequantize(&Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(x.device())?
    } else {
        weight.dequantize(x.device())?.to_dtype(dtype)?
    };
    let bias = bias
        .map(|b| b.to_device(x.device())?.to_dtype(dtype))
        .transpose()?;
    candle_nn::Linear::new(w, bias).forward(&x)
}

/// The weight and bias a `QMatMul` arm falls back on for one forward candle's
/// CUDA kernels decline (rank, or `FORCE_DMMV`).
///
/// It is `Option`al on that arm rather than unconditional because retaining
/// it is not always free: for a float-stored GGML dtype `QMatMul::from_arc`
/// dequantizes eagerly and drops the `QTensor`, so holding the `Arc` anyway
/// would keep bytes candle had just released. That case is Metal's — CUDA
/// sends float-stored weights down the dequant arm — and Metal never declines
/// a forward, so it never needs the fallback either.
#[derive(Clone)]
pub(crate) struct DequantFallback {
    pub(crate) weight: Arc<QTensor>,
    pub(crate) bias: Option<Tensor>,
}

// The LTX-2.5 GGUF linear arm (the following commit) is the first
// consumer of the assembled struct; the pure functions above are live
// through Qwen-Image and Z-Image already.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone)]
enum QuantizedLinearArm {
    /// Per-forward dequantization to the kernel dtype.
    Dequant {
        weight: Arc<QTensor>,
        bias: Option<Tensor>,
    },
    /// QMatMul-backed — the weight stays quantized on the device.
    QMatMul {
        inner: QMatMulLinear,
        fallback: Option<DequantFallback>,
    },
}

/// Device-dispatched quantized linear with the Wan `CastBoundary` rule: the
/// activation is normalized to `kernel_dtype` for the kernel (the CUDA
/// kernels return the dtype they were fed, and the bias is materialized at
/// `kernel_dtype`), and the output is cast back to the caller's dtype so the
/// arm choice never leaks into the surrounding model's dtype flow.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone)]
pub(crate) struct QuantizedLinear {
    arm: QuantizedLinearArm,
    kernel_dtype: DType,
}

impl std::fmt::Debug for QuantizedLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedLinear")
            .field("kind", &self.kind())
            .field("kernel_dtype", &self.kernel_dtype)
            .finish()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl QuantizedLinear {
    /// Build a linear over `weight`, choosing the arm through
    /// [`select_linear_kind`] for `device`. `bias` is expected dense
    /// (already dequantized) in any dtype; it is materialized at
    /// `kernel_dtype`.
    pub(crate) fn new(
        weight: Arc<QTensor>,
        bias: Option<Tensor>,
        device: &Device,
        kernel_dtype: DType,
        qmatmul_enabled: bool,
    ) -> Result<Self> {
        let bias = bias
            .map(|b| b.to_device(device)?.to_dtype(kernel_dtype))
            .transpose()?;
        let in_features = weight.shape().dims().last().copied().unwrap_or_default();
        let linear_device = LinearDevice::of(device);
        let arm = match select_linear_kind(
            linear_device,
            weight.dtype(),
            in_features,
            weight.device().same_device(device),
            qmatmul_enabled,
            crate::quantized_dmmv::force_dmmv_enabled(),
        ) {
            QuantizedLinearKind::QMatMul => {
                let fallback = (linear_device == LinearDevice::Cuda).then(|| DequantFallback {
                    weight: weight.clone(),
                    bias: bias.clone(),
                });
                QuantizedLinearArm::QMatMul {
                    inner: QMatMulLinear::from_arc(weight, bias)?,
                    fallback,
                }
            }
            QuantizedLinearKind::Dequant => QuantizedLinearArm::Dequant { weight, bias },
        };
        Ok(Self { arm, kernel_dtype })
    }

    /// The arm this linear resolved to at construction.
    pub(crate) fn kind(&self) -> QuantizedLinearKind {
        match &self.arm {
            QuantizedLinearArm::Dequant { .. } => QuantizedLinearKind::Dequant,
            QuantizedLinearArm::QMatMul { .. } => QuantizedLinearKind::QMatMul,
        }
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out_dtype = xs.dtype();
        let out = match &self.arm {
            QuantizedLinearArm::Dequant { weight, bias } => {
                dequant_forward(weight, bias.as_ref(), xs, self.kernel_dtype)?
            }
            QuantizedLinearArm::QMatMul { inner, fallback } => {
                if !qmatmul_forward_supported(
                    LinearDevice::of(xs.device()),
                    xs.rank(),
                    crate::quantized_dmmv::force_dmmv_enabled(),
                ) {
                    let Some(fallback) = fallback else {
                        // Unreachable: only CUDA declines, and only CUDA
                        // retains a fallback. Naming it beats an `unwrap`.
                        candle_core::bail!(
                            "candle declined a quantized forward on {:?}, which kept no dequant fallback",
                            xs.device()
                        );
                    };
                    dequant_forward(
                        &fallback.weight,
                        fallback.bias.as_ref(),
                        xs,
                        self.kernel_dtype,
                    )?
                } else {
                    let xs = if xs.dtype() == self.kernel_dtype {
                        xs.clone()
                    } else {
                        xs.to_dtype(self.kernel_dtype)?
                    };
                    // Both CUDA fast paths decline a non-contiguous rhs, and
                    // the fallback they decline into cannot read BF16.
                    if xs.is_contiguous() {
                        inner.forward(&xs)?
                    } else {
                        inner.forward(&xs.contiguous()?)?
                    }
                }
            }
        };
        if out.dtype() == out_dtype {
            Ok(out)
        } else {
            out.to_dtype(out_dtype)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantized(out_dim: usize, in_dim: usize, dtype: GgmlDType) -> Arc<QTensor> {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.125)
            .collect();
        let dense = Tensor::from_vec(values, (out_dim, in_dim), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&dense, dtype).unwrap())
    }

    /// The K-quant and legacy-quant block sizes match candle's weight-side
    /// table; float storage and `IQ*`-class types answer `None`.
    #[test]
    fn cuda_mmq_block_sizes_match_candles_table() {
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
        ] {
            assert_eq!(cuda_mmq_block_size(dtype), Some(32), "{dtype:?}");
        }
        for dtype in [
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            assert_eq!(cuda_mmq_block_size(dtype), Some(256), "{dtype:?}");
        }
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q8_1,
            GgmlDType::Q8K,
        ] {
            assert_eq!(cuda_mmq_block_size(dtype), None, "{dtype:?}");
        }
    }

    /// The LTX-2.5 GGUF tiers' accepted quantized dtypes are all
    /// MMQ-eligible at the checkpoints' own row widths (every quantized
    /// tensor in the real files has `in_features % 256 == 0`), so the arm
    /// choice is genuinely the env flag's, never a silent decline.
    #[test]
    fn ltx25_accepted_dtypes_over_devices_and_flags() {
        for dtype in [
            GgmlDType::Q8_0,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            // CUDA default: dequant, per the Qwen/Z-Image NaN precedent.
            assert_eq!(
                select_linear_kind(LinearDevice::Cuda, dtype, 4096, true, false, false),
                QuantizedLinearKind::Dequant,
                "{dtype:?}"
            );
            // CUDA opted in: the fast path, every precondition holding.
            assert_eq!(
                select_linear_kind(LinearDevice::Cuda, dtype, 4096, true, true, false),
                QuantizedLinearKind::QMatMul,
                "{dtype:?}"
            );
            // FORCE_DMMV routes everything into the f32-only fallback, so the
            // fast path must not be chosen even when opted in.
            assert_eq!(
                select_linear_kind(LinearDevice::Cuda, dtype, 4096, true, true, true),
                QuantizedLinearKind::Dequant,
                "{dtype:?}"
            );
            // A CPU-staged weight would hit candle's `unreachable!`.
            assert_eq!(
                select_linear_kind(LinearDevice::Cuda, dtype, 4096, false, true, false),
                QuantizedLinearKind::Dequant,
                "{dtype:?}"
            );
            // Metal keeps QMatMul regardless of the flag; CPU dequantizes.
            for enabled in [false, true] {
                assert_eq!(
                    select_linear_kind(LinearDevice::Metal, dtype, 4096, true, enabled, false),
                    QuantizedLinearKind::QMatMul,
                    "{dtype:?}"
                );
                assert_eq!(
                    select_linear_kind(LinearDevice::Other, dtype, 4096, true, enabled, false),
                    QuantizedLinearKind::Dequant,
                    "{dtype:?}"
                );
            }
        }
        // A row width the MMQ block size does not divide declines.
        assert_eq!(
            select_linear_kind(
                LinearDevice::Cuda,
                GgmlDType::Q4K,
                4096 + 32,
                true,
                true,
                false
            ),
            QuantizedLinearKind::Dequant,
        );
    }

    /// The per-forward half mirrors candle's activation-shaped declines and
    /// the process-global `FORCE_DMMV` mirror.
    #[test]
    fn qmatmul_forward_supported_mirrors_rank_and_force_dmmv() {
        assert!(qmatmul_forward_supported(LinearDevice::Cuda, 2, false));
        assert!(qmatmul_forward_supported(LinearDevice::Cuda, 3, false));
        assert!(!qmatmul_forward_supported(LinearDevice::Cuda, 4, false));
        assert!(!qmatmul_forward_supported(LinearDevice::Cuda, 3, true));
        for device in [LinearDevice::Metal, LinearDevice::Other] {
            for force in [false, true] {
                assert!(qmatmul_forward_supported(device, 4, force));
            }
        }
    }

    #[test]
    fn qmatmul_flag_parser_accepts_truthy_and_rejects_the_rest() {
        for value in ["1", "true", "on", "yes", " TRUE ", "Yes"] {
            assert!(parse_qmatmul_flag(Some(value)), "{value:?}");
        }
        for value in ["0", "false", "off", "no", "", "garbage"] {
            assert!(!parse_qmatmul_flag(Some(value)), "{value:?}");
        }
        assert!(!parse_qmatmul_flag(None));
    }

    /// The dequant arm is exactly `dequantize + dense linear` at the kernel
    /// dtype, cast back to the activation dtype at the boundary.
    #[test]
    fn quantized_linear_dequant_matches_dense_reference_and_keeps_input_dtype() {
        let device = Device::Cpu;
        let (out_dim, in_dim) = (4usize, 64usize);
        let weight = quantized(out_dim, in_dim, GgmlDType::Q8_0);
        let bias = Tensor::from_vec(vec![0.5f32, -0.5, 1.0, 0.0], out_dim, &device).unwrap();
        let linear = QuantizedLinear::new(
            weight.clone(),
            Some(bias.clone()),
            &device,
            DType::F32,
            false,
        )
        .unwrap();
        assert_eq!(linear.kind(), QuantizedLinearKind::Dequant);

        let x = Tensor::from_vec(
            (0..2 * in_dim)
                .map(|i| i as f32 * 0.01 - 0.3)
                .collect::<Vec<_>>(),
            (1, 2, in_dim),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let got = linear.forward(&x).unwrap();
        assert_eq!(
            got.dtype(),
            DType::BF16,
            "CastBoundary: output follows input"
        );

        let reference = candle_nn::Linear::new(weight.dequantize(&device).unwrap(), Some(bias))
            .forward(&x.to_dtype(DType::F32).unwrap())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let diff = (got.to_dtype(DType::F32).unwrap() - reference.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0);
    }

    /// The QMatMul arm agrees with the dequant arm on CPU (candle's CPU
    /// QMatMul path), inside quantization tolerance.
    #[test]
    fn qmatmul_arm_on_cpu_matches_dequant_arm() {
        let device = Device::Cpu;
        let (out_dim, in_dim) = (4usize, 64usize);
        let weight = quantized(out_dim, in_dim, GgmlDType::Q8_0);
        let qmatmul = QuantizedLinear {
            arm: QuantizedLinearArm::QMatMul {
                inner: QMatMulLinear::from_arc(weight.clone(), None).unwrap(),
                fallback: None,
            },
            kernel_dtype: DType::F32,
        };
        let dequant = QuantizedLinear::new(weight, None, &device, DType::F32, false).unwrap();

        let x = Tensor::from_vec(
            (0..3 * in_dim)
                .map(|i| i as f32 * 0.02 - 0.5)
                .collect::<Vec<_>>(),
            (1, 3, in_dim),
            &device,
        )
        .unwrap();
        let a = qmatmul.forward(&x).unwrap();
        let b = dequant.forward(&x).unwrap();
        let peak = b
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // Candle's CPU QMatMul quantizes the ACTIVATION (Q8_1) before its
        // integer dot products, so the two arms agree to activation-quant
        // tolerance, never bit-for-bit.
        assert!(
            diff <= 0.02 * peak + 1e-3,
            "QMatMul and dequant disagree by {diff} against peak {peak}"
        );
    }

    /// CPU construction with the flag on still resolves the dequant arm —
    /// the fast path is a CUDA/Metal question, and `new` must never hand CPU
    /// activations to a kernel candle routes through the f32-only fallback.
    #[test]
    fn cpu_construction_resolves_dequant_even_when_opted_in() {
        let weight = quantized(4, 64, GgmlDType::Q8_0);
        let linear = QuantizedLinear::new(weight, None, &Device::Cpu, DType::F32, true).unwrap();
        assert_eq!(linear.kind(), QuantizedLinearKind::Dequant);
    }
}
