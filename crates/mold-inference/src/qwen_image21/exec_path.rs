//! Qwen Image 2.1's resolved transformer execution path.
//!
//! The transformer used to carry three independent booleans (`fused_target`,
//! `fused_ops`, `compact_modulation`), each defaulted from
//! `attention::metal_fast_path_enabled()`. That shape could only describe two
//! paths — Metal-fast and everything-else — while CUDA needs a third, and a
//! benchmark harness needs to build any combination of the knobs directly.
//! [`Qwen21ExecPath`] is the one resolved value that replaces them.
//!
//! Three rules decide it, and each is pinned by a test below:
//!
//! * **Metal reproduces the path it shipped with exactly.** Fused SDPA target
//!   attention, fused projection/RoPE and compact modulation unless
//!   `MOLD_ATTN=math`; rotary tables are F32 on Metal in both cases; no fused
//!   adaLN. Nothing about Metal output moves.
//! * **CUDA takes the fast path unless `MOLD_ATTN=math`.** The target goes
//!   through `attention::attention_with_bias_for(FastStill, ..)` (FlashAttention
//!   wherever the kernel is compiled in), plus the fused projection, F32 RoPE
//!   tables, compact modulation, fused adaLN and the upstream dtype-rounded
//!   timestep. `MOLD_ATTN=math` selects
//!   [`Qwen21ExecPath::legacy`], which together with `MOLD_CONV=im2col`
//!   reproduces the v0.32 bytes.
//! * **CPU stays on the legacy path.** It has no fused kernel to gain from and
//!   is the reference the unit tests compare against.

// The transformer consumes this value once the joint-layout attention seam
// (`QwenImage21JointLayout` / `attend_segment`) lands; until then only the
// CUDA benchmark harness and the tests below read it.
#![allow(dead_code)]

use candle_core::Device;

use crate::attention::AttentionBackend;

/// How the transformer attends its target (and cached-decode) queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetAttention {
    /// v0.32: `attention::attention_with_bias` under the image policy — math
    /// in every build, softmax scale applied to the scores.
    Legacy,
    /// Metal's fused `candle_nn::ops::sdpa` for unbiased target calls; a
    /// biased call keeps the legacy math path.
    MetalSdpa,
    /// `attention::attention_with_bias_for(AttentionPolicy::FastStill, ..)`:
    /// FlashAttention for unbiased calls wherever the kernel is compiled in
    /// (CUDA BF16/F16), math with the scale folded into K otherwise.
    FastStill,
}

/// Every execution choice the Qwen Image 2.1 transformer makes that is not a
/// weight or a request field. Resolved once per engine from the device and the
/// process-frozen `MOLD_ATTN` request; the performance harness builds
/// arbitrary combinations directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen21ExecPath {
    /// Target-attention dispatch.
    pub attention: TargetAttention,
    /// Normalize Q/K with `rms_norm` on the contiguous BSHD projection before
    /// the transpose, then rotate with F32 `rope_i`, instead of flattening
    /// BHSD for the norm and rotating with Wan's composite RoPE.
    pub fused_projection: bool,
    /// Keep the shared per-timestep modulation as `[B, 1, D]` rows computed
    /// once per forward rather than broadcasting it to every image token.
    pub compact_modulation: bool,
    /// Fold LayerNorm and the `1 + scale` modulation into one fused kernel.
    pub fused_adaln: bool,
    /// Build the rotary tables in F32 regardless of the working dtype.
    pub f32_rope_tables: bool,
    /// Round the scheduler timestep to the latent dtype before dividing by
    /// 1000, as upstream does (diffusers `pipeline_qwenimage21.py` casts
    /// `t` to the latents' dtype before `timestep / 1000`). v0.32 passed the
    /// unrounded f64, so `legacy()` and Metal keep `false`; the pipeline
    /// reads this through `scheduler::transformer_timestep(sigma, dtype)`.
    pub round_timestep_to_dtype: bool,
}

impl Qwen21ExecPath {
    /// The v0.32 CUDA/CPU path, byte for byte: image-policy math attention,
    /// Wan RoPE on working-dtype tables, broadcast modulation and the
    /// hand-written F32 LayerNorm.
    pub(crate) const fn legacy() -> Self {
        Self {
            attention: TargetAttention::Legacy,
            fused_projection: false,
            compact_modulation: false,
            fused_adaln: false,
            f32_rope_tables: false,
            round_timestep_to_dtype: false,
        }
    }

    /// Metal's shipped path. `fast` is `attention::metal_fast_path_enabled()`;
    /// the rotary tables were F32 on Metal before either mode existed.
    pub(crate) const fn metal(fast: bool) -> Self {
        if fast {
            Self {
                attention: TargetAttention::MetalSdpa,
                fused_projection: true,
                compact_modulation: true,
                fused_adaln: false,
                f32_rope_tables: true,
                round_timestep_to_dtype: false,
            }
        } else {
            Self {
                f32_rope_tables: true,
                ..Self::legacy()
            }
        }
    }

    /// The CUDA fast path.
    pub(crate) const fn cuda_fast() -> Self {
        Self {
            attention: TargetAttention::FastStill,
            fused_projection: true,
            compact_modulation: true,
            fused_adaln: true,
            f32_rope_tables: true,
            round_timestep_to_dtype: true,
        }
    }

    /// Resolve for `device` under the process-frozen `MOLD_ATTN` request.
    pub(crate) fn resolve(device: &Device) -> Self {
        Self::resolve_for(
            ExecDevice::of(device),
            crate::attention::requested_backend(),
        )
    }

    /// Pure composition behind [`Self::resolve`], testable without a GPU and
    /// without poisoning the process-frozen `MOLD_ATTN` cache.
    pub(crate) const fn resolve_for(
        device: ExecDevice,
        requested: Option<AttentionBackend>,
    ) -> Self {
        let math = matches!(requested, Some(AttentionBackend::Math));
        match device {
            ExecDevice::Metal => Self::metal(!math),
            ExecDevice::Cuda if !math => Self::cuda_fast(),
            ExecDevice::Cuda | ExecDevice::Cpu => Self::legacy(),
        }
    }

    /// Whether this is exactly the v0.32 CUDA/CPU arithmetic.
    pub(crate) fn is_legacy(&self) -> bool {
        *self == Self::legacy()
    }

    /// A stable short label for logs, harness receipts and qualification
    /// records.
    pub(crate) fn label(&self) -> &'static str {
        if *self == Self::legacy() {
            "legacy"
        } else if *self == Self::cuda_fast() {
            "fast"
        } else if *self == Self::metal(true) {
            "metal-fast"
        } else if *self == Self::metal(false) {
            "metal-math"
        } else {
            "custom"
        }
    }
}

/// The device classes [`Qwen21ExecPath::resolve_for`] distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecDevice {
    Cpu,
    Cuda,
    Metal,
}

impl ExecDevice {
    pub(crate) fn of(device: &Device) -> Self {
        if device.is_metal() {
            Self::Metal
        } else if device.is_cuda() {
            Self::Cuda
        } else {
            Self::Cpu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUESTS: [Option<AttentionBackend>; 3] = [
        None,
        Some(AttentionBackend::Flash),
        Some(AttentionBackend::Math),
    ];

    /// Metal must keep the values its booleans defaulted to before this type
    /// existed: every fused knob follows `metal_fast_path_enabled()` (true
    /// unless `MOLD_ATTN=math`), the tables are always F32, and adaLN is never
    /// fused there.
    #[test]
    fn metal_reproduces_the_shipped_booleans() {
        for requested in REQUESTS {
            let fast = requested != Some(AttentionBackend::Math);
            let path = Qwen21ExecPath::resolve_for(ExecDevice::Metal, requested);
            assert_eq!(
                path.attention,
                if fast {
                    TargetAttention::MetalSdpa
                } else {
                    TargetAttention::Legacy
                }
            );
            assert_eq!(path.fused_projection, fast, "fused_ops default");
            assert_eq!(path.compact_modulation, fast, "compact_modulation default");
            assert!(!path.fused_adaln, "Metal never fused adaLN");
            assert!(
                !path.round_timestep_to_dtype,
                "Metal keeps its unrounded timestep"
            );
            assert!(path.f32_rope_tables, "Metal tables were always F32");
        }
    }

    #[test]
    fn cuda_is_fast_unless_math_is_requested() {
        for requested in [None, Some(AttentionBackend::Flash)] {
            assert_eq!(
                Qwen21ExecPath::resolve_for(ExecDevice::Cuda, requested),
                Qwen21ExecPath::cuda_fast()
            );
        }
        let math = Qwen21ExecPath::resolve_for(ExecDevice::Cuda, Some(AttentionBackend::Math));
        assert_eq!(math, Qwen21ExecPath::legacy());
        assert!(math.is_legacy());
    }

    #[test]
    fn cpu_is_always_legacy() {
        for requested in REQUESTS {
            assert!(Qwen21ExecPath::resolve_for(ExecDevice::Cpu, requested).is_legacy());
        }
    }

    /// `legacy()` is the v0.32 path: no fused knob at all, working-dtype
    /// tables, the image-policy attention.
    #[test]
    fn legacy_is_the_v032_arithmetic() {
        let legacy = Qwen21ExecPath::legacy();
        assert_eq!(legacy.attention, TargetAttention::Legacy);
        assert!(!legacy.fused_projection);
        assert!(!legacy.compact_modulation);
        assert!(!legacy.fused_adaln);
        assert!(!legacy.f32_rope_tables);
        assert!(!legacy.round_timestep_to_dtype);
    }

    #[test]
    fn the_fast_cuda_path_turns_every_knob_on() {
        let fast = Qwen21ExecPath::cuda_fast();
        assert_eq!(fast.attention, TargetAttention::FastStill);
        assert!(fast.fused_projection && fast.compact_modulation);
        assert!(fast.fused_adaln && fast.f32_rope_tables);
        assert!(fast.round_timestep_to_dtype);
    }

    #[test]
    fn labels_name_every_resolved_path() {
        assert_eq!(Qwen21ExecPath::legacy().label(), "legacy");
        assert_eq!(Qwen21ExecPath::cuda_fast().label(), "fast");
        assert_eq!(Qwen21ExecPath::metal(true).label(), "metal-fast");
        assert_eq!(Qwen21ExecPath::metal(false).label(), "metal-math");
        let custom = Qwen21ExecPath {
            fused_adaln: true,
            ..Qwen21ExecPath::legacy()
        };
        assert_eq!(custom.label(), "custom");
    }

    #[test]
    fn device_classification_on_cpu() {
        assert_eq!(ExecDevice::of(&Device::Cpu), ExecDevice::Cpu);
    }
}
