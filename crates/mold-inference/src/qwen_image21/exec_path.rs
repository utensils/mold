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
    /// Whether the scheduler timestep is rounded to the latent dtype before
    /// dividing by 1000, as upstream does (`pipeline_qwenimage21.py:770,775`
    /// casts `t` to the latents' dtype before `timestep / 1000`). A per-request
    /// decision: resolve it with [`Self::rounds_timestep`].
    pub timestep_rounding: TimestepRounding,
}

/// How a path decides whether to round the transformer timestep through the
/// working dtype (upstream `pipeline_qwenimage21.py:770,775`).
///
/// v0.32 passed the unrounded f64 `timestep / 1000`. Its bytes are archived
/// for exactly two cases — the legacy CUDA/CPU arithmetic and Metal's
/// base-tier (bf16) plain text-to-image render — so only those keep it.
/// Every render v0.32 could not make (a turbo tier, a quantized tier, a
/// reference, a transparent background, a LoRA) has no bytes to preserve and
/// follows upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimestepRounding {
    /// Never round: [`Qwen21ExecPath::legacy`], v0.32 byte for byte.
    Never,
    /// Always round: the CUDA fast path, which has no v0.32 bytes at all.
    Always,
    /// Metal: keep v0.32's unrounded value only for a request v0.32 could
    /// render ([`Qwen21RequestShape::has_v032_bytes`]), round everything else.
    UnlessV032Request,
}

/// The facts about a request that execution decisions read, resolved once
/// per render from the [`GenerateRequest`](mold_core::GenerateRequest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen21RequestShape {
    /// v0.32 shipped exactly `qwen-image-2.1:bf16` plain text-to-image: no
    /// reference images, no transparent background, no LoRA (v0.32 refused
    /// every adapter). Only such a request has archived bytes to reproduce.
    pub has_v032_bytes: bool,
}

impl Qwen21RequestShape {
    pub(crate) fn of(req: &mold_core::GenerateRequest) -> Self {
        let base_tier =
            mold_core::manifest::resolve_model_name(&req.model) == "qwen-image-2.1:bf16";
        let references = req
            .edit_images
            .as_ref()
            .is_some_and(|images| !images.is_empty());
        let transparent = req.transparent_background == Some(true);
        let lora = !req.caller_lora_stack().is_empty();
        Self {
            has_v032_bytes: base_tier && !references && !transparent && !lora,
        }
    }
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
            timestep_rounding: TimestepRounding::Never,
        }
    }

    /// Metal's shipped path. `fast` is `attention::metal_fast_path_enabled()`;
    /// the rotary tables were F32 on Metal before either mode existed. Both
    /// modes round the timestep except for a request with v0.32 bytes.
    pub(crate) const fn metal(fast: bool) -> Self {
        if fast {
            Self {
                attention: TargetAttention::MetalSdpa,
                fused_projection: true,
                compact_modulation: true,
                fused_adaln: false,
                f32_rope_tables: true,
                timestep_rounding: TimestepRounding::UnlessV032Request,
            }
        } else {
            Self {
                f32_rope_tables: true,
                timestep_rounding: TimestepRounding::UnlessV032Request,
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
            timestep_rounding: TimestepRounding::Always,
        }
    }

    /// Whether `request`'s denoise rounds its timestep through the working
    /// dtype on this path (see [`TimestepRounding`]).
    pub(crate) const fn rounds_timestep(&self, request: Qwen21RequestShape) -> bool {
        match self.timestep_rounding {
            TimestepRounding::Never => false,
            TimestepRounding::Always => true,
            TimestepRounding::UnlessV032Request => !request.has_v032_bytes,
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
            assert_eq!(
                path.timestep_rounding,
                TimestepRounding::UnlessV032Request,
                "Metal keeps its unrounded timestep only where v0.32 bytes exist"
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
        assert_eq!(legacy.timestep_rounding, TimestepRounding::Never);
    }

    #[test]
    fn the_fast_cuda_path_turns_every_knob_on() {
        let fast = Qwen21ExecPath::cuda_fast();
        assert_eq!(fast.attention, TargetAttention::FastStill);
        assert!(fast.fused_projection && fast.compact_modulation);
        assert!(fast.fused_adaln && fast.f32_rope_tables);
        assert_eq!(fast.timestep_rounding, TimestepRounding::Always);
    }

    fn request(model: &str) -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a red ceramic teapot",
            "model": model,
            "width": 1024,
            "height": 1024,
            "steps": 40,
            "guidance": 1.0,
        }))
        .expect("minimal request")
    }

    /// Only `qwen-image-2.1:bf16` plain text-to-image has v0.32 bytes; every
    /// tier, reference, transparency or LoRA added since has none.
    #[test]
    fn only_a_base_tier_plain_render_has_v032_bytes() {
        assert!(Qwen21RequestShape::of(&request("qwen-image-2.1:bf16")).has_v032_bytes);
        assert!(Qwen21RequestShape::of(&request("qwen-image-2.1")).has_v032_bytes);
        for model in [
            "qwen-image-2.1-turbo:bf16",
            "qwen-image-2.1:q8",
            "qwen-image-2.1:int8-conv",
            "qwen-image-2.1:fp8",
        ] {
            assert!(
                !Qwen21RequestShape::of(&request(model)).has_v032_bytes,
                "{model}"
            );
        }
        let mut refs = request("qwen-image-2.1:bf16");
        refs.edit_images = Some(vec![vec![0u8; 4]]);
        assert!(!Qwen21RequestShape::of(&refs).has_v032_bytes);
        let mut transparent = request("qwen-image-2.1:bf16");
        transparent.transparent_background = Some(true);
        assert!(!Qwen21RequestShape::of(&transparent).has_v032_bytes);
        let mut lora = request("qwen-image-2.1:bf16");
        lora.lora = Some(mold_core::LoraWeight {
            path: "/tmp/adapter.safetensors".into(),
            scale: 1.0,
            expert: None,
        });
        assert!(!Qwen21RequestShape::of(&lora).has_v032_bytes);
    }

    /// Metal rounds wherever there are no v0.32 bytes to preserve, in both of
    /// its modes; the legacy path never rounds, the CUDA fast path always does.
    #[test]
    fn timestep_rounding_is_decided_per_request() {
        let v032 = Qwen21RequestShape {
            has_v032_bytes: true,
        };
        let new = Qwen21RequestShape {
            has_v032_bytes: false,
        };
        for fast in [true, false] {
            let metal = Qwen21ExecPath::metal(fast);
            assert!(!metal.rounds_timestep(v032));
            assert!(metal.rounds_timestep(new));
        }
        assert!(!Qwen21ExecPath::legacy().rounds_timestep(v032));
        assert!(!Qwen21ExecPath::legacy().rounds_timestep(new));
        assert!(Qwen21ExecPath::cuda_fast().rounds_timestep(v032));
        assert!(Qwen21ExecPath::cuda_fast().rounds_timestep(new));
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
