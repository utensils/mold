//! Process-frozen environment inputs that can shape engine construction or
//! generation semantics.
//!
//! Planned execution must not observe a different process environment after
//! admission. Production captures this set once, when the first frozen engine
//! config is resolved. Unit-test builds intentionally capture per call so
//! existing parser tests can exercise multiple values in one process; the
//! immutable production behavior is covered by pure snapshot tests.

use std::collections::BTreeMap;
#[cfg(not(test))]
use std::sync::OnceLock;

/// Every environment variable whose value can alter device choice, concrete
/// weights, numeric execution, memory residency, or output semantics.
///
/// Every name added here must also gain a matching `RuntimeSemanticVariable`
/// arm in mold-server's `execution_plan.rs` (`runtime_semantic_variable`).
/// `every_engine_shaping_variable_has_a_semantic_class` over there fails CI on
/// a mismatch; at runtime an unclassified name fails the job with
/// `ExecutionPlanError::UnclassifiedRuntimeVariable` instead of panicking the
/// server (issue #685).
pub const ENGINE_SHAPING_VARIABLES: &[&str] = &[
    "MOLD_ATTN",
    "MOLD_ATTN_CHUNK",
    "MOLD_CFG_PLUS",
    "MOLD_DEVICE",
    "MOLD_EAGER",
    "MOLD_FLUX_DELTA_CACHE",
    "MOLD_FLUX_KEEP_TRANSFORMER",
    "MOLD_KEEP_TE_RAM",
    "MOLD_LONG_PROMPTS",
    "MOLD_LORA_BYPASS",
    "MOLD_LTX_DEBUG_ALT_PROMPT",
    "MOLD_LTX_DEBUG_COMPARE_UNCOND",
    "MOLD_LTX_DEBUG_DISABLE_AUDIO_BRANCH",
    "MOLD_LTX_DEBUG_DISABLE_CROSS_ATTENTION_ADALN",
    "MOLD_LTX2_DEBUG_DISABLE_TRANSFORMER_GATED_ATTENTION",
    "MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER",
    "MOLD_LTX2_DEBUG_LOAD_BLOCKS",
    "MOLD_LTX2_FORCE_EAGER",
    "MOLD_LTX2_FORCE_STREAMING",
    "MOLD_LTX2_FP8_INPUT_SCALE_MODE",
    "MOLD_LTX2_FP8_WEIGHT_SCALE_MODE",
    "MOLD_LTX2_GEMMA_DEVICE",
    "MOLD_LTX2_GEMMA_VARIANT",
    "MOLD_LTX2_SPATIAL_TILE",
    "MOLD_LTX2_VAE_DECODE_CHUNK_FRAMES",
    "MOLD_LTX2_VAE_DECODE_CONTEXT_FRAMES",
    "MOLD_LTX2_VAE_FORCE_FRAMEWISE",
    "MOLD_LTX2_VAE_FORCE_FULL_DECODE",
    "MOLD_NVFP4_BACKEND",
    "MOLD_OFFLOAD",
    "MOLD_OFFLOAD_PREFETCH",
    "MOLD_PINNED_VRAM_MAX_GB",
    "MOLD_QWEN2_TEXT_ENCODER_MODE",
    "MOLD_QWEN2_VARIANT",
    "MOLD_QWEN3_VARIANT",
    "MOLD_RESERVE_VRAM_MB",
    "MOLD_T5_VARIANT",
    "MOLD_UMT5_VARIANT",
    "MOLD_VAE_DTYPE",
    "MOLD_VAE_TILED",
    // #775 diagnostics: forcing the quantized-matmul fallback changes
    // numerics, runtime, and transient F32 memory; the profiler's per-phase
    // device syncs change runtime. Neither may share a fingerprint or a
    // learned-timing bucket with normal execution.
    "MOLD_WAN_FORCE_DMMV",
    // #802 item 3: the partner page-cache warm changes wall-clock (that is its
    // purpose), so a run with it off must not share a learned-timing bucket
    // with one that had it on.
    "MOLD_WAN_PREFETCH",
    // #801: residual reuse changes the rendered output, so it can never share
    // an execution-equivalence class or a learned-timing bucket with a run
    // that denoised every block.
    "MOLD_WAN_STEP_CACHE",
    "MOLD_WAN_STEP_PROFILE",
    "MOLD_WUERSTCHEN_DECODER_GUIDANCE",
];

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FrozenRuntimeEnvironment {
    values: BTreeMap<String, Option<String>>,
}

impl FrozenRuntimeEnvironment {
    fn capture() -> Self {
        Self {
            values: ENGINE_SHAPING_VARIABLES
                .iter()
                .map(|name| ((*name).to_string(), std::env::var(name).ok()))
                .collect(),
        }
    }

    pub fn value(&self, name: &str) -> Option<&str> {
        self.values.get(name).and_then(Option::as_deref)
    }
}

#[cfg(not(test))]
static PROCESS_ENVIRONMENT: OnceLock<FrozenRuntimeEnvironment> = OnceLock::new();

pub fn snapshot() -> FrozenRuntimeEnvironment {
    #[cfg(test)]
    {
        FrozenRuntimeEnvironment::capture()
    }
    #[cfg(not(test))]
    {
        PROCESS_ENVIRONMENT
            .get_or_init(FrozenRuntimeEnvironment::capture)
            .clone()
    }
}

pub fn value(name: &str) -> Option<String> {
    debug_assert!(
        ENGINE_SHAPING_VARIABLES.contains(&name),
        "runtime_env::value called for non-shaping variable {name}"
    );
    snapshot().value(name).map(str::to_string)
}

/// Canonical unsigned parser shared by runtime consumers and execution
/// equivalence. Keeping one authority prevents whitespace or invalid-value
/// handling from collapsing distinct runtime behavior into one descriptor.
pub fn parse_u64(value: &str) -> Option<u64> {
    value.trim().parse().ok()
}

/// Canonical floating-point parser shared by runtime consumers and execution
/// equivalence.
pub fn parse_f64(value: &str) -> Option<f64> {
    value.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_covers_required_multi_gpu_engine_shaping_inputs() {
        for required in [
            "MOLD_CFG_PLUS",
            "MOLD_EAGER",
            "MOLD_FLUX_KEEP_TRANSFORMER",
            "MOLD_KEEP_TE_RAM",
            "MOLD_LORA_BYPASS",
            "MOLD_LONG_PROMPTS",
            "MOLD_LTX2_FORCE_EAGER",
            "MOLD_LTX2_FP8_INPUT_SCALE_MODE",
            "MOLD_LTX2_FP8_WEIGHT_SCALE_MODE",
            "MOLD_NVFP4_BACKEND",
            "MOLD_OFFLOAD_PREFETCH",
            "MOLD_PINNED_VRAM_MAX_GB",
            "MOLD_RESERVE_VRAM_MB",
            "MOLD_WUERSTCHEN_DECODER_GUIDANCE",
            "MOLD_DEVICE",
        ] {
            assert!(
                ENGINE_SHAPING_VARIABLES.contains(&required),
                "missing engine-shaping environment input {required}"
            );
        }
    }

    #[test]
    fn logging_dump_and_preview_switches_are_not_engine_shaping_authority() {
        for diagnostic in [
            "MOLD_FLUX2_DUMP_LATENT",
            "MOLD_LTX2_DEBUG_TIMINGS",
            "MOLD_QWEN_DEBUG",
            "MOLD_SD3_DEBUG",
            "MOLD_STEP_PREVIEW",
            "MOLD_ZIMAGE_DEBUG",
        ] {
            assert!(!ENGINE_SHAPING_VARIABLES.contains(&diagnostic));
        }
    }

    #[test]
    fn shared_numeric_parsers_define_whitespace_and_invalid_value_semantics() {
        assert_eq!(parse_u64(" 4 "), Some(4));
        assert_eq!(parse_u64("4 frames"), None);
        assert_eq!(parse_f64("\t1.5\n"), Some(1.5));
        assert_eq!(parse_f64("automatic"), None);
    }
}
