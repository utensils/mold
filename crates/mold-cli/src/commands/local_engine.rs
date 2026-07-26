//! Shared local-engine preamble for `mold run` (single clip) and
//! `mold run --frames N` chains in `--local` mode: resolve-or-pull the
//! model, apply encoder-variant env overrides, resolve eager/offload,
//! pick a GPU, and construct the engine. One home so the generate and
//! chain paths can't drift.

#[cfg(any(feature = "cuda", feature = "metal"))]
use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "metal"))]
use colored::Colorize;
#[cfg(any(feature = "cuda", feature = "metal"))]
use mold_core::{Config, ModelPaths};

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::output::status;
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::theme;

/// Engine-construction knobs collected from CLI flags.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Clone, Debug, Default)]
pub(crate) struct EngineOverrides {
    pub gpus: Option<String>,
    pub t5_variant: Option<String>,
    pub qwen3_variant: Option<String>,
    pub qwen2_variant: Option<String>,
    pub qwen2_text_encoder_mode: Option<String>,
    pub eager: bool,
    pub offload: bool,
}

/// Export encoder-variant overrides as `MOLD_*` env vars so the engine
/// factory's auto-select picks them up during construction.
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn apply_local_engine_env_overrides(
    t5_variant_override: Option<&str>,
    qwen3_variant_override: Option<&str>,
    qwen2_variant_override: Option<&str>,
    qwen2_text_encoder_mode_override: Option<&str>,
) {
    if let Some(variant) = t5_variant_override {
        std::env::set_var("MOLD_T5_VARIANT", variant);
    }
    if let Some(variant) = qwen3_variant_override {
        std::env::set_var("MOLD_QWEN3_VARIANT", variant);
    }
    if let Some(variant) = qwen2_variant_override {
        std::env::set_var("MOLD_QWEN2_VARIANT", variant);
    }
    if let Some(mode) = qwen2_text_encoder_mode_override {
        std::env::set_var("MOLD_QWEN2_TEXT_ENCODER_MODE", mode);
    }
}

/// Resolve model paths, auto-pulling when the model is missing locally or
/// has missing assets (repair pull). Returns the paths, the effective
/// config (post-pull when a pull happened), and whether a pull ran — a
/// `true` tells callers to re-derive request defaults from the refreshed
/// model config.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) async fn resolve_or_pull_model(
    model: &str,
    config: &Config,
) -> Result<(ModelPaths, Config, bool)> {
    use mold_core::manifest::find_manifest;

    if config.manifest_model_needs_download(model) {
        status!(
            "{} Model '{}' is missing local assets, pulling repair...",
            theme::icon_info(),
            model.bold(),
        );
        let updated =
            super::pull::pull_and_configure(model, &mold_core::download::PullOptions::default())
                .await?;
        let paths = ModelPaths::resolve(model, &updated).ok_or_else(|| {
            anyhow::anyhow!("model '{model}' was pulled but paths could not be resolved")
        })?;
        Ok((paths, updated, true))
    } else if let Some(paths) = ModelPaths::resolve(model, config) {
        Ok((paths, config.clone(), false))
    } else if find_manifest(model).is_some() {
        status!(
            "{} Model '{}' not found locally, pulling...",
            theme::icon_info(),
            model.bold(),
        );
        let updated =
            super::pull::pull_and_configure(model, &mold_core::download::PullOptions::default())
                .await?;
        let paths = ModelPaths::resolve(model, &updated).ok_or_else(|| {
            anyhow::anyhow!("model '{model}' was pulled but paths could not be resolved")
        })?;
        Ok((paths, updated, true))
    } else {
        anyhow::bail!(
            "no model paths configured for '{model}'. Add [models.{model}] to \
             ~/.mold/config.toml, pull via `mold pull {model}`, or set \
             MOLD_TRANSFORMER_PATH / MOLD_VAE_PATH / MOLD_T5_PATH / MOLD_CLIP_PATH \
             / MOLD_T5_TOKENIZER_PATH / MOLD_CLIP_TOKENIZER_PATH env vars.",
        );
    }
}

/// Apply env overrides, resolve eager/offload, select the best GPU from
/// the allowed set (most free VRAM), and construct the engine.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) fn build_local_engine(
    model: &str,
    paths: ModelPaths,
    config: &Config,
    ov: &EngineOverrides,
) -> Result<Box<dyn mold_inference::InferenceEngine>> {
    let discovered = mold_inference::device::discover_gpus();
    let gpu_ordinal = selected_local_gpu_ordinals(config, ov)?
        .into_iter()
        .max_by_key(|ordinal| {
            discovered
                .iter()
                .find(|gpu| gpu.ordinal == *ordinal)
                .map_or(0, |gpu| gpu.free_vram_bytes)
        })
        .unwrap_or(0);
    build_local_engine_on_gpu(model, paths, config, ov, gpu_ordinal)
}

/// Resolve every selected runtime device. This is shared by single-item local
/// inference and the scheduler-backed local batch adapter; it has no 2-GPU
/// special case.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) fn selected_local_gpu_ordinals(
    config: &Config,
    ov: &EngineOverrides,
) -> Result<Vec<usize>> {
    let gpu_selection = match &ov.gpus {
        Some(s) => mold_core::types::GpuSelection::parse(s)?,
        None => config.gpu_selection(),
    };
    let discovered = mold_inference::device::discover_gpus();
    let mut available = mold_inference::device::resolve_gpu_selection(&discovered, &gpu_selection)?;
    if available.is_empty() {
        if matches!(gpu_selection, mold_core::types::GpuSelection::None) {
            anyhow::bail!("GPU selection 'none' cannot run local inference");
        }
        if !discovered.is_empty() {
            anyhow::bail!("no CUDA device with a stable identity is available for local inference");
        }
        return Ok(vec![0]);
    }
    available.sort_by_key(|gpu| gpu.ordinal);
    Ok(available.into_iter().map(|gpu| gpu.ordinal).collect())
}

/// Construct an engine pinned to one scheduler-selected local device.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) fn build_local_engine_on_gpu(
    model: &str,
    paths: ModelPaths,
    config: &Config,
    ov: &EngineOverrides,
    gpu_ordinal: usize,
) -> Result<Box<dyn mold_inference::InferenceEngine>> {
    use mold_inference::LoadStrategy;

    apply_local_engine_env_overrides(
        ov.t5_variant.as_deref(),
        ov.qwen3_variant.as_deref(),
        ov.qwen2_variant.as_deref(),
        ov.qwen2_text_encoder_mode.as_deref(),
    );

    let is_eager = ov.eager || std::env::var("MOLD_EAGER").is_ok_and(|v| v == "1");
    let load_strategy = if is_eager {
        LoadStrategy::Eager
    } else {
        LoadStrategy::Sequential
    };
    if is_eager {
        std::env::set_var("MOLD_EAGER", "1");
    }
    let is_offload = ov.offload || std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");

    mold_inference::create_engine(
        model.to_string(),
        paths,
        config,
        load_strategy,
        gpu_ordinal,
        is_offload,
    )
}

/// Assign local batch items through the same deterministic scheduler core
/// used by the server. Each returned value is a device ordinal for the item at
/// that index. Devices are reused in waves; no device-count ceiling exists.
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn local_batch_assignments(
    device_ordinals: &[usize],
    item_count: usize,
) -> anyhow::Result<Vec<usize>> {
    use mold_scheduler::{
        CandidatePlacement, DeviceSnapshot, HostMemorySnapshot, Planner, PlannerSnapshot,
        WorkSnapshot,
    };
    if item_count == 0 {
        return Ok(Vec::new());
    }
    if device_ordinals.is_empty() {
        anyhow::bail!("local scheduler has no eligible device");
    }
    let mut ordinals = device_ordinals.to_vec();
    ordinals.sort_unstable();
    ordinals.dedup();
    let devices = ordinals
        .iter()
        .map(|ordinal| DeviceSnapshot::idle(format!("local:{ordinal}"), u64::MAX))
        .collect::<Vec<_>>();
    let mut assignments = vec![usize::MAX; item_count];
    let planner = Planner::default();
    let mut next = 0_usize;
    let mut plan_version = 1_u64;
    while next < item_count {
        let work = (next..item_count)
            .map(|index| {
                WorkSnapshot::new(
                    format!("item:{index}"),
                    index as u64,
                    ordinals
                        .iter()
                        .map(|ordinal| {
                            CandidatePlacement::new(format!("local:{ordinal}"), "local-batch", 0)
                        })
                        .collect(),
                )
            })
            .collect();
        let snapshot = PlannerSnapshot {
            state_version: plan_version,
            next_plan_version: plan_version,
            now_ms: 0,
            next_replan_at_ms: None,
            host_memory: HostMemorySnapshot {
                headroom_bytes: u64::MAX,
                sample_generation: 1,
                ledger_sequence: 1,
            },
            devices: devices.clone(),
            work,
        };
        let plan = planner
            .plan(&snapshot)
            .map_err(|error| anyhow::anyhow!("local scheduler rejected batch: {error}"))?;
        if plan.immediate_leases.is_empty() {
            anyhow::bail!("local scheduler could not assign remaining batch items");
        }
        let mut assigned_indices = Vec::new();
        for lease in plan.immediate_leases {
            let index = lease
                .work_id
                .as_str()
                .strip_prefix("item:")
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| anyhow::anyhow!("local scheduler returned an invalid work id"))?;
            let ordinal = lease
                .device_id
                .as_str()
                .strip_prefix("local:")
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| anyhow::anyhow!("local scheduler returned an invalid device id"))?;
            assignments[index] = ordinal;
            assigned_indices.push(index);
        }
        assigned_indices.sort_unstable();
        let expected = (next..next + assigned_indices.len()).collect::<Vec<_>>();
        if assigned_indices != expected {
            anyhow::bail!("local scheduler violated strict batch order");
        }
        next += assigned_indices.len();
        plan_version = plan_version.saturating_add(1);
    }
    Ok(assignments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;

    #[test]
    fn apply_local_engine_env_overrides_sets_qwen2_overrides() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior_variant = std::env::var("MOLD_QWEN2_VARIANT").ok();
        let prior_mode = std::env::var("MOLD_QWEN2_TEXT_ENCODER_MODE").ok();

        std::env::remove_var("MOLD_QWEN2_VARIANT");
        std::env::remove_var("MOLD_QWEN2_TEXT_ENCODER_MODE");

        apply_local_engine_env_overrides(None, None, Some("q6"), Some("cpu-stage"));

        assert_eq!(
            std::env::var("MOLD_QWEN2_VARIANT").ok().as_deref(),
            Some("q6")
        );
        assert_eq!(
            std::env::var("MOLD_QWEN2_TEXT_ENCODER_MODE")
                .ok()
                .as_deref(),
            Some("cpu-stage")
        );

        match prior_variant {
            Some(value) => std::env::set_var("MOLD_QWEN2_VARIANT", value),
            None => std::env::remove_var("MOLD_QWEN2_VARIANT"),
        }
        match prior_mode {
            Some(value) => std::env::set_var("MOLD_QWEN2_TEXT_ENCODER_MODE", value),
            None => std::env::remove_var("MOLD_QWEN2_TEXT_ENCODER_MODE"),
        }
    }

    #[test]
    fn local_batch_scheduler_supports_zero_one_and_arbitrary_device_counts() {
        assert!(local_batch_assignments(&[], 1).is_err());
        assert_eq!(
            local_batch_assignments(&[7], 0).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(local_batch_assignments(&[7], 3).unwrap(), vec![7, 7, 7]);
        for count in [2_usize, 8, 16, 64] {
            let devices = (0..count).collect::<Vec<_>>();
            let assignments = local_batch_assignments(&devices, count * 2 + 1).unwrap();
            assert_eq!(assignments.len(), count * 2 + 1);
            for wave in assignments.chunks(count) {
                let unique = wave
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(unique.len(), wave.len());
            }
        }
    }
}
