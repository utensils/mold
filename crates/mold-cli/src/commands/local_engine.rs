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

    if ov.eager {
        std::env::set_var("MOLD_EAGER", "1");
    }
    if ov.offload {
        std::env::set_var("MOLD_OFFLOAD", "1");
    }
    let is_eager =
        ov.eager || mold_inference::runtime_env::value("MOLD_EAGER").is_some_and(|v| v == "1");
    let load_strategy = if is_eager {
        LoadStrategy::Eager
    } else {
        LoadStrategy::Sequential
    };
    let is_offload =
        ov.offload || mold_inference::runtime_env::value("MOLD_OFFLOAD").is_some_and(|v| v == "1");

    mold_inference::create_engine(
        model.to_string(),
        paths,
        config,
        load_strategy,
        gpu_ordinal,
        is_offload,
    )
}

/// Resolve the same concrete execution plans used by the server coordinator,
/// then run the pure scheduler against real discovered free VRAM and OS
/// available-RAM headroom. Each ordinal maps to exactly the immutable plan
/// that admitted it.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) async fn plan_local_batch(
    request: &mold_core::GenerateRequest,
    config: &Config,
    ov: &EngineOverrides,
    item_count: usize,
) -> Result<(
    Vec<usize>,
    std::collections::BTreeMap<usize, mold_server::execution_plan::ResolvedExecutionPlan>,
)> {
    use sysinfo::System;

    if ov.eager {
        std::env::set_var("MOLD_EAGER", "1");
    }
    if ov.offload {
        std::env::set_var("MOLD_OFFLOAD", "1");
    }
    let selected_ordinals = selected_local_gpu_ordinals(config, ov)?;
    let discovered = mold_inference::device::discover_gpus();
    let facts = discovered
        .iter()
        .filter(|gpu| selected_ordinals.contains(&gpu.ordinal))
        .filter_map(|gpu| {
            Some(mold_server::execution_plan::DeviceFact {
                id: gpu.stable_id.clone()?,
                ordinal: gpu.ordinal,
                available_vram_bytes: gpu.free_vram_bytes,
            })
        })
        .collect::<Vec<_>>();
    if facts.is_empty() {
        anyhow::bail!("local scheduler has no discovered device with stable identity");
    }
    let offload_requested = ov.offload
        || mold_inference::runtime_env::value("MOLD_OFFLOAD")
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
    let prepared = mold_server::variant_dependencies::prepare_local_execution_inputs(
        config,
        request,
        facts.clone(),
    )
    .await
    .map_err(anyhow::Error::msg)?;
    let plans = mold_server::execution_plan::resolve_execution_plans_with_prepared(
        config,
        request,
        &facts,
        offload_requested,
        Some(&prepared),
    )?;
    let by_ordinal = plans
        .into_iter()
        .map(|plan| (plan.device_ordinal, plan))
        .collect::<std::collections::BTreeMap<_, _>>();
    let candidates = by_ordinal
        .values()
        .map(|plan| LocalCandidate {
            ordinal: plan.device_ordinal,
            device_id: plan.device_id.clone(),
            execution_fingerprint: plan.execution_fingerprint.clone(),
            available_vram_bytes: plan.admitted_available_vram_bytes,
            predicted_vram_bytes: plan.predicted_vram_peak_bytes,
            predicted_host_ram_bytes: plan.predicted_host_increment_bytes,
        })
        .collect::<Vec<_>>();

    let mut system = System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    );
    system.refresh_memory();
    let total = system.total_memory();
    let safety_floor = (total.saturating_mul(15) / 100).max(8 << 30);
    let host_headroom = system.available_memory().saturating_sub(safety_floor);
    let assignments = local_batch_assignments(&candidates, item_count, host_headroom)?;
    Ok((assignments, by_ordinal))
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) fn build_local_engine_from_plan(
    request: &mold_core::GenerateRequest,
    config: &Config,
    plan: &mold_server::execution_plan::ResolvedExecutionPlan,
) -> Result<Box<dyn mold_inference::InferenceEngine>> {
    mold_server::execution_plan::validate_before_cuda(
        plan,
        &plan.device_id,
        plan.device_ordinal,
        config,
        request,
    )?;
    let current_free =
        mold_inference::device::free_vram_bytes(plan.device_ordinal).ok_or_else(|| {
            anyhow::anyhow!(
                "current free VRAM is unavailable for GPU {}",
                plan.device_ordinal
            )
        })?;
    if current_free < plan.predicted_vram_peak_bytes {
        anyhow::bail!(
            "local execution plan invalidated before CUDA: GPU {} now has {} bytes free but the exact plan requires {}",
            plan.device_ordinal,
            current_free,
            plan.predicted_vram_peak_bytes
        );
    }
    mold_inference::create_engine_with_frozen_config(
        request.model.clone(),
        plan.engine_paths.clone(),
        &plan.engine_config,
        plan.engine_load_strategy,
        plan.device_ordinal,
        plan.offload_mode == mold_server::execution_plan::OffloadMode::Block,
        None,
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) struct LocalCandidate {
    pub ordinal: usize,
    pub device_id: String,
    pub execution_fingerprint: String,
    pub available_vram_bytes: u64,
    pub predicted_vram_bytes: u64,
    pub predicted_host_ram_bytes: u64,
}

/// Partition payloads into immutable owner lanes. There is deliberately no
/// shared steal queue: a payload carrying GPU 0's materialized plan can never
/// be claimed by GPU 1.
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn partition_local_owner_lanes<T>(
    assignments: &[usize],
    items: Vec<T>,
) -> anyhow::Result<std::collections::BTreeMap<usize, Vec<T>>> {
    if assignments.len() != items.len() {
        anyhow::bail!(
            "local scheduler produced {} assignments for {} items",
            assignments.len(),
            items.len()
        );
    }
    let mut lanes = std::collections::BTreeMap::<usize, Vec<T>>::new();
    for (ordinal, item) in assignments.iter().copied().zip(items) {
        lanes.entry(ordinal).or_default().push(item);
    }
    Ok(lanes)
}

/// Assign local batch items through the shared deterministic scheduler using
/// concrete capacity and placement facts. The returned ordinal for each item
/// is its lease lane; callers must keep all work for that lane on one owner
/// thread.
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn local_batch_assignments(
    candidates: &[LocalCandidate],
    item_count: usize,
    host_headroom_bytes: u64,
) -> anyhow::Result<Vec<usize>> {
    use mold_scheduler::{
        CandidatePlacement, DeviceSnapshot, HostMemorySnapshot, Planner, PlannerSnapshot,
        WorkSnapshot,
    };
    if item_count == 0 {
        return Ok(Vec::new());
    }
    if candidates.is_empty() {
        anyhow::bail!("local scheduler has no eligible device");
    }
    let mut candidates = candidates.to_vec();
    candidates.sort_by(|left, right| {
        left.ordinal
            .cmp(&right.ordinal)
            .then_with(|| left.execution_fingerprint.cmp(&right.execution_fingerprint))
    });
    let devices = candidates
        .iter()
        .map(|candidate| {
            DeviceSnapshot::idle(candidate.device_id.clone(), candidate.available_vram_bytes)
        })
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
                    candidates
                        .iter()
                        .map(|candidate| {
                            CandidatePlacement::new(
                                candidate.device_id.clone(),
                                candidate.execution_fingerprint.clone(),
                                candidate.predicted_host_ram_bytes,
                            )
                            .with_vram(candidate.predicted_vram_bytes)
                            .with_device_available_vram(candidate.available_vram_bytes)
                            .with_static_timing(mold_scheduler::WorkKind::Generation)
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
                headroom_bytes: host_headroom_bytes,
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
            let ordinal = candidates
                .iter()
                .find(|candidate| candidate.device_id == lease.device_id.as_str())
                .map(|candidate| candidate.ordinal)
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
    fn local_batch_scheduler_supports_zero_one_and_arbitrary_real_capacities() {
        let candidates = |count: usize| {
            (0..count)
                .map(|ordinal| LocalCandidate {
                    ordinal,
                    device_id: format!("cuda:{ordinal}"),
                    execution_fingerprint: format!("exec:{ordinal}"),
                    available_vram_bytes: 24 << 30,
                    predicted_vram_bytes: 8 << 30,
                    predicted_host_ram_bytes: 1 << 30,
                })
                .collect::<Vec<_>>()
        };
        assert!(local_batch_assignments(&[], 1, 32 << 30).is_err());
        assert_eq!(
            local_batch_assignments(&candidates(1), 0, 32 << 30).unwrap(),
            Vec::<usize>::new()
        );
        let mut one = candidates(1);
        one[0].ordinal = 7;
        assert_eq!(
            local_batch_assignments(&one, 3, 32 << 30).unwrap(),
            vec![7, 7, 7]
        );
        for count in [2_usize, 8, 16, 64] {
            let assignments =
                local_batch_assignments(&candidates(count), count * 2 + 1, 128 << 30).unwrap();
            assert_eq!(assignments.len(), count * 2 + 1);
            for wave in assignments.chunks(count) {
                let unique = wave
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(unique.len(), wave.len());
            }
        }

        let mut oversized = candidates(1);
        oversized[0].predicted_vram_bytes = 25 << 30;
        assert!(local_batch_assignments(&oversized, 1, 32 << 30).is_err());

        let host_heavy = candidates(2);
        let host_limited = local_batch_assignments(&host_heavy, 2, (2 << 30) - 1).unwrap();
        assert_eq!(
            host_limited
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1,
            "host headroom may admit only one persistent local owner lane"
        );

        let pinned = vec![candidates(2)[1].clone()];
        assert_eq!(
            local_batch_assignments(&pinned, 2, 32 << 30).unwrap(),
            vec![1, 1]
        );
    }

    #[test]
    fn local_owner_lanes_cannot_steal_another_devices_concrete_plan() {
        let assignments = vec![0, 1, 0, 1];
        let items = vec![
            (0, "plan-gpu-0"),
            (1, "plan-gpu-1"),
            (2, "plan-gpu-0"),
            (3, "plan-gpu-1"),
        ];
        let lanes = partition_local_owner_lanes(&assignments, items).unwrap();
        assert_eq!(lanes[&0], vec![(0, "plan-gpu-0"), (2, "plan-gpu-0")]);
        assert_eq!(lanes[&1], vec![(1, "plan-gpu-1"), (3, "plan-gpu-1")]);
        for (owner, items) in lanes {
            assert!(items
                .iter()
                .all(|(_, plan)| *plan == format!("plan-gpu-{owner}")));
        }
    }
}
