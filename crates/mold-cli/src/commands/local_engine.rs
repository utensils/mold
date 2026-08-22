//! Shared local-engine preamble for `mold run` (single clip) and
//! `mold run --frames N` chains in `--local` mode: resolve-or-pull the
//! model, apply encoder-variant env overrides, resolve eager/offload,
//! freeze a scheduler-selected execution plan, and construct the engine on
//! its device owner thread. One home so generate and chain paths cannot drift.

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

/// Resolve the same concrete execution plans used by the server coordinator,
/// then run the pure scheduler against real discovered free VRAM and OS
/// available-RAM headroom. Each ordinal maps to exactly the immutable plan
/// that admitted it.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) async fn plan_local_batch(
    request: &mold_core::GenerateRequest,
    config: &Config,
    ov: &EngineOverrides,
) -> Result<LocalBatchPlan> {
    use sysinfo::System;

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
    let selected_ordinals = selected_local_gpu_ordinals(config, ov)?;
    let discovered = mold_inference::device::discover_gpus();
    let facts = discovered
        .iter()
        .filter(|gpu| selected_ordinals.contains(&gpu.ordinal))
        .filter_map(|gpu| {
            Some(mold_server::execution_plan::DeviceFact {
                id: gpu.stable_id.clone()?,
                ordinal: gpu.ordinal,
                backend: gpu.backend,
                compute_capability: gpu.compute_capability,
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
            execution_equivalence_fingerprint: plan.execution_equivalence_fingerprint.clone(),
            available_vram_bytes: plan.admitted_available_vram_bytes,
            // Backend-aware admission demands (#1038): on CUDA these are the
            // raw plan figures over two genuinely separate pools; on Metal
            // both claims fold onto the one unified pool — the demand is the
            // larger of the encoder phase and the denoise peak plus the
            // concurrent host charge, and the host figure is zero so the
            // same bytes are not proven twice against a second sample of the
            // same physical memory minus a safety floor the device gate
            // never paid. The gate predicate itself is unchanged.
            predicted_vram_bytes: plan.admission_vram_demand_bytes(),
            predicted_host_ram_bytes: plan.admission_host_demand_bytes(),
        })
        .collect::<Vec<_>>();

    let mut system = System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    );
    system.refresh_memory();
    let total = system.total_memory();
    let safety_floor = (total.saturating_mul(15) / 100).max(8 << 30);
    let host_headroom = system.available_memory().saturating_sub(safety_floor);
    Ok(LocalBatchPlan {
        candidates,
        execution_plans: by_ordinal,
        prepared_execution_inputs: prepared,
        host_headroom_bytes: host_headroom,
    })
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) fn build_local_engine_from_plan(
    request: &mold_core::GenerateRequest,
    config: &Config,
    plan: &mold_server::execution_plan::ResolvedExecutionPlan,
    prepared: &mold_server::execution_plan::PreparedExecutionInputs,
) -> Result<Box<dyn mold_inference::InferenceEngine>> {
    mold_server::execution_plan::validate_before_cuda(
        plan,
        &plan.device_id,
        plan.device_ordinal,
        config,
        request,
        Some(prepared),
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
    let mut engine = mold_inference::create_engine_with_frozen_config(
        request.model.clone(),
        plan.engine_paths.clone(),
        &plan.engine_config,
        plan.engine_load_strategy,
        plan.device_ordinal,
        plan.offload_mode == mold_server::execution_plan::OffloadMode::Block,
        None,
    )?;
    // The forced-local twin of the worker's install point. Local preparation
    // ran the extractor exactly once through the same
    // `prepare_local_execution_inputs` path the server uses, so this engine
    // gets the identical frozen value a remote render would have used — which
    // is what makes local/remote parity structural rather than reviewed.
    engine
        .install_identity_embedding(prepared.identity_embedding.as_ref())
        .map_err(|error| {
            anyhow::anyhow!("failed to install face-identity conditioning: {error:#}")
        })?;
    Ok(engine)
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) struct LocalCandidate {
    pub ordinal: usize,
    pub device_id: String,
    pub execution_fingerprint: String,
    pub execution_equivalence_fingerprint: mold_scheduler::ExecutionEquivalenceFingerprint,
    pub available_vram_bytes: u64,
    pub predicted_vram_bytes: u64,
    pub predicted_host_ram_bytes: u64,
}

#[derive(Clone, Debug)]
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) struct LocalBatchPlan {
    pub candidates: Vec<LocalCandidate>,
    pub execution_plans:
        std::collections::BTreeMap<usize, mold_server::execution_plan::ResolvedExecutionPlan>,
    pub prepared_execution_inputs: mold_server::execution_plan::PreparedExecutionInputs,
    pub host_headroom_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) struct LocalLease {
    pub ordinal: usize,
    pub item_index: usize,
}

/// Runtime adapter for forced-local batches. Future items remain in one global
/// queue. An owner enters the ready set only after its prior lease completes,
/// and every readiness event runs the same pure planner used by the server.
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) struct LocalBatchAdmission {
    candidates: std::collections::BTreeMap<usize, LocalCandidate>,
    pending: std::collections::VecDeque<usize>,
    ready: std::collections::BTreeSet<usize>,
    active: std::collections::BTreeMap<usize, usize>,
    resident_host_bytes: std::collections::BTreeMap<usize, u64>,
    host_headroom_bytes: u64,
    plan_version: u64,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl LocalBatchAdmission {
    pub(crate) fn new(
        candidates: &[LocalCandidate],
        item_count: usize,
        host_headroom_bytes: u64,
    ) -> anyhow::Result<Self> {
        if item_count > 0 && candidates.is_empty() {
            anyhow::bail!("local scheduler has no eligible device");
        }
        if item_count > 0
            && !candidates.iter().any(|candidate| {
                candidate.predicted_vram_bytes <= candidate.available_vram_bytes
                    && candidate.predicted_host_ram_bytes <= host_headroom_bytes
            })
        {
            // Name the shortfall the way `insufficient_vram_error` does — a bare
            // "cannot admit" leaves the user guessing which of the two budgets
            // was short, and by how much.
            let shortfall = candidates
                .iter()
                .map(|candidate| {
                    format!(
                        "{}: needs ~{:.1} GB VRAM of ~{:.1} GB usable and ~{:.1} GB host RAM of ~{:.1} GB headroom",
                        candidate.ordinal,
                        candidate.predicted_vram_bytes as f64 / 1_000_000_000.0,
                        candidate.available_vram_bytes as f64 / 1_000_000_000.0,
                        candidate.predicted_host_ram_bytes as f64 / 1_000_000_000.0,
                        host_headroom_bytes as f64 / 1_000_000_000.0,
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            anyhow::bail!(
                "local scheduler cannot admit an engine within current GPU and host-memory headroom ({shortfall})"
            );
        }
        Ok(Self {
            candidates: candidates
                .iter()
                .cloned()
                .map(|candidate| (candidate.ordinal, candidate))
                .collect(),
            pending: (0..item_count).collect(),
            ready: Default::default(),
            active: Default::default(),
            resident_host_bytes: Default::default(),
            host_headroom_bytes,
            plan_version: 0,
        })
    }

    pub(crate) fn owner_ready(&mut self, ordinal: usize) -> anyhow::Result<()> {
        if !self.candidates.contains_key(&ordinal) {
            anyhow::bail!("unknown local GPU owner {ordinal}");
        }
        if self.active.contains_key(&ordinal) {
            anyhow::bail!("local GPU owner {ordinal} became ready with an active lease");
        }
        self.ready.insert(ordinal);
        Ok(())
    }

    pub(crate) fn owner_completed(&mut self, ordinal: usize) -> anyhow::Result<()> {
        if self.active.remove(&ordinal).is_none() {
            anyhow::bail!("local GPU owner {ordinal} completed without an active lease");
        }
        Ok(())
    }

    pub(crate) fn owner_failed(&mut self, ordinal: usize) -> Option<usize> {
        let active_item = self.active.remove(&ordinal);
        self.resident_host_bytes.remove(&ordinal);
        self.ready.remove(&ordinal);
        self.candidates.remove(&ordinal);
        active_item
    }

    #[cfg_attr(test, allow(dead_code))]
    pub(crate) fn lease_transport_failed(&mut self, lease: LocalLease) {
        self.active.remove(&lease.ordinal);
        self.resident_host_bytes.remove(&lease.ordinal);
        self.ready.remove(&lease.ordinal);
        self.candidates.remove(&lease.ordinal);
        let position = self
            .pending
            .iter()
            .position(|index| *index > lease.item_index)
            .unwrap_or(self.pending.len());
        self.pending.insert(position, lease.item_index);
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub(crate) fn has_active(&self) -> bool {
        !self.active.is_empty()
    }

    pub(crate) fn has_owners(&self) -> bool {
        !self.candidates.is_empty()
    }

    pub(crate) fn lease_ready(&mut self) -> anyhow::Result<Vec<LocalLease>> {
        use mold_scheduler::{
            CandidatePlacement, DeviceSnapshot, HostMemorySnapshot, Planner, PlannerSnapshot,
            WorkSnapshot,
        };
        if self.pending.is_empty() || self.ready.is_empty() {
            return Ok(Vec::new());
        }
        let devices = self
            .candidates
            .values()
            .map(|candidate| {
                if self.ready.contains(&candidate.ordinal) {
                    DeviceSnapshot::idle(
                        candidate.device_id.clone(),
                        candidate.available_vram_bytes,
                    )
                } else {
                    DeviceSnapshot::busy(
                        candidate.device_id.clone(),
                        candidate.available_vram_bytes,
                        u64::MAX,
                    )
                }
            })
            .collect::<Vec<_>>();
        let work = self
            .pending
            .iter()
            .copied()
            .map(|index| {
                WorkSnapshot::new(
                    format!("item:{index}"),
                    index as u64,
                    self.candidates
                        .values()
                        .map(|candidate| {
                            let incremental_host_ram_bytes =
                                if self.resident_host_bytes.contains_key(&candidate.ordinal) {
                                    0
                                } else {
                                    candidate.predicted_host_ram_bytes
                                };
                            CandidatePlacement::new(
                                candidate.device_id.clone(),
                                candidate.execution_fingerprint.clone(),
                                incremental_host_ram_bytes,
                            )
                            .with_execution_equivalence(
                                candidate.execution_equivalence_fingerprint.clone(),
                            )
                            .with_vram(candidate.predicted_vram_bytes)
                            .with_device_available_vram(candidate.available_vram_bytes)
                            .with_static_timing(mold_scheduler::WorkKind::Generation)
                        })
                        .collect(),
                )
            })
            .collect();
        self.plan_version = self.plan_version.saturating_add(1);
        let reserved = self
            .resident_host_bytes
            .values()
            .copied()
            .fold(0_u64, u64::saturating_add);
        let snapshot = PlannerSnapshot {
            state_version: self.plan_version,
            next_plan_version: self.plan_version,
            now_ms: 0,
            next_replan_at_ms: None,
            queue_paused: false,
            host_memory: HostMemorySnapshot {
                headroom_bytes: self.host_headroom_bytes.saturating_sub(reserved),
                sample_generation: 1,
                ledger_sequence: 1,
            },
            devices,
            work,
        };
        let plan = Planner::default()
            .plan(&snapshot)
            .map_err(|error| anyhow::anyhow!("local scheduler rejected batch: {error}"))?;
        let mut leases = Vec::new();
        for lease in plan.immediate_leases {
            let index = lease
                .work_id
                .as_str()
                .strip_prefix("item:")
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| anyhow::anyhow!("local scheduler returned an invalid work id"))?;
            let ordinal = self
                .candidates
                .values()
                .find(|candidate| candidate.device_id == lease.device_id.as_str())
                .map(|candidate| candidate.ordinal)
                .ok_or_else(|| anyhow::anyhow!("local scheduler returned an invalid device id"))?;
            let position = self
                .pending
                .iter()
                .position(|pending| *pending == index)
                .ok_or_else(|| anyhow::anyhow!("local scheduler leased a non-pending item"))?;
            self.pending.remove(position);
            self.ready.remove(&ordinal);
            self.active.insert(ordinal, index);
            if let Some(candidate) = self.candidates.get(&ordinal) {
                self.resident_host_bytes
                    .entry(ordinal)
                    .or_insert(candidate.predicted_host_ram_bytes);
            }
            leases.push(LocalLease {
                ordinal,
                item_index: index,
            });
        }
        leases.sort_by_key(|lease| lease.item_index);
        Ok(leases)
    }
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

    fn candidates(count: usize) -> Vec<LocalCandidate> {
        (0..count)
            .map(|ordinal| LocalCandidate {
                ordinal,
                device_id: format!("cuda:{ordinal}"),
                execution_fingerprint: format!("exec:{ordinal}"),
                execution_equivalence_fingerprint:
                    mold_scheduler::ExecutionEquivalenceFingerprint::new("equivalent"),
                available_vram_bytes: 24 << 30,
                predicted_vram_bytes: 8 << 30,
                predicted_host_ram_bytes: 1 << 30,
            })
            .collect()
    }

    #[test]
    fn local_batch_admission_supports_zero_one_and_arbitrary_device_counts() {
        assert!(LocalBatchAdmission::new(&[], 1, 32 << 30).is_err());
        assert!(LocalBatchAdmission::new(&[], 0, 32 << 30)
            .unwrap()
            .lease_ready()
            .unwrap()
            .is_empty());

        for count in [1_usize, 2, 8, 16, 64] {
            let item_count = count * 2 + 1;
            let mut admission =
                LocalBatchAdmission::new(&candidates(count), item_count, 128 << 30).unwrap();
            for ordinal in 0..count {
                admission.owner_ready(ordinal).unwrap();
            }
            let mut leased = Vec::new();
            while admission.has_pending() || admission.has_active() {
                let wave = admission.lease_ready().unwrap();
                assert!(!wave.is_empty());
                for lease in wave {
                    leased.push(lease.item_index);
                    admission.owner_completed(lease.ordinal).unwrap();
                    admission.owner_ready(lease.ordinal).unwrap();
                }
            }
            leased.sort_unstable();
            assert_eq!(leased, (0..item_count).collect::<Vec<_>>());
        }
    }

    #[test]
    fn faster_owner_takes_next_global_item_instead_of_idling() {
        let mut admission = LocalBatchAdmission::new(&candidates(2), 4, 32 << 30).unwrap();
        admission.owner_ready(0).unwrap();
        admission.owner_ready(1).unwrap();
        let opening = admission.lease_ready().unwrap();
        assert_eq!(opening.len(), 2);
        let fast = opening.iter().find(|lease| lease.ordinal == 1).unwrap();
        admission.owner_completed(fast.ordinal).unwrap();
        admission.owner_ready(fast.ordinal).unwrap();

        let next = admission.lease_ready().unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].ordinal, 1);
        assert_eq!(next[0].item_index, 2);
    }

    #[test]
    fn failed_owner_does_not_strand_unstarted_global_items() {
        let mut admission = LocalBatchAdmission::new(&candidates(2), 4, 32 << 30).unwrap();
        admission.owner_ready(0).unwrap();
        admission.owner_ready(1).unwrap();
        let opening = admission.lease_ready().unwrap();
        assert_eq!(opening.len(), 2);
        let _ = admission.owner_failed(0);
        admission.owner_completed(1).unwrap();
        admission.owner_ready(1).unwrap();

        let next = admission.lease_ready().unwrap();
        assert_eq!(
            next,
            vec![LocalLease {
                ordinal: 1,
                item_index: 2,
            }]
        );
        assert!(admission.has_pending());
        assert!(admission.has_owners());
    }

    #[test]
    fn transport_failure_requeues_the_same_item_on_a_surviving_ready_owner() {
        let mut admission = LocalBatchAdmission::new(&candidates(2), 1, 32 << 30).unwrap();
        admission.owner_ready(0).unwrap();
        admission.owner_ready(1).unwrap();
        let failed = admission.lease_ready().unwrap().pop().unwrap();
        admission.lease_transport_failed(failed);

        let replacement = admission.lease_ready().unwrap();
        assert_eq!(replacement.len(), 1);
        assert_eq!(replacement[0].item_index, failed.item_index);
        assert_ne!(replacement[0].ordinal, failed.ordinal);
    }

    #[test]
    fn host_ram_admission_limits_only_current_leases_not_future_ownership() {
        let mut admission = LocalBatchAdmission::new(&candidates(2), 2, (2 << 30) - 1).unwrap();
        admission.owner_ready(0).unwrap();
        admission.owner_ready(1).unwrap();
        let first = admission.lease_ready().unwrap();
        assert_eq!(first.len(), 1);
        admission.owner_completed(first[0].ordinal).unwrap();
        admission.owner_ready(first[0].ordinal).unwrap();
        let second = admission.lease_ready().unwrap();
        assert_eq!(second.len(), 1);
        assert_ne!(first[0].item_index, second[0].item_index);
        assert_eq!(
            first[0].ordinal, second[0].ordinal,
            "the resident owner must retain its host-RAM charge and receive the next item"
        );
    }

    #[test]
    fn single_item_uses_the_same_gpu_and_host_admission_path() {
        let mut devices = candidates(2);
        devices[0].available_vram_bytes = 7 << 30;
        assert!(LocalBatchAdmission::new(&devices, 1, (1 << 30) - 1).is_err());

        let mut admission = LocalBatchAdmission::new(&devices, 1, 2 << 30).unwrap();
        admission.owner_ready(0).unwrap();
        admission.owner_ready(1).unwrap();
        assert_eq!(
            admission.lease_ready().unwrap(),
            vec![LocalLease {
                ordinal: 1,
                item_index: 0,
            }]
        );
    }
}
