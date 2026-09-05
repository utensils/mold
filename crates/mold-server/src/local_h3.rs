//! Owned, single-request local MiniMax H3 execution.

/// Local H3 attempts cannot be cached or reused across batch/chain requests.
/// Call before dependency preparation, which may open installed artifacts.
pub fn validate_invocation(
    is_h3: bool,
    batch: u32,
    chain: bool,
    references: bool,
) -> Result<(), &'static str> {
    if is_h3 && (batch != 1 || chain) {
        return Err("MiniMax H3 --local supports one request only; local batches and chains need per-request owned attempts");
    }
    if is_h3 && references {
        return Err(
            "MiniMax H3 local references require owned source bindings; use the server for Ref2VA",
        );
    }
    Ok(())
}

#[cfg(feature = "h3")]
pub use runtime::run_once;

#[cfg(feature = "h3")]
mod runtime {
    use crate::execution_plan::{PreparedExecutionInputs, ResolvedExecutionPlan};
    use crate::h3_attempt::{H3AttemptClaim, H3AttemptCurrent, H3AttemptRoot};
    use crate::h3_private_bridge::{self as bridge, H3AllocationCommit, H3PrivateUatPathSet};
    use anyhow::{anyhow, bail, Result};
    use mold_core::{Config, GenerateRequest, GenerateResponse};
    use mold_inference::progress::ProgressReporter;
    use mold_inference::{device, InferenceCancellationToken};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    #[cfg(feature = "cuda")]
    fn with_preparation_boundary<T>(
        ordinal: usize,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        use cudarc::driver::{CudaContext, CudaExecutionAttempt};
        let mut attempt = CudaExecutionAttempt::begin_unbound()?;
        // Drop this temporary retain while the containment boundary is still
        // installed. Retaining it past finish could run a CUDA destructor
        // after the boundary has deliberately retained a poisoned context.
        let adopted = CudaContext::new(ordinal)
            .map_err(|error| error.to_string())
            .and_then(|context| {
                attempt
                    .bind_context(&context)
                    .map_err(|error| error.to_string())
            });
        if let Err(error) = adopted {
            let status = attempt.finish();
            if status.resources_retained() {
                bail!("local H3 CUDA preparation retained resources; process restart required");
            }
            bail!("failed to adopt the local H3 CUDA context: {error}");
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
        if result.is_err() {
            attempt.mark_panicked();
        }
        let status = attempt.finish();
        if status.resources_retained() {
            // The driver boundary has retained unsafe CUDA resources. Never
            // drop a possibly GPU-bearing successful preparation in this arm.
            std::mem::forget(result);
            bail!("local H3 CUDA preparation retained resources; process restart required");
        }
        match result {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn with_preparation_boundary<T>(
        _ordinal: usize,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        operation()
    }

    /// Called on the selected local owner thread for exactly one admitted
    /// request. No executable authority is returned to the caller or cached.
    pub fn run_once(
        request: &GenerateRequest,
        config: &Config,
        plan: &ResolvedExecutionPlan,
        inputs: &PreparedExecutionInputs,
        cancellation: InferenceCancellationToken,
        progress: &mut ProgressReporter,
    ) -> Result<GenerateResponse> {
        cancellation.checkpoint()?;
        if device::thread_vram_grant_bytes().is_some() {
            bail!("local H3 cannot replace another owner's device grant");
        }
        super::validate_invocation(
            true,
            request.batch_size,
            false,
            request.references.is_some(),
        )
        .map_err(anyhow::Error::msg)?;
        crate::execution_plan::validate_before_cuda(
            plan,
            &plan.device_id,
            plan.device_ordinal,
            config,
            request,
            Some(inputs),
        )?;
        let grant = inputs
            .h3_private_ingress_grant
            .as_ref()
            .ok_or_else(|| anyhow!("local H3 lost its request authority"))?;
        grant
            .validate_bound_request(request)
            .map_err(anyhow::Error::msg)?;
        let evidence = inputs
            .h3_private_admission_by_device
            .get(&plan.device_id)
            .ok_or_else(|| anyhow!("local H3 lost its device admission evidence"))?;
        let factory = plan
            .engine_config
            .h3_factory_authority
            .as_ref()
            .ok_or_else(|| anyhow!("local H3 lost its factory authority"))?;
        let route = inputs
            .by_device
            .get(&plan.device_id)
            .ok_or_else(|| anyhow!("local H3 lost its prepared device route"))?;
        if route.engine_paths != plan.engine_paths
            || route.engine_config != plan.engine_config
            || factory != evidence.base_factory_authority()
            || plan.model_fingerprint != evidence.component_set_identity_sha256()
            || plan.execution_fingerprint != evidence.execution_fingerprint()
            || plan.predicted_vram_peak_bytes != evidence.predicted_device_peak_bytes()
            || plan.predicted_host_increment_bytes != evidence.predicted_host_increment_bytes()
            || plan.admitted_available_vram_bytes != evidence.admitted_available_device_bytes()
        {
            bail!("local H3 plan differs from immutable admission evidence");
        }
        let recheck = || -> Result<()> {
            cancellation.checkpoint()?;
            grant
                .validate_bound_request(request)
                .map_err(anyhow::Error::msg)?;
            let gpu = device::discover_gpus()
                .into_iter()
                .find(|gpu| gpu.ordinal == plan.device_ordinal)
                .ok_or_else(|| anyhow!("local H3 owner device disappeared"))?;
            if gpu.stable_id.as_deref() != Some(plan.device_id.as_str())
                || gpu.backend != plan.device_backend
                || gpu.compute_capability != evidence.compute_capability()
                || device::thread_gpu_ordinal() != Some(plan.device_ordinal)
            {
                bail!("local H3 owner device changed after admission");
            }
            let sampled = device::post_drop_free_vram_bytes(plan.device_ordinal)?;
            let available = if plan.device_backend == mold_core::GpuBackend::Metal {
                device::metal_unified_capacity_with_safety_floor(sampled)
            } else {
                sampled
            };
            let ram = crate::resources::ram_snapshot();
            let host = ram
                .available_with_evictable_arc()
                .saturating_sub((ram.total.saturating_mul(15) / 100).max(8 << 30));
            crate::gpu_worker::validate_private_h3_physical_capacity(
                &request.model,
                plan.predicted_vram_peak_bytes,
                available,
                plan.predicted_host_increment_bytes,
                host,
            )
            .map_err(|error| anyhow!(error.error))?;
            evidence.validate_for(
                request,
                mold_core::minimax_h3::ResolvedMediaPresence::from_request(request),
                &plan.device_id,
                plan.device_ordinal,
                gpu.compute_capability,
                available,
                host,
            )?;
            Ok(())
        };
        recheck()?;

        // One invocation owns one local reservation, sample, and cancellation
        // scope. A fresh work ID permits a later intentional repeat; the root
        // refuses duplicate claims/consumption of THIS work on the owner thread.
        let work_id = format!("local-h3:{}", uuid::Uuid::new_v4());
        let work = crate::h3_attempt::private_work_identity_sha256(&work_id);
        let cancel_scope = crate::h3_attempt::private_cancellation_scope_identity_sha256(
            &work,
            &plan.device_id,
            1,
            1,
            1,
            1,
            1,
            1,
        );
        let current = || H3AttemptCurrent {
            work_id: work_id.clone(),
            device_id: plan.device_id.clone(),
            device_ordinal: plan.device_ordinal,
            backend: plan.device_backend,
            owner_epoch: 1,
            worker_generation: 1,
            state_version: 1,
            plan_version: 1,
            memory_sample_generation: 1,
            memory_ledger_sequence: 1,
            execution_identity_sha256: factory.execution_fingerprint().into(),
            prepared_attempt_identity_sha256: evidence.prepared_attempt_identity_sha256().into(),
            target_budget_identity_sha256: evidence.target_budget_identity_sha256().into(),
            component_set_identity_sha256: factory.component_set_identity_sha256().into(),
            predicted_device_peak_bytes: plan.predicted_vram_peak_bytes,
            predicted_host_increment_bytes: plan.predicted_host_increment_bytes,
        };
        let root = H3AttemptRoot::claim(
            H3AttemptClaim {
                work_id: work_id.clone(),
                device_id: plan.device_id.clone(),
                device_ordinal: plan.device_ordinal,
                backend: plan.device_backend,
                owner_epoch: 1,
                worker_generation: 1,
                state_version: 1,
                plan_version: 1,
                memory_sample_generation: 1,
                memory_ledger_sequence: 1,
                execution_identity_sha256: evidence.execution_fingerprint().into(),
                prepared_attempt_identity_sha256: evidence
                    .prepared_attempt_identity_sha256()
                    .into(),
                target_budget_identity_sha256: evidence.target_budget_identity_sha256().into(),
                component_set_identity_sha256: evidence.component_set_identity_sha256().into(),
                predicted_device_peak_bytes: evidence.predicted_device_peak_bytes(),
                predicted_host_increment_bytes: evidence.predicted_host_increment_bytes(),
            },
            &current(),
            cancellation.clone(),
        )?;
        let paths = H3PrivateUatPathSet::resolve(plan.engine_config.artifact_root.clone());
        paths.ensure_staging_root();
        progress.set_cancellation_token(cancellation.clone());
        let prepared = with_preparation_boundary(plan.device_ordinal, || {
            bridge::prepare_bound_attempt(
                mold_inference::H3PrivateFl2VaPrepareInput {
                    request,
                    frozen_factory: factory,
                    admission_evidence: evidence,
                    paths: paths.inference_paths(),
                    references: Vec::new(),
                    owner_fence: mold_inference::H3PrivateFl2VaOwnerFenceFacts {
                        work_identity_sha256: work,
                        cancellation_scope_identity_sha256: cancel_scope,
                        device_id: plan.device_id.clone(),
                        device_ordinal: plan.device_ordinal,
                        compute_capability: evidence.compute_capability(),
                        memory_ledger_sequence: 1,
                        admission_evidence_identity_sha256: evidence.identity_sha256().into(),
                        artifact_qualification_identity_sha256: evidence
                            .artifact_qualification_identity_sha256()
                            .into(),
                        runtime_qualification_identity_sha256: evidence
                            .runtime_qualification_identity_sha256()
                            .into(),
                        prepared_attempt_identity_sha256: evidence
                            .prepared_attempt_identity_sha256()
                            .into(),
                        target_budget_identity_sha256: evidence
                            .target_budget_identity_sha256()
                            .into(),
                        predicted_device_peak_bytes: plan.predicted_vram_peak_bytes,
                        predicted_host_increment_bytes: plan.predicted_host_increment_bytes,
                    },
                },
                progress,
            )
            .map_err(anyhow::Error::msg)
        })?;
        let committed = Arc::new(AtomicBool::new(false));
        let commit_seen = Arc::clone(&committed);
        let expected_grant = plan.predicted_vram_peak_bytes;
        let commit_cancellation = cancellation.clone();
        let commit = H3AllocationCommit::new(move || {
            commit_cancellation.checkpoint()?;
            if device::thread_vram_grant_bytes() != Some(expected_grant)
                || commit_seen.swap(true, Ordering::SeqCst)
            {
                bail!("local H3 allocation commit lost its exact one-shot grant");
            }
            Ok(())
        });
        let mut prepared = std::mem::ManuallyDrop::new(prepared);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recheck()?;
            let _grant = crate::gpu_worker::ScopedThreadVramGrant::enter(Some(
                plan.predicted_vram_peak_bytes,
            ))
            .ok_or_else(|| anyhow!("local H3 has no exact device grant"))?;
            root.run_once(&current(), |scope| {
                bridge::run_bound_attempt(
                    &mut prepared,
                    scope,
                    request,
                    mold_core::minimax_h3::ResolvedMediaPresence::from_request(request),
                    progress,
                    commit,
                )
            })?
        }));
        progress.clear_cancellation_token();
        // Like the server owner, never destruct retained CUDA state after a
        // fatal driver error or panic. This local process exits on that error.
        match result {
            Err(_) => bail!("local H3 owner panicked; the attempt cannot be reused"),
            Ok(Err(error)) if crate::gpu_worker::is_fatal_cuda_error(&error) => Err(error),
            Ok(result) => {
                if result.as_ref().is_err_and(crate::gpu_worker::is_cuda_oom)
                    && device::post_drop_free_vram_bytes(plan.device_ordinal).is_err()
                {
                    bail!("local H3 CUDA owner could not synchronize after OOM; process restart required");
                }
                // SAFETY: exactly one ordinary completion reaches this arm.
                unsafe { std::mem::ManuallyDrop::drop(&mut prepared) };
                crate::gpu_worker::trim_malloc_arenas();
                let mut output = result?;
                if !committed.load(Ordering::SeqCst) {
                    bail!("local H3 returned without its allocation commit");
                }
                output.response.gpu = Some(plan.device_ordinal);
                Ok(output.response)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_h3_refuses_batches_and_chains_before_preparation() {
        for (batch, chain) in [(0, false), (2, false), (1, true)] {
            let mut prepared = false;
            let result = validate_invocation(true, batch, chain, false).map(|()| prepared = true);
            assert!(result.is_err());
            assert!(!prepared);
        }
        assert!(validate_invocation(true, 1, false, false).is_ok());
        assert!(validate_invocation(true, 1, false, true).is_err());
        assert!(validate_invocation(false, 2, true, true).is_ok());
    }

    #[test]
    fn cli_routes_h3_through_owned_attempts_before_the_engine_cache() {
        let source = include_str!("../../mold-cli/src/commands/generate.rs");
        let local = source
            .split("async fn generate_local_batch(")
            .nth(1)
            .unwrap();
        assert!(
            local.find("local_h3::validate_invocation").unwrap()
                < local.find("prepare_local_request(").unwrap()
        );
        assert!(
            local.find("base_req = local_batch_requests").unwrap()
                < local.find("plan_local_batch(").unwrap()
        );
        assert!(
            local.find("local_h3::run_once(").unwrap()
                < local.find("build_local_engine_from_plan(").unwrap()
        );
        let chain = include_str!("../../mold-cli/src/commands/chain.rs")
            .split("async fn run_chain_local(")
            .nth(1)
            .unwrap();
        assert!(
            chain.find("local_h3::validate_invocation").unwrap()
                < chain.find("resolve_or_pull_model(").unwrap()
        );
    }
}
