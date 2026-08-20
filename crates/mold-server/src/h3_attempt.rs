//! Owner-thread-only attempt authority for the future MiniMax H3 runtime.
//!
//! The scheduler transports cloneable plan facts to the GPU owner. This
//! module is the first boundary that turns those facts into singular behavior
//! ownership. The root deliberately cannot cross the worker transport or be
//! retained by an engine: it is non-`Clone`, non-`Send`, claimed only on the
//! concrete owner thread, and consumed once around one inference invocation.

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn private_work_identity_sha256(work_id: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-work.v1\0");
    digest.update((work_id.len() as u64).to_le_bytes());
    digest.update(work_id.as_bytes());
    format!("{:x}", digest.finalize())
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn private_cancellation_scope_identity_sha256(
    work_identity_sha256: &str,
    device_id: &str,
    owner_epoch: u64,
    state_version: u64,
    plan_version: u64,
    worker_generation: u64,
    memory_sample_generation: u64,
    memory_ledger_sequence: u64,
) -> String {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-cancellation-scope.v1\0");
    for value in [work_identity_sha256, device_id] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    for value in [
        owner_epoch,
        state_version,
        plan_version,
        worker_generation,
        memory_sample_generation,
        memory_ledger_sequence,
    ] {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3AttemptClaim {
    pub(crate) work_id: String,
    pub(crate) device_id: String,
    pub(crate) device_ordinal: usize,
    pub(crate) owner_epoch: u64,
    pub(crate) worker_generation: u64,
    pub(crate) state_version: u64,
    pub(crate) plan_version: u64,
    pub(crate) memory_sample_generation: u64,
    pub(crate) memory_ledger_sequence: u64,
    pub(crate) execution_identity_sha256: String,
    pub(crate) prepared_attempt_identity_sha256: String,
    pub(crate) target_budget_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
    pub(crate) predicted_device_peak_bytes: u64,
    pub(crate) predicted_host_increment_bytes: u64,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct H3AttemptCurrent {
    pub(crate) work_id: String,
    pub(crate) device_id: String,
    pub(crate) device_ordinal: usize,
    pub(crate) owner_epoch: u64,
    pub(crate) worker_generation: u64,
    pub(crate) state_version: u64,
    pub(crate) plan_version: u64,
    pub(crate) memory_sample_generation: u64,
    pub(crate) memory_ledger_sequence: u64,
    pub(crate) execution_identity_sha256: String,
    pub(crate) prepared_attempt_identity_sha256: String,
    pub(crate) target_budget_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
    pub(crate) predicted_device_peak_bytes: u64,
    pub(crate) predicted_host_increment_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3AttemptFenceKey {
    work_id: String,
    device_id: String,
    device_ordinal: usize,
    owner_epoch: u64,
    worker_generation: u64,
    state_version: u64,
    plan_version: u64,
    memory_sample_generation: u64,
    memory_ledger_sequence: u64,
}

impl From<&H3AttemptClaim> for H3AttemptFenceKey {
    fn from(claim: &H3AttemptClaim) -> Self {
        Self {
            work_id: claim.work_id.clone(),
            device_id: claim.device_id.clone(),
            device_ordinal: claim.device_ordinal,
            owner_epoch: claim.owner_epoch,
            worker_generation: claim.worker_generation,
            state_version: claim.state_version,
            plan_version: claim.plan_version,
            memory_sample_generation: claim.memory_sample_generation,
            memory_ledger_sequence: claim.memory_ledger_sequence,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3ActiveAttempt {
    fence: H3AttemptFenceKey,
    nonce: String,
}

#[derive(Default)]
struct H3OwnerAttemptSlot {
    active: Option<H3ActiveAttempt>,
    last_settled_fence: Option<H3AttemptFenceKey>,
    last_settled_nonce: Option<String>,
}

thread_local! {
    /// One slot per concrete owner thread. GPU workers already serialize
    /// grants; this local slot makes accidental nested/replayed H3 attempts a
    /// typed rejection without adding state to the worker transport or cache.
    static H3_OWNER_ATTEMPT_SLOT: RefCell<H3OwnerAttemptSlot> =
        RefCell::new(H3OwnerAttemptSlot::default());
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum H3AttemptError {
    InvalidClaim,
    StaleOwnerFence,
    IdentityMismatch,
    DuplicateActiveAttempt,
    DuplicateSettledFence,
    DuplicateNonce,
    Cancelled,
    RuntimeUnavailable,
}

impl fmt::Display for H3AttemptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidClaim => "MiniMax H3 attempt lease facts are incomplete",
            Self::StaleOwnerFence => "MiniMax H3 attempt lease no longer belongs to this owner",
            Self::IdentityMismatch => "MiniMax H3 attempt identity changed before consumption",
            Self::DuplicateActiveAttempt => "MiniMax H3 owner already has an active attempt lease",
            Self::DuplicateSettledFence => "MiniMax H3 attempt lease fence was already settled",
            Self::DuplicateNonce => "MiniMax H3 attempt nonce was already settled",
            Self::Cancelled => "MiniMax H3 attempt was cancelled before execution",
            Self::RuntimeUnavailable => {
                "MiniMax H3 claimed-attempt runtime bridge is not available"
            }
        })
    }
}

impl std::error::Error for H3AttemptError {}

#[derive(Debug, Default)]
struct H3AttemptSettlementProbe {
    #[cfg(test)]
    count: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
}

impl H3AttemptSettlementProbe {
    fn disabled() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn new(count: std::sync::Arc<std::sync::atomic::AtomicUsize>) -> Self {
        Self { count: Some(count) }
    }

    fn record(&self) {
        #[cfg(test)]
        if let Some(count) = &self.count {
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// Singular owner of one future H3 execution attempt.
///
/// `Rc` in the marker is intentional: even if every value field later becomes
/// `Send`, this root remains pinned to the OS thread that claimed it.
pub(crate) struct H3AttemptRoot {
    claim: H3AttemptClaim,
    nonce: String,
    cancellation: mold_inference::InferenceCancellationToken,
    owner_thread: std::thread::ThreadId,
    settled: bool,
    settlement_probe: H3AttemptSettlementProbe,
    _owner_thread_only: PhantomData<Rc<()>>,
}

impl fmt::Debug for H3AttemptRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("H3AttemptRoot")
            .field("work_id", &self.claim.work_id)
            .field("device_id", &self.claim.device_id)
            .field("device_ordinal", &self.claim.device_ordinal)
            .field("settled", &self.settled)
            .finish_non_exhaustive()
    }
}

impl H3AttemptRoot {
    pub(crate) fn claim(
        claim: H3AttemptClaim,
        current: &H3AttemptCurrent,
        cancellation: mold_inference::InferenceCancellationToken,
    ) -> Result<Self, H3AttemptError> {
        Self::claim_inner(
            claim,
            current,
            cancellation,
            uuid::Uuid::new_v4().to_string(),
            H3AttemptSettlementProbe::disabled(),
        )
    }

    #[cfg(test)]
    fn claim_for_test(
        claim: H3AttemptClaim,
        current: &H3AttemptCurrent,
        cancellation: mold_inference::InferenceCancellationToken,
        nonce: &str,
        settlement_probe: H3AttemptSettlementProbe,
    ) -> Result<Self, H3AttemptError> {
        Self::claim_inner(
            claim,
            current,
            cancellation,
            nonce.to_string(),
            settlement_probe,
        )
    }

    fn claim_inner(
        claim: H3AttemptClaim,
        current: &H3AttemptCurrent,
        cancellation: mold_inference::InferenceCancellationToken,
        nonce: String,
        settlement_probe: H3AttemptSettlementProbe,
    ) -> Result<Self, H3AttemptError> {
        validate_claim(&claim)?;
        validate_current(&claim, current)?;
        validate_nonce(&nonce)?;
        let fence = H3AttemptFenceKey::from(&claim);
        H3_OWNER_ATTEMPT_SLOT.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.active.is_some() {
                return Err(H3AttemptError::DuplicateActiveAttempt);
            }
            if slot.last_settled_fence.as_ref() == Some(&fence) {
                return Err(H3AttemptError::DuplicateSettledFence);
            }
            if slot.last_settled_nonce.as_deref() == Some(nonce.as_str()) {
                return Err(H3AttemptError::DuplicateNonce);
            }
            slot.active = Some(H3ActiveAttempt {
                fence,
                nonce: nonce.clone(),
            });
            Ok(())
        })?;
        Ok(Self {
            claim,
            nonce,
            cancellation,
            owner_thread: std::thread::current().id(),
            settled: false,
            settlement_probe,
            _owner_thread_only: PhantomData,
        })
    }

    /// Consume the root exactly once at the H3 invocation seam.
    ///
    /// The independently supplied current projection is deliberate: a future
    /// runtime must echo the exact execution, prepared-attempt, target-budget,
    /// and component-set identities instead of receiving a self-authenticating
    /// handle.
    pub(crate) fn run_once<T>(
        mut self,
        current: &H3AttemptCurrent,
        consume: impl FnOnce(H3AttemptScope<'_>) -> T,
    ) -> Result<T, H3AttemptError> {
        if std::thread::current().id() != self.owner_thread {
            return Err(H3AttemptError::StaleOwnerFence);
        }
        validate_current(&self.claim, current)?;
        self.cancellation
            .checkpoint()
            .map_err(|_| H3AttemptError::Cancelled)?;
        let result = consume(H3AttemptScope {
            cancellation: &self.cancellation,
            #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
            claim: &self.claim,
        });
        self.settle();
        Ok(result)
    }

    fn settle(&mut self) {
        if self.settled {
            return;
        }
        let fence = H3AttemptFenceKey::from(&self.claim);
        H3_OWNER_ATTEMPT_SLOT.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.active.as_ref()
                == Some(&H3ActiveAttempt {
                    fence: fence.clone(),
                    nonce: self.nonce.clone(),
                })
            {
                slot.active = None;
                slot.last_settled_fence = Some(fence);
                slot.last_settled_nonce = Some(self.nonce.clone());
            } else {
                debug_assert!(
                    false,
                    "H3 attempt root lost its exact owner-thread active slot"
                );
            }
        });
        self.settled = true;
        self.settlement_probe.record();
    }
}

impl Drop for H3AttemptRoot {
    fn drop(&mut self) {
        self.settle();
    }
}

pub(crate) struct H3AttemptScope<'a> {
    cancellation: &'a mold_inference::InferenceCancellationToken,
    #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
    claim: &'a H3AttemptClaim,
}

impl<'a> H3AttemptScope<'a> {
    #[cfg(test)]
    pub(crate) fn checkpoint(&self) -> Result<(), mold_inference::InferenceCancelled> {
        self.cancellation.checkpoint()
    }

    pub(crate) fn cancellation_token(&self) -> mold_inference::InferenceCancellationToken {
        self.cancellation.clone()
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn private_run_context(
        &self,
    ) -> anyhow::Result<mold_inference::H3PrivateFl2VaRunContext> {
        let work_identity_sha256 = private_work_identity_sha256(&self.claim.work_id);
        let cancellation_scope_identity_sha256 = private_cancellation_scope_identity_sha256(
            &work_identity_sha256,
            &self.claim.device_id,
            self.claim.owner_epoch,
            self.claim.state_version,
            self.claim.plan_version,
            self.claim.worker_generation,
            self.claim.memory_sample_generation,
            self.claim.memory_ledger_sequence,
        );
        mold_inference::H3PrivateFl2VaRunContext::new(
            work_identity_sha256,
            cancellation_scope_identity_sha256,
            self.claim.memory_ledger_sequence,
            self.cancellation.clone(),
        )
    }

    /// Payload-free, read-only owner facts exposed to the private inference
    /// facade. Scheduler sequence numbers and mutable worker state remain
    /// inside this module; the runtime receives only the identities it must
    /// independently echo at terminal publication.
    #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
    pub(crate) fn facts(&self) -> H3AttemptScopeFacts<'a> {
        H3AttemptScopeFacts {
            work_id: &self.claim.work_id,
            device_id: &self.claim.device_id,
            device_ordinal: self.claim.device_ordinal,
            owner_epoch: self.claim.owner_epoch,
            worker_generation: self.claim.worker_generation,
            state_version: self.claim.state_version,
            plan_version: self.claim.plan_version,
            memory_sample_generation: self.claim.memory_sample_generation,
            memory_ledger_sequence: self.claim.memory_ledger_sequence,
            execution_identity_sha256: &self.claim.execution_identity_sha256,
            prepared_attempt_identity_sha256: &self.claim.prepared_attempt_identity_sha256,
            target_budget_identity_sha256: &self.claim.target_budget_identity_sha256,
            component_set_identity_sha256: &self.claim.component_set_identity_sha256,
            predicted_device_peak_bytes: self.claim.predicted_device_peak_bytes,
            predicted_host_increment_bytes: self.claim.predicted_host_increment_bytes,
        }
    }
}

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct H3AttemptScopeFacts<'a> {
    work_id: &'a str,
    device_id: &'a str,
    device_ordinal: usize,
    owner_epoch: u64,
    worker_generation: u64,
    state_version: u64,
    plan_version: u64,
    memory_sample_generation: u64,
    memory_ledger_sequence: u64,
    execution_identity_sha256: &'a str,
    prepared_attempt_identity_sha256: &'a str,
    target_budget_identity_sha256: &'a str,
    component_set_identity_sha256: &'a str,
    predicted_device_peak_bytes: u64,
    predicted_host_increment_bytes: u64,
}

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
impl<'a> H3AttemptScopeFacts<'a> {
    pub(crate) fn device_id(self) -> &'a str {
        self.device_id
    }

    pub(crate) fn device_ordinal(self) -> usize {
        self.device_ordinal
    }

    pub(crate) fn matches_lease(self, lease: &crate::scheduler::LeaseFence) -> bool {
        lease.work_id == self.work_id
            && lease.device_id == self.device_id
            && lease.owner_epoch == self.owner_epoch
            && lease.worker_generation == self.worker_generation
            && lease.state_version == self.state_version
            && lease.plan_version == self.plan_version
            && lease.memory_sample_generation == self.memory_sample_generation
            && lease.memory_ledger_sequence == self.memory_ledger_sequence
    }

    pub(crate) fn execution_identity_sha256(self) -> &'a str {
        self.execution_identity_sha256
    }

    pub(crate) fn prepared_attempt_identity_sha256(self) -> &'a str {
        self.prepared_attempt_identity_sha256
    }

    pub(crate) fn target_budget_identity_sha256(self) -> &'a str {
        self.target_budget_identity_sha256
    }

    pub(crate) fn component_set_identity_sha256(self) -> &'a str {
        self.component_set_identity_sha256
    }

    pub(crate) fn predicted_device_peak_bytes(self) -> u64 {
        self.predicted_device_peak_bytes
    }

    pub(crate) fn predicted_host_increment_bytes(self) -> u64 {
        self.predicted_host_increment_bytes
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn matches_private_run_binding(
        self,
        work_identity_sha256: &str,
        cancellation_scope_identity_sha256: &str,
        memory_ledger_sequence: u64,
    ) -> bool {
        let expected_work_identity = private_work_identity_sha256(self.work_id);
        let expected_cancellation_scope = private_cancellation_scope_identity_sha256(
            &expected_work_identity,
            self.device_id,
            self.owner_epoch,
            self.state_version,
            self.plan_version,
            self.worker_generation,
            self.memory_sample_generation,
            self.memory_ledger_sequence,
        );
        work_identity_sha256 == expected_work_identity
            && cancellation_scope_identity_sha256 == expected_cancellation_scope
            && memory_ledger_sequence == self.memory_ledger_sequence
    }
}

/// Local stack value spanning the second pre-CUDA fence and exactly one H3
/// invocation. It is intentionally absent from every scheduler, job, plan,
/// registry, engine, and cache type.
pub(crate) struct H3GenerationAttempt {
    root: H3AttemptRoot,
}

impl H3GenerationAttempt {
    pub(crate) fn run_once<T>(
        self,
        current: H3AttemptCurrent,
        consume: impl FnOnce(H3AttemptScope<'_>) -> T,
    ) -> Result<T, H3AttemptError> {
        self.root.run_once(&current, consume)
    }
}

#[cfg(test)]
pub(crate) fn generation_attempt_for_test(
    work_id: &str,
    cancellation: mold_inference::InferenceCancellationToken,
) -> (
    H3GenerationAttempt,
    H3AttemptCurrent,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
) {
    let identity = |byte: char| std::iter::repeat_n(byte, 64).collect::<String>();
    let claim = H3AttemptClaim {
        work_id: work_id.to_string(),
        device_id: "cuda:0".to_string(),
        device_ordinal: 0,
        owner_epoch: 7,
        worker_generation: 11,
        state_version: 13,
        plan_version: 17,
        memory_sample_generation: 19,
        memory_ledger_sequence: 23,
        execution_identity_sha256: identity('a'),
        prepared_attempt_identity_sha256: identity('b'),
        target_budget_identity_sha256: identity('c'),
        component_set_identity_sha256: identity('d'),
        predicted_device_peak_bytes: 11_000_000_000,
        predicted_host_increment_bytes: 2_000_000_000,
    };
    let current = H3AttemptCurrent {
        work_id: claim.work_id.clone(),
        device_id: claim.device_id.clone(),
        device_ordinal: claim.device_ordinal,
        owner_epoch: claim.owner_epoch,
        worker_generation: claim.worker_generation,
        state_version: claim.state_version,
        plan_version: claim.plan_version,
        memory_sample_generation: claim.memory_sample_generation,
        memory_ledger_sequence: claim.memory_ledger_sequence,
        execution_identity_sha256: claim.execution_identity_sha256.clone(),
        prepared_attempt_identity_sha256: claim.prepared_attempt_identity_sha256.clone(),
        target_budget_identity_sha256: claim.target_budget_identity_sha256.clone(),
        component_set_identity_sha256: claim.component_set_identity_sha256.clone(),
        predicted_device_peak_bytes: claim.predicted_device_peak_bytes,
        predicted_host_increment_bytes: claim.predicted_host_increment_bytes,
    };
    let settlements = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let root = H3AttemptRoot::claim_inner(
        claim,
        &current,
        cancellation,
        uuid::Uuid::new_v4().to_string(),
        H3AttemptSettlementProbe::new(std::sync::Arc::clone(&settlements)),
    )
    .expect("synthetic H3 owner attempt must be internally valid");
    (H3GenerationAttempt { root }, current, settlements)
}

/// Claim an H3 attempt from exact owner-local facts.
///
/// Under the private feature, the final-dispatch inference facade must first
/// place its opaque attempt on the job. The claim copies only that value's
/// payload-free identities after comparing them with the still-frozen base
/// plan; an absent payload remains a non-H3/no-runtime outcome.
pub(crate) fn claim_generation_attempt(
    worker: &crate::gpu_pool::GpuWorker,
    current_worker_generation: u64,
    fence: &crate::scheduler::LeaseFence,
    job: &crate::gpu_pool::GpuJob,
    cancellation: mold_inference::InferenceCancellationToken,
) -> Result<Option<H3GenerationAttempt>, H3AttemptError> {
    if worker.owner_thread_id.get().copied() != Some(std::thread::current().id()) {
        return Err(H3AttemptError::StaleOwnerFence);
    }
    let Some(plan) = job.execution_plan.as_ref() else {
        return Ok(None);
    };
    if !mold_core::minimax_h3::is_family(&plan.model_family) {
        return Ok(None);
    }
    let authority = plan
        .engine_config
        .h3_factory_authority
        .as_ref()
        .ok_or(H3AttemptError::IdentityMismatch)?;
    #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
    let (prepared_attempt_identity, target_budget_identity) = {
        let Some(prepared) = job.h3_prepared_attempt.as_ref() else {
            return Ok(None);
        };
        let facts = prepared.facts();
        validate_prepared_facts_against_plan(&facts, plan, authority)?;
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        validate_private_prepared_binding(&facts, fence, job)?;
        (
            facts.prepared_attempt_identity_sha256,
            facts.target_budget_identity_sha256,
        )
    };
    #[cfg(not(any(test, feature = "h3-private-bridge", feature = "h3-private-uat")))]
    let (prepared_attempt_identity, target_budget_identity) = authority
        .prepared_target_attempt_identities()
        .map(|(attempt, budget)| (attempt.to_string(), budget.to_string()))
        .unwrap_or_else(|| (String::new(), String::new()));
    #[cfg(not(any(test, feature = "h3-private-bridge", feature = "h3-private-uat")))]
    if prepared_attempt_identity.is_empty() || target_budget_identity.is_empty() {
        return Ok(None);
    }
    let worker_device_id = crate::scheduler::worker_device_id(worker);
    let invocation = H3AttemptClaim {
        work_id: fence.work_id.clone(),
        device_id: fence.device_id.clone(),
        device_ordinal: worker.gpu.ordinal,
        owner_epoch: fence.owner_epoch,
        worker_generation: fence.worker_generation,
        state_version: fence.state_version,
        plan_version: fence.plan_version,
        memory_sample_generation: fence.memory_sample_generation,
        memory_ledger_sequence: fence.memory_ledger_sequence,
        execution_identity_sha256: plan.execution_fingerprint.clone(),
        prepared_attempt_identity_sha256: prepared_attempt_identity.clone(),
        target_budget_identity_sha256: target_budget_identity.clone(),
        component_set_identity_sha256: authority.component_set_identity_sha256().to_string(),
        predicted_device_peak_bytes: plan.predicted_vram_peak_bytes,
        predicted_host_increment_bytes: plan.predicted_host_increment_bytes,
    };
    let current = H3AttemptCurrent {
        work_id: job.id.clone(),
        device_id: worker_device_id.clone(),
        device_ordinal: plan.device_ordinal,
        owner_epoch: worker.owner_epoch,
        worker_generation: current_worker_generation,
        state_version: fence.state_version,
        plan_version: fence.plan_version,
        memory_sample_generation: fence.memory_sample_generation,
        memory_ledger_sequence: fence.memory_ledger_sequence,
        execution_identity_sha256: authority.execution_fingerprint().to_string(),
        prepared_attempt_identity_sha256: prepared_attempt_identity,
        target_budget_identity_sha256: target_budget_identity,
        component_set_identity_sha256: authority.component_set_identity_sha256().to_string(),
        predicted_device_peak_bytes: plan.predicted_vram_peak_bytes,
        predicted_host_increment_bytes: plan.predicted_host_increment_bytes,
    };
    if authority.device_id() != worker_device_id || authority.device_ordinal() != worker.gpu.ordinal
    {
        return Err(H3AttemptError::StaleOwnerFence);
    }
    let root = H3AttemptRoot::claim(invocation, &current, cancellation)?;
    Ok(Some(H3GenerationAttempt { root }))
}

/// Rebuild the consuming-seam projection without consulting the attempt root.
///
/// The lease fields come from the exact fence installed on the job after
/// acceptance. Owner and device facts come from the live worker, while the
/// execution and component identities come from the still-frozen plan while
/// prepared-attempt and target-budget identities are re-read from the opaque
/// inference value. This independently detects replacement between claim and
/// consumption without retaining the attempt inside the scheduler plan.
pub(crate) fn rebuild_generation_current(
    worker: &crate::gpu_pool::GpuWorker,
    current_worker_generation: u64,
    job: &crate::gpu_pool::GpuJob,
) -> Result<H3AttemptCurrent, H3AttemptError> {
    if worker.owner_thread_id.get().copied() != Some(std::thread::current().id()) {
        return Err(H3AttemptError::StaleOwnerFence);
    }
    let lease = job.lease.as_ref().ok_or(H3AttemptError::StaleOwnerFence)?;
    let plan = job
        .execution_plan
        .as_ref()
        .ok_or(H3AttemptError::IdentityMismatch)?;
    if !mold_core::minimax_h3::is_family(&plan.model_family) {
        return Err(H3AttemptError::IdentityMismatch);
    }
    let authority = plan
        .engine_config
        .h3_factory_authority
        .as_ref()
        .ok_or(H3AttemptError::IdentityMismatch)?;
    #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
    let (prepared_attempt_identity, target_budget_identity) = {
        let prepared = job
            .h3_prepared_attempt
            .as_ref()
            .ok_or(H3AttemptError::IdentityMismatch)?;
        let facts = prepared.facts();
        validate_prepared_facts_against_plan(&facts, plan, authority)?;
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        validate_private_prepared_binding(&facts, lease, job)?;
        (
            facts.prepared_attempt_identity_sha256,
            facts.target_budget_identity_sha256,
        )
    };
    #[cfg(not(any(test, feature = "h3-private-bridge", feature = "h3-private-uat")))]
    let (prepared_attempt_identity, target_budget_identity) = authority
        .prepared_target_attempt_identities()
        .map(|(attempt, budget)| (attempt.to_string(), budget.to_string()))
        .ok_or(H3AttemptError::IdentityMismatch)?;
    let worker_device_id = crate::scheduler::worker_device_id(worker);
    if lease.work_id != job.id
        || lease.device_id != worker_device_id
        || lease.owner_epoch != worker.owner_epoch
        || lease.worker_generation != current_worker_generation
        || plan.device_ordinal != worker.gpu.ordinal
        || authority.device_id() != worker_device_id
        || authority.device_ordinal() != worker.gpu.ordinal
    {
        return Err(H3AttemptError::StaleOwnerFence);
    }
    Ok(H3AttemptCurrent {
        work_id: lease.work_id.clone(),
        device_id: lease.device_id.clone(),
        device_ordinal: plan.device_ordinal,
        owner_epoch: lease.owner_epoch,
        worker_generation: lease.worker_generation,
        state_version: lease.state_version,
        plan_version: lease.plan_version,
        memory_sample_generation: lease.memory_sample_generation,
        memory_ledger_sequence: lease.memory_ledger_sequence,
        execution_identity_sha256: authority.execution_fingerprint().to_string(),
        prepared_attempt_identity_sha256: prepared_attempt_identity,
        target_budget_identity_sha256: target_budget_identity,
        component_set_identity_sha256: authority.component_set_identity_sha256().to_string(),
        predicted_device_peak_bytes: plan.predicted_vram_peak_bytes,
        predicted_host_increment_bytes: plan.predicted_host_increment_bytes,
    })
}

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
fn validate_prepared_facts_against_plan(
    facts: &crate::h3_private_bridge::H3PreparedAttemptFacts,
    plan: &crate::execution_plan::ResolvedExecutionPlan,
    authority: &mold_inference::FrozenH3FactoryAuthority,
) -> Result<(), H3AttemptError> {
    if facts.device_id != authority.device_id()
        || facts.device_ordinal != plan.device_ordinal
        || facts.device_ordinal != authority.device_ordinal()
        || facts.execution_identity_sha256 != plan.execution_fingerprint
        || facts.execution_identity_sha256 != authority.execution_fingerprint()
        || facts.component_set_identity_sha256 != authority.component_set_identity_sha256()
        || facts.predicted_device_peak_bytes != plan.predicted_vram_peak_bytes
        || facts.predicted_host_increment_bytes != plan.predicted_host_increment_bytes
        || !mold_inference::media_model_matches_h3_authority(
            &facts.media.canonical_model,
            authority,
        )
        || facts.media.task != authority.task()
        || facts.media.fps != mold_core::minimax_h3::FIXED_FPS
        || !valid_sha256(&facts.prepared_attempt_identity_sha256)
        || !valid_sha256(&facts.target_budget_identity_sha256)
        || !valid_sha256(&facts.admission_evidence_identity_sha256)
        || !valid_sha256(&facts.artifact_qualification_identity_sha256)
        || !valid_sha256(&facts.runtime_qualification_identity_sha256)
        || !valid_sha256(&facts.work_identity_sha256)
        || !valid_sha256(&facts.cancellation_scope_identity_sha256)
        || !valid_sha256(&facts.consumption_identity_sha256)
        || facts.memory_ledger_sequence == 0
    {
        return Err(H3AttemptError::IdentityMismatch);
    }
    match facts.media.task {
        mold_core::minimax_h3::Task::Fl2va => {
            if facts.media.mode == mold_core::minimax_h3::Mode::ReferenceToAudioVideo
                || facts.media.reference_count != 0
                || facts.media.reference_fingerprint_sha256.is_some()
                || facts.media.resolved_reference_fingerprint_sha256.is_some()
            {
                return Err(H3AttemptError::IdentityMismatch);
            }
        }
        mold_core::minimax_h3::Task::Ref2va => {
            if facts.media.mode != mold_core::minimax_h3::Mode::ReferenceToAudioVideo
                || facts.media.reference_count == 0
                || !facts
                    .media
                    .reference_fingerprint_sha256
                    .as_deref()
                    .is_some_and(valid_sha256)
                || !facts
                    .media
                    .resolved_reference_fingerprint_sha256
                    .as_deref()
                    .is_some_and(valid_sha256)
            {
                return Err(H3AttemptError::IdentityMismatch);
            }
        }
    }
    Ok(())
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn validate_private_prepared_binding(
    facts: &crate::h3_private_bridge::H3PreparedAttemptFacts,
    fence: &crate::scheduler::LeaseFence,
    job: &crate::gpu_pool::GpuJob,
) -> Result<(), H3AttemptError> {
    let prepared = job
        .prepared_execution_inputs
        .as_ref()
        .ok_or(H3AttemptError::IdentityMismatch)?;
    prepared
        .h3_private_ingress_grant
        .as_ref()
        .ok_or(H3AttemptError::IdentityMismatch)?
        .validate_bound_request(&job.request)
        .map_err(|_| H3AttemptError::IdentityMismatch)?;
    let evidence = prepared
        .h3_private_admission_by_device
        .get(&fence.device_id)
        .ok_or(H3AttemptError::IdentityMismatch)?;
    let work_identity_sha256 = private_work_identity_sha256(&fence.work_id);
    let cancellation_scope_identity_sha256 = private_cancellation_scope_identity_sha256(
        &work_identity_sha256,
        &fence.device_id,
        fence.owner_epoch,
        fence.state_version,
        fence.plan_version,
        fence.worker_generation,
        fence.memory_sample_generation,
        fence.memory_ledger_sequence,
    );
    if facts.admission_evidence_identity_sha256 != evidence.identity_sha256()
        || facts.artifact_qualification_identity_sha256
            != evidence.artifact_qualification_identity_sha256()
        || facts.runtime_qualification_identity_sha256
            != evidence.runtime_qualification_identity_sha256()
        || facts.prepared_attempt_identity_sha256 != evidence.prepared_attempt_identity_sha256()
        || facts.target_budget_identity_sha256 != evidence.target_budget_identity_sha256()
        || facts.work_identity_sha256 != work_identity_sha256
        || facts.cancellation_scope_identity_sha256 != cancellation_scope_identity_sha256
        || facts.memory_ledger_sequence != fence.memory_ledger_sequence
    {
        return Err(H3AttemptError::IdentityMismatch);
    }
    Ok(())
}

fn validate_claim(claim: &H3AttemptClaim) -> Result<(), H3AttemptError> {
    if claim.work_id.trim().is_empty()
        || claim.device_id.trim().is_empty()
        || claim.owner_epoch == 0
        || claim.worker_generation == 0
        || claim.state_version == 0
        || claim.plan_version == 0
        || claim.memory_sample_generation == 0
        || claim.memory_ledger_sequence == 0
        || !valid_sha256(&claim.execution_identity_sha256)
        || !valid_sha256(&claim.prepared_attempt_identity_sha256)
        || !valid_sha256(&claim.target_budget_identity_sha256)
        || !valid_sha256(&claim.component_set_identity_sha256)
        || claim.predicted_device_peak_bytes == 0
        || claim.predicted_host_increment_bytes == 0
    {
        return Err(H3AttemptError::InvalidClaim);
    }
    Ok(())
}

fn validate_current(
    claim: &H3AttemptClaim,
    current: &H3AttemptCurrent,
) -> Result<(), H3AttemptError> {
    if claim.work_id != current.work_id
        || claim.device_id != current.device_id
        || claim.device_ordinal != current.device_ordinal
        || claim.owner_epoch != current.owner_epoch
        || claim.worker_generation != current.worker_generation
        || claim.state_version != current.state_version
        || claim.plan_version != current.plan_version
        || claim.memory_sample_generation != current.memory_sample_generation
        || claim.memory_ledger_sequence != current.memory_ledger_sequence
    {
        return Err(H3AttemptError::StaleOwnerFence);
    }
    if claim.execution_identity_sha256 != current.execution_identity_sha256
        || claim.prepared_attempt_identity_sha256 != current.prepared_attempt_identity_sha256
        || claim.target_budget_identity_sha256 != current.target_budget_identity_sha256
        || claim.component_set_identity_sha256 != current.component_set_identity_sha256
        || claim.predicted_device_peak_bytes != current.predicted_device_peak_bytes
        || claim.predicted_host_increment_bytes != current.predicted_host_increment_bytes
    {
        return Err(H3AttemptError::IdentityMismatch);
    }
    Ok(())
}

fn validate_nonce(nonce: &str) -> Result<(), H3AttemptError> {
    let nonce = uuid::Uuid::parse_str(nonce).map_err(|_| H3AttemptError::InvalidClaim)?;
    if nonce.get_version_num() != 4 {
        return Err(H3AttemptError::InvalidClaim);
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn fixture(work_id: &str) -> H3AttemptClaim {
        H3AttemptClaim {
            work_id: work_id.into(),
            device_id: "cuda:0".into(),
            device_ordinal: 0,
            owner_epoch: 7,
            worker_generation: 11,
            state_version: 13,
            plan_version: 17,
            memory_sample_generation: 19,
            memory_ledger_sequence: 23,
            execution_identity_sha256: sha('a'),
            prepared_attempt_identity_sha256: sha('b'),
            target_budget_identity_sha256: sha('c'),
            component_set_identity_sha256: sha('d'),
            predicted_device_peak_bytes: 11_000_000_000,
            predicted_host_increment_bytes: 2_000_000_000,
        }
    }

    fn current(claim: &H3AttemptClaim) -> H3AttemptCurrent {
        H3AttemptCurrent {
            work_id: claim.work_id.clone(),
            device_id: claim.device_id.clone(),
            device_ordinal: claim.device_ordinal,
            owner_epoch: claim.owner_epoch,
            worker_generation: claim.worker_generation,
            state_version: claim.state_version,
            plan_version: claim.plan_version,
            memory_sample_generation: claim.memory_sample_generation,
            memory_ledger_sequence: claim.memory_ledger_sequence,
            execution_identity_sha256: claim.execution_identity_sha256.clone(),
            prepared_attempt_identity_sha256: claim.prepared_attempt_identity_sha256.clone(),
            target_budget_identity_sha256: claim.target_budget_identity_sha256.clone(),
            component_set_identity_sha256: claim.component_set_identity_sha256.clone(),
            predicted_device_peak_bytes: claim.predicted_device_peak_bytes,
            predicted_host_increment_bytes: claim.predicted_host_increment_bytes,
        }
    }

    fn probe() -> (H3AttemptSettlementProbe, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        (H3AttemptSettlementProbe::new(Arc::clone(&count)), count)
    }

    fn item_body<'a>(source: &'a str, start: &str, next: &str) -> &'a str {
        source
            .split_once(start)
            .unwrap_or_else(|| panic!("missing source item {start}"))
            .1
            .split_once(next)
            .unwrap_or_else(|| panic!("missing source item boundary {next}"))
            .0
    }

    #[test]
    fn attempt_root_is_absent_from_cloneable_transport_and_cache_owners() {
        let gpu_pool = include_str!("gpu_pool.rs");
        let gpu_job = item_body(
            gpu_pool,
            "pub struct GpuJob",
            "pub struct PromptExpansionJob",
        );
        let execution_plan = include_str!("execution_plan.rs");
        let resolved_plan = item_body(
            execution_plan,
            "pub struct ResolvedExecutionPlan",
            "pub struct PlannedLora",
        );
        let frozen_factory = include_str!("../../mold-inference/src/factory.rs");
        let frozen_config = item_body(
            frozen_factory,
            "pub struct FrozenEngineConfig",
            "impl FrozenEngineConfig",
        );
        for (label, source) in [
            ("GpuJob", gpu_job),
            ("ResolvedExecutionPlan", resolved_plan),
            ("FrozenEngineConfig", frozen_config),
            ("job registry", include_str!("job_registry.rs")),
            ("model cache", include_str!("model_cache.rs")),
            (
                "inference engine/session ownership",
                include_str!("../../mold-inference/src/engine.rs"),
            ),
        ] {
            assert!(
                !source.contains("H3AttemptRoot") && !source.contains("H3GenerationAttempt"),
                "{label} must not retain H3 one-shot attempt ownership",
            );
        }
    }

    #[test]
    fn consuming_seam_settles_exactly_once() {
        let claim = fixture("h3-attempt-settle-once");
        let (probe, settlements) = probe();
        let root = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-000000000001",
            probe,
        )
        .unwrap();

        let output = root
            .run_once(&current(&claim), |scope| {
                scope.checkpoint()?;
                Ok::<_, mold_inference::InferenceCancelled>(41 + 1)
            })
            .unwrap()
            .unwrap();

        assert_eq!(output, 42);
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_an_unconsumed_root_still_settles_once() {
        let claim = fixture("h3-attempt-drop");
        let (probe, settlements) = probe();
        let root = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-000000000002",
            probe,
        )
        .unwrap();

        drop(root);

        assert_eq!(settlements.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn closure_error_settles_once_without_replaying_the_closure() {
        let claim = fixture("h3-attempt-error");
        let (probe, settlements) = probe();
        let calls = AtomicUsize::new(0);
        let root = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-000000000003",
            probe,
        )
        .unwrap();

        let result = root.run_once(&current(&claim), |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Err::<(), _>("synthetic failure")
        });

        assert_eq!(result.unwrap(), Err("synthetic failure"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn panicking_consumer_unwinds_through_exactly_one_settlement() {
        let claim = fixture("h3-attempt-panic");
        let (probe, settlements) = probe();
        let root = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-00000000000b",
            probe,
        )
        .unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            root.run_once(&current(&claim), |_| -> () {
                panic!("synthetic H3 attempt panic")
            })
        }));

        assert!(result.is_err());
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancelled_scope_never_enters_the_consumer_and_settles_once() {
        let claim = fixture("h3-attempt-cancelled");
        let (probe, settlements) = probe();
        let called = AtomicBool::new(false);
        let cancellation = mold_inference::InferenceCancellationToken::default();
        cancellation.cancel();
        let root = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            cancellation,
            "00000000-0000-4000-8000-000000000004",
            probe,
        )
        .unwrap();

        let error = root
            .run_once(&current(&claim), |_| {
                called.store(true, Ordering::SeqCst);
            })
            .unwrap_err();

        assert_eq!(error, H3AttemptError::Cancelled);
        assert!(!called.load(Ordering::SeqCst));
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn duplicate_active_and_settled_fences_are_rejected() {
        let claim = fixture("h3-attempt-duplicate");
        let first = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-000000000005",
            H3AttemptSettlementProbe::disabled(),
        )
        .unwrap();

        let active = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-000000000006",
            H3AttemptSettlementProbe::disabled(),
        )
        .unwrap_err();
        assert_eq!(active, H3AttemptError::DuplicateActiveAttempt);

        drop(first);
        let settled = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-000000000007",
            H3AttemptSettlementProbe::disabled(),
        )
        .unwrap_err();
        assert_eq!(settled, H3AttemptError::DuplicateSettledFence);
    }

    #[test]
    fn settled_nonce_cannot_be_reused_by_a_different_fence() {
        let first_claim = fixture("h3-attempt-nonce-first");
        let nonce = "00000000-0000-4000-8000-00000000000a";
        let first = H3AttemptRoot::claim_for_test(
            first_claim.clone(),
            &current(&first_claim),
            mold_inference::InferenceCancellationToken::default(),
            nonce,
            H3AttemptSettlementProbe::disabled(),
        )
        .unwrap();
        drop(first);

        let second_claim = fixture("h3-attempt-nonce-second");
        let error = H3AttemptRoot::claim_for_test(
            second_claim.clone(),
            &current(&second_claim),
            mold_inference::InferenceCancellationToken::default(),
            nonce,
            H3AttemptSettlementProbe::disabled(),
        )
        .unwrap_err();

        assert_eq!(error, H3AttemptError::DuplicateNonce);
    }

    #[test]
    fn stale_owner_generation_is_rejected_before_claim() {
        let claim = fixture("h3-attempt-stale");
        let mut stale = current(&claim);
        stale.worker_generation += 1;

        let error = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &stale,
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-000000000008",
            H3AttemptSettlementProbe::disabled(),
        )
        .unwrap_err();

        assert_eq!(error, H3AttemptError::StaleOwnerFence);
    }

    #[test]
    fn mismatched_target_identity_is_rejected_at_the_consuming_seam() {
        let claim = fixture("h3-attempt-identity-mismatch");
        let root = H3AttemptRoot::claim_for_test(
            claim.clone(),
            &current(&claim),
            mold_inference::InferenceCancellationToken::default(),
            "00000000-0000-4000-8000-000000000009",
            H3AttemptSettlementProbe::disabled(),
        )
        .unwrap();
        let attempt = H3GenerationAttempt { root };
        let mut mismatch = current(&claim);
        mismatch.target_budget_identity_sha256 = claim.execution_identity_sha256.clone();
        let called = AtomicBool::new(false);

        let error = attempt
            .run_once(mismatch, |_| called.store(true, Ordering::SeqCst))
            .unwrap_err();

        assert_eq!(error, H3AttemptError::IdentityMismatch);
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn mutated_lease_versions_are_rejected_at_the_consuming_seam() {
        for (suffix, mutate) in [
            (
                "state",
                (|current: &mut H3AttemptCurrent| current.state_version += 1)
                    as fn(&mut H3AttemptCurrent),
            ),
            ("plan", |current: &mut H3AttemptCurrent| {
                current.plan_version += 1
            }),
            ("sample", |current: &mut H3AttemptCurrent| {
                current.memory_sample_generation += 1
            }),
            ("ledger", |current: &mut H3AttemptCurrent| {
                current.memory_ledger_sequence += 1
            }),
        ] {
            let claim = fixture(&format!("h3-attempt-mutated-{suffix}"));
            let root = H3AttemptRoot::claim(
                claim.clone(),
                &current(&claim),
                mold_inference::InferenceCancellationToken::default(),
            )
            .unwrap();
            let attempt = H3GenerationAttempt { root };
            let mut mismatch = current(&claim);
            mutate(&mut mismatch);
            let called = AtomicBool::new(false);

            let error = attempt
                .run_once(mismatch, |_| called.store(true, Ordering::SeqCst))
                .unwrap_err();

            assert_eq!(error, H3AttemptError::StaleOwnerFence);
            assert!(!called.load(Ordering::SeqCst));
        }
    }
}
