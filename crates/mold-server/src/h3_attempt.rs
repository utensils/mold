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
    /// runtime must echo the exact execution, prepared-attempt, and target-
    /// budget identities instead of receiving a self-authenticating handle.
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
}

impl H3AttemptScope<'_> {
    #[cfg(test)]
    pub(crate) fn checkpoint(&self) -> Result<(), mold_inference::InferenceCancelled> {
        self.cancellation.checkpoint()
    }

    pub(crate) fn cancellation_token(&self) -> mold_inference::InferenceCancellationToken {
        self.cancellation.clone()
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

/// Claim a future H3 attempt from exact owner-local facts.
///
/// The current production authority returns `None` because server admission
/// deliberately leaves the prepared-attempt/target-budget/attention triad
/// absent. That keeps this boundary testable without activating H3 or making
/// a legal authorization record sufficient to reach model bytes.
pub(crate) fn claim_generation_attempt(
    worker: &crate::gpu_pool::GpuWorker,
    current_worker_generation: u64,
    fence: &crate::scheduler::LeaseFence,
    job: &crate::gpu_pool::GpuJob,
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
    let Some((prepared_attempt_identity, target_budget_identity)) =
        authority.prepared_target_attempt_identities()
    else {
        return Ok(None);
    };
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
        prepared_attempt_identity_sha256: prepared_attempt_identity.to_string(),
        target_budget_identity_sha256: target_budget_identity.to_string(),
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
        prepared_attempt_identity_sha256: prepared_attempt_identity.to_string(),
        target_budget_identity_sha256: target_budget_identity.to_string(),
    };
    if authority.device_id() != worker_device_id || authority.device_ordinal() != worker.gpu.ordinal
    {
        return Err(H3AttemptError::StaleOwnerFence);
    }
    let cancellation = job.batch_child.as_ref().map_or_else(
        mold_inference::InferenceCancellationToken::default,
        |child| child.cancellation.clone(),
    );
    let root = H3AttemptRoot::claim(invocation, &current, cancellation)?;
    Ok(Some(H3GenerationAttempt { root }))
}

/// Rebuild the consuming-seam projection without consulting the attempt root.
///
/// The lease fields come from the exact fence installed on the job after
/// acceptance. Owner and device facts come from the live worker, while the
/// three execution identities come from the still-frozen plan authority. A
/// future H3 backend must replace those authority-side identity values with
/// its own echo before runtime activation; production cannot reach this helper
/// while the target authority triad remains absent.
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
    let (prepared_attempt_identity, target_budget_identity) = authority
        .prepared_target_attempt_identities()
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
        prepared_attempt_identity_sha256: prepared_attempt_identity.to_string(),
        target_budget_identity_sha256: target_budget_identity.to_string(),
    })
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
