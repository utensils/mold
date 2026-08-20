//! Singular-owner private MiniMax H3 FL2VA runtime composition.
//!
//! This module is available only to the non-shipping `h3-private-uat`
//! feature. It does not register a model, loader, capability, catalog,
//! download, server, CLI, or product-surface route.
//!
//! The concrete core is attempt-scoped and one-shot. The engine constructs and
//! consumes a fresh session per job; this private composer remains additionally
//! sealed until every artifact, memory, and execution authority is available.

use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::{Device, DeviceLocation, Tensor};
use mold_candle::minimax_h3::{
    H3AttentionDevice, H3AttentionRuntimeAuthority, H3AuthenticatedQwenNvfp4Authority,
    H3ComfyOpenedInt8Checkpoint, H3ForwardInput, H3FrozenPackedLayout, H3TransformerOutput,
    H3TransformerTask, StereoLatents, StereoWaveform,
};
use mold_core::minimax_h3::{self as contract, Task};
use sha2::{Digest, Sha256};

use super::backend::{H3BackendArtifactLease, H3BackendExecutionLease, H3CandleBackendDevice};
use super::engine::{H3BlockStreamedDenoiser, H3StreamedDenoiser};
use super::offload::H3BlockLease;
#[cfg(feature = "mp4")]
use super::pipeline::H3PipelineObserver;
use super::pipeline::{
    H3Fl2VaBackend, H3PipelineBackendIdentity, H3PipelineBackendKind, H3PipelineCheckpoint,
    H3PipelineEvent, H3PipelinePhase, H3PreparedEndpoint, H3TextConditioning, H3VideoEncodeSink,
};
use super::private_opened_evidence::{
    H3PrivateComfyStorageAuthority, H3PrivatePreparedFl2VaFactoryInputs,
    H3PrivatePreparedFl2VaRetention,
};
use super::private_qwen::{
    H3PrivateQwenAdapter, H3PrivateQwenArtifactLease, H3PrivateQwenConditionerLease,
};
use super::private_qwen_support::H3PrivateQwenSupport;
use super::private_runtime::{
    bind_private_comfy_stream, load_and_pair_private_comfy_stream, H3PrivateBoundComfyStream,
    H3PrivateComfyBindingExpectation, H3PrivateComfyBlockLoader, H3PrivateComfyCancellationGuard,
    H3PrivateComfyCancellationSlot, H3PrivateComfyCheckpointFacts, H3PrivateComfyStreamAuthority,
    H3PrivateComfyTransformerExecutor,
};
use super::private_server::{
    H3PrivateAllocationCommit, H3PrivateFactoryActivationEvidence, H3PrivateSchedulerLedgerIdentity,
};
use super::private_vae_adapter::H3PrivateVaeRuntime;
use super::sampler::H3SamplerKind;
#[cfg(test)]
use super::vae_runtime::H3ComfyVaeLoadPhase;
use super::vae_runtime::{
    load_h3_comfy_vae_runtime_from_authority, H3AuthenticatedComfyVaeAuthority,
    H3ComfyVaeLoadEvent, H3ComfyVaeLoadObserver, H3ComfyVaeRuntimeBundle,
};
use crate::h3_factory::{
    H3FactoryTargetLoadDropPolicy, H3PrivateFl2VaFactoryAuthority, H3PrivateVaeFactoryAuthority,
};
use crate::progress::ProgressReporter;
use crate::{FrozenH3FactoryAuthority, H3FactoryQuantizationAuthority};

/// Full opened-artifact authority needed by the streamed FL2VA attempt.
///
/// # Safety
///
/// In addition to the Qwen contract, implementers must retain the opened
/// transformer and both VAE artifacts named by these exact identities until
/// the complete attempt drops. The values must come from authenticated opened
/// descriptors, not copied model metadata.
pub(crate) unsafe trait H3PrivateFl2VaArtifactLease:
    H3PrivateQwenArtifactLease
{
    fn backend_plan_identity_sha256(&self) -> &str;
    fn transformer_component_content_sha256(&self) -> &str;
    fn transformer_component_validation_sha256(&self) -> &str;
    fn visual_vae_component_content_sha256(&self) -> &str;
    fn visual_vae_component_validation_sha256(&self) -> &str;
    fn audio_vae_component_content_sha256(&self) -> &str;
    fn audio_vae_component_validation_sha256(&self) -> &str;
    fn vae_artifact_plan_identity_sha256(&self) -> &str;
    fn transformer_task(&self) -> H3TransformerTask;
    fn transformer_checkpoint_content_sha256(&self) -> &str;
    fn transformer_checkpoint_layout_identity_sha256(&self) -> &str;
    fn transformer_checkpoint_identity_sha256(&self) -> &str;
    fn transformer_policy_identity_sha256(&self) -> &str;
    fn pruned_adaln_table_identity_sha256(&self) -> &str;
    fn attention_runtime_identity_sha256(&self) -> &str;
    fn attention_kernel_identity(&self) -> &str;
    fn attention_qualification_sha256(&self) -> &str;
    fn memory_overlap_identity_sha256(&self) -> &str;
}

/// Request-shaped conservative overlap record issued by private admission.
/// These are retained-lifetime facts, not estimates: composition rejects the
/// attempt unless the opened artifact lease names this exact fingerprint.
///
/// Production issuance is possible only from an authenticated prepared target
/// budget plus the exact scheduler-ledger identity that owns it. Callers never
/// provide the overlap byte fields independently.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateFl2VaMemoryOverlapAuthority {
    factory_identity_sha256: String,
    prepared_attempt_identity_sha256: String,
    target_budget_identity_sha256: String,
    condition_visual_rows: u64,
    condition_backing_host_bytes: u64,
    condition_backing_device_bytes: u64,
    target_audio_latent_device_bytes: u64,
    visual_vae_resident_device_bytes: u64,
    audio_vae_resident_device_bytes: u64,
    attempt_resident_vae_device_bytes: u64,
    visual_decode_peak_device_bytes: u64,
    normalized_endpoint_host_bytes: u64,
    scheduler_ledger_identity_sha256: String,
    identity_sha256: String,
    _activation: admitted_overlap_seal::Token,
}

mod admitted_overlap_seal {
    #[derive(Debug, Eq, PartialEq)]
    pub struct Token;
}

impl H3PrivateFl2VaMemoryOverlapAuthority {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        factory_identity_sha256: impl Into<String>,
        prepared_attempt_identity_sha256: impl Into<String>,
        target_budget_identity_sha256: impl Into<String>,
        condition_visual_rows: u64,
        condition_backing_host_bytes: u64,
        condition_backing_device_bytes: u64,
        target_audio_latent_device_bytes: u64,
        visual_vae_resident_device_bytes: u64,
        audio_vae_resident_device_bytes: u64,
        attempt_resident_vae_device_bytes: u64,
        visual_decode_peak_device_bytes: u64,
        normalized_endpoint_host_bytes: u64,
    ) -> Result<Self> {
        let mut authority = Self {
            factory_identity_sha256: factory_identity_sha256.into(),
            prepared_attempt_identity_sha256: prepared_attempt_identity_sha256.into(),
            target_budget_identity_sha256: target_budget_identity_sha256.into(),
            condition_visual_rows,
            condition_backing_host_bytes,
            condition_backing_device_bytes,
            target_audio_latent_device_bytes,
            visual_vae_resident_device_bytes,
            audio_vae_resident_device_bytes,
            attempt_resident_vae_device_bytes,
            visual_decode_peak_device_bytes,
            normalized_endpoint_host_bytes,
            scheduler_ledger_identity_sha256: std::iter::repeat_n('e', 64).collect(),
            identity_sha256: String::new(),
            _activation: admitted_overlap_seal::Token,
        };
        authority.identity_sha256 = memory_overlap_identity(&authority);
        authority.validate()?;
        Ok(authority)
    }

    pub(crate) fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    fn validate(&self) -> Result<()> {
        if !valid_sha256(&self.factory_identity_sha256)
            || !valid_sha256(&self.prepared_attempt_identity_sha256)
            || !valid_sha256(&self.target_budget_identity_sha256)
            || !valid_sha256(&self.scheduler_ledger_identity_sha256)
            || self.target_audio_latent_device_bytes == 0
            || self.visual_vae_resident_device_bytes == 0
            || self.audio_vae_resident_device_bytes == 0
        {
            bail!("private MiniMax H3 overlap authority lacks exact retained memory facts");
        }
        let condition_bytes = (
            self.condition_backing_host_bytes,
            self.condition_backing_device_bytes,
            self.normalized_endpoint_host_bytes,
        );
        if self.condition_visual_rows == 0 {
            if condition_bytes != (0, 0, 0) {
                bail!("private MiniMax H3 T2VA overlap authority invents condition backing");
            }
        } else if condition_bytes.0 == 0 || condition_bytes.1 == 0 || condition_bytes.2 == 0 {
            bail!("private MiniMax H3 FL2VA overlap authority undercharges condition backing");
        }
        let both_vaes = self
            .visual_vae_resident_device_bytes
            .checked_add(self.audio_vae_resident_device_bytes)
            .ok_or_else(|| anyhow::anyhow!("private MiniMax H3 VAE overlap bytes overflow"))?;
        if self.attempt_resident_vae_device_bytes != both_vaes {
            bail!("private MiniMax H3 overlap authority does not retain both VAEs");
        }
        let visual_decode_floor = both_vaes
            .checked_add(self.target_audio_latent_device_bytes)
            .ok_or_else(|| anyhow::anyhow!("private MiniMax H3 visual decode bytes overflow"))?;
        if self.visual_decode_peak_device_bytes < visual_decode_floor {
            bail!("private MiniMax H3 visual decode undercharges retained audio or VAE state");
        }
        if self.identity_sha256 != memory_overlap_identity(self) {
            bail!("private MiniMax H3 overlap authority changed after admission");
        }
        Ok(())
    }
}

const H3_PRIVATE_FL2VA_OVERLAP_IDENTITY_DOMAIN: &[u8] =
    b"mold.minimax-h3.private-fl2va-overlap.v3\0";

fn memory_overlap_identity(authority: &H3PrivateFl2VaMemoryOverlapAuthority) -> String {
    let mut hash = Sha256::new();
    hash.update(H3_PRIVATE_FL2VA_OVERLAP_IDENTITY_DOMAIN);
    hash.update(authority.factory_identity_sha256.as_bytes());
    hash.update(authority.prepared_attempt_identity_sha256.as_bytes());
    hash.update(authority.target_budget_identity_sha256.as_bytes());
    hash.update(authority.condition_visual_rows.to_le_bytes());
    for bytes in [
        authority.condition_backing_host_bytes,
        authority.condition_backing_device_bytes,
        authority.target_audio_latent_device_bytes,
        authority.visual_vae_resident_device_bytes,
        authority.audio_vae_resident_device_bytes,
        authority.attempt_resident_vae_device_bytes,
        authority.visual_decode_peak_device_bytes,
        authority.normalized_endpoint_host_bytes,
    ] {
        hash.update(bytes.to_le_bytes());
    }
    hash.update(authority.scheduler_ledger_identity_sha256.as_bytes());
    format!("{:x}", hash.finalize())
}

/// Issue the retained overlap record from one authenticated prepared budget
/// and the scheduler-ledger identity that owns exactly that budget. Fine-
/// grained memory fields are never accepted from the server.
pub(crate) fn issue_private_fl2va_memory_overlap(
    authority: &FrozenH3FactoryAuthority,
    prepared: &H3PrivatePreparedFl2VaFactoryInputs,
    ledger: &H3PrivateSchedulerLedgerIdentity,
) -> Result<H3PrivateFl2VaMemoryOverlapAuthority> {
    prepared.revalidate()?;
    ledger.revalidate()?;
    let identities = authority
        .prepared_target_attempt_identities()
        .ok_or_else(|| anyhow::anyhow!("private H3 factory has no prepared target identities"))?;
    if authority.execution_fingerprint() != ledger.execution_fingerprint()
        || identities.0 != prepared.prepared_attempt_identity_sha256()
        || identities.1 != prepared.target_budget_identity_sha256()
        || identities.0 != ledger.prepared_attempt_identity_sha256()
        || identities.1 != ledger.target_budget_identity_sha256()
        || authority.component_set_identity_sha256() != ledger.component_set_identity_sha256()
    {
        bail!("private H3 prepared budget differs from its scheduler ledger")
    }
    let request = &prepared.factory_attempt.request;
    let budget = &prepared.factory_attempt.target_budget;
    let mut overlap = H3PrivateFl2VaMemoryOverlapAuthority {
        factory_identity_sha256: authority.identity_sha256().into(),
        prepared_attempt_identity_sha256: identities.0.into(),
        target_budget_identity_sha256: identities.1.into(),
        condition_visual_rows: request.rows.condition_visual_rows,
        condition_backing_host_bytes: budget.condition_backing_host_bytes,
        condition_backing_device_bytes: budget.condition_latent_backing_device_bytes,
        target_audio_latent_device_bytes: budget.target_audio_latent_device_bytes,
        visual_vae_resident_device_bytes: budget.visual_vae_resident_device_bytes,
        audio_vae_resident_device_bytes: budget.audio_vae_resident_device_bytes,
        attempt_resident_vae_device_bytes: budget.attempt_resident_vae_device_bytes,
        visual_decode_peak_device_bytes: budget.visual_decode_phase_device_bytes,
        normalized_endpoint_host_bytes: budget.normalized_endpoint_host_bytes,
        scheduler_ledger_identity_sha256: ledger.identity_sha256().into(),
        identity_sha256: String::new(),
        _activation: admitted_overlap_seal::Token,
    };
    overlap.identity_sha256 = memory_overlap_identity(&overlap);
    overlap.validate()?;
    Ok(overlap)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateInnerAuthority {
    pub(crate) factory_identity_sha256: String,
    pub(crate) backend_plan_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
}

impl H3PrivateInnerAuthority {
    pub(crate) fn new(
        factory_identity_sha256: impl Into<String>,
        backend_plan_identity_sha256: impl Into<String>,
        component_set_identity_sha256: impl Into<String>,
    ) -> Result<Self> {
        let authority = Self {
            factory_identity_sha256: factory_identity_sha256.into(),
            backend_plan_identity_sha256: backend_plan_identity_sha256.into(),
            component_set_identity_sha256: component_set_identity_sha256.into(),
        };
        if !valid_sha256(&authority.factory_identity_sha256)
            || !valid_sha256(&authority.backend_plan_identity_sha256)
            || !valid_sha256(&authority.component_set_identity_sha256)
        {
            bail!("private MiniMax H3 VAE-free inner authority is not exact SHA-256 evidence");
        }
        Ok(authority)
    }

    pub(crate) fn matches(&self, admitted: &H3PrivateVaeFactoryAuthority) -> bool {
        self.factory_identity_sha256 == admitted.factory_identity_sha256
            && self.backend_plan_identity_sha256 == admitted.backend_plan_identity_sha256
            && self.component_set_identity_sha256 == admitted.component_set_identity_sha256
    }
}

#[cfg(not(test))]
mod vae_free_inner_seal {
    pub trait Sealed {}
}

#[cfg(test)]
pub(crate) mod vae_free_inner_seal {
    pub trait Sealed {}
}

/// Active-authority hook implemented only by the concrete private VAE-free
/// streamed core defined in this module (plus bounded unit-test fixtures).
///
/// # Safety
///
/// An implementer must own no visual or audio VAE weights. Every call must
/// revalidate its active execution lease and the artifact lease covering its
/// complete admitted component set, then return the exact full-factory,
/// backend-plan, and component-set identities from that live admission.
#[allow(private_bounds)]
pub(crate) unsafe trait H3PrivateVaeFreeInnerAuthority:
    H3Fl2VaBackend + vae_free_inner_seal::Sealed
{
    fn validate_private_vae_free_inner_authority(&self) -> Result<H3PrivateInnerAuthority>;
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// The only owner of the scheduler execution grant and the complete artifact
/// lease for one private attempt. All component-specific handles below are
/// projections over this allocation; none can clone either underlying lease.
/// Device selection is frozen from the validated route before construction so
/// safe lease implementations cannot redirect later component operations.
struct H3PrivateAttemptOwner<E, A> {
    execution: E,
    artifacts: A,
    device: Device,
}

/// Singular attempt authority retained until every model, streamed block, and
/// artifact-backed tensor has dropped.
pub(crate) struct H3PrivateAttemptAuthority<E, A> {
    owner: Arc<H3PrivateAttemptOwner<E, A>>,
}

impl<E, A> H3PrivateAttemptAuthority<E, A> {
    /// Construction stays inside the private runtime composition boundary so
    /// callers can never build projections from independently owned leases.
    fn new(bound: H3PrivateBoundExecution<E>, artifacts: A) -> Self {
        let H3PrivateBoundExecution { execution, device } = bound;
        Self {
            owner: Arc::new(H3PrivateAttemptOwner {
                execution,
                artifacts,
                device,
            }),
        }
    }

    fn qwen_projections(
        &self,
    ) -> (
        H3PrivateExecutionProjection<E, A>,
        H3PrivateArtifactProjection<E, A>,
    ) {
        (
            H3PrivateExecutionProjection {
                owner: Arc::clone(&self.owner),
            },
            H3PrivateArtifactProjection {
                owner: Arc::clone(&self.owner),
            },
        )
    }

    fn block_projection(&self) -> H3PrivateExecutionProjection<E, A> {
        H3PrivateExecutionProjection {
            owner: Arc::clone(&self.owner),
        }
    }

    fn artifact_projection(&self) -> H3PrivateArtifactProjection<E, A> {
        H3PrivateArtifactProjection {
            owner: Arc::clone(&self.owner),
        }
    }

    #[cfg(test)]
    fn new_unchecked_for_test(execution: E, artifacts: A, device: Device) -> Self {
        Self::new(H3PrivateBoundExecution { execution, device }, artifacts)
    }
}

/// Lightweight execution view. Manual `Clone` is intentional: deriving it
/// would incorrectly require the underlying execution and artifact types to
/// implement `Clone` even though only the `Arc` is duplicated.
pub(crate) struct H3PrivateExecutionProjection<E, A> {
    owner: Arc<H3PrivateAttemptOwner<E, A>>,
}

impl<E, A> Clone for H3PrivateExecutionProjection<E, A> {
    fn clone(&self) -> Self {
        Self {
            owner: Arc::clone(&self.owner),
        }
    }
}

impl<E, A> H3PrivateExecutionProjection<E, A> {
    fn belongs_to(&self, authority: &H3PrivateAttemptAuthority<E, A>) -> bool {
        Arc::ptr_eq(&self.owner, &authority.owner)
    }
}

impl<E, A> H3BackendExecutionLease for H3PrivateExecutionProjection<E, A>
where
    E: H3BackendExecutionLease,
{
    fn lease_id(&self) -> &str {
        self.owner.execution.lease_id()
    }

    fn device_id(&self) -> &str {
        self.owner.execution.device_id()
    }

    fn backend(&self) -> H3CandleBackendDevice {
        self.owner.execution.backend()
    }

    fn execution_fingerprint(&self) -> &str {
        self.owner.execution.execution_fingerprint()
    }

    fn device(&self) -> &Device {
        &self.owner.device
    }

    fn is_active(&self) -> bool {
        self.owner.execution.is_active()
    }
}

impl<E, A> H3BlockLease for H3PrivateExecutionProjection<E, A>
where
    E: H3BackendExecutionLease,
{
    fn lease_id(&self) -> &str {
        self.owner.execution.lease_id()
    }

    fn device_id(&self) -> &str {
        self.owner.execution.device_id()
    }

    fn execution_fingerprint(&self) -> &str {
        self.owner.execution.execution_fingerprint()
    }

    fn is_active(&self) -> bool {
        self.owner.execution.is_active()
    }
}

/// Lightweight artifact view over the same singular attempt allocation.
pub(crate) struct H3PrivateArtifactProjection<E, A> {
    owner: Arc<H3PrivateAttemptOwner<E, A>>,
}

impl<E, A> Clone for H3PrivateArtifactProjection<E, A> {
    fn clone(&self) -> Self {
        Self {
            owner: Arc::clone(&self.owner),
        }
    }
}

impl<E, A> H3PrivateArtifactProjection<E, A> {
    fn belongs_to(&self, authority: &H3PrivateAttemptAuthority<E, A>) -> bool {
        Arc::ptr_eq(&self.owner, &authority.owner)
    }
}

// SAFETY: the projection retains the exact `Arc` that owns `A`; therefore the
// underlying artifact lease outlives every projected reference and cannot be
// replaced independently.
unsafe impl<E, A> H3BackendArtifactLease for H3PrivateArtifactProjection<E, A>
where
    A: H3BackendArtifactLease,
{
    fn component_set_identity(&self) -> &str {
        self.owner.artifacts.component_set_identity()
    }

    fn is_active(&self) -> bool {
        self.owner.artifacts.is_active()
    }
}

// SAFETY: every method delegates to the one immutable artifact lease retained
// by the same singular attempt allocation.
unsafe impl<E, A> H3PrivateQwenArtifactLease for H3PrivateArtifactProjection<E, A>
where
    A: H3PrivateQwenArtifactLease,
{
    fn factory_identity_sha256(&self) -> &str {
        self.owner.artifacts.factory_identity_sha256()
    }

    fn conditioner_component_content_sha256(&self) -> &str {
        self.owner.artifacts.conditioner_component_content_sha256()
    }

    fn conditioner_component_validation_sha256(&self) -> &str {
        self.owner
            .artifacts
            .conditioner_component_validation_sha256()
    }

    fn support_identity_sha256(&self) -> &str {
        self.owner.artifacts.support_identity_sha256()
    }

    fn weight_identity_sha256(&self) -> &str {
        self.owner.artifacts.weight_identity_sha256()
    }

    fn weight_header_identity_sha256(&self) -> &str {
        self.owner.artifacts.weight_header_identity_sha256()
    }

    fn weight_policy_identity_sha256(&self) -> &str {
        self.owner.artifacts.weight_policy_identity_sha256()
    }
}

// SAFETY: the projection retains the same singular `Arc` for every full
// artifact identity and cannot outlive or independently replace `A`.
unsafe impl<E, A> H3PrivateFl2VaArtifactLease for H3PrivateArtifactProjection<E, A>
where
    A: H3PrivateFl2VaArtifactLease,
{
    fn backend_plan_identity_sha256(&self) -> &str {
        self.owner.artifacts.backend_plan_identity_sha256()
    }

    fn transformer_component_content_sha256(&self) -> &str {
        self.owner.artifacts.transformer_component_content_sha256()
    }

    fn transformer_component_validation_sha256(&self) -> &str {
        self.owner
            .artifacts
            .transformer_component_validation_sha256()
    }

    fn visual_vae_component_content_sha256(&self) -> &str {
        self.owner.artifacts.visual_vae_component_content_sha256()
    }

    fn visual_vae_component_validation_sha256(&self) -> &str {
        self.owner
            .artifacts
            .visual_vae_component_validation_sha256()
    }

    fn audio_vae_component_content_sha256(&self) -> &str {
        self.owner.artifacts.audio_vae_component_content_sha256()
    }

    fn audio_vae_component_validation_sha256(&self) -> &str {
        self.owner.artifacts.audio_vae_component_validation_sha256()
    }

    fn vae_artifact_plan_identity_sha256(&self) -> &str {
        self.owner.artifacts.vae_artifact_plan_identity_sha256()
    }

    fn transformer_task(&self) -> H3TransformerTask {
        self.owner.artifacts.transformer_task()
    }

    fn transformer_checkpoint_content_sha256(&self) -> &str {
        self.owner.artifacts.transformer_checkpoint_content_sha256()
    }

    fn transformer_checkpoint_layout_identity_sha256(&self) -> &str {
        self.owner
            .artifacts
            .transformer_checkpoint_layout_identity_sha256()
    }

    fn transformer_checkpoint_identity_sha256(&self) -> &str {
        self.owner
            .artifacts
            .transformer_checkpoint_identity_sha256()
    }

    fn transformer_policy_identity_sha256(&self) -> &str {
        self.owner.artifacts.transformer_policy_identity_sha256()
    }

    fn pruned_adaln_table_identity_sha256(&self) -> &str {
        self.owner.artifacts.pruned_adaln_table_identity_sha256()
    }

    fn attention_runtime_identity_sha256(&self) -> &str {
        self.owner.artifacts.attention_runtime_identity_sha256()
    }

    fn attention_kernel_identity(&self) -> &str {
        self.owner.artifacts.attention_kernel_identity()
    }

    fn attention_qualification_sha256(&self) -> &str {
        self.owner.artifacts.attention_qualification_sha256()
    }

    fn memory_overlap_identity_sha256(&self) -> &str {
        self.owner.artifacts.memory_overlap_identity_sha256()
    }
}

trait H3PrivateFl2VaConditioner<E, A>: Send + Sync
where
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    fn model(&self) -> &str;
    fn task(&self) -> Task;
    fn execution_projection(&self) -> &H3PrivateExecutionProjection<E, A>;
    fn artifact_projection(&self) -> &H3PrivateArtifactProjection<E, A>;
    fn encode_fl2va(
        &mut self,
        prompt: &str,
        endpoints: &[H3PreparedEndpoint],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning>;
    fn validate_continuing_authority(&self) -> Result<()>;
}

impl<'authority, C, E, A> H3PrivateFl2VaConditioner<E, A>
    for H3PrivateQwenAdapter<
        'authority,
        C,
        H3PrivateExecutionProjection<E, A>,
        H3PrivateArtifactProjection<E, A>,
    >
where
    C: H3PrivateQwenConditionerLease + Send + Sync,
    E: H3BackendExecutionLease + Send + Sync,
    A: H3PrivateFl2VaArtifactLease + Send + Sync,
{
    fn model(&self) -> &str {
        self.model()
    }

    fn task(&self) -> Task {
        self.task()
    }

    fn execution_projection(&self) -> &H3PrivateExecutionProjection<E, A> {
        self.execution_lease()
    }

    fn artifact_projection(&self) -> &H3PrivateArtifactProjection<E, A> {
        self.artifact_lease()
    }

    fn encode_fl2va(
        &mut self,
        prompt: &str,
        endpoints: &[H3PreparedEndpoint],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        self.encode_fl2va(prompt, endpoints, checkpoint)
    }

    fn validate_continuing_authority(&self) -> Result<()> {
        self.validate_continuing_authorities()
    }
}

/// Concrete VAE-free private component backend. The Qwen model is one-shot,
/// the main DiT is block-streamed, and every VAE method fails closed so the
/// outer authenticated VAE adapter is the only codec implementation.
struct H3PrivateVaeFreeStreamedCore<Q, D, E, A>
where
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    conditioner: Q,
    denoiser: D,
    cancellation_guard: H3PrivateComfyCancellationGuard,
    execution_authority: H3PrivateExecutionProjection<E, A>,
    artifact_authority: H3PrivateArtifactProjection<E, A>,
    frozen_identity: H3PipelineBackendIdentity,
    admitted: H3PrivateFl2VaFactoryAuthority,
    stream_authority: H3PrivateComfyStreamAuthority,
    memory_overlap: H3PrivateFl2VaMemoryOverlapAuthority,
    inner_authority: H3PrivateInnerAuthority,
    // Field order is load-bearing: this final anchor releases the one
    // execution grant and one full artifact lease only after every projection.
    attempt: H3PrivateAttemptAuthority<E, A>,
}

impl<Q, D, E, A> H3PrivateVaeFreeStreamedCore<Q, D, E, A>
where
    Q: H3PrivateFl2VaConditioner<E, A>,
    D: H3StreamedDenoiser,
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        conditioner: Q,
        denoiser: D,
        cancellation_guard: H3PrivateComfyCancellationGuard,
        execution_authority: H3PrivateExecutionProjection<E, A>,
        artifact_authority: H3PrivateArtifactProjection<E, A>,
        admitted: H3PrivateFl2VaFactoryAuthority,
        stream_authority: H3PrivateComfyStreamAuthority,
        memory_overlap: H3PrivateFl2VaMemoryOverlapAuthority,
        attempt: H3PrivateAttemptAuthority<E, A>,
    ) -> Result<Self> {
        let kind = if execution_authority.device().is_cuda() {
            H3PipelineBackendKind::Cuda
        } else if cfg!(test) && execution_authority.device().is_cpu() {
            H3PipelineBackendKind::SyntheticCpu
        } else {
            bail!("private MiniMax H3 streamed core requires one CUDA execution device");
        };
        let frozen_identity = H3PipelineBackendIdentity {
            kind,
            device_id: admitted.device_id.clone(),
            execution_fingerprint: admitted.execution_fingerprint.clone(),
        };
        let inner_authority = H3PrivateInnerAuthority::new(
            admitted.factory_identity_sha256.clone(),
            admitted.backend_plan_identity_sha256.clone(),
            admitted.component_set_identity_sha256.clone(),
        )?;
        let core = Self {
            conditioner,
            denoiser,
            cancellation_guard,
            execution_authority,
            artifact_authority,
            frozen_identity,
            admitted,
            stream_authority,
            memory_overlap,
            inner_authority,
            attempt,
        };
        core.validate_authority()?;
        Ok(core)
    }

    fn validate_authority(&self) -> Result<()> {
        let execution = &self.execution_authority;
        let artifacts = &self.artifact_authority;
        if !execution.belongs_to(&self.attempt)
            || !artifacts.belongs_to(&self.attempt)
            || !self
                .conditioner
                .execution_projection()
                .belongs_to(&self.attempt)
            || !self
                .conditioner
                .artifact_projection()
                .belongs_to(&self.attempt)
        {
            bail!("private MiniMax H3 component projections came from different attempts");
        }
        self.admitted.block_streaming.validate()?;
        self.memory_overlap.validate()?;
        if self.memory_overlap.factory_identity_sha256 != self.admitted.factory_identity_sha256
            || self.memory_overlap.condition_visual_rows != self.admitted.condition_visual_rows
            || self.memory_overlap.identity_sha256() != artifacts.memory_overlap_identity_sha256()
            || !H3BackendExecutionLease::is_active(execution)
            || H3BackendExecutionLease::lease_id(execution)
                .trim()
                .is_empty()
            || H3BackendExecutionLease::device_id(execution) != self.admitted.device_id
            || H3BackendExecutionLease::execution_fingerprint(execution)
                != self.admitted.execution_fingerprint
            || execution.backend()
                != H3CandleBackendDevice::from_compute_capability(self.admitted.compute_capability)
            || !(execution.backend().matches_candle(execution.device())
                || cfg!(test) && execution.device().is_cpu())
            || !artifacts.is_active()
            || artifacts.factory_identity_sha256() != self.admitted.factory_identity_sha256
            || artifacts.backend_plan_identity_sha256()
                != self.admitted.backend_plan_identity_sha256
            || artifacts.component_set_identity() != self.admitted.component_set_identity_sha256
            || artifacts.conditioner_component_content_sha256()
                != self.admitted.conditioner_component_content_sha256
            || artifacts.conditioner_component_validation_sha256()
                != self.admitted.conditioner_component_validation_sha256
            || artifacts.transformer_component_content_sha256()
                != self.admitted.transformer_component_content_sha256
            || artifacts.transformer_component_validation_sha256()
                != self.admitted.transformer_component_validation_sha256
            || artifacts.visual_vae_component_content_sha256()
                != self.admitted.visual_vae_component_content_sha256
            || artifacts.visual_vae_component_validation_sha256()
                != self.admitted.visual_vae_component_validation_sha256
            || artifacts.audio_vae_component_content_sha256()
                != self.admitted.audio_vae_component_content_sha256
            || artifacts.audio_vae_component_validation_sha256()
                != self.admitted.audio_vae_component_validation_sha256
            || artifacts.vae_artifact_plan_identity_sha256()
                != self.admitted.vae_artifact_plan_identity_sha256
            || artifacts.transformer_task() != self.stream_authority.task
            || artifacts.transformer_checkpoint_content_sha256()
                != self.stream_authority.transformer_content_sha256
            || artifacts.transformer_checkpoint_layout_identity_sha256()
                != self.stream_authority.transformer_layout_identity_sha256
            || artifacts.transformer_checkpoint_identity_sha256()
                != self.stream_authority.checkpoint_identity_sha256
            || artifacts.transformer_policy_identity_sha256()
                != self.stream_authority.transformer_policy_identity_sha256
            || artifacts.attention_runtime_identity_sha256()
                != self.stream_authority.attention_runtime_identity_sha256
            || artifacts.attention_runtime_identity_sha256()
                != self.admitted.attention.runtime_identity_sha256
            || artifacts.attention_kernel_identity()
                != self.admitted.attention.qualification_kernel_identity
            || artifacts.attention_qualification_sha256()
                != self.admitted.attention.qualification_sha256
            || self.stream_authority.task != H3TransformerTask::T2VaFl2Va
            || self.conditioner.model() != self.admitted.canonical_model
            || self.conditioner.task() != self.admitted.task
            || self.admitted.task != Task::Fl2va
            || self.admitted.canonical_model != contract::FL2VA_COMFY
            || self.denoiser.identity() != self.frozen_identity
        {
            bail!("private MiniMax H3 streamed runtime differs from frozen authority");
        }
        match &self.admitted.quantization {
            H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256,
                pruned_adaln_table_sha256,
                ..
            } if transformer_policy_sha256 == artifacts.transformer_policy_identity_sha256()
                && transformer_policy_sha256
                    == &self.stream_authority.transformer_policy_identity_sha256
                && pruned_adaln_table_sha256 == artifacts.pruned_adaln_table_identity_sha256() => {}
            _ => bail!("private MiniMax H3 transformer quantization authority differs"),
        }
        let _ = &self.cancellation_guard;
        Ok(())
    }
}

impl<Q, D, E, A> H3Fl2VaBackend for H3PrivateVaeFreeStreamedCore<Q, D, E, A>
where
    Q: H3PrivateFl2VaConditioner<E, A>,
    D: H3StreamedDenoiser,
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    fn identity(&self) -> H3PipelineBackendIdentity {
        self.frozen_identity.clone()
    }

    fn device(&self) -> &Device {
        self.execution_authority.device()
    }

    /// The integrator is a property of the frozen quantization authority,
    /// not of the layout: a reviewed Turbo tier distils to `comfy-euler`
    /// while the unadapted compact checkpoint keeps `comfy-res-multistep`.
    fn sampler_kind(&self) -> H3SamplerKind {
        self.admitted.quantization.sampler_kind()
    }

    fn sampler_video_shift(&self) -> f32 {
        self.admitted.quantization.video_shift()
    }

    fn encode_text(
        &mut self,
        prompt: &str,
        endpoints: &[H3PreparedEndpoint],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        self.validate_authority()?;
        let conditioning = self
            .conditioner
            .encode_fl2va(prompt, endpoints, checkpoint)?;
        // The one-shot adapter has dropped its model and released the
        // conditioner reservation before returning this tensor.
        self.conditioner.validate_continuing_authority()?;
        self.validate_authority()?;
        Ok(conditioning)
    }

    fn encode_visual_condition(
        &mut self,
        _endpoint: &H3PreparedEndpoint,
        _mode: mold_candle::minimax_h3::ConditionEncodeMode,
        _checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor> {
        bail!("private MiniMax H3 VAE-free core cannot encode visual conditions")
    }

    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput> {
        self.validate_authority()?;
        self.conditioner.validate_continuing_authority()?;
        let output = self.denoiser.denoise(input, layout, checkpoint)?;
        self.validate_authority()?;
        Ok(output)
    }

    fn decode_video(
        &mut self,
        _latents: &Tensor,
        _sink: &mut H3VideoEncodeSink,
        _checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        bail!("private MiniMax H3 VAE-free core cannot decode video")
    }

    fn decode_audio(
        &mut self,
        _latents: &StereoLatents,
        _checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform> {
        bail!("private MiniMax H3 VAE-free core cannot decode audio")
    }
}

impl<Q, D, E, A> vae_free_inner_seal::Sealed for H3PrivateVaeFreeStreamedCore<Q, D, E, A>
where
    Q: H3PrivateFl2VaConditioner<E, A>,
    D: H3StreamedDenoiser,
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
}

// SAFETY: the concrete core owns no VAE runtime and revalidates the exact
// singular execution/artifact attempt before and after every delegated phase.
unsafe impl<Q, D, E, A> H3PrivateVaeFreeInnerAuthority for H3PrivateVaeFreeStreamedCore<Q, D, E, A>
where
    Q: H3PrivateFl2VaConditioner<E, A>,
    D: H3StreamedDenoiser,
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    fn validate_private_vae_free_inner_authority(&self) -> Result<H3PrivateInnerAuthority> {
        self.validate_authority()?;
        Ok(self.inner_authority.clone())
    }
}

type H3PrivateComfyDenoiser<E, A> = H3BlockStreamedDenoiser<
    H3PrivateComfyBlockLoader,
    H3PrivateExecutionProjection<E, A>,
    H3PrivateComfyTransformerExecutor,
>;

type H3PrivateComfyCore<'authority, C, E, A> = H3PrivateVaeFreeStreamedCore<
    H3PrivateQwenAdapter<
        'authority,
        C,
        H3PrivateExecutionProjection<E, A>,
        H3PrivateArtifactProjection<E, A>,
    >,
    H3PrivateComfyDenoiser<E, A>,
    E,
    A,
>;

/// Consuming proof that one execution lease and one retained Candle device
/// passed route validation together. Keeping the fields private and this type
/// non-cloneable prevents production composition from mixing authorities.
struct H3PrivateBoundExecution<E> {
    execution: E,
    device: Device,
}

impl<E> H3PrivateBoundExecution<E> {
    fn device(&self) -> &Device {
        &self.device
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3PrivateExecutionRouteFacts {
    active: bool,
    lease_id: String,
    device_id: String,
    execution_fingerprint: String,
    backend: H3CandleBackendDevice,
    location: DeviceLocation,
    attention_device: H3AttentionDevice,
}

fn validate_private_execution_route_facts(
    actual: &H3PrivateExecutionRouteFacts,
    admitted: &H3PrivateFl2VaFactoryAuthority,
) -> Result<()> {
    let expected_attention_device = match admitted.compute_capability {
        Some(compute_capability) => H3AttentionDevice::Cuda {
            compute_capability: Some(compute_capability),
        },
        None => H3AttentionDevice::Metal,
    };
    if !actual.active
        || actual.lease_id.trim().is_empty()
        || actual.device_id != admitted.device_id
        || actual.execution_fingerprint != admitted.execution_fingerprint
        || actual.backend
            != H3CandleBackendDevice::from_compute_capability(admitted.compute_capability)
        || actual.location
            != (DeviceLocation::Cuda {
                gpu_id: admitted.device_ordinal,
            })
        || admitted.attention.device != expected_attention_device
        || actual.attention_device != admitted.attention.device
    {
        bail!("private MiniMax H3 execution route differs before component binding");
    }
    Ok(())
}

/// Snapshot the safe execution-lease surface once. In particular, the Candle
/// device is cloned from one call and becomes the only device the bound
/// transformer stream may later use.
fn capture_private_execution_route<E>(
    execution: E,
    admitted: &H3PrivateFl2VaFactoryAuthority,
) -> Result<H3PrivateBoundExecution<E>>
where
    E: H3BackendExecutionLease,
{
    let device = execution.device().clone();
    let facts = H3PrivateExecutionRouteFacts {
        active: execution.is_active(),
        lease_id: execution.lease_id().into(),
        device_id: execution.device_id().into(),
        execution_fingerprint: execution.execution_fingerprint().into(),
        backend: execution.backend(),
        location: device.location(),
        attention_device: H3AttentionDevice::from_candle(&device),
    };
    validate_private_execution_route_facts(&facts, admitted)?;
    Ok(H3PrivateBoundExecution { execution, device })
}

/// The second authenticated VAE open, retained from admission so both VAEs
/// can be parked across the whole denoise and reconstructed before visual
/// decode.
///
/// Reconstruction must never re-resolve a replaceable pathname, so this
/// authority carries its own opened source descriptors and privately staged
/// copies from the moment admission authenticated them. Its
/// artifact-validation identity — the one VAE-open identity defined for
/// comparing independently reopened artifacts — is pinned to the primary
/// open's, so a reload can only ever construct the same authenticated bytes
/// the attempt was admitted against.
pub(crate) struct H3PrivateRetainedVaeReload {
    authority: H3AuthenticatedComfyVaeAuthority,
    artifact_validation_identity_sha256: String,
    artifact_plan_identity_sha256: String,
}

impl H3PrivateRetainedVaeReload {
    pub(crate) fn bind(
        authority: H3AuthenticatedComfyVaeAuthority,
        primary: &H3AuthenticatedComfyVaeAuthority,
    ) -> Result<Self> {
        authority.validate()?;
        primary.validate()?;
        let retained = Self {
            artifact_validation_identity_sha256: authority
                .artifact_validation_identity_sha256()
                .into(),
            artifact_plan_identity_sha256: authority.artifact_plan_identity_sha256().into(),
            authority,
        };
        if retained.artifact_validation_identity_sha256
            != primary.artifact_validation_identity_sha256()
            || retained.artifact_plan_identity_sha256 != primary.artifact_plan_identity_sha256()
            || retained.authority.task() != primary.task()
            || retained.authority.canonical_model() != primary.canonical_model()
        {
            bail!("private H3 retained VAE reload authority opened different artifacts")
        }
        Ok(retained)
    }

    fn authority(&self) -> &H3AuthenticatedComfyVaeAuthority {
        &self.authority
    }

    fn validate(&self, admitted: &H3PrivateFl2VaFactoryAuthority) -> Result<()> {
        self.authority.validate()?;
        if self.authority.artifact_validation_identity_sha256()
            != self.artifact_validation_identity_sha256
            || self.authority.artifact_plan_identity_sha256() != self.artifact_plan_identity_sha256
            || self.artifact_plan_identity_sha256 != admitted.vae_artifact_plan_identity_sha256
            || self.authority.task() != Task::Fl2va
            || self.authority.canonical_model() != contract::FL2VA_COMFY
        {
            bail!("private H3 retained VAE reload authority differs from the admitted attempt")
        }
        Ok(())
    }

    fn into_authority(
        self,
        admitted: &H3PrivateFl2VaFactoryAuthority,
    ) -> Result<H3AuthenticatedComfyVaeAuthority> {
        self.validate(admitted)?;
        Ok(self.authority)
    }
}

/// One consuming scheduler-to-runtime handoff. It owns every opened artifact,
/// lease, prepared tensor, and typed factory record needed by one attempt.
/// Neither this root nor the retained overlap record implements `Clone`.
///
/// The safe constructor below remains unreachable in production today because
/// `private_fl2va_runtime_authority` retains the evidence-backed activation
/// prerequisites. The overlap authority itself now has one production issuer,
/// but only from a prepared budget and its scheduler-ledger identity.
#[allow(dead_code)]
pub(crate) struct H3PrivatePhaseRuntimeOwner<C, E, A> {
    authority: FrozenH3FactoryAuthority,
    activation_evidence: H3PrivateFactoryActivationEvidence,
    admitted: H3PrivateFl2VaFactoryAuthority,
    prepared: H3PrivatePreparedFl2VaFactoryInputs,
    storage: H3PrivateComfyStorageAuthority,
    qwen_support: H3PrivateQwenSupport,
    opened_qwen: H3AuthenticatedQwenNvfp4Authority,
    opened_vae: H3AuthenticatedComfyVaeAuthority,
    reload_vae: H3PrivateRetainedVaeReload,
    bound_transformer: H3PrivateBoundComfyStream,
    stream_authority: H3PrivateComfyStreamAuthority,
    qwen_artifact_authority: H3PrivateQwenArtifactAuthority,
    conditioner_lease: C,
    memory_overlap: H3PrivateFl2VaMemoryOverlapAuthority,
    allocation_commit: H3PrivateAllocationCommit,
    // The singular attempt root is declared last so all opened/component state
    // releases before the scheduler execution and artifact leases.
    attempt: H3PrivateAttemptAuthority<E, A>,
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn bind_private_comfy_fl2va_phase_owner<C, E, A>(
    authority: FrozenH3FactoryAuthority,
    activation_evidence: H3PrivateFactoryActivationEvidence,
    prepared: H3PrivatePreparedFl2VaFactoryInputs,
    storage: H3PrivateComfyStorageAuthority,
    qwen_support: H3PrivateQwenSupport,
    opened_transformer: H3ComfyOpenedInt8Checkpoint,
    opened_qwen: H3AuthenticatedQwenNvfp4Authority,
    opened_vae: H3AuthenticatedComfyVaeAuthority,
    reload_vae: H3PrivateRetainedVaeReload,
    attention: H3AttentionRuntimeAuthority,
    conditioner_lease: C,
    execution_lease: E,
    artifact_lease: A,
    memory_overlap: H3PrivateFl2VaMemoryOverlapAuthority,
    allocation_commit: H3PrivateAllocationCommit,
) -> Result<H3PrivatePhaseRuntimeOwner<C, E, A>>
where
    C: H3PrivateQwenConditionerLease + Send + Sync,
    E: H3BackendExecutionLease + Send + Sync,
    A: H3PrivateFl2VaArtifactLease + Send + Sync,
{
    let admitted =
        authority.private_fl2va_runtime_authority_with_activation(&activation_evidence)?;
    prepared.revalidate()?;
    storage.validate_opened_components(
        &qwen_support,
        &opened_transformer,
        &opened_qwen,
        &opened_vae,
    )?;
    // The reload authority is proved to belong to the same hidden storage
    // here, while the components its containment check needs are still owned.
    storage.validate_opened_components(
        &qwen_support,
        &opened_transformer,
        &opened_qwen,
        reload_vae.authority(),
    )?;
    reload_vae.validate(&admitted)?;
    let qwen_artifact_authority =
        H3PrivateQwenArtifactAuthority::capture(&qwen_support, &opened_qwen)?;
    validate_prepared_overlap_binding(&authority, &admitted, &prepared, &memory_overlap)?;
    let bound_execution = capture_private_execution_route(execution_lease, &admitted)?;
    validate_initial_artifact_authority(&admitted, &artifact_lease, &memory_overlap)?;
    let bound_transformer = bind_private_comfy_stream(
        opened_transformer,
        bound_execution.device(),
        attention,
        H3PrivateComfyBindingExpectation {
            attention: admitted.attention.clone(),
            checkpoint: H3PrivateComfyCheckpointFacts {
                task: artifact_lease.transformer_task(),
                content_sha256: artifact_lease
                    .transformer_checkpoint_content_sha256()
                    .into(),
                layout_identity_sha256: artifact_lease
                    .transformer_checkpoint_layout_identity_sha256()
                    .into(),
                opened_checkpoint_identity_sha256: artifact_lease
                    .transformer_checkpoint_identity_sha256()
                    .into(),
                quantization_policy_identity_sha256: artifact_lease
                    .transformer_policy_identity_sha256()
                    .into(),
            },
        },
        // Frozen at admission; the load never re-reads the environment.
        admitted.quantization.turbo_adapter().cloned(),
    )?;
    let stream_authority = bound_transformer.authority().clone();
    validate_private_artifact_authority(
        &admitted,
        &stream_authority,
        &qwen_artifact_authority,
        &memory_overlap,
        &artifact_lease,
    )?;
    let attempt = H3PrivateAttemptAuthority::new(bound_execution, artifact_lease);
    Ok(H3PrivatePhaseRuntimeOwner {
        authority,
        activation_evidence,
        admitted,
        prepared,
        storage,
        qwen_support,
        opened_qwen,
        opened_vae,
        reload_vae,
        bound_transformer,
        stream_authority,
        qwen_artifact_authority,
        conditioner_lease,
        memory_overlap,
        allocation_commit,
        attempt,
    })
}

fn validate_prepared_overlap_binding(
    authority: &FrozenH3FactoryAuthority,
    admitted: &H3PrivateFl2VaFactoryAuthority,
    prepared: &H3PrivatePreparedFl2VaFactoryInputs,
    overlap: &H3PrivateFl2VaMemoryOverlapAuthority,
) -> Result<()> {
    prepared.revalidate()?;
    overlap.validate()?;
    let identities = authority
        .prepared_target_attempt_identities()
        .ok_or_else(|| anyhow::anyhow!("private H3 factory has no prepared target identities"))?;
    let request = &prepared.factory_attempt.request;
    let budget = &prepared.factory_attempt.target_budget;
    if identities
        != (
            prepared.prepared_attempt_identity_sha256(),
            prepared.target_budget_identity_sha256(),
        )
        || overlap.factory_identity_sha256 != admitted.factory_identity_sha256
        || overlap.prepared_attempt_identity_sha256 != identities.0
        || overlap.target_budget_identity_sha256 != identities.1
        || request.canonical_model != contract::FL2VA_COMFY
        || request.task != Task::Fl2va
        || request.rows.condition_visual_rows != admitted.condition_visual_rows
        || overlap.condition_visual_rows != request.rows.condition_visual_rows
        || overlap.condition_backing_host_bytes != budget.condition_backing_host_bytes
        || overlap.condition_backing_device_bytes
            != budget.condition_latent_backing_device_bytes
        || overlap.target_audio_latent_device_bytes != budget.target_audio_latent_device_bytes
        || overlap.visual_vae_resident_device_bytes
            != budget.visual_vae_resident_device_bytes
        || overlap.audio_vae_resident_device_bytes != budget.audio_vae_resident_device_bytes
        || overlap.attempt_resident_vae_device_bytes
            != budget.attempt_resident_vae_device_bytes
        || overlap.visual_decode_peak_device_bytes != budget.visual_decode_phase_device_bytes
        || overlap.normalized_endpoint_host_bytes != budget.normalized_endpoint_host_bytes
        || budget.load_drop_policy
            != H3FactoryTargetLoadDropPolicy::LoadVaesLoadQwenEncodeTransferDropQwenEncodeConditionsParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux
    {
        bail!("private H3 prepared attempt, target budget, and overlap authority differ before VAE allocation")
    }
    Ok(())
}

fn validate_initial_artifact_authority<A>(
    admitted: &H3PrivateFl2VaFactoryAuthority,
    artifacts: &A,
    overlap: &H3PrivateFl2VaMemoryOverlapAuthority,
) -> Result<()>
where
    A: H3PrivateFl2VaArtifactLease,
{
    let expected_transformer_policy = match &admitted.quantization {
        H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
            transformer_policy_sha256,
            ..
        } => transformer_policy_sha256,
        H3FactoryQuantizationAuthority::OfficialBf16 => {
            bail!("private MiniMax H3 Comfy runtime requires quantized transformer authority")
        }
    };
    if !artifacts.is_active()
        || artifacts.factory_identity_sha256() != admitted.factory_identity_sha256
        || artifacts.component_set_identity() != admitted.component_set_identity_sha256
        || artifacts.memory_overlap_identity_sha256() != overlap.identity_sha256()
        || artifacts.transformer_task() != H3TransformerTask::T2VaFl2Va
        || artifacts.transformer_policy_identity_sha256() != expected_transformer_policy
        || artifacts.attention_runtime_identity_sha256()
            != admitted.attention.runtime_identity_sha256
        || artifacts.attention_kernel_identity() != admitted.attention.qualification_kernel_identity
        || artifacts.attention_qualification_sha256() != admitted.attention.qualification_sha256
    {
        bail!("private MiniMax H3 artifact lease differs before component loading")
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3PrivateQwenArtifactAuthority {
    support_identity_sha256: String,
    weight_identity_sha256: String,
    weight_header_identity_sha256: String,
    weight_policy_identity_sha256: String,
}

impl H3PrivateQwenArtifactAuthority {
    fn capture(
        support: &H3PrivateQwenSupport,
        opened: &H3AuthenticatedQwenNvfp4Authority,
    ) -> Result<Self> {
        opened.revalidate()?;
        if support.model() != contract::FL2VA_COMFY || support.task() != Task::Fl2va {
            bail!("private H3 retained Qwen support has the wrong task partition")
        }
        let authority = Self {
            support_identity_sha256: support.support_identity_sha256().into(),
            weight_identity_sha256: opened.artifact_identity_sha256().into(),
            weight_header_identity_sha256: opened.header_identity_sha256().into(),
            weight_policy_identity_sha256: opened.policy_identity_sha256().into(),
        };
        if [
            authority.support_identity_sha256.as_str(),
            authority.weight_identity_sha256.as_str(),
            authority.weight_header_identity_sha256.as_str(),
            authority.weight_policy_identity_sha256.as_str(),
        ]
        .into_iter()
        .any(|identity| !valid_sha256(identity))
        {
            bail!("private H3 retained Qwen artifact authority is not exact SHA-256 evidence")
        }
        Ok(authority)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3PrivateArtifactAuthorityFacts {
    active: bool,
    factory_identity_sha256: String,
    backend_plan_identity_sha256: String,
    component_set_identity_sha256: String,
    conditioner_component_content_sha256: String,
    conditioner_component_validation_sha256: String,
    support_identity_sha256: String,
    weight_identity_sha256: String,
    weight_header_identity_sha256: String,
    weight_policy_identity_sha256: String,
    transformer_component_content_sha256: String,
    transformer_component_validation_sha256: String,
    visual_vae_component_content_sha256: String,
    visual_vae_component_validation_sha256: String,
    audio_vae_component_content_sha256: String,
    audio_vae_component_validation_sha256: String,
    vae_artifact_plan_identity_sha256: String,
    transformer_task: H3TransformerTask,
    transformer_checkpoint_content_sha256: String,
    transformer_checkpoint_layout_identity_sha256: String,
    transformer_checkpoint_identity_sha256: String,
    transformer_policy_identity_sha256: String,
    pruned_adaln_table_identity_sha256: String,
    attention_runtime_identity_sha256: String,
    attention_kernel_identity: String,
    attention_qualification_sha256: String,
    memory_overlap_identity_sha256: String,
}

impl H3PrivateArtifactAuthorityFacts {
    fn capture<A>(artifacts: &A) -> Self
    where
        A: H3PrivateFl2VaArtifactLease,
    {
        Self {
            active: H3BackendArtifactLease::is_active(artifacts),
            factory_identity_sha256: artifacts.factory_identity_sha256().into(),
            backend_plan_identity_sha256: artifacts.backend_plan_identity_sha256().into(),
            component_set_identity_sha256: artifacts.component_set_identity().into(),
            conditioner_component_content_sha256: artifacts
                .conditioner_component_content_sha256()
                .into(),
            conditioner_component_validation_sha256: artifacts
                .conditioner_component_validation_sha256()
                .into(),
            support_identity_sha256: artifacts.support_identity_sha256().into(),
            weight_identity_sha256: artifacts.weight_identity_sha256().into(),
            weight_header_identity_sha256: artifacts.weight_header_identity_sha256().into(),
            weight_policy_identity_sha256: artifacts.weight_policy_identity_sha256().into(),
            transformer_component_content_sha256: artifacts
                .transformer_component_content_sha256()
                .into(),
            transformer_component_validation_sha256: artifacts
                .transformer_component_validation_sha256()
                .into(),
            visual_vae_component_content_sha256: artifacts
                .visual_vae_component_content_sha256()
                .into(),
            visual_vae_component_validation_sha256: artifacts
                .visual_vae_component_validation_sha256()
                .into(),
            audio_vae_component_content_sha256: artifacts
                .audio_vae_component_content_sha256()
                .into(),
            audio_vae_component_validation_sha256: artifacts
                .audio_vae_component_validation_sha256()
                .into(),
            vae_artifact_plan_identity_sha256: artifacts.vae_artifact_plan_identity_sha256().into(),
            transformer_task: artifacts.transformer_task(),
            transformer_checkpoint_content_sha256: artifacts
                .transformer_checkpoint_content_sha256()
                .into(),
            transformer_checkpoint_layout_identity_sha256: artifacts
                .transformer_checkpoint_layout_identity_sha256()
                .into(),
            transformer_checkpoint_identity_sha256: artifacts
                .transformer_checkpoint_identity_sha256()
                .into(),
            transformer_policy_identity_sha256: artifacts
                .transformer_policy_identity_sha256()
                .into(),
            pruned_adaln_table_identity_sha256: artifacts
                .pruned_adaln_table_identity_sha256()
                .into(),
            attention_runtime_identity_sha256: artifacts.attention_runtime_identity_sha256().into(),
            attention_kernel_identity: artifacts.attention_kernel_identity().into(),
            attention_qualification_sha256: artifacts.attention_qualification_sha256().into(),
            memory_overlap_identity_sha256: artifacts.memory_overlap_identity_sha256().into(),
        }
    }
}

fn validate_private_artifact_facts(
    admitted: &H3PrivateFl2VaFactoryAuthority,
    stream: &H3PrivateComfyStreamAuthority,
    qwen: &H3PrivateQwenArtifactAuthority,
    overlap: &H3PrivateFl2VaMemoryOverlapAuthority,
    facts: &H3PrivateArtifactAuthorityFacts,
) -> Result<()> {
    admitted.block_streaming.validate()?;
    overlap.validate()?;
    if admitted.task != Task::Fl2va
        || admitted.canonical_model != contract::FL2VA_COMFY
        || overlap.factory_identity_sha256 != admitted.factory_identity_sha256
        || overlap.condition_visual_rows != admitted.condition_visual_rows
        || facts.memory_overlap_identity_sha256 != overlap.identity_sha256()
        || !facts.active
        || facts.factory_identity_sha256 != admitted.factory_identity_sha256
        || facts.backend_plan_identity_sha256 != admitted.backend_plan_identity_sha256
        || facts.component_set_identity_sha256 != admitted.component_set_identity_sha256
        || facts.conditioner_component_content_sha256
            != admitted.conditioner_component_content_sha256
        || facts.conditioner_component_validation_sha256
            != admitted.conditioner_component_validation_sha256
        || facts.support_identity_sha256 != qwen.support_identity_sha256
        || facts.weight_identity_sha256 != qwen.weight_identity_sha256
        || facts.weight_header_identity_sha256 != qwen.weight_header_identity_sha256
        || facts.weight_policy_identity_sha256 != qwen.weight_policy_identity_sha256
        || facts.transformer_component_content_sha256
            != admitted.transformer_component_content_sha256
        || facts.transformer_component_validation_sha256
            != admitted.transformer_component_validation_sha256
        || facts.visual_vae_component_content_sha256 != admitted.visual_vae_component_content_sha256
        || facts.visual_vae_component_validation_sha256
            != admitted.visual_vae_component_validation_sha256
        || facts.audio_vae_component_content_sha256 != admitted.audio_vae_component_content_sha256
        || facts.audio_vae_component_validation_sha256
            != admitted.audio_vae_component_validation_sha256
        || facts.vae_artifact_plan_identity_sha256 != admitted.vae_artifact_plan_identity_sha256
        || stream.task != H3TransformerTask::T2VaFl2Va
        || facts.transformer_task != stream.task
        || facts.transformer_checkpoint_content_sha256 != stream.transformer_content_sha256
        || facts.transformer_checkpoint_layout_identity_sha256
            != stream.transformer_layout_identity_sha256
        || facts.transformer_checkpoint_identity_sha256 != stream.checkpoint_identity_sha256
        || facts.transformer_policy_identity_sha256 != stream.transformer_policy_identity_sha256
        || facts.attention_runtime_identity_sha256 != stream.attention_runtime_identity_sha256
        || facts.attention_runtime_identity_sha256 != admitted.attention.runtime_identity_sha256
        || facts.attention_kernel_identity != admitted.attention.qualification_kernel_identity
        || facts.attention_qualification_sha256 != admitted.attention.qualification_sha256
    {
        bail!("private MiniMax H3 continuing artifact authority differs from its bound stream")
    }
    match &admitted.quantization {
        H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
            transformer_policy_sha256,
            qwen_policy_sha256,
            pruned_adaln_table_sha256,
            ..
        } if transformer_policy_sha256 == &facts.transformer_policy_identity_sha256
            && transformer_policy_sha256 == &stream.transformer_policy_identity_sha256
            && qwen_policy_sha256 == &qwen.weight_policy_identity_sha256
            && pruned_adaln_table_sha256 == &facts.pruned_adaln_table_identity_sha256 => {}
        _ => bail!("private MiniMax H3 continuing quantization authority differs"),
    }
    Ok(())
}

fn validate_private_artifact_authority<A>(
    admitted: &H3PrivateFl2VaFactoryAuthority,
    stream: &H3PrivateComfyStreamAuthority,
    qwen: &H3PrivateQwenArtifactAuthority,
    overlap: &H3PrivateFl2VaMemoryOverlapAuthority,
    artifacts: &A,
) -> Result<String>
where
    A: H3PrivateFl2VaArtifactLease,
{
    let facts = H3PrivateArtifactAuthorityFacts::capture(artifacts);
    validate_private_artifact_facts(admitted, stream, qwen, overlap, &facts)?;
    Ok(facts.component_set_identity_sha256)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum H3PrivatePhaseState {
    Bound,
    VaesLoaded,
    QwenLoaded,
    QwenDropped,
    ConditionsEncoded,
    VaesParked,
    TransformerLoaded,
    TransformerDropped,
    VaesReloaded,
    VisualDecoded,
    Empty,
}

/// Task-neutral load/drop ledger shared by the FL2VA coordinator and the
/// future Ref2VA coordinator. It contains no artifact facts and cannot itself
/// activate either task partition.
struct H3PrivatePhaseLedger {
    state: H3PrivatePhaseState,
    expected_denoise_forwards: usize,
    completed_denoise_forwards: usize,
}

impl H3PrivatePhaseLedger {
    fn new(expected_denoise_forwards: usize) -> Result<Self> {
        if expected_denoise_forwards == 0 {
            bail!("private H3 phase ledger requires at least one denoise forward")
        }
        Ok(Self {
            state: H3PrivatePhaseState::Bound,
            expected_denoise_forwards,
            completed_denoise_forwards: 0,
        })
    }

    fn transition(
        &mut self,
        expected: &[H3PrivatePhaseState],
        next: H3PrivatePhaseState,
        label: &str,
    ) -> Result<()> {
        if !expected.contains(&self.state) {
            bail!("private H3 {label} occurred in phase {:?}", self.state)
        }
        self.state = next;
        Ok(())
    }

    fn vaes_loaded(&mut self) -> Result<()> {
        self.transition(
            &[H3PrivatePhaseState::Bound],
            H3PrivatePhaseState::VaesLoaded,
            "VAE load",
        )
    }

    fn qwen_loaded(&mut self) -> Result<()> {
        self.transition(
            &[H3PrivatePhaseState::VaesLoaded],
            H3PrivatePhaseState::QwenLoaded,
            "Qwen load",
        )
    }

    fn qwen_dropped(&mut self) -> Result<()> {
        self.transition(
            &[H3PrivatePhaseState::QwenLoaded],
            H3PrivatePhaseState::QwenDropped,
            "Qwen drop",
        )
    }

    fn conditions_encoded(&mut self) -> Result<()> {
        self.transition(
            &[
                H3PrivatePhaseState::QwenDropped,
                H3PrivatePhaseState::ConditionsEncoded,
            ],
            H3PrivatePhaseState::ConditionsEncoded,
            "condition encode",
        )
    }

    fn vaes_parked(&mut self) -> Result<()> {
        self.transition(
            &[
                H3PrivatePhaseState::QwenDropped,
                H3PrivatePhaseState::ConditionsEncoded,
            ],
            H3PrivatePhaseState::VaesParked,
            "VAE park",
        )
    }

    fn transformer_loaded(&mut self) -> Result<()> {
        self.transition(
            &[H3PrivatePhaseState::VaesParked],
            H3PrivatePhaseState::TransformerLoaded,
            "transformer load",
        )
    }

    fn vaes_reloaded(&mut self) -> Result<()> {
        self.transition(
            &[H3PrivatePhaseState::TransformerDropped],
            H3PrivatePhaseState::VaesReloaded,
            "VAE reload",
        )
    }

    fn visual_decoded(&mut self) -> Result<()> {
        self.transition(
            &[H3PrivatePhaseState::VaesReloaded],
            H3PrivatePhaseState::VisualDecoded,
            "visual decode",
        )
    }

    fn vaes_dropped(&mut self) -> Result<()> {
        self.transition(
            &[H3PrivatePhaseState::VisualDecoded],
            H3PrivatePhaseState::Empty,
            "VAE drop",
        )
    }

    fn denoise_completed(&mut self) -> Result<bool> {
        if self.state != H3PrivatePhaseState::TransformerLoaded
            || self.completed_denoise_forwards >= self.expected_denoise_forwards
        {
            bail!("private H3 denoise crossed its frozen phase count")
        }
        self.completed_denoise_forwards += 1;
        let last = self.completed_denoise_forwards == self.expected_denoise_forwards;
        if last {
            self.state = H3PrivatePhaseState::TransformerDropped;
        }
        Ok(last)
    }

    fn is_terminal(&self) -> bool {
        self.state == H3PrivatePhaseState::Empty
            && self.completed_denoise_forwards == self.expected_denoise_forwards
    }
}

type H3PrivatePhaseDenoiser<E, A> = H3BlockStreamedDenoiser<
    H3PrivateComfyBlockLoader,
    H3PrivateExecutionProjection<E, A>,
    H3PrivateComfyTransformerExecutor,
>;

struct H3PrivatePhaseBackend<C, E, A>
where
    C: H3PrivateQwenConditionerLease,
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    vae: Option<H3ComfyVaeRuntimeBundle>,
    denoiser: Option<H3PrivatePhaseDenoiser<E, A>>,
    opened_vae: Option<H3AuthenticatedComfyVaeAuthority>,
    reload_vae: Option<H3PrivateRetainedVaeReload>,
    opened_qwen: Option<H3AuthenticatedQwenNvfp4Authority>,
    qwen_support: Option<H3PrivateQwenSupport>,
    conditioner_lease: Option<C>,
    bound_transformer: Option<H3PrivateBoundComfyStream>,
    stream_authority: H3PrivateComfyStreamAuthority,
    qwen_artifact_authority: H3PrivateQwenArtifactAuthority,
    qwen_execution: Option<H3PrivateExecutionProjection<E, A>>,
    qwen_artifacts: Option<H3PrivateArtifactProjection<E, A>>,
    block_execution: Option<H3PrivateExecutionProjection<E, A>>,
    continuing_execution: H3PrivateExecutionProjection<E, A>,
    continuing_artifacts: H3PrivateArtifactProjection<E, A>,
    identity: H3PipelineBackendIdentity,
    authority: FrozenH3FactoryAuthority,
    activation_evidence: H3PrivateFactoryActivationEvidence,
    admitted: H3PrivateFl2VaFactoryAuthority,
    ledger: H3PrivatePhaseLedger,
    storage: H3PrivateComfyStorageAuthority,
    retention: H3PrivatePreparedFl2VaRetention,
    memory_overlap: H3PrivateFl2VaMemoryOverlapAuthority,
    allocation_commit: H3PrivateAllocationCommit,
    cancellation_slot: H3PrivateComfyCancellationSlot,
    cancellation_guard: H3PrivateComfyCancellationGuard,
    // Last: singular scheduler/artifact owner outlives every projection.
    attempt: H3PrivateAttemptAuthority<E, A>,
}

impl<C, E, A> H3PrivatePhaseRuntimeOwner<C, E, A>
where
    C: H3PrivateQwenConditionerLease + Send + Sync,
    E: H3BackendExecutionLease + Send + Sync,
    A: H3PrivateFl2VaArtifactLease + Send + Sync,
{
    fn into_backend(
        self,
        progress: &ProgressReporter,
    ) -> Result<(
        super::pipeline::H3PreparedFl2VaRequest,
        H3PrivatePhaseBackend<C, E, A>,
    )> {
        let Self {
            authority,
            activation_evidence,
            admitted,
            prepared,
            storage,
            qwen_support,
            opened_qwen,
            opened_vae,
            reload_vae,
            bound_transformer,
            stream_authority,
            qwen_artifact_authority,
            conditioner_lease,
            memory_overlap,
            allocation_commit,
            attempt,
        } = self;
        let (prepared, retention) = prepared.into_runtime_parts();
        retention.revalidate()?;
        let expected_denoise_forwards =
            super::sampler::H3DualSchedule::new_for_sampler_with_video_shift(
                prepared.grid_points,
                admitted.quantization.sampler_kind(),
                admitted.quantization.video_shift(),
            )?
            .counts()
            .transformer_evaluations;
        if expected_denoise_forwards != retention.denoise_forward_count()? {
            bail!("private H3 prepared denoise count differs from retained factory authority")
        }
        let (qwen_execution, qwen_artifacts) = attempt.qwen_projections();
        let block_execution = attempt.block_projection();
        let continuing_execution = attempt.block_projection();
        let continuing_artifacts = attempt.artifact_projection();
        let cancellation_slot = H3PrivateComfyCancellationSlot::default();
        let cancellation_guard = cancellation_slot.install(progress)?;
        let identity = H3PipelineBackendIdentity {
            kind: H3PipelineBackendKind::Cuda,
            device_id: admitted.device_id.clone(),
            execution_fingerprint: admitted.execution_fingerprint.clone(),
        };
        let backend = H3PrivatePhaseBackend {
            vae: None,
            denoiser: None,
            opened_vae: Some(opened_vae),
            reload_vae: Some(reload_vae),
            opened_qwen: Some(opened_qwen),
            qwen_support: Some(qwen_support),
            conditioner_lease: Some(conditioner_lease),
            bound_transformer: Some(bound_transformer),
            stream_authority,
            qwen_artifact_authority,
            qwen_execution: Some(qwen_execution),
            qwen_artifacts: Some(qwen_artifacts),
            block_execution: Some(block_execution),
            continuing_execution,
            continuing_artifacts,
            identity,
            authority,
            activation_evidence,
            admitted,
            ledger: H3PrivatePhaseLedger::new(expected_denoise_forwards)?,
            storage,
            retention,
            memory_overlap,
            allocation_commit,
            cancellation_slot,
            cancellation_guard,
            attempt,
        };
        backend.validate_continuing_authority()?;
        Ok((prepared, backend))
    }
}

impl<C, E, A> H3PrivatePhaseBackend<C, E, A>
where
    C: H3PrivateQwenConditionerLease + Send + Sync,
    E: H3BackendExecutionLease + Send + Sync,
    A: H3PrivateFl2VaArtifactLease + Send + Sync,
{
    fn validate_continuing_authority(&self) -> Result<()> {
        validate_private_continuing_authority(
            &self.authority,
            &self.activation_evidence,
            &self.admitted,
            &self.stream_authority,
            &self.qwen_artifact_authority,
            &self.storage,
            &self.retention,
            &self.memory_overlap,
            &self.continuing_execution,
            &self.continuing_artifacts,
            &self.attempt,
        )?;
        if let Some(vae) = self.vae.as_ref() {
            vae.validate_authority()?;
            if vae.task() != Task::Fl2va
                || vae.canonical_model() != contract::FL2VA_COMFY
                || vae.artifact_plan_identity_sha256()
                    != self.admitted.vae_artifact_plan_identity_sha256
                || !vae.device().same_device(self.continuing_execution.device())
            {
                bail!("private H3 retained VAE differs from the consuming attempt")
            }
        }
        Ok(())
    }

    /// Construct both VAEs on the execution device from one consumed
    /// authenticated authority. The initial load and the post-denoise reload
    /// share this path so a reconstructed pair is authenticated, progress
    /// reported, and authority checked exactly as the first pair was.
    fn construct_vaes(
        &mut self,
        opened_vae: H3AuthenticatedComfyVaeAuthority,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::VaeLoad,
            completed: 0,
            total: 1,
        })?;
        let vae = {
            let authority = &self.authority;
            let activation_evidence = &self.activation_evidence;
            let admitted = &self.admitted;
            let stream_authority = &self.stream_authority;
            let qwen_artifact_authority = &self.qwen_artifact_authority;
            let storage = &self.storage;
            let retention = &self.retention;
            let memory_overlap = &self.memory_overlap;
            let continuing_execution = &self.continuing_execution;
            let continuing_artifacts = &self.continuing_artifacts;
            let attempt = &self.attempt;
            let mut vae_observer = H3PrivateVaeLoadCheckpoint::new(
                checkpoint,
                || {
                    validate_private_continuing_authority(
                        authority,
                        activation_evidence,
                        admitted,
                        stream_authority,
                        qwen_artifact_authority,
                        storage,
                        retention,
                        memory_overlap,
                        continuing_execution,
                        continuing_artifacts,
                        attempt,
                    )
                    .map(|_| ())
                },
                &self.allocation_commit,
            );
            let loaded = load_h3_comfy_vae_runtime_from_authority(
                opened_vae,
                continuing_execution.device(),
                &mut vae_observer,
            );
            vae_observer.finish(loaded)?
        };
        if !self.allocation_commit.is_committed() {
            bail!("private H3 VAE construction returned without an allocation commitment")
        }
        self.vae = Some(vae);
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::VaeLoad,
            completed: 1,
            total: 1,
        })?;
        self.validate_continuing_authority()
    }

    fn into_empty(self) -> Result<H3PrivateTerminalAttempt<E, A>> {
        self.validate_continuing_authority()?;
        if !self.ledger.is_terminal()
            || self.vae.is_some()
            || self.denoiser.is_some()
            || self.opened_vae.is_some()
            || self.reload_vae.is_some()
            || self.opened_qwen.is_some()
            || self.qwen_support.is_some()
            || self.conditioner_lease.is_some()
            || self.bound_transformer.is_some()
            || self.qwen_execution.is_some()
            || self.qwen_artifacts.is_some()
            || self.block_execution.is_some()
        {
            bail!("private H3 mux requires an empty terminal component state")
        }
        let Self {
            vae: _,
            denoiser: _,
            opened_vae: _,
            reload_vae: _,
            opened_qwen: _,
            qwen_support: _,
            conditioner_lease: _,
            bound_transformer: _,
            stream_authority,
            qwen_artifact_authority,
            qwen_execution: _,
            qwen_artifacts: _,
            block_execution: _,
            continuing_execution,
            continuing_artifacts,
            identity: _,
            authority,
            activation_evidence,
            admitted,
            ledger: _,
            storage,
            retention,
            memory_overlap,
            allocation_commit: _,
            cancellation_slot: _,
            cancellation_guard,
            attempt,
        } = self;
        Ok(H3PrivateTerminalAttempt {
            continuing_execution,
            continuing_artifacts,
            authority,
            activation_evidence,
            admitted,
            stream_authority,
            qwen_artifact_authority,
            storage,
            retention,
            memory_overlap,
            cancellation_guard,
            attempt,
        })
    }

    fn validate_empty_terminal(&self) -> Result<()> {
        self.validate_continuing_authority()?;
        if !self.ledger.is_terminal()
            || self.vae.is_some()
            || self.denoiser.is_some()
            || self.opened_vae.is_some()
            || self.reload_vae.is_some()
            || self.opened_qwen.is_some()
            || self.qwen_support.is_some()
            || self.conditioner_lease.is_some()
            || self.bound_transformer.is_some()
            || self.qwen_execution.is_some()
            || self.qwen_artifacts.is_some()
            || self.block_execution.is_some()
        {
            bail!("private H3 mux requires an empty terminal component state")
        }
        Ok(())
    }

    fn terminal_identity_echo(&self) -> Result<H3PrivatePhaseIdentityEcho> {
        self.validate_empty_terminal()?;
        let live = validate_private_continuing_authority(
            &self.authority,
            &self.activation_evidence,
            &self.admitted,
            &self.stream_authority,
            &self.qwen_artifact_authority,
            &self.storage,
            &self.retention,
            &self.memory_overlap,
            &self.continuing_execution,
            &self.continuing_artifacts,
            &self.attempt,
        )?;
        let prepared_attempt_identity_sha256 =
            self.retention.prepared_attempt_identity_sha256().to_owned();
        let target_budget_identity_sha256 =
            self.retention.target_budget_identity_sha256().to_owned();
        if !valid_sha256(&prepared_attempt_identity_sha256)
            || !valid_sha256(&target_budget_identity_sha256)
            || !valid_sha256(&live.component_set_identity_sha256)
        {
            bail!("private H3 terminal identity echo contains an invalid digest")
        }
        Ok(H3PrivatePhaseIdentityEcho {
            device_id: live.device_id,
            execution_fingerprint: live.execution_fingerprint,
            prepared_attempt_identity_sha256,
            target_budget_identity_sha256,
            component_set_identity_sha256: live.component_set_identity_sha256,
        })
    }
}

impl<C, E, A> H3Fl2VaBackend for H3PrivatePhaseBackend<C, E, A>
where
    C: H3PrivateQwenConditionerLease + Send + Sync,
    E: H3BackendExecutionLease + Send + Sync,
    A: H3PrivateFl2VaArtifactLease + Send + Sync,
{
    fn identity(&self) -> H3PipelineBackendIdentity {
        self.identity.clone()
    }

    fn device(&self) -> &Device {
        self.continuing_execution.device()
    }

    fn sampler_kind(&self) -> H3SamplerKind {
        self.admitted.quantization.sampler_kind()
    }

    fn sampler_video_shift(&self) -> f32 {
        self.admitted.quantization.video_shift()
    }

    fn encode_text(
        &mut self,
        prompt: &str,
        endpoints: &[H3PreparedEndpoint],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        self.validate_continuing_authority()?;
        self.ledger.vaes_loaded()?;
        let opened_vae = self
            .opened_vae
            .take()
            .ok_or_else(|| anyhow::anyhow!("private H3 VAE authority was already consumed"))?;
        self.construct_vaes(opened_vae, checkpoint)?;

        self.ledger.qwen_loaded()?;
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::QwenLoad,
            completed: 0,
            total: 1,
        })?;
        let qwen = H3PrivateQwenAdapter::load_authorized_from_opened(
            self.opened_qwen
                .take()
                .ok_or_else(|| anyhow::anyhow!("private H3 Qwen authority was already consumed"))?,
            self.qwen_support
                .take()
                .ok_or_else(|| anyhow::anyhow!("private H3 Qwen support was already consumed"))?,
            self.conditioner_lease.take().ok_or_else(|| {
                anyhow::anyhow!("private H3 conditioner lease was already consumed")
            })?,
            self.qwen_execution
                .take()
                .ok_or_else(|| anyhow::anyhow!("private H3 Qwen execution was already consumed"))?,
            self.qwen_artifacts.take().ok_or_else(|| {
                anyhow::anyhow!("private H3 Qwen artifacts were already consumed")
            })?,
            &self.authority,
            checkpoint,
        )?;
        let mut qwen = ManuallyDrop::new(qwen);
        let text = (|| {
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::QwenLoad,
                completed: 1,
                total: 1,
            })?;
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::QwenEncode,
                completed: 0,
                total: 1,
            })?;
            let text = qwen.encode_fl2va(prompt, endpoints, checkpoint);
            let text = text.and_then(|text| {
                checkpoint.checkpoint(H3PipelineEvent {
                    phase: H3PipelinePhase::QwenEncode,
                    completed: 1,
                    total: 1,
                })?;
                Ok(text)
            });
            let continuing = qwen.validate_continuing_authorities();
            text.and_then(|text| continuing.map(|()| text))
        })();
        if text.as_ref().is_err_and(is_fatal_private_cuda_error) {
            return text;
        }
        // SAFETY: the fatal path above intentionally retains the concrete
        // conditioner. Every ordinary result releases it exactly once.
        unsafe { ManuallyDrop::drop(&mut qwen) };
        self.ledger.qwen_dropped()?;
        let text = text?;
        self.validate_continuing_authority()?;
        Ok(text)
    }

    fn encode_visual_condition(
        &mut self,
        endpoint: &H3PreparedEndpoint,
        mode: mold_candle::minimax_h3::ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor> {
        self.validate_continuing_authority()?;
        if !matches!(
            self.ledger.state,
            H3PrivatePhaseState::QwenDropped | H3PrivatePhaseState::ConditionsEncoded
        ) {
            bail!("private H3 condition encode occurred before Qwen drop")
        }
        let result = self
            .vae
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("private H3 VAE was not retained"))?
            .encode_visual_condition(endpoint, mode, checkpoint)?;
        self.ledger.conditions_encoded()?;
        self.validate_continuing_authority()?;
        Ok(result)
    }

    fn park_condition_components(
        &mut self,
        _checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        // Validate before releasing: a revoked authority leaves the resident
        // pair alone, mirroring the fatal-CUDA retention rule below. Nothing
        // between here and visual decode reads a VAE, so their ~5.8 GB is
        // removed from the transformer's own peak rather than added to it.
        self.validate_continuing_authority()?;
        drop(self.vae.take());
        self.ledger.vaes_parked()?;
        self.validate_continuing_authority()
    }

    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput> {
        self.validate_continuing_authority()?;
        if self.denoiser.is_none() {
            self.ledger.transformer_loaded()?;
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::TransformerLoad,
                completed: 0,
                total: 1,
            })?;
            let stream = load_and_pair_private_comfy_stream(
                self.bound_transformer.take().ok_or_else(|| {
                    anyhow::anyhow!("private H3 transformer authority was already consumed")
                })?,
                &self.admitted.block_streaming,
                self.cancellation_slot.clone(),
            )?;
            if stream.authority != self.stream_authority {
                bail!("private H3 loaded transformer differs from its retained bound authority")
            }
            let denoiser = H3BlockStreamedDenoiser::new(
                self.identity.clone(),
                self.admitted.block_streaming.clone(),
                self.block_execution.take().ok_or_else(|| {
                    anyhow::anyhow!("private H3 block execution was already consumed")
                })?,
                stream.loader,
                stream.executor,
            )?;
            self.denoiser = Some(denoiser);
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::TransformerLoad,
                completed: 1,
                total: 1,
            })?;
        }
        let output = self
            .denoiser
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("private H3 transformer was not loaded"))?
            .denoise(input, layout, checkpoint)?;
        if self.ledger.denoise_completed()? {
            drop(self.denoiser.take());
        }
        self.validate_continuing_authority()?;
        Ok(output)
    }

    fn decode_video(
        &mut self,
        latents: &Tensor,
        sink: &mut H3VideoEncodeSink,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        self.validate_continuing_authority()?;
        if self.vae.is_none() {
            let reload = self
                .reload_vae
                .take()
                .ok_or_else(|| {
                    anyhow::anyhow!("private H3 VAE reload authority was already consumed")
                })?
                .into_authority(&self.admitted)?;
            self.construct_vaes(reload, checkpoint)?;
            self.ledger.vaes_reloaded()?;
        }
        self.ledger.visual_decoded()?;
        self.vae
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("private H3 VAE was not retained for visual decode"))?
            .decode_video(latents, sink, checkpoint)?;
        self.validate_continuing_authority()
    }

    fn decode_audio(
        &mut self,
        latents: &StereoLatents,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform> {
        self.validate_continuing_authority()?;
        if self.ledger.state != H3PrivatePhaseState::VisualDecoded {
            bail!("private H3 audio decode occurred before visual decode")
        }
        let waveform = self
            .vae
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("private H3 VAE was not retained for audio decode"))?;
        let waveform = waveform.decode_audio(latents, checkpoint);
        if waveform.as_ref().is_err_and(is_fatal_private_cuda_error) {
            return waveform;
        }
        drop(self.vae.take());
        let waveform = waveform?;
        self.ledger.vaes_dropped()?;
        self.validate_continuing_authority()?;
        Ok(waveform)
    }
}

struct H3PrivateTerminalAttempt<E, A> {
    continuing_execution: H3PrivateExecutionProjection<E, A>,
    continuing_artifacts: H3PrivateArtifactProjection<E, A>,
    authority: FrozenH3FactoryAuthority,
    activation_evidence: H3PrivateFactoryActivationEvidence,
    admitted: H3PrivateFl2VaFactoryAuthority,
    stream_authority: H3PrivateComfyStreamAuthority,
    qwen_artifact_authority: H3PrivateQwenArtifactAuthority,
    storage: H3PrivateComfyStorageAuthority,
    retention: H3PrivatePreparedFl2VaRetention,
    memory_overlap: H3PrivateFl2VaMemoryOverlapAuthority,
    cancellation_guard: H3PrivateComfyCancellationGuard,
    // Last: leases remain active until mux and terminal validation finish.
    attempt: H3PrivateAttemptAuthority<E, A>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivatePhaseIdentityEcho {
    pub(crate) device_id: String,
    pub(crate) execution_fingerprint: String,
    pub(crate) prepared_attempt_identity_sha256: String,
    pub(crate) target_budget_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
}

#[cfg(feature = "mp4")]
pub(crate) struct H3PrivatePhaseRuntimeOutput {
    pub(crate) output: super::pipeline::H3PipelineOutput,
    pub(crate) identity_echo: H3PrivatePhaseIdentityEcho,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3PrivateValidatedLiveIdentity {
    device_id: String,
    execution_fingerprint: String,
    component_set_identity_sha256: String,
}

impl<E, A> H3PrivateTerminalAttempt<E, A>
where
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    fn validate(&self) -> Result<H3PrivateValidatedLiveIdentity> {
        let live = validate_private_continuing_authority(
            &self.authority,
            &self.activation_evidence,
            &self.admitted,
            &self.stream_authority,
            &self.qwen_artifact_authority,
            &self.storage,
            &self.retention,
            &self.memory_overlap,
            &self.continuing_execution,
            &self.continuing_artifacts,
            &self.attempt,
        )?;
        let _ = &self.cancellation_guard;
        Ok(live)
    }

    fn identity_echo(&self) -> Result<H3PrivatePhaseIdentityEcho> {
        let live = self.validate()?;
        let prepared_attempt_identity_sha256 =
            self.retention.prepared_attempt_identity_sha256().to_owned();
        let target_budget_identity_sha256 =
            self.retention.target_budget_identity_sha256().to_owned();
        if !valid_sha256(&prepared_attempt_identity_sha256)
            || !valid_sha256(&target_budget_identity_sha256)
            || !valid_sha256(&live.component_set_identity_sha256)
        {
            bail!("private H3 terminal identity echo contains an invalid digest")
        }
        Ok(H3PrivatePhaseIdentityEcho {
            device_id: live.device_id,
            execution_fingerprint: live.execution_fingerprint,
            prepared_attempt_identity_sha256,
            target_budget_identity_sha256,
            component_set_identity_sha256: live.component_set_identity_sha256,
        })
    }
}

fn snapshot_private_continuing_route<E, A>(
    execution: &H3PrivateExecutionProjection<E, A>,
) -> H3PrivateExecutionRouteFacts
where
    E: H3BackendExecutionLease,
{
    let device = H3BackendExecutionLease::device(execution);
    H3PrivateExecutionRouteFacts {
        active: H3BackendExecutionLease::is_active(execution),
        lease_id: H3BackendExecutionLease::lease_id(execution).into(),
        device_id: H3BackendExecutionLease::device_id(execution).into(),
        execution_fingerprint: H3BackendExecutionLease::execution_fingerprint(execution).into(),
        backend: H3BackendExecutionLease::backend(execution),
        location: device.location(),
        attention_device: H3AttentionDevice::from_candle(device),
    }
}

fn validate_private_continuing_execution_route_facts(
    actual: &H3PrivateExecutionRouteFacts,
    admitted: &H3PrivateFl2VaFactoryAuthority,
) -> Result<()> {
    if cfg!(test)
        && actual.location == DeviceLocation::Cpu
        && actual.attention_device == H3AttentionDevice::Cpu
    {
        if !actual.active
            || actual.lease_id.trim().is_empty()
            || actual.device_id != admitted.device_id
            || actual.execution_fingerprint != admitted.execution_fingerprint
            || actual.backend
                != H3CandleBackendDevice::from_compute_capability(admitted.compute_capability)
        {
            bail!("private MiniMax H3 continuing synthetic route differs from admission")
        }
        return Ok(());
    }
    validate_private_execution_route_facts(actual, admitted)
}

fn validate_private_live_attempt_authority<E, A>(
    admitted: &H3PrivateFl2VaFactoryAuthority,
    stream_authority: &H3PrivateComfyStreamAuthority,
    qwen_artifact_authority: &H3PrivateQwenArtifactAuthority,
    memory_overlap: &H3PrivateFl2VaMemoryOverlapAuthority,
    continuing_execution: &H3PrivateExecutionProjection<E, A>,
    continuing_artifacts: &H3PrivateArtifactProjection<E, A>,
    attempt: &H3PrivateAttemptAuthority<E, A>,
) -> Result<H3PrivateValidatedLiveIdentity>
where
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    if !continuing_execution.belongs_to(attempt) || !continuing_artifacts.belongs_to(attempt) {
        bail!("private H3 phase projections came from different attempts")
    }
    let route = snapshot_private_continuing_route(continuing_execution);
    validate_private_continuing_execution_route_facts(&route, admitted)?;
    let component_set_identity_sha256 = validate_private_artifact_authority(
        admitted,
        stream_authority,
        qwen_artifact_authority,
        memory_overlap,
        continuing_artifacts,
    )?;
    Ok(H3PrivateValidatedLiveIdentity {
        device_id: route.device_id,
        execution_fingerprint: route.execution_fingerprint,
        component_set_identity_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_private_continuing_authority<E, A>(
    authority: &FrozenH3FactoryAuthority,
    activation_evidence: &H3PrivateFactoryActivationEvidence,
    admitted: &H3PrivateFl2VaFactoryAuthority,
    stream_authority: &H3PrivateComfyStreamAuthority,
    qwen_artifact_authority: &H3PrivateQwenArtifactAuthority,
    storage: &H3PrivateComfyStorageAuthority,
    retention: &H3PrivatePreparedFl2VaRetention,
    memory_overlap: &H3PrivateFl2VaMemoryOverlapAuthority,
    continuing_execution: &H3PrivateExecutionProjection<E, A>,
    continuing_artifacts: &H3PrivateArtifactProjection<E, A>,
    attempt: &H3PrivateAttemptAuthority<E, A>,
) -> Result<H3PrivateValidatedLiveIdentity>
where
    E: H3BackendExecutionLease,
    A: H3PrivateFl2VaArtifactLease,
{
    activation_evidence.revalidate_for(authority)?;
    retention.revalidate()?;
    storage.validate()?;
    memory_overlap.validate()?;
    authority.validate_engine_seam(
        contract::FL2VA_COMFY,
        admitted.device_ordinal,
        authority.block_offload(),
    )?;
    if retention.prepared_attempt_identity_sha256()
        != memory_overlap.prepared_attempt_identity_sha256
        || retention.target_budget_identity_sha256() != memory_overlap.target_budget_identity_sha256
    {
        bail!("private H3 phase authority changed during the consuming attempt")
    }
    validate_private_live_attempt_authority(
        admitted,
        stream_authority,
        qwen_artifact_authority,
        memory_overlap,
        continuing_execution,
        continuing_artifacts,
        attempt,
    )
}

struct H3PrivateVaeLoadCheckpoint<'a, F> {
    checkpoint: &'a mut dyn H3PipelineCheckpoint,
    revalidate: F,
    allocation_commit: &'a H3PrivateAllocationCommit,
    first_error: Option<anyhow::Error>,
}

impl<'a, F> H3PrivateVaeLoadCheckpoint<'a, F>
where
    F: FnMut() -> Result<()>,
{
    fn new(
        checkpoint: &'a mut dyn H3PipelineCheckpoint,
        revalidate: F,
        allocation_commit: &'a H3PrivateAllocationCommit,
    ) -> Self {
        Self {
            checkpoint,
            revalidate,
            allocation_commit,
            first_error: None,
        }
    }

    fn finish<T>(
        self,
        result: std::result::Result<T, super::vae_runtime::H3ComfyVaeLoadError>,
    ) -> Result<T> {
        if let Some(error) = self.first_error {
            Err(error)
        } else {
            Ok(result?)
        }
    }
}

impl<F> H3ComfyVaeLoadObserver for H3PrivateVaeLoadCheckpoint<'_, F>
where
    F: FnMut() -> Result<()>,
{
    fn checkpoint(&mut self, event: H3ComfyVaeLoadEvent) -> bool {
        if self.first_error.is_some() {
            return false;
        }
        if let Err(error) = (self.revalidate)() {
            self.first_error = Some(error);
            return false;
        }
        if !self.allocation_commit.is_committed() {
            self.first_error = Some(anyhow::anyhow!(
                "private H3 VAE loading reached CUDA before the owner allocation commitment"
            ));
            return false;
        }
        let total = usize::try_from(event.total).unwrap_or(usize::MAX).max(1);
        let completed = usize::try_from(event.completed)
            .unwrap_or(usize::MAX)
            .min(total);
        if let Err(error) = self.checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::VaeLoadChunk,
            completed,
            total,
        }) {
            self.first_error = Some(error);
            false
        } else {
            true
        }
    }
}

fn is_fatal_private_cuda_error(error: &anyhow::Error) -> bool {
    let message = format!("{error:#}");
    [
        "CUDA_ERROR_ILLEGAL_ADDRESS",
        "CUDA_ERROR_ECC_UNCORRECTABLE",
        "CUDA_ERROR_LAUNCH_FAILED",
        "CUDA_ERROR_ASSERT",
        "CUDA_ERROR_MISALIGNED_ADDRESS",
        "CUDA_ERROR_HARDWARE_STACK_ERROR",
        "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        "CUDA_ERROR_INVALID_PC",
        "CUDA_ERROR_LAUNCH_TIMEOUT",
        "CUDA_ERROR_EXTERNAL_DEVICE",
        "CUDA_ERROR_MPS_CLIENT_TERMINATED",
        "CUDA_ERROR_CONTAINED",
        "CUDA_ERROR_TENSOR_MEMORY_LEAK",
        "CUBLAS_STATUS_MAPPING_ERROR",
        "CUBLAS_STATUS_EXECUTION_FAILED",
        "CUBLAS_STATUS_INTERNAL_ERROR",
        "CURAND_STATUS_LAUNCH_FAILURE",
        "CURAND_STATUS_PREEXISTING_FAILURE",
        "CURAND_STATUS_INTERNAL_ERROR",
        "CUDA execution attempt retained resources",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

/// Run with singular ownership of every concrete CUDA-bearing component.
/// Fatal driver errors and panics deliberately retain the whole resource graph
/// for process teardown; ordinary errors, cancellation, and success explicitly
/// release it on the owner thread.
fn with_contained_private_cuda_resources<R, T>(
    resources: R,
    operation: impl FnOnce(&mut R) -> Result<T>,
) -> Result<T> {
    let mut resources = ManuallyDrop::new(resources);
    let outcome = catch_unwind(AssertUnwindSafe(|| operation(&mut resources)));
    match outcome {
        Err(payload) => resume_unwind(payload),
        Ok(Err(error)) if is_fatal_private_cuda_error(&error) => Err(error),
        Ok(result) => {
            // SAFETY: fatal and panic paths return above without dropping. All
            // remaining paths reach this exactly once with the resource live.
            unsafe { ManuallyDrop::drop(&mut resources) };
            result
        }
    }
}

/// The sole private FL2VA mux path. It consumes an empty terminal proof and
/// retains the singular attempt authority across the complete AAC mux.
#[cfg(feature = "mp4")]
#[allow(dead_code)]
pub(crate) fn run_private_comfy_fl2va_attempt<C, E, A>(
    owner: H3PrivatePhaseRuntimeOwner<C, E, A>,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3PrivatePhaseRuntimeOutput>
where
    C: H3PrivateQwenConditionerLease + Send + Sync,
    E: H3BackendExecutionLease + Send + Sync,
    A: H3PrivateFl2VaArtifactLease + Send + Sync,
{
    let (prepared, backend) = owner.into_backend(progress)?;
    with_contained_private_cuda_resources(backend, |backend| {
        let staged = super::pipeline::execute_staged(&prepared, backend, progress, observer)?;
        let identity_echo = backend.terminal_identity_echo()?;
        let output = super::pipeline::finalize_av(staged, progress, observer)?;
        let post_mux_identity = backend.terminal_identity_echo()?;
        if output.provenance.device_id != identity_echo.device_id
            || output.provenance.execution_fingerprint != identity_echo.execution_fingerprint
            || post_mux_identity != identity_echo
        {
            bail!("private H3 mux output identity differs from the terminal authority")
        }
        Ok(H3PrivatePhaseRuntimeOutput {
            output,
            identity_echo,
        })
    })
}

mod ref2va_opened_seal {
    #[cfg(not(test))]
    pub enum Token {}

    #[cfg(test)]
    pub struct Token;
}

/// Shared phase machinery is ready for Ref2VA, but no safe owner can exist
/// until that partition lands its own exact opened/prepared evidence.
#[allow(dead_code)]
pub(crate) struct H3PrivateRef2VaPhaseOwner {
    _opened_evidence: ref2va_opened_seal::Token,
}

#[allow(dead_code)]
pub(crate) fn run_private_comfy_ref2va_attempt(_owner: H3PrivateRef2VaPhaseOwner) -> Result<()> {
    bail!("private H3 Ref2VA runtime has no exact opened/prepared evidence authority")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use candle_core::DType;
    use mold_candle::minimax_h3::{ConditionEncodeMode, H3Modality, H3ModalityTag, H3PackedLayout};

    use super::*;
    use crate::attention::{AttentionBackend, AttentionChunkPolicy};
    use crate::minimax_h3::engine::{H3StreamedFl2VaBackend, H3StreamedTransformerExecutor};
    use crate::minimax_h3::offload::H3BlockLoader;
    use crate::minimax_h3::pipeline::{
        H3EndpointAnchor, H3EndpointResize, H3PipelineEvent, H3PipelinePhase,
    };
    use crate::minimax_h3::private_vae_adapter::H3PrivateComfyVaeAdapter;
    use crate::progress::{is_inference_cancelled, InferenceCancelled};
    use crate::{
        H3FactoryAuthorityInput, H3FactoryComponentAuthority, H3FactoryComponentRole,
        H3FactoryConditionerPlacement,
    };

    const EXECUTION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn authority() -> FrozenH3FactoryAuthority {
        authority_with_condition_visual_rows(2)
    }

    fn authority_with_condition_visual_rows(
        condition_visual_rows: u64,
    ) -> FrozenH3FactoryAuthority {
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: contract::FL2VA_COMFY.into(),
            device_id: "test-cpu".into(),
            device_ordinal: 0,
            compute_capability: Some((8, 9)),
            execution_fingerprint: EXECUTION.into(),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 2_048,
            qwen_host_resident_parameter_bytes: 2_048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1_024,
            qwen_maximum_tensor_staging_bytes: 512,
            qwen_retained_raw_header_bytes: 64,
            qwen_output_text_rows: 1,
            qwen_vision_rows: if condition_visual_rows == 0 { 0 } else { 64 },
            condition_visual_rows,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: "synthetic-qualified-kernel".into(),
            attention_qualification_sha256: sha('b'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            block_offload: true,
            attention_runtime: None,
            prepared_attempt: None,
            execution_budget_echo: None,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('c'),
                qwen_policy_sha256: sha('d'),
                pruned_adaln_table_sha256: sha('e'),
                turbo_adapter: None,
            },
            components: [
                H3FactoryComponentRole::Conditioner,
                H3FactoryComponentRole::Transformer,
                H3FactoryComponentRole::VisualVae,
                H3FactoryComponentRole::AudioVae,
            ]
            .into_iter()
            .enumerate()
            .map(|(index, role)| {
                H3FactoryComponentAuthority::new(
                    role,
                    sha(char::from(b'1' + index as u8)),
                    sha(char::from(b'5' + index as u8)),
                )
                .unwrap()
            })
            .collect(),
        })
        .unwrap()
    }

    fn exact_execution_route_facts(
        admitted: &H3PrivateFl2VaFactoryAuthority,
    ) -> H3PrivateExecutionRouteFacts {
        H3PrivateExecutionRouteFacts {
            active: true,
            lease_id: "execution-lease".into(),
            device_id: admitted.device_id.clone(),
            execution_fingerprint: admitted.execution_fingerprint.clone(),
            backend: H3CandleBackendDevice::from_compute_capability(admitted.compute_capability),
            location: candle_core::DeviceLocation::Cuda {
                gpu_id: admitted.device_ordinal,
            },
            attention_device: admitted.attention.device,
        }
    }

    #[test]
    fn every_execution_route_axis_fails_closed_at_binding_and_continuing_validation() {
        let admitted = authority()
            .private_fl2va_runtime_authority_for_schema_tests()
            .unwrap();
        validate_private_execution_route_facts(&exact_execution_route_facts(&admitted), &admitted)
            .unwrap();
        validate_private_continuing_execution_route_facts(
            &exact_execution_route_facts(&admitted),
            &admitted,
        )
        .unwrap();

        for axis in [
            "active",
            "lease-id",
            "device-id",
            "execution",
            "backend",
            "location",
            "attention-device",
        ] {
            let mut facts = exact_execution_route_facts(&admitted);
            match axis {
                "active" => facts.active = false,
                "lease-id" => facts.lease_id = "   ".into(),
                "device-id" => facts.device_id = "other-device".into(),
                "execution" => facts.execution_fingerprint = sha('f'),
                "backend" => {
                    facts.backend = H3CandleBackendDevice::Cuda {
                        compute_capability: (8, 6),
                    }
                }
                "location" => {
                    facts.location = candle_core::DeviceLocation::Cuda {
                        gpu_id: admitted.device_ordinal + 1,
                    }
                }
                "attention-device" => {
                    facts.attention_device = mold_candle::minimax_h3::H3AttentionDevice::Cpu
                }
                _ => unreachable!(),
            }
            let error = validate_private_execution_route_facts(&facts, &admitted).expect_err(axis);
            assert!(
                error.to_string().contains("execution route"),
                "{axis}: {error}"
            );
            assert!(
                validate_private_continuing_execution_route_facts(&facts, &admitted).is_err(),
                "continuing {axis}"
            );
        }
    }

    struct AlternatingDeviceLease {
        devices: [Device; 2],
        calls: Arc<AtomicUsize>,
        device_id: String,
        execution_fingerprint: String,
        compute_capability: Option<(u16, u16)>,
    }

    impl H3BackendExecutionLease for AlternatingDeviceLease {
        fn lease_id(&self) -> &str {
            "alternating-execution-lease"
        }

        fn device_id(&self) -> &str {
            &self.device_id
        }

        fn backend(&self) -> H3CandleBackendDevice {
            H3CandleBackendDevice::from_compute_capability(self.compute_capability)
        }

        fn execution_fingerprint(&self) -> &str {
            &self.execution_fingerprint
        }

        fn device(&self) -> &Device {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            &self.devices[usize::from(call > 0)]
        }

        fn is_active(&self) -> bool {
            true
        }
    }

    #[test]
    fn execution_route_capture_reads_an_adversarial_device_lease_once() {
        let admitted = authority()
            .private_fl2va_runtime_authority_for_schema_tests()
            .unwrap();
        let lease = AlternatingDeviceLease {
            devices: [Device::Cpu, Device::Cpu],
            calls: Arc::new(AtomicUsize::new(0)),
            device_id: admitted.device_id.clone(),
            execution_fingerprint: admitted.execution_fingerprint.clone(),
            compute_capability: admitted.compute_capability,
        };
        let error = capture_private_execution_route(&lease, &admitted)
            .err()
            .expect("the CPU route must fail before binding");
        assert!(error.to_string().contains("execution route"));
        assert_eq!(lease.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn attempt_projections_reuse_the_retained_device_without_repolling_the_lease() {
        let calls = Arc::new(AtomicUsize::new(0));
        let lease = AlternatingDeviceLease {
            devices: [Device::Cpu, Device::Cpu],
            calls: Arc::clone(&calls),
            device_id: "synthetic-device".into(),
            execution_fingerprint: sha('a'),
            compute_capability: Some((8, 9)),
        };
        let retained_device = lease.device().clone();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let attempt = H3PrivateAttemptAuthority::new_unchecked_for_test(lease, (), retained_device);
        let projection = attempt.block_projection();
        assert!(projection.device().is_cpu());
        assert!(projection.clone().device().is_cpu());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    fn overlap(admitted: &H3PrivateFl2VaFactoryAuthority) -> H3PrivateFl2VaMemoryOverlapAuthority {
        let (condition_host, condition_device, normalized_endpoints) =
            if admitted.condition_visual_rows == 0 {
                (0, 0, 0)
            } else {
                (10, 20, 60)
            };
        H3PrivateFl2VaMemoryOverlapAuthority::new(
            admitted.factory_identity_sha256.clone(),
            sha('a'),
            sha('b'),
            admitted.condition_visual_rows,
            condition_host,
            condition_device,
            30,
            40,
            50,
            90,
            120,
            normalized_endpoints,
        )
        .unwrap()
    }

    fn stream_authority(
        admitted: &H3PrivateFl2VaFactoryAuthority,
    ) -> H3PrivateComfyStreamAuthority {
        let transformer_policy_identity_sha256 = match &admitted.quantization {
            H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256,
                ..
            } => transformer_policy_sha256.clone(),
            H3FactoryQuantizationAuthority::OfficialBf16 => unreachable!(),
        };
        H3PrivateComfyStreamAuthority {
            task: H3TransformerTask::T2VaFl2Va,
            // Raw checkpoint bytes deliberately use a different identity
            // domain from the admitted logical component aggregate.
            transformer_content_sha256: sha('7'),
            transformer_layout_identity_sha256: sha('8'),
            checkpoint_identity_sha256: sha('9'),
            transformer_policy_identity_sha256,
            attention_runtime_identity_sha256: admitted.attention.runtime_identity_sha256.clone(),
        }
    }

    struct FakeExecution {
        device: Device,
        device_id: String,
        execution_fingerprint: String,
        compute_capability: Option<(u16, u16)>,
        active: Arc<AtomicBool>,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Drop for FakeExecution {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("execution".into());
        }
    }

    impl H3BackendExecutionLease for FakeExecution {
        fn lease_id(&self) -> &str {
            "execution-lease"
        }

        fn device_id(&self) -> &str {
            &self.device_id
        }

        fn backend(&self) -> H3CandleBackendDevice {
            H3CandleBackendDevice::from_compute_capability(self.compute_capability)
        }

        fn execution_fingerprint(&self) -> &str {
            &self.execution_fingerprint
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn is_active(&self) -> bool {
            self.active.load(Ordering::SeqCst)
        }
    }

    struct FakeArtifacts {
        admitted: H3PrivateFl2VaFactoryAuthority,
        stream: H3PrivateComfyStreamAuthority,
        overlap_identity_sha256: String,
        active: Arc<AtomicBool>,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Drop for FakeArtifacts {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("artifacts".into());
        }
    }

    // SAFETY: the synthetic object is immutable after construction and its
    // active flag models revocation of the one retained artifact set.
    unsafe impl H3BackendArtifactLease for FakeArtifacts {
        fn component_set_identity(&self) -> &str {
            &self.admitted.component_set_identity_sha256
        }

        fn is_active(&self) -> bool {
            self.active.load(Ordering::SeqCst)
        }
    }

    // SAFETY: the fixture returns identities copied from one frozen synthetic
    // admission and never exposes artifact storage.
    unsafe impl H3PrivateQwenArtifactLease for FakeArtifacts {
        fn factory_identity_sha256(&self) -> &str {
            &self.admitted.factory_identity_sha256
        }

        fn conditioner_component_content_sha256(&self) -> &str {
            &self.admitted.conditioner_component_content_sha256
        }

        fn conditioner_component_validation_sha256(&self) -> &str {
            &self.admitted.conditioner_component_validation_sha256
        }

        fn support_identity_sha256(&self) -> &str {
            "1111111111111111111111111111111111111111111111111111111111111111"
        }

        fn weight_identity_sha256(&self) -> &str {
            "2222222222222222222222222222222222222222222222222222222222222222"
        }

        fn weight_header_identity_sha256(&self) -> &str {
            "3333333333333333333333333333333333333333333333333333333333333333"
        }

        fn weight_policy_identity_sha256(&self) -> &str {
            match &self.admitted.quantization {
                H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                    qwen_policy_sha256,
                    ..
                } => qwen_policy_sha256,
                H3FactoryQuantizationAuthority::OfficialBf16 => {
                    unreachable!("synthetic private FL2VA authority must be quantized")
                }
            }
        }
    }

    // SAFETY: every identity below belongs to the same immutable synthetic
    // artifact lease and remains stable until `Drop`.
    unsafe impl H3PrivateFl2VaArtifactLease for FakeArtifacts {
        fn backend_plan_identity_sha256(&self) -> &str {
            &self.admitted.backend_plan_identity_sha256
        }

        fn transformer_component_content_sha256(&self) -> &str {
            &self.admitted.transformer_component_content_sha256
        }

        fn transformer_component_validation_sha256(&self) -> &str {
            &self.admitted.transformer_component_validation_sha256
        }

        fn visual_vae_component_content_sha256(&self) -> &str {
            &self.admitted.visual_vae_component_content_sha256
        }

        fn visual_vae_component_validation_sha256(&self) -> &str {
            &self.admitted.visual_vae_component_validation_sha256
        }

        fn audio_vae_component_content_sha256(&self) -> &str {
            &self.admitted.audio_vae_component_content_sha256
        }

        fn audio_vae_component_validation_sha256(&self) -> &str {
            &self.admitted.audio_vae_component_validation_sha256
        }

        fn vae_artifact_plan_identity_sha256(&self) -> &str {
            &self.admitted.vae_artifact_plan_identity_sha256
        }

        fn transformer_task(&self) -> H3TransformerTask {
            self.stream.task
        }

        fn transformer_checkpoint_content_sha256(&self) -> &str {
            &self.stream.transformer_content_sha256
        }

        fn transformer_checkpoint_layout_identity_sha256(&self) -> &str {
            &self.stream.transformer_layout_identity_sha256
        }

        fn transformer_checkpoint_identity_sha256(&self) -> &str {
            &self.stream.checkpoint_identity_sha256
        }

        fn transformer_policy_identity_sha256(&self) -> &str {
            &self.stream.transformer_policy_identity_sha256
        }

        fn pruned_adaln_table_identity_sha256(&self) -> &str {
            match &self.admitted.quantization {
                H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                    pruned_adaln_table_sha256,
                    ..
                } => pruned_adaln_table_sha256,
                H3FactoryQuantizationAuthority::OfficialBf16 => unreachable!(),
            }
        }

        fn attention_runtime_identity_sha256(&self) -> &str {
            &self.stream.attention_runtime_identity_sha256
        }

        fn attention_kernel_identity(&self) -> &str {
            &self.admitted.attention.qualification_kernel_identity
        }

        fn attention_qualification_sha256(&self) -> &str {
            &self.admitted.attention.qualification_sha256
        }

        fn memory_overlap_identity_sha256(&self) -> &str {
            &self.overlap_identity_sha256
        }
    }

    struct FakeConditioner {
        execution: H3PrivateExecutionProjection<FakeExecution, FakeArtifacts>,
        artifacts: H3PrivateArtifactProjection<FakeExecution, FakeArtifacts>,
        model: String,
        task: Task,
        released: bool,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Drop for FakeConditioner {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("qwen".into());
        }
    }

    impl H3PrivateFl2VaConditioner<FakeExecution, FakeArtifacts> for FakeConditioner {
        fn model(&self) -> &str {
            &self.model
        }

        fn task(&self) -> Task {
            self.task
        }

        fn execution_projection(
            &self,
        ) -> &H3PrivateExecutionProjection<FakeExecution, FakeArtifacts> {
            &self.execution
        }

        fn artifact_projection(
            &self,
        ) -> &H3PrivateArtifactProjection<FakeExecution, FakeArtifacts> {
            &self.artifacts
        }

        fn encode_fl2va(
            &mut self,
            _prompt: &str,
            endpoints: &[H3PreparedEndpoint],
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TextConditioning> {
            if self.released {
                bail!("synthetic Qwen is one-shot");
            }
            self.released = true;
            self.events
                .lock()
                .unwrap()
                .push(format!("qwen-release:{}", endpoints.len()));
            Ok(H3TextConditioning {
                states: Tensor::zeros((1, 1, 5_120), DType::F32, &Device::Cpu)?,
                tags: vec![H3ModalityTag::Text],
                lifetime_probe: None,
            })
        }

        fn validate_continuing_authority(&self) -> Result<()> {
            if !self.released {
                bail!("synthetic Qwen was not released");
            }
            if !H3BackendExecutionLease::is_active(&self.execution) || !self.artifacts.is_active() {
                bail!("synthetic Qwen authority was revoked");
            }
            Ok(())
        }
    }

    struct FakeBlock(usize);

    struct FakeBlockLoader;

    impl H3BlockLoader for FakeBlockLoader {
        type Block = FakeBlock;

        fn load_block(
            &mut self,
            index: usize,
            device_id: &str,
            execution_fingerprint: &str,
        ) -> Result<Self::Block> {
            assert_eq!(device_id, "test-cpu");
            assert_eq!(execution_fingerprint, EXECUTION);
            Ok(FakeBlock(index))
        }
    }

    struct FakeExecutor {
        events: Arc<Mutex<Vec<String>>>,
        forwarded: Arc<Mutex<Vec<usize>>>,
        aborted: Arc<AtomicUsize>,
        input: Option<(Tensor, Tensor)>,
    }

    impl Drop for FakeExecutor {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("denoiser".into());
        }
    }

    impl H3StreamedTransformerExecutor<FakeBlock> for FakeExecutor {
        fn begin_step(
            &mut self,
            input: H3ForwardInput<'_>,
            _layout: &H3FrozenPackedLayout,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            assert!(self
                .events
                .lock()
                .unwrap()
                .iter()
                .any(|event| event.starts_with("qwen-release:")));
            self.input = Some((input.video_rows.clone(), input.audio_rows.clone()));
            Ok(())
        }

        fn forward_block(
            &mut self,
            index: usize,
            block: &FakeBlock,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            assert_eq!(index, block.0);
            self.forwarded.lock().unwrap().push(index);
            self.events.lock().unwrap().push(format!("block-{index}"));
            Ok(())
        }

        fn finish_step(
            &mut self,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TransformerOutput> {
            let (video, audio) = self.input.take().unwrap();
            Ok(H3TransformerOutput { video, audio })
        }

        fn abort_step(&mut self) {
            self.input = None;
            self.aborted.fetch_add(1, Ordering::SeqCst);
        }
    }

    type TestDenoiser = H3BlockStreamedDenoiser<
        FakeBlockLoader,
        H3PrivateExecutionProjection<FakeExecution, FakeArtifacts>,
        FakeExecutor,
    >;
    type TestCore =
        H3PrivateVaeFreeStreamedCore<FakeConditioner, TestDenoiser, FakeExecution, FakeArtifacts>;

    struct Fixture {
        authority: FrozenH3FactoryAuthority,
        core: Option<TestCore>,
        events: Arc<Mutex<Vec<String>>>,
        forwarded: Arc<Mutex<Vec<usize>>>,
        aborted: Arc<AtomicUsize>,
        artifact_active: Arc<AtomicBool>,
    }

    fn fixture() -> Result<Fixture> {
        fixture_with_condition_visual_rows(2)
    }

    fn fixture_with_condition_visual_rows(condition_visual_rows: u64) -> Result<Fixture> {
        fixture_with_mismatch_and_condition_visual_rows(None, condition_visual_rows)
    }

    #[derive(Clone, Copy, Debug)]
    enum AuthorityMismatch {
        Factory,
        BackendPlan,
        ComponentSet,
        DeviceExecution,
        TaskModel,
        ConditionShape,
        TransformerComponent,
        TransformerTask,
        TransformerContent,
        TransformerLayout,
        TransformerCheckpoint,
        TransformerPolicy,
        AttentionRuntime,
        AttentionQualification,
        VaePlan,
        MemoryOverlap,
        InactiveArtifact,
    }

    fn fixture_with_mismatch(mismatch: Option<AuthorityMismatch>) -> Result<Fixture> {
        fixture_with_mismatch_and_condition_visual_rows(mismatch, 2)
    }

    fn fixture_with_mismatch_and_condition_visual_rows(
        mismatch: Option<AuthorityMismatch>,
        condition_visual_rows: u64,
    ) -> Result<Fixture> {
        let authority = authority_with_condition_visual_rows(condition_visual_rows);
        let admitted = authority.private_fl2va_runtime_authority_for_schema_tests()?;
        let mut overlap = overlap(&admitted);
        let stream = stream_authority(&admitted);
        let mut artifact_admitted = admitted.clone();
        let mut artifact_stream = stream.clone();
        let mut artifact_overlap_identity = overlap.identity_sha256().to_owned();
        let mut execution_device_id = admitted.device_id.clone();
        let mut conditioner_model = admitted.canonical_model.clone();
        let mut conditioner_task = admitted.task;
        let events = Arc::new(Mutex::new(Vec::new()));
        let execution_active = Arc::new(AtomicBool::new(true));
        let artifact_active = Arc::new(AtomicBool::new(true));
        match mismatch {
            Some(AuthorityMismatch::Factory) => {
                artifact_admitted.factory_identity_sha256 = sha('f');
            }
            Some(AuthorityMismatch::BackendPlan) => {
                artifact_admitted.backend_plan_identity_sha256 = sha('f');
            }
            Some(AuthorityMismatch::ComponentSet) => {
                artifact_admitted.component_set_identity_sha256 = sha('f');
            }
            Some(AuthorityMismatch::DeviceExecution) => {
                execution_device_id = "wrong-device".into();
            }
            Some(AuthorityMismatch::TaskModel) => {
                conditioner_model = contract::REF2VA_COMFY.into();
                conditioner_task = Task::Ref2va;
            }
            Some(AuthorityMismatch::ConditionShape) => {
                overlap = H3PrivateFl2VaMemoryOverlapAuthority::new(
                    admitted.factory_identity_sha256.clone(),
                    sha('a'),
                    sha('b'),
                    0,
                    0,
                    0,
                    30,
                    40,
                    50,
                    90,
                    120,
                    0,
                )?;
                artifact_overlap_identity = overlap.identity_sha256().into();
            }
            Some(AuthorityMismatch::TransformerComponent) => {
                artifact_admitted.transformer_component_content_sha256 = sha('f');
            }
            Some(AuthorityMismatch::TransformerTask) => {
                artifact_stream.task = H3TransformerTask::Ref2Va;
            }
            Some(AuthorityMismatch::TransformerContent) => {
                artifact_stream.transformer_content_sha256 = sha('f');
            }
            Some(AuthorityMismatch::TransformerLayout) => {
                artifact_stream.transformer_layout_identity_sha256 = sha('f');
            }
            Some(AuthorityMismatch::TransformerCheckpoint) => {
                artifact_stream.checkpoint_identity_sha256 = sha('f');
            }
            Some(AuthorityMismatch::TransformerPolicy) => {
                artifact_stream.transformer_policy_identity_sha256 = sha('f');
            }
            Some(AuthorityMismatch::AttentionRuntime) => {
                artifact_stream.attention_runtime_identity_sha256 = sha('f');
            }
            Some(AuthorityMismatch::AttentionQualification) => {
                artifact_admitted.attention.qualification_sha256 = sha('f');
            }
            Some(AuthorityMismatch::VaePlan) => {
                artifact_admitted.vae_artifact_plan_identity_sha256 = sha('f');
            }
            Some(AuthorityMismatch::MemoryOverlap) => {
                artifact_overlap_identity = sha('f');
            }
            Some(AuthorityMismatch::InactiveArtifact) => {
                artifact_active.store(false, Ordering::SeqCst);
            }
            None => {}
        }
        let attempt = H3PrivateAttemptAuthority::new_unchecked_for_test(
            FakeExecution {
                device: Device::Cpu,
                device_id: execution_device_id,
                execution_fingerprint: admitted.execution_fingerprint.clone(),
                compute_capability: admitted.compute_capability,
                active: execution_active,
                events: Arc::clone(&events),
            },
            FakeArtifacts {
                admitted: artifact_admitted,
                stream: artifact_stream,
                overlap_identity_sha256: artifact_overlap_identity,
                active: Arc::clone(&artifact_active),
                events: Arc::clone(&events),
            },
            Device::Cpu,
        );
        let (qwen_execution, qwen_artifacts) = attempt.qwen_projections();
        let conditioner = FakeConditioner {
            execution: qwen_execution,
            artifacts: qwen_artifacts,
            model: conditioner_model,
            task: conditioner_task,
            released: false,
            events: Arc::clone(&events),
        };
        let forwarded = Arc::new(Mutex::new(Vec::new()));
        let aborted = Arc::new(AtomicUsize::new(0));
        let identity = H3PipelineBackendIdentity {
            kind: H3PipelineBackendKind::SyntheticCpu,
            device_id: admitted.device_id.clone(),
            execution_fingerprint: admitted.execution_fingerprint.clone(),
        };
        let denoiser = H3BlockStreamedDenoiser::new(
            identity,
            admitted.block_streaming.clone(),
            attempt.block_projection(),
            FakeBlockLoader,
            FakeExecutor {
                events: Arc::clone(&events),
                forwarded: Arc::clone(&forwarded),
                aborted: Arc::clone(&aborted),
                input: None,
            },
        )?;
        let cancellation_slot = H3PrivateComfyCancellationSlot::default();
        let cancellation_guard = cancellation_slot.install(&ProgressReporter::default())?;
        let core = H3PrivateVaeFreeStreamedCore::new(
            conditioner,
            denoiser,
            cancellation_guard,
            attempt.block_projection(),
            attempt.artifact_projection(),
            admitted,
            stream,
            overlap,
            attempt,
        )?;
        Ok(Fixture {
            authority,
            core: Some(core),
            events,
            forwarded,
            aborted,
            artifact_active,
        })
    }

    fn frozen_layout() -> H3FrozenPackedLayout {
        H3PackedLayout::new(
            vec![[0.0; 3]; 3],
            vec![0, 1, 2],
            vec![
                H3Modality::Video as u32,
                H3Modality::Audio as u32,
                H3Modality::Text as u32,
            ],
            vec![0],
            vec![1],
            vec![2],
        )
        .unwrap()
        .freeze(&Device::Cpu)
        .unwrap()
    }

    fn forward_inputs() -> (Tensor, Tensor, Tensor, Tensor) {
        (
            Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((1, 1, 3), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap(),
        )
    }

    fn endpoint(anchor: H3EndpointAnchor) -> H3PreparedEndpoint {
        H3PreparedEndpoint {
            anchor,
            source_width: 1,
            source_height: 1,
            resize: H3EndpointResize::Identity,
            pixels: Tensor::zeros((1, 3, 1, 1, 1), DType::U8, &Device::Cpu).unwrap(),
        }
    }

    #[derive(Default)]
    struct RecordingCheckpoint {
        events: Vec<H3PipelineEvent>,
        cancel_at_block_progress: Option<usize>,
    }

    impl H3PipelineCheckpoint for RecordingCheckpoint {
        fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()> {
            self.events.push(event);
            if event.phase == H3PipelinePhase::TransformerBlock
                && self.cancel_at_block_progress == Some(event.completed)
            {
                return Err(InferenceCancelled.into());
            }
            Ok(())
        }
    }

    struct FakeVaeRuntime {
        artifact_plan_identity_sha256: String,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Drop for FakeVaeRuntime {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("vae".into());
        }
    }

    impl H3PrivateVaeRuntime for FakeVaeRuntime {
        fn task(&self) -> Task {
            Task::Fl2va
        }

        fn canonical_model(&self) -> &str {
            contract::FL2VA_COMFY
        }

        fn plan_identity_sha256(&self) -> &str {
            "1111111111111111111111111111111111111111111111111111111111111111"
        }

        fn artifact_plan_identity_sha256(&self) -> &str {
            &self.artifact_plan_identity_sha256
        }

        fn authority_identity_sha256(&self) -> &str {
            "2222222222222222222222222222222222222222222222222222222222222222"
        }

        fn device(&self) -> &Device {
            &Device::Cpu
        }

        fn validate_authority(&self) -> Result<()> {
            Ok(())
        }

        fn encode_visual_condition(
            &self,
            _endpoint: &H3PreparedEndpoint,
            _mode: ConditionEncodeMode,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<Tensor> {
            self.events.lock().unwrap().push("vae-visual".into());
            Ok(Tensor::zeros(1, DType::F32, &Device::Cpu)?)
        }

        fn decode_video(
            &self,
            _latents: &Tensor,
            _sink: &mut H3VideoEncodeSink,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            self.events.lock().unwrap().push("vae-video".into());
            bail!("synthetic video stop")
        }

        fn decode_audio(
            &self,
            _latents: &StereoLatents,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<StereoWaveform> {
            self.events.lock().unwrap().push("vae-audio".into());
            bail!("synthetic audio stop")
        }
    }

    #[test]
    fn t2va_and_fl2va_release_qwen_before_all_fifty_streamed_blocks() {
        for (condition_visual_rows, endpoints) in [
            (0, Vec::new()),
            (
                2,
                vec![
                    endpoint(H3EndpointAnchor::First),
                    endpoint(H3EndpointAnchor::Last),
                ],
            ),
        ] {
            let mut fixture = fixture_with_condition_visual_rows(condition_visual_rows).unwrap();
            let mut core = fixture.core.take().unwrap();
            let mut checkpoint = RecordingCheckpoint::default();
            let text = core
                .encode_text("synthetic prompt", &endpoints, &mut checkpoint)
                .unwrap();
            let (video, audio, _, timesteps) = forward_inputs();
            let output = core
                .denoise(
                    H3ForwardInput {
                        video_rows: &video,
                        audio_rows: &audio,
                        text_states: &text.states,
                        timesteps: &timesteps,
                    },
                    &frozen_layout(),
                    &mut checkpoint,
                )
                .unwrap();
            assert_eq!(output.video.dims(), video.dims());
            assert_eq!(output.audio.dims(), audio.dims());
            assert_eq!(
                *fixture.forwarded.lock().unwrap(),
                (0..50).collect::<Vec<_>>()
            );
            assert_eq!(fixture.aborted.load(Ordering::SeqCst), 0);
            let events = fixture.events.lock().unwrap();
            let release = format!("qwen-release:{}", endpoints.len());
            let release_index = events.iter().position(|event| event == &release).unwrap();
            let block_zero_index = events.iter().position(|event| event == "block-0").unwrap();
            assert!(release_index < block_zero_index);
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event.starts_with("block-"))
                    .count(),
                50
            );
        }
    }

    #[test]
    fn block_twenty_five_cancellation_is_typed_aborts_and_produces_no_output() {
        let mut fixture = fixture().unwrap();
        let mut core = fixture.core.take().unwrap();
        let mut checkpoint = RecordingCheckpoint {
            cancel_at_block_progress: Some(25),
            ..RecordingCheckpoint::default()
        };
        let text = core
            .encode_text("synthetic cancellation", &[], &mut checkpoint)
            .unwrap();
        let (video, audio, _, timesteps) = forward_inputs();
        let error = core
            .denoise(
                H3ForwardInput {
                    video_rows: &video,
                    audio_rows: &audio,
                    text_states: &text.states,
                    timesteps: &timesteps,
                },
                &frozen_layout(),
                &mut checkpoint,
            )
            .unwrap_err();
        assert!(is_inference_cancelled(&error));
        assert_eq!(
            *fixture.forwarded.lock().unwrap(),
            (0..24).collect::<Vec<_>>()
        );
        assert_eq!(fixture.aborted.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn revoked_full_artifact_lease_fails_before_the_first_block() {
        let mut fixture = fixture().unwrap();
        let mut core = fixture.core.take().unwrap();
        let mut checkpoint = RecordingCheckpoint::default();
        let text = core
            .encode_text("synthetic revocation", &[], &mut checkpoint)
            .unwrap();
        fixture.artifact_active.store(false, Ordering::SeqCst);
        let (video, audio, _, timesteps) = forward_inputs();
        let error = core
            .denoise(
                H3ForwardInput {
                    video_rows: &video,
                    audio_rows: &audio,
                    text_states: &text.states,
                    timesteps: &timesteps,
                },
                &frozen_layout(),
                &mut checkpoint,
            )
            .unwrap_err();
        assert!(error.to_string().contains("frozen authority"));
        assert!(fixture.forwarded.lock().unwrap().is_empty());
    }

    #[test]
    fn one_shot_core_rejects_a_second_job_instead_of_reusing_released_qwen() {
        let mut fixture = fixture().unwrap();
        let mut core = fixture.core.take().unwrap();
        let mut checkpoint = RecordingCheckpoint::default();
        core.encode_text("first attempt", &[], &mut checkpoint)
            .unwrap();
        let error = core
            .encode_text("cached second attempt", &[], &mut checkpoint)
            .unwrap_err();
        assert!(error.to_string().contains("one-shot"));
        assert!(fixture.forwarded.lock().unwrap().is_empty());
    }

    #[test]
    fn conservative_overlap_rejects_condition_vae_and_audio_latent_undercharge() {
        let admitted = authority()
            .private_fl2va_runtime_authority_for_schema_tests()
            .unwrap();
        let error = H3PrivateFl2VaMemoryOverlapAuthority::new(
            admitted.factory_identity_sha256.clone(),
            sha('a'),
            sha('b'),
            admitted.condition_visual_rows,
            0,
            20,
            30,
            40,
            50,
            90,
            120,
            60,
        )
        .unwrap_err();
        assert!(error.to_string().contains("undercharges condition backing"));

        let error = H3PrivateFl2VaMemoryOverlapAuthority::new(
            admitted.factory_identity_sha256.clone(),
            sha('a'),
            sha('b'),
            admitted.condition_visual_rows,
            10,
            20,
            30,
            40,
            50,
            89,
            120,
            60,
        )
        .unwrap_err();
        assert!(error.to_string().contains("retain both VAEs"));

        let error = H3PrivateFl2VaMemoryOverlapAuthority::new(
            admitted.factory_identity_sha256,
            sha('a'),
            sha('b'),
            admitted.condition_visual_rows,
            10,
            20,
            30,
            40,
            50,
            90,
            119,
            60,
        )
        .unwrap_err();
        assert!(error.to_string().contains("undercharges retained audio"));

        let t2va = authority_with_condition_visual_rows(0)
            .private_fl2va_runtime_authority_for_schema_tests()
            .unwrap();
        H3PrivateFl2VaMemoryOverlapAuthority::new(
            t2va.factory_identity_sha256.clone(),
            sha('a'),
            sha('b'),
            0,
            0,
            0,
            30,
            40,
            50,
            90,
            120,
            0,
        )
        .unwrap();
        let error = H3PrivateFl2VaMemoryOverlapAuthority::new(
            t2va.factory_identity_sha256,
            sha('a'),
            sha('b'),
            0,
            1,
            0,
            30,
            40,
            50,
            90,
            120,
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("invents condition backing"));
    }

    #[test]
    fn raw_checkpoint_and_logical_transformer_identity_domains_remain_distinct() {
        let admitted = authority()
            .private_fl2va_runtime_authority_for_schema_tests()
            .unwrap();
        let stream = stream_authority(&admitted);
        assert_ne!(
            stream.transformer_content_sha256,
            admitted.transformer_component_content_sha256
        );
        assert!(valid_sha256(&stream.checkpoint_identity_sha256));
    }

    fn qwen_artifact_authority_from_facts(
        facts: &H3PrivateArtifactAuthorityFacts,
    ) -> H3PrivateQwenArtifactAuthority {
        H3PrivateQwenArtifactAuthority {
            support_identity_sha256: facts.support_identity_sha256.clone(),
            weight_identity_sha256: facts.weight_identity_sha256.clone(),
            weight_header_identity_sha256: facts.weight_header_identity_sha256.clone(),
            weight_policy_identity_sha256: facts.weight_policy_identity_sha256.clone(),
        }
    }

    fn exact_continuing_artifact_facts() -> (
        H3PrivateFl2VaFactoryAuthority,
        H3PrivateComfyStreamAuthority,
        H3PrivateQwenArtifactAuthority,
        H3PrivateFl2VaMemoryOverlapAuthority,
        H3PrivateArtifactAuthorityFacts,
    ) {
        let admitted = authority()
            .private_fl2va_runtime_authority_for_schema_tests()
            .unwrap();
        let stream = stream_authority(&admitted);
        let overlap = overlap(&admitted);
        let artifacts = FakeArtifacts {
            admitted: admitted.clone(),
            stream: stream.clone(),
            overlap_identity_sha256: overlap.identity_sha256().into(),
            active: Arc::new(AtomicBool::new(true)),
            events: Arc::new(Mutex::new(Vec::new())),
        };
        let facts = H3PrivateArtifactAuthorityFacts::capture(&artifacts);
        let qwen = qwen_artifact_authority_from_facts(&facts);
        (admitted, stream, qwen, overlap, facts)
    }

    #[test]
    fn every_continuing_artifact_and_bound_stream_axis_fails_closed() {
        let (admitted, stream, qwen, overlap, exact) = exact_continuing_artifact_facts();
        validate_private_artifact_facts(&admitted, &stream, &qwen, &overlap, &exact).unwrap();

        for axis in [
            "active",
            "factory",
            "backend-plan",
            "component-set",
            "conditioner-content",
            "conditioner-validation",
            "qwen-support",
            "qwen-weight",
            "qwen-header",
            "qwen-policy",
            "transformer-component-content",
            "transformer-component-validation",
            "visual-vae-content",
            "visual-vae-validation",
            "audio-vae-content",
            "audio-vae-validation",
            "vae-plan",
            "transformer-task",
            "checkpoint-content",
            "checkpoint-layout",
            "opened-checkpoint",
            "transformer-policy",
            "pruned-adaln",
            "attention-runtime",
            "attention-kernel",
            "attention-qualification",
            "memory-overlap",
        ] {
            let mut facts = exact.clone();
            match axis {
                "active" => facts.active = false,
                "factory" => facts.factory_identity_sha256 = sha('f'),
                "backend-plan" => facts.backend_plan_identity_sha256 = sha('f'),
                "component-set" => facts.component_set_identity_sha256 = sha('f'),
                "conditioner-content" => facts.conditioner_component_content_sha256 = sha('f'),
                "conditioner-validation" => {
                    facts.conditioner_component_validation_sha256 = sha('f')
                }
                "qwen-support" => facts.support_identity_sha256 = sha('f'),
                "qwen-weight" => facts.weight_identity_sha256 = sha('f'),
                "qwen-header" => facts.weight_header_identity_sha256 = sha('f'),
                "qwen-policy" => facts.weight_policy_identity_sha256 = sha('f'),
                "transformer-component-content" => {
                    facts.transformer_component_content_sha256 = sha('f')
                }
                "transformer-component-validation" => {
                    facts.transformer_component_validation_sha256 = sha('f')
                }
                "visual-vae-content" => facts.visual_vae_component_content_sha256 = sha('f'),
                "visual-vae-validation" => facts.visual_vae_component_validation_sha256 = sha('f'),
                "audio-vae-content" => facts.audio_vae_component_content_sha256 = sha('f'),
                "audio-vae-validation" => facts.audio_vae_component_validation_sha256 = sha('f'),
                "vae-plan" => facts.vae_artifact_plan_identity_sha256 = sha('f'),
                "transformer-task" => facts.transformer_task = H3TransformerTask::Ref2Va,
                "checkpoint-content" => facts.transformer_checkpoint_content_sha256 = sha('f'),
                "checkpoint-layout" => {
                    facts.transformer_checkpoint_layout_identity_sha256 = sha('f')
                }
                "opened-checkpoint" => facts.transformer_checkpoint_identity_sha256 = sha('f'),
                "transformer-policy" => facts.transformer_policy_identity_sha256 = sha('f'),
                "pruned-adaln" => facts.pruned_adaln_table_identity_sha256 = sha('f'),
                "attention-runtime" => facts.attention_runtime_identity_sha256 = sha('f'),
                "attention-kernel" => facts.attention_kernel_identity = "other-kernel".into(),
                "attention-qualification" => facts.attention_qualification_sha256 = sha('f'),
                "memory-overlap" => facts.memory_overlap_identity_sha256 = sha('f'),
                _ => unreachable!(),
            }
            assert!(
                validate_private_artifact_facts(&admitted, &stream, &qwen, &overlap, &facts)
                    .is_err(),
                "{axis}"
            );
        }

        for axis in ["support", "weight", "header", "policy"] {
            let mut changed = qwen.clone();
            match axis {
                "support" => changed.support_identity_sha256 = sha('f'),
                "weight" => changed.weight_identity_sha256 = sha('f'),
                "header" => changed.weight_header_identity_sha256 = sha('f'),
                "policy" => changed.weight_policy_identity_sha256 = sha('f'),
                _ => unreachable!(),
            }
            assert!(
                validate_private_artifact_facts(&admitted, &stream, &changed, &overlap, &exact)
                    .is_err(),
                "retained Qwen {axis}"
            );
        }

        for axis in [
            "task",
            "checkpoint-content",
            "checkpoint-layout",
            "opened-checkpoint",
            "transformer-policy",
            "attention-runtime",
        ] {
            let mut changed = stream.clone();
            match axis {
                "task" => changed.task = H3TransformerTask::Ref2Va,
                "checkpoint-content" => changed.transformer_content_sha256 = sha('f'),
                "checkpoint-layout" => changed.transformer_layout_identity_sha256 = sha('f'),
                "opened-checkpoint" => changed.checkpoint_identity_sha256 = sha('f'),
                "transformer-policy" => changed.transformer_policy_identity_sha256 = sha('f'),
                "attention-runtime" => changed.attention_runtime_identity_sha256 = sha('f'),
                _ => unreachable!(),
            }
            assert!(
                validate_private_artifact_facts(&admitted, &changed, &qwen, &overlap, &exact)
                    .is_err(),
                "stream {axis}"
            );
        }

        let mut changed = admitted.clone();
        changed.task = Task::Ref2va;
        assert!(
            validate_private_artifact_facts(&changed, &stream, &qwen, &overlap, &exact).is_err()
        );
        let mut changed = admitted;
        changed.canonical_model = contract::REF2VA_COMFY.into();
        assert!(
            validate_private_artifact_facts(&changed, &stream, &qwen, &overlap, &exact).is_err()
        );
    }

    #[test]
    fn overlap_identity_uses_the_version_two_serializer_domain() {
        assert_eq!(
            H3_PRIVATE_FL2VA_OVERLAP_IDENTITY_DOMAIN,
            b"mold.minimax-h3.private-fl2va-overlap.v3\0"
        );
    }

    struct MutableEchoExecution {
        device: Device,
        changed: Arc<AtomicBool>,
        admitted_device_id: String,
        changed_device_id: String,
        execution_fingerprint: String,
        compute_capability: Option<(u16, u16)>,
    }

    impl H3BackendExecutionLease for MutableEchoExecution {
        fn lease_id(&self) -> &str {
            "mutable-echo-lease"
        }

        fn device_id(&self) -> &str {
            if self.changed.load(Ordering::SeqCst) {
                &self.changed_device_id
            } else {
                &self.admitted_device_id
            }
        }

        fn backend(&self) -> H3CandleBackendDevice {
            H3CandleBackendDevice::from_compute_capability(self.compute_capability)
        }

        fn execution_fingerprint(&self) -> &str {
            &self.execution_fingerprint
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn is_active(&self) -> bool {
            true
        }
    }

    #[test]
    fn terminal_live_validation_rejects_mutated_echo_axis_before_publication() {
        let admitted = authority()
            .private_fl2va_runtime_authority_for_schema_tests()
            .unwrap();
        let stream = stream_authority(&admitted);
        let overlap = overlap(&admitted);
        let changed = Arc::new(AtomicBool::new(false));
        let artifact_lease = FakeArtifacts {
            admitted: admitted.clone(),
            stream: stream.clone(),
            overlap_identity_sha256: overlap.identity_sha256().into(),
            active: Arc::new(AtomicBool::new(true)),
            events: Arc::new(Mutex::new(Vec::new())),
        };
        let qwen = qwen_artifact_authority_from_facts(&H3PrivateArtifactAuthorityFacts::capture(
            &artifact_lease,
        ));
        let attempt = H3PrivateAttemptAuthority::new_unchecked_for_test(
            MutableEchoExecution {
                device: Device::Cpu,
                changed: Arc::clone(&changed),
                admitted_device_id: admitted.device_id.clone(),
                changed_device_id: "mutated-device".into(),
                execution_fingerprint: admitted.execution_fingerprint.clone(),
                compute_capability: admitted.compute_capability,
            },
            artifact_lease,
            Device::Cpu,
        );
        let execution = attempt.block_projection();
        let artifacts = attempt.artifact_projection();
        let initial = validate_private_live_attempt_authority(
            &admitted, &stream, &qwen, &overlap, &execution, &artifacts, &attempt,
        )
        .unwrap();
        assert_eq!(initial.device_id, admitted.device_id);
        assert_eq!(
            initial.component_set_identity_sha256,
            admitted.component_set_identity_sha256
        );

        changed.store(true, Ordering::SeqCst);
        let publications = AtomicUsize::new(0);
        let result = validate_private_live_attempt_authority(
            &admitted, &stream, &qwen, &overlap, &execution, &artifacts, &attempt,
        )
        .inspect(|_| {
            publications.fetch_add(1, Ordering::SeqCst);
        });
        assert!(result.unwrap_err().to_string().contains("route"));
        assert_eq!(publications.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn every_frozen_runtime_authority_axis_fails_closed_on_mismatch() {
        for mismatch in [
            AuthorityMismatch::Factory,
            AuthorityMismatch::BackendPlan,
            AuthorityMismatch::ComponentSet,
            AuthorityMismatch::DeviceExecution,
            AuthorityMismatch::TaskModel,
            AuthorityMismatch::ConditionShape,
            AuthorityMismatch::TransformerComponent,
            AuthorityMismatch::TransformerTask,
            AuthorityMismatch::TransformerContent,
            AuthorityMismatch::TransformerLayout,
            AuthorityMismatch::TransformerCheckpoint,
            AuthorityMismatch::TransformerPolicy,
            AuthorityMismatch::AttentionRuntime,
            AuthorityMismatch::AttentionQualification,
            AuthorityMismatch::VaePlan,
            AuthorityMismatch::MemoryOverlap,
            AuthorityMismatch::InactiveArtifact,
        ] {
            let error = fixture_with_mismatch(Some(mismatch))
                .err()
                .unwrap_or_else(|| panic!("{mismatch:?} unexpectedly composed"));
            assert!(!error.to_string().trim().is_empty(), "{mismatch:?}");
        }
    }

    #[test]
    fn projections_from_different_attempt_owners_never_compare_equal() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let first = H3PrivateAttemptAuthority::new_unchecked_for_test(
            DropCount {
                name: "execution",
                events: Arc::clone(&events),
            },
            DropCount {
                name: "artifacts",
                events: Arc::clone(&events),
            },
            Device::Cpu,
        );
        let second = H3PrivateAttemptAuthority::new_unchecked_for_test(
            DropCount {
                name: "execution",
                events: Arc::clone(&events),
            },
            DropCount {
                name: "artifacts",
                events,
            },
            Device::Cpu,
        );
        let (execution, artifacts) = first.qwen_projections();
        assert!(execution.belongs_to(&first));
        assert!(artifacts.belongs_to(&first));
        assert!(!execution.belongs_to(&second));
        assert!(!artifacts.belongs_to(&second));
    }

    #[test]
    fn outer_vae_is_the_streamed_backend_and_drops_before_the_singular_attempt() {
        fn assert_streamed<T: H3StreamedFl2VaBackend>() {}

        let mut fixture = fixture().unwrap();
        let core = fixture.core.take().unwrap();
        let vae = FakeVaeRuntime {
            artifact_plan_identity_sha256: fixture
                .authority
                .private_vae_adapter_authority()
                .unwrap()
                .vae_artifact_plan_identity_sha256,
            events: Arc::clone(&fixture.events),
        };
        assert_streamed::<H3PrivateComfyVaeAdapter<TestCore, FakeVaeRuntime>>();
        let mut adapter =
            H3PrivateComfyVaeAdapter::new_with_runtime(core, vae, &fixture.authority).unwrap();
        adapter
            .encode_visual_condition(
                &endpoint(H3EndpointAnchor::First),
                ConditionEncodeMode::OfficialFreshSeed42,
                &mut RecordingCheckpoint::default(),
            )
            .unwrap();
        assert_eq!(fixture.events.lock().unwrap().as_slice(), ["vae-visual"]);
        fixture.events.lock().unwrap().clear();
        drop(adapter);
        assert_eq!(
            fixture.events.lock().unwrap().as_slice(),
            ["vae", "qwen", "denoiser", "execution", "artifacts"]
        );
    }

    #[derive(Clone)]
    struct DropCount {
        name: &'static str,
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Drop for DropCount {
        fn drop(&mut self) {
            self.events.lock().unwrap().push(self.name);
        }
    }

    #[test]
    fn phase_ledger_enforces_exact_load_drop_order_and_terminal_mux_gate() {
        fn try_mux(ledger: &H3PrivatePhaseLedger, calls: &AtomicUsize) -> Result<()> {
            if !ledger.is_terminal() {
                bail!("synthetic mux requires Empty")
            }
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        let events = Arc::new(Mutex::new(Vec::new()));
        let mux_calls = AtomicUsize::new(0);
        let mut ledger = H3PrivatePhaseLedger::new(3).unwrap();
        assert!(H3PrivatePhaseLedger::new(0).is_err());
        assert!(ledger.qwen_loaded().is_err());
        assert!(ledger.denoise_completed().is_err());
        assert!(ledger.visual_decoded().is_err());
        assert!(ledger.vaes_dropped().is_err());
        assert!(try_mux(&ledger, &mux_calls).is_err());

        let vae = DropCount {
            name: "vae",
            events: Arc::clone(&events),
        };
        ledger.vaes_loaded().unwrap();
        assert!(ledger.vaes_loaded().is_err());
        assert!(try_mux(&ledger, &mux_calls).is_err());

        let qwen = DropCount {
            name: "qwen",
            events: Arc::clone(&events),
        };
        ledger.qwen_loaded().unwrap();
        assert!(ledger.transformer_loaded().is_err());
        drop(qwen);
        ledger.qwen_dropped().unwrap();
        assert_eq!(*events.lock().unwrap(), ["qwen"]);
        ledger.conditions_encoded().unwrap();
        ledger.conditions_encoded().unwrap();

        // The transformer may not load while either VAE is still resident.
        assert!(ledger.transformer_loaded().is_err());
        drop(vae);
        ledger.vaes_parked().unwrap();
        assert_eq!(*events.lock().unwrap(), ["qwen", "vae"]);
        assert!(ledger.vaes_parked().is_err());
        assert!(ledger.conditions_encoded().is_err());
        assert!(ledger.vaes_reloaded().is_err());
        assert!(try_mux(&ledger, &mux_calls).is_err());

        let transformer = DropCount {
            name: "transformer",
            events: Arc::clone(&events),
        };
        ledger.transformer_loaded().unwrap();
        assert!(!ledger.denoise_completed().unwrap());
        assert!(!ledger.denoise_completed().unwrap());
        // Denoise runs with neither VAE resident: only "qwen" and "vae" have
        // been released, and no reload has happened yet.
        assert_eq!(*events.lock().unwrap(), ["qwen", "vae"]);
        assert!(ledger.vaes_reloaded().is_err());
        assert!(ledger.visual_decoded().is_err());
        assert!(try_mux(&ledger, &mux_calls).is_err());

        assert!(ledger.denoise_completed().unwrap());
        drop(transformer);
        assert_eq!(*events.lock().unwrap(), ["qwen", "vae", "transformer"]);
        assert!(ledger.denoise_completed().is_err());

        // Visual decode requires the reconstructed pair, never the parked one.
        assert!(ledger.visual_decoded().is_err());
        let reloaded_vae = DropCount {
            name: "reloaded-vae",
            events: Arc::clone(&events),
        };
        ledger.vaes_reloaded().unwrap();
        assert!(ledger.vaes_reloaded().is_err());
        ledger.visual_decoded().unwrap();
        assert!(try_mux(&ledger, &mux_calls).is_err());

        drop(reloaded_vae);
        assert_eq!(
            *events.lock().unwrap(),
            ["qwen", "vae", "transformer", "reloaded-vae"]
        );
        ledger.vaes_dropped().unwrap();
        assert!(ledger.is_terminal());
        assert!(ledger.vaes_dropped().is_err());
        try_mux(&ledger, &mux_calls).unwrap();
        assert_eq!(mux_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn vae_load_revocation_cleans_attempt_scope_and_never_reaches_mux() {
        struct RevokeAfterFirstCheckpoint {
            active: Arc<AtomicBool>,
            events: usize,
        }

        impl H3PipelineCheckpoint for RevokeAfterFirstCheckpoint {
            fn checkpoint(&mut self, _event: H3PipelineEvent) -> Result<()> {
                self.events += 1;
                if self.events == 1 {
                    self.active.store(false, Ordering::SeqCst);
                }
                Ok(())
            }
        }

        let active = Arc::new(AtomicBool::new(true));
        let validator_active = Arc::clone(&active);
        let drops = Arc::new(Mutex::new(Vec::new()));
        let mux_calls = AtomicUsize::new(0);
        let result = (|| -> Result<()> {
            let _attempt_scope = DropCount {
                name: "attempt",
                events: Arc::clone(&drops),
            };
            let mut checkpoint = RevokeAfterFirstCheckpoint { active, events: 0 };
            let mut allocation_commit = H3PrivateAllocationCommit::new(|| Ok(()));
            allocation_commit.commit_once()?;
            let mut observer = H3PrivateVaeLoadCheckpoint::new(
                &mut checkpoint,
                move || {
                    if !validator_active.load(Ordering::SeqCst) {
                        bail!("synthetic VAE authority revoked")
                    }
                    Ok(())
                },
                &allocation_commit,
            );
            assert!(observer.checkpoint(H3ComfyVaeLoadEvent {
                role: super::super::vae_runtime::H3ComfyVaeArtifactRole::VisualConfig,
                phase: super::super::vae_runtime::H3ComfyVaeLoadPhase::Open,
                completed: 1,
                total: 2,
            }));
            assert!(!observer.checkpoint(H3ComfyVaeLoadEvent {
                role: super::super::vae_runtime::H3ComfyVaeArtifactRole::VisualWeights,
                phase: super::super::vae_runtime::H3ComfyVaeLoadPhase::Stage,
                completed: 2,
                total: 2,
            }));
            observer.finish(Ok(()))?;
            mux_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })();
        assert!(result.unwrap_err().to_string().contains("revoked"));
        assert_eq!(*drops.lock().unwrap(), ["attempt"]);
        assert_eq!(mux_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn vae_observer_requires_the_owner_commit_before_any_load_checkpoint() {
        struct AcceptAll;

        impl H3PipelineCheckpoint for AcceptAll {
            fn checkpoint(&mut self, _event: H3PipelineEvent) -> Result<()> {
                Ok(())
            }
        }

        let commits = Arc::new(AtomicUsize::new(0));
        let captured = Arc::clone(&commits);
        let mut allocation_commit = H3PrivateAllocationCommit::new(move || {
            captured.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        allocation_commit.commit_once().unwrap();
        assert_eq!(commits.load(Ordering::SeqCst), 1);
        let mut checkpoint = AcceptAll;
        let mut observer =
            H3PrivateVaeLoadCheckpoint::new(&mut checkpoint, || Ok(()), &allocation_commit);
        assert!(observer.checkpoint(H3ComfyVaeLoadEvent {
            role: super::super::vae_runtime::H3ComfyVaeArtifactRole::VisualWeights,
            phase: H3ComfyVaeLoadPhase::ValidateHeader,
            completed: 1,
            total: 1,
        }));
        for completed in [0, 1] {
            assert!(observer.checkpoint(H3ComfyVaeLoadEvent {
                role: super::super::vae_runtime::H3ComfyVaeArtifactRole::VisualWeights,
                phase: H3ComfyVaeLoadPhase::Construct,
                completed,
                total: 1,
            }));
        }
        assert_eq!(commits.load(Ordering::SeqCst), 1);
        drop(observer);
        assert!(allocation_commit.is_committed());
    }

    #[test]
    fn vae_observer_rejects_an_uncommitted_owner_without_running_callback() {
        struct AcceptAll;

        impl H3PipelineCheckpoint for AcceptAll {
            fn checkpoint(&mut self, _event: H3PipelineEvent) -> Result<()> {
                Ok(())
            }
        }

        let allocation_commit = H3PrivateAllocationCommit::new(|| {
            bail!("synthetic scheduler ledger rejected allocation")
        });
        let mut checkpoint = AcceptAll;
        let mut observer =
            H3PrivateVaeLoadCheckpoint::new(&mut checkpoint, || Ok(()), &allocation_commit);
        assert!(!observer.checkpoint(H3ComfyVaeLoadEvent {
            role: super::super::vae_runtime::H3ComfyVaeArtifactRole::VisualWeights,
            phase: H3ComfyVaeLoadPhase::Construct,
            completed: 0,
            total: 1,
        }));
        let error = observer.finish(Ok(())).unwrap_err();
        assert!(error
            .to_string()
            .contains("before the owner allocation commitment"));
        assert!(!allocation_commit.is_committed());
    }

    struct ConcreteCandleResources {
        _device: Device,
        _tensor: Tensor,
        retained: Arc<()>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for ConcreteCandleResources {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn concrete_candle_resources(
        drops: Arc<AtomicUsize>,
    ) -> (ConcreteCandleResources, std::sync::Weak<()>) {
        let retained = Arc::new(());
        let weak = Arc::downgrade(&retained);
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        (
            ConcreteCandleResources {
                _device: device,
                _tensor: tensor,
                retained,
                drops,
            },
            weak,
        )
    }

    #[test]
    fn concrete_candle_resources_are_retained_on_fatal_cuda_error() {
        let drops = Arc::new(AtomicUsize::new(0));
        let calls = AtomicUsize::new(0);
        let (resources, retained) = concrete_candle_resources(Arc::clone(&drops));
        let error = with_contained_private_cuda_resources(resources, |_| -> Result<()> {
            calls.fetch_add(1, Ordering::SeqCst);
            bail!("CUDA_ERROR_ILLEGAL_ADDRESS from synthetic concrete operation")
        })
        .unwrap_err();
        assert!(is_fatal_private_cuda_error(&error));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(retained.upgrade().is_some());
    }

    #[test]
    fn concrete_candle_resources_are_retained_when_operation_panics() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (resources, retained) = concrete_candle_resources(Arc::clone(&drops));
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = with_contained_private_cuda_resources(resources, |_| -> Result<()> {
                panic!("synthetic concrete CUDA owner panic")
            });
        }));
        assert!(panic.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(retained.upgrade().is_some());
    }

    #[test]
    fn concrete_candle_resources_drop_once_on_ordinary_error() {
        let drops = Arc::new(AtomicUsize::new(0));
        let calls = AtomicUsize::new(0);
        let (resources, retained) = concrete_candle_resources(Arc::clone(&drops));
        let error = with_contained_private_cuda_resources(resources, |_| -> Result<()> {
            calls.fetch_add(1, Ordering::SeqCst);
            bail!("ordinary private H3 validation error")
        })
        .unwrap_err();
        assert!(!is_fatal_private_cuda_error(&error));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn ref2va_phase_owner_remains_sealed_and_fail_closed() {
        let error = run_private_comfy_ref2va_attempt(H3PrivateRef2VaPhaseOwner {
            _opened_evidence: ref2va_opened_seal::Token,
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("no exact opened/prepared evidence"));
    }

    #[test]
    fn singular_attempt_owner_drops_each_underlying_lease_once() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let execution = DropCount {
            name: "execution",
            events: Arc::clone(&events),
        };
        let artifacts = DropCount {
            name: "artifacts",
            events: Arc::clone(&events),
        };
        let attempt =
            H3PrivateAttemptAuthority::new_unchecked_for_test(execution, artifacts, Device::Cpu);
        let (qwen_execution, qwen_artifacts) = attempt.qwen_projections();
        let block_execution = attempt.block_projection();

        drop(qwen_execution);
        drop(qwen_artifacts);
        drop(block_execution);
        assert!(events.lock().unwrap().is_empty());

        drop(attempt);
        assert_eq!(*events.lock().unwrap(), ["execution", "artifacts"]);
    }

    #[test]
    fn attempt_and_overlap_roots_are_consuming_while_projections_remain_cloneable() {
        trait AmbiguousIfClone<Marker> {
            fn assert_not_implemented() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        struct ImplementsClone;
        impl<T: Clone> AmbiguousIfClone<ImplementsClone> for T {}

        <H3PrivateAttemptAuthority<(), ()> as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3PrivateFl2VaMemoryOverlapAuthority as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3PrivatePhaseRuntimeOwner<(), (), ()> as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3PrivateRef2VaPhaseOwner as AmbiguousIfClone<_>>::assert_not_implemented();

        fn assert_clone<T: Clone>() {}
        assert_clone::<H3PrivateExecutionProjection<(), ()>>();
        assert_clone::<H3PrivateArtifactProjection<(), ()>>();
    }
}
