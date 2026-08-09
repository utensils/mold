//! Singular-owner private MiniMax H3 FL2VA runtime composition.
//!
//! This module is available only to the non-shipping `h3-private-uat`
//! feature. It does not register a model, loader, capability, catalog,
//! download, server, CLI, or product-surface route.
//!
//! The concrete core is attempt-scoped and one-shot. The engine constructs and
//! consumes a fresh session per job; this private composer remains additionally
//! sealed until every artifact, memory, and execution authority is available.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::{Device, DeviceLocation, Tensor};
use mold_candle::minimax_h3::{
    H3AttentionDevice, H3AttentionRuntimeAuthority, H3ComfyOpenedInt8Checkpoint, H3ForwardInput,
    H3FrozenPackedLayout, H3TransformerOutput, H3TransformerTask, StereoLatents, StereoWaveform,
};
use mold_core::minimax_h3::{self as contract, Task};
use sha2::{Digest, Sha256};

use super::backend::{H3BackendArtifactLease, H3BackendExecutionLease, H3CandleBackendDevice};
use super::engine::{H3BlockStreamedDenoiser, H3PipelineRuntime, H3StreamedDenoiser};
use super::offload::H3BlockLease;
use super::pipeline::{
    H3Fl2VaBackend, H3PipelineBackendIdentity, H3PipelineBackendKind, H3PipelineCheckpoint,
    H3PreparedEndpoint, H3TextConditioning, H3VideoEncodeSink,
};
use super::private_qwen::{
    H3PrivateQwenAdapter, H3PrivateQwenArtifactLease, H3PrivateQwenConditionerLease,
};
use super::private_qwen_support::H3PrivateQwenSupport;
use super::private_runtime::{
    bind_private_comfy_stream, load_and_pair_private_comfy_stream,
    H3PrivateComfyBindingExpectation, H3PrivateComfyBlockLoader, H3PrivateComfyCancellationGuard,
    H3PrivateComfyCancellationSlot, H3PrivateComfyCheckpointFacts, H3PrivateComfyStreamAuthority,
    H3PrivateComfyTransformerExecutor,
};
use super::private_vae_adapter::H3PrivateComfyVaeAdapter;
use super::vae_runtime::H3ComfyVaeRuntimeBundle;
use crate::h3_factory::{H3PrivateFl2VaFactoryAuthority, H3PrivateVaeFactoryAuthority};
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
/// This PR intentionally provides no production issuer. Outside unit tests,
/// `_activation` is an uninhabited type, making the otherwise concrete
/// composer unreachable from safe code until a scheduler-owned frozen-memory
/// projection is reviewed and added.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateFl2VaMemoryOverlapAuthority {
    factory_identity_sha256: String,
    condition_visual_rows: u64,
    condition_backing_host_bytes: u64,
    condition_backing_device_bytes: u64,
    target_audio_latent_device_bytes: u64,
    visual_vae_resident_device_bytes: u64,
    audio_vae_resident_device_bytes: u64,
    attempt_resident_vae_device_bytes: u64,
    visual_decode_peak_device_bytes: u64,
    normalized_endpoint_host_bytes: u64,
    identity_sha256: String,
    _activation: admitted_overlap_seal::Token,
}

mod admitted_overlap_seal {
    #[cfg(not(test))]
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum Token {}

    #[cfg(test)]
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct Token;
}

impl H3PrivateFl2VaMemoryOverlapAuthority {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        factory_identity_sha256: impl Into<String>,
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
            condition_visual_rows,
            condition_backing_host_bytes,
            condition_backing_device_bytes,
            target_audio_latent_device_bytes,
            visual_vae_resident_device_bytes,
            audio_vae_resident_device_bytes,
            attempt_resident_vae_device_bytes,
            visual_decode_peak_device_bytes,
            normalized_endpoint_host_bytes,
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

fn memory_overlap_identity(authority: &H3PrivateFl2VaMemoryOverlapAuthority) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.private-fl2va-overlap.v1\0");
    hash.update(authority.factory_identity_sha256.as_bytes());
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
    format!("{:x}", hash.finalize())
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

impl<E, A> Clone for H3PrivateAttemptAuthority<E, A> {
    fn clone(&self) -> Self {
        Self {
            owner: Arc::clone(&self.owner),
        }
    }
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
                != (H3CandleBackendDevice::Cuda {
                    compute_capability: self.admitted.compute_capability,
                })
            || !(execution.device().is_cuda() || cfg!(test) && execution.device().is_cpu())
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

type H3PrivateComfyPipelineRuntime<'authority, C, E, A> =
    H3PipelineRuntime<H3PrivateComfyVaeAdapter<H3PrivateComfyCore<'authority, C, E, A>>>;

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
    let expected_attention_device = H3AttentionDevice::Cuda {
        compute_capability: Some(admitted.compute_capability),
    };
    if !actual.active
        || actual.lease_id.trim().is_empty()
        || actual.device_id != admitted.device_id
        || actual.execution_fingerprint != admitted.execution_fingerprint
        || actual.backend
            != (H3CandleBackendDevice::Cuda {
                compute_capability: admitted.compute_capability,
            })
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

/// Structural composition proof for one private FL2VA attempt. It is private
/// to this module and its overlap argument is uninhabited in non-test builds,
/// so there is deliberately no production construction escape yet.
///
/// The engine already consumes one fresh session per job. Activating this
/// sealed composer must also accept authenticated VAE plans/opened descriptors
/// and load both VAEs inside this same owner and cancellation scope; a
/// preloaded bundle does not prove allocation or cancellation provenance and
/// is not sufficient. The binding below freezes the exact execution route,
/// Candle device, concrete attention authority, and opened transformer
/// identity before Qwen or transformer allocation, without activating any
/// shipping path.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn compose_private_comfy_fl2va_runtime<'authority, C, E, A>(
    qwen_weights_path: &Path,
    qwen_support: H3PrivateQwenSupport,
    conditioner_lease: C,
    opened_transformer: H3ComfyOpenedInt8Checkpoint,
    attention: H3AttentionRuntimeAuthority,
    vae: H3ComfyVaeRuntimeBundle,
    execution_lease: E,
    artifact_lease: A,
    memory_overlap: H3PrivateFl2VaMemoryOverlapAuthority,
    authority: &'authority FrozenH3FactoryAuthority,
    progress: &ProgressReporter,
    checkpoint: &mut dyn H3PipelineCheckpoint,
) -> Result<H3PrivateComfyPipelineRuntime<'authority, C, E, A>>
where
    C: H3PrivateQwenConditionerLease + Send + Sync,
    E: H3BackendExecutionLease + Send + Sync,
    A: H3PrivateFl2VaArtifactLease + Send + Sync,
{
    let admitted = authority.private_fl2va_runtime_authority()?;
    let bound_execution = capture_private_execution_route(execution_lease, &admitted)?;
    memory_overlap.validate()?;
    if memory_overlap.factory_identity_sha256 != admitted.factory_identity_sha256
        || memory_overlap.identity_sha256() != artifact_lease.memory_overlap_identity_sha256()
    {
        bail!("private MiniMax H3 overlap authority differs before component loading");
    }
    let expected_transformer_policy = match &admitted.quantization {
        H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
            transformer_policy_sha256,
            ..
        } => transformer_policy_sha256,
        H3FactoryQuantizationAuthority::OfficialBf16 => {
            bail!("private MiniMax H3 Comfy runtime requires quantized transformer authority")
        }
    };
    if !artifact_lease.is_active()
        || artifact_lease.transformer_task() != H3TransformerTask::T2VaFl2Va
        || artifact_lease.transformer_policy_identity_sha256() != expected_transformer_policy
        || artifact_lease.attention_runtime_identity_sha256()
            != admitted.attention.runtime_identity_sha256
        || artifact_lease.attention_kernel_identity()
            != admitted.attention.qualification_kernel_identity
        || artifact_lease.attention_qualification_sha256()
            != admitted.attention.qualification_sha256
    {
        bail!("private MiniMax H3 artifact lease differs before component loading");
    }
    let bound_stream = bind_private_comfy_stream(
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
    )?;
    let attempt = H3PrivateAttemptAuthority::new(bound_execution, artifact_lease);
    let (qwen_execution, qwen_artifacts) = attempt.qwen_projections();
    let block_execution = attempt.block_projection();
    let core_execution = attempt.block_projection();
    let core_artifacts = attempt.artifact_projection();
    let cancellation_slot = H3PrivateComfyCancellationSlot::default();
    let cancellation_guard = cancellation_slot.install(progress)?;
    let qwen = H3PrivateQwenAdapter::load_authorized(
        qwen_weights_path,
        qwen_support,
        conditioner_lease,
        qwen_execution,
        qwen_artifacts,
        authority,
        checkpoint,
    )?;
    let stream = load_and_pair_private_comfy_stream(
        bound_stream,
        &admitted.block_streaming,
        cancellation_slot,
    )?;
    let identity = H3PipelineBackendIdentity {
        kind: H3PipelineBackendKind::Cuda,
        device_id: admitted.device_id.clone(),
        execution_fingerprint: admitted.execution_fingerprint.clone(),
    };
    let denoiser = H3BlockStreamedDenoiser::new(
        identity,
        admitted.block_streaming.clone(),
        block_execution,
        stream.loader,
        stream.executor,
    )?;
    let core = H3PrivateVaeFreeStreamedCore::new(
        qwen,
        denoiser,
        cancellation_guard,
        core_execution,
        core_artifacts,
        admitted,
        stream.authority,
        memory_overlap,
        attempt,
    )?;
    let backend = H3PrivateComfyVaeAdapter::new(core, vae, authority)?;
    H3PipelineRuntime::new(backend, authority)
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
    use crate::minimax_h3::private_vae_adapter::H3PrivateVaeRuntime;
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
            compute_capability: (8, 9),
            execution_fingerprint: EXECUTION.into(),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 2_048,
            qwen_host_resident_parameter_bytes: 2_048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1_024,
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
            backend: H3CandleBackendDevice::Cuda {
                compute_capability: admitted.compute_capability,
            },
            location: candle_core::DeviceLocation::Cuda {
                gpu_id: admitted.device_ordinal,
            },
            attention_device: admitted.attention.device,
        }
    }

    #[test]
    fn every_execution_route_axis_fails_closed_before_binding() {
        let admitted = authority()
            .private_fl2va_runtime_authority_for_schema_tests()
            .unwrap();
        validate_private_execution_route_facts(&exact_execution_route_facts(&admitted), &admitted)
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
        }
    }

    struct AlternatingDeviceLease {
        devices: [Device; 2],
        calls: Arc<AtomicUsize>,
        device_id: String,
        execution_fingerprint: String,
        compute_capability: (u16, u16),
    }

    impl H3BackendExecutionLease for AlternatingDeviceLease {
        fn lease_id(&self) -> &str {
            "alternating-execution-lease"
        }

        fn device_id(&self) -> &str {
            &self.device_id
        }

        fn backend(&self) -> H3CandleBackendDevice {
            H3CandleBackendDevice::Cuda {
                compute_capability: self.compute_capability,
            }
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
            compute_capability: (8, 9),
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
        compute_capability: (u16, u16),
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
            H3CandleBackendDevice::Cuda {
                compute_capability: self.compute_capability,
            }
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
            "4444444444444444444444444444444444444444444444444444444444444444"
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
}
