//! Fail-closed Candle backend boundary for MiniMax H3 FL2VA.
//!
//! The orchestration in [`super::pipeline`] is deliberately runtime-neutral.
//! This module binds that contract to the audited Candle component types while
//! keeping construction behind every authority that is still missing from a
//! shipping Mold build. In particular, defining this adapter does not register
//! an engine, discover or read model files, or make H3 runnable.
//!
//! Component loaders receive only an immutable, fingerprinted plan. They are
//! not invoked until the legal policy, public capability contract, lossless
//! packed-attention runtime, and executable block-streaming DiT binding have
//! all been authorized. The exact #856 streaming plan is already part of the
//! frozen identity; today preflight still rejects before any loader callback.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use mold_candle::minimax_h3::{
    build_fl2va_presentation, pack_qwen_vision_u8, qwen_mrope_positions,
    AudioSoundtrackAssociation, AudioVae, AudioVaeCancellation, AudioVaePhase, DecodeSink,
    H3AdaLnMode, H3BlockProgress, H3BlockStack, H3ConditionerInput, H3ModalityTag,
    H3PrecisionProfile, H3RawTokenizer, H3Transformer, H3TransformerOutput, H3TransformerTask,
    H3VisionInput, MiniMaxH3VisualVae, StereoLatents, StereoWaveform, Uint8RgbPixels,
    VisualVaeEvent, VisualVaeObserver,
};
use mold_core::minimax_h3::{self as contract, Layout, Task};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::offload::FrozenH3BlockStreamingPlan;
use super::pipeline::{
    H3Fl2VaBackend, H3PipelineBackendIdentity, H3PipelineBackendKind, H3PipelineCheckpoint,
    H3PipelineEvent, H3PipelinePhase, H3PreparedEndpoint, H3TextConditioning, H3VideoEncodeSink,
};
use super::sampler::H3SamplerKind;
use super::{
    FrozenH3ConditionerPlacement, H3ConditionerExecution, H3ConditionerLease,
    H3ConditionerLifecycle,
};
use crate::progress::{InferenceCancelled, ProgressReporter};

const BACKEND_PLAN_SCHEMA_VERSION: u32 = 2;
const QWEN_MAXIMUM_TOKENS: usize = 262_144;
const QWEN_SPATIAL_MERGE_SIZE: usize = 2;
const PRODUCTION_TOKEN_REFINER_BLOCKS: usize = 2;
const PRODUCTION_TRANSFORMER_BLOCKS: usize = 50;
const PRODUCTION_OUTPUT_STAGES: usize = 1;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum H3ComponentRole {
    Conditioner,
    Transformer,
    VisualVae,
    AudioVae,
}

impl H3ComponentRole {
    const ALL: [Self; 4] = [
        Self::Conditioner,
        Self::Transformer,
        Self::VisualVae,
        Self::AudioVae,
    ];

    const fn stable_id(self) -> &'static str {
        match self {
            Self::Conditioner => "qwen3-vl-layer-50",
            Self::Transformer => "task-transformer",
            Self::VisualVae => "visual-vae",
            Self::AudioVae => "audio-vae",
        }
    }
}

/// Upstream validation authority for one exact component.
///
/// `content_sha256` identifies the landed bytes. `validation_sha256` binds the
/// header/config audit that produced the opaque Candle loader authority. This
/// type does not inspect paths or bytes itself: the compliance gate must run
/// before the loader that creates it is ever allowed to read an artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3ValidatedComponentAuthority {
    role: H3ComponentRole,
    content_sha256: String,
    validation_sha256: String,
}

impl H3ValidatedComponentAuthority {
    pub(crate) fn new(
        role: H3ComponentRole,
        content_sha256: impl Into<String>,
        validation_sha256: impl Into<String>,
    ) -> Result<Self> {
        let authority = Self {
            role,
            content_sha256: content_sha256.into(),
            validation_sha256: validation_sha256.into(),
        };
        authority.validate()?;
        Ok(authority)
    }

    fn validate(&self) -> Result<()> {
        require_sha256(&self.content_sha256, "H3 component content")?;
        require_sha256(&self.validation_sha256, "H3 component validation")
    }

    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn content_sha256(&self) -> &str {
        &self.content_sha256
    }

    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn validation_sha256(&self) -> &str {
        &self.validation_sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3ValidatedComponentSet {
    authorities: Vec<H3ValidatedComponentAuthority>,
    identity_sha256: String,
}

impl H3ValidatedComponentSet {
    pub(crate) fn new(
        authorities: impl IntoIterator<Item = H3ValidatedComponentAuthority>,
    ) -> Result<Self> {
        let mut authorities = authorities.into_iter().collect::<Vec<_>>();
        authorities.sort_by_key(|authority| authority.role);
        let roles = authorities
            .iter()
            .map(|authority| authority.role)
            .collect::<BTreeSet<_>>();
        if authorities.len() != H3ComponentRole::ALL.len()
            || roles != H3ComponentRole::ALL.into_iter().collect()
        {
            bail!(
                "MiniMax H3 backend requires exactly one conditioner, transformer, visual VAE, and audio VAE authority"
            );
        }
        for authority in &authorities {
            authority.validate()?;
        }
        let identity_sha256 = component_set_identity(&authorities);
        Ok(Self {
            authorities,
            identity_sha256,
        })
    }

    pub(crate) fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn authority(
        &self,
        role: H3ComponentRole,
    ) -> Option<&H3ValidatedComponentAuthority> {
        self.authorities
            .iter()
            .find(|authority| authority.role == role)
    }

    fn validate(&self) -> Result<()> {
        for authority in &self.authorities {
            authority.validate()?;
        }
        if self.authorities.len() != H3ComponentRole::ALL.len()
            || self
                .authorities
                .iter()
                .map(|authority| authority.role)
                .collect::<BTreeSet<_>>()
                != H3ComponentRole::ALL.into_iter().collect()
            || self.identity_sha256 != component_set_identity(&self.authorities)
        {
            bail!("MiniMax H3 component-set authority changed after validation");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum H3CandleBackendDevice {
    Cuda { compute_capability: (u16, u16) },
}

impl H3CandleBackendDevice {
    fn validate(self) -> Result<()> {
        match self {
            Self::Cuda {
                compute_capability: (major, _),
            } if major > 0 => Ok(()),
            Self::Cuda { .. } => bail!("MiniMax H3 CUDA route lacks a compute capability"),
        }
    }

    fn stable_id(self) -> String {
        match self {
            Self::Cuda {
                compute_capability: (major, minor),
            } => format!("cuda-sm{major}{minor}"),
        }
    }
}

/// Kernel prerequisites deliberately have no public `qualified` constructor
/// yet. The attention and quantization integrations own those authorities.
#[derive(Clone, Debug, Eq, PartialEq)]
struct H3DeferredRuntimeAuthorities {
    packed_attention_sha256: Option<String>,
    quantization_sha256: Option<String>,
}

impl H3DeferredRuntimeAuthorities {
    fn unavailable() -> Self {
        Self {
            packed_attention_sha256: None,
            quantization_sha256: None,
        }
    }

    fn validate(&self) -> Result<()> {
        for (label, fingerprint) in [
            ("packed attention", self.packed_attention_sha256.as_deref()),
            ("quantization", self.quantization_sha256.as_deref()),
        ] {
            if let Some(fingerprint) = fingerprint {
                require_sha256(fingerprint, label)?;
            }
        }
        Ok(())
    }
}

/// Complete immutable input to a future H3 Candle loader.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FrozenH3Fl2VaCandlePlan {
    schema_version: u32,
    canonical_model: String,
    task: Task,
    layout: Layout,
    device_id: String,
    backend: H3CandleBackendDevice,
    execution_fingerprint: String,
    conditioner_placement: FrozenH3ConditionerPlacement,
    block_streaming: FrozenH3BlockStreamingPlan,
    components: H3ValidatedComponentSet,
    runtime: H3DeferredRuntimeAuthorities,
    identity_sha256: String,
}

impl FrozenH3Fl2VaCandlePlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_unavailable(
        model: &str,
        device_id: impl Into<String>,
        backend: H3CandleBackendDevice,
        execution_fingerprint: impl Into<String>,
        conditioner_placement: FrozenH3ConditionerPlacement,
        block_streaming: FrozenH3BlockStreamingPlan,
        components: H3ValidatedComponentSet,
    ) -> Result<Self> {
        let contract = contract::capability_contract_for_model(model)
            .ok_or_else(|| anyhow!("{model:?} has no MiniMax H3 capability contract"))?;
        let device_id = device_id.into();
        let execution_fingerprint = execution_fingerprint.into();
        let mut plan = Self {
            schema_version: BACKEND_PLAN_SCHEMA_VERSION,
            canonical_model: contract.canonical_model.to_string(),
            task: contract.task,
            layout: contract.layout,
            device_id,
            backend,
            execution_fingerprint,
            conditioner_placement,
            block_streaming,
            components,
            runtime: H3DeferredRuntimeAuthorities::unavailable(),
            identity_sha256: String::new(),
        };
        plan.validate_fields()?;
        plan.identity_sha256 = backend_plan_identity(&plan);
        plan.validate()?;
        Ok(plan)
    }

    pub(crate) fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub(crate) fn component_set_identity(&self) -> &str {
        self.components.identity_sha256()
    }

    pub(crate) fn canonical_model(&self) -> &str {
        &self.canonical_model
    }

    pub(crate) const fn task(&self) -> Task {
        self.task
    }

    pub(crate) const fn layout(&self) -> Layout {
        self.layout
    }

    pub(crate) fn device_id(&self) -> &str {
        &self.device_id
    }

    pub(crate) const fn backend(&self) -> H3CandleBackendDevice {
        self.backend
    }

    pub(crate) fn execution_fingerprint(&self) -> &str {
        &self.execution_fingerprint
    }

    pub(crate) fn conditioner_placement(&self) -> &FrozenH3ConditionerPlacement {
        &self.conditioner_placement
    }

    pub(crate) fn block_streaming(&self) -> &FrozenH3BlockStreamingPlan {
        &self.block_streaming
    }

    pub(crate) fn components(&self) -> &H3ValidatedComponentSet {
        &self.components
    }

    fn validate_fields(&self) -> Result<()> {
        if self.schema_version != BACKEND_PLAN_SCHEMA_VERSION {
            bail!("MiniMax H3 backend plan uses an unsupported schema version");
        }
        let contract = contract::capability_contract_for_model(&self.canonical_model)
            .ok_or_else(|| anyhow!("H3 backend plan lost its canonical model"))?;
        if self.canonical_model != contract.canonical_model
            || self.task != contract.task
            || self.layout != contract.layout
        {
            bail!("MiniMax H3 backend model, task, or layout authority changed after freezing");
        }
        if self.device_id.trim().is_empty() {
            bail!("MiniMax H3 backend requires one frozen CUDA device ID");
        }
        self.backend.validate()?;
        require_sha256(&self.execution_fingerprint, "H3 scheduler execution")?;
        if self.conditioner_placement.execution_fingerprint != self.execution_fingerprint {
            bail!("MiniMax H3 conditioner and backend execution authorities disagree");
        }
        if self.conditioner_placement.execution == H3ConditionerExecution::CudaResident
            && self.conditioner_placement.device_id != self.device_id
        {
            bail!("MiniMax H3 resident conditioner is assigned to another CUDA device");
        }
        self.block_streaming.validate()?;
        if self.block_streaming.device_id != self.device_id
            || self.block_streaming.execution_fingerprint != self.execution_fingerprint
        {
            bail!("MiniMax H3 block streaming and backend admission authorities disagree");
        }
        self.components.validate()?;
        self.runtime.validate()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.validate_fields()?;
        require_sha256(&self.identity_sha256, "H3 backend plan")?;
        if self.identity_sha256 != backend_plan_identity(self) {
            bail!("MiniMax H3 backend plan identity changed after admission");
        }
        Ok(())
    }

    fn missing_requirements(&self) -> Vec<H3BackendRequirement> {
        let mut missing = Vec::new();
        if mold_core::require_model_activation(&self.canonical_model, Some(contract::FAMILY))
            .is_err()
        {
            missing.push(H3BackendRequirement::LicenseAuthorization);
        }
        if contract::runnable_capability_contract_for_model(&self.canonical_model).is_none() {
            missing.push(H3BackendRequirement::RunnableCapabilityContract);
        }
        if self.runtime.packed_attention_sha256.is_none() {
            missing.push(H3BackendRequirement::QualifiedLosslessPackedAttention);
        }
        // The #856 plan/lifecycle is bound above, but H3Transformer still owns
        // an eager Vec of blocks. Activation stays closed until the DiT loader
        // supplies those real block objects through that audited lifecycle.
        missing.push(H3BackendRequirement::IntegratedBlockStreamingTransformer);
        if self.layout == Layout::ComfyPrunedInt8ConvrotNvfp4Awq
            && self.runtime.quantization_sha256.is_none()
        {
            missing.push(H3BackendRequirement::QualifiedComfyQuantization);
        }
        missing
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum H3BackendRequirement {
    LicenseAuthorization,
    RunnableCapabilityContract,
    QualifiedLosslessPackedAttention,
    IntegratedBlockStreamingTransformer,
    QualifiedComfyQuantization,
}

#[derive(Debug, Error)]
pub(crate) enum H3BackendActivationError {
    #[error("invalid MiniMax H3 Candle backend plan: {0}")]
    InvalidPlan(String),
    #[error("MiniMax H3 Candle backend is not activated; missing authorities: {requirements:?}")]
    Unavailable {
        requirements: Vec<H3BackendRequirement>,
    },
    #[error("MiniMax H3 Candle component load failed: {0}")]
    Load(String),
}

/// Attempt-scoped scheduler authority for the one CUDA route used by the
/// complete FL2VA pipeline. It intentionally outlives the conditioner lease,
/// which is released immediately after layer-50 encoding.
pub(crate) trait H3BackendExecutionLease {
    fn lease_id(&self) -> &str;
    fn device_id(&self) -> &str;
    fn backend(&self) -> H3CandleBackendDevice;
    fn execution_fingerprint(&self) -> &str;
    fn device(&self) -> &Device;
    fn is_active(&self) -> bool;
}

impl<T> H3BackendExecutionLease for &T
where
    T: H3BackendExecutionLease + ?Sized,
{
    fn lease_id(&self) -> &str {
        T::lease_id(self)
    }

    fn device_id(&self) -> &str {
        T::device_id(self)
    }

    fn backend(&self) -> H3CandleBackendDevice {
        T::backend(self)
    }

    fn execution_fingerprint(&self) -> &str {
        T::execution_fingerprint(self)
    }

    fn device(&self) -> &Device {
        T::device(self)
    }

    fn is_active(&self) -> bool {
        T::is_active(self)
    }
}

/// Keeps every mmap-backed component immutable until all Candle tensors drop.
///
/// # Safety
///
/// Implementers must prevent replacement, mutation, or truncation of every
/// artifact covered by `component_set_identity` for the complete lease
/// lifetime.
pub(crate) unsafe trait H3BackendArtifactLease {
    fn component_set_identity(&self) -> &str;
    fn is_active(&self) -> bool;
}

// SAFETY: borrowing a valid artifact lease cannot shorten the underlying
// lease's lifetime or permit mutation of its protected artifact set.
unsafe impl<T> H3BackendArtifactLease for &T
where
    T: H3BackendArtifactLease + ?Sized,
{
    fn component_set_identity(&self) -> &str {
        T::component_set_identity(self)
    }

    fn is_active(&self) -> bool {
        T::is_active(self)
    }
}

/// Real component bundle returned by a future authorized loader. Field order
/// is load-bearing: mmap-backed models drop before the artifact lease.
pub(crate) struct H3CandleBackendComponents<C, E, A>
where
    C: H3ConditionerLease,
    E: H3BackendExecutionLease,
    A: H3BackendArtifactLease,
{
    pub(crate) conditioner: H3ConditionerLifecycle,
    pub(crate) transformer: H3Transformer,
    pub(crate) visual_vae: MiniMaxH3VisualVae,
    pub(crate) audio_vae: AudioVae,
    pub(crate) conditioner_lease: C,
    pub(crate) execution_lease: E,
    pub(crate) artifact_lease: A,
}

/// Load callbacks cannot run until [`FrozenH3Fl2VaCandlePlan`] has passed all
/// activation gates. Each callback receives the exact frozen component set,
/// route, conditioner placement, and block-streaming plan rather than
/// discovering paths or recomputing admission itself.
pub(crate) trait H3CandleBackendLoader {
    type ConditionerLease: H3ConditionerLease;
    type ExecutionLease: H3BackendExecutionLease;
    type ArtifactLease: H3BackendArtifactLease;

    fn load(
        self,
        plan: &FrozenH3Fl2VaCandlePlan,
    ) -> Result<
        H3CandleBackendComponents<
            Self::ConditionerLease,
            Self::ExecutionLease,
            Self::ArtifactLease,
        >,
    >;
}

pub(crate) struct H3CandleBackend<'a, C, E, A>
where
    C: H3ConditionerLease,
    E: H3BackendExecutionLease,
    A: H3BackendArtifactLease,
{
    plan: FrozenH3Fl2VaCandlePlan,
    progress: &'a ProgressReporter,
    // Drop the mmap-backed components before either authority lease.
    conditioner: H3ConditionerLifecycle,
    transformer: H3Transformer,
    visual_vae: MiniMaxH3VisualVae,
    audio_vae: AudioVae,
    conditioner_lease: C,
    execution_lease: E,
    artifact_lease: A,
}

type ActivatedH3CandleBackend<'a, L> = H3CandleBackend<
    'a,
    <L as H3CandleBackendLoader>::ConditionerLease,
    <L as H3CandleBackendLoader>::ExecutionLease,
    <L as H3CandleBackendLoader>::ArtifactLease,
>;

pub(crate) fn try_activate_h3_candle_backend<'a, L>(
    plan: FrozenH3Fl2VaCandlePlan,
    loader: L,
    progress: &'a ProgressReporter,
) -> std::result::Result<ActivatedH3CandleBackend<'a, L>, H3BackendActivationError>
where
    L: H3CandleBackendLoader,
{
    plan.validate()
        .map_err(|error| H3BackendActivationError::InvalidPlan(error.to_string()))?;
    let requirements = plan.missing_requirements();
    if !requirements.is_empty() {
        return Err(H3BackendActivationError::Unavailable { requirements });
    }
    if plan.task() != Task::Fl2va {
        return Err(H3BackendActivationError::InvalidPlan(
            "the eager MiniMax H3 Candle backend remains FL2VA-only".to_string(),
        ));
    }
    let components = loader
        .load(&plan)
        .map_err(|error| H3BackendActivationError::Load(error.to_string()))?;
    validate_loaded_components(&plan, &components)
        .map_err(|error| H3BackendActivationError::Load(error.to_string()))?;
    Ok(H3CandleBackend {
        plan,
        progress,
        conditioner: components.conditioner,
        transformer: components.transformer,
        visual_vae: components.visual_vae,
        audio_vae: components.audio_vae,
        conditioner_lease: components.conditioner_lease,
        execution_lease: components.execution_lease,
        artifact_lease: components.artifact_lease,
    })
}

impl<C, E, A> H3CandleBackend<'_, C, E, A>
where
    C: H3ConditionerLease,
    E: H3BackendExecutionLease,
    A: H3BackendArtifactLease,
{
    fn validate_runtime_authorities(&self) -> Result<()> {
        self.plan.validate()?;
        if !self.execution_lease.is_active()
            || self.execution_lease.lease_id().trim().is_empty()
            || self.execution_lease.device_id() != self.plan.device_id
            || self.execution_lease.backend() != self.plan.backend
            || self.execution_lease.execution_fingerprint() != self.plan.execution_fingerprint
            || !self.execution_lease.device().is_cuda()
        {
            bail!("MiniMax H3 execution lease changed after backend activation");
        }
        if !self.artifact_lease.is_active()
            || self.artifact_lease.component_set_identity()
                != self.plan.components.identity_sha256()
        {
            bail!("MiniMax H3 component authority changed after backend activation");
        }
        Ok(())
    }
}

impl<C, E, A> H3Fl2VaBackend for H3CandleBackend<'_, C, E, A>
where
    C: H3ConditionerLease,
    E: H3BackendExecutionLease,
    A: H3BackendArtifactLease,
{
    fn identity(&self) -> H3PipelineBackendIdentity {
        H3PipelineBackendIdentity {
            kind: H3PipelineBackendKind::Cuda,
            device_id: self.plan.device_id.clone(),
            execution_fingerprint: self.plan.execution_fingerprint.clone(),
        }
    }

    fn device(&self) -> &Device {
        self.execution_lease.device()
    }

    fn sampler_kind(&self) -> H3SamplerKind {
        match self.plan.layout() {
            Layout::OfficialBf16 => H3SamplerKind::OfficialEuler,
            Layout::ComfyPrunedInt8ConvrotNvfp4Awq => H3SamplerKind::ComfyResMultistep,
        }
    }

    fn encode_text(
        &mut self,
        prompt: &str,
        endpoints: &[H3PreparedEndpoint],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        self.validate_runtime_authorities()?;
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::PromptEncode,
            completed: 0,
            total: 1,
        })?;
        let (input, tags) = prepare_fl2va_conditioner_input(
            self.conditioner.tokenizer(),
            prompt,
            endpoints,
            self.conditioner_lease.device(),
        )?;
        // SAFETY: the backend owns `artifact_lease` after its exact component
        // identity was checked above. The unsafe-trait contract keeps every
        // conditioner shard immutable through this call.
        let states = unsafe {
            self.conditioner.encode_and_release_weights(
                &input,
                &self.plan.conditioner_placement,
                &mut self.conditioner_lease,
                self.progress,
            )?
        }
        .to_device(self.device())?;
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::PromptEncode,
            completed: 1,
            total: 1,
        })?;
        self.validate_runtime_authorities()?;
        Ok(H3TextConditioning {
            states,
            tags,
            #[cfg(test)]
            lifetime_probe: None,
        })
    }

    fn encode_visual_condition(
        &mut self,
        endpoint: &H3PreparedEndpoint,
        mode: mold_candle::minimax_h3::ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor> {
        self.validate_runtime_authorities()?;
        let pixels = Uint8RgbPixels::new(endpoint.pixels.to_device(self.device())?)?;
        let mut observer = H3VisualPipelineObserver {
            checkpoint,
            phase: H3PipelinePhase::VisualConditionEncodeChunk,
        };
        let latents = self
            .visual_vae
            .encode_condition_u8(pixels, mode, &mut observer)?;
        self.validate_runtime_authorities()?;
        Ok(latents)
    }

    fn denoise(
        &mut self,
        input: mold_candle::minimax_h3::H3ForwardInput<'_>,
        layout: &mold_candle::minimax_h3::H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput> {
        self.validate_runtime_authorities()?;
        let output = self
            .transformer
            .forward_with_observer(input, layout, |event| {
                transformer_checkpoint(checkpoint, event).map_err(candle_error)
            })?;
        self.validate_runtime_authorities()?;
        Ok(output)
    }

    fn decode_video(
        &mut self,
        latents: &Tensor,
        sink: &mut H3VideoEncodeSink,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        self.validate_runtime_authorities()?;
        let checkpoint = Rc::new(RefCell::new(checkpoint));
        let mut decode_sink = H3PipelineDecodeSink {
            sink,
            checkpoint: Rc::clone(&checkpoint),
            next_frame: 0,
        };
        let mut observer = H3SharedVisualPipelineObserver {
            checkpoint,
            phase: H3PipelinePhase::VisualDecodeChunk,
        };
        self.visual_vae
            .decode_normalized_to_sink(latents, &mut decode_sink, &mut observer)?;
        self.validate_runtime_authorities()
    }

    fn decode_audio(
        &mut self,
        latents: &StereoLatents,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform> {
        self.validate_runtime_authorities()?;
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::AudioDecodeChunk,
            completed: 0,
            total: 1,
        })?;
        let cancellation = H3AudioProgressCancellation(self.progress);
        let decoded = self.audio_vae.decode(
            latents,
            AudioSoundtrackAssociation::Generated,
            &cancellation,
        );
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(_) if self.progress.checkpoint().is_err() => return Err(InferenceCancelled.into()),
            Err(error) => return Err(error.into()),
        };
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::AudioDecodeChunk,
            completed: 1,
            total: 1,
        })?;
        self.validate_runtime_authorities()?;
        Ok(decoded)
    }
}

fn validate_loaded_components<C, E, A>(
    plan: &FrozenH3Fl2VaCandlePlan,
    components: &H3CandleBackendComponents<C, E, A>,
) -> Result<()>
where
    C: H3ConditionerLease,
    E: H3BackendExecutionLease,
    A: H3BackendArtifactLease,
{
    if !components.execution_lease.is_active()
        || components.execution_lease.lease_id().trim().is_empty()
        || components.execution_lease.device_id() != plan.device_id
        || components.execution_lease.backend() != plan.backend
        || components.execution_lease.execution_fingerprint() != plan.execution_fingerprint
        || !components.execution_lease.device().is_cuda()
    {
        bail!("MiniMax H3 loader returned a different or inactive execution lease");
    }
    if !components.artifact_lease.is_active()
        || components.artifact_lease.component_set_identity() != plan.components.identity_sha256()
    {
        bail!("MiniMax H3 loader returned a different or inactive artifact lease");
    }
    if components.conditioner_lease.device_id() != plan.conditioner_placement.device_id
        || !components.conditioner_lease.device().same_device(
            match plan.conditioner_placement.execution {
                H3ConditionerExecution::CudaResident => components.execution_lease.device(),
                H3ConditionerExecution::CpuOffloaded => &Device::Cpu,
            },
        )
    {
        bail!("MiniMax H3 loader returned a conditioner on a different frozen route");
    }
    if components.transformer.task() != H3TransformerTask::T2VaFl2Va {
        bail!("MiniMax H3 loader returned the Ref2VA transformer to the FL2VA backend");
    }
    if components.transformer.config() != &Default::default() {
        bail!("MiniMax H3 loader returned a non-production transformer architecture");
    }
    match plan.layout {
        Layout::OfficialBf16
            if components.transformer.precision() == H3PrecisionProfile::OfficialMixedBf16F32
                && components.transformer.adaln_mode() == H3AdaLnMode::Full => {}
        Layout::OfficialBf16 => {
            bail!("MiniMax H3 official plan requires mixed BF16/F32 full AdaLN")
        }
        Layout::ComfyPrunedInt8ConvrotNvfp4Awq => {
            bail!("MiniMax H3 Comfy quantized execution is not yet qualified")
        }
    }
    components
        .visual_vae
        .config()
        .validate_production_contract()?;
    if components.audio_vae.config() != &Default::default() {
        bail!("MiniMax H3 loader returned a non-production audio VAE configuration");
    }
    Ok(())
}

pub(super) fn prepare_fl2va_conditioner_input(
    tokenizer: &impl H3RawTokenizer,
    prompt: &str,
    endpoints: &[H3PreparedEndpoint],
    device: &Device,
) -> Result<(H3ConditionerInput, Vec<H3ModalityTag>)> {
    let mut packed_values = Vec::new();
    let mut grids = Vec::with_capacity(endpoints.len());
    let mut vision_tokens = Vec::with_capacity(endpoints.len());
    let mut patch_width = None;
    for endpoint in endpoints {
        let (_, channels, frames, height, width) = endpoint.pixels.dims5()?;
        if endpoint.pixels.dtype() != DType::U8
            || !endpoint.pixels.device().is_cpu()
            || (channels, frames) != (3, 1)
        {
            bail!("MiniMax H3 Qwen endpoint must be CPU U8 [1,3,1,H,W]");
        }
        let planar = endpoint.pixels.flatten_all()?.to_vec1::<u8>()?;
        let mut rgb = Vec::with_capacity(
            height
                .checked_mul(width)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| anyhow!("H3 Qwen endpoint byte count overflow"))?,
        );
        for row in 0..height {
            for column in 0..width {
                for channel in 0..3 {
                    rgb.push(planar[(channel * height + row) * width + column]);
                }
            }
        }
        let packed = pack_qwen_vision_u8(&rgb, 1, height, width)?;
        if patch_width
            .replace(packed.patch_width)
            .is_some_and(|expected| expected != packed.patch_width)
        {
            bail!("MiniMax H3 keyframes produced different Qwen patch widths");
        }
        vision_tokens.push(packed.grid.merged_tokens(QWEN_SPATIAL_MERGE_SIZE)?);
        grids.push(packed.grid);
        packed_values.extend(packed.values);
    }

    let presentation =
        build_fl2va_presentation(tokenizer, prompt, &vision_tokens, QWEN_MAXIMUM_TOKENS)?;
    let positions = qwen_mrope_positions(
        &presentation.qwen_mm_types,
        &grids,
        &[],
        QWEN_SPATIAL_MERGE_SIZE,
    )?;
    let sequence = presentation.token_ids.len();
    let input_ids = Tensor::from_vec(presentation.token_ids, (1, sequence), device)?;
    let position_values = positions.into_iter().flatten().collect::<Vec<_>>();
    let position_ids = Tensor::from_vec(position_values, (3, 1, sequence), device)?;
    let image = if grids.is_empty() {
        None
    } else {
        let patch_width = patch_width.context("H3 Qwen image patches lost their width")?;
        let patches = packed_values.len() / patch_width;
        let pixel_values = Tensor::from_vec(packed_values, (patches, patch_width), device)?;
        let grid_values = grids
            .into_iter()
            .flat_map(|grid| {
                [
                    u32::try_from(grid.temporal),
                    u32::try_from(grid.height),
                    u32::try_from(grid.width),
                ]
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("H3 Qwen grid exceeds U32")?;
        Some(H3VisionInput {
            pixel_values,
            grid_thw: Tensor::from_vec(grid_values, (vision_tokens.len(), 3), device)?,
        })
    };
    Ok((
        H3ConditionerInput {
            input_ids,
            position_ids,
            image,
            video: None,
        },
        presentation.h3_tags,
    ))
}

struct H3VisualPipelineObserver<'a> {
    checkpoint: &'a mut dyn H3PipelineCheckpoint,
    phase: H3PipelinePhase,
}

impl VisualVaeObserver for H3VisualPipelineObserver<'_> {
    fn checkpoint(&mut self, event: VisualVaeEvent) -> candle_core::Result<()> {
        self.checkpoint
            .checkpoint(H3PipelineEvent {
                phase: self.phase,
                completed: event.completed,
                total: event.total.max(1),
            })
            .map_err(candle_error)
    }
}

struct H3PipelineDecodeSink<'sink, 'checkpoint> {
    sink: &'sink mut H3VideoEncodeSink,
    checkpoint: Rc<RefCell<&'checkpoint mut dyn H3PipelineCheckpoint>>,
    next_frame: usize,
}

impl DecodeSink for H3PipelineDecodeSink<'_, '_> {
    fn write(&mut self, frame_offset: usize, frames: &Tensor) -> candle_core::Result<()> {
        if frame_offset != self.next_frame {
            candle_core::bail!(
                "MiniMax H3 decoder emitted frame offset {frame_offset}, expected {}",
                self.next_frame
            );
        }
        let (batch, channels, count, height, width) = frames.dims5()?;
        if batch != 1 || channels != 3 || frames.dtype() != DType::F32 {
            candle_core::bail!("MiniMax H3 decoded frames must be FP32 [1,3,T,H,W]");
        }
        let values = frames
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for frame_index in 0..count {
            let rgb_bytes = height
                .checked_mul(width)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| candle_core::Error::Msg("H3 RGB frame size overflow".into()))?;
            let mut rgb = Vec::with_capacity(rgb_bytes);
            for row in 0..height {
                for column in 0..width {
                    for channel in 0..3 {
                        let value = values
                            [((channel * count + frame_index) * height + row) * width + column];
                        rgb.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
                    }
                }
            }
            let width = u32::try_from(width)
                .map_err(|_| candle_core::Error::Msg("H3 RGB frame width exceeds U32".into()))?;
            let height = u32::try_from(height)
                .map_err(|_| candle_core::Error::Msg("H3 RGB frame height exceeds U32".into()))?;
            let image = RgbImage::from_raw(width, height, rgb)
                .ok_or_else(|| candle_core::Error::Msg("H3 RGB frame shape overflow".into()))?;
            let mut checkpoint = self.checkpoint.borrow_mut();
            self.sink
                .push(&image, &mut **checkpoint)
                .map_err(candle_error)?;
            self.next_frame += 1;
        }
        Ok(())
    }
}

struct H3SharedVisualPipelineObserver<'a> {
    checkpoint: Rc<RefCell<&'a mut dyn H3PipelineCheckpoint>>,
    phase: H3PipelinePhase,
}

impl VisualVaeObserver for H3SharedVisualPipelineObserver<'_> {
    fn checkpoint(&mut self, event: VisualVaeEvent) -> candle_core::Result<()> {
        self.checkpoint
            .borrow_mut()
            .checkpoint(H3PipelineEvent {
                phase: self.phase,
                completed: event.completed,
                total: event.total.max(1),
            })
            .map_err(candle_error)
    }
}

struct H3AudioProgressCancellation<'a>(&'a ProgressReporter);

impl AudioVaeCancellation for H3AudioProgressCancellation<'_> {
    fn is_cancelled(&self, _phase: AudioVaePhase) -> bool {
        self.0.checkpoint().is_err()
    }
}

fn transformer_checkpoint(
    checkpoint: &mut dyn H3PipelineCheckpoint,
    event: H3BlockProgress,
) -> Result<()> {
    let production_total =
        PRODUCTION_TOKEN_REFINER_BLOCKS + PRODUCTION_TRANSFORMER_BLOCKS + PRODUCTION_OUTPUT_STAGES;
    let expected_stack_total = match event.stack {
        H3BlockStack::TokenRefiner => PRODUCTION_TOKEN_REFINER_BLOCKS,
        H3BlockStack::Transformer => PRODUCTION_TRANSFORMER_BLOCKS,
        H3BlockStack::Output => PRODUCTION_OUTPUT_STAGES,
    };
    if event.total != expected_stack_total || event.completed > event.total {
        bail!("MiniMax H3 transformer reported non-production block progress");
    }
    let (completed, total) = match event.stack {
        H3BlockStack::TokenRefiner => (event.completed, production_total),
        H3BlockStack::Transformer => (
            event.completed + PRODUCTION_TOKEN_REFINER_BLOCKS,
            production_total,
        ),
        H3BlockStack::Output => (
            event.completed + PRODUCTION_TOKEN_REFINER_BLOCKS + PRODUCTION_TRANSFORMER_BLOCKS,
            production_total,
        ),
    };
    checkpoint.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::TransformerBlock,
        completed,
        total,
    })
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} identity must be a SHA-256 digest");
    }
    Ok(())
}

fn component_set_identity(authorities: &[H3ValidatedComponentAuthority]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.component-authorities.v1\0");
    for authority in authorities {
        digest.update(authority.role.stable_id().as_bytes());
        digest.update([0]);
        digest.update(authority.content_sha256.as_bytes());
        digest.update([0]);
        digest.update(authority.validation_sha256.as_bytes());
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

fn backend_plan_identity(plan: &FrozenH3Fl2VaCandlePlan) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.candle-backend-plan.v2\0");
    digest.update(plan.schema_version.to_le_bytes());
    digest.update(plan.canonical_model.as_bytes());
    digest.update([0]);
    digest.update(match plan.task {
        Task::Fl2va => b"fl2va".as_slice(),
        Task::Ref2va => b"ref2va".as_slice(),
    });
    digest.update([0]);
    digest.update(match plan.layout {
        Layout::OfficialBf16 => b"official-bf16".as_slice(),
        Layout::ComfyPrunedInt8ConvrotNvfp4Awq => b"comfy-pruned-int8".as_slice(),
    });
    digest.update([0]);
    digest.update(plan.device_id.as_bytes());
    digest.update([0]);
    digest.update(plan.backend.stable_id().as_bytes());
    digest.update([0]);
    digest.update(plan.execution_fingerprint.as_bytes());
    digest.update([0]);
    digest.update(plan.conditioner_placement.device_id.as_bytes());
    digest.update([0]);
    digest.update(match plan.conditioner_placement.execution {
        H3ConditionerExecution::CudaResident => b"cuda-resident".as_slice(),
        H3ConditionerExecution::CpuOffloaded => b"cpu-offloaded".as_slice(),
    });
    for value in [
        plan.conditioner_placement.memory.resident_parameter_bytes,
        plan.conditioner_placement.memory.activation_workspace_bytes,
        plan.conditioner_placement.memory.charged_host_bytes,
        plan.conditioner_placement.memory.charged_vram_bytes,
    ] {
        digest.update(value.to_le_bytes());
    }
    digest.update(plan.block_streaming.device_id.as_bytes());
    digest.update([0]);
    digest.update(plan.block_streaming.execution_fingerprint.as_bytes());
    digest.update(plan.block_streaming.resident_block_count.to_le_bytes());
    digest.update(plan.block_streaming.prefetch_depth.to_le_bytes());
    digest.update(plan.components.identity_sha256.as_bytes());
    for authority in [
        plan.runtime.packed_attention_sha256.as_deref(),
        plan.runtime.quantization_sha256.as_deref(),
    ] {
        digest.update([0]);
        digest.update(authority.unwrap_or("unavailable").as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn candle_error(error: anyhow::Error) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use mold_candle::minimax_h3::{H3RawTokenizer, PresentationError};

    use super::*;

    const EXECUTION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn authorities() -> H3ValidatedComponentSet {
        H3ValidatedComponentSet::new(H3ComponentRole::ALL.into_iter().enumerate().map(
            |(index, role)| {
                H3ValidatedComponentAuthority::new(
                    role,
                    digest(char::from(b'1' + index as u8)),
                    digest(char::from(b'5' + index as u8)),
                )
                .unwrap()
            },
        ))
        .unwrap()
    }

    fn plan(model: &str) -> FrozenH3Fl2VaCandlePlan {
        FrozenH3Fl2VaCandlePlan::new_unavailable(
            model,
            "gpu-0",
            H3CandleBackendDevice::Cuda {
                compute_capability: (8, 9),
            },
            EXECUTION,
            FrozenH3ConditionerPlacement::new(
                "cpu",
                H3ConditionerExecution::CpuOffloaded,
                EXECUTION,
                mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES,
                0,
                1_024,
            )
            .unwrap(),
            FrozenH3BlockStreamingPlan::new("gpu-0", EXECUTION, 8, 1).unwrap(),
            authorities(),
        )
        .unwrap()
    }

    #[test]
    fn plan_binds_route_component_and_conditioner_authorities() {
        let plan = plan(contract::FL2VA_OFFICIAL);
        plan.validate().unwrap();
        assert_eq!(plan.identity_sha256().len(), 64);
        assert_eq!(plan.component_set_identity().len(), 64);

        for (mutation, rejection) in [
            (
                (|plan: &mut FrozenH3Fl2VaCandlePlan| plan.device_id = "gpu-1".into())
                    as fn(&mut FrozenH3Fl2VaCandlePlan),
                "streaming and backend admission authorities disagree",
            ),
            (
                |plan: &mut FrozenH3Fl2VaCandlePlan| {
                    plan.backend = H3CandleBackendDevice::Cuda {
                        compute_capability: (9, 0),
                    }
                },
                "plan identity changed after admission",
            ),
            (
                |plan: &mut FrozenH3Fl2VaCandlePlan| {
                    plan.conditioner_placement.memory.activation_workspace_bytes += 1
                },
                "plan identity changed after admission",
            ),
            (
                |plan: &mut FrozenH3Fl2VaCandlePlan| {
                    plan.components.authorities[0].content_sha256 = digest('f')
                },
                "component-set authority changed after validation",
            ),
            (
                |plan: &mut FrozenH3Fl2VaCandlePlan| {
                    plan.components.authorities[1].validation_sha256 = digest('e')
                },
                "component-set authority changed after validation",
            ),
            (
                |plan: &mut FrozenH3Fl2VaCandlePlan| plan.block_streaming.resident_block_count += 1,
                "plan identity changed after admission",
            ),
        ] {
            let mut changed = plan.clone();
            mutation(&mut changed);
            let error = changed.validate().unwrap_err().to_string();
            assert!(error.contains(rejection), "unexpected rejection: {error}");
        }
    }

    #[test]
    fn component_set_rejects_missing_duplicate_and_malformed_authorities() {
        let malformed = H3ValidatedComponentAuthority::new(
            H3ComponentRole::AudioVae,
            "not-a-hash",
            digest('1'),
        );
        assert!(malformed.is_err());

        let duplicate = H3ValidatedComponentSet::new([
            H3ValidatedComponentAuthority::new(
                H3ComponentRole::Conditioner,
                digest('1'),
                digest('2'),
            )
            .unwrap(),
            H3ValidatedComponentAuthority::new(
                H3ComponentRole::Conditioner,
                digest('3'),
                digest('4'),
            )
            .unwrap(),
        ]);
        assert!(duplicate.is_err());
    }

    #[test]
    fn ref2va_plan_is_contract_only_and_route_mutations_fail_before_activation() {
        let ref2va = FrozenH3Fl2VaCandlePlan::new_unavailable(
            contract::REF2VA_OFFICIAL,
            "gpu-0",
            H3CandleBackendDevice::Cuda {
                compute_capability: (8, 9),
            },
            EXECUTION,
            FrozenH3ConditionerPlacement::new(
                "cpu",
                H3ConditionerExecution::CpuOffloaded,
                EXECUTION,
                mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES,
                0,
                1,
            )
            .unwrap(),
            FrozenH3BlockStreamingPlan::new("gpu-0", EXECUTION, 8, 1).unwrap(),
            authorities(),
        )
        .unwrap();
        assert_eq!(ref2va.task(), Task::Ref2va);
        ref2va.validate().unwrap();

        assert!(FrozenH3Fl2VaCandlePlan::new_unavailable(
            contract::FL2VA_OFFICIAL,
            "gpu-0",
            H3CandleBackendDevice::Cuda {
                compute_capability: (8, 9),
            },
            EXECUTION,
            FrozenH3ConditionerPlacement::new(
                "gpu-1",
                H3ConditionerExecution::CudaResident,
                EXECUTION,
                0,
                mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES,
                1,
            )
            .unwrap(),
            FrozenH3BlockStreamingPlan::new("gpu-0", EXECUTION, 8, 1).unwrap(),
            authorities(),
        )
        .is_err());

        let wrong_stream_route = FrozenH3Fl2VaCandlePlan::new_unavailable(
            contract::FL2VA_OFFICIAL,
            "gpu-0",
            H3CandleBackendDevice::Cuda {
                compute_capability: (8, 9),
            },
            EXECUTION,
            FrozenH3ConditionerPlacement::new(
                "cpu",
                H3ConditionerExecution::CpuOffloaded,
                EXECUTION,
                mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES,
                0,
                1,
            )
            .unwrap(),
            FrozenH3BlockStreamingPlan::new("gpu-1", EXECUTION, 8, 1).unwrap(),
            authorities(),
        )
        .unwrap_err();
        assert!(wrong_stream_route
            .to_string()
            .contains("streaming and backend admission authorities disagree"));

        let wrong_stream_execution = FrozenH3Fl2VaCandlePlan::new_unavailable(
            contract::FL2VA_OFFICIAL,
            "gpu-0",
            H3CandleBackendDevice::Cuda {
                compute_capability: (8, 9),
            },
            EXECUTION,
            FrozenH3ConditionerPlacement::new(
                "cpu",
                H3ConditionerExecution::CpuOffloaded,
                EXECUTION,
                mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES,
                0,
                1,
            )
            .unwrap(),
            FrozenH3BlockStreamingPlan::new("gpu-0", digest('b'), 8, 1).unwrap(),
            authorities(),
        )
        .unwrap_err();
        assert!(wrong_stream_execution
            .to_string()
            .contains("streaming and backend admission authorities disagree"));
    }

    struct NeverConditionerLease;

    impl H3ConditionerLease for NeverConditionerLease {
        fn lease_id(&self) -> &str {
            unreachable!()
        }

        fn device_id(&self) -> &str {
            unreachable!()
        }

        fn device(&self) -> &Device {
            unreachable!()
        }

        fn release(&mut self) {
            unreachable!()
        }
    }

    struct NeverExecutionLease;

    impl H3BackendExecutionLease for NeverExecutionLease {
        fn lease_id(&self) -> &str {
            unreachable!()
        }

        fn device_id(&self) -> &str {
            unreachable!()
        }

        fn backend(&self) -> H3CandleBackendDevice {
            unreachable!()
        }

        fn execution_fingerprint(&self) -> &str {
            unreachable!()
        }

        fn device(&self) -> &Device {
            unreachable!()
        }

        fn is_active(&self) -> bool {
            unreachable!()
        }
    }

    struct NeverArtifactLease;

    // SAFETY: this test lease is never constructed by a loader and therefore
    // never claims an artifact lifetime.
    unsafe impl H3BackendArtifactLease for NeverArtifactLease {
        fn component_set_identity(&self) -> &str {
            unreachable!()
        }

        fn is_active(&self) -> bool {
            unreachable!()
        }
    }

    struct NeverLoader(Arc<AtomicBool>);

    impl H3CandleBackendLoader for NeverLoader {
        type ConditionerLease = NeverConditionerLease;
        type ExecutionLease = NeverExecutionLease;
        type ArtifactLease = NeverArtifactLease;

        fn load(
            self,
            _plan: &FrozenH3Fl2VaCandlePlan,
        ) -> Result<
            H3CandleBackendComponents<
                Self::ConditionerLease,
                Self::ExecutionLease,
                Self::ArtifactLease,
            >,
        > {
            self.0.store(true, Ordering::SeqCst);
            unreachable!("the legal/runtime gate must reject before loading")
        }
    }

    #[test]
    fn loader_is_never_called_before_every_activation_authority() {
        for (model, quantized) in [
            (contract::FL2VA_OFFICIAL, false),
            (contract::FL2VA_COMFY, true),
            (contract::REF2VA_OFFICIAL, false),
        ] {
            let called = Arc::new(AtomicBool::new(false));
            let error = try_activate_h3_candle_backend(
                plan(model),
                NeverLoader(Arc::clone(&called)),
                &ProgressReporter::default(),
            )
            .err()
            .expect("H3 must remain unavailable");
            let H3BackendActivationError::Unavailable { requirements } = error else {
                panic!("unexpected activation error: {error}");
            };
            assert!(requirements.contains(&H3BackendRequirement::LicenseAuthorization));
            assert!(requirements.contains(&H3BackendRequirement::RunnableCapabilityContract));
            assert!(requirements.contains(&H3BackendRequirement::QualifiedLosslessPackedAttention));
            assert!(
                requirements.contains(&H3BackendRequirement::IntegratedBlockStreamingTransformer)
            );
            assert_eq!(
                requirements.contains(&H3BackendRequirement::QualifiedComfyQuantization),
                quantized
            );
            assert!(!called.load(Ordering::SeqCst));
        }
        assert!(!contract::capabilities(Task::Fl2va).runtime_available);
    }

    struct TinyTokenizer;

    impl H3RawTokenizer for TinyTokenizer {
        fn encode_raw(&self, text: &str) -> std::result::Result<Vec<u32>, PresentationError> {
            Ok(if text.is_empty() {
                Vec::new()
            } else {
                vec![42]
            })
        }

        fn token_id(&self, token: &str) -> Option<u32> {
            match token {
                "<|vision_start|>" => Some(151_652),
                "<|vision_end|>" => Some(151_653),
                "<|image_pad|>" => Some(151_655),
                "<|video_pad|>" => Some(151_656),
                _ => None,
            }
        }
    }

    #[test]
    fn conditioner_bridge_preserves_rgb_order_tags_and_mrope() {
        let mut planar = vec![0_u8; 3 * 32 * 32];
        planar[0] = 10;
        planar[32 * 32] = 20;
        planar[2 * 32 * 32] = 30;
        let endpoint = H3PreparedEndpoint {
            anchor: super::super::pipeline::H3EndpointAnchor::First,
            source_width: 32,
            source_height: 32,
            resize: super::super::pipeline::H3EndpointResize::Identity,
            pixels: Tensor::from_vec(planar, (1, 3, 1, 32, 32), &Device::Cpu).unwrap(),
        };
        let (input, tags) =
            prepare_fl2va_conditioner_input(&TinyTokenizer, "prompt", &[endpoint], &Device::Cpu)
                .unwrap();
        assert_eq!(input.input_ids.dims(), [1, 5]);
        assert_eq!(input.position_ids.dims(), [3, 1, 5]);
        assert_eq!(tags.len(), 5);
        assert_eq!(
            tags.iter()
                .filter(|tag| **tag == H3ModalityTag::Vision)
                .count(),
            3
        );
        let image = input.image.unwrap();
        assert_eq!(image.pixel_values.dims(), [4, 1536]);
        assert_eq!(
            image.grid_thw.to_vec2::<u32>().unwrap(),
            vec![vec![1, 2, 2]]
        );
        let values = image
            .pixel_values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((values[0] - (10.0 / 127.5 - 1.0)).abs() < 1e-6);
        assert!((values[512] - (20.0 / 127.5 - 1.0)).abs() < 1e-6);
        assert!((values[1024] - (30.0 / 127.5 - 1.0)).abs() < 1e-6);
    }

    #[derive(Default)]
    struct ProgressEvents(Vec<H3PipelineEvent>);

    impl H3PipelineCheckpoint for ProgressEvents {
        fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()> {
            self.0.push(event);
            Ok(())
        }
    }

    #[test]
    fn transformer_progress_maps_exact_production_stacks_without_gaps() {
        let mut checkpoint = ProgressEvents::default();
        for event in [
            H3BlockProgress {
                stack: H3BlockStack::TokenRefiner,
                completed: 0,
                total: 2,
            },
            H3BlockProgress {
                stack: H3BlockStack::TokenRefiner,
                completed: 2,
                total: 2,
            },
            H3BlockProgress {
                stack: H3BlockStack::Transformer,
                completed: 0,
                total: 50,
            },
            H3BlockProgress {
                stack: H3BlockStack::Transformer,
                completed: 50,
                total: 50,
            },
            H3BlockProgress {
                stack: H3BlockStack::Output,
                completed: 0,
                total: 1,
            },
            H3BlockProgress {
                stack: H3BlockStack::Output,
                completed: 1,
                total: 1,
            },
        ] {
            transformer_checkpoint(&mut checkpoint, event).unwrap();
        }
        assert_eq!(
            checkpoint
                .0
                .iter()
                .map(|event| (event.completed, event.total))
                .collect::<Vec<_>>(),
            [(0, 53), (2, 53), (2, 53), (52, 53), (52, 53), (53, 53)]
        );

        let error = transformer_checkpoint(
            &mut checkpoint,
            H3BlockProgress {
                stack: H3BlockStack::Transformer,
                completed: 0,
                total: 49,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("non-production block progress"));
    }
}
