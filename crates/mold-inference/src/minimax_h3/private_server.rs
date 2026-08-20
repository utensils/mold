//! Server-facing evidence boundary for private MiniMax H3 UAT.
//!
//! This non-shipping module deliberately exposes only payload-free facts. The
//! private runtime's opened descriptors, Candle tensors, paths, trait graph,
//! and singular owner roots remain crate-private. A reviewed CUDA campaign can
//! authorize the opaque preparation seam by adding the exact SHA-256 of one
//! record matching the schema below. The reader still fails before path access
//! whenever its reviewed authority set is empty.

#[cfg(all(feature = "mp4", feature = "cuda"))]
const H3_CUDA_ATTEMPT_RETAINED_MARKER: &str =
    "CUDA execution attempt retained resources; server restart required";
#[cfg(feature = "mp4")]
use std::collections::BTreeMap;
use std::fs::File;
#[cfg(not(feature = "h3"))]
use std::io::Read;
use std::path::{Component, Path, PathBuf};
#[cfg(feature = "mp4")]
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
#[cfg(feature = "mp4")]
use candle_core::Device;
#[cfg(feature = "mp4")]
use mold_candle::minimax_h3::{
    open_h3_comfy_published_int8_checkpoint, H3AuthenticatedQwenNvfp4Authority,
    H3ComfyInt8Cancellation, H3ComfyOpenedInt8Checkpoint, H3ComfyPublishedArtifact,
    H3TransformerTask, H3_QWEN_NVFP4_AWQ_POLICY_SHA256, H3_QWEN_NVFP4_AWQ_SHA256,
};
use mold_candle::minimax_h3::{
    H3AttentionDevice, H3AttentionModelContract, H3AttentionRuntimeAuthority,
};
use mold_core::minimax_h3::{self as contract, Mode, Task};
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use mold_core::{GenerateRequest, GenerateResponse};
#[cfg(feature = "mp4")]
use mold_core::{OutputFormat, VideoData};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "mp4")]
use super::backend::{H3BackendArtifactLease, H3BackendExecutionLease, H3CandleBackendDevice};
#[cfg(feature = "mp4")]
use super::engine::H3EngineProgressObserver;
#[cfg(feature = "mp4")]
use super::pipeline::{H3PipelineCheckpoint, H3PipelineEvent};
#[cfg(feature = "mp4")]
use super::private_fl2va_runtime::{
    bind_private_comfy_fl2va_phase_owner, issue_private_fl2va_memory_overlap,
    run_private_comfy_fl2va_attempt, H3PrivateFl2VaArtifactLease,
    H3PrivateFl2VaMemoryOverlapAuthority, H3PrivatePhaseRuntimeOutput, H3PrivateRetainedVaeReload,
};
#[cfg(feature = "mp4")]
use super::private_opened_evidence::{
    build_private_fl2va_admission_attempt, prepare_private_fl2va_admission_request,
    H3PrivateComfyStorageAuthority, H3PrivatePreparedFl2VaAttempt,
};
use super::private_opened_evidence::{
    H3PrivateOpenedActivationFacts, H3PrivatePreparedFl2VaFactoryInputs,
    H3PrivateQualifiedRuntimeBounds,
};
use super::private_qualification::validate_private_presentation_scope;
#[cfg(test)]
use super::private_qualification::validate_private_presentation_scope_against_evidence;
use super::private_qualification::H3PrivateArtifactQualificationReport;
#[cfg(feature = "mp4")]
use super::private_qualification::{qualify_private_artifacts_with_control, H3QualifiedArtifact};
#[cfg(feature = "mp4")]
use super::private_qwen::{
    open_authorized_private_qwen_authority, released_h3_private_qwen_loader_memory_authority,
    H3PrivateQwenArtifactLease, H3PrivateQwenConditionerLease, H3PrivateQwenLoaderMemoryRoute,
    H3PrivateQwenOpenRouteAuthority,
};
#[cfg(feature = "mp4")]
use super::private_qwen_support::{load_qualified_private_qwen_support, H3PrivateQwenSupport};
use super::private_runtime_observer::H3PrivateRuntimeProcessObservation;
#[cfg(feature = "mp4")]
use super::private_runtime_observer::{
    capture_process_observation, H3PrivateRuntimeAuthorityObservation,
    H3PrivateRuntimeBoundCapture, H3PrivateRuntimeEnvelopeObservation,
};
#[cfg(feature = "mp4")]
use super::sampler::H3SamplerKind;
#[cfg(feature = "mp4")]
use super::vae_runtime::{
    open_h3_comfy_vae_authority, H3AuthenticatedComfyVaeAuthority, H3ComfyVaeLoadError,
    H3ComfyVaeLoadEvent, H3ComfyVaeLoadObserver,
};
#[cfg(feature = "mp4")]
use super::H3ConditionerLease;
#[cfg(feature = "mp4")]
use crate::attention::{AttentionBackend, AttentionChunkPolicy};
#[cfg(feature = "mp4")]
use crate::h3_factory::H3FactoryTurboAdapterAuthority;
#[cfg(feature = "mp4")]
use crate::h3_factory::H3PrivateFl2VaFactoryAuthority;
use crate::progress::ProgressReporter;
use crate::{
    h3_factory_activation_prerequisites, FrozenH3FactoryAuthority, H3FactoryActivationPrerequisite,
    H3FactoryAttentionInput, InferenceCancellationToken,
};
#[cfg(feature = "mp4")]
use crate::{
    H3FactoryAuthorityInput, H3FactoryComponentAuthority, H3FactoryComponentRole,
    H3FactoryConditionerPlacement, H3FactoryEndpointAnchor, H3FactoryExecutionBudgetEchoInput,
    H3FactoryPreparedRequestInput, H3FactoryQuantizationAuthority,
};

pub(crate) const RUNTIME_QUALIFICATION_SCHEMA: &str =
    "mold.minimax-h3.private-runtime-qualification.v4";
pub(crate) const RUNTIME_QUALIFICATION_DECISION: &str = "qualified-private-fl2va-runtime";
pub(crate) const MAX_RUNTIME_QUALIFICATION_BYTES: u64 = 128 * 1024;

pub(crate) fn valid_stable_cuda_device_id(value: &str) -> bool {
    let Some(uuid) = value.strip_prefix("cuda:") else {
        return false;
    };
    uuid.len() == 32
        && uuid
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Exact reviewed runtime-qualification record hashes.
///
/// Each entry is an independently reviewed, content-addressed qualification
/// record. Accepting a hash supplied alongside a record would let the caller
/// self-authorize, so this list remains a source-controlled authority boundary.
///
/// The first record covers only the exact compact FL2VA quality envelope and
/// artifact/device/runtime/kernel identities retained by its v5 campaign.
const REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256: &[&str] =
    &["f624f71ce1eba7ebb75a13801da855a92f5eec0fccbcb9783f547479c7abfce5"];

const H3_ARTIFACT_VERIFICATION_PROGRESS: &str = "Verifying MiniMax H3 artifacts";
const H3_VAE_ARTIFACT_VERIFICATION_PROGRESS: &str = "Verifying MiniMax H3 VAE artifacts";

/// Report whether this binary contains at least one reviewed private-runtime
/// qualification record. This performs no filesystem access and is suitable
/// for an authenticated ingress gate before queue or dependency setup.
pub const fn reviewed_h3_private_runtime_available() -> bool {
    reviewed_h3_private_runtime_available_for_task(Task::Fl2va)
}

/// Report whether this binary contains a reviewed qualification for the
/// exact private task partition. A qualification for one H3 transformer task
/// must never authorize another task with different artifacts, conditioning,
/// and peak-memory behavior.
pub const fn reviewed_h3_private_runtime_available_for_task(task: Task) -> bool {
    match task {
        #[cfg(feature = "h3")]
        Task::Fl2va => true,
        #[cfg(not(feature = "h3"))]
        Task::Fl2va => !REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256.is_empty(),
        // Ref2VA remains sealed until its own artifact and runtime campaign is
        // reviewed. Do not infer authority from an FL2VA record.
        Task::Ref2va => false,
    }
}

/// One scheduler-visible CUDA route eligible for authenticated private H3
/// capability presentation. Supplying this record grants nothing: the
/// source-controlled runtime record must match it exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct H3PrivatePresentationRoute<'a> {
    pub device_id: &'a str,
    pub device_ordinal: usize,
    pub compute_capability: (u16, u16),
}

/// Payload-free proof that the exact reviewed FL2VA qualification, external
/// authorization scope, current runtime code, and one live scheduler route
/// were all crossed for capability presentation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivatePresentationAuthority {
    canonical_model: String,
    task: Task,
    device_id: String,
    device_ordinal: usize,
    compute_capability: (u16, u16),
}

impl H3PrivatePresentationAuthority {
    pub fn canonical_model(&self) -> &str {
        &self.canonical_model
    }

    pub const fn task(&self) -> Task {
        self.task
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub const fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub const fn compute_capability(&self) -> (u16, u16) {
        self.compute_capability
    }
}

const PRIVATE_ACTIVATION_COVERAGE: [H3FactoryActivationPrerequisite; 9] = [
    H3FactoryActivationPrerequisite::OpenedComponentMemoryEvidence,
    H3FactoryActivationPrerequisite::PreparedCheckpointExecutionEcho,
    H3FactoryActivationPrerequisite::ConsumingTargetLifetimeTransitions,
    H3FactoryActivationPrerequisite::RetainedTensorOverlapBudget,
    H3FactoryActivationPrerequisite::HostLayoutAndTransientBudget,
    H3FactoryActivationPrerequisite::EndpointPreprocessTransientBudget,
    H3FactoryActivationPrerequisite::PerAttemptRuntimeConstruction,
    H3FactoryActivationPrerequisite::OneShotSchedulerLease,
    H3FactoryActivationPrerequisite::SameAttemptCancellationCoverage,
];

/// Exact request/preprocessing envelope required from one reviewed campaign.
///
/// The first quality runtime record can qualify only the compact conditioned
/// route selected by the released Comfy workflow. Larger canvases, longer
/// clips, additional endpoints, and different prepared sequences require
/// their own reviewed campaign instead of inheriting these bounds.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3PrivateRuntimeEnvelopeRecord {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    pub(crate) fps: u32,
    pub(crate) batch_size: u32,
    pub(crate) max_steps: u32,
    pub(crate) endpoint_count: u32,
    pub(crate) endpoint_anchor: String,
    pub(crate) max_qwen_output_text_rows: u64,
    pub(crate) max_qwen_vision_rows: u64,
    pub(crate) max_condition_visual_rows: u64,
    pub(crate) max_target_video_rows: u64,
    pub(crate) max_target_audio_rows: u64,
    pub(crate) max_total_packed_rows: u64,
}

impl H3PrivateRuntimeEnvelopeRecord {
    /// Validate the reviewed compact-quality envelope.
    ///
    /// Kept as the no-adapter form so every existing caller is unchanged: the
    /// step count must be exactly [`contract::COMFY_DEFAULT_STEPS`].
    pub(crate) fn validate(&self) -> Result<()> {
        self.validate_with_reviewed_steps(None)
    }

    /// Validate the envelope against an optional authenticated Turbo adapter.
    ///
    /// The reviewed canvas is identical either way — same 1344x768, 124 frames,
    /// 24 fps, one first-frame endpoint. The ONLY axis a Turbo tier moves is
    /// the step count, and it may move it only to the count that tier's
    /// distillation was reviewed for. Without an adapter the 21-step pin is
    /// exactly as strict as before.
    pub(crate) fn validate_with_reviewed_steps(&self, turbo_steps: Option<u32>) -> Result<()> {
        let reviewed_steps = turbo_steps.unwrap_or(contract::COMFY_DEFAULT_STEPS);
        if self.max_steps != reviewed_steps {
            bail!(
                "private H3 runtime qualification envelope allows {reviewed_steps} steps, not {}",
                self.max_steps
            )
        }
        self.validate_shape()
    }

    fn validate_shape(&self) -> Result<()> {
        if self.width != contract::DEFAULT_WIDTH
            || self.height != contract::DEFAULT_HEIGHT
            || self.frames != contract::MIN_FRAMES
            || self.fps != contract::FIXED_FPS
            || self.batch_size != 1
            // The step axis is owned by `validate_with_turbo`, which is the
            // only place a reviewed Turbo tier may move it.
            || self.endpoint_count != 1
            || self.endpoint_anchor != "first"
            || [
                self.max_qwen_output_text_rows,
                self.max_qwen_vision_rows,
                self.max_condition_visual_rows,
                self.max_target_video_rows,
                self.max_target_audio_rows,
                self.max_total_packed_rows,
            ]
            .contains(&0)
        {
            bail!("private H3 runtime qualification has an invalid compact-quality envelope")
        }
        Ok(())
    }

    #[cfg(feature = "mp4")]
    fn validate_prepared(&self, request: &H3FactoryPreparedRequestInput) -> Result<()> {
        self.validate_prepared_with_reviewed_steps(request, None)
    }

    #[cfg(feature = "mp4")]
    fn validate_prepared_with_reviewed_steps(
        &self,
        request: &H3FactoryPreparedRequestInput,
        turbo_steps: Option<u32>,
    ) -> Result<()> {
        self.validate_with_reviewed_steps(turbo_steps)?;
        let endpoint = request.endpoints.first();
        if request.width != self.width
            || request.height != self.height
            || request.frames != self.frames
            || request.fps != self.fps
            || request.batch_size != self.batch_size
            || request.grid_points != self.max_steps
            || request.mode != Mode::FirstFrameToAudioVideo
            || !request.synchronized_audio
            || !request.mp4_output
            || request.endpoints.len() != usize::try_from(self.endpoint_count)?
            || !endpoint.is_some_and(|endpoint| endpoint.anchor == H3FactoryEndpointAnchor::First)
            || request.rows.qwen_output_text_rows > self.max_qwen_output_text_rows
            || request.rows.qwen_vision_rows > self.max_qwen_vision_rows
            || request.rows.condition_visual_rows > self.max_condition_visual_rows
            || request.rows.condition_audio_rows != 0
            || request.rows.target_video_rows > self.max_target_video_rows
            || request.rows.target_audio_rows > self.max_target_audio_rows
            || request.rows.total_packed_rows > self.max_total_packed_rows
        {
            bail!("private H3 request differs from the reviewed compact-quality envelope")
        }
        Ok(())
    }

    fn update_identity(&self, digest: &mut Sha256) {
        for value in [
            u64::from(self.width),
            u64::from(self.height),
            u64::from(self.frames),
            u64::from(self.fps),
            u64::from(self.batch_size),
            u64::from(self.max_steps),
            u64::from(self.endpoint_count),
            self.max_qwen_output_text_rows,
            self.max_qwen_vision_rows,
            self.max_condition_visual_rows,
            self.max_target_video_rows,
            self.max_target_audio_rows,
            self.max_total_packed_rows,
        ] {
            digest.update(value.to_le_bytes());
        }
        update_string(digest, &self.endpoint_anchor);
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3PrivateRuntimeBoundRecord {
    pub(crate) fixed_runtime_host_bytes: u64,
    pub(crate) fixed_runtime_device_bytes: u64,
    pub(crate) qwen_activation_workspace_bytes: u64,
    pub(crate) vae_construction_device_workspace_bytes: u64,
    pub(crate) condition_vae_workspace_device_bytes: u64,
    pub(crate) attention_workspace_device_bytes: u64,
    pub(crate) ffn_workspace_device_bytes: u64,
    pub(crate) decoder_tile_workspace_device_bytes: u64,
    pub(crate) audio_decode_workspace_device_bytes: u64,
    pub(crate) encoded_video_host_bytes_bound: u64,
    pub(crate) thumbnail_host_bytes_bound: u64,
    pub(crate) mux_output_host_bytes_bound: u64,
    pub(crate) aac_mux_staging_host_bytes: u64,
}

/// Device floor evaluated before any bulk artifact I/O.
///
/// Both terms appear in `denoise_phase_device_bytes`, and the predicted device
/// peak is the max over every phase, so this is a provable lower bound of the
/// exact figure the admission check compares. Every other term in that sum is
/// non-negative and request-derived, and is deliberately left out: the point is
/// to decline an impossible device in milliseconds rather than after hashing
/// ~37 GB of weights, never to make a decision the exact check would not.
#[cfg(feature = "mp4")]
pub(crate) fn private_h3_admission_device_floor_bytes(
    bounds: &H3PrivateRuntimeBoundRecord,
) -> Result<u64> {
    bounds
        .fixed_runtime_device_bytes
        .checked_add(crate::h3_factory::denoise_transient_workspace_device_bytes(
            bounds.attention_workspace_device_bytes,
            bounds.ffn_workspace_device_bytes,
        ))
        .ok_or_else(|| anyhow!("private H3 admission device floor overflow"))
}

/// Host floor evaluated before any bulk artifact I/O.
///
/// The Qwen phases charge `fixed_runtime_host_bytes` plus the conditioner's
/// host-resident parameters. Which placement admission picks is not known yet,
/// so this takes the smaller of the two published residencies — the floor must
/// hold whichever way placement goes. Every other term in that phase sum is
/// non-negative and omitted for the same reason as above.
#[cfg(feature = "mp4")]
pub(crate) fn private_h3_admission_host_floor_bytes(
    bounds: &H3PrivateRuntimeBoundRecord,
) -> Result<u64> {
    let cpu =
        released_h3_private_qwen_loader_memory_authority(H3PrivateQwenLoaderMemoryRoute::Cpu)?
            .host_resident_parameter_bytes;
    let accelerated =
        released_h3_private_qwen_loader_memory_authority(H3PrivateQwenLoaderMemoryRoute::Cuda)?
            .host_resident_parameter_bytes;
    bounds
        .fixed_runtime_host_bytes
        .checked_add(cpu.min(accelerated))
        .ok_or_else(|| anyhow!("private H3 admission host floor overflow"))
}

/// Refuse an attempt whose capacity sample cannot hold the floors above.
///
/// This runs before the artifact SHA-256 pass. It is one-directional by
/// construction: a refusal here implies the exact per-attempt check would also
/// refuse, so hoisting it can only turn a slow refusal into a fast one.
#[cfg(feature = "mp4")]
fn precheck_private_h3_admission_capacity(
    bounds: &H3PrivateRuntimeBoundRecord,
    available_device_bytes: u64,
    available_host_headroom_bytes: u64,
) -> Result<()> {
    let device_floor = private_h3_admission_device_floor_bytes(bounds)?;
    let host_floor = private_h3_admission_host_floor_bytes(bounds)?;
    if device_floor > available_device_bytes || host_floor > available_host_headroom_bytes {
        bail!(
            "private H3 admission needs at least {device_floor} device and {host_floor} host \
             bytes before any request-specific term, exceeding the {available_device_bytes} \
             device and {available_host_headroom_bytes} host admission sample"
        )
    }
    Ok(())
}

impl H3PrivateRuntimeBoundRecord {
    fn into_authority(self) -> H3PrivateQualifiedRuntimeBounds {
        H3PrivateQualifiedRuntimeBounds {
            fixed_runtime_host_bytes: self.fixed_runtime_host_bytes,
            fixed_runtime_device_bytes: self.fixed_runtime_device_bytes,
            qwen_activation_workspace_bytes: self.qwen_activation_workspace_bytes,
            vae_construction_device_workspace_bytes: self.vae_construction_device_workspace_bytes,
            condition_vae_workspace_device_bytes: self.condition_vae_workspace_device_bytes,
            attention_workspace_device_bytes: self.attention_workspace_device_bytes,
            ffn_workspace_device_bytes: self.ffn_workspace_device_bytes,
            decoder_tile_workspace_device_bytes: self.decoder_tile_workspace_device_bytes,
            audio_decode_workspace_device_bytes: self.audio_decode_workspace_device_bytes,
            encoded_video_host_bytes_bound: self.encoded_video_host_bytes_bound,
            thumbnail_host_bytes_bound: self.thumbnail_host_bytes_bound,
            mux_output_host_bytes_bound: self.mux_output_host_bytes_bound,
            aac_mux_staging_host_bytes: self.aac_mux_staging_host_bytes,
        }
    }

    fn update_identity(&self, digest: &mut Sha256) {
        for value in [
            self.fixed_runtime_host_bytes,
            self.fixed_runtime_device_bytes,
            self.qwen_activation_workspace_bytes,
            self.vae_construction_device_workspace_bytes,
            self.condition_vae_workspace_device_bytes,
            self.attention_workspace_device_bytes,
            self.ffn_workspace_device_bytes,
            self.decoder_tile_workspace_device_bytes,
            self.audio_decode_workspace_device_bytes,
            self.encoded_video_host_bytes_bound,
            self.thumbnail_host_bytes_bound,
            self.mux_output_host_bytes_bound,
            self.aac_mux_staging_host_bytes,
        ] {
            digest.update(value.to_le_bytes());
        }
    }

    fn validate(&self) -> Result<()> {
        if [
            self.fixed_runtime_host_bytes,
            self.fixed_runtime_device_bytes,
            self.qwen_activation_workspace_bytes,
            self.vae_construction_device_workspace_bytes,
            self.condition_vae_workspace_device_bytes,
            self.attention_workspace_device_bytes,
            self.ffn_workspace_device_bytes,
            self.decoder_tile_workspace_device_bytes,
            self.audio_decode_workspace_device_bytes,
            self.encoded_video_host_bytes_bound,
            self.thumbnail_host_bytes_bound,
            self.mux_output_host_bytes_bound,
            self.aac_mux_staging_host_bytes,
        ]
        .contains(&0)
        {
            bail!("private H3 runtime qualification bounds must all be nonzero")
        }
        if self.encoded_video_host_bytes_bound
            != super::pipeline::SMALL_ENCODED_VIDEO_HOST_BYTES_BOUND
            || self.thumbnail_host_bytes_bound != super::pipeline::SMALL_THUMBNAIL_HOST_BYTES_BOUND
            || self.mux_output_host_bytes_bound
                != super::pipeline::SMALL_MUX_OUTPUT_HOST_BYTES_BOUND
            || self.aac_mux_staging_host_bytes != super::pipeline::SMALL_AAC_MUX_STAGING_HOST_BYTES
        {
            bail!("private H3 runtime qualification media bounds must match enforced caps")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3PrivateRuntimeEvidenceArtifact {
    pub(crate) relative_path: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3PrivateRuntimeQualificationRecord {
    pub(crate) schema: String,
    pub(crate) decision: String,
    pub(crate) canonical_model: String,
    pub(crate) task: String,
    pub(crate) campaign_source_sha: String,
    pub(crate) campaign_runtime_code_identity_sha256: String,
    pub(crate) campaign_bootstrap_record_sha256: String,
    pub(crate) campaign_bootstrap_identity_sha256: String,
    pub(crate) measured_server_executable_relative_path: String,
    pub(crate) measured_server_executable_sha256: String,
    pub(crate) authorization_record_sha256: String,
    pub(crate) authorization_source_document_sha256: String,
    pub(crate) artifact_qualification_identity_sha256: String,
    pub(crate) artifact_total_bytes: u64,
    pub(crate) device_id: String,
    pub(crate) device_ordinal: usize,
    pub(crate) compute_capability: [u16; 2],
    pub(crate) attention_runtime_identity_sha256: String,
    pub(crate) attention_kernel_identity: String,
    pub(crate) attention_qualification_sha256: String,
    pub(crate) campaign_process: H3PrivateRuntimeProcessObservation,
    pub(crate) envelope: H3PrivateRuntimeEnvelopeRecord,
    pub(crate) bounds: H3PrivateRuntimeBoundRecord,
    pub(crate) evidence_artifacts: Vec<H3PrivateRuntimeEvidenceArtifact>,
    pub(crate) identity_sha256: String,
}

/// Payload-free identities expected by one private owner attempt.
///
/// This type is cloneable because it is only a comparison record. The opaque
/// preparation and every authority root that can execute it remain non-Clone.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivateFl2VaAttemptFacts {
    pub device_id: String,
    pub device_ordinal: usize,
    pub execution_fingerprint: String,
    pub prepared_attempt_identity_sha256: String,
    pub target_budget_identity_sha256: String,
    pub component_set_identity_sha256: String,
    pub predicted_device_peak_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub media: H3PrivateFl2VaMediaContract,
    admission_evidence_identity_sha256: String,
    artifact_qualification_identity_sha256: String,
    runtime_qualification_identity_sha256: String,
    work_identity_sha256: String,
    cancellation_scope_identity_sha256: String,
    memory_ledger_sequence: u64,
    consumption_identity_sha256: String,
}

impl H3PrivateFl2VaAttemptFacts {
    pub fn admission_evidence_identity_sha256(&self) -> &str {
        &self.admission_evidence_identity_sha256
    }

    pub fn artifact_qualification_identity_sha256(&self) -> &str {
        &self.artifact_qualification_identity_sha256
    }

    pub fn runtime_qualification_identity_sha256(&self) -> &str {
        &self.runtime_qualification_identity_sha256
    }

    pub fn work_identity_sha256(&self) -> &str {
        &self.work_identity_sha256
    }

    pub fn cancellation_scope_identity_sha256(&self) -> &str {
        &self.cancellation_scope_identity_sha256
    }

    pub const fn memory_ledger_sequence(&self) -> u64 {
        self.memory_ledger_sequence
    }

    pub fn consumption_identity_sha256(&self) -> &str {
        &self.consumption_identity_sha256
    }

    fn validate_against_binding(&self, binding: &H3PrivateAttemptConsumptionBinding) -> Result<()> {
        self.media.validate()?;
        if self.execution_fingerprint != binding.execution_fingerprint
            || self.prepared_attempt_identity_sha256 != binding.prepared_attempt_identity_sha256
            || self.target_budget_identity_sha256 != binding.target_budget_identity_sha256
            || self.component_set_identity_sha256 != binding.component_set_identity_sha256
            || self.admission_evidence_identity_sha256 != binding.admission_evidence_identity_sha256
            || self.artifact_qualification_identity_sha256
                != binding.artifact_qualification_identity_sha256
            || self.runtime_qualification_identity_sha256
                != binding.runtime_qualification_identity_sha256
            || self.work_identity_sha256 != binding.work_identity_sha256
            || self.cancellation_scope_identity_sha256 != binding.cancellation_scope_identity_sha256
            || self.memory_ledger_sequence != binding.memory_ledger_sequence
            || self.consumption_identity_sha256 != binding.identity_sha256
        {
            bail!("private H3 public attempt facts changed from the consumption binding")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivateFl2VaMediaContract {
    pub canonical_model: String,
    pub task: Task,
    pub mode: Mode,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
}

impl H3PrivateFl2VaMediaContract {
    fn validate(&self) -> Result<()> {
        if self.canonical_model != contract::FL2VA_COMFY
            || self.task != Task::Fl2va
            || self.mode == Mode::ReferenceToAudioVideo
            || self.width == 0
            || self.height == 0
            || self.frames == 0
            || self.fps != contract::FIXED_FPS
        {
            bail!("private H3 frozen media contract is not exact FL2VA")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivateFl2VaTerminalIdentityEcho {
    pub device_id: String,
    pub device_ordinal: usize,
    pub execution_fingerprint: String,
    pub prepared_attempt_identity_sha256: String,
    pub target_budget_identity_sha256: String,
    pub component_set_identity_sha256: String,
    pub media: H3PrivateFl2VaMediaContract,
    pub duration_ms: u64,
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
    pub synchronized_audio_video: bool,
    pub pipeline_provenance_sha256: String,
    pub admission_evidence_identity_sha256: String,
    pub artifact_qualification_identity_sha256: String,
    pub runtime_qualification_identity_sha256: String,
    pub consumption_identity_sha256: String,
}

#[derive(Debug)]
pub struct H3PrivateFl2VaRunOutput {
    pub response: GenerateResponse,
    pub identity_echo: H3PrivateFl2VaTerminalIdentityEcho,
}

#[derive(Debug, thiserror::Error)]
pub enum H3PrivateFl2VaPrepareError {
    #[error("private H3 runtime has no reviewed runtime qualification")]
    MissingReviewedRuntimeQualification,
    #[error("private H3 preparation evidence was rejected: {0}")]
    InvalidEvidence(String),
}

/// Complete input for the private production preparation seam. Paths remain
/// borrowed only for the duration of preparation; the returned attempt owns
/// every opened descriptor and prepared tensor it needs.
pub struct H3PrivateFl2VaPrepareInput<'a> {
    pub request: &'a GenerateRequest,
    pub frozen_factory: &'a FrozenH3FactoryAuthority,
    pub admission_evidence: &'a H3PrivateFl2VaAdmissionEvidence,
    pub paths: H3PrivateFl2VaUatPaths<'a>,
    pub owner_fence: H3PrivateFl2VaOwnerFenceFacts,
}

/// Canonical private-UAT filesystem inputs. Preparation resolves every model
/// component from the pinned compact manifest below `models_root`; callers cannot
/// supply individual checkpoint paths.
#[derive(Clone, Copy, Debug)]
pub struct H3PrivateFl2VaUatPaths<'a> {
    pub models_root: &'a Path,
    pub staging_root: &'a Path,
    pub authorization_record: &'a Path,
    pub runtime_qualification_record: &'a Path,
}

/// Inputs for allocation-free private FL2VA admission.
///
/// Capacity values are comparison facts only: they never alter the canonical
/// target budget. The returned evidence remains valid while a later capacity
/// sample is still at least the exact admitted peaks.
pub struct H3PrivateFl2VaAdmissionInput<'a> {
    pub request: &'a GenerateRequest,
    pub paths: H3PrivateFl2VaUatPaths<'a>,
    pub device_id: &'a str,
    pub device_ordinal: usize,
    pub compute_capability: (u16, u16),
    pub available_device_bytes: u64,
    /// Host RAM available after the server's canonical 15%-or-8-GiB safety
    /// floor, never the raw operating-system available-memory sample.
    pub available_host_headroom_bytes: u64,
}

/// Payload-free projection of the thirteen independently reviewed runtime
/// bounds used by the sole private FL2VA budget builder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivateFl2VaRuntimeBounds {
    fixed_runtime_host_bytes: u64,
    fixed_runtime_device_bytes: u64,
    qwen_activation_workspace_bytes: u64,
    vae_construction_device_workspace_bytes: u64,
    condition_vae_workspace_device_bytes: u64,
    attention_workspace_device_bytes: u64,
    ffn_workspace_device_bytes: u64,
    decoder_tile_workspace_device_bytes: u64,
    audio_decode_workspace_device_bytes: u64,
    encoded_video_host_bytes_bound: u64,
    thumbnail_host_bytes_bound: u64,
    mux_output_host_bytes_bound: u64,
    aac_mux_staging_host_bytes: u64,
}

impl H3PrivateFl2VaRuntimeBounds {
    pub const fn fixed_runtime_host_bytes(&self) -> u64 {
        self.fixed_runtime_host_bytes
    }

    pub const fn fixed_runtime_device_bytes(&self) -> u64 {
        self.fixed_runtime_device_bytes
    }

    pub const fn qwen_activation_workspace_bytes(&self) -> u64 {
        self.qwen_activation_workspace_bytes
    }

    pub const fn vae_construction_device_workspace_bytes(&self) -> u64 {
        self.vae_construction_device_workspace_bytes
    }

    pub const fn condition_vae_workspace_device_bytes(&self) -> u64 {
        self.condition_vae_workspace_device_bytes
    }

    pub const fn attention_workspace_device_bytes(&self) -> u64 {
        self.attention_workspace_device_bytes
    }

    pub const fn ffn_workspace_device_bytes(&self) -> u64 {
        self.ffn_workspace_device_bytes
    }

    pub const fn decoder_tile_workspace_device_bytes(&self) -> u64 {
        self.decoder_tile_workspace_device_bytes
    }

    pub const fn audio_decode_workspace_device_bytes(&self) -> u64 {
        self.audio_decode_workspace_device_bytes
    }

    pub const fn encoded_video_host_bytes_bound(&self) -> u64 {
        self.encoded_video_host_bytes_bound
    }

    pub const fn thumbnail_host_bytes_bound(&self) -> u64 {
        self.thumbnail_host_bytes_bound
    }

    pub const fn mux_output_host_bytes_bound(&self) -> u64 {
        self.mux_output_host_bytes_bound
    }

    pub const fn aac_mux_staging_host_bytes(&self) -> u64 {
        self.aac_mux_staging_host_bytes
    }
}

impl From<&H3PrivateQualifiedRuntimeBounds> for H3PrivateFl2VaRuntimeBounds {
    fn from(bounds: &H3PrivateQualifiedRuntimeBounds) -> Self {
        Self {
            fixed_runtime_host_bytes: bounds.fixed_runtime_host_bytes,
            fixed_runtime_device_bytes: bounds.fixed_runtime_device_bytes,
            qwen_activation_workspace_bytes: bounds.qwen_activation_workspace_bytes,
            vae_construction_device_workspace_bytes: bounds.vae_construction_device_workspace_bytes,
            condition_vae_workspace_device_bytes: bounds.condition_vae_workspace_device_bytes,
            attention_workspace_device_bytes: bounds.attention_workspace_device_bytes,
            ffn_workspace_device_bytes: bounds.ffn_workspace_device_bytes,
            decoder_tile_workspace_device_bytes: bounds.decoder_tile_workspace_device_bytes,
            audio_decode_workspace_device_bytes: bounds.audio_decode_workspace_device_bytes,
            encoded_video_host_bytes_bound: bounds.encoded_video_host_bytes_bound,
            thumbnail_host_bytes_bound: bounds.thumbnail_host_bytes_bound,
            mux_output_host_bytes_bound: bounds.mux_output_host_bytes_bound,
            aac_mux_staging_host_bytes: bounds.aac_mux_staging_host_bytes,
        }
    }
}

/// Immutable payload-free result of exact private FL2VA admission.
///
/// All fields stay private so request, route, identity, and budget facts
/// cannot be independently rewritten after the cross-bind succeeds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivateFl2VaAdmissionEvidence {
    identity_sha256: String,
    submitted_request_identity_sha256: String,
    resolved_request_identity_sha256: String,
    canonical_model: String,
    task: Task,
    mode: Mode,
    device_id: String,
    device_ordinal: usize,
    compute_capability: (u16, u16),
    admitted_available_device_bytes: u64,
    admitted_host_headroom_bytes: u64,
    base_factory_authority: FrozenH3FactoryAuthority,
    execution_fingerprint: String,
    component_set_identity_sha256: String,
    prepared_request_identity_sha256: String,
    prepared_attempt_identity_sha256: String,
    target_budget_identity_sha256: String,
    artifact_qualification_identity_sha256: String,
    runtime_qualification_identity_sha256: String,
    predicted_device_peak_bytes: u64,
    predicted_host_increment_bytes: u64,
    seed: u64,
    attention: H3FactoryAttentionInput,
    runtime_bounds: H3PrivateFl2VaRuntimeBounds,
}

impl H3PrivateFl2VaAdmissionEvidence {
    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn canonical_model(&self) -> &str {
        &self.canonical_model
    }

    pub const fn task(&self) -> Task {
        self.task
    }

    pub const fn mode(&self) -> Mode {
        self.mode
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub const fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub const fn compute_capability(&self) -> (u16, u16) {
        self.compute_capability
    }

    pub const fn admitted_available_device_bytes(&self) -> u64 {
        self.admitted_available_device_bytes
    }

    pub const fn admitted_host_headroom_bytes(&self) -> u64 {
        self.admitted_host_headroom_bytes
    }

    pub fn base_factory_authority(&self) -> &FrozenH3FactoryAuthority {
        &self.base_factory_authority
    }

    pub fn execution_fingerprint(&self) -> &str {
        &self.execution_fingerprint
    }

    pub fn component_set_identity_sha256(&self) -> &str {
        &self.component_set_identity_sha256
    }

    pub fn prepared_request_identity_sha256(&self) -> &str {
        &self.prepared_request_identity_sha256
    }

    pub fn prepared_attempt_identity_sha256(&self) -> &str {
        &self.prepared_attempt_identity_sha256
    }

    pub fn target_budget_identity_sha256(&self) -> &str {
        &self.target_budget_identity_sha256
    }

    pub fn artifact_qualification_identity_sha256(&self) -> &str {
        &self.artifact_qualification_identity_sha256
    }

    pub fn runtime_qualification_identity_sha256(&self) -> &str {
        &self.runtime_qualification_identity_sha256
    }

    pub const fn predicted_device_peak_bytes(&self) -> u64 {
        self.predicted_device_peak_bytes
    }

    pub const fn predicted_host_increment_bytes(&self) -> u64 {
        self.predicted_host_increment_bytes
    }

    pub const fn seed(&self) -> u64 {
        self.seed
    }

    pub fn attention(&self) -> &H3FactoryAttentionInput {
        &self.attention
    }

    pub fn runtime_bounds(&self) -> &H3PrivateFl2VaRuntimeBounds {
        &self.runtime_bounds
    }

    /// Return the exact request the owner must retain. An omitted seed is
    /// replaced by the one resolved during preprocessing; every other byte of
    /// the serialized request must still match the admitted submission.
    pub fn resolve_request(&self, request: &GenerateRequest) -> Result<GenerateRequest> {
        let supplied_identity = private_h3_request_identity(request)?;
        if supplied_identity != self.submitted_request_identity_sha256
            && supplied_identity != self.resolved_request_identity_sha256
        {
            bail!("private H3 request changed after allocation-free admission")
        }
        let mut resolved = request.clone();
        match resolved.seed {
            Some(seed) if seed != self.seed => {
                bail!("private H3 request seed differs from allocation-free admission")
            }
            Some(_) => {}
            None => resolved.seed = Some(self.seed),
        }
        if private_h3_request_identity(&resolved)? != self.resolved_request_identity_sha256 {
            bail!("private H3 resolved request differs from allocation-free admission")
        }
        Ok(resolved)
    }

    /// Revalidate this immutable DTO against an exact request and CUDA route.
    /// Current capacity may differ from the admission snapshot, but both
    /// current values must still cover the canonical target peaks.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        request: &GenerateRequest,
        device_id: &str,
        device_ordinal: usize,
        compute_capability: (u16, u16),
        available_device_bytes: u64,
        available_host_headroom_bytes: u64,
    ) -> Result<()> {
        let resolved = self.resolve_request(request)?;
        self.base_factory_authority.validate_engine_seam(
            &self.canonical_model,
            device_ordinal,
            self.base_factory_authority.block_offload(),
        )?;
        let attention_identity = H3AttentionRuntimeAuthority::expected_identity_for(
            self.attention.runtime_backend,
            self.attention.kernel,
            self.attention.activation,
            self.attention.device,
            self.attention.model_contract,
        )
        .map_err(|error| anyhow!(error.to_string()))?;
        if self.canonical_model != contract::FL2VA_COMFY
            || self.task != Task::Fl2va
            || contract::validate_request_contract(&resolved, Task::Fl2va)
                .map_err(|error| anyhow!("{}: {}", error.code, error.message))?
                != self.mode
            || private_h3_request_identity(&resolved)? != self.resolved_request_identity_sha256
            || self.device_id != device_id
            || self.device_ordinal != device_ordinal
            || self.compute_capability != compute_capability
            || self.base_factory_authority.canonical_model() != self.canonical_model
            || self.base_factory_authority.task() != self.task
            || self.base_factory_authority.device_id() != self.device_id
            || self.base_factory_authority.device_ordinal() != self.device_ordinal
            // The public runtime profile is CUDA SM89 only, so a Metal factory
            // authority (which carries no compute capability) can never match
            // here and fails closed without a separate branch.
            || self.base_factory_authority.compute_capability() != Some(self.compute_capability)
            || self.base_factory_authority.execution_fingerprint() != self.execution_fingerprint
            || self.base_factory_authority.component_set_identity_sha256()
                != self.component_set_identity_sha256
            || self
                .base_factory_authority
                .prepared_target_attempt_identities()
                .is_some()
            || !self
                .base_factory_authority
                .attention_runtime_identity_sha256()
                .is_empty()
            || self.attention.runtime_identity_sha256 != attention_identity
            || self.attention.qualification_sha256
                != self.base_factory_authority.attention_qualification_sha256()
            || available_device_bytes < self.predicted_device_peak_bytes
            || available_host_headroom_bytes < self.predicted_host_increment_bytes
            || self.admitted_available_device_bytes < self.predicted_device_peak_bytes
            || self.admitted_host_headroom_bytes < self.predicted_host_increment_bytes
            || !valid_sha256(&self.prepared_request_identity_sha256)
            || !valid_sha256(&self.prepared_attempt_identity_sha256)
            || !valid_sha256(&self.target_budget_identity_sha256)
            || !valid_sha256(&self.artifact_qualification_identity_sha256)
            || !valid_sha256(&self.runtime_qualification_identity_sha256)
            || self.identity_sha256 != private_h3_admission_evidence_identity(self)
        {
            bail!("private H3 allocation-free admission evidence changed or no longer fits")
        }
        Ok(())
    }
}

/// Produce exact, CUDA-allocation-free admission evidence for one private
/// FL2VA request and one concrete CUDA route. Endpoint normalization and noise
/// preparation deliberately remain here and may construct CPU-only Candle
/// tensors; this function never creates a CUDA `Device` or CUDA tensor. It
/// independently checks that reviewed authority exists before inspecting any
/// supplied path.
pub fn prepare_h3_private_fl2va_admission(
    input: H3PrivateFl2VaAdmissionInput<'_>,
    progress: &ProgressReporter,
) -> std::result::Result<H3PrivateFl2VaAdmissionEvidence, H3PrivateFl2VaPrepareError> {
    if !reviewed_h3_private_runtime_available() {
        return Err(H3PrivateFl2VaPrepareError::MissingReviewedRuntimeQualification);
    }
    #[cfg(not(feature = "mp4"))]
    {
        let _ = (input, progress);
        Err(H3PrivateFl2VaPrepareError::InvalidEvidence(
            "private H3 synchronized audio-video requires Mold's mp4 feature".into(),
        ))
    }
    #[cfg(feature = "mp4")]
    {
        prepare_reviewed_h3_private_fl2va_admission(input, progress)
            .map_err(|error| H3PrivateFl2VaPrepareError::InvalidEvidence(error.to_string()))
    }
}

#[cfg(feature = "mp4")]
fn prepare_reviewed_h3_private_fl2va_admission(
    input: H3PrivateFl2VaAdmissionInput<'_>,
    progress: &ProgressReporter,
) -> Result<H3PrivateFl2VaAdmissionEvidence> {
    let H3PrivateFl2VaAdmissionInput {
        request,
        paths,
        device_id,
        device_ordinal,
        compute_capability,
        available_device_bytes,
        available_host_headroom_bytes,
    } = input;
    if device_id.trim().is_empty()
        || compute_capability.0 == 0
        || available_device_bytes == 0
        || available_host_headroom_bytes == 0
    {
        bail!("private H3 admission requires one concrete nonempty CUDA capacity sample")
    }
    let mode = contract::validate_request_contract(request, Task::Fl2va)
        .map_err(|error| anyhow!("{}: {}", error.code, error.message))?;
    let submitted_request_identity_sha256 = private_h3_request_identity(request)?;

    // Private UAT retains its reviewed-record gate before bulk model I/O.
    // Ordinary public H3 derives authority from compiled policy, the exact
    // artifact graph, and the live SM89 attention route instead.
    #[cfg(not(feature = "h3"))]
    let reviewed_runtime_qualification = open_reviewed_h3_private_runtime_qualification(
        paths.runtime_qualification_record,
        REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256,
    )?;
    #[cfg(not(feature = "h3"))]
    reviewed_runtime_qualification.validate_route(device_id, device_ordinal, compute_capability)?;
    #[cfg(not(feature = "h3"))]
    let attention_qualification_sha256 = reviewed_runtime_qualification
        .record
        .attention_qualification_sha256
        .clone();
    // Cheap capacity floors BEFORE the artifact pass below hashes ~37 GB.
    // Refusing a hopeless device after several minutes of SHA-256 was the
    // worst failure this path had; both floors are provable lower bounds of
    // the exact per-attempt budget compared at the end of this function, so
    // this can only ever turn a slow refusal into a fast one.
    #[cfg(feature = "h3")]
    let precheck_bounds = public_runtime_bounds();
    #[cfg(not(feature = "h3"))]
    let precheck_bounds = reviewed_runtime_qualification.record.bounds.clone();
    precheck_private_h3_admission_capacity(
        &precheck_bounds,
        available_device_bytes,
        available_host_headroom_bytes,
    )?;
    let artifact_report = qualify_private_artifacts_with_control(
        paths.models_root,
        contract::FL2VA_COMFY,
        paths.authorization_record,
        |hash| {
            progress.checkpoint()?;
            progress.weight_load(
                H3_ARTIFACT_VERIFICATION_PROGRESS,
                hash.total_bytes_verified,
                hash.total_bytes,
            );
            progress.checkpoint()?;
            Ok(())
        },
    )?;

    let attention_device = H3AttentionDevice::Cuda {
        compute_capability: Some(compute_capability),
    };
    let attention_model = H3AttentionModelContract::released_bf16();
    let attention =
        H3AttentionRuntimeAuthority::qualify_flash_attention_v2(attention_device, attention_model)
            .map_err(|error| anyhow!(error.to_string()))?;
    #[cfg(feature = "h3")]
    let attention_qualification_sha256 = attention.identity_sha256().to_string();
    let attention_input = H3FactoryAttentionInput {
        generic_backend: AttentionBackend::Flash,
        generic_chunk: AttentionChunkPolicy::Off,
        runtime_backend: attention.backend(),
        kernel: attention.kernel(),
        activation: attention.activation(),
        device: attention.device(),
        model_contract: attention.contract(),
        runtime_identity_sha256: attention.identity_sha256().into(),
        qualification_kernel_identity: attention.kernel().identity().into(),
        qualification_sha256: attention_qualification_sha256.clone(),
        full_noncausal: true,
        lossless: true,
    };
    #[cfg(not(feature = "h3"))]
    let runtime_qualification = reviewed_runtime_qualification.authenticate(
        &artifact_report,
        device_id,
        device_ordinal,
        compute_capability,
        attention.identity_sha256(),
        attention.kernel().identity(),
        &attention_qualification_sha256,
    )?;
    #[cfg(feature = "h3")]
    // Authenticating here reads and digests the adapter but allocates no device
    // memory; the deltas are materialized in the transformer-load phase, which
    // is the phase whose budget charges them. It has to happen before the
    // runtime qualification so the envelope can be minted at the tier's own
    // reviewed step count.
    let turbo_adapter = super::turbo::resolve_requested_turbo_authority()?;
    let turbo_steps = turbo_adapter.as_ref().map(|turbo| turbo.grid_points);
    let runtime_qualification = public_runtime_qualification(
        &artifact_report,
        device_id,
        device_ordinal,
        compute_capability,
        attention.identity_sha256(),
        attention.kernel().identity(),
        &attention_qualification_sha256,
        turbo_steps,
    )?;
    progress.checkpoint()?;

    let storage = H3PrivateComfyStorageAuthority::resolve(paths.models_root)?;
    let qwen_support =
        load_qualified_private_qwen_support(paths.models_root, contract::FL2VA_COMFY)?;
    let transformer_cancellation = H3PrivatePreparationCancellation { progress };
    let opened_transformer = open_h3_comfy_published_int8_checkpoint(
        storage.transformer_path(),
        H3TransformerTask::T2VaFl2Va,
        H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
        &transformer_cancellation,
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    let vae_plan = storage.vae_plan(paths.staging_root)?;
    let mut vae_observer = H3PrivatePreparationVaeObserver::new(progress);
    let opened_vae = open_h3_comfy_vae_authority(&vae_plan, &mut vae_observer);
    let opened_vae = vae_observer.finish(opened_vae)?;

    let mut prepare_observer = H3EngineProgressObserver::new(progress);
    let admission_request = prepare_private_fl2va_admission_request(
        request,
        &qwen_support,
        progress,
        &mut prepare_observer,
    )?;
    runtime_qualification.validate_prepared_envelope(&admission_request.request)?;
    let seed = admission_request.seed;
    let mut resolved_request = request.clone();
    resolved_request.seed = Some(seed);
    let resolved_request_identity_sha256 = private_h3_request_identity(&resolved_request)?;

    let qwen_artifact = exact_qualified_qwen_artifact(&artifact_report)?;
    let qwen_header_identity = qwen_artifact
        .header_identity_sha256
        .as_deref()
        .ok_or_else(|| anyhow!("qualified private H3 Qwen artifact has no header identity"))?;
    let qwen_policy_identity = qwen_artifact
        .policy_identity_sha256
        .as_deref()
        .ok_or_else(|| anyhow!("qualified private H3 Qwen artifact has no policy identity"))?;
    let component_digests = private_h3_component_digests(
        &artifact_report,
        runtime_qualification.identity_sha256(),
        &qwen_support,
        &opened_transformer,
        &opened_vae,
        qwen_artifact,
    )?;
    let execution_fingerprint = private_h3_admission_execution_fingerprint(
        &resolved_request_identity_sha256,
        &artifact_report,
        runtime_qualification.identity_sha256(),
        device_id,
        device_ordinal,
        compute_capability,
        attention.identity_sha256(),
        &component_digests,
    );
    let qwen_memory =
        released_h3_private_qwen_loader_memory_authority(H3PrivateQwenLoaderMemoryRoute::Cpu)?;
    let transformer_policy_sha256 = opened_transformer
        .candidate()
        .strategy
        .quantization_policy
        .policy_sha256
        .clone();
    let pruned_adaln_table_sha256 = private_h3_pruned_adaln_identity(&opened_transformer)?;
    let components = component_digests
        .iter()
        .map(|component| {
            H3FactoryComponentAuthority::new(
                component.role,
                component.content_sha256.clone(),
                component.validation_sha256.clone(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let base_factory_authority =
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: contract::FL2VA_COMFY.into(),
            device_id: device_id.into(),
            device_ordinal,
            compute_capability,
            execution_fingerprint: execution_fingerprint.clone(),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: qwen_memory.source_parameter_bytes,
            qwen_host_resident_parameter_bytes: qwen_memory.host_resident_parameter_bytes,
            qwen_device_resident_parameter_bytes: qwen_memory.device_resident_parameter_bytes,
            qwen_activation_workspace_bytes: runtime_qualification
                .bounds()
                .qwen_activation_workspace_bytes,
            qwen_maximum_tensor_staging_bytes: qwen_memory.maximum_tensor_staging_bytes,
            qwen_retained_raw_header_bytes: qwen_memory.retained_raw_header_bytes,
            qwen_output_text_rows: admission_request.request.rows.qwen_output_text_rows,
            qwen_vision_rows: admission_request.request.rows.qwen_vision_rows,
            condition_visual_rows: admission_request.request.rows.condition_visual_rows,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: attention.kernel().identity().into(),
            attention_qualification_sha256: attention_qualification_sha256.clone(),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: u32::try_from(attention_model.heads)?,
            attention_head_dim: u32::try_from(attention_model.head_dim)?,
            attention_runtime: None,
            block_offload: true,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256,
                qwen_policy_sha256: qwen_policy_identity.into(),
                pruned_adaln_table_sha256,
                turbo_adapter: turbo_adapter.clone(),
            },
            prepared_attempt: None,
            execution_budget_echo: None,
            components,
        })?;

    let qwen_open_route = H3PrivateQwenOpenRouteAuthority::capture(
        &base_factory_authority,
        &qwen_support,
        device_id,
        device_ordinal,
        compute_capability,
        &qwen_artifact.sha256,
        qwen_header_identity,
        qwen_policy_identity,
    )?;
    let mut checkpoint = H3PrivatePreparationCheckpoint { progress };
    let opened_qwen = open_authorized_private_qwen_authority(
        storage.qwen_weights_path(),
        &qwen_support,
        &qwen_open_route,
        &base_factory_authority,
        &mut checkpoint,
    )?;
    let prepared_attempt = build_private_fl2va_admission_attempt(
        &execution_fingerprint,
        admission_request.request,
        &storage,
        &qwen_support,
        &opened_transformer,
        &opened_qwen,
        &opened_vae,
        runtime_qualification.bounds(),
        turbo_adapter.as_ref(),
    )?;
    let predicted_device_peak_bytes = prepared_attempt.target_budget.predicted_device_peak_bytes;
    let predicted_host_increment_bytes = prepared_attempt
        .target_budget
        .predicted_host_increment_bytes;
    if predicted_device_peak_bytes > available_device_bytes
        || predicted_host_increment_bytes > available_host_headroom_bytes
    {
        bail!(
            "private H3 canonical target needs {predicted_device_peak_bytes} device and \
             {predicted_host_increment_bytes} host bytes, exceeding the admission sample"
        )
    }
    let budget_echo = H3FactoryExecutionBudgetEchoInput {
        prepared_attempt_identity_sha256: prepared_attempt.identity_sha256.clone(),
        device_peak_bytes: predicted_device_peak_bytes,
        host_increment_bytes: predicted_host_increment_bytes,
    };
    let enriched = base_factory_authority.with_private_prepared_attempt(
        prepared_attempt.clone(),
        budget_echo,
        attention_input.clone(),
    )?;
    let enriched_identities = enriched
        .prepared_target_attempt_identities()
        .ok_or_else(|| anyhow!("private H3 admission did not bind the prepared target triad"))?;
    if enriched_identities.0 != prepared_attempt.identity_sha256
        || enriched_identities.1 != prepared_attempt.target_budget.identity_sha256
    {
        bail!("private H3 admission changed the canonical prepared target identities")
    }

    let mut evidence = H3PrivateFl2VaAdmissionEvidence {
        identity_sha256: String::new(),
        submitted_request_identity_sha256,
        resolved_request_identity_sha256,
        canonical_model: contract::FL2VA_COMFY.into(),
        task: Task::Fl2va,
        mode,
        device_id: device_id.into(),
        device_ordinal,
        compute_capability,
        admitted_available_device_bytes: available_device_bytes,
        admitted_host_headroom_bytes: available_host_headroom_bytes,
        component_set_identity_sha256: base_factory_authority
            .component_set_identity_sha256()
            .into(),
        prepared_request_identity_sha256: prepared_attempt.request.identity_sha256.clone(),
        prepared_attempt_identity_sha256: prepared_attempt.identity_sha256,
        target_budget_identity_sha256: prepared_attempt.target_budget.identity_sha256,
        artifact_qualification_identity_sha256: artifact_report
            .qualification_identity_sha256
            .clone(),
        runtime_qualification_identity_sha256: runtime_qualification.identity_sha256().into(),
        predicted_device_peak_bytes,
        predicted_host_increment_bytes,
        seed,
        attention: attention_input,
        runtime_bounds: H3PrivateFl2VaRuntimeBounds::from(runtime_qualification.bounds()),
        base_factory_authority,
        execution_fingerprint,
    };
    evidence.identity_sha256 = private_h3_admission_evidence_identity(&evidence);
    evidence.validate_for(
        &resolved_request,
        device_id,
        device_ordinal,
        compute_capability,
        available_device_bytes,
        available_host_headroom_bytes,
    )?;
    Ok(evidence)
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct H3PrivateComponentDigest {
    role: H3FactoryComponentRole,
    content_sha256: String,
    validation_sha256: String,
}

fn private_h3_request_identity(request: &GenerateRequest) -> Result<String> {
    let bytes =
        serde_json::to_vec(request).context("failed to serialize exact private H3 request")?;
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-admission-request.v1\0");
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(feature = "mp4")]
fn private_h3_component_digests(
    report: &H3PrivateArtifactQualificationReport,
    runtime_qualification_identity_sha256: &str,
    support: &H3PrivateQwenSupport,
    transformer: &H3ComfyOpenedInt8Checkpoint,
    vae: &H3AuthenticatedComfyVaeAuthority,
    qwen_artifact: &H3QualifiedArtifact,
) -> Result<Vec<H3PrivateComponentDigest>> {
    require_private_sha256(
        runtime_qualification_identity_sha256,
        "private H3 runtime qualification",
    )?;
    support.revalidate()?;
    transformer
        .revalidate()
        .map_err(|error| anyhow!(error.to_string()))?;
    vae.validate().map_err(|error| anyhow!(error.to_string()))?;
    let mut grouped = BTreeMap::<H3FactoryComponentRole, Vec<&H3QualifiedArtifact>>::new();
    for artifact in &report.artifacts {
        grouped
            .entry(private_h3_component_role(artifact))
            .or_default()
            .push(artifact);
    }
    let mut result = Vec::with_capacity(4);
    for role in [
        H3FactoryComponentRole::Conditioner,
        H3FactoryComponentRole::Transformer,
        H3FactoryComponentRole::VisualVae,
        H3FactoryComponentRole::AudioVae,
    ] {
        let members = grouped
            .get_mut(&role)
            .ok_or_else(|| anyhow!("private H3 logical component {role:?} has no artifacts"))?;
        members.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        let role_id = private_h3_component_role_id(role);
        let mut content = Sha256::new();
        content.update(b"mold.minimax-h3.private-logical-component-content.v1\0");
        update_string(&mut content, role_id);
        content.update((members.len() as u64).to_le_bytes());
        for artifact in members.iter() {
            update_private_h3_qualified_artifact(&mut content, artifact);
        }
        let content_sha256 = format!("{:x}", content.finalize());

        let mut validation = Sha256::new();
        validation.update(b"mold.minimax-h3.private-logical-component-validation.v1\0");
        for value in [
            role_id,
            report.qualification_identity_sha256.as_str(),
            runtime_qualification_identity_sha256,
            content_sha256.as_str(),
        ] {
            update_string(&mut validation, value);
        }
        match role {
            H3FactoryComponentRole::Conditioner => {
                for value in [
                    support.support_identity_sha256(),
                    qwen_artifact.sha256.as_str(),
                    qwen_artifact
                        .header_identity_sha256
                        .as_deref()
                        .ok_or_else(|| anyhow!("private H3 Qwen header identity is absent"))?,
                    qwen_artifact
                        .policy_identity_sha256
                        .as_deref()
                        .ok_or_else(|| anyhow!("private H3 Qwen policy identity is absent"))?,
                ] {
                    update_string(&mut validation, value);
                }
            }
            H3FactoryComponentRole::Transformer => {
                for value in [
                    transformer.checkpoint_identity_sha256(),
                    transformer.memory_evidence().identity_sha256.as_str(),
                    transformer.content_sha256(),
                    transformer.candidate().header_identity_sha256.as_str(),
                    transformer
                        .candidate()
                        .strategy
                        .quantization_policy
                        .policy_sha256
                        .as_str(),
                ] {
                    update_string(&mut validation, value);
                }
            }
            H3FactoryComponentRole::VisualVae | H3FactoryComponentRole::AudioVae => {
                let binding = private_h3_vae_artifact_binding_identity(
                    vae.artifact_validation_identity_sha256(),
                    vae.artifact_plan_identity_sha256(),
                )?;
                update_string(&mut validation, &binding);
            }
        }
        result.push(H3PrivateComponentDigest {
            role,
            content_sha256,
            validation_sha256: format!("{:x}", validation.finalize()),
        });
    }
    Ok(result)
}

#[cfg(feature = "mp4")]
fn private_h3_vae_artifact_binding_identity(
    artifact_validation_identity_sha256: &str,
    artifact_plan_identity_sha256: &str,
) -> Result<String> {
    require_private_sha256(
        artifact_validation_identity_sha256,
        "private H3 VAE artifact validation",
    )?;
    require_private_sha256(
        artifact_plan_identity_sha256,
        "private H3 VAE artifact plan",
    )?;
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-vae-artifact-binding.v1\0");
    update_string(&mut digest, artifact_validation_identity_sha256);
    update_string(&mut digest, artifact_plan_identity_sha256);
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(feature = "mp4")]
fn private_h3_component_digest(
    components: &[H3PrivateComponentDigest],
    role: H3FactoryComponentRole,
) -> Result<&H3PrivateComponentDigest> {
    let mut matches = components.iter().filter(|component| component.role == role);
    let component = matches
        .next()
        .ok_or_else(|| anyhow!("private H3 owner reopened no {role:?} component authority"))?;
    if matches.next().is_some() {
        bail!("private H3 owner reopened duplicate {role:?} component authorities")
    }
    Ok(component)
}

#[cfg(feature = "mp4")]
fn validate_owner_component_digests(
    factory: &FrozenH3FactoryAuthority,
    components: &[H3PrivateComponentDigest],
) -> Result<()> {
    if components.len() != 4 {
        bail!("private H3 owner reopened an incomplete component set")
    }
    for role in [
        H3FactoryComponentRole::Conditioner,
        H3FactoryComponentRole::Transformer,
        H3FactoryComponentRole::VisualVae,
        H3FactoryComponentRole::AudioVae,
    ] {
        let current = private_h3_component_digest(components, role)?;
        let admitted = factory.private_component_authority(role);
        if (
            current.content_sha256.as_str(),
            current.validation_sha256.as_str(),
        ) != admitted
        {
            bail!("private H3 owner-opened {role:?} authority differs from admission")
        }
    }
    Ok(())
}

#[cfg(feature = "mp4")]
fn private_h3_component_role(artifact: &H3QualifiedArtifact) -> H3FactoryComponentRole {
    match artifact.component {
        "conditioner" | "processor" => H3FactoryComponentRole::Conditioner,
        "visual-vae" => H3FactoryComponentRole::VisualVae,
        "audio-vae" => H3FactoryComponentRole::AudioVae,
        "component-config" if artifact.relative_path.starts_with("text_encoder/") => {
            H3FactoryComponentRole::Conditioner
        }
        "component-config" if artifact.relative_path.starts_with("vae/") => {
            H3FactoryComponentRole::VisualVae
        }
        "component-config" if artifact.relative_path.starts_with("audio_vae/") => {
            H3FactoryComponentRole::AudioVae
        }
        _ => H3FactoryComponentRole::Transformer,
    }
}

#[cfg(feature = "mp4")]
const fn private_h3_component_role_id(role: H3FactoryComponentRole) -> &'static str {
    match role {
        H3FactoryComponentRole::Conditioner => "conditioner",
        H3FactoryComponentRole::Transformer => "transformer",
        H3FactoryComponentRole::VisualVae => "visual-vae",
        H3FactoryComponentRole::AudioVae => "audio-vae",
    }
}

#[cfg(feature = "mp4")]
fn update_private_h3_qualified_artifact(digest: &mut Sha256, artifact: &H3QualifiedArtifact) {
    for value in [
        artifact.relative_path.as_str(),
        artifact.component,
        artifact.source_revision,
        artifact.sha256.as_str(),
        artifact.structural_contract,
        artifact.header_identity_sha256.as_deref().unwrap_or(""),
        artifact.policy_identity_sha256.as_deref().unwrap_or(""),
    ] {
        update_string(digest, value);
    }
    digest.update(artifact.size_bytes.to_le_bytes());
    digest.update(
        artifact
            .tensor_count
            .and_then(|count| u64::try_from(count).ok())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
}

#[cfg(feature = "mp4")]
#[allow(clippy::too_many_arguments)]
fn private_h3_admission_execution_fingerprint(
    resolved_request_identity_sha256: &str,
    artifact_report: &H3PrivateArtifactQualificationReport,
    runtime_qualification_identity_sha256: &str,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: (u16, u16),
    attention_runtime_identity_sha256: &str,
    components: &[H3PrivateComponentDigest],
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-admission-execution.v1\0");
    for value in [
        contract::FL2VA_COMFY,
        "fl2va",
        resolved_request_identity_sha256,
        artifact_report.qualification_identity_sha256.as_str(),
        runtime_qualification_identity_sha256,
        device_id,
        attention_runtime_identity_sha256,
        "host-cpu-then-drop",
    ] {
        update_string(&mut digest, value);
    }
    digest.update((device_ordinal as u64).to_le_bytes());
    digest.update(compute_capability.0.to_le_bytes());
    digest.update(compute_capability.1.to_le_bytes());
    for component in components {
        update_string(&mut digest, private_h3_component_role_id(component.role));
        update_string(&mut digest, &component.content_sha256);
        update_string(&mut digest, &component.validation_sha256);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(feature = "mp4")]
fn private_h3_pruned_adaln_identity(transformer: &H3ComfyOpenedInt8Checkpoint) -> Result<String> {
    let mold_candle::minimax_h3::H3AdaLnMode::Curve { grid, basis_dim } =
        transformer.candidate().strategy.adaln_mode
    else {
        bail!("private H3 FL2VA checkpoint did not retain the pruned AdaLN curve")
    };
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-pruned-adaln-table.v1\0");
    for value in [
        transformer.content_sha256(),
        transformer.candidate().header_identity_sha256.as_str(),
        transformer.checkpoint_identity_sha256(),
        transformer.candidate().strategy.stable_id.as_str(),
        "adaln_t_table",
        "clamp-unit-then-adjacent-linear",
    ] {
        update_string(&mut digest, value);
    }
    digest.update((grid as u64).to_le_bytes());
    digest.update((basis_dim as u64).to_le_bytes());
    Ok(format!("{:x}", digest.finalize()))
}

fn private_h3_admission_evidence_identity(evidence: &H3PrivateFl2VaAdmissionEvidence) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-admission-evidence.v2\0");
    for value in [
        evidence.submitted_request_identity_sha256.as_str(),
        evidence.resolved_request_identity_sha256.as_str(),
        evidence.canonical_model.as_str(),
        private_h3_mode_id(evidence.mode),
        evidence.device_id.as_str(),
        evidence.base_factory_authority.identity_sha256(),
        evidence.execution_fingerprint.as_str(),
        evidence.component_set_identity_sha256.as_str(),
        evidence.prepared_request_identity_sha256.as_str(),
        evidence.prepared_attempt_identity_sha256.as_str(),
        evidence.target_budget_identity_sha256.as_str(),
        evidence.artifact_qualification_identity_sha256.as_str(),
        evidence.runtime_qualification_identity_sha256.as_str(),
        evidence.attention.runtime_identity_sha256.as_str(),
        evidence.attention.qualification_kernel_identity.as_str(),
        evidence.attention.qualification_sha256.as_str(),
    ] {
        update_string(&mut digest, value);
    }
    digest.update(match evidence.task {
        Task::Fl2va => b"fl2va".as_slice(),
        Task::Ref2va => b"ref2va".as_slice(),
    });
    for value in [
        evidence.device_ordinal as u64,
        u64::from(evidence.compute_capability.0),
        u64::from(evidence.compute_capability.1),
        evidence.admitted_available_device_bytes,
        evidence.admitted_host_headroom_bytes,
        evidence.predicted_device_peak_bytes,
        evidence.predicted_host_increment_bytes,
        evidence.seed,
        evidence.runtime_bounds.fixed_runtime_host_bytes,
        evidence.runtime_bounds.fixed_runtime_device_bytes,
        evidence.runtime_bounds.qwen_activation_workspace_bytes,
        evidence
            .runtime_bounds
            .vae_construction_device_workspace_bytes,
        evidence.runtime_bounds.condition_vae_workspace_device_bytes,
        evidence.runtime_bounds.attention_workspace_device_bytes,
        evidence.runtime_bounds.ffn_workspace_device_bytes,
        evidence.runtime_bounds.decoder_tile_workspace_device_bytes,
        evidence.runtime_bounds.audio_decode_workspace_device_bytes,
        evidence.runtime_bounds.encoded_video_host_bytes_bound,
        evidence.runtime_bounds.thumbnail_host_bytes_bound,
        evidence.runtime_bounds.mux_output_host_bytes_bound,
        evidence.runtime_bounds.aac_mux_staging_host_bytes,
    ] {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

const fn private_h3_mode_id(mode: Mode) -> &'static str {
    match mode {
        Mode::TextToAudioVideo => "t2va",
        Mode::FirstFrameToAudioVideo => "first-frame-fl2va",
        Mode::LastFrameToAudioVideo => "last-frame-fl2va",
        Mode::FirstAndLastFrameToAudioVideo => "first-last-frame-fl2va",
        Mode::ReferenceToAudioVideo => "ref2va",
    }
}

fn require_private_sha256(value: &str, label: &str) -> Result<()> {
    if !valid_sha256(value) {
        bail!("{label} is not an exact SHA-256 identity")
    }
    Ok(())
}

/// Scheduler/owner facts available before endpoint preprocessing creates the
/// prepared-attempt and target-budget identities. The atomic preparation seam
/// derives those identities, enriches the private factory authority, and
/// returns them only through [`H3PrivateFl2VaAttemptFacts`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivateFl2VaOwnerFenceFacts {
    pub work_identity_sha256: String,
    pub cancellation_scope_identity_sha256: String,
    pub device_id: String,
    pub device_ordinal: usize,
    pub compute_capability: (u16, u16),
    pub memory_ledger_sequence: u64,
    pub admission_evidence_identity_sha256: String,
    pub artifact_qualification_identity_sha256: String,
    pub runtime_qualification_identity_sha256: String,
    pub prepared_attempt_identity_sha256: String,
    pub target_budget_identity_sha256: String,
    pub predicted_device_peak_bytes: u64,
    pub predicted_host_increment_bytes: u64,
}

impl H3PrivateFl2VaOwnerFenceFacts {
    pub fn validate(&self) -> Result<()> {
        if self.device_id.trim().is_empty()
            || self.compute_capability.0 == 0
            || self.memory_ledger_sequence == 0
            || self.predicted_device_peak_bytes == 0
            || self.predicted_host_increment_bytes == 0
            || !valid_sha256(&self.work_identity_sha256)
            || !valid_sha256(&self.cancellation_scope_identity_sha256)
            || !valid_sha256(&self.admission_evidence_identity_sha256)
            || !valid_sha256(&self.artifact_qualification_identity_sha256)
            || !valid_sha256(&self.runtime_qualification_identity_sha256)
            || !valid_sha256(&self.prepared_attempt_identity_sha256)
            || !valid_sha256(&self.target_budget_identity_sha256)
        {
            bail!("private H3 pre-prepare owner fence is incomplete")
        }
        Ok(())
    }
}

/// Non-Clone execution context for consuming exactly one prepared private H3
/// attempt. The cancellation token in this value is the sole runtime token:
/// `run_once` installs it into the progress reporter before any checkpoint.
pub struct H3PrivateFl2VaRunContext {
    work_identity_sha256: String,
    cancellation_scope_identity_sha256: String,
    memory_ledger_sequence: u64,
    cancellation: InferenceCancellationToken,
    identity_sha256: String,
}

impl std::fmt::Debug for H3PrivateFl2VaRunContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("H3PrivateFl2VaRunContext")
            .field("work_identity_sha256", &self.work_identity_sha256)
            .field(
                "cancellation_scope_identity_sha256",
                &self.cancellation_scope_identity_sha256,
            )
            .field("memory_ledger_sequence", &self.memory_ledger_sequence)
            .field("identity_sha256", &self.identity_sha256)
            .finish_non_exhaustive()
    }
}

impl H3PrivateFl2VaRunContext {
    pub fn new(
        work_identity_sha256: impl Into<String>,
        cancellation_scope_identity_sha256: impl Into<String>,
        memory_ledger_sequence: u64,
        cancellation: InferenceCancellationToken,
    ) -> Result<Self> {
        let mut context = Self {
            work_identity_sha256: work_identity_sha256.into(),
            cancellation_scope_identity_sha256: cancellation_scope_identity_sha256.into(),
            memory_ledger_sequence,
            cancellation,
            identity_sha256: String::new(),
        };
        context.identity_sha256 = private_h3_run_context_identity(&context);
        context.validate()?;
        Ok(context)
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    fn validate(&self) -> Result<()> {
        if self.memory_ledger_sequence == 0
            || !valid_sha256(&self.work_identity_sha256)
            || !valid_sha256(&self.cancellation_scope_identity_sha256)
            || self.identity_sha256 != private_h3_run_context_identity(self)
        {
            bail!("private H3 run context is incomplete or changed")
        }
        Ok(())
    }
}

fn private_h3_run_context_identity(context: &H3PrivateFl2VaRunContext) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-run-context.v1\0");
    update_string(&mut digest, &context.work_identity_sha256);
    update_string(&mut digest, &context.cancellation_scope_identity_sha256);
    digest.update(context.memory_ledger_sequence.to_le_bytes());
    format!("{:x}", digest.finalize())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3PrivateAttemptConsumptionBinding {
    work_identity_sha256: String,
    cancellation_scope_identity_sha256: String,
    memory_ledger_sequence: u64,
    admission_evidence_identity_sha256: String,
    artifact_qualification_identity_sha256: String,
    runtime_qualification_identity_sha256: String,
    execution_fingerprint: String,
    prepared_attempt_identity_sha256: String,
    target_budget_identity_sha256: String,
    component_set_identity_sha256: String,
    identity_sha256: String,
}

impl H3PrivateAttemptConsumptionBinding {
    fn new(
        owner: &H3PrivateFl2VaOwnerFacts,
        ledger: &H3PrivateSchedulerLedgerIdentity,
    ) -> Result<Self> {
        owner.validate()?;
        ledger.revalidate()?;
        let mut binding = Self {
            work_identity_sha256: owner.work_identity_sha256.clone(),
            cancellation_scope_identity_sha256: owner.cancellation_scope_identity_sha256.clone(),
            memory_ledger_sequence: ledger.memory_ledger_sequence(),
            admission_evidence_identity_sha256: owner.admission_evidence_identity_sha256.clone(),
            artifact_qualification_identity_sha256: owner
                .artifact_qualification_identity_sha256
                .clone(),
            runtime_qualification_identity_sha256: owner
                .runtime_qualification_identity_sha256
                .clone(),
            execution_fingerprint: owner.execution_fingerprint.clone(),
            prepared_attempt_identity_sha256: owner.prepared_attempt_identity_sha256.clone(),
            target_budget_identity_sha256: owner.target_budget_identity_sha256.clone(),
            component_set_identity_sha256: owner.component_set_identity_sha256.clone(),
            identity_sha256: String::new(),
        };
        binding.identity_sha256 = private_h3_consumption_binding_identity(&binding);
        binding.revalidate(owner, ledger)?;
        Ok(binding)
    }

    fn revalidate(
        &self,
        owner: &H3PrivateFl2VaOwnerFacts,
        ledger: &H3PrivateSchedulerLedgerIdentity,
    ) -> Result<()> {
        owner.validate()?;
        ledger.revalidate()?;
        if self.work_identity_sha256 != owner.work_identity_sha256
            || self.cancellation_scope_identity_sha256 != owner.cancellation_scope_identity_sha256
            || self.memory_ledger_sequence != ledger.memory_ledger_sequence()
            || self.admission_evidence_identity_sha256 != owner.admission_evidence_identity_sha256
            || self.artifact_qualification_identity_sha256
                != owner.artifact_qualification_identity_sha256
            || self.runtime_qualification_identity_sha256
                != owner.runtime_qualification_identity_sha256
            || self.execution_fingerprint != owner.execution_fingerprint
            || self.prepared_attempt_identity_sha256 != owner.prepared_attempt_identity_sha256
            || self.target_budget_identity_sha256 != owner.target_budget_identity_sha256
            || self.component_set_identity_sha256 != owner.component_set_identity_sha256
            || ledger.work_identity_sha256() != self.work_identity_sha256
            || ledger.cancellation_scope_identity_sha256()
                != self.cancellation_scope_identity_sha256
            || ledger.admission_evidence_identity_sha256()
                != self.admission_evidence_identity_sha256
            || ledger.artifact_qualification_identity_sha256()
                != self.artifact_qualification_identity_sha256
            || ledger.runtime_qualification_identity_sha256()
                != self.runtime_qualification_identity_sha256
            || self.identity_sha256 != private_h3_consumption_binding_identity(self)
        {
            bail!("private H3 prepared-attempt consumption binding changed")
        }
        Ok(())
    }

    fn validate_context(&self, context: &H3PrivateFl2VaRunContext) -> Result<()> {
        context.validate()?;
        if self.work_identity_sha256 != context.work_identity_sha256
            || self.cancellation_scope_identity_sha256 != context.cancellation_scope_identity_sha256
            || self.memory_ledger_sequence != context.memory_ledger_sequence
        {
            bail!("private H3 run context differs from the prepared owner binding")
        }
        Ok(())
    }
}

fn private_h3_consumption_binding_identity(binding: &H3PrivateAttemptConsumptionBinding) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-attempt-consumption.v1\0");
    for value in [
        binding.work_identity_sha256.as_str(),
        binding.cancellation_scope_identity_sha256.as_str(),
        binding.admission_evidence_identity_sha256.as_str(),
        binding.artifact_qualification_identity_sha256.as_str(),
        binding.runtime_qualification_identity_sha256.as_str(),
        binding.execution_fingerprint.as_str(),
        binding.prepared_attempt_identity_sha256.as_str(),
        binding.target_budget_identity_sha256.as_str(),
        binding.component_set_identity_sha256.as_str(),
    ] {
        update_string(&mut digest, value);
    }
    digest.update(binding.memory_ledger_sequence.to_le_bytes());
    format!("{:x}", digest.finalize())
}

trait H3PrivateFl2VaPreparedRunner: Send {
    fn consumption_binding(&self) -> &H3PrivateAttemptConsumptionBinding;

    fn run(
        self: Box<Self>,
        progress: &ProgressReporter,
        cancellation: InferenceCancellationToken,
        allocation_commit: H3PrivateAllocationCommit,
    ) -> Result<H3PrivateFl2VaRunOutput>;
}

/// Opaque, non-Clone, one-shot prepared attempt. The runner is removed before
/// invocation, so a second call is rejected even if the first call failed.
pub struct H3PrivateFl2VaPreparedAttempt {
    facts: H3PrivateFl2VaAttemptFacts,
    runner: Option<Box<dyn H3PrivateFl2VaPreparedRunner>>,
}

impl std::fmt::Debug for H3PrivateFl2VaPreparedAttempt {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("H3PrivateFl2VaPreparedAttempt")
            .field("facts", &self.facts)
            .field("consumed", &self.runner.is_none())
            .finish_non_exhaustive()
    }
}

impl H3PrivateFl2VaPreparedAttempt {
    pub fn facts(&self) -> &H3PrivateFl2VaAttemptFacts {
        &self.facts
    }

    pub fn run_once(
        &mut self,
        context: H3PrivateFl2VaRunContext,
        progress: &mut ProgressReporter,
        allocation_commit: H3PrivateAllocationCommit,
    ) -> Result<H3PrivateFl2VaRunOutput> {
        let runner = self
            .runner
            .take()
            .ok_or_else(|| anyhow!("private H3 prepared attempt was already consumed"))?;
        self.facts
            .validate_against_binding(runner.consumption_binding())?;
        runner.consumption_binding().validate_context(&context)?;
        let runner_cancellation = context.cancellation.clone();
        progress.with_attempt_cancellation_token(context.cancellation, |progress| {
            runner.run(progress, runner_cancellation, allocation_commit)
        })
    }

    #[allow(dead_code)]
    fn from_runner(
        facts: H3PrivateFl2VaAttemptFacts,
        runner: impl H3PrivateFl2VaPreparedRunner + 'static,
    ) -> Result<Self> {
        facts.validate_against_binding(runner.consumption_binding())?;
        Ok(Self {
            facts,
            runner: Some(Box::new(runner)),
        })
    }
}

/// Production preparation entrypoint. Exact endpoint preprocessing remains
/// CPU-only and intentionally precedes allocation commitment; CUDA `Device`
/// construction remains exclusively inside the one-shot runner after the
/// scheduler callback succeeds. An empty reviewed authority set
/// deterministically rejects before path access, hashing, preprocessing, or
/// device construction. Its full input and opaque success types keep that
/// boundary independent of the number of reviewed records.
pub fn prepare_h3_private_fl2va_attempt(
    input: H3PrivateFl2VaPrepareInput<'_>,
    progress: &ProgressReporter,
) -> std::result::Result<H3PrivateFl2VaPreparedAttempt, H3PrivateFl2VaPrepareError> {
    if !reviewed_h3_private_runtime_available() {
        return Err(H3PrivateFl2VaPrepareError::MissingReviewedRuntimeQualification);
    }
    #[cfg(not(feature = "mp4"))]
    {
        let _ = (input, progress);
        Err(H3PrivateFl2VaPrepareError::InvalidEvidence(
            "private H3 synchronized audio-video requires Mold's mp4 feature".into(),
        ))
    }
    #[cfg(feature = "mp4")]
    {
        prepare_reviewed_h3_private_fl2va_attempt(input, progress)
            .map_err(|error| H3PrivateFl2VaPrepareError::InvalidEvidence(error.to_string()))
    }
}

#[cfg(feature = "mp4")]
fn prepare_reviewed_h3_private_fl2va_attempt(
    input: H3PrivateFl2VaPrepareInput<'_>,
    progress: &ProgressReporter,
) -> Result<H3PrivateFl2VaPreparedAttempt> {
    progress.checkpoint()?;
    let H3PrivateFl2VaPrepareInput {
        request,
        frozen_factory,
        admission_evidence,
        paths,
        owner_fence,
    } = input;
    owner_fence.validate()?;
    admission_evidence.validate_for(
        request,
        &owner_fence.device_id,
        owner_fence.device_ordinal,
        owner_fence.compute_capability,
        admission_evidence.admitted_available_device_bytes(),
        admission_evidence.admitted_host_headroom_bytes(),
    )?;
    let resolved_request = admission_evidence.resolve_request(request)?;
    let resolved_request_identity_sha256 = private_h3_request_identity(&resolved_request)?;
    if frozen_factory != admission_evidence.base_factory_authority()
        || owner_fence.admission_evidence_identity_sha256 != admission_evidence.identity_sha256()
        || owner_fence.artifact_qualification_identity_sha256
            != admission_evidence.artifact_qualification_identity_sha256()
        || owner_fence.runtime_qualification_identity_sha256
            != admission_evidence.runtime_qualification_identity_sha256()
        || owner_fence.prepared_attempt_identity_sha256
            != admission_evidence.prepared_attempt_identity_sha256()
        || owner_fence.target_budget_identity_sha256
            != admission_evidence.target_budget_identity_sha256()
        || owner_fence.predicted_device_peak_bytes
            != admission_evidence.predicted_device_peak_bytes()
        || owner_fence.predicted_host_increment_bytes
            != admission_evidence.predicted_host_increment_bytes()
    {
        bail!("private H3 owner fence differs from immutable admission evidence")
    }
    validate_base_factory(frozen_factory, &owner_fence)?;
    let mode = contract::validate_request_contract(request, Task::Fl2va)
        .map_err(|error| anyhow!("{}: {}", error.code, error.message))?;
    #[cfg(not(feature = "h3"))]
    let reviewed_runtime_qualification = open_reviewed_h3_private_runtime_qualification(
        paths.runtime_qualification_record,
        REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256,
    )?;
    #[cfg(not(feature = "h3"))]
    reviewed_runtime_qualification.validate_route(
        &owner_fence.device_id,
        owner_fence.device_ordinal,
        owner_fence.compute_capability,
    )?;

    let artifact_report = qualify_private_artifacts_with_control(
        paths.models_root,
        contract::FL2VA_COMFY,
        paths.authorization_record,
        |hash| {
            progress.checkpoint()?;
            progress.weight_load(
                H3_ARTIFACT_VERIFICATION_PROGRESS,
                hash.total_bytes_verified,
                hash.total_bytes,
            );
            progress.checkpoint()?;
            Ok(())
        },
    )?;
    if artifact_report.qualification_identity_sha256
        != owner_fence.artifact_qualification_identity_sha256
    {
        bail!("private H3 reopened artifacts differ from admission evidence")
    }
    progress.checkpoint()?;

    let attention_device = H3AttentionDevice::Cuda {
        compute_capability: Some(owner_fence.compute_capability),
    };
    let attention_model = H3AttentionModelContract::released_bf16();
    let attention =
        H3AttentionRuntimeAuthority::qualify_flash_attention_v2(attention_device, attention_model)
            .map_err(|error| anyhow!(error.to_string()))?;
    #[cfg(feature = "h3")]
    let attention_qualification_sha256 = attention.identity_sha256().to_string();
    #[cfg(not(feature = "h3"))]
    let attention_qualification_sha256 =
        frozen_factory.attention_qualification_sha256().to_string();
    let attention_input = H3FactoryAttentionInput {
        generic_backend: frozen_factory.attention_backend(),
        generic_chunk: frozen_factory.attention_chunk(),
        runtime_backend: attention.backend(),
        kernel: attention.kernel(),
        activation: attention.activation(),
        device: attention.device(),
        model_contract: attention.contract(),
        runtime_identity_sha256: attention.identity_sha256().into(),
        qualification_kernel_identity: attention.kernel().identity().into(),
        qualification_sha256: attention_qualification_sha256.clone(),
        full_noncausal: true,
        lossless: true,
    };
    #[cfg(not(feature = "h3"))]
    let runtime_qualification = reviewed_runtime_qualification.authenticate(
        &artifact_report,
        &owner_fence.device_id,
        owner_fence.device_ordinal,
        owner_fence.compute_capability,
        attention.identity_sha256(),
        attention.kernel().identity(),
        &attention_qualification_sha256,
    )?;
    #[cfg(feature = "h3")]
    let runtime_qualification = public_runtime_qualification(
        &artifact_report,
        &owner_fence.device_id,
        owner_fence.device_ordinal,
        owner_fence.compute_capability,
        attention.identity_sha256(),
        attention.kernel().identity(),
        &attention_qualification_sha256,
        // Whatever admission froze, not a fresh environment read.
        frozen_factory
            .quantization()
            .turbo_adapter()
            .map(|turbo| turbo.grid_points),
    )?;
    if runtime_qualification.identity_sha256() != owner_fence.runtime_qualification_identity_sha256
        || runtime_qualification.artifact_qualification_identity_sha256()
            != owner_fence.artifact_qualification_identity_sha256
    {
        bail!("private H3 reopened runtime qualification differs from admission evidence")
    }
    progress.checkpoint()?;

    let storage = H3PrivateComfyStorageAuthority::resolve(paths.models_root)?;
    let qwen_support =
        load_qualified_private_qwen_support(paths.models_root, contract::FL2VA_COMFY)?;
    let transformer_cancellation = H3PrivatePreparationCancellation { progress };
    let opened_transformer = open_h3_comfy_published_int8_checkpoint(
        storage.transformer_path(),
        H3TransformerTask::T2VaFl2Va,
        H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
        &transformer_cancellation,
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    let vae_plan = storage.vae_plan(paths.staging_root)?;
    let mut vae_observer = H3PrivatePreparationVaeObserver::new(progress);
    let opened_vae = open_h3_comfy_vae_authority(&vae_plan, &mut vae_observer);
    let opened_vae = vae_observer.finish(opened_vae)?;
    // The runtime parks both VAEs for the whole denoise, so a second authority
    // is authenticated here — while the sources are still being inspected —
    // and retained with its own descriptors and staged copies. Reconstructing
    // from a pathname at reload time would reintroduce exactly the replacement
    // window the first open exists to close. The staged pair is process-cached,
    // so this second open re-reads no weights from the model root.
    let mut reload_vae_observer = H3PrivatePreparationVaeObserver::new(progress);
    let reload_vae = open_h3_comfy_vae_authority(&vae_plan, &mut reload_vae_observer);
    let reload_vae = reload_vae_observer.finish(reload_vae)?;
    let reload_vae = H3PrivateRetainedVaeReload::bind(reload_vae, &opened_vae)?;

    let qwen_artifact = exact_qualified_qwen_artifact(&artifact_report)?;
    let qwen_header_identity = qwen_artifact
        .header_identity_sha256
        .as_deref()
        .ok_or_else(|| anyhow!("qualified private H3 Qwen artifact has no header identity"))?;
    let qwen_policy_identity = qwen_artifact
        .policy_identity_sha256
        .as_deref()
        .ok_or_else(|| anyhow!("qualified private H3 Qwen artifact has no policy identity"))?;
    let owner_component_digests = private_h3_component_digests(
        &artifact_report,
        runtime_qualification.identity_sha256(),
        &qwen_support,
        &opened_transformer,
        &opened_vae,
        qwen_artifact,
    )?;
    validate_owner_component_digests(frozen_factory, &owner_component_digests)?;
    let owner_execution_fingerprint = private_h3_admission_execution_fingerprint(
        &resolved_request_identity_sha256,
        &artifact_report,
        runtime_qualification.identity_sha256(),
        &owner_fence.device_id,
        owner_fence.device_ordinal,
        owner_fence.compute_capability,
        attention.identity_sha256(),
        &owner_component_digests,
    );
    if owner_execution_fingerprint != frozen_factory.execution_fingerprint()
        || owner_execution_fingerprint != admission_evidence.execution_fingerprint()
    {
        bail!("private H3 owner-opened execution authority differs from admission")
    }
    let qwen_open_route = H3PrivateQwenOpenRouteAuthority::capture(
        frozen_factory,
        &qwen_support,
        &owner_fence.device_id,
        owner_fence.device_ordinal,
        owner_fence.compute_capability,
        &qwen_artifact.sha256,
        qwen_header_identity,
        qwen_policy_identity,
    )?;
    let mut checkpoint = H3PrivatePreparationCheckpoint { progress };
    let opened_qwen = open_authorized_private_qwen_authority(
        storage.qwen_weights_path(),
        &qwen_support,
        &qwen_open_route,
        frozen_factory,
        &mut checkpoint,
    )?;
    progress.checkpoint()?;

    let mut prepare_observer = H3EngineProgressObserver::new(progress);
    // Admission froze which adapter (if any) this attempt runs; re-reading the
    // environment here would let it change under a frozen plan.
    let frozen_turbo = frozen_factory.quantization().turbo_adapter();
    let prepared_attempt = H3PrivatePreparedFl2VaAttempt::prepare(
        request,
        frozen_factory.execution_fingerprint(),
        &storage,
        &qwen_support,
        &opened_transformer,
        &opened_qwen,
        &opened_vae,
        runtime_qualification.bounds(),
        frozen_turbo,
        progress,
        &mut prepare_observer,
    )?;
    let seed = prepared_attempt.seed();
    let prepared = prepared_attempt.into_factory_inputs();
    prepared.revalidate()?;
    runtime_qualification
        .validate_prepared_envelope_with_turbo(prepared.prepared_request_input(), frozen_turbo)?;
    let enriched_factory = frozen_factory.with_private_prepared_attempt(
        prepared.factory_attempt_input().clone(),
        prepared.budget_echo_input().clone(),
        attention_input,
    )?;
    if prepared.prepared_attempt_identity_sha256() != owner_fence.prepared_attempt_identity_sha256
        || prepared.target_budget_identity_sha256() != owner_fence.target_budget_identity_sha256
        || prepared.predicted_device_peak_bytes() != owner_fence.predicted_device_peak_bytes
        || prepared.predicted_host_increment_bytes() != owner_fence.predicted_host_increment_bytes
    {
        bail!("private H3 prepared budget differs from the scheduler owner fence")
    }

    let media = H3PrivateFl2VaMediaContract {
        canonical_model: contract::FL2VA_COMFY.into(),
        task: Task::Fl2va,
        mode,
        seed,
        width: request.width,
        height: request.height,
        frames: request.frames.unwrap_or(contract::MIN_FRAMES),
        fps: request.fps.unwrap_or(contract::FIXED_FPS),
    };
    media.validate()?;
    let memory_ledger_sequence = owner_fence.memory_ledger_sequence;
    let owner = H3PrivateFl2VaOwnerFacts {
        work_identity_sha256: owner_fence.work_identity_sha256,
        cancellation_scope_identity_sha256: owner_fence.cancellation_scope_identity_sha256,
        admission_evidence_identity_sha256: owner_fence.admission_evidence_identity_sha256,
        artifact_qualification_identity_sha256: owner_fence.artifact_qualification_identity_sha256,
        runtime_qualification_identity_sha256: owner_fence.runtime_qualification_identity_sha256,
        device_id: owner_fence.device_id,
        device_ordinal: owner_fence.device_ordinal,
        execution_fingerprint: enriched_factory.execution_fingerprint().into(),
        prepared_attempt_identity_sha256: prepared.prepared_attempt_identity_sha256().into(),
        target_budget_identity_sha256: prepared.target_budget_identity_sha256().into(),
        component_set_identity_sha256: enriched_factory.component_set_identity_sha256().into(),
        requested_grid_points: prepared.factory_attempt_input().request.grid_points,
        transformer_evaluations: prepared
            .factory_attempt_input()
            .request
            .denoise_forward_count,
        predicted_device_peak_bytes: prepared.predicted_device_peak_bytes(),
        predicted_host_increment_bytes: prepared.predicted_host_increment_bytes(),
        media,
    };
    owner.validate()?;
    let ledger = H3PrivateSchedulerLedgerIdentity::new(
        owner.work_identity_sha256.clone(),
        owner.cancellation_scope_identity_sha256.clone(),
        memory_ledger_sequence,
        owner.admission_evidence_identity_sha256.clone(),
        owner.artifact_qualification_identity_sha256.clone(),
        owner.runtime_qualification_identity_sha256.clone(),
        owner.execution_fingerprint.clone(),
        owner.prepared_attempt_identity_sha256.clone(),
        owner.target_budget_identity_sha256.clone(),
        owner.component_set_identity_sha256.clone(),
    )?;
    let consumption_binding = H3PrivateAttemptConsumptionBinding::new(&owner, &ledger)?;
    let memory_overlap = issue_private_fl2va_memory_overlap(&enriched_factory, &prepared, &ledger)?;
    let opened_facts = storage.opened_activation_facts(
        &qwen_support,
        &opened_transformer,
        &opened_qwen,
        &opened_vae,
    )?;
    let runtime_envelope = runtime_qualification.record.envelope.clone();
    let activation = H3PrivateFactoryActivationEvidence::derive(
        &enriched_factory,
        runtime_qualification,
        &opened_facts,
        &prepared,
        ledger,
        &owner,
    )?;
    let admitted = enriched_factory.private_fl2va_runtime_authority_with_activation(&activation)?;
    let artifact_lease = H3PrivateServerFl2VaArtifactLease::from_opened(
        &admitted,
        &owner_component_digests,
        &qwen_support,
        &opened_transformer,
        &opened_qwen,
        &opened_vae,
        &memory_overlap,
    )?;
    let facts = owner.attempt_facts(&consumption_binding);
    let runner = H3PrivateConcretePreparedRunner {
        authority: enriched_factory,
        activation,
        prepared,
        storage,
        qwen_support,
        opened_transformer,
        opened_qwen,
        opened_vae,
        reload_vae,
        attention,
        artifact_lease,
        memory_overlap,
        runtime_envelope,
        owner,
        consumption_binding,
    };
    H3PrivateFl2VaPreparedAttempt::from_runner(facts, runner)
}

#[cfg(feature = "mp4")]
fn validate_base_factory(
    factory: &FrozenH3FactoryAuthority,
    owner: &H3PrivateFl2VaOwnerFenceFacts,
) -> Result<()> {
    if factory.canonical_model() != contract::FL2VA_COMFY
        || factory.task() != Task::Fl2va
        || factory.device_id() != owner.device_id
        || factory.device_ordinal() != owner.device_ordinal
        || factory.compute_capability() != Some(owner.compute_capability)
        || factory.prepared_target_attempt_identities().is_some()
        || !factory.attention_runtime_identity_sha256().is_empty()
        || factory.attention_backend() != AttentionBackend::Flash
        || factory.attention_chunk() != AttentionChunkPolicy::Off
    {
        bail!("private H3 base factory differs from the exact FL2VA owner route")
    }
    Ok(())
}

#[cfg(feature = "mp4")]
fn exact_qualified_qwen_artifact(
    report: &H3PrivateArtifactQualificationReport,
) -> Result<&H3QualifiedArtifact> {
    let mut matches = report
        .artifacts
        .iter()
        .filter(|artifact| artifact.sha256 == H3_QWEN_NVFP4_AWQ_SHA256);
    let artifact = matches
        .next()
        .ok_or_else(|| anyhow!("private H3 qualification omits the released Qwen artifact"))?;
    if matches.next().is_some()
        || artifact.policy_identity_sha256.as_deref() != Some(H3_QWEN_NVFP4_AWQ_POLICY_SHA256)
        || artifact
            .header_identity_sha256
            .as_deref()
            .is_none_or(|identity| !valid_sha256(identity))
    {
        bail!("private H3 qualification has ambiguous or crossed Qwen authority")
    }
    Ok(artifact)
}

#[cfg(feature = "mp4")]
fn private_lease_id(
    label: &str,
    factory_identity_sha256: &str,
    work_identity_sha256: &str,
    cancellation_scope_identity_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-server-lease.v1\0");
    for value in [
        label,
        factory_identity_sha256,
        work_identity_sha256,
        cancellation_scope_identity_sha256,
    ] {
        update_string(&mut digest, value);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(feature = "mp4")]
struct H3PrivateServerConditionerLease {
    factory_identity_sha256: String,
    execution_fingerprint: String,
    lease_id: String,
    device_id: String,
    device: Device,
    active: bool,
}

#[cfg(feature = "mp4")]
impl H3PrivateServerConditionerLease {
    fn new(
        factory: &FrozenH3FactoryAuthority,
        device_id: String,
        device: Device,
        work_identity_sha256: &str,
        cancellation_scope_identity_sha256: &str,
        label: &str,
    ) -> Self {
        Self {
            factory_identity_sha256: factory.identity_sha256().into(),
            execution_fingerprint: factory.execution_fingerprint().into(),
            lease_id: private_lease_id(
                label,
                factory.identity_sha256(),
                work_identity_sha256,
                cancellation_scope_identity_sha256,
            ),
            device_id,
            device,
            active: true,
        }
    }
}

#[cfg(feature = "mp4")]
impl H3ConditionerLease for H3PrivateServerConditionerLease {
    fn lease_id(&self) -> &str {
        &self.lease_id
    }

    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn release(&mut self) {
        self.active = false;
    }
}

#[cfg(feature = "mp4")]
impl H3PrivateQwenConditionerLease for H3PrivateServerConditionerLease {
    fn factory_identity_sha256(&self) -> &str {
        &self.factory_identity_sha256
    }

    fn execution_fingerprint(&self) -> &str {
        &self.execution_fingerprint
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(feature = "mp4")]
struct H3PrivateServerExecutionLease {
    lease_id: String,
    device_id: String,
    compute_capability: (u16, u16),
    execution_fingerprint: String,
    device: Device,
    active: bool,
}

#[cfg(feature = "mp4")]
impl H3PrivateServerExecutionLease {
    fn new(
        factory: &FrozenH3FactoryAuthority,
        device: Device,
        work_identity_sha256: &str,
        cancellation_scope_identity_sha256: &str,
        label: &str,
    ) -> Self {
        Self {
            lease_id: private_lease_id(
                label,
                factory.identity_sha256(),
                work_identity_sha256,
                cancellation_scope_identity_sha256,
            ),
            device_id: factory.device_id().into(),
            compute_capability: factory
                .compute_capability()
                .ok_or_else(|| anyhow!("public MiniMax H3 runtime requires a CUDA route"))?,
            execution_fingerprint: factory.execution_fingerprint().into(),
            device,
            active: true,
        }
    }
}

#[cfg(feature = "mp4")]
impl H3BackendExecutionLease for H3PrivateServerExecutionLease {
    fn lease_id(&self) -> &str {
        &self.lease_id
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
        self.active
    }
}

#[cfg(feature = "mp4")]
struct H3PrivateServerQwenArtifactLease {
    factory_identity_sha256: String,
    component_set_identity_sha256: String,
    conditioner_component_content_sha256: String,
    conditioner_component_validation_sha256: String,
    support_identity_sha256: String,
    weight_identity_sha256: String,
    weight_header_identity_sha256: String,
    weight_policy_identity_sha256: String,
    active: bool,
}

#[cfg(feature = "mp4")]
unsafe impl H3BackendArtifactLease for H3PrivateServerQwenArtifactLease {
    fn component_set_identity(&self) -> &str {
        &self.component_set_identity_sha256
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(feature = "mp4")]
unsafe impl H3PrivateQwenArtifactLease for H3PrivateServerQwenArtifactLease {
    fn factory_identity_sha256(&self) -> &str {
        &self.factory_identity_sha256
    }

    fn conditioner_component_content_sha256(&self) -> &str {
        &self.conditioner_component_content_sha256
    }

    fn conditioner_component_validation_sha256(&self) -> &str {
        &self.conditioner_component_validation_sha256
    }

    fn support_identity_sha256(&self) -> &str {
        &self.support_identity_sha256
    }

    fn weight_identity_sha256(&self) -> &str {
        &self.weight_identity_sha256
    }

    fn weight_header_identity_sha256(&self) -> &str {
        &self.weight_header_identity_sha256
    }

    fn weight_policy_identity_sha256(&self) -> &str {
        &self.weight_policy_identity_sha256
    }
}

#[cfg(feature = "mp4")]
struct H3PrivateServerFl2VaArtifactLease {
    qwen: H3PrivateServerQwenArtifactLease,
    backend_plan_identity_sha256: String,
    transformer_component_content_sha256: String,
    transformer_component_validation_sha256: String,
    visual_vae_component_content_sha256: String,
    visual_vae_component_validation_sha256: String,
    audio_vae_component_content_sha256: String,
    audio_vae_component_validation_sha256: String,
    vae_artifact_plan_identity_sha256: String,
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

#[cfg(feature = "mp4")]
impl H3PrivateServerFl2VaArtifactLease {
    fn from_opened(
        admitted: &H3PrivateFl2VaFactoryAuthority,
        components: &[H3PrivateComponentDigest],
        support: &H3PrivateQwenSupport,
        transformer: &H3ComfyOpenedInt8Checkpoint,
        qwen: &H3AuthenticatedQwenNvfp4Authority,
        vae: &H3AuthenticatedComfyVaeAuthority,
        overlap: &H3PrivateFl2VaMemoryOverlapAuthority,
    ) -> Result<Self> {
        support.revalidate()?;
        transformer
            .revalidate()
            .map_err(|error| anyhow!(error.to_string()))?;
        qwen.revalidate()
            .map_err(|error| anyhow!(error.to_string()))?;
        vae.validate().map_err(|error| anyhow!(error.to_string()))?;
        let conditioner =
            private_h3_component_digest(components, H3FactoryComponentRole::Conditioner)?;
        let transformer_component =
            private_h3_component_digest(components, H3FactoryComponentRole::Transformer)?;
        let visual_vae =
            private_h3_component_digest(components, H3FactoryComponentRole::VisualVae)?;
        let audio_vae = private_h3_component_digest(components, H3FactoryComponentRole::AudioVae)?;
        if components.len() != 4
            || conditioner.content_sha256 != admitted.conditioner_component_content_sha256
            || conditioner.validation_sha256 != admitted.conditioner_component_validation_sha256
            || transformer_component.content_sha256 != admitted.transformer_component_content_sha256
            || transformer_component.validation_sha256
                != admitted.transformer_component_validation_sha256
            || visual_vae.content_sha256 != admitted.visual_vae_component_content_sha256
            || visual_vae.validation_sha256 != admitted.visual_vae_component_validation_sha256
            || audio_vae.content_sha256 != admitted.audio_vae_component_content_sha256
            || audio_vae.validation_sha256 != admitted.audio_vae_component_validation_sha256
        {
            bail!("private H3 owner-opened components differ from admitted runtime authority")
        }
        let pruned_adaln_table_identity_sha256 = match &admitted.quantization {
            H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                pruned_adaln_table_sha256,
                ..
            } => pruned_adaln_table_sha256.clone(),
            H3FactoryQuantizationAuthority::OfficialBf16 => {
                bail!("private H3 FL2VA runtime requires the reviewed Comfy quantization")
            }
        };
        let candidate = transformer.candidate();
        Ok(Self {
            qwen: H3PrivateServerQwenArtifactLease {
                factory_identity_sha256: admitted.factory_identity_sha256.clone(),
                component_set_identity_sha256: admitted.component_set_identity_sha256.clone(),
                conditioner_component_content_sha256: conditioner.content_sha256.clone(),
                conditioner_component_validation_sha256: conditioner.validation_sha256.clone(),
                support_identity_sha256: support.support_identity_sha256().into(),
                weight_identity_sha256: qwen.artifact_identity_sha256().into(),
                weight_header_identity_sha256: qwen.header_identity_sha256().into(),
                weight_policy_identity_sha256: qwen.policy_identity_sha256().into(),
                active: true,
            },
            backend_plan_identity_sha256: admitted.backend_plan_identity_sha256.clone(),
            transformer_component_content_sha256: transformer_component.content_sha256.clone(),
            transformer_component_validation_sha256: transformer_component
                .validation_sha256
                .clone(),
            visual_vae_component_content_sha256: visual_vae.content_sha256.clone(),
            visual_vae_component_validation_sha256: visual_vae.validation_sha256.clone(),
            audio_vae_component_content_sha256: audio_vae.content_sha256.clone(),
            audio_vae_component_validation_sha256: audio_vae.validation_sha256.clone(),
            vae_artifact_plan_identity_sha256: vae.artifact_plan_identity_sha256().into(),
            transformer_checkpoint_content_sha256: transformer.content_sha256().into(),
            transformer_checkpoint_layout_identity_sha256: candidate.header_identity_sha256.clone(),
            transformer_checkpoint_identity_sha256: transformer.checkpoint_identity_sha256().into(),
            transformer_policy_identity_sha256: candidate
                .strategy
                .quantization_policy
                .policy_sha256
                .clone(),
            pruned_adaln_table_identity_sha256,
            attention_runtime_identity_sha256: admitted.attention.runtime_identity_sha256.clone(),
            attention_kernel_identity: admitted.attention.qualification_kernel_identity.clone(),
            attention_qualification_sha256: admitted.attention.qualification_sha256.clone(),
            memory_overlap_identity_sha256: overlap.identity_sha256().into(),
        })
    }
}

#[cfg(feature = "mp4")]
unsafe impl H3BackendArtifactLease for H3PrivateServerFl2VaArtifactLease {
    fn component_set_identity(&self) -> &str {
        self.qwen.component_set_identity()
    }

    fn is_active(&self) -> bool {
        self.qwen.is_active()
    }
}

#[cfg(feature = "mp4")]
unsafe impl H3PrivateQwenArtifactLease for H3PrivateServerFl2VaArtifactLease {
    fn factory_identity_sha256(&self) -> &str {
        self.qwen.factory_identity_sha256()
    }

    fn conditioner_component_content_sha256(&self) -> &str {
        self.qwen.conditioner_component_content_sha256()
    }

    fn conditioner_component_validation_sha256(&self) -> &str {
        self.qwen.conditioner_component_validation_sha256()
    }

    fn support_identity_sha256(&self) -> &str {
        self.qwen.support_identity_sha256()
    }

    fn weight_identity_sha256(&self) -> &str {
        self.qwen.weight_identity_sha256()
    }

    fn weight_header_identity_sha256(&self) -> &str {
        self.qwen.weight_header_identity_sha256()
    }

    fn weight_policy_identity_sha256(&self) -> &str {
        self.qwen.weight_policy_identity_sha256()
    }
}

#[cfg(feature = "mp4")]
unsafe impl H3PrivateFl2VaArtifactLease for H3PrivateServerFl2VaArtifactLease {
    fn backend_plan_identity_sha256(&self) -> &str {
        &self.backend_plan_identity_sha256
    }

    fn transformer_component_content_sha256(&self) -> &str {
        &self.transformer_component_content_sha256
    }

    fn transformer_component_validation_sha256(&self) -> &str {
        &self.transformer_component_validation_sha256
    }

    fn visual_vae_component_content_sha256(&self) -> &str {
        &self.visual_vae_component_content_sha256
    }

    fn visual_vae_component_validation_sha256(&self) -> &str {
        &self.visual_vae_component_validation_sha256
    }

    fn audio_vae_component_content_sha256(&self) -> &str {
        &self.audio_vae_component_content_sha256
    }

    fn audio_vae_component_validation_sha256(&self) -> &str {
        &self.audio_vae_component_validation_sha256
    }

    fn vae_artifact_plan_identity_sha256(&self) -> &str {
        &self.vae_artifact_plan_identity_sha256
    }

    fn transformer_task(&self) -> H3TransformerTask {
        H3TransformerTask::T2VaFl2Va
    }

    fn transformer_checkpoint_content_sha256(&self) -> &str {
        &self.transformer_checkpoint_content_sha256
    }

    fn transformer_checkpoint_layout_identity_sha256(&self) -> &str {
        &self.transformer_checkpoint_layout_identity_sha256
    }

    fn transformer_checkpoint_identity_sha256(&self) -> &str {
        &self.transformer_checkpoint_identity_sha256
    }

    fn transformer_policy_identity_sha256(&self) -> &str {
        &self.transformer_policy_identity_sha256
    }

    fn pruned_adaln_table_identity_sha256(&self) -> &str {
        &self.pruned_adaln_table_identity_sha256
    }

    fn attention_runtime_identity_sha256(&self) -> &str {
        &self.attention_runtime_identity_sha256
    }

    fn attention_kernel_identity(&self) -> &str {
        &self.attention_kernel_identity
    }

    fn attention_qualification_sha256(&self) -> &str {
        &self.attention_qualification_sha256
    }

    fn memory_overlap_identity_sha256(&self) -> &str {
        &self.memory_overlap_identity_sha256
    }
}

#[cfg(feature = "mp4")]
struct H3PrivatePreparationCheckpoint<'a> {
    progress: &'a ProgressReporter,
}

#[cfg(feature = "mp4")]
struct H3PrivatePreparationCancellation<'a> {
    progress: &'a ProgressReporter,
}

#[cfg(feature = "mp4")]
impl H3ComfyInt8Cancellation for H3PrivatePreparationCancellation<'_> {
    fn is_cancelled(&self) -> bool {
        self.progress.checkpoint().is_err()
    }
}

#[cfg(feature = "mp4")]
impl H3PipelineCheckpoint for H3PrivatePreparationCheckpoint<'_> {
    fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()> {
        if event.total == 0 || event.completed > event.total {
            bail!("invalid private H3 preparation progress event {event:?}")
        }
        self.progress.checkpoint()?;
        Ok(())
    }
}

#[cfg(feature = "mp4")]
struct H3PrivatePreparationVaeObserver<'a> {
    progress: &'a ProgressReporter,
    first_error: Option<anyhow::Error>,
}

#[cfg(feature = "mp4")]
impl<'a> H3PrivatePreparationVaeObserver<'a> {
    fn new(progress: &'a ProgressReporter) -> Self {
        Self {
            progress,
            first_error: None,
        }
    }

    fn finish<T>(self, result: std::result::Result<T, H3ComfyVaeLoadError>) -> Result<T> {
        if let Some(error) = self.first_error {
            Err(error)
        } else {
            Ok(result?)
        }
    }
}

#[cfg(feature = "mp4")]
impl H3ComfyVaeLoadObserver for H3PrivatePreparationVaeObserver<'_> {
    fn checkpoint(&mut self, event: H3ComfyVaeLoadEvent) -> bool {
        if self.first_error.is_some() {
            return false;
        }
        if event.total == 0 || event.completed > event.total {
            self.first_error = Some(anyhow!(
                "invalid private H3 VAE preparation progress event {event:?}"
            ));
            return false;
        }
        if let Err(error) = self.progress.checkpoint() {
            self.first_error = Some(error.into());
            return false;
        }
        self.progress.weight_load(
            H3_VAE_ARTIFACT_VERIFICATION_PROGRESS,
            event.completed,
            event.total,
        );
        true
    }
}

#[cfg(feature = "mp4")]
fn commit_private_h3_allocation_then<T>(
    allocation_commit: &mut H3PrivateAllocationCommit,
    construct: impl FnOnce() -> Result<T>,
) -> Result<T> {
    allocation_commit.commit_once()?;
    construct()
}

#[cfg(all(feature = "mp4", feature = "cuda"))]
fn with_private_h3_cuda_execution_attempt<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    use cudarc::driver::CudaExecutionAttempt;

    let mut attempt = CudaExecutionAttempt::begin_unbound()
        .context("failed to install the private H3 CUDA execution boundary")?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
    match result {
        Err(payload) => {
            attempt.mark_panicked();
            let _status = attempt.finish();
            std::panic::resume_unwind(payload)
        }
        Ok(result) => {
            let status = attempt.finish();
            if status.resources_retained() {
                bail!(H3_CUDA_ATTEMPT_RETAINED_MARKER)
            }
            result
        }
    }
}

#[cfg(all(feature = "mp4", not(feature = "cuda")))]
fn with_private_h3_cuda_execution_attempt<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    operation()
}

#[cfg(feature = "mp4")]
struct H3PrivateConcretePreparedRunner {
    authority: FrozenH3FactoryAuthority,
    activation: H3PrivateFactoryActivationEvidence,
    prepared: H3PrivatePreparedFl2VaFactoryInputs,
    storage: H3PrivateComfyStorageAuthority,
    qwen_support: H3PrivateQwenSupport,
    opened_transformer: H3ComfyOpenedInt8Checkpoint,
    opened_qwen: H3AuthenticatedQwenNvfp4Authority,
    opened_vae: H3AuthenticatedComfyVaeAuthority,
    reload_vae: H3PrivateRetainedVaeReload,
    attention: H3AttentionRuntimeAuthority,
    artifact_lease: H3PrivateServerFl2VaArtifactLease,
    memory_overlap: H3PrivateFl2VaMemoryOverlapAuthority,
    runtime_envelope: H3PrivateRuntimeEnvelopeRecord,
    owner: H3PrivateFl2VaOwnerFacts,
    consumption_binding: H3PrivateAttemptConsumptionBinding,
}

#[cfg(feature = "mp4")]
fn runtime_envelope_observation(
    request: &H3FactoryPreparedRequestInput,
) -> Result<H3PrivateRuntimeEnvelopeObservation> {
    let endpoint = request
        .endpoints
        .first()
        .ok_or_else(|| anyhow!("private H3 runtime envelope has no endpoint"))?;
    if request.endpoints.len() != 1 || endpoint.anchor != H3FactoryEndpointAnchor::First {
        bail!("private H3 runtime envelope requires exactly one first-frame endpoint")
    }
    Ok(H3PrivateRuntimeEnvelopeObservation {
        width: request.width,
        height: request.height,
        frames: request.frames,
        fps: request.fps,
        batch_size: request.batch_size,
        steps: request.grid_points,
        endpoint_count: 1,
        endpoint_anchor: "first".into(),
        qwen_output_text_rows: request.rows.qwen_output_text_rows,
        qwen_vision_rows: request.rows.qwen_vision_rows,
        condition_visual_rows: request.rows.condition_visual_rows,
        target_video_rows: request.rows.target_video_rows,
        target_audio_rows: request.rows.target_audio_rows,
        total_packed_rows: request.rows.total_packed_rows,
    })
}

#[cfg(feature = "mp4")]
impl H3PrivateFl2VaPreparedRunner for H3PrivateConcretePreparedRunner {
    fn consumption_binding(&self) -> &H3PrivateAttemptConsumptionBinding {
        &self.consumption_binding
    }

    fn run(
        self: Box<Self>,
        progress: &ProgressReporter,
        cancellation: InferenceCancellationToken,
        mut allocation_commit: H3PrivateAllocationCommit,
    ) -> Result<H3PrivateFl2VaRunOutput> {
        cancellation.checkpoint()?;
        progress.checkpoint()?;
        let started = Instant::now();
        let Self {
            authority,
            activation,
            prepared,
            storage,
            qwen_support,
            opened_transformer,
            opened_qwen,
            opened_vae,
            reload_vae,
            attention,
            artifact_lease,
            memory_overlap,
            runtime_envelope,
            owner,
            consumption_binding,
        } = *self;
        runtime_envelope.validate_prepared(prepared.prepared_request_input())?;
        let observed_envelope = runtime_envelope_observation(prepared.prepared_request_input())?;
        let observed_compute_capability = match attention.device() {
            H3AttentionDevice::Cuda {
                compute_capability: Some((major, minor)),
            } => [major, minor],
            device => bail!(
                "private H3 terminal authority requires a concrete CUDA compute capability, got {device:?}"
            ),
        };
        let observed_authority = H3PrivateRuntimeAuthorityObservation {
            bootstrap_record_sha256: activation.runtime_qualification.record_file_sha256().into(),
            runtime_qualification_identity_sha256: activation
                .runtime_qualification
                .identity_sha256()
                .into(),
            device_id: owner.device_id.clone(),
            device_ordinal: owner.device_ordinal,
            compute_capability: observed_compute_capability,
            attention_runtime_identity_sha256: attention.identity_sha256().into(),
            attention_kernel_identity: attention.kernel().identity().into(),
            attention_qualification_sha256: artifact_lease.attention_qualification_sha256.clone(),
            process: capture_process_observation()?,
        };
        with_private_h3_cuda_execution_attempt(|| {
            consumption_binding.revalidate(&owner, &activation.scheduler_ledger)?;
            let cuda_device = commit_private_h3_allocation_then(&mut allocation_commit, || {
                Device::new_cuda(owner.device_ordinal)
                    .context("failed to construct the reviewed private H3 CUDA route")
            })?;
            let qwen_on_cpu = matches!(
                authority.conditioner_placement(),
                H3FactoryConditionerPlacement::HostCpuThenDrop
            );
            let runtime_bound_capture = H3PrivateRuntimeBoundCapture::begin(
                &cuda_device,
                qwen_on_cpu,
                observed_envelope,
                observed_authority,
            )?;
            // Keep one completion fence alive outside the concrete owner graph.
            // It runs only after a successful pipeline result and before any
            // CUDA-bearing local leaves the execution-attempt boundary.
            let completion_device = cuda_device.clone();
            let conditioner_device = match authority.conditioner_placement() {
                H3FactoryConditionerPlacement::AssignedCudaThenDrop => cuda_device.clone(),
                H3FactoryConditionerPlacement::HostCpuThenDrop => Device::Cpu,
            };
            let conditioner_device_id = match authority.conditioner_placement() {
                H3FactoryConditionerPlacement::AssignedCudaThenDrop => owner.device_id.clone(),
                H3FactoryConditionerPlacement::HostCpuThenDrop => "cpu".into(),
            };
            let conditioner_lease = H3PrivateServerConditionerLease::new(
                &authority,
                conditioner_device_id,
                conditioner_device,
                &owner.work_identity_sha256,
                &owner.cancellation_scope_identity_sha256,
                "runtime-conditioner",
            );
            let execution_lease = H3PrivateServerExecutionLease::new(
                &authority,
                cuda_device,
                &owner.work_identity_sha256,
                &owner.cancellation_scope_identity_sha256,
                "runtime-execution",
            );
            let phase_owner = bind_private_comfy_fl2va_phase_owner(
                authority,
                activation,
                prepared,
                storage,
                qwen_support,
                opened_transformer,
                opened_qwen,
                opened_vae,
                reload_vae,
                attention,
                conditioner_lease,
                execution_lease,
                artifact_lease,
                memory_overlap,
                allocation_commit,
            )?;
            let mut observer = H3EngineProgressObserver::new(progress);
            let output = run_private_comfy_fl2va_attempt(phase_owner, progress, &mut observer)?;
            completion_device
                .synchronize()
                .context("private H3 completion synchronization failed")?;
            cancellation.checkpoint()?;
            progress.checkpoint()?;
            let output = private_run_output(output, owner, &consumption_binding, started)?;
            let runtime_bound_observation = runtime_bound_capture.finish()?;
            tracing::info!(
                target: "mold::minimax_h3::private_runtime_bound",
                observation = %serde_json::to_string(&runtime_bound_observation)?,
                "captured private MiniMax H3 runtime bounds"
            );
            Ok(output)
        })
    }
}

#[cfg(feature = "mp4")]
fn private_run_output(
    runtime: H3PrivatePhaseRuntimeOutput,
    owner: H3PrivateFl2VaOwnerFacts,
    consumption: &H3PrivateAttemptConsumptionBinding,
    started: Instant,
) -> Result<H3PrivateFl2VaRunOutput> {
    owner.validate()?;
    if consumption.work_identity_sha256 != owner.work_identity_sha256
        || consumption.cancellation_scope_identity_sha256
            != owner.cancellation_scope_identity_sha256
        || consumption.admission_evidence_identity_sha256
            != owner.admission_evidence_identity_sha256
        || consumption.artifact_qualification_identity_sha256
            != owner.artifact_qualification_identity_sha256
        || consumption.runtime_qualification_identity_sha256
            != owner.runtime_qualification_identity_sha256
    {
        bail!("private H3 terminal owner changed from the consumption binding")
    }
    let output = runtime.output;
    let echo = runtime.identity_echo;
    let provenance = &output.provenance;
    let expected_mode = match owner.media.mode {
        Mode::TextToAudioVideo => "t2va",
        Mode::FirstFrameToAudioVideo => "first-frame-fl2va",
        Mode::LastFrameToAudioVideo => "last-frame-fl2va",
        Mode::FirstAndLastFrameToAudioVideo => "first-last-frame-fl2va",
        Mode::ReferenceToAudioVideo => "ref2va",
    };
    validate_private_sampler_provenance(
        provenance.sampler,
        provenance.requested_grid_points,
        provenance.transformer_evaluations,
        &owner,
    )?;
    if echo.device_id != owner.device_id
        || echo.execution_fingerprint != owner.execution_fingerprint
        || echo.prepared_attempt_identity_sha256 != owner.prepared_attempt_identity_sha256
        || echo.target_budget_identity_sha256 != owner.target_budget_identity_sha256
        || echo.component_set_identity_sha256 != owner.component_set_identity_sha256
        || provenance.device_id != owner.device_id
        || provenance.execution_fingerprint != owner.execution_fingerprint
        || provenance.mode != expected_mode
        || provenance.seed != owner.media.seed
        || provenance.width != usize::try_from(owner.media.width)?
        || provenance.height != usize::try_from(owner.media.height)?
        || provenance.frames != usize::try_from(owner.media.frames)?
        || provenance.fps != owner.media.fps
        || provenance.audio_sample_rate != contract::AUDIO_SAMPLE_RATE_HZ
        || provenance.audio_channels != contract::AUDIO_CHANNELS
        || output.mux_report.sample_rate != contract::AUDIO_SAMPLE_RATE_HZ
        || output.mux_report.channels
            != u16::try_from(contract::AUDIO_CHANNELS).expect("H3 channel count fits u16")
        || output.mp4.is_empty()
        || output.thumbnail_png.is_empty()
    {
        bail!("private H3 terminal output differs from the prepared attempt authority")
    }
    let duration_ms = crate::av_media::timeline_duration_ms(
        output.mux_report.video_duration_ticks,
        output.mux_report.video_timescale,
    )
    .context("private H3 mux returned an invalid duration")?;
    let pipeline_provenance_sha256 =
        format!("{:x}", Sha256::digest(serde_json::to_vec(provenance)?));
    let generation_time_ms = u64::try_from(started.elapsed().as_millis())
        .context("private H3 generation time exceeds u64")?;
    let identity_echo = H3PrivateFl2VaTerminalIdentityEcho {
        device_id: owner.device_id.clone(),
        device_ordinal: owner.device_ordinal,
        execution_fingerprint: owner.execution_fingerprint.clone(),
        prepared_attempt_identity_sha256: owner.prepared_attempt_identity_sha256.clone(),
        target_budget_identity_sha256: owner.target_budget_identity_sha256.clone(),
        component_set_identity_sha256: owner.component_set_identity_sha256.clone(),
        media: owner.media.clone(),
        duration_ms,
        audio_sample_rate: output.mux_report.sample_rate,
        audio_channels: output.mux_report.channels,
        synchronized_audio_video: true,
        pipeline_provenance_sha256: pipeline_provenance_sha256.clone(),
        admission_evidence_identity_sha256: owner.admission_evidence_identity_sha256.clone(),
        artifact_qualification_identity_sha256: owner
            .artifact_qualification_identity_sha256
            .clone(),
        runtime_qualification_identity_sha256: owner.runtime_qualification_identity_sha256.clone(),
        consumption_identity_sha256: consumption.identity_sha256.clone(),
    };
    let response = GenerateResponse {
        images: Vec::new(),
        video: Some(VideoData {
            data: output.mp4,
            format: OutputFormat::Mp4,
            width: owner.media.width,
            height: owner.media.height,
            frames: owner.media.frames,
            fps: owner.media.fps,
            pipeline: None,
            thumbnail: output.thumbnail_png,
            gif_preview: Vec::new(),
            has_audio: true,
            duration_ms: Some(duration_ms),
            audio_sample_rate: Some(output.mux_report.sample_rate),
            audio_channels: Some(u32::from(output.mux_report.channels)),
            pipeline_provenance_sha256: Some(pipeline_provenance_sha256),
            source_preprocessing: None,
        }),
        audio: None,
        generation_time_ms,
        model: owner.media.canonical_model,
        seed_used: owner.media.seed,
        gpu: Some(owner.device_ordinal),
    };
    Ok(H3PrivateFl2VaRunOutput {
        response,
        identity_echo,
    })
}

#[cfg(feature = "mp4")]
fn validate_private_sampler_provenance(
    sampler: &str,
    requested_grid_points: usize,
    transformer_evaluations: usize,
    owner: &H3PrivateFl2VaOwnerFacts,
) -> Result<()> {
    if sampler != H3SamplerKind::ComfyResMultistep.as_str()
        || requested_grid_points != usize::try_from(owner.requested_grid_points)?
        || transformer_evaluations != usize::try_from(owner.transformer_evaluations)?
    {
        bail!("private H3 sampler provenance differs from the prepared attempt authority")
    }
    Ok(())
}

/// Exact payload-free owner facts captured inside the singular server attempt
/// scope. This record is comparison data only; the prepared attempt and
/// activation evidence that consume it are non-Clone.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivateFl2VaOwnerFacts {
    pub work_identity_sha256: String,
    pub cancellation_scope_identity_sha256: String,
    pub admission_evidence_identity_sha256: String,
    pub artifact_qualification_identity_sha256: String,
    pub runtime_qualification_identity_sha256: String,
    pub device_id: String,
    pub device_ordinal: usize,
    pub execution_fingerprint: String,
    pub prepared_attempt_identity_sha256: String,
    pub target_budget_identity_sha256: String,
    pub component_set_identity_sha256: String,
    pub requested_grid_points: u32,
    pub transformer_evaluations: u32,
    pub predicted_device_peak_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub media: H3PrivateFl2VaMediaContract,
}

impl H3PrivateFl2VaOwnerFacts {
    pub fn validate(&self) -> Result<()> {
        self.media.validate()?;
        if self.device_id.trim().is_empty()
            || self.predicted_device_peak_bytes == 0
            || self.predicted_host_increment_bytes == 0
            || self.requested_grid_points < 2
            || self.transformer_evaluations != self.requested_grid_points - 1
            || [
                self.work_identity_sha256.as_str(),
                self.cancellation_scope_identity_sha256.as_str(),
                self.admission_evidence_identity_sha256.as_str(),
                self.artifact_qualification_identity_sha256.as_str(),
                self.runtime_qualification_identity_sha256.as_str(),
                self.execution_fingerprint.as_str(),
                self.prepared_attempt_identity_sha256.as_str(),
                self.target_budget_identity_sha256.as_str(),
                self.component_set_identity_sha256.as_str(),
            ]
            .into_iter()
            .any(|value| !valid_sha256(value))
        {
            bail!("private H3 owner facts are incomplete")
        }
        Ok(())
    }

    fn attempt_facts(
        &self,
        consumption: &H3PrivateAttemptConsumptionBinding,
    ) -> H3PrivateFl2VaAttemptFacts {
        H3PrivateFl2VaAttemptFacts {
            device_id: self.device_id.clone(),
            device_ordinal: self.device_ordinal,
            execution_fingerprint: self.execution_fingerprint.clone(),
            prepared_attempt_identity_sha256: self.prepared_attempt_identity_sha256.clone(),
            target_budget_identity_sha256: self.target_budget_identity_sha256.clone(),
            component_set_identity_sha256: self.component_set_identity_sha256.clone(),
            predicted_device_peak_bytes: self.predicted_device_peak_bytes,
            predicted_host_increment_bytes: self.predicted_host_increment_bytes,
            media: self.media.clone(),
            admission_evidence_identity_sha256: self.admission_evidence_identity_sha256.clone(),
            artifact_qualification_identity_sha256: self
                .artifact_qualification_identity_sha256
                .clone(),
            runtime_qualification_identity_sha256: self
                .runtime_qualification_identity_sha256
                .clone(),
            work_identity_sha256: self.work_identity_sha256.clone(),
            cancellation_scope_identity_sha256: self.cancellation_scope_identity_sha256.clone(),
            memory_ledger_sequence: consumption.memory_ledger_sequence,
            consumption_identity_sha256: consumption.identity_sha256.clone(),
        }
    }
}

/// Payload-free owner-ledger identity for one scheduler-issued attempt.
///
/// No memory byte count or caller boolean crosses this boundary. The complete
/// target budget is already covered by `target_budget_identity_sha256`; the
/// private overlap issuer derives every byte from that authenticated prepared
/// budget and uses this value only to prove which nonzero ledger sequence owns
/// it.
#[derive(Debug, Eq, PartialEq)]
pub struct H3PrivateSchedulerLedgerIdentity {
    work_identity_sha256: String,
    cancellation_scope_identity_sha256: String,
    memory_ledger_sequence: u64,
    admission_evidence_identity_sha256: String,
    artifact_qualification_identity_sha256: String,
    runtime_qualification_identity_sha256: String,
    execution_fingerprint: String,
    prepared_attempt_identity_sha256: String,
    target_budget_identity_sha256: String,
    component_set_identity_sha256: String,
    identity_sha256: String,
}

/// Non-Clone proof that one reviewed runtime record and the exact opened,
/// prepared, scheduler-ledger, and owner-scope facts jointly cover every
/// prerequisite retained by the public factory contract.
///
/// Only this module can issue the token. The ordinary public factory
/// projection never accepts it and remains fail-closed.
pub(crate) struct H3PrivateFactoryActivationEvidence {
    runtime_qualification: H3PrivateRuntimeQualificationAuthority,
    scheduler_ledger: H3PrivateSchedulerLedgerIdentity,
    coverage: [H3FactoryActivationPrerequisite; 9],
    factory_identity_sha256: String,
    opened_evidence_identity_sha256: String,
    work_identity_sha256: String,
    cancellation_scope_identity_sha256: String,
    admission_evidence_identity_sha256: String,
    artifact_qualification_identity_sha256: String,
    runtime_qualification_identity_sha256: String,
    device_id: String,
    device_ordinal: usize,
    execution_fingerprint: String,
    prepared_attempt_identity_sha256: String,
    target_budget_identity_sha256: String,
    component_set_identity_sha256: String,
    predicted_device_peak_bytes: u64,
    predicted_host_increment_bytes: u64,
    identity_sha256: String,
}

impl std::fmt::Debug for H3PrivateFactoryActivationEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("H3PrivateFactoryActivationEvidence")
            .field("identity_sha256", &self.identity_sha256)
            .field("device_id", &self.device_id)
            .field("device_ordinal", &self.device_ordinal)
            .finish_non_exhaustive()
    }
}

/// One-shot notification installed by the server owner immediately before a
/// private attempt is consumed. The runner must invoke it before constructing
/// even the Candle CUDA device; every later allocation boundary only verifies
/// that the commitment already succeeded. Dropping this value before that
/// boundary does not emit a false allocation commitment.
pub struct H3PrivateAllocationCommit {
    callback: Option<Box<dyn FnOnce() -> Result<()> + Send>>,
    committed: bool,
}

impl std::fmt::Debug for H3PrivateAllocationCommit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("H3PrivateAllocationCommit")
            .field("committed", &self.committed)
            .finish_non_exhaustive()
    }
}

impl H3PrivateAllocationCommit {
    pub fn new(callback: impl FnOnce() -> Result<()> + Send + 'static) -> Self {
        Self {
            callback: Some(Box::new(callback)),
            committed: false,
        }
    }

    pub(crate) fn commit_once(&mut self) -> Result<()> {
        if self.committed {
            return Ok(());
        }
        let callback = self
            .callback
            .take()
            .ok_or_else(|| anyhow!("private H3 allocation callback was already consumed"))?;
        callback()?;
        self.committed = true;
        Ok(())
    }

    pub(crate) const fn is_committed(&self) -> bool {
        self.committed
    }
}

impl H3PrivateSchedulerLedgerIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        work_identity_sha256: impl Into<String>,
        cancellation_scope_identity_sha256: impl Into<String>,
        memory_ledger_sequence: u64,
        admission_evidence_identity_sha256: impl Into<String>,
        artifact_qualification_identity_sha256: impl Into<String>,
        runtime_qualification_identity_sha256: impl Into<String>,
        execution_fingerprint: impl Into<String>,
        prepared_attempt_identity_sha256: impl Into<String>,
        target_budget_identity_sha256: impl Into<String>,
        component_set_identity_sha256: impl Into<String>,
    ) -> Result<Self> {
        let mut identity = Self {
            work_identity_sha256: work_identity_sha256.into(),
            cancellation_scope_identity_sha256: cancellation_scope_identity_sha256.into(),
            memory_ledger_sequence,
            admission_evidence_identity_sha256: admission_evidence_identity_sha256.into(),
            artifact_qualification_identity_sha256: artifact_qualification_identity_sha256.into(),
            runtime_qualification_identity_sha256: runtime_qualification_identity_sha256.into(),
            execution_fingerprint: execution_fingerprint.into(),
            prepared_attempt_identity_sha256: prepared_attempt_identity_sha256.into(),
            target_budget_identity_sha256: target_budget_identity_sha256.into(),
            component_set_identity_sha256: component_set_identity_sha256.into(),
            identity_sha256: String::new(),
        };
        identity.identity_sha256 = scheduler_ledger_identity(&identity);
        identity.revalidate()?;
        Ok(identity)
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub const fn memory_ledger_sequence(&self) -> u64 {
        self.memory_ledger_sequence
    }

    pub(crate) fn work_identity_sha256(&self) -> &str {
        &self.work_identity_sha256
    }

    pub(crate) fn cancellation_scope_identity_sha256(&self) -> &str {
        &self.cancellation_scope_identity_sha256
    }

    pub(crate) fn admission_evidence_identity_sha256(&self) -> &str {
        &self.admission_evidence_identity_sha256
    }

    pub(crate) fn artifact_qualification_identity_sha256(&self) -> &str {
        &self.artifact_qualification_identity_sha256
    }

    pub(crate) fn runtime_qualification_identity_sha256(&self) -> &str {
        &self.runtime_qualification_identity_sha256
    }

    pub(crate) fn execution_fingerprint(&self) -> &str {
        &self.execution_fingerprint
    }

    pub(crate) fn prepared_attempt_identity_sha256(&self) -> &str {
        &self.prepared_attempt_identity_sha256
    }

    pub(crate) fn target_budget_identity_sha256(&self) -> &str {
        &self.target_budget_identity_sha256
    }

    pub(crate) fn component_set_identity_sha256(&self) -> &str {
        &self.component_set_identity_sha256
    }

    pub(crate) fn revalidate(&self) -> Result<()> {
        if self.memory_ledger_sequence == 0
            || !valid_sha256(&self.work_identity_sha256)
            || !valid_sha256(&self.cancellation_scope_identity_sha256)
            || !valid_sha256(&self.admission_evidence_identity_sha256)
            || !valid_sha256(&self.artifact_qualification_identity_sha256)
            || !valid_sha256(&self.runtime_qualification_identity_sha256)
            || !valid_sha256(&self.execution_fingerprint)
            || !valid_sha256(&self.prepared_attempt_identity_sha256)
            || !valid_sha256(&self.target_budget_identity_sha256)
            || !valid_sha256(&self.component_set_identity_sha256)
            || self.identity_sha256 != scheduler_ledger_identity(self)
        {
            bail!("private H3 scheduler ledger identity is incomplete or changed")
        }
        Ok(())
    }
}

fn scheduler_ledger_identity(identity: &H3PrivateSchedulerLedgerIdentity) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-scheduler-ledger.v2\0");
    update_string(&mut digest, &identity.work_identity_sha256);
    update_string(&mut digest, &identity.cancellation_scope_identity_sha256);
    digest.update(identity.memory_ledger_sequence.to_le_bytes());
    update_string(&mut digest, &identity.admission_evidence_identity_sha256);
    update_string(
        &mut digest,
        &identity.artifact_qualification_identity_sha256,
    );
    update_string(&mut digest, &identity.runtime_qualification_identity_sha256);
    update_string(&mut digest, &identity.execution_fingerprint);
    update_string(&mut digest, &identity.prepared_attempt_identity_sha256);
    update_string(&mut digest, &identity.target_budget_identity_sha256);
    update_string(&mut digest, &identity.component_set_identity_sha256);
    format!("{:x}", digest.finalize())
}

impl H3PrivateFactoryActivationEvidence {
    pub(crate) fn derive(
        factory: &FrozenH3FactoryAuthority,
        runtime_qualification: H3PrivateRuntimeQualificationAuthority,
        opened: &H3PrivateOpenedActivationFacts,
        prepared: &H3PrivatePreparedFl2VaFactoryInputs,
        scheduler_ledger: H3PrivateSchedulerLedgerIdentity,
        owner: &H3PrivateFl2VaOwnerFacts,
    ) -> Result<Self> {
        runtime_qualification.revalidate()?;
        prepared.revalidate()?;
        scheduler_ledger.revalidate()?;
        owner.validate()?;
        let identities = factory
            .prepared_target_attempt_identities()
            .ok_or_else(|| anyhow!("private H3 factory has no prepared target identities"))?;
        if h3_factory_activation_prerequisites() != PRIVATE_ACTIVATION_COVERAGE.as_slice()
            || runtime_qualification.device_id() != owner.device_id
            || runtime_qualification.device_ordinal() != owner.device_ordinal
            || runtime_qualification.identity_sha256()
                != owner.runtime_qualification_identity_sha256
            || runtime_qualification.artifact_qualification_identity_sha256()
                != owner.artifact_qualification_identity_sha256
            || factory.device_id() != owner.device_id
            || factory.device_ordinal() != owner.device_ordinal
            || factory.execution_fingerprint() != owner.execution_fingerprint
            || factory.component_set_identity_sha256() != owner.component_set_identity_sha256
            || identities.0 != owner.prepared_attempt_identity_sha256
            || identities.1 != owner.target_budget_identity_sha256
            || prepared.execution_fingerprint() != owner.execution_fingerprint
            || prepared.prepared_attempt_identity_sha256() != owner.prepared_attempt_identity_sha256
            || prepared.target_budget_identity_sha256() != owner.target_budget_identity_sha256
            || prepared.predicted_device_peak_bytes() != owner.predicted_device_peak_bytes
            || prepared.predicted_host_increment_bytes() != owner.predicted_host_increment_bytes
            || scheduler_ledger.work_identity_sha256() != owner.work_identity_sha256
            || scheduler_ledger.cancellation_scope_identity_sha256()
                != owner.cancellation_scope_identity_sha256
            || scheduler_ledger.admission_evidence_identity_sha256()
                != owner.admission_evidence_identity_sha256
            || scheduler_ledger.artifact_qualification_identity_sha256()
                != owner.artifact_qualification_identity_sha256
            || scheduler_ledger.runtime_qualification_identity_sha256()
                != owner.runtime_qualification_identity_sha256
            || scheduler_ledger.execution_fingerprint() != owner.execution_fingerprint
            || scheduler_ledger.prepared_attempt_identity_sha256()
                != owner.prepared_attempt_identity_sha256
            || scheduler_ledger.target_budget_identity_sha256()
                != owner.target_budget_identity_sha256
            || scheduler_ledger.component_set_identity_sha256()
                != owner.component_set_identity_sha256
        {
            bail!("private H3 activation evidence crosses opened, prepared, ledger, or owner facts")
        }
        let mut evidence = Self {
            runtime_qualification,
            scheduler_ledger,
            coverage: PRIVATE_ACTIVATION_COVERAGE,
            factory_identity_sha256: factory.identity_sha256().into(),
            opened_evidence_identity_sha256: opened.identity_sha256.clone(),
            work_identity_sha256: owner.work_identity_sha256.clone(),
            cancellation_scope_identity_sha256: owner.cancellation_scope_identity_sha256.clone(),
            admission_evidence_identity_sha256: owner.admission_evidence_identity_sha256.clone(),
            artifact_qualification_identity_sha256: owner
                .artifact_qualification_identity_sha256
                .clone(),
            runtime_qualification_identity_sha256: owner
                .runtime_qualification_identity_sha256
                .clone(),
            device_id: owner.device_id.clone(),
            device_ordinal: owner.device_ordinal,
            execution_fingerprint: owner.execution_fingerprint.clone(),
            prepared_attempt_identity_sha256: owner.prepared_attempt_identity_sha256.clone(),
            target_budget_identity_sha256: owner.target_budget_identity_sha256.clone(),
            component_set_identity_sha256: owner.component_set_identity_sha256.clone(),
            predicted_device_peak_bytes: owner.predicted_device_peak_bytes,
            predicted_host_increment_bytes: owner.predicted_host_increment_bytes,
            identity_sha256: String::new(),
        };
        evidence.identity_sha256 = activation_evidence_identity(&evidence);
        evidence.revalidate_for(factory)?;
        Ok(evidence)
    }

    pub(crate) fn revalidate_for(&self, factory: &FrozenH3FactoryAuthority) -> Result<()> {
        self.runtime_qualification.revalidate()?;
        self.scheduler_ledger.revalidate()?;
        let identities = factory
            .prepared_target_attempt_identities()
            .ok_or_else(|| anyhow!("private H3 factory has no prepared target identities"))?;
        if self.coverage.as_slice() != h3_factory_activation_prerequisites()
            || self.coverage != PRIVATE_ACTIVATION_COVERAGE
            || self.factory_identity_sha256 != factory.identity_sha256()
            || self.device_id != factory.device_id()
            || self.device_ordinal != factory.device_ordinal()
            || self.execution_fingerprint != factory.execution_fingerprint()
            || self.prepared_attempt_identity_sha256 != identities.0
            || self.target_budget_identity_sha256 != identities.1
            || self.component_set_identity_sha256 != factory.component_set_identity_sha256()
            || self.runtime_qualification.device_id() != self.device_id
            || self.runtime_qualification.device_ordinal() != self.device_ordinal
            || self.runtime_qualification.identity_sha256()
                != self.runtime_qualification_identity_sha256
            || self
                .runtime_qualification
                .artifact_qualification_identity_sha256()
                != self.artifact_qualification_identity_sha256
            || self.scheduler_ledger.work_identity_sha256() != self.work_identity_sha256
            || self.scheduler_ledger.cancellation_scope_identity_sha256()
                != self.cancellation_scope_identity_sha256
            || self.scheduler_ledger.admission_evidence_identity_sha256()
                != self.admission_evidence_identity_sha256
            || self
                .scheduler_ledger
                .artifact_qualification_identity_sha256()
                != self.artifact_qualification_identity_sha256
            || self
                .scheduler_ledger
                .runtime_qualification_identity_sha256()
                != self.runtime_qualification_identity_sha256
            || self.scheduler_ledger.execution_fingerprint() != self.execution_fingerprint
            || self.scheduler_ledger.prepared_attempt_identity_sha256()
                != self.prepared_attempt_identity_sha256
            || self.scheduler_ledger.target_budget_identity_sha256()
                != self.target_budget_identity_sha256
            || self.scheduler_ledger.component_set_identity_sha256()
                != self.component_set_identity_sha256
            || !valid_sha256(&self.opened_evidence_identity_sha256)
            || !valid_sha256(&self.cancellation_scope_identity_sha256)
            || !valid_sha256(&self.admission_evidence_identity_sha256)
            || !valid_sha256(&self.artifact_qualification_identity_sha256)
            || !valid_sha256(&self.runtime_qualification_identity_sha256)
            || self.predicted_device_peak_bytes == 0
            || self.predicted_host_increment_bytes == 0
            || self.identity_sha256 != activation_evidence_identity(self)
        {
            bail!("private H3 activation evidence changed after issuance")
        }
        Ok(())
    }
}

fn activation_evidence_identity(evidence: &H3PrivateFactoryActivationEvidence) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-factory-activation.v2\0");
    for value in [
        evidence.runtime_qualification.identity_sha256(),
        evidence.runtime_qualification.record_file_sha256(),
        evidence.scheduler_ledger.identity_sha256(),
        evidence.factory_identity_sha256.as_str(),
        evidence.opened_evidence_identity_sha256.as_str(),
        evidence.work_identity_sha256.as_str(),
        evidence.cancellation_scope_identity_sha256.as_str(),
        evidence.admission_evidence_identity_sha256.as_str(),
        evidence.artifact_qualification_identity_sha256.as_str(),
        evidence.runtime_qualification_identity_sha256.as_str(),
        evidence.device_id.as_str(),
        evidence.execution_fingerprint.as_str(),
        evidence.prepared_attempt_identity_sha256.as_str(),
        evidence.target_budget_identity_sha256.as_str(),
        evidence.component_set_identity_sha256.as_str(),
    ] {
        update_string(&mut digest, value);
    }
    digest.update((evidence.device_ordinal as u64).to_le_bytes());
    digest.update(evidence.predicted_device_peak_bytes.to_le_bytes());
    digest.update(evidence.predicted_host_increment_bytes.to_le_bytes());
    for prerequisite in evidence.coverage {
        update_string(&mut digest, activation_prerequisite_id(prerequisite));
    }
    format!("{:x}", digest.finalize())
}

const fn activation_prerequisite_id(prerequisite: H3FactoryActivationPrerequisite) -> &'static str {
    match prerequisite {
        H3FactoryActivationPrerequisite::OpenedComponentMemoryEvidence => "opened-memory",
        H3FactoryActivationPrerequisite::PreparedCheckpointExecutionEcho => "checkpoint-echo",
        H3FactoryActivationPrerequisite::ConsumingTargetLifetimeTransitions => "lifetime",
        H3FactoryActivationPrerequisite::RetainedTensorOverlapBudget => "overlap",
        H3FactoryActivationPrerequisite::HostLayoutAndTransientBudget => "host-transients",
        H3FactoryActivationPrerequisite::EndpointPreprocessTransientBudget => "endpoint-transients",
        H3FactoryActivationPrerequisite::PerAttemptRuntimeConstruction => "per-attempt-runtime",
        H3FactoryActivationPrerequisite::OneShotSchedulerLease => "one-shot-lease",
        H3FactoryActivationPrerequisite::SameAttemptCancellationCoverage => "cancellation",
    }
}

/// Authenticated, non-Clone runtime-bound authority.
///
/// The opened record descriptor is retained so preparation and dispatch can
/// re-hash the same file. Raw bounds are not exposed to the server; the private
/// opened-evidence binder consumes them directly.
pub struct H3PrivateRuntimeQualificationAuthority {
    storage: RuntimeQualificationStorage,
    record_file_sha256: String,
    record: H3PrivateRuntimeQualificationRecord,
    bounds: H3PrivateQualifiedRuntimeBounds,
    device_id: String,
    device_ordinal: usize,
    compute_capability: (u16, u16),
}

impl std::fmt::Debug for H3PrivateRuntimeQualificationAuthority {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("H3PrivateRuntimeQualificationAuthority")
            .field("record_file_sha256", &self.record_file_sha256)
            .field("identity_sha256", &self.record.identity_sha256)
            .finish_non_exhaustive()
    }
}

impl H3PrivateRuntimeQualificationAuthority {
    pub fn identity_sha256(&self) -> &str {
        &self.record.identity_sha256
    }

    pub fn record_file_sha256(&self) -> &str {
        &self.record_file_sha256
    }

    pub fn artifact_qualification_identity_sha256(&self) -> &str {
        &self.record.artifact_qualification_identity_sha256
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub const fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub const fn compute_capability(&self) -> (u16, u16) {
        self.compute_capability
    }

    pub(crate) fn bounds(&self) -> &H3PrivateQualifiedRuntimeBounds {
        &self.bounds
    }

    #[cfg(feature = "mp4")]
    fn validate_prepared_envelope(&self, request: &H3FactoryPreparedRequestInput) -> Result<()> {
        self.validate_prepared_envelope_with_turbo(request, None)
    }

    #[cfg(feature = "mp4")]
    fn reviewed_turbo_steps(turbo: Option<&H3FactoryTurboAdapterAuthority>) -> Option<u32> {
        turbo.map(|turbo| turbo.grid_points)
    }

    /// Validate a prepared request against this qualification's envelope,
    /// admitting a reviewed Turbo tier's own step count when an authenticated
    /// adapter is present. `None` is byte-identical to the previous behaviour.
    #[cfg(feature = "mp4")]
    fn validate_prepared_envelope_with_turbo(
        &self,
        request: &H3FactoryPreparedRequestInput,
        turbo: Option<&H3FactoryTurboAdapterAuthority>,
    ) -> Result<()> {
        self.record
            .envelope
            .validate_prepared_with_reviewed_steps(request, Self::reviewed_turbo_steps(turbo))
    }

    pub(crate) fn revalidate(&self) -> Result<()> {
        self.storage
            .revalidate(&self.record_file_sha256, &self.record)
    }
}

enum RuntimeQualificationStorage {
    External {
        path: PathBuf,
        file: File,
    },
    #[cfg(feature = "h3")]
    Embedded,
    #[cfg(feature = "h3")]
    /// The compiled public profile. It remembers the reviewed Turbo step count
    /// it was minted with, so revalidation stays exactly as strict as minting.
    PublicCompiled {
        turbo_steps: Option<u32>,
    },
}

impl RuntimeQualificationStorage {
    fn revalidate(
        &self,
        record_file_sha256: &str,
        record: &H3PrivateRuntimeQualificationRecord,
    ) -> Result<()> {
        match self {
            Self::External { path, file } => {
                revalidate_runtime_qualification_file(path, file, record_file_sha256, record)
            }
            #[cfg(feature = "h3")]
            Self::Embedded => {
                let bytes = include_bytes!("../../assets/minimax-h3-runtime-qualification.json");
                if format!("{:x}", Sha256::digest(bytes)) != record_file_sha256
                    || runtime_qualification_identity(record) != record.identity_sha256
                {
                    bail!("embedded H3 runtime qualification changed after authentication")
                }
                record.bounds.validate()
            }
            #[cfg(feature = "h3")]
            Self::PublicCompiled { turbo_steps } => {
                validate_public_runtime_profile_with_turbo(record, record_file_sha256, *turbo_steps)
            }
        }
    }
}

#[cfg(feature = "h3")]
const PUBLIC_RUNTIME_PROFILE_SCHEMA: &str = "mold.minimax-h3.public-runtime-profile.v1";
#[cfg(feature = "h3")]
const PUBLIC_RUNTIME_PROFILE_DECISION: &str = "supported-compact-fl2va-sm89";

/// Margin applied to every measured workspace bound, then rounded up to
/// [`PUBLIC_RUNTIME_BOUND_GRID_BYTES`].
///
/// One render is one sample. 15% covers allocator and driver variance plus the
/// small shape headroom the envelope still allows — the qualifying render used
/// 39,768 of the envelope's 39,776 packed rows and 1,050 of its 1,058 text
/// rows, so it sat essentially at the ceiling and the margin does not have to
/// absorb a much larger shape.
#[cfg(feature = "h3")]
const PUBLIC_RUNTIME_BOUND_MARGIN_PERCENT: u64 = 115;

/// 64 MiB. Coarse enough that a re-measurement moves a bound only when it
/// moves materially, fine enough that a 200 MB workspace is not rounded to
/// three times its size.
#[cfg(feature = "h3")]
const PUBLIC_RUNTIME_BOUND_GRID_BYTES: u64 = 64 * 1024 * 1024;

/// `ceil_to_grid(observed * margin)`, the one policy every measured bound below
/// is derived by. Always at or above `observed`.
#[cfg(feature = "h3")]
const fn public_runtime_bound(observed_bytes: u64) -> u64 {
    let with_margin = observed_bytes * PUBLIC_RUNTIME_BOUND_MARGIN_PERCENT / 100;
    with_margin.next_multiple_of(PUBLIC_RUNTIME_BOUND_GRID_BYTES)
}

/// Compiled bounds for the reviewed compact FL2VA runtime.
///
/// Every workspace figure is `public_runtime_bound(observed)` over the first
/// real FL2VA render: 2026-08-19 on hal9000 (RTX 4090, SM89, 24 GB), 1344x768,
/// 124 frames at 24 fps, 21 steps, 1216 s, captured by
/// `private_runtime_observer` and recorded on issue #827. The observations are
/// named beside each value so a future re-measurement can be applied by
/// changing one number and re-running the same policy.
///
/// The policy reproduces four of the previous L40S-era caps exactly
/// (`fixed_runtime_device`, `condition_vae`, `decoder_tile`, `audio_decode`),
/// which is the reason to trust it, and corrects the two that were guesses:
/// attention fell 10.13 -> 7.11 GB and FFN 15.30 -> 8.79 GB. Two rose, because
/// the old values carried less than this margin over what the hardware
/// actually used; the policy is applied uniformly rather than only where it
/// flatters the result.
///
/// The three host capacity bounds are NOT measurements — they are the
/// pipeline's own allocation limits, so they stay tied to those constants. A
/// render that legitimately produces a larger MP4 must still be charged for
/// the buffer the pipeline is willing to allocate.
#[cfg(feature = "h3")]
fn public_runtime_bounds() -> H3PrivateRuntimeBoundRecord {
    H3PrivateRuntimeBoundRecord {
        // observed 659_701_760
        fixed_runtime_host_bytes: public_runtime_bound(659_701_760),
        // observed 477_298_688
        fixed_runtime_device_bytes: public_runtime_bound(477_298_688),
        // observed 3_400_171_520
        qwen_activation_workspace_bytes: public_runtime_bound(3_400_171_520),
        // Observed 0: the VAE construction transient never rose above the
        // weights themselves in the qualifying render. A zero bound is not
        // admissible (the reload stages through it and the validator refuses
        // zero), so the previous 64 MiB allowance is retained as a floor.
        vae_construction_device_workspace_bytes: 67_108_864,
        // observed 366_027_840
        condition_vae_workspace_device_bytes: public_runtime_bound(366_027_840),
        // observed 6_172_029_280
        attention_workspace_device_bytes: public_runtime_bound(6_172_029_280),
        // observed 7_641_748_832
        ffn_workspace_device_bytes: public_runtime_bound(7_641_748_832),
        // observed 1_338_688_660
        decoder_tile_workspace_device_bytes: public_runtime_bound(1_338_688_660),
        // observed 204_867_120
        audio_decode_workspace_device_bytes: public_runtime_bound(204_867_120),
        encoded_video_host_bytes_bound: super::pipeline::SMALL_ENCODED_VIDEO_HOST_BYTES_BOUND,
        thumbnail_host_bytes_bound: super::pipeline::SMALL_THUMBNAIL_HOST_BYTES_BOUND,
        mux_output_host_bytes_bound: super::pipeline::SMALL_MUX_OUTPUT_HOST_BYTES_BOUND,
        aac_mux_staging_host_bytes: super::pipeline::SMALL_AAC_MUX_STAGING_HOST_BYTES,
    }
}

#[cfg(feature = "h3")]
fn public_runtime_envelope() -> H3PrivateRuntimeEnvelopeRecord {
    public_runtime_envelope_for_steps(contract::COMFY_DEFAULT_STEPS)
}

/// The reviewed public envelope at a given step count.
///
/// A Turbo tier renders the same canvas — 1344x768, 124 frames, 24 fps, one
/// first-frame endpoint, identical row ceilings — and moves only the step
/// count, which is the whole point of the distillation. Callers may only pass
/// a count an authenticated adapter declares.
#[cfg(feature = "h3")]
fn public_runtime_envelope_for_steps(max_steps: u32) -> H3PrivateRuntimeEnvelopeRecord {
    H3PrivateRuntimeEnvelopeRecord {
        width: contract::DEFAULT_WIDTH,
        height: contract::DEFAULT_HEIGHT,
        frames: contract::MIN_FRAMES,
        fps: contract::FIXED_FPS,
        batch_size: 1,
        max_steps,
        endpoint_count: 1,
        endpoint_anchor: "first".into(),
        max_qwen_output_text_rows: 1_058,
        max_qwen_vision_rows: 4_032,
        max_condition_visual_rows: 1_008,
        max_target_video_rows: 37_296,
        max_target_audio_rows: 414,
        max_total_packed_rows: 39_776,
    }
}

#[cfg(feature = "h3")]
fn validate_public_runtime_profile(
    record: &H3PrivateRuntimeQualificationRecord,
    profile_sha256: &str,
) -> Result<()> {
    validate_public_runtime_profile_with_turbo(record, profile_sha256, None)
}

/// Validate the public profile, admitting a reviewed Turbo tier's step count
/// when an authenticated adapter declares it. `None` keeps the 21-step pin.
#[cfg(feature = "h3")]
fn validate_public_runtime_profile_with_turbo(
    record: &H3PrivateRuntimeQualificationRecord,
    profile_sha256: &str,
    turbo_steps: Option<u32>,
) -> Result<()> {
    record.envelope.validate_with_reviewed_steps(turbo_steps)?;
    record.bounds.validate()?;
    if record.schema != PUBLIC_RUNTIME_PROFILE_SCHEMA
        || record.decision != PUBLIC_RUNTIME_PROFILE_DECISION
        || record.canonical_model != contract::FL2VA_COMFY
        || record.task != "fl2va"
        || record.compute_capability != [8, 9]
        || record.identity_sha256 != runtime_qualification_identity(record)
        || profile_sha256 != record.identity_sha256
    {
        bail!("public H3 runtime profile changed or is not the supported SM89 profile")
    }
    Ok(())
}

#[cfg(feature = "h3")]
#[allow(clippy::too_many_arguments)]
fn public_runtime_qualification(
    artifact: &H3PrivateArtifactQualificationReport,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: (u16, u16),
    attention_runtime_identity_sha256: &str,
    attention_kernel_identity: &str,
    attention_qualification_sha256: &str,
    turbo_steps: Option<u32>,
) -> Result<H3PrivateRuntimeQualificationAuthority> {
    if compute_capability != (8, 9)
        || artifact.canonical_model != contract::FL2VA_COMFY
        || artifact.task != "fl2va"
        || !valid_sha256(attention_runtime_identity_sha256)
        || !valid_sha256(attention_qualification_sha256)
        || attention_kernel_identity.is_empty()
    {
        bail!("public H3 runtime requires the exact compact FL2VA SM89 authority")
    }
    let mut record = H3PrivateRuntimeQualificationRecord {
        schema: PUBLIC_RUNTIME_PROFILE_SCHEMA.into(),
        decision: PUBLIC_RUNTIME_PROFILE_DECISION.into(),
        canonical_model: contract::FL2VA_COMFY.into(),
        task: "fl2va".into(),
        campaign_source_sha: exact_h3_runtime_build_source_sha()?.into(),
        campaign_runtime_code_identity_sha256: super::PRIVATE_RUNTIME_CODE_IDENTITY_SHA256.into(),
        campaign_bootstrap_record_sha256: sha256_domain("public-profile-bootstrap"),
        campaign_bootstrap_identity_sha256: sha256_domain("public-profile-identity"),
        measured_server_executable_relative_path: "public-runtime-profile".into(),
        measured_server_executable_sha256: sha256_domain("public-runtime-executable"),
        authorization_record_sha256: artifact.authorization_record_sha256.clone(),
        authorization_source_document_sha256: artifact.authorization_source_document_sha256.clone(),
        artifact_qualification_identity_sha256: artifact.qualification_identity_sha256.clone(),
        artifact_total_bytes: artifact.total_bytes,
        device_id: device_id.into(),
        device_ordinal,
        compute_capability: [8, 9],
        attention_runtime_identity_sha256: attention_runtime_identity_sha256.into(),
        attention_kernel_identity: attention_kernel_identity.into(),
        attention_qualification_sha256: attention_qualification_sha256.into(),
        campaign_process: H3PrivateRuntimeProcessObservation {
            process_id: 0,
            process_start_time_ticks: 0,
            linux_boot_id_sha256: sha256_domain("public-runtime-boot"),
            executable_device: 0,
            executable_inode: 0,
            executable_bytes: 0,
            executable_sha256: sha256_domain("public-runtime-executable"),
            launch_argv_sha256: sha256_domain("public-runtime-argv"),
            launch_environment_sha256: sha256_domain("public-runtime-environment"),
            cuda_driver_version: 0,
            cuda_toolkit_version: 0,
        },
        envelope: public_runtime_envelope_for_steps(
            turbo_steps.unwrap_or(contract::COMFY_DEFAULT_STEPS),
        ),
        bounds: public_runtime_bounds(),
        evidence_artifacts: Vec::new(),
        identity_sha256: String::new(),
    };
    record.identity_sha256 = runtime_qualification_identity(&record);
    let profile_sha256 = record.identity_sha256.clone();
    validate_public_runtime_profile_with_turbo(&record, &profile_sha256, turbo_steps)?;
    let bounds = record.bounds.clone().into_authority();
    Ok(H3PrivateRuntimeQualificationAuthority {
        storage: RuntimeQualificationStorage::PublicCompiled { turbo_steps },
        record_file_sha256: profile_sha256,
        record,
        bounds,
        device_id: device_id.into(),
        device_ordinal,
        compute_capability,
    })
}

#[cfg(feature = "h3")]
fn sha256_domain(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn revalidate_runtime_qualification_file(
    path: &Path,
    file: &File,
    record_file_sha256: &str,
    record: &H3PrivateRuntimeQualificationRecord,
) -> Result<()> {
    let current = open_regular_file_no_follow(path).with_context(|| {
        format!(
            "failed to reopen private H3 runtime qualification {}",
            path.display()
        )
    })?;
    let opened_metadata = file.metadata()?;
    let current_metadata = current.metadata()?;
    if !opened_metadata.is_file()
        || !current_metadata.is_file()
        || opened_metadata.len() != current_metadata.len()
        || sha256_open_file(file)? != record_file_sha256
        || sha256_open_file(&current)? != record_file_sha256
        || runtime_qualification_identity(record) != record.identity_sha256
    {
        bail!("private H3 runtime qualification changed after authentication")
    }
    record.bounds.validate()?;
    Ok(())
}

/// Record bytes that passed the fixed reviewed-hash gate before any model
/// artifact path was opened. Cross-binding to the independently qualified
/// artifacts and live owner route happens only after the bulk qualification
/// completes.
struct H3PrivateReviewedRuntimeQualification {
    storage: RuntimeQualificationStorage,
    record_file_sha256: String,
    record: H3PrivateRuntimeQualificationRecord,
}

impl H3PrivateReviewedRuntimeQualification {
    fn revalidate(&self) -> Result<()> {
        self.storage
            .revalidate(&self.record_file_sha256, &self.record)
    }

    /// Reject a reviewed record that cannot authorize this concrete route
    /// before the caller starts the multi-artifact qualification pass.
    fn validate_route(
        &self,
        device_id: &str,
        device_ordinal: usize,
        compute_capability: (u16, u16),
    ) -> Result<()> {
        self.revalidate()?;
        if self.record.device_id != device_id
            || self.record.device_ordinal != device_ordinal
            || self.record.compute_capability != [compute_capability.0, compute_capability.1]
        {
            #[cfg(feature = "h3")]
            if self.record.compute_capability == [compute_capability.0, compute_capability.1] {
                return Ok(());
            }
            bail!("private H3 reviewed runtime qualification cannot authorize this CUDA route")
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn authenticate(
        self,
        artifact_qualification: &H3PrivateArtifactQualificationReport,
        device_id: &str,
        device_ordinal: usize,
        compute_capability: (u16, u16),
        attention_runtime_identity_sha256: &str,
        attention_kernel_identity: &str,
        attention_qualification_sha256: &str,
    ) -> Result<H3PrivateRuntimeQualificationAuthority> {
        self.revalidate()?;
        validate_runtime_qualification_record_binding(
            &self.record,
            artifact_qualification,
            device_id,
            device_ordinal,
            compute_capability,
            attention_runtime_identity_sha256,
            attention_kernel_identity,
            attention_qualification_sha256,
        )?;
        let bounds = self.record.bounds.clone().into_authority();
        let authority = H3PrivateRuntimeQualificationAuthority {
            storage: self.storage,
            record_file_sha256: self.record_file_sha256,
            record: self.record,
            bounds,
            device_id: device_id.to_string(),
            device_ordinal,
            compute_capability,
        };
        authority.revalidate()?;
        Ok(authority)
    }
}

fn open_reviewed_h3_private_runtime_qualification(
    path: &Path,
    reviewed_record_sha256: &[&str],
) -> Result<H3PrivateReviewedRuntimeQualification> {
    // This is deliberately the first check. An empty authorization set must
    // reject without inspecting even the record pathname.
    if reviewed_record_sha256.is_empty() {
        bail!("private H3 runtime qualification has no reviewed evidence allowlist")
    }
    let executing_source_sha = exact_h3_runtime_build_source_sha()?;
    open_reviewed_h3_private_runtime_qualification_for_source(
        path,
        reviewed_record_sha256,
        executing_source_sha,
        super::PRIVATE_RUNTIME_CODE_IDENTITY_SHA256,
    )
}

fn open_reviewed_h3_private_runtime_qualification_for_source(
    path: &Path,
    reviewed_record_sha256: &[&str],
    executing_source_sha: &str,
    executing_runtime_code_identity_sha256: &str,
) -> Result<H3PrivateReviewedRuntimeQualification> {
    if reviewed_record_sha256.is_empty() {
        bail!("private H3 runtime qualification has no reviewed evidence allowlist")
    }
    validate_h3_runtime_build_identity(
        executing_source_sha,
        executing_runtime_code_identity_sha256,
    )?;
    #[cfg(feature = "h3")]
    {
        let _ = path;
        let bytes = include_bytes!("../../assets/minimax-h3-runtime-qualification.json");
        let record_file_sha256 = format!("{:x}", Sha256::digest(bytes));
        if !reviewed_record_sha256.contains(&record_file_sha256.as_str()) {
            bail!("embedded H3 runtime qualification is not in the reviewed evidence allowlist")
        }
        let record: H3PrivateRuntimeQualificationRecord = serde_json::from_slice(bytes)
            .context("invalid embedded H3 runtime qualification record")?;
        validate_runtime_qualification_record_shape(&record)?;
        if record.campaign_runtime_code_identity_sha256 != executing_runtime_code_identity_sha256 {
            bail!("embedded H3 runtime qualification was measured by different runtime code")
        }
        let reviewed = H3PrivateReviewedRuntimeQualification {
            storage: RuntimeQualificationStorage::Embedded,
            record_file_sha256,
            record,
        };
        reviewed.revalidate()?;
        Ok(reviewed)
    }
    #[cfg(not(feature = "h3"))]
    {
        if !path.is_absolute() {
            bail!("private H3 runtime qualification path must be absolute")
        }
        let mut file = open_regular_file_no_follow(path).with_context(|| {
            format!(
                "failed to open private H3 runtime qualification {}",
                path.display()
            )
        })?;
        let metadata = file.metadata()?;
        if metadata.len() == 0 || metadata.len() > MAX_RUNTIME_QUALIFICATION_BYTES {
            bail!("private H3 runtime qualification record has an invalid size")
        }
        let record_file_sha256 = sha256_open_file(&file)?;
        if !reviewed_record_sha256.contains(&record_file_sha256.as_str()) {
            bail!(
                "private H3 runtime qualification record is not in the reviewed evidence allowlist"
            )
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.read_to_end(&mut bytes)?;
        let record: H3PrivateRuntimeQualificationRecord = serde_json::from_slice(&bytes)
            .context("invalid private H3 runtime qualification record")?;
        validate_runtime_qualification_record_shape(&record)?;
        if record.campaign_runtime_code_identity_sha256 != executing_runtime_code_identity_sha256 {
            bail!("private H3 runtime qualification was measured by different runtime code")
        }
        let reviewed = H3PrivateReviewedRuntimeQualification {
            storage: RuntimeQualificationStorage::External {
                path: path.to_path_buf(),
                file,
            },
            record_file_sha256,
            record,
        };
        reviewed.revalidate()?;
        Ok(reviewed)
    }
}

/// Authenticate the small, presentation-only private H3 authority chain.
///
/// This reads only the owner-protected authorization wrapper, its exact source
/// document, and the bounded reviewed runtime record. It never opens model
/// artifacts and cannot construct an engine or mutate runtime availability.
pub fn authenticate_h3_private_presentation(
    models_root: &Path,
    authorization_record: &Path,
    runtime_qualification_record: &Path,
    routes: &[H3PrivatePresentationRoute<'_>],
) -> Result<H3PrivatePresentationAuthority> {
    if !reviewed_h3_private_runtime_available_for_task(Task::Fl2va) {
        bail!("private H3 presentation has no reviewed FL2VA qualification")
    }
    if routes.is_empty() {
        bail!("private H3 presentation requires one live schedulable CUDA route")
    }
    let scope = validate_private_presentation_scope(
        models_root,
        authorization_record,
        runtime_qualification_record,
    )?;
    authenticate_h3_private_presentation_with_scope(
        scope,
        runtime_qualification_record,
        REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256,
        exact_h3_runtime_build_source_sha()?,
        super::PRIVATE_RUNTIME_CODE_IDENTITY_SHA256,
        routes,
        None,
    )
}

/// Authenticate the compiled public H3 profile against one live SM89 CUDA
/// route. This presentation check opens neither model weights nor external
/// compliance files; generation separately verifies every artifact and applies
/// the compiled conservative memory profile.
#[cfg(feature = "h3")]
pub fn authenticate_h3_public_presentation(
    routes: &[H3PrivatePresentationRoute<'_>],
) -> Result<H3PrivatePresentationAuthority> {
    if routes.is_empty() {
        bail!("MiniMax H3 presentation requires one live schedulable CUDA route")
    }
    let route = routes
        .iter()
        .find(|route| route.compute_capability == (8, 9))
        .ok_or_else(|| anyhow!("public H3 requires one live schedulable SM89 CUDA route"))?;
    H3AttentionRuntimeAuthority::qualify_flash_attention_v2(
        H3AttentionDevice::Cuda {
            compute_capability: Some(route.compute_capability),
        },
        H3AttentionModelContract::released_bf16(),
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    Ok(H3PrivatePresentationAuthority {
        canonical_model: contract::FL2VA_COMFY.into(),
        task: Task::Fl2va,
        device_id: route.device_id.to_string(),
        device_ordinal: route.device_ordinal,
        compute_capability: route.compute_capability,
    })
}

fn authenticate_h3_private_presentation_with_scope(
    scope: super::private_qualification::ValidatedPrivateScope,
    runtime_qualification_record: &Path,
    reviewed_record_sha256: &[&str],
    executing_source_sha: &str,
    executing_runtime_code_identity_sha256: &str,
    routes: &[H3PrivatePresentationRoute<'_>],
    test_attention_identity: Option<(&str, &str)>,
) -> Result<H3PrivatePresentationAuthority> {
    let reviewed = open_reviewed_h3_private_runtime_qualification_for_source(
        runtime_qualification_record,
        reviewed_record_sha256,
        executing_source_sha,
        executing_runtime_code_identity_sha256,
    )?;
    if reviewed.record.canonical_model != contract::FL2VA_COMFY
        || reviewed.record.task != "fl2va"
        || reviewed.record.authorization_record_sha256 != scope.authorization_record_sha256()
        || reviewed.record.authorization_source_document_sha256
            != scope.authorization_source_document_sha256()
    {
        bail!("private H3 presentation authority does not match the reviewed FL2VA scope")
    }
    let route = routes
        .iter()
        .find(|route| {
            reviewed.record.device_id == route.device_id
                && reviewed.record.device_ordinal == route.device_ordinal
                && reviewed.record.compute_capability
                    == [route.compute_capability.0, route.compute_capability.1]
        })
        .ok_or_else(|| {
            anyhow!("private H3 presentation has no live route matching reviewed evidence")
        })?;
    reviewed.validate_route(
        route.device_id,
        route.device_ordinal,
        route.compute_capability,
    )?;
    let attention;
    let (attention_runtime_identity_sha256, attention_kernel_identity) =
        if let Some(identity) = test_attention_identity {
            identity
        } else {
            attention = H3AttentionRuntimeAuthority::qualify_flash_attention_v2(
                H3AttentionDevice::Cuda {
                    compute_capability: Some(route.compute_capability),
                },
                H3AttentionModelContract::released_bf16(),
            )
            .map_err(|error| anyhow!(error.to_string()))?;
            (attention.identity_sha256(), attention.kernel().identity())
        };
    if reviewed.record.attention_runtime_identity_sha256 != attention_runtime_identity_sha256
        || reviewed.record.attention_kernel_identity != attention_kernel_identity
    {
        bail!("private H3 presentation attention authority does not match the live CUDA route")
    }
    scope.revalidate()?;
    reviewed.revalidate()?;
    Ok(H3PrivatePresentationAuthority {
        canonical_model: reviewed.record.canonical_model.clone(),
        task: Task::Fl2va,
        device_id: route.device_id.to_string(),
        device_ordinal: route.device_ordinal,
        compute_capability: route.compute_capability,
    })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn authenticate_h3_private_presentation_for_test(
    models_root: &Path,
    authorization_record: &Path,
    runtime_qualification_record: &Path,
    reviewed_authorization_evidence_sha256: &str,
    reviewed_runtime_record_sha256: &[&str],
    executing_source_sha: &str,
    executing_runtime_code_identity_sha256: &str,
    routes: &[H3PrivatePresentationRoute<'_>],
    attention_runtime_identity_sha256: &str,
    attention_kernel_identity: &str,
) -> Result<H3PrivatePresentationAuthority> {
    let scope = validate_private_presentation_scope_against_evidence(
        models_root,
        authorization_record,
        runtime_qualification_record,
        reviewed_authorization_evidence_sha256,
    )?;
    authenticate_h3_private_presentation_with_scope(
        scope,
        runtime_qualification_record,
        reviewed_runtime_record_sha256,
        executing_source_sha,
        executing_runtime_code_identity_sha256,
        routes,
        Some((attention_runtime_identity_sha256, attention_kernel_identity)),
    )
}

/// Authenticate an exact runtime-qualification record against the reviewed
/// record allowlist and the independently produced artifact qualification.
///
/// The source-controlled allowlist is fixed independently of the supplied
/// record, so this remains a fail-closed seam rather than a caller-configurable
/// approval gate.
#[allow(clippy::too_many_arguments)]
pub fn authenticate_h3_private_runtime_qualification(
    path: &Path,
    artifact_qualification: &H3PrivateArtifactQualificationReport,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: (u16, u16),
    attention_runtime_identity_sha256: &str,
    attention_kernel_identity: &str,
    attention_qualification_sha256: &str,
) -> Result<H3PrivateRuntimeQualificationAuthority> {
    authenticate_h3_private_runtime_qualification_with_allowlist(
        path,
        artifact_qualification,
        device_id,
        device_ordinal,
        compute_capability,
        attention_runtime_identity_sha256,
        attention_kernel_identity,
        attention_qualification_sha256,
        REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256,
    )
}

#[allow(clippy::too_many_arguments)]
fn authenticate_h3_private_runtime_qualification_with_allowlist(
    path: &Path,
    artifact_qualification: &H3PrivateArtifactQualificationReport,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: (u16, u16),
    attention_runtime_identity_sha256: &str,
    attention_kernel_identity: &str,
    attention_qualification_sha256: &str,
    reviewed_record_sha256: &[&str],
) -> Result<H3PrivateRuntimeQualificationAuthority> {
    open_reviewed_h3_private_runtime_qualification(path, reviewed_record_sha256)?.authenticate(
        artifact_qualification,
        device_id,
        device_ordinal,
        compute_capability,
        attention_runtime_identity_sha256,
        attention_kernel_identity,
        attention_qualification_sha256,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn authenticate_h3_private_runtime_qualification_for_source(
    path: &Path,
    artifact_qualification: &H3PrivateArtifactQualificationReport,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: (u16, u16),
    attention_runtime_identity_sha256: &str,
    attention_kernel_identity: &str,
    attention_qualification_sha256: &str,
    reviewed_record_sha256: &[&str],
    executing_source_sha: &str,
    executing_runtime_code_identity_sha256: &str,
) -> Result<H3PrivateRuntimeQualificationAuthority> {
    open_reviewed_h3_private_runtime_qualification_for_source(
        path,
        reviewed_record_sha256,
        executing_source_sha,
        executing_runtime_code_identity_sha256,
    )?
    .authenticate(
        artifact_qualification,
        device_id,
        device_ordinal,
        compute_capability,
        attention_runtime_identity_sha256,
        attention_kernel_identity,
        attention_qualification_sha256,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_runtime_qualification_record_binding(
    record: &H3PrivateRuntimeQualificationRecord,
    artifact: &H3PrivateArtifactQualificationReport,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: (u16, u16),
    attention_runtime_identity_sha256: &str,
    attention_kernel_identity: &str,
    attention_qualification_sha256: &str,
) -> Result<()> {
    validate_runtime_qualification_record_shape(record)?;
    #[cfg(feature = "h3")]
    let route_matches = record.compute_capability == [compute_capability.0, compute_capability.1];
    #[cfg(not(feature = "h3"))]
    let route_matches = record.device_id == device_id
        && record.device_ordinal == device_ordinal
        && record.compute_capability == [compute_capability.0, compute_capability.1];
    #[cfg(feature = "h3")]
    let _ = (device_id, device_ordinal);
    if artifact.canonical_model != contract::FL2VA_COMFY
        || artifact.task != "fl2va"
        || record.authorization_record_sha256 != artifact.authorization_record_sha256
        || record.authorization_source_document_sha256
            != artifact.authorization_source_document_sha256
        || record.artifact_qualification_identity_sha256 != artifact.qualification_identity_sha256
        || record.artifact_total_bytes != artifact.total_bytes
        || !route_matches
        || record.attention_runtime_identity_sha256 != attention_runtime_identity_sha256
        || record.attention_kernel_identity != attention_kernel_identity
        || record.attention_qualification_sha256 != attention_qualification_sha256
    {
        bail!("private H3 runtime qualification differs from artifact, device, or kernel authority")
    }
    Ok(())
}

pub(crate) fn validate_runtime_qualification_record_shape(
    record: &H3PrivateRuntimeQualificationRecord,
) -> Result<()> {
    record.envelope.validate()?;
    record.bounds.validate()?;
    let sha_values = [
        record.campaign_runtime_code_identity_sha256.as_str(),
        record.campaign_bootstrap_record_sha256.as_str(),
        record.campaign_bootstrap_identity_sha256.as_str(),
        record.measured_server_executable_sha256.as_str(),
        record.authorization_record_sha256.as_str(),
        record.authorization_source_document_sha256.as_str(),
        record.artifact_qualification_identity_sha256.as_str(),
        record.attention_runtime_identity_sha256.as_str(),
        record.attention_qualification_sha256.as_str(),
        record.campaign_process.linux_boot_id_sha256.as_str(),
        record.campaign_process.executable_sha256.as_str(),
        record.campaign_process.launch_argv_sha256.as_str(),
        record.campaign_process.launch_environment_sha256.as_str(),
        record.identity_sha256.as_str(),
    ];
    if !valid_lower_hex(&record.campaign_source_sha, 40)
        || sha_values
            .into_iter()
            .any(|value| !valid_lower_hex(value, 64))
        || record.evidence_artifacts.iter().any(|evidence| {
            !valid_runtime_evidence_relative_path(&evidence.relative_path)
                || evidence.bytes == 0
                || !valid_lower_hex(&evidence.sha256, 64)
        })
        || record.evidence_artifacts.is_empty()
        || record
            .evidence_artifacts
            .windows(2)
            .any(|pair| pair[0].relative_path >= pair[1].relative_path)
        || record.artifact_total_bytes == 0
        || record.campaign_process.process_id == 0
        || record.campaign_process.process_start_time_ticks == 0
        || record.campaign_process.executable_device == 0
        || record.campaign_process.executable_inode == 0
        || record.campaign_process.executable_bytes == 0
        || record.campaign_process.executable_sha256 != record.measured_server_executable_sha256
        || record.campaign_process.cuda_driver_version == 0
        || record.campaign_process.cuda_toolkit_version == 0
        || !valid_stable_cuda_device_id(&record.device_id)
        || !(1..=99).contains(&record.compute_capability[0])
        || record.compute_capability[1] > 99
        || !valid_runtime_evidence_relative_path(&record.measured_server_executable_relative_path)
        || record
            .evidence_artifacts
            .iter()
            .filter(|evidence| {
                evidence.relative_path == record.measured_server_executable_relative_path
                    && evidence.sha256 == record.measured_server_executable_sha256
                    && evidence.bytes == record.campaign_process.executable_bytes
            })
            .count()
            != 1
    {
        bail!("private H3 runtime qualification contains incomplete evidence identities")
    }
    if record.schema != RUNTIME_QUALIFICATION_SCHEMA
        || record.decision != RUNTIME_QUALIFICATION_DECISION
        || record.canonical_model != contract::FL2VA_COMFY
        || record.task != "fl2va"
        || record.identity_sha256 != runtime_qualification_identity(record)
    {
        bail!("private H3 runtime qualification record shape or identity is invalid")
    }
    Ok(())
}

pub(crate) fn runtime_qualification_identity(
    record: &H3PrivateRuntimeQualificationRecord,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-runtime-qualification.v4\0");
    for value in [
        record.schema.as_str(),
        record.decision.as_str(),
        record.canonical_model.as_str(),
        record.task.as_str(),
        record.campaign_source_sha.as_str(),
        record.campaign_runtime_code_identity_sha256.as_str(),
        record.campaign_bootstrap_record_sha256.as_str(),
        record.campaign_bootstrap_identity_sha256.as_str(),
        record.measured_server_executable_relative_path.as_str(),
        record.measured_server_executable_sha256.as_str(),
        record.authorization_record_sha256.as_str(),
        record.authorization_source_document_sha256.as_str(),
        record.artifact_qualification_identity_sha256.as_str(),
    ] {
        update_string(&mut digest, value);
    }
    digest.update(record.artifact_total_bytes.to_le_bytes());
    update_string(&mut digest, &record.device_id);
    digest.update((record.device_ordinal as u64).to_le_bytes());
    digest.update(record.compute_capability[0].to_le_bytes());
    digest.update(record.compute_capability[1].to_le_bytes());
    update_string(&mut digest, &record.attention_runtime_identity_sha256);
    update_string(&mut digest, &record.attention_kernel_identity);
    update_string(&mut digest, &record.attention_qualification_sha256);
    digest.update(record.campaign_process.process_id.to_le_bytes());
    digest.update(
        record
            .campaign_process
            .process_start_time_ticks
            .to_le_bytes(),
    );
    update_string(&mut digest, &record.campaign_process.linux_boot_id_sha256);
    digest.update(record.campaign_process.executable_device.to_le_bytes());
    digest.update(record.campaign_process.executable_inode.to_le_bytes());
    digest.update(record.campaign_process.executable_bytes.to_le_bytes());
    update_string(&mut digest, &record.campaign_process.executable_sha256);
    update_string(&mut digest, &record.campaign_process.launch_argv_sha256);
    update_string(
        &mut digest,
        &record.campaign_process.launch_environment_sha256,
    );
    digest.update(record.campaign_process.cuda_driver_version.to_le_bytes());
    digest.update(record.campaign_process.cuda_toolkit_version.to_le_bytes());
    record.envelope.update_identity(&mut digest);
    record.bounds.update_identity(&mut digest);
    digest.update((record.evidence_artifacts.len() as u64).to_le_bytes());
    for evidence in &record.evidence_artifacts {
        update_string(&mut digest, &evidence.relative_path);
        digest.update(evidence.bytes.to_le_bytes());
        update_string(&mut digest, &evidence.sha256);
    }
    format!("{:x}", digest.finalize())
}

fn update_string(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn exact_h3_runtime_build_source_sha() -> Result<&'static str> {
    let source_sha = mold_core::build_info::GIT_SHA;
    if !valid_lower_hex(source_sha, 40) {
        bail!("private H3 runtime activation requires an exact embedded source SHA")
    }
    Ok(source_sha)
}

fn validate_h3_runtime_build_identity(
    source_sha: &str,
    runtime_code_identity_sha256: &str,
) -> Result<()> {
    if !valid_lower_hex(source_sha, 40) || !valid_lower_hex(runtime_code_identity_sha256, 64) {
        bail!("private H3 runtime activation requires exact embedded build identities")
    }
    Ok(())
}

fn valid_runtime_evidence_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !value.is_empty()
        && !value.contains('\\')
        && !value.chars().any(char::is_control)
        && value
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;

    use super::*;
    use crate::attention::{AttentionBackend, AttentionChunkPolicy};
    use crate::{
        H3AttentionActivation, H3AttentionBackend, H3AttentionKernel, H3FactoryAuthorityInput,
        H3FactoryComponentAuthority, H3FactoryComponentRole, H3FactoryConditionerPlacement,
        H3FactoryQuantizationAuthority,
    };
    #[cfg(feature = "mp4")]
    use crate::{H3FactoryEndpointInput, H3FactoryEndpointPreprocess, H3FactoryPreparedRowsInput};

    fn reviewed_envelope(max_steps: u32) -> H3PrivateRuntimeEnvelopeRecord {
        H3PrivateRuntimeEnvelopeRecord {
            width: contract::DEFAULT_WIDTH,
            height: contract::DEFAULT_HEIGHT,
            frames: contract::MIN_FRAMES,
            fps: contract::FIXED_FPS,
            batch_size: 1,
            max_steps,
            endpoint_count: 1,
            endpoint_anchor: "first".into(),
            max_qwen_output_text_rows: 1_058,
            max_qwen_vision_rows: 4_032,
            max_condition_visual_rows: 1_008,
            max_target_video_rows: 37_296,
            max_target_audio_rows: 414,
            max_total_packed_rows: 39_776,
        }
    }

    /// Without an authenticated adapter the reviewed envelope is exactly as
    /// strict as before: 21 steps and nothing else.
    #[test]
    fn the_envelope_step_pin_is_unchanged_without_a_turbo_adapter() {
        reviewed_envelope(contract::COMFY_DEFAULT_STEPS)
            .validate()
            .unwrap();
        for steps in [5u32, 9, 20, 22, 50, 0] {
            let error = reviewed_envelope(steps).validate().unwrap_err().to_string();
            assert!(
                error.contains(&format!("allows {} steps", contract::COMFY_DEFAULT_STEPS)),
                "{steps}: {error}"
            );
        }
    }

    /// A Turbo tier moves the step axis and ONLY the step axis, and only to the
    /// count its own distillation was reviewed for.
    #[test]
    fn a_turbo_tier_moves_only_the_step_axis_of_the_reviewed_envelope() {
        for reviewed_steps in [5u32, 9] {
            reviewed_envelope(reviewed_steps)
                .validate_with_reviewed_steps(Some(reviewed_steps))
                .unwrap();

            // Any other count, including the baseline 21, is refused for that
            // tier: an adapter distilled for N steps may not run at M.
            for wrong in [contract::COMFY_DEFAULT_STEPS, reviewed_steps + 1, 4] {
                if wrong == reviewed_steps {
                    continue;
                }
                let error = reviewed_envelope(wrong)
                    .validate_with_reviewed_steps(Some(reviewed_steps))
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains(&format!("allows {reviewed_steps} steps")),
                    "{reviewed_steps}/{wrong}: {error}"
                );
            }

            // And no other axis relaxes just because an adapter is present.
            let mut widened = reviewed_envelope(reviewed_steps);
            widened.width += 64;
            assert!(widened
                .validate_with_reviewed_steps(Some(reviewed_steps))
                .is_err());
            let mut longer = reviewed_envelope(reviewed_steps);
            longer.frames += 4;
            assert!(longer
                .validate_with_reviewed_steps(Some(reviewed_steps))
                .is_err());
            let mut batched = reviewed_envelope(reviewed_steps);
            batched.batch_size = 2;
            assert!(batched
                .validate_with_reviewed_steps(Some(reviewed_steps))
                .is_err());
            let mut anchored = reviewed_envelope(reviewed_steps);
            anchored.endpoint_anchor = "last".into();
            assert!(anchored
                .validate_with_reviewed_steps(Some(reviewed_steps))
                .is_err());
            let mut zeroed = reviewed_envelope(reviewed_steps);
            zeroed.max_total_packed_rows = 0;
            assert!(zeroed
                .validate_with_reviewed_steps(Some(reviewed_steps))
                .is_err());
        }
    }

    /// The reviewed Turbo step counts are exactly the ones the tier table
    /// declares, so the envelope and the sampler cannot drift apart.
    #[test]
    fn reviewed_turbo_step_counts_come_from_the_tier_table() {
        for contract_entry in super::super::turbo::REVIEWED_TURBO_TIERS {
            reviewed_envelope(contract_entry.grid_points)
                .validate_with_reviewed_steps(Some(contract_entry.grid_points))
                .unwrap();
            assert!(matches!(contract_entry.grid_points, 5 | 9));
        }
    }

    const DEVICE_0: &str = "cuda:00000000000000000000000000000000";
    const DEVICE_1: &str = "cuda:00000000000000000000000000000001";

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn source_sha(byte: char) -> String {
        std::iter::repeat_n(byte, 40).collect()
    }

    #[test]
    fn runtime_record_requires_scheduler_stable_cuda_identity() {
        assert!(valid_stable_cuda_device_id(DEVICE_0));
        assert!(!valid_stable_cuda_device_id("cuda:0"));
        assert!(!valid_stable_cuda_device_id(
            "cuda:0000000000000000000000000000000A"
        ));
        assert!(!valid_stable_cuda_device_id(
            "cuda:0000000000000000000000000000000"
        ));
        assert!(!valid_stable_cuda_device_id(
            "runtime:gpu:00000000000000000000000000000000"
        ));
    }

    #[test]
    fn runtime_record_requires_the_enforced_media_caps() {
        let mut candidate = record();
        validate_runtime_qualification_record_shape(&candidate).unwrap();
        candidate.bounds.thumbnail_host_bytes_bound -= 1;
        assert!(validate_runtime_qualification_record_shape(&candidate)
            .unwrap_err()
            .to_string()
            .contains("media bounds must match enforced caps"));
    }

    fn media() -> H3PrivateFl2VaMediaContract {
        H3PrivateFl2VaMediaContract {
            canonical_model: contract::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            mode: Mode::TextToAudioVideo,
            seed: 42,
            width: 768,
            height: 512,
            frames: 97,
            fps: contract::FIXED_FPS,
        }
    }

    fn artifact_report() -> H3PrivateArtifactQualificationReport {
        H3PrivateArtifactQualificationReport {
            schema: "mold.minimax-h3.private-artifact-qualification.v2",
            claim_marker: "private-test",
            decision: "qualified-private-artifacts",
            authorization_scope: "private-h3-uat",
            authorization_schema: "mold.minimax-h3.authorization.v1",
            authorization_record_sha256: sha('a'),
            authorization_source_document_sha256: sha('b'),
            authorization_review_reference: "reviewed-test".into(),
            canonical_model: contract::FL2VA_COMFY.into(),
            task: "fl2va",
            layout: "comfy-pruned-int8-convrot-nvfp4-awq",
            official_model_revision: "test",
            comfy_model_revision: "test",
            official_implementation_revision: "test",
            comfy_implementation_revision: "test",
            artifacts: Vec::new(),
            total_bytes: 42,
            qualification_identity_sha256: sha('c'),
            runtime_constructed: false,
            generated_media: false,
            public_activation: "disabled",
            remaining_release_requirements: Vec::new(),
        }
    }

    fn record() -> H3PrivateRuntimeQualificationRecord {
        let mut record = H3PrivateRuntimeQualificationRecord {
            schema: RUNTIME_QUALIFICATION_SCHEMA.into(),
            decision: RUNTIME_QUALIFICATION_DECISION.into(),
            canonical_model: contract::FL2VA_COMFY.into(),
            task: "fl2va".into(),
            campaign_source_sha: source_sha('d'),
            campaign_runtime_code_identity_sha256: sha('5'),
            campaign_bootstrap_record_sha256: sha('6'),
            campaign_bootstrap_identity_sha256: sha('7'),
            measured_server_executable_relative_path: "bin/mold-campaign".into(),
            measured_server_executable_sha256: sha('4'),
            authorization_record_sha256: sha('a'),
            authorization_source_document_sha256: sha('b'),
            artifact_qualification_identity_sha256: sha('c'),
            artifact_total_bytes: 42,
            device_id: DEVICE_0.into(),
            device_ordinal: 0,
            compute_capability: [8, 9],
            attention_runtime_identity_sha256: sha('1'),
            attention_kernel_identity: "qualified-kernel".into(),
            attention_qualification_sha256: sha('2'),
            campaign_process: H3PrivateRuntimeProcessObservation {
                process_id: 42,
                process_start_time_ticks: 99,
                linux_boot_id_sha256: sha('8'),
                executable_device: 7,
                executable_inode: 8,
                executable_bytes: 1_024,
                executable_sha256: sha('4'),
                launch_argv_sha256: sha('9'),
                launch_environment_sha256: sha('0'),
                cuda_driver_version: 12_080,
                cuda_toolkit_version: 12_080,
            },
            envelope: H3PrivateRuntimeEnvelopeRecord {
                width: contract::DEFAULT_WIDTH,
                height: contract::DEFAULT_HEIGHT,
                frames: contract::MIN_FRAMES,
                fps: contract::FIXED_FPS,
                batch_size: 1,
                max_steps: contract::COMFY_DEFAULT_STEPS,
                endpoint_count: 1,
                endpoint_anchor: "first".into(),
                max_qwen_output_text_rows: 128,
                max_qwen_vision_rows: 1_024,
                max_condition_visual_rows: 1_024,
                max_target_video_rows: 16_384,
                max_target_audio_rows: 1_024,
                max_total_packed_rows: 19_560,
            },
            bounds: H3PrivateRuntimeBoundRecord {
                fixed_runtime_host_bytes: 1,
                fixed_runtime_device_bytes: 2,
                qwen_activation_workspace_bytes: 3,
                vae_construction_device_workspace_bytes: 4,
                condition_vae_workspace_device_bytes: 5,
                attention_workspace_device_bytes: 6,
                ffn_workspace_device_bytes: 7,
                decoder_tile_workspace_device_bytes: 8,
                audio_decode_workspace_device_bytes: 9,
                encoded_video_host_bytes_bound:
                    super::super::pipeline::SMALL_ENCODED_VIDEO_HOST_BYTES_BOUND,
                thumbnail_host_bytes_bound:
                    super::super::pipeline::SMALL_THUMBNAIL_HOST_BYTES_BOUND,
                mux_output_host_bytes_bound:
                    super::super::pipeline::SMALL_MUX_OUTPUT_HOST_BYTES_BOUND,
                aac_mux_staging_host_bytes:
                    super::super::pipeline::SMALL_AAC_MUX_STAGING_HOST_BYTES,
            },
            evidence_artifacts: vec![
                H3PrivateRuntimeEvidenceArtifact {
                    relative_path: "bin/mold-campaign".into(),
                    bytes: 1_024,
                    sha256: sha('4'),
                },
                H3PrivateRuntimeEvidenceArtifact {
                    relative_path: "runtime-report.json".into(),
                    bytes: 99,
                    sha256: sha('3'),
                },
            ],
            identity_sha256: String::new(),
        };
        record.identity_sha256 = runtime_qualification_identity(&record);
        record
    }

    #[cfg(feature = "mp4")]
    fn prepared_request_for_compact_quality_envelope() -> H3FactoryPreparedRequestInput {
        H3FactoryPreparedRequestInput {
            identity_sha256: sha('0'),
            canonical_model: contract::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            mode: Mode::FirstFrameToAudioVideo,
            prompt_sha256: sha('1'),
            seed: 42,
            grid_points: contract::COMFY_DEFAULT_STEPS,
            denoise_forward_count: contract::COMFY_DEFAULT_STEPS - 1,
            guidance_f64_bits: 0.0_f64.to_bits(),
            strength_f64_bits: 1.0_f64.to_bits(),
            batch_size: 1,
            width: contract::DEFAULT_WIDTH,
            height: contract::DEFAULT_HEIGHT,
            frames: contract::MIN_FRAMES,
            fps: contract::FIXED_FPS,
            synchronized_audio: true,
            mp4_output: true,
            video_latent_frames: 37,
            audio_latents_per_channel: 207,
            audio_samples_per_channel: 165_600,
            conditioning_fingerprint: sha('2'),
            reference_fingerprint: sha('3'),
            endpoints: vec![H3FactoryEndpointInput {
                anchor: H3FactoryEndpointAnchor::First,
                encoded_bytes: 128,
                encoded_content_sha256: sha('4'),
                preprocess: H3FactoryEndpointPreprocess::PillowLanczosRgbU8CpuV1,
                normalized_shape: [1, 3, 1, contract::DEFAULT_HEIGHT, contract::DEFAULT_WIDTH],
                normalized_cpu_bytes: u64::from(contract::DEFAULT_WIDTH)
                    * u64::from(contract::DEFAULT_HEIGHT)
                    * 3,
                normalized_cpu_content_sha256: sha('5'),
            }],
            rows: H3FactoryPreparedRowsInput {
                qwen_output_text_rows: 128,
                qwen_vision_rows: 1_024,
                condition_visual_rows: 1_024,
                condition_audio_rows: 0,
                target_video_rows: 16_384,
                target_audio_rows: 1_024,
                total_packed_rows: 18_560,
            },
        }
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn compact_quality_envelope_rejects_every_unreviewed_request_axis() {
        let envelope = record().envelope;
        let reviewed = prepared_request_for_compact_quality_envelope();
        envelope.validate_prepared(&reviewed).unwrap();

        let mut cases = Vec::new();
        let mut request = reviewed.clone();
        request.width = 544;
        request.height = 960;
        cases.push(request);
        let mut request = reviewed.clone();
        request.frames += 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.fps += 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.grid_points += 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.grid_points -= 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.batch_size += 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.mode = Mode::TextToAudioVideo;
        cases.push(request);
        let mut request = reviewed.clone();
        request.synchronized_audio = false;
        cases.push(request);
        let mut request = reviewed.clone();
        request.mp4_output = false;
        cases.push(request);
        let mut request = reviewed.clone();
        request.endpoints[0].anchor = H3FactoryEndpointAnchor::Last;
        cases.push(request);
        let mut request = reviewed.clone();
        request.rows.qwen_output_text_rows += 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.rows.qwen_vision_rows += 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.rows.condition_visual_rows += 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.rows.condition_audio_rows = 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.rows.target_video_rows += 1;
        cases.push(request);
        let mut request = reviewed.clone();
        request.rows.target_audio_rows += 1;
        cases.push(request);
        let mut request = reviewed;
        request.rows.total_packed_rows += 1_001;
        cases.push(request);

        for request in cases {
            assert!(envelope.validate_prepared(&request).is_err());
        }
    }

    fn write_record(record: &H3PrivateRuntimeQualificationRecord) -> (tempfile::TempDir, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("runtime-qualification.json");
        let mut file = File::create(&path).unwrap();
        serde_json::to_writer(&mut file, record).unwrap();
        file.flush().unwrap();
        (root, path)
    }

    #[cfg(unix)]
    struct PresentationFixture {
        _root: tempfile::TempDir,
        models_root: PathBuf,
        authorization_record: PathBuf,
        runtime_record: PathBuf,
        source_sha256: String,
        runtime_record_sha256: String,
        runtime_code_identity_sha256: String,
        route: H3PrivatePresentationRoute<'static>,
    }

    #[cfg(unix)]
    impl PresentationFixture {
        fn new() -> Self {
            use std::os::unix::fs::PermissionsExt;

            fn private_directory(path: &Path) {
                fs::create_dir_all(path).unwrap();
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
            }

            fn private_file(path: &Path, bytes: &[u8]) {
                fs::write(path, bytes).unwrap();
                fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
            }

            let root = tempfile::tempdir().unwrap();
            let campaign = root.path().canonicalize().unwrap().join("campaign");
            let mold_home = campaign.join("mold-home");
            let models_root = mold_home.join("models");
            let compliance = campaign.join("compliance");
            for directory in [&campaign, &mold_home, &models_root, &compliance] {
                private_directory(directory);
            }
            let source_document = compliance.join("authorization-evidence.bin");
            private_file(&source_document, b"presentation-review-evidence");
            let source_sha256 =
                format!("{:x}", Sha256::digest(fs::read(&source_document).unwrap()));
            let authorization_record = compliance.join("authorization.v1.json");
            private_file(
                &authorization_record,
                &serde_json::to_vec(&serde_json::json!({
                    "schema_version": "mold.minimax-h3.authorization.v1",
                    "family": contract::FAMILY,
                    "decision": "approved",
                    "license_revision": contract::OFFICIAL_REVISION,
                    "license_sha256": contract::LICENSE_SHA256,
                    "approved_scopes": [
                        "checkpoint-execution",
                        "fixture-capture",
                        "generated-output-retention"
                    ],
                    "source_document_path": source_document,
                    "source_document_sha256": source_sha256.clone(),
                    "review_reference": "presentation-test-review"
                }))
                .unwrap(),
            );
            let authorization_record_sha256 = format!(
                "{:x}",
                Sha256::digest(fs::read(&authorization_record).unwrap())
            );
            let mut record = record();
            record.authorization_record_sha256 = authorization_record_sha256;
            record.authorization_source_document_sha256 = source_sha256.clone();
            record.identity_sha256 = runtime_qualification_identity(&record);
            let runtime_code_identity_sha256 = record.campaign_runtime_code_identity_sha256.clone();
            let runtime_record = compliance.join("runtime-qualification.json");
            private_file(&runtime_record, &serde_json::to_vec(&record).unwrap());
            let runtime_record_sha256 =
                sha256_open_file(&open_regular_file_no_follow(&runtime_record).unwrap()).unwrap();
            Self {
                _root: root,
                models_root,
                authorization_record,
                runtime_record,
                source_sha256,
                runtime_record_sha256,
                runtime_code_identity_sha256,
                route: H3PrivatePresentationRoute {
                    device_id: DEVICE_0,
                    device_ordinal: 0,
                    compute_capability: (8, 9),
                },
            }
        }

        fn authenticate(&self) -> Result<H3PrivatePresentationAuthority> {
            authenticate_h3_private_presentation_for_test(
                &self.models_root,
                &self.authorization_record,
                &self.runtime_record,
                &self.source_sha256,
                &[self.runtime_record_sha256.as_str()],
                &source_sha('d'),
                &self.runtime_code_identity_sha256,
                &[self.route],
                &sha('1'),
                "qualified-kernel",
            )
        }
    }

    #[cfg(unix)]
    #[test]
    fn presentation_authenticates_exact_records_and_live_route() {
        let fixture = PresentationFixture::new();
        let authority = fixture.authenticate().unwrap();
        assert_eq!(authority.canonical_model(), contract::FL2VA_COMFY);
        assert_eq!(authority.task(), Task::Fl2va);
        assert_eq!(authority.device_id(), DEVICE_0);
        assert_eq!(authority.device_ordinal(), 0);
        assert_eq!(authority.compute_capability(), (8, 9));
    }

    #[cfg(unix)]
    #[test]
    fn presentation_rejects_record_scope_and_route_substitution() {
        use std::os::unix::fs::PermissionsExt;

        let fixture = PresentationFixture::new();
        fs::set_permissions(&fixture.runtime_record, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(fixture.authenticate().is_err());
        fs::set_permissions(&fixture.runtime_record, fs::Permissions::from_mode(0o600)).unwrap();

        let wrong_route = H3PrivatePresentationRoute {
            device_id: DEVICE_1,
            device_ordinal: 1,
            compute_capability: (8, 9),
        };
        assert!(authenticate_h3_private_presentation_for_test(
            &fixture.models_root,
            &fixture.authorization_record,
            &fixture.runtime_record,
            &fixture.source_sha256,
            &[fixture.runtime_record_sha256.as_str()],
            &source_sha('d'),
            &fixture.runtime_code_identity_sha256,
            &[wrong_route],
            &sha('1'),
            "qualified-kernel",
        )
        .is_err());

        fs::write(&fixture.runtime_record, b"{}").unwrap();
        assert!(fixture.authenticate().is_err());
    }

    fn base_factory() -> FrozenH3FactoryAuthority {
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: contract::FL2VA_COMFY.into(),
            device_id: DEVICE_0.into(),
            device_ordinal: 0,
            compute_capability: Some((8, 9)),
            execution_fingerprint: sha('4'),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 10,
            qwen_host_resident_parameter_bytes: 10,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 10,
            qwen_maximum_tensor_staging_bytes: 8,
            qwen_retained_raw_header_bytes: 4,
            qwen_output_text_rows: 1,
            qwen_vision_rows: 0,
            condition_visual_rows: 0,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: "qualified-test-kernel".into(),
            attention_qualification_sha256: sha('5'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            attention_runtime: None,
            block_offload: true,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('6'),
                qwen_policy_sha256: sha('7'),
                pruned_adaln_table_sha256: sha('8'),
                turbo_adapter: None,
            },
            prepared_attempt: None,
            execution_budget_echo: None,
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
                    sha(char::from(b'a' + index as u8)),
                    sha(char::from(b'1' + index as u8)),
                )
                .unwrap()
            })
            .collect(),
        })
        .unwrap()
    }

    fn admission_evidence(
        request: &GenerateRequest,
        factory: &FrozenH3FactoryAuthority,
    ) -> H3PrivateFl2VaAdmissionEvidence {
        let submitted_request_identity_sha256 = private_h3_request_identity(request).unwrap();
        let mut resolved = request.clone();
        resolved.seed = Some(42);
        let attention_device = H3AttentionDevice::Cuda {
            compute_capability: Some((8, 9)),
        };
        let attention_model = H3AttentionModelContract::released_bf16();
        let runtime_backend = H3AttentionBackend::FlashAttentionV2;
        let kernel = H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011;
        let activation = H3AttentionActivation::ReleaseCandidateQualificationOnly;
        let runtime_identity_sha256 = H3AttentionRuntimeAuthority::expected_identity_for(
            runtime_backend,
            kernel,
            activation,
            attention_device,
            attention_model,
        )
        .unwrap();
        let mut evidence = H3PrivateFl2VaAdmissionEvidence {
            identity_sha256: String::new(),
            submitted_request_identity_sha256,
            resolved_request_identity_sha256: private_h3_request_identity(&resolved).unwrap(),
            canonical_model: contract::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            mode: Mode::TextToAudioVideo,
            device_id: DEVICE_0.into(),
            device_ordinal: 0,
            compute_capability: (8, 9),
            admitted_available_device_bytes: 10,
            admitted_host_headroom_bytes: 10,
            base_factory_authority: factory.clone(),
            execution_fingerprint: factory.execution_fingerprint().into(),
            component_set_identity_sha256: factory.component_set_identity_sha256().into(),
            prepared_request_identity_sha256: sha('d'),
            prepared_attempt_identity_sha256: sha('e'),
            target_budget_identity_sha256: sha('f'),
            artifact_qualification_identity_sha256: sha('a'),
            runtime_qualification_identity_sha256: sha('b'),
            predicted_device_peak_bytes: 1,
            predicted_host_increment_bytes: 1,
            seed: 42,
            attention: H3FactoryAttentionInput {
                generic_backend: AttentionBackend::Flash,
                generic_chunk: AttentionChunkPolicy::Off,
                runtime_backend,
                kernel,
                activation,
                device: attention_device,
                model_contract: attention_model,
                runtime_identity_sha256,
                qualification_kernel_identity: kernel.identity().into(),
                qualification_sha256: factory.attention_qualification_sha256().into(),
                full_noncausal: true,
                lossless: true,
            },
            runtime_bounds: H3PrivateFl2VaRuntimeBounds::from(&record().bounds.into_authority()),
        };
        evidence.identity_sha256 = private_h3_admission_evidence_identity(&evidence);
        evidence
    }

    fn owner_facts() -> H3PrivateFl2VaOwnerFacts {
        H3PrivateFl2VaOwnerFacts {
            work_identity_sha256: sha('1'),
            cancellation_scope_identity_sha256: sha('2'),
            admission_evidence_identity_sha256: sha('3'),
            artifact_qualification_identity_sha256: sha('4'),
            runtime_qualification_identity_sha256: sha('5'),
            device_id: DEVICE_0.into(),
            device_ordinal: 0,
            execution_fingerprint: sha('6'),
            prepared_attempt_identity_sha256: sha('7'),
            target_budget_identity_sha256: sha('8'),
            component_set_identity_sha256: sha('9'),
            requested_grid_points: contract::COMFY_DEFAULT_STEPS,
            transformer_evaluations: contract::COMFY_DEFAULT_STEPS - 1,
            predicted_device_peak_bytes: 7,
            predicted_host_increment_bytes: 8,
            media: media(),
        }
    }

    fn scheduler_ledger(owner: &H3PrivateFl2VaOwnerFacts) -> H3PrivateSchedulerLedgerIdentity {
        H3PrivateSchedulerLedgerIdentity::new(
            owner.work_identity_sha256.clone(),
            owner.cancellation_scope_identity_sha256.clone(),
            9,
            owner.admission_evidence_identity_sha256.clone(),
            owner.artifact_qualification_identity_sha256.clone(),
            owner.runtime_qualification_identity_sha256.clone(),
            owner.execution_fingerprint.clone(),
            owner.prepared_attempt_identity_sha256.clone(),
            owner.target_budget_identity_sha256.clone(),
            owner.component_set_identity_sha256.clone(),
        )
        .unwrap()
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn terminal_sampler_provenance_is_exactly_bound_to_the_owner() {
        let owner = owner_facts();
        let grid_points = usize::try_from(contract::COMFY_DEFAULT_STEPS).unwrap();
        let evaluations = grid_points - 1;
        validate_private_sampler_provenance(
            "comfy-res-multistep",
            grid_points,
            evaluations,
            &owner,
        )
        .unwrap();
        for (sampler, grid_points, evaluations) in [
            ("official-euler", grid_points, evaluations),
            ("comfy-res-multistep", grid_points + 1, evaluations),
            ("comfy-res-multistep", grid_points, evaluations + 1),
        ] {
            assert!(
                validate_private_sampler_provenance(sampler, grid_points, evaluations, &owner)
                    .unwrap_err()
                    .to_string()
                    .contains("sampler provenance")
            );
        }
    }

    #[test]
    fn reviewed_allowlist_is_scoped_to_fl2va() {
        assert!(reviewed_h3_private_runtime_available());
        assert!(reviewed_h3_private_runtime_available_for_task(Task::Fl2va));
        assert!(!reviewed_h3_private_runtime_available_for_task(
            Task::Ref2va
        ));
    }

    #[test]
    fn empty_injected_allowlist_rejects_before_path_access() {
        let missing = Path::new("/private-h3-path-must-not-be-opened");
        let error = open_reviewed_h3_private_runtime_qualification_for_source(
            missing,
            &[],
            &source_sha('e'),
            &sha('6'),
        )
        .err()
        .expect("empty authority must reject");
        assert!(error.to_string().contains("reviewed evidence allowlist"));
    }

    #[test]
    fn production_reader_rejects_every_unreviewed_record() {
        let (_root, path) = write_record(&record());
        let error = authenticate_h3_private_runtime_qualification(
            &path,
            &artifact_report(),
            DEVICE_0,
            0,
            (8, 9),
            &sha('1'),
            "qualified-kernel",
            &sha('2'),
        )
        .unwrap_err();
        assert!(error.to_string().contains("reviewed evidence allowlist"));
    }

    #[test]
    fn reviewed_record_binds_all_thirteen_bounds_and_identity_axes() {
        let record = record();
        let (_root, path) = write_record(&record);
        let file = open_regular_file_no_follow(&path).unwrap();
        let digest = sha256_open_file(&file).unwrap();
        let reviewed = [digest.as_str()];
        let authority = authenticate_h3_private_runtime_qualification_for_source(
            &path,
            &artifact_report(),
            DEVICE_0,
            0,
            (8, 9),
            &sha('1'),
            "qualified-kernel",
            &sha('2'),
            &reviewed,
            &record.campaign_source_sha,
            &record.campaign_runtime_code_identity_sha256,
        )
        .unwrap();
        authority.revalidate().unwrap();
        assert_eq!(
            authority.bounds().aac_mux_staging_host_bytes,
            super::super::pipeline::SMALL_AAC_MUX_STAGING_HOST_BYTES
        );
    }

    #[test]
    fn reviewed_record_requires_exact_runtime_code_identity() {
        let record = record();
        let (_root, path) = write_record(&record);
        let digest = sha256_open_file(&open_regular_file_no_follow(&path).unwrap()).unwrap();
        let reviewed = [digest.as_str()];

        open_reviewed_h3_private_runtime_qualification_for_source(
            &path,
            &reviewed,
            &source_sha('e'),
            &record.campaign_runtime_code_identity_sha256,
        )
        .expect("an allowlist-only rebuild retains the normalized runtime identity");

        let error = open_reviewed_h3_private_runtime_qualification_for_source(
            &path,
            &reviewed,
            &record.campaign_source_sha,
            &sha('6'),
        )
        .err()
        .expect("changed runtime code must invalidate the campaign");
        assert!(error.to_string().contains("different runtime code"));
    }

    #[test]
    fn reviewed_record_rejects_crossed_device_or_artifact_authority() {
        let record = record();
        let (_root, path) = write_record(&record);
        let file = open_regular_file_no_follow(&path).unwrap();
        let digest = sha256_open_file(&file).unwrap();
        let reviewed = [digest.as_str()];
        let error = authenticate_h3_private_runtime_qualification_for_source(
            &path,
            &artifact_report(),
            DEVICE_1,
            1,
            (8, 9),
            &sha('1'),
            "qualified-kernel",
            &sha('2'),
            &reviewed,
            &record.campaign_source_sha,
            &record.campaign_runtime_code_identity_sha256,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("differs from artifact, device, or kernel authority"));
    }

    #[test]
    fn reviewed_record_rejects_each_crossed_route_before_artifact_qualification() {
        let record = record();
        let (_root, path) = write_record(&record);
        let file = open_regular_file_no_follow(&path).unwrap();
        let digest = sha256_open_file(&file).unwrap();
        let reviewed_records = [digest.as_str()];
        for (device_id, device_ordinal, compute_capability) in [
            (DEVICE_1, 0, (8, 9)),
            (DEVICE_0, 1, (8, 9)),
            (DEVICE_0, 0, (9, 0)),
        ] {
            let reviewed = open_reviewed_h3_private_runtime_qualification_for_source(
                &path,
                &reviewed_records,
                &record.campaign_source_sha,
                &record.campaign_runtime_code_identity_sha256,
            )
            .unwrap();
            let error = reviewed
                .validate_route(device_id, device_ordinal, compute_capability)
                .unwrap_err();
            assert!(error
                .to_string()
                .contains("cannot authorize this CUDA route"));
        }
    }

    #[test]
    fn reviewed_record_rejects_unbound_source_executable_and_gpu_claims() {
        for unsupported in [
            "source_tree_sha256",
            "executable_sha256",
            "gpu_identity_sha256",
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(format!("{unsupported}.json"));
            let mut value = serde_json::to_value(record()).unwrap();
            value
                .as_object_mut()
                .unwrap()
                .insert(unsupported.into(), serde_json::Value::String(sha('f')));
            let bytes = serde_json::to_vec(&value).unwrap();
            let mut file = File::create(&path).unwrap();
            file.write_all(&bytes).unwrap();
            file.flush().unwrap();
            let digest = sha256_open_file(&open_regular_file_no_follow(&path).unwrap()).unwrap();
            let canonical = record();
            let reviewed = [digest.as_str()];
            let error = open_reviewed_h3_private_runtime_qualification_for_source(
                &path,
                &reviewed,
                &canonical.campaign_source_sha,
                &canonical.campaign_runtime_code_identity_sha256,
            )
            .err()
            .expect("unsupported runtime identity claim must be rejected");
            assert!(
                error
                    .to_string()
                    .contains("invalid private H3 runtime qualification record"),
                "{unsupported}: {error:#}"
            );
        }
    }

    #[test]
    fn public_attempt_facts_are_payload_free_and_cloneable() {
        let owner = owner_facts();
        let ledger = scheduler_ledger(&owner);
        let binding = H3PrivateAttemptConsumptionBinding::new(&owner, &ledger).unwrap();
        let facts = owner.attempt_facts(&binding);
        assert_eq!(facts.clone(), facts);
        assert_eq!(facts.memory_ledger_sequence(), 9);
        assert_eq!(facts.consumption_identity_sha256(), binding.identity_sha256);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn mismatched_owner_opened_vae_validation_is_rejected() {
        let admitted_vae_binding =
            private_h3_vae_artifact_binding_identity(&sha('a'), &sha('b')).unwrap();
        let same_artifact_reopen =
            private_h3_vae_artifact_binding_identity(&sha('a'), &sha('b')).unwrap();
        let crossed_artifact_reopen =
            private_h3_vae_artifact_binding_identity(&sha('c'), &sha('b')).unwrap();
        assert_eq!(admitted_vae_binding, same_artifact_reopen);
        assert_ne!(admitted_vae_binding, crossed_artifact_reopen);

        let factory = base_factory();
        let mut components = [
            H3FactoryComponentRole::Conditioner,
            H3FactoryComponentRole::Transformer,
            H3FactoryComponentRole::VisualVae,
            H3FactoryComponentRole::AudioVae,
        ]
        .into_iter()
        .map(|role| {
            let (content_sha256, validation_sha256) = factory.private_component_authority(role);
            H3PrivateComponentDigest {
                role,
                content_sha256: content_sha256.into(),
                validation_sha256: validation_sha256.into(),
            }
        })
        .collect::<Vec<_>>();
        validate_owner_component_digests(&factory, &components).unwrap();

        let visual = components
            .iter_mut()
            .find(|component| component.role == H3FactoryComponentRole::VisualVae)
            .unwrap();
        visual.validation_sha256 = crossed_artifact_reopen;
        let error = validate_owner_component_digests(&factory, &components).unwrap_err();
        assert!(error
            .to_string()
            .contains("owner-opened VisualVae authority differs from admission"));
    }

    #[test]
    fn consumption_binding_rejects_each_crossed_provenance_authority() {
        let owner = owner_facts();
        let ledger = scheduler_ledger(&owner);
        let binding = H3PrivateAttemptConsumptionBinding::new(&owner, &ledger).unwrap();
        for field in ["admission", "artifact", "runtime"] {
            let mut crossed = owner.clone();
            match field {
                "admission" => crossed.admission_evidence_identity_sha256 = sha('a'),
                "artifact" => crossed.artifact_qualification_identity_sha256 = sha('a'),
                "runtime" => crossed.runtime_qualification_identity_sha256 = sha('a'),
                _ => unreachable!(),
            }
            let error = binding.revalidate(&crossed, &ledger).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("prepared-attempt consumption binding changed"),
                "{field}: {error:#}"
            );
        }
    }

    #[test]
    fn scheduler_ledger_identity_carries_no_caller_memory_totals() {
        let owner = owner_facts();
        let identity = H3PrivateSchedulerLedgerIdentity::new(
            owner.work_identity_sha256.clone(),
            owner.cancellation_scope_identity_sha256.clone(),
            9,
            owner.admission_evidence_identity_sha256.clone(),
            owner.artifact_qualification_identity_sha256.clone(),
            owner.runtime_qualification_identity_sha256.clone(),
            owner.execution_fingerprint.clone(),
            owner.prepared_attempt_identity_sha256.clone(),
            owner.target_budget_identity_sha256.clone(),
            owner.component_set_identity_sha256.clone(),
        )
        .unwrap();
        identity.revalidate().unwrap();
        assert_eq!(identity.memory_ledger_sequence(), 9);
        assert_eq!(identity.identity_sha256().len(), 64);
    }

    #[test]
    fn allocation_commit_is_exactly_once_and_never_runs_on_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let count = Arc::new(AtomicUsize::new(0));
        let captured = Arc::clone(&count);
        let mut commit = H3PrivateAllocationCommit::new(move || {
            captured.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        commit.commit_once().unwrap();
        commit.commit_once().unwrap();
        assert!(commit.is_committed());
        assert_eq!(count.load(Ordering::SeqCst), 1);

        let dropped_count = Arc::new(AtomicUsize::new(0));
        let captured = Arc::clone(&dropped_count);
        drop(H3PrivateAllocationCommit::new(move || {
            captured.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));
        assert_eq!(dropped_count.load(Ordering::SeqCst), 0);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn allocation_commit_precedes_concrete_resource_construction() {
        use std::sync::{Arc, Mutex};

        let order = Arc::new(Mutex::new(Vec::new()));
        let callback_order = Arc::clone(&order);
        let mut commit = H3PrivateAllocationCommit::new(move || {
            callback_order.lock().unwrap().push("commit");
            Ok(())
        });
        let constructed = commit_private_h3_allocation_then(&mut commit, || {
            order.lock().unwrap().push("construct");
            Ok(7_u8)
        })
        .unwrap();
        assert_eq!(constructed, 7);
        assert_eq!(order.lock().unwrap().as_slice(), ["commit", "construct"]);
    }

    #[test]
    fn reviewed_preparation_source_is_cuda_construction_free() {
        let source = include_str!("private_server.rs");
        let admission_start = source
            .find("fn prepare_reviewed_h3_private_fl2va_admission")
            .unwrap();
        let admission_end = source[admission_start..]
            .find("struct H3PrivateComponentDigest")
            .map(|offset| admission_start + offset)
            .unwrap();
        let admission = &source[admission_start..admission_end];
        assert!(!admission.contains("Device::new_cuda"));
        let runtime_open = admission
            .find("open_reviewed_h3_private_runtime_qualification")
            .unwrap();
        let route_check = admission.find(".validate_route(").unwrap();
        let artifact_qualification = admission
            .find("qualify_private_artifacts_with_control")
            .unwrap();
        assert!(runtime_open < route_check && route_check < artifact_qualification);
        let prepare_start = source
            .find("fn prepare_reviewed_h3_private_fl2va_attempt")
            .unwrap();
        let prepare_end = source[prepare_start..]
            .find("fn validate_base_factory")
            .map(|offset| prepare_start + offset)
            .unwrap();
        let prepare = &source[prepare_start..prepare_end];
        assert!(!prepare.contains("Device::new_cuda"));
        let runtime_open = prepare
            .find("open_reviewed_h3_private_runtime_qualification")
            .unwrap();
        let route_check = prepare.find(".validate_route(").unwrap();
        let artifact_qualification = prepare
            .find("qualify_private_artifacts_with_control")
            .unwrap();
        assert!(runtime_open < route_check && route_check < artifact_qualification);

        let opened_evidence = include_str!("private_opened_evidence.rs");
        let preflight_start = opened_evidence
            .find("fn prepare_private_fl2va_request_input")
            .unwrap();
        let preflight_end = opened_evidence[preflight_start..]
            .find("impl H3PrivatePreparedFl2VaFactoryInputs")
            .map(|offset| preflight_start + offset)
            .unwrap();
        assert!(!opened_evidence[preflight_start..preflight_end].contains("Device::new_cuda"));
        assert!(opened_evidence.contains(
            "vae_memory_evidence_identity_sha256: vae\n            .artifact_validation_identity_sha256()"
        ));
        let pipeline = include_str!("pipeline.rs");
        assert!(pipeline.contains("`Device::Cpu`; execution transfers happen only after CUDA"));

        let components_start = source.find("fn private_h3_component_digests").unwrap();
        let components_end = source[components_start..]
            .find("fn private_h3_component_role")
            .map(|offset| components_start + offset)
            .unwrap();
        let components = &source[components_start..components_end];
        assert!(components.contains("vae.artifact_validation_identity_sha256()"));
        assert!(!components.contains("vae.identity_sha256()"));

        let runner_start = source
            .find("impl H3PrivateFl2VaPreparedRunner for H3PrivateConcretePreparedRunner")
            .unwrap();
        let runner = &source[runner_start..];
        let boundary = runner
            .find("with_private_h3_cuda_execution_attempt(||")
            .unwrap();
        let commit = runner.find("commit_private_h3_allocation_then").unwrap();
        let device = runner.find("Device::new_cuda").unwrap();
        let execute = runner.find("run_private_comfy_fl2va_attempt").unwrap();
        let completion_sync = runner[execute..]
            .find(".synchronize()")
            .map(|offset| execute + offset)
            .unwrap();
        assert!(boundary < commit && commit < device && device < execute);
        assert!(execute < completion_sync);

        #[cfg(feature = "cuda")]
        {
            let attempt_start = source
                .find("fn with_private_h3_cuda_execution_attempt<T>(operation")
                .unwrap();
            let attempt_end = source[attempt_start..]
                .find("#[cfg(all(feature = \"mp4\", not(feature = \"cuda\")))]")
                .map(|offset| attempt_start + offset)
                .unwrap();
            let attempt = &source[attempt_start..attempt_end];
            let begin = attempt
                .find("CudaExecutionAttempt::begin_unbound()")
                .unwrap();
            let catch = attempt
                .find("catch_unwind(std::panic::AssertUnwindSafe(operation))")
                .unwrap();
            let mark = attempt.find("attempt.mark_panicked();").unwrap();
            let panic_finish = attempt[mark..]
                .find("attempt.finish()")
                .map(|offset| mark + offset)
                .unwrap();
            assert!(begin < catch && catch < mark && mark < panic_finish);
            assert_eq!(attempt.matches("attempt.finish()").count(), 2);
            assert!(attempt.contains("status.resources_retained()"));
        }
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn preparation_vae_observer_uses_the_same_attempt_cancellation() {
        let cancellation = InferenceCancellationToken::default();
        let mut progress = ProgressReporter::default();
        progress.set_cancellation_token(cancellation.clone());
        let mut observer = H3PrivatePreparationVaeObserver::new(&progress);
        assert!(observer.checkpoint(H3ComfyVaeLoadEvent {
            role: super::super::vae_runtime::H3ComfyVaeArtifactRole::VisualConfig,
            phase: super::super::vae_runtime::H3ComfyVaeLoadPhase::Open,
            completed: 0,
            total: 1,
        }));
        cancellation.cancel();
        assert!(!observer.checkpoint(H3ComfyVaeLoadEvent {
            role: super::super::vae_runtime::H3ComfyVaeArtifactRole::VisualConfig,
            phase: super::super::vae_runtime::H3ComfyVaeLoadPhase::Authenticate,
            completed: 1,
            total: 1,
        }));
        let error = observer.finish(Ok(())).unwrap_err();
        assert!(crate::progress::is_inference_cancelled(&error));
    }

    struct FailingRunner {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        binding: H3PrivateAttemptConsumptionBinding,
        check_tokens: bool,
    }

    impl H3PrivateFl2VaPreparedRunner for FailingRunner {
        fn consumption_binding(&self) -> &H3PrivateAttemptConsumptionBinding {
            &self.binding
        }

        fn run(
            self: Box<Self>,
            progress: &ProgressReporter,
            cancellation: InferenceCancellationToken,
            _allocation_commit: H3PrivateAllocationCommit,
        ) -> Result<H3PrivateFl2VaRunOutput> {
            if self.check_tokens {
                progress.checkpoint()?;
                cancellation.checkpoint()?;
            }
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            bail!("synthetic private H3 runner failure")
        }
    }

    fn failing_attempt(
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        check_tokens: bool,
    ) -> H3PrivateFl2VaPreparedAttempt {
        let owner = owner_facts();
        let ledger = scheduler_ledger(&owner);
        let binding = H3PrivateAttemptConsumptionBinding::new(&owner, &ledger).unwrap();
        let facts = owner.attempt_facts(&binding);
        H3PrivateFl2VaPreparedAttempt::from_runner(
            facts,
            FailingRunner {
                calls,
                binding,
                check_tokens,
            },
        )
        .unwrap()
    }

    fn matching_run_context(cancellation: InferenceCancellationToken) -> H3PrivateFl2VaRunContext {
        H3PrivateFl2VaRunContext::new(sha('1'), sha('2'), 9, cancellation).unwrap()
    }

    #[test]
    fn prepared_attempt_is_consumed_even_when_the_first_run_fails() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let calls = Arc::new(AtomicUsize::new(0));
        let mut attempt = failing_attempt(Arc::clone(&calls), false);
        let mut progress = ProgressReporter::default();
        let first = attempt
            .run_once(
                matching_run_context(InferenceCancellationToken::default()),
                &mut progress,
                H3PrivateAllocationCommit::new(|| Ok(())),
            )
            .unwrap_err();
        assert!(first
            .to_string()
            .contains("synthetic private H3 runner failure"));
        let second = attempt
            .run_once(
                matching_run_context(InferenceCancellationToken::default()),
                &mut progress,
                H3PrivateAllocationCommit::new(|| Ok(())),
            )
            .unwrap_err();
        assert!(second.to_string().contains("already consumed"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn work_cancellation_and_ledger_mismatch_consume_without_running() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        for context in [
            H3PrivateFl2VaRunContext::new(
                sha('f'),
                sha('2'),
                9,
                InferenceCancellationToken::default(),
            )
            .unwrap(),
            H3PrivateFl2VaRunContext::new(
                sha('1'),
                sha('f'),
                9,
                InferenceCancellationToken::default(),
            )
            .unwrap(),
            H3PrivateFl2VaRunContext::new(
                sha('1'),
                sha('2'),
                10,
                InferenceCancellationToken::default(),
            )
            .unwrap(),
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let mut attempt = failing_attempt(Arc::clone(&calls), false);
            let mut progress = ProgressReporter::default();
            let error = attempt
                .run_once(
                    context,
                    &mut progress,
                    H3PrivateAllocationCommit::new(|| Ok(())),
                )
                .unwrap_err();
            assert!(error.to_string().contains("run context differs"));
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            let replay = attempt
                .run_once(
                    matching_run_context(InferenceCancellationToken::default()),
                    &mut progress,
                    H3PrivateAllocationCommit::new(|| Ok(())),
                )
                .unwrap_err();
            assert!(replay.to_string().contains("already consumed"));
        }
    }

    #[test]
    fn run_once_installs_one_token_for_runner_and_progress_checkpoints() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let calls = Arc::new(AtomicUsize::new(0));
        let mut attempt = failing_attempt(Arc::clone(&calls), true);
        let prior = InferenceCancellationToken::default();
        prior.cancel();
        let mut progress = ProgressReporter::default();
        progress.set_cancellation_token(prior);
        let error = attempt
            .run_once(
                matching_run_context(InferenceCancellationToken::default()),
                &mut progress,
                H3PrivateAllocationCommit::new(|| Ok(())),
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("synthetic private H3 runner failure"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(progress.checkpoint().is_err());
    }

    #[test]
    fn runtime_qualification_authority_is_send_but_not_clone() {
        fn assert_send<T: Send>() {}
        assert_send::<H3PrivateRuntimeQualificationAuthority>();

        trait AmbiguousIfClone<A> {
            fn assert_not_implemented() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        struct Marker;
        impl<T: Clone> AmbiguousIfClone<Marker> for T {}
        <H3PrivateRuntimeQualificationAuthority as AmbiguousIfClone<_>>::assert_not_implemented();
    }

    #[cfg(feature = "h3")]
    #[test]
    fn public_presentation_uses_compiled_sm89_policy_without_a_runtime_record() {
        let route = H3PrivatePresentationRoute {
            device_id: "cuda:00000000000000000000000000000000",
            device_ordinal: 0,
            compute_capability: Some((8, 9)),
        };
        let authority = authenticate_h3_public_presentation(&[route]).unwrap();
        assert_eq!(authority.canonical_model(), contract::FL2VA_COMFY);
        assert_eq!(authority.task(), Task::Fl2va);
        assert_eq!(authority.compute_capability(), (8, 9));

        let unsupported = H3PrivatePresentationRoute {
            compute_capability: (9, 0),
            ..route
        };
        assert!(authenticate_h3_public_presentation(&[unsupported]).is_err());
    }

    /// The pre-SHA capacity precheck must never refuse work the exact
    /// per-attempt check would admit. That holds iff each floor is a lower
    /// bound of the exact phase sum it stands in for, so this sweeps bound
    /// records and both conditioner placements and asserts exactly that,
    /// rebuilding the phase sums from the ledger's own arithmetic.
    #[cfg(feature = "mp4")]
    #[test]
    fn admission_precheck_floors_never_exceed_the_exact_phase_sums() {
        fn record(attention: u64, ffn: u64, device: u64, host: u64) -> H3PrivateRuntimeBoundRecord {
            H3PrivateRuntimeBoundRecord {
                fixed_runtime_host_bytes: host,
                fixed_runtime_device_bytes: device,
                qwen_activation_workspace_bytes: 3_959_422_976,
                vae_construction_device_workspace_bytes: 67_108_864,
                condition_vae_workspace_device_bytes: 469_762_048,
                attention_workspace_device_bytes: attention,
                ffn_workspace_device_bytes: ffn,
                decoder_tile_workspace_device_bytes: 1_543_503_872,
                audio_decode_workspace_device_bytes: 268_435_456,
                encoded_video_host_bytes_bound:
                    super::super::pipeline::SMALL_ENCODED_VIDEO_HOST_BYTES_BOUND,
                thumbnail_host_bytes_bound:
                    super::super::pipeline::SMALL_THUMBNAIL_HOST_BYTES_BOUND,
                mux_output_host_bytes_bound:
                    super::super::pipeline::SMALL_MUX_OUTPUT_HOST_BYTES_BOUND,
                aac_mux_staging_host_bytes:
                    super::super::pipeline::SMALL_AAC_MUX_STAGING_HOST_BYTES,
            }
        }

        let sweep = [
            record(1, 1, 1, 1),
            record(7_113_539_584, 8_791_261_184, 603_979_776, 805_306_368),
            record(15_300_820_992, 10_133_438_464, 603_979_776, 671_088_640),
            record(u32::MAX.into(), 1, u32::MAX.into(), u32::MAX.into()),
        ];
        // Request-derived terms the exact sums add on top. Every one is
        // non-negative, so the floor has to hold for the empty case too.
        let request_extras = [0_u64, 1, 14_708_736, 1_710_342_144];

        for bounds in &sweep {
            let device_floor = private_h3_admission_device_floor_bytes(bounds).unwrap();
            let host_floor = private_h3_admission_host_floor_bytes(bounds).unwrap();

            for extra in request_extras {
                // The exact denoise phase, rebuilt from the ledger's own
                // helper, so switching the floor back to attention + FFN or
                // dropping the fixed-runtime term breaks this.
                let denoise_phase = bounds.fixed_runtime_device_bytes
                    + crate::h3_factory::denoise_transient_workspace_device_bytes(
                        bounds.attention_workspace_device_bytes,
                        bounds.ffn_workspace_device_bytes,
                    )
                    + extra;
                assert!(
                    device_floor <= denoise_phase,
                    "device floor {device_floor} exceeds the exact denoise phase {denoise_phase}"
                );

                // The exact Qwen host phase under EITHER placement. A floor
                // built from the CPU residency would fail the accelerated case.
                for route in [
                    H3PrivateQwenLoaderMemoryRoute::Cpu,
                    H3PrivateQwenLoaderMemoryRoute::Cuda,
                ] {
                    let qwen = released_h3_private_qwen_loader_memory_authority(route).unwrap();
                    let qwen_phase = bounds.fixed_runtime_host_bytes
                        + qwen.host_resident_parameter_bytes
                        + extra;
                    assert!(
                        host_floor <= qwen_phase,
                        "host floor {host_floor} exceeds the exact Qwen host phase {qwen_phase}"
                    );
                }
            }

            // The gate itself: it refuses exactly below each floor and admits
            // at it, so a refusal always implies the exact sums cannot fit.
            assert!(precheck_private_h3_admission_capacity(
                bounds,
                device_floor.saturating_sub(1),
                u64::MAX
            )
            .is_err());
            assert!(precheck_private_h3_admission_capacity(
                bounds,
                u64::MAX,
                host_floor.saturating_sub(1)
            )
            .is_err());
            assert!(
                precheck_private_h3_admission_capacity(bounds, device_floor, host_floor).is_ok()
            );
        }
    }

    #[cfg(feature = "h3")]
    #[test]
    fn public_runtime_authority_pins_every_bound_envelope_and_identity_axis() {
        let artifact = artifact_report();
        let authority = public_runtime_qualification(
            &artifact,
            DEVICE_0,
            0,
            (8, 9),
            &sha('d'),
            "flash-attention-v2-sm89",
            &sha('e'),
            None,
        )
        .unwrap();
        authority.revalidate().unwrap();
        assert_eq!(authority.device_id(), DEVICE_0);
        assert_eq!(authority.device_ordinal(), 0);
        assert_eq!(authority.compute_capability(), (8, 9));
        assert_eq!(authority.artifact_qualification_identity_sha256(), sha('c'));
        assert_eq!(
            H3PrivateFl2VaRuntimeBounds::from(authority.bounds()),
            H3PrivateFl2VaRuntimeBounds {
                fixed_runtime_host_bytes: 805_306_368,
                fixed_runtime_device_bytes: 603_979_776,
                qwen_activation_workspace_bytes: 3_959_422_976,
                vae_construction_device_workspace_bytes: 67_108_864,
                condition_vae_workspace_device_bytes: 469_762_048,
                attention_workspace_device_bytes: 7_113_539_584,
                ffn_workspace_device_bytes: 8_791_261_184,
                decoder_tile_workspace_device_bytes: 1_543_503_872,
                audio_decode_workspace_device_bytes: 268_435_456,
                encoded_video_host_bytes_bound:
                    super::super::pipeline::SMALL_ENCODED_VIDEO_HOST_BYTES_BOUND,
                thumbnail_host_bytes_bound:
                    super::super::pipeline::SMALL_THUMBNAIL_HOST_BYTES_BOUND,
                mux_output_host_bytes_bound:
                    super::super::pipeline::SMALL_MUX_OUTPUT_HOST_BYTES_BOUND,
                aac_mux_staging_host_bytes:
                    super::super::pipeline::SMALL_AAC_MUX_STAGING_HOST_BYTES,
            }
        );
        assert_eq!(
            authority.record.envelope,
            H3PrivateRuntimeEnvelopeRecord {
                width: 1_344,
                height: 768,
                frames: 124,
                fps: 24,
                batch_size: 1,
                max_steps: 21,
                endpoint_count: 1,
                endpoint_anchor: "first".into(),
                max_qwen_output_text_rows: 1_058,
                max_qwen_vision_rows: 4_032,
                max_condition_visual_rows: 1_008,
                max_target_video_rows: 37_296,
                max_target_audio_rows: 414,
                max_total_packed_rows: 39_776,
            }
        );

        for (mut crossed, cc, attention, kernel, qualification) in [
            (
                artifact.clone(),
                (9, 0),
                sha('d'),
                "flash-attention-v2-sm89",
                sha('e'),
            ),
            (
                artifact.clone(),
                (8, 9),
                "bad".into(),
                "flash-attention-v2-sm89",
                sha('e'),
            ),
            (artifact.clone(), (8, 9), sha('d'), "", sha('e')),
            (
                artifact.clone(),
                (8, 9),
                sha('d'),
                "flash-attention-v2-sm89",
                "bad".into(),
            ),
        ] {
            assert!(public_runtime_qualification(
                &crossed,
                DEVICE_0,
                0,
                cc,
                &attention,
                kernel,
                &qualification,
                None,
            )
            .is_err());
            crossed.task = "ref2va";
            assert!(public_runtime_qualification(
                &crossed,
                DEVICE_0,
                0,
                (8, 9),
                &sha('d'),
                "flash-attention-v2-sm89",
                &sha('e'),
                None,
            )
            .is_err());
        }
        let mut crossed_model = artifact;
        crossed_model.canonical_model = contract::REF2VA_COMFY.into();
        assert!(public_runtime_qualification(
            &crossed_model,
            DEVICE_0,
            0,
            (8, 9),
            &sha('d'),
            "flash-attention-v2-sm89",
            &sha('e'),
            None,
        )
        .is_err());
    }

    #[cfg(feature = "h3")]
    #[test]
    fn public_progress_labels_describe_artifact_verification_without_private_claims() {
        for label in [
            H3_ARTIFACT_VERIFICATION_PROGRESS,
            H3_VAE_ARTIFACT_VERIFICATION_PROGRESS,
        ] {
            assert!(label.starts_with("Verifying MiniMax H3"));
            assert!(!label.to_ascii_lowercase().contains("private"));
            assert!(!label.to_ascii_lowercase().contains("authenticat"));
        }
    }
}
