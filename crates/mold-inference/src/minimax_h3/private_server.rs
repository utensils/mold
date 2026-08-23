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
use super::metal_memory_guard::H3MetalMemoryGuard;
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
    prepare_private_ref2va_admission_request, qwen_activation_workspace_demand_bytes,
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
// Feature-independent: the envelope validators below name this in their
// signatures in every build, matching how the type itself is defined.
use crate::engine::GenerationReferenceBinding;
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
    H3FactoryPreparedRequestInput, H3FactoryPreparedRowsInput, H3FactoryQuantizationAuthority,
    H3FactoryTargetBudgetInput,
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
        // Ref2VA has no reviewed bounds and no qualified public route. It is
        // reachable only from the developer-only campaign build, whose whole
        // purpose is to measure the bounds a future reviewed record would
        // carry, and which admits under explicitly provisional ceilings.
        // Do not infer authority from an FL2VA record, and do not widen this
        // to `feature = "h3"`: a shipping build must keep refusing Ref2VA.
        #[cfg(feature = "h3-private-uat")]
        Task::Ref2va => true,
        #[cfg(not(feature = "h3-private-uat"))]
        Task::Ref2va => false,
    }
}

/// One scheduler-visible GPU route eligible for authenticated H3
/// capability presentation. Supplying this record grants nothing: the
/// source-controlled runtime record must match it exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct H3PrivatePresentationRoute<'a> {
    pub device_id: &'a str,
    pub device_ordinal: usize,
    /// `Some` identifies an exact CUDA architecture; `None` identifies Metal.
    pub compute_capability: Option<(u16, u16)>,
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
    compute_capability: Option<(u16, u16)>,
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

    pub const fn compute_capability(&self) -> Option<(u16, u16)> {
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
        self.validate_with_adapter(None)
    }

    /// Validate the envelope against an optional authenticated Turbo adapter.
    ///
    /// The reviewed canvas is identical either way — same 1344x768, 124 frames,
    /// 24 fps, one first-frame endpoint. The ONLY axis a Turbo tier moves is
    /// the step count, and it may move it only to the count that tier's
    /// distillation was reviewed for. Without an adapter the 21-step pin is
    /// exactly as strict as before.
    pub(crate) fn validate_with_adapter(
        &self,
        turbo: Option<&H3FactoryTurboAdapterAuthority>,
    ) -> Result<()> {
        self.validate_for_task_with_adapter(Task::Fl2va, turbo)
    }

    /// The record's serialized shape is task-neutral; what a task changes is
    /// which conditioning fields are meaningful. FL2VA pins exactly one
    /// first-frame endpoint and no condition audio, while Ref2VA carries no
    /// endpoint at all and prices its conditioning from ordered references.
    pub(crate) fn validate_for_task(&self, task: Task) -> Result<()> {
        self.validate_for_task_with_adapter(task, None)
    }

    /// The general form. The step axis (a reviewed Turbo tier) and the
    /// conditioning axis (the task) are orthogonal, but they are not
    /// independent: a tier may only move the step count of an envelope for the
    /// task its own distillation was reviewed for. Reducing the adapter to a
    /// bare step count loses exactly that, and since the FL2V 768p and Ref2V
    /// tiers are both 5-point schedules, an FL2V adapter would otherwise be
    /// indistinguishable from a Ref2V one on a Ref2VA envelope.
    pub(crate) fn validate_for_task_with_adapter(
        &self,
        task: Task,
        turbo: Option<&H3FactoryTurboAdapterAuthority>,
    ) -> Result<()> {
        let reviewed_steps = match turbo {
            None => None,
            Some(turbo) => {
                match turbo.reviewed_task() {
                    Some(adapter_task) if adapter_task == task => {}
                    // An unrecognised tier id is a mismatch, never a wildcard.
                    _ => bail!(
                        "private H3 Turbo adapter {} was not reviewed for the {task:?} envelope",
                        turbo.tier_stable_id()
                    ),
                }
                Some(turbo.grid_points())
            }
        };
        self.validate_for_task_with_reviewed_steps(task, reviewed_steps)
    }

    /// Apply a step count whose task scoping the CALLER has already
    /// established. Only two callers qualify: the adapter form above, which
    /// has just matched the tier's reviewed task, and the compiled public
    /// profile, which independently pins `record.task == "fl2va"`. Everything
    /// else must go through the adapter form so the tier identity is checked.
    fn validate_for_task_with_reviewed_steps(
        &self,
        task: Task,
        turbo_steps: Option<u32>,
    ) -> Result<()> {
        let reviewed_steps = turbo_steps.unwrap_or(contract::COMFY_DEFAULT_STEPS);
        if self.max_steps != reviewed_steps {
            bail!(
                "private H3 runtime qualification envelope allows {reviewed_steps} steps, not {}",
                self.max_steps
            )
        }
        self.validate_shape(task)
    }

    fn validate_shape(&self, task: Task) -> Result<()> {
        let conditioning_ok = match task {
            Task::Fl2va => {
                self.endpoint_count == 1
                    && self.endpoint_anchor == "first"
                    && self.max_condition_visual_rows > 0
            }
            Task::Ref2va => {
                self.endpoint_count == 0
                    && self.endpoint_anchor == "none"
                    && self.max_condition_visual_rows > 0
            }
        };
        if self.width != contract::DEFAULT_WIDTH
            || self.height != contract::DEFAULT_HEIGHT
            || self.frames != contract::REVIEWED_COMPACT_FRAMES
            || self.fps != contract::FIXED_FPS
            || self.batch_size != 1
            // The step axis is owned by `validate_for_task_with_adapter`,
            // which is the only place a reviewed Turbo tier may move it.
            || !conditioning_ok
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

    /// Validate a prepared request against this envelope.
    ///
    /// There is deliberately no `None`-defaulting wrapper: passing the wrong
    /// step authority silently rejects every Turbo render (it did, in the first
    /// cut of this wiring), so each caller has to name which one it holds.
    #[cfg(feature = "mp4")]
    fn validate_prepared_with_adapter(
        &self,
        request: &H3FactoryPreparedRequestInput,
        turbo: Option<&H3FactoryTurboAdapterAuthority>,
    ) -> Result<()> {
        // The request's own task selects the conditioning contract, and the
        // adapter must have been reviewed for that same task before its step
        // count is applied.
        self.validate_for_task_with_adapter(request.task, turbo)?;
        // Name every differing axis: one bare mismatch sentence made a
        // wrong-tier step count, an off-canvas size, and an over-cap prompt
        // all read identically, which is undebuggable from a client.
        let mut mismatches: Vec<String> = Vec::new();
        fn exact_into(mismatches: &mut Vec<String>, name: &str, requested: u64, reviewed: u64) {
            if requested != reviewed {
                mismatches.push(format!("{name} {requested} (reviewed {reviewed})"));
            }
        }
        exact_into(
            &mut mismatches,
            "width",
            u64::from(request.width),
            u64::from(self.width),
        );
        exact_into(
            &mut mismatches,
            "height",
            u64::from(request.height),
            u64::from(self.height),
        );
        exact_into(
            &mut mismatches,
            "frames",
            u64::from(request.frames),
            u64::from(self.frames),
        );
        exact_into(
            &mut mismatches,
            "fps",
            u64::from(request.fps),
            u64::from(self.fps),
        );
        exact_into(
            &mut mismatches,
            "batch_size",
            u64::from(request.batch_size),
            u64::from(self.batch_size),
        );
        exact_into(
            &mut mismatches,
            "grid_points",
            u64::from(request.grid_points),
            u64::from(self.max_steps),
        );
        if !request.synchronized_audio {
            mismatches.push("synchronized_audio false (reviewed true)".to_string());
        }
        if !request.mp4_output {
            mismatches.push("mp4_output false (reviewed true)".to_string());
        }
        exact_into(
            &mut mismatches,
            "endpoints",
            request.endpoints.len() as u64,
            u64::from(self.endpoint_count),
        );
        mismatches.extend(self.conditioning_mismatches(request));
        mismatches.extend(self.row_cap_mismatches(&request.rows));
        if !mismatches.is_empty() {
            bail!(
                "private H3 request differs from the reviewed compact-quality envelope: {}",
                mismatches.join("; ")
            )
        }
        Ok(())
    }

    /// Every conditioning axis this envelope pins, named individually.
    ///
    /// This used to be one boolean `conditioning_ok` reported as "conditioning
    /// shape for Fl2va", which told a client four different things could be
    /// wrong and which of them was not one of them. Each sub-axis is now its
    /// own entry carrying the requested value beside the reviewed one, and a
    /// request that breaks several names all of them.
    #[cfg(feature = "mp4")]
    fn conditioning_mismatches(&self, request: &H3FactoryPreparedRequestInput) -> Vec<String> {
        /// The envelope stores its anchor as one of these three names, so the
        /// requested side is rendered into the same domain to be comparable.
        fn requested_anchor(request: &H3FactoryPreparedRequestInput) -> &'static str {
            match request.endpoints.first().map(|endpoint| endpoint.anchor) {
                Some(H3FactoryEndpointAnchor::First) => "first",
                Some(H3FactoryEndpointAnchor::Last) => "last",
                None => "none",
            }
        }

        let mut mismatches = Vec::new();
        let reviewed_mode = match request.task {
            Task::Fl2va => Mode::FirstFrameToAudioVideo,
            Task::Ref2va => Mode::ReferenceToAudioVideo,
        };
        if request.mode != reviewed_mode {
            mismatches.push(format!(
                "mode {:?} (reviewed {reviewed_mode:?} for {:?})",
                request.mode, request.task
            ));
        }
        let anchor = requested_anchor(request);
        if anchor != self.endpoint_anchor {
            mismatches.push(format!(
                "endpoint_anchor {anchor} (reviewed {})",
                self.endpoint_anchor
            ));
        }
        match request.task {
            Task::Fl2va => {
                if request.rows.condition_audio_rows != 0 {
                    mismatches.push(format!(
                        "condition_audio_rows {} (reviewed 0 for Fl2va)",
                        request.rows.condition_audio_rows
                    ));
                }
            }
            Task::Ref2va => {
                if request.references.is_empty() {
                    mismatches.push("references 0 (reviewed at least 1 for Ref2va)".to_string());
                }
                // The reference soundtrack cap rides the target-audio cap:
                // both are 32 kHz stereo latents over the same duration.
                if request.rows.condition_audio_rows > self.max_target_audio_rows {
                    mismatches.push(format!(
                        "condition_audio_rows {} (cap {})",
                        request.rows.condition_audio_rows, self.max_target_audio_rows
                    ));
                }
            }
        }
        mismatches
    }

    /// Every row ceiling this envelope imposes, named individually.
    ///
    /// Split out of the shape check above because the row ceilings — unlike
    /// the canvas, the step count, or the conditioning shape — are knowable
    /// from the tokenized presentation alone, which admission produces long
    /// before it has an authenticated runtime qualification. The precheck at
    /// [`precheck_private_h3_prepared_rows`] asks exactly this question first
    /// so an over-budget prompt is refused in seconds instead of after the
    /// artifact pass has hashed ~37 GB; the shape check keeps asking it as the
    /// backstop, from the authenticated record rather than the compiled one.
    #[cfg(feature = "mp4")]
    fn row_cap_mismatches(&self, rows: &H3FactoryPreparedRowsInput) -> Vec<String> {
        [
            (
                "qwen_output_text_rows",
                rows.qwen_output_text_rows,
                self.max_qwen_output_text_rows,
            ),
            (
                "qwen_vision_rows",
                rows.qwen_vision_rows,
                self.max_qwen_vision_rows,
            ),
            (
                "condition_visual_rows",
                rows.condition_visual_rows,
                self.max_condition_visual_rows,
            ),
            (
                "target_video_rows",
                rows.target_video_rows,
                self.max_target_video_rows,
            ),
            (
                "target_audio_rows",
                rows.target_audio_rows,
                self.max_target_audio_rows,
            ),
            (
                "total_packed_rows",
                rows.total_packed_rows,
                self.max_total_packed_rows,
            ),
        ]
        .into_iter()
        .filter(|(_, requested, reviewed)| requested > reviewed)
        .map(|(name, requested, reviewed)| format!("{name} {requested} (cap {reviewed})"))
        .collect()
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
    /// Captured process RSS at runtime entry, before any conditioner parameter
    /// is retained. It is independent of how those parameters are represented,
    /// so #1316's narrowing of the NVFP4 block-scale cache does not invalidate
    /// this observation and required no re-capture campaign.
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
    let metal =
        released_h3_private_qwen_loader_memory_authority(H3PrivateQwenLoaderMemoryRoute::Metal)?
            .host_resident_parameter_bytes;
    bounds
        .fixed_runtime_host_bytes
        .checked_add(cpu.min(accelerated).min(metal))
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
    compute_capability: Option<(u16, u16)>,
    available_device_bytes: u64,
    available_host_headroom_bytes: u64,
) -> Result<()> {
    let device_floor = private_h3_admission_device_floor_bytes(bounds)?;
    let host_floor = private_h3_admission_host_floor_bytes(bounds)?;
    if compute_capability.is_none() {
        let unified_floor = device_floor.max(host_floor);
        if unified_floor > available_device_bytes {
            bail!(
                "private H3 Metal admission needs at least {unified_floor} unified-memory bytes \
                 before any request-specific term, exceeding the {available_device_bytes} byte \
                 admission sample"
            )
        }
        return Ok(());
    }
    if device_floor > available_device_bytes || host_floor > available_host_headroom_bytes {
        bail!(
            "private H3 admission needs at least {device_floor} device and {host_floor} host \
             bytes before any request-specific term, exceeding the {available_device_bytes} \
             device and {available_host_headroom_bytes} host admission sample"
        )
    }
    Ok(())
}

/// Refuse an over-budget prompt as soon as the tokenized presentation exists.
///
/// The row ceilings are the one part of the reviewed envelope a request can
/// violate purely by being typed, and the prompt is the only axis a user
/// controls: everything else in the packed sequence is the reviewed canvas.
/// Asking about them here — before the artifact pass walks ~37 GB of weights —
/// is what turns "the app hung for ninety seconds and then said the request
/// differs from the reviewed envelope" into an immediate sentence naming the
/// budget (#1245).
///
/// The budget is DERIVED, never transcribed: the presentation overhead is
/// whatever the tokenized sequence held beyond the prompt's own tokens (the
/// `"<Picture N>: "` labels and the endpoint's merged vision pads), so the
/// number reported to the user is the number this build would actually accept.
///
/// This is a lower bound of the authenticated envelope check the runtime
/// qualification performs later, not a replacement for it: the compiled
/// reviewed ceilings and the authenticated record's ceilings are the same
/// values, so a refusal here always implies a refusal there.
#[cfg(feature = "mp4")]
fn precheck_private_h3_prepared_rows(
    envelope: &H3PrivateRuntimeEnvelopeRecord,
    rows: &H3FactoryPreparedRowsInput,
    prompt_tokens: u64,
) -> Result<()> {
    if rows.qwen_output_text_rows > envelope.max_qwen_output_text_rows {
        let presentation_overhead_rows = rows.qwen_output_text_rows.saturating_sub(prompt_tokens);
        let prompt_budget_tokens = envelope
            .max_qwen_output_text_rows
            .saturating_sub(presentation_overhead_rows);
        bail!(
            "prompt is {prompt_tokens} tokens; the reviewed MiniMax H3 envelope has room for \
             {prompt_budget_tokens} prompt tokens (the conditioner sequence would be {} rows \
             against a reviewed ceiling of {})",
            rows.qwen_output_text_rows,
            envelope.max_qwen_output_text_rows
        )
    }
    let mismatches = envelope.row_cap_mismatches(rows);
    if !mismatches.is_empty() {
        bail!(
            "private H3 request exceeds the reviewed compact-quality envelope: {}",
            mismatches.join("; ")
        )
    }
    Ok(())
}

/// Refuse an exact prepared target the admission sample cannot hold.
///
/// Device VRAM and host headroom are two independent resources with two
/// independent samples, so they are two independent checks: one OR'd message
/// could not tell an operator which one fell short, and printed neither
/// sample (#1214). Both refusals are ordinary insufficient-memory errors —
/// nothing here may route through the fatal-CUDA quarantine.
#[cfg(feature = "mp4")]
fn check_private_h3_target_budget_fits(
    predicted_device_peak_bytes: u64,
    predicted_host_increment_bytes: u64,
    compute_capability: Option<(u16, u16)>,
    available_device_bytes: u64,
    available_host_headroom_bytes: u64,
) -> Result<()> {
    if compute_capability.is_none() {
        let unified_peak = predicted_device_peak_bytes.max(predicted_host_increment_bytes);
        if unified_peak > available_device_bytes {
            bail!(
                "private H3 Metal canonical target needs {unified_peak} unified-memory bytes but \
                 the admission sample offers {available_device_bytes}"
            )
        }
        return Ok(());
    }
    if predicted_device_peak_bytes > available_device_bytes {
        bail!(
            "private H3 canonical target needs {predicted_device_peak_bytes} device bytes but the \
             admission sample offers {available_device_bytes}"
        )
    }
    if predicted_host_increment_bytes > available_host_headroom_bytes {
        bail!(
            "private H3 canonical target needs {predicted_host_increment_bytes} host bytes but \
             the admission headroom sample offers {available_host_headroom_bytes}"
        )
    }
    Ok(())
}

/// Exact peak of H3's reviewed phase order on an Apple unified-memory device.
///
/// Host and device charges within one phase coexist and are therefore added;
/// different phases are mutually exclusive under the authenticated load/drop
/// policy and are therefore compared with `max`. Taking the maximum of the two
/// independent aggregate peaks would miss the smaller simultaneous charge.
#[cfg(feature = "mp4")]
fn private_h3_unified_target_peak_bytes(budget: &H3FactoryTargetBudgetInput) -> Result<u64> {
    let phases = [
        (
            budget.reference_decode_phase_device_bytes,
            budget.reference_decode_phase_host_bytes,
        ),
        (
            budget.reference_preprocess_phase_device_bytes,
            budget.reference_preprocess_phase_host_bytes,
        ),
        (
            budget.reference_visual_encode_phase_device_bytes,
            budget.reference_visual_encode_phase_host_bytes,
        ),
        (
            budget.reference_audio_encode_phase_device_bytes,
            budget.reference_audio_encode_phase_host_bytes,
        ),
        (
            budget.vae_load_phase_device_bytes,
            budget.vae_load_phase_host_bytes,
        ),
        (
            budget.qwen_encode_phase_device_bytes,
            budget.qwen_encode_phase_host_bytes,
        ),
        (
            budget.qwen_transfer_phase_device_bytes,
            budget.qwen_transfer_phase_host_bytes,
        ),
        (
            budget.condition_encode_phase_device_bytes,
            budget.condition_encode_phase_host_bytes,
        ),
        (
            budget.noise_allocation_phase_device_bytes,
            budget.noise_allocation_phase_host_bytes,
        ),
        (
            budget.transformer_load_phase_device_bytes,
            budget.transformer_load_phase_host_bytes,
        ),
        (
            budget.denoise_phase_device_bytes,
            budget.denoise_phase_host_bytes,
        ),
        (
            budget.visual_decode_phase_device_bytes,
            budget.visual_decode_phase_host_bytes,
        ),
        (
            budget.audio_decode_phase_device_bytes,
            budget.audio_decode_phase_host_bytes,
        ),
        (
            budget.waveform_transfer_phase_device_bytes,
            budget.waveform_transfer_phase_host_bytes,
        ),
        (budget.mux_phase_device_bytes, budget.mux_phase_host_bytes),
    ];
    phases.into_iter().try_fold(0, |peak, (device, host)| {
        device
            .checked_add(host)
            .map(|phase| peak.max(phase))
            .ok_or_else(|| anyhow!("private H3 unified-memory target phase overflow"))
    })
}

#[cfg(feature = "mp4")]
fn private_h3_qwen_route(
    compute_capability: Option<(u16, u16)>,
) -> (
    H3PrivateQwenLoaderMemoryRoute,
    H3FactoryConditionerPlacement,
) {
    if compute_capability.is_some() {
        (
            H3PrivateQwenLoaderMemoryRoute::Cpu,
            H3FactoryConditionerPlacement::HostCpuThenDrop,
        )
    } else {
        (
            H3PrivateQwenLoaderMemoryRoute::Metal,
            H3FactoryConditionerPlacement::AssignedMetalThenDrop,
        )
    }
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
        // The request keeps its full reviewed identity — a Turbo tag included
        // — while the engine partition it executes as must be its own task's
        // compact base, and the mode must be the task's own conditioning
        // shape. For FL2VA this is exactly the old constant pin.
        let mode_matches_task = match self.task {
            Task::Fl2va => self.mode != Mode::ReferenceToAudioVideo,
            Task::Ref2va => self.mode == Mode::ReferenceToAudioVideo,
        };
        if contract::base_compact_model(&self.canonical_model)
            != Some(contract::base_compact_model_for_task(self.task))
            || !contract::is_reviewed_compact_model(&self.canonical_model)
            || !mode_matches_task
            || self.width == 0
            || self.height == 0
            || self.frames == 0
            || self.fps != contract::FIXED_FPS
        {
            bail!("private H3 frozen media contract does not pair with its exact task")
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
    /// Verified bindings for the ordered Ref2VA references, minted by the
    /// caller from the staged set immediately before owner preparation —
    /// the same shape admission consumed, because the reopen re-derives the
    /// exact frozen factory request through the same decoder. Empty for
    /// FL2VA.
    pub references: &'a [GenerationReferenceBinding],
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
    /// `Some` identifies an exact CUDA architecture; `None` identifies Metal.
    pub compute_capability: Option<(u16, u16)>,
    pub available_device_bytes: u64,
    /// Host RAM available after the server's canonical 15%-or-8-GiB safety
    /// floor, never the raw operating-system available-memory sample.
    pub available_host_headroom_bytes: u64,
    /// Verified bindings for the ordered Ref2VA references, minted by the
    /// caller from the staged set immediately before admission.
    ///
    /// Empty for FL2VA, whose conditioning rides the endpoint contract. For
    /// Ref2VA these are what let admission derive the prepared shapes and the
    /// native/normalized retained-media geometry the target budget is sized
    /// from — it decodes through the same media adapter the runtime uses, so
    /// there is exactly one decoder. The bindings carry descriptors, never
    /// bytes, and nothing derived from them reaches the request, the queue
    /// journal, or the gallery.
    pub references: &'a [GenerationReferenceBinding],
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
    compute_capability: Option<(u16, u16)>,
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

    pub const fn compute_capability(&self) -> Option<(u16, u16)> {
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

    /// Revalidate this immutable DTO against an exact request and GPU route.
    /// Current capacity may differ from the admission snapshot, but both
    /// current values must still cover the canonical target peaks.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        request: &GenerateRequest,
        device_id: &str,
        device_ordinal: usize,
        compute_capability: Option<(u16, u16)>,
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
        // The recorded route must be a coherent pinned pair, and the resolved
        // request must satisfy THAT route. Re-asserting FL2VA here is what
        // killed the Ref2VA evidence the rest of admission had just derived.
        let recorded_contract = contract::capability_contract_for_model(&self.canonical_model);
        if !recorded_contract.is_some_and(|recorded| recorded.task == self.task)
            || contract::validate_resolved_request_contract(&resolved, self.task)
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
            || self.base_factory_authority.compute_capability() != self.compute_capability
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
/// The once-derived admission route, carrying BOTH model names.
///
/// `admitted_model` keeps the full reviewed request identity — a Turbo tag
/// included — for the media contract, provenance, and Turbo adapter
/// resolution. `partition_model` is the compact engine partition that
/// identity executes as (`base_compact_model`): artifact qualification, Qwen
/// support, the factory authority, execution fingerprints, and admission
/// evidence are all keyed on it, and the terminal media pairing
/// (`media_model_matches_h3_authority`) expects exactly that split. For the
/// base task models the two names are identical, so Ref2VA's task threading
/// is unchanged. Deriving either name downstream instead of here is what let
/// a Turbo tag reach `qualification_manifest` as itself and be refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct H3AdmittedRoute {
    pub(crate) admitted_model: &'static str,
    pub(crate) partition_model: &'static str,
    pub(crate) task: Task,
}

pub(crate) fn admitted_h3_route(model: &str) -> Result<H3AdmittedRoute> {
    let admitted = contract::capability_contract_for_model(model)
        .ok_or_else(|| anyhow!("private H3 admission names an unknown model"))?;
    let partition_model =
        contract::base_compact_model(admitted.canonical_model).ok_or_else(|| {
            anyhow!("private H3 admission names a model without a compact engine partition")
        })?;
    Ok(H3AdmittedRoute {
        admitted_model: admitted.canonical_model,
        partition_model,
        task: admitted.task,
    })
}

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
        references,
        device_id,
        device_ordinal,
        compute_capability,
        available_device_bytes,
        available_host_headroom_bytes,
    } = input;
    if device_id.trim().is_empty()
        || compute_capability.is_some_and(|capability| capability.0 == 0)
        || !private_h3_capacity_sample_is_concrete(
            compute_capability,
            available_device_bytes,
            available_host_headroom_bytes,
        )
    {
        bail!("private H3 admission requires one concrete nonempty GPU capacity sample")
    }
    // The admitted route is derived once, from the request model's own pinned
    // contract, and threaded through every step below. Reading it again from a
    // constant is what let a Ref2VA request load FL2VA support and then fail
    // its own cross-task check — and the route carries BOTH names, because a
    // Turbo tag is a first-class canonical identity whose engine partition is
    // the base compact model: qualification, Qwen support, the factory
    // authority, and admission evidence key on the partition, while media
    // facts and provenance keep the full tag. Threading the tag into the
    // partition consumers is what refused every Turbo admission with
    // "requires an exact Comfy H3 canonical model name".
    let route = admitted_h3_route(&request.model)?;
    let admitted_model = route.admitted_model;
    let partition_model = route.partition_model;
    let admitted_task = route.task;
    let (admitted_transformer_task, admitted_published_artifact) = match admitted_task {
        Task::Fl2va => (
            H3TransformerTask::T2VaFl2Va,
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
        ),
        Task::Ref2va => (
            H3TransformerTask::Ref2Va,
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot,
        ),
    };
    // Admission runs after reference resolution, so Ref2VA media are already
    // descriptor authorities — the only valid queue/worker form, and the one
    // the public-boundary validator refuses. FL2VA carries no references, so
    // this is the same check it always made.
    let mode = contract::validate_resolved_request_contract(request, admitted_task)
        .map_err(|error| anyhow!("{}: {}", error.code, error.message))?;
    let submitted_request_identity_sha256 = private_h3_request_identity(request)?;

    // Private UAT retains its reviewed-record gate before bulk model I/O —
    // resolved per task, because Ref2VA has no reviewed record and its only
    // authority is the compiled capture-scope profile. Ordinary public H3
    // derives authority from compiled policy, the exact artifact graph, and
    // the live SM89 attention route instead.
    #[cfg(not(feature = "h3"))]
    let runtime_qualification_source =
        private_runtime_qualification_source(admitted_task, paths.runtime_qualification_record)?;
    #[cfg(not(feature = "h3"))]
    runtime_qualification_source.validate_route(device_id, device_ordinal, compute_capability)?;
    // Cheap capacity floors BEFORE the artifact pass below hashes ~37 GB.
    // Refusing a hopeless device after several minutes of SHA-256 was the
    // worst failure this path had; both floors are provable lower bounds of
    // the exact per-attempt budget compared at the end of this function, so
    // this can only ever turn a slow refusal into a fast one.
    #[cfg(feature = "h3")]
    let precheck_bounds = public_runtime_bounds();
    #[cfg(not(feature = "h3"))]
    let precheck_bounds = runtime_qualification_source.precheck_bounds();
    precheck_private_h3_admission_capacity(
        &precheck_bounds,
        compute_capability,
        available_device_bytes,
        available_host_headroom_bytes,
    )?;
    // The prepared request is built BEFORE that artifact pass for the same
    // reason the capacity floors are checked before it: the conditioner
    // presentation is a tokenizer call and some CPU image normalization, while
    // the pass that follows walks ~37 GB of weights. An over-budget prompt is
    // the one refusal ordinary users hit repeatedly (#1245), and paying ninety
    // seconds of SHA-256 to learn it was undebuggable. Nothing here opens a
    // CUDA device, and nothing it produces is trusted on its own — the
    // authenticated envelope check below still validates the same request
    // against the record the artifact pass authorizes.
    let storage = H3PrivateComfyStorageAuthority::resolve(paths.models_root, admitted_task)?;
    let qwen_support = load_qualified_private_qwen_support(paths.models_root, partition_model)?;
    let mut prepare_observer = H3EngineProgressObserver::new(progress);
    // Conditioning is task-shaped: FL2VA normalizes its boundary endpoints
    // here, Ref2VA decodes and normalizes its ordered references through the
    // same media adapter the runtime uses. Both run CPU-only, before the
    // allocation commit, and neither opens a CUDA device.
    let admission_request = match admitted_task {
        Task::Fl2va => {
            if !references.is_empty() {
                bail!("private H3 FL2VA admission was handed Ref2VA reference bindings")
            }
            prepare_private_fl2va_admission_request(
                request,
                &qwen_support,
                progress,
                &mut prepare_observer,
            )?
        }
        Task::Ref2va => {
            if references.is_empty() {
                bail!("private H3 Ref2VA admission has no staged reference bindings")
            }
            prepare_private_ref2va_admission_request(
                request,
                references,
                &qwen_support,
                progress,
                &mut prepare_observer,
            )?
        }
    };
    // The default-step envelope is the right one to ask even for a Turbo tag:
    // a reviewed tier moves ONLY `max_steps`, and this precheck reads only row
    // ceilings. The authenticated backstop below still validates the step axis
    // against the tier's own minted envelope.
    #[cfg(feature = "h3")]
    let precheck_envelope = public_runtime_envelope();
    #[cfg(not(feature = "h3"))]
    let precheck_envelope = runtime_qualification_source.precheck_envelope();
    precheck_private_h3_prepared_rows(
        &precheck_envelope,
        &admission_request.request.rows,
        admission_request.prompt_tokens,
    )?;
    let artifact_report = qualify_private_artifacts_with_control(
        paths.models_root,
        partition_model,
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

    let attention_model = H3AttentionModelContract::released_bf16();
    let attention = match compute_capability {
        Some(compute_capability) => H3AttentionRuntimeAuthority::qualify_flash_attention_v2(
            H3AttentionDevice::Cuda {
                compute_capability: Some(compute_capability),
            },
            attention_model,
        ),
        None => H3AttentionRuntimeAuthority::metal_chunked_dense(attention_model),
    }
    .map_err(|error| anyhow!(error.to_string()))?;
    #[cfg(feature = "h3")]
    let attention_qualification_sha256 = attention.identity_sha256().to_string();
    #[cfg(not(feature = "h3"))]
    let attention_qualification_sha256 =
        runtime_qualification_source.attention_qualification_sha256(&attention);
    let attention_input = H3FactoryAttentionInput {
        generic_backend: if compute_capability.is_some() {
            AttentionBackend::Flash
        } else {
            AttentionBackend::Math
        },
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
    // Authenticating here reads and digests the adapter but allocates no device
    // memory; the deltas are materialized in the transformer-load phase, which
    // is the phase whose budget charges them. It has to happen before the
    // runtime qualification so the envelope can be minted at the tier's own
    // reviewed step count. Selection is by the request's model identity (a
    // reviewed Turbo manifest tag); the env pair survives only as the
    // capture-scope UAT override inside `resolve_turbo_selection`.
    let turbo_adapter =
        super::turbo::resolve_turbo_authority_for_request(admitted_model, paths.models_root)?;
    #[cfg(not(feature = "h3"))]
    let private_compute_capability = compute_capability
        .ok_or_else(|| anyhow!("private H3 reviewed evidence requires one concrete CUDA route"))?;
    #[cfg(not(feature = "h3"))]
    let runtime_qualification = runtime_qualification_source.authenticate(
        &artifact_report,
        device_id,
        device_ordinal,
        private_compute_capability,
        attention.identity_sha256(),
        attention.kernel().identity(),
        &attention_qualification_sha256,
    )?;
    #[cfg(feature = "h3")]
    let runtime_qualification = public_runtime_qualification(
        &artifact_report,
        device_id,
        device_ordinal,
        compute_capability,
        attention.identity_sha256(),
        attention.kernel().identity(),
        &attention_qualification_sha256,
        turbo_adapter.as_ref(),
    )?;
    progress.checkpoint()?;

    let transformer_cancellation = H3PrivatePreparationCancellation { progress };
    let opened_transformer = open_h3_comfy_published_int8_checkpoint(
        storage.transformer_path(),
        admitted_transformer_task,
        admitted_published_artifact,
        &transformer_cancellation,
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    let vae_plan = storage.vae_plan(paths.staging_root)?;
    let mut vae_observer = H3PrivatePreparationVaeObserver::new(progress);
    let opened_vae = open_h3_comfy_vae_authority(&vae_plan, &mut vae_observer);
    let opened_vae = vae_observer.finish(opened_vae)?;

    // The qualification above was minted at this tier's reviewed step count;
    // validating the prepared request against the baseline 21-step envelope
    // would reject every Turbo attempt. The row ceilings were already asked
    // before the artifact pass; this is the authenticated backstop over every
    // axis, the compiled precheck's ceilings included.
    runtime_qualification.validate_prepared_envelope_with_turbo(
        &admission_request.request,
        turbo_adapter.as_ref(),
    )?;
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
        partition_model,
        admitted_task,
        &resolved_request_identity_sha256,
        &artifact_report,
        runtime_qualification.identity_sha256(),
        device_id,
        device_ordinal,
        compute_capability,
        attention.identity_sha256(),
        &component_digests,
    );
    let (qwen_route, conditioner_placement) = private_h3_qwen_route(compute_capability);
    let qwen_memory = released_h3_private_qwen_loader_memory_authority(qwen_route)?;
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
            model: partition_model.into(),
            device_id: device_id.into(),
            device_ordinal,
            compute_capability,
            execution_fingerprint: execution_fingerprint.clone(),
            conditioner_placement,
            qwen_parameter_bytes: qwen_memory.source_parameter_bytes,
            qwen_host_resident_parameter_bytes: qwen_memory.host_resident_parameter_bytes,
            qwen_device_resident_parameter_bytes: qwen_memory.device_resident_parameter_bytes,
            // Request-derived for Ref2VA, the reviewed grant verbatim for
            // FL2VA — the same seam the budget builder charges through, so
            // the freeze-time projection comparison cannot drift.
            qwen_activation_workspace_bytes: qwen_activation_workspace_demand_bytes(
                &admission_request.request,
                runtime_qualification
                    .bounds()
                    .qwen_activation_workspace_bytes,
            )?,
            qwen_maximum_tensor_staging_bytes: qwen_memory.maximum_tensor_staging_bytes,
            qwen_retained_raw_header_bytes: qwen_memory.retained_raw_header_bytes,
            qwen_output_text_rows: admission_request.request.rows.qwen_output_text_rows,
            qwen_vision_rows: admission_request.request.rows.qwen_vision_rows,
            condition_visual_rows: admission_request.request.rows.condition_visual_rows,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: if compute_capability.is_some() {
                AttentionBackend::Flash
            } else {
                AttentionBackend::Math
            },
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
    let raw_device_peak_bytes = prepared_attempt.target_budget.predicted_device_peak_bytes;
    let raw_host_increment_bytes = prepared_attempt
        .target_budget
        .predicted_host_increment_bytes;
    let (predicted_device_peak_bytes, predicted_host_increment_bytes) =
        if compute_capability.is_none() {
            (
                private_h3_unified_target_peak_bytes(&prepared_attempt.target_budget)?,
                0,
            )
        } else {
            (raw_device_peak_bytes, raw_host_increment_bytes)
        };
    check_private_h3_target_budget_fits(
        predicted_device_peak_bytes,
        predicted_host_increment_bytes,
        compute_capability,
        available_device_bytes,
        available_host_headroom_bytes,
    )?;
    let budget_echo = H3FactoryExecutionBudgetEchoInput {
        prepared_attempt_identity_sha256: prepared_attempt.identity_sha256.clone(),
        device_peak_bytes: raw_device_peak_bytes,
        host_increment_bytes: raw_host_increment_bytes,
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
        // The engine partition, matching the factory authority the validate
        // path equates it with; the request identity hashes retain the full
        // reviewed tag.
        canonical_model: partition_model.into(),
        task: admitted_task,
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
fn private_h3_capacity_sample_is_concrete(
    compute_capability: Option<(u16, u16)>,
    available_device_bytes: u64,
    available_host_headroom_bytes: u64,
) -> bool {
    available_device_bytes > 0
        && (compute_capability.is_none() || available_host_headroom_bytes > 0)
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
    partition_model: &str,
    task: Task,
    resolved_request_identity_sha256: &str,
    artifact_report: &H3PrivateArtifactQualificationReport,
    runtime_qualification_identity_sha256: &str,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: Option<(u16, u16)>,
    attention_runtime_identity_sha256: &str,
    components: &[H3PrivateComponentDigest],
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-admission-execution.v1\0");
    // The route's engine partition and task, not constants: Ref2VA evidence
    // must mint in its own execution namespace even though the full-request
    // hash already separates the two tasks.
    for value in [
        partition_model,
        super::private_qualification::task_id(task),
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
    match compute_capability {
        Some((major, minor)) => {
            digest.update(b"cuda\0");
            digest.update(major.to_le_bytes());
            digest.update(minor.to_le_bytes());
        }
        None => digest.update(b"metal\0"),
    }
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
    let (backend, major, minor) = match evidence.compute_capability {
        Some((major, minor)) => (1, u64::from(major), u64::from(minor)),
        None => (2, 0, 0),
    };
    for value in [
        evidence.device_ordinal as u64,
        backend,
        major,
        minor,
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
    /// `Some` identifies an exact CUDA architecture; `None` identifies Metal.
    pub compute_capability: Option<(u16, u16)>,
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
            || self
                .compute_capability
                .is_some_and(|capability| capability.0 == 0)
            || self.memory_ledger_sequence == 0
            || self.predicted_device_peak_bytes == 0
            // Metal folds simultaneously-live host and device bytes into the
            // one reviewed unified-memory peak, so its independent host claim
            // is deliberately zero. CUDA retains two physical pools and must
            // carry a positive host increment.
            || (self.compute_capability.is_some() && self.predicted_host_increment_bytes == 0)
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
        references,
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
    // Every gate from here on is keyed on what admission froze — the frozen
    // factory's own task and its compact engine partition — never on an
    // FL2VA constant, so a frozen Ref2VA attempt reaches its own validators.
    let frozen_route = validate_frozen_reopen_route(request, frozen_factory, &owner_fence)?;
    let mode = frozen_route.mode;
    // The same per-task source admission resolved: the frozen FL2VA route
    // reopens the reviewed record file, the frozen Ref2VA route re-mints the
    // compiled capture-scope profile (deterministic, so the identity fence
    // below still compares it against what admission froze).
    #[cfg(not(feature = "h3"))]
    let runtime_qualification_source = private_runtime_qualification_source(
        frozen_route.task,
        paths.runtime_qualification_record,
    )?;
    #[cfg(not(feature = "h3"))]
    let private_compute_capability = owner_fence
        .compute_capability
        .ok_or_else(|| anyhow!("private H3 reviewed evidence requires one concrete CUDA route"))?;
    #[cfg(not(feature = "h3"))]
    runtime_qualification_source.validate_route(
        &owner_fence.device_id,
        owner_fence.device_ordinal,
        private_compute_capability,
    )?;

    let artifact_report = qualify_private_artifacts_with_control(
        paths.models_root,
        frozen_route.partition_model,
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

    let attention_model = H3AttentionModelContract::released_bf16();
    let attention = match owner_fence.compute_capability {
        Some(compute_capability) => H3AttentionRuntimeAuthority::qualify_flash_attention_v2(
            H3AttentionDevice::Cuda {
                compute_capability: Some(compute_capability),
            },
            attention_model,
        ),
        None => H3AttentionRuntimeAuthority::metal_chunked_dense(attention_model),
    }
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
    let runtime_qualification = runtime_qualification_source.authenticate(
        &artifact_report,
        &owner_fence.device_id,
        owner_fence.device_ordinal,
        private_compute_capability,
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
        frozen_factory.quantization().turbo_adapter(),
    )?;
    if runtime_qualification.identity_sha256() != owner_fence.runtime_qualification_identity_sha256
        || runtime_qualification.artifact_qualification_identity_sha256()
            != owner_fence.artifact_qualification_identity_sha256
    {
        bail!("private H3 reopened runtime qualification differs from admission evidence")
    }
    progress.checkpoint()?;

    // The reopen must follow what admission froze, never a fresh constant.
    // The route above already validated the base factory and keyed the
    // request contract and artifact qualification on the frozen task.
    let reopened_task = frozen_route.task;
    let storage = H3PrivateComfyStorageAuthority::resolve(paths.models_root, reopened_task)?;
    let qwen_support =
        load_qualified_private_qwen_support(paths.models_root, frozen_route.partition_model)?;
    let transformer_cancellation = H3PrivatePreparationCancellation { progress };
    let opened_transformer = open_h3_comfy_published_int8_checkpoint(
        storage.transformer_path(),
        frozen_route.transformer_task,
        frozen_route.published_artifact,
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
    // Recomputed under the frozen factory's own partition and task, so the
    // comparison stays self-consistent with what admission stamped.
    let owner_execution_fingerprint = private_h3_admission_execution_fingerprint(
        frozen_factory.canonical_model(),
        frozen_factory.task(),
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
        references,
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
    let (prepared_fence_device_bytes, prepared_fence_host_bytes) =
        private_h3_project_owner_fence_budget(
            owner_fence.compute_capability,
            prepared.predicted_device_peak_bytes(),
            prepared.predicted_host_increment_bytes(),
            private_h3_unified_target_peak_bytes(&prepared.factory_attempt_input().target_budget)?,
        );
    if prepared.prepared_attempt_identity_sha256() != owner_fence.prepared_attempt_identity_sha256
        || prepared.target_budget_identity_sha256() != owner_fence.target_budget_identity_sha256
        || prepared_fence_device_bytes != owner_fence.predicted_device_peak_bytes
        || prepared_fence_host_bytes != owner_fence.predicted_host_increment_bytes
    {
        bail!("private H3 prepared budget differs from the scheduler owner fence")
    }

    // Media facts carry the request's full reviewed identity — a Turbo tag
    // included — because the terminal gate compares them against the request
    // and the response's provenance records them. The identity must pair with
    // the frozen authority's adapter tier exactly.
    if !crate::h3_factory::media_model_matches_h3_authority(&request.model, frozen_factory) {
        bail!(
            "private H3 request model {:?} does not pair with the frozen factory's Turbo authority",
            request.model
        )
    }
    let media = H3PrivateFl2VaMediaContract {
        canonical_model: request.model.clone(),
        task: reopened_task,
        mode,
        seed,
        width: request.width,
        height: request.height,
        frames: request.frames.unwrap_or(contract::REVIEWED_COMPACT_FRAMES),
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
        sampler: frozen_factory.quantization().sampler_kind(),
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
    let facts = owner.attempt_facts_with_scheduler_budget(
        &consumption_binding,
        prepared_fence_device_bytes,
        prepared_fence_host_bytes,
    );
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
const fn private_h3_project_owner_fence_budget(
    compute_capability: Option<(u16, u16)>,
    split_device_bytes: u64,
    split_host_bytes: u64,
    unified_device_bytes: u64,
) -> (u64, u64) {
    if compute_capability.is_some() {
        (split_device_bytes, split_host_bytes)
    } else {
        (unified_device_bytes, 0)
    }
}

#[cfg(feature = "mp4")]
fn validate_base_factory(
    factory: &FrozenH3FactoryAuthority,
    owner: &H3PrivateFl2VaOwnerFenceFacts,
) -> Result<()> {
    // The gate is keyed on the FROZEN task: the canonical model must be that
    // task's own compact engine partition, and the task must have reviewed
    // runtime availability in this build — which is what keeps a frozen
    // Ref2VA attempt refused everywhere outside the developer campaign build.
    let task = factory.task();
    if !reviewed_h3_private_runtime_available_for_task(task)
        || factory.canonical_model() != contract::base_compact_model_for_task(task)
        || factory.device_id() != owner.device_id
        || factory.device_ordinal() != owner.device_ordinal
        || factory.compute_capability() != owner.compute_capability
        || factory.prepared_target_attempt_identities().is_some()
        || !factory.attention_runtime_identity_sha256().is_empty()
        || factory.attention_backend()
            != private_h3_factory_attention_backend(owner.compute_capability)
        || factory.attention_chunk() != AttentionChunkPolicy::Off
    {
        bail!("private H3 base factory differs from the exact frozen owner route")
    }
    Ok(())
}

#[cfg(feature = "mp4")]
const fn private_h3_factory_attention_backend(
    compute_capability: Option<(u16, u16)>,
) -> AttentionBackend {
    if compute_capability.is_some() {
        AttentionBackend::Flash
    } else {
        AttentionBackend::Math
    }
}

/// The route every reopen gate below the fence is keyed on, derived from the
/// frozen factory rather than from a task constant. `partition_model` is the
/// frozen task's compact engine partition — the identity artifact
/// qualification, storage, and Qwen support key on — mirroring admission's
/// `H3AdmittedRoute` split.
#[cfg(feature = "mp4")]
#[derive(Debug)]
struct H3FrozenReopenRoute {
    task: Task,
    partition_model: &'static str,
    transformer_task: H3TransformerTask,
    published_artifact: H3ComfyPublishedArtifact,
    mode: Mode,
}

/// Validate the frozen factory against the owner fence and the request
/// against the FROZEN task's contract, then hand back the task-keyed route.
/// The request at reopen is the resolved queue/worker form, so Ref2VA
/// references must be descriptor authorities — the same contract admission
/// validated; for FL2VA the resolved and submitted contracts are identical.
#[cfg(feature = "mp4")]
fn validate_frozen_reopen_route(
    request: &GenerateRequest,
    frozen_factory: &FrozenH3FactoryAuthority,
    owner_fence: &H3PrivateFl2VaOwnerFenceFacts,
) -> Result<H3FrozenReopenRoute> {
    validate_base_factory(frozen_factory, owner_fence)?;
    let task = frozen_factory.task();
    let (transformer_task, published_artifact) = match task {
        Task::Fl2va => (
            H3TransformerTask::T2VaFl2Va,
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
        ),
        Task::Ref2va => (
            H3TransformerTask::Ref2Va,
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot,
        ),
    };
    let mode = contract::validate_resolved_request_contract(request, task)
        .map_err(|error| anyhow!("{}: {}", error.code, error.message))?;
    Ok(H3FrozenReopenRoute {
        task,
        partition_model: contract::base_compact_model_for_task(task),
        transformer_task,
        published_artifact,
        mode,
    })
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
    /// Mirrors the factory's own optional capability: `None` is the Metal
    /// route. The lease reports the class the factory froze rather than
    /// asserting CUDA, so a crossed route is caught by the plan comparison
    /// instead of by an unwrap here.
    compute_capability: Option<(u16, u16)>,
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
            compute_capability: factory.compute_capability(),
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
        H3CandleBackendDevice::from_compute_capability(self.compute_capability)
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
pub(crate) struct H3PrivatePreparationCheckpoint<'a> {
    pub(crate) progress: &'a ProgressReporter,
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
        runtime_envelope.validate_prepared_with_adapter(
            prepared.prepared_request_input(),
            authority.quantization().turbo_adapter(),
        )?;
        let observed_envelope = runtime_envelope_observation(prepared.prepared_request_input())?;
        let observed_compute_capability = match attention.device() {
            H3AttentionDevice::Cuda {
                compute_capability: Some((major, minor)),
            } => [major, minor],
            H3AttentionDevice::Metal => [0, 0],
            device => bail!("private H3 terminal authority requires CUDA or Metal, got {device:?}"),
        };
        let observed_authority = authority
            .compute_capability()
            .map(|_| {
                Ok::<_, anyhow::Error>(H3PrivateRuntimeAuthorityObservation {
                    bootstrap_record_sha256: activation
                        .runtime_qualification
                        .record_file_sha256()
                        .into(),
                    runtime_qualification_identity_sha256: activation
                        .runtime_qualification
                        .identity_sha256()
                        .into(),
                    device_id: owner.device_id.clone(),
                    device_ordinal: owner.device_ordinal,
                    compute_capability: observed_compute_capability,
                    attention_runtime_identity_sha256: attention.identity_sha256().into(),
                    attention_kernel_identity: attention.kernel().identity().into(),
                    attention_qualification_sha256: artifact_lease
                        .attention_qualification_sha256
                        .clone(),
                    process: capture_process_observation()?,
                })
            })
            .transpose()?;
        with_private_h3_cuda_execution_attempt(|| {
            consumption_binding.revalidate(&owner, &activation.scheduler_ledger)?;
            let execution_device =
                commit_private_h3_allocation_then(&mut allocation_commit, || {
                    match authority.compute_capability() {
                        Some(_) => Device::new_cuda(owner.device_ordinal)
                            .context("failed to construct the reviewed H3 CUDA route"),
                        None => crate::device::metal_device(owner.device_ordinal)
                            .context("failed to construct the H3 Metal route"),
                    }
                })?;
            let metal_memory_guard =
                H3MetalMemoryGuard::start(&execution_device, cancellation.clone())?;
            let qwen_on_cpu = matches!(
                authority.conditioner_placement(),
                H3FactoryConditionerPlacement::HostCpuThenDrop
            );
            let runtime_bound_capture = observed_authority
                .map(|authority| {
                    H3PrivateRuntimeBoundCapture::begin(
                        &execution_device,
                        qwen_on_cpu,
                        observed_envelope,
                        authority,
                    )
                })
                .transpose()?;
            // Keep one completion fence alive outside the concrete owner graph.
            // It runs only after a successful pipeline result and before any
            // CUDA-bearing local leaves the execution-attempt boundary.
            let completion_device = execution_device.clone();
            let conditioner_device = match authority.conditioner_placement() {
                H3FactoryConditionerPlacement::AssignedCudaThenDrop
                | H3FactoryConditionerPlacement::AssignedMetalThenDrop => execution_device.clone(),
                H3FactoryConditionerPlacement::HostCpuThenDrop => Device::Cpu,
            };
            let conditioner_device_id = match authority.conditioner_placement() {
                H3FactoryConditionerPlacement::AssignedCudaThenDrop
                | H3FactoryConditionerPlacement::AssignedMetalThenDrop => owner.device_id.clone(),
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
                execution_device,
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
            let output = run_private_comfy_fl2va_attempt(phase_owner, progress, &mut observer);
            if let Some(violation) = metal_memory_guard.finish()? {
                bail!(violation);
            }
            let output = output?;
            completion_device
                .synchronize()
                .context("private H3 completion synchronization failed")?;
            cancellation.checkpoint()?;
            progress.checkpoint()?;
            let output = private_run_output(output, owner, &consumption_binding, started)?;
            if let Some(runtime_bound_capture) = runtime_bound_capture {
                let runtime_bound_observation = runtime_bound_capture.finish()?;
                tracing::info!(
                    target: "mold::minimax_h3::private_runtime_bound",
                    observation = %serde_json::to_string(&runtime_bound_observation)?,
                    "captured private MiniMax H3 runtime bounds"
                );
            }
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
        request_warnings: Vec::new(),
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
    if sampler != owner.sampler.as_str()
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
    /// The integrator the frozen quantization authority resolved for this
    /// attempt (RES-multistep without an adapter, first-order Euler under a
    /// reviewed Turbo tier). The terminal provenance gate compares against
    /// this, never a constant — a Turbo render legitimately executes Euler.
    pub(crate) sampler: H3SamplerKind,
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
            || !self.sampler.uses_comfy_simple_grid()
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
        self.attempt_facts_with_scheduler_budget(
            consumption,
            self.predicted_device_peak_bytes,
            self.predicted_host_increment_bytes,
        )
    }

    fn attempt_facts_with_scheduler_budget(
        &self,
        consumption: &H3PrivateAttemptConsumptionBinding,
        scheduler_device_peak_bytes: u64,
        scheduler_host_increment_bytes: u64,
    ) -> H3PrivateFl2VaAttemptFacts {
        H3PrivateFl2VaAttemptFacts {
            device_id: self.device_id.clone(),
            device_ordinal: self.device_ordinal,
            execution_fingerprint: self.execution_fingerprint.clone(),
            prepared_attempt_identity_sha256: self.prepared_attempt_identity_sha256.clone(),
            target_budget_identity_sha256: self.target_budget_identity_sha256.clone(),
            component_set_identity_sha256: self.component_set_identity_sha256.clone(),
            predicted_device_peak_bytes: scheduler_device_peak_bytes,
            predicted_host_increment_bytes: scheduler_host_increment_bytes,
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
    compute_capability: Option<(u16, u16)>,
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

    pub const fn compute_capability(&self) -> Option<(u16, u16)> {
        self.compute_capability
    }

    pub(crate) fn bounds(&self) -> &H3PrivateQualifiedRuntimeBounds {
        &self.bounds
    }

    #[cfg(feature = "mp4")]
    #[cfg(feature = "mp4")]
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
            .validate_prepared_with_adapter(request, turbo)
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
    /// Provisional, non-qualifying bounds that exist only to admit an
    /// instrumented campaign run so it can measure the real ones. Constructible
    /// only under the developer-only campaign feature; release verification
    /// rejects that feature's marker, which is the backstop that keeps this
    /// out of every shipping build.
    #[cfg(feature = "h3-private-uat")]
    CaptureCompiled,
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
            #[cfg(feature = "h3-private-uat")]
            Self::CaptureCompiled => validate_capture_runtime_profile(record, record_file_sha256),
        }
    }
}

#[cfg(feature = "h3")]
const PUBLIC_RUNTIME_PROFILE_SCHEMA: &str = "mold.minimax-h3.public-runtime-profile.v1";
#[cfg(feature = "h3")]
const PUBLIC_RUNTIME_PROFILE_DECISION: &str = "supported-compact-fl2va-cuda-sm89-or-metal";

/// Margin applied to every measured workspace bound, then rounded up to
/// [`PUBLIC_RUNTIME_BOUND_GRID_BYTES`].
///
/// One render is one sample. 15% covers allocator and driver variance plus the
/// small shape headroom the envelope still allows — the #1245 re-measurement
/// used 40,751 of the envelope's 40,766 packed rows and 2,033 of its 2,048
/// text rows, so it sat essentially at the ceiling and the margin does not
/// have to absorb a much larger shape. (The original #827 render sat the same
/// way against the old, smaller envelope.)
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
/// Every workspace figure is `public_runtime_bound(observed)` over a real
/// FL2VA render, captured by `private_runtime_observer`. The observations are
/// named beside each value so a re-measurement is applied by changing one
/// number and re-running the same policy — which is exactly what #1245 did.
///
/// Two campaigns contribute, and which one owns a bound follows from what the
/// bound depends on:
///
/// * #827, 2026-08-19 on hal9000 (RTX 4090, SM89, 24 GB), 1344x768, 124 frames
///   at 24 fps, 21 steps, 1216 s — the original qualification.
/// * #1245, 2026-08-21 on the same host, same canvas, 9 steps (`-turbo-8step`)
///   at the raised prompt budget: a 1,017-token prompt packing 2,033 text rows
///   and 40,751 of the envelope's 40,766 packed rows, 977 s. It re-measures
///   the three bounds the Qwen sequence and the packed sequence actually
///   move — attention, FFN, and the Qwen activation workspace — and
///   REPRODUCED `condition_vae`, `decoder_tile`, and `audio_decode` byte for
///   byte, which is the evidence that the two campaigns are comparable.
///
/// `fixed_runtime_device_bytes` deliberately keeps #827's observation. That
/// term is sampled as `global_total - global_free` at attempt entry, so on a
/// card shared with another process it measures the CO-TENANT, not Mold: the
/// #1245 host had ~1.5 GB of someone else's VRAM resident and reported
/// 2,075,197,440. It is not a sequence-dependent quantity, and re-pinning it
/// from a contaminated sample would inflate the admission device floor on
/// every host forever. `fixed_runtime_host_bytes` keeps #827's for the
/// opposite reason: #1245 observed 558,260,224, below it, so the existing
/// bound already covers it.
///
/// The policy reproduces four of the previous L40S-era caps exactly
/// (`fixed_runtime_device`, `condition_vae`, `decoder_tile`, `audio_decode`),
/// which is the reason to trust it, and corrects the two that were guesses:
/// attention fell 10.13 -> 7.31 GB and FFN 15.30 -> 9.06 GB. Two rose, because
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
        // observed 4_168_069_120 (#1245 re-measurement; 3_400_171_520 at the
        // old 1,058-row text ceiling — this is the one bound the raised
        // prompt budget moves materially, +22.6%)
        qwen_activation_workspace_bytes: public_runtime_bound(4_168_069_120),
        // Observed 0: the VAE construction transient never rose above the
        // weights themselves in the qualifying render. A zero bound is not
        // admissible (the reload stages through it and the validator refuses
        // zero), so the previous 64 MiB allowance is retained as a floor.
        vae_construction_device_workspace_bytes: 67_108_864,
        // observed 366_027_840
        condition_vae_workspace_device_bytes: public_runtime_bound(366_027_840),
        // observed 6_323_525_308 (#1245 re-measurement; 6_172_029_280 at the
        // old envelope — the denoise transients are linear in packed rows and
        // the sequence grew 2.5%)
        attention_workspace_device_bytes: public_runtime_bound(6_323_525_308),
        // observed 7_826_714_044 (#1245 re-measurement; 7_641_748_832 at the
        // old envelope)
        ffn_workspace_device_bytes: public_runtime_bound(7_826_714_044),
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

// ---------------------------------------------------------------------------
// The reviewed compact FL2VA row ceilings.
//
// Every one of these is a property of the reviewed canvas except the text
// ceiling, which is a reviewed BUDGET: the conditioner sequence is
// `"<Picture 1>: "` + the boundary endpoint's merged vision pads + the
// prompt's own tokens (`build_fl2va_presentation`), so whatever the ceiling
// leaves above that fixed presentation overhead is exactly the prompt a user
// may write. The original 1,058 was transcribed from the qualifying render's
// own 1,050-row observation, which left room for about forty prompt tokens and
// refused everything the apps actually send (#1245). The conditioner's own
// context is `QWEN_MAXIMUM_TOKENS` = 262,144, so the model was never the
// limit — the capture was.
// ---------------------------------------------------------------------------

/// One boundary endpoint's merged vision pads, the fixed part of the FL2VA
/// presentation overhead.
const REVIEWED_FL2VA_VISION_PAD_ROWS: u64 = 1_008;

/// The reviewed conditioner text ceiling: the vision pads above, the
/// `"<Picture 1>: "` label, and roughly a thousand prompt tokens on top — a
/// prompt long enough for the paragraph-scale descriptions the Studio surfaces
/// compose, and still two orders of magnitude inside the conditioner's own
/// 262,144-token context. The exact prompt budget is never hard-coded: the
/// admission precheck derives it as this ceiling minus the presentation
/// overhead the tokenizer actually produced, so a template change cannot make
/// the reported budget a lie.
const REVIEWED_MAX_QWEN_OUTPUT_TEXT_ROWS: u64 = 2_048;

/// Pre-merge vision patches for the same endpoint (4 x the merged pads).
const REVIEWED_MAX_QWEN_VISION_ROWS: u64 = 4 * REVIEWED_FL2VA_VISION_PAD_ROWS;

/// The boundary endpoint's conditioning latent rows.
const REVIEWED_MAX_CONDITION_VISUAL_ROWS: u64 = REVIEWED_FL2VA_VISION_PAD_ROWS;

/// 1344x768 x 124 frames of generated video latents.
const REVIEWED_MAX_TARGET_VIDEO_ROWS: u64 = 37_296;

/// The same duration of generated audio latents.
const REVIEWED_MAX_TARGET_AUDIO_ROWS: u64 = 414;

/// The packed sequence is exactly the four axes the FL2VA prepared request
/// sums (`prepared_request_input`), so it is derived rather than transcribed:
/// raising the text ceiling raises this by the same amount and nothing else
/// moves.
const REVIEWED_MAX_TOTAL_PACKED_ROWS: u64 = REVIEWED_MAX_QWEN_OUTPUT_TEXT_ROWS
    + REVIEWED_MAX_CONDITION_VISUAL_ROWS
    + REVIEWED_MAX_TARGET_VIDEO_ROWS
    + REVIEWED_MAX_TARGET_AUDIO_ROWS;

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
        frames: contract::REVIEWED_COMPACT_FRAMES,
        fps: contract::FIXED_FPS,
        batch_size: 1,
        max_steps,
        endpoint_count: 1,
        endpoint_anchor: "first".into(),
        max_qwen_output_text_rows: REVIEWED_MAX_QWEN_OUTPUT_TEXT_ROWS,
        max_qwen_vision_rows: REVIEWED_MAX_QWEN_VISION_ROWS,
        max_condition_visual_rows: REVIEWED_MAX_CONDITION_VISUAL_ROWS,
        max_target_video_rows: REVIEWED_MAX_TARGET_VIDEO_ROWS,
        max_target_audio_rows: REVIEWED_MAX_TARGET_AUDIO_ROWS,
        max_total_packed_rows: REVIEWED_MAX_TOTAL_PACKED_ROWS,
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
    // The record's own task is pinned to fl2va a few lines below, which is
    // what scopes this step count.
    record
        .envelope
        .validate_for_task_with_reviewed_steps(Task::Fl2va, turbo_steps)?;
    record.bounds.validate()?;
    if record.schema != PUBLIC_RUNTIME_PROFILE_SCHEMA
        || record.decision != PUBLIC_RUNTIME_PROFILE_DECISION
        || record.canonical_model != contract::FL2VA_COMFY
        || record.task != "fl2va"
        || !matches!(record.compute_capability, [8, 9] | [0, 0])
        || record.identity_sha256 != runtime_qualification_identity(record)
        || profile_sha256 != record.identity_sha256
    {
        bail!("public H3 runtime profile changed or is not the supported CUDA/Metal profile")
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Capture-scope Ref2VA profile.
//
// THESE BOUNDS ARE CEILINGS, NOT MEASUREMENTS. They exist for one purpose: an
// instrumented Ref2VA run must be admitted before it can observe anything, and
// admission's grant comes from a bounds record. Nothing here has been measured
// on hardware, and nothing here may ever be transcribed into a public profile.
// The reviewed record that eventually authorizes Ref2VA must transcribe the
// campaign's OBSERVED values, which the capture report prints beside these
// ceilings precisely so the two cannot be confused.
// ---------------------------------------------------------------------------
#[cfg(feature = "h3-private-uat")]
const CAPTURE_RUNTIME_PROFILE_SCHEMA: &str = "mold.minimax-h3.capture-runtime-profile.v1";
#[cfg(feature = "h3-private-uat")]
const CAPTURE_RUNTIME_PROFILE_DECISION: &str =
    "capture-scope-compact-ref2va-sm89-provisional-bounds";

/// Ref2VA's compact campaign envelope.
///
/// Target geometry is FL2VA's — same canvas, duration, and step count — so the
/// generated-side caps are identical and directly comparable. Conditioning is
/// what differs, and its caps come from the released reference limits rather
/// than from a measurement: at most [`contract::MAX_REFERENCE_FILES`] ordered
/// references, each normalized onto its own canvas. The text cap stays at
/// FL2VA's pre-#1245 value; see [`public_style_generated_caps`].
#[cfg(feature = "h3-private-uat")]
fn capture_runtime_envelope() -> H3PrivateRuntimeEnvelopeRecord {
    let public = public_style_generated_caps();
    // One 2048-square still is 4,096 rows and 4,096 vision pads — the largest
    // single reference the contract admits. Twelve of them is the ceiling the
    // reference count itself imposes.
    let max_reference_rows = 4_096 * contract::MAX_REFERENCE_FILES as u64;
    H3PrivateRuntimeEnvelopeRecord {
        width: contract::DEFAULT_WIDTH,
        height: contract::DEFAULT_HEIGHT,
        frames: contract::REVIEWED_COMPACT_FRAMES,
        fps: contract::FIXED_FPS,
        batch_size: 1,
        max_steps: contract::COMFY_DEFAULT_STEPS,
        // Ref2VA carries no boundary endpoint at all.
        endpoint_count: 0,
        endpoint_anchor: "none".into(),
        max_qwen_output_text_rows: public.0,
        max_qwen_vision_rows: max_reference_rows,
        max_condition_visual_rows: max_reference_rows,
        max_target_video_rows: public.1,
        max_target_audio_rows: public.2,
        max_total_packed_rows: public.0 + max_reference_rows + public.2 + public.1 + public.2,
    }
}

/// The generated-side caps the capture envelope prices against:
/// `(qwen_output_text_rows, target_video_rows, target_audio_rows)`.
///
/// The two generated-side rows are FL2VA's own — same canvas, same duration.
/// The text cap deliberately is NOT: it stays at the pre-#1245 1,058 rows
/// because every provisional ceiling in `capture_runtime_bounds` is derived
/// from this envelope's packed-row count, and re-deriving them at FL2VA's
/// raised prompt budget would move campaign-scope ceilings on the strength of
/// an FL2VA measurement. Ref2VA's own campaign owns that decision; until it
/// runs, its prompt budget is unchanged.
#[cfg(feature = "h3-private-uat")]
const fn public_style_generated_caps() -> (u64, u64, u64) {
    (
        1_058,
        REVIEWED_MAX_TARGET_VIDEO_ROWS,
        REVIEWED_MAX_TARGET_AUDIO_ROWS,
    )
}

/// FL2VA's observed workspace measurements — the identical figures
/// `public_runtime_bounds` applies its margin policy to. The two denoise
/// transients and their envelope come from the #1245 re-measurement at the
/// raised prompt budget (2026-08-21 on hal9000, RTX 4090 SM89, 1344x768, 124
/// frames, 9 steps); the condition-VAE transient is #827's, which #1245
/// reproduced byte for byte. Restated here because `public_runtime_bounds` is
/// compiled only under `h3` while the capture build runs without it; a
/// divergence is a review error.
///
/// The capture-scope ceilings these feed scale the observation by the ratio of
/// the two envelopes' packed rows, and both halves of that ratio moved by the
/// same ~2.5%, so the derived Ref2VA ceilings are unchanged.
#[cfg(feature = "h3-private-uat")]
mod fl2va_observed {
    pub(super) const ATTENTION_WORKSPACE_DEVICE_BYTES: u64 = 6_323_525_308;
    pub(super) const FFN_WORKSPACE_DEVICE_BYTES: u64 = 7_826_714_044;
    pub(super) const CONDITION_VAE_WORKSPACE_DEVICE_BYTES: u64 = 366_027_840;
    /// The packed-row count of the reviewed FL2VA envelope the two denoise
    /// transients above were measured under (`max_total_packed_rows`,
    /// `public_runtime_envelope_for_steps`); the qualifying render packed the
    /// envelope's own maximum sequence, so the observation IS the envelope's.
    pub(super) const ENVELOPE_TOTAL_PACKED_ROWS: u64 = super::REVIEWED_MAX_TOTAL_PACKED_ROWS;
    /// The conditioning canvas the FL2VA condition-VAE transient was
    /// measured encoding (one 512x384 boundary frame).
    pub(super) const CONDITION_ENCODE_PIXELS: u64 = 512 * 384;
}

/// The corrected public margin/rounding policy (`public_runtime_bound`),
/// restated for the capture build: round a derived ceiling up to the 64 MiB
/// grant grid. The capture derivation deliberately applies NO percentage
/// margin on top — its headroom is structural (see each term below), and a
/// percentage would push the denoise grant past what the 24 GiB campaign
/// card can hold beside the transformer itself.
#[cfg(feature = "h3-private-uat")]
const fn capture_grid_ceiling(bytes: u64) -> u64 {
    bytes.next_multiple_of(64 * 1024 * 1024)
}

/// Provisional device/host ceilings for one instrumented Ref2VA attempt,
/// derived term by term from the CORRECTED measurement-grounded FL2VA ledger
/// at the capture envelope. The pre-correction sizing (attention 10.13 GB x5,
/// FFN 15.30 GB x3) put the admission device floor at 51.3 GB — over double
/// the SM89 campaign card — so the campaign could never admit on the very
/// device it exists to measure.
///
/// Per-term provenance:
///
/// * `attention`/`ffn` — the denoise transients are per-forward workspaces
///   over the packed sequence, and the admission device floor is
///   `fixed_runtime_device + max(attention, ffn)`
///   (`denoise_transient_workspace_device_bytes`). Both scale LINEARLY with
///   packed rows: the route is FlashAttention v2 (no materialized score
///   matrix — the old x5 "quadratic" argument sized a matrix the kernel
///   never allocates) and the FFN materializes per-row projections. Ceiling =
///   observed x (capture envelope rows / FL2VA envelope rows) =
///   x 88,334/40,766, grid-rounded: attention 13.76 GB, FFN 16.98 GB —
///   unchanged by #1245, which grew observation and envelope by the same
///   2.5%. The
///   provisional headroom is structural: the envelope prices twelve
///   2048-square references while the instrumented campaign request packs a
///   small ordered set (~43k rows, barely above FL2VA's 40.8k), so the grant
///   sits ~2x above the expected observation. Device floor:
///   0.60 + 16.98 = 17.58 GB — clears the campaign card's 24.97 GB sample
///   with the transformer's own denoise-phase terms still fitting beside it.
/// * `condition_vae` — a per-encode transient, linear in the encoded canvas:
///   observed x (largest reference canvas / FL2VA's measured condition
///   canvas) = x 2048^2/(512x384) = x64/3, grid-rounded: 7.85 GB. Charged in
///   the reference visual-encode phase, far below the denoise peak.
/// * `qwen_activation` — 2x the public ceiling (2 x 4.83 GB = 9.66 GB; it
///   tracked 2 x 3.96 GB before #1245 re-measured the FL2VA observation at
///   the raised prompt budget, and the relationship is what is pinned, not
///   the number). This is the provisional GRANT ceiling only, not the charge:
///   the exact budget charges each request's own derived demand — the
///   observed per-row cost scaled by the request's text+vision rows
///   (`qwen_activation_workspace_demand_bytes`) — and a Ref2VA sequence whose
///   demand exceeds this grant is a named refusal at budget build, never an
///   undercharged admit. The 2x sizing covers the campaign's ordered sets
///   (~2x FL2VA's 6,065-row measured sequence); the envelope's twelve-still
///   maximum (50,210 Qwen rows) is
///   deliberately NOT the sizing point — granting it would exceed the
///   campaign host headroom beside the 19.1 GB CPU-placed conditioner.
/// * `fixed_runtime_host` / `fixed_runtime_device` /
///   `decoder_tile` / `audio_decode` — fixed runtime state and
///   generated-side output whose geometry the capture envelope shares with
///   FL2VA exactly; each keeps the corrected public ceiling
///   (`public_runtime_bound(observed)`): 805,306,368 / 603,979,776 /
///   1,543,503,872 / 268,435,456.
/// * `vae_construction` — the public profile's own 64 MiB floor for a
///   transient observed at zero (a zero bound is inadmissible).
/// * The four pipeline host bounds remain the pipeline's own allocation
///   limits, as everywhere else.
///
/// The audio reference encoder still has no FL2VA analogue; the budget
/// builder borrows `audio_decode_workspace_device_bytes` for it, which rides
/// the corrected value above.
#[cfg(feature = "h3-private-uat")]
fn capture_runtime_bounds() -> H3PrivateRuntimeBoundRecord {
    let envelope_rows = capture_runtime_envelope().max_total_packed_rows;
    let sequence_scaled = |observed_bytes: u64| {
        capture_grid_ceiling(
            observed_bytes * envelope_rows / fl2va_observed::ENVELOPE_TOTAL_PACKED_ROWS,
        )
    };
    let largest_reference_pixels = 2_048_u64 * 2_048;
    H3PrivateRuntimeBoundRecord {
        fixed_runtime_host_bytes: 805_306_368,
        fixed_runtime_device_bytes: 603_979_776,
        // 2x the FL2VA public ceiling, restated here for the same reason
        // `fl2va_observed` is: `public_runtime_bounds` is `h3`-only.
        qwen_activation_workspace_bytes: 2 * 4_831_838_208,
        vae_construction_device_workspace_bytes: 67_108_864,
        condition_vae_workspace_device_bytes: capture_grid_ceiling(
            fl2va_observed::CONDITION_VAE_WORKSPACE_DEVICE_BYTES * largest_reference_pixels
                / fl2va_observed::CONDITION_ENCODE_PIXELS,
        ),
        attention_workspace_device_bytes: sequence_scaled(
            fl2va_observed::ATTENTION_WORKSPACE_DEVICE_BYTES,
        ),
        ffn_workspace_device_bytes: sequence_scaled(fl2va_observed::FFN_WORKSPACE_DEVICE_BYTES),
        decoder_tile_workspace_device_bytes: 1_543_503_872,
        audio_decode_workspace_device_bytes: 268_435_456,
        encoded_video_host_bytes_bound: super::pipeline::SMALL_ENCODED_VIDEO_HOST_BYTES_BOUND,
        thumbnail_host_bytes_bound: super::pipeline::SMALL_THUMBNAIL_HOST_BYTES_BOUND,
        mux_output_host_bytes_bound: super::pipeline::SMALL_MUX_OUTPUT_HOST_BYTES_BOUND,
        aac_mux_staging_host_bytes: super::pipeline::SMALL_AAC_MUX_STAGING_HOST_BYTES,
    }
}

#[cfg(feature = "h3-private-uat")]
fn validate_capture_runtime_profile(
    record: &H3PrivateRuntimeQualificationRecord,
    profile_sha256: &str,
) -> Result<()> {
    record.envelope.validate_for_task(Task::Ref2va)?;
    record.bounds.validate()?;
    if record.schema != CAPTURE_RUNTIME_PROFILE_SCHEMA
        || record.decision != CAPTURE_RUNTIME_PROFILE_DECISION
        || record.canonical_model != contract::REF2VA_COMFY
        || record.task != "ref2va"
        || record.compute_capability != [8, 9]
        || record.identity_sha256 != runtime_qualification_identity(record)
        || profile_sha256 != record.identity_sha256
    {
        bail!("capture-scope H3 profile changed or is not the Ref2VA SM89 campaign profile")
    }
    Ok(())
}

/// Build the capture-scope authority for one instrumented Ref2VA attempt.
///
/// This deliberately mirrors `public_runtime_qualification`'s shape so the two
/// are directly comparable in review, and differs from it in exactly the ways
/// that matter: a different schema, a decision string that says the bounds are
/// provisional, the Ref2VA task and model, and a storage variant that cannot
/// exist in a shipping build.
#[cfg(feature = "h3-private-uat")]
#[allow(clippy::too_many_arguments)]
fn capture_runtime_qualification(
    artifact: &H3PrivateArtifactQualificationReport,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: (u16, u16),
    attention_runtime_identity_sha256: &str,
    attention_kernel_identity: &str,
    attention_qualification_sha256: &str,
) -> Result<H3PrivateRuntimeQualificationAuthority> {
    if compute_capability != (8, 9)
        || artifact.canonical_model != contract::REF2VA_COMFY
        || artifact.task != "ref2va"
        || !valid_sha256(attention_runtime_identity_sha256)
        || !valid_sha256(attention_qualification_sha256)
        || attention_kernel_identity.is_empty()
    {
        bail!("capture-scope H3 runtime requires the exact compact Ref2VA SM89 route")
    }
    let mut record = H3PrivateRuntimeQualificationRecord {
        schema: CAPTURE_RUNTIME_PROFILE_SCHEMA.into(),
        decision: CAPTURE_RUNTIME_PROFILE_DECISION.into(),
        canonical_model: contract::REF2VA_COMFY.into(),
        task: "ref2va".into(),
        campaign_source_sha: exact_h3_runtime_build_source_sha()?.into(),
        campaign_runtime_code_identity_sha256: super::PRIVATE_RUNTIME_CODE_IDENTITY_SHA256.into(),
        campaign_bootstrap_record_sha256: sha256_domain("capture-profile-bootstrap"),
        campaign_bootstrap_identity_sha256: sha256_domain("capture-profile-identity"),
        measured_server_executable_relative_path: "capture-runtime-profile".into(),
        measured_server_executable_sha256: sha256_domain("capture-runtime-executable"),
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
            linux_boot_id_sha256: sha256_domain("capture-runtime-boot"),
            executable_device: 0,
            executable_inode: 0,
            executable_bytes: 0,
            executable_sha256: sha256_domain("capture-runtime-executable"),
            launch_argv_sha256: sha256_domain("capture-runtime-argv"),
            launch_environment_sha256: sha256_domain("capture-runtime-environment"),
            cuda_driver_version: 0,
            cuda_toolkit_version: 0,
        },
        envelope: capture_runtime_envelope(),
        bounds: capture_runtime_bounds(),
        evidence_artifacts: Vec::new(),
        identity_sha256: String::new(),
    };
    record.identity_sha256 = runtime_qualification_identity(&record);
    let profile_sha256 = record.identity_sha256.clone();
    validate_capture_runtime_profile(&record, &profile_sha256)?;
    let bounds = record.bounds.clone().into_authority();
    Ok(H3PrivateRuntimeQualificationAuthority {
        storage: RuntimeQualificationStorage::CaptureCompiled,
        record_file_sha256: profile_sha256,
        record,
        bounds,
        device_id: device_id.into(),
        device_ordinal,
        compute_capability: Some(compute_capability),
    })
}

/// Render one campaign bound as `observed / ceiling`, so the reviewer reading
/// the capture transcribes the observation and never the ceiling.
#[cfg(feature = "h3-private-uat")]
pub fn h3_capture_bound_report(
    observed: &[(&'static str, u64)],
) -> Vec<(&'static str, u64, u64, f64)> {
    let ceilings = capture_runtime_bounds();
    let ceiling_for = |label: &str| match label {
        "fixed_runtime_host_bytes" => ceilings.fixed_runtime_host_bytes,
        "fixed_runtime_device_bytes" => ceilings.fixed_runtime_device_bytes,
        "qwen_activation_workspace_bytes" => ceilings.qwen_activation_workspace_bytes,
        "vae_construction_device_workspace_bytes" => {
            ceilings.vae_construction_device_workspace_bytes
        }
        "condition_vae_workspace_device_bytes" => ceilings.condition_vae_workspace_device_bytes,
        "attention_workspace_device_bytes" => ceilings.attention_workspace_device_bytes,
        "ffn_workspace_device_bytes" => ceilings.ffn_workspace_device_bytes,
        "decoder_tile_workspace_device_bytes" => ceilings.decoder_tile_workspace_device_bytes,
        "audio_decode_workspace_device_bytes" => ceilings.audio_decode_workspace_device_bytes,
        "encoded_video_host_bytes_bound" => ceilings.encoded_video_host_bytes_bound,
        "thumbnail_host_bytes_bound" => ceilings.thumbnail_host_bytes_bound,
        "mux_output_host_bytes_bound" => ceilings.mux_output_host_bytes_bound,
        "aac_mux_staging_host_bytes" => ceilings.aac_mux_staging_host_bytes,
        _ => 0,
    };
    observed
        .iter()
        .map(|(label, observed)| {
            let ceiling = ceiling_for(label);
            let headroom = if ceiling == 0 {
                0.0
            } else {
                *observed as f64 / ceiling as f64
            };
            (*label, *observed, ceiling, headroom)
        })
        .collect()
}

#[cfg(feature = "h3")]
#[allow(clippy::too_many_arguments)]
fn public_runtime_qualification(
    artifact: &H3PrivateArtifactQualificationReport,
    device_id: &str,
    device_ordinal: usize,
    compute_capability: Option<(u16, u16)>,
    attention_runtime_identity_sha256: &str,
    attention_kernel_identity: &str,
    attention_qualification_sha256: &str,
    turbo: Option<&H3FactoryTurboAdapterAuthority>,
) -> Result<H3PrivateRuntimeQualificationAuthority> {
    // This mints an FL2VA-shaped qualification, so a tier reviewed for another
    // task may not set its step count — the FL2V 768p and Ref2V tiers share a
    // 5-point schedule, so a bare count cannot tell them apart and the task
    // identity would be stripped across this boundary.
    if let Some(turbo) = turbo {
        if turbo.reviewed_task() != Some(Task::Fl2va) {
            bail!(
                "private H3 Turbo adapter {} was not reviewed for the fl2va runtime qualification",
                turbo.tier_stable_id()
            )
        }
    }
    let turbo_steps = turbo.map(H3FactoryTurboAdapterAuthority::grid_points);
    if !matches!(compute_capability, Some((8, 9)) | None)
        || artifact.canonical_model != contract::FL2VA_COMFY
        || artifact.task != "fl2va"
        || !valid_sha256(attention_runtime_identity_sha256)
        || !valid_sha256(attention_qualification_sha256)
        || attention_kernel_identity.is_empty()
    {
        bail!("public H3 runtime requires the exact compact FL2VA CUDA SM89 or Metal authority")
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
        compute_capability: compute_capability
            .map(|(major, minor)| [major, minor])
            .unwrap_or([0, 0]),
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

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
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
            compute_capability: Some(compute_capability),
        };
        authority.revalidate()?;
        Ok(authority)
    }
}

/// The runtime-qualification source for one task under the private-record
/// build (`not(feature = "h3")`), resolved BEFORE bulk model I/O.
///
/// FL2VA keeps its reviewed-record gate exactly as before. Ref2VA has no
/// reviewed record at all: its only authority is the compiled capture-scope
/// profile with provisional bounds, which exists precisely so the
/// instrumented campaign run can be admitted and measure the real ones — so
/// the Ref2VA arm never touches the record file. That arm is constructible
/// only under the developer-only `h3-private-uat` feature; every other build
/// keeps refusing the task here, and public `h3` builds never reach this type
/// (their Ref2VA refusal lives in `public_runtime_qualification`).
#[cfg(not(feature = "h3"))]
enum H3PrivateRuntimeQualificationSource {
    // Boxed: the opened record dwarfs the unit capture arm
    // (clippy::large_enum_variant).
    ReviewedRecord(Box<H3PrivateReviewedRuntimeQualification>),
    #[cfg(feature = "h3-private-uat")]
    CaptureCompiled,
}

#[cfg(not(feature = "h3"))]
fn private_runtime_qualification_source(
    task: Task,
    record_path: &Path,
) -> Result<H3PrivateRuntimeQualificationSource> {
    match task {
        Task::Fl2va => Ok(H3PrivateRuntimeQualificationSource::ReviewedRecord(
            Box::new(open_reviewed_h3_private_runtime_qualification(
                record_path,
                REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256,
            )?),
        )),
        #[cfg(feature = "h3-private-uat")]
        Task::Ref2va => Ok(H3PrivateRuntimeQualificationSource::CaptureCompiled),
        #[cfg(not(feature = "h3-private-uat"))]
        Task::Ref2va => bail!(
            "private H3 Ref2VA has no reviewed runtime qualification outside the campaign build"
        ),
    }
}

#[cfg(not(feature = "h3"))]
impl H3PrivateRuntimeQualificationSource {
    /// Reject a route this source cannot authorize before the caller starts
    /// the multi-artifact qualification pass. The capture arm mirrors the pin
    /// `capture_runtime_qualification` enforces at minting, so a wrong-arch
    /// device refuses cheaply instead of after hashing ~37 GB.
    fn validate_route(
        &self,
        device_id: &str,
        device_ordinal: usize,
        compute_capability: (u16, u16),
    ) -> Result<()> {
        match self {
            Self::ReviewedRecord(reviewed) => {
                reviewed.validate_route(device_id, device_ordinal, compute_capability)
            }
            #[cfg(feature = "h3-private-uat")]
            Self::CaptureCompiled => {
                let _ = device_ordinal;
                if device_id.trim().is_empty() || compute_capability != (8, 9) {
                    bail!("capture-scope H3 runtime requires the exact compact Ref2VA SM89 route")
                }
                Ok(())
            }
        }
    }

    /// The bounds the cheap admission capacity floors are derived from.
    fn precheck_bounds(&self) -> H3PrivateRuntimeBoundRecord {
        match self {
            Self::ReviewedRecord(reviewed) => reviewed.record.bounds.clone(),
            #[cfg(feature = "h3-private-uat")]
            Self::CaptureCompiled => capture_runtime_bounds(),
        }
    }

    /// The envelope the cheap prepared-row precheck is derived from. It is the
    /// same record `authenticate` later vouches for, read before the artifact
    /// pass so an over-budget prompt never pays for one.
    #[cfg(feature = "mp4")]
    fn precheck_envelope(&self) -> H3PrivateRuntimeEnvelopeRecord {
        match self {
            Self::ReviewedRecord(reviewed) => reviewed.record.envelope.clone(),
            #[cfg(feature = "h3-private-uat")]
            Self::CaptureCompiled => capture_runtime_envelope(),
        }
    }

    /// The attention qualification identity this source vouches for: the
    /// reviewed record carries its campaign's own value, while the compiled
    /// capture profile — like the compiled public profile — vouches for the
    /// live qualified attention route itself.
    fn attention_qualification_sha256(&self, attention: &H3AttentionRuntimeAuthority) -> String {
        match self {
            Self::ReviewedRecord(reviewed) => {
                reviewed.record.attention_qualification_sha256.clone()
            }
            #[cfg(feature = "h3-private-uat")]
            Self::CaptureCompiled => attention.identity_sha256().to_string(),
        }
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
        match self {
            Self::ReviewedRecord(reviewed) => reviewed.authenticate(
                artifact_qualification,
                device_id,
                device_ordinal,
                compute_capability,
                attention_runtime_identity_sha256,
                attention_kernel_identity,
                attention_qualification_sha256,
            ),
            #[cfg(feature = "h3-private-uat")]
            Self::CaptureCompiled => capture_runtime_qualification(
                artifact_qualification,
                device_id,
                device_ordinal,
                compute_capability,
                attention_runtime_identity_sha256,
                attention_kernel_identity,
                attention_qualification_sha256,
            ),
        }
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

/// Authenticate the compiled public H3 profile against one live SM89 CUDA or
/// Metal route.
/// route. This presentation check opens neither model weights nor external
/// compliance files; generation separately verifies every artifact and applies
/// the compiled conservative memory profile.
#[cfg(feature = "h3")]
pub fn authenticate_h3_public_presentation(
    routes: &[H3PrivatePresentationRoute<'_>],
) -> Result<H3PrivatePresentationAuthority> {
    if routes.is_empty() {
        bail!("MiniMax H3 presentation requires one live schedulable CUDA or Metal route")
    }
    let route = routes
        .iter()
        .find(|route| matches!(route.compute_capability, Some((8, 9)) | None))
        .ok_or_else(|| {
            anyhow!("public H3 requires one live schedulable SM89 CUDA or Metal route")
        })?;
    match route.compute_capability {
        Some(compute_capability) => H3AttentionRuntimeAuthority::qualify_flash_attention_v2(
            H3AttentionDevice::Cuda {
                compute_capability: Some(compute_capability),
            },
            H3AttentionModelContract::released_bf16(),
        ),
        None => H3AttentionRuntimeAuthority::metal_chunked_dense(
            H3AttentionModelContract::released_bf16(),
        ),
    }
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
                && route.compute_capability.is_some_and(|(major, minor)| {
                    reviewed.record.compute_capability == [major, minor]
                })
        })
        .ok_or_else(|| {
            anyhow!("private H3 presentation has no live route matching reviewed evidence")
        })?;
    let route_compute_capability = route
        .compute_capability
        .ok_or_else(|| anyhow!("private H3 reviewed presentation requires a CUDA route"))?;
    reviewed.validate_route(
        route.device_id,
        route.device_ordinal,
        route_compute_capability,
    )?;
    let attention;
    let (attention_runtime_identity_sha256, attention_kernel_identity) =
        if let Some(identity) = test_attention_identity {
            identity
        } else {
            attention = H3AttentionRuntimeAuthority::qualify_flash_attention_v2(
                H3AttentionDevice::Cuda {
                    compute_capability: Some(route_compute_capability),
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

    /// Build the reviewed adapter authority for one tier, so envelope tests
    /// exercise the same value admission passes rather than a bare step count.
    ///
    /// Ungated like its callers: every piece it touches — the authority type,
    /// its constructor, and the reviewed tier table — is feature-independent.
    fn turbo_authority_for(
        tier: mold_candle::minimax_h3::H3TurboLoraTier,
    ) -> H3FactoryTurboAdapterAuthority {
        H3FactoryTurboAdapterAuthority::for_reviewed_tier(
            tier.stable_id(),
            &sha('7'),
            &sha('8'),
            4_096,
            2_048,
            1_024,
        )
        .expect("reviewed tier must build an adapter authority")
    }

    fn reviewed_envelope(max_steps: u32) -> H3PrivateRuntimeEnvelopeRecord {
        H3PrivateRuntimeEnvelopeRecord {
            width: contract::DEFAULT_WIDTH,
            height: contract::DEFAULT_HEIGHT,
            frames: contract::REVIEWED_COMPACT_FRAMES,
            fps: contract::FIXED_FPS,
            batch_size: 1,
            max_steps,
            endpoint_count: 1,
            endpoint_anchor: "first".into(),
            max_qwen_output_text_rows: REVIEWED_MAX_QWEN_OUTPUT_TEXT_ROWS,
            max_qwen_vision_rows: REVIEWED_MAX_QWEN_VISION_ROWS,
            max_condition_visual_rows: REVIEWED_MAX_CONDITION_VISUAL_ROWS,
            max_target_video_rows: REVIEWED_MAX_TARGET_VIDEO_ROWS,
            max_target_audio_rows: REVIEWED_MAX_TARGET_AUDIO_ROWS,
            max_total_packed_rows: REVIEWED_MAX_TOTAL_PACKED_ROWS,
        }
    }

    /// The reviewed rows a maximum-length FL2VA request packs.
    #[cfg(feature = "mp4")]
    fn reviewed_rows(qwen_output_text_rows: u64) -> H3FactoryPreparedRowsInput {
        H3FactoryPreparedRowsInput {
            qwen_output_text_rows,
            qwen_vision_rows: REVIEWED_MAX_QWEN_VISION_ROWS,
            condition_visual_rows: REVIEWED_MAX_CONDITION_VISUAL_ROWS,
            condition_audio_rows: 0,
            target_video_rows: REVIEWED_MAX_TARGET_VIDEO_ROWS,
            target_audio_rows: REVIEWED_MAX_TARGET_AUDIO_ROWS,
            total_packed_rows: qwen_output_text_rows
                + REVIEWED_MAX_CONDITION_VISUAL_ROWS
                + REVIEWED_MAX_TARGET_VIDEO_ROWS
                + REVIEWED_MAX_TARGET_AUDIO_ROWS,
        }
    }

    /// The reviewed text ceiling is a PROMPT budget, and it has to be one a
    /// person can actually spend. The captured 1,058 left about forty tokens
    /// and refused ordinary app prompts after ninety seconds of hashing
    /// (#1245); the packed total is derived from it rather than transcribed,
    /// so the two can never disagree.
    #[test]
    fn the_reviewed_envelope_budgets_a_real_prompt_and_derives_its_packed_total() {
        let envelope = reviewed_envelope(contract::COMFY_DEFAULT_STEPS);
        // One `"<Picture 1>: "` label is a handful of tokens; whatever it is,
        // a thousand prompt tokens must survive beside the vision pads.
        let generous_label_allowance = 32;
        let prompt_budget = envelope.max_qwen_output_text_rows
            - REVIEWED_FL2VA_VISION_PAD_ROWS
            - generous_label_allowance;
        assert!(
            prompt_budget >= 1_000,
            "reviewed prompt budget is only {prompt_budget} tokens"
        );
        // Nothing but the text axis moved: the canvas rows are untouched and
        // the packed total is exactly their sum.
        assert_eq!(envelope.max_qwen_vision_rows, 4_032);
        assert_eq!(envelope.max_condition_visual_rows, 1_008);
        assert_eq!(envelope.max_target_video_rows, 37_296);
        assert_eq!(envelope.max_target_audio_rows, 414);
        assert_eq!(
            envelope.max_total_packed_rows,
            envelope.max_qwen_output_text_rows
                + envelope.max_condition_visual_rows
                + envelope.max_target_video_rows
                + envelope.max_target_audio_rows
        );
    }

    /// An over-budget prompt must be refused by NAME, with the budget it
    /// actually has, and the budget must be derived from the presentation the
    /// tokenizer produced rather than a second transcribed constant: the
    /// overhead here is 1,008 vision pads plus a ten-token label, so the
    /// reported budget moves with it.
    #[cfg(feature = "mp4")]
    #[test]
    fn an_over_budget_prompt_is_refused_by_name_with_its_own_budget() {
        let envelope = reviewed_envelope(contract::COMFY_DEFAULT_STEPS);
        let label_rows = 10;
        let overhead = REVIEWED_FL2VA_VISION_PAD_ROWS + label_rows;
        let expected_budget = envelope.max_qwen_output_text_rows - overhead;

        // Exactly at the budget: admitted.
        precheck_private_h3_prepared_rows(
            &envelope,
            &reviewed_rows(overhead + expected_budget),
            expected_budget,
        )
        .unwrap();

        // One token past it: refused, naming both numbers.
        let prompt_tokens = expected_budget + 1;
        let error = precheck_private_h3_prepared_rows(
            &envelope,
            &reviewed_rows(overhead + prompt_tokens),
            prompt_tokens,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(&format!("prompt is {prompt_tokens} tokens")),
            "{error}"
        );
        assert!(
            error.contains(&format!("room for {expected_budget} prompt tokens")),
            "{error}"
        );

        // A larger presentation overhead leaves a smaller budget, and the
        // message says so — nothing here is a constant.
        let wide_overhead = overhead + 500;
        let wide_budget = envelope.max_qwen_output_text_rows - wide_overhead;
        let wide = precheck_private_h3_prepared_rows(
            &envelope,
            &reviewed_rows(wide_overhead + wide_budget + 1),
            wide_budget + 1,
        )
        .unwrap_err()
        .to_string();
        assert!(
            wide.contains(&format!("room for {wide_budget} prompt tokens")),
            "{wide}"
        );
    }

    /// The precheck is only allowed to refuse what the authenticated envelope
    /// backstop would also refuse — that is what makes hoisting it before the
    /// artifact pass safe. Swept across the text axis, the two answers agree
    /// exactly.
    #[cfg(feature = "mp4")]
    #[test]
    fn the_prompt_precheck_and_the_envelope_backstop_agree_on_every_row_count() {
        let envelope = reviewed_envelope(contract::COMFY_DEFAULT_STEPS);
        for text_rows in [
            1,
            1_058,
            REVIEWED_MAX_QWEN_OUTPUT_TEXT_ROWS - 1,
            REVIEWED_MAX_QWEN_OUTPUT_TEXT_ROWS,
            REVIEWED_MAX_QWEN_OUTPUT_TEXT_ROWS + 1,
            REVIEWED_MAX_QWEN_OUTPUT_TEXT_ROWS * 4,
        ] {
            let rows = reviewed_rows(text_rows);
            let precheck = precheck_private_h3_prepared_rows(&envelope, &rows, 1).is_err();
            let backstop = !envelope.row_cap_mismatches(&rows).is_empty();
            assert_eq!(precheck, backstop, "text rows {text_rows}");
        }
    }

    /// The whole point of the precheck is WHERE it runs: a prompt refusal must
    /// not cost the ~37 GB artifact SHA-256 pass first. Ordering inside one
    /// function is not observable from a unit test, so this pins it in the one
    /// place it is decided — the source of the admission function itself.
    #[cfg(feature = "mp4")]
    #[test]
    fn the_prepared_row_precheck_runs_before_the_artifact_verification_pass() {
        let source = include_str!("private_server.rs");
        let body = source
            .split_once("fn prepare_reviewed_h3_private_fl2va_admission(")
            .expect("the admission function must exist")
            .1;
        let precheck = body
            .find("precheck_private_h3_prepared_rows(")
            .expect("admission must precheck the prepared rows");
        let artifacts = body
            .find("qualify_private_artifacts_with_control(")
            .expect("admission must qualify the artifacts");
        assert!(
            precheck < artifacts,
            "the prepared-row precheck must run before artifact verification"
        );
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

    /// The admitted route carries BOTH names: a Turbo tag stays the admitted
    /// request identity while its engine partition is the task's compact
    /// base — the name artifact qualification, Qwen support, the factory
    /// authority, and admission evidence key on. Threading the tag into
    /// those consumers refused every live Turbo admission with "requires an
    /// exact Comfy H3 canonical model name" even though every piece passed
    /// its own tests; this is the pin on the combination. Base models keep
    /// partition == admitted, so Ref2VA's task threading is unchanged.
    #[test]
    fn turbo_admission_routes_on_the_compact_engine_partition() {
        for tier in contract::REVIEWED_TURBO_MANIFEST_TIERS {
            let route = admitted_h3_route(tier.model).unwrap();
            assert_eq!(route.admitted_model, tier.model);
            assert_eq!(route.partition_model, contract::FL2VA_COMFY);
            assert_eq!(route.task, Task::Fl2va);
            // The partition is derived from the tier's own task, never a
            // constant — a future reviewed Ref2VA tier must partition to
            // REF2VA_COMFY through the same seam.
            assert_eq!(
                route.partition_model,
                contract::base_compact_model_for_task(route.task)
            );
        }
        assert_eq!(
            contract::base_compact_model_for_task(Task::Ref2va),
            contract::REF2VA_COMFY
        );

        let base = admitted_h3_route(contract::FL2VA_COMFY).unwrap();
        assert_eq!(base.admitted_model, contract::FL2VA_COMFY);
        assert_eq!(base.partition_model, contract::FL2VA_COMFY);
        assert_eq!(base.task, Task::Fl2va);

        let ref2va = admitted_h3_route(contract::REF2VA_COMFY).unwrap();
        assert_eq!(ref2va.admitted_model, contract::REF2VA_COMFY);
        assert_eq!(ref2va.partition_model, contract::REF2VA_COMFY);
        assert_eq!(ref2va.task, Task::Ref2va);

        assert!(
            admitted_h3_route("minimax-h3-fl2va:comfy-pruned-int8-turbo-2step").is_err(),
            "unreviewed lookalike tags stay outside the admitted route"
        );

        // The hidden official BF16 references have no compact engine
        // partition, so the route's None refusal fires HERE — not several
        // steps later inside artifact qualification.
        for official in [contract::FL2VA_OFFICIAL, contract::REF2VA_OFFICIAL] {
            let error = admitted_h3_route(official).unwrap_err().to_string();
            assert!(
                error.contains("compact engine partition"),
                "{official}: {error}"
            );
        }
    }

    /// A Turbo tier moves the step axis and ONLY the step axis, and only to the
    /// count its own distillation was reviewed for.
    #[test]
    fn a_turbo_tier_moves_only_the_step_axis_of_the_reviewed_envelope() {
        // Both FL2V tiers, so the default FL2VA task pin applies throughout.
        for tier in [
            mold_candle::minimax_h3::H3TurboLoraTier::Fl2v768p4StepV10,
            mold_candle::minimax_h3::H3TurboLoraTier::Fl2v8StepV10,
        ] {
            let adapter = turbo_authority_for(tier);
            let reviewed_steps = adapter.grid_points();
            reviewed_envelope(reviewed_steps)
                .validate_with_adapter(Some(&adapter))
                .unwrap();

            // Any other count, including the baseline 21, is refused for that
            // tier: an adapter distilled for N steps may not run at M.
            for wrong in [contract::COMFY_DEFAULT_STEPS, reviewed_steps + 1, 4] {
                if wrong == reviewed_steps {
                    continue;
                }
                let error = reviewed_envelope(wrong)
                    .validate_with_adapter(Some(&adapter))
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
            assert!(widened.validate_with_adapter(Some(&adapter)).is_err());
            let mut longer = reviewed_envelope(reviewed_steps);
            longer.frames += 4;
            assert!(longer.validate_with_adapter(Some(&adapter)).is_err());
            let mut batched = reviewed_envelope(reviewed_steps);
            batched.batch_size = 2;
            assert!(batched.validate_with_adapter(Some(&adapter)).is_err());
            let mut anchored = reviewed_envelope(reviewed_steps);
            anchored.endpoint_anchor = "last".into();
            assert!(anchored.validate_with_adapter(Some(&adapter)).is_err());
            let mut zeroed = reviewed_envelope(reviewed_steps);
            zeroed.max_total_packed_rows = 0;
            assert!(zeroed.validate_with_adapter(Some(&adapter)).is_err());
        }
    }

    /// The bug this pins: admission mints the qualification at the tier's step
    /// count and then has to validate the prepared request against THAT
    /// envelope. Validating it against the baseline 21-step envelope instead
    /// rejects every Turbo render.
    ///
    /// The earlier tests all asked "given an envelope, are wrong steps
    /// refused?", which is true whichever step authority the caller passes — so
    /// they passed while the positive admission path was broken. This one asks
    /// the question that actually distinguishes them: a Turbo envelope must
    /// ACCEPT its own tier's request and REJECT the same request under the
    /// no-adapter authority. If those two ever agree, the call-site choice
    /// stopped mattering and the wiring is unverifiable again.
    #[cfg(feature = "mp4")]
    #[test]
    fn a_turbo_prepared_request_validates_only_under_its_own_step_authority() {
        for tier in mold_candle::minimax_h3::H3TurboLoraTier::ALL {
            let adapter = turbo_authority_for(tier);
            let reviewed_steps = adapter.grid_points();
            // The real reviewed envelope with only the step axis moved, so
            // every other axis still matches the fixture request exactly.
            let mut envelope = record().envelope;
            envelope.max_steps = reviewed_steps;
            let mut request = prepared_request_for_compact_quality_envelope();
            request.grid_points = reviewed_steps;
            request.denoise_forward_count = reviewed_steps - 1;

            // The path admission must take.
            // FL2VA fixture request, so only an FL2V tier may move its steps.
            if adapter.reviewed_task() != Some(Task::Fl2va) {
                assert!(envelope
                    .validate_prepared_with_adapter(&request, Some(&adapter))
                    .is_err());
                continue;
            }
            envelope
                .validate_prepared_with_adapter(&request, Some(&adapter))
                .unwrap();

            // The path admission took before this fix. It must fail, loudly and
            // for the step reason — otherwise passing the wrong authority is
            // silently harmless and nothing pins the call site.
            let error = envelope
                .validate_prepared_with_adapter(&request, None)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("allows {} steps", contract::COMFY_DEFAULT_STEPS)),
                "{reviewed_steps}: {error}"
            );

            // And the baseline request is still refused under the Turbo
            // authority, so the tier's count is a pin rather than a widening.
            let mut baseline_request = prepared_request_for_compact_quality_envelope();
            baseline_request.grid_points = contract::COMFY_DEFAULT_STEPS;
            assert!(envelope
                .validate_prepared_with_adapter(&baseline_request, Some(&adapter))
                .is_err());
        }

        // The unadapted baseline is untouched: 21 steps under the None
        // authority still validates exactly as it always did.
        let baseline = record().envelope;
        let request = prepared_request_for_compact_quality_envelope();
        assert_eq!(request.grid_points, contract::COMFY_DEFAULT_STEPS);
        baseline
            .validate_prepared_with_adapter(&request, None)
            .unwrap();
    }

    /// The qualification boundary must not strip tier identity either.
    ///
    /// `public_runtime_qualification` mints an FL2VA-shaped record. Feeding it
    /// a bare step count let a Ref2V adapter set that record's envelope from a
    /// coincident 5-point schedule; the mismatch was caught later, but the
    /// record in between claimed an authority its adapter never had.
    #[cfg(all(feature = "h3", feature = "mp4"))]
    #[test]
    fn a_cross_task_turbo_adapter_cannot_mint_an_fl2va_qualification() {
        let ref2v = turbo_authority_for(mold_candle::minimax_h3::H3TurboLoraTier::Ref2v4StepV10);
        let fl2v = turbo_authority_for(mold_candle::minimax_h3::H3TurboLoraTier::Fl2v768p4StepV10);
        assert_eq!(ref2v.grid_points(), fl2v.grid_points());

        let artifact = artifact_report();
        let mint = |turbo| {
            public_runtime_qualification(
                &artifact,
                "gpu-0",
                0,
                Some((8, 9)),
                &sha('a'),
                "synthetic-qualified-kernel",
                &sha('b'),
                turbo,
            )
        };
        let error = mint(Some(&ref2v))
            .expect_err("a ref2v tier must not mint an fl2va qualification")
            .to_string();
        assert!(error.contains("was not reviewed for"), "{error}");
        // The FL2V tier at the identical step count is accepted, so the
        // refusal is about task identity and not about the number.
        assert!(mint(Some(&fl2v)).is_ok());
        assert!(mint(None).is_ok());
    }

    /// A tier's step authority is scoped to the task it was distilled for.
    ///
    /// The FL2V 768p and Ref2V tiers are both 5-point schedules, so an
    /// authority reduced to `Option<u32>` cannot tell them apart: an FL2V
    /// adapter would validate a Ref2VA envelope on a numeric coincidence.
    #[cfg(feature = "h3-private-uat")]
    #[test]
    fn a_turbo_tier_may_only_move_the_steps_of_its_own_task_envelope() {
        let fl2v = turbo_authority_for(mold_candle::minimax_h3::H3TurboLoraTier::Fl2v768p4StepV10);
        let ref2v = turbo_authority_for(mold_candle::minimax_h3::H3TurboLoraTier::Ref2v4StepV10);
        // Same step count - the coincidence that made this reachable.
        assert_eq!(fl2v.grid_points(), ref2v.grid_points());
        assert_eq!(fl2v.reviewed_task(), Some(Task::Fl2va));
        assert_eq!(ref2v.reviewed_task(), Some(Task::Ref2va));

        let mut ref2va_envelope = capture_runtime_envelope();
        ref2va_envelope.max_steps = ref2v.grid_points();
        ref2va_envelope
            .validate_for_task_with_adapter(Task::Ref2va, Some(&ref2v))
            .expect("the ref2v tier is the one reviewed for a Ref2VA envelope");
        let error = ref2va_envelope
            .validate_for_task_with_adapter(Task::Ref2va, Some(&fl2v))
            .expect_err("an fl2v tier must not carry a Ref2VA envelope")
            .to_string();
        assert!(error.contains("was not reviewed for"), "{error}");

        // And the mirror on the FL2VA side.
        let mut fl2va_envelope = record().envelope;
        fl2va_envelope.max_steps = fl2v.grid_points();
        fl2va_envelope
            .validate_for_task_with_adapter(Task::Fl2va, Some(&fl2v))
            .unwrap();
        assert!(fl2va_envelope
            .validate_for_task_with_adapter(Task::Fl2va, Some(&ref2v))
            .is_err());
    }

    /// Every reviewed tier's step count must survive the whole prepared-request
    /// envelope, not just the bare step comparison — this is the shape an
    /// admission-path request actually has.
    #[cfg(feature = "mp4")]
    #[test]
    fn every_reviewed_turbo_tier_admits_its_own_prepared_request() {
        for tier_contract in super::super::turbo::REVIEWED_TURBO_TIERS {
            let steps = tier_contract.grid_points;
            let adapter = turbo_authority_for(tier_contract.tier);
            let mut envelope = record().envelope;
            envelope.max_steps = steps;
            let mut request = prepared_request_for_compact_quality_envelope();
            request.grid_points = steps;
            request.denoise_forward_count = steps - 1;
            // The fixture request is FL2VA; a Ref2V tier is refused on it for
            // task identity rather than admitted on a matching step count.
            let outcome = envelope.validate_prepared_with_adapter(&request, Some(&adapter));
            if adapter.reviewed_task() == Some(Task::Fl2va) {
                outcome.unwrap_or_else(|error| {
                    panic!("{:?} at {steps} steps: {error}", tier_contract.tier)
                });
            } else {
                assert!(outcome.is_err(), "{:?} crossed tasks", tier_contract.tier);
            }
        }
    }

    /// The reviewed Turbo step counts are exactly the ones the tier table
    /// declares, so the envelope and the sampler cannot drift apart.
    #[test]
    fn reviewed_turbo_step_counts_come_from_the_tier_table() {
        for contract_entry in super::super::turbo::REVIEWED_TURBO_TIERS {
            let adapter = turbo_authority_for(contract_entry.tier);
            let outcome =
                reviewed_envelope(contract_entry.grid_points).validate_with_adapter(Some(&adapter));
            // `reviewed_envelope` is the FL2VA envelope, so only FL2V tiers
            // may carry it.
            if adapter.reviewed_task() == Some(Task::Fl2va) {
                outcome.unwrap();
            } else {
                assert!(outcome.is_err());
            }
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
                frames: contract::REVIEWED_COMPACT_FRAMES,
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
            frames: contract::REVIEWED_COMPACT_FRAMES,
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
            references: Vec::new(),
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
        envelope
            .validate_prepared_with_adapter(&reviewed, None)
            .unwrap();

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
            assert!(envelope
                .validate_prepared_with_adapter(&request, None)
                .is_err());
        }
    }

    /// Refusing is half the job: the message has to say WHICH axes differ and
    /// what each one's requested and reviewed values are. The composite
    /// "conditioning shape for Fl2va" entry this replaced named four possible
    /// faults at once and none of them specifically, and the two output flags
    /// read as a bare "off" with nothing to compare against.
    #[cfg(feature = "mp4")]
    #[test]
    fn an_envelope_refusal_names_every_failed_axis_with_both_values() {
        let envelope = record().envelope;
        let reviewed = prepared_request_for_compact_quality_envelope();

        // One request breaking six axes at once, three of them inside the old
        // conditioning composite. Every one must appear by name.
        let mut multi = reviewed.clone();
        multi.width = 544;
        multi.grid_points += 1;
        multi.synchronized_audio = false;
        multi.mode = Mode::TextToAudioVideo;
        multi.endpoints[0].anchor = H3FactoryEndpointAnchor::Last;
        multi.rows.condition_audio_rows = 7;
        let error = envelope
            .validate_prepared_with_adapter(&multi, None)
            .unwrap_err()
            .to_string();
        for expected in [
            &format!("width 544 (reviewed {})", envelope.width),
            "grid_points",
            "synchronized_audio false (reviewed true)",
            "mode TextToAudioVideo (reviewed FirstFrameToAudioVideo for Fl2va)",
            &format!(
                "endpoint_anchor last (reviewed {})",
                envelope.endpoint_anchor
            ),
            "condition_audio_rows 7 (reviewed 0 for Fl2va)",
        ] {
            assert!(error.contains(expected), "missing {expected:?} in {error}");
        }
        // The composite wording is gone, not merely supplemented.
        assert!(!error.contains("conditioning shape"), "{error}");

        // A Ref2VA envelope decomposes its own conditioning the same way: an
        // empty reference set and an over-cap soundtrack are separate,
        // separately numbered entries.
        let mut ref2va_envelope = envelope.clone();
        ref2va_envelope.endpoint_count = 0;
        ref2va_envelope.endpoint_anchor = "none".into();
        let mut ref2va = reviewed;
        ref2va.task = Task::Ref2va;
        ref2va.canonical_model = contract::REF2VA_COMFY.into();
        ref2va.mode = Mode::ReferenceToAudioVideo;
        ref2va.endpoints.clear();
        ref2va.rows.condition_audio_rows = ref2va_envelope.max_target_audio_rows + 3;
        let error = ref2va_envelope
            .validate_prepared_with_adapter(&ref2va, None)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("references 0 (reviewed at least 1 for Ref2va)"),
            "{error}"
        );
        assert!(
            error.contains(&format!(
                "condition_audio_rows {} (cap {})",
                ref2va_envelope.max_target_audio_rows + 3,
                ref2va_envelope.max_target_audio_rows
            )),
            "{error}"
        );
        // Its mode and anchor are the reviewed ones here, so neither is named:
        // the decomposition reports only what actually differs.
        assert!(!error.contains("mode "), "{error}");
        assert!(!error.contains("endpoint_anchor"), "{error}");
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
                    compute_capability: Some((8, 9)),
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
        assert_eq!(authority.compute_capability(), Some((8, 9)));
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
            compute_capability: Some((8, 9)),
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
        contract_factory(contract::FL2VA_COMFY)
    }

    /// The same contract-only frozen factory the FL2VA reopen fixtures use,
    /// parameterized so a Ref2VA route can freeze one too.
    fn contract_factory(model: &str) -> FrozenH3FactoryAuthority {
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: model.into(),
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
            compute_capability: Some((8, 9)),
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
            sampler: H3SamplerKind::ComfyResMultistep,
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
            ("comfy-euler", grid_points, evaluations),
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

    #[cfg(feature = "mp4")]
    #[test]
    fn turbo_owner_accepts_only_the_frozen_euler_sampler() {
        // A frozen Turbo attempt executes ComfyEuler; the terminal gate must
        // accept exactly that sampler and still refuse the non-Turbo one.
        let mut owner = owner_facts();
        owner.sampler = H3SamplerKind::ComfyEuler;
        owner.requested_grid_points = 9;
        owner.transformer_evaluations = 8;
        validate_private_sampler_provenance("comfy-euler", 9, 8, &owner).unwrap();
        for sampler in ["comfy-res-multistep", "official-euler"] {
            assert!(validate_private_sampler_provenance(sampler, 9, 8, &owner)
                .unwrap_err()
                .to_string()
                .contains("sampler provenance"));
        }
    }

    #[test]
    fn owner_facts_refuse_a_sampler_off_the_comfy_simple_grid() {
        let mut owner = owner_facts();
        owner.sampler = H3SamplerKind::OfficialEuler;
        assert!(owner
            .validate()
            .unwrap_err()
            .to_string()
            .contains("owner facts"));
    }

    #[test]
    fn reviewed_allowlist_is_scoped_to_fl2va() {
        assert!(reviewed_h3_private_runtime_available());
        assert!(reviewed_h3_private_runtime_available_for_task(Task::Fl2va));
        // Ref2VA has no reviewed bounds. It is reachable only from the
        // developer-only campaign build that exists to measure them, and a
        // shipping build — public `h3` or otherwise — must keep refusing it.
        assert_eq!(
            reviewed_h3_private_runtime_available_for_task(Task::Ref2va),
            cfg!(feature = "h3-private-uat")
        );
    }

    /// An owner fence agreeing with the frozen factory's own route, so a test
    /// isolates the task keying rather than a crossed device.
    #[cfg(feature = "mp4")]
    fn owner_fence_for(factory: &FrozenH3FactoryAuthority) -> H3PrivateFl2VaOwnerFenceFacts {
        H3PrivateFl2VaOwnerFenceFacts {
            work_identity_sha256: sha('1'),
            cancellation_scope_identity_sha256: sha('2'),
            device_id: factory.device_id().into(),
            device_ordinal: factory.device_ordinal(),
            compute_capability: factory.compute_capability(),
            memory_ledger_sequence: 9,
            admission_evidence_identity_sha256: sha('3'),
            artifact_qualification_identity_sha256: sha('4'),
            runtime_qualification_identity_sha256: sha('5'),
            prepared_attempt_identity_sha256: sha('6'),
            target_budget_identity_sha256: sha('7'),
            predicted_device_peak_bytes: 7,
            predicted_host_increment_bytes: 8,
        }
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn metal_owner_fence_accepts_the_unified_zero_host_claim() {
        let mut fence = owner_fence_for(&base_factory());
        fence.compute_capability = None;
        fence.predicted_host_increment_bytes = 0;
        fence.validate().unwrap();
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn cuda_owner_fence_requires_an_independent_host_claim() {
        let mut fence = owner_fence_for(&base_factory());
        fence.predicted_host_increment_bytes = 0;
        assert!(fence.validate().is_err());
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn metal_capacity_sample_needs_only_the_unified_pool() {
        assert!(private_h3_capacity_sample_is_concrete(None, 40, 0));
        assert!(!private_h3_capacity_sample_is_concrete(None, 0, 40));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn cuda_capacity_sample_still_needs_both_physical_pools() {
        assert!(private_h3_capacity_sample_is_concrete(Some((8, 9)), 40, 40));
        assert!(!private_h3_capacity_sample_is_concrete(Some((8, 9)), 40, 0,));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn frozen_factory_attention_backend_follows_the_device_route() {
        assert_eq!(
            private_h3_factory_attention_backend(None),
            AttentionBackend::Math,
        );
        assert_eq!(
            private_h3_factory_attention_backend(Some((8, 9))),
            AttentionBackend::Flash,
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn owner_fence_budget_preserves_cuda_and_projects_metal_unified_memory() {
        assert_eq!(
            private_h3_project_owner_fence_budget(Some((8, 9)), 30, 12, 38),
            (30, 12),
        );
        assert_eq!(
            private_h3_project_owner_fence_budget(None, 30, 12, 38),
            (38, 0),
        );
    }

    /// The resolved queue/worker form of an FL2VA request, matching what the
    /// reopen receives from the durable queue.
    #[cfg(feature = "mp4")]
    fn fl2va_reopen_request() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "model": contract::FL2VA_COMFY,
            "prompt": "a calm scene",
            "width": 1344,
            "height": 768,
            "steps": 21,
            "guidance": 0.0,
            "strength": 1.0,
            "batch_size": 1,
            "output_format": "mp4",
            "frames": 124,
            "fps": contract::FIXED_FPS,
            "seed": 42
        }))
        .unwrap()
    }

    /// The resolved Ref2VA form: references are descriptor authorities, which
    /// is the only shape the queue and worker may hold.
    #[cfg(feature = "mp4")]
    fn ref2va_reopen_request() -> GenerateRequest {
        use mold_core::{
            GenerationReference, GenerationReferenceAuthority, GenerationReferenceProvenance,
        };

        let mut request = fl2va_reopen_request();
        request.model = contract::REF2VA_COMFY.into();
        request.references = Some(vec![GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: GenerationReferenceProvenance {
                name: Some("portrait.png".to_string()),
                sha256: Some(sha('9')),
            },
            mime_type: "image/png".into(),
            width: 48,
            height: 48,
        }]);
        request
    }

    /// A frozen Ref2VA attempt must reach and pass the task-aware reopen
    /// validators: the base-factory gate, the resolved request contract, and
    /// the qualification partition are all keyed on the FROZEN task, never on
    /// an FL2VA constant. Validating the request as `Task::Fl2va` and
    /// qualifying `contract::FL2VA_COMFY` before the frozen-task logic ran is
    /// exactly what made the Ref2VA reopen path unreachable.
    #[cfg(all(feature = "mp4", feature = "h3-private-uat"))]
    #[test]
    fn frozen_ref2va_attempt_passes_the_task_aware_reopen_validators() {
        let factory = contract_factory(contract::REF2VA_COMFY);
        assert_eq!(factory.task(), Task::Ref2va);
        let fence = owner_fence_for(&factory);
        let route =
            validate_frozen_reopen_route(&ref2va_reopen_request(), &factory, &fence).unwrap();
        assert_eq!(route.task, Task::Ref2va);
        assert_eq!(route.partition_model, contract::REF2VA_COMFY);
        assert_eq!(route.transformer_task, H3TransformerTask::Ref2Va);
        assert_eq!(
            route.published_artifact,
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot
        );
        assert_eq!(route.mode, Mode::ReferenceToAudioVideo);

        // A crossed request — the other task's shape on this frozen route —
        // is refused by the same task-keyed validator.
        assert!(validate_frozen_reopen_route(&fl2va_reopen_request(), &factory, &fence).is_err());
    }

    /// The FL2VA reopen route is unchanged by the task keying.
    #[cfg(feature = "mp4")]
    #[test]
    fn frozen_fl2va_attempt_keeps_its_exact_reopen_route() {
        let factory = base_factory();
        let fence = owner_fence_for(&factory);
        let route =
            validate_frozen_reopen_route(&fl2va_reopen_request(), &factory, &fence).unwrap();
        assert_eq!(route.task, Task::Fl2va);
        assert_eq!(route.partition_model, contract::FL2VA_COMFY);
        assert_eq!(route.transformer_task, H3TransformerTask::T2VaFl2Va);
        assert_eq!(
            route.published_artifact,
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot
        );
        assert_eq!(route.mode, Mode::TextToAudioVideo);
        assert!(validate_frozen_reopen_route(&ref2va_reopen_request(), &factory, &fence).is_err());
    }

    /// Without the developer campaign build a frozen Ref2VA factory must stay
    /// refused at the reopen's first gate: task availability is per task, and
    /// an FL2VA qualification never authorizes Ref2VA.
    #[cfg(all(feature = "mp4", not(feature = "h3-private-uat")))]
    #[test]
    fn frozen_ref2va_attempt_stays_refused_without_the_campaign_build() {
        let factory = contract_factory(contract::REF2VA_COMFY);
        let fence = owner_fence_for(&factory);
        let error = validate_frozen_reopen_route(&ref2va_reopen_request(), &factory, &fence)
            .unwrap_err()
            .to_string();
        assert!(error.contains("frozen owner route"), "{error}");
    }

    /// The frozen media contract pairs mode and partition with its own task
    /// in both directions.
    #[test]
    fn frozen_media_contract_is_task_paired() {
        media().validate().unwrap();

        let ref2va = H3PrivateFl2VaMediaContract {
            canonical_model: contract::REF2VA_COMFY.into(),
            task: Task::Ref2va,
            mode: Mode::ReferenceToAudioVideo,
            ..media()
        };
        ref2va.validate().unwrap();

        // Each axis crossed against the task is refused: the FL2VA mode on a
        // Ref2VA contract, the Ref2VA mode on an FL2VA contract, and either
        // task carrying the other task's partition model.
        let crossed_mode = H3PrivateFl2VaMediaContract {
            mode: Mode::TextToAudioVideo,
            ..ref2va.clone()
        };
        assert!(crossed_mode.validate().is_err());
        let crossed_fl2va = H3PrivateFl2VaMediaContract {
            mode: Mode::ReferenceToAudioVideo,
            ..media()
        };
        assert!(crossed_fl2va.validate().is_err());
        let crossed_model = H3PrivateFl2VaMediaContract {
            canonical_model: contract::FL2VA_COMFY.into(),
            ..ref2va
        };
        assert!(crossed_model.validate().is_err());
        let crossed_task = H3PrivateFl2VaMediaContract {
            canonical_model: contract::REF2VA_COMFY.into(),
            ..media()
        };
        assert!(crossed_task.validate().is_err());
    }

    /// A Ref2VA admission under the campaign build must not require the
    /// reviewed-record FILE: Ref2VA has no reviewed record, and its authority
    /// is the compiled capture-scope profile. With the record path absent the
    /// admission must proceed past the runtime-qualification gate and fail
    /// later (here: at artifact qualification against an empty models root).
    #[cfg(all(feature = "mp4", not(feature = "h3")))]
    #[test]
    fn ref2va_admission_proceeds_past_the_missing_reviewed_record() {
        let models_root = tempfile::tempdir().unwrap();
        let staging_root = tempfile::tempdir().unwrap();
        let request = ref2va_reopen_request();
        let progress = ProgressReporter::default();
        let error = prepare_reviewed_h3_private_fl2va_admission(
            H3PrivateFl2VaAdmissionInput {
                request: &request,
                paths: H3PrivateFl2VaUatPaths {
                    models_root: models_root.path(),
                    staging_root: staging_root.path(),
                    authorization_record: Path::new("/nonexistent/authorization.json"),
                    runtime_qualification_record: Path::new(
                        "/nonexistent/runtime-qualification.json",
                    ),
                },
                references: &[],
                device_id: DEVICE_0,
                device_ordinal: 0,
                compute_capability: (8, 9),
                available_device_bytes: 1 << 60,
                available_host_headroom_bytes: 1 << 60,
            },
            &progress,
        )
        .unwrap_err()
        .to_string();
        assert!(
            !error.contains("runtime qualification"),
            "Ref2VA admission still died at the reviewed-record gate: {error}"
        );
    }

    /// Ref2VA under the campaign build resolves the compiled capture-scope
    /// authority: `CaptureCompiled` storage, the provisional decision string,
    /// and a record `validate_capture_runtime_profile` accepts — the same
    /// checks the constructor enforces at minting.
    #[cfg(all(not(feature = "h3"), feature = "h3-private-uat"))]
    #[test]
    fn ref2va_admission_resolves_the_compiled_capture_authority() {
        let source = private_runtime_qualification_source(
            Task::Ref2va,
            Path::new("/nonexistent/runtime-qualification.json"),
        )
        .unwrap();
        assert!(matches!(
            source,
            H3PrivateRuntimeQualificationSource::CaptureCompiled
        ));
        source.validate_route(DEVICE_0, 0, (8, 9)).unwrap();
        assert!(source.validate_route(DEVICE_0, 0, (9, 0)).is_err());
        assert_eq!(
            source.precheck_bounds(),
            capture_runtime_bounds(),
            "the capture arm's admission floors come from the capture ceilings"
        );

        let mut artifact = artifact_report();
        artifact.canonical_model = contract::REF2VA_COMFY.into();
        artifact.task = "ref2va";
        let attention_device = H3AttentionDevice::Cuda {
            compute_capability: Some((8, 9)),
        };
        let attention_model = H3AttentionModelContract::released_bf16();
        let kernel = H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011;
        let runtime_identity = H3AttentionRuntimeAuthority::expected_identity_for(
            H3AttentionBackend::FlashAttentionV2,
            kernel,
            H3AttentionActivation::ReleaseCandidateQualificationOnly,
            attention_device,
            attention_model,
        )
        .unwrap();
        let authority = source
            .authenticate(
                &artifact,
                DEVICE_0,
                0,
                (8, 9),
                &runtime_identity,
                kernel.identity(),
                &runtime_identity,
            )
            .unwrap();
        assert!(matches!(
            authority.storage,
            RuntimeQualificationStorage::CaptureCompiled
        ));
        assert_eq!(authority.record.decision, CAPTURE_RUNTIME_PROFILE_DECISION);
        assert!(authority.record.decision.contains("provisional"));
        assert_eq!(authority.record.canonical_model, contract::REF2VA_COMFY);
        assert_eq!(authority.record.task, "ref2va");
        validate_capture_runtime_profile(&authority.record, &authority.record_file_sha256).unwrap();
        authority.revalidate().unwrap();

        // A cross-task artifact report cannot mint the capture authority.
        let error = private_runtime_qualification_source(
            Task::Ref2va,
            Path::new("/nonexistent/runtime-qualification.json"),
        )
        .unwrap()
        .authenticate(
            &artifact_report(),
            DEVICE_0,
            0,
            (8, 9),
            &runtime_identity,
            kernel.identity(),
            &runtime_identity,
        )
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(error.contains("Ref2VA"), "{error}");
    }

    /// FL2VA under the same build keeps its reviewed-record FILE gate exactly
    /// as before: a missing record still refuses the task.
    #[cfg(not(feature = "h3"))]
    #[test]
    fn fl2va_admission_still_requires_the_reviewed_record_file() {
        let Err(error) = private_runtime_qualification_source(
            Task::Fl2va,
            Path::new("/nonexistent/runtime-qualification.json"),
        ) else {
            panic!("FL2VA must not resolve a runtime qualification without its record file")
        };
        let error = error.to_string();
        assert!(
            error.contains("failed to open private H3 runtime qualification"),
            "{error}"
        );
    }

    /// The capture-scope profile must never be mistakable for a qualified one.
    #[test]
    #[cfg(feature = "h3-private-uat")]
    fn capture_scope_profile_announces_that_its_bounds_are_provisional() {
        assert!(CAPTURE_RUNTIME_PROFILE_DECISION.contains("provisional"));
        assert!(CAPTURE_RUNTIME_PROFILE_DECISION.contains("capture-scope"));
        assert_ne!(CAPTURE_RUNTIME_PROFILE_SCHEMA, RUNTIME_QUALIFICATION_SCHEMA);
        assert_ne!(
            CAPTURE_RUNTIME_PROFILE_DECISION,
            RUNTIME_QUALIFICATION_DECISION
        );

        // Its envelope is Ref2VA-shaped: no boundary endpoint, and reference
        // conditioning priced from the released reference limits.
        let envelope = capture_runtime_envelope();
        envelope.validate_for_task(Task::Ref2va).unwrap();
        assert!(envelope.validate_for_task(Task::Fl2va).is_err());
        assert_eq!(envelope.endpoint_count, 0);
        assert_eq!(envelope.endpoint_anchor, "none");

        // Every ceiling is at or above the corrected FL2VA measurement it was
        // scaled from, and the sequence- and canvas-scaled ones are strictly
        // above it.
        let ceilings = capture_runtime_bounds();
        ceilings.validate().unwrap();
        assert!(ceilings.attention_workspace_device_bytes > 6_323_525_308);
        assert!(ceilings.ffn_workspace_device_bytes > 7_826_714_044);
        assert!(ceilings.condition_vae_workspace_device_bytes > 366_027_840);
        assert!(ceilings.qwen_activation_workspace_bytes > 4_168_069_120);

        // The admission floors these ceilings imply must clear the SM89
        // campaign card the profile exists to measure on. Campaign attempt 5
        // (RTX 4090, 24 GiB) sampled exactly these device/host figures and
        // was refused at a 51.3 GB device floor because the provisional
        // attention/FFN ceilings carried pre-correction ledger sizing; the
        // corrected derivation must admit on that same sample.
        #[cfg(feature = "mp4")]
        {
            const SM89_CAMPAIGN_DEVICE_SAMPLE_BYTES: u64 = 24_967_446_528;
            const SM89_CAMPAIGN_HOST_SAMPLE_BYTES: u64 = 46_225_700_250;
            let device_floor = private_h3_admission_device_floor_bytes(&ceilings).unwrap();
            let host_floor = private_h3_admission_host_floor_bytes(&ceilings).unwrap();
            assert!(
                device_floor <= 22_000_000_000,
                "capture device floor {device_floor} exceeds the ~22 GB campaign target"
            );
            precheck_private_h3_admission_capacity(
                &ceilings,
                Some((8, 9)),
                SM89_CAMPAIGN_DEVICE_SAMPLE_BYTES,
                SM89_CAMPAIGN_HOST_SAMPLE_BYTES,
            )
            .unwrap();
            // Pin the derived floors so any re-derivation of the ceilings is
            // a visible, reviewed decision rather than silent drift.
            assert_eq!(device_floor, 603_979_776 + 16_978_542_592);
            assert_eq!(host_floor, 805_306_368 + 1_052_855_836);
        }

        // The two sequence-linear denoise transients are the FL2VA
        // observations scaled by the exact envelope packed-row ratio, then
        // rounded up to the 64 MiB grant grid. #1245 moved BOTH halves of that
        // ratio — the observation by +2.5% and the FL2VA envelope by the same
        // +2.5% — so the derived Ref2VA ceilings are byte-identical to the
        // pre-#1245 ones, which is why the two device-floor pins below did not
        // move either.
        let rows = capture_runtime_envelope().max_total_packed_rows;
        assert_eq!(rows, 88_334);
        assert_eq!(
            ceilings.attention_workspace_device_bytes,
            (6_323_525_308_u64 * rows / REVIEWED_MAX_TOTAL_PACKED_ROWS)
                .next_multiple_of(64 * 1024 * 1024)
        );
        assert_eq!(
            ceilings.ffn_workspace_device_bytes,
            (7_826_714_044_u64 * rows / REVIEWED_MAX_TOTAL_PACKED_ROWS)
                .next_multiple_of(64 * 1024 * 1024)
        );

        // The report renders observation against ceiling so a reviewer
        // transcribes the measurement, never the ceiling.
        let report = h3_capture_bound_report(&[("attention_workspace_device_bytes", 1_000_000)]);
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].1, 1_000_000);
        assert_eq!(report[0].2, ceilings.attention_workspace_device_bytes);
        assert!(report[0].3 < 1.0);
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
            .find("private_runtime_qualification_source(")
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
            .find("private_runtime_qualification_source(")
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
        // The binding, not its line breaks: rustfmt reflows this expression
        // whenever the surrounding builder changes shape.
        let collapsed = opened_evidence
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(collapsed.contains(
            "vae_memory_evidence_identity_sha256:vae.artifact_validation_identity_sha256()"
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

    #[test]
    fn private_h3_metal_execution_reuses_the_process_device() {
        let source = include_str!("private_server.rs");
        let production = source
            .split_once("#[cfg(test)]\nmod tests")
            .expect("private server test module boundary")
            .0;
        assert!(production.contains("crate::device::metal_device(owner.device_ordinal)"));
        assert!(
            !production.contains("Device::new_metal"),
            "private H3 must not mint a second Candle Metal identity"
        );
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
    fn public_presentation_accepts_compiled_cuda_and_metal_policies() {
        let metal = H3PrivatePresentationRoute {
            device_id: "metal:00000000000000000000000000000000",
            device_ordinal: 0,
            compute_capability: None,
        };
        let metal_authority = authenticate_h3_public_presentation(&[metal]).unwrap();
        assert_eq!(metal_authority.canonical_model(), contract::FL2VA_COMFY);
        assert_eq!(metal_authority.task(), Task::Fl2va);
        assert_eq!(metal_authority.compute_capability(), None);

        let route = H3PrivatePresentationRoute {
            device_id: "cuda:00000000000000000000000000000000",
            device_ordinal: 0,
            compute_capability: Some((8, 9)),
        };
        #[cfg(feature = "cuda")]
        {
            let authority = authenticate_h3_public_presentation(&[route]).unwrap();
            assert_eq!(authority.canonical_model(), contract::FL2VA_COMFY);
            assert_eq!(authority.task(), Task::Fl2va);
            assert_eq!(authority.compute_capability(), Some((8, 9)));
        }

        let unsupported = H3PrivatePresentationRoute {
            compute_capability: Some((9, 0)),
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
                qwen_activation_workspace_bytes: 4_831_838_208,
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
            record(7_314_866_176, 9_059_696_640, 603_979_776, 805_306_368),
            record(15_300_820_992, 10_133_438_464, 603_979_776, 671_088_640),
            record(u32::MAX.into(), 1, u32::MAX.into(), u32::MAX.into()),
            // The derived capture-scope ceilings themselves, so the floor
            // invariant is checked at exactly the record the Ref2VA campaign
            // admits under.
            #[cfg(feature = "h3-private-uat")]
            capture_runtime_bounds(),
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
                    H3PrivateQwenLoaderMemoryRoute::Metal,
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
                Some((8, 9)),
                device_floor.saturating_sub(1),
                u64::MAX
            )
            .is_err());
            assert!(precheck_private_h3_admission_capacity(
                bounds,
                Some((8, 9)),
                u64::MAX,
                host_floor.saturating_sub(1)
            )
            .is_err());
            assert!(precheck_private_h3_admission_capacity(
                bounds,
                Some((8, 9)),
                device_floor,
                host_floor
            )
            .is_ok());

            let unified_floor = device_floor.max(host_floor);
            assert!(precheck_private_h3_admission_capacity(
                bounds,
                None,
                unified_floor.saturating_sub(1),
                1
            )
            .is_err());
            precheck_private_h3_admission_capacity(bounds, None, unified_floor, 1).unwrap();
        }
    }

    /// The exact per-attempt budget refusal has to tell an operator WHICH
    /// resource fell short and by how much. The single OR'd message named
    /// neither the failing resource nor either sample, so a host-headroom
    /// refusal read exactly like a VRAM one (#1214).
    #[cfg(feature = "mp4")]
    #[test]
    fn the_exact_target_budget_refusal_names_its_own_resource_and_both_numbers() {
        check_private_h3_target_budget_fits(
            9_000_000_000,
            7_000_000_000,
            Some((8, 9)),
            9_000_000_000,
            7_000_000_000,
        )
        .unwrap();

        let device = check_private_h3_target_budget_fits(
            9_000_000_001,
            7_000_000_000,
            Some((8, 9)),
            9_000_000_000,
            7_000_000_000,
        )
        .unwrap_err()
        .to_string();
        assert!(device.contains("9000000001"), "{device}");
        assert!(device.contains("9000000000"), "{device}");
        assert!(device.contains("device"), "{device}");
        assert!(!device.contains("host"), "{device}");

        let host = check_private_h3_target_budget_fits(
            9_000_000_000,
            7_000_000_001,
            Some((8, 9)),
            9_000_000_000,
            7_000_000_000,
        )
        .unwrap_err()
        .to_string();
        assert!(host.contains("7000000001"), "{host}");
        assert!(host.contains("7000000000"), "{host}");
        assert!(host.contains("host"), "{host}");
        assert!(!host.contains("device"), "{host}");

        check_private_h3_target_budget_fits(9_000_000_000, 7_000_000_000, None, 9_000_000_000, 1)
            .unwrap();
        let metal = check_private_h3_target_budget_fits(
            9_000_000_001,
            7_000_000_000,
            None,
            9_000_000_000,
            1,
        )
        .unwrap_err()
        .to_string();
        assert!(metal.contains("9000000001"), "{metal}");
        assert!(metal.contains("9000000000"), "{metal}");
        assert!(metal.contains("unified-memory"), "{metal}");
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn metal_runs_qwen_on_the_assigned_unified_memory_device() {
        assert_eq!(
            private_h3_qwen_route(None),
            (
                H3PrivateQwenLoaderMemoryRoute::Metal,
                H3FactoryConditionerPlacement::AssignedMetalThenDrop,
            )
        );
        assert_eq!(
            private_h3_qwen_route(Some((8, 9))),
            (
                H3PrivateQwenLoaderMemoryRoute::Cpu,
                H3FactoryConditionerPlacement::HostCpuThenDrop,
            )
        );
    }

    #[cfg(feature = "h3")]
    #[test]
    fn public_runtime_authority_pins_every_bound_envelope_and_identity_axis() {
        let artifact = artifact_report();
        let authority = public_runtime_qualification(
            &artifact,
            DEVICE_0,
            0,
            Some((8, 9)),
            &sha('d'),
            "flash-attention-v2-sm89",
            &sha('e'),
            None,
        )
        .unwrap();
        authority.revalidate().unwrap();
        assert_eq!(authority.device_id(), DEVICE_0);
        assert_eq!(authority.device_ordinal(), 0);
        assert_eq!(authority.compute_capability(), Some((8, 9)));
        assert_eq!(authority.artifact_qualification_identity_sha256(), sha('c'));
        assert_eq!(
            H3PrivateFl2VaRuntimeBounds::from(authority.bounds()),
            H3PrivateFl2VaRuntimeBounds {
                fixed_runtime_host_bytes: 805_306_368,
                fixed_runtime_device_bytes: 603_979_776,
                qwen_activation_workspace_bytes: 4_831_838_208,
                vae_construction_device_workspace_bytes: 67_108_864,
                condition_vae_workspace_device_bytes: 469_762_048,
                attention_workspace_device_bytes: 7_314_866_176,
                ffn_workspace_device_bytes: 9_059_696_640,
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
                // The reviewed prompt budget: 1,008 vision pads and the
                // `"<Picture 1>: "` label leave about a thousand prompt
                // tokens under this ceiling (#1245).
                max_qwen_output_text_rows: 2_048,
                max_qwen_vision_rows: 4_032,
                max_condition_visual_rows: 1_008,
                max_target_video_rows: 37_296,
                max_target_audio_rows: 414,
                max_total_packed_rows: 40_766,
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
                Some(cc),
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
                Some((8, 9)),
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
            Some((8, 9)),
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
