//! Contract-only MiniMax H3 factory authority.
//!
//! This module is the typed seam between server admission and the private H3
//! Candle backend. It carries only digests and immutable scheduling facts; it
//! never receives artifact paths or bytes. Construction deliberately creates
//! the backend's unavailable plan, so neither a qualified admission record nor
//! this public type can activate the family by itself.

use anyhow::{anyhow, bail, Result};
use mold_candle::minimax_h3::{
    H3AttentionActivation, H3AttentionBackend, H3AttentionDevice, H3AttentionKernel,
    H3AttentionModelContract, H3AttentionRuntimeAuthority,
};
use mold_core::minimax_h3::{self as contract, Layout, Mode, Task};
use sha2::{Digest, Sha256};

use crate::attention::{AttentionBackend, AttentionChunkPolicy};
use crate::minimax_h3::backend::{
    FrozenH3Fl2VaCandlePlan, H3CandleBackendDevice, H3ComponentRole, H3ValidatedComponentAuthority,
    H3ValidatedComponentSet,
};
use crate::minimax_h3::offload::FrozenH3BlockStreamingPlan;
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
use crate::minimax_h3::private_server::H3PrivateFactoryActivationEvidence;
use crate::minimax_h3::sampler::H3DualSchedule;
use crate::minimax_h3::vae_runtime::expected_h3_comfy_vae_artifact_plan_identity;
use crate::minimax_h3::{FrozenH3ConditionerPlacement, H3ConditionerExecution};

const H3_FACTORY_AUTHORITY_SCHEMA_VERSION: u32 = 5;
const H3_FACTORY_QWEN_MODEL_MAX_ROWS: u64 = 262_144;
const H3_FACTORY_MAX_GRID_POINTS: u32 = 4_096;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum H3FactoryComponentRole {
    Conditioner,
    Transformer,
    VisualVae,
    AudioVae,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryComponentAuthority {
    role: H3FactoryComponentRole,
    content_sha256: String,
    validation_sha256: String,
}

impl H3FactoryComponentAuthority {
    pub fn new(
        role: H3FactoryComponentRole,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryConditionerPlacement {
    AssignedCudaThenDrop,
    HostCpuThenDrop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryEndpointAnchor {
    First,
    Last,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryEndpointPreprocess {
    PillowLanczosRgbU8CpuV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryEndpointInput {
    pub anchor: H3FactoryEndpointAnchor,
    pub encoded_bytes: u64,
    pub encoded_content_sha256: String,
    pub preprocess: H3FactoryEndpointPreprocess,
    pub normalized_shape: [u32; 5],
    /// Exact CPU U8 `[1, 3, 1, height, width]` tensor retained after
    /// preprocessing. The later execution stack must recompute this charge.
    pub normalized_cpu_bytes: u64,
    pub normalized_cpu_content_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryPreparedRowsInput {
    pub qwen_output_text_rows: u64,
    pub qwen_vision_rows: u64,
    pub condition_visual_rows: u64,
    pub condition_audio_rows: u64,
    pub target_video_rows: u64,
    pub target_audio_rows: u64,
    pub total_packed_rows: u64,
}

/// Exact normalized request fields consumed by the private H3 pipeline.
/// `identity_sha256` is supplied by server admission but recomputed here from
/// every explicit field; an opaque caller-selected request identity is never
/// trusted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryPreparedRequestInput {
    pub identity_sha256: String,
    pub canonical_model: String,
    pub task: Task,
    pub mode: Mode,
    pub prompt_sha256: String,
    /// Concrete resolved attempt seed. Admission must resolve an omitted
    /// request seed before constructing this record.
    pub seed: u64,
    pub grid_points: u32,
    pub denoise_forward_count: u32,
    pub guidance_f64_bits: u64,
    pub strength_f64_bits: u64,
    pub batch_size: u32,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub synchronized_audio: bool,
    pub mp4_output: bool,
    pub video_latent_frames: u64,
    pub audio_latents_per_channel: u64,
    pub audio_samples_per_channel: u64,
    pub conditioning_fingerprint: String,
    pub reference_fingerprint: String,
    pub endpoints: Vec<H3FactoryEndpointInput>,
    pub rows: H3FactoryPreparedRowsInput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryBlockMemoryInput {
    pub index: u16,
    pub encoded_host_bytes: u64,
    pub protected_device_bytes: u64,
    pub max_device_weight_staging_bytes: u64,
    pub max_host_read_staging_bytes: u64,
    pub content_sha256: String,
}

/// Raw opened-checkpoint authority. This is a distinct typed domain from the
/// logical transformer component aggregate retained by the backend plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryRawCheckpointInput {
    pub identity_sha256: String,
    pub raw_content_sha256: String,
    pub verified_file_bytes: u64,
    pub raw_header_identity_sha256: String,
    pub opened_checkpoint_identity_sha256: String,
    pub quantization_policy_identity_sha256: String,
    pub config_identity_sha256: String,
    pub fixed_transformer_encoded_host_bytes: u64,
    pub fixed_transformer_protected_device_bytes: u64,
    pub fixed_transformer_max_host_read_staging_bytes: u64,
    pub fixed_transformer_max_device_weight_staging_bytes: u64,
    pub blocks: Vec<H3FactoryBlockMemoryInput>,
}

/// Required future consuming order whose conservative target overlaps are
/// priced below. The current private composer does not yet implement it. This
/// stack intentionally supports one policy only; adding another is a schema
/// change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryTargetLoadDropPolicy {
    LoadVaesLoadQwenEncodeTransferDropQwenEncodeConditionsAllocateNoiseLoadTransformerDenoiseDropTransformerDecodeVisualAudioDropVaesMux,
    /// Test-only marker proving that the policy discriminator participates in
    /// the target-budget identity without advertising another executable plan.
    #[cfg(test)]
    IdentityMutationSentinel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryTargetDenoiseCopyPolicy {
    /// Conservative nine-full-state-copy ceiling for the paired FP32 RES
    /// multistep update, including model output, the prior clean estimates,
    /// the continuously carried audio state, and intermediate blend tensors.
    CandleF32PairedResMultistepV2,
    /// Test-only marker proving that the policy discriminator participates in
    /// the target-budget identity without advertising another executable plan.
    #[cfg(test)]
    IdentityMutationSentinel,
}

/// Evidence and behavior that must land before a target budget may execute.
///
/// Keeping these blockers typed makes the schema useful without allowing a
/// caller-supplied budget to masquerade as current-runtime proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryActivationPrerequisite {
    /// Derive Qwen, transformer, and VAE memory facts from authenticated,
    /// opened runtime objects rather than caller assertions.
    OpenedComponentMemoryEvidence,
    /// Make the stream/artifact lease echo the prepared raw content, opened
    /// identity, fixed facts, policy, and all 50 block facts. Logical component
    /// digests remain a separate domain and need no equality relation.
    PreparedCheckpointExecutionEcho,
    /// Implement the target load/drop transitions with consuming ownership.
    ConsumingTargetLifetimeTransitions,
    /// Price every raw condition, noise, patch, concatenation, unpack, and
    /// decode copy that survives a phase boundary.
    RetainedTensorOverlapBudget,
    /// Derive packed-layout, tag, schedule, mapping, I/O, authentication, and
    /// construction transients on the host side.
    HostLayoutAndTransientBudget,
    /// Derive decoded-source, resize/filter, and encoded-endpoint lifetimes
    /// from the concrete preprocessing execution.
    EndpointPreprocessTransientBudget,
    /// Replace the reusable cached runtime session with a fresh component
    /// runtime and a self-consuming run API for each job.
    PerAttemptRuntimeConstruction,
    /// Consume a fresh non-Clone scheduler lease binding attempt nonce,
    /// host/device/ordinal, and cancellation-slot identity.
    OneShotSchedulerLease,
    /// Make VAE, Qwen, transformer/fifty-block loading, both decoders, and mux
    /// poll one shared active attempt/cancellation scope.
    SameAttemptCancellationCoverage,
}

const H3_FACTORY_ACTIVATION_PREREQUISITES: &[H3FactoryActivationPrerequisite] = &[
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

pub const fn h3_factory_activation_prerequisites() -> &'static [H3FactoryActivationPrerequisite] {
    H3_FACTORY_ACTIVATION_PREREQUISITES
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum H3FactoryArtifactHostRole {
    Conditioner,
    RawTransformerCheckpoint,
    TransformerSupport,
    VisualVae,
    AudioVae,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryArtifactHostInput {
    pub role: H3FactoryArtifactHostRole,
    pub index: u16,
    pub content_sha256: String,
    pub bytes: u64,
}

/// Conservative target budget for a future consuming execution stack.
///
/// This is not evidence about the current private runtime. Its phase fields
/// describe the required target order in [`H3FactoryTargetLoadDropPolicy`] and
/// recompute from the detailed planned charges. A later binder must derive
/// those charges from opened runtime objects before this can become executable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryTargetBudgetInput {
    pub identity_sha256: String,
    pub load_drop_policy: H3FactoryTargetLoadDropPolicy,
    pub artifacts: Vec<H3FactoryArtifactHostInput>,
    pub artifact_host_bytes: u64,
    pub fixed_runtime_host_bytes: u64,
    pub qwen_host_parameter_bytes: u64,
    pub qwen_host_activation_bytes: u64,
    pub qwen_host_output_state_bytes: u64,
    pub qwen_host_workspace_bytes: u64,
    pub condition_backing_host_bytes: u64,
    pub endpoint_encoded_host_bytes: u64,
    pub normalized_endpoint_host_bytes: u64,
    pub schedule_host_bytes: u64,
    pub packed_layout_host_bytes: u64,
    pub packed_layout_construction_staging_host_bytes: u64,
    pub packed_layout_freeze_staging_host_bytes: u64,
    pub text_modality_tags_host_bytes: u64,
    pub noise_cpu_staging_host_bytes: u64,
    pub vae_peak_host_io_buffer_bytes: u64,
    pub vae_peak_host_mapped_file_bytes: u64,
    pub vae_peak_staging_disk_bytes: u64,
    pub max_host_read_staging_bytes: u64,
    pub max_streamed_block_host_overlap_bytes: u64,
    pub fixed_transformer_load_host_staging_bytes: u64,
    pub encoded_video_host_bytes_bound: u64,
    pub thumbnail_host_bytes_bound: u64,
    pub waveform_host_bytes: u64,
    pub mux_output_host_bytes_bound: u64,
    pub aac_mux_staging_host_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub fixed_runtime_device_bytes: u64,
    pub fixed_transformer_device_bytes: u64,
    pub visual_vae_resident_device_bytes: u64,
    pub audio_vae_resident_device_bytes: u64,
    pub attempt_resident_vae_device_bytes: u64,
    pub vae_construction_device_workspace_bytes: u64,
    pub vae_memory_evidence_identity_sha256: String,
    pub qwen_device_parameter_bytes: u64,
    pub qwen_activation_device_bytes: u64,
    pub qwen_output_state_device_bytes: u64,
    pub qwen_output_transfer_device_bytes: u64,
    pub condition_vae_workspace_device_bytes: u64,
    pub condition_latent_backing_device_bytes: u64,
    pub target_video_latent_device_bytes: u64,
    pub target_audio_latent_device_bytes: u64,
    pub packed_layout_device_bytes: u64,
    pub packed_video_state_device_bytes: u64,
    pub packed_audio_state_device_bytes: u64,
    pub denoise_copy_policy: H3FactoryTargetDenoiseCopyPolicy,
    pub denoise_tensor_copy_workspace_device_bytes: u64,
    pub audio_waveform_device_bytes: u64,
    pub attention_workspace_device_bytes: u64,
    pub ffn_workspace_device_bytes: u64,
    pub decoder_tile_workspace_device_bytes: u64,
    pub audio_decode_workspace_device_bytes: u64,
    pub resident_block_device_bytes: u64,
    pub streamed_block_device_bytes: u64,
    pub prefetch_device_bytes: u64,
    pub dequantization_workspace_device_bytes: u64,
    pub protected_block_device_bytes: u64,
    pub streamed_block_device_overlap_bytes: u64,
    pub max_device_weight_staging_bytes: u64,
    pub fixed_transformer_load_device_staging_bytes: u64,
    pub vae_load_phase_device_bytes: u64,
    pub qwen_encode_phase_device_bytes: u64,
    pub qwen_transfer_phase_device_bytes: u64,
    pub condition_encode_phase_device_bytes: u64,
    pub noise_allocation_phase_device_bytes: u64,
    pub transformer_load_phase_device_bytes: u64,
    pub denoise_phase_device_bytes: u64,
    pub visual_decode_phase_device_bytes: u64,
    pub audio_decode_phase_device_bytes: u64,
    pub waveform_transfer_phase_device_bytes: u64,
    pub mux_phase_device_bytes: u64,
    pub predicted_device_peak_bytes: u64,
}

impl H3FactoryTargetBudgetInput {
    /// Append every semantically authoritative memory field to `hash`.
    ///
    /// The exhaustive destructure is intentional: adding a field to the
    /// schema must fail compilation here until its identity treatment is
    /// explicit. `identity_sha256` is the sole derived field.
    fn update_identity(&self, hash: &mut Sha256) {
        let Self {
            identity_sha256: _,
            load_drop_policy,
            artifacts,
            artifact_host_bytes,
            fixed_runtime_host_bytes,
            qwen_host_parameter_bytes,
            qwen_host_activation_bytes,
            qwen_host_output_state_bytes,
            qwen_host_workspace_bytes,
            condition_backing_host_bytes,
            endpoint_encoded_host_bytes,
            normalized_endpoint_host_bytes,
            schedule_host_bytes,
            packed_layout_host_bytes,
            packed_layout_construction_staging_host_bytes,
            packed_layout_freeze_staging_host_bytes,
            text_modality_tags_host_bytes,
            noise_cpu_staging_host_bytes,
            vae_peak_host_io_buffer_bytes,
            vae_peak_host_mapped_file_bytes,
            vae_peak_staging_disk_bytes,
            max_host_read_staging_bytes,
            max_streamed_block_host_overlap_bytes,
            fixed_transformer_load_host_staging_bytes,
            encoded_video_host_bytes_bound,
            thumbnail_host_bytes_bound,
            waveform_host_bytes,
            mux_output_host_bytes_bound,
            aac_mux_staging_host_bytes,
            predicted_host_increment_bytes,
            fixed_runtime_device_bytes,
            fixed_transformer_device_bytes,
            visual_vae_resident_device_bytes,
            audio_vae_resident_device_bytes,
            attempt_resident_vae_device_bytes,
            vae_construction_device_workspace_bytes,
            vae_memory_evidence_identity_sha256,
            qwen_device_parameter_bytes,
            qwen_activation_device_bytes,
            qwen_output_state_device_bytes,
            qwen_output_transfer_device_bytes,
            condition_vae_workspace_device_bytes,
            condition_latent_backing_device_bytes,
            target_video_latent_device_bytes,
            target_audio_latent_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            denoise_copy_policy,
            denoise_tensor_copy_workspace_device_bytes,
            audio_waveform_device_bytes,
            attention_workspace_device_bytes,
            ffn_workspace_device_bytes,
            decoder_tile_workspace_device_bytes,
            audio_decode_workspace_device_bytes,
            resident_block_device_bytes,
            streamed_block_device_bytes,
            prefetch_device_bytes,
            dequantization_workspace_device_bytes,
            protected_block_device_bytes,
            streamed_block_device_overlap_bytes,
            max_device_weight_staging_bytes,
            fixed_transformer_load_device_staging_bytes,
            vae_load_phase_device_bytes,
            qwen_encode_phase_device_bytes,
            qwen_transfer_phase_device_bytes,
            condition_encode_phase_device_bytes,
            noise_allocation_phase_device_bytes,
            transformer_load_phase_device_bytes,
            denoise_phase_device_bytes,
            visual_decode_phase_device_bytes,
            audio_decode_phase_device_bytes,
            waveform_transfer_phase_device_bytes,
            mux_phase_device_bytes,
            predicted_device_peak_bytes,
        } = self;

        hash.update(match load_drop_policy {
            H3FactoryTargetLoadDropPolicy::LoadVaesLoadQwenEncodeTransferDropQwenEncodeConditionsAllocateNoiseLoadTransformerDenoiseDropTransformerDecodeVisualAudioDropVaesMux => {
                b"load-vaes-load-qwen-encode-transfer-drop-qwen-encode-conditions-allocate-noise-load-transformer-denoise-drop-transformer-decode-visual-audio-drop-vaes-mux".as_slice()
            }
            #[cfg(test)]
            H3FactoryTargetLoadDropPolicy::IdentityMutationSentinel => {
                b"identity-mutation-sentinel".as_slice()
            }
        });
        hash.update((artifacts.len() as u64).to_le_bytes());
        for artifact in artifacts {
            hash.update(artifact_host_role_id(artifact.role));
            hash.update(artifact.index.to_le_bytes());
            update_string(hash, &artifact.content_sha256);
            hash.update(artifact.bytes.to_le_bytes());
        }
        for value in [
            artifact_host_bytes,
            fixed_runtime_host_bytes,
            qwen_host_parameter_bytes,
            qwen_host_activation_bytes,
            qwen_host_output_state_bytes,
            qwen_host_workspace_bytes,
            condition_backing_host_bytes,
            endpoint_encoded_host_bytes,
            normalized_endpoint_host_bytes,
            schedule_host_bytes,
            packed_layout_host_bytes,
            packed_layout_construction_staging_host_bytes,
            packed_layout_freeze_staging_host_bytes,
            text_modality_tags_host_bytes,
            noise_cpu_staging_host_bytes,
            vae_peak_host_io_buffer_bytes,
            vae_peak_host_mapped_file_bytes,
            vae_peak_staging_disk_bytes,
            max_host_read_staging_bytes,
            max_streamed_block_host_overlap_bytes,
            fixed_transformer_load_host_staging_bytes,
            encoded_video_host_bytes_bound,
            thumbnail_host_bytes_bound,
            waveform_host_bytes,
            mux_output_host_bytes_bound,
            aac_mux_staging_host_bytes,
            predicted_host_increment_bytes,
            fixed_runtime_device_bytes,
            fixed_transformer_device_bytes,
            visual_vae_resident_device_bytes,
            audio_vae_resident_device_bytes,
            attempt_resident_vae_device_bytes,
            vae_construction_device_workspace_bytes,
        ] {
            hash.update(value.to_le_bytes());
        }
        update_string(hash, vae_memory_evidence_identity_sha256);
        for value in [
            qwen_device_parameter_bytes,
            qwen_activation_device_bytes,
            qwen_output_state_device_bytes,
            qwen_output_transfer_device_bytes,
            condition_vae_workspace_device_bytes,
            condition_latent_backing_device_bytes,
            target_video_latent_device_bytes,
            target_audio_latent_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
        ] {
            hash.update(value.to_le_bytes());
        }
        hash.update(match denoise_copy_policy {
            H3FactoryTargetDenoiseCopyPolicy::CandleF32PairedResMultistepV2 => {
                b"candle-f32-paired-res-multistep-v2".as_slice()
            }
            #[cfg(test)]
            H3FactoryTargetDenoiseCopyPolicy::IdentityMutationSentinel => {
                b"identity-mutation-sentinel".as_slice()
            }
        });
        for value in [
            denoise_tensor_copy_workspace_device_bytes,
            audio_waveform_device_bytes,
            attention_workspace_device_bytes,
            ffn_workspace_device_bytes,
            decoder_tile_workspace_device_bytes,
            audio_decode_workspace_device_bytes,
            resident_block_device_bytes,
            streamed_block_device_bytes,
            prefetch_device_bytes,
            dequantization_workspace_device_bytes,
            protected_block_device_bytes,
            streamed_block_device_overlap_bytes,
            max_device_weight_staging_bytes,
            fixed_transformer_load_device_staging_bytes,
            vae_load_phase_device_bytes,
            qwen_encode_phase_device_bytes,
            qwen_transfer_phase_device_bytes,
            condition_encode_phase_device_bytes,
            noise_allocation_phase_device_bytes,
            transformer_load_phase_device_bytes,
            denoise_phase_device_bytes,
            visual_decode_phase_device_bytes,
            audio_decode_phase_device_bytes,
            waveform_transfer_phase_device_bytes,
            mux_phase_device_bytes,
            predicted_device_peak_bytes,
        ] {
            hash.update(value.to_le_bytes());
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryAttentionInput {
    pub generic_backend: AttentionBackend,
    pub generic_chunk: AttentionChunkPolicy,
    pub runtime_backend: H3AttentionBackend,
    pub kernel: H3AttentionKernel,
    pub activation: H3AttentionActivation,
    pub device: H3AttentionDevice,
    pub model_contract: H3AttentionModelContract,
    pub runtime_identity_sha256: String,
    pub qualification_kernel_identity: String,
    pub qualification_sha256: String,
    pub full_noncausal: bool,
    pub lossless: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Cloneable immutable value record for one prepared target attempt.
///
/// This is never an execution, lease, replay-prevention, or cancellation root.
/// The future non-`Clone` consuming scheduler root remains an explicit
/// activation prerequisite.
pub struct H3FactoryPreparedAttemptInput {
    pub identity_sha256: String,
    pub execution_fingerprint: String,
    pub request: H3FactoryPreparedRequestInput,
    pub raw_checkpoint: H3FactoryRawCheckpointInput,
    pub target_budget: H3FactoryTargetBudgetInput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Frozen cloneable value record, not behavior ownership or a scheduler lease.
pub(crate) struct H3FactoryPreparedAttemptAuthority {
    pub(crate) execution_fingerprint: String,
    pub(crate) request: H3FactoryPreparedRequestInput,
    pub(crate) raw_checkpoint: H3FactoryRawCheckpointInput,
    pub(crate) target_budget: H3FactoryTargetBudgetInput,
    pub(crate) identity_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H3FactoryQuantizationAuthority {
    OfficialBf16,
    ComfyPrunedInt8ConvrotNvfp4Awq {
        transformer_policy_sha256: String,
        qwen_policy_sha256: String,
        pruned_adaln_table_sha256: String,
    },
}

impl H3FactoryQuantizationAuthority {
    fn validate(&self, layout: Layout) -> Result<()> {
        match (self, layout) {
            (Self::OfficialBf16, Layout::OfficialBf16) => Ok(()),
            (
                Self::ComfyPrunedInt8ConvrotNvfp4Awq {
                    transformer_policy_sha256,
                    qwen_policy_sha256,
                    pruned_adaln_table_sha256,
                },
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
            ) => {
                require_sha256(transformer_policy_sha256, "H3 transformer quantization")?;
                require_sha256(qwen_policy_sha256, "H3 Qwen quantization")?;
                require_sha256(pruned_adaln_table_sha256, "H3 pruned AdaLN table")
            }
            _ => bail!("MiniMax H3 layout and quantization authorities disagree"),
        }
    }

    fn update_identity(&self, hash: &mut Sha256) {
        match self {
            Self::OfficialBf16 => hash.update(b"official-bf16"),
            Self::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256,
                qwen_policy_sha256,
                pruned_adaln_table_sha256,
            } => {
                hash.update(b"comfy-pruned-int8-convrot-nvfp4-awq\0");
                hash.update(transformer_policy_sha256.as_bytes());
                hash.update([0]);
                hash.update(qwen_policy_sha256.as_bytes());
                hash.update([0]);
                hash.update(pruned_adaln_table_sha256.as_bytes());
            }
        }
    }
}

/// Scheduler accounting projection for an admitted attempt.
///
/// This binds content and reserved byte ceilings only. It is deliberately not
/// replay protection and does not prove same-slot execution: the execution
/// fingerprint is reusable across warm-cache work. Activation must introduce
/// a non-`Clone`, one-shot scheduler-lease root with a fresh attempt nonce,
/// exact host/device/ordinal, and cancellation-slot identity before allocating
/// any component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryExecutionBudgetEchoInput {
    pub prepared_attempt_identity_sha256: String,
    pub device_peak_bytes: u64,
    pub host_increment_bytes: u64,
}

/// Server-owned inputs that may cross into the H3 factory boundary.
///
/// The constructor validates and fingerprints every field before retaining
/// it. In particular, component authorities are digests only; paths and model
/// bytes remain on the compliance-gated side of the boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryAuthorityInput {
    pub model: String,
    pub device_id: String,
    pub device_ordinal: usize,
    pub compute_capability: (u16, u16),
    pub execution_fingerprint: String,
    pub conditioner_placement: H3FactoryConditionerPlacement,
    pub qwen_parameter_bytes: u64,
    pub qwen_host_resident_parameter_bytes: u64,
    pub qwen_device_resident_parameter_bytes: u64,
    pub qwen_activation_workspace_bytes: u64,
    pub qwen_output_text_rows: u64,
    pub qwen_vision_rows: u64,
    pub condition_visual_rows: u64,
    pub resident_block_count: u32,
    pub prefetch_depth: u32,
    pub attention_backend: AttentionBackend,
    pub attention_chunk: AttentionChunkPolicy,
    pub attention_kernel_identity: String,
    pub attention_qualification_sha256: String,
    pub attention_full_noncausal: bool,
    pub attention_lossless: bool,
    pub attention_head_count: u32,
    pub attention_head_dim: u32,
    /// Exact typed runtime tuple. Existing contract-only callers supply
    /// `None`; activation authority requires `Some` and relationally binds it
    /// to the independent qualification projection above.
    pub attention_runtime: Option<H3FactoryAttentionInput>,
    pub block_offload: bool,
    pub quantization: H3FactoryQuantizationAuthority,
    /// Prepared target-budget records. Existing production admission deliberately
    /// supplies `None` until it owns normalized endpoints and opened-file
    /// evidence; no placeholder values are accepted.
    pub prepared_attempt: Option<H3FactoryPreparedAttemptInput>,
    pub execution_budget_echo: Option<H3FactoryExecutionBudgetEchoInput>,
    pub components: Vec<H3FactoryComponentAuthority>,
}

/// Immutable, contract-only authority carried by [`crate::FrozenEngineConfig`].
///
/// Its private backend plan always lacks executable attention/quantization
/// authorities. `validate_for_dispatch` also requires both the public capability
/// contract and production family registry before a future loader can be reached.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenH3FactoryAuthority {
    schema_version: u32,
    backend_plan: FrozenH3Fl2VaCandlePlan,
    comfy_vae_artifact_plan_identity_sha256: Option<String>,
    device_ordinal: usize,
    conditioner_placement: H3FactoryConditionerPlacement,
    qwen_parameter_bytes: u64,
    qwen_host_resident_parameter_bytes: u64,
    qwen_device_resident_parameter_bytes: u64,
    qwen_activation_workspace_bytes: u64,
    qwen_output_text_rows: u64,
    qwen_vision_rows: u64,
    condition_visual_rows: u64,
    attention_backend: AttentionBackend,
    attention_chunk: AttentionChunkPolicy,
    attention_kernel_identity: String,
    attention_qualification_sha256: String,
    attention_full_noncausal: bool,
    attention_lossless: bool,
    attention_head_count: u32,
    attention_head_dim: u32,
    attention_runtime: Option<H3FactoryAttentionInput>,
    block_offload: bool,
    quantization: H3FactoryQuantizationAuthority,
    prepared_attempt: Option<H3FactoryPreparedAttemptAuthority>,
    execution_budget_echo: Option<H3FactoryExecutionBudgetEchoInput>,
    identity_sha256: String,
}

/// Private-only projection of the exact admission record needed to compose
/// the opened-file Comfy VAEs with one already-authorized component backend.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateVaeFactoryAuthority {
    pub(crate) factory_identity_sha256: String,
    pub(crate) backend_plan_identity_sha256: String,
    pub(crate) vae_artifact_plan_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
    pub(crate) canonical_model: String,
    pub(crate) task: Task,
    pub(crate) device_id: String,
    pub(crate) execution_fingerprint: String,
}

/// Complete private projection consumed by the VAE-free streamed core. It is
/// still contract-only: no artifact path, opened descriptor, runtime object,
/// or public dispatch authority crosses this boundary.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateFl2VaFactoryAuthority {
    pub(crate) factory_identity_sha256: String,
    pub(crate) backend_plan_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
    pub(crate) vae_artifact_plan_identity_sha256: String,
    pub(crate) canonical_model: String,
    pub(crate) task: Task,
    pub(crate) device_id: String,
    pub(crate) device_ordinal: usize,
    pub(crate) compute_capability: (u16, u16),
    pub(crate) execution_fingerprint: String,
    pub(crate) condition_visual_rows: u64,
    pub(crate) block_streaming: FrozenH3BlockStreamingPlan,
    pub(crate) attention: H3FactoryAttentionInput,
    pub(crate) quantization: H3FactoryQuantizationAuthority,
    pub(crate) conditioner_component_content_sha256: String,
    pub(crate) conditioner_component_validation_sha256: String,
    pub(crate) transformer_component_content_sha256: String,
    pub(crate) transformer_component_validation_sha256: String,
    pub(crate) visual_vae_component_content_sha256: String,
    pub(crate) visual_vae_component_validation_sha256: String,
    pub(crate) audio_vae_component_content_sha256: String,
    pub(crate) audio_vae_component_validation_sha256: String,
}

#[derive(Clone, Copy)]
struct H3FactoryPreparedAttemptProjection<'a> {
    execution_fingerprint: &'a str,
    qwen_output_text_rows: u64,
    qwen_vision_rows: u64,
    condition_visual_rows: u64,
    resident_block_count: u32,
    prefetch_depth: u32,
    qwen_activation_workspace_bytes: u64,
    qwen_device_parameter_bytes: u64,
    qwen_host_parameter_bytes: u64,
    conditioner_placement: H3FactoryConditionerPlacement,
}

impl H3FactoryPreparedAttemptAuthority {
    fn freeze(
        input: H3FactoryPreparedAttemptInput,
        projection: H3FactoryPreparedAttemptProjection<'_>,
    ) -> Result<Self> {
        let authority = Self {
            execution_fingerprint: input.execution_fingerprint,
            request: input.request,
            raw_checkpoint: input.raw_checkpoint,
            target_budget: input.target_budget,
            identity_sha256: input.identity_sha256,
        };
        authority.validate(projection)?;
        Ok(authority)
    }

    fn validate(&self, projection: H3FactoryPreparedAttemptProjection<'_>) -> Result<()> {
        require_sha256(&self.execution_fingerprint, "H3 prepared execution")?;
        require_sha256(&self.identity_sha256, "H3 prepared attempt authority")?;
        validate_prepared_request(&self.request)?;
        validate_raw_checkpoint(&self.raw_checkpoint)?;
        validate_target_budget(
            &self.target_budget,
            &self.request,
            &self.raw_checkpoint,
            projection.resident_block_count,
            projection.prefetch_depth,
            projection.conditioner_placement,
        )?;
        if self.execution_fingerprint != projection.execution_fingerprint
            || self.request.rows.qwen_output_text_rows != projection.qwen_output_text_rows
            || self.request.rows.qwen_vision_rows != projection.qwen_vision_rows
            || self.request.rows.condition_visual_rows != projection.condition_visual_rows
            || self.target_budget.qwen_device_parameter_bytes
                != projection.qwen_device_parameter_bytes
            || self.target_budget.qwen_host_parameter_bytes != projection.qwen_host_parameter_bytes
            || match projection.conditioner_placement {
                H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
                    self.target_budget.qwen_activation_device_bytes
                        != projection.qwen_activation_workspace_bytes
                }
                H3FactoryConditionerPlacement::HostCpuThenDrop => {
                    self.target_budget.qwen_host_activation_bytes
                        != projection.qwen_activation_workspace_bytes
                }
            }
            || self.identity_sha256 != expected_prepared_attempt_identity(self)
        {
            bail!("MiniMax H3 prepared request or target budget changed after admission");
        }
        Ok(())
    }
}

fn validate_prepared_request(request: &H3FactoryPreparedRequestInput) -> Result<()> {
    for (value, label) in [
        (&request.identity_sha256, "H3 prepared request"),
        (&request.prompt_sha256, "H3 prepared prompt"),
        (
            &request.conditioning_fingerprint,
            "H3 prepared conditioning",
        ),
        (&request.reference_fingerprint, "H3 prepared references"),
    ] {
        require_sha256(value, label)?;
    }
    let model_contract = contract::capability_contract_for_model(&request.canonical_model)
        .ok_or_else(|| anyhow!("H3 prepared request names an unknown model"))?;
    let expected_anchors: &[H3FactoryEndpointAnchor] = match request.mode {
        Mode::TextToAudioVideo | Mode::ReferenceToAudioVideo => &[],
        Mode::FirstFrameToAudioVideo => &[H3FactoryEndpointAnchor::First],
        Mode::LastFrameToAudioVideo => &[H3FactoryEndpointAnchor::Last],
        Mode::FirstAndLastFrameToAudioVideo => &[
            H3FactoryEndpointAnchor::First,
            H3FactoryEndpointAnchor::Last,
        ],
    };
    let actual_anchors = request
        .endpoints
        .iter()
        .map(|endpoint| endpoint.anchor)
        .collect::<Vec<_>>();
    let mode_matches_task = matches!(
        (request.task, request.mode),
        (
            Task::Fl2va,
            Mode::TextToAudioVideo
                | Mode::FirstFrameToAudioVideo
                | Mode::LastFrameToAudioVideo
                | Mode::FirstAndLastFrameToAudioVideo
        ) | (Task::Ref2va, Mode::ReferenceToAudioVideo)
    );
    let expected_endpoint_bytes = u64::from(request.width)
        .checked_mul(u64::from(request.height))
        .and_then(|bytes| bytes.checked_mul(3))
        .ok_or_else(|| anyhow!("MiniMax H3 normalized endpoint bytes overflow"))?;
    for endpoint in &request.endpoints {
        require_sha256(&endpoint.encoded_content_sha256, "H3 endpoint content")?;
        require_sha256(
            &endpoint.normalized_cpu_content_sha256,
            "H3 normalized endpoint content",
        )?;
        if endpoint.encoded_bytes == 0 || endpoint.normalized_cpu_bytes != expected_endpoint_bytes {
            bail!("MiniMax H3 endpoint descriptor or normalized CPU charge is invalid");
        }
        if endpoint.preprocess != H3FactoryEndpointPreprocess::PillowLanczosRgbU8CpuV1
            || endpoint.normalized_shape != [1, 3, 1, request.height, request.width]
        {
            bail!("MiniMax H3 endpoint preprocessing contract changed after admission");
        }
    }
    let frames = u64::from(request.frames);
    let expected_video_latent_frames = frames
        .checked_sub(u64::from(contract::FRAME_OFFSET))
        .ok_or_else(|| anyhow!("MiniMax H3 video latent frames underflow"))?
        / u64::from(contract::FRAME_STEP)
        * 5
        + 2;
    let rows_per_video_latent = u64::from(request.width / 32)
        .checked_mul(u64::from(request.height / 32))
        .ok_or_else(|| anyhow!("MiniMax H3 rows per video latent overflow"))?;
    let expected_target_video_rows = expected_video_latent_frames
        .checked_mul(rows_per_video_latent)
        .ok_or_else(|| anyhow!("MiniMax H3 target video rows overflow"))?;
    let expected_audio_latents = frames
        .checked_mul(5)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| anyhow!("MiniMax H3 audio latent rows overflow"))?
        / 3;
    let expected_audio_rows = expected_audio_latents
        .checked_mul(u64::from(contract::AUDIO_CHANNELS))
        .ok_or_else(|| anyhow!("MiniMax H3 target audio rows overflow"))?;
    let expected_audio_samples = expected_audio_latents
        .checked_mul(800)
        .ok_or_else(|| anyhow!("MiniMax H3 audio samples overflow"))?;
    if !(2..=H3_FACTORY_MAX_GRID_POINTS).contains(&request.grid_points) {
        bail!("MiniMax H3 target budget supports 2..={H3_FACTORY_MAX_GRID_POINTS} grid points");
    }
    let schedule = H3DualSchedule::new(
        usize::try_from(request.grid_points)
            .map_err(|_| anyhow!("MiniMax H3 grid points exceed usize"))?,
    )?;
    let expected_denoise_forward_count =
        u32::try_from(schedule.counts().transformer_evaluations)
            .map_err(|_| anyhow!("MiniMax H3 denoise count exceeds u32"))?;
    let packed_rows = checked_u64_sum(
        [
            request.rows.qwen_output_text_rows,
            request.rows.condition_visual_rows,
            request.rows.condition_audio_rows,
            request.rows.target_video_rows,
            request.rows.target_audio_rows,
        ],
        "H3 packed rows",
    )?;
    let expected_condition_rows = u64::try_from(request.endpoints.len())
        .map_err(|_| anyhow!("MiniMax H3 endpoint count exceeds u64"))?
        .checked_mul(rows_per_video_latent)
        .ok_or_else(|| anyhow!("MiniMax H3 condition rows overflow"))?;
    let pixel_count = u64::from(request.width)
        .checked_mul(u64::from(request.height))
        .ok_or_else(|| anyhow!("MiniMax H3 pixel count overflow"))?;
    let aspect_ratio = request.width as f64 / request.height as f64;
    if request.canonical_model != model_contract.canonical_model
        || request.task != model_contract.task
        || request.task != Task::Fl2va
        || !mode_matches_task
        || actual_anchors != expected_anchors
        || request.denoise_forward_count != expected_denoise_forward_count
        || request.guidance_f64_bits != 0.0f64.to_bits()
        || request.strength_f64_bits != 1.0f64.to_bits()
        || request.batch_size != 1
        || request.fps != contract::FIXED_FPS
        || !request.synchronized_audio
        || !request.mp4_output
        || request.width == 0
        || request.height == 0
        || !request.width.is_multiple_of(contract::DIMENSION_ALIGNMENT)
        || !request.height.is_multiple_of(contract::DIMENSION_ALIGNMENT)
        || pixel_count > contract::MAX_PIXELS
        || !(contract::MIN_ASPECT_RATIO..=contract::MAX_ASPECT_RATIO).contains(&aspect_ratio)
        || !contract::valid_frame_count(request.frames)
        || request.video_latent_frames != expected_video_latent_frames
        || request.audio_latents_per_channel != expected_audio_latents
        || request.audio_samples_per_channel != expected_audio_samples
        || request.rows.qwen_output_text_rows == 0
        || request.rows.qwen_output_text_rows > H3_FACTORY_QWEN_MODEL_MAX_ROWS
        || request.rows.condition_audio_rows != 0
        || request.rows.target_video_rows != expected_target_video_rows
        || request.rows.target_audio_rows != expected_audio_rows
        || request.rows.total_packed_rows != packed_rows
        || request.task == Task::Fl2va
            && request.rows.condition_visual_rows != expected_condition_rows
        || request.identity_sha256 != expected_prepared_request_identity(request)
    {
        bail!("MiniMax H3 prepared request authority is internally inconsistent");
    }
    Ok(())
}

fn validate_raw_checkpoint(checkpoint: &H3FactoryRawCheckpointInput) -> Result<()> {
    for (value, label) in [
        (&checkpoint.identity_sha256, "H3 raw checkpoint"),
        (&checkpoint.raw_content_sha256, "H3 raw checkpoint content"),
        (
            &checkpoint.raw_header_identity_sha256,
            "H3 raw checkpoint header",
        ),
        (
            &checkpoint.opened_checkpoint_identity_sha256,
            "H3 opened checkpoint",
        ),
        (
            &checkpoint.quantization_policy_identity_sha256,
            "H3 raw checkpoint quantization",
        ),
        (&checkpoint.config_identity_sha256, "H3 checkpoint config"),
    ] {
        require_sha256(value, label)?;
    }
    if checkpoint.blocks.len() != 50
        || checkpoint.verified_file_bytes == 0
        || checkpoint.fixed_transformer_encoded_host_bytes == 0
        || checkpoint.fixed_transformer_protected_device_bytes == 0
        || checkpoint.fixed_transformer_max_host_read_staging_bytes == 0
        || checkpoint.fixed_transformer_max_device_weight_staging_bytes == 0
    {
        bail!("MiniMax H3 raw checkpoint requires one exact 50-block memory vector");
    }
    let encoded_bytes = checked_u64_sum(
        checkpoint
            .blocks
            .iter()
            .map(|block| block.encoded_host_bytes),
        "H3 encoded block bytes",
    )?;
    for (index, block) in checkpoint.blocks.iter().enumerate() {
        require_sha256(&block.content_sha256, "H3 raw checkpoint block")?;
        if usize::from(block.index) != index
            || block.encoded_host_bytes == 0
            || block.protected_device_bytes == 0
            || block.max_device_weight_staging_bytes == 0
            || block.max_host_read_staging_bytes == 0
        {
            bail!("MiniMax H3 raw checkpoint block memory facts are incomplete or unordered");
        }
    }
    let authenticated_encoded_bytes = checkpoint
        .fixed_transformer_encoded_host_bytes
        .checked_add(encoded_bytes)
        .ok_or_else(|| anyhow!("MiniMax H3 authenticated checkpoint bytes overflow"))?;
    if checkpoint.verified_file_bytes < authenticated_encoded_bytes
        || checkpoint.identity_sha256 != expected_raw_checkpoint_identity(checkpoint)
    {
        bail!("MiniMax H3 raw checkpoint identity or file charge is inconsistent");
    }
    Ok(())
}

fn validate_target_budget(
    memory: &H3FactoryTargetBudgetInput,
    request: &H3FactoryPreparedRequestInput,
    checkpoint: &H3FactoryRawCheckpointInput,
    resident_block_count: u32,
    prefetch_depth: u32,
    conditioner_placement: H3FactoryConditionerPlacement,
) -> Result<()> {
    require_sha256(&memory.identity_sha256, "H3 target budget")?;
    let resident_count = usize::try_from(resident_block_count)
        .map_err(|_| anyhow!("MiniMax H3 resident block count exceeds usize"))?;
    let prefetch_depth = usize::try_from(prefetch_depth)
        .map_err(|_| anyhow!("MiniMax H3 prefetch depth exceeds usize"))?;
    if resident_count != 0 || prefetch_depth != 0 {
        bail!("MiniMax H3 target budget requires fully streamed blocks without prefetch");
    }
    let retained_vaes = memory
        .visual_vae_resident_device_bytes
        .checked_add(memory.audio_vae_resident_device_bytes)
        .ok_or_else(|| anyhow!("MiniMax H3 VAE residency overflow"))?;
    let normalized_endpoint_bytes = checked_u64_sum(
        request
            .endpoints
            .iter()
            .map(|endpoint| endpoint.normalized_cpu_bytes),
        "H3 normalized endpoint bytes",
    )?;
    let endpoint_encoded_bytes = checked_u64_sum(
        request
            .endpoints
            .iter()
            .map(|endpoint| endpoint.encoded_bytes),
        "H3 encoded endpoint bytes",
    )?;
    let protected_blocks = checked_u64_sum(
        checkpoint
            .blocks
            .iter()
            .map(|block| block.protected_device_bytes),
        "H3 protected block bytes",
    )?;
    let resident_blocks = checked_u64_sum(
        checkpoint
            .blocks
            .iter()
            .take(resident_count)
            .map(|block| block.protected_device_bytes),
        "H3 resident block bytes",
    )?;
    let streamed_blocks = checked_u64_sum(
        checkpoint
            .blocks
            .iter()
            .skip(resident_count)
            .map(|block| block.protected_device_bytes),
        "H3 streamed block bytes",
    )?;
    let streamed = &checkpoint.blocks[resident_count..];
    let prefetch_slots = prefetch_depth.saturating_add(1);
    let expected_prefetch = if prefetch_depth == 0 || streamed.is_empty() {
        0
    } else {
        streamed
            .windows(prefetch_slots.min(streamed.len()).max(1))
            .map(|window| {
                checked_u64_sum(
                    window.iter().map(|block| block.protected_device_bytes),
                    "H3 prefetch window",
                )
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0)
    };
    let streamed_block_overlap = streamed
        .iter()
        .map(|block| block.protected_device_bytes)
        .max()
        .unwrap_or(0);
    let max_device_staging = checkpoint
        .blocks
        .iter()
        .map(|block| block.max_device_weight_staging_bytes)
        .max()
        .unwrap_or(0);
    let max_host_staging = checkpoint
        .blocks
        .iter()
        .map(|block| block.max_host_read_staging_bytes)
        .max()
        .unwrap_or(0);
    let max_streamed_block_host_overlap = checkpoint
        .blocks
        .iter()
        .map(|block| {
            block
                .encoded_host_bytes
                .checked_add(block.max_host_read_staging_bytes)
                .ok_or_else(|| anyhow!("H3 streamed block host overlap overflow"))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let predicted_host = checked_u64_sum(
        [
            memory.artifact_host_bytes,
            memory.fixed_runtime_host_bytes,
            memory.qwen_host_workspace_bytes,
            memory.condition_backing_host_bytes,
            memory.endpoint_encoded_host_bytes,
            memory.normalized_endpoint_host_bytes,
            memory.schedule_host_bytes,
            memory.packed_layout_host_bytes,
            memory.packed_layout_construction_staging_host_bytes,
            memory.packed_layout_freeze_staging_host_bytes,
            memory.text_modality_tags_host_bytes,
            memory.noise_cpu_staging_host_bytes,
            memory.vae_peak_host_io_buffer_bytes,
            memory.vae_peak_host_mapped_file_bytes,
            memory.max_streamed_block_host_overlap_bytes,
            memory.fixed_transformer_load_host_staging_bytes,
            memory.encoded_video_host_bytes_bound,
            memory.thumbnail_host_bytes_bound,
            memory.waveform_host_bytes,
            memory.mux_output_host_bytes_bound,
            memory.aac_mux_staging_host_bytes,
        ],
        "H3 host increment",
    )?;
    let qwen_host_workspace = memory
        .qwen_host_parameter_bytes
        .checked_add(memory.qwen_host_activation_bytes)
        .and_then(|bytes| bytes.checked_add(memory.qwen_host_output_state_bytes))
        .ok_or_else(|| anyhow!("MiniMax H3 Qwen host workspace overflow"))?;
    let schedule_host_bytes = u64::from(request.grid_points)
        .checked_mul(16)
        .ok_or_else(|| anyhow!("MiniMax H3 host schedule budget overflow"))?;
    let packed_layout_host_bytes = request
        .rows
        .total_packed_rows
        .checked_mul(24)
        .ok_or_else(|| anyhow!("MiniMax H3 host packed-layout budget overflow"))?;
    let packed_layout_construction_staging_host_bytes = request
        .rows
        .total_packed_rows
        .checked_mul(16)
        .ok_or_else(|| anyhow!("MiniMax H3 packed-layout construction staging overflow"))?;
    let packed_layout_freeze_staging_host_bytes = request
        .rows
        .total_packed_rows
        .checked_mul(12)
        .ok_or_else(|| anyhow!("MiniMax H3 packed-layout freeze staging overflow"))?;
    let text_modality_tags_host_bytes = request
        .rows
        .qwen_output_text_rows
        .checked_mul(8)
        .ok_or_else(|| anyhow!("MiniMax H3 text modality tags budget overflow"))?;
    if memory.artifacts.is_empty()
        || memory
            .artifacts
            .windows(2)
            .any(|pair| (pair[0].role, pair[0].index) >= (pair[1].role, pair[1].index))
    {
        bail!("MiniMax H3 authenticated artifact bytes must be nonempty and strictly ordered");
    }
    for artifact in &memory.artifacts {
        require_sha256(&artifact.content_sha256, "H3 authenticated host artifact")?;
        if artifact.bytes == 0 {
            bail!("MiniMax H3 authenticated host artifact has zero bytes");
        }
    }
    for required in [
        H3FactoryArtifactHostRole::Conditioner,
        H3FactoryArtifactHostRole::RawTransformerCheckpoint,
        H3FactoryArtifactHostRole::VisualVae,
        H3FactoryArtifactHostRole::AudioVae,
    ] {
        if !memory
            .artifacts
            .iter()
            .any(|artifact| artifact.role == required)
        {
            bail!("MiniMax H3 authenticated host artifact vector is incomplete");
        }
    }
    let authenticated_artifact_bytes = checked_u64_sum(
        memory.artifacts.iter().map(|artifact| artifact.bytes),
        "H3 authenticated artifact bytes",
    )?;
    let raw_artifacts = memory
        .artifacts
        .iter()
        .filter(|artifact| artifact.role == H3FactoryArtifactHostRole::RawTransformerCheckpoint)
        .collect::<Vec<_>>();
    if raw_artifacts.len() != 1
        || raw_artifacts[0].index != 0
        || raw_artifacts[0].content_sha256 != checkpoint.raw_content_sha256
        || raw_artifacts[0].bytes != checkpoint.verified_file_bytes
    {
        bail!("MiniMax H3 raw checkpoint host artifact does not match opened-file authority");
    }
    let vae_load = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.vae_construction_device_workspace_bytes,
        ],
        "H3 VAE load phase",
    )?;
    let qwen_encode = match conditioner_placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop => checked_u64_sum(
            [
                memory.fixed_runtime_device_bytes,
                retained_vaes,
                memory.qwen_device_parameter_bytes,
                memory.qwen_activation_device_bytes,
                memory.qwen_output_state_device_bytes,
            ],
            "H3 Qwen encode phase",
        )?,
        H3FactoryConditionerPlacement::HostCpuThenDrop => checked_u64_sum(
            [memory.fixed_runtime_device_bytes, retained_vaes],
            "H3 host Qwen encode phase",
        )?,
    };
    let qwen_transfer = match conditioner_placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop => 0,
        H3FactoryConditionerPlacement::HostCpuThenDrop => checked_u64_sum(
            [
                memory.fixed_runtime_device_bytes,
                retained_vaes,
                memory.qwen_output_transfer_device_bytes,
            ],
            "H3 Qwen transfer phase",
        )?,
    };
    let condition_encode = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.qwen_output_state_device_bytes,
            memory.condition_vae_workspace_device_bytes,
            memory.condition_latent_backing_device_bytes,
            memory.packed_layout_device_bytes,
        ],
        "H3 condition encode phase",
    )?;
    let noise_allocation = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.qwen_output_state_device_bytes,
            memory.condition_latent_backing_device_bytes,
            memory.condition_latent_backing_device_bytes,
            memory.packed_layout_device_bytes,
            memory.target_video_latent_device_bytes,
            memory.target_video_latent_device_bytes,
            memory.target_audio_latent_device_bytes,
            memory.target_audio_latent_device_bytes,
            memory.packed_video_state_device_bytes,
            memory.packed_audio_state_device_bytes,
        ],
        "H3 noise allocation phase",
    )?;
    let transformer_load = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.fixed_transformer_device_bytes,
            memory.qwen_output_state_device_bytes,
            memory.condition_latent_backing_device_bytes,
            memory.packed_layout_device_bytes,
            memory.packed_video_state_device_bytes,
            memory.packed_audio_state_device_bytes,
            resident_blocks,
            memory.fixed_transformer_load_device_staging_bytes,
        ],
        "H3 transformer load phase",
    )?;
    let denoise_copy_workspace = memory
        .packed_video_state_device_bytes
        .checked_add(memory.packed_audio_state_device_bytes)
        .and_then(|bytes| bytes.checked_mul(9))
        .ok_or_else(|| anyhow!("H3 paired RES multistep copy budget overflow"))?;
    let denoise = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.fixed_transformer_device_bytes,
            memory.qwen_output_state_device_bytes,
            memory.condition_latent_backing_device_bytes,
            memory.packed_layout_device_bytes,
            memory.packed_video_state_device_bytes,
            memory.packed_audio_state_device_bytes,
            memory.denoise_tensor_copy_workspace_device_bytes,
            memory.attention_workspace_device_bytes,
            memory.ffn_workspace_device_bytes,
            resident_blocks,
            streamed_block_overlap,
            expected_prefetch,
            max_device_staging,
        ],
        "H3 denoise phase",
    )?;
    let visual_decode = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.packed_video_state_device_bytes,
            memory.packed_audio_state_device_bytes,
            memory.target_video_latent_device_bytes,
            memory.target_audio_latent_device_bytes,
            memory.decoder_tile_workspace_device_bytes,
        ],
        "H3 visual decode phase",
    )?;
    let audio_decode = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.packed_video_state_device_bytes,
            memory.packed_audio_state_device_bytes,
            memory.target_audio_latent_device_bytes,
            memory.audio_decode_workspace_device_bytes,
            memory.audio_waveform_device_bytes,
        ],
        "H3 audio decode phase",
    )?;
    let waveform_transfer = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.audio_waveform_device_bytes,
        ],
        "H3 waveform transfer phase",
    )?;
    let predicted_device = [
        vae_load,
        qwen_encode,
        qwen_transfer,
        condition_encode,
        noise_allocation,
        transformer_load,
        denoise,
        visual_decode,
        audio_decode,
        waveform_transfer,
        0,
    ]
    .into_iter()
    .max()
    .unwrap_or_default();
    let condition_bytes = (
        memory.condition_backing_host_bytes,
        memory.condition_latent_backing_device_bytes,
    );
    if memory.load_drop_policy
        != H3FactoryTargetLoadDropPolicy::LoadVaesLoadQwenEncodeTransferDropQwenEncodeConditionsAllocateNoiseLoadTransformerDenoiseDropTransformerDecodeVisualAudioDropVaesMux
        || memory.artifact_host_bytes != authenticated_artifact_bytes
        || memory.fixed_runtime_host_bytes == 0
        || memory.fixed_runtime_device_bytes == 0
        || memory.fixed_transformer_device_bytes
            != checkpoint.fixed_transformer_protected_device_bytes
        || memory.visual_vae_resident_device_bytes == 0
        || memory.audio_vae_resident_device_bytes == 0
        || memory.vae_construction_device_workspace_bytes == 0
        || require_sha256(
            &memory.vae_memory_evidence_identity_sha256,
            "H3 VAE memory evidence",
        )
        .is_err()
        || memory.target_video_latent_device_bytes == 0
        || memory.target_audio_latent_device_bytes == 0
        || memory.attempt_resident_vae_device_bytes != retained_vaes
        || memory.qwen_host_workspace_bytes != qwen_host_workspace
        || memory.endpoint_encoded_host_bytes != endpoint_encoded_bytes
        || memory.normalized_endpoint_host_bytes != normalized_endpoint_bytes
        || memory.schedule_host_bytes != schedule_host_bytes
        || memory.packed_layout_host_bytes != packed_layout_host_bytes
        || memory.packed_layout_construction_staging_host_bytes
            != packed_layout_construction_staging_host_bytes
        || memory.packed_layout_freeze_staging_host_bytes != packed_layout_freeze_staging_host_bytes
        || memory.text_modality_tags_host_bytes != text_modality_tags_host_bytes
        || memory.protected_block_device_bytes != protected_blocks
        || memory.resident_block_device_bytes != resident_blocks
        || memory.streamed_block_device_bytes != streamed_blocks
        || memory.prefetch_device_bytes != expected_prefetch
        || memory.streamed_block_device_overlap_bytes != streamed_block_overlap
        || memory.dequantization_workspace_device_bytes != max_device_staging
        || memory.max_device_weight_staging_bytes != max_device_staging
        || memory.max_host_read_staging_bytes != max_host_staging
        || memory.max_streamed_block_host_overlap_bytes != max_streamed_block_host_overlap
        || memory.fixed_transformer_load_host_staging_bytes
            != checkpoint
                .fixed_transformer_encoded_host_bytes
                .checked_add(checkpoint.fixed_transformer_max_host_read_staging_bytes)
                .ok_or_else(|| anyhow!("H3 fixed transformer host staging overflow"))?
        || memory.fixed_transformer_load_device_staging_bytes
            != checkpoint.fixed_transformer_max_device_weight_staging_bytes
        || memory.predicted_host_increment_bytes != predicted_host
        || memory.vae_load_phase_device_bytes != vae_load
        || memory.qwen_encode_phase_device_bytes != qwen_encode
        || memory.qwen_transfer_phase_device_bytes != qwen_transfer
        || memory.condition_encode_phase_device_bytes != condition_encode
        || memory.noise_allocation_phase_device_bytes != noise_allocation
        || memory.transformer_load_phase_device_bytes != transformer_load
        || memory.denoise_phase_device_bytes != denoise
        || memory.visual_decode_phase_device_bytes != visual_decode
        || memory.audio_decode_phase_device_bytes != audio_decode
        || memory.waveform_transfer_phase_device_bytes != waveform_transfer
        || memory.mux_phase_device_bytes != 0
        || memory.predicted_device_peak_bytes != predicted_device
        || match conditioner_placement {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
                memory.qwen_host_parameter_bytes == 0
                    || memory.qwen_host_activation_bytes != 0
                    || memory.qwen_host_output_state_bytes != 0
                    || memory.qwen_device_parameter_bytes == 0
                    || memory.qwen_activation_device_bytes == 0
                    || memory.qwen_output_transfer_device_bytes != 0
            }
            H3FactoryConditionerPlacement::HostCpuThenDrop => {
                memory.qwen_host_parameter_bytes == 0
                    || memory.qwen_host_activation_bytes == 0
                    || memory.qwen_host_output_state_bytes != memory.qwen_output_state_device_bytes
                    || memory.qwen_device_parameter_bytes != 0
                    || memory.qwen_activation_device_bytes != 0
                    || memory.qwen_output_transfer_device_bytes
                        != memory.qwen_output_state_device_bytes
            }
        }
        || memory.target_video_latent_device_bytes
            != request
                .rows
                .target_video_rows
                .checked_mul(96 * 4)
                .ok_or_else(|| anyhow!("H3 target video latent bytes overflow"))?
        || memory.target_audio_latent_device_bytes
            != request
                .rows
                .target_audio_rows
                .checked_mul(32 * 4)
                .ok_or_else(|| anyhow!("H3 target audio latent bytes overflow"))?
        || memory.condition_latent_backing_device_bytes
            != request
                .rows
                .condition_visual_rows
                .checked_mul(96 * 4)
                .ok_or_else(|| anyhow!("H3 condition latent bytes overflow"))?
        || memory.packed_video_state_device_bytes
            != memory
                .condition_latent_backing_device_bytes
                .checked_add(memory.target_video_latent_device_bytes)
                .ok_or_else(|| anyhow!("H3 packed video state bytes overflow"))?
        || memory.packed_audio_state_device_bytes != memory.target_audio_latent_device_bytes
        || memory.packed_layout_device_bytes
            != request
                .rows
                .total_packed_rows
                .checked_mul(24)
                .ok_or_else(|| anyhow!("H3 packed layout bytes overflow"))?
        || memory.qwen_output_state_device_bytes
            != request
                .rows
                .qwen_output_text_rows
                .checked_mul(5_120 * 2)
                .ok_or_else(|| anyhow!("H3 Qwen output state bytes overflow"))?
        || memory.noise_cpu_staging_host_bytes
            != memory
                .condition_latent_backing_device_bytes
                .max(memory.target_video_latent_device_bytes)
                .max(memory.target_audio_latent_device_bytes)
        || memory.waveform_host_bytes
            != request
                .audio_samples_per_channel
                .checked_mul(u64::from(contract::AUDIO_CHANNELS))
                .and_then(|samples| samples.checked_mul(4))
                .ok_or_else(|| anyhow!("H3 waveform host bytes overflow"))?
        || memory.audio_waveform_device_bytes != memory.waveform_host_bytes
        || memory.denoise_copy_policy
            != H3FactoryTargetDenoiseCopyPolicy::CandleF32PairedResMultistepV2
        || memory.denoise_tensor_copy_workspace_device_bytes != denoise_copy_workspace
        || memory.vae_peak_host_io_buffer_bytes == 0
        || memory.vae_peak_host_mapped_file_bytes == 0
        || memory.vae_peak_staging_disk_bytes == 0
        || memory.encoded_video_host_bytes_bound == 0
        || memory.thumbnail_host_bytes_bound == 0
        || memory.mux_output_host_bytes_bound == 0
        || memory.aac_mux_staging_host_bytes == 0
        || memory.attention_workspace_device_bytes == 0
        || memory.ffn_workspace_device_bytes == 0
        || memory.decoder_tile_workspace_device_bytes == 0
        || memory.audio_decode_workspace_device_bytes == 0
        || request.rows.condition_visual_rows == 0
            && memory.condition_vae_workspace_device_bytes != 0
        || request.rows.condition_visual_rows > 0
            && memory.condition_vae_workspace_device_bytes == 0
        || request.rows.condition_visual_rows == 0 && condition_bytes != (0, 0)
        || request.rows.condition_visual_rows > 0
            && (condition_bytes.0 == 0 || condition_bytes.1 == 0)
        || memory.identity_sha256 != expected_target_budget_identity(memory)
    {
        bail!("MiniMax H3 target budget is internally inconsistent");
    }
    Ok(())
}

fn checked_u64_sum(values: impl IntoIterator<Item = u64>, label: &'static str) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| anyhow!("{label} overflow"))
    })
}

pub fn expected_h3_factory_prepared_request_identity(
    request: &H3FactoryPreparedRequestInput,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.prepared-request.v1\0");
    update_string(&mut hash, &request.canonical_model);
    hash.update(task_id(request.task));
    hash.update(mode_id(request.mode));
    hash.update(request.prompt_sha256.as_bytes());
    hash.update(request.seed.to_le_bytes());
    for value in [
        request.grid_points,
        request.denoise_forward_count,
        request.batch_size,
        request.width,
        request.height,
        request.frames,
        request.fps,
    ] {
        hash.update(value.to_le_bytes());
    }
    hash.update(request.guidance_f64_bits.to_le_bytes());
    hash.update(request.strength_f64_bits.to_le_bytes());
    hash.update([
        u8::from(request.synchronized_audio),
        u8::from(request.mp4_output),
    ]);
    for value in [
        request.video_latent_frames,
        request.audio_latents_per_channel,
        request.audio_samples_per_channel,
    ] {
        hash.update(value.to_le_bytes());
    }
    hash.update(request.conditioning_fingerprint.as_bytes());
    hash.update(request.reference_fingerprint.as_bytes());
    hash.update((request.endpoints.len() as u64).to_le_bytes());
    for endpoint in &request.endpoints {
        hash.update(endpoint_anchor_id(endpoint.anchor));
        hash.update(endpoint.encoded_bytes.to_le_bytes());
        hash.update(endpoint.encoded_content_sha256.as_bytes());
        hash.update(match endpoint.preprocess {
            H3FactoryEndpointPreprocess::PillowLanczosRgbU8CpuV1 => {
                b"pillow-lanczos-rgb-u8-cpu-v1".as_slice()
            }
        });
        for dimension in endpoint.normalized_shape {
            hash.update(dimension.to_le_bytes());
        }
        hash.update(endpoint.normalized_cpu_bytes.to_le_bytes());
        hash.update(endpoint.normalized_cpu_content_sha256.as_bytes());
    }
    for value in [
        request.rows.qwen_output_text_rows,
        request.rows.qwen_vision_rows,
        request.rows.condition_visual_rows,
        request.rows.condition_audio_rows,
        request.rows.target_video_rows,
        request.rows.target_audio_rows,
        request.rows.total_packed_rows,
    ] {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub fn expected_h3_factory_raw_checkpoint_identity(
    checkpoint: &H3FactoryRawCheckpointInput,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.raw-opened-checkpoint.v1\0");
    hash.update(checkpoint.raw_content_sha256.as_bytes());
    hash.update(checkpoint.verified_file_bytes.to_le_bytes());
    hash.update(checkpoint.raw_header_identity_sha256.as_bytes());
    hash.update(checkpoint.opened_checkpoint_identity_sha256.as_bytes());
    hash.update(checkpoint.quantization_policy_identity_sha256.as_bytes());
    hash.update(checkpoint.config_identity_sha256.as_bytes());
    for value in [
        checkpoint.fixed_transformer_encoded_host_bytes,
        checkpoint.fixed_transformer_protected_device_bytes,
        checkpoint.fixed_transformer_max_host_read_staging_bytes,
        checkpoint.fixed_transformer_max_device_weight_staging_bytes,
    ] {
        hash.update(value.to_le_bytes());
    }
    hash.update((checkpoint.blocks.len() as u64).to_le_bytes());
    for block in &checkpoint.blocks {
        hash.update(block.index.to_le_bytes());
        for value in [
            block.encoded_host_bytes,
            block.protected_device_bytes,
            block.max_device_weight_staging_bytes,
            block.max_host_read_staging_bytes,
        ] {
            hash.update(value.to_le_bytes());
        }
        hash.update(block.content_sha256.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub fn expected_h3_factory_target_budget_identity(memory: &H3FactoryTargetBudgetInput) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.target-attempt-budget.v1\0");
    memory.update_identity(&mut hash);
    format!("{:x}", hash.finalize())
}

pub fn expected_h3_factory_prepared_attempt_identity(
    attempt: &H3FactoryPreparedAttemptInput,
) -> String {
    expected_prepared_attempt_identity_fields(
        &attempt.execution_fingerprint,
        &attempt.request,
        &attempt.raw_checkpoint,
        &attempt.target_budget,
    )
}

fn expected_prepared_attempt_identity(authority: &H3FactoryPreparedAttemptAuthority) -> String {
    expected_prepared_attempt_identity_fields(
        &authority.execution_fingerprint,
        &authority.request,
        &authority.raw_checkpoint,
        &authority.target_budget,
    )
}

fn expected_prepared_attempt_identity_fields(
    execution_fingerprint: &str,
    request: &H3FactoryPreparedRequestInput,
    checkpoint: &H3FactoryRawCheckpointInput,
    memory: &H3FactoryTargetBudgetInput,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.prepared-target-attempt.v1\0");
    hash.update(execution_fingerprint.as_bytes());
    hash.update(request.identity_sha256.as_bytes());
    hash.update(checkpoint.identity_sha256.as_bytes());
    hash.update(memory.identity_sha256.as_bytes());
    format!("{:x}", hash.finalize())
}

fn expected_prepared_request_identity(request: &H3FactoryPreparedRequestInput) -> String {
    expected_h3_factory_prepared_request_identity(request)
}

fn expected_raw_checkpoint_identity(checkpoint: &H3FactoryRawCheckpointInput) -> String {
    expected_h3_factory_raw_checkpoint_identity(checkpoint)
}

fn expected_target_budget_identity(memory: &H3FactoryTargetBudgetInput) -> String {
    expected_h3_factory_target_budget_identity(memory)
}

fn task_id(task: Task) -> &'static [u8] {
    match task {
        Task::Fl2va => b"fl2va",
        Task::Ref2va => b"ref2va",
    }
}

fn mode_id(mode: Mode) -> &'static [u8] {
    match mode {
        Mode::TextToAudioVideo => b"text-to-audio-video",
        Mode::FirstFrameToAudioVideo => b"first-frame-to-audio-video",
        Mode::LastFrameToAudioVideo => b"last-frame-to-audio-video",
        Mode::FirstAndLastFrameToAudioVideo => b"first-and-last-frame-to-audio-video",
        Mode::ReferenceToAudioVideo => b"reference-to-audio-video",
    }
}

fn endpoint_anchor_id(anchor: H3FactoryEndpointAnchor) -> &'static [u8] {
    match anchor {
        H3FactoryEndpointAnchor::First => b"first",
        H3FactoryEndpointAnchor::Last => b"last",
    }
}

fn artifact_host_role_id(role: H3FactoryArtifactHostRole) -> &'static [u8] {
    match role {
        H3FactoryArtifactHostRole::Conditioner => b"conditioner",
        H3FactoryArtifactHostRole::RawTransformerCheckpoint => b"raw-transformer-checkpoint",
        H3FactoryArtifactHostRole::TransformerSupport => b"transformer-support",
        H3FactoryArtifactHostRole::VisualVae => b"visual-vae",
        H3FactoryArtifactHostRole::AudioVae => b"audio-vae",
    }
}

fn update_string(hash: &mut Sha256, value: &str) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value.as_bytes());
}

fn validate_attention(
    attention: &H3FactoryAttentionInput,
    compute_capability: (u16, u16),
) -> Result<()> {
    require_sha256(&attention.runtime_identity_sha256, "H3 attention runtime")?;
    require_sha256(
        &attention.qualification_sha256,
        "H3 attention qualification",
    )?;
    if attention.generic_backend != AttentionBackend::Flash
        || attention.generic_chunk != AttentionChunkPolicy::Off
        || attention.runtime_backend != H3AttentionBackend::FlashAttentionV2
        || attention.kernel != H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011
        || attention.activation != H3AttentionActivation::ReleaseCandidateQualificationOnly
        || attention.device
            != (H3AttentionDevice::Cuda {
                compute_capability: Some(compute_capability),
            })
        || attention.model_contract != H3AttentionModelContract::released_bf16()
        || attention.qualification_kernel_identity != attention.kernel.identity()
        || !attention.full_noncausal
        || !attention.lossless
    {
        bail!("MiniMax H3 factory requires one exact released CUDA attention tuple");
    }
    let expected = H3AttentionRuntimeAuthority::expected_identity_for(
        attention.runtime_backend,
        attention.kernel,
        attention.activation,
        attention.device,
        attention.model_contract,
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    if attention.runtime_identity_sha256 != expected {
        bail!("MiniMax H3 attention runtime identity does not match its typed tuple");
    }
    Ok(())
}

impl FrozenH3FactoryAuthority {
    pub fn new_contract_only(input: H3FactoryAuthorityInput) -> Result<Self> {
        let model_contract = contract::capability_contract_for_model(&input.model)
            .ok_or_else(|| anyhow!("{:?} has no MiniMax H3 capability contract", input.model))?;
        if model_contract.generation.runtime_available {
            bail!(
                "contract-only MiniMax H3 factory authority cannot be created for a runnable contract"
            );
        }
        if input.device_id.trim().is_empty() || input.compute_capability.0 == 0 {
            bail!("MiniMax H3 factory authority requires one concrete CUDA route");
        }
        require_sha256(&input.execution_fingerprint, "H3 scheduler execution")?;
        if input.qwen_parameter_bytes == 0
            || input
                .qwen_host_resident_parameter_bytes
                .checked_add(input.qwen_device_resident_parameter_bytes)
                .is_none_or(|bytes| bytes == 0)
            || input.qwen_activation_workspace_bytes == 0
            || input.qwen_output_text_rows == 0
        {
            bail!("MiniMax H3 factory authority requires exact nonzero Qwen memory facts");
        }
        if input.attention_kernel_identity.trim().is_empty()
            || !input.attention_full_noncausal
            || !input.attention_lossless
            || input.attention_head_count != 56
            || input.attention_head_dim != 128
        {
            bail!("MiniMax H3 factory authority requires qualified lossless 56x128 full attention");
        }
        require_sha256(
            &input.attention_qualification_sha256,
            "H3 attention qualification",
        )?;
        if let Some(attention) = input.attention_runtime.as_ref() {
            validate_attention(attention, input.compute_capability)?;
            if attention.generic_backend != input.attention_backend
                || attention.generic_chunk != input.attention_chunk
                || attention.qualification_kernel_identity != input.attention_kernel_identity
                || attention.qualification_sha256 != input.attention_qualification_sha256
                || attention.full_noncausal != input.attention_full_noncausal
                || attention.lossless != input.attention_lossless
                || attention.model_contract.heads != input.attention_head_count as usize
                || attention.model_contract.head_dim != input.attention_head_dim as usize
            {
                bail!("MiniMax H3 typed attention runtime crosses its qualification projection");
            }
        }
        if !input.block_offload {
            bail!("MiniMax H3 factory authority requires admitted block streaming");
        }
        input.quantization.validate(model_contract.layout)?;

        let (prepared_attempt, execution_budget_echo) =
            match (
                &input.prepared_attempt,
                &input.execution_budget_echo,
                &input.attention_runtime,
            ) {
                (Some(attempt), Some(budget), Some(_)) => {
                    if model_contract.layout != Layout::ComfyPrunedInt8ConvrotNvfp4Awq {
                        bail!(
                            "MiniMax H3 prepared target attempt requires the Comfy checkpoint layout"
                        );
                    }
                    let attempt = H3FactoryPreparedAttemptAuthority::freeze(
                        attempt.clone(),
                        H3FactoryPreparedAttemptProjection {
                            execution_fingerprint: &input.execution_fingerprint,
                            qwen_output_text_rows: input.qwen_output_text_rows,
                            qwen_vision_rows: input.qwen_vision_rows,
                            condition_visual_rows: input.condition_visual_rows,
                            resident_block_count: input.resident_block_count,
                            prefetch_depth: input.prefetch_depth,
                            qwen_activation_workspace_bytes: input.qwen_activation_workspace_bytes,
                            qwen_device_parameter_bytes: input
                                .qwen_device_resident_parameter_bytes,
                            qwen_host_parameter_bytes: input.qwen_host_resident_parameter_bytes,
                            conditioner_placement: input.conditioner_placement,
                        },
                    )?;
                    require_sha256(
                        &budget.prepared_attempt_identity_sha256,
                        "H3 execution budget attempt",
                    )?;
                    if attempt.request.canonical_model != model_contract.canonical_model
                        || attempt.request.task != model_contract.task
                        || budget.prepared_attempt_identity_sha256 != attempt.identity_sha256
                        || budget.device_peak_bytes
                            != attempt.target_budget.predicted_device_peak_bytes
                        || budget.host_increment_bytes
                            != attempt.target_budget.predicted_host_increment_bytes
                    {
                        bail!("MiniMax H3 execution budget does not bind the admitted attempt");
                    }
                    match (&input.quantization, model_contract.layout) {
                        (
                            H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                                transformer_policy_sha256,
                                ..
                            },
                            Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                        ) if transformer_policy_sha256
                            == &attempt.raw_checkpoint.quantization_policy_identity_sha256 => {}
                        (H3FactoryQuantizationAuthority::OfficialBf16, Layout::OfficialBf16) => {}
                        _ => bail!(
                        "MiniMax H3 raw checkpoint and factory quantization authorities disagree"
                    ),
                    }
                    (Some(attempt), Some(budget.clone()))
                }
                (None, None, None) => (None, None),
                _ => bail!(
                    "MiniMax H3 prepared target attempt, budget echo, and typed attention must be supplied together"
                ),
            };

        let components = H3ValidatedComponentSet::new(
            input
                .components
                .into_iter()
                .map(|component| {
                    component.validate()?;
                    let role = match component.role {
                        H3FactoryComponentRole::Conditioner => H3ComponentRole::Conditioner,
                        H3FactoryComponentRole::Transformer => H3ComponentRole::Transformer,
                        H3FactoryComponentRole::VisualVae => H3ComponentRole::VisualVae,
                        H3FactoryComponentRole::AudioVae => H3ComponentRole::AudioVae,
                    };
                    H3ValidatedComponentAuthority::new(
                        role,
                        component.content_sha256,
                        component.validation_sha256,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        let conditioner_execution = match input.conditioner_placement {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
                H3ConditionerExecution::CudaResident
            }
            H3FactoryConditionerPlacement::HostCpuThenDrop => H3ConditionerExecution::CpuOffloaded,
        };
        let conditioner_device = match input.conditioner_placement {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop => input.device_id.clone(),
            H3FactoryConditionerPlacement::HostCpuThenDrop => "cpu".to_string(),
        };
        let conditioner_placement = FrozenH3ConditionerPlacement::new(
            conditioner_device,
            conditioner_execution,
            input.execution_fingerprint.clone(),
            input.qwen_host_resident_parameter_bytes,
            input.qwen_device_resident_parameter_bytes,
            input.qwen_activation_workspace_bytes,
        )?;
        let block_streaming = FrozenH3BlockStreamingPlan::new(
            input.device_id.clone(),
            input.execution_fingerprint.clone(),
            usize::try_from(input.resident_block_count)
                .map_err(|_| anyhow!("MiniMax H3 resident block count exceeds usize"))?,
            usize::try_from(input.prefetch_depth)
                .map_err(|_| anyhow!("MiniMax H3 prefetch depth exceeds usize"))?,
        )?;
        let backend_plan = FrozenH3Fl2VaCandlePlan::new_unavailable(
            model_contract.canonical_model,
            input.device_id,
            H3CandleBackendDevice::Cuda {
                compute_capability: input.compute_capability,
            },
            input.execution_fingerprint,
            conditioner_placement,
            block_streaming,
            components,
        )?;
        let comfy_vae_artifact_plan_identity_sha256 =
            if model_contract.layout == Layout::ComfyPrunedInt8ConvrotNvfp4Awq {
                Some(
                    expected_h3_comfy_vae_artifact_plan_identity(model_contract.canonical_model)
                        .map_err(|error| anyhow!(error.to_string()))?,
                )
            } else {
                None
            };
        let mut frozen = Self {
            schema_version: H3_FACTORY_AUTHORITY_SCHEMA_VERSION,
            backend_plan,
            comfy_vae_artifact_plan_identity_sha256,
            device_ordinal: input.device_ordinal,
            conditioner_placement: input.conditioner_placement,
            qwen_parameter_bytes: input.qwen_parameter_bytes,
            qwen_host_resident_parameter_bytes: input.qwen_host_resident_parameter_bytes,
            qwen_device_resident_parameter_bytes: input.qwen_device_resident_parameter_bytes,
            qwen_activation_workspace_bytes: input.qwen_activation_workspace_bytes,
            qwen_output_text_rows: input.qwen_output_text_rows,
            qwen_vision_rows: input.qwen_vision_rows,
            condition_visual_rows: input.condition_visual_rows,
            attention_backend: input.attention_backend,
            attention_chunk: input.attention_chunk,
            attention_kernel_identity: input.attention_kernel_identity,
            attention_qualification_sha256: input.attention_qualification_sha256,
            attention_full_noncausal: input.attention_full_noncausal,
            attention_lossless: input.attention_lossless,
            attention_head_count: input.attention_head_count,
            attention_head_dim: input.attention_head_dim,
            attention_runtime: input.attention_runtime,
            block_offload: input.block_offload,
            quantization: input.quantization,
            prepared_attempt,
            execution_budget_echo,
            identity_sha256: String::new(),
        };
        frozen.identity_sha256 = frozen_identity(&frozen);
        frozen.validate_frozen()?;
        Ok(frozen)
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    /// Atomically enrich one already-validated scheduler authority with the
    /// opened/preprocessed attempt triad. The base authority remains
    /// immutable; the returned authority receives a new canonical identity.
    #[cfg(all(any(feature = "h3", feature = "h3-private-uat"), feature = "mp4"))]
    pub(crate) fn with_private_prepared_attempt(
        &self,
        prepared_attempt: H3FactoryPreparedAttemptInput,
        execution_budget_echo: H3FactoryExecutionBudgetEchoInput,
        attention_runtime: H3FactoryAttentionInput,
    ) -> Result<Self> {
        self.validate_frozen()?;
        if self.prepared_attempt.is_some()
            || self.execution_budget_echo.is_some()
            || self.attention_runtime.is_some()
        {
            bail!("private H3 factory base authority already contains an attempt triad")
        }
        validate_attention(&attention_runtime, self.compute_capability())?;
        let prepared_attempt = H3FactoryPreparedAttemptAuthority::freeze(
            prepared_attempt,
            H3FactoryPreparedAttemptProjection {
                execution_fingerprint: self.execution_fingerprint(),
                qwen_output_text_rows: self.qwen_output_text_rows,
                qwen_vision_rows: self.qwen_vision_rows,
                condition_visual_rows: self.condition_visual_rows,
                resident_block_count: u32::try_from(
                    self.backend_plan.block_streaming().resident_block_count,
                )
                .map_err(|_| anyhow!("private H3 resident block count exceeds u32"))?,
                prefetch_depth: u32::try_from(self.backend_plan.block_streaming().prefetch_depth)
                    .map_err(|_| anyhow!("private H3 prefetch depth exceeds u32"))?,
                qwen_activation_workspace_bytes: self.qwen_activation_workspace_bytes,
                qwen_device_parameter_bytes: self.qwen_device_resident_parameter_bytes,
                qwen_host_parameter_bytes: self.qwen_host_resident_parameter_bytes,
                conditioner_placement: self.conditioner_placement,
            },
        )?;
        let mut enriched = self.clone();
        enriched.prepared_attempt = Some(prepared_attempt);
        enriched.execution_budget_echo = Some(execution_budget_echo);
        enriched.attention_runtime = Some(attention_runtime);
        enriched.identity_sha256 = frozen_identity(&enriched);
        enriched.validate_frozen()?;
        Ok(enriched)
    }

    pub fn component_set_identity_sha256(&self) -> &str {
        self.backend_plan.component_set_identity()
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn backend_plan_identity_sha256(&self) -> &str {
        self.backend_plan.identity_sha256()
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn private_vae_adapter_authority(&self) -> Result<H3PrivateVaeFactoryAuthority> {
        self.validate_frozen()?;
        let vae_artifact_plan_identity_sha256 = self
            .comfy_vae_artifact_plan_identity_sha256
            .as_deref()
            .ok_or_else(|| anyhow!("MiniMax H3 factory authority has no private Comfy VAE plan"))?;
        if self.task() != Task::Fl2va {
            bail!("private MiniMax H3 VAE adapter currently requires the FL2VA task authority");
        }
        Ok(H3PrivateVaeFactoryAuthority {
            factory_identity_sha256: self.identity_sha256.clone(),
            backend_plan_identity_sha256: self.backend_plan_identity_sha256().into(),
            vae_artifact_plan_identity_sha256: vae_artifact_plan_identity_sha256.into(),
            component_set_identity_sha256: self.component_set_identity_sha256().into(),
            canonical_model: self.canonical_model().into(),
            task: self.task(),
            device_id: self.device_id().into(),
            execution_fingerprint: self.execution_fingerprint().into(),
        })
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    #[allow(dead_code)]
    pub(crate) fn private_fl2va_runtime_authority(&self) -> Result<H3PrivateFl2VaFactoryAuthority> {
        self.validate_frozen()?;
        let attention = match (
            &self.prepared_attempt,
            &self.execution_budget_echo,
            &self.attention_runtime,
        ) {
            (Some(_), Some(_), Some(attention)) => attention.clone(),
            _ => {
                bail!(
                    "private MiniMax H3 runtime requires the exact prepared attempt, budget echo, and typed attention triad"
                )
            }
        };
        if !h3_factory_activation_prerequisites().is_empty() {
            bail!(
                "private MiniMax H3 execution remains unavailable until target-budget prerequisites are verified"
            );
        }
        self.private_fl2va_runtime_authority_record(attention)
    }

    /// Private reviewed-evidence projection. Unlike the public/no-evidence
    /// projection above, this requires a non-Clone token issued from the exact
    /// opened, prepared, scheduler-ledger, owner-scope, and runtime-record
    /// authorities. The public prerequisite list remains intact and is part
    /// of the token identity.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn private_fl2va_runtime_authority_with_activation(
        &self,
        activation: &H3PrivateFactoryActivationEvidence,
    ) -> Result<H3PrivateFl2VaFactoryAuthority> {
        self.validate_frozen()?;
        activation.revalidate_for(self)?;
        let attention = match (
            &self.prepared_attempt,
            &self.execution_budget_echo,
            &self.attention_runtime,
        ) {
            (Some(_), Some(_), Some(attention)) => attention.clone(),
            _ => {
                bail!(
                    "private MiniMax H3 activation requires the exact prepared attempt, budget echo, and typed attention triad"
                )
            }
        };
        self.private_fl2va_runtime_authority_record(attention)
    }

    /// Test-only projection for preserving schema-neutral synthetic runtime
    /// unit coverage. Production and integration paths cannot call this seam.
    #[cfg(all(test, any(feature = "h3", feature = "h3-private-uat")))]
    pub(crate) fn private_fl2va_runtime_authority_for_schema_tests(
        &self,
    ) -> Result<H3PrivateFl2VaFactoryAuthority> {
        self.validate_frozen()?;
        let attention = match self.attention_runtime.as_ref() {
            Some(attention) => attention.clone(),
            None => {
                let runtime_backend = H3AttentionBackend::FlashAttentionV2;
                let kernel = H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011;
                let activation = H3AttentionActivation::ReleaseCandidateQualificationOnly;
                let device = H3AttentionDevice::Cuda {
                    compute_capability: Some(self.compute_capability()),
                };
                let model_contract = H3AttentionModelContract::released_bf16();
                H3FactoryAttentionInput {
                    generic_backend: self.attention_backend,
                    generic_chunk: self.attention_chunk,
                    runtime_backend,
                    kernel,
                    activation,
                    device,
                    model_contract,
                    runtime_identity_sha256: H3AttentionRuntimeAuthority::expected_identity_for(
                        runtime_backend,
                        kernel,
                        activation,
                        device,
                        model_contract,
                    )
                    .map_err(|error| anyhow!(error.to_string()))?,
                    qualification_kernel_identity: kernel.identity().into(),
                    qualification_sha256: self.attention_qualification_sha256.clone(),
                    full_noncausal: self.attention_full_noncausal,
                    lossless: self.attention_lossless,
                }
            }
        };
        self.private_fl2va_runtime_authority_record(attention)
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    fn private_fl2va_runtime_authority_record(
        &self,
        attention: H3FactoryAttentionInput,
    ) -> Result<H3PrivateFl2VaFactoryAuthority> {
        let vae_artifact_plan_identity_sha256 = self
            .comfy_vae_artifact_plan_identity_sha256
            .as_deref()
            .ok_or_else(|| anyhow!("MiniMax H3 factory authority has no private Comfy VAE plan"))?;
        if self.task() != Task::Fl2va {
            bail!("private MiniMax H3 streamed runtime currently requires FL2VA authority");
        }
        let (conditioner_content, conditioner_validation) =
            self.component_authority(H3ComponentRole::Conditioner);
        let (transformer_content, transformer_validation) =
            self.component_authority(H3ComponentRole::Transformer);
        let (visual_content, visual_validation) =
            self.component_authority(H3ComponentRole::VisualVae);
        let (audio_content, audio_validation) = self.component_authority(H3ComponentRole::AudioVae);
        Ok(H3PrivateFl2VaFactoryAuthority {
            factory_identity_sha256: self.identity_sha256.clone(),
            backend_plan_identity_sha256: self.backend_plan_identity_sha256().into(),
            component_set_identity_sha256: self.component_set_identity_sha256().into(),
            vae_artifact_plan_identity_sha256: vae_artifact_plan_identity_sha256.into(),
            canonical_model: self.canonical_model().into(),
            task: self.task(),
            device_id: self.device_id().into(),
            device_ordinal: self.device_ordinal,
            compute_capability: self.compute_capability(),
            execution_fingerprint: self.execution_fingerprint().into(),
            condition_visual_rows: self.condition_visual_rows,
            block_streaming: self.backend_plan.block_streaming().clone(),
            attention,
            quantization: self.quantization.clone(),
            conditioner_component_content_sha256: conditioner_content.into(),
            conditioner_component_validation_sha256: conditioner_validation.into(),
            transformer_component_content_sha256: transformer_content.into(),
            transformer_component_validation_sha256: transformer_validation.into(),
            visual_vae_component_content_sha256: visual_content.into(),
            visual_vae_component_validation_sha256: visual_validation.into(),
            audio_vae_component_content_sha256: audio_content.into(),
            audio_vae_component_validation_sha256: audio_validation.into(),
        })
    }

    /// Exact logical conditioner authority frozen by server admission.
    ///
    /// Private runtime adapters use this to cross-check the independently
    /// authenticated Qwen/support lease. It deliberately exposes only
    /// digests, never artifact paths or bytes.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn conditioner_component_authority(&self) -> (&str, &str) {
        self.private_component_authority(H3FactoryComponentRole::Conditioner)
    }

    /// Exact payload-free authority for independently reopening one private
    /// component. This comparison seam prevents an attempt-local artifact
    /// object from inheriting admission digests without recomputing them.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn private_component_authority(&self, role: H3FactoryComponentRole) -> (&str, &str) {
        let role = match role {
            H3FactoryComponentRole::Conditioner => H3ComponentRole::Conditioner,
            H3FactoryComponentRole::Transformer => H3ComponentRole::Transformer,
            H3FactoryComponentRole::VisualVae => H3ComponentRole::VisualVae,
            H3FactoryComponentRole::AudioVae => H3ComponentRole::AudioVae,
        };
        self.component_authority(role)
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    fn component_authority(&self, role: H3ComponentRole) -> (&str, &str) {
        let authority = self
            .backend_plan
            .components()
            .authority(role)
            .expect("validated H3 component set always contains every required role");
        (authority.content_sha256(), authority.validation_sha256())
    }

    pub fn canonical_model(&self) -> &str {
        self.backend_plan.canonical_model()
    }

    pub const fn task(&self) -> Task {
        self.backend_plan.task()
    }

    pub fn device_id(&self) -> &str {
        self.backend_plan.device_id()
    }

    pub const fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn compute_capability(&self) -> (u16, u16) {
        match self.backend_plan.backend() {
            H3CandleBackendDevice::Cuda { compute_capability } => compute_capability,
        }
    }

    pub fn execution_fingerprint(&self) -> &str {
        self.backend_plan.execution_fingerprint()
    }

    /// Clone-free identity projection for the future server-owned one-shot
    /// attempt root.
    ///
    /// Returning `None` is the production state today: server admission does
    /// not populate the prepared-attempt, target-budget echo, or typed
    /// attention triad. These value identities alone never make an authority
    /// executable and do not remove any activation prerequisite.
    pub fn prepared_target_attempt_identities(&self) -> Option<(&str, &str)> {
        match (
            &self.prepared_attempt,
            &self.execution_budget_echo,
            &self.attention_runtime,
        ) {
            (Some(attempt), Some(budget), Some(_))
                if budget.prepared_attempt_identity_sha256 == attempt.identity_sha256 =>
            {
                Some((
                    attempt.identity_sha256.as_str(),
                    attempt.target_budget.identity_sha256.as_str(),
                ))
            }
            _ => None,
        }
    }

    pub fn attention_qualification_sha256(&self) -> &str {
        &self.attention_qualification_sha256
    }

    pub fn attention_runtime_identity_sha256(&self) -> &str {
        self.attention_runtime
            .as_ref()
            .map_or("", |attention| &attention.runtime_identity_sha256)
    }

    pub const fn attention_backend(&self) -> AttentionBackend {
        self.attention_backend
    }

    pub const fn attention_chunk(&self) -> AttentionChunkPolicy {
        self.attention_chunk
    }

    pub const fn conditioner_placement(&self) -> H3FactoryConditionerPlacement {
        self.conditioner_placement
    }

    pub const fn qwen_parameter_bytes(&self) -> u64 {
        self.qwen_parameter_bytes
    }

    pub const fn qwen_host_resident_parameter_bytes(&self) -> u64 {
        self.qwen_host_resident_parameter_bytes
    }

    pub const fn qwen_device_resident_parameter_bytes(&self) -> u64 {
        self.qwen_device_resident_parameter_bytes
    }

    pub const fn qwen_activation_workspace_bytes(&self) -> u64 {
        self.qwen_activation_workspace_bytes
    }

    pub const fn qwen_output_text_rows(&self) -> u64 {
        self.qwen_output_text_rows
    }

    pub const fn qwen_vision_rows(&self) -> u64 {
        self.qwen_vision_rows
    }

    pub fn resident_block_count(&self) -> usize {
        self.backend_plan.block_streaming().resident_block_count
    }

    pub fn prefetch_depth(&self) -> usize {
        self.backend_plan.block_streaming().prefetch_depth
    }

    pub const fn block_offload(&self) -> bool {
        self.block_offload
    }

    /// Validate the immutable authority carried into the legal-neutral engine
    /// seam without claiming that H3 is runnable.
    ///
    /// Production factory dispatch must still call `validate_for_dispatch`,
    /// which additionally requires the public runtime capability and family
    /// registry. This narrower check exists only so an injected, weight-free
    /// runtime can exercise the engine/worker transaction while those gates
    /// remain closed.
    pub(crate) fn validate_engine_seam(
        &self,
        model: &str,
        gpu_ordinal: usize,
        offload: bool,
    ) -> Result<()> {
        self.validate_frozen()?;
        let request_contract = contract::capability_contract_for_model(model)
            .ok_or_else(|| anyhow!("{model:?} has no MiniMax H3 capability contract"))?;
        if request_contract.canonical_model != self.canonical_model()
            || request_contract.task != self.task()
            || gpu_ordinal != self.device_ordinal
            || offload != self.block_offload
        {
            bail!("MiniMax H3 frozen engine authority changed before construction");
        }
        Ok(())
    }

    pub fn quantization(&self) -> &H3FactoryQuantizationAuthority {
        &self.quantization
    }

    fn validate_frozen(&self) -> Result<()> {
        if self.schema_version != H3_FACTORY_AUTHORITY_SCHEMA_VERSION {
            bail!("MiniMax H3 factory authority uses an unsupported schema version");
        }
        self.backend_plan.validate()?;
        let expected_vae_plan =
            if self.backend_plan.layout() == Layout::ComfyPrunedInt8ConvrotNvfp4Awq {
                Some(
                    expected_h3_comfy_vae_artifact_plan_identity(self.canonical_model())
                        .map_err(|error| anyhow!(error.to_string()))?,
                )
            } else {
                None
            };
        if self.comfy_vae_artifact_plan_identity_sha256 != expected_vae_plan {
            bail!("MiniMax H3 factory VAE artifact plan changed after admission");
        }
        let conditioner_memory = &self.backend_plan.conditioner_placement().memory;
        if conditioner_memory.resident_parameter_bytes
            != self
                .qwen_host_resident_parameter_bytes
                .checked_add(self.qwen_device_resident_parameter_bytes)
                .ok_or_else(|| anyhow!("MiniMax H3 Qwen resident bytes overflow"))?
            || conditioner_memory.activation_workspace_bytes != self.qwen_activation_workspace_bytes
        {
            bail!("MiniMax H3 conditioner placement differs from frozen Qwen memory facts");
        }
        if self.backend_plan.block_streaming().resident_block_count > 50
            || self.backend_plan.block_streaming().prefetch_depth > 2
        {
            bail!("MiniMax H3 factory authority changed its streaming bounds");
        }
        if !self.block_offload
            || self.qwen_parameter_bytes == 0
            || self
                .qwen_host_resident_parameter_bytes
                .checked_add(self.qwen_device_resident_parameter_bytes)
                .is_none_or(|bytes| bytes == 0)
            || self.qwen_activation_workspace_bytes == 0
            || self.qwen_output_text_rows == 0
        {
            bail!("MiniMax H3 factory attention or offload authority changed after admission");
        }
        if self.attention_kernel_identity.trim().is_empty()
            || !self.attention_full_noncausal
            || !self.attention_lossless
            || self.attention_head_count != 56
            || self.attention_head_dim != 128
        {
            bail!("MiniMax H3 frozen attention qualification is incomplete");
        }
        require_sha256(
            &self.attention_qualification_sha256,
            "H3 attention qualification",
        )?;
        if let Some(attention) = self.attention_runtime.as_ref() {
            validate_attention(attention, self.compute_capability())?;
            if attention.generic_backend != self.attention_backend
                || attention.generic_chunk != self.attention_chunk
                || attention.qualification_kernel_identity != self.attention_kernel_identity
                || attention.qualification_sha256 != self.attention_qualification_sha256
                || attention.full_noncausal != self.attention_full_noncausal
                || attention.lossless != self.attention_lossless
                || attention.model_contract.heads != self.attention_head_count as usize
                || attention.model_contract.head_dim != self.attention_head_dim as usize
            {
                bail!("MiniMax H3 frozen typed attention crosses qualification evidence");
            }
        }
        self.quantization.validate(self.backend_plan.layout())?;
        match (
            &self.prepared_attempt,
            &self.execution_budget_echo,
            &self.attention_runtime,
        ) {
            (Some(attempt), Some(budget), Some(_)) => {
                attempt.validate(H3FactoryPreparedAttemptProjection {
                    execution_fingerprint: self.execution_fingerprint(),
                    qwen_output_text_rows: self.qwen_output_text_rows,
                    qwen_vision_rows: self.qwen_vision_rows,
                    condition_visual_rows: self.condition_visual_rows,
                    resident_block_count: u32::try_from(self.resident_block_count())
                        .map_err(|_| anyhow!("H3 resident block count exceeds u32"))?,
                    prefetch_depth: u32::try_from(self.prefetch_depth())
                        .map_err(|_| anyhow!("H3 prefetch depth exceeds u32"))?,
                    qwen_activation_workspace_bytes: self.qwen_activation_workspace_bytes,
                    qwen_device_parameter_bytes: self.qwen_device_resident_parameter_bytes,
                    qwen_host_parameter_bytes: self.qwen_host_resident_parameter_bytes,
                    conditioner_placement: self.conditioner_placement,
                })?;
                if budget.prepared_attempt_identity_sha256 != attempt.identity_sha256
                    || budget.device_peak_bytes != attempt.target_budget.predicted_device_peak_bytes
                    || budget.host_increment_bytes
                        != attempt.target_budget.predicted_host_increment_bytes
                {
                    bail!("MiniMax H3 frozen execution budget changed after admission");
                }
            }
            (None, None, None) => {}
            _ => bail!("MiniMax H3 frozen attempt and budget echo are incomplete"),
        }
        require_sha256(&self.identity_sha256, "H3 factory authority")?;
        if self.identity_sha256 != frozen_identity(self) {
            bail!("MiniMax H3 factory authority changed after admission");
        }
        Ok(())
    }

    pub(crate) fn validate_for_dispatch(
        &self,
        model: &str,
        family: &str,
        gpu_ordinal: usize,
        offload: bool,
        attention_backend: AttentionBackend,
        attention_chunk: AttentionChunkPolicy,
    ) -> Result<()> {
        self.validate_frozen()?;
        let request_contract = contract::capability_contract_for_model(model)
            .ok_or_else(|| anyhow!("{model:?} has no MiniMax H3 capability contract"))?;
        if !contract::is_family(family)
            || request_contract.canonical_model != self.canonical_model()
            || request_contract.task != self.task()
            || gpu_ordinal != self.device_ordinal
            || offload != self.block_offload
            || attention_backend != self.attention_backend
            || attention_chunk != self.attention_chunk
        {
            bail!(
                "MiniMax H3 frozen route, attention, or offload authority changed before dispatch"
            );
        }
        if self.prepared_attempt.is_none()
            || self.execution_budget_echo.is_none()
            || self.attention_runtime.is_none()
            || !h3_factory_activation_prerequisites().is_empty()
            || contract::runnable_capability_contract_for_model(model).is_none()
            || crate::production_family_capability_for_family(family).is_none()
        {
            bail!(
                "MiniMax H3 public capability or production factory registry remains runtime unavailable"
            );
        }
        Ok(())
    }
}

fn frozen_identity(authority: &FrozenH3FactoryAuthority) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.factory-authority.v5\0");
    hash.update(authority.schema_version.to_le_bytes());
    hash.update(authority.backend_plan.identity_sha256().as_bytes());
    hash.update([0]);
    hash.update(
        authority
            .comfy_vae_artifact_plan_identity_sha256
            .as_deref()
            .unwrap_or("no-comfy-vae-plan")
            .as_bytes(),
    );
    hash.update(authority.device_ordinal.to_le_bytes());
    hash.update(match authority.conditioner_placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop => b"qwen-cuda".as_slice(),
        H3FactoryConditionerPlacement::HostCpuThenDrop => b"qwen-cpu".as_slice(),
    });
    hash.update(authority.qwen_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_host_resident_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_device_resident_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_activation_workspace_bytes.to_le_bytes());
    hash.update(authority.qwen_output_text_rows.to_le_bytes());
    hash.update(authority.qwen_vision_rows.to_le_bytes());
    hash.update(authority.condition_visual_rows.to_le_bytes());
    hash.update(match authority.attention_backend {
        AttentionBackend::Math => b"math".as_slice(),
        AttentionBackend::Flash => b"flash".as_slice(),
    });
    match authority.attention_chunk {
        AttentionChunkPolicy::Auto => hash.update(b"chunk-auto"),
        AttentionChunkPolicy::Off => hash.update(b"chunk-off"),
        AttentionChunkPolicy::Size(size) => {
            hash.update(b"chunk-size\0");
            hash.update(size.to_le_bytes());
        }
    }
    hash.update(authority.attention_kernel_identity.as_bytes());
    hash.update([0]);
    hash.update(authority.attention_qualification_sha256.as_bytes());
    hash.update([
        u8::from(authority.attention_full_noncausal),
        u8::from(authority.attention_lossless),
        u8::from(authority.block_offload),
    ]);
    hash.update(authority.attention_head_count.to_le_bytes());
    hash.update(authority.attention_head_dim.to_le_bytes());
    if let Some(attention) = authority.attention_runtime.as_ref() {
        hash.update(b"typed-attention\0");
        hash.update(attention.runtime_identity_sha256.as_bytes());
        hash.update(attention.qualification_kernel_identity.as_bytes());
        hash.update(attention.qualification_sha256.as_bytes());
    } else {
        hash.update(b"typed-attention-unavailable");
    }
    match (
        &authority.prepared_attempt,
        &authority.execution_budget_echo,
    ) {
        (Some(attempt), Some(budget)) => {
            hash.update(b"prepared-target-attempt\0");
            hash.update(attempt.identity_sha256.as_bytes());
            hash.update(budget.prepared_attempt_identity_sha256.as_bytes());
            hash.update(budget.device_peak_bytes.to_le_bytes());
            hash.update(budget.host_increment_bytes.to_le_bytes());
        }
        (None, None) => hash.update(b"activation-authority-unavailable"),
        _ => hash.update(b"invalid-partial-activation-authority"),
    }
    authority.quantization.update_identity(&mut hash);
    format!("{:x}", hash.finalize())
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        bail!("{label} fingerprint is not SHA-256")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn sum(values: &[u64]) -> u64 {
        values.iter().copied().sum()
    }

    fn block_sha(index: u16) -> String {
        format!("{:064x}", u64::from(index) + 1)
    }

    fn prepared_request() -> H3FactoryPreparedRequestInput {
        let width = 768;
        let height = 512;
        let frames = 124;
        let video_latent_frames = 37;
        let rows_per_video_latent = u64::from(width / 32) * u64::from(height / 32);
        let target_video_rows = video_latent_frames * rows_per_video_latent;
        let audio_latents_per_channel = 207;
        let target_audio_rows = audio_latents_per_channel * u64::from(contract::AUDIO_CHANNELS);
        let condition_visual_rows = rows_per_video_latent;
        let mut request = H3FactoryPreparedRequestInput {
            identity_sha256: String::new(),
            canonical_model: contract::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            mode: Mode::FirstFrameToAudioVideo,
            prompt_sha256: sha('1'),
            seed: 42,
            grid_points: 5,
            denoise_forward_count: 4,
            guidance_f64_bits: 0.0f64.to_bits(),
            strength_f64_bits: 1.0f64.to_bits(),
            batch_size: 1,
            width,
            height,
            frames,
            fps: contract::FIXED_FPS,
            synchronized_audio: true,
            mp4_output: true,
            video_latent_frames,
            audio_latents_per_channel,
            audio_samples_per_channel: audio_latents_per_channel * 800,
            conditioning_fingerprint: sha('2'),
            reference_fingerprint: sha('3'),
            endpoints: vec![H3FactoryEndpointInput {
                anchor: H3FactoryEndpointAnchor::First,
                encoded_bytes: 128,
                encoded_content_sha256: sha('4'),
                preprocess: H3FactoryEndpointPreprocess::PillowLanczosRgbU8CpuV1,
                normalized_shape: [1, 3, 1, height, width],
                normalized_cpu_bytes: u64::from(width) * u64::from(height) * 3,
                normalized_cpu_content_sha256: sha('5'),
            }],
            rows: H3FactoryPreparedRowsInput {
                qwen_output_text_rows: 1,
                qwen_vision_rows: 64,
                condition_visual_rows,
                condition_audio_rows: 0,
                target_video_rows,
                target_audio_rows,
                total_packed_rows: 1
                    + condition_visual_rows
                    + target_video_rows
                    + target_audio_rows,
            },
        };
        request.identity_sha256 = expected_h3_factory_prepared_request_identity(&request);
        request
    }

    fn raw_checkpoint() -> H3FactoryRawCheckpointInput {
        let blocks = (0_u16..50)
            .map(|index| H3FactoryBlockMemoryInput {
                index,
                encoded_host_bytes: 100 + u64::from(index),
                protected_device_bytes: 20 + u64::from(index),
                max_device_weight_staging_bytes: 10 + u64::from(index),
                max_host_read_staging_bytes: 5 + u64::from(index),
                content_sha256: block_sha(index),
            })
            .collect::<Vec<_>>();
        let fixed_transformer_encoded_host_bytes = 1_000;
        let verified_file_bytes = fixed_transformer_encoded_host_bytes
            + blocks
                .iter()
                .map(|block| block.encoded_host_bytes)
                .sum::<u64>();
        let mut checkpoint = H3FactoryRawCheckpointInput {
            identity_sha256: String::new(),
            raw_content_sha256: sha('3'),
            verified_file_bytes,
            raw_header_identity_sha256: sha('6'),
            opened_checkpoint_identity_sha256: sha('7'),
            quantization_policy_identity_sha256: sha('d'),
            config_identity_sha256: sha('8'),
            fixed_transformer_encoded_host_bytes,
            fixed_transformer_protected_device_bytes: 1_000,
            fixed_transformer_max_host_read_staging_bytes: 100,
            fixed_transformer_max_device_weight_staging_bytes: 100,
            blocks,
        };
        checkpoint.identity_sha256 = expected_h3_factory_raw_checkpoint_identity(&checkpoint);
        checkpoint
    }

    fn target_budget(
        request: &H3FactoryPreparedRequestInput,
        checkpoint: &H3FactoryRawCheckpointInput,
    ) -> H3FactoryTargetBudgetInput {
        let artifacts = vec![
            H3FactoryArtifactHostInput {
                role: H3FactoryArtifactHostRole::Conditioner,
                index: 0,
                content_sha256: sha('a'),
                bytes: 4_096,
            },
            H3FactoryArtifactHostInput {
                role: H3FactoryArtifactHostRole::RawTransformerCheckpoint,
                index: 0,
                content_sha256: checkpoint.raw_content_sha256.clone(),
                bytes: checkpoint.verified_file_bytes,
            },
            H3FactoryArtifactHostInput {
                role: H3FactoryArtifactHostRole::VisualVae,
                index: 0,
                content_sha256: sha('4'),
                bytes: 8_192,
            },
            H3FactoryArtifactHostInput {
                role: H3FactoryArtifactHostRole::AudioVae,
                index: 0,
                content_sha256: sha('5'),
                bytes: 4_096,
            },
        ];
        let artifact_host_bytes = artifacts.iter().map(|artifact| artifact.bytes).sum();
        let qwen_host_parameter_bytes = 2_048;
        let qwen_host_activation_bytes = 1_024;
        let qwen_output_state_device_bytes = request.rows.qwen_output_text_rows * 5_120 * 2;
        let qwen_host_output_state_bytes = qwen_output_state_device_bytes;
        let qwen_host_workspace_bytes =
            qwen_host_parameter_bytes + qwen_host_activation_bytes + qwen_host_output_state_bytes;
        let condition_latent_backing_device_bytes = request.rows.condition_visual_rows * 96 * 4;
        let target_video_latent_device_bytes = request.rows.target_video_rows * 96 * 4;
        let target_audio_latent_device_bytes = request.rows.target_audio_rows * 32 * 4;
        let packed_video_state_device_bytes =
            condition_latent_backing_device_bytes + target_video_latent_device_bytes;
        let packed_audio_state_device_bytes = target_audio_latent_device_bytes;
        let packed_layout_device_bytes = request.rows.total_packed_rows * 24;
        let denoise_tensor_copy_workspace_device_bytes =
            (packed_video_state_device_bytes + packed_audio_state_device_bytes) * 9;
        let waveform_host_bytes =
            request.audio_samples_per_channel * u64::from(contract::AUDIO_CHANNELS) * 4;
        let retained_vaes = 900;
        let fixed_runtime_device_bytes = 100;
        let fixed_transformer_device_bytes = checkpoint.fixed_transformer_protected_device_bytes;
        let condition_vae_workspace_device_bytes = 300;
        let attention_workspace_device_bytes = 200;
        let ffn_workspace_device_bytes = 300;
        let decoder_tile_workspace_device_bytes = 500;
        let audio_decode_workspace_device_bytes = 400;
        let streamed_block_device_overlap_bytes = checkpoint
            .blocks
            .iter()
            .map(|block| block.protected_device_bytes)
            .max()
            .unwrap();
        let max_device_weight_staging_bytes = checkpoint
            .blocks
            .iter()
            .map(|block| block.max_device_weight_staging_bytes)
            .max()
            .unwrap();
        let fixed_transformer_load_device_staging_bytes =
            checkpoint.fixed_transformer_max_device_weight_staging_bytes;
        let vae_load_phase_device_bytes = fixed_runtime_device_bytes + retained_vaes + 100;
        let qwen_encode_phase_device_bytes = fixed_runtime_device_bytes + retained_vaes;
        let qwen_transfer_phase_device_bytes =
            fixed_runtime_device_bytes + retained_vaes + qwen_output_state_device_bytes;
        let condition_encode_phase_device_bytes = sum(&[
            fixed_runtime_device_bytes,
            retained_vaes,
            qwen_output_state_device_bytes,
            condition_vae_workspace_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
        ]);
        let noise_allocation_phase_device_bytes = sum(&[
            fixed_runtime_device_bytes,
            retained_vaes,
            qwen_output_state_device_bytes,
            condition_latent_backing_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
            target_video_latent_device_bytes,
            target_video_latent_device_bytes,
            target_audio_latent_device_bytes,
            target_audio_latent_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
        ]);
        let transformer_load_phase_device_bytes = sum(&[
            fixed_runtime_device_bytes,
            retained_vaes,
            fixed_transformer_device_bytes,
            qwen_output_state_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            fixed_transformer_load_device_staging_bytes,
        ]);
        let denoise_phase_device_bytes = sum(&[
            fixed_runtime_device_bytes,
            retained_vaes,
            fixed_transformer_device_bytes,
            qwen_output_state_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            denoise_tensor_copy_workspace_device_bytes,
            attention_workspace_device_bytes,
            ffn_workspace_device_bytes,
            streamed_block_device_overlap_bytes,
            max_device_weight_staging_bytes,
        ]);
        let visual_decode_phase_device_bytes = sum(&[
            fixed_runtime_device_bytes,
            retained_vaes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            target_video_latent_device_bytes,
            target_audio_latent_device_bytes,
            decoder_tile_workspace_device_bytes,
        ]);
        let audio_decode_phase_device_bytes = sum(&[
            fixed_runtime_device_bytes,
            retained_vaes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            target_audio_latent_device_bytes,
            audio_decode_workspace_device_bytes,
            waveform_host_bytes,
        ]);
        let waveform_transfer_phase_device_bytes =
            fixed_runtime_device_bytes + retained_vaes + waveform_host_bytes;
        let predicted_device_peak_bytes = *[
            vae_load_phase_device_bytes,
            qwen_encode_phase_device_bytes,
            qwen_transfer_phase_device_bytes,
            condition_encode_phase_device_bytes,
            noise_allocation_phase_device_bytes,
            transformer_load_phase_device_bytes,
            denoise_phase_device_bytes,
            visual_decode_phase_device_bytes,
            audio_decode_phase_device_bytes,
            waveform_transfer_phase_device_bytes,
        ]
        .iter()
        .max()
        .unwrap();
        let condition_backing_host_bytes = condition_latent_backing_device_bytes;
        let endpoint_encoded_host_bytes = request
            .endpoints
            .iter()
            .map(|endpoint| endpoint.encoded_bytes)
            .sum();
        let normalized_endpoint_host_bytes = request
            .endpoints
            .iter()
            .map(|endpoint| endpoint.normalized_cpu_bytes)
            .sum();
        let schedule_host_bytes = u64::from(request.grid_points) * 16;
        let packed_layout_host_bytes = request.rows.total_packed_rows * 24;
        let packed_layout_construction_staging_host_bytes = request.rows.total_packed_rows * 16;
        let packed_layout_freeze_staging_host_bytes = request.rows.total_packed_rows * 12;
        let text_modality_tags_host_bytes = request.rows.qwen_output_text_rows * 8;
        let noise_cpu_staging_host_bytes = condition_latent_backing_device_bytes
            .max(target_video_latent_device_bytes)
            .max(target_audio_latent_device_bytes);
        let max_host_read_staging_bytes = checkpoint
            .blocks
            .iter()
            .map(|block| block.max_host_read_staging_bytes)
            .max()
            .unwrap();
        let max_streamed_block_host_overlap_bytes = checkpoint
            .blocks
            .iter()
            .map(|block| block.encoded_host_bytes + block.max_host_read_staging_bytes)
            .max()
            .unwrap();
        let fixed_transformer_load_host_staging_bytes = checkpoint
            .fixed_transformer_encoded_host_bytes
            + checkpoint.fixed_transformer_max_host_read_staging_bytes;
        let predicted_host_increment_bytes = sum(&[
            artifact_host_bytes,
            100,
            qwen_host_workspace_bytes,
            condition_backing_host_bytes,
            endpoint_encoded_host_bytes,
            normalized_endpoint_host_bytes,
            schedule_host_bytes,
            packed_layout_host_bytes,
            packed_layout_construction_staging_host_bytes,
            packed_layout_freeze_staging_host_bytes,
            text_modality_tags_host_bytes,
            noise_cpu_staging_host_bytes,
            1_000,
            2_000,
            max_streamed_block_host_overlap_bytes,
            fixed_transformer_load_host_staging_bytes,
            5_000,
            1_000,
            waveform_host_bytes,
            10_000,
            2_000,
        ]);
        let mut budget = H3FactoryTargetBudgetInput {
            identity_sha256: String::new(),
            load_drop_policy: H3FactoryTargetLoadDropPolicy::LoadVaesLoadQwenEncodeTransferDropQwenEncodeConditionsAllocateNoiseLoadTransformerDenoiseDropTransformerDecodeVisualAudioDropVaesMux,
            artifacts,
            artifact_host_bytes,
            fixed_runtime_host_bytes: 100,
            qwen_host_parameter_bytes,
            qwen_host_activation_bytes,
            qwen_host_output_state_bytes,
            qwen_host_workspace_bytes,
            condition_backing_host_bytes,
            endpoint_encoded_host_bytes,
            normalized_endpoint_host_bytes,
            schedule_host_bytes,
            packed_layout_host_bytes,
            packed_layout_construction_staging_host_bytes,
            packed_layout_freeze_staging_host_bytes,
            text_modality_tags_host_bytes,
            noise_cpu_staging_host_bytes,
            vae_peak_host_io_buffer_bytes: 1_000,
            vae_peak_host_mapped_file_bytes: 2_000,
            vae_peak_staging_disk_bytes: 3_000,
            max_host_read_staging_bytes,
            max_streamed_block_host_overlap_bytes,
            fixed_transformer_load_host_staging_bytes,
            encoded_video_host_bytes_bound: 5_000,
            thumbnail_host_bytes_bound: 1_000,
            waveform_host_bytes,
            mux_output_host_bytes_bound: 10_000,
            aac_mux_staging_host_bytes: 2_000,
            predicted_host_increment_bytes,
            fixed_runtime_device_bytes,
            fixed_transformer_device_bytes,
            visual_vae_resident_device_bytes: 400,
            audio_vae_resident_device_bytes: 500,
            attempt_resident_vae_device_bytes: retained_vaes,
            vae_construction_device_workspace_bytes: 100,
            vae_memory_evidence_identity_sha256: sha('e'),
            qwen_device_parameter_bytes: 0,
            qwen_activation_device_bytes: 0,
            qwen_output_state_device_bytes,
            qwen_output_transfer_device_bytes: qwen_output_state_device_bytes,
            condition_vae_workspace_device_bytes,
            condition_latent_backing_device_bytes,
            target_video_latent_device_bytes,
            target_audio_latent_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            denoise_copy_policy: H3FactoryTargetDenoiseCopyPolicy::CandleF32PairedResMultistepV2,
            denoise_tensor_copy_workspace_device_bytes,
            audio_waveform_device_bytes: waveform_host_bytes,
            attention_workspace_device_bytes,
            ffn_workspace_device_bytes,
            decoder_tile_workspace_device_bytes,
            audio_decode_workspace_device_bytes,
            resident_block_device_bytes: 0,
            streamed_block_device_bytes: checkpoint
                .blocks
                .iter()
                .map(|block| block.protected_device_bytes)
                .sum(),
            prefetch_device_bytes: 0,
            dequantization_workspace_device_bytes: max_device_weight_staging_bytes,
            protected_block_device_bytes: checkpoint
                .blocks
                .iter()
                .map(|block| block.protected_device_bytes)
                .sum(),
            streamed_block_device_overlap_bytes,
            max_device_weight_staging_bytes,
            fixed_transformer_load_device_staging_bytes,
            vae_load_phase_device_bytes,
            qwen_encode_phase_device_bytes,
            qwen_transfer_phase_device_bytes,
            condition_encode_phase_device_bytes,
            noise_allocation_phase_device_bytes,
            transformer_load_phase_device_bytes,
            denoise_phase_device_bytes,
            visual_decode_phase_device_bytes,
            audio_decode_phase_device_bytes,
            waveform_transfer_phase_device_bytes,
            mux_phase_device_bytes: 0,
            predicted_device_peak_bytes,
        };
        budget.identity_sha256 = expected_h3_factory_target_budget_identity(&budget);
        budget
    }

    fn prepared_attempt() -> H3FactoryPreparedAttemptInput {
        let request = prepared_request();
        let raw_checkpoint = raw_checkpoint();
        let target_budget = target_budget(&request, &raw_checkpoint);
        let mut attempt = H3FactoryPreparedAttemptInput {
            identity_sha256: String::new(),
            execution_fingerprint: sha('a'),
            request,
            raw_checkpoint,
            target_budget,
        };
        attempt.identity_sha256 = expected_h3_factory_prepared_attempt_identity(&attempt);
        attempt
    }

    fn attention() -> H3FactoryAttentionInput {
        let runtime_backend = H3AttentionBackend::FlashAttentionV2;
        let kernel = H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011;
        let activation = H3AttentionActivation::ReleaseCandidateQualificationOnly;
        let device = H3AttentionDevice::Cuda {
            compute_capability: Some((8, 9)),
        };
        let model_contract = H3AttentionModelContract::released_bf16();
        let runtime_identity_sha256 = H3AttentionRuntimeAuthority::expected_identity_for(
            runtime_backend,
            kernel,
            activation,
            device,
            model_contract,
        )
        .unwrap();
        H3FactoryAttentionInput {
            generic_backend: AttentionBackend::Flash,
            generic_chunk: AttentionChunkPolicy::Off,
            runtime_backend,
            kernel,
            activation,
            device,
            model_contract,
            runtime_identity_sha256,
            qualification_kernel_identity: kernel.identity().into(),
            qualification_sha256: sha('b'),
            full_noncausal: true,
            lossless: true,
        }
    }

    fn components() -> Vec<H3FactoryComponentAuthority> {
        [
            (H3FactoryComponentRole::Conditioner, sha('a'), sha('6')),
            // Raw checkpoint and logical transformer digests may be equal;
            // their typed authority domains remain distinct.
            (H3FactoryComponentRole::Transformer, sha('3'), sha('7')),
            (H3FactoryComponentRole::VisualVae, sha('4'), sha('8')),
            (H3FactoryComponentRole::AudioVae, sha('5'), sha('9')),
        ]
        .into_iter()
        .map(|(role, content, validation)| {
            H3FactoryComponentAuthority::new(role, content, validation).unwrap()
        })
        .collect()
    }

    fn exact_input() -> H3FactoryAuthorityInput {
        let prepared_attempt = prepared_attempt();
        let attention = attention();
        let execution_budget_echo = H3FactoryExecutionBudgetEchoInput {
            prepared_attempt_identity_sha256: prepared_attempt.identity_sha256.clone(),
            device_peak_bytes: prepared_attempt.target_budget.predicted_device_peak_bytes,
            host_increment_bytes: prepared_attempt
                .target_budget
                .predicted_host_increment_bytes,
        };
        H3FactoryAuthorityInput {
            model: contract::FL2VA_COMFY.into(),
            device_id: "gpu-0".into(),
            device_ordinal: 0,
            compute_capability: (8, 9),
            execution_fingerprint: sha('a'),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 2048,
            qwen_host_resident_parameter_bytes: 2048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1024,
            qwen_output_text_rows: 1,
            qwen_vision_rows: 64,
            condition_visual_rows: 384,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: attention.qualification_kernel_identity.clone(),
            attention_qualification_sha256: sha('b'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            attention_runtime: Some(attention),
            block_offload: true,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('d'),
                qwen_policy_sha256: sha('c'),
                pruned_adaln_table_sha256: sha('e'),
            },
            prepared_attempt: Some(prepared_attempt),
            execution_budget_echo: Some(execution_budget_echo),
            components: components(),
        }
    }

    fn exact_cuda_input() -> H3FactoryAuthorityInput {
        let mut input = exact_input();
        input.conditioner_placement = H3FactoryConditionerPlacement::AssignedCudaThenDrop;
        input.qwen_device_resident_parameter_bytes = input.qwen_parameter_bytes;

        let budget = &mut input.prepared_attempt.as_mut().unwrap().target_budget;
        let prior_qwen_host_workspace_bytes = budget.qwen_host_workspace_bytes;
        budget.qwen_host_activation_bytes = 0;
        budget.qwen_host_output_state_bytes = 0;
        budget.qwen_host_workspace_bytes = budget.qwen_host_parameter_bytes;
        budget.qwen_device_parameter_bytes = input.qwen_device_resident_parameter_bytes;
        budget.qwen_activation_device_bytes = input.qwen_activation_workspace_bytes;
        budget.qwen_output_transfer_device_bytes = 0;
        budget.qwen_encode_phase_device_bytes = sum(&[
            budget.fixed_runtime_device_bytes,
            budget.attempt_resident_vae_device_bytes,
            budget.qwen_device_parameter_bytes,
            budget.qwen_activation_device_bytes,
            budget.qwen_output_state_device_bytes,
        ]);
        budget.qwen_transfer_phase_device_bytes = 0;
        budget.predicted_host_increment_bytes = budget
            .predicted_host_increment_bytes
            .checked_sub(prior_qwen_host_workspace_bytes)
            .unwrap()
            .checked_add(budget.qwen_host_workspace_bytes)
            .unwrap();
        budget.predicted_device_peak_bytes = [
            budget.vae_load_phase_device_bytes,
            budget.qwen_encode_phase_device_bytes,
            budget.qwen_transfer_phase_device_bytes,
            budget.condition_encode_phase_device_bytes,
            budget.noise_allocation_phase_device_bytes,
            budget.transformer_load_phase_device_bytes,
            budget.denoise_phase_device_bytes,
            budget.visual_decode_phase_device_bytes,
            budget.audio_decode_phase_device_bytes,
            budget.waveform_transfer_phase_device_bytes,
            budget.mux_phase_device_bytes,
        ]
        .into_iter()
        .max()
        .unwrap();
        refresh_nested_identities(&mut input);
        input
    }

    fn legacy_input(model: &str) -> H3FactoryAuthorityInput {
        let mut input = exact_input();
        input.model = model.into();
        input.resident_block_count = 8;
        input.prefetch_depth = 1;
        input.attention_runtime = None;
        input.prepared_attempt = None;
        input.execution_budget_echo = None;
        input
    }

    fn refresh_nested_identities(input: &mut H3FactoryAuthorityInput) {
        let attempt = input.prepared_attempt.as_mut().unwrap();
        attempt.request.identity_sha256 =
            expected_h3_factory_prepared_request_identity(&attempt.request);
        attempt.raw_checkpoint.identity_sha256 =
            expected_h3_factory_raw_checkpoint_identity(&attempt.raw_checkpoint);
        attempt.target_budget.identity_sha256 =
            expected_h3_factory_target_budget_identity(&attempt.target_budget);
        attempt.identity_sha256 = expected_h3_factory_prepared_attempt_identity(attempt);
        let budget = input.execution_budget_echo.as_mut().unwrap();
        budget.prepared_attempt_identity_sha256 = attempt.identity_sha256.clone();
        budget.device_peak_bytes = attempt.target_budget.predicted_device_peak_bytes;
        budget.host_increment_bytes = attempt.target_budget.predicted_host_increment_bytes;
    }

    fn refresh_attention_identity(attention: &mut H3FactoryAttentionInput) {
        attention.runtime_identity_sha256 = H3AttentionRuntimeAuthority::expected_identity_for(
            attention.runtime_backend,
            attention.kernel,
            attention.activation,
            attention.device,
            attention.model_contract,
        )
        .unwrap_or_else(|_| sha('f'));
    }

    fn authority() -> FrozenH3FactoryAuthority {
        FrozenH3FactoryAuthority::new_contract_only(exact_input()).unwrap()
    }

    #[test]
    fn target_authority_binds_typed_domains_but_execution_stays_closed() {
        let input = exact_input();
        let prepared = input.prepared_attempt.as_ref().unwrap();
        let logical_transformer = input
            .components
            .iter()
            .find(|component| component.role == H3FactoryComponentRole::Transformer)
            .unwrap();
        assert_eq!(
            prepared.raw_checkpoint.raw_content_sha256,
            logical_transformer.content_sha256
        );
        let prepared_identities = (
            prepared.identity_sha256.clone(),
            prepared.target_budget.identity_sha256.clone(),
        );
        let authority = FrozenH3FactoryAuthority::new_contract_only(input).unwrap();
        assert_eq!(authority.canonical_model(), contract::FL2VA_COMFY);
        assert_eq!(authority.device_id(), "gpu-0");
        assert_eq!(authority.device_ordinal(), 0);
        assert_eq!(authority.execution_fingerprint(), sha('a'));
        assert_eq!(
            authority.prepared_target_attempt_identities(),
            Some((
                prepared_identities.0.as_str(),
                prepared_identities.1.as_str(),
            ))
        );
        assert_ne!(
            authority.attention_runtime_identity_sha256(),
            authority.attention_qualification_sha256()
        );
        assert_eq!(authority.identity_sha256().len(), 64);
        assert_eq!(authority.component_set_identity_sha256().len(), 64);
        #[cfg(feature = "h3-private-uat")]
        {
            assert!(authority.private_fl2va_runtime_authority().is_err());
            let vae = authority.private_vae_adapter_authority().unwrap();
            assert_eq!(vae.factory_identity_sha256, authority.identity_sha256());
            assert_eq!(
                vae.backend_plan_identity_sha256,
                authority.backend_plan_identity_sha256()
            );
            assert_eq!(
                vae.component_set_identity_sha256,
                authority.component_set_identity_sha256()
            );
            assert_eq!(
                vae.vae_artifact_plan_identity_sha256,
                expected_h3_comfy_vae_artifact_plan_identity(contract::FL2VA_COMFY).unwrap()
            );
        }
        assert!(authority.block_offload());
        assert!(matches!(
            authority.quantization(),
            H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq { .. }
        ));
        assert_eq!(
            h3_factory_activation_prerequisites(),
            &[
                H3FactoryActivationPrerequisite::OpenedComponentMemoryEvidence,
                H3FactoryActivationPrerequisite::PreparedCheckpointExecutionEcho,
                H3FactoryActivationPrerequisite::ConsumingTargetLifetimeTransitions,
                H3FactoryActivationPrerequisite::RetainedTensorOverlapBudget,
                H3FactoryActivationPrerequisite::HostLayoutAndTransientBudget,
                H3FactoryActivationPrerequisite::EndpointPreprocessTransientBudget,
                H3FactoryActivationPrerequisite::PerAttemptRuntimeConstruction,
                H3FactoryActivationPrerequisite::OneShotSchedulerLease,
                H3FactoryActivationPrerequisite::SameAttemptCancellationCoverage,
            ]
        );
        assert!(authority
            .validate_for_dispatch(
                contract::FL2VA_COMFY,
                contract::FAMILY,
                0,
                true,
                AttentionBackend::Flash,
                AttentionChunkPolicy::Off,
            )
            .is_err());
    }

    #[test]
    fn cuda_conditioner_target_budget_succeeds_and_crossed_placements_fail_closed() {
        let cuda = FrozenH3FactoryAuthority::new_contract_only(exact_cuda_input()).unwrap();
        assert_eq!(
            cuda.conditioner_placement(),
            H3FactoryConditionerPlacement::AssignedCudaThenDrop
        );
        assert_eq!(
            cuda.qwen_device_resident_parameter_bytes(),
            cuda.qwen_parameter_bytes()
        );

        let mut cuda_route_with_host_budget = exact_input();
        cuda_route_with_host_budget.conditioner_placement =
            H3FactoryConditionerPlacement::AssignedCudaThenDrop;
        cuda_route_with_host_budget.qwen_device_resident_parameter_bytes =
            cuda_route_with_host_budget.qwen_parameter_bytes;
        assert!(FrozenH3FactoryAuthority::new_contract_only(cuda_route_with_host_budget).is_err());

        let mut host_route_with_cuda_budget = exact_cuda_input();
        host_route_with_cuda_budget.conditioner_placement =
            H3FactoryConditionerPlacement::HostCpuThenDrop;
        host_route_with_cuda_budget.qwen_device_resident_parameter_bytes = 0;
        assert!(FrozenH3FactoryAuthority::new_contract_only(host_route_with_cuda_budget).is_err());
    }

    #[test]
    fn post_admission_mutations_are_rejected() {
        for mutate in [
            (|value: &mut FrozenH3FactoryAuthority| value.device_ordinal += 1)
                as fn(&mut FrozenH3FactoryAuthority),
            |value| value.condition_visual_rows += 1,
            |value| {
                value
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .request
                    .rows
                    .target_audio_rows += 1;
            },
            |value| {
                value
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .target_budget
                    .visual_decode_phase_device_bytes -= 1;
            },
            |value| {
                value
                    .attention_runtime
                    .as_mut()
                    .unwrap()
                    .runtime_identity_sha256 = sha('8');
            },
            |value| {
                value
                    .execution_budget_echo
                    .as_mut()
                    .unwrap()
                    .device_peak_bytes += 1;
            },
            |value| value.attention_kernel_identity.push_str("-changed"),
            |value| value.block_offload = false,
            |value| {
                value.quantization = H3FactoryQuantizationAuthority::OfficialBf16;
            },
            |value| {
                value.comfy_vae_artifact_plan_identity_sha256 = Some(sha('f'));
            },
        ] {
            let mut changed = authority();
            mutate(&mut changed);
            assert!(changed.validate_frozen().is_err());
        }
    }

    #[test]
    fn legitimate_seed_endpoint_and_raw_domain_changes_form_distinct_authorities() {
        let baseline = authority().identity_sha256().to_owned();

        let mut seed = exact_input();
        seed.prepared_attempt.as_mut().unwrap().request.seed += 1;
        refresh_nested_identities(&mut seed);
        let seed = FrozenH3FactoryAuthority::new_contract_only(seed).unwrap();
        assert_ne!(seed.identity_sha256(), baseline);

        let mut normalized = exact_input();
        normalized
            .prepared_attempt
            .as_mut()
            .unwrap()
            .request
            .endpoints[0]
            .normalized_cpu_content_sha256 = sha('6');
        refresh_nested_identities(&mut normalized);
        let normalized = FrozenH3FactoryAuthority::new_contract_only(normalized).unwrap();
        assert_ne!(normalized.identity_sha256(), baseline);

        let mut raw_domain = exact_input();
        let attempt = raw_domain.prepared_attempt.as_mut().unwrap();
        attempt.raw_checkpoint.raw_content_sha256 = sha('f');
        let raw_artifact = attempt
            .target_budget
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.role == H3FactoryArtifactHostRole::RawTransformerCheckpoint)
            .unwrap();
        raw_artifact.content_sha256 = sha('f');
        refresh_nested_identities(&mut raw_domain);
        let raw_domain = FrozenH3FactoryAuthority::new_contract_only(raw_domain).unwrap();
        assert_ne!(raw_domain.identity_sha256(), baseline);
    }

    #[test]
    fn request_and_checkpoint_relations_reject_rehashed_mutations() {
        let mutations: &[fn(&mut H3FactoryAuthorityInput)] = &[
            |input| {
                input.prepared_attempt.as_mut().unwrap().request.mode = Mode::LastFrameToAudioVideo;
            },
            |input| {
                input.prepared_attempt.as_mut().unwrap().request.endpoints[0].normalized_shape
                    [4] -= 32;
            },
            |input| {
                input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .request
                    .audio_samples_per_channel += 800;
            },
            |input| {
                let request = &mut input.prepared_attempt.as_mut().unwrap().request;
                request.grid_points = H3_FACTORY_MAX_GRID_POINTS + 1;
                request.denoise_forward_count = H3_FACTORY_MAX_GRID_POINTS;
            },
            |input| {
                input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .request
                    .denoise_forward_count -= 1;
            },
            |input| {
                let request = &mut input.prepared_attempt.as_mut().unwrap().request;
                request.rows.condition_audio_rows = 1;
                request.rows.total_packed_rows += 1;
            },
            |input| {
                let request = &mut input.prepared_attempt.as_mut().unwrap().request;
                request.rows.target_video_rows += 1;
                request.rows.total_packed_rows += 1;
            },
            |input| {
                let request = &mut input.prepared_attempt.as_mut().unwrap().request;
                request.rows.target_audio_rows += 1;
                request.rows.total_packed_rows += 1;
            },
            |input| {
                input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .request
                    .rows
                    .total_packed_rows += 1;
            },
            |input| {
                input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .request
                    .rows
                    .qwen_vision_rows += 1;
            },
            |input| {
                let checkpoint = &mut input.prepared_attempt.as_mut().unwrap().raw_checkpoint;
                let encoded = checkpoint
                    .blocks
                    .iter()
                    .map(|block| block.encoded_host_bytes)
                    .sum::<u64>();
                checkpoint.verified_file_bytes =
                    checkpoint.fixed_transformer_encoded_host_bytes + encoded - 1;
            },
            |input| {
                input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .raw_checkpoint
                    .blocks
                    .swap(0, 1);
            },
            |input| {
                input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .raw_checkpoint
                    .blocks[0]
                    .protected_device_bytes += 1;
            },
            |input| {
                input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .raw_checkpoint
                    .fixed_transformer_encoded_host_bytes += 1;
            },
            |input| {
                let artifact = input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .target_budget
                    .artifacts
                    .iter_mut()
                    .find(|artifact| {
                        artifact.role == H3FactoryArtifactHostRole::RawTransformerCheckpoint
                    })
                    .unwrap();
                artifact.content_sha256 = sha('f');
            },
        ];
        for mutate in mutations {
            let mut input = exact_input();
            mutate(&mut input);
            refresh_nested_identities(&mut input);
            assert!(FrozenH3FactoryAuthority::new_contract_only(input).is_err());
        }
    }

    #[test]
    fn attention_axes_and_generic_projection_fail_closed() {
        let mutations: &[fn(&mut H3FactoryAuthorityInput)] = &[
            |input| {
                let attention = input.attention_runtime.as_mut().unwrap();
                attention.runtime_backend = H3AttentionBackend::BoundedDenseMath;
                refresh_attention_identity(attention);
            },
            |input| {
                let attention = input.attention_runtime.as_mut().unwrap();
                attention.kernel = H3AttentionKernel::CandleDenseF32V011;
                refresh_attention_identity(attention);
            },
            |input| {
                let attention = input.attention_runtime.as_mut().unwrap();
                attention.activation = H3AttentionActivation::SyntheticCorrectnessOnly;
                refresh_attention_identity(attention);
            },
            |input| {
                let attention = input.attention_runtime.as_mut().unwrap();
                attention.device = H3AttentionDevice::Cpu;
                refresh_attention_identity(attention);
            },
            |input| {
                let attention = input.attention_runtime.as_mut().unwrap();
                attention.model_contract.heads -= 1;
                refresh_attention_identity(attention);
            },
            |input| {
                input
                    .attention_runtime
                    .as_mut()
                    .unwrap()
                    .runtime_identity_sha256 = sha('f');
            },
            |input| {
                input.attention_runtime.as_mut().unwrap().generic_backend = AttentionBackend::Math;
            },
            |input| {
                input.attention_runtime.as_mut().unwrap().generic_chunk =
                    AttentionChunkPolicy::Auto;
            },
            |input| {
                input
                    .attention_runtime
                    .as_mut()
                    .unwrap()
                    .qualification_kernel_identity = "crossed-kernel".into();
            },
            |input| {
                input
                    .attention_runtime
                    .as_mut()
                    .unwrap()
                    .qualification_sha256 = sha('a');
            },
            |input| {
                input.attention_runtime.as_mut().unwrap().full_noncausal = false;
            },
            |input| {
                input.attention_runtime.as_mut().unwrap().lossless = false;
            },
            |input| {
                input.compute_capability = (9, 0);
            },
            |input| {
                let attention = input.attention_runtime.as_mut().unwrap();
                attention.runtime_backend = H3AttentionBackend::BoundedDenseMath;
                attention.kernel = H3AttentionKernel::CandleDenseF32V011;
                attention.activation = H3AttentionActivation::SyntheticCorrectnessOnly;
                attention.device = H3AttentionDevice::Cpu;
                refresh_attention_identity(attention);
            },
        ];
        for mutate in mutations {
            let mut input = exact_input();
            mutate(&mut input);
            assert!(FrozenH3FactoryAuthority::new_contract_only(input).is_err());
        }
    }

    #[test]
    fn partial_target_triad_and_budget_mismatches_are_rejected() {
        let mut missing_attempt = exact_input();
        missing_attempt.prepared_attempt = None;
        assert!(FrozenH3FactoryAuthority::new_contract_only(missing_attempt).is_err());

        let mut missing_budget = exact_input();
        missing_budget.execution_budget_echo = None;
        assert!(FrozenH3FactoryAuthority::new_contract_only(missing_budget).is_err());

        let mut missing_attention = exact_input();
        missing_attention.attention_runtime = None;
        assert!(FrozenH3FactoryAuthority::new_contract_only(missing_attention).is_err());

        let mut typed_only = legacy_input(contract::FL2VA_COMFY);
        typed_only.attention_runtime = Some(attention());
        assert!(FrozenH3FactoryAuthority::new_contract_only(typed_only).is_err());

        let mut device_budget = exact_input();
        device_budget
            .execution_budget_echo
            .as_mut()
            .unwrap()
            .device_peak_bytes += 1;
        assert!(FrozenH3FactoryAuthority::new_contract_only(device_budget).is_err());

        let mut host_budget = exact_input();
        host_budget
            .execution_budget_echo
            .as_mut()
            .unwrap()
            .host_increment_bytes += 1;
        assert!(FrozenH3FactoryAuthority::new_contract_only(host_budget).is_err());

        let mut jointly_rehashed = exact_input();
        jointly_rehashed
            .prepared_attempt
            .as_mut()
            .unwrap()
            .target_budget
            .predicted_device_peak_bytes += 1;
        refresh_nested_identities(&mut jointly_rehashed);
        assert!(FrozenH3FactoryAuthority::new_contract_only(jointly_rehashed).is_err());
    }

    #[test]
    fn target_budget_identity_hashes_every_numeric_and_evidence_field() {
        let base = exact_input().prepared_attempt.unwrap().target_budget;
        let baseline = expected_h3_factory_target_budget_identity(&base);
        macro_rules! assert_numeric_fields_hashed {
            ($($field:ident),+ $(,)?) => {$({
                let mut changed = base.clone();
                changed.$field = changed.$field.checked_add(1).unwrap();
                assert_ne!(
                    expected_h3_factory_target_budget_identity(&changed),
                    baseline,
                    "{} escaped the target-budget identity",
                    stringify!($field),
                );
            })+};
        }
        assert_numeric_fields_hashed!(
            artifact_host_bytes,
            fixed_runtime_host_bytes,
            qwen_host_parameter_bytes,
            qwen_host_activation_bytes,
            qwen_host_output_state_bytes,
            qwen_host_workspace_bytes,
            condition_backing_host_bytes,
            endpoint_encoded_host_bytes,
            normalized_endpoint_host_bytes,
            schedule_host_bytes,
            packed_layout_host_bytes,
            packed_layout_construction_staging_host_bytes,
            packed_layout_freeze_staging_host_bytes,
            text_modality_tags_host_bytes,
            noise_cpu_staging_host_bytes,
            vae_peak_host_io_buffer_bytes,
            vae_peak_host_mapped_file_bytes,
            vae_peak_staging_disk_bytes,
            max_host_read_staging_bytes,
            max_streamed_block_host_overlap_bytes,
            fixed_transformer_load_host_staging_bytes,
            encoded_video_host_bytes_bound,
            thumbnail_host_bytes_bound,
            waveform_host_bytes,
            mux_output_host_bytes_bound,
            aac_mux_staging_host_bytes,
            predicted_host_increment_bytes,
            fixed_runtime_device_bytes,
            fixed_transformer_device_bytes,
            visual_vae_resident_device_bytes,
            audio_vae_resident_device_bytes,
            attempt_resident_vae_device_bytes,
            vae_construction_device_workspace_bytes,
            qwen_device_parameter_bytes,
            qwen_activation_device_bytes,
            qwen_output_state_device_bytes,
            qwen_output_transfer_device_bytes,
            condition_vae_workspace_device_bytes,
            condition_latent_backing_device_bytes,
            target_video_latent_device_bytes,
            target_audio_latent_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            denoise_tensor_copy_workspace_device_bytes,
            audio_waveform_device_bytes,
            attention_workspace_device_bytes,
            ffn_workspace_device_bytes,
            decoder_tile_workspace_device_bytes,
            audio_decode_workspace_device_bytes,
            resident_block_device_bytes,
            streamed_block_device_bytes,
            prefetch_device_bytes,
            dequantization_workspace_device_bytes,
            protected_block_device_bytes,
            streamed_block_device_overlap_bytes,
            max_device_weight_staging_bytes,
            fixed_transformer_load_device_staging_bytes,
            vae_load_phase_device_bytes,
            qwen_encode_phase_device_bytes,
            qwen_transfer_phase_device_bytes,
            condition_encode_phase_device_bytes,
            noise_allocation_phase_device_bytes,
            transformer_load_phase_device_bytes,
            denoise_phase_device_bytes,
            visual_decode_phase_device_bytes,
            audio_decode_phase_device_bytes,
            waveform_transfer_phase_device_bytes,
            mux_phase_device_bytes,
            predicted_device_peak_bytes,
        );

        let mut load_drop_policy = base.clone();
        load_drop_policy.load_drop_policy = H3FactoryTargetLoadDropPolicy::IdentityMutationSentinel;
        assert_ne!(
            expected_h3_factory_target_budget_identity(&load_drop_policy),
            baseline
        );

        let mut denoise_copy_policy = base.clone();
        denoise_copy_policy.denoise_copy_policy =
            H3FactoryTargetDenoiseCopyPolicy::IdentityMutationSentinel;
        assert_ne!(
            expected_h3_factory_target_budget_identity(&denoise_copy_policy),
            baseline
        );

        for (field, mutate) in [
            (
                "role",
                (|artifact: &mut H3FactoryArtifactHostInput| {
                    artifact.role = H3FactoryArtifactHostRole::TransformerSupport;
                }) as fn(&mut H3FactoryArtifactHostInput),
            ),
            ("index", |artifact| artifact.index += 1),
            ("content_sha256", |artifact| {
                artifact.content_sha256 = sha('f');
            }),
            ("bytes", |artifact| artifact.bytes += 1),
        ] {
            let mut changed = base.clone();
            mutate(&mut changed.artifacts[0]);
            assert_ne!(
                expected_h3_factory_target_budget_identity(&changed),
                baseline,
                "artifact {field} escaped the target-budget identity",
            );
        }

        let mut evidence = base;
        evidence.vae_memory_evidence_identity_sha256 = sha('f');
        assert_ne!(
            expected_h3_factory_target_budget_identity(&evidence),
            baseline
        );
    }

    #[test]
    fn every_derived_target_budget_field_is_relationally_checked() {
        macro_rules! assert_rejected_fields {
            ($($field:ident),+ $(,)?) => {$({
                let mut input = exact_input();
                let budget = &mut input
                    .prepared_attempt
                    .as_mut()
                    .unwrap()
                    .target_budget;
                budget.$field = budget.$field.checked_add(1).unwrap();
                refresh_nested_identities(&mut input);
                assert!(
                    FrozenH3FactoryAuthority::new_contract_only(input).is_err(),
                    "{} was accepted after identities and budget echo were refreshed",
                    stringify!($field),
                );
            })+};
        }
        assert_rejected_fields!(
            artifact_host_bytes,
            qwen_host_output_state_bytes,
            qwen_host_workspace_bytes,
            endpoint_encoded_host_bytes,
            normalized_endpoint_host_bytes,
            schedule_host_bytes,
            packed_layout_host_bytes,
            packed_layout_construction_staging_host_bytes,
            packed_layout_freeze_staging_host_bytes,
            text_modality_tags_host_bytes,
            noise_cpu_staging_host_bytes,
            max_host_read_staging_bytes,
            max_streamed_block_host_overlap_bytes,
            fixed_transformer_load_host_staging_bytes,
            waveform_host_bytes,
            predicted_host_increment_bytes,
            fixed_transformer_device_bytes,
            attempt_resident_vae_device_bytes,
            qwen_output_state_device_bytes,
            qwen_output_transfer_device_bytes,
            condition_latent_backing_device_bytes,
            target_video_latent_device_bytes,
            target_audio_latent_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            denoise_tensor_copy_workspace_device_bytes,
            audio_waveform_device_bytes,
            resident_block_device_bytes,
            streamed_block_device_bytes,
            prefetch_device_bytes,
            dequantization_workspace_device_bytes,
            protected_block_device_bytes,
            streamed_block_device_overlap_bytes,
            max_device_weight_staging_bytes,
            fixed_transformer_load_device_staging_bytes,
            vae_load_phase_device_bytes,
            qwen_encode_phase_device_bytes,
            qwen_transfer_phase_device_bytes,
            condition_encode_phase_device_bytes,
            noise_allocation_phase_device_bytes,
            transformer_load_phase_device_bytes,
            denoise_phase_device_bytes,
            visual_decode_phase_device_bytes,
            audio_decode_phase_device_bytes,
            waveform_transfer_phase_device_bytes,
            mux_phase_device_bytes,
            predicted_device_peak_bytes,
        );
    }

    #[test]
    fn exact_authority_still_rejects_while_runtime_and_factory_registry_are_closed() {
        let authority = authority();
        let error = authority
            .validate_for_dispatch(
                contract::FL2VA_COMFY,
                contract::FAMILY,
                0,
                true,
                AttentionBackend::Flash,
                AttentionChunkPolicy::Off,
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("registry remains runtime unavailable"));
    }

    #[test]
    fn ref2va_contract_authority_is_distinct_while_missing_components_fail_closed() {
        let ref2va =
            FrozenH3FactoryAuthority::new_contract_only(legacy_input(contract::REF2VA_COMFY))
                .unwrap();
        assert_eq!(ref2va.task(), Task::Ref2va);
        assert_eq!(ref2va.canonical_model(), contract::REF2VA_COMFY);
        assert_ne!(ref2va.identity_sha256(), authority().identity_sha256());
        let error = ref2va
            .validate_for_dispatch(
                contract::REF2VA_COMFY,
                contract::FAMILY,
                0,
                true,
                AttentionBackend::Flash,
                AttentionChunkPolicy::Off,
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("registry remains runtime unavailable"));

        let mut exact_ref2va = exact_input();
        exact_ref2va.model = contract::REF2VA_COMFY.into();
        assert!(FrozenH3FactoryAuthority::new_contract_only(exact_ref2va).is_err());

        let mut missing_components = legacy_input(contract::REF2VA_COMFY);
        missing_components.components.clear();
        assert!(FrozenH3FactoryAuthority::new_contract_only(missing_components).is_err());
    }
}
