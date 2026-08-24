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
use crate::minimax_h3::sampler::{H3DualSchedule, H3SamplerKind, H3_VIDEO_SHIFT};
use crate::minimax_h3::vae_runtime::expected_h3_comfy_vae_artifact_plan_identity;
use crate::minimax_h3::{FrozenH3ConditionerPlacement, H3ConditionerExecution};

const H3_FACTORY_AUTHORITY_SCHEMA_VERSION: u32 = 6;
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

/// Where the scheduler placed the Qwen conditioner for one frozen attempt.
///
/// CUDA and Metal use distinct device-resident variants so a frozen authority
/// can never silently cross backend classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryConditionerPlacement {
    AssignedCudaThenDrop,
    AssignedMetalThenDrop,
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

/// Modality of one ordered Ref2VA reference. This mirrors
/// [`mold_core::GenerationReferenceKind`] as a factory-local domain so the
/// authority never inherits a wire enum's future variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryReferenceKind {
    Image,
    Video,
    Audio,
}

/// One ordered Ref2VA reference, frozen at its normalized preprocessing shape.
///
/// Every geometry field is the exact value
/// `mold_core::minimax_h3::reference_prepared_shapes_for_target` derived at
/// admission. The validator recomputes the row and retained-byte charges from
/// that geometry, so an understated charge is rejected rather than trusted.
/// There is deliberately no path, upload handle, or byte buffer here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryReferenceInput {
    /// One-based request order.
    pub index: u32,
    pub kind: H3FactoryReferenceKind,
    pub content_sha256: String,
    /// Prepared-shape contract version the admission geometry was derived at.
    pub preprocess_version: u32,
    /// Normalized visual canvas. Absent exactly for audio-only references.
    pub normalized_width: Option<u32>,
    pub normalized_height: Option<u32>,
    /// Exact CFR-normalized frame count, the visual-VAE prefix taken from it,
    /// and the 2 fps Qwen cursor sample count. Video references only.
    pub normalized_video_frames: Option<u32>,
    pub video_frames: Option<u32>,
    pub qwen_video_frames: Option<u32>,
    /// Normalized 32 kHz stereo sample count. Absent without a soundtrack.
    pub audio_samples_per_channel: Option<u64>,
    /// NATIVE decoded geometry, before any normalization.
    ///
    /// The decoder retains what it decoded, not what preprocessing will
    /// produce, and every reference stays retained until its own preprocess
    /// step consumes it. These fields are what make that peak derivable; a
    /// visual reference without them cannot be priced and is refused.
    pub native_width: Option<u32>,
    pub native_height: Option<u32>,
    /// Native soundtrack sample count and channel count, at the source rate.
    pub native_audio_samples_per_channel: Option<u64>,
    pub native_audio_channels: Option<u16>,
    /// Packed condition rows this reference contributes.
    pub visual_rows: u64,
    pub audio_rows: u64,
    /// Qwen vision pads this reference contributes to `qwen_vision_rows`.
    pub qwen_vision_rows: u64,
    /// Host bytes the backend retains for this reference between preprocessing
    /// and its two encoders: normalized RGB8 frames plus the normalized f32
    /// stereo waveform. Recomputed by the validator from the geometry.
    pub normalized_host_bytes: u64,
    /// Host bytes the backend retains for this reference between its decode
    /// and its preprocess: the NATIVE decoded frames it selected plus the
    /// native f32 soundtrack. Recomputed by the validator from the native
    /// geometry, and charged for every reference simultaneously because the
    /// orchestrator decodes the whole ordered set before preprocessing any.
    pub native_host_bytes: u64,
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
    /// Ordered Ref2VA references. Always empty for FL2VA, whose conditioning
    /// rides the endpoint contract instead.
    pub references: Vec<H3FactoryReferenceInput>,
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
    /// Parsed safetensors header the opened checkpoint retains for the whole
    /// stream lifetime. Anonymous host bytes, unlike the tensor payload the
    /// stream reads through a bounded buffer.
    pub retained_header_host_bytes: u64,
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
///
/// Both VAEs are parked once conditions are encoded and reconstructed after
/// the transformer is dropped, so the phase sums below charge their resident
/// weights on either side of denoise but never across it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryTargetLoadDropPolicy {
    LoadQwenEncodeTransferDropQwenLoadVaesEncodeConditionsParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux,
    /// Ref2VA's real order. It diverges from FL2VA before the conditioner:
    /// media is decoded and normalized on the host first (that retained media
    /// is what the two reference encoders later consume), Qwen encodes the
    /// prompt *with* the reference vision pads, and conditioning is then
    /// encoded twice — once through the visual VAE and once through the audio
    /// VAE — before the VAEs park for denoise and reload for decode.
    DecodeReferencesPreprocessReferencesLoadQwenEncodeVisionTransferDropQwenLoadVaesEncodeVisualReferencesEncodeAudioReferencesParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux,
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
    pub qwen_host_load_staging_bytes: u64,
    pub qwen_retained_header_host_bytes: u64,
    pub transformer_retained_header_host_bytes: u64,
    pub vae_retained_config_host_bytes: u64,
    /// Digest of the ordered reference facts this budget was derived from.
    ///
    /// Every reference term below is a byte total, and byte totals collide:
    /// a 6000x4000 native canvas and a 4000x6000 one price identically, as do
    /// any two reference sets summing to the same retained bytes. Binding the
    /// geometry itself into the budget identity is what stops a budget from
    /// being replayed against a different reference set that happens to cost
    /// the same. Empty for FL2VA. Mirrors `vae_memory_evidence_identity_sha256`.
    pub reference_media_identity_sha256: String,
    /// Ref2VA retained normalized media: the RGB8 frames and f32 stereo
    /// waveforms the backend holds from preprocessing until both reference
    /// encoders have consumed them. Derived from the frozen reference plan,
    /// never a flat constant — one 2048-square still is 12 MiB and a 768-square
    /// video contributes its whole VAE prefix plus Qwen cursor frames. Zero for
    /// FL2VA, which retains normalized endpoints instead.
    pub reference_normalized_media_host_bytes: u64,
    /// Transient host working set while one reference is decoded: the staged
    /// copy of its verified bytes plus the decoder's own output before
    /// normalization. Charged once because references decode one at a time.
    pub reference_decode_staging_host_bytes: u64,
    /// Transient host working set while one decoded reference is normalized
    /// (resize and resample scratch), likewise charged once.
    pub reference_preprocess_staging_host_bytes: u64,
    pub reference_decode_phase_host_bytes: u64,
    pub reference_preprocess_phase_host_bytes: u64,
    pub reference_visual_encode_phase_host_bytes: u64,
    pub reference_audio_encode_phase_host_bytes: u64,
    pub vae_load_phase_host_bytes: u64,
    pub qwen_encode_phase_host_bytes: u64,
    pub qwen_transfer_phase_host_bytes: u64,
    pub condition_encode_phase_host_bytes: u64,
    pub noise_allocation_phase_host_bytes: u64,
    pub transformer_load_phase_host_bytes: u64,
    pub denoise_phase_host_bytes: u64,
    pub visual_decode_phase_host_bytes: u64,
    pub audio_decode_phase_host_bytes: u64,
    pub waveform_transfer_phase_host_bytes: u64,
    pub mux_phase_host_bytes: u64,
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
    /// Resident device bytes of an authenticated Turbo LoRA adapter, charged
    /// from the transformer load through the whole denoise. Zero without one.
    pub turbo_adapter_device_bytes: u64,
    /// Transient device bytes the adapter upload peaks at *above* its
    /// residents: each module's transposed matrices are built while its
    /// originals are still live. Charged in the transformer-load phase only,
    /// because the transposes are released before the denoise begins. Zero
    /// without an adapter.
    pub turbo_adapter_device_staging_bytes: u64,
    /// Transient host bytes the Turbo adapter load peaks at while its deltas
    /// are read and staged one matrix at a time. Zero without one.
    pub turbo_adapter_host_staging_bytes: u64,
    /// Device workspace the audio VAE's encoder peaks at while one reference
    /// soundtrack is encoded. Distinct from `audio_decode_workspace_device_bytes`:
    /// the encoder runs DAC plus the posterior projection, the decoder runs
    /// BigVGAN. Zero for FL2VA, which never encodes audio.
    pub reference_audio_encode_workspace_device_bytes: u64,
    pub reference_decode_phase_device_bytes: u64,
    pub reference_preprocess_phase_device_bytes: u64,
    pub reference_visual_encode_phase_device_bytes: u64,
    pub reference_audio_encode_phase_device_bytes: u64,
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
            qwen_host_load_staging_bytes,
            qwen_retained_header_host_bytes,
            transformer_retained_header_host_bytes,
            vae_retained_config_host_bytes,
            reference_media_identity_sha256,
            reference_normalized_media_host_bytes,
            reference_decode_staging_host_bytes,
            reference_preprocess_staging_host_bytes,
            reference_decode_phase_host_bytes,
            reference_preprocess_phase_host_bytes,
            reference_visual_encode_phase_host_bytes,
            reference_audio_encode_phase_host_bytes,
            vae_load_phase_host_bytes,
            qwen_encode_phase_host_bytes,
            qwen_transfer_phase_host_bytes,
            condition_encode_phase_host_bytes,
            noise_allocation_phase_host_bytes,
            transformer_load_phase_host_bytes,
            denoise_phase_host_bytes,
            visual_decode_phase_host_bytes,
            audio_decode_phase_host_bytes,
            waveform_transfer_phase_host_bytes,
            mux_phase_host_bytes,
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
            turbo_adapter_device_bytes,
            turbo_adapter_device_staging_bytes,
            turbo_adapter_host_staging_bytes,
            reference_audio_encode_workspace_device_bytes,
            reference_decode_phase_device_bytes,
            reference_preprocess_phase_device_bytes,
            reference_visual_encode_phase_device_bytes,
            reference_audio_encode_phase_device_bytes,
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
            H3FactoryTargetLoadDropPolicy::LoadQwenEncodeTransferDropQwenLoadVaesEncodeConditionsParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux => {
                b"load-qwen-encode-transfer-drop-qwen-load-vaes-encode-conditions-park-vaes-allocate-noise-load-transformer-denoise-drop-transformer-reload-vaes-decode-visual-audio-drop-vaes-mux".as_slice()
            }
            H3FactoryTargetLoadDropPolicy::DecodeReferencesPreprocessReferencesLoadQwenEncodeVisionTransferDropQwenLoadVaesEncodeVisualReferencesEncodeAudioReferencesParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux => {
                b"decode-references-preprocess-references-load-qwen-encode-vision-transfer-drop-qwen-load-vaes-encode-visual-references-encode-audio-references-park-vaes-allocate-noise-load-transformer-denoise-drop-transformer-reload-vaes-decode-visual-audio-drop-vaes-mux".as_slice()
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
            qwen_host_load_staging_bytes,
            qwen_retained_header_host_bytes,
            transformer_retained_header_host_bytes,
            vae_retained_config_host_bytes,
            reference_normalized_media_host_bytes,
            reference_decode_staging_host_bytes,
            reference_preprocess_staging_host_bytes,
            reference_decode_phase_host_bytes,
            reference_preprocess_phase_host_bytes,
            reference_visual_encode_phase_host_bytes,
            reference_audio_encode_phase_host_bytes,
            vae_load_phase_host_bytes,
            qwen_encode_phase_host_bytes,
            qwen_transfer_phase_host_bytes,
            condition_encode_phase_host_bytes,
            noise_allocation_phase_host_bytes,
            transformer_load_phase_host_bytes,
            denoise_phase_host_bytes,
            visual_decode_phase_host_bytes,
            audio_decode_phase_host_bytes,
            waveform_transfer_phase_host_bytes,
            mux_phase_host_bytes,
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
        update_string(hash, reference_media_identity_sha256);
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
            turbo_adapter_device_bytes,
            turbo_adapter_device_staging_bytes,
            turbo_adapter_host_staging_bytes,
            reference_audio_encode_workspace_device_bytes,
            reference_decode_phase_device_bytes,
            reference_preprocess_phase_device_bytes,
            reference_visual_encode_phase_device_bytes,
            reference_audio_encode_phase_device_bytes,
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

/// Public mirror of the crate-private sampler kind, so a frozen authority can
/// name its integrator without leaking the runtime type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactorySamplerKind {
    OfficialEuler,
    ComfyResMultistep,
    ComfyEuler,
}

impl H3FactorySamplerKind {
    pub const fn as_str(self) -> &'static str {
        self.runtime_kind().as_str()
    }

    pub(crate) const fn runtime_kind(self) -> H3SamplerKind {
        match self {
            Self::OfficialEuler => H3SamplerKind::OfficialEuler,
            Self::ComfyResMultistep => H3SamplerKind::ComfyResMultistep,
            Self::ComfyEuler => H3SamplerKind::ComfyEuler,
        }
    }

    pub(crate) const fn from_runtime(kind: H3SamplerKind) -> Self {
        match kind {
            H3SamplerKind::OfficialEuler => Self::OfficialEuler,
            H3SamplerKind::ComfyResMultistep => Self::ComfyResMultistep,
            H3SamplerKind::ComfyEuler => Self::ComfyEuler,
        }
    }
}

/// Whether one media-facts model identity is exactly the identity a frozen H3
/// factory authority renders.
///
/// The frozen authority's `canonical_model` is the engine PARTITION (the base
/// compact task), while the request — and therefore media facts and saved
/// provenance — keeps the full reviewed identity, Turbo tag included. The
/// pairing is strict in both directions: a Turbo tag requires the authority to
/// have frozen exactly that tier's adapter, and a base identity requires no
/// adapter — except under the capture-scope `h3-private-uat` feature, whose
/// env override legitimately overlays an adapter on the base model.
pub fn media_model_matches_h3_authority(model: &str, authority: &FrozenH3FactoryAuthority) -> bool {
    if mold_core::minimax_h3::base_compact_model(model) != Some(authority.canonical_model()) {
        return false;
    }
    let adapter_tier = authority
        .quantization()
        .turbo_adapter()
        .map(H3FactoryTurboAdapterAuthority::tier_stable_id);
    match mold_core::minimax_h3::turbo_tier_for_model(model) {
        Some(tier) => adapter_tier == Some(tier.tier_stable_id),
        None => adapter_tier.is_none() || cfg!(feature = "h3-private-uat"),
    }
}

/// Identity of one authenticated Turbo LoRA adapter overlaid on the compact
/// INT8 checkpoint, together with the sampler contract its distillation
/// implies.
///
/// The base checkpoint file is byte-identical with or without Turbo, so this is
/// an *additive* artifact authority rather than a new checkpoint format. A
/// frozen plan that carries one has recorded exactly which adapter weights ran,
/// which integrator consumed them, and how many transformer evaluations the
/// distillation was reviewed for.
///
/// Every field is private and [`Self::for_reviewed_tier`] is the only
/// constructor. The distillation triple — sampler kind, step count, video
/// shift — is **read from the reviewed tier table**, never supplied, so a
/// genuine adapter cannot be paired with an arbitrary step count, an arbitrary
/// shift, or the RES-multistep integrator it was not distilled for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryTurboAdapterAuthority {
    tier_stable_id: String,
    adapter_identity_sha256: String,
    adapter_content_sha256: String,
    sampler_kind: H3FactorySamplerKind,
    grid_points: u32,
    video_shift_bits: u32,
    resident_device_bytes: u64,
    device_staging_peak_bytes: u64,
    host_staging_peak_bytes: u64,
}

impl H3FactoryTurboAdapterAuthority {
    /// Build the authority for a reviewed tier.
    ///
    /// The caller supplies only what the *file* proves — which tier it is, its
    /// two digests, and the three byte costs measured from its validated
    /// structure. Everything that decides how it is sampled comes from
    /// [`crate::minimax_h3::turbo::REVIEWED_TURBO_TIERS`].
    pub fn for_reviewed_tier(
        tier_stable_id: &str,
        adapter_identity_sha256: &str,
        adapter_content_sha256: &str,
        resident_device_bytes: u64,
        device_staging_peak_bytes: u64,
        host_staging_peak_bytes: u64,
    ) -> Result<Self> {
        let reviewed = reviewed_turbo_contract(tier_stable_id)?;
        let authority = Self {
            tier_stable_id: tier_stable_id.trim().to_owned(),
            adapter_identity_sha256: adapter_identity_sha256.to_owned(),
            adapter_content_sha256: adapter_content_sha256.to_owned(),
            sampler_kind: H3FactorySamplerKind::from_runtime(reviewed.sampler_kind),
            grid_points: reviewed.grid_points,
            video_shift_bits: reviewed.video_shift.to_bits(),
            resident_device_bytes,
            device_staging_peak_bytes,
            host_staging_peak_bytes,
        };
        authority.validate()?;
        Ok(authority)
    }

    pub fn tier_stable_id(&self) -> &str {
        &self.tier_stable_id
    }

    pub fn adapter_identity_sha256(&self) -> &str {
        &self.adapter_identity_sha256
    }

    pub fn adapter_content_sha256(&self) -> &str {
        &self.adapter_content_sha256
    }

    pub const fn sampler_kind(&self) -> H3FactorySamplerKind {
        self.sampler_kind
    }

    pub const fn grid_points(&self) -> u32 {
        self.grid_points
    }

    /// The task this adapter's tier was reviewed for, resolved from its stable
    /// id. An unrecognised id yields `None` and must be treated as a mismatch
    /// rather than as "any task".
    pub fn reviewed_task(&self) -> Option<Task> {
        mold_candle::minimax_h3::H3TurboLoraTier::ALL
            .into_iter()
            .find(|tier| tier.stable_id() == self.tier_stable_id)
            .map(|tier| match tier.task() {
                mold_candle::minimax_h3::H3TransformerTask::Ref2Va => Task::Ref2va,
                _ => Task::Fl2va,
            })
    }

    pub const fn resident_device_bytes(&self) -> u64 {
        self.resident_device_bytes
    }

    /// Transient device bytes the delta upload peaks at above the residents,
    /// because each module's transposed copies are built while its originals
    /// are still live.
    pub const fn device_staging_peak_bytes(&self) -> u64 {
        self.device_staging_peak_bytes
    }

    pub const fn host_staging_peak_bytes(&self) -> u64 {
        self.host_staging_peak_bytes
    }

    /// Re-prove every field against the reviewed tier table.
    ///
    /// The constructor already derives the distillation triple, so this only
    /// fails for a value that was mutated after construction — which is exactly
    /// what it is here to stop.
    fn validate(&self) -> Result<()> {
        let reviewed = reviewed_turbo_contract(&self.tier_stable_id)?;
        require_sha256(&self.adapter_identity_sha256, "H3 Turbo adapter identity")?;
        require_sha256(&self.adapter_content_sha256, "H3 Turbo adapter content")?;
        if self.sampler_kind != H3FactorySamplerKind::from_runtime(reviewed.sampler_kind) {
            bail!(
                "MiniMax H3 Turbo tier {:?} is distilled for {}, not {}",
                self.tier_stable_id,
                reviewed.sampler_kind.as_str(),
                self.sampler_kind.as_str()
            );
        }
        if self.grid_points != reviewed.grid_points {
            bail!(
                "MiniMax H3 Turbo tier {:?} is reviewed for {} grid points, not {}",
                self.tier_stable_id,
                reviewed.grid_points,
                self.grid_points
            );
        }
        if self.video_shift_bits != reviewed.video_shift.to_bits() {
            bail!(
                "MiniMax H3 Turbo tier {:?} is reviewed at video shift {}, not {}",
                self.tier_stable_id,
                reviewed.video_shift,
                f32::from_bits(self.video_shift_bits)
            );
        }
        if !self.sampler_kind.runtime_kind().uses_comfy_simple_grid() {
            bail!(
                "MiniMax H3 Turbo distillations require a Comfy sigma grid, got {}",
                self.sampler_kind.as_str()
            );
        }
        if !(2..=H3_FACTORY_MAX_GRID_POINTS).contains(&self.grid_points) {
            bail!(
                "MiniMax H3 Turbo adapter grid points must be 2..={H3_FACTORY_MAX_GRID_POINTS}, got {}",
                self.grid_points
            );
        }
        if self.resident_device_bytes == 0
            || self.device_staging_peak_bytes == 0
            || self.host_staging_peak_bytes == 0
        {
            bail!(
                "MiniMax H3 Turbo adapter must charge nonzero resident, device staging, and host staging bytes"
            );
        }
        if self.device_staging_peak_bytes > self.resident_device_bytes {
            bail!(
                "MiniMax H3 Turbo device staging {} exceeds the whole resident adapter {}",
                self.device_staging_peak_bytes,
                self.resident_device_bytes
            );
        }
        // The schedule has to be buildable before any weight is loaded, and its
        // evaluation count must not collapse below the reviewed step count.
        let counts = H3DualSchedule::new_for_sampler_with_video_shift(
            usize::try_from(self.grid_points)
                .map_err(|_| anyhow!("MiniMax H3 Turbo grid points exceed usize"))?,
            self.sampler_kind.runtime_kind(),
            self.video_shift(),
        )?
        .counts();
        if counts.effective_grid_points != counts.requested_grid_points {
            bail!(
                "MiniMax H3 Turbo schedule collapsed {} requested grid points to {}",
                counts.requested_grid_points,
                counts.effective_grid_points
            );
        }
        Ok(())
    }

    /// Consumed by the gated private FL2VA runtime; without the `h3` /
    /// `h3-private-uat` features only tests reach it.
    #[cfg_attr(not(any(feature = "h3", feature = "h3-private-uat")), allow(dead_code))]
    pub(crate) const fn resolved_sampler_kind(&self) -> H3SamplerKind {
        self.sampler_kind.runtime_kind()
    }

    /// Consumed by the gated private FL2VA runtime; without the `h3` /
    /// `h3-private-uat` features only tests reach it.
    #[cfg_attr(not(any(feature = "h3", feature = "h3-private-uat")), allow(dead_code))]
    pub(crate) fn video_shift(&self) -> f32 {
        f32::from_bits(self.video_shift_bits)
    }

    fn update_identity(&self, hash: &mut Sha256) {
        let Self {
            tier_stable_id,
            adapter_identity_sha256,
            adapter_content_sha256,
            sampler_kind,
            grid_points,
            video_shift_bits,
            resident_device_bytes,
            device_staging_peak_bytes,
            host_staging_peak_bytes,
        } = self;
        hash.update(b"turbo-adapter\0");
        for field in [
            tier_stable_id.as_str(),
            adapter_identity_sha256.as_str(),
            adapter_content_sha256.as_str(),
            sampler_kind.as_str(),
        ] {
            hash.update(field.as_bytes());
            hash.update([0]);
        }
        hash.update(grid_points.to_le_bytes());
        hash.update(video_shift_bits.to_le_bytes());
        hash.update(resident_device_bytes.to_le_bytes());
        hash.update(device_staging_peak_bytes.to_le_bytes());
        hash.update(host_staging_peak_bytes.to_le_bytes());
    }
}

/// The single reviewed-tier table both the authority and the runtime consult.
fn reviewed_turbo_contract(
    tier_stable_id: &str,
) -> Result<&'static crate::minimax_h3::turbo::H3TurboTierContract> {
    crate::minimax_h3::turbo::reviewed_contract_for_stable_id(tier_stable_id)
        .ok_or_else(|| anyhow!("MiniMax H3 Turbo tier {tier_stable_id:?} is not a reviewed tier"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H3FactoryQuantizationAuthority {
    OfficialBf16,
    ComfyPrunedInt8ConvrotNvfp4Awq {
        transformer_policy_sha256: String,
        qwen_policy_sha256: String,
        pruned_adaln_table_sha256: String,
        /// `None` for the reviewed 21-step baseline; `Some` only for an
        /// authenticated Turbo tier.
        turbo_adapter: Option<H3FactoryTurboAdapterAuthority>,
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
                    turbo_adapter,
                },
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
            ) => {
                require_sha256(transformer_policy_sha256, "H3 transformer quantization")?;
                require_sha256(qwen_policy_sha256, "H3 Qwen quantization")?;
                require_sha256(pruned_adaln_table_sha256, "H3 pruned AdaLN table")?;
                if let Some(turbo_adapter) = turbo_adapter {
                    turbo_adapter.validate()?;
                }
                Ok(())
            }
            _ => bail!("MiniMax H3 layout and quantization authorities disagree"),
        }
    }

    /// Append every semantically authoritative field to `hash`.
    ///
    /// The exhaustive destructure is intentional and mirrors
    /// [`H3FactoryTargetBudgetInput::update_identity`]: adding a variant field
    /// must fail compilation here until its identity treatment is explicit.
    /// Before Turbo this arm used `..`, so a new field would have been silently
    /// absent from the frozen identity.
    fn update_identity(&self, hash: &mut Sha256) {
        match self {
            Self::OfficialBf16 => hash.update(b"official-bf16"),
            Self::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256,
                qwen_policy_sha256,
                pruned_adaln_table_sha256,
                turbo_adapter,
            } => {
                hash.update(b"comfy-pruned-int8-convrot-nvfp4-awq\0");
                hash.update(transformer_policy_sha256.as_bytes());
                hash.update([0]);
                hash.update(qwen_policy_sha256.as_bytes());
                hash.update([0]);
                hash.update(pruned_adaln_table_sha256.as_bytes());
                match turbo_adapter {
                    None => hash.update(b"\0no-turbo-adapter"),
                    Some(turbo_adapter) => {
                        hash.update([0]);
                        turbo_adapter.update_identity(hash);
                    }
                }
            }
        }
    }

    /// The authenticated Turbo adapter overlaying this checkpoint, if any.
    pub fn turbo_adapter(&self) -> Option<&H3FactoryTurboAdapterAuthority> {
        match self {
            Self::OfficialBf16 => None,
            Self::ComfyPrunedInt8ConvrotNvfp4Awq { turbo_adapter, .. } => turbo_adapter.as_ref(),
        }
    }

    /// The integrator a frozen plan built from this authority must run.
    ///
    /// Without a Turbo adapter the compact layout keeps its reviewed
    /// RES-multistep rule; the official BF16 layout keeps first-order Euler.
    /// Only the private FL2VA runtime consumes this; it stays compiled in every
    /// build so the contract cannot drift, mirroring
    /// `private_fl2va_runtime_authority`'s treatment above.
    #[cfg_attr(not(any(feature = "h3", feature = "h3-private-uat")), allow(dead_code))]
    pub(crate) fn sampler_kind(&self) -> H3SamplerKind {
        match self {
            Self::OfficialBf16 => H3SamplerKind::OfficialEuler,
            Self::ComfyPrunedInt8ConvrotNvfp4Awq { turbo_adapter, .. } => match turbo_adapter {
                None => H3SamplerKind::ComfyResMultistep,
                Some(turbo_adapter) => turbo_adapter.resolved_sampler_kind(),
            },
        }
    }

    /// The video shift the sigma grid must be built with.
    #[cfg_attr(not(any(feature = "h3", feature = "h3-private-uat")), allow(dead_code))]
    pub(crate) fn video_shift(&self) -> f32 {
        self.turbo_adapter()
            .map_or(H3_VIDEO_SHIFT, |turbo| turbo.video_shift())
    }

    /// Resident device bytes this authority's Turbo adapter charges.
    pub fn turbo_adapter_device_bytes(&self) -> u64 {
        self.turbo_adapter()
            .map_or(0, H3FactoryTurboAdapterAuthority::resident_device_bytes)
    }

    /// Transient device bytes this authority's Turbo adapter upload charges.
    pub fn turbo_adapter_device_staging_bytes(&self) -> u64 {
        self.turbo_adapter()
            .map_or(0, H3FactoryTurboAdapterAuthority::device_staging_peak_bytes)
    }

    /// Transient host staging bytes this authority's Turbo adapter charges.
    pub fn turbo_adapter_host_staging_bytes(&self) -> u64 {
        self.turbo_adapter()
            .map_or(0, H3FactoryTurboAdapterAuthority::host_staging_peak_bytes)
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
    /// The CUDA compute capability of the frozen route, or `None` for Metal.
    ///
    /// Metal has no per-architecture attention qualification — Apple Silicon
    /// has one chunked dense correctness route — so `None` is the whole
    /// statement, not a missing value. Callers that gate on SM89 must treat it
    /// as "not a CUDA route" rather than substituting a default.
    pub compute_capability: Option<(u16, u16)>,
    pub execution_fingerprint: String,
    pub conditioner_placement: H3FactoryConditionerPlacement,
    pub qwen_parameter_bytes: u64,
    pub qwen_host_resident_parameter_bytes: u64,
    pub qwen_device_resident_parameter_bytes: u64,
    pub qwen_activation_workspace_bytes: u64,
    /// Largest single tensor the NVFP4 loader reads, and the raw header it
    /// retains. Both are opened-loader facts, pinned to the released runtime
    /// memory facts at the same seam that pins parameter residency.
    pub qwen_maximum_tensor_staging_bytes: u64,
    pub qwen_retained_raw_header_bytes: u64,
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
    qwen_maximum_tensor_staging_bytes: u64,
    qwen_retained_raw_header_bytes: u64,
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
    pub(crate) compute_capability: Option<(u16, u16)>,
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
    qwen_maximum_tensor_staging_bytes: u64,
    qwen_retained_raw_header_bytes: u64,
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
            // The loader holds the largest tensor twice while reading it: the
            // `Vec` and its `from_raw_buffer` copy. Bound to the opened loader
            // fact, never a caller-chosen number.
            || self.target_budget.qwen_host_load_staging_bytes
                != projection
                    .qwen_maximum_tensor_staging_bytes
                    .checked_mul(2)
                    .ok_or_else(|| anyhow!("H3 Qwen host load staging overflow"))?
            || self.target_budget.qwen_retained_header_host_bytes
                != projection.qwen_retained_raw_header_bytes
            || match projection.conditioner_placement {
                H3FactoryConditionerPlacement::AssignedCudaThenDrop
                | H3FactoryConditionerPlacement::AssignedMetalThenDrop => {
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

/// The charges one ordered reference contributes, derived from its frozen
/// geometry.
///
/// Exposed so the admission builder can populate a descriptor from the SAME
/// arithmetic `validate_prepared_request` re-derives, rather than restating it
/// and drifting. Every field here is computed; none is read back from the
/// input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(all(feature = "mp4", any(feature = "h3", feature = "h3-private-uat"))),
    allow(dead_code)
)]
pub struct H3FactoryReferenceCharges {
    pub visual_rows: u64,
    pub audio_rows: u64,
    pub qwen_vision_rows: u64,
    pub normalized_host_bytes: u64,
    pub native_host_bytes: u64,
}

/// Derive one reference's charges. The geometry fields of `reference` are the
/// only inputs read; its own charge fields are ignored.
#[cfg_attr(
    not(all(feature = "mp4", any(feature = "h3", feature = "h3-private-uat"))),
    allow(dead_code)
)]
pub fn expected_h3_factory_reference_charges(
    reference: &H3FactoryReferenceInput,
) -> Result<H3FactoryReferenceCharges> {
    let charges = h3_reference_charges(reference)?;
    Ok(H3FactoryReferenceCharges {
        visual_rows: charges.visual_rows,
        audio_rows: charges.audio_rows,
        qwen_vision_rows: charges.qwen_vision_rows,
        normalized_host_bytes: charges.normalized_host_bytes,
        native_host_bytes: charges.native_host_bytes,
    })
}

/// Recomputed charges for one ordered Ref2VA reference.
struct H3ReferenceCharges {
    visual_rows: u64,
    audio_rows: u64,
    qwen_vision_rows: u64,
    normalized_host_bytes: u64,
    native_host_bytes: u64,
}

/// Aggregate retained-media totals for one ordered reference set.
///
/// The two are charged in different phases and must not be collapsed: natives
/// are held from decode until preprocess, normalized ones from preprocess
/// until both encoders finish, and during preprocess the two overlap.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct H3ReferenceMediaTotals {
    pub(crate) normalized_host_bytes: u64,
    pub(crate) native_host_bytes: u64,
    /// The largest single native payload — the transient a decode peaks at
    /// while it materializes one reference on top of those already held.
    pub(crate) largest_native_host_bytes: u64,
    /// The largest single normalized payload, the transient a preprocess peaks
    /// at while it writes one reference's normalized form.
    pub(crate) largest_normalized_host_bytes: u64,
}

/// Re-derive every row and retained-byte charge for one reference from its
/// frozen normalized geometry.
///
/// This deliberately repeats `mold_core::minimax_h3`'s own arithmetic instead
/// of trusting the supplied totals: admission computes the shape, but the
/// factory is the authority that a scheduler grant is sized from, and a
/// reference whose rows were understated would be admitted into a budget that
/// cannot hold it.
fn h3_reference_charges(reference: &H3FactoryReferenceInput) -> Result<H3ReferenceCharges> {
    let visual_canvas = match (reference.normalized_width, reference.normalized_height) {
        (Some(width), Some(height)) => {
            if width == 0
                || height == 0
                || !width.is_multiple_of(32)
                || !height.is_multiple_of(32)
                // The visual VAE downsamples 16x and the DiT packs 2x2 latent
                // patches, so a canvas that is not 32-divisible cannot be
                // patchified at all.
                || reference.kind == H3FactoryReferenceKind::Audio
            {
                bail!(
                    "MiniMax H3 reference {} has an invalid normalized visual canvas",
                    reference.index
                );
            }
            Some((u64::from(width), u64::from(height)))
        }
        (None, None) => {
            if reference.kind != H3FactoryReferenceKind::Audio {
                bail!(
                    "MiniMax H3 visual reference {} lost its normalized canvas",
                    reference.index
                );
            }
            None
        }
        _ => bail!(
            "MiniMax H3 reference {} has a half-specified normalized canvas",
            reference.index
        ),
    };
    let rows_per_latent_frame = visual_canvas
        .map(|(width, height)| {
            (width / 32)
                .checked_mul(height / 32)
                .ok_or_else(|| anyhow!("MiniMax H3 reference rows per latent frame overflow"))
        })
        .transpose()?
        .unwrap_or(0);

    // Frame geometry. Images occupy exactly one latent frame; videos take the
    // `5n+2` causal latent count over their visual-VAE prefix; audio has none.
    let (latent_frames, qwen_blocks) = match reference.kind {
        H3FactoryReferenceKind::Image => {
            if reference.normalized_video_frames.is_some()
                || reference.video_frames.is_some()
                || reference.qwen_video_frames.is_some()
            {
                bail!(
                    "MiniMax H3 image reference {} carries video frame geometry",
                    reference.index
                );
            }
            (1_u64, 1_u64)
        }
        H3FactoryReferenceKind::Video => {
            let normalized_frames = reference.normalized_video_frames.ok_or_else(|| {
                anyhow!(
                    "MiniMax H3 video reference {} lost its normalized frame count",
                    reference.index
                )
            })?;
            let video_frames = reference.video_frames.ok_or_else(|| {
                anyhow!(
                    "MiniMax H3 video reference {} lost its visual-VAE frame prefix",
                    reference.index
                )
            })?;
            let qwen_video_frames = reference.qwen_video_frames.ok_or_else(|| {
                anyhow!(
                    "MiniMax H3 video reference {} lost its Qwen cursor sample count",
                    reference.index
                )
            })?;
            if normalized_frames == 0 || video_frames == 0 || video_frames > normalized_frames {
                bail!(
                    "MiniMax H3 video reference {} has an invalid frame prefix",
                    reference.index
                );
            }
            if qwen_video_frames != normalized_frames.div_ceil(contract::FIXED_FPS / 2) {
                bail!(
                    "MiniMax H3 video reference {} Qwen 2 fps sampling differs from the contract",
                    reference.index
                );
            }
            let latent_frames = if video_frames <= 5 {
                2_u64
            } else {
                u64::from((video_frames - 5) / contract::FRAME_STEP) * 5 + 2
            };
            // Qwen consumes temporal-patch-2 blocks over its 2 fps cursor.
            (latent_frames, u64::from(qwen_video_frames).div_ceil(2))
        }
        H3FactoryReferenceKind::Audio => {
            if reference.normalized_video_frames.is_some()
                || reference.video_frames.is_some()
                || reference.qwen_video_frames.is_some()
            {
                bail!(
                    "MiniMax H3 audio reference {} carries video frame geometry",
                    reference.index
                );
            }
            (0, 0)
        }
    };

    let visual_rows = latent_frames
        .checked_mul(rows_per_latent_frame)
        .ok_or_else(|| anyhow!("MiniMax H3 reference visual rows overflow"))?;
    let qwen_vision_rows = qwen_blocks
        .checked_mul(rows_per_latent_frame)
        .ok_or_else(|| anyhow!("MiniMax H3 reference Qwen vision rows overflow"))?;

    // Audio. Present exactly for standalone audio and for a video carrying a
    // soundtrack; the VAE compresses 800 normalized samples per stereo latent.
    let expects_audio = match reference.kind {
        H3FactoryReferenceKind::Image => false,
        H3FactoryReferenceKind::Video => reference.audio_samples_per_channel.is_some(),
        H3FactoryReferenceKind::Audio => true,
    };
    let audio_samples = match (reference.audio_samples_per_channel, expects_audio) {
        (Some(samples), true) if samples > 0 => samples,
        (None, false) => 0,
        _ => bail!(
            "MiniMax H3 reference {} audio geometry differs from its modality",
            reference.index
        ),
    };
    let audio_rows = audio_samples
        .div_ceil(800)
        .checked_mul(u64::from(contract::AUDIO_CHANNELS))
        .ok_or_else(|| anyhow!("MiniMax H3 reference audio rows overflow"))?;

    // Retained host bytes. The backend holds the normalized RGB8 frames the
    // visual VAE will encode plus the normalized f32 stereo waveform the audio
    // VAE will encode; for a video the Qwen cursor frames are borrowed from
    // that same prefix wherever they overlap and materialized beyond it.
    let frame_bytes = visual_canvas
        .map(|(width, height)| {
            width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| anyhow!("MiniMax H3 reference frame bytes overflow"))
        })
        .transpose()?
        .unwrap_or(0);
    let retained_frames = match reference.kind {
        H3FactoryReferenceKind::Image => 1,
        H3FactoryReferenceKind::Video => {
            let video_frames = u64::from(reference.video_frames.unwrap_or_default());
            let qwen_frames = u64::from(reference.qwen_video_frames.unwrap_or_default());
            // Cursor samples inside the VAE prefix are clones of frames that
            // are already retained, so both sets are charged.
            video_frames
                .checked_add(qwen_frames)
                .ok_or_else(|| anyhow!("MiniMax H3 reference retained frames overflow"))?
        }
        H3FactoryReferenceKind::Audio => 0,
    };
    let visual_host_bytes = retained_frames
        .checked_mul(frame_bytes)
        .ok_or_else(|| anyhow!("MiniMax H3 reference retained visual bytes overflow"))?;
    let audio_host_bytes = audio_samples
        .checked_mul(u64::from(contract::AUDIO_CHANNELS))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| anyhow!("MiniMax H3 reference retained audio bytes overflow"))?;
    let normalized_host_bytes = visual_host_bytes
        .checked_add(audio_host_bytes)
        .ok_or_else(|| anyhow!("MiniMax H3 reference retained host bytes overflow"))?;

    // Native retention. The decoder keeps what it decoded — the union of the
    // visual-VAE prefix and the Qwen cursor, at source resolution — plus the
    // source soundtrack, until this reference's own preprocess step runs.
    let native_canvas = match (reference.native_width, reference.native_height) {
        (Some(width), Some(height)) => {
            // Bound it here, at the request boundary, so the ledger term is
            // always derivable. An unbounded native payload would leave the
            // scheduler charging an estimate the runtime can exceed.
            if width == 0
                || height == 0
                || width > contract::MAX_REFERENCE_DIMENSION
                || height > contract::MAX_REFERENCE_DIMENSION
                || u64::from(width) * u64::from(height) > contract::MAX_REFERENCE_IMAGE_PIXELS
                || reference.kind == H3FactoryReferenceKind::Audio
            {
                bail!(
                    "MiniMax H3 reference {} native canvas is missing, oversized, or crossed",
                    reference.index
                );
            }
            Some((u64::from(width), u64::from(height)))
        }
        (None, None) => {
            if reference.kind != H3FactoryReferenceKind::Audio {
                bail!(
                    "MiniMax H3 visual reference {} has no native canvas to price its decode",
                    reference.index
                );
            }
            None
        }
        _ => bail!(
            "MiniMax H3 reference {} has a half-specified native canvas",
            reference.index
        ),
    };
    let native_frame_bytes = native_canvas
        .map(|(width, height)| {
            width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| anyhow!("MiniMax H3 native frame bytes overflow"))
        })
        .transpose()?
        .unwrap_or(0);
    // Charge what the decoder actually keeps, which is not the same shape for
    // the two visual modalities:
    //
    // * A still is retained exactly as decoded, at source resolution.
    // * A video's selected frames are resized to the normalized canvas DURING
    //   decode (`reference_media.rs` hands `decode_video_from_binding` a
    //   resize closure) and only the resized frames are retained. Pricing
    //   those at source resolution is wrong in both directions, and wrong in
    //   the dangerous direction whenever the normalized canvas is larger — a
    //   32x32 clip normalized to 384x384 retains 144x what source dims imply.
    let retained_frame_bytes = match reference.kind {
        H3FactoryReferenceKind::Image => native_frame_bytes,
        H3FactoryReferenceKind::Video => frame_bytes,
        H3FactoryReferenceKind::Audio => 0,
    };
    let native_visual_bytes = retained_frames
        .checked_mul(retained_frame_bytes)
        .ok_or_else(|| anyhow!("MiniMax H3 native retained visual bytes overflow"))?;
    let native_audio_bytes = match (
        reference.native_audio_samples_per_channel,
        reference.native_audio_channels,
        expects_audio,
    ) {
        (Some(samples), Some(channels), true) => {
            if samples == 0
                || channels == 0
                || channels > contract::MAX_REFERENCE_CHANNELS
                || samples
                    > u64::from(contract::MAX_REFERENCE_SAMPLE_RATE)
                        .saturating_mul(contract::MAX_REFERENCE_DURATION_MS)
                        .div_ceil(1_000)
            {
                bail!(
                    "MiniMax H3 reference {} native soundtrack is missing or oversized",
                    reference.index
                );
            }
            samples
                .checked_mul(u64::from(channels))
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>() as u64))
                .ok_or_else(|| anyhow!("MiniMax H3 native retained audio bytes overflow"))?
        }
        (None, None, false) => 0,
        _ => bail!(
            "MiniMax H3 reference {} native audio geometry differs from its modality",
            reference.index
        ),
    };
    let native_host_bytes = native_visual_bytes
        .checked_add(native_audio_bytes)
        .ok_or_else(|| anyhow!("MiniMax H3 native retained host bytes overflow"))?;

    Ok(H3ReferenceCharges {
        visual_rows,
        audio_rows,
        qwen_vision_rows,
        normalized_host_bytes,
        native_host_bytes,
    })
}

/// Validate the complete ordered reference list and return its aggregate
/// charges as `(visual_rows, audio_rows, qwen_vision_rows, host_bytes)`.
fn validate_prepared_references(
    references: &[H3FactoryReferenceInput],
) -> Result<(u64, u64, u64, H3ReferenceMediaTotals)> {
    if references.is_empty() {
        bail!("MiniMax H3 Ref2VA prepared request retains no ordered references");
    }
    if references.len() > contract::MAX_REFERENCE_FILES {
        bail!("MiniMax H3 Ref2VA prepared request exceeds the released reference count");
    }
    let mut totals = (0_u64, 0_u64, 0_u64, H3ReferenceMediaTotals::default());
    for (offset, reference) in references.iter().enumerate() {
        require_sha256(&reference.content_sha256, "H3 reference content")?;
        if reference.index != u32::try_from(offset + 1)? {
            bail!("MiniMax H3 Ref2VA reference indices are not contiguous request order");
        }
        if reference.preprocess_version != contract::REFERENCE_PREPROCESS_VERSION {
            bail!(
                "MiniMax H3 reference {} was prepared at a different preprocessing contract",
                reference.index
            );
        }
        let charges = h3_reference_charges(reference)?;
        if charges.visual_rows != reference.visual_rows
            || charges.audio_rows != reference.audio_rows
            || charges.qwen_vision_rows != reference.qwen_vision_rows
            || charges.normalized_host_bytes != reference.normalized_host_bytes
            || charges.native_host_bytes != reference.native_host_bytes
        {
            bail!(
                "MiniMax H3 reference {} row or retained-byte charges differ from its frozen geometry",
                reference.index
            );
        }
        totals.0 = totals
            .0
            .checked_add(charges.visual_rows)
            .ok_or_else(|| anyhow!("MiniMax H3 reference visual row total overflow"))?;
        totals.1 = totals
            .1
            .checked_add(charges.audio_rows)
            .ok_or_else(|| anyhow!("MiniMax H3 reference audio row total overflow"))?;
        totals.2 = totals
            .2
            .checked_add(charges.qwen_vision_rows)
            .ok_or_else(|| anyhow!("MiniMax H3 reference vision row total overflow"))?;
        totals.3.normalized_host_bytes = totals
            .3
            .normalized_host_bytes
            .checked_add(charges.normalized_host_bytes)
            .ok_or_else(|| anyhow!("MiniMax H3 reference retained host byte total overflow"))?;
        totals.3.native_host_bytes = totals
            .3
            .native_host_bytes
            .checked_add(charges.native_host_bytes)
            .ok_or_else(|| anyhow!("MiniMax H3 native retained host byte total overflow"))?;
        totals.3.largest_native_host_bytes = totals
            .3
            .largest_native_host_bytes
            .max(charges.native_host_bytes);
        totals.3.largest_normalized_host_bytes = totals
            .3
            .largest_normalized_host_bytes
            .max(charges.normalized_host_bytes);
    }
    if totals.0 == 0 {
        bail!("MiniMax H3 Ref2VA prepared request has no visual reference conditioning");
    }
    Ok(totals)
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
    // `mold_core::minimax_h3` is the one packed-row authority the server's
    // admission and the private runtime envelope also read.
    let expected_video_latent_frames = contract::video_latent_frames(request.frames)
        .ok_or_else(|| anyhow!("MiniMax H3 video latent frames underflow"))?;
    let rows_per_video_latent = contract::rows_per_video_latent(request.width, request.height)
        .ok_or_else(|| anyhow!("MiniMax H3 rows per video latent overflow"))?;
    let expected_target_video_rows =
        contract::target_video_rows(request.width, request.height, request.frames)
            .ok_or_else(|| anyhow!("MiniMax H3 target video rows overflow"))?;
    let expected_audio_latents = contract::audio_latents_per_channel(request.frames)
        .ok_or_else(|| anyhow!("MiniMax H3 audio latent rows overflow"))?;
    let expected_audio_rows = contract::target_audio_rows(request.frames)
        .ok_or_else(|| anyhow!("MiniMax H3 target audio rows overflow"))?;
    // Deliberately the VOCODER count, which is `latents * 800` and NOT the
    // duration-derived `audio_samples_per_channel` admission charges the AAC
    // staging against; the two differ by up to 800 samples.
    let expected_audio_samples = contract::vocoder_audio_samples_per_channel(request.frames)
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
    // Conditioning is task-shaped: FL2VA prices boundary endpoints against the
    // generated canvas, while Ref2VA prices every ordered reference against
    // its own normalized canvas. The two never mix.
    let reference_charges = match request.task {
        Task::Fl2va => {
            if !request.references.is_empty() {
                bail!("MiniMax H3 FL2VA prepared request carries Ref2VA references");
            }
            None
        }
        Task::Ref2va => {
            if !request.endpoints.is_empty() {
                bail!("MiniMax H3 Ref2VA prepared request carries FL2VA boundary endpoints");
            }
            Some(validate_prepared_references(&request.references)?)
        }
    };
    let pixel_count = u64::from(request.width)
        .checked_mul(u64::from(request.height))
        .ok_or_else(|| anyhow!("MiniMax H3 pixel count overflow"))?;
    let aspect_ratio = request.width as f64 / request.height as f64;
    if request.canonical_model != model_contract.canonical_model
        || request.task != model_contract.task
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
        || request.rows.target_video_rows != expected_target_video_rows
        || request.rows.target_audio_rows != expected_audio_rows
        || request.rows.total_packed_rows != packed_rows
        || match reference_charges {
            // FL2VA conditions only on boundary frames and never on audio.
            None => {
                request.rows.condition_visual_rows != expected_condition_rows
                    || request.rows.condition_audio_rows != 0
            }
            // Ref2VA's three condition row counts are exactly the ordered
            // reference totals the factory re-derived above.
            Some((visual_rows, audio_rows, qwen_vision_rows, _)) => {
                request.rows.condition_visual_rows != visual_rows
                    || request.rows.condition_audio_rows != audio_rows
                    || request.rows.qwen_vision_rows != qwen_vision_rows
            }
        }
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
        || checkpoint.retained_header_host_bytes == 0
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
    // Both VAEs are resident from their load through condition encoding, and
    // again from their post-denoise reload through the mux. The load/drop
    // policy parks them for everything in between, so the noise-allocation,
    // transformer-load, and denoise phases below charge no VAE weights —
    // charging them there priced ~5.8 GB inside the transformer's own peak
    // for memory the runtime has already released.
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
    // The one live packed block, plus the tensor currently being read held
    // twice: `read_tensor_bytes`' `Vec` and the `from_raw_buffer` CPU copy it
    // is turned into (`comfy_dit.rs:1373-1407`, `:1451-1462`). Charging one
    // copy undercounted every block load by its largest tensor.
    let max_streamed_block_host_overlap = checkpoint
        .blocks
        .iter()
        .map(|block| {
            block
                .max_host_read_staging_bytes
                .checked_mul(2)
                .and_then(|staging| staging.checked_add(block.encoded_host_bytes))
                .ok_or_else(|| anyhow!("H3 streamed block host overlap overflow"))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    // One dense non-block tensor at a time reaches host memory during the
    // fixed transformer load, and it lands on the device before the next is
    // read (`comfy_dit.rs:1410-1447`: read `Vec` -> `from_raw_buffer` CPU copy
    // -> optional `to_dtype` upcast -> `to_device`). The bound is therefore two
    // encoded copies of the largest tensor plus its widened form, NOT the sum
    // of every fixed tensor's bytes — those are device-resident weights that
    // `fixed_transformer_device_bytes` already charges.
    let fixed_transformer_host_staging = checked_u64_sum(
        [
            checkpoint.fixed_transformer_max_host_read_staging_bytes,
            checkpoint.fixed_transformer_max_host_read_staging_bytes,
            checkpoint.fixed_transformer_max_device_weight_staging_bytes,
        ],
        "H3 fixed transformer host staging",
    )?;
    // Host demand is a per-phase max, exactly as the device peak above is, and
    // it counts only ANONYMOUS bytes. Two whole classes are deliberately absent
    // from every phase sum:
    //
    // * `artifact_host_bytes` — the sum of every artifact's FILE size. Nothing
    //   in this pipeline holds a whole artifact in host RAM: the Qwen and the
    //   transformer stream through bounded `Vec`s with seek+read_exact
    //   (`qwen_nvfp4.rs:820-856`, `comfy_dit.rs:1373-1407`), and the VAEs are
    //   mmap'd. Charging ~42 GB of file bytes as anonymous demand is the #1108
    //   LTX-2 bug class verbatim.
    // * `vae_peak_host_mapped_file_bytes` — a genuine mapping
    //   (`visual_weights.rs:178`, `audio_weights.rs:413`), but file-backed and
    //   reclaimable, and `MemAvailable` — this ledger's own input via
    //   `h3_admission::current_h3_host_memory` — already counts those pages as
    //   available. See `ltx2_cpu_gemma_streams_from_mmap` for the precedent.
    //
    // `vae_peak_staging_disk_bytes` was already excluded: it is disk, fenced
    // separately by `ensure_staging_capacity`.
    let attempt_host = checked_u64_sum(
        [
            memory.fixed_runtime_host_bytes,
            memory.endpoint_encoded_host_bytes,
            memory.normalized_endpoint_host_bytes,
        ],
        "H3 attempt-long host demand",
    )?;
    // Opened-component metadata each authority retains alongside the payload
    // it streams. Small, but genuinely anonymous, so the phase sums carry it
    // for exactly as long as its authority is alive: the Qwen raw header until
    // the conditioner is dropped after encode, the transformer's parsed header
    // until the transformer is dropped after denoise, and the VAE authorities'
    // two config buffers until the post-denoise reload consumes the second one.
    let qwen_alive_metadata_host = checked_u64_sum(
        [
            memory.qwen_retained_header_host_bytes,
            memory.transformer_retained_header_host_bytes,
            memory.vae_retained_config_host_bytes,
        ],
        "H3 opened metadata host demand",
    )?;
    let transformer_alive_metadata_host = checked_u64_sum(
        [
            memory.transformer_retained_header_host_bytes,
            memory.vae_retained_config_host_bytes,
        ],
        "H3 post-Qwen metadata host demand",
    )?;
    let vae_alive_metadata_host = memory.vae_retained_config_host_bytes;
    // Ref2VA's four leading phases. Media is decoded and normalized on the
    // host before any model is loaded, so their device demand is the fixed
    // runtime alone; the retained normalized media then stays charged through
    // both reference encoders, which are what finally consume it.
    //
    // The retained-media charge is re-derived from the frozen reference plan
    // rather than taken from the budget, for the same reason the prepared
    // request re-derives its rows: this number is what the host ledger grants.
    let reference_media = match request.task {
        Task::Fl2va => H3ReferenceMediaTotals::default(),
        Task::Ref2va => validate_prepared_references(&request.references)?.3,
    };
    let expected_reference_media_host_bytes = reference_media.normalized_host_bytes;
    // The audio encoder has no measured bound of its own; it borrows the
    // decoder's, which runs the larger BigVGAN stack. Deriving it from that
    // already-bound field rather than accepting a caller value keeps it from
    // being a free parameter the budget can name.
    let expected_reference_audio_encode_workspace = if request.task == Task::Ref2va {
        memory.audio_decode_workspace_device_bytes
    } else {
        0
    };
    // The transients are derived here too, never named by the caller.
    let expected_reference_decode_staging = reference_media.largest_native_host_bytes;
    let expected_reference_preprocess_staging = reference_media.largest_normalized_host_bytes;
    // Decode retains EVERY reference's native payload at once: the
    // orchestrator decodes the whole ordered set before preprocessing any, and
    // each decoded slot survives until its own preprocess step removes it. The
    // transient on top is the one being materialized.
    let reference_decode_host = checked_u64_sum(
        [
            attempt_host,
            qwen_alive_metadata_host,
            reference_media.native_host_bytes,
            expected_reference_decode_staging,
        ],
        "H3 reference decode host phase",
    )?;
    // Preprocess converts one reference at a time, replacing its native slot
    // with a normalized one. For any point in that walk the held set is
    // (natives not yet converted) + (normalized already produced), which is
    // bounded by both totals together — the honest upper bound, and the only
    // one that does not depend on the order the two sizes happen to fall in.
    let reference_preprocess_host = checked_u64_sum(
        [
            attempt_host,
            qwen_alive_metadata_host,
            reference_media.native_host_bytes,
            reference_media.normalized_host_bytes,
            expected_reference_preprocess_staging,
        ],
        "H3 reference preprocess host phase",
    )?;
    // By the encoders every native payload has been released.
    let reference_encode_host = checked_u64_sum(
        [
            attempt_host,
            transformer_alive_metadata_host,
            reference_media.normalized_host_bytes,
            memory.condition_backing_host_bytes,
            memory.packed_layout_host_bytes,
            memory.text_modality_tags_host_bytes,
        ],
        "H3 reference encode host phase",
    )?;
    let vae_load_host = checked_u64_sum(
        [
            attempt_host,
            transformer_alive_metadata_host,
            memory.vae_peak_host_io_buffer_bytes,
        ],
        "H3 VAE load host phase",
    )?;
    let qwen_encode_host = checked_u64_sum(
        [
            attempt_host,
            qwen_alive_metadata_host,
            memory.qwen_host_workspace_bytes,
            memory.qwen_host_load_staging_bytes,
            memory.text_modality_tags_host_bytes,
        ],
        "H3 Qwen encode host phase",
    )?;
    let qwen_transfer_host = checked_u64_sum(
        [
            attempt_host,
            qwen_alive_metadata_host,
            memory.qwen_host_workspace_bytes,
            memory.text_modality_tags_host_bytes,
        ],
        "H3 Qwen transfer host phase",
    )?;
    let condition_encode_host = checked_u64_sum(
        [
            attempt_host,
            transformer_alive_metadata_host,
            memory.condition_backing_host_bytes,
            memory.packed_layout_host_bytes,
            memory.packed_layout_construction_staging_host_bytes,
            memory.packed_layout_freeze_staging_host_bytes,
            memory.text_modality_tags_host_bytes,
            memory.noise_cpu_staging_host_bytes,
        ],
        "H3 condition encode host phase",
    )?;
    let noise_allocation_host = checked_u64_sum(
        [
            attempt_host,
            transformer_alive_metadata_host,
            memory.condition_backing_host_bytes,
            memory.packed_layout_host_bytes,
            memory.text_modality_tags_host_bytes,
            memory.schedule_host_bytes,
            memory.noise_cpu_staging_host_bytes,
        ],
        "H3 noise allocation host phase",
    )?;
    let transformer_load_host = transformer_load_phase_host_bytes(H3TransformerLoadHostTerms {
        attempt_host_bytes: attempt_host,
        transformer_alive_metadata_host_bytes: transformer_alive_metadata_host,
        condition_backing_host_bytes: memory.condition_backing_host_bytes,
        packed_layout_host_bytes: memory.packed_layout_host_bytes,
        text_modality_tags_host_bytes: memory.text_modality_tags_host_bytes,
        schedule_host_bytes: memory.schedule_host_bytes,
        fixed_transformer_load_host_staging_bytes: memory.fixed_transformer_load_host_staging_bytes,
        turbo_adapter_host_staging_bytes: memory.turbo_adapter_host_staging_bytes,
    })?;
    let denoise_host = checked_u64_sum(
        [
            attempt_host,
            transformer_alive_metadata_host,
            memory.condition_backing_host_bytes,
            memory.packed_layout_host_bytes,
            memory.text_modality_tags_host_bytes,
            memory.schedule_host_bytes,
            memory.max_streamed_block_host_overlap_bytes,
        ],
        "H3 denoise host phase",
    )?;
    let visual_decode_host = checked_u64_sum(
        [
            attempt_host,
            vae_alive_metadata_host,
            memory.packed_layout_host_bytes,
            memory.vae_peak_host_io_buffer_bytes,
            memory.encoded_video_host_bytes_bound,
            memory.thumbnail_host_bytes_bound,
        ],
        "H3 visual decode host phase",
    )?;
    let audio_decode_host = checked_u64_sum(
        [
            attempt_host,
            memory.packed_layout_host_bytes,
            memory.encoded_video_host_bytes_bound,
            memory.thumbnail_host_bytes_bound,
            memory.waveform_host_bytes,
        ],
        "H3 audio decode host phase",
    )?;
    let waveform_transfer_host = checked_u64_sum(
        [
            attempt_host,
            memory.encoded_video_host_bytes_bound,
            memory.thumbnail_host_bytes_bound,
            memory.waveform_host_bytes,
        ],
        "H3 waveform transfer host phase",
    )?;
    let mux_host = checked_u64_sum(
        [
            attempt_host,
            memory.encoded_video_host_bytes_bound,
            memory.thumbnail_host_bytes_bound,
            memory.waveform_host_bytes,
            memory.mux_output_host_bytes_bound,
            memory.aac_mux_staging_host_bytes,
        ],
        "H3 mux host phase",
    )?;
    // A phase that is not in this task's order contributes nothing and must be
    // recorded as zero, so a budget can never carry a ledger for work the
    // worker will not perform.
    let ref2va = request.task == Task::Ref2va;
    let (
        expected_reference_decode_host,
        expected_reference_preprocess_host,
        expected_reference_visual_encode_host,
        expected_reference_audio_encode_host,
        expected_condition_encode_host,
    ) = if ref2va {
        (
            reference_decode_host,
            reference_preprocess_host,
            reference_encode_host,
            reference_encode_host,
            0,
        )
    } else {
        (0, 0, 0, 0, condition_encode_host)
    };
    let predicted_host = [
        expected_reference_decode_host,
        expected_reference_preprocess_host,
        expected_reference_visual_encode_host,
        expected_reference_audio_encode_host,
        vae_load_host,
        qwen_encode_host,
        qwen_transfer_host,
        expected_condition_encode_host,
        noise_allocation_host,
        transformer_load_host,
        denoise_host,
        visual_decode_host,
        audio_decode_host,
        waveform_transfer_host,
        mux_host,
    ]
    .into_iter()
    .max()
    .unwrap_or_default();
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
    // Host-only phases: nothing model-sized is resident on the device yet.
    let reference_decode = memory.fixed_runtime_device_bytes;
    let reference_preprocess = memory.fixed_runtime_device_bytes;
    // Both reference encoders run against the resident VAEs with the Qwen
    // output already transferred, exactly as FL2VA's condition encode does.
    let reference_visual_encode = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.qwen_output_state_device_bytes,
            memory.condition_vae_workspace_device_bytes,
            memory.condition_latent_backing_device_bytes,
            memory.packed_layout_device_bytes,
        ],
        "H3 reference visual encode phase",
    )?;
    let reference_audio_encode = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.qwen_output_state_device_bytes,
            memory.reference_audio_encode_workspace_device_bytes,
            memory.condition_latent_backing_device_bytes,
            memory.packed_layout_device_bytes,
        ],
        "H3 reference audio encode phase",
    )?;
    let vae_load = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.vae_construction_device_workspace_bytes,
            memory.qwen_output_state_device_bytes,
        ],
        "H3 VAE load phase",
    )?;
    let qwen_encode = match conditioner_placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop
        | H3FactoryConditionerPlacement::AssignedMetalThenDrop => checked_u64_sum(
            [
                memory.fixed_runtime_device_bytes,
                memory.qwen_device_parameter_bytes,
                memory.qwen_activation_device_bytes,
                memory.qwen_output_state_device_bytes,
            ],
            "H3 Qwen encode phase",
        )?,
        H3FactoryConditionerPlacement::HostCpuThenDrop => memory.fixed_runtime_device_bytes,
    };
    let qwen_transfer = match conditioner_placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop
        | H3FactoryConditionerPlacement::AssignedMetalThenDrop => 0,
        H3FactoryConditionerPlacement::HostCpuThenDrop => checked_u64_sum(
            [
                memory.fixed_runtime_device_bytes,
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
    let transformer_load = transformer_load_phase_device_bytes(H3TransformerLoadDeviceTerms {
        fixed_runtime_device_bytes: memory.fixed_runtime_device_bytes,
        fixed_transformer_device_bytes: memory.fixed_transformer_device_bytes,
        qwen_output_state_device_bytes: memory.qwen_output_state_device_bytes,
        condition_latent_backing_device_bytes: memory.condition_latent_backing_device_bytes,
        packed_layout_device_bytes: memory.packed_layout_device_bytes,
        packed_video_state_device_bytes: memory.packed_video_state_device_bytes,
        packed_audio_state_device_bytes: memory.packed_audio_state_device_bytes,
        resident_block_device_bytes: resident_blocks,
        fixed_transformer_load_device_staging_bytes: memory
            .fixed_transformer_load_device_staging_bytes,
        turbo_adapter_device_bytes: memory.turbo_adapter_device_bytes,
        turbo_adapter_device_staging_bytes: memory.turbo_adapter_device_staging_bytes,
    })?;
    let denoise_copy_workspace = memory
        .packed_video_state_device_bytes
        .checked_add(memory.packed_audio_state_device_bytes)
        .and_then(|bytes| bytes.checked_mul(9))
        .ok_or_else(|| anyhow!("H3 paired RES multistep copy budget overflow"))?;
    let denoise = denoise_phase_device_bytes(H3DenoiseDeviceTerms {
        fixed_runtime_device_bytes: memory.fixed_runtime_device_bytes,
        fixed_transformer_device_bytes: memory.fixed_transformer_device_bytes,
        qwen_output_state_device_bytes: memory.qwen_output_state_device_bytes,
        condition_latent_backing_device_bytes: memory.condition_latent_backing_device_bytes,
        packed_layout_device_bytes: memory.packed_layout_device_bytes,
        packed_video_state_device_bytes: memory.packed_video_state_device_bytes,
        packed_audio_state_device_bytes: memory.packed_audio_state_device_bytes,
        denoise_tensor_copy_workspace_device_bytes: memory
            .denoise_tensor_copy_workspace_device_bytes,
        denoise_transient_workspace_device_bytes: denoise_transient_workspace_device_bytes(
            memory.attention_workspace_device_bytes,
            memory.ffn_workspace_device_bytes,
        ),
        denoise_hidden_activation_device_bytes: denoise_hidden_activation_device_bytes(
            request.rows.total_packed_rows,
        )?,
        resident_block_device_bytes: resident_blocks,
        streamed_block_device_overlap_bytes: streamed_block_overlap,
        prefetch_device_bytes: expected_prefetch,
        max_device_weight_staging_bytes: max_device_staging,
        // The adapter never streams: it is loaded once at transformer load and
        // stays device-resident for every evaluation.
        turbo_adapter_device_bytes: memory.turbo_adapter_device_bytes,
    })?;
    let visual_decode = checked_u64_sum(
        [
            memory.fixed_runtime_device_bytes,
            retained_vaes,
            memory.vae_construction_device_workspace_bytes,
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
    let (
        expected_reference_decode,
        expected_reference_preprocess,
        expected_reference_visual_encode,
        expected_reference_audio_encode,
        expected_condition_encode,
    ) = if ref2va {
        (
            reference_decode,
            reference_preprocess,
            reference_visual_encode,
            reference_audio_encode,
            0,
        )
    } else {
        (0, 0, 0, 0, condition_encode)
    };
    let predicted_device = [
        expected_reference_decode,
        expected_reference_preprocess,
        expected_reference_visual_encode,
        expected_reference_audio_encode,
        vae_load,
        qwen_encode,
        qwen_transfer,
        expected_condition_encode,
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
    // Name the first mismatching field instead of reporting an opaque
    // inconsistency. This sweep compares ~110 terms between the builder's
    // budget and this validator's independent recomputation; when the two
    // disagree, the field name and both values are the only thing that makes
    // the disagreement locatable from a production error string.
    let mut mismatches: Vec<String> = Vec::new();
    macro_rules! expect_eq {
        ($label:literal, $observed:expr, $expected:expr) => {{
            let observed = $observed;
            let expected = $expected;
            if observed != expected {
                mismatches.push(format!("{} is {observed:?}, expected {expected:?}", $label));
            }
        }};
    }
    macro_rules! expect {
        ($label:literal, $cond:expr) => {
            if !($cond) {
                mismatches.push(format!("{} is invalid", $label));
            }
        };
    }

    // Each task has exactly one executable load/drop order, and the budget's
    // phase ledgers are only meaningful against that order.
    let expected_load_drop_policy = match request.task {
        Task::Fl2va => H3FactoryTargetLoadDropPolicy::LoadQwenEncodeTransferDropQwenLoadVaesEncodeConditionsParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux,
        Task::Ref2va => H3FactoryTargetLoadDropPolicy::DecodeReferencesPreprocessReferencesLoadQwenEncodeVisionTransferDropQwenLoadVaesEncodeVisualReferencesEncodeAudioReferencesParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux,
    };
    expect_eq!(
        "load_drop_policy",
        memory.load_drop_policy,
        expected_load_drop_policy
    );
    expect_eq!(
        "artifact_host_bytes",
        memory.artifact_host_bytes,
        authenticated_artifact_bytes
    );
    expect!(
        "fixed_runtime_host_bytes",
        memory.fixed_runtime_host_bytes != 0
    );
    expect!(
        "fixed_runtime_device_bytes",
        memory.fixed_runtime_device_bytes != 0
    );
    expect_eq!(
        "fixed_transformer_device_bytes",
        memory.fixed_transformer_device_bytes,
        checkpoint.fixed_transformer_protected_device_bytes
    );
    expect!(
        "visual_vae_resident_device_bytes",
        memory.visual_vae_resident_device_bytes != 0
    );
    expect!(
        "audio_vae_resident_device_bytes",
        memory.audio_vae_resident_device_bytes != 0
    );
    expect!(
        "vae_construction_device_workspace_bytes",
        memory.vae_construction_device_workspace_bytes != 0
    );
    expect!(
        "vae_memory_evidence_identity_sha256",
        require_sha256(
            &memory.vae_memory_evidence_identity_sha256,
            "H3 VAE memory evidence",
        )
        .is_ok()
    );
    expect!(
        "target_video_latent_device_bytes",
        memory.target_video_latent_device_bytes != 0
    );
    expect!(
        "target_audio_latent_device_bytes",
        memory.target_audio_latent_device_bytes != 0
    );
    expect_eq!(
        "attempt_resident_vae_device_bytes",
        memory.attempt_resident_vae_device_bytes,
        retained_vaes
    );
    expect_eq!(
        "qwen_host_workspace_bytes",
        memory.qwen_host_workspace_bytes,
        qwen_host_workspace
    );
    expect_eq!(
        "endpoint_encoded_host_bytes",
        memory.endpoint_encoded_host_bytes,
        endpoint_encoded_bytes
    );
    expect_eq!(
        "normalized_endpoint_host_bytes",
        memory.normalized_endpoint_host_bytes,
        normalized_endpoint_bytes
    );
    expect_eq!(
        "schedule_host_bytes",
        memory.schedule_host_bytes,
        schedule_host_bytes
    );
    expect_eq!(
        "packed_layout_host_bytes",
        memory.packed_layout_host_bytes,
        packed_layout_host_bytes
    );
    expect_eq!(
        "packed_layout_construction_staging_host_bytes",
        memory.packed_layout_construction_staging_host_bytes,
        packed_layout_construction_staging_host_bytes
    );
    expect_eq!(
        "packed_layout_freeze_staging_host_bytes",
        memory.packed_layout_freeze_staging_host_bytes,
        packed_layout_freeze_staging_host_bytes
    );
    expect_eq!(
        "text_modality_tags_host_bytes",
        memory.text_modality_tags_host_bytes,
        text_modality_tags_host_bytes
    );
    expect_eq!(
        "protected_block_device_bytes",
        memory.protected_block_device_bytes,
        protected_blocks
    );
    expect_eq!(
        "resident_block_device_bytes",
        memory.resident_block_device_bytes,
        resident_blocks
    );
    expect_eq!(
        "streamed_block_device_bytes",
        memory.streamed_block_device_bytes,
        streamed_blocks
    );
    expect_eq!(
        "prefetch_device_bytes",
        memory.prefetch_device_bytes,
        expected_prefetch
    );
    expect_eq!(
        "streamed_block_device_overlap_bytes",
        memory.streamed_block_device_overlap_bytes,
        streamed_block_overlap
    );
    expect_eq!(
        "dequantization_workspace_device_bytes",
        memory.dequantization_workspace_device_bytes,
        max_device_staging
    );
    expect_eq!(
        "max_device_weight_staging_bytes",
        memory.max_device_weight_staging_bytes,
        max_device_staging
    );
    expect_eq!(
        "max_host_read_staging_bytes",
        memory.max_host_read_staging_bytes,
        max_host_staging
    );
    expect_eq!(
        "max_streamed_block_host_overlap_bytes",
        memory.max_streamed_block_host_overlap_bytes,
        max_streamed_block_host_overlap
    );
    expect_eq!(
        "fixed_transformer_load_host_staging_bytes",
        memory.fixed_transformer_load_host_staging_bytes,
        fixed_transformer_host_staging
    );
    expect_eq!(
        "fixed_transformer_load_device_staging_bytes",
        memory.fixed_transformer_load_device_staging_bytes,
        checkpoint.fixed_transformer_max_device_weight_staging_bytes
    );
    expect_eq!(
        "reference_decode_phase_host_bytes",
        memory.reference_decode_phase_host_bytes,
        expected_reference_decode_host
    );
    expect_eq!(
        "reference_preprocess_phase_host_bytes",
        memory.reference_preprocess_phase_host_bytes,
        expected_reference_preprocess_host
    );
    expect_eq!(
        "reference_visual_encode_phase_host_bytes",
        memory.reference_visual_encode_phase_host_bytes,
        expected_reference_visual_encode_host
    );
    expect_eq!(
        "reference_audio_encode_phase_host_bytes",
        memory.reference_audio_encode_phase_host_bytes,
        expected_reference_audio_encode_host
    );
    expect_eq!(
        "reference_decode_phase_device_bytes",
        memory.reference_decode_phase_device_bytes,
        expected_reference_decode
    );
    expect_eq!(
        "reference_preprocess_phase_device_bytes",
        memory.reference_preprocess_phase_device_bytes,
        expected_reference_preprocess
    );
    expect_eq!(
        "reference_visual_encode_phase_device_bytes",
        memory.reference_visual_encode_phase_device_bytes,
        expected_reference_visual_encode
    );
    expect_eq!(
        "reference_audio_encode_phase_device_bytes",
        memory.reference_audio_encode_phase_device_bytes,
        expected_reference_audio_encode
    );
    // Reference charges exist exactly for the task that has references, and
    // the retained media is pinned to the re-derived plan total.
    let expected_reference_media_identity =
        expected_h3_factory_reference_media_identity(&request.references);
    expect_eq!(
        "reference_media_identity_sha256",
        memory.reference_media_identity_sha256.as_str(),
        expected_reference_media_identity.as_str()
    );
    expect_eq!(
        "reference_normalized_media_host_bytes",
        memory.reference_normalized_media_host_bytes,
        expected_reference_media_host_bytes
    );
    // Recomputed from the frozen media facts, not merely nonzero-checked: the
    // phase totals above are derived from these same values, so a caller that
    // could name them could name its own grant.
    expect_eq!(
        "reference_decode_staging_host_bytes",
        memory.reference_decode_staging_host_bytes,
        expected_reference_decode_staging
    );
    expect_eq!(
        "reference_preprocess_staging_host_bytes",
        memory.reference_preprocess_staging_host_bytes,
        expected_reference_preprocess_staging
    );
    // The audio VAE's encoder workspace is only ever charged by Ref2VA, and
    // its size comes from the qualified bounds record.
    expect_eq!(
        "reference_audio_encode_workspace_device_bytes",
        memory.reference_audio_encode_workspace_device_bytes,
        expected_reference_audio_encode_workspace
    );
    expect_eq!(
        "vae_load_phase_host_bytes",
        memory.vae_load_phase_host_bytes,
        vae_load_host
    );
    expect_eq!(
        "qwen_encode_phase_host_bytes",
        memory.qwen_encode_phase_host_bytes,
        qwen_encode_host
    );
    expect_eq!(
        "qwen_transfer_phase_host_bytes",
        memory.qwen_transfer_phase_host_bytes,
        qwen_transfer_host
    );
    expect_eq!(
        "condition_encode_phase_host_bytes",
        memory.condition_encode_phase_host_bytes,
        expected_condition_encode_host
    );
    expect_eq!(
        "noise_allocation_phase_host_bytes",
        memory.noise_allocation_phase_host_bytes,
        noise_allocation_host
    );
    expect_eq!(
        "transformer_load_phase_host_bytes",
        memory.transformer_load_phase_host_bytes,
        transformer_load_host
    );
    expect_eq!(
        "denoise_phase_host_bytes",
        memory.denoise_phase_host_bytes,
        denoise_host
    );
    expect_eq!(
        "visual_decode_phase_host_bytes",
        memory.visual_decode_phase_host_bytes,
        visual_decode_host
    );
    expect_eq!(
        "audio_decode_phase_host_bytes",
        memory.audio_decode_phase_host_bytes,
        audio_decode_host
    );
    expect_eq!(
        "waveform_transfer_phase_host_bytes",
        memory.waveform_transfer_phase_host_bytes,
        waveform_transfer_host
    );
    expect_eq!(
        "mux_phase_host_bytes",
        memory.mux_phase_host_bytes,
        mux_host
    );
    expect_eq!(
        "transformer_retained_header_host_bytes",
        memory.transformer_retained_header_host_bytes,
        checkpoint.retained_header_host_bytes
    );
    // The VAE's retained config bytes are bound by the same opened memory
    // evidence identity every other VAE-derived field is bound by.
    expect!(
        "vae_retained_config_host_bytes",
        memory.vae_retained_config_host_bytes != 0
    );
    expect_eq!(
        "predicted_host_increment_bytes",
        memory.predicted_host_increment_bytes,
        predicted_host
    );
    expect_eq!(
        "vae_load_phase_device_bytes",
        memory.vae_load_phase_device_bytes,
        vae_load
    );
    expect_eq!(
        "qwen_encode_phase_device_bytes",
        memory.qwen_encode_phase_device_bytes,
        qwen_encode
    );
    expect_eq!(
        "qwen_transfer_phase_device_bytes",
        memory.qwen_transfer_phase_device_bytes,
        qwen_transfer
    );
    expect_eq!(
        "condition_encode_phase_device_bytes",
        memory.condition_encode_phase_device_bytes,
        expected_condition_encode
    );
    expect_eq!(
        "noise_allocation_phase_device_bytes",
        memory.noise_allocation_phase_device_bytes,
        noise_allocation
    );
    expect_eq!(
        "transformer_load_phase_device_bytes",
        memory.transformer_load_phase_device_bytes,
        transformer_load
    );
    expect_eq!(
        "denoise_phase_device_bytes",
        memory.denoise_phase_device_bytes,
        denoise
    );
    expect_eq!(
        "visual_decode_phase_device_bytes",
        memory.visual_decode_phase_device_bytes,
        visual_decode
    );
    expect_eq!(
        "audio_decode_phase_device_bytes",
        memory.audio_decode_phase_device_bytes,
        audio_decode
    );
    expect_eq!(
        "waveform_transfer_phase_device_bytes",
        memory.waveform_transfer_phase_device_bytes,
        waveform_transfer
    );
    expect_eq!("mux_phase_device_bytes", memory.mux_phase_device_bytes, 0);
    expect_eq!(
        "predicted_device_peak_bytes",
        memory.predicted_device_peak_bytes,
        predicted_device
    );
    match conditioner_placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop
        | H3FactoryConditionerPlacement::AssignedMetalThenDrop => {
            expect!(
                "qwen_host_parameter_bytes (cuda placement)",
                memory.qwen_host_parameter_bytes != 0
            );
            expect_eq!(
                "qwen_host_activation_bytes (cuda placement)",
                memory.qwen_host_activation_bytes,
                0
            );
            expect_eq!(
                "qwen_host_output_state_bytes (cuda placement)",
                memory.qwen_host_output_state_bytes,
                0
            );
            expect!(
                "qwen_device_parameter_bytes (cuda placement)",
                memory.qwen_device_parameter_bytes != 0
            );
            expect!(
                "qwen_activation_device_bytes (cuda placement)",
                memory.qwen_activation_device_bytes != 0
            );
            expect_eq!(
                "qwen_output_transfer_device_bytes (cuda placement)",
                memory.qwen_output_transfer_device_bytes,
                0
            );
        }
        H3FactoryConditionerPlacement::HostCpuThenDrop => {
            expect!(
                "qwen_host_parameter_bytes (host placement)",
                memory.qwen_host_parameter_bytes != 0
            );
            expect!(
                "qwen_host_activation_bytes (host placement)",
                memory.qwen_host_activation_bytes != 0
            );
            expect_eq!(
                "qwen_host_output_state_bytes (host placement)",
                memory.qwen_host_output_state_bytes,
                memory.qwen_output_state_device_bytes
            );
            expect_eq!(
                "qwen_device_parameter_bytes (host placement)",
                memory.qwen_device_parameter_bytes,
                0
            );
            expect_eq!(
                "qwen_activation_device_bytes (host placement)",
                memory.qwen_activation_device_bytes,
                0
            );
            expect_eq!(
                "qwen_output_transfer_device_bytes (host placement)",
                memory.qwen_output_transfer_device_bytes,
                memory.qwen_output_state_device_bytes
            );
        }
    }
    expect_eq!(
        "target_video_latent_device_bytes",
        memory.target_video_latent_device_bytes,
        request
            .rows
            .target_video_rows
            .checked_mul(96 * 4)
            .ok_or_else(|| anyhow!("H3 target video latent bytes overflow"))?
    );
    expect_eq!(
        "target_audio_latent_device_bytes",
        memory.target_audio_latent_device_bytes,
        request
            .rows
            .target_audio_rows
            .checked_mul(32 * 4)
            .ok_or_else(|| anyhow!("H3 target audio latent bytes overflow"))?
    );
    expect_eq!(
        "condition_latent_backing_device_bytes",
        memory.condition_latent_backing_device_bytes,
        request
            .rows
            .condition_visual_rows
            .checked_mul(96 * 4)
            .ok_or_else(|| anyhow!("H3 condition latent bytes overflow"))?
    );
    expect_eq!(
        "packed_video_state_device_bytes",
        memory.packed_video_state_device_bytes,
        memory
            .condition_latent_backing_device_bytes
            .checked_add(memory.target_video_latent_device_bytes)
            .ok_or_else(|| anyhow!("H3 packed video state bytes overflow"))?
    );
    expect_eq!(
        "packed_audio_state_device_bytes",
        memory.packed_audio_state_device_bytes,
        memory.target_audio_latent_device_bytes
    );
    expect_eq!(
        "packed_layout_device_bytes",
        memory.packed_layout_device_bytes,
        request
            .rows
            .total_packed_rows
            .checked_mul(24)
            .ok_or_else(|| anyhow!("H3 packed layout bytes overflow"))?
    );
    expect_eq!(
        "qwen_output_state_device_bytes",
        memory.qwen_output_state_device_bytes,
        request
            .rows
            .qwen_output_text_rows
            .checked_mul(5_120 * 2)
            .ok_or_else(|| anyhow!("H3 Qwen output state bytes overflow"))?
    );
    expect_eq!(
        "noise_cpu_staging_host_bytes",
        memory.noise_cpu_staging_host_bytes,
        memory
            .condition_latent_backing_device_bytes
            .max(memory.target_video_latent_device_bytes)
            .max(memory.target_audio_latent_device_bytes)
    );
    expect_eq!(
        "waveform_host_bytes",
        memory.waveform_host_bytes,
        request
            .audio_samples_per_channel
            .checked_mul(u64::from(contract::AUDIO_CHANNELS))
            .and_then(|samples| samples.checked_mul(4))
            .ok_or_else(|| anyhow!("H3 waveform host bytes overflow"))?
    );
    expect_eq!(
        "audio_waveform_device_bytes",
        memory.audio_waveform_device_bytes,
        memory.waveform_host_bytes
    );
    expect!(
        "denoise_copy_policy",
        memory.denoise_copy_policy
            == H3FactoryTargetDenoiseCopyPolicy::CandleF32PairedResMultistepV2
    );
    expect_eq!(
        "denoise_tensor_copy_workspace_device_bytes",
        memory.denoise_tensor_copy_workspace_device_bytes,
        denoise_copy_workspace
    );
    expect!(
        "vae_peak_host_io_buffer_bytes",
        memory.vae_peak_host_io_buffer_bytes != 0
    );
    expect!(
        "vae_peak_host_mapped_file_bytes",
        memory.vae_peak_host_mapped_file_bytes != 0
    );
    expect!(
        "vae_peak_staging_disk_bytes",
        memory.vae_peak_staging_disk_bytes != 0
    );
    expect!(
        "encoded_video_host_bytes_bound",
        memory.encoded_video_host_bytes_bound != 0
    );
    expect!(
        "thumbnail_host_bytes_bound",
        memory.thumbnail_host_bytes_bound != 0
    );
    expect!(
        "mux_output_host_bytes_bound",
        memory.mux_output_host_bytes_bound != 0
    );
    expect!(
        "aac_mux_staging_host_bytes",
        memory.aac_mux_staging_host_bytes != 0
    );
    expect!(
        "attention_workspace_device_bytes",
        memory.attention_workspace_device_bytes != 0
    );
    expect!(
        "ffn_workspace_device_bytes",
        memory.ffn_workspace_device_bytes != 0
    );
    expect!(
        "decoder_tile_workspace_device_bytes",
        memory.decoder_tile_workspace_device_bytes != 0
    );
    expect!(
        "audio_decode_workspace_device_bytes",
        memory.audio_decode_workspace_device_bytes != 0
    );
    if request.rows.condition_visual_rows == 0 {
        expect_eq!(
            "condition_vae_workspace_device_bytes (no condition rows)",
            memory.condition_vae_workspace_device_bytes,
            0
        );
        expect_eq!(
            "condition_backing_host_bytes/condition_latent_backing_device_bytes (no condition rows)",
            condition_bytes,
            (0, 0)
        );
    } else {
        expect!(
            "condition_vae_workspace_device_bytes",
            memory.condition_vae_workspace_device_bytes != 0
        );
        expect!("condition_backing_host_bytes", condition_bytes.0 != 0);
        expect!(
            "condition_latent_backing_device_bytes",
            condition_bytes.1 != 0
        );
    }
    expect!(
        "identity_sha256",
        memory.identity_sha256 == expected_target_budget_identity(memory)
    );

    if let Some(first) = mismatches.first() {
        bail!("MiniMax H3 target budget is internally inconsistent: {first}");
    }
    Ok(())
}

fn checked_u64_sum(values: impl IntoIterator<Item = u64>, label: &'static str) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| anyhow!("{label} overflow"))
    })
}

/// The attention and FFN workspace bounds never coexist on the device: within
/// one transformer block the attention call's transients (QKV projection,
/// kernel auxiliaries, output-projection staging) are explicitly dropped
/// before the strictly-sequential MLP call allocates
/// (`mold_candle::minimax_h3::dit`, block forwards). The denoise phase
/// therefore charges the larger of the two per-workspace bounds, never their
/// sum. The hidden-sized tensors that DO stay live across that boundary are
/// charged by `denoise_hidden_activation_device_bytes`, not here.
/// Device-byte terms live during the transformer-load phase.
///
/// The evidence builder and [`validate_target_budget`] both fill this in and
/// call [`transformer_load_phase_device_bytes`], so the phase can no longer be
/// transcribed two different ways. That is not hypothetical: the Turbo staging
/// term was added to the validator's sum and missed in the builder's, and every
/// fixture test still passed because those build both sides from one reference.
/// Adding a term is now a struct field, i.e. a compile error at both call sites
/// until each supplies it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct H3TransformerLoadDeviceTerms {
    pub(crate) fixed_runtime_device_bytes: u64,
    pub(crate) fixed_transformer_device_bytes: u64,
    pub(crate) qwen_output_state_device_bytes: u64,
    pub(crate) condition_latent_backing_device_bytes: u64,
    pub(crate) packed_layout_device_bytes: u64,
    pub(crate) packed_video_state_device_bytes: u64,
    pub(crate) packed_audio_state_device_bytes: u64,
    pub(crate) resident_block_device_bytes: u64,
    pub(crate) fixed_transformer_load_device_staging_bytes: u64,
    pub(crate) turbo_adapter_device_bytes: u64,
    pub(crate) turbo_adapter_device_staging_bytes: u64,
}

pub(crate) fn transformer_load_phase_device_bytes(
    terms: H3TransformerLoadDeviceTerms,
) -> Result<u64> {
    let H3TransformerLoadDeviceTerms {
        fixed_runtime_device_bytes,
        fixed_transformer_device_bytes,
        qwen_output_state_device_bytes,
        condition_latent_backing_device_bytes,
        packed_layout_device_bytes,
        packed_video_state_device_bytes,
        packed_audio_state_device_bytes,
        resident_block_device_bytes,
        fixed_transformer_load_device_staging_bytes,
        turbo_adapter_device_bytes,
        turbo_adapter_device_staging_bytes,
    } = terms;
    checked_u64_sum(
        [
            fixed_runtime_device_bytes,
            fixed_transformer_device_bytes,
            qwen_output_state_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            resident_block_device_bytes,
            fixed_transformer_load_device_staging_bytes,
            turbo_adapter_device_bytes,
            turbo_adapter_device_staging_bytes,
        ],
        "H3 transformer load phase",
    )
}

/// Device-byte terms live during the denoise phase.
///
/// The Turbo adapter's residents are here but its upload staging is not: the
/// transposed copies are released before the first evaluation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct H3DenoiseDeviceTerms {
    pub(crate) fixed_runtime_device_bytes: u64,
    pub(crate) fixed_transformer_device_bytes: u64,
    pub(crate) qwen_output_state_device_bytes: u64,
    pub(crate) condition_latent_backing_device_bytes: u64,
    pub(crate) packed_layout_device_bytes: u64,
    pub(crate) packed_video_state_device_bytes: u64,
    pub(crate) packed_audio_state_device_bytes: u64,
    pub(crate) denoise_tensor_copy_workspace_device_bytes: u64,
    pub(crate) denoise_transient_workspace_device_bytes: u64,
    pub(crate) denoise_hidden_activation_device_bytes: u64,
    pub(crate) resident_block_device_bytes: u64,
    pub(crate) streamed_block_device_overlap_bytes: u64,
    pub(crate) prefetch_device_bytes: u64,
    pub(crate) max_device_weight_staging_bytes: u64,
    pub(crate) turbo_adapter_device_bytes: u64,
}

pub(crate) fn denoise_phase_device_bytes(terms: H3DenoiseDeviceTerms) -> Result<u64> {
    let H3DenoiseDeviceTerms {
        fixed_runtime_device_bytes,
        fixed_transformer_device_bytes,
        qwen_output_state_device_bytes,
        condition_latent_backing_device_bytes,
        packed_layout_device_bytes,
        packed_video_state_device_bytes,
        packed_audio_state_device_bytes,
        denoise_tensor_copy_workspace_device_bytes,
        denoise_transient_workspace_device_bytes,
        denoise_hidden_activation_device_bytes,
        resident_block_device_bytes,
        streamed_block_device_overlap_bytes,
        prefetch_device_bytes,
        max_device_weight_staging_bytes,
        turbo_adapter_device_bytes,
    } = terms;
    checked_u64_sum(
        [
            fixed_runtime_device_bytes,
            fixed_transformer_device_bytes,
            qwen_output_state_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            denoise_tensor_copy_workspace_device_bytes,
            denoise_transient_workspace_device_bytes,
            denoise_hidden_activation_device_bytes,
            resident_block_device_bytes,
            streamed_block_device_overlap_bytes,
            prefetch_device_bytes,
            max_device_weight_staging_bytes,
            turbo_adapter_device_bytes,
        ],
        "H3 denoise phase",
    )
}

/// Host-byte terms live during the transformer-load phase.
#[derive(Clone, Copy, Debug)]
pub(crate) struct H3TransformerLoadHostTerms {
    pub(crate) attempt_host_bytes: u64,
    pub(crate) transformer_alive_metadata_host_bytes: u64,
    pub(crate) condition_backing_host_bytes: u64,
    pub(crate) packed_layout_host_bytes: u64,
    pub(crate) text_modality_tags_host_bytes: u64,
    pub(crate) schedule_host_bytes: u64,
    pub(crate) fixed_transformer_load_host_staging_bytes: u64,
    pub(crate) turbo_adapter_host_staging_bytes: u64,
}

pub(crate) fn transformer_load_phase_host_bytes(terms: H3TransformerLoadHostTerms) -> Result<u64> {
    let H3TransformerLoadHostTerms {
        attempt_host_bytes,
        transformer_alive_metadata_host_bytes,
        condition_backing_host_bytes,
        packed_layout_host_bytes,
        text_modality_tags_host_bytes,
        schedule_host_bytes,
        fixed_transformer_load_host_staging_bytes,
        turbo_adapter_host_staging_bytes,
    } = terms;
    checked_u64_sum(
        [
            attempt_host_bytes,
            transformer_alive_metadata_host_bytes,
            condition_backing_host_bytes,
            packed_layout_host_bytes,
            text_modality_tags_host_bytes,
            schedule_host_bytes,
            fixed_transformer_load_host_staging_bytes,
            turbo_adapter_host_staging_bytes,
        ],
        "H3 transformer load host phase",
    )
}

pub(crate) fn denoise_transient_workspace_device_bytes(
    attention_workspace_device_bytes: u64,
    ffn_workspace_device_bytes: u64,
) -> u64 {
    attention_workspace_device_bytes.max(ffn_workspace_device_bytes)
}

/// The transformer's hidden width, pinned to the released config the
/// qualification path validates (`private_qualification.rs`,
/// `validate_transformer_config`: `hidden_size: 5_376`).
const H3_HIDDEN_SIZE: u64 = 5_376;

/// Hidden-sized (`rows x 5376` BF16) activations that remain live while the
/// denoise transient workspace peaks. With the explicit drops in the block
/// forwards, at most three such tensors coexist inside a block — the block's
/// residual hidden, the active normalized input, and the in-flight
/// projection — plus the caller's running sequence tensor across the block
/// boundary, where the returned output and the previous hidden overlap.
/// Charged as four, which also absorbs the six per-modality AdaLN parameter
/// tensors (orders of magnitude smaller than one hidden tensor).
const H3_DENOISE_LIVE_HIDDEN_ACTIVATIONS: u64 = 4;

/// Baseline device charge for the transformer's live hidden activations
/// during denoise. These are neither in the attention/FFN workspace bounds
/// (whose authority excludes borrowed inputs) nor in the packed-state terms
/// (which are latent-sized, 96/32 bytes per row, not `hidden_size`-sized).
pub(crate) fn denoise_hidden_activation_device_bytes(total_packed_rows: u64) -> Result<u64> {
    total_packed_rows
        .checked_mul(H3_HIDDEN_SIZE)
        .and_then(|elements| elements.checked_mul(2))
        .and_then(|bytes| bytes.checked_mul(H3_DENOISE_LIVE_HIDDEN_ACTIVATIONS))
        .ok_or_else(|| anyhow!("MiniMax H3 denoise hidden-activation budget overflow"))
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
    update_reference_identity(&mut hash, &request.references);
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
    hash.update(checkpoint.retained_header_host_bytes.to_le_bytes());
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

/// Append the complete ordered reference facts to `hash`.
///
/// Shared by the prepared-request identity and the target-budget's own
/// reference digest so the two can never describe different geometry.
fn update_reference_identity(hash: &mut Sha256, references: &[H3FactoryReferenceInput]) {
    hash.update((references.len() as u64).to_le_bytes());
    for reference in references {
        hash.update(reference.index.to_le_bytes());
        hash.update(reference_kind_id(reference.kind));
        update_string(hash, &reference.content_sha256);
        hash.update(reference.preprocess_version.to_le_bytes());
        for value in [
            reference.normalized_width,
            reference.normalized_height,
            reference.normalized_video_frames,
            reference.video_frames,
            reference.qwen_video_frames,
            reference.native_width,
            reference.native_height,
        ] {
            // Absent and zero are different geometries; tag the option so a
            // missing axis can never hash as a present zero.
            hash.update([u8::from(value.is_some())]);
            hash.update(value.unwrap_or_default().to_le_bytes());
        }
        for value in [
            reference.audio_samples_per_channel,
            reference.native_audio_samples_per_channel,
        ] {
            hash.update([u8::from(value.is_some())]);
            hash.update(value.unwrap_or_default().to_le_bytes());
        }
        hash.update([u8::from(reference.native_audio_channels.is_some())]);
        hash.update(
            reference
                .native_audio_channels
                .unwrap_or_default()
                .to_le_bytes(),
        );
        for value in [
            reference.visual_rows,
            reference.audio_rows,
            reference.qwen_vision_rows,
            reference.normalized_host_bytes,
            reference.native_host_bytes,
        ] {
            hash.update(value.to_le_bytes());
        }
    }
}

/// The digest a Ref2VA target budget carries so its byte totals are bound to
/// the exact geometry that produced them. Empty for a reference-free request.
pub fn expected_h3_factory_reference_media_identity(
    references: &[H3FactoryReferenceInput],
) -> String {
    if references.is_empty() {
        return String::new();
    }
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.reference-media-facts.v1\0");
    update_reference_identity(&mut hash, references);
    format!("{:x}", hash.finalize())
}

pub fn expected_h3_factory_target_budget_identity(memory: &H3FactoryTargetBudgetInput) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.target-attempt-budget.v2\0");
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

fn reference_kind_id(kind: H3FactoryReferenceKind) -> &'static [u8] {
    match kind {
        H3FactoryReferenceKind::Image => b"image",
        H3FactoryReferenceKind::Video => b"video",
        H3FactoryReferenceKind::Audio => b"audio",
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

/// Validate the frozen attention tuple against the plan's own device class.
///
/// `compute_capability` is `Some` for a CUDA plan and `None` for Metal. The
/// two routes are separate exact tuples, not a relaxation of one: CUDA
/// requires the SM89 FlashAttention kernel, Metal requires the chunked dense
/// correctness kernel, and neither device may present the other's.
fn validate_attention(
    attention: &H3FactoryAttentionInput,
    compute_capability: Option<(u16, u16)>,
) -> Result<()> {
    require_sha256(&attention.runtime_identity_sha256, "H3 attention runtime")?;
    require_sha256(
        &attention.qualification_sha256,
        "H3 attention qualification",
    )?;
    let expected_tuple = match compute_capability {
        Some(compute_capability) => (
            AttentionBackend::Flash,
            H3AttentionBackend::FlashAttentionV2,
            H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
            H3AttentionActivation::ReleaseCandidateQualificationOnly,
            H3AttentionDevice::Cuda {
                compute_capability: Some(compute_capability),
            },
        ),
        // H3 never routes through `crate::attention` on Metal — the frozen
        // plan carries its own query chunk — so the generic backend stays at
        // the process default and the generic chunk policy stays off.
        None => (
            AttentionBackend::Math,
            H3AttentionBackend::MetalChunkedDenseMath,
            H3AttentionKernel::CandleDenseChunkedF32V011,
            H3AttentionActivation::MetalCorrectnessOnly,
            H3AttentionDevice::Metal,
        ),
    };
    if attention.generic_backend != expected_tuple.0
        || attention.generic_chunk != AttentionChunkPolicy::Off
        || attention.runtime_backend != expected_tuple.1
        || attention.kernel != expected_tuple.2
        || attention.activation != expected_tuple.3
        || attention.device != expected_tuple.4
        || attention.model_contract != H3AttentionModelContract::released_bf16()
        || attention.qualification_kernel_identity != attention.kernel.identity()
        || !attention.full_noncausal
        || !attention.lossless
    {
        bail!("MiniMax H3 factory requires one exact released attention tuple for its device");
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
        #[cfg(not(feature = "h3"))]
        if model_contract.generation.runtime_available {
            bail!(
                "contract-only MiniMax H3 factory authority cannot be created for a runnable contract"
            );
        }
        if input.device_id.trim().is_empty() || matches!(input.compute_capability, Some((0, _))) {
            bail!("MiniMax H3 factory authority requires one concrete device route");
        }
        require_sha256(&input.execution_fingerprint, "H3 scheduler execution")?;
        if input.qwen_parameter_bytes == 0
            || input
                .qwen_host_resident_parameter_bytes
                .checked_add(input.qwen_device_resident_parameter_bytes)
                .is_none_or(|bytes| bytes == 0)
            || input.qwen_activation_workspace_bytes == 0
            || input.qwen_maximum_tensor_staging_bytes == 0
            || input.qwen_retained_raw_header_bytes == 0
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
                            qwen_maximum_tensor_staging_bytes: input
                                .qwen_maximum_tensor_staging_bytes,
                            qwen_retained_raw_header_bytes: input.qwen_retained_raw_header_bytes,
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
                                turbo_adapter,
                                ..
                            },
                            Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                        ) if transformer_policy_sha256
                            == &attempt.raw_checkpoint.quantization_policy_identity_sha256 =>
                        {
                            // The budget's Turbo terms are admissible only if
                            // this authority declared an adapter, and they must
                            // be exactly the bytes it declared.
                            let declared_device = turbo_adapter
                                .as_ref()
                                .map_or(0, H3FactoryTurboAdapterAuthority::resident_device_bytes);
                            let declared_device_staging = turbo_adapter.as_ref().map_or(
                                0,
                                H3FactoryTurboAdapterAuthority::device_staging_peak_bytes,
                            );
                            let declared_host = turbo_adapter
                                .as_ref()
                                .map_or(0, H3FactoryTurboAdapterAuthority::host_staging_peak_bytes);
                            if attempt.target_budget.turbo_adapter_device_bytes != declared_device
                                || attempt.target_budget.turbo_adapter_device_staging_bytes
                                    != declared_device_staging
                                || attempt.target_budget.turbo_adapter_host_staging_bytes
                                    != declared_host
                            {
                                bail!(
                                    "MiniMax H3 target budget Turbo adapter bytes disagree with the quantization authority"
                                );
                            }
                            if let Some(turbo_adapter) = turbo_adapter {
                                if attempt.request.grid_points != turbo_adapter.grid_points() {
                                    bail!(
                                        "MiniMax H3 prepared request uses {} grid points but its Turbo tier is reviewed for {}",
                                        attempt.request.grid_points,
                                        turbo_adapter.grid_points()
                                    );
                                }
                            }
                        }
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
        if input.compute_capability.is_none()
            && input.conditioner_placement == H3FactoryConditionerPlacement::AssignedCudaThenDrop
        {
            bail!("MiniMax H3 CUDA conditioner placement does not match its frozen Metal backend");
        }
        if input.compute_capability.is_some()
            && input.conditioner_placement == H3FactoryConditionerPlacement::AssignedMetalThenDrop
        {
            bail!("MiniMax H3 Metal conditioner placement does not match its frozen CUDA backend");
        }
        let conditioner_execution = match input.conditioner_placement {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
                H3ConditionerExecution::CudaResident
            }
            H3FactoryConditionerPlacement::AssignedMetalThenDrop => {
                H3ConditionerExecution::MetalResident
            }
            H3FactoryConditionerPlacement::HostCpuThenDrop => H3ConditionerExecution::CpuOffloaded,
        };
        let conditioner_device = match input.conditioner_placement {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop
            | H3FactoryConditionerPlacement::AssignedMetalThenDrop => input.device_id.clone(),
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
            match input.compute_capability {
                Some(compute_capability) => H3CandleBackendDevice::Cuda { compute_capability },
                None => H3CandleBackendDevice::Metal,
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
            qwen_maximum_tensor_staging_bytes: input.qwen_maximum_tensor_staging_bytes,
            qwen_retained_raw_header_bytes: input.qwen_retained_raw_header_bytes,
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
                qwen_maximum_tensor_staging_bytes: self.qwen_maximum_tensor_staging_bytes,
                qwen_retained_raw_header_bytes: self.qwen_retained_raw_header_bytes,
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

    // Shipping `h3` execution must consume activation evidence through
    // `private_fl2va_runtime_authority_with_activation`; only the private UAT
    // build retains this no-evidence projection.
    #[cfg(feature = "h3-private-uat")]
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
                // The default tuple mirrors the frozen device class: SM89
                // FlashAttention on CUDA, chunked dense correctness on Metal.
                let (runtime_backend, kernel, activation, device) = match self.compute_capability()
                {
                    Some(compute_capability) => (
                        H3AttentionBackend::FlashAttentionV2,
                        H3AttentionKernel::CandleFlashFwdHdim128Bf16Sm80V011,
                        H3AttentionActivation::ReleaseCandidateQualificationOnly,
                        H3AttentionDevice::Cuda {
                            compute_capability: Some(compute_capability),
                        },
                    ),
                    None => (
                        H3AttentionBackend::MetalChunkedDenseMath,
                        H3AttentionKernel::CandleDenseChunkedF32V011,
                        H3AttentionActivation::MetalCorrectnessOnly,
                        H3AttentionDevice::Metal,
                    ),
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

    /// The CUDA compute capability this plan froze, if it is a CUDA plan.
    ///
    /// Metal has none — its one attention route is qualified by backend, not
    /// by architecture — so callers that gate on SM89 must treat `None` as
    /// "not a CUDA route" rather than substituting a default.
    pub fn compute_capability(&self) -> Option<(u16, u16)> {
        match self.backend_plan.backend() {
            H3CandleBackendDevice::Cuda { compute_capability } => Some(compute_capability),
            H3CandleBackendDevice::Metal => None,
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

    pub const fn qwen_maximum_tensor_staging_bytes(&self) -> u64 {
        self.qwen_maximum_tensor_staging_bytes
    }

    pub const fn qwen_retained_raw_header_bytes(&self) -> u64 {
        self.qwen_retained_raw_header_bytes
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
        // Internal preparation authorities name the base engine partition,
        // while the final engine constructor carries the full reviewed media
        // identity (including its Turbo tier). This seam accepts either exact
        // representation; production dispatch below accepts only the latter
        // and proves the adapter tier with `media_model_matches_h3_authority`.
        let model_matches = request_contract.canonical_model == self.canonical_model()
            || media_model_matches_h3_authority(model, self);
        let mut drift = Vec::new();
        if !model_matches {
            drift.push(format!(
                "model {model:?} is not the frozen partition {:?}",
                self.canonical_model()
            ));
        }
        if request_contract.task != self.task() {
            drift.push(format!(
                "task {:?} vs frozen {:?}",
                request_contract.task,
                self.task()
            ));
        }
        if gpu_ordinal != self.device_ordinal {
            drift.push(format!(
                "device ordinal {gpu_ordinal} vs frozen {}",
                self.device_ordinal
            ));
        }
        if offload != self.block_offload {
            drift.push(format!(
                "block offload {offload} vs frozen {}",
                self.block_offload
            ));
        }
        if !drift.is_empty() {
            // Name every field that moved: a bare "authority changed" cannot be
            // acted on, and an ordinal or offload drift is a very different
            // bug from a model-identity drift.
            bail!(
                "MiniMax H3 frozen engine authority changed before construction: {}",
                drift.join("; ")
            );
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
            || self.qwen_maximum_tensor_staging_bytes == 0
            || self.qwen_retained_raw_header_bytes == 0
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
                    qwen_maximum_tensor_staging_bytes: self.qwen_maximum_tensor_staging_bytes,
                    qwen_retained_raw_header_bytes: self.qwen_retained_raw_header_bytes,
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
        let mut drift = Vec::new();
        if !contract::is_family(family) {
            drift.push(format!("family {family:?} is not MiniMax H3"));
        }
        if !media_model_matches_h3_authority(model, self) {
            drift.push(format!(
                "model {model:?} does not match the frozen partition {:?}",
                self.canonical_model()
            ));
        }
        if request_contract.task != self.task() {
            drift.push(format!(
                "task {:?} vs frozen {:?}",
                request_contract.task,
                self.task()
            ));
        }
        if gpu_ordinal != self.device_ordinal {
            drift.push(format!(
                "device ordinal {gpu_ordinal} vs frozen {}",
                self.device_ordinal
            ));
        }
        if offload != self.block_offload {
            drift.push(format!(
                "block offload {offload} vs frozen {}",
                self.block_offload
            ));
        }
        if attention_backend != self.attention_backend {
            drift.push(format!(
                "attention backend {attention_backend:?} vs frozen {:?}",
                self.attention_backend
            ));
        }
        if attention_chunk != self.attention_chunk {
            drift.push(format!(
                "attention chunk {attention_chunk:?} vs frozen {:?}",
                self.attention_chunk
            ));
        }
        if !drift.is_empty() {
            bail!(
                "MiniMax H3 frozen route, attention, or offload authority changed before dispatch: {}",
                drift.join("; ")
            );
        }
        let missing_prepared_attempt = self.prepared_attempt.is_none();
        let missing_budget_echo = self.execution_budget_echo.is_none();
        let missing_typed_attention = self.attention_runtime.is_none();
        let missing_runnable_contract =
            contract::runnable_capability_contract_for_model(model).is_none();
        let missing_family_registry =
            crate::production_family_capability_for_family(family).is_none();
        if missing_prepared_attempt
            || missing_budget_echo
            || missing_typed_attention
            || missing_runnable_contract
            || missing_family_registry
        {
            bail!(
                "MiniMax H3 public runtime registry is incomplete: prepared_attempt={}; budget_echo={}; typed_attention={}; runnable_contract={}; family_registry={}",
                !missing_prepared_attempt,
                !missing_budget_echo,
                !missing_typed_attention,
                !missing_runnable_contract,
                !missing_family_registry,
            );
        }
        Ok(())
    }
}

fn frozen_identity(authority: &FrozenH3FactoryAuthority) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.factory-authority.v6\0");
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
        H3FactoryConditionerPlacement::AssignedMetalThenDrop => b"qwen-metal".as_slice(),
        H3FactoryConditionerPlacement::HostCpuThenDrop => b"qwen-cpu".as_slice(),
    });
    hash.update(authority.qwen_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_host_resident_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_device_resident_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_activation_workspace_bytes.to_le_bytes());
    hash.update(authority.qwen_maximum_tensor_staging_bytes.to_le_bytes());
    hash.update(authority.qwen_retained_raw_header_bytes.to_le_bytes());
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

    /// Opened-loader facts the fixture stands in for. The budget's staging and
    /// retained-metadata terms are DERIVED from these, exactly as production
    /// derives them from the authenticated runtime facts, so a test cannot
    /// invent a value the validator would then have to accept.
    const FIXTURE_QWEN_MAX_TENSOR_STAGING_BYTES: u64 = 350;
    const FIXTURE_QWEN_RETAINED_HEADER_BYTES: u64 = 40;
    const FIXTURE_TRANSFORMER_RETAINED_HEADER_BYTES: u64 = 60;
    const FIXTURE_VAE_RETAINED_CONFIG_BYTES: u64 = 20;

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
            references: Vec::new(),
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

    /// One 2048-canvas image, one 768-canvas video with a soundtrack, and one
    /// standalone audio reference — the mixed ordered set the Ref2VA contract
    /// is specified against. Row and byte charges are written out longhand so
    /// this fixture is an independent statement of the arithmetic rather than
    /// a call to the code under test.
    ///
    /// Native geometry is deliberately much larger than normalized: a 6000x4000
    /// still normalizes to 2048x2048, and a 3840x2160 clip to 768x768. That gap
    /// is the whole point of charging the decode phase natively — it is what a
    /// normalized-only ledger silently loses.
    fn ref2va_references() -> Vec<H3FactoryReferenceInput> {
        let image_rows = (2_048 / 32) * (2_048 / 32);
        let video_rows_per_frame = (768 / 32) * (768 / 32);
        // 39 visual-VAE frames land on 5n+2 causal latents over the family's
        // own 17n+5 grid: (39-5)/17*5+2 = 12.
        let video_latent_frames = 12;
        // 48 normalized frames at 24 fps sample 4 cursor frames, which Qwen
        // consumes as 2 temporal-patch-2 blocks.
        let video_qwen_blocks = 2;
        vec![
            H3FactoryReferenceInput {
                index: 1,
                kind: H3FactoryReferenceKind::Image,
                content_sha256: sha('a'),
                preprocess_version: contract::REFERENCE_PREPROCESS_VERSION,
                normalized_width: Some(2_048),
                normalized_height: Some(2_048),
                normalized_video_frames: None,
                video_frames: None,
                qwen_video_frames: None,
                audio_samples_per_channel: None,
                native_width: Some(6_000),
                native_height: Some(4_000),
                native_audio_samples_per_channel: None,
                native_audio_channels: None,
                visual_rows: image_rows,
                audio_rows: 0,
                qwen_vision_rows: image_rows,
                normalized_host_bytes: 2_048 * 2_048 * 3,
                native_host_bytes: 6_000 * 4_000 * 3,
            },
            H3FactoryReferenceInput {
                index: 2,
                kind: H3FactoryReferenceKind::Video,
                content_sha256: sha('b'),
                preprocess_version: contract::REFERENCE_PREPROCESS_VERSION,
                normalized_width: Some(768),
                normalized_height: Some(768),
                normalized_video_frames: Some(48),
                video_frames: Some(39),
                qwen_video_frames: Some(4),
                audio_samples_per_channel: Some(64_000),
                native_width: Some(3_840),
                native_height: Some(2_160),
                native_audio_samples_per_channel: Some(96_000),
                native_audio_channels: Some(6),
                visual_rows: video_latent_frames * video_rows_per_frame,
                audio_rows: 64_000_u64.div_ceil(800) * 2,
                qwen_vision_rows: video_qwen_blocks * video_rows_per_frame,
                normalized_host_bytes: (39 + 4) * 768 * 768 * 3 + 64_000 * 2 * 4,
                // Video frames are resized to the normalized canvas DURING
                // decode, so what is retained between decode and preprocess is
                // the normalized frame; only the soundtrack stays native.
                native_host_bytes: (39 + 4) * 768 * 768 * 3 + 96_000 * 6 * 4,
            },
            H3FactoryReferenceInput {
                index: 3,
                kind: H3FactoryReferenceKind::Audio,
                content_sha256: sha('c'),
                preprocess_version: contract::REFERENCE_PREPROCESS_VERSION,
                normalized_width: None,
                normalized_height: None,
                normalized_video_frames: None,
                video_frames: None,
                qwen_video_frames: None,
                audio_samples_per_channel: Some(32_000),
                native_width: None,
                native_height: None,
                native_audio_samples_per_channel: Some(48_000),
                native_audio_channels: Some(2),
                visual_rows: 0,
                audio_rows: 32_000_u64.div_ceil(800) * 2,
                qwen_vision_rows: 0,
                normalized_host_bytes: 32_000 * 2 * 4,
                native_host_bytes: 48_000 * 2 * 4,
            },
        ]
    }

    fn ref2va_prepared_request() -> H3FactoryPreparedRequestInput {
        let base = prepared_request();
        let references = ref2va_references();
        let condition_visual_rows = references
            .iter()
            .map(|entry| entry.visual_rows)
            .sum::<u64>();
        let condition_audio_rows = references.iter().map(|entry| entry.audio_rows).sum::<u64>();
        let qwen_vision_rows = references
            .iter()
            .map(|entry| entry.qwen_vision_rows)
            .sum::<u64>();
        let mut request = H3FactoryPreparedRequestInput {
            canonical_model: contract::REF2VA_COMFY.into(),
            task: Task::Ref2va,
            mode: Mode::ReferenceToAudioVideo,
            endpoints: Vec::new(),
            references,
            rows: H3FactoryPreparedRowsInput {
                qwen_output_text_rows: base.rows.qwen_output_text_rows,
                qwen_vision_rows,
                condition_visual_rows,
                condition_audio_rows,
                target_video_rows: base.rows.target_video_rows,
                target_audio_rows: base.rows.target_audio_rows,
                total_packed_rows: base.rows.qwen_output_text_rows
                    + condition_visual_rows
                    + condition_audio_rows
                    + base.rows.target_video_rows
                    + base.rows.target_audio_rows,
            },
            ..base
        };
        request.identity_sha256 = expected_h3_factory_prepared_request_identity(&request);
        request
    }

    #[test]
    fn ref2va_prepared_request_prices_every_ordered_reference_modality() {
        let request = ref2va_prepared_request();
        validate_prepared_request(&request)
            .expect("the mixed ordered Ref2VA reference set must be admissible");

        // The three condition row counts are exactly the per-reference totals,
        // and mirror what mold-server's admission pricing already computes.
        assert_eq!(request.rows.condition_visual_rows, 4_096 + 12 * 576);
        assert_eq!(request.rows.condition_audio_rows, 80 * 2 + 40 * 2);
        assert_eq!(request.rows.qwen_vision_rows, 4_096 + 2 * 576);
        // Retained normalized media is the item-2 host charge: one 2048 RGB8
        // still, 43 768-square RGB8 frames, and two f32 stereo waveforms.
        let (_, _, _, media) = validate_prepared_references(&request.references).unwrap();
        assert_eq!(
            media.normalized_host_bytes,
            2_048 * 2_048 * 3 + 43 * 768 * 768 * 3 + 64_000 * 2 * 4 + 32_000 * 2 * 4
        );
        assert_eq!(
            media.normalized_host_bytes,
            12_582_912 + 76_087_296 + 512_000 + 256_000
        );
    }

    #[test]
    fn ref2va_prepared_request_rejects_every_reference_authority_mutation() {
        let base = ref2va_prepared_request();
        let reseal = |mut request: H3FactoryPreparedRequestInput| {
            request.identity_sha256 = expected_h3_factory_prepared_request_identity(&request);
            request
        };

        // An understated row charge is the whole point of re-deriving: it
        // would otherwise be admitted into a budget that cannot hold it.
        let mut understated = base.clone();
        understated.references[0].visual_rows -= 1;
        understated.rows.condition_visual_rows -= 1;
        understated.rows.total_packed_rows -= 1;
        assert!(validate_prepared_request(&reseal(understated)).is_err());

        // Same for the retained host bytes.
        let mut cheap_media = base.clone();
        cheap_media.references[1].normalized_host_bytes -= 1;
        assert!(validate_prepared_request(&reseal(cheap_media)).is_err());

        // Reordered or non-contiguous references break the packed sequence.
        let mut reordered = base.clone();
        reordered.references.swap(0, 1);
        assert!(validate_prepared_request(&reseal(reordered)).is_err());

        // Modality and geometry must agree.
        let mut audio_with_canvas = base.clone();
        audio_with_canvas.references[2].normalized_width = Some(64);
        audio_with_canvas.references[2].normalized_height = Some(64);
        assert!(validate_prepared_request(&reseal(audio_with_canvas)).is_err());

        let mut image_with_frames = base.clone();
        image_with_frames.references[0].video_frames = Some(9);
        assert!(validate_prepared_request(&reseal(image_with_frames)).is_err());

        // The Qwen 2 fps cursor contract is not negotiable.
        let mut resampled = base.clone();
        resampled.references[1].qwen_video_frames = Some(3);
        assert!(validate_prepared_request(&reseal(resampled)).is_err());

        // A prepared shape from a different preprocessing contract cannot be
        // priced by this factory at all.
        let mut stale_version = base.clone();
        stale_version.references[0].preprocess_version = contract::REFERENCE_PREPROCESS_VERSION + 1;
        assert!(validate_prepared_request(&reseal(stale_version)).is_err());

        // Every reference field participates in the frozen identity.
        for mutate in [
            (|request: &mut H3FactoryPreparedRequestInput| request.references[0].index = 9)
                as fn(&mut H3FactoryPreparedRequestInput),
            |request| request.references[0].content_sha256 = sha('9'),
            |request| request.references[2].audio_samples_per_channel = Some(800),
            |request| request.references.truncate(2),
        ] {
            let mut mutated = base.clone();
            mutate(&mut mutated);
            assert_ne!(
                expected_h3_factory_prepared_request_identity(&mutated),
                base.identity_sha256
            );
        }
    }

    #[test]
    fn conditioning_contracts_never_cross_between_the_two_tasks() {
        let reseal = |mut request: H3FactoryPreparedRequestInput| {
            request.identity_sha256 = expected_h3_factory_prepared_request_identity(&request);
            request
        };

        // FL2VA may not carry references, and Ref2VA may not carry endpoints.
        let mut fl2va_with_references = prepared_request();
        fl2va_with_references.references = ref2va_references();
        assert!(validate_prepared_request(&reseal(fl2va_with_references)).is_err());

        let mut ref2va_with_endpoints = ref2va_prepared_request();
        ref2va_with_endpoints.endpoints = prepared_request().endpoints;
        assert!(validate_prepared_request(&reseal(ref2va_with_endpoints)).is_err());

        // A Ref2VA request with no references at all has nothing to condition
        // on and must not be admitted as an unconditioned generation.
        let mut empty = ref2va_prepared_request();
        empty.references = Vec::new();
        empty.rows.condition_visual_rows = 0;
        empty.rows.condition_audio_rows = 0;
        empty.rows.qwen_vision_rows = 0;
        empty.rows.total_packed_rows = empty.rows.qwen_output_text_rows
            + empty.rows.target_video_rows
            + empty.rows.target_audio_rows;
        assert!(validate_prepared_request(&reseal(empty)).is_err());

        // Audio-only references cannot carry a Ref2VA generation either.
        let mut audio_only = ref2va_prepared_request();
        audio_only
            .references
            .retain(|entry| entry.kind == H3FactoryReferenceKind::Audio);
        audio_only.references[0].index = 1;
        audio_only.rows.condition_visual_rows = 0;
        audio_only.rows.qwen_vision_rows = 0;
        audio_only.rows.condition_audio_rows = audio_only.references[0].audio_rows;
        audio_only.rows.total_packed_rows = audio_only.rows.qwen_output_text_rows
            + audio_only.rows.condition_audio_rows
            + audio_only.rows.target_video_rows
            + audio_only.rows.target_audio_rows;
        assert!(validate_prepared_request(&reseal(audio_only)).is_err());
    }

    #[test]
    fn each_task_pins_exactly_one_executable_load_drop_order() {
        let fl2va = H3FactoryTargetLoadDropPolicy::LoadQwenEncodeTransferDropQwenLoadVaesEncodeConditionsParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux;
        let ref2va = H3FactoryTargetLoadDropPolicy::DecodeReferencesPreprocessReferencesLoadQwenEncodeVisionTransferDropQwenLoadVaesEncodeVisualReferencesEncodeAudioReferencesParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux;
        assert_ne!(fl2va, ref2va);

        // The discriminator participates in the target-budget identity, so a
        // budget measured against one order cannot be replayed as the other.
        let mut base = target_budget(&prepared_request(), &raw_checkpoint());
        base.load_drop_policy = fl2va;
        let fl2va_identity = expected_h3_factory_target_budget_identity(&base);
        base.load_drop_policy = ref2va;
        assert_ne!(
            expected_h3_factory_target_budget_identity(&base),
            fl2va_identity
        );
    }

    /// Drive the ledger authority directly. The prepared-attempt wrapper adds
    /// identity plumbing that is exercised elsewhere; what matters here is
    /// that the budget's phase arithmetic agrees with the request's task.
    fn check_budget(
        budget: &H3FactoryTargetBudgetInput,
        request: &H3FactoryPreparedRequestInput,
        checkpoint: &H3FactoryRawCheckpointInput,
    ) -> Result<()> {
        validate_target_budget(
            budget,
            request,
            checkpoint,
            0,
            0,
            H3FactoryConditionerPlacement::HostCpuThenDrop,
        )
    }

    #[test]
    fn ref2va_budget_charges_four_reference_phases_and_drops_condition_encode() {
        let request = ref2va_prepared_request();
        let checkpoint = raw_checkpoint();
        let budget = target_budget(&request, &checkpoint);
        check_budget(&budget, &request, &checkpoint)
            .expect("the Ref2VA reference ledgers must validate against their own load/drop order");

        // The retained normalized media is the headline host charge and is
        // pinned to the re-derived reference plan, not to a flat constant.
        let (_, _, _, plan_media) = validate_prepared_references(&request.references).unwrap();
        let plan_media_bytes = plan_media.normalized_host_bytes;
        assert_eq!(
            budget.reference_normalized_media_host_bytes,
            plan_media_bytes
        );
        assert!(plan_media_bytes > 88_000_000);

        // Every reference phase is charged, and it outlives decode into both
        // encoders, so the media appears in all four ledgers.
        for phase in [
            budget.reference_decode_phase_host_bytes,
            budget.reference_preprocess_phase_host_bytes,
            budget.reference_visual_encode_phase_host_bytes,
            budget.reference_audio_encode_phase_host_bytes,
        ] {
            assert!(phase > plan_media_bytes);
        }
        assert!(budget.reference_visual_encode_phase_device_bytes > 0);
        assert!(budget.reference_audio_encode_phase_device_bytes > 0);
        // The host-only leading phases hold no model weights.
        assert_eq!(
            budget.reference_decode_phase_device_bytes,
            budget.fixed_runtime_device_bytes
        );
        assert_eq!(
            budget.reference_preprocess_phase_device_bytes,
            budget.fixed_runtime_device_bytes
        );

        // Condition encode is not in Ref2VA's order at all.
        assert_eq!(budget.condition_encode_phase_host_bytes, 0);
        assert_eq!(budget.condition_encode_phase_device_bytes, 0);

        // The predicted peaks are a max over the phases that actually run, so
        // the retained media has to be visible in the host grant.
        assert!(budget.predicted_host_increment_bytes >= plan_media_bytes);
    }

    /// The decode phase holds EVERY reference's native payload at once.
    ///
    /// `pipeline/ref2va.rs` decodes all references before preprocessing any,
    /// and `reference_media.rs` keeps each decoded slot until that reference's
    /// own preprocess step removes it. So the peak is the sum of native
    /// payloads across the whole ordered set, not a per-reference maximum, and
    /// it is native rather than normalized — the gap between the two is
    /// unbounded in principle (a 100 MP still normalizes to 2048x2048) and is
    /// exactly what a normalized-only ledger loses. Under-charging here lets
    /// the scheduler grant less host memory than the runtime goes on to hold.
    #[test]
    fn reference_decode_charges_every_native_payload_held_at_once() {
        let request = ref2va_prepared_request();
        let checkpoint = raw_checkpoint();
        let budget = target_budget(&request, &checkpoint);
        check_budget(&budget, &request, &checkpoint).unwrap();

        let native_total =
            6_000 * 4_000 * 3 + (39 + 4) * 768 * 768 * 3 + 96_000 * 6 * 4 + 48_000 * 2 * 4;
        let normalized_total =
            2_048 * 2_048 * 3 + (39 + 4) * 768 * 768 * 3 + 64_000 * 2 * 4 + 32_000 * 2 * 4;
        // The still alone contributes 72 MB natively against 12.6 MB
        // normalized, which is the gap a normalized-only ledger loses.
        assert!(native_total > normalized_total);

        let (_, _, _, media) = validate_prepared_references(&request.references).unwrap();
        assert_eq!(media.native_host_bytes, native_total);
        assert_eq!(media.normalized_host_bytes, normalized_total);

        // Decode holds every native payload simultaneously.
        assert!(budget.reference_decode_phase_host_bytes > native_total);
        // Preprocess converts one at a time, so natives not yet converted and
        // normalized ones already produced are held together.
        assert!(budget.reference_preprocess_phase_host_bytes > native_total + normalized_total);
        // Both encoders run after every native payload is released.
        assert!(budget.reference_visual_encode_phase_host_bytes > normalized_total);
        assert!(
            budget.reference_visual_encode_phase_host_bytes
                < budget.reference_preprocess_phase_host_bytes
        );

        // The host grant must cover the true peak.
        assert!(budget.predicted_host_increment_bytes >= native_total + normalized_total);
    }

    /// The decoder resizes video frames to the normalized canvas while
    /// decoding, so a clip whose source is SMALLER than its normalized canvas
    /// retains more than its source dimensions imply. Pricing retained frames
    /// at source resolution is safe only when the canvas shrinks; it
    /// under-charges exactly when it grows, which is the direction that lets
    /// the runtime exceed its grant.
    #[test]
    fn an_upscaled_video_reference_is_charged_at_its_retained_canvas() {
        let mut references = ref2va_references();
        references.truncate(1);
        // A 32x32 source normalized up to 384x384: 144x the pixels per frame.
        references.push(H3FactoryReferenceInput {
            index: 2,
            kind: H3FactoryReferenceKind::Video,
            content_sha256: sha('d'),
            preprocess_version: contract::REFERENCE_PREPROCESS_VERSION,
            normalized_width: Some(384),
            normalized_height: Some(384),
            normalized_video_frames: Some(48),
            video_frames: Some(39),
            qwen_video_frames: Some(4),
            audio_samples_per_channel: None,
            native_width: Some(32),
            native_height: Some(32),
            native_audio_samples_per_channel: None,
            native_audio_channels: None,
            visual_rows: 12 * (384 / 32) * (384 / 32),
            audio_rows: 0,
            qwen_vision_rows: 2 * (384 / 32) * (384 / 32),
            normalized_host_bytes: (39 + 4) * 384 * 384 * 3,
            native_host_bytes: (39 + 4) * 384 * 384 * 3,
        });

        let charges = h3_reference_charges(&references[1]).unwrap();
        // Charged at the retained canvas, not the source one.
        assert_eq!(charges.native_host_bytes, (39 + 4) * 384 * 384 * 3);
        let at_source_dims = (39 + 4) * 32 * 32 * 3;
        assert_eq!(charges.native_host_bytes, at_source_dims * 144);

        // And it validates end to end through the ledger.
        let base = ref2va_prepared_request();
        let condition_visual_rows = references.iter().map(|r| r.visual_rows).sum::<u64>();
        let condition_audio_rows = references.iter().map(|r| r.audio_rows).sum::<u64>();
        let qwen_vision_rows = references.iter().map(|r| r.qwen_vision_rows).sum::<u64>();
        let mut request = H3FactoryPreparedRequestInput {
            references,
            rows: H3FactoryPreparedRowsInput {
                qwen_vision_rows,
                condition_visual_rows,
                condition_audio_rows,
                total_packed_rows: base.rows.qwen_output_text_rows
                    + condition_visual_rows
                    + condition_audio_rows
                    + base.rows.target_video_rows
                    + base.rows.target_audio_rows,
                ..base.rows
            },
            ..base
        };
        request.identity_sha256 = expected_h3_factory_prepared_request_identity(&request);
        let checkpoint = raw_checkpoint();
        let budget = target_budget(&request, &checkpoint);
        check_budget(&budget, &request, &checkpoint).unwrap();
        assert!(budget.reference_decode_phase_host_bytes > (39 + 4) * 384 * 384 * 3);
    }

    /// A rehashed budget must not be able to name its own transient charges.
    #[test]
    fn reference_transient_charges_are_recomputed_not_merely_nonzero() {
        let request = ref2va_prepared_request();
        let checkpoint = raw_checkpoint();
        let base = target_budget(&request, &checkpoint);

        for mutate in [
            (|budget: &mut H3FactoryTargetBudgetInput| {
                budget.reference_decode_staging_host_bytes = 1
            }) as fn(&mut H3FactoryTargetBudgetInput),
            |budget| budget.reference_preprocess_staging_host_bytes = 1,
            |budget| budget.reference_audio_encode_workspace_device_bytes = 1,
            // A doubled value is as wrong as a token one: both disagree with
            // the frozen media facts the term is derived from.
            |budget| budget.reference_decode_staging_host_bytes *= 2,
            |budget| budget.reference_preprocess_staging_host_bytes *= 2,
            |budget| budget.reference_audio_encode_workspace_device_bytes *= 2,
        ] {
            let mut budget = base.clone();
            mutate(&mut budget);
            budget.identity_sha256 = expected_h3_factory_target_budget_identity(&budget);
            let error = check_budget(&budget, &request, &checkpoint)
                .expect_err("a caller-named transient charge must be refused")
                .to_string();
            // The macro sweep names the offending field, which is what makes a
            // production failure locatable.
            assert!(error.contains("reference_"), "{error}");
        }
    }

    #[test]
    fn reference_ledgers_exist_exactly_for_the_task_that_has_references() {
        let fl2va = prepared_request();
        let checkpoint = raw_checkpoint();
        let fl2va_budget = target_budget(&fl2va, &checkpoint);
        check_budget(&fl2va_budget, &fl2va, &checkpoint).unwrap();

        // FL2VA charges nothing for references and keeps condition encode.
        assert_eq!(fl2va_budget.reference_normalized_media_host_bytes, 0);
        assert_eq!(fl2va_budget.reference_decode_phase_host_bytes, 0);
        assert_eq!(
            fl2va_budget.reference_audio_encode_workspace_device_bytes,
            0
        );
        assert!(fl2va_budget.condition_encode_phase_device_bytes > 0);

        // A reference ledger on an FL2VA budget is work the worker will never
        // perform, so it must be refused rather than silently granted.
        for mutate in [
            (|budget: &mut H3FactoryTargetBudgetInput| {
                budget.reference_normalized_media_host_bytes = 4_096
            }) as fn(&mut H3FactoryTargetBudgetInput),
            |budget| budget.reference_decode_staging_host_bytes = 1,
            |budget| budget.reference_preprocess_staging_host_bytes = 1,
            |budget| budget.reference_audio_encode_workspace_device_bytes = 1,
            |budget| budget.reference_decode_phase_host_bytes = 1,
            |budget| budget.reference_visual_encode_phase_device_bytes = 1,
        ] {
            let mut budget = fl2va_budget.clone();
            mutate(&mut budget);
            budget.identity_sha256 = expected_h3_factory_target_budget_identity(&budget);
            assert!(check_budget(&budget, &fl2va, &checkpoint).is_err());
        }

        // And the mirror: a Ref2VA budget may not drop its reference ledgers
        // or resurrect FL2VA's condition encode.
        let ref2va = ref2va_prepared_request();
        let ref2va_budget = target_budget(&ref2va, &checkpoint);
        for mutate in [
            (|budget: &mut H3FactoryTargetBudgetInput| {
                budget.reference_normalized_media_host_bytes = 0
            }) as fn(&mut H3FactoryTargetBudgetInput),
            // Understating the retained media is the failure that would admit
            // a job into a host grant too small to hold its own references.
            |budget| budget.reference_normalized_media_host_bytes -= 4_096,
            |budget| budget.reference_decode_phase_host_bytes = 0,
            |budget| budget.reference_audio_encode_phase_device_bytes = 0,
            |budget| budget.condition_encode_phase_device_bytes = 4_096,
            |budget| {
                budget.load_drop_policy = H3FactoryTargetLoadDropPolicy::LoadQwenEncodeTransferDropQwenLoadVaesEncodeConditionsParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux
            },
        ] {
            let mut budget = ref2va_budget.clone();
            mutate(&mut budget);
            budget.identity_sha256 = expected_h3_factory_target_budget_identity(&budget);
            assert!(check_budget(&budget, &ref2va, &checkpoint).is_err());
        }
    }

    /// Turbo and the reference ledgers are orthogonal, and this pins that on
    /// purpose rather than by accident.
    ///
    /// The task-shaped zero rule says a phase outside a task's order must be
    /// recorded as zero. Turbo is NOT such a phase: it charges into
    /// transformer load and denoise, which both tasks execute identically, and
    /// `REVIEWED_TURBO_TIERS` carries a first-class `ref2v-4step` tier with no
    /// task binding at all. So a Turbo term on a Ref2VA budget is legitimate,
    /// not a crossed contract, and must not be refused the way a stray
    /// condition-encode ledger is. Nothing wires a Ref2VA Turbo adapter yet,
    /// so in practice these terms are zero — allowed-but-zero-until-wired,
    /// which is a different statement from "forbidden" and must stay so.
    #[test]
    fn turbo_terms_are_task_neutral_while_reference_ledgers_are_not() {
        let checkpoint = raw_checkpoint();
        let ref2va = ref2va_prepared_request();
        let budget = target_budget(&ref2va, &checkpoint);

        // Baseline: no adapter is wired, so both terms are zero and the
        // Ref2VA budget validates.
        assert_eq!(budget.turbo_adapter_device_bytes, 0);
        assert_eq!(budget.turbo_adapter_host_staging_bytes, 0);
        check_budget(&budget, &ref2va, &checkpoint).unwrap();

        // Widening the two shared phases by an adapter's declared cost stays
        // arithmetically consistent on a Ref2VA budget exactly as it does on
        // an FL2VA one — the reference ledgers neither absorb nor block it.
        let mut turbocharged = budget.clone();
        turbocharged.turbo_adapter_device_bytes = TURBO_DEVICE_BYTES;
        turbocharged.turbo_adapter_host_staging_bytes = TURBO_HOST_STAGING_BYTES;
        turbocharged.transformer_load_phase_device_bytes += TURBO_DEVICE_BYTES;
        turbocharged.denoise_phase_device_bytes += TURBO_DEVICE_BYTES;
        turbocharged.transformer_load_phase_host_bytes += TURBO_HOST_STAGING_BYTES;
        turbocharged.predicted_device_peak_bytes = turbocharged
            .predicted_device_peak_bytes
            .max(turbocharged.denoise_phase_device_bytes)
            .max(turbocharged.transformer_load_phase_device_bytes);
        turbocharged.predicted_host_increment_bytes = turbocharged
            .predicted_host_increment_bytes
            .max(turbocharged.transformer_load_phase_host_bytes);
        turbocharged.identity_sha256 = expected_h3_factory_target_budget_identity(&turbocharged);
        check_budget(&turbocharged, &ref2va, &checkpoint)
            .expect("a Ref2VA budget must accept the reviewed ref2v Turbo tier's terms");

        // The reference ledgers remain task-shaped underneath it: a Ref2VA
        // budget that also resurrects FL2VA's condition encode is still wrong,
        // Turbo or not.
        let mut crossed = turbocharged;
        crossed.condition_encode_phase_device_bytes = 4_096;
        crossed.identity_sha256 = expected_h3_factory_target_budget_identity(&crossed);
        assert!(check_budget(&crossed, &ref2va, &checkpoint).is_err());
    }

    #[test]
    fn every_reference_ledger_field_participates_in_the_budget_identity() {
        let request = ref2va_prepared_request();
        let checkpoint = raw_checkpoint();
        let base = target_budget(&request, &checkpoint);
        for mutate in [
            (|budget: &mut H3FactoryTargetBudgetInput| {
                budget.reference_normalized_media_host_bytes += 1
            }) as fn(&mut H3FactoryTargetBudgetInput),
            |budget| budget.reference_decode_staging_host_bytes += 1,
            |budget| budget.reference_preprocess_staging_host_bytes += 1,
            |budget| budget.reference_decode_phase_host_bytes += 1,
            |budget| budget.reference_preprocess_phase_host_bytes += 1,
            |budget| budget.reference_visual_encode_phase_host_bytes += 1,
            |budget| budget.reference_audio_encode_phase_host_bytes += 1,
            |budget| budget.reference_audio_encode_workspace_device_bytes += 1,
            |budget| budget.reference_decode_phase_device_bytes += 1,
            |budget| budget.reference_preprocess_phase_device_bytes += 1,
            |budget| budget.reference_visual_encode_phase_device_bytes += 1,
            |budget| budget.reference_audio_encode_phase_device_bytes += 1,
            |budget| budget.reference_media_identity_sha256 = sha('e'),
        ] {
            let mut mutated = base.clone();
            mutate(&mut mutated);
            assert_ne!(
                expected_h3_factory_target_budget_identity(&mutated),
                base.identity_sha256
            );
        }

        // Byte totals collide under a transposed canvas, so the geometry has
        // to reach the BUDGET identity and not only the request's.
        let mut transposed = ref2va_prepared_request();
        let (width, height) = (
            transposed.references[0].native_width,
            transposed.references[0].native_height,
        );
        transposed.references[0].native_width = height;
        transposed.references[0].native_height = width;
        assert_ne!(width, height);
        let transposed_budget = target_budget(&transposed, &raw_checkpoint());
        // Same retained bytes...
        assert_eq!(
            transposed_budget.reference_decode_phase_host_bytes,
            base.reference_decode_phase_host_bytes
        );
        // ...but a different budget identity.
        assert_ne!(transposed_budget.identity_sha256, base.identity_sha256);
    }

    /// FL2VA's frozen plan must be byte-identical across the Ref2VA work.
    ///
    /// Two pinned literals, each with its own coverage:
    ///
    /// * The prepared-attempt digest (dc56e0a5…) is
    ///   `expected_h3_factory_prepared_attempt_identity` over the
    ///   hand-assembled fixture, so it covers the identity FUNCTIONS — the
    ///   request, endpoint, raw-checkpoint, and target-budget hash layouts and
    ///   every field they bind — but NOT the production budget builder
    ///   (`build_canonical_private_fl2va_target_budget`), which needs opened
    ///   weights and is exercised by the budget arithmetic tests instead.
    /// * The backend-plan digest (b9e5fe2e…) is taken from an authority built
    ///   by `FrozenH3FactoryAuthority::new_contract_only` — the same
    ///   constructor server admission calls — so it covers what the production
    ///   builder actually freezes for an FL2VA route: canonical model, task,
    ///   layout, device route, execution fingerprint, conditioner placement,
    ///   block streaming, and the component-authority set (the
    ///   `backend_plan_identity` domain in `minimax_h3/backend.rs`). Captured
    ///   2026-08-20 from that builder over `exact_input()`, which is path- and
    ///   machine-independent by construction.
    ///
    /// Still uncovered without weights on disk: storage-authority resolution
    /// and opened-component digests (pinned functionally by the
    /// `private_opened_evidence` tests).
    ///
    /// If either digest changes, an FL2VA frozen plan changed. That is either
    /// a deliberate FL2VA change — update the literal and say so — or a
    /// Ref2VA change that leaked, which is a bug.
    #[test]
    fn fl2va_frozen_plan_identity_is_pinned() {
        let attempt = prepared_attempt();
        assert_eq!(
            expected_h3_factory_prepared_attempt_identity(&attempt),
            "dc56e0a589b99d03fcb5d8415094690bcd7a85c6572afe15cb729e75a8456fd3"
        );

        // Through the production builder: the frozen authority must preserve
        // the pinned attempt identity unchanged, and its backend plan — the
        // part that hashes canonical model + task — must stay pinned too.
        let authority = FrozenH3FactoryAuthority::new_contract_only(exact_input()).unwrap();
        let (attempt_identity, budget_identity) =
            authority.prepared_target_attempt_identities().unwrap();
        assert_eq!(attempt_identity, attempt.identity_sha256);
        assert_eq!(budget_identity, attempt.target_budget.identity_sha256);
        assert_eq!(
            authority.backend_plan.identity_sha256(),
            "b9e5fe2ebf3a61852578c9d208c2b6e35ba8e7cce95016367a6600b00d82cb4e"
        );
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
            retained_header_host_bytes: FIXTURE_TRANSFORMER_RETAINED_HEADER_BYTES,
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
        target_budget_with_vae_residency(request, checkpoint, 400, 500)
    }

    /// The VAE residency is a parameter so a test can vary only that and read
    /// which phases the resident pair is charged in.
    fn target_budget_with_vae_residency(
        request: &H3FactoryPreparedRequestInput,
        checkpoint: &H3FactoryRawCheckpointInput,
        visual_vae_resident_device_bytes: u64,
        audio_vae_resident_device_bytes: u64,
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
        let retained_vaes = visual_vae_resident_device_bytes + audio_vae_resident_device_bytes;
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
        // The reviewed baseline attempt carries no Turbo adapter, so both
        // additive terms are zero here. `turbo_budget_terms_are_additive_...`
        // flips them against a declared adapter.
        let turbo_adapter_device_bytes = 0;
        let turbo_adapter_device_staging_bytes = 0;
        let turbo_adapter_host_staging_bytes = 0;
        let vae_load_phase_device_bytes =
            fixed_runtime_device_bytes + retained_vaes + 100 + qwen_output_state_device_bytes;
        let qwen_encode_phase_device_bytes = fixed_runtime_device_bytes;
        let qwen_transfer_phase_device_bytes =
            fixed_runtime_device_bytes + qwen_output_state_device_bytes;
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
        // Deliberately the PRODUCTION formula, not a fourth transcription of
        // it: a reference that restates the sum can only ever agree with
        // itself, which is how the Turbo staging term stayed missing from the
        // real builder while every fixture test passed.
        let transformer_load_phase_device_bytes =
            transformer_load_phase_device_bytes(H3TransformerLoadDeviceTerms {
                fixed_runtime_device_bytes,
                fixed_transformer_device_bytes,
                qwen_output_state_device_bytes,
                condition_latent_backing_device_bytes,
                packed_layout_device_bytes,
                packed_video_state_device_bytes,
                packed_audio_state_device_bytes,
                resident_block_device_bytes: 0,
                fixed_transformer_load_device_staging_bytes,
                turbo_adapter_device_bytes,
                turbo_adapter_device_staging_bytes,
            })
            .unwrap();
        let denoise_phase_device_bytes = denoise_phase_device_bytes(H3DenoiseDeviceTerms {
            fixed_runtime_device_bytes,
            fixed_transformer_device_bytes,
            qwen_output_state_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            denoise_tensor_copy_workspace_device_bytes,
            denoise_transient_workspace_device_bytes: denoise_transient_workspace_device_bytes(
                attention_workspace_device_bytes,
                ffn_workspace_device_bytes,
            ),
            denoise_hidden_activation_device_bytes: denoise_hidden_activation_device_bytes(
                request.rows.total_packed_rows,
            )
            .unwrap(),
            resident_block_device_bytes: 0,
            streamed_block_device_overlap_bytes,
            prefetch_device_bytes: 0,
            max_device_weight_staging_bytes,
            turbo_adapter_device_bytes,
        })
        .unwrap();
        let visual_decode_phase_device_bytes = sum(&[
            fixed_runtime_device_bytes,
            retained_vaes,
            100,
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
            .map(|block| block.encoded_host_bytes + 2 * block.max_host_read_staging_bytes)
            .max()
            .unwrap();
        let fixed_transformer_load_host_staging_bytes = 2 * checkpoint
            .fixed_transformer_max_host_read_staging_bytes
            + checkpoint.fixed_transformer_max_device_weight_staging_bytes;
        let qwen_host_load_staging_bytes = 2 * FIXTURE_QWEN_MAX_TENSOR_STAGING_BYTES;
        let qwen_retained_header_host_bytes = FIXTURE_QWEN_RETAINED_HEADER_BYTES;
        let transformer_retained_header_host_bytes = checkpoint.retained_header_host_bytes;
        let vae_retained_config_host_bytes = FIXTURE_VAE_RETAINED_CONFIG_BYTES;
        let qwen_alive_metadata_host_bytes = sum(&[
            qwen_retained_header_host_bytes,
            transformer_retained_header_host_bytes,
            vae_retained_config_host_bytes,
        ]);
        let transformer_alive_metadata_host_bytes = sum(&[
            transformer_retained_header_host_bytes,
            vae_retained_config_host_bytes,
        ]);
        // Anonymous host demand that outlives every phase: the process RSS
        // baseline and the request-owned endpoint buffers.
        let attempt_host_bytes = sum(&[
            100,
            endpoint_encoded_host_bytes,
            normalized_endpoint_host_bytes,
        ]);
        let vae_load_phase_host_bytes =
            attempt_host_bytes + transformer_alive_metadata_host_bytes + 1_000;
        let qwen_encode_phase_host_bytes = sum(&[
            attempt_host_bytes,
            qwen_alive_metadata_host_bytes,
            qwen_host_workspace_bytes,
            qwen_host_load_staging_bytes,
            text_modality_tags_host_bytes,
        ]);
        let qwen_transfer_phase_host_bytes = sum(&[
            attempt_host_bytes,
            qwen_alive_metadata_host_bytes,
            qwen_host_workspace_bytes,
            text_modality_tags_host_bytes,
        ]);
        let condition_encode_phase_host_bytes = sum(&[
            attempt_host_bytes,
            transformer_alive_metadata_host_bytes,
            condition_backing_host_bytes,
            packed_layout_host_bytes,
            packed_layout_construction_staging_host_bytes,
            packed_layout_freeze_staging_host_bytes,
            text_modality_tags_host_bytes,
            noise_cpu_staging_host_bytes,
        ]);
        let noise_allocation_phase_host_bytes = sum(&[
            attempt_host_bytes,
            transformer_alive_metadata_host_bytes,
            condition_backing_host_bytes,
            packed_layout_host_bytes,
            text_modality_tags_host_bytes,
            schedule_host_bytes,
            noise_cpu_staging_host_bytes,
        ]);
        let transformer_load_phase_host_bytes =
            transformer_load_phase_host_bytes(H3TransformerLoadHostTerms {
                attempt_host_bytes,
                transformer_alive_metadata_host_bytes,
                condition_backing_host_bytes,
                packed_layout_host_bytes,
                text_modality_tags_host_bytes,
                schedule_host_bytes,
                fixed_transformer_load_host_staging_bytes,
                turbo_adapter_host_staging_bytes,
            })
            .unwrap();
        let denoise_phase_host_bytes = sum(&[
            attempt_host_bytes,
            transformer_alive_metadata_host_bytes,
            condition_backing_host_bytes,
            packed_layout_host_bytes,
            text_modality_tags_host_bytes,
            schedule_host_bytes,
            max_streamed_block_host_overlap_bytes,
        ]);
        let visual_decode_phase_host_bytes = sum(&[
            attempt_host_bytes,
            vae_retained_config_host_bytes,
            packed_layout_host_bytes,
            1_000,
            5_000,
            1_000,
        ]);
        let audio_decode_phase_host_bytes = sum(&[
            attempt_host_bytes,
            packed_layout_host_bytes,
            5_000,
            1_000,
            waveform_host_bytes,
        ]);
        let waveform_transfer_phase_host_bytes =
            sum(&[attempt_host_bytes, 5_000, 1_000, waveform_host_bytes]);
        let mux_phase_host_bytes = sum(&[
            attempt_host_bytes,
            5_000,
            1_000,
            waveform_host_bytes,
            10_000,
            2_000,
        ]);
        // Ref2VA's four leading phases, written out independently of the code
        // under test so the ledger and its reference stay in lockstep.
        let ref2va = request.task == Task::Ref2va;
        let reference_normalized_media_host_bytes = request
            .references
            .iter()
            .map(|reference| reference.normalized_host_bytes)
            .sum::<u64>();
        // Every native payload is held at once — the orchestrator decodes the
        // whole ordered set before preprocessing any of it.
        let reference_native_media_host_bytes = request
            .references
            .iter()
            .map(|reference| reference.native_host_bytes)
            .sum::<u64>();
        let reference_decode_staging_host_bytes = request
            .references
            .iter()
            .map(|reference| reference.native_host_bytes)
            .max()
            .unwrap_or(0);
        let reference_preprocess_staging_host_bytes = request
            .references
            .iter()
            .map(|reference| reference.normalized_host_bytes)
            .max()
            .unwrap_or(0);
        let reference_audio_encode_workspace_device_bytes = if ref2va {
            audio_decode_workspace_device_bytes
        } else {
            0
        };
        let reference_decode_phase_host_bytes = if ref2va {
            sum(&[
                attempt_host_bytes,
                qwen_alive_metadata_host_bytes,
                reference_native_media_host_bytes,
                reference_decode_staging_host_bytes,
            ])
        } else {
            0
        };
        let reference_preprocess_phase_host_bytes = if ref2va {
            sum(&[
                attempt_host_bytes,
                qwen_alive_metadata_host_bytes,
                reference_native_media_host_bytes,
                reference_normalized_media_host_bytes,
                reference_preprocess_staging_host_bytes,
            ])
        } else {
            0
        };
        let reference_encode_phase_host_bytes = if ref2va {
            sum(&[
                attempt_host_bytes,
                transformer_alive_metadata_host_bytes,
                reference_normalized_media_host_bytes,
                condition_backing_host_bytes,
                packed_layout_host_bytes,
                text_modality_tags_host_bytes,
            ])
        } else {
            0
        };
        let reference_decode_phase_device_bytes = if ref2va {
            fixed_runtime_device_bytes
        } else {
            0
        };
        let reference_preprocess_phase_device_bytes = reference_decode_phase_device_bytes;
        let reference_visual_encode_phase_device_bytes = if ref2va {
            sum(&[
                fixed_runtime_device_bytes,
                retained_vaes,
                qwen_output_state_device_bytes,
                condition_vae_workspace_device_bytes,
                condition_latent_backing_device_bytes,
                packed_layout_device_bytes,
            ])
        } else {
            0
        };
        let reference_audio_encode_phase_device_bytes = if ref2va {
            sum(&[
                fixed_runtime_device_bytes,
                retained_vaes,
                qwen_output_state_device_bytes,
                reference_audio_encode_workspace_device_bytes,
                condition_latent_backing_device_bytes,
                packed_layout_device_bytes,
            ])
        } else {
            0
        };
        // A phase outside this task's order contributes nothing.
        let condition_encode_phase_host_bytes = if ref2va {
            0
        } else {
            condition_encode_phase_host_bytes
        };
        let condition_encode_phase_device_bytes = if ref2va {
            0
        } else {
            condition_encode_phase_device_bytes
        };
        let predicted_host_increment_bytes = *[
            reference_decode_phase_host_bytes,
            reference_preprocess_phase_host_bytes,
            reference_encode_phase_host_bytes,
            vae_load_phase_host_bytes,
            qwen_encode_phase_host_bytes,
            qwen_transfer_phase_host_bytes,
            condition_encode_phase_host_bytes,
            noise_allocation_phase_host_bytes,
            transformer_load_phase_host_bytes,
            denoise_phase_host_bytes,
            visual_decode_phase_host_bytes,
            audio_decode_phase_host_bytes,
            waveform_transfer_phase_host_bytes,
            mux_phase_host_bytes,
        ]
        .iter()
        .max()
        .unwrap();
        let predicted_device_peak_bytes = *[
            predicted_device_peak_bytes,
            reference_decode_phase_device_bytes,
            reference_preprocess_phase_device_bytes,
            reference_visual_encode_phase_device_bytes,
            reference_audio_encode_phase_device_bytes,
            condition_encode_phase_device_bytes,
        ]
        .iter()
        .max()
        .unwrap();
        let mut budget = H3FactoryTargetBudgetInput {
            identity_sha256: String::new(),
            load_drop_policy: if ref2va {
                H3FactoryTargetLoadDropPolicy::DecodeReferencesPreprocessReferencesLoadQwenEncodeVisionTransferDropQwenLoadVaesEncodeVisualReferencesEncodeAudioReferencesParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux
            } else {
                H3FactoryTargetLoadDropPolicy::LoadQwenEncodeTransferDropQwenLoadVaesEncodeConditionsParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux
            },
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
            reference_media_identity_sha256: expected_h3_factory_reference_media_identity(
                &request.references,
            ),
            reference_normalized_media_host_bytes,
            reference_decode_staging_host_bytes,
            reference_preprocess_staging_host_bytes,
            reference_decode_phase_host_bytes,
            reference_preprocess_phase_host_bytes,
            reference_visual_encode_phase_host_bytes: reference_encode_phase_host_bytes,
            reference_audio_encode_phase_host_bytes: reference_encode_phase_host_bytes,
            reference_audio_encode_workspace_device_bytes,
            reference_decode_phase_device_bytes,
            reference_preprocess_phase_device_bytes,
            reference_visual_encode_phase_device_bytes,
            reference_audio_encode_phase_device_bytes,
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
            qwen_host_load_staging_bytes,
            qwen_retained_header_host_bytes,
            transformer_retained_header_host_bytes,
            vae_retained_config_host_bytes,
            vae_load_phase_host_bytes,
            qwen_encode_phase_host_bytes,
            qwen_transfer_phase_host_bytes,
            condition_encode_phase_host_bytes,
            noise_allocation_phase_host_bytes,
            transformer_load_phase_host_bytes,
            denoise_phase_host_bytes,
            visual_decode_phase_host_bytes,
            audio_decode_phase_host_bytes,
            waveform_transfer_phase_host_bytes,
            mux_phase_host_bytes,
            predicted_host_increment_bytes,
            fixed_runtime_device_bytes,
            fixed_transformer_device_bytes,
            visual_vae_resident_device_bytes,
            audio_vae_resident_device_bytes,
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
            turbo_adapter_device_bytes,
            turbo_adapter_device_staging_bytes,
            turbo_adapter_host_staging_bytes,
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
            compute_capability: Some((8, 9)),
            execution_fingerprint: sha('a'),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 2048,
            qwen_host_resident_parameter_bytes: 2048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1024,
            qwen_maximum_tensor_staging_bytes: FIXTURE_QWEN_MAX_TENSOR_STAGING_BYTES,
            qwen_retained_raw_header_bytes: FIXTURE_QWEN_RETAINED_HEADER_BYTES,
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
                turbo_adapter: None,
            },
            prepared_attempt: Some(prepared_attempt),
            execution_budget_echo: Some(execution_budget_echo),
            components: components(),
        }
    }

    /// A Metal route's attention tuple, which the factory requires to agree
    /// with the absent compute capability.
    fn metal_attention() -> H3FactoryAttentionInput {
        let runtime_backend = H3AttentionBackend::MetalChunkedDenseMath;
        let kernel = H3AttentionKernel::CandleDenseChunkedF32V011;
        let activation = H3AttentionActivation::MetalCorrectnessOnly;
        let device = H3AttentionDevice::Metal;
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
            generic_backend: AttentionBackend::Math,
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

    /// Move an otherwise-exact input onto the Metal route: no compute
    /// capability, plus the Metal attention tuple the factory demands
    /// alongside it. Everything else -- budgets included -- is left exactly as
    /// the caller built it, so a test can isolate one disagreement.
    fn onto_metal(mut input: H3FactoryAuthorityInput) -> H3FactoryAuthorityInput {
        let attention = metal_attention();
        input.compute_capability = None;
        input.attention_backend = attention.generic_backend;
        input.attention_kernel_identity = attention.qualification_kernel_identity.clone();
        input.attention_runtime = Some(attention);
        input
    }

    /// A Metal route carries no compute capability, so a CUDA-NAMED
    /// conditioner placement is a contradiction rather than a translation
    /// problem: it would freeze as `CudaResident`, hash into the authority
    /// identity as `qwen-cuda`, and then pass a device-id-only check because
    /// the Metal device id is the one it names. Refuse it at construction.
    #[test]
    fn a_metal_route_refuses_the_cuda_assigned_conditioner_placement() {
        // `exact_cuda_input` is the fully consistent device-resident input, so
        // the ONLY thing wrong here is the route it is placed on.
        let input = onto_metal(exact_cuda_input());
        assert_eq!(
            input.conditioner_placement,
            H3FactoryConditionerPlacement::AssignedCudaThenDrop
        );
        let error = FrozenH3FactoryAuthority::new_contract_only(input)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Metal"), "{error}");
        assert!(error.contains("conditioner"), "{error}");

        // And the same input on its own CUDA route is accepted, which is what
        // proves the refusal is about the contradiction and not the fixture.
        FrozenH3FactoryAuthority::new_contract_only(exact_cuda_input()).unwrap();
    }

    /// The refusal is specific to the device-resident pair. A Metal route with
    /// a host-placed conditioner is not contradictory and must keep working,
    /// or the guard would simply ban Metal instead of banning the mismatch.
    #[test]
    fn a_metal_route_still_accepts_the_host_conditioner_placement() {
        let input = onto_metal(exact_input());
        assert_eq!(
            input.conditioner_placement,
            H3FactoryConditionerPlacement::HostCpuThenDrop
        );
        let authority = FrozenH3FactoryAuthority::new_contract_only(input).unwrap();
        assert_eq!(authority.compute_capability(), None);
    }

    fn exact_cuda_input() -> H3FactoryAuthorityInput {
        let mut input = exact_input();
        input.conditioner_placement = H3FactoryConditionerPlacement::AssignedCudaThenDrop;
        input.qwen_device_resident_parameter_bytes = input.qwen_parameter_bytes;

        let budget = &mut input.prepared_attempt.as_mut().unwrap().target_budget;
        budget.qwen_host_activation_bytes = 0;
        budget.qwen_host_output_state_bytes = 0;
        budget.qwen_host_workspace_bytes = budget.qwen_host_parameter_bytes;
        budget.qwen_device_parameter_bytes = input.qwen_device_resident_parameter_bytes;
        budget.qwen_activation_device_bytes = input.qwen_activation_workspace_bytes;
        budget.qwen_output_transfer_device_bytes = 0;
        budget.qwen_encode_phase_device_bytes = sum(&[
            budget.fixed_runtime_device_bytes,
            budget.qwen_device_parameter_bytes,
            budget.qwen_activation_device_bytes,
            budget.qwen_output_state_device_bytes,
        ]);
        budget.qwen_transfer_phase_device_bytes = 0;
        // Only the Qwen-bearing host phases move: the CUDA route keeps its
        // packed weights on the host but allocates no host activation or
        // output state, so the per-phase max must be re-taken rather than
        // patched by a delta.
        let attempt_host_bytes = sum(&[
            budget.fixed_runtime_host_bytes,
            budget.endpoint_encoded_host_bytes,
            budget.normalized_endpoint_host_bytes,
        ]);
        let qwen_alive_metadata_host_bytes = sum(&[
            budget.qwen_retained_header_host_bytes,
            budget.transformer_retained_header_host_bytes,
            budget.vae_retained_config_host_bytes,
        ]);
        budget.qwen_encode_phase_host_bytes = sum(&[
            attempt_host_bytes,
            qwen_alive_metadata_host_bytes,
            budget.qwen_host_workspace_bytes,
            budget.qwen_host_load_staging_bytes,
            budget.text_modality_tags_host_bytes,
        ]);
        budget.qwen_transfer_phase_host_bytes = sum(&[
            attempt_host_bytes,
            qwen_alive_metadata_host_bytes,
            budget.qwen_host_workspace_bytes,
            budget.text_modality_tags_host_bytes,
        ]);
        budget.predicted_host_increment_bytes = [
            budget.vae_load_phase_host_bytes,
            budget.qwen_encode_phase_host_bytes,
            budget.qwen_transfer_phase_host_bytes,
            budget.condition_encode_phase_host_bytes,
            budget.noise_allocation_phase_host_bytes,
            budget.transformer_load_phase_host_bytes,
            budget.denoise_phase_host_bytes,
            budget.visual_decode_phase_host_bytes,
            budget.audio_decode_phase_host_bytes,
            budget.waveform_transfer_phase_host_bytes,
            budget.mux_phase_host_bytes,
        ]
        .into_iter()
        .max()
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

    /// Exact resident and staging cost of a published Turbo adapter, from the
    /// mold-candle derivation: 1,956,118,528 BF16 matrix bytes (the published
    /// payload minus its 208 F32 alphas), and twice the widest single matrix
    /// (`lora_B` at `[21504, 384]`) staged on the host.
    const TURBO_DEVICE_BYTES: u64 = 1_956_118_528;
    /// The widest module (fused `attn.qkv_proj`): its transposed copies live
    /// beside its originals during the upload.
    const TURBO_DEVICE_STAGING_BYTES: u64 = 20_643_840;
    const TURBO_HOST_STAGING_BYTES: u64 = 33_030_144;
    /// The 4-step 768p tier, whose reviewed grid is 5 points at shift 6.
    const TURBO_4STEP_TIER: &str = "minimax-h3.turbo-lora.fl2v-4step-768p-v1.0.comfyui-bf16.v1";
    /// The 8-step tier, whose reviewed grid is 9 points at shift 12.
    const TURBO_8STEP_TIER: &str = "minimax-h3.turbo-lora.fl2v-8step-v1.0.comfyui-bf16.v1";

    fn turbo_authority_for(tier_stable_id: &str) -> H3FactoryTurboAdapterAuthority {
        H3FactoryTurboAdapterAuthority::for_reviewed_tier(
            tier_stable_id,
            &sha('7'),
            &sha('8'),
            TURBO_DEVICE_BYTES,
            TURBO_DEVICE_STAGING_BYTES,
            TURBO_HOST_STAGING_BYTES,
        )
        .unwrap()
    }

    /// Widen a baseline budget by exactly one adapter's declared cost.
    fn apply_turbo_budget(
        input: &mut H3FactoryAuthorityInput,
        turbo: &H3FactoryTurboAdapterAuthority,
    ) {
        let prepared = input.prepared_attempt.as_mut().unwrap();
        let budget = &mut prepared.target_budget;
        budget.turbo_adapter_device_bytes = turbo.resident_device_bytes();
        budget.turbo_adapter_device_staging_bytes = turbo.device_staging_peak_bytes();
        budget.turbo_adapter_host_staging_bytes = turbo.host_staging_peak_bytes();
        budget.transformer_load_phase_device_bytes +=
            turbo.resident_device_bytes() + turbo.device_staging_peak_bytes();
        // Only the residents survive into the denoise; the transposes are gone.
        budget.denoise_phase_device_bytes += turbo.resident_device_bytes();
        budget.transformer_load_phase_host_bytes += turbo.host_staging_peak_bytes();
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
            0,
        ]
        .into_iter()
        .max()
        .unwrap();
        budget.predicted_host_increment_bytes = [
            budget.vae_load_phase_host_bytes,
            budget.qwen_encode_phase_host_bytes,
            budget.qwen_transfer_phase_host_bytes,
            budget.condition_encode_phase_host_bytes,
            budget.noise_allocation_phase_host_bytes,
            budget.transformer_load_phase_host_bytes,
            budget.denoise_phase_host_bytes,
            budget.visual_decode_phase_host_bytes,
            budget.audio_decode_phase_host_bytes,
            budget.waveform_transfer_phase_host_bytes,
            budget.mux_phase_host_bytes,
        ]
        .into_iter()
        .max()
        .unwrap();
        budget.identity_sha256 = expected_h3_factory_target_budget_identity(budget);
        prepared.identity_sha256 = expected_h3_factory_prepared_attempt_identity(prepared);
        let echo = H3FactoryExecutionBudgetEchoInput {
            prepared_attempt_identity_sha256: prepared.identity_sha256.clone(),
            device_peak_bytes: prepared.target_budget.predicted_device_peak_bytes,
            host_increment_bytes: prepared.target_budget.predicted_host_increment_bytes,
        };
        input.execution_budget_echo = Some(echo);
    }

    fn with_turbo_adapter(
        input: &mut H3FactoryAuthorityInput,
        adapter: Option<H3FactoryTurboAdapterAuthority>,
    ) {
        input.quantization = H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
            transformer_policy_sha256: sha('d'),
            qwen_policy_sha256: sha('c'),
            pruned_adaln_table_sha256: sha('e'),
            turbo_adapter: adapter,
        };
    }

    /// Pin which phases each Turbo term reaches, through the production
    /// formulas the builder and the validator both call.
    ///
    /// The bug this replaces: `turbo_adapter_device_staging_bytes` was added to
    /// the validator's transformer-load sum and to this file's reference sum,
    /// but not to the evidence builder's — three transcriptions of one formula,
    /// two of which agreed. Every fixture test passed and the first real
    /// admission failed with a 20,643,840-byte gap. There is now one formula;
    /// this test pins its Turbo contribution per phase.
    #[test]
    fn turbo_terms_reach_exactly_the_phases_that_hold_them() {
        const RESIDENT: u64 = 1_956_118_528;
        const DEVICE_STAGING: u64 = 20_643_840;
        const HOST_STAGING: u64 = 33_030_144;

        let device_terms = |turbo_resident: u64, turbo_staging: u64| H3TransformerLoadDeviceTerms {
            fixed_runtime_device_bytes: 100,
            fixed_transformer_device_bytes: 200,
            qwen_output_state_device_bytes: 300,
            condition_latent_backing_device_bytes: 400,
            packed_layout_device_bytes: 500,
            packed_video_state_device_bytes: 600,
            packed_audio_state_device_bytes: 700,
            resident_block_device_bytes: 800,
            fixed_transformer_load_device_staging_bytes: 900,
            turbo_adapter_device_bytes: turbo_resident,
            turbo_adapter_device_staging_bytes: turbo_staging,
        };
        let baseline = transformer_load_phase_device_bytes(device_terms(0, 0)).unwrap();
        // The transformer load pays BOTH: the residents it uploads and the
        // transposed copies live beside their originals while it does.
        assert_eq!(
            transformer_load_phase_device_bytes(device_terms(RESIDENT, DEVICE_STAGING)).unwrap(),
            baseline + RESIDENT + DEVICE_STAGING
        );

        let denoise_terms = |turbo_resident: u64| H3DenoiseDeviceTerms {
            fixed_runtime_device_bytes: 100,
            fixed_transformer_device_bytes: 200,
            qwen_output_state_device_bytes: 300,
            condition_latent_backing_device_bytes: 400,
            packed_layout_device_bytes: 500,
            packed_video_state_device_bytes: 600,
            packed_audio_state_device_bytes: 700,
            denoise_tensor_copy_workspace_device_bytes: 800,
            denoise_transient_workspace_device_bytes: 900,
            denoise_hidden_activation_device_bytes: 1_000,
            resident_block_device_bytes: 1_100,
            streamed_block_device_overlap_bytes: 1_200,
            prefetch_device_bytes: 1_300,
            max_device_weight_staging_bytes: 1_400,
            turbo_adapter_device_bytes: turbo_resident,
        };
        let denoise_baseline = denoise_phase_device_bytes(denoise_terms(0)).unwrap();
        // The denoise pays only the residents; the transposes are long gone,
        // and `H3DenoiseDeviceTerms` has no field to charge them with.
        assert_eq!(
            denoise_phase_device_bytes(denoise_terms(RESIDENT)).unwrap(),
            denoise_baseline + RESIDENT
        );

        let host_terms = |turbo_host: u64| H3TransformerLoadHostTerms {
            attempt_host_bytes: 100,
            transformer_alive_metadata_host_bytes: 200,
            condition_backing_host_bytes: 300,
            packed_layout_host_bytes: 400,
            text_modality_tags_host_bytes: 500,
            schedule_host_bytes: 600,
            fixed_transformer_load_host_staging_bytes: 700,
            turbo_adapter_host_staging_bytes: turbo_host,
        };
        let host_baseline = transformer_load_phase_host_bytes(host_terms(0)).unwrap();
        assert_eq!(
            transformer_load_phase_host_bytes(host_terms(HOST_STAGING)).unwrap(),
            host_baseline + HOST_STAGING
        );

        // Every phase still refuses to wrap rather than silently truncating.
        assert!(transformer_load_phase_device_bytes(device_terms(u64::MAX, u64::MAX)).is_err());
        assert!(denoise_phase_device_bytes(denoise_terms(u64::MAX)).is_err());
        assert!(transformer_load_phase_host_bytes(host_terms(u64::MAX)).is_err());
    }

    /// Media facts keep the request's full reviewed identity, so the pairing
    /// between that identity and the frozen authority's adapter tier is the
    /// A seam refusal must name what moved. The message is the only evidence
    /// an operator gets when an admitted job is refused at construction, and
    /// a device-ordinal drift, an offload drift, and a model-identity drift
    /// are three different bugs.
    #[test]
    fn seam_refusals_name_every_field_that_drifted() {
        let frozen = FrozenH3FactoryAuthority::new_contract_only(exact_input()).unwrap();
        assert!(frozen
            .validate_engine_seam(
                contract::FL2VA_COMFY,
                frozen.device_ordinal,
                frozen.block_offload
            )
            .is_ok());

        let message = frozen
            .validate_engine_seam(
                contract::FL2VA_COMFY,
                frozen.device_ordinal + 3,
                !frozen.block_offload,
            )
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("authority changed before construction"),
            "{message}"
        );
        assert!(
            message.contains(&format!(
                "device ordinal {} vs frozen {}",
                frozen.device_ordinal + 3,
                frozen.device_ordinal
            )),
            "{message}"
        );
        assert!(
            message.contains(&format!(
                "block offload {} vs frozen {}",
                !frozen.block_offload, frozen.block_offload
            )),
            "{message}"
        );
        assert!(
            !message.contains("task"),
            "an agreeing field is not named: {message}"
        );

        let message = frozen
            .validate_engine_seam(
                contract::REF2VA_COMFY,
                frozen.device_ordinal,
                frozen.block_offload,
            )
            .unwrap_err()
            .to_string();
        assert!(message.contains("task"), "{message}");
        assert!(message.contains("is not the frozen partition"), "{message}");
    }

    /// gate that stops a Turbo tag rendering without its adapter (or with the
    /// wrong tier) and a base render claiming a Turbo identity.
    #[test]
    fn media_model_pairing_requires_the_exact_frozen_turbo_tier() {
        let baseline = FrozenH3FactoryAuthority::new_contract_only(exact_input()).unwrap();
        assert!(media_model_matches_h3_authority(
            contract::FL2VA_COMFY,
            &baseline
        ));
        for mismatched in [
            contract::FL2VA_COMFY_TURBO_8STEP,
            contract::FL2VA_COMFY_TURBO_4STEP_768P,
            contract::REF2VA_COMFY,
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-2step",
        ] {
            assert!(
                !media_model_matches_h3_authority(mismatched, &baseline),
                "{mismatched}"
            );
        }

        let turbo = turbo_authority_for(TURBO_4STEP_TIER);
        let mut input = exact_input();
        apply_turbo_budget(&mut input, &turbo);
        with_turbo_adapter(&mut input, Some(turbo));
        let frozen = FrozenH3FactoryAuthority::new_contract_only(input).unwrap();
        assert!(media_model_matches_h3_authority(
            contract::FL2VA_COMFY_TURBO_4STEP_768P,
            &frozen
        ));
        assert!(frozen
            .validate_engine_seam(contract::FL2VA_COMFY_TURBO_4STEP_768P, 0, true)
            .is_ok());
        assert!(frozen
            .validate_engine_seam(contract::FL2VA_COMFY, 0, true)
            .is_ok());
        assert!(frozen
            .validate_engine_seam(contract::FL2VA_COMFY_TURBO_8STEP, 0, true)
            .is_err());
        assert!(!media_model_matches_h3_authority(
            contract::FL2VA_COMFY_TURBO_8STEP,
            &frozen
        ));
        // A base identity over a frozen adapter is the capture-scope UAT
        // env-override shape and stays valid only under that feature.
        assert_eq!(
            media_model_matches_h3_authority(contract::FL2VA_COMFY, &frozen),
            cfg!(feature = "h3-private-uat")
        );
    }

    /// A budget whose phase fields were produced by the production formulas is
    /// accepted by the production validator, with and without a Turbo adapter.
    ///
    /// This is the round trip the fixture tests could not make before: the
    /// reference sums are now the same functions the evidence builder calls,
    /// so a term present in one side and missing in the other cannot pass.
    #[test]
    fn a_turbo_budget_round_trips_through_the_production_phase_formulas() {
        let turbo = turbo_authority_for(TURBO_4STEP_TIER);
        let mut input = exact_input();
        apply_turbo_budget(&mut input, &turbo);
        with_turbo_adapter(&mut input, Some(turbo.clone()));
        let frozen = FrozenH3FactoryAuthority::new_contract_only(input).unwrap();
        let budget = &frozen.prepared_attempt.as_ref().unwrap().target_budget;

        // The two phases that hold the adapter agree with the shared formulas
        // when they are re-derived from the frozen budget's own fields.
        assert_eq!(budget.turbo_adapter_device_bytes, TURBO_DEVICE_BYTES);
        assert_eq!(
            budget.turbo_adapter_device_staging_bytes,
            TURBO_DEVICE_STAGING_BYTES
        );
        assert_eq!(
            budget.turbo_adapter_host_staging_bytes,
            TURBO_HOST_STAGING_BYTES
        );
        // Transformer load carries resident + staging more than denoise does,
        // once the phases' non-Turbo differences are removed.
        let baseline = exact_input().prepared_attempt.unwrap().target_budget;
        assert_eq!(
            budget.transformer_load_phase_device_bytes
                - baseline.transformer_load_phase_device_bytes,
            TURBO_DEVICE_BYTES + TURBO_DEVICE_STAGING_BYTES
        );
        assert_eq!(
            budget.denoise_phase_device_bytes - baseline.denoise_phase_device_bytes,
            TURBO_DEVICE_BYTES
        );
        assert_eq!(
            budget.transformer_load_phase_host_bytes - baseline.transformer_load_phase_host_bytes,
            TURBO_HOST_STAGING_BYTES
        );
    }

    #[test]
    fn turbo_budget_terms_are_additive_and_bound_to_the_declaring_authority() {
        let grid_points = exact_input()
            .prepared_attempt
            .as_ref()
            .unwrap()
            .request
            .grid_points;
        // The fixture already renders on a reviewed 4-step Turbo grid.
        assert_eq!(grid_points, 5);
        let turbo = turbo_authority_for(TURBO_4STEP_TIER);
        assert_eq!(turbo.grid_points(), grid_points);

        // A baseline attempt charges nothing and still validates.
        let baseline = FrozenH3FactoryAuthority::new_contract_only(exact_input()).unwrap();
        assert_eq!(baseline.quantization().turbo_adapter_device_bytes(), 0);
        assert_eq!(baseline.quantization().turbo_adapter(), None);
        let baseline_peak = exact_input()
            .prepared_attempt
            .unwrap()
            .target_budget
            .predicted_device_peak_bytes;

        // Declaring the adapter and widening the budget by exactly its cost is
        // admissible, and the device peak grows by exactly that cost.
        let mut input = exact_input();
        apply_turbo_budget(&mut input, &turbo);
        with_turbo_adapter(&mut input, Some(turbo.clone()));
        let frozen = FrozenH3FactoryAuthority::new_contract_only(input).unwrap();
        assert_eq!(
            frozen.quantization().turbo_adapter_device_bytes(),
            TURBO_DEVICE_BYTES
        );
        assert_eq!(
            frozen.quantization().turbo_adapter_host_staging_bytes(),
            TURBO_HOST_STAGING_BYTES
        );
        assert_eq!(
            frozen
                .prepared_attempt
                .as_ref()
                .unwrap()
                .target_budget
                .predicted_device_peak_bytes,
            baseline_peak + TURBO_DEVICE_BYTES
        );
        // A distilled tier also changes the integrator and the shift.
        assert_eq!(
            frozen.quantization().sampler_kind(),
            H3SamplerKind::ComfyEuler
        );
        assert_eq!(frozen.quantization().video_shift(), 6.0);
        // ...and the identity moves with it.
        assert_ne!(frozen.identity_sha256(), baseline.identity_sha256());
    }

    #[test]
    fn a_turbo_budget_term_without_a_declaring_authority_is_refused() {
        let turbo = turbo_authority_for(TURBO_4STEP_TIER);

        // Budget charges the adapter, authority declares none.
        let mut input = exact_input();
        apply_turbo_budget(&mut input, &turbo);
        let error = FrozenH3FactoryAuthority::new_contract_only(input)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Turbo adapter bytes disagree"), "{error}");

        // Authority declares the adapter, budget charges nothing.
        let mut input = exact_input();
        with_turbo_adapter(&mut input, Some(turbo.clone()));
        let error = FrozenH3FactoryAuthority::new_contract_only(input)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Turbo adapter bytes disagree"), "{error}");

        // Half-charging is caught even earlier, by the phase arithmetic: the
        // host phase still carries the staging bytes the term now denies.
        let mut input = exact_input();
        apply_turbo_budget(&mut input, &turbo);
        {
            let prepared = input.prepared_attempt.as_mut().unwrap();
            prepared.target_budget.turbo_adapter_host_staging_bytes = 0;
            prepared.target_budget.identity_sha256 =
                expected_h3_factory_target_budget_identity(&prepared.target_budget);
            prepared.identity_sha256 = expected_h3_factory_prepared_attempt_identity(prepared);
            input.execution_budget_echo = Some(H3FactoryExecutionBudgetEchoInput {
                prepared_attempt_identity_sha256: prepared.identity_sha256.clone(),
                device_peak_bytes: prepared.target_budget.predicted_device_peak_bytes,
                host_increment_bytes: prepared.target_budget.predicted_host_increment_bytes,
            });
        }
        with_turbo_adapter(&mut input, Some(turbo));
        let error = FrozenH3FactoryAuthority::new_contract_only(input)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("target budget is internally inconsistent"),
            "{error}"
        );
    }

    #[test]
    fn a_turbo_attempt_must_use_its_tier_reviewed_step_count() {
        let mut input = exact_input();
        // The 8-step tier is 9 grid points; this fixture renders 5.
        // The 8-step tier is reviewed for 9 grid points; this fixture renders 5.
        let turbo = turbo_authority_for(TURBO_8STEP_TIER);
        apply_turbo_budget(&mut input, &turbo);
        with_turbo_adapter(&mut input, Some(turbo));
        let error = FrozenH3FactoryAuthority::new_contract_only(input)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Turbo tier is reviewed for 9"), "{error}");
    }

    #[test]
    fn a_turbo_authority_can_only_be_minted_for_a_reviewed_tier() {
        // The distillation triple is not an input at all: there is no way to
        // pair a genuine adapter with an arbitrary step count, an arbitrary
        // shift, or the RES-multistep integrator, because the constructor reads
        // all three from the reviewed table.
        let four_step = turbo_authority_for(TURBO_4STEP_TIER);
        assert_eq!(four_step.grid_points(), 5);
        assert_eq!(four_step.video_shift(), 6.0);
        assert_eq!(four_step.sampler_kind(), H3FactorySamplerKind::ComfyEuler);
        let eight_step = turbo_authority_for(TURBO_8STEP_TIER);
        assert_eq!(eight_step.grid_points(), 9);
        assert_eq!(eight_step.video_shift(), 12.0);
        assert_eq!(eight_step.sampler_kind(), H3FactorySamplerKind::ComfyEuler);

        let cases: [(&str, &str, &str, u64, u64, u64, &str); 6] = [
            (
                "minimax-h3.turbo-lora.fl2v-2step.v1",
                &"7".repeat(64),
                &"8".repeat(64),
                TURBO_DEVICE_BYTES,
                TURBO_DEVICE_STAGING_BYTES,
                TURBO_HOST_STAGING_BYTES,
                "not a reviewed tier",
            ),
            (
                TURBO_4STEP_TIER,
                "nope",
                &"8".repeat(64),
                TURBO_DEVICE_BYTES,
                TURBO_DEVICE_STAGING_BYTES,
                TURBO_HOST_STAGING_BYTES,
                "H3 Turbo adapter identity",
            ),
            (
                TURBO_4STEP_TIER,
                &"7".repeat(64),
                "nope",
                TURBO_DEVICE_BYTES,
                TURBO_DEVICE_STAGING_BYTES,
                TURBO_HOST_STAGING_BYTES,
                "H3 Turbo adapter content",
            ),
            (
                TURBO_4STEP_TIER,
                &"7".repeat(64),
                &"8".repeat(64),
                0,
                TURBO_DEVICE_STAGING_BYTES,
                TURBO_HOST_STAGING_BYTES,
                "nonzero resident",
            ),
            (
                TURBO_4STEP_TIER,
                &"7".repeat(64),
                &"8".repeat(64),
                TURBO_DEVICE_BYTES,
                0,
                TURBO_HOST_STAGING_BYTES,
                "nonzero resident",
            ),
            (
                TURBO_4STEP_TIER,
                &"7".repeat(64),
                &"8".repeat(64),
                TURBO_DEVICE_STAGING_BYTES,
                TURBO_DEVICE_BYTES,
                TURBO_HOST_STAGING_BYTES,
                "exceeds the whole resident adapter",
            ),
        ];
        for (tier, identity, content, resident, device_staging, host_staging, expected) in cases {
            let error = H3FactoryTurboAdapterAuthority::for_reviewed_tier(
                tier,
                identity,
                content,
                resident,
                device_staging,
                host_staging,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected), "{expected}: {error}");
        }
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
        let dispatch = authority.validate_for_dispatch(
            contract::FL2VA_COMFY,
            contract::FAMILY,
            0,
            true,
            AttentionBackend::Flash,
            AttentionChunkPolicy::Off,
        );
        #[cfg(feature = "h3")]
        assert!(dispatch.is_ok());
        #[cfg(not(feature = "h3"))]
        assert!(dispatch.is_err());
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
                input.compute_capability = Some((9, 0));
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
            qwen_host_load_staging_bytes,
            qwen_retained_header_host_bytes,
            transformer_retained_header_host_bytes,
            vae_retained_config_host_bytes,
            vae_load_phase_host_bytes,
            qwen_encode_phase_host_bytes,
            qwen_transfer_phase_host_bytes,
            condition_encode_phase_host_bytes,
            noise_allocation_phase_host_bytes,
            transformer_load_phase_host_bytes,
            denoise_phase_host_bytes,
            visual_decode_phase_host_bytes,
            audio_decode_phase_host_bytes,
            waveform_transfer_phase_host_bytes,
            mux_phase_host_bytes,
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
            turbo_adapter_device_bytes,
            turbo_adapter_device_staging_bytes,
            turbo_adapter_host_staging_bytes,
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
    fn host_demand_is_a_per_phase_max_of_anonymous_bytes_only() {
        let request = prepared_request();
        let checkpoint = raw_checkpoint();
        let budget = target_budget(&request, &checkpoint);
        let phases = [
            budget.vae_load_phase_host_bytes,
            budget.qwen_encode_phase_host_bytes,
            budget.qwen_transfer_phase_host_bytes,
            budget.condition_encode_phase_host_bytes,
            budget.noise_allocation_phase_host_bytes,
            budget.transformer_load_phase_host_bytes,
            budget.denoise_phase_host_bytes,
            budget.visual_decode_phase_host_bytes,
            budget.audio_decode_phase_host_bytes,
            budget.waveform_transfer_phase_host_bytes,
            budget.mux_phase_host_bytes,
        ];

        // The prediction is the peak phase, never everything ever allocated.
        assert_eq!(
            budget.predicted_host_increment_bytes,
            phases.into_iter().max().unwrap()
        );
        assert!(budget.predicted_host_increment_bytes < phases.into_iter().sum::<u64>());

        // The peak phase is named term by term, so the two classes that must
        // never be anonymous demand — whole-artifact file bytes and the
        // file-backed VAE mapping — are visibly absent from it.
        let attempt_host = budget.fixed_runtime_host_bytes
            + budget.endpoint_encoded_host_bytes
            + budget.normalized_endpoint_host_bytes;
        // Retained opened-component metadata is anonymous too, and lives for
        // exactly as long as its own authority.
        let qwen_alive_metadata = budget.qwen_retained_header_host_bytes
            + budget.transformer_retained_header_host_bytes
            + budget.vae_retained_config_host_bytes;
        let transformer_alive_metadata =
            budget.transformer_retained_header_host_bytes + budget.vae_retained_config_host_bytes;
        assert_eq!(
            budget.qwen_encode_phase_host_bytes,
            attempt_host
                + qwen_alive_metadata
                + budget.qwen_host_workspace_bytes
                + budget.qwen_host_load_staging_bytes
                + budget.text_modality_tags_host_bytes
        );
        // The Qwen header goes when the conditioner does; the VAE configs
        // outlive the transformer because the reload authority still holds one
        // copy until visual decode.
        assert_eq!(
            budget.visual_decode_phase_host_bytes
                - budget.vae_retained_config_host_bytes
                - budget.packed_layout_host_bytes
                - budget.vae_peak_host_io_buffer_bytes
                - budget.encoded_video_host_bytes_bound
                - budget.thumbnail_host_bytes_bound,
            attempt_host
        );
        assert_eq!(budget.mux_phase_host_bytes.min(attempt_host), attempt_host);

        // The Qwen is dropped before conditions are encoded, so its ~20 GB of
        // packed CPU parameters belong to no later phase. Denoise additionally
        // holds exactly one live packed block, never the whole checkpoint.
        assert_eq!(
            budget.denoise_phase_host_bytes,
            attempt_host
                + transformer_alive_metadata
                + budget.condition_backing_host_bytes
                + budget.packed_layout_host_bytes
                + budget.text_modality_tags_host_bytes
                + budget.schedule_host_bytes
                + budget.max_streamed_block_host_overlap_bytes
        );

        // A larger file-backed VAE mapping is reclaimable page cache that
        // `MemAvailable` already counts as free, so it must not change the
        // prediction — and the validator must still accept the budget.
        let mut mapped = exact_input();
        mapped
            .prepared_attempt
            .as_mut()
            .unwrap()
            .target_budget
            .vae_peak_host_mapped_file_bytes += 4 * 1024 * 1024 * 1024;
        refresh_nested_identities(&mut mapped);
        assert!(FrozenH3FactoryAuthority::new_contract_only(mapped).is_ok());
    }

    #[test]
    fn parked_vaes_are_charged_around_denoise_but_never_across_it() {
        let request = prepared_request();
        let checkpoint = raw_checkpoint();
        let base = target_budget_with_vae_residency(&request, &checkpoint, 400, 500);
        let heavier = target_budget_with_vae_residency(&request, &checkpoint, 401, 501);
        let delta =
            heavier.attempt_resident_vae_device_bytes - base.attempt_resident_vae_device_bytes;
        assert_eq!(delta, 2);

        // Qwen runs before either VAE is constructed. After condition encode,
        // both VAEs park until the transformer is dropped. Heavier VAE weights
        // therefore cannot move any of these disjoint phases.
        for (phase, base_bytes, heavier_bytes) in [
            (
                "Qwen encode",
                base.qwen_encode_phase_device_bytes,
                heavier.qwen_encode_phase_device_bytes,
            ),
            (
                "Qwen transfer",
                base.qwen_transfer_phase_device_bytes,
                heavier.qwen_transfer_phase_device_bytes,
            ),
            (
                "noise allocation",
                base.noise_allocation_phase_device_bytes,
                heavier.noise_allocation_phase_device_bytes,
            ),
            (
                "transformer load",
                base.transformer_load_phase_device_bytes,
                heavier.transformer_load_phase_device_bytes,
            ),
            (
                "denoise",
                base.denoise_phase_device_bytes,
                heavier.denoise_phase_device_bytes,
            ),
        ] {
            assert_eq!(
                base_bytes, heavier_bytes,
                "{phase} still overlaps the VAE pair"
            );
        }

        for (phase, base_bytes, heavier_bytes) in [
            (
                "VAE load",
                base.vae_load_phase_device_bytes,
                heavier.vae_load_phase_device_bytes,
            ),
            (
                "condition encode",
                base.condition_encode_phase_device_bytes,
                heavier.condition_encode_phase_device_bytes,
            ),
            (
                "visual decode",
                base.visual_decode_phase_device_bytes,
                heavier.visual_decode_phase_device_bytes,
            ),
            (
                "audio decode",
                base.audio_decode_phase_device_bytes,
                heavier.audio_decode_phase_device_bytes,
            ),
            (
                "waveform transfer",
                base.waveform_transfer_phase_device_bytes,
                heavier.waveform_transfer_phase_device_bytes,
            ),
        ] {
            assert_eq!(
                heavier_bytes - base_bytes,
                delta,
                "{phase} must charge the resident VAE pair"
            );
        }

        // Reload stages through the same construction workspace the first load
        // does, and only the visual-decode phase carries it after the park.
        assert_eq!(
            base.visual_decode_phase_device_bytes,
            base.fixed_runtime_device_bytes
                + base.attempt_resident_vae_device_bytes
                + base.vae_construction_device_workspace_bytes
                + base.packed_video_state_device_bytes
                + base.packed_audio_state_device_bytes
                + base.target_video_latent_device_bytes
                + base.target_audio_latent_device_bytes
                + base.decoder_tile_workspace_device_bytes
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
            qwen_host_load_staging_bytes,
            qwen_retained_header_host_bytes,
            transformer_retained_header_host_bytes,
            vae_retained_config_host_bytes,
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
            turbo_adapter_device_bytes,
            turbo_adapter_device_staging_bytes,
            turbo_adapter_host_staging_bytes,
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
    #[cfg(not(feature = "h3"))]
    fn exact_contract_only_authority_rejects_an_incomplete_runtime_registry() {
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
        let error = error.to_string();
        assert!(
            error.contains("public runtime registry is incomplete"),
            "{error}"
        );
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
        let error = error.to_string();
        assert!(
            error.contains("public runtime registry is incomplete"),
            "{error}"
        );

        let mut exact_ref2va = exact_input();
        exact_ref2va.model = contract::REF2VA_COMFY.into();
        assert!(FrozenH3FactoryAuthority::new_contract_only(exact_ref2va).is_err());

        let mut missing_components = legacy_input(contract::REF2VA_COMFY);
        missing_components.components.clear();
        assert!(FrozenH3FactoryAuthority::new_contract_only(missing_components).is_err());
    }
}
