//! Runtime-neutral ordered Ref2VA orchestration for MiniMax H3.
//!
//! This adapter is deliberately unreachable from the engine registry. It owns
//! no checkpoint paths, downloads, or model weights; a future legal-gated
//! runtime may implement [`H3Ref2VaBackend`] only after admission has frozen a
//! single device. The pipeline itself stores payload-free reference metadata
//! and proves ordering, geometry, noise, denoise, and output contracts with a
//! synthetic unit-test backend.

use super::*;
use crate::engine::GenerationReferenceBinding;
use crate::minimax_h3::reference_media::H3ReferenceMediaAdapter;
use crate::minimax_h3::sampler::H3_VISUAL_CONDITION_TIMESTEP;
use mold_candle::minimax_h3::{
    pack_h3_audio, sample_video_frames, AudioVaeConfig, RefPresentation, RefPresentationKind,
};
use mold_core::{
    generation_reference_fingerprint, GenerationReferenceKind, GenerationReferenceMetadata,
};

#[derive(Clone, Debug)]
pub(crate) struct H3PreparedRef2VaRequest {
    geometry: H3Fl2VaGeometry,
    references: Vec<H3PreparedReference>,
    reference_fingerprint: String,
    prompt: String,
    seed: u64,
    grid_points: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct H3PreparedReference {
    pub metadata: GenerationReferenceMetadata,
    pub shape: contract::GenerationReferencePreparedShape,
    pub target_frames: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct H3DecodedAudioFacts {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples_per_channel: u64,
}

/// Payload-free facts observed while the backend decodes its internally bound
/// media for one one-based reference. There is intentionally no place for a
/// path, upload handle, or byte buffer in this cross-layer contract.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct H3DecodedReferenceFacts {
    pub index: u32,
    pub kind: GenerationReferenceKind,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frame_count: Option<u32>,
    pub fps: Option<f64>,
    pub audio: Option<H3DecodedAudioFacts>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct H3ReferencePresentation {
    pub index: u32,
    pub presentation: RefPresentation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum H3AudioConditionEncodeMode {
    /// The official Ref2VA audio conditioner takes the FP32 posterior mode.
    /// It never consumes either the request RNG or a fresh posterior RNG.
    OfficialPosteriorModeF32,
}

/// A future Ref2VA runtime owns decoded media and every model component behind
/// one frozen device identity. Methods receive only one-based reference IDs
/// plus redacted, exact metadata; media authorities never cross this boundary.
pub(crate) trait H3Ref2VaBackend {
    fn identity(&self) -> H3PipelineBackendIdentity;
    fn device(&self) -> &Device;

    /// Hard admission bound for the complete packed sequence, including text,
    /// every reference block, and both generated suffixes.
    fn maximum_packed_rows(&self) -> usize;

    /// Concrete, weight-free media preprocessing owned by a future Ref2VA
    /// backend. Synthetic backends may override the two methods below; a
    /// production backend exposes this adapter and cannot silently substitute
    /// metadata-only preprocessing.
    fn reference_media_adapter(&mut self) -> Option<&mut H3ReferenceMediaAdapter> {
        None
    }

    fn decode_reference(
        &mut self,
        reference: &H3PreparedReference,
        binding: &GenerationReferenceBinding,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3DecodedReferenceFacts> {
        self.reference_media_adapter()
            .ok_or_else(|| anyhow!("MiniMax H3 Ref2VA backend has no reference-media adapter"))?
            .decode_reference(reference, binding, checkpoint)
    }

    /// Normalize the already-decoded media and retain it internally for the
    /// two VAE encoders. The returned presentation is the exact Qwen vision
    /// token geometry produced by that normalization.
    fn preprocess_reference(
        &mut self,
        reference: &H3PreparedReference,
        decoded: &H3DecodedReferenceFacts,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3ReferencePresentation> {
        self.reference_media_adapter()
            .ok_or_else(|| anyhow!("MiniMax H3 Ref2VA backend has no reference-media adapter"))?
            .preprocess_reference(reference, decoded, checkpoint)
    }

    fn encode_text(
        &mut self,
        prompt: &str,
        references: &[H3ReferencePresentation],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning>;

    fn encode_visual_reference(
        &mut self,
        reference: &H3PreparedReference,
        mode: ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor>;

    fn encode_audio_reference(
        &mut self,
        reference: &H3PreparedReference,
        mode: H3AudioConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoLatents>;

    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput>;

    fn decode_video(
        &mut self,
        latents: &Tensor,
        sink: &mut H3VideoEncodeSink,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()>;

    fn decode_audio(
        &mut self,
        latents: &StereoLatents,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform>;
}

#[derive(Clone, Debug)]
struct H3ReferenceLayoutSpec {
    index: u32,
    kind: GenerationReferenceKind,
    visual: Option<H3ReferenceVisualGeometry>,
    audio_latents_per_channel: usize,
}

#[derive(Clone, Copy, Debug)]
struct H3ReferenceVisualGeometry {
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    rows_per_frame: usize,
}

struct H3RefPackedSequence {
    layout: H3PackedLayout,
    condition_video_rows: usize,
    condition_audio_rows: usize,
    generated_video_rows: usize,
    generated_audio_rows: usize,
    #[cfg(test)]
    positions: Vec<[f32; 3]>,
    #[cfg(test)]
    timestep_indices: Vec<u32>,
    #[cfg(test)]
    video_indices: Vec<u32>,
    #[cfg(test)]
    audio_indices: Vec<u32>,
}

pub(crate) fn prepare_request(
    req: &GenerateRequest,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3PreparedRef2VaRequest> {
    prepare_request_with_authority(req, false, progress, observer)
}

pub(crate) fn prepare_resolved_request(
    req: &GenerateRequest,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3PreparedRef2VaRequest> {
    prepare_request_with_authority(req, true, progress, observer)
}

fn prepare_request_with_authority(
    req: &GenerateRequest,
    resolved_references: bool,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3PreparedRef2VaRequest> {
    let mut control = PipelineControl { progress, observer };
    phase_boundary(&mut control, H3PipelinePhase::Validate, false)?;
    let mode = if resolved_references {
        contract::validate_resolved_request_contract(req, Task::Ref2va)
    } else {
        contract::validate_request_contract(req, Task::Ref2va)
    }
    .map_err(|error| anyhow!("{}: {}", error.code, error.message))?;
    if mode != Mode::ReferenceToAudioVideo {
        bail!("MiniMax H3 Ref2VA preparation resolved the wrong mode {mode:?}");
    }
    if req.batch_size != 1 {
        bail!(
            "MiniMax H3 Ref2VA consumes one singleton sibling; batch size {} must be split by the authoritative batch router first",
            req.batch_size
        );
    }
    let source_references = req
        .references
        .as_deref()
        .ok_or_else(|| anyhow!("MiniMax H3 Ref2VA lost its validated references"))?;
    let frames = req.frames.unwrap_or(contract::MIN_FRAMES);
    let shapes = contract::reference_prepared_shapes_for_target(source_references, frames)
        .map_err(|error| anyhow!("{}: {}", error.code, error.message))?;
    let mut references = Vec::with_capacity(source_references.len());
    for (index, (reference, shape)) in source_references.iter().zip(shapes).enumerate() {
        let mut metadata = reference.redacted_metadata_lossless(index);
        if metadata.sha256.len() != 64 {
            bail!("MiniMax H3 Ref2VA reference {} lost its digest", index + 1);
        }
        metadata.prepared_shape = Some(shape.clone());
        references.push(H3PreparedReference {
            metadata,
            shape,
            target_frames: frames,
        });
    }
    let metadata = references
        .iter()
        .map(|reference| reference.metadata.clone())
        .collect::<Vec<_>>();
    let reference_fingerprint = generation_reference_fingerprint(&metadata);
    let geometry = H3Fl2VaGeometry::from_request(req, mode, 0)?;
    phase_boundary(&mut control, H3PipelinePhase::Validate, true)?;
    Ok(H3PreparedRef2VaRequest {
        geometry,
        references,
        reference_fingerprint,
        prompt: req.prompt.clone(),
        seed: req.seed.unwrap_or_else(rand_seed),
        grid_points: usize::try_from(req.steps).context("H3 grid points do not fit usize")?,
    })
}

pub(crate) fn execute_staged(
    prepared: &H3PreparedRef2VaRequest,
    bindings: &[GenerationReferenceBinding],
    backend: &mut dyn H3Ref2VaBackend,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3StagedAvOutput> {
    validate_reference_bindings(prepared, bindings)?;
    let frozen_identity = backend.identity();
    frozen_identity.validate(backend.device())?;
    let device = backend.device().clone();
    let mut control = PipelineControl { progress, observer };

    let total = prepared.references.len();
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::ReferenceDecode,
        completed: 0,
        total,
    })?;
    let mut decoded = Vec::with_capacity(total);
    for (offset, (reference, binding)) in prepared.references.iter().zip(bindings).enumerate() {
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        let facts = backend.decode_reference(reference, binding, &mut control)?;
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        validate_decoded_reference(reference, &facts)?;
        decoded.push(facts);
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::ReferenceDecode,
            completed: offset + 1,
            total,
        })?;
    }

    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::ReferencePreprocess,
        completed: 0,
        total,
    })?;
    let mut presentations = Vec::with_capacity(total);
    for (offset, (reference, facts)) in prepared.references.iter().zip(&decoded).enumerate() {
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        let presentation = backend.preprocess_reference(reference, facts, &mut control)?;
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        validate_reference_presentation(reference, &presentation)?;
        presentations.push(presentation);
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::ReferencePreprocess,
            completed: offset + 1,
            total,
        })?;
    }

    phase_boundary(&mut control, H3PipelinePhase::QwenEncode, false)?;
    ensure_ref_identity(backend, &frozen_identity, &device)?;
    let text = backend.encode_text(&prepared.prompt, &presentations, &mut control)?;
    ensure_ref_identity(backend, &frozen_identity, &device)?;
    text.validate(&device)?;
    validate_text_vision_rows(&text, &presentations)?;
    phase_boundary(&mut control, H3PipelinePhase::QwenEncode, true)?;

    let reference_layout = prepared
        .references
        .iter()
        .map(reference_layout_spec)
        .collect::<Result<Vec<_>>>()?;
    let packed = build_packed_sequence(&text.tags, &prepared.geometry, &reference_layout)?;
    if packed.layout.seq_len() > backend.maximum_packed_rows() {
        bail!(
            "MiniMax H3 Ref2VA packed {} rows, exceeding the frozen backend limit {}",
            packed.layout.seq_len(),
            backend.maximum_packed_rows()
        );
    }
    let frozen_layout = packed.layout.freeze(&device)?;

    let visual_total = reference_layout
        .iter()
        .filter(|reference| reference.visual.is_some())
        .count();
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::ReferenceVisualEncode,
        completed: 0,
        total: visual_total.max(1),
    })?;
    let mut visual_conditions = Vec::with_capacity(visual_total);
    for reference in prepared
        .references
        .iter()
        .filter(|reference| reference.metadata.kind != GenerationReferenceKind::Audio)
    {
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        let condition = backend.encode_visual_reference(
            reference,
            ConditionEncodeMode::OfficialFreshSeed42,
            &mut control,
        )?;
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        validate_visual_condition(reference, &condition, &device)?;
        visual_conditions.push(condition);
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::ReferenceVisualEncode,
            completed: visual_conditions.len(),
            total: visual_total.max(1),
        })?;
    }
    if visual_conditions.is_empty() {
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::ReferenceVisualEncode,
            completed: 1,
            total: 1,
        })?;
    }

    let audio_total = reference_layout
        .iter()
        .filter(|reference| reference.audio_latents_per_channel > 0)
        .count();
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::ReferenceAudioEncode,
        completed: 0,
        total: audio_total.max(1),
    })?;
    let mut audio_conditions = Vec::with_capacity(audio_total);
    for reference in prepared
        .references
        .iter()
        .filter(|reference| reference.shape.audio_rows > 0)
    {
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        let condition = backend.encode_audio_reference(
            reference,
            H3AudioConditionEncodeMode::OfficialPosteriorModeF32,
            &mut control,
        )?;
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        validate_audio_condition(reference, &condition, &device)?;
        audio_conditions.push(condition);
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::ReferenceAudioEncode,
            completed: audio_conditions.len(),
            total: audio_total.max(1),
        })?;
    }
    if audio_conditions.is_empty() {
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::ReferenceAudioEncode,
            completed: 1,
            total: 1,
        })?;
    }

    let mut posterior_draws = visual_conditions
        .iter()
        .enumerate()
        .map(|(ordinal, condition)| {
            Ok(H3NoiseDrawMetadata {
                stream: "condition-posterior",
                ordinal,
                shape: condition.dims().to_vec(),
                elements: checked_product(condition.dims(), "H3 condition posterior draw")?,
                seed: CONDITION_POSTERIOR_SEED,
                fresh_generator: true,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let draw_total = visual_conditions.len() + 2;
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::NoiseAllocation,
        completed: 0,
        total: draw_total,
    })?;
    let mut request_noise = H3RequestNoise::new(prepared.seed);
    let mut condition_video_rows = Vec::with_capacity(visual_conditions.len());
    for (ordinal, condition) in visual_conditions.iter().enumerate() {
        let noise = request_noise.draw("condition-noise", ordinal, condition.dims(), &device)?;
        let noised = condition
            .affine(f64::from(H3_VISUAL_CONDITION_TIMESTEP), 0.0)?
            .add(&noise.affine(f64::from(1.0 - H3_VISUAL_CONDITION_TIMESTEP), 0.0)?)?;
        condition_video_rows.push(patchify_h3_video(&noised, VIDEO_PATCH)?);
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::NoiseAllocation,
            completed: ordinal + 1,
            total: draw_total,
        })?;
    }
    let video_noise = request_noise.draw(
        "target-video",
        0,
        &prepared.geometry.generated_video_shape(),
        &device,
    )?;
    let generated_video = patchify_h3_video(&video_noise, VIDEO_PATCH)?;
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::NoiseAllocation,
        completed: visual_conditions.len() + 1,
        total: draw_total,
    })?;
    let generated_audio = request_noise.draw(
        "target-audio",
        0,
        &prepared.geometry.generated_audio_row_shape(),
        &device,
    )?;
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::NoiseAllocation,
        completed: draw_total,
        total: draw_total,
    })?;

    let mut video_parts = condition_video_rows.iter().collect::<Vec<_>>();
    video_parts.push(&generated_video);
    let mut video_rows = Tensor::cat(&video_parts, 1)?;
    let packed_audio_conditions = audio_conditions
        .iter()
        .map(|condition| pack_h3_audio(condition.normalized()))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let mut audio_parts = packed_audio_conditions.iter().collect::<Vec<_>>();
    audio_parts.push(&generated_audio);
    let mut audio_rows = Tensor::cat(&audio_parts, 1)?;
    validate_packed_tensors(&video_rows, &audio_rows, &text.states, &packed)?;

    let schedule = H3DualSchedule::new(prepared.grid_points)?;
    let counts = schedule.counts();
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::Denoise,
        completed: 0,
        total: counts.transformer_evaluations,
    })?;
    for step in schedule.steps() {
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        let timesteps = Tensor::from_slice(
            &[
                step.row_timesteps.generated_video,
                step.row_timesteps.visual_condition,
                step.row_timesteps.generated_audio,
                step.row_timesteps.audio_reference,
            ],
            4,
            &device,
        )?;
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        let output = backend.denoise(
            H3ForwardInput {
                video_rows: &video_rows,
                audio_rows: &audio_rows,
                text_states: &text.states,
                timesteps: &timesteps,
            },
            &frozen_layout,
            &mut control,
        )?;
        ensure_ref_identity(backend, &frozen_identity, &device)?;
        validate_transformer_output(&output, &video_rows, &audio_rows)?;

        let generated_video_rows =
            video_rows.narrow(1, packed.condition_video_rows, packed.generated_video_rows)?;
        let generated_video_velocity =
            output
                .video
                .narrow(1, packed.condition_video_rows, packed.generated_video_rows)?;
        let generated_audio_rows =
            audio_rows.narrow(1, packed.condition_audio_rows, packed.generated_audio_rows)?;
        let generated_audio_velocity =
            output
                .audio
                .narrow(1, packed.condition_audio_rows, packed.generated_audio_rows)?;
        let (next_video, next_audio) = euler_step_pair(
            &generated_video_rows,
            &generated_audio_rows,
            &generated_video_velocity,
            &generated_audio_velocity,
            step,
        )?;
        video_rows = preserve_prefix(&video_rows, packed.condition_video_rows, &next_video)?;
        audio_rows = preserve_prefix(&audio_rows, packed.condition_audio_rows, &next_audio)?;
        let after = step.progress_after_update();
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::Denoise,
            completed: after.completed_evaluations,
            total: after.total_evaluations,
        })?;
    }

    let generated_video_rows =
        video_rows.narrow(1, packed.condition_video_rows, packed.generated_video_rows)?;
    let video_latents = unpatchify_h3_video(
        &generated_video_rows,
        [
            prepared.geometry.latent_frames / VIDEO_PATCH[0],
            prepared.geometry.latent_height / VIDEO_PATCH[1],
            prepared.geometry.latent_width / VIDEO_PATCH[2],
        ],
        VIDEO_LATENT_CHANNELS,
        VIDEO_PATCH,
    )?;
    let generated_audio_rows =
        audio_rows.narrow(1, packed.condition_audio_rows, packed.generated_audio_rows)?;
    let audio_latents = StereoLatents::new(
        unpack_h3_audio(&generated_audio_rows, AUDIO_CHANNELS as usize)?.to_dtype(DType::F32)?,
        &AudioVaeConfig::default(),
    )?;

    phase_boundary(&mut control, H3PipelinePhase::VisualDecode, false)?;
    ensure_ref_identity(backend, &frozen_identity, &device)?;
    let mut sink = H3VideoEncodeSink::new_dimensions(
        prepared.geometry.width,
        prepared.geometry.height,
        prepared.geometry.frames,
    )?;
    backend.decode_video(&video_latents, &mut sink, &mut control)?;
    ensure_ref_identity(backend, &frozen_identity, &device)?;
    let encoded_video = sink.finish()?;
    phase_boundary(&mut control, H3PipelinePhase::VisualDecode, true)?;

    phase_boundary(&mut control, H3PipelinePhase::AudioDecode, false)?;
    ensure_ref_identity(backend, &frozen_identity, &device)?;
    let waveform = backend.decode_audio(&audio_latents, &mut control)?;
    ensure_ref_identity(backend, &frozen_identity, &device)?;
    validate_waveform(&waveform, prepared.geometry.audio_latents_per_channel)?;
    phase_boundary(&mut control, H3PipelinePhase::AudioDecode, true)?;

    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::Staged,
        completed: 1,
        total: 1,
    })?;
    Ok(H3StagedAvOutput {
        video_only_mp4: encoded_video.mp4,
        thumbnail_png: encoded_video.thumbnail_png,
        waveform,
        provenance: H3PipelineProvenance {
            mode: "ref2va",
            seed: prepared.seed,
            width: prepared.geometry.width,
            height: prepared.geometry.height,
            frames: prepared.geometry.frames,
            fps: FIXED_FPS,
            video_latent_frames: prepared.geometry.latent_frames,
            audio_latents_per_channel: prepared.geometry.audio_latents_per_channel,
            audio_sample_rate: AUDIO_SAMPLE_RATE_HZ,
            audio_channels: AUDIO_CHANNELS,
            requested_grid_points: counts.requested_grid_points,
            transformer_evaluations: counts.transformer_evaluations,
            endpoint_anchors: Vec::new(),
            references: prepared
                .references
                .iter()
                .map(reference_provenance)
                .collect(),
            reference_fingerprint: Some(prepared.reference_fingerprint.clone()),
            noise_domain_version: NOISE_DOMAIN_VERSION,
            noise_draws: {
                posterior_draws.extend(request_noise.draws);
                posterior_draws
            },
            device_id: frozen_identity.device_id,
            execution_fingerprint: frozen_identity.execution_fingerprint,
        },
    })
}

fn validate_reference_bindings(
    prepared: &H3PreparedRef2VaRequest,
    bindings: &[GenerationReferenceBinding],
) -> Result<()> {
    if bindings.len() != prepared.references.len() {
        bail!(
            "MiniMax H3 Ref2VA received {} private media bindings for {} ordered references",
            bindings.len(),
            prepared.references.len()
        );
    }
    for (reference, binding) in prepared.references.iter().zip(bindings) {
        let mut bound_metadata = binding.metadata().clone();
        bound_metadata.prepared_shape = Some(reference.shape.clone());
        if bound_metadata != reference.metadata {
            bail!(
                "MiniMax H3 Ref2VA private media binding {} differs from frozen reference provenance",
                reference.metadata.index
            );
        }
    }
    let metadata = bindings
        .iter()
        .zip(&prepared.references)
        .map(|(binding, reference)| {
            let mut metadata = binding.metadata().clone();
            metadata.prepared_shape = Some(reference.shape.clone());
            metadata
        })
        .collect::<Vec<_>>();
    if generation_reference_fingerprint(&metadata) != prepared.reference_fingerprint {
        bail!("MiniMax H3 Ref2VA private media binding order changed after admission");
    }
    Ok(())
}

fn validate_decoded_reference(
    reference: &H3PreparedReference,
    decoded: &H3DecodedReferenceFacts,
) -> Result<()> {
    let metadata = &reference.metadata;
    if decoded.index != metadata.index || decoded.kind != metadata.kind {
        bail!(
            "MiniMax H3 decoded reference identity {:?}/{} differs from frozen {:?}/{}",
            decoded.kind,
            decoded.index,
            metadata.kind,
            metadata.index
        );
    }
    let expected_audio = match metadata.kind {
        GenerationReferenceKind::Image => {
            if (decoded.width, decoded.height) != (metadata.width, metadata.height)
                || decoded.frame_count.is_some()
                || decoded.fps.is_some()
            {
                bail!("MiniMax H3 decoded image facts differ from the frozen descriptor");
            }
            None
        }
        GenerationReferenceKind::Video => {
            if (decoded.width, decoded.height, decoded.frame_count)
                != (metadata.width, metadata.height, metadata.frame_count)
                || decoded.fps.map(f64::to_bits) != metadata.fps.map(f64::to_bits)
            {
                bail!("MiniMax H3 decoded video facts differ from the frozen descriptor");
            }
            metadata.has_audio.then_some(H3DecodedAudioFacts {
                sample_rate: metadata.audio_sample_rate.unwrap_or_default(),
                channels: metadata.audio_channels.unwrap_or_default(),
                samples_per_channel: metadata.audio_sample_count.unwrap_or_default(),
            })
        }
        GenerationReferenceKind::Audio => {
            if decoded.width.is_some()
                || decoded.height.is_some()
                || decoded.frame_count.is_some()
                || decoded.fps.is_some()
            {
                bail!("MiniMax H3 decoded audio reference unexpectedly carries visual facts");
            }
            Some(H3DecodedAudioFacts {
                sample_rate: metadata.sample_rate.unwrap_or_default(),
                channels: metadata.channels.unwrap_or_default(),
                samples_per_channel: metadata.sample_count.unwrap_or_default(),
            })
        }
    };
    if decoded.audio != expected_audio {
        bail!("MiniMax H3 decoded audio facts differ from the frozen descriptor");
    }
    Ok(())
}

fn validate_reference_presentation(
    reference: &H3PreparedReference,
    presentation: &H3ReferencePresentation,
) -> Result<()> {
    if presentation.index != reference.metadata.index {
        bail!("MiniMax H3 Qwen presentation changed reference order");
    }
    match (&presentation.presentation.kind, reference.metadata.kind) {
        (RefPresentationKind::Image { vision_tokens }, GenerationReferenceKind::Image) => {
            if *vision_tokens == 0 || presentation.presentation.has_audio {
                bail!("MiniMax H3 image presentation has invalid vision/audio geometry");
            }
        }
        (RefPresentationKind::Audio, GenerationReferenceKind::Audio) => {
            if !presentation.presentation.has_audio {
                bail!("MiniMax H3 audio presentation omitted its audio label");
            }
        }
        (RefPresentationKind::Video { blocks }, GenerationReferenceKind::Video) => {
            if presentation.presentation.has_audio != reference.metadata.has_audio {
                bail!("MiniMax H3 video presentation changed soundtrack presence");
            }
            let normalized = usize::try_from(
                reference
                    .shape
                    .normalized_video_frames
                    .ok_or_else(|| anyhow!("video reference lost normalized frame count"))?,
            )?;
            let sampled = sample_video_frames(normalized, f64::from(FIXED_FPS))?;
            if reference.shape.qwen_video_frames != Some(sampled.frame_indices.len() as u32)
                || blocks.len() != sampled.block_timestamps_seconds.len()
                || blocks.iter().any(|block| block.vision_tokens == 0)
                || blocks
                    .iter()
                    .map(|block| block.timestamp_seconds.to_bits())
                    .ne(sampled
                        .block_timestamps_seconds
                        .iter()
                        .map(|timestamp| timestamp.to_bits()))
            {
                bail!("MiniMax H3 video Qwen sampling differs from the official 2 fps temporal-patch-2 contract");
            }
        }
        _ => bail!("MiniMax H3 Qwen presentation changed reference modality"),
    }
    Ok(())
}

fn validate_text_vision_rows(
    text: &H3TextConditioning,
    references: &[H3ReferencePresentation],
) -> Result<()> {
    let expected = references
        .iter()
        .map(|reference| match &reference.presentation.kind {
            RefPresentationKind::Audio => 0,
            RefPresentationKind::Image { vision_tokens } => *vision_tokens,
            RefPresentationKind::Video { blocks } => {
                blocks.iter().map(|block| block.vision_tokens).sum()
            }
        })
        .sum::<usize>();
    let actual = text
        .tags
        .iter()
        .filter(|tag| **tag == H3ModalityTag::Vision)
        .count();
    if actual != expected {
        bail!(
            "MiniMax H3 conditioner returned {actual} vision rows for {expected} presentation pads"
        );
    }
    Ok(())
}

fn reference_layout_spec(reference: &H3PreparedReference) -> Result<H3ReferenceLayoutSpec> {
    let shape = &reference.shape;
    let visual = match reference.metadata.kind {
        GenerationReferenceKind::Audio => None,
        GenerationReferenceKind::Image | GenerationReferenceKind::Video => {
            let width = usize::try_from(
                shape
                    .normalized_width
                    .ok_or_else(|| anyhow!("visual reference lost normalized width"))?,
            )?;
            let height = usize::try_from(
                shape
                    .normalized_height
                    .ok_or_else(|| anyhow!("visual reference lost normalized height"))?,
            )?;
            let latent_width = width / VIDEO_VAE_SPATIAL_COMPRESSION;
            let latent_height = height / VIDEO_VAE_SPATIAL_COMPRESSION;
            if !latent_width.is_multiple_of(VIDEO_PATCH[2])
                || !latent_height.is_multiple_of(VIDEO_PATCH[1])
            {
                bail!("MiniMax H3 reference latent canvas is not patch divisible");
            }
            let latent_frames = if reference.metadata.kind == GenerationReferenceKind::Image {
                1
            } else {
                VisualTemporalGeometry::default().encoded_frames(usize::try_from(
                    shape
                        .video_frames
                        .ok_or_else(|| anyhow!("video reference lost visual-VAE frame count"))?,
                )?)?
            };
            let rows_per_frame = checked_product(
                &[
                    latent_height / VIDEO_PATCH[1],
                    latent_width / VIDEO_PATCH[2],
                ],
                "H3 reference visual rows per frame",
            )?;
            let rows = latent_frames
                .checked_mul(rows_per_frame)
                .ok_or_else(|| anyhow!("H3 reference visual rows overflowed"))?;
            if rows as u64 != shape.visual_rows {
                bail!("MiniMax H3 reference visual geometry differs from prepared placement shape");
            }
            Some(H3ReferenceVisualGeometry {
                latent_frames,
                latent_height,
                latent_width,
                rows_per_frame,
            })
        }
    };
    let audio_latents_per_channel = usize::try_from(shape.audio_rows / u64::from(AUDIO_CHANNELS))?;
    let expected_audio = match reference.metadata.kind {
        GenerationReferenceKind::Image => false,
        GenerationReferenceKind::Video => reference.metadata.has_audio,
        GenerationReferenceKind::Audio => true,
    };
    if !shape.audio_rows.is_multiple_of(u64::from(AUDIO_CHANNELS))
        || (audio_latents_per_channel > 0) != expected_audio
    {
        bail!("MiniMax H3 reference audio geometry differs from its modality");
    }
    Ok(H3ReferenceLayoutSpec {
        index: reference.metadata.index,
        kind: reference.metadata.kind,
        visual,
        audio_latents_per_channel,
    })
}

fn build_packed_sequence(
    text_tags: &[H3ModalityTag],
    geometry: &H3Fl2VaGeometry,
    references: &[H3ReferenceLayoutSpec],
) -> Result<H3RefPackedSequence> {
    if text_tags.is_empty() {
        bail!("MiniMax H3 Ref2VA packed sequence requires text rows");
    }
    for (offset, reference) in references.iter().enumerate() {
        if reference.index != u32::try_from(offset + 1)? {
            bail!("MiniMax H3 Ref2VA reference indices are not contiguous request order");
        }
    }
    let condition_video_rows = references.iter().try_fold(0usize, |total, reference| {
        let rows = reference
            .visual
            .map_or(0, |visual| visual.latent_frames * visual.rows_per_frame);
        total
            .checked_add(rows)
            .ok_or_else(|| anyhow!("H3 condition video row count overflowed"))
    })?;
    let condition_audio_rows = references.iter().try_fold(0usize, |total, reference| {
        total
            .checked_add(reference.audio_latents_per_channel * AUDIO_CHANNELS as usize)
            .ok_or_else(|| anyhow!("H3 condition audio row count overflowed"))
    })?;
    let sequence_len = text_tags
        .len()
        .checked_add(condition_video_rows)
        .and_then(|rows| rows.checked_add(condition_audio_rows))
        .and_then(|rows| rows.checked_add(geometry.generated_audio_rows))
        .and_then(|rows| rows.checked_add(geometry.generated_video_rows))
        .ok_or_else(|| anyhow!("H3 Ref2VA packed row count overflowed"))?;
    if sequence_len > u32::MAX as usize {
        bail!("MiniMax H3 Ref2VA packed sequence exceeds u32 addressing");
    }

    let mut positions = vec![[0.0f32; 3]; sequence_len];
    let mut timestep_indices = vec![0u32; sequence_len];
    let mut tags = vec![H3Modality::Video as u32; sequence_len];
    let text_indices = checked_indices(0, text_tags.len())?;
    for (index, (position, tag)) in positions[..text_tags.len()]
        .iter_mut()
        .zip(text_tags)
        .enumerate()
    {
        position[0] = index as f32;
        tags[index] = match tag {
            H3ModalityTag::Vision => H3Modality::Video as u32,
            H3ModalityTag::Text => H3Modality::Text as u32,
        };
    }

    let (_, target_width_grid) = frame_position_grid(
        geometry.latent_height,
        geometry.latent_width,
        VIDEO_PATCH[1],
        VIDEO_PATCH[2],
    )?;
    let mut cursor = text_tags.len();
    let mut rotary_time = text_tags.len() as f64;
    let mut video_indices =
        Vec::with_capacity(condition_video_rows + geometry.generated_video_rows);
    let mut audio_indices =
        Vec::with_capacity(condition_audio_rows + geometry.generated_audio_rows);
    for reference in references {
        match reference.kind {
            GenerationReferenceKind::Image => {
                let visual = reference.visual.expect("image layout has visual geometry");
                let rows = visual.latent_frames * visual.rows_per_frame;
                let range = cursor..cursor + rows;
                let (frame_grid, _) = frame_position_grid(
                    visual.latent_height,
                    visual.latent_width,
                    VIDEO_PATCH[1],
                    VIDEO_PATCH[2],
                )?;
                for (position, spatial) in positions[range.clone()].iter_mut().zip(&frame_grid) {
                    *position = [rotary_time as f32, spatial[0], spatial[1]];
                }
                timestep_indices[range.clone()].fill(1);
                video_indices.extend(checked_indices(range.start, rows)?);
                cursor = range.end;
                rotary_time += 1.0;
            }
            GenerationReferenceKind::Audio => {
                let rows = reference.audio_latents_per_channel * AUDIO_CHANNELS as usize;
                let range = cursor..cursor + rows;
                fill_audio_positions(
                    &mut positions,
                    range.clone(),
                    reference.audio_latents_per_channel,
                    rotary_time,
                    &target_width_grid,
                )?;
                tags[range.clone()].fill(H3Modality::Audio as u32);
                timestep_indices[range.clone()].fill(3);
                audio_indices.extend(checked_indices(range.start, rows)?);
                cursor = range.end;
                rotary_time += reference.audio_latents_per_channel as f64;
            }
            GenerationReferenceKind::Video => {
                let visual = reference.visual.expect("video layout has visual geometry");
                let audio_rows = reference.audio_latents_per_channel * AUDIO_CHANNELS as usize;
                let video_rows = visual.latent_frames * visual.rows_per_frame;
                let audio_range = cursor..cursor + audio_rows;
                let video_range = audio_range.end..audio_range.end + video_rows;
                let (frame_grid, width_grid) = frame_position_grid(
                    visual.latent_height,
                    visual.latent_width,
                    VIDEO_PATCH[1],
                    VIDEO_PATCH[2],
                )?;
                fill_audio_positions(
                    &mut positions,
                    audio_range.clone(),
                    reference.audio_latents_per_channel,
                    rotary_time,
                    &width_grid,
                )?;
                let times = temporal_position_grid(visual.latent_frames, rotary_time);
                for (frame, time) in times.into_iter().enumerate() {
                    let start = video_range.start + frame * visual.rows_per_frame;
                    for (position, spatial) in positions[start..start + visual.rows_per_frame]
                        .iter_mut()
                        .zip(&frame_grid)
                    {
                        *position = [time, spatial[0], spatial[1]];
                    }
                }
                tags[audio_range.clone()].fill(H3Modality::Audio as u32);
                timestep_indices[audio_range.clone()].fill(3);
                timestep_indices[video_range.clone()].fill(1);
                audio_indices.extend(checked_indices(audio_range.start, audio_rows)?);
                video_indices.extend(checked_indices(video_range.start, video_rows)?);
                cursor = video_range.end;
                let mut sequential_span = 0.0;
                for frame in 0..visual.latent_frames {
                    sequential_span += temporal_span(frame);
                }
                rotary_time += (reference.audio_latents_per_channel as f64).max(sequential_span);
            }
        }
    }

    let target_audio_range = cursor..cursor + geometry.generated_audio_rows;
    let target_video_range =
        target_audio_range.end..target_audio_range.end + geometry.generated_video_rows;
    fill_audio_positions(
        &mut positions,
        target_audio_range.clone(),
        geometry.audio_latents_per_channel,
        rotary_time,
        &target_width_grid,
    )?;
    let (target_frame_grid, _) = frame_position_grid(
        geometry.latent_height,
        geometry.latent_width,
        VIDEO_PATCH[1],
        VIDEO_PATCH[2],
    )?;
    let target_times = temporal_position_grid(geometry.latent_frames, rotary_time);
    for (frame, time) in target_times.into_iter().enumerate() {
        let start = target_video_range.start + frame * geometry.rows_per_video_frame;
        for (position, spatial) in positions[start..start + geometry.rows_per_video_frame]
            .iter_mut()
            .zip(&target_frame_grid)
        {
            *position = [time, spatial[0], spatial[1]];
        }
    }
    tags[target_audio_range.clone()].fill(H3Modality::Audio as u32);
    timestep_indices[target_audio_range.clone()].fill(2);
    audio_indices.extend(checked_indices(
        target_audio_range.start,
        geometry.generated_audio_rows,
    )?);
    video_indices.extend(checked_indices(
        target_video_range.start,
        geometry.generated_video_rows,
    )?);
    if target_video_range.end != sequence_len {
        bail!("MiniMax H3 Ref2VA layout cursor did not consume the packed sequence");
    }

    let layout = H3PackedLayout::new(
        positions.clone(),
        timestep_indices.clone(),
        tags,
        video_indices.clone(),
        audio_indices.clone(),
        text_indices,
    )?;
    Ok(H3RefPackedSequence {
        layout,
        condition_video_rows,
        condition_audio_rows,
        generated_video_rows: geometry.generated_video_rows,
        generated_audio_rows: geometry.generated_audio_rows,
        #[cfg(test)]
        positions,
        #[cfg(test)]
        timestep_indices,
        #[cfg(test)]
        video_indices,
        #[cfg(test)]
        audio_indices,
    })
}

fn fill_audio_positions(
    positions: &mut [[f32; 3]],
    rows: std::ops::Range<usize>,
    latents_per_channel: usize,
    origin: f64,
    width_grid: &[f32],
) -> Result<()> {
    if rows.len() != latents_per_channel * AUDIO_CHANNELS as usize || width_grid.is_empty() {
        bail!("MiniMax H3 audio rotary block has invalid geometry");
    }
    for channel in 0..AUDIO_CHANNELS as usize {
        let width = if channel == 0 {
            width_grid[0]
        } else {
            *width_grid.last().expect("nonempty width grid")
        };
        let start = rows.start + channel * latents_per_channel;
        for (offset, position) in positions[start..start + latents_per_channel]
            .iter_mut()
            .enumerate()
        {
            *position = [(origin + offset as f64) as f32, 0.0, width];
        }
    }
    Ok(())
}

fn validate_visual_condition(
    reference: &H3PreparedReference,
    condition: &Tensor,
    device: &Device,
) -> Result<()> {
    let layout = reference_layout_spec(reference)?;
    let visual = layout
        .visual
        .ok_or_else(|| anyhow!("audio-only reference reached the visual VAE"))?;
    let expected = [
        1,
        VIDEO_LATENT_CHANNELS,
        visual.latent_frames,
        visual.latent_height,
        visual.latent_width,
    ];
    if condition.dtype() != DType::F32
        || condition.dims() != expected
        || !condition.device().same_device(device)
    {
        bail!(
            "MiniMax H3 visual reference {} encoded as {:?} {:?}, expected F32 {:?} on the frozen device",
            reference.metadata.index,
            condition.dtype(),
            condition.dims(),
            expected
        );
    }
    Ok(())
}

fn validate_audio_condition(
    reference: &H3PreparedReference,
    condition: &StereoLatents,
    device: &Device,
) -> Result<()> {
    let expected_latents = usize::try_from(reference.shape.audio_rows / u64::from(AUDIO_CHANNELS))?;
    let expected = [
        1,
        AUDIO_LATENT_CHANNELS,
        AUDIO_CHANNELS as usize,
        expected_latents,
    ];
    if condition.normalized().dtype() != DType::F32
        || condition.normalized().dims() != expected
        || !condition.normalized().device().same_device(device)
    {
        bail!(
            "MiniMax H3 audio reference {} encoded as {:?} {:?}, expected F32 {:?} on the frozen device",
            reference.metadata.index,
            condition.normalized().dtype(),
            condition.normalized().dims(),
            expected
        );
    }
    Ok(())
}

fn validate_packed_tensors(
    video: &Tensor,
    audio: &Tensor,
    text: &Tensor,
    packed: &H3RefPackedSequence,
) -> Result<()> {
    let (_, video_rows, video_width) = video.dims3()?;
    let (_, audio_rows, audio_width) = audio.dims3()?;
    let (_, text_rows, text_width) = text.dims3()?;
    if video_rows != packed.condition_video_rows + packed.generated_video_rows
        || video_width != VIDEO_LATENT_CHANNELS * 4
        || audio_rows != packed.condition_audio_rows + packed.generated_audio_rows
        || audio_width != AUDIO_LATENT_CHANNELS
        || text_rows == 0
        || text_width != TEXT_STATE_WIDTH
        || packed.layout.seq_len() != video_rows + audio_rows + text_rows
    {
        bail!("MiniMax H3 Ref2VA packed tensors differ from the frozen ordered layout");
    }
    Ok(())
}

fn preserve_prefix(rows: &Tensor, prefix: usize, generated: &Tensor) -> Result<Tensor> {
    if prefix == 0 {
        return Ok(generated.clone());
    }
    Tensor::cat(&[&rows.narrow(1, 0, prefix)?, generated], 1).map_err(Into::into)
}

fn reference_provenance(reference: &H3PreparedReference) -> H3ReferenceProvenance {
    let metadata = &reference.metadata;
    let original_samples_per_channel = match metadata.kind {
        GenerationReferenceKind::Video => metadata.audio_sample_count,
        GenerationReferenceKind::Audio => metadata.sample_count,
        GenerationReferenceKind::Image => None,
    };
    let used_samples_per_channel = reference.shape.audio_samples_per_channel;
    let used_duration_ms = match metadata.kind {
        GenerationReferenceKind::Image => None,
        GenerationReferenceKind::Video => reference
            .shape
            .normalized_video_frames
            .map(|frames| u64::from(frames).saturating_mul(1_000) / u64::from(FIXED_FPS)),
        GenerationReferenceKind::Audio => used_samples_per_channel
            .map(|samples| samples.saturating_mul(1_000) / u64::from(AUDIO_SAMPLE_RATE_HZ)),
    };
    H3ReferenceProvenance {
        index: metadata.index,
        kind: metadata.kind,
        name: metadata.name.clone(),
        sha256: metadata.sha256.clone(),
        original_duration_ms: metadata.duration_ms,
        used_duration_ms,
        original_samples_per_channel,
        used_samples_per_channel,
    }
}

fn ensure_ref_identity(
    backend: &dyn H3Ref2VaBackend,
    frozen: &H3PipelineBackendIdentity,
    frozen_device: &Device,
) -> Result<()> {
    let current = backend.identity();
    if &current != frozen {
        bail!(
            "MiniMax H3 Ref2VA backend identity changed from {:?} to {:?}; implicit reroute is forbidden",
            frozen,
            current
        );
    }
    current.validate(backend.device())?;
    if !backend.device().same_device(frozen_device) {
        bail!("MiniMax H3 Ref2VA backend device changed after placement was frozen");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};
    use mold_candle::minimax_h3::{AudioSoundtrackAssociation, VideoPresentationBlock};
    use mold_core::{GenerationReference, GenerationReferenceAuthority, OutputFormat};

    use super::*;
    use crate::progress::{is_inference_cancelled, InferenceCancellationToken, ProgressReporter};

    fn request() -> GenerateRequest {
        GenerateRequest {
            prompt: "a brass automaton conducting an orchestra".into(),
            negative_prompt: None,
            model: contract::REF2VA_COMFY.into(),
            width: 32,
            height: 32,
            steps: 4,
            guidance: 0.0,
            seed: Some(7),
            batch_size: 1,
            output_format: Some(OutputFormat::Mp4),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: Some(vec![
                video_reference(1),
                image_reference(2),
                audio_reference(3),
            ]),
            strength: 1.0,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: Some(124),
            fps: Some(FIXED_FPS),
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            guidance_overrides: None,
            placement: None,
        }
    }

    fn provenance(name: &str) -> mold_core::GenerationReferenceProvenance {
        mold_core::GenerationReferenceProvenance {
            name: Some(name.into()),
            sha256: None,
        }
    }

    fn video_reference(byte: u8) -> GenerationReference {
        GenerationReference::Video {
            media: GenerationReferenceAuthority::Inline {
                data: vec![byte; 16],
            },
            provenance: provenance("motion.mp4"),
            mime_type: "video/mp4".into(),
            width: 64,
            height: 64,
            frame_count: Some(48),
            duration_ms: 2_000,
            fps: 24.0,
            has_audio: true,
            audio_duration_ms: Some(2_000),
            audio_sample_count: Some(96_000),
            audio_sample_rate: Some(48_000),
            audio_channels: Some(2),
        }
    }

    fn image_reference(byte: u8) -> GenerationReference {
        GenerationReference::Image {
            media: GenerationReferenceAuthority::Inline {
                data: vec![byte; 12],
            },
            provenance: provenance("portrait.png"),
            mime_type: "image/png".into(),
            width: 48,
            height: 48,
        }
    }

    fn audio_reference(byte: u8) -> GenerationReference {
        GenerationReference::Audio {
            media: GenerationReferenceAuthority::Inline {
                data: vec![byte; 20],
            },
            provenance: provenance("voice.wav"),
            mime_type: "audio/wav".into(),
            duration_ms: 2_000,
            sample_rate: 48_000,
            channels: 1,
            sample_count: Some(96_000),
        }
    }

    fn prepare(req: &GenerateRequest) -> H3PreparedRef2VaRequest {
        prepare_request(
            req,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap()
    }

    fn bindings(prepared: &H3PreparedRef2VaRequest) -> Vec<GenerationReferenceBinding> {
        prepared
            .references
            .iter()
            .map(|reference| {
                let mut metadata = reference.metadata.clone();
                metadata.prepared_shape = None;
                GenerationReferenceBinding::synthetic(metadata)
            })
            .collect()
    }

    #[derive(Default)]
    struct RecordingObserver {
        events: Vec<H3PipelineEvent>,
        cancel_at: Option<H3PipelineEvent>,
        cancellation: Option<InferenceCancellationToken>,
    }

    impl H3PipelineObserver for RecordingObserver {
        fn observe(&mut self, event: H3PipelineEvent) {
            self.events.push(event);
            if self.cancel_at == Some(event) {
                self.cancellation.as_ref().unwrap().cancel();
            }
        }
    }

    struct SyntheticBackend {
        device: Device,
        identity: H3PipelineBackendIdentity,
        maximum_rows: usize,
        decoded_order: Vec<u32>,
        preprocessed_order: Vec<u32>,
        visual_order: Vec<u32>,
        audio_order: Vec<u32>,
        condition_video_checksums: Vec<f32>,
        condition_audio_checksums: Vec<f32>,
        reroute_after_decode: bool,
        reroute_after_denoise: bool,
    }

    impl SyntheticBackend {
        fn new() -> Self {
            Self {
                device: Device::Cpu,
                identity: H3PipelineBackendIdentity {
                    kind: H3PipelineBackendKind::SyntheticCpu,
                    device_id: "synthetic-ref2va-cpu-0".into(),
                    execution_fingerprint: "synthetic-ref2va-v1".into(),
                },
                maximum_rows: 100_000,
                decoded_order: Vec::new(),
                preprocessed_order: Vec::new(),
                visual_order: Vec::new(),
                audio_order: Vec::new(),
                condition_video_checksums: Vec::new(),
                condition_audio_checksums: Vec::new(),
                reroute_after_decode: false,
                reroute_after_denoise: false,
            }
        }
    }

    impl H3Ref2VaBackend for SyntheticBackend {
        fn identity(&self) -> H3PipelineBackendIdentity {
            let mut identity = self.identity.clone();
            if (self.reroute_after_decode && !self.decoded_order.is_empty())
                || (self.reroute_after_denoise && !self.condition_video_checksums.is_empty())
            {
                identity.device_id = "synthetic-ref2va-cpu-1".into();
            }
            identity
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn maximum_packed_rows(&self) -> usize {
            self.maximum_rows
        }

        fn decode_reference(
            &mut self,
            reference: &H3PreparedReference,
            binding: &GenerationReferenceBinding,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3DecodedReferenceFacts> {
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::ReferenceDecodeChunk,
                completed: 0,
                total: 1,
            })?;
            let metadata = &reference.metadata;
            let mut bound_metadata = binding.metadata().clone();
            bound_metadata.prepared_shape = Some(reference.shape.clone());
            assert_eq!(&bound_metadata, metadata);
            assert!(binding.file().metadata().unwrap().is_file());
            self.decoded_order.push(metadata.index);
            let audio = match metadata.kind {
                GenerationReferenceKind::Image => None,
                GenerationReferenceKind::Video if metadata.has_audio => Some(H3DecodedAudioFacts {
                    sample_rate: metadata.audio_sample_rate.unwrap(),
                    channels: metadata.audio_channels.unwrap(),
                    samples_per_channel: metadata.audio_sample_count.unwrap(),
                }),
                GenerationReferenceKind::Video => None,
                GenerationReferenceKind::Audio => Some(H3DecodedAudioFacts {
                    sample_rate: metadata.sample_rate.unwrap(),
                    channels: metadata.channels.unwrap(),
                    samples_per_channel: metadata.sample_count.unwrap(),
                }),
            };
            Ok(H3DecodedReferenceFacts {
                index: metadata.index,
                kind: metadata.kind,
                width: metadata.width,
                height: metadata.height,
                frame_count: metadata.frame_count,
                fps: metadata.fps,
                audio,
            })
        }

        fn preprocess_reference(
            &mut self,
            reference: &H3PreparedReference,
            _decoded: &H3DecodedReferenceFacts,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3ReferencePresentation> {
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::ReferencePreprocessChunk,
                completed: 0,
                total: 1,
            })?;
            self.preprocessed_order.push(reference.metadata.index);
            let presentation = match reference.metadata.kind {
                GenerationReferenceKind::Image => RefPresentation {
                    kind: RefPresentationKind::Image { vision_tokens: 2 },
                    has_audio: false,
                },
                GenerationReferenceKind::Audio => RefPresentation {
                    kind: RefPresentationKind::Audio,
                    has_audio: true,
                },
                GenerationReferenceKind::Video => {
                    let sampled = sample_video_frames(
                        reference.shape.normalized_video_frames.unwrap() as usize,
                        f64::from(FIXED_FPS),
                    )?;
                    RefPresentation {
                        kind: RefPresentationKind::Video {
                            blocks: sampled
                                .block_timestamps_seconds
                                .into_iter()
                                .map(|timestamp_seconds| VideoPresentationBlock {
                                    timestamp_seconds,
                                    vision_tokens: 3,
                                })
                                .collect(),
                        },
                        has_audio: reference.metadata.has_audio,
                    }
                }
            };
            Ok(H3ReferencePresentation {
                index: reference.metadata.index,
                presentation,
            })
        }

        fn encode_text(
            &mut self,
            _prompt: &str,
            references: &[H3ReferencePresentation],
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TextConditioning> {
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::QwenEncodeChunk,
                completed: 0,
                total: 1,
            })?;
            let mut tags = vec![H3ModalityTag::Text];
            for reference in references {
                tags.push(H3ModalityTag::Text);
                match &reference.presentation.kind {
                    RefPresentationKind::Audio => {}
                    RefPresentationKind::Image { vision_tokens } => {
                        tags.extend(std::iter::repeat_n(H3ModalityTag::Vision, *vision_tokens));
                    }
                    RefPresentationKind::Video { blocks } => {
                        for block in blocks {
                            tags.extend(std::iter::repeat_n(
                                H3ModalityTag::Vision,
                                block.vision_tokens,
                            ));
                        }
                    }
                }
            }
            Ok(H3TextConditioning {
                states: Tensor::zeros((1, tags.len(), TEXT_STATE_WIDTH), DType::F32, &self.device)?,
                tags,
            })
        }

        fn encode_visual_reference(
            &mut self,
            reference: &H3PreparedReference,
            mode: ConditionEncodeMode,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<Tensor> {
            assert_eq!(mode, ConditionEncodeMode::OfficialFreshSeed42);
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::ReferenceVisualEncodeChunk,
                completed: 0,
                total: 1,
            })?;
            self.visual_order.push(reference.metadata.index);
            let visual = reference_layout_spec(reference)?.visual.unwrap();
            Tensor::full(
                reference.metadata.index as f32 / 10.0,
                (
                    1,
                    VIDEO_LATENT_CHANNELS,
                    visual.latent_frames,
                    visual.latent_height,
                    visual.latent_width,
                ),
                &self.device,
            )
            .map_err(Into::into)
        }

        fn encode_audio_reference(
            &mut self,
            reference: &H3PreparedReference,
            mode: H3AudioConditionEncodeMode,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<StereoLatents> {
            assert_eq!(mode, H3AudioConditionEncodeMode::OfficialPosteriorModeF32);
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::ReferenceAudioEncodeChunk,
                completed: 0,
                total: 1,
            })?;
            self.audio_order.push(reference.metadata.index);
            let rows = reference.shape.audio_rows as usize / AUDIO_CHANNELS as usize;
            StereoLatents::new(
                Tensor::full(
                    reference.metadata.index as f32,
                    (1, AUDIO_LATENT_CHANNELS, AUDIO_CHANNELS as usize, rows),
                    &self.device,
                )?,
                &AudioVaeConfig::default(),
            )
            .map_err(Into::into)
        }

        fn denoise(
            &mut self,
            input: H3ForwardInput<'_>,
            _layout: &H3FrozenPackedLayout,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TransformerOutput> {
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::TransformerBlock,
                completed: 0,
                total: 1,
            })?;
            let video_prefix = input.video_rows.dims3()?.1 - 37;
            let audio_prefix = input.audio_rows.dims3()?.1 - 414;
            self.condition_video_checksums.push(
                input
                    .video_rows
                    .narrow(1, 0, video_prefix)?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
            self.condition_audio_checksums.push(
                input
                    .audio_rows
                    .narrow(1, 0, audio_prefix)?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
            Ok(H3TransformerOutput {
                video: Tensor::zeros_like(input.video_rows)?,
                audio: Tensor::zeros_like(input.audio_rows)?,
            })
        }

        fn decode_video(
            &mut self,
            latents: &Tensor,
            sink: &mut H3VideoEncodeSink,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            assert_eq!(latents.dims(), [1, 24, 37, 2, 2]);
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::VisualDecodeChunk,
                completed: 0,
                total: 1,
            })?;
            for frame in 0..124 {
                sink.push(
                    &RgbImage::from_pixel(32, 32, Rgb([frame as u8, 2, 3])),
                    checkpoint,
                )?;
            }
            Ok(())
        }

        fn decode_audio(
            &mut self,
            latents: &StereoLatents,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<StereoWaveform> {
            assert_eq!(latents.normalized().dims(), [1, 32, 2, 207]);
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::AudioDecodeChunk,
                completed: 0,
                total: 1,
            })?;
            StereoWaveform::new(
                Tensor::zeros(
                    (1, 2, 207 * AUDIO_SAMPLES_PER_LATENT),
                    DType::F32,
                    &self.device,
                )?,
                AUDIO_SAMPLE_RATE_HZ as usize,
                AudioSoundtrackAssociation::Generated,
            )
            .map_err(Into::into)
        }
    }

    #[test]
    fn preparation_freezes_order_shape_and_payload_free_fingerprint() {
        let original = request();
        let prepared = prepare(&original);
        assert_eq!(
            prepared
                .references
                .iter()
                .map(|reference| reference.metadata.kind)
                .collect::<Vec<_>>(),
            [
                GenerationReferenceKind::Video,
                GenerationReferenceKind::Image,
                GenerationReferenceKind::Audio
            ]
        );
        assert_eq!(
            prepared.references[0].shape.normalized_video_frames,
            Some(48)
        );
        assert_eq!(prepared.references[0].shape.video_frames, Some(39));
        assert_eq!(prepared.references[0].shape.qwen_video_frames, Some(4));
        assert_eq!(
            prepared.references[0].shape.audio_samples_per_channel,
            Some(64_000)
        );

        let metadata_json = serde_json::to_string(
            &prepared
                .references
                .iter()
                .map(|reference| &reference.metadata)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for secret in ["authority", "server_path", "handle", "/private/"] {
            assert!(!metadata_json.contains(secret));
        }

        let mut reordered = original.clone();
        reordered.references.as_mut().unwrap().swap(0, 1);
        let reordered = prepare(&reordered);
        assert_ne!(
            prepared.reference_fingerprint,
            reordered.reference_fingerprint
        );

        let mut mutated = original;
        mutated.prompt = "mutated after preparation".into();
        mutated.references.as_mut().unwrap().reverse();
        assert_eq!(prepared.prompt, "a brass automaton conducting an orchestra");
        assert_eq!(
            prepared.references[0].metadata.kind,
            GenerationReferenceKind::Video
        );
    }

    #[test]
    fn packed_layout_interleaves_reference_blocks_but_keeps_modality_indices_ordered() {
        let prepared = prepare(&request());
        let specs = prepared
            .references
            .iter()
            .map(reference_layout_spec)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let text = [H3ModalityTag::Text, H3ModalityTag::Vision];
        let packed = build_packed_sequence(&text, &prepared.geometry, &specs).unwrap();

        let video_audio_rows = specs[0].audio_latents_per_channel * 2;
        let video_visual_rows =
            specs[0].visual.unwrap().latent_frames * specs[0].visual.unwrap().rows_per_frame;
        let image_rows = specs[1].visual.unwrap().rows_per_frame;
        let standalone_audio_rows = specs[2].audio_latents_per_channel * 2;
        let video_audio_start = text.len();
        let video_visual_start = video_audio_start + video_audio_rows;
        let image_start = video_visual_start + video_visual_rows;
        let standalone_audio_start = image_start + image_rows;
        let target_audio_start = standalone_audio_start + standalone_audio_rows;
        let target_video_start = target_audio_start + prepared.geometry.generated_audio_rows;

        assert_eq!(packed.audio_indices[0] as usize, video_audio_start);
        assert_eq!(packed.video_indices[0] as usize, video_visual_start);
        assert_eq!(
            packed.video_indices[video_visual_rows] as usize,
            image_start
        );
        assert_eq!(
            packed.audio_indices[video_audio_rows] as usize,
            standalone_audio_start
        );
        assert_eq!(
            packed.audio_indices[video_audio_rows + standalone_audio_rows] as usize,
            target_audio_start
        );
        assert_eq!(
            packed.video_indices[video_visual_rows + image_rows] as usize,
            target_video_start
        );
        assert_eq!(
            packed.positions[video_audio_start][0],
            packed.positions[video_visual_start][0]
        );
        assert_eq!(
            packed.positions[target_audio_start][0],
            packed.positions[target_video_start][0]
        );
        assert!(
            packed.timestep_indices[video_audio_start..video_visual_start]
                .iter()
                .all(|index| *index == 3)
        );
        assert!(packed.timestep_indices[video_visual_start..target_audio_start].contains(&1));
        assert!(
            packed.timestep_indices[target_audio_start..target_video_start]
                .iter()
                .all(|index| *index == 2)
        );
    }

    #[test]
    fn execution_preserves_reference_prefixes_and_exact_noise_order() {
        let prepared = prepare(&request());
        let mut backend = SyntheticBackend::new();
        let staged = execute_staged(
            &prepared,
            &bindings(&prepared),
            &mut backend,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap();
        assert_eq!(backend.decoded_order, [1, 2, 3]);
        assert_eq!(backend.preprocessed_order, [1, 2, 3]);
        assert_eq!(backend.visual_order, [1, 2]);
        assert_eq!(backend.audio_order, [1, 3]);
        assert!(backend
            .condition_video_checksums
            .windows(2)
            .all(|pair| pair[0] == pair[1]));
        assert!(backend
            .condition_audio_checksums
            .windows(2)
            .all(|pair| pair[0] == pair[1]));
        assert_eq!(
            staged
                .provenance
                .noise_draws
                .iter()
                .map(|draw| draw.stream)
                .collect::<Vec<_>>(),
            [
                "condition-posterior",
                "condition-posterior",
                "condition-noise",
                "condition-noise",
                "target-video",
                "target-audio"
            ]
        );
        assert_eq!(
            staged
                .provenance
                .references
                .iter()
                .map(|reference| reference.kind)
                .collect::<Vec<_>>(),
            [
                GenerationReferenceKind::Video,
                GenerationReferenceKind::Image,
                GenerationReferenceKind::Audio
            ]
        );
        assert_eq!(
            staged.provenance.reference_fingerprint.as_deref(),
            Some(prepared.reference_fingerprint.as_str())
        );
        let json = serde_json::to_string(&staged.provenance).unwrap();
        for secret in ["authority", "handle", "server_path", "/private/"] {
            assert!(!json.contains(secret));
        }
        assert!(!staged.video_only_mp4.is_empty());
        assert!(!staged.thumbnail_png.is_empty());
    }

    #[test]
    fn private_reference_bindings_require_exact_count_order_and_provenance() {
        let prepared = prepare(&request());
        let mut missing = bindings(&prepared);
        missing.pop();
        let mut backend = SyntheticBackend::new();
        let error = execute_staged(
            &prepared,
            &missing,
            &mut backend,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap_err();
        assert!(error.to_string().contains("private media bindings"));
        assert!(backend.decoded_order.is_empty());

        let mut reordered = bindings(&prepared);
        reordered.swap(0, 1);
        let mut backend = SyntheticBackend::new();
        let error = execute_staged(
            &prepared,
            &reordered,
            &mut backend,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("differs from frozen reference provenance"));
        assert!(backend.decoded_order.is_empty());
    }

    #[test]
    fn every_long_ref2va_phase_has_a_cancellation_checkpoint() {
        let prepared = prepare(&request());
        for phase in [
            H3PipelinePhase::ReferenceDecodeChunk,
            H3PipelinePhase::ReferencePreprocessChunk,
            H3PipelinePhase::QwenEncodeChunk,
            H3PipelinePhase::ReferenceVisualEncodeChunk,
            H3PipelinePhase::ReferenceAudioEncodeChunk,
            H3PipelinePhase::TransformerBlock,
            H3PipelinePhase::VisualDecodeChunk,
            H3PipelinePhase::AudioDecodeChunk,
        ] {
            let cancellation = InferenceCancellationToken::default();
            let mut progress = ProgressReporter::default();
            progress.set_cancellation_token(cancellation.clone());
            let mut observer = RecordingObserver {
                cancel_at: Some(H3PipelineEvent {
                    phase,
                    completed: 0,
                    total: 1,
                }),
                cancellation: Some(cancellation),
                ..Default::default()
            };
            let error = execute_staged(
                &prepared,
                &bindings(&prepared),
                &mut SyntheticBackend::new(),
                &progress,
                &mut observer,
            )
            .unwrap_err();
            assert!(is_inference_cancelled(&error), "phase {phase:?}: {error:#}");
        }
    }

    #[test]
    fn frozen_backend_identity_and_packed_row_limit_fail_closed() {
        let prepared = prepare(&request());
        let mut rerouted = SyntheticBackend::new();
        rerouted.reroute_after_decode = true;
        let error = execute_staged(
            &prepared,
            &bindings(&prepared),
            &mut rerouted,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap_err();
        assert!(error.to_string().contains("implicit reroute is forbidden"));

        let mut denoise_reroute = SyntheticBackend::new();
        denoise_reroute.reroute_after_denoise = true;
        let error = execute_staged(
            &prepared,
            &bindings(&prepared),
            &mut denoise_reroute,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap_err();
        assert!(error.to_string().contains("implicit reroute is forbidden"));

        let mut too_small = SyntheticBackend::new();
        too_small.maximum_rows = 1;
        let error = execute_staged(
            &prepared,
            &bindings(&prepared),
            &mut too_small,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("exceeding the frozen backend limit"));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn mux_is_cancellable_before_publication() {
        let prepared = prepare(&request());
        let staged = execute_staged(
            &prepared,
            &bindings(&prepared),
            &mut SyntheticBackend::new(),
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap();
        let cancellation = InferenceCancellationToken::default();
        let mut progress = ProgressReporter::default();
        progress.set_cancellation_token(cancellation.clone());
        let mut observer = RecordingObserver {
            cancel_at: Some(H3PipelineEvent {
                phase: H3PipelinePhase::Mux,
                completed: 0,
                total: 1,
            }),
            cancellation: Some(cancellation),
            ..Default::default()
        };
        let error = match finalize_av(staged, &progress, &mut observer) {
            Ok(_) => panic!("cancelled Ref2VA mux unexpectedly published"),
            Err(error) => error,
        };
        assert!(is_inference_cancelled(&error));
    }
}
