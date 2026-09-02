//! Runtime-neutral T2VA/FL2VA orchestration for MiniMax H3.
//!
//! This module deliberately does not register an engine, discover artifacts,
//! or authorize H3 execution. It composes already-audited primitives behind a
//! single-device backend contract so tiny synthetic tests can prove the row,
//! noise, sampler, cancellation, and output lifecycle before the compliance,
//! attention, and admission gates permit a runnable family.

pub(crate) mod ref2va;

use std::io::Cursor;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{imageops, ImageReader, RgbImage};
use mold_candle::minimax_h3::{
    patchify_h3_video, unpack_h3_audio, unpatchify_h3_video, AudioSoundtrackAssociation,
    ConditionEncodeMode, EndpointResizePlan, H3ForwardInput, H3FrozenPackedLayout, H3Modality,
    H3ModalityTag, H3PackedLayout, H3TransformerOutput, StereoLatents, StereoWaveform,
    VisualTemporalGeometry,
};
use mold_core::minimax_h3::{
    self as contract, Mode, Task, AUDIO_CHANNELS, AUDIO_SAMPLE_RATE_HZ, CONDITION_POSTERIOR_SEED,
    FIXED_FPS, MAX_REFERENCE_DIMENSION, MAX_REFERENCE_IMAGE_PIXELS, NOISE_DOMAIN_VERSION,
};
use mold_core::GenerateRequest;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::Serialize;

use super::sampler::{H3DualSampler, H3DualSchedule, H3SamplerKind, H3_VISUAL_CONDITION_TIMESTEP};
use crate::engine::rand_seed;
use crate::ltx_video::video_enc::{self, Mp4StreamEncoder};
#[cfg(feature = "mp4")]
use crate::progress::InferenceCancelled;
use crate::progress::ProgressReporter;

const VIDEO_LATENT_CHANNELS: usize = 24;
const AUDIO_LATENT_CHANNELS: usize = 32;
const TEXT_STATE_WIDTH: usize = 5_120;
const VIDEO_VAE_SPATIAL_COMPRESSION: usize = 16;
const VIDEO_PATCH: [usize; 3] = [1, 2, 2];
const AUDIO_LATENTS_PER_SECOND: usize = 40;
const AUDIO_SAMPLES_PER_LATENT: usize = 800;

pub(crate) const SMALL_VIDEO_ONLY_MP4_MAX_BYTES: usize = 256 * 1024 * 1024;
pub(crate) const SMALL_THUMBNAIL_PNG_MAX_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const SMALL_FINAL_MP4_MAX_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const SMALL_ENCODED_VIDEO_HOST_BYTES_BOUND: u64 = 1024 * 1024 * 1024;
pub(crate) const SMALL_THUMBNAIL_HOST_BYTES_BOUND: u64 = 8 * 1024 * 1024;
pub(crate) const SMALL_MUX_OUTPUT_HOST_BYTES_BOUND: u64 = 512 * 1024 * 1024;
pub(crate) const SMALL_AAC_MUX_STAGING_HOST_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum H3EndpointAnchor {
    First,
    Last,
}

impl H3EndpointAnchor {
    fn as_str(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Last => "last",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum H3EndpointResize {
    Identity,
    GeometryAnchor(EndpointResizePlan),
    Follower(EndpointResizePlan),
}

#[derive(Clone, Debug)]
pub(crate) struct H3PreparedEndpoint {
    pub anchor: H3EndpointAnchor,
    pub source_width: u32,
    pub source_height: u32,
    pub resize: H3EndpointResize,
    /// Exact U8 `[1,3,1,H,W]` VAE/Qwen input on CPU.
    pub pixels: Tensor,
}

impl H3PreparedEndpoint {
    fn validate_for(&self, geometry: &H3Fl2VaGeometry) -> Result<()> {
        let (batch, channels, frames, height, width) = self.pixels.dims5()?;
        if self.pixels.dtype() != DType::U8
            || !self.pixels.device().is_cpu()
            || (batch, channels, frames) != (1, 3, 1)
            || (width, height) != (geometry.width, geometry.height)
        {
            bail!(
                "MiniMax H3 endpoint {:?} must be CPU U8 [1,3,1,{},{}], got {:?} {:?}",
                self.anchor,
                geometry.height,
                geometry.width,
                self.pixels.dtype(),
                self.pixels.dims()
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct H3Fl2VaGeometry {
    pub mode: Mode,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub latent_frames: usize,
    pub latent_width: usize,
    pub latent_height: usize,
    pub audio_latents_per_channel: usize,
    pub rows_per_video_frame: usize,
    pub condition_video_rows: usize,
    pub generated_video_rows: usize,
    pub generated_audio_rows: usize,
}

impl H3Fl2VaGeometry {
    fn from_request(req: &GenerateRequest, mode: Mode, endpoint_count: usize) -> Result<Self> {
        let width = usize::try_from(req.width).context("H3 width does not fit usize")?;
        let height = usize::try_from(req.height).context("H3 height does not fit usize")?;
        let frames = usize::try_from(req.frames.unwrap_or(contract::REVIEWED_COMPACT_FRAMES))
            .context("H3 frame count does not fit usize")?;
        let latent_frames = VisualTemporalGeometry::default().encoded_frames(frames)?;
        let latent_width = width / VIDEO_VAE_SPATIAL_COMPRESSION;
        let latent_height = height / VIDEO_VAE_SPATIAL_COMPRESSION;
        if !latent_width.is_multiple_of(VIDEO_PATCH[2])
            || !latent_height.is_multiple_of(VIDEO_PATCH[1])
        {
            bail!(
                "MiniMax H3 latent canvas {latent_width}x{latent_height} is not divisible by the transformer patch"
            );
        }
        let rows_per_video_frame = checked_product(
            &[
                latent_height / VIDEO_PATCH[1],
                latent_width / VIDEO_PATCH[2],
            ],
            "rows per H3 video frame",
        )?;
        let condition_video_rows = endpoint_count
            .checked_mul(rows_per_video_frame)
            .ok_or_else(|| anyhow!("H3 endpoint row count overflow"))?;
        let generated_video_rows = latent_frames
            .checked_mul(rows_per_video_frame)
            .ok_or_else(|| anyhow!("H3 generated video row count overflow"))?;
        let audio_latents_per_channel = rounded_ratio(
            frames,
            AUDIO_LATENTS_PER_SECOND,
            usize::try_from(FIXED_FPS).expect("fixed fps fits usize"),
        )?;
        let generated_audio_rows = audio_latents_per_channel
            .checked_mul(usize::try_from(AUDIO_CHANNELS).expect("audio channels fit usize"))
            .ok_or_else(|| anyhow!("H3 generated audio row count overflow"))?;
        Ok(Self {
            mode,
            width,
            height,
            frames,
            latent_frames,
            latent_width,
            latent_height,
            audio_latents_per_channel,
            rows_per_video_frame,
            condition_video_rows,
            generated_video_rows,
            generated_audio_rows,
        })
    }

    fn validate_mode_endpoints(&self, endpoints: &[H3PreparedEndpoint]) -> Result<()> {
        let anchors = endpoints
            .iter()
            .map(|endpoint| endpoint.anchor)
            .collect::<Vec<_>>();
        let expected: &[H3EndpointAnchor] = match self.mode {
            Mode::TextToAudioVideo => &[],
            Mode::FirstFrameToAudioVideo => &[H3EndpointAnchor::First],
            Mode::LastFrameToAudioVideo => &[H3EndpointAnchor::Last],
            Mode::FirstAndLastFrameToAudioVideo => {
                &[H3EndpointAnchor::First, H3EndpointAnchor::Last]
            }
            Mode::ReferenceToAudioVideo => {
                bail!("Ref2VA cannot enter the T2VA/FL2VA pipeline")
            }
        };
        if anchors != expected {
            bail!(
                "MiniMax H3 {:?} endpoint order is {:?}, expected {:?}",
                self.mode,
                anchors,
                expected
            );
        }
        for endpoint in endpoints {
            endpoint.validate_for(self)?;
        }
        Ok(())
    }

    fn generated_video_shape(&self) -> [usize; 5] {
        [
            1,
            VIDEO_LATENT_CHANNELS,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
        ]
    }

    fn generated_audio_row_shape(&self) -> [usize; 3] {
        [1, self.generated_audio_rows, AUDIO_LATENT_CHANNELS]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct H3PreparedFl2VaRequest {
    pub geometry: H3Fl2VaGeometry,
    pub endpoints: Vec<H3PreparedEndpoint>,
    pub prompt: String,
    pub seed: u64,
    pub grid_points: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum H3PipelinePhase {
    Validate,
    EndpointPreprocess,
    VaeLoad,
    VaeLoadChunk,
    ReferenceDecode,
    ReferenceDecodeChunk,
    ReferencePreprocess,
    ReferencePreprocessChunk,
    QwenLoad,
    QwenLoadChunk,
    QwenEncode,
    QwenEncodeChunk,
    /// The conditioner output was served from the in-process cache. It
    /// REPLACES `QwenLoad`/`QwenEncode` on a hit rather than joining them, so
    /// nothing downstream can mistake a served answer for a real encode.
    QwenConditioningCached,
    PromptEncode,
    VisualConditionEncode,
    VisualConditionEncodeChunk,
    ReferenceVisualEncode,
    ReferenceVisualEncodeChunk,
    ReferenceAudioEncode,
    ReferenceAudioEncodeChunk,
    NoiseAllocation,
    TransformerLoad,
    Denoise,
    TransformerBlock,
    VisualDecode,
    VisualDecodeChunk,
    AudioDecode,
    AudioDecodeChunk,
    VideoEncode,
    Mux,
    Staged,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct H3PipelineEvent {
    pub phase: H3PipelinePhase,
    pub completed: usize,
    pub total: usize,
}

pub(crate) trait H3PipelineObserver {
    fn observe(&mut self, event: H3PipelineEvent);
}

#[derive(Default)]
pub(crate) struct NoopH3PipelineObserver;

impl H3PipelineObserver for NoopH3PipelineObserver {
    fn observe(&mut self, _event: H3PipelineEvent) {}
}

pub(crate) trait H3PipelineCheckpoint {
    fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()>;
}

struct PipelineControl<'a> {
    progress: &'a ProgressReporter,
    observer: &'a mut dyn H3PipelineObserver,
}

impl H3PipelineCheckpoint for PipelineControl<'_> {
    fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()> {
        if event.total == 0 || event.completed > event.total {
            bail!("invalid MiniMax H3 pipeline progress event {event:?}");
        }
        self.progress.checkpoint()?;
        self.observer.observe(event);
        self.progress.checkpoint()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum H3PipelineBackendKind {
    Cuda,
    /// Apple Silicon's correctness route (#1164).
    Metal,
    /// This class is accepted only by `cfg(test)` unit binaries.
    SyntheticCpu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct H3PipelineBackendIdentity {
    pub kind: H3PipelineBackendKind,
    pub device_id: String,
    pub execution_fingerprint: String,
}

impl H3PipelineBackendIdentity {
    fn validate(&self, device: &Device) -> Result<()> {
        if self.device_id.trim().is_empty() || self.execution_fingerprint.trim().is_empty() {
            bail!("MiniMax H3 requires nonempty device and execution identities");
        }
        match self.kind {
            H3PipelineBackendKind::Cuda if device.is_cuda() => Ok(()),
            H3PipelineBackendKind::Cuda => {
                bail!("MiniMax H3 production pipeline requires one CUDA device")
            }
            H3PipelineBackendKind::Metal if device.is_metal() => Ok(()),
            H3PipelineBackendKind::Metal => {
                bail!("MiniMax H3 Metal pipeline requires one Metal device")
            }
            H3PipelineBackendKind::SyntheticCpu if cfg!(test) && device.is_cpu() => Ok(()),
            H3PipelineBackendKind::SyntheticCpu => {
                bail!("MiniMax H3 synthetic CPU execution is restricted to unit-test binaries")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct H3TextConditioning {
    pub states: Tensor,
    pub tags: Vec<H3ModalityTag>,
    #[cfg(test)]
    pub lifetime_probe: Option<std::sync::Arc<()>>,
}

impl H3TextConditioning {
    fn validate(&self, device: &Device) -> Result<()> {
        let (batch, rows, width) = self.states.dims3()?;
        if batch != 1 || rows == 0 || width != TEXT_STATE_WIDTH || self.tags.len() != rows {
            bail!(
                "MiniMax H3 text conditioning must be [1,rows,{TEXT_STATE_WIDTH}] with one tag per row"
            );
        }
        if !matches!(
            self.states.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ) || !self.states.device().same_device(device)
        {
            bail!("MiniMax H3 text conditioning dtype/device does not match the frozen route");
        }
        Ok(())
    }
}

/// One backend owns every H3 component and exposes exactly one device.
///
/// The real adapter will be introduced only after compliance, attention, and
/// admission are approved. Keeping the orchestration generic lets CI run a
/// weight-free synthetic adapter without a second implementation of the
/// pipeline math.
pub(crate) trait H3Fl2VaBackend {
    fn identity(&self) -> H3PipelineBackendIdentity;
    fn device(&self) -> &Device;

    fn sampler_kind(&self) -> H3SamplerKind {
        H3SamplerKind::OfficialEuler
    }

    /// Video shift the sigma grid is built with. Only a reviewed Turbo tier
    /// moves it (the 768p 4-step checkpoint is documented at shift 6); every
    /// other backend keeps the family constant.
    fn sampler_video_shift(&self) -> f32 {
        super::sampler::H3_VIDEO_SHIFT
    }

    fn encode_text(
        &mut self,
        prompt: &str,
        endpoints: &[H3PreparedEndpoint],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning>;

    /// Encode one endpoint with the requested posterior authority. The H3
    /// T2VA/FL2VA orchestrator always supplies
    /// [`ConditionEncodeMode::OfficialFreshSeed42`], which includes the
    /// independent fresh seed-42 posterior draw and FP16 round-trip before
    /// latent normalization. An adapter must not substitute ComfyUI's
    /// mean-only shortcut.
    fn encode_visual_condition(
        &mut self,
        endpoint: &H3PreparedEndpoint,
        mode: ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor>;

    /// Release every component that condition encoding was the last consumer
    /// of, before the orchestrator allocates noise and the backend loads the
    /// transformer. Nothing between here and visual decode reads a VAE, so a
    /// backend that keeps both VAEs device-resident through denoise pays their
    /// weights inside the transformer's own peak. Backends that hold no such
    /// component keep the default no-op; a backend that parks here must be
    /// able to reconstruct before `decode_video`.
    fn park_condition_components(
        &mut self,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        let _ = checkpoint;
        Ok(())
    }

    /// Borrowed inputs are call-scoped. Implementations must not clone or
    /// retain text states after returning; the orchestrator releases them
    /// immediately after the final forward to satisfy the frozen phase plan.
    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput>;

    /// Decode into the supplied bounded H.264 sink. Implementations must call
    /// `checkpoint` between temporal/tile chunks; the sink checks every frame.
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct H3NoiseDrawMetadata {
    pub stream: &'static str,
    pub ordinal: usize,
    pub shape: Vec<usize>,
    pub elements: usize,
    pub seed: u64,
    /// `true` for the independent seed-42 posterior generator recreated for
    /// every visual condition; request streams share one generator.
    pub fresh_generator: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct H3PipelineProvenance {
    pub mode: &'static str,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: u32,
    pub video_latent_frames: usize,
    pub audio_latents_per_channel: usize,
    pub audio_sample_rate: u32,
    pub audio_channels: u32,
    pub requested_grid_points: usize,
    pub transformer_evaluations: usize,
    pub sampler: &'static str,
    pub endpoint_anchors: Vec<H3EndpointAnchor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<H3ReferenceProvenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_fingerprint: Option<String>,
    pub noise_domain_version: &'static str,
    pub noise_draws: Vec<H3NoiseDrawMetadata>,
    pub device_id: String,
    pub execution_fingerprint: String,
}

/// Payload-free Ref2VA provenance. It intentionally cannot represent media
/// bytes, upload handles, or local filesystem paths.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct H3ReferenceProvenance {
    pub index: u32,
    pub kind: mold_core::GenerationReferenceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_samples_per_channel: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_samples_per_channel: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct H3StagedAvOutput {
    /// H.264-only MP4. The AAC track is attached atomically by `finalize_av`.
    pub video_only_mp4: Vec<u8>,
    pub thumbnail_png: Vec<u8>,
    pub waveform: StereoWaveform,
    pub provenance: H3PipelineProvenance,
}

#[cfg(feature = "mp4")]
pub(crate) struct H3PipelineOutput {
    pub mp4: Vec<u8>,
    pub thumbnail_png: Vec<u8>,
    pub provenance: H3PipelineProvenance,
    pub mux_report: crate::av_media::AacMuxReport,
}

struct H3RequestNoise {
    seed: u64,
    rng: StdRng,
    draws: Vec<H3NoiseDrawMetadata>,
}

impl H3RequestNoise {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            rng: StdRng::seed_from_u64(seed),
            draws: Vec::new(),
        }
    }

    fn draw(
        &mut self,
        stream: &'static str,
        ordinal: usize,
        shape: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let elements = checked_product(shape, "H3 random draw")?;
        let values = (0..elements)
            .map(|_| StandardNormal.sample(&mut self.rng))
            .collect::<Vec<f32>>();
        self.draws.push(H3NoiseDrawMetadata {
            stream,
            ordinal,
            shape: shape.to_vec(),
            elements,
            seed: self.seed,
            fresh_generator: false,
        });
        Tensor::from_vec(values, shape, &Device::Cpu)?
            .to_device(device)
            .map_err(Into::into)
    }
}

struct PackedSequence {
    layout: H3PackedLayout,
    condition_video_rows: usize,
    generated_video_rows: usize,
    generated_audio_rows: usize,
}

pub(crate) fn prepare_request(
    req: &GenerateRequest,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3PreparedFl2VaRequest> {
    // Keep endpoint decode/resize and prepared noise on the host side of the
    // allocation fence. Every Candle tensor constructed by this path uses
    // `Device::Cpu`; execution transfers happen only after CUDA allocation is
    // committed by the private runner.
    let mut control = PipelineControl { progress, observer };
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::Validate,
        completed: 0,
        total: 1,
    })?;
    let mode = contract::validate_request_contract(req, Task::Fl2va)
        .map_err(|error| anyhow!("{}: {}", error.code, error.message))?;
    if req.batch_size != 1 {
        bail!(
            "MiniMax H3 pipeline consumes one singleton sibling; batch size {} must be split by the authoritative batch router first",
            req.batch_size
        );
    }
    let raw = collect_endpoint_bytes(req, mode)?;
    let geometry = H3Fl2VaGeometry::from_request(req, mode, raw.len())?;
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::Validate,
        completed: 1,
        total: 1,
    })?;

    let total = raw.len().max(1);
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::EndpointPreprocess,
        completed: 0,
        total,
    })?;
    let mut endpoints = Vec::with_capacity(raw.len());
    for (index, (anchor, bytes)) in raw.into_iter().enumerate() {
        let decoded = decode_endpoint(bytes)
            .with_context(|| format!("failed to decode {} H3 endpoint", anchor.as_str()))?;
        let prepared = resize_endpoint(decoded, anchor, index, &geometry, progress)?;
        endpoints.push(prepared);
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::EndpointPreprocess,
            completed: index + 1,
            total,
        })?;
    }
    if endpoints.is_empty() {
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::EndpointPreprocess,
            completed: 1,
            total,
        })?;
    }
    geometry.validate_mode_endpoints(&endpoints)?;

    Ok(H3PreparedFl2VaRequest {
        geometry,
        endpoints,
        prompt: req.prompt.clone(),
        seed: req.seed.unwrap_or_else(rand_seed),
        grid_points: usize::try_from(req.steps).context("H3 grid points do not fit usize")?,
    })
}

pub(crate) fn execute_staged(
    prepared: &H3PreparedFl2VaRequest,
    backend: &mut dyn H3Fl2VaBackend,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3StagedAvOutput> {
    prepared
        .geometry
        .validate_mode_endpoints(&prepared.endpoints)?;
    let frozen_identity = backend.identity();
    frozen_identity.validate(backend.device())?;
    let device = backend.device().clone();
    let mut control = PipelineControl { progress, observer };

    phase_boundary(&mut control, H3PipelinePhase::PromptEncode, false)?;
    ensure_identity(backend, &frozen_identity, &device)?;
    let text = backend.encode_text(&prepared.prompt, &prepared.endpoints, &mut control)?;
    ensure_identity(backend, &frozen_identity, &device)?;
    text.validate(&device)?;
    phase_boundary(&mut control, H3PipelinePhase::PromptEncode, true)?;

    let anchors = prepared
        .endpoints
        .iter()
        .map(|endpoint| endpoint.anchor)
        .collect::<Vec<_>>();
    let packed = build_packed_sequence(&text.tags, &prepared.geometry, &anchors)?;
    let frozen_layout = packed.layout.freeze(&device)?;

    let visual_total = prepared.endpoints.len().max(1);
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::VisualConditionEncode,
        completed: 0,
        total: visual_total,
    })?;
    let mut conditions = Vec::with_capacity(prepared.endpoints.len());
    for (index, endpoint) in prepared.endpoints.iter().enumerate() {
        ensure_identity(backend, &frozen_identity, &device)?;
        let condition = backend.encode_visual_condition(
            endpoint,
            ConditionEncodeMode::OfficialFreshSeed42,
            &mut control,
        )?;
        ensure_identity(backend, &frozen_identity, &device)?;
        validate_condition_latent(&condition, &prepared.geometry)?;
        conditions.push(condition);
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::VisualConditionEncode,
            completed: index + 1,
            total: visual_total,
        })?;
    }
    if conditions.is_empty() {
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::VisualConditionEncode,
            completed: 1,
            total: visual_total,
        })?;
    }

    ensure_identity(backend, &frozen_identity, &device)?;
    backend.park_condition_components(&mut control)?;
    ensure_identity(backend, &frozen_identity, &device)?;

    let mut posterior_draws = conditions
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

    let draw_total = conditions.len() + 2;
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::NoiseAllocation,
        completed: 0,
        total: draw_total,
    })?;
    let mut request_noise = H3RequestNoise::new(prepared.seed);
    let mut condition_rows = Vec::with_capacity(conditions.len());
    for (index, condition) in conditions.iter().enumerate() {
        let noise = request_noise.draw("condition-noise", index, condition.dims(), &device)?;
        let condition = condition.to_device(&device)?.to_dtype(DType::F32)?;
        let noised = condition
            .affine(f64::from(H3_VISUAL_CONDITION_TIMESTEP), 0.0)?
            .add(&noise.affine(f64::from(1.0 - H3_VISUAL_CONDITION_TIMESTEP), 0.0)?)?;
        condition_rows.push(patchify_h3_video(&noised, VIDEO_PATCH)?);
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::NoiseAllocation,
            completed: index + 1,
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
        completed: conditions.len() + 1,
        total: draw_total,
    })?;
    let mut audio_rows = request_noise.draw(
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

    let mut video_rows = if condition_rows.is_empty() {
        generated_video
    } else {
        let mut parts = condition_rows.iter().collect::<Vec<_>>();
        parts.push(&generated_video);
        Tensor::cat(&parts, 1)?
    };
    validate_packed_rows(&video_rows, &audio_rows, &text.states, &packed)?;

    let sampler_kind = backend.sampler_kind();
    let schedule = H3DualSchedule::new_for_sampler_with_video_shift(
        prepared.grid_points,
        sampler_kind,
        backend.sampler_video_shift(),
    )?;
    let mut sampler = H3DualSampler::new(sampler_kind);
    let counts = schedule.counts();
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::Denoise,
        completed: 0,
        total: counts.transformer_evaluations,
    })?;
    for step in schedule.steps() {
        ensure_identity(backend, &frozen_identity, &device)?;
        let timesteps = Tensor::from_slice(
            &[
                step.row_timesteps.generated_video,
                step.row_timesteps.visual_condition,
                step.row_timesteps.generated_audio,
            ],
            3,
            &device,
        )?;
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
        ensure_identity(backend, &frozen_identity, &device)?;
        validate_transformer_output(&output, &video_rows, &audio_rows)?;

        let generated =
            video_rows.narrow(1, packed.condition_video_rows, packed.generated_video_rows)?;
        let generated_velocity =
            output
                .video
                .narrow(1, packed.condition_video_rows, packed.generated_video_rows)?;
        let (next_video, next_audio) = sampler.step_pair(
            &generated,
            &audio_rows,
            &generated_velocity,
            &output.audio,
            step,
        )?;
        video_rows = if packed.condition_video_rows == 0 {
            next_video
        } else {
            Tensor::cat(
                &[
                    &video_rows.narrow(1, 0, packed.condition_video_rows)?,
                    &next_video,
                ],
                1,
            )?
        };
        audio_rows = next_audio;
        let progress_after = step.progress_after_update();
        control.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::Denoise,
            completed: progress_after.completed_evaluations,
            total: progress_after.total_evaluations,
        })?;
    }
    // RES history is denoise-only workspace. Release previous clean estimates
    // and the carried audio state before transformer teardown and VAE decode.
    drop(sampler);
    // Text states are borrowed by every transformer forward but no decoder.
    // End their device residency at the exact denoise boundary on success;
    // ordinary unwinding provides the same guarantee for cancellation/errors.
    drop(text);

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
    let audio_latents = unpack_h3_audio(
        &audio_rows,
        usize::try_from(AUDIO_CHANNELS).expect("audio channels fit usize"),
    )?;
    let audio_latents = StereoLatents::new(
        audio_latents.to_dtype(DType::F32)?,
        &mold_candle::minimax_h3::AudioVaeConfig::default(),
    )?;

    phase_boundary(&mut control, H3PipelinePhase::VisualDecode, false)?;
    ensure_identity(backend, &frozen_identity, &device)?;
    let mut sink = H3VideoEncodeSink::new(&prepared.geometry)?;
    backend.decode_video(&video_latents, &mut sink, &mut control)?;
    ensure_identity(backend, &frozen_identity, &device)?;
    let encoded_video = sink.finish()?;
    phase_boundary(&mut control, H3PipelinePhase::VisualDecode, true)?;

    phase_boundary(&mut control, H3PipelinePhase::AudioDecode, false)?;
    ensure_identity(backend, &frozen_identity, &device)?;
    let waveform = backend.decode_audio(&audio_latents, &mut control)?;
    ensure_identity(backend, &frozen_identity, &device)?;
    validate_waveform(&waveform, prepared.geometry.audio_latents_per_channel)?;
    phase_boundary(&mut control, H3PipelinePhase::AudioDecode, true)?;

    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::Staged,
        completed: 1,
        total: 1,
    })?;
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    super::private_runtime_observer::observe_staged_host_bytes(
        encoded_video.mp4.capacity(),
        encoded_video.thumbnail_png.capacity(),
    )?;
    Ok(H3StagedAvOutput {
        video_only_mp4: encoded_video.mp4,
        thumbnail_png: encoded_video.thumbnail_png,
        waveform,
        provenance: H3PipelineProvenance {
            mode: mode_name(prepared.geometry.mode),
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
            sampler: sampler_kind.as_str(),
            endpoint_anchors: anchors,
            references: Vec::new(),
            reference_fingerprint: None,
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

#[cfg(feature = "mp4")]
pub(crate) fn finalize_av(
    staged: H3StagedAvOutput,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3PipelineOutput> {
    use crate::av_media::{mux_aac_track_to_mp4_bytes_bounded, AacMuxOptions, AvMuxLimits};

    let mut control = PipelineControl { progress, observer };
    phase_boundary(&mut control, H3PipelinePhase::Mux, false)?;
    let samples = interleaved_pcm(&staged.waveform)?;
    let sample_bytes = samples
        .capacity()
        .checked_mul(std::mem::size_of::<f32>())
        .context("H3 interleaved PCM capacity overflowed")?;
    let aac_staging_limit = usize::try_from(SMALL_AAC_MUX_STAGING_HOST_BYTES)
        .context("H3 AAC staging limit exceeded usize")?
        .checked_sub(sample_bytes)
        .context("H3 interleaved PCM exceeds the AAC staging limit")?;
    let muxed = mux_aac_track_to_mp4_bytes_bounded(
        &staged.video_only_mp4,
        &samples,
        AUDIO_SAMPLE_RATE_HZ,
        u16::try_from(AUDIO_CHANNELS).expect("audio channels fit u16"),
        AacMuxOptions::match_video(),
        AvMuxLimits {
            output_bytes: SMALL_FINAL_MP4_MAX_BYTES,
            aac_staging_bytes: aac_staging_limit,
        },
        &|_| progress.checkpoint().is_err(),
    )
    .map_err(|error| {
        if progress.checkpoint().is_err() {
            anyhow::Error::new(InferenceCancelled)
        } else {
            error
        }
    })?;
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    super::private_runtime_observer::observe_aac_staging_capacity(
        samples.capacity(),
        muxed.report.peak_aac_staging_host_bytes,
    )?;
    phase_boundary(&mut control, H3PipelinePhase::Mux, true)?;
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    super::private_runtime_observer::observe_mux_output_capacity(muxed.bytes.capacity())?;
    control.checkpoint(H3PipelineEvent {
        phase: H3PipelinePhase::Complete,
        completed: 1,
        total: 1,
    })?;
    Ok(H3PipelineOutput {
        mp4: muxed.bytes,
        thumbnail_png: staged.thumbnail_png,
        provenance: staged.provenance,
        mux_report: muxed.report,
    })
}

pub(crate) struct H3VideoEncodeSink {
    encoder: Option<Mp4StreamEncoder>,
    expected_frames: usize,
    width: u32,
    height: u32,
    pushed: usize,
    first_frame: Option<RgbImage>,
}

impl H3VideoEncodeSink {
    fn new(geometry: &H3Fl2VaGeometry) -> Result<Self> {
        Self::new_dimensions(geometry.width, geometry.height, geometry.frames)
    }

    pub(super) fn new_dimensions(
        width: usize,
        height: usize,
        expected_frames: usize,
    ) -> Result<Self> {
        let width = u32::try_from(width).context("H3 width does not fit u32")?;
        let height = u32::try_from(height).context("H3 height does not fit u32")?;
        Ok(H3VideoEncodeSink {
            encoder: Some(Mp4StreamEncoder::new_bounded(
                width,
                height,
                FIXED_FPS,
                SMALL_VIDEO_ONLY_MP4_MAX_BYTES,
            )?),
            expected_frames,
            width,
            height,
            pushed: 0,
            first_frame: None,
        })
    }

    pub(crate) fn push(
        &mut self,
        frame: &RgbImage,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        if self.pushed >= self.expected_frames {
            bail!("MiniMax H3 visual decoder emitted more frames than planned");
        }
        if frame.dimensions() != (self.width, self.height) {
            bail!(
                "MiniMax H3 visual decoder emitted {}x{}, expected {}x{}",
                frame.width(),
                frame.height(),
                self.width,
                self.height
            );
        }
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::VideoEncode,
            completed: self.pushed,
            total: self.expected_frames,
        })?;
        if self.first_frame.is_none() {
            self.first_frame = Some(frame.clone());
        }
        self.encoder
            .as_mut()
            .expect("encoder is present until finish")
            .push(frame)?;
        self.pushed += 1;
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::VideoEncode,
            completed: self.pushed,
            total: self.expected_frames,
        })
    }

    fn finish(mut self) -> Result<EncodedVideo> {
        if self.pushed != self.expected_frames {
            bail!(
                "MiniMax H3 visual decoder emitted {} frames, expected {}",
                self.pushed,
                self.expected_frames
            );
        }
        let first = self
            .first_frame
            .take()
            .ok_or_else(|| anyhow!("MiniMax H3 visual decoder emitted no frames"))?;
        let thumbnail_png = video_enc::first_frame_png_bounded(
            std::slice::from_ref(&first),
            SMALL_THUMBNAIL_PNG_MAX_BYTES,
        )?;
        Ok(EncodedVideo {
            mp4: self
                .encoder
                .take()
                .expect("encoder is present until finish")
                .finish()?,
            thumbnail_png,
        })
    }
}

struct EncodedVideo {
    mp4: Vec<u8>,
    thumbnail_png: Vec<u8>,
}

/// Solid endpoint placeholder for redacted placement probes.
///
/// Placement previews redact media, so a present FL2VA endpoint arrives as
/// zero bytes. Admission decodes endpoints for real submissions, which made
/// every H3 preview refuse with "endpoint is empty" even though the client
/// validated the real image locally. The probe substitutes this placeholder
/// at the request's exact target geometry so the preview prices the same
/// shape; real admission at submit still decodes the real bytes.
pub fn placeholder_endpoint_png(width: u32, height: u32) -> Vec<u8> {
    let width = width.clamp(1, MAX_REFERENCE_DIMENSION);
    let height = height.clamp(1, MAX_REFERENCE_DIMENSION);
    // Keep the placeholder inside the bounded image contract even for a
    // pathological request shape — admission must reject the SHAPE, not
    // choke on an undecodable probe substitute.
    let height = height
        .min(u32::try_from(MAX_REFERENCE_IMAGE_PIXELS / u64::from(width)).unwrap_or(u32::MAX));
    let height = height.max(1);
    let image = RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .expect("in-memory PNG encoding of a bounded solid image cannot fail");
    bytes.into_inner()
}

pub(crate) fn collect_endpoint_bytes(
    req: &GenerateRequest,
    mode: Mode,
) -> Result<Vec<(H3EndpointAnchor, &[u8])>> {
    let last_frame = req.frames.unwrap_or(contract::REVIEWED_COMPACT_FRAMES) - 1;
    let mut first = req.source_image.as_deref();
    let mut last = None;
    for keyframe in req.keyframes.as_deref().unwrap_or_default() {
        match keyframe.frame {
            0 => first = Some(&keyframe.image),
            frame if frame == last_frame => last = Some(&keyframe.image),
            _ => bail!("MiniMax H3 request contract admitted a non-endpoint keyframe"),
        }
    }
    let mut endpoints = Vec::with_capacity(2);
    if let Some(bytes) = first {
        if bytes.is_empty() {
            bail!("MiniMax H3 first-frame endpoint is empty");
        }
        endpoints.push((H3EndpointAnchor::First, bytes));
    }
    if let Some(bytes) = last {
        if bytes.is_empty() {
            bail!("MiniMax H3 last-frame endpoint is empty");
        }
        endpoints.push((H3EndpointAnchor::Last, bytes));
    }
    let expected = match mode {
        Mode::TextToAudioVideo => 0,
        Mode::FirstFrameToAudioVideo | Mode::LastFrameToAudioVideo => 1,
        Mode::FirstAndLastFrameToAudioVideo => 2,
        Mode::ReferenceToAudioVideo => bail!("Ref2VA cannot enter FL2VA endpoint collection"),
    };
    if endpoints.len() != expected {
        bail!(
            "MiniMax H3 mode {:?} resolved {} endpoints, expected {expected}",
            mode,
            endpoints.len()
        );
    }
    Ok(endpoints)
}

fn decode_endpoint(bytes: &[u8]) -> Result<RgbImage> {
    let dimensions_reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("unknown endpoint image format")?;
    let (width, height) = dimensions_reader
        .into_dimensions()
        .context("failed to inspect endpoint dimensions")?;
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| anyhow!("endpoint pixel count overflow"))?;
    if width == 0
        || height == 0
        || width > MAX_REFERENCE_DIMENSION
        || height > MAX_REFERENCE_DIMENSION
        || pixels > MAX_REFERENCE_IMAGE_PIXELS
    {
        bail!("MiniMax H3 endpoint dimensions {width}x{height} exceed the bounded image contract");
    }

    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("unknown endpoint image format")?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_REFERENCE_DIMENSION);
    limits.max_image_height = Some(MAX_REFERENCE_DIMENSION);
    limits.max_alloc = Some(MAX_REFERENCE_IMAGE_PIXELS.saturating_mul(4));
    reader.limits(limits);
    Ok(reader
        .decode()
        .context("endpoint image decode failed")?
        .to_rgb8())
}

const PILLOW_RESAMPLE_PRECISION_BITS: u32 = 22;

struct PillowResampleCoefficients {
    start: usize,
    values: Vec<i32>,
}

fn pillow_lanczos(x: f64) -> f64 {
    fn sinc(mut x: f64) -> f64 {
        if x == 0.0 {
            return 1.0;
        }
        x *= std::f64::consts::PI;
        x.sin() / x
    }

    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Pillow's U8 LANCZOS coefficient generation and fixed-point rounding.
///
/// H3's pinned preprocessing authority uses `PIL.Image.Resampling.LANCZOS`.
/// The `image` crate's similarly named Lanczos3 filter uses different edge and
/// quantization rules, which changes the endpoint tensor before seed-42 VAE
/// sampling. See Diffusers
/// `src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:134-158` at
/// `9c6a68c32b3b2a64db91800b624d33cec6e25ab8` and Pillow 12.3.0
/// (`bb1d8e8ab8d29048624d96e3ee53cecf7c13d13d`)
/// `src/libImaging/Resample.c:65-87,183-284,344-363,446-463`. The Pillow
/// attribution and license are preserved in `THIRD_PARTY_NOTICES.md`.
fn pillow_resample_coefficients(
    input: usize,
    output: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<PillowResampleCoefficients>> {
    if input == 0 || output == 0 {
        bail!("Pillow-compatible H3 resize dimensions must be non-zero");
    }
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    let coefficient_scale = f64::from(1_u32 << PILLOW_RESAMPLE_PRECISION_BITS);

    (0..output)
        .map(|destination| {
            checkpoint()?;
            let center = (destination as f64 + 0.5) * scale;
            // Match Pillow's C casts, which truncate toward zero before
            // clamping the bounds to the source image.
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as isize).clamp(0, input as isize) as usize;
            if end <= start {
                bail!("Pillow-compatible H3 resize produced an empty filter window");
            }
            let mut weights = (start..end)
                .map(|source| pillow_lanczos((source as f64 - center + 0.5) / filter_scale))
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            if sum != 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                }
            }
            let values = weights
                .into_iter()
                .map(|weight| {
                    let scaled = weight * coefficient_scale;
                    if weight < 0.0 {
                        (scaled - 0.5) as i32
                    } else {
                        (scaled + 0.5) as i32
                    }
                })
                .collect();
            Ok(PillowResampleCoefficients { start, values })
        })
        .collect()
}

fn pillow_resample_channel(samples: impl Iterator<Item = (u8, i32)>) -> u8 {
    let accumulator = samples.fold(
        1_i64 << (PILLOW_RESAMPLE_PRECISION_BITS - 1),
        |total, (sample, coefficient)| total + i64::from(sample) * i64::from(coefficient),
    );
    (accumulator >> PILLOW_RESAMPLE_PRECISION_BITS).clamp(0, 255) as u8
}

pub(super) fn pillow_lanczos_resize(
    source: &RgbImage,
    width: u32,
    height: u32,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<RgbImage> {
    let source_width =
        usize::try_from(source.width()).context("source width does not fit usize")?;
    let source_height =
        usize::try_from(source.height()).context("source height does not fit usize")?;
    let target_width = usize::try_from(width).context("target width does not fit usize")?;
    let target_height = usize::try_from(height).context("target height does not fit usize")?;
    let source_bytes = source.as_raw();

    let horizontal = if source_width == target_width {
        source_bytes.clone()
    } else {
        let coefficients = pillow_resample_coefficients(source_width, target_width, checkpoint)?;
        let output_len = checked_product(
            &[target_width, source_height, 3],
            "Pillow-compatible H3 horizontal resize",
        )?;
        let mut output = vec![0_u8; output_len];
        for y in 0..source_height {
            checkpoint()?;
            for (x, filter) in coefficients.iter().enumerate() {
                for channel in 0..3 {
                    output[(y * target_width + x) * 3 + channel] =
                        pillow_resample_channel(filter.values.iter().enumerate().map(
                            |(offset, &coefficient)| {
                                (
                                    source_bytes
                                        [(y * source_width + filter.start + offset) * 3 + channel],
                                    coefficient,
                                )
                            },
                        ));
                }
            }
        }
        output
    };

    let output = if source_height == target_height {
        horizontal
    } else {
        let coefficients = pillow_resample_coefficients(source_height, target_height, checkpoint)?;
        let output_len = checked_product(
            &[target_width, target_height, 3],
            "Pillow-compatible H3 vertical resize",
        )?;
        let mut output = vec![0_u8; output_len];
        for (y, filter) in coefficients.iter().enumerate() {
            checkpoint()?;
            for x in 0..target_width {
                for channel in 0..3 {
                    output[(y * target_width + x) * 3 + channel] =
                        pillow_resample_channel(filter.values.iter().enumerate().map(
                            |(offset, &coefficient)| {
                                (
                                    horizontal[((filter.start + offset) * target_width + x) * 3
                                        + channel],
                                    coefficient,
                                )
                            },
                        ));
                }
            }
        }
        output
    };

    RgbImage::from_raw(width, height, output)
        .ok_or_else(|| anyhow!("Pillow-compatible H3 resize produced an invalid RGB image"))
}

fn resize_endpoint(
    source: RgbImage,
    anchor: H3EndpointAnchor,
    packed_index: usize,
    geometry: &H3Fl2VaGeometry,
    progress: &ProgressReporter,
) -> Result<H3PreparedEndpoint> {
    let source_width = source.width();
    let source_height = source.height();
    let target_width = u32::try_from(geometry.width).context("H3 width does not fit u32")?;
    let target_height = u32::try_from(geometry.height).context("H3 height does not fit u32")?;
    let mut checkpoint = || progress.checkpoint().map_err(Into::into);
    let (image, resize) = if (source_width, source_height) == (target_width, target_height) {
        (source, H3EndpointResize::Identity)
    } else if packed_index == 0 {
        let plan = EndpointResizePlan::geometry_anchor(geometry.width, geometry.height)?;
        (
            pillow_lanczos_resize(&source, target_width, target_height, &mut checkpoint)?,
            H3EndpointResize::GeometryAnchor(plan),
        )
    } else {
        let plan = EndpointResizePlan::follower(
            usize::try_from(source_width).context("endpoint width does not fit usize")?,
            usize::try_from(source_height).context("endpoint height does not fit usize")?,
            geometry.width,
            geometry.height,
        )?;
        let EndpointResizePlan::CoverCrop {
            resized_width,
            resized_height,
            left,
            top,
            width,
            height,
        } = plan
        else {
            unreachable!("follower plan is always cover-crop")
        };
        let resized = pillow_lanczos_resize(
            &source,
            u32::try_from(resized_width).context("resized width does not fit u32")?,
            u32::try_from(resized_height).context("resized height does not fit u32")?,
            &mut checkpoint,
        )?;
        (
            imageops::crop_imm(
                &resized,
                u32::try_from(left).context("crop left does not fit u32")?,
                u32::try_from(top).context("crop top does not fit u32")?,
                u32::try_from(width).context("crop width does not fit u32")?,
                u32::try_from(height).context("crop height does not fit u32")?,
            )
            .to_image(),
            H3EndpointResize::Follower(plan),
        )
    };
    let pixels = Tensor::from_vec(
        image.into_raw(),
        (1, geometry.height, geometry.width, 3),
        &Device::Cpu,
    )?
    .permute((0, 3, 1, 2))?
    .unsqueeze(2)?;
    Ok(H3PreparedEndpoint {
        anchor,
        source_width,
        source_height,
        resize,
        pixels,
    })
}

/// Resolve an omitted canvas from the first packed endpoint. This helper sits
/// before Mold's normalized `GenerateRequest`, whose width/height are required,
/// so last-only FL2VA can still use the last image as its geometry anchor.
pub(crate) fn resolve_canvas(
    requested: Option<(u32, u32)>,
    first_packed_endpoint: Option<(u32, u32)>,
) -> Result<(u32, u32)> {
    match requested {
        Some((width, height)) if width > 0 && height > 0 => Ok((width, height)),
        Some(_) => bail!("MiniMax H3 explicit canvas dimensions must both be positive"),
        None => Ok(first_packed_endpoint.map_or(
            (contract::DEFAULT_WIDTH, contract::DEFAULT_HEIGHT),
            |(width, height)| contract::recommended_dimensions(width, height),
        )),
    }
}

fn build_packed_sequence(
    text_tags: &[H3ModalityTag],
    geometry: &H3Fl2VaGeometry,
    anchors: &[H3EndpointAnchor],
) -> Result<PackedSequence> {
    if text_tags.is_empty() {
        bail!("MiniMax H3 packed sequence requires at least one text row");
    }
    let text_rows = text_tags.len();
    let condition_start = text_rows;
    let audio_start = condition_start
        .checked_add(geometry.condition_video_rows)
        .ok_or_else(|| anyhow!("H3 packed condition offset overflow"))?;
    let video_start = audio_start
        .checked_add(geometry.generated_audio_rows)
        .ok_or_else(|| anyhow!("H3 packed audio offset overflow"))?;
    let sequence_len = video_start
        .checked_add(geometry.generated_video_rows)
        .ok_or_else(|| anyhow!("H3 packed video offset overflow"))?;
    if sequence_len > u32::MAX as usize {
        bail!("MiniMax H3 packed sequence exceeds u32 row addressing");
    }

    let mut positions = vec![[0.0f32; 3]; sequence_len];
    for (index, position) in positions[..text_rows].iter_mut().enumerate() {
        position[0] = index as f32;
    }
    let (frame_grid, width_grid) = frame_position_grid(
        geometry.latent_height,
        geometry.latent_width,
        VIDEO_PATCH[1],
        VIDEO_PATCH[2],
    )?;
    if anchors.len() * geometry.rows_per_video_frame != geometry.condition_video_rows {
        bail!("MiniMax H3 endpoint anchors do not match reserved condition rows");
    }
    for (index, anchor) in anchors.iter().copied().enumerate() {
        let anchor_time = match anchor {
            H3EndpointAnchor::First => text_rows as f64,
            H3EndpointAnchor::Last => {
                text_rows as f64 + temporal_span_sum(geometry.latent_frames) - 5.0 / 3.0
            }
        } as f32;
        let start = condition_start + index * geometry.rows_per_video_frame;
        for (row, spatial) in positions[start..start + geometry.rows_per_video_frame]
            .iter_mut()
            .zip(&frame_grid)
        {
            *row = [anchor_time, spatial[0], spatial[1]];
        }
    }

    for channel in 0..usize::try_from(AUDIO_CHANNELS).expect("audio channels fit usize") {
        let width = if channel == 0 {
            width_grid[0]
        } else {
            *width_grid.last().expect("target width grid is nonempty")
        };
        let start = audio_start + channel * geometry.audio_latents_per_channel;
        for (offset, row) in positions[start..start + geometry.audio_latents_per_channel]
            .iter_mut()
            .enumerate()
        {
            *row = [(text_rows + offset) as f32, 0.0, width];
        }
    }

    let temporal = temporal_position_grid(geometry.latent_frames, text_rows as f64);
    for (frame, &time) in temporal.iter().enumerate() {
        let start = video_start + frame * geometry.rows_per_video_frame;
        for (row, spatial) in positions[start..start + geometry.rows_per_video_frame]
            .iter_mut()
            .zip(&frame_grid)
        {
            *row = [time, spatial[0], spatial[1]];
        }
    }

    let text_indices = checked_indices(0, text_rows)?;
    let mut video_indices = checked_indices(condition_start, geometry.condition_video_rows)?;
    video_indices.extend(checked_indices(video_start, geometry.generated_video_rows)?);
    let audio_indices = checked_indices(audio_start, geometry.generated_audio_rows)?;
    let mut tags = vec![H3Modality::Video as u32; sequence_len];
    for (index, tag) in text_tags.iter().copied().enumerate() {
        tags[index] = match tag {
            H3ModalityTag::Vision => H3Modality::Video as u32,
            H3ModalityTag::Text => H3Modality::Text as u32,
        };
    }
    for &index in &audio_indices {
        tags[index as usize] = H3Modality::Audio as u32;
    }
    let mut timestep_indices = vec![0u32; sequence_len];
    for value in &mut timestep_indices[condition_start..audio_start] {
        *value = 1;
    }
    for value in &mut timestep_indices[audio_start..video_start] {
        *value = 2;
    }
    let layout = H3PackedLayout::new(
        positions,
        timestep_indices,
        tags,
        video_indices,
        audio_indices,
        text_indices,
    )?;
    Ok(PackedSequence {
        layout,
        condition_video_rows: geometry.condition_video_rows,
        generated_video_rows: geometry.generated_video_rows,
        generated_audio_rows: geometry.generated_audio_rows,
    })
}

fn frame_position_grid(
    latent_height: usize,
    latent_width: usize,
    patch_height: usize,
    patch_width: usize,
) -> Result<(Vec<[f32; 2]>, Vec<f32>)> {
    if latent_height == 0
        || latent_width == 0
        || patch_height == 0
        || patch_width == 0
        || !latent_height.is_multiple_of(patch_height)
        || !latent_width.is_multiple_of(patch_width)
    {
        bail!("invalid MiniMax H3 rotary spatial grid");
    }
    let sqrt_area = ((latent_height as f64) * (latent_width as f64)).sqrt();
    let axis = |dimension: usize, patch: usize| {
        let count = dimension / patch;
        let ratio = dimension as f64 / sqrt_area;
        let left = (1.0 - ratio) / 2.0;
        (0..count)
            .map(|index| ((left + index as f64 * ratio / count as f64) * 32.0) as f32)
            .collect::<Vec<_>>()
    };
    let heights = axis(latent_height, patch_height);
    let widths = axis(latent_width, patch_width);
    let grid = heights
        .iter()
        .flat_map(|&height| widths.iter().map(move |&width| [height, width]))
        .collect();
    Ok((grid, widths))
}

fn temporal_position_grid(frames: usize, origin: f64) -> Vec<f32> {
    let mut result = Vec::with_capacity(frames);
    let mut time = origin;
    for index in 0..frames {
        result.push(time as f32);
        time += temporal_span(index);
    }
    result
}

fn temporal_span(index: usize) -> f64 {
    const SPANS: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
    (5.0 / 3.0) * SPANS[index % SPANS.len()]
}

fn temporal_span_sum(frames: usize) -> f64 {
    // Balanced reduction matches the intent of NumPy's pairwise summation.
    fn sum(values: &[f64]) -> f64 {
        match values {
            [] => 0.0,
            [value] => *value,
            values => {
                let middle = values.len() / 2;
                sum(&values[..middle]) + sum(&values[middle..])
            }
        }
    }
    sum(&(0..frames).map(temporal_span).collect::<Vec<_>>())
}

fn checked_indices(start: usize, len: usize) -> Result<Vec<u32>> {
    (start
        ..start
            .checked_add(len)
            .ok_or_else(|| anyhow!("H3 row range overflow"))?)
        .map(|index| u32::try_from(index).context("H3 row index exceeds u32"))
        .collect()
}

fn validate_condition_latent(condition: &Tensor, geometry: &H3Fl2VaGeometry) -> Result<()> {
    if condition.dtype() != DType::F32
        || condition.dims()
            != [
                1,
                VIDEO_LATENT_CHANNELS,
                1,
                geometry.latent_height,
                geometry.latent_width,
            ]
    {
        bail!(
            "MiniMax H3 endpoint VAE latent has {:?} {:?}, expected F32 [1,{VIDEO_LATENT_CHANNELS},1,{},{}]",
            condition.dtype(),
            condition.dims(),
            geometry.latent_height,
            geometry.latent_width
        );
    }
    Ok(())
}

fn validate_packed_rows(
    video: &Tensor,
    audio: &Tensor,
    text: &Tensor,
    packed: &PackedSequence,
) -> Result<()> {
    let (_, video_rows, video_width) = video.dims3()?;
    let (_, audio_rows, audio_width) = audio.dims3()?;
    let (_, text_rows, text_width) = text.dims3()?;
    if video_rows != packed.condition_video_rows + packed.generated_video_rows
        || video_width != VIDEO_LATENT_CHANNELS * 4
        || audio_rows != packed.generated_audio_rows
        || audio_width != AUDIO_LATENT_CHANNELS
        || text_rows == 0
        || text_width != TEXT_STATE_WIDTH
    {
        bail!("MiniMax H3 packed tensors do not match the frozen layout");
    }
    Ok(())
}

fn validate_transformer_output(
    output: &H3TransformerOutput,
    video: &Tensor,
    audio: &Tensor,
) -> Result<()> {
    if output.video.dims() != video.dims()
        || output.audio.dims() != audio.dims()
        || !output.video.device().same_device(video.device())
        || !output.audio.device().same_device(audio.device())
    {
        bail!("MiniMax H3 transformer output does not match both packed modalities");
    }
    Ok(())
}

fn validate_waveform(waveform: &StereoWaveform, latent_rows: usize) -> Result<()> {
    let (batch, channels, samples) = waveform.samples().dims3()?;
    let expected = latent_rows
        .checked_mul(AUDIO_SAMPLES_PER_LATENT)
        .ok_or_else(|| anyhow!("H3 decoded audio sample count overflow"))?;
    if waveform.samples().dtype() != DType::F32
        || batch != 1
        || channels != AUDIO_CHANNELS as usize
        || samples != expected
        || waveform.association() != AudioSoundtrackAssociation::Generated
    {
        bail!(
            "MiniMax H3 audio decoder must emit the generated F32 timeline [1,2,{expected}], got {:?} {:?} {:?}",
            waveform.samples().dtype(),
            waveform.samples().dims(),
            waveform.association()
        );
    }
    Ok(())
}

#[cfg(feature = "mp4")]
fn interleaved_pcm(waveform: &StereoWaveform) -> Result<Vec<f32>> {
    let channels = waveform
        .samples()
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let [left, right]: [Vec<f32>; 2] = channels
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("H3 waveform lost its batch"))?
        .try_into()
        .map_err(|_| anyhow!("H3 waveform is not stereo"))?;
    Ok(left
        .into_iter()
        .zip(right)
        .flat_map(|(left, right)| [left, right])
        .collect())
}

fn phase_boundary(
    control: &mut dyn H3PipelineCheckpoint,
    phase: H3PipelinePhase,
    complete: bool,
) -> Result<()> {
    control.checkpoint(H3PipelineEvent {
        phase,
        completed: usize::from(complete),
        total: 1,
    })
}

fn ensure_identity(
    backend: &dyn H3Fl2VaBackend,
    frozen: &H3PipelineBackendIdentity,
    frozen_device: &Device,
) -> Result<()> {
    let current = backend.identity();
    if &current != frozen || !backend.device().same_device(frozen_device) {
        bail!(
            "MiniMax H3 backend identity changed from {:?} to {:?}; implicit reroute is forbidden",
            frozen,
            current
        );
    }
    current.validate(backend.device())
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or_else(|| anyhow!("{label} element count overflow"))
    })
}

fn rounded_ratio(value: usize, numerator: usize, denominator: usize) -> Result<usize> {
    if denominator == 0 {
        bail!("MiniMax H3 ratio denominator cannot be zero");
    }
    let scaled = value
        .checked_mul(numerator)
        .ok_or_else(|| anyhow!("H3 rational timeline overflow"))?;
    let quotient = scaled / denominator;
    let remainder = scaled % denominator;
    let twice = remainder
        .checked_mul(2)
        .ok_or_else(|| anyhow!("H3 rational rounding overflow"))?;
    if twice > denominator || (twice == denominator && !quotient.is_multiple_of(2)) {
        quotient
            .checked_add(1)
            .ok_or_else(|| anyhow!("H3 rounded timeline overflow"))
    } else {
        Ok(quotient)
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::TextToAudioVideo => "t2va",
        Mode::FirstFrameToAudioVideo => "first-frame-fl2va",
        Mode::LastFrameToAudioVideo => "last-frame-fl2va",
        Mode::FirstAndLastFrameToAudioVideo => "first-last-frame-fl2va",
        Mode::ReferenceToAudioVideo => "ref2va",
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, Weak};

    use image::{DynamicImage, ImageFormat, Rgb};
    use mold_candle::minimax_h3::{
        AudioSoundtrackAssociation, H3BlockProgress, H3BlockStack, H3TransformerOutput,
    };
    use mold_core::{KeyframeCondition, OutputFormat};

    use super::*;
    use crate::progress::{
        is_inference_cancelled, InferenceCancellationToken, InferenceCancelled, ProgressReporter,
    };

    fn request() -> GenerateRequest {
        GenerateRequest {
            mesh: None,
            video_only: None,
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            prompt: "a lighthouse in a storm".into(),
            negative_prompt: None,
            model: contract::FL2VA_COMFY.into(),
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
            references: None,
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
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    #[test]
    fn placeholder_endpoint_matches_requested_geometry_and_stays_bounded() {
        let bytes = placeholder_endpoint_png(1344, 768);
        let decoded = decode_endpoint(&bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (1344, 768));

        // Degenerate and oversized requests clamp inside the bounded image
        // contract instead of producing an undecodable or refused endpoint.
        let clamped = placeholder_endpoint_png(0, 10_000_000);
        let decoded = decode_endpoint(&clamped).unwrap();
        assert_eq!(decoded.width(), 1);
        assert_eq!(decoded.height(), MAX_REFERENCE_DIMENSION);
    }

    fn png(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
        let image = RgbImage::from_pixel(width, height, Rgb(color));
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
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
                if let Some(cancellation) = &self.cancellation {
                    cancellation.cancel();
                }
            }
        }
    }

    struct SyntheticBackend {
        device: Device,
        identity: H3PipelineBackendIdentity,
        sampler_kind: H3SamplerKind,
        forwards: usize,
        decoded_video: bool,
        decoded_audio: bool,
        reroute_after_audio_decode: bool,
        condition_values: Arc<Mutex<Vec<f32>>>,
        condition_rows_to_watch: usize,
        observed_condition_rows: Vec<Vec<f32>>,
        text_lifetime: Option<Weak<()>>,
        /// Stand in for a conditioner output served from the in-process
        /// cache: one `QwenConditioningCached` event and no encode at all.
        conditioning_from_cache: bool,
    }

    impl SyntheticBackend {
        fn new() -> Self {
            Self {
                device: Device::Cpu,
                identity: H3PipelineBackendIdentity {
                    kind: H3PipelineBackendKind::SyntheticCpu,
                    device_id: "synthetic-cpu-0".into(),
                    execution_fingerprint: "synthetic-h3-v1".into(),
                },
                sampler_kind: H3SamplerKind::OfficialEuler,
                forwards: 0,
                decoded_video: false,
                decoded_audio: false,
                reroute_after_audio_decode: false,
                condition_values: Arc::new(Mutex::new(Vec::new())),
                condition_rows_to_watch: 0,
                observed_condition_rows: Vec::new(),
                text_lifetime: None,
                conditioning_from_cache: false,
            }
        }

        fn text_was_dropped(&self) -> bool {
            self.text_lifetime
                .as_ref()
                .is_some_and(|lifetime| lifetime.upgrade().is_none())
        }
    }

    impl H3Fl2VaBackend for SyntheticBackend {
        fn identity(&self) -> H3PipelineBackendIdentity {
            self.identity.clone()
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn sampler_kind(&self) -> H3SamplerKind {
            self.sampler_kind
        }

        fn encode_text(
            &mut self,
            _prompt: &str,
            endpoints: &[H3PreparedEndpoint],
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TextConditioning> {
            checkpoint.checkpoint(H3PipelineEvent {
                phase: if self.conditioning_from_cache {
                    H3PipelinePhase::QwenConditioningCached
                } else {
                    H3PipelinePhase::PromptEncode
                },
                completed: if self.conditioning_from_cache { 1 } else { 0 },
                total: 1,
            })?;
            let mut tags = vec![H3ModalityTag::Text];
            for _ in endpoints {
                tags.extend([H3ModalityTag::Text, H3ModalityTag::Vision]);
            }
            let lifetime = Arc::new(());
            self.text_lifetime = Some(Arc::downgrade(&lifetime));
            Ok(H3TextConditioning {
                states: Tensor::zeros((1, tags.len(), TEXT_STATE_WIDTH), DType::F32, &self.device)?,
                tags,
                lifetime_probe: Some(lifetime),
            })
        }

        fn encode_visual_condition(
            &mut self,
            endpoint: &H3PreparedEndpoint,
            mode: ConditionEncodeMode,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<Tensor> {
            assert_eq!(mode, ConditionEncodeMode::OfficialFreshSeed42);
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::VisualConditionEncodeChunk,
                completed: 0,
                total: 1,
            })?;
            let (_, _, _, height, width) = endpoint.pixels.dims5()?;
            let value = match endpoint.anchor {
                H3EndpointAnchor::First => 0.25,
                H3EndpointAnchor::Last => -0.5,
            };
            self.condition_values.lock().unwrap().push(value);
            Tensor::full(
                value,
                (
                    1,
                    VIDEO_LATENT_CHANNELS,
                    1,
                    height / VIDEO_VAE_SPATIAL_COMPRESSION,
                    width / VIDEO_VAE_SPATIAL_COMPRESSION,
                ),
                &self.device,
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
            if self.condition_rows_to_watch > 0 {
                self.observed_condition_rows.push(
                    input
                        .video_rows
                        .narrow(1, 0, self.condition_rows_to_watch)?
                        .flatten_all()?
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?,
                );
            }
            self.forwards += 1;
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
            assert!(self.text_was_dropped());
            assert_eq!(latents.dims(), [1, 24, 37, 2, 2]);
            for frame in 0..124 {
                checkpoint.checkpoint(H3PipelineEvent {
                    phase: H3PipelinePhase::VisualDecodeChunk,
                    completed: frame,
                    total: 124,
                })?;
                sink.push(
                    &RgbImage::from_pixel(32, 32, Rgb([frame as u8, 2, 3])),
                    checkpoint,
                )?;
            }
            self.decoded_video = true;
            Ok(())
        }

        fn decode_audio(
            &mut self,
            latents: &StereoLatents,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<StereoWaveform> {
            assert!(self.text_was_dropped());
            assert_eq!(latents.normalized().dims(), [1, 32, 2, 207]);
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::AudioDecodeChunk,
                completed: 0,
                total: 1,
            })?;
            self.decoded_audio = true;
            let waveform = StereoWaveform::new(
                Tensor::zeros(
                    (1, 2, 207 * AUDIO_SAMPLES_PER_LATENT),
                    DType::F32,
                    &self.device,
                )?,
                AUDIO_SAMPLE_RATE_HZ as usize,
                AudioSoundtrackAssociation::Generated,
            )?;
            if self.reroute_after_audio_decode {
                self.identity.device_id = "other-gpu".into();
            }
            Ok(waveform)
        }
    }

    fn prepared(req: &GenerateRequest) -> H3PreparedFl2VaRequest {
        prepare_request(
            req,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap()
    }

    #[test]
    fn exact_geometry_covers_short_and_nominal_fifteen_second_boundaries() {
        // Geometry is a family property and the compact tags now span the
        // whole family grid, so every boundary is exercised on the shipping
        // compact layout rather than through the official reference.
        for (frames, video_latents, audio_latents) in
            [(107, 32, 178), (124, 37, 207), (345, 102, 575)]
        {
            let mut req = request();
            req.frames = Some(frames);
            let prepared = prepared(&req);
            assert_eq!(prepared.geometry.latent_frames, video_latents);
            assert_eq!(prepared.geometry.audio_latents_per_channel, audio_latents);
            assert_eq!(
                prepared.geometry.generated_video_rows,
                video_latents * prepared.geometry.rows_per_video_frame
            );
            assert_eq!(
                prepared.geometry.generated_audio_rows,
                audio_latents * AUDIO_CHANNELS as usize
            );
        }
    }

    #[test]
    fn low_level_pipeline_rejects_unsplit_batches() {
        let mut req = request();
        req.batch_size = 2;
        let error = prepare_request(
            &req,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("must be split by the authoritative batch router"));
    }

    #[test]
    fn omitted_canvas_uses_the_first_packed_endpoint_including_last_only() {
        assert_eq!(resolve_canvas(None, None).unwrap(), (1344, 768));
        assert_eq!(
            resolve_canvas(None, Some((600, 900))).unwrap(),
            contract::recommended_dimensions(600, 900)
        );
        assert_eq!(
            resolve_canvas(Some((640, 384)), Some((600, 900))).unwrap(),
            (640, 384)
        );
    }

    #[test]
    fn endpoints_are_canonical_and_use_anchor_then_follower_resize() {
        let mut req = request();
        req.source_image = Some(png(48, 80, [10, 20, 30]));
        req.keyframes = Some(vec![KeyframeCondition {
            frame: 123,
            image: png(80, 48, [40, 50, 60]),
            name: None,
        }]);
        let prepared = prepared(&req);
        assert_eq!(prepared.geometry.mode, Mode::FirstAndLastFrameToAudioVideo);
        assert_eq!(prepared.endpoints.len(), 2);
        assert_eq!(prepared.endpoints[0].anchor, H3EndpointAnchor::First);
        assert_eq!(prepared.endpoints[1].anchor, H3EndpointAnchor::Last);
        assert!(matches!(
            prepared.endpoints[0].resize,
            H3EndpointResize::GeometryAnchor(EndpointResizePlan::Stretch { .. })
        ));
        assert!(matches!(
            prepared.endpoints[1].resize,
            H3EndpointResize::Follower(EndpointResizePlan::CoverCrop { .. })
        ));
        assert_eq!(prepared.endpoints[0].pixels.dims(), [1, 3, 1, 32, 32]);
        assert_eq!(prepared.endpoints[1].pixels.dims(), [1, 3, 1, 32, 32]);
    }

    #[test]
    fn endpoint_resize_matches_pillow_lanczos_pixels() {
        let source = RgbImage::from_raw(
            5,
            7,
            (0..7)
                .flat_map(|y| {
                    (0..5).flat_map(move |x| {
                        (0..3).map(move |channel| (x * 37 + y * 19 + channel * 53) as u8)
                    })
                })
                .collect(),
        )
        .unwrap();
        let resized = pillow_lanczos_resize(&source, 8, 4, &mut || Ok(())).unwrap();
        assert_eq!(
            resized.into_raw(),
            vec![
                6, 57, 110, 18, 71, 125, 47, 100, 153, 71, 125, 168, 91, 142, 210, 115, 165, 243,
                144, 207, 188, 158, 228, 138, 36, 90, 144, 51, 104, 152, 80, 132, 183, 103, 153,
                225, 124, 186, 244, 148, 214, 192, 174, 191, 77, 186, 172, 6, 71, 124, 169, 85,
                138, 216, 114, 168, 231, 139, 190, 125, 157, 220, 40, 182, 206, 15, 222, 74, 52,
                244, 0, 84, 104, 157, 221, 118, 167, 183, 147, 203, 103, 166, 252, 25, 203, 172,
                20, 221, 16, 67, 156, 16, 99, 111, 67, 107,
            ]
        );
    }

    #[test]
    fn endpoint_resize_polls_cancellation_between_work_rows() {
        let source = RgbImage::from_pixel(64, 64, Rgb([10, 20, 30]));
        let mut polls = 0;
        let error = pillow_lanczos_resize(&source, 96, 32, &mut || {
            polls += 1;
            if polls == 4 {
                Err(anyhow::Error::new(InferenceCancelled))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(polls, 4);
        assert!(is_inference_cancelled(&error));
    }

    #[test]
    fn last_only_is_one_geometry_anchor_not_a_follower() {
        let mut req = request();
        req.keyframes = Some(vec![KeyframeCondition {
            frame: 123,
            image: png(80, 48, [40, 50, 60]),
            name: None,
        }]);
        let prepared = prepared(&req);
        assert_eq!(prepared.geometry.mode, Mode::LastFrameToAudioVideo);
        assert_eq!(prepared.endpoints[0].anchor, H3EndpointAnchor::Last);
        assert!(matches!(
            prepared.endpoints[0].resize,
            H3EndpointResize::GeometryAnchor(_)
        ));
    }

    #[test]
    fn packed_layout_counts_t2va_and_each_endpoint_signature() {
        for (mode, anchors, condition_rows) in [
            (Mode::TextToAudioVideo, vec![], 0),
            (
                Mode::FirstFrameToAudioVideo,
                vec![H3EndpointAnchor::First],
                1,
            ),
            (Mode::LastFrameToAudioVideo, vec![H3EndpointAnchor::Last], 1),
            (
                Mode::FirstAndLastFrameToAudioVideo,
                vec![H3EndpointAnchor::First, H3EndpointAnchor::Last],
                2,
            ),
        ] {
            let mut geometry =
                H3Fl2VaGeometry::from_request(&request(), mode, anchors.len()).unwrap();
            geometry.mode = mode;
            let layout =
                build_packed_sequence(&[H3ModalityTag::Text], &geometry, &anchors).unwrap();
            assert_eq!(
                layout.condition_video_rows,
                condition_rows * geometry.rows_per_video_frame
            );
            assert_eq!(
                layout.layout.seq_len(),
                1 + layout.condition_video_rows
                    + geometry.generated_audio_rows
                    + geometry.generated_video_rows
            );
        }
    }

    #[test]
    fn request_noise_draw_order_is_condition_then_video_then_audio() {
        let mut req = request();
        req.source_image = Some(png(32, 32, [10, 20, 30]));
        req.keyframes = Some(vec![KeyframeCondition {
            frame: 123,
            image: png(32, 32, [40, 50, 60]),
            name: None,
        }]);
        let prepared = prepared(&req);
        let mut backend = SyntheticBackend::new();
        backend.condition_rows_to_watch = 2;
        let staged = execute_staged(
            &prepared,
            &mut backend,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap();
        assert_eq!(backend.forwards, 3);
        assert!(backend.decoded_video && backend.decoded_audio);
        assert!(backend.text_was_dropped());
        assert_eq!(*backend.condition_values.lock().unwrap(), [0.25, -0.5]);
        assert_eq!(backend.observed_condition_rows.len(), 3);
        assert!(backend
            .observed_condition_rows
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
        assert!(staged.provenance.noise_draws[..2]
            .iter()
            .all(|draw| draw.seed == 42 && draw.fresh_generator));
        assert!(staged.provenance.noise_draws[2..]
            .iter()
            .all(|draw| draw.seed == 7 && !draw.fresh_generator));
        assert_eq!(staged.provenance.noise_domain_version, NOISE_DOMAIN_VERSION);
        assert_eq!(staged.provenance.endpoint_anchors.len(), 2);
        assert_eq!(staged.provenance.requested_grid_points, 4);
        assert_eq!(staged.provenance.transformer_evaluations, 3);
        assert_eq!(staged.provenance.sampler, "official-euler");
        assert!(!staged.video_only_mp4.is_empty());
        assert!(!staged.thumbnail_png.is_empty());
    }

    #[test]
    fn cached_conditioning_stages_without_a_conditioner_encode() {
        let prepared = prepared(&request());
        let mut backend = SyntheticBackend::new();
        backend.conditioning_from_cache = true;
        let mut observer = RecordingObserver::default();
        let staged = execute_staged(
            &prepared,
            &mut backend,
            &ProgressReporter::default(),
            &mut observer,
        )
        .unwrap();
        assert!(!staged.video_only_mp4.is_empty());
        assert!(backend.text_was_dropped());

        let cached = observer
            .events
            .iter()
            .filter(|event| event.phase == H3PipelinePhase::QwenConditioningCached)
            .collect::<Vec<_>>();
        assert_eq!(cached.len(), 1, "the hit is disclosed exactly once");
        assert_eq!((cached[0].completed, cached[0].total), (1, 1));
        assert!(
            !observer.events.iter().any(|event| matches!(
                event.phase,
                H3PipelinePhase::QwenLoad | H3PipelinePhase::QwenEncode
            )),
            "a served conditioner loads nothing and encodes nothing"
        );
        assert!(
            observer
                .events
                .iter()
                .any(|event| event.phase == H3PipelinePhase::PromptEncode && event.completed == 1),
            "the outer FL2VA prompt-encode boundary still closes"
        );
    }

    /// The restored tensor must satisfy the same gate the pipeline applies to
    /// a freshly encoded one — same shape, same BF16 dtype, same device.
    #[test]
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    fn cached_conditioning_restores_a_tensor_the_pipeline_accepts() {
        use crate::h3_factory::H3FactoryConditionerPlacement;
        use crate::minimax_h3::conditioner_cache::{
            H3CachedConditioning, H3ConditionerRouteIdentity,
        };

        let device = Device::Cpu;
        let rows = 3;
        let encoded = H3TextConditioning {
            states: Tensor::zeros((1, rows, TEXT_STATE_WIDTH), DType::BF16, &device).unwrap(),
            tags: vec![H3ModalityTag::Text; rows],
            lifetime_probe: None,
        };
        encoded.validate(&device).unwrap();
        let entry = H3CachedConditioning::capture(
            &encoded,
            rows as u64,
            0,
            H3ConditionerRouteIdentity {
                placement: H3FactoryConditionerPlacement::AssignedCudaThenDrop,
                device_id: "cuda:0".into(),
            },
        )
        .unwrap();
        let restored = entry.restore(&device).unwrap();
        restored.validate(&device).unwrap();
        assert_eq!(restored.states.dtype(), DType::BF16);
        assert_eq!(restored.tags, encoded.tags);
    }

    #[test]
    fn backend_selected_sampler_executes_and_is_retained_in_provenance() {
        let prepared = prepared(&request());
        let mut backend = SyntheticBackend::new();
        backend.sampler_kind = H3SamplerKind::ComfyResMultistep;
        let staged = execute_staged(
            &prepared,
            &mut backend,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap();

        assert_eq!(backend.forwards, 3);
        assert_eq!(staged.provenance.sampler, "comfy-res-multistep");
    }

    #[test]
    fn same_seed_reproduces_all_draws_and_added_endpoint_shifts_targets() {
        let t2va = prepared(&request());
        let mut with_endpoint_request = request();
        with_endpoint_request.source_image = Some(png(32, 32, [1, 2, 3]));
        let with_endpoint = prepared(&with_endpoint_request);

        let draw_all = |prepared: &H3PreparedFl2VaRequest| {
            let mut rng = H3RequestNoise::new(prepared.seed);
            for (index, _) in prepared.endpoints.iter().enumerate() {
                rng.draw("condition-noise", index, &[1, 24, 1, 2, 2], &Device::Cpu)
                    .unwrap();
            }
            let video = rng
                .draw(
                    "target-video",
                    0,
                    &prepared.geometry.generated_video_shape(),
                    &Device::Cpu,
                )
                .unwrap();
            let audio = rng
                .draw(
                    "target-audio",
                    0,
                    &prepared.geometry.generated_audio_row_shape(),
                    &Device::Cpu,
                )
                .unwrap();
            (
                video.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                audio.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            )
        };
        let first = draw_all(&t2va);
        let second = draw_all(&t2va);
        let shifted = draw_all(&with_endpoint);
        assert_eq!(first, second);
        assert_ne!(first.0, shifted.0);
        assert_ne!(first.1, shifted.1);
    }

    #[test]
    fn cancellation_between_coupled_forwards_never_decodes_or_publishes() {
        let prepared = prepared(&request());
        let cancellation = InferenceCancellationToken::default();
        let mut progress = ProgressReporter::default();
        progress.set_cancellation_token(cancellation.clone());
        let cancel_at = H3PipelineEvent {
            phase: H3PipelinePhase::Denoise,
            completed: 1,
            total: 3,
        };
        let mut observer = RecordingObserver {
            cancel_at: Some(cancel_at),
            cancellation: Some(cancellation),
            ..Default::default()
        };
        let mut backend = SyntheticBackend::new();
        let error = execute_staged(&prepared, &mut backend, &progress, &mut observer).unwrap_err();
        assert!(is_inference_cancelled(&error));
        assert_eq!(backend.forwards, 1);
        assert!(!backend.decoded_video);
        assert!(!backend.decoded_audio);
        assert!(backend.text_was_dropped());
    }

    #[test]
    fn backend_identity_cannot_change_mid_pipeline() {
        struct ReroutingBackend {
            inner: SyntheticBackend,
            identity_reads: usize,
        }
        impl H3Fl2VaBackend for ReroutingBackend {
            fn identity(&self) -> H3PipelineBackendIdentity {
                let mut identity = self.inner.identity.clone();
                if self.identity_reads > 0 {
                    identity.device_id = "other-gpu".into();
                }
                identity
            }
            fn device(&self) -> &Device {
                self.inner.device()
            }
            fn encode_text(
                &mut self,
                prompt: &str,
                endpoints: &[H3PreparedEndpoint],
                checkpoint: &mut dyn H3PipelineCheckpoint,
            ) -> Result<H3TextConditioning> {
                self.identity_reads += 1;
                self.inner.encode_text(prompt, endpoints, checkpoint)
            }
            fn encode_visual_condition(
                &mut self,
                endpoint: &H3PreparedEndpoint,
                mode: ConditionEncodeMode,
                checkpoint: &mut dyn H3PipelineCheckpoint,
            ) -> Result<Tensor> {
                self.inner
                    .encode_visual_condition(endpoint, mode, checkpoint)
            }
            fn denoise(
                &mut self,
                input: H3ForwardInput<'_>,
                layout: &H3FrozenPackedLayout,
                checkpoint: &mut dyn H3PipelineCheckpoint,
            ) -> Result<H3TransformerOutput> {
                self.inner.denoise(input, layout, checkpoint)
            }
            fn decode_video(
                &mut self,
                latents: &Tensor,
                sink: &mut H3VideoEncodeSink,
                checkpoint: &mut dyn H3PipelineCheckpoint,
            ) -> Result<()> {
                self.inner.decode_video(latents, sink, checkpoint)
            }
            fn decode_audio(
                &mut self,
                latents: &StereoLatents,
                checkpoint: &mut dyn H3PipelineCheckpoint,
            ) -> Result<StereoWaveform> {
                self.inner.decode_audio(latents, checkpoint)
            }
        }
        let mut backend = ReroutingBackend {
            inner: SyntheticBackend::new(),
            identity_reads: 0,
        };
        let error = execute_staged(
            &prepared(&request()),
            &mut backend,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap_err();
        assert!(error.to_string().contains("implicit reroute is forbidden"));
    }

    #[test]
    fn backend_identity_cannot_change_during_final_audio_decode() {
        let mut backend = SyntheticBackend::new();
        backend.reroute_after_audio_decode = true;
        let error = execute_staged(
            &prepared(&request()),
            &mut backend,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap_err();
        assert!(error.to_string().contains("implicit reroute is forbidden"));
    }

    #[test]
    fn decoded_audio_must_match_the_exact_generated_timeline() {
        let latent_rows = 207;
        let overlong = StereoWaveform::new(
            Tensor::zeros(
                (1, 2, latent_rows * AUDIO_SAMPLES_PER_LATENT + 1),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap(),
            AUDIO_SAMPLE_RATE_HZ as usize,
            AudioSoundtrackAssociation::Generated,
        )
        .unwrap();
        assert!(validate_waveform(&overlong, latent_rows).is_err());

        let unrelated = StereoWaveform::new(
            Tensor::zeros(
                (1, 2, latent_rows * AUDIO_SAMPLES_PER_LATENT),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap(),
            AUDIO_SAMPLE_RATE_HZ as usize,
            AudioSoundtrackAssociation::StandaloneReference { reference_index: 0 },
        )
        .unwrap();
        assert!(validate_waveform(&unrelated, latent_rows).is_err());
    }

    #[test]
    fn block_progress_can_be_reported_without_changing_phase_totals() {
        let _ = H3BlockProgress {
            stack: H3BlockStack::Transformer,
            completed: 1,
            total: 50,
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let mut progress = ProgressReporter::default();
        progress.set_callback(Box::new(move |event| captured.lock().unwrap().push(event)));
        let mut observer = RecordingObserver::default();
        let mut backend = SyntheticBackend::new();
        execute_staged(
            &prepared(&request()),
            &mut backend,
            &progress,
            &mut observer,
        )
        .unwrap();
        let denoise = observer
            .events
            .iter()
            .filter(|event| event.phase == H3PipelinePhase::Denoise)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(denoise.first().unwrap().completed, 0);
        assert_eq!(denoise.last().unwrap().completed, 3);
        assert!(denoise.iter().all(|event| event.total == 3));
        let mut totals = std::collections::HashMap::new();
        for event in observer.events {
            assert_eq!(
                *totals.entry(event.phase).or_insert(event.total),
                event.total,
                "phase {:?} changed progress denominator",
                event.phase
            );
        }
        assert_eq!(contract::CONDITION_POSTERIOR_SEED, 42);
        assert_eq!(
            (
                contract::FRAME_STEP,
                contract::FRAME_OFFSET,
                contract::DIMENSION_ALIGNMENT
            ),
            (17, 5, 32)
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn generated_channel_major_waveform_interleaves_left_then_right() {
        let waveform = StereoWaveform::new(
            Tensor::from_vec(
                vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0],
                (1, 2, 3),
                &Device::Cpu,
            )
            .unwrap(),
            AUDIO_SAMPLE_RATE_HZ as usize,
            AudioSoundtrackAssociation::Generated,
        )
        .unwrap();
        assert_eq!(
            interleaved_pcm(&waveform).unwrap(),
            [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn final_mux_matches_exact_video_timeline_and_preserves_stereo() {
        let mut backend = SyntheticBackend::new();
        let staged = execute_staged(
            &prepared(&request()),
            &mut backend,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap();
        let output = finalize_av(
            staged,
            &ProgressReporter::default(),
            &mut NoopH3PipelineObserver,
        )
        .unwrap();
        assert!(!output.mp4.is_empty());
        assert_eq!(output.mux_report.sample_rate, AUDIO_SAMPLE_RATE_HZ);
        assert_eq!(output.mux_report.channels, AUDIO_CHANNELS as u16);
        assert_eq!(output.mux_report.output_samples_per_channel, 165_333);
        assert_eq!(output.mux_report.trimmed_samples_per_channel, 267);
    }
}
