//! Legal-neutral MiniMax H3 FL2VA/Ref2VA engine and streamed-dispatch seam.
//!
//! This module deliberately has no production loader registration. The public
//! frozen factory still rejects H3 before paths or loader callbacks while the
//! #831 authorization gate, runnable capability, attention qualification, and
//! artifact identities are unavailable. Tests inject only in-memory runtimes.
//!
//! The streamed denoiser below consumes real [`H3BlockLoader`] objects one at a
//! time. It does not adapt the current eager `H3Transformer`: doing so would
//! falsely claim that its resident `Vec` satisfies block streaming. A future
//! authorized loader must provide an executor over independently owned block
//! objects before this seam can be registered by the production factory.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use mold_candle::minimax_h3::{
    H3ForwardInput, H3FrozenPackedLayout, H3TransformerOutput, StereoLatents, StereoWaveform,
};
use mold_core::minimax_h3::{self as contract, Mode, Task};
use mold_core::{GenerateRequest, GenerateResponse, ModelPaths, OutputFormat, VideoData};

use super::offload::{
    FrozenH3BlockStreamingPlan, H3BlockLease, H3BlockLoader, H3BlockStreamEvent,
    H3BlockStreamState, H3StreamCancellation, H3StreamCheckpoint,
};
#[cfg(feature = "mp4")]
use super::pipeline;
#[cfg(feature = "mp4")]
use super::pipeline::ref2va as ref2va_pipeline;
use super::pipeline::ref2va::{
    H3AudioConditionEncodeMode, H3DecodedReferenceFacts, H3PreparedReference, H3Ref2VaBackend,
    H3ReferencePresentation,
};
use super::pipeline::{
    H3Fl2VaBackend, H3PipelineBackendIdentity, H3PipelineCheckpoint, H3PipelineEvent,
    H3PipelineObserver, H3PipelinePhase, H3PreparedEndpoint, H3TextConditioning, H3VideoEncodeSink,
};
use crate::engine::{
    BatchExecutionCapability, GenerationReferenceBinding, InferenceEngine, LoadStrategy,
};
use crate::progress::{
    InferenceCancellationToken, ProgressCallback, ProgressEvent, ProgressPhase, ProgressReporter,
};
use crate::FrozenH3FactoryAuthority;

/// Stable failure used by the shipping factory until a separately reviewed
/// artifact locator/loader is registered. Keeping it here makes it impossible
/// for a path, header, network client, or queue side effect to be mistaken for
/// the legal-neutral runtime seam.
pub(crate) const H3_REAL_LOADER_DISABLED: &str =
    "MiniMax H3 real filesystem loader is runtime-disabled pending authorization and exact artifact identities";

#[derive(Clone, Copy)]
pub(crate) struct H3RuntimeLoadRequest<'a> {
    pub(crate) model_name: &'a str,
    pub(crate) paths: &'a ModelPaths,
    pub(crate) authority: &'a FrozenH3FactoryAuthority,
    pub(crate) load_strategy: LoadStrategy,
    pub(crate) gpu_ordinal: usize,
    pub(crate) block_offload: bool,
}

/// Injected component loader. No implementation is installed by the shipping
/// factory; callers inside tests use in-memory components only.
pub(crate) trait H3RuntimeLoader: Send + Sync {
    fn load(
        &mut self,
        request: H3RuntimeLoadRequest<'_>,
        progress: &ProgressReporter,
    ) -> Result<Box<dyn H3RuntimeSession>>;
}

/// One already-loaded, single-route H3 runtime.
pub(crate) trait H3RuntimeSession: Send + Sync {
    fn device_id(&self) -> &str;
    fn execution_fingerprint(&self) -> &str;
    fn component_set_identity_sha256(&self) -> &str;
    fn generate(
        &mut self,
        request: &GenerateRequest,
        bindings: &[GenerationReferenceBinding],
        progress: &ProgressReporter,
        observer: &mut dyn H3PipelineObserver,
    ) -> Result<H3EngineOutput>;
}

#[derive(Debug)]
pub(crate) struct H3EngineOutput {
    pub(crate) mp4: Vec<u8>,
    pub(crate) thumbnail_png: Vec<u8>,
    pub(crate) mode: Mode,
    pub(crate) seed: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    pub(crate) fps: u32,
    pub(crate) duration_ms: u64,
    pub(crate) audio_sample_rate: u32,
    pub(crate) audio_channels: u32,
    pub(crate) device_id: String,
    pub(crate) execution_fingerprint: String,
    pub(crate) component_set_identity_sha256: String,
    pub(crate) reference_fingerprint: Option<String>,
}

/// Concrete task-bound `InferenceEngine` adapter. It owns the exact
/// server-frozen route, but can be constructed only with an injected loader;
/// production factory dispatch deliberately has no such loader today.
pub(crate) struct H3Fl2VaEngine {
    task: Task,
    model_name: String,
    paths: ModelPaths,
    authority: FrozenH3FactoryAuthority,
    load_strategy: LoadStrategy,
    block_offload: bool,
    loader: Box<dyn H3RuntimeLoader>,
    runtime: Option<Box<dyn H3RuntimeSession>>,
    progress: ProgressReporter,
}

impl H3Fl2VaEngine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_injected(
        model_name: String,
        paths: ModelPaths,
        authority: FrozenH3FactoryAuthority,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        block_offload: bool,
        loader: Box<dyn H3RuntimeLoader>,
    ) -> Result<Self> {
        Self::new_task_injected(
            Task::Fl2va,
            model_name,
            paths,
            authority,
            load_strategy,
            gpu_ordinal,
            block_offload,
            loader,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_ref2va_injected(
        model_name: String,
        paths: ModelPaths,
        authority: FrozenH3FactoryAuthority,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        block_offload: bool,
        loader: Box<dyn H3RuntimeLoader>,
    ) -> Result<Self> {
        Self::new_task_injected(
            Task::Ref2va,
            model_name,
            paths,
            authority,
            load_strategy,
            gpu_ordinal,
            block_offload,
            loader,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_task_injected(
        task: Task,
        model_name: String,
        paths: ModelPaths,
        authority: FrozenH3FactoryAuthority,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        block_offload: bool,
        loader: Box<dyn H3RuntimeLoader>,
    ) -> Result<Self> {
        authority.validate_engine_seam(&model_name, gpu_ordinal, block_offload)?;
        let model_task = contract::task_for_model(&model_name)
            .context("MiniMax H3 engine model has no task contract")?;
        if model_task != task {
            bail!("MiniMax H3 engine constructor and model task disagree");
        }
        Ok(Self {
            task,
            model_name,
            paths,
            authority,
            load_strategy,
            block_offload,
            loader,
            runtime: None,
            progress: ProgressReporter::default(),
        })
    }

    pub(crate) fn from_factory_request(
        request: H3RuntimeLoadRequest<'_>,
        loader: Box<dyn H3RuntimeLoader>,
    ) -> Result<Self> {
        let task = contract::task_for_model(request.model_name)
            .context("MiniMax H3 factory request has no task contract")?;
        Self::new_task_injected(
            task,
            request.model_name.to_string(),
            request.paths.clone(),
            request.authority.clone(),
            request.load_strategy,
            request.gpu_ordinal,
            request.block_offload,
            loader,
        )
    }

    fn validate_loaded_runtime(&self, runtime: &dyn H3RuntimeSession) -> Result<()> {
        if runtime.device_id() != self.authority.device_id()
            || runtime.execution_fingerprint() != self.authority.execution_fingerprint()
            || runtime.component_set_identity_sha256()
                != self.authority.component_set_identity_sha256()
        {
            bail!("MiniMax H3 injected runtime differs from the frozen factory authority");
        }
        Ok(())
    }

    fn validate_output(
        &self,
        request: &GenerateRequest,
        expected_mode: Mode,
        expected_reference_fingerprint: Option<&str>,
        output: &H3EngineOutput,
    ) -> Result<()> {
        if output.mode != expected_mode
            || output.width != request.width
            || output.height != request.height
            || output.frames != request.frames.unwrap_or(contract::MIN_FRAMES)
            || output.fps != contract::FIXED_FPS
            || request.seed.is_some_and(|seed| seed != output.seed)
            || output.audio_sample_rate != contract::AUDIO_SAMPLE_RATE_HZ
            || output.audio_channels != contract::AUDIO_CHANNELS
            || output.device_id != self.authority.device_id()
            || output.execution_fingerprint != self.authority.execution_fingerprint()
            || output.component_set_identity_sha256
                != self.authority.component_set_identity_sha256()
            || output.reference_fingerprint.as_deref() != expected_reference_fingerprint
            || output.mp4.is_empty()
            || output.thumbnail_png.is_empty()
        {
            bail!("MiniMax H3 runtime output differs from the frozen request or factory authority");
        }
        Ok(())
    }

    fn generate_bound(
        &mut self,
        request: &GenerateRequest,
        bindings: &[GenerationReferenceBinding],
    ) -> Result<GenerateResponse> {
        let request_contract = match self.task {
            Task::Fl2va => contract::validate_request_contract(request, self.task),
            Task::Ref2va => contract::validate_resolved_request_contract(request, self.task),
        };
        let expected_mode = request_contract
            .map_err(|error| anyhow::anyhow!("{}: {}", error.code, error.message))?;
        let expected_reference_fingerprint = match self.task {
            Task::Fl2va => {
                if !bindings.is_empty() {
                    bail!("MiniMax H3 FL2VA cannot consume Ref2VA media bindings");
                }
                None
            }
            Task::Ref2va => {
                validate_ref2va_bindings(request, bindings)?;
                Some(ref2va_reference_fingerprint(request)?)
            }
        };
        if !self.is_loaded() {
            self.load_for_request(request)?;
        }
        self.progress.checkpoint()?;
        let started = Instant::now();
        let mut observer = H3EngineProgressObserver::new(&self.progress);
        let output = self
            .runtime
            .as_mut()
            .context("MiniMax H3 runtime disappeared after load")?
            .generate(request, bindings, &self.progress, &mut observer)?;
        self.progress.checkpoint()?;
        self.validate_output(
            request,
            expected_mode,
            expected_reference_fingerprint.as_deref(),
            &output,
        )?;
        Ok(GenerateResponse {
            images: Vec::new(),
            video: Some(VideoData {
                data: output.mp4,
                format: OutputFormat::Mp4,
                width: output.width,
                height: output.height,
                frames: output.frames,
                fps: output.fps,
                pipeline: None,
                thumbnail: output.thumbnail_png,
                gif_preview: Vec::new(),
                has_audio: true,
                duration_ms: Some(output.duration_ms),
                audio_sample_rate: Some(output.audio_sample_rate),
                audio_channels: Some(output.audio_channels),
            }),
            audio: None,
            generation_time_ms: started.elapsed().as_millis() as u64,
            model: self.model_name.clone(),
            seed_used: output.seed,
            gpu: None,
        })
    }
}

pub(crate) type H3Ref2VaEngine = H3Fl2VaEngine;

fn ref2va_reference_fingerprint(request: &GenerateRequest) -> Result<String> {
    let references = request
        .references
        .as_deref()
        .context("MiniMax H3 Ref2VA request lost ordered references")?;
    let shapes = contract::reference_prepared_shapes_for_target(
        references,
        request.frames.unwrap_or(contract::MIN_FRAMES),
    )
    .map_err(|error| anyhow::anyhow!("{}: {}", error.code, error.message))?;
    let metadata = references
        .iter()
        .zip(shapes)
        .enumerate()
        .map(|(index, (reference, shape))| {
            let mut metadata = reference.redacted_metadata_lossless(index);
            metadata.prepared_shape = Some(shape);
            metadata
        })
        .collect::<Vec<_>>();
    Ok(mold_core::generation_reference_fingerprint(&metadata))
}

fn validate_ref2va_bindings(
    request: &GenerateRequest,
    bindings: &[GenerationReferenceBinding],
) -> Result<()> {
    let references = request
        .references
        .as_deref()
        .context("MiniMax H3 Ref2VA request lost ordered reference descriptors")?;
    if references.len() != bindings.len() {
        bail!(
            "MiniMax H3 Ref2VA received {} private media bindings for {} ordered references",
            bindings.len(),
            references.len()
        );
    }
    let request_metadata = references
        .iter()
        .enumerate()
        .map(|(index, reference)| reference.redacted_metadata_lossless(index))
        .collect::<Vec<_>>();
    let binding_metadata = bindings
        .iter()
        .map(|binding| binding.metadata().clone())
        .collect::<Vec<_>>();
    if binding_metadata != request_metadata
        || mold_core::generation_reference_fingerprint(&binding_metadata)
            != mold_core::generation_reference_fingerprint(&request_metadata)
    {
        bail!("MiniMax H3 Ref2VA private media bindings changed order or provenance");
    }
    Ok(())
}

impl InferenceEngine for H3Fl2VaEngine {
    fn generate(&mut self, request: &GenerateRequest) -> Result<GenerateResponse> {
        if self.task == Task::Ref2va {
            bail!("MiniMax H3 Ref2VA requires authenticated server-resolved media bindings");
        }
        self.generate_bound(request, &[])
    }

    fn generate_with_reference_bindings(
        &mut self,
        request: &GenerateRequest,
        bindings: &[GenerationReferenceBinding],
    ) -> Result<GenerateResponse> {
        self.generate_bound(request, bindings)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_loaded(&self) -> bool {
        self.runtime.is_some()
    }

    fn load(&mut self) -> Result<()> {
        if self.runtime.is_some() {
            return Ok(());
        }
        self.progress.checkpoint()?;
        self.progress
            .stage_start("Loading MiniMax H3 frozen runtime");
        let started = Instant::now();
        let runtime = self.loader.load(
            H3RuntimeLoadRequest {
                model_name: &self.model_name,
                paths: &self.paths,
                authority: &self.authority,
                load_strategy: self.load_strategy,
                gpu_ordinal: self.authority.device_ordinal(),
                block_offload: self.block_offload,
            },
            &self.progress,
        )?;
        self.validate_loaded_runtime(runtime.as_ref())?;
        self.progress.checkpoint()?;
        self.runtime = Some(runtime);
        self.progress
            .stage_done("Loading MiniMax H3 frozen runtime", started.elapsed());
        Ok(())
    }

    fn unload(&mut self) {
        self.runtime = None;
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }

    fn clear_on_progress(&mut self) {
        self.progress.clear_callback();
    }

    fn set_cancellation_token(&mut self, token: InferenceCancellationToken) {
        self.progress.set_cancellation_token(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.progress.clear_cancellation_token();
    }

    fn batch_execution_capability(&self) -> BatchExecutionCapability {
        BatchExecutionCapability::SINGLETON_COOPERATIVE
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.paths)
    }

    fn configured_load_strategy(&self) -> Option<LoadStrategy> {
        Some(self.load_strategy)
    }

    fn configured_block_offload(&self) -> Option<bool> {
        Some(self.block_offload)
    }

    fn configured_execution_fingerprint(&self) -> Option<&str> {
        Some(self.authority.execution_fingerprint())
    }
}

struct H3EngineProgressObserver<'a> {
    progress: &'a ProgressReporter,
    started: HashMap<H3PipelinePhase, Instant>,
    last_completed: HashMap<H3PipelinePhase, usize>,
    denoise_step_started: Option<Instant>,
}

impl<'a> H3EngineProgressObserver<'a> {
    fn new(progress: &'a ProgressReporter) -> Self {
        Self {
            progress,
            started: HashMap::new(),
            last_completed: HashMap::new(),
            denoise_step_started: None,
        }
    }
}

impl H3PipelineObserver for H3EngineProgressObserver<'_> {
    fn observe(&mut self, event: H3PipelineEvent) {
        let now = Instant::now();
        let previous = self.last_completed.get(&event.phase).copied();
        if event.completed == 0 || previous.is_some_and(|value| event.completed < value) {
            self.started.insert(event.phase, now);
            if event.phase == H3PipelinePhase::Denoise {
                self.denoise_step_started = Some(now);
            }
            self.progress.stage_start(pipeline_phase_name(event.phase));
        }
        if event.phase == H3PipelinePhase::Denoise
            && event.completed > 0
            && previous != Some(event.completed)
        {
            let elapsed = self
                .denoise_step_started
                .replace(now)
                .map_or(Duration::ZERO, |started| now.duration_since(started));
            self.progress.emit(ProgressEvent::DenoiseStep {
                step: event.completed,
                total: event.total,
                elapsed,
            });
        }
        if event.completed == event.total && previous != Some(event.completed) {
            let elapsed = self
                .started
                .remove(&event.phase)
                .map_or(Duration::ZERO, |started| started.elapsed());
            match event.phase {
                H3PipelinePhase::QwenEncode | H3PipelinePhase::PromptEncode => {
                    self.progress.phase_done(
                        ProgressPhase::PromptEncode,
                        pipeline_phase_name(event.phase),
                        elapsed,
                    )
                }
                H3PipelinePhase::VisualConditionEncode
                | H3PipelinePhase::ReferenceVisualEncode
                | H3PipelinePhase::ReferenceAudioEncode => self.progress.phase_done(
                    ProgressPhase::Vae,
                    pipeline_phase_name(event.phase),
                    elapsed,
                ),
                H3PipelinePhase::VisualDecode => self.progress.phase_done(
                    ProgressPhase::VisualDecode,
                    pipeline_phase_name(event.phase),
                    elapsed,
                ),
                H3PipelinePhase::AudioDecode => self.progress.phase_done(
                    ProgressPhase::AudioDecode,
                    pipeline_phase_name(event.phase),
                    elapsed,
                ),
                H3PipelinePhase::Mux => self.progress.phase_done(
                    ProgressPhase::Mux,
                    pipeline_phase_name(event.phase),
                    elapsed,
                ),
                _ => self
                    .progress
                    .stage_done(pipeline_phase_name(event.phase), elapsed),
            }
        }
        self.last_completed.insert(event.phase, event.completed);
    }
}

fn pipeline_phase_name(phase: H3PipelinePhase) -> &'static str {
    match phase {
        H3PipelinePhase::Validate => "Validating MiniMax H3 request",
        H3PipelinePhase::EndpointPreprocess => "Preparing MiniMax H3 endpoints",
        H3PipelinePhase::ReferenceDecode => "Decoding MiniMax H3 references",
        H3PipelinePhase::ReferenceDecodeChunk => "Decoding MiniMax H3 reference chunk",
        H3PipelinePhase::ReferencePreprocess => "Preparing MiniMax H3 references",
        H3PipelinePhase::ReferencePreprocessChunk => "Preparing MiniMax H3 reference chunk",
        H3PipelinePhase::QwenEncode => "Encoding MiniMax H3 multimodal conditioning",
        H3PipelinePhase::QwenEncodeChunk => "Encoding MiniMax H3 conditioning chunk",
        H3PipelinePhase::PromptEncode => "Encoding MiniMax H3 prompt",
        H3PipelinePhase::VisualConditionEncode => "Encoding MiniMax H3 visual conditions",
        H3PipelinePhase::VisualConditionEncodeChunk => "Encoding MiniMax H3 visual condition chunk",
        H3PipelinePhase::ReferenceVisualEncode => "Encoding MiniMax H3 visual references",
        H3PipelinePhase::ReferenceVisualEncodeChunk => "Encoding MiniMax H3 visual reference chunk",
        H3PipelinePhase::ReferenceAudioEncode => "Encoding MiniMax H3 audio references",
        H3PipelinePhase::ReferenceAudioEncodeChunk => "Encoding MiniMax H3 audio reference chunk",
        H3PipelinePhase::NoiseAllocation => "Allocating MiniMax H3 synchronized noise",
        H3PipelinePhase::Denoise => "Denoising MiniMax H3 audio-video latents",
        H3PipelinePhase::TransformerBlock => "Streaming MiniMax H3 transformer blocks",
        H3PipelinePhase::VisualDecode => "Decoding MiniMax H3 video",
        H3PipelinePhase::VisualDecodeChunk => "Decoding MiniMax H3 video chunk",
        H3PipelinePhase::AudioDecode => "Decoding MiniMax H3 audio",
        H3PipelinePhase::AudioDecodeChunk => "Decoding MiniMax H3 audio chunk",
        H3PipelinePhase::VideoEncode => "Encoding MiniMax H3 H.264 video",
        H3PipelinePhase::Mux => "Muxing MiniMax H3 synchronized audio-video",
        H3PipelinePhase::Staged => "Staging MiniMax H3 audio-video output",
        H3PipelinePhase::Complete => "Completing MiniMax H3 generation",
    }
}

/// Runtime adapter over the landed, runtime-neutral FL2VA pipeline.
pub(crate) struct H3PipelineRuntime<B> {
    backend: B,
    identity: H3PipelineBackendIdentity,
    component_set_identity_sha256: String,
}

/// Explicit marker for a component backend whose main DiT path is backed by
/// `H3BlockStreamState`. The eager landed Candle transformer intentionally
/// does not implement this marker.
pub(crate) trait H3StreamedFl2VaBackend: H3Fl2VaBackend + Send + Sync {
    fn component_set_identity_sha256(&self) -> &str;
}

impl<B> H3PipelineRuntime<B>
where
    B: H3StreamedFl2VaBackend,
{
    pub(crate) fn new(backend: B, authority: &FrozenH3FactoryAuthority) -> Result<Self> {
        let identity = backend.identity();
        if identity.device_id != authority.device_id()
            || identity.execution_fingerprint != authority.execution_fingerprint()
            || backend.component_set_identity_sha256() != authority.component_set_identity_sha256()
        {
            bail!("MiniMax H3 pipeline backend differs from frozen factory authority");
        }
        Ok(Self {
            backend,
            identity,
            component_set_identity_sha256: authority.component_set_identity_sha256().to_string(),
        })
    }
}

impl<B> H3RuntimeSession for H3PipelineRuntime<B>
where
    B: H3StreamedFl2VaBackend,
{
    fn device_id(&self) -> &str {
        &self.identity.device_id
    }

    fn execution_fingerprint(&self) -> &str {
        &self.identity.execution_fingerprint
    }

    fn component_set_identity_sha256(&self) -> &str {
        &self.component_set_identity_sha256
    }

    fn generate(
        &mut self,
        request: &GenerateRequest,
        bindings: &[GenerationReferenceBinding],
        progress: &ProgressReporter,
        observer: &mut dyn H3PipelineObserver,
    ) -> Result<H3EngineOutput> {
        if !bindings.is_empty() {
            bail!("MiniMax H3 FL2VA runtime received Ref2VA media bindings");
        }
        #[cfg(not(feature = "mp4"))]
        {
            let _ = (request, progress, observer);
            bail!("MiniMax H3 synchronized audio-video requires Mold's mp4 feature")
        }
        #[cfg(feature = "mp4")]
        {
            let prepared = pipeline::prepare_request(request, progress, observer)?;
            let mode = prepared.geometry.mode;
            let staged =
                pipeline::execute_staged(&prepared, &mut self.backend, progress, observer)?;
            let output = pipeline::finalize_av(staged, progress, observer)?;
            let duration_ms = u64::from(1_000_u16)
                .checked_mul(output.mux_report.video_duration_ticks)
                .context("MiniMax H3 output duration overflow")?
                / u64::from(output.mux_report.video_timescale);
            Ok(H3EngineOutput {
                mp4: output.mp4,
                thumbnail_png: output.thumbnail_png,
                mode,
                seed: output.provenance.seed,
                width: u32::try_from(output.provenance.width)
                    .context("MiniMax H3 output width exceeds u32")?,
                height: u32::try_from(output.provenance.height)
                    .context("MiniMax H3 output height exceeds u32")?,
                frames: u32::try_from(output.provenance.frames)
                    .context("MiniMax H3 output frames exceed u32")?,
                fps: output.provenance.fps,
                duration_ms,
                audio_sample_rate: output.mux_report.sample_rate,
                audio_channels: u32::from(output.mux_report.channels),
                device_id: output.provenance.device_id,
                execution_fingerprint: output.provenance.execution_fingerprint,
                component_set_identity_sha256: self.component_set_identity_sha256.clone(),
                reference_fingerprint: output.provenance.reference_fingerprint,
            })
        }
    }
}

/// Runtime adapter over the landed ordered Ref2VA pipeline. Like the FL2VA
/// adapter, it is constructable only from an injected, block-streamed backend.
pub(crate) struct H3Ref2VaPipelineRuntime<B> {
    backend: B,
    identity: H3PipelineBackendIdentity,
    component_set_identity_sha256: String,
}

pub(crate) trait H3StreamedRef2VaBackend: H3Ref2VaBackend + Send + Sync {
    fn component_set_identity_sha256(&self) -> &str;
}

impl<B> H3Ref2VaPipelineRuntime<B>
where
    B: H3StreamedRef2VaBackend,
{
    pub(crate) fn new(backend: B, authority: &FrozenH3FactoryAuthority) -> Result<Self> {
        let identity = backend.identity();
        if authority.task() != Task::Ref2va
            || identity.device_id != authority.device_id()
            || identity.execution_fingerprint != authority.execution_fingerprint()
            || backend.component_set_identity_sha256() != authority.component_set_identity_sha256()
        {
            bail!("MiniMax H3 Ref2VA backend differs from frozen factory authority");
        }
        Ok(Self {
            backend,
            identity,
            component_set_identity_sha256: authority.component_set_identity_sha256().to_string(),
        })
    }
}

impl<B> H3RuntimeSession for H3Ref2VaPipelineRuntime<B>
where
    B: H3StreamedRef2VaBackend,
{
    fn device_id(&self) -> &str {
        &self.identity.device_id
    }

    fn execution_fingerprint(&self) -> &str {
        &self.identity.execution_fingerprint
    }

    fn component_set_identity_sha256(&self) -> &str {
        &self.component_set_identity_sha256
    }

    fn generate(
        &mut self,
        request: &GenerateRequest,
        bindings: &[GenerationReferenceBinding],
        progress: &ProgressReporter,
        observer: &mut dyn H3PipelineObserver,
    ) -> Result<H3EngineOutput> {
        #[cfg(not(feature = "mp4"))]
        {
            let _ = (request, bindings, progress, observer);
            bail!("MiniMax H3 synchronized audio-video requires Mold's mp4 feature")
        }
        #[cfg(feature = "mp4")]
        {
            let prepared = ref2va_pipeline::prepare_resolved_request(request, progress, observer)?;
            let staged = ref2va_pipeline::execute_staged(
                &prepared,
                bindings,
                &mut self.backend,
                progress,
                observer,
            )?;
            let output = pipeline::finalize_av(staged, progress, observer)?;
            h3_engine_output_from_pipeline(
                output,
                Mode::ReferenceToAudioVideo,
                &self.component_set_identity_sha256,
            )
        }
    }
}

#[cfg(feature = "mp4")]
fn h3_engine_output_from_pipeline(
    output: super::pipeline::H3PipelineOutput,
    mode: Mode,
    component_set_identity_sha256: &str,
) -> Result<H3EngineOutput> {
    let duration_ms = u64::from(1_000_u16)
        .checked_mul(output.mux_report.video_duration_ticks)
        .context("MiniMax H3 output duration overflow")?
        / u64::from(output.mux_report.video_timescale);
    Ok(H3EngineOutput {
        mp4: output.mp4,
        thumbnail_png: output.thumbnail_png,
        mode,
        seed: output.provenance.seed,
        width: u32::try_from(output.provenance.width)
            .context("MiniMax H3 output width exceeds u32")?,
        height: u32::try_from(output.provenance.height)
            .context("MiniMax H3 output height exceeds u32")?,
        frames: u32::try_from(output.provenance.frames)
            .context("MiniMax H3 output frames exceed u32")?,
        fps: output.provenance.fps,
        duration_ms,
        audio_sample_rate: output.mux_report.sample_rate,
        audio_channels: u32::from(output.mux_report.channels),
        device_id: output.provenance.device_id,
        execution_fingerprint: output.provenance.execution_fingerprint,
        component_set_identity_sha256: component_set_identity_sha256.to_string(),
        reference_fingerprint: output.provenance.reference_fingerprint,
    })
}

/// Main-block executor paired with a cancellation-safe block loader.
/// Implementations own any per-step hidden state; every block object passed to
/// `forward_block` is the exact object whose residency is managed by
/// `H3BlockStreamState`.
pub(crate) trait H3StreamedTransformerExecutor<Block>: Send + Sync {
    fn begin_step(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()>;

    fn forward_block(
        &mut self,
        index: usize,
        block: &Block,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()>;

    fn finish_step(
        &mut self,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput>;

    fn abort_step(&mut self);
}

pub(crate) trait H3StreamedDenoiser: Send + Sync {
    fn identity(&self) -> H3PipelineBackendIdentity;
    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput>;
}

pub(crate) struct H3BlockStreamedDenoiser<L, A, E>
where
    L: H3BlockLoader,
    A: H3BlockLease,
    E: H3StreamedTransformerExecutor<L::Block>,
{
    identity: H3PipelineBackendIdentity,
    stream: H3BlockStreamState<L, A>,
    executor: E,
}

impl<L, A, E> H3BlockStreamedDenoiser<L, A, E>
where
    L: H3BlockLoader,
    A: H3BlockLease,
    E: H3StreamedTransformerExecutor<L::Block>,
{
    pub(crate) fn new(
        identity: H3PipelineBackendIdentity,
        plan: FrozenH3BlockStreamingPlan,
        lease: A,
        loader: L,
        executor: E,
    ) -> Result<Self> {
        if identity.device_id != plan.device_id
            || identity.execution_fingerprint != plan.execution_fingerprint
        {
            bail!("MiniMax H3 streamed denoiser differs from the frozen backend route");
        }
        Ok(Self {
            identity,
            stream: H3BlockStreamState::new(plan, lease, loader)?,
            executor,
        })
    }
}

impl<L, A, E> H3StreamedDenoiser for H3BlockStreamedDenoiser<L, A, E>
where
    L: H3BlockLoader + Send + Sync,
    L::Block: Send + Sync,
    A: H3BlockLease + Send + Sync,
    E: H3StreamedTransformerExecutor<L::Block>,
{
    fn identity(&self) -> H3PipelineBackendIdentity {
        self.identity.clone()
    }

    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput> {
        let control = H3StreamPipelineControl::new(checkpoint);
        self.stream.prepare(&control, |_| {})?;
        if let Err(error) = control
            .with_checkpoint(|checkpoint| self.executor.begin_step(input, layout, checkpoint))
        {
            self.executor.abort_step();
            return Err(error);
        }
        let run = self.stream.run_step(
            &control,
            |event| control.observe(event),
            |index, block| {
                control.with_checkpoint(|checkpoint| {
                    self.executor.forward_block(index, block, checkpoint)
                })
            },
        );
        if let Err(error) = run {
            self.executor.abort_step();
            return Err(error);
        }
        match control.with_checkpoint(|checkpoint| self.executor.finish_step(checkpoint)) {
            Ok(output) => Ok(output),
            Err(error) => {
                self.executor.abort_step();
                Err(error)
            }
        }
    }
}

struct H3StreamPipelineControl<'a> {
    checkpoint: RefCell<&'a mut dyn H3PipelineCheckpoint>,
}

impl<'a> H3StreamPipelineControl<'a> {
    fn new(checkpoint: &'a mut dyn H3PipelineCheckpoint) -> Self {
        Self {
            checkpoint: RefCell::new(checkpoint),
        }
    }

    fn with_checkpoint<T>(
        &self,
        operation: impl FnOnce(&mut dyn H3PipelineCheckpoint) -> Result<T>,
    ) -> Result<T> {
        operation(&mut **self.checkpoint.borrow_mut())
    }

    fn observe(&self, _event: H3BlockStreamEvent) {
        // Before/after checkpoints below are the fallible progress authority.
        // The ownership observer remains side-effect free because the
        // H3BlockStreamState callback cannot return an error.
    }
}

impl H3StreamCancellation for H3StreamPipelineControl<'_> {
    fn checkpoint(&self, point: H3StreamCheckpoint) -> Result<()> {
        let index = match point {
            H3StreamCheckpoint::BeforeResidentLoad(index)
            | H3StreamCheckpoint::BeforePrefetch(index)
            | H3StreamCheckpoint::BeforePrefetchWait(index)
            | H3StreamCheckpoint::BeforeBlock(index) => index,
            H3StreamCheckpoint::AfterResidentLoad(index)
            | H3StreamCheckpoint::AfterPrefetch(index)
            | H3StreamCheckpoint::AfterPrefetchWait(index)
            | H3StreamCheckpoint::AfterBlock(index) => index.saturating_add(1),
        };
        self.checkpoint.borrow_mut().checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::TransformerBlock,
            completed: index.min(50),
            total: 50,
        })
    }
}

/// Combines the landed component backend with a separately streamed main DiT.
/// The eager backend's `denoise` method is deliberately never called.
pub(crate) struct H3StreamedPipelineBackend<B, D> {
    components: B,
    denoiser: D,
    component_set_identity_sha256: String,
}

impl<B, D> H3StreamedPipelineBackend<B, D>
where
    B: H3Fl2VaBackend,
    D: H3StreamedDenoiser,
{
    pub(crate) fn new(
        components: B,
        denoiser: D,
        component_set_identity_sha256: String,
    ) -> Result<Self> {
        if components.identity() != denoiser.identity() {
            bail!("MiniMax H3 component and streamed-transformer routes disagree");
        }
        if component_set_identity_sha256.len() != 64
            || !component_set_identity_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("MiniMax H3 component-set identity must be one SHA-256 digest");
        }
        Ok(Self {
            components,
            denoiser,
            component_set_identity_sha256,
        })
    }
}

impl<B, D> H3Fl2VaBackend for H3StreamedPipelineBackend<B, D>
where
    B: H3Fl2VaBackend,
    D: H3StreamedDenoiser,
{
    fn identity(&self) -> H3PipelineBackendIdentity {
        self.components.identity()
    }

    fn device(&self) -> &Device {
        self.components.device()
    }

    fn encode_text(
        &mut self,
        prompt: &str,
        endpoints: &[H3PreparedEndpoint],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        self.components.encode_text(prompt, endpoints, checkpoint)
    }

    fn encode_visual_condition(
        &mut self,
        endpoint: &H3PreparedEndpoint,
        mode: mold_candle::minimax_h3::ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor> {
        self.components
            .encode_visual_condition(endpoint, mode, checkpoint)
    }

    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput> {
        if self.components.identity() != self.denoiser.identity() {
            bail!("MiniMax H3 streamed-transformer route changed during inference");
        }
        self.denoiser.denoise(input, layout, checkpoint)
    }

    fn decode_video(
        &mut self,
        latents: &Tensor,
        sink: &mut H3VideoEncodeSink,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        self.components.decode_video(latents, sink, checkpoint)
    }

    fn decode_audio(
        &mut self,
        latents: &StereoLatents,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform> {
        self.components.decode_audio(latents, checkpoint)
    }
}

impl<B, D> H3StreamedFl2VaBackend for H3StreamedPipelineBackend<B, D>
where
    B: H3Fl2VaBackend + Send + Sync,
    D: H3StreamedDenoiser,
{
    fn component_set_identity_sha256(&self) -> &str {
        &self.component_set_identity_sha256
    }
}

/// Combines ordered-reference component operations with the same audited
/// block-streamed main DiT used by FL2VA. Each decode receives the exact
/// request-scoped private binding alongside its frozen payload-free metadata.
pub(crate) struct H3StreamedRef2VaPipelineBackend<B, D> {
    components: B,
    denoiser: D,
    component_set_identity_sha256: String,
}

impl<B, D> H3StreamedRef2VaPipelineBackend<B, D>
where
    B: H3Ref2VaBackend,
    D: H3StreamedDenoiser,
{
    pub(crate) fn new(
        components: B,
        denoiser: D,
        component_set_identity_sha256: String,
    ) -> Result<Self> {
        if components.identity() != denoiser.identity() {
            bail!("MiniMax H3 Ref2VA component and streamed-transformer routes disagree");
        }
        if component_set_identity_sha256.len() != 64
            || !component_set_identity_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("MiniMax H3 component-set identity must be one SHA-256 digest");
        }
        Ok(Self {
            components,
            denoiser,
            component_set_identity_sha256,
        })
    }
}

impl<B, D> H3Ref2VaBackend for H3StreamedRef2VaPipelineBackend<B, D>
where
    B: H3Ref2VaBackend,
    D: H3StreamedDenoiser,
{
    fn identity(&self) -> H3PipelineBackendIdentity {
        self.components.identity()
    }

    fn device(&self) -> &Device {
        self.components.device()
    }

    fn maximum_packed_rows(&self) -> usize {
        self.components.maximum_packed_rows()
    }

    fn decode_reference(
        &mut self,
        reference: &H3PreparedReference,
        binding: &GenerationReferenceBinding,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3DecodedReferenceFacts> {
        self.components
            .decode_reference(reference, binding, checkpoint)
    }

    fn preprocess_reference(
        &mut self,
        reference: &H3PreparedReference,
        decoded: &H3DecodedReferenceFacts,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3ReferencePresentation> {
        self.components
            .preprocess_reference(reference, decoded, checkpoint)
    }

    fn encode_text(
        &mut self,
        prompt: &str,
        references: &[H3ReferencePresentation],
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TextConditioning> {
        self.components.encode_text(prompt, references, checkpoint)
    }

    fn encode_visual_reference(
        &mut self,
        reference: &H3PreparedReference,
        mode: mold_candle::minimax_h3::ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor> {
        self.components
            .encode_visual_reference(reference, mode, checkpoint)
    }

    fn encode_audio_reference(
        &mut self,
        reference: &H3PreparedReference,
        mode: H3AudioConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoLatents> {
        self.components
            .encode_audio_reference(reference, mode, checkpoint)
    }

    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput> {
        if self.components.identity() != self.denoiser.identity() {
            bail!("MiniMax H3 Ref2VA streamed-transformer route changed during inference");
        }
        self.denoiser.denoise(input, layout, checkpoint)
    }

    fn decode_video(
        &mut self,
        latents: &Tensor,
        sink: &mut H3VideoEncodeSink,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        self.components.decode_video(latents, sink, checkpoint)
    }

    fn decode_audio(
        &mut self,
        latents: &StereoLatents,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform> {
        self.components.decode_audio(latents, checkpoint)
    }
}

impl<B, D> H3StreamedRef2VaBackend for H3StreamedRef2VaPipelineBackend<B, D>
where
    B: H3Ref2VaBackend + Send + Sync,
    D: H3StreamedDenoiser,
{
    fn component_set_identity_sha256(&self) -> &str {
        &self.component_set_identity_sha256
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use candle_core::DType;
    use image::{Rgb, RgbImage};
    use mold_candle::minimax_h3::{
        AudioSoundtrackAssociation, ConditionEncodeMode, H3Modality, H3ModalityTag, H3PackedLayout,
        RefPresentation, RefPresentationKind,
    };
    use serde_json::json;

    use super::*;
    use crate::progress::{is_inference_cancelled, InferenceCancelled};
    use crate::{
        H3FactoryAuthorityInput, H3FactoryComponentAuthority, H3FactoryComponentRole,
        H3FactoryConditionerPlacement, H3FactoryQuantizationAuthority,
    };

    const EXECUTION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn authority_for(model: &str) -> FrozenH3FactoryAuthority {
        let frozen = crate::FrozenEngineConfig::resolve(model, &mold_core::Config::default());
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: model.into(),
            device_id: "gpu-0".into(),
            device_ordinal: 0,
            compute_capability: (8, 9),
            execution_fingerprint: EXECUTION.into(),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 2_048,
            qwen_host_resident_parameter_bytes: 2_048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1_024,
            qwen_output_text_rows: 1,
            qwen_vision_rows: 0,
            condition_visual_rows: 0,
            resident_block_count: 2,
            prefetch_depth: 1,
            attention_backend: frozen.attention_backend,
            attention_chunk: frozen.attention_chunk,
            attention_kernel_identity: "synthetic-lossless-packed-attention-v1".into(),
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

    fn authority() -> FrozenH3FactoryAuthority {
        authority_for(contract::FL2VA_COMFY)
    }

    fn paths() -> ModelPaths {
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: PathBuf::from("/definitely/not/read/minimax-h3/transformer"),
            transformer_shards: Vec::new(),
            vae: PathBuf::from("/definitely/not/read/minimax-h3/vae"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        }
    }

    fn request() -> GenerateRequest {
        serde_json::from_value(json!({
            "prompt": "a lighthouse in a storm",
            "model": contract::FL2VA_COMFY,
            "width": 32,
            "height": 32,
            "steps": 4,
            "guidance": 0.0,
            "seed": 7,
            "batch_size": 1,
            "output_format": "mp4",
            "strength": 1.0,
            "frames": 124,
            "fps": contract::FIXED_FPS
        }))
        .unwrap()
    }

    fn ref2va_request() -> GenerateRequest {
        use mold_core::{
            GenerationReference, GenerationReferenceAuthority, GenerationReferenceProvenance,
        };

        let provenance = |name: &str, digest: char| GenerationReferenceProvenance {
            name: Some(name.to_string()),
            sha256: Some(sha(digest)),
        };
        let mut request = request();
        request.model = contract::REF2VA_COMFY.into();
        request.references = Some(vec![
            GenerationReference::Video {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: provenance("motion.mp4", '1'),
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
            },
            GenerationReference::Image {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: provenance("portrait.png", '2'),
                mime_type: "image/png".into(),
                width: 48,
                height: 48,
            },
            GenerationReference::Audio {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: provenance("voice.wav", '3'),
                mime_type: "audio/wav".into(),
                duration_ms: 2_000,
                sample_rate: 48_000,
                channels: 1,
                sample_count: Some(96_000),
            },
        ]);
        request
    }

    fn image_ref2va_request() -> GenerateRequest {
        use mold_core::{
            GenerationReference, GenerationReferenceAuthority, GenerationReferenceProvenance,
        };

        let mut request = request();
        request.model = contract::REF2VA_COMFY.into();
        request.references = Some(vec![GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: GenerationReferenceProvenance {
                name: Some("portrait.png".to_string()),
                sha256: Some(sha('2')),
            },
            mime_type: "image/png".into(),
            width: 48,
            height: 48,
        }]);
        request
    }

    fn ref2va_bindings(request: &GenerateRequest) -> Vec<GenerationReferenceBinding> {
        request
            .references
            .as_deref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, reference)| {
                GenerationReferenceBinding::synthetic(reference.redacted_metadata_lossless(index))
            })
            .collect()
    }

    struct SyntheticRuntime {
        device: String,
        execution: String,
        components: String,
    }

    impl H3RuntimeSession for SyntheticRuntime {
        fn device_id(&self) -> &str {
            &self.device
        }

        fn execution_fingerprint(&self) -> &str {
            &self.execution
        }

        fn component_set_identity_sha256(&self) -> &str {
            &self.components
        }

        fn generate(
            &mut self,
            request: &GenerateRequest,
            bindings: &[GenerationReferenceBinding],
            progress: &ProgressReporter,
            observer: &mut dyn H3PipelineObserver,
        ) -> Result<H3EngineOutput> {
            assert!(bindings.is_empty());
            progress.checkpoint()?;
            for event in [
                H3PipelineEvent {
                    phase: H3PipelinePhase::Validate,
                    completed: 0,
                    total: 1,
                },
                H3PipelineEvent {
                    phase: H3PipelinePhase::Validate,
                    completed: 1,
                    total: 1,
                },
                H3PipelineEvent {
                    phase: H3PipelinePhase::Denoise,
                    completed: 0,
                    total: 3,
                },
                H3PipelineEvent {
                    phase: H3PipelinePhase::Denoise,
                    completed: 1,
                    total: 3,
                },
                H3PipelineEvent {
                    phase: H3PipelinePhase::Denoise,
                    completed: 2,
                    total: 3,
                },
                H3PipelineEvent {
                    phase: H3PipelinePhase::Denoise,
                    completed: 3,
                    total: 3,
                },
                H3PipelineEvent {
                    phase: H3PipelinePhase::Complete,
                    completed: 1,
                    total: 1,
                },
            ] {
                observer.observe(event);
                progress.checkpoint()?;
            }
            Ok(H3EngineOutput {
                mp4: vec![0, 0, 0, 1],
                thumbnail_png: vec![137, 80, 78, 71],
                mode: Mode::TextToAudioVideo,
                seed: request.seed.unwrap(),
                width: request.width,
                height: request.height,
                frames: request.frames.unwrap(),
                fps: request.fps.unwrap(),
                duration_ms: 5_166,
                audio_sample_rate: contract::AUDIO_SAMPLE_RATE_HZ,
                audio_channels: contract::AUDIO_CHANNELS,
                device_id: self.device.clone(),
                execution_fingerprint: self.execution.clone(),
                component_set_identity_sha256: self.components.clone(),
                reference_fingerprint: None,
            })
        }
    }

    struct SyntheticLoader {
        loads: Arc<AtomicUsize>,
        wrong_device: bool,
    }

    struct SyntheticRefRuntime {
        device: String,
        execution: String,
        components: String,
        observed: Arc<Mutex<Vec<(u32, mold_core::GenerationReferenceKind)>>>,
        wrong_reference_fingerprint: bool,
    }

    impl H3RuntimeSession for SyntheticRefRuntime {
        fn device_id(&self) -> &str {
            &self.device
        }

        fn execution_fingerprint(&self) -> &str {
            &self.execution
        }

        fn component_set_identity_sha256(&self) -> &str {
            &self.components
        }

        fn generate(
            &mut self,
            request: &GenerateRequest,
            bindings: &[GenerationReferenceBinding],
            progress: &ProgressReporter,
            observer: &mut dyn H3PipelineObserver,
        ) -> Result<H3EngineOutput> {
            progress.checkpoint()?;
            let observed = bindings
                .iter()
                .map(|binding| {
                    assert!(binding.file().metadata().unwrap().is_file());
                    (binding.metadata().index, binding.metadata().kind)
                })
                .collect::<Vec<_>>();
            *self.observed.lock().unwrap() = observed;
            observer.observe(H3PipelineEvent {
                phase: H3PipelinePhase::QwenEncode,
                completed: 0,
                total: 1,
            });
            progress.checkpoint()?;
            observer.observe(H3PipelineEvent {
                phase: H3PipelinePhase::QwenEncode,
                completed: 1,
                total: 1,
            });
            progress.checkpoint()?;
            Ok(H3EngineOutput {
                mp4: vec![0, 0, 0, 1],
                thumbnail_png: vec![137, 80, 78, 71],
                mode: Mode::ReferenceToAudioVideo,
                seed: request.seed.unwrap(),
                width: request.width,
                height: request.height,
                frames: request.frames.unwrap(),
                fps: request.fps.unwrap(),
                duration_ms: 5_166,
                audio_sample_rate: contract::AUDIO_SAMPLE_RATE_HZ,
                audio_channels: contract::AUDIO_CHANNELS,
                device_id: self.device.clone(),
                execution_fingerprint: self.execution.clone(),
                component_set_identity_sha256: self.components.clone(),
                reference_fingerprint: Some(if self.wrong_reference_fingerprint {
                    sha('f')
                } else {
                    ref2va_reference_fingerprint(request)?
                }),
            })
        }
    }

    struct SyntheticRefLoader {
        loads: Arc<AtomicUsize>,
        observed: Arc<Mutex<Vec<(u32, mold_core::GenerationReferenceKind)>>>,
        wrong_reference_fingerprint: bool,
    }

    impl H3RuntimeLoader for SyntheticRefLoader {
        fn load(
            &mut self,
            request: H3RuntimeLoadRequest<'_>,
            progress: &ProgressReporter,
        ) -> Result<Box<dyn H3RuntimeSession>> {
            progress.checkpoint()?;
            assert_eq!(request.model_name, contract::REF2VA_COMFY);
            assert_eq!(request.authority.task(), Task::Ref2va);
            assert_eq!(request.gpu_ordinal, 0);
            assert!(request.block_offload);
            self.loads.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(SyntheticRefRuntime {
                device: request.authority.device_id().into(),
                execution: request.authority.execution_fingerprint().into(),
                components: request.authority.component_set_identity_sha256().into(),
                observed: Arc::clone(&self.observed),
                wrong_reference_fingerprint: self.wrong_reference_fingerprint,
            }))
        }
    }

    impl H3RuntimeLoader for SyntheticLoader {
        fn load(
            &mut self,
            request: H3RuntimeLoadRequest<'_>,
            progress: &ProgressReporter,
        ) -> Result<Box<dyn H3RuntimeSession>> {
            progress.checkpoint()?;
            assert_eq!(request.model_name, contract::FL2VA_COMFY);
            assert_eq!(request.load_strategy, LoadStrategy::Eager);
            assert_eq!(request.gpu_ordinal, 0);
            assert!(request.block_offload);
            assert_eq!(
                request.paths.transformer,
                PathBuf::from("/definitely/not/read/minimax-h3/transformer")
            );
            self.loads.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(SyntheticRuntime {
                device: if self.wrong_device {
                    "gpu-1".into()
                } else {
                    request.authority.device_id().into()
                },
                execution: request.authority.execution_fingerprint().into(),
                components: request.authority.component_set_identity_sha256().into(),
            }))
        }
    }

    fn engine(loads: Arc<AtomicUsize>, wrong_device: bool) -> H3Fl2VaEngine {
        H3Fl2VaEngine::new_injected(
            contract::FL2VA_COMFY.into(),
            paths(),
            authority(),
            LoadStrategy::Eager,
            0,
            true,
            Box::new(SyntheticLoader {
                loads,
                wrong_device,
            }),
        )
        .unwrap()
    }

    fn ref2va_engine(
        loads: Arc<AtomicUsize>,
        observed: Arc<Mutex<Vec<(u32, mold_core::GenerationReferenceKind)>>>,
    ) -> H3Ref2VaEngine {
        ref2va_engine_with_output_fingerprint(loads, observed, false)
    }

    fn ref2va_engine_with_output_fingerprint(
        loads: Arc<AtomicUsize>,
        observed: Arc<Mutex<Vec<(u32, mold_core::GenerationReferenceKind)>>>,
        wrong_reference_fingerprint: bool,
    ) -> H3Ref2VaEngine {
        H3Ref2VaEngine::new_ref2va_injected(
            contract::REF2VA_COMFY.into(),
            paths(),
            authority_for(contract::REF2VA_COMFY),
            LoadStrategy::Eager,
            0,
            true,
            Box::new(SyntheticRefLoader {
                loads,
                observed,
                wrong_reference_fingerprint,
            }),
        )
        .unwrap()
    }

    #[test]
    fn progress_observer_emits_independent_h3_runtime_phases() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let mut progress = ProgressReporter::default();
        progress.set_callback(Box::new(move |event| {
            captured.lock().unwrap().push(event);
        }));
        let mut observer = H3EngineProgressObserver::new(&progress);

        for phase in [
            H3PipelinePhase::PromptEncode,
            H3PipelinePhase::VisualDecode,
            H3PipelinePhase::AudioDecode,
            H3PipelinePhase::Mux,
        ] {
            observer.observe(H3PipelineEvent {
                phase,
                completed: 0,
                total: 1,
            });
            let chunk = match phase {
                H3PipelinePhase::VisualDecode => Some(H3PipelinePhase::VisualDecodeChunk),
                H3PipelinePhase::AudioDecode => Some(H3PipelinePhase::AudioDecodeChunk),
                _ => None,
            };
            if let Some(chunk) = chunk {
                observer.observe(H3PipelineEvent {
                    phase: chunk,
                    completed: 0,
                    total: 1,
                });
                observer.observe(H3PipelineEvent {
                    phase: chunk,
                    completed: 1,
                    total: 1,
                });
            }
            observer.observe(H3PipelineEvent {
                phase,
                completed: 1,
                total: 1,
            });
        }

        observer.observe(H3PipelineEvent {
            phase: H3PipelinePhase::Denoise,
            completed: 0,
            total: 1,
        });
        std::thread::sleep(Duration::from_millis(1));
        observer.observe(H3PipelineEvent {
            phase: H3PipelinePhase::Denoise,
            completed: 1,
            total: 1,
        });

        let events = events.lock().unwrap();
        let typed = events
            .iter()
            .filter_map(|event| match event {
                ProgressEvent::PhaseDone { phase, .. } => Some(*phase),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            typed,
            vec![
                ProgressPhase::PromptEncode,
                ProgressPhase::VisualDecode,
                ProgressPhase::AudioDecode,
                ProgressPhase::Mux,
            ],
            "nested decoder chunks must remain display-only instead of double-counting telemetry"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            ProgressEvent::DenoiseStep { elapsed, .. } if *elapsed > Duration::ZERO
        )));
    }

    #[test]
    fn inference_adapter_preserves_frozen_route_and_publishes_video_contract() {
        let loads = Arc::new(AtomicUsize::new(0));
        let mut engine = engine(Arc::clone(&loads), false);
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        engine.set_on_progress(Box::new(move |event| {
            captured.lock().unwrap().push(event);
        }));

        for _ in 0..2 {
            let response = engine.generate(&request()).unwrap();
            assert!(response.images.is_empty());
            assert_eq!(response.seed_used, 7);
            let video = response.video.unwrap();
            assert_eq!((video.width, video.height), (32, 32));
            assert_eq!((video.frames, video.fps), (124, contract::FIXED_FPS));
            assert!(video.has_audio);
            assert_eq!(
                video.audio_sample_rate,
                Some(contract::AUDIO_SAMPLE_RATE_HZ)
            );
            assert_eq!(video.audio_channels, Some(contract::AUDIO_CHANNELS));
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(
            engine.batch_execution_capability(),
            BatchExecutionCapability::SINGLETON_COOPERATIVE
        );
        assert_eq!(engine.configured_execution_fingerprint(), Some(EXECUTION));
        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            ProgressEvent::StageStart { name } if name.contains("frozen runtime")
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, ProgressEvent::DenoiseStep { .. }))
                .count(),
            6
        );
    }

    #[test]
    fn cancellation_and_runtime_reroute_fail_without_publication_or_cache_state() {
        let loads = Arc::new(AtomicUsize::new(0));
        let mut cancelled = engine(Arc::clone(&loads), false);
        let token = InferenceCancellationToken::default();
        token.cancel();
        cancelled.set_cancellation_token(token);
        let error = cancelled.generate(&request()).unwrap_err();
        assert!(is_inference_cancelled(&error));
        assert_eq!(loads.load(Ordering::SeqCst), 0);

        assert!(!cancelled.is_loaded());

        let mut rerouted = engine(Arc::clone(&loads), true);
        let error = rerouted.load().unwrap_err();
        assert!(error.to_string().contains("frozen factory authority"));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(!rerouted.is_loaded());
    }

    #[test]
    fn ref2va_adapter_requires_exact_private_bindings_and_preserves_ordered_av_contract() {
        let request = ref2va_request();
        let bindings = ref2va_bindings(&request);
        assert!(!format!("{:?}", bindings[0]).contains("/synthetic/not-read"));
        let loads = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut engine = ref2va_engine(Arc::clone(&loads), Arc::clone(&observed));

        let error = engine.generate(&request).unwrap_err();
        assert!(error.to_string().contains("server-resolved media bindings"));
        assert_eq!(loads.load(Ordering::SeqCst), 0);

        let response = engine
            .generate_with_reference_bindings(&request, &bindings)
            .unwrap();
        let video = response.video.unwrap();
        assert!(video.has_audio);
        assert_eq!(
            video.audio_sample_rate,
            Some(contract::AUDIO_SAMPLE_RATE_HZ)
        );
        assert_eq!(video.audio_channels, Some(contract::AUDIO_CHANNELS));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(
            *observed.lock().unwrap(),
            vec![
                (1, mold_core::GenerationReferenceKind::Video),
                (2, mold_core::GenerationReferenceKind::Image),
                (3, mold_core::GenerationReferenceKind::Audio),
            ]
        );
    }

    #[test]
    fn ref2va_binding_mutation_and_cancellation_fail_before_runtime_load() {
        let request = ref2va_request();
        let loads = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(Mutex::new(Vec::new()));

        let mut missing = ref2va_bindings(&request);
        missing.pop();
        let mut engine = ref2va_engine(Arc::clone(&loads), Arc::clone(&observed));
        let error = engine
            .generate_with_reference_bindings(&request, &missing)
            .unwrap_err();
        assert!(error.to_string().contains("private media bindings"));
        assert_eq!(loads.load(Ordering::SeqCst), 0);

        let mut reordered = ref2va_bindings(&request);
        reordered.swap(0, 1);
        let mut engine = ref2va_engine(Arc::clone(&loads), Arc::clone(&observed));
        let error = engine
            .generate_with_reference_bindings(&request, &reordered)
            .unwrap_err();
        assert!(error.to_string().contains("changed order or provenance"));
        assert_eq!(loads.load(Ordering::SeqCst), 0);

        let mut cancelled = ref2va_engine(Arc::clone(&loads), observed);
        let token = InferenceCancellationToken::default();
        token.cancel();
        cancelled.set_cancellation_token(token);
        let error = cancelled
            .generate_with_reference_bindings(&request, &ref2va_bindings(&request))
            .unwrap_err();
        assert!(is_inference_cancelled(&error));
        assert_eq!(loads.load(Ordering::SeqCst), 0);

        let wrong_loads = Arc::new(AtomicUsize::new(0));
        let mut wrong_output = ref2va_engine_with_output_fingerprint(
            Arc::clone(&wrong_loads),
            Arc::new(Mutex::new(Vec::new())),
            true,
        );
        let error = wrong_output
            .generate_with_reference_bindings(&request, &ref2va_bindings(&request))
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("differs from the frozen request or factory authority"));
        assert_eq!(wrong_loads.load(Ordering::SeqCst), 1);
    }

    #[derive(Clone)]
    struct SyntheticBlock(usize);

    struct SyntheticBlockLoader {
        loaded: Arc<Mutex<Vec<usize>>>,
    }

    impl H3BlockLoader for SyntheticBlockLoader {
        type Block = SyntheticBlock;

        fn load_block(
            &mut self,
            index: usize,
            device_id: &str,
            execution_fingerprint: &str,
        ) -> Result<Self::Block> {
            assert_eq!(device_id, "synthetic-cpu-0");
            assert_eq!(execution_fingerprint, EXECUTION);
            self.loaded.lock().unwrap().push(index);
            Ok(SyntheticBlock(index))
        }
    }

    struct SyntheticLease;

    impl H3BlockLease for SyntheticLease {
        fn lease_id(&self) -> &str {
            "synthetic-lease"
        }

        fn device_id(&self) -> &str {
            "synthetic-cpu-0"
        }

        fn execution_fingerprint(&self) -> &str {
            EXECUTION
        }

        fn is_active(&self) -> bool {
            true
        }
    }

    struct SyntheticExecutor {
        forwarded: Arc<Mutex<Vec<usize>>>,
        input: Option<(Tensor, Tensor)>,
        aborted: Arc<AtomicUsize>,
    }

    impl H3StreamedTransformerExecutor<SyntheticBlock> for SyntheticExecutor {
        fn begin_step(
            &mut self,
            input: H3ForwardInput<'_>,
            _layout: &H3FrozenPackedLayout,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            self.input = Some((input.video_rows.clone(), input.audio_rows.clone()));
            Ok(())
        }

        fn forward_block(
            &mut self,
            index: usize,
            block: &SyntheticBlock,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            assert_eq!(index, block.0);
            self.forwarded.lock().unwrap().push(index);
            Ok(())
        }

        fn finish_step(
            &mut self,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TransformerOutput> {
            let (video, audio) = self.input.take().unwrap();
            Ok(H3TransformerOutput {
                video: Tensor::zeros_like(&video)?,
                audio: Tensor::zeros_like(&audio)?,
            })
        }

        fn abort_step(&mut self) {
            self.input = None;
            self.aborted.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[derive(Default)]
    struct RecordingCheckpoint {
        events: Vec<H3PipelineEvent>,
        cancel_after_block: Option<usize>,
    }

    impl H3PipelineCheckpoint for RecordingCheckpoint {
        fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()> {
            self.events.push(event);
            if self.cancel_after_block == Some(event.completed) {
                Err(InferenceCancelled.into())
            } else {
                Ok(())
            }
        }
    }

    struct StructuralRefComponents {
        device: Device,
        identity: H3PipelineBackendIdentity,
        trace: Arc<Mutex<Vec<&'static str>>>,
    }

    impl H3Ref2VaBackend for StructuralRefComponents {
        fn identity(&self) -> H3PipelineBackendIdentity {
            self.identity.clone()
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn maximum_packed_rows(&self) -> usize {
            100_000
        }

        fn decode_reference(
            &mut self,
            reference: &H3PreparedReference,
            binding: &GenerationReferenceBinding,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3DecodedReferenceFacts> {
            self.trace.lock().unwrap().push("decode-reference");
            let mut bound = binding.metadata().clone();
            bound.prepared_shape = Some(reference.shape.clone());
            assert_eq!(bound, reference.metadata);
            assert!(binding.file().metadata()?.is_file());
            Ok(H3DecodedReferenceFacts {
                index: reference.metadata.index,
                kind: reference.metadata.kind,
                width: reference.metadata.width,
                height: reference.metadata.height,
                frame_count: None,
                fps: None,
                audio: None,
            })
        }

        fn preprocess_reference(
            &mut self,
            reference: &H3PreparedReference,
            decoded: &H3DecodedReferenceFacts,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3ReferencePresentation> {
            self.trace.lock().unwrap().push("preprocess-reference");
            assert_eq!(decoded.index, reference.metadata.index);
            Ok(H3ReferencePresentation {
                index: reference.metadata.index,
                presentation: RefPresentation {
                    kind: RefPresentationKind::Image { vision_tokens: 2 },
                    has_audio: false,
                },
            })
        }

        fn encode_text(
            &mut self,
            prompt: &str,
            references: &[H3ReferencePresentation],
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TextConditioning> {
            self.trace.lock().unwrap().push("encode-text");
            assert!(!prompt.is_empty());
            assert_eq!(references.len(), 1);
            let tags = vec![
                H3ModalityTag::Text,
                H3ModalityTag::Text,
                H3ModalityTag::Vision,
                H3ModalityTag::Vision,
            ];
            Ok(H3TextConditioning {
                states: Tensor::zeros((1, tags.len(), 5_120), DType::F32, &self.device)?,
                tags,
                #[cfg(test)]
                lifetime_probe: None,
            })
        }

        fn encode_visual_reference(
            &mut self,
            reference: &H3PreparedReference,
            mode: ConditionEncodeMode,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<Tensor> {
            self.trace.lock().unwrap().push("encode-visual");
            assert_eq!(mode, ConditionEncodeMode::OfficialFreshSeed42);
            let height = usize::try_from(reference.shape.normalized_height.unwrap())? / 16;
            let width = usize::try_from(reference.shape.normalized_width.unwrap())? / 16;
            Tensor::zeros((1, 24, 1, height, width), DType::F32, &self.device).map_err(Into::into)
        }

        fn encode_audio_reference(
            &mut self,
            _reference: &H3PreparedReference,
            _mode: H3AudioConditionEncodeMode,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<StereoLatents> {
            unreachable!("structural composition test does not encode audio")
        }

        fn denoise(
            &mut self,
            _input: H3ForwardInput<'_>,
            _layout: &H3FrozenPackedLayout,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TransformerOutput> {
            unreachable!("the composed backend must route denoise to its streamed authority")
        }

        fn decode_video(
            &mut self,
            latents: &Tensor,
            sink: &mut H3VideoEncodeSink,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            self.trace.lock().unwrap().push("decode-video");
            assert_eq!(latents.dims(), [1, 24, 37, 2, 2]);
            for frame in 0..124 {
                sink.push(
                    &RgbImage::from_pixel(32, 32, Rgb([frame as u8, 2, 3])),
                    _checkpoint,
                )?;
            }
            Ok(())
        }

        fn decode_audio(
            &mut self,
            latents: &StereoLatents,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<StereoWaveform> {
            self.trace.lock().unwrap().push("decode-audio");
            assert_eq!(latents.normalized().dims(), [1, 32, 2, 207]);
            StereoWaveform::new(
                Tensor::zeros((1, 2, 207 * 800), DType::F32, &self.device)?,
                contract::AUDIO_SAMPLE_RATE_HZ as usize,
                AudioSoundtrackAssociation::Generated,
            )
            .map_err(Into::into)
        }
    }

    struct StructuralDenoiser {
        identity: H3PipelineBackendIdentity,
        trace: Arc<Mutex<Vec<&'static str>>>,
    }

    impl H3StreamedDenoiser for StructuralDenoiser {
        fn identity(&self) -> H3PipelineBackendIdentity {
            self.identity.clone()
        }

        fn denoise(
            &mut self,
            input: H3ForwardInput<'_>,
            _layout: &H3FrozenPackedLayout,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TransformerOutput> {
            self.trace.lock().unwrap().push("streamed-denoise");
            Ok(H3TransformerOutput {
                video: Tensor::zeros_like(input.video_rows)?,
                audio: Tensor::zeros_like(input.audio_rows)?,
            })
        }
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn ref2va_runtime_executes_exact_composed_streamed_backend_authority() {
        let authority = authority_for(contract::REF2VA_COMFY);
        let identity = H3PipelineBackendIdentity {
            kind: crate::minimax_h3::pipeline::H3PipelineBackendKind::SyntheticCpu,
            device_id: authority.device_id().to_string(),
            execution_fingerprint: authority.execution_fingerprint().to_string(),
        };
        let trace = Arc::new(Mutex::new(Vec::new()));
        let composed = H3StreamedRef2VaPipelineBackend::new(
            StructuralRefComponents {
                device: Device::Cpu,
                identity: identity.clone(),
                trace: Arc::clone(&trace),
            },
            StructuralDenoiser {
                identity,
                trace: Arc::clone(&trace),
            },
            authority.component_set_identity_sha256().to_string(),
        )
        .unwrap();
        let mut runtime = H3Ref2VaPipelineRuntime::new(composed, &authority).unwrap();
        assert_eq!(runtime.device_id(), authority.device_id());
        assert_eq!(
            runtime.execution_fingerprint(),
            authority.execution_fingerprint()
        );
        assert_eq!(
            runtime.component_set_identity_sha256(),
            authority.component_set_identity_sha256()
        );

        let request = image_ref2va_request();
        let bindings = ref2va_bindings(&request);
        let output = runtime
            .generate(
                &request,
                &bindings,
                &ProgressReporter::default(),
                &mut pipeline::NoopH3PipelineObserver,
            )
            .unwrap();
        assert!(!output.mp4.is_empty());
        assert!(!output.thumbnail_png.is_empty());
        assert_eq!(output.mode, Mode::ReferenceToAudioVideo);
        assert_eq!(
            output.reference_fingerprint,
            Some(ref2va_reference_fingerprint(&request).unwrap())
        );
        assert_eq!(
            *trace.lock().unwrap(),
            vec![
                "decode-reference",
                "preprocess-reference",
                "encode-text",
                "encode-visual",
                "streamed-denoise",
                "streamed-denoise",
                "streamed-denoise",
                "decode-video",
                "decode-audio",
            ]
        );
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

    #[test]
    fn streamed_denoiser_executes_exact_blocks_and_reuses_only_resident_objects() {
        let loaded = Arc::new(Mutex::new(Vec::new()));
        let forwarded = Arc::new(Mutex::new(Vec::new()));
        let aborted = Arc::new(AtomicUsize::new(0));
        let identity = H3PipelineBackendIdentity {
            kind: super::super::pipeline::H3PipelineBackendKind::SyntheticCpu,
            device_id: "synthetic-cpu-0".into(),
            execution_fingerprint: EXECUTION.into(),
        };
        let plan = FrozenH3BlockStreamingPlan::new("synthetic-cpu-0", EXECUTION, 2, 1).unwrap();
        let mut denoiser = H3BlockStreamedDenoiser::new(
            identity,
            plan,
            SyntheticLease,
            SyntheticBlockLoader {
                loaded: Arc::clone(&loaded),
            },
            SyntheticExecutor {
                forwarded: Arc::clone(&forwarded),
                input: None,
                aborted: Arc::clone(&aborted),
            },
        )
        .unwrap();
        let layout = frozen_layout();
        let mut checkpoint = RecordingCheckpoint::default();

        for _ in 0..2 {
            let (video, audio, text, timesteps) = forward_inputs();
            let output = denoiser
                .denoise(
                    H3ForwardInput {
                        video_rows: &video,
                        audio_rows: &audio,
                        text_states: &text,
                        timesteps: &timesteps,
                    },
                    &layout,
                    &mut checkpoint,
                )
                .unwrap();
            assert_eq!(output.video.dims(), video.dims());
            assert_eq!(output.audio.dims(), audio.dims());
        }

        assert_eq!(
            *forwarded.lock().unwrap(),
            (0..50).chain(0..50).collect::<Vec<_>>()
        );
        let loaded = loaded.lock().unwrap();
        assert_eq!(loaded.iter().filter(|&&index| index < 2).count(), 2);
        assert_eq!(loaded.iter().filter(|&&index| index >= 2).count(), 48 * 2);
        drop(loaded);
        assert_eq!(aborted.load(Ordering::SeqCst), 0);
        assert!(checkpoint.events.iter().all(|event| {
            event.phase == H3PipelinePhase::TransformerBlock && event.total == 50
        }));

        let forwarded_before_cancel = forwarded.lock().unwrap().len();
        checkpoint.cancel_after_block = Some(3);
        let (video, audio, text, timesteps) = forward_inputs();
        let error = denoiser
            .denoise(
                H3ForwardInput {
                    video_rows: &video,
                    audio_rows: &audio,
                    text_states: &text,
                    timesteps: &timesteps,
                },
                &layout,
                &mut checkpoint,
            )
            .unwrap_err();
        assert!(is_inference_cancelled(&error));
        assert_eq!(aborted.load(Ordering::SeqCst), 1);
        assert_eq!(
            &forwarded.lock().unwrap()[forwarded_before_cancel..],
            &[0, 1]
        );

        checkpoint.cancel_after_block = None;
        let (video, audio, text, timesteps) = forward_inputs();
        denoiser
            .denoise(
                H3ForwardInput {
                    video_rows: &video,
                    audio_rows: &audio,
                    text_states: &text,
                    timesteps: &timesteps,
                },
                &layout,
                &mut checkpoint,
            )
            .expect("the streamed state must remain reusable after cooperative cancellation");
        assert_eq!(aborted.load(Ordering::SeqCst), 1);
        assert_eq!(
            &forwarded.lock().unwrap()[forwarded_before_cancel + 2..],
            &(0..50).collect::<Vec<_>>()
        );
    }
}
