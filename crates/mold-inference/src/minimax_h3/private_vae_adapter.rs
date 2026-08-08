//! Authorization-bound VAE adapter for the private MiniMax H3 CUDA runtime.
//!
//! This module is compiled only by the developer-only `h3-private-uat`
//! feature. It decorates an already-frozen, explicitly VAE-free FL2VA
//! component backend with the opened-file-bound Comfy visual and audio VAE
//! bundle. It has no registry, factory, capability, catalog, download, server,
//! or CLI integration.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use mold_candle::minimax_h3::{
    AudioSoundtrackAssociation, AudioVaePhase, DecodeSink, H3ForwardInput, H3FrozenPackedLayout,
    H3TransformerOutput, StereoLatents, StereoWaveform, Uint8RgbPixels, VisualVaeEvent,
    VisualVaeObserver,
};
use mold_core::minimax_h3::{self as contract, Task};

use crate::h3_factory::H3PrivateVaeFactoryAuthority;
use crate::FrozenH3FactoryAuthority;

use super::engine::H3StreamedFl2VaBackend;
use super::pipeline::{
    H3Fl2VaBackend, H3PipelineBackendIdentity, H3PipelineBackendKind, H3PipelineCheckpoint,
    H3PipelineEvent, H3PipelinePhase, H3PreparedEndpoint, H3TextConditioning, H3VideoEncodeSink,
};
use super::private_fl2va_runtime::H3PrivateVaeFreeInnerAuthority;
use super::vae_runtime::{H3ComfyVaeRuntimeBundle, H3ComfyVaeRuntimeMemory};

pub(crate) trait H3PrivateVaeRuntime: Send + Sync {
    fn task(&self) -> Task;
    fn canonical_model(&self) -> &str;
    fn plan_identity_sha256(&self) -> &str;
    fn artifact_plan_identity_sha256(&self) -> &str;
    fn authority_identity_sha256(&self) -> &str;
    fn device(&self) -> &Device;
    fn validate_authority(&self) -> Result<()>;

    fn encode_visual_condition(
        &self,
        endpoint: &H3PreparedEndpoint,
        mode: mold_candle::minimax_h3::ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor>;

    fn decode_video(
        &self,
        latents: &Tensor,
        sink: &mut H3VideoEncodeSink,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()>;

    fn decode_audio(
        &self,
        latents: &StereoLatents,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform>;
}

impl H3PrivateVaeRuntime for H3ComfyVaeRuntimeBundle {
    fn task(&self) -> Task {
        self.task()
    }

    fn canonical_model(&self) -> &str {
        self.canonical_model()
    }

    fn plan_identity_sha256(&self) -> &str {
        self.plan_identity_sha256()
    }

    fn artifact_plan_identity_sha256(&self) -> &str {
        self.artifact_plan_identity_sha256()
    }

    fn authority_identity_sha256(&self) -> &str {
        self.authority_identity_sha256()
    }

    fn device(&self) -> &Device {
        self.device()
    }

    fn validate_authority(&self) -> Result<()> {
        Ok(self.validate_authority()?)
    }

    fn encode_visual_condition(
        &self,
        endpoint: &H3PreparedEndpoint,
        mode: mold_candle::minimax_h3::ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor> {
        let pixels = Uint8RgbPixels::new(endpoint.pixels.to_device(self.device())?)?;
        let error_bridge = H3CallbackErrorBridge::default();
        let mut observer = H3PrivateVisualObserver {
            checkpoint,
            phase: H3PipelinePhase::VisualConditionEncodeChunk,
            error_bridge: error_bridge.clone(),
        };
        let result = self
            .visual_vae
            .encode_condition_u8(pixels, mode, &mut observer);
        error_bridge.finish(result)
    }

    fn decode_video(
        &self,
        latents: &Tensor,
        sink: &mut H3VideoEncodeSink,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        let checkpoint = Rc::new(RefCell::new(checkpoint));
        let error_bridge = H3CallbackErrorBridge::default();
        let mut decode_sink = H3PrivatePipelineDecodeSink {
            sink,
            checkpoint: Rc::clone(&checkpoint),
            next_frame: 0,
            error_bridge: error_bridge.clone(),
        };
        let mut observer = H3PrivateSharedVisualObserver {
            checkpoint,
            phase: H3PipelinePhase::VisualDecodeChunk,
            error_bridge: error_bridge.clone(),
        };
        let result =
            self.visual_vae
                .decode_normalized_to_sink(latents, &mut decode_sink, &mut observer);
        error_bridge.finish(result)
    }

    fn decode_audio(
        &self,
        latents: &StereoLatents,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform> {
        let error_bridge = H3CallbackErrorBridge::default();
        let mut phase_checkpoint =
            |phase| checkpoint_audio_decode_phase(checkpoint, phase, &error_bridge);
        let result = self.audio_vae.decode_with_phase_checkpoint(
            latents,
            AudioSoundtrackAssociation::Generated,
            &mut phase_checkpoint,
        );
        let waveform = error_bridge.finish(result)?;
        checkpoint.checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::AudioDecodeChunk,
            completed: 3,
            total: 3,
        })?;
        Ok(waveform)
    }
}

/// VAE-only decorator for the existing FL2VA/T2VA component contract.
///
/// Field order is load-bearing. The visual/audio models and their source-file
/// lease drop before the delegated backend, which may own the CUDA execution
/// lease. Inside the runtime bundle, both models drop before the opened source
/// descriptors.
pub(crate) struct H3PrivateComfyVaeAdapter<B, R = H3ComfyVaeRuntimeBundle> {
    vae: R,
    inner: B,
    frozen_identity: H3PipelineBackendIdentity,
    admitted: H3PrivateVaeFactoryAuthority,
    plan_identity_sha256: String,
    authority_identity_sha256: String,
}

impl<B> H3PrivateComfyVaeAdapter<B>
where
    B: H3PrivateVaeFreeInnerAuthority,
{
    pub(crate) fn new(
        inner: B,
        vae: H3ComfyVaeRuntimeBundle,
        authority: &FrozenH3FactoryAuthority,
    ) -> Result<Self> {
        Self::new_with_runtime(inner, vae, authority)
    }

    pub(crate) fn plan_identity_sha256(&self) -> &str {
        &self.plan_identity_sha256
    }

    pub(crate) fn authority_identity_sha256(&self) -> &str {
        &self.authority_identity_sha256
    }

    pub(crate) fn factory_identity_sha256(&self) -> &str {
        &self.admitted.factory_identity_sha256
    }

    pub(crate) fn backend_plan_identity_sha256(&self) -> &str {
        &self.admitted.backend_plan_identity_sha256
    }

    pub(crate) fn component_set_identity_sha256(&self) -> &str {
        &self.admitted.component_set_identity_sha256
    }

    pub(crate) fn memory(&self) -> &H3ComfyVaeRuntimeMemory {
        self.vae.memory()
    }
}

impl<B, R> H3PrivateComfyVaeAdapter<B, R>
where
    B: H3PrivateVaeFreeInnerAuthority,
    R: H3PrivateVaeRuntime,
{
    pub(crate) fn new_with_runtime(
        inner: B,
        vae: R,
        authority: &FrozenH3FactoryAuthority,
    ) -> Result<Self> {
        let frozen_identity = inner.identity();
        let admitted = authority.private_vae_adapter_authority()?;
        validate_initial_authority(&inner, &vae, &frozen_identity, &admitted)?;
        Ok(Self {
            plan_identity_sha256: vae.plan_identity_sha256().into(),
            authority_identity_sha256: vae.authority_identity_sha256().into(),
            vae,
            inner,
            frozen_identity,
            admitted,
        })
    }

    fn validate_authority(&self) -> Result<()> {
        self.vae.validate_authority()?;
        let inner_authority = self.inner.validate_private_vae_free_inner_authority()?;
        if self.inner.identity() != self.frozen_identity
            || !self.inner.device().same_device(self.vae.device())
            || self.vae.task() != Task::Fl2va
            || self.vae.canonical_model() != contract::FL2VA_COMFY
            || self.vae.artifact_plan_identity_sha256()
                != self.admitted.vae_artifact_plan_identity_sha256
            || !inner_authority.matches(&self.admitted)
            || self.vae.plan_identity_sha256() != self.plan_identity_sha256
            || self.vae.authority_identity_sha256() != self.authority_identity_sha256
        {
            bail!("private MiniMax H3 VAE authority changed after composition");
        }
        Ok(())
    }

    fn checked<T>(&self, operation: impl FnOnce(&R) -> Result<T>) -> Result<T> {
        self.validate_authority()?;
        let result = operation(&self.vae);
        self.validate_authority()?;
        result
    }
}

impl<B, R> H3Fl2VaBackend for H3PrivateComfyVaeAdapter<B, R>
where
    B: H3PrivateVaeFreeInnerAuthority,
    R: H3PrivateVaeRuntime,
{
    fn identity(&self) -> H3PipelineBackendIdentity {
        self.frozen_identity.clone()
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
        self.validate_authority()?;
        let result = self.inner.encode_text(prompt, endpoints, checkpoint);
        self.validate_authority()?;
        result
    }

    fn encode_visual_condition(
        &mut self,
        endpoint: &H3PreparedEndpoint,
        mode: mold_candle::minimax_h3::ConditionEncodeMode,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<Tensor> {
        self.checked(|vae| vae.encode_visual_condition(endpoint, mode, checkpoint))
    }

    fn denoise(
        &mut self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3TransformerOutput> {
        self.validate_authority()?;
        let result = self.inner.denoise(input, layout, checkpoint);
        self.validate_authority()?;
        result
    }

    fn decode_video(
        &mut self,
        latents: &Tensor,
        sink: &mut H3VideoEncodeSink,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<()> {
        self.checked(|vae| vae.decode_video(latents, sink, checkpoint))
    }

    fn decode_audio(
        &mut self,
        latents: &StereoLatents,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<StereoWaveform> {
        self.checked(|vae| vae.decode_audio(latents, checkpoint))
    }
}

impl<B, R> H3StreamedFl2VaBackend for H3PrivateComfyVaeAdapter<B, R>
where
    B: H3PrivateVaeFreeInnerAuthority + Send + Sync,
    R: H3PrivateVaeRuntime,
{
    fn component_set_identity_sha256(&self) -> &str {
        &self.admitted.component_set_identity_sha256
    }
}

fn validate_initial_authority<B, R>(
    inner: &B,
    vae: &R,
    identity: &H3PipelineBackendIdentity,
    admitted: &H3PrivateVaeFactoryAuthority,
) -> Result<()>
where
    B: H3PrivateVaeFreeInnerAuthority,
    R: H3PrivateVaeRuntime,
{
    vae.validate_authority()?;
    let inner_authority = inner.validate_private_vae_free_inner_authority()?;
    let supported_route = matches!(identity.kind, H3PipelineBackendKind::Cuda)
        || cfg!(test) && matches!(identity.kind, H3PipelineBackendKind::SyntheticCpu);
    if !supported_route
        || identity.device_id.trim().is_empty()
        || identity.execution_fingerprint.trim().is_empty()
        || identity.device_id != admitted.device_id
        || identity.execution_fingerprint != admitted.execution_fingerprint
        || !inner.device().same_device(vae.device())
        || admitted.task != Task::Fl2va
        || admitted.canonical_model != contract::FL2VA_COMFY
        || vae.task() != admitted.task
        || vae.canonical_model() != admitted.canonical_model
        || vae.artifact_plan_identity_sha256() != admitted.vae_artifact_plan_identity_sha256
        || !inner_authority.matches(admitted)
        || !valid_sha256(vae.plan_identity_sha256())
        || !valid_sha256(vae.authority_identity_sha256())
    {
        bail!("private MiniMax H3 VAE runtime does not match the frozen FL2VA route");
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn audio_decode_phase_progress(phase: AudioVaePhase) -> candle_core::Result<usize> {
    match phase {
        AudioVaePhase::DecodeProjection => Ok(0),
        AudioVaePhase::DecodeBigVgan => Ok(1),
        AudioVaePhase::DecodeOutput => Ok(2),
        _ => candle_core::bail!("MiniMax H3 audio decoder reported an encode-only phase"),
    }
}

fn checkpoint_audio_decode_phase(
    checkpoint: &mut dyn H3PipelineCheckpoint,
    phase: AudioVaePhase,
    error_bridge: &H3CallbackErrorBridge,
) -> candle_core::Result<()> {
    let completed = audio_decode_phase_progress(phase)?;
    checkpoint
        .checkpoint(H3PipelineEvent {
            phase: H3PipelinePhase::AudioDecodeChunk,
            completed,
            total: 3,
        })
        .map_err(|error| error_bridge.capture(error))
}

/// Candle callbacks can return only `candle_core::Error`. Keep the original
/// `anyhow::Error` beside that transport error so `InferenceCancelled` and any
/// future typed pipeline error survive the model boundary unchanged.
#[derive(Clone, Default)]
struct H3CallbackErrorBridge {
    original: Rc<RefCell<Option<anyhow::Error>>>,
}

impl H3CallbackErrorBridge {
    fn capture(&self, error: anyhow::Error) -> candle_core::Error {
        let message = error.to_string();
        let mut original = self.original.borrow_mut();
        if original.is_none() {
            *original = Some(error);
        }
        candle_core::Error::Msg(message)
    }

    fn finish<T>(&self, result: candle_core::Result<T>) -> Result<T> {
        let original = self.original.borrow_mut().take();
        match (result, original) {
            (Ok(value), None) => Ok(value),
            (Ok(_), Some(error)) | (Err(_), Some(error)) => Err(error),
            (Err(error), None) => Err(error.into()),
        }
    }
}

struct H3PrivateVisualObserver<'a> {
    checkpoint: &'a mut dyn H3PipelineCheckpoint,
    phase: H3PipelinePhase,
    error_bridge: H3CallbackErrorBridge,
}

impl VisualVaeObserver for H3PrivateVisualObserver<'_> {
    fn checkpoint(&mut self, event: VisualVaeEvent) -> candle_core::Result<()> {
        self.checkpoint
            .checkpoint(H3PipelineEvent {
                phase: self.phase,
                completed: event.completed,
                total: event.total.max(1),
            })
            .map_err(|error| self.error_bridge.capture(error))
    }
}

struct H3PrivatePipelineDecodeSink<'sink, 'checkpoint> {
    sink: &'sink mut H3VideoEncodeSink,
    checkpoint: Rc<RefCell<&'checkpoint mut dyn H3PipelineCheckpoint>>,
    next_frame: usize,
    error_bridge: H3CallbackErrorBridge,
}

impl DecodeSink for H3PrivatePipelineDecodeSink<'_, '_> {
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
                .map_err(|error| self.error_bridge.capture(error))?;
            self.next_frame += 1;
        }
        Ok(())
    }
}

struct H3PrivateSharedVisualObserver<'a> {
    checkpoint: Rc<RefCell<&'a mut dyn H3PipelineCheckpoint>>,
    phase: H3PipelinePhase,
    error_bridge: H3CallbackErrorBridge,
}

impl VisualVaeObserver for H3PrivateSharedVisualObserver<'_> {
    fn checkpoint(&mut self, event: VisualVaeEvent) -> candle_core::Result<()> {
        self.checkpoint
            .borrow_mut()
            .checkpoint(H3PipelineEvent {
                phase: self.phase,
                completed: event.completed,
                total: event.total.max(1),
            })
            .map_err(|error| self.error_bridge.capture(error))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use anyhow::anyhow;
    use mold_candle::minimax_h3::{
        AudioVaeConfig, ConditionEncodeMode, H3Modality, H3PackedLayout, LATENT_CHANNELS,
    };

    use super::*;
    use crate::attention::{AttentionBackend, AttentionChunkPolicy};
    use crate::minimax_h3::backend::{
        H3BackendArtifactLease, H3BackendExecutionLease, H3CandleBackend,
    };
    use crate::minimax_h3::private_fl2va_runtime::{vae_free_inner_seal, H3PrivateInnerAuthority};
    use crate::minimax_h3::H3ConditionerLease;
    use crate::progress::{is_inference_cancelled, InferenceCancelled};
    use crate::{
        H3FactoryAuthorityInput, H3FactoryComponentAuthority, H3FactoryComponentRole,
        H3FactoryConditionerPlacement, H3FactoryQuantizationAuthority,
    };

    const PLAN: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const AUTHORITY: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const EXECUTION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    // Compile-time negative assertion: if the eager backend ever implements
    // the VAE-free marker, inference of `_` becomes ambiguous and this test
    // module stops compiling. The private seal prevents accidental opt-in;
    // this assertion also guards an intentional attempt to open that seal.
    #[allow(dead_code)]
    fn assert_eager_h3_candle_backend_is_not_vae_free<C, E, A>()
    where
        C: H3ConditionerLease,
        E: H3BackendExecutionLease,
        A: H3BackendArtifactLease,
    {
        trait AmbiguousIfVaeFree<Marker> {
            fn assert_not_implemented() {}
        }
        impl<T: ?Sized> AmbiguousIfVaeFree<()> for T {}
        struct ImplementsVaeFree;
        impl<T: ?Sized + H3PrivateVaeFreeInnerAuthority> AmbiguousIfVaeFree<ImplementsVaeFree> for T {}

        <H3CandleBackend<'static, C, E, A> as AmbiguousIfVaeFree<_>>::assert_not_implemented();
    }

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn authority_with_qwen_parameter_bytes(qwen_parameter_bytes: u64) -> FrozenH3FactoryAuthority {
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: contract::FL2VA_COMFY.into(),
            device_id: "test-cpu".into(),
            device_ordinal: 0,
            compute_capability: (8, 9),
            execution_fingerprint: EXECUTION.into(),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes,
            qwen_host_resident_parameter_bytes: 2_048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1_024,
            qwen_output_text_rows: 1,
            qwen_vision_rows: 0,
            condition_visual_rows: 0,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: "synthetic-lossless-packed-attention-v1".into(),
            attention_qualification_sha256: sha('b'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            block_offload: true,
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
        authority_with_qwen_parameter_bytes(2_048)
    }

    #[derive(Default)]
    struct RecordingCheckpoint {
        events: Vec<H3PipelineEvent>,
        fail_at: Option<usize>,
    }

    impl H3PipelineCheckpoint for RecordingCheckpoint {
        fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()> {
            self.events.push(event);
            if self.fail_at == Some(self.events.len()) {
                bail!("cancelled by synthetic checkpoint");
            }
            Ok(())
        }
    }

    struct CancelCheckpoint;

    impl H3PipelineCheckpoint for CancelCheckpoint {
        fn checkpoint(&mut self, _event: H3PipelineEvent) -> Result<()> {
            Err(InferenceCancelled.into())
        }
    }

    struct FakeBackend {
        device: Device,
        identity: H3PipelineBackendIdentity,
        inner_authority: H3PrivateInnerAuthority,
        inner_authority_after_text: Option<H3PrivateInnerAuthority>,
        execution_active: Arc<AtomicBool>,
        artifact_active: Arc<AtomicBool>,
        delegated_text: Arc<Mutex<usize>>,
        denoise_calls: Arc<AtomicUsize>,
        drops: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Drop for FakeBackend {
        fn drop(&mut self) {
            self.drops.lock().unwrap().push("backend");
        }
    }

    impl vae_free_inner_seal::Sealed for FakeBackend {}

    // SAFETY: this synthetic backend owns no visual or audio VAE model. Its
    // only artifact/execution state is represented by the two active flags,
    // and it returns the exact frozen factory, backend-plan, and component-set
    // identities after revalidating both flags on every authority query.
    unsafe impl H3PrivateVaeFreeInnerAuthority for FakeBackend {
        fn validate_private_vae_free_inner_authority(&self) -> Result<H3PrivateInnerAuthority> {
            if !self.execution_active.load(Ordering::SeqCst) {
                bail!("synthetic execution lease revoked");
            }
            if !self.artifact_active.load(Ordering::SeqCst) {
                bail!("synthetic artifact lease revoked");
            }
            Ok(self.inner_authority.clone())
        }
    }

    impl H3Fl2VaBackend for FakeBackend {
        fn identity(&self) -> H3PipelineBackendIdentity {
            self.identity.clone()
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn encode_text(
            &mut self,
            _prompt: &str,
            _endpoints: &[H3PreparedEndpoint],
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TextConditioning> {
            *self.delegated_text.lock().unwrap() += 1;
            if let Some(authority) = self.inner_authority_after_text.take() {
                self.inner_authority = authority;
            }
            Err(anyhow!("synthetic delegated text stop"))
        }

        fn encode_visual_condition(
            &mut self,
            _endpoint: &H3PreparedEndpoint,
            _mode: ConditionEncodeMode,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<Tensor> {
            bail!("inner VAE must not be used")
        }

        fn denoise(
            &mut self,
            input: H3ForwardInput<'_>,
            _layout: &H3FrozenPackedLayout,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<H3TransformerOutput> {
            self.denoise_calls.fetch_add(1, Ordering::SeqCst);
            Ok(H3TransformerOutput {
                video: input.video_rows.clone(),
                audio: input.audio_rows.clone(),
            })
        }

        fn decode_video(
            &mut self,
            _latents: &Tensor,
            _sink: &mut H3VideoEncodeSink,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            bail!("inner VAE must not be used")
        }

        fn decode_audio(
            &mut self,
            _latents: &StereoLatents,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<StereoWaveform> {
            bail!("inner VAE must not be used")
        }
    }

    struct FakeVaeRuntime {
        device: Device,
        task: Task,
        model: &'static str,
        plan: &'static str,
        artifact_plan: String,
        authority: &'static str,
        active: Arc<AtomicBool>,
        visual_calls: Arc<Mutex<usize>>,
        audio_calls: Arc<AtomicUsize>,
        drops: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Drop for FakeVaeRuntime {
        fn drop(&mut self) {
            self.drops.lock().unwrap().push("vae");
        }
    }

    impl H3PrivateVaeRuntime for FakeVaeRuntime {
        fn task(&self) -> Task {
            self.task
        }

        fn canonical_model(&self) -> &str {
            self.model
        }

        fn plan_identity_sha256(&self) -> &str {
            self.plan
        }

        fn artifact_plan_identity_sha256(&self) -> &str {
            &self.artifact_plan
        }

        fn authority_identity_sha256(&self) -> &str {
            self.authority
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn validate_authority(&self) -> Result<()> {
            if self.active.load(Ordering::SeqCst) {
                Ok(())
            } else {
                bail!("synthetic VAE lease is inactive")
            }
        }

        fn encode_visual_condition(
            &self,
            _endpoint: &H3PreparedEndpoint,
            _mode: ConditionEncodeMode,
            checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<Tensor> {
            *self.visual_calls.lock().unwrap() += 1;
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::VisualConditionEncodeChunk,
                completed: 1,
                total: 1,
            })?;
            Ok(Tensor::zeros((1, 16, 1, 1, 1), DType::F32, &self.device)?)
        }

        fn decode_video(
            &self,
            _latents: &Tensor,
            _sink: &mut H3VideoEncodeSink,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<()> {
            Ok(())
        }

        fn decode_audio(
            &self,
            _latents: &StereoLatents,
            _checkpoint: &mut dyn H3PipelineCheckpoint,
        ) -> Result<StereoWaveform> {
            self.audio_calls.fetch_add(1, Ordering::SeqCst);
            Ok(StereoWaveform::new(
                Tensor::zeros((1, 2, 1), DType::F32, &self.device)?,
                32_000,
                AudioSoundtrackAssociation::Generated,
            )?)
        }
    }

    fn inner_authority(authority: &FrozenH3FactoryAuthority) -> H3PrivateInnerAuthority {
        let admitted = authority.private_vae_adapter_authority().unwrap();
        H3PrivateInnerAuthority::new(
            admitted.factory_identity_sha256,
            admitted.backend_plan_identity_sha256,
            admitted.component_set_identity_sha256,
        )
        .unwrap()
    }

    fn backend(
        drops: Arc<Mutex<Vec<&'static str>>>,
        authority: &FrozenH3FactoryAuthority,
    ) -> FakeBackend {
        FakeBackend {
            device: Device::Cpu,
            identity: H3PipelineBackendIdentity {
                kind: H3PipelineBackendKind::SyntheticCpu,
                device_id: "test-cpu".into(),
                execution_fingerprint: EXECUTION.into(),
            },
            inner_authority: inner_authority(authority),
            inner_authority_after_text: None,
            execution_active: Arc::new(AtomicBool::new(true)),
            artifact_active: Arc::new(AtomicBool::new(true)),
            delegated_text: Arc::new(Mutex::new(0)),
            denoise_calls: Arc::new(AtomicUsize::new(0)),
            drops,
        }
    }

    fn runtime(
        drops: Arc<Mutex<Vec<&'static str>>>,
        authority: &FrozenH3FactoryAuthority,
    ) -> FakeVaeRuntime {
        FakeVaeRuntime {
            device: Device::Cpu,
            task: Task::Fl2va,
            model: contract::FL2VA_COMFY,
            plan: PLAN,
            artifact_plan: authority
                .private_vae_adapter_authority()
                .unwrap()
                .vae_artifact_plan_identity_sha256,
            authority: AUTHORITY,
            active: Arc::new(AtomicBool::new(true)),
            visual_calls: Arc::new(Mutex::new(0)),
            audio_calls: Arc::new(AtomicUsize::new(0)),
            drops,
        }
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
    fn exact_private_vae_decorates_only_vae_operations() {
        let authority = authority();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let delegated_text = Arc::new(Mutex::new(0));
        let visual_calls = Arc::new(Mutex::new(0));
        let mut inner = backend(Arc::clone(&drops), &authority);
        inner.delegated_text = Arc::clone(&delegated_text);
        let mut vae = runtime(Arc::clone(&drops), &authority);
        vae.visual_calls = Arc::clone(&visual_calls);
        let mut adapter =
            H3PrivateComfyVaeAdapter::new_with_runtime(inner, vae, &authority).unwrap();
        let mut checkpoint = RecordingCheckpoint::default();

        let text_error = adapter
            .encode_text("prompt", &[], &mut checkpoint)
            .unwrap_err();
        assert!(text_error.to_string().contains("delegated text"));
        assert_eq!(*delegated_text.lock().unwrap(), 1);

        let endpoint = H3PreparedEndpoint {
            anchor: super::super::pipeline::H3EndpointAnchor::First,
            source_width: 1,
            source_height: 1,
            resize: super::super::pipeline::H3EndpointResize::Identity,
            pixels: Tensor::zeros((1, 3, 1, 1, 1), DType::U8, &Device::Cpu).unwrap(),
        };
        adapter
            .encode_visual_condition(
                &endpoint,
                ConditionEncodeMode::OfficialFreshSeed42,
                &mut checkpoint,
            )
            .unwrap();
        assert_eq!(*visual_calls.lock().unwrap(), 1);
        assert_eq!(
            checkpoint.events.last().unwrap().phase,
            H3PipelinePhase::VisualConditionEncodeChunk
        );
    }

    #[test]
    fn mixed_task_artifact_or_inactive_lease_is_rejected() {
        let authority = authority();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let mut wrong_task = runtime(Arc::clone(&drops), &authority);
        wrong_task.task = Task::Ref2va;
        assert!(H3PrivateComfyVaeAdapter::new_with_runtime(
            backend(Arc::clone(&drops), &authority),
            wrong_task,
            &authority,
        )
        .err()
        .unwrap()
        .to_string()
        .contains("frozen FL2VA route"));

        let mut wrong_artifact = runtime(Arc::clone(&drops), &authority);
        wrong_artifact.plan = "not-a-digest";
        assert!(H3PrivateComfyVaeAdapter::new_with_runtime(
            backend(Arc::clone(&drops), &authority),
            wrong_artifact,
            &authority,
        )
        .err()
        .unwrap()
        .to_string()
        .contains("frozen FL2VA route"));

        let inactive = runtime(Arc::clone(&drops), &authority);
        inactive.active.store(false, Ordering::SeqCst);
        assert!(H3PrivateComfyVaeAdapter::new_with_runtime(
            backend(Arc::clone(&drops), &authority),
            inactive,
            &authority,
        )
        .err()
        .unwrap()
        .to_string()
        .contains("inactive"));
    }

    #[test]
    fn cross_plan_and_cross_component_composition_fail_before_work() {
        let authority = authority();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let visual_calls = Arc::new(Mutex::new(0));

        let mut wrong_plan = runtime(Arc::clone(&drops), &authority);
        wrong_plan.artifact_plan = sha('f');
        wrong_plan.visual_calls = Arc::clone(&visual_calls);
        let error = H3PrivateComfyVaeAdapter::new_with_runtime(
            backend(Arc::clone(&drops), &authority),
            wrong_plan,
            &authority,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("frozen FL2VA route"));
        assert_eq!(*visual_calls.lock().unwrap(), 0);

        let mut wrong_components = backend(Arc::clone(&drops), &authority);
        wrong_components
            .inner_authority
            .component_set_identity_sha256 = sha('f');
        let error = H3PrivateComfyVaeAdapter::new_with_runtime(
            wrong_components,
            runtime(Arc::clone(&drops), &authority),
            &authority,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("frozen FL2VA route"));
        assert_eq!(*visual_calls.lock().unwrap(), 0);
    }

    #[test]
    fn outer_factory_admission_must_match_initial_before_and_after_work() {
        let authority = authority();
        let changed_outer_authority = authority_with_qwen_parameter_bytes(2_049);
        let exact = authority.private_vae_adapter_authority().unwrap();
        let changed = changed_outer_authority
            .private_vae_adapter_authority()
            .unwrap();
        assert_ne!(
            exact.factory_identity_sha256,
            changed.factory_identity_sha256
        );
        assert_eq!(
            exact.backend_plan_identity_sha256,
            changed.backend_plan_identity_sha256
        );
        assert_eq!(
            exact.component_set_identity_sha256,
            changed.component_set_identity_sha256
        );
        assert_eq!(exact.device_id, changed.device_id);
        assert_eq!(exact.execution_fingerprint, changed.execution_fingerprint);

        let drops = Arc::new(Mutex::new(Vec::new()));
        let error = H3PrivateComfyVaeAdapter::new_with_runtime(
            backend(Arc::clone(&drops), &authority),
            runtime(Arc::clone(&drops), &changed_outer_authority),
            &changed_outer_authority,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("frozen FL2VA route"));

        let audio_calls = Arc::new(AtomicUsize::new(0));
        let mut vae = runtime(Arc::clone(&drops), &authority);
        vae.audio_calls = Arc::clone(&audio_calls);
        let mut adapter = H3PrivateComfyVaeAdapter::new_with_runtime(
            backend(Arc::clone(&drops), &authority),
            vae,
            &authority,
        )
        .unwrap();
        adapter.inner.inner_authority.factory_identity_sha256 =
            changed.factory_identity_sha256.clone();
        let latents = StereoLatents::new(
            Tensor::zeros((1, LATENT_CHANNELS, 2, 1), DType::F32, &Device::Cpu).unwrap(),
            &AudioVaeConfig::default(),
        )
        .unwrap();
        let error = adapter
            .decode_audio(&latents, &mut RecordingCheckpoint::default())
            .unwrap_err();
        assert!(error.to_string().contains("authority changed"));
        assert_eq!(audio_calls.load(Ordering::SeqCst), 0);

        let delegated_text = Arc::new(Mutex::new(0));
        let mut inner = backend(Arc::clone(&drops), &authority);
        inner.delegated_text = Arc::clone(&delegated_text);
        inner.inner_authority_after_text = Some(inner_authority(&changed_outer_authority));
        let mut adapter = H3PrivateComfyVaeAdapter::new_with_runtime(
            inner,
            runtime(Arc::clone(&drops), &authority),
            &authority,
        )
        .unwrap();
        let error = adapter
            .encode_text("prompt", &[], &mut RecordingCheckpoint::default())
            .unwrap_err();
        assert!(error.to_string().contains("authority changed"));
        assert_eq!(*delegated_text.lock().unwrap(), 1);
    }

    #[test]
    fn revoked_inner_lease_after_denoise_blocks_vae_decode() {
        let authority = authority();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let artifact_active = Arc::new(AtomicBool::new(true));
        let denoise_calls = Arc::new(AtomicUsize::new(0));
        let audio_calls = Arc::new(AtomicUsize::new(0));
        let mut inner = backend(Arc::clone(&drops), &authority);
        inner.artifact_active = Arc::clone(&artifact_active);
        inner.denoise_calls = Arc::clone(&denoise_calls);
        let mut vae = runtime(Arc::clone(&drops), &authority);
        vae.audio_calls = Arc::clone(&audio_calls);
        let mut adapter =
            H3PrivateComfyVaeAdapter::new_with_runtime(inner, vae, &authority).unwrap();

        let (video, audio, text, timesteps) = forward_inputs();
        adapter
            .denoise(
                H3ForwardInput {
                    video_rows: &video,
                    audio_rows: &audio,
                    text_states: &text,
                    timesteps: &timesteps,
                },
                &frozen_layout(),
                &mut RecordingCheckpoint::default(),
            )
            .unwrap();
        assert_eq!(denoise_calls.load(Ordering::SeqCst), 1);

        artifact_active.store(false, Ordering::SeqCst);
        let latents = StereoLatents::new(
            Tensor::zeros((1, LATENT_CHANNELS, 2, 1), DType::F32, &Device::Cpu).unwrap(),
            &AudioVaeConfig::default(),
        )
        .unwrap();
        let error = adapter
            .decode_audio(&latents, &mut RecordingCheckpoint::default())
            .unwrap_err();
        assert!(error.to_string().contains("artifact lease revoked"));
        assert_eq!(audio_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn callback_bridge_preserves_typed_visual_video_and_audio_cancellation() {
        let bridge = H3CallbackErrorBridge::default();
        let mut visual_checkpoint = CancelCheckpoint;
        let mut observer = H3PrivateVisualObserver {
            checkpoint: &mut visual_checkpoint,
            phase: H3PipelinePhase::VisualConditionEncodeChunk,
            error_bridge: bridge.clone(),
        };
        let transport = observer
            .checkpoint(VisualVaeEvent {
                phase: mold_candle::minimax_h3::VisualVaePhase::EncodeChunk,
                completed: 0,
                total: 1,
            })
            .unwrap_err();
        let error = bridge.finish::<()>(Err(transport)).unwrap_err();
        assert!(is_inference_cancelled(&error));

        let bridge = H3CallbackErrorBridge::default();
        let mut video_checkpoint = CancelCheckpoint;
        let checkpoint: Rc<RefCell<&mut dyn H3PipelineCheckpoint>> =
            Rc::new(RefCell::new(&mut video_checkpoint));
        let mut sink = H3VideoEncodeSink::new_dimensions(1, 1, 1).unwrap();
        let mut decode_sink = H3PrivatePipelineDecodeSink {
            sink: &mut sink,
            checkpoint,
            next_frame: 0,
            error_bridge: bridge.clone(),
        };
        let transport = decode_sink
            .write(
                0,
                &Tensor::zeros((1, 3, 1, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            )
            .unwrap_err();
        let error = bridge.finish::<()>(Err(transport)).unwrap_err();
        assert!(is_inference_cancelled(&error));

        for (phase, expected_completed) in [
            (AudioVaePhase::DecodeProjection, 0),
            (AudioVaePhase::DecodeBigVgan, 1),
            (AudioVaePhase::DecodeOutput, 2),
        ] {
            assert_eq!(
                audio_decode_phase_progress(phase).unwrap(),
                expected_completed
            );
            let bridge = H3CallbackErrorBridge::default();
            let mut checkpoint = CancelCheckpoint;
            let transport =
                checkpoint_audio_decode_phase(&mut checkpoint, phase, &bridge).unwrap_err();
            let error = bridge.finish::<()>(Err(transport)).unwrap_err();
            assert!(is_inference_cancelled(&error), "phase {phase:?}");
        }
        assert!(audio_decode_phase_progress(AudioVaePhase::EncodeDac).is_err());
    }

    #[test]
    fn vae_models_and_artifact_lease_drop_before_execution_backend() {
        let authority = authority();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let adapter = H3PrivateComfyVaeAdapter::new_with_runtime(
            backend(Arc::clone(&drops), &authority),
            runtime(Arc::clone(&drops), &authority),
            &authority,
        )
        .unwrap();
        drop(adapter);
        assert_eq!(*drops.lock().unwrap(), ["vae", "backend"]);
    }
}
