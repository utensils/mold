//! Concrete, weight-free media preprocessing for the H3 Ref2VA path.
//!
//! This module deliberately contains no checkpoint discovery and activates no
//! model family. It turns server-bound, digest-verified file descriptors into
//! the exact media tensors the admitted backend encodes — the same decoder
//! admission runs to derive its row counts, so a preparation and its reopen
//! can only agree by reproducing each other.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, bail, Context, Result};
use image::RgbImage;
use mold_candle::minimax_h3::{
    sample_video_frames, RefPresentation, RefPresentationKind, VideoPresentationBlock,
};
use mold_core::minimax_h3::{AUDIO_SAMPLE_RATE_HZ, FIXED_FPS};
use mold_core::GenerationReferenceKind;

use super::pipeline::ref2va::{
    H3DecodedAudioFacts, H3DecodedReferenceFacts, H3PreparedReference, H3ReferencePresentation,
};
use super::pipeline::{
    pillow_lanczos_resize, H3PipelineCheckpoint, H3PipelineEvent, H3PipelinePhase,
};
use crate::engine::GenerationReferenceBinding;
use crate::ltx2::DecodedAudio;
use crate::reference_media::{
    cfr_source_indices, decode_audio_from_binding, decode_image_from_binding,
    decode_video_from_binding, normalize_audio, NormalizedStereoAudio,
};

#[derive(Debug)]
enum DecodedReference {
    Image(RgbImage),
    Video {
        selected_frames: BTreeMap<usize, RgbImage>,
        source_frame_count: u32,
        fps: f64,
        audio: Option<DecodedAudio>,
    },
    Audio(DecodedAudio),
}

#[derive(Debug)]
struct DecodedSlot {
    facts: H3DecodedReferenceFacts,
    media: DecodedReference,
}

#[derive(Debug)]
pub(crate) enum H3NormalizedReference {
    Image {
        image: RgbImage,
        vision_tokens: usize,
    },
    Video {
        /// Exact `17n+5` prefix consumed by the visual VAE.
        vae_frames: Vec<RgbImage>,
        /// Exact 2 fps cursor samples consumed by Qwen3-VL.
        qwen_frames: Vec<RgbImage>,
        qwen_source_indices: Vec<usize>,
        /// Exact normalized-media midpoint authority used to build the Qwen
        /// video presentation blocks, in presentation order.
        qwen_block_timestamps_seconds: Vec<f64>,
        audio: Option<NormalizedStereoAudio>,
    },
    Audio {
        audio: NormalizedStereoAudio,
    },
}

/// Attempt-local decoded/normalized media owned by one frozen backend.
/// BTreeMap preserves the one-based request order for diagnostics and future
/// encoder iteration; inserting a duplicate or preprocessing out of sequence
/// fails closed.
#[derive(Debug, Default)]
pub(crate) struct H3ReferenceMediaAdapter {
    decoded: BTreeMap<u32, DecodedSlot>,
    normalized: BTreeMap<u32, H3NormalizedReference>,
    decode_order: Vec<u32>,
    normalized_order: Vec<u32>,
}

impl H3ReferenceMediaAdapter {
    pub(crate) fn decode_reference(
        &mut self,
        reference: &H3PreparedReference,
        binding: &GenerationReferenceBinding,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3DecodedReferenceFacts> {
        let index = reference.metadata.index;
        let expected_index = u32::try_from(self.decode_order.len() + 1)
            .context("MiniMax H3 reference order exceeded u32")?;
        if index != expected_index {
            bail!(
                "MiniMax H3 reference decode order changed: received {index}, expected {expected_index}"
            );
        }
        if self.decoded.contains_key(&index) || self.normalized.contains_key(&index) {
            bail!("MiniMax H3 reference {index} was decoded more than once");
        }
        let mut poll = || {
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::ReferenceDecodeChunk,
                completed: 0,
                total: 1,
            })
        };
        let (facts, media) = match reference.metadata.kind {
            GenerationReferenceKind::Image => {
                let image = decode_image_from_binding(binding, &mut poll)?;
                let facts = H3DecodedReferenceFacts {
                    index,
                    kind: GenerationReferenceKind::Image,
                    width: Some(image.width()),
                    height: Some(image.height()),
                    frame_count: None,
                    fps: None,
                    audio: None,
                };
                (facts, DecodedReference::Image(image))
            }
            GenerationReferenceKind::Video => {
                let width = required_dimension(reference.shape.normalized_width, "width")?;
                let height = required_dimension(reference.shape.normalized_height, "height")?;
                let source_frame_count = required_count(
                    reference.metadata.frame_count,
                    "decoded source video frames",
                )?;
                let source_fps = reference
                    .metadata
                    .fps
                    .filter(|fps| fps.is_finite() && *fps > 0.0)
                    .ok_or_else(|| anyhow!("MiniMax H3 reference lost its decoded frame rate"))?;
                let selected_source_indices = selected_video_source_indices(
                    reference,
                    source_frame_count,
                    source_fps,
                    &mut poll,
                )?;
                let mut resize =
                    |frame: RgbImage| pillow_lanczos_resize(&frame, width, height, &mut || Ok(()));
                let decoded = decode_video_from_binding(
                    binding,
                    &selected_source_indices,
                    &mut resize,
                    &mut poll,
                )?;
                decoded
                    .frames
                    .first()
                    .context("decoded H3 reference video did not contain a selected frame")?;
                let audio_facts = decoded
                    .audio
                    .as_ref()
                    .map(decoded_audio_facts)
                    .transpose()?;
                let facts = H3DecodedReferenceFacts {
                    index,
                    kind: GenerationReferenceKind::Video,
                    width: Some(decoded.source_width),
                    height: Some(decoded.source_height),
                    frame_count: Some(decoded.source_frame_count),
                    fps: Some(decoded.fps),
                    audio: audio_facts,
                };
                let selected_frames = selected_source_indices
                    .into_iter()
                    .zip(decoded.frames)
                    .collect();
                (
                    facts,
                    DecodedReference::Video {
                        selected_frames,
                        source_frame_count: decoded.source_frame_count,
                        fps: decoded.fps,
                        audio: decoded.audio,
                    },
                )
            }
            GenerationReferenceKind::Audio => {
                let decoded =
                    decode_audio_from_binding(binding, &reference.metadata.mime_type, &mut poll)?;
                let facts = H3DecodedReferenceFacts {
                    index,
                    kind: GenerationReferenceKind::Audio,
                    width: None,
                    height: None,
                    frame_count: None,
                    fps: None,
                    audio: Some(decoded_audio_facts(&decoded)?),
                };
                (facts, DecodedReference::Audio(decoded))
            }
        };
        self.decoded.insert(
            index,
            DecodedSlot {
                facts: facts.clone(),
                media,
            },
        );
        self.decode_order.push(index);
        Ok(facts)
    }

    pub(crate) fn preprocess_reference(
        &mut self,
        reference: &H3PreparedReference,
        decoded: &H3DecodedReferenceFacts,
        checkpoint: &mut dyn H3PipelineCheckpoint,
    ) -> Result<H3ReferencePresentation> {
        let index = reference.metadata.index;
        let expected_index = self
            .decode_order
            .get(self.normalized_order.len())
            .copied()
            .ok_or_else(|| {
                anyhow!("MiniMax H3 reference {index} was not decoded in this request")
            })?;
        if index != expected_index {
            bail!(
                "MiniMax H3 reference preprocess order changed: received {index}, expected {expected_index}"
            );
        }
        if self.normalized.contains_key(&index) {
            bail!("MiniMax H3 reference {index} was preprocessed more than once");
        }
        let slot = self.decoded.remove(&index).ok_or_else(|| {
            anyhow!("MiniMax H3 reference {index} was preprocessed before decode")
        })?;
        if &slot.facts != decoded {
            bail!("MiniMax H3 reference {index} decoded facts changed before preprocessing");
        }
        let mut poll = || {
            checkpoint.checkpoint(H3PipelineEvent {
                phase: H3PipelinePhase::ReferencePreprocessChunk,
                completed: 0,
                total: 1,
            })
        };
        let (normalized, presentation) = match slot.media {
            DecodedReference::Image(image) => {
                let width = required_dimension(reference.shape.normalized_width, "width")?;
                let height = required_dimension(reference.shape.normalized_height, "height")?;
                let image = pillow_lanczos_resize(&image, width, height, &mut poll)?;
                let vision_tokens = vision_tokens_per_temporal_block(width, height)?;
                (
                    H3NormalizedReference::Image {
                        image,
                        vision_tokens,
                    },
                    RefPresentation {
                        kind: RefPresentationKind::Image { vision_tokens },
                        has_audio: false,
                    },
                )
            }
            DecodedReference::Video {
                selected_frames,
                source_frame_count,
                fps,
                audio,
            } => {
                let width = required_dimension(reference.shape.normalized_width, "width")?;
                let height = required_dimension(reference.shape.normalized_height, "height")?;
                let normalized_count = required_count(
                    reference.shape.normalized_video_frames,
                    "normalized video frames",
                )?;
                let vae_count = required_count(reference.shape.video_frames, "video VAE frames")?;
                let cfr_indices = cfr_source_indices(
                    usize::try_from(source_frame_count)?,
                    fps,
                    FIXED_FPS,
                    normalized_count,
                    &mut poll,
                )?;
                if vae_count > cfr_indices.len() {
                    bail!("MiniMax H3 visual-VAE frame prefix exceeds normalized video");
                }
                let mut vae_frames = Vec::with_capacity(vae_count);
                for &source_index in &cfr_indices[..vae_count] {
                    poll()?;
                    vae_frames.push(
                        selected_frames
                            .get(&source_index)
                            .with_context(|| {
                                format!(
                                    "MiniMax H3 decoder omitted selected source frame {source_index}"
                                )
                            })?
                            .clone(),
                    );
                }

                let sampled = sample_video_frames(normalized_count, f64::from(FIXED_FPS))?;
                let expected_qwen =
                    required_count(reference.shape.qwen_video_frames, "Qwen video frames")?;
                if sampled.frame_indices.len() != expected_qwen {
                    bail!("MiniMax H3 Qwen 2 fps sampling changed the prepared frame count");
                }
                let mut qwen_frames = Vec::with_capacity(sampled.frame_indices.len());
                let mut qwen_source_indices = Vec::with_capacity(sampled.frame_indices.len());
                for &normalized_index in &sampled.frame_indices {
                    poll()?;
                    let source_index = cfr_indices[normalized_index];
                    qwen_source_indices.push(source_index);
                    if normalized_index < vae_frames.len() {
                        qwen_frames.push(vae_frames[normalized_index].clone());
                    } else {
                        qwen_frames.push(
                            selected_frames
                                .get(&source_index)
                                .with_context(|| {
                                    format!(
                                        "MiniMax H3 decoder omitted Qwen source frame {source_index}"
                                    )
                                })?
                                .clone(),
                        );
                    }
                }
                let audio = match audio {
                    Some(audio) => Some(normalize_reference_audio(reference, &audio, &mut poll)?),
                    None => None,
                };
                let vision_tokens = vision_tokens_per_temporal_block(width, height)?;
                let qwen_block_timestamps_seconds = sampled.block_timestamps_seconds;
                let blocks = qwen_block_timestamps_seconds
                    .iter()
                    .copied()
                    .map(|timestamp_seconds| VideoPresentationBlock {
                        timestamp_seconds,
                        vision_tokens,
                    })
                    .collect();
                (
                    H3NormalizedReference::Video {
                        vae_frames,
                        qwen_frames,
                        qwen_source_indices,
                        qwen_block_timestamps_seconds,
                        audio,
                    },
                    RefPresentation {
                        kind: RefPresentationKind::Video { blocks },
                        has_audio: reference.metadata.has_audio,
                    },
                )
            }
            DecodedReference::Audio(audio) => {
                let audio = normalize_reference_audio(reference, &audio, &mut poll)?;
                (
                    H3NormalizedReference::Audio { audio },
                    RefPresentation {
                        kind: RefPresentationKind::Audio,
                        has_audio: true,
                    },
                )
            }
        };
        self.normalized.insert(index, normalized);
        self.normalized_order.push(index);
        Ok(H3ReferencePresentation {
            index,
            presentation,
        })
    }

    pub(crate) fn normalized(&self, index: u32) -> Option<&H3NormalizedReference> {
        self.normalized.get(&index)
    }

    pub(crate) fn normalized_indices(&self) -> Vec<u32> {
        self.normalized_order.clone()
    }
}

fn selected_video_source_indices(
    reference: &H3PreparedReference,
    source_frame_count: usize,
    source_fps: f64,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<usize>> {
    let normalized_count = required_count(
        reference.shape.normalized_video_frames,
        "normalized video frames",
    )?;
    let vae_count = required_count(reference.shape.video_frames, "video VAE frames")?;
    let cfr_indices = cfr_source_indices(
        source_frame_count,
        source_fps,
        FIXED_FPS,
        normalized_count,
        checkpoint,
    )?;
    if vae_count > cfr_indices.len() {
        bail!("MiniMax H3 visual-VAE frame prefix exceeds normalized video");
    }
    let sampled = sample_video_frames(normalized_count, f64::from(FIXED_FPS))?;
    let expected_qwen = required_count(reference.shape.qwen_video_frames, "Qwen video frames")?;
    if sampled.frame_indices.len() != expected_qwen {
        bail!("MiniMax H3 Qwen 2 fps sampling changed the prepared frame count");
    }
    let mut selected = BTreeSet::new();
    selected.extend(cfr_indices[..vae_count].iter().copied());
    selected.extend(
        sampled
            .frame_indices
            .into_iter()
            .map(|normalized_index| cfr_indices[normalized_index]),
    );
    Ok(selected.into_iter().collect())
}

fn decoded_audio_facts(decoded: &DecodedAudio) -> Result<H3DecodedAudioFacts> {
    Ok(H3DecodedAudioFacts {
        sample_rate: u32::try_from(decoded.sample_rate)
            .context("decoded H3 reference sample rate exceeded u32")?,
        channels: u16::try_from(decoded.channel_count())
            .context("decoded H3 reference channel count exceeded u16")?,
        samples_per_channel: u64::try_from(decoded.sample_count())
            .context("decoded H3 reference sample count exceeded u64")?,
    })
}

fn normalize_reference_audio(
    reference: &H3PreparedReference,
    decoded: &DecodedAudio,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<NormalizedStereoAudio> {
    let maximum_native_samples = u64::from(reference.target_frames)
        .checked_mul(u64::try_from(decoded.sample_rate)?)
        .context("MiniMax H3 native audio truncation count overflowed")?
        / u64::from(FIXED_FPS);
    let maximum_native_samples = usize::try_from(maximum_native_samples)
        .context("MiniMax H3 native audio truncation count exceeded usize")?;
    let expected_samples = usize::try_from(
        reference
            .shape
            .audio_samples_per_channel
            .ok_or_else(|| anyhow!("MiniMax H3 audio reference lost prepared sample count"))?,
    )
    .context("MiniMax H3 prepared audio sample count exceeded usize")?;
    let normalized = normalize_audio(
        decoded,
        maximum_native_samples,
        AUDIO_SAMPLE_RATE_HZ,
        expected_samples,
        checkpoint,
    )?;
    if normalized.sample_rate != AUDIO_SAMPLE_RATE_HZ
        || normalized.samples_per_channel() != expected_samples
    {
        bail!("MiniMax H3 audio normalization changed the prepared duration");
    }
    Ok(normalized)
}

fn required_dimension(value: Option<u32>, name: &str) -> Result<u32> {
    value
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow!("MiniMax H3 reference lost normalized {name}"))
}

fn required_count(value: Option<u32>, name: &str) -> Result<usize> {
    usize::try_from(
        value
            .filter(|value| *value > 0)
            .ok_or_else(|| anyhow!("MiniMax H3 reference lost {name}"))?,
    )
    .with_context(|| format!("MiniMax H3 {name} exceeded usize"))
}

fn vision_tokens_per_temporal_block(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 || !width.is_multiple_of(32) || !height.is_multiple_of(32) {
        bail!("MiniMax H3 Qwen canvas {width}x{height} is not divisible by 32");
    }
    usize::try_from(u64::from(width / 32) * u64::from(height / 32))
        .context("MiniMax H3 Qwen vision-token count exceeded usize")
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use image::{DynamicImage, ImageFormat, Rgb};
    use mold_core::minimax_h3::GenerationReferencePreparedShape;
    use mold_core::{GenerationReferenceAuthority, GenerationReferenceMetadata};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::progress::{is_inference_cancelled, InferenceCancelled};

    #[derive(Default)]
    struct TestCheckpoint {
        polls: usize,
        cancel_at: Option<usize>,
        phases: Vec<H3PipelinePhase>,
    }

    impl H3PipelineCheckpoint for TestCheckpoint {
        fn checkpoint(&mut self, event: H3PipelineEvent) -> Result<()> {
            self.polls += 1;
            self.phases.push(event.phase);
            if self.cancel_at == Some(self.polls) {
                return Err(InferenceCancelled.into());
            }
            Ok(())
        }
    }

    fn metadata(
        index: u32,
        kind: GenerationReferenceKind,
        mime_type: &str,
    ) -> GenerationReferenceMetadata {
        GenerationReferenceMetadata {
            kind,
            index,
            name: Some(format!("reference-{index}")),
            sha256: String::new(),
            mime_type: mime_type.into(),
            width: None,
            height: None,
            frame_count: None,
            duration_ms: None,
            fps: None,
            has_audio: false,
            audio_duration_ms: None,
            audio_sample_count: None,
            audio_sample_rate: None,
            audio_channels: None,
            sample_rate: None,
            channels: None,
            sample_count: None,
            prepared_shape: None,
            crop: None,
        }
    }

    fn binding(
        mut metadata: GenerationReferenceMetadata,
        bytes: &[u8],
    ) -> GenerationReferenceBinding {
        metadata.sha256 = format!("{:x}", Sha256::digest(bytes));
        let mut staged = tempfile::NamedTempFile::new().unwrap();
        staged.write_all(bytes).unwrap();
        staged.flush().unwrap();
        GenerationReferenceBinding::from_opened_file(
            metadata,
            staged.reopen().unwrap(),
            u64::try_from(bytes.len()).unwrap(),
            None,
        )
        .unwrap()
    }

    fn prepared(
        mut metadata: GenerationReferenceMetadata,
        shape: GenerationReferencePreparedShape,
        target_frames: u32,
        bytes: &[u8],
    ) -> (H3PreparedReference, GenerationReferenceBinding) {
        metadata.sha256 = format!("{:x}", Sha256::digest(bytes));
        metadata.prepared_shape = Some(shape.clone());
        let binding = binding(metadata.clone(), bytes);
        (
            H3PreparedReference {
                metadata,
                shape,
                target_frames,
            },
            binding,
        )
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = RgbImage::from_fn(width, height, |x, y| {
            Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
        });
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    #[test]
    fn image_preprocessing_uses_its_own_2048_short_edge_canvas() {
        let bytes = png(64, 64);
        let mut metadata = metadata(1, GenerationReferenceKind::Image, "image/png");
        metadata.width = Some(64);
        metadata.height = Some(64);
        let shape = GenerationReferencePreparedShape {
            version: mold_core::minimax_h3::REFERENCE_PREPROCESS_VERSION,
            normalized_width: Some(2_048),
            normalized_height: Some(2_048),
            normalized_video_frames: None,
            video_frames: None,
            qwen_video_frames: None,
            audio_samples_per_channel: None,
            visual_rows: 4_096,
            audio_rows: 0,
        };
        let (reference, binding) = prepared(metadata, shape, 124, &bytes);
        let mut adapter = H3ReferenceMediaAdapter::default();
        let mut checkpoint = TestCheckpoint::default();
        let facts = adapter
            .decode_reference(&reference, &binding, &mut checkpoint)
            .unwrap();
        let presentation = adapter
            .preprocess_reference(&reference, &facts, &mut checkpoint)
            .unwrap();

        assert_eq!(adapter.normalized_indices(), vec![1]);
        let H3NormalizedReference::Image {
            image,
            vision_tokens,
        } = adapter.normalized(1).unwrap()
        else {
            panic!("expected normalized image")
        };
        assert_eq!(image.dimensions(), (2_048, 2_048));
        assert_eq!(*vision_tokens, 4_096);
        assert_eq!(
            presentation.presentation.kind,
            RefPresentationKind::Image {
                vision_tokens: 4_096
            }
        );
        assert!(!presentation.presentation.has_audio);
    }

    #[test]
    fn preprocessing_cancellation_is_typed_and_does_not_publish_media() {
        let bytes = png(64, 64);
        let mut metadata = metadata(1, GenerationReferenceKind::Image, "image/png");
        metadata.width = Some(64);
        metadata.height = Some(64);
        let shape = GenerationReferencePreparedShape {
            version: mold_core::minimax_h3::REFERENCE_PREPROCESS_VERSION,
            normalized_width: Some(2_048),
            normalized_height: Some(2_048),
            normalized_video_frames: None,
            video_frames: None,
            qwen_video_frames: None,
            audio_samples_per_channel: None,
            visual_rows: 4_096,
            audio_rows: 0,
        };
        let (reference, binding) = prepared(metadata, shape, 124, &bytes);
        let mut adapter = H3ReferenceMediaAdapter::default();
        let facts = adapter
            .decode_reference(&reference, &binding, &mut TestCheckpoint::default())
            .unwrap();
        let mut checkpoint = TestCheckpoint {
            cancel_at: Some(4),
            ..Default::default()
        };
        let error = adapter
            .preprocess_reference(&reference, &facts, &mut checkpoint)
            .unwrap_err();
        assert!(is_inference_cancelled(&error));
        assert!(adapter.normalized(1).is_none());
        assert_eq!(checkpoint.polls, 4);
    }

    #[test]
    fn prepared_rows_freeze_native_small_media_and_exact_audio_duration() {
        let references = vec![
            mold_core::GenerationReference::Image {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: Default::default(),
                mime_type: "image/png".into(),
                width: 64,
                height: 64,
            },
            mold_core::GenerationReference::Video {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: Default::default(),
                mime_type: "video/mp4".into(),
                width: 64,
                height: 64,
                frame_count: Some(48),
                duration_ms: 2_000,
                fps: 24.0,
                has_audio: false,
                audio_duration_ms: None,
                audio_sample_count: None,
                audio_sample_rate: None,
                audio_channels: None,
            },
            mold_core::GenerationReference::Audio {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: Default::default(),
                mime_type: "audio/wav".into(),
                duration_ms: 1_000,
                sample_rate: 16_000,
                channels: 4,
                sample_count: Some(16_000),
            },
        ];
        let shapes =
            mold_core::minimax_h3::reference_prepared_shapes_for_target(&references, 124).unwrap();
        // Never upscaled: a 64x64 source keeps its native canvas on the 32
        // grid instead of blowing up to 2048x2048 (image) / 768x768 (video).
        assert_eq!(
            (shapes[0].normalized_width, shapes[0].normalized_height),
            (Some(64), Some(64))
        );
        assert_eq!(shapes[0].visual_rows, 4);
        assert_eq!(
            (shapes[1].normalized_width, shapes[1].normalized_height),
            (Some(64), Some(64))
        );
        assert_eq!(shapes[1].normalized_video_frames, Some(48));
        assert_eq!(shapes[1].video_frames, Some(39));
        assert_eq!(shapes[1].qwen_video_frames, Some(4));
        assert_eq!(shapes[1].visual_rows, 48);
        assert_eq!(shapes[2].audio_samples_per_channel, Some(32_000));
        assert_eq!(shapes[2].audio_rows, 80);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn ordinary_media_fixtures_preserve_order_soundtrack_and_duration() {
        use crate::av_media::attach_aac_track_to_mp4_bytes;
        use crate::ltx2::media::{encode_wav_f32_interleaved, probe_decoded_mp4_audio_file};
        use crate::ltx_video::video_enc;

        let image_bytes = png(32, 32);
        let frames = (0..24)
            .map(|index| RgbImage::from_pixel(32, 32, Rgb([index as u8, 32, 96])))
            .collect::<Vec<_>>();
        let video_only = video_enc::encode_mp4(&frames, 24).unwrap();
        let soundtrack = (0..32_000)
            .flat_map(|sample| {
                let value = ((sample as f32 / 32_000.0) * std::f32::consts::TAU * 220.0).sin();
                [value * 0.1, value * 0.2]
            })
            .collect::<Vec<_>>();
        let video_bytes =
            attach_aac_track_to_mp4_bytes(&video_only, &soundtrack, 32_000, 2).unwrap();
        let mut video_file = tempfile::NamedTempFile::new().unwrap();
        video_file.write_all(&video_bytes).unwrap();
        video_file.flush().unwrap();
        let audio_probe = probe_decoded_mp4_audio_file(video_file.path())
            .unwrap()
            .unwrap();

        let audio_samples = (0..8_000)
            .flat_map(|sample| {
                let value = sample as f32 / 8_000.0;
                [value * 2.0, value * 3.0]
            })
            .collect::<Vec<_>>();
        let audio_bytes = encode_wav_f32_interleaved(&audio_samples, 16_000, 2).unwrap();

        let mut image_metadata = metadata(1, GenerationReferenceKind::Image, "image/png");
        image_metadata.width = Some(32);
        image_metadata.height = Some(32);
        let image_shape = GenerationReferencePreparedShape {
            version: mold_core::minimax_h3::REFERENCE_PREPROCESS_VERSION,
            normalized_width: Some(32),
            normalized_height: Some(32),
            normalized_video_frames: None,
            video_frames: None,
            qwen_video_frames: None,
            audio_samples_per_channel: None,
            visual_rows: 1,
            audio_rows: 0,
        };
        let (image_reference, image_binding) =
            prepared(image_metadata, image_shape, 24, &image_bytes);

        let mut video_metadata = metadata(2, GenerationReferenceKind::Video, "video/mp4");
        video_metadata.width = Some(32);
        video_metadata.height = Some(32);
        video_metadata.frame_count = Some(24);
        video_metadata.duration_ms = Some(1_000);
        video_metadata.fps = Some(24.0);
        video_metadata.has_audio = true;
        video_metadata.audio_duration_ms = Some(1_000);
        video_metadata.audio_sample_count = Some(audio_probe.samples_per_channel);
        video_metadata.audio_sample_rate = Some(audio_probe.sample_rate);
        video_metadata.audio_channels = Some(audio_probe.channels);
        let expected_video_audio = audio_probe
            .samples_per_channel
            .min(u64::from(audio_probe.sample_rate));
        let expected_video_audio = expected_video_audio
            .checked_mul(u64::from(AUDIO_SAMPLE_RATE_HZ))
            .unwrap()
            .div_ceil(u64::from(audio_probe.sample_rate));
        let video_shape = GenerationReferencePreparedShape {
            version: mold_core::minimax_h3::REFERENCE_PREPROCESS_VERSION,
            normalized_width: Some(32),
            normalized_height: Some(32),
            normalized_video_frames: Some(24),
            video_frames: Some(22),
            qwen_video_frames: Some(2),
            audio_samples_per_channel: Some(expected_video_audio),
            visual_rows: 7,
            audio_rows: expected_video_audio.div_ceil(800) * 2,
        };
        let (video_reference, video_binding) =
            prepared(video_metadata, video_shape, 24, &video_bytes);

        let mut audio_metadata = metadata(3, GenerationReferenceKind::Audio, "audio/wav");
        audio_metadata.duration_ms = Some(500);
        audio_metadata.sample_rate = Some(16_000);
        audio_metadata.channels = Some(2);
        audio_metadata.sample_count = Some(8_000);
        let audio_shape = GenerationReferencePreparedShape {
            version: mold_core::minimax_h3::REFERENCE_PREPROCESS_VERSION,
            normalized_width: None,
            normalized_height: None,
            normalized_video_frames: None,
            video_frames: None,
            qwen_video_frames: None,
            audio_samples_per_channel: Some(16_000),
            visual_rows: 0,
            audio_rows: 40,
        };
        let (audio_reference, audio_binding) =
            prepared(audio_metadata, audio_shape, 24, &audio_bytes);

        let references = [image_reference, video_reference, audio_reference];
        let bindings = [image_binding, video_binding, audio_binding];
        let mut adapter = H3ReferenceMediaAdapter::default();
        let mut checkpoint = TestCheckpoint::default();
        let decoded = references
            .iter()
            .zip(&bindings)
            .map(|(reference, binding)| {
                adapter.decode_reference(reference, binding, &mut checkpoint)
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let presentations = references
            .iter()
            .zip(&decoded)
            .map(|(reference, decoded)| {
                adapter.preprocess_reference(reference, decoded, &mut checkpoint)
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(adapter.normalized_indices(), vec![1, 2, 3]);
        assert_eq!(
            presentations
                .iter()
                .map(|presentation| presentation.index)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        let H3NormalizedReference::Video {
            vae_frames,
            qwen_frames,
            qwen_source_indices,
            qwen_block_timestamps_seconds,
            audio: Some(video_audio),
        } = adapter.normalized(2).unwrap()
        else {
            panic!("expected normalized video with soundtrack")
        };
        assert_eq!(vae_frames.len(), 22);
        assert_eq!(qwen_frames.len(), 2);
        assert_eq!(qwen_source_indices, &[0, 12]);
        assert_eq!(qwen_block_timestamps_seconds, &[0.25]);
        assert_eq!(video_audio.sample_rate, 32_000);
        assert_eq!(
            video_audio.samples_per_channel(),
            usize::try_from(expected_video_audio).unwrap()
        );
        assert!(presentations[1].presentation.has_audio);
        let H3NormalizedReference::Audio { audio } = adapter.normalized(3).unwrap() else {
            panic!("expected normalized audio")
        };
        assert_eq!(audio.sample_rate, 32_000);
        assert_eq!(audio.samples_per_channel(), 16_000);
        assert!((audio.channels[0][100] - 0.0125).abs() < 1e-6);
        assert!((audio.channels[1][100] - 0.01875).abs() < 1e-6);
    }
}
