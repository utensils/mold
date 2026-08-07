//! Exact, payload-free MiniMax H3 reference descriptors for Discord.
//!
//! Discord exposes two ordered generic attachment options because `/generate`
//! already uses 23 of Discord's 25 option slots. Bytes are downloaded once,
//! probed locally, and retained only long enough to upload through a fresh,
//! authenticated server session. Debug output redacts both bytes and handles.

use anyhow::{Context, Result};
use image::ImageReader;
use mold_core::{
    minimax_h3, GenerateRequest, GenerationReference, GenerationReferenceAuthority,
    GenerationReferenceProvenance, MoldClient, ReferenceUploadSessionRequest,
};
use mp4_rs::{MediaType, Mp4Reader, TrackType};
use poise::serenity_prelude as serenity;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    io::{Cursor, Read, Seek, SeekFrom},
};
use symphonia::{
    core::{
        audio::SampleBuffer, codecs::DecoderOptions, errors::Error as SymphoniaError,
        formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
    },
    default::{get_codecs, get_probe},
};

pub(crate) const MAX_DISCORD_REFERENCES: usize = 2;
const MAX_REFERENCE_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const MAX_REFERENCE_TOTAL_BYTES: usize = 47 * 1024 * 1024;

pub(crate) struct PreparedReferences {
    pub descriptors: Vec<GenerationReference>,
    uploads: Vec<ReferenceUpload>,
}

impl std::fmt::Debug for PreparedReferences {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedReferences")
            .field("descriptors", &self.descriptors)
            .field(
                "uploads",
                &format_args!("<{} redacted bodies>", self.uploads.len()),
            )
            .finish()
    }
}

struct ReferenceUpload {
    bytes: Vec<u8>,
    mime_type: String,
}

pub(crate) async fn prepare_attachments(
    attachments: &[&serenity::Attachment],
) -> Result<PreparedReferences> {
    anyhow::ensure!(
        !attachments.is_empty() && attachments.len() <= MAX_DISCORD_REFERENCES,
        "Discord accepts one or two ordered MiniMax H3 references"
    );
    let declared_total = attachments
        .iter()
        .try_fold(0_usize, |total, attachment| {
            total.checked_add(attachment.size as usize)
        })
        .context("combined reference attachment size overflowed")?;
    anyhow::ensure!(
        declared_total <= MAX_REFERENCE_TOTAL_BYTES,
        "combined reference attachments must stay under 47 MiB"
    );

    let mut bodies = Vec::with_capacity(attachments.len());
    for (index, attachment) in attachments.iter().enumerate() {
        let bytes = attachment
            .download()
            .await
            .map_err(|_| anyhow::anyhow!("reference {} download failed", index + 1))?;
        anyhow::ensure!(!bytes.is_empty(), "reference {} is empty", index + 1);
        bodies.push((attachment.filename.as_str(), bytes));
    }
    prepare_named_bytes(bodies)
}

fn prepare_named_bytes<'a>(
    bodies: impl IntoIterator<Item = (&'a str, Vec<u8>)>,
) -> Result<PreparedReferences> {
    let bodies = bodies.into_iter().collect::<Vec<_>>();
    anyhow::ensure!(
        !bodies.is_empty() && bodies.len() <= MAX_DISCORD_REFERENCES,
        "Discord accepts one or two ordered MiniMax H3 references"
    );
    let actual_total = bodies
        .iter()
        .try_fold(0_usize, |total, (_, bytes)| total.checked_add(bytes.len()))
        .context("combined reference body size overflowed")?;
    anyhow::ensure!(
        actual_total <= MAX_REFERENCE_TOTAL_BYTES,
        "combined reference attachments must stay under 47 MiB"
    );

    let mut descriptors = Vec::with_capacity(bodies.len());
    let mut uploads = Vec::with_capacity(bodies.len());
    for (index, (name, bytes)) in bodies.into_iter().enumerate() {
        let provenance = GenerationReferenceProvenance {
            name: Some(
                safe_name(name)
                    .with_context(|| format!("reference {} filename is invalid", index + 1))?,
            ),
            sha256: Some(format!("{:x}", Sha256::digest(&bytes))),
        };
        let descriptor = probe_reference(&bytes, provenance)
            .with_context(|| format!("reference {} media probe failed", index + 1))?;
        if matches!(descriptor, GenerationReference::Image { .. }) {
            anyhow::ensure!(
                bytes.len() <= MAX_REFERENCE_IMAGE_BYTES,
                "reference {} image must stay under 10 MiB",
                index + 1
            );
        }
        uploads.push(ReferenceUpload {
            mime_type: reference_mime_type(&descriptor).to_string(),
            bytes,
        });
        descriptors.push(descriptor);
    }
    minimax_h3::validate_reference_descriptors(&descriptors).map_err(anyhow::Error::new)?;
    Ok(PreparedReferences {
        descriptors,
        uploads,
    })
}

pub(crate) async fn bind_remote_references(
    client: &MoldClient,
    request: &mut GenerateRequest,
    prepared: PreparedReferences,
) -> Result<String> {
    anyhow::ensure!(
        prepared.descriptors.len() == prepared.uploads.len(),
        "reference descriptor/body count mismatch"
    );
    minimax_h3::validate_reference_descriptors(&prepared.descriptors)
        .map_err(anyhow::Error::new)?;
    request.references = Some(prepared.descriptors.clone());
    let upload_references = (1..=prepared.descriptors.len())
        .map(|index| u32::try_from(index).context("too many reference uploads"))
        .collect::<Result<Vec<_>>>()?;
    let session = client
        .create_reference_upload_session(&ReferenceUploadSessionRequest {
            request: request.clone(),
            upload_references,
        })
        .await?;
    if session.instance_id.trim().is_empty()
        || session.expires_at_ms == 0
        || !is_sha256_hex(&session.request_scope_sha256)
        || !is_upload_handle(&session.session_handle)
        || session.uploads.len() != prepared.descriptors.len()
    {
        let _ = client
            .cancel_reference_upload_session(&session.session_handle)
            .await;
        anyhow::bail!("server returned an incomplete reference upload session");
    }

    let result: Result<Vec<GenerationReference>> = async {
        let mut bound = Vec::with_capacity(prepared.descriptors.len());
        let mut seen = HashSet::new();
        for (index, (descriptor, upload)) in prepared
            .descriptors
            .iter()
            .zip(prepared.uploads)
            .enumerate()
        {
            let reference = u32::try_from(index)
                .context("too many reference uploads")?
                .saturating_add(1);
            let slot = session
                .uploads
                .iter()
                .find(|slot| slot.reference == reference)
                .ok_or_else(|| anyhow::anyhow!("server omitted reference upload slot {reference}"))?;
            if !is_upload_handle(&slot.handle) || !seen.insert(slot.handle.as_str()) {
                anyhow::bail!("server returned an invalid or reused reference upload handle");
            }
            let completed = client
                .upload_reference_bytes(&slot.handle, upload.bytes, &upload.mime_type)
                .await
                .with_context(|| format!("reference {reference} upload failed"))?;
            if completed.instance_id != session.instance_id || completed.reference != reference {
                anyhow::bail!(
                    "reference {reference} upload response did not match its bound server session"
                );
            }
            let expected = descriptor
                .redacted_metadata(index)
                .ok_or_else(|| anyhow::anyhow!("reference {reference} lost digest provenance"))?;
            if completed.metadata != expected {
                anyhow::bail!(
                    "reference {reference} upload response drifted from the client-probed descriptor"
                );
            }
            bound.push(with_authority(
                descriptor,
                GenerationReferenceAuthority::Upload {
                    handle: slot.handle.clone(),
                },
            ));
        }
        Ok(bound)
    }
    .await;

    match result {
        Ok(bound) => {
            request.references = Some(bound);
            Ok(session.session_handle)
        }
        Err(error) => {
            let _ = client
                .cancel_reference_upload_session(&session.session_handle)
                .await;
            Err(error)
        }
    }
}

fn probe_reference(
    bytes: &[u8],
    provenance: GenerationReferenceProvenance,
) -> Result<GenerationReference> {
    if let Ok(reader) = ImageReader::new(Cursor::new(bytes)).with_guessed_format() {
        if let Some(format) = reader.format() {
            let mime_type = match format {
                image::ImageFormat::Png => Some("image/png"),
                image::ImageFormat::Jpeg => Some("image/jpeg"),
                image::ImageFormat::WebP => Some("image/webp"),
                image::ImageFormat::Gif => Some("image/gif"),
                _ => None,
            };
            if let Some(mime_type) = mime_type {
                let (width, height) = reader.into_dimensions()?;
                return Ok(GenerationReference::Image {
                    media: GenerationReferenceAuthority::Descriptor,
                    provenance,
                    mime_type: mime_type.to_string(),
                    width,
                    height,
                });
            }
        }
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        let wav = probe_wav(bytes)?;
        return Ok(GenerationReference::Audio {
            media: GenerationReferenceAuthority::Descriptor,
            provenance,
            mime_type: "audio/wav".to_string(),
            duration_ms: wav.duration_ms,
            sample_rate: wav.sample_rate,
            channels: wav.channels,
            sample_count: Some(wav.sample_count),
        });
    }
    probe_video(bytes, provenance)
}

fn probe_video(
    bytes: &[u8],
    provenance: GenerationReferenceProvenance,
) -> Result<GenerationReference> {
    let reader = Mp4Reader::read_header(Cursor::new(bytes), bytes.len() as u64)
        .context("unsupported reference; use PNG, JPEG, WebP, GIF, H.264 MP4, or RIFF/WAVE")?;
    let track = reader
        .tracks()
        .values()
        .find(|track| matches!(track.track_type(), Ok(TrackType::Video)))
        .context("MP4 did not contain a video track")?;
    anyhow::ensure!(
        track.media_type()? == MediaType::H264,
        "MP4 video must use H.264/AVC"
    );
    let width = track.width() as u32;
    let height = track.height() as u32;
    let frame_count = track.sample_count();
    let fps = track.frame_rate().round() as u32;
    let duration_ms = track.duration().as_millis() as u64;
    anyhow::ensure!(
        width > 0 && height > 0 && fps > 0 && duration_ms > 0,
        "MP4 video metadata is incomplete"
    );
    let has_audio = reader
        .tracks()
        .values()
        .any(|track| matches!(track.track_type(), Ok(TrackType::Audio)));
    let decoded_audio = if has_audio {
        Some(
            probe_decoded_mp4_audio(bytes)?
                .context("MP4 audio track could not be decoded exactly")?,
        )
    } else {
        None
    };
    let audio_duration_ms = decoded_audio
        .as_ref()
        .map(|audio| {
            audio
                .samples_per_channel
                .checked_mul(1_000)
                .context("MP4 soundtrack duration overflowed")
                .map(|samples| samples.div_ceil(u64::from(audio.sample_rate)))
        })
        .transpose()?;
    Ok(GenerationReference::Video {
        media: GenerationReferenceAuthority::Descriptor,
        provenance,
        mime_type: "video/mp4".to_string(),
        width,
        height,
        frame_count: Some(frame_count),
        duration_ms,
        fps: f64::from(fps),
        has_audio,
        audio_duration_ms,
        audio_sample_count: decoded_audio
            .as_ref()
            .map(|audio| audio.samples_per_channel),
        audio_sample_rate: decoded_audio.as_ref().map(|audio| audio.sample_rate),
        audio_channels: decoded_audio.as_ref().map(|audio| audio.channels),
    })
}

#[derive(Debug, Clone, Copy)]
struct ExactAudioProbe {
    sample_rate: u32,
    channels: u16,
    samples_per_channel: u64,
}

/// Decode attachment audio instead of estimating it from MP4 duration or AAC
/// packet count. Codec delay and the final packet make either estimate unsafe
/// for H3's frozen Ref2VA row calculation.
fn probe_decoded_mp4_audio(bytes: &[u8]) -> Result<Option<ExactAudioProbe>> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp4");
    let probed = get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("failed to probe MP4 attachment audio")?;
    let mut format = probed.format;
    let Some(track) = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.sample_rate.is_some())
        .cloned()
    else {
        return Ok(None);
    };
    let track_id = track.id;
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("failed to create MP4 attachment audio decoder")?;
    let mut sample_rate = None;
    let mut channels = None;
    let mut samples_per_channel = 0_u64;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(error) => return Err(error.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) | Err(SymphoniaError::IoError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                anyhow::bail!("MP4 attachment audio decoder requested a reset")
            }
            Err(error) => return Err(error.into()),
        };
        let observed_rate = decoded.spec().rate;
        let observed_channels = decoded.spec().channels.count();
        anyhow::ensure!(
            observed_rate > 0 && matches!(observed_channels, 1 | 2),
            "MP4 attachment audio must be mono or stereo with a positive sample rate"
        );
        if let Some(expected) = sample_rate {
            anyhow::ensure!(
                expected == observed_rate,
                "MP4 attachment audio changed sample rate mid-stream"
            );
        }
        if let Some(expected) = channels {
            anyhow::ensure!(
                expected == observed_channels,
                "MP4 attachment audio changed channel count mid-stream"
            );
        }
        sample_rate = Some(observed_rate);
        channels = Some(observed_channels);
        let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        samples.copy_interleaved_ref(decoded);
        anyhow::ensure!(
            samples.samples().len().is_multiple_of(observed_channels),
            "MP4 attachment audio decoder returned a partial frame"
        );
        samples_per_channel = samples_per_channel
            .checked_add((samples.samples().len() / observed_channels) as u64)
            .context("MP4 attachment decoded sample count overflowed")?;
    }

    let (Some(sample_rate), Some(channels)) = (sample_rate, channels) else {
        return Ok(None);
    };
    anyhow::ensure!(samples_per_channel > 0, "MP4 attachment audio was empty");
    Ok(Some(ExactAudioProbe {
        sample_rate,
        channels: u16::try_from(channels).context("MP4 audio channel count exceeded u16")?,
        samples_per_channel,
    }))
}

struct WavProbe {
    duration_ms: u64,
    sample_rate: u32,
    channels: u16,
    sample_count: u64,
}

fn probe_wav(bytes: &[u8]) -> Result<WavProbe> {
    let file_len = bytes.len() as u64;
    let mut reader = Cursor::new(bytes);
    let mut header = [0_u8; 12];
    reader.read_exact(&mut header)?;
    anyhow::ensure!(
        &header[..4] == b"RIFF" && &header[8..] == b"WAVE",
        "not RIFF/WAVE"
    );
    let riff_end = 8_u64
        .checked_add(u64::from(u32::from_le_bytes(header[4..8].try_into()?)))
        .context("WAVE container length overflowed")?;
    anyhow::ensure!(
        riff_end >= 12 && riff_end <= file_len,
        "WAVE container is truncated"
    );
    let mut sample_rate = None;
    let mut channels = None;
    let mut byte_rate = None;
    let mut block_align = None;
    let mut data_bytes = None;
    while reader.stream_position()?.saturating_add(8) <= riff_end {
        let mut chunk = [0_u8; 8];
        reader.read_exact(&mut chunk)?;
        let length = u64::from(u32::from_le_bytes(chunk[4..].try_into()?));
        let start = reader.stream_position()?;
        let next = start
            .checked_add(
                length
                    .checked_add(length % 2)
                    .context("WAVE chunk overflowed")?,
            )
            .context("WAVE chunk offset overflowed")?;
        anyhow::ensure!(
            next <= riff_end && next <= file_len,
            "WAVE chunk is truncated"
        );
        match &chunk[..4] {
            b"fmt " => {
                anyhow::ensure!(length >= 16, "WAVE fmt chunk is truncated");
                let mut format = [0_u8; 16];
                reader.read_exact(&mut format)?;
                let encoding = u16::from_le_bytes(format[..2].try_into()?);
                let observed_channels = u16::from_le_bytes(format[2..4].try_into()?);
                let observed_rate = u32::from_le_bytes(format[4..8].try_into()?);
                let observed_byte_rate = u32::from_le_bytes(format[8..12].try_into()?);
                let observed_align = u16::from_le_bytes(format[12..14].try_into()?);
                let bits = u16::from_le_bytes(format[14..16].try_into()?);
                anyhow::ensure!(matches!(encoding, 1 | 3), "unsupported WAVE encoding");
                anyhow::ensure!(
                    observed_channels > 0
                        && observed_rate > 0
                        && observed_align > 0
                        && bits > 0
                        && bits % 8 == 0,
                    "invalid WAVE fields"
                );
                anyhow::ensure!(
                    observed_align
                        == observed_channels
                            .checked_mul(bits / 8)
                            .context("WAVE alignment overflowed")?,
                    "inconsistent WAVE alignment"
                );
                anyhow::ensure!(
                    observed_byte_rate == observed_rate.saturating_mul(u32::from(observed_align)),
                    "inconsistent WAVE byte rate"
                );
                sample_rate = Some(observed_rate);
                channels = Some(observed_channels);
                byte_rate = Some(observed_byte_rate);
                block_align = Some(observed_align);
            }
            b"data" => data_bytes = Some(length),
            _ => {}
        }
        reader.seek(SeekFrom::Start(next))?;
        if sample_rate.is_some() && data_bytes.is_some() {
            break;
        }
    }
    let sample_rate = sample_rate.context("WAVE has no sample rate")?;
    let channels = channels.context("WAVE has no channels")?;
    let byte_rate = u64::from(byte_rate.context("WAVE has no byte rate")?);
    let block_align = u64::from(block_align.context("WAVE has no block alignment")?);
    let data_bytes = data_bytes.context("WAVE has no data chunk")?;
    anyhow::ensure!(
        data_bytes > 0 && data_bytes % block_align == 0,
        "WAVE data is empty or unaligned"
    );
    Ok(WavProbe {
        duration_ms: data_bytes.saturating_mul(1_000).div_ceil(byte_rate),
        sample_rate,
        channels,
        sample_count: data_bytes / block_align,
    })
}

fn safe_name(value: &str) -> Option<String> {
    value
        .rsplit(['/', '\\'])
        .next()
        .map(str::trim)
        .filter(|name| {
            !name.is_empty()
                && name.len() <= minimax_h3::MAX_REFERENCE_NAME_BYTES
                && *name != "."
                && *name != ".."
                && !name.chars().any(char::is_control)
        })
        .map(str::to_string)
}

fn reference_mime_type(reference: &GenerationReference) -> &str {
    match reference {
        GenerationReference::Image { mime_type, .. }
        | GenerationReference::Video { mime_type, .. }
        | GenerationReference::Audio { mime_type, .. } => mime_type,
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_upload_handle(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= minimax_h3::MAX_REFERENCE_UPLOAD_HANDLE_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b':' | b'.'))
}

fn with_authority(
    reference: &GenerationReference,
    media: GenerationReferenceAuthority,
) -> GenerationReference {
    match reference {
        GenerationReference::Image {
            provenance,
            mime_type,
            width,
            height,
            ..
        } => GenerationReference::Image {
            media,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            width: *width,
            height: *height,
        },
        GenerationReference::Video {
            provenance,
            mime_type,
            width,
            height,
            frame_count,
            duration_ms,
            fps,
            has_audio,
            audio_duration_ms,
            audio_sample_count,
            audio_sample_rate,
            audio_channels,
            ..
        } => GenerationReference::Video {
            media,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            width: *width,
            height: *height,
            frame_count: *frame_count,
            duration_ms: *duration_ms,
            fps: *fps,
            has_audio: *has_audio,
            audio_duration_ms: *audio_duration_ms,
            audio_sample_count: *audio_sample_count,
            audio_sample_rate: *audio_sample_rate,
            audio_channels: *audio_channels,
        },
        GenerationReference::Audio {
            provenance,
            mime_type,
            duration_ms,
            sample_rate,
            channels,
            sample_count,
            ..
        } => GenerationReference::Audio {
            media,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            duration_ms: *duration_ms,
            sample_rate: *sample_rate,
            channels: *channels,
            sample_count: *sample_count,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(seconds: u32) -> Vec<u8> {
        let sample_rate = 8_000_u32;
        let data_len = sample_rate * seconds * 2;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        bytes.resize(44 + data_len as usize, 0);
        bytes
    }

    #[test]
    fn mixed_image_audio_order_and_provenance_are_exact() {
        let mut png = Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(24, 16)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let prepared = prepare_named_bytes([
            ("../../anchor.png", png.into_inner()),
            ("voice.wav", wav_bytes(2)),
        ])
        .unwrap();
        assert!(matches!(
            &prepared.descriptors[0],
            GenerationReference::Image { provenance, width: 24, height: 16, .. }
                if provenance.name.as_deref() == Some("anchor.png")
                    && provenance.sha256.as_ref().is_some_and(|digest| digest.len() == 64)
        ));
        assert!(matches!(
            &prepared.descriptors[1],
            GenerationReference::Audio {
                provenance,
                duration_ms: 2_000,
                sample_count: Some(16_000),
                ..
            }
                if provenance.name.as_deref() == Some("voice.wav")
        ));
        assert!(!format!("{prepared:?}").contains("89504e47"));
    }

    #[test]
    fn audio_only_fails_closed() {
        let error = prepare_named_bytes([("voice.wav", wav_bytes(2))])
            .expect_err("audio-only input must fail");
        assert!(error.to_string().contains("audio references require"));
    }
}
