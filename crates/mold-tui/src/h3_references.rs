//! Legal-safe MiniMax H3 Ref2VA authoring for the terminal surface.
//!
//! Paths stay in transient TUI state. Before any request is sent, each path is
//! opened once, cloned for hashing/probing, and the original file descriptor is
//! streamed through an authenticated, request-bound upload session. Neither the
//! local path nor the one-use handles are persisted or logged.

use anyhow::{Context, Result};
use mold_core::{
    minimax_h3, GenerateRequest, GenerationReference, GenerationReferenceAuthority,
    GenerationReferenceProvenance, MoldClient, ReferenceUploadSessionRequest,
};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

const MAX_REFERENCE_FILE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_REFERENCE_TOTAL_BYTES: u64 = 1024 * 1024 * 1024;
pub(crate) const MAX_REFERENCE_INPUT_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferenceKind {
    Image,
    Video,
    Audio,
}

impl ReferenceKind {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "image" => Some(Self::Image),
            "video" => Some(Self::Video),
            "audio" => Some(Self::Audio),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ReferencePath {
    pub kind: ReferenceKind,
    pub path: PathBuf,
}

impl std::fmt::Debug for ReferencePath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferencePath")
            .field("kind", &self.kind.label())
            .field("path", &"<redacted>")
            .finish()
    }
}

/// Parse the one-line TUI syntax `image=/a.png; video=/b.mp4; audio=/c.wav`.
/// Semicolon order is semantic. `split_once` deliberately permits `=` inside a
/// pathname; newlines are accepted too for tests and future multiline widgets.
pub(crate) fn parse_reference_input(
    value: &str,
) -> std::result::Result<Vec<ReferencePath>, String> {
    if value.len() > MAX_REFERENCE_INPUT_BYTES {
        return Err("Ordered reference input is too long.".to_string());
    }
    let mut references = Vec::new();
    for (index, item) in value
        .split([';', '\n'])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .enumerate()
    {
        let (kind, path) = item.split_once('=').ok_or_else(|| {
            format!(
                "Reference {} must use KIND=PATH (image, video, or audio).",
                index + 1
            )
        })?;
        let kind = ReferenceKind::parse(kind).ok_or_else(|| {
            format!(
                "Reference {} has unknown kind '{}'; use image, video, or audio.",
                index + 1,
                kind.trim()
            )
        })?;
        let path = path.trim();
        if path.is_empty() || path == "-" {
            return Err(format!(
                "Reference {} requires a file path for authenticated streaming upload.",
                index + 1
            ));
        }
        references.push(ReferencePath {
            kind,
            path: PathBuf::from(path),
        });
    }
    if references.len() > minimax_h3::MAX_REFERENCE_FILES {
        return Err(format!(
            "MiniMax H3 accepts at most {} ordered references.",
            minimax_h3::MAX_REFERENCE_FILES
        ));
    }
    Ok(references)
}

pub(crate) fn format_reference_input(references: &[ReferencePath]) -> String {
    references
        .iter()
        .map(|reference| {
            format!(
                "{}={}",
                reference.kind.label(),
                reference.path.to_string_lossy()
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Safe identity for conditioning staleness checks. The digest retains order
/// and kind without putting local paths into prompt-transform snapshots/logs.
pub(crate) fn authority_fingerprint(references: &[ReferencePath]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.tui.minimax-h3.reference-paths.v1\0");
    for reference in references {
        digest.update(reference.kind.label().as_bytes());
        digest.update(b"\0");
        digest.update(reference.path.as_os_str().as_encoded_bytes());
        digest.update(b"\0");
    }
    format!("{:x}", digest.finalize())
}

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
                &format_args!("<{} redacted file descriptors>", self.uploads.len()),
            )
            .finish()
    }
}

struct ReferenceUpload {
    file: File,
    mime_type: String,
}

/// Probe exact payload-free descriptors while retaining an already-open file
/// descriptor for the subsequent upload. Server ingress probes the staged bytes
/// independently and rejects any drift.
pub(crate) fn prepare_references(inputs: &[ReferencePath]) -> Result<PreparedReferences> {
    let mut descriptors = Vec::with_capacity(inputs.len());
    let mut uploads = Vec::with_capacity(inputs.len());
    let mut total_bytes = 0_u64;
    for (index, input) in inputs.iter().enumerate() {
        let mut file = File::open(&input.path)
            .with_context(|| format!("reference {} could not be opened", index + 1))?;
        let metadata = file
            .metadata()
            .with_context(|| format!("reference {} could not be inspected", index + 1))?;
        anyhow::ensure!(
            metadata.is_file() && metadata.len() > 0 && metadata.len() <= MAX_REFERENCE_FILE_BYTES,
            "reference {} must be a non-empty regular file no larger than {} MiB",
            index + 1,
            MAX_REFERENCE_FILE_BYTES / (1024 * 1024)
        );
        total_bytes = total_bytes
            .checked_add(metadata.len())
            .context("combined reference size overflowed")?;
        anyhow::ensure!(
            total_bytes <= MAX_REFERENCE_TOTAL_BYTES,
            "combined references must stay under {} MiB",
            MAX_REFERENCE_TOTAL_BYTES / (1024 * 1024)
        );
        let sha256 = sha256_file(
            file.try_clone()
                .with_context(|| format!("reference {} could not be cloned", index + 1))?,
        )
        .with_context(|| format!("reference {} could not be hashed", index + 1))?;
        file.seek(SeekFrom::Start(0))
            .with_context(|| format!("reference {} could not be rewound", index + 1))?;
        let provenance = GenerationReferenceProvenance {
            name: Some(
                display_name(&input.path)
                    .with_context(|| format!("reference {} filename is invalid", index + 1))?,
            ),
            sha256: Some(sha256),
        };
        let probe_file = file
            .try_clone()
            .with_context(|| format!("reference {} could not be cloned", index + 1))?;
        let descriptor = match input.kind {
            ReferenceKind::Image => probe_image(probe_file, provenance),
            ReferenceKind::Video => probe_video(probe_file, provenance),
            ReferenceKind::Audio => probe_audio(probe_file, provenance),
        }
        .with_context(|| format!("reference {} media probe failed", index + 1))?;
        file.seek(SeekFrom::Start(0))
            .with_context(|| format!("reference {} could not be rewound", index + 1))?;
        uploads.push(ReferenceUpload {
            file,
            mime_type: reference_mime_type(&descriptor).to_string(),
        });
        descriptors.push(descriptor);
    }
    minimax_h3::validate_reference_descriptors(&descriptors).map_err(anyhow::Error::new)?;
    Ok(PreparedReferences {
        descriptors,
        uploads,
    })
}

/// Bind and upload one exact attempt. Callers must invoke this again from the
/// original paths/bytes for any retry; handles are one-use and request-scoped.
pub(crate) async fn bind_remote_references(
    client: &MoldClient,
    request: &mut GenerateRequest,
    prepared: PreparedReferences,
) -> Result<String> {
    let PreparedReferences {
        descriptors,
        uploads,
    } = prepared;
    anyhow::ensure!(
        !descriptors.is_empty() && descriptors.len() == uploads.len(),
        "reference descriptor/file count mismatch"
    );
    minimax_h3::validate_reference_descriptors(&descriptors).map_err(anyhow::Error::new)?;
    request.references = Some(descriptors.clone());
    let upload_references = (1..=descriptors.len())
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
        || session.uploads.len() != descriptors.len()
    {
        let _ = client
            .cancel_reference_upload_session(&session.session_handle)
            .await;
        anyhow::bail!("server returned an incomplete reference upload session");
    }

    let upload_result: Result<Vec<GenerationReference>> = async {
        let mut bound = Vec::with_capacity(descriptors.len());
        let mut seen_handles = HashSet::new();
        for (index, (descriptor, upload)) in
            descriptors.iter().zip(uploads).enumerate()
        {
            let reference = u32::try_from(index)
                .context("too many reference uploads")?
                .saturating_add(1);
            let slot = session
                .uploads
                .iter()
                .find(|slot| slot.reference == reference)
                .ok_or_else(|| anyhow::anyhow!("server omitted reference upload slot {reference}"))?;
            if !is_upload_handle(&slot.handle) || !seen_handles.insert(slot.handle.as_str()) {
                anyhow::bail!("server returned an invalid or reused reference upload handle");
            }
            let completed = client
                .upload_reference_open_file(&slot.handle, upload.file, &upload.mime_type)
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

    match upload_result {
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

fn sha256_file(mut file: File) -> Result<String> {
    file.seek(SeekFrom::Start(0))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn display_name(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::trim)
        .filter(|name| {
            !name.is_empty()
                && name.len() <= minimax_h3::MAX_REFERENCE_NAME_BYTES
                && *name != "."
                && *name != ".."
                && !name.contains(['/', '\\'])
                && !name.chars().any(char::is_control)
        })
        .map(str::to_string)
}

fn probe_image(
    file: File,
    provenance: GenerationReferenceProvenance,
) -> Result<GenerationReference> {
    let reader = image::ImageReader::new(BufReader::new(file)).with_guessed_format()?;
    let format = reader
        .format()
        .ok_or_else(|| anyhow::anyhow!("unknown image format"))?;
    let mime_type = match format {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::WebP => "image/webp",
        image::ImageFormat::Gif => "image/gif",
        _ => anyhow::bail!("unsupported image format; use PNG, JPEG, WebP, or GIF"),
    };
    let (width, height) = reader.into_dimensions()?;
    Ok(GenerationReference::Image {
        media: GenerationReferenceAuthority::Descriptor,
        provenance,
        mime_type: mime_type.to_string(),
        width,
        height,
    })
}

fn probe_video(
    file: File,
    provenance: GenerationReferenceProvenance,
) -> Result<GenerationReference> {
    let probe = mold_inference::ltx2::media::probe_video_file(file)?;
    let duration_ms = probe
        .duration_ms
        .ok_or_else(|| anyhow::anyhow!("MP4 video duration is unavailable"))?;
    Ok(GenerationReference::Video {
        media: GenerationReferenceAuthority::Descriptor,
        provenance,
        mime_type: "video/mp4".to_string(),
        width: probe.width,
        height: probe.height,
        duration_ms,
        fps: f64::from(probe.fps),
        has_audio: probe.has_audio,
        audio_duration_ms: probe.has_audio.then_some(duration_ms),
    })
}

fn probe_audio(
    file: File,
    provenance: GenerationReferenceProvenance,
) -> Result<GenerationReference> {
    let wav = probe_wav(file)?;
    Ok(GenerationReference::Audio {
        media: GenerationReferenceAuthority::Descriptor,
        provenance,
        mime_type: "audio/wav".to_string(),
        duration_ms: wav.duration_ms,
        sample_rate: wav.sample_rate,
        channels: wav.channels,
    })
}

struct WavProbe {
    duration_ms: u64,
    sample_rate: u32,
    channels: u16,
}

fn probe_wav(file: File) -> Result<WavProbe> {
    let file_len = file.metadata()?.len();
    anyhow::ensure!(file_len <= MAX_REFERENCE_FILE_BYTES);
    let mut reader = BufReader::new(file);
    let mut header = [0_u8; 12];
    reader.read_exact(&mut header)?;
    anyhow::ensure!(
        &header[..4] == b"RIFF" && &header[8..] == b"WAVE",
        "not a RIFF/WAVE file"
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
        let data_start = reader.stream_position()?;
        let padded_length = length
            .checked_add(length % 2)
            .context("WAVE chunk length overflowed")?;
        let next_chunk = data_start
            .checked_add(padded_length)
            .context("WAVE chunk offset overflowed")?;
        anyhow::ensure!(
            next_chunk <= riff_end && next_chunk <= file_len,
            "WAVE chunk is truncated"
        );
        match &chunk[..4] {
            b"fmt " => {
                anyhow::ensure!(length >= 16, "WAVE fmt chunk is truncated");
                let mut format = [0_u8; 16];
                reader.read_exact(&mut format)?;
                let encoding = u16::from_le_bytes(format[..2].try_into()?);
                anyhow::ensure!(
                    matches!(encoding, 1 | 3),
                    "unsupported WAVE sample encoding"
                );
                let observed_channels = u16::from_le_bytes(format[2..4].try_into()?);
                let observed_sample_rate = u32::from_le_bytes(format[4..8].try_into()?);
                let observed_byte_rate = u32::from_le_bytes(format[8..12].try_into()?);
                let observed_block_align = u16::from_le_bytes(format[12..14].try_into()?);
                let bits_per_sample = u16::from_le_bytes(format[14..16].try_into()?);
                anyhow::ensure!(
                    observed_channels > 0
                        && observed_sample_rate > 0
                        && observed_block_align > 0
                        && bits_per_sample > 0
                        && bits_per_sample % 8 == 0,
                    "WAVE format fields are invalid"
                );
                let expected_align = observed_channels
                    .checked_mul(bits_per_sample / 8)
                    .context("WAVE block alignment overflowed")?;
                anyhow::ensure!(
                    observed_block_align == expected_align,
                    "WAVE block alignment is inconsistent"
                );
                anyhow::ensure!(
                    observed_byte_rate
                        == observed_sample_rate.saturating_mul(u32::from(observed_block_align)),
                    "WAVE byte rate is inconsistent"
                );
                sample_rate = Some(observed_sample_rate);
                channels = Some(observed_channels);
                byte_rate = Some(observed_byte_rate);
                block_align = Some(observed_block_align);
            }
            b"data" => data_bytes = Some(length),
            _ => {}
        }
        reader.seek(SeekFrom::Start(next_chunk))?;
        if sample_rate.is_some() && data_bytes.is_some() {
            break;
        }
    }
    let sample_rate = sample_rate.context("WAVE file has no fmt chunk")?;
    let channels = channels.context("WAVE file has no channel count")?;
    let byte_rate = byte_rate.context("WAVE file has no byte rate")?;
    let block_align = block_align.context("WAVE file has no block alignment")?;
    let data_bytes = data_bytes.context("WAVE file has no data chunk")?;
    anyhow::ensure!(data_bytes > 0, "WAVE data chunk is empty");
    anyhow::ensure!(
        data_bytes % u64::from(block_align) == 0,
        "WAVE data is not block aligned"
    );
    let duration_ms = data_bytes
        .checked_mul(1_000)
        .context("WAVE duration overflowed")?
        .div_ceil(u64::from(byte_rate));
    anyhow::ensure!(duration_ms > 0, "WAVE duration is zero");
    Ok(WavProbe {
        duration_ms,
        sample_rate,
        channels,
    })
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
            duration_ms,
            fps,
            has_audio,
            audio_duration_ms,
            ..
        } => GenerationReference::Video {
            media,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            width: *width,
            height: *height,
            duration_ms: *duration_ms,
            fps: *fps,
            has_audio: *has_audio,
            audio_duration_ms: *audio_duration_ms,
        },
        GenerationReference::Audio {
            provenance,
            mime_type,
            duration_ms,
            sample_rate,
            channels,
            ..
        } => GenerationReference::Audio {
            media,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            duration_ms: *duration_ms,
            sample_rate: *sample_rate,
            channels: *channels,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(seconds: u32) -> Vec<u8> {
        let sample_rate = 8_000_u32;
        let channels = 1_u16;
        let samples = sample_rate * seconds;
        let data_len = samples * u32::from(channels) * 2;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
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
    fn parser_preserves_mixed_order_and_equals_inside_paths() {
        let references =
            parse_reference_input("video=/tmp/a=b.mp4; image=/tmp/frame.png; audio=/tmp/voice.wav")
                .unwrap();
        assert_eq!(references.len(), 3);
        assert_eq!(references[0].kind, ReferenceKind::Video);
        assert_eq!(references[0].path, PathBuf::from("/tmp/a=b.mp4"));
        assert_eq!(references[1].kind, ReferenceKind::Image);
        assert_eq!(references[2].kind, ReferenceKind::Audio);
        assert_eq!(
            format_reference_input(&references),
            "video=/tmp/a=b.mp4; image=/tmp/frame.png; audio=/tmp/voice.wav"
        );
        let fingerprint = authority_fingerprint(&references);
        assert_eq!(fingerprint.len(), 64);
        assert!(!fingerprint.contains("tmp"));
        assert!(!format!("{references:?}").contains("/tmp"));
    }

    #[test]
    fn preparation_keeps_image_then_audio_provenance_and_rejects_audio_only() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("anchor.png");
        let audio_path = dir.path().join("voice.wav");
        image::DynamicImage::new_rgb8(32, 16)
            .save(&image_path)
            .unwrap();
        std::fs::write(&audio_path, wav_bytes(2)).unwrap();

        let inputs = vec![
            ReferencePath {
                kind: ReferenceKind::Image,
                path: image_path,
            },
            ReferencePath {
                kind: ReferenceKind::Audio,
                path: audio_path.clone(),
            },
        ];
        let prepared = prepare_references(&inputs).unwrap();
        assert_eq!(prepared.descriptors.len(), 2);
        assert!(matches!(
            &prepared.descriptors[0],
            GenerationReference::Image { provenance, width: 32, height: 16, .. }
                if provenance.name.as_deref() == Some("anchor.png")
                    && provenance.sha256.as_ref().is_some_and(|value| value.len() == 64)
        ));
        assert!(matches!(
            &prepared.descriptors[1],
            GenerationReference::Audio { provenance, duration_ms: 2_000, .. }
                if provenance.name.as_deref() == Some("voice.wav")
                    && provenance.sha256.as_ref().is_some_and(|value| value.len() == 64)
        ));

        let error = prepare_references(&[ReferencePath {
            kind: ReferenceKind::Audio,
            path: audio_path,
        }])
        .expect_err("audio-only references must fail");
        assert!(error.to_string().contains("audio references require"));
    }

    #[test]
    fn parser_enforces_twelve_file_ceiling_before_io() {
        let input = (0..13)
            .map(|index| format!("image=/tmp/{index}.png"))
            .collect::<Vec<_>>()
            .join("; ");
        assert!(parse_reference_input(&input)
            .unwrap_err()
            .contains("at most 12"));
    }
}
