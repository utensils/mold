//! MiniMax H3 command-line authoring.
//!
//! This module deliberately owns only payload-free request construction and
//! local user-media inspection. It does not activate, advertise, resolve, or
//! download the compliance-gated H3 model family. Runtime activation remains
//! authoritative in `mold-core` and the server.

use anyhow::{Context, Result};
use mold_core::{
    minimax_h3, GenerationReference, GenerationReferenceAuthority, GenerationReferenceProvenance,
    KeyframeCondition, OutputFormat,
};
use std::{
    fs::File,
    io::{BufReader, Cursor, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    str::FromStr,
};

const MAX_REFERENCE_FILE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_REFERENCE_TOTAL_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_BOUNDARY_IMAGE_BYTES: u64 = minimax_h3::MAX_INLINE_REFERENCE_BYTES as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferenceKind {
    Image,
    Video,
    Audio,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ReferenceArg {
    pub kind: ReferenceKind,
    pub path: PathBuf,
}

impl std::fmt::Debug for ReferenceArg {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceArg")
            .field("kind", &self.kind)
            .field("path", &"<redacted>")
            .finish()
    }
}

impl FromStr for ReferenceArg {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let (kind, path) = value.split_once('=').ok_or_else(|| {
            "reference must use KIND=PATH (image=..., video=..., or audio=...)".to_string()
        })?;
        let kind = match kind.trim().to_ascii_lowercase().as_str() {
            "image" => ReferenceKind::Image,
            "video" => ReferenceKind::Video,
            "audio" => ReferenceKind::Audio,
            other => {
                return Err(format!(
                    "unknown reference kind '{other}'; expected image, video, or audio"
                ));
            }
        };
        if path.trim().is_empty() {
            return Err("reference path must not be empty".to_string());
        }
        if path == "-" {
            return Err(
                "H3 references require file paths so they can use authenticated streaming upload"
                    .to_string(),
            );
        }
        Ok(Self {
            kind,
            path: PathBuf::from(path),
        })
    }
}

pub(crate) struct ReferenceUpload {
    pub file: File,
}

impl std::fmt::Debug for ReferenceUpload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceUpload")
            .field("file", &"<redacted>")
            .finish()
    }
}

#[derive(Default)]
pub(crate) struct PreparedAuthoring {
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub source_image: Option<Vec<u8>>,
    pub source_image_name: Option<String>,
    pub keyframes: Option<Vec<KeyframeCondition>>,
    pub references: Option<Vec<GenerationReference>>,
    pub uploads: Vec<ReferenceUpload>,
}

impl std::fmt::Debug for PreparedAuthoring {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedAuthoring")
            .field("frames", &self.frames)
            .field("fps", &self.fps)
            .field("width", &self.width)
            .field("height", &self.height)
            .field(
                "source_image",
                &self
                    .source_image
                    .as_ref()
                    .map(|bytes| format!("<redacted {} bytes>", bytes.len())),
            )
            .field("source_image_name", &self.source_image_name)
            .field("keyframes", &self.keyframes.as_ref().map(Vec::len))
            .field("references", &self.references)
            .field("uploads", &self.uploads)
            .finish()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_authoring(
    model: &str,
    family: &str,
    duration_seconds: Option<f64>,
    frames: Option<u32>,
    fps: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    first_frame: Option<&Path>,
    legacy_image: Option<&str>,
    last_frame: Option<&Path>,
    references: &[ReferenceArg],
    reference_client: Option<&mold_core::MoldClient>,
) -> Result<PreparedAuthoring> {
    let is_h3 = minimax_h3::is_family(family);
    let has_h3_authoring = duration_seconds.is_some()
        || first_frame.is_some()
        || last_frame.is_some()
        || !references.is_empty();
    if !is_h3 {
        if has_h3_authoring {
            anyhow::bail!(
                "--duration, --first-frame, --last-frame, and --reference are MiniMax H3-only options"
            );
        }
        return Ok(PreparedAuthoring::default());
    }

    let task = minimax_h3::task_for_model(model)
        .ok_or_else(|| anyhow::anyhow!("unknown MiniMax H3 task for model '{model}'"))?;
    if frames.is_some() && duration_seconds.is_some() {
        anyhow::bail!("use either --duration or --frames for MiniMax H3, not both");
    }
    if fps.is_some_and(|value| value != minimax_h3::FIXED_FPS) {
        anyhow::bail!(
            "MiniMax H3 requires --fps {}; received {}",
            minimax_h3::FIXED_FPS,
            fps.unwrap_or_default()
        );
    }
    let frames = match duration_seconds {
        Some(seconds) => frames_for_duration(seconds)?,
        None => frames.unwrap_or(minimax_h3::MIN_FRAMES),
    };
    if !minimax_h3::valid_frame_count(frames) {
        anyhow::bail!(
            "MiniMax H3 --frames must be {}n+{} from {} through {}; received {} (nearest valid value: {})",
            minimax_h3::FRAME_STEP,
            minimax_h3::FRAME_OFFSET,
            minimax_h3::MIN_FRAMES,
            minimax_h3::MAX_FRAMES,
            frames,
            minimax_h3::recommended_frames(frames),
        );
    }
    if width.is_some() != height.is_some() {
        anyhow::bail!("MiniMax H3 requires --width and --height to be supplied together");
    }
    if let (Some(width), Some(height)) = (width, height) {
        validate_dimensions(width, height)?;
    }

    let first_path =
        first_frame.or_else(|| legacy_image.filter(|path| *path != "-").map(Path::new));
    match task {
        minimax_h3::Task::Fl2va => {
            if !references.is_empty() {
                anyhow::bail!(
                    "MiniMax H3 FL2VA accepts boundary frames, not --reference; select a minimax-h3-ref2va model for ordered references"
                );
            }
        }
        minimax_h3::Task::Ref2va => {
            if first_path.is_some() || legacy_image == Some("-") || last_frame.is_some() {
                anyhow::bail!(
                    "MiniMax H3 Ref2VA accepts ordered --reference inputs, not --image/--first-frame/--last-frame"
                );
            }
            if references.is_empty() {
                anyhow::bail!(
                    "MiniMax H3 Ref2VA requires at least one ordered --reference KIND=PATH"
                );
            }
            anyhow::ensure!(
                reference_client.is_some_and(mold_core::MoldClient::has_api_key),
                "MiniMax H3 reference preparation requires a configured, valid MOLD_API_KEY"
            );
        }
    }

    let (source_image, source_image_name, first_dimensions) = match first_path {
        Some(path) => {
            let (bytes, dimensions) = read_boundary_image(path, "first")?;
            (Some(bytes), display_name(path), Some(dimensions))
        }
        None if legacy_image == Some("-") => {
            // Stdin is read by the ordinary image path. It has no safe filename
            // provenance, and dimensions are resolved after that read.
            (None, None, None)
        }
        None => (None, None, None),
    };
    let (keyframes, last_dimensions) = match last_frame {
        Some(path) => {
            let (image, dimensions) = read_boundary_image(path, "last")?;
            (
                Some(vec![KeyframeCondition {
                    frame: frames - 1,
                    image,
                    name: display_name(path),
                }]),
                Some(dimensions),
            )
        }
        None => (None, None),
    };

    let inferred_dimensions = first_dimensions.or(last_dimensions);
    let (width, height) = match (width, height, inferred_dimensions) {
        (Some(width), Some(height), _) => (Some(width), Some(height)),
        (None, None, Some((source_width, source_height))) => {
            let (width, height) = minimax_h3::recommended_dimensions(source_width, source_height);
            (Some(width), Some(height))
        }
        (None, None, None) => (None, None),
        _ => unreachable!("partial MiniMax H3 dimensions were rejected above"),
    };

    let (reference_descriptors, uploads) = prepare_references(references)?;
    Ok(PreparedAuthoring {
        frames: Some(frames),
        fps: Some(minimax_h3::FIXED_FPS),
        width,
        height,
        source_image,
        source_image_name,
        keyframes,
        references: (!reference_descriptors.is_empty()).then_some(reference_descriptors),
        uploads,
    })
}

pub(crate) struct FlagValidation<'a> {
    pub family: &'a str,
    pub format: Option<OutputFormat>,
    pub guidance: Option<f64>,
    pub strength: Option<f64>,
    pub no_audio: bool,
    pub negative_prompt: Option<&'a str>,
    pub scheduler_set: bool,
    pub cfg_plus: bool,
    pub has_lora: bool,
    pub has_mask: bool,
    pub has_control: bool,
    pub has_audio_file: bool,
    pub has_video: bool,
    pub has_extend: bool,
    pub has_generic_keyframes: bool,
    pub has_ltx_pipeline_fields: bool,
    pub clip_frames: Option<u32>,
}

pub(crate) fn validate_h3_flags(flags: FlagValidation<'_>) -> Result<()> {
    if !minimax_h3::is_family(flags.family) {
        return Ok(());
    }
    if flags.format.is_some_and(|value| value != OutputFormat::Mp4) {
        anyhow::bail!("MiniMax H3 requires --format mp4 for synchronized audio-video output");
    }
    if flags.guidance.is_some_and(|value| value != 0.0) {
        anyhow::bail!("MiniMax H3 has no CFG path; omit --guidance or use --guidance 0");
    }
    if flags.strength.is_some_and(|value| value != 1.0) {
        anyhow::bail!("MiniMax H3 has no denoise-strength control; omit --strength or use 1");
    }
    if flags.no_audio {
        anyhow::bail!("MiniMax H3 always generates synchronized audio; --no-audio is unsupported");
    }
    if flags
        .negative_prompt
        .is_some_and(|value| !value.trim().is_empty())
    {
        anyhow::bail!("MiniMax H3 has no negative-prompt branch; remove --negative-prompt");
    }
    if flags.scheduler_set {
        anyhow::bail!(
            "MiniMax H3 uses fixed video/audio flow schedules; --scheduler is unsupported"
        );
    }
    if flags.cfg_plus {
        anyhow::bail!("MiniMax H3 has no CFG path; --cfg-plus is unsupported");
    }
    if flags.has_lora || flags.has_mask || flags.has_control {
        anyhow::bail!("MiniMax H3 does not accept LoRA, mask, or ControlNet options");
    }
    if flags.has_audio_file || flags.has_video || flags.has_extend {
        anyhow::bail!(
            "MiniMax H3 uses FL2VA boundary frames or Ref2VA --reference inputs; --audio-file, --video, and --extend are unsupported"
        );
    }
    if flags.has_generic_keyframes {
        anyhow::bail!(
            "MiniMax H3 accepts only endpoint frames; use --first-frame and/or --last-frame instead of --keyframe"
        );
    }
    if flags.has_ltx_pipeline_fields {
        anyhow::bail!(
            "MiniMax H3 does not accept LTX-2 pipeline, HDR, upscale, or guidance controls"
        );
    }
    if flags.clip_frames.is_some() {
        anyhow::bail!("MiniMax H3 is a single-clip request; --clip-frames is unsupported");
    }
    Ok(())
}

fn frames_for_duration(seconds: f64) -> Result<u32> {
    if !seconds.is_finite() {
        anyhow::bail!("MiniMax H3 --duration must be a finite number of seconds");
    }
    if seconds < f64::from(minimax_h3::MIN_DURATION_SECONDS)
        || seconds > f64::from(minimax_h3::MAX_DURATION_SECONDS)
    {
        anyhow::bail!(
            "MiniMax H3 --duration must be between {} and {} seconds",
            minimax_h3::MIN_DURATION_SECONDS,
            minimax_h3::MAX_DURATION_SECONDS
        );
    }
    let requested = seconds * f64::from(minimax_h3::FIXED_FPS);
    let offset = f64::from(minimax_h3::FRAME_OFFSET);
    let step = f64::from(minimax_h3::FRAME_STEP);
    let lower = minimax_h3::FRAME_OFFSET
        + (((requested - offset) / step).floor() as u32) * minimax_h3::FRAME_STEP;
    let upper = lower.saturating_add(minimax_h3::FRAME_STEP);
    let rounded = if requested - f64::from(lower) <= f64::from(upper) - requested {
        lower
    } else {
        upper
    };
    Ok(rounded.clamp(minimax_h3::MIN_FRAMES, minimax_h3::MAX_FRAMES))
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    let valid = width > 0
        && height > 0
        && width.is_multiple_of(minimax_h3::DIMENSION_ALIGNMENT)
        && height.is_multiple_of(minimax_h3::DIMENSION_ALIGNMENT)
        && u64::from(width) * u64::from(height) <= minimax_h3::MAX_PIXELS
        && (minimax_h3::MIN_ASPECT_RATIO..=minimax_h3::MAX_ASPECT_RATIO)
            .contains(&(f64::from(width) / f64::from(height)));
    if !valid {
        let (recommended_width, recommended_height) =
            minimax_h3::recommended_dimensions(width, height);
        anyhow::bail!(
            "MiniMax H3 dimensions must be positive multiples of {}, at most {} pixels, with aspect ratio 1:4 through 4:1; nearest official canvas is {}x{}",
            minimax_h3::DIMENSION_ALIGNMENT,
            minimax_h3::MAX_PIXELS,
            recommended_width,
            recommended_height,
        );
    }
    Ok(())
}

fn read_boundary_image(path: &Path, boundary: &str) -> Result<(Vec<u8>, (u32, u32))> {
    let mut file = mold_core::secure_file::open_regular_file_no_follow(path)
        .with_context(|| format!("failed to open {boundary} frame"))?;
    let length = file.metadata()?.len();
    anyhow::ensure!(
        length > 0 && length <= MAX_BOUNDARY_IMAGE_BYTES,
        "MiniMax H3 {boundary} frame must contain 1..={MAX_BOUNDARY_IMAGE_BYTES} bytes"
    );
    let mut bytes = Vec::with_capacity(usize::try_from(length)?);
    file.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {boundary} frame"))?;
    let dimensions = validate_boundary_image_bytes(&bytes, boundary).map_err(|error| {
        anyhow::anyhow!("invalid {boundary} frame '{}': {error}", path.display())
    })?;
    Ok((bytes, dimensions))
}

pub(crate) fn validate_boundary_image_bytes(bytes: &[u8], boundary: &str) -> Result<(u32, u32)> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_BOUNDARY_IMAGE_BYTES {
        anyhow::bail!(
            "MiniMax H3 {boundary} frame must contain 1..={MAX_BOUNDARY_IMAGE_BYTES} bytes"
        );
    }
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .with_context(|| format!("failed to identify MiniMax H3 {boundary} frame"))?;
    let format = reader
        .format()
        .ok_or_else(|| anyhow::anyhow!("unknown MiniMax H3 {boundary} frame format"))?;
    if !matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg) {
        anyhow::bail!("MiniMax H3 {boundary} frame must be PNG or JPEG; received {format:?}");
    }
    let dimensions = reader
        .into_dimensions()
        .with_context(|| format!("failed to decode MiniMax H3 {boundary} frame"))?;
    Ok(dimensions)
}

fn prepare_references(
    inputs: &[ReferenceArg],
) -> Result<(Vec<GenerationReference>, Vec<ReferenceUpload>)> {
    let mut descriptors = Vec::with_capacity(inputs.len());
    let mut uploads = Vec::with_capacity(inputs.len());
    let mut total_bytes = 0_u64;
    for (index, input) in inputs.iter().enumerate() {
        let mut file = mold_core::secure_file::open_regular_file_no_follow(&input.path)
            .with_context(|| format!("reference {} could not be opened no-follow", index + 1))?;
        let length = file.metadata()?.len();
        anyhow::ensure!(
            length > 0 && length <= MAX_REFERENCE_FILE_BYTES,
            "reference {} must be a non-empty regular file no larger than {} MiB",
            index + 1,
            MAX_REFERENCE_FILE_BYTES / (1024 * 1024)
        );
        total_bytes = total_bytes
            .checked_add(length)
            .context("combined reference size overflowed")?;
        anyhow::ensure!(
            total_bytes <= MAX_REFERENCE_TOTAL_BYTES,
            "combined references must stay under {} MiB",
            MAX_REFERENCE_TOTAL_BYTES / (1024 * 1024)
        );
        let sha256 = mold_core::secure_file::sha256_open_file(&file)
            .with_context(|| format!("failed to hash reference {}", index + 1))?;
        file.seek(SeekFrom::Start(0))?;
        let provenance = GenerationReferenceProvenance {
            name: display_name(&input.path),
            sha256: Some(sha256),
        };
        let descriptor = match input.kind {
            ReferenceKind::Image => probe_image(file.try_clone()?, provenance)
                .with_context(|| format!("reference {} image probe failed", index + 1))?,
            ReferenceKind::Video => probe_video(file.try_clone()?, provenance)
                .with_context(|| format!("reference {} video probe failed", index + 1))?,
            ReferenceKind::Audio => probe_audio(file.try_clone()?, provenance)
                .with_context(|| format!("reference {} audio probe failed", index + 1))?,
        };
        file.seek(SeekFrom::Start(0))?;
        uploads.push(ReferenceUpload { file });
        descriptors.push(descriptor);
    }
    if !descriptors.is_empty() {
        minimax_h3::validate_reference_descriptors(&descriptors).map_err(anyhow::Error::new)?;
    }
    Ok((descriptors, uploads))
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
    let probe = mold_inference::ltx2::media::probe_video_file(file.try_clone()?)?;
    let duration_ms = probe
        .duration_ms
        .ok_or_else(|| anyhow::anyhow!("MP4 video duration is unavailable"))?;
    let decoded_audio = if probe.has_audio {
        Some(
            mold_inference::ltx2::media::probe_decoded_mp4_audio_open_file(file)?
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
        width: probe.width,
        height: probe.height,
        frame_count: probe.frames,
        duration_ms,
        fps: f64::from(probe.fps),
        has_audio: probe.has_audio,
        audio_duration_ms,
        audio_sample_count: decoded_audio
            .as_ref()
            .map(|audio| audio.samples_per_channel),
        audio_sample_rate: decoded_audio.as_ref().map(|audio| audio.sample_rate),
        audio_channels: decoded_audio.as_ref().map(|audio| audio.channels),
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
        sample_count: Some(wav.sample_count),
    })
}

#[derive(Debug)]
struct WavProbe {
    duration_ms: u64,
    sample_rate: u32,
    channels: u16,
    sample_count: u64,
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
                channels = Some(observed_channels);
                sample_rate = Some(observed_sample_rate);
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
    let sample_rate = sample_rate
        .filter(|rate| *rate > 0)
        .context("WAVE sample rate is missing")?;
    let channels = channels
        .filter(|channels| *channels > 0)
        .context("WAVE channel count is missing")?;
    let byte_rate = u64::from(
        byte_rate
            .filter(|rate| *rate > 0)
            .context("WAVE byte rate is missing")?,
    );
    let block_align = u64::from(block_align.context("WAVE block alignment is missing")?);
    let data_bytes = data_bytes
        .filter(|bytes| *bytes > 0)
        .context("WAVE data chunk is missing or empty")?;
    anyhow::ensure!(
        data_bytes % block_align == 0,
        "WAVE data is not frame-aligned"
    );
    Ok(WavProbe {
        duration_ms: data_bytes.saturating_mul(1_000).div_ceil(byte_rate),
        sample_rate,
        channels,
        sample_count: data_bytes / block_align,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder as _;

    #[test]
    fn reference_arg_preserves_kind_and_paths_containing_equals() {
        let parsed: ReferenceArg = "video=/tmp/clip=final.mp4".parse().unwrap();
        assert_eq!(parsed.kind, ReferenceKind::Video);
        assert_eq!(parsed.path, PathBuf::from("/tmp/clip=final.mp4"));
        assert!(!format!("{parsed:?}").contains("clip=final"));
    }

    #[test]
    fn reference_arg_rejects_implicit_or_stdin_media() {
        assert!("/tmp/a.png".parse::<ReferenceArg>().is_err());
        assert!("audio=-".parse::<ReferenceArg>().is_err());
        assert!("document=/tmp/a.pdf".parse::<ReferenceArg>().is_err());
    }

    #[test]
    fn duration_resolves_onto_exact_h3_grid() {
        assert_eq!(frames_for_duration(5.0).unwrap(), 124);
        assert_eq!(frames_for_duration(10.0).unwrap(), 243);
        assert_eq!(frames_for_duration(15.0).unwrap(), 345);
        assert_eq!(frames_for_duration(200.5 / 24.0).unwrap(), 192);
        assert!(frames_for_duration(4.99).is_err());
        assert!(frames_for_duration(15.01).is_err());
        assert!(frames_for_duration(f64::NAN).is_err());
    }

    #[test]
    fn fl2va_last_frame_uses_resolved_final_index_and_source_canvas() {
        let dir = tempfile::tempdir().unwrap();
        let last = dir.path().join("closing.png");
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(
                &[0_u8; 640 * 360 * 3],
                640,
                360,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        std::fs::write(&last, png).unwrap();

        let prepared = prepare_authoring(
            minimax_h3::FL2VA_COMFY,
            minimax_h3::FAMILY,
            Some(5.0),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&last),
            &[],
            None,
        )
        .unwrap();
        assert_eq!(prepared.frames, Some(124));
        assert_eq!(prepared.fps, Some(24));
        assert_eq!(prepared.width.zip(prepared.height), Some((1344, 768)));
        let keyframe = &prepared.keyframes.unwrap()[0];
        assert_eq!(keyframe.frame, 123);
        assert_eq!(keyframe.name.as_deref(), Some("closing.png"));
    }

    #[test]
    fn fl2va_boundary_rejects_reference_only_image_formats() {
        let dir = tempfile::tempdir().unwrap();
        let last = dir.path().join("closing.webp");
        let mut webp = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(32, 32)
            .write_to(&mut webp, image::ImageFormat::WebP)
            .unwrap();
        std::fs::write(&last, webp.into_inner()).unwrap();

        let error = prepare_authoring(
            minimax_h3::FL2VA_COMFY,
            minimax_h3::FAMILY,
            Some(5.0),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&last),
            &[],
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("must be PNG or JPEG"));
    }

    #[test]
    fn h3_explicit_incompatible_shared_flags_fail_clearly() {
        let error = validate_h3_flags(FlagValidation {
            family: minimax_h3::FAMILY,
            format: Some(OutputFormat::Png),
            guidance: None,
            strength: None,
            no_audio: false,
            negative_prompt: None,
            scheduler_set: false,
            cfg_plus: false,
            has_lora: false,
            has_mask: false,
            has_control: false,
            has_audio_file: false,
            has_video: false,
            has_extend: false,
            has_generic_keyframes: false,
            has_ltx_pipeline_fields: false,
            clip_frames: None,
        })
        .unwrap_err();
        assert!(error.to_string().contains("--format mp4"));
    }

    #[cfg(unix)]
    #[test]
    fn reference_preparation_rejects_symlink_input() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("anchor.png");
        image::DynamicImage::new_rgb8(32, 16).save(&target).unwrap();
        let link = dir.path().join("linked.png");
        symlink(target, &link).unwrap();
        let error = prepare_references(&[ReferenceArg {
            kind: ReferenceKind::Image,
            path: link,
        }])
        .expect_err("symlink references must fail no-follow");
        assert!(error.to_string().contains("no-follow"));
    }

    #[test]
    fn ref2va_auth_preflight_wins_before_reference_file_access() {
        let missing = PathBuf::from("/definitely/not/opened/reference.png");
        let client = mold_core::MoldClient::new("http://127.0.0.1:9");
        let error = prepare_authoring(
            minimax_h3::REF2VA_COMFY,
            minimax_h3::FAMILY,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &[ReferenceArg {
                kind: ReferenceKind::Image,
                path: missing,
            }],
            Some(&client),
        )
        .expect_err("a missing API key must fail before touching the path");
        assert!(error.to_string().contains("MOLD_API_KEY"));
        assert!(!error.to_string().contains("opened no-follow"));
    }
}
