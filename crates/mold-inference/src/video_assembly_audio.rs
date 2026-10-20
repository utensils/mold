//! File-backed audio for long video assembly. One PCM spool and one continuous
//! AAC encoder avoid inserting an encoder's delay at every clip boundary.
//!
//! This module never owns publication: callers supply private staging files.

use crate::video_assembly_input::{checked_reader, PacketCursor};
use anyhow::{anyhow, ensure, Context, Result};
use fdk_aac::enc::{
    AudioObjectType as FdkObject, BitRate, ChannelMode, Encoder, EncoderParams, Transport,
};
use mp4_rs::{
    AacConfig, AudioObjectType, Bytes, ChannelConfig, MediaConfig, MediaType, Mp4Config, Mp4Sample,
    Mp4Writer, SampleFreqIndex, TrackConfig, TrackType,
};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_AAC},
    formats::Packet,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl PcmFormat {
    fn validate(self) -> Result<()> {
        let _ = frequency_index(self.sample_rate)?;
        ensure!(
            matches!(self.channels, 1 | 2),
            "assembly audio requires mono or stereo PCM"
        );
        Ok(())
    }
}

fn frequency_index(rate: u32) -> Result<SampleFreqIndex> {
    Ok(match rate {
        96_000 => SampleFreqIndex::Freq96000,
        88_200 => SampleFreqIndex::Freq88200,
        64_000 => SampleFreqIndex::Freq64000,
        48_000 => SampleFreqIndex::Freq48000,
        44_100 => SampleFreqIndex::Freq44100,
        32_000 => SampleFreqIndex::Freq32000,
        24_000 => SampleFreqIndex::Freq24000,
        22_050 => SampleFreqIndex::Freq22050,
        16_000 => SampleFreqIndex::Freq16000,
        12_000 => SampleFreqIndex::Freq12000,
        11_025 => SampleFreqIndex::Freq11025,
        8_000 => SampleFreqIndex::Freq8000,
        7_350 => SampleFreqIndex::Freq7350,
        _ => anyhow::bail!("unsupported AAC assembly sample rate {rate}"),
    })
}

pub struct PcmSpool {
    file: File,
    format: PcmFormat,
    frames: u64,
    poisoned: bool,
}

impl PcmSpool {
    pub fn new(file: File, format: PcmFormat) -> Result<Self> {
        Self::resume(file, format, 0)
    }

    /// Reopen an owned PCM spool after the caller authenticates its exact hash,
    /// format and committed parent checkpoint. Length is a consistency check,
    /// never permission to adopt a discovered file or uncommitted partial write.
    pub fn resume(file: File, format: PcmFormat, expected_frames: u64) -> Result<Self> {
        format.validate()?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "PCM staging must be a regular file");
        let bytes = expected_frames
            .checked_mul(u64::from(format.channels) * 4)
            .context("PCM checkpoint length overflowed")?;
        ensure!(
            metadata.len() == bytes,
            "PCM staging length differs from its committed checkpoint"
        );
        Ok(Self {
            file,
            format,
            frames: expected_frames,
            poisoned: false,
        })
    }
    pub fn frames(&self) -> u64 {
        self.frames
    }
    pub fn format(&self) -> PcmFormat {
        self.format
    }

    /// Append exact PCM frames. Missing source frames become silence; excess
    /// source frames are never read. Crossfade alters only the new movie's spool.
    pub fn append_pcm(
        &mut self,
        input: &mut dyn Read,
        take_frames: u64,
        crossfade_frames: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(!self.poisoned, "PCM assembly already failed");
        let result = self.append_pcm_inner(input, take_frames, crossfade_frames, checkpoint);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn append_pcm_inner(
        &mut self,
        input: &mut dyn Read,
        take_frames: u64,
        crossfade_frames: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let overlap = u64::from(crossfade_frames);
        ensure!(
            take_frames > overlap && self.frames >= overlap,
            "audio crossfade exceeds the available timeline"
        );
        // A join is bounded independently of the complete movie duration.
        ensure!(
            crossfade_frames <= self.format.sample_rate * 10,
            "audio crossfade exceeds ten seconds"
        );
        let channels = usize::from(self.format.channels);
        let stride = channels * 4;
        let start = self.frames - overlap;
        let output_frames = start
            .checked_add(take_frames)
            .context("PCM timeline overflowed")?;
        output_frames
            .checked_mul(stride as u64)
            .context("PCM byte length overflowed")?;
        let byte_start = start
            .checked_mul(stride as u64)
            .context("PCM offset overflowed")?;
        self.file.seek(SeekFrom::Start(byte_start))?;
        let mut tail = vec![0u8; usize::try_from(overlap)? * stride];
        self.file.read_exact(&mut tail)?;
        self.file.seek(SeekFrom::Start(byte_start))?;
        let mut writer = BufWriter::with_capacity(64 * 1024, &mut self.file);
        let mut frame = [0u8; 8];
        let mut exhausted = false;
        for index in 0..take_frames {
            if index % 4096 == 0 {
                checkpoint()?;
            }
            frame.fill(0);
            if !exhausted {
                let mut read = 0;
                while read < stride {
                    let count = input.read(&mut frame[read..stride])?;
                    if count == 0 {
                        exhausted = true;
                        break;
                    }
                    read += count;
                }
                ensure!(read == 0 || read == stride, "truncated PCM frame");
            }
            for channel in 0..channels {
                let offset = channel * 4;
                let mut value = f32::from_le_bytes(frame[offset..offset + 4].try_into()?);
                ensure!(value.is_finite(), "PCM contains a nonfinite sample");
                if index < overlap {
                    let old_offset = usize::try_from(index)? * stride + offset;
                    let old = f32::from_le_bytes(tail[old_offset..old_offset + 4].try_into()?);
                    ensure!(
                        old.is_finite(),
                        "PCM checkpoint contains a nonfinite sample"
                    );
                    let weight = (index + 1) as f32 / (overlap + 1) as f32;
                    value = old * (1.0 - weight) + value * weight;
                }
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        writer.flush()?;
        self.frames = output_frames;
        Ok(())
    }

    /// Decode an authenticated, preflighted AAC MP4 into a bounded temporary PCM
    /// file, then conform exactly one segment. Decoding never accumulates PCM.
    /// Mono/stereo and supported sample rates are conformed with a bounded linear
    /// interpolator, matching the existing source-audio ingest policy.
    pub fn append_mp4(
        &mut self,
        input: File,
        scratch: File,
        skip_frames: u64,
        take_frames: u64,
        crossfade_frames: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(!self.poisoned, "PCM assembly already failed");
        let result = self.append_mp4_inner(
            input,
            scratch,
            skip_frames,
            take_frames,
            crossfade_frames,
            checkpoint,
        );
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn append_mp4_inner(
        &mut self,
        input: File,
        mut scratch: File,
        skip_frames: u64,
        take_frames: u64,
        crossfade_frames: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            scratch.metadata()?.len() == 0,
            "decoded PCM scratch file must be empty"
        );
        checkpoint()?;
        let mut source_format = self.format;
        let packet_file = input.try_clone()?;
        let reader = checked_reader(input, checkpoint)?;
        let audios: Vec<_> = reader
            .tracks()
            .iter()
            .filter(|(_, t)| matches!(t.track_type(), Ok(TrackType::Audio)))
            .map(|(id, _)| *id)
            .collect();
        ensure!(
            audios.len() <= 1,
            "multiple audio tracks require explicit selection"
        );
        if let Some(id) = audios.first().copied() {
            let track = &reader.tracks()[&id];
            ensure!(
                track.media_type()? == MediaType::AAC,
                "assembly source audio must be AAC"
            );
            let audio = track
                .trak
                .mdia
                .minf
                .stbl
                .stsd
                .mp4a
                .as_ref()
                .context("missing AAC configuration")?;
            source_format = PcmFormat {
                sample_rate: track.sample_freq_index()?.freq(),
                channels: audio.channelcount,
            };
            source_format.validate()?;
            ensure!(
                track.timescale() == source_format.sample_rate,
                "source audio timescale differs from its sample rate"
            );
            let asc = &audio
                .esds
                .as_ref()
                .context("missing AAC decoder configuration")?
                .es_desc
                .dec_config
                .dec_specific;
            ensure!(
                asc.profile == 2 && asc.freq_index < 13 && matches!(asc.chan_conf, 1 | 2),
                "unsupported AAC decoder configuration"
            );
            let mut parameters = CodecParameters::new();
            parameters
                .for_codec(CODEC_TYPE_AAC)
                .with_sample_rate(source_format.sample_rate)
                .with_extra_data(Box::from([
                    (asc.profile << 3) | (asc.freq_index >> 1),
                    (asc.freq_index << 7) | (asc.chan_conf << 3),
                ]));
            let mut decoder =
                symphonia::default::get_codecs().make(&parameters, &DecoderOptions::default())?;
            let edits = track.trak.edts.as_ref().and_then(|e| e.elst.as_ref());
            let edit_skip = if let Some(edits) = edits {
                ensure!(
                    edits.entries.len() == 1
                        && edits.entries[0].media_rate == 1
                        && edits.entries[0].media_rate_fraction == 0,
                    "source audio has unsupported presentation edits"
                );
                ensure!(
                    edits.entries[0].media_time <= track.trak.mdia.mdhd.duration,
                    "source audio has an empty or out-of-range presentation edit"
                );
                edits.entries[0].media_time
            } else {
                0
            };
            let mut skip = edit_skip;
            let requested_end = skip_frames
                .checked_add(take_frames)
                .context("audio trim overflowed")?;
            let mut remaining = u64::try_from(
                (u128::from(requested_end) * u128::from(source_format.sample_rate))
                    .div_ceil(u128::from(self.format.sample_rate)),
            )
            .context("audio resample count overflowed")?
            .checked_add(1)
            .context("audio interpolation count overflowed")?;
            if let Some(edits) = edits {
                let movie_scale = reader.moov.mvhd.timescale;
                ensure!(movie_scale > 0, "source movie has zero timescale");
                let presented = crate::av_media::aligned_sample_count(
                    edits.entries[0].segment_duration,
                    movie_scale,
                    source_format.sample_rate,
                )?;
                remaining = remaining.min(presented);
            }
            let count = track.sample_count();
            let mut cursor = PacketCursor::new(track, packet_file)?;
            let mut writer = BufWriter::with_capacity(64 * 1024, &mut scratch);
            for _ in 1..=count {
                if remaining == 0 {
                    break;
                }
                checkpoint()?;
                let sample = cursor.next()?.context("missing AAC packet")?;
                let packet = Packet::new_from_boxed_slice(
                    id,
                    sample.start_time,
                    u64::from(sample.duration),
                    sample.bytes.to_vec().into_boxed_slice(),
                );
                let decoded = decoder
                    .decode(&packet)
                    .context("decoding AAC assembly source")?;
                ensure!(
                    decoded.spec().rate == source_format.sample_rate
                        && decoded.spec().channels.count() == usize::from(source_format.channels),
                    "decoded audio format changed"
                );
                ensure!(
                    decoded.capacity() <= 8192,
                    "decoded audio packet exceeds its memory budget"
                );
                let mut samples =
                    SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                samples.copy_interleaved_ref(decoded);
                // A legacy final AAC packet can declare fewer presentation
                // samples than its decoded codec frame. Honor that timeline.
                ensure!(track.timescale() > 0, "source audio timescale is zero");
                let presented = crate::av_media::aligned_sample_count(
                    u64::from(sample.duration),
                    track.timescale(),
                    source_format.sample_rate,
                )?;
                for frame in samples
                    .samples()
                    .chunks_exact(usize::from(source_format.channels))
                    .take(usize::try_from(presented)?)
                {
                    if skip > 0 {
                        skip -= 1;
                        continue;
                    }
                    if remaining == 0 {
                        break;
                    }
                    for sample in frame {
                        ensure!(
                            sample.is_finite(),
                            "decoded audio contains a nonfinite sample"
                        );
                        writer.write_all(&sample.to_le_bytes())?;
                    }
                    remaining -= 1;
                }
            }
            writer.flush()?;
        }
        scratch.seek(SeekFrom::Start(0))?;
        let mut conformed =
            ConformedPcmReader::new(scratch, source_format, self.format, skip_frames)?;
        self.append_pcm_inner(&mut conformed, take_frames, crossfade_frames, checkpoint)
    }

    pub fn into_file(mut self) -> Result<File> {
        ensure!(!self.poisoned, "cannot encode failed PCM assembly");
        self.file.flush()?;
        self.file.seek(SeekFrom::Start(0))?;
        Ok(self.file)
    }
}

/// A two-frame resampling window; PCM storage and decoder state never grow with
/// duration. Integer source positions prevent accumulated fractional drift.
struct ConformedPcmReader {
    reader: BufReader<File>,
    source: PcmFormat,
    target: PcmFormat,
    source_frames: u64,
    source_index: u64,
    output_index: u64,
    output_frames: u64,
    left: [f32; 2],
    right: [f32; 2],
    pending: [u8; 8],
    pending_start: usize,
    pending_end: usize,
}

impl ConformedPcmReader {
    fn new(mut file: File, source: PcmFormat, target: PcmFormat, skip: u64) -> Result<Self> {
        let stride = u64::from(source.channels) * 4;
        ensure!(
            file.metadata()?.len().is_multiple_of(stride),
            "decoded PCM has a partial frame"
        );
        let source_frames = file.metadata()?.len() / stride;
        let output_frames = u64::try_from(
            (u128::from(source_frames) * u128::from(target.sample_rate)
                + u128::from(source.sample_rate / 2))
                / u128::from(source.sample_rate),
        )?;
        let source_index = u64::try_from(
            u128::from(skip) * u128::from(source.sample_rate) / u128::from(target.sample_rate),
        )?
        .min(source_frames.saturating_sub(1));
        file.seek(SeekFrom::Start(
            source_index
                .checked_mul(stride)
                .context("PCM seek overflowed")?,
        ))?;
        let mut value = Self {
            reader: BufReader::with_capacity(64 * 1024, file),
            source,
            target,
            source_frames,
            source_index,
            output_index: skip,
            output_frames,
            left: [0.0; 2],
            right: [0.0; 2],
            pending: [0; 8],
            pending_start: 0,
            pending_end: 0,
        };
        if source_frames > 0 {
            value.left = value.read_frame()?;
            value.right = if source_index + 1 < source_frames {
                value.read_frame()?
            } else {
                value.left
            };
        }
        Ok(value)
    }
    fn read_frame(&mut self) -> std::io::Result<[f32; 2]> {
        let mut output = [0.0; 2];
        for channel in &mut output[..usize::from(self.source.channels)] {
            let mut bytes = [0u8; 4];
            self.reader.read_exact(&mut bytes)?;
            *channel = f32::from_le_bytes(bytes);
        }
        if self.source.channels == 1 {
            output[1] = output[0];
        }
        Ok(output)
    }
    fn produce(&mut self) -> std::io::Result<bool> {
        if self.output_index >= self.output_frames {
            return Ok(false);
        }
        let numerator = u128::from(self.output_index) * u128::from(self.source.sample_rate);
        let desired = u64::try_from(numerator / u128::from(self.target.sample_rate))
            .map_err(std::io::Error::other)?
            .min(self.source_frames - 1);
        while self.source_index < desired {
            self.left = self.right;
            self.source_index += 1;
            self.right = if self.source_index + 1 < self.source_frames {
                self.read_frame()?
            } else {
                self.left
            };
        }
        let weight = (numerator % u128::from(self.target.sample_rate)) as f32
            / self.target.sample_rate as f32;
        let mut frame = [0.0; 2];
        for (channel, value) in frame.iter_mut().enumerate() {
            *value = self.left[channel] + (self.right[channel] - self.left[channel]) * weight;
        }
        if self.target.channels == 1 {
            frame[0] = (frame[0] + frame[1]) * 0.5;
        }
        for (channel, value) in frame[..usize::from(self.target.channels)]
            .iter()
            .enumerate()
        {
            self.pending[channel * 4..channel * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        self.pending_start = 0;
        self.pending_end = usize::from(self.target.channels) * 4;
        self.output_index = self
            .output_index
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("PCM resample timeline overflowed"))?;
        Ok(true)
    }
}

impl Read for ConformedPcmReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.pending_start == self.pending_end && !self.produce()? {
            return Ok(0);
        }
        let count = output.len().min(self.pending_end - self.pending_start);
        output[..count]
            .copy_from_slice(&self.pending[self.pending_start..self.pending_start + count]);
        self.pending_start += count;
        Ok(count)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioMuxReceipt {
    pub samples: u64,
    pub encoder_delay: u32,
    pub packets: u64,
}

/// Encode one continuous PCM stream; interleave it with original H.264 packets.
/// The final edit list removes the codec delay and padding once for the movie.
pub fn mux_pcm(
    video: File,
    mut pcm: File,
    output: File,
    format: PcmFormat,
    samples: u64,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<AudioMuxReceipt> {
    format.validate()?;
    ensure!(
        output.metadata()?.len() == 0,
        "audio mux staging file must be empty"
    );
    ensure!(samples > 0, "cannot encode an empty PCM timeline");
    ensure!(
        pcm.metadata()?.len()
            == samples
                .checked_mul(u64::from(format.channels) * 4)
                .context("PCM size overflowed")?,
        "PCM spool length does not match its timeline"
    );
    pcm.seek(SeekFrom::Start(0))?;
    let mut pcm = BufReader::with_capacity(64 * 1024, pcm);
    let packet_file = video.try_clone()?;
    let reader = checked_reader(video, checkpoint)?;
    let videos: Vec<_> = reader
        .tracks()
        .iter()
        .filter(|(_, t)| matches!(t.track_type(), Ok(TrackType::Video)))
        .map(|(id, _)| *id)
        .collect();
    ensure!(videos.len() == 1, "audio mux requires one video track");
    let video_id = videos[0];
    let track = &reader.tracks()[&video_id];
    let video_config = crate::video_assembly::track_config(track, reader.moov.mvhd.timescale)?;
    let video_count = track.sample_count();
    let raw_stsd = crate::video_assembly_boxes::read_track_box(
        &mut packet_file.try_clone()?,
        video_id,
        &[*b"mdia", *b"minf", *b"stbl", *b"stsd"],
        4 * 1024 * 1024,
    )?;
    let mut video_cursor = PacketCursor::new(track, packet_file)?;
    let video_scale = track.timescale();
    let movie_duration = track.trak.mdia.mdhd.duration;
    ensure!(
        samples
            == crate::av_media::aligned_sample_count(
                movie_duration,
                video_scale,
                format.sample_rate
            )?,
        "PCM timeline does not match the video"
    );
    let bitrate = if format.channels == 1 {
        96_000
    } else {
        192_000
    };
    let encoder = Encoder::new(EncoderParams {
        bit_rate: BitRate::Cbr(bitrate),
        sample_rate: format.sample_rate,
        transport: Transport::Raw,
        channels: if format.channels == 1 {
            ChannelMode::Mono
        } else {
            ChannelMode::Stereo
        },
        audio_object_type: FdkObject::Mpeg4LowComplexity,
    })
    .map_err(|e| anyhow!("creating assembly AAC encoder: {e}"))?;
    let info = encoder
        .info()
        .map_err(|e| anyhow!("reading AAC encoder parameters: {e}"))?;
    ensure!(
        info.inputChannels == u32::from(format.channels),
        "AAC encoder channel configuration changed"
    );
    ensure!(
        info.frameLength > 0 && info.frameLength <= 8192 && info.maxOutBufBytes <= 1024 * 1024,
        "AAC encoder staging exceeds budget"
    );
    let frame_length = u64::from(info.frameLength);
    let packet_count = samples
        .checked_add(u64::from(info.nDelay))
        .context("AAC timeline overflowed")?
        .div_ceil(frame_length);
    let config = Mp4Config {
        major_brand: "isom".parse()?,
        minor_version: 0x200,
        compatible_brands: vec![
            "isom".parse()?,
            "iso2".parse()?,
            "avc1".parse()?,
            "mp41".parse()?,
        ],
        timescale: video_scale,
    };
    let mut writer = Mp4Writer::write_start(output, &config)?;
    writer.add_track(&video_config)?;
    writer.add_track(&TrackConfig {
        track_type: TrackType::Audio,
        timescale: format.sample_rate,
        language: "und".into(),
        media_conf: MediaConfig::AacConfig(AacConfig {
            bitrate,
            profile: AudioObjectType::AacLowComplexity,
            freq_index: frequency_index(format.sample_rate)?,
            chan_conf: if format.channels == 1 {
                ChannelConfig::Mono
            } else {
                ChannelConfig::Stereo
            },
        }),
    })?;
    let mut frame = vec![0i16; info.frameLength as usize * usize::from(format.channels)];
    let mut encoded = vec![0u8; info.maxOutBufBytes as usize];
    let mut audio_index = 0u64;
    let mut input_index = 0u64;
    let mut video_next = 1u32;
    let mut pending_video = None;
    while video_next <= video_count || pending_video.is_some() || audio_index < packet_count {
        checkpoint()?;
        if pending_video.is_none() && video_next <= video_count {
            pending_video = video_cursor.next()?;
            ensure!(
                pending_video.is_some(),
                "video packet disappeared during AAC mux"
            );
            video_next = video_next
                .checked_add(1)
                .context("video packet count overflowed")?;
        }
        let audio_time = audio_index
            .checked_mul(frame_length)
            .context("AAC packet timeline overflowed")?;
        let audio_due = audio_index < packet_count
            && pending_video.as_ref().is_none_or(|sample: &Mp4Sample| {
                u128::from(audio_time.saturating_sub(u64::from(info.nDelay)))
                    * u128::from(video_scale)
                    <= u128::from(sample.start_time) * u128::from(format.sample_rate)
            });
        if audio_due {
            frame.fill(0);
            for channel_frame in frame.chunks_exact_mut(usize::from(format.channels)) {
                if input_index < samples {
                    for target in channel_frame {
                        let mut bytes = [0u8; 4];
                        pcm.read_exact(&mut bytes)?;
                        let value = f32::from_le_bytes(bytes);
                        ensure!(value.is_finite(), "PCM contains a nonfinite sample");
                        *target = (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
                    }
                    input_index = input_index
                        .checked_add(1)
                        .context("PCM input timeline overflowed")?;
                }
            }
            let packet = encoder
                .encode(&frame, &mut encoded)
                .map_err(|e| anyhow!("encoding assembly AAC: {e}"))?;
            ensure!(
                packet.input_consumed == frame.len() && packet.output_size > 0,
                "AAC encoder did not produce a complete packet"
            );
            writer.write_sample(
                2,
                &Mp4Sample {
                    start_time: audio_time,
                    duration: info.frameLength,
                    rendering_offset: 0,
                    is_sync: true,
                    bytes: Bytes::copy_from_slice(&encoded[..packet.output_size]),
                },
            )?;
            audio_index = audio_index
                .checked_add(1)
                .context("AAC packet count overflowed")?;
        } else if let Some(sample) = pending_video.take() {
            writer.write_sample(1, &sample)?;
        }
    }
    ensure!(
        input_index == samples,
        "AAC encoder did not consume the complete PCM timeline"
    );
    writer.write_end()?;
    let mut output = writer.into_writer();
    apply_audio_edit(&mut output, movie_duration, info.nDelay, checkpoint)?;
    crate::video_assembly_boxes::replace_track_box(
        &mut output,
        1,
        &[*b"mdia", *b"minf", *b"stbl", *b"stsd"],
        &raw_stsd,
    )?;
    output.sync_all()?;
    Ok(AudioMuxReceipt {
        samples,
        encoder_delay: info.nDelay,
        packets: packet_count,
    })
}

fn apply_audio_edit(
    output: &mut File,
    video_ticks: u64,
    delay: u32,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    checkpoint()?;
    let mut edits = Vec::with_capacity(44);
    edits.extend_from_slice(&44u32.to_be_bytes());
    edits.extend_from_slice(b"edts");
    edits.extend_from_slice(&36u32.to_be_bytes());
    edits.extend_from_slice(b"elst");
    edits.extend_from_slice(&0x0100_0000u32.to_be_bytes());
    edits.extend_from_slice(&1u32.to_be_bytes());
    edits.extend_from_slice(&video_ticks.to_be_bytes());
    edits.extend_from_slice(&u64::from(delay).to_be_bytes());
    edits.extend_from_slice(&1u16.to_be_bytes());
    edits.extend_from_slice(&0u16.to_be_bytes());
    crate::video_assembly_boxes::replace_track_box(output, 2, &[*b"edts"], &edits)?;
    crate::video_assembly_boxes::set_durations(output, 2, video_ticks)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pcm_bytes(frames: usize, channels: usize, value: f32) -> Vec<u8> {
        (0..frames * channels)
            .flat_map(|_| value.to_le_bytes())
            .collect()
    }
    #[test]
    fn exact_pcm_joins_crossfade_without_growing_memory() {
        for rate in [16_000, 24_000, 32_000, 44_100, 48_000] {
            for channels in [1, 2] {
                let format = PcmFormat {
                    sample_rate: rate,
                    channels,
                };
                let mut spool = PcmSpool::new(tempfile::tempfile().unwrap(), format).unwrap();
                spool
                    .append_pcm(
                        &mut &pcm_bytes(1103, channels as usize, 0.5)[..],
                        1103,
                        0,
                        &mut || Ok(()),
                    )
                    .unwrap();
                spool
                    .append_pcm(
                        &mut &pcm_bytes(991, channels as usize, -0.5)[..],
                        991,
                        100,
                        &mut || Ok(()),
                    )
                    .unwrap();
                assert_eq!(spool.frames(), 1994);
                let mut file = spool.into_file().unwrap();
                assert_eq!(
                    file.metadata().unwrap().len(),
                    1994 * u64::from(channels) * 4
                );
                file.seek(SeekFrom::Start(1003 * u64::from(channels) * 4))
                    .unwrap();
                let mut value = [0u8; 4];
                file.read_exact(&mut value).unwrap();
                assert!((f32::from_le_bytes(value) - (0.5 - 1.0 / 101.0)).abs() < 1e-6);
            }
        }
    }
    #[test]
    fn cancelled_spool_cannot_be_encoded() {
        let mut spool = PcmSpool::new(
            tempfile::tempfile().unwrap(),
            PcmFormat {
                sample_rate: 48_000,
                channels: 2,
            },
        )
        .unwrap();
        assert!(spool
            .append_pcm(&mut std::io::empty(), 1000, 0, &mut || anyhow::bail!(
                "cancelled"
            ))
            .is_err());
        assert!(spool.into_file().is_err());
    }

    #[test]
    fn resume_requires_the_exact_committed_pcm_length() {
        let format = PcmFormat {
            sample_rate: 32_000,
            channels: 2,
        };
        let mut spool = PcmSpool::new(tempfile::tempfile().unwrap(), format).unwrap();
        spool
            .append_pcm(&mut &pcm_bytes(100, 2, 0.1)[..], 100, 0, &mut || Ok(()))
            .unwrap();
        let file = spool.into_file().unwrap();
        assert!(PcmSpool::resume(file.try_clone().unwrap(), format, 99).is_err());
        let mut resumed = PcmSpool::resume(file, format, 100).unwrap();
        resumed
            .append_pcm(&mut std::io::empty(), 17, 0, &mut || Ok(()))
            .unwrap();
        assert_eq!(resumed.frames(), 117);
        assert_eq!(
            resumed.into_file().unwrap().metadata().unwrap().len(),
            117 * 2 * 4
        );
    }

    #[test]
    fn rejects_unrepresentable_pcm_before_reading_or_writing() {
        let file = tempfile::tempfile().unwrap();
        let inspected = file.try_clone().unwrap();
        let mut spool = PcmSpool::new(
            file,
            PcmFormat {
                sample_rate: 48_000,
                channels: 2,
            },
        )
        .unwrap();
        let mut checkpoints = 0;
        let result = spool.append_pcm(&mut std::io::empty(), u64::MAX, 0, &mut || {
            checkpoints += 1;
            anyhow::bail!("should reject before starting PCM copy")
        });
        assert!(result.is_err());
        assert_eq!(checkpoints, 0);
        assert_eq!(inspected.metadata().unwrap().len(), 0);
    }

    #[test]
    fn rate_and_channel_conversion_uses_exact_rational_positions() {
        for source_rate in [16_000, 24_000, 32_000, 44_100, 48_000] {
            for target_rate in [16_000, 24_000, 32_000, 44_100, 48_000] {
                for source_channels in [1, 2] {
                    for target_channels in [1, 2] {
                        let mut source = tempfile::tempfile().unwrap();
                        for index in 0..3001 {
                            for channel in 0..source_channels {
                                source
                                    .write_all(
                                        &(0.1 + index as f32 * 0.00001 + channel as f32 * 0.02)
                                            .to_le_bytes(),
                                    )
                                    .unwrap();
                            }
                        }
                        let mut reader = ConformedPcmReader::new(
                            source,
                            PcmFormat {
                                sample_rate: source_rate,
                                channels: source_channels,
                            },
                            PcmFormat {
                                sample_rate: target_rate,
                                channels: target_channels,
                            },
                            17,
                        )
                        .unwrap();
                        let mut bytes = Vec::new();
                        reader.read_to_end(&mut bytes).unwrap();
                        let expected_count = (3001 * u64::from(target_rate)
                            + u64::from(source_rate / 2))
                            / u64::from(source_rate)
                            - 17;
                        assert_eq!(
                            bytes.len() as u64,
                            expected_count * u64::from(target_channels) * 4
                        );
                        let expected_first = 0.1
                            + 17.0 * source_rate as f32 / target_rate as f32 * 0.00001
                            + if source_channels == 2 && target_channels == 1 {
                                0.01
                            } else {
                                0.0
                            };
                        let actual = f32::from_le_bytes(bytes[..4].try_into().unwrap());
                        assert!((actual - expected_first).abs() < 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn continuous_aac_preserves_non_packet_aligned_tone_seams_and_exact_edit_timeline() {
        use image::{Rgb, RgbImage};
        for rate in [16_000, 24_000, 32_000, 44_100, 48_000] {
            for channels in [1, 2] {
                let frames = vec![RgbImage::from_pixel(32, 32, Rgb([20, 40, 60])); 27];
                let video = crate::ltx_video::video_enc::encode_mp4(&frames, 24).unwrap();
                let mut video_file = tempfile::tempfile().unwrap();
                video_file.write_all(&video).unwrap();
                video_file.seek(SeekFrom::Start(0)).unwrap();
                let format = PcmFormat {
                    sample_rate: rate,
                    channels,
                };
                let count = crate::av_media::aligned_sample_count(27, 24, rate).unwrap();
                let mut spool = PcmSpool::new(tempfile::tempfile().unwrap(), format).unwrap();
                let mut previous = 0;
                for end_video_frame in [7, 18, 27] {
                    let end =
                        crate::av_media::aligned_sample_count(end_video_frame, 24, rate).unwrap();
                    let bytes: Vec<_> = (previous..end)
                        .flat_map(|index| {
                            let value = (std::f64::consts::TAU * 997.0 * index as f64
                                / f64::from(rate))
                            .sin() as f32
                                * 0.2;
                            (0..channels).flat_map(move |_| value.to_le_bytes())
                        })
                        .collect();
                    spool
                        .append_pcm(&mut &bytes[..], end - previous, 0, &mut || Ok(()))
                        .unwrap();
                    previous = end;
                }
                assert_eq!(spool.frames(), count);
                let mut output = tempfile::tempfile().unwrap();
                let report = mux_pcm(
                    video_file,
                    spool.into_file().unwrap(),
                    output.try_clone().unwrap(),
                    format,
                    count,
                    &mut || Ok(()),
                )
                .unwrap();
                assert_eq!(report.samples, count);

                output.seek(SeekFrom::Start(0)).unwrap();
                let reader = checked_reader(output.try_clone().unwrap(), &mut || Ok(())).unwrap();
                let edit = &reader.tracks()[&2]
                    .trak
                    .edts
                    .as_ref()
                    .unwrap()
                    .elst
                    .as_ref()
                    .unwrap()
                    .entries[0];
                assert_eq!(edit.media_time, u64::from(report.encoder_delay));
                assert_eq!(edit.segment_duration, 27_000);
                assert_eq!(reader.moov.mvhd.duration, 27_000);
                output.seek(SeekFrom::Start(0)).unwrap();
                let mut decoded = PcmSpool::new(tempfile::tempfile().unwrap(), format).unwrap();
                decoded
                    .append_mp4(
                        output,
                        tempfile::tempfile().unwrap(),
                        0,
                        count,
                        0,
                        &mut || Ok(()),
                    )
                    .unwrap();
                let mut decoded = decoded.into_file().unwrap();
                for center in [
                    crate::av_media::aligned_sample_count(7, 24, rate).unwrap(),
                    crate::av_media::aligned_sample_count(18, 24, rate).unwrap(),
                    count - 64,
                ] {
                    decoded
                        .seek(SeekFrom::Start((center - 64) * u64::from(channels) * 4))
                        .unwrap();
                    let mut error = 0.0;
                    let mut energy = 0.0;
                    for index in center - 64..center + 64 {
                        let expected =
                            (std::f64::consts::TAU * 997.0 * index as f64 / f64::from(rate)).sin()
                                * 0.2;
                        for _ in 0..channels {
                            let mut bytes = [0u8; 4];
                            decoded.read_exact(&mut bytes).unwrap();
                            let value = f64::from(f32::from_le_bytes(bytes));
                            energy += value * value;
                            error += (value - expected).powi(2);
                        }
                    }
                    let denominator = 128.0 * f64::from(channels);
                    assert!(
                        (energy / denominator).sqrt() > 0.05,
                        "silent seam at {rate} Hz"
                    );
                    // An abruptly terminated tone has a normal AAC edge
                    // transient. FFmpeg independently decodes the same 16 kHz
                    // endpoint with 0.05166 RMS error; the clip joins remain
                    // below 0.002. Keep the seam check separate from that edge.
                    let tolerance = if center == count - 64 { 0.08 } else { 0.015 };
                    assert!(
                        (error / denominator).sqrt() < tolerance,
                        "tone seam shifted at {rate} Hz, {channels} channels, center {center}, delay {}: {}",
                        report.encoder_delay,
                        (error / denominator).sqrt()
                    );
                }
            }
        }
    }
}
