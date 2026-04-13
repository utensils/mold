#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{group_norm, ops, Conv2d, Conv2dConfig, GroupNorm, Module, VarBuilder};
use mp4_rs::{MediaType, Mp4Reader, TrackType};
use rustfft::{num_complex::Complex32, FftPlanner};
use serde::Deserialize;
use serde_json::Value;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{
    CodecParameters, Decoder as SymphoniaDecoder, DecoderOptions, CODEC_TYPE_AAC,
};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::Packet;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::TimeBase;
use symphonia::default::{get_codecs, get_probe};

use super::video_vae::PerChannelRmsNorm;
use super::{AudioLatentShape, AudioPatchifier};

const LATENT_DOWNSAMPLE_FACTOR: usize = 4;

fn silu(x: &Tensor) -> Result<Tensor> {
    ops::silu(x).map_err(Into::into)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioNormType {
    Group,
    Pixel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioCausalityAxis {
    None,
    Width,
    WidthCompatibility,
    Height,
}

#[derive(Debug, Clone)]
pub struct Ltx2AudioDecoderConfig {
    pub ch: usize,
    pub out_ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_resolutions: Vec<usize>,
    pub resolution: usize,
    pub z_channels: usize,
    pub norm_type: AudioNormType,
    pub causality_axis: AudioCausalityAxis,
    pub dropout: f64,
    pub mid_block_add_attention: bool,
    pub sample_rate: usize,
    pub mel_hop_length: usize,
    pub is_causal: bool,
    pub mel_bins: usize,
}

#[derive(Debug, Clone)]
pub struct Ltx2AudioEncoderConfig {
    pub ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_resolutions: Vec<usize>,
    pub resolution: usize,
    pub z_channels: usize,
    pub double_z: bool,
    pub norm_type: AudioNormType,
    pub causality_axis: AudioCausalityAxis,
    pub dropout: f64,
    pub mid_block_add_attention: bool,
    pub resamp_with_conv: bool,
    pub in_channels: usize,
    pub sample_rate: usize,
    pub mel_hop_length: usize,
    pub n_fft: usize,
    pub is_causal: bool,
    pub mel_bins: usize,
}

impl Ltx2AudioDecoderConfig {
    pub fn load(checkpoint_path: &Path) -> Result<Self> {
        let config_json = read_checkpoint_config_json(checkpoint_path)?;
        let checkpoint: CheckpointConfig =
            serde_json::from_str(&config_json).with_context(|| {
                format!(
                    "failed to parse LTX-2 checkpoint config metadata from {}",
                    checkpoint_path.display()
                )
            })?;
        let ddconfig = checkpoint.audio_vae.model.params.ddconfig;
        let preprocessing = checkpoint.audio_vae.preprocessing;
        Ok(Self {
            ch: ddconfig.ch,
            out_ch: ddconfig.out_ch,
            ch_mult: ddconfig.ch_mult,
            num_res_blocks: ddconfig.num_res_blocks,
            attn_resolutions: ddconfig.attn_resolutions,
            resolution: ddconfig.resolution,
            z_channels: ddconfig.z_channels,
            norm_type: ddconfig.norm_type,
            causality_axis: ddconfig.causality_axis,
            dropout: ddconfig.dropout,
            mid_block_add_attention: ddconfig.mid_block_add_attention,
            sample_rate: checkpoint.audio_vae.model.params.sampling_rate,
            mel_hop_length: preprocessing.stft.hop_length,
            is_causal: preprocessing.stft.causal,
            mel_bins: preprocessing.mel.n_mel_channels,
        })
    }
}

impl Ltx2AudioEncoderConfig {
    pub fn load(checkpoint_path: &Path) -> Result<Self> {
        let config_json = read_checkpoint_config_json(checkpoint_path)?;
        let checkpoint: CheckpointConfig =
            serde_json::from_str(&config_json).with_context(|| {
                format!(
                    "failed to parse LTX-2 checkpoint config metadata from {}",
                    checkpoint_path.display()
                )
            })?;
        let ddconfig = checkpoint.audio_vae.model.params.ddconfig;
        let preprocessing = checkpoint.audio_vae.preprocessing;
        Ok(Self {
            ch: ddconfig.ch,
            ch_mult: ddconfig.ch_mult,
            num_res_blocks: ddconfig.num_res_blocks,
            attn_resolutions: ddconfig.attn_resolutions,
            resolution: ddconfig.resolution,
            z_channels: ddconfig.z_channels,
            double_z: ddconfig.double_z.unwrap_or(true),
            norm_type: ddconfig.norm_type,
            causality_axis: ddconfig.causality_axis,
            dropout: ddconfig.dropout,
            mid_block_add_attention: ddconfig.mid_block_add_attention,
            resamp_with_conv: ddconfig.resamp_with_conv.unwrap_or(true),
            in_channels: ddconfig.in_channels,
            sample_rate: checkpoint.audio_vae.model.params.sampling_rate,
            mel_hop_length: preprocessing.stft.hop_length,
            n_fft: preprocessing.stft.filter_length.unwrap_or(1024),
            is_causal: preprocessing.stft.causal,
            mel_bins: preprocessing.mel.n_mel_channels,
        })
    }
}

#[derive(Debug, Deserialize)]
struct CheckpointConfig {
    audio_vae: CheckpointAudioVae,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioVae {
    model: CheckpointAudioVaeModel,
    preprocessing: CheckpointAudioPreprocessing,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioVaeModel {
    params: CheckpointAudioVaeParams,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioVaeParams {
    ddconfig: CheckpointAudioDdConfig,
    sampling_rate: usize,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioDdConfig {
    mel_bins: usize,
    z_channels: usize,
    resolution: usize,
    in_channels: usize,
    out_ch: usize,
    ch: usize,
    ch_mult: Vec<usize>,
    num_res_blocks: usize,
    attn_resolutions: Vec<usize>,
    double_z: Option<bool>,
    dropout: f64,
    resamp_with_conv: Option<bool>,
    mid_block_add_attention: bool,
    norm_type: AudioNormType,
    causality_axis: AudioCausalityAxis,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioPreprocessing {
    stft: CheckpointStftConfig,
    mel: CheckpointMelConfig,
}

#[derive(Debug, Deserialize)]
struct CheckpointStftConfig {
    hop_length: usize,
    causal: bool,
    filter_length: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CheckpointMelConfig {
    n_mel_channels: usize,
}

pub(crate) fn read_checkpoint_config_json(checkpoint_path: &Path) -> Result<String> {
    let mut file = File::open(checkpoint_path)
        .with_context(|| format!("failed to open {}", checkpoint_path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).with_context(|| {
        format!(
            "failed to read safetensors header length from {}",
            checkpoint_path.display()
        )
    })?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf).with_context(|| {
        format!(
            "failed to read safetensors header bytes from {}",
            checkpoint_path.display()
        )
    })?;
    let header: HashMap<String, Value> =
        serde_json::from_slice(&header_buf).with_context(|| {
            format!(
                "failed to parse safetensors header JSON from {}",
                checkpoint_path.display()
            )
        })?;
    let metadata = header
        .get("__metadata__")
        .and_then(Value::as_object)
        .context("safetensors metadata did not contain '__metadata__'")?;
    metadata
        .get("config")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .context("safetensors metadata did not contain a 'config' entry")
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    pub sample_rate: usize,
    pub channels: Vec<Vec<f32>>,
}

impl DecodedAudio {
    pub fn from_file(path: &Path, max_duration_seconds: Option<f32>) -> Result<Option<Self>> {
        let decoded = if is_mp4_audio_container(path) {
            match Self::decode_with_probe(path) {
                Ok(Some(decoded)) => Some(decoded),
                Ok(None) => Self::decode_aac_from_mp4(path)?,
                Err(probe_err) => match Self::decode_aac_from_mp4(path) {
                    Ok(decoded) => decoded,
                    Err(fallback_err) => {
                        return Err(anyhow!(
                            "failed to decode source audio '{}' via probe ({probe_err:#}) or native MP4 fallback ({fallback_err:#})",
                            path.display()
                        ));
                    }
                },
            }
        } else {
            Self::decode_with_probe(path)?
        };
        Ok(decoded.map(|decoded| decoded.trimmed(max_duration_seconds)))
    }

    fn decode_with_probe(path: &Path) -> Result<Option<Self>> {
        let file = File::open(path)
            .with_context(|| format!("failed to open source audio '{}'", path.display()))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            hint.with_extension(extension);
        }
        let probed = get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .with_context(|| format!("failed to probe source audio '{}'", path.display()))?;
        let mut format = probed.format;
        let track = match format
            .tracks()
            .iter()
            .find(|track| track.codec_params.sample_rate.is_some())
        {
            Some(track) => track.clone(),
            None => return Ok(None),
        };
        let track_id = track.id;
        let mut decoder = get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .with_context(|| {
                format!(
                    "failed to create decoder for source audio '{}'",
                    path.display()
                )
            })?;

        let mut accumulator = DecodedAudioAccumulator::new(
            track.codec_params.sample_rate.unwrap_or(16_000) as usize,
            track
                .codec_params
                .channels
                .map(|channels| channels.count())
                .unwrap_or(2),
        );

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(_)) => break,
                Err(err) => return Err(err.into()),
            };
            if packet.track_id() != track_id {
                continue;
            }
            accumulator.push_packet(path, decoder.as_mut(), &packet)?;
        }

        Ok(accumulator.finish())
    }

    fn decode_aac_from_mp4(path: &Path) -> Result<Option<Self>> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read source MP4 audio '{}'", path.display()))?;
        let mut reader = Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64)
            .with_context(|| format!("failed to parse MP4 container from '{}'", path.display()))?;
        let (track_id, track) = match reader
            .tracks()
            .iter()
            .find(|(_, track)| matches!(track.track_type(), Ok(TrackType::Audio)))
        {
            Some((track_id, track)) => (*track_id, track),
            None => return Ok(None),
        };
        if track.media_type()? != MediaType::AAC {
            bail!(
                "unsupported source audio codec in '{}': expected AAC in MP4 container, found {}",
                path.display(),
                track.media_type()?
            );
        }
        let mp4a = track
            .trak
            .mdia
            .minf
            .stbl
            .stsd
            .mp4a
            .as_ref()
            .context("MP4 audio track did not contain an mp4a sample entry")?;
        let esds = mp4a
            .esds
            .as_ref()
            .context("MP4 audio track did not contain an esds descriptor")?;
        let dec_specific = &esds.es_desc.dec_config.dec_specific;
        let mut codec_params = CodecParameters::new();
        codec_params
            .for_codec(CODEC_TYPE_AAC)
            .with_time_base(TimeBase::new(1, track.timescale()))
            .with_sample_rate(track.sample_freq_index()?.freq())
            .with_extra_data(build_aac_audio_specific_config(
                dec_specific.profile,
                dec_specific.freq_index,
                dec_specific.chan_conf,
            )?);
        let mut decoder = get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .with_context(|| {
                format!(
                    "failed to create AAC decoder for source audio '{}'",
                    path.display()
                )
            })?;
        let mut accumulator = DecodedAudioAccumulator::new(
            track.sample_freq_index()?.freq() as usize,
            mp4a.channelcount as usize,
        );

        for sample_id in 1..=track.sample_count() {
            let Some(sample) = reader.read_sample(track_id, sample_id)? else {
                continue;
            };
            let packet = Packet::new_from_boxed_slice(
                track_id,
                sample.start_time,
                u64::from(sample.duration),
                sample.bytes.to_vec().into_boxed_slice(),
            );
            accumulator.push_packet(path, decoder.as_mut(), &packet)?;
        }

        Ok(accumulator.finish())
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn sample_count(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    pub fn trimmed(&self, max_duration_seconds: Option<f32>) -> Self {
        let Some(max_duration_seconds) = max_duration_seconds else {
            return self.clone();
        };
        let max_samples =
            (max_duration_seconds.max(0.0) * self.sample_rate as f32).round() as usize;
        let channels = self
            .channels
            .iter()
            .map(|channel| channel[..channel.len().min(max_samples)].to_vec())
            .collect();
        Self {
            sample_rate: self.sample_rate,
            channels,
        }
    }

    pub fn to_tensor(
        &self,
        target_sample_rate: usize,
        target_channels: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let channels = conform_audio_channels(&self.channels, target_channels);
        let channels = if self.sample_rate == target_sample_rate {
            channels
        } else {
            resample_audio_channels_linear(&channels, self.sample_rate, target_sample_rate)
        };
        let sample_count = channels.first().map_or(0, Vec::len);
        let mut flat = Vec::with_capacity(target_channels * sample_count);
        for channel in &channels {
            flat.extend_from_slice(channel);
        }
        Tensor::from_vec(flat, (1, target_channels, sample_count), device).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct DecodedAudioAccumulator {
    sample_rate: usize,
    planar: Vec<Vec<f32>>,
}

impl DecodedAudioAccumulator {
    fn new(sample_rate: usize, channels: usize) -> Self {
        Self {
            sample_rate,
            planar: vec![Vec::new(); channels.max(1)],
        }
    }

    fn push_packet(
        &mut self,
        path: &Path,
        decoder: &mut dyn SymphoniaDecoder,
        packet: &Packet,
    ) -> Result<()> {
        let decoded = match decoder.decode(packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => return Ok(()),
            Err(SymphoniaError::IoError(_)) => return Ok(()),
            Err(SymphoniaError::ResetRequired) => {
                bail!(
                    "source audio decoder requested a reset while decoding '{}'",
                    path.display()
                );
            }
            Err(err) => return Err(err.into()),
        };

        self.sample_rate = decoded.spec().rate as usize;
        let channels = decoded.spec().channels.count();
        if self.planar.len() != channels {
            if self.planar.iter().all(Vec::is_empty) {
                self.planar = vec![Vec::new(); channels];
            } else {
                bail!(
                    "source audio '{}' changed channel count mid-stream from {} to {}",
                    path.display(),
                    self.planar.len(),
                    channels
                );
            }
        }

        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buf.copy_interleaved_ref(decoded);
        let frames = sample_buf.samples().len() / channels;
        let interleaved = sample_buf.samples();
        for frame_idx in 0..frames {
            let offset = frame_idx * channels;
            for channel_idx in 0..channels {
                self.planar[channel_idx].push(interleaved[offset + channel_idx]);
            }
        }
        Ok(())
    }

    fn finish(self) -> Option<DecodedAudio> {
        if self.planar.first().is_none_or(Vec::is_empty) {
            None
        } else {
            Some(DecodedAudio {
                sample_rate: self.sample_rate,
                channels: self.planar,
            })
        }
    }
}

fn is_mp4_audio_container(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .is_some_and(|ext| matches!(ext.as_str(), "mp4" | "m4a" | "mov"))
}

fn build_aac_audio_specific_config(
    profile: u8,
    freq_index: u8,
    chan_conf: u8,
) -> Result<Box<[u8]>> {
    if profile > 31 {
        bail!("unsupported AAC object type profile {profile} in MP4 fallback");
    }
    if freq_index > 15 {
        bail!("unsupported AAC sample frequency index {freq_index} in MP4 fallback");
    }
    if chan_conf > 15 {
        bail!("unsupported AAC channel config {chan_conf} in MP4 fallback");
    }
    Ok(Box::from([
        (profile << 3) | (freq_index >> 1),
        (freq_index << 7) | (chan_conf << 3),
    ]))
}

fn conform_audio_channels(channels: &[Vec<f32>], target_channels: usize) -> Vec<Vec<f32>> {
    if target_channels == 0 {
        return Vec::new();
    }
    if channels.is_empty() {
        return vec![Vec::new(); target_channels];
    }
    if channels.len() == target_channels {
        return channels.to_vec();
    }
    if channels.len() == 1 {
        return vec![channels[0].clone(); target_channels];
    }
    let sample_count = channels[0].len();
    let mut conformed = Vec::with_capacity(target_channels);
    for channel_idx in 0..target_channels {
        if let Some(channel) = channels.get(channel_idx) {
            conformed.push(channel.clone());
        } else {
            conformed.push(vec![0.0; sample_count]);
        }
    }
    conformed
}

fn resample_audio_channels_linear(
    channels: &[Vec<f32>],
    src_rate: usize,
    dst_rate: usize,
) -> Vec<Vec<f32>> {
    if src_rate == dst_rate {
        return channels.to_vec();
    }
    let src_len = channels.first().map_or(0, Vec::len);
    if src_len == 0 {
        return vec![Vec::new(); channels.len()];
    }
    let dst_len = (((src_len as f64) * (dst_rate as f64)) / (src_rate as f64))
        .round()
        .max(1.0) as usize;
    channels
        .iter()
        .map(|channel| {
            if channel.len() == 1 {
                return vec![channel[0]; dst_len];
            }
            let mut out = Vec::with_capacity(dst_len);
            for dst_idx in 0..dst_len {
                let src_pos = (dst_idx as f64) * (src_rate as f64) / (dst_rate as f64);
                let left = src_pos.floor() as usize;
                let right = (left + 1).min(channel.len() - 1);
                let frac = (src_pos - left as f64) as f32;
                let left_sample = channel[left];
                let right_sample = channel[right];
                out.push(left_sample + (right_sample - left_sample) * frac);
            }
            out
        })
        .collect()
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let max = (len - 1) as isize;
    while index < 0 || index > max {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * max - index;
        }
    }
    index as usize
}

fn reflect_pad_1d(samples: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 {
        return samples.to_vec();
    }
    let mut out = Vec::with_capacity(samples.len() + pad * 2);
    for idx in 0..pad {
        let reflected = reflect_index(idx as isize - pad as isize, samples.len());
        out.push(samples[reflected]);
    }
    out.extend_from_slice(samples);
    for idx in 0..pad {
        let reflected = reflect_index(samples.len() as isize + idx as isize, samples.len());
        out.push(samples[reflected]);
    }
    out
}

fn build_hann_window(win_length: usize) -> Vec<f32> {
    (0..win_length)
        .map(|idx| {
            let phase = 2.0 * std::f64::consts::PI * idx as f64 / win_length as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

fn hz_to_mel_slaney(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = (6.4f64).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / log_step
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = (6.4f64).ln() / 27.0;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (log_step * (mel - min_log_mel)).exp()
    }
}

fn build_slaney_mel_filterbank(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(sample_rate as f64 / 2.0);
    let mel_points = (0..(n_mels + 2))
        .map(|idx| {
            let ratio = idx as f64 / (n_mels + 1) as f64;
            mel_to_hz_slaney(mel_min + (mel_max - mel_min) * ratio)
        })
        .collect::<Vec<_>>();
    let fft_freqs = (0..n_freqs)
        .map(|idx| sample_rate as f64 * idx as f64 / n_fft as f64)
        .collect::<Vec<_>>();
    let mut filters = vec![vec![0.0f32; n_freqs]; n_mels];
    for mel_idx in 0..n_mels {
        let lower = mel_points[mel_idx];
        let center = mel_points[mel_idx + 1];
        let upper = mel_points[mel_idx + 2];
        let enorm = if upper > lower {
            2.0 / (upper - lower)
        } else {
            0.0
        } as f32;
        for (freq_idx, freq) in fft_freqs.iter().copied().enumerate() {
            let weight = if freq >= lower && freq <= center && center > lower {
                (freq - lower) / (center - lower)
            } else if freq >= center && freq <= upper && upper > center {
                (upper - freq) / (upper - center)
            } else {
                0.0
            };
            filters[mel_idx][freq_idx] = (weight as f32) * enorm;
        }
    }
    filters
}

fn waveform_to_log_mel(
    waveform: &Tensor,
    sample_rate: usize,
    n_mels: usize,
    hop_length: usize,
    n_fft: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let waveform = waveform
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let (batch, channels, samples) = waveform.dims3()?;
    if batch != 1 {
        bail!("native LTX-2 audio frontend currently expects batch=1, got {batch}");
    }
    let planar = waveform.i(0)?.to_vec2::<f32>()?;
    let pad = n_fft / 2;
    let window = build_hann_window(n_fft);
    let mel_filters = build_slaney_mel_filterbank(sample_rate, n_fft, n_mels);
    let mut fft_planner = FftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(n_fft);
    let mut flat = Vec::new();
    let mut expected_frames = None;

    for channel in planar.iter().take(channels) {
        let padded = reflect_pad_1d(channel, pad);
        if padded.len() < n_fft {
            bail!("source audio is too short to build a mel spectrogram");
        }
        let frame_count = 1 + (padded.len() - n_fft) / hop_length;
        if let Some(expected_frames) = expected_frames {
            if expected_frames != frame_count {
                bail!(
                    "native LTX-2 audio frontend produced inconsistent frame counts across channels"
                );
            }
        } else {
            expected_frames = Some(frame_count);
        }
        for frame_idx in 0..frame_count {
            let offset = frame_idx * hop_length;
            let mut spectrum = vec![Complex32::new(0.0, 0.0); n_fft];
            for (fft_idx, value) in padded[offset..(offset + n_fft)].iter().copied().enumerate() {
                spectrum[fft_idx].re = value * window[fft_idx];
            }
            fft.process(&mut spectrum);
            let magnitudes = spectrum[..(n_fft / 2 + 1)]
                .iter()
                .map(|bin| (bin.re * bin.re + bin.im * bin.im).sqrt())
                .collect::<Vec<_>>();
            for mel_filter in &mel_filters {
                let mel = mel_filter
                    .iter()
                    .zip(magnitudes.iter())
                    .map(|(weight, magnitude)| weight * magnitude)
                    .sum::<f32>()
                    .max(1e-5)
                    .ln();
                flat.push(mel);
            }
        }
    }

    let frames = expected_frames.unwrap_or_else(|| {
        let _ = samples;
        0
    });
    Tensor::from_vec(flat, (1, channels, frames, n_mels), device).map_err(Into::into)
}

#[derive(Debug, Clone)]
struct AudioPerChannelStatistics {
    mean: Tensor,
    std: Tensor,
}

impl AudioPerChannelStatistics {
    fn load(audio_vb: VarBuilder, features: usize) -> Result<Self> {
        let stats_vb = audio_vb.pp("per_channel_statistics");
        let mean = stats_vb
            .get(features, "mean-of-means")
            .context("failed to load audio per-channel mean statistics")?;
        let std = stats_vb
            .get(features, "std-of-means")
            .context("failed to load audio per-channel std statistics")?;
        Ok(Self { mean, std })
    }

    fn denormalize(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, features) = x.dims3()?;
        let mean = self
            .mean
            .reshape((1, 1, features))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        let std = self
            .std
            .reshape((1, 1, features))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        x.broadcast_mul(&std)?
            .broadcast_add(&mean)
            .map_err(Into::into)
    }

    fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, features) = x.dims3()?;
        let mean = self
            .mean
            .reshape((1, 1, features))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        let std = self
            .std
            .reshape((1, 1, features))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        x.broadcast_sub(&mean)?
            .broadcast_div(&std)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
enum AudioNorm {
    Group(GroupNorm),
    Pixel(PerChannelRmsNorm),
}

impl AudioNorm {
    fn load(channels: usize, norm_type: AudioNormType, vb: VarBuilder) -> Result<Self> {
        Ok(match norm_type {
            AudioNormType::Group => Self::Group(group_norm(32, channels, 1e-6, vb)?),
            AudioNormType::Pixel => Self::Pixel(PerChannelRmsNorm::new(1, 1e-6)),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Group(norm) => norm.forward(x).map_err(Into::into),
            Self::Pixel(norm) => norm.forward(x).map_err(Into::into),
        }
    }
}

#[derive(Debug, Clone)]
struct AudioCausalConv2d {
    conv: Conv2d,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

impl AudioCausalConv2d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad = kernel_size.saturating_sub(1);
        let (pad_left, pad_right, pad_top, pad_bottom) = match causality_axis {
            AudioCausalityAxis::None => (pad / 2, pad - (pad / 2), pad / 2, pad - (pad / 2)),
            AudioCausalityAxis::Width | AudioCausalityAxis::WidthCompatibility => {
                (pad, 0, pad / 2, pad - (pad / 2))
            }
            AudioCausalityAxis::Height => (pad / 2, pad - (pad / 2), pad, 0),
        };
        let conv_vb = vb.pp("conv");
        let weight = conv_vb.get(
            (out_channels, in_channels, kernel_size, kernel_size),
            "weight",
        )?;
        let bias = conv_vb.get(out_channels, "bias").ok();
        let conv = Conv2d::new(
            weight,
            bias,
            Conv2dConfig {
                stride,
                ..Default::default()
            },
        );
        Ok(Self {
            conv,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = x
            .pad_with_zeros(3, self.pad_left, self.pad_right)?
            .pad_with_zeros(2, self.pad_top, self.pad_bottom)?;
        padded.apply(&self.conv).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct AudioResnetBlock {
    in_channels: usize,
    out_channels: usize,
    norm1: AudioNorm,
    conv1: AudioCausalConv2d,
    norm2: AudioNorm,
    conv2: AudioCausalConv2d,
    nin_shortcut: Option<AudioCausalConv2d>,
}

impl AudioResnetBlock {
    fn load(
        in_channels: usize,
        out_channels: usize,
        norm_type: AudioNormType,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = AudioNorm::load(in_channels, norm_type, vb.pp("norm1"))?;
        let conv1 = AudioCausalConv2d::new(
            in_channels,
            out_channels,
            3,
            1,
            causality_axis,
            vb.pp("conv1"),
        )?;
        let norm2 = AudioNorm::load(out_channels, norm_type, vb.pp("norm2"))?;
        let conv2 = AudioCausalConv2d::new(
            out_channels,
            out_channels,
            3,
            1,
            causality_axis,
            vb.pp("conv2"),
        )?;
        let nin_shortcut = if in_channels != out_channels {
            Some(AudioCausalConv2d::new(
                in_channels,
                out_channels,
                1,
                1,
                causality_axis,
                vb.pp("nin_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            in_channels,
            out_channels,
            norm1,
            conv1,
            norm2,
            conv2,
            nin_shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.norm1.forward(x)?;
        h = silu(&h)?;
        h = self.conv1.forward(&h)?;
        h = self.norm2.forward(&h)?;
        h = silu(&h)?;
        h = self.conv2.forward(&h)?;
        let residual = match &self.nin_shortcut {
            Some(shortcut) => shortcut.forward(x)?,
            None => x.clone(),
        };
        residual.add(&h).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct AudioMidBlock {
    block_1: AudioResnetBlock,
    block_2: AudioResnetBlock,
}

impl AudioMidBlock {
    fn load(
        channels: usize,
        norm_type: AudioNormType,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            block_1: AudioResnetBlock::load(
                channels,
                channels,
                norm_type,
                causality_axis,
                vb.pp("block_1"),
            )?,
            block_2: AudioResnetBlock::load(
                channels,
                channels,
                norm_type,
                causality_axis,
                vb.pp("block_2"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block_1.forward(x)?;
        self.block_2.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct AudioDownsample {
    conv: Conv2d,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

impl AudioDownsample {
    fn load(
        in_channels: usize,
        with_conv: bool,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        if !with_conv {
            bail!("audio VAE downsample without convolution is not implemented for native LTX-2");
        }
        let (pad_left, pad_right, pad_top, pad_bottom) = match causality_axis {
            AudioCausalityAxis::None => (0, 1, 0, 1),
            AudioCausalityAxis::Width => (2, 0, 0, 1),
            AudioCausalityAxis::Height => (0, 1, 2, 0),
            AudioCausalityAxis::WidthCompatibility => (1, 0, 0, 1),
        };
        let conv_vb = vb.pp("conv");
        let weight = conv_vb.get((in_channels, in_channels, 3, 3), "weight")?;
        let bias = conv_vb.get(in_channels, "bias").ok();
        let conv = Conv2d::new(
            weight,
            bias,
            Conv2dConfig {
                stride: 2,
                ..Default::default()
            },
        );
        Ok(Self {
            conv,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = x
            .pad_with_zeros(3, self.pad_left, self.pad_right)?
            .pad_with_zeros(2, self.pad_top, self.pad_bottom)?;
        padded.apply(&self.conv).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct AudioUpsample {
    causality_axis: AudioCausalityAxis,
    conv: AudioCausalConv2d,
}

impl AudioUpsample {
    fn load(
        in_channels: usize,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            causality_axis,
            conv: AudioCausalConv2d::new(
                in_channels,
                in_channels,
                3,
                1,
                causality_axis,
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let mut x = x.upsample_nearest2d(h * 2, w * 2)?;
        x = self.conv.forward(&x)?;
        match self.causality_axis {
            AudioCausalityAxis::None | AudioCausalityAxis::WidthCompatibility => Ok(x),
            AudioCausalityAxis::Height => {
                let (_, _, up_h, _) = x.dims4()?;
                x.narrow(2, 1, up_h.saturating_sub(1)).map_err(Into::into)
            }
            AudioCausalityAxis::Width => {
                let (_, _, _, up_w) = x.dims4()?;
                x.narrow(3, 1, up_w.saturating_sub(1)).map_err(Into::into)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct AudioDecoderStage {
    blocks: Vec<AudioResnetBlock>,
    upsample: Option<AudioUpsample>,
}

#[derive(Debug, Clone)]
struct AudioEncoderStage {
    blocks: Vec<AudioResnetBlock>,
    downsample: Option<AudioDownsample>,
}

#[derive(Debug, Clone)]
pub struct Ltx2AudioEncoder {
    pub config: Ltx2AudioEncoderConfig,
    per_channel_statistics: AudioPerChannelStatistics,
    patchifier: AudioPatchifier,
    conv_in: AudioCausalConv2d,
    down: Vec<AudioEncoderStage>,
    mid: AudioMidBlock,
    norm_out: AudioNorm,
    conv_out: AudioCausalConv2d,
}

impl Ltx2AudioEncoder {
    pub fn load_from_checkpoint(
        checkpoint_path: &Path,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let config = Ltx2AudioEncoderConfig::load(checkpoint_path)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[PathBuf::from(checkpoint_path)], dtype, device)
        }
        .with_context(|| format!("failed to mmap {}", checkpoint_path.display()))?;
        Self::new(config, vb)
    }

    pub fn new(config: Ltx2AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        if !config.attn_resolutions.is_empty() {
            bail!("audio VAE encoder attention blocks are not implemented for native LTX-2");
        }
        if config.mid_block_add_attention {
            bail!("audio VAE encoder mid-block attention is not implemented for native LTX-2");
        }
        if config.dropout != 0.0 {
            bail!("audio VAE encoder dropout != 0.0 is not implemented for native LTX-2");
        }

        let audio_vb = vb.pp("audio_vae");
        let encoder_vb = audio_vb.pp("encoder");
        let per_channel_statistics = AudioPerChannelStatistics::load(audio_vb.clone(), config.ch)?;
        let patchifier = AudioPatchifier::new(
            config.sample_rate,
            config.mel_hop_length,
            LATENT_DOWNSAMPLE_FACTOR,
            config.is_causal,
            0,
        );
        let conv_in = AudioCausalConv2d::new(
            config.in_channels,
            config.ch,
            3,
            1,
            config.causality_axis,
            encoder_vb.pp("conv_in"),
        )?;

        let num_resolutions = config.ch_mult.len();
        let mut down = Vec::with_capacity(num_resolutions);
        let mut block_in = config.ch;
        for level in 0..num_resolutions {
            let block_out = config.ch * config.ch_mult[level];
            let mut blocks = Vec::with_capacity(config.num_res_blocks);
            for block_idx in 0..config.num_res_blocks {
                let block = AudioResnetBlock::load(
                    block_in,
                    block_out,
                    config.norm_type,
                    config.causality_axis,
                    encoder_vb.pp(format!("down.{level}.block.{block_idx}")),
                )?;
                block_in = block_out;
                blocks.push(block);
            }
            let downsample = if level + 1 < num_resolutions {
                Some(AudioDownsample::load(
                    block_in,
                    config.resamp_with_conv,
                    config.causality_axis,
                    encoder_vb.pp(format!("down.{level}.downsample")),
                )?)
            } else {
                None
            };
            down.push(AudioEncoderStage { blocks, downsample });
        }

        let mid = AudioMidBlock::load(
            block_in,
            config.norm_type,
            config.causality_axis,
            encoder_vb.pp("mid"),
        )?;
        let norm_out = AudioNorm::load(block_in, config.norm_type, encoder_vb.pp("norm_out"))?;
        let conv_out = AudioCausalConv2d::new(
            block_in,
            if config.double_z {
                2 * config.z_channels
            } else {
                config.z_channels
            },
            3,
            1,
            config.causality_axis,
            encoder_vb.pp("conv_out"),
        )?;

        Ok(Self {
            config,
            per_channel_statistics,
            patchifier,
            conv_in,
            down,
            mid,
            norm_out,
            conv_out,
        })
    }

    pub fn encode(&self, spectrogram: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(spectrogram)?;
        for stage in &self.down {
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
            if let Some(downsample) = stage.downsample.as_ref() {
                h = downsample.forward(&h)?;
            }
        }
        h = self.mid.forward(&h)?;
        h = self.norm_out.forward(&h)?;
        h = silu(&h)?;
        h = self.conv_out.forward(&h)?;
        self.normalize_latents(&h)
    }

    fn normalize_latents(&self, latent_output: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, mel_bins) = latent_output.dims4()?;
        let means = if self.config.double_z {
            latent_output.narrow(1, 0, self.config.z_channels.min(channels))?
        } else {
            latent_output.clone()
        };
        let (_, mean_channels, _, _) = means.dims4()?;
        let latent_shape = AudioLatentShape {
            batch,
            channels: mean_channels,
            frames,
            mel_bins,
        };
        let latent_patched = self.patchifier.patchify(&means)?;
        let latent_normalized = self.per_channel_statistics.normalize(&latent_patched)?;
        self.patchifier
            .unpatchify(&latent_normalized, latent_shape)
            .map_err(Into::into)
    }

    pub fn encode_audio(&self, audio: &DecodedAudio) -> Result<Tensor> {
        let device = self.per_channel_statistics.mean.device();
        let dtype = self.per_channel_statistics.mean.dtype();
        let waveform = audio.to_tensor(self.config.sample_rate, self.config.in_channels, device)?;
        let mel = waveform_to_log_mel(
            &waveform,
            self.config.sample_rate,
            self.config.mel_bins,
            self.config.mel_hop_length,
            self.config.n_fft,
            device,
        )?;
        self.encode(&mel.to_dtype(dtype)?)
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2AudioDecoder {
    pub config: Ltx2AudioDecoderConfig,
    per_channel_statistics: AudioPerChannelStatistics,
    patchifier: AudioPatchifier,
    conv_in: AudioCausalConv2d,
    mid: AudioMidBlock,
    up: Vec<AudioDecoderStage>,
    norm_out: AudioNorm,
    conv_out: AudioCausalConv2d,
}

impl Ltx2AudioDecoder {
    pub fn load_from_checkpoint(
        checkpoint_path: &Path,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let config = Ltx2AudioDecoderConfig::load(checkpoint_path)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[PathBuf::from(checkpoint_path)], dtype, device)
        }
        .with_context(|| format!("failed to mmap {}", checkpoint_path.display()))?;
        Self::new(config, vb)
    }

    pub fn new(config: Ltx2AudioDecoderConfig, vb: VarBuilder) -> Result<Self> {
        if !config.attn_resolutions.is_empty() {
            bail!("audio VAE attention blocks are not implemented for native LTX-2");
        }
        if config.mid_block_add_attention {
            bail!("audio VAE mid-block attention is not implemented for native LTX-2");
        }
        if config.dropout != 0.0 {
            bail!("audio VAE dropout != 0.0 is not implemented for native LTX-2");
        }

        let audio_vb = vb.pp("audio_vae");
        let decoder_vb = audio_vb.pp("decoder");
        let per_channel_statistics = AudioPerChannelStatistics::load(audio_vb.clone(), config.ch)?;
        let patchifier = AudioPatchifier::new(
            config.sample_rate,
            config.mel_hop_length,
            LATENT_DOWNSAMPLE_FACTOR,
            config.is_causal,
            0,
        );

        let base_block_channels = config.ch * config.ch_mult[config.ch_mult.len() - 1];
        let conv_in = AudioCausalConv2d::new(
            config.z_channels,
            base_block_channels,
            3,
            1,
            config.causality_axis,
            decoder_vb.pp("conv_in"),
        )?;
        let mid = AudioMidBlock::load(
            base_block_channels,
            config.norm_type,
            config.causality_axis,
            decoder_vb.pp("mid"),
        )?;

        let num_resolutions = config.ch_mult.len();
        let mut block_in = base_block_channels;
        let mut up = (0..num_resolutions)
            .map(|_| None)
            .collect::<Vec<Option<AudioDecoderStage>>>();
        for level in (0..num_resolutions).rev() {
            let block_out = config.ch * config.ch_mult[level];
            let mut blocks = Vec::with_capacity(config.num_res_blocks + 1);
            for block_idx in 0..(config.num_res_blocks + 1) {
                let block = AudioResnetBlock::load(
                    block_in,
                    block_out,
                    config.norm_type,
                    config.causality_axis,
                    decoder_vb.pp(format!("up.{level}.block.{block_idx}")),
                )?;
                block_in = block_out;
                blocks.push(block);
            }
            let upsample = if level != 0 {
                Some(AudioUpsample::load(
                    block_in,
                    config.causality_axis,
                    decoder_vb.pp(format!("up.{level}.upsample")),
                )?)
            } else {
                None
            };
            up[level] = Some(AudioDecoderStage { blocks, upsample });
        }
        let up = up
            .into_iter()
            .map(|stage| stage.context("audio decoder stage was not initialized"))
            .collect::<Result<Vec<_>>>()?;

        let norm_out = AudioNorm::load(block_in, config.norm_type, decoder_vb.pp("norm_out"))?;
        let conv_out = AudioCausalConv2d::new(
            block_in,
            config.out_ch,
            3,
            1,
            config.causality_axis,
            decoder_vb.pp("conv_out"),
        )?;

        Ok(Self {
            config,
            per_channel_statistics,
            patchifier,
            conv_in,
            mid,
            up,
            norm_out,
            conv_out,
        })
    }

    pub fn decode(&self, sample: &Tensor) -> Result<Tensor> {
        let (sample, target_shape) = self.denormalize_latents(sample)?;
        let mut h = self.conv_in.forward(&sample)?;
        h = self.mid.forward(&h)?;
        for level in (0..self.up.len()).rev() {
            let stage = &self.up[level];
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
            if let Some(upsample) = stage.upsample.as_ref() {
                h = upsample.forward(&h)?;
            }
        }
        h = self.norm_out.forward(&h)?;
        h = silu(&h)?;
        h = self.conv_out.forward(&h)?;
        adjust_decoded_output_shape(&h, target_shape)
    }

    fn denormalize_latents(&self, sample: &Tensor) -> Result<(Tensor, AudioLatentShape)> {
        let (batch, channels, frames, mel_bins) = sample.dims4()?;
        let latent_shape = AudioLatentShape {
            batch,
            channels,
            frames,
            mel_bins,
        };
        let sample_patched = self.patchifier.patchify(sample)?;
        let sample_denormalized = self.per_channel_statistics.denormalize(&sample_patched)?;
        let sample = self
            .patchifier
            .unpatchify(&sample_denormalized, latent_shape)?;
        let target_shape = decoder_target_shape(
            latent_shape,
            self.config.out_ch,
            self.config.mel_bins,
            self.config.causality_axis,
        );
        Ok((sample, target_shape))
    }
}

fn decoder_target_shape(
    latent_shape: AudioLatentShape,
    out_channels: usize,
    mel_bins: usize,
    causality_axis: AudioCausalityAxis,
) -> AudioLatentShape {
    let mut target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR;
    if causality_axis != AudioCausalityAxis::None {
        target_frames = target_frames
            .saturating_sub(LATENT_DOWNSAMPLE_FACTOR - 1)
            .max(1);
    }
    AudioLatentShape {
        batch: latent_shape.batch,
        channels: out_channels,
        frames: target_frames,
        mel_bins,
    }
}

fn adjust_decoded_output_shape(
    decoded_output: &Tensor,
    target_shape: AudioLatentShape,
) -> Result<Tensor> {
    let (batch, channels, current_time, current_freq) = decoded_output.dims4()?;
    let cropped = decoded_output
        .narrow(0, 0, batch.min(target_shape.batch))?
        .narrow(1, 0, channels.min(target_shape.channels))?
        .narrow(2, 0, current_time.min(target_shape.frames))?
        .narrow(3, 0, current_freq.min(target_shape.mel_bins))?;
    let (_, _, cropped_time, cropped_freq) = cropped.dims4()?;
    let padded = cropped
        .pad_with_zeros(3, 0, target_shape.mel_bins.saturating_sub(cropped_freq))?
        .pad_with_zeros(2, 0, target_shape.frames.saturating_sub(cropped_time))?
        .pad_with_zeros(
            1,
            0,
            target_shape
                .channels
                .saturating_sub(channels.min(target_shape.channels)),
        )?;
    let (_, _, padded_time, padded_freq) = padded.dims4()?;
    padded
        .narrow(1, 0, target_shape.channels)?
        .narrow(2, 0, padded_time.min(target_shape.frames))?
        .narrow(3, 0, padded_freq.min(target_shape.mel_bins))
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;

    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    #[cfg(feature = "mp4")]
    use image::{ImageBuffer, Rgb, RgbImage};
    #[cfg(feature = "mp4")]
    use std::path::PathBuf;

    use super::{
        adjust_decoded_output_shape, decoder_target_shape, waveform_to_log_mel, AudioCausalConv2d,
        AudioCausalityAxis, AudioDownsample, AudioNormType, AudioPerChannelStatistics,
        DecodedAudio, Ltx2AudioDecoder, Ltx2AudioDecoderConfig, Ltx2AudioEncoder,
        Ltx2AudioEncoderConfig,
    };
    #[cfg(feature = "mp4")]
    use crate::ltx2::media::attach_aac_track_from_f32_interleaved;
    use crate::ltx2::media::encode_wav_f32_interleaved;
    use crate::ltx2::model::AudioLatentShape;
    #[cfg(feature = "mp4")]
    use crate::ltx_video::video_enc;

    fn vb_from_tensors(tensors: Vec<(&str, Tensor)>) -> VarBuilder<'static> {
        let map = tensors
            .into_iter()
            .map(|(name, tensor)| (name.to_string(), tensor))
            .collect::<HashMap<_, _>>();
        VarBuilder::from_tensors(map, candle_core::DType::F32, &Device::Cpu)
    }

    #[cfg(feature = "mp4")]
    fn sample_frames() -> Vec<RgbImage> {
        [[255, 64, 32], [32, 192, 255], [16, 224, 96], [240, 224, 64]]
            .into_iter()
            .map(|rgb| ImageBuffer::from_pixel(64, 64, Rgb(rgb)))
            .collect()
    }

    #[cfg(feature = "mp4")]
    fn sample_audio_track(samples_per_channel: usize, sample_rate: u32, channels: u16) -> Vec<f32> {
        let mut samples = Vec::with_capacity(samples_per_channel * channels as usize);
        for idx in 0..samples_per_channel {
            let t = idx as f32 / sample_rate as f32;
            for channel in 0..channels {
                let freq = if channel % 2 == 0 { 440.0 } else { 660.0 };
                samples.push((t * std::f32::consts::TAU * freq).sin() * 0.25);
            }
        }
        samples
    }

    #[cfg(feature = "mp4")]
    fn write_mp4_with_native_aac_params(
        sample_rate: u32,
        channels: u16,
        extension: &str,
    ) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let video_path = dir.path().join("video.mp4");
        fs::write(
            &video_path,
            video_enc::encode_mp4(&sample_frames(), 12).unwrap(),
        )
        .unwrap();
        let audio_path = dir.path().join(format!("video-audio.{extension}"));
        attach_aac_track_from_f32_interleaved(
            &video_path,
            &audio_path,
            &sample_audio_track((sample_rate / 12).max(1024) as usize, sample_rate, channels),
            sample_rate,
            channels,
        )
        .unwrap();
        (dir, audio_path)
    }

    #[cfg(feature = "mp4")]
    fn write_mp4_with_native_aac() -> (tempfile::TempDir, PathBuf) {
        write_mp4_with_native_aac_params(48_000, 2, "mp4")
    }

    #[test]
    fn audio_per_channel_statistics_denormalize_patchified_latents() {
        let device = Device::Cpu;
        let stats = AudioPerChannelStatistics {
            mean: Tensor::from_vec(vec![1f32, 2.0], 2, &device).unwrap(),
            std: Tensor::from_vec(vec![2f32, 4.0], 2, &device).unwrap(),
        };
        let x = Tensor::from_vec(vec![3f32, 5.0], (1, 1, 2), &device).unwrap();
        let denorm = stats.denormalize(&x).unwrap();
        assert_eq!(denorm.to_vec3::<f32>().unwrap(), [[[7.0, 22.0]]]);
    }

    #[test]
    fn audio_per_channel_statistics_normalize_patchified_latents() {
        let device = Device::Cpu;
        let stats = AudioPerChannelStatistics {
            mean: Tensor::from_vec(vec![1f32, 2.0], 2, &device).unwrap(),
            std: Tensor::from_vec(vec![2f32, 4.0], 2, &device).unwrap(),
        };
        let x = Tensor::from_vec(vec![7f32, 22.0], (1, 1, 2), &device).unwrap();
        let norm = stats.normalize(&x).unwrap();
        assert_eq!(norm.to_vec3::<f32>().unwrap(), [[[3.0, 5.0]]]);
    }

    #[test]
    fn causal_height_conv2d_uses_top_only_padding() {
        let device = Device::Cpu;
        let vb = vb_from_tensors(vec![
            (
                "conv.weight",
                Tensor::ones((1, 1, 3, 3), candle_core::DType::F32, &device).unwrap(),
            ),
            (
                "conv.bias",
                Tensor::zeros(1, candle_core::DType::F32, &device).unwrap(),
            ),
        ]);
        let conv = AudioCausalConv2d::new(1, 1, 3, 1, AudioCausalityAxis::Height, vb).unwrap();
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0], (1, 1, 3, 1), &device).unwrap();
        let y = conv.forward(&x).unwrap();
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 3.0, 6.0]
        );
    }

    #[test]
    fn causal_height_downsample_uses_top_only_padding() {
        let device = Device::Cpu;
        let vb = vb_from_tensors(vec![
            (
                "conv.weight",
                Tensor::ones((1, 1, 3, 3), candle_core::DType::F32, &device).unwrap(),
            ),
            (
                "conv.bias",
                Tensor::zeros(1, candle_core::DType::F32, &device).unwrap(),
            ),
        ]);
        let downsample = AudioDownsample::load(1, true, AudioCausalityAxis::Height, vb).unwrap();
        let x = Tensor::from_vec(
            vec![1f32, 10.0, 2.0, 20.0, 3.0, 30.0],
            (1, 1, 3, 2),
            &device,
        )
        .unwrap();
        let y = downsample.forward(&x).unwrap();
        assert_eq!(y.dims4().unwrap(), (1, 1, 2, 1));
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![11.0, 66.0]
        );
    }

    #[test]
    fn decoder_target_shape_applies_causal_latent_trim() {
        let target = decoder_target_shape(
            AudioLatentShape {
                batch: 1,
                channels: 8,
                frames: 4,
                mel_bins: 16,
            },
            2,
            64,
            AudioCausalityAxis::Height,
        );
        assert_eq!(target.frames, 13);
        assert_eq!(target.mel_bins, 64);
        assert_eq!(target.channels, 2);
    }

    #[test]
    fn adjust_decoded_output_shape_crops_and_pads_to_target() {
        let device = Device::Cpu;
        let decoded = Tensor::from_vec(
            vec![
                1f32, 2.0, 3.0, 4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (1, 2, 2, 3),
            &device,
        )
        .unwrap();
        let adjusted = adjust_decoded_output_shape(
            &decoded,
            AudioLatentShape {
                batch: 1,
                channels: 2,
                frames: 3,
                mel_bins: 4,
            },
        )
        .unwrap();
        assert_eq!(adjusted.dims4().unwrap(), (1, 2, 3, 4));
    }

    #[test]
    fn audio_decoder_forward_respects_causal_target_shape() {
        let device = Device::Cpu;
        let config = Ltx2AudioDecoderConfig {
            ch: 2,
            out_ch: 2,
            ch_mult: vec![1],
            num_res_blocks: 1,
            attn_resolutions: vec![],
            resolution: 4,
            z_channels: 1,
            norm_type: AudioNormType::Pixel,
            causality_axis: AudioCausalityAxis::Height,
            dropout: 0.0,
            mid_block_add_attention: false,
            sample_rate: 16_000,
            mel_hop_length: 160,
            is_causal: true,
            mel_bins: 4,
        };
        let zero1 = |len| Tensor::zeros(len, DType::F32, &device).unwrap();
        let zero4 = |shape| Tensor::zeros(shape, DType::F32, &device).unwrap();
        let vb = vb_from_tensors(vec![
            ("audio_vae.per_channel_statistics.mean-of-means", zero1(2)),
            (
                "audio_vae.per_channel_statistics.std-of-means",
                Tensor::ones(2, DType::F32, &device).unwrap(),
            ),
            ("audio_vae.decoder.conv_in.conv.weight", zero4((2, 1, 3, 3))),
            ("audio_vae.decoder.conv_in.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.mid.block_1.conv1.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.mid.block_1.conv1.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.mid.block_1.conv2.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.mid.block_1.conv2.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.mid.block_2.conv1.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.mid.block_2.conv1.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.mid.block_2.conv2.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.mid.block_2.conv2.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.up.0.block.0.conv1.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.up.0.block.0.conv1.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.up.0.block.0.conv2.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.up.0.block.0.conv2.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.up.0.block.1.conv1.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.up.0.block.1.conv1.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.up.0.block.1.conv2.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.up.0.block.1.conv2.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.conv_out.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.conv_out.conv.bias", zero1(2)),
        ]);
        let decoder = Ltx2AudioDecoder::new(config, vb).unwrap();
        let latent = Tensor::zeros((1, 1, 3, 2), candle_core::DType::F32, &device).unwrap();
        let decoded = decoder.decode(&latent).unwrap();
        assert_eq!(decoded.dims4().unwrap(), (1, 2, 9, 4));
    }

    #[test]
    fn audio_encoder_forward_emits_normalized_latent_shape() {
        let device = Device::Cpu;
        let config = Ltx2AudioEncoderConfig {
            ch: 8,
            ch_mult: vec![1],
            num_res_blocks: 1,
            attn_resolutions: vec![],
            resolution: 4,
            z_channels: 2,
            double_z: true,
            norm_type: AudioNormType::Pixel,
            causality_axis: AudioCausalityAxis::Height,
            dropout: 0.0,
            mid_block_add_attention: false,
            resamp_with_conv: true,
            in_channels: 2,
            sample_rate: 16_000,
            mel_hop_length: 160,
            n_fft: 1024,
            is_causal: true,
            mel_bins: 4,
        };
        let zero1 = |len| Tensor::zeros(len, DType::F32, &device).unwrap();
        let zero4 = |shape| Tensor::zeros(shape, DType::F32, &device).unwrap();
        let vb = vb_from_tensors(vec![
            ("audio_vae.per_channel_statistics.mean-of-means", zero1(8)),
            (
                "audio_vae.per_channel_statistics.std-of-means",
                Tensor::ones(8, DType::F32, &device).unwrap(),
            ),
            ("audio_vae.encoder.conv_in.conv.weight", zero4((8, 2, 3, 3))),
            ("audio_vae.encoder.conv_in.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.down.0.block.0.conv1.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.down.0.block.0.conv1.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.down.0.block.0.conv2.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.down.0.block.0.conv2.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.mid.block_1.conv1.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.mid.block_1.conv1.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.mid.block_1.conv2.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.mid.block_1.conv2.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.mid.block_2.conv1.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.mid.block_2.conv1.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.mid.block_2.conv2.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.mid.block_2.conv2.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.conv_out.conv.weight",
                zero4((4, 8, 3, 3)),
            ),
            ("audio_vae.encoder.conv_out.conv.bias", zero1(4)),
        ]);
        let encoder = Ltx2AudioEncoder::new(config, vb).unwrap();
        let spectrogram = Tensor::zeros((1, 2, 9, 4), candle_core::DType::F32, &device).unwrap();
        let encoded = encoder.encode(&spectrogram).unwrap();
        assert_eq!(encoded.dims4().unwrap(), (1, 2, 9, 4));
    }

    #[test]
    fn decoded_audio_from_wav_trims_to_requested_duration() {
        let temp_dir = tempfile::tempdir().unwrap();
        let wav_path = temp_dir.path().join("source.wav");
        let mut samples = Vec::new();
        for idx in 0..1600 {
            let value = idx as f32 / 1600.0;
            samples.push(value);
            samples.push(-value);
        }
        fs::write(
            &wav_path,
            encode_wav_f32_interleaved(&samples, 16_000, 2).unwrap(),
        )
        .unwrap();

        let decoded = DecodedAudio::from_file(&wav_path, Some(0.05))
            .unwrap()
            .unwrap();

        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.channel_count(), 2);
        assert_eq!(decoded.sample_count(), 800);
    }

    #[test]
    fn decoded_audio_to_tensor_resamples_and_duplicates_mono() {
        let decoded = DecodedAudio {
            sample_rate: 8_000,
            channels: vec![vec![0.0, 1.0, 0.0, -1.0]],
        };

        let waveform = decoded.to_tensor(16_000, 2, &Device::Cpu).unwrap();

        assert_eq!(waveform.dims3().unwrap(), (1, 2, 8));
        let channels = waveform.i(0).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(channels[0], channels[1]);
    }

    #[test]
    fn waveform_to_log_mel_emits_expected_shape() {
        let waveform = Tensor::zeros((1, 2, 640), DType::F32, &Device::Cpu).unwrap();

        let mel = waveform_to_log_mel(&waveform, 16_000, 4, 160, 1024, &Device::Cpu).unwrap();

        assert_eq!(mel.dims4().unwrap(), (1, 2, 5, 4));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn decoded_audio_from_mp4_falls_back_when_probe_rejects_sl_descriptor() {
        let (_dir, bad_sl_path) = write_mp4_with_native_aac();

        let probe_err = DecodedAudio::decode_with_probe(&bad_sl_path).unwrap_err();
        assert!(
            format!("{probe_err:#}").contains("sl descriptor predefined not mp4"),
            "unexpected probe error: {probe_err:#}"
        );

        let decoded = DecodedAudio::from_file(&bad_sl_path, Some(0.05))
            .unwrap()
            .unwrap();

        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.channel_count(), 2);
        assert_eq!(decoded.sample_count(), 2_400);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn decoded_audio_from_mp4_fallback_handles_mono_24khz_native_aac() {
        let (_dir, path) = write_mp4_with_native_aac_params(24_000, 1, "mp4");

        let decoded = DecodedAudio::from_file(&path, Some(0.05)).unwrap().unwrap();

        assert_eq!(decoded.sample_rate, 24_000);
        assert_eq!(decoded.channel_count(), 1);
        assert_eq!(decoded.sample_count(), 1_200);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn decoded_audio_from_m4a_extension_uses_mp4_ingest_path() {
        let (_dir, path) = write_mp4_with_native_aac_params(16_000, 2, "m4a");

        let decoded = DecodedAudio::from_file(&path, Some(0.05)).unwrap().unwrap();

        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.channel_count(), 2);
        assert_eq!(decoded.sample_count(), 800);
    }
}
