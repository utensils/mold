//! File-backed assembly of compatible, independently decodable video parts.
//! Payloads are copied one packet at a time; neither decoded frames nor the
//! growing movie are retained in RAM. MP4 sample-table metadata grows with
//! frame count, as required by the progressive container.

use anyhow::{bail, ensure, Context, Result};
use mp4_rs::{
    AvcConfig, MediaConfig, MediaType, Mp4Config, Mp4Track, Mp4Writer, TrackConfig, TrackType,
};
use std::fs::File;

pub(crate) fn track_config(track: &Mp4Track, movie_timescale: u32) -> Result<TrackConfig> {
    ensure!(
        movie_timescale > 0 && track.timescale() > 0,
        "video presentation timescale is zero"
    );
    // Packet passthrough is safe only for an identity presentation timeline.
    // Nonidentity edits and reordered pictures require the decoded fallback.
    if let Some(edits) = track
        .trak
        .edts
        .as_ref()
        .and_then(|edits| edits.elst.as_ref())
    {
        ensure!(
            edits.entries.len() <= 1
                && edits.entries.iter().all(|edit| edit.media_time == 0
                    && edit.media_rate == 1
                    && edit.media_rate_fraction == 0
                    && u128::from(edit.segment_duration) * u128::from(track.timescale())
                        == u128::from(track.trak.mdia.mdhd.duration) * u128::from(movie_timescale)),
            "video part needs decoded assembly because its edit list changes presentation"
        );
    }
    ensure!(
        track.trak.tkhd.matrix == Default::default()
            && track.trak.tkhd.width.raw_value() == u32::from(track.width()) << 16
            && track.trak.tkhd.height.raw_value() == u32::from(track.height()) << 16,
        "rotated or transformed video requires decoded assembly"
    );
    let media_conf = match track.media_type()? {
        MediaType::H264 => {
            let config = &track
                .trak
                .mdia
                .minf
                .stbl
                .stsd
                .avc1
                .as_ref()
                .context("missing AVC decoder configuration")?
                .avcc;
            ensure!(
                config.length_size_minus_one & 3 == 3
                    && config.sequence_parameter_sets.len() == 1
                    && config.picture_parameter_sets.len() == 1,
                "video requires decoded assembly for its AVC decoder configuration"
            );
            MediaConfig::AvcConfig(AvcConfig {
                width: track.width(),
                height: track.height(),
                seq_param_set: track.sequence_parameter_set()?.to_vec(),
                pic_param_set: track.picture_parameter_set()?.to_vec(),
            })
        }
        _ => bail!("video part requires decoded assembly for its codec"),
    };
    Ok(TrackConfig {
        track_type: track.track_type()?,
        timescale: track.timescale(),
        language: "und".into(),
        media_conf,
    })
}

#[derive(PartialEq)]
struct PartFormat {
    config: TrackConfig,
    description: Vec<u8>,
    frame_duration: u32,
}

fn part_format(input: File, checkpoint: &mut dyn FnMut() -> Result<()>) -> Result<PartFormat> {
    let mut presentation = input.try_clone()?;
    let packets = input.try_clone()?;
    let reader = crate::video_assembly_input::checked_reader(input, checkpoint)?;
    let mut videos = reader
        .tracks()
        .values()
        .filter(|track| matches!(track.track_type(), Ok(TrackType::Video)));
    let track = videos.next().context("missing video track")?;
    ensure!(
        videos.next().is_none()
            && reader
                .tracks()
                .values()
                .filter(|track| matches!(track.track_type(), Ok(TrackType::Audio)))
                .count()
                <= 1,
        "video part has unsupported track layout"
    );
    let config = track_config(track, reader.moov.mvhd.timescale)?;
    let entries = &track.trak.mdia.minf.stbl.stts.entries;
    let frame_duration = entries
        .first()
        .context("missing video timeline")?
        .sample_delta;
    ensure!(
        frame_duration > 0
            && entries
                .iter()
                .all(|entry| entry.sample_count > 0 && entry.sample_delta == frame_duration),
        "variable frame rate requires normalization"
    );
    let frames = entries.iter().try_fold(0u64, |sum, entry| {
        sum.checked_add(u64::from(entry.sample_count))
            .context("video sample count overflow")
    })?;
    ensure!(
        frames > 0
            && frames == u64::from(track.sample_count())
            && frames.checked_mul(u64::from(frame_duration)) == Some(track.trak.mdia.mdhd.duration),
        "video sample timeline is inconsistent"
    );
    ensure!(
        track
            .trak
            .mdia
            .minf
            .stbl
            .stss
            .as_ref()
            .is_none_or(|sync| sync.entries.first() == Some(&1)),
        "video part does not begin with an independent picture"
    );
    crate::video_assembly_input::PacketCursor::new(track, packets)?;
    let description = crate::video_assembly_boxes::read_track_box(
        &mut presentation,
        track.track_id(),
        &[*b"mdia", *b"minf", *b"stbl", *b"stsd"],
        4 * 1024 * 1024,
    )?;
    Ok(PartFormat {
        config,
        description,
        frame_duration,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssemblyReceipt {
    pub video_frames: u64,
    pub video_ticks: u64,
    pub video_timescale: u32,
}

/// The caller supplies a private staging file and publishes it only after
/// `finish` succeeds. Any error poisons this writer; partial movies cannot be
/// finalized accidentally. Input files are already-open retained authorities.
pub struct Mp4FileAssembly {
    output: Option<File>,
    writer: Option<Mp4Writer<File>>,
    configs: Vec<TrackConfig>,
    sample_description: Option<Vec<u8>>,
    frame_duration: Option<u32>,
    video_ticks: u64,
    video_frames: u64,
    poisoned: bool,
}

impl Mp4FileAssembly {
    /// Compare presentation and decoder configuration before committing either
    /// file. This reads bounded metadata, without decoding picture payloads.
    pub fn compatible_parts(
        first: File,
        second: File,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<bool> {
        Ok(part_format(first, checkpoint)? == part_format(second, checkpoint)?)
    }

    /// A probe never changes or poisons the current staging movie.
    pub fn accepts(&self, input: File, checkpoint: &mut dyn FnMut() -> Result<()>) -> Result<bool> {
        ensure!(!self.poisoned, "video assembly already failed");
        let candidate = part_format(input, checkpoint)?;
        Ok(self.configs.is_empty()
            || (self.configs.as_slice() == [candidate.config]
                && self.sample_description.as_ref() == Some(&candidate.description)
                && self.frame_duration == Some(candidate.frame_duration)))
    }

    pub fn new(output: File) -> Self {
        Self {
            output: Some(output),
            writer: None,
            configs: Vec::new(),
            sample_description: None,
            frame_duration: None,
            video_ticks: 0,
            video_frames: 0,
            poisoned: false,
        }
    }

    /// Append the first `keep_frames` pictures of a bounded generated part
    /// or retained Library prefix. The first packet must be independently
    /// decodable; no frame seeking or growing-prefix decoding takes place.
    pub fn append(
        &mut self,
        input: File,
        keep_frames: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(!self.poisoned, "video assembly already failed");
        let result = self.append_inner(input, keep_frames, checkpoint);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn append_inner(
        &mut self,
        input: File,
        keep_frames: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        checkpoint()?;
        ensure!(keep_frames > 0, "cannot append an empty video part");
        let mut presentation = input.try_clone()?;
        let packets = input.try_clone()?;
        let input = crate::video_assembly_input::checked_reader(input, checkpoint)?;
        let videos: Vec<_> = input
            .tracks()
            .iter()
            .filter(|(_, track)| matches!(track.track_type(), Ok(TrackType::Video)))
            .map(|(id, _)| *id)
            .collect();
        let audios: Vec<_> = input
            .tracks()
            .iter()
            .filter(|(_, track)| matches!(track.track_type(), Ok(TrackType::Audio)))
            .map(|(id, _)| *id)
            .collect();
        ensure!(
            videos.len() == 1 && audios.len() <= 1,
            "video part must contain one video and at most one audio track"
        );
        let video_id = videos[0];
        let description = crate::video_assembly_boxes::read_track_box(
            &mut presentation,
            video_id,
            &[*b"mdia", *b"minf", *b"stbl", *b"stsd"],
            4 * 1024 * 1024,
        )?;
        ensure!(
            keep_frames <= input.tracks()[&video_id].sample_count(),
            "video part has fewer frames than planned"
        );
        let configs = vec![track_config(
            &input.tracks()[&video_id],
            input.moov.mvhd.timescale,
        )?];
        if self.writer.is_none() {
            let config = Mp4Config {
                major_brand: "isom".parse()?,
                minor_version: 0x200,
                compatible_brands: vec![
                    "isom".parse()?,
                    "iso2".parse()?,
                    "avc1".parse()?,
                    "mp41".parse()?,
                ],
                timescale: configs[0].timescale,
            };
            let mut writer = Mp4Writer::write_start(
                self.output.take().context("missing staging file")?,
                &config,
            )?;
            for track in &configs {
                writer.add_track(track)?;
            }
            self.configs = configs;
            self.sample_description = Some(description);
            self.writer = Some(writer);
        } else {
            ensure!(self.configs == configs && self.sample_description.as_ref() == Some(&description), "video parts require decoded assembly because their decoder configuration or presentation differ");
        }
        let video_scale = self.configs[0].timescale;
        ensure!(video_scale > 0, "video timescale is zero");
        let mut packets =
            crate::video_assembly_input::PacketCursor::new(&input.tracks()[&video_id], packets)?;
        let first = packets.next()?.context("missing first video packet")?;
        ensure!(
            first.is_sync && first.start_time == 0 && first.duration > 0,
            "video part must begin with an independent picture at time zero"
        );
        let frame_ticks = first.duration;
        ensure!(
            self.frame_duration.is_none_or(|ticks| ticks == frame_ticks),
            "video parts require decoded assembly because their frame rates differ"
        );
        self.frame_duration = Some(frame_ticks);
        let mut first = Some(first);
        let part_ticks = u64::from(keep_frames)
            .checked_mul(u64::from(frame_ticks))
            .context("video duration overflowed")?;
        let target_video_ticks = self
            .video_ticks
            .checked_add(part_ticks)
            .context("video duration overflowed")?;
        for video_next in 1..=keep_frames {
            checkpoint()?;
            let video_local = u64::from(video_next - 1) * u64::from(frame_ticks);
            let mut sample = if video_next == 1 {
                first.take().context("missing first video packet")?
            } else {
                packets.next()?.context("missing video packet")?
            };
            ensure!(
                sample.duration == frame_ticks
                    && sample.start_time == video_local
                    && sample.rendering_offset == 0,
                "video part requires decoded assembly for variable rate or reordered pictures"
            );
            sample.start_time = self
                .video_ticks
                .checked_add(video_local)
                .context("video timestamp overflowed")?;
            self.writer.as_mut().unwrap().write_sample(1, &sample)?;
        }
        self.video_ticks = target_video_ticks;
        self.video_frames = self
            .video_frames
            .checked_add(u64::from(keep_frames))
            .context("video frame count overflowed")?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<AssemblyReceipt> {
        ensure!(!self.poisoned, "cannot publish a failed video assembly");
        let mut writer = self
            .writer
            .take()
            .context("cannot publish an empty video assembly")?;
        writer.write_end()?;
        let mut output = writer.into_writer();
        crate::video_assembly_boxes::replace_track_box(
            &mut output,
            1,
            &[*b"mdia", *b"minf", *b"stbl", *b"stsd"],
            self.sample_description
                .as_deref()
                .context("missing retained video presentation")?,
        )?;
        output.sync_all()?;
        Ok(AssemblyReceipt {
            video_frames: self.video_frames,
            video_ticks: self.video_ticks,
            video_timescale: self.configs[0].timescale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ltx_video::video_enc;
    use image::{Rgb, RgbImage};

    fn reader(file: File) -> Result<mp4_rs::Mp4Reader<std::io::BufReader<File>>> {
        crate::video_assembly_input::checked_reader(file, &mut || Ok(()))
    }

    #[test]
    fn compatibility_probe_does_not_poison_an_ongoing_assembly() {
        let source = tempfile::NamedTempFile::new().unwrap();
        let other = tempfile::NamedTempFile::new().unwrap();
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([80, 90, 100])); 4];
        std::fs::write(source.path(), video_enc::encode_mp4(&frames, 24).unwrap()).unwrap();
        std::fs::write(other.path(), video_enc::encode_mp4(&frames, 48).unwrap()).unwrap();
        assert!(Mp4FileAssembly::compatible_parts(
            source.reopen().unwrap(),
            source.reopen().unwrap(),
            &mut || Ok(())
        )
        .unwrap());
        assert!(!Mp4FileAssembly::compatible_parts(
            source.reopen().unwrap(),
            other.reopen().unwrap(),
            &mut || Ok(())
        )
        .unwrap());
        let output = tempfile::tempfile().unwrap();
        let mut assembly = Mp4FileAssembly::new(output);
        assert!(assembly
            .accepts(source.reopen().unwrap(), &mut || Ok(()))
            .unwrap());
        assembly
            .append(source.reopen().unwrap(), 3, &mut || Ok(()))
            .unwrap();
        assert!(!assembly
            .accepts(other.reopen().unwrap(), &mut || Ok(()))
            .unwrap());
        assembly
            .append(source.reopen().unwrap(), 2, &mut || Ok(()))
            .unwrap();
        assert_eq!(assembly.finish().unwrap().video_frames, 5);
    }

    #[test]
    fn joins_parts_without_reencoding_the_original_packets() {
        let dir = tempfile::tempdir().unwrap();
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([80, 90, 100])); 12];
        let bytes = video_enc::encode_mp4(&frames, 24).unwrap();
        let source = dir.path().join("source.mp4");
        std::fs::write(&source, bytes).unwrap();
        let output = dir.path().join("output.mp4");
        let mut assembly = Mp4FileAssembly::new(
            std::fs::OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&output)
                .unwrap(),
        );
        assembly
            .append(std::fs::File::open(&source).unwrap(), 12, &mut || Ok(()))
            .unwrap();
        assembly
            .append(std::fs::File::open(&source).unwrap(), 7, &mut || Ok(()))
            .unwrap();
        let receipt = assembly.finish().unwrap();
        assert_eq!(receipt.video_frames, 19);
        let mut input = reader(std::fs::File::open(&source).unwrap()).unwrap();
        let mut result = reader(std::fs::File::open(&output).unwrap()).unwrap();
        assert_eq!(result.tracks()[&1].sample_count(), 19);
        let path = [*b"mdia", *b"minf", *b"stbl", *b"stsd"];
        assert_eq!(
            crate::video_assembly_boxes::read_track_box(
                &mut File::open(&source).unwrap(),
                1,
                &path,
                1024 * 1024
            )
            .unwrap(),
            crate::video_assembly_boxes::read_track_box(
                &mut File::open(&output).unwrap(),
                1,
                &path,
                1024 * 1024
            )
            .unwrap()
        );
        for frame in 1..=19 {
            let original = input.read_sample(1, (frame - 1) % 12 + 1).unwrap().unwrap();
            let joined = result.read_sample(1, frame).unwrap().unwrap();
            assert_eq!(original.bytes, joined.bytes);
            assert_eq!(joined.start_time, u64::from(frame - 1) * 1000);
        }
    }

    #[test]
    fn rejects_changed_geometry_and_cancelled_copy() {
        let dir = tempfile::tempdir().unwrap();
        let mut assembly = Mp4FileAssembly::new(tempfile::tempfile().unwrap());
        for (index, width) in [32, 64].into_iter().enumerate() {
            let path = dir.path().join(format!("{index}.mp4"));
            let frames = vec![RgbImage::from_pixel(width, 32, Rgb([0, 0, 0])); 2];
            std::fs::write(&path, video_enc::encode_mp4(&frames, 24).unwrap()).unwrap();
            let result = assembly.append(std::fs::File::open(path).unwrap(), 2, &mut || Ok(()));
            assert_eq!(result.is_ok(), index == 0);
        }
        let path = dir.path().join("0.mp4");
        assert!(assembly
            .append(
                std::fs::File::open(path).unwrap(),
                2,
                &mut || anyhow::bail!("cancelled")
            )
            .is_err());
    }

    #[test]
    fn changed_rate_with_identical_decoder_configuration_is_refused() {
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([10, 20, 30])); 3];
        let mut input = tempfile::tempfile().unwrap();
        std::io::Write::write_all(&mut input, &video_enc::encode_mp4(&frames, 24).unwrap())
            .unwrap();
        let mut source = tempfile::tempfile().unwrap();
        let mut first = Mp4FileAssembly::new(source.try_clone().unwrap());
        first.append(input, 3, &mut || Ok(())).unwrap();
        first.finish().unwrap();
        let mut movie = Mp4FileAssembly::new(tempfile::tempfile().unwrap());
        movie
            .append(source.try_clone().unwrap(), 3, &mut || Ok(()))
            .unwrap();
        let path = [*b"mdia", *b"minf", *b"stbl", *b"stts"];
        let mut timing =
            crate::video_assembly_boxes::read_track_box(&mut source, 1, &path, 100).unwrap();
        assert_eq!(timing.len(), 24);
        timing[20..24].copy_from_slice(&2000u32.to_be_bytes());
        crate::video_assembly_boxes::replace_track_box(&mut source, 1, &path, &timing).unwrap();
        let error = movie.append(source, 3, &mut || Ok(())).unwrap_err();
        assert!(error.to_string().contains("frame rates differ"));
        assert!(movie.finish().is_err());
    }

    #[test]
    fn hidden_pictures_and_custom_display_geometry_require_normalization() {
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([10, 20, 30])); 3];
        let mut input = tempfile::tempfile().unwrap();
        std::io::Write::write_all(&mut input, &video_enc::encode_mp4(&frames, 24).unwrap())
            .unwrap();
        let parsed = reader(input).unwrap();
        let original = &parsed.tracks()[&1];
        let mut track = Mp4Track {
            trak: original.trak.clone(),
            trafs: Vec::new(),
            default_sample_duration: 0,
        };
        assert!(track_config(&track, parsed.moov.mvhd.timescale).is_ok());
        track
            .trak
            .edts
            .as_mut()
            .unwrap()
            .elst
            .as_mut()
            .unwrap()
            .entries[0]
            .segment_duration -= 1;
        assert!(track_config(&track, parsed.moov.mvhd.timescale).is_err());
        track.trak = original.trak.clone();
        track.trak.tkhd.width = mp4_rs::FixedPointU16::new(0);
        assert!(track_config(&track, parsed.moov.mvhd.timescale).is_err());
    }
}
