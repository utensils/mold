//! Bounded inspection and delivered-pixel boundaries for long movies.

use anyhow::{ensure, Context, Result};
use image::RgbImage;
use openh264::formats::YUVSource;
use std::fs::File;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoFacts {
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    pub fps_numerator: u32,
    pub fps_denominator: u32,
}

fn video_track(reader: &mp4_rs::Mp4Reader<std::io::BufReader<File>>) -> Result<&mp4_rs::Mp4Track> {
    let mut videos = reader
        .tracks()
        .values()
        .filter(|track| matches!(track.track_type(), Ok(mp4_rs::TrackType::Video)));
    let video = videos.next().context("missing video track")?;
    ensure!(
        videos.next().is_none(),
        "multiple video tracks are unsupported"
    );
    Ok(video)
}

fn facts(track: &mp4_rs::Mp4Track, movie_timescale: u32) -> Result<VideoFacts> {
    crate::video_assembly::track_config(track, movie_timescale)?;
    ensure!(
        track.width() > 0
            && track.height() > 0
            && u64::from(track.width()) * u64::from(track.height()) * 3 <= 512 * 1024 * 1024,
        "source canvas exceeds decoded frame budget"
    );
    let entries = &track.trak.mdia.minf.stbl.stts.entries;
    let ticks = entries
        .first()
        .context("empty video timeline")?
        .sample_delta;
    ensure!(
        ticks > 0 && entries.iter().all(|entry| entry.sample_delta == ticks),
        "variable frame rate requires normalization"
    );
    let frames = entries.iter().try_fold(0u32, |sum, entry| {
        sum.checked_add(entry.sample_count)
            .context("frame count overflow")
    })?;
    ensure!(
        frames > 0
            && frames == track.sample_count()
            && u64::from(frames) * u64::from(ticks) == track.trak.mdia.mdhd.duration,
        "inconsistent video timeline"
    );
    let (mut a, mut b) = (track.timescale(), ticks);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Ok(VideoFacts {
        frames,
        width: track.width().into(),
        height: track.height().into(),
        fps_numerator: track.timescale() / a,
        fps_denominator: ticks / a,
    })
}

pub fn inspect(input: File, checkpoint: &mut dyn FnMut() -> Result<()>) -> Result<VideoFacts> {
    let reader = crate::video_assembly_input::checked_reader(input, checkpoint)?;
    facts(video_track(&reader)?, reader.moov.mvhd.timescale)
}

/// Decode only the GOPs required by the visible boundary. Hidden trimmed
/// pictures never condition the next generation.
pub fn tail(
    input: File,
    keep_frames: u32,
    tail_frames: u32,
    expected_width: u32,
    expected_height: u32,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<RgbImage>> {
    let packets = input.try_clone()?;
    let reader = crate::video_assembly_input::checked_reader(input, checkpoint)?;
    let track = video_track(&reader)?;
    let facts = facts(track, reader.moov.mvhd.timescale)?;
    ensure!(
        (facts.width, facts.height) == (expected_width, expected_height),
        "source canvas differs from the frozen boundary"
    );
    ensure!(
        tail_frames > 0 && tail_frames <= keep_frames && keep_frames <= facts.frames,
        "invalid visible boundary"
    );
    let budget = u64::from(tail_frames)
        .checked_mul(u64::from(facts.width))
        .and_then(|n| n.checked_mul(u64::from(facts.height)))
        .and_then(|n| n.checked_mul(3))
        .context("boundary size overflow")?;
    ensure!(
        budget <= 512 * 1024 * 1024,
        "boundary exceeds decoded frame budget"
    );
    let first = keep_frames - tail_frames;
    let start = match &track.trak.mdia.minf.stbl.stss {
        Some(sync) => {
            sync.entries
                .iter()
                .copied()
                .take_while(|&sample| sample <= first + 1)
                .last()
                .context("no independent picture before boundary")?
                - 1
        }
        None => first,
    };
    let mut result = Vec::with_capacity(tail_frames as usize);
    visit(
        track,
        packets,
        start,
        keep_frames,
        checkpoint,
        &mut |index, frame, _checkpoint| {
            if index >= first {
                result.push(frame);
            }
            Ok(())
        },
    )?;
    ensure!(
        result.len() == tail_frames as usize,
        "decoder did not deliver the complete boundary"
    );
    Ok(result)
}

type FrameConsumer<'a> =
    dyn FnMut(u32, RgbImage, &mut dyn FnMut() -> Result<()>) -> Result<()> + 'a;

fn visit(
    track: &mp4_rs::Mp4Track,
    packets: File,
    start: u32,
    end: u32,
    checkpoint: &mut dyn FnMut() -> Result<()>,
    consume: &mut FrameConsumer<'_>,
) -> Result<()> {
    use openh264::decoder::{Decoder, DecoderConfig, Flush};
    let avcc = &track
        .trak
        .mdia
        .minf
        .stbl
        .stsd
        .avc1
        .as_ref()
        .context("missing AVC configuration")?
        .avcc;
    let sps: Vec<_> = avcc
        .sequence_parameter_sets
        .iter()
        .map(|unit| unit.bytes.clone())
        .collect();
    let pps: Vec<_> = avcc
        .picture_parameter_sets
        .iter()
        .map(|unit| unit.bytes.clone())
        .collect();
    let mut decoder = Decoder::with_api_config(
        openh264::OpenH264API::from_source(),
        DecoderConfig::new().flush_after_decode(Flush::NoFlush),
    )?;
    let mut cursor = crate::video_assembly_input::PacketCursor::new(track, packets)?;
    cursor.skip_to(start, checkpoint)?;
    let (mut new_idr, mut sps_seen, mut pps_seen) = (true, false, false);
    let mut converted = Vec::new();
    let mut index = start;
    let mut consume_image = |image: openh264::decoder::DecodedYUV<'_>,
                             checkpoint: &mut dyn FnMut() -> Result<()>|
     -> Result<()> {
        ensure!(
            image.dimensions() == (usize::from(track.width()), usize::from(track.height())),
            "decoded canvas differs from track"
        );
        consume(
            index,
            crate::bt601::decoded_to_rgb(&image, track.width().into(), track.height().into())?,
            checkpoint,
        )?;
        index += 1;
        Ok(())
    };
    for _ in start..end {
        checkpoint()?;
        let sample = cursor.next()?.context("missing video packet")?;
        crate::ltx2::media::annex_b_convert_packet(
            &sample.bytes,
            avcc.length_size_minus_one + 1,
            &sps,
            &pps,
            &mut new_idr,
            &mut sps_seen,
            &mut pps_seen,
            &mut converted,
        );
        if let Some(image) = decoder.decode(&converted)? {
            consume_image(image, checkpoint)?;
        }
    }
    for image in decoder.flush_remaining()? {
        checkpoint()?;
        consume_image(image, checkpoint)?;
    }
    ensure!(index == end, "decoder picture count differs from timeline");
    Ok(())
}

/// Stream a visible source window to a model canvas and exact output count.
/// An incoming fade replaces the withheld tail of the preceding part. Memory
/// holds one decoded picture, one resized picture and the bounded fade tail.
#[allow(clippy::too_many_arguments)]
pub fn normalize(
    input: File,
    output: File,
    keep_source_frames: u32,
    output_frames: u32,
    width: u32,
    height: u32,
    fps: u32,
    incoming_fade: &[RgbImage],
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<u64> {
    normalize_inner(
        input,
        output,
        keep_source_frames,
        output_frames,
        width,
        height,
        fps,
        incoming_fade,
        false,
        checkpoint,
    )
}

/// Preserve a generated closing picture while shortening the final legal-grid
/// clip to its requested delivered duration. Ordinary frame-clock resampling
/// keeps its existing timestamp-based mapping.
#[allow(clippy::too_many_arguments)]
pub fn normalize_preserving_endpoint(
    input: File,
    output: File,
    keep_source_frames: u32,
    output_frames: u32,
    width: u32,
    height: u32,
    fps: u32,
    incoming_fade: &[RgbImage],
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<u64> {
    normalize_inner(
        input,
        output,
        keep_source_frames,
        output_frames,
        width,
        height,
        fps,
        incoming_fade,
        true,
        checkpoint,
    )
}

#[allow(clippy::too_many_arguments)]
fn normalize_inner(
    input: File,
    output: File,
    keep_source_frames: u32,
    output_frames: u32,
    width: u32,
    height: u32,
    fps: u32,
    incoming_fade: &[RgbImage],
    preserve_endpoint: bool,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<u64> {
    checkpoint()?;
    let packets = input.try_clone()?;
    let reader = crate::video_assembly_input::checked_reader(input, checkpoint)?;
    let track = video_track(&reader)?;
    let source = facts(track, reader.moov.mvhd.timescale)?;
    ensure!(
        keep_source_frames > 0 && keep_source_frames <= source.frames && output_frames > 0,
        "invalid normalization window"
    );
    ensure!(
        width > 0
            && height > 0
            && width.is_multiple_of(2)
            && height.is_multiple_of(2)
            && u64::from(width) * u64::from(height) * 3 <= 512 * 1024 * 1024,
        "invalid normalization canvas"
    );
    ensure!(
        incoming_fade.len() <= output_frames as usize
            && incoming_fade
                .iter()
                .all(|frame| frame.dimensions() == (width, height)),
        "invalid incoming fade canvas or length"
    );
    ensure!(
        incoming_fade.len() as u64 * u64::from(width) * u64::from(height) * 3 <= 512 * 1024 * 1024,
        "incoming fade exceeds decoded frame budget"
    );
    let mut encoder = crate::ltx_video::video_enc::Mp4FileEncoder::new(output, width, height, fps)?;
    let mut written = 0u32;
    visit(
        track,
        packets,
        0,
        keep_source_frames,
        checkpoint,
        &mut |index, frame, checkpoint| {
            let scale = (f64::from(width) / f64::from(frame.width()))
                .min(f64::from(height) / f64::from(frame.height()));
            let resized_width = ((f64::from(frame.width()) * scale).round() as u32).clamp(1, width);
            let resized_height =
                ((f64::from(frame.height()) * scale).round() as u32).clamp(1, height);
            let frame = if frame.dimensions() == (width, height) {
                frame
            } else {
                let resized = image::imageops::resize(
                    &frame,
                    resized_width,
                    resized_height,
                    image::imageops::FilterType::Triangle,
                );
                let mut canvas = RgbImage::new(width, height);
                image::imageops::replace(
                    &mut canvas,
                    &resized,
                    i64::from((width - resized_width) / 2),
                    i64::from((height - resized_height) / 2),
                );
                canvas
            };
            while written < output_frames
                && (if preserve_endpoint {
                    if output_frames == 1 {
                        u64::from(keep_source_frames - 1)
                    } else {
                        u64::from(written) * u64::from(keep_source_frames - 1)
                            / u64::from(output_frames - 1)
                    }
                } else {
                    u64::from(written) * u64::from(keep_source_frames) / u64::from(output_frames)
                }) == u64::from(index)
            {
                checkpoint()?;
                if let Some(previous) = incoming_fade.get(written as usize) {
                    let alpha = written as f32 / incoming_fade.len() as f32;
                    let mut blended = frame.clone();
                    for (pixel, prior) in blended.pixels_mut().zip(previous.pixels()) {
                        for channel in 0..3 {
                            pixel.0[channel] = (prior.0[channel] as f32 * (1.0 - alpha)
                                + pixel.0[channel] as f32 * alpha)
                                as u8;
                        }
                    }
                    encoder.push(&blended)?;
                } else {
                    encoder.push(&frame)?;
                }
                written += 1;
            }
            Ok(())
        },
    )?;
    ensure!(
        written == output_frames,
        "normalization did not deliver requested duration"
    );
    checkpoint()?;
    encoder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    #[test]
    fn normalization_streams_exact_duration_and_blends_the_incoming_boundary() {
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([0, 0, 220])); 12];
        let source = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            source.path(),
            crate::ltx_video::video_enc::encode_mp4(&frames, 12).unwrap(),
        )
        .unwrap();
        let output = tempfile::NamedTempFile::new().unwrap();
        let previous = vec![RgbImage::from_pixel(64, 32, Rgb([220, 0, 0])); 4];
        assert_eq!(
            normalize(
                source.reopen().unwrap(),
                output.reopen().unwrap(),
                12,
                24,
                64,
                32,
                24,
                &previous,
                &mut || Ok(())
            )
            .unwrap(),
            24
        );
        let result = inspect(output.reopen().unwrap(), &mut || Ok(())).unwrap();
        assert_eq!(
            (
                result.frames,
                result.width,
                result.height,
                result.fps_numerator
            ),
            (24, 64, 32, 24)
        );
        let (_, decoded) =
            crate::ltx2::media::decode_video_frames_from_path(output.path()).unwrap();
        assert!(decoded[0].get_pixel(32, 16).0[0] > 180);
        assert!(decoded[3].get_pixel(32, 16).0[2] > 100);
        assert!(decoded[23].get_pixel(32, 16).0[2] > 180);
        assert!(decoded[23].get_pixel(0, 16).0.iter().all(|&c| c < 8));
        let cancelled = tempfile::NamedTempFile::new().unwrap();
        assert!(normalize(
            source.reopen().unwrap(),
            cancelled.reopen().unwrap(),
            12,
            24,
            64,
            32,
            24,
            &[],
            &mut || anyhow::bail!("cancelled")
        )
        .is_err());
    }
    #[test]
    fn tail_uses_visible_frames_before_the_final_trim() {
        let frames: Vec<_> = (0..30)
            .map(|i| RgbImage::from_pixel(32, 32, Rgb([i * 7, 80, 40])))
            .collect();
        let bytes = crate::ltx_video::video_enc::encode_mp4(&frames, 24).unwrap();
        let input = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(input.path(), bytes).unwrap();
        let facts = inspect(std::fs::File::open(input.path()).unwrap(), &mut || Ok(())).unwrap();
        assert_eq!(
            (facts.frames, facts.fps_numerator, facts.fps_denominator),
            (30, 24, 1)
        );
        let (_, oracle) = crate::ltx2::media::decode_video_frames_from_path(input.path()).unwrap();
        let actual = tail(
            std::fs::File::open(input.path()).unwrap(),
            17,
            3,
            32,
            32,
            &mut || Ok(()),
        )
        .unwrap();
        assert_eq!(actual, oracle[14..17]);
        assert_ne!(actual, oracle[27..30]);
        assert!(tail(
            std::fs::File::open(input.path()).unwrap(),
            17,
            3,
            64,
            32,
            &mut || Ok(())
        )
        .is_err());
        assert!(tail(
            std::fs::File::open(input.path()).unwrap(),
            31,
            3,
            32,
            32,
            &mut || Ok(())
        )
        .is_err());
    }
    #[test]
    fn closing_frame_survives_an_off_grid_final_clip() {
        let frames = vec![
            RgbImage::from_pixel(32, 32, Rgb([220, 0, 0])),
            RgbImage::from_pixel(32, 32, Rgb([220, 0, 0])),
            RgbImage::from_pixel(32, 32, Rgb([220, 0, 0])),
            RgbImage::from_pixel(32, 32, Rgb([0, 0, 220])),
        ];
        let source = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            source.path(),
            crate::ltx_video::video_enc::encode_mp4(&frames, 24).unwrap(),
        )
        .unwrap();
        let output = tempfile::NamedTempFile::new().unwrap();
        normalize_preserving_endpoint(
            source.reopen().unwrap(),
            output.reopen().unwrap(),
            4,
            2,
            32,
            32,
            24,
            &[],
            &mut || Ok(()),
        )
        .unwrap();
        let (_, decoded) =
            crate::ltx2::media::decode_video_frames_from_path(output.path()).unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(decoded[0].get_pixel(16, 16).0[0] > 180);
        assert!(decoded[1].get_pixel(16, 16).0[2] > 180);
        let ordinary = tempfile::NamedTempFile::new().unwrap();
        normalize(
            source.reopen().unwrap(),
            ordinary.reopen().unwrap(),
            4,
            2,
            32,
            32,
            24,
            &[],
            &mut || Ok(()),
        )
        .unwrap();
        let (_, decoded) =
            crate::ltx2::media::decode_video_frames_from_path(ordinary.path()).unwrap();
        assert!(decoded[1].get_pixel(16, 16).0[0] > 180);
    }
}
