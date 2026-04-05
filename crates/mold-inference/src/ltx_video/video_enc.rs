//! Video encoding utilities for LTX Video output.

use anyhow::{Context, Result};
use image::RgbImage;

/// Generation metadata embedded as tEXt chunks in APNG output.
pub struct VideoMetadata {
    pub prompt: String,
    pub model: String,
    pub seed: u64,
    pub steps: u32,
    pub guidance: f64,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
}

/// Encode a sequence of RGB frames into an animated GIF.
///
/// Uses per-frame NeuQuant palette quantization (256 colors).
pub fn encode_gif(frames: &[RgbImage], fps: u32) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width() as u16, frames[0].height() as u16);
    let delay_cs = (100.0 / fps as f64).round() as u16;

    let mut buf = Vec::new();
    {
        let mut encoder = gif::Encoder::new(&mut buf, width, height, &[])
            .context("failed to create GIF encoder")?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .context("failed to set GIF repeat")?;

        for frame_img in frames {
            let rgba: image::RgbaImage =
                image::DynamicImage::ImageRgb8(frame_img.clone()).into_rgba8();
            let mut pixels = rgba.into_raw();

            let mut gif_frame = gif::Frame::from_rgba_speed(width, height, &mut pixels, 10);
            gif_frame.delay = delay_cs;
            gif_frame.dispose = gif::DisposalMethod::Any;

            encoder
                .write_frame(&gif_frame)
                .context("failed to write GIF frame")?;
        }
    }
    Ok(buf)
}

/// Extract the first frame as a PNG thumbnail.
pub fn first_frame_png(frames: &[RgbImage]) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames for thumbnail");

    let mut buf = std::io::Cursor::new(Vec::new());
    frames[0]
        .write_to(&mut buf, image::ImageFormat::Png)
        .context("failed to encode thumbnail PNG")?;
    Ok(buf.into_inner())
}

/// Encode a sequence of RGB frames into an animated PNG (APNG).
///
/// Loops infinitely. Optionally embeds generation metadata as tEXt/iTXt chunks.
pub fn encode_apng(
    frames: &[RgbImage],
    fps: u32,
    metadata: Option<&VideoMetadata>,
) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width(), frames[0].height());
    let num_frames = frames.len() as u32;

    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(num_frames, 0)?;
        encoder.set_frame_delay(1, fps as u16)?;

        if let Some(meta) = metadata {
            encoder.add_itxt_chunk("mold:prompt".to_string(), meta.prompt.clone())?;
            encoder.add_itxt_chunk("mold:model".to_string(), meta.model.clone())?;
            encoder.add_text_chunk("mold:seed".to_string(), meta.seed.to_string())?;
            encoder.add_text_chunk("mold:steps".to_string(), meta.steps.to_string())?;
            encoder.add_text_chunk("mold:guidance".to_string(), meta.guidance.to_string())?;
            encoder.add_text_chunk("mold:width".to_string(), meta.width.to_string())?;
            encoder.add_text_chunk("mold:height".to_string(), meta.height.to_string())?;
            encoder.add_text_chunk("mold:frames".to_string(), meta.frames.to_string())?;
            encoder.add_text_chunk("mold:fps".to_string(), meta.fps.to_string())?;
        }

        let mut writer = encoder.write_header().context("failed to write APNG header")?;

        for (i, frame) in frames.iter().enumerate() {
            if i > 0 {
                writer.set_blend_op(png::BlendOp::Source)?;
                writer.set_dispose_op(png::DisposeOp::Background)?;
            }
            writer
                .write_image_data(frame.as_raw())
                .with_context(|| format!("failed to write APNG frame {i}"))?;
        }

        writer.finish().context("failed to finalize APNG")?;
    }
    Ok(buf)
}

/// Encode a sequence of RGB frames into an animated WebP.
///
/// Uses the `webp-animation` crate (statically linked libwebp).
#[cfg(feature = "webp")]
pub fn encode_webp(frames: &[RgbImage], fps: u32) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width(), frames[0].height());
    let frame_duration_ms = (1000.0 / fps as f64).round() as i32;

    let mut encoder = webp_animation::Encoder::new((width, height))
        .map_err(|e| anyhow::anyhow!("failed to create WebP encoder: {e}"))?;

    for (i, frame_img) in frames.iter().enumerate() {
        let rgba: image::RgbaImage =
            image::DynamicImage::ImageRgb8(frame_img.clone()).into_rgba8();
        let timestamp_ms = i as i32 * frame_duration_ms;
        encoder
            .add_frame(rgba.as_raw(), timestamp_ms)
            .map_err(|e| anyhow::anyhow!("failed to add WebP frame {i}: {e}"))?;
    }

    let final_timestamp_ms = frames.len() as i32 * frame_duration_ms;
    let webp_data = encoder
        .finalize(final_timestamp_ms)
        .map_err(|e| anyhow::anyhow!("failed to finalize WebP animation: {e}"))?;

    Ok(webp_data.to_vec())
}

/// Encode a sequence of RGB frames into an MP4 (H.264/AVC) video.
///
/// Uses OpenH264 for H.264 frame encoding and muxide for MP4 container muxing.
#[cfg(feature = "mp4")]
pub fn encode_mp4(frames: &[RgbImage], fps: u32) -> Result<Vec<u8>> {
    use std::cell::RefCell;
    use std::io::{self, Cursor, Seek, Write};
    use std::rc::Rc;

    use muxide::api::{MuxerBuilder, VideoCodec};
    use openh264::encoder::{BitRate, EncoderConfig, FrameRate};
    use openh264::formats::{RgbSliceU8, YUVBuffer};

    /// Shared in-memory writer for muxide (needs Write + Seek).
    #[derive(Clone)]
    struct SharedWriter(Rc<RefCell<Cursor<Vec<u8>>>>);

    impl SharedWriter {
        fn new() -> Self {
            Self(Rc::new(RefCell::new(Cursor::new(Vec::new()))))
        }
        fn into_bytes(self) -> Vec<u8> {
            Rc::try_unwrap(self.0)
                .expect("SharedWriter still has multiple references")
                .into_inner()
                .into_inner()
        }
    }

    impl Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.borrow_mut().write(buf)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.0.borrow_mut().flush()
        }
    }

    impl Seek for SharedWriter {
        fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
            self.0.borrow_mut().seek(pos)
        }
    }

    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width(), frames[0].height());
    let frame_duration_s = 1.0 / fps as f64;

    let config = EncoderConfig::new()
        .max_frame_rate(FrameRate::from_hz(fps as f32))
        .bitrate(BitRate::from_bps(2_000_000));

    let api = openh264::OpenH264API::from_source();
    let mut h264_encoder = openh264::encoder::Encoder::with_api_config(api, config)
        .context("failed to create H.264 encoder")?;

    let writer = SharedWriter::new();
    let mut muxer = MuxerBuilder::new(writer.clone())
        .video(VideoCodec::H264, width, height, fps as f64)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to create MP4 muxer: {e}"))?;

    for (i, frame) in frames.iter().enumerate() {
        let rgb = RgbSliceU8::new(frame.as_raw(), (width as usize, height as usize));
        let yuv = YUVBuffer::from_rgb_source(rgb);
        let bitstream = h264_encoder
            .encode(&yuv)
            .context("failed to encode H.264 frame")?;
        let is_keyframe = matches!(bitstream.frame_type(), openh264::encoder::FrameType::IDR);
        let nal_data = bitstream.to_vec();
        if nal_data.is_empty() {
            continue;
        }
        let pts = i as f64 * frame_duration_s;
        muxer
            .write_video(pts, &nal_data, is_keyframe)
            .map_err(|e| anyhow::anyhow!("failed to mux MP4 frame {i}: {e}"))?;
    }

    muxer
        .finish_in_place()
        .map_err(|e| anyhow::anyhow!("failed to finalize MP4: {e}"))?;
    drop(muxer);

    Ok(writer.into_bytes())
}
