//! Video encoding utilities for LTX Video output.

use anyhow::{Context, Result};
use image::RgbImage;

/// Encode a sequence of RGB frames into an animated GIF.
///
/// Uses per-frame NeuQuant palette quantization (256 colors).
/// `delay_cs` is the delay between frames in centiseconds (100cs = 1 second).
pub fn encode_gif(frames: &[RgbImage], fps: u32) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width() as u16, frames[0].height() as u16);
    let delay_cs = (100.0 / fps as f64).round() as u16; // centiseconds per frame

    let mut buf = Vec::new();
    {
        let mut encoder = gif::Encoder::new(&mut buf, width, height, &[])
            .context("failed to create GIF encoder")?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .context("failed to set GIF repeat")?;

        for frame_img in frames {
            let rgba: image::RgbaImage = image::DynamicImage::ImageRgb8(frame_img.clone()).into_rgba8();
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
