use image::{imageops, Rgb, RgbImage};
use mold_core::{Ltx2SpatialUpscale, Ltx2TemporalUpscale};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stage1RenderShape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    pub(crate) fps: u32,
}

pub(crate) fn derive_stage1_render_shape(
    target_width: u32,
    target_height: u32,
    target_frames: u32,
    target_fps: u32,
    spatial_upscale: Option<Ltx2SpatialUpscale>,
    temporal_upscale: Option<Ltx2TemporalUpscale>,
) -> Stage1RenderShape {
    let (width, height) = match spatial_upscale {
        Some(Ltx2SpatialUpscale::X1_5) => (
            aligned_downsample(target_width, 1.5),
            aligned_downsample(target_height, 1.5),
        ),
        Some(Ltx2SpatialUpscale::X2) => (
            aligned_downsample(target_width, 2.0),
            aligned_downsample(target_height, 2.0),
        ),
        None => (target_width.max(16), target_height.max(16)),
    };
    let (frames, fps) = match temporal_upscale {
        Some(Ltx2TemporalUpscale::X2) => (
            target_frames.saturating_sub(1) / 2 + 1,
            (target_fps / 2).max(1),
        ),
        None => (target_frames.max(1), target_fps.max(1)),
    };
    Stage1RenderShape {
        width,
        height,
        frames,
        fps,
    }
}

fn aligned_downsample(target: u32, scale: f32) -> u32 {
    let raw = ((target as f32 / scale).floor() as u32).max(16);
    let aligned = (raw / 16) * 16;
    aligned.max(16).min(target.max(16))
}

pub(crate) fn spatially_upsample_frames(
    frames: &[RgbImage],
    target_width: u32,
    target_height: u32,
) -> Vec<RgbImage> {
    frames
        .iter()
        .map(|frame| {
            if frame.width() == target_width && frame.height() == target_height {
                frame.clone()
            } else {
                imageops::resize(
                    frame,
                    target_width,
                    target_height,
                    imageops::FilterType::CatmullRom,
                )
            }
        })
        .collect()
}

pub(crate) fn temporally_upsample_frames_x2(
    frames: &[RgbImage],
    target_frames: Option<u32>,
) -> Vec<RgbImage> {
    if frames.is_empty() {
        return Vec::new();
    }
    if frames.len() == 1 {
        return normalize_frame_count(vec![frames[0].clone()], target_frames);
    }

    let mut upsampled = Vec::with_capacity(frames.len() * 2 - 1);
    for pair in frames.windows(2) {
        let lhs = &pair[0];
        let rhs = &pair[1];
        upsampled.push(lhs.clone());
        upsampled.push(blend_frames(lhs, rhs));
    }
    upsampled.push(frames.last().cloned().expect("non-empty frames"));
    normalize_frame_count(upsampled, target_frames)
}

fn normalize_frame_count(mut frames: Vec<RgbImage>, target_frames: Option<u32>) -> Vec<RgbImage> {
    let Some(target_frames) = target_frames else {
        return frames;
    };
    let target_frames = target_frames.max(1) as usize;
    if frames.len() > target_frames {
        frames.truncate(target_frames);
        return frames;
    }
    while frames.len() < target_frames {
        frames.push(frames.last().cloned().expect("non-empty frames"));
    }
    frames
}

fn blend_frames(lhs: &RgbImage, rhs: &RgbImage) -> RgbImage {
    let mut blended = RgbImage::new(lhs.width(), lhs.height());
    for (dst, (a, b)) in blended.pixels_mut().zip(lhs.pixels().zip(rhs.pixels())) {
        *dst = Rgb([
            ((u16::from(a[0]) + u16::from(b[0])) / 2) as u8,
            ((u16::from(a[1]) + u16::from(b[1])) / 2) as u8,
            ((u16::from(a[2]) + u16::from(b[2])) / 2) as u8,
        ]);
    }
    blended
}

#[cfg(test)]
mod tests {
    use image::{ImageBuffer, Rgb};

    use super::{
        derive_stage1_render_shape, spatially_upsample_frames, temporally_upsample_frames_x2,
    };

    #[test]
    fn derives_stage_one_shape_for_x1_5_spatial_upscale() {
        let shape = derive_stage1_render_shape(
            1216,
            704,
            17,
            12,
            Some(mold_core::Ltx2SpatialUpscale::X1_5),
            None,
        );
        assert_eq!(shape.width, 800);
        assert_eq!(shape.height, 464);
        assert_eq!(shape.frames, 17);
        assert_eq!(shape.fps, 12);
    }

    #[test]
    fn derives_stage_one_shape_for_x2_temporal_upscale() {
        let shape = derive_stage1_render_shape(
            960,
            576,
            17,
            12,
            None,
            Some(mold_core::Ltx2TemporalUpscale::X2),
        );
        assert_eq!(shape.width, 960);
        assert_eq!(shape.height, 576);
        assert_eq!(shape.frames, 9);
        assert_eq!(shape.fps, 6);
    }

    #[test]
    fn spatial_upsample_resizes_frames_to_target_dimensions() {
        let frame = ImageBuffer::from_pixel(64, 32, Rgb([12, 34, 56]));
        let upsampled = spatially_upsample_frames(&[frame], 128, 64);
        assert_eq!(upsampled.len(), 1);
        assert_eq!(upsampled[0].dimensions(), (128, 64));
    }

    #[test]
    fn temporal_upsample_inserts_blended_inbetween_frame() {
        let lhs = ImageBuffer::from_pixel(1, 1, Rgb([0, 0, 0]));
        let rhs = ImageBuffer::from_pixel(1, 1, Rgb([200, 100, 50]));
        let upsampled = temporally_upsample_frames_x2(&[lhs, rhs], Some(3));
        assert_eq!(upsampled.len(), 3);
        assert_eq!(upsampled[0].get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(upsampled[1].get_pixel(0, 0).0, [100, 50, 25]);
        assert_eq!(upsampled[2].get_pixel(0, 0).0, [200, 100, 50]);
    }

    #[test]
    fn temporal_upsample_trims_to_requested_even_frame_count() {
        let frames = vec![
            ImageBuffer::from_pixel(1, 1, Rgb([0, 0, 0])),
            ImageBuffer::from_pixel(1, 1, Rgb([64, 64, 64])),
            ImageBuffer::from_pixel(1, 1, Rgb([255, 255, 255])),
        ];
        let upsampled = temporally_upsample_frames_x2(&frames, Some(4));
        assert_eq!(upsampled.len(), 4);
        assert_eq!(upsampled[0].get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(upsampled[3].get_pixel(0, 0).0, [159, 159, 159]);
    }
}
