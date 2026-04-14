use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};

use super::patchifiers::{AudioPatchifier, VideoLatentPatchifier};
use super::shapes::{AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum LtxRopeType {
    Interleaved,
    Split,
}

pub fn midpoint_positions(bounds: &Tensor) -> Result<Tensor> {
    let starts = bounds.narrow(3, 0, 1)?;
    let ends = bounds.narrow(3, 1, 1)?;
    starts
        .broadcast_add(&ends)?
        .affine(0.5, 0.0)
        .map_err(Into::into)
}

pub fn video_token_positions(
    patchifier: VideoLatentPatchifier,
    shape: VideoLatentShape,
    device: &Device,
) -> Result<Tensor> {
    patchifier.get_patch_grid_bounds(shape, device)
}

pub fn get_pixel_coords(
    latent_coords: &Tensor,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool,
) -> Result<Tensor> {
    let scale = Tensor::from_vec(
        vec![
            scale_factors.time as f32,
            scale_factors.height as f32,
            scale_factors.width as f32,
        ],
        (1, 3, 1, 1),
        latent_coords.device(),
    )?;
    let mut pixel_coords = latent_coords.to_dtype(DType::F32)?.broadcast_mul(&scale)?;
    if causal_fix {
        let temporal = pixel_coords.i((.., 0..1, .., ..))?;
        let corrected = temporal
            .affine(1.0, 1.0 - scale_factors.time as f64)?
            .clamp(0.0f32, f32::MAX)?;
        let height_width = pixel_coords.i((.., 1.., .., ..))?;
        pixel_coords = Tensor::cat(&[corrected, height_width], 1)?;
    }
    Ok(pixel_coords)
}

pub fn scale_video_time_to_seconds(pixel_coords: &Tensor, fps: f32) -> Result<Tensor> {
    let temporal = pixel_coords
        .i((.., 0..1, .., ..))?
        .affine(1.0 / fps as f64, 0.0)?;
    let height_width = pixel_coords.i((.., 1.., .., ..))?;
    Tensor::cat(&[temporal, height_width], 1).map_err(Into::into)
}

pub fn audio_temporal_positions(
    patchifier: AudioPatchifier,
    shape: AudioLatentShape,
    device: &Device,
) -> Result<Tensor> {
    patchifier.get_patch_grid_bounds(shape, device)
}

pub fn cross_modal_temporal_positions(
    video_bounds: &Tensor,
    audio_bounds: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let video = midpoint_positions(&video_bounds.narrow(1, 0, 1)?.to_dtype(DType::F32)?)?;
    let audio = midpoint_positions(&audio_bounds.to_dtype(DType::F32)?)?;
    Ok((video, audio))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, IndexOp};

    use super::{
        audio_temporal_positions, cross_modal_temporal_positions, get_pixel_coords,
        midpoint_positions, scale_video_time_to_seconds, video_token_positions,
    };
    use crate::ltx2::model::{
        AudioLatentShape, AudioPatchifier, SpatioTemporalScaleFactors, VideoLatentPatchifier,
        VideoLatentShape,
    };

    #[test]
    fn video_rope_positions_cover_3d_grid() {
        let device = Device::Cpu;
        let bounds = video_token_positions(
            VideoLatentPatchifier::new(1),
            VideoLatentShape {
                batch: 1,
                channels: 128,
                frames: 2,
                height: 2,
                width: 2,
            },
            &device,
        )
        .unwrap();

        assert_eq!(bounds.dims4().unwrap(), (1, 3, 8, 2));
        let mids = midpoint_positions(&bounds).unwrap();
        assert_eq!(mids.dims4().unwrap(), (1, 3, 8, 1));
    }

    #[test]
    fn audio_rope_positions_are_temporal_only() {
        let device = Device::Cpu;
        let bounds = audio_temporal_positions(
            AudioPatchifier::new(16_000, 160, 4, true, 0),
            AudioLatentShape {
                batch: 1,
                channels: 8,
                frames: 4,
                mel_bins: 16,
            },
            &device,
        )
        .unwrap();

        assert_eq!(bounds.dims4().unwrap(), (1, 1, 4, 2));
    }

    #[test]
    fn cross_modal_temporal_positions_keep_video_and_audio_sequences() {
        let device = Device::Cpu;
        let video = video_token_positions(
            VideoLatentPatchifier::new(1),
            VideoLatentShape {
                batch: 1,
                channels: 128,
                frames: 2,
                height: 1,
                width: 1,
            },
            &device,
        )
        .unwrap();
        let audio = audio_temporal_positions(
            AudioPatchifier::new(16_000, 160, 4, true, 0),
            AudioLatentShape {
                batch: 1,
                channels: 8,
                frames: 3,
                mel_bins: 16,
            },
            &device,
        )
        .unwrap();

        let (video_temporal, audio_temporal) =
            cross_modal_temporal_positions(&video, &audio).unwrap();
        assert_eq!(video_temporal.dims4().unwrap(), (1, 1, 2, 1));
        assert_eq!(audio_temporal.dims4().unwrap(), (1, 1, 3, 1));
    }

    #[test]
    fn pixel_coords_apply_causal_first_frame_fix() {
        let device = Device::Cpu;
        let latent = video_token_positions(
            VideoLatentPatchifier::new(1),
            VideoLatentShape {
                batch: 1,
                channels: 128,
                frames: 2,
                height: 1,
                width: 1,
            },
            &device,
        )
        .unwrap();

        let pixel = get_pixel_coords(&latent, SpatioTemporalScaleFactors::default(), true).unwrap();
        let temporal = pixel.i((0, 0, .., ..)).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(temporal, vec![vec![0.0, 1.0], vec![1.0, 9.0]]);
    }

    #[test]
    fn video_time_positions_scale_to_seconds() {
        let device = Device::Cpu;
        let latent = video_token_positions(
            VideoLatentPatchifier::new(1),
            VideoLatentShape {
                batch: 1,
                channels: 128,
                frames: 2,
                height: 1,
                width: 1,
            },
            &device,
        )
        .unwrap();
        let pixel = get_pixel_coords(&latent, SpatioTemporalScaleFactors::default(), true).unwrap();
        let seconds = scale_video_time_to_seconds(&pixel, 4.0).unwrap();
        let temporal = seconds.i((0, 0, .., ..)).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(temporal, vec![vec![0.0, 0.25], vec![0.25, 2.25]]);
    }
}
