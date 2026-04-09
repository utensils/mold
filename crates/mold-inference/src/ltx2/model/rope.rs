#![allow(dead_code)]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::patchifiers::{AudioPatchifier, VideoLatentPatchifier};
use super::shapes::{AudioLatentShape, VideoLatentShape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtxRopeType {
    Interleaved,
    Split,
}

pub fn midpoint_positions(bounds: &Tensor) -> Result<Tensor> {
    let starts = bounds.narrow(3, 0, 1)?;
    let ends = bounds.narrow(3, 1, 1)?;
    starts.broadcast_add(&ends)?.affine(0.5, 0.0).map_err(Into::into)
}

pub fn video_token_positions(
    patchifier: VideoLatentPatchifier,
    shape: VideoLatentShape,
    device: &Device,
) -> Result<Tensor> {
    patchifier.get_patch_grid_bounds(shape, device)
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
    use candle_core::Device;

    use super::{
        audio_temporal_positions, cross_modal_temporal_positions, midpoint_positions,
        video_token_positions,
    };
    use crate::ltx2::model::{
        AudioLatentShape, AudioPatchifier, VideoLatentPatchifier, VideoLatentShape,
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

        let (video_temporal, audio_temporal) = cross_modal_temporal_positions(&video, &audio).unwrap();
        assert_eq!(video_temporal.dims4().unwrap(), (1, 1, 2, 1));
        assert_eq!(audio_temporal.dims4().unwrap(), (1, 1, 3, 1));
    }
}
