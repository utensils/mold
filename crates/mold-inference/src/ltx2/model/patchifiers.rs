use anyhow::{bail, Result};
use candle_core::{Device, Tensor};

use super::shapes::{AudioLatentShape, VideoLatentShape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoLatentPatchifier {
    patch_size_t: usize,
    patch_size_h: usize,
    patch_size_w: usize,
}

impl VideoLatentPatchifier {
    pub fn new(patch_size: usize) -> Self {
        Self {
            patch_size_t: 1,
            patch_size_h: patch_size,
            patch_size_w: patch_size,
        }
    }

    #[allow(dead_code)]
    pub fn patch_size(self) -> (usize, usize, usize) {
        (self.patch_size_t, self.patch_size_h, self.patch_size_w)
    }

    pub fn get_token_count(self, shape: VideoLatentShape) -> usize {
        shape.frames * shape.height * shape.width
            / (self.patch_size_t * self.patch_size_h * self.patch_size_w)
    }

    pub fn patchify(self, latents: &Tensor) -> Result<Tensor> {
        let (b, c, f, h, w) = latents.dims5()?;
        if f % self.patch_size_t != 0 || h % self.patch_size_h != 0 || w % self.patch_size_w != 0 {
            bail!("video latent shape is not divisible by the configured patch size");
        }

        latents
            .reshape(&[
                b,
                c,
                f / self.patch_size_t,
                self.patch_size_t,
                h / self.patch_size_h,
                self.patch_size_h,
                w / self.patch_size_w,
                self.patch_size_w,
            ])?
            .permute([0, 2, 4, 6, 1, 3, 5, 7])?
            .reshape((
                b,
                (f / self.patch_size_t) * (h / self.patch_size_h) * (w / self.patch_size_w),
                c * self.patch_size_t * self.patch_size_h * self.patch_size_w,
            ))
            .map_err(Into::into)
    }

    pub fn unpatchify(self, latents: &Tensor, output_shape: VideoLatentShape) -> Result<Tensor> {
        let b = output_shape.batch;
        let c = output_shape.channels;
        let f = output_shape.frames;
        let h = output_shape.height;
        let w = output_shape.width;
        let patch_grid_f = f / self.patch_size_t;
        let patch_grid_h = h / self.patch_size_h;
        let patch_grid_w = w / self.patch_size_w;
        latents
            .reshape(&[
                b,
                patch_grid_f,
                patch_grid_h,
                patch_grid_w,
                c,
                self.patch_size_t,
                self.patch_size_h,
                self.patch_size_w,
            ])?
            .permute([0, 4, 1, 5, 2, 6, 3, 7])?
            .reshape((b, c, f, h, w))
            .map_err(Into::into)
    }

    pub fn get_patch_grid_bounds(self, shape: VideoLatentShape, device: &Device) -> Result<Tensor> {
        let patch_grid_f = shape.frames / self.patch_size_t;
        let patch_grid_h = shape.height / self.patch_size_h;
        let patch_grid_w = shape.width / self.patch_size_w;
        let token_count = patch_grid_f * patch_grid_h * patch_grid_w;
        let mut data = Vec::with_capacity(shape.batch * 3 * token_count * 2);
        for _batch in 0..shape.batch {
            for dim in 0..3 {
                for frame in 0..patch_grid_f {
                    let start_f = frame * self.patch_size_t;
                    let end_f = start_f + self.patch_size_t;
                    for height in 0..patch_grid_h {
                        let start_h = height * self.patch_size_h;
                        let end_h = start_h + self.patch_size_h;
                        for width in 0..patch_grid_w {
                            let start_w = width * self.patch_size_w;
                            let end_w = start_w + self.patch_size_w;
                            let (start, end) = match dim {
                                0 => (start_f as f32, end_f as f32),
                                1 => (start_h as f32, end_h as f32),
                                _ => (start_w as f32, end_w as f32),
                            };
                            data.push(start);
                            data.push(end);
                        }
                    }
                }
            }
        }
        Ok(Tensor::from_vec(
            data,
            (shape.batch, 3, token_count, 2),
            device,
        )?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioPatchifier {
    sample_rate: usize,
    hop_length: usize,
    audio_latent_downsample_factor: usize,
    is_causal: bool,
    shift: usize,
}

impl AudioPatchifier {
    pub fn new(
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
        is_causal: bool,
        shift: usize,
    ) -> Self {
        Self {
            sample_rate,
            hop_length,
            audio_latent_downsample_factor,
            is_causal,
            shift,
        }
    }

    #[allow(dead_code)]
    pub fn get_token_count(self, shape: AudioLatentShape) -> usize {
        shape.frames
    }

    pub fn patchify(self, latents: &Tensor) -> Result<Tensor> {
        let (b, c, t, f) = latents.dims4()?;
        latents
            .permute((0, 2, 1, 3))?
            .reshape((b, t, c * f))
            .map_err(Into::into)
    }

    pub fn unpatchify(self, latents: &Tensor, output_shape: AudioLatentShape) -> Result<Tensor> {
        latents
            .reshape((
                output_shape.batch,
                output_shape.frames,
                output_shape.channels,
                output_shape.mel_bins,
            ))?
            .permute((0, 2, 1, 3))
            .map_err(Into::into)
    }

    pub fn get_patch_grid_bounds(self, shape: AudioLatentShape, device: &Device) -> Result<Tensor> {
        let mut data = Vec::with_capacity(shape.batch * shape.frames * 2);
        for _batch in 0..shape.batch {
            for frame in 0..shape.frames {
                let start = self.audio_latent_time_seconds(self.shift + frame);
                let end = self.audio_latent_time_seconds(self.shift + frame + 1);
                data.push(start);
                data.push(end);
            }
        }
        Ok(Tensor::from_vec(
            data,
            (shape.batch, 1, shape.frames, 2),
            device,
        )?)
    }

    fn audio_latent_time_seconds(self, latent_index: usize) -> f32 {
        let mel_index = latent_index * self.audio_latent_downsample_factor;
        let adjusted = if self.is_causal {
            mel_index
                .saturating_add(1)
                .saturating_sub(self.audio_latent_downsample_factor)
        } else {
            mel_index
        };
        adjusted as f32 * self.hop_length as f32 / self.sample_rate as f32
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{AudioPatchifier, VideoLatentPatchifier};
    use crate::ltx2::model::{AudioLatentShape, VideoLatentShape};

    #[test]
    fn video_patchifier_round_trips_latents() {
        let device = Device::Cpu;
        let shape = VideoLatentShape {
            batch: 1,
            channels: 2,
            frames: 3,
            height: 2,
            width: 2,
        };
        let latents = Tensor::arange(0f32, 24f32, &device)
            .unwrap()
            .reshape((1, 2, 3, 2, 2))
            .unwrap();

        let patchifier = VideoLatentPatchifier::new(1);
        let patched = patchifier.patchify(&latents).unwrap();
        assert_eq!(patched.dims3().unwrap(), (1, 12, 2));

        let roundtrip = patchifier.unpatchify(&patched, shape).unwrap();
        assert_eq!(
            roundtrip.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            latents.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn audio_patchifier_round_trips_latents() {
        let device = Device::Cpu;
        let shape = AudioLatentShape {
            batch: 1,
            channels: 2,
            frames: 4,
            mel_bins: 3,
        };
        let latents = Tensor::arange(0f32, 24f32, &device)
            .unwrap()
            .reshape((1, 2, 4, 3))
            .unwrap();

        let patchifier = AudioPatchifier::new(16_000, 160, 4, true, 0);
        let patched = patchifier.patchify(&latents).unwrap();
        assert_eq!(patched.dims3().unwrap(), (1, 4, 6));

        let roundtrip = patchifier.unpatchify(&patched, shape).unwrap();
        assert_eq!(
            roundtrip.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            latents.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn video_patch_grid_bounds_match_3d_token_order() {
        let device = Device::Cpu;
        let bounds = VideoLatentPatchifier::new(1)
            .get_patch_grid_bounds(
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
        let flat = bounds.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&flat[..6], &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn audio_patch_grid_bounds_encode_seconds() {
        let device = Device::Cpu;
        let bounds = AudioPatchifier::new(16_000, 160, 4, true, 0)
            .get_patch_grid_bounds(
                AudioLatentShape {
                    batch: 1,
                    channels: 8,
                    frames: 3,
                    mel_bins: 16,
                },
                &device,
            )
            .unwrap();

        assert_eq!(bounds.dims4().unwrap(), (1, 1, 3, 2));
        let flat = bounds.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat, vec![0.0, 0.01, 0.01, 0.05, 0.05, 0.09]);
    }
}
