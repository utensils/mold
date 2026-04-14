#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoPixelShape {
    pub batch: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub fps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatioTemporalScaleFactors {
    pub time: usize,
    pub width: usize,
    pub height: usize,
}

impl Default for SpatioTemporalScaleFactors {
    fn default() -> Self {
        Self {
            time: 8,
            width: 32,
            height: 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoLatentShape {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl VideoLatentShape {
    #[allow(dead_code)]
    pub fn token_count(self) -> usize {
        self.frames * self.height * self.width
    }

    #[allow(dead_code)]
    pub fn mask_shape(self) -> Self {
        Self {
            channels: 1,
            ..self
        }
    }

    pub fn from_pixel_shape(
        shape: VideoPixelShape,
        latent_channels: usize,
        scale_factors: SpatioTemporalScaleFactors,
    ) -> Self {
        Self {
            batch: shape.batch,
            channels: latent_channels,
            frames: ((shape.frames - 1) / scale_factors.time) + 1,
            height: shape.height / scale_factors.height,
            width: shape.width / scale_factors.width,
        }
    }

    pub fn upscale(self, scale_factors: SpatioTemporalScaleFactors) -> Self {
        Self {
            channels: 3,
            frames: ((self.frames - 1) * scale_factors.time) + 1,
            height: self.height * scale_factors.height,
            width: self.width * scale_factors.width,
            ..self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioLatentShape {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub mel_bins: usize,
}

impl AudioLatentShape {
    #[allow(dead_code)]
    pub fn token_count(self) -> usize {
        self.frames
    }

    #[allow(dead_code)]
    pub fn mask_shape(self) -> Self {
        Self {
            channels: 1,
            mel_bins: 1,
            ..self
        }
    }

    pub fn from_duration(
        batch: usize,
        duration_seconds: f32,
        channels: usize,
        mel_bins: usize,
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
    ) -> Self {
        let latents_per_second =
            sample_rate as f32 / hop_length as f32 / audio_latent_downsample_factor as f32;
        Self {
            batch,
            channels,
            frames: (duration_seconds * latents_per_second).round() as usize,
            mel_bins,
        }
    }

    pub fn from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: usize,
        mel_bins: usize,
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
    ) -> Self {
        Self::from_duration(
            shape.batch,
            shape.frames as f32 / shape.fps,
            channels,
            mel_bins,
            sample_rate,
            hop_length,
            audio_latent_downsample_factor,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape};

    #[test]
    fn video_latent_shape_from_pixel_shape_matches_ltx2_contract() {
        let shape = VideoLatentShape::from_pixel_shape(
            VideoPixelShape {
                batch: 1,
                frames: 121,
                height: 704,
                width: 1216,
                fps: 24.0,
            },
            128,
            SpatioTemporalScaleFactors::default(),
        );

        assert_eq!(shape.batch, 1);
        assert_eq!(shape.channels, 128);
        assert_eq!(shape.frames, 16);
        assert_eq!(shape.height, 22);
        assert_eq!(shape.width, 38);
        assert_eq!(shape.token_count(), 16 * 22 * 38);
    }

    #[test]
    fn video_latent_shape_upscale_restores_pixel_grid() {
        let latent = VideoLatentShape {
            batch: 1,
            channels: 128,
            frames: 16,
            height: 22,
            width: 38,
        };
        let upscaled = latent.upscale(SpatioTemporalScaleFactors::default());

        assert_eq!(upscaled.channels, 3);
        assert_eq!(upscaled.frames, 121);
        assert_eq!(upscaled.height, 704);
        assert_eq!(upscaled.width, 1216);
    }

    #[test]
    fn audio_latent_shape_from_duration_rounds_to_expected_frame_count() {
        let shape = AudioLatentShape::from_duration(1, 5.0, 8, 16, 16_000, 160, 4);
        assert_eq!(shape.batch, 1);
        assert_eq!(shape.channels, 8);
        assert_eq!(shape.mel_bins, 16);
        assert_eq!(shape.frames, 125);
        assert_eq!(shape.token_count(), 125);
    }

    #[test]
    fn audio_latent_shape_tracks_video_duration() {
        let shape = AudioLatentShape::from_video_pixel_shape(
            VideoPixelShape {
                batch: 1,
                frames: 121,
                height: 704,
                width: 1216,
                fps: 24.0,
            },
            8,
            16,
            16_000,
            160,
            4,
        );

        assert_eq!(shape.frames, 126);
    }
}
