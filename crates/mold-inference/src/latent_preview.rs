//! Live denoise previews: a cheap latent→RGB linear projection emitted as a
//! small PNG through the progress callback after each denoise step, so
//! clients can show the image forming in real time.
//!
//! No VAE involved — the sequential pipelines drop the transformer before
//! the VAE ever loads, so a real decode mid-denoise is impossible under the
//! VRAM model. Instead each latent channel contributes linearly to RGB via
//! per-family factor matrices (the community-standard constants shared by
//! ComfyUI/diffusers previewers). The preview is latent-resolution
//! (width/8 × height/8); clients upscale it.
//!
//! Disable with `MOLD_STEP_PREVIEW=0`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, IndexOp, Tensor};

use crate::progress::{ProgressEvent, ProgressReporter};

/// Minimum wall-clock spacing between previews. Fast CUDA steps coalesce;
/// slow Metal steps (seconds each) get one preview per step.
const MIN_INTERVAL: Duration = Duration::from_millis(700);

/// FLUX.1 16-channel latent→RGB factors (row per channel).
const FLUX1_FACTORS: [[f32; 3]; 16] = [
    [-0.0346, 0.0244, 0.0681],
    [0.0034, 0.0210, 0.0687],
    [0.0275, -0.0668, -0.0433],
    [-0.0174, 0.0160, 0.0617],
    [0.0859, 0.0721, 0.0329],
    [0.0004, 0.0383, 0.0115],
    [0.0405, 0.0861, 0.0915],
    [-0.0236, -0.0185, -0.0259],
    [-0.0245, 0.0250, 0.1180],
    [0.1008, 0.0755, -0.0421],
    [-0.0515, 0.0201, 0.0011],
    [0.0428, -0.0012, -0.0036],
    [0.0817, 0.0765, 0.0749],
    [-0.1264, -0.0522, -0.1103],
    [-0.0280, -0.0881, -0.0499],
    [-0.1262, -0.0982, -0.0778],
];
const FLUX1_BIAS: [f32; 3] = [-0.0329, -0.0718, -0.0851];

/// Flux.2 32-channel latent→RGB factors.
const FLUX2_FACTORS: [[f32; 3]; 32] = [
    [0.0058, 0.0113, 0.0073],
    [0.0495, 0.0443, 0.0836],
    [-0.0099, 0.0096, 0.0644],
    [0.2144, 0.3009, 0.3652],
    [0.0166, -0.0039, -0.0054],
    [0.0157, 0.0103, -0.0160],
    [-0.0398, 0.0902, -0.0235],
    [-0.0052, 0.0095, 0.0109],
    [-0.3527, -0.2712, -0.1666],
    [-0.0301, -0.0356, -0.0180],
    [-0.0107, 0.0078, 0.0013],
    [0.0746, 0.0090, -0.0941],
    [0.0156, 0.0169, 0.0070],
    [-0.0034, -0.0040, -0.0114],
    [0.0032, 0.0181, 0.0080],
    [-0.0939, -0.0008, 0.0186],
    [0.0018, 0.0043, 0.0104],
    [0.0284, 0.0056, -0.0127],
    [-0.0024, -0.0022, -0.0030],
    [0.1207, -0.0026, 0.0065],
    [0.0128, 0.0101, 0.0142],
    [0.0137, -0.0072, -0.0007],
    [0.0095, 0.0092, -0.0059],
    [0.0000, -0.0077, -0.0049],
    [-0.0465, -0.0204, -0.0312],
    [0.0095, 0.0012, -0.0066],
    [0.0290, -0.0034, 0.0025],
    [0.0220, 0.0169, -0.0048],
    [-0.0332, -0.0457, -0.0468],
    [-0.0085, 0.0389, 0.0609],
    [-0.0076, 0.0003, -0.0043],
    [-0.0111, -0.0460, -0.0614],
];
const FLUX2_BIAS: [f32; 3] = [-0.0329, -0.0718, -0.0851];

type UnpackFn = Box<dyn Fn(&Tensor) -> anyhow::Result<Tensor> + Send + Sync>;

pub struct LatentPreviewer {
    factors: &'static [[f32; 3]],
    bias: [f32; 3],
    /// Family-specific transform from the sampler's working latent to a
    /// spatial `(B, C, H, W)` latent (unpack packed sequences, squeeze the
    /// video axis, …).
    to_spatial: UnpackFn,
    last_emit: Mutex<Option<Instant>>,
    enabled: bool,
}

fn preview_enabled() -> bool {
    !matches!(
        std::env::var("MOLD_STEP_PREVIEW").as_deref(),
        Ok("0") | Ok("false")
    )
}

impl LatentPreviewer {
    /// FLUX.1 family — packed `(B, seq, 64)` latents, 16 channels spatial.
    pub fn flux1(height: usize, width: usize) -> Self {
        Self {
            factors: &FLUX1_FACTORS,
            bias: FLUX1_BIAS,
            to_spatial: Box::new(move |t| {
                candle_transformers::models::flux::sampling::unpack(t, height, width)
                    .map_err(Into::into)
            }),
            last_emit: Mutex::new(None),
            enabled: preview_enabled(),
        }
    }

    /// Flux.2 family — packed `(B, seq, 128)` latents, 32 channels spatial.
    pub fn flux2(height: usize, width: usize) -> Self {
        Self {
            factors: &FLUX2_FACTORS,
            bias: FLUX2_BIAS,
            to_spatial: Box::new(move |t| {
                crate::flux2::sampling::unpack(t, height, width).map_err(Into::into)
            }),
            last_emit: Mutex::new(None),
            enabled: preview_enabled(),
        }
    }

    /// Z-Image — spatial 16-channel latents (FLUX.1 AE), either `(B, C, H, W)`
    /// or with the video axis still present `(B, C, 1, H, W)`.
    pub fn zimage() -> Self {
        Self {
            factors: &FLUX1_FACTORS,
            bias: FLUX1_BIAS,
            to_spatial: Box::new(|t| {
                if t.dims().len() == 5 {
                    t.squeeze(2).map_err(Into::into)
                } else {
                    Ok(t.clone())
                }
            }),
            last_emit: Mutex::new(None),
            enabled: preview_enabled(),
        }
    }

    /// True when the next `maybe_emit` for this step would render — lets
    /// callers skip building the x₀-estimate tensor for throttled steps.
    pub fn due(&self, step: usize, total: usize) -> bool {
        if !self.enabled {
            return false;
        }
        let last = self.last_emit.lock().expect("preview throttle mutex");
        step >= total || last.is_none_or(|t| t.elapsed() >= MIN_INTERVAL)
    }

    /// Emit a preview if due. Never fails the generation: conversion or
    /// encoding problems log a warning and are skipped. The final step is
    /// always emitted so the last preview matches the finished latent.
    pub fn maybe_emit(
        &self,
        progress: &ProgressReporter,
        latent: &Tensor,
        step: usize,
        total: usize,
    ) {
        if !self.due(step, total) {
            return;
        }
        {
            let mut last = self.last_emit.lock().expect("preview throttle mutex");
            *last = Some(Instant::now());
        }
        match self.render_png(latent) {
            Ok(png) => progress.emit(ProgressEvent::Preview {
                image_png: Arc::new(png),
                step,
                total,
            }),
            Err(e) => tracing::warn!("skipping denoise preview: {e:#}"),
        }
    }

    /// Project the latent to a latent-resolution RGB PNG.
    fn render_png(&self, latent: &Tensor) -> anyhow::Result<Vec<u8>> {
        let spatial = (self.to_spatial)(latent)?;
        let spatial = spatial
            .i(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let (channels, h, w) = spatial.dims3()?;
        anyhow::ensure!(
            channels == self.factors.len(),
            "latent has {channels} channels, preview factors expect {}",
            self.factors.len()
        );
        let data: Vec<f32> = spatial.flatten_all()?.to_vec1()?;
        let plane = h * w;
        let mut rgb = vec![0u8; plane * 3];
        for px in 0..plane {
            let (mut r, mut g, mut b) = (self.bias[0], self.bias[1], self.bias[2]);
            for (c, f) in self.factors.iter().enumerate() {
                let v = data[c * plane + px];
                r += v * f[0];
                g += v * f[1];
                b += v * f[2];
            }
            rgb[px * 3] = (((r + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0) as u8;
            rgb[px * 3 + 1] = (((g + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0) as u8;
            rgb[px * 3 + 2] = (((b + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0) as u8;
        }
        let img = ::image::RgbImage::from_raw(w as u32, h as u32, rgb)
            .ok_or_else(|| anyhow::anyhow!("preview buffer size mismatch"))?;
        let mut png = Vec::new();
        img.write_with_encoder(::image::codecs::png::PngEncoder::new(&mut png))?;
        Ok(png)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::sync::Arc as StdArc;

    fn capturing_reporter() -> (
        ProgressReporter,
        StdArc<std::sync::Mutex<Vec<ProgressEvent>>>,
    ) {
        let log = StdArc::new(std::sync::Mutex::new(Vec::new()));
        let sink = log.clone();
        let mut reporter = ProgressReporter::default();
        reporter.set_callback(Box::new(move |e| sink.lock().unwrap().push(e)));
        (reporter, log)
    }

    /// A previewer over already-spatial 16-channel latents (no unpack).
    fn spatial_previewer() -> LatentPreviewer {
        LatentPreviewer {
            factors: &FLUX1_FACTORS,
            bias: FLUX1_BIAS,
            to_spatial: Box::new(|t| Ok(t.clone())),
            last_emit: Mutex::new(None),
            enabled: true,
        }
    }

    #[test]
    fn renders_latent_to_png_at_latent_resolution() {
        let previewer = spatial_previewer();
        let latent = Tensor::zeros((1, 16, 4, 6), DType::F32, &Device::Cpu).unwrap();
        let png = previewer.render_png(&latent).unwrap();
        let decoded = ::image::load_from_memory(&png).unwrap();
        assert_eq!(decoded.width(), 6);
        assert_eq!(decoded.height(), 4);
        // All-zero latent = bias only → (bias+1)/2*255 per channel.
        let px = decoded.to_rgb8().get_pixel(0, 0).0;
        assert_eq!(px[0], (((FLUX1_BIAS[0] + 1.0) / 2.0) * 255.0) as u8);
    }

    #[test]
    fn rejects_channel_mismatch() {
        let previewer = spatial_previewer();
        let latent = Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(previewer.render_png(&latent).is_err());
    }

    #[test]
    fn throttles_but_always_emits_final_step() {
        let previewer = spatial_previewer();
        let (reporter, log) = capturing_reporter();
        let latent = Tensor::zeros((1, 16, 2, 2), DType::F32, &Device::Cpu).unwrap();

        previewer.maybe_emit(&reporter, &latent, 1, 10); // first: due
        previewer.maybe_emit(&reporter, &latent, 2, 10); // within interval: skipped
        previewer.maybe_emit(&reporter, &latent, 10, 10); // final: always
        let events = log.lock().unwrap();
        let previews = events
            .iter()
            .filter(|e| matches!(e, ProgressEvent::Preview { .. }))
            .count();
        assert_eq!(previews, 2, "first + final, middle throttled");
        match &events[0] {
            ProgressEvent::Preview {
                step,
                total,
                image_png,
            } => {
                assert_eq!((*step, *total), (1, 10));
                assert!(image_png.starts_with(b"\x89PNG"));
            }
            other => panic!("expected preview, got {other:?}"),
        }
    }

    #[test]
    fn disabled_previewer_emits_nothing() {
        let mut previewer = spatial_previewer();
        previewer.enabled = false;
        let (reporter, log) = capturing_reporter();
        let latent = Tensor::zeros((1, 16, 2, 2), DType::F32, &Device::Cpu).unwrap();
        previewer.maybe_emit(&reporter, &latent, 10, 10);
        assert!(log.lock().unwrap().is_empty());
    }
}
