//! Live denoise previews: a cheap latent→RGB linear projection emitted as a
//! small PNG through the progress callback after each denoise step, so
//! clients can show the image forming in real time.
//!
//! No VAE involved — the sequential pipelines drop the transformer before
//! the VAE ever loads, so a real decode mid-denoise is impossible under the
//! VRAM model. Instead each latent channel contributes linearly to RGB via
//! per-family factor matrices (the community-standard constants shared by
//! ComfyUI/diffusers previewers). The preview is latent-resolution — the
//! family VAE's spatial compression decides how small: width/8 for most
//! families, width/16 for Wan 2.2 TI2V — and clients upscale it. Video
//! families project one representative frame of the `(B, C, T, H, W)`
//! working latent (the middle latent frame).
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

/// Wan 2.1 16-channel latent→RGB factors, for latents in the normalized
/// space the denoise loop works in (ComfyUI `comfy/latent_formats.py`,
/// `Wan21.latent_rgb_factors` — ComfyUI samples in the same
/// `(latent - mean) / std` space, applied by `process_in`).
const WAN21_FACTORS: [[f32; 3]; 16] = [
    [-0.1299, -0.1692, 0.2932],
    [0.0671, 0.0406, 0.0442],
    [0.3568, 0.2548, 0.1747],
    [0.0372, 0.2344, 0.1420],
    [0.0313, 0.0189, -0.0328],
    [0.0296, -0.0956, -0.0665],
    [-0.3477, -0.4059, -0.2925],
    [0.0166, 0.1902, 0.1975],
    [-0.0412, 0.0267, -0.1364],
    [-0.1293, 0.0740, 0.1636],
    [0.0680, 0.3019, 0.1128],
    [0.0032, 0.0581, 0.0639],
    [-0.1251, 0.0927, 0.1699],
    [0.0060, -0.0633, 0.0005],
    [0.3477, 0.2275, 0.2950],
    [0.1984, 0.0913, 0.1861],
];
const WAN21_BIAS: [f32; 3] = [-0.1835, -0.0868, -0.3360];

/// Wan 2.2 TI2V 48-channel latent→RGB factors (ComfyUI
/// `comfy/latent_formats.py`, `Wan22.latent_rgb_factors`).
const WAN22_FACTORS: [[f32; 3]; 48] = [
    [0.0119, 0.0103, 0.0046],
    [-0.1062, -0.0504, 0.0165],
    [0.0140, 0.0409, 0.0491],
    [-0.0813, -0.0677, 0.0607],
    [0.0656, 0.0851, 0.0808],
    [0.0264, 0.0463, 0.0912],
    [0.0295, 0.0326, 0.0590],
    [-0.0244, -0.0270, 0.0025],
    [0.0443, -0.0102, 0.0288],
    [-0.0465, -0.0090, -0.0205],
    [0.0359, 0.0236, 0.0082],
    [-0.0776, 0.0854, 0.1048],
    [0.0564, 0.0264, 0.0561],
    [0.0006, 0.0594, 0.0418],
    [-0.0319, -0.0542, -0.0637],
    [-0.0268, 0.0024, 0.0260],
    [0.0539, 0.0265, 0.0358],
    [-0.0359, -0.0312, -0.0287],
    [-0.0285, -0.1032, -0.1237],
    [0.1041, 0.0537, 0.0622],
    [-0.0086, -0.0374, -0.0051],
    [0.0390, 0.0670, 0.2863],
    [0.0069, 0.0144, 0.0082],
    [0.0006, -0.0167, 0.0079],
    [0.0313, -0.0574, -0.0232],
    [-0.1454, -0.0902, -0.0481],
    [0.0714, 0.0827, 0.0447],
    [-0.0304, -0.0574, -0.0196],
    [0.0401, 0.0384, 0.0204],
    [-0.0758, -0.0297, -0.0014],
    [0.0568, 0.1307, 0.1372],
    [-0.0055, -0.0310, -0.0380],
    [0.0239, -0.0305, 0.0325],
    [-0.0663, -0.0673, -0.0140],
    [-0.0416, -0.0047, -0.0023],
    [0.0166, 0.0112, -0.0093],
    [-0.0211, 0.0011, 0.0331],
    [0.1833, 0.1466, 0.2250],
    [-0.0368, 0.0370, 0.0295],
    [-0.3441, -0.3543, -0.2008],
    [-0.0479, -0.0489, -0.0420],
    [-0.0660, -0.0153, 0.0800],
    [-0.0101, 0.0068, 0.0156],
    [-0.0690, -0.0452, -0.0927],
    [-0.0145, 0.0041, 0.0015],
    [0.0421, 0.0451, 0.0373],
    [0.0504, -0.0483, -0.0356],
    [-0.0837, 0.0168, 0.0055],
];
const WAN22_BIAS: [f32; 3] = [0.0317, -0.0878, -0.1388];

type UnpackFn = Box<dyn Fn(&Tensor) -> anyhow::Result<Tensor> + Send + Sync>;

pub struct LatentPreviewer {
    factors: &'static [[f32; 3]],
    bias: [f32; 3],
    /// Family-specific transform from the sampler's working latent to a
    /// spatial `(B, C, H, W)` latent (unpack packed sequences, squeeze the
    /// video axis, …).
    to_spatial: UnpackFn,
    last_emit: Mutex<Option<Instant>>,
    /// [`MIN_INTERVAL`] in production; tests shrink it to observe every step.
    min_interval: Duration,
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
            min_interval: MIN_INTERVAL,
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
            min_interval: MIN_INTERVAL,
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
            min_interval: MIN_INTERVAL,
            enabled: preview_enabled(),
        }
    }

    /// Wan video — spatial `(B, C, T, H, W)` working latents in the VAE's
    /// normalized space. The factor table follows the checkpoint's latent
    /// channel count: 16 → Wan 2.1 (1.3B / 14B / A14B), 48 → Wan 2.2 TI2V.
    /// An unknown channel count has no table, so no previewer.
    ///
    /// `to_spatial` projects the middle latent frame — for T2V that is the
    /// most representative slice of the clip (ComfyUI previews frame 0; the
    /// middle frame is issue #791's deliberate choice).
    pub fn wan(z_dim: usize) -> Option<Self> {
        let (factors, bias): (&'static [[f32; 3]], [f32; 3]) = match z_dim {
            16 => (&WAN21_FACTORS, WAN21_BIAS),
            48 => (&WAN22_FACTORS, WAN22_BIAS),
            _ => return None,
        };
        Some(Self {
            factors,
            bias,
            to_spatial: Box::new(|t| {
                if t.dims().len() == 5 {
                    let frames = t.dim(2)?;
                    t.narrow(2, frames / 2, 1)?.squeeze(2).map_err(Into::into)
                } else {
                    Ok(t.clone())
                }
            }),
            last_emit: Mutex::new(None),
            min_interval: MIN_INTERVAL,
            enabled: preview_enabled(),
        })
    }

    /// Test-only: ignore `MOLD_STEP_PREVIEW` so emission tests stay hermetic.
    #[cfg(test)]
    pub(crate) fn force_enabled(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Test-only: override the throttle so a fast CPU loop still emits every
    /// step — the intermediate steps are where preview math can go wrong.
    #[cfg(test)]
    pub(crate) fn with_min_interval(mut self, interval: Duration) -> Self {
        self.min_interval = interval;
        self
    }

    /// True when the next `maybe_emit` for this step would render — lets
    /// callers skip building the x₀-estimate tensor for throttled steps.
    pub fn due(&self, step: usize, total: usize) -> bool {
        if !self.enabled {
            return false;
        }
        let last = self.last_emit.lock().expect("preview throttle mutex");
        step >= total || last.is_none_or(|t| t.elapsed() >= self.min_interval)
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
            min_interval: MIN_INTERVAL,
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

    /// Decode one pixel of a rendered PNG.
    fn first_pixel(png: &[u8]) -> [u8; 3] {
        ::image::load_from_memory(png)
            .unwrap()
            .to_rgb8()
            .get_pixel(0, 0)
            .0
    }

    /// The same `(v + 1) / 2 * 255` mapping `render_png` applies.
    fn to_u8(v: f32) -> u8 {
        (((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0) as u8
    }

    /// The Wan constructor must pick the factor table from the checkpoint's
    /// latent channel count: 16 → Wan 2.1, 48 → Wan 2.2 TI2V. An all-zero
    /// latent renders each table's own bias color, which is how the test can
    /// see which table was selected (the biases differ between generations).
    #[test]
    fn wan_selects_the_factor_table_by_z_dim() {
        assert!(
            LatentPreviewer::wan(32).is_none(),
            "only the 16- and 48-channel Wan VAEs have preview tables"
        );

        let wan21 = LatentPreviewer::wan(16).expect("Wan 2.1 table");
        let latent = Tensor::zeros((1, 16, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let px = first_pixel(&wan21.render_png(&latent).unwrap());
        // ComfyUI comfy/latent_formats.py `Wan21.latent_rgb_factors_bias`.
        assert_eq!(px, [to_u8(-0.1835), to_u8(-0.0868), to_u8(-0.3360)]);

        let wan22 = LatentPreviewer::wan(48).expect("Wan 2.2 table");
        let latent = Tensor::zeros((1, 48, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let px = first_pixel(&wan22.render_png(&latent).unwrap());
        // ComfyUI comfy/latent_formats.py `Wan22.latent_rgb_factors_bias`.
        assert_eq!(px, [to_u8(0.0317), to_u8(-0.0878), to_u8(-0.1388)]);

        // Each generation rejects the other's channel count instead of
        // projecting through the wrong table.
        let wrong = Tensor::zeros((1, 48, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(wan21.render_png(&wrong).is_err());
        let wrong = Tensor::zeros((1, 16, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(wan22.render_png(&wrong).is_err());
    }

    /// The `(B, C, T, H, W)` working latent squeezes to the middle frame —
    /// decoy values on the first and last frame must not reach the preview.
    #[test]
    fn wan_to_spatial_previews_the_middle_latent_frame() {
        let wan21 = LatentPreviewer::wan(16).unwrap();
        // Channel-major layout: index = c * (T*H*W) + t.
        let mut data = vec![0f32; 16 * 5];
        data[0] = -1.0; // channel 0, frame 0 (decoy)
        data[2] = 1.0; // channel 0, frame 2 (the middle frame)
        data[4] = -1.0; // channel 0, frame 4 (decoy)
        let latent = Tensor::from_vec(data, (1, 16, 5, 1, 1), &Device::Cpu).unwrap();
        let px = first_pixel(&wan21.render_png(&latent).unwrap());
        // bias + factors[0]: the middle frame's channel 0 is exactly 1.
        assert_eq!(
            px,
            [
                to_u8(-0.1835 + -0.1299),
                to_u8(-0.0868 + -0.1692),
                to_u8(-0.3360 + 0.2932),
            ]
        );
    }
}
