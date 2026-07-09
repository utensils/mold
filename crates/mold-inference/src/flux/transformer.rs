use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::flux::{self, WithForward};
use std::time::Instant;

use crate::flux::offload::OffloadedFluxTransformer;
use crate::flux::quantized_transformer::QuantizedFluxTransformer;
use crate::img_utils::InpaintContext;
use crate::progress::{ProgressEvent, ProgressReporter};

/// BF16, quantized (GGUF), or offloaded FLUX transformer.
///
/// `QuantizedBypass` is the mold-owned GGUF path that supports
/// bypass-mode LoRA — it never touches base weights, applying LoRA
/// deltas at forward time instead. The legacy `Quantized` variant
/// (upstream `candle_transformers::flux::quantized_model`) is the
/// `MOLD_LORA_BYPASS=off` fallback.
#[allow(clippy::large_enum_variant)]
pub(crate) enum FluxTransformer {
    BF16(flux::model::Flux),
    Quantized(flux::quantized_model::Flux),
    QuantizedBypass(QuantizedFluxTransformer),
    /// Block-level offloading: blocks on CPU, streamed to GPU one at a time.
    Offloaded(OffloadedFluxTransformer),
}

impl FluxTransformer {
    /// Run the denoising loop with per-step progress reporting.
    ///
    /// Inlines the candle `flux::sampling::denoise` loop so we can emit
    /// `DenoiseStep` events for the CLI progress bar.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        vec_: &Tensor,
        timesteps: &[f64],
        guidance: f64,
        progress: &ProgressReporter,
        inpaint_ctx: Option<&InpaintContext>,
        preview: Option<&crate::latent_preview::LatentPreviewer>,
    ) -> Result<Tensor> {
        let b_sz = img.dim(0)?;
        let dev = img.device();
        let guidance_tensor = Tensor::full(guidance as f32, b_sz, dev)?;
        let mut img = img.clone();
        let total_steps = timesteps.len().saturating_sub(1);

        for (step, window) in timesteps.windows(2).enumerate() {
            let step_start = Instant::now();
            let (t_curr, t_prev) = match window {
                [a, b] => (a, b),
                _ => continue,
            };
            let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
            let pred = match self {
                Self::BF16(m) => m.forward(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                )?,
                Self::Quantized(m) => m.forward(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                )?,
                Self::QuantizedBypass(m) => m.forward(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                )?,
                Self::Offloaded(m) => m.forward(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                )?,
            };
            img = (img + &pred * (t_prev - t_curr))?;

            // Inpainting: blend preserved regions back at current noise level
            if let Some(ctx) = inpaint_ctx {
                img = apply_flux_inpaint_step(&img, ctx, *t_prev)?;
            }

            progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total: total_steps,
                elapsed: step_start.elapsed(),
            });
            if let Some(previewer) = preview {
                if previewer.due(step + 1, total_steps) {
                    // Preview the predicted clean image (flow matching:
                    // x0 = x_t - t*v), not the still-noisy working latent —
                    // composition is visible from the first step.
                    match &img - &(&pred * *t_prev)? {
                        Ok(x0_est) => {
                            previewer.maybe_emit(progress, &x0_est, step + 1, total_steps)
                        }
                        Err(e) => tracing::warn!("skipping denoise preview: {e}"),
                    }
                }
            }
        }
        Ok(img)
    }
}

fn apply_flux_inpaint_step(img: &Tensor, ctx: &InpaintContext, timestep: f64) -> Result<Tensor> {
    crate::img2img::apply_flow_match_inpaint(img, ctx, timestep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn packed_inpaint_mask_broadcasts_across_flux_channels() {
        let device = Device::Cpu;
        let img = Tensor::ones((1, 4, 64), DType::F32, &device).unwrap();
        let ctx = InpaintContext {
            original_latents: Tensor::zeros((1, 4, 64), DType::F32, &device).unwrap(),
            mask: Tensor::ones((1, 4, 1), DType::F32, &device).unwrap(),
            noise: Tensor::zeros((1, 4, 64), DType::F32, &device).unwrap(),
        };

        let blended = apply_flux_inpaint_step(&img, &ctx, 0.0).unwrap();

        assert_eq!(blended.dims(), &[1, 4, 64]);
    }
}
