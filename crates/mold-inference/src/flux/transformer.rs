use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::flux::{self, WithForward};
use std::time::Instant;

use crate::flux::offload::OffloadedFluxTransformer;
use crate::img_utils::InpaintContext;
use crate::progress::{ProgressEvent, ProgressReporter};

/// BF16, quantized (GGUF), or offloaded FLUX transformer.
#[allow(clippy::large_enum_variant)]
pub(crate) enum FluxTransformer {
    BF16(flux::model::Flux),
    Quantized(flux::quantized_model::Flux),
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
            img = (img + pred * (t_prev - t_curr))?;

            // Inpainting: blend preserved regions back at current noise level
            if let Some(ctx) = inpaint_ctx {
                let t = *t_prev;
                // Re-noise original latents to current timestep (flow-matching schedule)
                let noised_original = ((&ctx.original_latents * (1.0 - t))? + (&ctx.noise * t)?)?;
                // mask=1 -> repaint (use denoised), mask=0 -> preserve (use noised original)
                img = ((&ctx.mask * &img)? + (&(1.0 - &ctx.mask)? * &noised_original)?)?;
            }

            progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total: total_steps,
                elapsed: step_start.elapsed(),
            });
        }
        Ok(img)
    }
}
