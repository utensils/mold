use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::flux::{self, BlockHook, WithForward};
use std::time::Instant;

use crate::flux::offload::OffloadedFluxTransformer;
use crate::flux::pulid::PulidRuntime;
use crate::flux::quantized_transformer::QuantizedFluxTransformer;
use crate::img_utils::InpaintContext;
use crate::progress::{ProgressEvent, ProgressReporter};

/// BF16, quantized (GGUF), or offloaded FLUX transformer.
///
/// `QuantizedBypass` is the mold-owned GGUF path that supports
/// bypass-mode LoRA — it never touches base weights, applying LoRA
/// deltas at forward time instead. A merged GGUF LoRA also uses this Mold-owned
/// variant with no runtime registry; ordinary unmodified GGUF loads retain the
/// upstream `Quantized` variant.
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
        pulid: Option<PulidRuntime<'_>>,
    ) -> Result<Tensor> {
        let b_sz = img.dim(0)?;
        let dev = img.device();
        let guidance_tensor = Tensor::full(guidance as f32, b_sz, dev)?;
        let mut img = img.clone();
        let total_steps = timesteps.len().saturating_sub(1);

        for (step, window) in timesteps.windows(2).enumerate() {
            progress.checkpoint()?;
            let step_start = Instant::now();
            let (t_curr, t_prev) = match window {
                [a, b] => (a, b),
                _ => continue,
            };
            let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
            // The identity gate. A step before `id_start_step`, or an
            // effective `id_weight` of 0, yields no hook — and every arm below
            // answers `None` by calling the variant's ordinary `forward`, the
            // exact call a render with no identity request makes. Bit-identity
            // is therefore structural, not a numerical coincidence, and it
            // holds on the *same* transformer route rather than requiring a
            // separate no-PuLID load.
            let hook = pulid.and_then(|runtime| runtime.hook_for_step(step));
            let hook = hook.as_ref().map(|hook| hook as &dyn BlockHook);
            let pred = match (self, hook) {
                (Self::BF16(m), None) => m.forward(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                )?,
                (Self::BF16(m), Some(hook)) => m.forward_with_hook(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                    hook,
                )?,
                (Self::Quantized(m), None) => m.forward(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                )?,
                (Self::Quantized(m), Some(hook)) => m.forward_with_hook(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                    hook,
                )?,
                (Self::QuantizedBypass(m), hook) => m.forward_with_hook(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                    hook,
                )?,
                (Self::Offloaded(m), hook) => m.forward_with_hook(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                    hook,
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
        progress.checkpoint()?;
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
