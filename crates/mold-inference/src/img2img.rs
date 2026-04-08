use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub(crate) struct PreparedFlowMatchImg2Img {
    pub(crate) initial_latents: Tensor,
    pub(crate) inpaint_ctx: Option<crate::img_utils::InpaintContext>,
}

/// Convert an img2img strength into the number of denoising steps to keep.
///
/// This matches the scheduler semantics used by diffusers img2img pipelines:
/// `floor(total_steps * strength)`, clamped to `total_steps`.
pub(crate) fn img2img_effective_steps(total_steps: usize, strength: f64) -> usize {
    ((total_steps as f64 * strength).floor() as usize).min(total_steps)
}

/// Convert an img2img strength into the schedule start index.
///
/// `0` means full txt2img-style denoising, `total_steps` means preserve the
/// encoded image without any denoising steps.
pub(crate) fn img2img_start_index(total_steps: usize, strength: f64) -> usize {
    total_steps.saturating_sub(img2img_effective_steps(total_steps, strength))
}

/// Trim a full scheduler sequence down to the img2img tail selected by `strength`.
pub(crate) fn trim_schedule_tail<T: Clone>(
    schedule: &[T],
    total_steps: usize,
    strength: f64,
) -> (Vec<T>, usize) {
    let start_index = img2img_start_index(total_steps, strength);
    (schedule[start_index..].to_vec(), start_index)
}

/// Flow-matching img2img interpolation: `(1 - sigma) * encoded + sigma * noise`.
pub(crate) fn flow_match_interpolate(
    encoded: &Tensor,
    noise: &Tensor,
    sigma: f64,
) -> Result<Tensor> {
    Ok(((encoded * (1.0 - sigma))? + (noise * sigma)?)?)
}

/// Prepare the initial flow-matching latents and optional inpaint context from encoded latents.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_flow_match_img2img(
    encoded: &Tensor,
    seed: u64,
    noise_shape: &[usize],
    sigma: f64,
    mask_image: Option<&[u8]>,
    latent_h: usize,
    latent_w: usize,
    device: &Device,
    dtype: DType,
) -> Result<PreparedFlowMatchImg2Img> {
    let noise = crate::engine::seeded_randn(seed, noise_shape, device, dtype)?;
    let inpaint_ctx = maybe_build_inpaint_context(
        mask_image, encoded, &noise, latent_h, latent_w, device, dtype,
    )?;
    let initial_latents = flow_match_interpolate(encoded, &noise, sigma)?;
    Ok(PreparedFlowMatchImg2Img {
        initial_latents,
        inpaint_ctx,
    })
}

/// Re-noise original latents at the current flow-matching sigma.
pub(crate) fn flow_match_noised_original(
    ctx: &crate::img_utils::InpaintContext,
    sigma: f64,
) -> Result<Tensor> {
    flow_match_interpolate(&ctx.original_latents, &ctx.noise, sigma)
}

/// Blend the current denoised latents with preserved regions from the source image.
///
/// `mask=1` keeps the newly denoised region, `mask=0` restores the preserved region.
pub(crate) fn blend_inpaint_latents(
    current: &Tensor,
    ctx: &crate::img_utils::InpaintContext,
    preserved: &Tensor,
) -> Result<Tensor> {
    let repaint = current.broadcast_mul(&ctx.mask)?;
    let preserve_mask = (1.0 - &ctx.mask)?;
    let preserve = preserved.broadcast_mul(&preserve_mask)?;
    Ok((repaint + preserve)?)
}

/// Apply one flow-matching inpaint blend step at the given sigma.
pub(crate) fn apply_flow_match_inpaint(
    current: &Tensor,
    ctx: &crate::img_utils::InpaintContext,
    sigma: f64,
) -> Result<Tensor> {
    let preserved = flow_match_noised_original(ctx, sigma)?;
    blend_inpaint_latents(current, ctx, &preserved)
}

/// Decode an optional inpaint mask into the shared per-step blending context.
pub(crate) fn maybe_build_inpaint_context(
    mask_image: Option<&[u8]>,
    original_latents: &Tensor,
    noise: &Tensor,
    latent_h: usize,
    latent_w: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<crate::img_utils::InpaintContext>> {
    let Some(mask_bytes) = mask_image else {
        return Ok(None);
    };
    let mask = crate::img_utils::decode_mask_image(mask_bytes, latent_h, latent_w, device, dtype)?;
    Ok(Some(crate::img_utils::InpaintContext {
        original_latents: original_latents.clone(),
        mask,
        noise: noise.clone(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use image::{DynamicImage, ImageBuffer, ImageFormat, Luma};
    use std::io::Cursor;

    fn encode_mask_png(pixel: u8) -> Vec<u8> {
        let img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_pixel(1, 1, Luma([pixel]));
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageLuma8(img)
            .write_to(&mut out, ImageFormat::Png)
            .expect("encode PNG");
        out.into_inner()
    }

    #[test]
    fn full_strength_keeps_all_steps() {
        assert_eq!(img2img_effective_steps(20, 1.0), 20);
        assert_eq!(img2img_start_index(20, 1.0), 0);
    }

    #[test]
    fn low_strength_keeps_tail_steps() {
        assert_eq!(img2img_effective_steps(20, 0.1), 2);
        assert_eq!(img2img_start_index(20, 0.1), 18);
    }

    #[test]
    fn floor_semantics_match_reference_behavior() {
        assert_eq!(img2img_effective_steps(20, 0.3), 6);
        assert_eq!(img2img_start_index(20, 0.3), 14);
        assert_eq!(img2img_effective_steps(20, 0.75), 15);
        assert_eq!(img2img_start_index(20, 0.75), 5);
    }

    #[test]
    fn tiny_strength_can_skip_all_steps() {
        assert_eq!(img2img_effective_steps(20, 0.01), 0);
        assert_eq!(img2img_start_index(20, 0.01), 20);
    }

    #[test]
    fn trim_schedule_tail_returns_selected_tail() {
        let schedule = vec![0.9, 0.7, 0.5, 0.3, 0.0];
        let (trimmed, start_index) = trim_schedule_tail(&schedule, 4, 0.5);
        assert_eq!(start_index, 2);
        assert_eq!(trimmed, vec![0.5, 0.3, 0.0]);
    }

    #[test]
    fn flow_match_interpolate_blends_encoded_and_noise() {
        let device = Device::Cpu;
        let encoded = Tensor::from_vec(vec![2.0f32, 4.0], 2, &device).unwrap();
        let noise = Tensor::from_vec(vec![10.0f32, 20.0], 2, &device).unwrap();

        let zero = flow_match_interpolate(&encoded, &noise, 0.0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let half = flow_match_interpolate(&encoded, &noise, 0.5)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let full = flow_match_interpolate(&encoded, &noise, 1.0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(zero, vec![2.0, 4.0]);
        assert_eq!(half, vec![6.0, 12.0]);
        assert_eq!(full, vec![10.0, 20.0]);
    }

    #[test]
    fn prepare_flow_match_img2img_uses_seeded_noise_and_optional_mask() {
        let device = Device::Cpu;
        let encoded = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2, 1, 1), &device).unwrap();
        let mask = encode_mask_png(255);

        let prepared = prepare_flow_match_img2img(
            &encoded,
            123,
            &[1, 2, 1, 1],
            0.25,
            Some(&mask),
            1,
            1,
            &device,
            DType::F32,
        )
        .unwrap();

        let expected_noise = crate::engine::seeded_randn(123, &[1, 2, 1, 1], &device, DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual_noise = prepared
            .inpaint_ctx
            .as_ref()
            .unwrap()
            .noise
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual_noise, expected_noise);

        let expected_initial = flow_match_interpolate(
            &encoded,
            &crate::engine::seeded_randn(123, &[1, 2, 1, 1], &device, DType::F32).unwrap(),
            0.25,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let actual_initial = prepared
            .initial_latents
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual_initial, expected_initial);
        assert!(prepared.inpaint_ctx.is_some());
    }

    #[test]
    fn maybe_build_inpaint_context_returns_none_without_mask() {
        let device = Device::Cpu;
        let original = Tensor::zeros((1, 2, 1, 1), DType::F32, &device).unwrap();
        let noise = Tensor::ones((1, 2, 1, 1), DType::F32, &device).unwrap();

        let ctx = maybe_build_inpaint_context(None, &original, &noise, 1, 1, &device, DType::F32)
            .unwrap();
        assert!(ctx.is_none());
    }

    #[test]
    fn maybe_build_inpaint_context_decodes_mask_and_clones_inputs() {
        let device = Device::Cpu;
        let original = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2, 1, 1), &device).unwrap();
        let noise = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2, 1, 1), &device).unwrap();
        let mask = encode_mask_png(255);

        let ctx =
            maybe_build_inpaint_context(Some(&mask), &original, &noise, 1, 1, &device, DType::F32)
                .unwrap()
                .unwrap();

        assert_eq!(ctx.original_latents.dims(), &[1, 2, 1, 1]);
        assert_eq!(
            ctx.original_latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(ctx.noise.dims(), &[1, 2, 1, 1]);
        assert_eq!(
            ctx.noise.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![3.0, 4.0]
        );
        assert_eq!(ctx.mask.dims(), &[1, 1, 1, 1]);
        assert_eq!(
            ctx.mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0]
        );
    }

    #[test]
    fn flow_match_noised_original_uses_source_and_noise_mix() {
        let device = Device::Cpu;
        let ctx = crate::img_utils::InpaintContext {
            original_latents: Tensor::from_vec(vec![2.0f32, 4.0], (1, 2, 1, 1), &device).unwrap(),
            mask: Tensor::ones((1, 1, 1, 1), DType::F32, &device).unwrap(),
            noise: Tensor::from_vec(vec![10.0f32, 20.0], (1, 2, 1, 1), &device).unwrap(),
        };

        let preserved = flow_match_noised_original(&ctx, 0.25)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(preserved, vec![4.0, 8.0]);
    }

    #[test]
    fn blend_inpaint_latents_respects_mask() {
        let device = Device::Cpu;
        let current = Tensor::from_vec(vec![100.0f32, 200.0], (1, 2, 1, 1), &device).unwrap();
        let preserved = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2, 1, 1), &device).unwrap();
        let ctx = crate::img_utils::InpaintContext {
            original_latents: Tensor::zeros((1, 2, 1, 1), DType::F32, &device).unwrap(),
            mask: Tensor::from_vec(vec![1.0f32], (1, 1, 1, 1), &device).unwrap(),
            noise: Tensor::zeros((1, 2, 1, 1), DType::F32, &device).unwrap(),
        };

        let repainted = blend_inpaint_latents(&current, &ctx, &preserved)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(repainted, vec![100.0, 200.0]);

        let preserve_ctx = crate::img_utils::InpaintContext {
            mask: Tensor::zeros((1, 1, 1, 1), DType::F32, &device).unwrap(),
            ..ctx
        };
        let preserved_out = blend_inpaint_latents(&current, &preserve_ctx, &preserved)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(preserved_out, vec![1.0, 2.0]);
    }

    #[test]
    fn apply_flow_match_inpaint_reinserts_preserved_regions() {
        let device = Device::Cpu;
        let current = Tensor::from_vec(vec![100.0f32, 200.0], (1, 2, 1, 1), &device).unwrap();
        let ctx = crate::img_utils::InpaintContext {
            original_latents: Tensor::from_vec(vec![2.0f32, 4.0], (1, 2, 1, 1), &device).unwrap(),
            mask: Tensor::zeros((1, 1, 1, 1), DType::F32, &device).unwrap(),
            noise: Tensor::from_vec(vec![10.0f32, 20.0], (1, 2, 1, 1), &device).unwrap(),
        };

        let blended = apply_flow_match_inpaint(&current, &ctx, 0.25)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(blended, vec![4.0, 8.0]);
    }
}
