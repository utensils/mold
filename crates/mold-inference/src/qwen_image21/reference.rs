//! Reference-image preparation and Qwen3-VL conditioning for Qwen Image 2.1.
//!
//! Ported from diffusers `e0abab83b`
//! `pipelines/qwenimage21/pipeline_qwenimage21.py` (cited `P:`). One resize of
//! each reference feeds BOTH consumers (`P:655-663`):
//!
//! - the VAE reads all four channels, `u8 / 255 * 2 - 1` (`VaeImageProcessor`);
//! - the vision tower reads an RGB copy pasted over white with the alpha as
//!   the mask (`P:266-271`), packed by the Qwen3-VL processor.
//!
//! The vision tower is independent of the text, so it runs once per request
//! and its features serve both CFG branches (upstream runs it twice on the
//! same input). Each branch then runs the multimodal language model batch-1
//! (`P:689-696`) and keeps the final pre-norm state after the system prefix.

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, RgbaImage};
use mold_candle::qwen3_vl::{
    create_mm_token_type_ids, pack_qwen_vision_u8_torchvision, qwen_mrope_positions,
    ConditionerCheckpoint, GridThw, Qwen3VlVisionDimensions, Qwen3VlVisionModel,
};

use super::conditioning::{
    carries_alpha, expand_image_pad_tokens, image_conditioned_prompt_template, image_pad_count,
    reference_canvas,
};
use super::{system_message_prefix, QwenImage21TextConditioning, QWEN_IMAGE_21_VAE_SCALE_FACTOR};
use crate::encoders::qwen3::Qwen3Encoder;
use crate::encoders::qwen3_vl_inject::VisualInjection;
use crate::pillow_resize::{composite_over_white, resize_rgba_premultiplied, Filter};

/// Qwen3-VL's `<|image_pad|>` id (`text_encoder/config.json` `image_token_id`).
pub(crate) const QWEN3_VL_IMAGE_PAD_ID: u32 = 151_655;
/// `vision_config.spatial_merge_size`.
const SPATIAL_MERGE: usize = 2;
/// `vision_config.patch_size`.
const PATCH: usize = 16;

/// One reference, resized once for both consumers.
#[derive(Clone, Debug)]
pub(crate) struct PreparedReference {
    /// The Pillow-premultiplied LANCZOS resize to the reference canvas.
    pub rgba: RgbaImage,
    /// The vision tower's copy: `rgba` pasted over white.
    pub vision_rgb: RgbImage,
    /// Whether the SOURCE carried any transparency.
    pub carries_alpha: bool,
}

impl PreparedReference {
    pub(crate) fn width(&self) -> u32 {
        self.rgba.width()
    }

    pub(crate) fn height(&self) -> u32 {
        self.rgba.height()
    }

    /// Latent grid `(rows, columns)` of this reference: the canvas / 16.
    pub(crate) fn latent_shape(&self) -> (usize, usize) {
        (
            self.height() as usize / QWEN_IMAGE_21_VAE_SCALE_FACTOR,
            self.width() as usize / QWEN_IMAGE_21_VAE_SCALE_FACTOR,
        )
    }

    /// `<|image_pad|>` tokens this reference expands to.
    pub(crate) fn pad_count(&self) -> Result<usize> {
        image_pad_count(self.width(), self.height())
    }

    /// The Qwen3-VL processor grid in patches.
    pub(crate) fn grid(&self) -> GridThw {
        GridThw {
            temporal: 1,
            height: self.height() as usize / PATCH,
            width: self.width() as usize / PATCH,
        }
    }

    /// VAE input `[1, 4, H, W]` in `[-1, 1]`: `VaeImageProcessor.preprocess`
    /// (`pil_to_numpy` divides by 255 in f32, `normalize` is `2x - 1`).
    pub(crate) fn vae_input(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let (width, height) = self.rgba.dimensions();
        let values: Vec<f32> = self
            .rgba
            .as_raw()
            .iter()
            .map(|byte| 2.0 * (f32::from(*byte) / 255.0) - 1.0)
            .collect();
        Ok(
            Tensor::from_vec(values, (height as usize, width as usize, 4), device)?
                .permute((2, 0, 1))?
                .unsqueeze(0)?
                .contiguous()?
                .to_dtype(dtype)?,
        )
    }
}

/// Decode (EXIF orientation, ICC to sRGB) and resize one reference
/// (`P:653-663`): a non-RGBA source becomes opaque RGBA, the canvas is the
/// reference area at the source's own aspect, and the resize is Pillow's
/// premultiplied LANCZOS.
///
/// Upstream's PIL `open` does not apply EXIF orientation; mold does for every
/// source image, so a phone photo is conditioned the right way up.
pub(crate) fn prepare_reference(bytes: &[u8]) -> Result<PreparedReference> {
    let source = crate::img_utils::decode_oriented_srgb_rgba(bytes)
        .context("failed to decode a Qwen Image 2.1 reference image")?;
    prepare_decoded_reference(&source)
}

pub(crate) fn prepare_decoded_reference(source: &RgbaImage) -> Result<PreparedReference> {
    let (width, height) = reference_canvas(source.width(), source.height());
    let rgba = resize_rgba_premultiplied(source, width, height, Filter::Lanczos, &mut || Ok(()))?;
    let vision_rgb = composite_over_white(&rgba);
    Ok(PreparedReference {
        carries_alpha: carries_alpha(source.pixels().map(|pixel| pixel.0[3])),
        rgba,
        vision_rgb,
    })
}

/// Tensors of the vision tower inside the Qwen3-VL text-encoder shards.
pub(crate) fn is_vision_tensor(name: &str) -> bool {
    name.starts_with("model.visual.")
}

/// Load the Qwen3-VL vision tower (`model.visual`, ~1.07 GiB BF16) from the
/// text-encoder shards. Only its own tensors are mapped.
pub(crate) fn load_vision_tower(
    paths: &[std::path::PathBuf],
    device: &Device,
    dtype: DType,
    progress: &crate::progress::ProgressReporter,
) -> Result<Qwen3VlVisionModel> {
    let vb = crate::weight_loader::load_safetensors_with_filtered_progress(
        paths,
        dtype,
        device,
        "Qwen Image 2.1 vision tower",
        progress,
        is_vision_tensor,
    )?;
    Ok(Qwen3VlVisionModel::new(
        &Qwen3VlVisionDimensions::qwen_image_21(),
        vb.pp("model").pp("visual"),
    )?)
}

/// The vision tower's output for every reference of a request.
pub(crate) struct VisionFeatures {
    /// Merger rows `[Σ pads, 4096]`, in reference order.
    pub embeds: Tensor,
    /// DeepStack maps, each `[Σ pads, 4096]`, in tap order.
    pub deepstack: Vec<Tensor>,
    /// Processor grids, in reference order.
    pub grids: Vec<GridThw>,
}

impl VisionFeatures {
    fn rows(&self) -> Result<usize> {
        Ok(self.embeds.dim(0)?)
    }
}

/// The packed `pixel_values` `[Σ patches, 1536]` and `image_grid_thw`
/// `[refs, 3]` the Qwen3-VL processor produces for `references`.
pub(crate) fn pack_vision_inputs(
    references: &[PreparedReference],
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<GridThw>)> {
    ensure!(!references.is_empty(), "no reference images to pack");
    let mut values = Vec::new();
    let mut grids = Vec::with_capacity(references.len());
    let mut patch_width = 0;
    for reference in references {
        let (width, height) = reference.vision_rgb.dimensions();
        let packed = pack_qwen_vision_u8_torchvision(
            reference.vision_rgb.as_raw(),
            1,
            height as usize,
            width as usize,
        )
        .map_err(|error| anyhow::anyhow!("Qwen3-VL processor: {error}"))?;
        ensure!(packed.grid == reference.grid(), "Qwen3-VL grid mismatch");
        patch_width = packed.patch_width;
        values.extend(packed.values);
        grids.push(packed.grid);
    }
    let patches = values.len() / patch_width;
    let grid_rows: Vec<u32> = grids
        .iter()
        .flat_map(|grid| [grid.temporal, grid.height, grid.width])
        .map(|value| value as u32)
        .collect();
    Ok((
        Tensor::from_vec(values, (patches, patch_width), device)?,
        Tensor::from_vec(grid_rows, (grids.len(), 3), device)?,
        grids,
    ))
}

/// Run the vision tower once over every reference.
pub(crate) fn encode_vision(
    tower: &Qwen3VlVisionModel,
    references: &[PreparedReference],
    device: &Device,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<VisionFeatures> {
    let (pixels, grid, grids) = pack_vision_inputs(references, device)?;
    let mut cancelled = None;
    let result = tower.forward(&pixels, &grid, &mut |_: ConditionerCheckpoint| {
        if let Err(error) = checkpoint() {
            cancelled = Some(error);
            return Err(candle_core::Error::Msg("cancelled".into()));
        }
        Ok(())
    });
    if let Some(error) = cancelled {
        return Err(error);
    }
    let (embeds, deepstack) = result?;
    let expected = references
        .iter()
        .map(PreparedReference::pad_count)
        .sum::<Result<usize>>()?;
    ensure!(
        embeds.dim(0)? == expected,
        "Qwen3-VL merger produced {} rows for {expected} image pads",
        embeds.dim(0)?
    );
    Ok(VisionFeatures {
        embeds,
        deepstack,
        grids,
    })
}

/// The token ids of `prompt` with `pad_counts` expanded, exactly as the
/// processor builds them for one prompt.
pub(crate) fn tokenize_image_conditioned(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    pad_counts: &[usize],
) -> Result<Vec<u32>> {
    let template = image_conditioned_prompt_template(prompt, pad_counts.len());
    let text = expand_image_pad_tokens(&template, pad_counts)?;
    Ok(tokenizer
        .encode(text, true)
        .map_err(|error| anyhow::anyhow!("Qwen Image 2.1 prompt tokenization failed: {error}"))?
        .get_ids()
        .to_vec())
}

/// Encode one prompt with its references (`P:233-327`) into batch-1
/// conditioning: the final pre-norm state with the system prefix dropped, and
/// `image_slots` marking the retained `<|image_pad|>` rows.
pub(crate) fn encode_prompt_with_images(
    encoder: &mut Qwen3Encoder,
    vision: &VisionFeatures,
    prompt: &str,
) -> Result<QwenImage21TextConditioning> {
    let pad_counts: Vec<usize> = vision
        .grids
        .iter()
        .map(|grid| grid.height * grid.width / (SPATIAL_MERGE * SPATIAL_MERGE))
        .collect();
    let ids = tokenize_image_conditioned(&encoder.tokenizer, prompt, &pad_counts)?;
    let drop_idx = encoder
        .tokenizer
        .encode(system_message_prefix(), true)
        .map_err(|error| {
            anyhow::anyhow!("Qwen Image 2.1 system prompt tokenization failed: {error}")
        })?
        .len();
    let positions: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| (*id == QWEN3_VL_IMAGE_PAD_ID).then_some(index))
        .collect();
    ensure!(
        positions.len() == vision.rows()?,
        "Qwen Image 2.1 prompt carries {} image pads but the vision tower produced {} rows",
        positions.len(),
        vision.rows()?
    );
    ensure!(
        positions.first().is_none_or(|first| *first >= drop_idx),
        "Qwen Image 2.1 image pad inside the system prefix"
    );
    let mrope = qwen_mrope_positions(
        &create_mm_token_type_ids(&ids),
        &vision.grids,
        &[],
        SPATIAL_MERGE,
    )
    .map_err(|error| anyhow::anyhow!("Qwen3-VL MRoPE: {error}"))?;
    let sequence = ids.len();
    let input_ids = Tensor::from_vec(ids.clone(), (1, sequence), &encoder.device)?;
    let visual = VisualInjection {
        positions,
        embeds: vision.embeds.clone(),
        deepstack: vision.deepstack.clone(),
    };
    let hidden = encoder.forward_multimodal_final_pre_norm(&input_ids, Some(visual), &mrope)?;
    ensure!(
        sequence > drop_idx,
        "Qwen Image 2.1 system prefix consumes the whole prompt"
    );
    let retained = sequence - drop_idx;
    Ok(QwenImage21TextConditioning {
        embeddings: hidden.narrow(1, drop_idx, retained)?,
        valid_tokens: vec![vec![true; retained]],
        image_slots: vec![ids[drop_idx..]
            .iter()
            .map(|id| *id == QWEN3_VL_IMAGE_PAD_ID)
            .collect()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_resizes_once_and_flattens_only_the_vision_copy() {
        let mut source = RgbaImage::from_pixel(64, 48, image::Rgba([200, 30, 10, 255]));
        source.put_pixel(3, 3, image::Rgba([9, 9, 9, 0]));
        let prepared = prepare_decoded_reference(&source).unwrap();
        assert_eq!(
            (prepared.width(), prepared.height()),
            reference_canvas(64, 48)
        );
        assert!(prepared.carries_alpha);
        assert_eq!(prepared.vision_rgb.dimensions(), prepared.rgba.dimensions());
        assert_eq!(prepared.vision_rgb, composite_over_white(&prepared.rgba));
        assert_eq!(
            prepared.latent_shape(),
            (
                prepared.height() as usize / 16,
                prepared.width() as usize / 16
            )
        );
        assert_eq!(
            prepared.pad_count().unwrap(),
            prepared.grid().height * prepared.grid().width / 4
        );
        let opaque =
            prepare_decoded_reference(&RgbaImage::from_pixel(32, 32, image::Rgba([1, 2, 3, 255])))
                .unwrap();
        assert!(!opaque.carries_alpha);
    }

    #[test]
    fn vae_input_is_all_four_channels_in_minus_one_to_one() {
        let prepared = prepare_decoded_reference(&RgbaImage::from_pixel(
            32,
            32,
            image::Rgba([0, 255, 51, 102]),
        ))
        .unwrap();
        let input = prepared.vae_input(&Device::Cpu, DType::F32).unwrap();
        let (_, channels, height, width) = input.dims4().unwrap();
        assert_eq!(channels, 4);
        assert_eq!((width as u32, height as u32), prepared.rgba.dimensions());
        let pixel = prepared.rgba.get_pixel(0, 0).0;
        let values: Vec<f32> = (0..4)
            .map(|c| {
                input
                    .get(0)
                    .unwrap()
                    .get(c)
                    .unwrap()
                    .get(0)
                    .unwrap()
                    .get(0)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            })
            .collect();
        for (value, byte) in values.iter().zip(pixel) {
            assert_eq!(*value, 2.0 * (f32::from(byte) / 255.0) - 1.0);
        }
    }

    #[test]
    fn vision_inputs_pack_every_reference_in_order() {
        let first =
            prepare_decoded_reference(&RgbaImage::from_pixel(64, 32, image::Rgba([0, 0, 0, 255])))
                .unwrap();
        let second = prepare_decoded_reference(&RgbaImage::from_pixel(
            32,
            64,
            image::Rgba([255, 255, 255, 255]),
        ))
        .unwrap();
        let (pixels, grid, grids) =
            pack_vision_inputs(&[first.clone(), second.clone()], &Device::Cpu).unwrap();
        assert_eq!(grids, vec![first.grid(), second.grid()]);
        let patches: usize = grids.iter().map(|g| g.height * g.width).sum();
        assert_eq!(pixels.dims(), &[patches, 3 * 2 * 16 * 16]);
        assert_eq!(grid.dims(), &[2, 3]);
        let values = pixels.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values[0], -1.0);
        assert_eq!(*values.last().unwrap(), 1.0);
    }
}
