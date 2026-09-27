//! Pure (weight-free) pieces of Qwen Image 2.1's image-conditioned path.
//!
//! Every rule here is ported from diffusers `e0abab83b`
//! `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` (cited `P:`)
//! and is exercised without a checkpoint. The engine, the admission layer and
//! the clients must agree on these answers, so none of them may re-derive a
//! canvas or a template on its own.
//!
//! The reference-image engine path (reference preparation and Qwen3-VL
//! encoding) is the consumer of the canvas/template half; until it lands in
//! this module's caller, only the tests exercise those functions.
#![allow(dead_code)]

use anyhow::{ensure, Result};

use super::QWEN_IMAGE_21_SYSTEM_PROMPT;

/// `output_resolution`'s default (`P:527`). Its square is the pixel area every
/// condition image is resized to, and the output area when the caller names
/// no size — explicit `height`/`width` never change the reference area.
pub(crate) const QWEN_IMAGE_21_REFERENCE_RESOLUTION: u32 = 1024;

/// The pixel grid every Qwen Image 2.1 canvas lands on: the /16 VAE times the
/// 2x2 latent group one vision slot stands for (`P:631`).
pub(crate) const QWEN_IMAGE_21_CANVAS_GRID: u32 = 32;

/// The literal vision placeholder the Qwen3-VL processor expands.
pub(crate) const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";

/// diffusers `calculate_dimensions` (`P:149-156`): `sqrt(area * ratio)` and
/// `width / ratio`, each rounded with Python's ties-to-even `round()` onto the
/// 32 px grid. `ratio` is `width / height` of the source image.
///
/// The arithmetic lives in `mold_core::fit_to_target_area_ties_even`, which is
/// also what the CLI and Studio call to derive the output canvas from the last
/// reference, so the engine and every client land on the same side of a tie.
pub(crate) fn calculate_dimensions(target_area: u64, width: u32, height: u32) -> (u32, u32) {
    mold_core::fit_to_target_area_ties_even(width, height, target_area, QWEN_IMAGE_21_CANVAS_GRID)
}

/// The per-reference resize target (`P:656-660`): every condition image is
/// brought to `QWEN_IMAGE_21_REFERENCE_RESOLUTION²` area at its OWN aspect, and
/// that one resize feeds both the vision tower and the VAE.
pub(crate) fn reference_canvas(width: u32, height: u32) -> (u32, u32) {
    let side = u64::from(QWEN_IMAGE_21_REFERENCE_RESOLUTION);
    calculate_dimensions(side * side, width, height)
}

/// The output canvas (`P:621-633`).
///
/// Explicit dimensions win (`height = height or calculated_height`); otherwise
/// the LAST reference's aspect at the reference area; otherwise
/// `output_resolution` square. Either way the result is floored to the 32 px
/// grid (`width // multiple_of * multiple_of`).
pub(crate) fn derive_output_dimensions(
    last_reference: Option<(u32, u32)>,
    explicit: Option<(u32, u32)>,
) -> (u32, u32) {
    let (width, height) = explicit.unwrap_or_else(|| match last_reference {
        Some((width, height)) => reference_canvas(width, height),
        None => (
            QWEN_IMAGE_21_REFERENCE_RESOLUTION,
            QWEN_IMAGE_21_REFERENCE_RESOLUTION,
        ),
    });
    let floor = |value: u32| value / QWEN_IMAGE_21_CANVAS_GRID * QWEN_IMAGE_21_CANVAS_GRID;
    (floor(width), floor(height))
}

/// The raw image-conditioned processor string (`P:216-220`, `P:243`,
/// `P:252-258`).
///
/// The first image's vision block opens the user turn; every later one is
/// preceded by ONE space and its own literal `<imageN>` text marker (plain
/// text tokens, not specials). The prompt follows the last `<|vision_end|>`
/// with no separator, and an empty prompt becomes `" "` because Qwen has no
/// BOS token to read. With zero images this is exactly the text-to-image
/// template.
pub(crate) fn image_conditioned_prompt_template(prompt: &str, image_count: usize) -> String {
    if image_count == 0 {
        return super::t2i_prompt_template(prompt);
    }
    let prompt = if prompt.is_empty() { " " } else { prompt };
    let mut vision = String::from("<image1><|vision_start|><|image_pad|><|vision_end|>");
    for index in 2..=image_count {
        vision.push_str(&format!(
            " <image{index}><|vision_start|><|image_pad|><|vision_end|>"
        ));
    }
    format!(
        "<|im_start|>system\n{QWEN_IMAGE_21_SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{vision}{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// How many `<|image_pad|>` tokens the Qwen3-VL processor substitutes for one
/// condition image: `grid_thw.prod() // merge_size²` with patch 16 and merge
/// 2, i.e. one slot per 32x32 pixel group, which is exactly one slot per 2x2
/// group of /16 VAE latents. The transformer's `build_token_metadata` refuses
/// any disagreement between the two (`transformer_qwenimage21.py:827-833`).
pub(crate) fn image_pad_count(canvas_width: u32, canvas_height: u32) -> Result<usize> {
    ensure!(
        canvas_width > 0
            && canvas_height > 0
            && canvas_width.is_multiple_of(QWEN_IMAGE_21_CANVAS_GRID)
            && canvas_height.is_multiple_of(QWEN_IMAGE_21_CANVAS_GRID),
        "Qwen Image 2.1 condition canvas {canvas_width}x{canvas_height} is not on the {QWEN_IMAGE_21_CANVAS_GRID} px grid"
    );
    Ok((canvas_width / QWEN_IMAGE_21_CANVAS_GRID) as usize
        * (canvas_height / QWEN_IMAGE_21_CANVAS_GRID) as usize)
}

/// The processor's text substitution: the i-th `<|image_pad|>` placeholder, in
/// order of appearance, becomes `pad_counts[i]` copies of itself. The counts
/// must match the placeholders one-for-one.
pub(crate) fn expand_image_pad_tokens(text: &str, pad_counts: &[usize]) -> Result<String> {
    let pieces: Vec<&str> = text.split(IMAGE_PAD_TOKEN).collect();
    let placeholders = pieces.len() - 1;
    ensure!(
        placeholders == pad_counts.len(),
        "Qwen Image 2.1 template has {placeholders} image placeholders for {} images",
        pad_counts.len()
    );
    let mut expanded = String::with_capacity(
        text.len() + pad_counts.iter().sum::<usize>() * IMAGE_PAD_TOKEN.len(),
    );
    for (index, piece) in pieces.iter().enumerate() {
        expanded.push_str(piece);
        if let Some(&count) = pad_counts.get(index) {
            ensure!(
                count > 0,
                "Qwen Image 2.1 condition image {index} has no slots"
            );
            for _ in 0..count {
                expanded.push_str(IMAGE_PAD_TOKEN);
            }
        }
    }
    Ok(expanded)
}

/// What the decoded alpha plane becomes in the published artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputAlpha {
    /// Publish RGB. Chosen when transparency was not requested and every
    /// alpha byte is 255, which keeps a v0.32 text-to-image PNG/JPEG byte-for
    /// byte: the RGB channels are unchanged and PNG stays `ColorType::Rgb`.
    Rgb,
    /// Publish RGBA (PNG / WebP).
    Rgba,
    /// JPEG cannot carry alpha: composite over white — the convention the
    /// checkpoint itself uses for its vision input (`P:266-270`) — and tell
    /// the caller with a request warning.
    CompositeOverWhite,
}

impl OutputAlpha {
    /// The alpha rule. `format_carries_alpha` is true for PNG and WebP.
    ///
    /// A transparency request always publishes RGBA even when the model
    /// happened to paint an opaque canvas — the caller asked for an alpha
    /// channel. Without the request, an opaque plane is RGB (never an
    /// invented threshold: 254 is not 255). A JPEG with transparency
    /// requested is refused before any weights load; reaching here with one
    /// still composites rather than silently dropping alpha.
    pub(crate) fn decide(
        transparent_requested: bool,
        format_carries_alpha: bool,
        alpha: &[u8],
    ) -> Self {
        let opaque = alpha.iter().all(|&value| value == 255);
        match (format_carries_alpha, transparent_requested, opaque) {
            (_, false, true) => Self::Rgb,
            (true, _, _) => Self::Rgba,
            (false, _, _) => Self::CompositeOverWhite,
        }
    }

    /// The request warning attached when alpha had to be flattened.
    pub(crate) fn warning(self) -> Option<&'static str> {
        (self == Self::CompositeOverWhite).then_some(
            "Qwen Image 2.1 produced transparent pixels, which JPEG cannot store; they were composited over white. Choose PNG or WebP to keep the alpha channel.",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_dimensions_rounds_halves_to_even_like_python() {
        // `round(1040 / 32)` is `round(32.5)` = 32 in Python; `f64::round`
        // would give 33 (1056).
        assert_eq!(calculate_dimensions(1024 * 1024, 4225, 4096), (1024, 1024));
        assert_eq!(calculate_dimensions(1024 * 1024, 1600, 900), (1376, 768));
        assert_eq!(calculate_dimensions(1024 * 1024, 1080, 1920), (768, 1376));
    }

    #[test]
    fn reference_canvas_is_the_1024_area_at_the_references_own_aspect() {
        assert_eq!(reference_canvas(640, 480), (1184, 896));
        assert_eq!(reference_canvas(3000, 2000), (1248, 832));
        assert_eq!(reference_canvas(1024, 1024), (1024, 1024));
    }

    #[test]
    fn output_dimensions_follow_explicit_then_last_reference_then_square() {
        assert_eq!(derive_output_dimensions(None, None), (1024, 1024));
        assert_eq!(
            derive_output_dimensions(Some((1920, 1080)), None),
            (1376, 768)
        );
        // Explicit wins and is floored — not rounded — to the 32 px grid.
        assert_eq!(
            derive_output_dimensions(Some((1920, 1080)), Some((1000, 1500))),
            (992, 1472)
        );
        assert_eq!(
            derive_output_dimensions(None, Some((2048, 2048))),
            (2048, 2048)
        );
    }

    #[test]
    fn templates_match_the_upstream_processor_strings_byte_for_byte() {
        let system = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
        assert_eq!(
            image_conditioned_prompt_template("make it blue", 1),
            format!(
                "{system}<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>make it blue<|im_end|>\n<|im_start|>assistant\n"
            )
        );
        assert_eq!(
            image_conditioned_prompt_template("combine them", 2),
            format!(
                "{system}<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>combine them<|im_end|>\n<|im_start|>assistant\n"
            )
        );
        assert_eq!(
            image_conditioned_prompt_template("", 3),
            format!(
                "{system}<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|> <image3><|vision_start|><|image_pad|><|vision_end|> <|im_end|>\n<|im_start|>assistant\n"
            )
        );
        assert_eq!(
            image_conditioned_prompt_template("a cat", 0),
            super::super::t2i_prompt_template("a cat")
        );
    }

    #[test]
    fn pad_expansion_substitutes_each_placeholder_in_order() {
        assert_eq!(image_pad_count(1024, 1024).unwrap(), 1024);
        assert_eq!(image_pad_count(1376, 768).unwrap(), 43 * 24);
        assert!(image_pad_count(1000, 1024).is_err());
        let template = image_conditioned_prompt_template("x", 2);
        let expanded = expand_image_pad_tokens(&template, &[2, 3]).unwrap();
        assert_eq!(expanded.matches(IMAGE_PAD_TOKEN).count(), 5);
        assert!(expanded.contains(
            "<image1><|vision_start|><|image_pad|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>x"
        ));
        assert!(expand_image_pad_tokens(&template, &[2]).is_err());
        assert!(expand_image_pad_tokens(&template, &[2, 0]).is_err());
    }

    #[test]
    fn alpha_rule_keeps_opaque_text_to_image_rgb() {
        let opaque = [255u8; 16];
        let mut translucent = opaque;
        translucent[3] = 254;
        assert_eq!(OutputAlpha::decide(false, true, &opaque), OutputAlpha::Rgb);
        assert_eq!(OutputAlpha::decide(false, false, &opaque), OutputAlpha::Rgb);
        assert_eq!(
            OutputAlpha::decide(false, true, &translucent),
            OutputAlpha::Rgba
        );
        assert_eq!(OutputAlpha::decide(true, true, &opaque), OutputAlpha::Rgba);
        assert_eq!(
            OutputAlpha::decide(false, false, &translucent),
            OutputAlpha::CompositeOverWhite
        );
        assert_eq!(
            OutputAlpha::decide(true, false, &opaque),
            OutputAlpha::CompositeOverWhite
        );
        assert!(OutputAlpha::CompositeOverWhite.warning().is_some());
        assert!(OutputAlpha::Rgba.warning().is_none());
    }
}
