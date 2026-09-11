//! `mold run --fit` — client-side source-image preprocessing.
//!
//! The wire field `GenerateRequest.source_fit` has always been browser-side
//! PROVENANCE: the web and desktop apps resize the attached picture onto the
//! requested canvas with a `<canvas>` `drawImage`, then record which policy
//! they used so a reuse can restore the crop. No engine reads it. The CLI
//! could not do the same thing at all — it went the other way and derived the
//! CANVAS from the picture (`generate::source_image_model_dimensions`), so
//! `mold run --image wide.png --width 1024 --height 512` silently ignored the
//! requested shape.
//!
//! This module is the honest port of `studio/lib/sourceFit.ts`'s
//! `resolveSourceFitTransform` (`:175-227`): the same cover/contain scale, the
//! same centred offsets, the same negative offsets that trim the overflow.
//! The pixels are resampled with Lanczos3 and composited onto a target-sized
//! canvas, and the same `{"mode": …}` object the browsers record rides the
//! request, so a print made here reopens in the Library with its crop intact.
//!
//! Two of the five browser modes are refused BY NAME rather than silently
//! mapped: `pad-repaint` needs a generated repaint mask and `upscale-then-fit`
//! needs an upscaler pass, and both are features of the app, not of this
//! command.

use anyhow::Result;
use image::imageops::FilterType;
use image::{Rgba, RgbaImage};

/// The three source-fit policies a terminal can honour.
///
/// Deliberately NOT the whole `SourceFitPolicy` union: the two absent modes
/// are refused by [`parse_source_fit_mode`] with their reason, so a user who
/// read the browser's labels learns why instead of seeing "invalid value".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFitMode {
    /// Cover: keep proportions, scale until the canvas is filled, trim the
    /// overflow evenly on both sides.
    CropFill,
    /// Contain: keep proportions, scale until the whole picture fits, pad the
    /// remainder with black.
    PadFit,
    /// Resample straight onto the canvas. Proportions change.
    LanczosResize,
}

impl SourceFitMode {
    /// The wire spelling, identical to the browser union's `mode`.
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::CropFill => "crop-fill",
            Self::PadFit => "pad-fit",
            Self::LanczosResize => "lanczos-resize",
        }
    }
}

/// `--fit` value parser: the three honoured modes, plus a named refusal for
/// each browser-only mode.
pub fn parse_source_fit_mode(raw: &str) -> Result<SourceFitMode, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "crop-fill" => Ok(SourceFitMode::CropFill),
        "pad-fit" => Ok(SourceFitMode::PadFit),
        "lanczos-resize" => Ok(SourceFitMode::LanczosResize),
        "pad-repaint" => Err(
            "pad-repaint is not available on the command line: it pads the \
                              canvas and then asks the model to REPAINT the added borders, \
                              which needs a generated mask the apps draw. Use --fit pad-fit \
                              for the same framing with plain black borders."
                .to_string(),
        ),
        "upscale-then-fit" => Err("upscale-then-fit is not available on the command line: it \
                                   runs an upscaler over the source before cropping, which is \
                                   a separate render. Run `mold upscale` on the picture first, \
                                   then pass the result with --fit crop-fill."
            .to_string()),
        other => Err(format!(
            "unknown fit policy '{other}' (expected crop-fill, pad-fit, or lanczos-resize)"
        )),
    }
}

/// Where the resampled source lands on the target canvas.
///
/// Mirrors `SourceFitTransform` in `studio/lib/sourceFit.ts:156`. Offsets are
/// signed: `crop-fill` produces negative ones, which is how the overflow is
/// trimmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceFitTransform {
    pub output_width: u32,
    pub output_height: u32,
    pub draw_width: u32,
    pub draw_height: u32,
    pub offset_x: i64,
    pub offset_y: i64,
}

/// Port of `resolveSourceFitTransform` (`studio/lib/sourceFit.ts:175`).
///
/// The CLI never authors an alignment, so `crop-fill`'s offsets are the
/// browser's `align === undefined` branch: `Math.round(available / 2)`,
/// negated. JavaScript's `Math.round` is half-up rather than Rust's
/// half-away-from-zero, and the argument here is always non-negative
/// (`-available` for a crop, `available` for a pad), so `(v / 2.0).round()`
/// agrees on every input this can see.
pub fn resolve_source_fit_transform(
    source: (u32, u32),
    target: (u32, u32),
    mode: SourceFitMode,
) -> SourceFitTransform {
    let (source_width, source_height) = source;
    let (output_width, output_height) = target;
    if mode == SourceFitMode::LanczosResize {
        return SourceFitTransform {
            output_width,
            output_height,
            draw_width: output_width,
            draw_height: output_height,
            offset_x: 0,
            offset_y: 0,
        };
    }

    let source_ratio = f64::from(source_width) / f64::from(source_height);
    let target_ratio = f64::from(output_width) / f64::from(output_height);
    let crop = mode == SourceFitMode::CropFill;
    let scale = if crop {
        if target_ratio > source_ratio {
            f64::from(output_width) / f64::from(source_width)
        } else {
            f64::from(output_height) / f64::from(source_height)
        }
    } else if target_ratio < source_ratio {
        f64::from(output_width) / f64::from(source_width)
    } else {
        f64::from(output_height) / f64::from(source_height)
    };
    let draw_width = (f64::from(source_width) * scale).round().max(1.0) as u32;
    let draw_height = (f64::from(source_height) * scale).round().max(1.0) as u32;
    let available_x = i64::from(output_width) - i64::from(draw_width);
    let available_y = i64::from(output_height) - i64::from(draw_height);
    let centre = |available: i64| -> i64 { (available as f64 / 2.0).round() as i64 };
    let (offset_x, offset_y) = if crop {
        (-centre(-available_x), -centre(-available_y))
    } else {
        (centre(available_x), centre(available_y))
    };

    SourceFitTransform {
        output_width,
        output_height,
        draw_width,
        draw_height,
        offset_x,
        offset_y,
    }
}

/// Resample `bytes` onto a `target_width` x `target_height` canvas and encode
/// the result as PNG.
///
/// PNG because the fitted picture is conditioning, not a deliverable: a
/// re-encode through JPEG would add its own artefacts on top of the resample,
/// and every family that reads a source image reads PNG.
pub fn apply_source_fit(
    bytes: &[u8],
    target_width: u32,
    target_height: u32,
    mode: SourceFitMode,
) -> Result<(Vec<u8>, SourceFitTransform)> {
    if target_width == 0 || target_height == 0 {
        anyhow::bail!("--fit needs a canvas with a width and a height");
    }
    let source = image::load_from_memory(bytes)
        .map_err(|error| anyhow::anyhow!("failed to decode source image: {error}"))?
        .to_rgba8();
    if source.width() == 0 || source.height() == 0 {
        anyhow::bail!("source image has no pixels");
    }
    let transform = resolve_source_fit_transform(
        (source.width(), source.height()),
        (target_width, target_height),
        mode,
    );
    let scaled = image::imageops::resize(
        &source,
        transform.draw_width,
        transform.draw_height,
        FilterType::Lanczos3,
    );
    // Opaque black, the browsers' pad colour: their canvas starts cleared and
    // the surfaces composite onto a black-filled rect before drawing.
    let mut canvas = RgbaImage::from_pixel(target_width, target_height, Rgba([0, 0, 0, 255]));
    image::imageops::overlay(&mut canvas, &scaled, transform.offset_x, transform.offset_y);
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(canvas)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|error| anyhow::anyhow!("failed to encode the fitted source image: {error}"))?;
    Ok((png, transform))
}

/// The provenance object the request carries, byte-identical to what the
/// apps record so a print made here reuses the same way.
pub fn source_fit_provenance(mode: SourceFitMode) -> serde_json::Value {
    serde_json::json!({ "mode": mode.as_wire() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4x2 picture: left half red, right half blue, so the horizontal
    /// placement is readable from the output pixels.
    fn wide_source() -> Vec<u8> {
        let mut image = RgbaImage::new(4, 2);
        for y in 0..2 {
            for x in 0..4 {
                let colour = if x < 2 {
                    Rgba([255, 0, 0, 255])
                } else {
                    Rgba([0, 0, 255, 255])
                };
                image.put_pixel(x, y, colour);
            }
        }
        encode(image)
    }

    /// A 2x4 picture: top half green, bottom half white.
    fn tall_source() -> Vec<u8> {
        let mut image = RgbaImage::new(2, 4);
        for y in 0..4 {
            for x in 0..2 {
                let colour = if y < 2 {
                    Rgba([0, 255, 0, 255])
                } else {
                    Rgba([255, 255, 255, 255])
                };
                image.put_pixel(x, y, colour);
            }
        }
        encode(image)
    }

    fn encode(image: RgbaImage) -> Vec<u8> {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    fn decode(bytes: &[u8]) -> RgbaImage {
        image::load_from_memory(bytes).unwrap().to_rgba8()
    }

    #[test]
    fn every_mode_delivers_exactly_the_requested_canvas() {
        for source in [wide_source(), tall_source()] {
            for mode in [
                SourceFitMode::CropFill,
                SourceFitMode::PadFit,
                SourceFitMode::LanczosResize,
            ] {
                let (png, transform) = apply_source_fit(&source, 8, 8, mode).unwrap();
                let fitted = decode(&png);
                assert_eq!((fitted.width(), fitted.height()), (8, 8), "{mode:?}");
                assert_eq!((transform.output_width, transform.output_height), (8, 8));
            }
        }
    }

    /// Cover: a 4x2 source on an 8x8 canvas scales by 4 (height-limited), so
    /// the 16-wide render is trimmed by 4 columns on each side and every
    /// output pixel comes from the source.
    #[test]
    fn crop_fill_covers_the_canvas_and_trims_the_overflow() {
        let transform = resolve_source_fit_transform((4, 2), (8, 8), SourceFitMode::CropFill);
        assert_eq!(transform.draw_width, 16);
        assert_eq!(transform.draw_height, 8);
        assert_eq!(transform.offset_x, -4);
        assert_eq!(transform.offset_y, 0);

        let (png, _) = apply_source_fit(&wide_source(), 8, 8, SourceFitMode::CropFill).unwrap();
        let fitted = decode(&png);
        // The kept window is the middle half of the source, so the left
        // column is still red and the right column still blue, and nothing
        // is black.
        assert!(fitted.get_pixel(0, 4)[0] > 200, "left edge should stay red");
        assert!(
            fitted.get_pixel(7, 4)[2] > 200,
            "right edge should stay blue"
        );
        for pixel in fitted.pixels() {
            assert!(
                pixel[0] > 20 || pixel[2] > 20,
                "crop-fill must leave no padding"
            );
        }
    }

    /// Contain: the same 4x2 source scales by 2 (width-limited) to 8x4 and is
    /// centred vertically, leaving two black bands.
    #[test]
    fn pad_fit_centres_the_source_and_pads_the_rest_black() {
        let transform = resolve_source_fit_transform((4, 2), (8, 8), SourceFitMode::PadFit);
        assert_eq!(
            (transform.draw_width, transform.draw_height),
            (8, 4),
            "contain scale is width-limited here"
        );
        assert_eq!((transform.offset_x, transform.offset_y), (0, 2));

        let (png, _) = apply_source_fit(&wide_source(), 8, 8, SourceFitMode::PadFit).unwrap();
        let fitted = decode(&png);
        for y in [0_u32, 1, 6, 7] {
            for x in 0..8 {
                assert_eq!(
                    fitted.get_pixel(x, y),
                    &Rgba([0, 0, 0, 255]),
                    "row {y} should be padding"
                );
            }
        }
        assert!(fitted.get_pixel(1, 4)[0] > 200, "source lands in the band");
        assert!(fitted.get_pixel(6, 4)[2] > 200, "source lands in the band");
    }

    /// The tall source exercises the other branch of both scale rules.
    #[test]
    fn a_tall_source_pads_sideways_and_crops_vertically() {
        let pad = resolve_source_fit_transform((2, 4), (8, 8), SourceFitMode::PadFit);
        assert_eq!((pad.draw_width, pad.draw_height), (4, 8));
        assert_eq!((pad.offset_x, pad.offset_y), (2, 0));

        let crop = resolve_source_fit_transform((2, 4), (8, 8), SourceFitMode::CropFill);
        assert_eq!((crop.draw_width, crop.draw_height), (8, 16));
        assert_eq!((crop.offset_x, crop.offset_y), (0, -4));

        let (png, _) = apply_source_fit(&tall_source(), 8, 8, SourceFitMode::PadFit).unwrap();
        let fitted = decode(&png);
        assert_eq!(fitted.get_pixel(0, 4), &Rgba([0, 0, 0, 255]));
        assert_eq!(fitted.get_pixel(7, 4), &Rgba([0, 0, 0, 255]));
        assert!(fitted.get_pixel(4, 1)[1] > 200, "top band keeps the green");
        assert!(
            fitted.get_pixel(4, 6)[0] > 200,
            "bottom band keeps the white"
        );
    }

    /// Stretch: the whole source reaches every edge, so both halves survive
    /// at full width and nothing is padded or trimmed.
    #[test]
    fn lanczos_resize_stretches_the_whole_source_over_the_canvas() {
        let transform = resolve_source_fit_transform((4, 2), (8, 4), SourceFitMode::LanczosResize);
        assert_eq!((transform.draw_width, transform.draw_height), (8, 4));
        assert_eq!((transform.offset_x, transform.offset_y), (0, 0));

        let (png, _) =
            apply_source_fit(&wide_source(), 8, 4, SourceFitMode::LanczosResize).unwrap();
        let fitted = decode(&png);
        assert_eq!((fitted.width(), fitted.height()), (8, 4));
        assert!(fitted.get_pixel(0, 2)[0] > 200);
        assert!(fitted.get_pixel(7, 2)[2] > 200);
    }

    /// A source already the target shape is copied through unchanged, which
    /// is what makes `--fit` safe to leave on in a script.
    #[test]
    fn a_matching_source_is_unchanged_by_every_mode() {
        let mut square = RgbaImage::new(8, 8);
        for (x, y, pixel) in square.enumerate_pixels_mut() {
            *pixel = Rgba([(x * 30) as u8, (y * 30) as u8, 7, 255]);
        }
        let source = encode(square.clone());
        for mode in [
            SourceFitMode::CropFill,
            SourceFitMode::PadFit,
            SourceFitMode::LanczosResize,
        ] {
            let (png, _) = apply_source_fit(&source, 8, 8, mode).unwrap();
            assert_eq!(decode(&png), square, "{mode:?}");
        }
    }

    #[test]
    fn the_two_browser_only_modes_are_refused_by_name_with_the_reason() {
        let repaint = parse_source_fit_mode("pad-repaint").unwrap_err();
        assert!(repaint.contains("pad-repaint"));
        assert!(repaint.contains("mask"));
        assert!(repaint.contains("--fit pad-fit"));

        let upscale = parse_source_fit_mode("upscale-then-fit").unwrap_err();
        assert!(upscale.contains("upscale-then-fit"));
        assert!(upscale.contains("mold upscale"));
    }

    #[test]
    fn the_parser_accepts_the_three_wire_spellings_and_nothing_else() {
        for (raw, mode) in [
            ("crop-fill", SourceFitMode::CropFill),
            ("pad-fit", SourceFitMode::PadFit),
            ("lanczos-resize", SourceFitMode::LanczosResize),
            ("CROP-FILL", SourceFitMode::CropFill),
        ] {
            assert_eq!(parse_source_fit_mode(raw).unwrap(), mode);
            assert_eq!(mode.as_wire(), mode.as_wire());
        }
        assert!(parse_source_fit_mode("cover").is_err());
    }

    /// The provenance object is exactly what `parseSourceFitPolicy`
    /// (`studio/lib/sourceFit.ts:85`) accepts — a bare `mode` string, no
    /// alignment keys, so a reuse in the app restores the same policy.
    #[test]
    fn the_provenance_is_the_shape_the_apps_parse() {
        for (mode, wire) in [
            (SourceFitMode::CropFill, "crop-fill"),
            (SourceFitMode::PadFit, "pad-fit"),
            (SourceFitMode::LanczosResize, "lanczos-resize"),
        ] {
            assert_eq!(
                source_fit_provenance(mode),
                serde_json::json!({ "mode": wire })
            );
        }
    }

    #[test]
    fn a_canvasless_target_is_refused_rather_than_producing_no_pixels() {
        let error = apply_source_fit(&wide_source(), 0, 0, SourceFitMode::CropFill).unwrap_err();
        assert!(error.to_string().contains("canvas"));
    }
}
