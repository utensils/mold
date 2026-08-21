//! The three pixel operations the InsightFace pipeline performs outside its
//! ONNX graphs, ported from OpenCV's conventions rather than from a
//! general-purpose Rust image crate.
//!
//! Why not `image::imageops::resize` or `imageproc::warp`: both are correct
//! resamplers, and neither is OpenCV's. `imageops` scales its triangle filter
//! support with the ratio, so a 4x downscale averages a 8-tap window where
//! `cv2.resize(..., INTER_LINEAR)` takes exactly two taps at
//! `src = (dst + 0.5) * scale - 0.5`; the results differ by many LSBs on any
//! real photograph, and every landmark the detector reports is a function of
//! those pixels. `imageproc` is also not a mold dependency and is only in the
//! candle workspace, so adding it would pull a new tree into the Nix vendor set
//! for two functions. Both are implemented here instead, against upstream's
//! conventions, and pinned by goldens captured from OpenCV itself.
//!
//! Deliberate deviation: OpenCV evaluates both operations in fixed point
//! (`INTER_BITS = 5`, so interpolation weights are quantized to 1/32, and
//! `warpAffine` additionally quantizes source coordinates to `1/1024`). Mold
//! evaluates in `f64` and rounds once at the end. The fixtures record the
//! resulting per-pixel deltas; see `docs/architecture/pulid-face-extraction.md`.

use image::{Rgb, RgbImage};

/// A 2x3 affine matrix in OpenCV's row-major `warpAffine` convention: it maps
/// **source** coordinates to **destination** coordinates, and the warp inverts
/// it internally to sample.
pub type Affine2x3 = [[f64; 3]; 2];

/// Invert a 2x3 affine, matching `cv2.invertAffineTransform`.
///
/// Returns `None` for a singular matrix, which for a similarity fit means the
/// landmarks were collinear.
pub fn invert_affine(m: &Affine2x3) -> Option<Affine2x3> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < f64::EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    let a = m[1][1] * inv_det;
    let b = -m[0][1] * inv_det;
    let c = -m[1][0] * inv_det;
    let d = m[0][0] * inv_det;
    Some([
        [a, b, -a * m[0][2] - b * m[1][2]],
        [c, d, -c * m[0][2] - d * m[1][2]],
    ])
}

/// Apply a 2x3 affine to a point.
pub fn apply_affine(m: &Affine2x3, x: f64, y: f64) -> (f64, f64) {
    (
        m[0][0] * x + m[0][1] * y + m[0][2],
        m[1][0] * x + m[1][1] * y + m[1][2],
    )
}

#[inline]
fn clamp_u8(v: f64) -> u8 {
    // OpenCV saturates with round-half-away-from-zero.
    let r = v.round();
    if r <= 0.0 {
        0
    } else if r >= 255.0 {
        255
    } else {
        r as u8
    }
}

/// Bilinear sample with OpenCV's `BORDER_CONSTANT`: any tap outside the image
/// contributes `border`, exactly as `warpAffine` does.
fn sample_bilinear_constant(src: &RgbImage, x: f64, y: f64, border: [u8; 3]) -> [u8; 3] {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let x0 = x0 as i64;
    let y0 = y0 as i64;
    let (w, h) = (src.width() as i64, src.height() as i64);
    let tap = |px: i64, py: i64, channel: usize| -> f64 {
        if px < 0 || py < 0 || px >= w || py >= h {
            border[channel] as f64
        } else {
            src.get_pixel(px as u32, py as u32).0[channel] as f64
        }
    };
    let mut out = [0u8; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let v00 = tap(x0, y0, channel);
        let v01 = tap(x0 + 1, y0, channel);
        let v10 = tap(x0, y0 + 1, channel);
        let v11 = tap(x0 + 1, y0 + 1, channel);
        let top = v00 * (1.0 - fx) + v01 * fx;
        let bottom = v10 * (1.0 - fx) + v11 * fx;
        *slot = clamp_u8(top * (1.0 - fy) + bottom * fy);
    }
    out
}

/// Bilinear sample with edge clamping, which is what `cv2.resize` does at the
/// borders (it clamps the tap index rather than reading a constant).
fn sample_bilinear_clamped(src: &RgbImage, x: f64, y: f64) -> [u8; 3] {
    let (w, h) = (src.width() as i64, src.height() as i64);
    let mut x0 = x.floor() as i64;
    let mut y0 = y.floor() as i64;
    let mut fx = x - x0 as f64;
    let mut fy = y - y0 as f64;
    // OpenCV clamps the *coordinate*, which zeroes the fractional weight on
    // the far side. `resize.cpp`'s `computeResizeAreaTab` equivalent for
    // linear is `if (sx < 0) { fx = 0; sx = 0; }`.
    if x0 < 0 {
        x0 = 0;
        fx = 0.0;
    }
    if y0 < 0 {
        y0 = 0;
        fy = 0.0;
    }
    if x0 >= w - 1 {
        x0 = w - 1;
        fx = 0.0;
    }
    if y0 >= h - 1 {
        y0 = h - 1;
        fy = 0.0;
    }
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let mut out = [0u8; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let p = |px: i64, py: i64| src.get_pixel(px as u32, py as u32).0[channel] as f64;
        let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
        let bottom = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
        *slot = clamp_u8(top * (1.0 - fy) + bottom * fy);
    }
    out
}

/// `cv2.resize(src, (width, height))` with the default `INTER_LINEAR`.
///
/// Two taps per axis at `src = (dst + 0.5) * scale - 0.5`, clamped at the
/// borders. Deliberately not an area/`imageops` resample: see the module doc.
pub fn resize_bilinear(src: &RgbImage, width: u32, height: u32) -> RgbImage {
    assert!(width > 0 && height > 0, "resize target must be non-empty");
    let scale_x = src.width() as f64 / width as f64;
    let scale_y = src.height() as f64 / height as f64;
    let mut out = RgbImage::new(width, height);
    for y in 0..height {
        let sy = (y as f64 + 0.5) * scale_y - 0.5;
        for x in 0..width {
            let sx = (x as f64 + 0.5) * scale_x - 0.5;
            out.put_pixel(x, y, Rgb(sample_bilinear_clamped(src, sx, sy)));
        }
    }
    out
}

/// `cv2.warpAffine(src, m, (out_w, out_h), borderMode=BORDER_CONSTANT,
/// borderValue=border)` with the default `INTER_LINEAR`.
///
/// `m` maps source to destination; this inverts it to sample, exactly as
/// OpenCV does when `WARP_INVERSE_MAP` is not set.
pub fn warp_affine(
    src: &RgbImage,
    m: &Affine2x3,
    out_w: u32,
    out_h: u32,
    border: [u8; 3],
) -> Option<RgbImage> {
    let inv = invert_affine(m)?;
    let mut out = RgbImage::new(out_w, out_h);
    for y in 0..out_h {
        for x in 0..out_w {
            let (sx, sy) = apply_affine(&inv, x as f64, y as f64);
            out.put_pixel(x, y, Rgb(sample_bilinear_constant(src, sx, sy, border)));
        }
    }
    Some(out)
}

/// The detector's letterbox, ported from
/// `insightface/python-package/insightface/model_zoo/scrfd.py:459-470`
/// (`SCRFD._detect_candidates`).
///
/// The image is scaled to fit `size` on its longer axis preserving aspect and
/// pasted into the TOP-LEFT of a zero-filled `size x size` canvas — not
/// centred. Every coordinate the detector reports is divided by the returned
/// scale to land back in source pixels, which is why the offset must stay
/// zero.
#[derive(Debug, Clone)]
pub struct Letterboxed {
    /// The `size x size` canvas handed to the graph.
    pub image: RgbImage,
    /// `new_height / src_height`; detections divide by this.
    pub det_scale: f64,
}

/// Letterbox an image for SCRFD at `size` pixels square.
pub fn letterbox_top_left(src: &RgbImage, size: u32) -> Letterboxed {
    let (sw, sh) = (src.width() as f64, src.height() as f64);
    let im_ratio = sh / sw;
    // Square model input, so `model_ratio` is 1.0 and the comparison reduces
    // to "is the source taller than wide".
    let (new_w, new_h) = if im_ratio > 1.0 {
        let new_h = size as f64;
        // Upstream truncates with `int()`, not `round()`.
        ((new_h / im_ratio) as u32, size)
    } else {
        let new_w = size as f64;
        (size, (new_w * im_ratio) as u32)
    };
    let new_w = new_w.max(1);
    let new_h = new_h.max(1);
    let det_scale = new_h as f64 / sh;
    let resized = resize_bilinear(src, new_w, new_h);
    let mut canvas = RgbImage::new(size, size);
    for y in 0..new_h.min(size) {
        for x in 0..new_w.min(size) {
            canvas.put_pixel(x, y, *resized.get_pixel(x, y));
        }
    }
    Letterboxed {
        image: canvas,
        det_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, colour: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(colour))
    }

    #[test]
    fn an_identity_warp_reproduces_the_source() {
        let mut src = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                src.put_pixel(x, y, Rgb([(x * 60) as u8, (y * 60) as u8, 7]));
            }
        }
        let identity: Affine2x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out = warp_affine(&src, &identity, 4, 4, [0, 0, 0]).unwrap();
        assert_eq!(out.as_raw(), src.as_raw());
    }

    #[test]
    fn a_warp_outside_the_image_reads_the_border_colour() {
        let src = solid(2, 2, [10, 20, 30]);
        // Translate the source far off-canvas.
        let m: Affine2x3 = [[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]];
        let out = warp_affine(&src, &m, 2, 2, [135, 133, 132]).unwrap();
        for px in out.pixels() {
            assert_eq!(px.0, [135, 133, 132]);
        }
    }

    #[test]
    fn a_singular_affine_is_none_not_a_panic() {
        let src = solid(2, 2, [1, 2, 3]);
        let singular: Affine2x3 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        assert!(invert_affine(&singular).is_none());
        assert!(warp_affine(&src, &singular, 2, 2, [0, 0, 0]).is_none());
    }

    #[test]
    fn inverting_an_affine_round_trips_a_point() {
        let m: Affine2x3 = [[2.0, 0.5, -3.0], [-0.25, 1.5, 7.0]];
        let inv = invert_affine(&m).unwrap();
        let (x, y) = apply_affine(&m, 11.0, -4.0);
        let (rx, ry) = apply_affine(&inv, x, y);
        assert!((rx - 11.0).abs() < 1e-9, "{rx}");
        assert!((ry + 4.0).abs() < 1e-9, "{ry}");
    }

    #[test]
    fn resizing_a_solid_image_preserves_its_colour() {
        let src = solid(37, 91, [200, 100, 50]);
        let out = resize_bilinear(&src, 640, 640);
        for px in out.pixels() {
            assert_eq!(px.0, [200, 100, 50]);
        }
    }

    #[test]
    fn resize_uses_half_pixel_centres() {
        // A 2x1 ramp upscaled 2x: OpenCV's half-pixel rule puts the output
        // samples at src x = -0.25, 0.25, 0.75, 1.25 which clamp/interpolate
        // to 0, 0.25, 0.75, 1.0 of the way across the ramp.
        let mut src = RgbImage::new(2, 1);
        src.put_pixel(0, 0, Rgb([0, 0, 0]));
        src.put_pixel(1, 0, Rgb([100, 100, 100]));
        let out = resize_bilinear(&src, 4, 1);
        let values: Vec<u8> = out.pixels().map(|p| p.0[0]).collect();
        assert_eq!(values, vec![0, 25, 75, 100]);
    }

    #[test]
    fn a_landscape_letterbox_pins_the_image_to_the_top_left() {
        let src = solid(1000, 500, [9, 9, 9]);
        let boxed = letterbox_top_left(&src, 640);
        assert_eq!(boxed.image.dimensions(), (640, 640));
        // 640 wide, 320 tall.
        assert!((boxed.det_scale - 320.0 / 500.0).abs() < 1e-12);
        assert_eq!(boxed.image.get_pixel(0, 0).0, [9, 9, 9]);
        assert_eq!(boxed.image.get_pixel(639, 319).0, [9, 9, 9]);
        // Everything below the pasted region is the zero fill upstream uses.
        assert_eq!(boxed.image.get_pixel(0, 320).0, [0, 0, 0]);
        assert_eq!(boxed.image.get_pixel(639, 639).0, [0, 0, 0]);
    }

    #[test]
    fn a_portrait_letterbox_scales_on_height() {
        let src = solid(500, 1000, [4, 5, 6]);
        let boxed = letterbox_top_left(&src, 640);
        assert!((boxed.det_scale - 640.0 / 1000.0).abs() < 1e-12);
        assert_eq!(boxed.image.get_pixel(319, 639).0, [4, 5, 6]);
        assert_eq!(boxed.image.get_pixel(320, 639).0, [0, 0, 0]);
    }

    #[test]
    fn a_square_letterbox_fills_the_canvas() {
        let src = solid(700, 700, [1, 2, 3]);
        let boxed = letterbox_top_left(&src, 640);
        assert!((boxed.det_scale - 640.0 / 700.0).abs() < 1e-12);
        assert_eq!(boxed.image.get_pixel(639, 639).0, [1, 2, 3]);
    }
}
