//! Rounded BT.601 limited-range RGB <-> YUV 4:2:0, the one conversion pair
//! every H.264 encode and decode in mold uses.
//!
//! The openh264 crate's own helpers (`YUVBuffer::from_rgb_source`,
//! `DecodedYUV::write_rgb8`) TRUNCATE with `as u8` on both sides, so every
//! encode/decode cycle shaves about one code off every channel. One cycle is
//! invisible; a chained video pays it on every carried boundary frame, and a
//! dark scene lost ~13% of its luma per clip (2026-10-23, plato). Rounding
//! makes the round trip unbiased.

use anyhow::{Context, Result};
use image::RgbImage;

/// Planar YUV 4:2:0 for an even-sized RGB frame. Chroma is computed from the
/// 2x2 box-averaged RGB block, matching swscale's default subsampling closely
/// enough that flat colors survive within a code.
pub(crate) fn rgb_to_yuv420(rgb: &RgbImage) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);
    let src = rgb.as_raw();
    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; (w / 2) * (h / 2)];
    let mut v_plane = vec![0u8; (w / 2) * (h / 2)];
    let clamp = |v: f32| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for row in 0..h {
        for col in 0..w {
            let i = (row * w + col) * 3;
            let (r, g, b) = (
                f32::from(src[i]),
                f32::from(src[i + 1]),
                f32::from(src[i + 2]),
            );
            y_plane[row * w + col] = clamp(0.256_788 * r + 0.504_129 * g + 0.097_906 * b + 16.0);
        }
    }
    for row in 0..h / 2 {
        for col in 0..w / 2 {
            let mut sums = [0.0f32; 3];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let i = ((row * 2 + dy) * w + col * 2 + dx) * 3;
                sums[0] += f32::from(src[i]);
                sums[1] += f32::from(src[i + 1]);
                sums[2] += f32::from(src[i + 2]);
            }
            let (r, g, b) = (sums[0] / 4.0, sums[1] / 4.0, sums[2] / 4.0);
            u_plane[row * (w / 2) + col] =
                clamp(-0.148_223 * r - 0.290_993 * g + 0.439_216 * b + 128.0);
            v_plane[row * (w / 2) + col] =
                clamp(0.439_216 * r - 0.367_788 * g - 0.071_427 * b + 128.0);
        }
    }
    (y_plane, u_plane, v_plane)
}

/// RGB from a decoded picture, reading the decoder's strided planes directly
/// (each 2x2 block shares one chroma sample). Inverse of [`rgb_to_yuv420`] up
/// to codec quantization. `w`/`h` may be smaller than the decoded picture.
pub(crate) fn decoded_to_rgb(
    frame: &openh264::decoder::DecodedYUV<'_>,
    w: u32,
    h: u32,
) -> Result<RgbImage> {
    use openh264::formats::YUVSource;
    let (dec_w, dec_h) = frame.dimensions();
    anyhow::ensure!(
        dec_w >= w as usize && dec_h >= h as usize,
        "decoded frame ({dec_w}x{dec_h}) is smaller than the expected {w}x{h}"
    );
    let (stride_y, stride_u, stride_v) = frame.strides();
    let (y_plane, u_plane, v_plane) = (frame.y(), frame.u(), frame.v());
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0u8; w * h * 3];
    let clamp = |v: f32| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for row in 0..h {
        for col in 0..w {
            let y = f32::from(y_plane[row * stride_y + col]) - 16.0;
            let u = f32::from(u_plane[(row / 2) * stride_u + col / 2]) - 128.0;
            let v = f32::from(v_plane[(row / 2) * stride_v + col / 2]) - 128.0;
            let i = (row * w + col) * 3;
            out[i] = clamp(1.164_384 * y + 1.596_027 * v);
            out[i + 1] = clamp(1.164_384 * y - 0.391_762 * u - 0.812_968 * v);
            out[i + 2] = clamp(1.164_384 * y + 2.017_232 * u);
        }
    }
    RgbImage::from_raw(w as u32, h as u32, out).context("decoded frame size mismatch")
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    /// A dark, textured scene like the one that drifted: mean around 26.
    fn dark_scene() -> RgbImage {
        RgbImage::from_fn(64, 48, |x, y| {
            let base = 18 + ((x * 3 + y * 5) % 17) as u8;
            Rgb([base, base.saturating_add(4), base.saturating_sub(3)])
        })
    }

    fn mean(image: &RgbImage) -> f64 {
        image.as_raw().iter().map(|&v| f64::from(v)).sum::<f64>() / image.as_raw().len() as f64
    }

    /// Carrying a boundary frame through five encode/decode hops must not
    /// change its exposure. With the crate's truncating helpers each hop lost
    /// about a code; this is the chained-video darkening regression.
    #[test]
    fn five_carried_hops_through_the_mp4_path_keep_exposure() {
        let source = dark_scene();
        let mut carried = source.clone();
        for _ in 0..5 {
            let bytes =
                crate::ltx_video::video_enc::encode_mp4(&vec![carried.clone(); 3], 24).unwrap();
            let file = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(file.path(), bytes).unwrap();
            carried = crate::video_assembly_frames::tail(
                std::fs::File::open(file.path()).unwrap(),
                3,
                1,
                64,
                48,
                &mut || Ok(()),
            )
            .unwrap()
            .pop()
            .unwrap();
        }
        let drift = mean(&carried) - mean(&source);
        assert!(drift.abs() < 0.75, "five hops drifted {drift:+.2} codes");
    }
}
