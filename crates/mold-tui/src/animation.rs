//! Decoding helpers for animated previews (GIF/APNG/WebP).
//!
//! The TUI gallery detail view and generation viewport need frame-by-frame
//! playback for video previews, but `ratatui-image` only renders a single
//! image. This module decodes an animated file into a `Vec<AnimatedFrame>`
//! that callers can advance on a timer.

use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Seek};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use image::{AnimationDecoder, DynamicImage};

/// A single decoded animation frame plus its delay before the next frame.
#[derive(Debug, Clone)]
pub struct AnimatedFrame {
    pub image: DynamicImage,
    pub delay: Duration,
}

/// Per-source animation state used by the TUI to drive frame advancement.
pub struct AnimationState {
    pub frames: Vec<AnimatedFrame>,
    pub current: usize,
    pub last_advance: Instant,
}

impl AnimationState {
    pub fn new(frames: Vec<AnimatedFrame>) -> Option<Self> {
        if frames.len() < 2 {
            return None;
        }
        Some(Self {
            frames,
            current: 0,
            last_advance: Instant::now(),
        })
    }

    /// Advance to the next frame if the current frame's delay has elapsed.
    /// Returns `true` when the frame index changed (caller should rebuild
    /// the image protocol).
    pub fn tick(&mut self) -> bool {
        if self.frames.len() < 2 {
            return false;
        }
        let delay = self.frames[self.current].delay;
        if self.last_advance.elapsed() < delay {
            return false;
        }
        self.current = (self.current + 1) % self.frames.len();
        self.last_advance = Instant::now();
        true
    }

    pub fn current_image(&self) -> &DynamicImage {
        &self.frames[self.current].image
    }
}

/// Decode an animated file from disk. Supports GIF, APNG, and WebP. Returns
/// `Err` for non-animated or unsupported inputs.
pub fn decode_animation_path(path: &Path) -> Result<Vec<AnimatedFrame>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    decode_with_hint(reader, ext.as_deref(), Some(path))
}

/// Decode animation from raw bytes. `hint_ext` is an optional file extension
/// (without the dot) used to dispatch to the right codec; when omitted we
/// sniff the magic bytes.
pub fn decode_animation_bytes(data: &[u8], hint_ext: Option<&str>) -> Result<Vec<AnimatedFrame>> {
    let ext = hint_ext
        .map(|s| s.to_lowercase())
        .or_else(|| sniff_ext(data));
    decode_with_hint(Cursor::new(data), ext.as_deref(), None)
}

/// Cheap check: does this byte buffer look like an animated container we
/// know how to play (GIF/APNG/WebP)? Used by the CLI preview to decide
/// whether to call into the animated playback path.
pub fn is_animated_bytes(data: &[u8]) -> bool {
    match sniff_ext(data).as_deref() {
        Some("gif") => true,
        Some("png") => is_apng(data),
        Some("webp") => is_animated_webp(data),
        _ => false,
    }
}

fn sniff_ext(data: &[u8]) -> Option<String> {
    if data.len() >= 6 && (data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a")) {
        return Some("gif".into());
    }
    if data.len() >= 8 && data[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some("png".into());
    }
    if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return Some("webp".into());
    }
    None
}

fn is_apng(data: &[u8]) -> bool {
    // After the 8-byte PNG signature, scan chunks for an `acTL` (animation
    // control) chunk, which is the apng marker. Stop at IDAT (start of
    // image data) since acTL must precede IDAT.
    if data.len() < 8 || &data[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A].as_slice() {
        return false;
    }
    let mut i = 8usize;
    while i + 8 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        let kind = &data[i + 4..i + 8];
        if kind == b"acTL" {
            return true;
        }
        if kind == b"IDAT" {
            return false;
        }
        // Advance past length(4) + type(4) + data(len) + crc(4)
        let next = i.saturating_add(8).saturating_add(len).saturating_add(4);
        if next <= i {
            return false;
        }
        i = next;
    }
    false
}

fn is_animated_webp(data: &[u8]) -> bool {
    // RIFF....WEBPVP8X<flags>... — animated bit (0x02) set in VP8X flags.
    if data.len() < 21 || &data[12..16] != b"VP8X" {
        return false;
    }
    let flags = data[20];
    flags & 0x02 != 0
}

fn decode_with_hint<R: BufRead + Seek>(
    reader: R,
    ext: Option<&str>,
    path: Option<&Path>,
) -> Result<Vec<AnimatedFrame>> {
    match ext {
        Some("gif") => collect(image::codecs::gif::GifDecoder::new(reader)?.into_frames()),
        Some("png") | Some("apng") => {
            let decoder = image::codecs::png::PngDecoder::new(reader)?;
            let apng = decoder
                .apng()
                .map_err(|e| anyhow!("not an animated PNG: {e}"))?;
            collect(apng.into_frames())
        }
        Some("webp") => collect(image::codecs::webp::WebPDecoder::new(reader)?.into_frames()),
        other => Err(anyhow!(
            "unsupported animation format: {} ({})",
            other.unwrap_or("<unknown>"),
            path.map(|p| p.display().to_string())
                .unwrap_or_else(|| "<bytes>".into())
        )),
    }
}

fn collect<I: Iterator<Item = image::ImageResult<image::Frame>>>(
    frames: I,
) -> Result<Vec<AnimatedFrame>> {
    let mut out = Vec::new();
    for frame in frames {
        let frame = frame.context("decoding animation frame")?;
        let (num, den) = frame.delay().numer_denom_ms();
        // `numer_denom_ms()` returns the delay in milliseconds as a ratio
        // num/den. Convert to microseconds for sub-millisecond accuracy and
        // clamp at ~50 fps so a malformed `0/0` or absurdly small delay
        // can't peg the event loop.
        let micros = if den == 0 {
            100_000
        } else {
            (u64::from(num) * 1000) / u64::from(den.max(1))
        };
        let delay = Duration::from_micros(micros).max(Duration::from_millis(20));
        let image = DynamicImage::ImageRgba8(frame.into_buffer());
        out.push(AnimatedFrame { image, delay });
    }
    if out.is_empty() {
        return Err(anyhow!("animation contained no frames"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::gif::GifEncoder;
    use image::Frame;

    fn synth_gif(frames: u32, w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            enc.set_repeat(image::codecs::gif::Repeat::Infinite)
                .unwrap();
            for i in 0..frames {
                let img = image::RgbaImage::from_pixel(
                    w,
                    h,
                    image::Rgba([(i * 30) as u8 % 255, 0, 0, 255]),
                );
                let delay = image::Delay::from_numer_denom_ms(80, 1);
                enc.encode_frame(Frame::from_parts(img, 0, 0, delay))
                    .unwrap();
            }
        }
        buf
    }

    #[test]
    fn sniff_gif_signature() {
        let bytes = synth_gif(2, 4, 4);
        assert_eq!(sniff_ext(&bytes).as_deref(), Some("gif"));
        assert!(is_animated_bytes(&bytes));
    }

    #[test]
    fn decode_multiframe_gif() {
        let bytes = synth_gif(5, 8, 8);
        let frames = decode_animation_bytes(&bytes, Some("gif")).unwrap();
        assert_eq!(frames.len(), 5);
        for f in &frames {
            assert_eq!(f.image.width(), 8);
            assert_eq!(f.image.height(), 8);
            assert!(f.delay >= Duration::from_millis(20));
        }
    }

    #[test]
    fn animation_state_advances_after_delay() {
        let bytes = synth_gif(3, 4, 4);
        let frames = decode_animation_bytes(&bytes, Some("gif")).unwrap();
        let mut state = AnimationState::new(frames).expect("multi-frame");
        assert_eq!(state.current, 0);
        // Force the delay to have elapsed.
        state.last_advance = Instant::now() - Duration::from_secs(1);
        assert!(state.tick());
        assert_eq!(state.current, 1);
        // Immediately ticking again should be a no-op (delay not elapsed).
        assert!(!state.tick());
    }

    #[test]
    fn animation_state_wraps_around() {
        let bytes = synth_gif(2, 4, 4);
        let frames = decode_animation_bytes(&bytes, Some("gif")).unwrap();
        let mut state = AnimationState::new(frames).unwrap();
        state.last_advance = Instant::now() - Duration::from_secs(1);
        assert!(state.tick());
        assert_eq!(state.current, 1);
        state.last_advance = Instant::now() - Duration::from_secs(1);
        assert!(state.tick());
        assert_eq!(state.current, 0);
    }

    #[test]
    fn animation_state_rejects_single_frame() {
        let bytes = synth_gif(1, 4, 4);
        let frames = decode_animation_bytes(&bytes, Some("gif")).unwrap();
        assert!(AnimationState::new(frames).is_none());
    }

    #[test]
    fn sniff_ignores_non_animated_inputs() {
        assert!(sniff_ext(b"not an image").is_none());
        assert!(!is_animated_bytes(b"not an image"));
    }

    #[test]
    fn unsupported_format_errors() {
        // JPEG bytes — not in the supported animated set.
        let mut buf = Vec::new();
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Jpeg)
            .unwrap();
        assert!(decode_animation_bytes(&buf, Some("jpeg")).is_err());
    }
}
