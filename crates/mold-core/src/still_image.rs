//! Header-only facts about encoded still images.
//!
//! WebP is both a still and an animation container, so its
//! [`crate::OutputFormat`] alone cannot say which a file is: the bytes do.
//! These readers look at container headers only — never decode pixels — so
//! every surface (server publication, the CLI's terminal preview, gallery
//! classification) can ask the same question cheaply and agree.

/// Whether a RIFF/WEBP buffer is an animation rather than a still.
///
/// True when the `VP8X` extended header sets its animation flag or the file
/// carries an `ANIM`/`ANMF` chunk (libwebp's container spec,
/// `doc/webp-container-spec.txt`). A simple-format still (`VP8 ` / `VP8L`)
/// and a non-WebP buffer are both `false`.
pub fn webp_is_animated(bytes: &[u8]) -> bool {
    if !is_webp(bytes) {
        return false;
    }
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let fourcc = &bytes[offset..offset + 4];
        let len = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        match fourcc {
            // Flags byte: Rsv(2) I(1) L(1) E(1) X(1) A(1) R(1), MSB first;
            // `A` (0x02) is the animation bit.
            b"VP8X" if offset + 9 <= bytes.len() && bytes[offset + 8] & 0x02 != 0 => {
                return true;
            }
            b"ANIM" | b"ANMF" => return true,
            _ => {}
        }
        offset = match offset.checked_add(8 + len + (len & 1)) {
            Some(next) => next,
            None => break,
        };
    }
    false
}

/// Whether an encoded PNG or WebP still declares an alpha channel.
///
/// - PNG: the `IHDR` colour type is 4 (grey + alpha) or 6 (RGBA), or a
///   `tRNS` chunk precedes the image data.
/// - WebP: the `VP8X` alpha flag, or a lossless `VP8L` bitstream whose
///   `alpha_is_used` hint is set.
/// - Anything else (JPEG, unknown bytes): `false`.
///
/// mold's own still encoder writes an alpha-bearing container only when at
/// least one alpha byte is below 255, so for mold-written prints this is
/// exactly "the stored pixels carry transparency".
pub fn encoded_still_has_alpha(bytes: &[u8]) -> bool {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return png_has_alpha(bytes);
    }
    if is_webp(bytes) {
        return webp_has_alpha(bytes);
    }
    false
}

fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

fn png_has_alpha(bytes: &[u8]) -> bool {
    let mut offset = 8;
    while offset + 8 <= bytes.len() {
        let len = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let kind = &bytes[offset + 4..offset + 8];
        match kind {
            // IHDR: width(4) height(4) depth(1) colour type(1) ...
            b"IHDR" if offset + 18 <= bytes.len() && matches!(bytes[offset + 17], 4 | 6) => {
                return true;
            }
            b"tRNS" => return true,
            b"IDAT" | b"IEND" => return false,
            _ => {}
        }
        offset = match offset.checked_add(12 + len) {
            Some(next) => next,
            None => break,
        };
    }
    false
}

fn webp_has_alpha(bytes: &[u8]) -> bool {
    let Some(fourcc) = bytes.get(12..16) else {
        return false;
    };
    match fourcc {
        // Flags byte: `L` (0x10) is the alpha bit.
        b"VP8X" => bytes.get(20).is_some_and(|flags| flags & 0x10 != 0),
        // VP8L header: signature 0x2F at byte 20, then 14-bit width-1,
        // 14-bit height-1 and the 1-bit `alpha_is_used` hint (bit 28 of the
        // little-endian word following the signature).
        b"VP8L" => {
            let Some(header) = bytes.get(21..25) else {
                return false;
            };
            bytes.get(20) == Some(&0x2F)
                && (u32::from_le_bytes([header[0], header[1], header[2], header[3]]) >> 28) & 1 == 1
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn riff(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut body = b"WEBP".to_vec();
        for (fourcc, payload) in chunks {
            body.extend_from_slice(*fourcc);
            body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            body.extend_from_slice(payload);
            if payload.len() % 2 == 1 {
                body.push(0);
            }
        }
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn a_simple_webp_still_is_not_animated() {
        let still = riff(&[(b"VP8 ", vec![0; 10])]);
        assert!(!webp_is_animated(&still));
        assert!(!encoded_still_has_alpha(&still));
    }

    #[test]
    fn the_vp8x_animation_flag_or_an_anim_chunk_marks_an_animation() {
        let flagged = riff(&[(b"VP8X", vec![0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0])]);
        assert!(webp_is_animated(&flagged));
        let chunked = riff(&[
            (b"VP8X", vec![0; 10]),
            (b"ANIM", vec![0; 6]),
            (b"ANMF", vec![0; 16]),
        ]);
        assert!(webp_is_animated(&chunked));
    }

    #[test]
    fn webp_alpha_comes_from_the_vp8x_flag_or_the_vp8l_hint() {
        let alpha = riff(&[(b"VP8X", vec![0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0])]);
        assert!(encoded_still_has_alpha(&alpha));
        assert!(!webp_is_animated(&alpha));
        let lossless_alpha = riff(&[(b"VP8L", vec![0x2F, 0, 0, 0, 0x10])]);
        assert!(encoded_still_has_alpha(&lossless_alpha));
        let lossless_opaque = riff(&[(b"VP8L", vec![0x2F, 0, 0, 0, 0x00])]);
        assert!(!encoded_still_has_alpha(&lossless_opaque));
    }

    fn png_with(color_type: u8, extra: Option<&[u8; 4]>) -> Vec<u8> {
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        out.extend_from_slice(&13u32.to_be_bytes());
        out.extend_from_slice(b"IHDR");
        out.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 1, 8, color_type, 0, 0, 0]);
        out.extend_from_slice(&[0; 4]);
        if let Some(kind) = extra {
            out.extend_from_slice(&1u32.to_be_bytes());
            out.extend_from_slice(kind);
            out.push(0);
            out.extend_from_slice(&[0; 4]);
        }
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(b"IEND");
        out.extend_from_slice(&[0; 4]);
        out
    }

    #[test]
    fn png_alpha_is_the_colour_type_or_a_trns_chunk() {
        assert!(!encoded_still_has_alpha(&png_with(2, None)));
        assert!(encoded_still_has_alpha(&png_with(6, None)));
        assert!(encoded_still_has_alpha(&png_with(4, None)));
        assert!(encoded_still_has_alpha(&png_with(2, Some(b"tRNS"))));
    }

    #[test]
    fn jpeg_and_garbage_carry_no_alpha() {
        assert!(!encoded_still_has_alpha(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!encoded_still_has_alpha(b"RIFF"));
        assert!(!webp_is_animated(b"not a webp at all"));
    }
}
