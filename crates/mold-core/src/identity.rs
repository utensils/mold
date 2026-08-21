//! Face-identity conditioning contract (PuLID-FLUX).
//!
//! This module is the single authority for every identity-conditioning
//! constant on the wire: which models are qualified to accept an identity
//! image, the `id_weight` range, the `id_start_step` default, and the bounded
//! decode limits an `id_image` payload must respect before anything in the
//! process attempts a full decode.
//!
//! The runtime adapter lands separately. Everything here is contract only, so
//! it is deliberately feature-independent: a build without `pulid` still
//! validates the request shape identically and simply advertises
//! `supports_identity = false`, which is what stops a client from queueing
//! work the binary cannot execute.

use crate::types::GenerateRequest;

/// Models qualified to accept identity conditioning in milestone 1.
///
/// These are resolved manifest names. A request naming the bare `flux-dev`
/// resolves to `flux-dev:q8` through [`crate::manifest::resolve_model_name`],
/// and the legacy dash form `flux-dev-q4` resolves to `flux-dev:q4`, so both
/// are accepted; `flux-dev:bf16` and every other checkpoint are not.
pub const IDENTITY_QUALIFIED_MODELS: &[&str] = &["flux-dev:q4", "flux-dev:q8"];

/// Default `id_weight` when a request supplies an identity image without one.
pub const ID_WEIGHT_DEFAULT: f64 = 1.0;

/// Inclusive upper bound for `id_weight`. The range is `0.0..=ID_WEIGHT_MAX`.
pub const ID_WEIGHT_MAX: f64 = 3.0;

/// Default `id_start_step` — identity conditioning is applied from the first
/// denoise step unless the request delays it.
pub const ID_START_STEP_DEFAULT: u32 = 0;

/// Refusal a build without the identity adapter gives a request that asks
/// for identity conditioning.
///
/// Accept-and-ignore is not an option: the print would render without the
/// face and nothing would say so. The request is refused instead, and the
/// message names the missing build support rather than blaming the model.
pub const IDENTITY_BUILD_UNSUPPORTED: &str =
    "this server was built without PuLID face-identity support; remove id_image \
     (and any id_weight, id_start_step, or id_image_name) or use a server built \
     with the `pulid` feature";

/// Bounded-decode limits for an `id_image` payload.
///
/// Every limit is checked from the encoded header alone, before any decoder
/// is handed the bytes, so a malicious or malformed image can never make the
/// server allocate a decompression bomb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdImageLimits {
    /// Maximum length of the encoded payload as it arrives on the wire.
    pub max_encoded_bytes: usize,
    /// Maximum width or height, in pixels, read from the image header.
    pub max_axis_pixels: u32,
    /// Maximum `width * height` read from the image header.
    pub max_decoded_pixels: u64,
    /// Maximum bytes a full RGBA8 decode of the header-declared dimensions
    /// would occupy. Derived from [`Self::max_decoded_pixels`]; kept explicit
    /// so the refusal can name the allocation it prevented.
    pub max_decode_allocation_bytes: u64,
}

/// The one set of identity-image limits every surface enforces.
pub const ID_IMAGE_LIMITS: IdImageLimits = IdImageLimits {
    max_encoded_bytes: 16 * 1024 * 1024,
    max_axis_pixels: 8192,
    max_decoded_pixels: 32_000_000,
    max_decode_allocation_bytes: 32_000_000 * 4,
};

/// Whether `model` is qualified for identity conditioning.
///
/// The name is resolved first, so callers may pass whatever the request
/// carried — bare, tagged, or the legacy dash form.
pub fn identity_qualified_model(resolved_model: &str) -> bool {
    let resolved = crate::manifest::resolve_model_name(resolved_model);
    IDENTITY_QUALIFIED_MODELS.contains(&resolved.as_str())
}

/// Whether the request asks for identity conditioning in any way — including
/// the incomplete forms (a knob without an image) validation must refuse.
pub fn request_mentions_identity(req: &GenerateRequest) -> bool {
    req.id_image.is_some()
        || req.id_image_name.is_some()
        || req.id_weight.is_some()
        || req.id_start_step.is_some()
}

/// The `id_weight` that will actually be applied.
pub fn effective_id_weight(req: &GenerateRequest) -> f64 {
    req.id_weight.unwrap_or(ID_WEIGHT_DEFAULT)
}

/// The `id_start_step` that will actually be applied.
pub fn effective_id_start_step(req: &GenerateRequest) -> u32 {
    req.id_start_step.unwrap_or(ID_START_STEP_DEFAULT)
}

/// Header-declared pixel dimensions of a PNG or JPEG payload.
///
/// Only the container header is parsed — never the compressed image data — so
/// this cannot be made to allocate on the caller's behalf.
fn header_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    if bytes.starts_with(&PNG_SIGNATURE) {
        // The IHDR chunk is mandatory and must be first: 4-byte length,
        // 4-byte type, then width and height as big-endian u32.
        if bytes.len() < 24 || &bytes[12..16] != b"IHDR" {
            return Err("id_image is a truncated or malformed PNG: no IHDR header".to_string());
        }
        let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        return Ok((width, height));
    }

    // JPEG: walk the marker segments to the first start-of-frame, which is
    // the only place the dimensions live.
    let mut offset = 2usize;
    while offset + 1 < bytes.len() {
        if bytes[offset] != 0xFF {
            return Err("id_image is a malformed JPEG: lost marker alignment".to_string());
        }
        // Fill bytes before a marker are legal padding.
        let mut marker_at = offset + 1;
        while marker_at < bytes.len() && bytes[marker_at] == 0xFF {
            marker_at += 1;
        }
        let Some(&marker) = bytes.get(marker_at) else {
            break;
        };
        // Standalone markers carry no payload.
        if marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            offset = marker_at + 1;
            continue;
        }
        let length_at = marker_at + 1;
        if length_at + 1 >= bytes.len() {
            break;
        }
        let length = u16::from_be_bytes([bytes[length_at], bytes[length_at + 1]]) as usize;
        if length < 2 {
            return Err("id_image is a malformed JPEG: invalid segment length".to_string());
        }
        // SOF0..SOF15 except DHT (0xC4), JPG (0xC8) and DAC (0xCC).
        let is_start_of_frame =
            (0xC0..=0xCF).contains(&marker) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if is_start_of_frame {
            // precision(1) height(2) width(2)
            if length < 7 || length_at + 6 >= bytes.len() {
                return Err(
                    "id_image is a truncated JPEG: incomplete start-of-frame header".to_string(),
                );
            }
            let height = u16::from_be_bytes([bytes[length_at + 3], bytes[length_at + 4]]);
            let width = u16::from_be_bytes([bytes[length_at + 5], bytes[length_at + 6]]);
            return Ok((u32::from(width), u32::from(height)));
        }
        offset = length_at + length;
    }
    Err("id_image is a truncated JPEG: no start-of-frame header".to_string())
}

/// Validate an `id_image` payload without decoding it.
///
/// The checks run in a deliberate order — encoded length, then magic bytes,
/// then the header-declared dimensions — so an oversized payload is refused
/// on its size alone and no decoder ever sees the bytes.
pub fn validate_id_image_bytes(bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() {
        return Err("id_image must not be empty".to_string());
    }
    if bytes.len() > ID_IMAGE_LIMITS.max_encoded_bytes {
        return Err(format!(
            "id_image is {} bytes, which exceeds the {} byte (16 MiB) limit",
            bytes.len(),
            ID_IMAGE_LIMITS.max_encoded_bytes
        ));
    }

    const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];
    let is_png = bytes.starts_with(&PNG_SIGNATURE);
    let is_jpeg = bytes.starts_with(&JPEG_MAGIC);
    if !is_png && !is_jpeg {
        return Err("id_image must be a PNG or JPEG image".to_string());
    }

    let (width, height) = header_dimensions(bytes)?;
    if width == 0 || height == 0 {
        return Err("id_image declares a zero dimension".to_string());
    }
    if width > ID_IMAGE_LIMITS.max_axis_pixels || height > ID_IMAGE_LIMITS.max_axis_pixels {
        return Err(format!(
            "id_image is {width}x{height}; each axis must be at most {} pixels",
            ID_IMAGE_LIMITS.max_axis_pixels
        ));
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > ID_IMAGE_LIMITS.max_decoded_pixels {
        return Err(format!(
            "id_image is {width}x{height} ({pixels} pixels), which exceeds the {} pixel limit",
            ID_IMAGE_LIMITS.max_decoded_pixels
        ));
    }
    // Redundant with the pixel bound today, but it is the bound that actually
    // matters, so it is checked rather than assumed.
    let allocation = pixels.saturating_mul(4);
    if allocation > ID_IMAGE_LIMITS.max_decode_allocation_bytes {
        return Err(format!(
            "decoding id_image would allocate {allocation} bytes, above the {} byte limit",
            ID_IMAGE_LIMITS.max_decode_allocation_bytes
        ));
    }
    Ok(())
}

/// Validate the complete identity-conditioning partition of a request.
///
/// A request that mentions no identity field at all is untouched, so this is
/// safe to call unconditionally from the shared generate-request validator.
pub fn validate_identity_conditioning(req: &GenerateRequest) -> Result<(), String> {
    if !request_mentions_identity(req) {
        return Ok(());
    }

    // A binary that cannot execute identity conditioning refuses the request
    // rather than rendering a print that silently has no face in it.
    if !cfg!(feature = "pulid") {
        return Err(IDENTITY_BUILD_UNSUPPORTED.to_string());
    }

    let Some(bytes) = req.id_image.as_deref() else {
        return Err(
            "id_image is required when id_weight, id_start_step, or id_image_name is set"
                .to_string(),
        );
    };

    if !identity_qualified_model(&req.model) {
        return Err(format!(
            "{} does not support face-identity conditioning; identity is qualified only for {}",
            req.model,
            IDENTITY_QUALIFIED_MODELS.join(" and ")
        ));
    }

    let has_lora = req.lora.is_some() || req.loras.as_ref().is_some_and(|items| !items.is_empty());
    if has_lora {
        return Err(
            "face-identity conditioning combined with a LoRA is not yet qualified; \
             remove the LoRA or the id_image"
                .to_string(),
        );
    }
    if req.source_image.is_some() {
        return Err(
            "face-identity conditioning combined with img2img is not yet qualified; \
             remove the source_image or the id_image"
                .to_string(),
        );
    }

    let weight = effective_id_weight(req);
    if !weight.is_finite() || !(0.0..=ID_WEIGHT_MAX).contains(&weight) {
        return Err(format!(
            "id_weight ({weight}) must be a finite value in range [0.0, {ID_WEIGHT_MAX}]"
        ));
    }

    let start_step = effective_id_start_step(req);
    if start_step >= req.steps {
        return Err(format!(
            "id_start_step ({start_step}) must be less than steps ({})",
            req.steps
        ));
    }

    validate_id_image_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A genuine 1x1 RGBA PNG.
    fn png_1x1() -> Vec<u8> {
        vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR length + type
            0x00, 0x00, 0x00, 0x01, // width  = 1
            0x00, 0x00, 0x00, 0x01, // height = 1
            0x08, 0x06, 0x00, 0x00, 0x00, // bit depth, colour type, ...
            0x1F, 0x15, 0xC4, 0x89, // IHDR CRC
            0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, // IDAT
            0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4,
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82, // IEND
        ]
    }

    /// A baseline JPEG carrying JFIF + SOF0 for a 1x1 image.
    fn jpeg_1x1() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, // APP0/JFIF
            0xFF, 0xC0, 0x00, 0x11, 0x08, // SOF0, length 17, precision 8
            0x00, 0x01, // height = 1
            0x00, 0x01, // width  = 1
            0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, // 3 components
            0xFF, 0xD9, // EOI
        ]
    }

    fn png_with_dimensions(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = png_1x1();
        bytes[16..20].copy_from_slice(&width.to_be_bytes());
        bytes[20..24].copy_from_slice(&height.to_be_bytes());
        bytes
    }

    fn jpeg_with_dimensions(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = jpeg_1x1();
        // SOF0 payload: height then width, both u16 big-endian.
        bytes[25..27].copy_from_slice(&(height as u16).to_be_bytes());
        bytes[27..29].copy_from_slice(&(width as u16).to_be_bytes());
        bytes
    }

    fn identity_request(model: &str) -> GenerateRequest {
        let mut req = crate::test_support::minimal_generate_request(model);
        req.steps = 20;
        req.id_image = Some(png_1x1());
        req
    }

    #[test]
    fn qualified_models_accept_every_resolved_spelling() {
        for name in ["flux-dev", "flux-dev:q4", "flux-dev:q8", "flux-dev-q4"] {
            assert!(
                identity_qualified_model(name),
                "{name} must be identity-qualified"
            );
        }
    }

    #[test]
    fn unqualified_models_are_rejected() {
        for name in [
            "flux-dev:bf16",
            "flux-dev:fp8",
            "flux-schnell",
            "flux-schnell:q8",
            "flux2-klein",
            "sdxl",
            "sdxl-base",
            "qwen-image",
            "",
        ] {
            assert!(
                !identity_qualified_model(name),
                "{name} must not be identity-qualified"
            );
        }
    }

    #[test]
    fn id_image_limits_are_internally_consistent() {
        assert_eq!(
            ID_IMAGE_LIMITS.max_decode_allocation_bytes,
            ID_IMAGE_LIMITS.max_decoded_pixels * 4
        );
        assert!(
            u64::from(ID_IMAGE_LIMITS.max_axis_pixels) * u64::from(ID_IMAGE_LIMITS.max_axis_pixels)
                > ID_IMAGE_LIMITS.max_decoded_pixels,
            "the per-axis limit must not be the only bound that can bite"
        );
    }

    #[test]
    fn real_png_and_jpeg_pass() {
        validate_id_image_bytes(&png_1x1()).expect("1x1 png must validate");
        validate_id_image_bytes(&jpeg_1x1()).expect("1x1 jpeg must validate");
    }

    #[test]
    fn empty_payload_is_rejected() {
        let error = validate_id_image_bytes(&[]).unwrap_err();
        assert!(error.contains("empty"), "{error}");
    }

    #[test]
    fn encoded_length_boundary_and_boundary_plus_one() {
        // At the limit the payload is still judged on its header.
        let mut at_limit = png_1x1();
        at_limit.resize(ID_IMAGE_LIMITS.max_encoded_bytes, 0);
        validate_id_image_bytes(&at_limit).expect("exactly at the encoded limit must pass");

        let mut over_limit = png_1x1();
        over_limit.resize(ID_IMAGE_LIMITS.max_encoded_bytes + 1, 0);
        let error = validate_id_image_bytes(&over_limit).unwrap_err();
        assert!(
            error.contains("16 MiB") || error.contains("bytes"),
            "{error}"
        );
    }

    #[test]
    fn encoded_length_is_checked_before_the_magic_bytes() {
        // A giant non-image must be refused for its size, never sniffed.
        let bomb = vec![0u8; ID_IMAGE_LIMITS.max_encoded_bytes + 1];
        let error = validate_id_image_bytes(&bomb).unwrap_err();
        assert!(
            !error.contains("PNG or JPEG"),
            "size must be refused before the format sniff: {error}"
        );
    }

    #[test]
    fn non_png_jpeg_magic_bytes_are_rejected() {
        let gif = b"GIF89a\x01\x00\x01\x00".to_vec();
        let webp = {
            let mut bytes = b"RIFF".to_vec();
            bytes.extend_from_slice(&[0x1A, 0x00, 0x00, 0x00]);
            bytes.extend_from_slice(b"WEBPVP8 ");
            bytes
        };
        let garbage = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        // A truncated PNG signature must not squeak through on a 4-byte sniff.
        let near_png = vec![0x89, 0x50, 0x4E, 0x47, 0x00, 0x00, 0x00, 0x00];
        for candidate in [gif, webp, garbage, near_png] {
            let error = validate_id_image_bytes(&candidate).unwrap_err();
            assert!(
                error.contains("PNG") && error.contains("JPEG"),
                "the refusal must name the accepted formats: {error}"
            );
        }
    }

    #[test]
    fn per_axis_boundary_and_boundary_plus_one() {
        let max = ID_IMAGE_LIMITS.max_axis_pixels;
        validate_id_image_bytes(&png_with_dimensions(max, 1)).expect("exactly at the axis limit");
        validate_id_image_bytes(&png_with_dimensions(1, max)).expect("exactly at the axis limit");

        let error = validate_id_image_bytes(&png_with_dimensions(max + 1, 1)).unwrap_err();
        assert!(error.contains(&max.to_string()), "{error}");
        let error = validate_id_image_bytes(&png_with_dimensions(1, max + 1)).unwrap_err();
        assert!(error.contains(&max.to_string()), "{error}");
    }

    #[test]
    fn total_pixel_boundary_and_boundary_plus_one() {
        // 8000 x 4000 = 32 MP exactly, both axes inside the per-axis limit.
        validate_id_image_bytes(&png_with_dimensions(8000, 4000))
            .expect("exactly at the pixel limit");
        let error = validate_id_image_bytes(&png_with_dimensions(8000, 4001)).unwrap_err();
        assert!(
            error.contains("pixel") || error.contains("megapixel"),
            "{error}"
        );
    }

    #[test]
    fn zero_dimension_is_rejected() {
        let error = validate_id_image_bytes(&png_with_dimensions(0, 16)).unwrap_err();
        assert!(
            error.contains("dimension") || error.contains("zero"),
            "{error}"
        );
    }

    #[test]
    fn jpeg_header_dimensions_are_read_too() {
        validate_id_image_bytes(&jpeg_with_dimensions(4096, 4096)).expect("inside every limit");
        let error = validate_id_image_bytes(&jpeg_with_dimensions(8192, 8000)).unwrap_err();
        assert!(
            error.contains("pixel") || error.contains("megapixel"),
            "{error}"
        );
    }

    #[test]
    fn truncated_headers_are_rejected_rather_than_assumed_small() {
        let mut truncated_png = png_1x1();
        truncated_png.truncate(20);
        assert!(validate_id_image_bytes(&truncated_png).is_err());

        let truncated_jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert!(validate_id_image_bytes(&truncated_jpeg).is_err());
    }

    #[test]
    fn effective_values_apply_the_documented_defaults() {
        let mut req = identity_request("flux-dev:q8");
        assert_eq!(effective_id_weight(&req), ID_WEIGHT_DEFAULT);
        assert_eq!(effective_id_start_step(&req), ID_START_STEP_DEFAULT);
        req.id_weight = Some(0.25);
        req.id_start_step = Some(3);
        assert_eq!(effective_id_weight(&req), 0.25);
        assert_eq!(effective_id_start_step(&req), 3);
    }

    /// The rules below only apply to a build that can execute identity
    /// conditioning; without the adapter the request never reaches them.
    #[cfg(feature = "pulid")]
    #[test]
    fn identity_conditioning_accepts_a_qualified_request() {
        for model in ["flux-dev", "flux-dev:q4", "flux-dev:q8", "flux-dev-q4"] {
            let mut req = identity_request(model);
            req.id_weight = Some(ID_WEIGHT_MAX);
            req.id_start_step = Some(req.steps - 1);
            req.id_image_name = Some("face.png".to_string());
            validate_identity_conditioning(&req)
                .unwrap_or_else(|error| panic!("{model} must be accepted: {error}"));
        }
    }

    /// True on every build: a request that never mentions identity is not
    /// this contract's business, whether or not the adapter is linked.
    #[test]
    fn a_request_without_identity_is_untouched_on_every_build() {
        for model in ["sdxl", "flux-dev:q8", "ltx-2-19b-distilled:fp8"] {
            let mut req = identity_request(model);
            req.id_image = None;
            validate_identity_conditioning(&req)
                .unwrap_or_else(|error| panic!("{model}: nothing to validate, got {error}"));
        }
    }

    /// A build without the adapter refuses rather than rendering a print
    /// that silently has no face in it — and says which support is missing,
    /// not that the model is wrong.
    #[cfg(not(feature = "pulid"))]
    #[test]
    fn identity_is_refused_when_the_build_lacks_the_adapter() {
        // Even the otherwise perfectly qualified request is refused.
        let req = identity_request("flux-dev:q8");
        let error = validate_identity_conditioning(&req).unwrap_err();
        assert_eq!(error, IDENTITY_BUILD_UNSUPPORTED);
        assert!(error.contains("built without PuLID"), "{error}");
        assert!(
            !error.contains("flux-dev:q4"),
            "the refusal is about the build, not the model: {error}"
        );

        // Every incomplete form is refused for the same reason, so a client
        // never has to guess which field was the problem on this server.
        for mutate in [
            (|req: &mut GenerateRequest| req.id_image = None) as fn(&mut GenerateRequest),
            |req: &mut GenerateRequest| {
                req.id_image = None;
                req.id_weight = Some(1.0);
            },
            |req: &mut GenerateRequest| {
                req.id_image = None;
                req.id_start_step = Some(1);
            },
            |req: &mut GenerateRequest| {
                req.id_image = None;
                req.id_image_name = Some("face.png".to_string());
            },
            |req: &mut GenerateRequest| req.model = "sdxl".to_string(),
        ] {
            let mut req = identity_request("flux-dev:q8");
            mutate(&mut req);
            match validate_identity_conditioning(&req) {
                // The first case clears every identity field, so it is the
                // ordinary no-identity request and must still pass.
                Ok(()) => assert!(!request_mentions_identity(&req)),
                Err(error) => assert_eq!(error, IDENTITY_BUILD_UNSUPPORTED),
            }
        }
    }

    /// The rules below only apply to a build that can execute identity
    /// conditioning; without the adapter the request never reaches them.
    #[cfg(feature = "pulid")]
    #[test]
    fn identity_rules_are_table_driven() {
        struct Case {
            name: &'static str,
            mutate: fn(&mut GenerateRequest),
            expect: &'static str,
        }

        let cases = [
            Case {
                name: "unqualified quantization",
                mutate: |req| req.model = "flux-dev:bf16".to_string(),
                expect: "flux-dev:q8",
            },
            Case {
                name: "unqualified flux checkpoint",
                mutate: |req| req.model = "flux-schnell".to_string(),
                expect: "flux-dev:q8",
            },
            Case {
                name: "unqualified family flux2",
                mutate: |req| req.model = "flux2-klein".to_string(),
                expect: "flux-dev:q8",
            },
            Case {
                name: "unqualified family sdxl",
                mutate: |req| req.model = "sdxl".to_string(),
                expect: "flux-dev:q4",
            },
            Case {
                name: "weight above the range",
                mutate: |req| req.id_weight = Some(ID_WEIGHT_MAX + 0.01),
                expect: "id_weight",
            },
            Case {
                name: "negative weight",
                mutate: |req| req.id_weight = Some(-0.001),
                expect: "id_weight",
            },
            Case {
                name: "non-finite weight",
                mutate: |req| req.id_weight = Some(f64::NAN),
                expect: "id_weight",
            },
            Case {
                name: "start step at steps",
                mutate: |req| req.id_start_step = Some(req.steps),
                expect: "id_start_step",
            },
            Case {
                name: "start step past steps",
                mutate: |req| req.id_start_step = Some(req.steps + 5),
                expect: "id_start_step",
            },
            Case {
                name: "lora combination",
                mutate: |req| {
                    req.loras = Some(vec![crate::types::LoraWeight {
                        path: "/loras/a.safetensors".to_string(),
                        scale: 1.0,
                        expert: None,
                    }])
                },
                expect: "LoRA",
            },
            Case {
                name: "legacy single lora combination",
                mutate: |req| {
                    req.lora = Some(crate::types::LoraWeight {
                        path: "/loras/a.safetensors".to_string(),
                        scale: 1.0,
                        expert: None,
                    })
                },
                expect: "LoRA",
            },
            Case {
                name: "img2img combination",
                mutate: |req| req.source_image = Some(vec![0x89, 0x50, 0x4E, 0x47]),
                expect: "source_image",
            },
            Case {
                name: "weight without image",
                mutate: |req| {
                    req.id_image = None;
                    req.id_weight = Some(1.0);
                },
                expect: "id_image is required",
            },
            Case {
                name: "start step without image",
                mutate: |req| {
                    req.id_image = None;
                    req.id_start_step = Some(1);
                },
                expect: "id_image is required",
            },
            Case {
                name: "name without image",
                mutate: |req| {
                    req.id_image = None;
                    req.id_image_name = Some("face.png".to_string());
                },
                expect: "id_image is required",
            },
            Case {
                name: "unusable image bytes",
                mutate: |req| req.id_image = Some(b"GIF89a".to_vec()),
                expect: "PNG",
            },
        ];

        for case in cases {
            let mut req = identity_request("flux-dev:q8");
            (case.mutate)(&mut req);
            let error = validate_identity_conditioning(&req)
                .expect_err(&format!("{} must be refused", case.name));
            assert!(
                error.contains(case.expect),
                "{}: expected {:?} in {error:?}",
                case.name,
                case.expect
            );
        }
    }

    /// The rules below only apply to a build that can execute identity
    /// conditioning; without the adapter the request never reaches them.
    #[cfg(feature = "pulid")]
    #[test]
    fn empty_lora_list_is_not_a_lora_combination() {
        let mut req = identity_request("flux-dev:q8");
        req.loras = Some(Vec::new());
        validate_identity_conditioning(&req).expect("an empty list is not a merged adapter");
    }

    /// The rules below only apply to a build that can execute identity
    /// conditioning; without the adapter the request never reaches them.
    #[cfg(feature = "pulid")]
    #[test]
    fn the_model_refusal_names_every_supported_model() {
        let mut req = identity_request("sdxl");
        let error = validate_identity_conditioning(&req).unwrap_err();
        for model in IDENTITY_QUALIFIED_MODELS {
            assert!(error.contains(model), "{model} must be named: {error}");
        }
        req.model = "flux-dev:q4".to_string();
        assert!(validate_identity_conditioning(&req).is_ok());
    }
}
