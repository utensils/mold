//! Face-identity conditioning contract (PuLID-FLUX).
//!
//! This module is the single authority for every identity-conditioning
//! constant on the wire: which models are qualified to accept an identity
//! image, the `id_weight` range, the `id_start_step` default, and the bounded
//! decode limits an `id_image` payload must respect before anything in the
//! process attempts a full decode.
//!
//! A build without the `pulid` feature links this contract and nothing that
//! consumes it, so it refuses an identity request rather than rendering it
//! without the face. [`IDENTITY_RUNTIME_READY`] is the second half of that
//! gate: it records whether the runtime stack behind the feature is complete.
//!
//! It also owns [`FrozenIdentityEmbedding`], the extraction admission freezes
//! once per parent request and every sibling reuses.

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

/// Whether this binary can actually execute identity conditioning.
///
/// The `pulid` feature compiles the wire contract — the qualified-model list,
/// the bounds, the validator, the request fields.
///
/// #1221 landed the cross-attention adapter and #1223 wired the extractor to
/// it: a `pulid` build now turns `id_image` into the `[1, 32, 2048]` embedding
/// at admission, freezes it into the prepared plan, and installs it on the
/// engine before every dispatch. So a `pulid` build can execute identity
/// conditioning end to end, and this is `true`.
///
/// It stays a separate constant from the feature rather than collapsing into
/// it, because the two answer different questions: the feature is whether this
/// binary LINKED the stack, and this is whether the stack is complete enough to
/// honour a request. Advertising a control the worker cannot honour is the
/// failure `CLAUDE.md` records for wan's `supports_sequence`; here it would be
/// worse than a hard error, because the print would render, look fine, and
/// simply not be the person in the reference photo.
pub const IDENTITY_RUNTIME_READY: bool = true;

/// Whether identity conditioning can be both accepted and executed here.
///
/// The single predicate for every advertisement and every admission decision:
/// the feature must be linked *and* the runtime adapter must have landed.
/// Callers must never re-spell it as a bare `cfg!(feature = "pulid")`.
pub fn identity_runtime_available() -> bool {
    cfg!(feature = "pulid") && IDENTITY_RUNTIME_READY
}

/// Refusal a build compiled without the `pulid` feature gives a request that
/// asks for identity conditioning.
///
/// Accept-and-ignore is not an option: the print would render without the
/// face and nothing would say so. The request is refused instead, and the
/// message names the missing build support rather than blaming the model.
pub const IDENTITY_BUILD_UNSUPPORTED: &str =
    "this server was built without PuLID face-identity support; remove id_image \
     (and any id_weight, id_start_step, or id_image_name) or use a server built \
     with the `pulid` feature";

/// Refusal a build that links `pulid` but predates the runtime adapter gives
/// a request that asks for identity conditioning.
///
/// Deliberately distinct from [`IDENTITY_BUILD_UNSUPPORTED`]: the operator of
/// this server does not need a differently compiled binary, they need a newer
/// one, and a client that cannot tell the two apart would send them to the
/// wrong fix.
pub const IDENTITY_RUNTIME_PENDING: &str =
    "identity conditioning is not available in this build yet — the runtime \
     adapter has not landed; remove id_image (and any id_weight, id_start_step, \
     or id_image_name), or upgrade to a server that advertises \
     supports_identity";

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
    // rather than rendering a print that silently has no face in it. The two
    // reasons stay distinct: one needs a differently compiled binary, the
    // other simply a newer one.
    if !cfg!(feature = "pulid") {
        return Err(IDENTITY_BUILD_UNSUPPORTED.to_string());
    }
    if !IDENTITY_RUNTIME_READY {
        return Err(IDENTITY_RUNTIME_PENDING.to_string());
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

/// Token count of the identity embedding the IDFormer produces.
///
/// Mirrors `mold_inference::flux::pulid::ID_TOKENS`; it lives here as well
/// because the frozen embedding crosses the crate boundary as plain data and
/// the shape has to be checkable without linking the engine.
pub const ID_EMBEDDING_TOKENS: usize = 32;

/// Width of each identity token.
pub const ID_EMBEDDING_DIM: usize = 2048;

/// Number of `f32` values one frozen embedding carries.
pub const ID_EMBEDDING_VALUES: usize = ID_EMBEDDING_TOKENS * ID_EMBEDDING_DIM;

/// SHA-256 pins of every artifact one identity extraction consumed.
///
/// Recorded rather than assumed: the four PuLID files are pinned by the
/// manifest, but the *derived* vision tower is mold's own conversion output,
/// and an embedding is only reproducible from the exact bytes that produced
/// it. A frozen plan carrying these can be checked against the bundle the
/// worker actually holds instead of trusting that nothing was repaired
/// underneath it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdentityAssetDigests {
    /// `pulid_flux_v0.9.1.safetensors` — the IDFormer half.
    pub adapter: String,
    /// The derived `eva02_clip_l_336_vision.safetensors`, not the `.pt`
    /// source: the tower that ran is the artifact that matters.
    pub vision: String,
    /// `scrfd_10g_bnkps.onnx`.
    pub face_detector: String,
    /// `glintr100.onnx`.
    pub face_recognizer: String,
}

/// One identity, extracted once and frozen for every sibling of a batch.
///
/// This is the unit that makes "exactly one extraction per parent request"
/// enforceable rather than aspirational. Admission produces it before batch
/// fan-out; every singleton child, on every device, carries the same value and
/// the engine turns it back into a tensor on its own device. It is deliberately
/// **plain data** — little-endian `f32` bytes, not a `candle` tensor — so it is
/// `Clone`, `Eq`, device-independent, and safe to hold in a plan that outlives
/// any particular engine.
///
/// 32 x 2048 f32 is 256 KiB, which is why carrying it per child costs nothing
/// worth optimizing and why nothing here is lazily recomputed.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct FrozenIdentityEmbedding {
    /// `ID_EMBEDDING_VALUES` little-endian `f32`s, token-major.
    values_le: std::sync::Arc<Vec<u8>>,
    /// SHA-256 of the encoded `id_image` bytes the extraction consumed.
    source_sha256: String,
    assets: IdentityAssetDigests,
    /// SHA-256 over source, assets, and values — the identity half of the
    /// prepared plan's profile hash.
    fingerprint: String,
}

impl FrozenIdentityEmbedding {
    /// Freeze an extracted embedding.
    ///
    /// The shape is checked here, once, rather than at the twenty injection
    /// sites that eventually consume it.
    pub fn new(
        values: &[f32],
        source_sha256: impl Into<String>,
        assets: IdentityAssetDigests,
    ) -> Result<Self, String> {
        if values.len() != ID_EMBEDDING_VALUES {
            return Err(format!(
                "a frozen identity embedding must carry {ID_EMBEDDING_VALUES} values \
                 ({ID_EMBEDDING_TOKENS}x{ID_EMBEDDING_DIM}), got {}",
                values.len()
            ));
        }
        if let Some(bad) = values.iter().find(|value| !value.is_finite()) {
            return Err(format!(
                "the identity extractor produced a non-finite value ({bad}); \
                 the embedding is not usable"
            ));
        }
        let mut values_le = Vec::with_capacity(ID_EMBEDDING_VALUES * 4);
        for value in values {
            values_le.extend_from_slice(&value.to_le_bytes());
        }
        let source_sha256 = source_sha256.into();
        let fingerprint = fingerprint_of(&values_le, &source_sha256, &assets);
        Ok(Self {
            values_le: std::sync::Arc::new(values_le),
            source_sha256,
            assets,
            fingerprint,
        })
    }

    /// The embedding, token-major, as `ID_EMBEDDING_VALUES` `f32`s.
    pub fn values(&self) -> Vec<f32> {
        self.values_le
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// SHA-256 of the `id_image` this identity came from.
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    /// The artifacts the extraction ran on.
    pub fn assets(&self) -> &IdentityAssetDigests {
        &self.assets
    }

    /// The identity profile hash a prepared plan freezes.
    ///
    /// Two children of the same parent request must agree on it exactly; a
    /// disagreement means something re-extracted, which is the bug this whole
    /// type exists to make impossible.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

/// Redacted: the values are a biometric derivative and must never reach a log
/// line, an error body, or a probe payload.
impl std::fmt::Debug for FrozenIdentityEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenIdentityEmbedding")
            .field("fingerprint", &self.fingerprint)
            .field("source_sha256", &self.source_sha256)
            .field("values", &"<redacted>")
            .finish()
    }
}

fn fingerprint_of(values_le: &[u8], source_sha256: &str, assets: &IdentityAssetDigests) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"mold.identity.v1\0");
    hasher.update(source_sha256.as_bytes());
    hasher.update(b"\0");
    for digest in [
        &assets.adapter,
        &assets.vision,
        &assets.face_detector,
        &assets.face_recognizer,
    ] {
        hasher.update(digest.as_bytes());
        hasher.update(b"\0");
    }
    hasher.update(values_le);
    format!("{:x}", hasher.finalize())
}

/// SHA-256 of an `id_image` payload — the provenance a frozen embedding
/// records so a replayed or restored request can be told apart from a new one.
pub fn id_image_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
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

    /// Every `pulid` test below is written for BOTH values of
    /// [`IDENTITY_RUNTIME_READY`]: while the adapter is pending, the shape
    /// rules are unreachable and the one observable behaviour is this
    /// refusal, so asserting it is what keeps these tests honest rather than
    /// merely compiled.
    #[cfg(feature = "pulid")]
    fn assert_runtime_pending(result: Result<(), String>, context: &str) {
        let error = result.expect_err(&format!(
            "{context}: a pending adapter must refuse the request"
        ));
        assert_eq!(error, IDENTITY_RUNTIME_PENDING, "{context}");
        assert!(
            !error.contains("built without PuLID"),
            "{context}: a linked-but-pending build must not read as the wrong binary: {error}"
        );
    }

    /// The rules below only apply to a build that can execute identity
    /// conditioning; while the adapter is pending the request is refused
    /// before it reaches any of them.
    #[cfg(feature = "pulid")]
    #[test]
    fn identity_conditioning_accepts_a_qualified_request() {
        for model in ["flux-dev", "flux-dev:q4", "flux-dev:q8", "flux-dev-q4"] {
            let mut req = identity_request(model);
            req.id_weight = Some(ID_WEIGHT_MAX);
            req.id_start_step = Some(req.steps - 1);
            req.id_image_name = Some("face.png".to_string());
            let result = validate_identity_conditioning(&req);
            if IDENTITY_RUNTIME_READY {
                result.unwrap_or_else(|error| panic!("{model} must be accepted: {error}"));
            } else {
                assert_runtime_pending(result, model);
            }
        }
    }

    /// The feature alone never makes the capability real: a build that links
    /// `pulid` while the adapter is pending refuses exactly like one that
    /// does not link it, but with its own reason.
    #[cfg(feature = "pulid")]
    #[test]
    fn identity_is_refused_while_the_runtime_adapter_is_pending() {
        if IDENTITY_RUNTIME_READY {
            return;
        }
        let req = identity_request("flux-dev:q8");
        let error = validate_identity_conditioning(&req).unwrap_err();
        assert_eq!(error, IDENTITY_RUNTIME_PENDING);
        assert!(
            !error.contains("flux-dev:q4"),
            "the refusal is about the build, not the model: {error}"
        );
        // A request that mentions no identity field is still not this
        // contract's business, even while the adapter is pending.
        let mut plain = identity_request("flux-dev:q8");
        plain.id_image = None;
        validate_identity_conditioning(&plain).expect("no identity mentioned, nothing to refuse");
    }

    /// The two refusals must stay distinguishable — a client routes the
    /// operator to a different fix for each.
    #[test]
    fn the_two_unavailable_refusals_are_distinct() {
        assert_ne!(IDENTITY_BUILD_UNSUPPORTED, IDENTITY_RUNTIME_PENDING);
        assert!(IDENTITY_BUILD_UNSUPPORTED.contains("built without PuLID"));
        assert!(IDENTITY_RUNTIME_PENDING.contains("not available in this build yet"));
    }

    /// The predicate is the conjunction, never either half alone.
    #[test]
    fn runtime_availability_requires_both_the_feature_and_the_adapter() {
        assert_eq!(
            identity_runtime_available(),
            cfg!(feature = "pulid") && IDENTITY_RUNTIME_READY
        );
        if !IDENTITY_RUNTIME_READY {
            assert!(
                !identity_runtime_available(),
                "no build may claim identity while the adapter is pending"
            );
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
    /// conditioning; while the adapter is pending every case collapses to
    /// the one pending refusal, which is asserted instead.
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
            let result = validate_identity_conditioning(&req);
            if !IDENTITY_RUNTIME_READY {
                // Every one of these requests still mentions identity, so the
                // pending build refuses them all for the same reason before a
                // single shape rule runs.
                assert!(request_mentions_identity(&req), "{}", case.name);
                assert_runtime_pending(result, case.name);
                continue;
            }
            let error = result.expect_err(&format!("{} must be refused", case.name));
            assert!(
                error.contains(case.expect),
                "{}: expected {:?} in {error:?}",
                case.name,
                case.expect
            );
        }
    }

    /// The rules below only apply to a build that can execute identity
    /// conditioning; while the adapter is pending the request is refused
    /// before it reaches any of them.
    #[cfg(feature = "pulid")]
    #[test]
    fn empty_lora_list_is_not_a_lora_combination() {
        let mut req = identity_request("flux-dev:q8");
        req.loras = Some(Vec::new());
        let result = validate_identity_conditioning(&req);
        if IDENTITY_RUNTIME_READY {
            result.expect("an empty list is not a merged adapter");
        } else {
            assert_runtime_pending(result, "empty lora list");
        }
    }

    /// The rules below only apply to a build that can execute identity
    /// conditioning; while the adapter is pending the request is refused
    /// before it reaches any of them.
    #[cfg(feature = "pulid")]
    #[test]
    fn the_model_refusal_names_every_supported_model() {
        let mut req = identity_request("sdxl");
        if !IDENTITY_RUNTIME_READY {
            assert_runtime_pending(validate_identity_conditioning(&req), "unqualified model");
            req.model = "flux-dev:q4".to_string();
            assert_runtime_pending(validate_identity_conditioning(&req), "qualified model");
            return;
        }
        let error = validate_identity_conditioning(&req).unwrap_err();
        for model in IDENTITY_QUALIFIED_MODELS {
            assert!(error.contains(model), "{model} must be named: {error}");
        }
        req.model = "flux-dev:q4".to_string();
        assert!(validate_identity_conditioning(&req).is_ok());
    }

    fn digests() -> IdentityAssetDigests {
        IdentityAssetDigests {
            adapter: "a".repeat(64),
            vision: "b".repeat(64),
            face_detector: "c".repeat(64),
            face_recognizer: "d".repeat(64),
        }
    }

    fn embedding_values(seed: f32) -> Vec<f32> {
        (0..ID_EMBEDDING_VALUES)
            .map(|index| seed + index as f32 * 1e-4)
            .collect()
    }

    #[test]
    fn a_frozen_embedding_round_trips_its_values_exactly() {
        let values = embedding_values(0.5);
        let frozen = FrozenIdentityEmbedding::new(&values, "source-sha", digests())
            .expect("a correctly shaped embedding freezes");
        assert_eq!(frozen.values(), values);
        assert_eq!(frozen.source_sha256(), "source-sha");
        assert_eq!(frozen.assets(), &digests());
    }

    #[test]
    fn the_shape_is_checked_once_at_the_boundary() {
        let error = FrozenIdentityEmbedding::new(&[0.0; 16], "s", digests()).unwrap_err();
        assert!(error.contains(&ID_EMBEDDING_VALUES.to_string()), "{error}");
        assert!(error.contains("32x2048"), "{error}");
    }

    #[test]
    fn a_non_finite_extraction_is_refused_rather_than_rendered() {
        let mut values = embedding_values(0.5);
        values[7] = f32::NAN;
        let error = FrozenIdentityEmbedding::new(&values, "s", digests()).unwrap_err();
        assert!(error.contains("non-finite"), "{error}");
    }

    /// The whole point of the type: two children of one parent request agree
    /// bit for bit, so "exactly one extraction" is checkable.
    #[test]
    fn the_fingerprint_is_stable_across_clones_and_rebuilds() {
        let values = embedding_values(0.25);
        let first = FrozenIdentityEmbedding::new(&values, "s", digests()).unwrap();
        let second = FrozenIdentityEmbedding::new(&values, "s", digests()).unwrap();
        assert_eq!(first.fingerprint(), second.fingerprint());
        assert_eq!(first.clone().fingerprint(), first.fingerprint());
        assert_eq!(first, second);
    }

    #[test]
    fn the_fingerprint_moves_with_the_values_the_source_and_the_assets() {
        let base = FrozenIdentityEmbedding::new(&embedding_values(0.25), "s", digests()).unwrap();

        let other_values =
            FrozenIdentityEmbedding::new(&embedding_values(0.26), "s", digests()).unwrap();
        assert_ne!(base.fingerprint(), other_values.fingerprint());

        let other_source =
            FrozenIdentityEmbedding::new(&embedding_values(0.25), "t", digests()).unwrap();
        assert_ne!(base.fingerprint(), other_source.fingerprint());

        let mut repaired = digests();
        repaired.vision = "e".repeat(64);
        let other_assets =
            FrozenIdentityEmbedding::new(&embedding_values(0.25), "s", repaired).unwrap();
        assert_ne!(base.fingerprint(), other_assets.fingerprint());
    }

    /// A biometric derivative must never reach a log line or an error body.
    #[test]
    fn the_debug_form_redacts_the_biometric_values() {
        let frozen =
            FrozenIdentityEmbedding::new(&embedding_values(0.25), "source-sha", digests()).unwrap();
        let rendered = format!("{frozen:?}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
        assert!(rendered.contains(frozen.fingerprint()), "{rendered}");
        assert!(!rendered.contains("0.25"), "{rendered}");
    }

    #[test]
    fn the_source_digest_is_a_plain_sha256_of_the_payload() {
        // Well-known: sha256("abc").
        assert_eq!(
            id_image_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
