use anyhow::{bail, Context, Result};
use candle_core::Tensor;
use mold_core::{GenerateRequest, OutputFormat, OutputMetadata, Scheduler};

const MOLD_VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn build_output_metadata(
    req: &GenerateRequest,
    seed: u64,
    scheduler: Option<Scheduler>,
) -> Option<OutputMetadata> {
    if !req.embed_metadata.unwrap_or(true) {
        return None;
    }

    Some(OutputMetadata::from_generate_request(
        req,
        seed,
        scheduler,
        MOLD_VERSION,
    ))
}

pub(crate) fn update_output_metadata_size(
    metadata: &mut Option<OutputMetadata>,
    width: u32,
    height: u32,
) {
    if let Some(metadata) = metadata {
        metadata.width = width;
        metadata.height = height;
    }
}

/// What happens to the alpha channel of an RGBA render.
///
/// The ENGINE decides, because the answer is a property of the request, not
/// of the pixels: Qwen Image 2.1's decoder emits edge alpha of 204-252 on
/// ordinary opaque renders (measured in the M1 capture), so "any alpha below
/// 255" would turn every text-to-image render into an RGBA file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaOutput {
    /// No decision: an all-255 alpha channel encodes as RGB, anything else
    /// keeps it. The fallback for callers that do not decide.
    Infer,
    /// Keep the alpha channel: PNG is written RGBA and WebP with lossless
    /// alpha; JPEG, which cannot carry it, is composited over white.
    Keep,
    /// Discard the alpha channel: the RGB planes are encoded unchanged, so the
    /// file is byte-identical to the same pixels rendered as RGB.
    Drop,
}

/// The Qwen Image 2.1 output rule: keep alpha iff the request asks for a
/// transparent background, or at least one reference image carries a pixel
/// below full opacity (`capabilities.transparency.native_alpha` — an
/// alpha-carrying reference keeps alpha in the output). Otherwise drop it.
///
/// A reference that cannot be decoded is an error, never "no alpha": the
/// engine could not have conditioned on it either.
pub fn alpha_output_for_request(req: &mold_core::GenerateRequest) -> Result<AlphaOutput> {
    let mut references_carry_alpha = false;
    for bytes in req.edit_images.as_deref().unwrap_or_default() {
        if encoded_image_has_alpha(bytes)? {
            references_carry_alpha = true;
            break;
        }
    }
    Ok(
        if req.transparent_background == Some(true) || references_carry_alpha {
            AlphaOutput::Keep
        } else {
            AlphaOutput::Drop
        },
    )
}

/// Whether an encoded reference (PNG, JPEG, WebP) has any pixel below full
/// opacity. The container header is asked first, so an RGB file is never
/// decoded; an alpha-capable one is decoded by the SAME bounded decoder the
/// engine conditions on (`img_utils::decode_reference_rgba`), with Pillow's
/// 16-bit conversion, and scanned — an RGBA container whose alpha is all 255
/// carries no transparency.
pub fn encoded_image_has_alpha(bytes: &[u8]) -> Result<bool> {
    if !mold_core::still_image::encoded_still_has_alpha(bytes) {
        return Ok(false);
    }
    let decoded = crate::img_utils::decode_reference_rgba(bytes)
        .context("failed to decode a reference image to read its alpha")?;
    Ok(rgba_has_alpha(&decoded))
}

/// Encode a candle tensor of u8 values into still image bytes.
///
/// `[3, H, W]` is an RGB render. `[4, H, W]` is an RGBA render (a decoder
/// that emits alpha, e.g. Qwen Image 2.1's four-channel VAE), encoded under
/// [`AlphaOutput::Infer`]; an engine that has decided calls
/// [`encode_image_with_alpha`].
pub(crate) fn encode_image(
    img: &Tensor,
    format: OutputFormat,
    width: u32,
    height: u32,
    metadata: Option<&OutputMetadata>,
) -> Result<Vec<u8>> {
    encode_image_with_alpha(img, format, width, height, metadata, AlphaOutput::Infer)
}

/// [`encode_image`] with the engine's alpha decision. The decision applies to
/// a four-channel tensor only; a three-channel render has no alpha to keep.
pub(crate) fn encode_image_with_alpha(
    img: &Tensor,
    format: OutputFormat,
    width: u32,
    height: u32,
    metadata: Option<&OutputMetadata>,
    alpha: AlphaOutput,
) -> Result<Vec<u8>> {
    let (c, h, w) = img.dims3()?;
    if c != 3 && c != 4 {
        bail!("expected 3 (RGB) or 4 (RGBA) channels, got {c}");
    }
    let _ = (h, w); // dims used implicitly via from_raw

    let img_data = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    if c == 4 {
        let rgba_image = image::RgbaImage::from_raw(width, height, img_data)
            .ok_or_else(|| anyhow::anyhow!("failed to create image from tensor data"))?;
        return encode_rgba_image(&rgba_image, format, metadata, alpha);
    }
    let rgb_image = image::RgbImage::from_raw(width, height, img_data)
        .ok_or_else(|| anyhow::anyhow!("failed to create image from tensor data"))?;

    encode_rgb_image(&rgb_image, format, metadata)
}

/// Whether any pixel of an RGBA image is less than fully opaque.
pub(crate) fn rgba_has_alpha(rgba_image: &image::RgbaImage) -> bool {
    rgba_image.pixels().any(|pixel| pixel[3] != u8::MAX)
}

/// Drop the alpha channel, keeping the RGB planes byte for byte.
fn opaque_rgba_to_rgb(rgba_image: &image::RgbaImage) -> image::RgbImage {
    image::RgbImage::from_fn(rgba_image.width(), rgba_image.height(), |x, y| {
        let [r, g, b, _] = rgba_image.get_pixel(x, y).0;
        image::Rgb([r, g, b])
    })
}

/// Encode an RGBA still under an [`AlphaOutput`] decision.
///
/// The decision is resolved BEFORE any container is written, because the
/// embedded provenance records it (`OutputMetadata::has_alpha`):
///
/// - `Drop`, or `Infer` with every alpha byte 255: the image is encoded
///   exactly as the RGB image it is — a PNG stays `ColorType::Rgb` and is
///   byte-identical to an RGB render's.
/// - `Keep`, or `Infer` with any alpha below 255: PNG is written RGBA and WebP
///   as a still with lossless alpha; JPEG, which has no alpha, is composited
///   over white.
pub(crate) fn encode_rgba_image(
    rgba_image: &image::RgbaImage,
    format: OutputFormat,
    metadata: Option<&OutputMetadata>,
    alpha: AlphaOutput,
) -> Result<Vec<u8>> {
    let keep = match alpha {
        AlphaOutput::Keep => true,
        AlphaOutput::Drop => false,
        AlphaOutput::Infer => rgba_has_alpha(rgba_image),
    };
    if !keep {
        return encode_rgb_image(&opaque_rgba_to_rgb(rgba_image), format, metadata);
    }
    match format {
        OutputFormat::Png => {
            let metadata = metadata.map(|metadata| metadata_with_alpha(metadata, true));
            let mut buf = std::io::Cursor::new(Vec::new());
            write_png_color(
                rgba_image.width(),
                rgba_image.height(),
                png::ColorType::Rgba,
                rgba_image.as_raw(),
                &mut buf,
                metadata.as_ref(),
                png_profile(),
            )?;
            Ok(buf.into_inner())
        }
        OutputFormat::Webp => crate::webp_still::encode_webp_still_rgba(
            rgba_image,
            crate::webp_still::WEBP_STILL_QUALITY,
        ),
        OutputFormat::Jpeg => {
            // A JPEG carries no alpha, so the flattened pixels are what the
            // file holds and its provenance must not claim alpha.
            let metadata = metadata.map(|metadata| metadata_with_alpha(metadata, false));
            encode_rgb_image(
                &crate::pillow_resize::composite_over_white(rgba_image),
                format,
                metadata.as_ref(),
            )
        }
        OutputFormat::Gif
        | OutputFormat::Apng
        | OutputFormat::Mp4
        | OutputFormat::Wav
        | OutputFormat::Glb
        | OutputFormat::Obj => {
            anyhow::bail!("{format} encoding is not supported for single images")
        }
    }
}

/// Provenance for a still, stamped with whether its stored pixels carry
/// alpha.
fn metadata_with_alpha(metadata: &OutputMetadata, has_alpha: bool) -> OutputMetadata {
    let mut metadata = metadata.clone();
    metadata.has_alpha = has_alpha.then_some(true);
    metadata
}

pub(crate) fn encode_rgb_image(
    rgb_image: &image::RgbImage,
    format: OutputFormat,
    metadata: Option<&OutputMetadata>,
) -> Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    match format {
        OutputFormat::Png => write_png(rgb_image, &mut buf, metadata)?,
        OutputFormat::Jpeg => write_jpeg(rgb_image, &mut buf, metadata)?,
        // A WebP STILL: libwebp's simple API, never the animation encoder, so
        // the file is a plain `VP8 ` bitstream with no `ANIM` chunk. WebP
        // carries no embedded provenance; the gallery row is the authority,
        // exactly as for a video clip.
        OutputFormat::Webp => {
            return crate::webp_still::encode_webp_still_rgb(
                rgb_image,
                crate::webp_still::WEBP_STILL_QUALITY,
            );
        }
        OutputFormat::Gif
        | OutputFormat::Apng
        | OutputFormat::Mp4
        | OutputFormat::Wav
        | OutputFormat::Glb
        | OutputFormat::Obj => {
            anyhow::bail!("{format} encoding is not supported for single images")
        }
    }

    Ok(buf.into_inner())
}

/// Embed generation metadata as PNG text chunks: per-field tEXt/iTXt for
/// quick tooling plus the composite `mold:parameters` JSON block the
/// gallery readers parse back into `OutputMetadata`. Shared by still-PNG
/// output and APNG video output so the two can't drift.
pub(crate) fn add_metadata_chunks<W: std::io::Write>(
    encoder: &mut png::Encoder<W>,
    metadata: &OutputMetadata,
) -> Result<()> {
    encoder.add_itxt_chunk("mold:prompt".to_string(), metadata.prompt.clone())?;
    encoder.add_itxt_chunk("mold:model".to_string(), metadata.model.clone())?;
    encoder.add_text_chunk("mold:seed".to_string(), metadata.seed.to_string())?;
    encoder.add_text_chunk("mold:steps".to_string(), metadata.steps.to_string())?;
    encoder.add_text_chunk("mold:guidance".to_string(), metadata.guidance.to_string())?;
    encoder.add_text_chunk("mold:width".to_string(), metadata.width.to_string())?;
    encoder.add_text_chunk("mold:height".to_string(), metadata.height.to_string())?;
    if let Some(frames) = metadata.frames {
        encoder.add_text_chunk("mold:frames".to_string(), frames.to_string())?;
    }
    if let Some(fps) = metadata.fps {
        encoder.add_text_chunk("mold:fps".to_string(), fps.to_string())?;
    }
    if let Some(strength) = metadata.strength {
        encoder.add_text_chunk("mold:strength".to_string(), strength.to_string())?;
    }
    if let Some(scheduler) = metadata.scheduler {
        encoder.add_text_chunk("mold:scheduler".to_string(), scheduler.to_string())?;
    }
    if let Some(ref neg) = metadata.negative_prompt {
        encoder.add_itxt_chunk("mold:negative_prompt".to_string(), neg.clone())?;
    }
    if let Some(ref original) = metadata.original_prompt {
        encoder.add_itxt_chunk("mold:original_prompt".to_string(), original.clone())?;
    }
    encoder.add_itxt_chunk("mold:version".to_string(), metadata.version.clone())?;
    encoder.add_itxt_chunk(
        "mold:parameters".to_string(),
        serde_json::to_string(metadata)?,
    )?;
    Ok(())
}

/// How much CPU a saved PNG is worth.
///
/// PNG is lossless under both, so this only trades encode time against file
/// size — never pixels. A 1024² still spent ~1.0 s of the measured 16.6 s
/// server timeline at zlib-6 with adaptive per-row filter selection, which is
/// five filter evaluations per row plus a full-strength deflate on data that
/// is mostly incompressible noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PngEncoding {
    /// `fdeflate`'s PNG-tuned ultra-fast deflate with a fixed `Sub` filter.
    /// The default.
    #[default]
    Fast,
    /// zlib level 6 with adaptive per-row filter selection — the historical
    /// behaviour, for anyone who would rather spend the second.
    Balanced,
}

/// Resolve `MOLD_PNG_ENCODING`. Pure, so the contract is testable without
/// touching the process environment.
///
/// An unrecognized value is the default rather than an error: a finished
/// render must never fail to save because the variable was mistyped.
pub fn png_profile_from(value: Option<&str>) -> PngEncoding {
    match value.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("balanced") => PngEncoding::Balanced,
        _ => PngEncoding::Fast,
    }
}

/// The profile this process encodes stills with.
///
/// Read with `std::env::var`, not `runtime_env::value`: the choice is lossless
/// either way, so it changes no pixel and must never join the engine
/// fingerprint (`ENGINE_SHAPING_VARIABLES`) or split an execution-equivalence
/// class. `latent_preview.rs` reads `MOLD_STEP_PREVIEW` the same way.
pub fn png_profile() -> PngEncoding {
    png_profile_from(std::env::var("MOLD_PNG_ENCODING").ok().as_deref())
}

fn write_png(
    rgb_image: &image::RgbImage,
    writer: &mut std::io::Cursor<Vec<u8>>,
    metadata: Option<&OutputMetadata>,
) -> Result<()> {
    write_png_with(rgb_image, writer, metadata, png_profile())
}

fn write_png_with<W: std::io::Write>(
    rgb_image: &image::RgbImage,
    writer: W,
    metadata: Option<&OutputMetadata>,
    profile: PngEncoding,
) -> Result<()> {
    write_png_color(
        rgb_image.width(),
        rgb_image.height(),
        png::ColorType::Rgb,
        rgb_image.as_raw(),
        writer,
        metadata,
        profile,
    )
}

fn write_png_color<W: std::io::Write>(
    width: u32,
    height: u32,
    color: png::ColorType,
    pixels: &[u8],
    writer: W,
    metadata: Option<&OutputMetadata>,
    profile: PngEncoding,
) -> Result<()> {
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    match profile {
        PngEncoding::Fast => {
            // `fdeflate`'s PNG-tuned ultra-fast deflate, keeping the adaptive
            // per-row filter `Compression::Fast` already selects. The plan for
            // this change asked for a fixed `Filter::Sub` on the theory that
            // adaptive selection is five trial encodes per row; measured, it
            // is a pessimization in BOTH directions at this deflate level —
            // 1024² noise 2,759,598 B / 7.3 ms with `Sub` against 2,741,265 B
            // / 7.4 ms adaptive, a 512² photograph 380,599 B / 1.5 ms against
            // 322,448 B / 1.7 ms — because the smaller filtered stream costs
            // the deflate pass back what the filter search spent.
            encoder.set_compression(png::Compression::Fast);
            encoder.set_filter(png::Filter::Adaptive);
        }
        PngEncoding::Balanced => {
            encoder.set_compression(png::Compression::Balanced);
            encoder.set_filter(png::Filter::Adaptive);
        }
    }

    if let Some(metadata) = metadata {
        add_metadata_chunks(&mut encoder, metadata)?;
    }

    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(pixels)?;
    png_writer.finish()?;
    Ok(())
}

fn write_jpeg(
    rgb_image: &image::RgbImage,
    writer: &mut std::io::Cursor<Vec<u8>>,
    metadata: Option<&OutputMetadata>,
) -> Result<()> {
    rgb_image.write_to(writer, image::ImageFormat::Jpeg)?;

    let Some(metadata) = metadata else {
        return Ok(());
    };

    let jpeg_bytes = writer.get_ref().clone();
    // JPEG must start with SOI (0xFFD8)
    if jpeg_bytes.len() < 2 || jpeg_bytes[0] != 0xFF || jpeg_bytes[1] != 0xD8 {
        return Ok(());
    }

    let mut out = Vec::with_capacity(jpeg_bytes.len() + 4096);
    out.extend_from_slice(&jpeg_bytes[..2]); // SOI marker

    // Inject COM marker with JSON parameters (read by exiftool, identify, ffprobe)
    let json = serde_json::to_string(metadata)?;
    let comment = format!("mold:parameters {json}");
    write_jpeg_com_marker(&mut out, comment.as_bytes());

    // Inject XMP APP1 marker (read by Photoshop, Lightroom, GIMP, exiftool -xmp:all)
    let xmp = build_xmp_packet(metadata)?;
    write_jpeg_xmp_marker(&mut out, &xmp);

    // Append rest of JPEG data (everything after SOI)
    out.extend_from_slice(&jpeg_bytes[2..]);

    writer.get_mut().clear();
    writer.get_mut().extend_from_slice(&out);
    writer.set_position(out.len() as u64);
    Ok(())
}

/// Write a JPEG COM (comment) marker segment.
/// Truncates payload to 65533 bytes (JPEG segment limit: 65535 - 2 for length field).
fn write_jpeg_com_marker(out: &mut Vec<u8>, data: &[u8]) {
    const MAX_PAYLOAD: usize = 65533;
    let data = if data.len() > MAX_PAYLOAD {
        tracing::warn!(
            "JPEG COM marker truncated from {} to {MAX_PAYLOAD} bytes",
            data.len()
        );
        &data[..MAX_PAYLOAD]
    } else {
        data
    };
    let len = (data.len() + 2) as u16;
    out.push(0xFF);
    out.push(0xFE); // COM marker
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(data);
}

/// Build an XMP packet containing generation metadata as RDF/XML.
fn build_xmp_packet(metadata: &OutputMetadata) -> Result<Vec<u8>> {
    use std::fmt::Write;
    let mut xmp = String::with_capacity(1024);
    xmp.push_str(r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>"#);
    xmp.push_str(r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">"#);
    xmp.push_str(r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#);
    xmp.push_str(r#"<rdf:Description rdf:about="" xmlns:mold="https://github.com/utensils/mold">"#);
    let _ = write!(
        xmp,
        "<mold:prompt>{}</mold:prompt>",
        xml_escape(&metadata.prompt)
    );
    let _ = write!(
        xmp,
        "<mold:model>{}</mold:model>",
        xml_escape(&metadata.model)
    );
    let _ = write!(xmp, "<mold:seed>{}</mold:seed>", metadata.seed);
    let _ = write!(xmp, "<mold:steps>{}</mold:steps>", metadata.steps);
    let _ = write!(xmp, "<mold:guidance>{}</mold:guidance>", metadata.guidance);
    let _ = write!(xmp, "<mold:width>{}</mold:width>", metadata.width);
    let _ = write!(xmp, "<mold:height>{}</mold:height>", metadata.height);
    if let Some(strength) = metadata.strength {
        let _ = write!(xmp, "<mold:strength>{strength}</mold:strength>");
    }
    if let Some(scheduler) = metadata.scheduler {
        let _ = write!(xmp, "<mold:scheduler>{scheduler}</mold:scheduler>");
    }
    if let Some(ref neg) = metadata.negative_prompt {
        let _ = write!(
            xmp,
            "<mold:negativePrompt>{}</mold:negativePrompt>",
            xml_escape(neg)
        );
    }
    if let Some(ref original) = metadata.original_prompt {
        let _ = write!(
            xmp,
            "<mold:originalPrompt>{}</mold:originalPrompt>",
            xml_escape(original)
        );
    }
    let _ = write!(
        xmp,
        "<mold:version>{}</mold:version>",
        xml_escape(&metadata.version)
    );
    let json = serde_json::to_string(metadata)?;
    let _ = write!(
        xmp,
        "<mold:parameters>{}</mold:parameters>",
        xml_escape(&json)
    );
    xmp.push_str("</rdf:Description></rdf:RDF></x:xmpmeta>");
    xmp.push_str(r#"<?xpacket end="w"?>"#);
    Ok(xmp.into_bytes())
}

/// Write a JPEG APP1 marker with the standard XMP namespace prefix.
/// Skips the marker entirely if the payload exceeds the 65535-byte segment limit.
///
/// Per JPEG spec, the segment length field (u16) counts itself (2 bytes) plus the
/// payload (namespace + XMP data). The 0xFF 0xE1 marker bytes are NOT included in
/// the length field. So `total` = 2 + namespace + xmp, and max is 0xFFFF.
fn write_jpeg_xmp_marker(out: &mut Vec<u8>, xmp_data: &[u8]) {
    let namespace = b"http://ns.adobe.com/xap/1.0/\0";
    let total = namespace.len() + xmp_data.len() + 2; // +2 for the length field itself
    if total > 0xFFFF {
        tracing::warn!("XMP packet too large for JPEG APP1 marker ({total} bytes), skipping");
        return;
    }
    let total_len = total as u16;
    out.push(0xFF);
    out.push(0xE1); // APP1 marker
    out.extend_from_slice(&total_len.to_be_bytes());
    out.extend_from_slice(namespace);
    out.extend_from_slice(xmp_data);
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::io::Cursor;

    /// Build a 3xHxW solid-red tensor (R=255, G=0, B=0).
    fn solid_red_tensor(h: usize, w: usize) -> Tensor {
        let mut data = vec![0u8; 3 * h * w];
        // Channel 0 (R) = 255, channels 1 and 2 stay 0
        for value in data.iter_mut().take(h * w) {
            *value = 255;
        }
        Tensor::from_vec(data, (3, h, w), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap()
    }

    #[test]
    fn test_encode_png_valid_tensor() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Png, 4, 4, None).unwrap();
        assert!(bytes.len() >= 4);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    /// A photographic-ish 1024² RGB image: smooth gradients plus per-pixel
    /// noise, so neither profile can win on a degenerate solid colour.
    fn synthetic_render(width: u32, height: u32) -> image::RgbImage {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        image::RgbImage::from_fn(width, height, |x, y| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state & 0x1F) as u32;
            let r = ((x * 255) / width.max(1) + noise) % 256;
            let g = ((y * 255) / height.max(1) + noise) % 256;
            let b = (((x + y) * 255) / (width + height).max(1) + noise) % 256;
            image::Rgb([r as u8, g as u8, b as u8])
        })
    }

    #[test]
    fn png_profile_defaults_to_fast_and_reads_the_env() {
        assert_eq!(png_profile_from(None), PngEncoding::Fast);
        assert_eq!(png_profile_from(Some("fast")), PngEncoding::Fast);
        assert_eq!(png_profile_from(Some(" BALANCED ")), PngEncoding::Balanced);
        assert_eq!(png_profile_from(Some("Balanced")), PngEncoding::Balanced);
        // An unrecognized value is the default, not an error: a saved print
        // must never fail because someone typed the variable wrong.
        assert_eq!(png_profile_from(Some("maximum")), PngEncoding::Fast);
        assert_eq!(png_profile_from(Some("")), PngEncoding::Fast);
    }

    #[test]
    fn fast_png_round_trips_pixel_exactly() {
        // PNG is lossless under every profile. This is the whole safety
        // argument for making the fast one the default.
        let source = synthetic_render(97, 61);
        for profile in [PngEncoding::Fast, PngEncoding::Balanced] {
            let mut buf = Cursor::new(Vec::new());
            write_png_with(&source, &mut buf, None, profile).unwrap();
            let bytes = buf.into_inner();
            let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
                .unwrap()
                .to_rgb8();
            assert_eq!(
                decoded.as_raw(),
                source.as_raw(),
                "{profile:?} PNG must decode to the exact pixels it was handed"
            );
        }
    }

    #[test]
    fn fast_png_stays_within_a_third_of_the_balanced_size() {
        // A real 512^2 photograph is the honest proxy for a render: measured
        // fast 322_448 B against balanced 304_452 B, a ratio of 1.059, for
        // 1.7 ms against 59.0 ms. The bound is 1.35 so the test pins the
        // trade rather than the exact fdeflate build.
        let photo = image::load_from_memory(include_bytes!(
            "../testdata/pulid/faces/frank-rubio-official-portrait.eva512.png"
        ))
        .unwrap()
        .to_rgb8();
        let (fast_len, balanced_len) = encoded_sizes(&photo);
        eprintln!(
            "png sizes (photograph): fast={fast_len} balanced={balanced_len} ratio={:.3}",
            fast_len as f64 / balanced_len as f64
        );
        assert!(
            fast_len as f64 <= 1.35 * balanced_len as f64,
            "fast PNG is {fast_len} B against balanced {balanced_len} B"
        );

        // The adversarial case, stated rather than hidden: an image that is
        // per-pixel noise everywhere is where fdeflate's ultra-fast mode
        // gives up the most (measured 2_741_265 B against 1_426_861 B, 1.92,
        // for 7.4 ms against 203.2 ms). No render looks like this, but the
        // bound records how far the default can go.
        let noise = synthetic_render(1024, 1024);
        let (fast_len, balanced_len) = encoded_sizes(&noise);
        eprintln!(
            "png sizes (pure noise): fast={fast_len} balanced={balanced_len} ratio={:.3}",
            fast_len as f64 / balanced_len as f64
        );
        assert!(
            fast_len as f64 <= 2.0 * balanced_len as f64,
            "fast PNG is {fast_len} B against balanced {balanced_len} B"
        );
    }

    fn encoded_sizes(image: &image::RgbImage) -> (usize, usize) {
        let mut fast = Cursor::new(Vec::new());
        write_png_with(image, &mut fast, None, PngEncoding::Fast).unwrap();
        let mut balanced = Cursor::new(Vec::new());
        write_png_with(image, &mut balanced, None, PngEncoding::Balanced).unwrap();
        (fast.into_inner().len(), balanced.into_inner().len())
    }

    #[test]
    fn test_encode_jpeg_valid_tensor() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, None).unwrap();
        assert!(bytes.len() >= 2);
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_encode_wrong_channels_fails() {
        // Only RGB (3) and RGBA (4) tensors are images.
        let data = vec![0u8; 2 * 4 * 4];
        let tensor = Tensor::from_vec(data, (2, 4, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();
        let result = encode_image(&tensor, OutputFormat::Png, 4, 4, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("expected 3 (RGB) or 4 (RGBA) channels"),
            "unexpected error: {msg}"
        );
    }

    /// A `[4, H, W]` tensor from RGBA planes (channel-first).
    fn rgba_tensor(image: &image::RgbaImage) -> Tensor {
        let (w, h) = (image.width() as usize, image.height() as usize);
        Tensor::from_vec(image.as_raw().clone(), (h, w, 4), &Device::Cpu)
            .unwrap()
            .permute((2, 0, 1))
            .unwrap()
            .contiguous()
            .unwrap()
    }

    fn rgb_tensor(image: &image::RgbImage) -> Tensor {
        let (w, h) = (image.width() as usize, image.height() as usize);
        Tensor::from_vec(image.as_raw().clone(), (h, w, 3), &Device::Cpu)
            .unwrap()
            .permute((2, 0, 1))
            .unwrap()
            .contiguous()
            .unwrap()
    }

    /// A cut-out: an opaque square on a fully transparent black field, with
    /// one half-transparent column.
    fn cutout(width: u32, height: u32) -> image::RgbaImage {
        image::RgbaImage::from_fn(width, height, |x, y| {
            if (width / 4..3 * width / 4).contains(&x) && (height / 4..3 * height / 4).contains(&y)
            {
                image::Rgba([220, 60, 20, 255])
            } else if x == 0 {
                image::Rgba([0, 0, 255, 128])
            } else {
                image::Rgba([0, 0, 0, 0])
            }
        })
    }

    #[test]
    fn opaque_rgba_png_is_byte_identical_to_the_rgb_png() {
        let rgb = synthetic_render(33, 21);
        let rgba = image::RgbaImage::from_fn(33, 21, |x, y| {
            let [r, g, b] = rgb.get_pixel(x, y).0;
            image::Rgba([r, g, b, 255])
        });
        let metadata = test_metadata();
        for format in [OutputFormat::Png, OutputFormat::Jpeg] {
            let from_rgb =
                encode_image(&rgb_tensor(&rgb), format, 33, 21, Some(&metadata)).unwrap();
            let from_rgba =
                encode_image(&rgba_tensor(&rgba), format, 33, 21, Some(&metadata)).unwrap();
            assert_eq!(
                from_rgb, from_rgba,
                "{format}: an all-opaque RGBA render must encode exactly as RGB"
            );
        }
        let info = decode_png_info(
            &encode_image(&rgba_tensor(&rgba), OutputFormat::Png, 33, 21, None).unwrap(),
        );
        assert_eq!(info.color_type, png::ColorType::Rgb);
    }

    #[test]
    fn transparent_rgba_png_keeps_its_alpha_and_records_it() {
        let source = cutout(16, 12);
        let metadata = test_metadata();
        let bytes = encode_image(
            &rgba_tensor(&source),
            OutputFormat::Png,
            16,
            12,
            Some(&metadata),
        )
        .unwrap();
        assert_eq!(decode_png_info(&bytes).color_type, png::ColorType::Rgba);
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgba8();
        assert_eq!(decoded.as_raw(), source.as_raw(), "PNG alpha is lossless");
        assert!(mold_core::still_image::encoded_still_has_alpha(&bytes));

        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains("\"has_alpha\":true"),
            "embedded provenance records the alpha channel"
        );
    }

    /// Qwen Image 2.1's decoder leaves edge alpha of 204-252 on an ordinary
    /// opaque render, so the engine decides: `Drop` must encode exactly the
    /// RGB planes, byte-identical to the RGB render, whatever alpha says.
    #[test]
    fn a_dropped_alpha_channel_encodes_the_rgb_planes_unchanged() {
        let rgb = synthetic_render(29, 17);
        let rgba = image::RgbaImage::from_fn(29, 17, |x, y| {
            let [r, g, b] = rgb.get_pixel(x, y).0;
            image::Rgba([r, g, b, if x == 0 { 204 } else { 252 }])
        });
        let metadata = test_metadata();
        for format in [OutputFormat::Png, OutputFormat::Jpeg] {
            let from_rgb =
                encode_image(&rgb_tensor(&rgb), format, 29, 17, Some(&metadata)).unwrap();
            let dropped = encode_image_with_alpha(
                &rgba_tensor(&rgba),
                format,
                29,
                17,
                Some(&metadata),
                AlphaOutput::Drop,
            )
            .unwrap();
            assert_eq!(from_rgb, dropped, "{format}");
        }
    }

    #[test]
    fn a_kept_alpha_channel_is_written_even_when_fully_opaque() {
        let rgba = image::RgbaImage::from_pixel(8, 8, image::Rgba([10, 20, 30, 255]));
        let bytes = encode_image_with_alpha(
            &rgba_tensor(&rgba),
            OutputFormat::Png,
            8,
            8,
            Some(&test_metadata()),
            AlphaOutput::Keep,
        )
        .unwrap();
        assert_eq!(decode_png_info(&bytes).color_type, png::ColorType::Rgba);
        assert!(String::from_utf8_lossy(&bytes).contains("\"has_alpha\":true"));
        // JPEG cannot carry it: composited over white, provenance says so.
        let jpeg = encode_image_with_alpha(
            &rgba_tensor(&cutout(8, 8)),
            OutputFormat::Jpeg,
            8,
            8,
            Some(&test_metadata()),
            AlphaOutput::Keep,
        )
        .unwrap();
        assert!(!String::from_utf8_lossy(&jpeg).contains("\"has_alpha\":true"));
    }

    fn png_bytes(image: &image::RgbaImage) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        image.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn the_request_decides_whether_alpha_survives() {
        let mut req: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a lantern", "model": "qwen-image-2.1:bf16", "width": 64,
            "height": 64, "steps": 4, "guidance": 1.0, "batch_size": 1
        }))
        .unwrap();
        assert_eq!(alpha_output_for_request(&req).unwrap(), AlphaOutput::Drop);
        req.transparent_background = Some(true);
        assert_eq!(alpha_output_for_request(&req).unwrap(), AlphaOutput::Keep);
        req.transparent_background = None;

        // An RGBA container whose alpha is all 255 carries no transparency.
        let opaque = image::RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255]));
        req.edit_images = Some(vec![png_bytes(&opaque), vec![0xFF, 0xD8, 0xFF]]);
        assert_eq!(alpha_output_for_request(&req).unwrap(), AlphaOutput::Drop);
        // One reference with real transparency keeps alpha in the output.
        req.edit_images
            .as_mut()
            .unwrap()
            .push(png_bytes(&cutout(8, 8)));
        assert_eq!(alpha_output_for_request(&req).unwrap(), AlphaOutput::Keep);
    }

    /// Alpha is read the way upstream's PIL reads it, through the decoder the
    /// engine conditions on: a 16-bit alpha of `0xFF00` is 255 to Pillow (the
    /// high byte), so that reference is OPAQUE and the output drops alpha —
    /// the `image` crate's `x / 257` would have read 254 and kept it.
    #[test]
    fn sixteen_bit_alpha_is_read_as_pillow_reads_it() {
        let rgba16 = |alpha: u16| {
            let image: image::ImageBuffer<image::Rgba<u16>, Vec<u16>> =
                image::ImageBuffer::from_pixel(4, 4, image::Rgba([0x1234, 0x5678, 0x9abc, alpha]));
            let mut buf = Cursor::new(Vec::new());
            image.write_to(&mut buf, image::ImageFormat::Png).unwrap();
            buf.into_inner()
        };
        assert!(!encoded_image_has_alpha(&rgba16(0xff00)).unwrap());
        assert!(!encoded_image_has_alpha(&rgba16(0xffff)).unwrap());
        assert!(encoded_image_has_alpha(&rgba16(0xfeff)).unwrap());
    }

    /// An alpha-capable reference that cannot be decoded is an ERROR: the
    /// engine could not condition on it, and "no alpha" would silently
    /// flatten an output the reference may have asked to keep transparent.
    #[test]
    fn an_undecodable_alpha_reference_is_an_error_not_opaque() {
        let mut truncated = png_bytes(&cutout(8, 8));
        truncated.truncate(40);
        assert!(encoded_image_has_alpha(&truncated).is_err());
        let mut req: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a lantern", "model": "qwen-image-2.1:bf16", "width": 64,
            "height": 64, "steps": 4, "guidance": 1.0, "batch_size": 1
        }))
        .unwrap();
        req.edit_images = Some(vec![truncated]);
        assert!(alpha_output_for_request(&req).is_err());
    }

    #[test]
    fn transparent_rgba_jpeg_is_composited_over_white() {
        let source = image::RgbaImage::from_fn(8, 8, |_, _| image::Rgba([0, 0, 0, 0]));
        let bytes = encode_image(&rgba_tensor(&source), OutputFormat::Jpeg, 8, 8, None).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert!(
            decoded
                .pixels()
                .all(|pixel| pixel.0.iter().all(|&c| c >= 250)),
            "a fully transparent pixel lands on the white canvas"
        );
        assert_eq!(
            crate::pillow_resize::composite_over_white(&image::RgbaImage::from_pixel(
                1,
                1,
                image::Rgba([0, 100, 200, 128])
            ))
            .get_pixel(0, 0)
            .0,
            [127, 177, 227]
        );
    }

    #[cfg(feature = "webp")]
    #[test]
    fn webp_stills_encode_for_rgb_and_rgba_tensors() {
        // The all-family fix: an ordinary three-channel render encodes as a
        // still WebP instead of failing after the render.
        let rgb = synthetic_render(24, 16);
        let bytes = encode_image(&rgb_tensor(&rgb), OutputFormat::Webp, 24, 16, None).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[12..16], b"VP8 ");
        assert!(!crate::webp_still::has_anim_chunk(&bytes));
        assert!(!mold_core::still_image::webp_is_animated(&bytes));
        assert!(!OutputFormat::Webp.is_video_artifact(&bytes));
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (24, 16));

        let source = cutout(24, 16);
        let bytes = encode_image(&rgba_tensor(&source), OutputFormat::Webp, 24, 16, None).unwrap();
        assert!(matches!(&bytes[12..16], b"VP8X" | b"VP8L"));
        assert!(!crate::webp_still::has_anim_chunk(&bytes));
        assert!(mold_core::still_image::encoded_still_has_alpha(&bytes));
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgba8();
        let alpha = |image: &image::RgbaImage| image.pixels().map(|p| p[3]).collect::<Vec<_>>();
        assert_eq!(alpha(&decoded), alpha(&source));
    }

    #[cfg(not(feature = "webp"))]
    #[test]
    fn webp_stills_name_the_missing_feature() {
        let rgb = synthetic_render(4, 4);
        let err = encode_image(&rgb_tensor(&rgb), OutputFormat::Webp, 4, 4, None).unwrap_err();
        assert!(err.to_string().contains("'webp' feature"), "{err}");
    }

    #[test]
    fn test_encode_single_pixel() {
        let tensor = solid_red_tensor(1, 1);
        let bytes = encode_image(&tensor, OutputFormat::Png, 1, 1, None).unwrap();
        // Must be a valid PNG
        assert!(bytes.len() >= 4);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn test_encode_both_formats_succeed() {
        let tensor = solid_red_tensor(4, 4);
        let png = encode_image(&tensor, OutputFormat::Png, 4, 4, None).unwrap();
        let jpeg = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, None).unwrap();
        assert!(!png.is_empty(), "PNG output should not be empty");
        assert!(!jpeg.is_empty(), "JPEG output should not be empty");
        // Formats should differ in content
        assert_ne!(png, jpeg, "PNG and JPEG outputs should differ");
    }

    fn decode_png_info(bytes: &[u8]) -> png::Info<'static> {
        let decoder = png::Decoder::new(Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let out_size = reader.output_buffer_size().unwrap();
        let mut buf = vec![0; out_size];
        reader.next_frame(&mut buf).unwrap();
        reader.info().clone()
    }

    #[test]
    fn test_encode_png_with_metadata_chunks() {
        let tensor = solid_red_tensor(4, 4);
        let metadata = OutputMetadata {
            family: None,
            mesh_workflow: None,
            video_only: None,
            attention_path: None,
            int8_arm: None,
            collection: None,
            tags: None,
            title: None,
            generation_time_ms: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "hello \u{2603}".to_string(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: Some(mold_core::GenerationOutputMode::OneShot),
            model: "flux-schnell:q8".to_string(),
            seed: 42,
            steps: 4,
            guidance: 0.0,
            width: 4,
            height: 4,
            generation_width: Some(4),
            mesh: None,
            generation_height: Some(4),
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: None,
            output_format: Some(OutputFormat::Png),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            duration_prediction_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            chain_job_id: None,
            chain: None,
            version: "0.1.0".to_string(),
            frames: None,
            fps: None,
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
            has_alpha: None,
            prefix_cache: None,
            transparent_background: None,
        };

        let bytes = encode_image(&tensor, OutputFormat::Png, 4, 4, Some(&metadata)).unwrap();
        let info = decode_png_info(&bytes);

        assert!(info
            .utf8_text
            .iter()
            .any(|chunk| chunk.keyword == "mold:prompt"
                && chunk.get_text().unwrap() == "hello \u{2603}"));
        assert!(info
            .utf8_text
            .iter()
            .any(|chunk| chunk.keyword == "mold:model"
                && chunk.get_text().unwrap() == "flux-schnell:q8"));
        assert!(info
            .utf8_text
            .iter()
            .any(|chunk| chunk.keyword == "mold:parameters"
                && chunk
                    .get_text()
                    .unwrap()
                    .contains("\"model\":\"flux-schnell:q8\"")));
        assert!(info
            .uncompressed_latin1_text
            .iter()
            .any(|chunk| chunk.keyword == "mold:seed" && chunk.text == "42"));
    }

    #[test]
    fn test_encode_png_without_metadata_chunks() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Png, 4, 4, None).unwrap();
        let info = decode_png_info(&bytes);
        assert!(info.uncompressed_latin1_text.is_empty());
        assert!(info.utf8_text.is_empty());
    }

    #[test]
    fn test_build_output_metadata_respects_opt_out() {
        let req = GenerateRequest {
            mesh_workflow: None,
            offload: None,
            mesh: None,
            video_only: None,
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            prompt: "a cat".to_string(),
            negative_prompt: None,
            model: "flux-schnell:q8".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 0.0,
            seed: Some(42),
            batch_size: 1,
            output_format: Some(OutputFormat::Png),
            embed_metadata: Some(false),
            scheduler: None,
            cfg_plus: None,
            edit_images: None,
            reference_weight: None,
            references: None,
            source_image: None,
            source_image_name: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            save_to_gallery: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
            transparent_background: None,
        };

        assert!(build_output_metadata(&req, 42, None).is_none());
    }

    #[test]
    fn test_update_output_metadata_size_overrides_dimensions() {
        let mut metadata = Some(OutputMetadata {
            family: None,
            mesh_workflow: None,
            video_only: None,
            attention_path: None,
            int8_arm: None,
            collection: None,
            tags: None,
            title: None,
            generation_time_ms: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "a cat".to_string(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: Some(mold_core::GenerationOutputMode::OneShot),
            model: "wuerstchen-v2:fp16".to_string(),
            seed: 42,
            steps: 30,
            guidance: 0.0,
            width: 1024,
            height: 1024,
            generation_width: Some(1024),
            mesh: None,
            generation_height: Some(1024),
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: None,
            output_format: Some(OutputFormat::Png),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            duration_prediction_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            chain_job_id: None,
            chain: None,
            version: "0.1.0".to_string(),
            frames: None,
            fps: None,
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
            has_alpha: None,
            prefix_cache: None,
            transparent_background: None,
        });

        update_output_metadata_size(&mut metadata, 1008, 1008);

        let metadata = metadata.unwrap();
        assert_eq!(metadata.width, 1008);
        assert_eq!(metadata.height, 1008);
    }

    // ── JPEG metadata tests ───────────────────────────────────────────────

    fn test_metadata() -> OutputMetadata {
        OutputMetadata {
            family: None,
            mesh_workflow: None,
            video_only: None,
            attention_path: None,
            int8_arm: None,
            collection: None,
            tags: None,
            title: None,
            generation_time_ms: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "hello world".to_string(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: Some(mold_core::GenerationOutputMode::OneShot),
            model: "flux-schnell:q8".to_string(),
            seed: 42,
            steps: 4,
            guidance: 0.0,
            width: 4,
            height: 4,
            generation_width: Some(4),
            mesh: None,
            generation_height: Some(4),
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: None,
            output_format: Some(OutputFormat::Jpeg),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            duration_prediction_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            chain_job_id: None,
            chain: None,
            version: "0.1.0".to_string(),
            frames: None,
            fps: None,
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
            has_alpha: None,
            prefix_cache: None,
            transparent_background: None,
        }
    }

    /// Find the first JPEG COM (0xFFFE) marker and return its payload.
    fn find_jpeg_comment(bytes: &[u8]) -> Option<Vec<u8>> {
        let mut i = 2; // skip SOI
        while i + 3 < bytes.len() {
            if bytes[i] != 0xFF {
                break;
            }
            let marker = bytes[i + 1];
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            if marker == 0xFE {
                // COM marker
                return Some(bytes[i + 4..i + 2 + len].to_vec());
            }
            i += 2 + len;
        }
        None
    }

    /// Find the first JPEG APP1 XMP marker and return the XMP payload (after namespace).
    fn find_jpeg_xmp(bytes: &[u8]) -> Option<String> {
        let namespace = b"http://ns.adobe.com/xap/1.0/\0";
        let mut i = 2; // skip SOI
        while i + 3 < bytes.len() {
            if bytes[i] != 0xFF {
                break;
            }
            let marker = bytes[i + 1];
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            if marker == 0xE1 {
                let payload = &bytes[i + 4..i + 2 + len];
                if payload.starts_with(namespace) {
                    let xmp_bytes = &payload[namespace.len()..];
                    return String::from_utf8(xmp_bytes.to_vec()).ok();
                }
            }
            i += 2 + len;
        }
        None
    }

    #[test]
    fn test_encode_jpeg_with_metadata_has_comment() {
        let tensor = solid_red_tensor(4, 4);
        let metadata = test_metadata();
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, Some(&metadata)).unwrap();

        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "should be valid JPEG");
        let comment = find_jpeg_comment(&bytes).expect("should have COM marker");
        let comment_str = String::from_utf8(comment).unwrap();
        assert!(
            comment_str.starts_with("mold:parameters "),
            "comment should start with mold:parameters: {comment_str}"
        );
        let json_str = &comment_str["mold:parameters ".len()..];
        let parsed: OutputMetadata = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.prompt, "hello world");
        assert_eq!(parsed.model, "flux-schnell:q8");
        assert_eq!(parsed.seed, 42);
    }

    #[test]
    fn test_encode_jpeg_with_metadata_has_xmp() {
        let tensor = solid_red_tensor(4, 4);
        let metadata = test_metadata();
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, Some(&metadata)).unwrap();

        let xmp = find_jpeg_xmp(&bytes).expect("should have XMP APP1 marker");
        assert!(xmp.contains("mold:prompt"), "XMP should contain prompt");
        assert!(
            xmp.contains("hello world"),
            "XMP should contain prompt text"
        );
        assert!(xmp.contains("mold:seed"), "XMP should contain seed element");
        assert!(xmp.contains("<mold:seed>42</mold:seed>"), "seed value");
        assert!(
            xmp.contains("xmlns:mold=\"https://github.com/utensils/mold\""),
            "XMP should have mold namespace"
        );
        assert!(
            xmp.contains("mold:parameters"),
            "XMP should contain parameters JSON"
        );
    }

    #[test]
    fn test_encode_jpeg_without_metadata_no_extra_markers() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, None).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
        assert!(
            find_jpeg_comment(&bytes).is_none(),
            "no COM marker without metadata"
        );
        assert!(
            find_jpeg_xmp(&bytes).is_none(),
            "no XMP marker without metadata"
        );
    }

    #[test]
    fn test_encode_jpeg_metadata_roundtrip() {
        let tensor = solid_red_tensor(8, 8);
        let metadata = OutputMetadata {
            family: None,
            mesh_workflow: None,
            video_only: None,
            attention_path: None,
            int8_arm: None,
            collection: None,
            tags: None,
            title: None,
            generation_time_ms: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "a cat & a dog <br>".to_string(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: Some(mold_core::GenerationOutputMode::OneShot),
            model: "sdxl-turbo:fp16".to_string(),
            seed: 99999,
            steps: 25,
            guidance: 7.5,
            width: 8,
            height: 8,
            generation_width: Some(8),
            mesh: None,
            generation_height: Some(8),
            strength: Some(0.6),
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: Some(mold_core::Scheduler::EulerAncestral),
            output_format: Some(OutputFormat::Jpeg),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            duration_prediction_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            chain_job_id: None,
            chain: None,
            version: "0.2.0".to_string(),
            frames: None,
            fps: None,
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
            has_alpha: None,
            prefix_cache: None,
            transparent_background: None,
        };
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 8, 8, Some(&metadata)).unwrap();

        // Roundtrip via COM JSON
        let comment = find_jpeg_comment(&bytes).unwrap();
        let json_str = String::from_utf8(comment).unwrap();
        let json_str = &json_str["mold:parameters ".len()..];
        let parsed: OutputMetadata = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed, metadata);

        // XMP should have XML-escaped special characters
        let xmp = find_jpeg_xmp(&bytes).unwrap();
        assert!(
            xmp.contains("a cat &amp; a dog &lt;br&gt;"),
            "prompt should be XML-escaped in XMP: {xmp}"
        );
        assert!(
            xmp.contains("<mold:strength>0.6</mold:strength>"),
            "strength should be present"
        );
        assert!(
            xmp.contains("<mold:scheduler>euler-ancestral</mold:scheduler>"),
            "scheduler should be present"
        );
    }

    #[test]
    fn test_build_xmp_packet_returns_ok() {
        let metadata = test_metadata();
        let xmp = build_xmp_packet(&metadata);
        assert!(
            xmp.is_ok(),
            "build_xmp_packet should not fail for valid metadata"
        );
        let xmp_str = String::from_utf8(xmp.unwrap()).unwrap();
        assert!(xmp_str.contains("mold:parameters"));
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("hello"), "hello");
        assert_eq!(xml_escape("a & b"), "a &amp; b");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape(r#"say "hi""#), "say &quot;hi&quot;");
        assert_eq!(xml_escape("a < b & c > d"), "a &lt; b &amp; c &gt; d");
    }
}
