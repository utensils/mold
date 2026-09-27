import type { ImageInputFormat } from "./generated/generationProfileV1";

/** Pixel dimensions decoded directly from a PNG/JPEG/WebP base64 header. */
export interface ImageDimensions {
  width: number;
  height: number;
}

// Source images can be tens of MiB. Dimension discovery must not duplicate the
// whole payload into an `atob()` binary string on the UI thread. PNG needs only
// 24 bytes; JPEG normally places its SOF marker near the front, while 1 MiB
// still leaves room for unusually large EXIF/ICC metadata.
const MAX_HEADER_BYTES = 1024 * 1024;
const MAX_HEADER_BASE64_CHARS = Math.ceil(MAX_HEADER_BYTES / 3) * 4;

function normalizedPayload(base64: string): string {
  const comma = base64.indexOf(",");
  return (comma >= 0 ? base64.slice(comma + 1) : base64)
    .replace(/\s+/g, "")
    .replace(/-/g, "+")
    .replace(/_/g, "/");
}

/**
 * `length` bytes starting at byte `start` of a normalized base64 payload,
 * decoding only the quartets that cover them. Base64 quartets decode
 * independently, so any quartet-aligned window is valid on its own.
 */
function decodedRange(
  payload: string,
  start: number,
  length: number,
): Uint8Array | null {
  const skip = start % 3;
  const from = ((start - skip) / 3) * 4;
  const to = Math.min(
    payload.length,
    from + Math.ceil((skip + length) / 3) * 4,
  );
  if (from >= to) return null;
  try {
    const binary = globalThis.atob(payload.slice(from, to));
    return Uint8Array.from(binary, (character) =>
      character.charCodeAt(0),
    ).subarray(skip, skip + length);
  } catch {
    return null;
  }
}

function decodedPrefix(payload: string): Uint8Array | null {
  if (!payload) return null;
  // A quartet-aligned prefix remains valid even when the full image is much
  // larger than our metadata budget.
  const prefixLength = Math.min(payload.length, MAX_HEADER_BASE64_CHARS) & ~3;
  if (prefixLength === 0) return null;
  return decodedRange(payload, 0, (prefixLength / 4) * 3);
}

function u16be(bytes: Uint8Array, offset: number): number {
  return bytes[offset]! * 0x100 + bytes[offset + 1]!;
}

function u32be(bytes: Uint8Array, offset: number): number {
  return (
    bytes[offset]! * 0x1000000 +
    bytes[offset + 1]! * 0x10000 +
    bytes[offset + 2]! * 0x100 +
    bytes[offset + 3]!
  );
}

function pngDimensions(bytes: Uint8Array): ImageDimensions | null {
  const pngSignature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (
    bytes.length < 24 ||
    pngSignature.some((value, index) => bytes[index] !== value) ||
    u32be(bytes, 8) !== 13 ||
    bytes[12] !== 0x49 ||
    bytes[13] !== 0x48 ||
    bytes[14] !== 0x44 ||
    bytes[15] !== 0x52
  ) {
    return null;
  }
  const width = u32be(bytes, 16);
  const height = u32be(bytes, 20);
  return width > 0 && height > 0 ? { width, height } : null;
}

function isStartOfFrame(marker: number): boolean {
  return (
    (marker >= 0xc0 && marker <= 0xc3) ||
    (marker >= 0xc5 && marker <= 0xc7) ||
    (marker >= 0xc9 && marker <= 0xcb) ||
    (marker >= 0xcd && marker <= 0xcf)
  );
}

function jpegDimensions(bytes: Uint8Array): ImageDimensions | null {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    return null;
  }

  let offset = 2;
  while (offset < bytes.length) {
    while (offset < bytes.length && bytes[offset] !== 0xff) offset += 1;
    while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
    if (offset >= bytes.length) return null;

    const marker = bytes[offset]!;
    offset += 1;
    if (marker === 0xda || marker === 0xd9) return null;
    // Standalone markers carry no length or payload.
    if (
      marker === 0x01 ||
      marker === 0xd8 ||
      (marker >= 0xd0 && marker <= 0xd7)
    ) {
      continue;
    }
    if (offset + 2 > bytes.length) return null;
    const segmentLength = u16be(bytes, offset);
    if (segmentLength < 2) return null;

    if (isStartOfFrame(marker)) {
      if (segmentLength < 7 || offset + 7 > bytes.length) return null;
      const height = u16be(bytes, offset + 3);
      const width = u16be(bytes, offset + 5);
      return width > 0 && height > 0 ? { width, height } : null;
    }
    offset += segmentLength;
  }
  return null;
}

function u16le(bytes: Uint8Array, offset: number): number {
  return bytes[offset]! + bytes[offset + 1]! * 0x100;
}

function u24le(bytes: Uint8Array, offset: number): number {
  return u16le(bytes, offset) + bytes[offset + 2]! * 0x10000;
}

function fourcc(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + 4));
}

/**
 * WebP (`RIFF....WEBP`, the container `mold_core::validation::
 * sniff_image_input_format` accepts for a recipe advertising `webp`
 * references). The first chunk carries the canvas: `VP8X` (extended — alpha,
 * EXIF, ICC) stores it as 24-bit minus-one values; `VP8L` (lossless) packs
 * 14-bit minus-one values after its 0x2f signature; `VP8 ` (lossy) stores
 * 14-bit values after the 0x9d012a start code.
 */
function webpDimensions(bytes: Uint8Array): ImageDimensions | null {
  if (
    bytes.length < 30 ||
    fourcc(bytes, 0) !== "RIFF" ||
    fourcc(bytes, 8) !== "WEBP"
  ) {
    return null;
  }
  const chunk = fourcc(bytes, 12);
  let width = 0;
  let height = 0;
  if (chunk === "VP8X") {
    width = u24le(bytes, 24) + 1;
    height = u24le(bytes, 27) + 1;
  } else if (chunk === "VP8L") {
    if (bytes[20] !== 0x2f) return null;
    const bits =
      bytes[21]! +
      bytes[22]! * 0x100 +
      bytes[23]! * 0x10000 +
      bytes[24]! * 0x1000000;
    width = (bits & 0x3fff) + 1;
    height = ((bits >>> 14) & 0x3fff) + 1;
  } else if (chunk === "VP8 ") {
    if (bytes[23] !== 0x9d || bytes[24] !== 0x01 || bytes[25] !== 0x2a) {
      return null;
    }
    width = u16le(bytes, 26) & 0x3fff;
    height = u16le(bytes, 28) & 0x3fff;
  } else {
    return null;
  }
  return width > 0 && height > 0 ? { width, height } : null;
}

/** The containers a caller that names none accepts: `source_image`, masks,
 * keyframes and identity photos are PNG/JPEG at admission. */
const DEFAULT_FORMATS: readonly ImageInputFormat[] = ["png", "jpeg"];

/**
 * Decode PNG/JPEG (and, where `formats` names it, WebP) dimensions from raw
 * base64 or a data URL.
 *
 * The answer doubles as a FORMAT GATE for every well that calls it — a
 * `null` is how a source well refuses a GIF — so WebP is opt-in: only a
 * reference strip whose recipe advertises WebP (`reference_images.formats`)
 * passes it, and every PNG/JPEG-only door stays exactly as strict as before.
 *
 * Returns `null` for malformed/unsupported media or a JPEG whose SOF marker
 * falls beyond the bounded metadata prefix.
 */
export function imageDimensionsFromBase64(
  base64: string,
  formats: readonly ImageInputFormat[] = DEFAULT_FORMATS,
): ImageDimensions | null {
  const bytes = decodedPrefix(normalizedPayload(base64));
  if (!bytes) return null;
  return (
    (formats.includes("png") ? pngDimensions(bytes) : null) ??
    (formats.includes("jpeg") ? jpegDimensions(bytes) : null) ??
    (formats.includes("webp") ? webpDimensions(bytes) : null)
  );
}

/**
 * The EXIF `Orientation` a TIFF block carries, exactly as the Rust `image`
 * crate reads it (`Orientation::from_exif_chunk`, image 0.25.10
 * `src/metadata.rs`), because that is the reader the engines decode with:
 * `II*\0` or `MM\0*` magic, IFD0 only, the first entry tagged `0x0112` of
 * type SHORT (3) with count 1. A value outside 1-8, a truncated directory, or
 * any other magic is "no orientation".
 */
function exifOrientation(tiff: Uint8Array): number | null {
  if (tiff.length < 8) return null;
  let little: boolean;
  if (tiff[0] === 0x49 && tiff[1] === 0x49 && tiff[2] === 42 && tiff[3] === 0) {
    little = true;
  } else if (
    tiff[0] === 0x4d &&
    tiff[1] === 0x4d &&
    tiff[2] === 0 &&
    tiff[3] === 42
  ) {
    little = false;
  } else {
    return null;
  }
  const u16 = (offset: number) =>
    little ? u16le(tiff, offset) : u16be(tiff, offset);
  const u32 = (offset: number) =>
    little
      ? u16le(tiff, offset) + u16le(tiff, offset + 2) * 0x10000
      : u32be(tiff, offset);
  const directory = u32(4);
  if (directory + 2 > tiff.length) return null;
  const entries = u16(directory);
  for (let index = 0; index < entries; index += 1) {
    const entry = directory + 2 + index * 12;
    if (entry + 12 > tiff.length) return null;
    if (u16(entry) === 0x0112 && u16(entry + 2) === 3 && u32(entry + 4) === 1) {
      const value = u16(entry + 8);
      return value >= 1 && value <= 8 ? value : null;
    }
  }
  return null;
}

/**
 * The EXIF block of a JPEG: the payload after `Exif\0\0` of the LAST `APP1`
 * segment before the scan, which is the one zune-jpeg (the `image` crate's
 * JPEG decoder) keeps.
 */
function jpegExif(bytes: Uint8Array): Uint8Array | null {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  let exif: Uint8Array | null = null;
  let offset = 2;
  while (offset < bytes.length) {
    while (offset < bytes.length && bytes[offset] !== 0xff) offset += 1;
    while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
    if (offset >= bytes.length) break;
    const marker = bytes[offset]!;
    offset += 1;
    if (marker === 0xda || marker === 0xd9) break;
    if (
      marker === 0x01 ||
      marker === 0xd8 ||
      (marker >= 0xd0 && marker <= 0xd7)
    ) {
      continue;
    }
    if (offset + 2 > bytes.length) break;
    const segmentLength = u16be(bytes, offset);
    if (segmentLength < 2) break;
    if (
      marker === 0xe1 &&
      segmentLength - 2 > 6 &&
      offset + 8 <= bytes.length &&
      fourcc(bytes, offset + 2) === "Exif" &&
      bytes[offset + 6] === 0 &&
      bytes[offset + 7] === 0
    ) {
      exif = bytes.subarray(offset + 8, offset + segmentLength);
    }
    offset += segmentLength;
  }
  return exif;
}

/** A PNG's `eXIf` chunk (raw TIFF), which must precede the image data. */
function pngExif(bytes: Uint8Array): Uint8Array | null {
  let offset = 8;
  while (offset + 8 <= bytes.length) {
    const length = u32be(bytes, offset);
    const type = fourcc(bytes, offset + 4);
    if (type === "IDAT" || type === "IEND") return null;
    if (type === "eXIf") {
      return bytes.subarray(offset + 8, offset + 8 + length);
    }
    offset += 12 + length;
  }
  return null;
}

/**
 * An extended WebP's first `EXIF` chunk (raw TIFF; image-webp reads chunks
 * only behind a `VP8X` header). The container puts it AFTER the image data,
 * so it usually lies beyond the header prefix: each chunk header is decoded on
 * its own, eight bytes at a time, never the whole payload.
 */
function webpExif(payload: string, prefix: Uint8Array): Uint8Array | null {
  if (prefix.length < 30 || fourcc(prefix, 12) !== "VP8X") return null;
  const padding = payload.endsWith("==") ? 2 : payload.endsWith("=") ? 1 : 0;
  const total = (payload.length / 4) * 3 - padding;
  let position = 12;
  while (position + 8 <= total) {
    const header = decodedRange(payload, position, 8);
    if (!header || header.length < 8) return null;
    const size =
      header[4]! +
      header[5]! * 0x100 +
      header[6]! * 0x10000 +
      header[7]! * 0x1000000;
    if (fourcc(header, 0) === "EXIF") {
      return decodedRange(payload, position + 8, size);
    }
    position += 8 + size + (size & 1);
  }
  return null;
}

/**
 * {@link imageDimensionsFromBase64} with the EXIF orientation applied: the
 * size the picture has once it is decoded upright. Orientations 5-8 swap the
 * sides.
 *
 * Reference images are decoded upright by the engine
 * (`mold_inference::img_utils::decode_oriented_srgb_rgba`), and the CLI, the
 * server and the bots read a reference's size through the same `image` crate
 * reader (`mold_core::reference_image::oriented_dimensions`), so a canvas
 * derived from a reference must be read here. Source wells keep
 * {@link imageDimensionsFromBase64}: their engines read the stored pixels.
 */
export function orientedImageDimensionsFromBase64(
  base64: string,
  formats: readonly ImageInputFormat[] = DEFAULT_FORMATS,
): ImageDimensions | null {
  const payload = normalizedPayload(base64);
  const bytes = decodedPrefix(payload);
  if (!bytes) return null;
  let dimensions: ImageDimensions | null = null;
  let exif: Uint8Array | null = null;
  if (formats.includes("png") && (dimensions = pngDimensions(bytes))) {
    exif = pngExif(bytes);
  } else if (formats.includes("jpeg") && (dimensions = jpegDimensions(bytes))) {
    exif = jpegExif(bytes);
  } else if (formats.includes("webp") && (dimensions = webpDimensions(bytes))) {
    exif = webpExif(payload, bytes);
  }
  if (!dimensions) return null;
  const orientation = exif ? exifOrientation(exif) : null;
  return orientation !== null && orientation >= 5
    ? { width: dimensions.height, height: dimensions.width }
    : dimensions;
}
