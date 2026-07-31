/** Pixel dimensions decoded directly from a PNG/JPEG base64 header. */
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

function decodedPrefix(base64: string): Uint8Array | null {
  const comma = base64.indexOf(",");
  const payload = (comma >= 0 ? base64.slice(comma + 1) : base64)
    .replace(/\s+/g, "")
    .replace(/-/g, "+")
    .replace(/_/g, "/");
  if (!payload) return null;

  // Base64 quartets decode independently, so a quartet-aligned prefix remains
  // valid even when the full image is much larger than our metadata budget.
  const prefixLength = Math.min(payload.length, MAX_HEADER_BASE64_CHARS) & ~3;
  if (prefixLength === 0) return null;
  try {
    const binary = globalThis.atob(payload.slice(0, prefixLength));
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  } catch {
    return null;
  }
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

/**
 * Decode PNG/JPEG dimensions from raw base64 or a data URL.
 *
 * Returns `null` for malformed/unsupported media or a JPEG whose SOF marker
 * falls beyond the bounded metadata prefix.
 */
export function imageDimensionsFromBase64(
  base64: string,
): ImageDimensions | null {
  const bytes = decodedPrefix(base64);
  if (!bytes) return null;
  return pngDimensions(bytes) ?? jpegDimensions(bytes);
}
