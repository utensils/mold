/**
 * Browser-safe RFC 4122 v4 identifier.
 *
 * `crypto.randomUUID()` is limited to secure contexts in browsers, while Mold
 * is commonly opened from another LAN machine over plain HTTP. Keep native
 * UUIDs when available, then fall back to `getRandomValues()` (which remains
 * available in those contexts), with a last-resort random byte fill for older
 * WebViews.
 */
export interface UuidCrypto {
  randomUUID?: () => string;
  getRandomValues?: (bytes: Uint8Array) => Uint8Array;
}

export function createUuid(
  source: UuidCrypto | undefined = globalThis.crypto,
): string {
  if (typeof source?.randomUUID === "function") {
    try {
      return source.randomUUID();
    } catch {
      // Some older browsers expose the method but reject it off HTTPS.
    }
  }

  const bytes = new Uint8Array(16);
  if (typeof source?.getRandomValues === "function") {
    source.getRandomValues(bytes);
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }

  bytes[6] = (bytes[6]! & 0x0f) | 0x40;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, "0"));
  return [
    hex.slice(0, 4).join(""),
    hex.slice(4, 6).join(""),
    hex.slice(6, 8).join(""),
    hex.slice(8, 10).join(""),
    hex.slice(10).join(""),
  ].join("-");
}
