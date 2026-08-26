export type Base64DigestErrorCode = "invalid-base64" | "hash-unavailable";

export class Base64DigestError extends Error {
  constructor(
    readonly code: Base64DigestErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "Base64DigestError";
  }
}

function validBase64Character(code: number): boolean {
  return (
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122) ||
    (code >= 48 && code <= 57) ||
    code === 43 ||
    code === 47
  );
}

/** Decode strict, padded RFC 4648 standard base64. The Rust wire uses the
 * same alphabet and rejects unpadded, whitespace-containing, or URL-safe
 * inputs, so attachment hashing and upload validation cannot disagree. */
export function decodePaddedBase64(value: string): Uint8Array<ArrayBuffer> {
  if (value.length === 0 || value.length % 4 !== 0) {
    throw new Base64DigestError(
      "invalid-base64",
      "Media does not contain valid padded base64.",
    );
  }
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  for (let index = 0; index < value.length - padding; index += 1) {
    if (!validBase64Character(value.charCodeAt(index))) {
      throw new Base64DigestError(
        "invalid-base64",
        "Media does not contain valid padded base64.",
      );
    }
  }
  for (let index = value.length - padding; index < value.length; index += 1) {
    if (value.charCodeAt(index) !== 61) {
      throw new Base64DigestError(
        "invalid-base64",
        "Media does not contain valid padded base64.",
      );
    }
  }
  const finalAlphabetLength = value.length - padding;
  if (
    (padding === 2 && finalAlphabetLength % 4 !== 2) ||
    (padding === 1 && finalAlphabetLength % 4 !== 3)
  ) {
    throw new Base64DigestError(
      "invalid-base64",
      "Media does not contain valid padded base64.",
    );
  }

  let binary: string;
  try {
    binary = globalThis.atob(value);
  } catch {
    throw new Base64DigestError(
      "invalid-base64",
      "Media does not contain valid padded base64.",
    );
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

export async function sha256PaddedBase64(value: string): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Base64DigestError(
      "hash-unavailable",
      "Secure media hashing is unavailable on this device.",
    );
  }
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    decodePaddedBase64(value),
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}
