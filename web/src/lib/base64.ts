/**
 * Encode a Blob / File / ArrayBuffer as raw base64 (no data-URI prefix).
 * The server expects bare base64 in GenerateRequest.source_image.
 */
export async function blobToBase64(input: Blob): Promise<string> {
  const buffer = await input.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(
      ...bytes.subarray(i, Math.min(i + chunk, bytes.length)),
    );
  }
  return btoa(binary);
}

/** Inverse — base64 → Blob, used by the image picker's gallery tab. */
export function base64ToBlob(b64: string, mime: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}
