export { blobToBase64 } from "@studio/lib/base64";

/** Inverse — base64 → Blob, used by the image picker's gallery tab. */
export function base64ToBlob(b64: string, mime: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}
