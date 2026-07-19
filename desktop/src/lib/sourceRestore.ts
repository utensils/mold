/**
 * Reuse-settings source-image restore. Newer servers record two additive
 * provenance fields in `OutputMetadata`: `source_image_name` (the gallery
 * filename or upload name the source was picked from) and
 * `source_image_sha256` (hash of the exact bytes that shipped). Restoration
 * is best-effort, in order:
 *
 *   1. The local source stash by sha — covers uploads and canvas-fitted
 *      sources that exist nowhere else (`source_stash_put` at submit time).
 *   2. A gallery filename match, fetched from the print's own origin host —
 *      covers picks from any connected host's gallery.
 *
 * Old prints without either key are silently skipped; a failed attempt on a
 * keyed print is the caller's cue to toast.
 */
import type { OutputMetadata } from "./api/types";

export interface SourceRestoreDeps {
  /** Local stash lookup (ipc). Rejections are treated as misses. */
  stashGet(sha256: string): Promise<string | null>;
  /** Cross-host gallery lookup by filename → base64 bytes, or null. */
  galleryLookup(filename: string): Promise<string | null>;
}

export interface RestoredSource {
  base64: string;
  /** Label to carry back into the form (and future requests). */
  filename: string | null;
}

/** Whether a restore is even attemptable — old prints have no keys. */
export function metadataReferencesSource(meta: OutputMetadata): boolean {
  return Boolean(meta.source_image_sha256 || meta.source_image_name);
}

export async function restoreSourceImage(
  meta: OutputMetadata,
  deps: SourceRestoreDeps,
): Promise<RestoredSource | null> {
  const sha = meta.source_image_sha256 ?? null;
  if (sha) {
    const stashed = await deps.stashGet(sha).catch(() => null);
    if (stashed) return { base64: stashed, filename: meta.source_image_name ?? null };
  }
  const name = meta.source_image_name ?? null;
  if (name) {
    const fromGallery = await deps.galleryLookup(name).catch(() => null);
    if (fromGallery) return { base64: fromGallery, filename: name };
  }
  return null;
}

/**
 * SHA-256 (lowercase hex) of base64-encoded bytes, hashing the DECODED
 * payload — identical to the server's hash of the wire `source_image`, so
 * stash keys and `source_image_sha256` always agree.
 */
export async function sha256HexOfBase64(b64: string): Promise<string> {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
