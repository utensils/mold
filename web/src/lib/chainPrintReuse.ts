/**
 * Reusing a print that was rendered as several stitched clips.
 *
 * The web surface authors ONE SHOT prints only, but prints stitched from
 * clips still exist in every library — authored sequences from older builds,
 * and the auto-chained long videos the duration slider still produces today.
 * Their saved `metadata.prompt` is every clip's prompt newline-joined, which
 * is a description of the render and not a prompt anybody can re-submit: put
 * it back in the composer and the next generation asks for all of them at
 * once. `metadata.chain.stages` is the per-clip provenance the server
 * recorded, so the first clip's own prompt is the honest one-shot restore.
 */
import type { OutputMetadata } from "../types";

/**
 * The prompt to restore into the one-shot composer for `metadata`.
 *
 * A print with no chain provenance (every still, every single-pass video)
 * keeps its own prompt untouched — this is a narrowing, never a rewrite.
 * `applyMetadataToForm` is the ONE caller, so every reuse door on the web
 * surface (Library, the Lightbox, Create's Recent tiles, a recovered queue
 * selection) narrows identically.
 */
export function oneShotPromptForPrint(metadata: OutputMetadata): string {
  const firstClip = metadata.chain?.stages?.[0]?.prompt;
  return typeof firstClip === "string" && firstClip.trim() !== ""
    ? firstClip
    : (metadata.prompt ?? "");
}
