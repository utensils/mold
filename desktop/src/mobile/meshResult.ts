/**
 * What the phone needs to show a finished 3-D print in the Create result bed.
 *
 * Kept out of `MobileApp.vue` because both answers are ordering-sensitive and
 * worth pinning in isolation: a mesh must be recognized BEFORE the audio and
 * video probes (it carries neither samples nor frames, so a wider test running
 * first classifies it as a still and tries to draw glTF bytes into an `<img>`),
 * and its bytes arrive as base64 that has to become a real Blob before a
 * viewer can fetch it.
 */
import { isMeshCompletion } from "@studio/lib/meshCompletion";
import type { CompleteEvent } from "../lib/api/types";

/**
 * The advertised mesh export containers that are ANIMATED rather than a
 * geometry file: they carry the turntable options (playback, repeat, size,
 * frame rate) the existing export sheet already collects, so they route
 * through that sheet instead of exporting on a single tap.
 */
const ANIMATED_MESH_EXPORT_FORMATS: ReadonlySet<string> = new Set(["gif", "apng", "webp"]);

export function isAnimatedMeshExportFormat(format: string): boolean {
  return ANIMATED_MESH_EXPORT_FORMATS.has(format.trim().toLowerCase());
}

/**
 * The filename an exported mesh is shared or downloaded under: the print's own
 * stem with the requested container's extension. The advertised list is the
 * host's, never a client constant, so this deliberately does not validate it.
 */
export function meshExportFilename(filename: string, format: string): string {
  const stem = filename.replace(/\.[^.]+$/, "") || "mold-mesh";
  return `${stem}.${format.trim().toLowerCase()}`;
}

/** The binary glTF media type, as the container's own registration spells it. */
export const GLB_MIME_TYPE = "model/gltf-binary";

/**
 * Whether this completion is a 3-D mesh.
 *
 * The shared probe keys on `mesh_vertices`, which an inline completion
 * carries. A DURABLE completion is synthesized from the byte-free
 * presentation the phone persisted at submit time and has no mesh fields at
 * all — only the container the request asked for — so `glb` is the second,
 * equally authoritative answer here rather than a guess.
 */
export function isMobileMeshResult(result: CompleteEvent | null | undefined): boolean {
  return isMeshCompletion(result) || result?.format === "glb";
}

/**
 * The binary glTF a completion carried inline, as a Blob a viewer can fetch
 * through an object URL. `null` for a metadata-only completion, which
 * published no bytes and whose mesh is streamed from the host's gallery
 * instead.
 */
export function meshResultBlob(base64: string | null | undefined): Blob | null {
  if (!base64) return null;
  let binary: string;
  try {
    binary = atob(base64);
  } catch {
    // A truncated or non-base64 payload is a broken completion, not a crash:
    // the caller falls back to the host's own copy of the saved print.
    return null;
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: GLB_MIME_TYPE });
}
