/**
 * 3-D export menu policy, shared by web, desktop and the phone.
 *
 * GLB is the only stored form; OBJ / STL / PLY — and the animated turntables a
 * host renders — are produced on request through
 * `POST /api/gallery/export/:filename`. The menu is therefore built from the
 * holding host's own `/api/capabilities.mesh.export_formats` and never from a
 * client constant: a host that adds a container adds a menu entry with no
 * client release.
 *
 * What this module DOES own is the split of that advertised list. Animated
 * containers share the video export sheet's playback options (playback,
 * repeat, max dimension, fps), so they collapse into a single entry that opens
 * it; everything else is a one-click transcode with no options at all. The
 * stored container itself is dropped: the server lists it first so a client
 * can see what it holds, but "Export as GLB" beside Download is not an export.
 */
import type { VideoExportFormat } from "./videoExport";

const ANIMATED_MESH_EXPORTS: ReadonlySet<string> = new Set([
  "gif",
  "apng",
  "webp",
]);

/** The stored container, which no export menu offers. */
const STORED_MESH_FORMAT = "glb";

/** The binary glTF media type, as the container's own registration spells it. */
export const GLB_MIME_TYPE = "model/gltf-binary";

function normalise(format: string): string {
  return format.trim().toLowerCase();
}

/** Whether an advertised export container is an animated turntable. */
export function isAnimatedMeshExport(format: string): boolean {
  return ANIMATED_MESH_EXPORTS.has(normalise(format));
}

export interface MeshExportSplit {
  /** Direct one-click transcodes: one menu entry each, in the host's order. */
  files: string[];
  /** Animated turntables, which share the export sheet's playback options. */
  animations: VideoExportFormat[];
}

/**
 * The host's advertised list, lower-cased, minus the stored container, split
 * into the two kinds of menu entry.
 */
export function splitMeshExportFormats(
  advertised: readonly string[] | null | undefined,
): MeshExportSplit {
  const files: string[] = [];
  const animations: VideoExportFormat[] = [];
  for (const raw of advertised ?? []) {
    const format = normalise(raw);
    if (format === STORED_MESH_FORMAT) continue;
    if (ANIMATED_MESH_EXPORTS.has(format)) {
      animations.push(format as VideoExportFormat);
    } else {
      files.push(format);
    }
  }
  return { files, animations };
}

/**
 * The name an exported mesh is saved or shared under: the print's own stem
 * with the requested container's extension. The gallery filename never
 * changes — this only names the copy that leaves the app. The advertised list
 * is the host's, so this deliberately does not validate the format.
 */
export function meshExportFilename(filename: string, format: string): string {
  const stem = filename.replace(/\.[^.]+$/, "") || "mold-mesh";
  return `${stem}.${normalise(format)}`;
}
