/**
 * 3-D export menu policy.
 *
 * GLB is the only stored form; OBJ / STL / PLY — and any animated turntable
 * container a future host learns to render — are TRANSCODES the host performs
 * on request through `POST /api/gallery/export/:filename`. So the menu is
 * built from the holding host's own `/api/capabilities.mesh.export_formats`
 * and never from a client constant: a host that adds a container adds a menu
 * entry with no client release.
 *
 * The split itself — which advertised containers are ANIMATED turntables that
 * share the export sheet's options, and that the stored glb is never offered —
 * is the shared studio policy in `@studio/lib/meshExport`; this module only
 * keeps the desktop's names for it.
 */
import type { VideoExportFormat } from "@studio/lib/videoExport";
import {
  meshExportFilename as sharedMeshExportFilename,
  splitMeshExportFormats,
} from "@studio/lib/meshExport";

/** Direct one-click transcodes: one menu entry each, in the host's order. */
export function meshFileExportFormats(advertised: readonly string[] | null | undefined): string[] {
  return splitMeshExportFormats(advertised).files;
}

/** Animated turntables, which share the export sheet's playback options. */
export function meshAnimationExportFormats(
  advertised: readonly string[] | null | undefined,
): VideoExportFormat[] {
  return splitMeshExportFormats(advertised).animations;
}

/**
 * The saved name for a transcode: the print's own suggested save name with
 * its container swapped. The gallery filename never changes — this only names
 * the copy that lands in Downloads.
 */
export const meshExportFilename = sharedMeshExportFilename;
