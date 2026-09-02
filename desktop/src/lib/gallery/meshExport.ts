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
 * The one split this module does own is which advertised containers are
 * ANIMATED. Those share the video export sheet's playback options (playback,
 * repeat, max dimension, fps), so they collapse into a single entry that
 * opens it; everything else is a one-click transcode with no options at all.
 */
import type { VideoExportFormat } from "@studio/lib/videoExport";

const ANIMATED_MESH_EXPORTS: ReadonlySet<string> = new Set(["gif", "apng", "webp"]);

/** Direct one-click transcodes: one menu entry each, in the host's order. */
export function meshFileExportFormats(advertised: readonly string[] | null | undefined): string[] {
  return (advertised ?? []).filter((format) => !ANIMATED_MESH_EXPORTS.has(format));
}

/** Animated turntables, which share the export sheet's playback options. */
export function meshAnimationExportFormats(
  advertised: readonly string[] | null | undefined,
): VideoExportFormat[] {
  return (advertised ?? []).filter((format): format is VideoExportFormat =>
    ANIMATED_MESH_EXPORTS.has(format),
  );
}

/**
 * The saved name for a transcode: the print's own suggested save name with
 * its container swapped. The gallery filename never changes — this only names
 * the copy that lands in Downloads.
 */
export function meshExportFilename(saveName: string, format: string): string {
  const stem = saveName.replace(/\.[^.]+$/, "") || "mold-mesh";
  return `${stem}.${format}`;
}
