/*
 * The phone gallery viewer's bottom sheet — the pure parts.
 *
 * The viewer stages media at full screen and keeps every detail and action in
 * a sheet the user swipes up. What the sheet's one-line peek says, and what a
 * drag on it means, are decisions with no DOM in them, so they live here and
 * are tested directly.
 */
import type { GalleryImage } from "../lib/api/types";
import { formatCount } from "../lib/format";

/** The four things the viewer can stage. */
export type ViewerMediaKind = "image" | "video" | "audio" | "mesh";

/** The badge the peek row leads with. */
export function viewerKindLabel(kind: ViewerMediaKind): string {
  switch (kind) {
    case "mesh":
      return "3-D";
    case "video":
      return "Video";
    case "audio":
      return "Audio";
    default:
      return "Image";
  }
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** `m:ss`, the way a phone's own transport reads a clip. */
function mediaDuration(milliseconds: number): string {
  const total = Math.max(0, Math.floor(milliseconds / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/**
 * What the staged media itself reported once it loaded. A gallery row records
 * neither a mesh's triangle count nor a clip's running time, so the peek asks
 * the viewer that is already showing it and falls back to the row otherwise.
 */
export interface ViewerMeasurement {
  mesh?: { vertexCount: number; triangleCount: number } | null;
  durationMs?: number | null;
}

/**
 * The one fact the collapsed peek shows: what THIS kind is measured in — a
 * mesh in triangles, a clip or a score in running time, a still in pixels.
 * A print nothing has measured yet falls back to its container, which every
 * print has.
 */
export function viewerPeekSummary(
  kind: ViewerMediaKind,
  item: GalleryImage,
  measured: ViewerMeasurement = {},
): string {
  const metadata = item.metadata;
  if (kind === "mesh") {
    const stats = measured.mesh;
    if (stats) {
      return `${formatCount(stats.triangleCount)} tris · ${formatCount(stats.vertexCount)} verts`;
    }
  } else if (kind === "video" || kind === "audio") {
    if (finite(measured.durationMs) && measured.durationMs > 0) {
      return mediaDuration(measured.durationMs);
    }
    const frames = metadata.frames ?? metadata.video_frames ?? null;
    const fps = metadata.fps ?? metadata.video_fps ?? null;
    if (finite(frames) && finite(fps) && fps > 0) return mediaDuration((frames / fps) * 1000);
    if (finite(frames)) return `${frames} frames`;
  } else if (finite(metadata.width) && finite(metadata.height)) {
    return `${metadata.width}×${metadata.height}`;
  }
  return (item.format ?? metadata.output_format ?? "").toUpperCase();
}

/** What a finished drag on the sheet means. */
export type SheetGesture = "expand" | "collapse" | "none";

/** A drag has to travel this far, and mean it, before the sheet moves. */
const SHEET_SWIPE_DISTANCE = 40;

/**
 * A drag only moves the sheet when it is decisively vertical and points away
 * from where the sheet already is. A body scrolled past its top owns its own
 * downward drag — that is the list scrolling back, not a dismissal.
 */
export function resolveSheetGesture(drag: {
  deltaX: number;
  deltaY: number;
  expanded: boolean;
  scrolled: boolean;
}): SheetGesture {
  const { deltaX, deltaY, expanded, scrolled } = drag;
  if (Math.abs(deltaY) < SHEET_SWIPE_DISTANCE) return "none";
  if (Math.abs(deltaY) <= Math.abs(deltaX)) return "none";
  if (deltaY < 0) return expanded ? "none" : "expand";
  if (!expanded || scrolled) return "none";
  return "collapse";
}

/** One entry of the mesh export picker. */
export interface MeshExportChoice {
  value: string;
  label: string;
}

/**
 * The export picker's options: every geometry container the host advertises,
 * then ONE turntable entry standing for the animated containers, which carry
 * playback options and so go through the export sheet the phone already has.
 * Eight stacked buttons became one picker and two verbs.
 */
export function meshExportChoices(
  geometry: readonly string[],
  animated: readonly string[],
): MeshExportChoice[] {
  const choices = geometry.map((format) => ({ value: format, label: format.toUpperCase() }));
  if (animated.length) choices.push({ value: "turntable", label: "Turntable" });
  return choices;
}
