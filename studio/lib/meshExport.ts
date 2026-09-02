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

// ── Geometry options ───────────────────────────────────────────────────────
// A stored mesh is normalized model space (a unit cube, Y-up). A slicer that
// reads those coordinates as millimetres sees a 2 mm blob lying on its side,
// so OBJ / STL / PLY exports carry three optional knobs the host applies
// before it writes the file: a longest-side size in millimetres, the up axis,
// and where the origin sits. GLB (the stored form) and the animated
// turntables never take them.

/** Which axis the exported file calls up. */
export type MeshUpAxis = "y" | "z";
/** Where the exported file puts the origin relative to the mesh. */
export type MeshExportOrigin = "center" | "floor";

/** One resolved set of geometry knobs, as a client holds it in a form.
 * `size_mm: null` means "as stored" — model units, no scaling at all. */
export interface MeshGeometryOptions {
  size_mm: number | null;
  up_axis: MeshUpAxis;
  origin: MeshExportOrigin;
}

/**
 * `/api/capabilities.mesh.export_geometry` — the host's own bounds, the axes
 * and origins it accepts, and its per-format defaults. ABSENT (or null) on a
 * host that predates the feature, which is the ONLY gate: a client that
 * cannot see this block must post `{ format }` exactly as before, because an
 * older server drops unknown fields rather than refusing them and would
 * silently write the unscaled mesh the user thought they had resized.
 */
export interface MeshExportGeometryCapabilities {
  size_mm: { min: number; max: number; default: number };
  up_axes: MeshUpAxis[];
  origins: MeshExportOrigin[];
  /** Keyed by lower-case container; a format absent here takes no options. */
  defaults: Record<string, MeshGeometryOptions>;
}

/** The bounding box `MeshViewer` reports on `ready`, in model units. */
export interface MeshBounds {
  min: number[];
  max: number[];
}

/**
 * Whether a container is one geometry options could apply to at all. This is
 * the structural rule — the stored form and the turntables are excluded — and
 * it is deliberately permissive about containers this client has never heard
 * of. The host's own `defaults` table is the authority on what it will
 * actually accept, so `meshGeometryDefaults` is what a caller gates on.
 */
export function takesGeometryOptions(format: string): boolean {
  const value = normalise(format);
  return value !== STORED_MESH_FORMAT && !ANIMATED_MESH_EXPORTS.has(value);
}

function clampSize(
  value: number,
  bounds: MeshExportGeometryCapabilities["size_mm"],
): number {
  return Math.min(bounds.max, Math.max(bounds.min, value));
}

/**
 * The host's defaults for one container, or `null` meaning DO NOT OFFER the
 * options at all — a host that predates the block (`capabilities` absent or
 * null), a container it does not scale (glb, the turntables), or one it
 * simply does not list. A caller that gets `null` posts the bare `{ format }`.
 */
export function meshGeometryDefaults(
  capabilities: MeshExportGeometryCapabilities | null | undefined,
  format: string,
): MeshGeometryOptions | null {
  if (!capabilities) return null;
  if (!takesGeometryOptions(format)) return null;
  const entry = capabilities.defaults?.[normalise(format)];
  if (!entry) return null;
  const axes = capabilities.up_axes ?? [];
  const origins = capabilities.origins ?? [];
  return {
    size_mm:
      typeof entry.size_mm === "number" && Number.isFinite(entry.size_mm)
        ? clampSize(entry.size_mm, capabilities.size_mm)
        : null,
    up_axis: axes.includes(entry.up_axis) ? entry.up_axis : (axes[0] ?? "y"),
    origin: origins.includes(entry.origin)
      ? entry.origin
      : (origins[0] ?? "floor"),
  };
}

/**
 * The extents the exported file will have, along ITS OWN X, Y and Z axes.
 *
 * The stored mesh is Y-up, so a Z-up export rotates `(x, y, z) -> (x, -z, y)`
 * and its axes read width × depth × height; a Y-up export keeps the stored
 * order. Scaling is uniform: the longest stored extent becomes `sizeMm`, and
 * a null `sizeMm` means model units are written verbatim. Returns `null` when
 * the viewer has not reported a box yet or the box is degenerate — nothing
 * about the export changes, only what can be said about it.
 */
export function meshExportDimensionsMm(
  bounds: MeshBounds | null | undefined,
  sizeMm: number | null,
  upAxis: MeshUpAxis,
): [number, number, number] | null {
  const min = bounds?.min;
  const max = bounds?.max;
  if (!min || !max || min.length < 3 || max.length < 3) return null;
  const extents = [0, 1, 2].map((axis) => (max[axis] ?? 0) - (min[axis] ?? 0));
  if (extents.some((extent) => !Number.isFinite(extent) || extent < 0))
    return null;
  const longest = Math.max(...extents);
  if (!(longest > 0)) return null;
  const scale = sizeMm == null ? 1 : sizeMm / longest;
  const [x, y, z] = extents as [number, number, number];
  const ordered: [number, number, number] =
    upAxis === "z" ? [x, z, y] : [x, y, z];
  return ordered.map((extent) => extent * scale) as [number, number, number];
}

function trimNumber(value: number, decimals: number): string {
  return value.toFixed(decimals).replace(/\.?0+$/, "") || "0";
}

/**
 * The one live sentence under the size control: what the exported file will
 * actually measure. With a box from the viewer it names all three extents;
 * without one it can still name the knob itself.
 */
export function meshExportSizeLabel(
  bounds: MeshBounds | null | undefined,
  options: MeshGeometryOptions,
): string {
  const dimensions = meshExportDimensionsMm(
    bounds,
    options.size_mm,
    options.up_axis,
  );
  if (options.size_mm == null) {
    if (!dimensions) return "as stored";
    return `as stored (${dimensions
      .map((extent) => extent.toFixed(2))
      .join(" × ")})`;
  }
  if (!dimensions) return `longest side ${trimNumber(options.size_mm, 1)} mm`;
  return `${dimensions.map((extent) => extent.toFixed(1)).join(" × ")} mm`;
}

/**
 * The body posted to `POST /api/gallery/export/:filename`. With no geometry
 * (an older host, the stored container, a turntable) it is the bare format
 * this client has always sent. `size_mm` is OMITTED for "as stored", because
 * the wire has no way to ask a format whose default is a size to skip
 * scaling — the choice is only ever offered where the default is already null.
 */
export function meshExportRequest(
  format: string,
  geometry: MeshGeometryOptions | null,
): { format: string } & Partial<MeshGeometryOptions> {
  const body: { format: string } & Partial<MeshGeometryOptions> = {
    format: normalise(format),
  };
  if (!geometry) return body;
  if (geometry.size_mm != null) body.size_mm = geometry.size_mm;
  body.up_axis = geometry.up_axis;
  body.origin = geometry.origin;
  return body;
}
