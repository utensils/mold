/**
 * The 3-D mesh controls every Create surface renders from the recipe's
 * advertised `capabilities.mesh` block, and the request/metadata round trip
 * they share. Web, desktop and iPhone all hold a `MeshFormState`, serialize
 * it with `meshRequestFromForm` and restore it with `meshFormFromMetadata`,
 * so the three surfaces cannot disagree about what "untouched" means: a
 * value equal to the profile default is omitted from the wire and the print
 * records the default the engine actually applied.
 */
import type { MeshCapabilitiesProfile } from "./generated/generationProfileV1";

/**
 * `GenerateRequest.mesh` exactly as `mold-core`'s `MeshRequestOptions`
 * serializes; also the shape of `OutputMetadata.mesh`, which records the
 * RESOLVED values (request or engine default) for a mesh print.
 */
export interface MeshRequestOptions {
  octree_resolution?: number | null;
  threshold?: number | null;
  target_faces?: number | null;
  texture?: boolean | null;
  texture_resolution?: number | null;
}

/** Form state: `null` means "use the profile default". */
export interface MeshFormState {
  octreeResolution: number | null;
  threshold: number | null;
  targetFaces: number | null;
  /** Additive fields: older persisted drafts omit them. */
  texture?: boolean | null;
  textureResolution?: number | null;
}

export function emptyMeshForm(): MeshFormState {
  return { octreeResolution: null, threshold: null, targetFaces: null };
}

function explicit(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null;
}

/**
 * The request block for this form, or `undefined` when nothing differs from
 * the advertised defaults — the server applies the same defaults itself, so
 * an empty block would only be noise the print records anyway.
 */
export function meshRequestFromForm(
  form: MeshFormState,
  caps:
    | Pick<
        MeshCapabilitiesProfile,
        | "octree_default"
        | "threshold"
        | "texture"
        | "texture_resolutions"
        | "texture_default_resolution"
      >
    | null
    | undefined,
): MeshRequestOptions | undefined {
  const request: MeshRequestOptions = {};
  const octree = explicit(form.octreeResolution);
  if (octree !== null && octree !== caps?.octree_default) {
    request.octree_resolution = octree;
  }
  const threshold = explicit(form.threshold);
  if (threshold !== null && threshold !== caps?.threshold.default) {
    request.threshold = threshold;
  }
  const faces = explicit(form.targetFaces);
  if (faces !== null) request.target_faces = faces;
  if (form.texture === true && caps?.texture.mode !== "hidden") {
    request.texture = true;
    const selected = explicit(form.textureResolution);
    const fallback = explicit(caps?.texture_default_resolution);
    const resolution = selected ?? fallback;
    if (
      resolution !== null &&
      (!caps?.texture_resolutions?.length ||
        caps.texture_resolutions.includes(resolution))
    ) {
      request.texture_resolution = resolution;
    }
  }
  return Object.keys(request).length > 0 ? request : undefined;
}

/** Restore a form from a print's recorded `metadata.mesh`. */
export function meshFormFromMetadata(
  mesh: MeshRequestOptions | null | undefined,
): MeshFormState {
  const texture = mesh?.texture === true;
  const textureResolution = explicit(mesh?.texture_resolution);
  return {
    octreeResolution: explicit(mesh?.octree_resolution),
    threshold: explicit(mesh?.threshold),
    targetFaces: explicit(mesh?.target_faces),
    ...(texture ? { texture: true } : {}),
    ...(textureResolution !== null ? { textureResolution } : {}),
  };
}

export type MeshBounds = {
  min: readonly [number, number, number] | readonly number[];
  max: readonly [number, number, number] | readonly number[];
};

/**
 * `49,152 tris · 24,576 verts · 1.00×0.80×0.60` — the one caption every
 * surface writes under a mesh. Bounds are the wire's `[x, y, z]` pair, given
 * either as one object or as the separate min/max arrays the complete event
 * carries; the size shown is the axis-aligned extent.
 */
export function meshStatsLabel(
  vertices: number | null | undefined,
  faces: number | null | undefined,
  bounds: MeshBounds | readonly number[] | null | undefined,
  boundsMax?: readonly number[] | null,
): string {
  const parts: string[] = [];
  const count = (n: number) => n.toLocaleString("en-US");
  if (typeof faces === "number" && Number.isFinite(faces)) {
    parts.push(`${count(faces)} tris`);
  }
  if (typeof vertices === "number" && Number.isFinite(vertices)) {
    parts.push(`${count(vertices)} verts`);
  }
  const resolved = resolveBounds(bounds, boundsMax);
  if (resolved) {
    const extent = [0, 1, 2].map((axis) =>
      Math.abs((resolved.max[axis] ?? 0) - (resolved.min[axis] ?? 0)).toFixed(
        2,
      ),
    );
    parts.push(extent.join("×"));
  }
  return parts.join(" · ");
}

function resolveBounds(
  bounds: MeshBounds | readonly number[] | null | undefined,
  boundsMax?: readonly number[] | null,
): { min: readonly number[]; max: readonly number[] } | null {
  if (!bounds) return null;
  if (Array.isArray(bounds)) {
    return boundsMax && bounds.length >= 3 && boundsMax.length >= 3
      ? { min: bounds, max: boundsMax }
      : null;
  }
  const object = bounds as MeshBounds;
  return object.min?.length >= 3 && object.max?.length >= 3
    ? { min: object.min, max: object.max }
    : null;
}
