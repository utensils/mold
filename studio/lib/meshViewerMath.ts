/*
 * The arithmetic behind `MeshViewer.vue`'s auto-rotation and wireframe overlay,
 * kept out of the component so it can be tested without a GPU or a DOM. The
 * viewer itself owns only GL state and event wiring.
 */

const TAU = Math.PI * 2;

/** Folds any angle into `[-π, π)`. */
function wrapAngle(angle: number): number {
  if (!Number.isFinite(angle)) return 0;
  return ((((angle + Math.PI) % TAU) + TAU) % TAU) - Math.PI;
}

/**
 * The next yaw for an auto-rotating viewer, `elapsedMs` after the last frame.
 *
 * The result is always wrapped into `[-π, π)`: a gallery left open all day
 * would otherwise accumulate an ever-larger angle whose float precision — and
 * whose rotation matrix — quietly degrade.
 */
export function advanceAutoRotate(
  yaw: number,
  elapsedMs: number,
  radiansPerSecond = 0.25,
): number {
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) return wrapAngle(yaw);
  return wrapAngle(yaw + (radiansPerSecond * elapsedMs) / 1000);
}

/**
 * The deduplicated undirected edge list of a triangle index buffer, ready for
 * `gl.LINES`.
 *
 * Each edge appears exactly once as an ordered `[min, max]` pair, in first-seen
 * order, so a shared edge is drawn once rather than twice and the buffer is
 * stable across calls. Degenerate edges (a vertex joined to itself) and a
 * trailing partial triangle are dropped.
 *
 * One `Set` of packed `low * vertexCount + high` keys and one preallocated
 * output buffer: a two-million-face mesh used to allocate a `Set` per vertex
 * plus a growing array and freeze the tab for seconds on the first wireframe
 * toggle. The packing is exact while `vertexCount²` fits a double's integer
 * range, which every mesh under the viewer's 256 MiB cap does by orders of
 * magnitude.
 */
export function edgeIndices(
  indices: Uint32Array | ArrayLike<number>,
): Uint32Array {
  const triangles = indices.length - (indices.length % 3);
  if (triangles === 0) return new Uint32Array(0);

  let maxIndex = 0;
  for (let i = 0; i < triangles; i += 1) {
    const value = indices[i] ?? 0;
    if (value > maxIndex) maxIndex = value;
  }
  const vertexCount = maxIndex + 1;

  const seen = new Set<number>();
  // Three edges per triangle at most, two indices each.
  const out = new Uint32Array(triangles * 2);
  let count = 0;

  const add = (a: number, b: number): void => {
    if (a === b) return;
    const low = a < b ? a : b;
    const high = a < b ? b : a;
    const key = low * vertexCount + high;
    if (seen.has(key)) return;
    seen.add(key);
    out[count] = low;
    out[count + 1] = high;
    count += 2;
  };

  for (let i = 0; i < triangles; i += 3) {
    const a = indices[i] ?? 0;
    const b = indices[i + 1] ?? 0;
    const c = indices[i + 2] ?? 0;
    add(a, b);
    add(b, c);
    add(a, c);
  }
  return out.slice(0, count);
}

/**
 * Whether `edgeIndices` would emit anything at all: at least one complete
 * triangle joins two distinct vertices. A linear scan with no allocation, so
 * a viewer can decide whether to offer the wireframe toggle without paying
 * for the edge list a person may never ask for.
 */
export function meshHasEdges(
  indices: Uint32Array | ArrayLike<number>,
): boolean {
  const triangles = indices.length - (indices.length % 3);
  for (let i = 0; i < triangles; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    if (a !== b || b !== c) return true;
  }
  return false;
}
