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
 */
export function edgeIndices(
  indices: Uint32Array | ArrayLike<number>,
): Uint32Array {
  const seen = new Map<number, Set<number>>();
  const out: number[] = [];
  const triangles = indices.length - (indices.length % 3);

  const add = (a: number | undefined, b: number | undefined): void => {
    if (a === undefined || b === undefined || a === b) return;
    const low = a < b ? a : b;
    const high = a < b ? b : a;
    let partners = seen.get(low);
    if (!partners) {
      partners = new Set<number>();
      seen.set(low, partners);
    }
    if (partners.has(high)) return;
    partners.add(high);
    out.push(low, high);
  };

  for (let i = 0; i < triangles; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    add(a, b);
    add(b, c);
    add(a, c);
  }
  return Uint32Array.from(out);
}
