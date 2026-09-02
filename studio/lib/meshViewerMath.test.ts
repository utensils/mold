import { describe, expect, it } from "vitest";
import { advanceAutoRotate, edgeIndices, meshHasEdges } from "./meshViewerMath";

describe("advanceAutoRotate", () => {
  it("advances the yaw by the rate times the elapsed seconds", () => {
    expect(advanceAutoRotate(0, 1000)).toBeCloseTo(0.25, 10);
    expect(advanceAutoRotate(0, 500)).toBeCloseTo(0.125, 10);
    expect(advanceAutoRotate(0.1, 2000, 0.5)).toBeCloseTo(1.1, 10);
  });

  it("wraps into [-pi, pi) so a long rotation never grows unbounded", () => {
    const wrapped = advanceAutoRotate(Math.PI - 0.1, 1000);
    expect(wrapped).toBeCloseTo(-Math.PI + 0.15, 10);

    let yaw = 0;
    for (let i = 0; i < 10_000; i += 1) yaw = advanceAutoRotate(yaw, 100);
    expect(yaw).toBeGreaterThanOrEqual(-Math.PI);
    expect(yaw).toBeLessThan(Math.PI);
  });

  it("wraps a yaw the caller already pushed out of range", () => {
    expect(advanceAutoRotate(Math.PI * 4, 0)).toBeCloseTo(0, 10);
    expect(advanceAutoRotate(-Math.PI * 3, 0)).toBeCloseTo(-Math.PI, 10);
  });

  it("stands still for a non-positive or non-finite elapsed time", () => {
    expect(advanceAutoRotate(0.4, 0)).toBeCloseTo(0.4, 10);
    expect(advanceAutoRotate(0.4, -16)).toBeCloseTo(0.4, 10);
    expect(advanceAutoRotate(0.4, Number.NaN)).toBeCloseTo(0.4, 10);
  });
});

describe("edgeIndices", () => {
  it("gives one triangle its three edges", () => {
    expect(Array.from(edgeIndices(Uint32Array.from([0, 1, 2])))).toEqual([
      0, 1, 1, 2, 0, 2,
    ]);
  });

  it("emits a shared edge once, so two triangles give five edges", () => {
    const edges = edgeIndices(Uint32Array.from([0, 1, 2, 2, 1, 3]));
    expect(edges).toHaveLength(10);
    expect(Array.from(edges)).toEqual([0, 1, 1, 2, 0, 2, 1, 3, 2, 3]);
  });

  it("orders every edge low-to-high and keeps first-seen order", () => {
    const edges = Array.from(edgeIndices([7, 3, 5]));
    expect(edges).toEqual([3, 7, 3, 5, 5, 7]);
  });

  it("skips degenerate edges and a trailing partial triangle", () => {
    expect(Array.from(edgeIndices([0, 0, 1]))).toEqual([0, 1]);
    expect(Array.from(edgeIndices([4, 4, 4]))).toHaveLength(0);
    expect(Array.from(edgeIndices([0, 1, 2, 3, 4]))).toEqual([
      0, 1, 1, 2, 0, 2,
    ]);
  });

  /**
   * The readable Map-of-Sets version this replaced, kept as the oracle: the
   * production path packs each edge into one number so a two-million-face
   * mesh no longer allocates a Set per vertex, but the edge list it emits
   * must be identical, edge for edge, in first-seen order.
   */
  function referenceEdgeIndices(indices: ArrayLike<number>): number[] {
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
      add(indices[i], indices[i + 1]);
      add(indices[i + 1], indices[i + 2]);
      add(indices[i], indices[i + 2]);
    }
    return out;
  }

  it("matches the reference edge list on a large shared-edge grid", () => {
    // A 300×300 vertex grid, two triangles per cell: 179,400 triangles whose
    // interior edges are each shared by two of them.
    const side = 300;
    const cells = side - 1;
    const indices = new Uint32Array(cells * cells * 6);
    let cursor = 0;
    for (let row = 0; row < cells; row += 1) {
      for (let column = 0; column < cells; column += 1) {
        const a = row * side + column;
        const b = a + 1;
        const c = a + side;
        const d = c + 1;
        indices.set([a, b, c, b, d, c], cursor);
        cursor += 6;
      }
    }
    const edges = edgeIndices(indices);
    expect(edges).toBeInstanceOf(Uint32Array);
    expect(Array.from(edges)).toEqual(referenceEdgeIndices(indices));
    // Euler: a grid has 2·c·(c+1) axis edges plus one diagonal per cell.
    expect(edges.length / 2).toBe(2 * cells * (cells + 1) + cells * cells);
  });

  it("matches the reference on unordered indices that share edges both ways", () => {
    const indices: number[] = [];
    let seed = 7;
    const next = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed % 5000;
    };
    for (let i = 0; i < 60_000; i += 1) indices.push(next());
    expect(Array.from(edgeIndices(indices))).toEqual(
      referenceEdgeIndices(indices),
    );
  });

  it("keeps a vertex index above 2^16 exact when packing an edge", () => {
    const indices = [0, 70_000, 2_000_000, 2_000_000, 70_000, 1];
    expect(Array.from(edgeIndices(indices))).toEqual([
      0, 70_000, 70_000, 2_000_000, 0, 2_000_000, 1, 70_000, 1, 2_000_000,
    ]);
  });

  it("hands back an empty buffer for an empty index list", () => {
    const edges = edgeIndices(new Uint32Array(0));
    expect(edges).toBeInstanceOf(Uint32Array);
    expect(edges).toHaveLength(0);
  });
});

describe("meshHasEdges", () => {
  it("is true as soon as one complete triangle joins two distinct vertices", () => {
    expect(meshHasEdges([0, 1, 2])).toBe(true);
    expect(meshHasEdges([4, 4, 4, 4, 4, 5])).toBe(true);
    expect(meshHasEdges(Uint32Array.from([7, 7, 3]))).toBe(true);
  });

  it("is false for no triangles, degenerate triangles, or a partial one", () => {
    expect(meshHasEdges([])).toBe(false);
    expect(meshHasEdges([4, 4, 4])).toBe(false);
    expect(meshHasEdges([0, 1])).toBe(false);
    expect(meshHasEdges([2, 2, 2, 0, 1])).toBe(false);
  });

  it("agrees with edgeIndices on whether anything would be drawn", () => {
    for (const indices of [[0, 1, 2], [3, 3, 3], [], [1, 1, 1, 2, 2, 2, 5]]) {
      expect(meshHasEdges(indices)).toBe(edgeIndices(indices).length > 0);
    }
  });
});
