import { describe, expect, it } from "vitest";
import { advanceAutoRotate, edgeIndices } from "./meshViewerMath";

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

  it("hands back an empty buffer for an empty index list", () => {
    const edges = edgeIndices(new Uint32Array(0));
    expect(edges).toBeInstanceOf(Uint32Array);
    expect(edges).toHaveLength(0);
  });
});
