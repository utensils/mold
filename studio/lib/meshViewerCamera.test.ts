import { describe, expect, it } from "vitest";
import { parseGlb } from "./glb";
import { triangleGlb } from "./glbFixture";
import { advanceAutoRotate } from "./meshViewerMath";
import {
  azimuthDegOfYaw,
  homeCamera,
  identity,
  multiply,
  orthographic,
  orthographicScale,
  POSTER_AZIMUTH_DEG,
  POSTER_ELEVATION_DEG,
  POSTER_MARGIN,
  rotationX,
  rotationY,
  sweepExtent,
  translation,
  TURNTABLE_AZIMUTH_STEP_SIGN,
  upper3x3,
  type Mat4,
} from "./meshViewerCamera";

/** Column-major `m · v`, the multiplication a vertex shader performs. */
function apply(m: Mat4, v: [number, number, number, number]): number[] {
  const out = [0, 0, 0, 0];
  for (let row = 0; row < 4; row += 1) {
    let sum = 0;
    for (let k = 0; k < 4; k += 1) sum += (m[k * 4 + row] ?? 0) * (v[k] ?? 0);
    out[row] = sum;
  }
  return out;
}

/** The drag-to-yaw gain in `MeshViewer.vue`'s `onPointerMove`. */
const DRAG_RADIANS_PER_PIXEL = 0.008;

describe("homeCamera", () => {
  it("opens on the server's poster camera, exactly", () => {
    const home = homeCamera();
    expect(home.yaw).toBeCloseTo(-0.5235987755982988, 12);
    expect(home.pitch).toBeCloseTo(0.3490658503988659, 12);
    expect(home.zoom).toBe(1);
    // The poster's own numbers, not a second set that happens to agree.
    expect(home.yaw).toBeCloseTo((-POSTER_AZIMUTH_DEG * Math.PI) / 180, 15);
    expect(home.pitch).toBeCloseTo((POSTER_ELEVATION_DEG * Math.PI) / 180, 15);
  });

  it("hands back a fresh object so a viewer cannot mutate the home view", () => {
    const first = homeCamera();
    first.yaw = 1.5;
    expect(homeCamera().yaw).toBeCloseTo(-0.5235987755982988, 12);
  });
});

describe("azimuthDegOfYaw", () => {
  it("reads the home yaw back as the poster's azimuth", () => {
    expect(azimuthDegOfYaw(homeCamera().yaw)).toBeCloseTo(
      POSTER_AZIMUTH_DEG,
      10,
    );
  });

  it("is the exact inverse of the yaw conversion", () => {
    for (const azimuth of [-180, -34.4, 0, 30, 97.5, 180]) {
      expect(azimuthDegOfYaw((-azimuth * Math.PI) / 180)).toBeCloseTo(
        azimuth,
        10,
      );
    }
  });
});

describe("the turntable turns the way a rightward drag does", () => {
  it("lowers the azimuth for a rightward drag, and steps the sweep the same way", () => {
    const home = homeCamera();
    // +50 px to the right: `orbit(dx * 0.008, …)`.
    const dragged = home.yaw + 50 * DRAG_RADIANS_PER_PIXEL;
    expect(azimuthDegOfYaw(dragged)).toBeLessThan(azimuthDegOfYaw(home.yaw));
    // The server's sweep must step the azimuth the same direction, or the GIF
    // would spin opposite to the drag and to the auto-rotate tour.
    expect(TURNTABLE_AZIMUTH_STEP_SIGN).toBeLessThan(0);
  });

  it("agrees with advanceAutoRotate, which also raises the yaw", () => {
    const home = homeCamera();
    const toured = advanceAutoRotate(home.yaw, 1000);
    expect(toured).toBeGreaterThan(home.yaw);
    expect(azimuthDegOfYaw(toured)).toBeLessThan(azimuthDegOfYaw(home.yaw));
  });
});

describe("sweepExtent", () => {
  const elevation = (POSTER_ELEVATION_DEG * Math.PI) / 180;

  it("frames a unit box from every azimuth at the poster elevation", () => {
    // Every corner of [-1, 1]³ is radial √2 out and 1 above or below the
    // centre, so the bound is `cos e · 1 + sin e · √2` — larger than the
    // radial term, which is what makes the elevated view the binding one.
    const positions: number[] = [];
    for (const x of [-1, 1])
      for (const y of [-1, 1]) for (const z of [-1, 1]) positions.push(x, y, z);
    const extent = sweepExtent(positions, [0, 0, 0], elevation);
    expect(extent).toBeCloseTo(
      Math.cos(elevation) + Math.sin(elevation) * Math.SQRT2,
      12,
    );
    expect(extent).toBeCloseTo(1.4233823, 6);
    expect(extent).toBeGreaterThan(Math.SQRT2);
  });

  it("is the radial distance when the mesh is flat in the XZ plane", () => {
    // No height at all: `cos e · 0 + sin e · r` is smaller than `r`, so the
    // silhouette width wins and the extent is exactly the radius.
    const positions = [3, 0, 0, 0, 0, -3, -3, 0, 0];
    expect(sweepExtent(positions, [0, 0, 0], elevation)).toBeCloseTo(3, 12);
  });

  it("frames the parsed one-triangle fixture about its bounding-box centre", () => {
    const mesh = parseGlb(triangleGlb());
    const { min, max } = mesh.bounds;
    const center: [number, number, number] = [
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    ];
    expect(center).toEqual([1, 2, -0.5]);
    // Every vertex is radial √1.25 from the centre and 2 above or below it.
    const radial = Math.sqrt(1.25);
    expect(sweepExtent(mesh.positions, center, elevation)).toBeCloseTo(
      Math.cos(elevation) * 2 + Math.sin(elevation) * radial,
      6,
    );
    expect(sweepExtent(mesh.positions, center, elevation)).toBeCloseTo(
      2.2617754,
      6,
    );
  });

  it("skips a non-finite vertex rather than letting it poison the maximum", () => {
    const positions = [1, 0, 0, Number.NaN, 0, 0, 0, 0, Infinity, 2, 0, 0];
    expect(sweepExtent(positions, [0, 0, 0], elevation)).toBeCloseTo(2, 12);
  });

  it("is zero when there is nothing finite to frame", () => {
    expect(sweepExtent([], [0, 0, 0], elevation)).toBe(0);
    expect(sweepExtent([Number.NaN, 0, 0], [0, 0, 0], elevation)).toBe(0);
    // A trailing partial vertex is dropped, not read as zeros.
    expect(sweepExtent([0, 0], [0, 0, 0], elevation)).toBe(0);
  });

  it("reads a Float32Array the way it reads a plain array", () => {
    const values = [1, 2, 3, -4, 5, -6];
    expect(sweepExtent(Float32Array.from(values), [0, 0, 0], elevation)).toBe(
      sweepExtent(values, [0, 0, 0], elevation),
    );
  });
});

describe("orthographicScale", () => {
  it("matches the Rust fit_scale arithmetic on a 64×48 frame", () => {
    // `min(half_w, half_h) / extent * (1 - margin)` — the short axis binds.
    for (const extent of [0.5, 1, 2.2617754, 17]) {
      expect(orthographicScale(extent, 64, 48, POSTER_MARGIN)).toBeCloseTo(
        (24 / extent) * 0.92,
        10,
      );
    }
    expect(orthographicScale(2, 48, 64, POSTER_MARGIN)).toBeCloseTo(
      (24 / 2) * 0.92,
      10,
    );
  });

  it("leaves the margin's share of the short axis empty", () => {
    // The mesh's extent lands at `1 - margin` of the half-frame, which is what
    // makes the viewer's home frame the poster's frame.
    const scale = orthographicScale(3, 200, 200, POSTER_MARGIN);
    expect(3 * scale).toBeCloseTo(100 * (1 - POSTER_MARGIN), 10);
  });

  it("clamps the margin the way the rasterizer does", () => {
    expect(orthographicScale(1, 100, 100, -1)).toBeCloseTo(50, 10);
    expect(orthographicScale(1, 100, 100, 0.95)).toBeCloseTo(5, 10);
    expect(orthographicScale(1, 100, 100, Number.NaN)).toBeCloseTo(50, 10);
  });

  it("is zero rather than Infinity for a mesh with no extent", () => {
    expect(orthographicScale(0, 64, 48, POSTER_MARGIN)).toBe(0);
    expect(orthographicScale(-1, 64, 48, POSTER_MARGIN)).toBe(0);
    expect(orthographicScale(Number.NaN, 64, 48, POSTER_MARGIN)).toBe(0);
    expect(orthographicScale(Infinity, 64, 48, POSTER_MARGIN)).toBe(0);
    expect(orthographicScale(2, 0, 0, POSTER_MARGIN)).toBe(0);
  });
});

describe("orthographic", () => {
  it("maps the half-extents to the edges of the NDC cube", () => {
    const m = orthographic(4, 3, 1, 9);
    const corner = apply(m, [4, 3, -5, 1]);
    expect(corner[0]).toBeCloseTo(1, 6);
    expect(corner[1]).toBeCloseTo(1, 6);
    const opposite = apply(m, [-4, -3, -5, 1]);
    expect(opposite[0]).toBeCloseTo(-1, 6);
    expect(opposite[1]).toBeCloseTo(-1, 6);
  });

  it("maps the near and far planes to -1 and 1 with no perspective divide", () => {
    const m = orthographic(4, 3, 1, 9);
    const near = apply(m, [0, 0, -1, 1]);
    const far = apply(m, [0, 0, -9, 1]);
    expect(near[2]).toBeCloseTo(-1, 10);
    expect(far[2]).toBeCloseTo(1, 10);
    // `w` stays 1: the whole point of an orthographic frame.
    expect(near[3]).toBe(1);
    expect(far[3]).toBe(1);
    expect(m[11]).toBe(0);
    expect(m[15]).toBe(1);
  });
});

describe("matrix helpers", () => {
  it("multiplies column-major, so translation composes on the right", () => {
    const m = multiply(translation(1, 2, 3), translation(10, 20, 30));
    expect(apply(m, [0, 0, 0, 1])).toEqual([11, 22, 33, 1]);
  });

  it("leaves a vector alone under the identity", () => {
    expect(apply(identity(), [1, -2, 3, 1])).toEqual([1, -2, 3, 1]);
  });

  it("turns +Z toward +X under a positive rotationY", () => {
    const turned = apply(rotationY(Math.PI / 2), [0, 0, 1, 1]);
    expect(turned[0]).toBeCloseTo(1, 6);
    expect(turned[2]).toBeCloseTo(0, 6);
  });

  it("tips +Y toward +Z under a positive rotationX", () => {
    const tipped = apply(rotationX(Math.PI / 2), [0, 1, 0, 1]);
    expect(tipped[1]).toBeCloseTo(0, 6);
    expect(tipped[2]).toBeCloseTo(1, 6);
  });

  it("takes the upper-left 3×3 as the normal matrix", () => {
    const m = multiply(translation(5, 6, 7), rotationY(0.4));
    expect(Array.from(upper3x3(m))).toEqual(
      Array.from(upper3x3(rotationY(0.4))),
    );
  });
});
