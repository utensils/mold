/*
 * The ONE camera convention shared by the server's poster, the server's
 * turntable, and `MeshViewer.vue`'s home view.
 *
 * These four literals MIRROR `crates/mold-inference/src/hunyuan3d/poster.rs`
 * (`POSTER_AZIMUTH_DEG`, `POSTER_ELEVATION_DEG`, `POSTER_MARGIN`,
 * `TURNTABLE_AZIMUTH_STEP_SIGN`). A Rust test there reads THIS file and
 * fails the build when they drift, so change them in both places or not at
 * all. The conversion between the two frames lives here and nowhere else:
 * the server orbits the eye by `azimuth` about +Y with azimuth 0 on +Z; the
 * viewer rotates the MODEL by `yaw` about +Y, so `yaw = -azimuth` and
 * `pitch = elevation`.
 */

/** Orbit angle of the poster's eye about +Y, degrees. 0 places the eye on +Z. */
export const POSTER_AZIMUTH_DEG = 30;
/** Angle of the poster's eye above the XZ plane, degrees. */
export const POSTER_ELEVATION_DEG = 20;
/** Fraction of the frame left empty around the mesh's sweep extent. */
export const POSTER_MARGIN = 0.08;
/**
 * Sign of the turntable's per-frame azimuth step. Negative means the eye
 * orbits toward -X, so the object spins to the RIGHT on screen — the way a
 * rightward drag turns it in the viewer, and the way auto-rotate tours it.
 */
export const TURNTABLE_AZIMUTH_STEP_SIGN = -1;

/** The viewer's own camera state: model rotation plus a zoom-out factor. */
export interface ViewerCamera {
  /** Model rotation about +Y, radians. `yaw = -azimuth`. */
  yaw: number;
  /** Model rotation about +X, radians. `pitch = elevation`. */
  pitch: number;
  /** Multiplies the framed extent: 1 is the poster's own framing. */
  zoom: number;
}

/**
 * The camera the viewer opens on, and the one `0` returns it to: EXACTLY the
 * server's poster camera, so the gallery thumbnail, the viewer's first frame
 * and turntable frame 0 are the same picture.
 */
export function homeCamera(): ViewerCamera {
  return {
    yaw: (-POSTER_AZIMUTH_DEG * Math.PI) / 180,
    pitch: (POSTER_ELEVATION_DEG * Math.PI) / 180,
    zoom: 1,
  };
}

/**
 * The server-frame azimuth, in degrees, a viewer yaw is looking from. The one
 * place the two conventions are converted, in either direction.
 */
export function azimuthDegOfYaw(yaw: number): number {
  return (-yaw * 180) / Math.PI;
}

/**
 * The rotation-invariant half-extent that frames `positions` from EVERY
 * azimuth at `elevationRad`, about `center`.
 *
 * The closed form of the bounding cylinder about the bounding box's centre:
 * for each vertex, the larger of its radial distance (the widest that vertex
 * can ever project horizontally, at the azimuth that puts it on the silhouette)
 * and its projected height `cos e · |dy| + sin e · radial`. Mirrors
 * `sweep_fit_for` in `crates/mold-inference/src/hunyuan3d/raster.rs`; because
 * it depends on neither the azimuth nor the frame count, the poster, a 36-frame
 * turntable, a 72-frame one and this viewer all frame the mesh identically.
 *
 * The elevation is used by magnitude, so looking up at the mesh frames it
 * exactly as looking down does. `0` when there is nothing finite to frame,
 * which the caller reads as "draw nothing" rather than as a scale.
 */
export function sweepExtent(
  positions: Float32Array | ArrayLike<number>,
  center: readonly [number, number, number],
  elevationRad: number,
): number {
  return sweepExtentOfProfile(sweepProfile(positions, center), elevationRad);
}

/**
 * The `(radial, |dy|)` pair per finite vertex, interleaved — everything
 * [`sweepExtentOfProfile`] needs, and nothing else.
 *
 * Two thirds the size of the positions it replaces, and it lifts the centring,
 * the hypotenuse and the finiteness check out of the per-elevation loop. A
 * viewer that re-frames as the mesh tilts pays for those once, at upload.
 *
 * The pairs are stored single-precision, which is the precision `sweep_fit_for`
 * computes the same bound at.
 */
export function sweepProfile(
  positions: Float32Array | ArrayLike<number>,
  center: readonly [number, number, number],
): Float32Array {
  const [cx, cy, cz] = center;
  const triples = positions.length - (positions.length % 3);
  const out = new Float32Array((triples / 3) * 2);
  let count = 0;
  for (let i = 0; i < triples; i += 3) {
    const dx = (positions[i] ?? Number.NaN) - cx;
    const dy = (positions[i + 1] ?? Number.NaN) - cy;
    const dz = (positions[i + 2] ?? Number.NaN) - cz;
    // A NaN or infinite vertex would poison the max for the whole mesh.
    if (!Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(dz)) {
      continue;
    }
    out[count] = Math.hypot(dx, dz);
    out[count + 1] = Math.abs(dy);
    count += 2;
  }
  return count === out.length ? out : out.slice(0, count);
}

/**
 * [`sweepExtent`] evaluated against a prepared [`sweepProfile`], for a viewer
 * that has to re-frame every time the elevation changes.
 *
 * Identical to `sweepExtent` for the same inputs — a test pins that — so the
 * profile is an optimization and never a second definition of the framing.
 */
export function sweepExtentOfProfile(
  profile: Float32Array | ArrayLike<number>,
  elevationRad: number,
): number {
  // Absolute values, so a camera looking UP at the mesh is bounded exactly as
  // one looking down: only the magnitudes of the two basis terms matter.
  // `sweep_fit_for` takes the same absolutes.
  const sinE = Math.abs(Math.sin(elevationRad));
  const cosE = Math.abs(Math.cos(elevationRad));
  const pairs = profile.length - (profile.length % 2);
  let extent = 0;
  for (let i = 0; i < pairs; i += 2) {
    const radial = profile[i] ?? 0;
    const height = profile[i + 1] ?? 0;
    const candidate = Math.max(radial, cosE * height + sinE * radial);
    if (candidate > extent) extent = candidate;
  }
  return extent;
}

/**
 * Pixels per world unit for a frame of `width` × `height` that leaves `margin`
 * of itself empty around `extent`.
 *
 * `fit_scale`'s `FrameFit::Extent` arm in
 * `crates/mold-inference/src/hunyuan3d/raster.rs`, margin clamp included.
 * `0` — never `Infinity` — for a mesh with no extent, so a caller that
 * multiplies by it cannot produce a NaN projection matrix.
 */
export function orthographicScale(
  extent: number,
  width: number,
  height: number,
  margin: number,
): number {
  if (!Number.isFinite(extent) || extent <= 0) return 0;
  const half = Math.min(0.5 * width, 0.5 * height);
  const clamped = Math.min(
    Math.max(Number.isFinite(margin) ? margin : 0, 0),
    0.9,
  );
  const scale = (half / extent) * (1 - clamped);
  return Number.isFinite(scale) && scale > 0 ? scale : 0;
}

// ── Matrices (column-major, the order WebGL uniforms want) ─────────────────

export type Mat4 = Float32Array;

export function identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += (a[k * 4 + row] ?? 0) * (b[column * 4 + k] ?? 0);
      }
      out[column * 4 + row] = sum;
    }
  }
  return out;
}

export function translation(x: number, y: number, z: number): Mat4 {
  const m = identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

export function rotationX(angle: number): Mat4 {
  const m = identity();
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  m[5] = c;
  m[6] = s;
  m[9] = -s;
  m[10] = c;
  return m;
}

export function rotationY(angle: number): Mat4 {
  const m = identity();
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  m[0] = c;
  m[2] = -s;
  m[8] = s;
  m[10] = c;
  return m;
}

/**
 * A symmetric orthographic projection: the server renders the poster and every
 * turntable frame orthographically, so the viewer must too or its home view
 * would carry a perspective the thumbnail does not.
 */
export function orthographic(
  halfWidth: number,
  halfHeight: number,
  near: number,
  far: number,
): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1 / halfWidth;
  m[5] = 1 / halfHeight;
  m[10] = -2 / (far - near);
  m[14] = -(far + near) / (far - near);
  m[15] = 1;
  return m;
}

/** Upper-left 3×3. The camera only rotates, so this IS the normal matrix. */
export function upper3x3(m: Mat4): Float32Array {
  return new Float32Array([
    m[0] ?? 0,
    m[1] ?? 0,
    m[2] ?? 0,
    m[4] ?? 0,
    m[5] ?? 0,
    m[6] ?? 0,
    m[8] ?? 0,
    m[9] ?? 0,
    m[10] ?? 0,
  ]);
}
