<script setup lang="ts">
/*
 * Shared 3-D print viewer — the lightbox's answer to a `.glb` the way `<video>`
 * is its answer to an `.mp4`. Mounted by web, desktop and the iPhone gallery.
 *
 * Raw WebGL on purpose: this SPA is embedded in the `mold` binary, so a mesh
 * library would be paid for by every download of every install. What is here
 * is the small part of one that a single-primitive Hunyuan3D mesh needs — a
 * Lambert key/fill with a rim term, an orbit camera, and nothing else. The
 * container parsing lives in `@studio/lib/glb` so it can be tested without a
 * GPU or a DOM.
 *
 * A viewer that cannot start is NEVER a black rectangle: no WebGL, a refused
 * fetch, a corrupt file and a shader that will not link all land on the poster
 * the gallery already has, with one line saying why.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { GlbParseError, parseGlb, type ParsedMesh } from "../lib/glb";
import { meshStatsLabel } from "../lib/meshControls";
import {
  homeCamera,
  multiply,
  orthographic,
  orthographicScale,
  POSTER_MARGIN,
  rotationX,
  rotationY,
  sweepExtent,
  translation,
  upper3x3,
} from "../lib/meshViewerCamera";
import {
  advanceAutoRotate,
  edgeIndices,
  meshHasEdges,
} from "../lib/meshViewerMath";

const props = defineProps<{
  /** GLB URL the caller has already authorized (media-token URLs included). */
  src: string;
  /** Poster frame shown while loading and kept forever on any failure. */
  poster?: string;
  /** Describes the subject for assistive technology. */
  alt?: string;
  /** Slowly yaw the mesh until the first user interaction. */
  autoRotate?: boolean;
  /** Offer a fullscreen button in the controls. */
  expandable?: boolean;
}>();

/** What the caption and the `ready` event report about the loaded mesh. */
type MeshStats = {
  vertexCount: number;
  triangleCount: number;
  bounds: ParsedMesh["bounds"];
};

const emit = defineEmits<{
  /** The mesh is on the GPU and the first frame has been drawn. */
  ready: [stats: MeshStats];
  /** Rendering is impossible; the poster is showing instead. */
  fail: [message: string];
}>();

type Status = "loading" | "ready" | "failed";

const status = ref<Status>("loading");
const note = ref("");
const stats = ref<MeshStats | null>(null);
const canvas = ref<HTMLCanvasElement | null>(null);
const root = ref<HTMLElement | null>(null);
/** The edge overlay is a view option, not a property of the file. */
const wireframe = ref(false);
/** False for a mesh whose every triangle is degenerate: nothing to outline. */
const hasEdges = ref(true);
const WIREFRAME_UNAVAILABLE = "This mesh has no edges to outline.";
/** True only while the mesh is actually turning on its own. */
const autoRotating = ref(false);
const fullscreen = ref(false);

const label = computed(() => {
  const subject = props.alt?.trim() || "Generated 3-D mesh";
  if (status.value !== "ready") return subject;
  const base = `${subject}. Interactive 3-D view: drag or use the arrow keys to orbit, plus and minus to zoom.`;
  if (!autoRotating.value) return base;
  return `${base} It is rotating on its own until you interact with it.`;
});
// The one `tris · verts · bounds` caption every surface writes under a mesh.
const summary = computed(() => {
  const value = stats.value;
  if (!value) return "";
  return meshStatsLabel(value.vertexCount, value.triangleCount, value.bounds);
});

/*
 * The matrix helpers, the poster camera and the orthographic fit all live in
 * `@studio/lib/meshViewerCamera`, which mirrors the server's `poster.rs` under
 * a Rust contract test. This component owns GL state and event wiring only.
 */

// ── Shaders ────────────────────────────────────────────────────────────────
// GLSL ES 1.00 so one source compiles on both WebGL2 and a WebGL1 fallback.

const VERTEX_SHADER = `
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec3 aColor;
attribute vec2 aUv;
uniform mat4 uModelView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;
varying vec3 vNormal;
varying vec3 vView;
varying vec3 vColor;
varying vec2 vUv;
void main() {
  vec4 view = uModelView * vec4(aPosition, 1.0);
  vNormal = uNormalMatrix * aNormal;
  vView = view.xyz;
  vColor = aColor;
  vUv = aUv;
  gl_Position = uProjection * view;
}
`;

const FRAGMENT_SHADER = `
precision mediump float;
uniform sampler2D uTexture;
uniform float uHasTexture;
uniform float uWireframe;
varying vec3 vNormal;
varying vec3 vView;
varying vec3 vColor;
varying vec2 vUv;
void main() {
  // The edge pass reuses this program: one flag is cheaper than a second
  // compile, and the lines want a flat colour rather than the lighting.
  if (uWireframe > 0.5) {
    gl_FragColor = vec4(0.16, 0.85, 0.98, 1.0);
    return;
  }
  // Extracted surfaces are not reliably closed and the material is
  // doubleSided, so a back face is lit by its flipped normal, not left black.
  vec3 normal = normalize(vNormal);
  if (!gl_FrontFacing) normal = -normal;
  vec3 eye = normalize(-vView);
  vec3 key = normalize(vec3(0.45, 0.72, 0.85));
  vec3 fill = normalize(vec3(-0.65, -0.15, 0.35));
  float kd = max(dot(normal, key), 0.0);
  float fd = max(dot(normal, fill), 0.0);
  float rim = pow(1.0 - max(dot(normal, eye), 0.0), 2.5);
  vec3 albedo = vColor;
  if (uHasTexture > 0.5) albedo *= texture2D(uTexture, vUv).rgb;
  vec3 lit = albedo * (0.20 + 0.72 * kd + 0.22 * fd) + vec3(0.16, 0.19, 0.24) * rim;
  gl_FragColor = vec4(clamp(lit, 0.0, 1.0), 1.0);
}
`;

// ── GL state ───────────────────────────────────────────────────────────────

type GL = WebGLRenderingContext | WebGL2RenderingContext;

interface Scene {
  gl: GL;
  program: WebGLProgram;
  buffers: WebGLBuffer[];
  texture: WebGLTexture | null;
  indexBuffer: WebGLBuffer;
  indexCount: number;
  indexType: number;
  /** The triangle list as parsed, kept so the edge list can be built later. */
  sourceIndices: Uint32Array;
  /** Built on the first wireframe toggle and never rebuilt. */
  edgeBuffer: WebGLBuffer | null;
  edgeCount: number;
  /** World-space bounding-box centre: the point every camera orbits. */
  center: [number, number, number];
  /** Half the bounding-box diagonal. The depth range only — never the fit. */
  radius: number;
  /**
   * The rotation-invariant half-extent the poster and the turntable frame to,
   * so this viewer's home view IS the thumbnail and IS turntable frame 0.
   */
  extent: number;
  uniforms: {
    modelView: WebGLUniformLocation | null;
    projection: WebGLUniformLocation | null;
    normalMatrix: WebGLUniformLocation | null;
    texture: WebGLUniformLocation | null;
    hasTexture: WebGLUniformLocation | null;
    wireframe: WebGLUniformLocation | null;
  };
}

let scene: Scene | null = null;
let controller: AbortController | null = null;
let frame = 0;
let intersection: IntersectionObserver | null = null;
let resize: ResizeObserver | null = null;
/** RAF is only ever scheduled while both of these hold. */
let onScreen = true;
let pageVisible = true;

/** The poster's camera. `resetView` and every fresh mount start here. */
const HOME = homeCamera();
const camera = { ...HOME };
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 6;
/** A mesh larger than this is refused rather than allowed to wedge the tab. */
const MAX_BYTES = 256 * 1024 * 1024;

function fail(message: string): void {
  if (status.value === "failed") return;
  status.value = "failed";
  note.value = message;
  releaseGl();
  emit("fail", message);
}

// ── Render ─────────────────────────────────────────────────────────────────

function requestFrame(): void {
  if (frame !== 0 || !scene || !onScreen || !pageVisible) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    draw();
  });
}

function resizeCanvas(gl: GL): void {
  const element = canvas.value;
  if (!element) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round((element.clientWidth || 320) * ratio));
  const height = Math.max(1, Math.round((element.clientHeight || 320) * ratio));
  if (element.width !== width || element.height !== height) {
    element.width = width;
    element.height = height;
  }
  gl.viewport(0, 0, element.width, element.height);
}

function draw(): void {
  if (!scene) return;
  const { gl, program, uniforms } = scene;
  resizeCanvas(gl);
  const element = canvas.value;
  // Backing-store pixels, not CSS pixels: the devicePixelRatio cancels between
  // the fit and the half-extents, so a retina canvas frames the mesh exactly
  // as the server's poster does.
  const width = element?.width ?? 0;
  const height = element?.height ?? 0;
  // `camera.zoom` is the pull-back factor the wheel, the pinch and the +/-
  // keys have always spoken — larger means further away — so it DIVIDES the
  // fit here exactly as it multiplied the eye distance under perspective.
  const scale =
    orthographicScale(scene.extent, width, height, POSTER_MARGIN) / camera.zoom;
  // A mesh with no extent, or a canvas with no area, has nothing to frame:
  // clear rather than build a projection out of a division by zero.
  if (!(scale > 0)) {
    gl.useProgram(program);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    return;
  }

  const distance = scene.radius * 3;
  const modelView = multiply(
    multiply(
      translation(0, 0, -distance),
      multiply(rotationX(camera.pitch), rotationY(camera.yaw)),
    ),
    translation(-scene.center[0], -scene.center[1], -scene.center[2]),
  );
  // The mesh sits within `radius` of the eye axis' centre, so these planes
  // bracket it whatever the orbit angle.
  const projection = orthographic(
    width / 2 / scale,
    height / 2 / scale,
    distance - scene.radius * 2,
    distance + scene.radius * 2,
  );

  gl.useProgram(program);
  gl.uniformMatrix4fv(uniforms.modelView, false, modelView);
  gl.uniformMatrix4fv(uniforms.projection, false, projection);
  gl.uniformMatrix3fv(uniforms.normalMatrix, false, upper3x3(modelView));
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const overlay = wireframe.value && scene.edgeBuffer && scene.edgeCount > 0;
  // Pushing the filled triangles away from the eye by one depth unit is what
  // keeps the edges from z-fighting the very surface they outline.
  if (overlay) {
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(1, 1);
  }
  gl.uniform1f(uniforms.wireframe, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, scene.indexBuffer);
  gl.drawElements(gl.TRIANGLES, scene.indexCount, scene.indexType, 0);
  if (!overlay) return;
  gl.disable(gl.POLYGON_OFFSET_FILL);
  gl.uniform1f(uniforms.wireframe, 1);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, scene.edgeBuffer);
  gl.drawElements(gl.LINES, scene.edgeCount, scene.indexType, 0);
}

/**
 * Uploads the edge list, once, the first time the overlay is switched on. A
 * mesh nobody wireframes never pays for the deduplication or the buffer.
 */
function ensureEdges(): boolean {
  const current = scene;
  if (!current) return false;
  if (current.edgeBuffer) return true;
  const { gl } = current;
  const edges = edgeIndices(current.sourceIndices);
  if (edges.length === 0) return false;
  const data =
    current.indexType === gl.UNSIGNED_SHORT ? Uint16Array.from(edges) : edges;
  const buffer = gl.createBuffer();
  if (!buffer) return false;
  current.buffers.push(buffer);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW);
  current.edgeBuffer = buffer;
  current.edgeCount = data.length;
  return true;
}

function toggleWireframe(): void {
  if (!hasEdges.value) return;
  if (wireframe.value) {
    wireframe.value = false;
  } else {
    // A GPU that refuses the edge buffer leaves the button where it was
    // rather than promising an overlay that will never draw.
    if (!ensureEdges()) return;
    wireframe.value = true;
  }
  requestFrame();
}

// ── Upload ─────────────────────────────────────────────────────────────────

function compile(gl: GL, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("the GPU refused a shader object");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? "unknown error";
    gl.deleteShader(shader);
    throw new Error(`shader failed to compile: ${log}`);
  }
  return shader;
}

function link(gl: GL): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error("the GPU refused a program object");
  const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  // The shaders are owned by the program once attached; flagging them for
  // delete here is what frees them when the program goes.
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? "unknown error";
    gl.deleteProgram(program);
    throw new Error(`shader program failed to link: ${log}`);
  }
  return program;
}

function attribute(
  gl: GL,
  program: WebGLProgram,
  name: string,
  data: Float32Array | null,
  size: number,
  fallback: number[],
  buffers: WebGLBuffer[],
): void {
  const location = gl.getAttribLocation(program, name);
  if (location < 0) return;
  if (!data) {
    // A constant vertex attribute costs no buffer: an untextured or
    // uncolored mesh takes the same code path as a full one.
    gl.disableVertexAttribArray(location);
    if (size === 2)
      gl.vertexAttrib2f(location, fallback[0] ?? 0, fallback[1] ?? 0);
    else
      gl.vertexAttrib3f(
        location,
        fallback[0] ?? 1,
        fallback[1] ?? 1,
        fallback[2] ?? 1,
      );
    return;
  }
  const buffer = gl.createBuffer();
  if (!buffer) throw new Error(`the GPU refused a buffer for ${name}`);
  buffers.push(buffer);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
}

async function decodeTexture(blob: Blob): Promise<ImageBitmap | null> {
  // No createImageBitmap (very old WebKit): the mesh still shades, it just
  // shades with its vertex colors. Worth strictly less than a second fetch.
  if (typeof createImageBitmap !== "function") return null;
  try {
    return await createImageBitmap(blob);
  } catch {
    return null;
  }
}

async function upload(mesh: ParsedMesh): Promise<void> {
  const element = canvas.value;
  if (!element) return;
  const options: WebGLContextAttributes = {
    alpha: true,
    antialias: true,
    depth: true,
    premultipliedAlpha: true,
    powerPreference: "low-power",
  };
  const gl =
    (element.getContext("webgl2", options) as WebGL2RenderingContext | null) ??
    (element.getContext("webgl", options) as WebGLRenderingContext | null);
  if (!gl) {
    fail("This browser can't display 3-D previews, so here's the poster.");
    return;
  }

  const isWebgl2 =
    typeof WebGL2RenderingContext !== "undefined" &&
    gl instanceof WebGL2RenderingContext;
  let indices: Uint32Array | Uint16Array = mesh.indices;
  let indexType: number = gl.UNSIGNED_INT;
  if (!isWebgl2 && !gl.getExtension("OES_element_index_uint")) {
    if (mesh.vertexCount > 65536) {
      fail("This mesh is too detailed for this browser's WebGL 1 renderer.");
      return;
    }
    indices = Uint16Array.from(mesh.indices);
    indexType = gl.UNSIGNED_SHORT;
  }

  const buffers: WebGLBuffer[] = [];
  let program: WebGLProgram;
  try {
    program = link(gl);
  } catch (error) {
    fail(errorNote(error));
    return;
  }
  gl.useProgram(program);
  // Nothing below has reached `scene` yet, so `releaseGl` cannot see it: every
  // exit before the assignment hands the program and buffers back here.
  const abandon = (): void => {
    for (const buffer of buffers) gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
  };

  try {
    attribute(gl, program, "aPosition", mesh.positions, 3, [0, 0, 0], buffers);
    attribute(gl, program, "aNormal", mesh.normals, 3, [0, 0, 1], buffers);
    attribute(
      gl,
      program,
      "aColor",
      mesh.colors,
      3,
      [0.82, 0.82, 0.86],
      buffers,
    );
    attribute(gl, program, "aUv", mesh.uvs, 2, [0, 0], buffers);
  } catch (error) {
    abandon();
    fail(errorNote(error));
    return;
  }

  const indexBuffer = gl.createBuffer();
  if (!indexBuffer) {
    abandon();
    fail("The GPU refused the mesh's index buffer.");
    return;
  }
  buffers.push(indexBuffer);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  const uniforms = {
    modelView: gl.getUniformLocation(program, "uModelView"),
    projection: gl.getUniformLocation(program, "uProjection"),
    normalMatrix: gl.getUniformLocation(program, "uNormalMatrix"),
    texture: gl.getUniformLocation(program, "uTexture"),
    hasTexture: gl.getUniformLocation(program, "uHasTexture"),
    wireframe: gl.getUniformLocation(program, "uWireframe"),
  };

  let texture: WebGLTexture | null = null;
  const bitmap =
    mesh.uvs && mesh.baseColorTexture
      ? await decodeTexture(mesh.baseColorTexture)
      : null;
  // The await above is the one place this function yields, so an unmount
  // between the fetch and here must not upload into a dead context.
  if (controller?.signal.aborted) {
    bitmap?.close();
    abandon();
    return;
  }
  if (bitmap) {
    texture = gl.createTexture();
    if (texture) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      // glTF's UV origin is the image's top-left, which is exactly what an
      // unflipped upload gives: never enable UNPACK_FLIP_Y_WEBGL here.
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        bitmap,
      );
      // No mipmaps: mold's textures are not power-of-two, and WebGL1 renders
      // an NPOT texture black unless it is CLAMP_TO_EDGE + non-mipmap.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(uniforms.texture, 0);
    }
    bitmap.close();
  }
  gl.uniform1f(uniforms.hasTexture, texture ? 1 : 0);

  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE); // mold writes doubleSided materials
  gl.clearColor(0, 0, 0, 0);

  const { min, max } = mesh.bounds;
  const center: [number, number, number] = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const radius =
    Math.max(
      Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2,
      1e-4,
    ) || 1;

  scene = {
    gl,
    program,
    buffers,
    texture,
    indexBuffer,
    indexCount: indices.length,
    indexType,
    sourceIndices: mesh.indices,
    edgeBuffer: null,
    edgeCount: 0,
    center,
    radius,
    extent: sweepExtent(mesh.positions, center, HOME.pitch),
    uniforms,
  };
  hasEdges.value = meshHasEdges(mesh.indices);
  stats.value = {
    vertexCount: mesh.vertexCount,
    triangleCount: mesh.triangleCount,
    bounds: mesh.bounds,
  };
  status.value = "ready";
  note.value = "";
  draw();
  startAutoRotate();
  emit("ready", stats.value);
}

function errorNote(error: unknown): string {
  if (error instanceof GlbParseError) return "This mesh file couldn't be read.";
  if (error instanceof Error && error.name === "AbortError") return "";
  return "The 3-D view couldn't start, so here's the poster.";
}

// ── Auto-rotation ──────────────────────────────────────────────────────────

let autoFrame = 0;
let autoStamp = -1;
/** Once a person has touched the viewer it is theirs for the rest of the mount. */
let interacted = false;
/** A stalled tab hands back a huge delta; a jump is worse than a dropped frame. */
const MAX_AUTO_STEP_MS = 100;

function reducedMotionQuery(): MediaQueryList | null {
  // Absent in tests and in older WebViews: no query is not a preference.
  const query = typeof window !== "undefined" ? window.matchMedia : undefined;
  if (typeof query !== "function") return null;
  try {
    return query.call(window, "(prefers-reduced-motion: reduce)");
  } catch {
    return null;
  }
}

function prefersReducedMotion(): boolean {
  return reducedMotionQuery()?.matches === true;
}

/** Live for the mount: the tour parks or resumes as the OS setting changes. */
let motionQuery: MediaQueryList | null = null;

function onReducedMotionChange(event: { matches: boolean }): void {
  if (event.matches) stopAutoRotate();
  else startAutoRotate();
}

function stepAutoRotate(stamp: number): void {
  autoFrame = 0;
  if (!autoRotating.value || !scene) return;
  // Scrolled away or in a background tab: park rather than burn a callback a
  // frame. The observers below start the loop again when the viewer returns.
  if (!onScreen || !pageVisible) {
    autoStamp = -1;
    return;
  }
  const elapsed = autoStamp < 0 ? 0 : stamp - autoStamp;
  camera.yaw = advanceAutoRotate(
    camera.yaw,
    Math.min(elapsed, MAX_AUTO_STEP_MS),
  );
  draw();
  autoStamp = stamp;
  autoFrame = requestAnimationFrame(stepAutoRotate);
}

function scheduleAutoRotate(): void {
  if (autoFrame !== 0 || !autoRotating.value || !onScreen || !pageVisible) {
    return;
  }
  autoStamp = -1;
  autoFrame = requestAnimationFrame(stepAutoRotate);
}

function startAutoRotate(): void {
  if (autoRotating.value || interacted || !props.autoRotate || !scene) return;
  if (prefersReducedMotion()) return;
  autoRotating.value = true;
  scheduleAutoRotate();
}

function stopAutoRotate(): void {
  autoRotating.value = false;
  autoStamp = -1;
  if (autoFrame !== 0) {
    cancelAnimationFrame(autoFrame);
    autoFrame = 0;
  }
}

/** The first drag, key or wheel ends the tour, permanently, for this mount. */
function noteInteraction(): void {
  if (interacted) return;
  interacted = true;
  stopAutoRotate();
}

// ── Fullscreen ─────────────────────────────────────────────────────────────

const canFullscreen = computed(
  () =>
    props.expandable === true &&
    typeof document !== "undefined" &&
    document.fullscreenEnabled === true,
);

function toggleFullscreen(): void {
  const element = root.value;
  if (!element) return;
  // Both calls reject when the browser refuses the gesture; the button simply
  // stays where it was, and `fullscreenchange` remains the only truth.
  if (document.fullscreenElement === element) {
    void Promise.resolve(document.exitFullscreen?.()).catch(() => {});
  } else {
    void Promise.resolve(element.requestFullscreen?.()).catch(() => {});
  }
}

function onFullscreenChange(): void {
  fullscreen.value = !!root.value && document.fullscreenElement === root.value;
  // The viewport just changed size; `draw` re-fits the backing store for it.
  requestFrame();
}

function onWindowResize(): void {
  requestFrame();
}

// ── Lifecycle ──────────────────────────────────────────────────────────────

async function load(): Promise<void> {
  const request = new AbortController();
  controller = request;
  status.value = "loading";
  note.value = "";
  stats.value = null;
  try {
    const response = await fetch(props.src, { signal: request.signal });
    if (!response.ok) {
      throw new Error(`mesh request failed with HTTP ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    if (request.signal.aborted) return;
    if (buffer.byteLength > MAX_BYTES) {
      fail("This mesh is too large to preview in the browser.");
      return;
    }
    const mesh = parseGlb(buffer);
    if (request.signal.aborted) return;
    await upload(mesh);
  } catch (error) {
    if (request.signal.aborted) return;
    fail(errorNote(error) || "The 3-D view couldn't start.");
  }
}

function releaseGl(): void {
  stopAutoRotate();
  if (frame !== 0) {
    cancelAnimationFrame(frame);
    frame = 0;
  }
  wireframe.value = false;
  const current = scene;
  scene = null;
  if (!current) return;
  const { gl } = current;
  for (const buffer of current.buffers) gl.deleteBuffer(buffer);
  if (current.texture) gl.deleteTexture(current.texture);
  gl.deleteProgram(current.program);
  // A lightbox opens and closes all session; without this the browser hits its
  // hard context limit (~16) and silently stops giving this page a GPU.
  gl.getExtension("WEBGL_lose_context")?.loseContext();
}

function onPageVisibility(): void {
  pageVisible = !document.hidden;
  requestFrame();
  scheduleAutoRotate();
}

onMounted(() => {
  if (typeof IntersectionObserver === "function" && root.value) {
    intersection = new IntersectionObserver((entries) => {
      onScreen = entries.some((entry) => entry.isIntersecting);
      requestFrame();
      scheduleAutoRotate();
    });
    intersection.observe(root.value);
  }
  if (typeof ResizeObserver === "function" && root.value) {
    resize = new ResizeObserver(() => requestFrame());
    resize.observe(root.value);
  }
  document.addEventListener("visibilitychange", onPageVisibility);
  document.addEventListener("fullscreenchange", onFullscreenChange);
  window.addEventListener("resize", onWindowResize);
  motionQuery = reducedMotionQuery();
  motionQuery?.addEventListener?.("change", onReducedMotionChange);
  pageVisible = !document.hidden;
  void load();
});

onBeforeUnmount(() => {
  controller?.abort();
  controller = null;
  intersection?.disconnect();
  intersection = null;
  resize?.disconnect();
  resize = null;
  document.removeEventListener("visibilitychange", onPageVisibility);
  document.removeEventListener("fullscreenchange", onFullscreenChange);
  window.removeEventListener("resize", onWindowResize);
  motionQuery?.removeEventListener?.("change", onReducedMotionChange);
  motionQuery = null;
  releaseGl();
  pointers.clear();
});

watch(
  () => props.src,
  () => {
    controller?.abort();
    releaseGl();
    resetView();
    void load();
  },
);

// A caller that turns the tour on late still gets one, unless this viewer has
// already been handled — an interaction is never taken back.
watch(
  () => props.autoRotate,
  (wanted) => {
    if (wanted) startAutoRotate();
    else stopAutoRotate();
  },
);

// ── Orbit ──────────────────────────────────────────────────────────────────

const PITCH_LIMIT = Math.PI / 2 - 0.01;
const pointers = new Map<number, { x: number; y: number }>();
let pinchDistance = 0;

function orbit(dx: number, dy: number): void {
  camera.yaw += dx;
  camera.pitch = Math.min(
    PITCH_LIMIT,
    Math.max(-PITCH_LIMIT, camera.pitch + dy),
  );
  requestFrame();
}

function zoomBy(factor: number): void {
  camera.zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, camera.zoom * factor));
  requestFrame();
}

function resetView(): void {
  camera.yaw = HOME.yaw;
  camera.pitch = HOME.pitch;
  camera.zoom = HOME.zoom;
  requestFrame();
}

function spread(): number {
  const [a, b] = [...pointers.values()];
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function onPointerDown(event: PointerEvent): void {
  noteInteraction();
  if (status.value !== "ready") return;
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (pointers.size === 2) pinchDistance = spread();
  (event.currentTarget as Element | null)?.setPointerCapture?.(event.pointerId);
}

function onPointerMove(event: PointerEvent): void {
  const previous = pointers.get(event.pointerId);
  if (!previous) return;
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (pointers.size >= 2) {
    const next = spread();
    if (pinchDistance > 0 && next > 0) zoomBy(pinchDistance / next);
    pinchDistance = next;
    return;
  }
  orbit(
    (event.clientX - previous.x) * 0.008,
    (event.clientY - previous.y) * 0.008,
  );
}

function onPointerUp(event: PointerEvent): void {
  pointers.delete(event.pointerId);
  pinchDistance = pointers.size === 2 ? spread() : 0;
  (event.currentTarget as Element | null)?.releasePointerCapture?.(
    event.pointerId,
  );
}

function onWheel(event: WheelEvent): void {
  noteInteraction();
  if (status.value !== "ready") return;
  event.preventDefault();
  zoomBy(Math.exp(event.deltaY * 0.0015));
}

function onKeydown(event: KeyboardEvent): void {
  noteInteraction();
  if (status.value !== "ready") return;
  const step = event.shiftKey ? 0.3 : 0.12;
  switch (event.key) {
    case "ArrowLeft":
      orbit(-step, 0);
      break;
    case "ArrowRight":
      orbit(step, 0);
      break;
    case "ArrowUp":
      orbit(0, -step);
      break;
    case "ArrowDown":
      orbit(0, step);
      break;
    case "+":
    case "=":
      zoomBy(1 / 1.15);
      break;
    case "-":
    case "_":
      zoomBy(1.15);
      break;
    case "0":
      resetView();
      break;
    default:
      return;
  }
  event.preventDefault();
}
</script>

<template>
  <!-- `data-gesture="own"`: this surface reads drags as camera orbit, so an
       ancestor must not also read them as a swipe. It pairs with
       `touch-action: none` below, which only stops the BROWSER's own panning
       and never a JavaScript pointer handler on a parent. -->
  <div
    ref="root"
    class="mesh-viewer"
    data-test="mesh-viewer"
    data-gesture="own"
    :data-status="status"
  >
    <img
      v-if="poster && status !== 'ready'"
      :src="poster"
      :alt="alt || 'Poster frame for the generated mesh'"
      class="mesh-viewer__poster"
      data-test="mesh-viewer-poster"
      draggable="false"
    />
    <canvas
      v-show="status === 'ready'"
      ref="canvas"
      class="mesh-viewer__canvas"
      data-test="mesh-viewer-canvas"
      role="img"
      tabindex="0"
      :aria-label="label"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
      @wheel="onWheel"
      @keydown="onKeydown"
      @dblclick="resetView"
    />
    <p
      v-if="status !== 'ready'"
      class="mesh-viewer__note"
      data-test="mesh-viewer-note"
      role="status"
    >
      {{ status === "loading" ? "Loading the 3-D view…" : note }}
    </p>
    <div v-else class="mesh-viewer__controls">
      <span class="mesh-viewer__stats" data-test="mesh-viewer-stats">
        {{ summary }}
      </span>
      <span class="mesh-viewer__buttons">
        <button
          type="button"
          class="mesh-viewer__button"
          data-test="mesh-viewer-wireframe"
          :aria-pressed="wireframe ? 'true' : 'false'"
          :disabled="!hasEdges"
          :title="hasEdges ? undefined : WIREFRAME_UNAVAILABLE"
          @click="toggleWireframe"
        >
          Wireframe
        </button>
        <button
          v-if="canFullscreen"
          type="button"
          class="mesh-viewer__button"
          data-test="mesh-viewer-fullscreen"
          :aria-label="fullscreen ? 'Exit fullscreen' : 'Enter fullscreen'"
          @click="toggleFullscreen"
        >
          {{ fullscreen ? "Exit fullscreen" : "Fullscreen" }}
        </button>
        <button
          type="button"
          class="mesh-viewer__button"
          data-test="mesh-viewer-reset"
          @click="resetView"
        >
          Reset view
        </button>
      </span>
    </div>
  </div>
</template>

<style scoped>
.mesh-viewer {
  position: relative;
  display: block;
  width: 100%;
  height: 100%;
  min-height: 180px;
  overflow: hidden;
  border-radius: var(--radius-card, 12px);
  /* The bed prints are always viewed on; the canvas clears to transparent so
     this token shows through and follows the theme without a GL reupload. */
  background: var(--print, #141110);
  color: var(--on-media, #f5efff);
}
.mesh-viewer__poster,
.mesh-viewer__canvas {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.mesh-viewer__canvas {
  touch-action: none;
  cursor: grab;
  outline-offset: -3px;
}
.mesh-viewer__canvas:active {
  cursor: grabbing;
}
.mesh-viewer__note {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  margin: 0;
  padding: 8px 12px;
  background: linear-gradient(transparent, rgba(0, 0, 0, 0.55));
  color: inherit;
  font-size: 12px;
  text-align: center;
}
.mesh-viewer__controls {
  position: absolute;
  right: 8px;
  bottom: 8px;
  left: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  pointer-events: none;
}
.mesh-viewer__stats {
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  opacity: 0.72;
}
.mesh-viewer__buttons {
  display: flex;
  align-items: center;
  gap: 6px;
}
.mesh-viewer__button {
  min-height: 32px;
  padding: 0 12px;
  border: 1px solid var(--edge, rgba(255, 255, 255, 0.2));
  border-radius: var(--radius-control, 9px);
  background: rgba(0, 0, 0, 0.42);
  color: inherit;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  pointer-events: auto;
}
.mesh-viewer__button[aria-pressed="true"] {
  border-color: var(--accent, #29d9fa);
  background: rgba(41, 217, 250, 0.24);
}
</style>
