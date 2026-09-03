import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseGlb } from "../lib/glb";
import { assemble, buildDocument, triangleGlb } from "../lib/glbFixture";
import { meshStatsLabel } from "../lib/meshControls";
import {
  homeCamera,
  multiply,
  orthographicScale,
  POSTER_MARGIN,
  rotationX,
  rotationY,
  sweepExtent,
  sweepExtentOfProfile,
  sweepProfile,
  translation,
} from "../lib/meshViewerCamera";
import MeshViewer from "./MeshViewer.vue";

/*
 * The GPU half of this component is covered by the browser, not here: happy-dom
 * has no WebGL, so every mount in this file lands on the degraded path. That is
 * exactly the path worth pinning — a lightbox must never show a black rectangle,
 * and the poster it falls back to is the only thing a viewer without WebGL,
 * without the file, or with a corrupt file ever sees.
 */

interface FetchCall {
  url: string;
  signal: AbortSignal | undefined;
}

const calls: FetchCall[] = [];

function stubFetch(respond: () => Promise<unknown>): void {
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    calls.push({ url, signal: init?.signal ?? undefined });
    return respond();
  });
}

function ok(body: ArrayBuffer): Promise<unknown> {
  return Promise.resolve({
    ok: true,
    status: 200,
    arrayBuffer: () => Promise.resolve(body),
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  calls.length = 0;
});

describe("MeshViewer", () => {
  it("shows the poster and a status line while the mesh loads", () => {
    stubFetch(() => new Promise(() => {}));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", poster: "/media/mesh.png" },
    });
    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("loading");
    expect(
      wrapper.get("[data-test=mesh-viewer-poster]").attributes("src"),
    ).toBe("/media/mesh.png");
    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toContain(
      "Loading",
    );
    wrapper.unmount();
  });

  it("keeps the poster and says so when the file cannot be read", async () => {
    stubFetch(() => ok(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]).buffer));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", poster: "/media/mesh.png" },
    });
    await flushPromises();

    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("failed");
    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toBe(
      "This mesh file couldn't be read.",
    );
    expect(wrapper.find("[data-test=mesh-viewer-poster]").exists()).toBe(true);
    expect(wrapper.emitted("fail")).toHaveLength(1);
    wrapper.unmount();
  });

  it("falls back to the poster when the browser has no WebGL", async () => {
    // happy-dom hands back null already; pinning it keeps the assertion about
    // the component rather than about the DOM stand-in.
    const getContext = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(null);
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", poster: "/media/mesh.png" },
    });
    await flushPromises();

    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("failed");
    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toContain(
      "can't display 3-D previews",
    );
    expect(wrapper.find("[data-test=mesh-viewer-poster]").exists()).toBe(true);
    expect(wrapper.emitted("fail")).toHaveLength(1);
    getContext.mockRestore();
    wrapper.unmount();
  });

  it("falls back without blaming the file when the host refuses it", async () => {
    stubFetch(() =>
      Promise.resolve({
        ok: false,
        status: 404,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
      }),
    );
    const wrapper = mount(MeshViewer, { props: { src: "/media/gone.glb" } });
    await flushPromises();

    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toContain(
      "couldn't start",
    );
    expect(wrapper.emitted("fail")).toHaveLength(1);
    wrapper.unmount();
  });

  it("aborts the in-flight fetch on unmount", async () => {
    stubFetch(() => new Promise(() => {}));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    expect(calls).toHaveLength(1);
    expect(calls[0]?.signal?.aborted).toBe(false);

    wrapper.unmount();
    await flushPromises();
    expect(calls[0]?.signal?.aborted).toBe(true);
    // An aborted load is not a failure the viewer should announce.
    expect(wrapper.emitted("fail")).toBeUndefined();
  });

  it("refetches when the src changes", async () => {
    stubFetch(() => new Promise(() => {}));
    const wrapper = mount(MeshViewer, { props: { src: "/media/one.glb" } });
    await wrapper.setProps({ src: "/media/two.glb" });
    await flushPromises();

    expect(calls.map((call) => call.url)).toEqual([
      "/media/one.glb",
      "/media/two.glb",
    ]);
    expect(calls[0]?.signal?.aborted).toBe(true);
    wrapper.unmount();
  });

  it("labels the canvas for assistive technology", () => {
    stubFetch(() => new Promise(() => {}));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", alt: "A ceramic owl" },
    });
    const canvas = wrapper.get("[data-test=mesh-viewer-canvas]");
    expect(canvas.attributes("role")).toBe("img");
    expect(canvas.attributes("tabindex")).toBe("0");
    expect(canvas.attributes("aria-label")).toContain("A ceramic owl");
    wrapper.unmount();
  });
});

/*
 * The interactive half — auto-rotation, fullscreen and the wireframe overlay —
 * needs a viewer that actually reached `ready`, which happy-dom cannot give us.
 * The stub below is a recording WebGL context: every constant the component
 * touches carries its real value, every entry point answers, and the calls
 * worth asserting on (draws, index binds, polygon offset) are logged by name.
 */

const GL_CONSTANTS: Record<string, number> = {
  VERTEX_SHADER: 0x8b31,
  FRAGMENT_SHADER: 0x8b30,
  COMPILE_STATUS: 0x8b81,
  LINK_STATUS: 0x8b82,
  ARRAY_BUFFER: 0x8892,
  ELEMENT_ARRAY_BUFFER: 0x8893,
  STATIC_DRAW: 0x88e4,
  FLOAT: 0x1406,
  UNSIGNED_INT: 0x1405,
  UNSIGNED_SHORT: 0x1403,
  UNSIGNED_BYTE: 0x1401,
  DEPTH_TEST: 0x0b71,
  CULL_FACE: 0x0b44,
  POLYGON_OFFSET_FILL: 0x8037,
  COLOR_BUFFER_BIT: 0x4000,
  DEPTH_BUFFER_BIT: 0x0100,
  TRIANGLES: 0x0004,
  LINES: 0x0001,
  TEXTURE_2D: 0x0de1,
  RGBA: 0x1908,
  TEXTURE_MIN_FILTER: 0x2801,
  TEXTURE_MAG_FILTER: 0x2800,
  LINEAR: 0x2601,
  TEXTURE_WRAP_S: 0x2802,
  TEXTURE_WRAP_T: 0x2803,
  CLAMP_TO_EDGE: 0x812f,
  TEXTURE0: 0x84c0,
  UNPACK_FLIP_Y_WEBGL: 0x9240,
};

const GL_NAMES = new Map(
  Object.entries(GL_CONSTANTS).map(([name, value]) => [value, name]),
);

interface GlRecorder {
  log: string[];
  modelViews: number[][];
  projections: number[][];
  restore: () => void;
}

function stubWebgl(extra: Record<string, unknown> = {}): GlRecorder {
  const log: string[] = [];
  const modelViews: number[][] = [];
  const projections: number[][] = [];
  const named = new Map<object, string>();
  const overrides: Record<string, unknown> = {
    deleteBuffer: () => {
      log.push("deleteBuffer");
    },
    deleteProgram: () => {
      log.push("deleteProgram");
    },
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getShaderInfoLog: () => "",
    getProgramInfoLog: () => "",
    createShader: () => ({}),
    createProgram: () => ({}),
    createTexture: () => ({}),
    createBuffer: () => ({}),
    getExtension: () => ({ loseContext: () => {} }),
    getAttribLocation: (_program: unknown, name: string) =>
      ["aPosition", "aNormal", "aColor", "aUv"].indexOf(name),
    getUniformLocation: (_program: unknown, name: string) => name,
    uniformMatrix4fv: (
      location: string,
      _transpose: boolean,
      value: Float32Array,
    ) => {
      // `getUniformLocation` below hands back the uniform's own name, so the
      // two matrices are told apart by the name the component asked for.
      if (location === "uModelView") modelViews.push(Array.from(value));
      if (location === "uProjection") projections.push(Array.from(value));
    },
    uniform1f: (location: string, value: number) => {
      log.push(`uniform1f:${location}:${value}`);
    },
    enable: (cap: number) => {
      log.push(`enable:${GL_NAMES.get(cap) ?? cap}`);
    },
    disable: (cap: number) => {
      log.push(`disable:${GL_NAMES.get(cap) ?? cap}`);
    },
    bufferData: (target: number, data: ArrayBufferView) => {
      if (target === GL_CONSTANTS["ELEMENT_ARRAY_BUFFER"]) {
        log.push(`indexData:${(data as unknown as { length: number }).length}`);
      }
    },
    bindBuffer: (target: number, buffer: object | null) => {
      if (target !== GL_CONSTANTS["ELEMENT_ARRAY_BUFFER"] || !buffer) return;
      let name = named.get(buffer);
      if (!name) {
        name = `index${named.size}`;
        named.set(buffer, name);
      }
      log.push(`bindIndex:${name}`);
    },
    drawElements: (mode: number, count: number) => {
      log.push(`draw:${GL_NAMES.get(mode) ?? mode}:${count}`);
    },
    ...extra,
  };

  const gl = new Proxy(
    {},
    {
      get(_target, property) {
        if (typeof property !== "string") return undefined;
        if (property in overrides) return overrides[property];
        if (property in GL_CONSTANTS) return GL_CONSTANTS[property];
        return () => undefined;
      },
    },
  );

  const getContext = vi
    .spyOn(HTMLCanvasElement.prototype, "getContext")
    .mockImplementation(((kind: string) =>
      kind === "webgl2" ? gl : null) as never);

  return {
    log,
    modelViews,
    projections,
    restore: () => getContext.mockRestore(),
  };
}

interface RafHarness {
  cancelled: number[];
  pending: () => number;
  run: (timestamp: number) => void;
}

function stubRaf(): RafHarness {
  let next = 1;
  const queue = new Map<number, FrameRequestCallback>();
  const cancelled: number[] = [];
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    const id = next;
    next += 1;
    queue.set(id, callback);
    return id;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    cancelled.push(id);
    queue.delete(id);
  });
  return {
    cancelled,
    pending: () => queue.size,
    run(timestamp: number) {
      const due = [...queue.values()];
      queue.clear();
      for (const callback of due) callback(timestamp);
    },
  };
}

interface ReducedMotionHarness {
  /** Flip the preference and tell every listener, the way a browser does. */
  set: (reduce: boolean) => void;
  listeners: () => number;
}

function stubReducedMotion(reduce: boolean): ReducedMotionHarness {
  let current = reduce;
  const listeners = new Set<(event: { matches: boolean }) => void>();
  vi.stubGlobal("matchMedia", (query: string) => {
    const reducedMotion = query.includes("prefers-reduced-motion");
    return {
      get matches() {
        return reducedMotion && current;
      },
      media: query,
      addEventListener: (
        type: string,
        listener: (event: { matches: boolean }) => void,
      ) => {
        if (type === "change" && reducedMotion) listeners.add(listener);
      },
      removeEventListener: (
        type: string,
        listener: (event: { matches: boolean }) => void,
      ) => {
        if (type === "change") listeners.delete(listener);
      },
    };
  });
  return {
    set(next) {
      current = next;
      for (const listener of listeners) listener({ matches: next });
    },
    listeners: () => listeners.size,
  };
}

/**
 * One triangle carrying every attribute the shader binds — positions, normals,
 * colors and UVs — so `attribute()` builds four buffers and the index buffer
 * is the FIFTH `createBuffer` call.
 */
function fullyAttributedGlb(): ArrayBuffer {
  const { json, bin } = buildDocument({
    positions: [0, 0, 0, 2, 0, 0, 0, 4, -1],
    indices: [0, 1, 2],
    normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
    colors: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    uvs: [0, 0, 1, 0, 0, 1],
  });
  return assemble(json, bin);
}

/** A unit cube centred on the origin: something with a real silhouette to tilt. */
function boxGlb(): ArrayBuffer {
  const positions: number[] = [];
  for (const x of [-0.5, 0.5])
    for (const y of [-0.5, 0.5])
      for (const z of [-0.5, 0.5]) positions.push(x, y, z);
  const { json, bin } = buildDocument({
    positions,
    indices: [
      0, 1, 3, 0, 3, 2, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0,
      2, 6, 0, 6, 4, 1, 3, 7, 1, 7, 5,
    ],
  });
  return assemble(json, bin);
}

/** The eight corners of `boxGlb`, in the order it writes them. */
function boxCorners(): [number, number, number][] {
  const corners: [number, number, number][] = [];
  for (const x of [-0.5, 0.5])
    for (const y of [-0.5, 0.5])
      for (const z of [-0.5, 0.5]) corners.push([x, y, z]);
  return corners;
}

/** A well-formed GLB whose one triangle is fully degenerate: no edges at all. */
function edgelessGlb(): ArrayBuffer {
  const { json, bin } = buildDocument({
    positions: [0, 0, 0, 2, 0, 0, 0, 4, -1],
    indices: [1, 1, 1],
  });
  return assemble(json, bin);
}

function stubFullscreen(enabled: boolean): { element: Element | null } {
  const state: { element: Element | null } = { element: null };
  Object.defineProperty(document, "fullscreenEnabled", {
    configurable: true,
    get: () => enabled,
  });
  Object.defineProperty(document, "fullscreenElement", {
    configurable: true,
    get: () => state.element,
  });
  return state;
}

afterEach(() => {
  Reflect.deleteProperty(document, "fullscreenEnabled");
  Reflect.deleteProperty(document, "fullscreenElement");
});

describe("MeshViewer viewport controls", () => {
  it("offers a wireframe toggle whose aria-pressed follows the overlay", async () => {
    const gl = stubWebgl();
    const raf = stubRaf();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();

    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("ready");
    const button = wrapper.get("[data-test=mesh-viewer-wireframe]");
    expect(button.attributes("aria-pressed")).toBe("false");
    expect(button.text()).toContain("Wireframe");
    expect(gl.log.some((entry) => entry.startsWith("draw:LINES"))).toBe(false);

    await button.trigger("click");
    raf.run(16);
    expect(
      wrapper
        .get("[data-test=mesh-viewer-wireframe]")
        .attributes("aria-pressed"),
    ).toBe("true");
    // One triangle: three deduplicated edges, six indices, drawn as LINES over
    // a solid pass that had polygon offset on so the lines cannot z-fight.
    expect(gl.log).toContain("indexData:6");
    expect(gl.log).toContain("enable:POLYGON_OFFSET_FILL");
    expect(gl.log).toContain("draw:LINES:6");

    await wrapper.get("[data-test=mesh-viewer-wireframe]").trigger("click");
    expect(
      wrapper
        .get("[data-test=mesh-viewer-wireframe]")
        .attributes("aria-pressed"),
    ).toBe("false");

    gl.restore();
    wrapper.unmount();
  });

  it("builds the edge buffer only on the first wireframe toggle", async () => {
    const gl = stubWebgl();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();

    const edgeUploads = (): number =>
      gl.log.filter((entry) => entry === "indexData:6").length;
    expect(edgeUploads()).toBe(0);
    await wrapper.get("[data-test=mesh-viewer-wireframe]").trigger("click");
    expect(edgeUploads()).toBe(1);
    await wrapper.get("[data-test=mesh-viewer-wireframe]").trigger("click");
    await wrapper.get("[data-test=mesh-viewer-wireframe]").trigger("click");
    expect(edgeUploads()).toBe(1);

    gl.restore();
    wrapper.unmount();
  });

  it("hides the fullscreen button without the prop or without the API", async () => {
    const gl = stubWebgl();
    stubFullscreen(true);
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();
    expect(wrapper.find("[data-test=mesh-viewer-fullscreen]").exists()).toBe(
      false,
    );
    wrapper.unmount();

    stubFullscreen(false);
    const refused = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", expandable: true },
    });
    await flushPromises();
    expect(refused.find("[data-test=mesh-viewer-fullscreen]").exists()).toBe(
      false,
    );

    gl.restore();
    refused.unmount();
  });

  it("enters fullscreen and reflects the browser's own state change", async () => {
    const gl = stubWebgl();
    const state = stubFullscreen(true);
    const request = vi.fn(() => Promise.resolve());
    const exit = vi.fn(() => Promise.resolve());
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      writable: true,
      value: request,
    });
    Object.defineProperty(document, "exitFullscreen", {
      configurable: true,
      writable: true,
      value: exit,
    });
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", expandable: true },
      attachTo: document.body,
    });
    await flushPromises();

    const button = wrapper.get("[data-test=mesh-viewer-fullscreen]");
    expect(button.attributes("aria-label")).toBe("Enter fullscreen");
    await button.trigger("click");
    await flushPromises();
    expect(request).toHaveBeenCalledTimes(1);

    state.element = wrapper.get("[data-test=mesh-viewer]").element;
    document.dispatchEvent(new Event("fullscreenchange"));
    await flushPromises();
    expect(
      wrapper
        .get("[data-test=mesh-viewer-fullscreen]")
        .attributes("aria-label"),
    ).toBe("Exit fullscreen");

    await wrapper.get("[data-test=mesh-viewer-fullscreen]").trigger("click");
    await flushPromises();
    expect(exit).toHaveBeenCalledTimes(1);

    Reflect.deleteProperty(HTMLElement.prototype, "requestFullscreen");
    Reflect.deleteProperty(document, "exitFullscreen");
    gl.restore();
    wrapper.unmount();
  });
});

describe("MeshViewer upload failures", () => {
  // `attribute()` has built four buffers and `link()` a program by the time
  // the index buffer is requested; a refusal there must hand all of them back
  // rather than leaving them on a context the poster is about to cover.
  it("releases the program and attribute buffers when the index buffer is refused", async () => {
    let created = 0;
    const gl = stubWebgl({
      createBuffer: () => {
        created += 1;
        return created === 5 ? null : {};
      },
    });
    stubFetch(() => ok(fullyAttributedGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", poster: "/media/mesh.png" },
    });
    await flushPromises();

    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("failed");
    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toBe(
      "The GPU refused the mesh's index buffer.",
    );
    expect(gl.log.filter((entry) => entry === "deleteBuffer")).toHaveLength(4);
    expect(gl.log.filter((entry) => entry === "deleteProgram")).toHaveLength(1);
    expect(wrapper.emitted("fail")).toHaveLength(1);

    gl.restore();
    wrapper.unmount();
  });
});

describe("MeshViewer caption", () => {
  // Every surface writes the same `tris · verts · bounds` line under a mesh;
  // the viewer's own caption is that line, not a second wording of it.
  it("captions the mesh with the shared stats label, bounds included", async () => {
    const gl = stubWebgl();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();

    const parsed = parseGlb(triangleGlb());
    const expected = meshStatsLabel(
      parsed.vertexCount,
      parsed.triangleCount,
      parsed.bounds,
    );
    expect(expected).toContain("tris");
    expect(expected).toContain("×");
    expect(wrapper.get("[data-test=mesh-viewer-stats]").text()).toBe(expected);
    expect(wrapper.emitted("ready")?.[0]?.[0]).toMatchObject({
      vertexCount: parsed.vertexCount,
      triangleCount: parsed.triangleCount,
      bounds: parsed.bounds,
    });

    gl.restore();
    wrapper.unmount();
  });
});

describe("MeshViewer wireframe availability", () => {
  it("disables the wireframe toggle, with a reason, for a mesh with no edges", async () => {
    const gl = stubWebgl();
    stubFetch(() => ok(edgelessGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();

    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("ready");
    const button = wrapper.get("[data-test=mesh-viewer-wireframe]");
    expect(button.attributes("disabled")).toBeDefined();
    expect(button.attributes("title")).toContain("no edges");
    await button.trigger("click");
    expect(button.attributes("aria-pressed")).toBe("false");
    expect(gl.log.some((entry) => entry.startsWith("draw:LINES"))).toBe(false);

    gl.restore();
    wrapper.unmount();
  });

  it("keeps the toggle enabled and untitled for an ordinary mesh", async () => {
    const gl = stubWebgl();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();
    const button = wrapper.get("[data-test=mesh-viewer-wireframe]");
    expect(button.attributes("disabled")).toBeUndefined();
    expect(button.attributes("title")).toBeUndefined();
    gl.restore();
    wrapper.unmount();
  });
});

describe("MeshViewer auto-rotation", () => {
  it("yaws the camera on every frame while nobody has touched it", async () => {
    const gl = stubWebgl();
    stubReducedMotion(false);
    const raf = stubRaf();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", autoRotate: true },
    });
    await flushPromises();

    const settled = gl.modelViews.length;
    expect(settled).toBeGreaterThan(0);
    raf.run(0);
    raf.run(1000);
    expect(gl.modelViews.length).toBeGreaterThan(settled);
    expect(gl.modelViews.at(-1)).not.toEqual(gl.modelViews[settled - 1]);
    expect(
      wrapper.get("[data-test=mesh-viewer-canvas]").attributes("aria-label"),
    ).toContain("rotat");

    gl.restore();
    wrapper.unmount();
  });

  it("never starts when the viewer asked for reduced motion", async () => {
    const gl = stubWebgl();
    stubReducedMotion(true);
    const raf = stubRaf();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", autoRotate: true },
    });
    await flushPromises();

    const settled = gl.modelViews.length;
    raf.run(0);
    raf.run(1000);
    raf.run(2000);
    expect(gl.modelViews.length).toBe(settled);

    gl.restore();
    wrapper.unmount();
  });

  it("stops for good at the first pointer interaction", async () => {
    const gl = stubWebgl();
    stubReducedMotion(false);
    const raf = stubRaf();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", autoRotate: true },
    });
    await flushPromises();
    raf.run(0);
    raf.run(1000);

    await wrapper
      .get("[data-test=mesh-viewer-canvas]")
      .trigger("pointerdown", { pointerId: 1, clientX: 4, clientY: 4 });
    const stopped = gl.modelViews.length;
    raf.run(2000);
    raf.run(3000);
    raf.run(4000);
    expect(gl.modelViews.length).toBe(stopped);

    gl.restore();
    wrapper.unmount();
  });

  it("parks the tour the moment the viewer turns reduced motion on", async () => {
    const gl = stubWebgl();
    const motion = stubReducedMotion(false);
    const raf = stubRaf();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", autoRotate: true },
    });
    await flushPromises();
    raf.run(0);
    raf.run(1000);
    expect(motion.listeners()).toBe(1);

    motion.set(true);
    await flushPromises();
    const parked = gl.modelViews.length;
    raf.run(2000);
    raf.run(3000);
    expect(gl.modelViews.length).toBe(parked);
    expect(
      wrapper.get("[data-test=mesh-viewer-canvas]").attributes("aria-label"),
    ).not.toContain("rotat");

    // Turning the preference back off resumes the tour for an untouched
    // viewer — the person never interacted, so it is still on offer.
    motion.set(false);
    raf.run(4000);
    raf.run(5000);
    expect(gl.modelViews.length).toBeGreaterThan(parked);

    gl.restore();
    wrapper.unmount();
    expect(motion.listeners()).toBe(0);
  });

  it("cancels the auto-rotate frame on unmount", async () => {
    const gl = stubWebgl();
    stubReducedMotion(false);
    const raf = stubRaf();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", autoRotate: true },
    });
    await flushPromises();
    expect(raf.pending()).toBeGreaterThan(0);

    wrapper.unmount();
    expect(raf.cancelled.length).toBeGreaterThan(0);
    expect(raf.pending()).toBe(0);
    gl.restore();
  });
});

/*
 * The home view is a CONTRACT, not a preference: the gallery thumbnail is the
 * server's poster, turntable frame 0 is the same pixels, and this viewer's
 * first frame has to be that picture too. These assertions read the matrices
 * the component actually hands the GPU and rebuild them from the shared
 * `meshViewerCamera` definitions the server's `poster.rs` is pinned against.
 */
describe("MeshViewer camera parity", () => {
  const HOME = homeCamera();

  /** The fixture's bounding-box centre, radius and sweep extent. */
  function fixtureFrame(): {
    center: [number, number, number];
    radius: number;
    extent: number;
  } {
    const mesh = parseGlb(triangleGlb());
    const { min, max } = mesh.bounds;
    const center: [number, number, number] = [
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    ];
    const radius = Math.max(
      Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2,
      1e-4,
    );
    return {
      center,
      radius,
      extent: sweepExtent(mesh.positions, center, HOME.pitch),
    };
  }

  /** The box fixture's own profile and poster-elevation extent. */
  function boxFrame(): { profile: Float32Array; extent: number } {
    const mesh = parseGlb(boxGlb());
    const profile = sweepProfile(mesh.positions, [0, 0, 0]);
    return { profile, extent: sweepExtentOfProfile(profile, HOME.pitch) };
  }

  /** `T(0,0,-d) · Rx(pitch) · Ry(yaw) · T(-centre)`, the viewer's modelView. */
  function expectedModelView(
    yaw: number,
    pitch: number,
    distance: number,
    center: [number, number, number],
  ): Float32Array {
    return multiply(
      multiply(
        translation(0, 0, -distance),
        multiply(rotationX(pitch), rotationY(yaw)),
      ),
      translation(-center[0], -center[1], -center[2]),
    );
  }

  function expectMatrix(actual: number[] | undefined, expected: Float32Array) {
    expect(actual).toBeDefined();
    for (let i = 0; i < 16; i += 1) {
      expect(actual?.[i]).toBeCloseTo(expected[i] ?? 0, 5);
    }
  }

  it("projects orthographically, framed to the sweep extent at the poster margin", async () => {
    const gl = stubWebgl();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();

    const canvas = wrapper.get("[data-test=mesh-viewer-canvas]")
      .element as HTMLCanvasElement;
    const { extent } = fixtureFrame();
    const scale = orthographicScale(
      extent,
      canvas.width,
      canvas.height,
      POSTER_MARGIN,
    );
    expect(scale).toBeGreaterThan(0);

    const projection = gl.projections[0];
    expect(projection).toBeDefined();
    // No perspective divide anywhere in the matrix.
    expect(projection?.[11]).toBe(0);
    expect(projection?.[15]).toBe(1);
    // The half-extents ARE the fit: `1 / halfWidth` down the diagonal.
    expect(projection?.[0]).toBeCloseTo(1 / (canvas.width / 2 / scale), 5);
    expect(projection?.[5]).toBeCloseTo(1 / (canvas.height / 2 / scale), 5);
    // The margin the server leaves is the margin drawn here: the mesh's
    // extent lands at `1 - POSTER_MARGIN` of the half-frame.
    expect(extent * scale).toBeCloseTo(
      (Math.min(canvas.width, canvas.height) / 2) * (1 - POSTER_MARGIN),
      5,
    );

    gl.restore();
    wrapper.unmount();
  });

  it("opens on the poster's own camera", async () => {
    const gl = stubWebgl();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();

    const { center, radius } = fixtureFrame();
    expectMatrix(
      gl.modelViews[0],
      expectedModelView(HOME.yaw, HOME.pitch, radius * 3, center),
    );

    gl.restore();
    wrapper.unmount();
  });

  it("turns the mesh by the drag distance and comes back home on 0", async () => {
    const gl = stubWebgl();
    const raf = stubRaf();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();

    const { center, radius } = fixtureFrame();
    const canvas = wrapper.get("[data-test=mesh-viewer-canvas]");
    await canvas.trigger("pointerdown", {
      pointerId: 1,
      clientX: 0,
      clientY: 0,
    });
    await canvas.trigger("pointermove", {
      pointerId: 1,
      clientX: 50,
      clientY: 0,
    });
    raf.run(16);
    // +50 px to the right at 0.008 rad/px: the yaw rises by 0.4, which lowers
    // the server-frame azimuth — the direction the turntable now sweeps.
    expectMatrix(
      gl.modelViews.at(-1),
      expectedModelView(HOME.yaw + 0.4, HOME.pitch, radius * 3, center),
    );

    await canvas.trigger("keydown", { key: "0" });
    raf.run(32);
    expectMatrix(
      gl.modelViews.at(-1),
      expectedModelView(HOME.yaw, HOME.pitch, radius * 3, center),
    );

    gl.restore();
    wrapper.unmount();
  });

  it("re-frames as the mesh tilts, so a 45-degree view is never clipped", async () => {
    const gl = stubWebgl();
    const raf = stubRaf();
    stubFetch(() => ok(boxGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/box.glb" } });
    await flushPromises();

    const home = gl.projections[0];
    expect(home).toBeDefined();
    const homeHalfHeight = 1 / (home?.[5] ?? 1);

    // `orbit(0, dy * 0.008)`: the pixels that take the poster's 20° to 45°.
    const pixels = (Math.PI / 4 - HOME.pitch) / 0.008;
    const pitch = HOME.pitch + pixels * 0.008;
    const canvas = wrapper.get("[data-test=mesh-viewer-canvas]");
    await canvas.trigger("pointerdown", {
      pointerId: 1,
      clientX: 0,
      clientY: 0,
    });
    await canvas.trigger("pointermove", {
      pointerId: 1,
      clientX: 0,
      clientY: pixels,
    });
    raf.run(16);

    const tilted = gl.projections.at(-1);
    const modelView = gl.modelViews.at(-1);
    expect(tilted).toBeDefined();
    expect(modelView).toBeDefined();
    const halfWidth = 1 / (tilted?.[0] ?? 1);
    const halfHeight = 1 / (tilted?.[5] ?? 1);
    const view = Float32Array.from(modelView ?? []);

    let tallest = 0;
    for (const corner of boxCorners()) {
      const projected = [0, 1].map((row) => {
        let sum = view[12 + row] ?? 0;
        for (let k = 0; k < 3; k += 1) {
          sum += (view[k * 4 + row] ?? 0) * (corner[k] ?? 0);
        }
        return sum;
      });
      // Nothing runs off the frame at the tilt the person is holding.
      expect(Math.abs(projected[0] ?? 0)).toBeLessThanOrEqual(halfWidth + 1e-5);
      expect(Math.abs(projected[1] ?? 0)).toBeLessThanOrEqual(
        halfHeight + 1e-5,
      );
      tallest = Math.max(tallest, Math.abs(projected[1] ?? 0));
    }
    // Non-vacuous: the poster's own framing WOULD have clipped this view, so
    // the assertions above are describing the re-frame and not a coincidence.
    expect(tallest).toBeGreaterThan(homeHalfHeight);
    expect(halfHeight).toBeGreaterThan(homeHalfHeight);
    // The tilt only ever pulls back, and by the sweep bound at that pitch.
    const { extent } = boxFrame();
    expect(extent).toBeCloseTo(0.7117, 4);
    const tiltedExtent = sweepExtentOfProfile(boxFrame().profile, pitch);
    expect(halfHeight).toBeCloseTo(
      160 / orthographicScale(tiltedExtent, 320, 320, POSTER_MARGIN),
      4,
    );

    // Home is still the poster's exact framing after a round trip.
    await canvas.trigger("keydown", { key: "0" });
    raf.run(32);
    const restored = gl.projections.at(-1);
    for (let i = 0; i < 16; i += 1) {
      expect(restored?.[i]).toBeCloseTo(home?.[i] ?? 0, 6);
    }

    gl.restore();
    wrapper.unmount();
  });

  it("keeps the zoom keys pointing the way they always did", async () => {
    const gl = stubWebgl();
    const raf = stubRaf();
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    await flushPromises();

    const canvas = wrapper.get("[data-test=mesh-viewer-canvas]");
    const homeHalfWidth = 1 / (gl.projections[0]?.[0] ?? 1);

    await canvas.trigger("keydown", { key: "+" });
    raf.run(16);
    const zoomedIn = 1 / (gl.projections.at(-1)?.[0] ?? 1);
    // Zooming in shows LESS of the world, so the half-extent shrinks.
    expect(zoomedIn).toBeLessThan(homeHalfWidth);

    await canvas.trigger("keydown", { key: "-" });
    await canvas.trigger("keydown", { key: "-" });
    raf.run(32);
    expect(1 / (gl.projections.at(-1)?.[0] ?? 1)).toBeGreaterThan(
      homeHalfWidth,
    );

    gl.restore();
    wrapper.unmount();
  });
});
