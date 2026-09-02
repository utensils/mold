import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { triangleGlb } from "../lib/glbFixture";
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
  restore: () => void;
}

function stubWebgl(): GlRecorder {
  const log: string[] = [];
  const modelViews: number[][] = [];
  const named = new Map<object, string>();
  const overrides: Record<string, unknown> = {
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
      if (location === "uModelView") modelViews.push(Array.from(value));
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

  return { log, modelViews, restore: () => getContext.mockRestore() };
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

function stubReducedMotion(reduce: boolean): void {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: reduce && query.includes("prefers-reduced-motion"),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
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
