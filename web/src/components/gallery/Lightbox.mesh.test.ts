import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Lightbox from "./Lightbox.vue";
import { __resetGalleryMediaForTests } from "../../lib/galleryMedia";
import { __testing__ as routingTesting } from "../../composables/useHostRouting";
import { ORIGIN_HOST_ID } from "../../lib/hostRegistry";
import type { GalleryImage, ServerCapabilities } from "../../types";

/**
 * A `.glb` print viewed in the lightbox: no "Use as source" (the engines read
 * pixels, not geometry), and an export menu built from the HOST's advertised
 * `mesh.export_formats` — never a client constant, so a host that later adds
 * an animated turntable container offers it with no web release.
 */

const mesh: GalleryImage = {
  filename: "chair.glb",
  timestamp: 1_700_000_000,
  format: "glb",
  metadata: {
    prompt: "a wooden chair",
    model: "hunyuan3d-mini-turbo:fp16",
    seed: 7,
    steps: 5,
    guidance: 5,
    width: 512,
    height: 512,
    version: "test",
  },
};

const still: GalleryImage = { ...mesh, filename: "still.png", format: "png" };
const secondMesh: GalleryImage = { ...mesh, filename: "table.glb" };

function setViewportWidth(px: number) {
  Object.defineProperty(window, "innerWidth", {
    value: px,
    configurable: true,
    writable: true,
  });
}

const MeshViewerStub = {
  name: "MeshViewer",
  props: {
    src: { type: String, default: "" },
    poster: { type: String, default: "" },
    alt: { type: String, default: "" },
    autoRotate: { type: Boolean, default: false },
    expandable: { type: Boolean, default: false },
  },
  template: "<div data-test='mesh-viewer-stub' />",
};

function mountNarrow(props: Record<string, unknown> = {}) {
  setViewportWidth(480);
  return mount(Lightbox, {
    props: {
      item: mesh,
      index: 0,
      total: 1,
      hasPrev: false,
      hasNext: false,
      muted: true,
      ...props,
    },
    global: { stubs: { Transition: false, MeshViewer: MeshViewerStub } },
  });
}

function mountWide(props: Record<string, unknown> = {}) {
  setViewportWidth(1200);
  return mount(Lightbox, {
    props: {
      item: mesh,
      index: 0,
      total: 1,
      hasPrev: false,
      hasNext: false,
      muted: true,
      ...props,
    },
    global: { stubs: { Transition: false, MeshViewer: MeshViewerStub } },
  });
}

const originalFetch = globalThis.fetch;
let requests: { url: string; init?: RequestInit }[] = [];

function mockCapabilities(
  exportFormats: string[],
  exportResponse: () => Promise<unknown> = async () => ({
    ok: true,
    status: 200,
    blob: async () => new Blob(["MESH"], { type: "model/obj" }),
  }),
  /** `mesh.export_geometry`; omitted entirely for an older host. */
  exportGeometry?: unknown,
) {
  requests = [];
  globalThis.fetch = vi.fn(async (url: string, init?: RequestInit) => {
    requests.push({ url, init });
    if (init?.method === "POST") return exportResponse();
    if (url.endsWith("/api/capabilities")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          mesh: {
            generation: true,
            formats: ["glb"],
            export_formats: exportFormats,
            ...(exportGeometry ? { export_geometry: exportGeometry } : {}),
            textures: false,
          },
        }),
      };
    }
    return {
      ok: true,
      status: 200,
      blob: async () => new Blob(["MESH"], { type: "model/obj" }),
    };
  }) as never;
}

beforeEach(() => {
  localStorage.clear();
  __resetGalleryMediaForTests();
  routingTesting.reset();
  setViewportWidth(1200);
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("Lightbox 3-D prints", () => {
  it("refuses Use as source for a mesh print", async () => {
    mockCapabilities([]);
    const wrapper = mountWide();
    await flushPromises();
    const button = wrapper
      .findAll("button")
      .find((candidate) => candidate.text() === "Use as source")!;
    expect(button.attributes("disabled")).toBeDefined();
    await button.trigger("click");
    expect(wrapper.emitted("use-source")).toBeUndefined();
  });

  // A mesh has no raster to upscale: the overflow never offers it, and the
  // handler is a no-op even if something reaches it.
  it("offers no Upscale entry for a mesh print", async () => {
    mockCapabilities([]);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    const labels = wrapper
      .findAll("[role='menuitem']")
      .map((entry) => entry.text());
    expect(labels.some((label) => /upscale/i.test(label))).toBe(false);
    expect(wrapper.emitted("upscale")).toBeUndefined();
  });

  it("keeps the Upscale entry for a raster print", async () => {
    mockCapabilities([]);
    const wrapper = mountWide({ item: still });
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    const labels = wrapper
      .findAll("[role='menuitem']")
      .map((entry) => entry.text());
    expect(labels.some((label) => /upscale/i.test(label))).toBe(true);
  });

  it("keeps Use as source for a raster print", async () => {
    mockCapabilities([]);
    const wrapper = mountWide({ item: still });
    await flushPromises();
    const button = wrapper
      .findAll("button")
      .find((candidate) => candidate.text() === "Use as source")!;
    expect(button.attributes("disabled")).toBeUndefined();
  });

  it("builds the export menu from the host's advertised formats", async () => {
    mockCapabilities(["obj", "stl", "ply"]);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    const labels = wrapper
      .findAll("[data-test^='mesh-export-']")
      .map((entry) => entry.text());
    expect(labels).toEqual([
      "Export as OBJ…",
      "Export as STL…",
      "Export as PLY…",
    ]);
  });

  it("offers nothing when the host advertises no mesh exports", async () => {
    mockCapabilities([]);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    expect(wrapper.findAll("[data-test^='mesh-export-']")).toHaveLength(0);
  });

  it("posts the chosen format and downloads the returned bytes", async () => {
    mockCapabilities(["obj", "stl"]);
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    const createObjectURL = vi.fn(() => "blob:export");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL,
      revokeObjectURL,
    });
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();

    const post = requests.find((request) => request.init?.method === "POST")!;
    expect(post.url).toBe("/api/gallery/export/chair.glb");
    expect(JSON.parse(String(post.init!.body))).toEqual({ format: "stl" });
    expect(anchorClick).toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it("shows the host's own refusal and clears it on the next print", async () => {
    mockCapabilities(["obj"], async () => ({
      ok: false,
      status: 422,
      json: async () => ({ error: "This print has no geometry to convert." }),
    }));
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-obj']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='mesh-export-error']").text()).toBe(
      "This print has no geometry to convert.",
    );

    // Arrowing on: the refusal belonged to the print that is gone.
    await wrapper.setProps({ item: secondMesh });
    await flushPromises();
    expect(wrapper.find("[data-test='mesh-export-error']").exists()).toBe(
      false,
    );
  });

  // A mesh transcode is not a video export; a bare status must not say so.
  it("falls back to a neutral export failure when the host sends no reason", async () => {
    mockCapabilities(["obj"], async () => ({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error("not json");
      },
    }));
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-obj']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='mesh-export-error']").text()).toBe(
      "Export failed (500)",
    );
  });

  it("routes an advertised animated container through the options sheet", async () => {
    mockCapabilities(["obj", "gif", "webp"]);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    // One entry for the animation family, not one per container.
    expect(wrapper.findAll("[data-test^='mesh-export-']").length).toBe(2);
    await wrapper.get("[data-test='mesh-export-animation']").trigger("click");
    await flushPromises();
    const dialog = wrapper.getComponent({ name: "VideoExportDialog" });
    expect(dialog.props("open")).toBe(true);
    expect(dialog.props("formats")).toEqual(["gif", "webp"]);
  });

  // The server lists the stored container first so a client can see what it
  // holds; "Export as GLB" beside Download is not an export.
  it("never offers the stored glb as an export", async () => {
    mockCapabilities(["glb", "obj", "gif"]);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    const labels = wrapper
      .findAll("[data-test^='mesh-export-']")
      .map((entry) => entry.text());
    expect(labels).toEqual(["Export as OBJ…", "Export turntable…"]);
  });

  // The shell already polls every host's capabilities; arrowing between
  // prints must reuse that snapshot rather than probe the host per step.
  it("builds the menu from the routing snapshot without probing the host", async () => {
    mockCapabilities([]);
    routingTesting.seedCapabilities(ORIGIN_HOST_ID, {
      mesh: {
        generation: true,
        formats: ["glb"],
        export_formats: ["glb", "stl", "ply"],
        textures: false,
      },
    } as ServerCapabilities);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.setProps({ item: secondMesh });
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    const labels = wrapper
      .findAll("[data-test^='mesh-export-']")
      .map((entry) => entry.text());
    expect(labels).toEqual(["Export as STL…", "Export as PLY…"]);
    expect(
      requests.some((request) => request.url.endsWith("/api/capabilities")),
    ).toBe(false);
  });

  it("does not probe capabilities for a raster print", async () => {
    mockCapabilities(["obj"]);
    mountWide({ item: still });
    await flushPromises();
    expect(
      requests.some((request) => request.url.endsWith("/api/capabilities")),
    ).toBe(false);
  });
});

describe("Lightbox 3-D prints (mobile full-screen)", () => {
  it("offers no Upscale entry for a mesh print", async () => {
    mockCapabilities(["obj"]);
    const wrapper = mountNarrow();
    await flushPromises();
    expect(wrapper.find(".lb__full").exists()).toBe(true);
    await wrapper.get("[aria-label='More actions']").trigger("click");
    const labels = wrapper
      .findAll("[role='menuitem']")
      .map((entry) => entry.text());
    expect(labels.some((label) => /upscale/i.test(label))).toBe(false);
    expect(labels).toContain("Export as OBJ…");
  });

  it("keeps the Upscale entry for a raster print", async () => {
    mockCapabilities([]);
    const wrapper = mountNarrow({ item: still });
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    const labels = wrapper
      .findAll("[role='menuitem']")
      .map((entry) => entry.text());
    expect(labels.some((label) => /upscale/i.test(label))).toBe(true);
  });

  it("shows an export refusal in the bottom sheet", async () => {
    mockCapabilities(["obj"], async () => ({
      ok: false,
      status: 422,
      json: async () => ({ error: "This print has no geometry to convert." }),
    }));
    const wrapper = mountNarrow();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-obj']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='mesh-export-error']").text()).toBe(
      "This print has no geometry to convert.",
    );
  });
});

/**
 * Geometry options. A stored mesh is normalized model space, so an STL sent
 * to a slicer verbatim is a 2 mm blob on its side. A host that advertises
 * `mesh.export_geometry` lets the lightbox ask for a size, an up axis and an
 * origin; a host that does not gets exactly the body this client always sent,
 * because an older server DROPS unknown keys instead of refusing them.
 */
const GEOMETRY = {
  size_mm: { min: 1, max: 1000, default: 100 },
  up_axes: ["y", "z"],
  origins: ["center", "floor"],
  defaults: {
    obj: { size_mm: null, up_axis: "y", origin: "floor" },
    stl: { size_mm: 100, up_axis: "z", origin: "floor" },
    ply: { size_mm: 100, up_axis: "z", origin: "floor" },
  },
};

describe("Lightbox 3-D geometry options", () => {
  it("opens the options sheet for a container the host will scale", async () => {
    mockCapabilities(["obj", "stl", "gif"], undefined, GEOMETRY);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();

    const dialog = wrapper.get("[data-test='mesh-export-dialog']");
    expect(dialog.text()).toContain("Export as STL");
    // Nothing is posted until the sheet is submitted.
    expect(requests.some((request) => request.init?.method === "POST")).toBe(
      false,
    );
  });

  it("posts the three keys the user settled on", async () => {
    mockCapabilities(["stl"], undefined, GEOMETRY);
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn(() => "blob:export"),
      revokeObjectURL: vi.fn(),
    });
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mesh-geometry-size-200']").setValue();
    await wrapper.get("[data-test='mesh-geometry-origin-center']").setValue();
    await wrapper
      .get("[data-test='mesh-export-dialog'] form")
      .trigger("submit");
    await flushPromises();

    const post = requests.find((request) => request.init?.method === "POST")!;
    expect(post.url).toBe("/api/gallery/export/chair.glb");
    expect(JSON.parse(String(post.init!.body))).toEqual({
      format: "stl",
      size_mm: 200,
      up_axis: "z",
      origin: "center",
    });
    expect(anchorClick).toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  // The old-server contract, byte for byte: no block, no options, no keys.
  it("posts the bare format when the host advertises no geometry block", async () => {
    mockCapabilities(["stl"]);
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
      () => undefined,
    );
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn(() => "blob:export"),
      revokeObjectURL: vi.fn(),
    });
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='mesh-export-dialog']").exists()).toBe(
      false,
    );
    const post = requests.find((request) => request.init?.method === "POST")!;
    expect(JSON.parse(String(post.init!.body))).toEqual({ format: "stl" });
    vi.unstubAllGlobals();
  });

  // Geometry belongs to a file, not to a rendered animation.
  it("keeps the turntable on the playback sheet", async () => {
    mockCapabilities(["stl", "gif"], undefined, GEOMETRY);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-animation']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mesh-export-dialog']").exists()).toBe(
      false,
    );
    expect(
      wrapper.getComponent({ name: "VideoExportDialog" }).props("open"),
    ).toBe(true);
  });

  // The host's own defaults table is the authority; a container it does not
  // name there is the one-click transcode it has always been.
  it("posts the bare format for a container the host lists no defaults for", async () => {
    mockCapabilities(["usdz"], undefined, GEOMETRY);
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
      () => undefined,
    );
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn(() => "blob:export"),
      revokeObjectURL: vi.fn(),
    });
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-usdz']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mesh-export-dialog']").exists()).toBe(
      false,
    );
    const post = requests.find((request) => request.init?.method === "POST")!;
    expect(JSON.parse(String(post.init!.body))).toEqual({ format: "usdz" });
    vi.unstubAllGlobals();
  });

  it("reads the geometry block from the routing snapshot without probing", async () => {
    mockCapabilities([]);
    routingTesting.seedCapabilities(ORIGIN_HOST_ID, {
      mesh: {
        generation: true,
        formats: ["glb"],
        export_formats: ["stl"],
        export_geometry: GEOMETRY,
        textures: false,
      },
    } as unknown as ServerCapabilities);
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mesh-export-dialog']").exists()).toBe(
      true,
    );
    expect(
      requests.some((request) => request.url.endsWith("/api/capabilities")),
    ).toBe(false);
  });

  it("shows the host's refusal inside the sheet and leaves it open", async () => {
    mockCapabilities(
      ["stl"],
      async () => ({
        ok: false,
        status: 422,
        json: async () => ({ error: "That size is out of range." }),
      }),
      GEOMETRY,
    );
    const wrapper = mountWide();
    await flushPromises();
    await wrapper.get("[aria-label='More actions']").trigger("click");
    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();
    await wrapper
      .get("[data-test='mesh-export-dialog'] form")
      .trigger("submit");
    await flushPromises();
    const dialog = wrapper.get("[data-test='mesh-export-dialog']");
    expect(dialog.get("[role='alert']").text()).toBe(
      "That size is out of range.",
    );
  });
});
