import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Lightbox from "./Lightbox.vue";
import { __resetGalleryMediaForTests } from "../../lib/galleryMedia";
import type { GalleryImage } from "../../types";

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

function mockCapabilities(exportFormats: string[]) {
  requests = [];
  globalThis.fetch = vi.fn(async (url: string, init?: RequestInit) => {
    requests.push({ url, init });
    if (url.endsWith("/api/capabilities")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          mesh: {
            generation: true,
            formats: ["glb"],
            export_formats: exportFormats,
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

  it("does not probe capabilities for a raster print", async () => {
    mockCapabilities(["obj"]);
    mountWide({ item: still });
    await flushPromises();
    expect(
      requests.some((request) => request.url.endsWith("/api/capabilities")),
    ).toBe(false);
  });
});
