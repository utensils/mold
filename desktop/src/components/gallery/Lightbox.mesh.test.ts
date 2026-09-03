/**
 * 3-D exports in the Library lightbox. GLB is the only stored form; OBJ / STL
 * / PLY and any animated turntable are TRANSCODES the holding host performs,
 * so the menu is built from that host's own advertised
 * `capabilities.mesh.export_formats` — never a client constant — and each
 * entry posts `{ format }` to `POST /api/gallery/export/:filename` through
 * the same authenticated save path every other download uses.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const { saveGalleryMedia, apiJsonTo } = vi.hoisted(() => ({
  saveGalleryMedia: vi.fn(),
  apiJsonTo: vi.fn(),
}));

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../../lib/ipc", () => ({
  inTauri: () => true,
  ipc: {
    getOutputDir: vi.fn().mockResolvedValue(null),
    revealOutputFile: vi.fn(),
    revealSavedMedia: vi.fn(),
  },
}));
vi.mock("../../lib/mediaSave", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/mediaSave")>()),
  saveGalleryMedia,
}));
vi.mock("../../lib/api/client", () => ({
  currentTarget: vi.fn(() => ({ baseUrl: "http://local", apiKey: "key" })),
  apiJson: vi.fn(),
  apiJsonTo,
}));
// Only the mesh-poster describe block below mounts AuthedMedia unstubbed;
// every other test in this file stubs it out, so these mocks never fire.
vi.mock("../../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/gallery/media")>()),
  authedMediaUrl: vi.fn().mockResolvedValue("blob:poster"),
  fullSizeMediaUrl: vi.fn().mockResolvedValue("blob:mesh-src"),
  prepareNativeThumbnail: vi.fn().mockResolvedValue(null),
}));

import Lightbox from "./Lightbox.vue";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
import MeshExportDialog from "@ui/components/MeshExportDialog.vue";
import MeshViewer from "@studio/components/MeshViewer.vue";
import type { MeshExportGeometryCapabilities } from "@studio/lib/meshExport";
import type { GalleryImage } from "../../lib/api/types";

const meshItem: GalleryImage = {
  filename: "print-0007.glb",
  timestamp: 1_700_000_000,
  format: "glb",
  metadata: {
    prompt: "an armchair",
    model: "hunyuan3d-mini-turbo:fp16",
    seed: 7,
    steps: 5,
    guidance: 5,
    width: 512,
    height: 512,
  },
};

const target = { baseUrl: "http://plato", apiKey: "plato-key" };

beforeEach(() => {
  setActivePinia(createPinia());
  saveGalleryMedia.mockReset().mockResolvedValue({
    filename: "print-0007.obj",
    path: "/Users/test/Downloads/print-0007.obj",
    directory: "Downloads",
  });
  apiJsonTo.mockReset().mockResolvedValue({});
});

/**
 * The holding host's own geometry contract, exactly as
 * `/api/capabilities.mesh.export_geometry` spells it. Its ABSENCE is the only
 * gate: an older host never sees a geometry field.
 */
const geometry: MeshExportGeometryCapabilities = {
  size_mm: { min: 1, max: 1000, default: 100 },
  up_axes: ["y", "z"],
  origins: ["center", "floor"],
  defaults: {
    obj: { size_mm: null, up_axis: "y", origin: "floor" },
    stl: { size_mm: 100, up_axis: "z", origin: "floor" },
    ply: { size_mm: 100, up_axis: "z", origin: "floor" },
  },
};

function mountMesh(
  meshExportFormats: string[],
  meshExportGeometry: MeshExportGeometryCapabilities | null = null,
) {
  return mount(Lightbox, {
    props: {
      item: meshItem,
      index: 0,
      count: 1,
      video: false,
      mesh: true,
      target,
      meshExportFormats,
      meshExportGeometry,
    },
    global: { stubs: { AuthedMedia: { template: "<div />" } } },
  });
}

describe("Lightbox — mesh exports", () => {
  it("builds one entry per advertised geometry container", () => {
    const wrapper = mountMesh(["obj", "stl", "ply"]);
    expect(wrapper.get("[data-test='mesh-export-obj']").text()).toBe("Export as OBJ…");
    expect(wrapper.get("[data-test='mesh-export-stl']").text()).toBe("Export as STL…");
    expect(wrapper.get("[data-test='mesh-export-ply']").text()).toBe("Export as PLY…");
    expect(wrapper.find("[data-test='mesh-export-animation']").exists()).toBe(false);
  });

  it("offers nothing when the holding host advertises no transcodes", () => {
    const wrapper = mountMesh([]);
    expect(wrapper.find("[data-test='mesh-exports']").exists()).toBe(false);
  });

  // The server lists the stored container first (`glb`) so a CLI can name it;
  // Save already hands over that exact file, so "Export as GLB…" beside it
  // would be a no-op transcode.
  it("never offers the stored GLB as an export beside Save", () => {
    const wrapper = mountMesh(["glb", "obj", "stl", "ply", "gif", "apng", "webp"]);
    expect(wrapper.find("[data-test='mesh-export-glb']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mesh-export-obj']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mesh-export-animation']").exists()).toBe(true);
  });

  // The export route reads the live gallery only; a trashed print's bytes are
  // under `.trash`, so every export control disappears with Upscale.
  it("hides every export for a trashed mesh print", () => {
    const wrapper = mount(Lightbox, {
      props: {
        item: meshItem,
        index: 0,
        count: 1,
        video: false,
        mesh: true,
        trashed: true,
        target,
        meshExportFormats: ["glb", "obj", "stl", "ply", "gif", "apng", "webp"],
      },
      global: { stubs: { AuthedMedia: { template: "<div />" } } },
    });
    expect(wrapper.find("[data-test='mesh-exports']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mesh-export-animation']").exists()).toBe(false);
  });

  it("hides Export format… for a trashed clip", () => {
    const wrapper = mount(Lightbox, {
      props: {
        item: { ...meshItem, filename: "clip-0002.mp4", format: "mp4" },
        index: 0,
        count: 1,
        video: true,
        trashed: true,
        target,
      },
      global: { stubs: { AuthedMedia: { template: "<div />" } } },
    });
    expect(wrapper.find("[data-test='export-video']").exists()).toBe(false);
  });

  // A .glb has no raster to stage as conditioning: the primary button must
  // agree with its context-menu twin, or the GLB bytes reach the source well.
  it("disables the primary Use as source for a mesh print", () => {
    const wrapper = mountMesh(["obj"]);
    expect(wrapper.get("[data-test='lightbox-use-source']").attributes()).toHaveProperty(
      "disabled",
    );
  });

  it("keeps the primary Use as source live for a raster print", () => {
    const wrapper = mount(Lightbox, {
      props: {
        item: { ...meshItem, filename: "print-0001.png", format: "png" },
        index: 0,
        count: 1,
        video: false,
        target,
      },
      global: { stubs: { AuthedMedia: { template: "<div />" } } },
    });
    expect(wrapper.get("[data-test='lightbox-use-source']").attributes()).not.toHaveProperty(
      "disabled",
    );
  });

  it("offers only what this host advertises, never a client list", () => {
    const wrapper = mountMesh(["stl"]);
    expect(wrapper.find("[data-test='mesh-export-obj']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mesh-export-stl']").exists()).toBe(true);
  });

  // A mesh has no raster to upscale, so the button is not offered at all.
  it("offers no Upscale for a mesh print", () => {
    const wrapper = mountMesh(["obj"]);
    expect(wrapper.find("[data-test='lightbox-upscale']").exists()).toBe(false);
  });

  it("keeps Upscale for a raster print", () => {
    const wrapper = mount(Lightbox, {
      props: {
        item: { ...meshItem, filename: "print-0001.png", format: "png" },
        index: 0,
        count: 1,
        video: false,
        target,
      },
      global: { stubs: { AuthedMedia: { template: "<div />" } } },
    });
    expect(wrapper.find("[data-test='lightbox-upscale']").exists()).toBe(true);
  });

  it("keeps every export off a raster print", () => {
    const wrapper = mount(Lightbox, {
      props: {
        item: { ...meshItem, filename: "print-0001.png", format: "png" },
        index: 0,
        count: 1,
        video: false,
        target,
        meshExportFormats: ["obj", "stl"],
      },
      global: { stubs: { AuthedMedia: { template: "<div />" } } },
    });
    expect(wrapper.find("[data-test='mesh-exports']").exists()).toBe(false);
  });

  it("posts { format } to the holding host and saves the transcoded name", async () => {
    const wrapper = mountMesh(["obj", "stl", "ply"]);
    await wrapper.get("[data-test='mesh-export-obj']").trigger("click");
    await flushPromises();
    expect(saveGalleryMedia).toHaveBeenCalledTimes(1);
    const [usedTarget, filename, outputName, options] = saveGalleryMedia.mock.calls[0]!;
    expect(usedTarget).toEqual(target);
    expect(filename).toBe("print-0007.glb");
    expect(outputName.endsWith(".obj")).toBe(true);
    expect(options).toEqual({ format: "obj" });
  });

  it("opens the shared export sheet for an animated turntable", async () => {
    const wrapper = mountMesh(["obj", "gif", "webp"]);
    expect(wrapper.findComponent(VideoExportDialog).props("open")).toBe(false);
    await wrapper.get("[data-test='mesh-export-animation']").trigger("click");
    await flushPromises();
    const dialog = wrapper.findComponent(VideoExportDialog);
    expect(dialog.props("open")).toBe(true);
    // Only the containers this host advertises — the sheet never invents one.
    expect(dialog.props("formats")).toEqual(["gif", "webp"]);
    // And no capability probe: the host already told us on connect.
    expect(apiJsonTo).not.toHaveBeenCalled();

    dialog.vm.$emit("export", {
      format: "gif",
      playback: "bounce",
      repeat: "once",
      max_dimension: 512,
      fps: 12,
    });
    await flushPromises();
    expect(saveGalleryMedia).toHaveBeenCalledTimes(1);
    expect(saveGalleryMedia.mock.calls[0]![3]).toMatchObject({
      format: "gif",
      playback: "bounce",
      repeat: "once",
      max_dimension: 512,
    });
  });
});

/**
 * The geometry knobs a stored mesh needs before a slicer can read it. They
 * exist only where the holding host advertises `mesh.export_geometry`; its
 * absence is the ONLY gate, because an older server silently DROPS unknown
 * body fields rather than refusing them — so a client that guessed would post
 * a size the host ignores and hand back the unscaled mesh the user thought
 * they had resized.
 */
describe("Lightbox — mesh export geometry", () => {
  it("opens the geometry sheet for a container the host scales", async () => {
    const wrapper = mountMesh(["obj", "stl", "ply"], geometry);
    expect(wrapper.findComponent(MeshExportDialog).props("open")).toBe(false);

    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();

    const dialog = wrapper.findComponent(MeshExportDialog);
    expect(dialog.props("open")).toBe(true);
    expect(dialog.props("format")).toBe("stl");
    expect(dialog.props("capabilities")).toEqual(geometry);
    // Nothing is exported until the user says so.
    expect(saveGalleryMedia).not.toHaveBeenCalled();
  });

  it("posts the chosen geometry beside the format", async () => {
    const wrapper = mountMesh(["obj", "stl", "ply"], geometry);
    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();
    // The sheet's own submit, so the posted body is the one its draft holds.
    expect(wrapper.find("[data-test='mesh-export-submit']").exists()).toBe(true);
    await wrapper.get("[data-test='mesh-export-dialog'] form").trigger("submit");
    await flushPromises();

    expect(saveGalleryMedia).toHaveBeenCalledTimes(1);
    const [usedTarget, filename, outputName, options] = saveGalleryMedia.mock.calls[0]!;
    expect(usedTarget).toEqual(target);
    expect(filename).toBe("print-0007.glb");
    expect(outputName.endsWith(".stl")).toBe(true);
    // The host's own defaults for STL, which is what the sheet opened on.
    expect(options).toEqual({
      format: "stl",
      size_mm: 100,
      up_axis: "z",
      origin: "floor",
    });
    expect(wrapper.findComponent(MeshExportDialog).props("open")).toBe(false);
  });

  // THE regression that matters: a host that predates the feature must keep
  // receiving the body this client has always sent.
  it("posts exactly { format } to a host that advertises no geometry", async () => {
    const wrapper = mountMesh(["obj", "stl", "ply"]);
    await wrapper.get("[data-test='mesh-export-stl']").trigger("click");
    await flushPromises();

    // No contract, so the sheet is not even mounted.
    expect(wrapper.findComponent(MeshExportDialog).exists()).toBe(false);
    expect(saveGalleryMedia).toHaveBeenCalledTimes(1);
    expect(saveGalleryMedia.mock.calls[0]![3]).toEqual({ format: "stl" });
  });

  // A format the host lists no defaults for is one it does not scale.
  it("exports straight through for a container the host omits from defaults", async () => {
    const wrapper = mountMesh(["obj", "stl", "3mf"], {
      ...geometry,
      defaults: { stl: geometry.defaults.stl! },
    });
    await wrapper.get("[data-test='mesh-export-3mf']").trigger("click");
    await flushPromises();

    expect(wrapper.findComponent(MeshExportDialog).props("open")).toBe(false);
    expect(saveGalleryMedia.mock.calls[0]![3]).toEqual({ format: "3mf" });
  });

  // A turntable is a render with playback options, not a geometry file: it
  // keeps the video sheet even where geometry is advertised.
  it("keeps the turntable on the playback sheet", async () => {
    const wrapper = mountMesh(["obj", "gif", "webp"], geometry);
    await wrapper.get("[data-test='mesh-export-animation']").trigger("click");
    await flushPromises();

    expect(wrapper.findComponent(VideoExportDialog).props("open")).toBe(true);
    expect(wrapper.findComponent(MeshExportDialog).props("open")).toBe(false);
  });

  // The stored container is never an export entry, so it can never reach the
  // geometry sheet either.
  it("still never offers the stored GLB", () => {
    const wrapper = mountMesh(["glb", "obj"], geometry);
    expect(wrapper.find("[data-test='mesh-export-glb']").exists()).toBe(false);
  });
});

/**
 * The desktop Lightbox promises a poster while a mesh loads (and keeps it on
 * failure); this guards that the print's thumbnail actually reaches
 * `MeshViewer`, not just that `AuthedMedia` accepts a `poster` prop. Every
 * test above stubs `AuthedMedia` out entirely, so it never exercises this —
 * these two mount the real component tree instead.
 */
describe("Lightbox — mesh poster", () => {
  it("passes a resolved poster to the mesh viewer", async () => {
    const wrapper = mount(Lightbox, {
      props: {
        item: meshItem,
        index: 0,
        count: 1,
        video: false,
        mesh: true,
        target,
        meshExportFormats: [],
      },
    });
    await vi.waitFor(() => expect(wrapper.findComponent(MeshViewer).exists()).toBe(true));
    await flushPromises();

    const viewer = wrapper.findComponent(MeshViewer);
    expect(viewer.props("poster")).toBeTruthy();
  });

  it("leaves the poster undefined for a non-mesh print", async () => {
    const wrapper = mount(Lightbox, {
      props: {
        item: { ...meshItem, filename: "print-0001.png", format: "png" },
        index: 0,
        count: 1,
        video: false,
        target,
      },
    });
    await flushPromises();

    expect(wrapper.findComponent(MeshViewer).exists()).toBe(false);
  });
});
