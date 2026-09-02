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

import Lightbox from "./Lightbox.vue";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
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

function mountMesh(meshExportFormats: string[]) {
  return mount(Lightbox, {
    props: {
      item: meshItem,
      index: 0,
      count: 1,
      video: false,
      mesh: true,
      target,
      meshExportFormats,
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

  it("offers only what this host advertises, never a client list", () => {
    const wrapper = mountMesh(["stl"]);
    expect(wrapper.find("[data-test='mesh-export-obj']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mesh-export-stl']").exists()).toBe(true);
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
