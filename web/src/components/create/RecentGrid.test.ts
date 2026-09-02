import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import RecentGrid from "./RecentGrid.vue";
import type { GalleryImage, OutputFormat } from "../../types";

vi.mock("../../api", () => ({
  thumbnailUrl: (name: string) => `/api/gallery/thumbnail/${name}`,
}));

function entry(filename: string, format: OutputFormat = "png"): GalleryImage {
  return {
    filename,
    format,
    timestamp: 0,
    metadata: {
      prompt: `prompt for ${filename}`,
      model: "flux2-klein",
      seed: 1,
    } as GalleryImage["metadata"],
  };
}

const stubs = {
  RouterLink: { template: "<a><slot /></a>" },
  Icon: { template: "<i />" },
};

function mountGrid(entries: GalleryImage[], limit?: number) {
  return mount(RecentGrid, {
    props: { entries, ...(limit != null ? { limit } : {}) },
    global: { stubs },
  });
}

describe("RecentGrid", () => {
  it("renders one uniform tile per entry (aspect-fit thumbnail source)", () => {
    const w = mountGrid([entry("a.png"), entry("b.png"), entry("c.png")]);
    const tiles = w.findAll("[data-test='recent-tile']");
    expect(tiles).toHaveLength(3);
    // Every tile pulls the cached square thumbnail — not the full-res file —
    // so the grid stays uniform regardless of the print's real aspect ratio.
    const img = tiles[0].get("img");
    expect(img.attributes("src")).toBe("/api/gallery/thumbnail/a.png");
  });

  it("caps the grid at `limit` and links the overflow to the gallery", () => {
    const entries = Array.from({ length: 30 }, (_, i) => entry(`p${i}.png`));
    const w = mountGrid(entries, 12);
    expect(w.findAll("[data-test='recent-tile']")).toHaveLength(12);
    const more = w.find("[data-test='recent-view-all']");
    expect(more.exists()).toBe(true);
    expect(more.text()).toContain("30");
  });

  it("fills large web workspaces when given the desktop history limit", () => {
    const entries = Array.from({ length: 60 }, (_, i) => entry(`p${i}.png`));
    const w = mountGrid(entries, 50);
    expect(w.findAll("[data-test='recent-tile']")).toHaveLength(50);
  });

  it("does not render a view-all link when everything fits", () => {
    const w = mountGrid([entry("a.png")], 12);
    expect(w.find("[data-test='recent-view-all']").exists()).toBe(false);
  });

  it("marks video prints with a single non-overlapping badge", () => {
    const w = mountGrid([entry("clip.mp4", "mp4"), entry("still.png")]);
    const badges = w.findAll("[data-test='recent-video-badge']");
    expect(badges).toHaveLength(1);
  });

  // The gallery grid already badges a `.glb` tile 3D; Create's strip is the
  // same print list and must not leave a mesh looking like a still.
  it("marks a 3-D print with the same 3D badge the gallery grid uses", () => {
    const w = mountGrid([
      entry("chair.glb", "glb"),
      entry("clip.mp4", "mp4"),
      entry("still.png"),
    ]);
    const mesh = w.findAll("[data-test='recent-mesh-badge']");
    expect(mesh).toHaveLength(1);
    expect(mesh[0].text()).toContain("3D");
    expect(w.findAll("[data-test='recent-video-badge']")).toHaveLength(1);
  });

  it("emits open with the clicked entry", async () => {
    const item = entry("a.png");
    const w = mountGrid([item]);
    await w.get("[data-test='recent-tile']").trigger("click");
    expect(w.emitted("open")?.[0]?.[0]).toStrictEqual(item);
  });

  it("emits a positioned context menu request for a right-clicked entry", async () => {
    const item = entry("a.png");
    const w = mountGrid([item]);
    await w.get("[data-test='recent-tile']").trigger("contextmenu", {
      clientX: 120,
      clientY: 80,
    });
    expect(w.emitted("context-menu")?.[0]?.[0]).toStrictEqual({
      item,
      x: 120,
      y: 80,
      trigger: w.get("[data-test='recent-tile']").element,
    });
  });

  it("shows an empty hint when there are no prints", () => {
    const w = mountGrid([]);
    expect(w.find("[data-test='recent-empty']").exists()).toBe(true);
    expect(w.findAll("[data-test='recent-tile']")).toHaveLength(0);
  });
});
