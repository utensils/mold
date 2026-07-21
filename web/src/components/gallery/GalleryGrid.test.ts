import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import GalleryGrid from "./GalleryGrid.vue";
import type { GalleryImage } from "../../types";

class FakeIntersectionObserver {
  observe() {}
  disconnect() {}
}

function meta(overrides: Partial<GalleryImage["metadata"]> = {}) {
  return {
    prompt: "a print",
    model: "flux-dev:fp16",
    seed: 7,
    steps: 20,
    guidance: 3.5,
    width: 1024,
    height: 1024,
    version: "test",
    ...overrides,
  };
}

const image: GalleryImage = {
  filename: "recent.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: meta({ prompt: "a wandering cat" }),
};
const video: GalleryImage = {
  filename: "clip.mp4",
  timestamp: 1_700_000_100,
  format: "mp4",
  metadata: meta({ prompt: "a drifting cloud", frames: 25, fps: 24 }),
};

describe("GalleryGrid", () => {
  beforeEach(() => {
    (globalThis as any).IntersectionObserver = FakeIntersectionObserver;
  });
  afterEach(() => {
    delete (globalThis as Partial<typeof globalThis>).IntersectionObserver;
  });

  it("renders a tile per entry", () => {
    const wrapper = mount(GalleryGrid, {
      props: { entries: [image, video], loading: false },
    });
    expect(wrapper.findAll(".gg__cell")).toHaveLength(2);
  });

  it("badges only fresh prints as NEW", () => {
    const wrapper = mount(GalleryGrid, {
      props: {
        entries: [image, video],
        loading: false,
        fresh: new Set(["recent.png"]),
      },
    });
    const badges = wrapper.findAll(".ms-tile__fresh");
    expect(badges).toHaveLength(1);
    expect(badges[0]!.text()).toBe("New");
  });

  it("shows a play glyph + duration overlay on motion prints", () => {
    const wrapper = mount(GalleryGrid, {
      props: { entries: [image, video], loading: false },
    });
    const badges = wrapper.findAll(".gg__vbadge");
    expect(badges).toHaveLength(1);
    expect(badges[0]!.text()).toContain("1.0s");
  });

  it("opens a print on tile click when not selecting", async () => {
    const wrapper = mount(GalleryGrid, {
      props: { entries: [image], loading: false },
    });
    await wrapper.find(".ms-tile").trigger("click");
    const open = wrapper.emitted("open");
    expect(open).toBeTruthy();
    expect((open![0]![0] as GalleryImage).filename).toBe("recent.png");
  });

  it("toggles selection (with shift) via the hit layer in select mode", async () => {
    const wrapper = mount(GalleryGrid, {
      props: {
        entries: [image, video],
        loading: false,
        selectMode: true,
        selection: new Set<string>(),
      },
    });
    const hits = wrapper.findAll(".gg__hit");
    expect(hits).toHaveLength(2);
    await hits[1]!.trigger("click", { shiftKey: true });
    const evt = wrapper.emitted("toggle-select");
    expect(evt).toBeTruthy();
    const payload = evt![0]![0] as {
      item: GalleryImage;
      shift: boolean;
      meta: boolean;
    };
    expect(payload.item.filename).toBe("clip.mp4");
    expect(payload.shift).toBe(true);
  });

  it("renders skeletons while loading an empty gallery", () => {
    const wrapper = mount(GalleryGrid, {
      props: { entries: [], loading: true },
    });
    expect(wrapper.findAll(".gg__skel").length).toBeGreaterThan(0);
    expect(wrapper.findAll(".gg__cell")).toHaveLength(0);
  });
});
