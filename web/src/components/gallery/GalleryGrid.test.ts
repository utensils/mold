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
        fresh: new Set(["origin|recent.png"]),
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

describe("GalleryGrid multi-host thumbnails", () => {
  beforeEach(() => {
    (globalThis as any).IntersectionObserver = FakeIntersectionObserver;
    localStorage.clear();
  });
  afterEach(() => {
    delete (globalThis as Partial<typeof globalThis>).IntersectionObserver;
    localStorage.clear();
  });

  it("addresses a remote print's thumbnail on its own host", async () => {
    // A merged remote entry resolved against the origin 404s — the print
    // lives on the other machine, so its thumbnail must too.
    localStorage.setItem(
      "mold.web.hosts.v1",
      JSON.stringify([
        { id: "studio-7680", name: "studio", url: "http://studio:7680" },
      ]),
    );
    const remote = { ...image, hostId: "studio-7680", hostLabel: "studio" };
    const w = mount(GalleryGrid, {
      props: { entries: [remote], loading: false },
    });
    await new Promise((r) => setTimeout(r, 0));
    const src = w.find("img").attributes("src") ?? "";
    expect(src).toContain("http://studio:7680/api/gallery/thumbnail/");
  });

  it("selects only the collided print that belongs to the selected host", async () => {
    // Same filename on two machines. Selection is keyed on (host, filename),
    // so highlighting the remote copy must leave the local one untouched.
    localStorage.setItem(
      "mold.web.hosts.v1",
      JSON.stringify([
        { id: "studio-7680", name: "studio", url: "http://studio:7680" },
      ]),
    );
    const mine = { ...image, hostId: "origin", hostLabel: "this server" };
    const theirs = { ...image, hostId: "studio-7680", hostLabel: "studio" };
    const w = mount(GalleryGrid, {
      props: {
        entries: [mine, theirs],
        loading: false,
        selectMode: true,
        selection: new Set(["studio-7680|recent.png"]),
      },
    });
    const hits = w.findAll(".gg__hit");
    expect(hits).toHaveLength(2);
    expect(hits[0]!.classes()).not.toContain("gg__hit--on");
    expect(hits[1]!.classes()).toContain("gg__hit--on");
  });

  it("emits toggle-select carrying the host-tagged entry", async () => {
    const mine = { ...image, hostId: "origin", hostLabel: "this server" };
    const theirs = { ...image, hostId: "studio-7680", hostLabel: "studio" };
    const w = mount(GalleryGrid, {
      props: {
        entries: [mine, theirs],
        loading: false,
        selectMode: true,
        selection: new Set<string>(),
      },
    });
    await w.findAll(".gg__hit")[1]!.trigger("click");
    const payload = w.emitted("toggle-select")![0]![0] as {
      item: { hostId?: string };
    };
    expect(payload.item.hostId).toBe("studio-7680");
  });

  it("badges only the collided print that is actually fresh", () => {
    const mine = { ...image, hostId: "origin", hostLabel: "this server" };
    const theirs = { ...image, hostId: "studio-7680", hostLabel: "studio" };
    const w = mount(GalleryGrid, {
      props: {
        entries: [mine, theirs],
        loading: false,
        fresh: new Set(["studio-7680|recent.png"]),
      },
    });
    expect(w.findAll(".ms-tile__fresh")).toHaveLength(1);
  });

  it("keeps origin prints on relative URLs", async () => {
    const w = mount(GalleryGrid, {
      props: {
        entries: [{ ...image, hostId: "origin", hostLabel: "this server" }],
        loading: false,
      },
    });
    await new Promise((r) => setTimeout(r, 0));
    const src = w.find("img").attributes("src") ?? "";
    expect(src.startsWith("/api/gallery/thumbnail/")).toBe(true);
  });
});
