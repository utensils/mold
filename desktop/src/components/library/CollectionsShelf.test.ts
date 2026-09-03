import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import CollectionCard from "./CollectionCard.vue";
import CollectionsShelf, { type ShelfCard } from "./CollectionsShelf.vue";

const authedMediaStub = {
  name: "AuthedMedia",
  props: ["path", "target", "cacheKey", "video", "alt"],
  template: "<img :data-path='path' :data-cache='cacheKey' />",
};

const NOW = 1_700_000_000_000;

const cards: ShelfCard[] = [
  {
    slug: "smurfs",
    name: "Smurfs",
    count: 9,
    hostLabels: ["This Mac", "plato"],
    updatedAt: NOW / 1000 - 7200,
    covers: [
      { path: "/api/gallery/thumbnail/a.png", target: null, cacheKey: "local" },
      { path: "/api/gallery/thumbnail/b.png", target: null, cacheKey: "local" },
      { path: "/api/gallery/thumbnail/c.png", target: null, cacheKey: "plato" },
      { path: "/api/gallery/thumbnail/d.png", target: null, cacheKey: "plato" },
      { path: "/api/gallery/thumbnail/e.png", target: null, cacheKey: "plato" },
    ],
    hidden: true,
  },
  {
    slug: "river-studies",
    name: "River studies",
    count: 1,
    hostLabels: ["This Mac"],
    updatedAt: null,
    covers: [],
    hidden: false,
  },
];

describe("CollectionCard", () => {
  it("renders a 2×2 mosaic (max four), name, mono meta, and updated time", () => {
    const wrapper = mount(CollectionCard, {
      props: { ...cards[0]!, nowMs: NOW },
      global: { stubs: { AuthedMedia: authedMediaStub } },
    });
    expect(wrapper.findAll("img")).toHaveLength(4);
    expect(wrapper.get("[data-test='collection-mosaic']").classes()).toContain("grid-cols-2");
    expect(wrapper.get("[data-test='collection-name']").text()).toContain("Smurfs");
    expect(wrapper.get("[data-test='collection-hidden-badge']").text()).toBe("Hidden");
    const meta = wrapper.get("[data-test='collection-meta']");
    expect(meta.text()).toBe("9 prints · This Mac · plato");
    expect(meta.classes()).toContain("font-mono");
    expect(wrapper.get("[data-test='collection-updated']").text()).toBe("Updated 2h ago");
  });

  it("falls back to a glyph with no covers and a singular noun", () => {
    const wrapper = mount(CollectionCard, {
      props: { ...cards[1]!, nowMs: NOW },
      global: { stubs: { AuthedMedia: authedMediaStub } },
    });
    expect(wrapper.findAll("img")).toHaveLength(0);
    expect(wrapper.get("[data-test='collection-meta']").text()).toBe("1 print · This Mac");
    expect(wrapper.find("[data-test='collection-updated']").exists()).toBe(false);
  });

  it("opens on click and hands the right-click to the parent", async () => {
    const wrapper = mount(CollectionCard, {
      props: { ...cards[1]! },
      global: { stubs: { AuthedMedia: authedMediaStub } },
    });
    await wrapper.get("[data-test='collection-card']").trigger("click");
    expect(wrapper.emitted("open")).toHaveLength(1);
    await wrapper.get("[data-test='collection-card']").trigger("contextmenu");
    expect(wrapper.emitted("contextmenu")).toHaveLength(1);
  });
});

describe("CollectionsShelf", () => {
  function mountShelf(props: Record<string, unknown> = {}) {
    return mount(CollectionsShelf, {
      props: { cards, nowMs: NOW, ...props },
      global: { stubs: { AuthedMedia: authedMediaStub } },
    });
  }

  it("lists one card per collection and a dashed New collection card", async () => {
    const wrapper = mountShelf();
    const items = wrapper.findAll("[data-test='collection-card']");
    expect(items.map((c) => c.attributes("data-slug"))).toEqual(["smurfs", "river-studies"]);
    expect(wrapper.get("[data-test='new-collection-card']").text()).toContain("New collection");
    await items[0]!.trigger("click");
    expect(wrapper.emitted("open")).toEqual([["smurfs"]]);
    await items[1]!.trigger("contextmenu");
    expect(wrapper.emitted("contextmenu")?.[0]?.[0]).toBe("river-studies");
  });

  it("creates from the inline name input on Enter and cancels on Escape / empty", async () => {
    const wrapper = mountShelf();
    await wrapper.get("[data-test='new-collection-label']").trigger("click");
    const input = wrapper.get("[data-test='new-collection-input']");
    await input.setValue("Film grain tests");
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("create")).toEqual([["Film grain tests"]]);
    expect(wrapper.find("[data-test='new-collection-input']").exists()).toBe(false);

    await wrapper.get("[aria-label='New collection']").trigger("click");
    await wrapper.get("[data-test='new-collection-input']").trigger("keydown", { key: "Escape" });
    expect(wrapper.find("[data-test='new-collection-input']").exists()).toBe(false);

    await wrapper.get("[aria-label='New collection']").trigger("click");
    await wrapper.get("[data-test='new-collection-input']").trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("create")).toHaveLength(1);
  });

  it("hides the New card when no host can organize", () => {
    const wrapper = mountShelf({ canCreate: false });
    expect(wrapper.find("[data-test='new-collection-card']").exists()).toBe(false);
  });
});
