import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import GalleryPage from "./GalleryPage.vue";
import type { GalleryImage } from "../types";

const entry: GalleryImage = {
  filename: "gallery-visible.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "gallery visible",
    model: "flux-dev:fp16",
    seed: 1,
    steps: 20,
    guidance: 3.5,
    width: 1024,
    height: 1024,
    version: "test",
  },
};

vi.mock("../api", () => ({
  listGallery: vi.fn(async () => [entry]),
  deleteGalleryImage: vi.fn(async () => undefined),
}));

// The page seeds its search from the global nav's `?q=`. No router is
// installed in these isolated page mounts, so stub the route.
vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {} }),
}));

const GalleryFeedStub = defineComponent({
  name: "GalleryFeed",
  props: {
    entries: { type: Array, required: true },
    loading: { type: Boolean, required: true },
    view: { type: String, required: true },
    muted: { type: Boolean, required: true },
    hideMode: { type: Boolean, default: undefined },
    revealed: { type: Object, default: undefined },
  },
  template: '<div data-test="gallery-feed">{{ entries.length }}</div>',
});

describe("GalleryPage visibility", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("does not pass persisted hide state into gallery content", async () => {
    localStorage.setItem("mold.gallery.hide", "true");

    const wrapper = mount(GalleryPage, {
      global: {
        stubs: {
          GalleryFeed: GalleryFeedStub,
          DetailDrawer: { template: "<aside />" },
          Transition: false,
        },
      },
    });
    await flushPromises();

    const feed = wrapper.findComponent(GalleryFeedStub);
    expect(feed.props("entries")).toEqual([entry]);
    expect(feed.props("hideMode")).toBeUndefined();
    expect(feed.props("revealed")).toBeUndefined();
  });
});
