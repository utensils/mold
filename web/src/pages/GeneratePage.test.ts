import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import GeneratePage from "./GeneratePage.vue";
import type { GalleryImage } from "../types";

const entry: GalleryImage = {
  filename: "generate-visible.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "generate visible",
    model: "flux-dev:fp16",
    seed: 2,
    steps: 20,
    guidance: 3.5,
    width: 1024,
    height: 1024,
    version: "test",
  },
};

vi.mock("../api", () => ({
  fetchModels: vi.fn(async () => []),
  listGallery: vi.fn(async () => [entry]),
  deleteGalleryImage: vi.fn(async () => undefined),
}));

vi.mock("../composables/useGenerateStream", () => ({
  useGenerateStream: () => ({
    jobs: { value: [] },
    submit: vi.fn(),
    cancel: vi.fn(),
    remove: vi.fn(),
    clearDone: vi.fn(),
  }),
}));

vi.mock("../composables/useStatusPoll", () => ({
  useStatusPoll: () => ({ status: { value: null } }),
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

describe("GeneratePage layout and visibility", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("uses a wider large-screen shell and controls rail", () => {
    const wrapper = mount(GeneratePage, {
      global: { stubs: pageStubs() },
    });

    expect(wrapper.get("[data-test='generate-shell']").classes()).toContain(
      "max-w-[2400px]",
    );
    expect(wrapper.get("[data-test='generate-workspace']").classes()).toContain(
      "2xl:grid-cols-[22rem_minmax(0,1fr)_30rem]",
    );
  });

  it("keeps the compact recent gallery visible after refreshes", async () => {
    localStorage.setItem("mold.gallery.hide", "true");

    const wrapper = mount(GeneratePage, {
      global: { stubs: pageStubs() },
    });
    await flushPromises();

    const feed = wrapper.findComponent(GalleryFeedStub);
    expect(feed.props("entries")).toEqual([entry]);
    expect(feed.props("hideMode")).toBeUndefined();
    expect(feed.props("revealed")).toBeUndefined();
  });
});

function pageStubs() {
  return {
    TopBar: { template: "<header />" },
    Composer: { template: "<section />" },
    GenerateParamsPanel: {
      template: "<aside />",
      methods: { setExpanded: vi.fn() },
    },
    LoraPicker: { template: "<div />" },
    ModelPicker: { template: "<div />" },
    PreferencesModal: { template: "<div />" },
    ExpandModal: { template: "<div />" },
    ImagePickerModal: { template: "<div />" },
    RunningStrip: { template: "<div />" },
    GalleryFeed: GalleryFeedStub,
    DetailDrawer: { template: "<div />" },
  };
}
