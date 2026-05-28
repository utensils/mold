import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, nextTick } from "vue";
import GeneratePage from "./GeneratePage.vue";
import type {
  GalleryImage,
  GenerateFormState,
  ModelInfoExtended,
} from "../types";

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

const submitMock = vi.hoisted(() => vi.fn());
const fetchModelsMock = vi.hoisted(() => vi.fn(async () => []));

vi.mock("../api", () => ({
  fetchModels: fetchModelsMock,
  listGallery: vi.fn(async () => [entry]),
  deleteGalleryImage: vi.fn(async () => undefined),
  updateQueueJobTargetGpu: vi.fn(async () => undefined),
}));

vi.mock("../composables/useGenerateStream", () => ({
  useGenerateStream: () => ({
    jobs: { value: [] },
    submit: submitMock,
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

const fluxModel: ModelInfoExtended = {
  name: "flux-dev:q4",
  family: "flux",
  size_gb: 12,
  is_loaded: false,
  last_used: null,
  hf_repo: "black-forest-labs/FLUX.1-dev",
  downloaded: true,
  default_steps: 28,
  default_guidance: 3.5,
  default_width: 1024,
  default_height: 1024,
  description: "",
};

const qwenEditModel: ModelInfoExtended = {
  ...fluxModel,
  name: "qwen-image-edit:q4",
  family: "qwen-image-edit",
  hf_repo: "Qwen/Qwen-Image-Edit",
};

describe("GeneratePage layout and visibility", () => {
  beforeEach(() => {
    localStorage.clear();
    submitMock.mockClear();
    fetchModelsMock.mockResolvedValue([]);
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

  it("blocks non-Qwen mask submissions until a source image is selected", async () => {
    fetchModelsMock.mockResolvedValue([fluxModel]);
    const wrapper = mount(GeneratePage, {
      global: { stubs: pageStubs() },
    });
    await flushPromises();

    const params = wrapper.getComponent({ name: "GenerateParamsPanel" });
    const current = params.props("modelValue") as GenerateFormState;
    params.vm.$emit("update:modelValue", {
      ...current,
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("Mask image needs a source image.");
  });

  it("submits Qwen edit images without sending stale mask state", async () => {
    fetchModelsMock.mockResolvedValue([qwenEditModel]);
    const wrapper = mount(GeneratePage, {
      global: { stubs: pageStubs() },
    });
    await flushPromises();

    const params = wrapper.getComponent({ name: "GenerateParamsPanel" });
    const current = params.props("modelValue") as GenerateFormState;
    params.vm.$emit("update:modelValue", {
      ...current,
      imageAttachments: [
        { kind: "upload", filename: "target.png", base64: "TARGET" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");

    expect(submitMock).toHaveBeenCalledTimes(1);
    const req = submitMock.mock.calls[0][0];
    expect(req.edit_images).toEqual(["TARGET"]);
    expect(req.mask_image).toBeUndefined();
    expect(req.source_image).toBeUndefined();
  });
});

function pageStubs() {
  return {
    TopBar: { template: "<header />" },
    Composer: {
      props: ["submitError"],
      emits: ["submit"],
      template:
        '<section><p v-if="submitError">{{ submitError }}</p><button data-test="composer-submit" @click="$emit(\'submit\')">submit</button></section>',
    },
    GenerateParamsPanel: {
      name: "GenerateParamsPanel",
      props: ["modelValue"],
      emits: ["update:modelValue"],
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
