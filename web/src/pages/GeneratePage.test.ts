import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, nextTick } from "vue";
import GeneratePage from "./GeneratePage.vue";
import { __testing__ as generateFormTesting } from "../composables/useGenerateForm";
import {
  resetNotifications,
  settleConfirm,
  useNotifications,
} from "../lib/toasts";
import type {
  ChainJobDetail,
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
const createChainJobMock = vi.hoisted(() =>
  vi.fn(async () => ({ job_id: "job-1" })),
);
const chainJobDetailRef = vi.hoisted(
  () =>
    ({
      __v_isRef: true,
      value: null as ChainJobDetail | null,
    }) as { __v_isRef: true; value: ChainJobDetail | null },
);
const fetchModelsMock = vi.hoisted(() =>
  vi.fn(async (): Promise<ModelInfoExtended[]> => []),
);

vi.mock("../api", () => ({
  createChainJob: createChainJobMock,
  fetchModels: fetchModelsMock,
  fetchQueue: vi.fn(async () => ({ entries: [] })),
  listGallery: vi.fn(async () => [entry]),
  deleteGalleryImage: vi.fn(async () => undefined),
  upscaleStream: vi.fn(async () => undefined),
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

vi.mock("../composables/useChainJobStream", () => ({
  useChainJobStream: () => ({
    detail: chainJobDetailRef,
    connected: { __v_isRef: true, value: true },
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
    generateFormTesting.resetForTest();
    resetNotifications();
    submitMock.mockClear();
    createChainJobMock.mockClear();
    createChainJobMock.mockResolvedValue({ job_id: "job-1" });
    chainJobDetailRef.value = null;
    fetchModelsMock.mockResolvedValue([]);
    vi.stubGlobal("prompt", vi.fn());
  });

  it("uses a wider large-screen shell and controls rail", () => {
    const wrapper = mount(GeneratePage, {
      global: { stubs: pageStubs() },
    });

    expect(wrapper.get("[data-test='generate-shell']").classes()).toContain(
      "max-w-[2400px]",
    );
    expect(wrapper.get("[data-test='generate-workspace']").classes()).toContain(
      "2xl:grid-cols-[26rem_minmax(0,1fr)_38rem]",
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

  it("asks before replacing a source image while a mask exists", async () => {
    fetchModelsMock.mockResolvedValue([fluxModel]);
    const wrapper = mount(GeneratePage, {
      global: { stubs: pageStubs() },
    });
    await flushPromises();

    const params = wrapper.getComponent({ name: "GenerateParamsPanel" });
    const current = params.props("modelValue") as GenerateFormState;
    params.vm.$emit("update:modelValue", {
      ...current,
      imageAttachments: [
        { kind: "upload", filename: "old.png", base64: "OLD" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });
    await nextTick();

    wrapper.getComponent({ name: "Composer" }).vm.$emit("open-image-picker");
    await nextTick();
    wrapper
      .getComponent({ name: "ImagePickerModal" })
      .vm.$emit("pick", [
        { kind: "upload", filename: "new.png", base64: "NEW" },
      ]);
    await nextTick();

    // The mask conflict now flows through the app dialog store.
    expect(useNotifications().confirm?.kind).toBe("choice");
    settleConfirm("reset");
    await flushPromises();

    const updated = wrapper
      .getComponent({ name: "GenerateParamsPanel" })
      .props("modelValue") as GenerateFormState;
    expect(updated.imageAttachments[0]?.filename).toBe("new.png");
    expect(updated.maskImage).toBeNull();
  });

  it("submits Script mode through createChainJob instead of the legacy stream", async () => {
    const wrapper = mount(GeneratePage, {
      global: { stubs: pageStubs() },
    });
    await flushPromises();

    wrapper.getComponent({ name: "Composer" }).vm.$emit("submit-script", {
      chain: {
        model: "ltx-2-19b-distilled:fp8",
        width: 64,
        height: 64,
        fps: 12,
        seed: 42,
        steps: 4,
        guidance: 3,
        strength: 1,
        output_format: "mp4",
        motion_tail_frames: 0,
      },
      stage: [{ prompt: "stage zero", frames: 9, transition: "cut" }],
    });
    await flushPromises();

    expect(createChainJobMock).toHaveBeenCalledTimes(1);
    expect(createChainJobMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "ltx-2-19b-distilled:fp8",
        output_format: "mp4",
        stages: [
          expect.objectContaining({
            prompt: "stage zero",
            frames: 9,
            transition: "cut",
          }),
        ],
      }),
    );
    expect(submitMock).not.toHaveBeenCalled();
  });

  it("renders the submitted durable chain job card from useChainJobStream", async () => {
    chainJobDetailRef.value = {
      id: "job-1",
      state: "queued",
      model: "ltx-2",
      stage_count: 1,
      current_stage: 0,
      created_at_unix_ms: 1,
      updated_at_unix_ms: 2,
      error: null,
      ephemeral: false,
      stages: [],
      finalizes: [],
      retakes: [],
      script: {
        schema: "mold.chain.v1",
        chain: {},
        stage: [],
      },
    };
    const wrapper = mount(GeneratePage, {
      global: { stubs: pageStubs() },
    });
    await flushPromises();

    expect(
      wrapper.getComponent({ name: "ChainJobCard" }).props("job"),
    ).toMatchObject({
      id: "job-1",
      state: "queued",
    });
  });
});

function pageStubs() {
  return {
    ResourceStrip: { template: "<div />" },
    Composer: {
      name: "Composer",
      props: ["submitError"],
      emits: ["submit", "submit-script", "open-image-picker", "clear-source"],
      template:
        '<section><p v-if="submitError">{{ submitError }}</p><button data-test="composer-submit" @click="$emit(\'submit\')">submit</button></section>',
    },
    ChainJobCard: {
      name: "ChainJobCard",
      props: ["job"],
      template: '<div data-test="chain-job-card">{{ job.id }}</div>',
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
    ExpandModal: { template: "<div />" },
    ImagePickerModal: {
      name: "ImagePickerModal",
      props: ["open"],
      emits: ["pick", "close"],
      template: "<div v-if='open' />",
    },
    RunningStrip: { template: "<div />" },
    GalleryFeed: GalleryFeedStub,
    DetailDrawer: { template: "<div />" },
  };
}
