import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, nextTick } from "vue";
import GeneratePage from "./GeneratePage.vue";
import {
  useGenerateForm,
  __testing__ as generateFormTesting,
} from "../composables/useGenerateForm";
import {
  resetNotifications,
  settleConfirm,
  useNotifications,
} from "../lib/toasts";
import type { ChainJobDetail, GalleryImage } from "../types";

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
    ({ __v_isRef: true, value: null as ChainJobDetail | null }) as {
      __v_isRef: true;
      value: ChainJobDetail | null;
    },
);

vi.mock("../api", () => ({
  createChainJob: createChainJobMock,
  fetchModels: vi.fn(async () => []),
  fetchQueue: vi.fn(async () => ({ entries: [] })),
  listGallery: vi.fn(async () => [entry]),
  deleteGalleryImage: vi.fn(async () => undefined),
  upscaleStream: vi.fn(async () => undefined),
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
    showRecreate: { type: Boolean, default: undefined },
  },
  template: '<div data-test="gallery-feed">{{ entries.length }}</div>',
});

describe("GeneratePage layout and behavior", () => {
  beforeEach(() => {
    localStorage.clear();
    generateFormTesting.resetForTest();
    resetNotifications();
    submitMock.mockClear();
    createChainJobMock.mockClear();
    createChainJobMock.mockResolvedValue({ job_id: "job-1" });
    chainJobDetailRef.value = null;
    vi.stubGlobal("prompt", vi.fn());
  });

  it("uses the Mold Studio three-column workspace", () => {
    const wrapper = mount(GeneratePage, { global: { stubs: pageStubs() } });
    expect(wrapper.get("[data-test='generate-shell']").classes()).toContain(
      "max-w-[1600px]",
    );
    expect(wrapper.get("[data-test='generate-workspace']").classes()).toContain(
      "xl:grid-cols-[238px_minmax(0,1fr)_296px]",
    );
  });

  it("keeps the recent gallery visible after refreshes", async () => {
    const wrapper = mount(GeneratePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const feed = wrapper.findComponent(GalleryFeedStub);
    expect(feed.props("entries")).toEqual([entry]);
  });

  it("blocks non-Qwen mask submissions until a source image is selected", async () => {
    const wrapper = mount(GeneratePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK",
    };
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("Mask image needs a source image.");
  });

  it("submits Qwen edit images without sending stale mask state", async () => {
    const wrapper = mount(GeneratePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "qwen-image-edit:q4";
    form.state.value.modelFamily = "qwen-image-edit";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "target.png", base64: "TARGET" },
    ];
    form.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK",
    };
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(1);
    const req = submitMock.mock.calls[0][0];
    expect(req.edit_images).toEqual(["TARGET"]);
    expect(req.mask_image).toBeUndefined();
    expect(req.source_image).toBeUndefined();
  });

  it("asks before replacing a source image while a mask exists", async () => {
    const wrapper = mount(GeneratePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "old.png", base64: "OLD" },
    ];
    form.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK",
    };
    await nextTick();

    wrapper
      .getComponent({ name: "ImagePickerModal" })
      .vm.$emit("pick", [
        { kind: "upload", filename: "new.png", base64: "NEW" },
      ]);
    await nextTick();

    expect(useNotifications().confirm?.kind).toBe("choice");
    settleConfirm("reset");
    await flushPromises();

    expect(form.state.value.imageAttachments[0]?.filename).toBe("new.png");
    expect(form.state.value.maskImage).toBeNull();
  });

  it("submits Sequence mode through createChainJob instead of the legacy stream", async () => {
    const wrapper = mount(GeneratePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    // Switch to Sequence so the ScriptComposer renders.
    const seqButton = wrapper
      .findAll("[data-test='composer-mode'] button")
      .find((b) => b.text() === "Sequence")!;
    await seqButton.trigger("click");

    wrapper.getComponent({ name: "ScriptComposer" }).vm.$emit("submit", {
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
        stages: [expect.objectContaining({ prompt: "stage zero", frames: 9 })],
      }),
    );
    expect(submitMock).not.toHaveBeenCalled();
  });

  it("fans a batch out into variations and queues one print per edited prompt", async () => {
    const wrapper = mount(GeneratePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    form.state.value.batchSize = 3;
    await nextTick();

    // Expand fans out into 3 client-side variations instead of submitting.
    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();
    expect(submitMock).not.toHaveBeenCalled();
    expect(
      wrapper.getComponent({ name: "ResultCanvas" }).props("variations"),
    ).toHaveLength(3);

    // Queue submits one single print per variation.
    await wrapper.get("[data-test='queue-variations']").trigger("click");
    await flushPromises();
    expect(submitMock).toHaveBeenCalledTimes(3);
    for (const call of submitMock.mock.calls) {
      expect(call[0].batch_size).toBe(1);
      expect(call[0].prompt).toContain("a lighthouse");
    }
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
      script: { schema: "mold.chain.v1", chain: {}, stage: [] },
    };
    const wrapper = mount(GeneratePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(
      wrapper.getComponent({ name: "ChainJobCard" }).props("job"),
    ).toMatchObject({ id: "job-1", state: "queued" });
  });
});

function pageStubs() {
  return {
    ResourceStrip: { template: "<div />" },
    ComposerCard: {
      name: "ComposerCard",
      template:
        '<div><button data-test="composer-submit" @click="$emit(\'submit\')">go</button><button data-test="composer-expand" @click="$emit(\'expand\')">expand</button></div>',
    },
    ResultCanvas: {
      name: "ResultCanvas",
      props: ["mode", "variations"],
      template:
        '<div data-test="result-canvas" :data-count="(variations||[]).length"><button data-test="queue-variations" @click="$emit(\'queue\')">queue</button></div>',
    },
    ControlsAside: { name: "ControlsAside", template: "<aside />" },
    AdvancedDrawer: { name: "AdvancedDrawer", template: "<div />" },
    ActivityStrip: { name: "ActivityStrip", template: "<div />" },
    ScriptComposer: {
      name: "ScriptComposer",
      template: "<div data-test='script-composer' />",
      methods: { setStagePrompt: vi.fn() },
    },
    ChainJobCard: {
      name: "ChainJobCard",
      props: ["job"],
      template: '<div data-test="chain-job-card">{{ job.id }}</div>',
    },
    ExpandModal: { name: "ExpandModal", template: "<div />" },
    ImagePickerModal: {
      name: "ImagePickerModal",
      props: ["open"],
      emits: ["pick", "close"],
      template: "<div />",
    },
    MaskEditorModal: { name: "MaskEditorModal", template: "<div />" },
    GenerationTemplatesPanel: { template: "<div />" },
    GalleryFeed: GalleryFeedStub,
    Lightbox: { template: "<div />" },
  };
}
