import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, nextTick, type Component } from "vue";
import CreatePage from "./CreatePage.vue";
import {
  useGenerateForm,
  __testing__ as generateFormTesting,
} from "../composables/useGenerateForm";
import {
  resetNotifications,
  settleConfirm,
  useNotifications,
} from "../lib/toasts";
import { styleHint } from "../lib/stylePresets";
import { __testing__ as hostRoutingTesting } from "../composables/useHostRouting";
import { addHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { AUTO_TARGET_ID, CAPABLE_TARGET_ID } from "../lib/hostRouting";
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
const upscaleStreamMock = vi.hoisted(() =>
  vi.fn<
    (
      request: unknown,
      handlers: { onComplete: (event: { image: string }) => void },
    ) => Promise<void>
  >(async () => undefined),
);
const createChainJobMock = vi.hoisted(() =>
  vi.fn(async () => ({ job_id: "job-1" })),
);
const expandPromptMock = vi.hoisted(() =>
  vi.fn(async (request: { variations: number }) => ({
    original: "a lighthouse",
    expanded: ["north light", "storm light", "harbor light"].slice(
      0,
      request.variations,
    ),
  })),
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
  expandPrompt: expandPromptMock,
  fetchModels: vi.fn(async () => []),
  fetchQueue: vi.fn(async () => ({ entries: [] })),
  listGallery: vi.fn(async () => [entry]),
  deleteGalleryImage: vi.fn(async () => undefined),
  upscaleStream: upscaleStreamMock,
  fetchPromptHistory: vi.fn(async () => []),
  imageUrl: (name: string) => `/api/gallery/image/${name}`,
  thumbnailUrl: (name: string) => `/api/gallery/thumbnail/${name}`,
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

// Create now reads its model list (and routing inputs) from the per-host poll.
// Canned responses keep the page deterministic and off the network.
const hostStatusMock = vi.hoisted(() =>
  vi.fn(async (_host: { id: string }): Promise<Record<string, unknown>> => ({
    version: "test",
    models_loaded: [],
    busy: false,
    uptime_secs: 1,
    queue_depth: 0,
  })),
);
const hostModelsMock = vi.hoisted(() =>
  vi.fn(async (_host: { id: string }): Promise<unknown[]> => []),
);

vi.mock("../components/machines/hostClient", () => ({
  hostStatus: hostStatusMock,
  hostModels: hostModelsMock,
}));

const RecentGridStub = defineComponent({
  name: "RecentGrid",
  props: {
    entries: { type: Array, required: true },
    limit: { type: Number, default: undefined },
  },
  template: '<div data-test="recent-grid">{{ entries.length }}</div>',
});

describe("CreatePage layout and behavior", () => {
  beforeEach(async () => {
    // The routing singleton outlives a test's component; let any poll still in
    // flight from the previous test land, then discard what it wrote.
    hostRoutingTesting.reset();
    await flushPromises();
    hostRoutingTesting.reset();
    localStorage.clear();
    generateFormTesting.resetForTest();
    resetNotifications();
    submitMock.mockClear();
    upscaleStreamMock.mockReset();
    upscaleStreamMock.mockResolvedValue(undefined);
    createChainJobMock.mockClear();
    createChainJobMock.mockResolvedValue({ job_id: "job-1" });
    expandPromptMock.mockClear();
    expandPromptMock.mockImplementation(
      async (request: { variations: number }) => ({
        original: "a lighthouse",
        expanded: ["north light", "storm light", "harbor light"].slice(
          0,
          request.variations,
        ),
      }),
    );
    chainJobDetailRef.value = null;
    hostStatusMock.mockClear();
    hostModelsMock.mockClear();
    hostModelsMock.mockResolvedValue([]);
    vi.stubGlobal("prompt", vi.fn());
  });

  it("uses the Mold Studio composer + controls-region workspace", () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    expect(wrapper.get("[data-test='generate-shell']").classes()).toContain(
      "max-w-[1600px]",
    );
    expect(wrapper.get("[data-test='generate-workspace']").classes()).toContain(
      "md:grid-cols-[minmax(0,1fr)_340px]",
    );
  });

  it("keeps the recent gallery visible after refreshes", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const feed = wrapper.findComponent(RecentGridStub);
    expect(feed.props("entries")).toEqual([entry]);
  });

  it("dismisses the Templates popover with Escape and outside click", async () => {
    const wrapper = mount(CreatePage, {
      attachTo: document.body,
      global: { stubs: pageStubs() },
    });

    await wrapper.get("[data-test='templates-toggle']").trigger("click");
    expect(wrapper.find("[data-test='templates-popover']").exists()).toBe(true);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await nextTick();
    expect(wrapper.find("[data-test='templates-popover']").exists()).toBe(
      false,
    );

    await wrapper.get("[data-test='templates-toggle']").trigger("click");
    document.body.dispatchEvent(
      new MouseEvent("pointerdown", { bubbles: true }),
    );
    await nextTick();
    expect(wrapper.find("[data-test='templates-popover']").exists()).toBe(
      false,
    );
    wrapper.unmount();
  });

  it("configures an upscaler when the lightbox Upscale action is chosen", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "real-esrgan-x4plus:fp16",
        family: "real-esrgan",
        size_gb: 0.1,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 1,
        default_guidance: 1,
        default_width: 1024,
        default_height: 1024,
        description: "Upscaler",
      },
    ]);
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(["image"]),
    })) as never;
    const stubs: Record<string, Component> = pageStubs();
    stubs.RecentGrid = defineComponent({
      props: ["entries"],
      template:
        '<button data-test="open-recent" @click="$emit(\'open\', entries[0])">open</button>',
    });
    stubs.Lightbox = defineComponent({
      props: ["item"],
      template:
        '<button v-if="item" data-test="lightbox-upscale" @click="$emit(\'upscale\', item)">upscale</button>',
    });
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();
    await wrapper.get('[data-test="open-recent"]').trigger("click");
    await wrapper.get('[data-test="lightbox-upscale"]').trigger("click");
    await flushPromises();

    expect(useGenerateForm().state.value.imageAttachments[0]?.filename).toBe(
      entry.filename,
    );
    expect(useGenerateForm().state.value.upscaleModel).toBe(
      "real-esrgan-x4plus:fp16",
    );
    globalThis.fetch = originalFetch;
  });

  it("guides a first pull when no models are installed (cold start)", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(wrapper.find("[data-test='cold-start-stub']").exists()).toBe(true);
  });

  it("blocks non-Qwen mask submissions until a source image is selected", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
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
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
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

  it("reuses source preprocessing without replacing the editable source", async () => {
    upscaleStreamMock.mockImplementation(async (_request, handlers) => {
      handlers.onComplete({ image: "UPSCALED" });
    });
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.frames = 9;
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "source.png", base64: "SOURCE" },
    ];
    form.state.value.sourceFitPolicy = {
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x4plus:fp16",
      fit: { mode: "crop-fill" },
    };
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await vi.waitFor(() => expect(submitMock).toHaveBeenCalledTimes(2));

    expect(upscaleStreamMock).toHaveBeenCalledTimes(1);
    expect(
      submitMock.mock.calls.map(([request]) => request.source_image),
    ).toEqual(["UPSCALED", "UPSCALED"]);
    expect(form.state.value.imageAttachments[0]).toMatchObject({
      filename: "source.png",
      base64: "SOURCE",
    });
  });

  it("asks before replacing a source image while a mask exists", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
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

  it("submits Sequence mode through the durable chain endpoint", async () => {
    // Sequence only offers the ScriptComposer for a chain-capable (video)
    // model; a non-chain model gets the "sequences need a video model" panel.
    useGenerateForm().state.value.modelFamily = "ltx2";
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
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

  it("explains Sequence for a non-chain model instead of a dead composer", async () => {
    useGenerateForm().state.value.modelFamily = "flux2";
    useGenerateForm().state.value.model = "flux2-klein:q4";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const seqButton = wrapper
      .findAll("[data-test='composer-mode'] button")
      .find((b) => b.text() === "Sequence")!;
    await seqButton.trigger("click");
    // The Sequence tab stays reachable, but a non-chain model gets a clear
    // explanation — not the ScriptComposer with a live-looking Generate button.
    expect(wrapper.find("[data-test='chain-unsupported']").exists()).toBe(true);
    expect(wrapper.find("[data-test='script-composer']").exists()).toBe(false);
    // "back to single" returns to the composer.
    await wrapper.get("[data-test='chain-back-to-single']").trigger("click");
    expect(wrapper.find("[data-test='chain-unsupported']").exists()).toBe(
      false,
    );
  });

  it("prepares a batch on the server and queues provenance on every sibling", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    form.state.value.batchSize = 3;
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();
    expect(submitMock).not.toHaveBeenCalled();
    expect(expandPromptMock).toHaveBeenCalledWith(
      {
        prompt: "a lighthouse",
        model_family: "flux",
        variations: 3,
      },
      undefined,
      undefined,
    );
    expect(
      wrapper.getComponent({ name: "ResultCanvas" }).props("variations"),
    ).toEqual(["north light", "storm light", "harbor light"]);

    // Queue submits one single print per variation.
    await wrapper.get("[data-test='queue-variations']").trigger("click");
    await flushPromises();
    expect(submitMock).toHaveBeenCalledTimes(3);
    const batchIds = new Set<string>();
    for (const [index, call] of submitMock.mock.calls.entries()) {
      expect(call[0].batch_size).toBe(1);
      expect(call[0].prompt).toBe(
        ["north light", "storm light", "harbor light"][index],
      );
      expect(call[0].original_prompt).toBe("a lighthouse");
      expect(call[0].batch_index).toBe(index + 1);
      expect(call[0].batch_count).toBe(3);
      batchIds.add(call[0].batch_id);
    }
    expect(batchIds.size).toBe(1);
  });

  it("preserves reviewed variations as stale when the model changes", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    form.state.value.batchSize = 3;
    await nextTick();
    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();

    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    await nextTick();
    await wrapper.get("[data-test='queue-variations']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("Model changed");
    expect(
      wrapper.getComponent({ name: "ResultCanvas" }).props("variations"),
    ).toHaveLength(3);
  });

  it("sends the active style as a directive on the main-prompt expand", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    form.state.value.prompt = "a lighthouse";
    form.state.value.stylePreset = "cinematic";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();

    const modal = wrapper.getComponent({ name: "ExpandModal" });
    expect(modal.props("open")).toBe(true);
    // The chip travels as natural language the server weaves into the
    // expander's system message — never as a literal prompt suffix.
    expect(modal.props("styleDirective")).toBe(styleHint("cinematic"));
  });

  it("never steers a chain-stage expand with the composer's style chip", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "ltx2:q8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.stylePreset = "cinematic";
    await nextTick();

    const seqButton = wrapper
      .findAll("[data-test='composer-mode'] button")
      .find((b) => b.text() === "Sequence")!;
    await seqButton.trigger("click");
    await wrapper.get("[data-test='stage-expand']").trigger("click");
    await nextTick();

    const modal = wrapper.getComponent({ name: "ExpandModal" });
    expect(modal.props("prompt")).toBe("a stage prompt");
    // The style row belongs to the single-print composer, not to stage text.
    expect(modal.props("styleDirective")).toBeNull();
  });

  it("bakes and clears the chip when a quick expansion is applied", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    form.state.value.prompt = "a lighthouse";
    form.state.value.negativePrompt = "text";
    form.state.value.stylePreset = "cinematic";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();
    wrapper
      .getComponent({ name: "ExpandModal" })
      .vm.$emit("apply-prompt", "storm light over a cinematic coast");
    await nextTick();

    expect(form.state.value.prompt).toBe("storm light over a cinematic coast");
    // Bake-and-clear: the rewrite absorbed the look, so the chip drops — and
    // the curated negative moves into the form, its only remaining home.
    expect(form.state.value.stylePreset).toBeNull();
    expect(form.state.value.negativePrompt).toBe(
      "text, anime, cartoon, graphic, washed out",
    );
    // Applied exactly once — the cleared chip can't merge it again at submit.
    expect(form.toRequest().negative_prompt).toBe(
      "text, anime, cartoon, graphic, washed out",
    );

    await wrapper.get("[data-test='composer-undo']").trigger("click");
    await nextTick();
    expect(form.state.value.prompt).toBe("a lighthouse");
    expect(form.state.value.stylePreset).toBe("cinematic");
    expect(form.state.value.negativePrompt).toBe("text");
  });

  it("freezes a quick expansion to its host and sends original-prompt provenance", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "sk-studio",
    });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    hostModelsMock.mockResolvedValue([
      {
        name: "flux-dev:q4",
        family: "flux",
        description: "Flux Dev Q4",
        size_gb: 4,
        default_width: 1024,
        default_height: 1024,
        default_steps: 20,
        default_guidance: 3.5,
        is_loaded: false,
        hf_repo: "example/flux",
        downloaded: true,
      },
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    wrapper
      .getComponent({ name: "ExpandModal" })
      .vm.$emit("apply-prompt", "storm light over the harbor");
    await nextTick();
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(1);
    expect(submitMock.mock.calls[0]?.[0].original_prompt).toBe("a lighthouse");
    expect(submitMock.mock.calls[0]?.[2]).toEqual({
      hostId: studio.id,
      label: "Studio",
      target: { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
    });
  });

  it("keeps a stage expansion out of the composer's prompt and style", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "ltx2:q8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.prompt = "a lighthouse";
    form.state.value.stylePreset = "cinematic";
    await nextTick();

    const seqButton = wrapper
      .findAll("[data-test='composer-mode'] button")
      .find((b) => b.text() === "Sequence")!;
    await seqButton.trigger("click");
    await wrapper.get("[data-test='stage-expand']").trigger("click");
    wrapper
      .getComponent({ name: "ExpandModal" })
      .vm.$emit("apply-prompt", "a rewritten stage");
    await nextTick();

    expect(form.state.value.prompt).toBe("a lighthouse");
    expect(form.state.value.stylePreset).toBe("cinematic");
  });

  it("carries the preset negative when a variation is adopted into the composer", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    form.state.value.prompt = "a lighthouse";
    form.state.value.negativePrompt = "text";
    form.state.value.stylePreset = "cinematic";
    form.state.value.batchSize = 3;
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();
    const canvas = wrapper.getComponent({ name: "ResultCanvas" });
    const variations = canvas.props("variations") as string[];
    canvas.vm.$emit("use-variation", 0);
    await nextTick();

    // The variation already carries the baked look, so the chip clears — the
    // curated negative has to come with it.
    expect(form.state.value.prompt).toBe(variations[0]);
    expect(form.state.value.stylePreset).toBeNull();
    expect(form.state.value.negativePrompt).toBe(
      "text, anime, cartoon, graphic, washed out",
    );
  });

  it("resets to a fresh print on the mold:new-print event, keeping the model", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises(); // onMounted registers the mold:new-print listener
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    form.state.value.batchSize = 2;
    await nextTick();

    // Fan the batch out into variations so there's review state to clear.
    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();
    expect(
      wrapper.get("[data-test='result-canvas']").attributes("data-count"),
    ).toBe("2");

    window.dispatchEvent(new CustomEvent("mold:new-print"));
    await nextTick();

    expect(form.state.value.prompt).toBe("");
    // The selected model survives — New print is a fresh canvas, not a reset.
    expect(form.state.value.model).toBe("flux-dev:q4");
    // Variations cleared → the variations canvas gives way to the cold-start guide.
    expect(wrapper.find("[data-test='result-canvas']").exists()).toBe(false);
    expect(wrapper.find("[data-test='cold-start-stub']").exists()).toBe(true);
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
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(
      wrapper.getComponent({ name: "ChainJobCard" }).props("job"),
    ).toMatchObject({ id: "job-1", state: "queued" });
  });
});

// ── Multi-host generation routing (spec §08) ────────────────────────────────
describe("CreatePage host routing", () => {
  const flux = {
    name: "flux2-klein:q4",
    family: "flux2",
    description: "Flux.2 Klein Q4",
    size_gb: 4,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 3.5,
    is_loaded: false,
    hf_repo: "unsloth/FLUX.2-klein-4B-GGUF",
    downloaded: true,
  };
  const zimage = { ...flux, name: "z-image:bf16", family: "z-image" };

  beforeEach(async () => {
    // The routing singleton outlives a test's component; let any poll still in
    // flight from the previous test land, then discard what it wrote.
    hostRoutingTesting.reset();
    await flushPromises();
    hostRoutingTesting.reset();
    localStorage.clear();
    generateFormTesting.resetForTest();
    resetNotifications();
    submitMock.mockClear();
    hostStatusMock.mockReset();
    hostStatusMock.mockResolvedValue({
      version: "test",
      models_loaded: [],
      busy: false,
      uptime_secs: 1,
      queue_depth: 0,
    });
    hostModelsMock.mockReset();
    hostModelsMock.mockResolvedValue([]);
    vi.stubGlobal("prompt", vi.fn());
  });

  it("submits unrouted when this server is the only machine", async () => {
    hostModelsMock.mockResolvedValue([flux]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(1);
    // Third argument is the route — null means "stay on the serving origin".
    expect(submitMock.mock.calls[0]?.[2]).toBeNull();
  });

  it("dispatches to the pinned machine with its base URL and key", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "sk-studio",
    });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    hostModelsMock.mockResolvedValue([flux]);

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock.mock.calls[0]?.[2]).toEqual({
      hostId: studio.id,
      label: "Studio",
      target: { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
    });
  });

  it("refuses to reroute when the pinned machine is unreachable", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    hostModelsMock.mockResolvedValue([flux]);
    hostStatusMock.mockImplementation(async (host: { id: string }) => {
      if (host.id !== ORIGIN_HOST_ID) throw new Error("unreachable");
      return {
        version: "test",
        models_loaded: [],
        busy: false,
        uptime_secs: 1,
        queue_depth: 0,
      };
    });

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    const notifications = useNotifications();
    expect(notifications.toasts.map((t) => t.text).join(" ")).toMatch(
      /isn't reachable/i,
    );
  });

  // An offline pinned machine reports no models, so a model-first check would
  // blame the model. Name the machine — that's the thing the user can fix.
  it("blames the unreachable machine, not the empty model list", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    hostStatusMock.mockImplementation(async (host: { id: string }) => {
      if (host.id !== ORIGIN_HOST_ID) throw new Error("unreachable");
      return {
        version: "test",
        models_loaded: [],
        busy: false,
        uptime_secs: 1,
        queue_depth: 0,
      };
    });
    hostModelsMock.mockImplementation(async (host: { id: string }) => {
      if (host.id !== ORIGIN_HOST_ID) throw new Error("unreachable");
      return [flux];
    });

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    const notifications = useNotifications();
    expect(notifications.toasts.map((t) => t.text).join(" ")).toMatch(
      /isn't reachable/i,
    );
    // The generic "Pick a model to start." never fires in its place.
    expect(wrapper.find("[data-test='composer-submit-error']").exists()).toBe(
      false,
    );
  });

  it("routes Auto to the machine that already holds the model", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    localStorage.setItem("mold.web.generateTarget.v1", AUTO_TARGET_ID);
    // Only the remote has the weights; the origin is idle but empty.
    hostModelsMock.mockImplementation(async (host: { id: string }) =>
      host.id === ORIGIN_HOST_ID ? [] : [zimage],
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock.mock.calls[0]?.[2]).toMatchObject({ hostId: studio.id });
  });

  it("offers the union of every ready machine's models under Auto", async () => {
    addHost({ url: "http://studio:7680", name: "Studio" });
    localStorage.setItem("mold.web.generateTarget.v1", AUTO_TARGET_ID);
    hostModelsMock.mockImplementation(async (host: { id: string }) =>
      host.id === ORIGIN_HOST_ID ? [flux] : [zimage],
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    const picker = wrapper.getComponent({ name: "CreateModelPicker" });
    const names = (picker.props("models") as { name: string }[]).map(
      (m) => m.name,
    );
    expect(names.sort()).toEqual(["flux2-klein:q4", "z-image:bf16"]);
  });

  it("shows only the pinned machine's models", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    hostModelsMock.mockImplementation(async (host: { id: string }) =>
      host.id === ORIGIN_HOST_ID ? [flux] : [zimage],
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    const picker = wrapper.getComponent({ name: "CreateModelPicker" });
    expect(
      (picker.props("models") as { name: string }[]).map((m) => m.name),
    ).toEqual(["z-image:bf16"]);
  });

  // Regression: a persisted model the server no longer has left the <select>
  // with no matching <option>, which renders BLANK — the picker looked empty
  // even though models were installed.
  it("re-homes the form when the persisted model isn't installed", async () => {
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q8";
    form.state.value.modelFamily = "flux";
    hostModelsMock.mockResolvedValue([flux]);

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    await nextTick();

    expect(form.state.value.model).toBe("flux2-klein:q4");
    expect(
      wrapper.getComponent({ name: "CreateModelPicker" }).props("model"),
    ).toBe("flux2-klein:q4");
  });

  it("leaves the selection alone when the persisted model is installed", async () => {
    const form = useGenerateForm();
    form.state.value.model = "flux2-klein:q4";
    form.state.value.modelFamily = "flux2";
    hostModelsMock.mockResolvedValue([flux, zimage]);

    mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    await nextTick();

    expect(form.state.value.model).toBe("flux2-klein:q4");
  });

  it("keeps the cold-start guide hidden while a machine is still answering", async () => {
    addHost({ url: "http://studio:7680", name: "Studio" });
    const deferred: { release: (models: unknown[]) => void } = {
      release: () => {},
    };
    const pendingRemote = new Promise<unknown[]>((resolve) => {
      deferred.release = resolve;
    });
    hostModelsMock.mockImplementation((host: { id: string }) =>
      host.id === ORIGIN_HOST_ID ? Promise.resolve([]) : pendingRemote,
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(wrapper.find("[data-test='cold-start-stub']").exists()).toBe(false);

    deferred.release([]);
    await flushPromises();
    await nextTick();
    expect(wrapper.find("[data-test='cold-start-stub']").exists()).toBe(true);
  });

  it("says sequences stay on this server when the route points elsewhere", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    localStorage.setItem("mold.composer.mode", "script");
    hostModelsMock.mockResolvedValue([
      { ...flux, name: "ltx-2:fp8", family: "ltx2" },
    ]);

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    await nextTick();

    expect(wrapper.find("[data-test='sequence-origin-note']").exists()).toBe(
      true,
    );
  });

  it("routes Most capable to the strongest GPU", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    localStorage.setItem("mold.web.generateTarget.v1", CAPABLE_TARGET_ID);
    hostModelsMock.mockResolvedValue([flux]);
    hostStatusMock.mockImplementation(async (host: { id: string }) => ({
      version: "test",
      models_loaded: [],
      busy: false,
      uptime_secs: 1,
      queue_depth: 0,
      gpu_info:
        host.id === ORIGIN_HOST_ID
          ? { name: "Apple M3", vram_total_mb: 65536, vram_used_mb: 0 }
          : {
              name: "NVIDIA RTX 4090",
              vram_total_mb: 24576,
              vram_used_mb: 0,
              backend: "cuda",
            },
    }));

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock.mock.calls[0]?.[2]).toMatchObject({ hostId: studio.id });
  });
});

function pageStubs() {
  return {
    ColdStartGuide: {
      name: "ColdStartGuide",
      template: "<div data-test='cold-start-stub' />",
    },
    ComposerCard: {
      name: "ComposerCard",
      template:
        '<div><button data-test="composer-submit" @click="$emit(\'submit\')">go</button><button data-test="composer-expand" @click="$emit(\'expand\')">expand</button><button data-test="composer-undo" @click="$emit(\'undo-expand\')">undo</button></div>',
      // The page calls these through its template ref on submit / new-print;
      // a stub without them throws an unhandled TypeError mid-run.
      methods: { record: vi.fn(), focus: vi.fn() },
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
      template:
        "<div data-test='script-composer'><button data-test='stage-expand' @click=\"$emit('expand', 0, 'a stage prompt')\">expand stage</button></div>",
      methods: { setStagePrompt: vi.fn() },
    },
    ChainJobCard: {
      name: "ChainJobCard",
      props: ["job"],
      template: '<div data-test="chain-job-card">{{ job.id }}</div>',
    },
    ExpandModal: {
      name: "ExpandModal",
      props: [
        "open",
        "prompt",
        "expand",
        "currentModel",
        "queueBusy",
        "styleDirective",
        "target",
      ],
      template: "<div />",
    },
    ImagePickerModal: {
      name: "ImagePickerModal",
      props: ["open"],
      emits: ["pick", "close"],
      template: "<div />",
    },
    MaskEditorModal: { name: "MaskEditorModal", template: "<div />" },
    GenerationTemplatesPanel: { template: "<div />" },
    RecentGrid: RecentGridStub,
    Lightbox: { template: "<div />" },
    RouterLink: { template: "<a><slot /></a>" },
  };
}
