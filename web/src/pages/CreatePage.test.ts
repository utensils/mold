import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, nextTick, type Component } from "vue";
import { createPinia, setActivePinia } from "pinia";
import CreatePage from "./CreatePage.vue";
import {
  useGenerateForm,
  __testing__ as generateFormTesting,
} from "../composables/useGenerateForm";
import {
  resetNotifications,
  runToastAction,
  settleConfirm,
  useNotifications,
} from "../lib/toasts";
import { styleHint } from "../lib/stylePresets";
import { __testing__ as hostRoutingTesting } from "../composables/useHostRouting";
import {
  __testing__ as chainJobsTesting,
  useChainJobs,
} from "../composables/useChainJobs";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  pendingSequenceHandoff,
  setSequenceHandoff,
  takeSequenceHandoff,
} from "../composables/useSequenceHandoff";
import { ApiHttpError } from "../api";
import { addHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { AUTO_TARGET_ID, CAPABLE_TARGET_ID } from "../lib/hostRouting";
import type { GalleryImage, ModelInfoExtended, OutputMetadata } from "../types";
import type { Job } from "../composables/useGenerateStream";
import type { ChainJobDetail } from "@studio/lib/api/chainTypes";

const routeQuery = vi.hoisted(() => ({ value: {} as Record<string, unknown> }));
const routerReplaceMock = vi.hoisted(() =>
  vi.fn((to: { query?: Record<string, unknown> }) => {
    routeQuery.value = to.query ?? {};
    return Promise.resolve();
  }),
);
vi.mock("vue-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("vue-router")>()),
  useRoute: () =>
    new Proxy(
      {},
      { get: (_t, key) => (key === "query" ? routeQuery.value : undefined) },
    ),
  useRouter: () => ({ replace: routerReplaceMock, push: vi.fn() }),
}));

vi.mock("@microsoft/fetch-event-source", () => ({
  fetchEventSource: vi.fn(async () => undefined),
}));

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
const streamJobsRef = vi.hoisted(() => ({ value: [] as Job[] }));
const placementPreviewMock = vi.hoisted(() =>
  vi.fn(async () => ({
    version: 1,
    authoritative: true,
    state_version: 1,
    plan_version: 1,
    outcome: "planned",
    candidate: {
      device_id: "cuda:0",
      execution_fingerprint: "test",
      predicted_start_after_ms: 0,
      predicted_completion_after_ms: 100,
      setup_ms: 0,
      setup_kind: "warm",
      estimate_confidence: "high",
    },
  })),
);

vi.mock("@studio/api/generationPlacement", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationPlacement")>()),
  previewGenerationPlacement: placementPreviewMock,
  previewChainPlacement: placementPreviewMock,
}));
const fetchChainLimitsMock = vi.hoisted(() =>
  vi.fn(async () => ({
    model: "ltx-2-19b-distilled:fp8",
    frames_per_clip_cap: 97,
    frames_per_clip_recommended: 97,
    max_stages: 16,
    max_total_frames: 97 * 16,
    fade_frames_max: 32,
    transition_modes: ["smooth", "cut", "fade"],
    quantization_family: "fp8",
    supports_audio: true,
    supports_sequence: true,
  })),
);
const listChainJobsMock = vi.hoisted(() =>
  vi.fn(async () => ({ jobs: [] as unknown[] })),
);
const getChainJobMock = vi.hoisted(() => vi.fn());
const cancelChainJobMock = vi.hoisted(() => vi.fn(async () => ({})));
const resumeChainJobMock = vi.hoisted(() => vi.fn(async () => ({})));
const deleteChainJobMock = vi.hoisted(() => vi.fn(async () => undefined));
const gcChainJobsMock = vi.hoisted(() =>
  vi.fn(async () => ({ swept_ephemeral_jobs: 0, pruned_artifact_dirs: 0 })),
);
const amendChainJobMock = vi.hoisted(() => vi.fn());

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    // Real error class so `instanceof ApiHttpError` branches stay honest.
    ApiHttpError: actual.ApiHttpError,
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
    fetchChainLimits: fetchChainLimitsMock,
    listChainJobs: listChainJobsMock,
    getChainJob: getChainJobMock,
    cancelChainJob: cancelChainJobMock,
    resumeChainJob: resumeChainJobMock,
    retakeChainJob: vi.fn(async () => ({})),
    deleteChainJob: deleteChainJobMock,
    gcChainJobs: gcChainJobsMock,
    amendChainJob: amendChainJobMock,
    chainJobEventsUrl: (id: string) => `/api/chain-jobs/${id}/events`,
    chainJobStagePreviewUrl: (id: string, idx: number) =>
      `/api/chain-jobs/${id}/stages/${idx}/preview`,
  };
});

vi.mock("../composables/useGenerateStream", async (importOriginal) => ({
  // Keep the real pure helpers (activeCanvasJob) — only the singleton
  // stream is replaced.
  ...(await importOriginal<
    typeof import("../composables/useGenerateStream")
  >()),
  useGenerateStream: () => ({
    jobs: streamJobsRef,
    submit: submitMock,
    cancel: vi.fn(),
    remove: vi.fn(),
    clearDone: vi.fn(),
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
  hostQueue: () => Promise.resolve({ entries: [], plan: null }),
  hostDevices: () => Promise.reject(new Error("legacy server in tests")),
  hostModelComponents: (_host: unknown, model: string) =>
    Promise.resolve({ model, components: [] }),
  hostGenerationEstimate: (_host: unknown, request: { model: string }) =>
    Promise.resolve({
      model: request.model,
      peak_memory_bytes: 1,
      activation_memory_bytes: 1,
      fits_available_memory: true,
      load_strategy: "resident",
    }),
}));

const RecentGridStub = defineComponent({
  name: "RecentGrid",
  props: {
    entries: { type: Array, required: true },
    limit: { type: Number, default: undefined },
  },
  template: '<div data-test="recent-grid">{{ entries.length }}</div>',
});

/** Flip the shared draft store into Sequence output (the Output card's job)
 * without going through a mounted ControlsAside. */
function enterSequenceMode() {
  const draft = useSequenceDraftStore();
  draft.hydrate();
  draft.setOutput("sequence", { getPrompt: () => "", setPrompt: () => {} }, 97);
  return draft;
}

function installedSequenceModel() {
  return {
    name: "ltx-2-19b-distilled:fp8",
    family: "ltx2",
    downloaded: true,
    supports_sequence: true,
    default_width: 1216,
    default_height: 704,
    default_steps: 8,
    default_guidance: 3,
  };
}

describe("CreatePage layout and behavior", () => {
  beforeEach(async () => {
    // The routing singleton outlives a test's component; let any poll still in
    // flight from the previous test land, then discard what it wrote.
    hostRoutingTesting.reset();
    await flushPromises();
    hostRoutingTesting.reset();
    localStorage.clear();
    setActivePinia(createPinia());
    chainJobsTesting.reset();
    takeSequenceHandoff();
    generateFormTesting.resetForTest();
    resetNotifications();
    submitMock.mockClear();
    streamJobsRef.value = [];
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
    placementPreviewMock.mockClear();
    fetchChainLimitsMock.mockClear();
    listChainJobsMock.mockClear();
    listChainJobsMock.mockResolvedValue({ jobs: [] });
    getChainJobMock.mockReset();
    cancelChainJobMock.mockClear();
    resumeChainJobMock.mockClear();
    deleteChainJobMock.mockClear();
    gcChainJobsMock.mockClear();
    amendChainJobMock.mockReset();
    routeQuery.value = {};
    routerReplaceMock.mockClear();
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

  it("uses a human-readable catalog model in the completed generation caption", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "cv:1759168",
        family: "sdxl",
        display_name: "Juggernaut XL - Ragnarok",
        description: "Juggernaut XL - Ragnarok by RunDiffusion",
        size_gb: 6.9,
        default_width: 1024,
        default_height: 1024,
        default_steps: 25,
        default_guidance: 7,
        is_loaded: true,
        hf_repo: "",
        downloaded: true,
      },
    ]);
    streamJobsRef.value = [
      {
        id: "done-1",
        request: {
          model: "cv:1759168",
          prompt: "a camera",
          width: 1024,
          height: 1024,
          steps: 25,
          guidance: 7,
          batch_size: 1,
          output_format: "png",
        },
        startedAt: 1,
        controller: new AbortController(),
        progress: {
          stage: "complete",
          step: 25,
          totalSteps: 25,
          weightBytesLoaded: null,
          weightBytesTotal: null,
          queuePosition: null,
          gpu: null,
          elapsedMs: 11_800,
        },
        result: {
          type: "complete",
          image: "image-bytes",
          format: "png",
          seed_used: 42,
          model: "cv:1759168",
          width: 1024,
          height: 1024,
          generation_time_ms: 11_800,
        },
        error: null,
        state: "done",
        settledAt: Date.now(),
        chain: null,
        lastProgressAt: Date.now(),
        workStarted: true,
        hostId: null,
        hostLabel: null,
        target: null,
        serverId: "server-1",
        previewUrl: null,
        seedVisual: "42",
      } as Job,
    ];

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    const caption = String(
      wrapper.getComponent({ name: "ResultCanvas" }).props("resultCaption"),
    );
    expect(caption).toContain("Juggernaut XL - Ragnarok");
    expect(caption).not.toContain("cv:1759168");
  });

  it("orders phone Create as prompt, controls, actions, canvas, then recent", async () => {
    vi.stubGlobal(
      "matchMedia",
      vi.fn(() => ({
        matches: true,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      })),
    );
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    expect(wrapper.find("[data-test='phone-create-title']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='phone-create-controls']").exists()).toBe(
      true,
    );
    await flushPromises();
    const canvas = wrapper.find("[data-test='result-canvas']").exists()
      ? wrapper.get("[data-test='result-canvas']").element
      : wrapper.get("[data-test='cold-start-stub']").element;
    const markers = [
      "phone-create-title",
      "prompt-style-stub",
      "model-picker-stub",
      "controls-stub",
      "composer-submit",
      "recent-grid",
    ].map((test) => wrapper.get(`[data-test='${test}']`).element);
    markers.splice(markers.length - 1, 0, canvas);
    for (let index = 1; index < markers.length; index += 1) {
      expect(
        markers[index - 1]!.compareDocumentPosition(markers[index]!) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }
    wrapper.unmount();
    vi.unstubAllGlobals();
    vi.stubGlobal("prompt", vi.fn());
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

  it("resets the rail settings to the model defaults, undoably", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "sdxl:fp16",
        family: "sdxl",
        size_gb: 6,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 30,
        default_guidance: 7.5,
        default_width: 1024,
        default_height: 1024,
        description: "SDXL",
      },
    ]);
    const stubs: Record<string, Component> = pageStubs();
    stubs.ControlsAside = defineComponent({
      name: "ControlsAside",
      template:
        "<aside data-test='controls-stub'><button data-test='controls-reset' @click=\"$emit('reset-settings')\">reset</button></aside>",
    });
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();

    const form = useGenerateForm();
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.steps = 3;
    form.state.value.guidance = 11;
    form.state.value.seedMode = "static";
    form.state.value.seed = 42;
    form.state.value.negativePrompt = "blurry";

    await wrapper.get("[data-test='controls-reset']").trigger("click");
    expect(form.state.value.steps).toBe(30);
    expect(form.state.value.guidance).toBe(7.5);
    expect(form.state.value.seedMode).toBe("random");
    expect(form.state.value.negativePrompt).toBe("");
    expect(form.state.value.prompt).toBe("a lighthouse in a storm");
    expect(form.state.value.model).toBe("sdxl:fp16");

    const notifications = useNotifications();
    const settingsToast = notifications.toasts.find((t) =>
      /settings/i.test(t.text),
    );
    expect(settingsToast?.actionLabel).toBe("Undo");
    runToastAction(settingsToast!.id);
    expect(form.state.value.steps).toBe(3);
    expect(form.state.value.seed).toBe(42);
    expect(form.state.value.negativePrompt).toBe("blurry");
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

  it("keeps the visible Output control synchronized with sequence mode", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    enterSequenceMode();
    await flushPromises();

    const controls = wrapper.getComponent({ name: "ControlsAside" });
    expect(controls.props("output")).toBe("sequence");
    expect(controls.props("clipCount")).toBe(2);
  });

  it("selects a chain-capable model from the pinned host when Sequence is restored", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
    });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    const still = {
      name: "flux-schnell:q8",
      family: "flux",
      downloaded: true,
      default_width: 1024,
      default_height: 1024,
      default_steps: 4,
      default_guidance: 1,
    };
    const video = {
      name: "ltx-2.3-22b-distilled:fp8",
      family: "ltx2",
      downloaded: true,
      supports_sequence: true,
      default_width: 1216,
      default_height: 704,
      default_steps: 8,
      default_guidance: 3,
      default_frames: 97,
    };
    hostModelsMock.mockImplementation(async (host: { id: string }) =>
      host.id === studio.id ? [still, video] : [still],
    );
    const form = useGenerateForm();
    form.state.value.model = still.name;
    form.state.value.modelFamily = still.family;
    enterSequenceMode();

    mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    expect(form.state.value.model).toBe(video.name);
    expect(form.state.value.modelFamily).toBe("ltx2");
  });

  it("clears an image-only pinned-host model and renders the browse state without submitting", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
    });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    const still = {
      name: "flux-schnell:q8",
      family: "flux",
      downloaded: true,
      default_width: 1024,
      default_height: 1024,
      default_steps: 4,
      default_guidance: 1,
    };
    const videoElsewhere = {
      name: "ltx-video",
      family: "ltx-video",
      downloaded: true,
      supports_sequence: true,
      default_width: 1024,
      default_height: 576,
      default_steps: 25,
      default_guidance: 3,
    };
    hostModelsMock.mockImplementation(async (host: { id: string }) =>
      host.id === studio.id ? [still] : [videoElsewhere],
    );
    const form = useGenerateForm();
    form.state.value.model = still.name;
    form.state.value.modelFamily = still.family;
    enterSequenceMode();

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    expect(form.state.value.model).toBe("");
    expect(
      wrapper.get("[data-test='chain-unsupported']").attributes("data-empty"),
    ).toBe("true");
    expect(wrapper.get("[data-test='chain-unsupported']").text()).toContain(
      "No chain-capable video model is installed on the selected machine",
    );
    expect(wrapper.find("[data-test='sequence-generate']").exists()).toBe(
      false,
    );
    expect(createChainJobMock).not.toHaveBeenCalled();
  });

  it("submits the sequence with the LIVE inspector values (stale-inspector regression)", async () => {
    hostModelsMock.mockResolvedValue([installedSequenceModel()]);
    useGenerateForm().state.value.modelFamily = "ltx2";
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const draft = enterSequenceMode();
    draft.clips[0]!.prompt = "the opening";
    draft.clips[1]!.prompt = "the landing";
    // Turn the inspector's knobs WHILE in sequence mode — the old
    // ScriptComposer kept private copies that silently ignored these.
    const form = useGenerateForm();
    form.state.value.width = 1216;
    form.state.value.height = 704;
    form.state.value.steps = 4;
    form.state.value.guidance = 5.5;
    form.state.value.fps = 24;
    await flushPromises();

    await wrapper.get("[data-test='sequence-generate']").trigger("click");
    await flushPromises();

    expect(createChainJobMock).toHaveBeenCalledTimes(1);
    expect(createChainJobMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "ltx-2-19b-distilled:fp8",
        width: 1216,
        height: 704,
        steps: 4,
        guidance: 5.5,
        fps: 24,
        output_format: "mp4",
        stages: [
          expect.objectContaining({ prompt: "the opening" }),
          expect.objectContaining({ prompt: "the landing" }),
        ],
      }),
      expect.objectContaining({ baseUrl: expect.any(String) }),
    );
    expect(placementPreviewMock).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: expect.any(String) }),
      expect.objectContaining({
        model: "ltx-2-19b-distilled:fp8",
        stages: [
          expect.objectContaining({ prompt: "the opening" }),
          expect.objectContaining({ prompt: "the landing" }),
        ],
      }),
      1,
    );
    expect(submitMock).not.toHaveBeenCalled();
  });

  it("keeps a successfully submitted sequence when tracked-job persistence fails", async () => {
    hostModelsMock.mockResolvedValue([installedSequenceModel()]);
    useGenerateForm().state.value.modelFamily = "ltx2";
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const draft = enterSequenceMode();
    draft.clips[0]!.prompt = "one";
    draft.clips[1]!.prompt = "two";
    await flushPromises();
    const setItem = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(function (this: Storage, key, value) {
        if (key === "mold.create.tracked-sequences.v1")
          throw new DOMException("blocked", "QuotaExceededError");
        return Reflect.apply(
          Object.getOwnPropertyDescriptor(Storage.prototype, "setItem")!.value,
          this,
          [key, value],
        );
      });

    await wrapper.get("[data-test='sequence-generate']").trigger("click");
    await flushPromises();

    // The durable job was created and is being watched despite storage
    // being unavailable — persistence is recovery convenience only.
    expect(createChainJobMock).toHaveBeenCalledTimes(1);
    expect(useChainJobs().state.watching).toMatchObject({ jobId: "job-1" });
    setItem.mockRestore();
  });

  it("redirects legacy ?mode=sequence deep links to ?output=sequence", async () => {
    routeQuery.value = { mode: "sequence" };
    mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(routerReplaceMock).toHaveBeenCalledWith({
      query: { output: "sequence" },
    });
  });

  it("consumes ?output=sequence once and strips it from the URL", async () => {
    useGenerateForm().state.value.modelFamily = "ltx2";
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    routeQuery.value = { output: "sequence" };
    mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(useSequenceDraftStore().output).toBe("sequence");
    expect(routerReplaceMock).toHaveBeenCalledWith({ query: {} });
  });

  it("reuses a Library sequence print as a NEW draft with no edit session", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "ltx-2.3-22b-distilled:fp8",
        family: "ltx2",
        size_gb: 35,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 8,
        default_guidance: 3,
        default_width: 1216,
        default_height: 704,
        default_frames: 97,
        description: "Sequence model",
        supports_sequence: true,
      },
    ]);
    setSequenceHandoff({
      kind: "reuse",
      metadata: {
        // A sequence print records every clip's prompt newline-joined; the
        // reuse path must never surface that join.
        prompt: "a harbour at dawn\nthe boats leave",
        model: "ltx-2.3-22b-distilled:fp8",
        seed: 4242,
        steps: 8,
        guidance: 3,
        width: 1216,
        height: 704,
        negative_prompt: "blurry",
        chain_job_id: "job-9",
        chain: {
          stage_count: 2,
          motion_tail_frames: 17,
          stages: [
            { prompt: "a harbour at dawn", frames: 97, transition: "smooth" },
            { prompt: "the boats leave", frames: 9, transition: "cut" },
          ],
        },
      } as OutputMetadata,
    });
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    const draft = useSequenceDraftStore();
    expect(draft.output).toBe("sequence");
    expect(draft.clips.map((c) => c.prompt)).toEqual([
      "a harbour at dawn",
      "the boats leave",
    ]);
    // Clip 2 was 9 frames — at/below the model's 17-frame tail — so it is
    // raised, and the surface says so instead of silently resizing.
    expect(draft.clips[1]!.frames).toBeGreaterThan(17);
    expect(draft.editing).toBeNull();
    expect(useGenerateForm().state.value.prompt).toBe("a harbour at dawn");
    const note = wrapper.get("[data-test='sequence-reuse-note']").text();
    expect(note).toContain("reused 2 clips");
    expect(note).toContain("Clip durations raised to fit");
    // One-shot: a back-nav must not replay the handoff.
    expect(pendingSequenceHandoff().value).toBeNull();
  });

  it("explains Sequence for a non-chain model instead of a dead composer", async () => {
    useGenerateForm().state.value.modelFamily = "flux2";
    useGenerateForm().state.value.model = "flux2-klein:q4";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    enterSequenceMode();
    await nextTick();
    // Sequence output stays reachable, but a non-chain model gets a clear
    // explanation — not a composer with a live-looking Generate button.
    expect(wrapper.find("[data-test='chain-unsupported']").exists()).toBe(true);
    expect(wrapper.find("[data-test='sequence-composer-stub']").exists()).toBe(
      false,
    );
    // "back to one shot" returns to the composer.
    await wrapper.get("[data-test='chain-back-to-single']").trigger("click");
    expect(wrapper.find("[data-test='chain-unsupported']").exists()).toBe(
      false,
    );
  });

  it("switches an image model to an installed sequence-compatible model and filters the picker", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "flux2-klein:q4",
        family: "flux2",
        size_gb: 4,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 4,
        default_guidance: 1,
        default_width: 1024,
        default_height: 1024,
        description: "Image model",
        supports_sequence: false,
      },
      {
        name: "ltx-2.3-22b-dev:fp8",
        family: "ltx2",
        size_gb: 34,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 20,
        default_guidance: 3,
        default_width: 1216,
        default_height: 704,
        description: "Two-stage video model",
        supports_sequence: false,
      },
      {
        name: "ltx-2.3-22b-distilled:fp8",
        family: "ltx2",
        size_gb: 35,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 8,
        default_guidance: 3,
        default_width: 1216,
        default_height: 704,
        description: "Sequence model",
        supports_sequence: true,
      },
    ]);
    useGenerateForm().state.value.model = "flux2-klein:q4";
    useGenerateForm().state.value.modelFamily = "flux2";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    enterSequenceMode();
    await flushPromises();

    expect(useGenerateForm().state.value.model).toBe(
      "ltx-2.3-22b-distilled:fp8",
    );
    const picker = wrapper.getComponent({ name: "CreateModelPicker" });
    expect(
      (picker.props("models") as ModelInfoExtended[]).map(
        (model) => model.name,
      ),
    ).toEqual(["ltx-2.3-22b-distilled:fp8"]);
    expect(wrapper.find("[data-test='chain-unsupported']").exists()).toBe(
      false,
    );
  });

  it("links an empty sequence setup to filtered video checkpoint discovery", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "flux2-klein:q4",
        family: "flux2",
        size_gb: 4,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 4,
        default_guidance: 1,
        default_width: 1024,
        default_height: 1024,
        description: "Image model",
        supports_sequence: false,
      },
    ]);
    useGenerateForm().state.value.model = "flux2-klein:q4";
    useGenerateForm().state.value.modelFamily = "flux2";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    enterSequenceMode();
    await nextTick();

    expect(
      wrapper.get("[data-test='browse-sequence-models']").attributes("to"),
    ).toBe("/models?tab=discover&type=video&kind=checkpoint&intent=sequence");
  });

  it("does not treat a restored 19B dev checkpoint as sequence-capable before inventory loads", async () => {
    useGenerateForm().state.value.model = "ltx-2-19b-dev:fp8";
    useGenerateForm().state.value.modelFamily = "ltx2";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    enterSequenceMode();
    await nextTick();

    expect(wrapper.find("[data-test='chain-unsupported']").exists()).toBe(true);
  });

  it("fans an ordinary batch out into one routed request per print", async () => {
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
    form.state.value.prompt = "three lighthouses";
    form.state.value.batchSize = 3;
    form.state.value.seedMode = "static";
    form.state.value.seed = 41;
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(3);
    const batchIds = new Set<string>();
    for (const [index, call] of submitMock.mock.calls.entries()) {
      expect(call[0]).toMatchObject({
        prompt: "three lighthouses",
        batch_size: 1,
        batch_index: index + 1,
        batch_count: 3,
        seed: 41 + index,
      });
      batchIds.add(call[0].batch_id);
    }
    expect(batchIds.size).toBe(1);
    expect(placementPreviewMock).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ batch_size: 1 }),
      3,
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

  it.each([
    [
      "authoritative infeasible",
      {
        version: 1,
        authoritative: true,
        state_version: 2,
        plan_version: 2,
        outcome: "infeasible",
        candidate: null,
        reason: "insufficient_vram",
      },
    ],
    ["null", null],
    ["malformed unsupported", { outcome: "unsupported" }],
    [
      "planned without a candidate",
      {
        version: 1,
        authoritative: true,
        state_version: 2,
        plan_version: 2,
        outcome: "planned",
      },
    ],
  ])(
    "preserves every reviewed variation and submits zero siblings after %s revalidation",
    async (_case, response) => {
      const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
      const form = useGenerateForm();
      form.state.value.model = "flux-dev:q4";
      form.state.value.modelFamily = "flux";
      form.state.value.prompt = "a lighthouse";
      form.state.value.batchSize = 3;
      await nextTick();
      await wrapper.get("[data-test='composer-expand']").trigger("click");
      await flushPromises();
      placementPreviewMock.mockResolvedValueOnce(response as never);

      await wrapper.get("[data-test='queue-variations']").trigger("click");
      await flushPromises();

      expect(submitMock).not.toHaveBeenCalled();
      expect(
        wrapper.getComponent({ name: "ResultCanvas" }).props("variations"),
      ).toHaveLength(3);
      expect(wrapper.text()).toContain("Nothing was queued");
    },
  );

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

  it("never steers a clip expand with the composer's style chip", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "ltx2:q8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.stylePreset = "cinematic";
    await nextTick();

    enterSequenceMode();
    await nextTick();
    await wrapper.get("[data-test='clip-expand']").trigger("click");
    await nextTick();

    const modal = wrapper.getComponent({ name: "ExpandModal" });
    expect(modal.props("prompt")).toBe("a stage prompt");
    // The style row belongs to the single-print composer, not to clip text.
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

  it("offers a readable explicit override for a stale quick expansion", async () => {
    const fluxModel = {
      name: "flux-dev:q4",
      family: "flux",
      description: "Inverse Mix",
      size_gb: 4,
      default_width: 1024,
      default_height: 1024,
      default_steps: 20,
      default_guidance: 3.5,
      is_loaded: false,
      hf_repo: "",
      downloaded: true,
      last_used: null,
    };
    const catalogModel = {
      ...fluxModel,
      name: "cv:1759168",
      family: "sdxl",
      display_name: "Juggernaut XL - Ragnarok",
    };
    hostModelsMock.mockResolvedValue([fluxModel, catalogModel]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = fluxModel.name;
    form.state.value.modelFamily = fluxModel.family;
    form.state.value.prompt = "a lighthouse";
    await nextTick();
    await wrapper.get("[data-test='composer-expand']").trigger("click");
    wrapper
      .getComponent({ name: "ExpandModal" })
      .vm.$emit("apply-prompt", "storm light over the harbor");
    await nextTick();
    form.applyModelDefaults(catalogModel);
    await nextTick();

    const notice = wrapper.get("[data-test='web-quick-expansion-stale']");
    expect(notice.text()).toContain("Juggernaut XL - Ragnarok");
    expect(notice.text()).not.toContain("cv:1759168");
    await notice
      .get("[data-test='web-generate-expanded-anyway']")
      .trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(1);
    expect(submitMock.mock.calls[0]?.[0]).toMatchObject({
      model: catalogModel.name,
      prompt: "storm light over the harbor",
      original_prompt: "a lighthouse",
    });
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
      instanceId: null,
      target: { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
    });
  });

  it("applies a clip expansion to the clip, never the composer's prompt or style", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    const form = useGenerateForm();
    form.state.value.model = "ltx2:q8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.prompt = "a lighthouse";
    form.state.value.stylePreset = "cinematic";
    await nextTick();

    const draft = enterSequenceMode();
    await nextTick();
    const clip = draft.clips[0]!;
    wrapper
      .getComponent({ name: "SequenceComposer" })
      .vm.$emit("expand-clip", clip.id, clip.prompt);
    await nextTick();
    wrapper
      .getComponent({ name: "ExpandModal" })
      .vm.$emit("apply-prompt", "a rewritten stage");
    await nextTick();

    expect(clip.prompt).toBe("a rewritten stage");
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
    await flushPromises();
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
    await flushPromises();
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

  it("feeds durable sequence jobs from every host into the activity strip", async () => {
    listChainJobsMock.mockResolvedValue({
      jobs: [
        {
          id: "chain-9",
          state: "running",
          model: "ltx-2-19b-distilled:fp8",
          stage_count: 3,
          current_stage: 1,
          created_at_unix_ms: 5,
          updated_at_unix_ms: 5,
        },
      ],
    });
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const strip = wrapper.getComponent({ name: "ActivityStrip" });
    expect(
      (strip.props("sequences") as { jobId: string }[]).map((s) => s.jobId),
    ).toEqual(["chain-9"]);
  });

  it("clears stale advanced fields when a selected job omits optional keys", async () => {
    const job = {
      id: "print-9",
      request: {
        prompt: "a clean request",
        model: "flux-dev:q4",
        width: 768,
        height: 512,
        steps: 20,
        guidance: 3,
      },
      startedAt: 0,
      controller: new AbortController(),
      progress: {
        stage: "Queued",
        step: null,
        totalSteps: null,
        weightBytesLoaded: null,
        weightBytesTotal: null,
        queuePosition: null,
        gpu: null,
        elapsedMs: null,
      },
      result: null,
      error: null,
      state: "running",
      chain: null,
      lastProgressAt: 0,
      workStarted: false,
      serverId: null,
    } as Job;
    streamJobsRef.value = [job];
    const form = useGenerateForm();
    form.state.value.negativePrompt = "stale negative";
    form.state.value.controlModel = "stale-control";
    form.state.value.frames = 97;
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "stale.png", base64: "stale" },
    ];
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    wrapper.getComponent({ name: "ActivityStrip" }).vm.$emit("open", job);
    await nextTick();

    expect(form.state.value.prompt).toBe("a clean request");
    expect(form.state.value.negativePrompt).toBe("");
    expect(form.state.value.controlModel).toBe("");
    expect(form.state.value.frames).toBeNull();
    expect(form.state.value.imageAttachments).toEqual([]);
  });

  it("deletes settled durable jobs through the strip's Delete action", async () => {
    listChainJobsMock.mockResolvedValue({
      jobs: [
        {
          id: "failed-job",
          state: "failed",
          model: "ltx-2.3-22b-dev:fp8",
          stage_count: 1,
          current_stage: 0,
          created_at_unix_ms: 1,
          updated_at_unix_ms: 2,
          error: "TwoStage is unsupported",
        },
      ],
    });
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    wrapper
      .getComponent({ name: "ActivityStrip" })
      .vm.$emit("sequence-action", "delete", {
        kind: "sequence",
        hostId: ORIGIN_HOST_ID,
        jobId: "failed-job",
      });
    await flushPromises();

    expect(deleteChainJobMock).toHaveBeenCalledWith(
      "failed-job",
      expect.objectContaining({ baseUrl: expect.any(String) }),
    );
    expect(cancelChainJobMock).not.toHaveBeenCalled();
  });

  it("reattaches to a tracked in-flight sequence after a reload", async () => {
    localStorage.setItem(
      "mold.create.tracked-sequences.v1",
      JSON.stringify([{ hostId: ORIGIN_HOST_ID, jobId: "durable-job-7" }]),
    );
    listChainJobsMock.mockResolvedValue({
      jobs: [
        {
          id: "durable-job-7",
          state: "running",
          model: "ltx-2-19b-distilled:fp8",
          stage_count: 2,
          current_stage: 0,
          created_at_unix_ms: 1,
          updated_at_unix_ms: 1,
        },
      ],
    });
    mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(useChainJobs().state.watching).toEqual({
      hostId: ORIGIN_HOST_ID,
      jobId: "durable-job-7",
    });
  });

  it("loads a durable job into an edit session and amends through it", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "ltx-2-19b-distilled:fp8",
        family: "ltx2",
        size_gb: 20,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 8,
        default_guidance: 3,
        default_width: 1216,
        default_height: 704,
        description: "Sequence model",
        supports_sequence: true,
      },
    ]);
    getChainJobMock.mockResolvedValue({
      id: "job-9",
      state: "completed",
      model: "ltx-2-19b-distilled:fp8",
      stage_count: 2,
      current_stage: 1,
      created_at_unix_ms: 1,
      updated_at_unix_ms: 2,
      error: null,
      stages: [
        { idx: 0, state: "completed" },
        { idx: 1, state: "completed" },
      ],
      script: {
        schema: "mold.chain.v1",
        chain: {
          model: "ltx-2-19b-distilled:fp8",
          width: 640,
          height: 384,
          fps: 24,
          seed: 7,
          steps: 6,
          guidance: 4,
          strength: 1,
          motion_tail_frames: 17,
          output_format: "mp4",
        },
        stages: [
          { prompt: "one", frames: 97 },
          { prompt: "two", frames: 97, transition: "cut" },
        ],
      },
    });
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    wrapper
      .getComponent({ name: "ActivityStrip" })
      .vm.$emit("sequence-action", "edit", {
        kind: "sequence",
        hostId: ORIGIN_HOST_ID,
        jobId: "job-9",
      });
    await flushPromises();

    const draft = useSequenceDraftStore();
    expect(draft.editing).toMatchObject({
      jobId: "job-9",
      hostId: ORIGIN_HOST_ID,
      completedStages: 2,
    });
    expect(draft.clips.map((c) => c.prompt)).toEqual(["one", "two"]);
    expect(draft.clips[1]?.transition).toBe("cut");
    // The job's shared params landed on the LIVE form.
    const form = useGenerateForm();
    expect(form.state.value.width).toBe(640);
    expect(form.state.value.height).toBe(384);
    expect(form.state.value.steps).toBe(6);
    expect(form.state.value.seedMode).toBe("static");
    expect(form.state.value.seed).toBe(7);

    amendChainJobMock.mockResolvedValue({
      id: "job-9",
      state: "queued",
      model: "ltx-2-19b-distilled:fp8",
      stage_count: 2,
      current_stage: 0,
      created_at_unix_ms: 1,
      updated_at_unix_ms: 3,
      preserved_stages: 1,
    });
    draft.clips[1]!.prompt = "two, but stormier";
    await wrapper.get("[data-test='sequence-generate']").trigger("click");
    await flushPromises();

    expect(amendChainJobMock).toHaveBeenCalledWith(
      "job-9",
      expect.objectContaining({
        stages: [
          expect.objectContaining({ prompt: "one" }),
          expect.objectContaining({ prompt: "two, but stormier" }),
        ],
      }),
      expect.objectContaining({ baseUrl: expect.any(String) }),
    );
    expect(createChainJobMock).not.toHaveBeenCalled();
    expect(draft.editing).toBeNull();
  });

  it("preserves the edit session when an amend conflicts (409)", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "ltx-2-19b-distilled:fp8",
        family: "ltx2",
        size_gb: 20,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 8,
        default_guidance: 3,
        default_width: 1216,
        default_height: 704,
        description: "Sequence model",
        supports_sequence: true,
      },
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const draft = enterSequenceMode();
    draft.clips[0]!.prompt = "one";
    draft.clips[1]!.prompt = "two";
    draft.loadFromJob(
      {
        jobId: "job-9",
        hostId: ORIGIN_HOST_ID,
        baseline: draft.clips.map((c) => ({ ...c })),
        completedStages: 1,
      },
      draft.clips.map((c) => ({ ...c })),
      false,
    );
    await flushPromises();
    amendChainJobMock.mockRejectedValue(
      new ApiHttpError("POST /api/chain-jobs/job-9/amend", 409, "moved on"),
    );

    await wrapper.get("[data-test='sequence-generate']").trigger("click");
    await flushPromises();

    expect(amendChainJobMock).toHaveBeenCalledTimes(1);
    expect(createChainJobMock).not.toHaveBeenCalled();
    expect(draft.editing).toMatchObject({
      jobId: "job-9",
      hostId: ORIGIN_HOST_ID,
    });
    expect(wrapper.get("[data-test='sequence-submit-error']").text()).toContain(
      "Your edits are still here",
    );
  });

  it("keeps one stage playing while later stages finish and stops invalidated media", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "ltx-2-19b-distilled:fp8",
        family: "ltx2",
        downloaded: true,
        supports_sequence: true,
        default_width: 1216,
        default_height: 704,
        default_steps: 8,
        default_guidance: 3,
      },
    ]);
    const draft = enterSequenceMode();
    draft.clips[0]!.prompt = "opening";
    draft.clips[1]!.prompt = "arrival";
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    useGenerateForm().state.value.modelFamily = "ltx2";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    const chains = useChainJobs();
    chains.watch(ORIGIN_HOST_ID, "job-playback");
    const playbackDetail = {
      id: "job-playback",
      state: "running",
      model: "ltx-2-19b-distilled:fp8",
      stage_count: 2,
      current_stage: 1,
      created_at_unix_ms: 1,
      updated_at_unix_ms: 2,
      error: null,
      stages: [
        {
          idx: 0,
          state: "completed",
          seed: "1",
          frames_emitted: 97,
          generation_time_ms: 1,
          has_preview: false,
          has_media: true,
          cache_ready: true,
          error: null,
        },
        {
          idx: 1,
          state: "running",
          seed: "2",
          frames_emitted: null,
          generation_time_ms: null,
          has_preview: false,
          has_media: false,
          cache_ready: false,
          error: null,
        },
      ],
      script: {
        schema: "mold.chain.v1",
        chain: { model: "ltx-2-19b-distilled:fp8", fps: 24 },
        stages: [
          { prompt: "opening", frames: 97 },
          { prompt: "arrival", frames: 97 },
        ],
      },
    } satisfies ChainJobDetail;
    chains.state.live.detail = playbackDetail;
    await flushPromises();

    await wrapper.get("[data-test='stub-play-first']").trigger("click");
    await flushPromises();
    const firstSrc = wrapper
      .get("[data-test='sequence-stage-player'] video")
      .attributes("src");
    expect(firstSrc).toContain("/api/chain-jobs/job-playback/stages/0/media");

    chains.state.live.detail = {
      ...playbackDetail,
      stages: [
        playbackDetail.stages[0]!,
        {
          ...playbackDetail.stages[1]!,
          state: "completed",
          has_media: true,
          cache_ready: true,
        },
      ],
    };
    await flushPromises();
    expect(
      wrapper
        .get("[data-test='sequence-stage-player'] video")
        .attributes("src"),
    ).toBe(firstSrc);

    const completedDetail = chains.state.live.detail!;
    chains.state.live.detail = {
      ...completedDetail,
      stages: [
        {
          ...completedDetail.stages[0]!,
          state: "pending",
          has_media: false,
          cache_ready: false,
        },
        completedDetail.stages[1]!,
      ],
    };
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-stage-player']").exists()).toBe(
      false,
    );
  });

  it("restores cached filmstrip previews after Create unmounts and remounts", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "ltx-2-19b-distilled:fp8",
        family: "ltx2",
        downloaded: true,
        supports_sequence: true,
        default_width: 1216,
        default_height: 704,
        default_steps: 8,
        default_guidance: 3,
      },
    ]);
    const draft = enterSequenceMode();
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    useGenerateForm().state.value.modelFamily = "ltx2";
    draft.clips[0]!.prompt = "edited opening";
    draft.clips[1]!.prompt = "edited ending";
    draft.loadFromJob(
      {
        jobId: "job-remount",
        hostId: ORIGIN_HOST_ID,
        baseline: draft.clips.map((clip) => ({ ...clip })),
        completedStages: 2,
      },
      draft.clips.map((clip) => ({ ...clip })),
      false,
    );
    const detail = {
      id: "job-remount",
      state: "completed",
      model: "ltx-2-19b-distilled:fp8",
      stage_count: 2,
      current_stage: 2,
      created_at_unix_ms: 1,
      updated_at_unix_ms: 2,
      error: null,
      stages: [
        {
          idx: 0,
          state: "completed",
          seed: "1",
          frames_emitted: 97,
          generation_time_ms: 1,
          has_preview: true,
          has_media: false,
          cache_ready: true,
          error: null,
        },
        {
          idx: 1,
          state: "completed",
          seed: "2",
          frames_emitted: 97,
          generation_time_ms: 1,
          has_preview: false,
          has_media: false,
          cache_ready: true,
          error: null,
        },
      ],
      script: {
        schema: "mold.chain.v1",
        chain: { model: "ltx-2-19b-distilled:fp8", fps: 24 },
        stages: [
          { prompt: "server opening", frames: 97 },
          { prompt: "server ending", frames: 97 },
        ],
      },
    } satisfies ChainJobDetail;
    const chains = useChainJobs();
    chains.watch(ORIGIN_HOST_ID, detail.id);
    chains.state.live.detail = detail;
    // Earlier wrappers in this file share the chain-job singleton. Let their
    // watchers settle before measuring the two mounts owned by this test.
    const settleFetch = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      blob: async () => new Blob(["settled"], { type: "image/jpeg" }),
    } as Response);
    await flushPromises();
    settleFetch.mockRestore();
    const fetchPreview = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      blob: async () => new Blob(["preview"], { type: "image/jpeg" }),
    } as Response);
    let urlIndex = 0;
    const createUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockImplementation(() => `blob:preview-${++urlIndex}`);
    const revokeUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});

    const first = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const firstMedia = first
      .getComponent({ name: "SequenceComposer" })
      .props("stageMediaByClipId") as Record<string, { posterUrl?: string }>;
    const firstPoster = firstMedia[draft.clips[0]!.id]?.posterUrl;
    expect(firstPoster).toMatch(/^blob:preview-/);
    first.unmount();
    expect(revokeUrl).toHaveBeenCalledWith(firstPoster);

    const second = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const secondMedia = second
      .getComponent({ name: "SequenceComposer" })
      .props("stageMediaByClipId") as Record<string, { posterUrl?: string }>;
    const secondPoster = secondMedia[draft.clips[0]!.id]?.posterUrl;
    const previewFetchCount = fetchPreview.mock.calls.filter(([url]) =>
      String(url).includes("/stages/0/preview"),
    ).length;
    fetchPreview.mockRestore();
    createUrl.mockRestore();
    revokeUrl.mockRestore();
    second.unmount();
    expect(secondPoster).toMatch(/^blob:preview-/);
    expect(secondPoster).not.toBe(firstPoster);
    expect(previewFetchCount).toBe(2);
  });

  it("keeps the Output control reachable on phones in sequence mode", async () => {
    vi.stubGlobal(
      "matchMedia",
      vi.fn(() => ({
        matches: true,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      })),
    );
    hostModelsMock.mockResolvedValue([installedSequenceModel()]);
    useGenerateForm().state.value.modelFamily = "ltx2";
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    enterSequenceMode();
    await nextTick();

    expect(wrapper.find("[data-test='phone-sequence-controls']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='sequence-composer-stub']").exists()).toBe(
      true,
    );
    vi.unstubAllGlobals();
    vi.stubGlobal("prompt", vi.fn());
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
    setActivePinia(createPinia());
    chainJobsTesting.reset();
    takeSequenceHandoff();
    generateFormTesting.resetForTest();
    resetNotifications();
    submitMock.mockClear();
    listChainJobsMock.mockResolvedValue({ jobs: [] });
    routeQuery.value = {};
    routerReplaceMock.mockClear();
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
      instanceId: null,
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

  it("does not show the obsolete origin-only sequence warning", async () => {
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
      false,
    );
  });

  it("routes Most capable to the strongest GPU", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    placementPreviewMock.mockImplementation(async (...args: unknown[]) => ({
      version: 1,
      authoritative: true,
      state_version: 1,
      plan_version: 1,
      outcome: "planned",
      candidate: {
        device_id: "cuda:0",
        execution_fingerprint: "test",
        predicted_start_after_ms: 0,
        predicted_completion_after_ms: (
          args[0] as { baseUrl: string }
        ).baseUrl.includes("studio")
          ? 50
          : 100,
        setup_ms: 0,
        setup_kind: "warm",
        estimate_confidence: "high",
      },
    }));
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
        '<div><div data-test="prompt-style-stub"/><slot name="mobile-controls"/><button data-test="composer-submit" @click="$emit(\'submit\')">go</button><button data-test="composer-expand" @click="$emit(\'expand\')">expand</button><button data-test="composer-undo" @click="$emit(\'undo-expand\')">undo</button></div>',
      // The page calls these through its template ref on submit / new-print;
      // a stub without them throws an unhandled TypeError mid-run.
      methods: { record: vi.fn(), focus: vi.fn() },
    },
    ResultCanvas: {
      name: "ResultCanvas",
      props: ["mode", "variations", "resultCaption"],
      template:
        '<div data-test="result-canvas" :data-count="(variations||[]).length" :data-caption="resultCaption"><button data-test="queue-variations" @click="$emit(\'queue\')">queue</button></div>',
    },
    CreateModelPicker: {
      name: "CreateModelPicker",
      props: ["models", "model", "browseTo", "emptyLabel"],
      template: "<div data-test='model-picker-stub' />",
    },
    ControlsAside: {
      name: "ControlsAside",
      props: ["output", "clipCount"],
      template: "<aside data-test='controls-stub' />",
    },
    AdvancedDrawer: { name: "AdvancedDrawer", template: "<div />" },
    ActivityStrip: {
      name: "ActivityStrip",
      props: ["jobs", "sequences"],
      emits: [
        "cancel",
        "dismiss",
        "open",
        "sequence-action",
        "clear-inactive",
        "cleanup-disk",
      ],
      template:
        "<div data-test='activity-stub' :data-sequences='(sequences||[]).length' />",
    },
    SequenceComposer: {
      name: "SequenceComposer",
      props: [
        "model",
        "family",
        "shared",
        "modelDefaultFrames",
        "target",
        "chainLevelDirty",
        "stageMediaByClipId",
        "playingClipId",
      ],
      emits: [
        "submit",
        "duplicate-as-new",
        "discard-edit",
        "expand-clip",
        "import-shared",
        "play-clip",
      ],
      template:
        "<div data-test='sequence-composer-stub'>" +
        "<button data-test='sequence-generate' @click=\"$emit('submit')\">go</button>" +
        "<button v-if='Object.keys(stageMediaByClipId || {}).length' data-test='stub-play-first' @click=\"$emit('play-clip', Object.keys(stageMediaByClipId)[0])\">play</button>" +
        "<button data-test='clip-expand' @click=\"$emit('expand-clip', 'clip-x', 'a stage prompt')\">expand</button>" +
        "</div>",
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
