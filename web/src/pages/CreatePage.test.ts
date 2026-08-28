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
import { useHostRouting } from "../composables/useHostRouting";
import { usePullResume } from "../composables/usePullResume";
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
import {
  setGenerationHandoff,
  takeGenerationHandoff,
} from "../composables/useGenerationHandoff";
import { ApiHttpError } from "../api";
import { addHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { autoTagTitle, reloadAutoTagTitle } from "../lib/fileUnder";
import { AUTO_TARGET_ID, CAPABLE_TARGET_ID } from "../lib/hostRouting";
import type {
  GalleryImage,
  GenerateFormState,
  GenerateRequestWire,
  ModelInfoExtended,
  OutputMetadata,
} from "../types";
import type { Job } from "../composables/useGenerateStream";
import type {
  ChainJobDetail,
  ChainRequestWire,
  CreateChainJobResponse,
} from "@studio/lib/api/chainTypes";
import type { StreamTarget } from "../api";

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
  vi.fn<
    (
      request: ChainRequestWire,
      target?: StreamTarget,
      operationId?: string,
    ) => Promise<CreateChainJobResponse>
  >(async () => ({ job_id: "job-1" })),
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
const streamCanvasErrorJobIdRef = vi.hoisted(() => ({
  value: null as string | null,
}));
const streamSelectedJobRef = vi.hoisted(() => ({
  value: null as Job | null,
}));
const streamSelectMock = vi.hoisted(() => vi.fn());
const placementPreviewMock = vi.hoisted(() =>
  vi.fn(async (..._args: unknown[]): Promise<Record<string, unknown>> => ({
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
const promptHistoryApiMock = vi.hoisted(() =>
  vi.fn<(...args: unknown[]) => Promise<unknown>>(async () => ({
    entries: [],
  })),
);

vi.mock("@studio/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/client")>()),
  apiJsonTo: promptHistoryApiMock,
}));

const listCollectionsMock = vi.hoisted(() =>
  vi.fn(async () => [] as unknown[]),
);
const listTagsMock = vi.hoisted(() => vi.fn(async () => [] as unknown[]));

vi.mock("@studio/api/galleryOrganization", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/galleryOrganization")>()),
  listCollections: listCollectionsMock,
  listTags: listTagsMock,
}));

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
const cancelChainJobMutationMock = vi.hoisted(() =>
  vi.fn(async () => undefined),
);
const resumeChainJobMock = vi.hoisted(() => vi.fn(async () => ({})));
const deleteChainJobMock = vi.hoisted(() => vi.fn(async () => undefined));
const gcChainJobsMock = vi.hoisted(() =>
  vi.fn(async () => ({ swept_ephemeral_jobs: 0, pruned_artifact_dirs: 0 })),
);
const amendChainJobMock = vi.hoisted(() => vi.fn());
const cancelPrintMock = vi.hoisted(() => vi.fn(async () => undefined));
const postDownloadMock = vi.hoisted(() => vi.fn(async () => undefined));
const postCatalogDownloadMock = vi.hoisted(() => vi.fn(async () => ({})));

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    // Real error class so `instanceof ApiHttpError` branches stay honest.
    ApiHttpError: actual.ApiHttpError,
    looksLikeCatalogId: actual.looksLikeCatalogId,
    postDownload: postDownloadMock,
    postCatalogDownload: postCatalogDownloadMock,
    fetchDownloads: vi.fn(async () => ({
      active: null,
      active_jobs: [],
      queued: [],
      history: [],
    })),
    downloadsStreamUrl: () => "/api/downloads/events",
    cancelDownload: vi.fn(async () => undefined),
    createChainJob: createChainJobMock,
    expandPrompt: expandPromptMock,
    fetchModels: vi.fn(async () => []),
    fetchQueue: vi.fn(async () => ({ entries: [] })),
    listGallery: vi.fn(async () => [entry]),
    deleteGalleryImage: vi.fn(async () => undefined),
    upscaleStream: upscaleStreamMock,
    imageUrl: (name: string) => `/api/gallery/image/${name}`,
    thumbnailUrl: (name: string) => `/api/gallery/thumbnail/${name}`,
    fetchChainLimits: fetchChainLimitsMock,
    listChainJobs: listChainJobsMock,
    getChainJob: getChainJobMock,
    cancelChainJob: cancelChainJobMock,
    cancelChainJobMutation: cancelChainJobMutationMock,
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
    canvasErrorJobId: streamCanvasErrorJobIdRef,
    selectedJob: streamSelectedJobRef,
    submit: submitMock,
    submitBatch: (
      requests: GenerateRequestWire[],
      decision: unknown,
      route: unknown,
    ) => requests.map((request) => submitMock(request, decision, route)),
    cancel: cancelPrintMock,
    failRunning: vi.fn(),
    remove: vi.fn(),
    clearDone: vi.fn(),
    select: streamSelectMock,
  }),
}));

/** The default machine's inventory for a test that actually queues a print:
 *  routing is inventory-driven now, so the machine must hold the checkpoint. */
function installedModelRow(name: string, family: string) {
  return {
    name,
    family,
    size_gb: 12,
    is_loaded: false,
    last_used: null,
    hf_repo: "",
    downloaded: true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 3.5,
  };
}

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

const hostCapabilitiesMock = vi.hoisted(() =>
  vi.fn(
    async (_host: { id: string }): Promise<Record<string, unknown>> => ({}),
  ),
);
const hostModelDownloadMock = vi.hoisted(() => vi.fn(async () => null));

vi.mock("../components/machines/hostClient", () => ({
  hostStatus: hostStatusMock,
  hostModels: hostModelsMock,
  hostDownloads: vi.fn(async () => ({
    active: null,
    active_jobs: [],
    queued: [],
    history: [],
  })),
  hostModelDownload: hostModelDownloadMock,
  hostCapabilities: hostCapabilitiesMock,
  // The File under group reads each host's organization snapshot through
  // the same helpers the Library uses.
  hostApiTarget: (host: { url: string; apiKey?: string | null }) => ({
    baseUrl: host.url,
    apiKey: host.apiKey ?? null,
  }),
  hostGallery: vi.fn(async () => []),
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
    takeGenerationHandoff();
    usePullResume().cancel();
    generateFormTesting.resetForTest();
    resetNotifications();
    submitMock.mockClear();
    promptHistoryApiMock.mockReset();
    promptHistoryApiMock.mockResolvedValue({ entries: [] });
    placementPreviewMock.mockReset();
    placementPreviewMock.mockResolvedValue({
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
    });
    streamJobsRef.value = [];
    streamCanvasErrorJobIdRef.value = null;
    streamSelectedJobRef.value = null;
    streamSelectMock.mockReset();
    cancelPrintMock.mockReset();
    cancelPrintMock.mockResolvedValue(undefined);
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
    cancelChainJobMutationMock.mockClear();
    resumeChainJobMock.mockClear();
    deleteChainJobMock.mockClear();
    gcChainJobsMock.mockClear();
    amendChainJobMock.mockReset();
    routeQuery.value = {};
    routerReplaceMock.mockClear();
    hostStatusMock.mockClear();
    hostModelsMock.mockClear();
    hostModelsMock.mockResolvedValue([]);
    hostCapabilitiesMock.mockClear();
    hostCapabilitiesMock.mockResolvedValue({
      queue: { heterogeneous_batch_max_outputs: 64 },
    });
    hostModelDownloadMock.mockClear();
    listCollectionsMock.mockClear();
    listCollectionsMock.mockResolvedValue([]);
    listTagsMock.mockClear();
    listTagsMock.mockResolvedValue([]);
    reloadAutoTagTitle();
    vi.stubGlobal("prompt", vi.fn());
  });

  it("uses the Mold Studio composer + controls-region workspace", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(wrapper.get("[data-test='generate-shell']").classes()).toContain(
      "max-w-[1600px]",
    );
    expect(wrapper.get("[data-test='generate-workspace']").classes()).toContain(
      "md:grid-cols-[minmax(0,1fr)_340px]",
    );
  });

  it("applies settings selected from recovered Now developing work", async () => {
    enterSequenceMode();
    promptHistoryApiMock.mockImplementation(async (...args: unknown[]) =>
      args[1] === "/api/queue/remote-print/preview"
        ? {
            preview_image: "UFJFVklFVw==",
            step: 8,
            total: 20,
            stage: "Denoising",
            updated_at_ms: 1,
          }
        : { entries: [] },
    );
    setGenerationHandoff({
      seedPinned: true,
      queueSelection: {
        hostId: ORIGIN_HOST_ID,
        jobId: "remote-print",
        running: true,
      },
      metadata: {
        version: "1",
        model: "flux-dev",
        prompt: "recovered lighthouse",
        seed: 42,
        steps: 20,
        guidance: 3.5,
        width: 1024,
        height: 1024,
      } as OutputMetadata,
    });

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    expect(useSequenceDraftStore().output).toBe("single");
    expect(useGenerateForm().state.value).toMatchObject({
      prompt: "recovered lighthouse",
      seedMode: "static",
      seed: 42,
      steps: 20,
      guidance: 3.5,
    });
    expect(promptHistoryApiMock).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: window.location.origin }),
      "/api/queue/remote-print/preview",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    await flushPromises();
    await wrapper.vm.$nextTick();
    const previewSrc = wrapper
      .getComponent({ name: "ResultCanvas" })
      .props("previewSrc");
    const stage = wrapper.getComponent({ name: "ResultCanvas" }).props("stage");
    wrapper.unmount();
    expect(stage).toBe("Developing 8 / 20");
    expect(previewSrc).toBe("data:image/png;base64,UFJFVklFVw==");
    expect(takeGenerationHandoff()).toBeNull();
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

  it("renders a settled failure only while it has live canvas authority", async () => {
    streamJobsRef.value = [
      {
        id: "stale-error",
        request: {
          model: "cv:2937936",
          prompt: "old failed print",
          width: 512,
          height: 512,
          steps: 1,
          guidance: 1,
        },
        startedAt: 100,
        controller: new AbortController(),
        progress: {
          stage: "Loading model",
          step: null,
          totalSteps: null,
          queuePosition: null,
          gpu: null,
          elapsedMs: null,
        },
        result: null,
        error:
          "model load error: stable device placement requires a scheduler-bound GPU owner thread",
        state: "error",
        settledAt: 200,
        chain: null,
        lastProgressAt: 200,
        workStarted: true,
        hostId: null,
        hostLabel: null,
        target: null,
        serverId: "failed-server-job",
        previewUrl: null,
        seedVisual: "old",
      },
    ];
    // The successful card has already auto-removed from the rail. Explicit
    // live canvas authority is clear, so settled history cannot retake it.
    streamCanvasErrorJobIdRef.value = null;

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    // ColdStartGuide and ResultCanvas are mutually exclusive in production;
    // reaching the empty-state guide is the component-level proof that the
    // stale job did not select error mode.
    expect(wrapper.find("[data-test='cold-start-stub']").exists()).toBe(true);
    wrapper.unmount();

    const staleError = streamJobsRef.value[0];
    streamJobsRef.value = [
      {
        ...staleError,
        id: "newer-done",
        startedAt: 300,
        result: {
          type: "complete",
          image: "newer-image",
          format: "png",
          seed_used: 7,
          model: "cv:2937936",
          width: 512,
          height: 512,
          generation_time_ms: 1_000,
        },
        error: null,
        state: "done",
        settledAt: 400,
        lastProgressAt: 400,
        serverId: "successful-server-job",
      } as Job,
      staleError,
    ];
    streamCanvasErrorJobIdRef.value = "stale-error";
    const liveWrapper = mount(CreatePage, {
      global: { stubs: pageStubs() },
    });
    await flushPromises();

    const canvas = liveWrapper.getComponent({ name: "ResultCanvas" });
    expect(canvas.props("mode")).toBe("error");
    expect(liveWrapper.find("[data-test='cold-start-stub']").exists()).toBe(
      false,
    );

    liveWrapper
      .getComponent({ name: "ActivityStrip" })
      .vm.$emit("open", staleError);
    await nextTick();
    expect(streamSelectMock).toHaveBeenCalledWith("stale-error");
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
    await flushPromises();
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
    expect(wrapper.findComponent(RecentGridStub).props("limit")).toBe(18);
    wrapper.unmount();
    vi.unstubAllGlobals();
    vi.stubGlobal("prompt", vi.fn());
  });

  it("keeps the recent gallery visible after refreshes", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const feed = wrapper.findComponent(RecentGridStub);
    expect(feed.props("entries")).toEqual([entry]);
    expect(feed.props("limit")).toBe(50);
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

  it("offers the gallery actions when a Recent tile is right-clicked", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(["image"]),
    })) as never;
    const stubs: Record<string, Component> = pageStubs();
    stubs.RecentGrid = defineComponent({
      props: ["entries"],
      template:
        '<button data-test="context-recent" @contextmenu.prevent="$emit(\'context-menu\', { item: entries[0], x: 99999, y: 99999, trigger: $event.currentTarget })">context</button>',
    });
    const wrapper = mount(CreatePage, {
      attachTo: document.body,
      global: { stubs },
    });
    await flushPromises();

    await wrapper.get('[data-test="context-recent"]').trigger("contextmenu");
    const menu = wrapper.get('[data-test="recent-context-menu"]');
    const style = menu.attributes("style") ?? "";
    const left = Number(/left: (\d+(?:\.\d+)?)px/.exec(style)?.[1]);
    const top = Number(/top: (\d+(?:\.\d+)?)px/.exec(style)?.[1]);
    expect(left).toBeLessThan(window.innerWidth);
    expect(top).toBeLessThan(window.innerHeight);
    expect(menu.text()).toContain("Open");
    expect(menu.text()).toContain("Reuse settings");
    expect(menu.text()).toContain("Use as source");
    expect(menu.text()).toContain("Delete");
    expect((document.activeElement as HTMLElement).textContent?.trim()).toBe(
      "Open",
    );
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect((document.activeElement as HTMLElement).textContent?.trim()).toBe(
      "Reuse settings",
    );
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect((document.activeElement as HTMLElement).textContent?.trim()).toBe(
      "Use as source",
    );

    await wrapper.get('[data-test="recent-context-source"]').trigger("click");
    await flushPromises();
    expect(useGenerateForm().state.value.imageAttachments[0]?.filename).toBe(
      entry.filename,
    );
    expect(wrapper.find('[data-test="recent-context-menu"]').exists()).toBe(
      false,
    );

    await wrapper.get('[data-test="context-recent"]').trigger("contextmenu");
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await nextTick();
    expect(document.activeElement).toBe(
      wrapper.get('[data-test="context-recent"]').element,
    );
    wrapper.unmount();
    globalThis.fetch = originalFetch;
  });

  it("routes Recent videos into the source-video field", async () => {
    const video = { ...entry, filename: "clip.mp4", format: "mp4" as const };
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(["video"], { type: "video/mp4" }),
    })) as never;
    const stubs: Record<string, Component> = pageStubs();
    stubs.RecentGrid = defineComponent({
      props: ["entries"],
      setup: () => ({ video }),
      template:
        '<button data-test="context-recent-video" @contextmenu.prevent="$emit(\'context-menu\', { item: video, x: 20, y: 20 })">context</button>',
    });
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();

    await wrapper
      .get('[data-test="context-recent-video"]')
      .trigger("contextmenu");
    await wrapper.get('[data-test="recent-context-source"]').trigger("click");
    await flushPromises();
    expect(useGenerateForm().state.value.sourceVideo).toMatchObject({
      filename: "clip.mp4",
      mime: "video/mp4",
    });
    expect(useGenerateForm().state.value.imageAttachments).toHaveLength(0);
    wrapper.unmount();
    globalThis.fetch = originalFetch;
  });

  it("keeps animated Recent images in the image-source field", async () => {
    const animated = {
      ...entry,
      filename: "loop.gif",
      format: "gif" as const,
    };
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(["image"], { type: "image/gif" }),
    })) as never;
    const stubs: Record<string, Component> = pageStubs();
    stubs.RecentGrid = defineComponent({
      props: ["entries"],
      setup: () => ({ animated }),
      template:
        '<button data-test="context-recent-gif" @contextmenu.prevent="$emit(\'context-menu\', { item: animated, x: 20, y: 20 })">context</button>',
    });
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();

    await wrapper
      .get('[data-test="context-recent-gif"]')
      .trigger("contextmenu");
    await wrapper.get('[data-test="recent-context-source"]').trigger("click");
    await flushPromises();
    expect(useGenerateForm().state.value.imageAttachments[0]).toMatchObject({
      filename: "loop.gif",
    });
    expect(useGenerateForm().state.value.sourceVideo).toBeNull();
    wrapper.unmount();
    globalThis.fetch = originalFetch;
  });

  it("uses a Recent still as the Sequence opening image", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(["image"], { type: "image/png" }),
    })) as never;
    const stubs: Record<string, Component> = pageStubs();
    stubs.RecentGrid = defineComponent({
      props: ["entries"],
      template:
        '<button data-test="context-recent-sequence" @contextmenu.prevent="$emit(\'context-menu\', { item: entries[0], x: 20, y: 20 })">context</button>',
    });
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();
    const draft = enterSequenceMode();

    await wrapper
      .get('[data-test="context-recent-sequence"]')
      .trigger("contextmenu");
    await wrapper.get('[data-test="recent-context-source"]').trigger("click");
    await flushPromises();
    expect(draft.openingImage).toMatchObject({ filename: entry.filename });
    expect(useGenerateForm().state.value.imageAttachments).toHaveLength(0);
    wrapper.unmount();
    globalThis.fetch = originalFetch;
  });

  it("disables Recent video sources while Sequence is active", async () => {
    const video = { ...entry, filename: "clip.mp4", format: "mp4" as const };
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as never;
    const stubs: Record<string, Component> = pageStubs();
    stubs.RecentGrid = defineComponent({
      props: ["entries"],
      setup: () => ({ video }),
      template:
        '<button data-test="context-recent-sequence-video" @contextmenu.prevent="$emit(\'context-menu\', { item: video, x: 20, y: 20 })">context</button>',
    });
    const wrapper = mount(CreatePage, {
      attachTo: document.body,
      global: { stubs },
    });
    await flushPromises();
    enterSequenceMode();

    await wrapper
      .get('[data-test="context-recent-sequence-video"]')
      .trigger("contextmenu");
    const action = wrapper.get('[data-test="recent-context-source"]');
    expect(action.attributes("disabled")).toBeDefined();
    expect(action.attributes("title")).toContain("must be an image");
    expect((document.activeElement as HTMLElement).textContent?.trim()).toBe(
      "Open",
    );
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect((document.activeElement as HTMLElement).textContent?.trim()).toBe(
      "Reuse settings",
    );
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect((document.activeElement as HTMLElement).textContent?.trim()).toBe(
      "Delete",
    );
    await action.trigger("click");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(useGenerateForm().state.value.sourceVideo).toBeNull();
    wrapper.unmount();
    globalThis.fetch = originalFetch;
  });

  it("routes a Recent still into the active MiniMax H3 first frame", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(["image"], { type: "image/png" }),
    })) as never;
    const stubs: Record<string, Component> = pageStubs();
    stubs.RecentGrid = defineComponent({
      props: ["entries"],
      template:
        '<button data-test="context-recent-h3" @contextmenu.prevent="$emit(\'context-menu\', { item: entries[0], x: 20, y: 20 })">context</button>',
    });
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();
    const form = useGenerateForm().state.value;
    form.model = "minimax-h3-fl2va:comfy-pruned-int8";
    form.modelFamily = "minimax-h3";
    form.h3Authoring = {
      firstFrame: null,
      lastFrame: null,
      references: [],
    };

    await wrapper.get('[data-test="context-recent-h3"]').trigger("contextmenu");
    await wrapper.get('[data-test="recent-context-source"]').trigger("click");
    await flushPromises();
    expect(form.h3Authoring.firstFrame).toMatchObject({
      filename: entry.filename,
      mimeType: "image/png",
      width: entry.metadata.width,
      height: entry.metadata.height,
    });
    expect(form.imageAttachments).toHaveLength(0);
    wrapper.unmount();
    globalThis.fetch = originalFetch;
  });

  it("restores a recent normal print into One shot while Sequence is active", async () => {
    const stubs: Record<string, Component> = pageStubs();
    stubs.RecentGrid = defineComponent({
      props: ["entries"],
      template:
        '<button data-test="open-recent" @click="$emit(\'open\', entries[0])">open</button>',
    });
    stubs.Lightbox = defineComponent({
      props: ["item"],
      template:
        '<button v-if="item" data-test="lightbox-reuse" @click="$emit(\'reuse\', item)">reuse</button>',
    });
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();
    const draft = enterSequenceMode();
    await flushPromises();

    await wrapper.get('[data-test="open-recent"]').trigger("click");
    await wrapper.get('[data-test="lightbox-reuse"]').trigger("click");
    await flushPromises();

    expect(draft.output).toBe("single");
    expect(useGenerateForm().state.value.model).toBe(entry.metadata.model);
    expect(useGenerateForm().state.value.prompt).toBe(entry.metadata.prompt);
    expect(useGenerateForm().state.value.seed).toBe(entry.metadata.seed);
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
    form.state.value.batchSize = 4;
    await wrapper.get("[data-test='controls-reset']").trigger("click");
    expect(form.state.value.steps).toBe(30);
    expect(form.state.value.guidance).toBe(7.5);
    expect(form.state.value.seedMode).toBe("random");
    expect(form.state.value.negativePrompt).toBe("");
    expect(form.state.value.prompt).toBe("a lighthouse in a storm");
    expect(form.state.value.model).toBe("sdxl:fp16");
    expect(form.state.value.batchSize).toBe(1);

    const notifications = useNotifications();
    const settingsToast = notifications.toasts.find((t) =>
      /settings/i.test(t.text),
    );
    expect(settingsToast?.actionLabel).toBe("Undo");
    runToastAction(settingsToast!.id);
    expect(form.state.value.steps).toBe(3);
    expect(form.state.value.seed).toBe(42);
    expect(form.state.value.negativePrompt).toBe("blurry");
    expect(form.state.value.batchSize).toBe(4);

    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.enableAudio = true;
    await flushPromises();
    await wrapper.get("[data-test='controls-reset']").trigger("click");
    expect(draft.enableAudio).toBe(false);
    const sequenceToast = [...notifications.toasts]
      .reverse()
      .find((t) => /settings/i.test(t.text));
    runToastAction(sequenceToast!.id);
    expect(draft.enableAudio).toBe(true);
  });

  it("returns the canvas authority to the model on reset, and undo restores it (#1166)", async () => {
    const stubs: Record<string, Component> = pageStubs();
    stubs.ControlsAside = defineComponent({
      name: "ControlsAside",
      props: { canvasIntent: { type: String, default: "" } },
      template:
        "<aside data-test='controls-stub' :data-intent='canvasIntent'>" +
        "<button data-test='controls-reset' @click=\"$emit('reset-settings')\">reset</button>" +
        "<button data-test='controls-source' @click=\"$emit('canvas-intent', 'source')\">source</button>" +
        "</aside>",
    });
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();

    await wrapper.get("[data-test='controls-source']").trigger("click");
    expect(
      wrapper.get("[data-test='controls-stub']").attributes("data-intent"),
    ).toBe("source");

    await wrapper.get("[data-test='controls-reset']").trigger("click");
    expect(
      wrapper.get("[data-test='controls-stub']").attributes("data-intent"),
    ).toBe("model-default");

    const notifications = useNotifications();
    const settingsToast = notifications.toasts.find((t) =>
      /settings/i.test(t.text),
    );
    runToastAction(settingsToast!.id);
    await flushPromises();
    expect(
      wrapper.get("[data-test='controls-stub']").attributes("data-intent"),
    ).toBe("source");
  });

  it("guides a first pull when no models are installed (cold start)", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(wrapper.find("[data-test='cold-start-stub']").exists()).toBe(true);
  });

  it("shows the H3 first-frame blocker before prompt entry", async () => {
    const h3Model: ModelInfoExtended = {
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
      display_name: "MiniMax H3 FL2VA",
      size_gb: 42.5,
      is_loaded: false,
      last_used: null,
      hf_repo: "Comfy-Org/MiniMax-H3",
      downloaded: true,
      default_steps: 21,
      default_guidance: 0,
      default_width: 1344,
      default_height: 768,
      default_frames: 124,
      default_fps: 24,
      description: "Reviewed first-frame H3 runtime",
      source_image: "required",
    };
    hostModelsMock.mockResolvedValue([h3Model]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = h3Model.name;
    form.state.value.modelFamily = h3Model.family;
    form.state.value.prompt = "";
    form.state.value.sourceImageCapability = "required";
    form.state.value.h3Authoring = {
      firstFrame: null,
      lastFrame: null,
      references: [],
    };
    await nextTick();

    expect(
      wrapper.get("[data-test='page-generation-blocker']").text(),
    ).toContain("requires a first frame");

    form.state.value.h3Authoring.firstFrame = {
      filename: "opening.png",
      mimeType: "image/png",
      width: 1344,
      height: 768,
      data: "FIRST",
    };
    await nextTick();
    expect(wrapper.get("[data-test='page-generation-blocker']").text()).toBe(
      "Add a prompt before generating.",
    );
  });

  it("acknowledges Generate immediately and swallows a double click", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a cat";
    await nextTick();

    const submit = wrapper.get("[data-test='composer-submit']");
    // Two rapid clicks with no microtask flush between them — the second
    // must be swallowed by the in-flight guard, never double-queued.
    void submit.trigger("click");
    void submit.trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(1);
  });

  it("submits an off-profile custom size — the server is the authority", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a cat";
    // Off any recipe grid and below common minimums — advisory only.
    form.state.value.width = 320;
    form.state.value.height = 320;
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalled();
  });

  it("carries the print title field into the generate request and validates it inline", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a cat";
    await nextTick();

    const title = wrapper.get("[data-test='print-title']");
    expect(title.attributes("placeholder")).toBe("Untitled print");
    await title.setValue("  Smurf 04  ");
    expect(form.state.value.title).toBe("  Smurf 04  ");
    expect(wrapper.find("[data-test='print-title-error']").exists()).toBe(
      false,
    );

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    expect(submitMock.mock.calls[0]?.[0]).toMatchObject({ title: "Smurf 04" });

    await title.setValue("x".repeat(121));
    expect(wrapper.get("[data-test='print-title-error']").text()).toContain(
      "120",
    );
    submitMock.mockClear();
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    // An invalid title BLOCKS the submit — it is never silently dropped
    // from the wire while Generate fires anyway (codex review).
    expect(submitMock).not.toHaveBeenCalled();

    // Fixing the title unblocks Generate and the value rides the request.
    await title.setValue("Valid again");
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    expect(submitMock.mock.calls[0]?.[0]).toMatchObject({
      title: "Valid again",
    });
  });

  it("blocks a submit for an invalid title that never went through the field", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a cat";
    // A restored draft can carry an invalid title without an input event.
    form.state.value.title = "bad\u0007title";
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='print-title-error']").text()).toContain(
      "control characters",
    );
  });

  it("still blocks a malformed non-integer size", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a cat";
    form.state.value.width = 1024.5;
    form.state.value.height = 576;
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("whole numbers");
  });

  it("blocks non-Qwen mask submissions until a source image is selected", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK",
    };
    await nextTick();
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("Mask image needs a source image.");
  });

  it("blocks submission on an STG block list it cannot parse", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("ltx-2-19b-distilled:fp8", "ltx2"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.guidanceOverrides = {
      stgScale: 1.5,
      stgBlocks: "28,twenty-nine",
      rescaleScale: null,
      modalityScale: null,
      skipStep: null,
    };
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("STG blocks:");
  });

  it("blocks submission on a skip stride the wire cannot carry", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("ltx-2-19b-distilled:fp8", "ltx2"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.guidanceOverrides = {
      stgScale: null,
      stgBlocks: "",
      rescaleScale: null,
      modalityScale: null,
      skipStep: 1.5,
    };
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("Guidance skip stride:");
  });

  // #772: the advertised contract gates submit, and #779's first/last-frame
  // pair rides the existing keyframes field.
  it("holds Generate until an image-to-video checkpoint has its first frame", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("wan22-i2v-a14b:q8", "wan"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "wan22-i2v-a14b:q8";
    form.state.value.modelFamily = "wan";
    form.state.value.sourceImageCapability = "required";
    form.state.value.frames = 81;
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain(
      "This checkpoint is image-to-video only. Attach a source image to use as the first frame.",
    );

    form.state.value.imageAttachments = [
      { kind: "upload", filename: "open.png", base64: "FIRST" },
    ];
    await nextTick();
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    expect(submitMock).toHaveBeenCalled();
  });

  // #783: a continuation supplies its own first frames from the tail of the
  // clip it continues, which is why admission counts it as carrying source
  // (`mold_core::validation::request_carries_source_frames`). Without the same
  // reading here, the Continue-a-video control this branch made visible for a
  // Wan I2V checkpoint offered work submit refused.
  it("lets a Wan I2V continuation through with no attached image", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("wan22-i2v-a14b:q8", "wan"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "wan22-i2v-a14b:q8";
    form.state.value.modelFamily = "wan";
    form.state.value.sourceImageCapability = "required";
    form.state.value.frames = 49;
    form.state.value.extendVideo = {
      kind: "upload",
      filename: "clip.mp4",
      base64: "Q0xJUA==",
    };
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(wrapper.text()).not.toContain("This checkpoint is image-to-video");
    expect(submitMock).toHaveBeenCalled();
  });

  it("holds Generate for a Wan continuation on a text-to-video checkpoint", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("wan22-t2v-a14b:q8", "wan"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "wan22-t2v-a14b:q8";
    form.state.value.modelFamily = "wan";
    form.state.value.sourceImageCapability = "unsupported";
    form.state.value.frames = 49;
    form.state.value.extendVideoPath = "/srv/mold/clip.mp4";
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("text-to-video only and cannot continue");
  });

  it("holds Generate when a text-to-video checkpoint carries a source image", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("wan22-t2v-a14b:q5", "wan"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "wan22-t2v-a14b:q5";
    form.state.value.modelFamily = "wan";
    form.state.value.sourceImageCapability = "unsupported";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "stale.png", base64: "STALE" },
    ];
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("text-to-video only");
  });

  it("holds Generate for an end frame with no first frame", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("wan22-ti2v-5b:fp16", "wan"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "wan22-ti2v-5b:fp16";
    form.state.value.modelFamily = "wan";
    form.state.value.sourceImageCapability = "optional";
    form.state.value.frames = 81;
    form.state.value.endFrame = {
      kind: "upload",
      filename: "close.png",
      base64: "LAST",
    };
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("An end frame needs a first frame.");
  });

  it("submits a first/last-frame pair as two keyframes plus the source image", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("wan22-ti2v-5b:fp16", "wan"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "wan22-ti2v-5b:fp16";
    form.state.value.modelFamily = "wan";
    form.state.value.sourceImageCapability = "optional";
    form.state.value.frames = 81;
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "open.png", base64: "FIRST" },
    ];
    form.state.value.endFrame = {
      kind: "upload",
      filename: "close.png",
      base64: "LAST",
    };
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalled();
    const request = submitMock.mock.calls.at(-1)![0] as {
      source_image?: string | null;
      keyframes?: { frame: number; image: string }[];
    };
    expect(request.source_image).toBe("FIRST");
    expect(request.keyframes).toEqual([
      { frame: 0, image: "FIRST", name: "open.png" },
      { frame: 80, image: "LAST", name: "close.png" },
    ]);
  });

  it("says the end frame could not be restored when reusing a first/last-frame print", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "wan22-ti2v-5b:fp16",
        family: "wan",
        size_gb: 10,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
        default_steps: 20,
        default_guidance: 3.5,
        default_width: 1280,
        default_height: 704,
        default_frames: 81,
        default_fps: 24,
        description: "Wan 2.2 TI2V 5B",
        source_image: "optional",
      },
    ]);
    setGenerationHandoff({
      seedPinned: true,
      metadata: {
        version: "1",
        model: "wan22-ti2v-5b:fp16",
        prompt: "a heron takes off",
        seed: 42,
        steps: 20,
        guidance: 3.5,
        width: 1280,
        height: 704,
        frames: 81,
        keyframes: [
          { frame: 0, name: "open.png", sha256: "a".repeat(64) },
          { frame: 80, name: "close.png", sha256: "b".repeat(64) },
        ],
      } as OutputMetadata,
    });

    mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    expect(useGenerateForm().state.value.endFrame).toBeNull();
    // A first/last print carries its opening frame only in keyframes[0]
    // (no source provenance), so BOTH endpoints are named.
    const toastText = useNotifications()
      .toasts.map((item) => item.text)
      .join(" ");
    expect(toastText).toContain(
      "The end frame (close.png) and the first frame (open.png) can't be restored",
    );
  });

  it("fits only Qwen edit's target and preserves ordered references", async () => {
    hostModelsMock.mockResolvedValue([
      {
        name: "qwen-image-edit:q4",
        family: "qwen-image-edit",
        size_gb: 12,
        is_loaded: false,
        last_used: null,
        hf_repo: "",
        downloaded: true,
      },
    ]);
    class LoadedImage {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      naturalWidth = 2048;
      naturalHeight = 1024;
      set src(_value: string) {
        queueMicrotask(() => this.onload?.());
      }
    }
    vi.stubGlobal("Image", LoadedImage);
    const getContext = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue({
        fillStyle: "",
        fillRect: vi.fn(),
        drawImage: vi.fn(),
      } as unknown as CanvasRenderingContext2D);
    const toDataUrl = vi
      .spyOn(HTMLCanvasElement.prototype, "toDataURL")
      .mockReturnValue("data:image/png;base64,FITTED_TARGET");
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "qwen-image-edit:q4";
    form.state.value.modelFamily = "qwen-image-edit";
    form.state.value.imageAttachments = [
      {
        kind: "upload",
        filename: "target.png",
        base64: "TARGET",
        width: 2048,
        height: 1024,
      },
      { kind: "upload", filename: "reference.png", base64: "REFERENCE" },
    ];
    form.state.value.sourceFitPolicy = { mode: "crop-fill" };
    form.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK",
    };
    await nextTick();
    expect(form.state.value).toMatchObject({ width: 1664, height: 928 });

    try {
      await wrapper.get("[data-test='composer-submit']").trigger("click");
      await vi.waitFor(() => expect(submitMock).toHaveBeenCalledTimes(1), {
        timeout: 5_000,
      });

      const req = submitMock.mock.calls[0][0];
      expect(req.edit_images).toEqual(["FITTED_TARGET", "REFERENCE"]);
      expect(req.mask_image).toBeUndefined();
      expect(req.source_image).toBeUndefined();
      expect(getContext.mock.results[0]?.value?.drawImage).toHaveBeenCalledWith(
        expect.anything(),
        -72,
        0,
        1504,
        752,
      );
      expect(
        form.state.value.imageAttachments.map((image) => image.base64),
      ).toEqual(["TARGET", "REFERENCE"]);
    } finally {
      getContext.mockRestore();
      toDataUrl.mockRestore();
      vi.unstubAllGlobals();
    }
  });

  it("applies a pending Ref2VA reference crop at the original resolution before submitting", async () => {
    hostModelsMock.mockResolvedValue([
      {
        ...installedModelRow(
          "minimax-h3-ref2va:comfy-pruned-int8",
          "minimax-h3",
        ),
        default_width: 1344,
        default_height: 768,
      },
    ]);
    class LoadedImage {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      naturalWidth = 1024;
      naturalHeight = 768;
      set src(_value: string) {
        queueMicrotask(() => this.onload?.());
      }
    }
    vi.stubGlobal("Image", LoadedImage);
    const drawImage = vi.fn();
    const getContext = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue({
        fillStyle: "",
        fillRect: vi.fn(),
        drawImage,
      } as unknown as CanvasRenderingContext2D);
    const toDataUrl = vi
      .spyOn(HTMLCanvasElement.prototype, "toDataURL")
      .mockReturnValue("data:image/png;base64,Q1JPUFBFRA==");
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "minimax-h3-ref2va:comfy-pruned-int8";
    form.state.value.modelFamily = "minimax-h3";
    form.state.value.prompt = "a subject in a new shot";
    form.state.value.h3Authoring = {
      firstFrame: null,
      lastFrame: null,
      references: [
        {
          reference: {
            kind: "image",
            media: { authority: "inline", data: "SU1BR0U=" },
            provenance: { name: "subject.png", sha256: "a".repeat(64) },
            mime_type: "image/png",
            width: 1024,
            height: 768,
          },
          crop: { x: 256, y: 0, width: 512, height: 768 },
        },
      ],
    };
    await nextTick();

    try {
      await wrapper.get("[data-test='composer-submit']").trigger("click");
      await vi.waitFor(() => expect(submitMock).toHaveBeenCalledTimes(1), {
        timeout: 5_000,
      });
      const req = submitMock.mock.calls[0][0];
      expect(drawImage).toHaveBeenCalledWith(
        expect.anything(),
        -256,
        0,
        1024,
        768,
      );
      expect(req.references?.[0]).toMatchObject({
        kind: "image",
        media: { authority: "inline", data: "Q1JPUFBFRA==" },
        width: 512,
        height: 768,
        provenance: {
          name: "subject.png",
          crop: {
            x: 256,
            y: 0,
            width: 512,
            height: 768,
            source_width: 1024,
            source_height: 768,
            source_sha256: "a".repeat(64),
          },
        },
      });
      // The live form keeps the uncropped original and its pending crop.
      expect(form.state.value.h3Authoring.references[0]).toMatchObject({
        reference: { width: 1024, media: { data: "SU1BR0U=" } },
        crop: { x: 256, y: 0, width: 512, height: 768 },
      });
    } finally {
      getContext.mockRestore();
      toDataUrl.mockRestore();
      vi.unstubAllGlobals();
    }
  });

  it("keeps a long advanced video request single-shot and preserves its settings", async () => {
    const guidedModel = {
      ...installedSequenceModel(),
      name: "ltx-2-19b-dev:fp8",
    };
    hostModelsMock.mockResolvedValue([guidedModel]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = guidedModel.name;
    form.state.value.modelFamily = "ltx2";
    form.state.value.frames = 153;
    form.state.value.fps = 24;
    form.state.value.negativePrompt = "flicker";
    await nextTick();

    expect(
      wrapper.get("[data-test='single-shot-preservation-cue']").text(),
    ).toContain("one 153-frame clip to preserve negative prompt");

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(1);
    expect(submitMock.mock.calls[0]?.[0]).toMatchObject({
      frames: 153,
      negative_prompt: "flicker",
    });
    expect(submitMock.mock.calls[0]?.[1]).toEqual({
      kind: "single",
      preservedAutoChainFields: ["negative_prompt"],
    });
  });

  it("reuses source preprocessing without replacing the editable source", async () => {
    upscaleStreamMock.mockImplementation(async (_request, handlers) => {
      handlers.onComplete({ image: "UPSCALED" });
    });
    hostModelsMock.mockResolvedValue([
      installedModelRow("ltx-2-19b-distilled:fp8", "ltx2"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
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

    const submitButton = wrapper.get("[data-test='composer-submit']");
    await submitButton.trigger("click");
    await vi.waitFor(
      () => {
        expect(submitMock).toHaveBeenCalledTimes(1);
        expect(submitButton.attributes("disabled")).toBeUndefined();
      },
      { timeout: 5_000 },
    );
    await submitButton.trigger("click");
    await vi.waitFor(() => expect(submitMock).toHaveBeenCalledTimes(2), {
      timeout: 5_000,
    });

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
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
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

  it("renders the sequence opening image in the primary form, never in Advanced", async () => {
    hostModelsMock.mockResolvedValue([installedSequenceModel()]);
    enterSequenceMode();
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    expect(
      wrapper.find("[data-test='sequence-opening-image-panel']").exists(),
    ).toBe(true);
    // The one-shot well steps aside in Sequence, and Advanced no longer owns
    // an opening-image section at all.
    expect(wrapper.find("[data-test='source-media-panel']").exists()).toBe(
      false,
    );
    expect(
      wrapper.find("[data-test='sequence-section-opening-image']").exists(),
    ).toBe(false);
  });

  it("parks a retained opening image out of the request for an unsupported checkpoint", async () => {
    // The well is gone, so the user can neither see nor remove the image; the
    // request must not carry conditioning the server would refuse.
    hostModelsMock.mockResolvedValue([
      { ...installedSequenceModel(), source_image: "unsupported" },
    ]);
    useGenerateForm().state.value.modelFamily = "ltx2";
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    const draft = enterSequenceMode();
    draft.clips[0]!.prompt = "the opening";
    draft.clips[1]!.prompt = "the landing";
    draft.openingImage = { filename: "opening.png", base64: "U1RBTEU=" };
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(
      wrapper.find("[data-test='sequence-opening-image-panel']").exists(),
    ).toBe(false);

    await wrapper.get("[data-test='sequence-generate']").trigger("click");
    await flushPromises();

    expect(createChainJobMock).toHaveBeenCalledTimes(1);
    const request = createChainJobMock.mock.calls[0]?.[0] as unknown as {
      stages: Array<Record<string, unknown>>;
    };
    expect(request.stages).toHaveLength(2);
    for (const stage of request.stages) {
      expect(stage).not.toHaveProperty("source_image");
    }
    // The draft keeps the image for a checkpoint that can use it later.
    expect(draft.openingImage).toMatchObject({ filename: "opening.png" });
  });

  it("hides the opening image when the checkpoint advertises no source-image contract", async () => {
    hostModelsMock.mockResolvedValue([
      { ...installedSequenceModel(), source_image: "unsupported" },
    ]);
    enterSequenceMode();
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    expect(
      wrapper.find("[data-test='sequence-opening-image-panel']").exists(),
    ).toBe(false);
  });

  it("keeps the opening image out of the Advanced count and clears it on Reset", async () => {
    hostModelsMock.mockResolvedValue([installedSequenceModel()]);
    const stubs: Record<string, Component> = pageStubs();
    stubs.ControlsAside = defineComponent({
      name: "ControlsAside",
      template:
        "<aside data-test='controls-stub'><button data-test='controls-reset' @click=\"$emit('reset-settings')\">reset</button></aside>",
    });
    stubs.AdvancedDrawer = defineComponent({
      name: "AdvancedDrawer",
      props: { advCount: { type: Number, default: 0 } },
      template: "<div data-test='advanced-stub' :data-count='advCount' />",
    });
    const draft = enterSequenceMode();
    const wrapper = mount(CreatePage, { global: { stubs } });
    await flushPromises();

    draft.openingImage = {
      filename: "opening.png",
      base64: "QUJD",
      width: 1216,
      height: 704,
    };
    await flushPromises();
    // Advanced badges Advanced content; primary-form source media is not it.
    expect(
      wrapper.get("[data-test='advanced-stub']").attributes("data-count"),
    ).toBe("0");

    await wrapper.get("[data-test='controls-reset']").trigger("click");
    expect(draft.openingImage).toBeNull();

    const notifications = useNotifications();
    const toast = [...notifications.toasts]
      .reverse()
      .find((t) => /settings/i.test(t.text));
    runToastAction(toast!.id);
    expect(draft.openingImage).toMatchObject({ filename: "opening.png" });
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
    draft.clips[1]!.cameraControl = "dolly-in";
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

    expect(fetchChainLimitsMock).toHaveBeenCalledWith(
      "ltx-2-19b-distilled:fp8",
      expect.anything(),
      24,
    );
    expect(createChainJobMock).toHaveBeenCalledTimes(1);
    expect(createChainJobMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "ltx-2-19b-distilled:fp8",
        width: 1216,
        height: 704,
        steps: 4,
        // Distilled recipes materialize their fixed guidance at submission
        // while preserving the inspector's adjustable value for other models.
        guidance: 1,
        fps: 24,
        output_format: "mp4",
        stages: [
          expect.objectContaining({ prompt: "the opening" }),
          expect.objectContaining({
            prompt: "the landing",
            loras: [
              {
                path: "camera-control:dolly-in",
                scale: 1,
                name: "Dolly in",
              },
            ],
          }),
        ],
      }),
      expect.objectContaining({ baseUrl: expect.any(String) }),
      expect.stringMatching(/^[0-9a-f-]{36}$/),
      expect.any(Function),
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
      expect.any(Object),
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
    useGenerateForm().state.value.prompt = "parked one shot";
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
    expect(useGenerateForm().state.value.prompt).toBe("parked one shot");
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
  });

  it("prepares a batch on the server and queues provenance on every sibling", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    form.state.value.batchSize = 3;
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();
    expect(submitMock).not.toHaveBeenCalled();
    const expandCall = expandPromptMock.mock.calls[0] as unknown as unknown[];
    const [expandPayload, expandTarget] = expandCall;
    expect(expandPayload).toEqual({
      prompt: "a lighthouse",
      model_family: "flux",
      variations: 3,
      task: "text-to-image",
    });
    // The origin expands through relative dispatch: same machine, no target.
    expect(expandTarget).toBeUndefined();
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

  it("does not offer a model pull for an exact-count expansion failure", async () => {
    expandPromptMock.mockRejectedValueOnce(
      new Error(
        "expected exactly 3 distinct non-empty prompts, but the expansion backend returned 2. " +
          "The model may need re-downloading: mold pull qwen3-expand",
      ),
    );
    hostModelsMock.mockResolvedValue([installedModelRow("flux-dev:q4", "flux")]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    form.state.value.batchSize = 3;
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='web-expansion-pull']").exists()).toBe(false);
    expect(wrapper.getComponent({ name: "ResultCanvas" }).props("variations")).toEqual([]);
  });

  it("blocks ordinary submit while variations are preparing and awaiting review", async () => {
    let releaseExpansion!: (value: {
      original: string;
      expanded: string[];
    }) => void;
    expandPromptMock.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseExpansion = resolve;
        }),
    );
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    form.state.value.batchSize = 3;
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();
    const composer = wrapper.getComponent({ name: "ComposerCard" });
    expect(composer.props("busy")).toBe(true);
    composer.vm.$emit("submit");
    await flushPromises();
    expect(submitMock).not.toHaveBeenCalled();

    releaseExpansion({
      original: "a lighthouse",
      expanded: ["north light", "storm light", "harbor light"],
    });
    await flushPromises();
    expect(composer.props("busy")).toBe(true);
    composer.vm.$emit("submit");
    await flushPromises();
    expect(submitMock).not.toHaveBeenCalled();
  });

  it("opens prompt expansion while an earlier print is running", async () => {
    streamJobsRef.value = [
      {
        id: "already-running",
        request: {
          model: "flux-dev:q4",
          prompt: "earlier work",
          width: 512,
          height: 512,
          steps: 20,
          guidance: 3,
          batch_size: 1,
          output_format: "png",
        },
        startedAt: 0,
        controller: new AbortController(),
        progress: {
          stage: "Queued",
          step: null,
          totalSteps: null,
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
      } as Job,
    ];
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.prompt = "new work";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();
    expect(wrapper.getComponent({ name: "ExpandModal" }).props("open")).toBe(
      true,
    );
  });

  it("preserves reviewed variations as stale when the model changes", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
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
    hostModelsMock.mockResolvedValue([
      installedModelRow("sdxl-base:fp16", "sdxl"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
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
    hostModelsMock.mockResolvedValue([installedSequenceModel()]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
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
    expect(modal.props("task")).toBe("text-to-video");
    // The style row belongs to the single-print composer, not to clip text.
    expect(modal.props("styleDirective")).toBeNull();
  });

  it("resolves image-conditioned video expansion without sending source bytes", async () => {
    hostModelsMock.mockResolvedValue([installedModelRow("ltx2:q8", "ltx2")]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "ltx2:q8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.prompt = "she turns toward the window";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "opening.png", base64: "secret-image-bytes" },
    ];
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();

    const modal = wrapper.getComponent({ name: "ExpandModal" });
    expect(modal.props("task")).toBe("image-to-video");
    expect(JSON.stringify(modal.props())).not.toContain("secret-image-bytes");
  });

  it("bakes and clears the chip when a quick expansion is applied", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
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

  it("retires dormant original-prompt provenance when a new prompt is authored", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.originalPrompt =
      "the source for an earlier generated print";
    await nextTick();

    wrapper
      .getComponent({ name: "ComposerCard" })
      .vm.$emit("update:prompt", "a completely new print");
    await nextTick();

    expect(form.state.value.originalPrompt).toBeNull();
    expect(form.toRequest().original_prompt).toBeUndefined();
  });

  it("retires dormant provenance when a LoRA trigger phrase edits the prompt", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.prompt = "a portrait";
    form.state.value.originalPrompt = "an earlier generated print";
    await nextTick();

    wrapper
      .getComponent({ name: "AdvancedDrawer" })
      .vm.$emit("append-prompt", "cinematic light");
    await nextTick();

    expect(form.state.value.prompt).toBe("a portrait, cinematic light");
    expect(form.state.value.originalPrompt).toBeNull();
    expect(form.toRequest().original_prompt).toBeUndefined();
  });

  it("preserves the source while an active quick expansion becomes stale", async () => {
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    form.state.value.prompt = "a lighthouse";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await nextTick();
    wrapper
      .getComponent({ name: "ExpandModal" })
      .vm.$emit("apply-prompt", "a lighthouse in storm light");
    await nextTick();

    wrapper
      .getComponent({ name: "ComposerCard" })
      .vm.$emit("update:prompt", "a hand-edited storm lighthouse");
    await nextTick();

    expect(form.state.value.originalPrompt).toBe("a lighthouse");
    const stale = wrapper.get("[data-test='web-quick-expansion-stale']");
    expect(
      stale.find("[data-test='web-reexpand-current-prompt']").exists(),
    ).toBe(true);
  });

  it("undoes an original-source Remix to the latest live composer edit", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("sdxl-base:fp16", "sdxl"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    form.state.value.originalPrompt = "root lighthouse idea";
    form.state.value.prompt = "expanded lighthouse draft";
    await nextTick();

    wrapper.getComponent({ name: "ComposerCard" }).vm.$emit("remix");
    await flushPromises();
    form.state.value.prompt = "latest hand edit while Remix was open";
    wrapper.getComponent({ name: "RemixModal" }).vm.$emit("apply", {
      prompt: "subject-preserving remix",
      response: {
        source_prompt: "root lighthouse idea",
        root_prompt: "root lighthouse idea",
        source_kind: "original",
        variants: [
          { prompt: "subject-preserving remix", dimensions: ["camera"] },
          { prompt: "second", dimensions: ["lighting"] },
          { prompt: "third", dimensions: ["mood"] },
        ],
      },
    });
    await nextTick();
    expect(form.state.value.prompt).toBe("subject-preserving remix");

    await wrapper.get("[data-test='composer-undo']").trigger("click");
    await nextTick();
    expect(form.state.value.prompt).toBe(
      "latest hand edit while Remix was open",
    );
    expect(form.state.value.originalPrompt).toBe("root lighthouse idea");
  });

  it("bakes and clears an active style when a Remix is applied", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("sdxl-base:fp16", "sdxl"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    form.state.value.prompt = "a lighthouse";
    form.state.value.negativePrompt = "text";
    form.state.value.stylePreset = "cinematic";
    await nextTick();

    wrapper.getComponent({ name: "ComposerCard" }).vm.$emit("remix");
    await flushPromises();
    wrapper.getComponent({ name: "RemixModal" }).vm.$emit("apply", {
      prompt: "storm light over a cinematic coast",
      response: {
        source_prompt: "a lighthouse",
        source_kind: "direct",
        variants: [
          {
            prompt: "storm light over a cinematic coast",
            dimensions: ["camera"],
          },
          { prompt: "second", dimensions: ["lighting"] },
          { prompt: "third", dimensions: ["mood"] },
        ],
      },
    });
    await nextTick();

    expect(form.state.value.stylePreset).toBeNull();
    expect(form.state.value.negativePrompt).toBe(
      "text, anime, cartoon, graphic, washed out",
    );
    expect(form.toRequest()).toMatchObject({
      prompt: "storm light over a cinematic coast",
      negative_prompt: "text, anime, cartoon, graphic, washed out",
    });

    await wrapper.get("[data-test='composer-undo']").trigger("click");
    await nextTick();
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
    expect(submitMock.mock.calls[0]?.[2]).toMatchObject({
      hostId: studio.id,
      label: "Studio",
      instanceId: null,
      modelFamily: "flux",
      referenceUploads: null,
      target: { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
    });
  });

  it("reuses a quick transformed prompt on a newly selected host without stale recovery", async () => {
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
        description: "Flux Dev",
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
    useHostRouting().setTarget(ORIGIN_HOST_ID);
    await nextTick();

    expect(
      wrapper.find("[data-test='web-quick-expansion-stale']").exists(),
    ).toBe(false);
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(1);
    expect(submitMock.mock.calls[0]?.[0]).toMatchObject({
      prompt: "storm light over the harbor",
      original_prompt: "a lighthouse",
    });
    expect(submitMock.mock.calls[0]?.[2]).toMatchObject({
      hostId: ORIGIN_HOST_ID,
    });
  });

  it("applies a clip expansion to the clip, never the composer's prompt or style", async () => {
    hostModelsMock.mockResolvedValue([installedSequenceModel()]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
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
    hostModelsMock.mockResolvedValue([
      installedModelRow("sdxl-base:fp16", "sdxl"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
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
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
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
    // Variations cleared → nothing reviewed is left on the canvas.
    const canvas = wrapper.find("[data-test='result-canvas']");
    expect(canvas.exists() ? canvas.attributes("data-count") : "0").toBe("0");
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

  it("reports an unconfirmed print cancellation instead of claiming success", async () => {
    cancelPrintMock.mockRejectedValueOnce(new Error("job is already running"));
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    wrapper
      .getComponent({ name: "ActivityStrip" })
      .vm.$emit("cancel", "print-9");
    await flushPromises();

    expect(cancelPrintMock).toHaveBeenCalledWith("print-9");
    expect(useNotifications().toasts.map((item) => item.text)).toContain(
      "job is already running",
    );
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
        source_fit: { mode: "crop-fill", alignX: "right", alignY: "bottom" },
        enable_audio: true,
      },
      startedAt: 0,
      controller: new AbortController(),
      progress: {
        stage: "Queued",
        step: null,
        totalSteps: null,
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
    expect(form.state.value.sourceFitPolicy).toEqual({
      mode: "crop-fill",
      alignX: "right",
      alignY: "bottom",
    });
    expect(form.state.value.enableAudio).toBe(true);
  });

  it("restores a queued camera LoRA into both the picker and visible stack", async () => {
    const job = {
      id: "camera-print",
      request: {
        prompt: "a tracking shot",
        model: "ltx-2-19b-distilled:fp8",
        width: 768,
        height: 512,
        steps: 8,
        guidance: 3,
        loras: [{ path: "camera-control:dolly-in", scale: 0.45 }],
      },
      startedAt: 0,
      controller: new AbortController(),
      progress: {
        stage: "Queued",
        step: null,
        totalSteps: null,
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
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    wrapper.getComponent({ name: "ActivityStrip" }).vm.$emit("open", job);
    await nextTick();

    expect(form.state.value.cameraControl).toBe("dolly-in");
    expect(form.state.value.loras).toEqual([
      {
        path: "camera-control:dolly-in",
        scale: 0.45,
        trainedWords: [],
      },
    ]);
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
          {
            prompt: "one",
            frames: 97,
            source_image_b64: "iVBORw0KGgoAAAANSUhEUgAAAoAAAAGA",
            source_image_path: "opening.png",
          },
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
    expect(draft.openingImage).toEqual({
      filename: "opening image",
      base64: "iVBORw0KGgoAAAANSUhEUgAAAoAAAAGA",
      width: 640,
      height: 384,
    });
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
          expect.objectContaining({
            prompt: "one",
            source_image: "iVBORw0KGgoAAAANSUhEUgAAAoAAAAGA",
          }),
          expect.objectContaining({ prompt: "two, but stormier" }),
        ],
      }),
      expect.objectContaining({ baseUrl: expect.any(String) }),
      expect.stringMatching(/^[0-9a-f-]{36}$/),
    );
    expect(amendChainJobMock.mock.calls[0]?.[1]).toEqual(
      expect.objectContaining({ strength: 1 }),
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

  it("compensates a cancelled in-flight sequence amendment on its frozen target", async () => {
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
        baseline: draft.clips.map((clip) => ({ ...clip })),
        completedStages: 1,
      },
      draft.clips.map((clip) => ({ ...clip })),
      false,
    );
    await flushPromises();
    let finishAmend!: (value: { preserved_stages: number }) => void;
    amendChainJobMock.mockReturnValueOnce(
      new Promise((resolve) => (finishAmend = resolve)),
    );

    await wrapper.get("[data-test='sequence-generate']").trigger("click");
    await vi.waitFor(() => expect(amendChainJobMock).toHaveBeenCalledTimes(1));
    const frozenTarget = amendChainJobMock.mock.calls[0]?.[2];
    const operationId = amendChainJobMock.mock.calls[0]?.[3];
    expect(operationId).toMatch(/^[0-9a-f-]{36}$/);
    const cancel = wrapper.get("[data-test='sequence-generate']");
    expect(cancel.text()).toContain("Cancel");
    await cancel.trigger("click");
    await vi.waitFor(() =>
      expect(cancelChainJobMutationMock).toHaveBeenCalledWith(
        "job-9",
        operationId,
        frozenTarget,
      ),
    );
    finishAmend({ preserved_stages: 0 });
    await flushPromises();

    expect(cancelChainJobMock).toHaveBeenCalledWith("job-9", frozenTarget);
    expect(draft.editing).toMatchObject({ jobId: "job-9" });
  });

  it("cancels sequence creation before a lost job-id response", async () => {
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
    createChainJobMock.mockReturnValueOnce(new Promise(() => {}));
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const draft = enterSequenceMode();
    draft.clips[0]!.prompt = "one";
    draft.clips[1]!.prompt = "two";
    await flushPromises();

    await wrapper.get("[data-test='sequence-generate']").trigger("click");
    await vi.waitFor(() => expect(createChainJobMock).toHaveBeenCalledTimes(1));
    const target = createChainJobMock.mock.calls[0]?.[1];
    const operationId = createChainJobMock.mock.calls[0]?.[2];
    expect(operationId).toMatch(/^[0-9a-f-]{36}$/);
    await wrapper.get("[data-test='sequence-generate']").trigger("click");

    await vi.waitFor(() =>
      expect(cancelChainJobMutationMock).toHaveBeenCalledWith(
        operationId,
        operationId,
        target,
      ),
    );
    expect(cancelChainJobMock).not.toHaveBeenCalled();
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

  // ── File under (Create-time Library organization) ─────────────────────
  const filingCapabilities = {
    gallery: { organize: true },
    queue: { heterogeneous_batch_max_outputs: 64 },
  };

  function filingFleet() {
    hostCapabilitiesMock.mockResolvedValue(filingCapabilities);
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    listTagsMock.mockResolvedValue([
      { name: "blue", count: 9 },
      { name: "dusk", count: 2 },
    ]);
    listCollectionsMock.mockResolvedValue([
      {
        id: "c1",
        name: "Smurfs",
        slug: "smurfs",
        description: null,
        cover_filename: null,
        count: 12,
        created_at: 1,
        updated_at: 1,
      },
    ]);
  }

  async function mountFiling(title = "Smurfs") {
    filingFleet();
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a cat";
    form.state.value.title = title;
    await flushPromises();
    return wrapper;
  }

  it("hides File under entirely on a fleet that cannot organize", async () => {
    hostModelsMock.mockResolvedValue([
      installedModelRow("flux-dev:q4", "flux"),
    ]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    expect(wrapper.find("[data-test='file-under-group']").exists()).toBe(false);
  });

  it("renders File under inside the controls region once a host can file", async () => {
    const wrapper = await mountFiling();
    // Its home is the controls rail's slot — after the essentials, above the
    // inline Advanced column (spec §06 web note).
    expect(
      wrapper
        .get("[data-test='controls-stub']")
        .find("[data-test='file-under-group']")
        .exists(),
    ).toBe(true);
    expect(wrapper.get("[data-test='file-under-ghost']").text()).toContain(
      "smurfs",
    );
  });

  it("files a one-shot print under the ghost tag and the matched collection", async () => {
    const wrapper = await mountFiling();
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    expect(submitMock.mock.calls[0]?.[0]).toMatchObject({
      title: "Smurfs",
      tags: ["smurfs"],
      collection: { name: "Smurfs" },
    });
  });

  it("carries a typed tag added in the group", async () => {
    const wrapper = await mountFiling();
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.setValue("#blue");
    await input.trigger("keydown", { key: "Enter" });
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    expect(submitMock.mock.calls[0]?.[0].tags).toEqual(["smurfs", "blue"]);
  });

  it("files nothing once the ghost chip and the collection are cleared", async () => {
    const wrapper = await mountFiling();
    await wrapper.get("[data-test='file-under-ghost-remove']").trigger("click");
    await wrapper
      .get("[data-test='file-under-collection-clear']")
      .trigger("click");
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    const request = submitMock.mock.calls[0]?.[0];
    expect(request.tags).toBeUndefined();
    expect(request.collection).toBeUndefined();
  });

  it("gives every batch sibling the same filing", async () => {
    const wrapper = await mountFiling();
    useGenerateForm().state.value.batchSize = 3;
    await nextTick();
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    expect(submitMock).toHaveBeenCalledTimes(3);
    for (const call of submitMock.mock.calls) {
      expect(call[0]).toMatchObject({
        tags: ["smurfs"],
        collection: { name: "Smurfs" },
      });
    }
  });

  it("files every prepared variation the same way", async () => {
    const wrapper = await mountFiling();
    useGenerateForm().state.value.batchSize = 3;
    useGenerateForm().state.value.prompt = "a lighthouse";
    await nextTick();
    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='queue-variations']").trigger("click");
    await flushPromises();
    expect(submitMock).toHaveBeenCalledTimes(3);
    for (const call of submitMock.mock.calls) {
      expect(call[0]).toMatchObject({
        tags: ["smurfs"],
        collection: { name: "Smurfs" },
      });
    }
  });

  it("files the stitched print of a sequence, never its clips", async () => {
    filingFleet();
    hostModelsMock.mockResolvedValue([installedSequenceModel()]);
    useGenerateForm().state.value.modelFamily = "ltx2";
    useGenerateForm().state.value.model = "ltx-2-19b-distilled:fp8";
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const draft = enterSequenceMode();
    draft.clips[0]!.prompt = "the opening";
    draft.clips[1]!.prompt = "the landing";
    useGenerateForm().state.value.title = "Smurfs";
    await flushPromises();

    await wrapper.get("[data-test='sequence-generate']").trigger("click");
    await flushPromises();

    expect(createChainJobMock).toHaveBeenCalledTimes(1);
    const calls = createChainJobMock.mock.calls as unknown as Record<
      string,
      unknown
    >[][];
    const body = calls[0]![0]!;
    expect(body).toMatchObject({
      title: "Smurfs",
      tags: ["smurfs"],
      collection: { name: "Smurfs" },
    });
    for (const stage of body.stages as Record<string, unknown>[]) {
      expect(stage.tags).toBeUndefined();
      expect(stage.collection).toBeUndefined();
    }
  });

  it("drops the ghost tag when the title auto-tag preference is off", async () => {
    const wrapper = await mountFiling();
    autoTagTitle.value = false;
    await nextTick();
    expect(wrapper.find("[data-test='file-under-ghost']").exists()).toBe(false);
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    const request = submitMock.mock.calls[0]?.[0];
    expect(request.tags).toBeUndefined();
    expect(request.collection).toEqual({ name: "Smurfs" });
  });

  it("restores a print's filing with Reuse settings", async () => {
    filingFleet();
    setGenerationHandoff({
      seedPinned: true,
      metadata: {
        version: "1",
        model: "flux-dev",
        prompt: "recovered lighthouse",
        title: "Smurfs",
        tags: ["smurfs", "blue"],
        collection: "River studies",
        seed: 42,
        steps: 20,
        guidance: 3.5,
        width: 1024,
        height: 1024,
      } as OutputMetadata,
    });
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const chips = wrapper
      .findAll("[data-test='file-under-tag'], [data-test='file-under-ghost']")
      .map((chip) => chip.text());
    expect(chips.join(" ")).toContain("blue");
    expect(wrapper.get("[data-test='file-under-collection']").text()).toContain(
      "River studies",
    );
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
    promptHistoryApiMock.mockReset();
    promptHistoryApiMock.mockResolvedValue({ entries: [] });
    placementPreviewMock.mockReset();
    placementPreviewMock.mockResolvedValue({
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
    });
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
    // Every routed machine speaks the durable contract; a host that answered
    // /api/capabilities with no queue is refused by name, never routed.
    hostCapabilitiesMock.mockReset();
    hostCapabilitiesMock.mockResolvedValue({
      queue: { heterogeneous_batch_max_outputs: 64 },
    });
    hostModelsMock.mockResolvedValue([]);
    vi.stubGlobal("prompt", vi.fn());
  });

  it("submits against this server's own route when it is the only machine", async () => {
    hostModelsMock.mockResolvedValue([flux]);
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).toHaveBeenCalledTimes(1);
    // Durable admission reconciles against a machine's instance identity, so
    // the origin carries a real route now — never `null`.
    expect(submitMock.mock.calls[0]?.[2]).toMatchObject({
      hostId: ORIGIN_HOST_ID,
    });
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

    expect(submitMock.mock.calls[0]?.[2]).toMatchObject({
      hostId: studio.id,
      label: "Studio",
      instanceId: null,
      modelFamily: "flux2",
      referenceUploads: null,
      target: { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
    });
  });

  // ── Expansion routing (issue #1162 §5) ──────────────────────────────
  function expandCapabilityFor(present: Record<string, boolean>) {
    return async (host: { id: string }) => ({
      gallery: { can_delete: true },
      // Every machine in this fleet can queue a print; the axis under test
      // is which of them holds the expander.
      queue: { heterogeneous_batch_max_outputs: 64 },
      expand: {
        configured: true,
        model_present: present[host.id] ?? null,
        backend: "local",
        model: "qwen3-expand:q8",
      },
    });
  }

  it("expands on a machine that has the expander while the print stays put", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "sk-studio",
    });
    hostModelsMock.mockResolvedValue([flux]);
    hostCapabilitiesMock.mockImplementation(
      expandCapabilityFor({ [ORIGIN_HOST_ID]: false, [studio.id]: true }),
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();

    const modal = wrapper.getComponent({ name: "ExpandModal" });
    expect(modal.props("open")).toBe(true);
    expect(modal.props("target")).toEqual({
      baseUrl: "http://studio:7680",
      apiKey: "sk-studio",
    });
    expect(wrapper.find("[data-test='web-expansion-pull']").exists()).toBe(
      false,
    );
  });

  it("keeps the generation route when its expand capability is unknown", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    hostModelsMock.mockResolvedValue([flux]);
    hostCapabilitiesMock.mockImplementation(
      expandCapabilityFor({ [studio.id]: true }),
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();

    // Unknown is never "missing": the route the generation earned is kept.
    expect(
      wrapper.getComponent({ name: "ExpandModal" }).props("target"),
    ).toEqual({ baseUrl: "http://localhost:3000" });
  });

  it("keeps the print on the generation machine after a rerouted quick expansion", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "sk-studio",
    });
    hostModelsMock.mockResolvedValue([flux]);
    hostCapabilitiesMock.mockImplementation(
      expandCapabilityFor({ [ORIGIN_HOST_ID]: false, [studio.id]: true }),
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();
    expect(
      wrapper.getComponent({ name: "ExpandModal" }).props("target"),
    ).toEqual({ baseUrl: "http://studio:7680", apiKey: "sk-studio" });

    wrapper
      .getComponent({ name: "ExpandModal" })
      .vm.$emit("apply-prompt", "storm light over the harbor");
    await nextTick();
    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    // Rewritten on Studio, printed here: the quick snapshot must freeze the
    // generation route, which for the origin is this server's own route.
    expect(submitMock).toHaveBeenCalledTimes(1);
    expect(submitMock.mock.calls[0]?.[2]).toMatchObject({
      hostId: ORIGIN_HOST_ID,
    });
  });

  it("routes Remix through the same expansion policy", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "sk-studio",
    });
    hostModelsMock.mockResolvedValue([flux]);
    hostCapabilitiesMock.mockImplementation(
      expandCapabilityFor({ [ORIGIN_HOST_ID]: false, [studio.id]: true }),
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux2-klein:q4";
    form.state.value.modelFamily = "flux2";
    form.state.value.prompt = "a lighthouse";
    await nextTick();

    wrapper.getComponent({ name: "ComposerCard" }).vm.$emit("remix");
    await flushPromises();

    // `/api/remix` runs on the same expander, so it follows the same rule.
    expect(
      wrapper.getComponent({ name: "RemixModal" }).props("target"),
    ).toEqual({ baseUrl: "http://studio:7680", apiKey: "sk-studio" });
  });

  it("offers the expander pull on a single-host install", async () => {
    hostModelsMock.mockResolvedValue([flux]);
    hostCapabilitiesMock.mockImplementation(
      expandCapabilityFor({ [ORIGIN_HOST_ID]: false }),
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();

    expect(wrapper.getComponent({ name: "ExpandModal" }).props("open")).toBe(
      false,
    );
    expect(wrapper.get("[data-test='web-expansion-pull']").text()).toContain(
      "qwen3-expand:q8",
    );
  });

  it("offers the expander pull on a pinned machine instead of leaving it", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "sk-studio",
    });
    localStorage.setItem("mold.web.generateTarget.v1", studio.id);
    hostModelsMock.mockResolvedValue([flux]);
    hostCapabilitiesMock.mockImplementation(
      expandCapabilityFor({ [ORIGIN_HOST_ID]: true, [studio.id]: false }),
    );

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a lighthouse";
    await nextTick();

    await wrapper.get("[data-test='composer-expand']").trigger("click");
    await flushPromises();

    expect(wrapper.getComponent({ name: "ExpandModal" }).props("open")).toBe(
      false,
    );
    const notice = wrapper.get("[data-test='web-expansion-pull']");
    expect(notice.text()).toContain("qwen3-expand:q8");
    expect(notice.text()).toContain("Studio");

    await wrapper
      .get("[data-test='web-expansion-pull-action']")
      .trigger("click");
    await flushPromises();
    expect(hostModelDownloadMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: studio.id }),
      "qwen3-expand:q8",
    );
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

  // Regression (#1162): a persisted or restored model the fleet no longer has
  // used to be silently swapped for `installedModels[0]` — along with that
  // model's size/steps/guidance. The id is kept and disclosed instead, and the
  // picker renders it as a not-installed option so the <select> is never blank.
  it("keeps a restored model the fleet doesn't have and discloses it", async () => {
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q8";
    form.state.value.modelFamily = "flux";
    hostModelsMock.mockResolvedValue([flux]);

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    await nextTick();

    expect(form.state.value.model).toBe("flux-dev:q8");
    const picker = wrapper.getComponent({ name: "CreateModelPicker" });
    expect(picker.props("model")).toBe("flux-dev:q8");
    expect(picker.props("missingModel")).toBe("flux-dev:q8");
    expect(
      useNotifications().toasts.some((t) => /isn't installed/.test(t.text)),
    ).toBe(true);
  });

  // #1162: Auto must never dead-end. Nobody has the model, so the pull is the
  // recovery — the print is not queued, and the resume waits on the download.
  it("offers the pull when no machine has the model instead of dead-ending", async () => {
    hostModelsMock.mockResolvedValue([flux]);
    placementPreviewMock.mockResolvedValue({
      version: 1,
      authoritative: true,
      state_version: 1,
      plan_version: 1,
      outcome: "infeasible",
      reason: "model 'z-image-turbo:q6' has no concrete local artifacts",
      missing_components: [
        {
          kind: "transformer",
          name: "transformer",
          present: false,
          repair_model: "z-image-turbo:q6",
        },
      ],
    });
    postDownloadMock.mockClear();

    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "z-image-turbo:q6";
    form.state.value.modelFamily = "zimage";
    form.state.value.prompt = "a lighthouse at dusk";
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='install-target-option']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(postDownloadMock).toHaveBeenCalledWith("z-image-turbo:q6");
    const toasts = useNotifications().toasts;
    expect(toasts.some((t) => /Pulling z-image-turbo:q6/.test(t.text))).toBe(
      true,
    );
    expect(toasts.some((t) => /can't run this print/.test(t.text))).toBe(false);
  });

  it("does not arm a late missing-model resume after planning is cancelled", async () => {
    hostModelsMock.mockResolvedValue([flux]);
    placementPreviewMock.mockResolvedValue({
      version: 1,
      authoritative: true,
      state_version: 1,
      plan_version: 1,
      outcome: "infeasible",
      reason: "model 'z-image-turbo:q6' has no concrete local artifacts",
      missing_components: [
        {
          kind: "transformer",
          name: "transformer",
          present: false,
          repair_model: "z-image-turbo:q6",
        },
      ],
    });
    let finishDownload!: () => void;
    postDownloadMock.mockClear();
    postDownloadMock.mockImplementationOnce(
      () =>
        new Promise<undefined>(
          (resolve) => (finishDownload = () => resolve(undefined)),
        ),
    );
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    const form = useGenerateForm();
    form.state.value.model = "z-image-turbo:q6";
    form.state.value.modelFamily = "zimage";
    form.state.value.prompt = "cancel the repair";
    await nextTick();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='install-target-option']").trigger("click");
    await vi.waitFor(() => expect(finishDownload).toBeTypeOf("function"));
    const cancel = wrapper.get("[data-test='composer-submit']");
    expect(cancel.text()).toContain("Cancel");
    await cancel.trigger("click");
    finishDownload();
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(cancel.text()).toContain("Generate");
    expect(
      useNotifications().toasts.some((toast) =>
        /generation starts when it's ready/.test(toast.text),
      ),
    ).toBe(false);
  });

  // The pull is only ever offered on a machine that reported the model
  // absent; one that refused for capacity would refuse again after a repair.

  it("still re-homes a genuinely unset model onto an installed one", async () => {
    const form = useGenerateForm();
    form.state.value.model = "";
    form.state.value.modelFamily = "";
    hostModelsMock.mockResolvedValue([flux]);

    mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();
    await nextTick();

    expect(form.state.value.model).toBe("flux2-klein:q4");
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

  // ── Face identity (PuLID, #1224) ──────────────────────────────────────
  // The photo is dropped from the wire whenever the combination is invalid,
  // so the submit gate is what stops Generate from quietly rendering a
  // different face under the same prompt and seed.
  const pulid = {
    ...flux,
    name: "flux-dev-pulid:bf16",
    family: "flux",
    supports_identity: true,
  };
  const PHOTO = "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC";

  function stageIdentity(extra: Partial<GenerateFormState> = {}) {
    const form = useGenerateForm();
    form.state.value.model = pulid.name;
    form.state.value.modelFamily = "flux";
    form.state.value.prompt = "a portrait";
    form.state.value.identityImage = {
      kind: "upload",
      filename: "ada.png",
      base64: PHOTO,
    };
    Object.assign(form.state.value, extra);
    return form;
  }

  it("ships the identity photo on an identity-qualified checkpoint", async () => {
    hostModelsMock.mockResolvedValue([pulid]);
    stageIdentity({ identityWeight: 1.2 });
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock.mock.calls[0]?.[0]).toMatchObject({
      id_image: PHOTO,
      id_image_name: "ada.png",
      id_weight: 1.2,
    });
  });

  it("blocks Generate for a combination admission would refuse", async () => {
    hostModelsMock.mockResolvedValue([pulid]);
    stageIdentity({ loras: [{ path: "style.safetensors", scale: 1 }] });
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    await wrapper.get("[data-test='composer-submit']").trigger("click");
    await flushPromises();

    expect(submitMock).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='composer-submit-error']").text()).toContain(
      "cannot be combined with a LoRA",
    );
  });

  it("renders the identity well beside the source media, not in Advanced", async () => {
    hostModelsMock.mockResolvedValue([pulid]);
    stageIdentity();
    const wrapper = mount(CreatePage, { global: { stubs: pageStubs() } });
    await flushPromises();

    // AdvancedDrawer is stubbed out entirely here, so finding the well proves
    // it lives in the primary form.
    expect(wrapper.find("[data-test='identity-panel']").exists()).toBe(true);
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
      props: ["busy", "cancellable", "busyLabel", "disabledReason"],
      template:
        '<div><div data-test="prompt-style-stub"/><slot name="mobile-controls"/><p v-if="disabledReason" data-test="page-generation-blocker">{{ disabledReason }}</p><p v-if="cancellable">{{ busyLabel }}</p><button data-test="composer-submit" @click="$emit(cancellable ? \'cancel\' : \'submit\')">{{ cancellable ? "Cancel" : "Generate" }}</button><button data-test="composer-expand" @click="$emit(\'expand\')">expand</button><button data-test="composer-undo" @click="$emit(\'undo-expand\')">undo</button></div>',
      // The page calls these through its template ref on submit / new-print;
      // a stub without them throws an unhandled TypeError mid-run.
      methods: { record: vi.fn(), focus: vi.fn() },
    },
    ResultCanvas: {
      name: "ResultCanvas",
      props: ["mode", "variations", "resultCaption", "previewSrc", "stage"],
      template:
        '<div data-test="result-canvas" :data-count="(variations||[]).length" :data-caption="resultCaption" :data-preview-src="previewSrc" :data-stage="stage"><button data-test="queue-variations" @click="$emit(\'queue\')">queue</button></div>',
    },
    CreateModelPicker: {
      name: "CreateModelPicker",
      props: ["models", "model", "browseTo", "emptyLabel", "missingModel"],
      template: "<div data-test='model-picker-stub' />",
    },
    ControlsAside: {
      name: "ControlsAside",
      props: ["output", "clipCount"],
      // The File under group rides the rail's `file-under` slot, so the stub
      // has to render it for placement to be observable.
      template:
        "<aside data-test='controls-stub'><slot name='file-under'/></aside>",
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
        "submitting",
      ],
      emits: [
        "submit",
        "cancel",
        "duplicate-as-new",
        "discard-edit",
        "expand-clip",
        "import-shared",
        "play-clip",
      ],
      template:
        "<div data-test='sequence-composer-stub'>" +
        "<button data-test='sequence-generate' @click=\"$emit(submitting ? 'cancel' : 'submit')\">{{ submitting ? 'Cancel' : 'go' }}</button>" +
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
        "styleDirective",
        "task",
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
