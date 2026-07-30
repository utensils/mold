import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";

const { routerPush, routerReplace, routeQuery } = vi.hoisted(() => ({
  routerPush: vi.fn(),
  routerReplace: vi.fn(),
  routeQuery: { value: {} as Record<string, unknown> },
}));
vi.mock("vue-router", () => ({
  useRouter: () => ({ push: routerPush, replace: routerReplace }),
  useRoute: () => ({ query: routeQuery.value }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
const apiFetchTo = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
  apiFetchTo: (...args: unknown[]) => apiFetchTo(...args),
  ApiError: class ApiError extends Error {
    status = 0;
  },
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
  },
}));
vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn().mockResolvedValue(undefined) }));

import GenerateView from "./GenerateView.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useGenerateFormStore } from "../stores/generateForm";
import { useChainJobsStore } from "../stores/chainJobs";
import { useComposerStore } from "../stores/composer";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useModelStore } from "../stores/models";
import { useToastStore } from "../stores/toasts";
import { useUiStore } from "../stores/ui";
import { useContextMenuStore } from "../stores/contextMenu";
import { newJob, useGenerationStore } from "../stores/generation";
import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import type { ChainJobDetail } from "@studio/lib/api/chainTypes";

enableAutoUnmount(afterEach);

const videoModel: ModelEntry = {
  name: "ltx-video",
  family: "ltx-video",
  downloaded: true,
  default_width: 1024,
  default_height: 576,
  default_steps: 25,
  default_guidance: 3,
} as ModelEntry;

const imageModel: ModelEntry = {
  name: "flux-schnell:q8",
  family: "flux",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 4,
  default_guidance: 1,
} as ModelEntry;

function readyLocal() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
}

function mountView() {
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: { stubs: { SequenceComposer: true, ComposerCard: true } },
  });
}

let installedPayload: ModelEntry[] = [];

beforeEach(() => {
  setActivePinia(createPinia());
  routerPush.mockClear();
  routerReplace.mockClear();
  routeQuery.value = {};
  installedPayload = [];
  apiJson.mockReset();
  apiJson.mockImplementation((path: unknown) =>
    Promise.resolve(path === "/api/models" ? installedPayload : []),
  );
  apiJsonTo.mockReset();
  apiFetchTo.mockReset();
  apiJsonTo.mockImplementation((_target: unknown, path: unknown) => {
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    if (path === "/api/models") return Promise.resolve(installedPayload);
    return Promise.resolve({});
  });
  window.localStorage?.clear?.();
});
afterEach(() => (document.body.innerHTML = ""));

describe("GenerateView — sequence output", () => {
  it("offers Save image and Copy file path on a completed still", async () => {
    readyLocal();
    installedPayload = [imageModel];
    useModelStore().all = [imageModel];
    const job = newJob({
      prompt: "moonlit alley",
      model: imageModel.name,
      width: 1024,
      height: 1024,
      steps: 4,
    });
    job.clientId = 1;
    job.status = "complete";
    job.resultUrl = "blob:result";
    job.result = {
      image: "aW1hZ2U=",
      format: "png",
      width: 1024,
      height: 1024,
      seed_used: 9,
      generation_time_ms: 10,
      model: imageModel.name,
      filename: "remote-print.png",
    };
    const generation = useGenerationStore();
    generation.jobs = [job];
    generation.selectedClientId = 1;

    const wrapper = mountView();
    await flushPromises();
    await wrapper.get("[data-test='preview-frame']").trigger("contextmenu");
    const labels = useContextMenuStore().entries.flatMap((entry) =>
      "separator" in entry ? [] : [entry.label],
    );
    expect(labels).toContain("Save image…");
    expect(labels).toContain("Copy file path");
  });

  it.each([
    {
      kind: "image",
      model: imageModel,
      metadata: {
        prompt: "a lighthouse at dawn",
        model: imageModel.name,
        seed: 101,
        steps: 4,
        guidance: 1,
        width: 1024,
        height: 1024,
      } as OutputMetadata,
    },
    {
      kind: "video",
      model: videoModel,
      metadata: {
        prompt: "a plane crosses the runway",
        model: videoModel.name,
        seed: 202,
        steps: 31,
        guidance: 4,
        width: 1024,
        height: 576,
        frames: 121,
        fps: 30,
      } as OutputMetadata,
    },
  ])(
    "restores a normal $kind print into One shot while Sequence is active",
    async ({ model, metadata }) => {
      readyLocal();
      installedPayload = [imageModel, videoModel];
      useModelStore().all = [imageModel, videoModel];
      const formStore = useGenerateFormStore();
      formStore.form.model = videoModel.name;
      formStore.form.family = videoModel.family;
      const draft = useSequenceDraftStore();
      draft.output = "sequence";
      draft.ensureClips(25);
      draft.clips[0]!.prompt = "stale sequence opening";
      draft.loadFromJob(
        {
          jobId: "sequence-being-edited",
          hostId: "local",
          baseline: draft.clips.map((clip) => ({ ...clip })),
          completedStages: 0,
        },
        draft.clips.map((clip) => ({ ...clip })),
        false,
      );
      useComposerStore().set({ metadata });

      mountView();
      await flushPromises();

      expect(draft.output).toBe("single");
      expect(draft.editing).toBeNull();
      expect(draft.lastSingleModel).toBeNull();
      expect(formStore.form.model).toBe(model.name);
      expect(formStore.form.prompt).toBe(metadata.prompt);
      expect(formStore.form.seed).toBe(String(metadata.seed));
      expect(formStore.form.steps).toBe(metadata.steps);
      if (metadata.frames) {
        expect(formStore.form.frames).toBe(metadata.frames);
        expect(formStore.form.fps).toBe(metadata.fps);
      }
    },
  );

  it("consumes ?output=sequence without leaking the one-shot prompt, then strips the query", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.prompt = "a storm rolls in";
    routeQuery.value = { output: "sequence" };
    mountView();
    await flushPromises();

    const draft = useSequenceDraftStore();
    expect(draft.output).toBe("sequence");
    expect(draft.clips.length).toBeGreaterThanOrEqual(2);
    expect(draft.clips[0]!.prompt).toBe("");
    expect(useGenerateFormStore().form.prompt).toBe("a storm rolls in");
    expect(routerReplace).toHaveBeenCalledWith({ path: "/create" });
  });

  it("swaps to a sequence-capable model BEFORE seeding clips on deep-link", async () => {
    // Deep-linking ?output=sequence while a still model is selected must
    // not seed the clips with the still model's (absent) frame default —
    // the capable model is applied first so defaultClipFrames sees it.
    const stillModel = {
      name: "flux-dev:q8",
      family: "flux",
      downloaded: true,
      default_width: 1024,
      default_height: 1024,
      default_steps: 25,
      default_guidance: 3.5,
    } as ModelEntry;
    const ltx2 = {
      ...videoModel,
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      default_frames: 97,
    } as ModelEntry;
    readyLocal();
    installedPayload = [stillModel, ltx2];
    useModelStore().all = [stillModel, ltx2];
    const formStore = useGenerateFormStore();
    formStore.form.model = "flux-dev:q8";
    routeQuery.value = { output: "sequence" };
    mountView();
    await flushPromises();

    const draft = useSequenceDraftStore();
    expect(formStore.form.model).toBe("ltx-2-19b-distilled:fp8");
    expect(draft.lastSingleModel).toBe("flux-dev:q8");
    // 97 comes from the swapped-in model's server-advertised default; the
    // pre-fix ordering seeded 25 (the generic floor) from the still model.
    expect(draft.clips[0]!.frames).toBe(97);
  });

  it("re-homes a restored still model onto a compatible model installed on the pinned host", async () => {
    const stillModel = {
      name: "flux-schnell:q8",
      family: "flux",
      downloaded: true,
      default_width: 1024,
      default_height: 1024,
      default_steps: 4,
      default_guidance: 1,
    } as ModelEntry;
    const platoVideo = {
      ...videoModel,
      name: "ltx-2.3-22b-distilled:fp8",
      family: "ltx2",
      default_frames: 97,
    } as ModelEntry;
    readyLocal();
    useHostsStore().extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    useAppPrefsStore().settings = {
      generateTargetHost: "plato-7680",
    } as never;
    installedPayload = [stillModel];
    useModelStore().all = [stillModel];
    apiJsonTo.mockImplementation((target: { baseUrl?: string }, path: unknown) => {
      if (path === "/api/models") {
        return Promise.resolve(
          target.baseUrl === "http://plato:7680" ? [stillModel, platoVideo] : [stillModel],
        );
      }
      if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
      return Promise.resolve({});
    });
    const formStore = useGenerateFormStore();
    formStore.form.model = stillModel.name;
    formStore.form.family = stillModel.family;
    useSequenceDraftStore().output = "sequence";

    mountView();
    await flushPromises();

    expect(formStore.form.model).toBe(platoVideo.name);
    expect(formStore.form.family).toBe("ltx2");
  });

  it("clears an incompatible pinned-host model, shows the empty state, and refuses submit", async () => {
    const stillModel = {
      name: "flux-schnell:q8",
      family: "flux",
      downloaded: true,
      default_width: 1024,
      default_height: 1024,
      default_steps: 4,
      default_guidance: 1,
    } as ModelEntry;
    readyLocal();
    useHostsStore().extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    useAppPrefsStore().settings = {
      generateTargetHost: "plato-7680",
    } as never;
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    apiJsonTo.mockImplementation((target: { baseUrl?: string }, path: unknown) => {
      if (path === "/api/models") {
        return Promise.resolve(
          target.baseUrl === "http://plato:7680" ? [stillModel] : [videoModel],
        );
      }
      if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
      return Promise.resolve({});
    });
    const formStore = useGenerateFormStore();
    formStore.form.model = stillModel.name;
    formStore.form.family = stillModel.family;
    useSequenceDraftStore().output = "sequence";

    const wrapper = mountView();
    await flushPromises();

    expect(formStore.form.model).toBe("");
    expect(wrapper.find("[data-test='sequence-empty']").exists()).toBe(true);
    expect(wrapper.find("sequence-composer-stub").exists()).toBe(false);

    apiJsonTo.mockClear();
    useUiStore().generateTick += 1;
    await flushPromises();
    expect(
      apiJsonTo.mock.calls.some(
        ([, path, init]) =>
          path === "/api/chain-jobs" && (init as RequestInit | undefined)?.method === "POST",
      ),
    ).toBe(false);
    expect(
      useToastStore().items.some((toast) => toast.message.includes("sequence-capable video model")),
    ).toBe(true);
  });

  it("renders the sequence bench instead of the single composer", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.model = "ltx-video";
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.find("sequence-composer-stub").exists()).toBe(true);
    expect(wrapper.find("composer-card-stub").exists()).toBe(false);
  });

  it("keeps the single composer for one-shot output", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const wrapper = mountView();
    await flushPromises();
    expect(wrapper.find("composer-card-stub").exists()).toBe(true);
    expect(wrapper.find("sequence-composer-stub").exists()).toBe(false);
  });

  it("amends with an explicit enable_audio boolean so edits can turn audio off", async () => {
    // null means "keep current" server-side — sending it when the draft's
    // audio is off would make disabling audio via edit-in-place impossible.
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const formStore = useGenerateFormStore();
    formStore.form.model = "ltx-video";
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "opening";
    draft.clips[1]!.prompt = "landing";
    draft.enableAudio = false; // user turned audio OFF during the edit
    draft.loadFromJob(
      {
        jobId: "job-1",
        hostId: "local",
        baseline: draft.clips.map((c) => ({ ...c })),
        completedStages: 2,
      },
      draft.clips.map((c) => ({ ...c })),
      false,
    );

    const amendCalls: unknown[] = [];
    apiJsonTo.mockImplementation((_target: unknown, path: unknown, init?: unknown) => {
      if (typeof path === "string" && path.endsWith("/amend")) {
        amendCalls.push(init);
        return Promise.resolve({
          id: "job-1",
          state: "queued",
          model: "ltx-video",
          stage_count: 2,
          current_stage: 0,
          created_at_unix_ms: 1,
          updated_at_unix_ms: 2,
          preserved_stages: 1,
        });
      }
      if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
      if (path === "/api/models") return Promise.resolve(installedPayload);
      return Promise.resolve({});
    });

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent({ name: "SequenceComposer" }).vm.$emit("submit");
    await flushPromises();

    expect(amendCalls.length).toBe(1);
    const body = JSON.parse((amendCalls[0] as { body: string }).body);
    expect(body.enable_audio).toBe(false);
  });

  it("guides to Discover when no chain-capable video model is installed", async () => {
    readyLocal();
    installedPayload = [];
    apiJsonTo.mockRejectedValue(new Error("offline"));
    useModelStore().all = [];
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.find("[data-test='sequence-empty']").exists()).toBe(true);
    expect(wrapper.find("sequence-composer-stub").exists()).toBe(false);
  });
});

// Settling must never blank the canvas: the strip no longer keeps a settled
// row, so the canvas is where the finished sequence lands.
describe("GenerateView — settled sequence canvas", () => {
  function watchSequence(state: ChainJobDetail["state"], extra: Partial<ChainJobDetail> = {}) {
    const chains = useChainJobsStore();
    chains.watching = { hostId: "local", jobId: "job-1" };
    chains.live = {
      detail: {
        id: "job-1",
        state,
        model: "ltx-video",
        stage_count: 2,
        current_stage: 1,
        created_at_unix_ms: 1,
        updated_at_unix_ms: 2,
        stages: [],
        ...extra,
      } as unknown as ChainJobDetail,
      progress: {},
      activeStage: null,
    };
    return chains;
  }

  async function sequenceView() {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    useGenerateFormStore().form.model = "ltx-video";
    const wrapper = mountView();
    await flushPromises();
    return wrapper;
  }

  it("holds the finished sequence with Edit sequence and Show in library", async () => {
    watchSequence("completed");
    const wrapper = await sequenceView();

    const result = wrapper.get("[data-test='sequence-result']");
    expect(result.find("[data-test='sequence-edit']").exists()).toBe(true);
    expect(result.find("[data-test='sequence-show-in-library']").exists()).toBe(true);
    expect(wrapper.find("[data-test='empty-canvas']").exists()).toBe(false);
  });

  it("keeps a failed sequence inspectable with Resume", async () => {
    const chains = watchSequence("failed", { error: "CUDA ran out of memory" });
    vi.spyOn(chains, "resume").mockResolvedValue();
    const wrapper = await sequenceView();

    const notice = wrapper.get("[data-test='sequence-failed']");
    expect(notice.attributes("message")).toContain("GPU memory");
    await wrapper.get("[data-test='sequence-resume']").trigger("click");
    expect(chains.resume).toHaveBeenCalledWith("local", "job-1");
    expect(wrapper.find("[data-test='empty-canvas']").exists()).toBe(false);
  });

  it("re-enters a job handed over from the Library", async () => {
    const chains = useChainJobsStore();
    const detail = vi.spyOn(chains, "fetchDetail").mockRejectedValue(new Error("not in this test"));
    useComposerStore().setSequence({ kind: "edit", hostId: "okra-7680", jobId: "job-9" });
    await sequenceView();

    expect(detail).toHaveBeenCalledWith("okra-7680", "job-9");
    // One-shot: a back-nav must not replay the handoff.
    expect(useComposerStore().pendingSequence).toBeNull();
  });
});

describe("GenerateView — cached filmstrip remount", () => {
  function cachedDetail(id = "job-cached"): ChainJobDetail {
    return {
      id,
      state: "completed",
      model: "ltx-video",
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
          frames_emitted: 25,
          generation_time_ms: 10,
          has_preview: true,
          has_media: false,
          cache_ready: true,
          error: null,
        },
        {
          idx: 1,
          state: "completed",
          seed: "2",
          frames_emitted: 25,
          generation_time_ms: 10,
          has_preview: false,
          has_media: false,
          cache_ready: true,
          error: null,
        },
      ],
      script: {
        schema: "mold.chain.v1",
        chain: { model: "ltx-video", fps: 24 },
        stages: [
          { prompt: "server opening", frames: 25 },
          { prompt: "server ending", frames: 25 },
        ],
      },
      finalizes: [],
      retakes: [],
      amends: [],
    } as ChainJobDetail;
  }

  function prepareCachedEdit(detail: ChainJobDetail) {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.model = "ltx-video";
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "edited opening";
    draft.clips[1]!.prompt = "edited ending";
    draft.loadFromJob(
      {
        jobId: detail.id,
        hostId: "local",
        baseline: draft.clips.map((clip) => ({ ...clip })),
        completedStages: 2,
      },
      draft.clips.map((clip) => ({ ...clip })),
      false,
    );
    const chains = useChainJobsStore();
    chains.watching = { hostId: "local", jobId: detail.id };
    chains.live = { detail, progress: {}, activeStage: null };
    return { chains, draft };
  }

  it("restores durable preview bindings after unmount and remount", async () => {
    const detail = cachedDetail();
    const { draft } = prepareCachedEdit(detail);
    apiFetchTo.mockResolvedValue({
      blob: async () => new Blob(["preview"], { type: "image/jpeg" }),
    });
    const createUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValueOnce("blob:first")
      .mockReturnValueOnce("blob:second");
    const revokeUrl = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    const first = mountView();
    await flushPromises();
    const firstMedia = first
      .getComponent({ name: "SequenceComposer" })
      .props("stageMediaByClipId") as Record<string, { posterUrl?: string }>;
    expect(firstMedia[draft.clips[0]!.id]?.posterUrl).toBe("blob:first");

    first.unmount();
    expect(revokeUrl).toHaveBeenCalledWith("blob:first");

    const second = mountView();
    await flushPromises();
    const secondMedia = second
      .getComponent({ name: "SequenceComposer" })
      .props("stageMediaByClipId") as Record<string, { posterUrl?: string }>;
    expect(secondMedia[draft.clips[0]!.id]?.posterUrl).toBe("blob:second");
    expect(apiFetchTo).toHaveBeenCalledTimes(2);
    createUrl.mockRestore();
    revokeUrl.mockRestore();
  });

  it("discards a late preview from the previously watched job", async () => {
    const detail = cachedDetail("job-a");
    const { chains } = prepareCachedEdit(detail);
    let resolvePreview!: (response: { blob: () => Promise<Blob> }) => void;
    apiFetchTo.mockReturnValue(
      new Promise((resolve) => {
        resolvePreview = resolve;
      }),
    );
    const createUrl = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:stale");
    const revokeUrl = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const wrapper = mountView();
    await flushPromises();

    const other = cachedDetail("job-b");
    other.stages = other.stages.map((stage) => ({ ...stage, has_preview: false }));
    chains.watching = { hostId: "local", jobId: "job-b" };
    chains.live = { detail: other, progress: {}, activeStage: null };
    await flushPromises();
    resolvePreview({ blob: async () => new Blob(["late"], { type: "image/jpeg" }) });
    await flushPromises();

    expect(revokeUrl).toHaveBeenCalledWith("blob:stale");
    expect(wrapper.getComponent({ name: "SequenceComposer" }).props("stageMediaByClipId")).toEqual(
      {},
    );
    createUrl.mockRestore();
    revokeUrl.mockRestore();
  });
});

// Reuse settings on a sequence print: a NEW draft from the recorded clips.
// The load-bearing difference from Edit is that nothing is cached and no edit
// session exists — Generate sequence queues a fresh job.
describe("GenerateView — sequence reuse handoff", () => {
  function chainMetadata(frames: number[], extra: Partial<OutputMetadata> = {}): OutputMetadata {
    return {
      prompt: frames.map((_, i) => `clip ${i + 1}`).join("\n"),
      model: "ltx-video",
      seed: 4242,
      steps: 25,
      guidance: 3,
      width: 1024,
      height: 576,
      chain_job_id: "job-9",
      chain: {
        stage_count: frames.length,
        motion_tail_frames: 8,
        stages: frames.map((f, i) => ({
          prompt: `clip ${i + 1}`,
          frames: f,
          transition: "smooth" as const,
        })),
      },
      ...extra,
    } as OutputMetadata;
  }

  async function reuseView(metadata: OutputMetadata, model: ModelEntry = videoModel) {
    readyLocal();
    installedPayload = [model];
    useModelStore().all = [model];
    useGenerateFormStore().form.prompt = "parked one shot";
    const draft = useSequenceDraftStore();
    draft.hydrate();
    useComposerStore().setSequence({ kind: "reuse", metadata });
    const wrapper = mountView();
    await flushPromises();
    return { wrapper, draft };
  }

  it("loads the recorded clips as a fresh draft with no edit session", async () => {
    const { wrapper, draft } = await reuseView(chainMetadata([97, 65, 33]));

    expect(draft.output).toBe("sequence");
    expect(draft.clips.map((c) => c.prompt)).toEqual(["clip 1", "clip 2", "clip 3"]);
    expect(draft.clips.map((c) => c.frames)).toEqual([97, 65, 33]);
    expect(draft.editing).toBeNull();
    expect(useGenerateFormStore().form.seed).toBe("4242");
    expect(useGenerateFormStore().form.prompt).toBe("parked one shot");
    expect(useComposerStore().pendingSequence).toBeNull();
    // The confirmation line is always there; it just has nothing to disclaim.
    expect(wrapper.get("[data-test='sequence-reuse-note']").text()).toBe("reused 3 clips");
  });

  it("discloses what the print could not give back, once", async () => {
    const { wrapper } = await reuseView(
      chainMetadata([97, 65], {
        negative_prompt: "blurry",
        source_image_sha256: "deadbeef",
      } as Partial<OutputMetadata>),
    );

    const note = wrapper.get("[data-test='sequence-reuse-note']");
    expect(note.text()).toBe(
      "reused 2 clips · negatives and clip sources aren't recorded in prints",
    );
    expect(wrapper.findAll("[data-test='sequence-reuse-note']")).toHaveLength(1);
  });

  it("raises clips that no longer clear the current model's motion tail, and says so", async () => {
    // The print was rendered on a zero-tail LTX-Video model; the reuse lands
    // on an LTX-2 model whose tail is 17, so a 9-frame clip is now invalid.
    const ltx2 = {
      ...videoModel,
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      default_frames: 97,
    } as ModelEntry;
    const { wrapper, draft } = await reuseView(
      chainMetadata([9, 65], { model: "ltx-2-19b-distilled:fp8" } as Partial<OutputMetadata>),
      ltx2,
    );

    // 9 → the first 8n+1 duration that clears the 17-frame tail; 65 is fine.
    expect(draft.clips.map((c) => c.frames)).toEqual([25, 65]);
    expect(draft.clips.every((c) => c.frames > 17)).toBe(true);
    expect(wrapper.get("[data-test='sequence-reuse-note']").text()).toContain(
      "Clip durations raised to fit",
    );
  });

  it("ignores a legacy print with no recorded clips", async () => {
    const { draft } = await reuseView({
      prompt: "one shot",
      model: "ltx-video",
      seed: 1,
      steps: 25,
      guidance: 3,
      width: 1024,
      height: 576,
    } as OutputMetadata);

    expect(draft.output).toBe("single");
    expect(useComposerStore().pendingSequence).toBeNull();
  });
});
