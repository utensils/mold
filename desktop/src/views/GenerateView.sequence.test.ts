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
vi.mock("../lib/api/client", () => ({
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
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
import { useModelStore } from "../stores/models";
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
  apiJsonTo.mockImplementation((_target: unknown, path: unknown) => {
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    if (path === "/api/models") return Promise.resolve(installedPayload);
    return Promise.resolve({});
  });
  window.localStorage?.clear?.();
});
afterEach(() => (document.body.innerHTML = ""));

describe("GenerateView — sequence output", () => {
  it("consumes ?output=sequence once, then strips the query", async () => {
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
    expect(draft.clips[0]!.prompt).toBe("a storm rolls in");
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
    // Never the newline join — that is the wart this path exists to avoid.
    expect(useGenerateFormStore().form.prompt).toBe("clip 1");
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
