import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";

const { routerPush, routerReplace, routeQuery, licenseRequest, placementDownloads } = vi.hoisted(
  () => ({
    routerPush: vi.fn(),
    routerReplace: vi.fn(),
    routeQuery: { value: {} as Record<string, unknown> },
    licenseRequest: vi.fn().mockResolvedValue({ accepted: true, downloaded: false }),
    placementDownloads: { value: [] as unknown[] },
  }),
);
vi.mock("vue-router", () => ({
  useRouter: () => ({ push: routerPush, replace: routerReplace }),
  useRoute: () => ({ query: routeQuery.value }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
const apiFetchTo = vi.fn();
const applySourceFitPreprocess = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
  apiFetchTo: (...args: unknown[]) => apiFetchTo(...args),
  ApiError: class ApiError extends Error {
    status = 0;
  },
}));
vi.mock("@studio/api/generationPlacement", async (importOriginal) => {
  const original = await importOriginal<typeof import("@studio/api/generationPlacement")>();
  const planned = () =>
    Promise.resolve({
      version: 1,
      authoritative: true,
      state_version: 1,
      plan_version: 1,
      outcome: "planned",
      candidate: {
        device_id: "local",
        execution_fingerprint: "sequence-test",
        predicted_start_after_ms: 0,
        predicted_completion_after_ms: 1,
        setup_ms: 0,
        setup_kind: "warm",
        estimate_confidence: "high",
      },
      pending_downloads: placementDownloads.value,
    });
  return {
    ...original,
    previewGenerationPlacement: vi.fn(planned),
    previewChainPlacement: vi.fn(planned),
  };
});
vi.mock("@studio/composables/useLicenseAcceptance", () => ({
  useLicenseAcceptance: () => ({ request: licenseRequest }),
  // The downloads store wraps its enqueue in this; the mock must keep the
  // module's whole surface or every test in the file loses the store.
  runWithLicenseConsent: async (options: { start: () => Promise<unknown> }) => ({
    kind: "ok" as const,
    value: await options.start(),
  }),
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    saveMediaBytes: vi.fn(),
    revealSavedMedia: vi.fn(),
  },
}));
vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn().mockResolvedValue(undefined) }));
vi.mock("../lib/sourceFitPreprocess", () => ({
  applySourceFitPreprocess: (...args: unknown[]) => applySourceFitPreprocess(...args),
}));

import GenerateView from "./GenerateView.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { addTag } from "@studio/lib/fileUnder";
import { useGenerateFormStore } from "../stores/generateForm";
import { useChainJobsStore } from "../stores/chainJobs";
import { useComposerStore } from "../stores/composer";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useHostModelsStore } from "../stores/hostModels";
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
  licenseRequest.mockReset().mockResolvedValue({ accepted: true, downloaded: false });
  placementDownloads.value = [];
  installedPayload = [];
  apiJson.mockReset();
  apiJson.mockImplementation((path: unknown) =>
    Promise.resolve(path === "/api/models" ? installedPayload : []),
  );
  apiJsonTo.mockReset();
  apiFetchTo.mockReset();
  applySourceFitPreprocess
    .mockReset()
    .mockImplementation((input) =>
      Promise.resolve({ source: input.source, mask: input.mask, changed: false }),
    );
  apiJsonTo.mockImplementation((_target: unknown, path: unknown) => {
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    if (path === "/api/models") return Promise.resolve(installedPayload);
    if (path === "/api/generate/placement-preview") {
      return Promise.resolve({
        version: 1,
        authoritative: true,
        state_version: 1,
        plan_version: 1,
        outcome: "planned",
        candidate: {
          device_id: "local",
          execution_fingerprint: "sequence-test",
          predicted_start_after_ms: 0,
          predicted_completion_after_ms: 1,
          setup_ms: 0,
          setup_kind: "warm",
          estimate_confidence: "high",
        },
      });
    }
    return Promise.resolve({});
  });
  window.localStorage?.clear?.();
});
afterEach(() => (document.body.innerHTML = ""));

describe("GenerateView — sequence output", () => {
  /** A ready clip draft on a chain-capable model, mounted. */
  async function clipMode() {
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
    return { wrapper, draft };
  }

  it("queues nothing on license cancel and resumes exactly once after acceptance", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.model = videoModel.name;
    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "opening shot";
    draft.clips[1]!.prompt = "closing shot";
    placementDownloads.value = [
      {
        kind: "future_runtime",
        name: "future-runtime.bin",
        repo: "future/repository",
        bytes: 1024,
        install_model: "future-video-assets",
        licenses: [
          {
            id: "future-video-license",
            name: "Future video terms",
            url: "https://example.test/pinned",
            canonical: "https://example.test/project",
            sha256: "c".repeat(64),
            summary: "Research use only.",
          },
        ],
      },
    ];
    licenseRequest
      .mockResolvedValueOnce({ accepted: false, downloaded: false })
      .mockResolvedValueOnce({ accepted: true, downloaded: false });
    apiFetchTo.mockResolvedValue(Response.json({ job_id: "licensed-sequence" }));

    const wrapper = mountView();
    await flushPromises();
    const composer = wrapper.findComponent({ name: "ComposerCard" });
    apiJsonTo.mockClear();
    composer.vm.$emit("generate");
    await flushPromises();
    await vi.waitFor(() => expect(licenseRequest).toHaveBeenCalledTimes(1));
    expect(
      apiFetchTo.mock.calls.filter(
        ([, path, init]) =>
          path === "/api/chain-jobs" && (init as RequestInit | undefined)?.method === "POST",
      ),
    ).toHaveLength(0);

    composer.vm.$emit("generate");
    await flushPromises();
    await vi.waitFor(() => expect(licenseRequest).toHaveBeenCalledTimes(2));
    expect(
      apiFetchTo.mock.calls.filter(
        ([, path, init]) =>
          path === "/api/chain-jobs" && (init as RequestInit | undefined)?.method === "POST",
      ),
    ).toHaveLength(1);
    expect(licenseRequest).toHaveBeenLastCalledWith(
      expect.objectContaining({
        requirements: [expect.objectContaining({ installModel: "future-video-assets" })],
      }),
    );
  });

  it("offers Save image, Use as source, and Copy file path on a completed still", async () => {
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
    expect(labels).toContain("Save image");
    expect(labels).toContain("Start from this photo");
    expect(labels).toContain("Copy file path");

    const useAsSource = useContextMenuStore().entries.find(
      (entry) => !("separator" in entry) && entry.label === "Start from this photo",
    );
    expect(useAsSource).toMatchObject({ disabled: false });
    useContextMenuStore().activate(useAsSource!);
    await flushPromises();
    expect(useGenerateFormStore().form.sourceImage).toBe("aW1hZ2U=");
    expect(useGenerateFormStore().form.sourceImageName).toBe("remote-print.png");
    expect(useGenerateFormStore().form.sourceFit).toEqual({ mode: "crop-fill" });
  });

  // The Library has always taken a rendered clip back in as LTX source video;
  // the finished render on the canvas is the same print, so its menu offers
  // the same thing instead of graying the item out.
  it("offers a completed video render as source video", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const job = newJob({
      prompt: "a plane crosses the runway",
      model: videoModel.name,
      width: 1024,
      height: 576,
      steps: 31,
    });
    job.clientId = 1;
    job.status = "complete";
    job.resultUrl = "blob:clip";
    job.result = {
      image: "dmlkZW8=",
      format: "mp4",
      width: 1024,
      height: 576,
      seed_used: 11,
      generation_time_ms: 10,
      model: videoModel.name,
      filename: "remote-clip.mp4",
      video_frames: 97,
    };
    const generation = useGenerationStore();
    generation.jobs = [job];
    generation.selectedClientId = 1;

    const wrapper = mountView();
    await flushPromises();
    await wrapper.get("[data-test='preview-frame']").trigger("contextmenu");
    const useAsSource = useContextMenuStore().entries.find(
      (entry) => !("separator" in entry) && entry.label === "Start from this photo",
    );
    expect(useAsSource).toMatchObject({ disabled: false });
    useContextMenuStore().activate(useAsSource!);
    await flushPromises();
    expect(useGenerateFormStore().form.sourceVideo).toMatchObject({
      filename: "remote-clip.mp4",
      base64: "dmlkZW8=",
    });
    expect(useGenerateFormStore().form.sourceImage).toBeNull();
  });

  // Every ordinary desktop submission settles through `applyDurableCompletion`,
  // which records the FILE the host saved and NO inline bytes. A menu that
  // asked `result.image` therefore refused the prints it exists for, and its
  // handler returned early. The print's bytes come from its host instead.
  it.each([
    {
      kind: "still",
      filename: "durable-still.png",
      format: "png" as const,
      bytes: new Uint8Array([65, 66, 67]),
      expected: { sourceImage: "QUJD", sourceImageName: "durable-still.png" },
    },
    {
      kind: "clip",
      filename: "durable-clip.mp4",
      format: "mp4" as const,
      bytes: new Uint8Array([65, 66, 67]),
      expected: { sourceVideo: { filename: "durable-clip.mp4", base64: "QUJD" } },
    },
  ])("uses a durable $kind completion as source by reading its host bytes", async (row) => {
    readyLocal();
    const model = row.format === "mp4" ? videoModel : imageModel;
    installedPayload = [model];
    useModelStore().all = [model];
    apiFetchTo.mockImplementation((_target: unknown, path: string) =>
      String(path).includes(row.filename)
        ? Promise.resolve(new Response(row.bytes))
        : Promise.resolve(new Response("{}")),
    );
    const job = newJob({
      prompt: "a durable print",
      model: model.name,
      width: 1024,
      height: 576,
      steps: 4,
    });
    job.clientId = 1;
    job.status = "complete";
    job.resultUrl = "blob:durable";
    // Exactly what `applyDurableCompletion` writes: no bytes, a filename.
    job.result = {
      image: "",
      format: row.format,
      width: 1024,
      height: 576,
      seed_used: 5,
      generation_time_ms: 10,
      model: model.name,
      filename: row.filename,
      metadata: null,
    };
    const generation = useGenerationStore();
    generation.jobs = [job];
    generation.selectedClientId = 1;

    const wrapper = mountView();
    await flushPromises();
    await wrapper.get("[data-test='preview-frame']").trigger("contextmenu");
    const useAsSource = useContextMenuStore().entries.find(
      (entry) => !("separator" in entry) && entry.label === "Start from this photo",
    );
    expect(useAsSource).toMatchObject({ disabled: false });
    useContextMenuStore().activate(useAsSource!);
    await flushPromises();
    expect(useGenerateFormStore().form).toMatchObject(row.expected);
  });

  it.each([
    { kind: "audio", filename: "durable-score.wav", format: "wav" as const },
    { kind: "mesh", filename: "durable-bust.glb", format: "glb" as const },
  ])("still refuses a durable $kind completion as a source", async (row) => {
    readyLocal();
    installedPayload = [imageModel];
    useModelStore().all = [imageModel];
    const job = newJob({
      prompt: "a durable print",
      model: imageModel.name,
      width: 1024,
      height: 1024,
      steps: 4,
    });
    job.clientId = 1;
    job.status = "complete";
    job.resultUrl = "blob:durable";
    job.result = {
      image: "",
      format: row.format,
      width: 1024,
      height: 1024,
      seed_used: 5,
      generation_time_ms: 10,
      model: imageModel.name,
      filename: row.filename,
      metadata: null,
    };
    const generation = useGenerationStore();
    generation.jobs = [job];
    generation.selectedClientId = 1;

    const wrapper = mountView();
    await flushPromises();
    await wrapper.get("[data-test='preview-frame']").trigger("contextmenu");
    const useAsSource = useContextMenuStore().entries.find(
      (entry) => !("separator" in entry) && entry.label === "Start from this photo",
    );
    expect(useAsSource).toMatchObject({ disabled: true });
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

  /*
   * The palette's Make a short clip and File ▸ New Clip open the SHORT CLIP
   * DOOR through an intent — the same door the toolbar's segment opens, onto
   * the remembered way and the remembered style — whether or not New image
   * was already open. The old `?output=sequence` deep link landed in Scenes
   * when the door opens onto Simple, and did nothing from inside New image,
   * where the query is consumed only on mount.
   */
  it("opens the Short clip door from the palette while New image is already open", async () => {
    readyLocal();
    installedPayload = [imageModel, videoModel];
    useModelStore().all = [imageModel, videoModel];
    const form = useGenerateFormStore().form;
    form.model = imageModel.name;
    form.family = imageModel.family;
    const draft = useSequenceDraftStore();
    draft.hydrate();
    mountView();
    await flushPromises();

    useUiStore().shortClip();
    await flushPromises();

    // Simple: the clip style in the one-shot output, not Scenes.
    expect(draft.output).toBe("single");
    expect(form.model).toBe(videoModel.name);
    expect(draft.lastStillModel).toBe(imageModel.name);
  });

  it("opens the Short clip door onto Scenes when that is the remembered way", async () => {
    readyLocal();
    installedPayload = [imageModel, videoModel];
    useModelStore().all = [imageModel, videoModel];
    const form = useGenerateFormStore().form;
    form.model = imageModel.name;
    form.family = imageModel.family;
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.clipMode = "scenes";
    // Scenes is the inspector's swap; the shallow mount stubs the inspector,
    // so this stub records what the door handed it.
    const setOutputMode = vi.fn();
    mount(GenerateView, {
      shallow: true,
      attachTo: document.body,
      global: {
        stubs: {
          SequenceComposer: true,
          ComposerCard: true,
          InspectorPanel: {
            name: "InspectorPanel",
            template: "<div />",
            setup: () => ({ setOutputMode }),
          },
        },
      },
    });
    await flushPromises();

    useUiStore().shortClip();
    await flushPromises();
    expect(setOutputMode).toHaveBeenCalledWith("sequence");
    expect(form.model).toBe(imageModel.name);
  });

  it("opens the Short clip door onto Scenes when raised from another workspace, before the inspector exists", async () => {
    // The palette raises the intent and THEN navigates here, so the view
    // consumes it during setup, when the inspector ref is still null. The
    // swap has to wait for the inspector rather than fall on the floor.
    readyLocal();
    installedPayload = [imageModel, videoModel];
    useModelStore().all = [imageModel, videoModel];
    const form = useGenerateFormStore().form;
    form.model = imageModel.name;
    form.family = imageModel.family;
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.clipMode = "scenes";
    const setOutputMode = vi.fn();
    useUiStore().shortClip();
    mount(GenerateView, {
      shallow: true,
      attachTo: document.body,
      global: {
        stubs: {
          SequenceComposer: true,
          ComposerCard: true,
          InspectorPanel: {
            name: "InspectorPanel",
            template: "<div />",
            setup: () => ({ setOutputMode }),
          },
        },
      },
    });
    await flushPromises();
    expect(setOutputMode).toHaveBeenCalledWith("sequence");
  });

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

  it("keeps the composer on screen with the timeline above it", async () => {
    const { wrapper } = await clipMode();

    expect(wrapper.find("sequence-composer-stub").exists()).toBe(true);
    expect(wrapper.find("composer-card-stub").exists()).toBe(true);
  });

  it("writes the composer's words onto the selected scene", async () => {
    const { wrapper, draft } = await clipMode();
    draft.activeClipId = draft.clips[1]!.id;
    await flushPromises();

    const composer = wrapper.findComponent({ name: "ComposerCard" });
    expect(composer.props("promptValue")).toBe("");
    expect(composer.props("placeholder")).toContain("Scene 2");
    // A chain has no batch, so Make is hidden outright rather than showing a
    // count nothing reads.
    expect(composer.props("showCount")).toBe(false);

    composer.vm.$emit("update:promptValue", "the rain picks up");
    await flushPromises();
    expect(draft.clips[1]!.prompt).toBe("the rain picks up");
  });

  it("submits the clip from the composer's Generate", async () => {
    apiFetchTo.mockResolvedValue(Response.json({ job_id: "composed-clip" }));
    const { wrapper, draft } = await clipMode();
    draft.clips[0]!.prompt = "a paper boat sets off";
    draft.clips[1]!.prompt = "the rain picks up";
    await flushPromises();

    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();

    expect(
      apiFetchTo.mock.calls.filter(
        ([, path, init]) =>
          path === "/api/chain-jobs" && (init as RequestInit | undefined)?.method === "POST",
      ),
    ).toHaveLength(1);
  });

  it("keeps the composer alone for one-shot output", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const wrapper = mountView();
    await flushPromises();
    expect(wrapper.find("composer-card-stub").exists()).toBe(true);
    expect(wrapper.find("sequence-composer-stub").exists()).toBe(false);
  });

  it("detaches an amend session before switching its model authority", async () => {
    readyLocal();
    const wanModel = {
      ...videoModel,
      name: "wan22-i2v-a14b:q5",
      family: "wan",
      default_frames: 53,
    } as ModelEntry;
    installedPayload = [videoModel, wanModel];
    useModelStore().all = [videoModel, wanModel];
    const formStore = useGenerateFormStore();
    formStore.form.model = videoModel.name;
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(97);
    draft.loadFromJob(
      {
        jobId: "job-model-a",
        hostId: "local",
        baseline: draft.clips.map((clip) => ({ ...clip })),
        completedStages: 0,
      },
      draft.clips.map((clip) => ({ ...clip })),
      false,
    );

    mountView();
    await flushPromises();
    formStore.form.model = wanModel.name;
    await flushPromises();

    expect(draft.editing).toBeNull();
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
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
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
      draft.openingImage,
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
    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();

    expect(amendCalls.length).toBe(1);
    const body = JSON.parse((amendCalls[0] as { body: string }).body);
    expect(body.enable_audio).toBe(false);
    expect(body.strength).toBe(0.75);
    expect(body.stages[0].source_image).toBe("QUJD");
    expect(body.stages[1].source_image).toBeUndefined();
  });

  /**
   * Amending a clip that already exists is not a new print, and the button
   * that does it said "Generate" — the one word that promises a new one.
   */
  it("names the amend by what it does to the clip already made", async () => {
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
    const composer = wrapper.findComponent({ name: "ComposerCard" });
    expect(composer.props("buttonLabel")).toBe("Generate");

    draft.loadFromJob(
      {
        jobId: "job-1",
        hostId: "local",
        baseline: draft.clips.map((clip) => ({ ...clip })),
        completedStages: 1,
      },
      draft.clips.map((clip) => ({ ...clip })),
      false,
    );
    await flushPromises();

    expect(composer.props("buttonLabel")).toBe("Update clip");

    // Parking the edit behind Simple keeps the session, but Simple's Generate
    // makes a NEW print, so the button must not promise an amendment there.
    draft.setOutput("single", { getPrompt: () => "", setPrompt: () => undefined }, 25);
    await flushPromises();
    expect(draft.editing).not.toBeNull();
    expect(composer.props("buttonLabel")).toBe("Generate");
  });

  it("compensates a cancelled in-flight amendment on its frozen target", async () => {
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
    draft.loadFromJob(
      {
        jobId: "job-1",
        hostId: "local",
        baseline: draft.clips.map((clip) => ({ ...clip })),
        completedStages: 2,
      },
      draft.clips.map((clip) => ({ ...clip })),
      false,
    );
    let finishAmend!: (value: Record<string, unknown>) => void;
    apiJsonTo.mockImplementation((_target: unknown, path: unknown) => {
      if (typeof path === "string" && path.endsWith("/amend")) {
        return new Promise((resolve) => (finishAmend = resolve));
      }
      if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
      if (path === "/api/models") return Promise.resolve(installedPayload);
      return Promise.resolve({});
    });
    apiFetchTo.mockResolvedValue({});
    const wrapper = mountView();
    await flushPromises();
    const composer = wrapper.findComponent({ name: "ComposerCard" });

    composer.vm.$emit("generate");
    await vi.waitFor(() => expect(finishAmend).toBeTypeOf("function"));
    const amendCall = apiJsonTo.mock.calls.find(
      (call) => typeof call[1] === "string" && call[1].endsWith("/amend"),
    );
    const operationId = new Headers((amendCall?.[2] as RequestInit).headers).get(
      "x-mold-operation-id",
    );
    expect(operationId).toMatch(/^[0-9a-f-]{36}$/);
    composer.vm.$emit("cancel");
    await vi.waitFor(() =>
      expect(apiFetchTo).toHaveBeenCalledWith(
        { baseUrl: "http://127.0.0.1:7680", apiKey: "k" },
        `/api/chain-jobs/job-1/operations/${operationId}/cancel`,
        { method: "POST", keepalive: true },
      ),
    );
    finishAmend({ preserved_stages: 0 });
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://127.0.0.1:7680", apiKey: "k" },
      "/api/chain-jobs/job-1/cancel",
      { method: "POST" },
    );
    expect(draft.editing).toMatchObject({ jobId: "job-1" });
  });

  it("fits the opening image before a sequence request is submitted", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const formStore = useGenerateFormStore();
    formStore.form.model = "ltx-video";
    formStore.form.sourceFit = { mode: "lanczos-resize" };
    formStore.form.strength = 0.55;
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "opening";
    draft.clips[1]!.prompt = "landing";
    draft.openingImage = { filename: "opening.png", base64: "ORIGINAL" };
    applySourceFitPreprocess.mockResolvedValue({
      source: "FITTED",
      mask: null,
      changed: true,
    });
    const create = vi.spyOn(useChainJobsStore(), "create").mockResolvedValue("job-1");

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();

    expect(applySourceFitPreprocess).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "ORIGINAL",
        mask: null,
        policy: { mode: "lanczos-resize" },
      }),
      expect.any(Object),
    );
    expect(create.mock.calls[0]?.[1]).toMatchObject({
      strength: 0.55,
      stages: [{ source_image: "FITTED" }, expect.any(Object)],
    });
  });

  it("files the STITCHED print: title, tags, and collection ride the create body", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const formStore = useGenerateFormStore();
    formStore.form.model = "ltx-video";
    formStore.form.title = "Smurf Village";
    formStore.form.fileUnderAutoTag = true;
    formStore.form.fileUnder = addTag(formStore.form.fileUnder, "blue");
    formStore.form.fileUnderMatch = { name: "River studies", slug: "river-studies" };
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "opening";
    draft.clips[1]!.prompt = "landing";
    const create = vi.spyOn(useChainJobsStore(), "create").mockResolvedValue("job-1");

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();

    expect(create.mock.calls[0]?.[1]).toMatchObject({
      title: "Smurf Village",
      tags: ["smurf-village", "blue"],
    });
    // The title match only wins when the title's own slug names it; here the
    // user's typed title slugs to `smurf-village`, so nothing is filed.
    expect(create.mock.calls[0]?.[1].collection).toBeUndefined();
  });

  it("leaves the create body unfiled for an untitled, untagged sequence", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.model = "ltx-video";
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "opening";
    draft.clips[1]!.prompt = "landing";
    const create = vi.spyOn(useChainJobsStore(), "create").mockResolvedValue("job-1");

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();

    const body = create.mock.calls[0]?.[1];
    expect(body?.title).toBeUndefined();
    expect(body?.tags).toBeUndefined();
    expect(body?.collection).toBeUndefined();
  });

  it("parks Sequence images instead of sending them to an unsupported checkpoint", async () => {
    readyLocal();
    const t2vModel = {
      ...videoModel,
      name: "wan22-t2v-a14b:q4",
      family: "wan",
      source_image: "unsupported",
    } as ModelEntry;
    installedPayload = [t2vModel];
    useModelStore().all = [t2vModel];
    const formStore = useGenerateFormStore();
    formStore.form.model = t2vModel.name;
    formStore.form.family = t2vModel.family;
    formStore.form.sourceImageCapability = "unsupported";
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "opening";
    draft.clips[1]!.prompt = "landing";
    draft.openingImage = { filename: "opening.png", base64: "OPENING" };
    draft.clips[1]!.sourceImage = { filename: "second.png", base64: "SECOND" };
    const create = vi.spyOn(useChainJobsStore(), "create").mockResolvedValue("job-1");

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();

    expect(applySourceFitPreprocess).not.toHaveBeenCalled();
    expect(create).toHaveBeenCalledTimes(1);
    expect(create.mock.calls[0]?.[1].stages).toEqual([
      expect.not.objectContaining({ source_image: expect.anything() }),
      expect.not.objectContaining({ source_image: expect.anything() }),
    ]);
    expect(draft.openingImage?.base64).toBe("OPENING");
    expect(draft.clips[1]!.sourceImage?.base64).toBe("SECOND");
  });

  it("aborts when the sequence route changes during source preprocessing", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const formStore = useGenerateFormStore();
    formStore.form.model = "ltx-video";
    formStore.form.sourceFit = { mode: "lanczos-resize" };
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "opening";
    draft.clips[1]!.prompt = "landing";
    draft.openingImage = { filename: "opening.png", base64: "ORIGINAL" };
    let finishPreprocessing!: (value: { source: string; mask: null; changed: boolean }) => void;
    applySourceFitPreprocess.mockReturnValue(
      new Promise((resolve) => {
        finishPreprocessing = resolve;
      }),
    );
    const create = vi.spyOn(useChainJobsStore(), "create").mockResolvedValue("job-1");

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();
    useConnectionStore().info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:8765",
      apiKey: "replacement-key",
    };
    finishPreprocessing({ source: "FITTED", mask: null, changed: true });
    await flushPromises();

    expect(create).not.toHaveBeenCalled();
    expect(useToastStore().items.at(-1)?.message).toContain(
      "machine changed during source preparation",
    );
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

/**
 * The one composer is the ONLY Generate button in clip mode, so what the
 * timeline refuses has to reach it — and so does what the view refuses on the
 * timeline's behalf. With no installed style that can chain, the timeline is
 * `v-else`'d away entirely: nothing emits, and Generate used to sit live over
 * an empty bench and fail with a toast after the click.
 */
describe("GenerateView — what locks the one composer in clip mode", () => {
  /** The real timeline, so its own refusal is proved on both sides of the seam. */
  function mountWithTimeline() {
    return mount(GenerateView, {
      shallow: true,
      attachTo: document.body,
      global: { stubs: { SequenceComposer: false, ComposerCard: true } },
    });
  }

  function composerProps(wrapper: ReturnType<typeof mountWithTimeline>) {
    const card = wrapper.findComponent({ name: "ComposerCard" });
    return {
      disabled: card.props("disabled") as boolean,
      reason: card.props("disabledReason") as string | null,
    };
  }

  async function clipDraft(installed: ModelEntry[]) {
    readyLocal();
    installedPayload = installed;
    useModelStore().all = installed;
    const form = useGenerateFormStore().form;
    form.model = installed[0]?.name ?? "";
    form.family = installed[0]?.family ?? "";
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    return draft;
  }

  it("refuses with no style that can make a clip, where the timeline never mounts", async () => {
    await clipDraft([imageModel]);
    const wrapper = mountWithTimeline();
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-empty']").exists()).toBe(true);
    expect(composerProps(wrapper)).toEqual({
      disabled: true,
      reason: "Pick a video style first.",
    });
  });

  it("carries the timeline's own refusal while a scene is blank", async () => {
    const draft = await clipDraft([videoModel]);
    draft.clips[0]!.prompt = "a kite over the harbour";
    draft.clips[1]!.prompt = "";
    const wrapper = mountWithTimeline();
    await flushPromises();
    expect(composerProps(wrapper)).toEqual({
      disabled: true,
      reason: "Describe scene 2 before generating.",
    });
  });

  it("unlocks once every scene has words", async () => {
    const draft = await clipDraft([videoModel]);
    draft.clips[0]!.prompt = "a kite over the harbour";
    draft.clips[1]!.prompt = "the kite falls into the water";
    const wrapper = mountWithTimeline();
    await flushPromises();
    expect(composerProps(wrapper)).toEqual({ disabled: false, reason: null });
  });

  /** Losing the style unmounts the timeline mid-draft; the last thing it
   *  emitted was `null`, and that stale value used to stand. */
  it("locks again when the style that could chain goes away", async () => {
    const draft = await clipDraft([videoModel]);
    draft.clips[0]!.prompt = "a kite over the harbour";
    draft.clips[1]!.prompt = "the kite falls into the water";
    const wrapper = mountWithTimeline();
    await flushPromises();
    expect(composerProps(wrapper).disabled).toBe(false);

    useModelStore().all = [imageModel];
    useHostModelsStore().byHost.local = {
      entries: [imageModel],
      fetchedAt: Date.now(),
      error: null,
    };
    await flushPromises();

    expect(wrapper.find("[data-test='sequence-empty']").exists()).toBe(true);
    expect(composerProps(wrapper)).toEqual({
      disabled: true,
      reason: "Pick a video style first.",
    });
  });
});

/**
 * The Recent tab's door says "Use these settings again", so it is the same
 * path the Lightbox takes — including the branch that matters most: a stitched
 * clip restores as a CLIP. Routing it through the metadata prefill forced the
 * output back to `single` and kept one scene's words as the whole prompt.
 */
describe("GenerateView — Recent restores what the print actually was", () => {
  function sequenceMetadata(): OutputMetadata {
    return {
      model: videoModel.name,
      prompt: "a kite over the harbour\nthe kite falls into the water",
      output_mode: "sequence",
      seed: 4242,
      steps: 25,
      guidance: 3,
      width: 1024,
      height: 576,
      chain: {
        stage_count: 2,
        motion_tail_frames: 8,
        stages: [
          { prompt: "a kite over the harbour", frames: 25, transition: "smooth" as const },
          { prompt: "the kite falls into the water", frames: 25, transition: "smooth" as const },
        ],
      },
    } as OutputMetadata;
  }

  function recentPrint(metadata: OutputMetadata) {
    return {
      sourceKey: "local",
      hostLabel: "This device",
      item: { filename: "harbour.mp4", metadata, timestamp: 1, size_bytes: 10 },
    } as never;
  }

  it("hands a stitched clip to the sequence path, not to a metadata prefill", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const wrapper = mountView();
    await flushPromises();
    const composer = useComposerStore();

    wrapper
      .findComponent({ name: "InspectorPanel" })
      .vm.$emit("reuse-print", recentPrint(sequenceMetadata()));
    await flushPromises();

    expect(useSequenceDraftStore().output).toBe("sequence");
    expect(useSequenceDraftStore().clips.map((clip) => clip.prompt)).toEqual([
      "a kite over the harbour",
      "the kite falls into the water",
    ]);
    expect(composer.prefill).toBeNull();
  });

  it("still restores a still print through the metadata prefill", async () => {
    readyLocal();
    installedPayload = [imageModel];
    useModelStore().all = [imageModel];
    const wrapper = mountView();
    await flushPromises();

    const metadata = { model: imageModel.name, prompt: "a brass teapot" } as OutputMetadata;
    wrapper
      .findComponent({ name: "InspectorPanel" })
      .vm.$emit("reuse-print", recentPrint(metadata));
    await flushPromises();

    expect(useSequenceDraftStore().output).toBe("single");
    expect(useGenerateFormStore().form.prompt).toBe("a brass teapot");
  });
});

/**
 * `ModalPanel` is `position: absolute; inset: 0`, and the bench strip declares
 * `container-type: size` — which implies `contain: layout` and makes it the
 * containing block for every absolutely positioned descendant. A confirm
 * rendered inside the timeline was therefore centred in a 320px strip, inside
 * an `overflow-hidden` box that clipped a longer message. The timeline emits
 * what it needs asked; the workbench renders it as a sibling of the bench.
 */
describe("GenerateView — the timeline's dialogs belong to the workbench", () => {
  it("renders the clear confirm outside the bench strip", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.model = videoModel.name;
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    const wrapper = mount(GenerateView, {
      shallow: true,
      attachTo: document.body,
      global: { stubs: { SequenceComposer: false, ConfirmDialog: false, ModalPanel: false } },
    });
    await flushPromises();

    await wrapper.get("[data-test='sequence-clear']").trigger("click");
    await flushPromises();

    const dialog = wrapper.get("[data-test='confirm-dialog']");
    expect(dialog.text()).toContain("Clear the clip?");
    expect(dialog.element.closest("[data-test='create-bottom-panel']")).toBeNull();
    expect(dialog.element.closest("[data-test='generate-workbench']")).not.toBeNull();
  });
});

/**
 * The palette and the native menu raise a Create intent and then navigate, so
 * New image is MOUNTING when the tick lands. A plain watcher registers on the
 * already-incremented value and sees no change — "Generate from these words"
 * run from My images did nothing at all, and ⌥↩ from another workspace
 * navigated to Create and made no variations.
 */
describe("GenerateView — an intent raised on the way here", () => {
  it("generates once for a tick raised before this view mounted", async () => {
    readyLocal();
    installedPayload = [imageModel];
    useModelStore().all = [imageModel];
    const ui = useUiStore();
    ui.generate();

    const form = useGenerateFormStore().form;
    form.model = imageModel.name;
    form.family = imageModel.family;
    form.prompt = "a brass teapot";
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    mountView();
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
  });

  it("acts on the intent exactly once, however many views see it", async () => {
    readyLocal();
    installedPayload = [imageModel];
    useModelStore().all = [imageModel];
    const form = useGenerateFormStore().form;
    form.model = imageModel.name;
    form.family = imageModel.family;
    form.prompt = "a brass teapot";
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    useUiStore().generate();
    const first = mountView();
    await flushPromises();
    expect(submit).toHaveBeenCalledTimes(1);

    // Leaving Create and coming back must not replay it.
    first.unmount();
    mountView();
    await flushPromises();
    expect(submit).toHaveBeenCalledTimes(1);
  });
});

/*
 * Simple | Scenes — the two ways of making a clip. Simple is the plain
 * one-shot render (a prompt, a clip style, a length) and is the default;
 * Scenes is the authored sequence. Switching either way destroys nothing.
 */
describe("GenerateView — Simple | Scenes", () => {
  /** A clip style selected with the output still one shot: the Simple mode. */
  async function simpleClip() {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const form = useGenerateFormStore().form;
    form.model = videoModel.name;
    form.family = videoModel.family;
    form.prompt = "a kingfisher drops off the branch";
    form.frames = 97;
    form.fps = 24;
    const draft = useSequenceDraftStore();
    draft.hydrate();
    const wrapper = mountView();
    await flushPromises();
    return { wrapper, draft, form };
  }

  function chainPosts() {
    return apiFetchTo.mock.calls.filter(
      ([, path, init]) =>
        path === "/api/chain-jobs" && (init as RequestInit | undefined)?.method === "POST",
    );
  }

  it("submits Simple as one ordinary print carrying the clip's length", async () => {
    const { wrapper, form } = await simpleClip();
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });
    apiFetchTo.mockClear();

    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    const request = submit.mock.calls[0]![0];
    expect(request.model).toBe(videoModel.name);
    expect(request.prompt).toBe(form.prompt);
    expect(request.frames).toBe(97);
    // No chain job: the plain render is one call to the ordinary door.
    expect(chainPosts()).toHaveLength(0);
  });

  it("submits Scenes as a durable chain job, never an ordinary print", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.model = videoModel.name;
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "the gate opens";
    draft.clips[1]!.prompt = "the road bends away";
    apiFetchTo.mockResolvedValue(Response.json({ job_id: "scenes-job" }));
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    const wrapper = mountView();
    await flushPromises();
    apiFetchTo.mockClear();
    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("generate");
    await flushPromises();

    expect(chainPosts()).toHaveLength(1);
    expect(submit).not.toHaveBeenCalled();
  });

  it("offers the Length chip in Simple and hides Make in Scenes", async () => {
    const { wrapper, draft } = await simpleClip();
    let composer = wrapper.findComponent({ name: "ComposerCard" });
    expect(composer.props("lengthContract")).not.toBeNull();
    expect(composer.props("lengthFrames")).toBe(97);
    expect(composer.props("showCount")).toBe(true);
    expect(composer.props("placeholder")).toBe("Describe the clip");

    draft.output = "sequence";
    draft.ensureClips(25);
    await flushPromises();
    composer = wrapper.findComponent({ name: "ComposerCard" });
    // A sequence's lengths are per-scene, and a chain has no batch at all.
    expect(composer.props("lengthContract")).toBeNull();
    expect(composer.props("showCount")).toBe(false);
  });

  it("writes the Length chip through to the same field the inspector writes", async () => {
    const { wrapper, form } = await simpleClip();
    wrapper.findComponent({ name: "ComposerCard" }).vm.$emit("update:lengthFrames", 121);
    await flushPromises();
    expect(form.frames).toBe(121);
  });

  it("seeds scene 1 from the words and the length already on the composer", async () => {
    const { wrapper, draft, form } = await simpleClip();

    wrapper.findComponent({ name: "ClipModeStrip" }).vm.$emit("set-clip-mode", "scenes");
    await flushPromises();

    expect(draft.output).toBe("sequence");
    expect(draft.clipMode).toBe("scenes");
    expect(draft.clips[0]?.prompt).toBe(form.prompt);
    // The seeded length is a real scene length on the family's grid.
    expect((draft.clips[0]?.frames ?? 0) % 8).toBe(1);
    expect(draft.clips[0]?.frames).toBeLessThanOrEqual(97);
    // The one-shot's own words are untouched: the two are separate authorities.
    expect(form.prompt).toBe("a kingfisher drops off the branch");
  });

  it("leaves written scenes alone rather than overwriting scene 1", async () => {
    const { wrapper, draft } = await simpleClip();
    draft.ensureClips(97);
    draft.clips[0]!.prompt = "the gate opens";

    wrapper.findComponent({ name: "ClipModeStrip" }).vm.$emit("set-clip-mode", "scenes");
    await flushPromises();

    expect(draft.clips[0]?.prompt).toBe("the gate opens");
  });

  it("parks the scenes when the switch goes back to Simple", async () => {
    const { wrapper, draft } = await simpleClip();
    draft.output = "sequence";
    draft.ensureClips(97);
    draft.clips[0]!.prompt = "the gate opens";
    draft.clips[1]!.prompt = "the road bends away";
    await flushPromises();

    wrapper.findComponent({ name: "ClipModeStrip" }).vm.$emit("set-clip-mode", "simple");
    await flushPromises();

    expect(draft.output).toBe("single");
    expect(draft.clipMode).toBe("simple");
    // Nothing destroyed — the timeline's own Clear the clip is the only eraser.
    expect(draft.clips.map((clip) => clip.prompt)).toEqual([
      "the gate opens",
      "the road bends away",
    ]);
    // And the clip style stays: Simple is still a clip.
    expect(useGenerateFormStore().form.model).toBe(videoModel.name);
  });

  it("takes the palette's own door into Scenes", async () => {
    const { draft } = await simpleClip();
    useUiStore().clipScenes();
    await flushPromises();
    expect(draft.output).toBe("sequence");
    expect(draft.clipMode).toBe("scenes");
  });
});
