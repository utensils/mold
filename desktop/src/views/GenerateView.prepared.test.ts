import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { nextTick } from "vue";
import GenerateView from "./GenerateView.vue";
import ExpandControl from "../components/generate/ExpandControl.vue";
import PreparedExpansionBatch from "../components/generate/PreparedExpansionBatch.vue";
import ExpansionPullStatus from "../components/generate/ExpansionPullStatus.vue";
import MissingModelDialog from "../components/generate/MissingModelDialog.vue";
import { useConnectionStore } from "../stores/connection";
import { useGenerateFormStore } from "../stores/generateForm";
import { newJob, useGenerationStore } from "../stores/generation";
import { useHostsStore, type FeasibleRouteResult } from "../stores/hosts";
import { useHostModelsStore } from "../stores/hostModels";
import { useModelStore } from "../stores/models";
import { useToastStore } from "../stores/toasts";
import { useDownloadsStore } from "../stores/downloads";
import { useUiStore } from "../stores/ui";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useComposerStore } from "../stores/composer";
import { expandPrompt } from "../lib/api/expand";
import { remixPrompt } from "../lib/api/remix";
import { startCatalogDownload } from "../lib/api/catalog";
import { ApiError } from "../lib/api/client";
import { applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import type { ModelEntry } from "../lib/api/types";

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
const placementPreview = vi.hoisted(() => vi.fn());
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {
    constructor(
      message: string,
      public readonly status: number,
      public readonly body: unknown = null,
    ) {
      super(message);
    }
  },
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("@studio/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/client")>()),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));
vi.mock("../lib/api/expand", () => ({ expandPrompt: vi.fn() }));
vi.mock("../lib/api/remix", () => ({ remixPrompt: vi.fn() }));
vi.mock("../lib/api/catalog", () => ({ startCatalogDownload: vi.fn() }));
vi.mock("../lib/sourceFitPreprocess", () => ({ applySourceFitPreprocess: vi.fn() }));
vi.mock("@studio/api/generationPlacement", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationPlacement")>()),
  previewGenerationPlacement: (...args: unknown[]) => placementPreview(...args),
  previewChainPlacement: (...args: unknown[]) => placementPreview(...args),
}));

enableAutoUnmount(afterEach);

const model: ModelEntry = {
  name: "flux-dev:q8",
  family: "flux",
  downloaded: true,
  default_width: 768,
  default_height: 768,
  default_steps: 20,
  default_guidance: 4.5,
} as ModelEntry;

const catalogModel: ModelEntry = {
  ...model,
  name: "cv:1759168",
  family: "sdxl",
  display_name: "Juggernaut XL - Ragnarok",
  description: "Juggernaut XL - Ragnarok by RunDiffusion",
} as ModelEntry;

function mountView() {
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: {
      // The composer textarea + Generate button now live in ComposerCard and
      // the model picker in InspectorPanel → ModelPicker; keep them real so
      // the view's DOM hooks and focus targets resolve. Prepared/pull
      // surfaces stay real too.
      stubs: {
        ExpandControl: false,
        PreparedExpansionBatch: false,
        ExpansionPullStatus: false,
        GenerateErrorNotice: false,
        ErrorNotice: false,
        ComposerCard: false,
        InspectorPanel: false,
        ModelPicker: false,
      },
    },
  });
}

describe("GenerateView prepared expansion batches", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    apiJson.mockReset();
    apiJson.mockImplementation((path: string) =>
      Promise.resolve(path === "/api/models" ? [model] : []),
    );
    apiJsonTo.mockReset();
    apiJsonTo.mockImplementation((_target: unknown, path: string) =>
      Promise.resolve(path === "/api/models" ? [model] : []),
    );
    placementPreview.mockReset().mockResolvedValue({
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
    vi.mocked(expandPrompt).mockReset();
    vi.mocked(remixPrompt).mockReset();
    vi.mocked(startCatalogDownload).mockReset();
    vi.mocked(startCatalogDownload).mockResolvedValue("pull-job");
    vi.mocked(applySourceFitPreprocess).mockReset();
    vi.mocked(applySourceFitPreprocess).mockResolvedValue({
      source: "fitted-source",
      mask: null,
      changed: true,
    });

    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    connection.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [model];
    // A print routes from inventory now, not from a placement preview, so
    // This device must have listed its own models before one can be queued.
    useHostModelsStore().byHost.local = {
      entries: [model],
      fetchedAt: Date.now(),
      error: null,
    };
    const form = useGenerateFormStore().form;
    form.prompt = "a lighthouse at dusk";
    form.model = model.name;
    form.family = model.family;
    form.batchSize = 3;
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("prepares exactly Batch N prompts on one concrete host without generating", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const generation = useGenerationStore();
    const submit = vi.spyOn(generation, "submitBatch");
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith(
      "a lighthouse at dusk",
      { variations: 3, modelFamily: "flux", task: "text-to-image" },
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    );
    expect(submit).not.toHaveBeenCalled();
    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.exists()).toBe(true);
    expect(prepared.props("batch").prompts.map((prompt: { text: string }) => prompt.text)).toEqual([
      "storm light",
      "sea mist",
      "aerial coast",
    ]);
    expect(prepared.props("batch").route.hostId).toBe("local");
  });

  it("restores every selected queue setting and shows its live render", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) =>
      Promise.resolve(
        path === "/api/queue/remote-print/preview"
          ? { image: "UFJFVklFVw==", step: 8, total: 20 }
          : path === "/api/models"
            ? [model]
            : [],
      ),
    );
    useComposerStore().set({
      metadata: {
        version: "1",
        model: model.name,
        prompt: "queue lighthouse",
        title: "Queue study",
        negative_prompt: "fog",
        seed: 42,
        steps: 20,
        guidance: 3.5,
        width: 1024,
        height: 576,
      },
      queueSelection: { hostId: "local", jobId: "remote-print", running: true },
    });

    const wrapper = mountView();
    await flushPromises();

    expect(useGenerateFormStore().form).toMatchObject({
      prompt: "queue lighthouse",
      title: "Queue study",
      negativePrompt: "fog",
      seed: "42",
      steps: 20,
      guidance: 3.5,
      width: 1024,
      height: 576,
    });
    expect(wrapper.get('[data-test="develop-preview"]').attributes("src")).toBe(
      "data:image/png;base64,UFJFVklFVw==",
    );
    expect(wrapper.get('[data-test="generation-live-status"]').text()).toBe("Developing 8/20");
  });

  it("expands in Auto when model owners differ only by patch version", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: "remote-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    hosts.telemetry.local = { queueDepth: 1, queueCapacity: 8, version: "0.23.1" };
    hosts.telemetry["hal9000-7680"] = {
      queueDepth: 0,
      queueCapacity: 8,
      version: "0.23.0",
    };
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = {
      entries: [{ ...model, generation_profile: { profile_hash: "local-profile" } } as ModelEntry],
      fetchedAt: Date.now(),
      error: null,
    };
    hostModels.byHost["hal9000-7680"] = {
      entries: [{ ...model, generation_profile: { profile_hash: "remote-profile" } } as ModelEntry],
      fetchedAt: Date.now(),
      error: null,
    };
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith(
      "a lighthouse at dusk",
      { variations: 3, modelFamily: "flux", task: "text-to-image" },
      { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
    );
    expect(wrapper.findComponent(PreparedExpansionBatch).props("batch").route.hostId).toBe(
      "hal9000-7680",
    );
  });

  it("generates Batch N directly without implicitly expanding the prompt", async () => {
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();

    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    expect(expandPrompt).not.toHaveBeenCalled();
    expect(submit).toHaveBeenCalledTimes(1);
    const [request, batch, , , options] = submit.mock.calls[0]!;
    expect(request).toMatchObject({ prompt: "a lighthouse at dusk" });
    expect(batch).toBe(3);
    expect(options).toEqual({});
  });

  it("retires dormant original-prompt provenance when a new prompt is authored", async () => {
    const form = useGenerateFormStore().form;
    form.originalPrompt = "the source for an earlier generated print";
    const wrapper = mountView();
    await flushPromises();

    await wrapper.get("textarea[aria-label='Prompt']").setValue("a completely new print");

    expect(form.originalPrompt).toBeNull();
    expect(form.prompt).toBe("a completely new print");
  });

  it("preserves quick-expansion provenance when the rewrite is hand-edited", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 1;
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light"],
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    await wrapper.get("textarea[aria-label='Prompt']").setValue("hand-edited storm light");
    await nextTick();

    expect(form.originalPrompt).toBe("a lighthouse at dusk");
    expect(wrapper.find("[data-test='quick-expansion-stale']").exists()).toBe(true);
  });

  it("prepares and preserves exactly eight reviewed prompts", async () => {
    const prompts = Array.from({ length: 8 }, (_, index) => `variation ${index + 1}`);
    useGenerateFormStore().form.batchSize = prompts.length;
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: prompts,
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith(
      "a lighthouse at dusk",
      { variations: 8, modelFamily: "flux", task: "text-to-image" },
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    );
    expect(
      wrapper
        .findComponent(PreparedExpansionBatch)
        .props("batch")
        .prompts.map((prompt: { text: string }) => prompt.text),
    ).toEqual(prompts);
  });

  it("runs exactly one finalized placement preview before submitting edited prepared prompts", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const generation = useGenerationStore();
    const submit = vi.spyOn(generation, "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    const preparedRoute = prepared.props("batch").route;
    const preview = vi
      .spyOn(useHostsStore(), "resolveFeasible")
      .mockResolvedValue({ kind: "route", route: preparedRoute });
    const firstId = prepared.props("batch").prompts[0]!.id;
    prepared.vm.$emit("edit", { id: firstId, text: "edited storm light" });
    await flushPromises();
    prepared.vm.$emit("generate");
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    expect(preview).toHaveBeenCalledTimes(1);
    expect(preview).toHaveBeenCalledWith(
      "local",
      expect.anything(),
      3,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    const [, count, submittedRoute, , options] = submit.mock.calls[0]!;
    expect(count).toBe(3);
    expect(submittedRoute).toMatchObject({
      hostId: "local",
      target: { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    });
    expect(options).toEqual({
      prompts: ["edited storm light", "sea mist", "aerial coast"],
      originalPrompt: "a lighthouse at dusk",
      batchId: expect.any(String),
    });
    expect(document.activeElement).toBe(wrapper.get('textarea[aria-label="Prompt"]').element);
  });

  it("releases accepted work so another prepared batch can queue before the first settles", async () => {
    vi.mocked(expandPrompt)
      .mockResolvedValueOnce({
        original: "a lighthouse at dusk",
        expanded: ["first one", "first two", "first three"],
      })
      .mockResolvedValueOnce({
        original: "a lighthouse at dusk",
        expanded: ["second one", "second two", "second three"],
      });
    const first = deferred<never[]>();
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValueOnce({ jobs: [], settled: first.promise })
      .mockReturnValueOnce({ jobs: [], settled: Promise.resolve([]) });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    wrapper.findComponent(PreparedExpansionBatch).vm.$emit("generate");
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(false);
    expect(wrapper.findComponent(ExpandControl).props("blocked")).toBe(false);

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(true);
    wrapper.findComponent(PreparedExpansionBatch).vm.$emit("generate");
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(2);
    expect(submit.mock.calls[0]![4]).toMatchObject({
      prompts: ["first one", "first two", "first three"],
      originalPrompt: "a lighthouse at dusk",
    });
    expect(submit.mock.calls[1]![4]).toMatchObject({
      prompts: ["second one", "second two", "second three"],
      originalPrompt: "a lighthouse at dusk",
    });
    first.resolve([]);
    await flushPromises();
  });

  it("does not let an older out-of-order completion replace newer recovery authority", async () => {
    const form = useGenerateFormStore().form;
    form.model = "missing-old:q8";
    vi.mocked(expandPrompt)
      .mockResolvedValueOnce({
        original: "a lighthouse at dusk",
        expanded: ["old one", "old two", "old three"],
      })
      .mockResolvedValueOnce({
        original: "a lighthouse at dusk",
        expanded: ["new one", "new two", "new three"],
      });
    const older = deferred<ReturnType<typeof newJob>[]>();
    const newer = deferred<ReturnType<typeof newJob>[]>();
    vi.spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValueOnce({ jobs: [], settled: older.promise })
      .mockReturnValueOnce({ jobs: [], settled: newer.promise });
    const failedJob = (modelName: string) => {
      const job = newJob({
        prompt: "a lighthouse at dusk",
        model: modelName,
        width: 768,
        height: 768,
        steps: 20,
      });
      job.status = "error";
      job.error = `model '${modelName}' not found`;
      return job;
    };
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    wrapper.findComponent(PreparedExpansionBatch).vm.$emit("generate");
    await flushPromises();

    form.model = "missing-new:q8";
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    wrapper.findComponent(PreparedExpansionBatch).vm.$emit("generate");
    await flushPromises();

    newer.resolve([failedJob("missing-new:q8")]);
    await flushPromises();
    expect(wrapper.findComponent(MissingModelDialog).props("model")).toBe("missing-new:q8");

    older.resolve([failedJob("missing-old:q8")]);
    await flushPromises();
    expect(wrapper.findComponent(MissingModelDialog).props("model")).toBe("missing-new:q8");
  });

  it.each([
    {
      label: "local",
      route: {
        hostId: "local",
        label: "This device",
        kind: "local" as const,
        target: { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
        instanceId: null,
      },
    },
    {
      label: "remote",
      route: {
        hostId: "remote-one",
        label: "Studio GPU",
        kind: "remote" as const,
        target: { baseUrl: "http://studio:7680", apiKey: "remote-key" },
        instanceId: "studio-id",
      },
    },
  ])(
    "preserves reviewed prompts and queues nothing when the finalized $label preview rejects",
    async ({ route }) => {
      vi.mocked(expandPrompt).mockResolvedValue({
        original: "a lighthouse at dusk",
        expanded: ["storm light", "sea mist", "aerial coast"],
      });
      const hosts = useHostsStore();
      if (route.kind === "remote") {
        hosts.extras = [
          {
            id: route.hostId,
            label: route.label,
            url: route.target.baseUrl,
            apiKey: route.target.apiKey,
            status: "ready",
            error: null,
            instanceId: route.instanceId,
          },
        ];
      }
      vi.spyOn(hosts, "resolveRoute").mockReturnValue(route);
      const placementFailure: FeasibleRouteResult =
        route.kind === "local"
          ? {
              kind: "infeasible",
              perHost: [
                {
                  kind: "infeasible",
                  hostId: route.hostId,
                  label: route.label,
                  route,
                  reason: "required VAE is not installed",
                  missingComponents: [
                    {
                      kind: "vae",
                      name: "ae.safetensors",
                      present: false,
                      repair_model: "flux-dev:q4",
                    },
                  ],
                  missingModel: null,
                },
              ],
            }
          : {
              kind: "mixed",
              perHost: [
                {
                  kind: "infeasible",
                  hostId: route.hostId,
                  label: route.label,
                  route,
                  reason: "required VAE is not installed",
                  missingComponents: [
                    {
                      kind: "vae",
                      name: "ae.safetensors",
                      present: false,
                      repair_model: "flux-dev:q4",
                    },
                  ],
                  missingModel: null,
                },
                {
                  kind: "unreachable",
                  hostId: "local",
                  label: "This device",
                  error: "placement preview returned HTTP 401",
                },
              ],
            };
      const finalizedPreview = vi
        .spyOn(hosts, "resolveFeasible")
        .mockResolvedValue(placementFailure);
      const submit = vi.spyOn(useGenerationStore(), "submitBatch");
      const wrapper = mountView();
      await flushPromises();

      wrapper.findComponent(ExpandControl).vm.$emit("expand");
      await flushPromises();
      const before = wrapper
        .findComponent(PreparedExpansionBatch)
        .props("batch")
        .prompts.map((prompt: { text: string }) => prompt.text);
      wrapper.findComponent(PreparedExpansionBatch).vm.$emit("generate");
      await flushPromises();

      expect(finalizedPreview).toHaveBeenCalledTimes(1);
      expect(finalizedPreview).toHaveBeenCalledWith(
        route.hostId,
        expect.anything(),
        3,
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
      expect(submit).not.toHaveBeenCalled();
      const preserved = wrapper.findComponent(PreparedExpansionBatch);
      expect(preserved.exists()).toBe(true);
      expect(
        preserved.props("batch").prompts.map((prompt: { text: string }) => prompt.text),
      ).toEqual(before);
      const message = useToastStore().items.at(-1)?.message ?? "";
      expect(message).toContain("required VAE is not installed (missing: ae.safetensors)");
      expect(message).toContain("Nothing was queued");
      if (route.kind === "remote") {
        expect(message).toContain("failed for different reasons");
        expect(message).toContain("This device did not answer");
        expect(message).not.toContain("No selected machine can run this print");
      }
    },
  );

  it("freezes the Batch 1 expansion route through the next Generate", async () => {
    useGenerateFormStore().form.batchSize = 1;
    const hosts = useHostsStore();
    hosts.extras = [
      {
        id: "remote-one",
        label: "Studio GPU",
        url: "http://studio:7680",
        apiKey: "remote-key",
        status: "ready",
        error: null,
        instanceId: "studio-id",
      },
    ];
    const localRoute = {
      hostId: "local",
      label: "This device",
      kind: "local" as const,
      target: { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
      instanceId: null,
      referenceUploads: null,
      heterogeneousBatchMaxOutputs: null,
    };
    const remoteRoute = {
      hostId: "remote-one",
      label: "Studio GPU",
      kind: "remote" as const,
      target: { baseUrl: "http://studio:7680", apiKey: "remote-key" },
    };
    const resolveRoute = vi.spyOn(hosts, "resolveRoute").mockReturnValue(localRoute);
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light"],
    });
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    resolveRoute.mockImplementation((selection) =>
      selection === "local" ? localRoute : remoteRoute,
    );
    useHostsStore().telemetry.local = {
      queueDepth: 10,
      queueCapacity: 10,
      version: null,
    };
    await flushPromises();
    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit.mock.calls[0]![2]).toEqual({
      ...localRoute,
      modelFamily: "flux",
    });
  });

  it("reuses a quick transformed prompt after a host-only selection change", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 1;
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light"],
    });
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    useAppPrefsStore().settings = { generateTargetHost: "capable" } as never;
    await nextTick();
    expect(wrapper.find("[data-test='quick-expansion-stale']").exists()).toBe(false);

    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();
    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit.mock.calls[0]?.[0]).toMatchObject({
      prompt: "storm light",
      original_prompt: "a lighthouse at dusk",
    });
  });

  it("remixes Batch 1 in place with the style snapshotted before the request", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 1;
    form.stylePreset = "cinematic";
    const pending = deferred<Awaited<ReturnType<typeof remixPrompt>>>();
    vi.mocked(remixPrompt).mockReturnValue(pending.promise);
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("remix");
    await nextTick();
    form.stylePreset = "anime";
    pending.resolve({
      source_prompt: "a lighthouse at dusk",
      source_kind: "direct",
      variants: ["one"].map((prompt) => ({
        prompt,
        dimensions: ["camera"],
      })),
    });
    await flushPromises();

    expect(vi.mocked(remixPrompt)).toHaveBeenCalledWith(
      expect.objectContaining({ variations: 1 }),
      expect.anything(),
    );
    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(false);
    expect(form.prompt).toBe("one");
    expect(form.batchSize).toBe(1);
    expect(form.stylePreset).toBe("");

    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();
    expect(submit.mock.calls[0]?.[0]).toMatchObject({
      prompt: "one",
      prompt_transform: {
        operation: "remix",
        source_prompt: "a lighthouse at dusk",
        dimensions: ["camera"],
      },
    });
  });

  it("prepares the explicitly selected number of Remix variants", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 3;
    vi.mocked(remixPrompt).mockResolvedValue({
      source_prompt: "a lighthouse at dusk",
      source_kind: "direct",
      variants: ["one", "two", "three"].map((prompt) => ({
        prompt,
        dimensions: ["camera"],
      })),
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("remix");
    await flushPromises();

    expect(vi.mocked(remixPrompt)).toHaveBeenCalledWith(
      expect.objectContaining({ variations: 3 }),
      expect.anything(),
    );
    expect(wrapper.findComponent(PreparedExpansionBatch).props("batch").prompts).toHaveLength(3);
    expect(form.batchSize).toBe(3);
  });

  it("offers immediate human-readable recovery when a quick expansion changes models", async () => {
    useGenerateFormStore().form.batchSize = 1;
    apiJson.mockResolvedValue([model, catalogModel]);
    apiJsonTo.mockResolvedValue([model, catalogModel]);
    useModelStore().all = [model, catalogModel];
    useHostModelsStore().byHost.local = {
      entries: [model, catalogModel],
      fetchedAt: Date.now(),
      error: null,
    };
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light"],
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    useGenerateFormStore().applyModel(catalogModel);
    await flushPromises();

    const notice = wrapper.get("[data-test='quick-expansion-stale']");
    expect(notice.text()).toContain("Juggernaut XL - Ragnarok");
    expect(notice.text()).not.toContain("cv:1759168");
    expect(notice.find("[data-test='reexpand-and-generate']").exists()).toBe(true);
    expect(notice.find("[data-test='generate-expanded-anyway']").exists()).toBe(true);
    expect(notice.find("[data-test='restore-expanded-original']").exists()).toBe(true);
  });

  it("explicitly generates the visible expanded prompt with the current model and route", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 1;
    apiJson.mockResolvedValue([model, catalogModel]);
    apiJsonTo.mockResolvedValue([model, catalogModel]);
    useModelStore().all = [model, catalogModel];
    useHostModelsStore().byHost.local = {
      entries: [model, catalogModel],
      fetchedAt: Date.now(),
      error: null,
    };
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light"],
    });
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    useGenerateFormStore().applyModel(catalogModel);
    await flushPromises();

    await wrapper.get("[data-test='generate-expanded-anyway']").trigger("click");
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit.mock.calls[0]![0]).toMatchObject({
      model: "cv:1759168",
      prompt: "storm light",
      original_prompt: "a lighthouse at dusk",
    });
    expect(submit.mock.calls[0]![2]).toMatchObject({
      hostId: "local",
      target: { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    });
  });

  it("re-expands the original prompt for the current model before generating", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 1;
    apiJson.mockResolvedValue([model, catalogModel]);
    apiJsonTo.mockResolvedValue([model, catalogModel]);
    useModelStore().all = [model, catalogModel];
    useHostModelsStore().byHost.local = {
      entries: [model, catalogModel],
      fetchedAt: Date.now(),
      error: null,
    };
    vi.mocked(expandPrompt)
      .mockResolvedValueOnce({
        original: "a lighthouse at dusk",
        expanded: ["storm light"],
      })
      .mockResolvedValueOnce({
        original: "a lighthouse at dusk",
        expanded: ["cinematic lighthouse for SDXL"],
      });
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    useGenerateFormStore().applyModel(catalogModel);
    await flushPromises();

    await wrapper.get("[data-test='reexpand-and-generate']").trigger("click");
    await flushPromises();

    expect(expandPrompt).toHaveBeenNthCalledWith(
      2,
      "a lighthouse at dusk",
      { variations: 1, modelFamily: "sdxl", task: "text-to-image" },
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    );
    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit.mock.calls[0]![0]).toMatchObject({
      model: "cv:1759168",
      prompt: "cinematic lighthouse for SDXL",
      original_prompt: "a lighthouse at dusk",
    });
  });

  it("shows a human-readable catalog model in generation chrome", async () => {
    apiJson.mockResolvedValue([catalogModel]);
    apiJsonTo.mockResolvedValue([catalogModel]);
    useModelStore().all = [catalogModel];
    const job = newJob({
      model: catalogModel.name,
      prompt: "a camera",
      width: 1024,
      height: 1024,
      steps: 25,
      guidance: 7,
      batch_size: 1,
      output_format: "png",
    });
    job.status = "denoising";
    job.step = 20;
    job.total = 25;
    useGenerationStore().jobs = [job];

    const wrapper = mountView();
    await flushPromises();

    const chrome = wrapper.get("[data-test='generation-edge-code']");
    expect(chrome.text()).toContain("Juggernaut XL - Ragnarok");
    expect(chrome.text()).not.toContain("CV·1759168");
    expect(chrome.text()).not.toContain("cv:1759168");
  });

  it("keeps the live generation status below the developing image", async () => {
    const job = newJob({
      model: catalogModel.name,
      prompt: "a camera",
      width: 1024,
      height: 1024,
      steps: 25,
      guidance: 7,
      batch_size: 1,
      output_format: "png",
    });
    job.status = "denoising";
    job.step = 20;
    job.total = 25;
    job.previewUrl = "blob:latent-preview";
    useGenerationStore().jobs = [job];

    const wrapper = mountView();
    await flushPromises();

    const frame = wrapper.get("[data-test='preview-frame']");
    const status = wrapper.get("[data-test='generation-live-status']");
    expect(status.text()).toBe("Developing 20/25");
    expect(frame.find("[data-test='generation-live-status']").exists()).toBe(false);
    expect(
      frame.element.compareDocumentPosition(status.element) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("cancels a deferred Batch 1 submission when the expansion is restored", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 1;
    form.sourceImage = "source-bytes";
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light"],
    });
    let resolvePreprocess!: (value: { source: string; mask: null; changed: boolean }) => void;
    vi.mocked(applySourceFitPreprocess).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePreprocess = resolve;
        }),
    );
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("restore");
    resolvePreprocess({ source: "fitted-source", mask: null, changed: true });
    await flushPromises();

    expect(submit).not.toHaveBeenCalled();
    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.originalPrompt).toBeNull();
  });

  it("cancels an older deferred Batch 1 submission without clearing a newer expansion", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 1;
    form.sourceImage = "source-bytes";
    vi.mocked(expandPrompt)
      .mockResolvedValueOnce({
        original: "a lighthouse at dusk",
        expanded: ["storm light"],
      })
      .mockResolvedValueOnce({
        original: "storm light",
        expanded: ["storm light over silver water"],
      });
    let resolvePreprocess!: (value: { source: string; mask: null; changed: boolean }) => void;
    vi.mocked(applySourceFitPreprocess).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePreprocess = resolve;
        }),
    );
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    expect(form.prompt).toBe("storm light over silver water");
    resolvePreprocess({ source: "fitted-source", mask: null, changed: true });
    await flushPromises();

    expect(submit).not.toHaveBeenCalled();
    expect(form.prompt).toBe("storm light over silver water");
    wrapper.findComponent(ExpandControl).vm.$emit("restore");
    await flushPromises();
    expect(form.prompt).toBe("storm light");
  });

  it("does not let Batch 1 quick Expand erase a preserved prepared set", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    useGenerateFormStore().form.batchSize = 1;
    await flushPromises();
    const control = wrapper.findComponent(ExpandControl);
    expect(control.props("blocked")).toBe(true);
    control.vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledTimes(1);
    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(true);
  });

  it("focuses the composer when an explicit refresh replaces a stale batch with Batch 1", async () => {
    vi.mocked(expandPrompt).mockResolvedValueOnce({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    useGenerateFormStore().form.batchSize = 1;
    vi.mocked(expandPrompt).mockResolvedValueOnce({
      original: "a lighthouse at dusk",
      expanded: ["one reviewed lighthouse"],
    });
    await flushPromises();
    wrapper.findComponent(PreparedExpansionBatch).vm.$emit("refresh");
    await flushPromises();

    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(false);
    expect(document.activeElement).toBe(wrapper.get('textarea[aria-label="Prompt"]').element);
  });

  it("does not steal focus after a delayed Batch 1 refresh replacement", async () => {
    vi.mocked(expandPrompt).mockResolvedValueOnce({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    useGenerateFormStore().form.batchSize = 1;
    let resolveRefresh!: (value: { original: string; expanded: string[] }) => void;
    vi.mocked(expandPrompt).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveRefresh = resolve;
        }),
    );
    await flushPromises();
    const refresh = wrapper
      .findComponent(PreparedExpansionBatch)
      .get('[data-test="refresh-prepared"]');
    await refresh.trigger("click");
    const modelButton = wrapper.get('[data-test="selected-model-name"]').element.closest("button")!;
    modelButton.focus();
    resolveRefresh({
      original: "a lighthouse at dusk",
      expanded: ["one reviewed lighthouse"],
    });
    await flushPromises();

    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(false);
    expect(document.activeElement).toBe(modelButton);
  });

  it("cancels a prepared submit discarded during async source preprocessing", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    let resolvePreprocess!: (value: { source: string; mask: null; changed: boolean }) => void;
    vi.mocked(applySourceFitPreprocess).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePreprocess = resolve;
        }),
    );
    const form = useGenerateFormStore().form;
    form.sourceImage = "source-bytes";
    const submit = vi.spyOn(useGenerationStore(), "submitBatch");
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    let prepared = wrapper.findComponent(PreparedExpansionBatch);
    prepared.vm.$emit("generate");
    await flushPromises();
    prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.props("submitting")).toBe(true);
    prepared.vm.$emit("discard");
    resolvePreprocess({ source: "fitted-source", mask: null, changed: true });
    await flushPromises();

    expect(submit).not.toHaveBeenCalled();
    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(false);
  });

  it("revalidates upstream prepared inputs after async source preprocessing", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    let resolvePreprocess!: (value: { source: string; mask: null; changed: boolean }) => void;
    vi.mocked(applySourceFitPreprocess).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePreprocess = resolve;
        }),
    );
    const form = useGenerateFormStore().form;
    form.sourceImage = "source-bytes";
    const submit = vi.spyOn(useGenerationStore(), "submitBatch");
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    wrapper.findComponent(PreparedExpansionBatch).vm.$emit("generate");
    await flushPromises();
    form.prompt = "a lighthouse at dawn";
    resolvePreprocess({ source: "fitted-source", mask: null, changed: true });
    await flushPromises();

    expect(submit).not.toHaveBeenCalled();
    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.exists()).toBe(true);
    expect(prepared.props("staleReasons")).toContain(
      "Source prompt changed after these variations were prepared.",
    );
    expect(wrapper.text()).toContain("Nothing was queued");
  });

  it("preserves reviewed prompts and blocks submission when the source changes", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const generation = useGenerationStore();
    const submit = vi.spyOn(generation, "submitBatch");
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    useGenerateFormStore().form.prompt = "a lighthouse at dawn";
    await flushPromises();
    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.props("staleReasons")).toContain(
      "Source prompt changed after these variations were prepared.",
    );
    expect(prepared.props("batch").prompts).toHaveLength(3);

    prepared.vm.$emit("generate");
    await flushPromises();
    expect(submit).not.toHaveBeenCalled();
  });

  it("rejects a late response after the composer is cleared", async () => {
    let resolveExpansion!: (value: { original: string; expanded: string[] }) => void;
    vi.mocked(expandPrompt).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveExpansion = resolve;
        }),
    );
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    useUiStore().newGenerationTick += 1;
    await flushPromises();
    resolveExpansion({
      original: "a lighthouse at dusk",
      expanded: ["late one", "late two", "late three"],
    });
    await flushPromises();

    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(false);
    expect(useGenerateFormStore().form.prompt).toBe("");
  });

  it("keeps the requested count when the host returns a malformed batch", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["only one", "only two"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(useGenerateFormStore().form.batchSize).toBe(3);
    expect(wrapper.findComponent(PreparedExpansionBatch).exists()).toBe(false);
    expect(wrapper.get('[role="alert"]').text()).toContain(
      "Expected exactly 3 non-empty prompts, but the host returned 2.",
    );
  });

  it("offers a missing expansion-model pull on the exact resolved host", async () => {
    const subscribe = vi.spyOn(useDownloadsStore(), "subscribe").mockResolvedValue();
    vi.mocked(expandPrompt).mockRejectedValue(
      new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
    );
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(wrapper.get('[data-test="expansion-pull-status"]').text()).toContain(
      "qwen3-expand:q8 isn't installed on This device",
    );
    wrapper.findComponent(ExpansionPullStatus).vm.$emit("pull");
    await flushPromises();
    expect(subscribe).toHaveBeenCalled();
    expect(startCatalogDownload).toHaveBeenCalledWith(
      "qwen3-expand:q8",
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
      false,
    );
  });

  it("keeps Batch 1 pull progress inline from connection through exact-job readiness", async () => {
    useGenerateFormStore().form.batchSize = 1;
    const stream = deferred<void>();
    const start = deferred<string | null>();
    vi.spyOn(useDownloadsStore(), "subscribe").mockReturnValue(stream.promise);
    vi.mocked(startCatalogDownload).mockReturnValue(start.promise);
    vi.mocked(expandPrompt)
      .mockRejectedValueOnce(new Error("local expand model not found, run: mold pull qwen3-expand"))
      .mockResolvedValueOnce({ original: "a lighthouse at dusk", expanded: ["storm light"] });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    wrapper.findComponent(ExpansionPullStatus).vm.$emit("pull");
    await flushPromises();
    expect(wrapper.findComponent(ExpansionPullStatus).props("status")).toMatchObject({
      kind: "connecting",
    });

    stream.resolve();
    await flushPromises();
    expect(wrapper.findComponent(ExpansionPullStatus).props("status")).toMatchObject({
      kind: "starting",
    });

    useDownloadsStore().apply({
      type: "snapshot",
      listing: {
        active_jobs: [],
        queued: [
          {
            id: "snapshot-before-id",
            model: "qwen3-expand:q8",
            status: "queued",
            files_done: 0,
            files_total: 0,
            bytes_done: 0,
            bytes_total: 0,
          },
        ],
        history: [],
      },
    });
    await flushPromises();
    expect(wrapper.findComponent(ExpansionPullStatus).props("status")).toMatchObject({
      kind: "queued",
      job: { id: "snapshot-before-id" },
    });

    start.resolve(null);
    await flushPromises();
    useDownloadsStore().apply({
      type: "started",
      id: "snapshot-before-id",
      files_total: 2,
      bytes_total: 1_000,
    });
    useDownloadsStore().apply({
      type: "progress",
      id: "snapshot-before-id",
      files_done: 1,
      bytes_done: 750,
      current_file: "model.safetensors",
    });
    await flushPromises();
    expect(wrapper.findComponent(ExpansionPullStatus).props("status")).toMatchObject({
      kind: "pulling",
      job: { bytes_done: 750, current_file: "model.safetensors" },
    });

    useDownloadsStore().apply({
      type: "job_done",
      id: "snapshot-before-id",
      model: "qwen3-expand:q8",
    });
    await flushPromises();
    expect(wrapper.findComponent(ExpansionPullStatus).props("status")).toMatchObject({
      kind: "ready",
    });
    wrapper.findComponent(ExpansionPullStatus).vm.$emit("retry-expansion");
    await flushPromises();
    expect(expandPrompt).toHaveBeenLastCalledWith(
      "a lighthouse at dusk",
      { variations: 1, modelFamily: "flux", task: "text-to-image" },
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    );
    expect(useGenerateFormStore().form.prompt).toBe("storm light");
  });

  it("adopts the exact already-queued job id from a 409 body for the real bare alias", async () => {
    useGenerateFormStore().form.batchSize = 1;
    const downloads = useDownloadsStore();
    downloads.queued = [
      {
        id: "existing-A",
        model: "qwen3-expand:q8",
        status: "queued",
        files_done: 0,
        files_total: 0,
        bytes_done: 0,
        bytes_total: 0,
      },
      {
        id: "competing-B",
        model: "qwen3-expand:q8",
        status: "queued",
        files_done: 0,
        files_total: 0,
        bytes_done: 0,
        bytes_total: 0,
      },
    ];
    vi.spyOn(downloads, "subscribe").mockResolvedValue();
    vi.mocked(startCatalogDownload).mockRejectedValue(
      new ApiError("already queued", 409, { id: "existing-A" }),
    );
    vi.mocked(expandPrompt).mockRejectedValue(
      new Error("local expand model not found, run: mold pull qwen3-expand"),
    );
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    wrapper.findComponent(ExpansionPullStatus).vm.$emit("pull");
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalledWith(
      "qwen3-expand",
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
      false,
    );
    expect(wrapper.findComponent(ExpansionPullStatus).props("status")).toMatchObject({
      kind: "queued",
      job: { id: "existing-A" },
    });
  });

  it("places the same inline pull surface inside a preserved prepared batch", async () => {
    vi.mocked(expandPrompt)
      .mockResolvedValueOnce({
        original: "a lighthouse at dusk",
        expanded: ["one", "two", "three"],
      })
      .mockRejectedValueOnce(
        new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
      );
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    wrapper.findComponent(PreparedExpansionBatch).vm.$emit("regenerate");
    await flushPromises();

    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.exists()).toBe(true);
    expect(prepared.props("pullStatus")).toEqual({ kind: "missing", job: null });
    expect(prepared.props("pullModel")).toBe("qwen3-expand:q8");
    expect(prepared.findComponent(ExpansionPullStatus).exists()).toBe(true);
  });

  it("ignores matching download activity from every host except the frozen route", async () => {
    useGenerateFormStore().form.batchSize = 1;
    const hosts = useHostsStore();
    hosts.extras = [
      {
        id: "studio",
        label: "Studio GPU",
        url: "http://studio:7680",
        apiKey: "studio-key",
        status: "ready",
        error: null,
        instanceId: "studio-instance",
      },
    ];
    vi.spyOn(hosts, "resolveRoute").mockReturnValue({
      hostId: "studio",
      label: "Studio GPU",
      kind: "remote",
      target: { baseUrl: "http://studio:7680", apiKey: "studio-key" },
    });
    const downloads = useDownloadsStore();
    downloads.hostStates.studio = {
      label: "Studio GPU",
      target: { baseUrl: "http://studio:7680", apiKey: "studio-key" },
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
      activeJobs: [],
      queued: [],
      history: [],
    };
    const subscribe = vi.spyOn(downloads, "subscribe").mockResolvedValue();
    vi.mocked(startCatalogDownload).mockReturnValue(new Promise(() => {}));
    vi.mocked(expandPrompt).mockRejectedValue(
      new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
    );
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();
    wrapper.findComponent(ExpansionPullStatus).vm.$emit("pull");
    await flushPromises();
    expect(subscribe).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "studio",
        baseUrl: "http://studio:7680",
        apiKey: "studio-key",
      }),
    );

    downloads.apply({
      type: "enqueued",
      id: "wrong-host-job",
      model: "qwen3-expand:q8",
      position: 1,
    });
    await flushPromises();
    expect(wrapper.findComponent(ExpansionPullStatus).props("status")).toMatchObject({
      kind: "starting",
    });

    downloads.applyForHost("studio", {
      type: "enqueued",
      id: "exact-host-job",
      model: "qwen3-expand:q8",
      position: 1,
    });
    await flushPromises();
    expect(wrapper.findComponent(ExpansionPullStatus).props("status")).toMatchObject({
      kind: "queued",
      job: { id: "exact-host-job" },
    });
    expect(startCatalogDownload).toHaveBeenCalledWith(
      "qwen3-expand:q8",
      { baseUrl: "http://studio:7680", apiKey: "studio-key" },
      true,
    );
  });

  it("updates Batch on intentional removal and reports partial sibling failure", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    let prepared = wrapper.findComponent(PreparedExpansionBatch);
    prepared.vm.$emit("remove", prepared.props("batch").prompts[2]!.id);
    await flushPromises();
    expect(useGenerateFormStore().form.batchSize).toBe(2);
    prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.props("batch").prompts).toHaveLength(2);

    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([
        { status: "complete", error: null },
        { status: "error", error: "out of memory" },
      ]),
    } as never);
    prepared.vm.$emit("generate");
    await flushPromises();

    expect(
      useToastStore().items.some(
        (toast) => toast.kind === "error" && toast.message.includes("1 of 2 variations"),
      ),
    ).toBe(true);
  });
});
