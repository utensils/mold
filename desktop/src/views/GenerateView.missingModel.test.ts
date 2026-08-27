import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import GenerateView from "./GenerateView.vue";
import MissingModelDialog from "../components/generate/MissingModelDialog.vue";
import DownloadTargetDialog from "../components/models/DownloadTargetDialog.vue";
import InspectorPanel from "../components/create/InspectorPanel.vue";
import { useComposerStore } from "../stores/composer";
import { useHostModelsStore } from "../stores/hostModels";
import { useConnectionStore } from "../stores/connection";
import { useGenerateFormStore } from "../stores/generateForm";
import { useGenerationStore } from "../stores/generation";
import { useHostsStore, type FeasibleRouteResult, type HostRoute } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useToastStore } from "../stores/toasts";
import { useDownloadsStore } from "../stores/downloads";
import { usePullResumeStore } from "../stores/pullResume";
import { startCatalogDownload } from "../lib/api/catalog";
import { applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));
vi.mock("../lib/api/expand", () => ({ expandPrompt: vi.fn() }));
vi.mock("../lib/api/remix", () => ({ remixPrompt: vi.fn() }));
vi.mock("../lib/api/catalog", () => ({ startCatalogDownload: vi.fn() }));
vi.mock("../lib/sourceFitPreprocess", () => ({ applySourceFitPreprocess: vi.fn() }));

enableAutoUnmount(afterEach);

const model: ModelEntry = {
  name: "z-image-turbo:q6",
  family: "zimage",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 8,
  default_guidance: 1,
} as ModelEntry;

const localRoute: HostRoute = {
  hostId: "local",
  label: "This device",
  kind: "local",
  target: { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
  instanceId: null,
};
const remoteRoute: HostRoute = {
  hostId: "hal9000-7680",
  label: "hal9000",
  kind: "remote",
  target: { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
  instanceId: "hal-instance",
};

function missingModelFailure(route: HostRoute) {
  return {
    kind: "infeasible" as const,
    hostId: route.hostId,
    label: route.label,
    route,
    reason: `model '${model.name}' has no concrete local artifacts`,
    missingComponents: [
      {
        kind: "transformer",
        name: "transformer",
        present: false,
        repair_model: model.name,
      },
    ],
    missingModel: { model: model.name, missingComponents: [] },
  };
}

function capacityFailure(route: HostRoute) {
  return {
    kind: "infeasible" as const,
    hostId: route.hostId,
    label: route.label,
    route,
    reason: "no device can host this generation: needs 48.0 GB",
    missingComponents: [],
    missingModel: null,
  };
}

function mountView() {
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: {
      stubs: {
        ComposerCard: false,
        InspectorPanel: false,
        ModelPicker: false,
        GenerateErrorNotice: false,
        ErrorNotice: false,
      },
    },
  });
}

function addRemoteHost() {
  useHostsStore().extras.push({
    id: remoteRoute.hostId,
    label: remoteRoute.label,
    url: remoteRoute.target.baseUrl!,
    apiKey: remoteRoute.target.apiKey ?? null,
    status: "ready",
    error: null,
    instanceId: remoteRoute.instanceId ?? null,
  });
}

describe("GenerateView missing-model pull before submit", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    apiJson
      .mockReset()
      .mockImplementation((path: string) => Promise.resolve(path === "/api/models" ? [model] : []));
    apiJsonTo
      .mockReset()
      .mockImplementation((_target: unknown, path: string) =>
        Promise.resolve(path === "/api/models" ? [model] : []),
      );
    vi.mocked(startCatalogDownload).mockReset().mockResolvedValue("pull-job-1");
    vi.mocked(applySourceFitPreprocess).mockReset().mockResolvedValue({
      source: null,
      mask: null,
      changed: false,
    });

    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: localRoute.target.baseUrl!,
      apiKey: localRoute.target.apiKey ?? null,
    };
    connection.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [model];
    const form = useGenerateFormStore().form;
    form.prompt = "a lighthouse at dusk";
    form.model = model.name;
    form.family = model.family;
    form.batchSize = 1;
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("offers the pull on the one machine that only lacks the model", async () => {
    const hosts = useHostsStore();
    const infeasible: FeasibleRouteResult = {
      kind: "infeasible",
      perHost: [missingModelFailure(localRoute)],
    };
    vi.spyOn(hosts, "resolveFeasible").mockResolvedValue(infeasible);
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });
    const toasts = useToastStore();
    const wrapper = mountView();
    await flushPromises();

    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    expect(submit).not.toHaveBeenCalled();
    expect(wrapper.findComponent(DownloadTargetDialog).exists()).toBe(false);
    const dialog = wrapper.findComponent(MissingModelDialog);
    expect(dialog.exists()).toBe(true);
    expect(dialog.props("model")).toBe(model.name);
    expect(dialog.props("hostLabel")).toBe("This device");
    expect(toasts.items.some((item) => item.kind === "error")).toBe(false);
  });

  it("asks which machine to pull onto when more than one could run it", async () => {
    addRemoteHost();
    const hosts = useHostsStore();
    vi.spyOn(hosts, "resolveFeasible").mockResolvedValue({
      kind: "infeasible",
      perHost: [missingModelFailure(localRoute), missingModelFailure(remoteRoute)],
    } as FeasibleRouteResult);
    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const downloads = useDownloadsStore();
    vi.spyOn(downloads, "subscribe").mockResolvedValue(undefined);
    const wrapper = mountView();
    await flushPromises();

    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    const picker = wrapper.findComponent(DownloadTargetDialog);
    expect(picker.exists()).toBe(true);
    expect(picker.props("modelName")).toBe(model.name);
    expect(
      (picker.props("targets") as Array<{ host: { id: string } }>).map((target) => target.host.id),
    ).toEqual(expect.arrayContaining(["local", remoteRoute.hostId]));

    const remoteHost = useHostsStore().all.find((host) => host.id === remoteRoute.hostId);
    picker.vm.$emit("select", remoteHost);
    await flushPromises();

    expect(wrapper.findComponent(DownloadTargetDialog).exists()).toBe(false);
    const dialog = wrapper.findComponent(MissingModelDialog);
    expect(dialog.exists()).toBe(true);
    expect(dialog.props("hostLabel")).toBe("hal9000");

    dialog.vm.$emit("confirm");
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalledWith(
      model.name,
      { baseUrl: remoteRoute.target.baseUrl, apiKey: remoteRoute.target.apiKey },
      true,
    );
    const pullResume = usePullResumeStore();
    expect(pullResume.pending).toMatchObject({
      model: model.name,
      hostId: remoteRoute.hostId,
      hostLabel: "hal9000",
      jobId: "pull-job-1",
    });
    expect(pullResume.pending?.request.prompt).toBe("a lighthouse at dusk");
    expect(Object.values(useDownloadsStore().notificationActions)).toContainEqual(
      expect.objectContaining({ action: { kind: "create" } }),
    );
  });

  it("offers the pull from the picker's Not installed row without queueing", async () => {
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = { entries: [], fetchedAt: Date.now(), error: null };
    vi.spyOn(hostModels, "refresh").mockResolvedValue(undefined);
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(InspectorPanel).vm.$emit("pull-missing-model", model.name);
    await flushPromises();

    expect(submit).not.toHaveBeenCalled();
    const dialog = wrapper.findComponent(MissingModelDialog);
    expect(dialog.exists()).toBe(true);
    expect(dialog.props("model")).toBe(model.name);
    expect(dialog.props("hostLabel")).toBe("This device");
  });

  it("discloses a restored model that no machine has, keeping its raw id", async () => {
    const toasts = useToastStore();
    // Nothing is installed anywhere: the restored id has no inventory entry.
    apiJson.mockResolvedValue([]);
    apiJsonTo.mockResolvedValue([]);
    useModelStore().all = [];
    const form = useGenerateFormStore().form;
    form.model = "";
    const wrapper = mountView();
    await flushPromises();

    useComposerStore().set({
      prompt: "a lighthouse at dusk",
      model: model.name,
      seed: null,
      width: 1024,
      height: 1024,
      steps: 8,
      guidance: 1,
    });
    await flushPromises();

    expect(form.model).toBe(model.name);
    expect(toasts.items.some((item) => item.message.includes("isn't installed"))).toBe(true);
    // The picker's Not installed rendering is covered by InspectorPanel's own
    // test; here the contract is the raw id plus the disclosure.
    expect(wrapper.exists()).toBe(true);
  });

  // The source is fitted against the machine that runs it, so a request that
  // has not been preprocessed yet is not the one that would render: download,
  // but never promise the generation.
  it("downloads without promising a resume when the source still needs fitting", async () => {
    const hosts = useHostsStore();
    vi.spyOn(hosts, "resolveFeasible").mockResolvedValue({
      kind: "infeasible",
      perHost: [missingModelFailure(localRoute)],
    } as FeasibleRouteResult);
    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const downloads = useDownloadsStore();
    vi.spyOn(downloads, "subscribe").mockResolvedValue(undefined);
    const form = useGenerateFormStore().form;
    form.family = "flux";
    form.model = model.name;
    form.sourceImage = "SOURCE";
    form.sourceFit = {
      mode: "upscale-then-fit",
      upscalerModel: "realesrgan-x4",
      fit: { mode: "pad-repaint" },
    };
    const wrapper = mountView();
    await flushPromises();

    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    const dialog = wrapper.findComponent(MissingModelDialog);
    expect(dialog.exists()).toBe(true);
    expect(dialog.props("resumeAfterPull")).toBe(false);

    dialog.vm.$emit("confirm");
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalled();
    expect(usePullResumeStore().pending).toBeNull();
    expect(useToastStore().items.at(-1)?.message).toContain("press Generate again");
  });

  it("never offers a pull for a machine that simply cannot fit the print", async () => {
    const hosts = useHostsStore();
    vi.spyOn(hosts, "resolveFeasible").mockResolvedValue({
      kind: "infeasible",
      perHost: [capacityFailure(localRoute)],
    } as FeasibleRouteResult);
    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const toasts = useToastStore();
    const wrapper = mountView();
    await flushPromises();

    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    expect(wrapper.findComponent(MissingModelDialog).exists()).toBe(false);
    expect(wrapper.findComponent(DownloadTargetDialog).exists()).toBe(false);
    expect(toasts.items.at(-1)?.message).toContain("No selected machine can run this print");
  });

  it("offers the pull when a slow machine did not answer but another lacks only the model", async () => {
    addRemoteHost();
    const hosts = useHostsStore();
    vi.spyOn(hosts, "resolveFeasible").mockResolvedValue({
      kind: "mixed",
      perHost: [
        missingModelFailure(localRoute),
        {
          kind: "unreachable",
          hostId: remoteRoute.hostId,
          label: remoteRoute.label,
          error: "Auto placement timed out after 20 seconds",
        },
      ],
    } as FeasibleRouteResult);
    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();

    await wrapper.get('[data-test="generate-button"]').trigger("click");
    await flushPromises();

    expect(wrapper.findComponent(MissingModelDialog).props("hostLabel")).toBe("This device");
  });
});

describe("GenerateView missing-model pull after a durable hold", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    apiJson
      .mockReset()
      .mockImplementation((path: string) => Promise.resolve(path === "/api/models" ? [model] : []));
    apiJsonTo
      .mockReset()
      .mockImplementation((_target: unknown, path: string) =>
        Promise.resolve(path === "/api/models" ? [model] : []),
      );
    vi.mocked(startCatalogDownload).mockReset().mockResolvedValue("pull-job-1");
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: localRoute.target.baseUrl!,
      apiKey: localRoute.target.apiKey ?? null,
    };
    connection.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [model];
    const form = useGenerateFormStore().form;
    form.prompt = "a lighthouse at dusk";
    form.model = model.name;
    form.family = model.family;
    form.batchSize = 1;
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  /** A print is admitted before the machine resolves its model, so a missing
   *  model parks the child instead of refusing the request. */
  function holdJob(generation: ReturnType<typeof useGenerationStore>, reason: string) {
    const job = generation.startJob({
      prompt: "a lighthouse at dusk",
      model: model.name,
      width: 1024,
      height: 1024,
      steps: 8,
    });
    job.hostId = "local";
    job.status = "queued";
    job.holdError = reason;
    job.retryable = true;
    return job;
  }

  it("offers the pull once for a child held because the model is missing", async () => {
    const generation = useGenerationStore();
    const wrapper = mountView();
    await flushPromises();

    const job = holdJob(generation, "UNKNOWN_MODEL: z-image-turbo:q6 is not installed");
    await flushPromises();

    const dialog = wrapper.findComponent(MissingModelDialog);
    expect(dialog.exists()).toBe(true);
    expect(dialog.props("model")).toBe(model.name);

    // Re-reporting the same hold must not raise a second offer.
    job.holdError = "UNKNOWN_MODEL: z-image-turbo:q6 is not installed (again)";
    await flushPromises();
    expect(wrapper.findAllComponents(MissingModelDialog)).toHaveLength(1);
  });

  it("offers nothing for a hold that is not about the model", async () => {
    const generation = useGenerationStore();
    const wrapper = mountView();
    await flushPromises();

    holdJob(generation, "insufficient VRAM on this device");
    await flushPromises();

    expect(wrapper.findComponent(MissingModelDialog).exists()).toBe(false);
  });

  it("retries the held child rather than queueing a second print", async () => {
    const generation = useGenerationStore();
    const retryHeld = vi.spyOn(generation, "retryHeld").mockResolvedValue(undefined);
    const submit = vi
      .spyOn(generation, "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });
    const downloads = useDownloadsStore();
    vi.spyOn(downloads, "subscribe").mockResolvedValue(undefined);
    const pullResume = usePullResumeStore();
    const wrapper = mountView();
    await flushPromises();
    const job = holdJob(generation, "MODEL_NOT_FOUND");
    await flushPromises();

    wrapper.findComponent(MissingModelDialog).vm.$emit("confirm");
    await flushPromises();

    expect(pullResume.pending?.retryClientId).toBe(job.clientId);
    downloads.history = [{ id: "pull-job-1", model: model.name, status: "completed" }] as never;
    pullResume.check();
    await flushPromises();

    expect(retryHeld).toHaveBeenCalledWith(job.clientId);
    expect(submit).not.toHaveBeenCalled();
  });
});
