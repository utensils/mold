import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import GenerateView from "./GenerateView.vue";
import ExpandControl from "../components/generate/ExpandControl.vue";
import PreparedExpansionBatch from "../components/generate/PreparedExpansionBatch.vue";
import ExpansionPullStatus from "../components/generate/ExpansionPullStatus.vue";
import { useConnectionStore } from "../stores/connection";
import { useGenerateFormStore } from "../stores/generateForm";
import { useHostsStore } from "../stores/hosts";
import { useHostModelsStore } from "../stores/hostModels";
import { useModelStore } from "../stores/models";
import { useDownloadsStore } from "../stores/downloads";
import { useAppPrefsStore } from "../stores/appPrefs";
import { expandPrompt } from "../lib/api/expand";
import { startCatalogDownload } from "../lib/api/catalog";
import type { ModelEntry, ServerCapabilities } from "../lib/api/types";

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

function expandCapability(
  modelPresent: boolean | null,
  named = "qwen3-expand:q8",
): ServerCapabilities {
  return {
    gallery: { can_delete: true },
    expand: {
      configured: true,
      model_present: modelPresent,
      backend: "local",
      ...(named ? { model: named } : {}),
    },
  } as ServerCapabilities;
}

function mountView() {
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: {
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

/** Both machines hold the checkpoint, so generation routing is free to choose
 *  on queue depth alone — the expander is the only thing that differs. */
function addRemoteHost(): void {
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
  hosts.telemetry.local = { queueDepth: 0, queueCapacity: 8, version: "0.23.3" };
  hosts.telemetry["hal9000-7680"] = { queueDepth: 5, queueCapacity: 8, version: "0.23.3" };
  const hostModels = useHostModelsStore();
  hostModels.byHost.local = { entries: [model], fetchedAt: Date.now(), error: null };
  hostModels.byHost["hal9000-7680"] = { entries: [model], fetchedAt: Date.now(), error: null };
}

describe("GenerateView expansion routing", () => {
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
    vi.mocked(startCatalogDownload).mockReset();
    vi.mocked(startCatalogDownload).mockResolvedValue("pull-job");

    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    connection.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [model];
    const form = useGenerateFormStore().form;
    form.prompt = "a lighthouse at dusk";
    form.model = model.name;
    form.family = model.family;
    form.batchSize = 3;
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("expands on a peer that has the expand model while the print stays on the generation route", async () => {
    addRemoteHost();
    const hosts = useHostsStore();
    hosts.capabilities.local = expandCapability(false);
    hosts.capabilities["hal9000-7680"] = expandCapability(true);
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
    const batch = wrapper.findComponent(PreparedExpansionBatch).props("batch");
    // The print is queued where the generation router sent it — never on the
    // machine that merely rewrote the prompt.
    expect(batch.route.hostId).toBe("local");
    expect(batch.expansionRoute?.hostId).toBe("hal9000-7680");
  });

  it("keeps the generation route when its expand capability was never read", async () => {
    addRemoteHost();
    const hosts = useHostsStore();
    hosts.capabilities["hal9000-7680"] = expandCapability(true);
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith("a lighthouse at dusk", expect.anything(), {
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    });
    expect(
      wrapper.findComponent(PreparedExpansionBatch).props("batch").expansionRoute,
    ).toBeUndefined();
  });

  it("never leaves a pinned machine that lacks the expander", async () => {
    addRemoteHost();
    const hosts = useHostsStore();
    hosts.capabilities.local = expandCapability(false);
    hosts.capabilities["hal9000-7680"] = expandCapability(true);
    useAppPrefsStore().settings = { generateTargetHost: "local" } as never;
    const subscribe = vi.spyOn(useDownloadsStore(), "subscribe").mockResolvedValue();

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).not.toHaveBeenCalled();
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

  it("offers the pull without a doomed request when no machine has the expander", async () => {
    addRemoteHost();
    const hosts = useHostsStore();
    hosts.capabilities.local = expandCapability(false);
    hosts.capabilities["hal9000-7680"] = expandCapability(false);

    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).not.toHaveBeenCalled();
    // The pull defaults to the generation route so prepared work freezes ONE
    // machine, even though every reachable machine is a legitimate target.
    expect(wrapper.get('[data-test="expansion-pull-status"]').text()).toContain(
      "isn't installed on This device",
    );
  });
});
