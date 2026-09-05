/**
 * Expansion and style presets never meet on this surface.
 *
 * The desktop composer has no preset strip — the word "Style" belongs to the
 * model — so nothing here writes `form.stylePreset` and nothing may read one.
 * A preset left behind by a pre-redesign template or a persisted draft must
 * not travel to the expander as a directive, must not be baked into the
 * rewritten prompt, must not merge its curated negative into the field, and
 * must not make a reviewed batch go stale. The phone keeps its own chips and
 * its own bake, which is why `lib/stylePresets.ts` and the form field stay.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import GenerateView from "./GenerateView.vue";
import ExpandControl from "../components/generate/ExpandControl.vue";
import PreparedExpansionBatch from "../components/generate/PreparedExpansionBatch.vue";
import { useConnectionStore } from "../stores/connection";
import { useGenerateFormStore } from "../stores/generateForm";
import { useGenerationStore } from "../stores/generation";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { expandPrompt } from "../lib/api/expand";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
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
        device_id: "cuda:0",
        execution_fingerprint: "test",
        predicted_start_after_ms: 0,
        predicted_completion_after_ms: 100,
        setup_ms: 0,
        setup_kind: "warm",
        estimate_confidence: "high",
      },
    });
  return {
    ...original,
    previewGenerationPlacement: planned,
    previewChainPlacement: planned,
  };
});
vi.mock("../lib/ipc", () => ({ ipc: {} }));
vi.mock("../lib/api/expand", () => ({ expandPrompt: vi.fn() }));
vi.mock("../lib/api/catalog", () => ({ startCatalogDownload: vi.fn() }));
vi.mock("../lib/sourceFitPreprocess", () => ({ applySourceFitPreprocess: vi.fn() }));

enableAutoUnmount(afterEach);

const fluxModel: ModelEntry = {
  name: "flux-dev:q8",
  family: "flux",
  downloaded: true,
  default_width: 768,
  default_height: 768,
  default_steps: 20,
  default_guidance: 4.5,
} as ModelEntry;

const sdxlModel: ModelEntry = {
  name: "sdxl-base:fp16",
  family: "sdxl",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 30,
  default_guidance: 7.0,
} as ModelEntry;

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
      },
    },
  });
}

describe("GenerateView — a leftover style preset reaches nothing", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    apiJson.mockReset();
    apiJson.mockImplementation((path: string) =>
      Promise.resolve(path === "/api/models" ? [fluxModel, sdxlModel] : []),
    );
    apiJsonTo.mockReset();
    apiJsonTo.mockImplementation((_target: unknown, path: string) =>
      Promise.resolve(path === "/api/models" ? [fluxModel, sdxlModel] : []),
    );
    vi.mocked(expandPrompt).mockReset();

    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    connection.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [fluxModel, sdxlModel];
    const form = useGenerateFormStore().form;
    form.prompt = "a lighthouse at dusk";
    form.model = sdxlModel.name;
    form.family = sdxlModel.family;
    form.batchSize = 1;
    // Only a stale draft or an old template can put this here now.
    form.stylePreset = "cinematic";
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("sends no style directive to the expander, and bakes nothing on the way back", async () => {
    const form = useGenerateFormStore().form;
    form.negativePrompt = "text";
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light over a rocky coast"],
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith(
      "a lighthouse at dusk",
      {
        variations: 1,
        modelFamily: "sdxl",
        task: "text-to-image",
        context: expect.any(Object),
      },
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    );
    expect(form.prompt).toBe("storm light over a rocky coast");
    // No curated negative merged in behind the user's back.
    expect(form.negativePrompt).toBe("text");
  });

  it("freezes no style on a prepared batch, so changing one stales nothing", async () => {
    useGenerateFormStore().form.batchSize = 3;
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.props("batch").stylePreset).toBeNull();

    useGenerateFormStore().form.stylePreset = "anime";
    await flushPromises();
    expect(prepared.props("staleReasons")).toEqual([]);
  });

  it("ships the reviewed prompts verbatim, with the composer's own negative", async () => {
    const form = useGenerateFormStore().form;
    form.batchSize = 3;
    form.negativePrompt = "text";
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    wrapper.findComponent(PreparedExpansionBatch).vm.$emit("generate");
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    const [request, , , , options] = submit.mock.calls[0]!;
    expect(options).toMatchObject({
      prompts: ["storm light", "sea mist", "aerial coast"],
      originalPrompt: "a lighthouse at dusk",
    });
    expect(request.prompt).toBe("a lighthouse at dusk");
    expect(request.negative_prompt).toBe("text");
  });
});
