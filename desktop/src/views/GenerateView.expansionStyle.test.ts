/**
 * Style-aware expansion (bake-and-clear): the active style chip travels to the
 * server as a natural-language directive (`styleHint`), never as the literal
 * suffix. A quick Batch 1 apply bakes the style into the rewritten prompt and
 * clears the chip (undo restores it); a prepared Batch N keeps the chip as the
 * frozen-style indicator, marks a style change as stale work, and ships its
 * reviewed prompts verbatim while the preset negative still merges.
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
import { useUiStore } from "../stores/ui";
import { expandPrompt } from "../lib/api/expand";
import { styleHint } from "../lib/stylePresets";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
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

const cinematicHint = styleHint("cinematic")!;

function mountView() {
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: {
      stubs: {
        ExpandControl: false,
        PreparedExpansionBatch: false,
        ExpansionPullStatus: false,
        ComposerCard: false,
        InspectorPanel: false,
      },
    },
  });
}

describe("GenerateView style-aware expansion", () => {
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
    form.model = fluxModel.name;
    form.family = fluxModel.family;
    form.batchSize = 1;
    form.stylePreset = "cinematic";
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("quick Batch 1 carries the style hint, then bakes and clears the chip", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light over a cinematic coast"],
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith(
      "a lighthouse at dusk",
      { variations: 1, modelFamily: "flux", style: cinematicHint },
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    );
    const form = useGenerateFormStore().form;
    expect(form.prompt).toBe("storm light over a cinematic coast");
    // Bake-and-clear: the expansion output absorbed the style.
    expect(form.stylePreset).toBe("");
  });

  it("merges the preset negative into the composer when a quick apply clears the chip", async () => {
    const form = useGenerateFormStore().form;
    form.model = sdxlModel.name;
    form.family = sdxlModel.family;
    form.negativePrompt = "text";
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light over a cinematic coast"],
    });
    const submit = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    // Bake-and-clear drops the chip, so the curated negative has to land in the
    // form now — the submit-time merge no longer sees a preset.
    expect(form.stylePreset).toBe("");
    expect(form.negativePrompt).toBe("text, anime, cartoon, graphic, washed out");

    useUiStore().generateTick++;
    await flushPromises();
    expect(submit).toHaveBeenCalledTimes(1);
    const [request] = submit.mock.calls[0]!;
    // …and exactly once — the cleared chip can't merge it a second time.
    expect(request.negative_prompt).toBe("text, anime, cartoon, graphic, washed out");
  });

  it("leaves the negative alone when the family takes no negative prompt", async () => {
    const form = useGenerateFormStore().form;
    form.negativePrompt = "text";
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light over a cinematic coast"],
    });
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    // flux ignores a negative prompt — baking one in would be noise the user
    // then has to delete by hand.
    expect(form.negativePrompt).toBe("text");
  });

  it("restoring the quick expansion restores the negative alongside the chip", async () => {
    const form = useGenerateFormStore().form;
    form.model = sdxlModel.name;
    form.family = sdxlModel.family;
    form.negativePrompt = "text";
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light over a cinematic coast"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("restore");
    await flushPromises();

    expect(form.negativePrompt).toBe("text");
    expect(form.stylePreset).toBe("cinematic");
  });

  it("restoring the quick expansion restores the preset chip", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light over a cinematic coast"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    wrapper.findComponent(ExpandControl).vm.$emit("restore");
    await flushPromises();

    const form = useGenerateFormStore().form;
    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.stylePreset).toBe("cinematic");
  });

  it("discards a quick apply when the style changes while expansion is in flight", async () => {
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

    const form = useGenerateFormStore().form;
    form.stylePreset = "anime";
    resolveExpansion({
      original: "a lighthouse at dusk",
      expanded: ["late styled rewrite"],
    });
    await flushPromises();

    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.stylePreset).toBe("anime");
    expect(wrapper.get('[role="alert"]').text()).toContain("changed while expansion was running");
  });

  it("prepared Batch N carries the style hint and keeps the chip as the frozen-style indicator", async () => {
    useGenerateFormStore().form.batchSize = 3;
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
      { variations: 3, modelFamily: "flux", style: cinematicHint },
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
    );
    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.exists()).toBe(true);
    expect(prepared.props("batch").stylePreset).toBe("cinematic");
    // Prepared keeps the chip — it is the frozen-style indicator.
    expect(useGenerateFormStore().form.stylePreset).toBe("cinematic");
  });

  it("marks the prepared batch stale and blocks Generate when the style changes", async () => {
    useGenerateFormStore().form.batchSize = 3;
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist", "aerial coast"],
    });
    const submit = vi.spyOn(useGenerationStore(), "submitBatch");
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    useGenerateFormStore().form.stylePreset = "anime";
    await flushPromises();

    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    expect(prepared.props("staleReasons")).toContain("Style changed from Cinematic to Anime.");
    prepared.vm.$emit("generate");
    await flushPromises();
    expect(submit).not.toHaveBeenCalled();
  });

  it("ships reviewed prompts verbatim while merging the preset negative for prepared batches", async () => {
    const form = useGenerateFormStore().form;
    form.model = sdxlModel.name;
    form.family = sdxlModel.family;
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
    // Reviewed prompt text is never suffixed or templated.
    expect(options).toMatchObject({
      prompts: ["storm light", "sea mist", "aerial coast"],
      originalPrompt: "a lighthouse at dusk",
    });
    expect(request.prompt).toBe("a lighthouse at dusk");
    // negative_prompt is separate from the reviewed prompt text, so the
    // frozen style's negative still merges (user fragments first).
    expect(request.negative_prompt).toBe("text, anime, cartoon, graphic, washed out");
  });

  it("clears the chip when a prepared pair collapses into the composer", async () => {
    useGenerateFormStore().form.batchSize = 2;
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    prepared.vm.$emit("collapse", prepared.props("batch").prompts[1]!.id);
    await flushPromises();

    const form = useGenerateFormStore().form;
    expect(form.batchSize).toBe(1);
    expect(form.prompt).toBe("storm light");
    // The surviving reviewed text absorbed the style — same bake-and-clear.
    expect(form.stylePreset).toBe("");
  });

  it("carries the preset negative across a prepared-pair collapse", async () => {
    const form = useGenerateFormStore().form;
    form.model = sdxlModel.name;
    form.family = sdxlModel.family;
    form.negativePrompt = "text";
    form.batchSize = 2;
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "a lighthouse at dusk",
      expanded: ["storm light", "sea mist"],
    });
    const wrapper = mountView();
    await flushPromises();
    wrapper.findComponent(ExpandControl).vm.$emit("expand");
    await flushPromises();

    const prepared = wrapper.findComponent(PreparedExpansionBatch);
    prepared.vm.$emit("collapse", prepared.props("batch").prompts[1]!.id);
    await flushPromises();

    // Collapse clears the chip too, so the frozen style's negative has to move
    // into the form with it.
    expect(form.stylePreset).toBe("");
    expect(form.negativePrompt).toBe("text, anime, cartoon, graphic, washed out");
  });
});
