/**
 * The composer style preset is a look modifier composed into the OUTGOING
 * request at submit — the textarea (form.prompt) and negative field are never
 * mutated. The preset negative merges only for families that accept one.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import GenerateView from "./GenerateView.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useGenerationStore } from "../stores/generation";
import { useHostModelsStore } from "../stores/hostModels";
import { useModelStore } from "../stores/models";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useUiStore } from "../stores/ui";
import type { GenerateRequest, ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
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

const fluxModel: ModelEntry = {
  name: "flux-dev:q8",
  family: "flux",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
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

async function submitStyled(overrides: {
  model: ModelEntry;
  stylePreset: string;
  negativePrompt?: string;
}): Promise<GenerateRequest> {
  const submit = vi
    .spyOn(useGenerationStore(), "submitBatch")
    .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

  mount(GenerateView, { shallow: true, attachTo: document.body });
  await flushPromises();
  const form = useGenerateFormStore().form;
  form.prompt = "a lighthouse";
  form.model = overrides.model.name;
  form.family = overrides.model.family;
  form.batchSize = 1;
  form.stylePreset = overrides.stylePreset;
  form.negativePrompt = overrides.negativePrompt ?? "";

  useUiStore().generateTick++;
  await flushPromises();

  expect(submit).toHaveBeenCalledTimes(1);
  return submit.mock.calls[0]![0];
}

describe("GenerateView style-at-submit", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
    conn.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [fluxModel, sdxlModel];
    // A print is routed from inventory, not from a placement preview, so This
    // device must have listed its own models before one can be queued.
    useHostModelsStore().byHost.local = {
      entries: [fluxModel, sdxlModel],
      fetchedAt: Date.now(),
      error: null,
    };
  });
  afterEach(() => (document.body.innerHTML = ""));

  it("bakes the style template into the request prompt, leaving the textarea unchanged", async () => {
    const request = await submitStyled({ model: fluxModel, stylePreset: "cinematic" });
    expect(request.prompt).toBe(
      "cinematic film still of a lighthouse, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
    // The live composer text is untouched.
    expect(useGenerateFormStore().form.prompt).toBe("a lighthouse");
  });

  it("merges the preset negative after the user's fragments when the family accepts one", async () => {
    const request = await submitStyled({
      model: sdxlModel,
      stylePreset: "cinematic",
      negativePrompt: "text, watermark, anime",
    });
    // User fragments first, exact duplicates dropped, preset fragments appended.
    expect(request.negative_prompt).toBe("text, watermark, anime, cartoon, graphic, washed out");
    // The visible negative field is untouched.
    expect(useGenerateFormStore().form.negativePrompt).toBe("text, watermark, anime");
  });

  it("never ships the preset negative to a family that rejects negative prompts", async () => {
    const request = await submitStyled({
      model: fluxModel,
      stylePreset: "cinematic",
      negativePrompt: "text",
    });
    expect(request.negative_prompt).toBeUndefined();
    expect(useGenerateFormStore().form.negativePrompt).toBe("text");
  });
});
