/**
 * "Make 4 variations" is the same picture, made again as a batch of four —
 * nothing about the words, the style, the size or the seed policy changes,
 * the persisted batch size included. It is reachable from the canvas caption,
 * from ⌥↩, and from the command palette, and all three are the one action on
 * the UI store. It is offered only where it means something: a finished still
 * on a recipe that can repeat.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises, mount } from "@vue/test-utils";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

import GenerateView from "./GenerateView.vue";
import { newJob } from "../lib/generationJob";
import { useConnectionStore } from "../stores/connection";
import { useGenerateFormStore } from "../stores/generateForm";
import { useGenerationStore } from "../stores/generation";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useUiStore } from "../stores/ui";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
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
vi.mock("../lib/ipc", () => ({ ipc: {} }));

const sdxlModel: ModelEntry = {
  name: "sdxl-base:fp16",
  family: "sdxl",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 30,
  default_guidance: 7.0,
} as ModelEntry;

beforeEach(() => {
  setActivePinia(createPinia());
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
  conn.status = "ready";
  useHostsStore().initialized = true;
  useModelStore().all = [sdxlModel];
  useHostModelsStore().byHost.local = {
    entries: [sdxlModel],
    fetchedAt: Date.now(),
    error: null,
  };
});
afterEach(() => (document.body.innerHTML = ""));

/** A finished print on the canvas — the only state the action is offered in. */
function finishPrint(result: Record<string, unknown> = {}) {
  const generation = useGenerationStore();
  const job = newJob({
    prompt: "a brass teapot on a rainy windowsill",
    model: sdxlModel.name,
    width: 1024,
    height: 1024,
    steps: 30,
  } as GenerateRequest);
  Object.assign(job, {
    clientId: 1,
    batchId: 1,
    id: "finished-print",
    status: "complete",
    result: {
      image: "cGl4ZWxz",
      filename: "teapot.png",
      model: sdxlModel.name,
      format: "png",
      seed_used: 4821,
      ...result,
    },
  });
  generation.jobs.push(job);
  generation.selectedClientId = job.clientId;
  return job;
}

async function mountWithAPrint(shallow = true) {
  mount(GenerateView, { shallow, attachTo: document.body });
  await flushPromises();
  const form = useGenerateFormStore().form;
  form.prompt = "a brass teapot on a rainy windowsill";
  form.model = sdxlModel.name;
  form.family = sdxlModel.family;
  form.batchSize = 1;
  form.seed = "4821";
  await flushPromises();
  return form;
}

describe("GenerateView — Make 4 variations", () => {
  it("re-submits the print at batch four, leaving every other setting alone", async () => {
    const form = await mountWithAPrint();
    finishPrint();
    await flushPromises();
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    useUiStore().makeVariations();
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit.mock.calls[0]![1]).toBe(4);
    const request = submit.mock.calls[0]![0];
    expect(request.prompt).toBe("a brass teapot on a rainy windowsill");
    expect(request.model).toBe(sdxlModel.name);
    expect(form.seed).toBe("4821");
  });

  /**
   * The count rides the ONE submission. Writing it to the form left every
   * later Generate quietly making four pictures.
   */
  it("leaves the form's own batch size where the user set it", async () => {
    const form = await mountWithAPrint();
    finishPrint();
    await flushPromises();
    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });

    useUiStore().makeVariations();
    await flushPromises();

    expect(form.batchSize).toBe(1);
  });

  it("stays out of the way in clip mode, where Generate does not mean a batch", async () => {
    const form = await mountWithAPrint();
    useSequenceDraftStore().output = "sequence";
    await flushPromises();
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    useUiStore().makeVariations();
    await flushPromises();

    expect(form.batchSize).toBe(1);
    expect(submit).not.toHaveBeenCalled();
  });
});

describe("GenerateView — where Make 4 variations is offered", () => {
  it("sits in the caption beside Make bigger for a finished still", async () => {
    await mountWithAPrint(false);
    finishPrint();
    await flushPromises();
    expect(document.querySelector("[data-test='canvas-variations']")).not.toBeNull();
    expect(document.querySelector("[data-test='canvas-upscale']")).not.toBeNull();
  });

  /**
   * `effectiveBatchSize` coerces a batch-locked recipe's count to one, so the
   * button would have promised four pictures and made exactly one.
   */
  it("is hidden on a batch-locked recipe", async () => {
    const form = await mountWithAPrint(false);
    finishPrint();
    form.family = "qwen-image-edit";
    form.model = "qwen-image-edit-2511:q8";
    await flushPromises();
    expect(document.querySelector("[data-test='canvas-variations']")).toBeNull();
  });

  it("is hidden for a clip, which has no batch", async () => {
    await mountWithAPrint(false);
    finishPrint({ video_frames: 97 });
    await flushPromises();
    expect(document.querySelector("[data-test='canvas-variations']")).toBeNull();
  });
});
