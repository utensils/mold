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

/** A second installed style, so the form can hold something the print does
 *  not and the two answers can be told apart. */
const fluxModel: ModelEntry = {
  ...sdxlModel,
  name: "flux-dev:q8",
  family: "flux",
} as ModelEntry;

/** An edit recipe: it renders one at a time, whatever the form says. */
const editModel: ModelEntry = {
  ...sdxlModel,
  name: "qwen-image-edit-2511:q8",
  family: "qwen-image-edit",
} as ModelEntry;

beforeEach(() => {
  setActivePinia(createPinia());
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
  conn.status = "ready";
  useHostsStore().initialized = true;
  useModelStore().all = [sdxlModel, fluxModel, editModel];
  useHostModelsStore().byHost.local = {
    entries: [sdxlModel, fluxModel, editModel],
    fetchedAt: Date.now(),
    error: null,
  };
});
afterEach(() => (document.body.innerHTML = ""));

/** A finished print on the canvas — the only state the action is offered in. */
function finishPrint(result: Record<string, unknown> = {}, request: Partial<GenerateRequest> = {}) {
  const generation = useGenerationStore();
  const job = newJob({
    prompt: "a brass teapot on a rainy windowsill",
    model: sdxlModel.name,
    width: 1024,
    height: 1024,
    steps: 30,
    ...request,
  } as GenerateRequest);
  Object.assign(job, {
    clientId: 1,
    batchId: 1,
    id: "finished-print",
    status: "complete",
    result: {
      image: "cGl4ZWxz",
      filename: "teapot.png",
      model: (request.model as string | undefined) ?? sdxlModel.name,
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
   * The action's promise is "this picture again", and the picture was made by
   * the print's own saved request. Building the resubmission from the live
   * form instead made four of whatever the composer happened to be holding —
   * a different prompt, on a different style, at a different detail.
   */
  it("resubmits the print's own request, not whatever the form now says", async () => {
    const form = await mountWithAPrint();
    finishPrint();
    await flushPromises();
    form.prompt = "a different picture entirely";
    form.model = fluxModel.name;
    form.family = fluxModel.family;
    form.steps = 7;
    await flushPromises();
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    useUiStore().makeVariations();
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    const request = submit.mock.calls[0]![0];
    expect(request.prompt).toBe("a brass teapot on a rainy windowsill");
    expect(request.model).toBe(sdxlModel.name);
    expect(request.steps).toBe(30);
    expect(submit.mock.calls[0]![1]).toBe(4);
  });

  /**
   * A print made on this device carries `hostId: null`, and to placement
   * null means AUTOMATIC — any ready machine. Variations promise the print's
   * own machine, so the resubmit pins the primary rather than letting a
   * remote box be handed a LoRA path that only exists here.
   */
  it("pins this device for a print this device made", async () => {
    const form = await mountWithAPrint();
    finishPrint();
    await flushPromises();
    const feasible = vi.spyOn(useHostsStore(), "resolveFeasible");
    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    void form;

    useUiStore().makeVariations();
    await flushPromises();

    expect(feasible).toHaveBeenCalled();
    expect(feasible.mock.calls[0]![0]).toBe(useHostsStore().primaryHost?.id ?? "local");
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
   * A batch-locked recipe coerces the count to one, so the button would have
   * promised four pictures and made exactly one. The recipe that answers is
   * the PRINT'S, read from its own request and its own machine's contract.
   */
  it("is hidden for a print an edit recipe made", async () => {
    await mountWithAPrint(false);
    finishPrint({}, { model: editModel.name, edit_images: ["cGl4ZWxz"] });
    await flushPromises();
    expect(document.querySelector("[data-test='canvas-variations']")).toBeNull();
  });

  it("stays offered for a repeatable print while the composer holds an edit recipe", async () => {
    const form = await mountWithAPrint(false);
    finishPrint();
    form.family = editModel.family;
    form.model = editModel.name;
    await flushPromises();
    expect(document.querySelector("[data-test='canvas-variations']")).not.toBeNull();
  });

  /**
   * `job.request` is the print's recipe only when it carries everything the
   * print was made from. A job whose media authority lived OUTSIDE the request
   * (a same-host retained-media relay), a print restored from the gallery
   * with conditioning its snapshot never held, or a job recovered after a
   * restart under a placeholder prompt would repeat as four unrelated or
   * unconditioned pictures. Such a job is not repeatable, and says so.
   */
  it("is hidden for a job whose request cannot reproduce the print", async () => {
    await mountWithAPrint(false);
    finishPrint();
    await flushPromises();
    expect(document.querySelector("[data-test='canvas-variations']")).not.toBeNull();
    const shown = useGenerationStore().active!;
    shown.repeatable = false;
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
