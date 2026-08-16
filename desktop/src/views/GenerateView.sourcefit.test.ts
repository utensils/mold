/**
 * Submit-path integration for the img2img source-fit policies: the view must
 * resolve the CONCRETE target host before host-bound upscale preprocessing,
 * while local fit policies preprocess first so placement sees the finalized
 * request exactly once.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import GenerateView from "./GenerateView.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useGenerationStore } from "../stores/generation";
import { useModelStore } from "../stores/models";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useToastStore } from "../stores/toasts";
import { useUiStore } from "../stores/ui";
import { useComposerStore } from "../stores/composer";
import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import type { SourceFitInput } from "../lib/sourceFitPreprocess";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
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
const { sourceStashGet } = vi.hoisted(() => ({ sourceStashGet: vi.fn() }));
vi.mock("../lib/ipc", () => ({ ipc: { sourceStashGet } }));
vi.mock("../lib/api/history", () => ({ fetchHistory: vi.fn(() => Promise.resolve([])) }));

const upscaleImage = vi.fn();
vi.mock("../lib/api/upscale", () => ({
  upscaleImage: (...args: unknown[]) => upscaleImage(...args),
}));

interface PreprocessDeps {
  upscale?: (image: string, model: string) => Promise<string>;
  cache?: unknown;
  onStatus?: (message: string) => void;
}
const applySourceFitPreprocess = vi.fn();
vi.mock("../lib/sourceFitPreprocess", () => ({
  applySourceFitPreprocess: (...args: unknown[]) => applySourceFitPreprocess(...args),
}));

const model: ModelEntry = {
  name: "sd15:fp16",
  family: "sd15",
  downloaded: true,
  default_width: 512,
  default_height: 512,
  default_steps: 20,
  default_guidance: 7,
} as ModelEntry;

function setupMultiHost() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
  conn.status = "ready";
  const hosts = useHostsStore();
  hosts.initialized = true;
  hosts.extras.push({
    id: "hal9000-7680",
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: "remote-key",
    status: "ready",
    error: null,
    instanceId: null,
  });
  return hosts;
}

function primeForm() {
  const form = useGenerateFormStore().form;
  form.prompt = "a lighthouse";
  form.model = "sd15:fp16";
  form.family = "sd15";
  form.width = 512;
  form.height = 512;
  form.sourceImage = "SRC";
  form.sourceFit = {
    mode: "upscale-then-fit",
    upscalerModel: "real-esrgan-x2plus:fp16",
    fit: { mode: "pad-repaint" },
  };
  return form;
}

describe("GenerateView source-fit submit path", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    apiJsonTo.mockReset();
    apiJsonTo.mockResolvedValue([model]);
    upscaleImage.mockReset();
    upscaleImage.mockResolvedValue("UPSCALED");
    applySourceFitPreprocess.mockReset();
    sourceStashGet.mockReset();
    useModelStore().all = [model];
  });
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("preserves an attached source and its fit policy when Create remounts", async () => {
    const form = primeForm();
    form.sourceImageName = "camera.png";
    form.sourceFit = { mode: "crop-fill", alignX: "right", alignY: "bottom" };

    const first = mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    first.unmount();

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();

    expect(form.sourceImage).toBe("SRC");
    expect(form.sourceImageName).toBe("camera.png");
    expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "right", alignY: "bottom" });
  });

  it("preserves the fit policy while Reuse settings restores source bytes", async () => {
    const form = useGenerateFormStore().form;
    form.sourceFit = { mode: "crop-fill", alignX: "right", alignY: "bottom" };
    sourceStashGet.mockResolvedValue("RESTORED");
    useComposerStore().set({
      metadata: {
        prompt: "a camera",
        model: model.name,
        seed: 42,
        steps: 20,
        guidance: 7,
        width: 512,
        height: 512,
        generation_width: 768,
        generation_height: 1344,
        version: "test",
        source_image_sha256: "a".repeat(64),
        source_image_name: "camera.png",
      } as OutputMetadata,
    });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();

    expect(sourceStashGet).toHaveBeenCalledWith("a".repeat(64));
    expect(form.sourceImage).toBe("RESTORED");
    expect(form.sourceImageName).toBe("camera.png");
    expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "right", alignY: "bottom" });
    expect(form.width).toBe(768);
    expect(form.height).toBe(1344);
  });

  it("resolves the host before preprocessing and routes the upscale to it", async () => {
    const hosts = setupMultiHost();
    const resolveFeasible = vi.spyOn(hosts, "resolveFeasible");
    const generation = useGenerationStore();
    const submitBatch = vi
      .spyOn(generation, "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    applySourceFitPreprocess.mockImplementation(
      async (_input: SourceFitInput, deps: PreprocessDeps) => {
        // Drive the injected upscale fn so the host binding is observable.
        await deps.upscale?.("SRC", "real-esrgan-x2plus:fp16");
        return { source: "FIT", mask: "PADMASK", changed: true };
      },
    );

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm();
    // Let the model-change watcher settle before selecting the host-bound
    // policy, matching the user's interaction order.
    await flushPromises();
    form.sourceFit = {
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x2plus:fp16",
      fit: { mode: "pad-repaint" },
    };
    useUiStore().generateTick++;
    await flushPromises();

    // Host resolution happened BEFORE preprocessing (same-host invariant).
    expect(resolveFeasible).toHaveBeenCalled();
    expect(applySourceFitPreprocess).toHaveBeenCalled();
    expect(Math.min(...resolveFeasible.mock.invocationCallOrder)).toBeLessThan(
      Math.min(...applySourceFitPreprocess.mock.invocationCallOrder),
    );

    // The upscale hit the SAME host the generation routed to.
    expect(submitBatch).toHaveBeenCalledTimes(1);
    const [req, , batchRoute] = submitBatch.mock.calls[0]!;
    expect(batchRoute).toBeTruthy();
    if (!batchRoute) throw new Error("unreachable: submit did not retain its route");
    expect(upscaleImage).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "real-esrgan-x2plus:fp16",
        image: "SRC",
        target: batchRoute.target,
      }),
    );
    expect(req.source_image).toBe("FIT");
    expect(req.mask_image).toBe("PADMASK");
  });

  it("preprocesses local fit policies before one finalized placement preview", async () => {
    const hosts = setupMultiHost();
    const resolveFeasible = vi.spyOn(hosts, "resolveFeasible");
    const submitBatch = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    applySourceFitPreprocess.mockResolvedValue({
      source: "FIT",
      mask: "PADMASK",
      changed: true,
    });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm();
    form.sourceFit = { mode: "lanczos-resize" };
    useUiStore().generateTick++;
    await flushPromises();

    expect(applySourceFitPreprocess).toHaveBeenCalledTimes(1);
    expect(resolveFeasible).toHaveBeenCalledTimes(1);
    expect(Math.min(...applySourceFitPreprocess.mock.invocationCallOrder)).toBeLessThan(
      Math.min(...resolveFeasible.mock.invocationCallOrder),
    );
    expect(submitBatch).toHaveBeenCalledTimes(1);
    expect(submitBatch.mock.calls[0]![0]).toMatchObject({
      source_image: "FIT",
      mask_image: "PADMASK",
    });
  });

  it("passes the form's source, mask, policy, and target size to the preprocess", async () => {
    setupMultiHost();
    const generation = useGenerationStore();
    vi.spyOn(generation, "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    applySourceFitPreprocess.mockResolvedValue({ source: "FIT", mask: null, changed: true });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm();
    form.maskImage = "USERMASK";
    useUiStore().generateTick++;
    await flushPromises();

    const [input] = applySourceFitPreprocess.mock.calls[0]! as [SourceFitInput];
    expect(input).toMatchObject({
      source: "SRC",
      mask: "USERMASK",
      policy: form.sourceFit,
      target: { width: 512, height: 512 },
    });
    // Processed bytes belong to the request draft; the editable source stays original.
    expect(form.sourceImage).toBe("SRC");
    expect(form.maskImage).toBe("USERMASK");
  });

  it("reuses one per-composer cache across repeat submits", async () => {
    setupMultiHost();
    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    applySourceFitPreprocess.mockImplementation(async (input: SourceFitInput) => ({
      source: input.source,
      mask: input.mask,
      changed: false,
    }));

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    primeForm();
    useUiStore().generateTick++;
    await flushPromises();
    useUiStore().generateTick++;
    await flushPromises();

    expect(applySourceFitPreprocess).toHaveBeenCalledTimes(2);
    const firstDeps = applySourceFitPreprocess.mock.calls[0]![1] as PreprocessDeps;
    const secondDeps = applySourceFitPreprocess.mock.calls[1]![1] as PreprocessDeps;
    expect(firstDeps.cache).toBeDefined();
    expect(secondDeps.cache).toBe(firstDeps.cache);
  });

  it("submits the tapped form snapshot when preprocessing is still running", async () => {
    setupMultiHost();
    const submitBatch = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    let finishPreprocess!: () => void;
    applySourceFitPreprocess.mockImplementation(
      () =>
        new Promise((resolve) => {
          finishPreprocess = () => resolve({ source: "FIT", mask: null, changed: true });
        }),
    );

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm();
    // Batch N now requires a reviewed prepared expansion set. This regression
    // isolates the ordinary Batch 1 source-preprocess snapshot boundary.
    form.batchSize = 1;
    useUiStore().generateTick++;
    await flushPromises();
    expect(applySourceFitPreprocess).toHaveBeenCalledTimes(1);

    form.prompt = "a different live prompt";
    form.batchSize = 3;
    finishPreprocess();
    await flushPromises();

    expect(submitBatch).toHaveBeenCalledTimes(1);
    const [request, batch] = submitBatch.mock.calls[0]!;
    expect(request).toMatchObject({ prompt: "a lighthouse", source_image: "FIT" });
    expect(batch).toBe(1);
  });

  it("does not overwrite a mask or fit policy edited while preprocessing", async () => {
    setupMultiHost();
    const submitBatch = vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    });
    let finishPreprocess!: () => void;
    applySourceFitPreprocess.mockImplementation(
      () =>
        new Promise((resolve) => {
          finishPreprocess = () => resolve({ source: "FIT", mask: "OLD-PAD-MASK", changed: true });
        }),
    );

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm();
    form.maskImage = "OLD-MASK";
    useUiStore().generateTick++;
    await flushPromises();

    form.maskImage = "NEW-MASK";
    form.sourceFit = { mode: "crop-fill", alignX: "right", alignY: "bottom" };
    finishPreprocess();
    await flushPromises();

    expect(form.sourceImage).toBe("SRC");
    expect(form.maskImage).toBe("NEW-MASK");
    expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "right", alignY: "bottom" });
    expect(submitBatch).toHaveBeenCalledTimes(1);
    expect(submitBatch.mock.calls[0]?.[0]).toMatchObject({
      source_image: "FIT",
      mask_image: "OLD-PAD-MASK",
    });
  });

  it("skips preprocessing when no source image is attached", async () => {
    setupMultiHost();
    const generation = useGenerationStore();
    const submitBatch = vi
      .spyOn(generation, "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm();
    form.sourceImage = null;
    useUiStore().generateTick++;
    await flushPromises();

    expect(applySourceFitPreprocess).not.toHaveBeenCalled();
    expect(submitBatch).toHaveBeenCalledTimes(1);
  });

  it("aborts the submit and toasts when preprocessing fails", async () => {
    setupMultiHost();
    const generation = useGenerationStore();
    const submitBatch = vi
      .spyOn(generation, "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });
    applySourceFitPreprocess.mockRejectedValue(new Error("unknown upscaler model"));

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    primeForm();
    useUiStore().generateTick++;
    await flushPromises();

    expect(submitBatch).not.toHaveBeenCalled();
    const toasts = useToastStore();
    expect(JSON.stringify(toasts.$state)).toContain("unknown upscaler model");
  });
});
