/**
 * Submit-path integration for the img2img source-fit policies: the view must
 * resolve the CONCRETE target host BEFORE preprocessing (so upscale-then-fit
 * runs the upscaler on the same host the generation routes to), apply the
 * preprocessed source/mask to the form, and only then build + submit.
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
import type { ModelEntry } from "../lib/api/types";
import type { SourceFitInput } from "../lib/sourceFitPreprocess";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));
vi.mock("../lib/api/history", () => ({ fetchHistory: vi.fn(() => Promise.resolve([])) }));

const upscaleImage = vi.fn();
vi.mock("../lib/api/upscale", () => ({
  upscaleImage: (...args: unknown[]) => upscaleImage(...args),
}));

interface PreprocessDeps {
  upscale?: (image: string, model: string) => Promise<string>;
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
    useModelStore().all = [model];
  });
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("resolves the host before preprocessing and routes the upscale to it", async () => {
    const hosts = setupMultiHost();
    const resolveRoute = vi.spyOn(hosts, "resolveRoute");
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
    primeForm();
    useUiStore().generateTick++;
    await flushPromises();

    // Host resolution happened BEFORE preprocessing (same-host invariant).
    expect(resolveRoute).toHaveBeenCalled();
    expect(applySourceFitPreprocess).toHaveBeenCalled();
    expect(Math.min(...resolveRoute.mock.invocationCallOrder)).toBeLessThan(
      Math.min(...applySourceFitPreprocess.mock.invocationCallOrder),
    );

    // The upscale hit the SAME host the generation routed to.
    const route = resolveRoute.mock.results.find((r) => r.type === "return")!.value;
    expect(route).not.toBeNull();
    if (!route) throw new Error("unreachable: resolveRoute returned null");
    expect(upscaleImage).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "real-esrgan-x2plus:fp16",
        image: "SRC",
        target: route.target,
      }),
    );
    expect(submitBatch).toHaveBeenCalledTimes(1);
    const [req, , batchRoute] = submitBatch.mock.calls[0]!;
    expect(req.source_image).toBe("FIT");
    expect(req.mask_image).toBe("PADMASK");
    expect(batchRoute).toEqual(route);
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
    // The preprocess result replaced the form's images (visible in the well).
    expect(form.sourceImage).toBe("FIT");
    expect(form.maskImage).toBeNull();
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
