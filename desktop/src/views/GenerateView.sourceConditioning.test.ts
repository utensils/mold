/**
 * Per-model source-image conditioning (#772) and the wan first/last-frame
 * layout riding on it (#779), as the Create view enforces them: Generate stays
 * gated while the attached conditioning would be refused, and a reused
 * first/last-frame print says out loud that its closing still cannot come back.
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
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { useUiStore } from "../stores/ui";
import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import ComposerCard from "../components/create/ComposerCard.vue";

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
  return { ...original, previewGenerationPlacement: planned, previewChainPlacement: planned };
});
const { sourceStashGet } = vi.hoisted(() => ({ sourceStashGet: vi.fn() }));
vi.mock("../lib/ipc", () => ({ ipc: { sourceStashGet } }));
vi.mock("../lib/api/history", () => ({ fetchHistory: vi.fn(() => Promise.resolve([])) }));
// jsdom has no canvas; source fitting is not what these tests are about, so
// the preprocessor passes the attached bytes straight through.
vi.mock("../lib/sourceFitPreprocess", () => ({
  applySourceFitPreprocess: vi.fn((input: { source: string; mask: string | null }) =>
    Promise.resolve({ source: input.source, mask: input.mask, changed: false }),
  ),
}));

function wanModel(name: string, sourceImage?: string): ModelEntry {
  return {
    name,
    family: "wan",
    downloaded: true,
    default_width: 1280,
    default_height: 720,
    default_steps: 20,
    default_guidance: 5,
    default_frames: 81,
    ...(sourceImage === undefined ? {} : { source_image: sourceImage }),
  } as ModelEntry;
}

function primeForm(model: ModelEntry) {
  const form = useGenerateFormStore().form;
  form.prompt = "a lighthouse at dusk";
  form.model = model.name;
  form.family = "wan";
  form.width = 1280;
  form.height = 720;
  form.frames = 81;
  form.sourceImageCapability = model.source_image ?? null;
  return form;
}

function readyConnection() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
  conn.status = "ready";
}

describe("GenerateView source-conditioning gating (#772, #779)", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    sourceStashGet.mockReset();
    readyConnection();
  });
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("parks a source image while submitting an advertised text-to-video checkpoint", async () => {
    const model = wanModel("wan22-t2v-a14b", "unsupported");
    useModelStore().all = [model];
    // A print routes from inventory now, not from a placement preview.
    useHostModelsStore().byHost.local = {
      entries: [model],
      fetchedAt: Date.now(),
      error: null,
    };
    const submitBatch = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm(model);
    form.sourceImage = "PARKED";
    form.sourceImageName = "previous.png";
    await flushPromises();

    useUiStore().generateTick++;
    await flushPromises();
    expect(submitBatch).toHaveBeenCalledTimes(1);
    expect(submitBatch.mock.calls[0]![0].source_image).toBeUndefined();
    expect(submitBatch.mock.calls[0]![0].source_image_name).toBeUndefined();
    expect(form.sourceImage).toBe("PARKED");
    expect(form.sourceImageName).toBe("previous.png");

    const imageModel = wanModel("wan22-i2v-a14b", "optional");
    form.model = imageModel.name;
    form.sourceImageCapability = imageModel.source_image ?? null;
    await flushPromises();
    expect(form.sourceImage).toBe("PARKED");
    expect(form.sourceImageName).toBe("previous.png");

    useUiStore().generateTick++;
    await flushPromises();
    expect(submitBatch).toHaveBeenCalledTimes(2);
    expect(submitBatch.mock.calls[1]![0].source_image).toBe("PARKED");
    expect(submitBatch.mock.calls[1]![0].source_image_name).toBe("previous.png");
  });

  it("blocks Generate until a required source image is attached", async () => {
    const model = wanModel("wan22-i2v-a14b", "required");
    useModelStore().all = [model];
    // A print routes from inventory now, not from a placement preview.
    useHostModelsStore().byHost.local = {
      entries: [model],
      fetchedAt: Date.now(),
      error: null,
    };
    const submitBatch = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm(model);
    await flushPromises();

    useUiStore().generateTick++;
    await flushPromises();
    expect(submitBatch).not.toHaveBeenCalled();

    form.sourceImage = "SRC";
    await flushPromises();
    useUiStore().generateTick++;
    await flushPromises();
    expect(submitBatch).toHaveBeenCalledTimes(1);
  });

  it("shows the required H3 first frame before the user starts typing", async () => {
    const model = {
      ...wanModel("minimax-h3-fl2va:comfy-pruned-int8", "required"),
      family: "minimax-h3",
      default_width: 1344,
      default_height: 768,
      default_steps: 21,
      default_frames: 124,
      default_fps: 24,
    } as ModelEntry;
    useModelStore().all = [model];
    // A print routes from inventory now, not from a placement preview.
    useHostModelsStore().byHost.local = {
      entries: [model],
      fetchedAt: Date.now(),
      error: null,
    };
    const wrapper = mount(GenerateView, {
      shallow: true,
      attachTo: document.body,
    });
    await flushPromises();
    const form = useGenerateFormStore().form;
    form.prompt = "";
    form.model = model.name;
    form.family = model.family;
    form.sourceImageCapability = "required";
    await flushPromises();

    expect(wrapper.getComponent(ComposerCard).props("disabledReason")).toBe(
      "This reviewed MiniMax H3 runtime requires a first frame.",
    );
  });

  it("blocks an end-frame-only draft until a first frame joins it", async () => {
    const model = wanModel("wan22-flf2v-a14b", "optional");
    useModelStore().all = [model];
    // A print routes from inventory now, not from a placement preview.
    useHostModelsStore().byHost.local = {
      entries: [model],
      fetchedAt: Date.now(),
      error: null,
    };
    const submitBatch = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = primeForm(model);
    form.endFrame = { filename: "close.png", base64: "ENDB64" };
    await flushPromises();

    useUiStore().generateTick++;
    await flushPromises();
    expect(submitBatch).not.toHaveBeenCalled();

    form.sourceImage = "SRC";
    form.sourceImageName = "open.png";
    await flushPromises();
    useUiStore().generateTick++;
    await flushPromises();
    expect(submitBatch).toHaveBeenCalledTimes(1);
    const [request] = submitBatch.mock.calls[0]!;
    expect(request.keyframes).toEqual([
      { frame: 0, image: "SRC", name: "open.png" },
      { frame: 80, image: "ENDB64", name: "close.png" },
    ]);
    // The engine refuses `source_image` + `keyframes` together ("not both").
    expect(request.source_image).toBeUndefined();
  });

  it("says the end frame cannot be restored when reusing a first/last-frame print", async () => {
    const model = wanModel("wan22-flf2v-a14b", "optional");
    useModelStore().all = [model];
    // A print routes from inventory now, not from a placement preview.
    useHostModelsStore().byHost.local = {
      entries: [model],
      fetchedAt: Date.now(),
      error: null,
    };
    sourceStashGet.mockResolvedValue("RESTORED");
    useComposerStore().set({
      metadata: {
        prompt: "a lighthouse at dusk",
        model: model.name,
        seed: 7,
        steps: 20,
        guidance: 5,
        width: 1280,
        height: 720,
        version: "test",
        source_image_sha256: "a".repeat(64),
        source_image_name: "open.png",
        keyframes: [
          { frame: 0, name: "open.png", sha256: "a".repeat(64) },
          { frame: 80, name: "close.png", sha256: "b".repeat(64) },
        ],
      } as OutputMetadata,
    });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();

    const notice = useToastStore().items.find((t) => t.message.includes("end frame"));
    expect(notice?.message).toContain("close.png");
    expect(useGenerateFormStore().form.endFrame).toBeNull();
  });

  it("stays silent about an end frame for an ordinary single-source print", async () => {
    const model = wanModel("wan22-flf2v-a14b", "optional");
    useModelStore().all = [model];
    // A print routes from inventory now, not from a placement preview.
    useHostModelsStore().byHost.local = {
      entries: [model],
      fetchedAt: Date.now(),
      error: null,
    };
    sourceStashGet.mockResolvedValue("RESTORED");
    useComposerStore().set({
      metadata: {
        prompt: "a lighthouse at dusk",
        model: model.name,
        seed: 7,
        steps: 20,
        guidance: 5,
        width: 1280,
        height: 720,
        version: "test",
        source_image_sha256: "a".repeat(64),
        source_image_name: "open.png",
      } as OutputMetadata,
    });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();

    expect(useToastStore().items.some((t) => t.message.includes("end frame"))).toBe(false);
  });
});
