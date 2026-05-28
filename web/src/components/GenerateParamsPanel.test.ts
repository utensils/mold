import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GenerateParamsPanel from "./GenerateParamsPanel.vue";
import type { GenerateFormState, ModelInfoExtended } from "../types";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    downloadsStreamUrl: vi.fn(() => "/api/downloads/stream"),
    fetchDownloads: vi.fn(async () => ({
      active: null,
      queued: [],
      history: [],
    })),
    fetchCatalogInstalled: vi.fn(async () => ({
      entries: [],
      page: 1,
      page_size: 0,
      total: 0,
    })),
  };
});

class MockEventSource {
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(_url: string) {}

  addEventListener(_type: string, _listener: EventListener) {}

  close() {}
}

vi.stubGlobal("EventSource", MockEventSource);

const baseModel: ModelInfoExtended = {
  name: "flux-dev:q4",
  family: "flux",
  size_gb: 12,
  is_loaded: false,
  last_used: null,
  hf_repo: "black-forest-labs/FLUX.1-dev",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 28,
  default_guidance: 3.5,
  description: "",
};

const altModel: ModelInfoExtended = {
  name: "sdxl:fp16",
  family: "sdxl",
  size_gb: 7,
  is_loaded: false,
  last_used: null,
  hf_repo: "stabilityai/stable-diffusion-xl-base-1.0",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 30,
  default_guidance: 7.5,
  description: "",
};

function makeForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  return {
    version: 2,
    model: "flux-dev:q4",
    modelFamily: "flux",
    prompt: "",
    negativePrompt: "",
    width: 1024,
    height: 1024,
    steps: 28,
    guidance: 3.5,
    seedMode: "random",
    seed: null,
    batchSize: 1,
    strength: 0.75,
    imageAttachments: [],
    maskImage: null,
    controlImage: null,
    controlModel: "",
    controlScale: 1,
    upscaleModel: "",
    gifPreview: false,
    audioFile: null,
    audioFilePath: "",
    sourceVideo: null,
    sourceVideoPath: "",
    keyframes: [],
    pipeline: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    scheduler: null,
    cfgPlus: false,
    frames: null,
    fps: null,
    outputFormat: "png",
    expand: { enabled: false, variations: 1, familyOverride: null },
    placement: null,
    loras: [],
    enableAudio: null,
    ...overrides,
  };
}

function mountPanel(
  form: GenerateFormState = makeForm(),
  models: ModelInfoExtended[] = [baseModel, altModel],
) {
  return mount(GenerateParamsPanel, {
    props: { modelValue: form, models },
  });
}

describe("GenerateParamsPanel", () => {
  beforeEach(() => {
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });

  it("renders a one-line summary when collapsed", () => {
    const w = mountPanel();
    const summary = w.find("[data-test='params-summary']");
    expect(summary.exists()).toBe(true);
    expect(summary.text()).toBe(
      "flux-dev:q4 · 1024×1024 · 28 · g 3.5 · seed random",
    );
    // Body is hidden when collapsed.
    expect(w.find("[data-test='params-body']").exists()).toBe(false);
    w.unmount();
  });

  it("formats a numeric seed in the summary", () => {
    const w = mountPanel(makeForm({ seedMode: "static", seed: 42 }));
    expect(w.find("[data-test='params-summary']").text()).toBe(
      "flux-dev:q4 · 1024×1024 · 28 · g 3.5 · seed static 42",
    );
    w.unmount();
  });

  it("expands when the summary header is clicked", async () => {
    const w = mountPanel();
    expect(w.find("[data-test='params-body']").exists()).toBe(false);
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    expect(w.find("[data-test='params-body']").exists()).toBe(true);
    w.unmount();
  });

  it("shows a dirty dot when steps deviates from the model default", () => {
    const clean = mountPanel(makeForm({ steps: 28 }));
    expect(clean.find("[data-test='params-dirty-dot']").exists()).toBe(false);
    clean.unmount();

    const dirty = mountPanel(makeForm({ steps: 50 }));
    expect(dirty.find("[data-test='params-dirty-dot']").exists()).toBe(true);
    dirty.unmount();
  });

  it("clears LoRAs when selecting a different model via ModelPicker", async () => {
    const form = makeForm({
      model: "flux-dev:q4",
      loras: [{ path: "/loras/foo.safetensors", scale: 1.0 }],
    });
    const w = mountPanel(form);
    // Expand so the ModelPicker mounts.
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    // Surface the panel's `selectModel` handler via the ModelPicker child's
    // emit. ModelPicker fires `select` with a ModelInfoExtended payload.
    const picker = w.findComponent({ name: "ModelPicker" });
    expect(picker.exists()).toBe(true);
    picker.vm.$emit("select", altModel);
    await w.vm.$nextTick();

    const events = w.emitted("update:modelValue");
    expect(events).toBeTruthy();
    const last = events![events!.length - 1][0] as GenerateFormState;
    expect(last.model).toBe("sdxl:fp16");
    expect(last.loras).toEqual([]);
    expect(last.modelFamily).toBe("sdxl");
    // Defaults from altModel are applied.
    expect(last.steps).toBe(30);
    expect(last.guidance).toBe(7.5);
  });

  it("hides strength for Qwen image edit even with a target attachment", async () => {
    const qwenEdit = {
      ...baseModel,
      name: "qwen-image-edit:q4",
      family: "qwen-image-edit",
    };
    const w = mountPanel(
      makeForm({
        model: "qwen-image-edit:q4",
        modelFamily: "qwen-image-edit",
        imageAttachments: [
          { kind: "upload", filename: "target.png", base64: "TARGET" },
        ],
      }),
      [qwenEdit],
    );

    await w.find("[data-test='params-summary-toggle']").trigger("click");

    expect(w.text()).not.toContain("Strength");
  });

  it("shows strength for non-edit img2img attachments", async () => {
    const w = mountPanel(
      makeForm({
        model: "sdxl:fp16",
        modelFamily: "sdxl",
        imageAttachments: [
          { kind: "upload", filename: "source.png", base64: "SOURCE" },
        ],
      }),
      [altModel],
    );

    await w.find("[data-test='params-summary-toggle']").trigger("click");

    expect(w.text()).toContain("Strength");
  });

  it("renders the LoRA picker for Z-Image models", async () => {
    const zImageModel: ModelInfoExtended = {
      ...baseModel,
      name: "z-image-turbo:q8",
      family: "z-image",
      hf_repo: "Tongyi-MAI/Z-Image-Turbo",
      default_steps: 9,
    };
    const w = mount(GenerateParamsPanel, {
      props: {
        modelValue: makeForm({
          model: "z-image-turbo:q8",
          steps: 9,
        }),
        models: [zImageModel],
      },
      global: {
        stubs: {
          LoraPicker: {
            props: ["family", "modelValue"],
            template: '<div data-test="lora-picker-stub" />',
          },
        },
      },
    });

    await w.find("[data-test='params-summary-toggle']").trigger("click");

    expect(w.find("[data-test='lora-picker-stub']").exists()).toBe(true);
    w.unmount();
  });

  it("persists expanded state to localStorage and restores it on remount", async () => {
    const w = mountPanel();
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    expect(localStorage.getItem("mold.generate.params.expanded")).toBe("true");
    w.unmount();

    // Remount: localStorage value should drive initial expanded state.
    const w2 = mountPanel();
    expect(w2.find("[data-test='params-body']").exists()).toBe(true);
    w2.unmount();
  });
});
