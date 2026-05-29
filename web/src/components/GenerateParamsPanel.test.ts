import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GenerateParamsPanel from "./GenerateParamsPanel.vue";
import type { GenerateFormState, ModelInfoExtended } from "../types";
import { fetchCatalogInstalled } from "../api";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    fetchGenerationEstimate: vi.fn(async () => ({
      model: "flux-dev:q4",
      peak_memory_bytes: 12_500_000_000,
      activation_memory_bytes: 500_000_000,
      available_memory_bytes: 24_000_000_000,
      load_strategy: "sequential",
      fits_available_memory: true,
    })),
    fetchModelComponents: vi.fn(async () => ({
      model: "flux-dev:q4",
      components: [
        {
          kind: "transformer",
          name: "transformer",
          present: true,
          path: "/models/flux/transformer.safetensors",
          repair_model: "flux-dev:q4",
          options: [
            {
              label: "transformer.safetensors",
              path: "/models/flux/transformer.safetensors",
              present: true,
            },
          ],
        },
        {
          kind: "vae",
          name: "vae",
          present: false,
          path: "/models/flux/vae.safetensors",
          repair_model: "flux-dev:q4",
          options: [
            {
              label: "vae.safetensors",
              path: "/models/flux/vae.safetensors",
              present: false,
            },
            {
              label: "alternate-vae.safetensors",
              path: "/models/flux/alternate-vae.safetensors",
              present: true,
            },
          ],
        },
        {
          kind: "clip",
          name: "clip encoder",
          present: true,
          path: "/models/flux/clip_l.safetensors",
          repair_model: "flux-dev:q4",
          options: [
            {
              label: "clip_l.safetensors",
              path: "/models/flux/clip_l.safetensors",
              present: true,
            },
          ],
        },
        {
          kind: "clip",
          name: "clip-g encoder",
          present: true,
          path: "/models/flux/clip_g.safetensors",
          repair_model: "flux-dev:q4",
          options: [
            {
              label: "clip_g.safetensors",
              path: "/models/flux/clip_g.safetensors",
              present: true,
            },
          ],
        },
      ],
    })),
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

const upscalerModel: ModelInfoExtended = {
  name: "real-esrgan-x4plus:fp16",
  family: "upscaler",
  size_gb: 0.08,
  is_loaded: false,
  last_used: null,
  hf_repo: "ai-forever/Real-ESRGAN",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 1,
  default_guidance: 1,
  description: "Real-ESRGAN 4x upscaler",
};

const controlNetModel: ModelInfoExtended = {
  name: "controlnet-canny-sd15:fp16",
  family: "controlnet",
  size_gb: 0.7,
  is_loaded: false,
  last_used: null,
  hf_repo: "lllyasviel/control_v11p_sd15_canny",
  downloaded: true,
  default_width: 512,
  default_height: 512,
  default_steps: 25,
  default_guidance: 7.5,
  description: "ControlNet Canny edge detection for SD1.5",
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
    props: {
      modelValue: form,
      models,
      placementGpus: [{ ordinal: 0, name: "GPU 0" }],
    },
    global: {
      stubs: {
        MaskEditorModal: {
          props: ["open", "sourceImage"],
          emits: ["apply", "close"],
          template:
            "<div v-if=\"open\" data-test=\"mask-editor-stub\"><button data-test=\"apply-mask-edit\" @click=\"$emit('apply', { kind: 'upload', filename: 'edited-mask.png', base64: 'EDITED_MASK' })\">apply</button></div>",
        },
        ImagePickerModal: {
          props: ["open", "title", "multiple"],
          emits: ["pick", "close"],
          template:
            '<div v-if="open" data-test="mask-base-picker-stub" :data-title="title" :data-multiple="String(multiple)"><button data-test="stub-pick-mask-base" @click="$emit(\'pick\', [{ kind: \'gallery\', filename: \'gallery-base.png\', base64: \'GALLERY_BASE\' }])">pick</button></div>',
        },
      },
    },
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

  it("keeps direct mask upload available and stores the uploaded mask", async () => {
    const w = mountPanel();
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    const input = w.get("[data-test='mask-upload']")
      .element as HTMLInputElement;
    Object.defineProperty(input, "files", {
      value: [new File(["mask"], "mask.png", { type: "image/png" })],
    });
    await w.get("[data-test='mask-upload']").trigger("change");
    await flushPromises();

    const events = w.emitted("update:modelValue");
    expect(events).toBeTruthy();
    const last = events![events!.length - 1][0] as GenerateFormState;
    expect(last.maskImage?.filename).toBe("mask.png");
    expect(last.maskImage?.base64).toMatch(/^[A-Za-z0-9+/]+=*$/);
  });

  it("opens the mask editor from the current source image and applies the edited mask", async () => {
    const w = mountPanel(
      makeForm({
        imageAttachments: [
          { kind: "upload", filename: "source.png", base64: "SOURCE" },
        ],
      }),
    );
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    await w.get("[data-test='edit-source-mask']").trigger("click");
    expect(w.find("[data-test='mask-editor-stub']").exists()).toBe(true);

    await w.get("[data-test='apply-mask-edit']").trigger("click");

    const events = w.emitted("update:modelValue");
    expect(events).toBeTruthy();
    const last = events![events!.length - 1][0] as GenerateFormState;
    expect(last.maskImage).toEqual({
      kind: "upload",
      filename: "edited-mask.png",
      base64: "EDITED_MASK",
    });
  });

  it("disables source-based mask editing until a source image is selected", async () => {
    const w = mountPanel();
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    const button = w.get("[data-test='edit-source-mask']");
    expect((button.element as HTMLButtonElement).disabled).toBe(true);
  });

  it("opens a single-image picker for mask editor base images", async () => {
    const w = mountPanel();
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    await w.get("[data-test='pick-mask-base']").trigger("click");
    const picker = w.get("[data-test='mask-base-picker-stub']");
    expect(picker.attributes("data-title")).toBe("Mask base image");
    expect(picker.attributes("data-multiple")).toBe("false");
  });

  it("hides mask controls for Qwen image edit", async () => {
    const qwenEdit = {
      ...baseModel,
      name: "qwen-image-edit:q4",
      family: "qwen-image-edit",
    };
    const w = mountPanel(
      makeForm({ model: "qwen-image-edit:q4", modelFamily: "qwen-image-edit" }),
      [qwenEdit],
    );

    await w.find("[data-test='params-summary-toggle']").trigger("click");

    expect(w.find("[data-test='mask-upload']").exists()).toBe(false);
    expect(w.find("[data-test='edit-source-mask']").exists()).toBe(false);
    expect(w.find("[data-test='pick-mask-base']").exists()).toBe(false);
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

  it("renders installed ControlNet choices only for SD1.5-capable models", async () => {
    vi.mocked(fetchCatalogInstalled).mockResolvedValueOnce({
      entries: [
        {
          id: "cv:8101",
          source: "civitai",
          source_id: "8101",
          name: "Catalog Canny",
          author: "alice",
          family: "sd15",
          family_role: "finetune",
          sub_family: null,
          modality: "image",
          kind: "control-net",
          file_format: "safetensors",
          bundling: "single-file",
          size_bytes: 100,
          download_count: 0,
          rating: null,
          likes: 0,
          nsfw: false,
          thumbnail_url: null,
          description: null,
          license: null,
          license_flags: null,
          tags: [],
          companions: [],
          download_recipe: { files: [], needs_token: null },
          engine_phase: 1,
          installed: true,
          primary_path: "/models/controlnet/catalog-canny.safetensors",
          created_at: null,
          updated_at: null,
          added_at: 0,
          trained_words: [],
        },
      ],
      page: 1,
      page_size: 1,
      total: 1,
    });

    const w = mountPanel(
      makeForm({
        model: "sd15:fp16",
        modelFamily: "sd15",
        controlImage: {
          kind: "upload",
          filename: "control.png",
          base64: "CONTROL",
        },
      }),
      [{ ...baseModel, name: "sd15:fp16", family: "sd15" }, controlNetModel],
    );
    await flushPromises();
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    expect(fetchCatalogInstalled).toHaveBeenCalledWith({
      family: "sd15",
      kind: "control-net",
    });
    const select = w.get("[data-test='controlnet-select']");
    expect(select.text()).toContain("controlnet-canny-sd15:fp16");
    expect(select.text()).toContain("Catalog Canny");

    const flux = mountPanel(makeForm({ modelFamily: "flux" }), [
      baseModel,
      controlNetModel,
    ]);
    await flushPromises();
    await flux.find("[data-test='params-summary-toggle']").trigger("click");
    expect(flux.find("[data-test='control-upload']").exists()).toBe(false);

    w.unmount();
    flux.unmount();
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

  it("renders compact generation template controls in the expanded panel", async () => {
    const w = mountPanel(makeForm({ prompt: "template source" }));
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    expect(w.find("[data-test='generation-templates']").exists()).toBe(true);
    await w.find("[data-test='template-name']").setValue("Base");
    await w.find("[data-test='template-save']").trigger("click");

    expect(localStorage.getItem("mold.generation.templates.v1")).toContain(
      "template source",
    );
    w.unmount();
  });

  it("shows scheduler controls only for scheduler-capable families", async () => {
    const flux = mountPanel(
      makeForm({ model: "flux-dev:q4", modelFamily: "flux" }),
    );
    await flux.find("[data-test='params-summary-toggle']").trigger("click");
    expect(flux.find("[data-test='scheduler-select']").exists()).toBe(false);
    flux.unmount();
    localStorage.clear();

    const sdxl = mountPanel(
      makeForm({ model: "sdxl:fp16", modelFamily: "sdxl" }),
      [altModel],
    );
    await sdxl.find("[data-test='params-summary-toggle']").trigger("click");
    await sdxl.find("[data-test='advanced-toggle']").trigger("click");
    const labels = sdxl
      .findAll("[data-test='scheduler-select'] option")
      .map((option) => option.text());
    expect(labels).toEqual(["default", "ddim", "euler-ancestral", "unipc"]);
    sdxl.unmount();
  });

  it("renders seed mode as a stable segmented control with non-truncating labels", async () => {
    const w = mountPanel();
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    const control = w.get("[data-test='seed-mode-control']");
    expect(control.classes().join(" ")).toContain(
      "grid-cols-[repeat(3,minmax(0,1fr))]",
    );
    const labels = w
      .findAll("[data-test='seed-mode-option']")
      .map((button) => button.text());
    expect(labels).toEqual(["random", "static", "increment"]);
    for (const button of w.findAll("[data-test='seed-mode-option']")) {
      expect(button.classes()).toContain("min-w-0");
      expect(button.classes()).toContain("whitespace-nowrap");
    }
    w.unmount();
  });

  it("uses installed upscaler models for the upscaler dropdown", async () => {
    const w = mountPanel(makeForm(), [baseModel, upscalerModel]);
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    const select = w.get("[data-test='upscaler-select']");
    const labels = select.findAll("option").map((option) => option.text());
    expect(labels).toEqual(["None", "real-esrgan-x4plus:fp16"]);

    await select.setValue("real-esrgan-x4plus:fp16");
    const events = w.emitted("update:modelValue");
    const last = events![events!.length - 1][0] as GenerateFormState;
    expect(last.upscaleModel).toBe("real-esrgan-x4plus:fp16");
    w.unmount();
  });

  it("does not render a separate Device placement section in Controls", async () => {
    const w = mountPanel(makeForm(), [baseModel]);
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    await vi.dynamicImportSettled();

    expect(w.find("[data-test='placement-section-toggle']").exists()).toBe(
      false,
    );
    expect(w.find("[data-test='component-status']").text()).not.toContain(
      "Device placement",
    );
    w.unmount();
  });

  it("shows component status, omits transformer asset dropdown, and keeps CLIP rows distinct", async () => {
    const w = mountPanel(makeForm(), [baseModel]);
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    await vi.dynamicImportSettled();

    expect(w.find("[data-test='memory-estimate']").text()).toContain("12.5 GB");
    expect(w.find("[data-test='component-status']").text()).toContain(
      "transformer",
    );
    expect(w.find("[data-test='component-status']").text()).toContain(
      "Missing",
    );
    expect(w.find("[data-test='component-catalog-link']").exists()).toBe(true);
    expect(
      w.find("[data-test='component-repair-link']").attributes("href"),
    ).toBe("/catalog?q=flux-dev%3Aq4");
    const rows = w.findAll("[data-test='component-row']");
    expect(rows.map((row) => row.text())).toEqual(
      expect.arrayContaining([
        expect.stringContaining("transformer"),
        expect.stringContaining("vae"),
        expect.stringContaining("clip encoder"),
        expect.stringContaining("clip-g encoder"),
      ]),
    );
    const transformerRow = rows.find((row) =>
      row.text().includes("transformer"),
    );
    expect(
      transformerRow?.find("[data-test='component-option-select']").exists(),
    ).toBe(false);
    expect(w.findAll("[data-test='component-option-select']")).toHaveLength(3);
    expect(
      w.findAll("[data-test='component-option-select']")[0].text(),
    ).toContain("alternate-vae.safetensors");
    expect(w.findAll("[data-test='component-placement-select']")).toHaveLength(
      3,
    );
    w.unmount();
  });

  it("renders estimated peak memory as a used over available progress bar", async () => {
    const w = mountPanel(makeForm(), [baseModel]);
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    await vi.dynamicImportSettled();

    const bar = w.get("[data-test='memory-estimate-bar']");
    expect(bar.attributes("aria-valuenow")).toBe("52");
    expect(bar.attributes("aria-valuemax")).toBe("100");
    expect(bar.text()).toContain("52%");
    expect(bar.text()).toContain("12.5 GB used");
    expect(bar.text()).toContain("24.0 GB available");
    expect(
      w.get("[data-test='memory-estimate-bar-fill']").attributes("style"),
    ).toContain("width: 52.1%");
    w.unmount();
  });

  it("updates placement from component-row pin controls", async () => {
    const w = mountPanel(makeForm(), [baseModel]);
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    await vi.dynamicImportSettled();

    const rows = w.findAll("[data-test='component-row']");
    const vaeRow = rows.find((row) => row.text().includes("vae"));
    expect(vaeRow).toBeTruthy();

    await vaeRow!
      .get("[data-test='component-placement-select']")
      .setValue("cpu");

    const events = w.emitted("update:modelValue");
    expect(events).toBeTruthy();
    const last = events![events!.length - 1][0] as GenerateFormState;
    expect(last.placement).toEqual({
      text_encoders: { kind: "auto" },
      advanced: {
        transformer: { kind: "auto" },
        vae: { kind: "cpu" },
        clip_l: null,
        clip_g: null,
        t5: null,
        qwen: null,
      },
    });
    w.unmount();
  });

  it("maps live-style CLIP component names to distinct placement fields", async () => {
    const w = mountPanel(makeForm(), [baseModel]);
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    await vi.dynamicImportSettled();

    const rows = w.findAll("[data-test='component-row']");
    const clipGRow = rows.find((row) => row.text().includes("clip-g encoder"));
    expect(clipGRow).toBeTruthy();

    await clipGRow!
      .get("[data-test='component-placement-select']")
      .setValue("gpu:0");

    const events = w.emitted("update:modelValue");
    const last = events![events!.length - 1][0] as GenerateFormState;
    expect(last.placement?.advanced?.clip_g).toEqual({
      kind: "gpu",
      ordinal: 0,
    });
    expect(last.placement?.advanced?.clip_l).toBeNull();
    w.unmount();
  });
});
