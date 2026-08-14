import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AdvancedDrawer from "./AdvancedDrawer.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { GenerateFormState, ModelInfoExtended } from "../../types";
import { createPinia, setActivePinia } from "pinia";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";

// The upscale section reads the host's model list. Only upscalers matter here.
const UPSCALERS: ModelInfoExtended[] = [
  {
    name: "real-esrgan-x4plus:fp16",
    family: "upscaler",
    size_gb: 0.03,
    is_loaded: false,
    last_used: null,
    hf_repo: "",
    downloaded: true,
    default_steps: 0,
    default_guidance: 0,
    default_width: 0,
    default_height: 0,
    description: "Real-ESRGAN x4+ FP16",
  } as ModelInfoExtended,
];
vi.mock("../../api", () => ({
  fetchModels: vi.fn(async () => UPSCALERS),
  fetchCatalogInstalled: vi.fn(async () => ({
    entries: [],
    page: 1,
    page_size: 0,
    total: 0,
  })),
}));

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, ...overrides };
}

function factory(
  family: string,
  overrides: Partial<GenerateFormState> = {},
  extra: Record<string, unknown> = {},
) {
  const pinia = createPinia();
  setActivePinia(pinia);
  if (extra.output === "sequence") useSequenceDraftStore().ensureClips(97);
  return mount(AdvancedDrawer, {
    props: { open: true, modelValue: baseForm(overrides), family, ...extra },
    global: {
      plugins: [pinia],
      stubs: {
        LoraPicker: { template: "<div data-test='lora-picker-stub' />" },
        RouterLink: { template: "<a><slot /></a>" },
      },
    },
  });
}

describe("AdvancedDrawer sequence contract", () => {
  it("delegates the H3 opening frame to the page's existing upload/gallery picker", async () => {
    const model = {
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
      source_image: "required",
      downloaded: true,
    } as ModelInfoExtended;
    const wrapper = factory(
      model.family,
      { model: model.name, modelFamily: model.family },
      { models: [model] },
    );

    expect(wrapper.find("[data-test='h3-first-well']").exists()).toBe(true);
    await wrapper.get("[data-test='h3-first-gallery']").trigger("click");
    expect(wrapper.emitted("open-h3-first-frame-picker")).toHaveLength(1);
  });

  it("delegates the optional H3 closing frame to the page's picker as well", async () => {
    const model = {
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
      downloaded: true,
    } as ModelInfoExtended;
    const wrapper = factory(
      model.family,
      { model: model.name, modelFamily: model.family },
      { models: [model] },
    );

    await wrapper.get("[data-test='h3-last-gallery']").trigger("click");
    expect(wrapper.emitted("open-h3-last-frame-picker")).toHaveLength(1);
  });

  it("preserves but disables clip negatives for a distilled recipe", () => {
    const wrapper = factory(
      "ltx2",
      { model: "ltx-2.3-22b-distilled:fp8" },
      { output: "sequence" },
    );
    useSequenceDraftStore().clips[0]!.negativePrompt = "flicker";
    const input = wrapper.get("[data-test='sequence-negative-input']");
    expect(input.attributes("disabled")).toBeDefined();
    expect(useSequenceDraftStore().clips[0]!.negativePrompt).toBe("flicker");
    expect(
      wrapper.get("[data-test='sequence-negative-unavailable-hint']").text(),
    ).toContain("does not use negative-prompt guidance");
  });

  it("shows only sequence-owned controls", () => {
    const wrapper = factory(
      "ltx2",
      {},
      { output: "sequence", sequenceSupportsAudio: true },
    );
    expect(
      wrapper.find("[data-test='sequence-section-opening-image']").exists(),
    ).toBe(true);
    expect(
      wrapper.find("[data-test='sequence-section-negative']").exists(),
    ).toBe(true);
    expect(wrapper.find("[data-test='sequence-section-audio']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='section-source']").exists()).toBe(false);
    expect(wrapper.find("[data-test='section-lora']").exists()).toBe(false);
    expect(wrapper.find("[data-test='section-output']").exists()).toBe(false);
    expect(wrapper.find("[data-test='section-video']").exists()).toBe(false);
  });

  it("exposes strength and maskless fit controls for an opening image", async () => {
    const wrapper = factory(
      "ltx2",
      { strength: 0.8, sourceFitPolicy: { mode: "crop-fill" } },
      { output: "sequence" },
    );
    useSequenceDraftStore().openingImage = {
      filename: "opening.png",
      base64: "QUJD",
    };
    await wrapper.vm.$nextTick();

    expect(
      wrapper.get("[data-test='sequence-source-strength']").text(),
    ).toContain("0.80");
    const fit = wrapper.get("[data-test='sequence-source-fit']");
    expect((fit.element as HTMLSelectElement).value).toBe("crop-fill");
    expect(fit.findAll("option").map((option) => option.text())).toEqual([
      "Crop fill",
      "Scale to fit",
      "Resize to fill",
      "Upscale + crop",
    ]);
    await fit.setValue("lanczos-resize");
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      sourceFitPolicy: { mode: "lanczos-resize" },
    });
  });

  it("edits and resets camera motion on the active clip", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => [
        {
          id: "dolly-in",
          label: "Dolly in",
          size_bytes: 327_309_208,
          installed: false,
          download_model: "ltx2-camera-control-dolly-in-19b",
          download_repo: "Lightricks/camera",
          download_filename: "dolly-in.safetensors",
          download_sha256: "a".repeat(64),
        },
      ],
    } as Response);
    const wrapper = factory(
      "ltx2",
      { model: "ltx-2-19b-distilled:fp8" },
      { output: "sequence" },
    );
    const draft = useSequenceDraftStore();
    draft.clips[1]!.cameraControl = "jib-up";
    await vi.waitFor(() =>
      expect(
        wrapper.get("[data-test='sequence-camera-motion']").findAll("option"),
      ).toHaveLength(3),
    );
    expect(draft.clips[1]?.cameraControl).toBeNull();
    await wrapper
      .get("[data-test='sequence-camera-motion']")
      .setValue("dolly-in");
    expect(draft.clips[0]?.cameraControl).toBe("dolly-in");
    expect(
      wrapper.get("[data-test='sequence-camera-motion']").text(),
    ).toContain("downloads on first use");
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    expect(draft.clips[0]?.cameraControl).toBeNull();
  });
});

describe("AdvancedDrawer video duration", () => {
  it("keeps the seconds slider beside exact frames and FPS", async () => {
    const model = {
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      default_frames: 97,
      default_fps: 24,
      max_runtime_seconds: 20,
      max_frames_absolute: 604,
      frame_step: 8,
    } as ModelInfoExtended;
    const wrapper = factory(
      "ltx2",
      { model: model.name, frames: 97, fps: 24 },
      { models: [model] },
    );
    expect(wrapper.get("[data-test='video-duration']").text()).toContain(
      "4.0s",
    );
    await wrapper
      .get("[data-test='video-duration'] input[type='range']")
      .setValue("241");
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      frames: 241,
    });
    expect(wrapper.find("[data-test='video-frames']").exists()).toBe(true);
    expect(wrapper.find("[data-test='video-fps']").exists()).toBe(true);
  });
});

function sections(family: string, extra: Record<string, unknown> = {}) {
  const wrapper = factory(family, {}, extra);
  const has = (key: string) =>
    wrapper.find(`[data-test='section-${key}']`).exists();
  return {
    scheduler: has("scheduler"),
    negative: has("negative"),
    source: has("source"),
    lora: has("lora"),
    upscale: has("upscale"),
    output: has("output"),
    video: has("video"),
    placement: has("placement"),
  };
}

// The canonical Advanced section order shared with the desktop inspector
// (which has no placement section — desktop owns GPU placement in Settings).
const SECTION_ORDER = [
  "scheduler",
  "negative",
  "source",
  "lora",
  "upscale",
  "output",
  "video",
  "placement",
] as const;

function sectionIds(wrapper: ReturnType<typeof factory>): string[] {
  return wrapper
    .findAll("[data-test^='section-']")
    .map((node) => node.attributes("data-test")!.replace("section-", ""));
}

describe("AdvancedDrawer section ordering contract", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  const gpus = { placementGpus: [{ ordinal: 0, name: "GPU 0" }] };

  it("renders still-image sections in the canonical order", () => {
    expect(sectionIds(factory("sdxl", {}, gpus))).toEqual([
      "scheduler",
      "negative",
      "source",
      "lora",
      "upscale",
      "output",
      "placement",
    ]);
  });

  it("renders video sections in the canonical order", () => {
    expect(sectionIds(factory("ltx2", {}, gpus))).toEqual([
      "negative",
      "source",
      "lora",
      "output",
      "video",
      "placement",
    ]);
  });

  it("keeps every family's rendered sections a subsequence of the canon", () => {
    const families = [
      "sdxl",
      "sd15",
      "sd3.5",
      "flux",
      "qwen-image-edit",
      "ltx-video",
      "ltx2",
    ];
    for (const family of families) {
      const rendered = sectionIds(factory(family, {}, gpus));
      expect(rendered.length).toBeGreaterThan(0);
      expect(rendered).toEqual(
        SECTION_ORDER.filter((id) => rendered.includes(id)),
      );
    }
  });
});

describe("AdvancedDrawer capability matrix", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  it("preserves but disables a saved negative prompt for distilled LTX", () => {
    const wrapper = factory("ltx2", {
      model: "ltx-2.3-22b-distilled:fp8",
      negativePrompt: "watermark",
    });
    const input = wrapper.get("[data-test='negative-input']");
    expect(input.attributes("disabled")).toBeDefined();
    expect((input.element as HTMLTextAreaElement).value).toBe("watermark");
    expect(
      wrapper.get("[data-test='negative-unavailable-hint']").text(),
    ).toContain("Saved for reuse");
  });

  it("matches desktop with always-open icon sections and no nested disclosure", () => {
    const wrapper = factory("sdxl");
    const sections = wrapper.findAll(".adv__sections > .ms-acc");

    expect(sections.length).toBeGreaterThan(0);
    expect(
      wrapper.findAll(".adv__sections > .ms-acc .ms-acc__plate svg"),
    ).toHaveLength(sections.length);
    expect(wrapper.find(".adv__sections .ms-acc__chevron").exists()).toBe(
      false,
    );
    expect(wrapper.find(".adv__sections button.ms-acc__head").exists()).toBe(
      false,
    );
    expect(wrapper.find("[data-test='negative-input']").exists()).toBe(true);
    expect(wrapper.find("[data-test='exact-width']").exists()).toBe(true);
  });

  it("sdxl exposes scheduler, negative, source and output", () => {
    const s = sections("sdxl");
    expect(s.scheduler).toBe(true);
    expect(s.negative).toBe(true);
    expect(s.source).toBe(true);
    expect(s.output).toBe(true);
    expect(s.video).toBe(false);
  });

  it("flux hides scheduler and negative but keeps source, lora and output", () => {
    const s = sections("flux");
    expect(s.scheduler).toBe(false);
    expect(s.negative).toBe(false);
    expect(s.source).toBe(true);
    expect(s.lora).toBe(true);
    expect(s.output).toBe(true);
  });

  it("qwen-image-edit exposes its required edit-image attachment section", async () => {
    const wrapper = factory("qwen-image-edit");
    expect(sections("qwen-image-edit").source).toBe(true);
    expect(wrapper.find("[data-test='source-attach']").exists()).toBe(true);
    expect(wrapper.text()).toContain("Edit images");
  });

  it("ltx2 exposes the video section", () => {
    expect(sections("ltx2").video).toBe(true);
  });

  it("offers post-generate upscale for stills but not for video", () => {
    // The server skips the post-upscaler whenever the response is a video
    // (queue.rs), so the section must not pretend otherwise.
    expect(sections("flux").upscale).toBe(true);
    expect(sections("ltx2").upscale).toBe(false);
  });

  it("shows GPU placement only when GPUs are reported", () => {
    expect(sections("flux").placement).toBe(false);
    expect(
      sections("flux", { placementGpus: [{ ordinal: 0, name: "GPU 0" }] })
        .placement,
    ).toBe(true);
  });

  it("renders the real placement child with the selected placement", async () => {
    const placement = { text_encoders: { kind: "cpu" as const } };
    const wrapper = factory(
      "flux",
      { placement },
      { placementGpus: [{ ordinal: 0, name: "GPU 0" }] },
    );
    await wrapper
      .get("[data-test='section-placement'] .ms-acc__head")
      .trigger("click");
    expect(
      wrapper.getComponent({ name: "PlacementPanel" }).props("modelValue"),
    ).toEqual(placement);
  });

  it("offers all five source-fit policies", async () => {
    const wrapper = factory("sdxl", {
      imageAttachments: [
        { kind: "upload", filename: "source.png", base64: "AA" },
      ],
    });
    await wrapper
      .get("[data-test='section-source'] .ms-acc__head")
      .trigger("click");
    const labels = wrapper
      .findAll("[aria-label='Fit to canvas'] button")
      .map((button) => button.text());
    expect(labels).toEqual([
      "Contain",
      "Cover",
      "Pad + repaint",
      "Lanczos",
      "Upscale + fit",
    ]);
  });
});

describe("AdvancedDrawer always-open sections", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  it("shows every available control without aria disclosure state", () => {
    const w = factory("flux");
    expect(w.find("[data-test='source-attach']").exists()).toBe(true);
    expect(w.find("[data-test='lora-picker-stub']").exists()).toBe(true);
    expect(w.find("[data-test='exact-width']").exists()).toBe(true);
    expect(w.find("[aria-expanded]").exists()).toBe(false);
  });

  it("still hides sections unavailable to the selected family", () => {
    const w = factory("sd3.5");
    expect(w.find("[data-test='section-lora']").exists()).toBe(false);
    expect(w.find("[data-test='cfg-plus']").exists()).toBe(true);
  });
});

describe("AdvancedDrawer ControlNet block", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  async function openSource(family: string, overrides = {}) {
    const wrapper = factory(family, overrides);
    const head = wrapper.find("[data-test='section-source'] .ms-acc__head");
    if (head.exists()) await head.trigger("click");
    return wrapper;
  }

  it("shows the ControlNet block for a controlnet family (sd15)", async () => {
    const wrapper = await openSource("sd15");
    expect(wrapper.find("[data-test='controlnet-block']").exists()).toBe(true);
  });

  it("hides the ControlNet block for flux", async () => {
    const wrapper = await openSource("flux");
    expect(wrapper.find("[data-test='controlnet-block']").exists()).toBe(false);
  });

  it("only surfaces control model + scale once an image is attached", async () => {
    const bare = await openSource("sd15");
    expect(bare.find("[data-test='control-attach']").exists()).toBe(true);
    expect(bare.find("[data-test='control-model']").exists()).toBe(false);

    const withImage = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    expect(withImage.find("[data-test='control-model']").exists()).toBe(true);
  });

  it("round-trips the control model text field", async () => {
    const wrapper = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    const input = wrapper.get("[data-test='control-model']");
    (input.element as HTMLInputElement).value = "control_v11p_sd15_canny";
    await input.trigger("input");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.controlModel).toBe("control_v11p_sd15_canny");
  });

  it("suggests installed ControlNet models while retaining custom input", async () => {
    const wrapper = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    await wrapper.setProps({
      models: [
        {
          ...UPSCALERS[0],
          name: "controlnet-canny-sd15:fp16",
          family: "controlnet",
        },
      ],
    });
    expect(wrapper.get("[data-test='control-model']").attributes("list")).toBe(
      "installed-controlnet-models",
    );
    expect(
      wrapper.get("#installed-controlnet-models option").attributes("value"),
    ).toBe("controlnet-canny-sd15:fp16");
  });

  it("round-trips the control scale slider", async () => {
    const wrapper = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    const range = wrapper.get("[data-test='control-scale'] input");
    (range.element as HTMLInputElement).value = "1.5";
    await range.trigger("input");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.controlScale).toBe(1.5);
  });

  it("removes the control image", async () => {
    const wrapper = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
      controlModel: "control_v11p_sd15_canny",
    });
    await wrapper.get("[data-test='control-remove']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.controlImage).toBe(null);
  });
});

describe("AdvancedDrawer video suite", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  async function openVideo(family: string, overrides = {}) {
    const wrapper = factory(family, overrides);
    await wrapper
      .get("[data-test='section-video'] .ms-acc__head")
      .trigger("click");
    return wrapper;
  }

  it("shows the LTX-2 suite for ltx2 but not for ltx-video", async () => {
    const ltx2 = await openVideo("ltx2");
    expect(ltx2.find("[data-test='ltx2-suite']").exists()).toBe(true);
    const ltxVideo = await openVideo("ltx-video");
    expect(ltxVideo.find("[data-test='ltx2-suite']").exists()).toBe(false);
    // The GIF preview toggle is shared by every video family.
    expect(ltxVideo.find("[data-test='video-gif-preview']").exists()).toBe(
      true,
    );
  });

  it("hides the video suite entirely for flux", () => {
    expect(sections("flux").video).toBe(false);
  });

  it("toggles the GIF preview", async () => {
    const wrapper = await openVideo("ltx2");
    await wrapper.get("[data-test='video-gif-preview']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.gifPreview).toBe(true);
  });

  it("round-trips the LTX-2 pipeline select", async () => {
    const wrapper = await openVideo("ltx2");
    const select = wrapper.get("[data-test='ltx2-pipeline']");
    (select.element as HTMLSelectElement).value = "two-stage-hq";
    await select.trigger("change");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.pipeline).toBe("two-stage-hq");
  });

  it("toggles LTX-2 audio decode off (null default reads as on)", async () => {
    const wrapper = await openVideo("ltx2");
    await wrapper.get("[data-test='ltx2-enable-audio']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.enableAudio).toBe(false);
  });

  it("edits an existing keyframe's frame", async () => {
    const wrapper = await openVideo("ltx2", {
      keyframes: [
        {
          frame: 0,
          image: { kind: "upload", filename: "k.png", base64: "AA" },
        },
      ],
    });
    const input = wrapper.get("[data-test='ltx2-keyframe-frame']");
    (input.element as HTMLInputElement).value = "48";
    await input.trigger("input");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.keyframes[0]?.frame).toBe(48);
  });

  it("Reset restores the advertised default negative, not the opt-out", async () => {
    // Shipping "" from Reset would silently disable wan's tuned default —
    // the explicit opt-out belongs to the user, not to a reset action.
    const wrapper = factory("wan", {
      negativePrompt: "hand-typed",
      negativePromptDefault: "TUNED_DEFAULT",
    });
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.negativePrompt).toBe("TUNED_DEFAULT");
  });

  it("Reset also clears the video suite", async () => {
    const wrapper = factory("ltx2", {
      pipeline: "two-stage",
      gifPreview: true,
      spatialUpscale: "x2",
    });
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.pipeline).toBe(null);
    expect(next.gifPreview).toBe(false);
    expect(next.spatialUpscale).toBe(null);
  });
});

describe("AdvancedDrawer interactions", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  it("keeps scheduler and negative controls visible together", () => {
    const wrapper = factory("sdxl");
    expect(wrapper.find("[data-test='negative-input']").exists()).toBe(true);
    expect(wrapper.find("[data-test='scheduler-select']").exists()).toBe(true);
  });

  it("adds negative-prompt quick chips", async () => {
    const wrapper = factory("sdxl", { negativePrompt: "" });
    await wrapper
      .get("[data-test='section-negative'] .ms-acc__head")
      .trigger("click");
    await wrapper.get("[data-test='neg-chip-blurry']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.negativePrompt).toBe("blurry");
  });

  it("snaps video frames to 8n+1", async () => {
    const wrapper = factory("ltx2");
    await wrapper
      .get("[data-test='section-video'] .ms-acc__head")
      .trigger("click");
    const input = wrapper.get("[data-test='video-frames']");
    (input.element as HTMLInputElement).value = "100";
    await input.trigger("change");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.frames).toBe(97);
  });

  it("snaps wan video frames to the advertised 4n+1 grid", async () => {
    const model = {
      name: "wan22-i2v-a14b:q5",
      family: "wan",
      frame_step: 4,
    };
    const wrapper = factory(
      "wan",
      { model: model.name, modelFamily: "wan", frames: 45 },
      { models: [model] },
    );
    expect(wrapper.get("[data-test='section-video']").text()).toContain(
      "Frames (4n+1)",
    );
    const input = wrapper.get("[data-test='video-frames']");
    (input.element as HTMLInputElement).value = "46";
    await input.trigger("change");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.frames).toBe(45);
  });

  it("Reset clears advanced fields but preserves the prompt", async () => {
    const wrapper = factory("sdxl", {
      prompt: "a lighthouse in a storm",
      negativePrompt: "blurry",
      loras: [{ path: "a", scale: 1 }],
      upscaleModel: "real-esrgan-x4plus",
    });
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.prompt).toBe("a lighthouse in a storm");
    expect(next.negativePrompt).toBe("");
    expect(next.loras).toEqual([]);
    expect(next.upscaleModel).toBe("");
  });

  it("exact-size inputs snap to 16px and swap orientation", async () => {
    const wrapper = factory("flux");
    await wrapper
      .get("[data-test='section-output'] .ms-acc__head")
      .trigger("click");
    const width = wrapper.get("[data-test='exact-width']");
    await width.setValue("1000");
    await width.trigger("change");
    let next = wrapper.emitted("update:modelValue")?.at(-1)?.[0] as {
      width: number;
    };
    expect(next.width).toBe(1008); // 1000 → nearest multiple of 16

    await wrapper.setProps({
      modelValue: { ...baseForm({ width: 1216, height: 704 }) },
    });
    await wrapper.get("[data-test='exact-swap']").trigger("click");
    const swapped = wrapper.emitted("update:modelValue")?.at(-1)?.[0] as {
      width: number;
      height: number;
    };
    expect(swapped.width).toBe(704);
    expect(swapped.height).toBe(1216);
  });

  it("snaps exact dimensions to the selected model's advertised grid", async () => {
    const wrapper = factory(
      "ltx2",
      { model: "ltx-2-19b-distilled:fp8", width: 1000 },
      {
        models: [
          {
            name: "ltx-2-19b-distilled:fp8",
            family: "ltx2",
            dimension_alignment: 32,
          },
        ],
      },
    );
    const width = wrapper.get("[data-test='exact-width']");
    expect(width.attributes("step")).toBe("32");
    await width.trigger("change");
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      width: 992,
    });
  });

  it("round-trips 'Upscale after generate' into the generate request", async () => {
    // The always-visible picker must survive into the wire request and the
    // explicit Off option must remove it again.
    const form = useGenerateForm();
    const wrapper = mount(AdvancedDrawer, {
      props: { open: true, modelValue: form.state.value, family: "flux" },
      global: {
        stubs: {
          LoraPicker: { template: "<div />" },
          PlacementPanel: { template: "<div />" },
          RouterLink: { template: "<a><slot /></a>" },
        },
      },
    });
    await flushPromises();

    await wrapper
      .get("[data-test='upscale-model']")
      .setValue("real-esrgan-x4plus:fp16");
    let next = wrapper.emitted("update:modelValue")!.at(-1)![0] as
      GenerateFormState | undefined;
    Object.assign(form.state.value, next);
    expect(form.state.value.upscaleModel).toBe("real-esrgan-x4plus:fp16");
    expect(form.toRequest().upscale_model).toBe("real-esrgan-x4plus:fp16");

    await wrapper.setProps({ modelValue: { ...form.state.value } });
    await wrapper.get("[data-test='upscale-model']").setValue("");
    next = wrapper.emitted("update:modelValue")!.at(-1)![0] as
      GenerateFormState | undefined;
    Object.assign(form.state.value, next);
    expect(form.state.value.upscaleModel).toBe("");
    expect(form.toRequest().upscale_model).toBeUndefined();
  });

  it("puts Reset at the top of the phone sheet, above the sections", async () => {
    // SheetPanel's full variant has no header slot, so the reset has to lead
    // the sheet body — buried under every advanced section it is unreachable
    // without scrolling the whole sheet.
    const wrapper = factory(
      "sdxl",
      { prompt: "a lighthouse in a storm", negativePrompt: "blurry" },
      { mobile: true },
    );
    const reset = wrapper.get("[data-test='advanced-reset']");
    const sections = wrapper.get("[data-test='section-negative']");
    expect(
      reset.element.compareDocumentPosition(sections.element) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    await reset.trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.prompt).toBe("a lighthouse in a storm");
    expect(next.negativePrompt).toBe("");
  });

  it("gives wan its sampler recipe and no second scheduler picker", async () => {
    const wrapper = factory("wan", { model: "wan22-t2v-a14b:q5" });
    // The solver writes the same `scheduler` field the generic section owns,
    // so exactly one picker for it may ever render.
    expect(wrapper.find("[data-test='scheduler-select']").exists()).toBe(false);
    const solver = wrapper.get("[data-test='wan-solver-select']");
    expect(solver.findAll("option").map((o) => o.text())).toEqual([
      "Default",
      "UniPC",
      "Euler",
      "DPM++",
    ]);
    expect(wrapper.find("[data-test='wan-sample-shift']").exists()).toBe(true);
    expect(wrapper.find("[data-test='wan-distill-high']").exists()).toBe(true);

    await solver.setValue("dpm-pp");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.scheduler).toBe("dpm-pp");
  });

  it("hides the distill strengths on a wan tier that ships no distill", () => {
    const wrapper = factory("wan", { model: "wan22-ti2v-5b:fp16" });
    expect(wrapper.find("[data-test='wan-sample-shift']").exists()).toBe(true);
    expect(wrapper.find("[data-test='wan-distill-high']").exists()).toBe(false);
    expect(wrapper.find("[data-test='wan-distill-low']").exists()).toBe(false);
  });

  it("hides the sampler recipe entirely off-family", () => {
    for (const family of ["sdxl", "ltx2", "flux"]) {
      const wrapper = factory(family);
      expect(wrapper.find("[data-test='section-wan-recipe']").exists()).toBe(
        false,
      );
    }
  });

  it("leaves the flow shift absent until it is typed into", async () => {
    const wrapper = factory("wan", { model: "wan22-t2v-a14b:q5" });
    const shift = wrapper.get("[data-test='wan-sample-shift']");
    expect((shift.element as HTMLInputElement).value).toBe("");

    await shift.setValue("12");
    let [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.wanRecipe?.sampleShift).toBe(12);

    // Clearing it must return to absent, not zero — zero is a value the engine
    // would apply.
    await wrapper.setProps({ modelValue: next });
    await wrapper.get("[data-test='wan-sample-shift']").setValue("");
    [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.wanRecipe?.sampleShift).toBeNull();
  });

  it("names the out-of-band distill strength inline", async () => {
    const wrapper = factory("wan", { model: "wan22-t2v-a14b:q5" });
    await wrapper.get("[data-test='wan-distill-high']").setValue("9");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    await wrapper.setProps({ modelValue: next });
    expect(wrapper.get("[data-test='wan-recipe-error']").text()).toBe(
      "High-noise distill strength must be at most 4.",
    );
  });

  it("emits close from Done in the phone sheet", async () => {
    // Inline (desktop) has no Done — it's always visible. The Done button
    // lives only in the phone Advanced sheet (mobile: true).
    const wrapper = factory("sdxl", {}, { mobile: true });
    await wrapper.get("[data-test='advanced-done']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });
});

/**
 * Per-model source-image conditioning (#772) and the wan End frame well
 * (#779). Every gate here comes from the shared capability kit — the drawer
 * never reads `source_image` or a family set of its own.
 */
describe("AdvancedDrawer source-image contract", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  function wanModel(source_image?: string | null): ModelInfoExtended {
    return {
      name: "wan22-ti2v-5b:fp16",
      family: "wan",
      size_gb: 10,
      is_loaded: false,
      last_used: null,
      hf_repo: "",
      downloaded: true,
      default_steps: 20,
      default_guidance: 3.5,
      default_width: 1280,
      default_height: 704,
      default_frames: 81,
      default_fps: 24,
      description: "Wan 2.2 TI2V 5B",
      ...(source_image === undefined ? {} : { source_image }),
    } as ModelInfoExtended;
  }

  function wan(
    source_image?: string | null,
    overrides: Partial<GenerateFormState> = {},
  ) {
    return factory(
      "wan",
      {
        model: "wan22-ti2v-5b:fp16",
        modelFamily: "wan",
        frames: 81,
        fps: 24,
        sourceImageCapability: source_image ?? null,
        ...overrides,
      },
      { models: [wanModel(source_image)] },
    );
  }

  it("keeps today's source well when the server advertises nothing", () => {
    const wrapper = wan(undefined);
    expect(wrapper.find("[data-test='section-source']").exists()).toBe(true);
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      false,
    );
    // An older server rejects wan keyframes outright, so absence hides the well.
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);
  });

  it("hides the whole source section for a text-to-video checkpoint", () => {
    const wrapper = wan("unsupported");
    expect(wrapper.find("[data-test='section-source']").exists()).toBe(false);
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);
  });

  it("marks the source required and names why the request would be refused", () => {
    const wrapper = wan("required");
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      true,
    );
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toBe(
      "This checkpoint is image-to-video only. Attach a source image to use as the first frame.",
    );
  });

  // #783: the inline notice has to read a continuation the way submit and
  // admission do (`request_carries_source_frames`), or the well contradicts a
  // Generate button that accepts the draft.
  it("clears the inline message for a continuation with no attached image", () => {
    const wrapper = wan("required", {
      extendVideo: { kind: "upload", filename: "clip.mp4", base64: "Q0xJUA==" },
    });
    expect(
      wrapper.find("[data-test='source-conditioning-error']").exists(),
    ).toBe(false);
  });

  it("clears the inline message once the opening frame is attached", () => {
    const wrapper = wan("required", {
      imageAttachments: [
        { kind: "upload", filename: "open.png", base64: "FIRST" },
      ],
    });
    expect(
      wrapper.find("[data-test='source-conditioning-error']").exists(),
    ).toBe(false);
  });

  it("offers the optional End frame well and its attach/remove affordances", async () => {
    const wrapper = wan("optional");
    expect(wrapper.get("[data-test='end-frame-well']").text()).toContain(
      "Optional",
    );
    await wrapper.get("[data-test='end-frame-attach']").trigger("click");
    expect(wrapper.emitted("open-end-frame-picker")).toHaveLength(1);

    const attached = wan("optional", {
      imageAttachments: [
        { kind: "upload", filename: "open.png", base64: "FIRST" },
      ],
      endFrame: { kind: "upload", filename: "close.png", base64: "LAST" },
    });
    expect(attached.get("[data-test='end-frame-well']").text()).toContain(
      "close.png",
    );
    await attached.get("[data-test='end-frame-remove']").trigger("click");
    expect(attached.emitted("clear-end-frame")).toHaveLength(1);
  });

  it("explains an end frame with no first frame", () => {
    const wrapper = wan("optional", {
      endFrame: { kind: "upload", filename: "close.png", base64: "LAST" },
    });
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toBe(
      "An end frame needs a first frame. Attach a source image, or remove the end frame.",
    );
  });

  it("never offers an End frame well outside wan", () => {
    const wrapper = factory("ltx2", {
      model: "ltx-2-19b:fp8",
      modelFamily: "ltx2",
      sourceImageCapability: "optional",
    });
    expect(wrapper.find("[data-test='section-source']").exists()).toBe(true);
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);
  });
});

/**
 * Continuation is not an LTX-2 control (#783). Wan continues by seeding the
 * render with the source clip's final frame, which its image-conditioned
 * checkpoints accept — but the extend field lived inside the LTX-2 advanced
 * suite, whose gate is the audio families, so no wan checkpoint could reach
 * it however the host advertised `supports_extend`.
 */
describe("AdvancedDrawer — continue a video", () => {
  it("offers continuation for a wan checkpoint the host advertises", () => {
    const wrapper = factory(
      "wan",
      { model: "wan22-i2v-a14b:q5", modelFamily: "wan" },
      { canExtend: true, extendDefaultOverlapFrames: 1 },
    );
    expect(wrapper.find("[data-test='extend-field']").exists()).toBe(true);
    // …without dragging the LTX-2 suite in behind it.
    expect(wrapper.find("[data-test='ltx2-pipeline']").exists()).toBe(false);
  });

  it("keeps the control hidden when the host advertises nothing", () => {
    const wrapper = factory("wan", {
      model: "wan22-t2v-a14b:q5",
      modelFamily: "wan",
    });
    expect(wrapper.find("[data-test='extend-field']").exists()).toBe(false);
  });

  it("still offers continuation on LTX-2", () => {
    const wrapper = factory(
      "ltx2",
      { model: "ltx-2-19b-distilled:fp8", modelFamily: "ltx2" },
      { canExtend: true },
    );
    expect(wrapper.find("[data-test='extend-field']").exists()).toBe(true);
  });

  // Wan's engine refuses any overlap but the one frame it was seeded with,
  // so the 8k+1 ladder offered four choices and three of them failed.
  it("offers wan only the single overlap its engine accepts", () => {
    const wrapper = factory(
      "wan",
      {
        model: "wan22-i2v-a14b:q5",
        modelFamily: "wan",
        frames: 53,
        extendVideoPath: "/srv/mold/clip.mp4",
      },
      { canExtend: true, extendDefaultOverlapFrames: 1 },
    );
    const options = wrapper
      .get("[data-test='extend-overlap']")
      .findAll("option")
      .map((option) => option.attributes("value"));
    expect(options).toEqual(["1"]);
  });
});
