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
  it("keeps only the Ref2VA ordered-reference panel — boundaries live in the primary form", () => {
    const fl2va = {
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
      source_image: "required",
      downloaded: true,
    } as ModelInfoExtended;
    const boundaries = factory(
      fl2va.family,
      { model: fl2va.name, modelFamily: fl2va.family },
      { models: [fl2va] },
    );
    expect(boundaries.find("[data-test='section-h3-authoring']").exists()).toBe(
      false,
    );

    const ref2va = {
      name: "minimax-h3-ref2va:comfy-pruned-int8",
      family: "minimax-h3",
      downloaded: true,
    } as ModelInfoExtended;
    const references = factory(
      ref2va.family,
      { model: ref2va.name, modelFamily: ref2va.family },
      { models: [ref2va] },
    );
    expect(references.find("[data-test='section-h3-authoring']").exists()).toBe(
      true,
    );
    expect(references.text()).toContain("Ordered references");
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

  it("sequence Reset preserves the opening image and source conditioning", async () => {
    const wrapper = factory(
      "ltx2",
      { model: "ltx-2-19b-distilled:fp8", strength: 0.6 },
      { output: "sequence" },
    );
    const draft = useSequenceDraftStore();
    draft.openingImage = {
      kind: "upload",
      filename: "open.png",
      base64: "CCCC",
    } as never;
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    expect(draft.openingImage).not.toBeNull();
    // Strength/fit render beside the opening-image well, not in Advanced.
    const patched = wrapper.emitted("update:modelValue")?.at(-1)?.[0] as
      | GenerateFormState
      | undefined;
    expect(patched?.strength ?? 0.6).toBe(0.6);
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
      "lora",
      "upscale",
      "output",
      "placement",
    ]);
  });

  it("renders video sections in the canonical order", () => {
    expect(sectionIds(factory("ltx2", {}, gpus))).toEqual([
      "negative",
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

  it("sdxl exposes scheduler, negative and output — never a source section", () => {
    const s = sections("sdxl");
    expect(s.scheduler).toBe(true);
    expect(s.negative).toBe(true);
    expect(s.source).toBe(false);
    expect(s.output).toBe(true);
    expect(s.video).toBe(false);
  });

  it("flux hides scheduler and negative but keeps lora and output", () => {
    const s = sections("flux");
    expect(s.scheduler).toBe(false);
    expect(s.negative).toBe(false);
    expect(s.source).toBe(false);
    expect(s.lora).toBe(true);
    expect(s.output).toBe(true);
  });

  it("qwen-image-edit keeps its edit images in the primary form, not Advanced", () => {
    const wrapper = factory("qwen-image-edit");
    expect(sections("qwen-image-edit").source).toBe(false);
    expect(wrapper.find("[data-test='source-attach']").exists()).toBe(false);
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
});

describe("AdvancedDrawer always-open sections", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  it("shows every available control without aria disclosure state", () => {
    const w = factory("flux");
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

  it("Reset preserves source media — it lives in the primary form now", async () => {
    const image = {
      kind: "upload",
      filename: "frame.png",
      base64: "AAAA",
      width: 1024,
      height: 576,
    } as const;
    const clip = {
      kind: "upload",
      filename: "clip.mp4",
      base64: "BBBB",
    } as const;
    const wrapper = factory("ltx2", {
      imageAttachments: [image],
      endFrame: { ...image, filename: "end.png" },
      maskImage: { ...image, filename: "mask.png" },
      controlImage: { ...image, filename: "control.png" },
      controlModel: "canny",
      controlScale: 0.8,
      strength: 0.5,
      sourceFitPolicy: { mode: "crop-fill" },
      audioFile: { ...clip, filename: "voice.wav" },
      audioFilePath: "/tmp/voice.wav",
      sourceVideo: clip,
      sourceVideoPath: "/tmp/clip.mp4",
      keyframes: [{ frame: 9, image }],
      negativePrompt: "blurry",
    });
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.imageAttachments).toEqual([image]);
    expect(next.endFrame?.filename).toBe("end.png");
    expect(next.maskImage?.filename).toBe("mask.png");
    expect(next.controlImage?.filename).toBe("control.png");
    expect(next.controlModel).toBe("canny");
    expect(next.controlScale).toBe(0.8);
    expect(next.strength).toBe(0.5);
    expect(next.sourceFitPolicy).toEqual({ mode: "crop-fill" });
    expect(next.audioFile?.filename).toBe("voice.wav");
    expect(next.audioFilePath).toBe("/tmp/voice.wav");
    expect(next.sourceVideo?.filename).toBe("clip.mp4");
    expect(next.sourceVideoPath).toBe("/tmp/clip.mp4");
    expect(next.keyframes).toHaveLength(1);
    // Advanced fields still reset.
    expect(next.negativePrompt).toBe("");
  });

  it("Reset clears guidance overrides and camera motion — they count toward the badge", async () => {
    const wrapper = factory("ltx2", {
      guidanceOverrides: {
        stgScale: 1.5,
        stgBlocks: "3,4",
        rescaleScale: null,
        modalityScale: null,
        skipStep: null,
      },
      cameraControl: "orbit_left",
    });
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.cameraControl).toBe(null);
    expect(next.guidanceOverrides?.stgScale ?? null).toBe(null);
    expect(next.guidanceOverrides?.stgBlocks ?? "").toBe("");
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
