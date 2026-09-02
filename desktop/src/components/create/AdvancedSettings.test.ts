import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import AdvancedSettings from "./AdvancedSettings.vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import ImagePickerModal from "../generate/ImagePickerModal.vue";
import { buildRequest, newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import type {
  Ltx2CameraControlInfo,
  Ltx2ControlAdapterInfo,
  ModelEntry,
} from "../../lib/api/types";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../../lib/ipc", () => ({
  ipc: {
    localGalleryList: vi.fn(() => Promise.resolve({ images: [], target: null })),
    localGalleryDelete: vi.fn(),
  },
  inTauri: () => false,
}));

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

function formFor(family: string): GenerateForm {
  return reactive({ ...newGenerateForm(), family });
}

function accordionTitles(wrapper: VueWrapper): string[] {
  return wrapper.findAllComponents(AccordionSection).map((a) => a.props("title") as string);
}

async function openSection(wrapper: VueWrapper, title: string) {
  const acc = wrapper.findAllComponents(AccordionSection).find((a) => a.props("title") === title);
  if (!acc) throw new Error(`section "${title}" not found`);
  expect(acc.props("open")).toBe(true);
  expect(acc.props("headerInteractive")).toBe(false);
  await flushPromises();
}

function mountSettings(form: GenerateForm, extra: Record<string, unknown> = {}) {
  return mount(AdvancedSettings, {
    props: { form, ...extra },
    attachTo: document.body,
  });
}

const cameraControl = (id = "dolly-in", installed = false): Ltx2CameraControlInfo => ({
  id,
  label: id === "dolly-in" ? "Dolly in" : id,
  size_bytes: 327_309_208,
  installed,
  download_model: `ltx2-camera-control-${id}-19b`,
  download_repo: `Lightricks/${id}`,
  download_filename: `${id}.safetensors`,
  download_sha256: "a".repeat(64),
});

describe("AdvancedSettings — capability matrix", () => {
  it("renders every applicable section open with an icon and no nested disclosure", () => {
    const wrapper = mountSettings(formFor("sdxl"));
    const sections = wrapper.findAllComponents(AccordionSection);

    expect(sections.length).toBeGreaterThan(0);
    for (const section of sections) {
      expect(section.props("open")).toBe(true);
      expect(section.props("headerInteractive")).toBe(false);
      expect(section.props("icon")).toBeTruthy();
    }
    expect(wrapper.findAll(".ms-acc__plate svg")).toHaveLength(sections.length);
    expect(wrapper.find(".ms-acc__chevron").exists()).toBe(false);
    expect(wrapper.find("button.ms-acc__head").exists()).toBe(false);
    expect(wrapper.find("input[aria-label='Width']").exists()).toBe(true);
  });

  it("shows scheduler, negative, and no video for SDXL", () => {
    const titles = accordionTitles(mountSettings(formFor("sdxl")));
    expect(titles).toEqual([
      "Scheduler & sampling",
      "Negative prompt",
      "LoRA stack",
      "Output & seed",
    ]);
  });

  it("hides scheduler and negative for FLUX", () => {
    const titles = accordionTitles(mountSettings(formFor("flux")));
    expect(titles).not.toContain("Scheduler & sampling");
    expect(titles).not.toContain("Negative prompt");
    expect(titles).not.toContain("Source image");
    expect(titles).toContain("Output & seed");
  });

  it("exposes Video (and hides Upscale) for LTX-2", () => {
    const titles = accordionTitles(mountSettings(formFor("ltx2")));
    expect(titles).toContain("Video");
    expect(titles).not.toContain("Upscale after generate");
  });

  it("keeps qwen-edit free of scheduler, negative, and video", () => {
    const titles = accordionTitles(mountSettings(formFor("qwen-image-edit")));
    expect(titles).toEqual(["LoRA stack", "Output & seed"]);
  });

  it("gives wan a sampler recipe instead of a second scheduler picker", async () => {
    const form = reactive({
      ...newGenerateForm(),
      family: "wan",
      model: "wan22-t2v-a14b:q5",
    });
    const wrapper = mountSettings(form);
    const titles = accordionTitles(wrapper);
    expect(titles).toContain("Sampler recipe");
    expect(titles).not.toContain("Scheduler & sampling");

    const solver = wrapper.get("[data-test='wan-solver-select']");
    expect(solver.findAll("option").map((option) => option.text())).toEqual([
      "Default",
      "UniPC",
      "Euler",
      "DPM++",
    ]);
    await solver.setValue("dpm-pp");
    expect(form.scheduler).toBe("dpm-pp");
  });

  it("keeps the wan flow shift absent until it is typed into", async () => {
    const form = reactive({
      ...newGenerateForm(),
      family: "wan",
      model: "wan22-t2v-a14b:q5",
    });
    const wrapper = mountSettings(form);
    const shift = wrapper.get("[data-test='wan-sample-shift']");
    expect((shift.element as HTMLInputElement).value).toBe("");

    await shift.setValue("12");
    expect(form.wanRecipe.sampleShift).toBe(12);

    // Clearing returns to absent, never to zero — zero is a value the engine
    // would apply.
    await shift.setValue("");
    expect(form.wanRecipe.sampleShift).toBeNull();
  });

  it("hides the distill strengths on a wan tier that ships none", () => {
    const wrapper = mountSettings(
      reactive({ ...newGenerateForm(), family: "wan", model: "wan22-ti2v-5b:fp16" }),
    );
    expect(wrapper.find("[data-test='wan-sample-shift']").exists()).toBe(true);
    expect(wrapper.find("[data-test='wan-distill-high']").exists()).toBe(false);
  });

  it("names an out-of-band distill strength inline", async () => {
    const form = reactive({
      ...newGenerateForm(),
      family: "wan",
      model: "wan22-t2v-a14b:q5",
    });
    const wrapper = mountSettings(form);
    await wrapper.get("[data-test='wan-distill-high']").setValue("9");
    await flushPromises();
    expect(wrapper.get("[data-test='wan-recipe-error']").text()).toBe(
      "High-noise distill strength must be at most 4.",
    );
  });

  it("hides the sampler recipe entirely off-family", () => {
    for (const family of ["sdxl", "ltx2", "flux"]) {
      expect(accordionTitles(mountSettings(formFor(family)))).not.toContain("Sampler recipe");
    }
  });
});

describe("AdvancedSettings — section ordering contract", () => {
  // The canonical Advanced section order shared with the web drawer (which
  // additionally renders a trailing placement section — desktop owns GPU
  // placement in Settings instead).
  const SECTION_ORDER = [
    "scheduler",
    "wan-recipe",
    "negative",
    "lora",
    "upscale",
    "output",
    "video",
  ] as const;
  const upscaler = { name: "real-esrgan-x4plus", downloaded: true } as ModelEntry;

  function sectionIds(wrapper: VueWrapper): string[] {
    return wrapper
      .findAll("[data-test^='section-']")
      .map((node) => node.attributes("data-test")!.replace("section-", ""));
  }

  it("renders still-image sections in the canonical order", () => {
    const wrapper = mountSettings(formFor("sdxl"), { upscalers: [upscaler] });
    expect(sectionIds(wrapper)).toEqual(["scheduler", "negative", "lora", "upscale", "output"]);
  });

  it("renders video sections in the canonical order", () => {
    const wrapper = mountSettings(formFor("ltx2"), { upscalers: [upscaler] });
    expect(sectionIds(wrapper)).toEqual(["negative", "lora", "upscale", "output", "video"]);
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
      "wan",
    ];
    for (const family of families) {
      const rendered = sectionIds(mountSettings(formFor(family), { upscalers: [upscaler] }));
      expect(rendered.length).toBeGreaterThan(0);
      expect(rendered).toEqual(SECTION_ORDER.filter((id) => rendered.includes(id)));
    }
  });
});

describe("AdvancedSettings — output & seed", () => {
  it("offers the family output formats and exact size override with a swap", async () => {
    const form = formFor("flux");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Output & seed");
    expect(wrapper.find("input[aria-label='Width']").exists()).toBe(true);
    expect(wrapper.find("input[aria-label='Height']").exists()).toBe(true);
    form.width = 1024;
    form.height = 768;
    await wrapper.get("button[aria-label='Swap width and height']").trigger("click");
    expect(form.width).toBe(768);
    expect(form.height).toBe(1024);
  });

  it("snaps exact dimensions to the selected model's advertised grid", async () => {
    const form = formFor("ltx2");
    form.width = 1000;
    const wrapper = mountSettings(form, {
      selectedModel: {
        name: "ltx-2-19b-distilled:fp8",
        family: "ltx2",
        dimension_alignment: 32,
      } as ModelEntry,
    });
    const width = wrapper.get("input[aria-label='Width']");
    expect(width.attributes("step")).toBe("32");
    await width.trigger("change");
    expect(form.width).toBe(992);
  });

  it("shows the fixed-seed value only when the seed is locked", async () => {
    const form = formFor("flux");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Output & seed");
    expect(wrapper.find("[data-test='advanced-seed-value']").exists()).toBe(false);
    form.seed = "42";
    await flushPromises();
    expect(wrapper.find("[data-test='advanced-seed-value']").exists()).toBe(true);
  });
});

describe("AdvancedSettings — negative prompt", () => {
  it("preserves but disables a saved negative prompt for a distilled recipe", () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    form.negativePrompt = "watermark";
    const wrapper = mountSettings(form);
    const input = wrapper.get("textarea[aria-label='Negative prompt']");
    expect(input.attributes("disabled")).toBeDefined();
    expect(form.negativePrompt).toBe("watermark");
    expect(wrapper.get("[data-test='negative-unavailable-hint']").text()).toContain(
      "Saved for reuse",
    );
  });

  it("appends quick-add words to the negative prompt", async () => {
    const form = formFor("sdxl");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Negative prompt");
    await wrapper.get("[data-test='neg-add-blurry']").trigger("click");
    await wrapper.get("[data-test='neg-add-watermark']").trigger("click");
    expect(form.negativePrompt).toBe("blurry, watermark");
  });
});

describe("AdvancedSettings — upscale", () => {
  const upscaler = { name: "real-esrgan-x4plus", downloaded: false } as ModelEntry;

  it("offers Off plus the known upscalers for image families", async () => {
    const form = formFor("flux");
    const wrapper = mountSettings(form, { upscalers: [upscaler] });
    await openSection(wrapper, "Upscale after generate");
    const select = wrapper.get("[data-test='upscale-select']");
    expect(select.findAll("option").map((o) => o.text())).toEqual([
      "Off",
      "real-esrgan-x4plus (downloads on first use)",
    ]);
    await select.setValue("real-esrgan-x4plus");
    expect(form.upscaleModel).toBe("real-esrgan-x4plus");
  });

  it("offers Framewise upscale for video families", () => {
    expect(accordionTitles(mountSettings(formFor("ltx2"), { upscalers: [upscaler] }))).toContain(
      "Framewise upscale after generate",
    );
  });
});

describe("AdvancedSettings — video (LTX-2)", () => {
  it("shows the LTX-2 pipeline controls without another disclosure", async () => {
    const wrapper = mountSettings(formFor("ltx2"));
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-disclosure']").exists()).toBe(false);
    expect(wrapper.find("[data-test='ltx2-pipeline']").exists()).toBe(true);
    expect(wrapper.find("[data-test='ltx2-spatial']").exists()).toBe(true);
    expect(wrapper.find("[data-test='ltx2-temporal']").exists()).toBe(true);
  });

  it("reveals retake range only when the pipeline is retake", async () => {
    const form = formFor("ltx2");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-retake-start']").exists()).toBe(false);
    await wrapper.get("[data-test='ltx2-pipeline']").setValue("retake");
    expect(form.pipeline).toBe("retake");
    expect(wrapper.find("[data-test='ltx2-retake-start']").exists()).toBe(true);
  });

  it("resets model-owned controls when the recipe changes", async () => {
    const form = formFor("ltx2");
    form.width = 640;
    form.height = 640;
    form.steps = 7;
    form.guidance = 8;
    const selectedModel = {
      name: "ltx2:test",
      family: "ltx2",
      default_width: 1024,
      default_height: 576,
      default_steps: 30,
      default_guidance: 3,
      size_gb: 1,
      is_loaded: false,
      hf_repo: "fixture",
      description: "",
      downloaded: true,
    } as ModelEntry;
    const wrapper = mountSettings(form, { selectedModel });
    await wrapper.get("[data-test='ltx2-pipeline']").setValue("two-stage");
    expect(form).toMatchObject({
      pipeline: "two-stage",
      width: 1024,
      height: 576,
      steps: 30,
      guidance: 3,
    });
  });

  it("edits, validates, counts, and resets guidance overrides", async () => {
    const form = formFor("ltx2");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");

    await wrapper.get("[data-test='ltx2-stg-scale']").setValue("1.5");
    await wrapper.get("[data-test='ltx2-stg-blocks']").setValue("28, nope");
    expect(form.guidanceOverrides.stgScale).toBe(1.5);
    expect(wrapper.get("[data-test='ltx2-stg-blocks-error']").text()).toContain(
      "not a block index",
    );
    expect(wrapper.get("[data-test='ltx2-guidance-count']").text()).toBe("2");
    expect(wrapper.text()).toContain("1 active");

    await wrapper.get("[data-test='ltx2-guidance-reset']").trigger("click");
    expect(form.guidanceOverrides.stgScale).toBeNull();
    expect(form.guidanceOverrides.stgBlocks).toBe("");
    expect(wrapper.find("[data-test='ltx2-stg-blocks-error']").exists()).toBe(false);
  });

  it("renders host-provided reference controls and guide copy", async () => {
    const form = formFor("ltx2");
    const controlAdapters: Ltx2ControlAdapterInfo[] = [
      {
        id: "pose",
        label: "Pose control",
        guide: "A frame-aligned pose guide video.",
        size_bytes: 654_465_256,
        installed: true,
        download_model: "ltx2-control-pose-19b",
        download_repo: "Lightricks/control",
        download_filename: "control.safetensors",
        download_sha256: "a".repeat(64),
      },
    ];
    const wrapper = mountSettings(form, { controlAdapters });
    await openSection(wrapper, "Video");
    await wrapper.get("[data-test='ltx2-reference-control']").setValue("pose");
    expect(form.icLoraControl).toBe("pose");
    expect(form.pipeline).toBe("ic-lora");
    expect(wrapper.get("[data-test='ltx2-reference-guide']").text()).toContain("pose guide video");
  });

  it("selects the lip-dub pipeline for the lip-dub adapter and says where timing comes from", async () => {
    const form = formFor("ltx2");
    const controlAdapters: Ltx2ControlAdapterInfo[] = [
      {
        id: "lipdub",
        label: "Lip dub",
        guide: "A reference video with speech; the mouth is re-timed to new audio.",
        size_bytes: 2_466_665_072,
        installed: true,
        download_model: "ltx2-control-lipdub-23",
        download_repo: "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt",
        download_filename: "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
        download_sha256: "a".repeat(64),
      },
    ];
    const wrapper = mountSettings(form, { controlAdapters });
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-lip-dub-hint']").exists()).toBe(false);

    await wrapper.get("[data-test='ltx2-reference-control']").setValue("lipdub");

    expect(form.icLoraControl).toBe("lipdub");
    expect(form.pipeline).toBe("lip-dub");
    // The frames/fps fields above are not in charge here, so say so.
    expect(wrapper.get("[data-test='ltx2-lip-dub-hint']").text()).toContain("reference video");
  });

  it("keeps the reference control when switching between the two adapter pipelines", async () => {
    const form = formFor("ltx2");
    form.icLoraControl = "lipdub";
    form.pipeline = "lip-dub";
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");

    await wrapper.get("[data-test='ltx2-pipeline']").setValue("ic-lora");
    expect(form.icLoraControl).toBe("lipdub");

    await wrapper.get("[data-test='ltx2-pipeline']").setValue("two-stage");
    expect(form.icLoraControl).toBeNull();
  });

  it("shows the a2vid conditioning-audio input only for that pipeline", async () => {
    const form = formFor("ltx2");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-audio-file']").exists()).toBe(false);
    await wrapper.get("[data-test='ltx2-pipeline']").setValue("a2-vid");
    expect(wrapper.find("[data-test='ltx2-audio-file']").exists()).toBe(true);
  });

  it("adds a keyframe from the picker", async () => {
    const form = formFor("ltx2");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    await wrapper.get("[data-test='ltx2-add-keyframe']").trigger("click");
    const picker = wrapper
      .findAllComponents(ImagePickerModal)
      .find((candidate) => candidate.props("title") === "Keyframe image");
    if (!picker) throw new Error("keyframe picker not found");
    picker.vm.$emit("pick", [{ filename: "k.png", base64: "KB" }]);
    await flushPromises();
    expect(form.keyframes).toHaveLength(1);
    expect(form.keyframes[0]!.frame).toBe(0);
  });

  it("keeps all H3 media controls out of Advanced", () => {
    const fl2va = formFor("minimax-h3");
    fl2va.model = "minimax-h3-fl2va:comfy-pruned-int8";
    const fl2vaWrapper = mountSettings(fl2va, {
      selectedModel: {
        name: fl2va.model,
        family: fl2va.family,
        source_image: "required",
      } as ModelEntry,
    });
    // Boundaries render as the shared source wells in the primary form.
    expect(fl2vaWrapper.find("[data-test='section-h3-authoring']").exists()).toBe(false);

    const ref2va = formFor("minimax-h3");
    ref2va.model = "minimax-h3-ref2va:comfy-pruned-int8";
    const ref2vaWrapper = mountSettings(ref2va, {
      selectedModel: { name: ref2va.model, family: ref2va.family } as ModelEntry,
    });
    expect(ref2vaWrapper.find("[data-test='section-h3-authoring']").exists()).toBe(false);
    expect(accordionTitles(ref2vaWrapper)).not.toContain("Ordered references");
  });

  it("shows the chained-clips cue when frames exceed one clip for a chainable model", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    form.frames = 241;
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    expect(wrapper.get("[data-test='chain-cue']").text()).toContain(
      "chained clips of up to 97 frames",
    );
  });

  it("explains when advanced settings keep a long request single-shot", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-dev:fp8";
    form.frames = 153;
    form.negativePrompt = "flicker";
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");

    expect(wrapper.get("[data-test='single-shot-preservation-cue']").text()).toContain(
      "one 153-frame clip to preserve negative prompt",
    );
    expect(wrapper.find("[data-test='chain-compatibility-error']").exists()).toBe(false);
  });

  it("shows the reject message when a chain would exceed the stage ceiling", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2-19b:fp8";
    // Every ltx2 checkpoint chains now, so the remaining reject case is the
    // server's sixteen-stage ceiling: 1305 frames needs a seventeenth clip.
    form.frames = 1305;
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    expect(wrapper.get("[data-test='chain-reject']").text()).toContain("at most 1297 frames");
  });

  it("renders compatible host camera controls enabled with first-use download copy", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2-19b-distilled:fp8";
    const wrapper = mountSettings(form, {
      cameraControls: [cameraControl("dolly-in", false)],
      cameraControlsLoaded: true,
    });
    await openSection(wrapper, "Video");
    const options = wrapper.get("[data-test='camera-motion']").findAll("option");
    expect(options.map((option) => option.text())).toContain("Dolly in · downloads on first use");
    expect(options[1]?.attributes("disabled")).toBeUndefined();
    await wrapper.get("[data-test='camera-motion']").setValue("dolly-in");
    expect(form.cameraControl).toBe("dolly-in");
    expect(form.loras).toEqual([
      {
        path: "camera-control:dolly-in",
        name: "Dolly in camera control",
        scale: 1,
        trainedWords: [],
      },
    ]);
  });

  it("keeps custom camera motion available when the host reports no built-ins", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    const wrapper = mountSettings(form, {
      cameraControls: [],
      cameraControlsLoaded: true,
    });
    await openSection(wrapper, "Video");
    expect(wrapper.get("[data-test='camera-motion-19b-hint']").text()).toContain("LTX-2 19B");
    expect(
      wrapper
        .get("[data-test='camera-motion']")
        .findAll("option")
        .map((option) => option.text()),
    ).toEqual(["None", "Custom LoRA path…"]);
  });

  it("reveals a custom camera-motion path input", async () => {
    const form = formFor("ltx2");
    form.cameraControl = "dolly-in";
    form.loras = [
      {
        path: "camera-control:dolly-in",
        name: "Dolly in camera control",
        scale: 0.45,
        trainedWords: [],
      },
    ];
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    await wrapper.get("[data-test='camera-motion']").setValue("custom");
    await wrapper.get("[data-test='camera-motion-custom']").setValue("/loras/pan.safetensors");
    expect(form.cameraControl).toBe("/loras/pan.safetensors");
    expect(form.loras).toEqual([
      {
        path: "/loras/pan.safetensors",
        name: "/loras/pan.safetensors",
        scale: 0.45,
        trainedWords: [],
      },
    ]);
  });

  it("disables new camera choices when all four LoRA slots are occupied", async () => {
    const form = formFor("ltx2");
    form.loras = ["one", "two", "three", "four"].map((path) => ({
      path,
      name: path,
      scale: 1,
      trainedWords: [],
    }));
    const wrapper = mountSettings(form, {
      cameraControls: [cameraControl("dolly-in", false)],
      cameraControlsLoaded: true,
    });
    await openSection(wrapper, "Video");
    const options = wrapper.get("[data-test='camera-motion']").findAll("option");
    expect(options[1]?.attributes()).toHaveProperty("disabled");
    expect(options[2]?.attributes()).toHaveProperty("disabled");
  });
});

describe("AdvancedSettings — reset and summary", () => {
  const model: ModelEntry = {
    name: "sdxl:base",
    family: "sdxl",
    downloaded: true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 30,
    default_guidance: 7,
  } as ModelEntry;

  it("restores model defaults but preserves the prompt and prepared batch size", () => {
    const form = formFor("sdxl");
    form.prompt = "a lighthouse at dusk";
    form.negativePrompt = "blurry";
    form.steps = 12;
    form.batchSize = 4;
    const wrapper = mountSettings(form, { selectedModel: model });
    wrapper.get("[data-test='advanced-reset']").trigger("click");
    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.negativePrompt).toBe("");
    expect(form.steps).toBe(30); // model default
    expect(form.batchSize).toBe(4); // prepared state preserved
  });

  it("preserves staged source media across Reset — it lives in the primary form", () => {
    const form = formFor("sdxl");
    form.sourceImage = "SRC";
    form.sourceImageName = "pic.png";
    form.strength = 0.4;
    form.sourceFit = { mode: "crop-fill" };
    form.maskImage = "MASK";
    form.negativePrompt = "blurry";
    form.enableAudio = true;
    const wrapper = mountSettings(form, { selectedModel: model });
    wrapper.get("[data-test='advanced-reset']").trigger("click");
    expect(form.sourceImage).toBe("SRC");
    expect(form.sourceImageName).toBe("pic.png");
    expect(form.strength).toBe(0.4);
    expect(form.sourceFit).toEqual({ mode: "crop-fill" });
    expect(form.maskImage).toBe("MASK");
    expect(form.negativePrompt).toBe("");
    expect(form.enableAudio).toBe(true);
  });

  it("badges the active advanced count in the header", () => {
    const form = formFor("sdxl");
    form.negativePrompt = "blurry";
    form.scheduler = "ddim";
    const wrapper = mountSettings(form);
    expect(wrapper.text()).toContain("2 active");
  });
});

describe("AdvancedSettings — continue a video", () => {
  const extendModel = {
    name: "ltx-2-19b-distilled:fp8",
    family: "ltx2",
    supports_extend: true,
    extend_default_overlap_frames: 17,
  };

  // A host that predates continuation omits `supports_extend`, so rendering
  // the control would only let the user build a request the host rejects.
  it("stays hidden until the selected model advertises continuation", async () => {
    const legacy = mountSettings(formFor("ltx2"), {
      selectedModel: { name: "ltx-2-19b-distilled:fp8", family: "ltx2" },
    });
    await openSection(legacy, "Video");
    expect(legacy.find("[data-test='ltx2-extend-video']").exists()).toBe(false);

    const wrapper = mountSettings(formFor("ltx2"), { selectedModel: extendModel });
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-extend-video']").exists()).toBe(true);
  });

  it("reports how many frames the continuation appends", async () => {
    const form = formFor("ltx2");
    form.frames = 97;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    const wrapper = mountSettings(form, { selectedModel: extendModel });
    await openSection(wrapper, "Video");

    // 97 rendered frames minus the 17 that re-render the source tail.
    expect(wrapper.get("[data-test='ltx2-extend-summary']").text()).toContain("80 new frames");
  });

  it("explains a conflicting source image before submission", async () => {
    const form = formFor("ltx2");
    form.frames = 97;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    form.sourceImage = "AAAA";
    const wrapper = mountSettings(form, { selectedModel: extendModel });
    await openSection(wrapper, "Video");

    expect(wrapper.get("[data-test='ltx2-extend-error']").text()).toContain("source image");
  });

  it("explains a conflicting source video before submission", async () => {
    const form = formFor("ltx2");
    form.frames = 97;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    form.sourceVideo = { filename: "guide.mp4", base64: "BBBB" };
    const wrapper = mountSettings(form, { selectedModel: extendModel });
    await openSection(wrapper, "Video");

    expect(wrapper.get("[data-test='ltx2-extend-error']").text()).toContain("source video");
  });

  it("resets the overlap when the attached video is cleared", async () => {
    const form = formFor("ltx2");
    form.frames = 97;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    form.extendOverlapFrames = 33;
    const wrapper = mountSettings(form, { selectedModel: extendModel });
    await openSection(wrapper, "Video");

    await wrapper.get("[data-test='ltx2-extend-clear']").trigger("click");
    expect(form.extendVideo).toBeNull();
    expect(form.extendOverlapFrames).toBeNull();
  });

  /**
   * Wan continues too (#783) — its handoff is the source clip's final frame
   * as image conditioning — but the field lived inside the LTX-2 suite, so no
   * wan checkpoint could reach it however the host advertised the capability.
   */
  it("offers continuation for an image-conditioned wan checkpoint", async () => {
    const form = formFor("wan");
    form.model = "wan22-i2v-a14b:q5";
    const wrapper = mountSettings(form, {
      selectedModel: {
        name: "wan22-i2v-a14b:q5",
        family: "wan",
        source_image: "required",
        supports_extend: true,
        extend_default_overlap_frames: 17,
      },
    });
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-extend-video']").exists()).toBe(true);
    // …without the LTX-2 pipeline suite coming with it.
    expect(wrapper.find("[data-test='ltx2-controls']").exists()).toBe(false);
  });

  // Wan's engine refuses any overlap but the one frame it was seeded with,
  // and the server's family-wide default of 17 is one of the rejected ones.
  it("offers wan only the single overlap its engine accepts", async () => {
    const form = formFor("wan");
    form.model = "wan22-i2v-a14b:q5";
    form.frames = 53;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    const wrapper = mountSettings(form, {
      selectedModel: {
        name: "wan22-i2v-a14b:q5",
        family: "wan",
        source_image: "required",
        supports_extend: true,
        extend_default_overlap_frames: 17,
      },
    });
    await openSection(wrapper, "Video");

    const select = wrapper.get("[data-test='ltx2-extend-overlap']");
    expect(select.findAll("option").map((option) => option.text())).toEqual(["1"]);
    expect((select.element as HTMLSelectElement).value).toBe("1");
    expect(wrapper.get("[data-test='ltx2-extend-summary']").text()).toContain("52 new frames");
  });

  /**
   * The clamp is only worth anything if the clamped value is the one that
   * leaves the app. Wan's select carries a single option, so `@change` never
   * fires and `extendOverlapFrames` stays null — an absent wire field handed
   * the host its own family-wide default of 17, which `extend_inner` refuses.
   */
  it("submits the overlap it is showing for an untouched wan continuation", async () => {
    const form = formFor("wan");
    form.model = "wan22-i2v-a14b:q5";
    form.frames = 53;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    // The host's advertised value travels on the form, exactly as the picker
    // would have snapshotted it.
    form.extendDefaultOverlapFrames = 17;
    const wrapper = mountSettings(form, {
      selectedModel: {
        name: "wan22-i2v-a14b:q5",
        family: "wan",
        source_image: "required",
        supports_extend: true,
        extend_default_overlap_frames: 17,
      },
    });
    await openSection(wrapper, "Video");

    expect(form.extendOverlapFrames).toBeNull();
    const shown = Number(
      (wrapper.get("[data-test='ltx2-extend-overlap']").element as HTMLSelectElement).value,
    );
    expect(shown).toBe(1);
    expect(buildRequest(form).extend_overlap_frames).toBe(shown);
  });

  it("stays hidden for a text-to-video wan checkpoint", async () => {
    const form = formFor("wan");
    form.model = "wan22-t2v-a14b:q5";
    const wrapper = mountSettings(form, {
      selectedModel: {
        name: "wan22-t2v-a14b:q5",
        family: "wan",
        source_image: "unsupported",
        supports_extend: false,
      },
    });
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-extend-video']").exists()).toBe(false);
  });
});

describe("AdvancedSettings — per-model source-image contract (#772)", () => {
  const wanModel = (sourceImage?: string): ModelEntry =>
    ({
      name: "wan22-t2v-a14b",
      family: "wan",
      downloaded: true,
      default_width: 1280,
      default_height: 720,
      default_steps: 20,
      default_guidance: 5,
      ...(sourceImage === undefined ? {} : { source_image: sourceImage }),
    }) as ModelEntry;

  function wanForm(sourceImage?: string): GenerateForm {
    return reactive({
      ...newGenerateForm(),
      family: "wan",
      model: "wan22-t2v-a14b",
      frames: 81,
      sourceImageCapability: sourceImage ?? null,
    });
  }

  it("never renders a source section — the wells live in the primary form", () => {
    for (const contract of [undefined, "optional", "required", "unsupported"]) {
      const wrapper = mountSettings(wanForm(contract), {
        selectedModel: wanModel(contract),
      });
      expect(accordionTitles(wrapper)).not.toContain("Source image");
      expect(wrapper.find("[data-test='section-source']").exists()).toBe(false);
    }
  });
});

describe("AdvancedSettings — identity conditioning", () => {
  function identityForm(supported: boolean | null = true): GenerateForm {
    return reactive({
      ...newGenerateForm(),
      family: "flux",
      model: "flux-dev:q8",
      steps: 20,
      identitySupported: supported,
    });
  }

  it("renders the Identity section only for a qualified checkpoint", () => {
    expect(accordionTitles(mountSettings(identityForm()))).toContain("Identity");
    for (const supported of [false, null] as const) {
      const wrapper = mountSettings(identityForm(supported));
      expect(accordionTitles(wrapper)).not.toContain("Identity");
      expect(wrapper.find("[data-test='section-identity']").exists()).toBe(false);
    }
  });

  it("leaves both knobs absent until touched so the server default is authoritative", async () => {
    const form = identityForm();
    const wrapper = mountSettings(form);
    const weight = wrapper.get("[data-test='identity-weight']");
    const startStep = wrapper.get("[data-test='identity-start-step']");
    expect((weight.element as HTMLInputElement).value).toBe("");
    expect(weight.attributes("placeholder")).toBe("1.0");
    expect((startStep.element as HTMLInputElement).value).toBe("");
    expect(startStep.attributes("placeholder")).toBe("0");
    // The start step can never reach the step count this print renders.
    expect(startStep.attributes("max")).toBe("19");

    await weight.setValue("0.6");
    await startStep.setValue("3");
    expect(form.identityWeight).toBe(0.6);
    expect(form.identityStartStep).toBe(3);

    await weight.setValue("");
    await startStep.setValue("");
    expect(form.identityWeight).toBeNull();
    expect(form.identityStartStep).toBeNull();
  });

  it("names the refusal inline in the section, never as a toast", async () => {
    const form = identityForm();
    form.identityWeight = 2;
    const wrapper = mountSettings(form);
    // A knob with no photo is exactly what admission refuses.
    expect(wrapper.get("[data-test='identity-error']").text()).toContain(
      "Attach an identity photo",
    );
  });
});

describe("AdvancedSettings — exact size", () => {
  it("records a typed exact size as a manual canvas intent (#1166)", async () => {
    const form = formFor("flux");
    form.width = 1024;
    form.height = 1024;
    const wrapper = mountSettings(form);
    const width = wrapper.get("input[aria-label='Width']");
    await width.setValue(900);
    await width.trigger("change");
    expect(form.width).toBe(896);
    expect(wrapper.emitted("canvas-intent")?.at(-1)).toEqual(["manual"]);

    await wrapper.get("[title='Swap width and height']").trigger("click");
    expect([form.width, form.height]).toEqual([1024, 896]);
    expect(wrapper.emitted("canvas-intent")?.at(-1)).toEqual(["manual"]);
  });
});
