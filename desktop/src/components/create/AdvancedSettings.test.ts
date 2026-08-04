import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import AdvancedSettings from "./AdvancedSettings.vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import ImagePickerModal from "../generate/ImagePickerModal.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
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
      "Source image",
      "LoRA stack",
      "Output & seed",
    ]);
  });

  it("hides scheduler and negative for FLUX", () => {
    const titles = accordionTitles(mountSettings(formFor("flux")));
    expect(titles).not.toContain("Scheduler & sampling");
    expect(titles).not.toContain("Negative prompt");
    expect(titles).toContain("Source image");
    expect(titles).toContain("Output & seed");
  });

  it("exposes Video (and hides Upscale) for LTX-2", () => {
    const titles = accordionTitles(mountSettings(formFor("ltx2")));
    expect(titles).toContain("Video");
    expect(titles).not.toContain("Upscale after generate");
  });

  it("keeps qwen-edit free of scheduler, negative, and video", () => {
    const titles = accordionTitles(mountSettings(formFor("qwen-image-edit")));
    expect(titles).toEqual(["Source image", "LoRA stack", "Output & seed"]);
  });
});

describe("AdvancedSettings — section ordering contract", () => {
  // The canonical Advanced section order shared with the web drawer (which
  // additionally renders a trailing placement section — desktop owns GPU
  // placement in Settings instead).
  const SECTION_ORDER = [
    "scheduler",
    "negative",
    "source",
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
    expect(sectionIds(wrapper)).toEqual([
      "scheduler",
      "negative",
      "source",
      "lora",
      "upscale",
      "output",
    ]);
  });

  it("renders video sections in the canonical order", () => {
    const wrapper = mountSettings(formFor("ltx2"), { upscalers: [upscaler] });
    expect(sectionIds(wrapper)).toEqual(["negative", "source", "lora", "output", "video"]);
  });

  it("keeps every family's rendered sections a subsequence of the canon", () => {
    const families = ["sdxl", "sd15", "sd3.5", "flux", "qwen-image-edit", "ltx-video", "ltx2"];
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

  it("hides the upscale accordion for video families", () => {
    expect(
      accordionTitles(mountSettings(formFor("ltx2"), { upscalers: [upscaler] })),
    ).not.toContain("Upscale after generate");
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

  it("shows the chained-clips cue when frames exceed one clip for a chainable model", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    form.frames = 241;
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    expect(wrapper.get("[data-test='chain-cue']").text()).toContain("chained clips of 97 frames");
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
});
