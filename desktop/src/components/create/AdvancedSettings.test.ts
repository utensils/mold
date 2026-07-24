import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import AdvancedSettings from "./AdvancedSettings.vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import ImagePickerModal from "../generate/ImagePickerModal.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../../lib/ipc", () => ({
  ipc: { localGalleryList: vi.fn(() => Promise.resolve([])), localGalleryDelete: vi.fn() },
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

  it("shows the a2vid conditioning-audio input only for that pipeline", async () => {
    const form = formFor("ltx2");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-audio-file']").exists()).toBe(false);
    await wrapper.get("[data-test='ltx2-pipeline']").setValue("a2vid");
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

  it("shows the reject message for a non-chainable model over budget", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2-19b:fp8";
    form.frames = 241;
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    expect(wrapper.get("[data-test='chain-reject']").text()).toContain(
      "does not support chained video generation",
    );
  });

  it("offers the seven camera presets, disabling them on LTX-2.3", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    const options = wrapper.get("[data-test='camera-motion']").findAll("option");
    for (const o of options) {
      const value = o.attributes("value");
      const shouldDisable = value !== "" && value !== "custom";
      expect(o.attributes("disabled") !== undefined).toBe(shouldDisable);
    }
    expect(wrapper.get("[data-test='camera-motion-23-hint']").text()).toContain("LTX-2 19B");
  });

  it("reveals a custom camera-motion path input", async () => {
    const form = formFor("ltx2");
    const wrapper = mountSettings(form);
    await openSection(wrapper, "Video");
    await wrapper.get("[data-test='camera-motion']").setValue("custom");
    await wrapper.get("[data-test='camera-motion-custom']").setValue("/loras/pan.safetensors");
    expect(form.cameraControl).toBe("/loras/pan.safetensors");
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
