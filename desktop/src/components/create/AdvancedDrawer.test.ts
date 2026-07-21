import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import AdvancedDrawer from "./AdvancedDrawer.vue";
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
  if (!acc) throw new Error(`accordion "${title}" not found`);
  acc.vm.$emit("toggle");
  await flushPromises();
}

function mountDrawer(form: GenerateForm, extra: Record<string, unknown> = {}) {
  return mount(AdvancedDrawer, {
    props: { open: true, form, ...extra },
    attachTo: document.body,
  });
}

describe("AdvancedDrawer — capability matrix", () => {
  it("shows scheduler, negative, and no video for SDXL", () => {
    const titles = accordionTitles(mountDrawer(formFor("sdxl")));
    expect(titles).toEqual([
      "Scheduler & sampling",
      "Negative prompt",
      "Source image",
      "LoRA stack",
      "Output & seed",
    ]);
  });

  it("hides scheduler and negative for FLUX", () => {
    const titles = accordionTitles(mountDrawer(formFor("flux")));
    expect(titles).not.toContain("Scheduler & sampling");
    expect(titles).not.toContain("Negative prompt");
    expect(titles).toContain("Source image");
    expect(titles).toContain("Output & seed");
  });

  it("exposes Video (and hides Upscale) for LTX-2", () => {
    const titles = accordionTitles(mountDrawer(formFor("ltx2")));
    expect(titles).toContain("Video");
    expect(titles).not.toContain("Upscale after generate");
  });

  it("keeps qwen-edit free of scheduler, negative, and video", () => {
    const titles = accordionTitles(mountDrawer(formFor("qwen-image-edit")));
    expect(titles).toEqual(["Source image", "LoRA stack", "Output & seed"]);
  });
});

describe("AdvancedDrawer — output & seed", () => {
  it("offers the family output formats and exact size override with a swap", async () => {
    const form = formFor("flux");
    const wrapper = mountDrawer(form);
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
    const wrapper = mountDrawer(form);
    await openSection(wrapper, "Output & seed");
    expect(wrapper.find("[data-test='advanced-seed-value']").exists()).toBe(false);
    form.seed = "42";
    await flushPromises();
    expect(wrapper.find("[data-test='advanced-seed-value']").exists()).toBe(true);
  });
});

describe("AdvancedDrawer — negative prompt", () => {
  it("appends quick-add words to the negative prompt", async () => {
    const form = formFor("sdxl");
    const wrapper = mountDrawer(form);
    await openSection(wrapper, "Negative prompt");
    await wrapper.get("[data-test='neg-add-blurry']").trigger("click");
    await wrapper.get("[data-test='neg-add-watermark']").trigger("click");
    expect(form.negativePrompt).toBe("blurry, watermark");
  });
});

describe("AdvancedDrawer — upscale", () => {
  const upscaler = { name: "real-esrgan-x4plus", downloaded: false } as ModelEntry;

  it("offers Off plus the known upscalers for image families", async () => {
    const form = formFor("flux");
    const wrapper = mountDrawer(form, { upscalers: [upscaler] });
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
    expect(accordionTitles(mountDrawer(formFor("ltx2"), { upscalers: [upscaler] }))).not.toContain(
      "Upscale after generate",
    );
  });
});

describe("AdvancedDrawer — video (LTX-2)", () => {
  it("reveals the pipeline controls behind the LTX-2 disclosure", async () => {
    const wrapper = mountDrawer(formFor("ltx2"));
    await openSection(wrapper, "Video");
    expect(wrapper.find("[data-test='ltx2-disclosure']").exists()).toBe(true);
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    expect(wrapper.find("[data-test='ltx2-pipeline']").exists()).toBe(true);
    expect(wrapper.find("[data-test='ltx2-spatial']").exists()).toBe(true);
    expect(wrapper.find("[data-test='ltx2-temporal']").exists()).toBe(true);
  });

  it("reveals retake range only when the pipeline is retake", async () => {
    const form = formFor("ltx2");
    const wrapper = mountDrawer(form);
    await openSection(wrapper, "Video");
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    expect(wrapper.find("[data-test='ltx2-retake-start']").exists()).toBe(false);
    await wrapper.get("[data-test='ltx2-pipeline']").setValue("retake");
    expect(form.pipeline).toBe("retake");
    expect(wrapper.find("[data-test='ltx2-retake-start']").exists()).toBe(true);
  });

  it("shows the a2vid conditioning-audio input only for that pipeline", async () => {
    const form = formFor("ltx2");
    const wrapper = mountDrawer(form);
    await openSection(wrapper, "Video");
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    expect(wrapper.find("[data-test='ltx2-audio-file']").exists()).toBe(false);
    await wrapper.get("[data-test='ltx2-pipeline']").setValue("a2vid");
    expect(wrapper.find("[data-test='ltx2-audio-file']").exists()).toBe(true);
  });

  it("adds a keyframe from the picker", async () => {
    const form = formFor("ltx2");
    const wrapper = mountDrawer(form);
    await openSection(wrapper, "Video");
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    await wrapper.get("[data-test='ltx2-add-keyframe']").trigger("click");
    wrapper.findComponent(ImagePickerModal).vm.$emit("pick", [{ filename: "k.png", base64: "KB" }]);
    await flushPromises();
    expect(form.keyframes).toHaveLength(1);
    expect(form.keyframes[0]!.frame).toBe(0);
  });

  it("shows the chained-clips cue when frames exceed one clip for a chainable model", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    form.frames = 241;
    const wrapper = mountDrawer(form);
    await openSection(wrapper, "Video");
    expect(wrapper.get("[data-test='chain-cue']").text()).toContain("chained clips of 97 frames");
  });

  it("shows the reject message for a non-chainable model over budget", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2-19b:fp8";
    form.frames = 241;
    const wrapper = mountDrawer(form);
    await openSection(wrapper, "Video");
    expect(wrapper.get("[data-test='chain-reject']").text()).toContain(
      "does not support chained video generation",
    );
  });

  it("offers the seven camera presets, disabling them on LTX-2.3", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    const wrapper = mountDrawer(form);
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
    const wrapper = mountDrawer(form);
    await openSection(wrapper, "Video");
    await wrapper.get("[data-test='camera-motion']").setValue("custom");
    await wrapper.get("[data-test='camera-motion-custom']").setValue("/loras/pan.safetensors");
    expect(form.cameraControl).toBe("/loras/pan.safetensors");
  });
});

describe("AdvancedDrawer — reset and badge", () => {
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
    const wrapper = mountDrawer(form, { selectedModel: model });
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
    const wrapper = mountDrawer(form);
    expect(wrapper.find("[data-test='advanced-active']").text()).toContain("2 active");
  });

  it("emits close from Done", () => {
    const wrapper = mountDrawer(formFor("flux"));
    wrapper.get("[data-test='advanced-done']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });
});
