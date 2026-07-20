import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import ParamPanel from "./ParamPanel.vue";
import ImagePickerModal from "./ImagePickerModal.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));

vi.mock("../../lib/ipc", () => ({
  ipc: {
    localGalleryList: vi.fn(() => Promise.resolve([])),
    localGalleryDelete: vi.fn(),
  },
}));

// The embedded ImagePickerModal reads the multi-host gallery store.
beforeEach(() => setActivePinia(createPinia()));

function formFor(family: string): GenerateForm {
  return reactive({ ...newGenerateForm(), family });
}

describe("ParamPanel — LTX-2 advanced disclosure", () => {
  afterEach(() => (document.body.innerHTML = ""));

  it("hides the disclosure for a still-image family", () => {
    const wrapper = mount(ParamPanel, { props: { form: formFor("flux") } });
    expect(wrapper.find("[data-test='ltx2-disclosure']").exists()).toBe(false);
  });

  it("hides the disclosure for plain ltx-video (not ltx2)", () => {
    const wrapper = mount(ParamPanel, { props: { form: formFor("ltx-video") } });
    expect(wrapper.find("[data-test='ltx2-disclosure']").exists()).toBe(false);
  });

  it("shows the pipeline + upscale controls for ltx2 when expanded", async () => {
    const wrapper = mount(ParamPanel, { props: { form: formFor("ltx2") } });
    expect(wrapper.find("[data-test='ltx2-disclosure']").exists()).toBe(true);
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    expect(wrapper.find("[data-test='ltx2-pipeline']").exists()).toBe(true);
    expect(wrapper.find("[data-test='ltx2-spatial']").exists()).toBe(true);
    expect(wrapper.find("[data-test='ltx2-temporal']").exists()).toBe(true);
  });

  it("reveals retake range inputs only when the pipeline is retake", async () => {
    const form = formFor("ltx2");
    const wrapper = mount(ParamPanel, { props: { form } });
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    expect(wrapper.find("[data-test='ltx2-retake-start']").exists()).toBe(false);

    await wrapper.get("[data-test='ltx2-pipeline']").setValue("retake");
    expect(form.pipeline).toBe("retake");
    expect(wrapper.find("[data-test='ltx2-retake-start']").exists()).toBe(true);
  });

  it("adds a keyframe from the ImagePickerModal pick", async () => {
    const form = formFor("ltx2");
    const wrapper = mount(ParamPanel, { props: { form }, attachTo: document.body });
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    await wrapper.get("[data-test='ltx2-add-keyframe']").trigger("click");
    wrapper
      .findComponent(ImagePickerModal)
      .vm.$emit("pick", [{ filename: "k.png", base64: "KB64" }]);
    await flushPromises();

    expect(form.keyframes).toHaveLength(1);
    expect(form.keyframes[0]!.image.base64).toBe("KB64");
    expect(form.keyframes[0]!.frame).toBe(0);
  });

  it("shows the conditioning-audio input only for the a2vid pipeline", async () => {
    const form = formFor("ltx2");
    const wrapper = mount(ParamPanel, { props: { form } });
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    // Auto pipeline → no audio input.
    expect(wrapper.find("[data-test='ltx2-audio-file']").exists()).toBe(false);

    await wrapper.get("[data-test='ltx2-pipeline']").setValue("a2vid");
    expect(form.pipeline).toBe("a2vid");
    expect(wrapper.find("[data-test='ltx2-audio-file']").exists()).toBe(true);

    // Switching away hides it again.
    await wrapper.get("[data-test='ltx2-pipeline']").setValue("keyframe");
    expect(wrapper.find("[data-test='ltx2-audio-file']").exists()).toBe(false);
  });

  it("gives the size and LTX-2 controls accessible names", async () => {
    const wrapper = mount(ParamPanel, { props: { form: formFor("ltx2") } });
    // The width/height pair shares one visible label, so each needs its own name.
    expect(wrapper.find("input[aria-label='Width']").exists()).toBe(true);
    expect(wrapper.find("input[aria-label='Height']").exists()).toBe(true);
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    expect(wrapper.get("[data-test='ltx2-pipeline']").attributes("aria-label")).toBe(
      "LTX-2 pipeline mode",
    );
    expect(wrapper.get("[data-test='ltx2-spatial']").attributes("aria-label")).toBe(
      "Spatial upscale",
    );
  });

  it("maps the spatial upscale select onto the form (native → null)", async () => {
    const form = formFor("ltx2");
    const wrapper = mount(ParamPanel, { props: { form } });
    await wrapper.get("[data-test='ltx2-disclosure']").trigger("click");
    await wrapper.get("[data-test='ltx2-spatial']").setValue("x2");
    expect(form.spatialUpscale).toBe("x2");
    await wrapper.get("[data-test='ltx2-spatial']").setValue("");
    expect(form.spatialUpscale).toBeNull();
  });
});

describe("ParamPanel — size presets", () => {
  it("applies a preset to width and height and reads back as that preset", async () => {
    const form = formFor("sdxl");
    const wrapper = mount(ParamPanel, { props: { form } });
    await wrapper.get("[data-test='size-preset']").setValue("3:4 · 896×1152");
    expect(form.width).toBe(896);
    expect(form.height).toBe(1152);
    expect((wrapper.get("[data-test='size-preset']").element as HTMLSelectElement).value).toBe(
      "3:4 · 896×1152",
    );
  });

  it("shows Custom when the manual inputs leave the preset list", async () => {
    const form = formFor("sdxl");
    form.width = 1000;
    form.height = 1000;
    const wrapper = mount(ParamPanel, { props: { form } });
    expect((wrapper.get("[data-test='size-preset']").element as HTMLSelectElement).value).toBe(
      "custom",
    );
  });

  it("shows a live visual aspect and orientation helper", () => {
    const form = formFor("qwen-image-edit");
    form.width = 1328;
    form.height = 1328;
    const wrapper = mount(ParamPanel, { props: { form } });
    expect(wrapper.get("[data-test='aspect-helper']").text()).toContain("1:1");
    expect(wrapper.get("[data-test='aspect-helper']").text()).toContain("Square");
    expect(wrapper.get("[data-test='aspect-shape']").attributes("style")).toContain(
      "aspect-ratio: 1328 / 1328",
    );
    expect(wrapper.get("[data-test='aspect-shape']").attributes("style")).toContain("height: 26px");
    expect(wrapper.get("[data-test='aspect-shape']").attributes("style")).not.toContain(
      "width: 30px",
    );
  });
});

describe("ParamPanel — seed mode", () => {
  it("starts Random with an empty seed and hides the input", () => {
    const wrapper = mount(ParamPanel, { props: { form: formFor("flux") } });
    expect(wrapper.get("[data-test='seed-mode-random']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.find("[data-test='seed-input']").exists()).toBe(false);
    expect(wrapper.text()).toContain("New seed every print");
  });

  it("switching to Fixed fills the field (last seed preferred) and locks it", async () => {
    const form = formFor("flux");
    const wrapper = mount(ParamPanel, { props: { form, lastSeed: 1234 } });
    await wrapper.get("[data-test='seed-mode-fixed']").trigger("click");
    expect(form.seed).toBe("1234");
    expect(wrapper.get("[data-test='seed-mode-fixed']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.find("[data-test='seed-input']").exists()).toBe(true);
  });

  it("lock-last shortcut jumps straight from Random to that seed", async () => {
    const form = formFor("flux");
    const wrapper = mount(ParamPanel, { props: { form, lastSeed: 77 } });
    await wrapper.get("[data-test='lock-last-seed']").trigger("click");
    expect(form.seed).toBe("77");
    expect(wrapper.get("[data-test='seed-mode-fixed']").attributes("aria-pressed")).toBe("true");
  });

  it("switching back to Random clears the seed", async () => {
    const form = formFor("flux");
    form.seed = "42";
    const wrapper = mount(ParamPanel, { props: { form } });
    await wrapper.get("[data-test='seed-mode-random']").trigger("click");
    expect(form.seed).toBe("");
  });
});

describe("ParamPanel — post-generate upscale", () => {
  const upscaler = {
    name: "real-esrgan-x4plus",
    family: "upscaler",
    size_gb: 0.06,
    is_loaded: false,
    hf_repo: "r",
    default_steps: 1,
    default_guidance: 1,
    default_width: 0,
    default_height: 0,
    description: "",
    downloaded: false,
  };

  it("offers Off plus the known upscalers for image families", async () => {
    const form = formFor("flux");
    const wrapper = mount(ParamPanel, { props: { form, upscalers: [upscaler] } });
    const select = wrapper.get("[data-test='upscale-select']");
    expect(select.findAll("option").map((o) => o.text())).toEqual([
      "Off",
      "real-esrgan-x4plus (downloads on first use)",
    ]);
    await select.setValue("real-esrgan-x4plus");
    expect(form.upscaleModel).toBe("real-esrgan-x4plus");
  });

  it("hides the control for video families and when no upscalers exist", () => {
    expect(
      mount(ParamPanel, { props: { form: formFor("ltx2"), upscalers: [upscaler] } })
        .find("[data-test='upscale-select']")
        .exists(),
    ).toBe(false);
    expect(
      mount(ParamPanel, { props: { form: formFor("flux") } })
        .find("[data-test='upscale-select']")
        .exists(),
    ).toBe(false);
  });
});

describe("ParamPanel — seed mode edge cases", () => {
  it("clearing the field in Fixed mode keeps the input mounted with a hint", async () => {
    const form = formFor("flux");
    form.seed = "42";
    const wrapper = mount(ParamPanel, { props: { form } });
    await wrapper.get("[data-test='seed-input']").setValue("");
    // The input must NOT unmount mid-edit, and the mode stays Fixed…
    expect(wrapper.find("[data-test='seed-input']").exists()).toBe(true);
    expect(wrapper.get("[data-test='seed-mode-fixed']").attributes("aria-pressed")).toBe("true");
    // …with an honest note that the wire will treat it as random.
    expect(wrapper.get("[data-test='seed-hint']").text()).toContain("random seed will be used");
  });

  it("non-numeric seed text warns instead of silently generating random", async () => {
    const form = formFor("flux");
    form.seed = "42";
    const wrapper = mount(ParamPanel, { props: { form } });
    await wrapper.get("[data-test='seed-input']").setValue("banana");
    expect(wrapper.get("[data-test='seed-hint']").text()).toContain("Not a number");
    await wrapper.get("[data-test='seed-input']").setValue("1234");
    expect(wrapper.find("[data-test='seed-hint']").exists()).toBe(false);
  });
});

describe("ParamPanel — chained-clip routing cue", () => {
  function distilledForm(): GenerateForm {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    return form;
  }

  it("shows no cue at or below the single-clip cap", () => {
    const form = distilledForm();
    form.frames = 97;
    const wrapper = mount(ParamPanel, { props: { form } });
    expect(wrapper.find("[data-test='chain-cue']").exists()).toBe(false);
    expect(wrapper.find("[data-test='chain-reject']").exists()).toBe(false);
  });

  it("shows the chained-clips cue when frames exceed one clip for a chainable model", () => {
    const form = distilledForm();
    form.frames = 241;
    const wrapper = mount(ParamPanel, { props: { form } });
    const cue = wrapper.get("[data-test='chain-cue']");
    expect(cue.text()).toContain("Will render as");
    expect(cue.text()).toContain("3");
    expect(cue.text()).toContain("chained clips of 97 frames (motion-tail 17)");
    expect(cue.text()).toContain("substantially longer");
  });

  it("shows the cue with motion-tail 0 for ltx-video (no context handoff)", () => {
    const form = formFor("ltx-video");
    form.model = "ltx-video-0.9.8-13b-dev:bf16";
    form.frames = 241;
    const wrapper = mount(ParamPanel, { props: { form } });
    expect(wrapper.get("[data-test='chain-cue']").text()).toContain("(motion-tail 0)");
  });

  it("shows the reject message for a non-chainable model over budget", () => {
    const form = formFor("ltx2");
    form.model = "ltx-2-19b:fp8";
    form.frames = 241;
    const wrapper = mount(ParamPanel, { props: { form } });
    expect(wrapper.find("[data-test='chain-cue']").exists()).toBe(false);
    expect(wrapper.get("[data-test='chain-reject']").text()).toContain(
      "does not support chained video generation",
    );
  });

  it("replaces the chain cue with a directed error for unsupported advanced settings", () => {
    const form = distilledForm();
    form.frames = 177;
    form.negativePrompt = "flicker";

    const wrapper = mount(ParamPanel, { props: { form } });
    expect(wrapper.find("[data-test='chain-cue']").exists()).toBe(false);
    expect(wrapper.get("[data-test='chain-compatibility-error']").text()).toContain(
      "can’t preserve",
    );
  });
});

describe("ParamPanel — camera motion (LTX-2)", () => {
  it("is hidden for still-image and plain ltx-video families", () => {
    expect(
      mount(ParamPanel, { props: { form: formFor("flux") } })
        .find("[data-test='camera-motion']")
        .exists(),
    ).toBe(false);
    expect(
      mount(ParamPanel, { props: { form: formFor("ltx-video") } })
        .find("[data-test='camera-motion']")
        .exists(),
    ).toBe(false);
  });

  it("offers None, the seven presets, and a custom-path option for ltx2", () => {
    const wrapper = mount(ParamPanel, { props: { form: formFor("ltx2") } });
    const options = wrapper
      .get("[data-test='camera-motion']")
      .findAll("option")
      .map((o) => o.attributes("value"));
    expect(options).toEqual([
      "",
      "dolly-in",
      "dolly-left",
      "dolly-out",
      "dolly-right",
      "jib-down",
      "jib-up",
      "static",
      "custom",
    ]);
  });

  it("maps a preset pick onto the form and None back to null", async () => {
    const form = formFor("ltx2");
    const wrapper = mount(ParamPanel, { props: { form } });
    await wrapper.get("[data-test='camera-motion']").setValue("dolly-in");
    expect(form.cameraControl).toBe("dolly-in");
    await wrapper.get("[data-test='camera-motion']").setValue("");
    expect(form.cameraControl).toBeNull();
  });

  it("reveals a path input for Custom and writes the typed path to the form", async () => {
    const form = formFor("ltx2");
    const wrapper = mount(ParamPanel, { props: { form } });
    expect(wrapper.find("[data-test='camera-motion-custom']").exists()).toBe(false);
    await wrapper.get("[data-test='camera-motion']").setValue("custom");
    const input = wrapper.get("[data-test='camera-motion-custom']");
    await input.setValue("/loras/pan-up.safetensors");
    expect(form.cameraControl).toBe("/loras/pan-up.safetensors");
  });

  it("hydrates the select from an existing form value (preset and custom path)", () => {
    const preset = formFor("ltx2");
    preset.cameraControl = "jib-down";
    expect(
      (
        mount(ParamPanel, { props: { form: preset } }).get("[data-test='camera-motion']")
          .element as HTMLSelectElement
      ).value,
    ).toBe("jib-down");

    const custom = formFor("ltx2");
    custom.cameraControl = "/loras/x.safetensors";
    const wrapper = mount(ParamPanel, { props: { form: custom } });
    expect((wrapper.get("[data-test='camera-motion']").element as HTMLSelectElement).value).toBe(
      "custom",
    );
    expect(
      (wrapper.get("[data-test='camera-motion-custom']").element as HTMLInputElement).value,
    ).toBe("/loras/x.safetensors");
  });

  it("disables presets for LTX-2.3 models and explains why (custom path stays open)", () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    const wrapper = mount(ParamPanel, { props: { form } });
    const options = wrapper.get("[data-test='camera-motion']").findAll("option");
    for (const o of options) {
      const value = o.attributes("value");
      const shouldDisable = value !== "" && value !== "custom";
      expect(o.attributes("disabled") !== undefined).toBe(shouldDisable);
    }
    expect(wrapper.get("[data-test='camera-motion-23-hint']").text()).toContain("LTX-2 19B");
  });
});
