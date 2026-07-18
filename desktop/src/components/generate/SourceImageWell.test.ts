import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import { reactive } from "vue";
import SourceImageWell from "./SourceImageWell.vue";
import ImagePickerModal from "./ImagePickerModal.vue";
import MaskEditorModal from "./MaskEditorModal.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
}));

function formFor(family: string): GenerateForm {
  return reactive({ ...newGenerateForm(), family, model: "m" });
}

describe("SourceImageWell", () => {
  beforeEach(() => setActivePinia(createPinia()));
  afterEach(() => (document.body.innerHTML = ""));

  it("sets the source image from an ImagePickerModal pick", async () => {
    const form = formFor("sd15");
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    await wrapper.get("[data-test='source-choose-gallery']").trigger("click");
    wrapper
      .findComponent(ImagePickerModal)
      .vm.$emit("pick", [{ filename: "pick.png", base64: "PICKED" }]);
    await flushPromises();

    expect(form.sourceImage).toBe("PICKED");
  });

  it("gates Edit mask on a source image and applies the painted mask", async () => {
    const form = formFor("sd15");
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    // No source yet → the Edit mask control is absent.
    expect(wrapper.find("[data-test='source-edit-mask']").exists()).toBe(false);

    form.sourceImage = "SRC";
    await flushPromises();
    await wrapper.get("[data-test='source-edit-mask']").trigger("click");
    wrapper.findComponent(MaskEditorModal).vm.$emit("apply", "MASKB64");
    await flushPromises();

    expect(form.maskImage).toBe("MASKB64");
  });

  it("hides the mask controls for families without inpaint support", () => {
    // qwen-image-edit uses qwen-edit source mode → no mask.
    const form = formFor("qwen-image-edit");
    form.sourceImage = "SRC";
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
    expect(wrapper.find("[data-test='source-edit-mask']").exists()).toBe(false);
  });

  describe("source-fit selector", () => {
    it("appears only once a source image is attached", async () => {
      const form = formFor("sd15");
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      expect(wrapper.find("[data-test='source-fit-policy']").exists()).toBe(false);

      form.sourceImage = "SRC";
      await flushPromises();
      const select = wrapper.get("[data-test='source-fit-policy']").element as HTMLSelectElement;
      expect(select.value).toBe("pad-repaint");
      expect(Array.from(select.options).map((o) => o.value)).toEqual([
        "pad-repaint",
        "crop-fill",
        "pad-fit",
        "lanczos-resize",
        "upscale-then-fit",
      ]);
    });

    it("stays visible alongside an existing mask (web parity: the mask is refit)", async () => {
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      form.maskImage = "MASK";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await flushPromises();
      expect(wrapper.find("[data-test='source-fit-policy']").exists()).toBe(true);
    });

    it("maps crop-fill to a centered policy", async () => {
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='source-fit-policy']").setValue("crop-fill");
      expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "center", alignY: "center" });

      await wrapper.get("[data-test='source-fit-policy']").setValue("lanczos-resize");
      expect(form.sourceFit).toEqual({ mode: "lanczos-resize" });

      await wrapper.get("[data-test='source-fit-policy']").setValue("pad-fit");
      expect(form.sourceFit).toEqual({ mode: "pad-fit" });
    });

    it("upscale-then-fit seeds the upscaler from the form's post-generate pick", async () => {
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      form.upscaleModel = "real-esrgan-x2plus:fp16";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='source-fit-policy']").setValue("upscale-then-fit");
      expect(form.sourceFit).toEqual({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x2plus:fp16",
        fit: { mode: "pad-repaint" },
      });
    });

    it("upscale-then-fit falls back to the first known upscaler model", async () => {
      const { useModelStore } = await import("../../stores/models");
      const models = useModelStore();
      models.all = [
        { name: "real-esrgan-x4plus:fp16", family: "upscaler" },
      ] as (typeof models.all)[number][];
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='source-fit-policy']").setValue("upscale-then-fit");
      expect(form.sourceFit).toEqual({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x4plus:fp16",
        fit: { mode: "pad-repaint" },
      });
    });
  });
});
