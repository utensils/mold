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
});
