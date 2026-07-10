import { afterEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { reactive } from "vue";
import ParamPanel from "./ParamPanel.vue";
import ImagePickerModal from "./ImagePickerModal.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
}));

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
