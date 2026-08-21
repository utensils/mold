import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it } from "vitest";
import { reactive } from "vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import ImageDropWell from "@studio/components/ImageDropWell.vue";
import SequenceOpeningImageWell from "./SequenceOpeningImageWell.vue";
import { newGenerateForm } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";

// A real 7x4 PNG so the dimension probe reports a size instead of rejecting.
const PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC";

function pngFile(name = "opening.png"): File {
  const binary = atob(PNG_BASE64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new File([bytes], name, { type: "image/png" });
}

beforeEach(() => {
  setActivePinia(createPinia());
  useSequenceDraftStore().ensureClips(97);
});

describe("SequenceOpeningImageWell", () => {
  it("renders in the primary form with a labelled well and no accordion", () => {
    const wrapper = mount(SequenceOpeningImageWell, { props: { form: newGenerateForm() } });
    expect(wrapper.find("[data-test='sequence-opening-image-field']").exists()).toBe(true);
    expect(wrapper.find("[data-test='sequence-opening-image-well']").exists()).toBe(true);
    expect(wrapper.findComponent({ name: "AccordionSection" }).exists()).toBe(false);
    expect(wrapper.text()).toContain("Opening image");
  });

  it("keeps strength and fit controls out of the form until an image is attached", async () => {
    const draft = useSequenceDraftStore();
    const wrapper = mount(SequenceOpeningImageWell, { props: { form: newGenerateForm() } });
    expect(wrapper.find("[data-test='sequence-source-strength']").exists()).toBe(false);
    expect(wrapper.find("[data-test='sequence-source-fit']").exists()).toBe(false);

    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-source-strength']").exists()).toBe(true);
    expect(wrapper.find("[data-test='sequence-source-fit']").exists()).toBe(true);
    expect(wrapper.text()).toContain("Applied to the opening image before clip 1 renders.");
  });

  it("attaches a gallery pick and coerces the maskless source fit", async () => {
    const draft = useSequenceDraftStore();
    const form = reactive(newGenerateForm());
    form.sourceFit = { mode: "pad-repaint" };
    const wrapper = mount(SequenceOpeningImageWell, { props: { form } });

    await wrapper.get("[data-test='sequence-opening-image-gallery']").trigger("click");
    const picker = wrapper.getComponent({ name: "ImagePickerModal" });
    expect(picker.props("open")).toBe(true);
    picker.vm.$emit("pick", [{ filename: "opening.jpg", base64: "wire-bytes" }]);
    await flushPromises();

    expect(draft.openingImage).toEqual({ filename: "opening.jpg", base64: "wire-bytes" });
    expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "center", alignY: "center" });
    expect(wrapper.get(".image-well__preview img").attributes("src")).toBe(
      "data:image/jpeg;base64,wire-bytes",
    );
  });

  it("attaches a dropped file with its probed dimensions and clears back to none", async () => {
    const draft = useSequenceDraftStore();
    const form = reactive(newGenerateForm());
    const wrapper = mount(SequenceOpeningImageWell, { props: { form } });

    wrapper.getComponent(ImageDropWell).vm.$emit("file", pngFile());
    await flushPromises();
    expect(draft.openingImage).toMatchObject({
      filename: "opening.png",
      base64: PNG_BASE64,
      width: 7,
      height: 4,
    });

    await wrapper.get("[data-test='sequence-opening-image-remove']").trigger("click");
    expect(draft.openingImage).toBeNull();
  });

  it("rejects a file that is not a still image", async () => {
    const draft = useSequenceDraftStore();
    const wrapper = mount(SequenceOpeningImageWell, { props: { form: newGenerateForm() } });
    wrapper
      .getComponent(ImageDropWell)
      .vm.$emit("file", new File(["nope"], "clip.txt", { type: "text/plain" }));
    await flushPromises();
    expect(draft.openingImage).toBeNull();
  });

  it("edits strength and fit against the live form", async () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    const form = reactive(newGenerateForm());
    const wrapper = mount(SequenceOpeningImageWell, {
      props: { form, upscalers: [{ name: "real-esrgan-x4plus:fp16" } as ModelEntry] },
    });

    expect(
      (wrapper.get("[data-test='sequence-source-fit']").element as HTMLSelectElement).value,
    ).toBe("crop-fill");
    await wrapper.get("[data-test='sequence-source-strength']").setValue("0.55");
    expect(form.strength).toBe(0.55);
    await wrapper.get("[data-test='sequence-source-fit']").setValue("upscale-then-fit");
    expect(form.sourceFit).toEqual({
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x4plus:fp16",
      fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    });
  });

  it("disables upscale fit when no upscaler is available", () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    const wrapper = mount(SequenceOpeningImageWell, { props: { form: newGenerateForm() } });
    const upscale = wrapper
      .get("[data-test='sequence-source-fit']")
      .findAll("option")
      .find((option) => option.attributes("value") === "upscale-then-fit");
    expect(upscale?.attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("Install an upscaler");
  });

  it("renders the shared thumbnail with the opening-image alt text", () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.jpg", base64: "QUJD" };
    const wrapper = mount(SequenceOpeningImageWell, { props: { form: newGenerateForm() } });
    const image = wrapper.get(".image-well__preview img");
    expect(image.attributes("src")).toBe("data:image/jpeg;base64,QUJD");
    expect(image.attributes("alt")).toBe("Opening sequence image");
  });
});
