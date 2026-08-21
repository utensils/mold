import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import SequenceOpeningImagePanel from "./SequenceOpeningImagePanel.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { GenerateFormState } from "../../types";

const PNG_7x4 = "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC";

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, ...overrides };
}

function factory(overrides: Partial<GenerateFormState> = {}) {
  const pinia = createPinia();
  setActivePinia(pinia);
  return mount(SequenceOpeningImagePanel, {
    props: { modelValue: baseForm(overrides) },
    global: { plugins: [pinia] },
  });
}

function pngFile(name = "opening.png") {
  const bytes = Uint8Array.from(atob(PNG_7x4), (c) => c.charCodeAt(0));
  return new File([bytes], name, { type: "image/png" });
}

beforeEach(() => localStorage.clear());
afterEach(() => __testing__.resetForTest());

describe("SequenceOpeningImagePanel — the primary-form opening frame", () => {
  it("renders in the primary form with the shared source-media language", () => {
    const wrapper = factory();
    expect(
      wrapper.find("[data-test='sequence-opening-image-panel']").exists(),
    ).toBe(true);
    expect(
      wrapper.find("[data-test='sequence-opening-image-well']").exists(),
    ).toBe(true);
  });

  it("stores a dropped PNG on the shared draft and coerces the maskless fit", async () => {
    const wrapper = factory({ sourceFitPolicy: { mode: "pad-repaint" } });
    await wrapper
      .get("[data-test='sequence-opening-image-well']")
      .trigger("drop", { dataTransfer: { files: [pngFile()] } });
    await flushPromises();

    const draft = useSequenceDraftStore();
    expect(draft.openingImage).toMatchObject({
      filename: "opening.png",
      base64: PNG_7x4,
      width: 7,
      height: 4,
    });
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      sourceFitPolicy: { mode: "crop-fill" },
    });
  });

  it("refuses a non-PNG/JPEG file with a visible error and no draft write", async () => {
    const wrapper = factory();
    await wrapper
      .get("[data-test='sequence-opening-image-well']")
      .trigger("drop", {
        dataTransfer: {
          files: [new File(["gif"], "bad.gif", { type: "image/gif" })],
        },
      });
    await flushPromises();

    expect(useSequenceDraftStore().openingImage).toBeNull();
    expect(
      wrapper.get("[data-test='sequence-opening-image-error']").text(),
    ).toContain("PNG or JPEG");
  });

  it("clears the error and asks the page for its gallery picker", async () => {
    const wrapper = factory();
    await wrapper
      .get("[data-test='sequence-opening-image-well']")
      .trigger("drop", {
        dataTransfer: {
          files: [new File(["gif"], "bad.gif", { type: "image/gif" })],
        },
      });
    await flushPromises();
    expect(
      wrapper.find("[data-test='sequence-opening-image-error']").exists(),
    ).toBe(true);

    await wrapper
      .get("[data-test='sequence-opening-image-gallery']")
      .trigger("click");
    expect(
      wrapper.find("[data-test='sequence-opening-image-error']").exists(),
    ).toBe(false);
    expect(wrapper.emitted("open-picker")).toHaveLength(1);
  });

  it("removes the attached opening image", async () => {
    const wrapper = factory();
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: PNG_7x4 };
    await wrapper.vm.$nextTick();

    await wrapper
      .get("[data-test='sequence-opening-image-remove']")
      .trigger("click");
    expect(draft.openingImage).toBeNull();
  });

  it("exposes strength and maskless fit only once an image is attached", async () => {
    const wrapper = factory({
      strength: 0.8,
      sourceFitPolicy: { mode: "crop-fill" },
    });
    expect(
      wrapper.find("[data-test='sequence-source-strength']").exists(),
    ).toBe(false);
    expect(wrapper.find("[data-test='sequence-source-fit']").exists()).toBe(
      false,
    );

    useSequenceDraftStore().openingImage = {
      filename: "opening.png",
      base64: PNG_7x4,
    };
    await wrapper.vm.$nextTick();

    expect(
      wrapper.get("[data-test='sequence-source-strength']").text(),
    ).toContain("0.80");
    const fit = wrapper.get("[data-test='sequence-source-fit']");
    expect((fit.element as HTMLSelectElement).value).toBe("crop-fill");
    expect(fit.findAll("option").map((option) => option.text())).toEqual([
      "Crop to fill",
      "Fit with borders",
      "Stretch to fill",
      "Upscale, then crop",
    ]);
    await fit.setValue("lanczos-resize");
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      sourceFitPolicy: { mode: "lanczos-resize" },
    });
  });

  it("renders the shared thumbnail for a JPEG opening image", async () => {
    const wrapper = factory();
    useSequenceDraftStore().openingImage = {
      filename: "opening.jpg",
      base64: "QUJD",
    };
    await wrapper.vm.$nextTick();

    const image = wrapper.get(
      "[data-test='sequence-opening-image-panel'] .image-well__preview img",
    );
    expect(image.attributes("src")).toBe("data:image/jpeg;base64,QUJD");
    expect(image.attributes("alt")).toBe("Opening sequence image");
  });
});
