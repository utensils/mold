import { mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import type { ModelEntry } from "../lib/api/types";
import { newGenerateForm } from "../lib/generateForm";
import MobileSequenceOpeningImage from "./MobileSequenceOpeningImage.vue";

installMemoryLocalStorage();

const upscaler = {
  name: "real-esrgan-x4plus:fp16",
  family: "upscaler",
  size_gb: 1,
  is_loaded: false,
  hf_repo: "example/upscaler",
  downloaded: true,
} as ModelEntry;

let wrapper: VueWrapper | null = null;

function mountOpeningImage(props: Record<string, unknown> = {}): VueWrapper {
  wrapper = mount(MobileSequenceOpeningImage, {
    props: {
      form: newGenerateForm(),
      target: { baseUrl: "http://studio:7680", apiKey: "secret" },
      upscalers: [],
      locked: false,
      ...props,
    },
  });
  return wrapper;
}

beforeEach(() => {
  localStorage.clear();
  setActivePinia(createPinia());
  const draft = useSequenceDraftStore();
  draft.hydrate();
  draft.ensureClips(97);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

describe("MobileSequenceOpeningImage", () => {
  it("round-trips an attached opening image through the shared draft", async () => {
    const draft = useSequenceDraftStore();
    mountOpeningImage();
    expect(wrapper!.find("[data-test='mobile-sequence-source-strength']").exists()).toBe(false);

    await wrapper!.get("[data-test='mobile-sequence-source-pick']").trigger("click");
    const picker = wrapper!.getComponent({ name: "MobileImagePickerSheet" });
    expect(picker.props("open")).toBe(true);
    picker.vm.$emit("pick", { filename: "opening.jpg", base64: "wire-bytes" });
    await wrapper!.vm.$nextTick();

    expect(draft.openingImage).toEqual({ filename: "opening.jpg", base64: "wire-bytes" });
    expect(wrapper!.get("[data-test='mobile-sequence-source-preview']").attributes("src")).toBe(
      "data:image/jpeg;base64,wire-bytes",
    );

    await wrapper!.get("[data-test='mobile-sequence-source-clear']").trigger("click");
    expect(draft.openingImage).toBeNull();
  });

  it("forwards the fleet galleries independently of the sequence render target", async () => {
    const gallerySources = [
      {
        id: "studio",
        label: "Studio",
        target: { baseUrl: "http://studio:7680", apiKey: "secret" },
      },
      {
        id: "render",
        label: "Render",
        target: { baseUrl: "http://render:7680", apiKey: "render-secret" },
      },
    ];
    mountOpeningImage({ gallerySources });
    await wrapper!.get("[data-test='mobile-sequence-source-pick']").trigger("click");
    expect(
      wrapper!.getComponent({ name: "MobileImagePickerSheet" }).props("gallerySources"),
    ).toEqual(gallerySources);
  });

  it("writes strength and fit onto the host form", async () => {
    const form = newGenerateForm();
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    mountOpeningImage({ form, upscalers: [upscaler] });

    await wrapper!.get("[data-test='mobile-sequence-source-strength']").setValue("0.6");
    expect(form.strength).toBe(0.6);
    await wrapper!.get("[data-test='mobile-sequence-source-fit']").setValue("upscale-then-fit");
    expect(form.sourceFit).toEqual({
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x4plus:fp16",
      fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    });
  });

  it("disables every control while the sequence is locked", async () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    mountOpeningImage({ locked: true });
    await wrapper!.vm.$nextTick();

    for (const id of [
      "mobile-sequence-source-pick",
      "mobile-sequence-source-clear",
      "mobile-sequence-source-strength",
      "mobile-sequence-source-fit",
    ]) {
      expect((wrapper!.get(`[data-test='${id}']`).element as HTMLButtonElement).disabled, id).toBe(
        true,
      );
    }
  });

  it("sizes its controls from the shared mobile primitives", () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    mountOpeningImage();

    for (const id of ["mobile-sequence-source-pick", "mobile-sequence-source-clear"]) {
      expect(wrapper!.get(`[data-test='${id}']`).classes(), id).toContain("secondary-button");
    }
    expect(wrapper!.get("[data-test='mobile-sequence-source-fit']").classes()).toContain("control");
    expect(
      wrapper!
        .get("[data-test='mobile-sequence-source-strength']")
        .element.closest(".mobile-range-field"),
    ).not.toBeNull();
  });

  it("explains a disabled upscale fit when no upscaler is installed", () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    mountOpeningImage();

    const option = wrapper!
      .get("[data-test='mobile-sequence-source-fit']")
      .find("option[value='upscale-then-fit']");
    expect((option.element as HTMLOptionElement).disabled).toBe(true);
    expect(wrapper!.text()).toContain("Install an upscaler");
    expect(wrapper!.text()).toContain("Applied to the opening image before clip 1 renders.");
  });
});
