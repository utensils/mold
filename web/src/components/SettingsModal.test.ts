import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SettingsModal from "./SettingsModal.vue";

const defaultModelValue = {
  version: 1 as const,
  model: "flux-dev:q4",
  prompt: "",
  negativePrompt: "",
  width: 1024,
  height: 1024,
  steps: 20,
  guidance: 7.5,
  seed: null,
  batchSize: 1,
  strength: 0.75,
  sourceImage: null,
  scheduler: null,
  frames: null,
  fps: null,
  outputFormat: "png" as const,
  expand: { enabled: false, variations: 1 as const, familyOverride: null },
  placement: null,
};

// Teleport renders outside the wrapper root; attach to document.body and
// query the DOM directly to find teleported content.
function mountModal() {
  return mount(SettingsModal, {
    props: { open: true, modelValue: defaultModelValue, models: [] },
    attachTo: document.body,
  });
}

describe("SettingsModal — catalog auth", () => {
  it("renders the Hugging Face token input", () => {
    const w = mountModal();
    expect(document.body.textContent).toContain("Hugging Face");
    expect(document.body.querySelector("input[name=hf_token]")).not.toBeNull();
    w.unmount();
  });

  it("renders the Civitai token input", () => {
    const w = mountModal();
    expect(document.body.textContent).toContain("Civitai");
    expect(
      document.body.querySelector("input[name=civitai_token]"),
    ).not.toBeNull();
    w.unmount();
  });

  it("renders the Show NSFW toggle", () => {
    const w = mountModal();
    expect(
      document.body.querySelector("input[name=catalog_show_nsfw]"),
    ).not.toBeNull();
    w.unmount();
  });
});
