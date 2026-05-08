import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it } from "vitest";
import GenerateParamsPanel from "./GenerateParamsPanel.vue";
import type { GenerateFormState, ModelInfoExtended } from "../types";

const baseModel: ModelInfoExtended = {
  name: "flux-dev:q4",
  family: "flux",
  size_gb: 12,
  is_loaded: false,
  last_used: null,
  hf_repo: "black-forest-labs/FLUX.1-dev",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 28,
  default_guidance: 3.5,
  description: "",
};

const altModel: ModelInfoExtended = {
  name: "sdxl:fp16",
  family: "sdxl",
  size_gb: 7,
  is_loaded: false,
  last_used: null,
  hf_repo: "stabilityai/stable-diffusion-xl-base-1.0",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 30,
  default_guidance: 7.5,
  description: "",
};

function makeForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  return {
    version: 1,
    model: "flux-dev:q4",
    prompt: "",
    negativePrompt: "",
    width: 1024,
    height: 1024,
    steps: 28,
    guidance: 3.5,
    seed: null,
    batchSize: 1,
    strength: 0.75,
    sourceImage: null,
    scheduler: null,
    frames: null,
    fps: null,
    outputFormat: "png",
    expand: { enabled: false, variations: 1, familyOverride: null },
    placement: null,
    loras: [],
    enableAudio: null,
    ...overrides,
  };
}

function mountPanel(
  form: GenerateFormState = makeForm(),
  models: ModelInfoExtended[] = [baseModel, altModel],
) {
  return mount(GenerateParamsPanel, {
    props: { modelValue: form, models },
  });
}

describe("GenerateParamsPanel", () => {
  beforeEach(() => {
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });

  it("renders a one-line summary when collapsed", () => {
    const w = mountPanel();
    const summary = w.find("[data-test='params-summary']");
    expect(summary.exists()).toBe(true);
    expect(summary.text()).toBe(
      "flux-dev:q4 · 1024×1024 · 28 · g 3.5 · seed random",
    );
    // Body is hidden when collapsed.
    expect(w.find("[data-test='params-body']").exists()).toBe(false);
    w.unmount();
  });

  it("formats a numeric seed in the summary", () => {
    const w = mountPanel(makeForm({ seed: 42 }));
    expect(w.find("[data-test='params-summary']").text()).toBe(
      "flux-dev:q4 · 1024×1024 · 28 · g 3.5 · seed 42",
    );
    w.unmount();
  });

  it("expands when the summary header is clicked", async () => {
    const w = mountPanel();
    expect(w.find("[data-test='params-body']").exists()).toBe(false);
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    expect(w.find("[data-test='params-body']").exists()).toBe(true);
    w.unmount();
  });

  it("shows a dirty dot when steps deviates from the model default", () => {
    const clean = mountPanel(makeForm({ steps: 28 }));
    expect(clean.find("[data-test='params-dirty-dot']").exists()).toBe(false);
    clean.unmount();

    const dirty = mountPanel(makeForm({ steps: 50 }));
    expect(dirty.find("[data-test='params-dirty-dot']").exists()).toBe(true);
    dirty.unmount();
  });

  it("clears LoRAs when selecting a different model via ModelPicker", async () => {
    const form = makeForm({
      model: "flux-dev:q4",
      loras: [{ path: "/loras/foo.safetensors", scale: 1.0 }],
    });
    const w = mountPanel(form);
    // Expand so the ModelPicker mounts.
    await w.find("[data-test='params-summary-toggle']").trigger("click");

    // Surface the panel's `selectModel` handler via the ModelPicker child's
    // emit. ModelPicker fires `select` with a ModelInfoExtended payload.
    const picker = w.findComponent({ name: "ModelPicker" });
    expect(picker.exists()).toBe(true);
    picker.vm.$emit("select", altModel);
    await w.vm.$nextTick();

    const events = w.emitted("update:modelValue");
    expect(events).toBeTruthy();
    const last = events![events!.length - 1][0] as GenerateFormState;
    expect(last.model).toBe("sdxl:fp16");
    expect(last.loras).toEqual([]);
    // Defaults from altModel are applied.
    expect(last.steps).toBe(30);
    expect(last.guidance).toBe(7.5);
  });

  it("persists expanded state to localStorage and restores it on remount", async () => {
    const w = mountPanel();
    await w.find("[data-test='params-summary-toggle']").trigger("click");
    expect(localStorage.getItem("mold.generate.params.expanded")).toBe("true");
    w.unmount();

    // Remount: localStorage value should drive initial expanded state.
    const w2 = mountPanel();
    expect(w2.find("[data-test='params-body']").exists()).toBe(true);
    w2.unmount();
  });
});
