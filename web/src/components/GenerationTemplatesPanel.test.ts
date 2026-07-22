import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GenerationTemplatesPanel from "./GenerationTemplatesPanel.vue";
import type { GenerateFormState } from "../types";
import { saveGenerationTemplate } from "../lib/generationTemplates";
import {
  resetNotifications,
  settleConfirm,
  useNotifications,
} from "../lib/toasts";

function makeForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  return {
    version: 3,
    stylePreset: null,
    prompt: "template prompt",
    negativePrompt: "",
    model: "flux-dev:q4",
    modelFamily: "flux",
    width: 1024,
    height: 1024,
    steps: 28,
    guidance: 3.5,
    seedMode: "static",
    seed: 123,
    batchSize: 1,
    strength: 0.75,
    frames: null,
    fps: null,
    scheduler: null,
    cfgPlus: false,
    outputFormat: "png",
    expand: { enabled: false, variations: 1, familyOverride: null },
    imageAttachments: [],
    maskImage: null,
    controlImage: null,
    controlModel: "",
    controlScale: 1,
    upscaleModel: "",
    gifPreview: false,
    audioFile: null,
    audioFilePath: "",
    sourceVideo: null,
    sourceVideoPath: "",
    keyframes: [],
    pipeline: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    cameraControl: null,
    placement: null,
    loras: [],
    enableAudio: null,
    ...overrides,
  };
}

describe("GenerationTemplatesPanel", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    resetNotifications();
  });

  it("saves the current form as a named template", async () => {
    const w = mount(GenerationTemplatesPanel, {
      props: { modelValue: makeForm({ prompt: "saved prompt" }) },
    });

    await w.find("[data-test='template-name']").setValue("Portrait");
    await w.find("[data-test='template-save']").trigger("click");

    expect(w.text()).toContain("Portrait");
    expect(localStorage.getItem("mold.generation.templates.v1")).toContain(
      "saved prompt",
    );
  });

  it("loads a template into the bound form", async () => {
    saveGenerationTemplate("Portrait", makeForm({ prompt: "from template" }));
    const w = mount(GenerationTemplatesPanel, {
      props: { modelValue: makeForm({ prompt: "current" }) },
    });

    await w.find("[data-test='template-load']").trigger("click");

    const emitted = w
      .emitted("update:modelValue")
      ?.at(-1)?.[0] as GenerateFormState;
    expect(emitted.prompt).toBe("from template");
  });

  it("searches, sorts, renames, and deletes templates", async () => {
    saveGenerationTemplate("Zebra", makeForm({ prompt: "z" }));
    saveGenerationTemplate("Alpha", makeForm({ prompt: "a" }));
    const w = mount(GenerationTemplatesPanel, {
      props: { modelValue: makeForm() },
    });

    await w.find("[data-test='template-search']").setValue("alp");
    expect(w.text()).toContain("Alpha");
    expect(w.text()).not.toContain("Zebra");

    // Rename now flows through the app dialog store (requestText).
    await w.find("[data-test='template-rename']").trigger("click");
    expect(useNotifications().confirm?.kind).toBe("text");
    settleConfirm("Beta");
    await w.vm.$nextTick();
    await w.find("[data-test='template-search']").setValue("");
    expect(w.text()).toContain("Beta");

    await w.find("[data-test='template-sort']").setValue("name-asc");
    const names = w
      .findAll("[data-test='template-row-name']")
      .map((n) => n.text());
    expect(names).toEqual(["Beta", "Zebra"]);

    await w.find("[data-test='template-delete']").trigger("click");
    expect(w.text()).not.toContain("Beta");
  });
});
