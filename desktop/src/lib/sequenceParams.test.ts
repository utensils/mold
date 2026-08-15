import { describe, expect, it } from "vitest";
import { newGenerateForm } from "./generateForm";
import { sequenceParams } from "./sequenceParams";
import type { ModelEntry } from "./api/types";

describe("sequenceParams", () => {
  it("projects the LIVE generate form onto the shared sequence params", () => {
    const form = newGenerateForm();
    form.model = "ltx-video";
    form.family = "ltx-video";
    form.width = 1216;
    form.height = 704;
    form.fps = 24;
    form.steps = 8;
    form.guidance = 3;
    form.strength = 0.9;
    form.seed = "42";

    expect(sequenceParams(form)).toEqual({
      model: "ltx-video",
      family: "ltx-video",
      width: 1216,
      height: 704,
      fps: 24,
      steps: 8,
      guidance: 3,
      strength: 0.9,
      sourceFitPolicy: { mode: "pad-repaint" },
      upscalerModel: "",
      seed: "42",
    });

    // LIVE values: later edits are reflected, never a snapshot.
    form.steps = 12;
    expect(sequenceParams(form).steps).toBe(12);
  });

  it("prefers the selected installed entry's family over the form copy", () => {
    const form = newGenerateForm();
    form.model = "cv:12345";
    form.family = "";
    const entry = { name: "cv:12345", family: "ltx2" } as ModelEntry;
    expect(sequenceParams(form, entry).family).toBe("ltx2");
  });

  it("projects the effective fixed guidance for a distilled sequence", () => {
    const form = newGenerateForm();
    form.model = "hf:opaque/distilled-checkpoint";
    form.family = "ltx2";
    form.guidance = 7;
    form.guidanceCapabilities = {
      adjustable: false,
      supports_negative_prompt: false,
      fixed_scale: 1,
    };

    expect(sequenceParams(form).guidance).toBe(1);
    expect(form.guidance).toBe(7);
  });
});
