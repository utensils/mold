import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useGenerateFormStore } from "./generateForm";
import type { ModelEntry } from "../lib/api/types";

const model: ModelEntry = {
  name: "flux-dev:q8",
  family: "flux",
  default_width: 768,
  default_height: 1152,
  default_steps: 20,
  default_guidance: 4.5,
} as ModelEntry;

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("generateForm store", () => {
  it("persists edits across store re-acquisition", () => {
    const store = useGenerateFormStore();
    store.form.prompt = "a lighthouse at dusk";
    store.form.model = "flux-dev:q8";

    // Re-acquire the store as a fresh view would — Pinia hands back the same
    // singleton, so the form must still hold the edits.
    const again = useGenerateFormStore();
    expect(again.form.prompt).toBe("a lighthouse at dusk");
    expect(again.form.model).toBe("flux-dev:q8");
  });

  it("applyModel routes through applyModelDefaults", () => {
    const store = useGenerateFormStore();
    store.applyModel(model);
    expect(store.form.model).toBe("flux-dev:q8");
    expect(store.form.family).toBe("flux");
    expect(store.form.width).toBe(768);
    expect(store.form.height).toBe(1152);
    expect(store.form.steps).toBe(20);
  });

  it("clearComposer clears prompt/seed/media but keeps model and params", () => {
    const store = useGenerateFormStore();
    store.applyModel(model);
    store.form.prompt = "a cat";
    store.form.originalPrompt = "orig";
    store.form.negativePrompt = "blurry";
    store.form.seed = "42";
    store.form.sourceImage = "base64";
    store.form.maskImage = "base64";

    store.clearComposer();

    expect(store.form.prompt).toBe("");
    expect(store.form.originalPrompt).toBeNull();
    expect(store.form.negativePrompt).toBe("");
    expect(store.form.seed).toBe("");
    expect(store.form.sourceImage).toBeNull();
    expect(store.form.maskImage).toBeNull();
    // Model + shape params survive a ⌘N.
    expect(store.form.model).toBe("flux-dev:q8");
    expect(store.form.family).toBe("flux");
    expect(store.form.width).toBe(768);
    expect(store.form.steps).toBe(20);
  });

  it("resetAll restores defaults while preserving the form reference", () => {
    const store = useGenerateFormStore();
    // A view/child panel captures the reference before the reset.
    const captured = store.form;
    store.form.prompt = "a cat";
    store.form.model = "flux-dev:q8";

    store.resetAll();

    // Same object identity — the captured reference observes the reset.
    expect(store.form).toBe(captured);
    expect(captured.prompt).toBe("");
    expect(captured.model).toBe("");
  });
});
