import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useComposerStore, type ComposerPrefill } from "./composer";
import type { OutputMetadata } from "../lib/api/types";

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("composer store", () => {
  it("keeps the legacy scalar prefill shape working (palette / history / jobs)", () => {
    const composer = useComposerStore();
    const scalar: ComposerPrefill = {
      prompt: "a cat",
      model: "flux-dev:q8",
      seed: null,
      width: 1024,
      height: 1024,
      steps: 4,
      guidance: 3.5,
    };
    composer.set(scalar);
    expect(composer.prefill).toMatchObject({ prompt: "a cat", model: "flux-dev:q8" });
    expect(composer.take()).toEqual(scalar);
    expect(composer.prefill).toBeNull();
  });

  it("carries a full-metadata prefill for gallery Reuse settings", () => {
    const composer = useComposerStore();
    const metadata: OutputMetadata = {
      prompt: "a lighthouse at dusk",
      negative_prompt: "blurry",
      model: "sd15:fp16",
      seed: 42,
      steps: 30,
      guidance: 7,
      width: 512,
      height: 512,
      scheduler: "ddim",
      loras: [{ path: "detail.safetensors", scale: 0.8 }],
    };
    composer.set({ metadata });
    const taken = composer.take();
    expect(taken && "metadata" in taken ? taken.metadata : null).toEqual(metadata);
    expect(composer.take()).toBeNull();
  });
});
