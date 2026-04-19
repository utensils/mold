import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { useGenerateForm } from "./useGenerateForm";
import type { ModelInfoExtended } from "../types";

const STORAGE_KEY = "mold.generate.form";

function makeModel(
  overrides: Partial<ModelInfoExtended> = {},
): ModelInfoExtended {
  return {
    name: "flux2-klein:q4",
    family: "flux2",
    size_gb: 6,
    is_loaded: false,
    last_used: null,
    hf_repo: "black-forest-labs/FLUX.2-Klein",
    downloaded: true,
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    ...overrides,
  };
}

describe("useGenerateForm", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("hydrates defaults when localStorage is empty", () => {
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
    expect(form.state.value.width).toBe(1024);
    expect(form.state.value.height).toBe(1024);
    expect(form.state.value.steps).toBe(20);
    expect(form.state.value.batchSize).toBe(1);
    expect(form.state.value.outputFormat).toBe("png");
    expect(form.state.value.sourceImage).toBeNull();
  });

  it("merges a persisted snapshot over defaults but drops the sourceImage field", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 1,
        prompt: "a cat",
        model: "flux-dev:q4",
        width: 512,
        height: 768,
        // sourceImage should never be read back from storage even if someone
        // injects it — base64 lives in memory only.
        sourceImage: { kind: "upload", filename: "x.png", base64: "AAAA" },
      }),
    );

    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("a cat");
    expect(form.state.value.model).toBe("flux-dev:q4");
    expect(form.state.value.width).toBe(512);
    expect(form.state.value.height).toBe(768);
    expect(form.state.value.sourceImage).toBeNull();
    // Untouched fields fall back to defaults.
    expect(form.state.value.steps).toBe(20);
  });

  it("discards a snapshot with a mismatched version to avoid stale schemas", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: 99, prompt: "stale" }),
    );
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
  });

  it("swallows malformed JSON without throwing", () => {
    localStorage.setItem(STORAGE_KEY, "{not json");
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
  });

  it("debounces persistence and strips sourceImage from the written snapshot", async () => {
    const form = useGenerateForm();
    form.state.value.prompt = "a fox";
    form.state.value.sourceImage = {
      kind: "upload",
      filename: "x.png",
      base64: "SECRETBYTES",
    };
    await nextTick();

    // Watch fires but persist is debounced by 300ms.
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();

    vi.advanceTimersByTime(300);

    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!);
    expect(parsed.prompt).toBe("a fox");
    expect(parsed.sourceImage).toBeUndefined();
    expect(raw).not.toContain("SECRETBYTES");
  });

  it("applyModelDefaults copies model defaults and clears video fields for non-video families", () => {
    const form = useGenerateForm();
    form.state.value.frames = 25;
    form.state.value.fps = 24;

    form.applyModelDefaults(
      makeModel({
        name: "sdxl:fp16",
        family: "sdxl",
        default_width: 1024,
        default_height: 1024,
        default_steps: 30,
        default_guidance: 7.5,
      }),
    );

    expect(form.state.value.model).toBe("sdxl:fp16");
    expect(form.state.value.steps).toBe(30);
    expect(form.state.value.guidance).toBe(7.5);
    expect(form.state.value.frames).toBeNull();
    expect(form.state.value.fps).toBeNull();
  });

  it("applyModelDefaults seeds frame/fps for video families when absent", () => {
    const form = useGenerateForm();
    form.state.value.frames = null;
    form.state.value.fps = null;

    form.applyModelDefaults(
      makeModel({ name: "ltx-video:fp16", family: "ltx-video" }),
    );

    expect(form.state.value.frames).toBe(25);
    expect(form.state.value.fps).toBe(24);
  });

  it("applyModelDefaults preserves user-chosen frame/fps for video families", () => {
    const form = useGenerateForm();
    form.state.value.frames = 49;
    form.state.value.fps = 30;

    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));

    expect(form.state.value.frames).toBe(49);
    expect(form.state.value.fps).toBe(30);
  });

  it("toRequest maps camelCase state to snake_case wire payload", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      prompt: "a cat",
      negativePrompt: "blurry",
      model: "sdxl:fp16",
      width: 1024,
      height: 1024,
      steps: 30,
      guidance: 7.5,
      seed: 42,
      batchSize: 2,
      outputFormat: "png",
      scheduler: "ddim",
      strength: 0.8,
      frames: null,
      fps: null,
      expand: { enabled: true, variations: 3, familyOverride: null },
      sourceImage: {
        kind: "upload",
        filename: "src.png",
        base64: "AAAA",
      },
    });

    const wire = form.toRequest();
    expect(wire).toMatchObject({
      prompt: "a cat",
      negative_prompt: "blurry",
      model: "sdxl:fp16",
      width: 1024,
      height: 1024,
      steps: 30,
      guidance: 7.5,
      seed: 42,
      batch_size: 2,
      output_format: "png",
      scheduler: "ddim",
      strength: 0.8,
      source_image: "AAAA",
      expand: true,
    });
  });

  it("toRequest omits expand entirely when disabled (server treats missing/false the same, but this keeps payload minimal)", () => {
    const form = useGenerateForm();
    form.state.value.expand.enabled = false;
    const wire = form.toRequest();
    expect(wire.expand).toBeUndefined();
  });

  it("toRequest maps empty negativePrompt to null so server skips CFG", () => {
    const form = useGenerateForm();
    form.state.value.negativePrompt = "";
    expect(form.toRequest().negative_prompt).toBeNull();
  });

  it("family-capability helpers match the documented allow-lists", () => {
    const form = useGenerateForm();
    // Video families.
    expect(form.isVideoFamily("ltx-video")).toBe(true);
    expect(form.isVideoFamily("ltx2")).toBe(true);
    expect(form.isVideoFamily("flux")).toBe(false);

    // CFG (negative prompt) support — flow-matching families skip CFG.
    expect(form.supportsNegativePrompt("sdxl")).toBe(true);
    expect(form.supportsNegativePrompt("sd15")).toBe(true);
    expect(form.supportsNegativePrompt("flux")).toBe(false);
    expect(form.supportsNegativePrompt("z-image")).toBe(false);

    // Scheduler override — UNet families only.
    expect(form.supportsScheduler("sdxl")).toBe(true);
    expect(form.supportsScheduler("sd15")).toBe(true);
    expect(form.supportsScheduler("flux")).toBe(false);
  });

  it("reset() restores defaults", () => {
    const form = useGenerateForm();
    form.state.value.prompt = "dirty";
    form.state.value.steps = 99;
    form.reset();
    expect(form.state.value.prompt).toBe("");
    expect(form.state.value.steps).toBe(20);
  });
});
