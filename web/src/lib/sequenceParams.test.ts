import { describe, expect, it } from "vitest";
import { sequenceSharedParams } from "./sequenceParams";
import type { GenerateFormState } from "../types";

function formState(): GenerateFormState {
  return {
    version: 3,
    stylePreset: null,
    prompt: "a heron lifts off",
    negativePrompt: "",
    model: "ltx-2-19b-distilled:fp8",
    modelFamily: "ltx2",
    width: 1216,
    height: 704,
    steps: 8,
    guidance: 3.5,
    sourceImageCapability: null,
    endFrame: null,
    seedMode: "random",
    seed: null,
    batchSize: 1,
    strength: 0.9,
    frames: 97,
    fps: 24,
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
    sourceFitPolicy: { mode: "crop-fill" },
    gifPreview: false,
    audioFile: null,
    audioFilePath: "",
    sourceVideo: null,
    sourceVideoPath: "",
    extendVideo: null,
    extendVideoPath: "",
    extendOverlapFrames: null,
    keyframes: [],
    pipeline: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    cameraControl: null,
    placement: null,
    loras: [],
    enableAudio: null,
  };
}

describe("sequenceSharedParams", () => {
  it("projects the LIVE generate form onto the shared chain params", () => {
    const state = formState();
    const shared = sequenceSharedParams(state, "ltx2");
    expect(shared).toEqual({
      model: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      width: 1216,
      height: 704,
      fps: 24,
      steps: 8,
      guidance: 3.5,
      strength: 0.9,
      sourceFitPolicy: { mode: "crop-fill" },
      upscalerModel: "",
      seed: "",
    });
  });

  it("keeps a fixed seed and maps random mode to the empty string", () => {
    const state = formState();
    state.seedMode = "static";
    state.seed = 42;
    expect(sequenceSharedParams(state, "ltx2").seed).toBe("42");

    state.seedMode = "random";
    expect(sequenceSharedParams(state, "ltx2").seed).toBe("");
  });

  it("defaults a null fps to 24", () => {
    const state = formState();
    state.fps = null;
    expect(sequenceSharedParams(state, "ltx2").fps).toBe(24);
  });
});
