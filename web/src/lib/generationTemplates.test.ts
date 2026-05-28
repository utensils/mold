import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GenerateFormState } from "../types";
import {
  deleteGenerationTemplate,
  loadGenerationTemplates,
  renameGenerationTemplate,
  saveGenerationTemplate,
  searchGenerationTemplates,
} from "./generationTemplates";

function makeForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  return {
    version: 2,
    prompt: "cinematic cat",
    negativePrompt: "blurry",
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
    placement: null,
    loras: [],
    enableAudio: null,
    ...overrides,
  };
}

describe("generation templates", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.setSystemTime(new Date("2026-05-28T12:00:00Z"));
  });

  it("saves named templates with sanitized form config and media references", () => {
    const saved = saveGenerationTemplate(
      "Portrait Base",
      makeForm({
        imageAttachments: [
          { kind: "gallery", filename: "source.png", base64: "SOURCE_BYTES" },
        ],
        maskImage: {
          kind: "upload",
          filename: "mask.png",
          base64: "MASK_BYTES",
        },
        audioFilePath: "/srv/audio.wav",
        loras: [
          {
            path: "/loras/a.safetensors",
            scale: 0.7,
            trainedWords: ["style a"],
          },
        ],
      }),
    );

    expect(saved.name).toBe("Portrait Base");
    expect(saved.form.prompt).toBe("cinematic cat");
    expect(saved.form.imageAttachments).toEqual([]);
    expect(saved.form.maskImage).toBeNull();
    expect(saved.form.audioFilePath).toBe("/srv/audio.wav");
    expect(saved.form.loras).toEqual([
      { path: "/loras/a.safetensors", scale: 0.7, trainedWords: ["style a"] },
    ]);
    expect(saved.mediaReferences).toEqual(
      expect.arrayContaining([
        {
          field: "imageAttachments",
          kind: "gallery",
          filename: "source.png",
        },
        { field: "maskImage", kind: "upload", filename: "mask.png" },
      ]),
    );
    expect(localStorage.getItem("mold.generation.templates.v1")).not.toContain(
      "SOURCE_BYTES",
    );
  });

  it("loads, searches, renames, and deletes templates from localStorage", () => {
    const first = saveGenerationTemplate("Portrait Base", makeForm());
    const second = saveGenerationTemplate(
      "Video Motion",
      makeForm({ prompt: "moving clouds", outputFormat: "mp4" }),
    );

    expect(loadGenerationTemplates().map((t) => t.id)).toEqual([
      second.id,
      first.id,
    ]);
    expect(searchGenerationTemplates("portrait").map((t) => t.id)).toEqual([
      first.id,
    ]);

    const renamed = renameGenerationTemplate(first.id, "Portrait Final");
    expect(renamed?.name).toBe("Portrait Final");

    deleteGenerationTemplate(second.id);
    expect(loadGenerationTemplates().map((t) => t.name)).toEqual([
      "Portrait Final",
    ]);
  });
});
