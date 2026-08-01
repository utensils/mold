import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GenerateFormState } from "../types";
import {
  deleteGenerationTemplate,
  deleteGenerationTemplateWithMedia,
  hydrateGenerationTemplate,
  loadGenerationTemplates,
  renameGenerationTemplate,
  saveGenerationTemplate,
  saveGenerationTemplateWithMedia,
  searchGenerationTemplates,
} from "./generationTemplates";
import type {
  StoredTemplateMedia,
  TemplateMediaPersistence,
} from "@studio/lib/templateMediaStore";

function makeForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  return {
    version: 3,
    stylePreset: null,
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
    ...overrides,
  };
}

function memoryMedia(): TemplateMediaPersistence & {
  records: Map<string, StoredTemplateMedia>;
} {
  const records = new Map<string, StoredTemplateMedia>();
  return {
    records,
    async put(record) {
      records.set(record.assetId, structuredClone(record));
      return true;
    },
    async get(assetId) {
      return records.get(assetId) ?? null;
    },
    async delete(assetId) {
      records.delete(assetId);
    },
  };
}

describe("generation templates", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.setSystemTime(new Date("2026-05-28T12:00:00Z"));
  });

  it("restores ordered local and gallery sources from durable template media", async () => {
    const persistence = memoryMedia();
    const saved = await saveGenerationTemplateWithMedia(
      "Sources",
      makeForm({
        imageAttachments: [
          { kind: "upload", filename: "local.png", base64: "LOCAL" },
          { kind: "gallery", filename: "remote.jpg", base64: "REMOTE" },
        ],
      }),
      persistence,
    );

    expect(saved.form.imageAttachments.every((image) => !image.base64)).toBe(
      true,
    );
    expect(JSON.stringify(saved)).not.toContain('"LOCAL"');
    const hydrated = await hydrateGenerationTemplate(saved, persistence);
    expect(hydrated.sourceMissing).toBe(false);
    expect(hydrated.form.imageAttachments).toEqual([
      { kind: "upload", filename: "local.png", base64: "LOCAL" },
      { kind: "gallery", filename: "remote.jpg", base64: "REMOTE" },
    ]);

    await deleteGenerationTemplateWithMedia(saved, persistence);
    expect(persistence.records.size).toBe(0);
  });

  it("does not expose byte-free legacy source markers as usable images", async () => {
    const legacy = saveGenerationTemplate(
      "Legacy",
      makeForm({
        imageAttachments: [
          { kind: "gallery", filename: "gone.png", base64: "GONE" },
        ],
      }),
    );
    const hydrated = await hydrateGenerationTemplate(legacy, memoryMedia());
    expect(hydrated.form.imageAttachments).toEqual([]);
    expect(hydrated.sourceMissing).toBe(true);
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
    expect(saved.form.imageAttachments).toEqual([
      { kind: "gallery", filename: "source.png" },
    ]);
    expect(saved.form.maskImage).toEqual({
      kind: "upload",
      filename: "mask.png",
    });
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
