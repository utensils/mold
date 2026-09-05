import { beforeEach, describe, expect, it } from "vitest";
import { newGenerateForm, type GenerateForm } from "./generateForm";
import { installMemoryLocalStorage } from "./testSupport/memoryLocalStorage";

// happy-dom v20 does not expose `localStorage`; the real Tauri webview does.
installMemoryLocalStorage();
import {
  GENERATION_TEMPLATES_STORAGE_KEY,
  collectTemplateMediaReferences,
  deleteGenerationTemplate,
  formatTemplateMediaReferences,
  loadGenerationTemplates,
  hydrateGenerationTemplate,
  renameGenerationTemplate,
  saveGenerationTemplate,
  saveGenerationTemplateWithMedia,
  deleteGenerationTemplateWithMedia,
  searchGenerationTemplates,
  sortGenerationTemplates,
  type GenerationTemplate,
} from "./generationTemplates";
import type { StoredTemplateMedia, TemplateMediaPersistence } from "@studio/lib/templateMediaStore";

const B64 = "aGVsbG8td29ybGQtYmFzZTY0"; // "hello-world-base64"

function formWith(overrides: Partial<GenerateForm> = {}): GenerateForm {
  return { ...newGenerateForm(), ...overrides };
}

function memoryMedia(): TemplateMediaPersistence & { records: Map<string, StoredTemplateMedia> } {
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

beforeEach(() => {
  localStorage.clear();
});

describe("save / load round-trip", () => {
  it("restores a saved source image independently of its original gallery host", async () => {
    const persistence = memoryMedia();
    const saved = await saveGenerationTemplateWithMedia(
      "Portable source",
      formWith({
        sourceImage: B64,
        sourceImageName: "remote-gallery.png",
      }),
      GENERATION_TEMPLATES_STORAGE_KEY,
      "host-a",
      persistence,
    );

    expect(saved.form.sourceImage).toBeNull();
    expect(JSON.stringify(saved)).not.toContain(B64);
    const hydrated = await hydrateGenerationTemplate(saved, persistence);
    expect(hydrated.form.sourceImage).toBe(B64);
    expect(hydrated.form.sourceImageName).toBe("remote-gallery.png");
    expect(hydrated.missingMediaReferences).toEqual([]);

    await deleteGenerationTemplateWithMedia(saved, GENERATION_TEMPLATES_STORAGE_KEY, persistence);
    expect(persistence.records.size).toBe(0);
  });

  it("restores H3 endpoint and mixed-reference bytes in authoritative order", async () => {
    const persistence = memoryMedia();
    const saved = await saveGenerationTemplateWithMedia(
      "H3 portable",
      formWith({
        model: "minimax-h3-ref2va:official-bf16",
        family: "minimax-h3",
        h3Authoring: {
          firstFrame: null,
          lastFrame: {
            filename: "last.png",
            mimeType: "image/png",
            width: 1280,
            height: 720,
            data: "H3_LAST_BYTES",
          },
          references: [
            {
              reference: {
                kind: "audio",
                media: { authority: "inline", data: "H3_AUDIO_BYTES" },
                provenance: { name: "voice.wav", sha256: "a".repeat(64) },
                mime_type: "audio/wav",
                duration_ms: 2_500,
                sample_rate: 48_000,
                channels: 2,
                sample_count: 120_000,
              },
            },
            {
              reference: {
                kind: "image",
                media: { authority: "inline", data: "H3_IMAGE_BYTES" },
                provenance: { name: "subject.png", sha256: "b".repeat(64) },
                mime_type: "image/png",
                width: 1024,
                height: 768,
              },
            },
          ],
        },
      }),
      GENERATION_TEMPLATES_STORAGE_KEY,
      "host-a",
      persistence,
    );

    expect(JSON.stringify(saved)).not.toContain("H3_LAST_BYTES");
    expect(saved.mediaReferences).toEqual(["h3LastFrame", "h3References"]);
    const hydrated = await hydrateGenerationTemplate(saved, persistence);
    expect(hydrated.missingMediaReferences).toEqual([]);
    expect(hydrated.form.h3Authoring?.lastFrame?.data).toBe("H3_LAST_BYTES");
    expect(
      hydrated.form.h3Authoring?.references.map((draft) => [
        draft.reference.kind,
        draft.reference.provenance?.name,
        draft.reference.media.authority === "inline" ? draft.reference.media.data : null,
      ]),
    ).toEqual([
      ["audio", "voice.wav", "H3_AUDIO_BYTES"],
      ["image", "subject.png", "H3_IMAGE_BYTES"],
    ]);
  });

  it("keeps the legacy re-add warning when no durable source asset exists", async () => {
    const legacy = saveGenerationTemplate("Legacy", formWith({ sourceImage: B64 }));
    const hydrated = await hydrateGenerationTemplate(legacy, memoryMedia());
    expect(hydrated.form.sourceImage).toBeNull();
    expect(hydrated.missingMediaReferences).toEqual(["source"]);
  });

  it("strips base64 media and records references", () => {
    const form = formWith({
      prompt: "a lighthouse at dusk",
      model: "flux-dev:q8",
      sourceImage: B64,
      maskImage: B64,
      controlImage: B64,
    });

    const saved = saveGenerationTemplate("Dusk shot", form);

    expect(saved.form.sourceImage).toBeNull();
    expect(saved.form.maskImage).toBeNull();
    expect(saved.form.controlImage).toBeNull();
    expect(saved.mediaReferences).toEqual(["source", "mask", "control"]);

    // The base64 blob must never touch localStorage.
    const raw = localStorage.getItem(GENERATION_TEMPLATES_STORAGE_KEY) ?? "";
    expect(raw).not.toContain(B64);

    const [reloaded] = loadGenerationTemplates();
    expect(reloaded?.form.prompt).toBe("a lighthouse at dusk");
    expect(reloaded?.form.sourceImage).toBeNull();
    expect(reloaded?.mediaReferences).toEqual(["source", "mask", "control"]);
  });

  it("strips the qwen-edit picture strip and records an editImages reference", () => {
    const form = formWith({
      prompt: "make the sky pink",
      model: "qwen-image-edit-2511:q4",
      imageAttachments: [B64, B64],
    });

    const saved = saveGenerationTemplate("Sky edit", form);

    expect(saved.form.imageAttachments).toEqual([]);
    expect(saved.mediaReferences).toEqual(["editImages"]);
    const raw = localStorage.getItem(GENERATION_TEMPLATES_STORAGE_KEY) ?? "";
    expect(raw).not.toContain(B64);
  });

  it("preserves non-media fields (prompt, model, params, loras)", () => {
    const form = formWith({
      prompt: "neon city",
      negativePrompt: "blurry",
      model: "sdxl:fp16",
      family: "sdxl",
      steps: 30,
      guidance: 7,
      seed: "42",
      loras: [{ path: "/loras/foo.safetensors", name: "Foo", scale: 0.8, trainedWords: ["foo"] }],
    });

    saveGenerationTemplate("City", form);
    const [tpl] = loadGenerationTemplates();

    expect(tpl?.form.prompt).toBe("neon city");
    expect(tpl?.form.negativePrompt).toBe("blurry");
    expect(tpl?.form.steps).toBe(30);
    expect(tpl?.form.seed).toBe("42");
    expect(tpl?.form.loras[0]?.path).toBe("/loras/foo.safetensors");
    expect(tpl?.mediaReferences).toEqual([]);
  });

  it("hydrates an empty guidance override state for legacy templates", () => {
    const legacyForm: Partial<GenerateForm> = newGenerateForm();
    delete legacyForm.guidanceOverrides;
    localStorage.setItem(
      GENERATION_TEMPLATES_STORAGE_KEY,
      JSON.stringify([
        {
          id: "legacy",
          name: "Legacy",
          createdAt: 1,
          updatedAt: 1,
          form: legacyForm,
          mediaReferences: [],
        },
      ]),
    );

    expect(loadGenerationTemplates()[0]?.form.guidanceOverrides).toEqual({
      stgScale: null,
      stgBlocks: "",
      rescaleScale: null,
      modalityScale: null,
      skipStep: null,
    });
  });

  /**
   * A template saved before "Save every result" existed carries no
   * preference. Hydrating it `undefined` let `buildRequest`'s falsy check
   * read it as an opt-out and send every print from a legacy starting point
   * straight to the trash.
   */
  it("hydrates Save every result on for legacy templates", () => {
    const legacyForm: Partial<GenerateForm> = newGenerateForm();
    delete legacyForm.saveResult;
    localStorage.setItem(
      GENERATION_TEMPLATES_STORAGE_KEY,
      JSON.stringify([
        {
          id: "legacy",
          name: "Legacy",
          createdAt: 1,
          updatedAt: 1,
          form: legacyForm,
          mediaReferences: [],
        },
      ]),
    );

    expect(loadGenerationTemplates()[0]?.form.saveResult).toBe(true);
  });

  it("keeps an explicit opt-out saved in a template", () => {
    const form = newGenerateForm();
    form.saveResult = false;
    localStorage.setItem(
      GENERATION_TEMPLATES_STORAGE_KEY,
      JSON.stringify([
        {
          id: "opted-out",
          name: "Throwaways",
          createdAt: 1,
          updatedAt: 1,
          form,
          mediaReferences: [],
        },
      ]),
    );

    expect(loadGenerationTemplates()[0]?.form.saveResult).toBe(false);
  });

  it("strips video, keyframe, and conditioning-audio media", () => {
    const form = formWith({
      prompt: "a river",
    });
    form.sourceVideo = { filename: "clip.mp4", base64: B64 };
    form.keyframes = [{ frame: 8, image: { filename: "kf.png", base64: B64 } }];
    form.audioFile = { filename: "score.wav", base64: B64 };

    const saved = saveGenerationTemplate("River", form);

    expect(saved.mediaReferences).toContain("sourceVideo");
    expect(saved.mediaReferences).toContain("keyframes");
    expect(saved.mediaReferences).toContain("audioFile");
    expect(saved.form.audioFile).toBeNull();
    const raw = localStorage.getItem(GENERATION_TEMPLATES_STORAGE_KEY) ?? "";
    expect(raw).not.toContain(B64);
  });

  it("gives a template a fallback name from the prompt when name is blank", () => {
    const saved = saveGenerationTemplate("", formWith({ prompt: "  a quiet forest  " }));
    expect(saved.name).toBe("a quiet forest");

    const untitled = saveGenerationTemplate("", formWith({ prompt: "" }));
    expect(untitled.name).toBe("Untitled template");
  });

  it("can scope a remote template to its owning host", () => {
    const saved = saveGenerationTemplate(
      "Remote",
      formWith({ prompt: "remote" }),
      "mold.mobile.templates.test",
      "host-a",
    );
    expect(saved.scopeId).toBe("host-a");
  });
});

describe("sort", () => {
  const now = Date.now();
  const templates: GenerationTemplate[] = [
    {
      id: "a",
      name: "Banana",
      createdAt: now,
      updatedAt: now + 100,
      form: newGenerateForm(),
      mediaReferences: [],
    },
    {
      id: "b",
      name: "apple",
      createdAt: now,
      updatedAt: now + 300,
      form: newGenerateForm(),
      mediaReferences: [],
    },
    {
      id: "c",
      name: "Cherry",
      createdAt: now,
      updatedAt: now + 200,
      form: newGenerateForm(),
      mediaReferences: [],
    },
  ];

  it("sorts newest-first by default", () => {
    expect(sortGenerationTemplates(templates).map((t) => t.id)).toEqual(["b", "c", "a"]);
  });
  it("sorts oldest-first", () => {
    expect(sortGenerationTemplates(templates, "updated-asc").map((t) => t.id)).toEqual([
      "a",
      "c",
      "b",
    ]);
  });
  it("sorts A-Z case-insensitively", () => {
    expect(sortGenerationTemplates(templates, "name-asc").map((t) => t.name)).toEqual([
      "apple",
      "Banana",
      "Cherry",
    ]);
  });
  it("sorts Z-A case-insensitively", () => {
    expect(sortGenerationTemplates(templates, "name-desc").map((t) => t.name)).toEqual([
      "Cherry",
      "Banana",
      "apple",
    ]);
  });
  it("does not mutate the input array", () => {
    const before = templates.map((t) => t.id);
    sortGenerationTemplates(templates, "name-asc");
    expect(templates.map((t) => t.id)).toEqual(before);
  });
});

describe("search", () => {
  beforeEach(() => {
    saveGenerationTemplate(
      "Portrait",
      formWith({ prompt: "a woman by a window", model: "flux-dev:q8" }),
    );
    saveGenerationTemplate("Landscape", formWith({ prompt: "rolling hills", model: "sdxl:fp16" }));
  });

  it("returns all templates for a blank query", () => {
    expect(searchGenerationTemplates("").length).toBe(2);
  });
  it("matches on name", () => {
    expect(searchGenerationTemplates("portra").map((t) => t.name)).toEqual(["Portrait"]);
  });
  it("matches on prompt text", () => {
    expect(searchGenerationTemplates("hills").map((t) => t.name)).toEqual(["Landscape"]);
  });
  it("matches on model name", () => {
    expect(searchGenerationTemplates("sdxl").map((t) => t.name)).toEqual(["Landscape"]);
  });
  it("returns nothing when there is no match", () => {
    expect(searchGenerationTemplates("zzznope")).toEqual([]);
  });
});

describe("corrupted / malformed storage", () => {
  it("returns [] for non-JSON", () => {
    localStorage.setItem(GENERATION_TEMPLATES_STORAGE_KEY, "{not json");
    expect(loadGenerationTemplates()).toEqual([]);
  });
  it("returns [] when the stored value is not an array", () => {
    localStorage.setItem(GENERATION_TEMPLATES_STORAGE_KEY, JSON.stringify({ id: "x" }));
    expect(loadGenerationTemplates()).toEqual([]);
  });
  it("drops entries missing required fields", () => {
    localStorage.setItem(
      GENERATION_TEMPLATES_STORAGE_KEY,
      JSON.stringify([
        { id: "ok", name: "n", createdAt: 1, updatedAt: 1, form: {}, mediaReferences: [] },
        { id: "bad" },
      ]),
    );
    const loaded = loadGenerationTemplates();
    expect(loaded.length).toBe(1);
    expect(loaded[0]?.id).toBe("ok");
  });
  it("returns [] when nothing is stored", () => {
    expect(loadGenerationTemplates()).toEqual([]);
  });
});

describe("rename", () => {
  it("renames and touches updatedAt", () => {
    const saved = saveGenerationTemplate("Old", formWith({ prompt: "x" }));
    const renamed = renameGenerationTemplate(saved.id, "New");
    expect(renamed?.name).toBe("New");
    expect(loadGenerationTemplates()[0]?.name).toBe("New");
  });
  it("falls back to Untitled for a blank name", () => {
    const saved = saveGenerationTemplate("Old", formWith({ prompt: "x" }));
    expect(renameGenerationTemplate(saved.id, "   ")?.name).toBe("Untitled template");
  });
  it("returns null for an unknown id", () => {
    expect(renameGenerationTemplate("nope", "New")).toBeNull();
  });
});

describe("delete", () => {
  it("removes a template by id", () => {
    const a = saveGenerationTemplate("A", formWith({ prompt: "x" }));
    saveGenerationTemplate("B", formWith({ prompt: "y" }));
    deleteGenerationTemplate(a.id);
    const names = loadGenerationTemplates().map((t) => t.name);
    expect(names).toEqual(["B"]);
  });
  it("is a no-op for an unknown id", () => {
    saveGenerationTemplate("A", formWith({ prompt: "x" }));
    deleteGenerationTemplate("nope");
    expect(loadGenerationTemplates().length).toBe(1);
  });
});

describe("storage namespaces", () => {
  it("can isolate mobile templates from desktop templates", () => {
    const mobileKey = "mold.mobile.generation.templates.v1";

    saveGenerationTemplate("Desktop", formWith({ prompt: "desktop prompt" }));
    saveGenerationTemplate("Mobile", formWith({ prompt: "mobile prompt" }), mobileKey);

    expect(loadGenerationTemplates().map((template) => template.name)).toEqual(["Desktop"]);
    expect(
      loadGenerationTemplates("updated-desc", mobileKey).map((template) => template.name),
    ).toEqual(["Mobile"]);
  });
});

describe("collectTemplateMediaReferences", () => {
  it("lists only the media fields that are populated", () => {
    expect(collectTemplateMediaReferences(formWith({ sourceImage: B64 }))).toEqual(["source"]);
    expect(collectTemplateMediaReferences(formWith())).toEqual([]);
  });

  it("formats internal media keys as readable instructions", () => {
    expect(formatTemplateMediaReferences(["editImages", "audioFile"])).toBe(
      "edit pictures, conditioning audio",
    );
  });
});
