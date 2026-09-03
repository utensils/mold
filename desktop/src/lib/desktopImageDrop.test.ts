import { describe, expect, it } from "vitest";
import { newGenerateForm } from "./generateForm";
import { applyDesktopImageDrop, type DesktopImageImport } from "./desktopImageDrop";
import type { ModelEntry, OutputMetadata } from "./api/types";
import { flux2KleinRecipe } from "@studio/lib/generationProfile.testFixtures";

const sd15 = {
  name: "sd15:fp16",
  family: "sd15",
  downloaded: true,
  default_width: 512,
  default_height: 512,
  default_steps: 20,
  default_guidance: 7,
} as ModelEntry;

const qwenEdit = {
  name: "qwen-image-edit:q4",
  family: "qwen-image-edit",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 20,
  default_guidance: 4,
} as ModelEntry;

function dropped(
  metadata: OutputMetadata | null = null,
  filename = "reference.png",
): DesktopImageImport {
  return {
    filename,
    base64: filename === "reference.png" ? "IMAGE_BYTES" : `BYTES_${filename}`,
    width: 896,
    height: 1152,
    metadata,
  };
}

function metadata(overrides: Partial<OutputMetadata> = {}): OutputMetadata {
  return {
    prompt: "a brass observatory",
    model: "sd15:fp16",
    seed: 42,
    steps: 31,
    guidance: 6.5,
    width: 768,
    height: 512,
    version: "test",
    ...overrides,
  };
}

/**
 * Klein's two-well layout comes from the ADVERTISED recipe on the selected
 * row — the interim legacy sniff deliberately answers "no references" for
 * Klein, because an older host has no Klein reference engine.
 */
const kleinModel = {
  name: "flux2-klein:bf16",
  family: "flux2",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 4,
  default_guidance: 1,
  generation_profile: {
    schema_version: 1,
    profile_id: "flux2-klein",
    profile_hash: "test",
    default_recipe_id: "default",
    recipes: [flux2KleinRecipe()],
  },
} as unknown as ModelEntry;

function kleinForm() {
  const form = newGenerateForm();
  Object.assign(form, { model: kleinModel.name, family: "flux2" });
  return form;
}

describe("applyDesktopImageDrop", () => {
  it("attaches a plain dropped image as the current model's source", async () => {
    const form = newGenerateForm();
    Object.assign(form, { model: sd15.name, family: sd15.family });

    await expect(applyDesktopImageDrop(form, dropped(), [sd15])).resolves.toMatchObject({
      attached: true,
      metadataApplied: false,
      target: "source",
    });
    expect(form.sourceImage).toBe("IMAGE_BYTES");
    expect(form.sourceImageName).toBe("reference.png");
    expect(form.sourceFit).toEqual({ mode: "crop-fill" });
    expect(form.sourceImageWidth).toBe(896);
    expect(form.sourceImageHeight).toBe(1152);
  });

  it("loads embedded generation metadata and keeps the dropped image as source", async () => {
    const form = newGenerateForm();

    await expect(applyDesktopImageDrop(form, dropped(metadata()), [sd15])).resolves.toMatchObject({
      attached: true,
      metadataApplied: true,
    });
    expect(form.prompt).toBe("a brass observatory");
    expect(form.model).toBe("sd15:fp16");
    expect(form.seed).toBe("42");
    expect(form.sourceImage).toBe("IMAGE_BYTES");
    expect(form.sourceImageName).toBe("reference.png");
  });

  it("uses a dropped image as the Qwen edit target", async () => {
    const form = newGenerateForm();
    Object.assign(form, { model: qwenEdit.name, family: qwenEdit.family });

    const result = await applyDesktopImageDrop(form, dropped(), [qwenEdit]);
    expect(result.attached).toBe(true);
    expect(form.imageAttachments).toEqual(["IMAGE_BYTES"]);
    expect(form.sourceImage).toBeNull();
    expect(form.sourceFit).toEqual({ mode: "crop-fill" });
  });

  it("APPENDS to a strip that already holds pictures", async () => {
    // The bug: routing by capability replaced the whole strip, so dropping a
    // second reference onto a Target + reference lost both.
    const form = newGenerateForm();
    Object.assign(form, { model: qwenEdit.name, family: qwenEdit.family });
    form.imageAttachments = ["TARGET", "REF_1"];

    await applyDesktopImageDrop(form, dropped(), [qwenEdit]);
    expect(form.imageAttachments).toEqual(["TARGET", "REF_1", "IMAGE_BYTES"]);
  });

  it("still loads metadata when that model cannot accept a source image", async () => {
    const form = newGenerateForm();
    const textOnly = { ...sd15, name: "ltx-video:bf16", family: "ltx-video" } as ModelEntry;

    const result = await applyDesktopImageDrop(form, dropped(metadata({ model: textOnly.name })), [
      textOnly,
    ]);
    expect(result).toMatchObject({ attached: false, metadataApplied: true });
    expect(result.refused).toBeTruthy();
    expect(form.prompt).toBe("a brass observatory");
    expect(form.sourceImage).toBeNull();
  });
});

describe("applyDesktopImageDrop routes to the well under the cursor", () => {
  it("reaches the end-frame well of a first/last checkpoint", async () => {
    const form = newGenerateForm();
    Object.assign(form, {
      model: "wan22-ti2v-5b:fp16",
      family: "wan",
      sourceImageCapability: "optional",
    });

    await applyDesktopImageDrop(form, dropped(), [], "end");
    expect(form.endFrame).toEqual({ filename: "reference.png", base64: "IMAGE_BYTES" });
    expect(form.sourceImage).toBeNull();
  });

  it("reaches the identity well only while it is rendering", async () => {
    const form = newGenerateForm();
    Object.assign(form, { model: sd15.name, family: sd15.family });

    await applyDesktopImageDrop(form, dropped(), [sd15], "identity", {
      identityVisible: true,
    });
    expect(form.identityImage).toEqual({
      filename: "reference.png",
      base64: "IMAGE_BYTES",
    });

    // Not rendering: the drop falls back to the plan default rather than
    // writing a field nothing can show.
    const other = newGenerateForm();
    Object.assign(other, { model: sd15.name, family: sd15.family });
    await applyDesktopImageDrop(other, dropped(), [sd15], "identity");
    expect(other.identityImage).toBeNull();
    expect(other.sourceImage).toBe("IMAGE_BYTES");
  });

  it("reaches the sequence opening image on the draft that owns it", async () => {
    const form = newGenerateForm();
    Object.assign(form, { model: sd15.name, family: sd15.family });
    const draft = { openingImage: null as { filename: string; base64: string } | null };

    await applyDesktopImageDrop(form, dropped(), [sd15], "opening", {
      openingVisible: true,
      sequenceDraft: draft,
    });
    expect(draft.openingImage).toEqual({
      filename: "reference.png",
      base64: "IMAGE_BYTES",
    });
    expect(form.sourceImage).toBeNull();
  });

  it("writes H3 boundaries and references to h3Authoring, never imageAttachments", async () => {
    // The old bridge wrote `imageAttachments` for every non-single mode — a
    // field the H3 request builder never reads, so the drop did nothing.
    const first = newGenerateForm();
    Object.assign(first, {
      model: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
    });
    await applyDesktopImageDrop(first, dropped(), []);
    expect(first.imageAttachments).toEqual([]);
    expect(first.h3Authoring?.firstFrame?.data).toBe("IMAGE_BYTES");

    // A second drop with the first frame filled lands on the last frame.
    await applyDesktopImageDrop(first, dropped(null, "closing.png"), []);
    expect(first.h3Authoring?.lastFrame?.data).toBe("BYTES_closing.png");

    const ref2va = newGenerateForm();
    Object.assign(ref2va, {
      model: "minimax-h3-ref2va:official-bf16",
      family: "minimax-h3",
    });
    // The ordered-reference panel hashes its media, so this leg carries real
    // padded base64 rather than the placeholder the other cases use.
    const result = await applyDesktopImageDrop(ref2va, { ...dropped(), base64: "QUJD" }, []);
    expect(result).toMatchObject({ attached: true, target: "h3-reference" });
    expect(ref2va.imageAttachments).toEqual([]);
    expect(ref2va.h3Authoring?.references).toHaveLength(1);
  });
});

describe("applyDesktopImageDrop on an exclusive (Klein) recipe", () => {
  it("defaults to the source well on an empty Klein form", async () => {
    const form = kleinForm();
    const result = await applyDesktopImageDrop(form, dropped(), [kleinModel]);
    expect(result.target).toBe("source");
    expect(form.exclusiveWell).toBe("source");
  });

  it("parks the source when a reference is dropped on the strip", async () => {
    const form = kleinForm();
    form.sourceImage = "SOURCE_BYTES";
    form.sourceImageName = "source.png";

    await applyDesktopImageDrop(form, dropped(), [kleinModel], "references");
    // The source is PARKED, not discarded — it comes back when the
    // references clear.
    expect(form.sourceImage).toBe("SOURCE_BYTES");
    expect(form.imageAttachments).toEqual(["IMAGE_BYTES"]);
    expect(form.exclusiveWell).toBe("references");
  });

  it("parks the references when a source is dropped on the source well", async () => {
    const form = kleinForm();
    form.imageAttachments = ["REF_1"];

    await applyDesktopImageDrop(form, dropped(), [kleinModel], "source");
    expect(form.imageAttachments).toEqual(["REF_1"]);
    expect(form.sourceImage).toBe("IMAGE_BYTES");
    expect(form.exclusiveWell).toBe("source");
  });
});
