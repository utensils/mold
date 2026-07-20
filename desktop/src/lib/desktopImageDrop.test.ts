import { describe, expect, it } from "vitest";
import { newGenerateForm } from "./generateForm";
import { applyDesktopImageDrop, type DesktopImageImport } from "./desktopImageDrop";
import type { ModelEntry, OutputMetadata } from "./api/types";

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

function dropped(metadata: OutputMetadata | null = null): DesktopImageImport {
  return { filename: "reference.png", base64: "IMAGE_BYTES", metadata };
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

describe("applyDesktopImageDrop", () => {
  it("attaches a plain dropped image as the current model's source", () => {
    const form = newGenerateForm();
    Object.assign(form, { model: sd15.name, family: sd15.family });

    expect(applyDesktopImageDrop(form, dropped(), [sd15])).toEqual({
      attached: true,
      metadataApplied: false,
    });
    expect(form.sourceImage).toBe("IMAGE_BYTES");
    expect(form.sourceImageName).toBe("reference.png");
  });

  it("loads embedded generation metadata and keeps the dropped image as source", () => {
    const form = newGenerateForm();

    expect(applyDesktopImageDrop(form, dropped(metadata()), [sd15])).toEqual({
      attached: true,
      metadataApplied: true,
    });
    expect(form.prompt).toBe("a brass observatory");
    expect(form.model).toBe("sd15:fp16");
    expect(form.seed).toBe("42");
    expect(form.steps).toBe(31);
    expect(form.guidance).toBe(6.5);
    expect(form.width).toBe(768);
    expect(form.height).toBe(512);
    expect(form.sourceImage).toBe("IMAGE_BYTES");
    expect(form.sourceImageName).toBe("reference.png");
  });

  it("uses a dropped image as the Qwen edit target", () => {
    const form = newGenerateForm();
    Object.assign(form, { model: qwenEdit.name, family: qwenEdit.family });

    expect(applyDesktopImageDrop(form, dropped(), [qwenEdit]).attached).toBe(true);
    expect(form.imageAttachments).toEqual(["IMAGE_BYTES"]);
    expect(form.sourceImage).toBeNull();
  });

  it("still loads metadata when that model cannot accept a source image", () => {
    const form = newGenerateForm();
    const textOnly = { ...sd15, name: "ltx-video:bf16", family: "ltx-video" } as ModelEntry;

    expect(
      applyDesktopImageDrop(form, dropped(metadata({ model: textOnly.name })), [textOnly]),
    ).toEqual({ attached: false, metadataApplied: true });
    expect(form.prompt).toBe("a brass observatory");
    expect(form.sourceImage).toBeNull();
  });
});
