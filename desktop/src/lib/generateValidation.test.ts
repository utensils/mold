import { describe, expect, it } from "vitest";
import { newGenerateForm } from "./generateForm";
import { inlineGenerationMediaBytes } from "./generateValidation";

function formWithEveryH3MediaField() {
  const form = newGenerateForm();
  form.h3Authoring = {
    firstFrame: {
      filename: "first.png",
      mimeType: "image/png",
      width: 32,
      height: 32,
      data: "QUJD",
    },
    lastFrame: {
      filename: "last.png",
      mimeType: "image/png",
      width: 32,
      height: 32,
      data: "REVG",
    },
    references: [
      {
        reference: {
          kind: "image",
          media: { authority: "inline", data: "R0hJ" },
          provenance: { name: "reference.png" },
          mime_type: "image/png",
          width: 32,
          height: 32,
        },
      },
    ],
  };
  return form;
}

describe("inlineGenerationMediaBytes — MiniMax H3 active partition", () => {
  it("counts only FL2VA boundaries and preserves replacement exclusion", () => {
    const form = formWithEveryH3MediaField();
    form.model = "minimax-h3-fl2va:official-bf16";

    expect(inlineGenerationMediaBytes(form)).toBe(6);
    expect(inlineGenerationMediaBytes(form, "h3FirstFrame")).toBe(3);
    expect(inlineGenerationMediaBytes(form, "h3References")).toBe(6);
  });

  it("counts only ordered Ref2VA media and preserves replacement exclusion", () => {
    const form = formWithEveryH3MediaField();
    form.model = "minimax-h3-ref2va:comfy-pruned-int8";

    expect(inlineGenerationMediaBytes(form)).toBe(3);
    expect(inlineGenerationMediaBytes(form, "h3References")).toBe(0);
    expect(inlineGenerationMediaBytes(form, "h3FirstFrame")).toBe(3);
  });

  it.each(["flux:replacement", "minimax-h3-ref2va:future-layout"])(
    "ignores parked H3 media when %s has no released H3 wire task",
    (model) => {
      const form = formWithEveryH3MediaField();
      form.model = model;

      expect(inlineGenerationMediaBytes(form)).toBe(0);
    },
  );
});
