import { describe, expect, it } from "vitest";

import {
  OPTIONAL_PROMPT_GUIDANCE,
  OPTIONAL_PROMPT_PLACEHOLDER,
  familyAllowsEmptyPrompt,
  hasVisualConditioning,
  promptOptional,
  promptPlaceholder,
  promptRequired,
} from "./promptRequirement";

describe("familyAllowsEmptyPrompt", () => {
  it("accepts the video families whose engines carry visual conditioning", () => {
    expect(familyAllowsEmptyPrompt("ltx2")).toBe(true);
    expect(familyAllowsEmptyPrompt("ltx-2")).toBe(true);
    expect(familyAllowsEmptyPrompt("ltx-video")).toBe(true);
  });

  it("normalizes case and surrounding whitespace like the server", () => {
    expect(familyAllowsEmptyPrompt("  LTX2 ")).toBe(true);
    expect(familyAllowsEmptyPrompt("LTX-Video")).toBe(true);
  });

  it("rejects image families and an unknown family", () => {
    expect(familyAllowsEmptyPrompt("flux")).toBe(false);
    expect(familyAllowsEmptyPrompt("qwen-image-edit")).toBe(false);
    expect(familyAllowsEmptyPrompt("sd35")).toBe(false);
    expect(familyAllowsEmptyPrompt("")).toBe(false);
    expect(familyAllowsEmptyPrompt(null)).toBe(false);
    expect(familyAllowsEmptyPrompt(undefined)).toBe(false);
  });
});

describe("hasVisualConditioning", () => {
  it("counts every conditioning channel the server accepts", () => {
    expect(
      hasVisualConditioning({ sourceImage: "data:image/png;base64," }),
    ).toBe(true);
    expect(hasVisualConditioning({ imageAttachments: [{ base64: "x" }] })).toBe(
      true,
    );
    expect(hasVisualConditioning({ keyframes: [{ frame: 0 }] })).toBe(true);
    expect(hasVisualConditioning({ sourceVideo: { base64: "x" } })).toBe(true);
    expect(hasVisualConditioning({ sourceVideoPath: "/clips/a.mp4" })).toBe(
      true,
    );
    expect(hasVisualConditioning({ extendVideo: { base64: "x" } })).toBe(true);
    expect(hasVisualConditioning({ extendVideoPath: "/clips/a.mp4" })).toBe(
      true,
    );
  });

  it("treats empty collections, blank paths, and absence as no conditioning", () => {
    expect(hasVisualConditioning({})).toBe(false);
    expect(
      hasVisualConditioning({
        sourceImage: null,
        imageAttachments: [],
        keyframes: [],
        sourceVideo: null,
        sourceVideoPath: "",
        extendVideo: null,
        extendVideoPath: "   ",
      }),
    ).toBe(false);
    expect(hasVisualConditioning(null)).toBe(false);
    expect(hasVisualConditioning(undefined)).toBe(false);
  });
});

describe("promptRequired / promptOptional", () => {
  // Parity with `validate_generate_request_with_family`: an empty prompt is
  // admitted only when a promptless-capable video family is paired with real
  // visual conditioning.
  it("allows an empty prompt for a conditioned LTX-2 render", () => {
    expect(promptOptional({ family: "ltx2", sourceImage: "b64" })).toBe(true);
    expect(promptOptional({ family: "ltx-2", keyframes: [{ frame: 0 }] })).toBe(
      true,
    );
    expect(promptOptional({ family: "ltx2", sourceVideoPath: "/a.mp4" })).toBe(
      true,
    );
    expect(
      promptOptional({ family: "ltx2", extendVideo: { base64: "x" } }),
    ).toBe(true);
    expect(promptOptional({ family: "ltx-video", sourceImage: "b64" })).toBe(
      true,
    );
  });

  it("keeps the prompt required for pure text-to-video", () => {
    expect(promptRequired({ family: "ltx2" })).toBe(true);
    expect(promptRequired({ family: "ltx-video" })).toBe(true);
    expect(promptRequired({ family: "ltx2", imageAttachments: [] })).toBe(true);
  });

  it("keeps the prompt required for every image family, conditioned or not", () => {
    expect(promptRequired({ family: "flux", sourceImage: "b64" })).toBe(true);
    expect(
      promptRequired({
        family: "qwen-image-edit",
        imageAttachments: [{ base64: "x" }],
      }),
    ).toBe(true);
    expect(promptRequired({ family: "sdxl", sourceImage: "b64" })).toBe(true);
  });

  it("reads web's `modelFamily` as well as desktop's `family`", () => {
    expect(
      promptOptional({
        modelFamily: "ltx2",
        imageAttachments: [{ base64: "x" }],
      }),
    ).toBe(true);
    expect(
      promptRequired({
        modelFamily: "flux",
        imageAttachments: [{ base64: "x" }],
      }),
    ).toBe(true);
  });

  it("requires a prompt when the family is unknown or the input is missing", () => {
    expect(promptRequired({ sourceImage: "b64" })).toBe(true);
    expect(promptRequired({})).toBe(true);
    expect(promptRequired(null)).toBe(true);
    expect(promptRequired(undefined)).toBe(true);
  });

  it("is the exact complement of promptOptional", () => {
    const inputs = [
      { family: "ltx2", sourceImage: "b64" },
      { family: "ltx2" },
      { family: "flux", sourceImage: "b64" },
      null,
    ];
    for (const input of inputs) {
      expect(promptRequired(input)).toBe(!promptOptional(input));
    }
  });
});

describe("copy helpers", () => {
  it("swaps the surface's own placeholder only once the prompt is optional", () => {
    expect(promptPlaceholder({ family: "flux" }, "Describe the print…")).toBe(
      "Describe the print…",
    );
    expect(
      promptPlaceholder(
        { family: "ltx2", sourceImage: "b64" },
        "Describe the print…",
      ),
    ).toBe(OPTIONAL_PROMPT_PLACEHOLDER);
    expect(
      promptPlaceholder(
        { model: "minimax-h3-fl2va:official-bf16" },
        "Describe the print…",
      ),
    ).toContain("synchronized shot");
  });

  // Two things the copy must never imply: that a blank prompt saves memory
  // (the Gemma context is a fixed [1, 1024, 4096] tensor either way), or that
  // it produces the same motion as a described one.
  it("sets honest expectations about motion and memory", () => {
    expect(OPTIONAL_PROMPT_GUIDANCE.toLowerCase()).toContain("near-static");
    expect(OPTIONAL_PROMPT_GUIDANCE.toLowerCase()).toContain("memory");
    expect(OPTIONAL_PROMPT_PLACEHOLDER.toLowerCase()).toContain("optional");
  });
});
