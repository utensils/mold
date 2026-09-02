import { describe, expect, it } from "vitest";

import { hunyuan3dRecipe, sdxlRecipe } from "./generationProfile.testFixtures";
import {
  IGNORED_PROMPT_PLACEHOLDER,
  IGNORED_PROMPT_GUIDANCE,
  OPTIONAL_PROMPT_GUIDANCE,
  promptGuidance,
  OPTIONAL_PROMPT_PLACEHOLDER,
  hasVisualConditioning,
  promptOptional,
  promptPlaceholder,
  promptRequired,
  promptRequirementFor,
  promptRequirementForRecipe,
} from "./promptRequirement";

/** An LTX-2 recipe as the host advertises it: optional WITH conditioning. */
function ltxRecipe() {
  const recipe = sdxlRecipe();
  recipe.capabilities.prompt = {
    mode: "optional",
    reason: "A conditioned render animates what it sees.",
  };
  return recipe;
}

describe("promptRequirementForRecipe", () => {
  // The advertised mode answers for a CONDITIONED request, because that is
  // the only case that can differ; the caller resolves it against the
  // request it is building.
  it("resolves an optional recipe to required until conditioning arrives", () => {
    expect(promptRequirementForRecipe(ltxRecipe(), false)).toBe("required");
    expect(promptRequirementForRecipe(ltxRecipe(), true)).toBe("optional");
  });

  it("keeps ignored ignored and required required regardless of conditioning", () => {
    expect(promptRequirementForRecipe(hunyuan3dRecipe(), false)).toBe(
      "ignored",
    );
    expect(promptRequirementForRecipe(hunyuan3dRecipe(), true)).toBe("ignored");
    expect(promptRequirementForRecipe(sdxlRecipe(), true)).toBe("required");
    expect(promptRequirementForRecipe(sdxlRecipe(), false)).toBe("required");
  });

  it("reads an older host's recipe that never advertised a prompt mode as required", () => {
    const recipe = sdxlRecipe();
    delete (recipe.capabilities as { prompt?: unknown }).prompt;
    expect(promptRequirementForRecipe(recipe, true)).toBe("required");
    expect(promptRequirementForRecipe(null, true)).toBe("required");
    expect(promptRequirementForRecipe(undefined, true)).toBe("required");
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

  it("prefers the recipe carried on the input over the family rule", () => {
    // A hunyuan3d recipe ignores the prompt entirely: optional even without
    // conditioning, and the family string is never consulted.
    expect(
      promptRequirementFor({ family: "flux", recipe: hunyuan3dRecipe() }),
    ).toBe("ignored");
    expect(promptOptional({ family: "flux", recipe: hunyuan3dRecipe() })).toBe(
      true,
    );
    // An optional recipe still needs conditioning to resolve.
    expect(promptRequired({ family: "ltx2", recipe: ltxRecipe() })).toBe(true);
    expect(
      promptOptional({
        family: "ltx2",
        recipe: ltxRecipe(),
        sourceImage: "b64",
      }),
    ).toBe(true);
    // A required recipe overrides a legacy-optional family: the host knows.
    expect(
      promptRequired({
        family: "ltx2",
        recipe: sdxlRecipe(),
        sourceImage: "b64",
      }),
    ).toBe(true);
  });

  it("uses the legacy family rule when the input carries no recipe", () => {
    expect(promptRequirementFor({ family: "ltx2", sourceImage: "b64" })).toBe(
      "optional",
    );
    expect(promptRequirementFor({ family: "ltx2" })).toBe("required");
    expect(
      promptRequirementFor({
        family: "ltx2",
        recipe: null,
        sourceImage: "b64",
      }),
    ).toBe("optional");
    expect(promptRequirementFor(null)).toBe("required");
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

  it("tells the user the prompt is a note when the recipe ignores it", () => {
    expect(
      promptPlaceholder(
        { family: "hunyuan3d", recipe: hunyuan3dRecipe() },
        "Describe the print…",
      ),
    ).toBe(IGNORED_PROMPT_PLACEHOLDER);
    expect(IGNORED_PROMPT_PLACEHOLDER.toLowerCase()).toContain("note");
  });

  it("explains the empty canvas from the prompt rule, never the optional wording for an ignored prompt", () => {
    const required = "Describe an image below and press Generate.";
    expect(promptGuidance({ family: "flux" }, required)).toBe(required);
    expect(
      promptGuidance({ family: "ltx2", sourceImage: "b64" }, required),
    ).toBe(OPTIONAL_PROMPT_GUIDANCE);
    expect(
      promptGuidance(
        { family: "hunyuan3d", recipe: hunyuan3dRecipe() },
        required,
      ),
    ).toBe(IGNORED_PROMPT_GUIDANCE);
    expect(
      promptGuidance({ family: "sdxl", recipe: sdxlRecipe() }, required),
    ).toBe(required);
    // The ignored wording is about the image, not motion, and never claims
    // the model reads anything typed.
    expect(IGNORED_PROMPT_GUIDANCE.toLowerCase()).toContain("source image");
    expect(IGNORED_PROMPT_GUIDANCE.toLowerCase()).toContain("no prompt");
    expect(IGNORED_PROMPT_GUIDANCE.toLowerCase()).not.toContain("animates");
    expect(IGNORED_PROMPT_GUIDANCE).not.toBe(OPTIONAL_PROMPT_GUIDANCE);
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
