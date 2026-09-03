import { describe, expect, it } from "vitest";
import { baseGenerationCapabilities } from "./generationCapabilities";
import {
  EXCLUSIVE_WELLS_NOTE,
  resolveExclusiveWells,
  sourceMediaPlan,
} from "./sourceMediaPlan";
import {
  flux2DevRecipe,
  flux2KleinRecipe,
  qwenImageEditRecipe,
} from "./generationProfile.testFixtures";
import type { GenerationRecipeProfile } from "./generationProfile";

function plan(
  family: string,
  model = "",
  advertisedSourceImage: string | null = null,
  recipe: GenerationRecipeProfile | null = null,
) {
  return sourceMediaPlan(
    baseGenerationCapabilities(
      family,
      model,
      null,
      null,
      advertisedSourceImage,
      recipe,
    ),
  );
}

describe("sourceMediaPlan", () => {
  it("renders one optional well for image families", () => {
    expect(plan("flux")).toEqual({
      kind: "single",
      required: false,
      endFrame: false,
      video: false,
    });
  });

  it("renders nothing for a text-to-video family without image conditioning", () => {
    expect(plan("ltx-video")).toEqual({ kind: "none" });
  });

  it("follows the advertised per-model contract over the family heuristic", () => {
    // wan T2V says unsupported; wan I2V says required with keyframes.
    expect(plan("wan", "wan-t2v", "unsupported")).toEqual({ kind: "none" });
    expect(plan("wan", "wan-i2v", "required")).toEqual({
      kind: "single",
      required: true,
      endFrame: true,
      video: true,
    });
  });

  it("caps FLUX.2 Dev references and leaves Qwen edit unbounded", () => {
    expect(plan("flux2", "flux2-dev")).toEqual({
      kind: "attachments",
      max: 4,
      required: false,
      primary: null,
    });
    expect(plan("qwen-image-edit")).toEqual({
      kind: "attachments",
      max: null,
      required: true,
      primary: "target",
    });
  });

  it("maps MiniMax H3 tasks to boundary wells and the reference panel", () => {
    for (const model of [
      "minimax-h3-fl2va:official-bf16",
      "minimax-h3-fl2va:comfy-pruned-int8",
      "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step",
      "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
      "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1",
      "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p",
      "minimax-h3-fl2va:comfy-pruned-nvfp4",
    ]) {
      expect(plan("minimax-h3", model)).toEqual({
        kind: "h3-boundaries",
        requiredEndpoint: null,
      });
    }
    expect(
      plan("minimax-h3", "minimax-h3-fl2va:comfy-pruned-int8", "required"),
    ).toEqual({ kind: "h3-boundaries", requiredEndpoint: "first" });
    for (const model of [
      "minimax-h3-ref2va:official-bf16",
      "minimax-h3-ref2va:comfy-pruned-int8",
      "minimax-h3-ref2va:comfy-pruned-nvfp4",
    ]) {
      expect(plan("minimax-h3", model)).toEqual({ kind: "h3-references" });
    }
  });
});

describe("sourceMediaPlan reference bounds come from the recipe", () => {
  it("takes the strip ceiling, requiredness and target from the profile", () => {
    expect(plan("flux2", "flux2-dev:bf16", null, flux2DevRecipe())).toEqual({
      kind: "attachments",
      max: 4,
      required: false,
      primary: null,
    });
    expect(
      plan(
        "qwen-image-edit",
        "qwen-image-edit-2511:q4",
        null,
        qwenImageEditRecipe(),
      ),
    ).toEqual({
      kind: "attachments",
      max: null,
      required: true,
      primary: "target",
    });
  });

  it("reads a bound this client never hard-coded", () => {
    const recipe = flux2DevRecipe();
    const caps = baseGenerationCapabilities(
      "flux2",
      "flux2-dev:bf16",
      null,
      null,
      null,
      {
        ...recipe,
        capabilities: {
          ...recipe.capabilities,
          reference_images: {
            ...recipe.capabilities.reference_images!,
            max_count: 6,
          },
        },
      } as GenerationRecipeProfile,
    );
    expect(sourceMediaPlan(caps)).toEqual({
      kind: "attachments",
      max: 6,
      required: false,
      primary: null,
    });
  });

  it("renders BOTH wells for an exclusive recipe", () => {
    expect(plan("flux2", "flux2-klein:bf16", null, flux2KleinRecipe())).toEqual(
      {
        kind: "single-or-references",
        single: { required: false, endFrame: false, video: false },
        references: {
          max: 4,
          maxPixelsSingle: 4_096_576,
          maxPixelsMulti: 1_048_576,
        },
      },
    );
  });
});

describe("resolveExclusiveWells", () => {
  it("parks nothing while both wells are empty", () => {
    expect(
      resolveExclusiveWells({ hasSource: false, referenceCount: 0 }),
    ).toEqual({
      active: null,
      parked: null,
      note: null,
    });
  });

  it("makes the well that holds media the active one", () => {
    expect(
      resolveExclusiveWells({ hasSource: true, referenceCount: 0 }),
    ).toEqual({
      active: "source",
      parked: "references",
      note: EXCLUSIVE_WELLS_NOTE,
    });
    expect(
      resolveExclusiveWells({ hasSource: false, referenceCount: 2 }),
    ).toEqual({
      active: "references",
      parked: "source",
      note: EXCLUSIVE_WELLS_NOTE,
    });
  });

  it("is last-write-wins when both hold media, and the parked media survives", () => {
    // A reference dropped onto a form that already had a source: the source
    // is PARKED, not discarded — it comes back when the references clear.
    expect(
      resolveExclusiveWells({
        hasSource: true,
        referenceCount: 1,
        lastWrite: "references",
      }),
    ).toEqual({
      active: "references",
      parked: "source",
      note: EXCLUSIVE_WELLS_NOTE,
    });
    // …and the other direction, which a "references win when present" rule
    // would get wrong.
    expect(
      resolveExclusiveWells({
        hasSource: true,
        referenceCount: 1,
        lastWrite: "source",
      }),
    ).toEqual({
      active: "source",
      parked: "references",
      note: EXCLUSIVE_WELLS_NOTE,
    });
  });

  it("falls back to the source well when nothing recorded the last write", () => {
    // A restored draft carrying both (an older snapshot has no marker).
    expect(
      resolveExclusiveWells({
        hasSource: true,
        referenceCount: 1,
        lastWrite: null,
      }),
    ).toEqual({
      active: "source",
      parked: "references",
      note: EXCLUSIVE_WELLS_NOTE,
    });
  });

  it("ignores a stale last write for a well that no longer holds media", () => {
    expect(
      resolveExclusiveWells({
        hasSource: false,
        referenceCount: 3,
        lastWrite: "source",
      }).active,
    ).toBe("references");
  });
});
