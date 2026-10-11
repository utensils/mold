import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { GenerationProfileSet } from "./generationProfile";
import { baseGenerationCapabilities } from "./generationCapabilities";
import { sourceMediaPlan } from "./sourceMediaPlan";
import {
  emptyMinimaxH3AuthoringState,
  minimaxH3AuthoringCapabilities,
  minimaxH3AuthoringError,
  serializeMinimaxH3Authoring,
} from "./minimaxH3Authoring";

let directory = process.cwd();
const relative = "docs/generated/generation-profiles-v1.json";
while (!existsSync(resolve(directory, relative))) {
  const parent = dirname(directory);
  if (parent === directory) throw new Error(`Missing ${relative}`);
  directory = parent;
}
const registry = JSON.parse(
  readFileSync(resolve(directory, relative), "utf8"),
) as {
  profiles: { models: { model: string }[]; profile: GenerationProfileSet }[];
};

// Exercise the actual Rust-generated profiles through the shared web, desktop
// and mobile authoring/serialization path, rather than hand-copying controls.
describe("H3 distilled tiers across browser and Tauri authoring", () => {
  it.each(["required", "unsupported", "optional"] as const)(
    "honors explicit %s even when the saved row disagrees",
    (source) => {
      const generated = registry.profiles.find((entry) =>
        entry.models.some(
          (row) => row.model === "minimax-h3-fl2va:fasth3-v2-int8",
        ),
      )!.profile.recipes[0]!;
      const caps = baseGenerationCapabilities(
        "minimax-h3",
        "minimax-h3-fl2va:fasth3-v2-int8",
        null,
        null,
        "required",
        {
          ...generated,
          capabilities: { ...generated.capabilities, source_image: source },
        },
      );
      expect(caps.sourceImageCapability).toBe(source);
      expect(caps.requiresSourceImage).toBe(source === "required");
      expect(caps.supportsSourceImage).toBe(source !== "unsupported");
      expect(
        baseGenerationCapabilities(
          "minimax-h3",
          "minimax-h3-fl2va:fasth3-v2-int8",
          null,
          null,
          source,
          generated,
        ).sourceImageCapability,
      ).toBe(source);
    },
  );

  it.each([
    ["minimax-h3-fl2va:fasth3-v2-int8", 9],
    ["minimax-h3-fl2va:comfy-pruned-int8-dmad-4step", 5],
    ["minimax-h3-fl2va:comfy-pruned-int8-pdmd-2step", 3],
    ["minimax-h3-fl2va:comfy-pruned-int8-pdmd-4step", 5],
  ] as const)(
    "uses %s's authoritative optional-input recipe and exact schedule",
    (model, steps) => {
      const profile = registry.profiles.find((entry) =>
        entry.models.some((row) => row.model === model),
      )!.profile;
      const generated = profile.recipes.find(
        (row) => row.id === profile.default_recipe_id,
      )!;
      // h3_private_bridge's public-runtime overlay adds this additive field;
      // the static registry deliberately has no installed-runtime authority.
      const recipe = {
        ...generated,
        capabilities: {
          ...generated.capabilities,
          source_image: "optional" as const,
        },
      };
      const caps = baseGenerationCapabilities(
        "minimax-h3",
        model,
        null,
        null,
        "required",
        recipe,
      );
      expect(caps.requiresSourceImage).toBe(false);
      expect(sourceMediaPlan(caps)).toEqual({
        kind: "h3-boundaries",
        requiredEndpoint: null,
      });
      const authoring = emptyMinimaxH3AuthoringState();
      expect(
        minimaxH3AuthoringError(
          "minimax-h3",
          model,
          authoring,
          caps.requiresSourceImage,
        ),
      ).toBeNull();
      expect(recipe.steps).toMatchObject({
        mode: "fixed",
        min: steps,
        max: steps,
        default: steps,
      });
      expect(
        minimaxH3AuthoringCapabilities({ name: model, family: "minimax-h3" }),
      ).toMatchObject({ minSteps: steps, maxSteps: steps });
      const request = serializeMinimaxH3Authoring(
        {
          ...recipe.defaults,
          model,
          prompt: "A lighthouse in a storm",
          guidance: 9,
          batch_size: 4,
        },
        "minimax-h3",
        model,
        authoring,
      );
      expect(request).toMatchObject({
        model,
        steps,
        fps: 24,
        guidance: 0,
        batch_size: 1,
        output_format: "mp4",
      });
      expect(request).not.toHaveProperty("source_image");
      expect(request).not.toHaveProperty("references");
      expect(request).not.toHaveProperty("keyframes");
      const conditioned = serializeMinimaxH3Authoring(
        request,
        "minimax-h3",
        model,
        {
          ...authoring,
          firstFrame: {
            filename: "source.png",
            mimeType: "image/png",
            width: 1344,
            height: 768,
            data: "SOURCE",
          },
        },
      );
      expect(conditioned).toMatchObject({
        source_image: "SOURCE",
        source_image_name: "source.png",
        steps,
      });
    },
  );
});
