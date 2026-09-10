import { describe, expect, it } from "vitest";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import {
  IGNORED_PROMPT_PLACEHOLDER,
  promptOptional,
  promptPlaceholder,
} from "@studio/lib/promptRequirement";
import { recipeCapabilitiesSnapshot } from "./capabilities";
import { newGenerateForm } from "./generateForm";
import { promptInputForForm, promptRecipeFromForm } from "./promptRecipe";

/** A recipe as a host advertises it for a family whose prompt is optional
 * ONCE the request carries visual conditioning (LTX-2, Wan, MiniMax H3). */
function optionalPromptRecipe() {
  const recipe = sdxlRecipe();
  recipe.capabilities.prompt = {
    mode: "optional",
    reason: "The conditioning decides the render.",
  };
  return recipe;
}

function formWith(snapshot: ReturnType<typeof recipeCapabilitiesSnapshot>) {
  const form = newGenerateForm();
  form.recipeCapabilities = snapshot;
  return form;
}

describe("promptRecipeFromForm", () => {
  it("projects the snapshotted mode back into the shared recipe shape", () => {
    const form = formWith(recipeCapabilitiesSnapshot(hunyuan3dRecipe(), "hunyuan3d"));
    expect(promptRecipeFromForm(form)).toEqual({
      capabilities: { prompt: { mode: "ignored" } },
    });
  });

  it("answers null for a host that advertises no recipe", () => {
    expect(promptRecipeFromForm(formWith(null))).toBeNull();
  });
});

describe("promptInputForForm", () => {
  it("lets an ignored-prompt recipe submit blank and names the note placeholder", () => {
    const form = formWith(recipeCapabilitiesSnapshot(hunyuan3dRecipe(), "hunyuan3d"));
    form.family = "hunyuan3d";
    form.prompt = "";
    const input = promptInputForForm(form);
    expect(promptOptional(input)).toBe(true);
    expect(promptPlaceholder(input, "Describe the image…")).toBe(IGNORED_PROMPT_PLACEHOLDER);
  });

  // The adapter is an explicit projection, not a spread: a spread only ever
  // carried the fields whose names already matched, so an LTX-2 end frame and
  // every H3 boundary frame were invisible to the rule.
  it("lets a conditioned Wan render submit blank", () => {
    const form = formWith(recipeCapabilitiesSnapshot(optionalPromptRecipe(), "wan"));
    form.family = "wan";
    form.prompt = "";
    expect(promptOptional(promptInputForForm(form))).toBe(false);
    form.sourceImage = "b64";
    expect(promptOptional(promptInputForForm(form))).toBe(true);
  });

  // A lone end frame ships nothing (keyframes need a first frame, and the
  // source well blocks submit), so the composer must not promise an optional
  // prompt it cannot deliver.
  it("does not read a lone LTX-2 end frame as conditioning", () => {
    const form = formWith(recipeCapabilitiesSnapshot(optionalPromptRecipe(), "ltx2"));
    form.family = "ltx2";
    form.prompt = "";
    form.endFrame = { filename: "last.png", base64: "b64" };
    expect(promptOptional(promptInputForForm(form))).toBe(false);
  });

  it("reads an H3 last frame and an H3 reference as the conditioning", () => {
    const form = formWith(recipeCapabilitiesSnapshot(optionalPromptRecipe(), "minimax-h3"));
    form.family = "minimax-h3";
    form.model = "minimax-h3-fl2va:official-bf16";
    form.prompt = "";
    form.h3Authoring = {
      firstFrame: null,
      lastFrame: {
        filename: "last.png",
        mimeType: "image/png",
        width: 1,
        height: 1,
        data: "b64",
      },
      references: [],
    };
    expect(promptOptional(promptInputForForm(form))).toBe(true);

    form.h3Authoring.lastFrame = null;
    expect(promptOptional(promptInputForForm(form))).toBe(false);
    form.h3Authoring.references = [{ reference: { kind: "image", image: "b64" } } as never];
    expect(promptOptional(promptInputForForm(form))).toBe(true);
  });

  it("keeps a raster recipe's prompt required", () => {
    const form = formWith(recipeCapabilitiesSnapshot(sdxlRecipe(), "sdxl"));
    form.family = "sdxl";
    const input = promptInputForForm(form);
    expect(promptOptional(input)).toBe(false);
    expect(promptPlaceholder(input, "Describe the image…")).toBe("Describe the image…");
  });
});
