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

  it("keeps a raster recipe's prompt required", () => {
    const form = formWith(recipeCapabilitiesSnapshot(sdxlRecipe(), "sdxl"));
    form.family = "sdxl";
    const input = promptInputForForm(form);
    expect(promptOptional(input)).toBe(false);
    expect(promptPlaceholder(input, "Describe the image…")).toBe("Describe the image…");
  });
});
