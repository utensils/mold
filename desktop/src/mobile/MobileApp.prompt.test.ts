import { describe, expect, it } from "vitest";
import { hunyuan3dRecipe } from "@studio/lib/generationProfile.testFixtures";
import {
  IGNORED_PROMPT_PLACEHOLDER,
  promptPlaceholder,
  promptRequired,
} from "@studio/lib/promptRequirement";
import mobileAppSource from "./MobileApp.vue?raw";

/**
 * iPhone half of the optional-prompt rule. A conditioned LTX-2 render may go
 * out undescribed, so the Develop button and the pre-submit guard both stop
 * demanding a prompt the host would accept — and, like desktop, they have to
 * move together or the enabled control becomes a dead end.
 */
describe("MobileApp prompt requirement", () => {
  it("derives the Develop gate from the shared predicate", () => {
    expect(mobileAppSource).toContain('from "@studio/lib/promptRequirement"');
    expect(mobileAppSource).toMatch(
      /const promptMissing = computed\(\s*\(\) => promptRequired\(promptConditioning\.value\) && !form\.prompt\.trim\(\),?\s*\);/,
    );
  });

  /**
   * The advertised recipe is the AUTHORITY on the prompt rule — a family
   * allowlist cannot see `ignored`, which is what a text-encoder-free 3-D
   * checkpoint reports. Reading the form alone left Develop disabled forever
   * on a model whose prompt the host never encodes.
   */
  it("resolves the rule against the selected model's advertised recipe", () => {
    expect(mobileAppSource).toMatch(
      /const promptConditioning = computed\(\(\) => \(\{\s*\.\.\.form,\s*recipe: effectiveGenerationRecipe\(selectedGenerationModel\.value, form\.pipeline\),\s*\}\)\);/,
    );
  });

  it("uses that one gate for both the button and the pre-submit guard", () => {
    const occurrences = mobileAppSource.match(/promptMissing(\.value)?\b/g) ?? [];
    // definition + the shared Develop-disabled computation + the `generate()` guard
    expect(occurrences.length).toBeGreaterThanOrEqual(3);
    expect(mobileAppSource).toMatch(/promptMissing\.value \|\|\s*!form\.model\.trim\(\)/);
    expect(mobileAppSource).toContain(':disabled="developDisabled"');
    // The bare check must not survive anywhere in the generation path; only
    // "Use as prompt" may still skip a print with no recorded prompt.
    expect(mobileAppSource).not.toContain("!form.prompt.trim() ||");
  });

  it("softens the prompt placeholder through the shared helper", () => {
    expect(mobileAppSource).toMatch(
      /const promptFieldPlaceholder = computed\(\(\) =>\s*promptPlaceholder\(promptConditioning\.value, "Describe the print…"\),?\s*\);/,
    );
    expect(mobileAppSource).toContain(':placeholder="promptFieldPlaceholder"');
  });
});

/**
 * The recipe answers, so the shared helpers are exercised here directly with
 * the exact input `MobileApp` builds: a 3-D checkpoint has no text encoder
 * anywhere, so an empty prompt must never block Develop and the bed says so.
 */
describe("mobile prompt rule on a text-encoder-free recipe", () => {
  const conditioning = {
    family: "hunyuan3d",
    model: "hunyuan3d-mini-turbo:fp16",
    sourceImage: "c291cmNl",
    recipe: hunyuan3dRecipe(),
  };

  it("never demands a prompt and names the field a note", () => {
    expect(promptRequired(conditioning)).toBe(false);
    expect(promptPlaceholder(conditioning, "Describe the print…")).toBe(IGNORED_PROMPT_PLACEHOLDER);
  });

  it("still demands one without the recipe's answer", () => {
    const { recipe: _recipe, ...legacy } = conditioning;
    expect(promptRequired(legacy)).toBe(true);
  });
});
