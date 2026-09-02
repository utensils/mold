/**
 * The prompt rule's recipe authority, read off the form.
 *
 * `@studio/lib/promptRequirement` takes the resolved recipe as the authority
 * and falls back to the pre-profile family rule without one. Desktop's
 * composer surfaces do not all hold the selected `ModelEntry`, but every one
 * of them holds the form — and `applyModelDefaults` already snapshotted the
 * recipe's advertised mode into `recipeCapabilities.promptMode`. Projecting
 * that snapshot back into the narrow `PromptRecipe` shape keeps the composer,
 * the view's submit gate, and the sequence rail on one answer, so a recipe
 * that IGNORES the prompt (Hunyuan3D has no text encoder) enables Generate
 * with an empty prompt everywhere at once.
 *
 * A form with no snapshot is an older host that advertises no recipe: return
 * `null` and the legacy family rule answers, exactly as before.
 */
import type { PromptConditioningInput, PromptRecipe } from "@studio/lib/promptRequirement";
import type { GenerateForm } from "./generateForm";

export function promptRecipeFromForm(
  form: Pick<GenerateForm, "recipeCapabilities">,
): PromptRecipe | null {
  const snapshot = form.recipeCapabilities;
  return snapshot ? { capabilities: { prompt: { mode: snapshot.promptMode } } } : null;
}

/**
 * The form as the shared prompt rule reads it: its own conditioning fields
 * (`GenerateForm` satisfies `PromptConditioningInput` structurally) plus the
 * advertised recipe when the host sent one.
 */
export function promptInputForForm(form: GenerateForm): PromptConditioningInput {
  const recipe = promptRecipeFromForm(form);
  return recipe ? { ...form, recipe } : form;
}
