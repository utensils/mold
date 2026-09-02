/**
 * Browser-safe reading of the server's prompt rule.
 *
 * The generation profile is the single authority: `capabilities.prompt.mode`
 * is emitted from ONE core function (`prompt_requirement_for_family`), which
 * server validation also calls, so admission and every client necessarily
 * agree and nobody carries a family allowlist. The advertised mode answers
 * for a CONDITIONED request, because that is the only case that can differ;
 * this module resolves it against the request being built.
 *
 * Why a prompt can be optional at all: an empty prompt is a well-defined
 * trained context for LTX-2, not a degenerate one — the Gemma tokenizer pads
 * to a fixed 1024 tokens and the embeddings connector replaces every padded
 * position with learned register embeddings, so the transformer always sees
 * a full context. What makes a promptless render meaningful is the *visual*
 * conditioning — a source image, keyframes, a source video, or a
 * continuation — which is why `optional` resolves back to `required` until
 * the request carries one. `ignored` is different in kind: the family has no
 * text encoder anywhere (Hunyuan3D), so the prompt is saved as a note and
 * conditioning does not enter into it.
 *
 * Two things the surfaces must stay honest about:
 *  - It buys **zero** VRAM. The Gemma context is a fixed `[1, 1024, 4096]`
 *    tensor whose size is independent of the prompt's token count.
 *  - Expect near-static output (a blink, micro-motion). The right answer is
 *    guidance, never a synthesized placeholder prompt.
 *
 * A host that predates the field gets the rule every client applied before
 * it existed (`legacyRecipeRules.ts`), so behaviour there is unchanged.
 */

import type {
  PromptCapabilitiesProfile,
  PromptRequirement,
} from "./generated/generationProfileV1";
import { legacyPromptRequirementForFamily } from "./legacyRecipeRules";
import {
  MINIMAX_H3_PROMPT_PLACEHOLDER,
  isMinimaxH3Identity,
} from "./minimaxH3Authoring";

export type { PromptRequirement } from "./generated/generationProfileV1";

/**
 * Just enough of a resolved recipe to answer the prompt question. Narrower
 * than `GenerationRecipeProfile` so a caller (or a test) can ask without
 * building a complete profile; `prompt` is optional because an older host's
 * recipe never carried it, and its absence means `required`.
 */
export type PromptRecipe = {
  capabilities: { prompt?: PromptCapabilitiesProfile | null };
};

/**
 * The prompt requirement for THIS request, resolved from the recipe's
 * advertised mode: `optional` holds only once the request carries visual
 * conditioning, `ignored` and `required` hold regardless. A missing recipe or
 * a recipe that predates the field answers `required` — the server's own
 * serde default, and the answer that was true of every recipe before then.
 */
export function promptRequirementForRecipe(
  recipe: PromptRecipe | null | undefined,
  hasVisualConditioning: boolean,
): PromptRequirement {
  const advertised = recipe?.capabilities.prompt?.mode ?? "required";
  if (advertised === "optional" && !hasVisualConditioning) return "required";
  return advertised;
}

/**
 * The request/form fields the rule reads. Structurally satisfied by desktop's
 * `GenerateForm` (`family`, `sourceImage`) and web's form state
 * (`modelFamily`, `imageAttachments`) alike, so neither surface needs an
 * adapter that could drift from the other.
 */
export type PromptConditioningInput = {
  /**
   * The resolved generation recipe for the selected model. When present it
   * is the authority and the family fields below are never consulted for
   * the prompt rule; absent (or `null`) means an older host, and the legacy
   * family rule answers.
   */
  recipe?: PromptRecipe | null;
  /** Desktop / mobile `GenerateForm`. */
  family?: string | null;
  /** Web form state. */
  modelFamily?: string | null;
  /** Exact request identity remains available when family metadata is not. */
  model?: string | null;
  sourceImage?: unknown;
  imageAttachments?: readonly unknown[] | null;
  keyframes?: readonly unknown[] | null;
  sourceVideo?: unknown;
  sourceVideoPath?: string | null;
  extendVideo?: unknown;
  extendVideoPath?: string | null;
};

function conditioningFamily(
  input: PromptConditioningInput | null | undefined,
): string | null {
  if (!input) return null;
  return input.family ?? input.modelFamily ?? null;
}

/** Whether the request carries anything for the model to animate. */
export function hasVisualConditioning(
  input: PromptConditioningInput | null | undefined,
): boolean {
  if (!input) return false;
  return Boolean(
    input.sourceImage ||
    (input.imageAttachments?.length ?? 0) > 0 ||
    (input.keyframes?.length ?? 0) > 0 ||
    input.sourceVideo ||
    input.sourceVideoPath?.trim() ||
    input.extendVideo ||
    input.extendVideoPath?.trim(),
  );
}

/**
 * The prompt requirement for the request this input describes: the recipe's
 * advertised mode resolved against the input's conditioning, or the legacy
 * family rule when the input carries no recipe.
 */
export function promptRequirementFor(
  input: PromptConditioningInput | null | undefined,
): PromptRequirement {
  if (!input) return "required";
  const conditioned = hasVisualConditioning(input);
  if (input.recipe)
    return promptRequirementForRecipe(input.recipe, conditioned);
  const legacy = legacyPromptRequirementForFamily(conditioningFamily(input));
  return legacy === "optional" && !conditioned ? "required" : legacy;
}

/** Whether this request may be submitted with a blank prompt. */
export function promptOptional(
  input: PromptConditioningInput | null | undefined,
): boolean {
  return promptRequirementFor(input) !== "required";
}

/** Whether a non-empty prompt is a precondition for submitting. */
export function promptRequired(
  input: PromptConditioningInput | null | undefined,
): boolean {
  return !promptOptional(input);
}

/**
 * Prompt-bed placeholder once the prompt is optional. Deliberately does not
 * suggest leaving it blank is free or equivalent — the guidance line owns the
 * expectation-setting; this only says the field can be skipped.
 */
export const OPTIONAL_PROMPT_PLACEHOLDER =
  "Describe the motion — optional with a source…";

/**
 * The one shared explanation of what a blank prompt does. Web, desktop, and
 * iPhone all render this string so the three surfaces cannot set different
 * expectations.
 */
export const OPTIONAL_PROMPT_GUIDANCE =
  "With a source the prompt is optional — leave it blank and the model animates what it sees, which usually means near-static motion. It does not reduce memory use.";

/**
 * Placeholder for a recipe that IGNORES the prompt: there is no text encoder
 * to feed, so the field is a note the print carries, never guidance.
 */
export const IGNORED_PROMPT_PLACEHOLDER =
  "Optional note — this model has no text encoder and renders from the source image";

/**
 * The prompt bed's placeholder: the surface's own wording while the prompt is
 * required, the shared optional wording once it is not, and the note wording
 * for a recipe that never reads it.
 */
export function promptPlaceholder(
  input: PromptConditioningInput | null | undefined,
  requiredPlaceholder: string,
): string {
  if (isMinimaxH3Identity(conditioningFamily(input), input?.model)) {
    return MINIMAX_H3_PROMPT_PLACEHOLDER;
  }
  switch (promptRequirementFor(input)) {
    case "required":
      return requiredPlaceholder;
    case "optional":
      return OPTIONAL_PROMPT_PLACEHOLDER;
    case "ignored":
      return IGNORED_PROMPT_PLACEHOLDER;
  }
}
