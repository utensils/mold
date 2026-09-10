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
 * Why a prompt can be optional at all: for the families that advertise it,
 * an empty prompt is a well-defined trained context rather than a degenerate
 * one — each family's text encoder has its own reason (LTX-2's Gemma pads to
 * a fixed 1024 positions and the connector fills them with learned register
 * embeddings; Wan's umT5 encodes the empty string on its unconditional
 * branch). What makes a promptless render meaningful is the *visual*
 * conditioning — a source image, keyframes, boundary frames, an ordered
 * reference, a source video, or a continuation — which is why `optional`
 * resolves back to `required` until the request carries one. `ignored` is
 * different in kind: the family has no text encoder anywhere (Hunyuan3D), so
 * the prompt is saved as a note and conditioning does not enter into it.
 *
 * What the surfaces must stay honest about: expect near-static output (a
 * blink, micro-motion). The right answer is guidance, never a synthesized
 * placeholder prompt. The copy names no memory saving — that was an LTX-2
 * fact about a fixed-size Gemma context and it does not generalize.
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
   * The resolved generation recipe for the selected model. A present prompt
   * block is authoritative. An absent recipe or prompt block uses the legacy
   * family rule for older hosts.
   */
  recipe?: PromptRecipe | null;
  /** Desktop / mobile `GenerateForm`. */
  family?: string | null;
  /** Web form state. */
  modelFamily?: string | null;
  /** Exact request identity remains available when family metadata is not. */
  model?: string | null;
  sourceImage?: unknown;
  /**
   * Web's single well as well as an ordered reference strip: on a
   * `single`-layout web recipe `imageAttachments[0]` IS the source image
   * (`web/src/composables/useGenerateForm.ts`), so dropping it to match the
   * server's field list would un-condition every web img2img render. It also
   * counts references on a reference-only recipe, which is inert: those
   * families advertise `required` whatever they carry.
   */
  imageAttachments?: readonly unknown[] | null;
  keyframes?: readonly unknown[] | null;
  sourceVideo?: unknown;
  sourceVideoPath?: string | null;
  extendVideo?: unknown;
  extendVideoPath?: string | null;
  /**
   * MiniMax H3 Ref2VA's ordered references — the ONLY place a Ref2VA
   * request carries its frames. (An LTX-2 end frame is deliberately NOT an
   * input: it ships as a keyframe only beside a first frame, which already
   * counts, and alone it ships nothing — so it must not read as conditioning.)
   */
  references?: readonly unknown[] | null;
  /** MiniMax H3 first/last-frame authoring, which lives outside
   * `sourceImage` and `keyframes` on every surface. */
  h3FirstFrame?: unknown;
  h3LastFrame?: unknown;
};

/**
 * A surface form as this module reads it. Desktop's and iPhone's
 * `GenerateForm` and web's form state both satisfy it structurally; the
 * fields the shared input does not name by the same name are projected by
 * {@link promptConditioningInputFor}.
 */
export type PromptConditioningSource = Omit<
  PromptConditioningInput,
  "h3FirstFrame" | "h3LastFrame"
> & {
  /** Desktop / web / iPhone MiniMax H3 authoring state. */
  h3Authoring?: {
    firstFrame?: unknown;
    lastFrame?: unknown;
    references?: readonly unknown[] | null;
  } | null;
  /** A surface that names its reference strip differently. */
  referenceImages?: readonly unknown[] | null;
  /** Accepted so a form can be handed over whole; deliberately not read (see
   * `references` above). */
  endFrame?: unknown;
};

/**
 * Web initialises `h3Authoring` with an EMPTY reference list, and `[] ?? x`
 * is `[]`, so a plain nullish chain would hide every later strip. The first
 * list that actually holds something is the one the request will carry.
 */
function firstNonEmpty(
  ...lists: (readonly unknown[] | null | undefined)[]
): readonly unknown[] | null {
  return lists.find((list) => (list?.length ?? 0) > 0) ?? null;
}

/**
 * The ONE projection from a surface's form onto the shared rule's input.
 *
 * Desktop, web and iPhone all hold the same conditioning under different
 * field names (H3 frames sit inside `h3Authoring`, references may be
 * `references` or `referenceImages`), and a spread only ever carried the
 * fields that already matched — which is how every H3 frame stayed invisible
 * to the rule. Reading the form here keeps that map in one place instead of
 * three.
 */
export function promptConditioningInputFor(
  source: PromptConditioningSource | null | undefined,
  recipe?: PromptRecipe | null,
): PromptConditioningInput {
  if (!source) return recipe ? { recipe } : {};
  return {
    recipe: recipe ?? source.recipe ?? null,
    family: source.family ?? null,
    modelFamily: source.modelFamily ?? null,
    model: source.model ?? null,
    sourceImage: source.sourceImage ?? null,
    imageAttachments: source.imageAttachments ?? null,
    keyframes: source.keyframes ?? null,
    sourceVideo: source.sourceVideo ?? null,
    sourceVideoPath: source.sourceVideoPath ?? null,
    extendVideo: source.extendVideo ?? null,
    extendVideoPath: source.extendVideoPath ?? null,
    references: firstNonEmpty(
      source.h3Authoring?.references,
      source.references,
      source.referenceImages,
    ),
    h3FirstFrame: source.h3Authoring?.firstFrame ?? null,
    h3LastFrame: source.h3Authoring?.lastFrame ?? null,
  };
}

function conditioningFamily(
  input: PromptConditioningInput | null | undefined,
): string | null {
  if (!input) return null;
  return input.family ?? input.modelFamily ?? null;
}

/** Whether the request carries anything for the model to render from. */
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
    input.extendVideoPath?.trim() ||
    (input.references?.length ?? 0) > 0 ||
    input.h3FirstFrame ||
    input.h3LastFrame,
  );
}

/**
 * The prompt requirement for the request this input describes: the recipe's
 * advertised mode resolved against the input's conditioning, or the legacy
 * family rule when the input carries no advertised prompt block.
 */
export function promptRequirementFor(
  input: PromptConditioningInput | null | undefined,
): PromptRequirement {
  if (!input) return "required";
  const conditioned = hasVisualConditioning(input);
  // A profile may predate this additive block. Match generationCapabilities:
  // only a present prompt rule overrides the legacy family contract.
  if (input.recipe?.capabilities.prompt)
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
 * expectation-setting; this only says the field can be skipped, naming the
 * attachment rather than any one family's conditioning channel.
 */
export const OPTIONAL_PROMPT_PLACEHOLDER =
  "Describe the motion — optional with what you attached…";

/**
 * The one shared explanation of what a blank prompt does. Web, desktop, and
 * iPhone all render this string so the three surfaces cannot set different
 * expectations.
 *
 * It names the conditioning generically because the rule now covers a source
 * image, a source video, a continuation, boundary frames and an ordered
 * reference across three families, and it claims no memory saving: that was
 * an LTX-2 fact about a fixed-size Gemma context.
 */
export const OPTIONAL_PROMPT_GUIDANCE =
  "The prompt is optional with what you attached — leave it blank and the model works from the attached image or frames, which usually means near-static motion.";

/**
 * Placeholder for a recipe that IGNORES the prompt: there is no text encoder
 * to feed, so the field is a note the print carries, never guidance.
 */
export const IGNORED_PROMPT_PLACEHOLDER =
  "Optional note — this model has no text encoder and renders from the source image";

/**
 * The one shared explanation for a recipe that IGNORES the prompt, shown
 * where the empty canvas tells a first-time user what to do. It must not
 * borrow the optional-prompt wording: nothing here "animates what it sees",
 * and there is no prompt to describe motion with. The source image is the
 * whole input, so the advice is about preparing it.
 */
export const IGNORED_PROMPT_GUIDANCE =
  "This model reads no prompt — it renders from the source image alone. Attach a clean cutout on a plain background and press Generate; anything typed is saved as a note.";

/**
 * The empty-canvas guidance for the request this input describes: the
 * surface's own wording while the prompt is required, the shared optional
 * wording once it is not, and the image-preparation wording for a recipe
 * that never reads it.
 */
export function promptGuidance(
  input: PromptConditioningInput | null | undefined,
  requiredGuidance: string,
): string {
  switch (promptRequirementFor(input)) {
    case "required":
      return requiredGuidance;
    case "optional":
      return OPTIONAL_PROMPT_GUIDANCE;
    case "ignored":
      return IGNORED_PROMPT_GUIDANCE;
  }
}

/**
 * The prompt bed's placeholder: the surface's own wording while the prompt is
 * required, the shared optional wording once it is not, and the note wording
 * for a recipe that never reads it.
 */
export function promptPlaceholder(
  input: PromptConditioningInput | null | undefined,
  requiredPlaceholder: string,
): string {
  const requirement = promptRequirementFor(input);
  // MiniMax H3's own wording is what a request that still NEEDS a prompt is
  // told. Once its conditioning makes the prompt optional the shared optional
  // wording wins, or a conditioned H3 render would never learn it can skip
  // the field.
  if (
    requirement === "required" &&
    isMinimaxH3Identity(conditioningFamily(input), input?.model)
  ) {
    return MINIMAX_H3_PROMPT_PLACEHOLDER;
  }
  switch (requirement) {
    case "required":
      return requiredPlaceholder;
    case "optional":
      return OPTIONAL_PROMPT_PLACEHOLDER;
    case "ignored":
      return IGNORED_PROMPT_PLACEHOLDER;
  }
}
