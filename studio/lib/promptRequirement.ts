/**
 * Browser-safe mirror of the server's optional-prompt rule
 * (`validate_generate_request_with_family` in `crates/mold-core/src/validation.rs`).
 *
 * An empty prompt is a well-defined trained context for LTX-2, not a
 * degenerate one: the Gemma tokenizer pads to a fixed 1024 tokens and the
 * embeddings connector replaces every padded position with learned register
 * embeddings, so the transformer always sees a full context. Upstream ships
 * `""` as an encodable prompt value itself. What actually makes a promptless
 * render meaningful is the *visual* conditioning — a source image, keyframes,
 * a source video, or a continuation — which is why the two must travel
 * together. Pure text-to-video and every image family keep the prompt
 * required, because there would be nothing left to describe the print.
 *
 * Two things the surfaces must stay honest about:
 *  - It buys **zero** VRAM. The Gemma context is a fixed `[1, 1024, 4096]`
 *    tensor whose size is independent of the prompt's token count.
 *  - Expect near-static output (a blink, micro-motion). The right answer is
 *    guidance, never a synthesized placeholder prompt.
 *
 * The server stays authoritative; this exists so the Create surfaces stop
 * disabling Generate on a request the server would happily admit.
 */

import {
  MINIMAX_H3_PROMPT_PLACEHOLDER,
  isMinimaxH3Identity,
} from "./minimaxH3Authoring";

/** Families whose engines can render from visual conditioning alone. */
const PROMPT_OPTIONAL_FAMILIES: ReadonlySet<string> = new Set([
  "ltx2",
  "ltx-2",
  "ltx-video",
]);

export function familyAllowsEmptyPrompt(
  family: string | null | undefined,
): boolean {
  return PROMPT_OPTIONAL_FAMILIES.has((family ?? "").trim().toLowerCase());
}

/**
 * The request/form fields the rule reads. Structurally satisfied by desktop's
 * `GenerateForm` (`family`, `sourceImage`) and web's form state
 * (`modelFamily`, `imageAttachments`) alike, so neither surface needs an
 * adapter that could drift from the other.
 */
export type PromptConditioningInput = {
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

/** Whether this request may be submitted with a blank prompt. */
export function promptOptional(
  input: PromptConditioningInput | null | undefined,
): boolean {
  if (!input) return false;
  return (
    familyAllowsEmptyPrompt(conditioningFamily(input)) &&
    hasVisualConditioning(input)
  );
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
 * The prompt bed's placeholder: the surface's own wording while the prompt is
 * required, the shared optional wording once it is not.
 */
export function promptPlaceholder(
  input: PromptConditioningInput | null | undefined,
  requiredPlaceholder: string,
): string {
  if (isMinimaxH3Identity(conditioningFamily(input), input?.model)) {
    return MINIMAX_H3_PROMPT_PLACEHOLDER;
  }
  return promptRequired(input)
    ? requiredPlaceholder
    : OPTIONAL_PROMPT_PLACEHOLDER;
}
