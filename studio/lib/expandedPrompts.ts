/**
 * The one place Create decides whether an Expand answer is complete, shared
 * by web and desktop so the two surfaces cannot disagree about a full batch.
 *
 * A host normally returns exactly the requested number of rewrites. A recipe
 * that IGNORES the prompt is the single exception: the family has no text
 * encoder, so the host answers with ONE result — the guide's image-preparation
 * advice — whatever count was asked for. `transformCountAccepted` owns that
 * tolerance, the same rule `validateRemixVariants` applies.
 */
import {
  transformCountAccepted,
  type TransformCountOptions,
} from "./promptTransform";

/**
 * Validate the expansion response as one indivisible batch. Whitespace is
 * normalized only after the response has proven it contains exactly the
 * requested number of prompts, and a malformed response never changes the
 * requested batch size on the user's behalf. A rewrite that arrives as a
 * one-element JSON array (some expanders answer that way) is unwrapped; any
 * other text is kept verbatim apart from edge trimming.
 */
export function validateExpandedPrompts(
  prompts: readonly string[],
  expected: number,
  options?: TransformCountOptions,
): string[] {
  if (!transformCountAccepted(prompts.length, expected, options)) {
    throw new Error(
      `Expected exactly ${expected} non-empty prompts, but the host returned ${prompts.length}.`,
    );
  }
  const normalized = prompts.map(unwrapRewrite);
  const emptyIndex = normalized.findIndex((prompt) => !prompt);
  if (emptyIndex >= 0) {
    throw new Error(
      `Prompt ${emptyIndex + 1} was empty. Expected exactly ${expected} non-empty prompts.`,
    );
  }
  return normalized;
}

function unwrapRewrite(prompt: string): string {
  const trimmed = prompt.trim();
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (
      Array.isArray(parsed) &&
      parsed.length === 1 &&
      typeof parsed[0] === "string"
    ) {
      return parsed[0].trim();
    }
  } catch {
    // Ordinary prompt text is not JSON and needs only edge trimming.
  }
  return trimmed;
}
