/**
 * The one place Create decides whether an Expand answer is complete.
 *
 * A host normally returns exactly the requested number of rewrites. A recipe
 * that IGNORES the prompt is the single exception: the family has no text
 * encoder, so the host answers with ONE result — the guide's
 * image-preparation advice — whatever count was asked for. Accepting that
 * through `transformCountAccepted` keeps the tolerance in the shared studio
 * rule rather than in a second copy per surface.
 */
import {
  transformCountAccepted,
  type TransformCountOptions,
} from "@studio/lib/promptTransform";

export function validateExpandedPrompts(
  prompts: readonly string[],
  expected: number,
  options?: TransformCountOptions,
): string[] {
  const normalized = prompts.map((prompt) => prompt.trim());
  if (
    !transformCountAccepted(normalized.length, expected, options) ||
    normalized.some((prompt) => !prompt)
  ) {
    throw new Error(`Expected exactly ${expected} non-empty expanded prompts.`);
  }
  return normalized;
}
