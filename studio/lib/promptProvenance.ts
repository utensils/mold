/** Minimal mutable shape shared by the web, desktop, and iPhone composers. */
export interface PromptProvenanceDraft {
  prompt: string;
  originalPrompt?: string | null;
}

/**
 * How prompt text reached the composer.
 *
 * - `typed`: the user edited the textarea by hand (or inserted a trigger word).
 * - `recalled`: a ↑/↓ prompt-history step replaced the whole prompt.
 */
export type PromptAuthoringSource = "typed" | "recalled";

/**
 * Whether an active quick transform (Expand/Remix at batch 1) keeps its
 * authority across an authoring event.
 *
 * A hand edit keeps it: the rewrite is still on screen, so stale-work recovery
 * can offer re-expand, generate anyway, or restore. A history recall is a
 * wholesale replacement — the prepared rewrite, its frozen route, its undo,
 * and the style it baked describe nothing the user is looking at any more —
 * so the transform is released outright instead of nagging about a prompt
 * the user deliberately walked away from.
 */
export function quickTransformSurvivesAuthoring(
  source: PromptAuthoringSource,
): boolean {
  return source === "typed";
}

/**
 * Apply prompt text authored directly by the user.
 *
 * `originalPrompt` describes the provenance of one concrete transformed
 * prompt. Once that transform has been submitted its route snapshot is
 * retired, so later typing starts a new print and must retire the old
 * provenance too. While a quick transform snapshot is still active, keep the
 * root so stale-work recovery can offer re-expand, generate anyway, or restore
 * — unless the authoring event itself released the transform (a history
 * recall), in which case nothing prepared survives to be described.
 */
export function applyAuthoredPrompt(
  draft: PromptProvenanceDraft,
  prompt: string,
  activeQuickTransform: boolean,
  source: PromptAuthoringSource = "typed",
): void {
  draft.prompt = prompt;
  if (!activeQuickTransform || !quickTransformSurvivesAuthoring(source)) {
    draft.originalPrompt = null;
  }
}
