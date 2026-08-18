/** Minimal mutable shape shared by the web, desktop, and iPhone composers. */
export interface PromptProvenanceDraft {
  prompt: string;
  originalPrompt?: string | null;
}

/**
 * Apply prompt text authored directly by the user.
 *
 * `originalPrompt` describes the provenance of one concrete transformed
 * prompt. Once that transform has been submitted its route snapshot is
 * retired, so later typing starts a new print and must retire the old
 * provenance too. While a quick transform snapshot is still active, keep the
 * root so stale-work recovery can offer re-expand, generate anyway, or restore.
 */
export function applyAuthoredPrompt(
  draft: PromptProvenanceDraft,
  prompt: string,
  activeQuickTransform: boolean,
): void {
  draft.prompt = prompt;
  if (!activeQuickTransform) draft.originalPrompt = null;
}
