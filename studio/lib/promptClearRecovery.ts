import type {
  PromptAuthoringSource,
  PromptProvenanceDraft,
} from "./promptProvenance";

/** One composer's immediate Clear recovery, never persisted or shared. */
export class PromptClearRecovery {
  private receipt: { originalPrompt: string | null | undefined } | null = null;
  apply(
    draft: PromptProvenanceDraft,
    text: string,
    source: PromptAuthoringSource,
  ): boolean {
    if (source === "clear") {
      this.receipt = { originalPrompt: draft.originalPrompt };
      draft.prompt = text;
      draft.originalPrompt = null;
      return true;
    }
    if (source === "undo-clear" && this.receipt) {
      draft.prompt = text;
      if (this.receipt.originalPrompt === undefined)
        delete draft.originalPrompt;
      else draft.originalPrompt = this.receipt.originalPrompt;
      this.receipt = null;
      return true;
    }
    this.receipt = null;
    return false;
  }
}
