/**
 * ↑/↓ prompt-history cycling for the Generate composer, shell-history style:
 * ↑ walks from the newest recorded prompt toward older ones, ↓ walks back
 * and finally restores the draft the user was typing. Pure state machine —
 * the view feeds it entries from GET /api/history and caret position checks.
 */
export class PromptCycler {
  /** Newest first, matching the /api/history wire order. */
  private entries: string[] = [];
  /** Index into entries while navigating; null = not navigating. */
  private cursor: number | null = null;
  /** What the user had typed before navigation started. */
  private draft = "";

  setEntries(prompts: string[]) {
    this.entries = prompts;
    // Entries changing mid-navigation (a refetch) would make the cursor
    // point at a different prompt than the one on screen; start over.
    this.cursor = null;
  }

  get navigating(): boolean {
    return this.cursor !== null;
  }

  /** User edited the prompt by hand — abandon navigation. */
  reset() {
    this.cursor = null;
  }

  /** ↑ — older. Returns the prompt to show, or null when there is none. */
  prev(current: string): string | null {
    if (this.entries.length === 0) return null;
    if (this.cursor === null) {
      this.draft = current;
      this.cursor = 0;
    } else if (this.cursor + 1 < this.entries.length) {
      this.cursor += 1;
    } else {
      return null; // already at the oldest — stay put
    }
    return this.entries[this.cursor] ?? null;
  }

  /** ↓ — newer; past the newest restores the draft. Null when not navigating. */
  next(): string | null {
    if (this.cursor === null) return null;
    if (this.cursor === 0) {
      this.cursor = null;
      return this.draft;
    }
    this.cursor -= 1;
    return this.entries[this.cursor] ?? null;
  }
}

/** True when the caret sits on the first line of the textarea. */
export function caretOnFirstLine(el: HTMLTextAreaElement): boolean {
  return !el.value.slice(0, el.selectionStart ?? 0).includes("\n");
}

/** True when the caret sits on the last line of the textarea. */
export function caretOnLastLine(el: HTMLTextAreaElement): boolean {
  return !el.value.slice(el.selectionEnd ?? el.value.length).includes("\n");
}
