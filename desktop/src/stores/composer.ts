import { defineStore } from "pinia";
import type { GeneratePrefill } from "../lib/generateForm";

/**
 * Either the legacy scalar shape (CommandPalette, history, jobs, "Generate
 * with <model>") or `{ metadata }` carrying a gallery item's full embedded
 * metadata for full-fidelity "Reuse settings". `GenerateView` routes both
 * through `applyPrefillToForm`.
 */
export type ComposerPrefill = GeneratePrefill;

/**
 * A sequence handed to Create from somewhere else in the app. `edit` re-enters
 * a durable job in place (its cached clips are preserved); the Library's
 * settings-reuse variant lands on this same slot in a follow-up.
 */
export type SequenceHandoff = { kind: "edit"; hostId: string; jobId: string };

/** Carries "Reuse settings" from the gallery into the Generate composer. */
export const useComposerStore = defineStore("composer", {
  state: () => ({
    prefill: null as ComposerPrefill | null,
    /** One-shot: Create consumes it on arrival so a back-nav can't replay it. */
    pendingSequence: null as SequenceHandoff | null,
  }),
  actions: {
    set(prefill: ComposerPrefill) {
      this.prefill = prefill;
    },
    take(): ComposerPrefill | null {
      const p = this.prefill;
      this.prefill = null;
      return p;
    },
    setSequence(handoff: SequenceHandoff) {
      this.pendingSequence = handoff;
    },
    takeSequence(): SequenceHandoff | null {
      const h = this.pendingSequence;
      this.pendingSequence = null;
      return h;
    },
  },
});
