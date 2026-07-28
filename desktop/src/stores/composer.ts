import { defineStore } from "pinia";
import type { GeneratePrefill } from "../lib/generateForm";
import type { OutputMetadata } from "../lib/api/types";

/**
 * Either the legacy scalar shape (CommandPalette, history, jobs, "Generate
 * with <model>") or `{ metadata }` carrying a gallery item's full embedded
 * metadata for full-fidelity "Reuse settings". `GenerateView` routes both
 * through `applyPrefillToForm`.
 */
export type ComposerPrefill = GeneratePrefill;

/**
 * A sequence handed to Create from somewhere else in the app.
 *
 * - `edit` re-enters a durable job in place, preserving its cached clips.
 * - `reuse` starts a FRESH draft from a print's recorded per-clip provenance
 *   (`metadata.chain`) — a new sequence, no edit session, nothing cached.
 *
 * Keeping both on one slot is deliberate: Create consumes exactly one
 * sequence intent per arrival, and two slots could disagree.
 */
export type SequenceHandoff =
  | { kind: "edit"; hostId: string; jobId: string }
  | { kind: "reuse"; metadata: OutputMetadata };

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
