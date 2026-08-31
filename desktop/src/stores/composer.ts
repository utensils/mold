import { defineStore } from "pinia";
import type { GeneratePrefill } from "../lib/generateForm";
import type { OutputMetadata } from "../lib/api/types";
import type { ApiTarget } from "@studio/api/client";
import type { RetainedSourceMediaInventory } from "@studio/api/gallerySourceMedia";

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
 * - `inspect` loads and watches an exact job without granting amend authority.
 * - `reuse` starts a FRESH draft from a print's recorded per-clip provenance
 *   (`metadata.chain`) — a new sequence, no edit session, nothing cached.
 *
 * Keeping both on one slot is deliberate: Create consumes exactly one
 * sequence intent per arrival, and two slots could disagree.
 */
export type SequenceHandoff =
  | { kind: "edit"; hostId: string; jobId: string }
  | { kind: "inspect"; hostId: string; jobId: string }
  | { kind: "reuse"; metadata: OutputMetadata };

export interface RetainedSourceReuseHandoff {
  /** Exact output archive identity; never a source path or storage id. */
  filename: string;
  origin: ApiTarget;
  inventory: RetainedSourceMediaInventory;
}

/** Carries "Reuse settings" from the gallery into the Generate composer. */
export const useComposerStore = defineStore("composer", {
  state: () => ({
    prefill: null as ComposerPrefill | null,
    /** One-shot: Create consumes it on arrival so a back-nav can't replay it. */
    pendingSequence: null as SequenceHandoff | null,
    retainedSource: null as RetainedSourceReuseHandoff | null,
    retainedSourceVersion: 0,
    pendingRetainedSourceApplyVersion: null as number | null,
  }),
  actions: {
    invalidateRetainedSource(): number {
      this.retainedSourceVersion += 1;
      this.retainedSource = null;
      this.pendingRetainedSourceApplyVersion = null;
      return this.retainedSourceVersion;
    },
    beginRetainedSourceReuse(prefill: ComposerPrefill): number {
      const version = this.invalidateRetainedSource();
      this.pendingRetainedSourceApplyVersion = version;
      this.prefill = prefill;
      return version;
    },
    takePendingRetainedSourceApplyVersion(): number | null {
      const version = this.pendingRetainedSourceApplyVersion;
      this.pendingRetainedSourceApplyVersion = null;
      return version;
    },
    setRetainedSourceIfCurrent(version: number, handoff: RetainedSourceReuseHandoff): boolean {
      if (version !== this.retainedSourceVersion) return false;
      this.retainedSource = handoff;
      return true;
    },
    isRetainedSourceCurrent(version: number): boolean {
      return version === this.retainedSourceVersion;
    },
    set(prefill: ComposerPrefill) {
      // Any new handoff supersedes the prior print's private-media authority.
      // Gallery Reuse settings attaches its exact inventory immediately after
      // setting the metadata prefill.
      this.invalidateRetainedSource();
      this.prefill = prefill;
    },
    setRetainedSource(handoff: RetainedSourceReuseHandoff | null) {
      this.retainedSource = handoff;
    },
    take(): ComposerPrefill | null {
      const p = this.prefill;
      this.prefill = null;
      return p;
    },
    setSequence(handoff: SequenceHandoff) {
      this.invalidateRetainedSource();
      this.pendingSequence = handoff;
    },
    takeSequence(): SequenceHandoff | null {
      const h = this.pendingSequence;
      this.pendingSequence = null;
      return h;
    },
  },
});
