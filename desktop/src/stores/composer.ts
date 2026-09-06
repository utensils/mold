import { defineStore } from "pinia";
import type { GeneratePrefill } from "../lib/generateForm";
import type { ApiTarget } from "@studio/api/client";
import type { RetainedSourceMediaInventory } from "@studio/api/gallerySourceMedia";

/**
 * Either the legacy scalar shape (CommandPalette, history, jobs, "Generate
 * with <model>") or `{ metadata }` carrying a gallery item's full embedded
 * metadata for full-fidelity "Reuse settings". `GenerateView` routes both
 * through `applyPrefillToForm`.
 */
export type ComposerPrefill = GeneratePrefill;

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
  },
});
