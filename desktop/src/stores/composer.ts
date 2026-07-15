import { defineStore } from "pinia";

export interface ComposerPrefill {
  prompt: string;
  model: string;
  seed: number | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  upscaleModel?: string;
}

/** Carries "Reuse settings" from the gallery into the Generate composer. */
export const useComposerStore = defineStore("composer", {
  state: () => ({ prefill: null as ComposerPrefill | null }),
  actions: {
    set(prefill: ComposerPrefill) {
      this.prefill = prefill;
    },
    take(): ComposerPrefill | null {
      const p = this.prefill;
      this.prefill = null;
      return p;
    },
  },
});
