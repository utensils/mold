import { defineStore } from "pinia";
import { emptyFileUnderState } from "@studio/lib/fileUnder";
import { applyModelDefaults, keepingPrintIdentity, newGenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";

/**
 * Holds the Generate workspace form outside any component so it survives
 * navigation — `GenerateView` unmounts on every route change and a
 * component-local `reactive()` would take the model, prompt, and params with
 * it. Pinia state is reactive, so the SSE/closure-mutation rule is satisfied.
 */
export const useGenerateFormStore = defineStore("generateForm", {
  state: () => ({ form: newGenerateForm() }),
  actions: {
    /** Pick a model, applying its defaults and pruning unsupported fields. */
    applyModel(m: ModelEntry) {
      applyModelDefaults(this.form, m);
    },
    /**
     * ⌘N — clear the composer for a fresh print, keeping the model and its
     * shape/params. Mirrors the fields the GenerateView ⌘N watcher cleared.
     */
    clearComposer() {
      this.form.prompt = "";
      this.form.originalPrompt = null;
      // A fresh print is untitled. This is the ONLY place the title is
      // cleared implicitly — generating keeps it so siblings share the name.
      this.form.title = "";
      // …and unfiled: "File under" is a per-print choice, and the ghost chip
      // it re-derives from the (now empty) title is nothing.
      this.form.fileUnder = emptyFileUnderState();
      this.form.fileUnderMatch = null;
      // A cleared composer starts from the model's advertised default
      // negative (wan), never the explicit empty opt-out.
      this.form.negativePrompt = this.form.negativePromptDefault;
      this.form.seed = "";
      this.form.sourceImage = null;
      this.form.sourceImageName = null;
      this.form.maskImage = null;
      this.form.imageAttachments = [];
    },
    /**
     * Reset every field to defaults. `Object.assign` preserves the form's
     * object identity — the view and child panels hold references to it.
     */
    resetAll() {
      keepingPrintIdentity(this.form, () => Object.assign(this.form, newGenerateForm()));
    },
  },
});
