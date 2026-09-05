import { computed, type ComputedRef } from "vue";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import type { OutputMode } from "@studio/lib/sequence";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useGenerateFormStore } from "../stores/generateForm";

/**
 * The three-way output kind the New image view is in — the one decision
 * behind the toolbar's Still picture | Short clip | 3-D object control, the
 * title bar's "New image" / "New clip" / "New 3-D object", and the print
 * title's placeholder. A clip is the AUTHORED output kind and outranks the
 * style's family; 3-D is a property of the chosen style.
 */
export type OutputKind = "still" | "clip" | "mesh";

export function outputKindFor(output: OutputMode, family: string | null | undefined): OutputKind {
  if (output === "sequence") return "clip";
  return isMeshFamily(family) ? "mesh" : "still";
}

/** The view's mono title in the unified toolbar. */
export const OUTPUT_KIND_TITLE: Readonly<Record<OutputKind, string>> = {
  still: "New image",
  clip: "New clip",
  mesh: "New 3-D object",
};

/** The print title's placeholder before the user names the result. */
export const OUTPUT_KIND_PLACEHOLDER: Readonly<Record<OutputKind, string>> = {
  still: "Untitled picture",
  clip: "Untitled clip",
  mesh: "Untitled 3-D object",
};

/** The live output kind, read from the sequence draft and the form store. */
export function useCreateOutputKind(): ComputedRef<OutputKind> {
  const draft = useSequenceDraftStore();
  const generateForm = useGenerateFormStore();
  return computed(() => outputKindFor(draft.output, generateForm.form.family));
}
