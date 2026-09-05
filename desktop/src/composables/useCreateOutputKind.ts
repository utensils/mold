import { computed, type ComputedRef } from "vue";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import type { OutputMode } from "@studio/lib/sequence";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { isVideoFamily } from "../lib/capabilities";
import { useGenerateFormStore } from "../stores/generateForm";

/**
 * The three-way output kind the New image view is in — the one decision
 * behind the toolbar's Still picture | Short clip | 3-D object control, the
 * title bar's "New image" / "New clip" / "New 3-D object", and the print
 * title's placeholder. An authored sequence is always a clip; otherwise the
 * chosen style says which kind this is.
 */
export type OutputKind = "still" | "clip" | "mesh";

export function outputKindFor(output: OutputMode, family: string | null | undefined): OutputKind {
  // A sequence is a clip whatever style is loaded (a moment mid-swap can hold
  // a picture style). Everywhere else the STYLE decides, by the same partition
  // the picker sorts its rows with — a one-shot on a clip style IS a clip, the
  // Simple sub-mode, and calling that view "Still picture" is the mislabelling
  // this one authority exists to end.
  if (output === "sequence") return "clip";
  return outputKindForModel({ family: family ?? "" });
}

/**
 * The three kinds' names — ONE set of words wherever a kind is chosen or
 * filtered. The Create toolbar's Still picture | Short clip | 3-D object
 * control and the Styles view's kind filter both read it, so a person learns
 * the mapping once: what you pick in Create is what you filter by in Styles.
 */
export const OUTPUT_KIND_LABEL: Readonly<Record<OutputKind, string>> = {
  still: "Still picture",
  clip: "Short clip",
  mesh: "3-D object",
};

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

/** The least a row needs to be sorted into a section. */
export interface SectionModel {
  family: string;
}

/**
 * Which section a STYLE belongs to — the other half of the same decision, and
 * the reason it lives here rather than in the picker.
 *
 * It is a PARTITION: every installed style lands in exactly one section, so a
 * style can never be offered where it cannot deliver and, just as important,
 * can never become unreachable by belonging nowhere. The video test is the one
 * existing helper (`isVideoFamily` → the shared `supportsVideo`), never a
 * second family list.
 *
 * A clip style that cannot author a multi-scene sequence — MiniMax H3
 * advertises `supports_sequence: false` — still belongs to **clip**: what it
 * makes is a clip, and that is where a person looks for it. Sorting it into
 * Still picture is exactly the mislabelling this narrowing exists to end, and
 * dropping it from every section would hide a working style. Its inability to
 * chain is a refusal the picker spells out on the row itself.
 */
export function outputKindForModel(model: SectionModel): OutputKind {
  if (isMeshFamily(model.family)) return "mesh";
  return isVideoFamily(model.family) ? "clip" : "still";
}

/** Narrow an installed-style list to one section, in the order handed in. */
export function modelsForOutputKind<M extends SectionModel>(
  models: readonly M[],
  kind: OutputKind,
): M[] {
  return models.filter((model) => outputKindForModel(model) === kind);
}

/** The menu's mono kicker — what this section holds, in the binding lexicon. */
export const OUTPUT_KIND_SECTION_LABEL: Readonly<Record<OutputKind, string>> = {
  still: `${OUTPUT_KIND_LABEL.still.toLowerCase()} styles`,
  clip: `${OUTPUT_KIND_LABEL.clip.toLowerCase()} styles`,
  mesh: `${OUTPUT_KIND_LABEL.mesh} styles`,
};

/** The menu's sentence when the section holds nothing on any machine. */
export const OUTPUT_KIND_EMPTY: Readonly<Record<OutputKind, string>> = {
  still: `No ${OUTPUT_KIND_SECTION_LABEL.still} on this machine.`,
  clip: `No ${OUTPUT_KIND_SECTION_LABEL.clip} on this machine.`,
  mesh: `No ${OUTPUT_KIND_SECTION_LABEL.mesh} on this machine.`,
};

/**
 * Where **Browse more** goes from each section — the Styles view filtered to
 * the same kind. The values are `mediaTypeFromQuery`'s own (`image` / `video`
 * / `mesh`) and nothing else.
 */
export const OUTPUT_KIND_BROWSE_TARGET: Readonly<Record<OutputKind, string>> = {
  still: "/models?type=image",
  clip: "/models?type=video",
  mesh: "/models?type=mesh",
};

/** The live output kind, read from the sequence draft and the form store. */
export function useCreateOutputKind(): ComputedRef<OutputKind> {
  const draft = useSequenceDraftStore();
  const generateForm = useGenerateFormStore();
  return computed(() => outputKindFor(draft.output, generateForm.form.family));
}
