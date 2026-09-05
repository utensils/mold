import { computed, type ComputedRef } from "vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useLastUsedStylesStore } from "@studio/stores/lastUsedStyles";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import { isModelRuntimeUnavailable } from "@studio/lib/modelRuntimeAvailability";
import type { GenerateForm } from "../lib/generateForm";
import { findInstalledModel } from "../lib/generateModels";
import { useGenerateFormStore } from "../stores/generateForm";
import { useStylePicker } from "./useStylePicker";
import {
  modelsForOutputKind,
  OUTPUT_KIND_LABEL,
  outputKindFor,
  type OutputKind,
} from "./useCreateOutputKind";

/**
 * The three doors of New image — Still picture, Short clip, 3-D object — as
 * ONE decision, whoever opens them. The view toolbar's segmented control
 * renders `outputOptions` and calls `setOutputKind`; the ⌘K palette's
 * "Make a short clip" and File ▸ New Clip raise the `shortClip` intent, which
 * the view answers with the same `setOutputKind("clip")`. Before this the
 * palette and the menu deep-linked `?output=sequence`, which put a person in
 * Scenes when the Short clip door opens onto Simple, and did nothing at all
 * when New image was already open, because the query is consumed on mount.
 *
 * Still picture ↔ Scenes is the inspector's swap (`setOutput`); Simple and
 * 3-D adopt a style HERE, because handing the inspector `single` restores a
 * PICTURE style, the opposite of what those doors ask for.
 */
export function useOutputKindDoor(
  form: () => GenerateForm,
  setOutput: (mode: "single" | "sequence") => void,
): {
  outputKind: ComputedRef<OutputKind>;
  outputOptions: ComputedRef<{ value: OutputKind; label: string }[]>;
  setOutputKind: (kind: string | number) => void;
} {
  const draft = useSequenceDraftStore();
  const lastUsed = useLastUsedStylesStore();
  const formStore = useGenerateFormStore();
  // ONE inventory for the toolbar and the inspector. The header partitioned
  // every style any machine has while the inspector partitioned the picker's
  // own target rows, so a 3-D door could open onto a style the machine Create
  // is aimed at cannot run.
  const { targetModels } = useStylePicker(form);

  const isSequence = computed(() => draft.output === "sequence");
  const isMesh = computed(() => isMeshFamily(form().family));
  // Which styles each section holds is `useCreateOutputKind`'s one answer —
  // the same one the picker narrows on, so the door and the menu behind it
  // agree.
  // `targetModels` keeps downloaded-but-unrunnable rows so the picker can
  // disclose them, disabled; a door never opens onto one of those, whether it
  // is the remembered style or the first in the section.
  const runnable = computed(() =>
    targetModels.value.filter((model) => !isModelRuntimeUnavailable(model)),
  );
  const meshModels = computed(() => modelsForOutputKind(runnable.value, "mesh"));
  const stillModels = computed(() => modelsForOutputKind(runnable.value, "still"));
  const clipModels = computed(() => modelsForOutputKind(runnable.value, "clip"));

  // The same decision the title bar reads (`useCreateOutputKind`), from this
  // form rather than the store so the header answers for the form it renders.
  const outputKind = computed<OutputKind>(() => outputKindFor(draft.output, form().family));
  // The words are `OUTPUT_KIND_LABEL`'s, shared with the Styles view's kind
  // filter, so a person learns the three kinds once.
  const outputOptions = computed(() => [
    { value: "still" as const, label: OUTPUT_KIND_LABEL.still },
    { value: "clip" as const, label: OUTPUT_KIND_LABEL.clip },
    // The 3-D door only exists where a 3-D style is installed; a style the
    // machine cannot run would be a dead end.
    ...(meshModels.value.length > 0 || isMesh.value
      ? [{ value: "mesh" as const, label: OUTPUT_KIND_LABEL.mesh }]
      : []),
  ]);

  /** The plain one-shot clip: the output stays `single` and the STYLE is
   *  what makes it a clip. */
  const isSimpleClip = computed(() => !isSequence.value && outputKind.value === "clip");

  /** Remember the picture style while another kind holds the form. Parked
   *  on the DRAFT: leaving New image unmounts the toolbar, and a
   *  component-local ref took the parked style with it. A kind that is
   *  already not Still picture holds no picture style worth parking. */
  function parkStillModel() {
    if (isMesh.value || isSimpleClip.value) return;
    draft.lastStillModel = form().model || null;
  }

  function setOutputKind(kind: string | number) {
    if (kind === outputKind.value) return;
    if (kind === "clip") {
      // Which of the two the door opens onto is the remembered sub-mode.
      // Scenes is the inspector's swap, exactly as before; Simple keeps the
      // one-shot output and adopts a clip style here, the way the 3-D door
      // adopts a 3-D one — the inspector's `setOutputMode("single")` would
      // restore a PICTURE style, which is the opposite of what Simple asks
      // for.
      if (draft.clipMode === "scenes") {
        setOutput("sequence");
        return;
      }
      // The clip style this section was last used with, else the first.
      const pick = lastUsed.pick("clip", clipModels.value);
      // No clip style on this machine: Scenes owns that empty state and says
      // where to get one, so the door still opens rather than doing nothing.
      if (!pick) {
        setOutput("sequence");
        return;
      }
      parkStillModel();
      formStore.applyModel(pick);
      return;
    }
    if (isSequence.value) setOutput("single");
    if (kind === "mesh") {
      const pick = lastUsed.pick("mesh", meshModels.value);
      if (!pick) return;
      parkStillModel();
      formStore.applyModel(pick);
      return;
    }
    if (isMesh.value || isSimpleClip.value) {
      // Whatever we restore has to be a style Still picture can make. The
      // old "anything that is not 3-D" fallback reached for the first row on
      // the machine, which on a box with a clip style installed put a video
      // style under a Still picture label. The parked style covers the
      // immediate round trip; the last-used memory covers every later visit
      // and the next launch.
      const restored =
        (draft.lastStillModel && findInstalledModel(stillModels.value, draft.lastStillModel)) ||
        lastUsed.pick("still", stillModels.value);
      if (restored) formStore.applyModel(restored);
      draft.lastStillModel = null;
    }
  }

  return { outputKind, outputOptions, setOutputKind };
}
