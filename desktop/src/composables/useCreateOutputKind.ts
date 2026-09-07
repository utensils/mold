import { computed, type ComputedRef } from "vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { outputKindFor, type OutputKind } from "@studio/lib/outputKind";
export * from "@studio/lib/outputKind";

/** The live output kind, read from the form store. */
export function useCreateOutputKind(): ComputedRef<OutputKind> {
  const generateForm = useGenerateFormStore();
  return computed(() => outputKindFor(generateForm.form.family));
}
