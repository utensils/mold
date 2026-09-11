import { computed, ref, watch, type Ref } from "vue";
import { useLastUsedStylesStore } from "@studio/stores/lastUsedStyles";
import {
  modelsForOutputKind,
  outputKindFor,
  outputKindForModel,
  OUTPUT_KIND_LABEL,
  OUTPUT_KIND_MISSING,
  OUTPUT_KIND_TITLE,
  OUTPUT_KIND_BROWSE_TARGET,
  type OutputKind,
} from "@studio/lib/outputKind";
export * from "@studio/lib/outputKind";
import { isModelRuntimeUnavailable } from "@studio/lib/modelRuntimeAvailability";
import type { ModelInfoExtended } from "../types";
import type { UseGenerateForm } from "./useGenerateForm";

/**
 * Three doors into one form. Media parking and request policy stay in the form.
 *
 * The three kinds' vocabulary — the labels, the titles, the partition, the
 * Browse-more targets and the "nothing to open onto" sentence — is
 * `@studio/lib/outputKind`'s and is re-exported above, the same shape desktop's
 * shim has. Nothing here re-derives a word or a partition; what is left is the
 * web form's own wiring.
 */
export function useCreateOutputKind(
  form: UseGenerateForm,
  models: Ref<ModelInfoExtended[]>,
  family: Ref<string>,
) {
  const memory = useLastUsedStylesStore();
  const kind = computed(() => outputKindFor(family.value));
  const options = (["still", "clip", "mesh"] as const).map((value) => ({
    value,
    label: OUTPUT_KIND_LABEL[value],
  }));
  const notice = ref("");
  const requestedKind = ref<OutputKind | null>(null);
  const title = computed(() => OUTPUT_KIND_TITLE[kind.value]);
  const pickerModels = computed(() =>
    modelsForOutputKind(models.value, kind.value),
  );
  const browseTo = computed(
    () => OUTPUT_KIND_BROWSE_TARGET[requestedKind.value ?? kind.value],
  );
  const runnable = computed(() =>
    models.value.filter((model) => !isModelRuntimeUnavailable(model)),
  );
  let fallbackName: string | null = null;

  function selectStyle(model: ModelInfoExtended) {
    if (isModelRuntimeUnavailable(model)) return;
    fallbackName = null;
    notice.value = "";
    requestedKind.value = null;
    form.applyModelDefaults(model);
  }
  function selectKind(value: string | number) {
    if (value !== "still" && value !== "clip" && value !== "mesh") return;
    if (value === kind.value) {
      requestedKind.value = null;
      notice.value = "";
      return;
    }
    const selected = memory.pick(
      value,
      modelsForOutputKind(runnable.value, value),
    );
    if (!selected) {
      requestedKind.value = value;
      notice.value = OUTPUT_KIND_MISSING[value];
      return;
    }
    selectStyle(selected);
    const remembered = memory.bySection[value];
    fallbackName =
      remembered && remembered !== selected.name ? selected.name : null;
  }
  function initialStyle(): ModelInfoExtended | null {
    const section = memory.lastSection;
    const selected =
      (section
        ? memory.pick(section, modelsForOutputKind(runnable.value, section))
        : null) ??
      runnable.value[0] ??
      null;
    const remembered = section ? memory.bySection[section] : null;
    fallbackName =
      selected && remembered && remembered !== selected.name
        ? selected.name
        : null;
    return selected;
  }
  watch(
    () => [form.state.value.model, models.value] as const,
    ([name, entries]) => {
      const model = entries.find((entry) => entry.name === name);
      if (model && name !== fallbackName)
        memory.remember(outputKindForModel(model), name);
    },
  );
  return {
    kind,
    options,
    title,
    pickerModels,
    browseTo,
    notice,
    selectKind,
    selectStyle,
    initialStyle,
  };
}
