<script setup lang="ts">
import { computed } from "vue";
import Icon from "@ui/components/Icon.vue";
import ModelPicker from "./ModelPicker.vue";
import { useStylePicker } from "../../composables/useStylePicker";
import {
  OUTPUT_KIND_BROWSE_TARGET,
  OUTPUT_KIND_EMPTY,
  OUTPUT_KIND_SECTION_LABEL,
} from "../../composables/useCreateOutputKind";
import { useGenerateFormStore } from "../../stores/generateForm";
import { familyLabel } from "@studio/lib/modelFamily";
import { modelDisplayName, modelDisplayNameForId } from "../../lib/models";
import type { GenerateForm } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";

/**
 * The composer's Style chip AND the menu it opens — the ONE style selector on
 * Create (docs/design/mold-studio-desktop.dc.html, the composer control row:
 * layers glyph · plain name · mono id · ▼). The inspector's Settings tab has
 * no style field; two selectors is what made one of them look broken.
 *
 * The menu opens UPWARD because the composer sits on the bottom edge of the
 * canvas.
 */
const props = defineProps<{ form: GenerateForm }>();
const emit = defineEmits<{ "pull-missing-model": [model: string] }>();

const formStore = useGenerateFormStore();
const picker = useStylePicker(() => props.form);

/**
 * The menu shows ONE section of the view's three, so it says which one and
 * sends Browse more to the Styles view filtered the same way. All three come
 * from `useCreateOutputKind`, the one authority for which styles a section
 * holds — the picker never re-derives it from a family name.
 */
const sectionLabel = computed(() => OUTPUT_KIND_SECTION_LABEL[picker.outputKind.value]);
const sectionEmpty = computed(() => OUTPUT_KIND_EMPTY[picker.outputKind.value]);
const browseTarget = computed(() => OUTPUT_KIND_BROWSE_TARGET[picker.outputKind.value]);

/** Plain name first. A bare manifest id is not a plain word, so the family's
 *  friendly label stands in for it and the id still rides beside in mono. */
const styleLabel = computed(() => {
  const model = picker.selectedPickerModel.value;
  if (model) {
    const name = modelDisplayName(model);
    return name === model.name ? familyLabel(model.family) : name;
  }
  const missing = picker.missingModelId.value;
  if (!missing) return "";
  const name = modelDisplayNameForId(missing, picker.pickerModels.value);
  return name === missing ? familyLabel(props.form.family) : name;
});

const styleId = computed(
  () => picker.selectedPickerModel.value?.name ?? picker.missingModelId.value ?? "",
);

function pickModel(model: ModelEntry) {
  formStore.applyModel(model);
}
</script>

<template>
  <ModelPicker
    class="ms-style"
    placement="up"
    :models="picker.pickerModels.value"
    :selected="picker.selectedPickerModel.value"
    :missing-model="picker.missingModelId.value"
    :disabled-reason="picker.pickerDisabledReason"
    :show-availability="!picker.stickyTarget.value || picker.stickyTarget.value === 'capable'"
    :kicker="sectionLabel"
    :empty-label="sectionEmpty"
    :browse-target="browseTarget"
    @pick="pickModel"
    @pick-missing="emit('pull-missing-model', $event)"
  >
    <template #trigger="{ open, toggle }">
      <button
        type="button"
        data-test="style-chip"
        class="ms-chip ms-chip--style"
        :aria-expanded="open"
        aria-haspopup="listbox"
        title="Style"
        @click="toggle"
      >
        <Icon name="layers" :size="13" />
        <span data-test="selected-model-name" class="ms-chip__label">{{
          styleLabel || "Choose a style"
        }}</span>
        <span v-if="styleId && styleId !== styleLabel" class="ms-chip__id">{{ styleId }}</span>
        <!-- A real fact the user needs before Generate, beside the chip. -->
        <span
          v-if="picker.missingModelId.value"
          data-test="style-not-installed"
          class="ms-chip__tag"
          >Not on this machine</span
        >
        <span
          v-else-if="picker.stickyHostMissingModel.value"
          data-test="style-will-download"
          class="ms-chip__tag"
          >Not on {{ picker.stickyHostMissingModel.value }} — will download there</span
        >
        <span class="ms-chip__caret">▼</span>
      </button>
    </template>
  </ModelPicker>
</template>

<style scoped>
/* The chip lives in the composer's control row, but the picker owns it now,
 * so its look is defined here rather than in ComposerCard's scoped sheet —
 * slot content is compiled in the parent and never inherits the child's. */
.ms-style {
  flex-shrink: 0;
}
.ms-chip {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 10px;
  flex-shrink: 0;
  white-space: nowrap;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-2);
  cursor: pointer;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-chip:hover {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}
.ms-chip--style {
  background: var(--mold-surface);
  color: var(--mold-text);
  font-weight: 500;
}
.ms-chip__label {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ms-chip__id {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
.ms-chip__tag {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-warning);
}
.ms-chip__caret {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
</style>
