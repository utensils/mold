<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { validatePrintTitle } from "@studio/lib/libraryOrganization";
import type { GenerateForm } from "../../lib/generateForm";
import { outputFamilyLabel } from "@studio/lib/outputShape";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import HostChip from "./HostChip.vue";

/**
 * Create header (Mold Studio): the editable print/sequence title, a live
 * summary pill, and the shared generation-host chip. Output (One shot |
 * Sequence) is a setting in the inspector, not a place — the old Single |
 * Sequence route switch is gone.
 *
 * The title is `form.title` (Library organization, D5): click to edit, Enter
 * or blur commits, Escape reverts; the value ships as `GenerateRequest.title`
 * on every print built from this form. An invalid title (control characters,
 * > 120 chars) keeps the editor open with the reason instead of committing.
 */
const props = defineProps<{ form: GenerateForm }>();

const draft = useSequenceDraftStore();
const isSequence = computed(() => draft.output === "sequence");

const placeholder = computed(() => (isSequence.value ? "Untitled sequence" : "Untitled print"));
const title = computed(() => props.form.title?.trim() ?? "");

const editing = ref(false);
const pendingTitle = ref("");
const titleError = ref<string | null>(null);
const inputRef = ref<HTMLInputElement | null>(null);

function startEdit() {
  pendingTitle.value = title.value;
  titleError.value = null;
  editing.value = true;
  void nextTick(() => {
    inputRef.value?.focus();
    inputRef.value?.select();
  });
}

function commitEdit() {
  if (!editing.value) return;
  const result = validatePrintTitle(pendingTitle.value);
  if (!result.ok) {
    titleError.value = result.reason;
    return;
  }
  props.form.title = result.value ?? "";
  titleError.value = null;
  editing.value = false;
}

function revertEdit() {
  titleError.value = null;
  editing.value = false;
}

function onBlur() {
  // A refused commit keeps the editor open; clicking away after that simply
  // discards the invalid text rather than trapping focus.
  if (!editing.value) return;
  const result = validatePrintTitle(pendingTitle.value);
  if (result.ok) commitEdit();
  else revertEdit();
}

const summary = computed(() => {
  const { width, height, steps } = props.form;
  if (isSequence.value) {
    const aspect = outputFamilyLabel(width, height);
    return `${aspect} · ${width}×${height} · ${draft.clips.length} clips · ${props.form.fps} fps`;
  }
  // A canvasless recipe (a 3-D mesh) has no aspect or pixel canvas — its
  // width/height default to 0×0, so the ordinary summary read nonsense like
  // "1:1 · 0×0 · 5 steps". Show its octree resolution instead, using the
  // same canvasless predicate the rest of the form relies on (recipe
  // snapshot first, family fallback for a form restored pre-snapshot).
  const canvasless = props.form.recipeCapabilities?.canvasless ?? isMeshFamily(props.form.family);
  if (canvasless) {
    const octree =
      props.form.mesh?.octreeResolution ?? props.form.recipeCapabilities?.mesh?.octree_default;
    return octree != null ? `octree ${octree} · ${steps} steps` : `${steps} steps`;
  }
  const aspect = outputFamilyLabel(width, height);
  return `${aspect} · ${width}×${height} · ${steps} steps`;
});
</script>

<template>
  <header data-test="create-header" class="ms-header">
    <div class="ms-header__title">
      <template v-if="editing">
        <input
          ref="inputRef"
          v-model="pendingTitle"
          data-test="print-title-input"
          data-selectable
          type="text"
          class="ms-header__title-input"
          :class="{ 'ms-header__title-input--invalid': titleError }"
          aria-label="Print title"
          :aria-invalid="titleError ? 'true' : undefined"
          :aria-describedby="titleError ? 'create-print-title-error' : undefined"
          :placeholder="placeholder"
          autocomplete="off"
          spellcheck="false"
          @keydown.enter.prevent="commitEdit"
          @keydown.esc.prevent="revertEdit"
          @blur="onBlur"
        />
        <span
          v-if="titleError"
          id="create-print-title-error"
          data-test="print-title-error"
          class="ms-header__title-error"
          role="alert"
          >{{ titleError }}</span
        >
      </template>
      <button
        v-else
        type="button"
        data-test="print-title"
        class="ms-header__title-button"
        :class="{ 'ms-header__title-button--placeholder': !title }"
        aria-label="Print title"
        :title="title ? `${title} — click to rename` : 'Click to name this print'"
        @click="startEdit"
      >
        <span class="ms-header__title-text">{{ title || placeholder }}</span>
        <Icon name="pencil" :size="13" class="ms-header__title-pencil" aria-hidden="true" />
      </button>
    </div>
    <span class="ms-header__summary data-mono">{{ summary }}</span>
    <div class="ms-header__spacer" />
    <HostChip />
  </header>
</template>

<style scoped>
.ms-header {
  height: 52px;
  flex: 0 0 52px;
  border-bottom: 1px solid var(--edge);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 22px;
}
.ms-header__title {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  font-family: var(--f-display);
  font-size: 15px;
  font-weight: 600;
}
.ms-header__title-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  max-width: 40vw;
  height: 30px;
  margin-left: -6px;
  padding: 0 6px;
  border-radius: var(--r-control, 6px);
  font: inherit;
  color: inherit;
  background: transparent;
  border: 1px solid transparent;
  cursor: text;
}
.ms-header__title-button:hover,
.ms-header__title-button:focus-visible {
  border-color: var(--edge);
  background: var(--bench, transparent);
}
.ms-header__title-button--placeholder .ms-header__title-text {
  color: var(--ink-3);
  font-weight: 500;
}
.ms-header__title-text {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-header__title-pencil {
  flex: 0 0 auto;
  color: var(--ink-3);
  opacity: 0;
  transition: opacity 120ms ease;
}
.ms-header__title-button:hover .ms-header__title-pencil,
.ms-header__title-button:focus-visible .ms-header__title-pencil {
  opacity: 1;
}
.ms-header__title-input {
  width: min(40vw, 420px);
  height: 30px;
  margin-left: -6px;
  padding: 0 6px;
  border-radius: var(--r-control, 6px);
  border: 1px solid var(--edge);
  background: var(--bench, transparent);
  color: var(--ink);
  font: inherit;
}
.ms-header__title-input:focus {
  outline: none;
  border-color: var(--safelight);
}
.ms-header__title-input--invalid,
.ms-header__title-input--invalid:focus {
  border-color: var(--stop);
}
.ms-header__title-error {
  font-family: var(--f-body, inherit);
  font-size: 11px;
  font-weight: 500;
  color: var(--stop);
  white-space: nowrap;
}
.ms-header__summary {
  font-size: 10px;
  color: var(--ink-3);
  padding: 3px 8px;
  border: 1px solid var(--edge);
  border-radius: 20px;
}
.ms-header__spacer {
  flex: 1;
}
</style>
