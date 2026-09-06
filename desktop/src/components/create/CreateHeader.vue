<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import { validatePrintTitle } from "@studio/lib/libraryOrganization";
import type { GenerateForm } from "../../lib/generateForm";
import { OUTPUT_KIND_PLACEHOLDER } from "../../composables/useCreateOutputKind";
import { useOutputKindDoor } from "../../composables/useOutputKindDoor";
import HostChip from "./HostChip.vue";
import type { InspectorTab } from "./inspectorTabs";

/**
 * The New image view toolbar (README §04): the editable print title, the
 * output kind as a segmented control — Still picture | Short clip | 3-D
 * object — the two inspector doors, Starting points and Use these settings
 * again, and last the Where it runs chip. Nothing floats over the canvas.
 *
 * There are exactly three kinds, and a clip has ONE way of being made — the
 * retired Simple | Scenes strip is gone with the timeline it opened. This row
 * holds the same children in every kind, so nothing on it ever moves.
 *
 * Where it runs is chrome, not a setting: at the foot of the inspector's
 * Settings list nobody found it, and which machine a print goes to is
 * something to know at a glance, so the chip stays on screen in every tab
 * and every output kind.
 *
 * The title is `form.title` (Library organization, D5): click to edit, Enter
 * or blur commits, Escape reverts; the value ships as `GenerateRequest.title`
 * on every print built from this form. An invalid title (control characters,
 * > 120 chars) keeps the editor open with the reason instead of committing.
 */
const props = defineProps<{ form: GenerateForm }>();
const emit = defineEmits<{
  "open-tab": [tab: InspectorTab];
}>();

// The three doors are ONE decision (`useOutputKindDoor`), shared with the
// palette's Make a short clip and File ▸ New Clip, which open the Short clip
// door through the view rather than deep-linking anything.
const { outputKind, outputOptions, setOutputKind } = useOutputKindDoor(() => props.form);

const placeholder = computed(() => OUTPUT_KIND_PLACEHOLDER[outputKind.value]);
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
        :title="title ? `${title} — click to rename` : 'Click to name this picture'"
        @click="startEdit"
      >
        <span class="ms-header__title-text">{{ title || placeholder }}</span>
        <Icon name="pencil" :size="13" class="ms-header__title-pencil" aria-hidden="true" />
      </button>
    </div>

    <div class="ms-header__spacer" />

    <SegmentedControl
      data-test="output-kind"
      class="ms-header__seg"
      :model-value="outputKind"
      :options="outputOptions"
      variant="neutral"
      compact
      label="What to make"
      @update:model-value="setOutputKind"
    />

    <span class="ms-header__divider" aria-hidden="true" />

    <button
      type="button"
      data-test="open-starters"
      class="ms-header__door"
      title="Starting points"
      aria-label="Starting points"
      @click="emit('open-tab', 'starters')"
    >
      <Icon name="grid" :size="14" />
      <span class="ms-header__door-label">Starting points</span>
    </button>
    <button
      type="button"
      data-test="open-recent"
      class="ms-header__door"
      title="Use these settings again"
      aria-label="Use these settings again"
      @click="emit('open-tab', 'recent')"
    >
      <Icon name="reuse" :size="14" />
      <span class="ms-header__door-label">Use these settings again</span>
    </button>

    <span class="ms-header__divider" aria-hidden="true" />

    <HostChip />
  </header>
</template>

<style scoped>
/* The toolbar is a size container: what it can fit is the window minus the
 * sidebar and a user-dragged inspector, so a viewport media query would be
 * measuring the wrong box. One row at every width, yielding in this order —
 * the title truncates, then the doors drop to icons; the segments never
 * wrap. */
.ms-header {
  height: var(--mold-shell-viewbar-h);
  flex: 0 0 var(--mold-shell-viewbar-h);
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 14px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-chrome);
  container-type: inline-size;
  container-name: create-header;
}
.ms-header__title {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  font-family: var(--mold-font-sans);
  font-size: var(--mold-fs-sm);
  font-weight: 600;
  color: var(--mold-text);
}
.ms-header__title-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  max-width: 34vw;
  height: 26px;
  margin-left: -6px;
  padding: 0 6px;
  border-radius: var(--mold-radius-2);
  font: inherit;
  color: inherit;
  background: transparent;
  border: var(--mold-bw) solid transparent;
  cursor: text;
}
.ms-header__title-button:hover,
.ms-header__title-button:focus-visible {
  border-color: var(--mold-border);
}
.ms-header__title-button--placeholder .ms-header__title-text {
  color: var(--mold-text-dim);
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
  color: var(--mold-text-faint);
}
.ms-header__title-input {
  width: min(34vw, 420px);
  height: 26px;
  margin-left: -6px;
  padding: 0 6px;
  border-radius: var(--mold-radius-2);
  border: var(--mold-bw) solid var(--mold-border-control);
  background: var(--mold-bg);
  color: var(--mold-text);
  font: inherit;
}
.ms-header__title-input:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}
.ms-header__title-input--invalid,
.ms-header__title-input--invalid:focus {
  border-color: var(--mold-error);
}
.ms-header__title-error {
  font-size: var(--mold-fs-micro);
  font-weight: 500;
  color: var(--mold-error);
  white-space: nowrap;
}
.ms-header__spacer {
  flex: 1 1 0;
  min-width: 0;
}
/* Never shrink: its segments are nowrap, so shrinking it only overflows. */
.ms-header__seg {
  flex: 0 0 auto;
}
.ms-header__divider {
  flex: 0 0 auto;
  width: var(--mold-bw);
  height: 20px;
  background: var(--mold-border);
}
.ms-header__door {
  display: inline-flex;
  cursor: pointer;
  align-items: center;
  gap: 6px;
  height: var(--mold-ctl-md);
  padding: 0 10px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-xs);
  font-weight: 500;
  white-space: nowrap;
  color: var(--mold-text-2);
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-header__door {
  flex: 0 0 auto;
}
.ms-header__door:hover {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}

/* The last thing to yield. Both doors keep their icon, their tooltip and
 * their accessible name; only the visible words go. */
@container create-header (max-width: 620px) {
  .ms-header__door-label {
    display: none;
  }
  .ms-header__door {
    padding: 0 8px;
  }
}
</style>
