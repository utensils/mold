<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useLastUsedStylesStore } from "@studio/stores/lastUsedStyles";
import { validatePrintTitle } from "@studio/lib/libraryOrganization";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import type { GenerateForm } from "../../lib/generateForm";
import { findInstalledModel } from "../../lib/generateModels";
import {
  modelsForOutputKind,
  OUTPUT_KIND_LABEL,
  OUTPUT_KIND_PLACEHOLDER,
  outputKindFor,
  type OutputKind,
} from "../../composables/useCreateOutputKind";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useStylePicker } from "../../composables/useStylePicker";
import HostChip from "./HostChip.vue";
import type { InspectorTab } from "./inspectorTabs";

/**
 * The New image view toolbar (README §04): the editable print title, the
 * output kind as a segmented control — Still picture | Short clip | 3-D
 * object — the two inspector doors, Starting points and Use these settings
 * again, and last the Where it runs chip. Nothing floats over the canvas.
 *
 * Simple | Scenes — how a video gets made — is NOT on this row: it is
 * `ClipModeStrip`, the row beneath. Beside the kind control it pushed the
 * whole right-hand cluster left whenever Short clip was chosen, so the control a
 * person had just clicked jumped away from the pointer. This row holds the
 * same children in every kind, so nothing on it ever moves.
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
  /** Still picture ↔ Short clip: the inspector owns the model swap. */
  "set-output": [mode: "single" | "sequence"];
}>();

const draft = useSequenceDraftStore();
const lastUsed = useLastUsedStylesStore();
const formStore = useGenerateFormStore();
// ONE inventory for the toolbar and the inspector. The header partitioned
// every style any machine has while the inspector partitioned the picker's own
// target rows, so a 3-D door could open onto a style the machine Create is
// aimed at cannot run.
const { targetModels } = useStylePicker(() => props.form);

const isSequence = computed(() => draft.output === "sequence");
const isMesh = computed(() => isMeshFamily(props.form.family));
// Which styles each section holds is `useCreateOutputKind`'s one answer — the
// same one the picker narrows on, so the door and the menu behind it agree.
const meshModels = computed(() => modelsForOutputKind(targetModels.value, "mesh"));
const stillModels = computed(() => modelsForOutputKind(targetModels.value, "still"));
const clipModels = computed(() => modelsForOutputKind(targetModels.value, "clip"));

// The same decision the title bar reads (`useCreateOutputKind`), from this
// form rather than the store so the header answers for the form it renders.
const outputKind = computed<OutputKind>(() => outputKindFor(draft.output, props.form.family));
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

/** The plain one-shot clip: the output stays `single` and the STYLE is what
 *  makes it a clip. */
const isSimpleClip = computed(() => !isSequence.value && outputKind.value === "clip");

function setOutputKind(kind: string | number) {
  if (kind === outputKind.value) return;
  if (kind === "clip") {
    // Which of the two the door opens onto is the remembered sub-mode. Scenes
    // is the inspector's swap, exactly as before; Simple keeps the one-shot
    // output and adopts a clip style here, the way the 3-D door adopts a 3-D
    // one — the inspector's `setOutputMode("single")` would restore a
    // PICTURE style, which is the opposite of what Simple asks for.
    if (draft.clipMode === "scenes") {
      emit("set-output", "sequence");
      return;
    }
    // The clip style this section was last used with, else the first.
    const pick = lastUsed.pick("clip", clipModels.value);
    // No clip style on this machine: Scenes owns that empty state and says
    // where to get one, so the door still opens rather than doing nothing.
    if (!pick) {
      emit("set-output", "sequence");
      return;
    }
    parkStillModel();
    formStore.applyModel(pick);
    return;
  }
  if (isSequence.value) emit("set-output", "single");
  if (kind === "mesh") {
    const pick = lastUsed.pick("mesh", meshModels.value);
    if (!pick) return;
    parkStillModel();
    formStore.applyModel(pick);
    return;
  }
  if (isMesh.value || isSimpleClip.value) {
    // Whatever we restore has to be a style Still picture can make. The old
    // "anything that is not 3-D" fallback reached for the first row on the
    // machine, which on a box with a clip style installed put a video style
    // under a Still picture label.
    // The parked style covers the immediate round trip; the last-used memory
    // covers every later visit and the next launch.
    const restored =
      (draft.lastStillModel && findInstalledModel(stillModels.value, draft.lastStillModel)) ||
      lastUsed.pick("still", stillModels.value);
    if (restored) formStore.applyModel(restored);
    draft.lastStillModel = null;
  }
}

/** Remember the picture style while another kind holds the form. Parked on
 *  the DRAFT: leaving New image unmounts this toolbar, and a component-local
 *  ref took the parked style with it. A kind that is already not Still
 *  picture holds no picture style worth parking. */
function parkStillModel() {
  if (isMesh.value || isSimpleClip.value) return;
  draft.lastStillModel = props.form.model || null;
}

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
