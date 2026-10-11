<script setup lang="ts">
import PromptEditor from "@studio/components/PromptEditor.vue";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { promptPlaceholder } from "@studio/lib/promptRequirement";
import { promptTransformBlockedReason } from "@studio/lib/promptTransform";
import {
  clampVideoFrames,
  formatVideoDuration,
  maxVideoFrames,
  minVideoFrames,
  videoFrameStep,
  type VideoFrameContract,
} from "@studio/lib/videoDuration";
import Icon from "@ui/components/Icon.vue";
import ActionBlocker from "@ui/components/ActionBlocker.vue";
import Stepper from "@ui/components/Stepper.vue";
import ExpandControl from "../generate/ExpandControl.vue";
import EstimateBadge from "@studio/components/EstimateBadge.vue";
import { estimateGeneration } from "../../lib/api/estimate";
import type { GenerateForm } from "../../lib/generateForm";
import { promptInputForForm } from "../../lib/promptRecipe";
import type { GenerateRequest } from "../../lib/api/types";
import type { ApiTarget } from "../../lib/api/client";
import { MAX_BATCH_SIZE } from "../../lib/generateForm";
import {
  PROMPT_HEIGHT_INITIAL,
  PROMPT_HEIGHT_MIN,
  PROMPT_HEIGHT_STEP,
  availablePromptHeight,
  draggedPromptHeight,
  loadPromptHeight,
  resolvePromptHeight,
  savePromptHeight,
} from "../../lib/promptEditorHeight";
import { PromptCycler, caretOnFirstLine, caretOnLastLine } from "@studio/lib/promptCycler";
import type { PromptAuthoringSource } from "@studio/lib/promptProvenance";
import { primaryModifierPressed, shortcutLabel } from "../../lib/platform";

/**
 * The composer (README §04): the prompt line carries the estimate on its
 * right; the control row below carries the chips — Style, Shape, Length,
 * Make N, Write more for me — then Generate. Style is the ONE style picker and is
 * supplied through the `style` slot, so this component stays presentational.
 * Advisory text never shares a row with controls. Owns the prompt textarea and its editing affordances — ⌘↵
 * generate, ⌘E expand, ↑/↓ shell-style prompt history, and its resize handle. The view
 * keeps all orchestration; this component only surfaces intent.
 */
const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    effectiveBatchSize: number;
    /** The recipe renders one at a time: the Make chip reads 1 and locks. */
    batchLocked?: boolean;
    expansionRunning: boolean;
    expansionHostLabel: string | null;
    canUndo: boolean;
    preparedBlocked: boolean;
    disabled: boolean;
    disabledReason: string | null;
    /** Non-blocking advisory (e.g. an off-profile custom size) shown when no
     * blocker is active. Generate stays enabled. */
    warningReason?: string | null;
    submitting: boolean;
    buttonLabel: string;
    /** Queue depth, said beside the one-word button rather than inside it. */
    queuedNote?: string | null;
    estimateRequest: GenerateRequest | null;
    estimateTarget: ApiTarget | null;
    preprocessingStatus: string | null;
    remixSource?: "original" | "current";
    /** Recent prompts for ↑/↓ history cycling. */
    history?: string[];
    /** The kind's own invitation — "Describe the clip". Null keeps the
     * recipe's own prompt placeholder. */
    placeholder?: string | null;
    /**
     * The clip's length, as the chip beside Shape. `lengthContract` absent
     * hides it — a still has no length. It is the same `form.frames` the
     * inspector's Clip card slider writes, so the two are one control shown
     * twice.
     */
    lengthFrames?: number | null;
    lengthFps?: number;
    lengthContract?: VideoFrameContract | null;
  }>(),
  {
    history: () => [],
    remixSource: "original",
    warningReason: null,
    batchLocked: false,
    placeholder: null,
    lengthFrames: null,
    lengthFps: 24,
    lengthContract: null,
  },
);

const emit = defineEmits<{
  generate: [];
  cancel: [];
  expand: [];
  remix: [];
  restore: [];
  /** Tagged with how the text arrived: a ↑/↓ recall replaces the whole
   * prompt and releases any quick expansion, where typing keeps it. */
  "prompt-authored": [value: string, source: PromptAuthoringSource];
  /** The Length chip's new frame count, already on the family's grid. */
  "update:lengthFrames": [frames: number];
  "update:remixSource": [value: "original" | "current"];
}>();

// Disabled state and corrective guidance are intentionally separate: obvious
// requirements such as an empty prompt do not need a persistent warning.
const generateDisabled = computed(() => props.disabled && !props.submitting);
// The recipe is the prompt rule's authority (a mesh family has no text
// encoder to feed), so the form's snapshot of it rides along — without it the
// pre-profile family rule would ask for a description nothing reads.
const invitation = computed(
  () =>
    props.placeholder ??
    promptPlaceholder(
      promptInputForForm(props.form),
      "Describe the picture you want — “a brass teapot on a rainy windowsill, evening light”",
    ),
);
/** The composer always carries the FORM's prompt: there is no second set of
 * words behind it any more, so its provenance, its quick rewrite and its
 * ↑/↓ history never stand down. */
const promptText = computed({
  get: () => props.form.prompt,
  set: (value: string) => {
    props.form.prompt = value;
  },
});
// Expand and Remix rewrite the prompt, so the same recipe snapshot answers
// for them: a family with no text encoder reads nothing a rewrite could say.
const transformBlockedReason = computed(() =>
  promptTransformBlockedReason(props.form.recipeCapabilities?.promptMode),
);

/*
 * Length — the clip as a chip. Every bound comes from the shared video-duration
 * authority, so the chip snaps to exactly the grid the request validator
 * enforces and the inspector's slider offers. An intentional above-ceiling
 * count (the auto-chained long clip) is shown as authored while the slider
 * itself stays inside the one-render range, which is `VideoDurationSlider`'s
 * own rule.
 */
const lengthRate = computed(() => Math.max(1, Math.round(props.lengthFps) || 24));
const lengthMin = computed(() => minVideoFrames(props.lengthContract));
const lengthMax = computed(() => maxVideoFrames(props.lengthContract, lengthRate.value));
const lengthStep = computed(() => videoFrameStep(props.lengthContract));
const lengthValue = computed(() =>
  clampVideoFrames(props.lengthFrames ?? lengthMin.value, lengthRate.value, props.lengthContract),
);
const lengthShown = computed(() =>
  (props.lengthFrames ?? 0) > lengthMax.value ? props.lengthFrames! : lengthValue.value,
);
const lengthReadout = computed(
  () => `${lengthShown.value}f · ${formatVideoDuration(lengthShown.value, lengthRate.value)}`,
);
const lengthFill = computed(() => {
  const span = lengthMax.value - lengthMin.value;
  if (span <= 0) return "100%";
  return `${Math.round(((lengthValue.value - lengthMin.value) / span) * 100)}%`;
});
function setLength(raw: string) {
  emit(
    "update:lengthFrames",
    clampVideoFrames(Number(raw), lengthRate.value, props.lengthContract),
  );
}

const promptEl = ref<HTMLTextAreaElement | null>(null);
const promptEditorOpen = ref(false);
watch(
  () => props.form.recipeCapabilities?.promptMode === "ignored",
  (ignored) => {
    if (ignored) promptEditorOpen.value = false;
  },
);
const expandControl = ref<InstanceType<typeof ExpandControl> | null>(null);
const cycler = new PromptCycler();

watch(
  () => props.history,
  (entries) => cycler.setEntries(entries ?? []),
  { immediate: true },
);

// The prompt opens three lines tall and the handle above it resizes it,
// growing upward from the bottom-anchored composer. The saved height is the
// person's; a short window only clamps what is drawn.
const preferredPromptHeight = ref(loadPromptHeight());
const viewportHeight = ref(window.innerHeight);
const promptRoom = computed(() => availablePromptHeight(viewportHeight.value));
const promptHeight = computed(() =>
  resolvePromptHeight(preferredPromptHeight.value, promptRoom.value),
);
function setPromptHeight(next: number) {
  preferredPromptHeight.value = resolvePromptHeight(next, promptRoom.value);
  savePromptHeight(preferredPromptHeight.value);
}
function onViewportResize() {
  viewportHeight.value = window.innerHeight;
}
let dragStart: { y: number; height: number } | null = null;
function onHandlePointerDown(event: PointerEvent) {
  if (event.button !== 0) return;
  event.preventDefault();
  dragStart = { y: event.clientY, height: promptHeight.value };
  (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
}
function onHandlePointerMove(event: PointerEvent) {
  if (!dragStart) return;
  setPromptHeight(
    draggedPromptHeight(dragStart.height, event.clientY - dragStart.y, promptRoom.value),
  );
}
function onHandlePointerUp(event: PointerEvent) {
  dragStart = null;
  (event.currentTarget as HTMLElement).releasePointerCapture?.(event.pointerId);
}
function onHandleKeydown(event: KeyboardEvent) {
  const delta =
    event.key === "ArrowUp"
      ? PROMPT_HEIGHT_STEP
      : event.key === "ArrowDown"
        ? -PROMPT_HEIGHT_STEP
        : 0;
  if (delta) {
    event.preventDefault();
    setPromptHeight(promptHeight.value + delta);
  } else if (event.key === "Home") {
    event.preventDefault();
    setPromptHeight(PROMPT_HEIGHT_INITIAL);
  }
}
onMounted(() => {
  window.addEventListener("resize", onViewportResize);
  promptEl.value?.focus();
});
onBeforeUnmount(() => window.removeEventListener("resize", onViewportResize));

function cycleHistory(direction: "prev" | "next"): boolean {
  const replacement = direction === "prev" ? cycler.prev(promptText.value) : cycler.next();
  if (replacement === null) return false;
  promptText.value = replacement;
  emit("prompt-authored", replacement, "recalled");
  void nextTick(() => {
    const el = promptEl.value;
    el?.setSelectionRange(el.value.length, el.value.length);
  });
  return true;
}

function onPromptInput(event: Event) {
  const value = (event.target as HTMLTextAreaElement).value;
  cycler.reset();
  promptText.value = value;
  emit("prompt-authored", value, "typed");
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && primaryModifierPressed(e)) {
    e.preventDefault();
    if (!generateDisabled.value) submitOrCancel();
  } else if ((e.key === "e" || e.key === "E") && primaryModifierPressed(e)) {
    e.preventDefault();
    // Blocked by the recipe: the view owns the refusal, so the shortcut still
    // reaches it and answers with the reason rather than dying silently here.
    if (transformBlockedReason.value) emit("expand");
    else expandControl.value?.expand();
  } else if (e.key === "ArrowUp" && promptEl.value && caretOnFirstLine(promptEl.value)) {
    if (cycleHistory("prev")) e.preventDefault();
  } else if (e.key === "ArrowDown" && promptEl.value && caretOnLastLine(promptEl.value)) {
    if (cycleHistory("next")) e.preventDefault();
  }
}

function submitOrCancel() {
  if (props.submitting) emit("cancel");
  else emit("generate");
}

function focus() {
  promptEl.value?.focus();
}
function expand() {
  expandControl.value?.expand();
}
/** Record a submitted prompt so ↑ recalls it next time. */
function record(prompt: string) {
  cycler.record(prompt);
}
defineExpose({ focus, expand, record, isEditingPrompt: () => promptEditorOpen.value });
</script>

<template>
  <div data-test="composer-card" class="ms-composer">
    <div class="ms-composer__card">
      <div
        role="separator"
        aria-orientation="horizontal"
        aria-label="Prompt height"
        :aria-valuemin="Math.min(PROMPT_HEIGHT_MIN, Math.round(promptRoom))"
        :aria-valuenow="Math.round(promptHeight)"
        :aria-valuemax="Math.round(promptRoom)"
        tabindex="0"
        title="Drag to make the prompt taller or shorter · double-click to reset"
        class="prompt-resize-handle"
        data-test="prompt-resize-handle"
        @pointerdown="onHandlePointerDown"
        @pointermove="onHandlePointerMove"
        @pointerup="onHandlePointerUp"
        @pointercancel="onHandlePointerUp"
        @dblclick="setPromptHeight(PROMPT_HEIGHT_INITIAL)"
        @keydown="onHandleKeydown"
      >
        <span class="prompt-resize-handle__grip" aria-hidden="true" />
      </div>
      <div class="ms-composer__prompt-row">
        <textarea
          ref="promptEl"
          :value="promptText"
          data-selectable
          aria-label="Prompt"
          :placeholder="invitation"
          class="ms-composer__input prompt-input"
          :style="{ height: `${promptHeight}px` }"
          @keydown="onKeydown"
          @input="onPromptInput"
        />
        <div class="prompt-side">
          <EstimateBadge
            class="ms-composer__estimate"
            :request="estimateRequest"
            :target="estimateTarget"
            :estimate="estimateGeneration"
          />
          <button
            v-if="form.recipeCapabilities?.promptMode !== 'ignored'"
            type="button"
            class="prompt-editor-opener"
            data-test="edit-prompt"
            title="Open the prompt in a full-window editor"
            @click="promptEditorOpen = true"
          >
            Edit prompt
          </button>
        </div>
      </div>
      <div class="ms-composer__controls">
        <!-- The Style chip IS the picker (StylePicker.vue, filled by the view);
             it opens its menu in place rather than being a door to a second
             selector in the inspector. Shape keeps its door. -->
        <slot name="style" />
        <!-- Shape is the same: its chip IS the picker (ShapeChip.vue, filled
             by the view), reading and writing the one output-shape resolver
             the inspector's rail reads. It used to be a door whose only
             handler set the inspector tab it was already on. -->
        <slot name="shape" />
        <span v-if="lengthContract" class="ms-chip ms-chip--slider" data-test="length-chip">
          <span>Length</span>
          <input
            type="range"
            class="ms-chip__slider"
            data-test="length-slider"
            :min="lengthMin"
            :max="lengthMax"
            :step="lengthStep"
            :value="lengthValue"
            :style="{ '--ms-chip-slider-fill': lengthFill }"
            aria-label="How long the clip is"
            :aria-valuetext="lengthReadout"
            @input="setLength(($event.target as HTMLInputElement).value)"
          />
          <span class="ms-chip__id" data-test="length-readout">{{ lengthReadout }}</span>
        </span>
        <span
          class="ms-chip ms-chip--stepper"
          data-test="batch-chip"
          :class="{ 'ms-chip--locked': batchLocked }"
        >
          <span>Make</span>
          <Stepper
            :model-value="batchLocked ? 1 : form.batchSize"
            :min="1"
            :max="batchLocked ? 1 : MAX_BATCH_SIZE"
            :editable="!batchLocked"
            compact
            label="How many to make"
            @update:model-value="form.batchSize = $event"
          />
        </span>
        <ExpandControl
          ref="expandControl"
          :prompt="form.prompt"
          :batch-size="effectiveBatchSize"
          :running="expansionRunning"
          :host-label="expansionHostLabel"
          :can-undo="canUndo"
          :blocked="preparedBlocked"
          :transform-blocked-reason="transformBlockedReason"
          :original-available="!!form.originalPrompt"
          :remix-source="remixSource"
          @expand="emit('expand')"
          @remix="emit('remix')"
          @update:remix-source="emit('update:remixSource', $event)"
          @restore="emit('restore')"
        />
        <span class="ms-composer__spacer" />
        <span
          v-if="preprocessingStatus"
          class="ms-composer__status"
          data-test="preprocessing-status"
          >{{ preprocessingStatus }}</span
        >
        <span v-if="queuedNote" class="ms-composer__queued" data-test="generate-queued-note">{{
          queuedNote
        }}</span>
        <button
          type="button"
          data-test="generate-button"
          class="ms-composer__generate"
          :disabled="generateDisabled"
          @click="submitOrCancel"
        >
          <Icon v-if="submitting" name="close" :size="14" />
          {{ buttonLabel }}
          <kbd class="ms-composer__key">{{ shortcutLabel("↩") }}</kbd>
        </button>
      </div>
    </div>
    <ActionBlocker v-if="disabledReason" class="ms-composer__blocker" :reason="disabledReason" />
    <ActionBlocker
      v-else-if="warningReason"
      class="ms-composer__blocker"
      variant="warn"
      :reason="warningReason"
    />
  </div>
  <PromptEditor
    :open="promptEditorOpen"
    :prompt="promptText"
    :history="history"
    @authored="
      (text, source) => {
        cycler.reset();
        promptText = text;
        emit('prompt-authored', text, source);
      }
    "
    @close="promptEditorOpen = false"
  >
    <template #tools
      ><button
        type="button"
        :disabled="
          !promptText.trim() || expansionRunning || preparedBlocked || !!transformBlockedReason
        "
        @click="expand()"
      >
        Expand</button
      ><button
        type="button"
        :disabled="
          !promptText.trim() || expansionRunning || preparedBlocked || !!transformBlockedReason
        "
        @click="emit('remix')"
      >
        Remix</button
      ><button v-if="canUndo" type="button" @click="emit('restore')">Undo rewrite</button></template
    >
  </PromptEditor>
</template>

<style scoped>
.ms-composer__estimate {
  flex-shrink: 0;
  white-space: nowrap;
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
.ms-chip--locked {
  color: var(--mold-text-dim);
  cursor: default;
}
.ms-chip__caret {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
/* The chip IS the track's frame: no label row, no ticks, the mono readout
 * beside it. The full control stays in the inspector's Clip card. */
.ms-chip--slider {
  cursor: default;
  gap: 8px;
}
/* Same track vocabulary as the kit's SliderRow — accent behind the thumb,
 * control ink ahead of it — so the chip and the inspector's slider read as
 * the one control they are. */
.ms-chip__slider {
  -webkit-appearance: none;
  appearance: none;
  width: 84px;
  height: 4px;
  margin: 0;
  padding: 0;
  background: var(--mold-border-control);
  background-image: linear-gradient(
    to right,
    var(--mold-blue) var(--ms-chip-slider-fill, 0%),
    transparent var(--ms-chip-slider-fill, 0%)
  );
  cursor: pointer;
}
.ms-chip__slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 12px;
  height: 12px;
  border-radius: var(--mold-radius-1);
  background: var(--mold-text);
  cursor: pointer;
}
.ms-chip__slider::-moz-range-thumb {
  width: 12px;
  height: 12px;
  border: 0;
  border-radius: var(--mold-radius-1);
  background: var(--mold-text);
  cursor: pointer;
}
.ms-chip__slider:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}
.ms-chip--stepper {
  cursor: default;
  gap: 2px;
  /* The compact Stepper's − button carries its own 22px hit box, so the
   * chip's right padding is only the 2px that keeps it off the border. */
  padding-right: 2px;
}
.ms-composer__status {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-composer__queued {
  flex-shrink: 0;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  white-space: nowrap;
}
.ms-composer__blocker {
  margin-top: 8px;
}
</style>

<style scoped>
.prompt-resize-handle {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 12px;
  margin-bottom: -6px;
  cursor: ns-resize;
  touch-action: none;
  border-radius: var(--mold-radius-2) var(--mold-radius-2) 0 0;
}
.prompt-resize-handle__grip {
  width: 38px;
  height: 4px;
  border-radius: 999px; /* literal: pill */
  background: var(--mold-border);
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}
.prompt-resize-handle:hover .prompt-resize-handle__grip,
.prompt-resize-handle:focus-visible .prompt-resize-handle__grip {
  background: var(--mold-text-dim);
}
.prompt-resize-handle:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: -2px;
}
/* A multi-line prompt reads from the top; the estimate and the editor door
   stay pinned to its first line rather than floating to a baseline. */
.ms-composer__prompt-row {
  align-items: flex-start;
}
.prompt-input {
  resize: none;
  overflow-y: auto;
}
.prompt-side {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
  flex: none;
}
.prompt-editor-opener {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  min-height: 26px;
  padding: 3px 8px;
  font: inherit;
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
  background: transparent;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  cursor: pointer;
}
.prompt-editor-opener:hover {
  color: var(--mold-text);
  background: var(--mold-border-control);
}
.prompt-editor-opener:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}
@media (max-width: 600px) {
  .prompt-editor-opener {
    min-height: 44px;
  }
}
</style>
