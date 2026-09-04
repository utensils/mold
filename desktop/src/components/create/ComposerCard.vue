<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { promptPlaceholder } from "@studio/lib/promptRequirement";
import { promptTransformBlockedReason } from "@studio/lib/promptTransform";
import { outputFamilyLabel } from "@studio/lib/outputShape";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import Icon from "@ui/components/Icon.vue";
import ActionBlocker from "@ui/components/ActionBlocker.vue";
import Stepper from "@ui/components/Stepper.vue";
import ExpandControl from "../generate/ExpandControl.vue";
import EstimateBadge from "../generate/EstimateBadge.vue";
import type { GenerateForm } from "../../lib/generateForm";
import { promptInputForForm } from "../../lib/promptRecipe";
import type { GenerateRequest } from "../../lib/api/types";
import type { ApiTarget } from "../../lib/api/client";
import { MAX_BATCH_SIZE } from "../../lib/generateForm";
import { autoGrowRows } from "../../lib/autogrow";
import { PromptCycler, caretOnFirstLine, caretOnLastLine } from "@studio/lib/promptCycler";
import type { PromptAuthoringSource } from "@studio/lib/promptProvenance";
import { primaryModifierPressed, shortcutLabel } from "../../lib/platform";

/**
 * The composer (README §04): the prompt line carries the estimate on its
 * right; the control row below carries the chips — Style, Shape, Make N,
 * Write more for me — then Generate. Style is the ONE style picker and is
 * supplied through the `style` slot, so this component stays presentational.
 * Advisory text never shares a row with controls. Owns the prompt textarea and its editing affordances — ⌘↵
 * generate, ⌘E expand, ↑/↓ shell-style prompt history, autogrow. The view
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
    /** Clip mode: the composer writes the SELECTED SCENE's words rather than
     * the form's prompt. Null keeps the one-shot binding. */
    promptValue?: string | null;
    /** Clip mode's own invitation — "Scene 2 — describe what happens next". */
    placeholder?: string | null;
    /** Stands in for the Make stepper where the recipe makes exactly one
     * thing: "Make 1 clip". */
    countLabel?: string | null;
    /** Rewriting reaches the one-shot prompt only, so a scene has no expander. */
    showExpand?: boolean;
  }>(),
  {
    history: () => [],
    remixSource: "original",
    warningReason: null,
    batchLocked: false,
    promptValue: null,
    placeholder: null,
    countLabel: null,
    showExpand: true,
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
  /** Clip mode: the selected scene's new words. */
  "update:promptValue": [value: string];
  "update:remixSource": [value: "original" | "current"];
  /** The Shape chip is a door to the inspector's Settings tab. Style is not:
   *  its chip opens the picker itself, in the `style` slot. */
  "open-shape": [];
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
/** One writable prompt whichever words the composer is carrying. */
const promptText = computed({
  get: () => props.promptValue ?? props.form.prompt,
  set: (value: string) => {
    if (props.promptValue === null) props.form.prompt = value;
    else emit("update:promptValue", value);
  },
});
// Expand and Remix rewrite the prompt, so the same recipe snapshot answers
// for them: a family with no text encoder reads nothing a rewrite could say.
const transformBlockedReason = computed(() =>
  promptTransformBlockedReason(props.form.recipeCapabilities?.promptMode),
);

/** "Square · 1024" — the canvas as a chip; a 3-D style has no canvas. */
const shapeLabel = computed(() => {
  const canvasless = props.form.recipeCapabilities?.canvasless ?? isMeshFamily(props.form.family);
  if (canvasless) return null;
  const { width, height } = props.form;
  const family = outputFamilyLabel(width, height);
  const size = width === height ? `${width}` : `${width}×${height}`;
  return `${family === "1:1" ? "Square" : family} · ${size}`;
});

const promptEl = ref<HTMLTextAreaElement | null>(null);
const expandControl = ref<InstanceType<typeof ExpandControl> | null>(null);
const cycler = new PromptCycler();

watch(
  () => props.history,
  (entries) => cycler.setEntries(entries ?? []),
  { immediate: true },
);

function growPrompt() {
  if (promptEl.value) autoGrowRows(promptEl.value);
}
watch(promptText, () => void nextTick(growPrompt), { flush: "post" });
onMounted(() => {
  growPrompt();
  promptEl.value?.focus();
});

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
  growPrompt();
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
defineExpose({ focus, expand, record });
</script>

<template>
  <div data-test="composer-card" class="ms-composer">
    <div class="ms-composer__card">
      <div class="ms-composer__prompt-row">
        <textarea
          ref="promptEl"
          :value="promptText"
          data-selectable
          rows="1"
          aria-label="Prompt"
          :placeholder="invitation"
          class="ms-composer__input"
          @keydown="onKeydown"
          @input="onPromptInput"
        />
        <EstimateBadge
          class="ms-composer__estimate"
          :request="estimateRequest"
          :target="estimateTarget"
        />
      </div>
      <div class="ms-composer__controls">
        <!-- The Style chip IS the picker (StylePicker.vue, filled by the view);
             it opens its menu in place rather than being a door to a second
             selector in the inspector. Shape keeps its door. -->
        <slot name="style" />
        <button
          v-if="shapeLabel"
          type="button"
          data-test="shape-chip"
          class="ms-chip"
          title="Shape and size"
          @click="emit('open-shape')"
        >
          {{ shapeLabel }} <span class="ms-chip__caret">▼</span>
        </button>
        <span v-if="countLabel" class="ms-chip ms-chip--locked" data-test="batch-chip">{{
          countLabel
        }}</span>
        <span
          v-else
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
            label="How many to make"
            @update:model-value="form.batchSize = $event"
          />
        </span>
        <ExpandControl
          v-if="showExpand"
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
</template>

<style scoped>
.ms-composer {
  display: flex;
  flex-direction: column;
  border-top: var(--mold-bw) solid var(--mold-border);
  padding: 12px 14px 14px;
  background: var(--mold-bg);
}
.ms-composer__card {
  display: flex;
  flex-direction: column;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  transition: border-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-composer__card:focus-within {
  border-color: var(--mold-border-focus);
}
.ms-composer__prompt-row {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 12px 14px 6px;
}
.ms-composer__input {
  flex: 1;
  min-width: 0;
  width: 100%;
  box-sizing: border-box;
  border: 0;
  background: transparent;
  color: var(--mold-text);
  font-family: var(--mold-font-sans);
  font-size: var(--mold-fs-base);
  line-height: var(--mold-lh-body);
  resize: none;
  outline: none;
  min-height: 24px;
  max-height: 160px;
  overflow-x: hidden;
}
.ms-composer__input::placeholder {
  color: var(--mold-text-dim);
}
.ms-composer__estimate {
  flex-shrink: 0;
  white-space: nowrap;
}
.ms-composer__controls {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 8px 10px 10px 12px;
}
.ms-composer__spacer {
  flex: 1;
  min-width: 16px;
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
.ms-chip--stepper {
  cursor: default;
  padding-right: 4px;
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
.ms-composer__generate {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 9px;
  flex-shrink: 0;
  height: var(--mold-ctl-lg);
  padding: 0 16px;
  border: 0;
  border-radius: var(--mold-radius-2);
  background: var(--mold-blue);
  color: var(--mold-on-accent);
  font-size: var(--mold-fs-sm);
  font-weight: 600;
  letter-spacing: -0.005em;
  white-space: nowrap;
  cursor: pointer;
  transition: filter var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-composer__generate:hover:not(:disabled) {
  filter: brightness(1.05);
}
.ms-composer__generate:active:not(:disabled) {
  transform: translateY(1px);
}
.ms-composer__generate:disabled {
  opacity: 0.55;
  cursor: default;
}
.ms-composer__key {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  opacity: 0.7;
}
</style>
