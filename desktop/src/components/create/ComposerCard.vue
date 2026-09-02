<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { promptPlaceholder } from "@studio/lib/promptRequirement";
import Keycap from "@ui/components/Keycap.vue";
import Icon from "@ui/components/Icon.vue";
import ActionBlocker from "@ui/components/ActionBlocker.vue";
import ExpandControl from "../generate/ExpandControl.vue";
import EstimateBadge from "../generate/EstimateBadge.vue";
import StyleChips from "./StyleChips.vue";
import type { GenerateForm } from "../../lib/generateForm";
import { promptInputForForm } from "../../lib/promptRecipe";
import type { GenerateRequest } from "../../lib/api/types";
import type { ApiTarget } from "../../lib/api/client";
import { autoGrowRows } from "../../lib/autogrow";
import { PromptCycler, caretOnFirstLine, caretOnLastLine } from "@studio/lib/promptCycler";
import { primaryModifierPressed, shortcutLabel } from "../../lib/platform";

/**
 * The Create composer (Mold Studio). Owns the prompt textarea and its editing
 * affordances — ⌘↵ generate, ⌘E expand, ↑/↓ shell-style prompt history,
 * autogrow — plus the style row, estimate, expand control, and Generate button.
 * The view keeps all orchestration; this component only surfaces intent.
 */
const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    effectiveBatchSize: number;
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
    estimateRequest: GenerateRequest | null;
    estimateTarget: ApiTarget | null;
    preprocessingStatus: string | null;
    remixSource?: "original" | "current";
    /** Recent prompts for ↑/↓ history cycling. */
    history?: string[];
  }>(),
  { history: () => [], remixSource: "original", warningReason: null },
);

const emit = defineEmits<{
  generate: [];
  cancel: [];
  expand: [];
  remix: [];
  restore: [];
  "prompt-authored": [value: string];
  "update:remixSource": [value: "original" | "current"];
}>();

// Disabled state and corrective guidance are intentionally separate: obvious
// requirements such as an empty prompt do not need a persistent warning.
const generateDisabled = computed(() => props.disabled && !props.submitting);
// The recipe is the prompt rule's authority (a mesh family has no text
// encoder to feed), so the form's snapshot of it rides along — without it the
// pre-profile family rule would ask for a description nothing reads.
const placeholder = computed(() =>
  promptPlaceholder(promptInputForForm(props.form), "Describe the image you want to create…"),
);

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
watch(
  () => props.form.prompt,
  () => void nextTick(growPrompt),
  { flush: "post" },
);
onMounted(() => {
  growPrompt();
  promptEl.value?.focus();
});

function cycleHistory(direction: "prev" | "next"): boolean {
  const replacement = direction === "prev" ? cycler.prev(props.form.prompt) : cycler.next();
  if (replacement === null) return false;
  props.form.prompt = replacement;
  emit("prompt-authored", replacement);
  void nextTick(() => {
    const el = promptEl.value;
    el?.setSelectionRange(el.value.length, el.value.length);
  });
  return true;
}

function onPromptInput(event: Event) {
  cycler.reset();
  growPrompt();
  emit("prompt-authored", (event.target as HTMLTextAreaElement).value);
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && primaryModifierPressed(e)) {
    e.preventDefault();
    if (!generateDisabled.value) submitOrCancel();
  } else if ((e.key === "e" || e.key === "E") && primaryModifierPressed(e)) {
    e.preventDefault();
    expandControl.value?.expand();
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
    <div class="ms-composer__bench">
      <textarea
        ref="promptEl"
        v-model="form.prompt"
        data-selectable
        rows="2"
        aria-label="Prompt"
        :placeholder="placeholder"
        class="ms-composer__input"
        @keydown="onKeydown"
        @input="onPromptInput"
      />
      <StyleChips v-model="form.stylePreset" />
    </div>
    <div class="ms-composer__actions">
      <ExpandControl
        ref="expandControl"
        :prompt="form.prompt"
        :batch-size="effectiveBatchSize"
        :running="expansionRunning"
        :host-label="expansionHostLabel"
        :can-undo="canUndo"
        :blocked="preparedBlocked"
        :original-available="!!form.originalPrompt"
        :remix-source="remixSource"
        @expand="emit('expand')"
        @remix="emit('remix')"
        @update:remix-source="emit('update:remixSource', $event)"
        @restore="emit('restore')"
      />
      <div class="ms-composer__right">
        <span
          v-if="preprocessingStatus"
          class="ms-composer__status"
          data-test="preprocessing-status"
          >{{ preprocessingStatus }}</span
        >
        <EstimateBadge :request="estimateRequest" :target="estimateTarget" />
        <button
          type="button"
          data-test="generate-button"
          class="ms-composer__generate"
          :disabled="generateDisabled"
          @click="submitOrCancel"
        >
          <Icon name="sparkle" :size="16" />
          {{ buttonLabel }}
          <Keycap on-accent>{{ shortcutLabel("↩") }}</Keycap>
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
  border-top: 1px solid var(--edge);
  padding: 16px 22px;
  background: var(--bench);
}
.ms-composer__bench {
  display: flex;
  flex: 1;
  flex-direction: column;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: 12px;
  padding: 12px 14px;
  transition: border-color 0.1s;
}
.ms-composer__bench:focus-within {
  border-color: var(--safelight);
}
.ms-composer__input {
  flex: 1;
  width: 100%;
  box-sizing: border-box;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 14.5px;
  resize: none;
  outline: none;
  min-height: 44px;
  line-height: 1.45;
  overflow-x: hidden;
}
.ms-composer__actions {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-top: auto;
  padding-top: 12px;
  min-width: 0;
  flex-wrap: wrap;
}
.ms-composer__right {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-left: auto;
  min-width: 0;
}
.ms-composer__blocker {
  order: 3;
  margin-top: 10px;
}
.ms-composer__status {
  font-size: 11px;
  color: var(--ink-3);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-composer__generate {
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 0 20px;
  height: 42px;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 9px;
  cursor: pointer;
  transition: filter 0.1s;
}
.ms-composer__generate:hover:not(:disabled) {
  filter: brightness(1.05);
}
.ms-composer__generate:active:not(:disabled) {
  transform: translateY(1px);
}
.ms-composer__generate:disabled {
  opacity: 0.6;
  cursor: default;
}
</style>
