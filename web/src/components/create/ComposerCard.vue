<script setup lang="ts">
/*
 * Composer card (Mold Studio Create) — the prompt bed. Autogrow textarea, a
 * mono summary line with an inline "expanded · undo" affordance, and the
 * chip + Write-more / Generate action row. Generate carries the ⌘↵ keycap;
 * ⌘↵ / Ctrl+↵ inside the textarea submits.
 *
 * The action row's first three positions are SLOTS — `style`, `shape`,
 * `count` — because the style picker, the resolved output shape and the batch
 * count all belong to the form the page owns. The composer decides only where
 * they sit. Slot content is compiled in the PARENT and never inherits this
 * component's scoped CSS, so each chip carries its own look.
 *
 * Where the composer sits is the page's decision too: `composer--sticky`
 * parks it at the bottom of the wide column, `composer--docked` fixes it to
 * the bottom of a narrow one. The card takes neither on its own.
 *
 * There is no prompt-preset strip: Style is the style the picture is made
 * with, and a second "Photoreal" on the same screen meaning a phrase appended
 * to the prompt is two different things wearing one word.
 */
import { computed, nextTick, ref, watch } from "vue";
import Icon from "@ui/components/Icon.vue";
import Keycap from "@ui/components/Keycap.vue";
import ActionBlocker from "@ui/components/ActionBlocker.vue";
import {
  PromptCycler,
  caretOnFirstLine,
  caretOnLastLine,
} from "@studio/lib/promptCycler";
import { OPTIONAL_PROMPT_PLACEHOLDER } from "@studio/lib/promptRequirement";
import type { PromptAuthoringSource } from "@studio/lib/promptProvenance";

const props = withDefaults(
  defineProps<{
    /** Prompt text (v-model). */
    prompt: string;
    /** Aspect label for the summary line (e.g. "1:1" or "Custom"). */
    aspectLabel: string;
    width: number;
    height: number;
    steps: number;
    batchSize: number;
    /** An in-place expansion is undoable (batch = 1 rewrite). */
    expanded?: boolean;
    /** Disable submit/expand (e.g. a job is mid-flight). */
    busy?: boolean;
    cancellable?: boolean;
    busyLabel?: string;
    /** Actionable request prerequisite shown beside Generate. */
    disabledReason?: string | null;
    /** The visual conditioning lets this render go out undescribed. */
    promptOptional?: boolean;
    /** Model-specific required-prompt wording. */
    requiredPlaceholder?: string;
    /** Fully resolved prompt-bed placeholder from the page that holds the
     * recipe (`promptPlaceholder`). It wins over the two props above, which
     * remain the fallback for a caller with no recipe in scope — a recipe
     * that IGNORES the prompt is neither "required" nor "optional" wording. */
    placeholder?: string | null;
    /** Why Expand and Remix are unavailable for the resolved recipe (a family
     * that IGNORES the prompt has no text encoder to rewrite for), or `null`
     * when they are available. Shown as the tooltip on both controls and as a
     * visible line beside the summary. */
    transformBlockedReason?: string | null;
    /** Prompt history (newest first) for ↑/↓ recall. */
    history?: string[];
  }>(),
  {
    expanded: false,
    busy: false,
    cancellable: false,
    busyLabel: "Planning generation…",
    disabledReason: null,
    promptOptional: false,
    requiredPlaceholder: "Describe the image you want to create…",
    placeholder: null,
    transformBlockedReason: null,
    history: () => [],
  },
);

const emit = defineEmits<{
  /** Tagged with how the text arrived: a ↑/↓ recall replaces the whole
   * prompt and releases any quick expansion, where typing keeps it. */
  "update:prompt": [value: string, source: PromptAuthoringSource];
  submit: [];
  cancel: [];
  expand: [];
  remix: [];
  "undo-expand": [];
}>();

const textarea = ref<HTMLTextAreaElement | null>(null);

const summaryLine = computed(() => {
  // A canvasless recipe (a 3-D mesh) renders at no pixel size at all, so the
  // shape/pixel clause is dropped rather than reading "3-D · 0×0".
  const canvas =
    props.width > 0 && props.height > 0
      ? `${props.aspectLabel} · ${props.width}×${props.height} · `
      : "";
  const base = `${canvas}${props.steps} passes`;
  return props.batchSize > 1 ? `${base} · ×${props.batchSize}` : base;
});

const promptFieldPlaceholder = computed(
  () =>
    props.placeholder?.trim() ||
    (props.promptOptional
      ? OPTIONAL_PROMPT_PLACEHOLDER
      : props.requiredPlaceholder),
);

const expandLabel = computed(() =>
  props.batchSize > 1 ? `Write ${props.batchSize} for me` : "Write more for me",
);
const generateDisabled = computed(
  () => !props.cancellable && (props.busy || Boolean(props.disabledReason)),
);
// Generate is deliberately untouched: only the two prompt TRANSFORMS are
// unavailable when the recipe reads no prompt — the render itself is fine.
const transformsDisabled = computed(
  () =>
    props.busy || !props.prompt.trim() || Boolean(props.transformBlockedReason),
);
const transformTitle = computed(
  () => props.transformBlockedReason?.trim() || undefined,
);

// Shell-style ↑/↓ prompt-history recall. The cycler is fed the latest history
// and gated on caret line so ↑/↓ still move the caret within a multi-line
// prompt; only an edge caret walks history.
const cycler = new PromptCycler();
watch(
  () => props.history,
  (h) => cycler.setEntries(h ?? []),
  { immediate: true },
);

function onInput(event: Event) {
  cycler.reset(); // hand-editing abandons history navigation
  emit("update:prompt", (event.target as HTMLTextAreaElement).value, "typed");
}

function onKeydown(event: KeyboardEvent) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (!generateDisabled.value) submitOrCancel();
    return;
  }
  // ⌘E / Ctrl+E is desktop's shortcut for the same rewrite, and the chip
  // carries the keycap — so the keycap has to be true here too.
  if (
    (event.metaKey || event.ctrlKey) &&
    (event.key === "e" || event.key === "E")
  ) {
    event.preventDefault();
    if (!transformsDisabled.value) emit("expand");
    return;
  }
  const el = event.target as HTMLTextAreaElement;
  if (event.key === "ArrowUp" && caretOnFirstLine(el)) {
    const recalled = cycler.prev(props.prompt);
    if (recalled !== null) {
      event.preventDefault();
      emit("update:prompt", recalled, "recalled");
    }
  } else if (
    event.key === "ArrowDown" &&
    cycler.navigating &&
    caretOnLastLine(el)
  ) {
    const recalled = cycler.next();
    if (recalled !== null) {
      event.preventDefault();
      emit("update:prompt", recalled, "recalled");
    }
  }
}

function submitOrCancel() {
  if (props.cancellable) emit("cancel");
  else emit("submit");
}

// Let the parent focus the prompt bed (⌘K "New print" starts here) and push a
// just-submitted prompt to the front of history for instant ↑ recall.
defineExpose({
  focus() {
    textarea.value?.focus();
  },
  record(prompt: string) {
    cycler.record(prompt);
  },
});

// Autogrow up to ~8 lines, tracking external prompt changes (Expand, Recreate).
watch(
  () => props.prompt,
  async () => {
    await nextTick();
    const el = textarea.value;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 8 * 28)}px`;
  },
  { immediate: true },
);
</script>

<template>
  <div class="composer">
    <textarea
      ref="textarea"
      class="composer__prompt"
      data-test="composer-prompt"
      :value="prompt"
      :placeholder="promptFieldPlaceholder"
      rows="2"
      @input="onInput"
      @keydown="onKeydown"
    />

    <div class="composer__actions">
      <!-- C1 STUB — the chip row's three slots land here at merge. Style is
           the composer's own control on every width (desktop's pattern), and
           Shape and Make ride beside it instead of living in the rail. -->
      <slot name="style" />
      <slot name="shape" />
      <slot name="count" />
      <span class="composer__summary" data-test="composer-summary">{{
        summaryLine
      }}</span>
      <span
        v-if="transformBlockedReason"
        class="composer__summary"
        data-test="composer-transform-blocked"
        >{{ transformBlockedReason }}</span
      >
      <button
        v-if="expanded"
        type="button"
        class="composer__undo"
        data-test="composer-undo"
        @click="emit('undo-expand')"
      >
        <Icon name="sparkle" :size="12" />
        expanded · undo
      </button>
      <slot name="style" />
      <slot name="shape" />
      <slot name="count" />
      <span class="composer__spacer" />
      <button
        type="button"
        class="composer__expand"
        data-test="composer-expand"
        :disabled="transformsDisabled"
        :title="transformTitle"
        @click="emit('expand')"
      >
        <Icon name="sparkle" :size="15" />
        {{ expandLabel }}
        <Keycap>⌘E</Keycap>
      </button>
      <button
        type="button"
        class="composer__expand"
        data-test="composer-remix"
        :disabled="transformsDisabled"
        :title="transformTitle"
        @click="emit('remix')"
      >
        <Icon name="sparkle" :size="15" />
        Remix
      </button>
      <button
        type="button"
        class="composer__generate"
        data-test="composer-submit"
        :disabled="generateDisabled"
        @click="submitOrCancel"
      >
        <Icon
          :name="cancellable ? 'close' : 'sparkle'"
          :size="16"
          :stroke-width="2"
        />
        {{ cancellable ? "Cancel" : "Generate" }}
        <Keycap on-accent>⌘<span class="composer__return">↵</span></Keycap>
      </button>
      <span
        v-if="cancellable"
        class="composer__summary"
        data-test="composer-busy-status"
        >{{ busyLabel }}</span
      >
    </div>
    <ActionBlocker
      v-if="disabledReason"
      class="composer__blocker"
      :reason="disabledReason"
    />
  </div>
</template>

<style scoped>
.composer {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 16px 18px;
}

/* Applied by the page, never by the card. The wide column's composer stays
 * on screen as the advisories and Recent scroll under it; the narrow one is
 * docked to the bottom of the viewport and the page pads for its height. */
.composer--sticky {
  position: sticky;
  bottom: 0;
  z-index: 1;
  box-shadow: 0 -10px 24px -18px rgba(0, 0, 0, 0.8);
}

.composer--docked {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 2;
  border-radius: 0;
  border-left: 0;
  border-right: 0;
  border-bottom: 0;
}

.composer__prompt {
  width: 100%;
  box-sizing: border-box;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 15px;
  line-height: 1.45;
  min-height: 58px;
  resize: none;
  outline: none;
}

.composer__spacer {
  flex: 1;
}

.composer__actions {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-top: 12px;
  flex-wrap: wrap;
}

.composer__blocker {
  margin-top: 12px;
}

.composer__summary {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}

.composer__undo {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 0;
  background: transparent;
  color: var(--safelight);
  font-family: var(--f-mono);
  font-size: 10px;
  padding: 0;
  cursor: pointer;
}

.composer__expand {
  /* literal: the mock's 28px chip row — the two prompt transforms stand at
   * the same height as the Style, Shape and Make chips beside them. */
  --composer-chip-h: 28px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 0 15px;
  height: var(--composer-chip-h);
  border-radius: var(--radius-control-lg);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.composer__expand:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.composer__generate {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 0 12px 0 22px;
  /* The kit has three control heights and the primary action is the tallest
   * of them; 42px was a fourth height nothing else on the screen used. */
  height: var(--mold-ctl-lg, 32px);
  border-radius: var(--radius-control-lg);
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
}

.composer__generate:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.composer__return {
  font-size: 15px;
}
</style>
