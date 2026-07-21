<script setup lang="ts">
/*
 * Composer card (Mold Studio Create) — the prompt bed. Autogrow textarea,
 * a collapsible Style row (tapping the active preset deselects it), a mono
 * summary line with an inline "expanded · undo" affordance, and the Expand /
 * Generate action row. Generate carries the ⌘↵ keycap; ⌘↵ / Ctrl+↵ inside the
 * textarea submits. The card never rewrites the prompt for style — the active
 * preset is applied at request time by `useGenerateForm.promptWithStyle`.
 */
import { computed, nextTick, ref, watch } from "vue";
import Chip from "@ui/components/Chip.vue";
import Icon from "@ui/components/Icon.vue";
import Keycap from "@ui/components/Keycap.vue";
import { STYLE_PRESETS, stylePresetById } from "../../lib/stylePresets";

const props = withDefaults(
  defineProps<{
    /** Prompt text (v-model). */
    prompt: string;
    /** Active style preset id (v-model:stylePreset). */
    stylePreset: string | null;
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
  }>(),
  { expanded: false, busy: false },
);

const emit = defineEmits<{
  "update:prompt": [value: string];
  "update:stylePreset": [value: string | null];
  submit: [];
  expand: [];
  "undo-expand": [];
}>();

const textarea = ref<HTMLTextAreaElement | null>(null);
const stylesOpen = ref(false);

const activePreset = computed(() => stylePresetById(props.stylePreset));
const styleLabel = computed(() => activePreset.value?.name ?? "None");

const summaryLine = computed(() => {
  const base = `${props.aspectLabel} · ${props.width}×${props.height} · ${props.steps} steps`;
  return props.batchSize > 1 ? `${base} · ×${props.batchSize}` : base;
});

const expandLabel = computed(() =>
  props.batchSize > 1 ? `Expand to ${props.batchSize}` : "Expand prompt",
);

function onInput(event: Event) {
  emit("update:prompt", (event.target as HTMLTextAreaElement).value);
}

function onKeydown(event: KeyboardEvent) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (!props.busy) emit("submit");
  }
}

function pickStyle(id: string) {
  // Tapping the active preset deselects it (→ null); otherwise select it.
  emit("update:stylePreset", props.stylePreset === id ? null : id);
}

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
      placeholder="Describe the image you want to create…"
      rows="2"
      @input="onInput"
      @keydown="onKeydown"
    />

    <div class="composer__style">
      <button
        type="button"
        class="composer__style-head"
        :aria-expanded="stylesOpen"
        data-test="style-toggle"
        @click="stylesOpen = !stylesOpen"
      >
        <span class="composer__kicker">Style</span>
        <Chip :active="!!activePreset" tabindex="-1" data-test="style-active">{{
          styleLabel
        }}</Chip>
        <span class="composer__spacer" />
        <Icon :name="stylesOpen ? 'chevron-up' : 'chevron-down'" :size="15" />
      </button>
      <div v-if="stylesOpen" class="composer__chips" data-test="style-chips">
        <Chip
          v-for="preset in STYLE_PRESETS"
          :key="preset.id"
          :active="stylePreset === preset.id"
          :data-test="`style-chip-${preset.id}`"
          @click="pickStyle(preset.id)"
          >{{ preset.name }}</Chip
        >
      </div>
    </div>

    <div class="composer__actions">
      <span class="composer__summary" data-test="composer-summary">{{
        summaryLine
      }}</span>
      <button
        v-if="expanded"
        type="button"
        class="composer__undo"
        data-test="composer-undo"
        @click="emit('undo-expand')"
      >
        ✨ expanded · undo
      </button>
      <span class="composer__spacer" />
      <button
        type="button"
        class="composer__expand"
        data-test="composer-expand"
        :disabled="busy"
        @click="emit('expand')"
      >
        <Icon name="sparkle" :size="15" />
        {{ expandLabel }}
      </button>
      <button
        type="button"
        class="composer__generate"
        data-test="composer-submit"
        :disabled="busy"
        @click="emit('submit')"
      >
        <Icon name="sparkle" :size="16" :stroke-width="2" />
        Generate
        <Keycap on-accent>⌘<span class="composer__return">↵</span></Keycap>
      </button>
    </div>
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

.composer__style {
  padding-top: 4px;
}

.composer__style-head {
  display: flex;
  align-items: center;
  gap: 9px;
  width: 100%;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  padding: 4px 0;
  text-align: left;
  cursor: pointer;
}

.composer__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.composer__spacer {
  flex: 1;
}

.composer__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 9px;
}

.composer__actions {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-top: 12px;
  flex-wrap: wrap;
}

.composer__summary {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}

.composer__undo {
  border: 0;
  background: transparent;
  color: var(--safelight);
  font-family: var(--f-mono);
  font-size: 10px;
  padding: 0;
  cursor: pointer;
}

.composer__expand {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 0 15px;
  height: 42px;
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
  height: 42px;
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
