<script setup lang="ts">
import { ref } from "vue";
import { STYLE_PRESETS, stylePresetLabel } from "../lib/stylePresets";

/**
 * Composer style row (Mold Studio Create, iPhone). A collapsible strip of look
 * presets mirroring the desktop StyleChips. The selection is a prompt modifier
 * composed at submit — this component only owns the preset id, never the
 * textarea. Single-select; tapping the active chip clears it back to "None".
 */
const props = defineProps<{ modelValue: string }>();
const emit = defineEmits<{ "update:modelValue": [value: string] }>();

const open = ref(false);

function pick(id: string): void {
  emit("update:modelValue", props.modelValue === id ? "" : id);
}
</script>

<template>
  <div class="mobile-style">
    <button
      type="button"
      class="mobile-style-head"
      data-test="mobile-style-toggle"
      :aria-expanded="open"
      @click="open = !open"
    >
      <span class="mobile-style-kicker">Style</span>
      <span
        class="mobile-style-chip"
        :aria-pressed="modelValue !== ''"
        data-test="mobile-style-active"
        >{{ stylePresetLabel(modelValue) }}</span
      >
      <span class="mobile-style-spacer" />
      <svg
        class="mobile-style-chevron"
        :class="{ 'is-open': open }"
        width="17"
        height="17"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2.2"
        stroke-linecap="round"
        aria-hidden="true"
      >
        <path d="M6 9l6 6 6-6" />
      </svg>
    </button>
    <div v-if="open" class="mobile-style-row" data-test="mobile-style-row">
      <button
        v-for="preset in STYLE_PRESETS"
        :key="preset.id"
        type="button"
        class="mobile-style-chip"
        :aria-pressed="modelValue === preset.id"
        :data-test="`mobile-style-${preset.id}`"
        @click="pick(preset.id)"
      >
        {{ preset.label }}
      </button>
    </div>
  </div>
</template>
