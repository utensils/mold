<script setup lang="ts">
import { computed } from "vue";
import type { ModelSource } from "../../lib/modelSource";

/**
 * 12px source mark for model rows: where did this model come from?
 * Neutral monograms in currentColor (not trademark logos): rounded tile for
 * Hugging Face, diamond for Civitai, disk for local files.
 */
const props = defineProps<{ source: ModelSource }>();

const label = computed(
  () => ({ hf: "Hugging Face", civitai: "Civitai", local: "Local file" })[props.source],
);
</script>

<template>
  <svg
    viewBox="0 0 12 12"
    class="h-3 w-3 shrink-0"
    role="img"
    :aria-label="label"
    :data-source="source"
  >
    <title>{{ label }}</title>
    <template v-if="source === 'hf'">
      <rect x="0.5" y="0.5" width="11" height="11" rx="2.5" fill="currentColor" opacity="0.18" />
      <text
        x="6"
        y="8.8"
        text-anchor="middle"
        font-size="6.4"
        font-weight="700"
        fill="currentColor"
        font-family="inherit"
      >
        HF
      </text>
    </template>
    <template v-else-if="source === 'civitai'">
      <path d="M6 0.8 L11.2 6 L6 11.2 L0.8 6 Z" fill="currentColor" opacity="0.18" />
      <text
        x="6"
        y="8.6"
        text-anchor="middle"
        font-size="6.4"
        font-weight="700"
        fill="currentColor"
        font-family="inherit"
      >
        C
      </text>
    </template>
    <template v-else>
      <ellipse cx="6" cy="3.4" rx="4.6" ry="1.9" fill="currentColor" opacity="0.35" />
      <path
        d="M1.4 3.4 v5.2 c0 1.05 2.06 1.9 4.6 1.9 s4.6 -0.85 4.6 -1.9 v-5.2"
        fill="none"
        stroke="currentColor"
        stroke-width="1"
        opacity="0.55"
      />
    </template>
  </svg>
</template>
