<script setup lang="ts">
import { computed, useId } from "vue";
import type { ModelSource } from "../../lib/modelSource";

/**
 * Source mark for model rows and catalog cards: where did this model come
 * from? Hugging Face and Civitai render their official brand marks — the
 * 🤗 face (paths from huggingface.co's logo SVG, hands omitted at glyph
 * size) and the blue hexagonal C (paths from Civitai's open-source Logo
 * component). Local files keep the neutral currentColor disk monogram.
 */
const props = withDefaults(defineProps<{ source: ModelSource; size?: number }>(), {
  size: 12,
});

const label = computed(
  () => ({ hf: "Hugging Face", civitai: "Civitai", local: "Local file" })[props.source],
);

// Gradient ids must be document-unique — this glyph renders once per row.
const uid = useId();
</script>

<template>
  <!-- Hugging Face: brand-yellow smiley face. -->
  <svg
    v-if="source === 'hf'"
    viewBox="8.46 3 77.5 77.5"
    :width="size"
    :height="size"
    class="shrink-0"
    role="img"
    :aria-label="label"
    :data-source="source"
  >
    <title>{{ label }}</title>
    <path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z" />
    <path
      fill="#FF9D0B"
      d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z"
    />
    <path
      fill="#3A3B45"
      d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z"
    />
    <path
      fill="#FF323D"
      d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z"
    />
    <path
      fill="#3A3B45"
      fill-rule="evenodd"
      d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z"
      clip-rule="evenodd"
    />
    <path
      fill="#FF9D0B"
      d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5Z"
    />
  </svg>

  <!-- Civitai: blue hexagonal C. -->
  <svg
    v-else-if="source === 'civitai'"
    viewBox="0.2 1.6 20 20"
    :width="size"
    :height="size"
    class="shrink-0"
    role="img"
    :aria-label="label"
    :data-source="source"
  >
    <title>{{ label }}</title>
    <defs>
      <linearGradient :id="`cv-inner-${uid}`" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#081692" />
        <stop offset="1" stop-color="#1E043C" />
      </linearGradient>
      <linearGradient :id="`cv-outer-${uid}`" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#1284F7" />
        <stop offset="1" stop-color="#0A20C9" />
      </linearGradient>
    </defs>
    <path :fill="`url(#cv-inner-${uid})`" d="M1.5,6.6v10l8.7,5l8.7-5v-10l-8.7-5L1.5,6.6z" />
    <path
      :fill="`url(#cv-outer-${uid})`"
      d="M10.2,4.7l5.9,3.4V15l-5.9,3.4L4.2,15V8.1L10.2,4.7 M10.2,1.6l-8.7,5v10l8.7,5l8.7-5v-10C18.8,6.6,10.2,1.6,10.2,1.6z"
    />
    <path
      fill="#fff"
      d="M11.8,12.4l-1.7,1l-1.7-1v-1.9l1.7-1l1.7,1h2.1V9.3l-3.8-2.2L6.4,9.3v4.3l3.8,2.2l3.8-2.2v-1.2H11.8z"
    />
  </svg>

  <!-- Local file: neutral currentColor disk. -->
  <svg
    v-else
    viewBox="0 0 12 12"
    :width="size"
    :height="size"
    class="shrink-0"
    role="img"
    :aria-label="label"
    :data-source="source"
  >
    <title>{{ label }}</title>
    <ellipse cx="6" cy="3.4" rx="4.6" ry="1.9" fill="currentColor" opacity="0.35" />
    <path
      d="M1.4 3.4 v5.2 c0 1.05 2.06 1.9 4.6 1.9 s4.6 -0.85 4.6 -1.9 v-5.2"
      fill="none"
      stroke="currentColor"
      stroke-width="1"
      opacity="0.55"
    />
  </svg>
</template>
