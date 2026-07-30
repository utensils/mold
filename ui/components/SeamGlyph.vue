<script setup lang="ts">
/*
 * Seam transition glyph — the shared bar icon drawn inside the circular
 * badge on the seam pill and the seam editor's teaching rows (mockup
 * sequence-filmstrip). Smooth carries motion through the seam (rising into
 * a peak), cut is two hard bars with a gap, fade tapers away. Size and
 * color come from the consumer (`fill: currentColor`).
 */
import type { SequenceTransition } from "../lib/seam";

defineProps<{ transition: SequenceTransition }>();
</script>

<template>
  <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <g v-if="transition === 'smooth'">
      <rect x="2.5" y="5" width="2" height="6" rx="1" />
      <rect x="7" y="2.5" width="2" height="11" rx="1" />
      <rect x="11.5" y="5" width="2" height="6" rx="1" />
    </g>
    <g v-else-if="transition === 'cut'">
      <rect x="4.2" y="2.5" width="2.4" height="11" rx="1" />
      <rect x="9.4" y="2.5" width="2.4" height="11" rx="1" />
    </g>
    <g v-else>
      <rect x="2.5" y="2.5" width="2" height="11" rx="1" />
      <rect x="7" y="4.5" width="2" height="7" rx="1" opacity="0.55" />
      <rect x="11.5" y="6" width="2" height="4" rx="1" opacity="0.28" />
    </g>
  </svg>
</template>
