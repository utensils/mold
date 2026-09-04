<script setup lang="ts">
/*
 * Card surface — flat bench card with an inset top highlight (spec §04).
 * `large` bumps the radius, `dashed` swaps to a dashed cutting-edge border
 * for empty/offer cards, `padded: false` lets content own its own padding.
 */
withDefaults(
  defineProps<{
    large?: boolean;
    /** Dashed border for empty/offer cards. */
    dashed?: boolean;
    padded?: boolean;
  }>(),
  { large: false, dashed: false, padded: true },
);
</script>

<template>
  <div
    class="ms-card"
    :class="{
      'ms-card--large': large,
      'ms-card--dashed': dashed,
      'ms-card--padded': padded,
    }"
  >
    <slot />
  </div>
</template>

<style scoped>
.ms-card {
  background: var(--mold-bg);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
}

.ms-card--large {
  border-radius: var(--mold-radius-3);
}

.ms-card--dashed {
  border: 1px dashed var(--mold-border-control);
  box-shadow: none;
}

.ms-card--padded {
  padding: 16px;
}
</style>
