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
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  box-shadow: inset 0 1px 0 var(--card-hi);
}

.ms-card--large {
  border-radius: var(--radius-card-lg);
}

.ms-card--dashed {
  border: 1px dashed var(--ce);
  box-shadow: none;
}

.ms-card--padded {
  padding: 16px;
}
</style>
