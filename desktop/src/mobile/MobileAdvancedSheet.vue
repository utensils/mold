<script setup lang="ts">
defineProps<{
  open: boolean;
  count: number;
}>();

defineEmits<{
  (event: "close"): void;
  (event: "reset"): void;
}>();
</script>

<template>
  <!--
    Full-screen Advanced surface (Mold Studio iOS spec). The hosted controls
    stay mounted whether the sheet is open or not so their validity wiring keeps
    reporting and the Generate flow can read every advanced field; the overlay
    only toggles visibility.
  -->
  <div
    class="mobile-advanced-sheet"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-label="Advanced settings"
    :aria-hidden="open ? undefined : 'true'"
    data-test="mobile-advanced-sheet"
  >
    <header class="mobile-advanced-sheet-head">
      <div class="mobile-advanced-sheet-title">
        <strong>Advanced</strong>
        <span
          v-if="count > 0"
          class="mobile-advanced-sheet-badge"
          data-test="mobile-advanced-count"
        >
          {{ count }}
        </span>
      </div>
      <button
        class="mobile-advanced-sheet-close"
        type="button"
        aria-label="Close advanced settings"
        data-test="mobile-advanced-close"
        @click="$emit('close')"
      >
        <span aria-hidden="true">✕</span>
      </button>
    </header>
    <div class="mobile-advanced-sheet-body">
      <slot />
      <button
        class="mobile-advanced-sheet-reset"
        type="button"
        data-test="mobile-advanced-reset"
        @click="$emit('reset')"
      >
        Reset advanced
      </button>
    </div>
  </div>
</template>
