<script setup lang="ts">
withDefaults(
  defineProps<{
    loading?: boolean;
  }>(),
  { loading: false },
);
</script>

<template>
  <span
    class="mobile-media-placeholder"
    :class="{ 'is-loading': loading }"
    aria-hidden="true"
    data-test="mobile-media-placeholder"
  >
    <svg viewBox="0 0 32 32" fill="none">
      <rect x="5.5" y="6.5" width="21" height="19" rx="4.5" />
      <circle cx="20.5" cy="12" r="2.25" />
      <path d="m8.5 22 6.1-6.1 3.8 3.8 2.1-2.1 3 3" />
    </svg>
  </span>
</template>
