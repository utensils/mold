<script setup lang="ts">
/*
 * A settings section: a mono group label, one dim sentence saying what the
 * section is for, and a bordered panel holding its rows.
 */
defineProps<{ label: string; summary?: string | undefined }>();
</script>

<template>
  <div class="ms-settings-section">
    <div class="ms-settings-section__head">
      <span class="ms-settings-section__label">{{ label }}</span>
      <span v-if="summary" class="ms-settings-section__summary">{{
        summary
      }}</span>
    </div>
    <div class="ms-settings-section__panel">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.ms-settings-section {
  display: flex;
  flex-direction: column;
  gap: var(--mold-sp-2);
}
.ms-settings-section__head {
  display: flex;
  flex-direction: column;
  gap: var(--mold-sp-1);
}
.ms-settings-section__label {
  color: var(--mold-text-2);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.ms-settings-section__summary {
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
}
.ms-settings-section__panel {
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-panel, var(--mold-bg-deep));
}
</style>
