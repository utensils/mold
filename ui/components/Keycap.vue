<script setup lang="ts">
/*
 * Keycap — a keyboard shortcut hint. Two contexts: `on-accent` for placement
 * inside an accent button (the Generate button's ⌘↵ chip), otherwise the
 * subtle outlined hint. Purely decorative text — the shortcut itself is wired
 * by the parent.
 */
withDefaults(defineProps<{ onAccent?: boolean }>(), { onAccent: false });
</script>

<template>
  <kbd class="ms-keycap" :data-on-accent="onAccent ? 'true' : undefined">
    <slot />
  </kbd>
</template>

<style scoped>
.ms-keycap {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  line-height: 1;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  border: var(--mold-bw) solid var(--mold-border);
  background: transparent;
  color: var(--mold-text-dim);
  padding: 2px 6px;
  border-radius: var(--mold-radius-1);
  white-space: nowrap;
}

.ms-keycap[data-on-accent="true"] {
  font-family: var(--mold-font-sans);
  font-size: var(--mold-fs-base);
  font-weight: 700;
  border: 0;
  background: var(--mold-on-accent);
  color: var(--mold-blue);
  padding: 6px 9px;
  border-radius: var(--mold-radius-1);
}
</style>
