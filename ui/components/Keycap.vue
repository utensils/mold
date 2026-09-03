<script setup lang="ts">
/*
 * Keycap — keyboard shortcut hint (spec §03). Two contexts: `on-accent` for
 * placement inside accent buttons (the Generate button's ⌘↵ chip), otherwise
 * the subtle outlined hint (the command palette's "esc"). Purely decorative
 * text — the shortcut itself is wired by the parent.
 */
withDefaults(defineProps<{ onAccent?: boolean }>(), { onAccent: false });
</script>

<template>
  <kbd class="ms-keycap" :data-on-accent="onAccent ? 'true' : undefined">
    <slot />
  </kbd>
</template>

<style scoped>
/* Prototype radii (5px hint / 7px on-accent) collapse to the control-sm token. */
.ms-keycap {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  line-height: 1;
  font-family: var(--mold-font-mono);
  font-size: 11px;
  border: 1px solid var(--mold-border);
  background: transparent;
  color: var(--mold-text-dim);
  padding: 2px 6px;
  border-radius: var(--mold-radius-1);
  white-space: nowrap;
}

.ms-keycap[data-on-accent="true"] {
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 14px;
  font-weight: 700;
  border: 0;
  background: var(--mold-on-accent);
  color: var(--mold-blue);
  padding: 6px 9px;
  border-radius: var(--mold-radius-1);
}
</style>
