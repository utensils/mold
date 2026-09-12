<script setup lang="ts">
/*
 * The composer's Shape chip — "Square · 1024 ▼".
 *
 * Presentational on purpose. The page resolves ONE `resolveOutputShape`
 * answer and every control on the screen reads that same object; the chip is
 * handed the family it named and the size it wrote (already marked `≈` when
 * the resolver called it approximate) so it can never state a second,
 * disagreeing reading of the canvas. It opens the rail's Shape and size
 * group; it does not own a picker of its own.
 */
withDefaults(
  defineProps<{
    /** The resolved shape family, in plain words: "Square", "Landscape". */
    label: string;
    /** The resolved size, already formatted: "1024", "≈1216×704". A recipe
     * whose canvas follows its source has none to name. */
    sublabel?: string | null;
    disabled?: boolean;
  }>(),
  { sublabel: null, disabled: false },
);

const emit = defineEmits<{ open: [] }>();
</script>

<template>
  <button
    type="button"
    class="shape-chip"
    data-test="shape-chip"
    title="Shape and size"
    :disabled="disabled"
    @click="emit('open')"
  >
    <span class="shape-chip__label">{{ label }}</span>
    <span
      v-if="sublabel"
      class="shape-chip__size"
      data-test="shape-chip-size"
      >{{ sublabel }}</span
    >
    <span class="shape-chip__caret" aria-hidden="true">▼</span>
  </button>
</template>

<style scoped>
.shape-chip {
  /* literal: the mock's 28px chip row, the one metric between --mold-ctl-sm
   * (24px, dense icon buttons) and --mold-ctl-md (26px, toolbars). */
  --composer-chip-h: 28px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: var(--composer-chip-h);
  padding: 0 10px;
  flex-shrink: 0;
  white-space: nowrap;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}

.shape-chip:hover:not(:disabled) {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}

.shape-chip:disabled {
  opacity: 0.55;
  cursor: default;
}

.shape-chip__label {
  overflow: hidden;
  text-overflow: ellipsis;
}

.shape-chip__size {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.shape-chip__caret {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
</style>
