<script setup lang="ts">
/*
 * Quality — Draft / Good / Best, the recipe's own recommended ladder read
 * through `studio/lib/qualityPresets.ts`. The rows are the pass counts the
 * HOST stands behind: the control's raw floor and ceiling are admission
 * bounds, not advice, so the ladder never reaches them, and a recipe that
 * pins its passes offers no rows at all (the profile's own note under the
 * Detail slider stays the whole explanation).
 *
 * Timing belongs to the machine, not to this component. `estimateFor` is
 * asked per row and a `null` answer renders nothing — a row is then its name
 * and its pass count, never an invented duration.
 */
import type { QualityPreset } from "@studio/lib/qualityPresets";

withDefaults(
  defineProps<{
    presets: QualityPreset[];
    /** The pass count in the form; the row that matches it is the live one. */
    steps: number;
    /** The machine's own estimate for a row, or null where it has none. */
    estimateFor?: ((steps: number) => string | null) | null;
    disabled?: boolean;
  }>(),
  { estimateFor: null, disabled: false },
);

const emit = defineEmits<{ select: [steps: number] }>();
</script>

<template>
  <div
    v-if="presets.length"
    class="quality"
    data-test="quality-ladder"
    role="radiogroup"
    aria-label="Quality"
  >
    <button
      v-for="preset in presets"
      :key="preset.key"
      type="button"
      class="quality__row"
      :class="{ 'quality__row--live': preset.steps === steps }"
      :data-test="`quality-row-${preset.key}`"
      role="radio"
      :aria-checked="preset.steps === steps"
      :disabled="disabled"
      @click="emit('select', preset.steps)"
    >
      <span class="quality__label">{{ preset.label }}</span>
      <span class="quality__meta">
        <template v-if="estimateFor?.(preset.steps)"
          >{{ estimateFor(preset.steps) }} · </template
        >{{ preset.steps }} passes
      </span>
    </button>
  </div>
</template>

<style scoped>
.quality {
  display: flex;
  flex-direction: column;
}

.quality__row {
  /* literal: the mock's 34px preset row. */
  --quality-row-h: 34px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  height: var(--quality-row-h);
  padding: 0 10px;
  border: 0;
  border-radius: var(--mold-radius-1);
  background: transparent;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  text-align: left;
  cursor: pointer;
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}

.quality__row:hover:not(:disabled) {
  background: var(--mold-row-hover);
  color: var(--mold-text);
}

.quality__row:disabled {
  opacity: 0.55;
  cursor: default;
}

.quality__row--live {
  background: var(--mold-surface);
  color: var(--mold-text);
  font-weight: 600;
}

.quality__meta {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  white-space: nowrap;
}
</style>
