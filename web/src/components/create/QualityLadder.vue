<script setup lang="ts">
// C1 STUB — replaced at merge by lane C1's QualityLadder.vue.
/*
 * The rail's Draft / Good / Best rows. The rungs come from
 * `@studio/lib/qualityPresets` — the recipe's advertised `steps.recommended`,
 * standing in with the host's own half/default/1.5× formula for a server that
 * predates the field. A pinned-steps recipe yields no rows and the profile's
 * own note under Detail stays the whole explanation.
 */
import { computed } from "vue";
import {
  activeQualityPreset,
  type QualityPreset,
} from "@studio/lib/qualityPresets";

const props = withDefaults(
  defineProps<{
    presets: QualityPreset[];
    steps: number;
    disabled?: boolean;
    /** Optional per-row seconds estimate, e.g. `(steps) => "~3s"`. */
    estimateFor?: ((steps: number) => string | null) | null;
  }>(),
  { disabled: false, estimateFor: null },
);

const emit = defineEmits<{ select: [steps: number] }>();

const active = computed(() => activeQualityPreset(props.presets, props.steps));

function meta(preset: QualityPreset): string {
  const estimate = props.estimateFor?.(preset.steps) ?? null;
  const passes = `${preset.steps} passes`;
  return estimate ? `${estimate} · ${passes}` : passes;
}
</script>

<template>
  <div v-if="presets.length" class="ladder" data-test="quality-ladder">
    <div class="ladder__kicker">Quality</div>
    <button
      v-for="preset in presets"
      :key="preset.key"
      type="button"
      class="ladder__row"
      :class="{ 'ladder__row--on': active === preset.key }"
      :data-test="`quality-${preset.key}`"
      :aria-pressed="active === preset.key"
      :disabled="disabled"
      @click="emit('select', preset.steps)"
    >
      <span class="ladder__label">{{ preset.label }}</span>
      <span class="ladder__meta">{{ meta(preset) }}</span>
    </button>
  </div>
</template>

<style scoped>
.ladder {
  background: var(--mold-surface);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  padding: 14px;
}
.ladder__kicker {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mold-text-dim);
  margin-bottom: 9px;
}
.ladder__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  width: 100%;
  min-height: var(--mold-row-h-preset, 34px);
  padding: 0 10px;
  border: 1px solid transparent;
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
}
.ladder__row--on {
  border-color: var(--mold-blue);
  color: var(--mold-text);
}
.ladder__row:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.ladder__meta {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
}
</style>
