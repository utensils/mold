<script setup lang="ts">
// C1 STUB — replaced at merge by lane C1's ShapeChip.vue.
/*
 * Composer shape chip — the mock's `Square · 1024 ▼`. It renders ONE
 * `resolveOutputShape` result the page hands down (label = the family badge,
 * sublabel = the pixels) and opens the rail's Shape/Size group; it never
 * computes a size of its own.
 */
import Icon from "@ui/components/Icon.vue";

defineProps<{
  label: string;
  sublabel?: string;
  disabled?: boolean;
}>();

const emit = defineEmits<{ open: [] }>();
</script>

<template>
  <button
    type="button"
    class="chip"
    data-test="composer-shape-chip"
    :disabled="disabled"
    :title="sublabel ? `${label} · ${sublabel}` : label"
    @click="emit('open')"
  >
    <span class="chip__label">{{ label }}</span>
    <span v-if="sublabel" class="chip__sub" data-test="composer-shape-sublabel">
      {{ sublabel }}
    </span>
    <Icon name="chevron-down" :size="12" />
  </button>
</template>

<style scoped>
.chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: var(--mold-ctl-md, 28px);
  padding: 0 10px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--ink-2);
  font-size: 12px;
  cursor: pointer;
}
.chip:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.chip__sub {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}
</style>
