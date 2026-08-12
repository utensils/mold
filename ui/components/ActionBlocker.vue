<script setup lang="ts">
withDefaults(
  defineProps<{
    reason: string;
    title?: string;
    compact?: boolean;
  }>(),
  { title: "Before you generate", compact: false },
);
</script>

<template>
  <div
    class="ms-action-blocker"
    :class="{ 'ms-action-blocker--compact': compact }"
    role="status"
    data-test="action-blocker"
  >
    <span class="ms-action-blocker__mark" aria-hidden="true">!</span>
    <span class="ms-action-blocker__copy">
      <strong>{{ title }}</strong>
      <span>{{ reason }}</span>
    </span>
  </div>
</template>

<style scoped>
.ms-action-blocker {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  padding: 9px 11px;
  border: 1px solid color-mix(in srgb, var(--stop) 28%, var(--edge));
  border-radius: 9px;
  background: color-mix(in srgb, var(--stop) 7%, var(--bath));
  color: var(--ink-2);
}
.ms-action-blocker__mark {
  display: grid;
  flex: 0 0 22px;
  width: 22px;
  height: 22px;
  place-items: center;
  border-radius: 50%;
  background: color-mix(in srgb, var(--stop) 14%, transparent);
  color: var(--stop);
  font-family: var(--f-mono);
  font-size: 12px;
  font-weight: 700;
}
.ms-action-blocker__copy {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 3px 7px;
  font-size: 11px;
  line-height: 1.35;
}
.ms-action-blocker__copy strong {
  flex: 0 0 auto;
  color: var(--ink);
  font-weight: 700;
}
.ms-action-blocker__copy span {
  min-width: 0;
}
.ms-action-blocker--compact {
  padding: 6px 9px;
}
.ms-action-blocker--compact .ms-action-blocker__mark {
  flex-basis: 18px;
  width: 18px;
  height: 18px;
  font-size: 10px;
}
</style>
