<script setup lang="ts">
// C1 STUB — replaced at merge by lane C1's DisclosureRow.vue.
/*
 * One 44px rail row: plain label and note on the left, the mono value and a
 * chevron on the right. It opens something over the page — the mock's Modals
 * rule — and owns no state of its own.
 */
import Icon from "@ui/components/Icon.vue";

defineProps<{
  label: string;
  note?: string;
  value?: string;
  testId?: string;
}>();

const emit = defineEmits<{ open: [] }>();
</script>

<template>
  <button type="button" class="row" :data-test="testId" @click="emit('open')">
    <span class="row__text">
      <span class="row__label">{{ label }}</span>
      <span v-if="note" class="row__note">{{ note }}</span>
    </span>
    <span v-if="value" class="row__value">{{ value }}</span>
    <Icon name="chevron-right" :size="13" />
  </button>
</template>

<style scoped>
.row {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: var(--mold-row-h-disclosure, 44px);
  padding: 6px 12px;
  border: 0;
  border-top: 1px solid var(--mold-border);
  background: transparent;
  color: var(--mold-text-2);
  text-align: left;
  cursor: pointer;
}
.row:first-child {
  border-top: 0;
}
.row:hover {
  background: var(--mold-row-hover);
}
.row__text {
  display: grid;
  min-width: 0;
  flex: 1;
}
.row__label {
  font-size: var(--mold-fs-xs);
  font-weight: 600;
  color: var(--mold-text);
}
.row__note {
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
}
.row__value {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
  white-space: nowrap;
}
</style>
