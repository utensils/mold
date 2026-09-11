<script setup lang="ts">
/*
 * One 44px row of the Create rail's disclosure list: plain words on the left
 * over the sentence that says what opening it does, the mono value it holds
 * on the right, and a chevron.
 *
 * The rail used to render these as always-open blocks — a source-media panel,
 * an identity panel and a `<details>` of everything else — so the settings
 * column was longer than the picture. Each row is now a door to the modal or
 * sheet that already existed.
 */
withDefaults(
  defineProps<{
    label: string;
    /** What opening it does. */
    note?: string;
    /** What it currently holds, in mono: "None", "2", "4821". */
    value?: string | null;
    /** The owner's own probe, so a page test can name the row it means. */
    testId?: string;
  }>(),
  { note: "", value: null, testId: "disclosure-row" },
);

const emit = defineEmits<{ open: [] }>();
</script>

<template>
  <div
    class="disclosure"
    :data-test="testId"
    role="button"
    tabindex="0"
    @click="emit('open')"
    @keydown.enter.prevent="emit('open')"
    @keydown.space.prevent="emit('open')"
  >
    <span class="disclosure__words">
      <span class="disclosure__label" data-test="disclosure-label">{{
        label
      }}</span>
      <span v-if="note" class="disclosure__note" data-test="disclosure-note">{{
        note
      }}</span>
    </span>
    <span class="disclosure__right">
      <span
        v-if="value"
        class="disclosure__value"
        data-test="disclosure-value"
        >{{ value }}</span
      >
      <span class="disclosure__chevron" aria-hidden="true">›</span>
    </span>
  </div>
</template>

<style scoped>
.disclosure {
  /* literal: the mock's 44px disclosure row. */
  --disclosure-row-h: 44px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: var(--disclosure-row-h);
  padding: 0 12px;
  cursor: pointer;
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}

.disclosure:hover {
  background: var(--mold-row-hover);
}

.disclosure:focus-visible {
  outline: var(--mold-bw) solid var(--mold-border-focus);
  outline-offset: -1px;
}

.disclosure__words {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}

.disclosure__label {
  font-size: var(--mold-fs-xs);
  color: var(--mold-text);
}

.disclosure__note {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.disclosure__right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.disclosure__value {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.disclosure__chevron {
  color: var(--mold-text-faint);
  font-size: var(--mold-fs-xs);
}
</style>
