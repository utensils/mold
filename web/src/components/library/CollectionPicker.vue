<script setup lang="ts">
/*
 * "In collections" checklist (Library organization, V3 "Shelf"). One row per
 * merged (cross-host) collection with a checked / mixed state, plus a trailing
 * "New collection…" row. Used by the Lightbox aside and the bulk bar's
 * Add-to-collection popover; the parent owns the fan-out.
 */
import Icon from "@ui/components/Icon.vue";

export interface CollectionPickerRow {
  slug: string;
  name: string;
  /** Every print in question is in it. */
  checked: boolean;
  /** Some of the prints are in it (bulk). */
  mixed?: boolean;
  /** Shown in mono after the name ("plato only", a count…). */
  note?: string;
}

withDefaults(
  defineProps<{
    rows: readonly CollectionPickerRow[];
    disabled?: boolean;
    /** Mono footer line ("fans out to This Mac · plato"). */
    footer?: string;
  }>(),
  { disabled: false, footer: "" },
);

const emit = defineEmits<{
  (e: "toggle", slug: string, member: boolean): void;
  (e: "new"): void;
}>();
</script>

<template>
  <div class="cp" role="group" aria-label="Collections">
    <p v-if="rows.length === 0" class="cp__empty">No collections yet.</p>
    <label
      v-for="row in rows"
      :key="row.slug"
      class="cp__row"
      :data-on="row.checked ? 'true' : undefined"
      :data-mixed="row.mixed && !row.checked ? 'true' : undefined"
      data-test="collection-row"
    >
      <input
        type="checkbox"
        class="cp__box"
        :checked="row.checked"
        :indeterminate.prop="!!row.mixed && !row.checked"
        :disabled="disabled"
        :aria-label="`In ${row.name}`"
        data-test="collection-toggle"
        @change="
          emit('toggle', row.slug, ($event.target as HTMLInputElement).checked)
        "
      />
      <span class="cp__name">{{ row.name }}</span>
      <span v-if="row.note" class="cp__note">{{ row.note }}</span>
    </label>
    <button
      v-if="!disabled"
      type="button"
      class="cp__new"
      data-test="collection-new"
      @click="emit('new')"
    >
      <Icon name="plus" :size="13" /> New collection…
    </button>
    <p v-if="footer" class="cp__foot">{{ footer }}</p>
  </div>
</template>

<style scoped>
.cp {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.cp__empty {
  margin: 0 0 4px;
  font-size: 12px;
  font-style: italic;
  color: var(--ink-3);
}
.cp__row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 28px;
  padding: 0 6px;
  border-radius: var(--radius-control-sm);
  font-size: 12.5px;
  color: var(--rebate);
  cursor: pointer;
}
.cp__row:hover {
  background: color-mix(in srgb, var(--rebate) 5%, transparent);
}
.cp__row[data-on="true"] {
  color: var(--sel-ink);
}
.cp__box {
  width: 15px;
  height: 15px;
  margin: 0;
  accent-color: var(--safelight);
}
.cp__name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cp__note {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}
.cp__new {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 28px;
  margin-top: 2px;
  padding: 0 6px;
  border: 0;
  border-radius: var(--radius-control-sm);
  background: transparent;
  color: var(--safelight);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  text-align: left;
  cursor: pointer;
}
.cp__new:hover {
  background: var(--sel-bg);
}
.cp__foot {
  margin: 6px 0 0;
  padding: 0 6px;
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}
</style>
