<script setup lang="ts">
/*
 * One style's overrides, collapsed.
 *
 * A machine with a dozen styles reports `models.<style>.<field>` for each of
 * them — 104 rows on hal9000, which flat is most of the Settings page and
 * tells you nothing about any one style. Collapsed, it is 13 rows saying which
 * styles have been tuned and by how much; the fields arrive only when a style
 * is opened.
 *
 * The fields are named the way `mold config list` names them, because that is
 * what `mold config set models.<style>.<field>` takes and what a person tuning
 * a single style has already seen.
 */
import { computed, ref } from "vue";
import SettingRow from "./SettingRow.vue";
import NumberControl from "./NumberControl.vue";
import TextControl from "./TextControl.vue";
import { PER_STYLE_FIELDS, parsePerStyleKey } from "../../lib/settingsSchema";
import {
  canResetConfig,
  type ConfigRow,
  type ConfigValue,
} from "../../api/config";

const props = defineProps<{
  style: string;
  rows: readonly ConfigRow[];
  /** Friendlier name for the style, from `studio/lib/styleLabel.ts`. */
  displayName?: string | null | undefined;
}>();
const emit = defineEmits<{
  (e: "save", key: string, value: ConfigValue): void;
  (e: "reset", key: string): void;
}>();

const open = ref(false);

/** Numbers where the engine's own type is numeric; text everywhere else. */
const NUMERIC_FIELDS = new Set([
  "default_steps",
  "default_guidance",
  "default_width",
  "default_height",
  "lora_scale",
]);

const byField = computed(() => {
  const map = new Map<string, ConfigRow>();
  for (const row of props.rows) {
    const parsed = parsePerStyleKey(row.key);
    if (parsed) map.set(parsed.field, row);
  }
  return map;
});

/** Only the fields this style has actually set: a null is the engine saying
 *  "nothing here", not an override. */
const overrides = computed(
  () =>
    props.rows.filter((row) => row.value !== null && row.value !== "").length,
);
const overrideLabel = computed(
  () =>
    `${overrides.value} ${overrides.value === 1 ? "override" : "overrides"}`,
);

const fields = computed(() =>
  PER_STYLE_FIELDS.map((field) => ({
    field,
    key: `models.${props.style}.${field}`,
    row: byField.value.get(field) ?? null,
    numeric: NUMERIC_FIELDS.has(field),
  })),
);

function numberOf(row: ConfigRow | null): number | null {
  const value = row?.value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function textOf(row: ConfigRow | null): string {
  return row?.value == null ? "" : String(row.value);
}
</script>

<template>
  <details
    class="ms-per-style"
    @toggle="open = ($event.target as HTMLDetailsElement).open"
  >
    <summary class="ms-per-style__summary">
      <span class="ms-per-style__names">
        <span class="ms-per-style__name" data-test="per-style-name">
          {{ displayName || style }}
        </span>
        <span class="ms-per-style__id" data-test="per-style-id">{{
          style
        }}</span>
      </span>
      <span class="ms-per-style__count" data-test="per-style-count">{{
        overrideLabel
      }}</span>
    </summary>

    <div v-if="open" class="ms-per-style__body">
      <SettingRow
        v-for="field in fields"
        :key="field.key"
        :label="field.field"
        :source="field.row?.source"
        :resettable="canResetConfig(field.key)"
        :data-test="`per-style-row-${field.field}`"
        @reset="emit('reset', field.key)"
      >
        <template #default>
          <NumberControl
            v-if="field.numeric"
            :model-value="numberOf(field.row)"
            :aria-label="field.key"
            @commit="(v) => emit('save', field.key, v)"
          />
          <TextControl
            v-else
            :model-value="textOf(field.row)"
            wide
            :aria-label="field.key"
            @commit="(v) => emit('save', field.key, v === '' ? null : v)"
          />
        </template>
      </SettingRow>
    </div>
  </details>
</template>

<style scoped>
.ms-per-style {
  border-bottom: var(--mold-bw) solid var(--mold-border);
}
.ms-per-style:last-child {
  border-bottom: none;
}
.ms-per-style__summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--mold-sp-3);
  min-height: var(--mold-row-h-table, 52px);
  padding: var(--mold-sp-2) var(--mold-sp-3);
  cursor: pointer;
  list-style: none;
}
.ms-per-style__summary::-webkit-details-marker {
  display: none;
}
.ms-per-style__summary::after {
  content: "▾";
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}
.ms-per-style[open] .ms-per-style__summary::after {
  content: "▴";
}
.ms-per-style__names {
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 2px;
}
.ms-per-style__name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--mold-text);
  font-size: var(--mold-fs-sm);
  font-weight: 500;
}
.ms-per-style__id {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}
.ms-per-style__count {
  flex: none;
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}
.ms-per-style__body {
  border-top: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-bg-deep);
}
</style>
