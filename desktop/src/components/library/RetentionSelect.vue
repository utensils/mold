<script setup lang="ts">
/*
 * RetentionSelect — the `gallery.trash_retention_days` control (Settings ▸
 * Library for This device, Machines ▸ host detail for remotes). Options are
 * the shared `RETENTION_OPTIONS` (1 / 7 / 30 / 90 / 365 days / Forever);
 * a value the server reports outside that list is still shown, never
 * silently rewritten. Pure: emits the new number, the parent saves it.
 */
import { computed } from "vue";
import { RETENTION_OPTIONS, retentionLabel } from "@studio/lib/libraryOrganization";

const props = withDefaults(
  defineProps<{
    /** Days; `0` = keep forever. */
    modelValue: number;
    disabled?: boolean;
    /** Secondary line under the control. */
    hint?: string | null;
    /** Accessible name — the control has no visible <label> of its own. */
    ariaLabel?: string;
  }>(),
  { disabled: false, hint: null, ariaLabel: "Trash retention" },
);

const emit = defineEmits<{ "update:modelValue": [days: number] }>();

const current = computed(() =>
  Number.isFinite(props.modelValue) && props.modelValue > 0 ? Math.floor(props.modelValue) : 0,
);

/** The shared ladder plus the current value when it is off-ladder. */
const options = computed<{ value: number; label: string }[]>(() => {
  const values = [...RETENTION_OPTIONS];
  if (!values.includes(current.value)) {
    values.splice(values.length - 1, 0, current.value);
    values.sort((a, b) => (a === 0 ? 1 : b === 0 ? -1 : a - b));
  }
  return values.map((value) => ({ value, label: retentionLabel(value) }));
});

function onChange(event: Event) {
  const raw = (event.target as HTMLSelectElement).value;
  const days = Number(raw);
  if (!Number.isFinite(days) || days < 0) return;
  emit("update:modelValue", days);
}
</script>

<template>
  <div class="flex flex-col gap-1" data-test="retention-select">
    <select
      :value="String(current)"
      :disabled="disabled"
      :aria-label="ariaLabel"
      class="border-edge h-7 max-w-56 rounded-control border bg-bath px-2 text-body text-ink disabled:opacity-40"
      @change="onChange"
    >
      <option v-for="option in options" :key="option.value" :value="String(option.value)">
        {{ option.label }}
      </option>
    </select>
    <p v-if="hint" class="text-caption text-ink-3" data-test="retention-hint">{{ hint }}</p>
  </div>
</template>
