<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { isRowLocked, provenance } from "../../lib/config";
import type { ConfigRow } from "../../lib/api/types";

const props = defineProps<{ row: ConfigRow }>();
const emit = defineEmits<{
  (e: "save", value: ConfigRow["value"]): void;
  (e: "reset"): void;
}>();

const locked = computed(() => isRowLocked(props.row));
const tag = computed(() => provenance(props.row.source));
const isBool = computed(() => typeof props.row.value === "boolean");
const isNumber = computed(() => typeof props.row.value === "number");

const draft = ref(props.row.value);
watch(
  () => props.row.value,
  (v) => (draft.value = v),
);

/** Commit on Enter or blur — but only a real change, and only a value the
 *  engine can honour. Every row a user tabs past raises `blur`, so an
 *  unguarded commit PUT and toasted rows nobody edited; and on a numeric row
 *  `Number("")` is 0 while `Number("abc")` is NaN, which `lib/api/config.ts`
 *  serialises as null — a cleared field silently rewrote the setting. */
function commitText(e: Event) {
  const input = e.target as HTMLInputElement;
  const raw = input.value;
  if (raw === String(props.row.value ?? "")) return;
  if (isNumber.value) {
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      // Show what the engine still holds rather than leaving a blank field
      // that looks saved.
      input.value = String(props.row.value ?? "");
      return;
    }
    emit("save", parsed);
    return;
  }
  emit("save", raw);
}
function commitBool(e: Event) {
  emit("save", (e.target as HTMLInputElement).checked);
}
</script>

<template>
  <!-- Rows are full-bleed inside the section card, which has no padding of
       its own; the inset is the row's, exactly as `SettingRow`'s is. -->
  <div class="border-border flex items-center gap-3 border-b px-3.5 py-2 last:border-b-0">
    <div class="min-w-0 flex-1">
      <div class="font-mono truncate text-sm text-fg" :title="row.key">{{ row.key }}</div>
      <div v-if="locked" class="text-micro text-fg-dim">
        Set by {{ row.env_var ?? "an environment variable" }} — unset it to edit here.
      </div>
    </div>

    <!-- editor -->
    <label v-if="isBool" class="flex items-center">
      <input
        type="checkbox"
        :checked="!!draft"
        :disabled="locked"
        class="accent-accent disabled:opacity-50"
        @change="commitBool"
      />
    </label>
    <input
      v-else
      :value="draft ?? ''"
      :type="isNumber ? 'number' : 'text'"
      :disabled="locked"
      data-selectable
      class="border-border font-mono text-xs h-7 w-48 rounded-control border bg-bg-deep px-1.5 text-fg disabled:opacity-50"
      @keydown.enter="commitText"
      @blur="commitText"
    />

    <!-- provenance tag -->
    <span
      class="font-mono text-micro text-fg-dim whitespace-nowrap w-14 shrink-0 text-right"
      :title="tag.label"
    >
      {{ tag.glyph }} {{ tag.label.toUpperCase() }}
    </span>

    <!-- reset (db rows only) -->
    <button
      v-if="row.source === 'db'"
      type="button"
      class="w-12 shrink-0 text-micro text-fg-dim hover:text-fg"
      title="Reset to its default"
      @click="emit('reset')"
    >
      Reset
    </button>
    <span v-else class="w-12 shrink-0" />
  </div>
</template>
