<script setup lang="ts">
/*
 * A raw engine-config row: the forward-compatibility path, and nothing else.
 *
 * `settingsSchema.contract.test.ts` fails when any key in the engine's own
 * registry has no curated schema, so on a current machine this renders for
 * NOTHING. When it does render, "Server-provided configuration key." means
 * exactly one thing: a key newer than this client.
 *
 * It commits on blur, so every row a person tabs past is a write unless it
 * guards — and on a numeric row `Number("")` is 0 while `Number("abc")` is
 * NaN, which the config client serialises as null. A cleared field silently
 * rewrote the setting.
 */
import { computed, ref, watch } from "vue";
import { isRowLocked, provenance, type ConfigRow } from "../../api/config";

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
  (value) => (draft.value = value),
);

function commitText(event: Event) {
  const input = event.target as HTMLInputElement;
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

function commitBool(event: Event) {
  emit("save", (event.target as HTMLInputElement).checked);
}
</script>

<template>
  <div class="ms-raw-row">
    <div class="ms-raw-row__text">
      <div class="ms-raw-row__key" :title="row.key">{{ row.key }}</div>
      <p class="ms-raw-row__help">Server-provided configuration key.</p>
      <p v-if="locked" class="ms-raw-row__help">
        Set by {{ row.env_var ?? "an environment variable" }} — unset it to edit
        here.
      </p>
    </div>

    <label v-if="isBool" class="ms-raw-row__check">
      <input
        type="checkbox"
        :checked="!!draft"
        :disabled="locked"
        :aria-label="row.key"
        @change="commitBool"
      />
    </label>
    <input
      v-else
      class="ms-raw-row__field"
      :value="draft ?? ''"
      :type="isNumber ? 'number' : 'text'"
      :disabled="locked"
      :aria-label="row.key"
      data-selectable
      @keydown.enter="commitText"
      @blur="commitText"
    />

    <span class="ms-raw-row__tag" :title="tag.label">
      {{ tag.glyph }} {{ tag.label.toUpperCase() }}
    </span>

    <button
      v-if="row.source === 'db'"
      type="button"
      class="ms-raw-row__reset"
      data-test="raw-row-reset"
      title="Reset to its default"
      @click="emit('reset')"
    >
      Reset
    </button>
    <span v-else class="ms-raw-row__reset" />
  </div>
</template>

<style scoped>
.ms-raw-row {
  display: flex;
  align-items: center;
  gap: var(--mold-sp-3);
  min-height: var(--mold-row-h-table, 52px);
  padding: var(--mold-sp-2) var(--mold-sp-3);
  border-bottom: var(--mold-bw) solid var(--mold-border);
}
.ms-raw-row:last-child {
  border-bottom: none;
}
.ms-raw-row__text {
  flex: 1 1 auto;
  min-width: 0;
}
.ms-raw-row__key {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-sm);
}
.ms-raw-row__help {
  margin: 0;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
}
.ms-raw-row__check {
  display: flex;
  align-items: center;
}
.ms-raw-row__check input {
  accent-color: var(--mold-blue);
}
.ms-raw-row__field {
  width: 192px;
  height: var(--mold-ctl-lg, 32px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
}
.ms-raw-row__field:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}
.ms-raw-row__field:disabled {
  opacity: 0.5;
}
.ms-raw-row__tag {
  width: 56px;
  flex: none;
  text-align: right;
  white-space: nowrap;
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}
.ms-raw-row__reset {
  width: 48px;
  flex: none;
  border: none;
  background: none;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
  cursor: pointer;
}
.ms-raw-row__reset:hover {
  color: var(--mold-text);
}
</style>
