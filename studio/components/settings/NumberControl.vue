<script setup lang="ts">
/*
 * A number that writes on blur, which means every row a person TABS PAST is a
 * write. So it commits only a REAL change, and a blank or non-numeric field
 * snaps back to the value the engine still holds rather than being sent:
 * `Number("")` is 0 and `Number("abc")` is NaN, and either would silently
 * rewrite the setting (`ConfigRowItem.test.ts`'s original finding).
 */
const props = defineProps<{
  modelValue: number | null;
  min?: number | undefined;
  max?: number | undefined;
  step?: number | undefined;
  placeholder?: string | undefined;
  disabled?: boolean | undefined;
  /** Accessible name — the control has no visible <label> beside it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: number): void }>();

function commit(event: Event) {
  const input = event.target as HTMLInputElement;
  const raw = input.value.trim();
  const parsed = Number(raw);
  if (raw === "" || !Number.isFinite(parsed)) {
    // Show what the engine still holds rather than leaving a blank field that
    // looks saved.
    input.value = props.modelValue === null ? "" : String(props.modelValue);
    return;
  }
  if (parsed === props.modelValue) return;
  emit("commit", parsed);
}
</script>

<template>
  <input
    class="ms-setting-field"
    type="number"
    data-selectable
    :value="modelValue ?? ''"
    :min="min"
    :max="max"
    :step="step"
    :placeholder="placeholder"
    :disabled="disabled"
    :aria-label="ariaLabel"
    @change="commit"
    @keydown.enter="($event.target as HTMLInputElement).blur()"
  />
</template>

<style scoped>
.ms-setting-field {
  width: 112px;
  height: var(--mold-ctl-lg, 32px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
}
.ms-setting-field::placeholder {
  color: var(--mold-text-dim);
}
.ms-setting-field:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}
.ms-setting-field:disabled {
  opacity: 0.4;
}
</style>
