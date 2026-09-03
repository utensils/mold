<script setup lang="ts">
const props = defineProps<{
  modelValue: number | null;
  min?: number | undefined;
  max?: number | undefined;
  step?: number | undefined;
  placeholder?: string | undefined;
  disabled?: boolean | undefined;
  /** Accessible name — the control has no visible <label> associated with it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: number | null): void }>();

function commit(e: Event) {
  const raw = (e.target as HTMLInputElement).value.trim();
  if (raw === "") {
    emit("commit", null);
    return;
  }
  const n = Number(raw);
  if (!Number.isFinite(n) || n === props.modelValue) return;
  emit("commit", n);
}
</script>

<template>
  <input
    type="number"
    :value="modelValue ?? ''"
    :min="min"
    :max="max"
    :step="step"
    :placeholder="placeholder"
    :disabled="disabled"
    :aria-label="ariaLabel"
    data-selectable
    class="border-border font-mono text-xs h-7 w-28 rounded-control border bg-bg-deep px-2 text-fg placeholder:text-fg-dim disabled:opacity-40"
    @change="commit"
    @keydown.enter="($event.target as HTMLInputElement).blur()"
  />
</template>
