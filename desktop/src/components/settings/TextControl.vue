<script setup lang="ts">
const props = defineProps<{
  modelValue: string;
  placeholder?: string | undefined;
  wide?: boolean | undefined;
  disabled?: boolean | undefined;
  /** Accessible name — the control has no visible <label> associated with it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: string): void }>();

function commit(e: Event) {
  const v = (e.target as HTMLInputElement).value;
  if (v !== props.modelValue) emit("commit", v);
}
</script>

<template>
  <input
    type="text"
    :value="modelValue"
    :placeholder="placeholder"
    :disabled="disabled"
    :aria-label="ariaLabel"
    data-selectable
    class="border-edge data-mono h-7 rounded-control border bg-bath px-2 text-ink placeholder:text-ink-3 disabled:opacity-40"
    :class="wide ? 'w-72' : 'w-44'"
    @change="commit"
    @keydown.enter="($event.target as HTMLInputElement).blur()"
  />
</template>
