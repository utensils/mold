<script setup lang="ts">
import { ipc } from "../../lib/ipc";

const props = defineProps<{ modelValue: string; title: string; disabled?: boolean | undefined }>();
const emit = defineEmits<{ (e: "commit", value: string): void }>();

async function pick() {
  const dir = await ipc.pickDirectory(props.title);
  if (dir && dir !== props.modelValue) emit("commit", dir);
}
</script>

<template>
  <div class="flex items-center gap-2">
    <span
      class="data-mono max-w-64 truncate text-caption text-ink-2"
      :title="props.modelValue"
      dir="rtl"
    >
      {{ props.modelValue || "—" }}
    </span>
    <button
      type="button"
      :disabled="disabled"
      class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink disabled:opacity-40"
      @click="pick"
    >
      Choose…
    </button>
  </div>
</template>
