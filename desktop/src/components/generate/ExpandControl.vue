<script setup lang="ts">
import { ref } from "vue";
import { expandPrompt } from "../../lib/api/expand";
import { ApiError } from "../../lib/api/client";
import { useToastStore } from "../../stores/toasts";

const props = defineProps<{ prompt: string; family: string }>();
const emit = defineEmits<{
  (e: "apply", payload: { expanded: string; original: string }): void;
  (e: "restore", original: string): void;
}>();

const toasts = useToastStore();
const running = ref(false);
const lastOriginal = ref<string | null>(null);

async function expand() {
  const original = props.prompt.trim();
  if (!original || running.value) return;
  running.value = true;
  try {
    const res = await expandPrompt(original, props.family || undefined);
    const expanded = res.expanded[0]?.trim();
    if (!expanded) {
      toasts.push("Expansion returned nothing.", "error");
      return;
    }
    lastOriginal.value = original;
    emit("apply", { expanded, original });
  } catch (err) {
    if (err instanceof ApiError && (err.status === 404 || err.status === 503)) {
      toasts.push("Expansion model isn't installed.", "error");
    } else {
      toasts.push("Couldn't expand the prompt.", "error");
    }
  } finally {
    running.value = false;
  }
}

function undo() {
  if (lastOriginal.value === null) return;
  emit("restore", lastOriginal.value);
  lastOriginal.value = null;
}

// ⌘E in the composer triggers the same expand.
defineExpose({ expand });
</script>

<template>
  <div class="flex items-center gap-1">
    <button
      type="button"
      class="border-edge h-7 rounded-control border px-2 text-body text-ink-2 hover:text-ink disabled:opacity-50"
      :disabled="running || !prompt.trim()"
      title="Expand prompt"
      @click="expand"
    >
      {{ running ? "Expanding…" : "Expand" }}
      <kbd v-if="!running" class="data-mono ml-1 opacity-60">⌘E</kbd>
    </button>
    <button
      v-if="lastOriginal !== null"
      type="button"
      class="h-7 rounded-control px-1 text-body text-halide hover:text-ink"
      title="Restore original prompt"
      @click="undo"
    >
      ↩
    </button>
  </div>
</template>
