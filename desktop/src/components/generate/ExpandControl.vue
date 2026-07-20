<script setup lang="ts">
import { computed } from "vue";
import { shortcutLabel } from "../../lib/platform";

const props = defineProps<{
  prompt: string;
  batchSize: number;
  running: boolean;
  hostLabel: string | null;
  canUndo: boolean;
  blocked?: boolean;
}>();

const emit = defineEmits<{
  (e: "expand"): void;
  (e: "restore"): void;
}>();

const isPreparedBatch = computed(() => props.batchSize > 1);
const actionLabel = computed(() =>
  isPreparedBatch.value ? `Prepare ${props.batchSize} variations` : "Expand",
);
const progressLabel = computed(() => {
  const host = props.hostLabel ?? "the selected host";
  return isPreparedBatch.value
    ? `Expanding ${props.batchSize} prompts on ${host}…`
    : `Expanding on ${host}…`;
});

function expand() {
  if (!props.blocked && !props.running && props.prompt.trim()) emit("expand");
}

defineExpose({ expand });
</script>

<template>
  <div class="flex min-w-0 flex-wrap items-center gap-2">
    <button
      type="button"
      data-test="expand-action"
      class="border-edge min-h-7 rounded-control border px-2 text-body text-ink-2 transition-colors duration-100 hover:border-safelight hover:text-ink active:translate-y-px disabled:opacity-50"
      :disabled="blocked || running || !prompt.trim()"
      :title="
        blocked
          ? 'Refresh or discard the preserved prepared batch first'
          : isPreparedBatch
            ? `Prepare ${batchSize} prompt variations`
            : 'Expand prompt'
      "
      @click="expand"
    >
      {{ actionLabel }}
      <kbd v-if="!running" class="kbd-hint ml-1">{{ shortcutLabel("E") }}</kbd>
    </button>

    <button
      v-if="!isPreparedBatch && canUndo"
      type="button"
      class="min-h-7 rounded-control px-1.5 text-body text-halide transition-colors duration-100 hover:text-ink"
      title="Restore original prompt"
      aria-label="Restore original prompt"
      @click="emit('restore')"
    >
      ↩
    </button>

    <span v-if="running" role="status" aria-live="polite" class="min-w-0 text-caption text-halide">
      {{ progressLabel }}
    </span>
  </div>
</template>
