<script setup lang="ts">
import { ref, watch } from "vue";
import type { ApiTarget } from "../api/client";
import {
  inspectQueueBatch,
  cancelInspectedQueueBatch,
  type QueueBatchSelection,
} from "../api/queueBatchCancel";
const props = defineProps<{
  target?: ApiTarget | null | undefined;
  instanceId?: string | null | undefined;
  jobId: string;
  online?: boolean | undefined;
  disabled?: boolean;
}>();
const emit = defineEmits<{ cancelled: [] }>();
const selection = ref<QueueBatchSelection | null>(null);
const armed = ref(false);
const busy = ref(false);
const error = ref<string | null>(null);
let revision = 0;
watch(
  [
    () => props.target?.baseUrl,
    () => props.target?.apiKey,
    () => props.target?.connectionMode,
    () => props.instanceId,
    () => props.jobId,
    () => props.online,
  ],
  async (_, __, onCleanup) => {
    const version = ++revision;
    onCleanup(() => {
      ++revision;
    });
    selection.value = null;
    armed.value = false;
    error.value = null;
    const target = props.target;
    if (!target || !props.instanceId || props.online !== true) return;
    try {
      const result = await inspectQueueBatch(
        target,
        props.instanceId,
        props.jobId,
      );
      if (version === revision) selection.value = result;
    } catch {
      /* Legacy hosts without batch lookup keep individual cancellation. */
    }
  },
  { immediate: true },
);
async function cancel(): Promise<void> {
  const selected = selection.value;
  const target = props.target;
  if (
    !selected ||
    !target ||
    props.disabled ||
    busy.value ||
    props.online !== true
  )
    return;
  if (!armed.value) {
    armed.value = true;
    return;
  }
  armed.value = false;
  busy.value = true;
  error.value = null;
  const version = revision;
  try {
    await cancelInspectedQueueBatch(
      target,
      selected,
      () => version === revision && !props.disabled && props.online === true,
    );
    if (version === revision) {
      selection.value = null;
      emit("cancelled");
    }
  } catch (failure) {
    if (version === revision)
      error.value =
        failure instanceof Error ? failure.message : String(failure);
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <div v-if="selection" class="queue-batch-cancel">
    <button
      type="button"
      class="qed__danger"
      data-test="queue-detail-cancel-batch"
      :disabled="disabled || busy"
      @click="cancel"
      @blur="armed = false"
    >
      {{
        busy
          ? "Cancelling batch…"
          : armed
            ? "Cancel entire batch?"
            : `Cancel batch (${selection.liveCount} jobs)`
      }}
    </button>
    <p v-if="error" role="alert">{{ error }}</p>
  </div>
</template>
<style scoped>
.queue-batch-cancel button {
  min-height: 44px;
  padding: 8px 12px;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  background: var(--mold-surface);
  color: var(--mold-error);
  font: inherit;
  cursor: pointer;
}
.queue-batch-cancel button:disabled {
  opacity: 0.5;
  cursor: default;
}
.queue-batch-cancel p {
  margin: 4px 0;
  color: var(--mold-error);
}
</style>
