<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { ApiTarget } from "../api/client";
import {
  missingQueueModel,
  queueDownloadState,
  startQueueDownloadRecovery,
} from "../composables/useQueueDownloadRecovery";
const props = withDefaults(
  defineProps<{
    target?: ApiTarget | null | undefined;
    instance?: string | null | undefined;
    job: string;
    host: string;
    online?: boolean | undefined;
    controls?: boolean;
  }>(),
  { target: null, instance: null, online: true, controls: false },
);
const emit = defineEmits<{ available: [value: boolean] }>();
const available = ref(false);
const state = computed(() =>
  queueDownloadState(props.target, props.instance, props.job),
);
watch(
  () => [
    props.target?.baseUrl,
    props.target?.apiKey,
    props.instance,
    props.job,
    props.online,
    props.controls,
  ],
  async (_, __, onCleanup) => {
    let current = true;
    onCleanup(() => {
      current = false;
    });
    available.value = false;
    emit("available", false);
    if (!props.controls || !props.online || !props.target || !props.instance)
      return;
    async function probe() {
      const missing = await missingQueueModel(
        props.target!,
        props.instance!,
        props.job,
      ).catch(() => false);
      if (current) {
        available.value = missing;
        emit("available", missing);
      }
    }
    await probe();
    if (!current) return;
    const timer = setInterval(() => void probe(), 5000);
    onCleanup(() => {
      current = false;
      clearInterval(timer);
    });
  },
  { immediate: true },
);
function start() {
  if (props.target && props.instance && props.online)
    void startQueueDownloadRecovery(
      props.target,
      props.instance,
      props.job,
      props.host,
    );
}
</script>
<template>
  <span
    v-if="state || (controls && available)"
    class="queue-download"
    data-test="queue-download-feedback"
    @click.stop
    @keydown.stop
  >
    <span v-if="state" role="status" aria-live="polite">{{
      state.message
    }}</span>
    <progress
      v-if="state?.busy"
      :value="state.fraction ?? undefined"
      :max="1"
      aria-label="Model download"
    />
    <button
      v-if="controls && available"
      type="button"
      :disabled="state?.busy || !online"
      @click="start"
    >
      {{ state?.busy ? "Downloading…" : "Download and Retry" }}
    </button>
  </span>
</template>
<style scoped>
.queue-download {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}
[role="status"] {
  margin: 0;
  overflow-wrap: anywhere;
}
progress {
  width: 100%;
  height: 6px;
  accent-color: var(--mold-accent);
}
button {
  min-height: 44px;
  padding: 8px 12px;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-surface);
  color: var(--mold-fg);
  cursor: pointer;
}
button:disabled {
  cursor: default;
  opacity: 0.7;
}
</style>
