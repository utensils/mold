<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from "vue";
import type { ApiTarget } from "../api/client";
import { queueSourceThumbnail } from "../api/queueSourceThumbnail";

const props = withDefaults(
  defineProps<{
    target?: ApiTarget | null | undefined;
    instanceId?: string | null | undefined;
    jobId: string;
    online?: boolean | undefined;
  }>(),
  { online: true },
);
const url = ref<string | null>(null);
let controller: AbortController | null = null;
function release() {
  controller?.abort();
  controller = null;
  if (url.value) URL.revokeObjectURL(url.value);
  url.value = null;
}
watch(
  // Compare each primitive identity separately. Polling creates fresh target
  // objects; their identity must not restart an unchanged media request.
  [
    () => props.target?.baseUrl,
    () => props.target?.apiKey,
    () => props.instanceId,
    () => props.jobId,
    () => props.online,
  ],
  async (_, __, onCleanup) => {
    release();
    const target = props.target;
    if (!target || !props.jobId || props.online === false) return;
    const request = new AbortController();
    controller = request;
    onCleanup(() => request.abort());
    try {
      const blob = await queueSourceThumbnail(
        target,
        props.jobId,
        request.signal,
      );
      if (!request.signal.aborted) url.value = URL.createObjectURL(blob);
    } catch {
      // Jobs without retained inputs and older hosts have no source preview.
    }
  },
  { immediate: true },
);
onBeforeUnmount(release);
</script>

<template>
  <figure v-if="url" class="queue-source-thumbnail">
    <img :src="url" alt="Source image for this render" decoding="async" />
    <figcaption>Source</figcaption>
  </figure>
  <slot v-else />
</template>

<style scoped>
.queue-source-thumbnail {
  margin: 0;
  flex: none;
  width: 3.5rem;
}
.queue-source-thumbnail img {
  display: block;
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--mold-radius-3);
}
.queue-source-thumbnail figcaption {
  margin-top: 0.25rem;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-xs);
}
</style>
