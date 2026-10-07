<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import type { ApiTarget } from "../api/client";
import {
  queueInputs,
  queueSourceThumbnail,
  type QueueInput,
} from "../api/queueSourceThumbnail";

const props = withDefaults(
  defineProps<{
    target?: ApiTarget | null | undefined;
    instanceId?: string | null | undefined;
    jobId: string;
    online?: boolean | undefined;
    detailed?: boolean | undefined;
  }>(),
  { online: true },
);
const failed = ref(false);
const retry = ref(0);
const items = ref<(QueueInput & { url?: string })[]>([]);
const visible = computed(() =>
  items.value.length === 0
    ? []
    : props.detailed
      ? items.value
      : [
          items.value.find((item) => item.url) ??
            items.value.find((item) => item.preview) ??
            items.value[0]!,
        ],
);
let controller: AbortController | null = null;
function release() {
  controller?.abort();
  controller = null;
  for (const item of items.value) if (item.url) URL.revokeObjectURL(item.url);
  items.value = [];
  failed.value = false;
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
    () => props.detailed,
    () => retry.value,
  ],
  async (_, __, onCleanup) => {
    release();
    const target = props.target;
    if (!target || !props.jobId || props.online === false) return;
    const request = new AbortController();
    controller = request;
    onCleanup(() => request.abort());
    try {
      const descriptors = await queueInputs(
        target,
        props.jobId,
        request.signal,
      );
      if (request.signal.aborted) return;
      items.value = descriptors;
      const previews = descriptors;
      for (const item of previews) {
        if (!item.preview) continue;
        try {
          const blob = await queueSourceThumbnail(
            target,
            props.jobId,
            request.signal,
            item.index,
          );
          if (request.signal.aborted) return;
          const current = items.value.find(
            (candidate) => candidate.index === item.index,
          );
          if (current) current.url = URL.createObjectURL(blob);
          if (!props.detailed) break;
        } catch {
          if (request.signal.aborted) return;
        }
      }
      if (
        descriptors.length === 1 &&
        descriptors[0]?.index === undefined &&
        !items.value[0]?.url
      )
        items.value = [];
    } catch {
      if (!request.signal.aborted) failed.value = true;
      // Missing/corrupt inputs never masquerade as a denoise preview.
    }
  },
  { immediate: true },
);
onBeforeUnmount(release);
</script>

<template>
  <div
    v-if="detailed ? items.length : visible.some((item) => item.url)"
    class="queue-inputs"
    :class="{ 'queue-inputs--detail': detailed }"
  >
    <figure
      v-for="item in visible"
      :key="item.index ?? 'legacy'"
      class="queue-source-thumbnail"
    >
      <img
        v-if="item.url"
        :src="item.url"
        :alt="
          item.label === 'Source' ? 'Source image for this render' : item.label
        "
        decoding="async"
      />
      <figcaption>
        {{ item.label
        }}<template v-if="!detailed && items.length > 1">
          +{{ items.length - 1 }}</template
        >
      </figcaption>
      <span v-if="detailed && !item.url" class="queue-input-notice">{{
        item.preview ? "Preview unavailable" : "No still preview"
      }}</span>
    </figure>
    <button
      v-if="detailed && items.some((item) => item.preview && !item.url)"
      type="button"
      @click="retry++"
    >
      Retry input previews
    </button>
  </div>
  <div v-else-if="detailed && failed" role="status">
    Input previews unavailable.
    <button type="button" @click="retry++">Try again</button>
  </div>
  <slot v-else />
</template>

<style scoped>
.queue-inputs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}
.queue-inputs--detail .queue-source-thumbnail {
  width: min(100%, 16rem);
}
.queue-input-notice {
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
}
.queue-source-thumbnail {
  margin: 0;
  flex: none;
  width: 3.5rem;
}
.queue-source-thumbnail img {
  display: block;
  width: 100%;
  aspect-ratio: 1;
  object-fit: contain;
  border-radius: var(--mold-radius-3);
}
.queue-source-thumbnail figcaption {
  margin-top: 0.25rem;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-xs);
}
</style>
