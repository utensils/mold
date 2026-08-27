<script setup lang="ts">
/**
 * Right-side info panel for one server-queue row — the queue counterpart of
 * the model detail drawer. The panel is chrome only: the shared
 * `QueueEntryDetail` decides what a queued job says about itself so desktop,
 * web, and iPhone cannot disagree, and the actions are emitted for
 * `HostQueuePanel` to route to the exact selected host.
 */
import { onMounted, onUnmounted } from "vue";
import QueueEntryDetail from "@studio/components/QueueEntryDetail.vue";
import type { QueueEntryDetailModel } from "@studio/lib/queueEntryDetail";
import type { QueueJobProgress } from "@studio/api/generationSelection";

defineProps<{
  model: QueueEntryDetailModel;
  preview?: QueueJobProgress | null;
  cancelling?: boolean;
  retrying?: boolean;
  error?: string | null;
}>();
const emit = defineEmits<{
  (e: "close"): void;
  (e: "reuse"): void;
  (e: "cancel"): void;
  (e: "retry"): void;
}>();

function onKeydown(event: KeyboardEvent): void {
  if (event.key === "Escape") emit("close");
}
onMounted(() => window.addEventListener("keydown", onKeydown));
onUnmounted(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <aside
    class="border-edge fixed inset-y-0 right-0 z-40 flex w-96 max-w-full flex-col border-l bg-bench shadow-raised"
    role="dialog"
    aria-modal="false"
    :aria-label="`Queued job — ${model.modelLabel}`"
    data-test="queue-entry-drawer"
  >
    <QueueEntryDetail
      :model="model"
      :preview="preview ?? null"
      :cancelling="cancelling ?? false"
      :retrying="retrying ?? false"
      :error="error ?? null"
      confirm="delegate"
      @close="emit('close')"
      @reuse="emit('reuse')"
      @cancel="emit('cancel')"
      @retry="emit('retry')"
    />
  </aside>
</template>
