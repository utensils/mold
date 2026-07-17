<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, onUnmounted, ref } from "vue";
import type { HostView } from "../../stores/hosts";

defineProps<{ modelName: string; hosts: HostView[] }>();
const emit = defineEmits<{
  (e: "select", host: HostView): void;
  (e: "close"): void;
}>();
const closeBtn = ref<HTMLButtonElement | null>(null);
let restoreFocusEl: HTMLElement | null = null;

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") emit("close");
}

onMounted(() => {
  restoreFocusEl = document.activeElement as HTMLElement | null;
  window.addEventListener("keydown", onKeydown);
  void nextTick(() => closeBtn.value?.focus());
});
onUnmounted(() => window.removeEventListener("keydown", onKeydown));
onBeforeUnmount(() => restoreFocusEl?.focus?.());
</script>

<template>
  <Teleport to="body">
    <div
      class="fixed inset-0 z-40 flex items-center justify-center bg-black/55 p-5"
      @click.self="emit('close')"
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="download-target-title"
        class="border-edge z-50 w-full max-w-md rounded-chrome border bg-bench p-4 shadow-raised"
      >
        <div class="mb-3 flex items-start justify-between gap-4">
          <div>
            <h2 id="download-target-title" class="text-body-lg font-semibold text-ink">
              Choose where to download {{ modelName }}
            </h2>
            <p class="mt-1 text-caption text-ink-2">
              The model and its required components will be stored on the selected host.
            </p>
          </div>
          <button
            ref="closeBtn"
            type="button"
            class="h-7 rounded-control px-2 text-ink-2 hover:bg-bath hover:text-ink"
            aria-label="Close download target picker"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <div class="flex flex-col gap-1" role="list">
          <button
            v-for="host in hosts"
            :key="host.id"
            type="button"
            role="listitem"
            :data-test="`download-target-${host.id}`"
            class="border-edge flex min-h-12 items-center gap-3 rounded-control border px-3 py-2 text-left transition-colors duration-150 hover:border-safelight hover:bg-bath"
            @click="emit('select', host)"
          >
            <span
              class="h-2 w-2 shrink-0 rounded-full"
              :class="host.status === 'ready' ? 'bg-halide' : 'bg-stop'"
              aria-hidden="true"
            />
            <span class="min-w-0 flex-1">
              <span class="block truncate text-body font-medium text-ink">{{ host.label }}</span>
              <span class="block truncate text-caption text-ink-3">
                {{ host.kind === "local" ? "This device" : host.baseUrl }}
              </span>
            </span>
            <span v-if="host.primary" class="edge-code">Current</span>
            <span v-if="host.queueDepth != null" class="data-mono text-caption text-ink-3">
              {{ host.queueDepth }} queued
            </span>
          </button>
        </div>
      </section>
    </div>
  </Teleport>
</template>
