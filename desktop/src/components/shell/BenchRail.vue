<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useConnectionStore } from "../../stores/connection";
import { useToastStore } from "../../stores/toasts";
import { apiJson } from "../../lib/api/client";
import { sseStream } from "../../lib/api/sse";
import { formatGB, percent, vramLevel } from "../../lib/format";
import { shouldRestartEmbeddedEngine } from "../../lib/connectionRecovery";
import type { ResourceSnapshot, ServerStatus } from "../../lib/api/types";

const conn = useConnectionStore();
const snapshot = ref<ResourceSnapshot | null>(null);
const status = ref<ServerStatus | null>(null);

let abort: AbortController | null = null;
let statusTimer: ReturnType<typeof setInterval> | null = null;

const gpu = computed(() => snapshot.value?.gpus[0] ?? null);
const vramPct = computed(() =>
  gpu.value ? percent(gpu.value.vram_used, gpu.value.vram_total) : 0,
);
const vramCritical = computed(
  () => gpu.value !== null && vramLevel(gpu.value.vram_used, gpu.value.vram_total) === "critical",
);

const engineChip = computed(() => {
  if (conn.status === "starting") return "⌁ starting…";
  if (conn.status === "error") return "⌁ engine error";
  switch (conn.mode) {
    case "local":
      return "⌁ local";
    case "external":
      return "⌁ local (shared)";
    case "remote":
      return `⇄ ${conn.baseUrl?.replace(/^https?:\/\//, "") ?? "remote"}`;
    default:
      return "⌁ engine off";
  }
});

let statusFailures = 0;

async function refreshStatus() {
  if (!conn.ready) return;
  try {
    status.value = await apiJson<ServerStatus>("/api/status");
    statusFailures = 0;
  } catch {
    // Two consecutive failures = the engine is gone, not a blip. For the
    // built-in engine, restart it (the backend detects the dead thread);
    // remote hosts surface the error chip instead.
    statusFailures += 1;
    if (shouldRestartEmbeddedEngine(conn.mode, statusFailures)) {
      statusFailures = 0;
      status.value = null;
      await conn.useLocal();
      if (conn.ready) {
        useToastStore().push("Engine restarted");
        startTelemetry();
      }
    }
  }
}

function startTelemetry() {
  stopTelemetry();
  abort = new AbortController();
  void sseStream("/api/resources/stream", {
    signal: abort.signal,
    onEvent(event, data) {
      if (event === "snapshot") {
        try {
          snapshot.value = JSON.parse(data) as ResourceSnapshot;
        } catch {
          /* skip malformed frame */
        }
      }
    },
  });
  void refreshStatus();
  statusTimer = setInterval(refreshStatus, 10_000);
}

function stopTelemetry() {
  abort?.abort();
  abort = null;
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = null;
  snapshot.value = null;
  status.value = null;
}

watch(
  () => conn.ready,
  (ready) => (ready ? startTelemetry() : stopTelemetry()),
);
onMounted(() => {
  if (conn.ready) startTelemetry();
});
onUnmounted(stopTelemetry);
</script>

<template>
  <footer
    class="border-edge flex items-center gap-4 border-t bg-bath px-3"
    role="complementary"
    aria-label="Engine and resource status"
  >
    <span
      class="data-mono"
      :class="conn.status === 'error' ? 'text-stop' : conn.ready ? 'text-ink-2' : 'text-ink-3'"
      :title="conn.error ?? conn.baseUrl ?? undefined"
      role="status"
      aria-live="polite"
    >
      <span class="sr-only">Engine status: </span>{{ engineChip }}
    </span>

    <template v-if="gpu">
      <span class="data-mono text-ink-3">{{ gpu.name }}</span>
      <div class="flex items-center gap-2" :title="`VRAM ${vramPct.toFixed(0)}%`">
        <div class="h-1.5 w-20 overflow-hidden rounded-full bg-bench">
          <div
            class="h-full transition-[width] duration-500"
            :class="vramCritical ? 'bg-stop' : 'bg-halide'"
            :style="{ width: `${vramPct}%` }"
          />
        </div>
        <span class="data-mono text-ink-3">
          {{ formatGB(gpu.vram_used) }}/{{ formatGB(gpu.vram_total) }}
        </span>
      </div>
    </template>
    <span v-if="snapshot" class="data-mono text-ink-3">
      RAM {{ formatGB(snapshot.system_ram.used) }}
    </span>

    <div class="flex-1" />
    <span v-if="status?.models_loaded?.length" class="data-mono text-ink-3">
      {{ status.models_loaded.join(" · ") }}
    </span>
    <span class="edge-code">
      QUEUE {{ status?.queue_depth ?? "—"
      }}<template v-if="status?.queue_capacity">/{{ status.queue_capacity }}</template>
    </span>
  </footer>
</template>
