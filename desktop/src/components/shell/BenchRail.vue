<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useConnectionStore } from "../../stores/connection";
import { useGenerationStore } from "../../stores/generation";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import { apiJson } from "../../lib/api/client";
import { sseStream } from "../../lib/api/sse";
import { formatGB, percent, vramLevel } from "../../lib/format";
import { pickDisplayHost } from "../../lib/hosts";
import { shouldRestartEmbeddedEngine } from "../../lib/connectionRecovery";
import type { ResourceSnapshot, ServerStatus } from "../../lib/api/types";

const conn = useConnectionStore();
const generation = useGenerationStore();
const hosts = useHostsStore();
const snapshot = ref<ResourceSnapshot | null>(null);
const status = ref<ServerStatus | null>(null);

let resourceAbort: AbortController | null = null;
let statusTimer: ReturnType<typeof setInterval> | null = null;

/**
 * The bar follows the action: while a routed job is live, it mirrors THAT
 * host (chip, VRAM, queue, models) and reverts to the primary once the last
 * remote job settles. The embedded-engine recovery poll below stays bound to
 * the primary connection no matter what is displayed.
 */
const displayHost = computed(() => {
  const liveIds = generation.jobs
    .filter((j) => j.status !== "complete" && j.status !== "error")
    .map((j) => j.hostId);
  const primaryId = hosts.primaryHost?.id ?? "local";
  const id = pickDisplayHost(liveIds, primaryId);
  return hosts.all.find((h) => h.id === id) ?? hosts.primaryHost;
});
const displayingRemote = computed(() => !!displayHost.value && !displayHost.value.primary);

/** MB-based `/api/status` GPU summary as a snapshot-shaped fallback, for
 *  display hosts whose resources stream is unavailable (older servers).
 *  Covers the primary too — a remote primary on an old server would
 *  otherwise show no GPU at all despite `gpu_info` sitting in `status`. */
const telemetryGpu = computed(() => {
  const host = displayHost.value;
  if (!host) return null;
  const info = displayingRemote.value ? hosts.telemetry[host.id]?.gpuInfo : status.value?.gpu_info;
  if (!info) return null;
  // Decimal MB, matching formatGB (/1e9) and the server's resources path
  // ("1 MiB = 1_000_000 for display consistency with the rest of mold") —
  // binary here would show ~5% more VRAM than the live stream.
  return {
    name: info.name,
    vram_total: info.vram_total_mb * 1_000_000,
    vram_used: info.vram_used_mb * 1_000_000,
  };
});

const gpu = computed(() => snapshot.value?.gpus[0] ?? telemetryGpu.value);
const vramPct = computed(() =>
  gpu.value ? percent(gpu.value.vram_used, gpu.value.vram_total) : 0,
);
const vramCritical = computed(
  () => gpu.value !== null && vramLevel(gpu.value.vram_used, gpu.value.vram_total) === "critical",
);

const engineChip = computed(() => {
  if (conn.status === "starting") return "⌁ starting…";
  if (conn.status === "error") return "⌁ engine error";
  if (displayingRemote.value) return `⇄ ${displayHost.value!.label}`;
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

/** Queue + loaded models for whichever host the bar is displaying. */
const displayStatus = computed(() => {
  if (displayingRemote.value) {
    const t = hosts.telemetry[displayHost.value!.id];
    return {
      modelsLoaded: t?.modelsLoaded ?? [],
      queueDepth: t?.queueDepth ?? null,
      queueCapacity: t?.queueCapacity ?? null,
    };
  }
  return {
    modelsLoaded: status.value?.models_loaded ?? [],
    queueDepth: status.value?.queue_depth ?? null,
    queueCapacity: status.value?.queue_capacity ?? null,
  };
});

let statusFailures = 0;

/** PRIMARY-only status poll — this drives embedded-engine recovery and must
 *  never be re-targeted at the display host. */
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

/** Live VRAM/RAM for the DISPLAY host; reopened whenever it changes. */
function startResourceStream() {
  resourceAbort?.abort();
  resourceAbort = new AbortController();
  snapshot.value = null;
  const host = displayHost.value;
  void sseStream("/api/resources/stream", {
    signal: resourceAbort.signal,
    // Default (primary) target unless the bar is following a remote host.
    ...(displayingRemote.value && host?.baseUrl
      ? { target: { baseUrl: host.baseUrl, apiKey: host.apiKey } }
      : {}),
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
}

function startTelemetry() {
  stopTelemetry();
  startResourceStream();
  void refreshStatus();
  statusTimer = setInterval(refreshStatus, 10_000);
}

function stopTelemetry() {
  resourceAbort?.abort();
  resourceAbort = null;
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = null;
  snapshot.value = null;
  status.value = null;
}

watch(
  () => conn.ready,
  (ready) => (ready ? startTelemetry() : stopTelemetry()),
);
// Only the resources stream follows the display host; the status poll stays
// on the primary (recovery invariant).
watch(
  () => displayHost.value?.id,
  () => {
    if (conn.ready) startResourceStream();
  },
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
      :class="
        conn.status === 'error'
          ? 'text-stop'
          : displayingRemote
            ? 'text-halide'
            : conn.ready
              ? 'text-ink-2'
              : 'text-ink-3'
      "
      :title="conn.error ?? displayHost?.baseUrl ?? conn.baseUrl ?? undefined"
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
    <span v-if="displayStatus.modelsLoaded.length" class="data-mono text-ink-3">
      {{ displayStatus.modelsLoaded.join(" · ") }}
    </span>
    <span class="edge-code">
      QUEUE {{ displayStatus.queueDepth ?? "—"
      }}<template v-if="displayStatus.queueCapacity">/{{ displayStatus.queueCapacity }}</template>
    </span>
  </footer>
</template>
