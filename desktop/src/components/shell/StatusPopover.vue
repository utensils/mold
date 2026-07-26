<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useGenerationStore } from "../../stores/generation";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import { apiJson } from "../../lib/api/client";
import { sseStream } from "../../lib/api/sse";
import { formatGB, percent, vramLevel } from "../../lib/format";
import { normalizeTargetHost, pickDisplayHost } from "../../lib/hosts";
import { shouldRestartEmbeddedEngine } from "../../lib/connectionRecovery";
import type { ResourceSnapshot, ServerStatus } from "../../lib/api/types";

defineProps<{ collapsed?: boolean }>();

const conn = useConnectionStore();
const appPrefs = useAppPrefsStore();
const generation = useGenerationStore();
const hosts = useHostsStore();
const snapshot = ref<ResourceSnapshot | null>(null);
const status = ref<ServerStatus | null>(null);
const open = ref(false);
const rootEl = ref<HTMLElement | null>(null);

let resourceAbort: AbortController | null = null;
let statusTimer: ReturnType<typeof setInterval> | null = null;

/**
 * The status mirrors the concrete host selected in the Create header. Auto
 * and Most capable have no concrete host until routing happens, so those modes
 * follow the most recently submitted live job with a routed host id and
 * otherwise show the primary. The embedded-engine recovery poll below stays
 * bound to the primary connection no matter what is displayed.
 */
const displayHost = computed(() => {
  const selected = normalizeTargetHost(appPrefs.settings?.generateTargetHost, hosts.all);
  if (selected && selected !== "capable") {
    return hosts.all.find((host) => host.id === selected) ?? hosts.primaryHost;
  }
  const liveIds = generation.jobs
    .filter((j) => j.status !== "complete" && j.status !== "error")
    .map((j) => j.hostId);
  const primaryId = hosts.primaryHost?.id ?? "local";
  const id = pickDisplayHost(liveIds, primaryId);
  return hosts.all.find((h) => h.id === id) ?? hosts.primaryHost;
});
const displayingRemote = computed(() => !!displayHost.value && !displayHost.value.primary);
const anyJobRunning = computed(() =>
  generation.jobs.some((j) => j.status !== "complete" && j.status !== "error"),
);

/** MB-based `/api/status` GPU summary as a snapshot-shaped fallback, for
 *  display hosts whose resources stream is unavailable (older servers). */
const telemetryGpu = computed(() => {
  const host = displayHost.value;
  if (!host) return null;
  const info = displayingRemote.value ? hosts.telemetry[host.id]?.gpuInfo : status.value?.gpu_info;
  if (!info) return null;
  return {
    name: info.name,
    vram_total: info.vram_total_mb * 1_000_000,
    vram_used: info.vram_used_mb * 1_000_000,
  };
});

/** Every GPU on the display host — the popover gets one VRAM row per GPU and
 *  the trigger mini-bar aggregates them, so a job denoising on any ordinal is
 *  visible (jobs land on whichever worker is free, not GPU 0). */
const gpus = computed(() => {
  if (snapshot.value?.gpus.length) return snapshot.value.gpus;
  const t = telemetryGpu.value;
  return t ? [{ ordinal: 0, ...t }] : [];
});
const vramUsed = computed(() => gpus.value.reduce((sum, g) => sum + g.vram_used, 0));
const vramTotal = computed(() => gpus.value.reduce((sum, g) => sum + g.vram_total, 0));
const vramPct = computed(() => (gpus.value.length ? percent(vramUsed.value, vramTotal.value) : 0));
const vramCritical = computed(() =>
  gpus.value.some((g) => vramLevel(g.vram_used, g.vram_total) === "critical"),
);
/** Halide when idle/latent, safelight while a job develops, stop past 92%. */
const vramTone = computed(() => {
  if (vramCritical.value) return "danger";
  return anyJobRunning.value ? "accent" : "info";
});
function gpuTone(used: number, total: number) {
  if (vramLevel(used, total) === "critical") return "danger";
  return anyJobRunning.value ? "accent" : "info";
}

const engineChip = computed(() => {
  if (conn.status === "starting") return "⌁ starting…";
  if (conn.status === "error") return "⌁ engine error";
  if (displayingRemote.value) return `⇄ ${displayHost.value!.label}`;
  switch (conn.mode) {
    case "local":
      return "⌁ local";
    case "external":
      return "⌁ local (shared)";
    default:
      return "⌁ engine off";
  }
});

/** Trigger dot color mirrors the engine/host status. */
const dotClass = computed(() => {
  if (conn.status === "error") return "bg-stop";
  if (conn.status === "starting") return "bg-halide animate-pulse";
  if (displayingRemote.value) return "bg-halide";
  return conn.ready ? "bg-success" : "bg-ink-3";
});

/** Queue + loaded models for whichever host the status is displaying. */
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

function toggle() {
  open.value = !open.value;
}
function onDocPointer(e: MouseEvent) {
  if (open.value && rootEl.value && !rootEl.value.contains(e.target as Node)) open.value = false;
}
function onKey(e: KeyboardEvent) {
  if (e.key === "Escape" && open.value) open.value = false;
}

onMounted(() => {
  if (conn.ready) startTelemetry();
  document.addEventListener("mousedown", onDocPointer);
  document.addEventListener("keydown", onKey);
});
onUnmounted(() => {
  stopTelemetry();
  document.removeEventListener("mousedown", onDocPointer);
  document.removeEventListener("keydown", onKey);
});
</script>

<template>
  <div ref="rootEl" class="relative">
    <!-- popover: engine + telemetry, contained inside the app frame (§05) -->
    <div
      v-if="open"
      class="ms-fade-up absolute bottom-[calc(100%+8px)] left-0 z-50 w-64 rounded-card border border-ce bg-bench p-3.5 shadow-raised"
      role="dialog"
      aria-label="Engine and resource status"
    >
      <div class="mb-2.5 flex items-center justify-between">
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
        >
          {{ engineChip }}
        </span>
        <span class="edge-code">
          queue {{ displayStatus.queueDepth ?? "—"
          }}<template v-if="displayStatus.queueCapacity"
            >/{{ displayStatus.queueCapacity }}</template
          >
        </span>
      </div>

      <template v-if="gpus.length">
        <div
          v-for="(g, i) in gpus"
          :key="g.ordinal"
          data-test="gpu-row"
          :class="i > 0 ? 'mt-2.5' : ''"
        >
          <div class="mb-1 flex min-w-0 items-baseline gap-2">
            <span v-if="gpus.length > 1" class="edge-code shrink-0">GPU {{ g.ordinal }}</span>
            <span class="min-w-0 truncate text-caption text-ink-2" :title="g.name">{{
              g.name
            }}</span>
          </div>
          <ProgressBar
            :value="percent(g.vram_used, g.vram_total)"
            :tone="gpuTone(g.vram_used, g.vram_total)"
            :height="6"
            :label="`VRAM in use on ${g.name}`"
          />
          <div class="mt-1.5 flex items-center justify-between">
            <span class="edge-code">vram</span>
            <span class="data-mono text-ink-3">
              {{ formatGB(g.vram_used) }} / {{ formatGB(g.vram_total) }}
            </span>
          </div>
        </div>
      </template>
      <p v-else class="text-caption text-ink-3">no gpu telemetry</p>

      <div v-if="snapshot" class="mt-2 flex items-center justify-between">
        <span class="edge-code">ram</span>
        <span class="data-mono text-ink-3">{{ formatGB(snapshot.system_ram.used) }}</span>
      </div>

      <template v-if="displayStatus.modelsLoaded.length">
        <div class="mt-3 mb-1 border-t border-edge pt-2">
          <span class="edge-code">loaded</span>
        </div>
        <div class="flex flex-col gap-1">
          <div v-for="m in displayStatus.modelsLoaded" :key="m" class="flex items-center gap-2">
            <span class="h-1.5 w-1.5 shrink-0 rounded-full bg-safelight" />
            <span class="min-w-0 truncate text-caption text-ink-2">{{ m }}</span>
          </div>
        </div>
      </template>
    </div>

    <!-- trigger: always visible; mini VRAM + queue when expanded, dot only when collapsed -->
    <button
      type="button"
      data-test="status-trigger"
      class="flex h-8 w-full items-center rounded-[9px] hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
      :class="collapsed ? 'justify-center px-0' : 'gap-2.5 px-2.5'"
      :aria-expanded="open"
      aria-label="Engine and resource status"
      :title="engineChip"
      @click="toggle"
    >
      <span class="h-[7px] w-[7px] shrink-0 rounded-full" :class="dotClass" />
      <template v-if="!collapsed">
        <span v-if="gpus.length" class="min-w-0 flex-1">
          <ProgressBar :value="vramPct" :tone="vramTone" :height="4" label="VRAM in use" />
        </span>
        <span v-else class="flex-1 text-left text-caption text-ink-3">status</span>
        <span class="shrink-0 font-utility text-[9.5px] text-ink-3">
          {{ displayStatus.queueDepth ?? 0 }}
        </span>
      </template>
    </button>
  </div>
</template>
