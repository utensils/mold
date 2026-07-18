<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import { RouterLink, useRoute, useRouter } from "vue-router";
import CatalogDetailDrawer from "../components/models/CatalogDetailDrawer.vue";
import DownloadsTray from "../components/models/DownloadsTray.vue";
import ModelTableRow from "../components/models/ModelTableRow.vue";
import RenameDialog from "../components/shell/RenameDialog.vue";
import { startCatalogDownload } from "../lib/api/catalog";
import { ApiError, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { installedModelToEntry } from "../lib/catalogDetail";
import { sseStream } from "../lib/api/sse";
import { formatEta, formatGB, formatUptime, percent, vramLevel } from "../lib/format";
import { inferBackendFromGpuName } from "../lib/hosts";
import { modelDiskBytes, modelSizeLabels } from "../lib/models";
import { modelSource } from "../lib/modelSource";
import { ipc } from "../lib/ipc";
import type { GpuSnapshot, ModelEntry, ResourceSnapshot, ServerStatus } from "../lib/api/types";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useDownloadsStore } from "../stores/downloads";
import { useGenerationStore } from "../stores/generation";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { enrichQueueEntries, useJobsStore, type EnrichedQueueEntry } from "../stores/jobs";
import { useToastStore } from "../stores/toasts";

/** `ResourceSnapshot` plus the additive `cpu` wire field (mold-core
 *  `CpuSnapshot`); `null`/absent until the server's sampler has two frames,
 *  and always absent on servers that predate CPU sampling. */
type DetailSnapshot = ResourceSnapshot & {
  cpu?: { cores: number; usage_percent: number } | null;
};

const route = useRoute();
const router = useRouter();
const appPrefs = useAppPrefsStore();
const downloads = useDownloadsStore();
const generation = useGenerationStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
const jobs = useJobsStore();
const toasts = useToastStore();

const hostId = computed(() => String(route.params.id ?? ""));
const host = computed(() => hosts.all.find((h) => h.id === hostId.value) ?? null);
const telemetry = computed(() => hosts.telemetry[hostId.value]);

// ── Live telemetry (this host's resources stream) ─────────────────────────

const snapshot = ref<DetailSnapshot | null>(null);
const status = ref<ServerStatus | null>(null);
let resourceAbort: AbortController | null = null;

function hostTarget(): ApiTarget | null {
  const h = host.value;
  return h?.baseUrl ? { baseUrl: h.baseUrl, apiKey: h.apiKey } : null;
}

/** Reopen the stream against the CURRENT host; prior subscription aborts.
 *  Keep the last frame mounted when only credentials change so reconnecting
 *  updates values in place instead of collapsing the telemetry layout. */
function startResourceStream(reset = false) {
  resourceAbort?.abort();
  resourceAbort = null;
  if (reset) snapshot.value = null;
  const target = hostTarget();
  if (!target) return;
  resourceAbort = new AbortController();
  void sseStream("/api/resources/stream", {
    signal: resourceAbort.signal,
    target,
    onEvent(event, data) {
      if (event === "snapshot") {
        try {
          snapshot.value = JSON.parse(data) as DetailSnapshot;
        } catch {
          /* skip malformed frame */
        }
      }
    },
  });
}

// ── Live queue (this host's server queue via the jobs store) ──────────────

let queueTimer: ReturnType<typeof setInterval> | null = null;

function tickQueue() {
  const current = host.value;
  if (current) void jobs.refreshHost(current);
}

/** Poll only THIS host's queue while the page is open (Jobs polls them all). */
function startQueuePolling() {
  if (queueTimer) clearInterval(queueTimer);
  tickQueue();
  queueTimer = setInterval(tickQueue, 5_000);
}

let statusAbort: AbortController | null = null;

/** One-shot status fetch for models-disk stats (and fresher queue fields).
 *  Guarded per request: the :id param retargets this reused component in
 *  place, and a slow host's late response must never populate the page of
 *  the host the user navigated to next. */
async function fetchStatus(reset = false) {
  statusAbort?.abort();
  const abort = new AbortController();
  statusAbort = abort;
  if (reset) status.value = null;
  const target = hostTarget();
  if (!target) return;
  try {
    const res = await apiJsonTo<ServerStatus>(target, "/api/status", { signal: abort.signal });
    if (!abort.signal.aborted) status.value = res;
  } catch {
    // Unreachable, superseded, or older server — the storage card simply
    // hides and the telemetry cards fall back to the polled gpu_info summary.
  }
}

function startReadyServices() {
  const current = host.value;
  if (current?.status !== "ready") return;
  void downloads.subscribe(current).catch(() => {
    // Host status and the model list still render if an older server lacks
    // the download stream; reconnect retries are owned by the SSE helper.
  });
  void hostModels.refresh(true);
}

// Clicking a model row opens the shared detail drawer against THIS host.
// Declared before the identity watch so its immediate run can reset it.
const detailModel = ref<ModelEntry | null>(null);
const drawerRepairing = ref(false);

// Connection identity retargeting is the only event that replaces page data.
// A health poll changing ready/error state must not clear and rebuild the
// telemetry, storage, or models sections: those components keep their last
// snapshot mounted while their own live sources update them in place.
watch(
  [hostId, () => host.value?.baseUrl, () => host.value?.apiKey],
  (identity, previous) => {
    const hostChanged = !previous || identity[0] !== previous[0] || identity[1] !== previous[1];
    // A drawer left open for the previous host must not retarget its model
    // (and Repair action) at the next one.
    if (hostChanged) detailModel.value = null;
    startResourceStream(hostChanged);
    void fetchStatus(hostChanged);
    startQueuePolling();
    startReadyServices();
  },
  { immediate: true },
);

// Reconnect side effects follow readiness, without touching rendered data or
// reopening the already self-healing resource stream.
watch(
  () => host.value?.status,
  (current, previous) => {
    if (current === "ready" && previous !== "ready") {
      // A host may have been unreachable during the identity watch's initial
      // request. Recover status-only fields without clearing the last snapshot.
      void fetchStatus();
      tickQueue();
      startReadyServices();
    }
  },
);
onUnmounted(() => {
  resourceAbort?.abort();
  resourceAbort = null;
  statusAbort?.abort();
  statusAbort = null;
  if (queueTimer) clearInterval(queueTimer);
  queueTimer = null;
});

// ── Derived display data ──────────────────────────────────────────────────

/** Stream frames win; before the first frame (or on older servers without the
 *  stream) fall back to the status poll's MB-based `gpu_info` summary. */
const gpus = computed<GpuSnapshot[]>(() => {
  if (snapshot.value) return snapshot.value.gpus;
  const info = telemetry.value?.gpuInfo;
  if (!info) return [];
  // Decimal MB → bytes, matching formatGB and the server's resources path.
  return [
    {
      ordinal: 0,
      name: info.name,
      backend: info.backend ?? inferBackendFromGpuName(info.name),
      vram_total: info.vram_total_mb * 1_000_000,
      vram_used: info.vram_used_mb * 1_000_000,
      gpu_utilization: null,
    },
  ];
});

function backendLabel(gpu: GpuSnapshot): string {
  return (gpu.backend || inferBackendFromGpuName(gpu.name)).toUpperCase();
}

const cpu = computed(() => snapshot.value?.cpu ?? null);
const ram = computed(() => snapshot.value?.system_ram ?? null);
const modelsDisk = computed(() => status.value?.models_disk ?? null);
const diskUsedPct = computed(() => {
  const d = modelsDisk.value;
  return d ? percent(d.total_bytes - d.free_bytes, d.total_bytes) : 0;
});

const queueDepth = computed(() => status.value?.queue_depth ?? host.value?.queueDepth ?? null);
const queueCapacity = computed(
  () => status.value?.queue_capacity ?? host.value?.queueCapacity ?? null,
);
const modelsLoaded = computed(() => telemetry.value?.modelsLoaded ?? []);
const installedModels = computed(() => hostModels.installedOn(hostId.value));

const queueSnapshot = computed(() => jobs.queues[hostId.value] ?? null);
const queuePaused = computed(() => queueSnapshot.value?.paused === true);

/** This host's whole server queue, tagged with what belongs to this app. */
const queueEntries = computed<EnrichedQueueEntry[]>(() => {
  const snap = queueSnapshot.value;
  if (!snap) return [];
  return enrichQueueEntries(
    snap.entries,
    hostId.value,
    generation.jobs,
    hosts.primaryHost?.id ?? "local",
  );
});

/** Same state vocabulary as the Jobs view: RUNNING · GPU 0 / QUEUED #2. */
function entryCode(entry: EnrichedQueueEntry): string {
  if (entry.state === "running") {
    return entry.gpu !== undefined ? `RUNNING · GPU ${entry.gpu}` : "RUNNING";
  }
  return entry.position > 0 ? `QUEUED #${entry.position}` : "QUEUED";
}

/** Elapsed wall-clock for running entries; re-evaluates on each poll frame. */
function entryElapsed(entry: EnrichedQueueEntry): string | null {
  if (entry.state !== "running" || !entry.started_at_unix_ms) return null;
  return formatEta((Date.now() - entry.started_at_unix_ms) / 1000);
}

const uptime = computed(() => status.value?.uptime_secs ?? null);
const hasTelemetry = computed(
  () => gpus.value.length > 0 || !!cpu.value || !!ram.value || !!modelsDisk.value,
);

/** Jobs on this host mean its GPU is developing — the VRAM meter warms.
 *  The 5 s queue poll is the live signal; the one-shot status depth only
 *  covers the window before the first snapshot lands. */
const hostBusy = computed(() => {
  const snap = queueSnapshot.value;
  if (snap) return snap.entries.length > 0;
  return (queueDepth.value ?? 0) > 0;
});

function vramFill(gpu: GpuSnapshot): string {
  if (vramLevel(gpu.vram_used, gpu.vram_total) === "critical") return "bg-stop";
  return hostBusy.value ? "bg-safelight" : "bg-halide";
}

/** `14 · 96.4 GB` summary for the models section header; null without sizes. */
const installedTotalLabel = computed(() => {
  const bytes = installedModels.value.reduce(
    (sum, m) => sum + (m.size_gb > 0 ? m.size_gb * 1_000_000_000 : 0),
    0,
  );
  return bytes > 0 ? formatGB(bytes) : null;
});

/** Denominator for the per-row relative usage bar, as on the Installed shelf. */
const maxModelDiskBytes = computed(() =>
  installedModels.value.reduce((max, m) => Math.max(max, modelDiskBytes(m)), 0),
);

async function repairFromDrawer() {
  const m = detailModel.value;
  const h = host.value;
  if (!m || !h) return;
  drawerRepairing.value = true;
  try {
    await startCatalogDownload(m.name, hostTarget() ?? undefined, h.kind === "remote");
    toasts.push(`Repairing ${m.name} on ${h.label}`);
  } catch (err) {
    toasts.push(
      err instanceof ApiError && err.status === 409 ? `${m.name} is already queued.` : String(err),
      "error",
    );
  } finally {
    drawerRepairing.value = false;
  }
}

function statusDot(s: "connecting" | "ready" | "error"): string {
  switch (s) {
    case "ready":
      return "bg-safelight";
    case "connecting":
      return "bg-halide animate-pulse";
    default:
      return "bg-stop";
  }
}

// ── Actions ───────────────────────────────────────────────────────────────

const isTarget = computed(() => (appPrefs.settings?.generateTargetHost ?? null) === hostId.value);

function toggleTarget() {
  void appPrefs.update({ generateTargetHost: isTarget.value ? null : hostId.value });
}

const renameOpen = ref(false);

function onRenameSave(name: string) {
  renameOpen.value = false;
  const h = host.value;
  if (h) void hosts.rename(h.id, name);
}

/** Open the host's web UI in the default browser. */
async function openHostUrl(url: string) {
  try {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  } catch {
    window.open(url, "_blank", "noopener");
  }
}

async function disconnect() {
  const h = host.value;
  if (!h) return;
  await hosts.disconnect(h.id);
  toasts.push(`Disconnected from ${h.label}`);
  void router.push("/settings");
}

// Forget drops the saved entry AND the stored API key — two-step confirm.
const forgetPending = ref(false);

async function forget() {
  const h = host.value;
  if (!h) return;
  if (!forgetPending.value) {
    forgetPending.value = true;
    return;
  }
  forgetPending.value = false;
  await hosts.disconnect(h.id);
  await ipc.forgetRemoteHost(h.id);
  toasts.push(`Forgot ${h.label}`);
  void router.push("/settings");
}
</script>

<template>
  <div class="h-full overflow-y-auto p-6">
    <div class="mx-auto max-w-3xl">
      <template v-if="host">
        <!-- Header -->
        <div class="flex items-center gap-3">
          <span
            class="h-2 w-2 shrink-0 rounded-full"
            :class="statusDot(host.status)"
            data-test="host-status-dot"
          />
          <h1
            class="min-w-0 truncate font-display text-display font-bold text-ink"
            style="font-stretch: 90%"
            data-test="host-title"
          >
            {{ host.label }}
          </h1>
          <span class="edge-code shrink-0">
            {{ host.kind === "local" ? "THIS DEVICE" : "REMOTE" }}
          </span>
          <span v-if="host.version" class="edge-code shrink-0" data-test="host-version">
            v{{ host.version }}
          </span>
        </div>
        <div class="mt-1.5 flex items-center gap-3 pl-5">
          <span class="data-mono text-ink-3" data-selectable data-test="host-url">{{
            host.baseUrl
          }}</span>
          <span
            v-if="host.instanceId"
            class="edge-code max-w-40 truncate"
            :title="host.instanceId"
            data-test="host-instance-id"
          >
            {{ host.instanceId }}
          </span>
        </div>
        <p v-if="host.status === 'error'" class="mt-2 pl-5 text-caption text-stop">
          Unreachable — reconnect below or check the server.
        </p>

        <!-- Actions -->
        <div class="mt-4 flex flex-wrap items-center gap-2 pl-5">
          <button
            type="button"
            data-test="target-toggle"
            class="h-7 rounded-control border px-2.5 text-body transition-colors active:translate-y-px disabled:opacity-40"
            :class="
              isTarget
                ? 'border-safelight/50 text-safelight'
                : 'border-edge text-ink-2 hover:text-ink'
            "
            :aria-pressed="isTarget"
            :disabled="!isTarget && host.status !== 'ready'"
            @click="toggleTarget"
          >
            {{ isTarget ? "Used for generations" : "Use for generations" }}
          </button>
          <button
            v-if="host.kind === 'remote'"
            type="button"
            data-test="rename-host"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 transition-colors hover:text-ink active:translate-y-px"
            @click="renameOpen = true"
          >
            Rename…
          </button>
          <button
            type="button"
            data-test="open-web-ui"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 transition-colors hover:text-ink active:translate-y-px disabled:opacity-40"
            :disabled="!host.baseUrl"
            @click="openHostUrl(host.baseUrl ?? '')"
          >
            Open web UI
          </button>
          <button
            v-if="host.kind === 'remote' && host.status === 'error'"
            type="button"
            data-test="reconnect-host"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 transition-colors hover:text-ink active:translate-y-px"
            @click="hosts.reconnect(host.id)"
          >
            Reconnect
          </button>
          <div class="flex-1" />
          <button
            v-if="host.kind === 'remote'"
            type="button"
            data-test="disconnect-host"
            class="h-7 rounded-control px-2.5 text-body text-ink-3 transition-colors hover:text-stop active:translate-y-px"
            @click="disconnect"
          >
            Disconnect
          </button>
          <button
            v-if="host.kind === 'remote'"
            type="button"
            data-test="forget-host"
            class="h-7 rounded-control px-2.5 text-body transition-colors active:translate-y-px"
            :class="forgetPending ? 'text-stop' : 'text-ink-3 hover:text-stop'"
            @click="forget"
            @blur="forgetPending = false"
          >
            {{ forgetPending ? "Forget?" : "Forget" }}
          </button>
        </div>

        <!-- Telemetry — one instrument panel: GPU(s), CPU, RAM, models disk.
             A shared grid keeps every meter's label, bar, and value in the
             same column tracks, so the panel reads as one machine. -->
        <div class="mt-8 flex items-center gap-2">
          <h2 class="edge-code">TELEMETRY</h2>
          <div class="border-edge h-px flex-1 border-t" />
          <span v-if="uptime !== null" class="edge-code uppercase" data-test="host-uptime">
            UP {{ formatUptime(uptime) }}
          </span>
          <span v-if="snapshot" class="edge-code flex items-center gap-1.5"
            ><span class="h-1.5 w-1.5 rounded-full bg-safelight" aria-hidden="true" />LIVE</span
          >
        </div>
        <div
          v-if="hasTelemetry"
          data-test="telemetry-panel"
          class="border-edge mt-2 grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-2.5 rounded-control border bg-bench px-4 py-3"
        >
          <template v-for="gpu in gpus" :key="gpu.ordinal">
            <div class="contents" data-test="gpu-card">
              <div class="col-span-full flex min-w-0 items-baseline gap-2">
                <span v-if="gpus.length > 1" class="edge-code">GPU {{ gpu.ordinal }}</span>
                <span class="min-w-0 truncate text-body font-medium text-ink">{{ gpu.name }}</span>
                <span class="edge-code shrink-0">{{ backendLabel(gpu) }}</span>
                <div class="flex-1" />
                <span
                  v-if="gpu.gpu_utilization !== null && gpu.gpu_utilization !== undefined"
                  class="data-mono shrink-0 text-ink-3"
                >
                  <span class="text-ink-2" data-test="gpu-utilization"
                    >{{ gpu.gpu_utilization }}%</span
                  >
                  util
                </span>
              </div>
              <span class="edge-code">VRAM</span>
              <div
                class="h-1.5 overflow-hidden rounded-full bg-bath"
                role="meter"
                aria-valuemin="0"
                aria-valuemax="100"
                :aria-valuenow="Math.round(percent(gpu.vram_used, gpu.vram_total))"
                :aria-label="`VRAM used on ${gpu.name}`"
              >
                <div
                  class="h-full transition-[width] duration-300"
                  :class="vramFill(gpu)"
                  :style="{ width: `${percent(gpu.vram_used, gpu.vram_total)}%` }"
                />
              </div>
              <span class="data-mono text-right text-ink-3">
                {{ formatGB(gpu.vram_used) }}/{{ formatGB(gpu.vram_total) }}
              </span>
            </div>
          </template>
          <div v-if="cpu" class="contents" data-test="cpu-card">
            <span class="edge-code">CPU</span>
            <div
              class="h-1.5 overflow-hidden rounded-full bg-bath"
              role="meter"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="Math.round(cpu.usage_percent)"
              aria-label="CPU usage"
            >
              <div
                class="h-full bg-halide transition-[width] duration-300"
                :style="{ width: `${cpu.usage_percent}%` }"
              />
            </div>
            <span class="data-mono text-right text-ink-3">
              {{ cpu.usage_percent.toFixed(0) }}% · {{ cpu.cores }} CORES
            </span>
          </div>
          <div v-if="ram" class="contents" data-test="ram-card">
            <span class="edge-code">RAM</span>
            <div
              class="h-1.5 overflow-hidden rounded-full bg-bath"
              role="meter"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="Math.round(percent(ram.used, ram.total))"
              aria-label="System RAM used"
            >
              <div
                class="h-full bg-halide transition-[width] duration-300"
                :style="{ width: `${percent(ram.used, ram.total)}%` }"
              />
            </div>
            <span class="data-mono text-right text-ink-3">
              {{ formatGB(ram.used) }}/{{ formatGB(ram.total) }}
            </span>
          </div>
          <div v-if="modelsDisk" class="contents" data-test="storage-card">
            <span class="edge-code" title="Models disk">DISK</span>
            <div
              class="h-1.5 overflow-hidden rounded-full bg-bath"
              role="meter"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="Math.round(diskUsedPct)"
              aria-label="Models disk used"
            >
              <div
                class="h-full transition-[width] duration-300"
                :class="diskUsedPct >= 92 ? 'bg-stop' : 'bg-halide'"
                :style="{ width: `${diskUsedPct}%` }"
              />
            </div>
            <span class="data-mono text-right text-ink-3">
              {{ formatGB(modelsDisk.free_bytes) }} free of {{ formatGB(modelsDisk.total_bytes) }}
            </span>
          </div>
        </div>
        <p v-else class="mt-2 text-caption text-ink-3">No live telemetry from this host yet.</p>

        <!-- Queue — the host's whole server queue (other clients' jobs
             included), with resident models labeled LOADED so they can't
             read as queued jobs. Management stays in the Jobs view. -->
        <div class="mt-8 flex items-center gap-2">
          <h2 class="edge-code">QUEUE</h2>
          <div class="border-edge h-px flex-1 border-t" />
          <span
            v-if="queuePaused"
            class="data-mono text-caption text-stop"
            data-test="queue-paused"
          >
            PAUSED
          </span>
          <span class="edge-code" data-test="queue-depth">
            {{ queueDepth ?? "—" }}<template v-if="queueCapacity">/{{ queueCapacity }}</template>
          </span>
          <RouterLink to="/jobs" class="text-caption text-ink-3 hover:text-ink">
            Jobs →
          </RouterLink>
        </div>
        <ul
          v-if="queueEntries.length"
          data-test="host-queue"
          class="border-edge divide-edge mt-2 divide-y overflow-hidden rounded-control border bg-bench"
        >
          <li
            v-for="entry in queueEntries"
            :key="entry.id"
            data-test="host-queue-row"
            class="flex items-center gap-2.5 px-3 py-1.5"
          >
            <span
              class="h-1.5 w-1.5 shrink-0 rounded-full"
              :class="entry.state === 'running' ? 'bg-safelight' : 'bg-halide'"
              aria-hidden="true"
            />
            <span class="edge-code shrink-0">{{ entryCode(entry) }}</span>
            <span class="min-w-0 truncate text-body text-ink">{{ entry.model }}</span>
            <span v-if="!entry.mine" class="edge-code shrink-0">OTHER CLIENT</span>
            <div class="flex-1" />
            <span v-if="entryElapsed(entry)" class="data-mono shrink-0 text-ink-3">
              {{ entryElapsed(entry) }}
            </span>
          </li>
        </ul>
        <p v-else class="mt-2 text-caption text-ink-3" data-test="queue-empty">Queue is empty.</p>
        <div v-if="modelsLoaded.length" class="mt-2 flex flex-wrap items-center gap-1.5">
          <span class="edge-code mr-1" data-test="loaded-label">LOADED</span>
          <span
            v-for="m in modelsLoaded"
            :key="m"
            data-test="loaded-model-chip"
            class="border-edge flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-caption text-ink-2"
            ><span class="h-1.5 w-1.5 shrink-0 rounded-full bg-safelight" aria-hidden="true" />{{
              m
            }}</span
          >
        </div>
        <p v-else class="mt-2 text-caption text-ink-3">
          No models loaded — the next generation loads one first.
        </p>

        <!-- Models installed on this host, with this host's pulls above them -->
        <div class="mt-8 flex items-center gap-2">
          <h2 class="edge-code">MODELS ON THIS HOST</h2>
          <div class="border-edge h-px flex-1 border-t" />
          <span v-if="installedModels.length" class="edge-code" data-test="models-summary">
            {{ installedModels.length
            }}<template v-if="installedTotalLabel"> · {{ installedTotalLabel }}</template>
          </span>
          <RouterLink to="/models" class="text-caption text-ink-3 hover:text-ink">
            Catalog →
          </RouterLink>
        </div>
        <DownloadsTray
          :host-id="hostId"
          data-test="host-downloads"
          class="mt-2 rounded-control border"
        />
        <ul
          v-if="installedModels.length"
          class="border-edge divide-edge mt-2 divide-y overflow-hidden rounded-control border bg-bench"
        >
          <li v-for="m in installedModels" :key="m.name" data-test="model-row">
            <ModelTableRow
              :name="m.name"
              :source="modelSource(m)"
              :loaded="m.is_loaded"
              :family="m.family"
              :page-url="m.hf_repo ? `https://huggingface.co/${m.hf_repo}` : null"
              :size-primary="
                modelSizeLabels(m).weights ?? modelSizeLabels(m).runtime ?? 'Size unavailable'
              "
              :size-secondary="
                modelSizeLabels(m).weights && modelSizeLabels(m).runtime
                  ? modelSizeLabels(m).runtime
                  : null
              "
              :bar-percent="percent(modelDiskBytes(m), maxModelDiskBytes)"
              clickable
              class="px-3 py-2"
              @open="detailModel = m"
            />
          </li>
        </ul>
        <p v-else class="mt-2 text-caption text-ink-3">No installed models reported</p>

        <!-- One consistent model-detail drawer, shared with the catalog. -->
        <CatalogDetailDrawer
          v-if="detailModel"
          :entry="installedModelToEntry(detailModel)"
          :pulling="drawerRepairing"
          :target="hostTarget() ?? undefined"
          :forward-credentials="host.kind === 'remote'"
          @close="detailModel = null"
          @pull="repairFromDrawer"
        />

        <RenameDialog
          :open="renameOpen"
          title="Rename host"
          :initial="host.label"
          @save="onRenameSave"
          @cancel="renameOpen = false"
        />
      </template>

      <!-- Unknown id — quiet empty state -->
      <div v-else class="mt-16 text-center" data-test="host-missing">
        <h1 class="font-display text-display font-bold text-ink" style="font-stretch: 90%">
          Host not found
        </h1>
        <p class="mt-2 text-body text-ink-2">
          This host isn't connected. It may have been disconnected or forgotten.
        </p>
        <RouterLink
          to="/settings"
          data-test="back-to-hosts"
          class="mt-4 inline-block text-body text-safelight hover:brightness-110"
        >
          Back to Hosts
        </RouterLink>
      </div>
    </div>
  </div>
</template>
