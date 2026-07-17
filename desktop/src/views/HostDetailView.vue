<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { RouterLink, useRoute, useRouter } from "vue-router";
import DownloadsTray from "../components/models/DownloadsTray.vue";
import RenameDialog from "../components/shell/RenameDialog.vue";
import { apiJsonTo, type ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import { formatGB, percent, vramLevel } from "../lib/format";
import { inferBackendFromGpuName } from "../lib/hosts";
import { modelSizeLabels } from "../lib/models";
import { ipc } from "../lib/ipc";
import type { GpuSnapshot, ResourceSnapshot, ServerStatus } from "../lib/api/types";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useDownloadsStore } from "../stores/downloads";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
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
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
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

/** Reopen the stream against the CURRENT host; prior subscription aborts. */
function startResourceStream() {
  resourceAbort?.abort();
  resourceAbort = null;
  snapshot.value = null;
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

let statusAbort: AbortController | null = null;

/** One-shot status fetch for models-disk stats (and fresher queue fields).
 *  Guarded per request: the :id param retargets this reused component in
 *  place, and a slow host's late response must never populate the page of
 *  the host the user navigated to next. */
async function fetchStatus() {
  statusAbort?.abort();
  const abort = new AbortController();
  statusAbort = abort;
  status.value = null;
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

// Immediate + retargeting: covers first render, the :id param changing in
// place, and a host whose baseUrl appears after a late connect.
watch(
  () => [hostId.value, host.value?.baseUrl] as const,
  () => {
    startResourceStream();
    void fetchStatus();
    const current = host.value;
    if (current?.status === "ready") {
      void downloads.subscribe(current).catch(() => {
        // Host status and the model list still render if an older server lacks
        // the download stream; reconnect retries are owned by the SSE helper.
      });
      void hostModels.refresh(true);
    }
  },
  { immediate: true },
);
onUnmounted(() => {
  resourceAbort?.abort();
  resourceAbort = null;
  statusAbort?.abort();
  statusAbort = null;
});
onMounted(() => void hostModels.refresh(true));

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
        <DownloadsTray :host-id="hostId" data-test="host-downloads" />
        <!-- Header -->
        <div class="flex items-center gap-3">
          <span
            class="h-2 w-2 shrink-0 rounded-full"
            :class="statusDot(host.status)"
            data-test="host-status-dot"
          />
          <h1
            class="min-w-0 truncate font-display text-display-md font-bold text-ink"
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
        <div class="mt-1 flex items-center gap-3 pl-5">
          <span class="data-mono text-ink-3" data-test="host-url">{{ host.baseUrl }}</span>
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
            class="border-edge h-7 rounded-control border px-2.5 text-body"
            :class="isTarget ? 'text-safelight' : 'text-ink-2 hover:text-ink'"
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
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            @click="renameOpen = true"
          >
            Rename…
          </button>
          <button
            type="button"
            data-test="open-web-ui"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            :disabled="!host.baseUrl"
            @click="openHostUrl(host.baseUrl ?? '')"
          >
            Open web UI
          </button>
          <button
            v-if="host.kind === 'remote' && host.status === 'error'"
            type="button"
            data-test="reconnect-host"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            @click="hosts.reconnect(host.id)"
          >
            Reconnect
          </button>
          <div class="flex-1" />
          <button
            v-if="host.kind === 'remote'"
            type="button"
            data-test="disconnect-host"
            class="h-7 rounded-control px-2.5 text-body text-ink-3 hover:text-stop"
            @click="disconnect"
          >
            Disconnect
          </button>
          <button
            v-if="host.kind === 'remote'"
            type="button"
            data-test="forget-host"
            class="h-7 rounded-control px-2.5 text-body"
            :class="forgetPending ? 'text-stop' : 'text-ink-3 hover:text-stop'"
            @click="forget"
            @blur="forgetPending = false"
          >
            {{ forgetPending ? "Forget?" : "Forget" }}
          </button>
        </div>

        <!-- Telemetry -->
        <div class="mt-8 flex items-center gap-2">
          <span class="edge-code">TELEMETRY</span>
          <div class="border-edge h-px flex-1 border-t" />
          <span v-if="snapshot" class="edge-code">LIVE</span>
        </div>
        <div
          v-for="gpu in gpus"
          :key="gpu.ordinal"
          data-test="gpu-card"
          class="border-edge mt-2 rounded-control border bg-bench px-3 py-2"
        >
          <div class="flex items-center gap-2">
            <span class="data-mono text-ink-2">{{ gpu.name }}</span>
            <span class="edge-code">{{ backendLabel(gpu) }}</span>
            <div class="flex-1" />
            <span
              v-if="gpu.gpu_utilization !== null && gpu.gpu_utilization !== undefined"
              class="data-mono text-ink-3"
              data-test="gpu-utilization"
            >
              {{ gpu.gpu_utilization }}%
            </span>
          </div>
          <div class="mt-2 flex items-center gap-2">
            <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-bath">
              <div
                class="h-full"
                :class="
                  vramLevel(gpu.vram_used, gpu.vram_total) === 'critical' ? 'bg-stop' : 'bg-halide'
                "
                :style="{ width: `${percent(gpu.vram_used, gpu.vram_total)}%` }"
              />
            </div>
            <span class="data-mono text-ink-3">
              VRAM {{ formatGB(gpu.vram_used) }}/{{ formatGB(gpu.vram_total) }}
            </span>
          </div>
        </div>
        <div v-if="cpu || ram" class="mt-2 grid grid-cols-2 gap-2">
          <div
            v-if="cpu"
            data-test="cpu-card"
            class="border-edge rounded-control border bg-bench px-3 py-2"
          >
            <div class="flex items-center gap-2">
              <span class="edge-code">CPU</span>
              <span class="data-mono text-ink-3">{{ cpu.cores }} CORES</span>
            </div>
            <div class="mt-2 flex items-center gap-2">
              <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-bath">
                <div class="h-full bg-halide" :style="{ width: `${cpu.usage_percent}%` }" />
              </div>
              <span class="data-mono text-ink-3">{{ cpu.usage_percent.toFixed(0) }}%</span>
            </div>
          </div>
          <div
            v-if="ram"
            data-test="ram-card"
            class="border-edge rounded-control border bg-bench px-3 py-2"
          >
            <span class="edge-code">RAM</span>
            <div class="mt-2 flex items-center gap-2">
              <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-bath">
                <div
                  class="h-full bg-halide"
                  :style="{ width: `${percent(ram.used, ram.total)}%` }"
                />
              </div>
              <span class="data-mono text-ink-3">
                {{ formatGB(ram.used) }}/{{ formatGB(ram.total) }}
              </span>
            </div>
          </div>
        </div>
        <p v-if="gpus.length === 0 && !cpu && !ram" class="mt-2 text-caption text-ink-3">
          No live telemetry from this host yet.
        </p>

        <!-- Storage (models disk; absent on older servers) -->
        <template v-if="modelsDisk">
          <div class="mt-8 flex items-center gap-2">
            <span class="edge-code">STORAGE</span>
            <div class="border-edge h-px flex-1 border-t" />
          </div>
          <div
            data-test="storage-card"
            class="border-edge mt-2 rounded-control border bg-bench px-3 py-2"
          >
            <div class="flex items-center gap-2">
              <span class="text-caption text-ink-2">Models disk</span>
              <div class="flex-1" />
              <span class="data-mono text-ink-3">
                {{ formatGB(modelsDisk.free_bytes) }} free of {{ formatGB(modelsDisk.total_bytes) }}
              </span>
            </div>
            <div class="mt-2 h-1.5 overflow-hidden rounded-full bg-bath">
              <div
                class="h-full"
                :class="diskUsedPct >= 92 ? 'bg-stop' : 'bg-halide'"
                :style="{ width: `${diskUsedPct}%` }"
              />
            </div>
          </div>
        </template>

        <!-- Queue -->
        <div class="mt-8 flex items-center gap-2">
          <span class="edge-code">QUEUE</span>
          <div class="border-edge h-px flex-1 border-t" />
          <span class="edge-code" data-test="queue-depth">
            {{ queueDepth ?? "—" }}<template v-if="queueCapacity">/{{ queueCapacity }}</template>
          </span>
        </div>
        <div v-if="modelsLoaded.length" class="mt-2 flex flex-wrap gap-1.5">
          <span
            v-for="m in modelsLoaded"
            :key="m"
            data-test="loaded-model-chip"
            class="border-edge rounded-full border px-2 py-0.5 text-caption text-ink-2"
          >
            {{ m }}
          </span>
        </div>
        <p v-else class="mt-2 text-caption text-ink-3">No models loaded</p>

        <!-- Models installed on this host -->
        <div class="mt-8 flex items-center gap-2">
          <span class="edge-code">MODELS ON THIS HOST</span>
          <div class="border-edge h-px flex-1 border-t" />
          <RouterLink to="/models" class="text-caption text-ink-3 hover:text-ink">
            Catalog
          </RouterLink>
        </div>
        <ul v-if="installedModels.length" class="mt-2 space-y-1">
          <li
            v-for="m in installedModels"
            :key="m.name"
            data-test="model-row"
            class="border-edge flex items-center gap-2 rounded-control border bg-bench px-3 py-1.5"
          >
            <span class="min-w-0 truncate text-body text-ink">{{ m.name }}</span>
            <span class="text-caption text-ink-3">{{ m.family }}</span>
            <div class="flex-1" />
            <span class="shrink-0 text-right">
              <span class="data-mono block text-caption text-ink-2">
                {{ modelSizeLabels(m).weights ?? modelSizeLabels(m).runtime ?? "Size unavailable" }}
              </span>
              <span
                v-if="modelSizeLabels(m).runtime && modelSizeLabels(m).weights"
                class="data-mono block text-[10px] text-ink-3"
              >
                {{ modelSizeLabels(m).runtime }}
              </span>
            </span>
          </li>
        </ul>
        <p v-else class="mt-2 text-caption text-ink-3">No installed models reported</p>

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
        <h1 class="font-display text-display-md font-bold text-ink" style="font-stretch: 90%">
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
