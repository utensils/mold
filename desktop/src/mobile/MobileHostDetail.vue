<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { parseDeviceListResponse, setDeviceEnabled, type DeviceInfo } from "@studio/api/devices";
import {
  cancelQueueJob,
  parseQueueListing,
  setQueueDevicePin,
  type QueuePlan,
} from "@studio/api/queuePlan";
import DevicePanel from "@studio/components/DevicePanel.vue";
import MinimaxH3InventoryPanel from "@studio/components/MinimaxH3InventoryPanel.vue";
import { canMutateDevice } from "@studio/lib/deviceLifecycle";
import { apiJsonTo } from "../lib/api/client";
import { describeTransportError } from "../lib/api/errors";
import { gpuSnapshotsFromStatus } from "../lib/api/gpuStatus";
import { sseStream } from "../lib/api/sse";
import { subscribeToDeviceSnapshots } from "../lib/api/deviceEvents";
import type {
  DownloadEvent,
  DownloadJob,
  GpuSnapshot,
  ModelEntry,
  ResourceSnapshot,
  ServerCapabilities,
  ServerStatus,
} from "../lib/api/types";
import { formatGB, formatUptime, percent } from "../lib/format";
import { unifiedMemoryHost } from "@studio/lib/telemetryMemory";
import { inferBackendFromGpuName } from "../lib/hosts";
import { unloadModel } from "../lib/api/models";
import { modelDisplayName, modelDisplayNameForId, modelSizeLabels } from "../lib/models";
import { applyDownloadEvent, emptyDownloadsState, type DownloadsState } from "../stores/downloads";
import type { QueueEntry } from "../stores/jobs";
import { mobileHostTarget, type MobileHost } from "./hosts";

type DetailSnapshot = ResourceSnapshot & {
  cpu?: { cores: number; usage_percent: number } | null;
};

const props = defineProps<{
  host: MobileHost;
  active: boolean;
}>();

const emit = defineEmits<{
  (event: "back"): void;
  (event: "select", id: string): void;
  (event: "rename", payload: { id: string; name: string }): void;
  (event: "disconnect", id: string): void;
  (event: "reconnect", id: string): void;
  (event: "forget", id: string): void;
  (event: "catalog", hostId: string): void;
  (event: "status", payload: { id: string; status: ServerStatus | null }): void;
}>();

const status = ref<ServerStatus | null>(null);
const snapshot = ref<DetailSnapshot | null>(null);
const devices = ref<DeviceInfo[] | null>(null);
const deviceCapabilities = ref<ServerCapabilities | null>(null);
const deviceMutations = ref(new Set<string>());
const deviceError = ref("");
const installed = ref<ModelEntry[]>([]);
const queue = ref<QueueEntry[]>([]);
const queuePlan = ref<QueuePlan | null>(null);
const queueApiAvailable = ref(false);
const downloads = ref<DownloadsState>(emptyDownloadsState());
const loading = ref(false);
const error = ref("");
const renaming = ref(false);
const renameValue = ref("");
const forgetPending = ref(false);
const unloading = ref<Set<string>>(new Set());
const cancelConfirmId = ref<string | null>(null);
const cancellingQueueIds = ref(new Set<string>());
let loadEpoch = 0;
let queueRequestGeneration = 0;
let deviceRequestGeneration = 0;
let resourceAbort: AbortController | null = null;
let downloadsAbort: AbortController | null = null;
let deviceEventsAbort: AbortController | null = null;
let livePollTimer: ReturnType<typeof setTimeout> | null = null;
let deviceRefreshPromise: Promise<void> | null = null;
let deviceRefreshQueued = false;

const target = computed(() => mobileHostTarget(props.host));
const inFlightDownloads = computed(() => [
  ...downloads.value.activeJobs,
  ...downloads.value.queued,
]);
const loadedModels = computed(() => status.value?.models_loaded ?? []);
const uptime = computed(() => status.value?.uptime_secs ?? null);
const queueDepth = computed(() =>
  queueApiAvailable.value ? queue.value.length : (status.value?.queue_depth ?? queue.value.length),
);
const modelLabel = (name: string) => modelDisplayNameForId(name, installed.value);

function downloadStatus(job: DownloadJob): string {
  if (job.status === "queued") return "Waiting";
  if (!job.bytes_total) {
    return job.current_file ? `Preparing… · ${job.current_file}` : "Preparing…";
  }
  return `${job.files_done}/${job.files_total} files`;
}

const gpus = computed<GpuSnapshot[]>(() => {
  if (snapshot.value?.gpus.length) return snapshot.value.gpus;
  return gpuSnapshotsFromStatus(status.value);
});

const ram = computed(() => snapshot.value?.system_ram ?? null);
const cpu = computed(() => snapshot.value?.cpu ?? null);
/** Apple Metal shares one physical pool — a VRAM row and a RAM row would
 *  show the same numbers twice, so unified hosts render one Memory row. */
const unifiedMemory = computed(() => unifiedMemoryHost(gpus.value));
const disk = computed(() => status.value?.models_disk ?? null);
const h3Host = computed(() => [
  {
    id: props.host.id,
    label: props.host.name,
    capabilities: deviceCapabilities.value,
  },
]);

function stopLiveServices(): void {
  queueRequestGeneration += 1;
  deviceRequestGeneration += 1;
  resourceAbort?.abort();
  downloadsAbort?.abort();
  deviceEventsAbort?.abort();
  resourceAbort = null;
  downloadsAbort = null;
  deviceEventsAbort = null;
  if (livePollTimer) clearTimeout(livePollTimer);
  livePollTimer = null;
  deviceRefreshPromise = null;
  deviceRefreshQueued = false;
}

async function refreshQueue(epoch = loadEpoch): Promise<void> {
  const generation = ++queueRequestGeneration;
  const requestTarget = target.value;
  try {
    const listing = parseQueueListing(await apiJsonTo<unknown>(requestTarget, "/api/queue"));
    if (
      epoch === loadEpoch &&
      generation === queueRequestGeneration &&
      requestTarget.baseUrl === target.value.baseUrl &&
      requestTarget.apiKey === target.value.apiKey
    ) {
      queue.value = listing.entries as QueueEntry[];
      queuePlan.value = listing.plan;
      queueApiAvailable.value = true;
    }
  } catch {
    if (
      epoch === loadEpoch &&
      generation === queueRequestGeneration &&
      requestTarget.baseUrl === target.value.baseUrl &&
      requestTarget.apiKey === target.value.apiKey
    ) {
      // A plan is tentative server authority, so it cannot survive a failed
      // refresh. Older hosts still fall back to status queue depth.
      queue.value = [];
      queuePlan.value = null;
      queueApiAvailable.value = false;
    }
  }
}

async function refreshDeviceState(epoch = loadEpoch): Promise<void> {
  const generation = ++deviceRequestGeneration;
  const requestHost = props.host;
  const requestTarget = target.value;
  const [deviceResult, capabilityResult] = await Promise.allSettled([
    apiJsonTo<unknown>(requestTarget, "/api/devices"),
    apiJsonTo<ServerCapabilities>(requestTarget, "/api/capabilities"),
  ]);
  const isCurrent = () =>
    epoch === loadEpoch &&
    generation === deviceRequestGeneration &&
    props.host.id === requestHost.id &&
    props.host.baseUrl === requestHost.baseUrl &&
    props.host.apiKey === requestHost.apiKey &&
    requestTarget.baseUrl === target.value.baseUrl &&
    requestTarget.apiKey === target.value.apiKey;
  if (!isCurrent()) return;

  let capabilityPayload: ServerCapabilities | null = null;
  let capabilityFailure: unknown = null;
  if (
    capabilityResult.status === "fulfilled" &&
    capabilityResult.value !== null &&
    typeof capabilityResult.value === "object" &&
    !Array.isArray(capabilityResult.value)
  ) {
    capabilityPayload = capabilityResult.value;
    deviceCapabilities.value = capabilityPayload;
  } else {
    // A prior successful response is no longer mutation authority once the
    // current route cannot verify its lifecycle contract.
    deviceCapabilities.value = null;
    capabilityFailure =
      capabilityResult.status === "rejected"
        ? capabilityResult.reason
        : new SyntaxError("Malformed capabilities response");
  }

  let nextDevices: DeviceInfo[] | null = null;
  let deviceFailure: unknown = null;
  if (deviceResult.status === "fulfilled") {
    try {
      nextDevices = parseDeviceListResponse(deviceResult.value).devices;
    } catch (caught) {
      deviceFailure = caught;
    }
  } else {
    deviceFailure = deviceResult.reason;
  }

  if (deviceFailure === null) {
    devices.value = nextDevices;
    if (capabilityFailure === null) {
      deviceError.value = "";
    } else {
      const detail = describeTransportError(capabilityFailure, requestHost.name);
      deviceError.value = `Couldn’t verify compute device controls. ${detail} Mold will try again automatically.`;
    }
    return;
  }

  // Only a successful capabilities response can prove that this is a host
  // from before the additive device API. A failed capabilities request is
  // uncertainty, so retain the last good snapshot and keep retrying visibly.
  const legacyDeviceApi =
    capabilityResult.status === "fulfilled" &&
    capabilityPayload !== null &&
    capabilityPayload.devices?.available !== true;
  if (legacyDeviceApi) {
    devices.value = null;
    deviceError.value = "";
    return;
  }

  const detail = describeTransportError(deviceFailure, requestHost.name);
  deviceError.value = `Couldn’t refresh compute devices. ${detail} Mold will try again automatically.`;
}

function requestDeviceRefresh(epoch = loadEpoch): Promise<void> {
  if (epoch !== loadEpoch) return Promise.resolve();
  if (deviceRefreshPromise) {
    deviceRefreshQueued = true;
    // The catch-up pass owns authority once an invalidation arrives. Fence the
    // older request immediately so it cannot flash stale telemetry first.
    deviceRequestGeneration += 1;
    return deviceRefreshPromise;
  }

  const refresh = (async () => {
    do {
      deviceRefreshQueued = false;
      await refreshDeviceState(epoch);
    } while (deviceRefreshQueued && epoch === loadEpoch);
  })();
  deviceRefreshPromise = refresh;
  return refresh.finally(() => {
    if (deviceRefreshPromise === refresh) deviceRefreshPromise = null;
  });
}

async function refreshDevicesSafely(epoch: number): Promise<void> {
  try {
    await requestDeviceRefresh(epoch);
  } catch (caught) {
    if (epoch !== loadEpoch) return;
    const detail = describeTransportError(caught, props.host.name);
    deviceError.value = `Couldn’t refresh compute devices. ${detail} Mold will try again automatically.`;
  }
}

function scheduleLivePoll(epoch: number): void {
  if (epoch !== loadEpoch) return;
  livePollTimer = setTimeout(async () => {
    livePollTimer = null;
    // Queue authority has its own generation fence and must not hold device
    // freshness hostage if an older endpoint never settles.
    void refreshQueue(epoch);
    await refreshDevicesSafely(epoch);
    scheduleLivePoll(epoch);
  }, 5_000);
}

function startLiveServices(epoch: number): void {
  resourceAbort = new AbortController();
  void sseStream("/api/resources/stream", {
    target: target.value,
    signal: resourceAbort.signal,
    retry: true,
    onEvent(event, data) {
      if (event !== "snapshot" || epoch !== loadEpoch) return;
      try {
        snapshot.value = JSON.parse(data) as DetailSnapshot;
      } catch {
        // A malformed telemetry frame must not tear down the detail view.
      }
    },
  });

  downloadsAbort = new AbortController();
  void sseStream("/api/downloads/stream", {
    target: target.value,
    signal: downloadsAbort.signal,
    retry: true,
    onEvent(_event, data) {
      if (epoch !== loadEpoch) return;
      try {
        downloads.value = applyDownloadEvent(downloads.value, JSON.parse(data) as DownloadEvent);
      } catch {
        // Keep the last good snapshot when an older host sends an unknown frame.
      }
    },
  });

  deviceEventsAbort = new AbortController();
  subscribeToDeviceSnapshots(target.value, deviceEventsAbort.signal, () => {
    void refreshQueue(epoch);
    void refreshDevicesSafely(epoch);
  });

  void refreshQueue(epoch);
  scheduleLivePoll(epoch);
}

async function loadHost(): Promise<void> {
  const epoch = ++loadEpoch;
  stopLiveServices();
  loading.value = true;
  error.value = "";
  status.value = null;
  snapshot.value = null;
  devices.value = null;
  deviceCapabilities.value = null;
  deviceError.value = "";
  installed.value = [];
  queue.value = [];
  queuePlan.value = null;
  queueApiAvailable.value = false;
  downloads.value = emptyDownloadsState();
  renameValue.value = props.host.name;
  renaming.value = false;
  forgetPending.value = false;
  cancelConfirmId.value = null;
  cancellingQueueIds.value = new Set();
  if (props.host.connected === false) {
    loading.value = false;
    return;
  }

  try {
    const [nextStatus, models] = await Promise.all([
      apiJsonTo<ServerStatus>(target.value, "/api/status"),
      apiJsonTo<ModelEntry[]>(target.value, "/api/models"),
    ]);
    if (epoch !== loadEpoch) return;
    status.value = nextStatus;
    installed.value = models.filter((model) => model.downloaded);
    emit("status", { id: props.host.id, status: nextStatus });
    await refreshDevicesSafely(epoch);
    if (epoch !== loadEpoch) return;
    startLiveServices(epoch);
  } catch (caught) {
    if (epoch !== loadEpoch) return;
    error.value = describeTransportError(caught, props.host.name);
    emit("status", { id: props.host.id, status: null });
  } finally {
    if (epoch === loadEpoch) loading.value = false;
  }
}

async function toggleDevice(device: DeviceInfo): Promise<void> {
  if (!canMutateDevice(device, deviceCapabilities.value)) return;
  const enabled = !device.desired_enabled;
  deviceMutations.value = new Set(deviceMutations.value).add(device.id);
  try {
    await setDeviceEnabled(target.value, device.id, enabled);
    await requestDeviceRefresh();
  } catch (caught) {
    error.value = describeTransportError(caught, props.host.name);
  } finally {
    const next = new Set(deviceMutations.value);
    next.delete(device.id);
    deviceMutations.value = next;
  }
}

async function toggleDeviceById(deviceId: string, enabled: boolean): Promise<void> {
  const device = devices.value?.find((candidate) => candidate.id === deviceId);
  if (!device || enabled === device.desired_enabled) return;
  await toggleDevice(device);
}

async function unpinWork(workId: string): Promise<void> {
  try {
    await setQueueDevicePin(target.value, workId, null);
    await refreshQueue();
  } catch (caught) {
    error.value = describeTransportError(caught, props.host.name);
  }
}

async function cancelQueuedJob(entry: QueueEntry): Promise<void> {
  if (entry.state !== "queued" || cancellingQueueIds.value.has(entry.id)) return;
  if (cancelConfirmId.value !== entry.id) {
    cancelConfirmId.value = entry.id;
    return;
  }

  cancelConfirmId.value = null;
  const epoch = loadEpoch;
  const requestTarget = target.value;
  error.value = "";
  cancellingQueueIds.value = new Set(cancellingQueueIds.value).add(entry.id);
  try {
    await cancelQueueJob(requestTarget, entry.id);
    if (
      epoch === loadEpoch &&
      requestTarget.baseUrl === target.value.baseUrl &&
      requestTarget.apiKey === target.value.apiKey
    ) {
      await refreshQueue(epoch);
    }
  } catch (caught) {
    if (
      epoch === loadEpoch &&
      requestTarget.baseUrl === target.value.baseUrl &&
      requestTarget.apiKey === target.value.apiKey
    ) {
      error.value = describeTransportError(caught, props.host.name);
    }
  } finally {
    if (epoch === loadEpoch) {
      const next = new Set(cancellingQueueIds.value);
      next.delete(entry.id);
      cancellingQueueIds.value = next;
    }
  }
}

function clearCancelConfirmation(id: string): void {
  if (cancelConfirmId.value === id) cancelConfirmId.value = null;
}

function saveRename(): void {
  const name = renameValue.value.trim();
  if (!name) return;
  emit("rename", { id: props.host.id, name });
  renaming.value = false;
}

function requestForget(): void {
  if (!forgetPending.value) {
    forgetPending.value = true;
    return;
  }
  emit("forget", props.host.id);
}

async function unload(name: string): Promise<void> {
  if (unloading.value.has(name)) return;
  unloading.value.add(name);
  try {
    await unloadModel(name, target.value);
    if (status.value) {
      const nextStatus = {
        ...status.value,
        models_loaded: status.value.models_loaded.filter((model) => model !== name),
      };
      status.value = nextStatus;
      installed.value = installed.value.map((model) =>
        model.name === name ? { ...model, is_loaded: false } : model,
      );
      emit("status", { id: props.host.id, status: nextStatus });
    }
  } catch (caught) {
    error.value = describeTransportError(caught, props.host.name);
  } finally {
    unloading.value.delete(name);
  }
}

function queueCode(entry: QueueEntry): string {
  if (entry.state === "running")
    return entry.gpu == null ? "RUNNING" : `RUNNING · GPU ${entry.gpu}`;
  return entry.position > 0 ? `QUEUED #${entry.position}` : "QUEUED";
}

function downloadPercent(done: number, total: number): number {
  return total > 0 ? percent(done, total) : 0;
}

watch(
  () => [props.host.id, props.host.baseUrl, props.host.apiKey, props.host.connected] as const,
  loadHost,
  {
    immediate: true,
  },
);
onBeforeUnmount(() => {
  loadEpoch += 1;
  stopLiveServices();
});
</script>

<template>
  <div class="mobile-detail" data-test="mobile-host-detail">
    <div class="mobile-detail-nav">
      <button
        class="mobile-back-button"
        type="button"
        data-test="host-detail-back"
        @click="emit('back')"
      >
        <span aria-hidden="true">‹</span> Hosts
      </button>
      <span class="host-chip">{{
        host.connected === false
          ? "disconnected"
          : host.online
            ? `v${host.version ?? ""}`
            : "offline"
      }}</span>
    </div>

    <div class="mobile-detail-title">
      <span class="status-dot" :class="host.online ? 'is-ready' : 'is-error'" aria-hidden="true" />
      <div>
        <h1 class="section-title">{{ host.name }}</h1>
        <p class="host-url">{{ host.baseUrl }}</p>
      </div>
    </div>

    <div class="row-actions mobile-host-actions">
      <button
        class="secondary-button"
        type="button"
        :disabled="active || host.connected === false || !host.online"
        data-test="host-detail-select"
        @click="emit('select', host.id)"
      >
        {{ active ? "Used for generations" : "Use for generations" }}
      </button>
      <button class="secondary-button" type="button" @click="renaming = !renaming">Rename</button>
      <button
        v-if="host.connected !== false"
        class="secondary-button"
        type="button"
        data-test="host-detail-disconnect"
        @click="emit('disconnect', host.id)"
      >
        Disconnect
      </button>
      <button
        v-else
        class="secondary-button"
        type="button"
        data-test="host-detail-reconnect"
        @click="emit('reconnect', host.id)"
      >
        Reconnect
      </button>
    </div>

    <form v-if="renaming" class="mobile-inline-form" @submit.prevent="saveRename">
      <label class="field">
        <span>Host name</span>
        <input v-model="renameValue" class="control" autocomplete="off" />
      </label>
      <div class="row-actions">
        <button class="primary-button" type="submit">Save name</button>
        <button class="secondary-button" type="button" @click="renaming = false">Cancel</button>
      </div>
    </form>

    <p v-if="host.connected === false" class="status-line">
      Disconnected. This host stays out of generation, Library, Models, and background checks.
    </p>
    <p v-else-if="loading" class="status-line">Reading host…</p>
    <div v-if="error" class="row-actions">
      <p class="status-line error-text" role="alert">{{ error }}</p>
      <button
        class="secondary-button"
        type="button"
        data-test="host-detail-retry"
        :disabled="loading"
        @click="loadHost"
      >
        Retry connection
      </button>
    </div>

    <template v-if="status">
      <MinimaxH3InventoryPanel :hosts="h3Host" heading="H3 on this machine" />

      <section class="mobile-detail-section" aria-labelledby="host-telemetry-title">
        <div class="mobile-section-head">
          <h2 id="host-telemetry-title">Telemetry</h2>
          <span v-if="uptime !== null">UP {{ formatUptime(uptime) }}</span>
        </div>
        <div v-if="gpus.length || cpu || ram || disk" class="telemetry-panel">
          <template v-for="gpu in gpus" :key="gpu.ordinal">
            <div class="telemetry-label">
              <strong>{{ gpu.name }}</strong>
              <span>{{ (gpu.backend || inferBackendFromGpuName(gpu.name)).toUpperCase() }}</span>
            </div>
            <div class="telemetry-meter-row">
              <span>{{ unifiedMemory ? "MEMORY" : "VRAM" }}</span>
              <div
                class="meter"
                role="meter"
                :aria-label="
                  unifiedMemory
                    ? `Unified memory usage for ${gpu.name}`
                    : `VRAM usage for ${gpu.name}`
                "
                :aria-valuenow="Math.round(percent(gpu.vram_used, gpu.vram_total))"
                aria-valuemin="0"
                aria-valuemax="100"
              >
                <span :style="{ width: `${percent(gpu.vram_used, gpu.vram_total)}%` }" />
              </div>
              <strong>{{ formatGB(gpu.vram_used) }}/{{ formatGB(gpu.vram_total) }}</strong>
            </div>
          </template>
          <div v-if="cpu" class="telemetry-meter-row">
            <span>CPU</span>
            <div
              class="meter"
              role="meter"
              aria-label="CPU usage"
              :aria-valuenow="Math.round(cpu.usage_percent)"
              aria-valuemin="0"
              aria-valuemax="100"
            >
              <span :style="{ width: `${cpu.usage_percent}%` }" />
            </div>
            <strong>{{ cpu.usage_percent.toFixed(0) }}% · {{ cpu.cores }} cores</strong>
          </div>
          <div v-if="ram && !unifiedMemory" class="telemetry-meter-row">
            <span>RAM</span>
            <div
              class="meter"
              role="meter"
              aria-label="RAM usage"
              :aria-valuenow="Math.round(percent(ram.used, ram.total))"
              aria-valuemin="0"
              aria-valuemax="100"
            >
              <span :style="{ width: `${percent(ram.used, ram.total)}%` }" />
            </div>
            <strong>{{ formatGB(ram.used) }}/{{ formatGB(ram.total) }}</strong>
          </div>
          <div v-if="disk" class="telemetry-meter-row">
            <span>DISK</span>
            <div
              class="meter"
              role="meter"
              aria-label="Models disk usage"
              :aria-valuenow="
                Math.round(percent(disk.total_bytes - disk.free_bytes, disk.total_bytes))
              "
              aria-valuemin="0"
              aria-valuemax="100"
            >
              <span
                :style="{
                  width: `${percent(disk.total_bytes - disk.free_bytes, disk.total_bytes)}%`,
                }"
              />
            </div>
            <strong>{{ formatGB(disk.free_bytes) }} free</strong>
          </div>
        </div>
        <p v-else class="mobile-empty-note">No live telemetry from this host yet.</p>
      </section>

      <section class="mobile-detail-section" aria-labelledby="host-queue-title">
        <div class="mobile-section-head">
          <h2 id="host-queue-title">Queue</h2>
          <span
            >{{ queueDepth
            }}<template v-if="status.queue_capacity">/{{ status.queue_capacity }}</template></span
          >
        </div>
        <div
          v-if="devices !== null || queuePlan !== null || deviceError"
          data-test="host-detail-devices"
        >
          <DevicePanel
            :devices="devices ?? []"
            :plan="queuePlan"
            :mutable="
              deviceCapabilities?.devices?.lifecycle === true &&
              deviceCapabilities?.dispatch?.v2_authoritative === true
            "
            :restart-enable="deviceCapabilities?.devices?.restart_enable === true"
            show-controls
            :busy-device-ids="[...deviceMutations]"
            @unpin="unpinWork"
            @toggle="toggleDeviceById"
          />
          <p
            v-if="deviceError"
            class="status-line error-text"
            role="alert"
            data-test="host-detail-device-error"
          >
            {{ deviceError }}
          </p>
        </div>
        <ul v-if="queue.length" class="mobile-data-list" data-test="host-detail-queue">
          <li v-for="entry in queue" :key="entry.id">
            <div>
              <strong>{{ modelLabel(entry.model) }}</strong>
              <span>{{ queueCode(entry) }}</span>
            </div>
            <button
              v-if="entry.state === 'queued'"
              type="button"
              class="mobile-inline-danger mobile-host-queue-cancel"
              :data-test="`host-detail-queue-cancel-${entry.id}`"
              :disabled="cancellingQueueIds.has(entry.id)"
              :aria-label="
                cancellingQueueIds.has(entry.id)
                  ? `Cancelling queued ${modelLabel(entry.model)} job`
                  : cancelConfirmId === entry.id
                    ? `Confirm cancellation of queued ${modelLabel(entry.model)} job`
                    : `Cancel queued ${modelLabel(entry.model)} job`
              "
              @click="cancelQueuedJob(entry)"
              @blur="clearCancelConfirmation(entry.id)"
            >
              {{
                cancellingQueueIds.has(entry.id)
                  ? "Cancelling…"
                  : cancelConfirmId === entry.id
                    ? "Cancel?"
                    : "Cancel"
              }}
            </button>
          </li>
        </ul>
        <p v-else class="mobile-empty-note">Queue is empty.</p>

        <div v-if="loadedModels.length" class="mobile-chip-list" aria-label="Loaded models">
          <span v-for="model in loadedModels" :key="model" class="mobile-model-chip">
            <span>{{ modelLabel(model) }}</span>
            <button
              type="button"
              :disabled="unloading.has(model)"
              :aria-label="`Unload ${modelLabel(model)}`"
              @click="unload(model)"
            >
              {{ unloading.has(model) ? "…" : "×" }}
            </button>
          </span>
        </div>
        <p v-else class="mobile-empty-note">No models are loaded on the GPU.</p>
      </section>

      <section
        v-if="inFlightDownloads.length"
        class="mobile-detail-section"
        aria-labelledby="host-downloads-title"
      >
        <div class="mobile-section-head">
          <h2 id="host-downloads-title">Downloads</h2>
          <span>{{ inFlightDownloads.length }}</span>
        </div>
        <ul class="mobile-data-list">
          <li v-for="job in inFlightDownloads" :key="job.id">
            <div class="download-row-copy">
              <strong>{{ modelLabel(job.model) }}</strong>
              <span>{{ downloadStatus(job) }}</span>
              <div
                v-if="job.status === 'active'"
                class="meter"
                role="meter"
                :aria-label="`Download progress for ${modelLabel(job.model)}`"
                :aria-valuenow="downloadPercent(job.bytes_done, job.bytes_total)"
                aria-valuemin="0"
                aria-valuemax="100"
              >
                <span :style="{ width: `${downloadPercent(job.bytes_done, job.bytes_total)}%` }" />
              </div>
            </div>
          </li>
        </ul>
      </section>

      <section class="mobile-detail-section" aria-labelledby="host-models-title">
        <div class="mobile-section-head">
          <h2 id="host-models-title">Models on this host</h2>
          <button type="button" data-test="host-detail-catalog" @click="emit('catalog', host.id)">
            Catalog ›
          </button>
        </div>
        <ul v-if="installed.length" class="mobile-data-list" data-test="host-detail-models">
          <li v-for="model in installed" :key="model.name">
            <div>
              <strong>{{ modelDisplayName(model) }}</strong>
              <span
                >{{ model.family
                }}<template v-if="modelSizeLabels(model).weights">
                  · {{ modelSizeLabels(model).weights }}</template
                ></span
              >
            </div>
            <span v-if="model.is_loaded" class="status-badge">LOADED</span>
          </li>
        </ul>
        <p v-else class="mobile-empty-note">No installed models reported.</p>
      </section>
    </template>

    <section class="mobile-danger-zone">
      <button
        class="danger-button"
        type="button"
        data-test="host-detail-forget"
        @click="requestForget"
        @blur="forgetPending = false"
      >
        {{ forgetPending ? `Forget ${host.name}?` : "Forget host" }}
      </button>
      <p>Forgetting removes this address and its API key from this iPhone.</p>
    </section>
  </div>
</template>
