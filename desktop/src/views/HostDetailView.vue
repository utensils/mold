<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import { RouterLink, useRoute, useRouter } from "vue-router";
import Tooltip from "@ui/components/Tooltip.vue";
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import DevicePanel from "@studio/components/DevicePanel.vue";
import MinimaxH3InventoryPanel from "@studio/components/MinimaxH3InventoryPanel.vue";
import { setQueueDevicePin } from "@studio/api/queuePlan";
import { queuePlanOnlyWork } from "@studio/lib/queuePlanPresentation";
import { modelKindLabel, modelKindValue } from "@studio/lib/modelMetadata";
import { setDeviceEnabled } from "@studio/api/devices";
import { emptyTrash, listTrash } from "@studio/api/galleryOrganization";
import { RETENTION_OPTIONS, retentionLabel } from "@studio/lib/libraryOrganization";
import CatalogDetailDrawer from "../components/models/CatalogDetailDrawer.vue";
import DownloadsTray from "../components/models/DownloadsTray.vue";
import HostQueuePanel from "../components/machines/HostQueuePanel.vue";
import ModelTableRow from "../components/models/ModelTableRow.vue";
import RenameDialog from "../components/shell/RenameDialog.vue";
import ConfirmDialog from "../components/shell/ConfirmDialog.vue";
import { startCatalogDownload } from "../lib/api/catalog";
import { unloadModel } from "../lib/api/models";
import { ApiError, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { fetchHostConfigKey, setHostConfigKey } from "../lib/api/hostConfig";
import { gpuSnapshotsFromWorkers } from "../lib/api/gpuStatus";
import { installedModelToEntry } from "../lib/catalogDetail";
import { sseStream } from "../lib/api/sse";
import { subscribeToDeviceSnapshots } from "../lib/api/deviceEvents";
import { hostMemoryLevel, hostMemoryScheduleLabel } from "@studio/lib/hostMemory";
import { formatGB, formatGBPair, percent, vramLevel } from "../lib/format";
import { unifiedMemoryHost } from "@studio/lib/telemetryMemory";
import { inferBackendFromGpuName } from "../lib/hosts";
import { machineSentence } from "../lib/machineSentence";
import {
  isOpaqueModelId,
  modelDiskBytes,
  modelDisplayName,
  modelDisplayNameForId,
  modelSizeLabels,
} from "../lib/models";
import { modelSource } from "../lib/modelSource";
import { ipc } from "../lib/ipc";
import type {
  ConfigRow,
  GpuSnapshot,
  ModelEntry,
  ResourceSnapshot,
  ServerStatus,
} from "../lib/api/types";
import { useDownloadsStore } from "../stores/downloads";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useJobsStore } from "../stores/jobs";
import { useToastStore } from "../stores/toasts";

/** `ResourceSnapshot` plus the additive `cpu` wire field (mold-core
 *  `CpuSnapshot`); `null`/absent until the server's sampler has two frames,
 *  and always absent on servers that predate CPU sampling. */
type DetailSnapshot = ResourceSnapshot & {
  cpu?: { cores: number; usage_percent: number } | null;
};

const route = useRoute();
const router = useRouter();
const downloads = useDownloadsStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
const jobs = useJobsStore();
const toasts = useToastStore();

function modelAccessibilityLabel(model: ModelEntry): string {
  const classification = [
    modelKindLabel(modelKindValue({ kind: model.kind ?? null, family: model.family })),
    ...(model.nsfw ? ["18+ NSFW"] : []),
  ].join(", ");
  return `${modelDisplayName(model)} — ${classification}, view details`;
}

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
let deviceEventsAbort: AbortController | null = null;

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

function startDeviceEvents() {
  deviceEventsAbort?.abort();
  deviceEventsAbort = null;
  const target = hostTarget();
  if (!target) return;
  deviceEventsAbort = new AbortController();
  subscribeToDeviceSnapshots(target, deviceEventsAbort.signal, tickQueue);
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
  void refreshStorage();
}

// ── Storage: this host's Library trash (retention + count) ────────────────
//
// Organization state lives per host in that host's mold.db (D1), so retention
// is THAT host's `gallery.trash_retention_days` — edited here through the
// per-host `/api/config` helper, never through the primary-only settings
// store. Hidden entirely when the host's capabilities lack `gallery.trash`
// (older servers keep today's hard-delete wording everywhere).

const TRASH_RETENTION_KEY = "gallery.trash_retention_days";

const trashCapability = computed(() => hosts.capabilities[hostId.value]?.gallery?.trash ?? null);
const storageAvailable = computed(() => trashCapability.value?.enabled === true);
const retentionRow = ref<ConfigRow | null>(null);
const retentionSaving = ref(false);
const trashCount = ref<number | null>(null);
const trashLoadError = ref<string | null>(null);
const emptyTrashOpen = ref(false);
const emptyingTrash = ref(false);
let storageAbort: AbortController | null = null;

/** The retention the select shows: the host's config row, else the value
 *  its capabilities advertised, else the engine default. */
const retentionDays = computed(() => {
  const raw = retentionRow.value?.value;
  const fromRow =
    typeof raw === "number" ? raw : raw != null && raw !== "" ? Number(raw) : Number.NaN;
  if (Number.isFinite(fromRow) && fromRow >= 0) return Math.floor(fromRow);
  return trashCapability.value?.retention_days ?? 30;
});
const retentionLocked = computed(() => retentionRow.value?.source === "env");
/** Curated choices plus the current value when it is not one of them, so the
 *  select always tells the truth about the host. */
const retentionOptions = computed(() => {
  const days = retentionDays.value;
  const values = RETENTION_OPTIONS.includes(days)
    ? [...RETENTION_OPTIONS]
    : [...RETENTION_OPTIONS.filter((d) => d !== 0), days, 0];
  return values.map((value) => ({ value, label: retentionLabel(value) }));
});

async function refreshStorage() {
  storageAbort?.abort();
  const abort = new AbortController();
  storageAbort = abort;
  const target = hostTarget();
  if (!target || !storageAvailable.value) return;
  const [row, trashed] = await Promise.allSettled([
    fetchHostConfigKey(target, TRASH_RETENTION_KEY, abort.signal),
    listTrash(target, abort.signal),
  ]);
  if (abort.signal.aborted) return;
  if (row.status === "fulfilled") retentionRow.value = row.value;
  if (trashed.status === "fulfilled") {
    trashCount.value = trashed.value.length;
    trashLoadError.value = null;
  } else {
    trashLoadError.value =
      trashed.reason instanceof Error ? trashed.reason.message : String(trashed.reason);
  }
}

async function onRetentionChange(event: Event) {
  const target = hostTarget();
  const h = host.value;
  if (!target || !h) return;
  const select = event.target as HTMLSelectElement;
  const days = Number(select.value);
  if (!Number.isFinite(days) || days < 0) return;
  retentionSaving.value = true;
  try {
    await setHostConfigKey(target, TRASH_RETENTION_KEY, days);
    retentionRow.value = {
      key: TRASH_RETENTION_KEY,
      source: "db",
      ...(retentionRow.value ?? {}),
      value: days,
    } as ConfigRow;
    toasts.push(`Trash retention on ${h.label}: ${retentionLabel(days)}`);
    void refreshStorage();
  } catch (error) {
    toasts.push(
      `Couldn't change trash retention on ${h.label}: ${
        error instanceof Error ? error.message : String(error)
      }`,
      "error",
    );
    // Put the select back on the host's truth.
    select.value = String(retentionDays.value);
  } finally {
    retentionSaving.value = false;
  }
}

const emptyTrashMessage = computed(() => {
  const n = trashCount.value ?? 0;
  const label = host.value?.label ?? "this machine";
  return `Delete ${n} ${n === 1 ? "print" : "prints"} in the trash on ${label} forever? This can't be undone.`;
});

async function confirmEmptyTrash() {
  const target = hostTarget();
  const h = host.value;
  if (!target || !h) return;
  emptyingTrash.value = true;
  try {
    const result = await emptyTrash(target);
    emptyTrashOpen.value = false;
    trashCount.value = 0;
    toasts.push(
      `Emptied the trash on ${h.label} — ${result.purged} ${
        result.purged === 1 ? "print" : "prints"
      } deleted forever`,
    );
    void refreshStorage();
  } catch (error) {
    toasts.push(
      `Couldn't empty the trash on ${h.label}: ${
        error instanceof Error ? error.message : String(error)
      }`,
      "error",
    );
  } finally {
    emptyingTrash.value = false;
  }
}

// Capabilities often arrive after the identity watch's first run (the hosts
// store polls them); the card appears — and loads — as soon as they do.
watch(storageAvailable, (available) => {
  if (available) void refreshStorage();
});

// Clicking a model row opens the shared detail drawer against THIS host.
// Declared before the identity watch so its immediate run can reset it.
const detailModel = ref<ModelEntry | null>(null);
const drawerRepairing = ref(false);

// Loaded-model chip unload state — declared here (not with its handlers)
// because the identity watch's immediate run resets it.
const unloading = ref<Set<string>>(new Set());
/** Name awaiting the confirming second click before it unloads. */
const unloadPending = ref<string | null>(null);
/** Optimistically hidden until the telemetry poll confirms the unload. */
const recentlyUnloaded = ref<Set<string>>(new Set());

// Connection identity retargeting is the only event that replaces page data.
// A health poll changing ready/error state must not clear and rebuild the
// telemetry, storage, or models sections: those components keep their last
// snapshot mounted while their own live sources update them in place.
watch(
  [hostId, () => host.value?.baseUrl, () => host.value?.apiKey],
  (identity, previous) => {
    const hostChanged = !previous || identity[0] !== previous[0] || identity[1] !== previous[1];
    // A drawer left open for the previous host must not retarget its model
    // (and Repair action) at the next one — nor may a queue drawer or an
    // optimistic unload survive a host switch.
    if (hostChanged) {
      detailModel.value = null;
      recentlyUnloaded.value.clear();
      retentionRow.value = null;
      trashCount.value = null;
      trashLoadError.value = null;
      emptyTrashOpen.value = false;
    }
    startResourceStream(hostChanged);
    startDeviceEvents();
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
  storageAbort?.abort();
  storageAbort = null;
  resourceAbort?.abort();
  resourceAbort = null;
  deviceEventsAbort?.abort();
  deviceEventsAbort = null;
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
  return gpuSnapshotsFromWorkers(telemetry.value?.gpuInfo, telemetry.value?.gpuWorkers);
});

function backendLabel(gpu: GpuSnapshot): string {
  return (gpu.backend || inferBackendFromGpuName(gpu.name)).toUpperCase();
}

/** Which card this meter belongs to, for the tile's tooltip. */
function gpuDetail(gpu: GpuSnapshot): string {
  const util =
    gpu.gpu_utilization === null || gpu.gpu_utilization === undefined
      ? ""
      : `${gpu.gpu_utilization}% util`;
  return [gpu.name, backendLabel(gpu), util].filter(Boolean).join(" · ");
}

const cpu = computed(() => snapshot.value?.cpu ?? null);
const ram = computed(() => snapshot.value?.system_ram ?? null);
/** Apple Metal shares one physical pool — a VRAM row and a RAM row would
 *  show the same numbers twice, so unified hosts render one Memory row. */
const unifiedMemory = computed(() => unifiedMemoryHost(gpus.value));
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
// This is an inventory surface (drill-in opens the read-only Pull/Repair
// drawer, never a Load action), so it must show every downloaded artifact —
// including a runtime-restricted download-only row such as an NVFP4 H3
// partition — never `installedOn`'s runtime-filtered view Create/routing use.
const installedModels = computed(() => hostModels.downloadedOn(hostId.value));
const modelLabel = (name: string) => modelDisplayNameForId(name, hostModels.modelsOn(hostId.value));
const h3Host = computed(() => [
  {
    id: hostId.value,
    label: host.value?.label ?? hostId.value,
    capabilities: hosts.capabilities[hostId.value],
  },
]);

const queueSnapshot = computed(() => jobs.queues[hostId.value] ?? null);
const scheduledWorkCount = computed(() => queuePlanOnlyWork(queueSnapshot.value?.plan, []).length);
const mutatingDeviceIds = ref(new Set<string>());
const queuePaused = computed(
  () =>
    queueSnapshot.value?.paused === true ||
    queueSnapshot.value?.entries.some((entry) => entry.state === "paused") === true,
);

async function toggleDeviceById(deviceId: string, enabled: boolean) {
  const target = hostTarget();
  if (!target) return;
  mutatingDeviceIds.value = new Set(mutatingDeviceIds.value).add(deviceId);
  try {
    await setDeviceEnabled(target, deviceId, enabled);
    await hosts.refresh();
    tickQueue();
  } catch (error) {
    toasts.push(
      `Couldn't ${enabled ? "enable" : "disable"} device: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  } finally {
    const next = new Set(mutatingDeviceIds.value);
    next.delete(deviceId);
    mutatingDeviceIds.value = next;
  }
}

async function unpinWork(workId: string) {
  const target = hostTarget();
  if (!target) return;
  try {
    await setQueueDevicePin(target, workId, null);
    tickQueue();
  } catch (error) {
    toasts.push(
      `Queue pin was not changed: ${error instanceof Error ? error.message : String(error)}`,
      "error",
    );
  }
}

// ── Loaded-model chips: per-host unload ───────────────────────────────────

const loadedChips = computed(() =>
  modelsLoaded.value.filter((name) => !recentlyUnloaded.value.has(name)),
);

watch(modelsLoaded, (models) => {
  for (const name of [...recentlyUnloaded.value]) {
    if (!models.includes(name)) recentlyUnloaded.value.delete(name);
  }
});

async function unloadChip(name: string) {
  const h = host.value;
  if (!h || unloading.value.has(name)) return;
  // Inline confirm: the first click arms, the second unloads.
  if (unloadPending.value !== name) {
    unloadPending.value = name;
    return;
  }
  unloadPending.value = null;
  unloading.value.add(name);
  try {
    await unloadModel(name, hostTarget() ?? undefined);
    recentlyUnloaded.value.add(name);
    toasts.push(`Unloaded ${name} on ${h.label}`);
    void fetchStatus();
    void hostModels.refresh(true);
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  } finally {
    unloading.value.delete(name);
  }
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
  if (vramLevel(gpu.vram_used, gpu.vram_total) === "critical") return "bg-error";
  return hostBusy.value ? "bg-accent" : "bg-sapphire";
}

/** The RAM meter colors off the scheduler's own ledger rather than used/total:
 *  reservations that have not allocated yet still park the queue, and the OS
 *  cannot see them. Absent on older servers, which keeps today's plain bar. */
const hostMemoryPressure = computed(() => hostMemoryLevel(queueSnapshot.value?.plan?.host_memory));
const ramFill = computed(() => {
  switch (hostMemoryPressure.value) {
    case "critical":
      return "bg-error";
    case "warn":
      return "bg-accent";
    default:
      return "bg-sapphire";
  }
});
const ramPressureLabel = computed(() => {
  const memory = queueSnapshot.value?.plan?.host_memory;
  if (!memory) return null;
  return hostMemoryScheduleLabel(memory, formatGB);
});

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
    toasts.push(`Repairing ${modelDisplayName(m)} on ${h.label}`);
  } catch (err) {
    toasts.push(
      err instanceof ApiError && err.status === 409
        ? `${modelDisplayName(m)} is already queued.`
        : String(err),
      "error",
    );
  } finally {
    drawerRepairing.value = false;
  }
}

function statusDot(s: "connecting" | "ready" | "error"): string {
  switch (s) {
    case "ready":
      return "bg-success";
    case "connecting":
      return "bg-sapphire ms-pulse";
    default:
      return "bg-error";
  }
}

/**
 * The toolbar's one plain sentence — what the machine is, where it lives, how
 * long it has been up. It is the machine card's own sentence plus the uptime:
 * one builder, so the list and the page cannot describe the same box two ways.
 * The address stays off it and rides the name's tooltip instead.
 */
const hostSentence = computed(() =>
  host.value ? machineSentence(host.value, gpus.value, { uptimeSeconds: uptime.value }) : "",
);

/** The wire facts a person only needs when something is wrong: address,
 *  version, instance id. They ride the name's tooltip, not the toolbar. */
const identityDetail = computed(() => {
  const h = host.value;
  if (!h) return "";
  const facts = [h.baseUrl, h.version ? `v${h.version}` : "", h.instanceId ?? ""].filter(Boolean);
  return h.instanceId ? `${facts.join(" · ")} — click to copy the instance id` : facts.join(" · ");
});

/** The id the downloads store tags THIS host's rows with. A remote's rows
 *  carry its own host id, but the primary's carry whatever scope the store
 *  last subscribed as — `"primary"` until a host row claims it, never this
 *  route's `"local"` — so the store is asked rather than assumed. Comparing
 *  the route id left "Downloads here" permanently empty for this device
 *  whenever another surface had subscribed without a host. */
const downloadsHostId = computed(() =>
  host.value?.primary ? downloads.primaryHostId : hostId.value,
);

/** Whether the Downloads-here card has anything to show under its tray. */
const hostDownloading = computed(() =>
  downloads.hostedInFlight.some((row) => row.hostId === downloadsHostId.value),
);

// ── Actions ───────────────────────────────────────────────────────────────

const renameOpen = ref(false);

async function copyInstanceId() {
  const id = host.value?.instanceId;
  if (!id) return;
  try {
    await navigator.clipboard.writeText(id);
    toasts.push("Instance ID copied");
  } catch {
    toasts.push("Couldn't copy the instance ID", "error");
  }
}

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

// Forget drops the saved entry AND the stored API key — confirmed first.
const forgetOpen = ref(false);

async function forget() {
  const h = host.value;
  forgetOpen.value = false;
  if (!h) return;
  await hosts.disconnect(h.id);
  await ipc.forgetRemoteHost(h.id);
  toasts.push(`Forgot ${h.label}`);
  void router.push("/machines");
}
</script>

<template>
  <div class="flex h-full min-h-0 w-full flex-col" data-test="host-detail-content">
    <template v-if="host">
      <!-- toolbar: dot · mono name · one plain sentence · the machine's actions -->
      <div
        class="flex h-[var(--mold-shell-viewbar-h)] shrink-0 items-center gap-2.5 border-b border-border bg-chrome px-3.5"
        data-test="host-toolbar"
      >
        <span
          class="h-2 w-2 shrink-0 rounded-full"
          :class="statusDot(host.status)"
          data-test="host-status-dot"
        />
        <Tooltip :text="identityDetail" data-test="host-identity" class="shrink-0">
          <h1 class="font-mono text-base font-bold text-fg">
            <button type="button" data-test="host-title" @click="copyInstanceId">
              {{ host.label }}
            </button>
          </h1>
        </Tooltip>
        <span class="min-w-0 truncate text-xs text-fg-dim" data-test="host-sentence">
          {{ hostSentence }}
        </span>
        <span class="flex-1" />
        <!-- Three actions, the mock's. "Make images here" is reached from the
             machine card's context menu: a fourth nowrap button here needs
             more room than the pane has at the app's minimum width, and what
             it pushes off the end is Forget…. -->
        <button
          v-if="host.kind === 'remote'"
          type="button"
          data-test="rename-host"
          class="ms-toolbar-button"
          @click="renameOpen = true"
        >
          Rename
        </button>
        <button
          type="button"
          data-test="open-web-ui"
          class="ms-toolbar-button"
          :disabled="!host.baseUrl"
          @click="openHostUrl(host.baseUrl ?? '')"
        >
          Open web UI
        </button>
        <!-- Disconnect stays on the machine card's context menu: this toolbar
             carries the four actions the mock names and no more. -->
        <button
          v-if="host.kind === 'remote'"
          type="button"
          data-test="forget-host"
          class="ms-toolbar-button ms-toolbar-button--danger-hover"
          @click="forgetOpen = true"
        >
          Forget…
        </button>
      </div>

      <div class="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
        <p
          v-if="host.status === 'error'"
          class="flex items-center gap-3 rounded-control border border-error/50 bg-panel px-3 py-2 text-xs text-error"
        >
          Unreachable — it keeps retrying on its own, or check the server.
          <button
            v-if="host.kind === 'remote'"
            type="button"
            data-test="reconnect-host"
            class="ms-toolbar-button ml-auto"
            @click="hosts.reconnect(host.id)"
          >
            Try now
          </button>
        </p>

        <!-- Right now: one tile per meter, a shared shape so the row reads as one machine -->
        <section class="flex flex-col gap-2.5">
          <div class="flex items-center gap-2">
            <span class="ms-group-label uppercase">Right now</span>
            <span
              v-if="snapshot"
              class="flex items-center gap-1.5 font-mono text-micro text-success"
              ><span class="h-[5px] w-[5px] rounded-full bg-success ms-pulse" aria-hidden="true" />
              LIVE</span
            >
          </div>
          <div
            v-if="hasTelemetry"
            data-test="telemetry-panel"
            class="grid grid-cols-[repeat(auto-fit,minmax(11rem,1fr))] gap-2.5"
          >
            <div
              v-for="gpu in gpus"
              :key="gpu.ordinal"
              class="tile ms-card-edge"
              data-test="gpu-card"
            >
              <span class="ms-group-label uppercase">
                {{ unifiedMemory ? "Memory" : "Graphics memory"
                }}<template v-if="gpus.length > 1"> · GPU {{ gpu.ordinal }}</template>
              </span>
              <span class="text-lg font-semibold text-fg">
                {{ Math.round(percent(gpu.vram_used, gpu.vram_total)) }}%
              </span>
              <span
                class="block h-[5px] overflow-hidden bg-surface"
                role="meter"
                aria-valuemin="0"
                aria-valuemax="100"
                :aria-valuenow="Math.round(percent(gpu.vram_used, gpu.vram_total))"
                :aria-label="
                  unifiedMemory ? `Unified memory used on ${gpu.name}` : `VRAM used on ${gpu.name}`
                "
              >
                <span
                  class="block h-full transition-[width] duration-300"
                  :class="vramFill(gpu)"
                  :style="{ width: `${Math.round(percent(gpu.vram_used, gpu.vram_total))}%` }"
                />
              </span>
              <!-- One reading, the way the mock keeps it. The card, the GPU
                   and its backend ride the tooltip rather than truncating. -->
              <span
                class="font-mono text-micro text-fg-dim"
                :title="gpuDetail(gpu)"
                data-test="gpu-note"
              >
                {{ formatGBPair(gpu.vram_used, gpu.vram_total) }}
              </span>
            </div>
            <div v-if="cpu" class="tile ms-card-edge" data-test="cpu-card">
              <span class="ms-group-label uppercase">Processor</span>
              <span class="text-lg font-semibold text-fg">{{ cpu.usage_percent.toFixed(0) }}%</span>
              <span
                class="block h-[5px] overflow-hidden bg-surface"
                role="meter"
                aria-valuemin="0"
                aria-valuemax="100"
                :aria-valuenow="Math.round(cpu.usage_percent)"
                aria-label="CPU usage"
              >
                <span
                  class="block h-full bg-sapphire transition-[width] duration-300"
                  :style="{ width: `${Math.round(cpu.usage_percent)}%` }"
                />
              </span>
              <span class="font-mono text-micro text-fg-dim">{{ cpu.cores }} cores</span>
            </div>
            <div v-if="ram && !unifiedMemory" class="tile ms-card-edge" data-test="ram-card">
              <span class="ms-group-label uppercase">System memory</span>
              <span class="text-lg font-semibold text-fg">
                {{ Math.round(percent(ram.used, ram.total)) }}%
              </span>
              <span
                class="block h-[5px] overflow-hidden bg-surface"
                role="meter"
                aria-valuemin="0"
                aria-valuemax="100"
                :aria-valuenow="Math.round(percent(ram.used, ram.total))"
                aria-label="System RAM used"
                :title="ramPressureLabel ?? undefined"
                :data-pressure="hostMemoryPressure ?? undefined"
              >
                <span
                  class="block h-full transition-[width] duration-300"
                  :class="ramFill"
                  :style="{ width: `${Math.round(percent(ram.used, ram.total))}%` }"
                />
              </span>
              <span class="font-mono text-micro text-fg-dim">
                {{ formatGBPair(ram.used, ram.total) }}
              </span>
            </div>
            <div v-if="modelsDisk" class="tile ms-card-edge" data-test="storage-card">
              <span class="ms-group-label uppercase">Disk for styles</span>
              <span class="text-lg font-semibold text-fg">{{ Math.round(diskUsedPct) }}%</span>
              <span
                class="block h-[5px] overflow-hidden bg-surface"
                role="meter"
                aria-valuemin="0"
                aria-valuemax="100"
                :aria-valuenow="Math.round(diskUsedPct)"
                aria-label="Disk for styles"
              >
                <span
                  class="block h-full transition-[width] duration-300"
                  :class="diskUsedPct >= 92 ? 'bg-error' : 'bg-mauve'"
                  :style="{ width: `${Math.round(diskUsedPct)}%` }"
                />
              </span>
              <span class="font-mono text-micro text-fg-dim">
                {{ formatGB(modelsDisk.free_bytes) }} free of {{ formatGB(modelsDisk.total_bytes) }}
              </span>
            </div>
          </div>
          <p v-else class="text-xs text-fg-dim">No live readings from this machine yet.</p>
        </section>

        <!-- Loaded and ready -->
        <section class="flex flex-col gap-2">
          <span class="ms-group-label uppercase" data-test="loaded-label">Loaded and ready</span>
          <div v-if="loadedChips.length" class="flex flex-wrap gap-2">
            <span
              v-for="m in loadedChips"
              :key="m"
              data-test="loaded-model-chip"
              class="inline-flex h-[30px] max-w-full items-center gap-2 rounded-control border border-border bg-panel px-2.5 text-xs text-fg"
            >
              <span class="font-mono text-star" aria-hidden="true">★</span>
              <span class="min-w-0 truncate" data-test="loaded-model-name">{{
                modelLabel(m)
              }}</span>
              <span
                v-if="modelLabel(m) !== m && !isOpaqueModelId(m)"
                class="font-mono text-micro text-fg-dim"
                >{{ m }}</span
              >
              <button
                type="button"
                data-test="unload-chip"
                class="shrink-0 text-micro transition-colors hover:text-error disabled:opacity-40"
                :class="unloadPending === m ? 'font-semibold text-error' : 'text-fg-dim'"
                :aria-label="`Unload ${modelLabel(m)}`"
                :title="`Unload ${modelLabel(m)} from this machine's GPU`"
                :disabled="unloading.has(m)"
                @click="unloadChip(m)"
                @blur="unloadPending = null"
              >
                {{ unloading.has(m) ? "…" : unloadPending === m ? "Unload?" : "Unload" }}
              </button>
            </span>
          </div>
          <p v-else class="text-xs text-fg-dim">
            Nothing loaded — the next image loads its style first.
          </p>
        </section>

        <!-- Waiting on this machine: the whole server queue, other clients included -->
        <section class="flex flex-col gap-2">
          <div class="flex items-center gap-2">
            <span class="ms-group-label uppercase">Waiting on this machine</span>
            <span
              v-if="queuePaused"
              class="font-mono text-micro text-error"
              data-test="queue-paused"
              >PAUSED</span
            >
            <span class="flex-1" />
            <!-- The noun rides the number in every case: a bare "3/8" beside
                 a heading is not a reading. -->
            <span class="font-mono text-micro text-fg-dim" data-test="queue-depth">
              <template v-if="scheduledWorkCount">{{ scheduledWorkCount }} work · </template>
              {{ queueDepth ?? "—" }}<template v-if="queueCapacity">/{{ queueCapacity }}</template>
              queued
            </span>
          </div>
          <!-- The panel's rows draw their own border and radius; a second one
               here would be a border inside a border, and the mock has no
               Pause / Cancel-all row on this page. -->
          <HostQueuePanel
            :host="host"
            row-test-id="host-queue-row"
            empty-label="Queue is empty."
            :thumbnails="false"
            :controls="false"
          />
        </section>

        <div class="rounded-control border border-border bg-panel p-3.5">
          <DevicePanel
            :devices="queueSnapshot?.devices ?? []"
            :plan="queueSnapshot?.plan ?? null"
            :mutable="
              queueSnapshot?.devices !== null &&
              hosts.capabilities[hostId]?.devices?.lifecycle === true &&
              hosts.capabilities[hostId]?.dispatch?.v2_authoritative === true
            "
            :restart-enable="hosts.capabilities[hostId]?.devices?.restart_enable === true"
            show-controls
            :busy-device-ids="[...mutatingDeviceIds]"
            @unpin="unpinWork"
            @toggle="toggleDeviceById"
          />
        </div>

        <!-- Storage · Downloads here. The pair pairs up when THIS PANE is wide
             enough, never when the window is: `minWidth` is 1080, so a
             viewport breakpoint is always true and would put two columns in a
             484px pane. -->
        <div class="host-pair-shell">
          <div class="host-pair grid grid-cols-1 gap-3">
            <!-- Storage — this machine's own trash retention and count, behind the
               shared plain confirm (never a typed phrase). -->
            <section
              v-if="storageAvailable"
              class="flex flex-col gap-2.5 rounded-control border border-border bg-panel p-3.5"
              data-test="host-storage"
            >
              <span class="ms-group-label uppercase">Storage</span>
              <span v-if="installedTotalLabel" class="text-xs text-fg-2">
                Styles take {{ installedTotalLabel }}
              </span>
              <div class="flex items-center gap-2">
                <label for="host-trash-retention" class="text-xs text-fg">
                  Keep deleted pictures for
                </label>
                <select
                  id="host-trash-retention"
                  data-test="host-trash-retention"
                  class="h-[26px] rounded-control border border-border bg-bg px-1.5 font-mono text-micro text-fg-2 disabled:opacity-50"
                  :value="String(retentionDays)"
                  :disabled="retentionLocked || retentionSaving || host.status !== 'ready'"
                  :title="
                    retentionLocked
                      ? `Set by ${retentionRow?.env_var ?? 'the environment'} on ${host.label}`
                      : undefined
                  "
                  aria-label="Trash retention"
                  @change="onRetentionChange"
                >
                  <option
                    v-for="option in retentionOptions"
                    :key="option.value"
                    :value="String(option.value)"
                  >
                    {{ option.label }}
                  </option>
                </select>
                <span class="flex-1" />
                <button
                  type="button"
                  data-test="host-empty-trash"
                  class="ms-toolbar-button ms-toolbar-button--danger"
                  :disabled="!trashCount || emptyingTrash || host.status !== 'ready'"
                  @click="emptyTrashOpen = true"
                >
                  Empty trash
                </button>
              </div>
              <span class="text-micro text-fg-dim" data-test="host-trash-count">
                Pictures in trash: <span class="font-mono">{{ trashCount ?? "—" }}</span>
                <span v-if="trashLoadError" class="block text-error">{{ trashLoadError }}</span>
              </span>
            </section>

            <section
              class="flex flex-col gap-2.5 rounded-control border border-border bg-panel p-3.5"
            >
              <span class="flex items-center gap-2">
                <span class="ms-group-label uppercase">Downloads here</span>
                <span class="flex-1" />
                <RouterLink to="/models" class="text-micro text-fg-dim hover:text-fg">
                  Browse more styles →
                </RouterLink>
              </span>
              <DownloadsTray :host-id="downloadsHostId" compact data-test="host-downloads" />
              <span v-if="!hostDownloading" class="text-micro text-fg-dim">
                Nothing on its way to this machine.
              </span>
            </section>
          </div>
        </div>

        <!-- Styles on this machine -->
        <section class="flex flex-col gap-2" data-test="host-model-column">
          <div class="flex items-center gap-2">
            <span class="ms-group-label uppercase">Styles on this machine</span>
            <span class="flex-1" />
            <span
              v-if="installedModels.length"
              class="font-mono text-micro text-fg-dim"
              data-test="models-summary"
            >
              {{ installedModels.length
              }}<template v-if="installedTotalLabel"> · {{ installedTotalLabel }}</template>
            </span>
          </div>
          <ul
            v-if="installedModels.length"
            class="divide-y divide-border border border-border bg-panel"
          >
            <li v-for="m in installedModels" :key="m.name" data-test="model-row">
              <ModelTableRow
                :name="modelDisplayName(m)"
                :id="m.name"
                :source="modelSource(m)"
                :loaded="m.is_loaded"
                :family="m.family"
                :page-url="m.hf_repo ? `https://huggingface.co/${m.hf_repo}` : null"
                :note="m.description || null"
                :size-primary="
                  modelSizeLabels(m).weights ?? modelSizeLabels(m).runtime ?? 'Size unavailable'
                "
                :size-secondary="
                  modelSizeLabels(m).weights && modelSizeLabels(m).runtime
                    ? modelSizeLabels(m).runtime
                    : null
                "
                :bar-percent="percent(modelDiskBytes(m), maxModelDiskBytes)"
                :accessibility-label="modelAccessibilityLabel(m)"
                clickable
                class="px-3"
                @open="detailModel = m"
              >
                <template #meta>
                  <ModelMetadataBadges
                    :kind="m.kind ?? null"
                    :family="m.family"
                    :nsfw="m.nsfw ?? null"
                    :show-modality="false"
                  />
                </template>
              </ModelTableRow>
            </li>
          </ul>
          <p v-else class="text-xs text-fg-dim">No styles reported</p>
        </section>

        <!-- Specialized capability detail reads below the live instruments. -->
        <MinimaxH3InventoryPanel :hosts="h3Host" heading="H3 on this machine" />
      </div>

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
        title="Rename machine"
        :initial="host.label"
        @save="onRenameSave"
        @cancel="renameOpen = false"
      />

      <ConfirmDialog
        :open="emptyTrashOpen"
        title="Empty trash?"
        :message="emptyTrashMessage"
        confirm-label="Delete forever"
        danger
        :busy="emptyingTrash"
        @confirm="confirmEmptyTrash"
        @cancel="emptyTrashOpen = false"
      />

      <ConfirmDialog
        :open="forgetOpen"
        :title="`Forget ${host.label}?`"
        message="Its saved API key is discarded."
        confirm-label="Forget"
        danger
        @confirm="forget"
        @cancel="forgetOpen = false"
      />
    </template>

    <!-- Unknown id — quiet empty state -->
    <div v-else class="mt-16 text-center" data-test="host-missing">
      <h1 class="text-lg font-semibold text-fg">Machine not found</h1>
      <p class="mt-2 text-sm text-fg-2">
        This machine isn't connected. It may have been disconnected or forgotten.
      </p>
      <RouterLink
        to="/machines"
        data-test="back-to-hosts"
        class="mt-4 inline-block text-sm text-accent hover:brightness-110"
      >
        Back to Machines
      </RouterLink>
    </div>
  </div>
</template>

<style scoped>
/* Right-now tile (README §04 telemetry): label · value · meter · note. */
.tile {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 7px;
  padding: 13px;
  background: var(--mold-panel);
}

/* Storage · Downloads here: two columns only once the PANE can hold them.
   The shell is the query container, so the pair's own box is measured and
   nothing else in the view gains a containing block. */
.host-pair-shell {
  container-type: inline-size;
}
@container (min-width: 34rem) {
  .host-pair {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
