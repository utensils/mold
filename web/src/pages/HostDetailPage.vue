<script setup lang="ts">
/*
 * Host detail (spec §04 / §08 G3). Live telemetry, the relocated queue
 * management card, active downloads, and installed models for one machine —
 * the primary origin or any remembered remote, all through the same per-host
 * client. Absent metrics render as em dashes and a reconnecting host keeps its
 * last-good data dimmed behind a retry banner (G4).
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import CardSurface from "@ui/components/CardSurface.vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import DevicePanel from "@studio/components/DevicePanel.vue";
import MinimaxH3InventoryPanel from "@studio/components/MinimaxH3InventoryPanel.vue";
import {
  mergeQueueEntries,
  queuePageRequestForCapacity,
  setQueueDevicePin,
  type QueuePlan,
} from "@studio/api/queuePlan";
import { hostMemoryLevel } from "@studio/lib/hostMemory";
import {
  modelDisplayName,
  modelDisplayNameForId,
} from "@studio/lib/modelDisplay";
import StatusDot from "../components/machines/StatusDot.vue";
import QueueCard from "../components/machines/QueueCard.vue";
import QueueEntryDetail from "@studio/components/QueueEntryDetail.vue";
import {
  queueEntryDetailModel,
  type QueueDetailMetadata,
} from "@studio/lib/queueEntryDetail";
import {
  settingsRestoreMetadata,
  watchSelectedQueuePreview,
  type QueueJobPreview,
} from "@studio/api/generationSelection";
import { setGenerationHandoff } from "../composables/useGenerationHandoff";
import {
  cancelQueueJob,
  cancelAllHostQueue,
  hostApiTarget,
  hostCapabilities,
  hostConfigValue,
  hostDownloads,
  hostGallery,
  hostModels,
  hostQueue,
  hostWriteConfig,
  moveQueueJob,
  pauseHostQueue,
  resumeHostQueue,
  setQueueJobLane,
  useHostPoll,
  type HostCapabilities,
} from "../components/machines/hostClient";
import { emptyTrash } from "@studio/api/galleryOrganization";
import {
  RETENTION_OPTIONS,
  retentionLabel,
} from "@studio/lib/libraryOrganization";
import { deriveTelemetry } from "../components/machines/machineTelemetry";
import {
  ORIGIN_HOST_ID,
  HOSTS_CHANGED_EVENT,
  HOSTS_STORAGE_KEY,
  getGenerateTargetId,
  getKnownHost,
  removeHost,
  setHostConnected,
  setGenerateTargetId,
  updateHost,
} from "../lib/hostRegistry";
import { ApiHttpError, reorderQueueJob, updateQueueJobTargetGpu } from "../api";
import { requestConfirm, requestText, toast } from "../lib/toasts";
import { subscribeToDeviceSnapshots } from "../lib/deviceEvents";
import type {
  DownloadJobWire,
  ModelInfoExtended,
  OutputMetadata,
  QueueEntry,
} from "../types";
import { setDeviceEnabled } from "@studio/api/devices";

const route = useRoute();
const router = useRouter();
const registryRevision = ref(0);
const hostId = computed(() => String(route.params.id));
const host = computed(() => {
  void registryRevision.value;
  return getKnownHost(hostId.value);
});
const isOrigin = computed(() => hostId.value === ORIGIN_HOST_ID);
const disconnected = computed(() => host.value?.connected === false);
const liveHost = computed(() => (disconnected.value ? null : host.value));
const hostName = ref(host.value?.name ?? "");

const caps = ref<HostCapabilities | null>(null);
const queue = ref<QueueEntry[]>([]);
const queuePlan = ref<QueuePlan | null>(null);
const queueTail = ref<QueueEntry[]>([]);
const queuePageLimit = ref<number | null>(null);
const queueNextCursor = ref<string | null>(null);
const queueContinued = ref(false);
const loadingMoreQueue = ref(false);
const loadMoreQueueError = ref("");
const cancellingIds = ref<string[]>([]);
const mutatingDeviceIds = ref(new Set<string>());
const models = ref<ModelInfoExtended[]>([]);
const downloads = ref<DownloadJobWire[]>([]);
const targetId = ref(getGenerateTargetId());

const poll = useHostPoll(liveHost, { withResources: true, intervalMs: 4000 });

const online = computed(() => poll.online.value);
const loading = computed(() => poll.loading.value);
const reconnecting = computed(
  () =>
    !!host.value &&
    !disconnected.value &&
    !loading.value &&
    (!online.value || poll.stale.value),
);
const dotState = computed<"online" | "offline" | "unknown">(() => {
  if (!host.value || disconnected.value || loading.value || reconnecting.value)
    return "unknown";
  return "online";
});

const telemetry = computed(() =>
  deriveTelemetry(poll.status.value ?? null, poll.resources.value ?? null),
);

/** RAM pressure from the scheduler's ledger rather than used/total — a
 * reservation that has not allocated yet still parks the queue and the OS
 * cannot see it. Absent on older servers, which keeps the plain info bar. */
const hostMemoryPressure = computed(() =>
  hostMemoryLevel(queuePlan.value?.host_memory),
);
const ramTone = computed<"info" | "warning" | "danger">(() => {
  switch (hostMemoryPressure.value) {
    case "critical":
      return "danger";
    case "warn":
      return "warning";
    default:
      return "info";
  }
});
const gpuOrdinals = computed(() => {
  const devices = poll.devices.value;
  if (devices !== null && devices !== undefined)
    return [
      ...new Set(
        devices
          .filter((device) => device.schedulable && device.ordinal !== null)
          .map((device) => device.ordinal as number),
      ),
    ].sort((a, b) => a - b);
  const status = poll.status.value;
  if (status?.gpus != null)
    return status.gpus
      .filter((gpu) => gpu.state !== "degraded")
      .map((gpu) => gpu.ordinal);
  return status?.gpu_info ? [0] : [];
});
const canReorder = computed(() => !!caps.value?.queue?.can_reorder);
const isTarget = computed(() => targetId.value === hostId.value);
const paused = computed(() => poll.status.value?.queue_paused === true);

const address = computed(() => {
  if (!host.value) return "";
  try {
    return new URL(host.value.url).host;
  } catch {
    return host.value.url;
  }
});

const installed = computed(() => models.value.filter((m) => m.downloaded));
const modelLabel = (name: string) => modelDisplayNameForId(name, models.value);
const h3Host = computed(() => [
  {
    id: hostId.value,
    label: hostName.value,
    capabilities: caps.value,
  },
]);

let sessionEpoch = 0;
let queueRequestGeneration = 0;
let capabilityRequestGeneration = 0;
let modelRequestGeneration = 0;
let downloadRequestGeneration = 0;
let queueLoadMoreGeneration = 0;

function isCurrentSession(
  entry: NonNullable<typeof host.value>,
  epoch: number,
) {
  const current = host.value;
  return (
    epoch === sessionEpoch &&
    current?.id === entry.id &&
    current.url === entry.url &&
    current.apiKey === entry.apiKey &&
    current.connected !== false
  );
}

async function reloadQueue(
  entry = host.value,
  epoch = sessionEpoch,
  signal?: AbortSignal,
) {
  if (!entry) return;
  const generation = ++queueRequestGeneration;
  try {
    const page = queuePageRequestForCapacity(poll.status.value?.queue_capacity);
    const listing = page
      ? await hostQueue(entry, signal, page)
      : await hostQueue(entry, signal);
    if (
      generation === queueRequestGeneration &&
      isCurrentSession(entry, epoch)
    ) {
      const head = mergeQueueEntries(
        listing.entries,
        listing.live_only_entries ?? [],
      ) as QueueEntry[];
      queuePlan.value = listing.plan ?? null;
      if (queueContinued.value || loadingMoreQueue.value) {
        // A routine head poll can refresh the live window without destroying
        // continuation pages the user already loaded or invalidating a page
        // currently in flight. Replace the head, retain the continuation
        // snapshot, and de-duplicate rows that advanced into the live window.
        const seen = new Set<string>();
        queue.value = [...head, ...queueTail.value].filter(
          ({ id }) => !seen.has(id) && !!seen.add(id),
        );
      } else {
        queue.value = head;
        queuePageLimit.value = listing.page?.limit ?? null;
        queueNextCursor.value = listing.page?.next_cursor ?? null;
        queueTail.value = [];
        queueContinued.value = false;
        loadMoreQueueError.value = "";
      }
    }
  } catch {
    // Keep the last-good queue; the reconnecting banner covers the failure.
  }
}

async function loadMoreQueue() {
  const entry = host.value;
  const cursor = queueNextCursor.value;
  const limit = queuePageLimit.value;
  if (!entry || !cursor || !limit || loadingMoreQueue.value) return;
  loadingMoreQueue.value = true;
  loadMoreQueueError.value = "";
  const epoch = sessionEpoch;
  const generation = ++queueLoadMoreGeneration;
  try {
    const listing = await hostQueue(entry, undefined, { limit, cursor });
    if (
      generation !== queueLoadMoreGeneration ||
      !isCurrentSession(entry, epoch) ||
      queueNextCursor.value !== cursor
    )
      return;
    if (!listing.page) {
      queue.value = listing.entries;
      queueTail.value = [];
      queuePageLimit.value = null;
      queueNextCursor.value = null;
      queueContinued.value = false;
      return;
    }
    const seenTail = new Set(queueTail.value.map(({ id }) => id));
    queueTail.value = [
      ...queueTail.value,
      ...(listing.entries as QueueEntry[]).filter(
        ({ id }) => !seenTail.has(id),
      ),
    ];
    const seen = new Set<string>();
    queue.value = [
      ...queue.value,
      ...queueTail.value,
      ...(listing.live_only_entries ?? []),
    ].filter(({ id }) => !seen.has(id) && !!seen.add(id)) as QueueEntry[];
    queueNextCursor.value = listing.page.next_cursor ?? null;
    queueContinued.value = true;
    queuePlan.value = listing.plan ?? queuePlan.value;
  } catch (error) {
    if (
      generation === queueLoadMoreGeneration &&
      isCurrentSession(entry, epoch)
    )
      loadMoreQueueError.value = errMsg(error);
  } finally {
    if (
      generation === queueLoadMoreGeneration &&
      isCurrentSession(entry, epoch)
    )
      loadingMoreQueue.value = false;
  }
}

async function onToggleDevice(deviceId: string, enabled: boolean) {
  const entry = host.value;
  if (!entry) return;
  const epoch = sessionEpoch;
  mutatingDeviceIds.value = new Set(mutatingDeviceIds.value).add(deviceId);
  try {
    const accepted = await setDeviceEnabled(
      { baseUrl: entry.url, apiKey: entry.apiKey ?? null },
      deviceId,
      enabled,
    );
    if (isCurrentSession(entry, epoch)) {
      const devices = (poll.devices.value ?? []).map((device) =>
        device.id === accepted.id ? accepted : device,
      );
      if (!devices.some((device) => device.id === accepted.id)) {
        devices.push(accepted);
      }
      poll.devices.value = devices;
      poll.deviceState.value = {
        devices,
        plan_version: poll.deviceState.value?.plan_version ?? 0,
      };
    }
    await poll.refresh();
    await reloadQueue();
  } catch (e) {
    toast(
      "error",
      `Couldn't ${enabled ? "enable" : "disable"} device: ${errMsg(e)}`,
    );
  } finally {
    const next = new Set(mutatingDeviceIds.value);
    next.delete(deviceId);
    mutatingDeviceIds.value = next;
  }
}

async function onUnpinWork(workId: string) {
  const entry = host.value;
  if (!entry) return;
  try {
    await setQueueDevicePin(
      { baseUrl: entry.url, apiKey: entry.apiKey ?? null },
      workId,
      null,
    );
    await reloadQueue();
  } catch (error) {
    toast("error", `Couldn't use Auto for queued work: ${errMsg(error)}`);
  }
}

async function reloadModels(
  entry = host.value,
  epoch = sessionEpoch,
  signal?: AbortSignal,
) {
  if (!entry) return;
  const generation = ++modelRequestGeneration;
  try {
    const next = await hostModels(entry, signal);
    if (generation === modelRequestGeneration && isCurrentSession(entry, epoch))
      models.value = next;
  } catch {
    /* keep stale */
  }
}

async function reloadDownloads(
  entry = host.value,
  epoch = sessionEpoch,
  signal?: AbortSignal,
) {
  if (!entry) return;
  const generation = ++downloadRequestGeneration;
  try {
    const listing = await hostDownloads(entry, signal);
    const active = [
      ...(listing.active ? [listing.active] : []),
      ...(listing.active_jobs ?? []),
      ...listing.queued,
    ];
    if (
      generation === downloadRequestGeneration &&
      isCurrentSession(entry, epoch)
    )
      downloads.value = active;
  } catch {
    /* keep stale */
  }
}

async function reloadAllOnce(
  entry = host.value,
  epoch = sessionEpoch,
  signal?: AbortSignal,
) {
  await Promise.all([
    reloadCapabilities(entry, epoch, signal).then(() =>
      reloadLibrary(entry, epoch, signal),
    ),
    reloadQueue(entry, epoch, signal),
    reloadModels(entry, epoch, signal),
    reloadDownloads(entry, epoch, signal),
  ]);
}

// ── Library (trash retention + trash count) ────────────────────────────────
// Shown only when the host advertises `gallery.trash`; the retention value is
// that host's own `gallery.trash_retention_days`, read and written with its
// key (per-host, like every other setting on this page).
const RETENTION_KEY = "gallery.trash_retention_days";
// `trash.enabled === false` is a real answer (MOLD_DB_DISABLE, output
// disabled): delete stays permanent there, so the Library card and its
// config/trash polling must stay hidden — presence alone is not consent.
const trashCapable = computed(
  () => caps.value?.gallery?.trash?.enabled === true,
);
const trashRetentionDays = ref<number | null>(null);
const trashCount = ref<number | null>(null);
const savingRetention = ref(false);
const retentionChoices = computed(() => {
  const current = trashRetentionDays.value;
  const options = [...RETENTION_OPTIONS];
  if (current !== null && !options.includes(current)) options.push(current);
  return options.map((days) => ({ value: days, label: retentionLabel(days) }));
});

async function reloadLibrary(
  entry = host.value,
  epoch = sessionEpoch,
  signal?: AbortSignal,
) {
  if (!entry || !isCurrentSession(entry, epoch)) return;
  if (!caps.value?.gallery?.trash) {
    trashRetentionDays.value = null;
    trashCount.value = null;
    return;
  }
  const [retention, trashed] = await Promise.all([
    hostConfigValue(entry, RETENTION_KEY, signal).catch(() => null),
    hostGallery(entry, signal, "trash").catch(() => null),
  ]);
  if (!isCurrentSession(entry, epoch)) return;
  const value =
    typeof retention === "number"
      ? retention
      : typeof retention === "string" && retention.trim() !== ""
        ? Number(retention)
        : null;
  trashRetentionDays.value =
    value !== null && Number.isFinite(value)
      ? value
      : (caps.value?.gallery?.trash?.retention_days ?? null);
  trashCount.value = trashed ? trashed.length : null;
}

async function onRetentionChange(raw: string) {
  const entry = host.value;
  if (!entry) return;
  const days = Number(raw);
  if (!Number.isFinite(days)) return;
  savingRetention.value = true;
  try {
    await hostWriteConfig(entry, RETENTION_KEY, days);
    trashRetentionDays.value = days;
    toast(
      "success",
      `Trash retention on ${hostName.value}: ${retentionLabel(days)}`,
    );
    await reloadCapabilities();
    await reloadLibrary();
  } catch (e) {
    toast("error", `Couldn't change trash retention: ${errMsg(e)}`);
  } finally {
    savingRetention.value = false;
  }
}

async function onEmptyTrash() {
  const entry = host.value;
  if (!entry) return;
  const count = trashCount.value ?? 0;
  const accepted = await requestConfirm({
    title: "Empty trash?",
    body: `Delete ${count} ${count === 1 ? "print" : "prints"} in the trash on ${hostName.value} forever? This can't be undone.`,
    confirmLabel: "Delete forever",
    danger: true,
  });
  if (!accepted) return;
  try {
    const result = await emptyTrash(hostApiTarget(entry));
    toast(
      "success",
      `Emptied the trash on ${hostName.value} (${result.purged} purged)`,
    );
    trashCount.value = 0;
    await reloadLibrary();
  } catch (e) {
    toast("error", `Couldn't empty the trash: ${errMsg(e)}`);
  }
}

async function reloadCapabilities(
  entry = host.value,
  epoch = sessionEpoch,
  signal?: AbortSignal,
) {
  if (!entry) return;
  const generation = ++capabilityRequestGeneration;
  try {
    const next = await hostCapabilities(entry, signal);
    if (
      generation === capabilityRequestGeneration &&
      isCurrentSession(entry, epoch)
    )
      caps.value = next;
  } catch (error) {
    // A transport blip cannot erase verified capability policy. HTTP auth is
    // different authority and must fail closed until the credential changes.
    if (
      generation === capabilityRequestGeneration &&
      isCurrentSession(entry, epoch) &&
      error instanceof ApiHttpError &&
      (error.status === 401 || error.status === 403)
    )
      caps.value = null;
  }
}

function toggleTarget() {
  const next = isTarget.value ? ORIGIN_HOST_ID : hostId.value;
  setGenerateTargetId(next);
  targetId.value = next;
}

async function onCancel(id: string) {
  const entry = host.value;
  if (!entry || cancellingIds.value.includes(id)) return;
  cancellingIds.value = [...cancellingIds.value, id];
  try {
    await cancelQueueJob(entry, id);
    await reloadQueue();
  } catch (e) {
    toast("error", `Couldn't cancel job: ${errMsg(e)}`);
  } finally {
    cancellingIds.value = cancellingIds.value.filter(
      (pending) => pending !== id,
    );
  }
}

// ── Queue row detail ─────────────────────────────────────────────────────
const inspectedId = ref<string | null>(null);
const inspectError = ref<string | null>(null);
const inspectPreview = ref<QueueJobPreview | null>(null);
const inspectNowMs = ref(Date.now());
let stopInspectPreview: (() => void) | null = null;
let inspectTimer: ReturnType<typeof setInterval> | null = null;

const inspectedEntry = computed(
  () => queue.value.find((entry) => entry.id === inspectedId.value) ?? null,
);

const inspectedModel = computed(() => {
  const entry = inspectedEntry.value;
  if (!entry) return null;
  return queueEntryDetailModel({
    entry,
    hostLabel: host.value?.name || host.value?.url || "this machine",
    modelLabel: modelDisplayNameForId(entry.model, models.value),
    nowMs: inspectNowMs.value,
    plan: queuePlan.value,
    metadata:
      (entry.metadata as QueueDetailMetadata | null | undefined) ?? null,
    mine: false,
    canCancelRunning: caps.value?.queue?.cooperative_cancellation === true,
  });
});

// Elapsed and estimate lines are wall-clock; the 4 s host poll is too coarse.
watch(inspectedId, (id) => {
  inspectError.value = null;
  inspectPreview.value = null;
  stopInspectPreview?.();
  stopInspectPreview = null;
  if (inspectTimer !== null) clearInterval(inspectTimer);
  inspectTimer = null;
  if (id === null) return;
  inspectNowMs.value = Date.now();
  inspectTimer = setInterval(() => (inspectNowMs.value = Date.now()), 1_000);

  const entry = host.value;
  const row = queue.value.find((candidate) => candidate.id === id);
  if (!entry || row?.state !== "running") return;
  stopInspectPreview = watchSelectedQueuePreview(
    hostApiTarget(entry),
    id,
    (preview) => (inspectPreview.value = preview),
    750,
    () => (inspectPreview.value = null),
  );
});

onBeforeUnmount(() => {
  stopInspectPreview?.();
  if (inspectTimer !== null) clearInterval(inspectTimer);
});

async function reuseInspected() {
  const entry = inspectedEntry.value;
  const metadata = entry?.metadata as OutputMetadata | null | undefined;
  if (!entry || !metadata) return;
  const pinned = entry.seed_pinned ?? metadata.seed !== 0;
  setGenerationHandoff({
    metadata: settingsRestoreMetadata(metadata, { seedPinned: pinned }),
    seedPinned: pinned,
    queueSelection: {
      hostId: hostId.value,
      jobId: entry.id,
      running: entry.state === "running",
    },
  });
  inspectedId.value = null;
  await router.push("/create");
}

async function cancelInspected() {
  const entry = inspectedEntry.value;
  if (!entry) return;
  inspectError.value = null;
  const accepted = await requestConfirm({
    title: entry.state === "running" ? "Stop this job?" : "Cancel this job?",
    body:
      entry.state === "running"
        ? "The machine stops at its next safe point and nothing is saved."
        : "It leaves the queue and is not rendered.",
    confirmLabel: entry.state === "running" ? "Stop job" : "Cancel job",
    danger: true,
  });
  if (!accepted) return;
  await onCancel(entry.id);
  inspectedId.value = null;
}

async function onSetLane(id: string, gpu: number | null) {
  const entry = host.value;
  if (!entry) return;
  try {
    if (isOrigin.value) await updateQueueJobTargetGpu(id, gpu);
    else await setQueueJobLane(entry, id, gpu);
    await reloadQueue();
  } catch (e) {
    toast("error", `Couldn't change lane: ${errMsg(e)}`);
  }
}

async function onMove(id: string, position: number) {
  const entry = host.value;
  if (!entry) return;
  try {
    if (isOrigin.value) await reorderQueueJob(id, position);
    else await moveQueueJob(entry, id, position);
    await reloadQueue();
  } catch (e) {
    toast("error", `Couldn't reorder job: ${errMsg(e)}`);
  }
}

async function onTogglePause() {
  const entry = host.value;
  if (!entry) return;
  try {
    if (paused.value) await resumeHostQueue(entry);
    else await pauseHostQueue(entry);
    await poll.refresh();
  } catch (e) {
    toast(
      "error",
      `Couldn't ${paused.value ? "resume" : "pause"} queue: ${errMsg(e)}`,
    );
  }
}

async function onCancelAll() {
  const entry = host.value;
  if (!entry) return;
  const accepted = await requestConfirm({
    title: "Cancel all queued jobs?",
    body: `Running work on ${hostName.value} will continue.`,
    confirmLabel: "Cancel queued jobs",
    danger: true,
  });
  if (!accepted) return;
  try {
    await cancelAllHostQueue(entry);
    await reloadQueue();
  } catch (e) {
    toast("error", `Couldn't cancel queued jobs: ${errMsg(e)}`);
  }
}

async function renameMachine() {
  if (!host.value || isOrigin.value) return;
  const next = await requestText({
    title: "Rename machine",
    label: "Machine name",
    initial: hostName.value,
  });
  const name = next?.trim();
  if (!name) return;
  if (updateHost(hostId.value, { name })) hostName.value = name;
}

async function forgetMachine() {
  if (!host.value || isOrigin.value) return;
  const accepted = await requestConfirm({
    title: "Forget this machine?",
    body: `${hostName.value} and its saved API key will be removed from this browser.`,
    confirmLabel: "Forget machine",
    danger: true,
  });
  if (!accepted) return;
  removeHost(hostId.value);
  if (isTarget.value) setGenerateTargetId(ORIGIN_HOST_ID);
  await router.push("/machines");
}

async function disconnectMachine() {
  if (!host.value || isOrigin.value) return;
  setHostConnected(hostId.value, false);
  if (isTarget.value) {
    setGenerateTargetId(ORIGIN_HOST_ID);
    targetId.value = ORIGIN_HOST_ID;
  }
  stopHostSession();
  toast("success", `${hostName.value} disconnected.`);
  await router.push("/machines");
}

function reconnectMachine() {
  if (!host.value || isOrigin.value) return;
  setHostConnected(hostId.value, true);
  toast("success", `${hostName.value} reconnected.`);
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function downloadPct(job: DownloadJobWire): number {
  if (!job.bytes_total) return 0;
  return Math.min(100, (job.bytes_done / job.bytes_total) * 100);
}

let timer: ReturnType<typeof setTimeout> | null = null;
let deviceEventsAbort: AbortController | null = null;
let sessionAbort: AbortController | null = null;
let reloadAllInFlight: Promise<void> | null = null;
let reloadAllPending = false;

function reloadAll(
  entry: NonNullable<typeof host.value>,
  epoch: number,
  signal: AbortSignal,
): Promise<void> {
  if (reloadAllInFlight) {
    reloadAllPending = true;
    return reloadAllInFlight;
  }
  const run = (async () => {
    do {
      reloadAllPending = false;
      await reloadAllOnce(entry, epoch, signal);
    } while (
      reloadAllPending &&
      !signal.aborted &&
      isCurrentSession(entry, epoch)
    );
  })().finally(() => {
    if (reloadAllInFlight === run) reloadAllInFlight = null;
  });
  reloadAllInFlight = run;
  return run;
}

function scheduleReload(
  entry: NonNullable<typeof host.value>,
  epoch: number,
  signal: AbortSignal,
) {
  if (timer || signal.aborted || !isCurrentSession(entry, epoch)) return;
  timer = setTimeout(() => {
    timer = null;
    void reloadAll(entry, epoch, signal).finally(() =>
      scheduleReload(entry, epoch, signal),
    );
  }, 4000);
}

function stopHostSession() {
  sessionEpoch += 1;
  queueRequestGeneration += 1;
  queueLoadMoreGeneration += 1;
  capabilityRequestGeneration += 1;
  modelRequestGeneration += 1;
  downloadRequestGeneration += 1;
  if (timer) clearTimeout(timer);
  timer = null;
  deviceEventsAbort?.abort();
  deviceEventsAbort = null;
  sessionAbort?.abort();
  sessionAbort = null;
  reloadAllPending = false;
  reloadAllInFlight = null;
}

function startHostSession() {
  stopHostSession();
  const entry = liveHost.value;
  const epoch = sessionEpoch;
  caps.value = null;
  queue.value = [];
  queuePlan.value = null;
  queueTail.value = [];
  queuePageLimit.value = null;
  queueNextCursor.value = null;
  queueContinued.value = false;
  loadMoreQueueError.value = "";
  models.value = [];
  downloads.value = [];
  hostName.value = host.value?.name ?? "";
  if (!entry) return;
  sessionAbort = new AbortController();
  const signal = sessionAbort.signal;
  void reloadAll(entry, epoch, signal).finally(() =>
    scheduleReload(entry, epoch, signal),
  );
  deviceEventsAbort = new AbortController();
  subscribeToDeviceSnapshots(
    { baseUrl: entry.url, apiKey: entry.apiKey ?? null },
    deviceEventsAbort.signal,
    () => {
      void poll.refresh();
      // The queued follow-up is single-flight, but the now-stale queue answer
      // must not become visible while the current wave settles.
      queueRequestGeneration += 1;
      void reloadAll(entry, epoch, signal);
    },
  );
}

function onHostsChanged() {
  registryRevision.value += 1;
}

function onHostStorage(event: StorageEvent) {
  if (event.key === HOSTS_STORAGE_KEY) onHostsChanged();
}

onMounted(() => {
  window.addEventListener(HOSTS_CHANGED_EVENT, onHostsChanged);
  window.addEventListener("storage", onHostStorage);
});

watch(
  () => {
    const entry = host.value;
    return entry
      ? `${entry.id}\u0000${entry.url}\u0000${entry.apiKey ?? ""}\u0000${entry.connected !== false}`
      : `missing:${hostId.value}`;
  },
  () => startHostSession(),
  { immediate: true },
);

onBeforeUnmount(() => {
  window.removeEventListener(HOSTS_CHANGED_EVENT, onHostsChanged);
  window.removeEventListener("storage", onHostStorage);
  stopHostSession();
});
</script>

<template>
  <!-- w-full: see MachinesPage — an mx-auto child of the column-flex app
       frame shrinks to content width without it. -->
  <div class="mx-auto w-full max-w-[1800px] px-4 pb-40 pt-6 sm:px-6 lg:px-10">
    <router-link to="/machines" class="md-back" data-test="detail-back">
      <Icon name="chevron-left" :size="16" /> Machines
    </router-link>

    <div v-if="!host" class="mt-10">
      <CardSurface data-test="detail-not-found">
        <p class="md-missing">That machine isn't in your list anymore.</p>
      </CardSurface>
    </div>

    <template v-else>
      <div class="md-head">
        <StatusDot :state="dotState" />
        <span class="md-name" data-test="machine-detail-title">
          {{ hostName }}
        </span>
        <span class="md-addr">{{ address }}</span>
        <div class="md-spacer" />
        <button
          v-if="!disconnected"
          type="button"
          class="md-target"
          data-test="detail-target"
          :data-on="isTarget ? 'true' : undefined"
          @click="toggleTarget"
        >
          {{ isTarget ? "Generation target" : "Set as generation target" }}
        </button>
        <button
          v-if="!isOrigin"
          type="button"
          class="md-action"
          data-test="detail-rename"
          @click="renameMachine"
        >
          Rename
        </button>
        <button
          v-if="!isOrigin && !disconnected"
          type="button"
          class="md-action"
          data-test="detail-disconnect"
          @click="disconnectMachine"
        >
          Disconnect
        </button>
        <button
          v-if="!isOrigin && disconnected"
          type="button"
          class="md-action"
          data-test="detail-reconnect"
          @click="reconnectMachine"
        >
          Reconnect
        </button>
        <button
          v-if="!isOrigin"
          type="button"
          class="md-action md-action--danger"
          data-test="detail-forget"
          @click="forgetMachine"
        >
          Forget
        </button>
      </div>

      <div
        v-if="reconnecting"
        class="md-offline"
        data-test="detail-reconnecting"
      >
        <span
          >This machine is reconnecting. Showing the last known values.</span
        >
        <button type="button" class="md-retry" @click="poll?.refresh()">
          Retry
        </button>
      </div>

      <div
        class="md-grid"
        :data-dimmed="reconnecting ? 'true' : undefined"
        :data-has-library="trashCapable ? 'true' : undefined"
      >
        <CardSurface class="md-telemetry">
          <div class="md-label">Telemetry</div>
          <div class="md-gpu" data-test="telemetry-gpu">
            {{ telemetry.gpuLine }}
          </div>

          <div class="md-metric">
            <span>GPU load</span>
            <span class="md-accent" data-test="telemetry-load">
              {{ telemetry.loadLabel }}
            </span>
          </div>
          <ProgressBar
            :value="telemetry.loadPct ?? 0"
            tone="accent"
            label="GPU load"
          />

          <div class="md-metric md-metric--mt">
            <span>
              {{ telemetry.unifiedMemory ? "Unified memory" : "Memory" }}
              {{ telemetry.memLabel }}
            </span>
            <span class="md-halide" data-test="telemetry-mem">
              {{
                telemetry.memPct != null
                  ? `${Math.round(telemetry.memPct)}%`
                  : "—"
              }}
            </span>
          </div>
          <ProgressBar
            :value="telemetry.memPct ?? 0"
            tone="info"
            label="Memory"
          />

          <!-- On unified-memory hosts this would repeat the Memory row. -->
          <div v-if="!telemetry.unifiedMemory" class="md-metric md-metric--mt">
            <span>System RAM {{ telemetry.ramLabel }}</span>
            <span class="md-halide" data-test="telemetry-ram">
              {{
                telemetry.ramPct != null
                  ? `${Math.round(telemetry.ramPct)}%`
                  : "—"
              }}
            </span>
          </div>
          <ProgressBar
            v-if="!telemetry.unifiedMemory"
            :value="telemetry.ramPct ?? 0"
            :tone="ramTone"
            label="System RAM"
            :data-pressure="hostMemoryPressure ?? undefined"
          />

          <div class="md-tiles">
            <div class="md-tile">
              <div class="md-tile__k">CPU</div>
              <div class="md-tile__v" data-test="telemetry-cpu">
                {{ telemetry.cpuLabel }}
              </div>
            </div>
            <div class="md-tile">
              <div class="md-tile__k">Queue</div>
              <div class="md-tile__v" data-test="telemetry-queue">
                {{ telemetry.queue }}
              </div>
            </div>
            <div class="md-tile">
              <div class="md-tile__k">Uptime</div>
              <div class="md-tile__v" data-test="telemetry-uptime">
                {{ telemetry.uptime }}
              </div>
            </div>
          </div>

          <div
            v-if="telemetry.storageLabel"
            class="md-storage"
            data-test="telemetry-storage"
          >
            <Icon name="download" :size="13" /> {{ telemetry.storageLabel }}
          </div>
        </CardSurface>

        <CardSurface class="md-models">
          <div class="md-label">Installed models</div>
          <p
            v-if="installed.length === 0"
            class="md-models__empty"
            data-test="models-empty"
          >
            No models installed.
          </p>
          <div v-else class="md-models__list">
            <div
              v-for="model in installed"
              :key="model.name"
              class="md-models__row"
              data-test="model-row"
            >
              <span class="md-models__name">{{ modelDisplayName(model) }}</span>
              <span
                v-if="model.is_loaded"
                class="md-models__loaded"
                data-test="model-loaded"
              >
                loaded
              </span>
            </div>
          </div>
        </CardSurface>

        <CardSurface
          v-if="trashCapable"
          class="md-library"
          data-test="library-card"
        >
          <div class="md-label">Library</div>
          <div class="md-library__row">
            <label class="md-library__k" for="library-retention"
              >Trash retention</label
            >
            <select
              id="library-retention"
              class="md-library__select"
              data-test="library-retention"
              :value="String(trashRetentionDays ?? '')"
              :disabled="savingRetention || !online"
              @change="
                onRetentionChange(($event.target as HTMLSelectElement).value)
              "
            >
              <option
                v-for="choice in retentionChoices"
                :key="choice.value"
                :value="String(choice.value)"
              >
                {{ choice.label }}
              </option>
            </select>
          </div>
          <p class="md-library__help">
            Prints moved to the trash are deleted forever after this long.
            Forever keeps them until you empty the trash.
          </p>
          <div class="md-library__row">
            <span class="md-library__k">Prints in trash</span>
            <span class="md-library__v" data-test="library-trash-count">{{
              trashCount ?? "—"
            }}</span>
            <button
              type="button"
              class="md-library__empty"
              :disabled="!trashCount || !online"
              data-test="library-empty-trash"
              @click="onEmptyTrash"
            >
              <Icon name="trash" :size="13" /> Empty trash
            </button>
          </div>
        </CardSurface>
      </div>

      <div class="md-queue" :data-dimmed="reconnecting ? 'true' : undefined">
        <CardSurface class="mb-4">
          <DevicePanel
            :devices="poll?.devices.value ?? []"
            :plan="queuePlan"
            :mutable="
              caps?.devices?.lifecycle === true &&
              caps?.dispatch?.v2_authoritative === true
            "
            :restart-enable="caps?.devices?.restart_enable === true"
            show-controls
            :busy-device-ids="[...mutatingDeviceIds]"
            @unpin="onUnpinWork"
            @toggle="onToggleDevice"
          />
        </CardSurface>
        <QueueCard
          :entries="queue"
          :plan="queuePlan"
          :models="models"
          :gpu-ordinals="gpuOrdinals"
          :can-reorder="canReorder"
          :can-pause="caps?.queue?.can_pause === true"
          :can-cancel-all="caps?.queue?.can_cancel_all === true"
          :can-cancel-running="caps?.queue?.cooperative_cancellation === true"
          :cancelling-ids="cancellingIds"
          :paused="paused"
          :dimmed="reconnecting"
          @cancel="onCancel"
          @inspect="inspectedId = $event"
          @set-lane="onSetLane"
          @move="onMove"
          @toggle-pause="onTogglePause"
          @cancel-all="onCancelAll"
        />
        <button
          v-if="queueNextCursor"
          type="button"
          class="md-library__empty mt-2"
          data-test="queue-load-more"
          :disabled="loadingMoreQueue || !online"
          @click="loadMoreQueue"
        >
          {{ loadingMoreQueue ? "Loading…" : "Load more jobs" }}
        </button>
        <p v-if="loadMoreQueueError" class="error-text" role="alert">
          {{ loadMoreQueueError }}
        </p>
      </div>

      <CardSurface
        class="md-downloads"
        :data-dimmed="reconnecting ? 'true' : undefined"
      >
        <div class="md-label">Downloads</div>
        <p
          v-if="downloads.length === 0"
          class="md-dl__empty"
          data-test="downloads-empty"
        >
          No active downloads.
        </p>
        <div v-else class="md-dl__list">
          <div
            v-for="job in downloads"
            :key="job.id"
            class="md-dl__row"
            data-test="download-row"
          >
            <div class="md-dl__head">
              <span class="md-dl__name">{{ modelLabel(job.model) }}</span>
              <BadgePill tone="info" outline>{{ job.status }}</BadgePill>
            </div>
            <ProgressBar
              :value="downloadPct(job)"
              tone="info"
              :label="`${modelLabel(job.model)} download`"
            />
          </div>
        </div>
      </CardSurface>

      <!-- Specialized capability detail reads below the live instruments. -->
      <MinimaxH3InventoryPanel :hosts="h3Host" heading="H3 on this machine" />
    </template>
    <aside
      v-if="inspectedModel"
      class="md-queue-detail"
      role="dialog"
      aria-modal="false"
      :aria-label="`Queued job — ${inspectedModel.modelLabel}`"
      data-test="queue-entry-drawer"
    >
      <QueueEntryDetail
        :model="inspectedModel"
        :preview="inspectPreview"
        :cancelling="
          inspectedEntry ? cancellingIds.includes(inspectedEntry.id) : false
        "
        :error="inspectError"
        confirm="delegate"
        @close="inspectedId = null"
        @reuse="reuseInspected"
        @cancel="cancelInspected"
      />
    </aside>
  </div>
</template>

<style scoped>
.md-back {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  color: var(--safelight);
  font-size: 13px;
  font-weight: 600;
}

.md-missing,
.md-models__empty,
.md-dl__empty {
  margin: 0;
  font-size: 12.5px;
  color: var(--ink-3);
}

.md-head {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 18px 0;
}

.md-name {
  font-family: var(--f-display);
  font-size: 19px;
  font-weight: 700;
  color: var(--rebate);
}

.md-addr {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}

.md-spacer {
  flex: 1;
}

.md-target {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 8px 13px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.md-action {
  min-height: 36px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 0 11px;
  border-radius: 8px;
  font-size: 12px;
  cursor: pointer;
}

.md-action--danger {
  color: var(--stop);
}

.md-target[data-on="true"] {
  border-color: var(--sel-border);
  background: var(--sel-bg);
  color: var(--sel-ink);
  box-shadow: var(--sel-ring);
}

.md-offline {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
  padding: 10px 14px;
  border: 1px solid var(--ce);
  border-radius: 10px;
  background: color-mix(in srgb, var(--stop) 10%, transparent);
  font-size: 12.5px;
  color: var(--ink-2);
}

.md-retry {
  margin-left: auto;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 5px 12px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.md-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  grid-template-areas:
    "telemetry library"
    "models models";
  gap: 16px;
  align-items: flex-start;
}

.md-grid:not([data-has-library="true"]) {
  grid-template-areas:
    "telemetry telemetry"
    "models models";
}

.md-grid[data-dimmed="true"],
.md-queue[data-dimmed="true"],
.md-downloads[data-dimmed="true"] {
  opacity: 0.6;
}

.md-telemetry {
  grid-area: telemetry;
  min-width: 280px;
}

.md-models {
  grid-area: models;
  min-width: 0;
}

.md-library {
  grid-area: library;
  width: 300px;
}

@media (max-width: 820px) {
  .md-grid {
    grid-template-columns: minmax(0, 1fr);
    grid-template-areas:
      "telemetry"
      "library"
      "models";
  }

  .md-grid:not([data-has-library="true"]) {
    grid-template-areas:
      "telemetry"
      "models";
  }

  .md-library {
    width: 100%;
  }
}

.md-library__row {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 32px;
}
.md-library__row + .md-library__row {
  margin-top: 8px;
}
.md-library__k {
  flex: 1;
  font-size: 12.5px;
  color: var(--ink-2);
}
.md-library__v {
  font-family: var(--f-mono);
  font-size: 12px;
  color: var(--rebate);
}
.md-library__select {
  height: 30px;
  padding: 0 8px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
}
.md-library__help {
  margin: 6px 0 10px;
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--ink-3);
}
.md-library__empty {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 30px;
  padding: 0 10px;
  border: 1px solid color-mix(in srgb, var(--stop) 50%, transparent);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--stop);
  font-family: var(--f-body);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.md-library__empty:disabled {
  opacity: 0.5;
  cursor: default;
}
.md-library__empty:focus-visible,
.md-library__select:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.md-label {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin-bottom: 12px;
}

.md-gpu {
  font-family: var(--f-mono);
  font-size: 11.5px;
  color: var(--ink-3);
  margin-bottom: 16px;
}

.md-metric {
  display: flex;
  justify-content: space-between;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-2);
  margin-bottom: 6px;
}

.md-metric--mt {
  margin-top: 16px;
}

.md-accent {
  color: var(--safelight);
}

.md-halide {
  color: var(--halide);
}

.md-tiles {
  display: flex;
  gap: 10px;
  margin-top: 18px;
}

.md-tile {
  flex: 1;
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: 9px;
  padding: 11px;
}

.md-tile__k {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.md-tile__v {
  font-family: var(--f-mono);
  font-size: 16px;
  margin-top: 3px;
  color: var(--rebate);
}

.md-storage {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 16px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}

.md-models__list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 2px;
  max-height: clamp(180px, 32vh, 320px);
  overflow-y: auto;
  overscroll-behavior: contain;
  padding-right: 4px;
}

.md-models__row {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 8px 6px;
  border-radius: 7px;
}

.md-models__row:hover {
  background: color-mix(in srgb, var(--rebate) 5%, transparent);
}

.md-models__name {
  flex: 1;
  min-width: 0;
  font-family: var(--f-mono);
  font-size: 12.5px;
  color: var(--rebate);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.md-models__loaded {
  font-family: var(--f-mono);
  font-size: 9px;
  color: var(--safelight);
}

.md-queue {
  margin-top: 16px;
}

.md-downloads {
  margin-top: 16px;
}

.md-queue-detail {
  position: fixed;
  inset-block: 0;
  inset-inline-end: 0;
  z-index: 40;
  display: flex;
  width: min(384px, 100%);
  flex-direction: column;
  border-inline-start: 1px solid var(--edge);
  background: var(--bench);
}

.md-dl__list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.md-dl__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.md-dl__name {
  font-family: var(--f-mono);
  font-size: 12.5px;
  color: var(--rebate);
}
</style>
