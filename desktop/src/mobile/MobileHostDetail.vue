<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { parseDeviceListResponse, setDeviceEnabled, type DeviceInfo } from "@studio/api/devices";
import {
  cancelQueueJob,
  listQueue,
  mergeQueueEntries,
  moveQueueJobToBack,
  queuePageRequestForCapacity,
  setQueuePaused as setQueueDispatchPaused,
  setQueueDevicePin,
  type QueuePlan,
} from "@studio/api/queuePlan";
import { watchSelectedQueuePreview, type QueueJobProgress } from "@studio/api/generationSelection";
import DevicePanel from "@studio/components/DevicePanel.vue";
import QueuePlanWorkList from "@studio/components/QueuePlanWorkList.vue";
import QueueEntryDetail from "@studio/components/QueueEntryDetail.vue";
import SwipeActionRow from "@studio/components/SwipeActionRow.vue";
import MobileLibrarySheet from "./MobileLibrarySheet.vue";
import { queueEntryDetailModel, type QueueDetailMetadata } from "@studio/lib/queueEntryDetail";
import type { SwipeRowAction } from "@studio/lib/swipeAction";
import MinimaxH3InventoryPanel from "@studio/components/MinimaxH3InventoryPanel.vue";
import { canMutateDevice } from "@studio/lib/deviceLifecycle";
import { queueWaitCode, resolveQueueWait } from "@studio/lib/queuePosition";
import { queuePlanOnlyWork } from "@studio/lib/queuePlanPresentation";
import { hostMemoryLevel, hostMemoryScheduleLabel } from "@studio/lib/hostMemory";
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
import { mobileHostHealthLabel, mobileHostTarget, type MobileHost } from "./hosts";
import { emptyTrash as emptyHostTrash, listTrash } from "@studio/api/galleryOrganization";
import { RETENTION_OPTIONS, retentionLabel } from "@studio/lib/libraryOrganization";
import {
  TRASH_RETENTION_CONFIG_KEY,
  fetchHostConfigKey,
  hostConfigEditable,
  hostConfigLocked,
  retentionDaysFromConfigValue,
  setHostConfigKey,
  type HostConfigEntry,
} from "./hostConfig";
import { galleryCapabilitiesOf } from "./libraryOrganization";

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
const queueTail = ref<QueueEntry[]>([]);
const queuePageLimit = ref<number | null>(null);
const queueNextCursor = ref<string | null>(null);
const queueContinued = ref(false);
const loadingMoreQueue = ref(false);
const loadMoreQueueError = ref("");
const downloads = ref<DownloadsState>(emptyDownloadsState());
const loading = ref(false);
const error = ref("");
const renaming = ref(false);
const renameValue = ref("");
const forgetPending = ref(false);
const unloading = ref<Set<string>>(new Set());
const cancellingQueueIds = ref(new Set<string>());
const queueControlBusy = ref(false);
// ── Library card (per-host trash retention, #4 iPhone V3) ───────────────────
const retentionEntry = ref<HostConfigEntry | null>(null);
const retentionValue = ref<number | null>(null);
const retentionProbeFailed = ref(false);
const retentionError = ref("");
const retentionSaving = ref(false);
const trashCount = ref<number | null>(null);
const emptyTrashArmed = ref(false);
const emptyingTrash = ref(false);
let loadEpoch = 0;
let queueRequestGeneration = 0;
let queueLoadMoreGeneration = 0;
let deviceRequestGeneration = 0;
let statusRequestGeneration = 0;
let resourceAbort: AbortController | null = null;
let downloadsAbort: AbortController | null = null;
let deviceEventsAbort: AbortController | null = null;
let livePollTimer: ReturnType<typeof setTimeout> | null = null;
let queueRefreshPromise: Promise<void> | null = null;
let queueRefreshQueued = false;
let deviceRefreshPromise: Promise<void> | null = null;
let deviceRefreshQueued = false;

const target = computed(() => mobileHostTarget(props.host));
const inFlightDownloads = computed(() => [
  ...downloads.value.activeJobs,
  ...downloads.value.queued,
]);
const loadedModels = computed(() => status.value?.models_loaded ?? []);
const uptime = computed(() => status.value?.uptime_secs ?? null);
const durableBacklog = computed(() => {
  const depth = status.value?.queue_depth;
  return typeof depth === "number" && Number.isSafeInteger(depth) && depth >= 0 ? depth : null;
});
const runtimeWindow = computed(() => {
  const capacity = status.value?.queue_capacity;
  return typeof capacity === "number" && Number.isSafeInteger(capacity) && capacity > 0
    ? capacity
    : null;
});
const queueEntryIds = computed(() => queue.value.map((entry) => entry.id));
const planOnlyWork = computed(() => queuePlanOnlyWork(queuePlan.value, queueEntryIds.value));
const queueSummary = computed(() => {
  const visibleWork = queue.value.length + planOnlyWork.value.length;
  if (durableBacklog.value !== null) return `${Math.max(durableBacklog.value, visibleWork)} total`;
  return `${visibleWork} loaded`;
});
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
/** RAM pressure from the scheduler's own ledger, not used/total: a committed
 *  reservation that has not allocated yet still parks the queue and the OS
 *  cannot see it. Absent on older servers, which keeps the plain meter. */
const hostMemoryPressure = computed(() => hostMemoryLevel(queuePlan.value?.host_memory));
const hostMemoryLabel = computed(() => {
  const memory = queuePlan.value?.host_memory;
  return memory ? hostMemoryScheduleLabel(memory, formatGB) : null;
});
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
  queueRefreshPromise = null;
  queueRefreshQueued = false;
  deviceRefreshPromise = null;
  deviceRefreshQueued = false;
}

async function refreshQueue(epoch = loadEpoch): Promise<void> {
  const generation = ++queueRequestGeneration;
  const requestTarget = target.value;
  try {
    const listing = await listQueue(
      requestTarget,
      queuePageRequestForCapacity(status.value?.queue_capacity) ?? null,
    );
    if (
      epoch === loadEpoch &&
      generation === queueRequestGeneration &&
      requestTarget.baseUrl === target.value.baseUrl &&
      requestTarget.apiKey === target.value.apiKey
    ) {
      const head = mergeQueueEntries(
        listing.entries,
        listing.live_only_entries ?? [],
      ) as QueueEntry[];
      queue.value = head;
      queuePlan.value = listing.plan;
      queuePageLimit.value = listing.page?.limit ?? null;
      queueNextCursor.value = listing.page?.next_cursor ?? null;
      queueTail.value = [];
      queueContinued.value = false;
      loadingMoreQueue.value = false;
      queueLoadMoreGeneration += 1;
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

function requestQueueRefresh(epoch = loadEpoch): Promise<void> {
  if (queueRefreshPromise) {
    queueRefreshQueued = true;
    queueRequestGeneration += 1;
    return queueRefreshPromise;
  }
  const refresh = (async () => {
    do {
      queueRefreshQueued = false;
      await refreshQueue(epoch);
    } while (queueRefreshQueued && epoch === loadEpoch);
  })();
  queueRefreshPromise = refresh;
  return refresh.finally(() => {
    if (queueRefreshPromise === refresh) queueRefreshPromise = null;
  });
}

async function loadMoreQueue(): Promise<void> {
  const cursor = queueNextCursor.value;
  const limit = queuePageLimit.value;
  if (!cursor || !limit || loadingMoreQueue.value) return;
  loadingMoreQueue.value = true;
  loadMoreQueueError.value = "";
  const epoch = loadEpoch;
  const requestTarget = target.value;
  const generation = ++queueLoadMoreGeneration;
  try {
    const listing = await listQueue(requestTarget, { limit, cursor });
    if (
      generation !== queueLoadMoreGeneration ||
      epoch !== loadEpoch ||
      queueNextCursor.value !== cursor ||
      requestTarget.baseUrl !== target.value.baseUrl ||
      requestTarget.apiKey !== target.value.apiKey
    )
      return;
    const seenTail = new Set(queueTail.value.map(({ id }) => id));
    queueTail.value = [
      ...queueTail.value,
      ...(listing.entries as QueueEntry[]).filter(({ id }) => !seenTail.has(id)),
    ];
    const seen = new Set<string>();
    queue.value = [...queue.value, ...queueTail.value, ...(listing.live_only_entries ?? [])].filter(
      ({ id }) => {
        if (seen.has(id)) return false;
        seen.add(id);
        return true;
      },
    ) as QueueEntry[];
    queueNextCursor.value = listing.page?.next_cursor ?? null;
    queueContinued.value = listing.page !== undefined;
    queuePlan.value = listing.plan ?? queuePlan.value;
  } catch (error) {
    if (
      generation === queueLoadMoreGeneration &&
      epoch === loadEpoch &&
      requestTarget.baseUrl === target.value.baseUrl &&
      requestTarget.apiKey === target.value.apiKey
    )
      loadMoreQueueError.value = describeTransportError(error, props.host.name);
  } finally {
    if (
      generation === queueLoadMoreGeneration &&
      epoch === loadEpoch &&
      requestTarget.baseUrl === target.value.baseUrl &&
      requestTarget.apiKey === target.value.apiKey
    )
      loadingMoreQueue.value = false;
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

async function refreshStatusSafely(epoch: number): Promise<void> {
  if (queueControlBusy.value) return;
  const generation = ++statusRequestGeneration;
  const requestTarget = target.value;
  try {
    const nextStatus = await apiJsonTo<ServerStatus>(requestTarget, "/api/status");
    if (
      generation !== statusRequestGeneration ||
      epoch !== loadEpoch ||
      queueControlBusy.value ||
      requestTarget.baseUrl !== target.value.baseUrl ||
      requestTarget.apiKey !== target.value.apiKey
    )
      return;
    const expectedInstanceId = props.host.instanceId?.trim() || status.value?.instance_id?.trim();
    const reportedInstanceId = nextStatus.instance_id?.trim();
    if (expectedInstanceId && reportedInstanceId && expectedInstanceId !== reportedInstanceId) {
      emit("status", { id: props.host.id, status: nextStatus });
      error.value = "This address now reports a different Mold server identity.";
      return;
    }
    status.value = nextStatus;
    emit("status", { id: props.host.id, status: nextStatus });
  } catch {
    // Queue and device polling remain useful during a transient status failure.
    // The next five-second pass retries without clearing the last good state.
  }
}

function scheduleLivePoll(epoch: number): void {
  if (epoch !== loadEpoch) return;
  livePollTimer = setTimeout(async () => {
    livePollTimer = null;
    // Queue authority has its own generation fence and must not hold device
    // freshness hostage if an older endpoint never settles.
    await Promise.all([
      refreshStatusSafely(epoch),
      requestQueueRefresh(epoch),
      refreshDevicesSafely(epoch),
    ]);
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
    void requestQueueRefresh(epoch);
    void refreshDevicesSafely(epoch);
  });

  void requestQueueRefresh(epoch);
  scheduleLivePoll(epoch);
}

/** The host's own trash capability; the Library card renders only when the
 * exact Keychain-authenticated host advertises `gallery.trash.enabled`. */
const hostTrashCapability = computed(
  () => galleryCapabilitiesOf(deviceCapabilities.value)?.trash ?? null,
);
const libraryCardVisible = computed(() => hostTrashCapability.value?.enabled === true);
/** An env-pinned key is read-only on every client (`mold config` semantics). */
const retentionLocked = computed(() => hostConfigLocked(retentionEntry.value));
/** Unknown authority (probe failed / unanswered) is read-only too: enabling
 * the selector on a failed probe could expose an env-pinned key to edits. */
const retentionEditable = computed(() => hostConfigEditable(retentionEntry.value));
const retentionChoices = computed(() => {
  const days = retentionValue.value;
  return days !== null && !RETENTION_OPTIONS.includes(days)
    ? [...RETENTION_OPTIONS, days].sort((a, b) => (a === 0 ? 1 : b === 0 ? -1 : a - b))
    : RETENTION_OPTIONS;
});

async function refreshLibraryCard(epoch = loadEpoch): Promise<void> {
  if (!libraryCardVisible.value) return;
  const requestTarget = target.value;
  const [entry, trashed] = await Promise.allSettled([
    fetchHostConfigKey(requestTarget, TRASH_RETENTION_CONFIG_KEY),
    listTrash(requestTarget),
  ]);
  if (epoch !== loadEpoch) return;
  if (entry.status === "fulfilled") {
    retentionProbeFailed.value = false;
    retentionEntry.value = entry.value;
    retentionValue.value = retentionDaysFromConfigValue(entry.value.value);
  } else {
    // No entry = unknown authority: the selector stays disabled (see
    // `retentionEditable`) and the failure line offers an explicit Retry.
    retentionProbeFailed.value = true;
    retentionEntry.value = null;
    retentionValue.value = hostTrashCapability.value?.retention_days ?? null;
  }
  trashCount.value = trashed.status === "fulfilled" ? trashed.value.length : null;
}

async function saveRetention(rawDays: string): Promise<void> {
  const days = Number(rawDays);
  if (!Number.isFinite(days) || days < 0 || retentionSaving.value) return;
  retentionSaving.value = true;
  retentionError.value = "";
  try {
    const entry = await setHostConfigKey(target.value, TRASH_RETENTION_CONFIG_KEY, days);
    retentionEntry.value = entry;
    retentionValue.value = retentionDaysFromConfigValue(entry.value);
  } catch (caught) {
    retentionError.value = describeTransportError(caught, props.host.name);
  } finally {
    retentionSaving.value = false;
  }
}

/** Two-step inline confirm: first tap arms, the second purges the trash. */
async function emptyTrashNow(): Promise<void> {
  if (emptyingTrash.value) return;
  if (!emptyTrashArmed.value) {
    emptyTrashArmed.value = true;
    return;
  }
  emptyTrashArmed.value = false;
  emptyingTrash.value = true;
  retentionError.value = "";
  try {
    await emptyHostTrash(target.value);
    trashCount.value = 0;
  } catch (caught) {
    retentionError.value = describeTransportError(caught, props.host.name);
  } finally {
    emptyingTrash.value = false;
  }
}

async function loadHost(): Promise<void> {
  const epoch = ++loadEpoch;
  statusRequestGeneration += 1;
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
  queueTail.value = [];
  queuePageLimit.value = null;
  queueNextCursor.value = null;
  queueContinued.value = false;
  loadingMoreQueue.value = false;
  loadMoreQueueError.value = "";
  downloads.value = emptyDownloadsState();
  renameValue.value = props.host.name;
  renaming.value = false;
  forgetPending.value = false;
  cancellingQueueIds.value = new Set();
  queueControlBusy.value = false;
  retentionEntry.value = null;
  retentionValue.value = null;
  retentionProbeFailed.value = false;
  retentionError.value = "";
  trashCount.value = null;
  emptyTrashArmed.value = false;
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
    const expectedInstanceId = props.host.instanceId?.trim();
    const reportedInstanceId = nextStatus.instance_id?.trim();
    if (expectedInstanceId && reportedInstanceId && expectedInstanceId !== reportedInstanceId) {
      emit("status", { id: props.host.id, status: nextStatus });
      throw new Error("This address now reports a different Mold server identity.");
    }
    status.value = nextStatus;
    installed.value = models.filter((model) => model.downloaded);
    emit("status", { id: props.host.id, status: nextStatus });
    await refreshDevicesSafely(epoch);
    if (epoch !== loadEpoch) return;
    void refreshLibraryCard(epoch);
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
    await requestQueueRefresh();
  } catch (caught) {
    error.value = describeTransportError(caught, props.host.name);
  }
}

function queueEntryCancellable(entry: QueueEntry): boolean {
  return (
    entry.state === "queued" ||
    entry.state === "paused" ||
    entry.state === "held" ||
    (entry.state === "running" &&
      deviceCapabilities.value?.queue?.cooperative_cancellation === true)
  );
}

/**
 * Cancel with no confirmation of its own. The caller owns step two — the
 * swipe row's reveal, or the detail sheet's armed button — so a destructive
 * action is still never one tap.
 */
async function cancelQueuedJob(entry: QueueEntry): Promise<void> {
  if (!queueEntryCancellable(entry) || cancellingQueueIds.value.has(entry.id)) return;

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
      await requestQueueRefresh(epoch);
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

// ── Queue row swipe actions and detail sheet ─────────────────────────────
const inspectedQueueId = ref<string | null>(null);
const queueRowError = ref<string | null>(null);
const reorderingQueueIds = ref(new Set<string>());
const queuePreview = ref<QueueJobProgress | null>(null);
const queueDetailNowMs = ref(Date.now());
let stopQueuePreview: (() => void) | null = null;
let queueDetailTimer: ReturnType<typeof setInterval> | null = null;

const canReorderQueue = computed(() => deviceCapabilities.value?.queue?.can_reorder === true);
const canPauseQueue = computed(() => deviceCapabilities.value?.queue?.can_pause === true);
const dispatchPaused = computed(() => status.value?.queue_paused === true);
const restartPaused = computed(() => queue.value.some((entry) => entry.state === "paused"));
const resumeNeeded = computed(() => dispatchPaused.value || restartPaused.value);

const inspectedQueueEntry = computed(
  () => queue.value.find((entry) => entry.id === inspectedQueueId.value) ?? null,
);

const inspectedQueueModel = computed(() => {
  const entry = inspectedQueueEntry.value;
  if (!entry) return null;
  return queueEntryDetailModel({
    entry,
    hostLabel: props.host.name,
    modelLabel: modelLabel(entry.model),
    nowMs: queueDetailNowMs.value,
    plan: queuePlan.value,
    metadata: (entry.metadata as QueueDetailMetadata | null | undefined) ?? null,
    mine: false,
    canCancelRunning: deviceCapabilities.value?.queue?.cooperative_cancellation === true,
  });
});

/** Trailing actions for one row. Cancel is the only destructive one and the
 *  only one a full swipe commits; a reorder-capable host also offers the
 *  non-destructive Move to back. */
function queueRowActions(entry: QueueEntry): SwipeRowAction[] {
  const actions: SwipeRowAction[] = [];
  if (canPauseQueue.value && ["queued", "paused", "held"].includes(entry.state)) {
    actions.push({
      id: resumeNeeded.value ? "queue-resume" : "queue-pause",
      label: resumeNeeded.value ? "Resume" : "Pause",
    });
  }
  if (canReorderQueue.value && entry.state === "queued") {
    actions.push({ id: "back", label: "To back" });
  }
  if (queueEntryCancellable(entry)) {
    actions.push({
      id: "cancel",
      label: "Cancel",
      tone: "danger",
      commitOnFullSwipe: true,
    });
  }
  return actions;
}

function queueRowBusy(entry: QueueEntry): boolean {
  return (
    queueControlBusy.value ||
    cancellingQueueIds.value.has(entry.id) ||
    reorderingQueueIds.value.has(entry.id)
  );
}

async function setHostQueuePaused(paused: boolean): Promise<void> {
  if (!canPauseQueue.value || queueControlBusy.value) return;
  const epoch = loadEpoch;
  const requestTarget = target.value;
  statusRequestGeneration += 1;
  queueControlBusy.value = true;
  queueRowError.value = null;
  try {
    const liveStatus = await apiJsonTo<ServerStatus>(requestTarget, "/api/status");
    const expectedInstanceId = props.host.instanceId?.trim() || status.value?.instance_id?.trim();
    if (expectedInstanceId && liveStatus.instance_id?.trim() !== expectedInstanceId) {
      throw new Error("This address now reports a different Mold server identity.");
    }
    await setQueueDispatchPaused(requestTarget, paused);
    if (epoch !== loadEpoch || requestTarget.baseUrl !== target.value.baseUrl) return;
    const nextStatus = { ...liveStatus, queue_paused: paused };
    status.value = nextStatus;
    emit("status", { id: props.host.id, status: nextStatus });
    await requestQueueRefresh(epoch);
  } catch (caught) {
    if (epoch === loadEpoch) {
      queueRowError.value = describeTransportError(caught, props.host.name);
    }
  } finally {
    if (epoch === loadEpoch) queueControlBusy.value = false;
  }
}

async function onQueueRowAction(entry: QueueEntry, action: string) {
  queueRowError.value = null;
  if (action === "queue-pause" || action === "queue-resume") {
    await setHostQueuePaused(action === "queue-pause");
    return;
  }
  if (action === "cancel") {
    await cancelQueuedJob(entry);
    if (inspectedQueueId.value === entry.id) inspectedQueueId.value = null;
    return;
  }
  if (action !== "back") return;
  const epoch = loadEpoch;
  const requestTarget = target.value;
  reorderingQueueIds.value = new Set(reorderingQueueIds.value).add(entry.id);
  try {
    await moveQueueJobToBack(requestTarget, entry.id);
    if (
      epoch === loadEpoch &&
      requestTarget.baseUrl === target.value.baseUrl &&
      requestTarget.apiKey === target.value.apiKey
    ) {
      await requestQueueRefresh(epoch);
    }
  } catch (caught) {
    if (epoch === loadEpoch) {
      queueRowError.value = describeTransportError(caught, props.host.name);
    }
  } finally {
    if (epoch === loadEpoch) {
      const next = new Set(reorderingQueueIds.value);
      next.delete(entry.id);
      reorderingQueueIds.value = next;
    }
  }
}

// Elapsed and estimate lines are wall-clock; the queue poll is far too coarse.
watch(inspectedQueueId, (id) => {
  queueRowError.value = null;
  queuePreview.value = null;
  stopQueuePreview?.();
  stopQueuePreview = null;
  if (queueDetailTimer !== null) clearInterval(queueDetailTimer);
  queueDetailTimer = null;
  if (id === null) return;
  queueDetailNowMs.value = Date.now();
  queueDetailTimer = setInterval(() => (queueDetailNowMs.value = Date.now()), 1_000);

  const row = queue.value.find((candidate) => candidate.id === id);
  if (row?.state !== "running") return;
  stopQueuePreview = watchSelectedQueuePreview(
    target.value,
    id,
    (preview) => (queuePreview.value = preview),
    750,
    () => (queuePreview.value = null),
  );
});

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
  if (dispatchPaused.value && entry.state === "queued") return "PAUSED";
  // Same waiting vocabulary as the Create queue, resolved once in studio —
  // a held row reads HELD, never a place in line.
  return queueWaitCode(resolveQueueWait({ state: entry.state, position: entry.position }));
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
  stopQueuePreview?.();
  if (queueDetailTimer !== null) clearInterval(queueDetailTimer);
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
      <span class="host-chip" data-test="host-detail-health">{{
        mobileHostHealthLabel(host)
      }}</span>
    </div>

    <div class="mobile-detail-title">
      <span
        class="status-dot"
        :class="host.stale ? 'is-reconnecting' : host.online ? 'is-ready' : 'is-error'"
        aria-hidden="true"
      />
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
    <p v-else-if="host.instanceMismatch" class="status-line error-text" role="alert">
      This address now reports a different Mold server identity. Remove and re-add the machine to
      trust that replacement.
    </p>
    <p v-else-if="host.stale" class="status-line" role="status">
      Reconnecting… Showing the last verified host state.
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
              :data-pressure="hostMemoryPressure ?? undefined"
              :title="hostMemoryLabel ?? undefined"
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
          <div class="mobile-host-queue-controls">
            <span v-if="resumeNeeded" data-test="host-detail-queue-paused">
              {{ restartPaused && !dispatchPaused ? "PAUSED AFTER RESTART" : "PAUSED" }}
            </span>
            <span data-test="host-detail-queue-total">{{ queueSummary }}</span>
            <button
              v-if="canPauseQueue"
              type="button"
              class="secondary-button mobile-host-queue-pause"
              data-test="host-detail-queue-pause"
              :disabled="queueControlBusy"
              @click="setHostQueuePaused(!resumeNeeded)"
            >
              {{ queueControlBusy ? "Working…" : resumeNeeded ? "Resume" : "Pause" }}
            </button>
          </div>
        </div>
        <p class="mobile-empty-note" data-test="host-detail-queue-scope">
          <template v-if="queueApiAvailable">{{ queue.length }} loaded</template>
          <template v-else>Queue page unavailable</template>
          <template v-if="runtimeWindow"> · Runtime window {{ runtimeWindow }}</template>
        </p>
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
          <li v-for="entry in queue" :key="entry.id" class="mobile-queue-item">
            <SwipeActionRow
              :actions="queueRowActions(entry)"
              :label="`${modelLabel(entry.model)} job`"
              :disabled="queueRowBusy(entry)"
              :data-test="`host-detail-queue-row-${entry.id}`"
              @act="onQueueRowAction(entry, $event)"
            >
              <button
                type="button"
                class="mobile-queue-open"
                :data-test="`host-detail-queue-open-${entry.id}`"
                :aria-label="`Job details for ${modelLabel(entry.model)}`"
                @click="inspectedQueueId = entry.id"
              >
                <strong>{{ modelLabel(entry.model) }}</strong>
                <span>{{ queueRowBusy(entry) ? "WORKING…" : queueCode(entry) }}</span>
              </button>
            </SwipeActionRow>
          </li>
        </ul>
        <p
          v-if="queueRowError"
          class="status-line error-text"
          role="alert"
          data-test="host-detail-queue-row-error"
        >
          {{ queueRowError }}
        </p>
        <QueuePlanWorkList
          :plan="queuePlan"
          :exclude-ids="queueEntryIds"
          data-test="host-detail-planned-work"
        />
        <p v-if="queue.length === 0 && planOnlyWork.length === 0" class="mobile-empty-note">
          {{ durableBacklog === 0 ? "Queue is empty." : "No queue rows are loaded." }}
        </p>
        <button
          v-if="queueNextCursor"
          type="button"
          class="secondary-button mobile-host-queue-more"
          data-test="host-detail-queue-load-more"
          :disabled="loadingMoreQueue"
          @click="loadMoreQueue"
        >
          {{ loadingMoreQueue ? "Loading…" : "Load more jobs" }}
        </button>
        <p v-if="loadMoreQueueError" class="status-line error-text" role="alert">
          {{ loadMoreQueueError }}
        </p>

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

      <section
        v-if="libraryCardVisible"
        class="mobile-detail-section"
        aria-labelledby="host-library-title"
        data-test="host-detail-library"
      >
        <div class="mobile-section-head">
          <h2 id="host-library-title">Library</h2>
        </div>
        <label class="field">
          <span>Trash retention</span>
          <select
            class="control"
            :value="retentionValue ?? ''"
            :disabled="retentionSaving || !retentionEditable"
            data-test="host-detail-retention"
            @change="saveRetention(($event.target as HTMLSelectElement).value)"
          >
            <option v-for="days in retentionChoices" :key="days" :value="days">
              {{ retentionLabel(days) }}
            </option>
          </select>
        </label>
        <p
          v-if="retentionLocked"
          class="mobile-empty-note"
          data-test="host-detail-retention-locked"
        >
          Set by {{ retentionEntry?.env_var }} on this host — edit it there.
        </p>
        <p
          v-if="retentionProbeFailed"
          class="status-line error-text"
          role="alert"
          data-test="host-detail-retention-error"
        >
          Couldn't read this host's retention setting — it stays read-only.
          <button
            class="secondary-button"
            type="button"
            data-test="host-detail-retention-retry"
            @click="refreshLibraryCard()"
          >
            Retry
          </button>
        </p>
        <p v-if="retentionError" class="status-line error-text" role="alert">
          {{ retentionError }}
        </p>
        <div class="mobile-library-trash-row" data-test="host-detail-trash-row">
          <span
            >Prints in trash: <span class="mobile-library-mono">{{ trashCount ?? "–" }}</span></span
          >
          <button
            class="secondary-button mobile-inline-danger"
            type="button"
            :class="{ 'is-armed': emptyTrashArmed }"
            :disabled="emptyingTrash || trashCount === 0"
            data-test="host-detail-empty-trash"
            @click="emptyTrashNow"
            @blur="emptyTrashArmed = false"
          >
            {{ emptyingTrash ? "Emptying…" : emptyTrashArmed ? "Confirm" : "Empty trash" }}
          </button>
        </div>
        <p v-if="emptyTrashArmed" class="status-line" data-test="host-detail-empty-prompt">
          Delete everything in this host's trash forever?
        </p>
      </section>

      <!-- Specialized capability detail reads below the live instruments. -->
      <MinimaxH3InventoryPanel :hosts="h3Host" heading="H3 on this machine" />
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
      <p>Forgetting removes this address and its API key from this phone.</p>
    </section>

    <MobileLibrarySheet
      :open="inspectedQueueModel !== null"
      title="Job details"
      :focus-first-control="false"
      test-id="host-detail-queue-sheet"
      @close="inspectedQueueId = null"
    >
      <QueueEntryDetail
        v-if="inspectedQueueModel"
        :model="inspectedQueueModel"
        :preview="queuePreview"
        :cancelling="inspectedQueueEntry ? cancellingQueueIds.has(inspectedQueueEntry.id) : false"
        :error="queueRowError"
        confirm="inline"
        compact
        @close="inspectedQueueId = null"
        @cancel="inspectedQueueEntry && onQueueRowAction(inspectedQueueEntry, 'cancel')"
      />
    </MobileLibrarySheet>
  </div>
</template>
