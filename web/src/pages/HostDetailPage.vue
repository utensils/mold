<script setup lang="ts">
/*
 * Host detail (spec §04 / §08 G3). Live telemetry, the relocated queue
 * management card, active downloads, and installed models for one machine —
 * the primary origin or any remembered remote, all through the same per-host
 * client. Absent metrics render as em dashes and an offline host keeps its
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
import { setQueueDevicePin, type QueuePlan } from "@studio/api/queuePlan";
import {
  modelDisplayName,
  modelDisplayNameForId,
} from "@studio/lib/modelDisplay";
import StatusDot from "../components/machines/StatusDot.vue";
import QueueCard from "../components/machines/QueueCard.vue";
import {
  cancelQueueJob,
  cancelAllHostQueue,
  hostCapabilities,
  hostDownloads,
  hostModels,
  hostQueue,
  moveQueueJob,
  pauseHostQueue,
  resumeHostQueue,
  setQueueJobLane,
  useHostPoll,
  type HostCapabilities,
} from "../components/machines/hostClient";
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
import { reorderQueueJob, updateQueueJobTargetGpu } from "../api";
import { requestConfirm, requestText, toast } from "../lib/toasts";
import { subscribeToDeviceSnapshots } from "../lib/deviceEvents";
import type { DownloadJobWire, ModelInfoExtended, QueueEntry } from "../types";
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
const mutatingDeviceIds = ref(new Set<string>());
const models = ref<ModelInfoExtended[]>([]);
const downloads = ref<DownloadJobWire[]>([]);
const targetId = ref(getGenerateTargetId());

const poll = useHostPoll(liveHost, { withResources: true, intervalMs: 4000 });

const online = computed(() => poll.online.value);
const loading = computed(() => poll.loading.value);
const offline = computed(
  () => !!host.value && !disconnected.value && !loading.value && !online.value,
);
const dotState = computed<"online" | "offline" | "unknown">(() => {
  if (!host.value || disconnected.value || loading.value) return "unknown";
  return online.value ? "online" : "offline";
});

const telemetry = computed(() =>
  deriveTelemetry(poll.status.value ?? null, poll.resources.value ?? null),
);
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
    const listing = await hostQueue(entry, signal);
    if (
      generation === queueRequestGeneration &&
      isCurrentSession(entry, epoch)
    ) {
      queue.value = listing.entries;
      queuePlan.value = listing.plan ?? null;
    }
  } catch {
    // Keep the last-good queue; the offline banner covers the failure.
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
  try {
    const next = await hostModels(entry, signal);
    if (isCurrentSession(entry, epoch)) models.value = next;
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
  try {
    const listing = await hostDownloads(entry, signal);
    const active = [
      ...(listing.active ? [listing.active] : []),
      ...(listing.active_jobs ?? []),
      ...listing.queued,
    ];
    if (isCurrentSession(entry, epoch)) downloads.value = active;
  } catch {
    /* keep stale */
  }
}

async function reloadAll(
  entry = host.value,
  epoch = sessionEpoch,
  signal?: AbortSignal,
) {
  await Promise.all([
    reloadCapabilities(entry, epoch, signal),
    reloadQueue(entry, epoch, signal),
    reloadModels(entry, epoch, signal),
    reloadDownloads(entry, epoch, signal),
  ]);
}

async function reloadCapabilities(
  entry = host.value,
  epoch = sessionEpoch,
  signal?: AbortSignal,
) {
  if (!entry) return;
  try {
    const next = await hostCapabilities(entry, signal);
    if (isCurrentSession(entry, epoch)) caps.value = next;
  } catch {
    if (isCurrentSession(entry, epoch)) caps.value = null;
  }
}

function toggleTarget() {
  const next = isTarget.value ? ORIGIN_HOST_ID : hostId.value;
  setGenerateTargetId(next);
  targetId.value = next;
}

async function onCancel(id: string) {
  const entry = host.value;
  if (!entry) return;
  try {
    await cancelQueueJob(entry, id);
    await reloadQueue();
  } catch (e) {
    toast("error", `Couldn't cancel job: ${errMsg(e)}`);
  }
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

let timer: ReturnType<typeof setInterval> | null = null;
let deviceEventsAbort: AbortController | null = null;
let sessionAbort: AbortController | null = null;

function stopHostSession() {
  sessionEpoch += 1;
  queueRequestGeneration += 1;
  if (timer) clearInterval(timer);
  timer = null;
  deviceEventsAbort?.abort();
  deviceEventsAbort = null;
  sessionAbort?.abort();
  sessionAbort = null;
}

function startHostSession() {
  stopHostSession();
  const entry = liveHost.value;
  const epoch = sessionEpoch;
  caps.value = null;
  queue.value = [];
  queuePlan.value = null;
  models.value = [];
  downloads.value = [];
  hostName.value = host.value?.name ?? "";
  if (!entry) return;
  sessionAbort = new AbortController();
  const signal = sessionAbort.signal;
  void reloadAll(entry, epoch, signal);
  deviceEventsAbort = new AbortController();
  subscribeToDeviceSnapshots(
    { baseUrl: entry.url, apiKey: entry.apiKey ?? null },
    deviceEventsAbort.signal,
    () => {
      void poll.refresh();
      void reloadAll(entry, epoch, signal);
    },
  );
  timer = setInterval(() => void reloadAll(entry, epoch, signal), 4000);
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
  <div class="mx-auto max-w-[1800px] px-4 pb-40 pt-6 sm:px-6 lg:px-10">
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

      <div v-if="offline" class="md-offline" data-test="detail-offline">
        <span>This machine is offline. Showing the last known values.</span>
        <button type="button" class="md-retry" @click="poll?.refresh()">
          Retry
        </button>
      </div>

      <MinimaxH3InventoryPanel :hosts="h3Host" heading="H3 on this machine" />

      <div class="md-grid" :data-dimmed="offline ? 'true' : undefined">
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
            <span>Memory {{ telemetry.memLabel }}</span>
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

          <div class="md-metric md-metric--mt">
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
            :value="telemetry.ramPct ?? 0"
            tone="info"
            label="System RAM"
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
      </div>

      <div class="md-queue" :data-dimmed="offline ? 'true' : undefined">
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
          :models="models"
          :gpu-ordinals="gpuOrdinals"
          :can-reorder="canReorder"
          :can-pause="caps?.queue?.can_pause === true"
          :can-cancel-all="caps?.queue?.can_cancel_all === true"
          :paused="paused"
          :dimmed="offline"
          @cancel="onCancel"
          @set-lane="onSetLane"
          @move="onMove"
          @toggle-pause="onTogglePause"
          @cancel-all="onCancelAll"
        />
      </div>

      <CardSurface
        class="md-downloads"
        :data-dimmed="offline ? 'true' : undefined"
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
    </template>
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
  display: flex;
  gap: 16px;
  align-items: flex-start;
  flex-wrap: wrap;
}

.md-grid[data-dimmed="true"],
.md-queue[data-dimmed="true"],
.md-downloads[data-dimmed="true"] {
  opacity: 0.6;
}

.md-telemetry {
  flex: 1;
  min-width: 280px;
}

.md-models {
  width: 300px;
  flex: 0 0 300px;
}

@media (max-width: 640px) {
  .md-models {
    width: 100%;
    flex: 1 1 100%;
  }
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
  display: flex;
  flex-direction: column;
  gap: 2px;
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
