<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { DeviceInfo } from "../api/devices";
import type { QueuePlan, QueueWorkItem } from "../api/queuePlan";
import { normalizeBlockedReason } from "../lib/queuePosition";
import {
  queueCompletionLabel,
  queueLanePositionLabel,
  queuePlanUpdateLabel,
} from "../lib/queuePlanPresentation";

const props = withDefaults(
  defineProps<{
    devices: DeviceInfo[];
    plan?: QueuePlan | null;
    mutable?: boolean;
    restartEnable?: boolean;
    showControls?: boolean;
    busyDeviceIds?: readonly string[];
  }>(),
  {
    plan: null,
    mutable: false,
    restartEnable: false,
    showControls: false,
    busyDeviceIds: () => [],
  },
);

const emit = defineEmits<{
  toggle: [deviceId: string, enabled: boolean];
  unpin: [workId: string];
}>();

const nowUnixMs = ref(Date.now());
let clockTimer: ReturnType<typeof setInterval> | null = null;
const blocked = computed(
  () => props.plan?.work_items.filter((work) => blockedReason(work)) ?? [],
);
const cpuUtilityWork = computed(
  () =>
    props.plan?.work_items
      .filter(
        (work) =>
          work.planned_lane_kind === "host_utility" &&
          !hasUnmatchedDevice(work) &&
          !blockedReason(work),
      )
      .sort((a, b) => (a.lane_order ?? 0) - (b.lane_order ?? 0)) ?? [],
);
const otherComputeWork = computed(
  () =>
    props.plan?.work_items
      .filter(
        (work) =>
          work.planned_lane_kind != null &&
          work.planned_lane_kind !== "device" &&
          work.planned_lane_kind !== "host_utility" &&
          !hasUnmatchedDevice(work) &&
          !blockedReason(work),
      )
      .sort((a, b) => (a.lane_order ?? 0) - (b.lane_order ?? 0)) ?? [],
);
const unassignedDeviceWork = computed(
  () =>
    props.plan?.work_items
      .filter((work) => hasUnmatchedDevice(work) && !blockedReason(work))
      .sort((a, b) => (a.lane_order ?? 0) - (b.lane_order ?? 0)) ?? [],
);
const compact = computed(
  () =>
    props.devices.length +
      Number(cpuUtilityWork.value.length > 0) +
      Number(otherComputeWork.value.length > 0) +
      Number(unassignedDeviceWork.value.length > 0) <=
    1,
);
const hasComputeLanes = computed(
  () =>
    props.devices.length > 0 ||
    cpuUtilityWork.value.length > 0 ||
    otherComputeWork.value.length > 0 ||
    unassignedDeviceWork.value.length > 0,
);
const laneCount = computed(
  () =>
    props.devices.length +
    Number(cpuUtilityWork.value.length > 0) +
    Number(otherComputeWork.value.length > 0) +
    Number(unassignedDeviceWork.value.length > 0),
);
const lifecycleNote = computed(() => {
  if (!props.showControls || props.devices.length === 0) return null;
  if (props.mutable) {
    return "Disabling a busy GPU lets its current stage finish, then removes it from scheduling.";
  }
  if (props.restartEnable) {
    return "Live GPU controls require Scheduler V2. Disabled GPUs can be enabled for the next server restart.";
  }
  return "Live GPU controls are unavailable on this server.";
});

function gib(bytes: number | null): string {
  return bytes === null ? "—" : `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function memoryLabel(device: DeviceInfo): string {
  return `${gib(device.memory.used_bytes)} of ${gib(device.memory.total_bytes)}`;
}

function stateLabel(device: DeviceInfo): string {
  if (device.admin_state === "draining") return "Finishing current work";
  if (device.restart_required) return "Restart required";
  if (device.admin_state === "starting") return "Starting";
  if (device.health !== "healthy") return device.health;
  return device.admin_state.replace("_", " ");
}

function planned(device: DeviceInfo): QueueWorkItem[] {
  return (
    props.plan?.work_items
      .filter(
        (work) =>
          work.planned_lane_kind === "device" &&
          work.planned_device_id === device.id &&
          work.work_id !== device.active_work_id &&
          !blockedReason(work),
      )
      .sort((a, b) => (a.lane_order ?? 0) - (b.lane_order ?? 0)) ?? []
  );
}

function activeWorkLabel(device: DeviceInfo): string {
  const work = props.plan?.work_items.find(
    (candidate) => candidate.work_id === device.active_work_id,
  );
  return work ? workLabel(work) : "Generation in progress";
}

/**
 * Work the scheduler has not placed on any lane yet carries neither a lane
 * kind nor a device id — it is queued, not lost. Only a device-lane item that
 * names no device, or names one this snapshot lacks, is genuinely unmatched.
 */
function hasUnmatchedDevice(work: QueueWorkItem): boolean {
  if (work.planned_device_id == null)
    return work.planned_lane_kind === "device";
  return !props.devices.some((device) => device.id === work.planned_device_id);
}

function blockedReason(work: QueueWorkItem): string | null {
  return normalizeBlockedReason(work.blocked_reason ?? work.reason);
}

function pinnedDevice(work: QueueWorkItem): DeviceInfo | null {
  return (
    props.devices.find((device) => device.id === work.hard_pinned_device_id) ??
    null
  );
}

function pinnedDeviceLabel(work: QueueWorkItem): string {
  return pinnedDevice(work)?.name ?? "an unavailable machine";
}

function eta(work: QueueWorkItem): string {
  return queueCompletionLabel(
    work.estimated_finish_unix_ms,
    work.estimate_confidence,
    nowUnixMs.value,
  );
}

function workKindLabel(kind: unknown): string {
  const words = typeof kind === "string" ? kind.replaceAll("_", " ") : "";
  return words.length ? `${words[0]!.toUpperCase()}${words.slice(1)}` : "Work";
}

function workLabel(work: QueueWorkItem): string {
  const kind = workKindLabel(work.work_kind);
  return work.chain_stage == null
    ? kind
    : `${kind} · stage ${work.chain_stage + 1}`;
}

function replanLabel(): string | null {
  return queuePlanUpdateLabel(
    props.plan?.next_replan_at_unix_ms,
    nowUnixMs.value,
  );
}

function canToggle(): boolean {
  return props.mutable || props.restartEnable || props.showControls;
}

function toggleDisabled(device: DeviceInfo): boolean {
  return (
    device.admin_state === "startup_excluded" ||
    props.busyDeviceIds.includes(device.id) ||
    (!props.mutable && !(props.restartEnable && !device.desired_enabled))
  );
}

function toggleLabel(device: DeviceInfo): string {
  if (props.mutable) return device.desired_enabled ? "Disable" : "Enable";
  if (!device.desired_enabled)
    return props.restartEnable ? "Enable on restart" : "Enable";
  return device.restart_required ? "Enabled on restart" : "Disable";
}

onMounted(() => {
  clockTimer = setInterval(() => {
    nowUnixMs.value = Date.now();
  }, 1_000);
});

onBeforeUnmount(() => {
  if (clockTimer !== null) clearInterval(clockTimer);
});
</script>

<template>
  <section
    class="device-panel"
    :class="{ 'device-panel--compact': compact }"
    :data-device-count="devices.length"
    :data-lane-count="laneCount"
    data-test="device-panel"
  >
    <header class="device-panel__head">
      <span>Compute plan</span>
      <span
        v-if="replanLabel()"
        class="device-panel__tentative"
        data-test="replan-countdown"
      >
        {{ replanLabel() }}
      </span>
    </header>

    <p
      v-if="lifecycleNote"
      class="device-panel__lifecycle"
      data-test="device-lifecycle-note"
    >
      {{ lifecycleNote }}
    </p>

    <p v-if="!hasComputeLanes" class="device-panel__empty">
      No compute devices visible.
    </p>
    <div v-else class="device-panel__grid">
      <article
        v-for="device in devices"
        :key="device.id"
        class="device-card"
        :data-state="device.admin_state"
        :data-health="device.health"
        data-test="device-card"
      >
        <div class="device-card__title">
          <span class="device-card__name">{{ device.name }}</span>
          <span class="device-card__badge">{{
            device.backend.toUpperCase()
          }}</span>
          <span v-if="device.device_kind === 'mig'" class="device-card__badge">
            MIG {{ device.mig_profile || "" }}
          </span>
        </div>
        <div class="device-card__meta">
          <span v-if="device.ordinal !== null">GPU {{ device.ordinal }}</span>
          <span>{{ stateLabel(device) }}</span>
        </div>

        <div class="device-card__metrics">
          <span>GPU memory {{ memoryLabel(device) }}</span>
          <span>
            GPU use
            {{
              device.telemetry.utilization_percent === null
                ? "—"
                : `${Math.round(device.telemetry.utilization_percent)}%`
            }}
          </span>
        </div>
        <div v-if="device.loaded_models.length" class="device-card__line">
          <span class="device-card__line-label">Loaded</span>
          <span
            class="device-card__line-value"
            :title="device.loaded_models.join(', ')"
          >
            {{ device.loaded_models.join(", ") }}
          </span>
        </div>
        <div v-if="device.active_work_id" class="device-card__line">
          <span class="device-card__line-label">Running now</span>
          <span class="device-card__line-value">{{
            activeWorkLabel(device)
          }}</span>
        </div>
        <ol
          v-if="planned(device).length"
          class="device-card__lane"
          data-test="device-lane"
        >
          <li v-for="(work, index) in planned(device)" :key="work.work_id">
            <span class="device-card__work">
              {{ queueLanePositionLabel(index) }} · {{ workLabel(work) }}
            </span>
            <span class="device-card__eta">{{ eta(work) }}</span>
          </li>
        </ol>
        <button
          v-if="canToggle()"
          type="button"
          class="device-card__toggle"
          :disabled="toggleDisabled(device)"
          :aria-pressed="device.desired_enabled"
          :data-test="`device-toggle-${device.ordinal ?? device.id}`"
          @click="emit('toggle', device.id, !device.desired_enabled)"
        >
          {{ toggleLabel(device) }}
        </button>
      </article>
      <article
        v-if="unassignedDeviceWork.length"
        class="device-card device-card--utility"
        data-test="unassigned-device-lane"
      >
        <div class="device-card__title">
          <span class="device-card__name">Unassigned / unknown device</span>
          <span class="device-card__badge">UNKNOWN</span>
        </div>
        <div class="device-card__meta">
          <span>No matching compute device in this snapshot</span>
        </div>
        <ol class="device-card__lane">
          <li v-for="(work, index) in unassignedDeviceWork" :key="work.work_id">
            <span class="device-card__work">
              {{ queueLanePositionLabel(index) }} · {{ workLabel(work) }}
            </span>
            <span class="device-card__eta">{{ eta(work) }}</span>
          </li>
        </ol>
      </article>
      <article
        v-if="cpuUtilityWork.length"
        class="device-card device-card--utility"
        data-test="cpu-utility-lane"
      >
        <div class="device-card__title">
          <span class="device-card__name">Machine utility</span>
          <span class="device-card__badge">CPU</span>
        </div>
        <div class="device-card__meta">
          <span>One task at a time</span>
        </div>
        <ol class="device-card__lane" data-test="cpu-utility-lane-list">
          <li v-for="(work, index) in cpuUtilityWork" :key="work.work_id">
            <span class="device-card__work">
              {{ queueLanePositionLabel(index) }} · {{ workLabel(work) }}
            </span>
            <span class="device-card__eta">{{ eta(work) }}</span>
          </li>
        </ol>
      </article>
      <article
        v-if="otherComputeWork.length"
        class="device-card device-card--utility"
        data-test="other-compute-lane"
      >
        <div class="device-card__title">
          <span class="device-card__name">Scheduled work</span>
          <span class="device-card__badge">OTHER</span>
        </div>
        <ol class="device-card__lane">
          <li v-for="(work, index) in otherComputeWork" :key="work.work_id">
            <span class="device-card__work">
              {{ queueLanePositionLabel(index) }} · {{ workLabel(work) }}
            </span>
            <span class="device-card__eta">{{ eta(work) }}</span>
          </li>
        </ol>
      </article>
    </div>

    <div
      v-if="blocked.length"
      class="device-panel__blocked"
      data-test="blocked-work"
    >
      <strong>Blocked</strong>
      <div v-for="work in blocked" :key="work.work_id">
        {{ workLabel(work) }} · {{ blockedReason(work) }}
        <template v-if="work.hard_pinned_device_id">
          · pinned to {{ pinnedDeviceLabel(work) }}
          <button
            v-if="mutable"
            type="button"
            class="device-panel__blocked-action"
            @click="emit('unpin', work.parent_id || work.work_id)"
          >
            Use Auto
          </button>
          <button
            v-if="
              mutable &&
              pinnedDevice(work) &&
              !pinnedDevice(work)!.desired_enabled
            "
            type="button"
            class="device-panel__blocked-action"
            @click="emit('toggle', pinnedDevice(work)!.id, true)"
          >
            Re-enable
          </button>
        </template>
      </div>
    </div>
  </section>
</template>

<style scoped>
.device-panel {
  display: grid;
  gap: 12px;
  color: var(--mold-text);
}
.device-panel__head {
  display: flex;
  gap: 12px;
  align-items: baseline;
  justify-content: space-between;
  font-weight: 650;
}
.device-panel__tentative,
.device-panel__empty {
  color: var(--mold-text-dim, #777);
  font-size: var(--mold-fs-xs);
  font-weight: 500;
}
.device-panel__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
  align-items: stretch;
  gap: 10px;
}
.device-panel--compact .device-panel__grid {
  grid-template-columns: minmax(0, 1fr);
}
.device-card {
  display: grid;
  min-width: 0;
  min-height: 226px;
  align-content: start;
  gap: 8px;
  padding: 12px;
  /* `--line` and `--surface-2` were never defined anywhere, so these cards
     drew a #d5d5d5 hairline on every dark theme. The token the old name
     meant is the theme's own border. */
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: transparent;
}
.device-card--utility {
  border-style: dashed;
}
.device-card[data-state="draining"],
.device-card[data-health="degraded"] {
  border-color: var(--mold-warning, #b87800);
}
.device-card[data-state="disabled"],
.device-card[data-health="unavailable"],
.device-card[data-health="poisoned"] {
  opacity: 0.68;
}
.device-card__title {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 6px;
  align-items: center;
}
.device-card__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
  align-items: center;
}
.device-card__metrics {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 6px 10px;
  align-items: center;
}
.device-card__name {
  min-width: 0;
  font-weight: 650;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.device-card__badge {
  white-space: nowrap;
  padding: 2px 6px;
  border: var(--mold-bw) solid var(--mold-border);
  /* Never a pill (ui/mold-desktop.css): only knobs and dots are circles. */
  border-radius: var(--mold-radius-1);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.04em;
}
.device-card__meta,
.device-card__metrics,
.device-card__line,
.device-card__lane,
.device-panel__blocked {
  color: var(--mold-text-2, #555);
  font-size: var(--mold-fs-xs);
}
.device-card__line {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 6px;
  min-width: 0;
}
.device-card__line-label::after {
  content: ":";
}
.device-card__line-value {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.device-card__lane {
  display: grid;
  gap: 4px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.device-card__lane li {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  min-width: 0;
}
.device-card__work {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.device-card__eta {
  white-space: nowrap;
}
.device-card__toggle {
  margin-top: auto;
  justify-self: start;
  min-height: 32px;
  padding: 4px 10px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
}
.device-panel__blocked {
  display: grid;
  gap: 4px;
  padding: 10px 12px;
  border-left: 3px solid var(--mold-warning, #b87800);
}
.device-panel__blocked-action {
  margin-left: 8px;
  min-height: 28px;
  padding: 2px 8px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
}
@media (max-width: 639px) {
  .device-card__toggle {
    min-height: 44px;
  }
}
</style>
