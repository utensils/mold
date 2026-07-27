<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { DeviceInfo } from "../api/devices";
import type { QueuePlan, QueueWorkItem } from "../api/queuePlan";

const props = withDefaults(
  defineProps<{
    devices: DeviceInfo[];
    plan?: QueuePlan | null;
    mutable?: boolean;
    restartEnable?: boolean;
    showControls?: boolean;
    busyDeviceId?: string | null;
  }>(),
  {
    plan: null,
    mutable: false,
    restartEnable: false,
    showControls: false,
    busyDeviceId: null,
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
          work.planned_lane_kind === "host_utility" && !blockedReason(work),
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
          !blockedReason(work),
      )
      .sort((a, b) => (a.lane_order ?? 0) - (b.lane_order ?? 0)) ?? [],
);
const compact = computed(
  () =>
    props.devices.length +
      Number(cpuUtilityWork.value.length > 0) +
      Number(otherComputeWork.value.length > 0) <=
    1,
);
const hasComputeLanes = computed(
  () =>
    props.devices.length > 0 ||
    cpuUtilityWork.value.length > 0 ||
    otherComputeWork.value.length > 0,
);
const laneCount = computed(
  () =>
    props.devices.length +
    Number(cpuUtilityWork.value.length > 0) +
    Number(otherComputeWork.value.length > 0),
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

function shortId(id: string): string {
  const tail = id.split(":").at(-1) ?? id;
  return tail.length > 10 ? `…${tail.slice(-8)}` : tail;
}

function gib(bytes: number | null): string {
  return bytes === null ? "—" : `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function memoryLabel(device: DeviceInfo): string {
  return `${gib(device.memory.used_bytes)} / ${gib(device.memory.total_bytes)}`;
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
          (work.planned_lane_kind === "device" ||
            work.planned_lane_kind == null) &&
          work.planned_device_id === device.id &&
          !blockedReason(work),
      )
      .sort((a, b) => (a.lane_order ?? 0) - (b.lane_order ?? 0)) ?? []
  );
}

function blockedReason(work: QueueWorkItem): string | null {
  const reason = work.blocked_reason ?? work.reason;
  if (
    !reason ||
    ["priority", "starvation_forced", "warm_resident"].includes(reason)
  ) {
    return null;
  }
  return reason.replaceAll("_", " ");
}

function pinnedDevice(work: QueueWorkItem): DeviceInfo | null {
  return (
    props.devices.find((device) => device.id === work.hard_pinned_device_id) ??
    null
  );
}

function eta(work: QueueWorkItem): string {
  const finish = work.estimated_finish_unix_ms;
  if (finish == null) return "ETA pending";
  const seconds = Math.max(0, Math.ceil((finish - nowUnixMs.value) / 1000));
  return `~${seconds}s · ${work.estimate_confidence} confidence`;
}

function workKindLabel(kind: unknown): string {
  const words = typeof kind === "string" ? kind.replaceAll("_", " ") : "";
  return words.length ? `${words[0]!.toUpperCase()}${words.slice(1)}` : "Work";
}

function replanLabel(): string | null {
  const deadline = props.plan?.next_replan_at_unix_ms;
  if (deadline == null) return null;
  return `Tentative plan · optimizing in ${Math.max(
    0,
    Math.ceil((deadline - nowUnixMs.value) / 1000),
  )}s`;
}

function canToggle(): boolean {
  return props.mutable || props.restartEnable || props.showControls;
}

function toggleDisabled(device: DeviceInfo): boolean {
  return (
    device.admin_state === "startup_excluded" ||
    props.busyDeviceId === device.id ||
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
      <span>Compute</span>
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
          <code :title="device.id">{{ shortId(device.id) }}</code>
          <span v-if="device.ordinal !== null">GPU {{ device.ordinal }}</span>
          <span>{{ stateLabel(device) }}</span>
        </div>

        <div class="device-card__metrics">
          <span>VRAM {{ memoryLabel(device) }}</span>
          <span>
            Utilization
            {{
              device.telemetry.utilization_percent === null
                ? "—"
                : `${Math.round(device.telemetry.utilization_percent)}%`
            }}
          </span>
        </div>
        <div v-if="device.loaded_models.length" class="device-card__line">
          Loaded: {{ device.loaded_models.join(", ") }}
        </div>
        <div v-if="device.active_work_id" class="device-card__line">
          Active: {{ device.active_work_id }}
        </div>
        <ol
          v-if="planned(device).length"
          class="device-card__lane"
          data-test="device-lane"
        >
          <li v-for="work in planned(device)" :key="work.work_id">
            {{ work.work_id }} · {{ eta(work) }}
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
        v-if="cpuUtilityWork.length"
        class="device-card device-card--utility"
        data-test="cpu-utility-lane"
      >
        <div class="device-card__title">
          <span class="device-card__name">Host utility</span>
          <span class="device-card__badge">CPU</span>
        </div>
        <div class="device-card__meta">
          <span>One task at a time</span>
        </div>
        <ol class="device-card__lane" data-test="cpu-utility-lane-list">
          <li v-for="work in cpuUtilityWork" :key="work.work_id">
            {{ work.work_id }} · {{ workKindLabel(work.work_kind) }} ·
            {{ eta(work) }}
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
          <li v-for="work in otherComputeWork" :key="work.work_id">
            {{ work.work_id }} · {{ workKindLabel(work.work_kind) }} ·
            {{ eta(work) }}
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
        {{ work.work_id }} · {{ blockedReason(work) }}
        <template v-if="work.hard_pinned_device_id">
          · pinned to {{ shortId(work.hard_pinned_device_id) }}
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
  color: var(--ink, currentColor);
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
  color: var(--ink-3, #777);
  font-size: 12px;
  font-weight: 500;
}
.device-panel__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 10px;
}
.device-panel--compact .device-panel__grid {
  grid-template-columns: minmax(0, 1fr);
}
.device-card {
  display: grid;
  gap: 8px;
  padding: 12px;
  border: 1px solid var(--line, #d5d5d5);
  border-radius: 10px;
  background: var(--surface-2, transparent);
}
.device-card--utility {
  border-style: dashed;
}
.device-card[data-state="draining"],
.device-card[data-health="degraded"] {
  border-color: var(--warning, #b87800);
}
.device-card[data-state="disabled"],
.device-card[data-health="unavailable"],
.device-card[data-health="poisoned"] {
  opacity: 0.68;
}
.device-card__title,
.device-card__meta,
.device-card__metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
  align-items: center;
}
.device-card__name {
  min-width: 0;
  flex: 1;
  font-weight: 650;
}
.device-card__badge {
  padding: 2px 6px;
  border: 1px solid var(--line, #aaa);
  border-radius: 999px;
  font-size: 10px;
  letter-spacing: 0.04em;
}
.device-card__meta,
.device-card__metrics,
.device-card__line,
.device-card__lane,
.device-panel__blocked {
  color: var(--ink-2, #555);
  font-size: 12px;
}
.device-card__metrics {
  justify-content: space-between;
}
.device-card__lane {
  margin: 0;
  padding-left: 18px;
}
.device-card__toggle {
  justify-self: start;
  min-height: 32px;
  padding: 4px 10px;
  border: 1px solid var(--line, #aaa);
  border-radius: 7px;
}
.device-panel__blocked {
  display: grid;
  gap: 4px;
  padding: 10px 12px;
  border-left: 3px solid var(--warning, #b87800);
}
.device-panel__blocked-action {
  margin-left: 8px;
  min-height: 28px;
  padding: 2px 8px;
  border: 1px solid var(--line, #aaa);
  border-radius: 7px;
}
@media (max-width: 639px) {
  .device-card__toggle {
    min-height: 44px;
  }
}
</style>
