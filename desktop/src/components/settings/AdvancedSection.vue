<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from "vue";
import DevicePanel from "@studio/components/DevicePanel.vue";
import { listDevices, setDeviceEnabled, type DeviceInfo } from "@studio/api/devices";
import {
  listQueue,
  queuePageRequestForCapacity,
  setQueueDevicePin,
  type QueuePlan,
} from "@studio/api/queuePlan";
import ConfigRowItem from "./ConfigRowItem.vue";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import PlacementSection from "./PlacementSection.vue";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";
import { useConnectionStore } from "../../stores/connection";
import { schemaFor } from "../../lib/settingsSchema";
import type { ConfigRow } from "../../lib/api/types";
import { subscribeToDeviceSnapshots } from "../../lib/api/deviceEvents";
import { fetchServerCapabilities } from "../../lib/api/serverCapabilities";
import { apiJsonTo } from "../../lib/api/client";
import type { ServerStatus } from "../../lib/api/types";

defineProps<{ filter?: ((row: ConfigRow) => boolean) | undefined }>();

const config = useSettingsConfigStore();
const toasts = useToastStore();
const connection = useConnectionStore();
const devices = ref<DeviceInfo[]>([]);
const plan = ref<QueuePlan | null>(null);
const mutatingDeviceIds = ref(new Set<string>());
const lifecycleMutable = ref(false);
const restartEnable = ref(false);
let deviceEventsAbort: AbortController | null = null;
let deviceRequestGeneration = 0;

function target() {
  return connection.baseUrl ? { baseUrl: connection.baseUrl, apiKey: connection.apiKey } : null;
}

async function loadDevices() {
  const apiTarget = target();
  const generation = ++deviceRequestGeneration;
  if (!apiTarget) {
    devices.value = [];
    plan.value = null;
    lifecycleMutable.value = false;
    restartEnable.value = false;
    return;
  }
  const isCurrent = () => {
    const current = target();
    return (
      generation === deviceRequestGeneration &&
      current?.baseUrl === apiTarget.baseUrl &&
      current.apiKey === apiTarget.apiKey
    );
  };
  const statusRequest = apiJsonTo<ServerStatus>(apiTarget, "/api/status");
  const queueRequest = statusRequest.then((status) =>
    listQueue(apiTarget, queuePageRequestForCapacity(status.queue_capacity) ?? null),
  );
  const [deviceResult, queueResult, capabilityResult] = await Promise.allSettled([
    listDevices(apiTarget),
    queueRequest,
    fetchServerCapabilities(apiTarget),
  ]);
  if (!isCurrent()) return;
  if (deviceResult.status === "fulfilled") devices.value = deviceResult.value.devices;
  if (queueResult.status === "fulfilled") plan.value = queueResult.value.plan;
  lifecycleMutable.value =
    capabilityResult.status === "fulfilled" &&
    capabilityResult.value.devices?.lifecycle === true &&
    capabilityResult.value.dispatch?.v2_authoritative === true;
  restartEnable.value =
    capabilityResult.status === "fulfilled" &&
    capabilityResult.value.devices?.restart_enable === true;
}

async function toggleDevice(deviceId: string, enabled: boolean) {
  const apiTarget = target();
  if (!apiTarget) return;
  mutatingDeviceIds.value = new Set(mutatingDeviceIds.value).add(deviceId);
  try {
    const accepted = await setDeviceEnabled(apiTarget, deviceId, enabled);
    devices.value = devices.value.map((device) => (device.id === accepted.id ? accepted : device));
    await loadDevices();
  } catch (error) {
    toasts.push(`Device state was not changed: ${String(error)}`, "error");
  } finally {
    const next = new Set(mutatingDeviceIds.value);
    next.delete(deviceId);
    mutatingDeviceIds.value = next;
  }
}

async function unpinWork(workId: string) {
  const apiTarget = target();
  if (!apiTarget) return;
  try {
    await setQueueDevicePin(apiTarget, workId, null);
    await loadDevices();
  } catch (error) {
    toasts.push(`Queue pin was not changed: ${String(error)}`, "error");
  }
}

watch(
  [() => connection.baseUrl, () => connection.apiKey],
  () => {
    deviceEventsAbort?.abort();
    deviceEventsAbort = null;
    const apiTarget = target();
    if (!apiTarget) {
      void loadDevices();
      return;
    }
    // Establish an authoritative snapshot even when this is an older server
    // without `/api/events`; onOpen and relevant deltas repair it thereafter.
    void loadDevices();
    deviceEventsAbort = new AbortController();
    subscribeToDeviceSnapshots(apiTarget, deviceEventsAbort.signal, () => {
      void loadDevices();
    });
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  deviceRequestGeneration += 1;
  deviceEventsAbort?.abort();
});

async function save(row: ConfigRow, value: ConfigRow["value"]) {
  const error = await config.save(row.key, value);
  toasts.push(error ?? `Saved ${row.key}`, error ? "error" : "info");
}
async function reset(row: ConfigRow) {
  const error = await config.reset(row.key);
  toasts.push(error ?? `Reset ${row.key}`, error ? "error" : "info");
}
</script>

<template>
  <div>
    <DevicePanel
      v-if="!filter"
      class="mb-5"
      :devices="devices"
      :plan="plan"
      :mutable="lifecycleMutable"
      :restart-enable="restartEnable"
      show-controls
      :busy-device-ids="[...mutatingDeviceIds]"
      @unpin="unpinWork"
      @toggle="toggleDevice"
    />
    <ConfigSettingRow schema-key="server_port" />
    <PlacementSection v-if="!filter" class="mt-5" />
    <p class="mt-4 mb-1 text-micro text-fg-dim">
      Everything the engine exposes that has no curated control — including keys added by newer
      engines. Provenance: ⌂ database · ⛁ config.toml · ⚿ environment.
    </p>
    <template v-for="row in config.advancedRows" :key="row.key">
      <ConfigRowItem
        v-if="!schemaFor(row.key) && (!filter || filter(row))"
        :row="row"
        @save="(v) => save(row, v)"
        @reset="reset(row)"
      />
    </template>
  </div>
</template>
