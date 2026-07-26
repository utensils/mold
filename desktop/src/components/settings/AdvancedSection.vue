<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from "vue";
import DevicePanel from "@studio/components/DevicePanel.vue";
import { listDevices, setDeviceEnabled, type DeviceInfo } from "@studio/api/devices";
import { listQueue, setQueueDevicePin, type QueuePlan } from "@studio/api/queuePlan";
import ConfigRowItem from "./ConfigRowItem.vue";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import PlacementSection from "./PlacementSection.vue";
import DeviceSettingsPanel from "./DeviceSettingsPanel.vue";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";
import { useConnectionStore } from "../../stores/connection";
import { schemaFor } from "../../lib/settingsSchema";
import type { ConfigRow } from "../../lib/api/types";
import { subscribeToDeviceSnapshots } from "../../lib/api/deviceEvents";

defineProps<{ filter?: ((row: ConfigRow) => boolean) | undefined }>();

const config = useSettingsConfigStore();
const toasts = useToastStore();
const connection = useConnectionStore();
const devices = ref<DeviceInfo[]>([]);
const plan = ref<QueuePlan | null>(null);
const mutatingDeviceId = ref<string | null>(null);
let deviceEventsAbort: AbortController | null = null;

function target() {
  return connection.baseUrl ? { baseUrl: connection.baseUrl, apiKey: connection.apiKey } : null;
}

async function loadDevices() {
  const apiTarget = target();
  if (!apiTarget) {
    devices.value = [];
    plan.value = null;
    return;
  }
  const [deviceResult, queueResult] = await Promise.allSettled([
    listDevices(apiTarget),
    listQueue(apiTarget),
  ]);
  if (deviceResult.status === "fulfilled") devices.value = deviceResult.value.devices;
  if (queueResult.status === "fulfilled") plan.value = queueResult.value.plan;
}

async function toggleDevice(deviceId: string, enabled: boolean) {
  const apiTarget = target();
  if (!apiTarget) return;
  mutatingDeviceId.value = deviceId;
  try {
    devices.value = (await setDeviceEnabled(apiTarget, deviceId, enabled)).devices;
    await loadDevices();
  } catch (error) {
    toasts.push(`Device state was not changed: ${String(error)}`, "error");
  } finally {
    mutatingDeviceId.value = null;
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

onBeforeUnmount(() => deviceEventsAbort?.abort());

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
      v-if="!filter && devices.length"
      class="mb-5"
      :devices="devices"
      :plan="plan"
      mutable
      :busy-device-id="mutatingDeviceId"
      @unpin="unpinWork"
      @toggle="toggleDevice"
    />
    <ConfigSettingRow schema-key="server_port" />
    <DeviceSettingsPanel v-if="!filter" class="mt-5" />
    <PlacementSection v-if="!filter" class="mt-5" />
    <p class="mt-4 mb-1 text-caption text-ink-3">
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
