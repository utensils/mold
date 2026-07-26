<script setup lang="ts">
import { computed, ref } from "vue";
import { setDeviceEnabled, type DeviceInfo } from "@studio/api/devices";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";

const hosts = useHostsStore();
const toasts = useToastStore();
const mutations = ref(new Set<string>());
const localHost = computed(() => hosts.all.find((host) => host.kind === "local") ?? null);
const devices = computed(() => {
  const host = localHost.value;
  return host ? (hosts.telemetry[host.id]?.devices ?? null) : null;
});
const capabilities = computed(() => {
  const host = localHost.value;
  return host ? (hosts.capabilities[host.id] ?? null) : null;
});

function canMutate(device: DeviceInfo): boolean {
  if (device.admin_state === "startup_excluded") return false;
  if (capabilities.value?.devices?.lifecycle && capabilities.value?.dispatch?.v2_authoritative)
    return true;
  return !device.desired_enabled && capabilities.value?.devices?.restart_enable === true;
}

function actionLabel(device: DeviceInfo): string {
  if (
    !device.desired_enabled &&
    !capabilities.value?.devices?.lifecycle &&
    capabilities.value?.devices?.restart_enable
  )
    return "Enable on restart";
  return device.desired_enabled ? "Disable" : "Enable";
}

function stateLabel(device: DeviceInfo): string {
  if (device.admin_state === "draining") return "Finishing current work";
  if (device.admin_state === "starting") return "Starting";
  if (device.admin_state === "startup_excluded") return "Excluded at startup";
  if (device.health !== "healthy") return device.health;
  return device.admin_state;
}

async function toggle(device: DeviceInfo) {
  const host = localHost.value;
  if (!host?.baseUrl || !canMutate(device)) return;
  const enabled = !device.desired_enabled;
  mutations.value = new Set(mutations.value).add(device.id);
  try {
    await setDeviceEnabled(
      { baseUrl: host.baseUrl, apiKey: host.apiKey ?? null },
      device.id,
      enabled,
    );
    await hosts.refresh();
  } catch (error) {
    toasts.push(`Could not update ${device.name}: ${String(error)}`, "error");
  } finally {
    const next = new Set(mutations.value);
    next.delete(device.id);
    mutations.value = next;
  }
}
</script>

<template>
  <section v-if="devices !== null" class="mb-5" data-test="settings-device-controls">
    <h3 class="mb-1 text-body font-semibold text-ink">GPU devices</h3>
    <p class="mb-2 text-caption text-ink-3">
      <template v-if="capabilities?.devices?.lifecycle">
        Disabling a busy GPU lets its current stage finish, then removes it from scheduling.
      </template>
      <template v-else>
        <template v-if="capabilities?.devices?.restart_enable">
          Live GPU controls require Scheduler V2. Disabled GPUs can be enabled for the next server
          restart.
        </template>
        <template v-else> Live GPU controls are unavailable on this server. </template>
      </template>
    </p>
    <div class="divide-edge divide-y rounded-control border border-edge">
      <div
        v-for="device in devices"
        :key="device.id"
        class="flex min-h-11 items-center gap-3 px-3 py-2"
        data-test="device-row"
      >
        <div class="min-w-0 flex-1">
          <div class="truncate text-body text-ink">{{ device.name }}</div>
          <div class="edge-code mt-0.5">
            {{ device.ordinal == null ? device.backend : `GPU ${device.ordinal}` }}
            · {{ stateLabel(device) }}
          </div>
        </div>
        <button
          type="button"
          class="rounded-control border border-edge px-3 py-1.5 text-caption font-semibold text-ink disabled:opacity-40"
          :data-test="`settings-device-toggle-${device.ordinal ?? device.id}`"
          :disabled="!canMutate(device) || mutations.has(device.id)"
          @click="toggle(device)"
        >
          {{ actionLabel(device) }}
        </button>
      </div>
    </div>
  </section>
</template>
