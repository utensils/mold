<script setup lang="ts">
import { computed, ref, watch } from "vue";
import MeshWorkflowStudio from "@studio/components/MeshWorkflowStudio.vue";
import MeshWorkflowHostPicker from "@studio/components/MeshWorkflowHostPicker.vue";
import { useHostRouting } from "../composables/useHostRouting";
import { AUTO_TARGET_ID, CAPABLE_TARGET_ID } from "../lib/hostRouting";
import { ORIGIN_HOST_ID } from "../lib/hostRegistry";

const routing = useHostRouting();
const selectedHostId = ref("");
const selectedHost = computed(
  () =>
    routing.hosts.value.find((host) => host.id === selectedHostId.value) ??
    null,
);
const target = computed(() => {
  const host = selectedHost.value;
  if (!host || (host.status !== "ready" && !host.stale)) return null;
  return { baseUrl: host.url, apiKey: host.apiKey ?? null };
});

watch(
  () => routing.hosts.value.map((host) => host.id).join("|"),
  () => {
    if (selectedHost.value) return;
    const preferred = routing.targetId.value;
    selectedHostId.value =
      preferred !== AUTO_TARGET_ID &&
      preferred !== CAPABLE_TARGET_ID &&
      routing.hosts.value.some((host) => host.id === preferred)
        ? preferred
        : (routing.hosts.value.find((host) => host.id === ORIGIN_HOST_ID)?.id ??
          routing.hosts.value[0]?.id ??
          "");
  },
  { immediate: true },
);

function selectHost(id: string): void {
  selectedHostId.value = id;
  routing.setTarget(id);
}

function hostStatus(): string {
  const host = selectedHost.value;
  if (!host) return "The selected machine is no longer connected.";
  if (host.stale || host.status === "connecting")
    return `${host.label} is reconnecting. Its workflows stay on that machine.`;
  return `${host.label} is unavailable. Its workflows stay on that machine.`;
}
</script>

<template>
  <MeshWorkflowStudio
    v-if="target"
    :key="`${selectedHostId}:${target.baseUrl}:${target.apiKey ?? ''}`"
    :target="target"
  >
    <template #machine>
      <MeshWorkflowHostPicker
        :model-value="selectedHostId"
        :hosts="routing.hosts.value"
        @update:model-value="selectHost"
      />
    </template>
  </MeshWorkflowStudio>
  <div v-else class="mesh-workflow-unavailable">
    <MeshWorkflowHostPicker
      :model-value="selectedHostId"
      :hosts="routing.hosts.value"
      @update:model-value="selectHost"
    />
    <p>{{ hostStatus() }}</p>
  </div>
</template>

<style scoped>
.mesh-workflow-unavailable {
  display: grid;
  place-content: center;
  gap: 14px;
  height: 100%;
  color: var(--mold-text-2);
}
.mesh-workflow-unavailable p {
  margin: 0;
}
</style>
