<script setup lang="ts">
import { computed, ref, watch } from "vue";
import MeshWorkflowStudio from "@studio/components/MeshWorkflowStudio.vue";
import MeshWorkflowHostPicker from "@studio/components/MeshWorkflowHostPicker.vue";
import { useHostsStore } from "../stores/hosts";

const hosts = useHostsStore();
const selectedHostId = ref("");
const selectedHost = computed(
  () => hosts.all.find((host) => host.id === selectedHostId.value) ?? null,
);
const target = computed(() => {
  const host = selectedHost.value;
  if (!host?.baseUrl || (host.status !== "ready" && !host.stale)) return null;
  return { baseUrl: host.baseUrl, apiKey: host.apiKey };
});

watch(
  () => hosts.all.map((host) => host.id).join("|"),
  () => {
    if (selectedHost.value) return;
    selectedHostId.value = hosts.primaryHost?.id ?? hosts.all[0]?.id ?? "";
  },
  { immediate: true },
);

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
      <MeshWorkflowHostPicker v-model="selectedHostId" :hosts="hosts.all" />
    </template>
  </MeshWorkflowStudio>
  <div v-else class="mesh-workflow-unavailable text-fg-dim">
    <MeshWorkflowHostPicker v-model="selectedHostId" :hosts="hosts.all" />
    <p>{{ hostStatus() }}</p>
  </div>
</template>

<style scoped>
.mesh-workflow-unavailable {
  display: grid;
  place-content: center;
  gap: 14px;
  height: 100%;
}
.mesh-workflow-unavailable p {
  margin: 0;
}
</style>
