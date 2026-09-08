<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import MeshWorkflowStudio from "@studio/components/MeshWorkflowStudio.vue";
import { meshWorkflowModes, type WorkflowModel } from "@studio/lib/meshWorkflowAuthoring";
import {
  supportsMeshWorkflow,
  type MeshWorkflowRequirements,
  type MeshWorkflowRoute,
} from "@studio/lib/meshWorkflowRouting";
import HostChip from "../components/create/HostChip.vue";
import { useHostsStore } from "../stores/hosts";
import { useHostModelsStore } from "../stores/hostModels";

const hosts = useHostsStore();
const inventory = useHostModelsStore();
const routing = ref<string | null>(null);
const browseHostId = ref("");
const selectedHost = computed(
  () => hosts.all.find((host) => host.id === browseHostId.value) ?? null,
);
const target = computed(() => {
  const host = selectedHost.value;
  return host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : null;
});
// Telemetry never changes the browsing machine or remounts the draft.
watch(
  () => hosts.all.map((host) => host.id).join("|"),
  () => {
    if (!selectedHost.value) browseHostId.value = hosts.primaryHost?.id ?? hosts.all[0]?.id ?? "";
  },
  { immediate: true },
);
watch(routing, (value) => {
  if (value && value !== "capable") browseHostId.value = value;
});
onMounted(() => void inventory.refresh());

const availableModels = computed(() => {
  const byName = new Map<string, WorkflowModel>();
  for (const host of hosts.all) {
    if (routing.value && routing.value !== "capable" && host.id !== routing.value) continue;
    if (host.status !== "ready" || host.stale) continue;
    for (const model of inventory.byHost[host.id]?.entries ?? []) {
      if (!model.downloaded || model.runtime_available === false) continue;
      const existing = byName.get(model.name);
      if (!existing || meshWorkflowModes(model).length > meshWorkflowModes(existing).length)
        byName.set(model.name, model);
    }
  }
  return [...byName.values()];
});

async function resolveTarget(requirements: MeshWorkflowRequirements): Promise<MeshWorkflowRoute> {
  await inventory.refresh(true);
  const eligible = hosts.all
    .filter(
      (host) =>
        host.status === "ready" &&
        !host.stale &&
        !inventory.byHost[host.id]?.error &&
        supportsMeshWorkflow(inventory.byHost[host.id]?.entries ?? [], requirements),
    )
    .map((host) => host.id);
  const route = hosts.resolveRoute(routing.value, requirements.meshModel, eligible);
  if (!route) {
    const label =
      routing.value && routing.value !== "capable"
        ? hosts.all.find((host) => host.id === routing.value)?.label
        : null;
    throw new Error(
      label
        ? `${label} cannot run all the selected 3-D stages. Connect it and check its installed styles, or choose Auto.`
        : "No ready machine can run all the selected 3-D stages. Choose styles installed together on one machine.",
    );
  }
  return { target: route.target, label: route.label };
}
</script>

<template>
  <MeshWorkflowStudio
    v-if="target"
    :target="target"
    :available-models="availableModels"
    :resolve-target="resolveTarget"
    :host-label="selectedHost?.label ?? ''"
    desktop
  >
    <template #machine="{ busy }">
      <HostChip v-model="routing" :disabled="busy" always-show-routing />
    </template>
  </MeshWorkflowStudio>
  <div v-else class="mesh-workflow-unavailable text-fg-dim">
    <HostChip v-model="routing" always-show-routing />
    <p>Connect a machine to make a 3-D object.</p>
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
