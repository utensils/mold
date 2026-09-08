<script setup lang="ts">
import { computed, ref, watch } from "vue";
import MeshWorkflowStudio from "@studio/components/MeshWorkflowStudio.vue";
import CreateModelPicker from "../components/create/CreateModelPicker.vue";
import {
  supportsMeshWorkflow,
  type MeshWorkflowRequirements,
} from "@studio/lib/meshWorkflowRouting";
import type { WorkflowModel } from "@studio/lib/meshWorkflowAuthoring";
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
  if (!host) return null;
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
  if (id !== AUTO_TARGET_ID && id !== CAPABLE_TARGET_ID)
    selectedHostId.value = id;
  routing.setTarget(id);
}

function pickerModels(filtered: WorkflowModel[]) {
  const names = new Set(filtered.map((model) => model.name));
  return routing.targetModels.value.filter((model) => names.has(model.name));
}
async function resolveTarget(requirements: MeshWorkflowRequirements) {
  await routing.refresh();
  const eligible = routing.hosts.value
    .filter(
      (host) =>
        host.status === "ready" &&
        !host.stale &&
        supportsMeshWorkflow(routing.modelsForHost(host.id), requirements),
    )
    .map((host) => host.id);
  const route = routing.resolve(requirements.meshModel, eligible);
  if (!route)
    throw new Error(
      "No selected machine can run all these 3-D stages. Choose styles installed together on one ready machine.",
    );
  return {
    target: { baseUrl: route.target.baseUrl, apiKey: route.target.apiKey ?? null },
    label: route.label,
  };
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
    :target="target"
    :available-models="routing.targetModels.value"
    :resolve-target="resolveTarget"
    :host-label="selectedHost?.label ?? ''"
  >
    <template #mesh-picker="{ models, selected, select }">
      <CreateModelPicker
        :models="pickerModels(models)"
        :model="selected"
        browse-to="/models?kind=mesh"
        @select="(model) => select(model.name)"
      />
    </template>
    <template #image-picker="{ models, selected, select }">
      <CreateModelPicker
        :models="pickerModels(models)"
        :model="selected"
        browse-to="/models?kind=image"
        @select="(model) => select(model.name)"
      />
    </template>
    <template #machine="{ busy }">
      <MeshWorkflowHostPicker
        :model-value="routing.targetId.value"
        automatic
        :hosts="routing.hosts.value"
        :disabled="busy"
        @update:model-value="selectHost"
      />
    </template>
  </MeshWorkflowStudio>
  <div v-else class="mesh-workflow-unavailable">
    <MeshWorkflowHostPicker
      :model-value="routing.targetId.value"
      automatic
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
