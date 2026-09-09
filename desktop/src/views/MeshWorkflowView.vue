<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useRoute } from "vue-router";
import MeshWorkflowStudio from "@studio/components/MeshWorkflowStudio.vue";
import { useMeshWorkflowDraftStore } from "@studio/stores/meshWorkflowDraft";
import {
  meshWorkflowHostFromQuery,
  meshWorkflowIdFromQuery,
} from "@studio/lib/meshWorkflowProvenance";
import { meshWorkflowModes, type WorkflowModel } from "@studio/lib/meshWorkflowAuthoring";
import {
  supportsMeshWorkflow,
  type MeshWorkflowRequirements,
  type MeshWorkflowRoute,
} from "@studio/lib/meshWorkflowRouting";
import ModelPicker from "../components/create/ModelPicker.vue";
import PanelResizeHandle from "../components/shell/PanelResizeHandle.vue";
import { useAppPrefsStore } from "../stores/appPrefs";
import { dragWidth } from "../lib/panelResize";
import type { ModelEntry } from "../lib/api/types";
import HostChip from "../components/create/HostChip.vue";
import { useHostsStore } from "../stores/hosts";
import { useHostModelsStore } from "../stores/hostModels";
import { useUiStore } from "../stores/ui";

const hosts = useHostsStore();
const prefs = useAppPrefsStore();
const route = useRoute();
const ui = useUiStore();
const studio = ref<{ generate: () => void } | null>(null);
// The queue row's route back here (`?workflow=`), read by the shell because
// routing is the shell's job — the shared studio is handed the id.
const openWorkflow = computed(() => meshWorkflowIdFromQuery(route.query));

/*
 * ⌘↩ used to raise the Generate intent AND push `/create`, so pressing it
 * here left the view and rendered a picture — while the status bar advertised
 * the hint. The shell now stays put on this route and this view consumes the
 * intent, exactly as New image does.
 */
watch(
  () => ui.generateTick,
  () => {
    if (ui.consumeIntent("generate")) studio.value?.generate();
  },
);
const draftWidth = ref<number | null>(null);
const inspectorWidth = computed(() => draftWidth.value ?? prefs.generateParamsWidth);
function resizeInspector(dx: number) {
  draftWidth.value = dragWidth("generateParams", prefs.generateParamsWidth, dx, "left");
}
async function commitInspector() {
  if (draftWidth.value !== null) await prefs.update({ generateParamsWidth: draftWidth.value });
  draftWidth.value = null;
}
function resetInspector() {
  draftWidth.value = null;
  void prefs.update({ generateParamsWidth: null });
}
function pickerModels(filtered: WorkflowModel[]): ModelEntry[] {
  const names = new Set(filtered.map((model) => model.name));
  const rows = new Map<string, ModelEntry>();
  for (const snapshot of Object.values(inventory.byHost))
    for (const model of snapshot.entries) if (names.has(model.name)) rows.set(model.name, model);
  return [...rows.values()];
}
const inventory = useHostModelsStore();
/*
 * The machine pin and the machine being browsed belong to the draft, not to
 * this mount: leaving for the Queue and coming back used to drop the workflow
 * back onto whichever host answered first. `HostChip` writes `routing`, and
 * the watcher below still follows it to the browsing host.
 */
const draft = useMeshWorkflowDraftStore();
const { routing, browseHostId } = storeToRefs(draft);
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
/*
 * A durable workflow lives on ONE machine, so the link that opens it names
 * that machine too. Without this a queue row from another host asked whichever
 * machine the studio happened to be browsing and was told the workflow does
 * not exist. A link that does not say (an older one) leaves the pin alone.
 */
watch(
  [() => meshWorkflowHostFromQuery(route.query), () => hosts.all.map((h) => h.id).join("|")],
  ([hostId]) => {
    if (!hostId || browseHostId.value === hostId) return;
    // The machine may not be known yet: `extras` fills in asynchronously and
    // the file's own comment below says connections become ready AFTER this
    // view mounts on a cold launch. The query never changes, so a query-only
    // watcher would drop the pin here and never look again — which is why the
    // host list is a source too.
    if (!hosts.all.some((host) => host.id === hostId)) return;
    browseHostId.value = hostId;
    routing.value = hostId;
    draft.persist();
  },
  { immediate: true },
);

watch(routing, (value) => {
  if (value && value !== "capable") browseHostId.value = value;
  draft.persist();
});
// Connections become ready after the view mounts on a cold launch. Refresh
// only when that authority set changes, never on queue/GPU telemetry ticks.
watch(
  () =>
    JSON.stringify(
      hosts.all
        .filter((host) => host.status === "ready" && !host.stale)
        .map((host) => [host.id, host.baseUrl, host.apiKey]),
    ),
  () => void inventory.refresh(),
  { immediate: true },
);

const availableModels = computed(() => {
  const byName = new Map<string, WorkflowModel>();
  for (const host of hosts.all) {
    if (routing.value && routing.value !== "capable" && host.id !== routing.value) continue;
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
    ref="studio"
    :target="target"
    :open-workflow="openWorkflow"
    :style="{ '--mesh-inspector-width': inspectorWidth + 'px' }"
    :available-models="availableModels"
    :resolve-target="resolveTarget"
    :host-label="selectedHost?.label ?? ''"
    desktop
  >
    <template #inspector-resize>
      <PanelResizeHandle
        class="mesh-inspector-resize"
        :style="{ right: inspectorWidth - 2 + 'px' }"
        label="Resize 3-D settings"
        @resize="resizeInspector"
        @commit="commitInspector"
        @reset="resetInspector"
      />
    </template>
    <template #mesh-picker="{ models, selected, select, disabled }">
      <div class="mesh-style-field">
        <span class="ms-group-label">3-D style</span>
        <ModelPicker
          :models="pickerModels(models)"
          :selected="pickerModels(models).find((m) => m.name === selected) ?? null"
          :disabled-reason="disabled ? () => 'Preparing workflow' : null"
          kicker="3-D object styles"
          browse-target="/models?type=mesh"
          @pick="(model) => select(model.name)"
        />
      </div>
    </template>
    <template #image-picker="{ models, selected, select, disabled }">
      <div class="mesh-style-field">
        <span class="ms-group-label">Picture style</span>
        <ModelPicker
          :models="pickerModels(models)"
          :selected="pickerModels(models).find((m) => m.name === selected) ?? null"
          :disabled-reason="disabled ? () => 'Preparing workflow' : null"
          kicker="Still picture styles"
          browse-target="/models?type=image"
          @pick="(model) => select(model.name)"
        />
      </div>
    </template>
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
.mesh-inspector-resize {
  position: absolute;
  top: 0;
  bottom: 0;
  z-index: 10;
}
.mesh-style-field {
  display: grid;
  gap: 8px;
  min-width: 0;
}

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
