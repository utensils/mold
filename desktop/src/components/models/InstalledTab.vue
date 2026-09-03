<script setup lang="ts">
/*
 * Ready to use — every style on any connected machine, grouped by family
 * under a mono heading (README §04 table). A disk meter opens the shelf when
 * this machine reports its models disk; the ⋯ at the end of a row holds the
 * page link and the one destructive action behind a plain confirm.
 */
import { computed, ref } from "vue";
import { planModelInstall } from "@studio/lib/modelInstallTargets";
import { useModelStore } from "../../stores/models";
import { useInventoryKnown } from "../../lib/modelInventory";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostStatusStore } from "../../stores/hostStatus";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import Icon from "@ui/components/Icon.vue";
import ConfirmDialog from "../shell/ConfirmDialog.vue";
import CatalogDetailDrawer from "./CatalogDetailDrawer.vue";
import DownloadTargetDialog from "./DownloadTargetDialog.vue";
import ModelTableRow from "./ModelTableRow.vue";
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import { modelKindLabel, modelKindValue } from "@studio/lib/modelMetadata";
import {
  isModelRuntimeUnavailable,
  modelRuntimeNotice,
  RUNTIME_UNAVAILABLE_BADGE,
} from "@studio/lib/modelRuntimeAvailability";
import { isVideoFamily } from "../../lib/capabilities";
import { installedModelToEntry } from "../../lib/catalogDetail";
import {
  groupInstalledModels,
  isUtilityModel,
  modelDiskBytes,
  modelDisplayName,
  modelSizeLabels,
} from "../../lib/models";
import { modelSource } from "../../lib/modelSource";
import { openExternal } from "../../lib/openExternal";
import { loadModel, removeModel, unloadModel } from "../../lib/api/models";
import { startCatalogDownload } from "../../lib/api/catalog";
import { ApiError } from "../../lib/api/client";
import { formatGB, percent } from "../../lib/format";
import { type MediaType } from "../../lib/modelAvailability";
import type { ModelEntry } from "../../lib/api/types";
import type { HostView } from "../../stores/hosts";

type LibraryModelEntry = ModelEntry & { hostIds?: string[] };

const props = defineProps<{
  query?: string | undefined;
  mediaType?: MediaType | undefined;
  entries?: LibraryModelEntry[] | undefined;
}>();
const emit = defineEmits<{ (e: "browse-catalog"): void }>();

const models = useModelStore();
const hostModels = useHostModelsStore();
const hostStatus = useHostStatusStore();
const hosts = useHostsStore();
const toasts = useToastStore();
const contextMenu = useContextMenuStore();
const inventoryKnown = useInventoryKnown();
const sourceEntries = computed(() => props.entries ?? models.installed);

const filtered = computed(() => {
  const q = (props.query ?? "").trim().toLowerCase();
  const searched = q
    ? sourceEntries.value.filter(
        (m) => m.name.toLowerCase().includes(q) || modelDisplayName(m).toLowerCase().includes(q),
      )
    : sourceEntries.value;
  const type = props.mediaType ?? "all";
  if (type === "all") return searched;
  // Utility rows aren't picture or clip makers — they only show under All.
  return searched.filter(
    (m) => !isUtilityModel(m) && isVideoFamily(m.family) === (type === "video"),
  );
});
const groups = computed(() => groupInstalledModels(filtered.value));

/** Family groups plus a trailing helper section, as [heading, models] rows. */
const sections = computed<[string, ModelEntry[]][]>(() =>
  [
    ...groups.value.families,
    ...((props.mediaType ?? "all") === "all"
      ? [["SHARED / UTILITY", groups.value.utility] as [string, ModelEntry[]]]
      : []),
  ].filter(([, list]) => list.length > 0),
);

/**
 * Disk used by styles on THIS machine (the primary's /api/status is the only
 * disk the shell reads): one segment per family, sized by each style's
 * primary weights. Weights are unique per style, so a T5 encoder three
 * styles share is never counted three times; the tooltip says so.
 */
const SEGMENT_TONES = ["bg-accent", "bg-sapphire", "bg-mauve", "bg-teal", "bg-lavender"];
const disk = computed(() => hostStatus.status?.models_disk ?? null);
const weightsBytes = (m: ModelEntry) => Math.max(0, m.size_gb) * 1_000_000_000;
const diskSegments = computed(() =>
  sections.value
    .map(([heading, list], index) => ({
      heading,
      bytes: list
        .filter((m) => ((m as LibraryModelEntry).hostIds ?? ["local"]).includes("local"))
        .reduce((sum, m) => sum + weightsBytes(m), 0),
      tone: SEGMENT_TONES[index % SEGMENT_TONES.length]!,
    }))
    .filter((segment) => segment.bytes > 0),
);
const stylesBytes = computed(() =>
  diskSegments.value.reduce((sum, segment) => sum + segment.bytes, 0),
);

// A row opens the shared model detail drawer (same layout as Browse more).
const detailModel = ref<LibraryModelEntry | null>(null);
const pendingRepair = ref<LibraryModelEntry | null>(null);
const busy = ref<string | null>(null);
const removing = ref<LibraryModelEntry | null>(null);

function targetHost(m: LibraryModelEntry) {
  const ids = m.hostIds ?? ["local"];
  const preferred = ids.includes("local") ? "local" : ids[0];
  return hosts.all.find((host) => host.id === preferred) ?? null;
}

function targetFor(m: LibraryModelEntry) {
  return targetForHost(targetHost(m));
}

function targetForHost(target: HostView | null) {
  return target && !target.primary && target.baseUrl
    ? { baseUrl: target.baseUrl, apiKey: target.apiKey }
    : undefined;
}

const readyHosts = computed(() =>
  hosts.all.filter((host) => host.status === "ready" && Boolean(host.baseUrl)),
);

/**
 * A row here is merged across machines, so "ready" never means "ready
 * everywhere". Every ready machine that lacks the style stays an install
 * target; only the owners degrade to repair.
 */
function installPlan(m: LibraryModelEntry) {
  return planModelInstall(readyHosts.value, m.hostIds ?? ["local"], { inventoryKnown });
}

function hostLabels(m: LibraryModelEntry): string[] {
  return (m.hostIds ?? ["local"]).map(
    (id) => hosts.all.find((host) => host.id === id)?.label ?? id,
  );
}

/** `runtime_available: false` (download-only rows such as the NVFP4 H3
 * partitions) means the server rejects every load/generate attempt with a
 * 501 — hide Load/Unload for the row and say why instead of a toast. Get it,
 * Repair, and Remove stay reachable. The obstacle itself is the server's to
 * name (#1276); the row repeats its sentence rather than guessing. */
function runtimeUnavailable(m: LibraryModelEntry): boolean {
  return isModelRuntimeUnavailable(m);
}

function runtimeUnavailableTitle(m: LibraryModelEntry): string | undefined {
  return modelRuntimeNotice(m)?.message;
}

function modelAccessibilityLabel(m: LibraryModelEntry): string {
  const kind = modelKindLabel(modelKindValue(m));
  return `${modelDisplayName(m)} — ${kind}${m.nsfw ? ", 18+ NSFW" : ""} — view details`;
}

function pageUrl(m: LibraryModelEntry): string | null {
  return m.hf_repo ? `https://huggingface.co/${m.hf_repo}` : null;
}

/** The ⋯ at the end of a row: the page link and the destructive action. */
function openRowMenu(event: MouseEvent, m: LibraryModelEntry) {
  const url = pageUrl(m);
  const entries: MenuEntry[] = [
    ...(url ? [{ label: "Open model page", action: () => void openExternal(url) }] : []),
    { label: "Remove from disk…", danger: true, action: () => (removing.value = m) },
  ];
  contextMenu.open(event, entries);
}

async function refreshAfterAction(m: LibraryModelEntry) {
  if (!targetFor(m)) await models.fetch();
  await hostModels.refresh(true);
}

async function remove(m: LibraryModelEntry) {
  busy.value = m.name;
  try {
    const result = await removeModel(m.name, targetFor(m));
    const kept = result.kept.length;
    toasts.push(
      kept > 0
        ? `Removed ${modelDisplayName(m)} — freed ${formatGB(result.freed_bytes)}, kept ${kept} shared helper${kept === 1 ? "" : "s"}`
        : `Removed ${modelDisplayName(m)} — freed ${formatGB(result.freed_bytes)}`,
    );
    await refreshAfterAction(m);
  } catch (err) {
    // 409 MODEL_LOADED → tell the user the one concrete fix.
    toasts.push(
      err instanceof ApiError && err.status === 409
        ? `${modelDisplayName(m)} is on the GPU. Unload it first.`
        : String(err),
      "error",
    );
  } finally {
    busy.value = null;
    removing.value = null;
  }
}

/** Whole-model install/repair from the drawer or the row's Get it action. */
const drawerRepairing = ref(false);

function requestDownload(m: LibraryModelEntry) {
  const candidates = installPlan(m).targets;
  if (candidates.length === 0) {
    toasts.push("No online machine is available for this style.", "error");
    return;
  }
  if (candidates.length > 1) {
    pendingRepair.value = m;
    return;
  }
  void downloadOnHost(m, candidates[0]!.host);
}

async function downloadOnHost(m: LibraryModelEntry, host: HostView | null) {
  pendingRepair.value = null;
  drawerRepairing.value = true;
  const owns = (m.hostIds ?? ["local"]).includes(host?.id ?? "local");
  try {
    const target = targetForHost(host);
    await startCatalogDownload(m.name, target, !!target);
    toasts.push(
      `${owns ? "Repairing" : "Getting"} ${modelDisplayName(m)}${host ? ` on ${host.label}` : ""}`,
    );
  } catch (err) {
    toasts.push(
      err instanceof ApiError && err.status === 409
        ? `${modelDisplayName(m)} is already on its way.`
        : String(err),
      "error",
    );
  } finally {
    drawerRepairing.value = false;
  }
}

async function load(m: LibraryModelEntry) {
  busy.value = m.name;
  try {
    await loadModel(m.name, targetFor(m));
    await refreshAfterAction(m);
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    busy.value = null;
  }
}

async function unload(m: LibraryModelEntry) {
  busy.value = m.name;
  try {
    await unloadModel(m.name, targetFor(m));
    await refreshAfterAction(m);
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    busy.value = null;
  }
}
</script>

<template>
  <div v-if="filtered.length === 0" class="p-8 text-center">
    <p class="text-sm text-fg-2">
      <template v-if="(query ?? '').trim()">No ready style matches "{{ query }}".</template>
      <template v-else>No styles ready yet.</template>
    </p>
    <button
      v-if="!(query ?? '').trim()"
      type="button"
      class="mt-2 text-sm font-semibold text-accent hover:text-fg"
      @click="emit('browse-catalog')"
    >
      Browse more styles
    </button>
  </div>

  <div v-else class="flex flex-col gap-4 p-3.5">
    <div v-if="disk" class="flex items-center gap-2.5 px-3" data-test="styles-disk-meter">
      <span class="ms-group-label uppercase">Disk used by styles</span>
      <span
        class="flex h-2 flex-1 overflow-hidden border border-border"
        role="meter"
        aria-label="Disk used by styles on this machine"
        :aria-valuenow="Math.round(percent(stylesBytes, disk.total_bytes))"
        aria-valuemin="0"
        aria-valuemax="100"
        title="Each style's own weights. Helpers that several styles share are counted once."
      >
        <span
          v-for="segment in diskSegments"
          :key="segment.heading"
          :class="segment.tone"
          :style="{ width: `${percent(segment.bytes, disk.total_bytes)}%` }"
          :title="`${segment.heading} · ${formatGB(segment.bytes)}`"
        />
      </span>
      <span class="font-mono text-micro text-fg-dim">
        {{ formatGB(stylesBytes) }} of {{ formatGB(disk.total_bytes) }}
      </span>
    </div>

    <section v-for="[heading, list] in sections" :key="heading">
      <div class="mb-2 flex items-center gap-2.5 px-3">
        <span class="ms-group-label uppercase">{{ heading.toUpperCase() }}</span>
        <div class="h-px flex-1 border-t border-border" />
      </div>

      <ul class="divide-y divide-border border border-border bg-panel">
        <li v-for="m in list" :key="m.name">
          <ModelTableRow
            class="px-3"
            :name="modelDisplayName(m)"
            :id="m.name"
            :source="modelSource(m)"
            :loaded="m.is_loaded"
            :family="m.family"
            :host-labels="hostLabels(m)"
            :page-url="pageUrl(m)"
            :note="m.description || null"
            :size-primary="
              modelSizeLabels(m).weights ?? modelSizeLabels(m).runtime ?? 'Size unavailable'
            "
            :size-secondary="
              modelSizeLabels(m).weights && modelSizeLabels(m).runtime
                ? modelSizeLabels(m).runtime
                : null
            "
            :bar-percent="percent(modelDiskBytes(m), groups.maxDiskBytes)"
            :accessibility-label="modelAccessibilityLabel(m)"
            clickable
            @open="detailModel = m"
          >
            <template #meta>
              <ModelMetadataBadges
                :kind="m.kind ?? null"
                :family="m.family"
                :nsfw="m.nsfw ?? false"
                :show-modality="false"
              />
            </template>
            <template #actions>
              <!-- Ready on one machine says nothing about the others:
                   offer it until every ready machine has the style. -->
              <button
                v-if="installPlan(m).canInstall"
                type="button"
                data-test="install-elsewhere"
                class="ms-toolbar-button ms-toolbar-button--accent"
                title="Get this style on another machine"
                :disabled="busy === m.name"
                @click="requestDownload(m)"
              >
                Get it
              </button>
              <span
                v-if="runtimeUnavailable(m)"
                data-test="runtime-unavailable-note"
                class="flex h-[26px] items-center text-micro text-fg-dim"
                :title="runtimeUnavailableTitle(m)"
              >
                {{ RUNTIME_UNAVAILABLE_BADGE }}
              </span>
              <button
                v-else-if="!m.is_loaded"
                type="button"
                data-test="load-btn"
                class="ms-toolbar-button"
                :disabled="busy === m.name"
                @click="load(m)"
              >
                Load
              </button>
              <button
                v-else
                type="button"
                data-test="unload-btn"
                class="ms-toolbar-button"
                :disabled="busy === m.name"
                @click="unload(m)"
              >
                Unload
              </button>
              <button
                type="button"
                data-test="row-menu"
                class="inline-flex h-[26px] w-[26px] shrink-0 items-center justify-center rounded-control text-fg-dim transition-colors duration-100 hover:bg-surface-2 hover:text-fg"
                title="Open model page, remove from disk…"
                aria-label="Style actions"
                :disabled="busy === m.name"
                @click="openRowMenu($event, m)"
              >
                <Icon name="more" :size="14" />
              </button>
            </template>
          </ModelTableRow>
        </li>
      </ul>
    </section>
  </div>

  <!-- One consistent model-detail drawer, shared with Browse more. -->
  <CatalogDetailDrawer
    v-if="detailModel"
    :entry="installedModelToEntry(detailModel)"
    :pulling="drawerRepairing"
    :target="targetFor(detailModel)"
    :forward-credentials="!!targetFor(detailModel)"
    :action="installPlan(detailModel).label"
    :runtime-notice="modelRuntimeNotice(detailModel)"
    @close="detailModel = null"
    @pull="detailModel && requestDownload(detailModel)"
  />

  <DownloadTargetDialog
    v-if="pendingRepair"
    :model-name="modelDisplayName(pendingRepair)"
    :targets="installPlan(pendingRepair).targets"
    @close="pendingRepair = null"
    @select="(host) => pendingRepair && void downloadOnHost(pendingRepair, host)"
  />

  <ConfirmDialog
    :open="removing !== null"
    :title="removing ? `Remove ${modelDisplayName(removing)} from disk?` : ''"
    :message="
      removing
        ? `Frees ${modelSizeLabels(removing).weights ?? formatGB(modelDiskBytes(removing))}. Helpers other styles use stay. You can get it again from Browse more.`
        : ''
    "
    confirm-label="Remove"
    danger
    :busy="removing !== null && busy === removing.name"
    @confirm="removing && remove(removing)"
    @cancel="removing = null"
  />
</template>
