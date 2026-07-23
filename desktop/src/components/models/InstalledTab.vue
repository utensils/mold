<script setup lang="ts">
import { computed, ref } from "vue";
import { useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import CatalogDetailDrawer from "./CatalogDetailDrawer.vue";
import ModelTableRow from "./ModelTableRow.vue";
import { isVideoFamily } from "../../lib/capabilities";
import { installedModelToEntry } from "../../lib/catalogDetail";
import {
  groupInstalledModels,
  isUtilityModel,
  modelDiskBytes,
  modelDisplayName,
  modelSizeLabels,
  quantTag,
} from "../../lib/models";
import { modelSource } from "../../lib/modelSource";
import { loadModel, removeModel, unloadModel } from "../../lib/api/models";
import { startCatalogDownload } from "../../lib/api/catalog";
import { ApiError } from "../../lib/api/client";
import { formatGB, percent } from "../../lib/format";
import { type MediaType } from "../../lib/modelAvailability";
import type { ModelEntry } from "../../lib/api/types";

type LibraryModelEntry = ModelEntry & { hostIds?: string[] };

const props = defineProps<{
  query?: string | undefined;
  mediaType?: MediaType | undefined;
  entries?: LibraryModelEntry[] | undefined;
}>();
const emit = defineEmits<{ (e: "browse-catalog"): void }>();

const models = useModelStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const toasts = useToastStore();
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
  // Utility rows aren't image or video generators — they only show under All.
  return searched.filter(
    (m) => !isUtilityModel(m) && isVideoFamily(m.family) === (type === "video"),
  );
});
const groups = computed(() => groupInstalledModels(filtered.value));

/** Family groups plus a trailing utility section, as [heading, models] rows. */
const sections = computed<[string, ModelEntry[]][]>(() => [
  ...groups.value.families,
  ...((props.mediaType ?? "all") === "all"
    ? [["SHARED / UTILITY", groups.value.utility] as [string, ModelEntry[]]]
    : []),
]);

// Info opens the shared model detail drawer (same layout as the catalog).
const detailModel = ref<LibraryModelEntry | null>(null);
const busy = ref<string | null>(null);
const confirmingRemove = ref<string | null>(null);

function targetHost(m: LibraryModelEntry) {
  const ids = m.hostIds ?? ["local"];
  const preferred = ids.includes("local") ? "local" : ids[0];
  return hosts.all.find((host) => host.id === preferred) ?? null;
}

function targetFor(m: LibraryModelEntry) {
  const target = targetHost(m);
  return target && !target.primary && target.baseUrl
    ? { baseUrl: target.baseUrl, apiKey: target.apiKey }
    : undefined;
}

function hostLabels(m: LibraryModelEntry): string[] {
  return (m.hostIds ?? ["local"]).map(
    (id) => hosts.all.find((host) => host.id === id)?.label ?? id,
  );
}

async function refreshAfterAction(m: LibraryModelEntry) {
  if (!targetFor(m)) await models.fetch();
  await hostModels.refresh(true);
}

async function remove(m: LibraryModelEntry) {
  if (confirmingRemove.value !== m.name) {
    confirmingRemove.value = m.name;
    return;
  }
  confirmingRemove.value = null;
  busy.value = m.name;
  try {
    const result = await removeModel(m.name, targetFor(m));
    const kept = result.kept.length;
    toasts.push(
      kept > 0
        ? `Removed ${modelDisplayName(m)} — freed ${formatGB(result.freed_bytes)}, kept ${kept} shared component${kept === 1 ? "" : "s"}`
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
  }
}

/** Whole-model repair from the drawer: re-fetches only missing files. */
const drawerRepairing = ref(false);

async function repairFromDrawer(m: LibraryModelEntry) {
  drawerRepairing.value = true;
  try {
    await startCatalogDownload(m.name, targetFor(m), !!targetFor(m));
    toasts.push(`Repairing ${modelDisplayName(m)}`);
  } catch (err) {
    toasts.push(
      err instanceof ApiError && err.status === 409
        ? `${modelDisplayName(m)} is already queued.`
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
    <p class="text-body text-ink-2">
      <template v-if="(query ?? '').trim()">Nothing installed matches "{{ query }}".</template>
      <template v-else>Nothing on the shelf yet.</template>
    </p>
    <button
      v-if="!(query ?? '').trim()"
      type="button"
      class="mt-2 text-body text-halide hover:text-ink"
      @click="emit('browse-catalog')"
    >
      Browse the catalog
    </button>
  </div>

  <div v-else class="flex flex-col gap-4 p-4">
    <template v-for="[heading, list] in sections" :key="heading">
      <section v-if="list.length">
        <div class="mb-2 flex items-center gap-2">
          <span class="edge-code">{{ heading.toUpperCase() }}</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>

        <ul
          class="border-edge divide-edge divide-y overflow-hidden rounded-control border bg-bench"
        >
          <li v-for="m in list" :key="m.name">
            <ModelTableRow
              class="px-3 py-2"
              :name="modelDisplayName(m)"
              :source="modelSource(m)"
              :loaded="m.is_loaded"
              :host-labels="hostLabels(m)"
              :quant="quantTag(m.name)"
              :page-url="m.hf_repo ? `https://huggingface.co/${m.hf_repo}` : null"
              :size-primary="
                modelSizeLabels(m).weights ?? modelSizeLabels(m).runtime ?? 'Size unavailable'
              "
              :size-secondary="
                modelSizeLabels(m).weights && modelSizeLabels(m).runtime
                  ? modelSizeLabels(m).runtime
                  : null
              "
              :bar-percent="percent(modelDiskBytes(m), groups.maxDiskBytes)"
              clickable
              @open="detailModel = m"
            >
              <template #actions>
                <button
                  v-if="!m.is_loaded"
                  type="button"
                  class="border-edge h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-ink active:translate-y-px disabled:opacity-40"
                  :disabled="busy === m.name"
                  @click="load(m)"
                >
                  Load
                </button>
                <button
                  v-else
                  type="button"
                  class="border-edge h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-ink active:translate-y-px disabled:opacity-40"
                  :disabled="busy === m.name"
                  @click="unload(m)"
                >
                  Unload
                </button>
                <button
                  type="button"
                  class="h-7 rounded-control border px-2 text-caption transition-colors duration-100 active:translate-y-px"
                  :class="
                    confirmingRemove === m.name
                      ? 'border-stop bg-stop font-semibold text-on-accent'
                      : 'border-edge text-ink-2 hover:text-stop'
                  "
                  :disabled="busy === m.name"
                  @blur="confirmingRemove = null"
                  @click="remove(m)"
                >
                  {{ confirmingRemove === m.name ? "Remove from disk?" : "✕" }}
                </button>
              </template>
            </ModelTableRow>
          </li>
        </ul>
      </section>
    </template>
  </div>

  <!-- One consistent model-detail drawer, shared with the catalog. -->
  <CatalogDetailDrawer
    v-if="detailModel"
    :entry="installedModelToEntry(detailModel)"
    :pulling="drawerRepairing"
    :target="targetFor(detailModel)"
    :forward-credentials="!!targetFor(detailModel)"
    @close="detailModel = null"
    @pull="detailModel && repairFromDrawer(detailModel)"
  />
</template>
