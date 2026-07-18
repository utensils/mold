<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import SourceGlyph from "../generate/SourceGlyph.vue";
import { isVideoFamily } from "../../lib/capabilities";
import {
  groupInstalledModels,
  isUtilityModel,
  modelDiskBytes,
  modelSizeLabels,
  quantTag,
} from "../../lib/models";
import { modelSource } from "../../lib/modelSource";
import { fetchModelComponents, loadModel, removeModel, unloadModel } from "../../lib/api/models";
import { startCatalogDownload } from "../../lib/api/catalog";
import { ApiError } from "../../lib/api/client";
import { formatGB, percent } from "../../lib/format";
import { openExternal } from "../../lib/openExternal";
import { type MediaType } from "../../lib/modelAvailability";
import type { ModelComponentStatus, ModelEntry } from "../../lib/api/types";

type LibraryModelEntry = ModelEntry & { hostIds?: string[] };

const props = defineProps<{
  query?: string;
  mediaType?: MediaType;
  entries?: LibraryModelEntry[];
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
    ? sourceEntries.value.filter((m) => m.name.toLowerCase().includes(q))
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

// Per-row Info expansion + lazily-loaded component lists.
const expanded = reactive<Record<string, boolean>>({});
const components = reactive<Record<string, ModelComponentStatus[] | "loading" | "error">>({});
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
        ? `Removed ${m.name} — freed ${formatGB(result.freed_bytes)}, kept ${kept} shared component${kept === 1 ? "" : "s"}`
        : `Removed ${m.name} — freed ${formatGB(result.freed_bytes)}`,
    );
    await refreshAfterAction(m);
  } catch (err) {
    // 409 MODEL_LOADED → tell the user the one concrete fix.
    toasts.push(
      err instanceof ApiError && err.status === 409
        ? `${m.name} is on the GPU. Unload it first.`
        : String(err),
      "error",
    );
  } finally {
    busy.value = null;
  }
}

async function toggleInfo(m: LibraryModelEntry) {
  expanded[m.name] = !expanded[m.name];
  if (expanded[m.name] && components[m.name] === undefined) {
    components[m.name] = "loading";
    try {
      components[m.name] = (await fetchModelComponents(m.name, targetFor(m))).components;
    } catch {
      components[m.name] = "error";
    }
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

function barWidth(m: ModelEntry): string {
  return `${percent(modelDiskBytes(m), groups.value.maxDiskBytes)}%`;
}

function componentList(name: string): ModelComponentStatus[] {
  const c = components[name];
  return Array.isArray(c) ? c : [];
}

/** Component rows currently mid-repair, keyed `model/component`. */
const repairing = ref<Set<string>>(new Set());

function repairKey(m: ModelEntry, c: ModelComponentStatus): string {
  return `${m.name}/${c.name}`;
}

/**
 * Re-fetch a missing component without deleting the model: the download
 * queue is keyed on the server-provided `repair_model` and skips files
 * already on disk, so only what's missing is pulled. Targets the same API
 * host the component listing came from (this view's current target) —
 * ModelsView already holds the downloads stream open, so progress lands in
 * the tray. On success the row flips once the listing is re-read.
 */
async function repairComponent(m: ModelEntry, c: ModelComponentStatus) {
  if (!c.repair_model) return;
  const key = repairKey(m, c);
  repairing.value.add(key);
  try {
    await startCatalogDownload(c.repair_model);
    toasts.push(`Repairing ${m.name} — re-fetching ${c.name}`);
    // Re-read presence after the pull lands; cheap and self-healing even if
    // the download is still in flight (the user can re-open Info later).
    components[m.name] = (await fetchModelComponents(m.name)).components;
  } catch (err) {
    toasts.push(
      err instanceof ApiError && err.status === 409
        ? `${c.repair_model} is already queued.`
        : String(err),
      "error",
    );
  } finally {
    repairing.value.delete(key);
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

        <div v-for="m in list" :key="m.name" class="mb-1.5">
          <div
            class="-mx-1 flex items-center gap-2 rounded-control px-1 transition-colors duration-100 hover:bg-bath"
          >
            <!-- residency -->
            <span
              class="h-1.5 w-1.5 shrink-0 rounded-full"
              :class="m.is_loaded ? 'bg-safelight' : 'bg-transparent'"
              role="img"
              :title="m.is_loaded ? 'On GPU' : 'Cold'"
              :aria-label="m.is_loaded ? 'On GPU' : 'Cold'"
            />
            <SourceGlyph :source="modelSource(m)" class="text-ink-3" />
            <span class="truncate text-body text-ink" :title="m.name">{{ m.name }}</span>
            <span
              v-for="label in hostLabels(m)"
              :key="label"
              data-test="installed-host"
              class="border-edge data-mono shrink-0 rounded-full border px-1.5 text-caption text-ink-3"
            >
              {{ label }}
            </span>
            <span
              v-if="quantTag(m.name)"
              class="border-edge data-mono rounded-full border px-1.5 text-caption text-ink-2"
            >
              {{ quantTag(m.name) }}
            </span>
            <button
              v-if="m.hf_repo"
              type="button"
              class="shrink-0 text-ink-3 transition-colors duration-100 hover:text-ink"
              :aria-label="`Open ${m.name} on Hugging Face`"
              title="Open on Hugging Face"
              data-test="model-page-link"
              @click="void openExternal(`https://huggingface.co/${m.hf_repo}`)"
            >
              <svg
                viewBox="0 0 12 12"
                width="11"
                height="11"
                fill="none"
                stroke="currentColor"
                stroke-width="1.2"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
              >
                <path d="M8.5 6.75v2.75a1 1 0 0 1-1 1H2.5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1h2.75" />
                <path d="M7 1.5h3.5V5" />
                <path d="M10.5 1.5 5.75 6.25" />
              </svg>
            </button>

            <!-- Primary weights and full runtime footprint are deliberately
                 separate: the latter includes shared encoders/VAEs. -->
            <div class="ml-auto flex w-64 items-center gap-2">
              <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-bath" aria-hidden="true">
                <div class="h-full bg-halide" :style="{ width: barWidth(m) }" />
              </div>
              <span class="w-40 shrink-0 text-right">
                <span class="data-mono block text-caption text-ink-2">
                  {{
                    modelSizeLabels(m).weights ?? modelSizeLabels(m).runtime ?? "Size unavailable"
                  }}
                </span>
                <span
                  v-if="modelSizeLabels(m).runtime && modelSizeLabels(m).weights"
                  class="data-mono block text-[10px] text-ink-3"
                >
                  {{ modelSizeLabels(m).runtime }}
                </span>
              </span>
            </div>

            <!-- actions -->
            <div class="flex shrink-0 items-center gap-1">
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
                class="border-edge h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-ink active:translate-y-px"
                @click="toggleInfo(m)"
              >
                Info
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
            </div>
          </div>

          <!-- components -->
          <div v-if="expanded[m.name]" class="mt-1 ml-4 border-l border-edge pl-3">
            <p v-if="components[m.name] === 'loading'" class="text-caption text-ink-3">Loading…</p>
            <p v-else-if="components[m.name] === 'error'" class="text-caption text-stop">
              Couldn't read components.
            </p>
            <div
              v-for="c in componentList(m.name)"
              v-else
              :key="c.name"
              class="flex items-center gap-2 py-0.5"
            >
              <span
                class="h-1.5 w-1.5 shrink-0 rounded-full"
                :class="c.present ? 'bg-halide' : 'bg-stop'"
                role="img"
                :title="c.present ? 'Present' : 'Missing'"
                :aria-label="c.present ? 'Present' : 'Missing'"
              />
              <span class="text-caption text-ink-2">{{ c.name }}</span>
              <span class="edge-code ml-auto">{{ c.kind }}</span>
              <button
                v-if="!c.present && c.repair_model"
                type="button"
                data-test="component-repair"
                class="border-edge h-6 shrink-0 rounded-control border px-2 text-caption text-safelight transition-colors duration-150 hover:border-safelight active:translate-y-px disabled:opacity-40"
                :disabled="repairing.has(repairKey(m, c))"
                :title="`Re-download the missing ${c.name} for ${m.name}`"
                @click="repairComponent(m, c)"
              >
                {{ repairing.has(repairKey(m, c)) ? "Repairing…" : "Repair" }}
              </button>
            </div>
          </div>
        </div>
      </section>
    </template>
  </div>
</template>
