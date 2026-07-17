<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { useModelStore } from "../../stores/models";
import { useToastStore } from "../../stores/toasts";
import SourceGlyph from "../generate/SourceGlyph.vue";
import { groupInstalledModels, modelDiskBytes, quantTag } from "../../lib/models";
import { modelSource } from "../../lib/modelSource";
import { fetchModelComponents, loadModel, removeModel, unloadModel } from "../../lib/api/models";
import { ApiError } from "../../lib/api/client";
import { formatGB, percent } from "../../lib/format";
import { openExternal } from "../../lib/openExternal";
import type { ModelComponentStatus, ModelEntry } from "../../lib/api/types";

const props = defineProps<{ query?: string }>();
const emit = defineEmits<{ (e: "browse-catalog"): void }>();

const models = useModelStore();
const toasts = useToastStore();

const filtered = computed(() => {
  const q = (props.query ?? "").trim().toLowerCase();
  return q ? models.installed.filter((m) => m.name.toLowerCase().includes(q)) : models.installed;
});
const groups = computed(() => groupInstalledModels(filtered.value));

/** Family groups plus a trailing utility section, as [heading, models] rows. */
const sections = computed<[string, ModelEntry[]][]>(() => [
  ...groups.value.families,
  ["SHARED / UTILITY", groups.value.utility],
]);

// Per-row Info expansion + lazily-loaded component lists.
const expanded = reactive<Record<string, boolean>>({});
const components = reactive<Record<string, ModelComponentStatus[] | "loading" | "error">>({});
const busy = ref<string | null>(null);
const confirmingRemove = ref<string | null>(null);

async function remove(m: ModelEntry) {
  if (confirmingRemove.value !== m.name) {
    confirmingRemove.value = m.name;
    return;
  }
  confirmingRemove.value = null;
  busy.value = m.name;
  try {
    const result = await removeModel(m.name);
    const kept = result.kept.length;
    toasts.push(
      kept > 0
        ? `Removed ${m.name} — freed ${formatGB(result.freed_bytes)}, kept ${kept} shared component${kept === 1 ? "" : "s"}`
        : `Removed ${m.name} — freed ${formatGB(result.freed_bytes)}`,
    );
    await models.fetch();
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

async function toggleInfo(m: ModelEntry) {
  expanded[m.name] = !expanded[m.name];
  if (expanded[m.name] && components[m.name] === undefined) {
    components[m.name] = "loading";
    try {
      components[m.name] = (await fetchModelComponents(m.name)).components;
    } catch {
      components[m.name] = "error";
    }
  }
}

async function load(m: ModelEntry) {
  busy.value = m.name;
  try {
    await loadModel(m.name);
    await models.fetch();
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    busy.value = null;
  }
}

async function unload(m: ModelEntry) {
  busy.value = m.name;
  try {
    await unloadModel(m.name);
    await models.fetch();
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

            <!-- disk usage bar -->
            <div class="ml-auto flex w-40 items-center gap-2">
              <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-bath" aria-hidden="true">
                <div class="h-full bg-halide" :style="{ width: barWidth(m) }" />
              </div>
              <span class="data-mono w-16 shrink-0 text-right text-ink-3">
                {{ formatGB(modelDiskBytes(m)) }}
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
            </div>
          </div>
        </div>
      </section>
    </template>
  </div>
</template>
