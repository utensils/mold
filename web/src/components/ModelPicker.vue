<script setup lang="ts">
import { computed, ref } from "vue";
import type { ModelInfoExtended } from "../types";
import { VIDEO_FAMILIES } from "../types";
import { useDownloads } from "../composables/useDownloads";

const props = defineProps<{
  models: ModelInfoExtended[];
  modelValue: string;
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: string): void;
  (e: "select", model: ModelInfoExtended): void;
}>();

const SHOW_ALL_KEY = "mold.generate.showAllModels";
const showAll = ref(localStorage.getItem(SHOW_ALL_KEY) === "true");

function setShowAll(v: boolean) {
  showAll.value = v;
  try {
    localStorage.setItem(SHOW_ALL_KEY, String(v));
  } catch {
    /* ignore */
  }
}

const visibleModels = computed(() =>
  props.models.filter((m) => (showAll.value ? true : m.downloaded)),
);

const imageModels = computed(() =>
  visibleModels.value.filter((m) => !VIDEO_FAMILIES.includes(m.family)),
);
const videoModels = computed(() =>
  visibleModels.value.filter((m) => VIDEO_FAMILIES.includes(m.family)),
);

const downloads = useDownloads();

type DownloadUiState =
  | { kind: "idle" }
  | { kind: "active"; pct: number }
  | { kind: "queued"; position: number; id: string }
  | { kind: "failed"; id: string };

function downloadStateFor(name: string): DownloadUiState {
  if (downloads.active.value && downloads.active.value.model === name) {
    const a = downloads.active.value;
    const pct = a.bytes_total
      ? Math.min(100, Math.round((a.bytes_done / a.bytes_total) * 100))
      : 0;
    return { kind: "active", pct };
  }
  const q = downloads.queued.value.findIndex((j) => j.model === name);
  if (q >= 0)
    return {
      kind: "queued",
      position: q + 1,
      id: downloads.queued.value[q].id,
    };
  const failed = downloads.history.value.find(
    (j) => j.model === name && j.status === "failed",
  );
  if (failed) return { kind: "failed", id: failed.id };
  return { kind: "idle" };
}

function fmtSize(m: ModelInfoExtended): string {
  if (m.size_gb >= 1) return `${m.size_gb.toFixed(1)} GB`;
  return `${(m.size_gb * 1024).toFixed(0)} MB`;
}

function onPick(model: ModelInfoExtended) {
  if (!model.downloaded) return;
  emit("update:modelValue", model.name);
  emit("select", model);
}

async function startDownload(model: ModelInfoExtended) {
  try {
    await downloads.enqueue(model.name);
  } catch (err) {
    console.error("failed to enqueue download", err);
  }
}

async function cancelQueued(id: string) {
  await downloads.cancel(id);
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <label
      class="flex items-center justify-between text-xs uppercase text-slate-400"
    >
      <span>Model</span>
      <span class="flex items-center gap-2 normal-case">
        <input
          id="mold-show-all-models"
          type="checkbox"
          :checked="showAll"
          @change="setShowAll(($event.target as HTMLInputElement).checked)"
        />
        <label for="mold-show-all-models">Show all</label>
      </span>
    </label>

    <div class="flex max-h-80 flex-col gap-3 overflow-y-auto pr-1">
      <div>
        <div class="text-xs font-medium text-slate-500">Images</div>
        <ul class="mt-1 flex flex-col gap-1">
          <li v-for="m in imageModels" :key="m.name">
            <div
              class="group w-full rounded-xl px-3 py-2 text-left text-sm"
              :class="[
                modelValue === m.name
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200',
              ]"
            >
              <button
                type="button"
                class="flex w-full items-center justify-between gap-2"
                :disabled="!m.downloaded"
                :class="!m.downloaded ? 'cursor-default opacity-70' : ''"
                :title="m.description"
                @click="onPick(m)"
              >
                <span class="flex items-center gap-2">
                  <span>{{ m.name }}</span>
                  <span class="text-xs text-slate-400">({{ fmtSize(m) }})</span>
                </span>
                <span class="text-xs text-slate-400">{{ m.family }}</span>
              </button>
              <div class="text-xs text-slate-400">{{ m.description }}</div>

              <!-- Download affordance row -->
              <div v-if="!m.downloaded" class="mt-2">
                <template v-if="downloadStateFor(m.name).kind === 'idle'">
                  <button
                    class="rounded-full bg-brand-500/20 px-2 py-0.5 text-xs text-brand-100 hover:bg-brand-500/40"
                    @click="startDownload(m)"
                  >
                    Download
                  </button>
                </template>
                <template
                  v-else-if="downloadStateFor(m.name).kind === 'active'"
                >
                  <div
                    class="h-1.5 w-full overflow-hidden rounded-full bg-white/10"
                    role="progressbar"
                  >
                    <div
                      class="h-full bg-brand-400"
                      :style="{
                        width:
                          (
                            downloadStateFor(m.name) as Extract<
                              DownloadUiState,
                              { kind: 'active' }
                            >
                          ).pct + '%',
                      }"
                    />
                  </div>
                  <div class="mt-1 text-xs text-slate-400">
                    Downloading…
                    {{
                      (
                        downloadStateFor(m.name) as Extract<
                          DownloadUiState,
                          { kind: "active" }
                        >
                      ).pct
                    }}%
                  </div>
                </template>
                <template
                  v-else-if="downloadStateFor(m.name).kind === 'queued'"
                >
                  <span
                    class="inline-flex items-center gap-1 rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-200"
                  >
                    Queued (#{{
                      (
                        downloadStateFor(m.name) as Extract<
                          DownloadUiState,
                          { kind: "queued" }
                        >
                      ).position
                    }})
                    <button
                      class="ml-1 text-slate-300 hover:text-white"
                      aria-label="Cancel queued download"
                      @click="
                        cancelQueued(
                          (
                            downloadStateFor(m.name) as Extract<
                              DownloadUiState,
                              { kind: 'queued' }
                            >
                          ).id,
                        )
                      "
                    >
                      ×
                    </button>
                  </span>
                </template>
                <template
                  v-else-if="downloadStateFor(m.name).kind === 'failed'"
                >
                  <button
                    class="rounded-full bg-red-500/20 px-2 py-0.5 text-xs text-red-200 hover:bg-red-500/40"
                    @click="startDownload(m)"
                  >
                    Retry
                  </button>
                </template>
              </div>
            </div>
          </li>
        </ul>
      </div>

      <div v-if="videoModels.length">
        <div class="flex items-center gap-2 text-xs font-medium text-slate-500">
          <span>🎬</span><span>Video</span>
        </div>
        <ul class="mt-1 flex flex-col gap-1">
          <li v-for="m in videoModels" :key="m.name">
            <div
              class="w-full rounded-xl px-3 py-2 text-left text-sm"
              :class="[
                modelValue === m.name
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200',
              ]"
            >
              <button
                type="button"
                class="flex w-full items-center justify-between gap-2"
                :disabled="!m.downloaded"
                :class="!m.downloaded ? 'cursor-default opacity-70' : ''"
                :title="m.description"
                @click="onPick(m)"
              >
                <span>
                  {{ m.name }}
                  <span class="italic text-xs text-slate-400">video</span>
                  <span class="ml-1 text-xs text-slate-400"
                    >({{ fmtSize(m) }})</span
                  >
                </span>
                <span class="text-xs text-slate-400">{{ m.family }}</span>
              </button>
              <div class="text-xs text-slate-400">{{ m.description }}</div>
              <div v-if="!m.downloaded" class="mt-2">
                <button
                  class="rounded-full bg-brand-500/20 px-2 py-0.5 text-xs text-brand-100 hover:bg-brand-500/40"
                  @click="startDownload(m)"
                >
                  Download
                </button>
              </div>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>
