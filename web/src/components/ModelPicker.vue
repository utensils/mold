<script setup lang="ts">
import { computed, ref } from "vue";
import type { ModelInfoExtended } from "../types";
import {
  applyModelFilters,
  familyLabel,
  groupModelsByFamily,
  isStandaloneGenerationModel,
  modelQuantization,
  sortModels,
  type DownloadedFilter,
  type ModelFilterMode,
  type ModelSortKey,
  type SizeBucket,
} from "../lib/modelFilters";

defineOptions({ name: "ModelPicker" });

const props = defineProps<{
  models: ModelInfoExtended[];
  modelValue: string;
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: string): void;
  (e: "select", model: ModelInfoExtended): void;
}>();

const COLLAPSED_FAMILIES_KEY = "mold.generate.modelPicker.collapsedFamilies";

const query = ref("");
const filterMode = ref<ModelFilterMode>("all");
const downloadedFilter = ref<DownloadedFilter>("downloaded");
const selectedFamily = ref("");
const selectedQuantization = ref("");
const selectedSize = ref<SizeBucket | "">("");
const sortKeys = ref<ModelSortKey[]>(["family", "variantRank", "name"]);
const collapsedFamilies = ref<Set<string>>(loadCollapsedFamilies());

function loadCollapsedFamilies(): Set<string> {
  try {
    const raw = localStorage.getItem(COLLAPSED_FAMILIES_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? new Set(parsed.filter(isString)) : new Set();
  } catch {
    return new Set();
  }
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function persistCollapsedFamilies(next: Set<string>) {
  try {
    localStorage.setItem(
      COLLAPSED_FAMILIES_KEY,
      JSON.stringify(Array.from(next).sort()),
    );
  } catch {
    /* ignore */
  }
}

function toggleFamily(family: string) {
  const next = new Set(collapsedFamilies.value);
  if (next.has(family)) next.delete(family);
  else next.add(family);
  collapsedFamilies.value = next;
  persistCollapsedFamilies(next);
}

const generationModels = computed(() =>
  props.models.filter(isStandaloneGenerationModel),
);
const downloadedGenerationCount = computed(
  () => generationModels.value.filter((m) => m.downloaded).length,
);
const showEmptyState = computed(() => downloadedGenerationCount.value === 0);

const familyOptions = computed(() =>
  groupModelsByFamily(generationModels.value).map((group) => ({
    family: group.family,
    label: group.label,
  })),
);

const quantizationOptions = computed(() =>
  Array.from(new Set(generationModels.value.map(modelQuantization))).sort(),
);

const filterState = computed(() => ({
  mode: filterMode.value,
  query: query.value,
  families: selectedFamily.value ? [selectedFamily.value] : [],
  quantizations: selectedQuantization.value ? [selectedQuantization.value] : [],
  sizeBuckets: selectedSize.value ? [selectedSize.value] : [],
  downloaded: downloadedFilter.value,
}));

const filteredModels = computed(() =>
  sortModels(
    applyModelFilters(generationModels.value, filterState.value),
    sortKeys.value,
  ),
);

const groupedModels = computed(() => groupModelsByFamily(filteredModels.value));

const activeSortKeys = computed(() => new Set(sortKeys.value));

function toggleSort(key: ModelSortKey) {
  const current = sortKeys.value.filter((existing) => existing !== key);
  sortKeys.value = activeSortKeys.value.has(key) ? current : [key, ...current];
}

function onPick(model: ModelInfoExtended) {
  if (!model.downloaded) return;
  emit("update:modelValue", model.name);
  emit("select", model);
}

function fmtSize(m: ModelInfoExtended): string {
  if (m.size_gb >= 1) return `${m.size_gb.toFixed(1)} GB`;
  return `${(m.size_gb * 1024).toFixed(0)} MB`;
}

function sizeLabel(bucket: SizeBucket): string {
  if (bucket === "small") return "<5 GB";
  if (bucket === "medium") return "5-10 GB";
  if (bucket === "large") return "10-20 GB";
  return "20+ GB";
}
</script>

<template>
  <div class="flex flex-col gap-2 text-sm">
    <div class="text-xs uppercase text-slate-400">Model</div>

    <input
      v-model="query"
      type="search"
      class="rounded-md bg-slate-900/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500"
      placeholder="Search models"
      aria-label="Search models"
      data-test="filter-query"
    />

    <div class="grid grid-cols-2 gap-1.5">
      <select
        v-model="filterMode"
        class="rounded-md bg-slate-900/60 px-2 py-1.5 text-xs text-slate-100"
        aria-label="Filter logic"
        data-test="filter-mode"
      >
        <option value="all">All filters</option>
        <option value="any">Any filter</option>
        <option value="not">Not filters</option>
      </select>
      <select
        v-model="downloadedFilter"
        class="rounded-md bg-slate-900/60 px-2 py-1.5 text-xs text-slate-100"
        aria-label="Downloaded state"
        data-test="filter-downloaded"
      >
        <option value="downloaded">Downloaded</option>
        <option value="missing">Missing</option>
        <option value="any">Any state</option>
      </select>
      <select
        v-model="selectedFamily"
        class="rounded-md bg-slate-900/60 px-2 py-1.5 text-xs text-slate-100"
        aria-label="Family filter"
        data-test="filter-family"
      >
        <option value="">Any family</option>
        <option
          v-for="family in familyOptions"
          :key="family.family"
          :value="family.family"
        >
          {{ family.label }}
        </option>
      </select>
      <select
        v-model="selectedQuantization"
        class="rounded-md bg-slate-900/60 px-2 py-1.5 text-xs text-slate-100"
        aria-label="Quantization filter"
        data-test="filter-quantization"
      >
        <option value="">Any quant</option>
        <option v-for="q in quantizationOptions" :key="q" :value="q">
          {{ q }}
        </option>
      </select>
      <select
        v-model="selectedSize"
        class="col-span-2 rounded-md bg-slate-900/60 px-2 py-1.5 text-xs text-slate-100"
        aria-label="Size filter"
      >
        <option value="">Any size</option>
        <option
          v-for="bucket in ['small', 'medium', 'large', 'xlarge']"
          :key="bucket"
          :value="bucket"
        >
          {{ sizeLabel(bucket as SizeBucket) }}
        </option>
      </select>
    </div>

    <div class="flex flex-wrap gap-1" aria-label="Sort models">
      <button
        v-for="sort in [
          ['name', 'Name'],
          ['family', 'Family'],
          ['size', 'Size'],
          ['quantization', 'Quant'],
          ['downloaded', 'State'],
          ['variantRank', 'Variant'],
        ]"
        :key="sort[0]"
        type="button"
        class="rounded-md px-2 py-1 text-xs"
        :class="
          activeSortKeys.has(sort[0] as ModelSortKey)
            ? 'bg-brand-500/30 text-brand-100'
            : 'bg-slate-900/60 text-slate-300 hover:bg-white/10'
        "
        :data-test="`sort-${sort[0]}`"
        @click="toggleSort(sort[0] as ModelSortKey)"
      >
        {{ sort[1] }}
      </button>
    </div>

    <div
      v-if="showEmptyState"
      data-test="model-picker-empty"
      class="rounded-md border border-dashed border-slate-700 bg-slate-900/50 px-3 py-2 text-xs text-slate-300"
    >
      No downloaded generation models.
      <RouterLink
        to="/catalog"
        data-test="model-catalog-link"
        class="text-brand-200 underline decoration-brand-400/60 underline-offset-2"
      >
        Open model catalog
      </RouterLink>
    </div>

    <div v-else class="flex max-h-96 flex-col gap-2 overflow-y-auto pr-1">
      <div
        v-if="groupedModels.length === 0"
        class="rounded-md bg-slate-900/50 px-3 py-2 text-xs text-slate-400"
      >
        No models match these filters.
      </div>

      <section
        v-for="group in groupedModels"
        v-else
        :key="group.family"
        class="overflow-hidden rounded-md bg-slate-950/30"
      >
        <button
          type="button"
          class="flex w-full items-center justify-between gap-2 px-2 py-1.5 text-left text-xs font-medium text-slate-300 hover:bg-white/5"
          :aria-expanded="!collapsedFamilies.has(group.family)"
          :data-test="`family-toggle-${group.family}`"
          @click="toggleFamily(group.family)"
        >
          <span>{{ group.label }}</span>
          <span class="text-slate-500">
            {{ group.models.length }}
            {{ collapsedFamilies.has(group.family) ? "+" : "-" }}
          </span>
        </button>
        <ul
          v-if="!collapsedFamilies.has(group.family)"
          class="flex flex-col gap-1 px-1 pb-1"
        >
          <li v-for="m in group.models" :key="m.name" data-test="model-option">
            <div
              class="w-full rounded-md px-2 py-2 text-left"
              :class="[
                modelValue === m.name
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/70 text-slate-200',
              ]"
            >
              <button
                type="button"
                class="flex w-full items-start justify-between gap-2 text-left"
                :disabled="!m.downloaded"
                :class="!m.downloaded ? 'cursor-default opacity-60' : ''"
                :title="m.description"
                :data-test="`model-option-${m.name}`"
                @click="onPick(m)"
              >
                <span class="min-w-0">
                  <span class="block truncate">{{ m.name }}</span>
                  <span class="text-xs text-slate-400">
                    {{ familyLabel(m.family) }} · {{ fmtSize(m) }} ·
                    {{ modelQuantization(m) }}
                  </span>
                </span>
                <span
                  v-if="!m.downloaded"
                  class="shrink-0 rounded bg-white/10 px-1.5 py-0.5 text-xs text-slate-300"
                >
                  Missing
                </span>
              </button>
              <div
                v-if="m.description"
                class="mt-1 line-clamp-2 text-xs text-slate-400"
              >
                {{ m.description }}
              </div>
            </div>
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>
