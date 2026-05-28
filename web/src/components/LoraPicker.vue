<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { fetchCatalogInstalled } from "../api";
import type { CatalogEntryWire, LoraSelection } from "../types";
import { MAX_LORA_STACK } from "../types";

defineOptions({ name: "LoraPicker" });

/// Multi-LoRA picker. The component owns its own catalog fetch (filtered
/// to `kind=lora` for the current model family) and exposes the user-
/// chosen stack via `modelValue` as an array. The parent feeds back a
/// fresh array on every change — Vue's reactivity layer handles the rest.
///
/// Trigger words: when the chosen LoRA was picked from the catalog (not
/// arbitrary path), `trainedWords` is populated and we render each phrase
/// as a small clickable chip. Clicking a chip emits `append-prompt` with
/// the phrase so the parent can append it to the active prompt.
const props = defineProps<{
  family: string;
  modelValue: LoraSelection[];
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: LoraSelection[]): void;
  (e: "append-prompt", phrase: string): void;
}>();

const loras = ref<CatalogEntryWire[]>([]);
const loading = ref(false);
const search = ref("");

async function loadLoras(family: string) {
  loras.value = [];
  if (!family) return;
  loading.value = true;
  try {
    // Live-search bypasses the bulk-scrape catalog DB; the picker reads
    // from per-install sidecar files so an obscure LoRA that never showed
    // up on page 1 of search still appears here.
    const res = await fetchCatalogInstalled({ family, kind: "lora" });
    loras.value = res.entries.filter((e) => e.installed && e.primary_path);
  } catch {
    // silently ignore — picker just stays empty
  } finally {
    loading.value = false;
  }
}

watch(() => props.family, loadLoras, { immediate: true });

const canAddMore = computed(
  () => props.modelValue.length < MAX_LORA_STACK && loras.value.length > 0,
);

const filteredLoras = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return loras.value;
  return loras.value.filter((entry) =>
    [
      entry.name,
      entry.author ?? "",
      entry.id,
      entry.primary_path ?? "",
      ...(entry.trained_words ?? []),
    ]
      .join(" ")
      .toLowerCase()
      .includes(q),
  );
});

/// Resolve the catalog entry for a given LoRA path so we can surface its
/// trigger words on every selected row, even after a fresh page load (the
/// stack persists in localStorage but `trainedWords` is a derived field).
function entryForPath(path: string): CatalogEntryWire | undefined {
  return loras.value.find((e) => e.primary_path === path);
}

function trainedWordsForRow(row: LoraSelection): string[] {
  // Prefer the row's own snapshot (set when the user picked from the
  // dropdown); fall back to the catalog lookup so a re-rendered row
  // still gets chips after a refresh.
  if (row.trainedWords && row.trainedWords.length > 0) return row.trainedWords;
  const entry = entryForPath(row.path);
  return entry?.trained_words ?? [];
}

function selectAt(index: number, event: Event) {
  const path = (event.target as HTMLSelectElement).value;
  const next = props.modelValue.slice();
  if (!path) {
    next.splice(index, 1);
    emit("update:modelValue", next);
    return;
  }
  const entry = entryForPath(path);
  next[index] = {
    path,
    scale: next[index]?.scale ?? 1.0,
    trainedWords: entry?.trained_words ?? [],
  };
  emit("update:modelValue", next);
}

function changeScale(index: number, event: Event) {
  const scale = Number((event.target as HTMLInputElement).value);
  const row = props.modelValue[index];
  if (!row) return;
  const next = props.modelValue.slice();
  next[index] = { ...row, scale };
  emit("update:modelValue", next);
}

function removeAt(index: number) {
  const next = props.modelValue.slice();
  next.splice(index, 1);
  emit("update:modelValue", next);
}

function addAnother() {
  if (!canAddMore.value) return;
  // Default to the first LoRA not yet in the stack to keep the picker
  // useful out of the box; user can change it via the dropdown.
  const used = new Set(props.modelValue.map((l) => l.path));
  const fresh = loras.value.find((e) => !used.has(e.primary_path!));
  if (!fresh || !fresh.primary_path) return;
  const next = [
    ...props.modelValue,
    {
      path: fresh.primary_path,
      scale: 1.0,
      trainedWords: fresh.trained_words ?? [],
    },
  ];
  emit("update:modelValue", next);
}
</script>

<template>
  <section
    v-if="loras.length > 0 || modelValue.length > 0"
    class="mt-4 space-y-3"
  >
    <div class="flex items-center justify-between">
      <label class="text-xs uppercase text-slate-400">
        LoRA{{ modelValue.length > 1 ? ` stack (${modelValue.length})` : "" }}
      </label>
      <button
        v-if="canAddMore"
        type="button"
        class="rounded-md bg-slate-800 px-2 py-0.5 text-xs text-slate-200 hover:bg-slate-700"
        @click="addAnother"
      >
        + Add LoRA
      </button>
    </div>

    <input
      v-if="loras.length > 6"
      v-model="search"
      type="search"
      class="w-full rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100 placeholder:text-slate-500"
      placeholder="Search LoRAs"
      aria-label="Search LoRAs"
    />

    <div
      v-for="(row, index) in modelValue"
      :key="`${row.path}-${index}`"
      class="rounded-lg border border-slate-800 bg-slate-900/40 p-2"
    >
      <div class="flex items-center gap-2">
        <select
          :value="row.path"
          class="flex-1 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
          @change="selectAt(index, $event)"
        >
          <option value="">— remove —</option>
          <option
            v-for="e in filteredLoras"
            :key="e.id"
            :value="e.primary_path!"
          >
            {{ e.name }}
          </option>
        </select>
        <button
          type="button"
          class="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300 hover:bg-rose-700/50"
          aria-label="Remove this LoRA"
          @click="removeAt(index)"
        >
          ✕
        </button>
      </div>

      <div class="mt-2">
        <label class="text-xs text-slate-400">
          Scale — {{ row.scale.toFixed(2) }}
        </label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          :value="row.scale"
          class="w-full"
          @input="changeScale(index, $event)"
        />
      </div>

      <div
        v-if="trainedWordsForRow(row).length > 0"
        class="mt-2 flex flex-wrap gap-1"
        aria-label="Trigger words — click to append to prompt"
      >
        <button
          v-for="phrase in trainedWordsForRow(row)"
          :key="phrase"
          type="button"
          class="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-200 hover:bg-emerald-700/40"
          :title="`Append “${phrase}” to the prompt`"
          @click="emit('append-prompt', phrase)"
        >
          {{ phrase }}
        </button>
      </div>
    </div>

    <p
      v-if="modelValue.length === 0 && loras.length > 0"
      class="text-xs text-slate-500"
    >
      No LoRA selected — click <span class="text-slate-400">+ Add LoRA</span> to
      stack one.
    </p>
  </section>
</template>
