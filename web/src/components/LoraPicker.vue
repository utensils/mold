<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { RouterLink } from "vue-router";
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

function moveAt(from: number, to: number) {
  if (from === to) return;
  if (from < 0 || from >= props.modelValue.length) return;
  if (to < 0 || to >= props.modelValue.length) return;
  const next = props.modelValue.slice();
  const [row] = next.splice(from, 1);
  next.splice(to, 0, row);
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
  <section class="lp" data-test="lora-picker">
    <div class="lp__head">
      <label class="lp__kicker">
        LoRA{{ modelValue.length > 1 ? ` stack (${modelValue.length})` : "" }}
      </label>
      <button
        v-if="canAddMore"
        type="button"
        class="lp__add"
        @click="addAnother"
      >
        + Add LoRA
      </button>
    </div>

    <input
      v-if="loras.length > 6"
      v-model="search"
      type="search"
      class="lp__search"
      placeholder="Search LoRAs"
      aria-label="Search LoRAs"
    />

    <div
      v-for="(row, index) in modelValue"
      :key="`${row.path}-${index}`"
      class="lp__row"
      data-test="lora-row"
    >
      <div class="lp__row-top">
        <select
          :value="row.path"
          class="lp__select"
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
        <div class="lp__row-actions">
          <button
            type="button"
            class="lp__icon-btn"
            aria-label="Move LoRA up"
            :disabled="index === 0"
            @click="moveAt(index, index - 1)"
          >
            ↑
          </button>
          <button
            type="button"
            class="lp__icon-btn"
            aria-label="Move LoRA down"
            :disabled="index === modelValue.length - 1"
            @click="moveAt(index, index + 1)"
          >
            ↓
          </button>
          <button
            type="button"
            class="lp__icon-btn lp__icon-btn--danger"
            aria-label="Remove this LoRA"
            @click="removeAt(index)"
          >
            ✕
          </button>
        </div>
      </div>

      <div class="lp__scale">
        <div class="lp__scale-head">
          <span class="lp__scale-label">Weight</span>
          <span class="lp__scale-val">{{ row.scale.toFixed(2) }}×</span>
        </div>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          :value="row.scale"
          class="lp__range"
          @input="changeScale(index, $event)"
        />
      </div>

      <div
        v-if="trainedWordsForRow(row).length > 0"
        class="lp__triggers"
        aria-label="Trigger words — click to append to prompt"
      >
        <button
          v-for="phrase in trainedWordsForRow(row)"
          :key="phrase"
          type="button"
          class="lp__trigger"
          :title="`Append “${phrase}” to the prompt`"
          @click="emit('append-prompt', phrase)"
        >
          + {{ phrase }}
        </button>
      </div>
    </div>

    <p
      v-if="modelValue.length === 0 && loras.length > 0"
      class="lp__hint"
      data-test="lora-hint-select"
    >
      no lora selected — <span class="lp__accent">+ Add LoRA</span> to stack
      one.
    </p>
    <p
      v-if="modelValue.length === 0 && loras.length === 0"
      class="lp__hint"
      data-test="lora-hint-empty"
    >
      no loras installed for this model. pull one from
      <RouterLink to="/models" class="lp__accent">Models</RouterLink> and it
      shows up here.
    </p>
  </section>
</template>

<style scoped>
.lp {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.lp__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.lp__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.lp__add {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 5px 12px;
  border-radius: var(--radius-pill);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.lp__add:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}
.lp__search {
  width: 100%;
  box-sizing: border-box;
  height: 36px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  padding: 0 12px;
  font-size: 13px;
  font-family: var(--f-mono);
  outline: none;
}
.lp__row {
  border: 1px solid var(--edge);
  background: var(--bench);
  border-radius: var(--radius-control-lg);
  padding: 11px;
}
.lp__row-top {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  align-items: center;
}
.lp__select {
  min-width: 0;
  height: 36px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  padding: 0 10px;
  font-size: 13px;
  font-family: var(--f-mono);
}
.lp__row-actions {
  display: flex;
  align-items: center;
  gap: 5px;
}
.lp__icon-btn {
  height: 32px;
  width: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control);
  font-size: 13px;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.lp__icon-btn:hover:not(:disabled) {
  border-color: var(--safelight);
  color: var(--rebate);
}
.lp__icon-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.lp__icon-btn--danger:hover:not(:disabled) {
  border-color: var(--stop);
  color: var(--stop);
}
.lp__scale {
  margin-top: 12px;
}
.lp__scale-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}
.lp__scale-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink-2);
}
.lp__scale-val {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--safelight);
}
.lp__range {
  width: 100%;
  accent-color: var(--safelight);
}
.lp__triggers {
  margin-top: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.lp__trigger {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 4px 10px;
  border-radius: var(--radius-pill);
  font-size: 11px;
  font-family: var(--f-mono);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.lp__trigger:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}
.lp__hint {
  font-size: 12px;
  color: var(--ink-3);
  line-height: 1.5;
}
.lp__accent {
  color: var(--safelight);
  font-weight: 600;
  text-decoration: none;
}
</style>
