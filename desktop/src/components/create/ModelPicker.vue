<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import type { ModelEntry } from "../../lib/api/types";
import { modelAvailabilityTag } from "../../lib/hosts";
import { modelDisplayName, modelDisplayNameForId } from "../../lib/models";
import { modelSource } from "../../lib/modelSource";
import { formatGB } from "../../lib/format";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import SourceGlyph from "../generate/SourceGlyph.vue";

/**
 * The Mold Studio installed-model picker (extracted from the Create
 * inspector so the chain composer shares the exact same control): a
 * family-grouped dropdown with source glyphs, multi-host availability tags,
 * on-GPU dots, and a Browse footer. The menu dismisses on outside
 * pointerdown and Escape.
 */
const props = withDefaults(
  defineProps<{
    models: ModelEntry[];
    selected: ModelEntry | null;
    /** Multi-host availability tags; parents suppress them for a sticky host. */
    showAvailability?: boolean;
    /** Non-null marks the entry unpickable and explains why, inline. */
    disabledReason?: ((m: ModelEntry) => string | null) | null;
    browseTarget?: string;
    browseLabel?: string;
    /**
     * A model id the form carries that no machine has installed — a restored
     * print, a template, or a deleted checkpoint. It renders as the selected
     * entry with a Not installed tag instead of reading "Choose a model",
     * which made the restore look like it had silently dropped the model.
     */
    missingModel?: string | null;
  }>(),
  {
    showAvailability: true,
    disabledReason: null,
    browseTarget: "/models",
    browseLabel: "Browse all models →",
    missingModel: null,
  },
);

const emit = defineEmits<{ pick: [model: ModelEntry]; "pick-missing": [model: string] }>();

const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const router = useRouter();

const pickerEl = ref<HTMLDivElement | null>(null);
const open = ref(false);

/** The phantom entry is only shown when nothing real is selected. */
const phantom = computed(() => (props.selected ? null : (props.missingModel ?? null)));
const phantomLabel = computed(() =>
  phantom.value ? modelDisplayNameForId(phantom.value, props.models) : "",
);

function pickMissing() {
  const name = phantom.value;
  if (!name) return;
  open.value = false;
  emit("pick-missing", name);
}

const families = computed<Map<string, ModelEntry[]>>(() => {
  const byName = new Map<string, ModelEntry>();
  for (const m of props.models) byName.set(m.name, m);
  const groups = new Map<string, ModelEntry[]>();
  for (const m of byName.values()) {
    const list = groups.get(m.family) ?? [];
    list.push(m);
    groups.set(m.family, list);
  }
  return groups;
});

function availabilityTag(m: ModelEntry): string | null {
  if (!hosts.multiHost || !props.showAvailability) return null;
  return modelAvailabilityTag(hostModels.hostsFor(m.name), hosts.all);
}

function pick(m: ModelEntry) {
  if (props.disabledReason?.(m)) return;
  emit("pick", m);
  open.value = false;
}

function browse() {
  open.value = false;
  void router.push(props.browseTarget);
}

function onDocumentPointerDown(event: PointerEvent) {
  if (!open.value || !pickerEl.value) return;
  if (!event.composedPath().includes(pickerEl.value)) open.value = false;
}
function onDocumentKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") open.value = false;
}

// Force-fresh availability when the picker opens — a model pulled on an
// extra host by another client shows up the moment the user looks.
watch(open, (isOpen) => {
  if (isOpen) void hostModels.refresh(true);
});

onMounted(() => {
  document.addEventListener("pointerdown", onDocumentPointerDown);
  document.addEventListener("keydown", onDocumentKeydown);
});
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onDocumentPointerDown);
  document.removeEventListener("keydown", onDocumentKeydown);
});
</script>

<template>
  <div ref="pickerEl" class="ms-model">
    <button type="button" :aria-expanded="open" class="ms-model__button" @click="open = !open">
      <span data-test="selected-model-name" class="min-w-0 break-all text-left">{{
        selected ? modelDisplayName(selected) : phantom ? phantomLabel : "Choose a model"
      }}</span>
      <span v-if="selected?.disk_usage_bytes" class="data-mono ms-model__size">
        {{ formatGB(selected.disk_usage_bytes) }}
      </span>
      <span v-else-if="phantom" data-test="selected-model-missing" class="edge-code shrink-0">
        Not installed
      </span>
    </button>
    <div v-if="open" data-test="model-picker-menu" class="ms-model__menu">
      <!-- The model the form actually carries, kept visible so a restored
           print never reads as "no model". Picking it offers the pull. -->
      <button
        v-if="phantom"
        type="button"
        data-test="model-option-missing"
        class="ms-model__option"
        @click="pickMissing"
      >
        <span class="min-w-0 flex-1">
          <span class="block break-all text-ink" :title="phantomLabel">{{ phantomLabel }}</span>
          <span class="edge-code mt-0.5 block break-all whitespace-normal">
            Not installed — download it
          </span>
        </span>
      </button>
      <template v-for="[family, list] in families" :key="family">
        <div class="ms-model__group">{{ family.toUpperCase() }}</div>
        <button
          v-for="m in list"
          :key="m.name"
          type="button"
          class="ms-model__option"
          :class="{ 'ms-model__option--disabled': disabledReason?.(m) }"
          :disabled="!!disabledReason?.(m)"
          @click="pick(m)"
        >
          <SourceGlyph :source="modelSource(m)" class="mt-0.5 shrink-0 text-ink-3" />
          <span class="min-w-0 flex-1">
            <span
              data-test="model-option-name"
              class="block break-all text-ink"
              :title="modelDisplayName(m)"
            >
              {{ modelDisplayName(m) }}
            </span>
            <span
              v-if="disabledReason?.(m)"
              data-test="model-disabled-reason"
              class="edge-code mt-0.5 block break-all whitespace-normal"
            >
              {{ disabledReason?.(m) }}
            </span>
            <span
              v-else-if="availabilityTag(m)"
              data-test="model-availability"
              class="edge-code mt-0.5 block break-all whitespace-normal"
            >
              {{ availabilityTag(m) }}
            </span>
          </span>
          <span
            class="ms-model__loaded"
            :class="m.is_loaded ? 'bg-safelight' : 'bg-transparent'"
            :title="m.is_loaded ? 'On GPU' : ''"
          />
        </button>
      </template>
      <button type="button" data-test="browse-catalog" class="ms-model__browse" @click="browse">
        {{ browseLabel }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.ms-model {
  position: relative;
}
.ms-model__button {
  display: flex;
  min-height: 40px;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  border: 1px solid var(--ce);
  border-radius: 9px;
  background: var(--bath);
  padding: 0 12px;
  font-size: 13px;
  color: var(--rebate);
}
.ms-model__size {
  flex-shrink: 0;
  color: var(--ink-3);
}
.ms-model__menu {
  position: absolute;
  z-index: 30;
  margin-top: 4px;
  max-height: 18rem;
  width: 100%;
  min-width: 16rem;
  overflow-y: auto;
  border: 1px solid var(--edge);
  border-radius: 12px;
  background: var(--bench);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
}
.ms-model__group {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  padding: 8px 8px 4px;
}
.ms-model__option {
  display: flex;
  width: 100%;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 8px;
  text-align: left;
  font-size: 13px;
  color: var(--ink-2);
}
.ms-model__option:hover:not(:disabled) {
  background: var(--bath);
  color: var(--rebate);
}
.ms-model__option--disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
.ms-model__loaded {
  margin-top: 8px;
  height: 6px;
  width: 6px;
  flex-shrink: 0;
  border-radius: 9999px;
}
.ms-model__browse {
  display: flex;
  width: 100%;
  align-items: center;
  border-top: 1px solid var(--edge);
  padding: 8px;
  text-align: left;
  font-size: 13px;
  color: var(--halide);
}
.ms-model__browse:hover {
  background: var(--bath);
}
</style>
