<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import type { ModelEntry } from "../../lib/api/types";
import { modelAvailabilityTag } from "../../lib/hosts";
import { modelDisplayName, modelDisplayNameForId } from "../../lib/models";
import { familyLabel } from "@studio/lib/modelFamily";
import { modelSource } from "../../lib/modelSource";
import { formatGB } from "../../lib/format";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import SourceGlyph from "../generate/SourceGlyph.vue";

/**
 * The Mold Studio installed-model picker — the ONE style picker on Create.
 * A family-grouped menu with source glyphs, multi-host availability tags,
 * on-GPU state and a Browse footer, dismissed on outside pointerdown and
 * Escape.
 *
 * The trigger is a SLOT: the composer's Style chip opens it in place, above
 * the composer (`placement="up"`), so there is no second selector anywhere.
 * The default trigger stays for any consumer that wants a plain field.
 *
 * Every row says the plain thing in sans and the technical truth in mono, on
 * the same row (style guide): friendly name, then id · size · state.
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
     * entry with a "Not on this machine" tag instead of reading "Choose a
     * style", which made the restore look like it had silently dropped the
     * model.
     */
    missingModel?: string | null;
    /**
     * Which way the menu opens. The composer sits on the bottom edge of the
     * canvas, so its chip must open UPWARD or the menu falls off the window.
     */
    placement?: "down" | "up";
    /**
     * The mono kicker naming what this menu holds — the New image view's
     * section ("still picture styles"). Absent on a consumer that offers
     * every style.
     */
    kicker?: string | null;
    /**
     * The sentence for a menu whose whole list is empty, which is a different
     * fact from a filter that matched nothing: the section holds no styles at
     * all, and Browse more below is the way out. Absent falls back to the
     * generic line.
     */
    emptyLabel?: string | null;
  }>(),
  {
    showAvailability: true,
    disabledReason: null,
    browseTarget: "/models",
    browseLabel: "Browse more →",
    missingModel: null,
    placement: "down",
    kicker: null,
    emptyLabel: null,
  },
);

const emit = defineEmits<{ pick: [model: ModelEntry]; "pick-missing": [model: string] }>();

const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const router = useRouter();

const pickerEl = ref<HTMLDivElement | null>(null);
const filterEl = ref<HTMLInputElement | null>(null);
const open = ref(false);
const query = ref("");
const activeIndex = ref(0);

/** The phantom entry is only shown when nothing real is selected. */
const phantom = computed(() => (props.selected ? null : (props.missingModel ?? null)));
const phantomLabel = computed(() =>
  phantom.value ? modelDisplayNameForId(phantom.value, props.models) : "",
);

/** A short menu is faster to read than to type into; a long one is not. */
const FILTER_THRESHOLD = 8;
const showFilter = computed(() => props.models.length > FILTER_THRESHOLD);

function matches(m: ModelEntry, needle: string): boolean {
  if (!needle) return true;
  const haystack = `${m.name} ${modelDisplayName(m)} ${m.family} ${familyLabel(m.family)}`;
  return haystack.toLocaleLowerCase().includes(needle);
}

/** Deduped by id, in the order the parent handed them, narrowed by the filter. */
const visibleModels = computed<ModelEntry[]>(() => {
  const byName = new Map<string, ModelEntry>();
  for (const m of props.models) byName.set(m.name, m);
  const needle = query.value.trim().toLocaleLowerCase();
  return [...byName.values()].filter((m) => matches(m, needle));
});

const families = computed<Map<string, ModelEntry[]>>(() => {
  const groups = new Map<string, ModelEntry[]>();
  for (const m of visibleModels.value) {
    const list = groups.get(m.family) ?? [];
    list.push(m);
    groups.set(m.family, list);
  }
  return groups;
});

/** The phantom row (when shown) then every family's rows, in render order —
 *  the list ↑/↓ walks and Enter picks from. */
const rows = computed<ModelEntry[]>(() => [...families.value.values()].flat());
const hasPhantomRow = computed(() => phantom.value !== null && !query.value.trim());
/** Row 0 is the phantom when it renders, so model i sits at i + offset. */
const rowOffset = computed(() => (hasPhantomRow.value ? 1 : 0));
const rowCount = computed(() => rows.value.length + rowOffset.value);

/**
 * An empty list and an empty FILTER RESULT are different facts. The section
 * holding nothing is answered by the caller's own sentence, which names the
 * section; a filter that matched nothing keeps the generic line with the
 * needle in it.
 */
const emptyMessage = computed(() =>
  props.models.length === 0 && props.emptyLabel
    ? props.emptyLabel
    : `No style matches “${query.value}”.`,
);

function rowIndexFor(m: ModelEntry): number {
  return rows.value.indexOf(m) + rowOffset.value;
}

function availabilityTag(m: ModelEntry): string | null {
  if (!hosts.multiHost || !props.showAvailability) return null;
  return modelAvailabilityTag(hostModels.hostsFor(m.name), hosts.all);
}

function sizeLabel(m: ModelEntry): string | null {
  return m.disk_usage_bytes ? formatGB(m.disk_usage_bytes) : null;
}

/**
 * The entry's second line. Only a description that says something the title
 * does not — the catalog synthesises a name-shaped one for `cv:`/`hf:` rows,
 * which `modelDisplayName` already promoted into the title.
 */
function description(m: ModelEntry): string | null {
  const text = m.description?.trim();
  if (!text || modelDisplayName(m) !== m.name) return null;
  return text;
}

function isSelected(m: ModelEntry): boolean {
  return props.selected?.name === m.name;
}

function toggle() {
  open.value = !open.value;
}
function close() {
  open.value = false;
}

function pick(m: ModelEntry) {
  if (props.disabledReason?.(m)) return;
  emit("pick", m);
  close();
}

function pickMissing() {
  const name = phantom.value;
  if (!name) return;
  close();
  emit("pick-missing", name);
}

function activateRow(index: number) {
  if (hasPhantomRow.value && index === 0) {
    pickMissing();
    return;
  }
  const model = rows.value[index - rowOffset.value];
  if (model) pick(model);
}

function move(delta: number) {
  const count = rowCount.value;
  if (count === 0) return;
  activeIndex.value = (activeIndex.value + delta + count) % count;
}

function browse() {
  close();
  void router.push(props.browseTarget);
}

/** Keys reach here from the filter field or from whatever trigger has focus,
 *  so the menu never needs a document-level arrow listener. */
function onKeydown(event: KeyboardEvent) {
  if (!open.value) return;
  switch (event.key) {
    case "ArrowDown":
      event.preventDefault();
      move(1);
      break;
    case "ArrowUp":
      event.preventDefault();
      move(-1);
      break;
    case "Enter":
      event.preventDefault();
      activateRow(activeIndex.value);
      break;
    case "Escape":
      event.preventDefault();
      close();
      break;
  }
}

function onDocumentPointerDown(event: PointerEvent) {
  if (!open.value || !pickerEl.value) return;
  if (!event.composedPath().includes(pickerEl.value)) close();
}
function onDocumentKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") close();
}

// Force-fresh availability when the picker opens — a model pulled on an
// extra host by another client shows up the moment the user looks.
watch(open, (isOpen) => {
  if (!isOpen) return;
  query.value = "";
  const selectedRow = props.selected ? rowIndexFor(props.selected) : -1;
  activeIndex.value = selectedRow >= rowOffset.value ? selectedRow : 0;
  void hostModels.refresh(true);
  void nextTick(() => filterEl.value?.focus());
});

// A narrowed list can be shorter than where the cursor was.
watch(rowCount, (count) => {
  if (activeIndex.value >= count) activeIndex.value = Math.max(0, count - 1);
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
  <div ref="pickerEl" class="ms-model" data-test="model-picker" @keydown="onKeydown">
    <!-- The composer's Style chip fills this; the plain field is the default. -->
    <slot name="trigger" :open="open" :toggle="toggle">
      <button type="button" :aria-expanded="open" class="ms-model__button" @click="toggle">
        <span data-test="selected-model-name" class="min-w-0 break-all text-left">{{
          selected ? modelDisplayName(selected) : phantom ? phantomLabel : "Choose a style"
        }}</span>
        <span v-if="selected?.disk_usage_bytes" class="font-mono text-xs ms-model__size">
          {{ formatGB(selected.disk_usage_bytes) }}
        </span>
        <span
          v-else-if="phantom"
          data-test="selected-model-missing"
          class="font-mono text-micro text-fg-dim whitespace-nowrap shrink-0"
        >
          Not on this machine
        </span>
      </button>
    </slot>
    <div
      v-if="open"
      data-test="model-picker-menu"
      class="ms-model__menu"
      :class="placement === 'up' ? 'ms-model__menu--up' : 'ms-model__menu--down'"
      :data-placement="placement"
      role="listbox"
    >
      <!-- What this menu holds, in the section's own words. -->
      <p v-if="kicker" data-test="model-picker-kicker" class="ms-model__kicker">{{ kicker }}</p>
      <div v-if="showFilter" class="ms-model__filter">
        <input
          ref="filterEl"
          v-model="query"
          data-test="model-filter"
          data-selectable
          type="text"
          autocomplete="off"
          spellcheck="false"
          aria-label="Find a style"
          placeholder="Find a style…"
        />
      </div>
      <!-- The model the form actually carries, kept visible so a restored
           print never reads as "no model". Picking it offers the pull. -->
      <button
        v-if="hasPhantomRow"
        type="button"
        data-test="model-option-missing"
        class="ms-model__option"
        :class="{ 'ms-model__option--active': activeIndex === 0 }"
        role="option"
        :aria-selected="activeIndex === 0"
        @click="pickMissing"
      >
        <span class="min-w-0 flex-1">
          <span class="block break-all text-fg" :title="phantomLabel">{{ phantomLabel }}</span>
          <span class="font-mono text-micro text-fg-dim mt-0.5 block break-all">
            Not on this machine — get it
          </span>
        </span>
      </button>
      <template v-for="[family, list] in families" :key="family">
        <div class="ms-model__group">{{ familyLabel(family) }}</div>
        <button
          v-for="m in list"
          :key="m.name"
          type="button"
          class="ms-model__option"
          :class="{
            'ms-model__option--disabled': disabledReason?.(m),
            'ms-model__option--active': activeIndex === rowIndexFor(m),
            'ms-model__option--selected': isSelected(m),
          }"
          role="option"
          :aria-selected="isSelected(m)"
          :disabled="!!disabledReason?.(m)"
          @click="pick(m)"
          @mousemove="activeIndex = rowIndexFor(m)"
        >
          <SourceGlyph :source="modelSource(m)" class="mt-0.5 shrink-0 text-fg-dim" />
          <span class="min-w-0 flex-1">
            <span
              data-test="model-option-name"
              class="block break-all text-fg"
              :title="modelDisplayName(m)"
            >
              {{ modelDisplayName(m) }}
            </span>
            <span class="ms-model__meta">
              <span data-test="model-option-id" class="break-all">{{ m.name }}</span>
              <span v-if="sizeLabel(m)" data-test="model-option-size">{{ sizeLabel(m) }}</span>
              <span v-if="m.is_loaded" data-test="model-option-loaded" class="text-accent">
                on GPU
              </span>
            </span>
            <span
              v-if="disabledReason?.(m)"
              data-test="model-disabled-reason"
              class="font-mono text-micro text-fg-dim mt-0.5 block break-all"
            >
              {{ disabledReason?.(m) }}
            </span>
            <span
              v-else-if="availabilityTag(m)"
              data-test="model-availability"
              class="font-mono text-micro text-fg-dim mt-0.5 block break-all"
            >
              {{ availabilityTag(m) }}
            </span>
            <span v-if="description(m)" data-test="model-option-description" class="ms-model__desc">
              {{ description(m) }}
            </span>
          </span>
          <span
            v-if="isSelected(m)"
            data-test="model-option-current"
            class="ms-model__current"
            title="Current style"
            aria-hidden="true"
            >✓</span
          >
        </button>
      </template>
      <p v-if="rowCount === 0" data-test="model-picker-empty" class="ms-model__empty">
        {{ emptyMessage }}
      </p>
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
  cursor: pointer;
  min-height: 40px;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-3);
  background: var(--mold-bg-deep);
  padding: 0 12px;
  font-size: var(--mold-fs-sm);
  color: var(--mold-text);
}
.ms-model__size {
  flex-shrink: 0;
  color: var(--mold-text-dim);
}
.ms-model__menu {
  position: absolute;
  z-index: 30;
  max-height: 22rem;
  width: 100%;
  min-width: 20rem;
  overflow-y: auto;
  overflow-x: hidden;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  background: var(--mold-bg);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
}
.ms-model__menu--down {
  top: 100%;
  margin-top: 4px;
}
/* The composer is on the bottom edge of the canvas: downward would leave the
 * window. Anchored to the trigger's top, growing up. */
.ms-model__menu--up {
  bottom: 100%;
  margin-bottom: 4px;
}
.ms-model__filter {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 8px;
  background: var(--mold-bg);
  border-bottom: 1px solid var(--mold-border);
}
.ms-model__filter input {
  width: 100%;
  height: 28px;
  padding: 0 8px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-size: var(--mold-fs-xs);
}
.ms-model__filter input:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}
/* The section caption: quieter than a family heading, same mono vocabulary. */
.ms-model__kicker {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--mold-text-faint);
  padding: 8px 8px 0;
}
.ms-model__group {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--mold-text-dim);
  padding: 8px 8px 4px;
}
.ms-model__option {
  display: flex;
  cursor: pointer;
  width: 100%;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 8px;
  text-align: left;
  font-size: var(--mold-fs-sm);
  color: var(--mold-text-2);
}
.ms-model__option:hover:not(:disabled),
.ms-model__option--active:not(:disabled) {
  background: var(--mold-bg-deep);
  color: var(--mold-text);
}
.ms-model__option--selected {
  box-shadow: inset 2px 0 0 var(--mold-blue);
}
.ms-model__option--disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
/* Technical truth in mono, beside the plain name. */
.ms-model__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0 8px;
  margin-top: 2px;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
.ms-model__desc {
  display: block;
  margin-top: 2px;
  font-size: var(--mold-fs-micro);
  line-height: var(--mold-lh-snug);
  color: var(--mold-text-dim);
}
.ms-model__current {
  flex-shrink: 0;
  margin-top: 2px;
  color: var(--mold-blue);
}
.ms-model__empty {
  padding: 12px 8px;
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
}
.ms-model__browse {
  display: flex;
  cursor: pointer;
  width: 100%;
  align-items: center;
  border-top: 1px solid var(--mold-border);
  padding: 8px;
  text-align: left;
  font-size: var(--mold-fs-sm);
  color: var(--mold-sapphire);
}
.ms-model__browse:hover {
  background: var(--mold-bg-deep);
}
</style>
