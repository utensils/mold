<script setup lang="ts" generic="M extends StyleMenuModel">
import { computed, nextTick, onMounted, ref, useId, watch } from "vue";
import { familyLabel } from "../lib/modelFamily";
import { formatBytes } from "../lib/formatBytes";
import { modelDisplayName, modelDisplayNameForId } from "../lib/modelDisplay";
import { styleDisplayName } from "../lib/styleLabel";
import { modelSource } from "../lib/modelSource";
import SourceGlyph from "@ui/components/SourceGlyph.vue";
import type { StyleMenuModel } from "../lib/styleMenu";

/**
 * The ONE style list — the menu desktop's composer chip, web's Create chip and
 * the phone's Style sheet all open. Only the LIST is shared: each surface keeps
 * its own container (a popover anchored above the composer on desktop, a
 * teleported popover on web, a bottom sheet on the phone) because that is the
 * part that genuinely differs.
 *
 * It is always-rendered content. Visibility, placement, dismissal and the
 * refresh-on-open belong to the host, which mounts this only while open.
 *
 * Every row says the plain thing in sans and the technical truth in mono, on
 * the same row: friendly name, then id · size · state.
 *
 * Nothing here may reach a store, a router or a shell — `studio/` stays
 * browser-safe and application-shell independent
 * (`scripts/tests/frontend-architecture.sh`), so the availability tag and the
 * refusal reason are INJECTED by the host. The source glyph is drawn by the
 * menu itself from `modelSource(model)` — `studio/` may import `@ui`
 * (precedent: `MeshWorkflowStudio.vue`, `NotificationsCenter.vue`) — and a
 * host may still override it through the `glyph` slot, which desktop does
 * to add its own tone classes. The rule behind the availability tag is now
 * ONE shared rule,
 * `@studio/lib/modelAvailability`, but it still arrives as a function: the
 * menu cannot reach a machine list, and each surface spells reachability and
 * the "is this even a fleet" guard differently.
 */
const props = withDefaults(
  defineProps<{
    models: M[];
    selected: M | null;
    /**
     * A style id the form carries that no machine has installed — a restored
     * print, a template, or a deleted checkpoint. It renders as a phantom row
     * at the top instead of the menu reading as if the style were dropped.
     */
    missingModel?: string | null;
    /**
     * The mono kicker naming what this menu holds — the New image view's
     * section ("still picture styles"). Absent on a host that offers every
     * style.
     */
    kicker?: string | null;
    /**
     * The sentence for a menu whose whole list is empty, which is a different
     * fact from a filter that matched nothing.
     */
    emptyLabel?: string | null;
    /** Non-null marks the row unpickable and explains why, inline. */
    disabledReason?: ((model: M) => string | null) | null;
    /** `@studio/lib/modelAvailability`, bound by the host to its own
     *  reachable machines; null renders nothing at all. */
    availabilityTag?: ((model: M) => string | null) | null;
    /** Finger-sized rows and body text, for the phone's sheet. */
    touch?: boolean;
    /** Take focus into the filter field on mount (a host that opens with a
     *  pointer wants this; one whose own trigger keeps focus does not). */
    autofocusFilter?: boolean;
    browseLabel?: string;
  }>(),
  {
    missingModel: null,
    kicker: null,
    emptyLabel: null,
    disabledReason: null,
    availabilityTag: null,
    touch: false,
    autofocusFilter: false,
    browseLabel: "Browse more →",
  },
);

const emit = defineEmits<{
  pick: [model: M];
  "pick-missing": [model: string];
  browse: [];
}>();

const rootEl = ref<HTMLElement | null>(null);
const filterEl = ref<HTMLInputElement | null>(null);

/* A screen reader follows the cursor through `aria-activedescendant`, so
 * every row needs a stable id of its own and the listbox has to be
 * addressable — including by a host chip's `aria-controls`. */
const menuId = useId();
const rowId = (index: number) => `${menuId}-row-${index}`;
const query = ref("");
const activeIndex = ref(0);

/** The phantom row is only shown when nothing real is selected. */
const phantom = computed(() =>
  props.selected ? null : (props.missingModel ?? null),
);
const phantomLabel = computed(() =>
  phantom.value ? modelDisplayNameForId(phantom.value, props.models) : "",
);

/** A short menu is faster to read than to type into; a long one is not. */
const FILTER_THRESHOLD = 8;
const showFilter = computed(() => props.models.length > FILTER_THRESHOLD);

function matches(model: M, needle: string): boolean {
  if (!needle) return true;
  const haystack = `${model.name} ${modelDisplayName(model)} ${styleDisplayName(model)} ${
    model.description ?? ""
  } ${model.family} ${familyLabel(model.family)}`;
  return haystack.toLocaleLowerCase().includes(needle);
}

/** Deduped by id, in the order the host handed them, narrowed by the filter. */
const visibleModels = computed<M[]>(() => {
  const byName = new Map<string, M>();
  for (const model of props.models) byName.set(model.name, model);
  const needle = query.value.trim().toLocaleLowerCase();
  return [...byName.values()].filter((model) => matches(model, needle));
});

const families = computed<Map<string, M[]>>(() => {
  const groups = new Map<string, M[]>();
  for (const model of visibleModels.value) {
    const list = groups.get(model.family) ?? [];
    list.push(model);
    groups.set(model.family, list);
  }
  return groups;
});

/** Every family's rows in render order — the list ↑/↓ walks and Enter picks
 *  from, with the phantom (when shown) occupying index 0 ahead of them. */
const rows = computed<M[]>(() => [...families.value.values()].flat());
const hasPhantomRow = computed(
  () => phantom.value !== null && !query.value.trim(),
);
const rowOffset = computed(() => (hasPhantomRow.value ? 1 : 0));
const rowCount = computed(() => rows.value.length + rowOffset.value);

/**
 * An empty list and an empty FILTER RESULT are different facts. The section
 * holding nothing is answered by the host's own sentence, which names the
 * section; a filter that matched nothing keeps the generic line with the
 * needle in it.
 */
const emptyMessage = computed(() =>
  props.models.length === 0 && props.emptyLabel
    ? props.emptyLabel
    : `No style matches “${query.value}”.`,
);

function rowIndexFor(model: M): number {
  return rows.value.indexOf(model) + rowOffset.value;
}

function sizeLabel(model: M): string | null {
  return model.disk_usage_bytes ? formatBytes(model.disk_usage_bytes) : null;
}

/**
 * The row's TITLE, in the lexicon's own order: the description first, then a
 * curated `display_name`, then the family's friendly name — never the id,
 * which is a "never say" as a primary label (`docs/design/README.md` §2) and
 * already rides beneath in mono. `modelDisplayName` answers a different
 * question — "what is this row called" — and for every manifest style the
 * answer is its id, so using it here printed `flux-schnell:q8` over
 * `flux-schnell:q8 · 23.1 GB` while the chip above already said the
 * description.
 */
function title(model: M): string {
  return styleDisplayName(model);
}

/**
 * The row's second line: the OTHER name it has.
 *
 * `styleDisplayName` ranks the description above `display_name`, so a catalog
 * row carrying both would otherwise lose the curated one entirely. Nothing is
 * repeated — a name equal to the title, or to the id already in mono, is not
 * a second fact.
 */
function secondaryName(model: M): string | null {
  const other = modelDisplayName(model);
  if (other === title(model) || other === model.name) return null;
  return other;
}

function isSelected(model: M): boolean {
  return props.selected?.name === model.name;
}

function pick(model: M) {
  if (props.disabledReason?.(model)) return;
  emit("pick", model);
}

function pickMissing() {
  const name = phantom.value;
  if (name) emit("pick-missing", name);
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

/**
 * ↑/↓/Enter, and nothing else. Escape belongs to the host: web's popover
 * consumes it in the CAPTURE phase before it could ever reach this root, so a
 * menu that thought it owned Escape would be wrong on one of the three
 * surfaces.
 *
 * A host whose trigger keeps focus (desktop's chip, when the list is too short
 * to carry a filter field) forwards keys here through the exposed handler; it
 * checks `defaultPrevented` first so a key that already walked the list inside
 * this root is never counted twice.
 */
function onKeydown(event: KeyboardEvent) {
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
  }
}

/**
 * A host whose container does not move focus (web's popover teleports the
 * panel to <body> and leaves focus on the chip) calls `focus()` on open, and
 * forwards its trigger's keys through `handleKeydown`. Without BOTH, a list
 * too short to carry a filter field could be opened and then walked nowhere.
 */
defineExpose({
  handleKeydown: onKeydown,
  focus: () => rootEl.value?.focus(),
});

// A narrowed list can be shorter than where the cursor was.
watch(rowCount, (count) => {
  if (activeIndex.value >= count) activeIndex.value = Math.max(0, count - 1);
});

onMounted(() => {
  const selectedRow = props.selected ? rowIndexFor(props.selected) : -1;
  activeIndex.value = selectedRow >= rowOffset.value ? selectedRow : 0;
  if (props.autofocusFilter) void nextTick(() => filterEl.value?.focus());
});
</script>

<template>
  <div
    :id="menuId"
    ref="rootEl"
    class="ms-model__menu"
    :class="{ 'ms-model__menu--touch': touch }"
    data-test="model-picker-menu"
    role="listbox"
    tabindex="-1"
    :aria-activedescendant="rowCount ? rowId(activeIndex) : undefined"
    @keydown="onKeydown"
  >
    <!-- What this menu holds, in the section's own words. -->
    <p v-if="kicker" data-test="model-picker-kicker" class="ms-model__kicker">
      {{ kicker }}
    </p>
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
    <!-- The style the form actually carries, kept visible so a restored print
         never reads as "no style". Picking it offers the pull. -->
    <button
      v-if="hasPhantomRow"
      :id="rowId(0)"
      type="button"
      data-test="model-option-missing"
      class="ms-model__option"
      :class="{ 'ms-model__option--active': activeIndex === 0 }"
      role="option"
      :aria-selected="activeIndex === 0"
      @click="pickMissing"
      @mousemove="activeIndex = 0"
    >
      <span class="ms-model__body">
        <span class="ms-model__name" :title="phantomLabel">{{
          phantomLabel
        }}</span>
        <span class="ms-model__tag">Not on this machine — get it</span>
      </span>
    </button>
    <template v-for="[family, list] in families" :key="family">
      <div class="ms-model__group">{{ familyLabel(family) }}</div>
      <button
        v-for="model in list"
        :id="rowId(rowIndexFor(model))"
        :key="model.name"
        type="button"
        class="ms-model__option"
        :class="{
          'ms-model__option--disabled': disabledReason?.(model),
          'ms-model__option--active': activeIndex === rowIndexFor(model),
          'ms-model__option--selected': isSelected(model),
        }"
        role="option"
        :aria-selected="isSelected(model)"
        :disabled="!!disabledReason?.(model)"
        @click="pick(model)"
        @mousemove="activeIndex = rowIndexFor(model)"
      >
        <!-- The menu draws a source glyph per row by default; a host may
             override with its own (desktop fills this with tone classes). -->
        <slot name="glyph" :model="model">
          <SourceGlyph :source="modelSource(model)" class="ms-model__glyph" />
        </slot>
        <span class="ms-model__body">
          <span
            data-test="model-option-name"
            class="ms-model__name"
            :title="title(model)"
          >
            {{ title(model) }}
          </span>
          <span class="ms-model__meta">
            <span data-test="model-option-id">{{ model.name }}</span>
            <span v-if="sizeLabel(model)" data-test="model-option-size">{{
              sizeLabel(model)
            }}</span>
            <span
              v-if="model.is_loaded"
              data-test="model-option-loaded"
              class="ms-model__gpu"
            >
              on GPU
            </span>
          </span>
          <span
            v-if="disabledReason?.(model)"
            data-test="model-disabled-reason"
            class="ms-model__tag"
          >
            {{ disabledReason?.(model) }}
          </span>
          <span
            v-else-if="availabilityTag?.(model)"
            data-test="model-availability"
            class="ms-model__tag"
          >
            {{ availabilityTag?.(model) }}
          </span>
          <span
            v-if="secondaryName(model)"
            data-test="model-option-description"
            class="ms-model__desc"
          >
            {{ secondaryName(model) }}
          </span>
        </span>
        <span
          v-if="isSelected(model)"
          data-test="model-option-current"
          class="ms-model__current"
          title="Current style"
          aria-hidden="true"
          >✓</span
        >
      </button>
    </template>
    <p
      v-if="rowCount === 0"
      data-test="model-picker-empty"
      class="ms-model__empty"
    >
      {{ emptyMessage }}
    </p>
    <button
      type="button"
      data-test="browse-catalog"
      class="ms-model__browse"
      @click="emit('browse')"
    >
      {{ browseLabel }}
    </button>
  </div>
</template>

<style scoped>
/* The SURFACE — placement, width, border, background, elevation — belongs to
 * the host's popover or sheet; this only owns the list inside it. */
.ms-model__menu {
  display: block;
  overflow-x: hidden;
}
/* It takes focus so the arrow keys reach it, but it is not a tab stop and it
 * draws no ring of its own — the active ROW is what the cursor marks. */
.ms-model__menu:focus {
  outline: none;
}
/* The section caption: quieter than a family heading, same mono vocabulary. */
.ms-model__kicker {
  margin: 0;
  font-family: var(--mold-font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro, 0.6875rem);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--mold-text-faint, #6b7280);
  padding: 8px 8px 0;
}
.ms-model__filter {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 8px;
  background: var(--mold-bg, #111);
  border-bottom: 1px solid var(--mold-border, #2a2a2a);
}
.ms-model__filter input {
  width: 100%;
  box-sizing: border-box;
  height: 28px;
  padding: 0 8px;
  border: 1px solid var(--mold-border-control, #3a3a3a);
  border-radius: var(--mold-radius-2, 4px); /* literal: token fallback only */
  background: var(--mold-bg-deep, #0b0b0b);
  color: var(--mold-text, #e5e5e5);
  font-size: var(--mold-fs-xs, 0.75rem);
}
.ms-model__filter input:focus {
  outline: none;
  border-color: var(--mold-border-focus, #6b8afd);
}
.ms-model__group {
  font-family: var(--mold-font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro, 0.6875rem);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--mold-text-dim, #9ca3af);
  padding: 8px 8px 4px;
}
.ms-model__option {
  display: flex;
  cursor: pointer;
  width: 100%;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 8px;
  border: 0;
  background: none;
  text-align: left;
  font-size: var(--mold-fs-sm, 0.8125rem);
  color: var(--mold-text-2, #d1d5db);
}
.ms-model__option:hover:not(:disabled),
.ms-model__option--active:not(:disabled) {
  background: var(--mold-bg-deep, #0b0b0b);
  color: var(--mold-text, #e5e5e5);
}
.ms-model__option--selected {
  box-shadow: inset 2px 0 0 var(--mold-blue, #6b8afd);
}
.ms-model__option--disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
.ms-model__body {
  min-width: 0;
  flex: 1;
}
/* The default glyph, drawn when a host does not fill the slot itself. */
.ms-model__glyph {
  margin-top: 2px;
  color: var(--mold-text-dim);
}
/* A style id is one long unbroken token. It WRAPS rather than being cut: the
 * id is the thing a person copies, so an ellipsis makes the row useless. */
.ms-model__name {
  display: block;
  overflow-wrap: anywhere;
  color: var(--mold-text, #e5e5e5);
}
/* Technical truth in mono, beneath the plain name. */
.ms-model__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0 8px;
  margin-top: 2px;
  overflow-wrap: anywhere;
  font-family: var(--mold-font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro, 0.6875rem);
  color: var(--mold-text-dim, #9ca3af);
}
.ms-model__gpu {
  color: var(--mold-accent, #6b8afd);
}
.ms-model__tag {
  display: block;
  margin-top: 2px;
  overflow-wrap: anywhere;
  font-family: var(--mold-font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro, 0.6875rem);
  color: var(--mold-text-dim, #9ca3af);
}
.ms-model__desc {
  display: block;
  margin-top: 2px;
  font-size: var(--mold-fs-micro, 0.6875rem);
  line-height: var(--mold-lh-snug, 1.3);
  color: var(--mold-text-dim, #9ca3af);
}
.ms-model__current {
  flex-shrink: 0;
  margin-top: 2px;
  color: var(--mold-blue, #6b8afd);
}
.ms-model__empty {
  margin: 0;
  padding: 12px 8px;
  font-size: var(--mold-fs-xs, 0.75rem);
  color: var(--mold-text-dim, #9ca3af);
}
.ms-model__browse {
  display: flex;
  cursor: pointer;
  width: 100%;
  align-items: center;
  border: 0;
  border-top: 1px solid var(--mold-border, #2a2a2a);
  background: none;
  padding: 8px;
  text-align: left;
  font-size: var(--mold-fs-sm, 0.8125rem);
  color: var(--mold-sapphire, #6b8afd);
}
.ms-model__browse:hover {
  background: var(--mold-bg-deep, #0b0b0b);
}

/* A finger is not a cursor: the phone's sheet gets iOS's 44px target and body
 * text it can read, without changing a pixel on desktop or web. */
.ms-model__menu--touch .ms-model__option {
  min-height: 44px;
  padding: 10px 12px;
  font-size: var(--mold-fs-md, 1rem);
}
.ms-model__menu--touch .ms-model__browse {
  min-height: 44px;
  padding: 10px 12px;
  font-size: var(--mold-fs-md, 1rem);
}
.ms-model__menu--touch .ms-model__filter input {
  height: 40px;
  font-size: var(--mold-fs-md, 1rem);
}
</style>
