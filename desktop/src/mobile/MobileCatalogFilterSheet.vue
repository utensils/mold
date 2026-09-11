<script setup lang="ts">
/*
 * Filters — every way of narrowing the Styles list, in one sheet.
 *
 * The phone stacked seven separate narrowing surfaces above the results in
 * three different visual languages: `aria-pressed` pills for the shelf, media
 * type, catalog source and model kind; native `<select>` chrome for the
 * machine, family and sort; and a bare checkbox for NSFW. Four of them
 * appeared and vanished with the shelf, so the list jumped down the screen
 * every time you switched. On a 393pt screen that left barely two results
 * visible.
 *
 * What stays in the scroll is what you touch constantly — the shelf and the
 * search field. Everything else lives here, behind one chip that says how many
 * are set. The controls themselves are unchanged: same classes, same
 * data-tests, same semantics, so the shelf rules still read the same way.
 *
 * The body stays mounted whether the sheet is open or closed, like More
 * settings: the catalog reads these controls' state to build its query.
 */
import { ref, toRef } from "vue";
import { useOverlayStack } from "@ui/lib/overlayStack";
import { useMobileBack } from "./useMobileBack";
import { useSheetDismiss } from "./useSheetDismiss";
import { useSheetFocus } from "./useSheetFocus";
import {
  CATALOG_KIND_OPTIONS,
  type CatalogKindFilter,
  CATALOG_SORT_OPTIONS,
  type CatalogSortOption,
  type CatalogSource,
} from "../lib/catalogFilters";
import type { MobileHost } from "./hosts";

const props = defineProps<{
  open: boolean;
  hosts: MobileHost[];
  selectedHostId: string;
  /** False on the Ready-to-use shelf, where a catalog query has no meaning. */
  discover: boolean;
  familyOptions: string[];
}>();

const emit = defineEmits<{
  close: [];
  reset: [];
  "select-host": [hostId: string];
}>();

const source = defineModel<CatalogSource>("source", { required: true });
const kind = defineModel<CatalogKindFilter | "">("kind", { required: true });
const family = defineModel<string>("family", { required: true });
const sort = defineModel<CatalogSortOption>("sort", { required: true });
const includeNsfw = defineModel<boolean>("includeNsfw", { required: true });

useMobileBack(toRef(props, "open"), () => emit("close"));
const { isTop } = useOverlayStack(toRef(props, "open"), "mobile-catalog-filters");
const panel = ref<HTMLElement | null>(null);
const body = ref<HTMLElement | null>(null);

const { dragging, panelStyle, backdropStyle, beginDismiss, moveDismiss, finishDismiss, resetDrag } =
  useSheetDismiss({ body, onDismiss: () => emit("close") });

/* `aria-modal="true"` over a background that is not inert is a promise only
 * this keeps: focus in on open, Tab held inside, Escape out, focus restored. */
const { onKeydown } = useSheetFocus({
  panel,
  open: () => props.open,
  isTop,
  onClose: () => emit("close"),
  onBeforeClose: resetDrag,
});
</script>

<template>
  <div
    class="mobile-sheet mobile-catalog-filter-sheet"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-modal="true"
    :inert="!open"
    aria-label="Filters"
    :aria-hidden="open ? undefined : 'true'"
    data-test="mobile-catalog-filters"
    @keydown="onKeydown"
  >
    <button
      class="mobile-sheet-scrim"
      type="button"
      data-sheet-close
      aria-label="Close Filters"
      :style="backdropStyle"
      @click="emit('close')"
    />
    <div
      ref="panel"
      class="mobile-sheet-panel"
      :class="{ 'is-dragging': dragging }"
      :style="panelStyle"
      tabindex="-1"
      @touchstart="beginDismiss"
      @touchmove="moveDismiss"
      @touchend="finishDismiss"
      @touchcancel="resetDrag"
    >
      <span class="mobile-sheet-grabber" aria-hidden="true" />
      <header class="mobile-sheet-head">
        <div class="mobile-sheet-head-slot">
          <button
            class="mobile-sheet-action"
            type="button"
            data-test="mobile-catalog-filters-reset"
            @click="emit('reset')"
          >
            Reset
          </button>
        </div>
        <div class="mobile-sheet-heading">
          <h2 class="mobile-sheet-title">Filters</h2>
        </div>
        <div class="mobile-sheet-head-slot">
          <button
            class="mobile-sheet-action is-strong"
            type="button"
            data-sheet-close
            data-test="mobile-catalog-filters-done"
            @click="emit('close')"
          >
            Done
          </button>
        </div>
      </header>
      <div ref="body" class="mobile-sheet-body">
        <label v-if="hosts.length > 1" class="mobile-catalog-host-picker">
          <span>Browse on</span>
          <select
            :value="selectedHostId"
            aria-label="Machine"
            @change="emit('select-host', ($event.target as HTMLSelectElement).value)"
          >
            <option v-for="host in hosts" :key="host.id" :value="host.id">
              {{ host.name }}{{ host.online ? "" : " · offline" }}
            </option>
          </select>
        </label>

        <div
          v-if="discover"
          class="mobile-catalog-sources"
          role="group"
          aria-label="Catalog source"
        >
          <button
            v-for="option in ['all', 'hf', 'civitai'] as const"
            :key="option"
            type="button"
            :aria-pressed="source === option"
            @click="source = option"
          >
            {{ option === "all" ? "All" : option === "hf" ? "HuggingFace" : "Civitai" }}
          </button>
        </div>

        <div
          v-if="discover"
          class="mobile-catalog-kinds"
          role="group"
          aria-label="Model kind"
          data-test="mobile-catalog-kind-chips"
        >
          <button type="button" :aria-pressed="kind === ''" @click="kind = ''">All</button>
          <button
            v-for="option in CATALOG_KIND_OPTIONS"
            :key="option.value"
            type="button"
            :aria-pressed="kind === option.value"
            @click="kind = option.value"
          >
            {{ option.label }}
          </button>
        </div>

        <div v-if="discover" class="mobile-catalog-filters">
          <label>
            <span>Family</span>
            <select v-model="family" data-test="mobile-catalog-family">
              <option value="">All families</option>
              <option v-for="option in familyOptions" :key="option" :value="option">
                {{ option }}
              </option>
            </select>
          </label>
          <label>
            <span>Sort</span>
            <select v-model="sort" data-test="mobile-catalog-sort">
              <option
                v-for="option in CATALOG_SORT_OPTIONS"
                :key="option.value"
                :value="option.value"
              >
                {{ option.label }}
              </option>
            </select>
          </label>
          <label class="mobile-catalog-nsfw">
            <input v-model="includeNsfw" type="checkbox" />
            <span>Include NSFW</span>
          </label>
        </div>
      </div>
    </div>
  </div>
</template>
