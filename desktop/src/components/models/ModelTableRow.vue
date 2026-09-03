<script setup lang="ts">
import { computed, useId, useSlots } from "vue";

import { familyLabel } from "@studio/lib/modelFamily";

import Tooltip from "@ui/components/Tooltip.vue";

import SourceGlyph from "../generate/SourceGlyph.vue";
import ModelFootprintBar from "./ModelFootprintBar.vue";
import { isOpaqueModelId } from "../../lib/models";
import { openExternal } from "../../lib/openExternal";
import type { ModelSource } from "../../lib/modelSource";

/**
 * The one row shape for models shown as a table anywhere in the app —
 * the Styles shelf, a machine's detail page, and the catalog's table
 * layout all render this component so model rows stay visually identical.
 *
 * Purely presentational: parents own data fetching, size resolution, and
 * actions (the `#actions` slot). README §04 table row, left to right:
 * residency star · source glyph · name stacked over its mono id, quant,
 * and family · machines (dot + mono name) · page link · `#meta` · a plain
 * one-line note · relative footprint + two-line size block · `#actions`.
 */
const props = withDefaults(
  defineProps<{
    name: string;
    /** The request id, shown in mono under the name when the two differ. */
    id?: string | null;
    source: ModelSource;
    /** GPU residency star: true = warm, false = cold placeholder, omit = no column. */
    loaded?: boolean | undefined;
    hostLabels?: string[];
    quant?: string | null;
    family?: string | null;
    /** External model page; renders the link-out icon when present. */
    pageUrl?: string | null;
    /** One line in plain words — what the style is good for. */
    note?: string | null;
    /** Right-aligned size block: primary line + smaller secondary line. */
    sizePrimary?: string | null;
    sizeSecondary?: string | null;
    /** 0–100 fill for the relative-usage bar; omit to hide the bar. */
    barPercent?: number | null;
    /** Row and name become buttons that emit `open` (catalog detail drawer). */
    clickable?: boolean;
    /** When false, only the title is a semantic button. Use this when slots
     * contain their own interactive controls such as selection checkboxes. */
    interactiveContainer?: boolean;
    /** The row currently backing an open detail drawer. */
    selected?: boolean;
    /** Rich accessible name when visible metadata adds context to the row. */
    accessibilityLabel?: string | null;
  }>(),
  {
    id: null,
    loaded: undefined,
    hostLabels: () => [],
    quant: null,
    family: null,
    pageUrl: null,
    note: null,
    sizePrimary: null,
    sizeSecondary: null,
    barPercent: null,
    clickable: false,
    interactiveContainer: true,
    selected: false,
    accessibilityLabel: null,
  },
);

const emit = defineEmits<{ (e: "open"): void }>();
const slots = useSlots();
const footprintDescriptionId = `model-footprint-${useId()}`;

/**
 * The family reads as its name, not its wire slug (#806) — a Wan row read
 * "wan" here while web read "Wan Video".
 */
const familyChip = computed(() => (props.family ? familyLabel(props.family) : ""));
/** A manifest model's display name IS its id — one line, never two copies —
 * and an opaque install id is a number nobody recognises, so it stays off. */
const showId = computed(() =>
  Boolean(props.id && props.id !== props.name && !isOpaqueModelId(props.id)),
);

/**
 * Enter/Space on a clickable row opens it, but keydown bubbles — a keypress
 * on an inner control (Get it, the link-out) reaches here too. Only act when
 * the row itself is focused so those controls keep their own single action.
 */
function onRowKeydown(event: KeyboardEvent): void {
  if (!props.clickable || event.target !== event.currentTarget) return;
  event.preventDefault();
  emit("open");
}
</script>

<template>
  <div
    class="model-table-row transition-colors duration-100 hover:bg-row-hover"
    :class="[
      clickable ? 'cursor-pointer focus-visible:outline-2 focus-visible:outline-accent' : '',
      barPercent != null ? 'model-table-row--has-footprint' : '',
      note ? 'model-table-row--has-note' : '',
      slots.actions ? 'model-table-row--has-actions' : '',
      selected ? 'model-table-row--selected' : '',
    ]"
    :role="clickable && interactiveContainer ? 'button' : undefined"
    :tabindex="clickable && interactiveContainer ? 0 : undefined"
    :aria-label="
      clickable && interactiveContainer
        ? (accessibilityLabel ?? `${name} — view details`)
        : undefined
    "
    :aria-describedby="barPercent != null ? footprintDescriptionId : undefined"
    :aria-current="selected ? 'true' : undefined"
    :data-selected="selected ? 'true' : undefined"
    data-test="model-table-row"
    @click="clickable && emit('open')"
    @keydown.enter="onRowKeydown"
    @keydown.space="onRowKeydown"
  >
    <div class="model-table-row__identity">
      <Tooltip v-if="loaded !== undefined" :text="loaded ? 'On GPU' : 'Cold'" class="shrink-0">
        <span
          class="block w-3 text-center font-mono text-xs"
          :class="loaded ? 'text-star' : 'text-transparent'"
          role="img"
          :aria-label="loaded ? 'On GPU' : 'Cold'"
          >★</span
        >
      </Tooltip>
      <SourceGlyph :source="source" class="shrink-0 text-fg-dim" />
      <div class="model-table-row__name">
        <button
          v-if="clickable"
          type="button"
          class="block max-w-full truncate text-left text-sm font-medium text-fg transition-colors duration-100 hover:text-accent"
          :title="`${name} — view details`"
          :aria-label="
            !interactiveContainer ? (accessibilityLabel ?? `${name} — view details`) : undefined
          "
          data-test="row-title"
          @click.stop="emit('open')"
        >
          {{ name }}
        </button>
        <span
          v-else
          class="block truncate text-sm font-medium text-fg"
          :title="name"
          data-test="row-title"
        >
          {{ name }}
        </span>
        <span
          v-if="showId || quant || familyChip"
          class="flex min-w-0 items-center gap-1.5 font-mono text-micro text-fg-dim"
        >
          <span v-if="showId" class="truncate" data-test="row-id">{{ id }}</span>
          <span v-if="quant" class="shrink-0 text-fg-2">{{ quant }}</span>
          <span v-if="familyChip" class="shrink-0 whitespace-nowrap" data-test="row-family">{{
            familyChip
          }}</span>
        </span>
      </div>
      <span
        v-for="label in hostLabels"
        :key="label"
        data-test="installed-host"
        class="flex shrink-0 items-center gap-1.5 font-mono text-xs text-fg-dim"
      >
        <span class="h-1.5 w-1.5 rounded-full bg-success" aria-hidden="true" />
        {{ label }}
      </span>
      <Tooltip v-if="pageUrl" text="Open model page" class="shrink-0">
        <button
          type="button"
          class="text-fg-dim transition-colors duration-100 hover:text-fg"
          :aria-label="`Open ${name} model page`"
          data-test="model-page-link"
          @click.stop="pageUrl && void openExternal(pageUrl)"
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
      </Tooltip>

      <slot name="meta" />
    </div>

    <span
      v-if="note"
      class="model-table-row__note truncate text-xs text-fg-2"
      :title="note"
      data-test="row-note"
    >
      {{ note }}
    </span>

    <!-- Primary weights and full runtime footprint are deliberately separate
         lines: the latter includes shared encoders/VAEs (SIZE vs FETCH on
         catalog rows). The grid owns this column so it cannot overlap the
         identity metadata; narrow rows hide only the secondary comparison. -->
    <div class="model-table-row__footprint">
      <ModelFootprintBar
        v-if="barPercent != null"
        :percent="barPercent"
        :size-label="sizeSecondary ?? sizePrimary"
        :description-id="footprintDescriptionId"
      />
      <span class="min-w-0 max-w-40 shrink text-right" data-test="row-sizes">
        <span
          v-if="sizePrimary"
          class="block truncate font-mono text-xs text-fg-2"
          :title="sizePrimary"
        >
          {{ sizePrimary }}
        </span>
        <span
          v-if="sizeSecondary"
          class="block truncate font-mono text-micro text-fg-dim"
          :title="sizeSecondary"
        >
          {{ sizeSecondary }}
        </span>
      </span>
    </div>

    <!-- Clicks on parent-provided actions must never open the row. -->
    <div v-if="slots.actions" class="flex shrink-0 items-center gap-1.5" @click.stop>
      <slot name="actions" />
    </div>
  </div>
</template>

<style scoped>
.model-table-row {
  container-type: inline-size;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 12px;
  min-width: 0;
  min-height: 52px;
}

.model-table-row--has-note {
  grid-template-columns: minmax(0, 1fr) minmax(8rem, 13rem) auto;
}

.model-table-row--has-actions {
  grid-template-columns: minmax(0, 1fr) auto auto;
}

.model-table-row--has-note.model-table-row--has-actions {
  grid-template-columns: minmax(0, 1fr) minmax(8rem, 13rem) auto auto;
}

.model-table-row__identity {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 10px;
  overflow: hidden;
}

.model-table-row__name {
  display: flex;
  min-width: 48px;
  flex: 1 1 10rem;
  flex-direction: column;
  gap: 1px;
  overflow: hidden;
}

.model-table-row--selected,
.model-table-row--selected:hover {
  background: var(--mold-accent-tint);
  box-shadow: inset 3px 0 var(--mold-blue);
}

.model-table-row__footprint {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.model-table-row--has-footprint .model-table-row__footprint {
  width: clamp(9rem, 33cqw, 16rem);
}

/* At machine-card widths, protect identity and sizes first. The note and
   the footprint are secondary and return as soon as the row has room. */
@container (max-width: 30rem) {
  .model-table-row__note,
  .model-table-row__footprint :deep(.model-footprint) {
    display: none;
  }

  .model-table-row--has-note {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .model-table-row--has-note.model-table-row--has-actions {
    grid-template-columns: minmax(0, 1fr) auto auto;
  }

  .model-table-row--has-footprint .model-table-row__footprint {
    width: auto;
  }
}
</style>
