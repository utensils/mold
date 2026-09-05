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
 * residency star · name stacked over a mono line carrying the source mark,
 * the id and the family · page link · `#meta` · a plain one-line note ·
 * relative footprint + size · machines (dot + mono name) · `#actions`.
 *
 * Every cell is a grid track, and a list that wants one axis for all its rows
 * sets `--model-row-columns` on an ancestor (the Styles shelf does, under a
 * mono header on the same template). Without it each row sizes its own
 * tracks, which is what a machine card's narrow embedding wants. A PINNED
 * axis fixes the track COUNT, so a row that skips a cell shifts every column
 * after it one track left — which is why `noteColumn` exists: the parent says
 * the table has a Good-for column and the cell is emitted empty rather than
 * dropped.
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
    /** Give the machine chips a column of their own instead of trailing the
     * name — what a table with a MACHINE header needs, and only that. */
    machinesColumn?: boolean;
    family?: string | null;
    /** External model page; renders the link-out icon when present. */
    pageUrl?: string | null;
    /** One line in plain words — what the style is good for. */
    note?: string | null;
    /** The table pins a note track: emit the cell on every row, empty or not,
     * so a row without a description cannot shift the columns after it. */
    noteColumn?: boolean;
    /** How long the style typically takes (`~20s`), read from the prints
     * already made with it; null when nobody has timed it. */
    speed?: string | null;
    /** The table pins a speed track: emit the cell on every row, empty or
     * not, for the same reason as `noteColumn`. */
    speedColumn?: boolean;
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
    machinesColumn: false,
    family: null,
    pageUrl: null,
    note: null,
    noteColumn: false,
    speed: null,
    speedColumn: false,
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
 * The mono sub-line under the name. The source mark rides it so the identity
 * cell starts at the name rather than behind a star + a 12px colour glyph +
 * two 10px gaps; when a row has neither an id nor a family there is no line to
 * ride, and the mark sits beside the title rather than inventing one.
 */
const showMonoLine = computed(() => showId.value || Boolean(familyChip.value));
/** One cell per pinned track: the note is emitted whenever its track exists. */
const showNoteCell = computed(() => props.noteColumn || props.note != null);
const showSpeedCell = computed(() => props.speedColumn || props.speed != null);

/**
 * The machines cell is ONE line — the mock's row is 52px and every other cell
 * obeys that. The rest of the machines are named in the overflow chip's tip.
 */
const MACHINES_SHOWN = 1;
const machinesShown = computed(() => props.hostLabels.slice(0, MACHINES_SHOWN));
const machinesHidden = computed(() => props.hostLabels.slice(MACHINES_SHOWN));

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
      showNoteCell ? 'model-table-row--has-note' : '',
      showSpeedCell ? 'model-table-row--has-speed' : '',
      machinesColumn ? 'model-table-row--has-machines' : '',
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
          v-if="showMonoLine"
          class="flex min-w-0 items-center gap-1.5 font-mono text-micro text-fg-dim"
          data-test="row-mono-line"
        >
          <SourceGlyph :source="source" :size="10" class="shrink-0 text-fg-dim" />
          <span v-if="showId" class="truncate" data-test="row-id">{{ id }}</span>
          <span v-if="familyChip" class="shrink-0 whitespace-nowrap" data-test="row-family">{{
            familyChip
          }}</span>
        </span>
      </div>
      <!-- No mono line to ride: the mark keeps its place beside the name
           rather than adding a second line to a one-line row. -->
      <SourceGlyph v-if="!showMonoLine" :source="source" :size="10" class="shrink-0 text-fg-dim" />
      <template v-if="!machinesColumn">
        <span
          v-for="label in hostLabels"
          :key="label"
          data-test="installed-host"
          class="flex shrink-0 items-center gap-1.5 font-mono text-xs text-fg-dim"
        >
          <span class="h-1.5 w-1.5 rounded-full bg-success" aria-hidden="true" />
          {{ label }}
        </span>
      </template>
      <Tooltip v-if="pageUrl" text="Open the style's page" class="shrink-0">
        <button
          type="button"
          class="text-fg-dim transition-colors duration-100 hover:text-fg"
          :aria-label="`Open ${name}'s page`"
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
      v-if="showNoteCell"
      class="model-table-row__note truncate text-xs text-fg-2"
      :title="note ?? undefined"
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

    <!-- Typical time from the prints already made with this style. -->
    <span
      v-if="showSpeedCell"
      class="model-table-row__speed truncate text-right font-mono text-xs text-fg-2"
      :title="speed ? 'Typical time over your recent prints with this style' : undefined"
      data-test="row-speed"
    >
      {{ speed }}
    </span>

    <div v-if="machinesColumn" class="model-table-row__machines">
      <span
        v-for="label in machinesShown"
        :key="label"
        data-test="installed-host"
        class="flex min-w-0 items-center gap-1.5 font-mono text-xs text-fg-dim"
      >
        <span class="h-1.5 w-1.5 shrink-0 rounded-full bg-success" aria-hidden="true" />
        <span class="truncate" :title="label">{{ label }}</span>
      </span>
      <span
        v-if="machinesHidden.length"
        data-test="installed-host-more"
        class="shrink-0 font-mono text-micro text-fg-dim"
        :title="machinesHidden.join(', ')"
        >+{{ machinesHidden.length }}</span
      >
    </div>

    <!-- Clicks on parent-provided actions must never open the row. -->
    <div v-if="slots.actions" class="model-table-row__actions" @click.stop>
      <slot name="actions" />
    </div>
  </div>
</template>

<style scoped>
/* One track per cell. Absent cells contribute an empty custom property, so a
   row's own template collapses to what it renders while a list that sets
   --model-row-columns pins every row (and its header) to one axis. */
.model-table-row {
  --mtr-note: ;
  --mtr-speed: ;
  --mtr-machines: ;
  --mtr-actions: ;
  container-type: inline-size;
  display: grid;
  grid-template-columns: var(
    --model-row-columns,
    minmax(0, 1fr) var(--mtr-note) auto var(--mtr-speed) var(--mtr-machines) var(--mtr-actions)
  );
  align-items: center;
  gap: 12px;
  min-width: 0;
  min-height: 52px;
}

.model-table-row--has-note {
  --mtr-note: minmax(8rem, 13rem);
}

.model-table-row--has-speed {
  --mtr-speed: auto;
}

.model-table-row--has-machines {
  --mtr-machines: auto;
}

.model-table-row--has-actions {
  --mtr-actions: auto;
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
  max-width: 100%;
}

/* One LINE, never a stack: the mock's row is 52px and a style on three
   machines used to draw three lines here. The overflow chip names the rest. */
.model-table-row__machines {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 6px;
}

.model-table-row__actions {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
}

/* At machine-card widths, protect identity and sizes first. The note and
   the footprint are secondary and return as soon as the row has room. */
@container (max-width: 30rem) {
  .model-table-row__note,
  .model-table-row__footprint :deep(.model-footprint) {
    display: none;
  }

  /* A pinned table's axis is wider than this embedding — fall back to the
     row's own tracks rather than overflowing the card. */
  .model-table-row {
    --model-row-columns: initial;
  }

  .model-table-row--has-note {
    --mtr-note: ;
  }

  .model-table-row__speed {
    display: none;
  }

  .model-table-row--has-speed {
    --mtr-speed: ;
  }

  .model-table-row--has-footprint .model-table-row__footprint {
    width: auto;
  }
}
</style>
