<script setup lang="ts">
import { computed, useId } from "vue";

import { familyLabel } from "@studio/lib/modelFamily";

import Tooltip from "@ui/components/Tooltip.vue";

import SourceGlyph from "../generate/SourceGlyph.vue";
import ModelFootprintBar from "./ModelFootprintBar.vue";
import { openExternal } from "../../lib/openExternal";
import type { ModelSource } from "../../lib/modelSource";

/**
 * The one row shape for models shown as a table anywhere in the app —
 * the Installed shelf, a host's detail page, and the catalog's table
 * layout all render this component so model rows stay visually identical.
 *
 * Purely presentational: parents own data fetching, size resolution, and
 * actions (the `#actions` slot). Column recipe, left to right: residency
 * dot · source glyph · name · host chips · quant chip · family code ·
 * page link · `#meta` · relative footprint + two-line size block · `#actions`.
 */
const props = withDefaults(
  defineProps<{
    name: string;
    source: ModelSource;
    /** GPU residency dot: true = warm, false = cold placeholder, omit = no dot column. */
    loaded?: boolean | undefined;
    hostLabels?: string[];
    quant?: string | null;
    family?: string | null;
    /** External model page; renders the link-out icon when present. */
    pageUrl?: string | null;
    /** Right-aligned size block: primary line + smaller secondary line. */
    sizePrimary?: string | null;
    sizeSecondary?: string | null;
    /** 0–100 fill for the relative-usage bar; omit to hide the bar. */
    barPercent?: number | null;
    /** Row and name become buttons that emit `open` (catalog detail drawer). */
    clickable?: boolean;
    /** The row currently backing an open detail drawer. */
    selected?: boolean;
    /** Rich accessible name when visible metadata adds context to the row. */
    accessibilityLabel?: string | null;
  }>(),
  {
    loaded: undefined,
    hostLabels: () => [],
    quant: null,
    family: null,
    pageUrl: null,
    sizePrimary: null,
    sizeSecondary: null,
    barPercent: null,
    clickable: false,
    selected: false,
    accessibilityLabel: null,
  },
);

const emit = defineEmits<{ (e: "open"): void }>();
const footprintDescriptionId = `model-footprint-${useId()}`;

/**
 * The chip shows the family's name, not its wire slug (#806) — a Wan row read
 * "wan" here while web read "Wan Video". No width concern: the longest label
 * ("Qwen Image Edit") is exactly as long as the slug it replaces, and every
 * other label is the same length or shorter. Only wan grows.
 */
const familyChip = computed(() => (props.family ? familyLabel(props.family) : ""));

/**
 * Enter/Space on a clickable row opens it, but keydown bubbles — a keypress
 * on an inner control (Pull, the link-out) reaches here too. Only act when
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
    class="model-table-row rounded-control transition-colors duration-100 hover:bg-bath"
    :class="[
      clickable ? 'cursor-pointer focus-visible:outline-2 focus-visible:outline-safelight' : '',
      barPercent != null ? 'model-table-row--has-footprint' : '',
      $slots.actions ? 'model-table-row--has-actions' : '',
      selected ? 'model-table-row--selected' : '',
    ]"
    :role="clickable ? 'button' : undefined"
    :tabindex="clickable ? 0 : undefined"
    :aria-label="clickable ? (accessibilityLabel ?? `${name} — view details`) : undefined"
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
          class="h-1.5 w-1.5 rounded-full"
          :class="loaded ? 'bg-safelight' : 'bg-transparent'"
          role="img"
          :aria-label="loaded ? 'On GPU' : 'Cold'"
        />
      </Tooltip>
      <SourceGlyph :source="source" class="shrink-0 text-ink-3" />
      <button
        v-if="clickable"
        type="button"
        class="model-table-row__title truncate text-left text-body text-ink transition-colors duration-100 hover:text-safelight"
        :title="`${name} — view details`"
        data-test="row-title"
        @click.stop="emit('open')"
      >
        {{ name }}
      </button>
      <span
        v-else
        class="model-table-row__title truncate text-body text-ink"
        :title="name"
        data-test="row-title"
      >
        {{ name }}
      </span>
      <span
        v-for="label in hostLabels"
        :key="label"
        data-test="installed-host"
        class="border-edge data-mono shrink-0 rounded-full border px-1.5 text-caption text-ink-3"
      >
        {{ label }}
      </span>
      <span
        v-if="quant"
        class="border-edge data-mono shrink-0 rounded-full border px-1.5 text-caption text-ink-2"
      >
        {{ quant }}
      </span>
      <span v-if="familyChip" class="edge-code shrink-0" data-test="row-family">{{
        familyChip
      }}</span>
      <Tooltip v-if="pageUrl" text="Open model page" class="shrink-0">
        <button
          type="button"
          class="text-ink-3 transition-colors duration-100 hover:text-ink"
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
          class="data-mono block truncate text-caption text-ink-2"
          :title="sizePrimary"
        >
          {{ sizePrimary }}
        </span>
        <span
          v-if="sizeSecondary"
          class="data-mono block truncate text-[10px] text-ink-3"
          :title="sizeSecondary"
        >
          {{ sizeSecondary }}
        </span>
      </span>
    </div>

    <!-- Clicks on parent-provided actions must never open the row. -->
    <div v-if="$slots.actions" class="flex shrink-0 items-center gap-1" @click.stop>
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
  gap: 8px;
  min-width: 0;
}

.model-table-row--has-actions {
  grid-template-columns: minmax(0, 1fr) auto auto;
}

.model-table-row__identity {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
  overflow: hidden;
}

.model-table-row__title {
  min-width: 48px;
  flex: 1 1 10rem;
}

.model-table-row--selected,
.model-table-row--selected:hover {
  background: var(--sel-bg);
  box-shadow: inset 3px 0 var(--sel-border);
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

/* At host-card widths, protect metadata and sizes first. The footprint is a
   secondary comparison and returns as soon as its own row has enough room. */
@container (max-width: 30rem) {
  .model-table-row__footprint :deep(.model-footprint) {
    display: none;
  }

  .model-table-row--has-footprint .model-table-row__footprint {
    width: auto;
  }
}
</style>
