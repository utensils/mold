<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import { modelKindLabel, modelKindValue } from "@studio/lib/modelMetadata";
import ModelTableRow from "./ModelTableRow.vue";
import { catalogPageUrl, catalogSizeInfo } from "../../lib/catalog";
import { resolveEntrySize } from "../../lib/catalogSizes";
import { formatCount, formatGB } from "../../lib/format";
import { RUNTIME_UNAVAILABLE_BADGE } from "@studio/lib/modelRuntimeAvailability";
import type { ModelRuntimeNotice } from "@studio/lib/modelRuntimeAvailability";
import type { ModelSource } from "../../lib/modelSource";
import type { CatalogEntry } from "../../lib/api/types";

/**
 * One catalog search result in the table layout — the shared model-row
 * shape with catalog specifics on top: no preview image (that's the grid
 * card's job), lazy size resolution (HF summary rows arrive without
 * `size_bytes`), SIZE/FETCH as the two size lines, and Get it / ready
 * state in the actions column — Get it is the filled accent button the card
 * uses, one verb and one treatment. Clicking the row opens the detail drawer.
 */
const props = withDefaults(
  defineProps<{
    entry: CatalogEntry;
    pulling: boolean;
    /** Labels of hosts that have this model installed — the unified list's
     *  "you have this" indicator. */
    hosts?: string[];
    /** Whether some machine can still receive this model. An installed row
     *  keeps its Pull action until every reachable machine has it; the parent
     *  decides, because only it knows the fleet. */
    installable?: boolean | undefined;
    /** The row currently backing the open detail drawer. */
    selected?: boolean;
    selectable?: boolean;
    checked?: boolean;
    /** This machine's own answer for the model behind the row, resolved by
     *  the parent through `@studio/lib/modelRuntimeAvailability`. `null`
     *  means runnable or unknown; nothing here derives it from the family,
     *  and it never disables Pull — the model is downloadable (#1276). */
    runtimeNotice?: ModelRuntimeNotice | null;
  }>(),
  // Explicit so Vue's boolean casting doesn't turn "not supplied" into false.
  {
    installable: undefined,
    selected: false,
    selectable: true,
    checked: false,
    runtimeNotice: null,
  },
);
const emit = defineEmits<{
  (e: "pull", entry: CatalogEntry): void;
  (e: "open", entry: CatalogEntry): void;
  (e: "toggle-select", entry: CatalogEntry, checked: boolean): void;
}>();

const glyphSource = computed<ModelSource>(() =>
  props.entry.source === "civitai" ? "civitai" : "hf",
);
/** Without an explicit answer from the parent, fall back to single-machine
 *  truth: installed here means there is nothing left to pull. */
const showAction = computed(() => props.installable ?? !props.entry.installed);
const pageUrl = computed(() => catalogPageUrl(props.entry));
const kindValue = computed(() => modelKindValue(props.entry));
const accessibilityLabel = computed(
  () =>
    `${props.entry.display_name ?? props.entry.name} — ${modelKindLabel(kindValue.value)}${
      props.entry.nsfw ? ", 18+ NSFW" : ""
    } — view details`,
);

/** `undefined` = still resolving (skeleton); `number | null` = resolved. */
const resolvedBytes = ref<number | null | undefined>(props.entry.size_bytes ?? undefined);
onMounted(() => {
  if (resolvedBytes.value !== undefined) return;
  void resolveEntrySize(props.entry).then((bytes) => {
    resolvedBytes.value = bytes;
  });
});

const sizeInfo = computed(() =>
  catalogSizeInfo({ ...props.entry, size_bytes: resolvedBytes.value ?? null }),
);
/** SIZE = primary weights; unresolvable sizes omit the line, no "SIZE —" noise. */
const sizePrimary = computed(() => {
  if (resolvedBytes.value === undefined) return "SIZE …";
  const weights = sizeInfo.value.weightsBytes;
  return weights != null ? `SIZE ${formatGB(weights)}` : null;
});
/** FETCH = the honest total download, shown only when shared deps add to it. */
const sizeSecondary = computed(() => {
  if (resolvedBytes.value === undefined) return null;
  const info = sizeInfo.value;
  return info.differs && info.fetchBytes != null ? `FETCH ${formatGB(info.fetchBytes)}` : null;
});

const counts = computed(() => {
  const parts: string[] = [];
  const downloads = props.entry.download_count ?? 0;
  const likes = props.entry.likes ?? 0;
  if (downloads > 0) parts.push(`↓ ${formatCount(downloads)}`);
  if (likes > 0) parts.push(`♥ ${formatCount(likes)}`);
  return parts.join(" · ");
});
</script>

<template>
  <ModelTableRow
    :name="entry.display_name ?? entry.name"
    :source="glyphSource"
    :family="entry.family"
    :host-labels="hosts ?? []"
    :page-url="pageUrl"
    :size-primary="sizePrimary"
    :size-secondary="sizeSecondary"
    :accessibility-label="accessibilityLabel"
    :selected="selected"
    clickable
    :interactive-container="false"
    data-test="catalog-table-row"
    @open="emit('open', entry)"
  >
    <template #meta>
      <label
        class="flex h-6 w-6 shrink-0 items-center justify-center"
        :title="selectable ? 'Select style to get in a batch' : 'No machine can take it'"
        @click.stop
      >
        <input
          type="checkbox"
          class="h-4 w-4 accent-accent"
          :checked="checked"
          :disabled="!selectable"
          :aria-label="`Select ${entry.display_name ?? entry.name}`"
          data-test="catalog-select"
          @change="emit('toggle-select', entry, ($event.target as HTMLInputElement).checked)"
        />
      </label>
      <ModelMetadataBadges
        :kind="entry.kind"
        :family="entry.family"
        :modality="entry.modality ?? null"
        :nsfw="entry.nsfw"
        :show-modality="false"
        :data-test="entry.nsfw ? 'nsfw-tag' : undefined"
      />
      <span v-if="entry.author" class="min-w-0 shrink truncate text-micro text-fg-dim">
        {{ entry.author }}
      </span>
      <span v-if="counts" class="font-mono shrink-0 text-micro text-fg-dim">{{ counts }}</span>
    </template>
    <template #actions>
      <span v-if="entry.installed" class="font-mono text-micro text-success">● ready</span>
      <span
        v-if="props.runtimeNotice"
        data-test="runtime-unavailable-badge"
        class="font-mono text-micro text-fg-dim"
        :title="props.runtimeNotice.message"
      >
        {{ RUNTIME_UNAVAILABLE_BADGE }}
      </span>
      <button
        v-if="showAction"
        type="button"
        data-test="pull"
        class="ms-toolbar-button ms-toolbar-button--on disabled:opacity-50"
        :disabled="pulling"
        @click.stop="emit('pull', entry)"
      >
        {{ pulling ? "Getting it…" : "Get it" }}
      </button>
    </template>
  </ModelTableRow>
</template>
