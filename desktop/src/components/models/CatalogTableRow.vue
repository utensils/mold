<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import ModelTableRow from "./ModelTableRow.vue";
import { catalogPageUrl, catalogSizeInfo } from "../../lib/catalog";
import { resolveEntrySize } from "../../lib/catalogSizes";
import { formatCount, formatGB } from "../../lib/format";
import type { ModelSource } from "../../lib/modelSource";
import type { CatalogEntry } from "../../lib/api/types";

/**
 * One catalog search result in the table layout — the shared model-row
 * shape with catalog specifics on top: no preview image (that's the grid
 * card's job), lazy size resolution (HF summary rows arrive without
 * `size_bytes`), SIZE/FETCH as the two size lines, and Pull / installed
 * state in the actions column. Clicking the row opens the detail drawer.
 */
const props = defineProps<{ entry: CatalogEntry; pulling: boolean }>();
const emit = defineEmits<{
  (e: "pull", entry: CatalogEntry): void;
  (e: "open", entry: CatalogEntry): void;
}>();

const glyphSource = computed<ModelSource>(() =>
  props.entry.source === "civitai" ? "civitai" : "hf",
);
const pageUrl = computed(() => catalogPageUrl(props.entry));

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
    :name="entry.name"
    :source="glyphSource"
    :family="entry.family"
    :page-url="pageUrl"
    :size-primary="sizePrimary"
    :size-secondary="sizeSecondary"
    clickable
    data-test="catalog-table-row"
    @open="emit('open', entry)"
  >
    <template #meta>
      <span v-if="entry.author" class="min-w-0 shrink truncate text-caption text-ink-3">
        {{ entry.author }}
      </span>
      <span v-if="counts" class="data-mono shrink-0 text-caption text-ink-3">{{ counts }}</span>
    </template>
    <template #actions>
      <span v-if="entry.installed" class="data-mono text-caption text-halide">● installed</span>
      <button
        v-else
        type="button"
        data-test="pull"
        class="border-edge h-7 rounded-control border px-2.5 text-caption text-safelight transition-colors duration-150 hover:border-safelight active:translate-y-px disabled:opacity-50"
        :disabled="pulling"
        @click.stop="emit('pull', entry)"
      >
        {{ pulling ? "Pulling…" : "Pull" }}
      </button>
    </template>
  </ModelTableRow>
</template>
