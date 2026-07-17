<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import SourceGlyph from "../generate/SourceGlyph.vue";
import ModelFamilyPlaceholder from "./ModelFamilyPlaceholder.vue";
import {
  catalogFetchCaption,
  catalogPageUrl,
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
} from "../../lib/catalog";
import { resolveEntrySize } from "../../lib/catalogSizes";
import { catalogThumbnailUrl } from "../../lib/catalogThumbnails";
import { formatCount } from "../../lib/format";
import { openExternal } from "../../lib/openExternal";
import type { ModelSource } from "../../lib/modelSource";
import type { CatalogEntry } from "../../lib/api/types";

/**
 * One catalog search result. Owns its lazy size resolution (HF summary
 * rows arrive without `size_bytes`). Clicking the card body or title emits
 * `open` — the in-app detail drawer; the external-link icon stays a
 * secondary action out to huggingface.co / civitai.com. Install state and
 * the Pull action stay with the parent tab via the `pull` event.
 */
const props = withDefaults(
  defineProps<{ entry: CatalogEntry; pulling: boolean; layout?: "grid" | "table" }>(),
  { layout: "grid" },
);
const emit = defineEmits<{
  (e: "pull", entry: CatalogEntry): void;
  (e: "open", entry: CatalogEntry): void;
}>();

const glyphSource = computed<ModelSource>(() =>
  props.entry.source === "civitai" ? "civitai" : "hf",
);

const pageUrl = computed(() => catalogPageUrl(props.entry));
const thumbnailUrl = computed(() =>
  props.entry.thumbnail_url ? catalogThumbnailUrl(props.entry.thumbnail_url) : null,
);

/** `undefined` = still resolving (skeleton); `number | null` = resolved. */
const resolvedBytes = ref<number | null | undefined>(props.entry.size_bytes ?? undefined);
onMounted(() => {
  if (resolvedBytes.value !== undefined) return;
  void resolveEntrySize(props.entry).then((bytes) => {
    resolvedBytes.value = bytes;
  });
});

const sizePending = computed(() => resolvedBytes.value === undefined);
const sizeInfo = computed(() =>
  catalogSizeInfo({ ...props.entry, size_bytes: resolvedBytes.value ?? null }),
);
/** Unresolvable sizes omit the line entirely — no "SIZE —" noise. */
const hasSizeLine = computed(() => !sizePending.value && resolvedBytes.value != null);

const counts = computed(() => {
  const parts: string[] = [];
  const downloads = props.entry.download_count ?? 0;
  const likes = props.entry.likes ?? 0;
  if (downloads > 0) parts.push(`↓ ${formatCount(downloads)}`);
  if (likes > 0) parts.push(`♥ ${formatCount(likes)}`);
  return parts.join(" · ");
});

const thumbLoaded = ref(false);
/** A failed thumbnail falls back to the same family mark as a missing image. */
const thumbFailed = ref(false);

function openPage(): void {
  if (pageUrl.value) void openExternal(pageUrl.value);
}
</script>

<template>
  <div
    class="catalog-card-contained border-edge cursor-pointer rounded-chrome border bg-bath transition-colors duration-150 hover:bg-bench"
    :class="layout === 'grid' ? 'flex flex-col' : 'flex min-h-16 items-center gap-3 p-2'"
    data-test="catalog-card"
    :data-layout="layout"
    @click="emit('open', entry)"
  >
    <!-- Civitai preview image (public URL); shimmer while loading and use a
         local family mark when no custom image is available. -->
    <div
      v-if="thumbnailUrl && !thumbFailed"
      :class="
        layout === 'grid'
          ? 'relative h-32 w-full overflow-hidden rounded-t-chrome'
          : 'relative h-12 w-16 shrink-0 overflow-hidden rounded-media'
      "
    >
      <div v-if="!thumbLoaded" class="grain-shimmer absolute inset-0" aria-hidden="true" />
      <img
        :src="thumbnailUrl"
        alt=""
        loading="lazy"
        decoding="async"
        fetchpriority="low"
        class="h-full w-full object-cover"
        @load="thumbLoaded = true"
        @error="thumbFailed = true"
      />
    </div>
    <ModelFamilyPlaceholder v-else :family="entry.family" :layout="layout" />

    <div :class="layout === 'grid' ? 'flex min-h-32 flex-1 flex-col gap-1.5 p-3' : 'contents'">
      <div
        class="flex min-w-0 items-start justify-between gap-2"
        :class="layout === 'table' ? 'flex-1' : ''"
      >
        <span class="flex min-w-0 items-center gap-1.5">
          <SourceGlyph :source="glyphSource" :size="16" class="text-ink-3" />
          <button
            type="button"
            class="truncate text-left text-body text-ink transition-colors duration-100 hover:text-safelight"
            :title="`${entry.name} — view details`"
            data-test="card-title"
            @click.stop="emit('open', entry)"
          >
            {{ entry.name }}
          </button>
        </span>
        <span class="flex shrink-0 items-center gap-1.5">
          <button
            v-if="pageUrl"
            type="button"
            class="text-ink-3 transition-colors duration-100 hover:text-ink"
            :aria-label="`Open ${entry.name} model page`"
            title="Open model page"
            data-test="page-link"
            @click.stop="openPage"
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
          <span class="border-edge data-mono rounded-full border px-1.5 text-caption text-ink-2">
            {{ entry.family }}
          </span>
        </span>
      </div>

      <div
        v-if="entry.author || counts"
        class="flex items-center gap-2"
        :class="layout === 'table' ? 'w-44 shrink-0' : ''"
      >
        <span v-if="entry.author" class="truncate text-caption text-ink-3">{{ entry.author }}</span>
        <span v-if="counts" class="data-mono ml-auto shrink-0 text-caption text-ink-3">
          {{ counts }}
        </span>
      </div>

      <div
        v-if="sizePending"
        class="data-mono text-caption text-ink-3"
        :class="layout === 'table' ? 'w-24 shrink-0' : ''"
        data-test="size-skeleton"
        aria-label="Resolving size"
      >
        SIZE …
      </div>
      <div
        v-else-if="hasSizeLine"
        class="text-caption"
        :class="layout === 'table' ? 'w-28 shrink-0' : ''"
      >
        <div class="data-mono text-ink-2">{{ catalogSizeLabel(sizeInfo) }}</div>
        <div v-if="catalogFetchCaption(sizeInfo) && layout === 'grid'" class="text-ink-3">
          {{ catalogFetchCaption(sizeInfo) }}
        </div>
      </div>
      <div v-else-if="layout === 'table'" class="w-28 shrink-0" />

      <div class="flex shrink-0 justify-end" :class="layout === 'grid' ? 'mt-auto pt-1' : 'w-24'">
        <span v-if="entry.installed" class="data-mono text-caption text-halide">● installed</span>
        <button
          v-else
          type="button"
          data-test="pull"
          class="border-edge h-7 rounded-control border px-2.5 text-caption text-safelight transition-colors duration-150 hover:border-safelight active:translate-y-px disabled:opacity-50"
          :disabled="pulling"
          @click.stop="emit('pull', entry)"
        >
          {{ pulling ? "Pulling…" : layout === "table" ? "Pull" : catalogPullLabel(sizeInfo) }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.catalog-card-contained {
  contain: layout paint style;
}
</style>
