<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import SourceGlyph from "../generate/SourceGlyph.vue";
import {
  catalogFetchCaption,
  catalogPageUrl,
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
} from "../../lib/catalog";
import { resolveEntrySize } from "../../lib/catalogSizes";
import { formatCount } from "../../lib/format";
import { openExternal } from "../../lib/openExternal";
import type { ModelSource } from "../../lib/modelSource";
import type { CatalogEntry } from "../../lib/api/types";

/**
 * One catalog search result. Owns its lazy size resolution (HF summary
 * rows arrive without `size_bytes`) and the link-out to the model's page
 * on huggingface.co / civitai.com; install state and the Pull action stay
 * with the parent tab via the `pull` event.
 */
const props = defineProps<{ entry: CatalogEntry; pulling: boolean }>();
const emit = defineEmits<{ (e: "pull", entry: CatalogEntry): void }>();

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
/** A failed thumbnail drops the whole image block — no broken-image box. */
const thumbFailed = ref(false);

function openPage(): void {
  if (pageUrl.value) void openExternal(pageUrl.value);
}
</script>

<template>
  <div
    class="border-edge flex flex-col gap-1.5 rounded-chrome border bg-bath p-3 transition-colors duration-100 hover:bg-bench"
    data-test="catalog-card"
  >
    <!-- Civitai preview image (public URL); shimmer placeholder while it
         loads, dropped entirely if it fails. -->
    <div
      v-if="entry.thumbnail_url && !thumbFailed"
      class="relative -mx-3 -mt-3 mb-0.5 aspect-[5/3] overflow-hidden rounded-t-chrome"
    >
      <div v-if="!thumbLoaded" class="grain-shimmer absolute inset-0" aria-hidden="true" />
      <img
        :src="entry.thumbnail_url"
        alt=""
        loading="lazy"
        class="h-full w-full object-cover"
        @load="thumbLoaded = true"
        @error="thumbFailed = true"
      />
    </div>

    <div class="flex items-start justify-between gap-2">
      <span class="flex min-w-0 items-center gap-1.5">
        <SourceGlyph :source="glyphSource" :size="16" class="text-ink-3" />
        <button
          v-if="pageUrl"
          type="button"
          class="truncate text-left text-body text-ink transition-colors duration-100 hover:text-safelight"
          :title="`${entry.name} — open model page`"
          @click="openPage"
        >
          {{ entry.name }}
        </button>
        <span v-else class="truncate text-body text-ink" :title="entry.name">{{ entry.name }}</span>
      </span>
      <span class="flex shrink-0 items-center gap-1.5">
        <button
          v-if="pageUrl"
          type="button"
          class="text-ink-3 transition-colors duration-100 hover:text-ink"
          :aria-label="`Open ${entry.name} model page`"
          title="Open model page"
          data-test="page-link"
          @click="openPage"
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

    <div v-if="entry.author || counts" class="flex items-center gap-2">
      <span v-if="entry.author" class="truncate text-caption text-ink-3">{{ entry.author }}</span>
      <span v-if="counts" class="data-mono ml-auto shrink-0 text-caption text-ink-3">
        {{ counts }}
      </span>
    </div>

    <div
      v-if="sizePending"
      class="data-mono text-caption text-ink-3"
      data-test="size-skeleton"
      aria-label="Resolving size"
    >
      SIZE …
    </div>
    <template v-else-if="hasSizeLine">
      <div class="data-mono text-caption text-ink-2">{{ catalogSizeLabel(sizeInfo) }}</div>
      <div v-if="catalogFetchCaption(sizeInfo)" class="text-caption text-ink-3">
        {{ catalogFetchCaption(sizeInfo) }}
      </div>
    </template>

    <div class="mt-1 flex justify-end">
      <span v-if="entry.installed" class="data-mono text-caption text-halide">● installed</span>
      <button
        v-else
        type="button"
        data-test="pull"
        class="border-edge h-7 rounded-control border px-2.5 text-caption text-safelight transition-colors duration-100 hover:border-safelight active:translate-y-px disabled:opacity-50"
        :disabled="pulling"
        @click="emit('pull', entry)"
      >
        {{ pulling ? "Pulling…" : catalogPullLabel(sizeInfo) }}
      </button>
    </div>
  </div>
</template>
