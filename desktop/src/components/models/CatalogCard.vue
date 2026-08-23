<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import { modelKindLabel, modelKindValue } from "@studio/lib/modelMetadata";
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
import { RUNTIME_UNAVAILABLE_BADGE } from "@studio/lib/modelRuntimeAvailability";
import type { ModelRuntimeNotice } from "@studio/lib/modelRuntimeAvailability";
import type { ModelSource } from "../../lib/modelSource";
import type { CatalogEntry } from "../../lib/api/types";

/**
 * One catalog search result in the grid layout (the table layout uses
 * `CatalogTableRow`, the shared model-row shape). Owns its lazy size
 * resolution (HF summary rows arrive without `size_bytes`). Clicking the
 * card body or title emits `open` — the in-app detail drawer; the
 * external-link icon stays a secondary action out to huggingface.co /
 * civitai.com. Install state and the Pull action stay with the parent tab
 * via the `pull` event.
 */
const props = withDefaults(
  defineProps<{
    entry: CatalogEntry;
    pulling: boolean;
    /** Labels of hosts that have this model installed (unified-list tags). */
    hosts?: string[];
    /** Whether some machine can still receive this model. An installed card
     *  keeps its Pull action until every reachable machine has it; the parent
     *  decides, because only it knows the fleet. */
    installable?: boolean | undefined;
    /** The card currently backing the open detail drawer. */
    selected?: boolean;
    /** Batch-download checkbox state. */
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
const displayName = computed(() => props.entry.display_name ?? props.entry.name);
const kindValue = computed(() => modelKindValue(props.entry));
const description = computed(() => props.entry.description?.trim() || null);
const accessibilityLabel = computed(
  () =>
    `${displayName.value} — ${modelKindLabel(kindValue.value)}${
      props.entry.nsfw ? ", 18+ NSFW" : ""
    } — view details`,
);
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
  <article
    class="catalog-card-contained border-edge flex cursor-pointer flex-col rounded-chrome border bg-bath transition-colors duration-150 hover:bg-bench"
    :class="selected ? 'catalog-card-contained--selected' : ''"
    data-test="catalog-card"
    data-layout="grid"
    :aria-label="accessibilityLabel"
    :aria-current="selected ? 'true' : undefined"
    :data-selected="selected ? 'true' : undefined"
    @click="emit('open', entry)"
  >
    <label
      class="catalog-card-checkbox border-edge absolute left-2 top-2 z-10 flex h-7 w-7 items-center justify-center rounded-control border bg-bench/95"
      :title="selectable ? 'Select model for batch download' : 'No download target available'"
      @click.stop
    >
      <input
        type="checkbox"
        class="h-4 w-4 accent-[var(--safelight)]"
        :checked="checked"
        :disabled="!selectable"
        :aria-label="`Select ${displayName}`"
        data-test="catalog-select"
        @change="emit('toggle-select', entry, ($event.target as HTMLInputElement).checked)"
      />
    </label>
    <!-- Civitai preview image (public URL); shimmer while loading and use a
         local family mark when no custom image is available. -->
    <div
      v-if="thumbnailUrl && !thumbFailed"
      class="relative h-32 w-full overflow-hidden rounded-t-chrome"
    >
      <div v-if="!thumbLoaded" class="grain-shimmer absolute inset-0" aria-hidden="true" />
      <img
        :src="thumbnailUrl"
        alt=""
        loading="lazy"
        decoding="async"
        fetchpriority="low"
        class="catalog-thumb h-full w-full object-cover"
        @load="thumbLoaded = true"
        @error="thumbFailed = true"
      />
    </div>
    <ModelFamilyPlaceholder v-else :family="entry.family" layout="grid" />

    <div class="flex min-h-32 flex-1 flex-col gap-1.5 p-3">
      <div class="flex min-w-0 items-start justify-between gap-2">
        <span class="flex min-w-0 items-center gap-1.5">
          <SourceGlyph :source="glyphSource" :size="16" class="text-ink-3" />
          <button
            type="button"
            class="truncate text-left text-body text-ink transition-colors duration-100 hover:text-safelight"
            :title="`${entry.display_name ?? entry.name} — view details`"
            data-test="card-title"
            @click.stop="emit('open', entry)"
          >
            {{ entry.display_name ?? entry.name }}
          </button>
        </span>
        <span class="flex shrink-0 items-center gap-1.5">
          <button
            v-if="pageUrl"
            type="button"
            class="text-ink-3 transition-colors duration-100 hover:text-ink"
            :aria-label="`Open ${entry.display_name ?? entry.name} model page`"
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

      <ModelMetadataBadges
        :kind="entry.kind"
        :family="entry.family"
        :modality="entry.modality ?? null"
        :nsfw="entry.nsfw"
        :data-test="entry.nsfw ? 'nsfw-tag' : undefined"
      />

      <div v-if="entry.author || counts" class="flex items-center gap-2">
        <span v-if="entry.author" class="truncate text-caption text-ink-3">{{ entry.author }}</span>
        <span v-if="counts" class="data-mono ml-auto shrink-0 text-caption text-ink-3">
          {{ counts }}
        </span>
      </div>

      <p
        v-if="description"
        class="line-clamp-2 text-caption leading-snug text-ink-2"
        data-test="catalog-description"
        :title="description"
      >
        {{ description }}
      </p>

      <div v-if="hosts?.length" class="flex flex-wrap items-center gap-1">
        <span
          v-for="host in hosts"
          :key="host"
          class="border-edge data-mono rounded-full border px-1.5 text-caption text-ink-2"
          data-test="installed-host"
        >
          {{ host }}
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
      <div v-else-if="hasSizeLine" class="text-caption">
        <div class="data-mono text-ink-2">{{ catalogSizeLabel(sizeInfo) }}</div>
        <div v-if="catalogFetchCaption(sizeInfo)" class="text-ink-3">
          {{ catalogFetchCaption(sizeInfo) }}
        </div>
      </div>

      <div class="mt-auto flex shrink-0 items-center justify-end gap-2 pt-1">
        <span v-if="entry.installed" class="data-mono text-caption text-halide">● installed</span>
        <span
          v-if="props.runtimeNotice"
          data-test="runtime-unavailable-badge"
          class="data-mono text-caption text-ink-3"
          :title="props.runtimeNotice.message"
        >
          {{ RUNTIME_UNAVAILABLE_BADGE }}
        </span>
        <button
          v-if="showAction"
          type="button"
          data-test="pull"
          class="border-edge h-7 rounded-control border px-2.5 text-caption text-safelight transition-colors duration-150 hover:border-safelight active:translate-y-px disabled:opacity-50"
          :disabled="pulling"
          @click.stop="emit('pull', entry)"
        >
          {{ pulling ? "Pulling…" : catalogPullLabel(sizeInfo) }}
        </button>
      </div>
    </div>
  </article>
</template>

<style scoped>
.catalog-card-contained {
  position: relative;
  contain: layout paint style;
}

.catalog-card-contained:has(.catalog-card-checkbox input:checked) {
  border-color: var(--safelight);
  box-shadow: inset 0 0 0 1px var(--safelight);
}

.catalog-card-contained--selected,
.catalog-card-contained--selected:hover {
  border-color: var(--sel-border);
  background: var(--sel-bg);
  box-shadow: inset 0 0 0 1px var(--sel-border);
}

/* Model previews are overwhelmingly portrait subjects, and a centred cover
   crop of a portrait in this short landscape box slices heads off at the
   chin. Bias the frame upward so the subject survives the crop. */
.catalog-thumb {
  object-position: 50% 28%;
}
</style>
