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
 * `CatalogTableRow`, the shared model-row shape). README §04 catalog card:
 * name with the size in mono, the id in mono, a plain note, then the
 * machines that already have it and **Get it**. Owns its lazy size
 * resolution (HF summary rows arrive without `size_bytes`). Clicking the
 * card body or title emits `open` — the in-app detail drawer; the
 * external-link icon stays a secondary action out to huggingface.co /
 * civitai.com. Install state and the Get it action stay with the parent
 * tab via the `pull` event.
 */
const props = withDefaults(
  defineProps<{
    entry: CatalogEntry;
    pulling: boolean;
    /** Labels of hosts that have this model installed (unified-list tags). */
    hosts?: string[];
    /** Whether some machine can still receive this model. A ready card
     *  keeps its Get it action until every reachable machine has it; the
     *  parent decides, because only it knows the fleet. */
    installable?: boolean | undefined;
    /** The card currently backing the open detail drawer. */
    selected?: boolean;
    /** Batch-download checkbox state. */
    selectable?: boolean;
    checked?: boolean;
    /** This machine's own answer for the model behind the row, resolved by
     *  the parent through `@studio/lib/modelRuntimeAvailability`. `null`
     *  means runnable or unknown; nothing here derives it from the family,
     *  and it never disables Get it — the model is downloadable (#1276). */
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
 *  truth: ready here means there is nothing left to get. */
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
    class="catalog-card-contained flex cursor-pointer flex-col rounded-control border border-border bg-panel transition-colors duration-100 hover:border-border-focus"
    :class="selected ? 'catalog-card-contained--selected' : ''"
    data-test="catalog-card"
    data-layout="grid"
    :aria-label="accessibilityLabel"
    :aria-current="selected ? 'true' : undefined"
    :data-selected="selected ? 'true' : undefined"
    @click="emit('open', entry)"
  >
    <label
      class="catalog-card-checkbox absolute left-2 top-2 z-10 flex h-7 w-7 items-center justify-center rounded-control border border-border bg-bg/95"
      :title="selectable ? 'Select style to get in a batch' : 'No machine can take it'"
      @click.stop
    >
      <input
        type="checkbox"
        class="h-4 w-4 accent-accent"
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
      class="relative h-32 w-full overflow-hidden rounded-t-control"
    >
      <div v-if="!thumbLoaded" class="ms-shimmer absolute inset-0" aria-hidden="true" />
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

    <div class="flex min-h-32 flex-1 flex-col gap-2 p-3.5">
      <div class="flex min-w-0 items-start justify-between gap-2">
        <span class="flex min-w-0 items-center gap-1.5">
          <SourceGlyph :source="glyphSource" :size="16" class="text-fg-dim" />
          <button
            type="button"
            class="truncate text-left text-sm font-semibold text-fg transition-colors duration-100 hover:text-accent"
            :title="`${displayName} — view details`"
            data-test="card-title"
            @click.stop="emit('open', entry)"
          >
            {{ displayName }}
          </button>
        </span>
        <span class="flex shrink-0 items-center gap-1.5">
          <button
            v-if="pageUrl"
            type="button"
            class="text-fg-dim transition-colors duration-100 hover:text-fg"
            :aria-label="`Open ${displayName}'s page`"
            title="Open the style's page"
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
          <span
            v-if="sizePending"
            class="font-mono text-micro text-fg-dim"
            data-test="size-skeleton"
            aria-label="Resolving size"
          >
            SIZE …
          </span>
          <span v-else-if="hasSizeLine" class="font-mono text-micro text-fg-dim">
            {{ catalogSizeLabel(sizeInfo) }}
          </span>
        </span>
      </div>

      <div class="flex min-w-0 items-center gap-2 font-mono text-micro text-fg-dim">
        <span v-if="entry.name !== displayName" class="truncate" :title="entry.name">
          {{ entry.name }}
        </span>
        <span class="shrink-0">{{ entry.family }}</span>
        <span v-if="counts" class="ml-auto shrink-0">{{ counts }}</span>
      </div>

      <ModelMetadataBadges
        :kind="entry.kind"
        :family="entry.family"
        :modality="entry.modality ?? null"
        :nsfw="entry.nsfw"
        :data-test="entry.nsfw ? 'nsfw-tag' : undefined"
      />

      <p
        v-if="description"
        class="line-clamp-2 text-xs leading-relaxed text-fg-2"
        data-test="catalog-description"
        :title="description"
      >
        {{ description }}
      </p>
      <p v-if="entry.author" class="truncate text-micro text-fg-dim">by {{ entry.author }}</p>
      <p v-if="hasSizeLine && catalogFetchCaption(sizeInfo)" class="text-micro text-fg-dim">
        {{ catalogFetchCaption(sizeInfo) }}
      </p>

      <div class="mt-auto flex min-w-0 items-center gap-2 pt-1">
        <span
          v-if="entry.installed"
          class="shrink-0 font-mono text-micro text-success"
          data-test="catalog-ready"
          >● ready</span
        >
        <span
          v-for="host in hosts"
          :key="host"
          class="shrink-0 font-mono text-micro text-fg-dim"
          data-test="installed-host"
        >
          {{ host }}
        </span>
        <span
          v-if="props.runtimeNotice"
          data-test="runtime-unavailable-badge"
          class="shrink-0 font-mono text-micro text-fg-dim"
          :title="props.runtimeNotice.message"
        >
          {{ RUNTIME_UNAVAILABLE_BADGE }}
        </span>
        <span class="flex-1" />
        <button
          v-if="showAction"
          type="button"
          data-test="pull"
          class="ms-toolbar-button ms-toolbar-button--on disabled:opacity-50"
          :disabled="pulling"
          @click.stop="emit('pull', entry)"
        >
          {{ pulling ? "Getting it…" : catalogPullLabel(sizeInfo, "Get it") }}
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
  border-color: var(--mold-blue);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
}

.catalog-card-contained--selected,
.catalog-card-contained--selected:hover {
  border-color: var(--mold-blue);
  background: var(--mold-accent-tint);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
}

/* Model previews are overwhelmingly portrait subjects, and a centred cover
   crop of a portrait in this short landscape box slices heads off at the
   chin. Bias the frame upward so the subject survives the crop. */
.catalog-thumb {
  object-position: 50% 28%;
}
</style>
