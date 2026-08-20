<script setup lang="ts">
/*
 * LibraryHeader — the Library's 52px top bar (V3 "Shelf"). Title, the
 * scope control (Prints | Collections | Trash with mono counts), a per-scope
 * count label, then the right cluster: thumbnail slider, media-kind control,
 * search, History / Select / Refresh. Collections hides the slider and kind
 * control; Trash keeps slider + Select and swaps Refresh for **Empty trash**.
 * Scope options are whatever the parent says the connected hosts can do —
 * with a single option the control is not rendered at all.
 */
import { computed, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import ThumbnailSizeSlider from "@ui/components/ThumbnailSizeSlider.vue";
import {
  GALLERY_THUMBNAIL_SIZE_MAX,
  GALLERY_THUMBNAIL_SIZE_MIN,
  GALLERY_THUMBNAIL_SIZE_STEP,
} from "@studio/lib/galleryThumbnailSize";
import type { GalleryKindFilter, LibraryScope } from "../../stores/gallery";

export interface ScopeCounts {
  prints: number;
  collections: number;
  trash: number;
}

const props = withDefaults(
  defineProps<{
    scope: LibraryScope;
    /** Scopes the connected hosts support, in display order. */
    scopes: readonly LibraryScope[];
    counts: ScopeCounts;
    /** Per-scope count sentence ("24 prints · 3.1 GB"). */
    countLabel: string;
    error?: string | null;
    thumbnailSize: number;
    mediaKind: GalleryKindFilter;
    kindOptions: readonly { value: GalleryKindFilter; label: string }[];
    search: string;
    selectMode: boolean;
    /** Enables **Empty trash** (Trash scope). */
    trashCount?: number;
    busy?: boolean;
  }>(),
  { error: null, trashCount: 0, busy: false },
);

const emit = defineEmits<{
  "update:scope": [scope: LibraryScope];
  "update:thumbnailSize": [px: number];
  "update:mediaKind": [kind: GalleryKindFilter];
  "update:search": [value: string];
  openHistory: [];
  toggleSelect: [];
  refresh: [];
  emptyTrash: [];
}>();

const SCOPE_LABELS: Record<LibraryScope, string> = {
  prints: "Prints",
  collections: "Collections",
  trash: "Trash",
};

const scopeOptions = computed(() =>
  props.scopes.map((scope) => ({
    value: scope,
    label: SCOPE_LABELS[scope],
    sub: String(props.counts[scope]),
  })),
);

const searchPlaceholder = computed(() =>
  props.scope === "collections"
    ? "Search collections…"
    : props.scope === "trash"
      ? "Search trash…"
      : "Search prompts…",
);

const searchEl = ref<HTMLInputElement | null>(null);
function focusSearch() {
  searchEl.value?.focus();
}
defineExpose({ focusSearch });
</script>

<template>
  <header
    class="flex h-[52px] shrink-0 items-center gap-3 border-b border-edge px-6"
    data-test="library-header"
  >
    <span class="font-display text-[17px] font-semibold text-ink" style="font-stretch: 92%">
      Library
    </span>
    <SegmentedControl
      v-if="scopes.length > 1"
      class="ms-lib-scope"
      :model-value="scope"
      :options="scopeOptions"
      label="Library scope"
      compact
      data-test="library-scope"
      @update:model-value="emit('update:scope', $event)"
    />
    <span class="data-mono text-caption text-ink-3" data-test="library-count">{{
      countLabel
    }}</span>
    <span v-if="error" class="text-caption text-stop">{{ error }}</span>

    <div class="flex-1" />

    <ThumbnailSizeSlider
      v-if="scope !== 'collections'"
      :model-value="thumbnailSize"
      :min="GALLERY_THUMBNAIL_SIZE_MIN"
      :max="GALLERY_THUMBNAIL_SIZE_MAX"
      :step="GALLERY_THUMBNAIL_SIZE_STEP"
      @update:model-value="emit('update:thumbnailSize', $event)"
    />

    <SegmentedControl
      v-if="scope === 'prints'"
      :model-value="mediaKind"
      :options="kindOptions"
      label="Media kind"
      @update:model-value="emit('update:mediaKind', $event)"
    />

    <label
      class="flex h-[34px] w-[180px] items-center gap-2 rounded-chrome border border-ce bg-bench px-2.5"
    >
      <Icon name="search" :size="14" class="shrink-0 text-ink-3" />
      <input
        ref="searchEl"
        :value="search"
        data-selectable
        type="search"
        :placeholder="searchPlaceholder"
        aria-label="Search prints"
        class="min-w-0 flex-1 bg-transparent text-body text-ink outline-none placeholder:text-ink-3"
        @input="emit('update:search', ($event.target as HTMLInputElement).value)"
      />
    </label>

    <button
      type="button"
      class="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-chrome text-ink-3 transition-colors duration-100 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)] hover:text-ink"
      title="History"
      aria-label="Open history"
      @click="emit('openHistory')"
    >
      <Icon name="history" :size="17" />
    </button>
    <button
      type="button"
      class="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-chrome transition-colors duration-100"
      :class="
        selectMode
          ? 'bg-[color-mix(in_srgb,var(--safelight)_16%,transparent)] text-safelight'
          : 'text-ink-3 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)] hover:text-ink'
      "
      :aria-pressed="selectMode"
      title="Select"
      aria-label="Toggle select mode"
      @click="emit('toggleSelect')"
    >
      <Icon name="check" :size="17" />
    </button>
    <button
      v-if="scope === 'trash'"
      type="button"
      data-test="empty-trash"
      class="flex h-[34px] shrink-0 items-center gap-1.5 rounded-chrome border px-3 text-body text-stop transition-colors duration-100 hover:bg-[color-mix(in_srgb,var(--stop)_10%,transparent)] disabled:cursor-default disabled:opacity-40 disabled:hover:bg-transparent"
      style="border-color: color-mix(in srgb, var(--stop) 50%, transparent)"
      :disabled="trashCount === 0 || busy"
      @click="emit('emptyTrash')"
    >
      <Icon name="trash" :size="14" />
      Empty trash
    </button>
    <button
      v-else
      type="button"
      class="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-chrome text-ink-3 transition-colors duration-100 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)] hover:text-ink"
      title="Refresh"
      aria-label="Refresh library"
      @click="emit('refresh')"
    >
      <Icon name="refresh" :size="17" />
    </button>
  </header>
</template>

<style scoped>
/* The scope control reads as tabs: label + mono count side by side. */
.ms-lib-scope :deep(.ms-seg__btn) {
  flex-direction: row;
  gap: 6px;
}

.ms-lib-scope :deep(.ms-seg__sub) {
  font-family: var(--f-mono);
  font-size: 10.5px;
  color: inherit;
  opacity: 0.8;
}
</style>
