<script setup lang="ts">
/*
 * LibraryHeader — the 40px per-view toolbar over the grid: the scope control
 * (Everything | Favourites | Albums | Trash with mono counts), search, and the
 * media-kind control on the left, then a spacer and the right cluster
 * (thumbnail slider, Select, History, Refresh). No count label — the shell's
 * title bar already carries "312 pictures · 6 albums". Scope options are
 * whatever the parent says the connected hosts can do — with a single option
 * the control is not rendered at all.
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
  favorites: number;
  collections: number;
  trash: number;
}

const props = withDefaults(
  defineProps<{
    scope: LibraryScope;
    /** Scopes the connected hosts support, in display order. */
    scopes: readonly LibraryScope[];
    counts: ScopeCounts;
    error?: string | null;
    thumbnailSize: number;
    mediaKind: GalleryKindFilter;
    kindOptions: readonly { value: GalleryKindFilter; label: string }[];
    search: string;
    selectMode: boolean;
    historyOpen?: boolean;
  }>(),
  { error: null, historyOpen: false },
);

const emit = defineEmits<{
  "update:scope": [scope: LibraryScope];
  "update:thumbnailSize": [px: number];
  "update:mediaKind": [kind: GalleryKindFilter];
  "update:search": [value: string];
  toggleHistory: [];
  toggleSelect: [];
  refresh: [];
}>();

const SCOPE_LABELS: Record<LibraryScope, string> = {
  prints: "Everything",
  favorites: "Favourites",
  collections: "Albums",
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
    ? "Search albums…"
    : props.scope === "trash"
      ? "Search the trash…"
      : "Search words or tags…",
);

const searchEl = ref<HTMLInputElement | null>(null);
function focusSearch() {
  searchEl.value?.focus();
}
defineExpose({ focusSearch });
</script>

<template>
  <header
    class="flex h-[var(--mold-shell-viewbar-h)] shrink-0 items-center gap-2.5 border-b border-border bg-chrome px-3.5"
    data-test="library-header"
  >
    <SegmentedControl
      v-if="scopes.length > 1"
      inline
      :model-value="scope"
      :options="scopeOptions"
      label="My images scope"
      compact
      data-test="library-scope"
      @update:model-value="emit('update:scope', $event)"
    />

    <label
      class="flex h-[26px] w-[170px] items-center gap-1.5 rounded-control border border-border bg-bg px-2 focus-within:border-border-focus"
    >
      <Icon name="search" :size="14" class="shrink-0 text-fg-dim" />
      <input
        ref="searchEl"
        :value="search"
        data-selectable
        type="search"
        :placeholder="searchPlaceholder"
        aria-label="Search pictures"
        class="min-w-0 flex-1 bg-transparent text-xs text-fg outline-none placeholder:text-fg-dim"
        @input="emit('update:search', ($event.target as HTMLInputElement).value)"
      />
    </label>

    <SegmentedControl
      :model-value="mediaKind"
      :options="kindOptions"
      label="Media kind"
      @update:model-value="emit('update:mediaKind', $event)"
    />

    <span v-if="error" class="truncate text-micro text-error">{{ error }}</span>

    <div class="flex-1" />

    <ThumbnailSizeSlider
      :model-value="thumbnailSize"
      :min="GALLERY_THUMBNAIL_SIZE_MIN"
      :max="GALLERY_THUMBNAIL_SIZE_MAX"
      :step="GALLERY_THUMBNAIL_SIZE_STEP"
      @update:model-value="emit('update:thumbnailSize', $event)"
    />

    <button
      type="button"
      class="ms-toolbar-button"
      :class="selectMode ? 'ms-toolbar-button--on' : ''"
      :aria-pressed="selectMode"
      title="Select pictures"
      aria-label="Toggle select mode"
      @click="emit('toggleSelect')"
    >
      Select
    </button>
    <button
      type="button"
      class="ms-toolbar-button"
      :class="historyOpen ? 'ms-toolbar-button--on' : ''"
      :aria-pressed="historyOpen"
      title="History"
      :aria-label="historyOpen ? 'Close history' : 'Open history'"
      @click="emit('toggleHistory')"
    >
      History
    </button>
    <!-- Refresh is not the poll: `pollExtras` deliberately skips the primary
         bucket because it is live over SSE, so this is the only way back to a
         current grid when that stream has dropped. -->
    <button
      type="button"
      class="ms-toolbar-button ms-toolbar-button--icon"
      title="Refresh"
      aria-label="Refresh my images"
      @click="emit('refresh')"
    >
      <Icon name="refresh" :size="14" />
    </button>
  </header>
</template>
