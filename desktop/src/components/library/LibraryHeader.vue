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
  prints: "Everything",
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
      class="ms-lib-scope"
      :model-value="scope"
      :options="scopeOptions"
      label="Library scope"
      compact
      data-test="library-scope"
      @update:model-value="emit('update:scope', $event)"
    />
    <span class="font-mono text-xs text-micro text-fg-dim" data-test="library-count">{{
      countLabel
    }}</span>
    <span v-if="error" class="text-micro text-error">{{ error }}</span>

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
      class="flex h-[26px] w-[170px] items-center gap-1.5 rounded-control border border-border bg-bg px-2 focus-within:border-border-focus"
    >
      <Icon name="search" :size="14" class="shrink-0 text-fg-dim" />
      <input
        ref="searchEl"
        :value="search"
        data-selectable
        type="search"
        :placeholder="searchPlaceholder"
        aria-label="Search prints"
        class="min-w-0 flex-1 bg-transparent text-xs text-fg outline-none placeholder:text-fg-dim"
        @input="emit('update:search', ($event.target as HTMLInputElement).value)"
      />
    </label>

    <button
      type="button"
      class="lib-button"
      :class="selectMode ? 'lib-button--on' : ''"
      :aria-pressed="selectMode"
      title="Select pictures"
      aria-label="Toggle select mode"
      @click="emit('toggleSelect')"
    >
      Select
    </button>
    <button
      type="button"
      class="lib-button"
      title="History"
      aria-label="Open history"
      @click="emit('openHistory')"
    >
      History
    </button>
    <button
      v-if="scope === 'trash'"
      type="button"
      data-test="empty-trash"
      class="lib-button lib-button--danger"
      :disabled="trashCount === 0 || busy"
      @click="emit('emptyTrash')"
    >
      <Icon name="trash" :size="14" />
      Empty trash
    </button>
    <button
      v-else
      type="button"
      class="flex h-[26px] w-[26px] shrink-0 items-center justify-center rounded-control border border-border text-fg-2 transition-colors duration-100 hover:border-border-focus hover:text-fg"
      title="Refresh"
      aria-label="Refresh library"
      @click="emit('refresh')"
    >
      <Icon name="refresh" :size="14" />
    </button>
  </header>
</template>

<style scoped>
.lib-button {
  display: inline-flex;
  height: var(--mold-ctl-md);
  flex-shrink: 0;
  align-items: center;
  gap: 6px;
  padding: 0 11px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-xs);
  font-weight: 500;
  white-space: nowrap;
  color: var(--mold-text-2);
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.lib-button:hover:not(:disabled) {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}
.lib-button--on {
  border-color: var(--mold-blue);
  background: var(--mold-blue);
  color: var(--mold-on-accent);
}
.lib-button--danger {
  border-color: var(--mold-error);
  color: var(--mold-error);
  font-weight: 600;
}
.lib-button--danger:hover:not(:disabled) {
  border-color: var(--mold-error);
  background: color-mix(in srgb, var(--mold-error) 12%, transparent);
  color: var(--mold-error);
}
.lib-button:disabled {
  opacity: 0.45;
  cursor: default;
}

/* The scope control reads as tabs: label + mono count side by side. */
.ms-lib-scope :deep(.ms-seg__btn) {
  flex-direction: row;
  gap: 6px;
}

.ms-lib-scope :deep(.ms-seg__sub) {
  font-family: var(--mold-font-mono);
  font-size: 10.5px;
  color: inherit;
  opacity: 0.8;
}
</style>
